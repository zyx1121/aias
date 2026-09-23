"""aias MCP server: pull, start and stop local models on one GPU.

Runs in a container next to the engines and drives them through the Docker
socket with the same compose file a person would use by hand.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import httpx
from huggingface_hub import HfApi, scan_cache_dir, snapshot_download
from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from starlette.requests import Request
from starlette.responses import JSONResponse

Engine = Literal["ollama", "vllm"]
ENGINES: tuple[Engine, ...] = ("ollama", "vllm")

COMPOSE = ["docker", "compose", "-f", os.environ.get("AIAS_COMPOSE", "/opt/aias/compose.yaml")]
PORT = 11400
# Inside the compose network the engines answer on their service names.
INTERNAL = {"ollama": "http://ollama:11434", "vllm": "http://vllm:8000"}
# What a client on the Windows host uses.
PUBLIC = {"ollama": "http://127.0.0.1:11434/v1", "vllm": "http://127.0.0.1:8000/v1"}
OLLAMA_MANIFESTS = Path("/ollama/models/manifests")
VLLM_STARTUP_SECS = 900

mcp = MCPServer(
    "aias",
    instructions=(
        "Local model host on one NVIDIA GPU. Two engines: ollama (names like qwen3:8b) "
        "and vllm (Hugging Face ids like Qwen/Qwen3-0.6B). Only one model is up at a time; "
        "model_up stops whatever else is running. Pulls and startups are jobs, one at a time: "
        "poll job_status until it finishes before starting the next. "
        "Once a model is up, call it through the OpenAI compatible base_url that status returns."
    ),
)


# ---------------------------------------------------------------- helpers


def _run(*args: str, env: dict[str, str] | None = None, timeout: float = 600) -> str:
    proc = subprocess.run(
        [*COMPOSE, *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, **(env or {})},
    )
    if proc.returncode != 0:
        raise RuntimeError(f"docker compose {' '.join(args)} failed: {proc.stderr.strip()[-2000:]}")
    return proc.stdout


def _running() -> dict[str, dict[str, Any]]:
    """Engine containers that exist, keyed by service, with their state."""
    out = _run("ps", "--all", "--format", "json", "ollama", "vllm")
    rows = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        parsed = json.loads(line)
        rows.extend(parsed if isinstance(parsed, list) else [parsed])
    return {r["Service"]: r for r in rows if r.get("State") == "running"}


def _vllm_model_from_container() -> str | None:
    out = subprocess.run(
        ["docker", "inspect", "-f", "{{json .Args}}", "aias-vllm-1"],
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        return None
    for arg in json.loads(out.stdout or "[]"):
        if arg.startswith("--model="):
            return arg.split("=", 1)[1]
    return None


def _gpu() -> dict[str, Any] | None:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        return None
    name, used, total = (x.strip() for x in out.stdout.splitlines()[0].split(","))
    return {"name": name, "memory_used_mib": int(used), "memory_total_mib": int(total)}


def _stop(*services: str) -> None:
    if services:
        _run("stop", *services, timeout=120)


async def _wait_http(url: str, secs: float, alive: Any = None) -> bool:
    """Poll url until it answers 200; give up early if alive() turns false."""
    deadline = time.monotonic() + secs
    async with httpx.AsyncClient(timeout=5) as client:
        while time.monotonic() < deadline:
            try:
                if (await client.get(url)).status_code == 200:
                    return True
            except httpx.HTTPError:
                pass
            if alive is not None and not await asyncio.to_thread(alive):
                return False
            await asyncio.sleep(2)
    return False


def _ollama_models() -> list[dict[str, Any]]:
    models = []
    if not OLLAMA_MANIFESTS.exists():
        return models
    for manifest in OLLAMA_MANIFESTS.rglob("*"):
        if not manifest.is_file():
            continue
        parts = manifest.relative_to(OLLAMA_MANIFESTS).parts  # host/namespace/name/tag
        if len(parts) != 4:
            continue
        host, namespace, name, tag = parts
        full = f"{name}:{tag}" if namespace == "library" else f"{namespace}/{name}:{tag}"
        if host != "registry.ollama.ai":
            full = f"{host}/{full}"
        size = sum(layer.get("size", 0) for layer in json.loads(manifest.read_text()).get("layers", []))
        models.append({"engine": "ollama", "model": full, "size_gb": round(size / 1e9, 2)})
    return sorted(models, key=lambda m: m["model"])


def _vllm_models() -> list[dict[str, Any]]:
    try:
        cache = scan_cache_dir()
    except Exception:
        return []
    return sorted(
        (
            {"engine": "vllm", "model": repo.repo_id, "size_gb": round(repo.size_on_disk / 1e9, 2)}
            for repo in cache.repos
            if repo.repo_type == "model"
        ),
        key=lambda m: m["model"],
    )


# ---------------------------------------------------------------- jobs


@dataclass
class Job:
    kind: str
    engine: Engine
    model: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    state: Literal["running", "done", "error"] = "running"
    detail: str = ""
    started: float = field(default_factory=time.time)
    finished: float | None = None
    task: asyncio.Task | None = None

    def view(self) -> dict[str, Any]:
        end = self.finished or time.time()
        return {
            "job_id": self.id,
            "kind": self.kind,
            "engine": self.engine,
            "model": self.model,
            "state": self.state,
            "detail": self.detail,
            "elapsed_s": round(end - self.started),
        }


JOBS: dict[str, Job] = {}


def _busy() -> Job | None:
    return next((j for j in JOBS.values() if j.state == "running"), None)


def _refuse_if_busy() -> None:
    # One job at a time: a pull may start the ollama server and an up stops the
    # other engine, so overlapping jobs could leave two engines on one GPU.
    if (job := _busy()) is not None:
        raise RuntimeError(
            f"job {job.id} ({job.kind} {job.model}) is still running; "
            "wait for it with job_status, then retry"
        )


def _start_job(job: Job, work: Any) -> dict[str, Any]:
    _refuse_if_busy()

    async def runner() -> None:
        try:
            job.detail = await work(job) or "done"
            job.state = "done"
        except Exception as exc:  # reported through job_status
            job.state = "error"
            job.detail = str(exc)[-3000:]
        finally:
            job.finished = time.time()

    JOBS[job.id] = job
    job.task = asyncio.create_task(runner())
    return job.view()


async def _ensure_ollama_server() -> bool:
    """Start the ollama server if needed. Returns True when this call started it."""
    started = False
    if "ollama" not in await asyncio.to_thread(_running):
        await asyncio.to_thread(_run, "up", "-d", "ollama")
        started = True
    if not await _wait_http(f"{INTERNAL['ollama']}/api/version", 60):
        raise RuntimeError("ollama server did not come up within 60 s")
    return started


async def _pull_ollama(job: Job) -> str:
    started = await _ensure_ollama_server()
    try:
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST", f"{INTERNAL['ollama']}/api/pull", json={"model": job.model}
            ) as resp:
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    msg = json.loads(line)
                    if "error" in msg:
                        raise RuntimeError(msg["error"])
                    status = msg.get("status", "")
                    if msg.get("total"):
                        pct = 100 * msg.get("completed", 0) / msg["total"]
                        status = f"{status} {pct:.0f}% of {msg['total'] / 1e9:.2f} GB"
                    job.detail = status
    finally:
        if started:
            # A server this job started would sit idle next to a vLLM model.
            await asyncio.to_thread(_stop, "ollama")
    return f"pulled {job.model}"


async def _pull_vllm(job: Job) -> str:
    files = await asyncio.to_thread(HfApi().list_repo_files, job.model)
    ignore = ["*.pth", "original/*"]
    if any(f.endswith(".safetensors") for f in files):
        ignore.append("*.bin")
    job.detail = f"downloading {len(files)} files"
    path = await asyncio.to_thread(snapshot_download, job.model, ignore_patterns=ignore)
    size = sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())
    return f"pulled {job.model} ({size / 1e9:.2f} GB)"


async def _up_ollama(job: Job) -> str:
    await asyncio.to_thread(_stop, "vllm")
    await _ensure_ollama_server()
    job.detail = "loading into VRAM"
    async with httpx.AsyncClient(timeout=600) as client:
        resp = await client.post(
            f"{INTERNAL['ollama']}/api/generate", json={"model": job.model, "keep_alive": -1}
        )
        if resp.status_code != 200:
            raise RuntimeError(f"ollama could not load {job.model}: {resp.text[:500]}")
    return f"{job.model} is up at {PUBLIC['ollama']}"


async def _up_vllm(job: Job, max_model_len: int, gpu_memory_utilization: float) -> str:
    await asyncio.to_thread(_stop, "ollama")
    env = {
        "VLLM_MODEL": job.model,
        "VLLM_MAX_LEN": str(max_model_len),
        "VLLM_GPU_UTIL": str(gpu_memory_utilization),
    }
    await asyncio.to_thread(_run, "up", "-d", "--force-recreate", "vllm", env=env)
    job.detail = "starting (weights, compile, warmup)"

    def alive() -> bool:
        return "vllm" in _running()

    if await _wait_http(f"{INTERNAL['vllm']}/v1/models", VLLM_STARTUP_SECS, alive):
        return f"{job.model} is up at {PUBLIC['vllm']}"
    tail = await asyncio.to_thread(_run, "logs", "--no-log-prefix", "--tail", "40", "vllm")
    raise RuntimeError(f"vLLM did not become ready. Last log lines:\n{tail}")


# ---------------------------------------------------------------- tools


@mcp.tool()
async def status() -> dict[str, Any]:
    """What is running now: the engine and model that are up, their OpenAI compatible
    base_url, GPU memory, and jobs still in progress. Call this first."""
    running = await asyncio.to_thread(_running)
    up = []
    if "ollama" in running:
        loaded = []
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                ps = (await client.get(f"{INTERNAL['ollama']}/api/ps")).json()
            loaded = [m["name"] for m in ps.get("models", [])]
        except httpx.HTTPError:
            pass
        up.append({"engine": "ollama", "models": loaded, "base_url": PUBLIC["ollama"]})
    if "vllm" in running:
        ready = await _wait_http(f"{INTERNAL['vllm']}/v1/models", 0.1)
        up.append(
            {
                "engine": "vllm",
                "models": [await asyncio.to_thread(_vllm_model_from_container)],
                "ready": ready,
                "base_url": PUBLIC["vllm"],
            }
        )
    return {
        "up": up,
        "gpu": await asyncio.to_thread(_gpu),
        "jobs": [j.view() for j in JOBS.values() if j.state == "running"],
    }


@mcp.tool()
async def model_list(engine: Engine | None = None) -> list[dict[str, Any]]:
    """Models already downloaded on this machine, with size on disk. Filter by engine."""
    models: list[dict[str, Any]] = []
    if engine in (None, "ollama"):
        models += await asyncio.to_thread(_ollama_models)
    if engine in (None, "vllm"):
        models += await asyncio.to_thread(_vllm_models)
    return models


@mcp.tool()
async def model_pull(engine: Engine, model: str) -> dict[str, Any]:
    """Download a model. ollama takes library names like qwen3:8b; vllm takes Hugging
    Face repo ids like Qwen/Qwen3-0.6B. Returns a job; poll job_status."""
    work = _pull_ollama if engine == "ollama" else _pull_vllm
    return _start_job(Job("pull", engine, model), work)


@mcp.tool()
async def model_up(
    engine: Engine,
    model: str,
    max_model_len: int = 8192,
    gpu_memory_utilization: float = 0.8,
) -> dict[str, Any]:
    """Load a model so it serves requests, stopping any other engine first. The two
    tuning arguments apply to vllm only. vLLM startup takes 40 s to several minutes.
    Returns a job; when it is done, status shows the base_url."""
    job = Job("up", engine, model)
    if engine == "ollama":
        return _start_job(job, _up_ollama)
    return _start_job(job, lambda j: _up_vllm(j, max_model_len, gpu_memory_utilization))


@mcp.tool()
async def model_down() -> dict[str, Any]:
    """Stop every engine and free the GPU. Refused while a job is running."""
    _refuse_if_busy()
    await asyncio.to_thread(_stop, *ENGINES)
    return {"stopped": list(ENGINES), "gpu": await asyncio.to_thread(_gpu)}


@mcp.tool()
async def job_status(job_id: str, wait_seconds: int = 30) -> dict[str, Any]:
    """Progress of a pull or up job. Blocks up to wait_seconds (max 120) for it to finish."""
    job = JOBS.get(job_id)
    if job is None:
        raise ValueError(f"no job {job_id}; jobs do not survive a server restart")
    if job.task is not None and job.state == "running":
        await asyncio.wait({job.task}, timeout=max(0, min(wait_seconds, 120)))
    return job.view()


@mcp.tool()
async def logs(engine: Engine, lines: int = 100) -> str:
    """Recent log lines of an engine, for diagnosing a failed start."""
    lines = max(1, min(lines, 1000))
    return await asyncio.to_thread(_run, "logs", "--no-log-prefix", "--tail", str(lines), engine)


@mcp.custom_route("/health", methods=["GET"])
async def health(_: Request) -> JSONResponse:
    return JSONResponse({"ok": True})


if __name__ == "__main__":
    mcp.run(
        "streamable-http",
        host="0.0.0.0",
        port=PORT,
        stateless_http=True,
        json_response=True,
        # Published on 127.0.0.1 only; still refuse other Host headers so a web
        # page cannot reach it through DNS rebinding.
        transport_security=TransportSecuritySettings(
            allowed_hosts=[f"localhost:{PORT}", f"127.0.0.1:{PORT}"],
            allowed_origins=[f"http://localhost:{PORT}", f"http://127.0.0.1:{PORT}"],
        ),
    )
