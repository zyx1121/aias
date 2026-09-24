"""aias MCP server: pull, start and stop local models on one GPU, and diarize audio.

Runs in a container next to the engines and drives them through the Docker
socket with the same compose file a person would use by hand.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import httpx
from huggingface_hub import HfApi, scan_cache_dir, snapshot_download
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from starlette.requests import Request
from starlette.responses import JSONResponse

Engine = Literal["ollama", "vllm", "nemo"]
ENGINES: tuple[Engine, ...] = ("ollama", "vllm", "nemo")
Mode = Literal["offline", "low", "verylow", "ultralow"]

COMPOSE = ["docker", "compose", "-f", os.environ.get("AIAS_COMPOSE", "/opt/aias/compose.yaml")]
PORT = 11400
# Inside the compose network the engines answer on their service names.
INTERNAL = {"ollama": "http://ollama:11434", "vllm": "http://vllm:8000", "nemo": "http://nemo:8100"}
# What a client on the Windows host uses. nemo has no port; diarize reaches it.
PUBLIC = {"ollama": "http://127.0.0.1:11434/v1", "vllm": "http://127.0.0.1:8000/v1"}
OLLAMA_MANIFESTS = Path("/ollama/models/manifests")
VLLM_STARTUP_SECS = 900
NEMO_STARTUP_SECS = 600
NEMO_BUILD_SECS = 3600
# nemo loads .nemo archives, which can carry pickled code: only NVIDIA's repos.
# One path segment under nvidia/, so `nvidia/../other/repo` cannot slip through.
NEMO_REPO = re.compile(r"nvidia/(?!\.+$)[A-Za-z0-9_.-]+")
# One diarize call end to end: the engine caps the download at 10 minutes and
# the audio at 2 hours, which ultralow mode needs about 25 minutes for.
DIARIZE_SECS = 3600

mcp = MCPServer(
    "aias",
    instructions=(
        "Local model host on one NVIDIA GPU. Three engines: ollama (names like qwen3:8b), "
        "vllm (Hugging Face ids like Qwen/Qwen3-0.6B) and nemo, speaker diarization "
        "(nvidia/Nemotron-3-Diarization). Only one model is up at a time; "
        "model_up stops whatever else is running. Pulls, startups and diarize runs are jobs, "
        "one at a time: poll job_status until it finishes before starting the next. "
        "Once an ollama or vllm model is up, call it through the OpenAI compatible base_url "
        "that status returns. Once nemo is up, call diarize with an audio URL; the finished "
        "job carries the RTTM and seconds per speaker."
    ),
)


# ---------------------------------------------------------------- helpers


_NO_LOCK = contextlib.nullcontext()
# Serializes compose calls that build, start or stop containers. A cancelled
# job's `up` keeps running in its worker thread, so a later `stop` must wait for it.
_COMPOSE_LOCK = threading.Lock()
# The nemo image build in progress, so model_down can end it instead of
# waiting minutes for the lock.
_BUILD: subprocess.Popen[str] | None = None
# Set by model_down so a build still waiting for the lock never starts.
_BUILD_CANCELLED = threading.Event()


def _run(*args: str, env: dict[str, str] | None = None, timeout: float = 600) -> str:
    mutating = args[0] in ("up", "stop", "build")
    with _COMPOSE_LOCK if mutating else _NO_LOCK:
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
    out = _run("ps", "--all", "--format", "json", *ENGINES)
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


def _nemo_model_from_container() -> str | None:
    out = subprocess.run(
        ["docker", "inspect", "-f", "{{json .Config.Env}}", "aias-nemo-1"],
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        return None
    for var in json.loads(out.stdout or "[]"):
        if var.startswith("NEMO_MODEL="):
            return var.split("=", 1)[1]
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


def _build_nemo() -> None:
    """Build the nemo image. Runs on every model_up: an unchanged nemo/ is a cache
    hit in seconds, and a changed one (after an upgrade) gets rebuilt."""
    global _BUILD
    with _COMPOSE_LOCK:
        if _BUILD_CANCELLED.is_set():
            raise RuntimeError("nemo image build cancelled by model_down")
        proc = subprocess.Popen(
            [*COMPOSE, "build", "nemo"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        _BUILD = proc
        try:
            out, _ = proc.communicate(timeout=NEMO_BUILD_SECS)
        except subprocess.TimeoutExpired:
            proc.kill()
            out, _ = proc.communicate()
        finally:
            _BUILD = None
    if proc.returncode != 0:
        raise RuntimeError(f"building the nemo image failed (exit {proc.returncode}): {out.strip()[-2000:]}")


def _cancel_build() -> None:
    _BUILD_CANCELLED.set()
    if (proc := _BUILD) is not None and proc.poll() is None:
        proc.terminate()


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


def _hf_models() -> list[dict[str, Any]]:
    """Hugging Face repos in the shared cache. A repo with a .nemo file is for nemo."""
    try:
        cache = scan_cache_dir()
    except Exception:
        return []
    models = []
    for repo in cache.repos:
        if repo.repo_type != "model":
            continue
        files = {f.file_name for rev in repo.revisions for f in rev.files}
        engine = "nemo" if any(name.endswith(".nemo") for name in files) else "vllm"
        models.append({"engine": engine, "model": repo.repo_id, "size_gb": round(repo.size_on_disk / 1e9, 2)})
    return sorted(models, key=lambda m: m["model"])


# ---------------------------------------------------------------- jobs


@dataclass
class Job:
    kind: str
    engine: Engine
    model: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    state: Literal["running", "done", "error"] = "running"
    detail: str = ""
    result: dict[str, Any] | None = None
    started: float = field(default_factory=time.time)
    finished: float | None = None
    task: asyncio.Task | None = None

    def view(self) -> dict[str, Any]:
        end = self.finished or time.time()
        view = {
            "job_id": self.id,
            "kind": self.kind,
            "engine": self.engine,
            "model": self.model,
            "state": self.state,
            "detail": self.detail,
            "elapsed_s": round(end - self.started),
        }
        if self.result is not None:
            view["result"] = self.result
        return view


JOBS: dict[str, Job] = {}


def _busy() -> Job | None:
    return next((j for j in JOBS.values() if j.state == "running"), None)


def _refuse_if_busy() -> None:
    # One job at a time: a pull may start the ollama server and an up stops the
    # other engine, so overlapping jobs could leave two engines on one GPU.
    if (job := _busy()) is not None:
        raise ToolError(
            f"job {job.id} ({job.kind} {job.model}) is still running; "
            "wait for it with job_status, then retry"
        )


def _start_job(job: Job, work: Any) -> dict[str, Any]:
    _refuse_if_busy()

    async def runner() -> None:
        try:
            job.detail = await work(job) or "done"
            job.state = "done"
        except asyncio.CancelledError:
            job.state = "error"
            job.detail = "cancelled by model_down"
        except Exception as exc:  # reported through job_status
            job.state = "error"
            # Some exceptions (httpx timeouts) carry no message.
            job.detail = (str(exc) or type(exc).__name__)[-3000:]
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


async def _pull_hf(job: Job) -> str:
    files = await asyncio.to_thread(HfApi().list_repo_files, job.model)
    if job.engine == "nemo":
        # NeMo loads the single .nemo archive; skip the safetensors, gguf and demo video.
        patterns: dict[str, Any] = {"allow_patterns": ["*.nemo"]}
        if not any(f.endswith(".nemo") for f in files):
            raise RuntimeError(f"{job.model} has no .nemo file, so the nemo engine cannot load it")
    else:
        ignore = ["*.pth", "original/*"]
        if any(f.endswith(".safetensors") for f in files):
            ignore.append("*.bin")
        patterns = {"ignore_patterns": ignore}
    job.detail = f"downloading from {len(files)} files"
    path = await asyncio.to_thread(snapshot_download, job.model, **patterns)
    size = sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())
    return f"pulled {job.model} ({size / 1e9:.2f} GB)"


def _others(engine: Engine) -> list[Engine]:
    return [e for e in ENGINES if e != engine]


async def _up_ollama(job: Job) -> str:
    await asyncio.to_thread(_stop, *_others("ollama"))
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
    await asyncio.to_thread(_stop, *_others("vllm"))
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


def _check_nemo_repo(model: str) -> None:
    if not NEMO_REPO.fullmatch(model):
        raise ToolError(f"nemo only loads Hugging Face repos named nvidia/<name>, not {model}")


async def _up_nemo(job: Job) -> str:
    await asyncio.to_thread(_stop, *_others("nemo"))
    job.detail = "building the nemo image (about 5 minutes the first time, seconds after)"
    _BUILD_CANCELLED.clear()
    await asyncio.to_thread(_build_nemo)
    await asyncio.to_thread(_run, "up", "-d", "--force-recreate", "nemo", env={"NEMO_MODEL": job.model})
    job.detail = "loading the model onto the GPU"

    def alive() -> bool:
        return "nemo" in _running()

    if await _wait_http(f"{INTERNAL['nemo']}/health", NEMO_STARTUP_SECS, alive):
        return f"{job.model} is up; call diarize"
    tail = await asyncio.to_thread(_run, "logs", "--no-log-prefix", "--tail", "40", "nemo")
    raise RuntimeError(f"nemo did not become ready. Last log lines:\n{tail}")


async def _diarize(job: Job, audio_url: str, mode: str) -> str:
    job.detail = f"downloading and diarizing ({mode})"
    try:
        async with asyncio.timeout(DIARIZE_SECS):
            async with httpx.AsyncClient(timeout=httpx.Timeout(DIARIZE_SECS, connect=10)) as client:
                resp = await client.post(
                    f"{INTERNAL['nemo']}/diarize", json={"audio_url": audio_url, "mode": mode}
                )
    except TimeoutError:
        raise RuntimeError(f"diarize did not finish within {DIARIZE_SECS} s") from None
    except httpx.HTTPError as exc:
        raise RuntimeError(f"could not reach the nemo engine: {type(exc).__name__} {exc}".rstrip()) from exc
    try:
        body = resp.json()
    except ValueError:
        body = {"error": resp.text[-1000:]}
    if resp.status_code != 200:
        raise RuntimeError(body.get("error") or f"nemo answered HTTP {resp.status_code}")
    job.result = body
    speakers = ", ".join(f"{s['speaker']} {s['seconds']:.0f} s" for s in body["speakers"]) or "no speech"
    return f"{len(body['speakers'])} speakers in {body['audio_s']:.0f} s of audio: {speakers}"


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
    if "nemo" in running:
        up.append(
            {
                "engine": "nemo",
                "models": [await asyncio.to_thread(_nemo_model_from_container)],
                "ready": await _wait_http(f"{INTERNAL['nemo']}/health", 0.1),
                "base_url": None,
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
    if engine != "ollama":
        models += [m for m in await asyncio.to_thread(_hf_models) if engine in (None, m["engine"])]
    return models


@mcp.tool()
async def model_pull(engine: Engine, model: str) -> dict[str, Any]:
    """Download a model. ollama takes library names like qwen3:8b; vllm and nemo take Hugging
    Face repo ids like Qwen/Qwen3-0.6B or nvidia/Nemotron-3-Diarization. Returns a job;
    poll job_status."""
    if engine == "nemo":
        _check_nemo_repo(model)
    work = _pull_ollama if engine == "ollama" else _pull_hf
    return _start_job(Job("pull", engine, model), work)


@mcp.tool()
async def model_up(
    engine: Engine,
    model: str,
    max_model_len: int = 8192,
    gpu_memory_utilization: float = 0.8,
) -> dict[str, Any]:
    """Load a model so it serves requests, stopping any other engine first. The two
    tuning arguments apply to vllm only. vLLM startup takes 40 s to several minutes; the
    first nemo start builds its image and takes about 5 minutes.
    Returns a job; when it is done, status shows the base_url (nemo has none: use diarize)."""
    job = Job("up", engine, model)
    if engine == "ollama":
        return _start_job(job, _up_ollama)
    if engine == "nemo":
        _check_nemo_repo(model)
        return _start_job(job, _up_nemo)
    return _start_job(job, lambda j: _up_vllm(j, max_model_len, gpu_memory_utilization))


@mcp.tool()
async def model_down() -> dict[str, Any]:
    """Stop every engine and free the GPU. Cancels a pull, up or diarize job that is still running."""
    if (job := _busy()) is not None and job.task is not None:
        job.task.cancel()
        # A build holds the compose lock for minutes; end it so _stop gets the lock.
        _cancel_build()
        await asyncio.wait({job.task}, timeout=30)
    await asyncio.to_thread(_stop, *ENGINES)
    return {"stopped": list(ENGINES), "gpu": await asyncio.to_thread(_gpu)}


@mcp.tool()
async def diarize(audio_url: str, mode: Mode = "offline") -> dict[str, Any]:
    """Label who spoke when in an audio file, up to 8 speakers, with the nemo engine
    (model_up engine=nemo first). audio_url is an http(s) link the server downloads; any
    format ffmpeg reads, up to 2 GB. mode trades accuracy for latency: offline (best),
    low (1.04 s), verylow (0.64 s), ultralow (0.32 s); the streaming modes are slower on
    a whole file. Returns a job; when it is done, job_status carries result with the
    RTTM text and the seconds each speaker talked."""
    running = await asyncio.to_thread(_running)
    if "nemo" not in running:
        raise ToolError("nemo is not up: run model_up engine=nemo model=nvidia/Nemotron-3-Diarization first")
    model = await asyncio.to_thread(_nemo_model_from_container) or "nemo"
    return _start_job(Job("diarize", "nemo", model), lambda j: _diarize(j, audio_url, mode))


@mcp.tool()
async def job_status(job_id: str, wait_seconds: int = 30) -> dict[str, Any]:
    """Progress of a pull, up or diarize job. Blocks up to wait_seconds (max 120) for it to
    finish. A finished diarize job carries its output in result."""
    job = JOBS.get(job_id)
    if job is None:
        raise ToolError(f"no job {job_id}; jobs do not survive a server restart")
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
