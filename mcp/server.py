"""aias MCP server: pull, start and stop local models on one GPU, and diarize or
transcribe audio.

Runs in a container next to the engines and drives them through the Docker
socket with the same compose file a person would use by hand. Several models
share the card: one Ollama server (any number of models), one vLLM model and
one nemo model, admitted against a VRAM budget (vram.py).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import os
import re
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import httpx
import opencc
from huggingface_hub import HfApi, scan_cache_dir, snapshot_download, try_to_load_from_cache
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import StrictInt
from starlette.requests import Request
from starlette.responses import JSONResponse

import vram
from audio import AudioError, fetch_wav

Engine = Literal["ollama", "vllm", "nemo"]
ENGINES: tuple[Engine, ...] = ("ollama", "vllm", "nemo")
Mode = Literal["offline", "low", "verylow", "ultralow"]
Evict = Literal["never", "auto"]

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
# One diarize or transcribe call end to end: audio.py caps the download at
# 10 minutes and the audio at 2 hours, which ultralow diarization needs about
# 25 minutes for.
AUDIO_JOB_SECS = 3600
# model_up defaults for vllm. Whisper's decoder takes 448 tokens; the others
# get 80 % of the card, as vLLM's own default would.
VLLM_DEFAULTS = {"max_model_len": 8192, "gpu_memory_utilization": 0.8}
WHISPER_DEFAULTS = {"max_model_len": 448, "gpu_memory_utilization": 0.4}
# vLLM sizes its KV cache from "whole card minus what is in use", which next to
# another engine is not this instance's share. aias always passes
# --kv-cache-memory instead, so an instance takes a fixed amount whatever else
# is on the card. Whisper needs at least 0.38 GiB; other models get what their
# budget leaves after the weights and about 1.5 GiB of context and activations.
WHISPER_KV_BYTES = 450_000_000
VLLM_OVERHEAD_MIB = 1536
MIN_KV_MIB = 512
# Transcriptions a vLLM Whisper instance runs at once; nemo runs one diarize.
TRANSCRIBE_CONCURRENCY = int(os.environ.get("AIAS_TRANSCRIBE_CONCURRENCY", "4"))
RECONCILE_SECS = 10
# A container list taken just before a model finished starting must not drop it.
SETTLE_SECS = 2 * RECONCILE_SECS
# Whisper drifts from traditional to simplified Chinese after about 30 s.
# s2tw converts characters only; s2twp would also swap mainland vocabulary for
# Taiwanese words, which rewrites what the speaker said.
_S2TW = opencc.OpenCC("s2tw")

log = logging.getLogger("aias")

mcp = MCPServer(
    "aias",
    instructions=(
        "Local model host on one NVIDIA GPU. Three engines: ollama (names like qwen3:8b), "
        "vllm (Hugging Face ids like Qwen/Qwen3-0.6B, or openai/whisper-large-v3 for speech "
        "to text) and nemo, speaker diarization (nvidia/Nemotron-3-Diarization). Models "
        "share the card within a VRAM budget: any number of ollama models, one vllm model "
        "and one nemo model at a time. model_up refuses a model that does not fit and "
        "returns a plan (fits, need_mib, free_mib, evict); pass evict=\"auto\" to stop the "
        "least recently used models in that plan, dry_run=true to only see it, and "
        "pin=true to keep a model from being evicted. Pulls, startups, diarize and "
        "transcribe runs are jobs: poll job_status until each finishes. Once an ollama or "
        "vllm model is up, call it through the OpenAI compatible base_url that status "
        "returns. Once nemo is up, call diarize with an audio URL; once a whisper model is "
        "up on vllm, call transcribe. The finished job carries the output in result. "
        "model_down with a model stops only that one; without one it stops everything."
    ),
)


# ---------------------------------------------------------------- helpers


_NO_LOCK = contextlib.nullcontext()
# Serializes compose calls that build, start or stop containers. A cancelled
# job's `up` keeps running in its worker thread, so a later `stop` must wait for it.
# Reentrant, so a check-then-stop can hold it across both steps.
_COMPOSE_LOCK = threading.RLock()
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


def _inspect(container: str, fmt: str) -> list[str]:
    out = subprocess.run(["docker", "inspect", "-f", fmt, container], capture_output=True, text=True)
    if out.returncode != 0:
        return []
    return json.loads(out.stdout or "[]") or []


def _vllm_args_from_container() -> dict[str, str]:
    """--model, --max-model-len and --kv-cache-memory of the running vLLM."""
    args = {}
    for arg in _inspect("aias-vllm-1", "{{json .Args}}"):
        if arg.startswith("--") and "=" in arg:
            key, value = arg[2:].split("=", 1)
            args[key] = value
    return args


def _vllm_model_from_container() -> str | None:
    return _vllm_args_from_container().get("model")


def _nemo_model_from_container() -> str | None:
    for var in _inspect("aias-nemo-1", "{{json .Config.Env}}"):
        if var.startswith("NEMO_MODEL="):
            return var.split("=", 1)[1]
    return None


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


def _disk_mib(engine: str, model: str) -> int | None:
    if engine != "ollama":
        return _hf_weights_mib(model)
    size = next((m["size_gb"] for m in _ollama_models() if m["model"] == model), None)
    return None if size is None else round(size * 1e9 / 2**20)


def _hf_weights_mib(model: str) -> int | None:
    """Weights of the revision main points at (safetensors, else .bin), not every
    revision in the cache."""
    try:
        cache = scan_cache_dir()
    except Exception:
        return None
    repo = next((r for r in cache.repos if r.repo_id == model and r.repo_type == "model"), None)
    if repo is None or not repo.revisions:
        return None
    rev = max(repo.revisions, key=lambda r: ("main" in r.refs, r.last_modified))
    files = [f for f in rev.files if f.file_name.endswith(".safetensors")]
    files = files or [f for f in rev.files if f.file_name.endswith(".bin")]
    return round(sum(f.size_on_disk for f in files) / 2**20) if files else None


def _kv_mib_for(model: str, max_len: int) -> int | None:
    """KV cache one max_len sequence needs, from the model's config.json in the
    cache (None if it is not there or lacks the fields)."""
    path = try_to_load_from_cache(model, "config.json")
    if not isinstance(path, str):
        return None
    try:
        cfg = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    text = cfg.get("text_config") or cfg
    layers, heads = text.get("num_hidden_layers"), text.get("num_attention_heads")
    kv_heads = text.get("num_key_value_heads") or heads
    head_dim = text.get("head_dim") or (text["hidden_size"] // heads if text.get("hidden_size") and heads else None)
    if not (layers and kv_heads and head_dim):
        return None
    dtype = str(text.get("torch_dtype") or cfg.get("torch_dtype") or "bfloat16")
    size = 4 if dtype == "float32" else 2
    # K and V, plus 5 % for vLLM's block rounding.
    return math.ceil(max_len * layers * kv_heads * head_dim * 2 * size * 1.05 / 2**20)


def _ollama_name(model: str) -> str:
    """Ollama reports qwen3 as qwen3:latest; key both the same way."""
    return model if ":" in model.rsplit("/", 1)[-1] else f"{model}:latest"


def _is_whisper(model: str | None) -> bool:
    return model is not None and "whisper" in model.lower()


def _check_nemo_repo(model: str) -> None:
    if not NEMO_REPO.fullmatch(model):
        raise ToolError(f"nemo only loads Hugging Face repos named nvidia/<name>, not {model}")


# ---------------------------------------------------------------- ledger


@dataclass
class Instance:
    """A model that holds (or is about to hold) GPU memory."""

    engine: Engine
    model: str
    vram_mib: int
    vram_source: str  # measured | declared | estimate | ollama_ps
    record_key: str  # where its measured footprint is stored
    last_used: float = field(default_factory=time.time)
    starting: bool = False
    managed: bool = True  # False: an Ollama model a client loaded, not aias
    cpu_offload: bool = False  # Ollama put part of it in system memory
    since: float = field(default_factory=time.time)  # created or last started

    def settled(self) -> bool:
        return not self.starting and time.time() - self.since > SETTLE_SECS

    @property
    def key(self) -> str:
        return f"{self.engine}:{self.model}"


INSTANCES: dict[str, Instance] = {}
# Extra memory a running job holds on top of its instance (nemo's per-file peak).
BURSTS: dict[str, int] = {}
STORE = vram.Store()
# Held while models start, get evicted or bursts get reserved, so two
# admissions never plan against the same free memory.
PLACEMENT = asyncio.Lock()
QUEUES: dict[str, asyncio.Semaphore] = {}
CARD: dict[str, Any] = {"smi": None, "pressure": False, "unaccounted_mib": 0}


def _reserved() -> int:
    return sum(i.vram_mib for i in INSTANCES.values()) + sum(BURSTS.values())


def _total() -> int:
    return (CARD["smi"] or {}).get("total", 0)


def _vllm_config(
    model: str, max_model_len: int | None, util: float | None, vram_mib: int | None
) -> dict[str, Any]:
    """Launch settings and the memory need of a vLLM model. The need comes from a
    measurement of these exact settings, else the caller's declaration, else
    util x card."""
    total = _total()
    defaults = WHISPER_DEFAULTS if _is_whisper(model) else VLLM_DEFAULTS
    max_len = max_model_len or defaults["max_model_len"]
    declared = vram_mib or (round(util * total) if util else None)
    target = declared or round(defaults["gpu_memory_utilization"] * total)
    if _is_whisper(model):
        kv = WHISPER_KV_BYTES
    else:
        weights = _disk_mib("vllm", model) or 0
        kv_mib = max(MIN_KV_MIB, target - weights - VLLM_OVERHEAD_MIB)
        if (needed := _kv_mib_for(model, max_len)) and kv_mib < needed:
            raise ToolError(
                f"{model} would get a {kv_mib} MiB KV cache, but max_model_len {max_len} needs "
                f"{needed} MiB; pass vram_mib of at least {weights + VLLM_OVERHEAD_MIB + needed} "
                "or a smaller max_model_len"
            )
        kv = kv_mib * 2**20
    key = f"vllm:{model}:{max_len}:{kv}"
    # A declaration wins over a measurement, so a wrong measurement can be overridden.
    measured = STORE.measured(key)
    if declared:
        need, source = declared, "declared"
    elif measured:
        need, source = measured, "measured"
    else:
        need, source = target, "estimate"
    return {"max_len": max_len, "kv": kv, "record_key": key, "need": need, "source": source}


async def _ollama_model_info(model: str, start_server: bool) -> dict[str, Any] | None:
    """GGUF metadata from /api/show. Starting the server for it costs no GPU memory."""
    if start_server:
        with contextlib.suppress(RuntimeError):
            await _ensure_ollama_server()
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(f"{INTERNAL['ollama']}/api/show", json={"model": model})
        return resp.json().get("model_info") if resp.status_code == 200 else None
    except (httpx.HTTPError, ValueError):
        return None


async def _simple_need(
    engine: str, model: str, vram_mib: int | None, start_server: bool = True
) -> tuple[int, str, str]:
    """(need, source, record key) for ollama and nemo."""
    key = f"{engine}:{model}"
    if vram_mib:
        return vram_mib, "declared", key
    if measured := STORE.measured(key):
        return measured, "measured", key
    if engine == "nemo":
        return vram.NEMO_BASE_MIB, "estimate", key
    disk = await asyncio.to_thread(_disk_mib, "ollama", model) or 1024
    info = await _ollama_model_info(model, start_server)
    need, _ = vram.ollama_estimate_mib(disk, info)
    return need, "estimate", key


async def _ollama_ps() -> list[dict[str, Any]]:
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            return (await client.get(f"{INTERNAL['ollama']}/api/ps")).json().get("models", [])
    except (httpx.HTTPError, ValueError):
        return []


async def _sync_single(engine: Engine, running: dict[str, Any]) -> None:
    """Match the ledger to the vllm or nemo container, which may have been started,
    stopped or have crashed without aias (or before an MCP restart)."""
    inst = next((i for i in INSTANCES.values() if i.engine == engine), None)
    if engine not in running:
        if inst is not None and inst.settled():
            del INSTANCES[inst.key]
        return
    if inst is not None and inst.starting:
        return
    if engine == "vllm":
        args = await asyncio.to_thread(_vllm_args_from_container)
        model = args.get("model")
    else:
        model = await asyncio.to_thread(_nemo_model_from_container)
    if inst is not None:
        if inst.model == model:
            return
        # The container runs something else than the ledger says: rebuild from it.
        del INSTANCES[inst.key]
    if not model:
        return
    if engine == "vllm":
        key = f"vllm:{model}:{args.get('max-model-len')}:{args.get('kv-cache-memory')}"
        util = float(args.get("gpu-memory-utilization", 0) or 0)
        need, source = STORE.measured(key), "measured"
        if not need:
            need, source = round(util * _total()), "estimate"
    else:
        need, source, key = await _simple_need("nemo", model, None)
    INSTANCES[f"{engine}:{model}"] = Instance(engine, model, need, source, key)


async def _sync_ollama(running: dict[str, Any]) -> None:
    """Every loaded Ollama model is an instance, whoever loaded it; /api/ps is the
    truth for its size."""
    loaded = await _ollama_ps() if "ollama" in running else []
    seen = set()
    for m in loaded:
        name = m.get("name") or m.get("model")
        key = f"ollama:{name}"
        seen.add(key)
        mib = round(m.get("size_vram", 0) / 2**20) + vram.OLLAMA_OVERHEAD_MIB
        inst = INSTANCES.get(key)
        if inst is None:
            inst = INSTANCES[key] = Instance("ollama", name, mib, "ollama_ps", key)
        elif not inst.starting:
            inst.vram_mib, inst.vram_source = mib, "ollama_ps"
        inst.managed = STORE.managed(name)
        inst.cpu_offload = m.get("size_vram", 0) < m.get("size", 0)
    for key, inst in list(INSTANCES.items()):
        if inst.engine == "ollama" and key not in seen and inst.settled():
            # Unloaded behind aias's back (a client, or Ollama making room itself).
            STORE.manage(inst.model, False)
            del INSTANCES[key]


_RECONCILE_LOCK = asyncio.Lock()


async def _reconcile() -> None:
    async with _RECONCILE_LOCK:
        await _reconcile_once()


async def _reconcile_once() -> None:
    running = await asyncio.to_thread(_running)
    await _sync_single("vllm", running)
    await _sync_single("nemo", running)
    await _sync_ollama(running)
    card = await asyncio.to_thread(vram.smi)
    CARD["smi"] = card
    if card:
        # What nvidia-smi shows beyond the ledger is the desktop and anything
        # aias does not know about. Past the desktop reserve, stop admitting.
        CARD["unaccounted_mib"] = card["used"] - _reserved()
        CARD["pressure"] = CARD["unaccounted_mib"] > vram.DESKTOP_RESERVE_MIB


async def _reconcile_forever() -> None:
    while True:
        try:
            await _reconcile()
        except Exception:  # docker or nvidia-smi briefly unavailable
            log.exception("reconcile failed")
        await asyncio.sleep(RECONCILE_SECS)


_RECONCILER: asyncio.Task | None = None


def _ensure_reconciler() -> None:
    global _RECONCILER
    if _RECONCILER is None or _RECONCILER.done():
        _RECONCILER = asyncio.get_running_loop().create_task(_reconcile_forever())


def _jobs_of(key: str, exclude: Job | None = None) -> list[Job]:
    return [j for j in JOBS.values() if j.state == "running" and j.instance == key and j is not exclude]


# Instances an admission is stopping; no job may start on them and no pin may
# land on them meanwhile, so a plan is carried out whole or not at all.
EVICTING: set[str] = set()


def _evictable(key: str) -> bool:
    inst = INSTANCES.get(key)
    return (
        inst is not None
        and not inst.starting
        and key not in EVICTING
        and not STORE.pinned(key)
        and not _jobs_of(key)
    )


async def _carry_out(job: Job, plan: vram.Plan) -> list[str]:
    """Evict what plan lists, all or nothing: every target is checked again (in
    PLACEMENT, after the plan) and marked, so none can gain a job or a pin while
    the others are being stopped."""
    targets = [k for k in plan.replace + plan.evict if k in INSTANCES]
    blocked = [k for k in targets if not _evictable(k)]
    if blocked:
        raise RuntimeError(f"{', '.join(blocked)} became pinned or busy; plan again")
    EVICTING.update(targets)
    try:
        for key in targets:
            job.detail = f"stopping {key} to make room"
            await _evict(key)
    finally:
        EVICTING.difference_update(targets)
    return targets


def _plan(need: int, protect: set[str], replace: list[str]) -> vram.Plan:
    card = CARD["smi"] or {}
    candidates = [
        vram.Candidate(k, i.vram_mib, i.last_used)
        for k, i in INSTANCES.items()
        if k not in protect and k not in replace and _evictable(k)
    ]
    return vram.make_plan(
        need,
        vram.budget(card.get("total", 0)),
        _reserved(),
        card.get("free"),
        CARD["pressure"],
        candidates,
        [vram.Candidate(k, INSTANCES[k].vram_mib, INSTANCES[k].last_used) for k in replace if k in INSTANCES],
    )


def _refusal(plan: vram.Plan) -> dict[str, Any]:
    return {"refused": True, **plan.view()}


async def _evict(key: str) -> None:
    inst = INSTANCES.get(key)
    if inst is None:
        return
    if inst.engine == "ollama":
        await _unload_ollama(inst.model)
        STORE.manage(inst.model, False)
    else:
        await asyncio.to_thread(_stop, inst.engine)
    INSTANCES.pop(key, None)
    STORE.pin(key, False)


async def _wait_card_free(mib: int, secs: float = 30) -> None:
    """Freed memory shows on nvidia-smi a moment after a process exits."""
    deadline = time.monotonic() + secs
    while time.monotonic() < deadline:
        card = await asyncio.to_thread(vram.smi)
        if card is None or card["free"] >= mib:
            return
        await asyncio.sleep(1)


def _queue(key: str, size: int) -> asyncio.Semaphore:
    if key not in QUEUES:
        QUEUES[key] = asyncio.Semaphore(size)
    return QUEUES[key]


# ---------------------------------------------------------------- jobs


@dataclass
class Job:
    kind: str
    engine: Engine
    model: str
    instance: str | None = None  # the Instance key the job runs on or starts
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


def _start_job(job: Job, work: Any) -> dict[str, Any]:
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
            BURSTS.pop(job.id, None)
            if job.instance in INSTANCES and job.kind != "up":
                INSTANCES[job.instance].last_used = time.time()

    JOBS[job.id] = job
    job.task = asyncio.create_task(runner())
    return job.view()


def _replacements(engine: Engine, key: str) -> list[str]:
    """vllm and nemo hold one model: the other one, if any, has to go. Refused if it
    is pinned, starting or running a job."""
    if engine not in ("vllm", "nemo"):
        return []
    others = [i for i in INSTANCES.values() if i.engine == engine and i.key != key]
    for other in others:
        if other.starting or STORE.pinned(other.key) or _jobs_of(other.key) or other.key in EVICTING:
            raise ToolError(
                f"{engine} holds {other.model}, which is pinned, starting or running a job; "
                f"model_down model={other.model} first"
            )
    return [o.key for o in others]


async def _place(job: Job, inst: Instance, evict: str, start: Any) -> str:
    """Admit inst against the budget, evicting as the plan says if allowed, then
    start it. Holds PLACEMENT throughout: what to replace or evict is decided on
    the ledger as it is now, and the footprint measured around the start is this
    model's alone (it is not measured when anything was stopped for it)."""
    async with PLACEMENT:
        job.detail = "waiting for room on the GPU"
        await _reconcile()
        if (existing := INSTANCES.get(inst.key)) is not None:
            if existing.starting:
                raise RuntimeError(f"{inst.model} is already starting in another job")
            job.result = {"already_up": True, "evicted": [], "vram_mib": existing.vram_mib}
            return f"{inst.model} was already up"
        plan = _plan(inst.vram_mib, {inst.key}, _replacements(inst.engine, inst.key))
        if not plan.can_fit or (not plan.fits and evict != "auto"):
            job.result = _refusal(plan)
            raise RuntimeError(plan.reason or f"{inst.model} does not fit")
        evicted = await _carry_out(job, plan)
        if evicted:
            await _wait_card_free(inst.vram_mib + vram.SAFETY_MIB)
        quiet = not evicted and not any(
            j.state == "running" and j is not job and j.kind != "pull" for j in JOBS.values()
        )
        before = await asyncio.to_thread(vram.smi)
        inst.starting = True
        INSTANCES[inst.key] = inst
        try:
            detail = await start(job, inst)
        except BaseException:
            if INSTANCES.get(inst.key) is inst:
                del INSTANCES[inst.key]
            raise
        inst.starting = False
        inst.last_used = inst.since = time.time()
        if quiet and inst.engine != "ollama" and before:
            await asyncio.sleep(3)
            after = await asyncio.to_thread(vram.smi)
            if after and after["used"] > before["used"]:
                measured = after["used"] - before["used"]
                STORE.record(inst.record_key, measured)
                if inst.vram_source != "declared":  # a declaration stays in charge
                    inst.vram_mib, inst.vram_source = measured, "measured"
        job.result = {"evicted": evicted, "vram_mib": inst.vram_mib, "vram_source": inst.vram_source}
        return detail


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
        # Leave the server alone if a model got loaded in the meantime.
        busy = any(i.engine == "ollama" for i in INSTANCES.values()) or any(
            j.state == "running" and j.kind == "up" and j.engine == "ollama" for j in JOBS.values()
        )
        if started and not busy:
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


async def _start_ollama(job: Job, inst: Instance) -> str:
    await _ensure_ollama_server()
    job.detail = "loading into VRAM"
    async with httpx.AsyncClient(timeout=600) as client:
        resp = await client.post(
            f"{INTERNAL['ollama']}/api/generate", json={"model": inst.model, "keep_alive": -1}
        )
        if resp.status_code != 200:
            raise RuntimeError(f"ollama could not load {inst.model}: {resp.text[:500]}")
    STORE.manage(inst.model, True)
    inst.managed = True
    loaded = next((m for m in await _ollama_ps() if (m.get("name") or m.get("model")) == inst.model), None)
    if loaded:
        inst.vram_mib = round(loaded.get("size_vram", 0) / 2**20) + vram.OLLAMA_OVERHEAD_MIB
        inst.vram_source = "ollama_ps"
        inst.cpu_offload = loaded.get("size_vram", 0) < loaded.get("size", 0)
        STORE.record(inst.record_key, inst.vram_mib)
    return f"{inst.model} is up at {PUBLIC['ollama']}"


def _start_vllm(cfg: dict[str, Any]) -> Any:
    async def start(job: Job, inst: Instance) -> str:
        total = _total() or 1
        # vLLM refuses to start unless util x card is free; with --kv-cache-memory
        # that check is all util still does, so ask for exactly the budgeted need.
        util = min(0.95, math.ceil(inst.vram_mib / total * 1000) / 1000)
        env = {
            "VLLM_MODEL": inst.model,
            "VLLM_MAX_LEN": str(cfg["max_len"]),
            "VLLM_GPU_UTIL": str(util),
            "VLLM_KV_BYTES": str(cfg["kv"]),
        }
        await asyncio.to_thread(_run, "up", "-d", "--force-recreate", "vllm", env=env)
        job.detail = "starting (weights, compile, warmup)"

        def alive() -> bool:
            return "vllm" in _running()

        if await _wait_http(f"{INTERNAL['vllm']}/v1/models", VLLM_STARTUP_SECS, alive):
            return f"{inst.model} is up at {PUBLIC['vllm']}"
        tail = await asyncio.to_thread(_run, "logs", "--no-log-prefix", "--tail", "40", "vllm")
        with contextlib.suppress(RuntimeError):
            await asyncio.to_thread(_stop, "vllm")
        raise RuntimeError(f"vLLM did not become ready. Last log lines:\n{tail}")

    return start


async def _start_nemo(job: Job, inst: Instance) -> str:
    job.detail = "building the nemo image (about 5 minutes the first time, seconds after)"
    _BUILD_CANCELLED.clear()
    await asyncio.to_thread(_build_nemo)
    await asyncio.to_thread(_run, "up", "-d", "--force-recreate", "nemo", env={"NEMO_MODEL": inst.model})
    job.detail = "loading the model onto the GPU"

    def alive() -> bool:
        return "nemo" in _running()

    if await _wait_http(f"{INTERNAL['nemo']}/health", NEMO_STARTUP_SECS, alive):
        return f"{inst.model} is up; call diarize"
    tail = await asyncio.to_thread(_run, "logs", "--no-log-prefix", "--tail", "40", "nemo")
    with contextlib.suppress(RuntimeError):
        await asyncio.to_thread(_stop, "nemo")
    raise RuntimeError(f"nemo did not become ready. Last log lines:\n{tail}")


async def _audio_job(job: Job, audio_url: str, engine: str, send: Any) -> dict[str, Any]:
    """Fetch audio_url as 16 kHz mono WAV (audio.py enforces every limit), then hand
    it to send(client, wav, audio_s), which calls the engine and returns its reply."""
    try:
        async with asyncio.timeout(AUDIO_JOB_SECS):
            with tempfile.TemporaryDirectory() as tmp:
                job.detail = "downloading and decoding the audio"
                wav, audio_s = await fetch_wav(audio_url, Path(tmp))
                job.detail = f"processing {audio_s / 60:.0f} min of audio"
                async with httpx.AsyncClient(timeout=httpx.Timeout(AUDIO_JOB_SECS, connect=10)) as client:
                    resp = await send(client, wav, audio_s)
    except AudioError as exc:
        raise RuntimeError(str(exc)) from None
    except TimeoutError:
        raise RuntimeError(f"{job.kind} did not finish within {AUDIO_JOB_SECS} s") from None
    except httpx.HTTPError as exc:
        raise RuntimeError(f"could not reach the {engine} engine: {type(exc).__name__} {exc}".rstrip()) from exc
    try:
        body = resp.json()
    except ValueError:
        body = {"error": resp.text[-1000:]}
    if resp.status_code != 200:
        err = body.get("error")
        if isinstance(err, dict):  # vLLM wraps errors as {"error": {"message": ...}}
            err = err.get("message")
        raise RuntimeError(err or f"{engine} answered HTTP {resp.status_code}")
    return body


async def _wav_chunks(wav: Path) -> Any:
    with wav.open("rb") as f:
        while chunk := await asyncio.to_thread(f.read, 1 << 20):
            yield chunk


async def _reserve_burst(job: Job, mib: int, evict: str) -> None:
    """Hold mib on top of the job's instance for as long as the job runs."""
    async with PLACEMENT:
        await _reconcile()
        plan = _plan(mib, {job.instance or ""}, [])
        if not plan.can_fit or (not plan.fits and evict != "auto"):
            job.result = _refusal(plan)
            raise RuntimeError(plan.reason or f"the {mib} MiB this job needs does not fit")
        await _carry_out(job, plan)
        BURSTS[job.id] = mib


async def _diarize(job: Job, audio_url: str, mode: str, evict: str) -> str:
    reserved_burst: dict[str, int] = {}

    async def send(client: httpx.AsyncClient, wav: Path, audio_s: float) -> httpx.Response:
        job.detail = "waiting for the nemo engine"
        async with _queue(job.instance or "nemo", 1):
            burst = vram.nemo_burst_mib(audio_s, STORE.burst_rate(job.model))
            await _reserve_burst(job, burst, evict)
            reserved_burst["mib"] = burst
            job.detail = f"diarizing {audio_s / 60:.0f} min of audio"
            try:
                return await client.post(
                    f"{INTERNAL['nemo']}/diarize",
                    params={"mode": mode},
                    content=_wav_chunks(wav),
                    headers={"Content-Type": "audio/wav"},
                )
            finally:
                BURSTS.pop(job.id, None)

    body = await _audio_job(job, audio_url, "nemo", send)
    if body.get("burst_mib") and body.get("audio_s"):
        STORE.record_burst(job.model, body["burst_mib"] / (body["audio_s"] / 60))
    body["reserved_burst_mib"] = reserved_burst.get("mib")
    job.result = body
    speakers = ", ".join(f"{s['speaker']} {s['seconds']:.0f} s" for s in body["speakers"]) or "no speech"
    return f"{len(body['speakers'])} speakers in {body['audio_s']:.0f} s of audio: {speakers}"


def _whisper_language(language: str) -> str:
    """zh-TW, zh-Hant, zh_CN and the like to the bare ISO 639-1 code Whisper takes."""
    return language.replace("_", "-").split("-")[0].strip().lower()


async def _transcribe(job: Job, audio_url: str, language: str, traditional: bool) -> str:
    timing: dict[str, float] = {}
    code = _whisper_language(language)

    async def send(client: httpx.AsyncClient, wav: Path, audio_s: float) -> httpx.Response:
        timing["audio_s"] = audio_s
        job.detail = "waiting for the vllm engine"
        async with _queue(job.instance or "vllm", TRANSCRIBE_CONCURRENCY):
            job.detail = f"transcribing {audio_s / 60:.0f} min of audio"
            started = time.monotonic()
            with wav.open("rb") as f:
                resp = await client.post(
                    f"{INTERNAL['vllm']}/v1/audio/transcriptions",
                    data={
                        "model": job.model,
                        "language": code,
                        "response_format": "verbose_json",
                        "temperature": "0",
                    },
                    files={"file": ("audio.wav", f, "audio/wav")},
                )
            timing["elapsed_s"] = time.monotonic() - started
        return resp

    body = await _audio_job(job, audio_url, "vllm", send)
    # OpenCC only makes sense for Chinese; it would leave other text alone, but skip it.
    traditional = traditional and code == "zh"
    convert = _S2TW.convert if traditional else (lambda text: text)
    segments = [
        {"start": round(seg["start"], 2), "end": round(seg["end"], 2), "text": convert(seg["text"].strip())}
        for seg in body.get("segments") or []
    ]
    job.result = {
        "model": job.model,
        "language": code,
        "traditional": traditional,
        "audio_s": round(timing["audio_s"], 1),
        "elapsed_s": round(timing["elapsed_s"], 2),
        "text": convert(body.get("text", "").strip()),
        "segments": segments,
    }
    return f"{len(segments)} segments from {timing['audio_s']:.0f} s of audio in {timing['elapsed_s']:.0f} s"


# ---------------------------------------------------------------- tools


def _iso(ts: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))


@mcp.tool()
async def status() -> dict[str, Any]:
    """What is running now: one entry per model that is up (engine, OpenAI compatible
    base_url, vram_mib and where that number comes from, pinned, last_used, its jobs),
    the GPU budget (budget_mib, reserved_mib, free_mib, pressure) and jobs in progress.
    Call this first."""
    _ensure_reconciler()
    await _reconcile()
    running = await asyncio.to_thread(_running)
    up = []
    for inst in sorted(INSTANCES.values(), key=lambda i: i.key):
        if inst.engine == "vllm":
            ready = not inst.starting and await _wait_http(f"{INTERNAL['vllm']}/v1/models", 0.1)
        elif inst.engine == "nemo":
            ready = not inst.starting and await _wait_http(f"{INTERNAL['nemo']}/health", 0.1)
        else:
            ready = not inst.starting
        entry = {
            "engine": inst.engine,
            "model": inst.model,
            "models": [inst.model],
            "base_url": PUBLIC.get(inst.engine),
            "ready": ready,
            "vram_mib": inst.vram_mib,
            "vram_source": inst.vram_source,
            "pinned": STORE.pinned(inst.key),
            "last_used": _iso(inst.last_used),
            "jobs": [j.id for j in _jobs_of(inst.key)],
        }
        if inst.engine == "ollama":
            entry["managed"] = inst.managed
            entry["cpu_offload"] = inst.cpu_offload
        up.append(entry)
    if "ollama" in running and not any(i.engine == "ollama" for i in INSTANCES.values()):
        # The server is up with nothing loaded; keep the old one-entry shape.
        up.append({"engine": "ollama", "model": None, "models": [], "base_url": PUBLIC["ollama"],
                   "ready": True, "vram_mib": 0, "vram_source": "ollama_ps", "pinned": False,
                   "last_used": None, "jobs": []})
    card = CARD["smi"]
    gpu = None
    if card:
        budget = vram.budget(card["total"])
        gpu = {
            "name": card["name"],
            "memory_used_mib": card["used"],
            "memory_total_mib": card["total"],
            "budget_mib": budget,
            "reserved_mib": _reserved(),
            "free_mib": budget - _reserved(),
            "pressure": CARD["pressure"],
            "unaccounted_mib": CARD["unaccounted_mib"],
            # A client loaded more into Ollama than the budget allows; under WSL
            # the driver spills it to system memory instead of failing.
            "over_budget": _reserved() > budget,
        }
    return {"up": up, "gpu": gpu, "jobs": [j.view() for j in JOBS.values() if j.state == "running"]}


@mcp.tool()
async def model_list(engine: Engine | None = None) -> list[dict[str, Any]]:
    """Models already downloaded on this machine, with size on disk, the GPU memory each
    needs (vram_mib, vram_source) and whether it is loaded. Filter by engine."""
    _ensure_reconciler()
    if CARD["smi"] is None:
        CARD["smi"] = await asyncio.to_thread(vram.smi)
    models: list[dict[str, Any]] = []
    if engine in (None, "ollama"):
        models += await asyncio.to_thread(_ollama_models)
    if engine != "ollama":
        models += [m for m in await asyncio.to_thread(_hf_models) if engine in (None, m["engine"])]
    for m in models:
        inst = INSTANCES.get(f"{m['engine']}:{m['model']}")
        m["loaded"] = inst is not None
        if inst is not None:
            m["vram_mib"], m["vram_source"] = inst.vram_mib, inst.vram_source
        elif m["engine"] == "vllm":
            try:
                cfg = await asyncio.to_thread(_vllm_config, m["model"], None, None, None)
                m["vram_mib"], m["vram_source"] = cfg["need"], cfg["source"]
            except ToolError:  # the defaults do not fit its context; model_up says why
                m["vram_mib"], m["vram_source"] = None, "too_small"
        else:
            m["vram_mib"], m["vram_source"], _ = await _simple_need(m["engine"], m["model"], None, start_server=False)
    return models


@mcp.tool()
async def model_pull(engine: Engine, model: str) -> dict[str, Any]:
    """Download a model. ollama takes library names like qwen3:8b; vllm and nemo take Hugging
    Face repo ids like Qwen/Qwen3-0.6B or nvidia/Nemotron-3-Diarization. Returns a job;
    poll job_status."""
    _ensure_reconciler()
    if engine == "nemo":
        _check_nemo_repo(model)
    work = _pull_ollama if engine == "ollama" else _pull_hf
    return _start_job(Job("pull", engine, model), work)


@mcp.tool()
async def model_up(
    engine: Engine,
    model: str,
    max_model_len: int | None = None,
    gpu_memory_utilization: float | None = None,
    vram_mib: StrictInt | None = None,  # strict: true or "3000" is refused, not coerced
    pin: bool | None = None,
    evict: Evict = "never",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Load a model so it serves requests, next to whatever else fits on the GPU.

    Admission: the models already up plus this one must fit the VRAM budget. If it
    does not, nothing happens and the reply is a plan {refused, fits, need_mib,
    free_mib, evict: [...]}; call again with evict="auto" to stop the models in evict
    (least recently used first, never pinned ones or ones running a job). dry_run
    returns the plan without doing anything. vram_mib (MiB, at most the card) declares
    the need and wins over aias's own measurement, to correct a wrong one. pin=true
    keeps it from being evicted (false unpins).
    vllm holds one model: asking for another replaces it. max_model_len and
    gpu_memory_utilization apply to vllm only; left out, they are 8192 and 0.8, or 448
    and 0.4 for a Whisper model. vLLM startup takes 40 s to several minutes; the first
    nemo start builds its image and takes about 5 minutes. Returns a job; when it is
    done, status shows the base_url (nemo has none: use diarize)."""
    _ensure_reconciler()
    await _reconcile()
    if engine == "nemo":
        _check_nemo_repo(model)
    if engine == "ollama":
        model = _ollama_name(model)
    total = _total()
    if vram_mib is not None and (vram_mib <= 0 or (total and vram_mib > total)):
        raise ToolError(f"vram_mib must be a whole number of MiB from 1 to {total or 'the card size'}, not {vram_mib}")
    if gpu_memory_utilization is not None and not 0 < gpu_memory_utilization <= 1:
        raise ToolError(f"gpu_memory_utilization must be above 0 and at most 1, not {gpu_memory_utilization}")
    if max_model_len is not None and max_model_len <= 0:
        raise ToolError(f"max_model_len must be positive, not {max_model_len}")
    if engine == "vllm":
        cfg = await asyncio.to_thread(_vllm_config, model, max_model_len, gpu_memory_utilization, vram_mib)
        need, source, record_key = cfg["need"], cfg["source"], cfg["record_key"]
    else:
        cfg = {}
        need, source, record_key = await _simple_need(engine, model, vram_mib)
    key = f"{engine}:{model}"

    current = INSTANCES.get(key)
    if current is not None:
        if current.starting:
            raise ToolError(f"{model} is already starting; follow its model_up job with job_status")
        if current.key in EVICTING:
            raise ToolError(f"{model} is being stopped to make room for another model")
        if pin is not None and not dry_run:
            STORE.pin(key, pin)
        if engine == "ollama" and not current.managed and not dry_run:
            # A client loaded it; take it over so it stays loaded.
            job = Job("up", engine, model, instance=key)
            return _start_job(job, lambda j: _start_ollama(j, current))
        return {"already_up": True, "model": model, "engine": engine, "vram_mib": current.vram_mib,
                "pinned": STORE.pinned(key)}

    # A preview for the reply; the job decides again once it holds PLACEMENT.
    plan = _plan(need, {key}, _replacements(engine, key))
    if dry_run:
        return {"dry_run": True, "model": model, "engine": engine, "vram_source": source, **plan.view()}
    if not plan.can_fit or (not plan.fits and evict != "auto"):
        return {"model": model, "engine": engine, "vram_source": source, **_refusal(plan)}

    inst = Instance(engine, model, need, source, record_key)
    start = _start_ollama if engine == "ollama" else _start_nemo if engine == "nemo" else _start_vllm(cfg)

    async def work(job: Job) -> str:
        detail = await _place(job, inst, evict, start)
        if pin:
            STORE.pin(key, True)
        return detail

    return _start_job(Job("up", engine, model, instance=key), work)


def _cancel(jobs: list[Job]) -> list[asyncio.Task]:
    tasks = []
    for job in jobs:
        if job.task is not None and not job.task.done():
            if job.kind == "up" and job.engine == "nemo":
                # A build holds the compose lock for minutes; end it so _stop gets the lock.
                _cancel_build()
            job.task.cancel()
            tasks.append(job.task)
    return tasks


def _stop_if_running(engine: Engine, names: set[str]) -> bool:
    """Stop the vllm or nemo container if it runs one of names. The check and the
    stop share the compose lock, so an up still finishing in a cancelled job's
    thread completes first and is then seen."""
    with _COMPOSE_LOCK:
        if engine not in _running():
            return False
        running = _vllm_model_from_container() if engine == "vllm" else _nemo_model_from_container()
        if running not in names:
            return False
        _stop(engine)
        return True


async def _unload_ollama(name: str) -> None:
    async with httpx.AsyncClient(timeout=60) as client:
        with contextlib.suppress(httpx.HTTPError):
            await client.post(f"{INTERNAL['ollama']}/api/generate", json={"model": name, "keep_alive": 0})
    for _ in range(30):
        if not any((m.get("name") or m.get("model")) == name for m in await _ollama_ps()):
            return
        await asyncio.sleep(0.5)


@mcp.tool()
async def model_down(model: str | None = None) -> dict[str, Any]:
    """Stop models and free their GPU memory. With model, stop only that one (pinned or
    not, loaded or still starting) and cancel only its jobs. Without, stop every engine
    and cancel every pull, up, diarize or transcribe job, as before."""
    _ensure_reconciler()
    if model is None:
        tasks = _cancel([j for j in JOBS.values() if j.state == "running"])
        _cancel_build()
        if tasks:
            await asyncio.wait(tasks, timeout=30)
        # No reconcile may read the containers mid-stop and put a model back.
        async with _RECONCILE_LOCK:
            await asyncio.to_thread(_stop, *ENGINES)
            for inst in list(INSTANCES.values()):
                STORE.pin(inst.key, False)
                if inst.engine == "ollama":
                    STORE.manage(inst.model, False)
            INSTANCES.clear()
            BURSTS.clear()
            CARD["smi"] = await asyncio.to_thread(vram.smi)
        return {"stopped": list(ENGINES), "gpu": _gpu_view()}

    names = {model, _ollama_name(model)}
    jobs = [j for j in JOBS.values() if j.state == "running" and j.kind != "pull" and j.model in names]
    tasks = _cancel(jobs)
    if tasks:
        await asyncio.wait(tasks, timeout=30)
    stopped: list[str] = []
    async with _RECONCILE_LOCK:
        # Go by what the engines run, not the ledger: a model still starting (or
        # one whose job was just cancelled) may have no ledger entry.
        for engine in ("vllm", "nemo"):
            if await asyncio.to_thread(_stop_if_running, engine, names):
                stopped.append(f"{engine}:{model}")
        loaded = {(m.get("name") or m.get("model")) for m in await _ollama_ps()}
        for name in names & loaded:
            await _unload_ollama(name)
            STORE.manage(name, False)
            stopped.append(f"ollama:{name}")
        for inst in [i for i in INSTANCES.values() if i.model in names]:
            INSTANCES.pop(inst.key, None)
            STORE.pin(inst.key, False)
            if inst.key not in stopped:
                stopped.append(inst.key)
        CARD["smi"] = await asyncio.to_thread(vram.smi)
    if not stopped and not jobs:
        raise ToolError(f"{model} is not up; status lists what is")
    return {"stopped": stopped, "cancelled_jobs": [j.id for j in jobs], "gpu": _gpu_view()}


def _gpu_view() -> dict[str, Any] | None:
    card = CARD["smi"]
    if not card:
        return None
    return {"name": card["name"], "memory_used_mib": card["used"], "memory_total_mib": card["total"]}


def _up_instance(engine: Engine, check: Any) -> Instance | None:
    inst = next((i for i in INSTANCES.values() if i.engine == engine and check(i.model)), None)
    return None if inst is None or inst.starting else inst


@mcp.tool()
async def diarize(audio_url: str, mode: Mode = "offline", evict: Evict = "never") -> dict[str, Any]:
    """Label who spoke when in an audio file, up to 8 speakers, with the nemo engine
    (model_up engine=nemo first). audio_url is an http(s) link the server downloads; any
    format ffmpeg reads, up to 2 GB. mode trades accuracy for latency: offline (best),
    low (1.04 s), verylow (0.64 s), ultralow (0.32 s); the streaming modes are slower on
    a whole file. A run needs about 40 MiB of GPU memory per minute of audio on top of
    the model (measured after the first run); if that does not fit, the job fails with a
    plan, and evict="auto" lets it stop least recently used models. Returns a job; when it is done, job_status carries
    result with the RTTM text and the seconds each speaker talked."""
    _ensure_reconciler()
    await _reconcile()
    inst = _up_instance("nemo", lambda m: True)
    if inst is None:
        raise ToolError("nemo is not up: run model_up engine=nemo model=nvidia/Nemotron-3-Diarization first")
    if inst.key in EVICTING:
        raise ToolError(f"{inst.model} is being stopped to make room for another model")
    job = Job("diarize", "nemo", inst.model, instance=inst.key)
    inst.last_used = time.time()
    return _start_job(job, lambda j: _diarize(j, audio_url, mode, evict))


@mcp.tool()
async def transcribe(audio_url: str, language: str = "zh", traditional: bool = True) -> dict[str, Any]:
    """Speech to text with timestamps, with a Whisper model on vllm (model_up engine=vllm
    model=openai/whisper-large-v3 first). audio_url is an http(s) link the server
    downloads; any format ffmpeg reads, up to 2 GB and 2 hours. language is an ISO 639-1
    code; a tag like zh-TW or zh_CN is cut to zh. traditional converts simplified Chinese
    characters to traditional (Taiwan), because Whisper drifts to simplified; it applies
    only to zh. Several transcriptions run at once. Returns a job; when it is done,
    job_status carries result with the full text and segments (start, end, text) in
    seconds."""
    _ensure_reconciler()
    await _reconcile()
    inst = _up_instance("vllm", _is_whisper)
    if inst is None:
        raise ToolError(
            "no Whisper model is up: run model_up engine=vllm model=openai/whisper-large-v3 first"
        )
    if inst.key in EVICTING:
        raise ToolError(f"{inst.model} is being stopped to make room for another model")
    if not await _wait_http(f"{INTERNAL['vllm']}/v1/models", 0.1):
        raise ToolError(f"{inst.model} is still starting; wait for its model_up job to finish")
    inst.last_used = time.time()
    return _start_job(
        Job("transcribe", "vllm", inst.model, instance=inst.key),
        lambda j: _transcribe(j, audio_url, language, traditional),
    )


@mcp.tool()
async def job_status(job_id: str, wait_seconds: int = 30) -> dict[str, Any]:
    """Progress of a pull, up, diarize or transcribe job. Blocks up to wait_seconds (max 120)
    for it to finish. A finished diarize or transcribe job carries its output in result;
    a refused one carries the plan."""
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
    _ensure_reconciler()
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
