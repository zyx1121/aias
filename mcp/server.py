"""aias MCP server: pull, start and stop local models on one GPU, diarize or
transcribe audio, and generate sound effects.

Runs in a container next to the engines and drives them through the Docker
socket with the same compose file a person would use by hand. Several models
share the card: one Ollama server (any number of models), up to 10 vLLM models
(one container and port each), one nemo model and one audio model, admitted
against a VRAM budget (vram.py).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import os
import re
import shutil
import stat
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
from starlette.responses import FileResponse, JSONResponse, Response

import align
import vram
from audio import MAX_AUDIO_SECS, WORK, AudioError, fetch_wav

Engine = Literal["ollama", "vllm", "nemo", "audio"]
ENGINES: tuple[Engine, ...] = ("ollama", "vllm", "nemo", "audio")
# Engines that run one model in their compose service's single container.
SINGLE: tuple[Engine, ...] = ("nemo", "audio")
Mode = Literal["offline", "low", "verylow", "ultralow"]
Evict = Literal["never", "auto"]

COMPOSE = ["docker", "compose", "-f", os.environ.get("AIAS_COMPOSE", "/opt/aias/compose.yaml")]
PORT = 11400
# Inside the compose network the engines answer on their service names; each
# vLLM model is its own container, named by its port (VLLM_PORTS).
INTERNAL = {"ollama": "http://ollama:11434", "nemo": "http://nemo:8100", "audio": "http://audio:8200"}
# What a client on the Windows host uses. nemo and audio have no port; diarize
# and generate_audio reach them.
PUBLIC = {"ollama": "http://127.0.0.1:11434/v1"}
# One port per vLLM model, lowest free first, so a lone model is on 8000 as before.
VLLM_PORTS = range(8000, 8010)
# Labels on vLLM containers are the truth about them: `compose stop` does not
# reach containers made by `compose run`, and an MCP restart rebuilds from them.
LABEL = "aias"
OLLAMA_MANIFESTS = Path("/ollama/models/manifests")
VLLM_STARTUP_SECS = 900
NEMO_STARTUP_SECS = 600
# Image builds of the single-model engines (nemo, audio), on their first start.
BUILD_SECS = 3600
# nemo loads .nemo archives, which can carry pickled code: only NVIDIA's repos.
# One path segment under nvidia/, so `nvidia/../other/repo` cannot slip through.
NEMO_REPO = re.compile(r"nvidia/(?!\.+$)[A-Za-z0-9_.-]+")
# AudioCraft loads its weights with torch.load (pickle, which can carry code),
# so audio takes only these repos, each with the text encoder it loads too.
AUDIO_REPOS: dict[str, list[str]] = {"facebook/audiogen-medium": ["t5-large"]}
AUDIO_HELPERS = {h for helpers in AUDIO_REPOS.values() for h in helpers}
AUDIO_HELPER_FILES = ["config.json", "tokenizer.json", "spiece.model", "model.safetensors"]
AUDIO_STARTUP_SECS = 600
# One generate_audio request: a 30 s clip takes about 100 s on an RTX 3080.
GENERATE_SECS = 600
AUDIO_MIN_S, AUDIO_MAX_S = 0.5, 30.0
# AudioCraft pins torch 2.1 (CUDA 12.1), which has kernels up to compute
# capability 9.0: RTX 50 series cards (12.0) cannot run it.
AUDIO_MAX_COMPUTE_CAP = 9.0
AUDIO_MAX_PROMPT = 500
# Generated sound files, served on /files/<name> to the host (and through an SSH
# tunnel to the agent's machine). Kept a day and at most 1 GB, oldest out first.
OUT = Path(os.environ.get("AIAS_OUT", "/out"))
OUT_KEEP_SECS = 24 * 3600
OUT_MAX_BYTES = 1 << 30
OUT_NAME = re.compile(r"[0-9a-f]{32}\.wav")
LOCAL_HOSTS = [f"localhost:{PORT}", f"127.0.0.1:{PORT}"]
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
# Audio work directories; any left over from a previous run are removed at start.
AUDIO_TMP_PREFIX = "aias-audio-"
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
        "Local model host on one NVIDIA GPU. Four engines: ollama (names like qwen3:8b), "
        "vllm (Hugging Face ids like Qwen/Qwen3-0.6B, or openai/whisper-large-v3 for speech "
        "to text), nemo, speaker diarization (nvidia/Nemotron-3-Diarization), and audio, "
        "sound effects from text (facebook/audiogen-medium). Models share the card within "
        "a VRAM budget: any number of ollama models, up to 10 vllm models, one nemo and "
        "one audio model at a time. model_up refuses a model that does not fit and "
        "returns a plan (fits, need_mib, free_mib, evict); pass evict=\"auto\" to stop the "
        "least recently used models in that plan, dry_run=true to only see it, and "
        "pin=true to keep a model from being evicted. Pulls, startups, diarize, "
        "transcribe and generate_audio runs are jobs: poll job_status until each finishes. Once an ollama or "
        "vllm model is up, call it through the OpenAI compatible base_url that status "
        "returns. Once nemo is up, call diarize with an audio URL; once a whisper model is "
        "up on vllm, call transcribe; once audio is up, call generate_audio with a prompt "
        "and download the WAV from the url in its result. The finished job carries the "
        "output in result. "
        "model_down with a model stops only that one; without one it stops everything."
    ),
)


# ---------------------------------------------------------------- helpers


_NO_LOCK = contextlib.nullcontext()
# Serializes compose calls that build, start or stop containers. A cancelled
# job's `up` keeps running in its worker thread, so a later `stop` must wait for it.
# Reentrant, so a check-then-stop can hold it across both steps.
_COMPOSE_LOCK = threading.RLock()
# The nemo or audio image build in progress, per service, so model_down can end
# it instead of waiting minutes for the lock, without touching the other one's.
_BUILDS: dict[str, subprocess.Popen[str]] = {}
# Set by model_down so a build still waiting for the lock never starts.
_BUILD_CANCELLED: dict[str, threading.Event] = {engine: threading.Event() for engine in SINGLE}


def _run(*args: str, env: dict[str, str] | None = None, timeout: float = 600) -> str:
    mutating = args[0] in ("up", "stop", "build", "run")
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
    # `compose run` containers (the vLLM models) show up as the vllm service too;
    # they are tracked by their labels instead.
    return {
        r["Service"]: r for r in rows
        if r.get("State") == "running" and "com.docker.compose.oneoff=True" not in (r.get("Labels") or "")
    }


def _inspect(container: str, fmt: str) -> list[str]:
    out = subprocess.run(["docker", "inspect", "-f", fmt, container], capture_output=True, text=True)
    if out.returncode != 0:
        return []
    return json.loads(out.stdout or "[]") or []


def _vllm_name(port: int) -> str:
    return f"aias-vllm-{port}"


def _vllm_url(port: int) -> str:
    return f"http://{_vllm_name(port)}:8000"


def _vllm_containers() -> list[dict[str, Any]]:
    """Every vLLM container aias made, running or not: name, state and labels."""
    ids = subprocess.run(
        ["docker", "ps", "-aq", "--filter", f"label={LABEL}.engine=vllm"], capture_output=True, text=True
    ).stdout.split()
    if not ids:
        return []
    out = subprocess.run(["docker", "inspect", *ids], capture_output=True, text=True)
    rows = []
    for c in json.loads(out.stdout or "[]"):
        labels = c.get("Config", {}).get("Labels") or {}
        rows.append({
            "name": c.get("Name", "").lstrip("/"),
            "running": c.get("State", {}).get("Running", False),
            "labels": {k[len(LABEL) + 1:]: v for k, v in labels.items() if k.startswith(f"{LABEL}.")},
        })
    return rows


def _remove_vllm(names: list[str]) -> None:
    if names:
        with _COMPOSE_LOCK:
            subprocess.run(["docker", "rm", "-f", *names], capture_output=True, text=True, timeout=120)


def _model_env(engine: str) -> str:
    """The variable that names the model of a single-model engine (NEMO_MODEL)."""
    return f"{engine.upper()}_MODEL"


def _single_model(engine: str) -> str | None:
    """The model the nemo or audio container was started with."""
    for var in _inspect(f"aias-{engine}-1", "{{json .Config.Env}}"):
        if var.startswith(f"{_model_env(engine)}="):
            return var.split("=", 1)[1]
    return None


def _build(service: str) -> None:
    """Build the nemo or audio image. Runs on every model_up: an unchanged
    directory is a cache hit in seconds, and a changed one (after an upgrade)
    gets rebuilt."""
    with _COMPOSE_LOCK:
        if _BUILD_CANCELLED[service].is_set():
            raise RuntimeError(f"{service} image build cancelled by model_down")
        proc = subprocess.Popen(
            [*COMPOSE, "build", service], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        _BUILDS[service] = proc
        try:
            out, _ = proc.communicate(timeout=BUILD_SECS)
        except subprocess.TimeoutExpired:
            proc.kill()
            out, _ = proc.communicate()
        finally:
            _BUILDS.pop(service, None)
    if proc.returncode != 0:
        raise RuntimeError(f"building the {service} image failed (exit {proc.returncode}): {out.strip()[-2000:]}")


def _cancel_build(service: str | None = None) -> None:
    """End the build of service, or of every single-model engine."""
    for name in (service,) if service else SINGLE:
        _BUILD_CANCELLED[name].set()
        if (proc := _BUILDS.get(name)) is not None and proc.poll() is None:
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
    """Hugging Face repos in the shared cache. A repo with a .nemo file is for nemo;
    the AudioGen repos and their text encoder are for audio."""
    try:
        cache = scan_cache_dir()
    except Exception:
        return []
    models = []
    for repo in cache.repos:
        if repo.repo_type != "model":
            continue
        files = {f.file_name for rev in repo.revisions for f in rev.files}
        if any(name.endswith(".nemo") for name in files):
            engine = "nemo"
        elif repo.repo_id in AUDIO_REPOS or repo.repo_id in AUDIO_HELPERS:
            engine = "audio"
        else:
            engine = "vllm"
        entry = {"engine": engine, "model": repo.repo_id, "size_gb": round(repo.size_on_disk / 1e9, 2)}
        if repo.repo_id in AUDIO_HELPERS:
            entry["used_by"] = sorted(m for m, helpers in AUDIO_REPOS.items() if repo.repo_id in helpers)
        models.append(entry)
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


def _check_audio_repo(model: str) -> None:
    if model not in AUDIO_REPOS:
        raise ToolError(f"audio loads only {', '.join(AUDIO_REPOS)}, not {model}")


def _check_audio_gpu() -> None:
    cap = vram.compute_cap()
    if cap is not None and cap > AUDIO_MAX_COMPUTE_CAP:
        raise ToolError(
            f"the audio engine runs on torch 2.1, which has no kernels for this GPU (compute capability "
            f"{cap:g}, above {AUDIO_MAX_COMPUTE_CAP:g}); RTX 50 series cards are not supported yet"
        )


def _check_repo(engine: str, model: str) -> None:
    if engine == "nemo":
        _check_nemo_repo(model)
    elif engine == "audio":
        _check_audio_repo(model)


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
    port: int | None = None  # vLLM only

    @property
    def url(self) -> str:
        return _vllm_url(self.port) if self.engine == "vllm" and self.port else INTERNAL[self.engine]

    @property
    def base_url(self) -> str | None:
        if self.engine == "vllm":
            return f"http://127.0.0.1:{self.port}/v1" if self.port else None
        return PUBLIC.get(self.engine)

    def settled(self) -> bool:
        return not self.starting and time.time() - self.since > SETTLE_SECS

    @property
    def key(self) -> str:
        return f"{self.engine}:{self.model}"


INSTANCES: dict[str, Instance] = {}
# Extra memory a running job holds on top of its instance (nemo's per-file peak,
# audio's per-clip peak).
BURSTS: dict[str, int] = {}
STORE = vram.Store()
# Held while models start, get evicted or bursts get reserved, so two
# admissions never plan against the same free memory.
PLACEMENT = asyncio.Lock()
QUEUES: dict[str, asyncio.Semaphore] = {}
# Bumped whenever GPU memory is claimed or released (a start, an eviction, a
# burst), so a start can tell whether its before/after measurement is its own.
_EPOCH = [0]


def _touch() -> None:
    _EPOCH[0] += 1
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
    """(need, source, record key) for ollama, nemo and audio."""
    key = f"{engine}:{model}"
    if vram_mib:
        return vram_mib, "declared", key
    if measured := STORE.measured(key):
        return measured, "measured", key
    if engine == "nemo":
        return vram.NEMO_BASE_MIB, "estimate", key
    if engine == "audio":
        return vram.AUDIO_BASE_MIB, "estimate", key
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


async def _sync_vllm() -> None:
    """Match the ledger to the vLLM containers aias labelled, which may have been
    stopped or have crashed without aias, or outlived an MCP restart."""
    rows = [r for r in await asyncio.to_thread(_vllm_containers) if r["running"]]
    live = {}
    for r in rows:
        lab = r["labels"]
        model, port = lab.get("model"), lab.get("port", "")
        if model and port.isdigit():
            live[f"vllm:{model}"] = (model, int(port), lab)
    for key, inst in list(INSTANCES.items()):
        if inst.engine == "vllm" and key not in live and inst.settled():
            del INSTANCES[key]
    for key, (model, port, lab) in live.items():
        inst = INSTANCES.get(key)
        if inst is not None:
            if not inst.starting:
                inst.port = port
            continue
        # The labels carry what the ledger booked at start, declaration included.
        record_key = lab.get("record_key", f"vllm:{model}")
        booked, source = int(lab.get("vram_mib", "0") or 0), lab.get("vram_source", "estimate")
        measured = STORE.measured(record_key)
        if measured and source != "declared":
            booked, source = measured, "measured"
        if booked <= 0:
            continue
        INSTANCES[key] = Instance("vllm", model, booked, source, record_key, port=port)


async def _sync_single(engine: Engine, running: dict[str, Any]) -> None:
    """Match the ledger to the nemo or audio container, which may have been
    started, stopped or have crashed without aias (or before an MCP restart)."""
    inst = next((i for i in INSTANCES.values() if i.engine == engine), None)
    if engine not in running:
        if inst is not None and inst.settled():
            del INSTANCES[inst.key]
        return
    if inst is not None and inst.starting:
        return
    model = await asyncio.to_thread(_single_model, engine)
    if not model:
        return
    need, source, key = await _simple_need(engine, model, None)
    if inst is not None:
        if inst.model == model:
            if inst.vram_source == "estimate" and (inst.vram_mib, inst.record_key) != (need, key):
                # An estimate made on stale inputs must not stick.
                inst.vram_mib, inst.vram_source, inst.record_key = need, source, key
            return
        # The container runs something else than the ledger says: rebuild from it.
        del INSTANCES[inst.key]
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


LEGACY_VLLM = "aias-vllm-1"  # the compose vllm service container of aias before 0.2


def _is_legacy_vllm() -> bool:
    """aias-vllm-1 exists and is not one of ours (no aias labels)."""
    out = subprocess.run(["docker", "inspect", "-f", "{{json .Config.Labels}}", LEGACY_VLLM],
                         capture_output=True, text=True)
    if out.returncode != 0:
        return False
    return f"{LABEL}.engine" not in (json.loads(out.stdout or "{}") or {})


async def _reconcile_once() -> None:
    # The card first: rebuilding a vLLM entry after a restart needs its size.
    card = await asyncio.to_thread(vram.smi)
    if card:
        CARD["smi"] = card
    running = await asyncio.to_thread(_running)
    if "vllm" in running and await asyncio.to_thread(_is_legacy_vllm):
        # Started from the compose file by hand or by an old aias: it holds port
        # 8000 and GPU memory outside the ledger.
        log.warning("stopping %s, the old single vLLM container", LEGACY_VLLM)
        await asyncio.to_thread(_stop, "vllm")
    await _sync_vllm()
    for engine in SINGLE:
        await _sync_single(engine, running)
    await _sync_ollama(running)
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
    return [
        j for j in JOBS.values()
        if j.state == "running" and (j.instance == key or key in j.also) and j is not exclude
    ]


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


def _diarize_capacity(model: str, freed_mib: int = 0) -> float:
    """Longest file (minutes, rounded down to 0.1) whose diarization is admitted now
    without evicting anything, by the same check the reservation makes. freed_mib
    counts memory as already released (the bursts of jobs still running)."""
    card = CARD["smi"] or {}
    free = card.get("free")
    return vram.diarize_capacity_min(
        STORE.burst_rate(model),
        vram.budget(card.get("total", 0)),
        _reserved() - freed_mib,
        None if free is None else free + freed_mib,
        CARD["pressure"],
        MAX_AUDIO_SECS / 60,
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
    elif inst.engine == "vllm":
        await asyncio.to_thread(_stop_vllm_models, {inst.model})
    else:
        await asyncio.to_thread(_stop, inst.engine)
    INSTANCES.pop(key, None)
    STORE.pin(key, False)
    _touch()


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
    also: list[str] = field(default_factory=list)  # other instances it uses (nemo for a diarized transcript)
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
            if BURSTS.pop(job.id, None) is not None:
                _touch()
            if job.kind != "up":
                for key in (job.instance, *job.also):
                    if key in INSTANCES:
                        INSTANCES[key].last_used = time.time()

    JOBS[job.id] = job
    job.task = asyncio.create_task(runner())
    return job.view()


def _replacements(engine: Engine, key: str) -> list[str]:
    """nemo and audio hold one model each: the other one, if any, has to go.
    Refused if it is pinned, starting or running a job. vLLM models sit side by side."""
    if engine not in SINGLE:
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
    start it. PLACEMENT is held only to decide: what to replace or evict comes from
    the ledger as it is now, and the model is booked (a placeholder, not ready)
    before the lock is released, so the minutes a start takes do not hold up other
    admissions. A failed or cancelled start takes its booking back at once. The footprint
    is measured only when nothing else claimed or released memory meanwhile."""
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
        for key in plan.replace:
            # Tell whoever started the model that was just swapped out.
            for other in JOBS.values():
                if other.kind == "up" and other.instance == key and other.state == "done":
                    other.result = {**(other.result or {}), "replaced_by": inst.model, "replaced_by_job": job.id}
        if evicted:
            await _wait_card_free(inst.vram_mib + vram.SAFETY_MIB)
        if inst.engine == "vllm":
            inst.port = await asyncio.to_thread(_free_vllm_port)
        quiet = not evicted and not any(
            j.state == "running" and j is not job and j.kind != "pull" for j in JOBS.values()
        )
        before = await asyncio.to_thread(vram.smi)
        inst.starting = True
        inst.since = time.time()
        INSTANCES[inst.key] = inst
        _touch()
        epoch = _EPOCH[0]

    try:
        detail = await start(job, inst)
    except BaseException:
        # Take the booking back at once, with no await: waiting for PLACEMENT here
        # could itself be cancelled (model_down cancels, then waits) and leave the
        # model booked and "starting" for good. Dropping a booking only frees
        # ledger room, so a plan being made under the lock meanwhile stays safe.
        if INSTANCES.get(inst.key) is inst:
            del INSTANCES[inst.key]
            _touch()
        raise
    inst.starting = False
    inst.last_used = inst.since = time.time()
    others_busy = any(j.state == "running" and j is not job and j.kind != "pull" for j in JOBS.values())
    if quiet and not others_busy and _EPOCH[0] == epoch and inst.engine != "ollama" and before:
        await asyncio.sleep(3)
        after = await asyncio.to_thread(vram.smi)
        if _EPOCH[0] == epoch and after and after["used"] > before["used"]:
            measured = after["used"] - before["used"]
            STORE.record(inst.record_key, measured)
            if inst.vram_source != "declared":  # a declaration stays in charge
                inst.vram_mib, inst.vram_source = measured, "measured"
    job.result = {"evicted": evicted, "vram_mib": inst.vram_mib, "vram_source": inst.vram_source}
    if inst.engine == "vllm":
        job.result["port"] = inst.port
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
    if job.engine == "audio":
        return await _pull_audio(job)
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


async def _pull_audio(job: Job) -> str:
    """The AudioGen repo (two .bin state dicts, 3.7 GB) and the t5-large text
    encoder it loads at startup (2.8 GB), so the engine starts without a download."""
    size = 0
    repos = [job.model, *AUDIO_REPOS[job.model]]
    for n, repo in enumerate(repos, 1):
        job.detail = f"downloading {repo} ({n} of {len(repos)})"
        # Only what transformers loads, not the TensorFlow, Flax and ONNX copies.
        patterns: dict[str, Any] = {} if repo == job.model else {"allow_patterns": AUDIO_HELPER_FILES}
        path = await asyncio.to_thread(snapshot_download, repo, **patterns)
        size += sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())
    return f"pulled {job.model} and {', '.join(repos[1:])} ({size / 1e9:.2f} GB)"


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


def _free_vllm_port(skip: set[int] | frozenset[int] = frozenset()) -> int:
    """Lowest port no running vLLM container or booked model holds (and not in
    skip). A stopped container still on it is removed first (its name is taken by
    the port)."""
    rows = _vllm_containers()
    busy = {int(r["labels"].get("port", -1)) for r in rows if r["running"]}
    busy |= {i.port for i in INSTANCES.values() if i.engine == "vllm" and i.port}
    for port in VLLM_PORTS:
        if port not in busy and port not in skip:
            _remove_vllm([r["name"] for r in rows if r["name"] == _vllm_name(port)])
            return port
    raise RuntimeError(f"all {len(VLLM_PORTS)} vLLM ports ({VLLM_PORTS[0]}-{VLLM_PORTS[-1]}) are in use")


def _start_vllm(cfg: dict[str, Any]) -> Any:
    async def start(job: Job, inst: Instance) -> str:
        total = _total() or 1
        # vLLM refuses to start unless util x card is free now (the card minus what
        # everything else uses); with --kv-cache-memory that check is all util still
        # does, so ask for exactly the budgeted need.
        util = min(0.95, math.ceil(inst.vram_mib / total * 1000) / 1000)
        tried: set[int] = set()
        while True:
            port = inst.port  # booked by _place, or the next one after a clash
            labels = {
                "engine": "vllm", "model": inst.model, "port": str(port), "vram_mib": str(inst.vram_mib),
                "vram_source": inst.vram_source, "record_key": inst.record_key,
            }
            args = [
                "run", "-d", "--no-deps", "--name", _vllm_name(port), "-p", f"127.0.0.1:{port}:8000",
                *[a for k, v in labels.items() for a in ("--label", f"{LABEL}.{k}={v}")],
                "vllm",
                f"--model={inst.model}", f"--max-model-len={cfg['max_len']}",
                f"--gpu-memory-utilization={util}", f"--kv-cache-memory={cfg['kv']}",
            ]
            try:
                await asyncio.to_thread(_run, *args)
                break
            except RuntimeError as exc:
                text = str(exc).lower()
                if "port is already allocated" not in text and "address already in use" not in text:
                    raise
                # Something outside aias holds the port: drop the half-made
                # container and take the next free one.
                await asyncio.to_thread(_remove_vllm, [_vllm_name(port)])
                tried.add(port)
                async with PLACEMENT:
                    inst.port = await asyncio.to_thread(_free_vllm_port, tried)
        job.detail = f"starting on port {port} (weights, compile, warmup)"

        def alive() -> bool:
            return any(r["name"] == _vllm_name(port) and r["running"] for r in _vllm_containers())

        if await _wait_http(f"{inst.url}/v1/models", VLLM_STARTUP_SECS, alive):
            return f"{inst.model} is up at {inst.base_url}"
        tail = subprocess.run(
            ["docker", "logs", "--tail", "40", _vllm_name(port)], capture_output=True, text=True
        )
        await asyncio.to_thread(_remove_vllm, [_vllm_name(port)])
        raise RuntimeError(f"vLLM did not become ready. Last log lines:\n{(tail.stdout + tail.stderr)[-4000:]}")

    return start


async def _start_single(job: Job, inst: Instance) -> str:
    """Start nemo or audio: build its image (a cache hit after the first time), then
    recreate its one container with the model and wait for it to load."""
    engine = inst.engine
    job.detail = f"building the {engine} image (about 5 minutes the first time, seconds after)"
    _BUILD_CANCELLED[engine].clear()
    await asyncio.to_thread(_build, engine)
    await asyncio.to_thread(_run, "up", "-d", "--force-recreate", engine, env={_model_env(engine): inst.model})
    job.detail = "loading the model onto the GPU"

    def alive() -> bool:
        return engine in _running()

    secs = NEMO_STARTUP_SECS if engine == "nemo" else AUDIO_STARTUP_SECS
    if await _wait_http(f"{INTERNAL[engine]}/health", secs, alive):
        return f"{inst.model} is up; call {'diarize' if engine == 'nemo' else 'generate_audio'}"
    tail = await asyncio.to_thread(_run, "logs", "--no-log-prefix", "--tail", "40", engine)
    with contextlib.suppress(RuntimeError):
        await asyncio.to_thread(_stop, engine)
    raise RuntimeError(f"{engine} did not become ready. Last log lines:\n{tail}")


async def _with_audio(job: Job, audio_url: str, work: Any) -> Any:
    """Fetch audio_url once as 16 kHz mono WAV (audio.py enforces every limit), then
    run work(client, wav, audio_s), which calls one engine or several."""
    try:
        async with asyncio.timeout(AUDIO_JOB_SECS):
            with tempfile.TemporaryDirectory(prefix=AUDIO_TMP_PREFIX, dir=WORK) as tmp:
                job.detail = "downloading and decoding the audio"
                wav, audio_s = await fetch_wav(audio_url, Path(tmp))
                job.detail = f"processing {audio_s / 60:.0f} min of audio"
                async with httpx.AsyncClient(timeout=httpx.Timeout(AUDIO_JOB_SECS, connect=10)) as client:
                    return await work(client, wav, audio_s)
    except AudioError as exc:
        raise RuntimeError(str(exc)) from None
    except TimeoutError:
        raise RuntimeError(f"{job.kind} did not finish within {AUDIO_JOB_SECS} s") from None
    except httpx.HTTPError as exc:
        raise RuntimeError(f"could not reach an engine: {type(exc).__name__} {exc}".rstrip()) from exc


def _reply(resp: httpx.Response, engine: str) -> dict[str, Any]:
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


async def _reserve_burst(job: Job, mib: int, evict: str, protect: set[str]) -> None:
    """Hold mib on top of the job's instances for as long as the job runs."""
    async with PLACEMENT:
        await _reconcile()
        plan = _plan(mib, protect, [])
        if not plan.can_fit or (not plan.fits and evict != "auto"):
            job.result = _refusal(plan)
            nemo = next((INSTANCES[k] for k in protect if k.startswith("nemo:") and k in INSTANCES), None)
            if nemo is not None:
                job.result["max_audio_minutes"] = _diarize_capacity(nemo.model)
            limit = f"; at most {math.floor(job.result['max_audio_minutes'])} minutes fit now" if nemo else ""
            raise RuntimeError((plan.reason or f"the {mib} MiB this job needs does not fit") + limit)
        await _carry_out(job, plan)
        BURSTS[job.id] = mib
        _touch()


async def _run_nemo(job: Job, client: httpx.AsyncClient, wav: Path, audio_s: float,
                    nemo: str, model: str, mode: str, evict: str) -> dict[str, Any]:
    """One diarization on the nemo engine: queue, reserve its burst, send the WAV."""
    job.detail = "waiting for the nemo engine"
    async with _queue(nemo, 1):
        burst = vram.nemo_burst_mib(audio_s, STORE.burst_rate(model))
        await _reserve_burst(job, burst, evict, {job.instance or "", *job.also})
        job.detail = f"diarizing {audio_s / 60:.0f} min of audio"
        try:
            resp = await client.post(
                f"{INTERNAL['nemo']}/diarize",
                params={"mode": mode},
                content=_wav_chunks(wav),
                headers={"Content-Type": "audio/wav"},
            )
        finally:
            if BURSTS.pop(job.id, None) is not None:
                _touch()
    body = _reply(resp, "nemo")
    if body.get("burst_mib") and body.get("audio_s"):
        STORE.record_burst(model, body["burst_mib"], body["audio_s"])
    body["reserved_burst_mib"] = burst
    return body


async def _run_whisper(job: Job, client: httpx.AsyncClient, wav: Path, audio_s: float,
                       vllm: str, code: str) -> tuple[dict[str, Any], float]:
    """One transcription on vLLM. Returns the reply and the seconds it took."""
    job.detail = "waiting for the vllm engine"
    inst = INSTANCES.get(vllm)
    if inst is None or not inst.port:
        raise RuntimeError(f"{job.model} is no longer up")
    url = inst.url
    async with _queue(vllm, TRANSCRIBE_CONCURRENCY):
        job.detail = f"transcribing {audio_s / 60:.0f} min of audio"
        started = time.monotonic()
        with wav.open("rb") as f:
            resp = await client.post(
                f"{url}/v1/audio/transcriptions",
                data={"model": job.model, "language": code, "response_format": "verbose_json", "temperature": "0"},
                files={"file": ("audio.wav", f, "audio/wav")},
            )
        elapsed = time.monotonic() - started
    return _reply(resp, "vllm"), elapsed


async def _diarize(job: Job, audio_url: str, mode: str, evict: str) -> str:
    async def work(client: httpx.AsyncClient, wav: Path, audio_s: float) -> dict[str, Any]:
        return await _run_nemo(job, client, wav, audio_s, job.instance or "nemo", job.model, mode, evict)

    body = await _with_audio(job, audio_url, work)
    job.result = body
    speakers = ", ".join(f"{s['speaker']} {s['seconds']:.0f} s" for s in body["speakers"]) or "no speech"
    return f"{len(body['speakers'])} speakers in {body['audio_s']:.0f} s of audio: {speakers}"


def _whisper_language(language: str) -> str:
    """zh-TW, zh-Hant, zh_CN and the like to the bare ISO 639-1 code Whisper takes."""
    return language.replace("_", "-").split("-")[0].strip().lower()


async def _transcribe(
    job: Job, audio_url: str, language: str, traditional: bool,
    diarize: dict[str, str] | None = None, evict: str = "never",
) -> str:
    """Whisper on vLLM; with diarize ({"instance", "model"} of nemo), nemo runs on
    the same decoded WAV at the same time and the segments get speakers."""
    code = _whisper_language(language)
    timing: dict[str, float] = {}

    async def work(client: httpx.AsyncClient, wav: Path, audio_s: float) -> Any:
        timing["audio_s"] = audio_s
        started = time.monotonic()
        if diarize is None:
            body, timing["transcribe_s"] = await _run_whisper(job, client, wav, audio_s, job.instance or "vllm", code)
            return body, None
        job.detail = f"transcribing and diarizing {audio_s / 60:.0f} min of audio"
        try:
            async with asyncio.TaskGroup() as group:
                whisper = group.create_task(_run_whisper(job, client, wav, audio_s, job.instance or "vllm", code))
                nemo = group.create_task(
                    _run_nemo(job, client, wav, audio_s, diarize["instance"], diarize["model"], "offline", evict)
                )
        except ExceptionGroup as failed:  # one engine failed; the other was cancelled
            raise failed.exceptions[0] from None
        body, timing["transcribe_s"] = whisper.result()
        timing["diarize_s"] = nemo.result().get("elapsed_s", 0)
        timing["elapsed_s"] = time.monotonic() - started
        return body, nemo.result()

    body, diarized = await _with_audio(job, audio_url, work)
    # OpenCC only makes sense for Chinese; it would leave other text alone, but skip it.
    traditional = traditional and code == "zh"
    convert = _S2TW.convert if traditional else (lambda text: text)
    segments = [
        {"start": round(seg["start"], 2), "end": round(seg["end"], 2), "text": convert(seg["text"].strip())}
        for seg in body.get("segments") or []
    ]
    result = {
        "model": job.model,
        "language": code,
        "traditional": traditional,
        "audio_s": round(timing["audio_s"], 1),
        "elapsed_s": round(timing.get("elapsed_s", timing["transcribe_s"]), 2),
        "text": convert(body.get("text", "").strip()),
        "segments": segments,
    }
    summary = f"{len(segments)} segments from {timing['audio_s']:.0f} s of audio in {result['elapsed_s']:.0f} s"
    if diarized is not None:
        labelled = align.label(segments, diarized["rttm"])
        result.update({
            "diarization_model": diarized["model"],
            "transcribe_s": round(timing["transcribe_s"], 2),
            "diarize_s": timing["diarize_s"],
            "gpu_peak_mib": diarized.get("gpu_peak_mib"),
            "reserved_burst_mib": diarized.get("reserved_burst_mib"),
            "segments": labelled,
            "speakers": align.speakers(diarized["speakers"], labelled),
            "turns": align.turns(labelled, code),
            "rttm": diarized["rttm"],
        })
        uncertain = sum(s["speaker_uncertain"] for s in labelled)
        summary += f", {len(diarized['speakers'])} speakers, {uncertain} segments uncertain"
    job.result = result
    return summary


def _prune_out() -> None:
    """Drop generated files past OUT_KEEP_SECS, then the oldest past OUT_MAX_BYTES.
    A .tmp left by a crash mid-write goes with the expired ones."""
    now = time.time()
    files = []
    for f in [*OUT.glob("*.wav"), *OUT.glob(".*.tmp")]:
        with contextlib.suppress(OSError):
            st = f.stat()
            if now - st.st_mtime > OUT_KEEP_SECS:
                f.unlink()
            elif f.suffix == ".wav":
                files.append((st.st_mtime, st.st_size, f))
    total = sum(size for _, size, _ in files)
    for _, size, f in sorted(files):
        if total <= OUT_MAX_BYTES:
            break
        with contextlib.suppress(OSError):
            f.unlink()
        total -= size


def _save_out(data: bytes) -> tuple[str, float]:
    """Write a generated WAV under a random name; returns the name and when it expires."""
    OUT.mkdir(parents=True, exist_ok=True)
    _prune_out()
    name = f"{uuid.uuid4().hex}.wav"
    tmp = OUT / f".{name}.tmp"
    tmp.write_bytes(data)
    os.replace(tmp, OUT / name)
    return name, time.time() + OUT_KEEP_SECS


async def _generate(job: Job, prompt: str, duration_s: float, seed: int | None, cfg_coef: float, evict: str) -> str:
    """One clip on the audio engine: queue, reserve its burst, generate, keep the WAV."""
    key = job.instance or "audio"
    job.detail = "waiting for the audio engine"
    async with _queue(key, 1):
        burst = vram.audio_burst_mib(duration_s, STORE.audio_burst_factor(job.model))
        await _reserve_burst(job, burst, evict, {key})
        job.detail = f"generating {duration_s:g} s of audio"
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(GENERATE_SECS, connect=10)) as client:
                resp = await client.post(
                    f"{INTERNAL['audio']}/generate",
                    json={"prompt": prompt, "duration_s": duration_s, "seed": seed, "cfg_coef": cfg_coef},
                )
        except httpx.HTTPError as exc:
            raise RuntimeError(f"could not reach the audio engine: {type(exc).__name__} {exc}".rstrip()) from exc
        finally:
            if BURSTS.pop(job.id, None) is not None:
                _touch()
    if resp.status_code != 200 or not resp.content.startswith(b"RIFF"):
        _reply(resp, "audio")
        raise RuntimeError("the audio engine did not answer with a WAV file")
    try:
        stats = json.loads(resp.headers.get("x-aias-result") or "{}")
    except ValueError:
        stats = {}
    if stats.get("burst_mib") and stats.get("audio_s"):
        STORE.record_audio_burst(job.model, stats["burst_mib"], stats["audio_s"])
    name, expires = await asyncio.to_thread(_save_out, resp.content)
    job.result = {
        "url": f"http://127.0.0.1:{PORT}/files/{name}",
        "file": name,
        "bytes": len(resp.content),
        "expires": _iso(expires),
        "prompt": prompt,
        **stats,
        "reserved_burst_mib": burst,
    }
    return f"{stats.get('audio_s', duration_s):g} s of audio in {stats.get('elapsed_s', 0):.0f} s: {job.result['url']}"


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
            ready = not inst.starting and bool(inst.port) and await _wait_http(f"{inst.url}/v1/models", 0.1)
        elif inst.engine in SINGLE:
            ready = not inst.starting and await _wait_http(f"{INTERNAL[inst.engine]}/health", 0.1)
        else:
            ready = not inst.starting
        entry = {
            "engine": inst.engine,
            "model": inst.model,
            "models": [inst.model],
            "base_url": inst.base_url,
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
        if inst.engine == "nemo":
            # Longest file a diarize (alone or with transcribe) is admitted for now.
            entry["max_audio_minutes"] = _diarize_capacity(inst.model)
            in_flight = sum(BURSTS.get(j.id, 0) for j in _jobs_of(inst.key))
            if in_flight:
                # nemo runs one file at a time; a new one waits, then gets this.
                entry["max_audio_minutes_next"] = _diarize_capacity(inst.model, freed_mib=in_flight)
        if inst.engine == "audio":
            entry["max_duration_s"] = AUDIO_MAX_S
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
        elif "used_by" in m:  # a text encoder, counted in the model that loads it
            m["vram_mib"], m["vram_source"] = None, "part_of_model"
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
    """Download a model. ollama takes library names like qwen3:8b; vllm, nemo and audio take
    Hugging Face repo ids like Qwen/Qwen3-0.6B, nvidia/Nemotron-3-Diarization or
    facebook/audiogen-medium (which also fetches its t5-large text encoder). Returns a
    job; poll job_status."""
    _ensure_reconciler()
    _check_repo(engine, model)
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
    Each vllm model gets its own container and port (8000 first, up to 8009; status
    gives each its base_url); nemo and audio hold one model each, so asking for
    another replaces it. max_model_len and
    gpu_memory_utilization apply to vllm only; left out, they are 8192 and 0.8, or 448
    and 0.4 for a Whisper model. vLLM startup takes 40 s to several minutes; the first
    nemo or audio start builds its image and takes about 5 minutes. Returns a job; when
    it is done, status shows the base_url (nemo and audio have none: use diarize or
    generate_audio)."""
    _ensure_reconciler()
    await _reconcile()
    _check_repo(engine, model)
    if engine == "audio":
        await asyncio.to_thread(_check_audio_gpu)
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
    start = _start_ollama if engine == "ollama" else _start_single if engine in SINGLE else _start_vllm(cfg)

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
            if job.kind == "up" and job.engine in SINGLE:
                # A build holds the compose lock for minutes; end it so _stop gets the lock.
                _cancel_build(job.engine)
            job.task.cancel()
            tasks.append(job.task)
    return tasks


def _stop_vllm_models(names: set[str]) -> list[str]:
    """Remove every vLLM container (running or not) labelled with one of names.
    Under the compose lock, so a start still finishing in a cancelled job's
    thread completes first and is then seen."""
    with _COMPOSE_LOCK:
        gone = [r for r in _vllm_containers() if r["labels"].get("model") in names]
        _remove_vllm([r["name"] for r in gone])
    return sorted({r["labels"]["model"] for r in gone})


def _stop_if_running(engine: Engine, names: set[str]) -> bool:
    """Stop the vllm containers, or the nemo or audio container, that run one of names."""
    if engine == "vllm":
        return bool(_stop_vllm_models(names))
    with _COMPOSE_LOCK:
        if engine not in _running():
            return False
        if _single_model(engine) not in names:
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
    and cancel every pull, up, diarize, transcribe or generate_audio job, as before."""
    _ensure_reconciler()
    if model is None:
        tasks = _cancel([j for j in JOBS.values() if j.state == "running"])
        _cancel_build()
        if tasks:
            await asyncio.wait(tasks, timeout=30)
        # No reconcile may read the containers mid-stop and put a model back.
        async with _RECONCILE_LOCK:
            await asyncio.to_thread(_stop, *ENGINES)
            await asyncio.to_thread(lambda: _remove_vllm([r["name"] for r in _vllm_containers()]))
            if await asyncio.to_thread(_is_legacy_vllm):
                await asyncio.to_thread(_remove_vllm, [LEGACY_VLLM])
            for inst in list(INSTANCES.values()):
                STORE.pin(inst.key, False)
                if inst.engine == "ollama":
                    STORE.manage(inst.model, False)
            INSTANCES.clear()
            BURSTS.clear()
            _touch()
            CARD["smi"] = await asyncio.to_thread(vram.smi)
        return {"stopped": list(ENGINES), "gpu": _gpu_view()}

    names = {model, _ollama_name(model)}
    jobs = [
        j for j in JOBS.values()
        if j.state == "running" and j.kind != "pull"
        and (j.model in names or any(k.split(":", 1)[1] in names for k in j.also))
    ]
    tasks = _cancel(jobs)
    if tasks:
        await asyncio.wait(tasks, timeout=30)
    stopped: list[str] = []
    async with _RECONCILE_LOCK:
        # Go by what the engines run, not the ledger: a model still starting (or
        # one whose job was just cancelled) may have no ledger entry.
        for engine in ("vllm", *SINGLE):
            if await asyncio.to_thread(_stop_if_running, engine, names):
                stopped.append(f"{engine}:{model}")
        loaded = {(m.get("name") or m.get("model")) for m in await _ollama_ps()}
        for name in names & loaded:
            await _unload_ollama(name)
            STORE.manage(name, False)
            stopped.append(f"ollama:{name}")
        _touch()
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
async def transcribe(
    audio_url: str,
    language: str = "zh",
    traditional: bool = True,
    diarize: bool = False,
    evict: Evict = "never",
    model: str | None = None,
) -> dict[str, Any]:
    """Speech to text with timestamps, with a Whisper model on vllm (model_up engine=vllm
    model=openai/whisper-large-v3 first). model picks one of several Whisper models
    that are up; left out, the one used most recently; the result names it. audio_url is an http(s) link the server
    downloads; any format ffmpeg reads, up to 2 GB and 2 hours. language is an ISO 639-1
    code; a tag like zh-TW or zh_CN is cut to zh. traditional converts simplified Chinese
    characters to traditional (Taiwan), because Whisper drifts to simplified; it applies
    only to zh. diarize=true also labels speakers: the same audio goes to nemo at the
    same time (model_up engine=nemo first; neither model is started for you), each
    segment gets speaker, speaker_confidence and speaker_uncertain, and the result adds
    speakers, turns (adjacent segments of one speaker merged) and the rttm; evict
    applies to the diarization's GPU reservation as in diarize. Several transcriptions
    run at once. Returns a job; when it is done, job_status carries result with the full
    text and segments (start, end, text) in seconds."""
    _ensure_reconciler()
    await _reconcile()
    whispers = [i for i in INSTANCES.values() if i.engine == "vllm" and _is_whisper(i.model) and not i.starting]
    if model is not None:
        if not _is_whisper(model):
            raise ToolError(f"{model} is not a Whisper model")
        whispers = [i for i in whispers if i.model == model]
        if not whispers:
            raise ToolError(f"{model} is not up: run model_up engine=vllm model={model} first")
    inst = max(whispers, key=lambda i: i.last_used, default=None)
    nemo = _up_instance("nemo", lambda m: True) if diarize else None
    missing = []
    if inst is None:
        missing.append("model_up engine=vllm model=openai/whisper-large-v3")
    if diarize and nemo is None:
        missing.append("model_up engine=nemo model=nvidia/Nemotron-3-Diarization")
    if missing:
        what = "Whisper and nemo are" if len(missing) == 2 else "Whisper is" if inst is None else "nemo is"
        need = " for diarize=true" if diarize else ""
        raise ToolError(f"{what} not up{need}: run {' and '.join(missing)} first")
    for up in (inst, nemo):
        if up is not None and up.key in EVICTING:
            raise ToolError(f"{up.model} is being stopped to make room for another model")
    if not inst.port or not await _wait_http(f"{inst.url}/v1/models", 0.1):
        raise ToolError(f"{inst.model} is still starting; wait for its model_up job to finish")
    inst.last_used = time.time()
    job = Job("transcribe", "vllm", inst.model, instance=inst.key)
    target = None
    if nemo is not None:
        nemo.last_used = time.time()
        job.also = [nemo.key]
        target = {"instance": nemo.key, "model": nemo.model}
    return _start_job(job, lambda j: _transcribe(j, audio_url, language, traditional, target, evict))


@mcp.tool()
async def generate_audio(
    prompt: str,
    duration_s: float = 5.0,
    seed: StrictInt | None = None,
    cfg_coef: float = 3.0,
    evict: Evict = "never",
) -> dict[str, Any]:
    """Generate a sound effect from an English text prompt with the audio engine
    (model_up engine=audio model=facebook/audiogen-medium first), e.g. "dog barking in
    the distance, light rain". duration_s is 0.5 to 30; AudioGen is trained on 10 s
    clips and extends longer ones, which takes longer and drifts more. seed makes a run
    repeatable (the result gives the one used); cfg_coef (0 to 10, default 3) is how
    closely it follows the prompt. A run needs some GPU memory on top of the model; if
    that does not fit, the job fails with a plan, and evict="auto" lets it stop least
    recently used models. One clip at a time; later ones wait. Returns a job; when it is
    done, result.url is a 16 kHz mono WAV to download (kept 24 hours), with seed,
    audio_s and elapsed_s. Output is licensed CC BY-NC 4.0 (the model's license)."""
    _ensure_reconciler()
    await _reconcile()
    prompt = prompt.strip()
    if not prompt or len(prompt) > AUDIO_MAX_PROMPT:
        raise ToolError(f"prompt must be 1 to {AUDIO_MAX_PROMPT} characters")
    if not AUDIO_MIN_S <= duration_s <= AUDIO_MAX_S:
        raise ToolError(f"duration_s must be from {AUDIO_MIN_S:g} to {AUDIO_MAX_S:g}, not {duration_s:g}")
    if not 0 <= cfg_coef <= 10:
        raise ToolError(f"cfg_coef must be from 0 to 10, not {cfg_coef:g}")
    if seed is not None and not 0 <= seed < 2**31:
        raise ToolError("seed must be from 0 to 2147483647")
    inst = _up_instance("audio", lambda m: True)
    if inst is None:
        raise ToolError(f"audio is not up: run model_up engine=audio model={next(iter(AUDIO_REPOS))} first")
    if inst.key in EVICTING:
        raise ToolError(f"{inst.model} is being stopped to make room for another model")
    job = Job("generate", "audio", inst.model, instance=inst.key)
    inst.last_used = time.time()
    return _start_job(job, lambda j: _generate(j, prompt, duration_s, seed, cfg_coef, evict))


@mcp.tool()
async def job_status(job_id: str, wait_seconds: int = 30) -> dict[str, Any]:
    """Progress of a pull, up, diarize, transcribe or generate job. Blocks up to
    wait_seconds (max 120) for it to finish. A finished diarize, transcribe or generate
    job carries its output in result; a refused one carries the plan."""
    job = JOBS.get(job_id)
    if job is None:
        raise ToolError(f"no job {job_id}; jobs do not survive a server restart")
    if job.task is not None and job.state == "running":
        await asyncio.wait({job.task}, timeout=max(0, min(wait_seconds, 120)))
    return job.view()


@mcp.tool()
async def logs(engine: Engine, lines: int = 100, model: str | None = None) -> str:
    """Recent log lines of an engine, for diagnosing a failed start. For vllm, model
    picks which model's container (default: the one on the lowest port)."""
    lines = max(1, min(lines, 1000))
    if engine != "vllm":
        return await asyncio.to_thread(_run, "logs", "--no-log-prefix", "--tail", str(lines), engine)
    rows = sorted(await asyncio.to_thread(_vllm_containers), key=lambda r: int(r["labels"].get("port", 0) or 0))
    rows = [r for r in rows if model is None or r["labels"].get("model") == model]
    if not rows:
        raise ToolError(f"no vLLM container{' for ' + model if model else ''}; status lists what is up")
    out = subprocess.run(["docker", "logs", "--tail", str(lines), rows[0]["name"]], capture_output=True, text=True)
    return out.stdout + out.stderr


@mcp.custom_route("/health", methods=["GET"])
async def health(_: Request) -> JSONResponse:
    _ensure_reconciler()
    return JSONResponse({"ok": True})


@mcp.custom_route("/files/{name}", methods=["GET"])
async def files(request: Request) -> Response:
    """A WAV generate_audio made. Same Host check as /mcp, so a web page cannot read
    it through DNS rebinding; names are random and unlisted."""
    if request.headers.get("host") not in LOCAL_HOSTS:
        return JSONResponse({"error": "forbidden host"}, status_code=421)
    name = request.path_params.get("name", "")
    path = OUT / name
    gone = JSONResponse({"error": "no such file; generated files are kept 24 hours"}, status_code=404)
    if not OUT_NAME.fullmatch(name):
        return gone
    try:
        st = path.stat()  # once: a prune may remove it at any moment
    except OSError:
        return gone
    if not stat.S_ISREG(st.st_mode) or time.time() - st.st_mtime > OUT_KEEP_SECS:
        return gone
    return FileResponse(path, media_type="audio/wav", filename=name)


if __name__ == "__main__":
    # Audio a previous run was working on when it stopped.
    for leftover in Path(WORK).glob(f"{AUDIO_TMP_PREFIX}*"):
        shutil.rmtree(leftover, ignore_errors=True)
    # Job directories are 0700 once decoded; the decoder may enter only its own.
    with contextlib.suppress(OSError):
        os.chmod(WORK, 0o711)
    with contextlib.suppress(OSError):
        _prune_out()
    mcp.run(
        "streamable-http",
        host="0.0.0.0",
        port=PORT,
        stateless_http=True,
        json_response=True,
        # Published on 127.0.0.1 only; still refuse other Host headers so a web
        # page cannot reach it through DNS rebinding.
        transport_security=TransportSecuritySettings(
            allowed_hosts=LOCAL_HOSTS,
            allowed_origins=[f"http://localhost:{PORT}", f"http://127.0.0.1:{PORT}"],
        ),
    )
