"""aias audio engine: text to sound effects and music over HTTP.

Loads one AudioCraft model onto the GPU at startup, picked by AUDIO_MODEL:
AudioGen (facebook/audiogen-medium, sound effects) or MusicGen
(facebook/musicgen-medium, music). It answers POST /generate for the MCP server
with a WAV file. It is reachable only on the compose network; the MCP tools are
its public face.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import random
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import soundfile as sf
import torch
import uvicorn
from audiocraft.models import AudioGen, MusicGen
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route


@dataclass(frozen=True)
class Spec:
    cls: type
    max_duration_s: float
    default_cfg_coef: float
    # Clips past the training length are extended with a sliding window, this
    # many seconds at a time; they get slower and less coherent.
    extend_stride_s: float


SPECS = {
    # Trained on 10 s clips.
    "facebook/audiogen-medium": Spec(AudioGen, 30.0, 3.0, 5.0),
    # Trained on 30 s clips, so up to 30 s it never extends; 18 is MusicGen's own stride.
    "facebook/musicgen-medium": Spec(MusicGen, 30.0, 3.0, 18.0),
}

MODEL = os.environ.get("AUDIO_MODEL", "facebook/audiogen-medium")
if MODEL not in SPECS:
    sys.exit(f"AUDIO_MODEL={MODEL!r} is not supported; set it to one of: {', '.join(SPECS)}")
SPEC = SPECS[MODEL]
PORT = 8200
MIN_DURATION_S = 0.5
MAX_CFG_COEF = 10.0
MAX_PROMPT_CHARS = 500
# The loudest sample ends up here (-1 dBFS), so clips come out at a usable level.
PEAK = 10 ** (-1 / 20)

model = SPEC.cls.get_pretrained(MODEL, device="cuda")
SAMPLE_RATE = int(model.sample_rate)
CHANNELS = 1  # both are mono models
# Loading leaves checkpoint buffers in PyTorch's cache; hand them back so the
# idle footprint matches the one between jobs.
torch.cuda.empty_cache()
# Generation settings live on the shared model, so one request at a time.
_LOCK = threading.Lock()

log = logging.getLogger("aias.audio")


class BadRequest(Exception):
    pass


class GenerateError(Exception):
    """A failed generation, as text only: no traceback holding GPU tensors."""


def _number(body: dict[str, Any], name: str, default: float, lo: float, hi: float) -> float:
    value = body.get(name)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not lo <= value <= hi:
        raise BadRequest(f"{name} must be a number from {lo:g} to {hi:g}")
    return float(value)


def _run(prompt: str, duration: float, seed: int, cfg_coef: float) -> np.ndarray:
    model.set_generation_params(
        duration=duration, use_sampling=True, top_k=250, cfg_coef=cfg_coef, extend_stride=SPEC.extend_stride_s
    )
    torch.manual_seed(seed)
    wav = model.generate([prompt], progress=False)[0]
    return wav.float().mean(dim=0).cpu().numpy()


def _generate(prompt: str, duration: float, seed: int, cfg_coef: float) -> tuple[bytes, dict[str, Any]]:
    with _LOCK:
        torch.cuda.reset_peak_memory_stats()
        base_reserved = torch.cuda.memory_reserved()
        started = time.monotonic()
        failure = None
        try:
            with torch.inference_mode():
                samples = _run(prompt, duration, seed, cfg_coef)
            elapsed = time.monotonic() - started
            peak_mib = torch.cuda.max_memory_allocated() / 2**20
            # What this clip added on top of the idle engine, as nvidia-smi sees it.
            burst_mib = (torch.cuda.max_memory_reserved() - base_reserved) / 2**20
        except torch.cuda.OutOfMemoryError:
            failure = GenerateError("out of GPU memory; try a shorter duration")
        except Exception as exc:
            log.exception("generate failed")
            failure = GenerateError(
                f"generation failed: {type(exc).__name__}: {str(exc)[:300]}. "
                "If it repeats, restart the engine with model_up engine=audio."
            )
        # Past the except blocks the traceback, and the tensors its frames hold,
        # are gone: hand the peak back, so the engine returns to the footprint the
        # MCP server budgets it at between jobs.
        torch.cuda.empty_cache()
        if failure is not None:
            raise failure

    loudest = float(np.abs(samples).max()) if samples.size else 0.0
    if loudest > 0:
        samples = samples * (PEAK / loudest)
    out = io.BytesIO()
    sf.write(out, samples, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return out.getvalue(), {
        "model": MODEL,
        "sample_rate": SAMPLE_RATE,
        "channels": CHANNELS,
        "audio_s": round(samples.shape[0] / SAMPLE_RATE, 2),
        "seed": seed,
        "cfg_coef": cfg_coef,
        "elapsed_s": round(elapsed, 2),
        "gpu_peak_mib": round(peak_mib),
        "burst_mib": round(burst_mib),
    }


async def generate(request: Request) -> Response:
    """Body: JSON {prompt, duration_s, seed, cfg_coef}; a null or missing cfg_coef
    is the model's default. Answers
    audio/wav, with the run's numbers in the X-Aias-Result header (JSON)."""
    try:
        try:
            body = await request.json()
        except ValueError:
            raise BadRequest("body must be JSON") from None
        if not isinstance(body, dict):
            raise BadRequest("body must be a JSON object")
        prompt = body.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise BadRequest("prompt must be a non-empty string")
        if len(prompt) > MAX_PROMPT_CHARS:
            raise BadRequest(f"prompt is longer than {MAX_PROMPT_CHARS} characters")
        duration = _number(body, "duration_s", 5.0, MIN_DURATION_S, SPEC.max_duration_s)
        cfg_coef = _number(body, "cfg_coef", SPEC.default_cfg_coef, 0.0, MAX_CFG_COEF)
        seed = body.get("seed")
        if seed is None:
            seed = random.randrange(2**31)
        elif isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**31:
            raise BadRequest("seed must be a whole number from 0 to 2147483647")
        data, stats = await asyncio.to_thread(_generate, prompt.strip(), duration, seed, cfg_coef)
    except BadRequest as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except GenerateError as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)
    return Response(data, media_type="audio/wav", headers={"X-Aias-Result": json.dumps(stats)})


async def health(_: Request) -> JSONResponse:
    return JSONResponse({
        "ok": True,
        "model": MODEL,
        "sample_rate": SAMPLE_RATE,
        "channels": CHANNELS,
        "max_duration_s": SPEC.max_duration_s,
        "default_cfg_coef": SPEC.default_cfg_coef,
    })


app = Starlette(routes=[Route("/generate", generate, methods=["POST"]), Route("/health", health)])

if __name__ == "__main__":
    # One short clip before /health answers: the first generation leaves about
    # 120 MiB of CUDA workspace that empty_cache does not return, and the MCP
    # server measures the footprint as soon as the engine is ready.
    try:
        _generate("warmup", 1.0, 0, SPEC.default_cfg_coef)
    except GenerateError as exc:
        sys.exit(f"warmup clip failed, so {MODEL} is not ready: {exc}")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
