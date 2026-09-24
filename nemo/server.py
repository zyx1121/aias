"""aias nemo engine: speaker diarization over HTTP.

Loads one NeMo Sortformer model (nvidia/Nemotron-3-Diarization by default)
onto the GPU at startup, then answers POST /diarize for the MCP server, which
downloads and decodes the audio first. It is reachable only on the compose
network; the MCP tools are its public face.
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import soundfile as sf
import torch
import uvicorn
from nemo.collections.asr.models import SortformerEncLabelModel
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

MODEL = os.environ.get("NEMO_MODEL", "nvidia/Nemotron-3-Diarization")
PORT = 8100
# The MCP server sends at most 2 hours of 16 kHz mono s16 WAV (230.4 MB).
MAX_BODY_BYTES = 240_000_000
# Streaming settings from the model card, in 80 ms frames. Latency is
# (chunk_len + chunk_right_context) frames: 30.4 s, 1.04 s, 0.64 s, 0.32 s.
MODES: dict[str, dict[str, int]] = {
    "offline": dict(chunk_len=340, chunk_right_context=40, fifo_len=40, spkcache_len=264, spkcache_update_period=300),
    "low": dict(chunk_len=9, chunk_right_context=4, fifo_len=264, spkcache_len=264, spkcache_update_period=222),
    "verylow": dict(chunk_len=6, chunk_right_context=2, fifo_len=264, spkcache_len=264, spkcache_update_period=222),
    "ultralow": dict(chunk_len=3, chunk_right_context=1, fifo_len=264, spkcache_len=264, spkcache_update_period=222),
}

model = SortformerEncLabelModel.from_pretrained(MODEL).eval().cuda()
# The streaming settings live on the shared model, so one request at a time.
_LOCK = threading.Lock()


log = logging.getLogger("aias.nemo")


class BadRequest(Exception):
    pass


def _diarize(wav: Path, mode: str) -> dict[str, Any]:
    try:
        duration = sf.info(str(wav)).duration
    except RuntimeError as exc:
        raise BadRequest(f"body is not a readable WAV file: {exc}") from exc

    with _LOCK:
        for key, value in MODES[mode].items():
            setattr(model.sortformer_modules, key, value)
        model._check_streaming_parameters()
        torch.cuda.reset_peak_memory_stats()
        base_reserved = torch.cuda.memory_reserved()
        started = time.monotonic()
        try:
            with torch.inference_mode():
                segments = model.diarize(audio=[str(wav)], batch_size=1)[0]
            elapsed = time.monotonic() - started
            peak_mib = torch.cuda.max_memory_allocated() / 2**20
            # What this file added on top of the idle engine, as nvidia-smi sees it.
            burst_mib = (torch.cuda.max_memory_reserved() - base_reserved) / 2**20
        finally:
            # Hand the per-file peak back, so the engine returns to the base
            # footprint the MCP server budgets it at between jobs.
            torch.cuda.empty_cache()

    lines = []
    seconds: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for seg in segments:
        start, end, speaker = seg.split() if isinstance(seg, str) else seg
        start, end = float(start), float(end)
        seconds[speaker] += end - start
        counts[speaker] += 1
        lines.append(f"SPEAKER audio 1 {start:.3f} {end - start:.3f} <NA> <NA> {speaker} <NA> <NA>")
    return {
        "model": MODEL,
        "mode": mode,
        "audio_s": round(duration, 1),
        "elapsed_s": round(elapsed, 2),
        "gpu_peak_mib": round(peak_mib),
        "burst_mib": round(burst_mib),
        "speakers": [
            {"speaker": s, "seconds": round(seconds[s], 1), "segments": counts[s]} for s in sorted(seconds)
        ],
        "rttm": "\n".join(lines) + ("\n" if lines else ""),
    }


async def diarize(request: Request) -> JSONResponse:
    """Body: 16 kHz mono WAV, already fetched and bounded by the MCP server
    (mcp/audio.py). Query: mode."""
    mode = request.query_params.get("mode", "offline")
    try:
        if mode not in MODES:
            raise BadRequest(f"mode must be one of {', '.join(MODES)}")
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "audio.wav"
            size = 0
            with wav.open("wb") as f:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > MAX_BODY_BYTES:
                        raise BadRequest("body is larger than 2 hours of 16 kHz mono WAV")
                    f.write(chunk)
            if size == 0:
                raise BadRequest("body is empty; send the audio as 16 kHz mono WAV")
            return JSONResponse(await asyncio.to_thread(_diarize, wav, mode))
    except BadRequest as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return JSONResponse({"error": "out of GPU memory; try a shorter file or a streaming mode"}, status_code=500)
    except Exception as exc:
        log.exception("diarize failed")
        torch.cuda.empty_cache()
        return JSONResponse(
            {
                "error": f"diarization failed: {type(exc).__name__}: {str(exc)[:300]}. "
                "If it repeats, restart the engine with model_up engine=nemo."
            },
            status_code=500,
        )


async def health(_: Request) -> JSONResponse:
    return JSONResponse({"ok": True, "model": MODEL})


app = Starlette(routes=[Route("/diarize", diarize, methods=["POST"]), Route("/health", health)])

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
