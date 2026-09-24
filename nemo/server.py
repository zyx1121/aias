"""aias nemo engine: speaker diarization over HTTP.

Loads one NeMo Sortformer model (nvidia/Nemotron-3-Diarization by default)
onto the GPU at startup, then answers POST /diarize for the MCP server. It is
reachable only on the compose network; the MCP tools are its public face.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import tempfile
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import httpx
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
MAX_BYTES = 2 * 1024**3
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


class BadRequest(Exception):
    pass


async def _download(url: str, dest: Path) -> None:
    if not url.startswith(("https://", "http://")):
        raise BadRequest("audio_url must be an http or https URL")
    size = 0
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=httpx.Timeout(120, connect=30)) as client:
            async with client.stream("GET", url) as resp:
                if resp.status_code != 200:
                    raise BadRequest(f"audio_url answered HTTP {resp.status_code}")
                with dest.open("wb") as f:
                    async for chunk in resp.aiter_bytes():
                        size += len(chunk)
                        if size > MAX_BYTES:
                            raise BadRequest("audio is larger than 2 GB")
                        f.write(chunk)
    except httpx.HTTPError as exc:
        # Timeouts carry no message, so name the exception type too.
        raise BadRequest(f"could not download audio_url: {type(exc).__name__} {exc}".rstrip()) from exc


def _diarize(src: Path, mode: str) -> dict[str, Any]:
    wav = src.with_suffix(".16k.wav")
    proc = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(src), "-ac", "1", "-ar", "16000", str(wav)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise BadRequest(f"ffmpeg could not decode the audio: {proc.stderr.strip()[-500:]}")
    duration = sf.info(str(wav)).duration

    with _LOCK:
        for key, value in MODES[mode].items():
            setattr(model.sortformer_modules, key, value)
        model._check_streaming_parameters()
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        with torch.inference_mode():
            segments = model.diarize(audio=[str(wav)], batch_size=1)[0]
        elapsed = time.monotonic() - started
        peak_mib = torch.cuda.max_memory_allocated() / 2**20

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
        "speakers": [
            {"speaker": s, "seconds": round(seconds[s], 1), "segments": counts[s]} for s in sorted(seconds)
        ],
        "rttm": "\n".join(lines) + ("\n" if lines else ""),
    }


async def diarize(request: Request) -> JSONResponse:
    try:
        body = await request.json()
        url, mode = body.get("audio_url"), body.get("mode", "offline")
        if not isinstance(url, str) or not url:
            raise BadRequest("audio_url is required")
        if mode not in MODES:
            raise BadRequest(f"mode must be one of {', '.join(MODES)}")
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "input"
            await _download(url, src)
            return JSONResponse(await asyncio.to_thread(_diarize, src, mode))
    except BadRequest as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except ValueError:
        return JSONResponse({"error": "body must be JSON: {audio_url, mode}"}, status_code=400)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return JSONResponse({"error": "out of GPU memory; try a shorter file or a streaming mode"}, status_code=500)


async def health(_: Request) -> JSONResponse:
    return JSONResponse({"ok": True, "model": MODEL})


app = Starlette(routes=[Route("/diarize", diarize, methods=["POST"]), Route("/health", health)])

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
