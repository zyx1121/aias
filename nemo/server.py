"""aias nemo engine: speaker diarization over HTTP.

Loads one NeMo Sortformer model (nvidia/Nemotron-3-Diarization by default)
onto the GPU at startup, then answers POST /diarize for the MCP server. It is
reachable only on the compose network; the MCP tools are its public face.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import os
import socket
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
# Offline peak GPU memory grows about 40 MiB per minute of audio (3.5 GB at
# 87 minutes), so 2 hours stays inside a 10 GB card next to the desktop.
MAX_AUDIO_SECS = 2 * 3600
# 2 hours of 16 kHz mono s16 is 230.4 MB; -fs stops ffmpeg a little past that
# even when the input's timestamps defeat -t (chained Ogg streams restart them).
MAX_WAV_BYTES = 240_000_000
# ffmpeg checks -fs after writing each packet, so leave room for one.
FS_HEADROOM = 64 * 1024
DOWNLOAD_SECS = 600
DECODE_SECS = 600
MAX_REDIRECTS = 5
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


def _public_ip(host: str) -> str:
    """Resolve host and return an address to connect to, refusing the whole name if
    any address is private, loopback, link-local or reserved. The caller connects to
    the returned address, so a second lookup cannot swap in an internal one."""
    try:
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except (socket.gaierror, UnicodeError) as exc:
        raise BadRequest(f"cannot resolve {host}") from exc
    addrs = sorted({info[4][0] for info in infos}, key=lambda a: ":" in a)  # IPv4 first
    for addr in addrs:
        ip = ipaddress.ip_address(addr.split("%")[0])
        if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
            ip = ip.ipv4_mapped
        if not ip.is_global or ip.is_multicast:
            raise BadRequest(f"audio_url host {host} resolves to {ip}, which is not a public address")
    return addrs[0]


async def _download(url: str, dest: Path) -> None:
    """Download url to dest. Follows redirects by hand so every hop gets the same
    public address check, and gives up after DOWNLOAD_SECS in total."""
    try:
        target = httpx.URL(url)
    except httpx.InvalidURL as exc:
        raise BadRequest(f"audio_url is not a valid URL: {exc}") from exc
    try:
        async with asyncio.timeout(DOWNLOAD_SECS):
            async with httpx.AsyncClient(
                follow_redirects=False, trust_env=False, timeout=httpx.Timeout(60, connect=15)
            ) as client:
                for _ in range(MAX_REDIRECTS + 1):
                    if target.scheme not in ("http", "https") or not target.host:
                        raise BadRequest("audio_url must be an http or https URL")
                    ip = await asyncio.to_thread(_public_ip, target.host)
                    request = client.build_request(
                        "GET",
                        target.copy_with(host=ip),
                        headers={"Host": target.netloc.decode("ascii")},
                        # TLS still verifies the certificate against the real name.
                        extensions={"sni_hostname": target.host} if target.scheme == "https" else {},
                    )
                    resp = await client.send(request, stream=True)
                    try:
                        if resp.is_redirect:
                            location = resp.headers.get("location")
                            if not location:
                                raise BadRequest(
                                    f"audio_url answered HTTP {resp.status_code} without a Location header"
                                )
                            target = target.join(location)
                            continue
                        if resp.status_code != 200:
                            raise BadRequest(f"audio_url answered HTTP {resp.status_code}")
                        size = 0
                        with dest.open("wb") as f:
                            async for chunk in resp.aiter_bytes():
                                size += len(chunk)
                                if size > MAX_BYTES:
                                    raise BadRequest("audio is larger than 2 GB")
                                f.write(chunk)
                        return
                    finally:
                        await resp.aclose()
                raise BadRequest(f"audio_url redirected more than {MAX_REDIRECTS} times")
    except TimeoutError:
        raise BadRequest(f"downloading audio_url took longer than {DOWNLOAD_SECS} s") from None
    except httpx.HTTPError as exc:
        # Timeouts carry no message, so name the exception type too.
        raise BadRequest(f"could not download audio_url: {type(exc).__name__} {exc}".rstrip()) from exc


def _probe_seconds(src: Path) -> float | None:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(src)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    try:
        return float(proc.stdout.strip())
    except ValueError:
        return None


def _too_long(seconds: float) -> BadRequest:
    return BadRequest(f"audio is {seconds / 3600:.1f} hours; the limit is {MAX_AUDIO_SECS // 3600} hours")


def _diarize(src: Path, mode: str) -> dict[str, Any]:
    probed = _probe_seconds(src)
    if probed is not None and probed > MAX_AUDIO_SECS:
        raise _too_long(probed)
    wav = src.with_suffix(".16k.wav")
    # -t and -fs bound the decoded size even when the container's duration is missing or wrong.
    proc = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(src),
         "-t", str(MAX_AUDIO_SECS + 1), "-fs", str(MAX_WAV_BYTES - FS_HEADROOM), "-ac", "1", "-ar", "16000", str(wav)],
        capture_output=True,
        text=True,
        timeout=DECODE_SECS,
    )
    if proc.returncode != 0:
        raise BadRequest(f"ffmpeg could not decode the audio: {proc.stderr.strip()[-500:]}")
    duration = sf.info(str(wav)).duration
    if duration > MAX_AUDIO_SECS:
        # Decoding stopped at the cap, so the real length is unknown.
        raise BadRequest(f"audio is longer than the {MAX_AUDIO_SECS // 3600} hour limit")

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
    except json.JSONDecodeError:
        body = None
    if not isinstance(body, dict):
        return JSONResponse({"error": "body must be a JSON object: {audio_url, mode}"}, status_code=400)
    try:
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
    except subprocess.TimeoutExpired:
        return JSONResponse({"error": f"decoding the audio took longer than {DECODE_SECS} s"}, status_code=400)
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
