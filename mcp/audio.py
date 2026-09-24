"""Fetch an audio URL and decode it to 16 kHz mono WAV for the engines.

Every tool that takes audio (diarize, transcribe) goes through fetch_wav, so
the URL checks and size limits live in one place and the engines only ever
see a bounded local file.
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
import subprocess
import wave
from pathlib import Path

import httpx

MAX_BYTES = 2 * 1024**3
# Offline diarization peaks about 40 MiB of GPU memory per minute of audio
# (3.5 GB at 87 minutes), so 2 hours stays inside a 10 GB card.
MAX_AUDIO_SECS = 2 * 3600
# 2 hours of 16 kHz mono s16 is 230.4 MB; -fs stops ffmpeg a little past that
# even when the input's timestamps defeat -t (chained Ogg streams restart them).
MAX_WAV_BYTES = 240_000_000
# ffmpeg checks -fs only after its muxer flushes, which overshot by 190 KB on
# ffmpeg 7.1; stopping 5 MB early still leaves room for a full 2 hours.
FS_HEADROOM = 5_000_000
DOWNLOAD_SECS = 600
DECODE_SECS = 600
MAX_REDIRECTS = 5


class AudioError(Exception):
    """The audio URL or its content is not acceptable; the message says why."""


def _public_ip(host: str) -> str:
    """Resolve host and return an address to connect to, refusing the whole name if
    any address is private, loopback, link-local or reserved. The caller connects to
    the returned address, so a second lookup cannot swap in an internal one."""
    try:
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except (socket.gaierror, UnicodeError) as exc:
        raise AudioError(f"cannot resolve {host}") from exc
    addrs = sorted({info[4][0] for info in infos}, key=lambda a: ":" in a)  # IPv4 first
    for addr in addrs:
        ip = ipaddress.ip_address(addr.split("%")[0])
        if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
            ip = ip.ipv4_mapped
        if not ip.is_global or ip.is_multicast:
            raise AudioError(f"audio_url host {host} resolves to {ip}, which is not a public address")
    return addrs[0]


async def _download(url: str, dest: Path) -> None:
    """Download url to dest. Follows redirects by hand so every hop gets the same
    public address check, and gives up after DOWNLOAD_SECS in total."""
    try:
        target = httpx.URL(url)
    except httpx.InvalidURL as exc:
        raise AudioError(f"audio_url is not a valid URL: {exc}") from exc
    try:
        async with asyncio.timeout(DOWNLOAD_SECS):
            async with httpx.AsyncClient(
                follow_redirects=False, trust_env=False, timeout=httpx.Timeout(60, connect=15)
            ) as client:
                for _ in range(MAX_REDIRECTS + 1):
                    if target.scheme not in ("http", "https") or not target.host:
                        raise AudioError("audio_url must be an http or https URL")
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
                                raise AudioError(
                                    f"audio_url answered HTTP {resp.status_code} without a Location header"
                                )
                            target = target.join(location)
                            continue
                        if resp.status_code != 200:
                            raise AudioError(f"audio_url answered HTTP {resp.status_code}")
                        size = 0
                        with dest.open("wb") as f:
                            async for chunk in resp.aiter_bytes():
                                size += len(chunk)
                                if size > MAX_BYTES:
                                    raise AudioError("audio is larger than 2 GB")
                                f.write(chunk)
                        return
                    finally:
                        await resp.aclose()
                raise AudioError(f"audio_url redirected more than {MAX_REDIRECTS} times")
    except TimeoutError:
        raise AudioError(f"downloading audio_url took longer than {DOWNLOAD_SECS} s") from None
    except httpx.HTTPError as exc:
        # Timeouts carry no message, so name the exception type too.
        raise AudioError(f"could not download audio_url: {type(exc).__name__} {exc}".rstrip()) from exc


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


def _decode(src: Path, wav: Path) -> float:
    """Decode src to 16 kHz mono s16 WAV at wav and return its length in seconds."""
    probed = _probe_seconds(src)
    if probed is not None and probed > MAX_AUDIO_SECS:
        raise AudioError(f"audio is {probed / 3600:.1f} hours; the limit is {MAX_AUDIO_SECS // 3600} hours")
    # -t and -fs bound the decoded size even when the container's duration is missing or wrong.
    try:
        proc = subprocess.run(
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(src),
             "-t", str(MAX_AUDIO_SECS + 1), "-fs", str(MAX_WAV_BYTES - FS_HEADROOM),
             "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav)],
            capture_output=True,
            text=True,
            timeout=DECODE_SECS,
        )
    except subprocess.TimeoutExpired:
        raise AudioError(f"decoding the audio took longer than {DECODE_SECS} s") from None
    if proc.returncode != 0:
        raise AudioError(f"ffmpeg could not decode the audio: {proc.stderr.strip()[-500:]}")
    with wave.open(str(wav)) as w:
        duration = w.getnframes() / w.getframerate()
    if duration > MAX_AUDIO_SECS:
        # Decoding stopped at the cap, so the real length is unknown.
        raise AudioError(f"audio is longer than the {MAX_AUDIO_SECS // 3600} hour limit")
    if duration == 0:
        raise AudioError("audio has no samples")
    return duration


async def fetch_wav(url: str, workdir: Path) -> tuple[Path, float]:
    """Download url into workdir and decode it. Returns the WAV path and its length."""
    src, wav = workdir / "input", workdir / "audio.wav"
    await _download(url, src)
    duration = await asyncio.to_thread(_decode, src, wav)
    src.unlink(missing_ok=True)
    return wav, duration
