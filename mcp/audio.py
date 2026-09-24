"""Fetch an audio URL and decode it to 16 kHz mono WAV for the engines.

Every tool that takes audio (diarize, transcribe) goes through fetch_wav, so
the URL checks and size limits live in one place and the engines only ever
see a bounded local file. The MCP server downloads; the decoding itself runs
in the decoder container (decoder/, compose.yaml), which has no network, no
Docker socket, no capabilities and a read-only root.
"""

from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import os
import socket
import stat
import subprocess
import uuid
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
PROBE_SECS = 60
# How long to wait for a killed decoder to be reaped.
REAP_SECS = 5
MAX_REDIRECTS = 5
# The decoder runs as nobody. 2 GiB of address space decodes 2 hours of FLAC.
NOBODY = 65534
DECODER_MAX_AS = 2 * 1024**3
# The volume the MCP server and the decoder share, and the decoder's service.
WORK = Path(os.environ.get("AIAS_AUDIO_WORK", "/work"))
COMPOSE = ["docker", "compose", "-f", os.environ.get("AIAS_COMPOSE", "/opt/aias/compose.yaml")]
TOO_LONG_EXIT = 3  # decode.sh: the probed duration is over the limit
FSIZE_EXIT = 128 + 25  # killed by SIGXFSZ: the output hit RLIMIT_FSIZE


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


def _remove_container(name: str) -> None:
    subprocess.run(["docker", "rm", "-f", name], capture_output=True, text=True, timeout=60)


async def _run_decoder(src: Path, wav: Path, timeout: float) -> tuple[int, str, str]:
    """Run decode.sh in a fresh decoder container named for this run. When it
    ends, however it ends (done, failed, timed out, cancelled), the container is
    removed, and with it every process the decoder started."""
    name = f"aias-decoder-{uuid.uuid4().hex[:12]}"
    proc = await asyncio.create_subprocess_exec(
        *COMPOSE, "run", "--rm", "--no-deps", "-T", "--name", name, "decoder",
        str(src), str(wav), str(MAX_AUDIO_SECS), str(MAX_WAV_BYTES), str(DECODER_MAX_AS),
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout)
    finally:
        if proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(proc.wait(), REAP_SECS)
        # --rm covers a normal end; a killed client leaves the container running.
        await asyncio.to_thread(_remove_container, name)
    return proc.returncode, out.decode(errors="replace"), err.decode(errors="replace")


def _seconds(seconds: float) -> str:
    return f"{seconds:.1f} seconds"


def _wav_seconds(wav: Path) -> float:
    """Length of the decoder's WAV. It was written by nobody, so open it without
    following a symlink and check it is a plain file that nobody owns."""
    try:
        fd = os.open(wav, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as exc:
        raise AudioError(f"the decoder left no readable WAV: {exc.strerror}") from None
    st = os.fstat(fd)
    if not stat.S_ISREG(st.st_mode) or st.st_uid != NOBODY:
        os.close(fd)
        raise AudioError("the decoder's output is not a plain file it owns")
    with os.fdopen(fd, "rb") as f, wave.open(f) as w:
        return w.getnframes() / w.getframerate()


async def _decode(src: Path, wav: Path) -> None:
    """Decode src to 16 kHz mono s16 WAV at wav in the decoder container."""
    limit = f"the limit is {MAX_AUDIO_SECS} seconds"
    try:
        code, out, err = await _run_decoder(src, wav, PROBE_SECS + DECODE_SECS)
    except TimeoutError:
        raise AudioError(f"decoding the audio took longer than {PROBE_SECS + DECODE_SECS} s") from None
    if code == TOO_LONG_EXIT:
        probed = next((line.split("=", 1)[1] for line in out.splitlines() if line.startswith("duration=")), "")
        raise AudioError(f"audio is {_seconds(float(probed))}; {limit}")
    if code == FSIZE_EXIT:
        raise AudioError(f"audio is longer than the {MAX_AUDIO_SECS} second limit")
    if code != 0:
        raise AudioError(f"ffmpeg could not decode the audio: {err.strip()[-500:]}")


def _check_wav(wav: Path) -> float:
    """Length of the decoded WAV, within the limits. Call once the decoder
    container is gone and the directory is root's again."""
    duration = _wav_seconds(wav)
    if duration > MAX_AUDIO_SECS:
        # Decoding stopped at the cap, so the real length is unknown.
        raise AudioError(f"audio is longer than the {MAX_AUDIO_SECS} second limit")
    if duration == 0:
        raise AudioError("audio has no samples")
    return duration


async def fetch_wav(url: str, workdir: Path) -> tuple[Path, float]:
    """Download url into workdir and decode it. Returns the WAV path and its length."""
    # The decoder runs as nobody and writes its WAV here.
    os.chown(workdir, NOBODY, NOBODY)
    src, wav = workdir / "input", workdir / "audio.wav"
    await _download(url, src)
    # The decoder reads the input as nobody.
    os.chmod(src, 0o644)
    try:
        await _decode(src, wav)
    finally:
        # The decoder container is gone by now; the directory goes back to root
        # before the WAV is checked, so the checked file is the one the engine reads.
        os.chown(workdir, 0, 0)
        os.chmod(workdir, 0o700)
    duration = _check_wav(wav)
    src.unlink(missing_ok=True)
    return wav, duration
