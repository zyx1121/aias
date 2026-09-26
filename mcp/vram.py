"""GPU memory budget: what the card can hold, what is reserved, and what to evict.

The accounting is the ledger, not nvidia-smi: under WSL the driver quietly
spills an overcommitted card into system memory instead of failing, so
nvidia-smi alone would let models pile up until everything is slow. It is
still consulted as a second check, and to notice memory nobody accounted for.
Per-process numbers (--query-compute-apps) are N/A under WSL, so whole-card
used and free are all there is.
"""

from __future__ import annotations

import itertools
import json
import math
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# The Windows desktop and other apps hold this much of the card.
DESKTOP_RESERVE_MIB = int(os.environ.get("AIAS_DESKTOP_RESERVE_MIB", "1024"))
# Kept free on top of every admission, and required free on nvidia-smi.
SAFETY_MIB = int(os.environ.get("AIAS_SAFETY_MIB", "512"))
# An Ollama runner's CUDA context on top of the size_vram it reports.
OLLAMA_OVERHEAD_MIB = 250
# Ollama's context and KV cache type; keep in step with the ollama service.
OLLAMA_NUM_CTX = int(os.environ.get("AIAS_OLLAMA_NUM_CTX", "32768"))
OLLAMA_NUM_PARALLEL = int(os.environ.get("AIAS_OLLAMA_NUM_PARALLEL", "1"))
OLLAMA_KV_TYPE = os.environ.get("AIAS_OLLAMA_KV_CACHE_TYPE", "q8_0")
# Bytes per cached element; q8_0 and q4_0 store a 2 byte scale per 32 values.
KV_BYTES = {"f16": 2.0, "q8_0": 34 / 32, "q4_0": 18 / 32}
# Without model metadata: the weights twice over, or the weights + 2 GiB.
OLLAMA_FALLBACK_FACTOR = 2.0
OLLAMA_FALLBACK_MIN_EXTRA_MIB = 2048
# nemo with the model loaded and no job, at the nvidia-smi level.
NEMO_BASE_MIB = int(os.environ.get("AIAS_NEMO_BASE_MIB", "1300"))
# Offline diarization adds about 40 MiB of GPU memory per minute of audio
# (39.2 to 40.25 measured on king). Used until a run has been measured; the
# 512 MiB safety margin absorbs the difference.
NEMO_BURST_MIB_PER_MIN = 40
# Short files carry a fixed overhead that would inflate a per-minute rate
# (a 5 s clip measured 240 MiB/min), so only runs this long set the rate,
# and no reservation is smaller than the floor.
NEMO_BURST_MIN_AUDIO_S = 600
NEMO_BURST_FLOOR_MIB = 128


@dataclass(frozen=True)
class AudioBudget:
    """One audio model: its idle footprint at the nvidia-smi level, and the burst a
    clip adds, fixed_mib + per_s up to window_s + extend_per_s past it."""

    base_mib: int
    per_s: float
    window_s: float = 0.0
    extend_per_s: float = 0.0
    fixed_mib: int = 0


AUDIO_BUDGETS: dict[str, AudioBudget] = {
    # Idle is measured on king after the engine's warmup clip, which leaves about
    # 120 MiB of CUDA workspace for good.
    # AudioGen: 5441 MiB loaded (LM fp16 3457, t5-large 1278, EnCodec 225, the CUDA
    # context), 5562 after a clip. The burst grows steeply up to the 10 s window
    # (KV cache), then slower as longer clips are extended 5 s at a time: 370 MiB
    # at 5 s, 732 at 10 s, 1488 at 30 s; the rates cover each.
    "facebook/audiogen-medium": AudioBudget(5570, 76, 10.0, 40),
    # MusicGen: 4613 MiB loaded, 4736 after a clip. Trained on 30 s clips, so its
    # burst grows all the way: 428 MiB at 5 s, 930 to 972 at 10 s, 2718 at 30 s.
    "facebook/musicgen-medium": AudioBudget(4740, 100),
}
# A model without a measured entry is budgeted like the largest known one.
AUDIO_FALLBACK = max(AUDIO_BUDGETS.values(), key=lambda b: b.base_mib)
AUDIO_BURST_FLOOR_MIB = 128
# Only clips this long set the measured factor; shorter ones are mostly noise.
AUDIO_BURST_MIN_S = 5.0

STATE_FILE = Path(os.environ.get("AIAS_STATE", "/state")) / "vram.json"
# Up to this many evictable models, the plan tries every subset for the fewest
# evictions; past it (unlikely on one card) it falls back to a greedy walk.
EXHAUSTIVE_MAX = 12

log = logging.getLogger("aias.vram")


def smi() -> dict[str, Any] | None:
    """Whole-card numbers from nvidia-smi, in MiB."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None
    name, used, total, free = (x.strip() for x in out.stdout.splitlines()[0].split(","))
    return {"name": name, "used": int(used), "total": int(total), "free": int(free)}


def compute_cap() -> float | None:
    """The card's CUDA compute capability (8.6 for an RTX 3080)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=15)
        return float(out.stdout.splitlines()[0])
    except (OSError, subprocess.TimeoutExpired, ValueError, IndexError):
        return None


def budget(total_mib: int) -> int:
    return total_mib - DESKTOP_RESERVE_MIB - SAFETY_MIB


def ollama_kv_mib(model_info: dict[str, Any]) -> int | None:
    """KV cache of an Ollama model at OLLAMA_NUM_CTX, from /api/show's model_info
    (GGUF metadata). None if the fields are missing."""
    arch = model_info.get("general.architecture")
    get = lambda name: model_info.get(f"{arch}.{name}")  # noqa: E731
    layers, heads = get("block_count"), get("attention.head_count")
    kv_heads = get("attention.head_count_kv") or heads
    if not (arch and layers and kv_heads):
        return None
    head_dim = get("attention.key_length") or (
        get("embedding_length") // heads if get("embedding_length") and heads else None
    )
    value_dim = get("attention.value_length") or head_dim
    if not head_dim:
        return None
    ctx = min(OLLAMA_NUM_CTX, get("context_length") or OLLAMA_NUM_CTX) * OLLAMA_NUM_PARALLEL
    per_token = layers * kv_heads * (head_dim + value_dim) * KV_BYTES.get(OLLAMA_KV_TYPE, 2.0)
    return round(ctx * per_token / 2**20)


def ollama_estimate_mib(file_mib: int, model_info: dict[str, Any] | None) -> tuple[int, bool]:
    """(need, used the metadata) for an Ollama model that has not been loaded yet:
    weights + KV cache + runner overhead, or a conservative multiple."""
    kv = ollama_kv_mib(model_info or {})
    if kv is not None:
        return file_mib + kv + OLLAMA_OVERHEAD_MIB, True
    extra = max(file_mib * (OLLAMA_FALLBACK_FACTOR - 1), OLLAMA_FALLBACK_MIN_EXTRA_MIB)
    return round(file_mib + extra) + OLLAMA_OVERHEAD_MIB, False


def nemo_burst_mib(audio_s: float, measured_per_min: float | None = None) -> int:
    """Memory a diarize run adds on top of the idle engine: the highest rate seen in
    a measured run, else NEMO_BURST_MIB_PER_MIN."""
    # Rounded up, so a file fits exactly when rate x minutes fits.
    return max(NEMO_BURST_FLOOR_MIB, math.ceil(burst_rate(measured_per_min) * audio_s / 60))


def audio_base_mib(model: str) -> int:
    return AUDIO_BUDGETS.get(model, AUDIO_FALLBACK).base_mib


def audio_burst_estimate_mib(model: str, duration_s: float) -> float:
    b = AUDIO_BUDGETS.get(model, AUDIO_FALLBACK)
    inside = min(duration_s, b.window_s) if b.window_s else duration_s
    return b.fixed_mib + b.per_s * inside + b.extend_per_s * (duration_s - inside)


def audio_burst_mib(model: str, duration_s: float, measured_factor: float | None = None) -> int:
    """Memory one generate_audio run adds on top of the idle engine: the estimate
    for the clip's length, scaled up (never down) by the highest measured/estimate
    ratio seen on this card."""
    factor = max(1.0, measured_factor or 1.0)
    return max(AUDIO_BURST_FLOOR_MIB, math.ceil(audio_burst_estimate_mib(model, duration_s) * factor))


def burst_rate(measured_per_min: float | None) -> float:
    return measured_per_min or NEMO_BURST_MIB_PER_MIN


def burst_admitted(audio_s: float, measured_per_min: float | None, budget_mib: int,
                   reserved_mib: int, smi_free_mib: int | None, pressure: bool) -> bool:
    """Whether the diarization burst of a file this long is admitted without evicting
    anything: the same make_plan the reservation itself goes through."""
    need = nemo_burst_mib(audio_s, measured_per_min)
    return make_plan(need, budget_mib, reserved_mib, smi_free_mib, pressure, []).fits


def diarize_capacity_min(measured_per_min: float | None, budget_mib: int, reserved_mib: int,
                         smi_free_mib: int | None, pressure: bool, max_minutes: float) -> float:
    """Longest file, in tenths of a minute rounded down, whose burst burst_admitted
    would take: 0 under pressure or when not even the 128 MiB floor fits. Found by
    bisecting on burst_admitted, so it cannot drift from admission."""
    def ok(tenths: int) -> bool:
        return burst_admitted(tenths * 6, measured_per_min, budget_mib, reserved_mib, smi_free_mib, pressure)

    lo, hi = 0, int(max_minutes * 10)  # ok(lo) holds (nothing to fit); find the last ok
    if not ok(1):
        return 0.0
    if ok(hi):
        return hi / 10
    lo = 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    return lo / 10


class Store:
    """Measured footprints, pins and the Ollama models aias loaded, kept in the
    aias-state volume so they survive a restart of the MCP server."""

    def __init__(self, path: Path = STATE_FILE) -> None:
        self.path = path
        self.data: dict[str, Any] = self._defaults()
        try:
            loaded = json.loads(path.read_text())
        except FileNotFoundError:
            return
        except (OSError, ValueError) as exc:
            log.warning("ignoring unreadable %s: %s", path, exc)
            return
        self._adopt(loaded)

    @staticmethod
    def _defaults() -> dict[str, Any]:
        return {"measured": {}, "nemo_burst_per_min": {}, "audio_burst_factor": {}, "pins": [], "managed_ollama": []}

    def _adopt(self, loaded: Any) -> None:
        """Take each field of the state file only if it has the right shape, so a
        damaged file costs its bad fields, not the MCP server."""
        if not isinstance(loaded, dict):
            log.warning("ignoring %s: not a JSON object", self.path)
            return

        def good_record(v: Any) -> bool:
            return isinstance(v, dict) and isinstance(v.get("mib"), int) and v["mib"] > 0

        checks = {
            "measured": lambda v: isinstance(v, dict) and all(isinstance(k, str) and good_record(r) for k, r in v.items()),
            # burst_per_min, the field before, also took short files' rates; not read.
            "nemo_burst_per_min": lambda v: isinstance(v, dict) and all(
                isinstance(k, str) and isinstance(r, (int, float)) and r > 0 for k, r in v.items()
            ),
            "audio_burst_factor": lambda v: isinstance(v, dict) and all(
                isinstance(k, str) and isinstance(r, (int, float)) and r > 0 for k, r in v.items()
            ),
            "pins": lambda v: isinstance(v, list) and all(isinstance(x, str) for x in v),
            "managed_ollama": lambda v: isinstance(v, list) and all(isinstance(x, str) for x in v),
        }
        for name, ok in checks.items():
            if name not in loaded:
                continue
            if ok(loaded[name]):
                self.data[name] = loaded[name]
            else:
                log.warning("ignoring field %r of %s: unexpected shape", name, self.path)

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.data, indent=1, sort_keys=True))
            os.replace(tmp, self.path)
        except OSError:
            pass  # the ledger still works from estimates

    def measured(self, key: str) -> int | None:
        rec = self.data["measured"].get(key)
        return rec["mib"] if rec else None

    def record(self, key: str, mib: int) -> None:
        self.data["measured"][key] = {"mib": int(mib), "at": round(time.time())}
        self.save()

    def burst_rate(self, model: str) -> float | None:
        return self.data["nemo_burst_per_min"].get(model)

    def record_burst(self, model: str, burst_mib: float, audio_s: float) -> None:
        """Keep the highest per-minute burst of a long enough run, so a heavy file
        sets the bar and a short one's fixed overhead does not."""
        if audio_s < NEMO_BURST_MIN_AUDIO_S:
            return
        per_min = burst_mib / (audio_s / 60)
        if per_min > (self.burst_rate(model) or 0):
            self.data["nemo_burst_per_min"][model] = round(per_min, 2)
            self.save()

    def audio_burst_factor(self, model: str) -> float | None:
        return self.data["audio_burst_factor"].get(model)

    def record_audio_burst(self, model: str, burst_mib: float, audio_s: float) -> None:
        """Keep the highest measured/estimate ratio of a long enough clip, so a card
        that needs more than the estimate raises every later reservation."""
        if audio_s < AUDIO_BURST_MIN_S:
            return
        factor = burst_mib / audio_burst_estimate_mib(model, audio_s)
        if factor > (self.audio_burst_factor(model) or 1.0):
            self.data["audio_burst_factor"][model] = round(factor, 3)
            self.save()

    def _toggle(self, name: str, item: str, on: bool) -> None:
        items = set(self.data[name])
        items.add(item) if on else items.discard(item)
        self.data[name] = sorted(items)
        self.save()

    def pinned(self, key: str) -> bool:
        return key in self.data["pins"]

    def pin(self, key: str, on: bool) -> None:
        self._toggle("pins", key, on)

    def managed(self, model: str) -> bool:
        return model in self.data["managed_ollama"]

    def manage(self, model: str, on: bool) -> None:
        self._toggle("managed_ollama", model, on)


@dataclass
class Candidate:
    """Something that could be stopped to make room."""

    key: str
    mib: int
    last_used: float


@dataclass
class Plan:
    need_mib: int
    budget_mib: int
    reserved_mib: int
    smi_free_mib: int | None
    pressure: bool
    replace: list[str] = field(default_factory=list)
    evict: list[str] = field(default_factory=list)
    fits: bool = False  # without evicting anything
    can_fit: bool = False  # after evicting `evict`
    reason: str = ""

    @property
    def free_mib(self) -> int:
        return self.budget_mib - self.reserved_mib

    def view(self) -> dict[str, Any]:
        return {
            "fits": self.fits,
            "can_fit": self.can_fit,
            "need_mib": self.need_mib,
            "free_mib": self.free_mib,
            "budget_mib": self.budget_mib,
            "reserved_mib": self.reserved_mib,
            "smi_free_mib": self.smi_free_mib,
            "replace": self.replace,
            "evict": self.evict,
            "reason": self.reason,
        }


def make_plan(
    need_mib: int,
    budget_mib: int,
    reserved_mib: int,
    smi_free_mib: int | None,
    pressure: bool,
    candidates: list[Candidate],
    replace: list[Candidate] | None = None,
) -> Plan:
    """Admission: ledger reserved + need <= budget, and nvidia-smi free >= need +
    SAFETY_MIB. `replace` is freed whatever happens (the same engine's single
    slot); `candidates` are evicted least recently used first, only as far as
    needed. Callers leave out anything pinned or with a job."""
    replace = replace or []
    plan = Plan(need_mib, budget_mib, reserved_mib, smi_free_mib, pressure,
                replace=[c.key for c in replace])
    if pressure:
        plan.reason = "memory pressure: nvidia-smi shows more in use than the ledger and desktop reserve explain"
        return plan
    if need_mib > budget_mib:
        plan.reason = f"needs {need_mib} MiB, more than the whole {budget_mib} MiB budget"
        return plan

    def ok(freed: int) -> bool:
        ledger = reserved_mib - freed + need_mib <= budget_mib
        card = smi_free_mib is None or smi_free_mib + freed >= need_mib + SAFETY_MIB
        return ledger and card

    base = sum(c.mib for c in replace)
    plan.fits = ok(base)
    if plan.fits:
        plan.can_fit = True
        return plan
    chosen = _fewest(sorted(candidates, key=lambda c: c.last_used), lambda cs: ok(base + sum(c.mib for c in cs)))
    if chosen is None:
        plan.reason = "does not fit even after evicting every model that is not pinned and has no job"
        return plan
    plan.evict = [c.key for c in chosen]
    plan.can_fit = True
    plan.reason = f"fits after evicting {', '.join(plan.evict)}; pass evict=\"auto\" to do it"
    return plan


def _staleness(combo: tuple[Candidate, ...]) -> tuple[float, ...]:
    """Sort key among sets of one size: last use of the newest member first, then
    the next newest, and so on; smaller (longer idle) wins."""
    return tuple(sorted((c.last_used for c in combo), reverse=True))


def _fewest(by_age: list[Candidate], fits: Any) -> list[Candidate] | None:
    """The fewest candidates whose eviction makes room. Among sets of that size, the
    one whose most recently used member was used longest ago (then the next most
    recent, and so on), so recently used models are the last to go."""
    if not fits(by_age):
        return None
    if len(by_age) <= EXHAUSTIVE_MAX:
        for size in range(1, len(by_age) + 1):
            fitting = [c for c in itertools.combinations(by_age, size) if fits(list(c))]
            if fitting:
                return sorted(min(fitting, key=_staleness), key=lambda c: c.last_used)
    # Greedy fallback: oldest first until it fits, then drop what the rest can do without.
    chosen: list[Candidate] = []
    for cand in by_age:
        chosen.append(cand)
        if fits(chosen):
            break
    for cand in reversed(chosen[:]):
        rest = [c for c in chosen if c is not cand]
        if fits(rest):
            chosen = rest
    return chosen
