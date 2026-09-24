"""Put speaker labels from a diarization RTTM on Whisper's transcript segments.

Each segment takes the speaker who talks the most during it. Whisper cuts on
pauses, not on speaker changes, so a segment can straddle two speakers or fall
in a gap nobody was labelled in; those are kept whole and marked uncertain
rather than split at a guessed word boundary.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

# Below this share of the segment, the best speaker is a guess.
MIN_SHARE = 0.3
# A second speaker holding this share of the segment makes it mixed.
MIXED_SHARE = 0.3
# Scripts written without spaces between words.
NO_SPACE_LANGUAGES = {"zh", "ja", "ko", "th", "lo", "my", "km", "bo"}


def parse_rttm(rttm: str) -> list[tuple[float, float, str]]:
    """(start, end, speaker) for every SPEAKER line."""
    turns = []
    for line in rttm.splitlines():
        parts = line.split()
        if len(parts) >= 8 and parts[0] == "SPEAKER":
            start, dur = float(parts[3]), float(parts[4])
            turns.append((start, start + dur, parts[7]))
    return sorted(turns)


def label(segments: list[dict[str, Any]], rttm: str) -> list[dict[str, Any]]:
    """segments with speaker, speaker_confidence (share of the segment the speaker
    talks) and speaker_uncertain (under MIN_SHARE, or a second speaker over
    MIXED_SHARE)."""
    turns = parse_rttm(rttm)
    out = []
    for seg in segments:
        start, end = seg["start"], seg["end"]
        if end <= start:
            out.append({**seg, **_at_instant(start, turns)})
            continue
        dur = end - start
        talk: dict[str, float] = defaultdict(float)
        for t_start, t_end, speaker in turns:
            if t_start >= end:
                break  # sorted by start: the rest begin later still
            overlap = min(end, t_end) - max(start, t_start)
            if overlap > 0:
                talk[speaker] += overlap
        ranked = sorted(talk.items(), key=lambda kv: kv[1], reverse=True)
        best, best_s = ranked[0] if ranked else (None, 0.0)
        second_s = ranked[1][1] if len(ranked) > 1 else 0.0
        share = min(best_s / dur, 1.0)
        out.append({
            **seg,
            "speaker": best,
            "speaker_confidence": round(share, 2),
            "speaker_uncertain": share < MIN_SHARE or second_s / dur >= MIXED_SHARE,
        })
    return out


def _at_instant(t: float, turns: list[tuple[float, float, str]]) -> dict[str, Any]:
    """A segment with no length: whoever is talking at that instant."""
    here = sorted({speaker for start, end, speaker in turns if start <= t < end})
    if len(here) == 1:
        return {"speaker": here[0], "speaker_confidence": 1.0, "speaker_uncertain": False}
    # Nobody, or an overlap: take the first for a label, but flag it.
    return {"speaker": here[0] if here else None, "speaker_confidence": 0.0, "speaker_uncertain": True}


def turns(segments: list[dict[str, Any]], language: str) -> list[dict[str, Any]]:
    """Adjacent segments of one speaker merged into a turn."""
    joiner = "" if language in NO_SPACE_LANGUAGES else " "
    merged: list[dict[str, Any]] = []
    for seg in segments:
        if merged and merged[-1]["speaker"] == seg["speaker"]:
            last = merged[-1]
            last["end"] = seg["end"]
            last["text"] = joiner.join(t for t in (last["text"], seg["text"]) if t)
            last["segments"] += 1
        else:
            merged.append({"speaker": seg["speaker"], "start": seg["start"], "end": seg["end"],
                           "text": seg["text"], "segments": 1})
    return merged


def speakers(diarized: list[dict[str, Any]], segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per speaker: seconds talked (from diarization) and transcript segments given."""
    count: dict[str | None, int] = defaultdict(int)
    for seg in segments:
        count[seg["speaker"]] += 1
    rows = [{"speaker": s["speaker"], "seconds": s["seconds"], "segments": count.get(s["speaker"], 0)}
            for s in diarized]
    if count.get(None):
        rows.append({"speaker": None, "seconds": 0, "segments": count[None]})
    return rows
