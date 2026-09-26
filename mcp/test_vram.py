"""Unit tests for the VRAM planner and estimates: python3 -m unittest discover -s mcp"""

import unittest

import vram
from vram import Candidate, make_plan

BUDGET = vram.budget(10240)  # 8704 on a 10 GB card


def plan(need, reserved, candidates, smi_free=None, replace=None, pressure=False):
    return make_plan(need, BUDGET, reserved, smi_free, pressure, candidates, replace)


class EvictionPlan(unittest.TestCase):
    def test_fits_without_eviction(self):
        p = plan(4000, 4000, [Candidate("a", 1000, 1)])
        self.assertTrue(p.fits)
        self.assertEqual(p.evict, [])

    def test_single_minimal(self):
        # The king case: qwen3:8b next to nemo (older) and Whisper. LRU alone would
        # take both; stopping Whisper is enough.
        nemo, whisper = Candidate("nemo", 843, 1), Candidate("whisper", 4187, 2)
        p = plan(7686, 843 + 4187, [whisper, nemo])
        self.assertFalse(p.fits)
        self.assertTrue(p.can_fit)
        self.assertEqual(p.evict, ["whisper"])

    def test_needs_two(self):
        cands = [Candidate("a", 2000, 1), Candidate("b", 2000, 2), Candidate("c", 2000, 3)]
        p = plan(5000, 7000, cands)
        self.assertTrue(p.can_fit)
        # 1704 free; two of the 2000s are needed, and the two idle longest are chosen.
        self.assertEqual(p.evict, ["a", "b"])

    def test_prefers_least_recently_used_among_minimal(self):
        cands = [Candidate("old", 3000, 1), Candidate("new", 3000, 2)]
        p = plan(4000, 6000, cands)
        self.assertEqual(p.evict, ["old"])

    def test_fewest_not_greedy(self):
        # Oldest first would take both 1500s; the single 3000 is fewer.
        cands = [Candidate("a", 1500, 1), Candidate("b", 1500, 2), Candidate("c", 3000, 3)]
        p = plan(3000, 8704, cands)
        self.assertEqual(p.evict, ["c"])

    def test_same_count_prefers_oldest(self):
        cands = [Candidate("new", 3000, 9), Candidate("old", 3000, 1), Candidate("mid", 3000, 5)]
        p = plan(3000, 8704, cands)
        self.assertEqual(p.evict, ["old"])

    def test_tie_goes_to_the_set_whose_newest_is_oldest(self):
        # By age A (oldest) .. D (newest). Freeing 4000 takes two; (A, D) and (B, C)
        # both fit, but (A, D) holds the newest model, so (B, C) goes.
        cands = [Candidate("A", 1000, 1), Candidate("B", 2000, 2), Candidate("C", 2000, 3), Candidate("D", 3000, 4)]
        p = plan(4000, 8704, cands)
        self.assertEqual(p.evict, ["B", "C"])

    def test_greedy_fallback_past_the_limit(self):
        cands = [Candidate(f"m{i}", 100, i) for i in range(vram.EXHAUSTIVE_MAX + 3)]
        p = plan(1000, 8704, cands)
        self.assertTrue(p.can_fit)
        self.assertEqual(len(p.evict), 10)
        self.assertEqual(p.evict, [f"m{i}" for i in range(10)])

    def test_nothing_is_enough(self):
        p = plan(8000, 6000, [Candidate("a", 1000, 1), Candidate("b", 1000, 2)])
        self.assertFalse(p.can_fit)
        self.assertEqual(p.evict, [])

    def test_larger_than_budget(self):
        p = plan(BUDGET + 1, 0, [])
        self.assertFalse(p.can_fit)

    def test_nvidia_smi_free_also_counts(self):
        # The ledger has room, the card does not until something is stopped.
        p = plan(3000, 1000, [Candidate("a", 1000, 1)], smi_free=2600)
        self.assertEqual(p.evict, ["a"])

    def test_replace_is_freed_first(self):
        p = plan(4000, 7000, [Candidate("a", 1000, 1)], replace=[Candidate("vllm:old", 4000, 0)])
        self.assertTrue(p.fits)
        self.assertEqual(p.replace, ["vllm:old"])
        self.assertEqual(p.evict, [])

    def test_pressure_refuses(self):
        p = plan(100, 0, [], pressure=True)
        self.assertFalse(p.can_fit)


QWEN3_8B = {
    "general.architecture": "qwen3",
    "qwen3.block_count": 36,
    "qwen3.attention.head_count": 32,
    "qwen3.attention.head_count_kv": 8,
    "qwen3.attention.key_length": 128,
    "qwen3.attention.value_length": 128,
    "qwen3.context_length": 40960,
    "qwen3.embedding_length": 4096,
}


class NemoBurst(unittest.TestCase):
    def test_short_runs_do_not_set_the_rate(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            store = vram.Store(Path(tmp) / "vram.json")
            store.record_burst("m", 20, 5)  # 240 MiB/min from fixed overhead
            self.assertIsNone(store.burst_rate("m"))
            store.record_burst("m", 3400, 5203)
            self.assertAlmostEqual(store.burst_rate("m"), 39.21, places=2)

    def test_default_rate_fits_ep103_next_to_whisper(self):
        # king: Whisper 4187 + nemo 843; EP103 is 5203 s.
        burst = vram.nemo_burst_mib(5203, None)
        self.assertEqual(burst, 3469)
        self.assertLessEqual(4187 + 843 + burst, BUDGET)
        self.assertEqual(vram.diarize_capacity_min(None, BUDGET, 4187 + 843, None, False, 120), 91.8)

    def test_floor(self):
        self.assertEqual(vram.nemo_burst_mib(5, 39.2), vram.NEMO_BURST_FLOOR_MIB)
        self.assertEqual(vram.nemo_burst_mib(5203, 39.2), 3400)  # rounded up


AG = "facebook/audiogen-medium"
# (model, clip seconds, burst MiB) measured on king, RTX 3080.
MEASURED_BURSTS = [
    (AG, 5, 370), (AG, 10, 732), (AG, 30, 1488),
    ("facebook/musicgen-medium", 5, 428), ("facebook/musicgen-medium", 10, 972),
    ("facebook/musicgen-medium", 30, 2718),
]


class AudioBurst(unittest.TestCase):
    def test_covers_what_king_measured(self):
        # 370 MiB at 5 s, 732 at 10 s, 1488 at 30 s (RTX 3080, AudioGen medium).
        for seconds, measured in [(5, 370), (10, 732), (30, 1488)]:
            with self.subTest(seconds=seconds):
                self.assertGreaterEqual(vram.audio_burst_mib(AG, seconds), measured)
        self.assertEqual(vram.audio_burst_mib(AG, 0.5), vram.AUDIO_BURST_FLOOR_MIB)

    def test_slower_past_the_window(self):
        per_s_inside = vram.audio_burst_mib(AG, 10) - vram.audio_burst_mib(AG, 9)
        per_s_past = vram.audio_burst_mib(AG, 21) - vram.audio_burst_mib(AG, 20)
        self.assertLess(per_s_past, per_s_inside)

    def test_factor_raises_but_never_lowers(self):
        self.assertEqual(vram.audio_burst_mib(AG, 10, 1.5), 1140)
        self.assertEqual(vram.audio_burst_mib(AG, 10, 0.5), vram.audio_burst_mib(AG, 10))

    def test_only_long_enough_clips_set_the_factor(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            store = vram.Store(Path(tmp) / "vram.json")
            store.record_audio_burst(AG, 900, 2)
            self.assertIsNone(store.audio_burst_factor(AG))
            store.record_audio_burst(AG, 700, 10)  # under the estimate: nothing to raise
            self.assertIsNone(store.audio_burst_factor(AG))
            store.record_audio_burst(AG, 1140, 10)
            store.record_audio_burst(AG, 1600, 30)  # a lower ratio: the highest stays
            self.assertEqual(store.audio_burst_factor(AG), 1.5)
            self.assertEqual(vram.Store(Path(tmp) / "vram.json").audio_burst_factor(AG), 1.5)

    def test_bad_state_field_dropped(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "vram.json"
            path.write_text('{"audio_burst_factor": {"m": "big"}}')
            with self.assertLogs("aias.vram", "WARNING"):
                self.assertEqual(vram.Store(path).data["audio_burst_factor"], {})

    def test_unknown_model_budgeted_as_the_largest(self):
        largest = max(b.base_mib for b in vram.AUDIO_BUDGETS.values())
        self.assertEqual(vram.audio_base_mib("someone/else"), largest)

    def test_every_model_covers_its_measured_bursts(self):
        for model, seconds, measured in MEASURED_BURSTS:
            with self.subTest(model=model, seconds=seconds):
                self.assertGreaterEqual(vram.audio_burst_mib(model, seconds), measured)

    def test_does_not_fit_next_to_whisper_on_10_gb(self):
        # Whisper 4187 + AudioGen 5570 (or MusicGen 4740) is over the 8704 MiB budget: model_up must
        # refuse with Whisper in the eviction plan, not overcommit the card.
        p = plan(vram.audio_base_mib(AG), 4187, [Candidate("whisper", 4187, 1)])
        self.assertFalse(p.fits)
        self.assertEqual(p.evict, ["whisper"])


class DiarizeCapacity(unittest.TestCase):
    def cap(self, reserved, rate=39.3, smi_free=None, pressure=False):
        return vram.diarize_capacity_min(rate, BUDGET, reserved, smi_free, pressure, 120)

    def admitted(self, minutes, reserved, rate=39.3, smi_free=None, pressure=False):
        return vram.burst_admitted(minutes * 60, rate, BUDGET, reserved, smi_free, pressure)

    def test_pressure_is_zero(self):
        self.assertEqual(self.cap(843, pressure=True), 0.0)

    def test_room_under_the_floor_is_zero(self):
        # 8704 - 843 - 7761 = 100 MiB, less than the 128 MiB every run reserves.
        self.assertEqual(self.cap(843 + 7761), 0.0)
        self.assertFalse(self.admitted(0.1, 843 + 7761))

    def test_capped_at_the_audio_limit(self):
        self.assertEqual(self.cap(843 + 2000), 120.0)

    def test_boundary_matches_admission(self):
        for reserved, smi_free in [(843 + 4457, None), (843 + 4187, None), (843 + 4457, 3000), (843, 3000)]:
            n = self.cap(reserved, smi_free=smi_free)
            with self.subTest(reserved=reserved, smi_free=smi_free, n=n):
                self.assertTrue(self.admitted(n, reserved, smi_free=smi_free))
                self.assertFalse(self.admitted(round(n + 0.1, 1), reserved, smi_free=smi_free))

    def test_rounds_down(self):
        # 3404 MiB / 39.3 = 86.61...: 86.6, never 86.7.
        self.assertEqual(self.cap(843 + 4457), 86.6)


class StateFile(unittest.TestCase):
    def load(self, text):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "vram.json"
            path.write_text(text)
            with self.assertLogs("aias.vram", "WARNING") if text.strip() not in ("{}",) else _nolog():
                return vram.Store(path).data

    def test_not_an_object(self):
        self.assertEqual(self.load("[1, 2]"), vram.Store._defaults())

    def test_not_json(self):
        self.assertEqual(self.load("{oops"), vram.Store._defaults())

    def test_bad_field_dropped_good_kept(self):
        data = self.load('{"measured": {"a": {"mib": "big"}}, "pins": ["vllm:x"]}')
        self.assertEqual(data["measured"], {})
        self.assertEqual(data["pins"], ["vllm:x"])


class _nolog:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class OllamaEstimate(unittest.TestCase):
    def test_kv_from_metadata(self):
        # 32768 tokens x 36 layers x 8 heads x (128 + 128) x 34/32 bytes
        self.assertEqual(vram.ollama_kv_mib(QWEN3_8B), 2448)

    def test_head_dim_from_embedding(self):
        info = {k: v for k, v in QWEN3_8B.items() if "_length" not in k or "context" in k or "embedding" in k}
        self.assertEqual(vram.ollama_kv_mib(info), 2448)

    def test_estimate_with_metadata(self):
        need, used = vram.ollama_estimate_mib(4988, QWEN3_8B)
        self.assertTrue(used)
        self.assertEqual(need, 4988 + 2448 + vram.OLLAMA_OVERHEAD_MIB)

    def test_conservative_without_metadata(self):
        need, used = vram.ollama_estimate_mib(500, None)
        self.assertFalse(used)
        self.assertEqual(need, 500 + 2048 + vram.OLLAMA_OVERHEAD_MIB)


if __name__ == "__main__":
    unittest.main()
