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
