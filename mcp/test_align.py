"""Unit tests for speaker alignment: python3 -m unittest discover -s mcp"""

import unittest

import align

RTTM = "\n".join([
    "SPEAKER audio 1 0.000 5.000 <NA> <NA> host <NA> <NA>",
    "SPEAKER audio 1 5.200 3.000 <NA> <NA> guest <NA> <NA>",
    "SPEAKER audio 1 8.500 2.000 <NA> <NA> host <NA> <NA>",
])


def seg(start, end, text):
    return {"start": start, "end": end, "text": text}


class Label(unittest.TestCase):
    def test_question_and_answer(self):
        out = align.label([seg(0.5, 4.5, "有間接翻臉嗎?"), seg(5.3, 8.0, "可能也是有錄下去的時候")], RTTM)
        self.assertEqual([s["speaker"] for s in out], ["host", "guest"])
        self.assertFalse(any(s["speaker_uncertain"] for s in out))
        self.assertEqual(out[0]["speaker_confidence"], 1.0)

    def test_straddling_segment_is_uncertain(self):
        (out,) = align.label([seg(3.0, 7.0, "跨兩人")], RTTM)
        self.assertEqual(out["speaker"], "host")  # 2.0 s against 1.8 s
        self.assertTrue(out["speaker_uncertain"])

    def test_gap_is_uncertain(self):
        (out,) = align.label([seg(10.6, 12.0, "沒人標到")], RTTM)
        self.assertIsNone(out["speaker"])
        self.assertTrue(out["speaker_uncertain"])

    def test_mostly_silence_is_uncertain(self):
        (out,) = align.label([seg(8.0, 12.0, "尾巴")], RTTM)
        self.assertEqual(out["speaker"], "host")
        self.assertEqual(out["speaker_confidence"], 0.5)
        self.assertFalse(out["speaker_uncertain"])
        (short,) = align.label([seg(10.3, 11.0, "只重疊一點")], RTTM)
        self.assertTrue(short["speaker_uncertain"])


class Turns(unittest.TestCase):
    def test_merges_adjacent_same_speaker(self):
        segs = align.label([seg(0.5, 2.0, "你好"), seg(2.0, 4.5, "歡迎"), seg(5.3, 8.0, "謝謝")], RTTM)
        turns = align.turns(segs, "zh")
        self.assertEqual([(t["speaker"], t["text"], t["segments"]) for t in turns],
                         [("host", "你好歡迎", 2), ("guest", "謝謝", 1)])
        self.assertEqual((turns[0]["start"], turns[0]["end"]), (0.5, 4.5))

    def test_spaces_for_english(self):
        segs = align.label([seg(0.5, 2.0, "hello"), seg(2.0, 4.5, "there")], RTTM)
        self.assertEqual(align.turns(segs, "en")[0]["text"], "hello there")

    def test_speaker_summary(self):
        segs = align.label([seg(0.5, 2.0, "a"), seg(5.3, 8.0, "b"), seg(10.6, 12.0, "c")], RTTM)
        rows = align.speakers([{"speaker": "guest", "seconds": 3.0}, {"speaker": "host", "seconds": 7.0}], segs)
        self.assertEqual(rows, [{"speaker": "guest", "seconds": 3.0, "segments": 1},
                                {"speaker": "host", "seconds": 7.0, "segments": 1},
                                {"speaker": None, "seconds": 0, "segments": 1}])


if __name__ == "__main__":
    unittest.main()
