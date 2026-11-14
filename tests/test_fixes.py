# Tests of transcript_fixes.py with a fake engine (no model needed):
#   python -m unittest discover -s tests
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from transcript_fixes import Fixer, collapse, is_looped, words_of  # noqa: E402


def seg(start, end, text, **k):
    d = {"start": start, "end": end, "text": " " + text, "tokens": [1] * max(1, len(text.split()) * 2),
         "avg_logprob": -0.3, "no_speech_prob": 0.05}
    d.update(k)
    return d


class FakeEngine:
    """decode_range gives what's really said in the audio (self.truth), but
    loops again if it starts before `trap` (the word that makes it loop)."""

    def __init__(self, truth, trap=None, words=None, speech=None):
        self.truth = truth
        self.trap = trap
        self.words = words or {}
        self.speech = speech
        self.calls = []

    def decode_range(self, audio, start, end, language):
        self.calls.append((start, end))
        if self.trap is not None and start < self.trap:
            return [seg(start, end, "no no no no no no no no no")]
        return [dict(s) for s in self.truth if s["start"] >= start - 0.2 and s["start"] < end]

    def align(self, audio, s, nxt, language):
        return self.words.get(round(s["start"], 2))

    def speech_in_range(self, audio, start, end):
        return self.speech


def run(fixer, segments, step=True):
    out = []
    for s in segments:
        out += fixer.feed(s)
    out += fixer.flush()
    return out


AUDIO = np.zeros(16000 * 120, np.float32)


class TestDetection(unittest.TestCase):
    def test_looped(self):
        self.assertTrue(is_looped(seg(0, 5, "no, no, no, no, no, no")))
        self.assertTrue(is_looped(seg(0, 5, "I'm sorry. I'm sorry. I'm sorry. I'm sorry.")))
        self.assertFalse(is_looped(seg(0, 5, "No, no, I said that I would come tomorrow.")))
        self.assertFalse(is_looped(seg(0, 5, "Very very good.")))

    def test_collapse(self):
        self.assertEqual(collapse(seg(0, 5, "no, no, no, no, no, no")), "no,")
        self.assertEqual(collapse(seg(0, 5, "Well, I'm sorry. I'm sorry. I'm sorry. I'm sorry.")),
                         "Well, I'm sorry.")

    def test_words(self):
        self.assertEqual(words_of("Don't stop, ¡Vamos!"), ["don't", "stop", "vamos"])


class TestLoops(unittest.TestCase):
    def test_loop_recovered(self):
        truth = [seg(12.0, 14.0, "Wait a moment."), seg(15.0, 18.0, "I need to tell you something."),
                 seg(19.0, 24.0, "The car was not there when I came back.")]
        eng = FakeEngine(truth, trap=11.2)
        f = Fixer(eng, AUDIO, "en")
        out = run(f, [seg(5.0, 9.0, "Hello, how are you?"),
                      seg(11.0, 30.0, "no no no no no no no no no no no no"),
                      seg(30.0, 33.0, "Fine, thanks.")])
        texts = [s["text"].strip() for s in out]
        self.assertEqual(texts[0], "Hello, how are you?")
        self.assertIn("Wait a moment.", texts)
        self.assertIn("The car was not there when I came back.", texts)
        self.assertEqual(texts[-1], "Fine, thanks.")
        self.assertEqual(texts[1], "no")
        self.assertEqual(f.loops_fixed, 1)
        # all in order, nothing over the next segment
        self.assertTrue(all(a["start"] <= b["start"] for a, b in zip(out, out[1:])))
        self.assertTrue(all(s["end"] <= 30.0 for s in out[:-1]))

    def test_same_text_run(self):
        truth = [seg(41.0, 44.0, "We have to go now."), seg(45.0, 49.0, "Get your coat.")]
        eng = FakeEngine(truth)
        f = Fixer(eng, AUDIO, "en")
        out = run(f, [seg(38.0, 40.0, "Thank you."), seg(40.0, 42.0, "Thank you."),
                      seg(42.0, 44.0, "Thank you."), seg(44.0, 46.0, "Thank you."),
                      seg(50.0, 52.0, "Bye.")])
        texts = [s["text"].strip() for s in out]
        self.assertEqual(texts.count("Thank you."), 1)
        self.assertIn("We have to go now.", texts)
        self.assertIn("Get your coat.", texts)

    def test_two_same_is_not_a_loop(self):
        eng = FakeEngine([])
        f = Fixer(eng, AUDIO, "en")
        out = run(f, [seg(1, 2, "Yes."), seg(3, 4, "Yes."), seg(5, 7, "Of course.")])
        self.assertEqual([s["text"].strip() for s in out], ["Yes.", "Yes.", "Of course."])
        self.assertEqual(eng.calls, [])

    def test_no_voice_only_collapses(self):
        eng = FakeEngine([seg(20, 22, "Hallucination.")], speech=0.0)
        f = Fixer(eng, AUDIO, "en")
        out = run(f, [seg(10.0, 30.0, "la la la la la la la la la la"), seg(31.0, 33.0, "Okay.")])
        self.assertEqual([s["text"].strip() for s in out], ["la", "Okay."])
        self.assertEqual(eng.calls, [])

    def test_recovery_that_loops_again_is_rejected(self):
        eng = FakeEngine([], trap=1000)  # always loops
        f = Fixer(eng, AUDIO, "en")
        out = run(f, [seg(10.0, 30.0, "no no no no no no no no no no"), seg(31.0, 33.0, "Okay.")])
        self.assertEqual([s["text"].strip() for s in out], ["no", "Okay."])
        self.assertEqual(len(eng.calls), 3)  # three starts tried

    def test_off(self):
        eng = FakeEngine([])
        f = Fixer(eng, AUDIO, "en", loop_recovery=False, retime=False)
        segs = [seg(10.0, 30.0, "no no no no no no no no no no"), seg(31.0, 33.0, "Okay.")]
        self.assertEqual(len(run(f, segs)), 2)
        self.assertEqual(eng.calls, [])

    def test_streaming_delay(self):
        # a clean segment comes out as soon as the next one arrives
        f = Fixer(FakeEngine([]), AUDIO, "en")
        self.assertEqual(f.feed(seg(1, 2, "One.")), [])
        self.assertEqual([s["text"] for s in f.feed(seg(3, 4, "Two."))], [" One."])
        self.assertEqual([s["text"] for s in f.flush()], [" Two."])


class TestRetime(unittest.TestCase):
    def test_stretched_moved(self):
        words = {10.0: [{"word": " Where", "start": 27.1, "end": 27.4, "probability": 0.9},
                        {"word": " were", "start": 27.4, "end": 27.6, "probability": 0.8},
                        {"word": " you?", "start": 27.6, "end": 28.1, "probability": 0.85}]}
        f = Fixer(FakeEngine([], words=words), AUDIO, "en")
        out = run(f, [seg(10.0, 28.2, "Where were you?"), seg(29.0, 31.0, "At home.")])
        self.assertAlmostEqual(out[0]["start"], 27.1)
        self.assertAlmostEqual(out[0]["end"], 28.1)
        self.assertEqual(f.retimed, 1)

    def test_low_confidence_kept(self):
        words = {10.0: [{"word": " Where", "start": 27.1, "end": 27.4, "probability": 0.1},
                        {"word": " were", "start": 27.4, "end": 27.6, "probability": 0.2}]}
        f = Fixer(FakeEngine([], words=words), AUDIO, "en")
        out = run(f, [seg(10.0, 28.2, "Where were you?"), seg(29.0, 31.0, "At home.")])
        self.assertEqual(out[0]["start"], 10.0)

    def test_normal_not_touched(self):
        words = {10.0: [{"word": " x", "start": 12.0, "end": 13.0, "probability": 0.9}]}
        f = Fixer(FakeEngine([], words=words), AUDIO, "en")
        out = run(f, [seg(10.0, 14.0, "This is a normal sentence of some words.")])
        self.assertEqual(out[0]["start"], 10.0)


if __name__ == "__main__":
    unittest.main()
