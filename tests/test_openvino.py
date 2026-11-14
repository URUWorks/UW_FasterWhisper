# The OpenVINO backend with a fake openvino_genai (no model, no Intel GPU):
# the segments, words, blocks, language and the CPU fallback.
import os
import sys
import types
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import openvino_backend as ob  # noqa: E402


class Chunk:
    def __init__(self, a, b, text):
        self.start_ts, self.end_ts, self.text, self.token_ids = a, b, text, [1, 2]


class Word:
    def __init__(self, a, b, w):
        self.start_ts, self.end_ts, self.word = a, b, w


class Result:
    def __init__(self, n):
        self.chunks = [Chunk(1.0, 2.5, " Hello there."), Chunk(3.0, 4.0, " Bye.")]
        self.words = [Word(1.0, 1.5, " Hello"), Word(1.5, 2.5, " there."), Word(3.0, 4.0, " Bye.")]
        self.texts = [" Hello there. Bye."]
        self.languages = ["<|es|>"]
        self.n = n


class Config:
    num_beams = 1


class Pipe:
    created = []
    generated = []

    def __init__(self, path, device, **kw):
        if device == "GPU" and os.environ.get("FAKE_GPU_FAILS"):
            raise RuntimeError("no GPU driver")
        Pipe.created.append((device, kw))

    def get_generation_config(self):
        return Config()

    def generate(self, samples, cfg):
        Pipe.generated.append((len(samples), getattr(cfg, "language", None)))
        return Result(len(samples))


def install_fakes(devices):
    g = types.ModuleType("openvino_genai")
    g.WhisperPipeline = Pipe
    ov = types.ModuleType("openvino")
    ov.Core = lambda: types.SimpleNamespace(available_devices=devices)
    sys.modules["openvino_genai"] = g
    sys.modules["openvino"] = ov


def args(**k):
    a = dict(model="/fake", model_dir=None, local_files_only=True, device="openvino", task="transcribe",
             beam_size=5, initial_prompt=None, hotwords=None, repetition_penalty=1.0, no_repeat_ngram_size=0,
             word_timestamps=False, sentence=False, max_line_width=None, max_line_count=None,
             language=None, multilingual=False)
    a.update(k)
    return types.SimpleNamespace(**a)


class TestOpenVINO(unittest.TestCase):
    def setUp(self):
        Pipe.created.clear()
        Pipe.generated.clear()
        os.environ.pop("FAKE_GPU_FAILS", None)
        self._resolve = ob.resolve_model
        ob.resolve_model = lambda *a: "/fake"

    def tearDown(self):
        ob.resolve_model = self._resolve

    def test_gpu_first_and_language(self):
        install_fakes(["CPU", "GPU"])
        e = ob.OpenVINOEngine(args(), "/nowhere", lambda m: None)
        self.assertEqual(e.device, "GPU")
        audio = np.zeros(16000 * 500, np.float32)  # 3 blocks
        segs, lang, _p = e.transcribe(audio, args())
        segs = list(segs)
        self.assertEqual(lang, "es")
        self.assertEqual(len(Pipe.generated), 3)
        # the blocks after the first use the detected language
        self.assertEqual(Pipe.generated[1][1], "<|es|>")
        self.assertEqual(len(segs), 6)
        self.assertTrue(all(a["start"] <= b["start"] for a, b in zip(segs, segs[1:])))
        self.assertGreater(segs[2]["start"], 200)

    def test_cpu_fallback_and_words(self):
        install_fakes(["CPU", "GPU"])
        os.environ["FAKE_GPU_FAILS"] = "1"
        e = ob.OpenVINOEngine(args(word_timestamps=True), "/nowhere", lambda m: None)
        self.assertEqual(e.device, "CPU")
        self.assertEqual(Pipe.created[-1][1], {"word_timestamps": True})
        segs, _l, _p = e.transcribe(np.zeros(16000 * 10, np.float32), args(language="en"))
        segs = list(segs)
        self.assertEqual([len(s["words"]) for s in segs], [2, 1])

    def test_decode_range_offset(self):
        install_fakes(["CPU"])
        e = ob.OpenVINOEngine(args(), "/nowhere", lambda m: None)
        segs = e.decode_range(np.zeros(16000 * 60, np.float32), 20.0, 30.0, "en")
        self.assertEqual(segs[0]["start"], 21.0)
        self.assertEqual(Pipe.generated[-1][0], 160000)

    def test_blocks(self):
        audio = np.ones(16000 * 600, np.float32)
        audio[16000 * 230:16000 * 230 + 1600] = 0  # a silence in the last 15 s of the first block
        parts = list(ob.blocks(audio))
        self.assertEqual(parts[0][0], 0)
        self.assertTrue(230 * 16000 <= parts[0][1] <= 230 * 16000 + 1600)
        self.assertEqual(parts[-1][1], len(audio))
        self.assertTrue(all(a[1] == b[0] for a, b in zip(parts, parts[1:])))

    def test_repo_names(self):
        self.assertEqual(ob.repo_for("medium"), "OpenVINO/whisper-medium-int8-ov")
        self.assertEqual(ob.repo_for("distil-large-v3"), "OpenVINO/distil-whisper-large-v3-int8-ov")


if __name__ == "__main__":
    unittest.main()
