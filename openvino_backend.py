#
#  UWFaster-Whisper
#
#  OpenVINO GenAI backend (--device openvino): Whisper on Intel GPUs (Arc,
#  Iris Xe, the integrated ones), NPUs and CPUs, with the WhisperPipeline of
#  openvino-genai. faster-whisper (CTranslate2) can't use these GPUs.
#
#  openvino-genai is not in the main build (it's about 200 MB): it's the
#  optional fasterwhisper_openvino_<system>.zip, extracted next to the
#  executable (an "openvino" folder), as the CUDA libraries. Running from
#  the sources, "pip install openvino-genai" is enough.
#
#  The models are OpenVINO IR folders (openvino_encoder_model.xml...), not
#  the CTranslate2 ones: <model_dir>/whisper-<model>-int8-ov, downloaded from
#  the OpenVINO organization of Hugging Face if they're not there.
#
#  The contents of this file are used with permission, subject to
#  the Mozilla Public License Version 2.0 (the "License"); you may
#  not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  http://www.mozilla.org/MPL/2.0.html
#
#  Copyright (C) 2026 URUWorks, uruworks@gmail.com.
#

import os
import sys

import numpy as np

SAMPLE_RATE = 16000
BLOCK_SECONDS = 240  # the audio is given by blocks, so the segments come out while transcribing


def add_extras_folder(app_dir):
    """The "openvino" folder of the optional zip, next to the executable."""
    d = os.path.join(app_dir, "openvino")
    if os.path.isdir(d) and d not in sys.path:
        sys.path.insert(0, d)
        libs = os.path.join(d, "openvino", "libs")
        if sys.platform == "win32" and os.path.isdir(libs):
            try:
                os.add_dll_directory(libs)
            except Exception:
                pass
            os.environ["PATH"] = libs + os.pathsep + os.environ.get("PATH", "")


def available(app_dir):
    add_extras_folder(app_dir)
    try:
        import openvino_genai  # noqa: F401
        return True
    except Exception:
        return False


def repo_for(model):
    """Model name -> the Hugging Face repository of its OpenVINO version."""
    m = model.lower()
    if m in ("turbo", "large-v3-turbo"):
        m = "large-v3-turbo"
    if m.startswith("distil-"):
        return "OpenVINO/distil-whisper-%s-int8-ov" % m[len("distil-"):]
    return "OpenVINO/whisper-%s-int8-ov" % m


def is_ov_folder(d):
    return os.path.isfile(os.path.join(d, "openvino_encoder_model.xml"))


def resolve_model(model, model_dir, local_files_only, log):
    if os.path.isdir(model):
        if is_ov_folder(model):
            return model
        raise RuntimeError("'%s' is not an OpenVINO model (no openvino_encoder_model.xml)" % model)
    repo = repo_for(model)
    name = repo.split("/")[1]
    if model_dir:
        for d in (name, "openvino-whisper-" + model, "openvino-" + model):
            p = os.path.join(model_dir, d)
            if is_ov_folder(p):
                return p
    if local_files_only:
        raise RuntimeError("OpenVINO model '%s' not found in '%s'" % (name, model_dir))
    from huggingface_hub import snapshot_download
    target = os.path.join(model_dir, name) if model_dir else None
    log("Downloading the OpenVINO model %s" % repo)
    return snapshot_download(repo, local_dir=target)


def pick_device(wanted):
    """openvino -> GPU if there is one, else CPU. openvino:NPU, openvino:GPU.1..."""
    import openvino as ov
    devices = ov.Core().available_devices
    if wanted:
        return wanted.upper(), devices
    for d in devices:
        if d.startswith("GPU"):
            return d, devices
    return "CPU", devices


class OpenVINOEngine:
    name = "openvino"

    def __init__(self, args, app_dir, log):
        add_extras_folder(app_dir)
        import openvino_genai as ov_genai

        self.args = args
        self.log = log
        self.path = resolve_model(args.model, args.model_dir, args.local_files_only, log)
        wanted = args.device.split(":", 1)[1] if ":" in args.device else None
        device, devices = pick_device(wanted)
        self.words = bool(args.word_timestamps or args.sentence or (args.max_line_width and args.max_line_count))
        kwargs = {"word_timestamps": True} if self.words else {}
        try:
            self.pipe = ov_genai.WhisperPipeline(self.path, device, **kwargs)
        except Exception as e:
            if device == "CPU":
                raise
            log("OpenVINO %s not available (%s), using the CPU." % (device, e))
            device = "CPU"
            self.pipe = ov_genai.WhisperPipeline(self.path, device, **kwargs)
        self.device = device
        log("Model: %s, device: openvino %s (devices: %s)" % (self.path, device, ", ".join(devices)))

    # -- generation ------------------------------------------------------------

    def _config(self, language):
        a = self.args
        cfg = self.pipe.get_generation_config()
        cfg.task = a.task
        cfg.return_timestamps = True
        if language:
            cfg.language = "<|%s|>" % language
        if a.beam_size and a.beam_size > 1:
            cfg.num_beams = a.beam_size
        if a.initial_prompt:
            cfg.initial_prompt = a.initial_prompt
        if a.hotwords:
            cfg.hotwords = a.hotwords
        if a.repetition_penalty and a.repetition_penalty != 1.0:
            cfg.repetition_penalty = a.repetition_penalty
        if a.no_repeat_ngram_size:
            cfg.no_repeat_ngram_size = a.no_repeat_ngram_size
        if self.words:
            cfg.word_timestamps = True
        return cfg

    def _generate(self, samples, language):
        cfg = self._config(language)
        try:
            return self.pipe.generate(samples, cfg)
        except Exception:
            if getattr(cfg, "num_beams", 1) <= 1:
                raise
            cfg.num_beams = 1  # a version or device without beam search
            return self.pipe.generate(samples, cfg)

    def _segments(self, result, offset, language):
        chunks = list(getattr(result, "chunks", None) or [])
        if not chunks:
            text = result.texts[0] if result.texts else ""
            chunks = [type("C", (), {"start_ts": 0.0, "end_ts": len(text) and -1, "text": text})()]
        words = list(getattr(result, "words", None) or []) if self.words else []
        segs = []
        for c in chunks:
            text = c.text if c.text.startswith(" ") else " " + c.text
            if not text.strip():
                continue
            start = offset + max(0.0, float(c.start_ts))
            end = offset + float(c.end_ts) if c.end_ts is not None and c.end_ts >= 0 else start
            segs.append({"id": 0, "seek": 0, "start": round(start, 3), "end": round(max(end, start), 3),
                         "text": text, "tokens": list(getattr(c, "token_ids", []) or []),
                         "temperature": 0.0, "avg_logprob": None, "compression_ratio": None,
                         "no_speech_prob": None})
        if words and segs:
            k = 0
            for s in segs:
                s["words"] = []
            for w in words:
                mid = offset + (float(w.start_ts) + float(w.end_ts)) / 2
                while k + 1 < len(segs) and mid >= segs[k + 1]["start"]:
                    k += 1
                segs[k]["words"].append({"start": round(offset + float(w.start_ts), 3),
                                         "end": round(offset + float(w.end_ts), 3),
                                         "word": w.word if w.word.startswith(" ") else " " + w.word,
                                         "probability": 1.0})
        return segs

    def _language_of(self, result):
        langs = getattr(result, "languages", None) or []
        if langs:
            return str(langs[0]).strip("<|>") or None
        return None

    # -- the engine interface (see whisper_faster.py) -------------------------

    def transcribe(self, audio, args):
        """(segments iterator, language, probability). The first block is
        transcribed here: it gives the language (when it's not given)."""
        parts = list(blocks(audio))
        if not parts:
            return iter([]), args.language or "en", None
        language = args.language
        a, b = parts[0]
        result = self._generate(audio[a:b], language)
        if language is None:
            language = self._language_of(result)
        first = self._segments(result, a / SAMPLE_RATE, language)
        # multilingual: each block detects its own language
        lang = None if args.multilingual and args.language is None else language

        def gen():
            for s in first:
                yield s
            for a, b in parts[1:]:
                for s in self._segments(self._generate(audio[a:b], lang), a / SAMPLE_RATE, lang):
                    yield s
        return gen(), language or "en", None

    def decode_range(self, audio, start, end, language):
        a, b = int(start * SAMPLE_RATE), int(end * SAMPLE_RATE)
        result = self._generate(audio[a:b], language)
        return self._segments(result, start, language)


def blocks(audio, seconds=BLOCK_SECONDS):
    """(start, end) samples of blocks of about `seconds`, cut at the quietest
    moment of the last 15 s of each block (not in the middle of a word)."""
    n = len(audio)
    size = seconds * SAMPLE_RATE
    a = 0
    while a < n:
        b = a + size
        if b >= n:
            yield a, n
            return
        lo = max(a + SAMPLE_RATE, b - 15 * SAMPLE_RATE)
        win = SAMPLE_RATE // 10
        seg = audio[lo:b]
        m = len(seg) // win
        if m > 0:
            energy = (seg[:m * win].reshape(m, win) ** 2).mean(axis=1)
            b = lo + int(np.argmin(energy)) * win + win // 2
        yield a, b
        a = b
