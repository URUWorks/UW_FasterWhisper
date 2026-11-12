#!/usr/bin/env python3
#
#  UWFaster-Whisper
#
#  A standalone faster-whisper command line for Tero Subtitler, compatible
#  with the parameters of the Purfview's whisper-faster that Tero uses.
#
#  The contents of this file are used with permission, subject to
#  the Mozilla Public License Version 2.0 (the "License"); you may
#  not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  http://www.mozilla.org/MPL/2.0.html
#
#  Copyright (C) 2026 URUWorks, uruworks@gmail.com.
#

import argparse
import json
import os
import re
import sys
import time

__version__ = "1.0.2"

PROG = "whisper-faster"

# -----------------------------------------------------------------------------
# output: utf-8, line by line (Tero reads the "[from --> to] text" lines
# while transcribing)
# -----------------------------------------------------------------------------

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:
        pass


def log(msg):
    print(msg, file=sys.stderr, flush=True)


# -----------------------------------------------------------------------------
# GPU libraries (cuBLAS/cuDNN) next to the executable, the optional download
# -----------------------------------------------------------------------------

def app_dir():
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def add_library_folders():
    folders = [app_dir(), os.path.join(app_dir(), "cuda")]
    for d in folders:
        if not os.path.isdir(d):
            continue
        if sys.platform == "win32":
            try:
                os.add_dll_directory(d)
            except Exception:
                pass
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
        else:
            key = "LD_LIBRARY_PATH"
            os.environ[key] = d + os.pathsep + os.environ.get(key, "")


# -----------------------------------------------------------------------------
# audio: Tero gives a wav (extracted with ffmpeg), read here so the build
# doesn't need PyAV and its ffmpeg libraries (about 130 MB). Other formats
# need PyAV installed (running from the sources).
# -----------------------------------------------------------------------------

def stub_av_if_missing():
    try:
        import av  # noqa: F401
    except ImportError:
        import types
        av = types.ModuleType("av")

        def _missing(*a, **k):
            raise RuntimeError("only wav files are supported in this build (convert it with ffmpeg)")
        av.open = _missing
        av.audio = types.SimpleNamespace(resampler=types.SimpleNamespace(AudioResampler=_missing),
                                         fifo=types.SimpleNamespace(AudioFifo=_missing))
        av.error = types.SimpleNamespace(InvalidDataError=Exception)
        sys.modules["av"] = av


def read_wav(path, sample_rate=16000):
    """A PCM wav -> mono float32 at 16 kHz (numpy), None if it's not a PCM wav."""
    import wave
    import numpy as np

    try:
        with wave.open(path, "rb") as w:
            channels, width, rate, frames = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
            raw = w.readframes(frames)
    except (wave.Error, EOFError):
        return None

    if width == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        i = (b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8) | (b[:, 2].astype(np.int32) << 16))
        i = np.where(i >= 1 << 23, i - (1 << 24), i)
        data = i.astype(np.float32) / float(1 << 23)
    elif width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / float(1 << 31)
    else:
        return None

    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    if rate != sample_rate and len(data) > 0:
        data = resample(data, rate, sample_rate)
    return np.ascontiguousarray(data, dtype=np.float32)


def resample(data, rate, new_rate):
    """Low-pass (windowed sinc, against aliasing) and linear interpolation."""
    import numpy as np

    if new_rate < rate:
        taps = 127
        cutoff = 0.45 * new_rate / rate  # of the input rate, a bit under nyquist
        n = np.arange(taps) - (taps - 1) / 2
        h = (2 * cutoff * np.sinc(2 * cutoff * n) * np.blackman(taps)).astype(np.float32)
        h /= h.sum()
        # FFT convolution by blocks (fast, and a long file doesn't need all
        # the memory at once)
        nfft = 1 << 18
        step = nfft - taps + 1
        H = np.fft.rfft(h, nfft)
        out = np.empty_like(data)
        pad = np.concatenate([np.zeros(taps // 2, np.float32), data, np.zeros(taps, np.float32)])
        for i in range(0, len(data), step):
            seg = pad[i:i + nfft]
            y = np.fft.irfft(np.fft.rfft(seg, nfft) * H, nfft)[taps - 1:taps - 1 + step]
            j = min(i + step, len(data))
            out[i:j] = y[:j - i]
        data = out

    n = int(round(len(data) * new_rate / rate))
    return np.interp(np.arange(n) * (rate / new_rate), np.arange(len(data)), data).astype(np.float32)


# -----------------------------------------------------------------------------
# arguments
# -----------------------------------------------------------------------------

def str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("true", "1", "yes", "y", "on"):
        return True
    if s in ("false", "0", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError("expected True or False, got %r" % v)


def optional_float(v):
    return None if str(v).lower() == "none" else float(v)


def optional_int(v):
    return None if str(v).lower() == "none" else int(v)


OUTPUT_FORMATS = ["txt", "vtt", "srt", "tsv", "json", "all"]


def build_parser():
    p = argparse.ArgumentParser(
        prog=PROG,
        description="UWFaster-Whisper %s, speech recognition with faster-whisper." % __version__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument("audio", nargs="*", help="audio or video file(s) to transcribe")
    p.add_argument("--version", action="version", version="%s %s" % (PROG, __version__))

    g = p.add_argument_group("model")
    g.add_argument("--model", "-m", default="medium", help="model name (tiny, base, small, medium, large-v3...) or a model folder")
    g.add_argument("--model_dir", default=None, help="folder of the models, with a 'faster-whisper-<model>' folder for each one")
    g.add_argument("--device", "-d", default="auto", help="auto, cpu or cuda")
    g.add_argument("--device_index", type=int, default=0, help="GPU number")
    g.add_argument("--compute_type", "-ct", default="auto", help="auto, int8, int8_float16, float16, float32...")
    g.add_argument("--threads", type=int, default=0, help="CPU threads, 0 = automatic")
    g.add_argument("--local_files_only", type=str2bool, default=False, help="don't download the model")

    g = p.add_argument_group("output")
    g.add_argument("--output_dir", "-o", default=".", help="folder for the output files")
    g.add_argument("--output_format", "-f", default="srt", choices=OUTPUT_FORMATS, help="output format")
    g.add_argument("--verbose", type=str2bool, default=True, help="print each segment while transcribing")
    g.add_argument("--sentence", action="store_true", help="one subtitle per sentence (uses the word times)")
    g.add_argument("--max_line_width", type=optional_int, default=None, help="characters per line (srt/vtt)")
    g.add_argument("--max_line_count", type=optional_int, default=None, help="lines per subtitle (srt/vtt)")
    g.add_argument("--max_comma", type=optional_int, default=None, help="with --sentence, split at a comma after this many characters")
    g.add_argument("--beep_off", action="store_true", help="accepted for compatibility, there is no beep")

    g = p.add_argument_group("transcription")
    g.add_argument("--task", default="transcribe", choices=["transcribe", "translate"], help="translate = to english")
    g.add_argument("--language", "-l", default=None, help="language code (en, es...), detected if not given")
    g.add_argument("--word_timestamps", type=str2bool, default=False, help="times of each word (json)")
    g.add_argument("--beam_size", type=int, default=5)
    g.add_argument("--best_of", type=int, default=5)
    g.add_argument("--patience", type=float, default=1.0)
    g.add_argument("--length_penalty", type=float, default=1.0)
    g.add_argument("--repetition_penalty", type=float, default=1.0)
    g.add_argument("--no_repeat_ngram_size", type=int, default=0)
    g.add_argument("--temperature", type=float, default=0.0)
    g.add_argument("--temperature_increment_on_fallback", type=optional_float, default=0.2)
    g.add_argument("--compression_ratio_threshold", type=optional_float, default=2.4)
    g.add_argument("--logprob_threshold", type=optional_float, default=-1.0)
    g.add_argument("--no_speech_threshold", type=optional_float, default=0.6)
    g.add_argument("--condition_on_previous_text", type=str2bool, default=True)
    g.add_argument("--initial_prompt", default=None)
    g.add_argument("--hotwords", default=None)
    g.add_argument("--suppress_blank", type=str2bool, default=True)
    g.add_argument("--suppress_tokens", default="-1")
    g.add_argument("--hallucination_silence_threshold", type=optional_float, default=None)
    g.add_argument("--chunk_length", type=optional_int, default=None)
    g.add_argument("--clip_timestamps", default="0")

    g = p.add_argument_group("voice activity detection")
    g.add_argument("--vad_filter", type=str2bool, default=False, help="skip the parts without voice")
    g.add_argument("--vad_threshold", type=float, default=None)
    g.add_argument("--vad_min_speech_duration_ms", type=int, default=None)
    g.add_argument("--vad_max_speech_duration_s", type=float, default=None)
    g.add_argument("--vad_min_silence_duration_ms", type=int, default=None)
    g.add_argument("--vad_speech_pad_ms", type=int, default=None)

    return p


# -----------------------------------------------------------------------------
# times
# -----------------------------------------------------------------------------

def ts(seconds, sep=",", always_hours=True):
    ms = max(0, int(round(seconds * 1000)))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    if always_hours or h > 0:
        return "%02d:%02d:%02d%s%03d" % (h, m, s, sep, ms)
    return "%02d:%02d%s%03d" % (m, s, sep, ms)


# -----------------------------------------------------------------------------
# subtitles: lines and sentences
# -----------------------------------------------------------------------------

SENTENCE_END = re.compile(r"[.!?…。！？]['\"”’)\]]*$")


def wrap_words(words, width):
    """Word list -> lines of at most width characters (a longer word alone)."""
    lines, cur = [], ""
    for w in words:
        w = w.strip()
        if not w:
            continue
        cand = w if not cur else cur + " " + w
        if width and len(cand) > width and cur:
            lines.append(cur)
            cur = w
        else:
            cur = cand
    if cur:
        lines.append(cur)
    return lines


def balance_lines(text, width, count):
    """Text in lines of width characters, balanced (similar lengths)."""
    words = text.split()
    if not width:
        return [" ".join(words)]
    lines = wrap_words(words, width)
    if count and len(lines) == count and count > 1:
        # same number of lines, as even as possible
        total = len(" ".join(words))
        target = total / count
        best = lines
        best_diff = max(len(l) for l in lines)
        for w in range(int(target), width + 1):
            cand = wrap_words(words, w)
            if len(cand) <= count and max(len(l) for l in cand) < best_diff:
                best, best_diff = cand, max(len(l) for l in cand)
        lines = best
    return lines


def chunk_words(words, width, count, sentence, max_comma):
    """Timed words [(start, end, text)] -> subtitles [(start, end, [words])]."""
    subs, cur = [], []

    def fits(ws):
        if not width:
            return True
        return len(wrap_words([w[2] for w in ws], width)) <= count

    for w in words:
        if not w[2].strip():
            continue
        cand = cur + [w]
        if cur and width and count and not fits(cand):
            subs.append(cur)
            cur = [w]
        else:
            cur = cand
        text = " ".join(x[2].strip() for x in cur)
        if sentence and SENTENCE_END.search(w[2].strip()):
            subs.append(cur)
            cur = []
        elif sentence and max_comma and w[2].strip().endswith(",") and len(text) >= max_comma:
            subs.append(cur)
            cur = []
    if cur:
        subs.append(cur)
    return [(s[0][0], s[-1][1], [x[2].strip() for x in s]) for s in subs]


def make_subtitles(segments, args):
    """Segments -> [(start, end, text with \\n)] for srt/vtt."""
    width, count = args.max_line_width, args.max_line_count
    has_words = all(s.get("words") for s in segments) and segments
    result = []

    if has_words and (args.sentence or (width and count)):
        words = [(w["start"], w["end"], w["word"]) for s in segments for w in s["words"]]
        for start, end, ws in chunk_words(words, width, count, args.sentence, args.max_comma):
            text = " ".join(ws)
            lines = balance_lines(text, width, count) if width else [text]
            result.append((start, end, "\n".join(lines)))
        return result

    for s in segments:
        text = s["text"].strip()
        if not text:
            continue
        if width:
            lines = balance_lines(text, width, count)
            if count and len(lines) > count:
                # more lines than allowed: split the time by characters
                groups = [lines[i:i + count] for i in range(0, len(lines), count)]
                total = sum(len(" ".join(g)) for g in groups)
                t = s["start"]
                for g in groups:
                    d = (s["end"] - s["start"]) * len(" ".join(g)) / max(1, total)
                    result.append((t, t + d, "\n".join(g)))
                    t += d
                continue
            text = "\n".join(lines)
        result.append((s["start"], s["end"], text))
    return result


# -----------------------------------------------------------------------------
# writers
# -----------------------------------------------------------------------------

def write_srt(path, subs):
    with open(path, "w", encoding="utf-8") as f:
        for i, (a, b, t) in enumerate(subs, 1):
            f.write("%d\n%s --> %s\n%s\n\n" % (i, ts(a), ts(b), t))


def write_vtt(path, subs):
    with open(path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for a, b, t in subs:
            f.write("%s --> %s\n%s\n\n" % (ts(a, "."), ts(b, "."), t))


def write_txt(path, segments):
    with open(path, "w", encoding="utf-8") as f:
        for s in segments:
            f.write(s["text"].strip() + "\n")


def write_tsv(path, segments):
    with open(path, "w", encoding="utf-8") as f:
        f.write("start\tend\ttext\n")
        for s in segments:
            f.write("%d\t%d\t%s\n" % (round(s["start"] * 1000), round(s["end"] * 1000),
                                      s["text"].strip().replace("\t", " ")))


def write_json(path, segments, language):
    # the openai-whisper json: {"text", "segments": [{"start", "end", "text",
    # "words": [{"start", "end", "word", "probability"}]}], "language"}
    data = {
        "text": "".join(s["text"] for s in segments),
        "segments": segments,
        "language": language,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def write_outputs(audio, segments, language, args):
    os.makedirs(args.output_dir, exist_ok=True)
    base = os.path.join(args.output_dir, os.path.splitext(os.path.basename(audio))[0])
    fmts = ["txt", "vtt", "srt", "tsv", "json"] if args.output_format == "all" else [args.output_format]
    for fmt in fmts:
        path = base + "." + fmt
        if fmt == "srt":
            write_srt(path, make_subtitles(segments, args))
        elif fmt == "vtt":
            write_vtt(path, make_subtitles(segments, args))
        elif fmt == "txt":
            write_txt(path, segments)
        elif fmt == "tsv":
            write_tsv(path, segments)
        elif fmt == "json":
            write_json(path, segments, language)
        log("Subtitles are written to '%s'" % os.path.abspath(path))


# -----------------------------------------------------------------------------
# model
# -----------------------------------------------------------------------------

def resolve_model(args):
    """A model folder (Tero: <model_dir>/faster-whisper-<model>) or the name to download."""
    m = args.model
    if os.path.isdir(m):
        return m, None
    if args.model_dir:
        for name in ("faster-whisper-" + m, "faster-distil-whisper-" + m.replace("distil-", ""), m):
            d = os.path.join(args.model_dir, name)
            if os.path.isfile(os.path.join(d, "model.bin")):
                return d, None
    # not found, faster-whisper downloads it to model_dir (or its cache)
    return m, args.model_dir


# GPU libraries that CTranslate2 4.x loads (CUDA 12, cuDNN 9). cuDNN loads its
# parts only when they're used and, if one is missing, the process ends without
# an exception (Tero gets no text), so they are looked for before using the GPU.
CUDA_LIBRARIES = {
    "win32": ["cublas64_12.dll", "cublasLt64_12.dll", "cudnn64_9.dll", "cudnn_ops64_9.dll",
              "cudnn_cnn64_9.dll"],
    "linux": ["libcublas.so.12", "libcublasLt.so.12", "libcudnn.so.9", "libcudnn_ops.so.9",
              "libcudnn_cnn.so.9"],
}


def missing_cuda_libraries():
    import ctypes

    names = CUDA_LIBRARIES.get("linux" if sys.platform.startswith("linux") else sys.platform, [])
    missing = []
    for name in names:
        try:
            if sys.platform == "win32":
                # winmode=0: the usual search, PATH included (the CUDA installation)
                ctypes.WinDLL(name, winmode=0)
            else:
                ctypes.CDLL(name)
        except OSError:
            missing.append(name)
    return missing


def cpu_without_fast_int8():
    """An Intel Mac without AVX2 (Mac Pro 2009-2012...): the int8 of the macOS
    10.11 build (Ruy) has no fast code there, float32 (Accelerate) is faster."""
    if sys.platform != "darwin":
        return False
    import platform
    import subprocess
    if platform.machine() != "x86_64":
        return False
    try:
        out = subprocess.run(["sysctl", "-n", "hw.optional.avx2_0"], capture_output=True, text=True)
        return out.stdout.strip() != "1"
    except Exception:
        return False


def check_gpu(model):
    """Encodes 30 s of silence: a GPU that CTranslate2 can't use fails here, not halfway."""
    import numpy as np

    features = model.feature_extractor(np.zeros(16000 * 30, dtype=np.float32))
    features = features[:, :model.feature_extractor.nb_max_frames]
    model.encode(features)


def load_model(args):
    from faster_whisper import WhisperModel
    import ctranslate2

    path, download_root = resolve_model(args)

    device = args.device
    if device == "auto":
        try:
            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:
            device = "cpu"
    if device == "cuda":
        missing = missing_cuda_libraries()
        if missing:
            log("GPU libraries not found (%s), using the CPU. For the GPU, extract "
                "fasterwhisper_cuda_win64.zip next to whisper-faster." % ", ".join(missing))
            device = "cpu"

    def compute_for(dev):
        if args.compute_type != "auto":
            return args.compute_type
        if dev == "cuda":
            return "float16"
        return "float32" if cpu_without_fast_int8() else "int8"

    def create(dev):
        return WhisperModel(path, device=dev, device_index=args.device_index,
                            compute_type=compute_for(dev), cpu_threads=args.threads,
                            download_root=download_root, local_files_only=args.local_files_only)

    try:
        model = create(device)
        if device == "cuda":
            check_gpu(model)
    except Exception as e:
        if device != "cuda":
            raise
        # an old GPU, not enough GPU memory, a float16 the GPU doesn't have...: on the CPU
        log("GPU not available (%s), using the CPU." % e)
        model = None  # frees the GPU memory
        device = "cpu"
        model = create(device)

    log("Model: %s, device: %s, compute type: %s" % (path, device, compute_for(device)))
    return model, device


# -----------------------------------------------------------------------------
# transcription
# -----------------------------------------------------------------------------

def transcribe_options(args):
    if args.temperature_increment_on_fallback:
        temps, t = [], args.temperature
        while t <= 1.0 + 1e-6:
            temps.append(round(t, 2))
            t += args.temperature_increment_on_fallback
    else:
        temps = [args.temperature]

    vad = {}
    for k in ("threshold", "min_speech_duration_ms", "max_speech_duration_s",
              "min_silence_duration_ms", "speech_pad_ms"):
        v = getattr(args, "vad_" + k)
        if v is not None:
            vad[k] = v

    suppress = [int(x) for x in str(args.suppress_tokens).split(",") if x.strip()]

    opts = dict(
        language=args.language, task=args.task, beam_size=args.beam_size, best_of=args.best_of,
        patience=args.patience, length_penalty=args.length_penalty,
        repetition_penalty=args.repetition_penalty, no_repeat_ngram_size=args.no_repeat_ngram_size,
        temperature=temps, compression_ratio_threshold=args.compression_ratio_threshold,
        log_prob_threshold=args.logprob_threshold, no_speech_threshold=args.no_speech_threshold,
        condition_on_previous_text=args.condition_on_previous_text,
        initial_prompt=args.initial_prompt, hotwords=args.hotwords,
        suppress_blank=args.suppress_blank, suppress_tokens=suppress,
        word_timestamps=args.word_timestamps or args.sentence or bool(args.max_line_width and args.max_line_count),
        hallucination_silence_threshold=args.hallucination_silence_threshold,
        chunk_length=args.chunk_length, clip_timestamps=args.clip_timestamps,
        vad_filter=args.vad_filter, vad_parameters=vad or None,
    )
    return opts


def segment_to_dict(s):
    d = {
        "id": s.id, "seek": s.seek, "start": round(s.start, 3), "end": round(s.end, 3),
        "text": s.text, "tokens": list(s.tokens), "temperature": s.temperature,
        "avg_logprob": s.avg_logprob, "compression_ratio": s.compression_ratio,
        "no_speech_prob": s.no_speech_prob,
    }
    if s.words:
        d["words"] = [{"start": round(w.start, 3), "end": round(w.end, 3), "word": w.word,
                       "probability": round(w.probability, 4)} for w in s.words]
    return d


def vad_available():
    try:
        import onnxruntime  # noqa: F401
        return True
    except ImportError:
        return False


def transcribe_file(model, audio, args, progress=None):
    log("Transcribing: %s" % audio)
    if args.vad_filter and not vad_available():
        # the macOS 10.11 build has no onnxruntime (there is none for it)
        log("The VAD filter is not available in this build, transcribing without it.")
        args.vad_filter = False
    start = time.time()
    data = read_wav(audio) if audio.lower().endswith(".wav") else None
    segments, info = model.transcribe(audio if data is None else data, **transcribe_options(args))
    if args.language is None:
        log("Detected language '%s' with probability %.2f" % (info.language, info.language_probability))

    result = []
    for s in segments:
        d = segment_to_dict(s)
        result.append(d)
        if progress is not None:
            progress.append(d)
        if args.verbose:
            # mm:ss.zzz until one hour, then hh:mm:ss.zzz (as Tero reads it)
            print("[%s --> %s] %s" % (ts(s.start, ".", False), ts(s.end, ".", False), s.text.strip()), flush=True)

    write_outputs(audio, result, info.language, args)
    log("Operation finished in: %.1f s" % (time.time() - start))


def main(argv=None):
    add_library_folders()
    stub_av_if_missing()
    parser = build_parser()
    # unknown parameters of other versions don't stop the transcription
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        log("Ignored parameters: %s" % " ".join(unknown))
    if not args.audio:
        parser.print_help()
        return 1

    try:
        model, device = load_model(args)
    except Exception as e:
        log("Error loading the model '%s': %s" % (args.model, e))
        return 2

    code = 0
    for audio in args.audio:
        if not os.path.isfile(audio):
            log("File not found: %s" % audio)
            code = 3
            continue
        done = []
        try:
            transcribe_file(model, audio, args, done)
        except Exception as e:
            if device == "cuda" and not done:
                # the GPU failed before the first segment: again on the CPU
                log("GPU error (%s), transcribing on the CPU." % e)
                try:
                    args.device = "cpu"
                    model, device = load_model(args)
                    transcribe_file(model, audio, args)
                    continue
                except Exception as e2:
                    e = e2
            log("Error transcribing '%s': %s" % (audio, e))
            code = 4
    return code


if __name__ == "__main__":
    sys.exit(main())
