#
#  UWFaster-Whisper
#
#  Speaker diarization (who speaks in each segment), without a Hugging Face
#  token and without new libraries: onnxruntime (already used by the VAD)
#  runs a speaker embedding model, one voice "fingerprint" for each
#  segment, and the fingerprints are grouped into speakers.
#
#  The model is downloaded once from the sherpa-onnx releases (GitHub) to
#  <model_dir>/diarization: 3D-Speaker CAM++ trained with VoxCeleb (Apache
#  2.0, about 28 MB). The features (kaldi fbank, 80 mel bins) are computed
#  here with numpy, they give the same numbers as kaldi-native-fbank.
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

import numpy as np

SAMPLE_RATE = 16000

MODEL_FILE = "3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx"
MODEL_URL = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/"
             "speaker-recongition-models/" + MODEL_FILE)  # (sic, the tag name)


# -----------------------------------------------------------------------------
# model
# -----------------------------------------------------------------------------

def model_path(folder, download=True, log=print):
    """The embedding model in <folder>, downloaded if it's not there."""
    path = os.path.join(folder, MODEL_FILE)
    if os.path.isfile(path) or not download:
        return path
    os.makedirs(folder, exist_ok=True)
    log("Downloading the speaker model (about 28 MB): %s" % MODEL_URL)
    import urllib.request
    tmp = path + ".part"
    with urllib.request.urlopen(MODEL_URL, timeout=60) as r, open(tmp, "wb") as f:
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b)
    os.replace(tmp, path)
    return path


# -----------------------------------------------------------------------------
# features: kaldi fbank (povey window, 25 ms / 10 ms, preemphasis 0.97, mel
# 20-7600 Hz, without snip edges), as kaldi-native-fbank computes it
# -----------------------------------------------------------------------------

_MEL_CACHE = {}


def _mel_banks(nmel, nfft, sr, low=20.0, high=-400.0):
    key = (nmel, nfft, sr, low, high)
    if key in _MEL_CACHE:
        return _MEL_CACHE[key]
    if high <= 0:
        high += sr / 2

    def mel(f):
        return 1127.0 * np.log(1.0 + f / 700.0)
    pts = np.linspace(mel(low), mel(high), nmel + 2)
    fm = mel(np.arange(nfft // 2 + 1) * sr / nfft)
    w = np.zeros((nmel, nfft // 2 + 1))
    for m in range(nmel):
        left, center, right = pts[m], pts[m + 1], pts[m + 2]
        w[m] = np.maximum(0.0, np.minimum((fm - left) / (center - left), (right - fm) / (right - center)))
    w[:, -1] = 0.0  # kaldi doesn't use the nyquist bin
    _MEL_CACHE[key] = w.T.astype(np.float32)
    return _MEL_CACHE[key]


def fbank(x, sr=SAMPLE_RATE, nmel=80):
    flen, shift, nfft = int(sr * 0.025), int(sr * 0.010), 512
    n = (len(x) + shift // 2) // shift
    if n <= 0:
        return np.zeros((0, nmel), np.float32)
    # the frames are centered (snip_edges false): the edges are mirrored
    pad = np.concatenate([x[:flen // 2][::-1], x, x[::-1][:flen]])
    off = (shift - flen) // 2 + flen // 2
    idx = np.arange(flen)[None, :] + shift * np.arange(n)[:, None] + off
    frames = pad[idx].astype(np.float64)
    frames -= frames.mean(axis=1, keepdims=True)
    frames[:, 1:] -= 0.97 * frames[:, :-1].copy()
    frames[:, 0] -= 0.97 * frames[:, 0]
    win = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(flen) / (flen - 1))) ** 0.85
    power = np.abs(np.fft.rfft(frames * win, nfft)) ** 2
    feats = np.log(np.maximum(power.astype(np.float32) @ _mel_banks(nmel, nfft, sr),
                              np.finfo(np.float32).eps))
    return feats.astype(np.float32)


# -----------------------------------------------------------------------------
# embeddings
# -----------------------------------------------------------------------------

class Embedder:
    def __init__(self, path, threads=0):
        import onnxruntime as ort
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        so.log_severity_level = 3
        self.session = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name
        meta = self.session.get_modelmeta().custom_metadata_map
        self.global_mean = meta.get("feature_normalize_type", "global-mean") == "global-mean"

    def __call__(self, samples):
        f = fbank(samples)
        if len(f) < 10:
            return None
        if self.global_mean:
            f = f - f.mean(axis=0)
        e = self.session.run(None, {self.input: f[None]})[0][0].astype(np.float64)
        norm = np.linalg.norm(e)
        return e / norm if norm > 0 else None


# -----------------------------------------------------------------------------
# clustering: agglomerative, average linkage on the cosine similarity. First
# the embeddings that are almost the same are joined (a few hundred groups at
# most, so a long film is quick), then the groups.
# -----------------------------------------------------------------------------

def cluster(emb, weights, threshold=0.5, num_speakers=None, max_speakers=None):
    n = len(emb)
    if n == 0:
        return []
    if n == 1:
        return [0]

    # 1. pre-groups (leader clustering with a high similarity)
    labels = np.full(n, -1)
    centers, sums, counts = [], [], []
    for i in np.argsort(-np.asarray(weights)):
        if centers:
            sims = np.array(centers) @ emb[i]
            k = int(np.argmax(sims))
            if sims[k] >= 0.75:
                labels[i] = k
                sums[k] += emb[i] * weights[i]
                counts[k] += weights[i]
                c = sums[k] / np.linalg.norm(sums[k])
                centers[k] = c
                continue
        labels[i] = len(centers)
        centers.append(emb[i].copy())
        sums.append(emb[i] * weights[i])
        counts.append(weights[i])

    # 2. average linkage between the groups (weighted by the speech duration):
    # the average of the cosine similarities is the dot of the weighted sums
    groups = [[i for i in range(n) if labels[i] == k] for k in range(len(centers))]
    gw = np.array([float(sum(weights[i] for i in g)) for g in groups])
    gm = np.array([np.sum([emb[i] * weights[i] for i in g], axis=0) for g in groups])
    alive = np.ones(len(groups), dtype=bool)
    sim = (gm @ gm.T) / np.outer(gw, gw)
    np.fill_diagonal(sim, -np.inf)

    target = num_speakers if num_speakers else 1
    while alive.sum() > target:
        k = int(np.argmax(sim))
        a, b = divmod(k, len(groups))
        best = sim[a, b]
        if not num_speakers and best < threshold and not (max_speakers and alive.sum() > max_speakers):
            break
        groups[a] += groups[b]
        gw[a] += gw[b]
        gm[a] = gm[a] + gm[b]
        alive[b] = False
        row = (gm @ gm[a]) / (gw * gw[a])
        row[~alive] = -np.inf
        row[a] = -np.inf
        sim[a, :] = row
        sim[:, a] = row
        sim[b, :] = -np.inf
        sim[:, b] = -np.inf
    alive = [g for g in range(len(groups)) if alive[g]]

    # the speaker who speaks first is speaker 1
    out = np.zeros(n, dtype=int)
    order = sorted(alive, key=lambda g: min(groups[g]))
    for k, g in enumerate(order):
        for i in groups[g]:
            out[i] = k
    return out.tolist()


# -----------------------------------------------------------------------------
# segments -> speakers
# -----------------------------------------------------------------------------

def diarize(audio, segments, model_folder, num_speakers=None, max_speakers=None,
            threshold=0.5, threads=0, download=True, log=print):
    """Adds "speaker" (1, 2...) to each segment. Returns the number of speakers."""
    if not segments:
        return 0
    emb_model = Embedder(model_path(model_folder, download, log), threads)

    embs, weights, idx = [], [], []
    for k, s in enumerate(segments):
        a, b = s["start"], s["end"]
        # the words have better limits than the segment, if there are
        if s.get("words"):
            a, b = s["words"][0]["start"], s["words"][-1]["end"]
        if b - a < 0.5:
            # very short: a bit of what's around it
            c = (a + b) / 2
            a, b = c - 0.4, c + 0.4
        a, b = max(0, int(a * SAMPLE_RATE)), min(len(audio), int(b * SAMPLE_RATE))
        # long segments: up to 20 s from the middle is enough for the voice
        if b - a > 20 * SAMPLE_RATE:
            c = (a + b) // 2
            a, b = c - 10 * SAMPLE_RATE, c + 10 * SAMPLE_RATE
        e = emb_model(audio[a:b]) if b > a else None
        if e is not None:
            embs.append(e)
            weights.append(min(10.0, (b - a) / SAMPLE_RATE))
            idx.append(k)

    if not embs:
        return 0
    embs = np.array(embs)
    weights = np.array(weights)

    # the short segments (< 1 s) have unreliable voices: the speakers are
    # found with the others and the short ones go to the closest speaker
    long_ = weights >= 1.0
    if long_.sum() >= 2:
        lab_long = cluster(embs[long_], weights[long_], threshold, num_speakers, max_speakers)
        nspk = max(lab_long) + 1
        cent = np.zeros((nspk, embs.shape[1]))
        for e, w, l in zip(embs[long_], weights[long_], lab_long):
            cent[l] += e * w
        cent /= np.linalg.norm(cent, axis=1, keepdims=True)
        labels = np.argmax(embs @ cent.T, axis=1)
        it = iter(lab_long)
        labels = [next(it) if lg else int(l) for lg, l in zip(long_, labels)]
    else:
        labels = cluster(embs, weights, threshold, num_speakers, max_speakers)

    # numbered by order of appearance
    order = {}
    for l in labels:
        if l not in order:
            order[l] = len(order) + 1
    for k, l in zip(idx, labels):
        segments[k]["speaker"] = order[l]

    # the ones without a voice (too short, silence): the speaker before them
    last = None
    for s in segments:
        if "speaker" in s:
            last = s["speaker"]
        elif last is not None:
            s["speaker"] = last
    first = next((s["speaker"] for s in segments if "speaker" in s), 1)
    for s in segments:
        s.setdefault("speaker", first)
    return len(order)
