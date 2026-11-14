#
#  UWFaster-Whisper
#
#  Fixes on the segments while they come out of Whisper:
#
#  - Loops: Whisper sometimes gets stuck repeating a word or a phrase
#    ("no, no, no, no...", the same "Thank you." again and again) and the rest
#    of that 30 s window is lost. The looped segments are collapsed to one
#    and the audio they covered, up to the next good segment, is decoded
#    again starting a bit later (several starts are tried), so the dialog
#    that was there comes back.
#
#  - Stretched segments: after a silence, a segment can start long before
#    its words are said (a line that appears 10 or 20 s too early). Only the
#    segments that last much more than their words need are looked at: their
#    own text (and the next segment's, as an anchor) is aligned with the
#    audio and, if the alignment is reliable, the start and end are moved to
#    where the words are.
#
#  The contents of this file are used with permission, subject to
#  the Mozilla Public License Version 2.0 (the "License"); you may
#  not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  http://www.mozilla.org/MPL/2.0.html
#
#  Copyright (C) 2026 URUWorks, uruworks@gmail.com.
#

import re
import zlib

SAMPLE_RATE = 16000

_WORD = re.compile(r"\w+(?:['’]\w+)*", re.UNICODE)
_CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]")


def words_of(text):
    """Normalized words of a text (CJK: one character = one word)."""
    text = text.lower()
    if _CJK.search(text):
        return [c for c in text if c.strip() and (c.isalnum() or _CJK.match(c))]
    return _WORD.findall(text)


def compression_ratio(text):
    b = text.encode("utf-8")
    return len(b) / max(1, len(zlib.compress(b)))


def repeated_unit(words, min_repeats=3):
    """(n, repeats) of the n-gram repeated back to back the most times, if it
    covers at least half of the words, else None."""
    best = None
    total = len(words)
    for n in range(1, min(8, total // min_repeats) + 1):
        i = 0
        while i + n <= total:
            unit = words[i:i + n]
            r = 1
            while words[i + r * n:i + (r + 1) * n] == unit:
                r += 1
            if r >= min_repeats and r * n * 2 >= total and (best is None or r * n > best[1] * best[0]):
                best = (n, r, i)
            i += max(1, (r - 1) * n) if r > 1 else 1
    return best


def is_looped(seg, cr_threshold=2.4):
    """A segment that is a word or phrase repeated over and over."""
    if seg.get("_checked"):
        return False
    text = seg["text"].strip()
    words = words_of(text)
    if len(words) >= 4 and repeated_unit(words) is not None:
        return True
    return len(text) >= 40 and compression_ratio(text) > cr_threshold


def same_text(a, b):
    return words_of(a["text"]) == words_of(b["text"]) and bool(words_of(a["text"]))


def collapse(seg):
    """A looped segment -> its text once ("no, no, no, no" -> "no")."""
    text = seg["text"].strip()
    words = words_of(text)
    found = repeated_unit(words) if len(words) >= 4 else None
    if found is None:
        return text
    n, _r, i = found
    # the same words in the original text (with their case and punctuation)
    tokens = text.split()
    if len(tokens) == len(words):
        unit = " ".join(tokens[i:i + n])
        prefix = " ".join(tokens[:i])
        return (prefix + " " + unit).strip()
    return " ".join(words[:i + n])


def speech_duration(text, tokens=None):
    """About how long it takes to say a text (seconds)."""
    tokens = [t for t in tokens or [] if t < 50257]  # without the special/timestamp tokens
    if tokens:
        return 0.22 * len(tokens)
    words = words_of(text)
    if words and _CJK.search(text):
        return 0.2 * len(words)
    return 0.32 * len(words)


class Fixer:
    """Takes the segments one by one (feed) and gives them back fixed, with
    the smallest delay possible: a segment waits only for the next one (the
    stretched check needs it) or, inside a loop, until the loop ends."""

    def __init__(self, engine, audio, language, loop_recovery=True, retime=True,
                 retime_threshold=0.4, cr_threshold=2.4, no_speech_threshold=0.6,
                 log=None):
        self.engine = engine
        self.audio = audio
        self.duration = len(audio) / SAMPLE_RATE if audio is not None else 0.0
        self.language = language
        self.loop_recovery = loop_recovery and audio is not None
        self.retime = retime and audio is not None and hasattr(engine, "align")
        self.retime_threshold = retime_threshold
        self.cr_threshold = cr_threshold
        self.no_speech_threshold = no_speech_threshold
        self.log = log or (lambda m: None)
        self.buf = []
        self.loops_fixed = 0
        self.retimed = 0

    # -- the buffer ----------------------------------------------------------

    def feed(self, seg):
        self.buf.append(seg)
        return self._process(final=False)

    def flush(self):
        return self._process(final=True)

    def _loop_run(self, i):
        """End (exclusive) of the loop that starts at i, None if there's none
        there. A loop: looped segments (and the same text after them), or the
        same short text 3 or more times in a row."""
        buf = self.buf
        if not self.loop_recovery or buf[i].get("_checked"):
            return None
        if is_looped(buf[i], self.cr_threshold):
            j = i + 1
            while j < len(buf) and not buf[j].get("_checked") and (
                    is_looped(buf[j], self.cr_threshold) or same_text(buf[j], buf[j - 1])):
                j += 1
            return j
        j = i + 1
        while j < len(buf) and not buf[j].get("_checked") and same_text(buf[j], buf[i]):
            j += 1
        if j - i >= 3:
            return j
        return None

    def _process(self, final):
        out = []
        while self.buf:
            buf = self.buf
            run = None
            i = 0
            while i < len(buf):
                j = self._loop_run(i)
                if j is not None:
                    # still open (it can go on with the next segment)?
                    if j == len(buf) and not final:
                        run = (i, None)
                    else:
                        run = (i, j)
                    break
                # the same text twice at the end: it can become a loop
                if (not final and i == len(buf) - 2 and self.loop_recovery
                        and not buf[i].get("_checked") and same_text(buf[i], buf[i + 1])):
                    run = (i, None)
                    break
                i += 1

            ready = i if run else len(buf)
            if not final and run is None:
                ready -= 1  # the last one waits for the next (stretched check)
            for k in range(ready):
                nxt = buf[k + 1] if k + 1 < len(buf) else None
                out.append(self._retime(buf[k], nxt))
            del buf[:ready]

            if run is None or run[1] is None:
                break
            # a loop, closed by the next segment (or the end of the audio)
            n = run[1] - run[0]
            looped = buf[:n]
            nxt = buf[n] if n < len(buf) else None
            del buf[:n]
            fixed = self._recover(looped, nxt)
            for s in fixed:
                s["_checked"] = True
            buf[0:0] = fixed
        for s in out:
            s.pop("_checked", None)
            s.pop("_retimed", None)
        return out

    # -- loops ---------------------------------------------------------------

    def _recover(self, looped, nxt):
        start = looped[0]["start"]
        end = nxt["start"] if nxt is not None else self.duration
        end = max(end, looped[-1]["end"])
        once = dict(looped[0])
        once["text"] = " " + collapse(looped[0])
        once.pop("words", None)
        once["tokens"] = []  # the loop's tokens, they're not this text anymore
        said = speech_duration(once["text"])
        once["end"] = min(looped[0]["end"], start + said + 0.5)
        self.loops_fixed += 1

        # time that the loop's text can't explain: something was lost there
        if end - start - said < 1.5:
            self.log("Loop at %s collapsed: %r" % (_t(start), once["text"].strip()))
            return [once]

        # no voice after the loop's own words (music, noise): nothing to find
        has_speech = getattr(self.engine, "speech_in_range", None)
        if has_speech is not None:
            try:
                voice = has_speech(self.audio, min(end, start + said), end)
            except Exception:
                voice = None
            if voice is not None and voice < 0.5:
                self.log("Loop at %s collapsed (no voice until %s): %r"
                         % (_t(start), _t(end), once["text"].strip()))
                return [once]

        best = None
        for offset in self._starts(start, end, said):
            try:
                segs = self.engine.decode_range(self.audio, offset, end, self.language)
            except Exception as e:  # a failed try leaves the loop collapsed
                self.log("Loop recovery: decoding from %s failed (%s)" % (_t(offset), e))
                continue
            segs = [s for s in segs if s["text"].strip() and s["start"] < end - 0.05
                    and not self._silence(s)]
            if not segs or any(is_looped(s, self.cr_threshold) for s in segs):
                continue
            if any(same_text(segs[k], segs[k + 1]) and same_text(segs[k], segs[k + 2])
                   for k in range(len(segs) - 2)):
                continue
            for s in segs:
                s["end"] = min(s["end"], end)
            n = sum(len(words_of(s["text"])) for s in segs)
            if best is None or n > best[0]:
                best = (n, offset, segs)
            if n >= 3:
                break

        if best is None:
            self.log("Loop at %s collapsed, nothing more found until %s: %r"
                     % (_t(start), _t(end), once["text"].strip()))
            return [once]

        _n, offset, segs = best
        # the loop's own words are kept if the new text starts later
        result = []
        first = segs[0]
        if first["start"] - start >= 0.3 and not self._repeats(once, first):
            once["end"] = min(once["end"], first["start"])
            result.append(once)
        if result and once["end"] <= once["start"]:
            result = []
        result += segs
        self.log("Loop at %s recovered (decoded again from %s): %d segment(s) until %s"
                 % (_t(start), _t(offset), len(segs), _t(end)))
        return result

    def _starts(self, start, end, said):
        """Where to start decoding again: a bit after the loop's start, so the
        decoder doesn't fall in the same loop."""
        starts = []
        for d in (max(0.4, min(said, 1.0)), 1.5, 3.0):
            t = start + d
            if t < end - 1.0 and (not starts or t - starts[-1] >= 0.4):
                starts.append(round(t, 2))
        return starts

    def _repeats(self, once, first):
        a, b = words_of(once["text"]), words_of(first["text"])
        return bool(a) and b[:len(a)] == a

    def _silence(self, s):
        nsp = s.get("no_speech_prob")
        lp = s.get("avg_logprob")
        if nsp is None or lp is None:
            return False
        return nsp > self.no_speech_threshold and lp < -1.0

    # -- stretched segments ----------------------------------------------------

    def _retime(self, seg, nxt):
        if not self.retime or seg.get("_retimed"):
            return seg
        dur = seg["end"] - seg["start"]
        tokens = seg.get("tokens")
        need = speech_duration(seg["text"], tokens)
        if dur < 3.0 or dur < 2.5 * need + 1.0:
            return seg
        try:
            words = self.engine.align(self.audio, seg, nxt, self.language)
        except Exception as e:
            self.log("Retime at %s failed (%s)" % (_t(seg["start"]), e))
            return seg
        if not words:
            return seg
        probs = [w["probability"] for w in words]
        conf = sum(probs) / len(probs)
        if conf < self.retime_threshold:
            return seg
        new_start = max(seg["start"], words[0]["start"])
        new_end = min(seg["end"], max(words[-1]["end"], new_start + 0.2))
        if nxt is not None:
            new_end = min(new_end, max(nxt["start"], new_start + 0.2))
        # only when it really moves (the point is the lines that come early)
        if new_start - seg["start"] < 1.0 and seg["end"] - new_end < 1.0:
            return seg
        self.log("Stretched segment %s --> %s moved to %s --> %s (confidence %.2f)"
                 % (_t(seg["start"]), _t(seg["end"]), _t(new_start), _t(new_end), conf))
        seg = dict(seg)
        seg["start"], seg["end"] = round(new_start, 3), round(new_end, 3)
        if seg.get("words"):
            seg["words"] = [{"start": round(w["start"], 3), "end": round(w["end"], 3),
                             "word": w["word"], "probability": round(w["probability"], 4)}
                            for w in words]
        seg["_retimed"] = True
        self.retimed += 1
        return seg


def _t(sec):
    m, s = divmod(max(0.0, sec), 60)
    return "%02d:%06.3f" % (m, s)
