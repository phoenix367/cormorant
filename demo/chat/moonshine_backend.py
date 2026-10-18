# Copyright 2026 Ivan Gubochkin
# SPDX-License-Identifier: Apache-2.0

"""moonshine_backend.py — speech to text with Moonshine on the KV260 for the
chat server (POST /v1/audio/transcriptions; doc/plans/ASR_PLAN.md §7).

  host (prepare_transcription, outside the FPGA lock)
      the uploaded file -> 16 kHz mono int16 (speech_input.py: WAV here,
      other formats through ffmpeg); its length checked
  FPGA (transcribe, under the lock)
      libmoonshine_<size>.so (demo/asr/src/asr_api.h): the audio split at
      quiet points into parts of at most 24.6 s (asr_chunks), each part
      encoded in the smallest bucket that holds it and decoded greedily
      (asr_transcribe); a Segment per part (cancellable between parts),
      the token ids decoded on the host (tokenizer.json, Tokenizer)

The library's transcripts equal the specification demo/asr/scripts/
moonshine_q.py transcribe_long token for token (ASR_PLAN §6: 200 / 200
LibriSpeech utterances on the board).  English only; greedy decoding (no
temperature, no prompt).  Needs numpy on the board (ffmpeg for formats other
than WAV).

The phrase check (non-standard form fields kv260_wake_phrase and
kv260_wake_margin; phrase_decode): Moonshine takes no prompt, so a phrase
expected at the start of the audio — a wake phrase, "Hello Miranda" — is fed
to the decoder as its first tokens and scored: the margin is the sum, over
the phrase's tokens the model would not have chosen, of log P(phrase token) -
log P(the model's choice), the raw logit difference / 2^frac (0: the model
writes the phrase itself).  Punctuation the model puts between the phrase's
words is taken as it is.  A margin >= kv260_wake_margin keeps the phrase and
decodes on greedily from it; below, the transcript is the plain greedy one
(the same ids as without the check).  The response's kv260.wake says which.
"""

from __future__ import annotations

import ctypes
import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from chat_backend import Backend, BackendError, CancelToken, Finish, Segment, TranscriptionRequest

LANGUAGES = ("en", "english")


class AsrLibraryError(RuntimeError):
    pass


class _Stats(ctypes.Structure):
    _fields_ = [("encode_ms", ctypes.c_double), ("decode_ms", ctypes.c_double), ("total_ms", ctypes.c_double),
                ("parts", ctypes.c_int), ("tokens", ctypes.c_int), ("steps", ctypes.c_int)]


class LibAsrEngine:
    """libmoonshine_<size>.so through ctypes (demo/asr/src/asr_api.h)."""

    def __init__(self, lib_path: str, weights_dir: Optional[str] = None):
        self.lib_path = lib_path
        self.weights_dir = weights_dir
        self.lib = None
        self.sample_rate = self.max_samples = 0
        self.model_name = None
        self.can_check = False                 # the library has the step functions of phrase_decode
        self._pcm = None

    def _bind(self):
        lib = ctypes.CDLL(os.path.abspath(self.lib_path))
        i16p, i32p = ctypes.POINTER(ctypes.c_int16), ctypes.POINTER(ctypes.c_int32)
        sig = {"asr_open": ([ctypes.c_char_p], ctypes.c_int), "asr_close": ([], None),
               "asr_last_error": ([], ctypes.c_char_p), "asr_sample_rate": ([], ctypes.c_int),
               "asr_max_samples": ([], ctypes.c_int), "asr_num_buckets": ([], ctypes.c_int),
               "asr_bucket": ([ctypes.c_int], ctypes.c_int), "asr_eos": ([], ctypes.c_int),
               "asr_transcribe": ([i16p, ctypes.c_int, i32p, ctypes.c_int], ctypes.c_int),
               "asr_chunks": ([i16p, ctypes.c_int, i32p, i32p, ctypes.c_int], ctypes.c_int),
               "asr_last_stats": ([ctypes.POINTER(_Stats)], None),
               "asr_model_name": ([], ctypes.c_char_p)}
        steps = {"asr_bos": ([], ctypes.c_int), "asr_encode": ([i16p, ctypes.c_int], ctypes.c_int),
                 "asr_decode_step": ([ctypes.c_int, ctypes.c_int], ctypes.c_int),
                 "asr_last_logits": ([i32p, ctypes.c_int, i32p], ctypes.c_int),
                 "asr_logit_frac": ([], ctypes.c_int), "asr_max_steps": ([ctypes.c_int], ctypes.c_int)}
        for name, (args, res) in sig.items():
            f = getattr(lib, name)
            f.argtypes, f.restype = args, res
        self.can_check = all(hasattr(lib, name) for name in steps)   # libraries before the phrase check: no
        if self.can_check:
            for name, (args, res) in steps.items():
                f = getattr(lib, name)
                f.argtypes, f.restype = args, res
        return lib

    def _err(self, what: str, rc: int) -> AsrLibraryError:
        msg = self.lib.asr_last_error() if self.lib is not None else None
        return AsrLibraryError(f"{what}: {(msg or b'').decode('utf-8', 'replace') or 'error'} (rc {rc})")

    def open(self) -> None:
        if self.lib is not None:
            return
        lib = self._bind()
        self.lib = lib
        rc = lib.asr_open(self.weights_dir.encode() if self.weights_dir else None)
        if rc < 0:
            err = self._err("asr_open", rc)
            self.lib = None
            raise err
        self.sample_rate, self.max_samples = int(lib.asr_sample_rate()), int(lib.asr_max_samples())
        self.buckets = [int(lib.asr_bucket(i)) for i in range(int(lib.asr_num_buckets()))]
        self.model_name = (lib.asr_model_name() or b"").decode() or None
        self.eos = int(lib.asr_eos())
        if self.can_check:
            self.bos, self.logit_frac = int(lib.asr_bos()), int(lib.asr_logit_frac())

    def close(self) -> None:
        if self.lib is not None:
            self.lib.asr_close()
            self.lib = None

    @property
    def is_open(self) -> bool:
        return self.lib is not None

    @staticmethod
    def _ptr(a, ctype):
        return a.ctypes.data_as(ctypes.POINTER(ctype))

    def chunks(self, pcm) -> List[Tuple[int, int]]:
        """The parts asr_transcribe() would split pcm into: [(start, length)]."""
        import numpy as np
        x = np.ascontiguousarray(pcm, np.int16)
        k = len(x) // max(self.max_samples, 1) + 2
        st, ln = np.zeros(k, np.int32), np.zeros(k, np.int32)
        n = self.lib.asr_chunks(self._ptr(x, ctypes.c_int16), len(x), self._ptr(st, ctypes.c_int32),
                                self._ptr(ln, ctypes.c_int32), k)
        if n < 0:
            raise self._err("asr_chunks", n)
        return [(int(st[i]), int(ln[i])) for i in range(n)]

    def transcribe(self, pcm) -> Tuple[List[int], Dict[str, Any]]:
        """Token ids of pcm (one part: at most max_samples, or any length) and the
        library's stats."""
        import numpy as np
        x = np.ascontiguousarray(pcm, np.int16)
        cap = (len(x) // max(self.max_samples, 1) + 1) * 256
        ids = np.zeros(cap, np.int32)
        n = self.lib.asr_transcribe(self._ptr(x, ctypes.c_int16), len(x), self._ptr(ids, ctypes.c_int32), cap)
        if n < 0:
            raise self._err("asr_transcribe", n)
        st = _Stats()
        self.lib.asr_last_stats(ctypes.byref(st))
        return [int(v) for v in ids[:n]], {k: getattr(st, k) for k, _ in _Stats._fields_}


    # ── the steps of phrase_decode (one part: at most max_samples) ──

    def encode(self, pcm) -> None:
        import numpy as np
        self._pcm = np.ascontiguousarray(pcm, np.int16)
        rc = self.lib.asr_encode(self._ptr(self._pcm, ctypes.c_int16), len(self._pcm))
        if rc < 0:
            raise self._err("asr_encode", rc)

    def step(self, token: int, pos: int) -> int:
        rc = self.lib.asr_decode_step(token, pos)
        if rc < 0:
            raise self._err("asr_decode_step", rc)
        return int(rc)

    def logits(self, tokens) -> List[int]:
        """The last step's raw logits of tokens (int16 at 2^-logit_frac)."""
        t = (ctypes.c_int32 * len(tokens))(*tokens)
        out = (ctypes.c_int32 * len(tokens))()
        rc = self.lib.asr_last_logits(t, len(tokens), out)
        if rc < 0:
            raise self._err("asr_last_logits", rc)
        return list(out)

    def max_steps(self, n: int) -> int:
        return int(self.lib.asr_max_steps(n))


class Tokenizer:
    """A tokenizer.json with a BPE model whose decoder is Replace("▁" -> " "),
    ByteFallback, Fuse, Strip(1 leading space) and whose normalizer prepends "▁"
    and replaces " " by "▁" (no pre-tokenizer) — Moonshine's (and Llama's).
    decode: the token ids -> text, special tokens skipped, as tokenizers'
    decode(..., skip_special_tokens=True); encode: a short text -> ids, as
    tokenizers' encode(..., add_special_tokens=False) (the merges by rank,
    bytes for unknown characters) — the phrase of the phrase check."""

    _BYTE = re.compile(r"<0x([0-9A-Fa-f]{2})>")

    def __init__(self, path: str):
        with open(path, encoding="utf-8") as f:
            t = json.load(f)
        if t.get("model", {}).get("type") != "BPE":
            raise ValueError(f"{path}: a BPE tokenizer expected, not {t.get('model', {}).get('type')}")
        vocab = t["model"]["vocab"]
        size = max([*vocab.values(), *(a["id"] for a in t.get("added_tokens", []))]) + 1
        self.pieces: List[Optional[str]] = [None] * size
        for p, i in vocab.items():
            self.pieces[i] = p
        self.special = set()
        for a in t.get("added_tokens", []):
            self.pieces[a["id"]] = a["content"]
            if a.get("special"):
                self.special.add(a["id"])
        steps = [d.get("type") for d in (t.get("decoder") or {}).get("decoders", [])]
        if steps != ["Replace", "ByteFallback", "Fuse", "Strip"]:
            raise ValueError(f"{path}: decoder {steps} is not Moonshine's")
        self.vocab = vocab
        self._merges = t["model"].get("merges", [])
        self._rank: Optional[Dict[Tuple[str, str], int]] = None
        self.unk = vocab.get(t["model"].get("unk_token") or "<unk>", 0)

    def encode(self, text: str) -> List[int]:
        if self._rank is None:
            self._rank = {tuple(m.split(" ", 1)) if isinstance(m, str) else tuple(m): i
                          for i, m in enumerate(self._merges)}
        sym = list("▁" + text.replace(" ", "▁")) if text else []
        while len(sym) > 1:
            best = min(range(len(sym) - 1), key=lambda i: self._rank.get((sym[i], sym[i + 1]), 1 << 30))
            if (sym[best], sym[best + 1]) not in self._rank:
                break
            sym[best:best + 2] = [sym[best] + sym[best + 1]]
        ids: List[int] = []
        for p in sym:
            if p in self.vocab:
                ids.append(self.vocab[p])
            else:
                ids += [self.vocab.get(f"<0x{b:02X}>", self.unk) for b in p.encode("utf-8")]
        return ids

    def decode(self, ids) -> str:
        out: List[str] = []
        buf = bytearray()

        def flush():
            if buf:
                try:
                    out.append(buf.decode("utf-8"))
                except UnicodeDecodeError:
                    out.append("�" * len(buf))
                buf.clear()
        for i in ids:
            if i in self.special or not 0 <= i < len(self.pieces) or self.pieces[i] is None:
                continue
            p = self.pieces[i]
            m = self._BYTE.fullmatch(p)
            if m:
                buf.append(int(m.group(1), 16))
                continue
            flush()
            out.append(p.replace("▁", " "))
        flush()
        s = "".join(out)
        return s[1:] if s.startswith(" ") else s


PUNCT = frozenset(",.!?;:")
WAKE_MARGIN = -10.0                                # kv260_wake_margin's default (nats; ASR_PLAN §8)
MAX_PHRASE = 64                                    # kv260_wake_phrase's characters


@dataclass
class Phrase:
    """A phrase to check at the start of the audio: its tokens, and which of
    them start a word (punctuation the model writes may come before those)."""
    text: str
    ids: List[int]
    word_start: List[bool]

    @classmethod
    def of(cls, tok: Tokenizer, text: str) -> "Phrase":
        ids = tok.encode(text)
        return cls(text, ids, [(tok.pieces[i] or "").startswith("▁") for i in ids])


def punct_ids(tok: Tokenizer) -> frozenset:
    """The tokens that are punctuation alone (",", ".", "?" ...)."""
    return frozenset(i for i, p in enumerate(tok.pieces) if p and set(p) <= PUNCT)


def phrase_decode(eng, pcm, phrase: Phrase, min_margin: float,
                  punct: frozenset) -> Tuple[List[int], Dict[str, Any], Dict[str, Any]]:
    """One part (at most eng.max_samples): the phrase fed as the first tokens and
    scored (the module docstring), then greedy on from it when the margin holds,
    else the plain greedy transcript (asr_transcribe's ids: the decoder rewrites
    its cache from position 1 on).  The margin stops being summed once it is
    below min_margin (complete: False).  ``eng``: LibAsrEngine's encode / step /
    logits / max_steps / bos / eos / logit_frac.  -> (ids, the engine's stats,
    the check's result)."""
    t0 = time.monotonic()
    S = eng.max_steps(len(pcm))
    res: Dict[str, Any] = {"phrase": phrase.text, "margin": 0.0, "complete": False, "passed": False}
    if S < 1 or not phrase.ids:
        return [], {"encode_ms": 0.0, "decode_ms": 0.0, "steps": 0}, res
    eng.encode(pcm)
    t1 = time.monotonic()
    scale = 2.0 ** -eng.logit_frac
    first = a = eng.step(eng.bos, 0)
    steps, ids, j, margin, run = 1, [], 0, 0, 0
    while j < len(phrase.ids):
        want = phrase.ids[j]
        if a != want and j > 0 and phrase.word_start[j] and a in punct and run < 2:
            t, run = a, run + 1                    # "Hello, Miranda": the model's comma
        else:
            if a != want:
                lw, la = eng.logits([want, a])
                margin += lw - la
                if margin * scale < min_margin:
                    break
            t, j, run = want, j + 1, 0
        ids.append(t)
        if len(ids) >= S:
            break
        a = eng.step(t, len(ids))
        steps += 1
    res.update(margin=margin * scale, complete=j == len(phrase.ids))
    if res["complete"] and margin * scale >= min_margin:
        res["passed"] = True                       # on greedily from the phrase
    else:
        ids, a = [], first                         # the plain transcript from its first step
    while len(ids) < S and a != eng.eos:          # a: the model's token after ids
        ids.append(a)
        if len(ids) >= S:
            break
        a = eng.step(a, len(ids))
        steps += 1
    t2 = time.monotonic()
    return ids, {"encode_ms": (t1 - t0) * 1000.0, "decode_ms": (t2 - t1) * 1000.0, "steps": steps}, res


@dataclass
class TranscriptionJob:
    pcm: Any                                       # 16 kHz mono int16
    response_format: str = "json"
    audio_s: float = 0.0
    decode_ms: float = 0.0                         # the upload decoded / resampled (host)
    source: Dict[str, Any] = field(default_factory=dict)
    wake: Optional[Phrase] = None                  # kv260_wake_phrase
    wake_margin: float = WAKE_MARGIN


class MoonshineBackend(Backend):
    """One Moonshine library on the FPGA; see the module docstring."""

    transcription = True
    sample_rate = 16000

    def __init__(self, engine, tokenizer: str, *, cma_mb: float = 200.0, model_id: Optional[str] = None,
                 ffmpeg: Optional[str] = None, max_seconds: float = 600.0, detokenizer=None):
        self.engine = engine
        self.tokenizer_path = tokenizer
        self.cma_mb = cma_mb
        self.model_id = model_id or self._default_id()
        self.ffmpeg = ffmpeg
        self.max_seconds = max_seconds
        self.detok = detokenizer
        self._punct: Optional[frozenset] = None
        self.stats = {"requests": 0, "audio_s": 0.0, "fpga_s": 0.0, "tokens": 0, "parts": 0}

    def _default_id(self) -> str:
        """moonshine-base / moonshine-tiny from the library's file name (the id
        must be known before load())."""
        base = os.path.basename(getattr(self.engine, "lib_path", "") or "")
        m = re.fullmatch(r"libmoonshine_(\w+)\.so", base)
        return f"moonshine-{m.group(1)}" if m else "moonshine-base"

    # ── host side ──

    def load_host(self) -> None:
        if self.detok is None:
            if not os.path.exists(self.tokenizer_path):
                raise BackendError(f"{self.tokenizer_path} not found (the Moonshine tokenizer.json)")
            self.detok = Tokenizer(self.tokenizer_path)

    def prepare(self, req):
        raise BackendError(f"The model '{self.model_id}' is a speech-to-text model: send audio to "
                           f"/v1/audio/transcriptions.", "model")

    def prepare_transcription(self, req: TranscriptionRequest) -> TranscriptionJob:
        from speech_input import AudioError, decode_audio
        if req.language and req.language.strip().lower() not in LANGUAGES:
            raise BackendError(f"Invalid 'language': '{req.language}' — {self.model_id} transcribes "
                               f"English only ('en').", "language", code="unsupported_language")
        t0 = time.monotonic()
        try:
            pcm, src = decode_audio(req.audio, self.ffmpeg)
        except AudioError as e:
            raise BackendError(f"Invalid file: {e}.", "file", code="invalid_audio") from None
        audio_s = len(pcm) / self.sample_rate
        if len(pcm) == 0:
            raise BackendError("Invalid file: it holds no audio samples.", "file", code="invalid_audio")
        if audio_s > self.max_seconds:
            raise BackendError(f"Invalid file: {audio_s:.0f} s of audio; this server transcribes at most "
                               f"{self.max_seconds:.0f} s.", "file", code="audio_too_long")
        wake, margin = self._wake_fields(req.raw)
        return TranscriptionJob(pcm=pcm, response_format=req.response_format, audio_s=audio_s,
                                decode_ms=(time.monotonic() - t0) * 1000.0, source=src, wake=wake,
                                wake_margin=margin)

    def _wake_fields(self, raw: Dict[str, Any]) -> Tuple[Optional[Phrase], float]:
        """kv260_wake_phrase / kv260_wake_margin (non-standard form fields)."""
        text = raw.get("kv260_wake_phrase")
        if text in (None, ""):
            return None, WAKE_MARGIN
        if not isinstance(text, str) or not text.strip() or len(text.strip()) > MAX_PHRASE:
            raise BackendError(f"Invalid 'kv260_wake_phrase': a phrase of 1 to {MAX_PHRASE} characters "
                               f"expected.", "kv260_wake_phrase")
        margin = raw.get("kv260_wake_margin")
        if margin in (None, ""):
            margin = WAKE_MARGIN
        else:
            try:
                margin = float(margin)
            except (TypeError, ValueError):
                raise BackendError(f"Invalid type for 'kv260_wake_margin': expected a number, but got "
                                   f"'{margin}' instead.", "kv260_wake_margin", code="invalid_type") from None
            if math.isnan(margin) or margin > 0:
                raise BackendError(f"Invalid 'kv260_wake_margin': expected a number <= 0 (nats), but got "
                                   f"{margin} instead.", "kv260_wake_margin", code="invalid_value")
        if self.detok is None:
            self.load_host()
        return Phrase.of(self.detok, " ".join(text.split())), margin

    # ── FPGA side ──

    def load(self) -> None:
        self.engine.open()
        if self.engine.sample_rate and self.engine.sample_rate != self.sample_rate:
            self.engine.close()
            raise BackendError(f"{self.engine.lib_path} runs at {self.engine.sample_rate} Hz, not "
                               f"{self.sample_rate}")

    def unload(self) -> None:
        self.engine.close()

    def close(self) -> None:
        self.engine.close()

    def transcribe(self, job: TranscriptionJob, cancel: CancelToken) -> Iterator[Union[Segment, Finish]]:
        t0 = time.monotonic()
        sr = self.sample_rate
        ids_all: List[int] = []
        enc_ms = dec_ms = 0.0
        steps = 0
        parts = self.engine.chunks(job.pcm)
        wake = None
        for i, (a, n) in enumerate(parts):
            cancel.check()
            if job.wake is not None and i == 0:
                ids, st, wake = self._phrase_part(job, job.pcm[a:a + n], len(parts))
            else:
                ids, st = self.engine.transcribe(job.pcm[a:a + n])
            enc_ms += st.get("encode_ms", 0.0)
            dec_ms += st.get("decode_ms", 0.0)
            steps += st.get("steps", 0)
            ids_all += ids
            yield Segment(start=a / sr, end=(a + n) / sr, text=self.detok.decode(ids).strip(), tokens=ids)
        fpga_s = time.monotonic() - t0
        st = self.stats
        st["requests"] += 1
        st["audio_s"] += job.audio_s
        st["fpga_s"] += fpga_s
        st["tokens"] += len(ids_all)
        st["parts"] += len(parts)
        rtf = fpga_s / job.audio_s if job.audio_s else None
        info = {"text": self.detok.decode(ids_all).strip(), "audio_s": round(job.audio_s, 3),
                "parts": len(parts), "tokens": len(ids_all), "encode_ms": round(enc_ms, 1),
                "decode_ms": round(dec_ms, 1), "fpga_ms": round(fpga_s * 1000.0, 1),
                "input_ms": round(job.decode_ms, 1), "rtf": round(rtf, 3) if rtf else None,
                "source": job.source.get("source")}
        info["log"] = {"parts": len(parts), "steps": steps, "input": f"{job.decode_ms:.0f}ms",
                       "fpga": f"{fpga_s * 1000.0:.0f}ms", "rtf": f"{rtf:.3f}" if rtf else "-"}
        if wake is not None:
            info["wake"] = wake
            if "margin" in wake:
                info["log"]["wake"] = f"{wake['margin']:.2f}{'+' if wake['passed'] else '-'}"
        yield Finish("stop", prompt_tokens=0, completion_tokens=len(ids_all), info=info)

    def _phrase_part(self, job: TranscriptionJob, pcm, parts: int):
        """The first part with the phrase check -> (ids, stats, kv260.wake)."""
        eng = self.engine
        why = (None if getattr(eng, "can_check", False) else
               f"{getattr(eng, 'lib_path', 'the library')} has no asr_last_logits (regenerate it)")
        if why is None and parts > 1:
            why = "the audio is longer than one encoder pass"
        if why is not None:
            ids, st = eng.transcribe(pcm)
            return ids, st, {"phrase": job.wake.text, "passed": False, "error": why}
        if self._punct is None:
            self._punct = punct_ids(self.detok)
        ids, st, res = phrase_decode(eng, pcm, job.wake, job.wake_margin, self._punct)
        res["margin"] = round(res["margin"], 3)
        res["min_margin"] = job.wake_margin
        return ids, st, res

    def health(self) -> Dict[str, Any]:
        st = self.stats
        return {"type": "transcription", "library": getattr(self.engine, "lib_path", None),
                "tokenizer": self.tokenizer_path, "sample_rate": self.sample_rate,
                "max_seconds": self.max_seconds, "requests": st["requests"],
                "audio_s": round(st["audio_s"], 2), "tokens": st["tokens"], "parts": st["parts"],
                "rtf": round(st["fpga_s"] / st["audio_s"], 3) if st["audio_s"] else None}
