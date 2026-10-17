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
      the token ids decoded on the host (tokenizer.json, Detokenizer)

The library's transcripts equal the specification demo/asr/scripts/
moonshine_q.py transcribe_long token for token (ASR_PLAN §6: 200 / 200
LibriSpeech utterances on the board).  English only; greedy decoding (no
temperature, no prompt).  Needs numpy on the board (ffmpeg for formats other
than WAV).
"""

from __future__ import annotations

import ctypes
import json
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
        for name, (args, res) in sig.items():
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


class Detokenizer:
    """The token ids -> text of a tokenizer.json with a BPE model whose decoder is
    Replace("▁" -> " "), ByteFallback, Fuse, Strip(1 leading space) — Moonshine's
    (and Llama's); special tokens skipped, as tokenizers' decode(...,
    skip_special_tokens=True)."""

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


@dataclass
class TranscriptionJob:
    pcm: Any                                       # 16 kHz mono int16
    response_format: str = "json"
    audio_s: float = 0.0
    decode_ms: float = 0.0                         # the upload decoded / resampled (host)
    source: Dict[str, Any] = field(default_factory=dict)


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
            self.detok = Detokenizer(self.tokenizer_path)

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
        return TranscriptionJob(pcm=pcm, response_format=req.response_format, audio_s=audio_s,
                                decode_ms=(time.monotonic() - t0) * 1000.0, source=src)

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
        for a, n in parts:
            cancel.check()
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
        yield Finish("stop", prompt_tokens=0, completion_tokens=len(ids_all), info=info)

    def health(self) -> Dict[str, Any]:
        st = self.stats
        return {"type": "transcription", "library": getattr(self.engine, "lib_path", None),
                "tokenizer": self.tokenizer_path, "sample_rate": self.sample_rate,
                "max_seconds": self.max_seconds, "requests": st["requests"],
                "audio_s": round(st["audio_s"], 2), "tokens": st["tokens"], "parts": st["parts"],
                "rtf": round(st["fpga_s"] / st["audio_s"], 3) if st["audio_s"] else None}
