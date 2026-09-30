"""piper_backend.py — text to speech with Piper (VITS, en_US-lessac-medium) on
the KV260 for the chat server (POST /v1/audio/speech; doc/plans/TTS_PLAN.md §5).

  host (prepare_speech, outside the FPGA lock)
      text -> espeak-ng phonemes (piper_phonemize.py) -> Piper ids, packed by
      sentence into utterances of <= 400 ids
  FPGA + host (synthesize, under the lock)
      per utterance: the text encoder on the FPGA (tts_encode, TTS_PLAN §6),
      the stochastic duration predictor in the library's C (tts_duration,
      §7), the length regulator and the noise in numpy -> z_p
      [192][frames]; then libpiper_tts.so: the reverse flow + HiFi-GAN
      decoder in chunks of 128 frames (1.49 s of audio, ~0.7 s each); every
      chunk's samples go to the client as soon as they exist

The voice directory (--tts-weights) holds the library's weights/*.dat (the
duration predictor's weights/dp.dat among them) and voice.json (phoneme ids,
espeak voice, noise / length scales), written by
demo/tts/scripts/generate_tts_project.py and installed by tts_board.py.  The
library must export tts_encode and tts_duration (built since TTS_PLAN §6 /
§7); load() refuses an older one.  Deterministic: the noise is seeded by the
request's seed (default 0).  Needs numpy (the length regulator and the
noise) and libespeak-ng.so.1 (+ espeak-ng-data) on the board.
"""

from __future__ import annotations

import ctypes
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Union

from chat_backend import Backend, BackendError, CancelToken, Finish, SpeechRequest

MAX_IDS = 400                    # phoneme ids per front-end pass (whole sentences)
HOP = 256                        # samples per frame (the library's tts_hop())


class TtsLibraryError(RuntimeError):
    pass


class LibTtsEngine:
    """libpiper_tts.so through ctypes (demo/tts/src/tts_api.h)."""

    def __init__(self, lib_path: str, weights_dir: Optional[str] = None):
        self.lib_path = lib_path
        self.weights_dir = weights_dir
        self.lib = None
        self.channels = self.hop = self.sample_rate = self.chunk_frames = 0
        self.buf = None

    def _bind(self):
        lib = ctypes.CDLL(os.path.abspath(self.lib_path))
        fp, i16p = ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int16)
        sig = {"tts_open": ([ctypes.c_char_p], ctypes.c_int), "tts_close": ([], None),
               "tts_last_error": ([], ctypes.c_char_p), "tts_channels": ([], ctypes.c_int),
               "tts_sample_rate": ([], ctypes.c_int), "tts_hop": ([], ctypes.c_int),
               "tts_chunk_frames": ([], ctypes.c_int), "tts_num_chunks": ([ctypes.c_int], ctypes.c_int),
               "tts_synthesize_chunk": ([fp, ctypes.c_int, ctypes.c_int, i16p], ctypes.c_int),
               "tts_model_name": ([], ctypes.c_char_p)}
        for name, (args, res) in sig.items():
            f = getattr(lib, name)
            f.argtypes, f.restype = args, res
        # The text encoder (TTS_PLAN §6) and the duration predictor (§7): an
        # older library lacks them and PiperBackend.load() refuses it.
        if hasattr(lib, "tts_encode"):
            lib.tts_encode.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_int, fp, fp, fp]
            lib.tts_encode.restype = ctypes.c_int
            lib.tts_max_ids.argtypes, lib.tts_max_ids.restype = [], ctypes.c_int
        if hasattr(lib, "tts_duration"):
            dp = ctypes.POINTER(ctypes.c_double)
            lib.tts_duration.argtypes = [fp, ctypes.c_int, dp, dp]
            lib.tts_duration.restype = ctypes.c_int
        return lib

    def _err(self, what: str, rc: int) -> TtsLibraryError:
        msg = self.lib.tts_last_error() if self.lib is not None else None
        return TtsLibraryError(f"{what}: {(msg or b'').decode('utf-8', 'replace') or 'error'} (rc {rc})")

    def open(self) -> None:
        if self.lib is not None:
            return
        lib = self._bind()
        self.lib = lib
        rc = lib.tts_open(self.weights_dir.encode() if self.weights_dir else None)
        if rc < 0:
            err = self._err("tts_open", rc)
            self.lib = None
            raise err
        self.channels, self.hop = int(lib.tts_channels()), int(lib.tts_hop())
        self.sample_rate, self.chunk_frames = int(lib.tts_sample_rate()), int(lib.tts_chunk_frames())
        self.buf = (ctypes.c_int16 * (self.chunk_frames * self.hop))()
        self.max_ids = int(lib.tts_max_ids()) if hasattr(lib, "tts_encode") else 0
        self.has_duration = hasattr(lib, "tts_duration")

    @property
    def has_encode(self) -> bool:
        return self.lib is not None and self.max_ids > 0

    def encode(self, ids):
        """The text encoder on the FPGA: ids -> (x, m_p, logs_p) [192][n] float64."""
        import numpy as np
        n = len(ids)
        a = np.ascontiguousarray(ids, np.int32)
        out = np.empty((3, self.channels, n), np.float32)
        fp = ctypes.POINTER(ctypes.c_float)
        rc = self.lib.tts_encode(a.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), n,
                                 out[0].ctypes.data_as(fp), out[1].ctypes.data_as(fp), out[2].ctypes.data_as(fp))
        if rc < 0:
            raise self._err("tts_encode", rc)
        return out[0].astype(np.float64), out[1].astype(np.float64), out[2].astype(np.float64)

    def duration(self, x, z):
        """The C duration predictor: x [192][n] (float32 values), z [2][n] -> logw [n]."""
        import numpy as np
        n = x.shape[1]
        xa = np.ascontiguousarray(x, np.float32)
        za = np.ascontiguousarray(z, np.float64)
        out = np.empty(n, np.float64)
        dp = ctypes.POINTER(ctypes.c_double)
        rc = self.lib.tts_duration(xa.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), n, za.ctypes.data_as(dp),
                                   out.ctypes.data_as(dp))
        if rc < 0:
            raise self._err("tts_duration", rc)
        return out

    def close(self) -> None:
        if self.lib is not None:
            self.lib.tts_close()
            self.lib = None

    @property
    def is_open(self) -> bool:
        return self.lib is not None

    def num_chunks(self, frames: int) -> int:
        return -(-frames // self.chunk_frames) if frames > 0 else 0

    def chunk(self, zp, k: int) -> bytes:
        """Chunk k of the utterance z_p (numpy float32 [192][frames], C order):
        its int16 little-endian samples."""
        frames = zp.shape[1]
        n = self.lib.tts_synthesize_chunk(zp.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), frames, k,
                                          self.buf)
        if n < 0:
            raise self._err("tts_synthesize_chunk", n)
        return ctypes.string_at(self.buf, 2 * n)


@dataclass
class SpeechJob:
    groups: List[List[int]]                    # phoneme ids per utterance
    speed: float = 1.0
    seed: int = 0
    utterances: List[Any] = field(default_factory=list)   # z_p per utterance [192][frames] (synthesize)
    samples: Optional[int] = None              # total, for the Content-Length / WAV header (synthesize)
    frontend_ms: float = 0.0
    voice: Optional[str] = None
    info: Dict[str, Any] = field(default_factory=dict)

    @property
    def ids(self) -> int:
        return sum(len(g) for g in self.groups)


class PiperBackend(Backend):
    """One Piper voice on the FPGA; see the module docstring."""

    speech = True
    sample_rate = 22050

    def __init__(self, engine, voice_dir: str, *, cma_mb: float = 55.0, model_id: Optional[str] = None,
                 espeak_lib: Optional[str] = None, espeak_data: Optional[str] = None,
                 max_ids: int = MAX_IDS, frontend=None, phonemizer=None, encoder=None, duration=None):
        self.engine = engine
        self.voice_dir = voice_dir
        self.cma_mb = cma_mb
        self.espeak_lib, self.espeak_data = espeak_lib, espeak_data
        self.max_ids = max_ids
        self.voice: Dict[str, Any] = {}
        if os.path.exists(os.path.join(voice_dir, "voice.json")):
            with open(os.path.join(voice_dir, "voice.json")) as f:
                self.voice = json.load(f)
        self.model_id = model_id or self.voice.get("model", "piper-lessac-medium")
        self.sample_rate = int(self.voice.get("audio", {}).get("sample_rate", 22050))
        self._frontend = frontend              # tests: a stand-in front_end(ids, speed, seed) -> z_p
        self._espeak = phonemizer              # tests: a stand-in with clauses(text)
        self._encoder = encoder                # ids -> (x, m_p, logs_p); default: the library's (FPGA)
        self._duration = duration              # (x, z) -> logw; default: the library's (C)
        self.stats = {"requests": 0, "audio_s": 0.0, "chunks": 0, "fpga_s": 0.0, "frontend_s": 0.0}

    # ── host side ──

    def load_host(self) -> None:
        if not self.voice:
            raise BackendError(f"{self.voice_dir}/voice.json not found (tts_board.py --install-only)")
        if self._espeak is None:
            from piper_phonemize import Espeak
            self._espeak = Espeak(self.voice.get("espeak", {}).get("voice", "en-us"),
                                  self.espeak_lib, self.espeak_data)
        if self._frontend is None:
            import piper_vits
            inf = self.voice.get("inference", {})

            def fe(ids, speed, seed):
                # the encoder and the duration predictor are the library's (or the
                # caller's stand-ins): front_end reads no weights of its own (W=None)
                return piper_vits.front_end(
                    None, ids, noise_scale=inf.get("noise_scale", 0.667),
                    length_scale=inf.get("length_scale", 1.0) / speed,
                    noise_w=inf.get("noise_w", 0.8), seed=seed, fast_erf=True,
                    encoder=self._encoder or self.engine.encode,
                    duration=self._duration or self.engine.duration)
            self._frontend = fe

    def phoneme_ids(self, text: str) -> List[List[int]]:
        from piper_phonemize import sentences, utterances
        return utterances(sentences(self._espeak.clauses(text)), self.voice["phoneme_id_map"], self.max_ids)

    def prepare(self, req):
        raise BackendError(f"The model '{self.model_id}' is a text-to-speech model: send its input "
                           f"to /v1/audio/speech.", "model")

    def prepare_speech(self, req: SpeechRequest) -> SpeechJob:
        groups = self.phoneme_ids(req.input)
        if not groups:
            raise BackendError("Invalid 'input': nothing to speak (no phonemes).", "input")
        seed = 0 if req.seed is None else req.seed & 0x7FFFFFFFFFFFFFFF
        return SpeechJob(groups=groups, speed=req.speed, seed=seed, voice=req.voice)

    def front_end(self, job: SpeechJob, cancel: Optional[CancelToken] = None) -> None:
        """z_p of every utterance (the encoder on the FPGA: under the lock),
        then the job's sample count."""
        import numpy as np
        t0 = time.monotonic()
        job.utterances = []
        for i, ids in enumerate(job.groups):
            if cancel is not None:
                cancel.check()
            job.utterances.append(np.ascontiguousarray(self._frontend(ids, job.speed, job.seed + i), np.float32))
        frames = sum(z.shape[1] for z in job.utterances)
        job.samples = frames * (getattr(self.engine, "hop", 0) or HOP)
        job.frontend_ms = (time.monotonic() - t0) * 1000.0
        job.info = {"utterances": len(job.utterances), "frames": frames}

    # ── FPGA side ──

    def load(self) -> None:
        self.engine.open()
        if self.engine.sample_rate and self.engine.sample_rate != self.sample_rate:
            raise BackendError(f"libpiper_tts.so runs at {self.engine.sample_rate} Hz, voice.json says "
                               f"{self.sample_rate}")
        eng = self.engine
        lib = getattr(eng, "lib_path", "libpiper_tts.so")
        missing = [f for f, have, own in (("tts_encode", getattr(eng, "has_encode", True), self._encoder),
                                          ("tts_duration", getattr(eng, "has_duration", True), self._duration))
                   if not have and own is None]
        if missing:
            eng.close()
            raise BackendError(f"{lib} has no {' / '.join(missing)} (built before TTS_PLAN §6 / §7): "
                               f"regenerate it with demo/tts/scripts/generate_tts_project.py and install it "
                               f"with tts_board.py --install-only")
        if 0 < getattr(eng, "max_ids", 0) < self.max_ids:
            eng.close()
            raise BackendError(f"{lib} encodes at most {eng.max_ids} phoneme ids; the server packs "
                               f"utterances of up to {self.max_ids}")

    def unload(self) -> None:
        self.engine.close()

    def close(self) -> None:
        self.engine.close()

    def synthesize(self, job: SpeechJob, cancel: CancelToken) -> Iterator[Union[bytes, Finish]]:
        self.front_end(job, cancel)                   # before the first piece: the server reads job.samples
        t0 = time.monotonic()
        chunks = 0
        for zp in job.utterances:
            for k in range(self.engine.num_chunks(zp.shape[1])):
                cancel.check()
                data = self.engine.chunk(zp, k)
                chunks += 1
                yield data
        fpga_s = time.monotonic() - t0
        audio_s = job.samples / self.sample_rate
        st = self.stats
        st["requests"] += 1
        st["audio_s"] += audio_s
        st["chunks"] += chunks
        st["fpga_s"] += fpga_s
        st["frontend_s"] += job.frontend_ms / 1000.0
        info = {"audio_s": round(audio_s, 3), "chunks": chunks, "frontend_ms": round(job.frontend_ms, 1),
                "fpga_ms": round(fpga_s * 1000.0, 1), "rtf": round(fpga_s / audio_s, 3) if audio_s else None,
                **job.info}
        info["log"] = {"ids": job.ids, "chunks": chunks, "frontend": f"{job.frontend_ms:.0f}ms",
                       "fpga": f"{fpga_s * 1000.0:.0f}ms",
                       "rtf": f"{fpga_s / audio_s:.2f}" if audio_s else "-"}
        yield Finish("stop", prompt_tokens=job.ids, completion_tokens=job.info.get("frames", 0), info=info)

    def health(self) -> Dict[str, Any]:
        st = self.stats
        return {"type": "speech", "library": getattr(self.engine, "lib_path", None),
                "voice_dir": self.voice_dir, "sample_rate": self.sample_rate,
                "espeak_voice": self.voice.get("espeak", {}).get("voice"),
                "requests": st["requests"], "audio_s": round(st["audio_s"], 2), "chunks": st["chunks"],
                "rtf": round(st["fpga_s"] / st["audio_s"], 3) if st["audio_s"] else None,
                "frontend_s": round(st["frontend_s"], 2)}
