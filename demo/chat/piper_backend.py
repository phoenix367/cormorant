"""piper_backend.py — text to speech with Piper (VITS, en_US-lessac-medium) on
the KV260 for the chat server (POST /v1/audio/speech; doc/plans/TTS_PLAN.md §5).

  host (prepare_speech, outside the FPGA lock)
      text -> espeak-ng phonemes (piper_phonemize.py) -> Piper ids, packed by
      sentence into utterances of <= 400 ids -> per utterance the float64
      front end of piper_vits.py (text encoder, stochastic duration
      predictor, length regulator, noise; erf_fast) -> z_p [192][frames]
  FPGA (synthesize, under the lock)
      libpiper_tts.so: the reverse flow + HiFi-GAN decoder in chunks of 128
      frames (1.49 s of audio, ~1 s each); every chunk's samples go to the
      client as soon as they exist

The voice directory (--tts-weights) holds the library's weights/*.dat,
frontend.npz (the text encoder's and duration predictor's weights) and
voice.json (phoneme ids, espeak voice, noise / length scales), written by
demo/tts/scripts/generate_tts_project.py and installed by tts_board.py.
Deterministic: the noise is seeded by the request's seed (default 0).
Needs numpy and libespeak-ng.so.1 (+ espeak-ng-data) on the board.
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
    utterances: List[Any]                      # z_p per utterance, float32 [192][frames]
    samples: int                               # total, for the Content-Length / WAV header
    ids: int
    frontend_ms: float
    voice: Optional[str] = None
    info: Dict[str, Any] = field(default_factory=dict)


class PiperBackend(Backend):
    """One Piper voice on the FPGA; see the module docstring."""

    speech = True
    sample_rate = 22050

    def __init__(self, engine, voice_dir: str, *, cma_mb: float = 40.0, model_id: Optional[str] = None,
                 espeak_lib: Optional[str] = None, espeak_data: Optional[str] = None,
                 max_ids: int = MAX_IDS, frontend=None, phonemizer=None):
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
        self._frontend = frontend              # tests: a stand-in front_end(ids, ...) -> z_p
        self._espeak = phonemizer              # tests: a stand-in with clauses(text)
        self.W = None
        self.stats = {"requests": 0, "audio_s": 0.0, "chunks": 0, "fpga_s": 0.0, "frontend_s": 0.0}

    # ── host side ──

    def load_host(self) -> None:
        import numpy as np
        if not self.voice:
            raise BackendError(f"{self.voice_dir}/voice.json not found (tts_board.py --install-only)")
        if self._espeak is None:
            from piper_phonemize import Espeak
            self._espeak = Espeak(self.voice.get("espeak", {}).get("voice", "en-us"),
                                  self.espeak_lib, self.espeak_data)
        if self._frontend is None:
            import piper_vits
            with np.load(os.path.join(self.voice_dir, "frontend.npz")) as z:
                self.W = {k: z[k].astype(np.float64) for k in z.files}
            inf = self.voice.get("inference", {})

            def fe(ids, speed, seed):
                return piper_vits.front_end(
                    self.W, ids, noise_scale=inf.get("noise_scale", 0.667),
                    length_scale=inf.get("length_scale", 1.0) / speed,
                    noise_w=inf.get("noise_w", 0.8), seed=seed, fast_erf=True)
            self._frontend = fe

    def phoneme_ids(self, text: str) -> List[List[int]]:
        from piper_phonemize import sentences, utterances
        return utterances(sentences(self._espeak.clauses(text)), self.voice["phoneme_id_map"], self.max_ids)

    def prepare(self, req):
        raise BackendError(f"The model '{self.model_id}' is a text-to-speech model: send its input "
                           f"to /v1/audio/speech.", "model")

    def prepare_speech(self, req: SpeechRequest) -> SpeechJob:
        import numpy as np
        t0 = time.monotonic()
        groups = self.phoneme_ids(req.input)
        if not groups:
            raise BackendError("Invalid 'input': nothing to speak (no phonemes).", "input")
        seed = 0 if req.seed is None else req.seed & 0x7FFFFFFFFFFFFFFF
        zps = [np.ascontiguousarray(self._frontend(ids, req.speed, seed + i), np.float32)
               for i, ids in enumerate(groups)]
        frames = sum(z.shape[1] for z in zps)
        hop = getattr(self.engine, "hop", 0) or HOP
        return SpeechJob(utterances=zps, samples=frames * hop, ids=sum(len(g) for g in groups),
                         frontend_ms=(time.monotonic() - t0) * 1000.0, voice=req.voice,
                         info={"utterances": len(zps), "frames": frames})

    # ── FPGA side ──

    def load(self) -> None:
        self.engine.open()
        if self.engine.sample_rate and self.engine.sample_rate != self.sample_rate:
            raise BackendError(f"libpiper_tts.so runs at {self.engine.sample_rate} Hz, voice.json says "
                               f"{self.sample_rate}")

    def unload(self) -> None:
        self.engine.close()

    def close(self) -> None:
        self.engine.close()

    def synthesize(self, job: SpeechJob, cancel: CancelToken) -> Iterator[Union[bytes, Finish]]:
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
