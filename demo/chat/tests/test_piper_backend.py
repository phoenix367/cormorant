"""piper_backend.py with a Python stand-in for libpiper_tts.so and the front
end: sentence packing, the chunk loop, cancellation, the Finish info and the
server log; piper_phonemize.py against espeak-ng when it is installed."""

import ctypes.util
import os
import unittest

from _util import LOG, RunningServer

from chat_backend import BackendError, Cancelled, CancelToken, SpeechRequest

try:
    import numpy as np
except ImportError:                                            # pragma: no cover
    np = None

ID_MAP = {"^": [1], "_": [0], "$": [2], " ": [3], ".": [10], "!": [4], ",": [8],
          **{c: [20 + i] for i, c in enumerate("abcdefghijklmnopqrstuvwxyzəɪˈ")}}


class FakeEngine:
    """tts_synthesize_chunk in Python: chunk k of z_p -> 128 * 256 samples
    whose value is the chunk's first z_p value (int16), fewer at the end."""

    hop, sample_rate, chunk_frames = 256, 22050, 128

    def __init__(self):
        self.opened = self.closed = 0
        self.calls = []

    def open(self):
        self.opened += 1

    def close(self):
        self.closed += 1

    def num_chunks(self, frames):
        return -(-frames // 128)

    def chunk(self, zp, k):
        self.calls.append((zp.shape[1], k))
        n = min(128, zp.shape[1] - 128 * k) * 256
        return np.full(n, int(zp[0, 128 * k]), "<i2").tobytes()


class FakeEspeak:
    """clauses(): one clause per '.', '!' or ','-terminated piece."""

    def clauses(self, text):
        out, cur = [], ""
        for ch in text:
            if ch in ".!,":
                out.append((cur.strip().lower(), ch))
                cur = ""
            else:
                cur += ch
        if cur.strip():
            out.append((cur.strip().lower(), ""))
        return out


def fake_frontend(ids, speed, seed):
    """frames = 3 per id / speed; z_p[0] = the utterance's seed."""
    frames = int(3 * len(ids) / speed)
    zp = np.zeros((192, frames), np.float32)
    zp[0] = seed
    return zp


@unittest.skipIf(np is None, "needs numpy")
class TestPiperBackend(unittest.TestCase):
    def backend(self, **kw):
        import piper_backend as pb
        b = pb.PiperBackend(FakeEngine(), "/nonexistent", frontend=fake_frontend, phonemizer=FakeEspeak(),
                            **kw)
        b.voice = {"phoneme_id_map": ID_MAP, "audio": {"sample_rate": 22050}}
        b.load_host()
        return b

    def test_job_and_chunks(self):
        b = self.backend(max_ids=40)
        req = SpeechRequest(model=b.model_id, input="Hello there. This is a test! Bye.", seed=5)
        job = b.prepare_speech(req)
        self.assertEqual(len(job.groups), 3)                      # 40 ids per pass: one sentence each
        self.assertIsNone(job.samples)                            # the front end runs under the lock
        b.load()
        out = list(b.synthesize(job, CancelToken()))
        frames = [z.shape[1] for z in job.utterances]
        self.assertEqual(job.samples, sum(frames) * 256)
        pcm, fin = out[:-1], out[-1]
        self.assertEqual(sum(len(p) for p in pcm), 2 * job.samples)
        self.assertEqual(len(pcm), sum(-(-f // 128) for f in frames))
        vals = [int(np.frombuffer(p, "<i2")[0]) for p in pcm]
        self.assertEqual(sorted(set(vals)), [5, 6, 7])           # seed + utterance index
        self.assertEqual(fin.reason, "stop")
        self.assertEqual(fin.info["chunks"], len(pcm))
        self.assertEqual(fin.info["utterances"], 3)
        self.assertEqual(b.health()["requests"], 1)

    def test_packing_and_speed(self):
        b = self.backend()
        req = SpeechRequest(model=b.model_id, input="Hello there. This is a test! Bye.", speed=2.0)
        job = b.prepare_speech(req)
        b.front_end(job)
        self.assertEqual(len(job.utterances), 1)                  # 400 ids: all in one pass
        ids = b.phoneme_ids(req.input)[0]
        self.assertEqual(ids[:2], [1, 0])                         # ^ _
        self.assertEqual(ids[-1], 2)
        self.assertEqual(job.utterances[0].shape[1], int(3 * len(ids) / 2.0))

    def test_cancel_and_empty(self):
        b = self.backend(max_ids=12)                              # one sentence per pass
        job = b.prepare_speech(SpeechRequest(model=b.model_id, input="aaa. bbb. ccc."))
        tok = CancelToken()
        gen = b.synthesize(job, tok)
        next(gen)
        tok.cancel()
        with self.assertRaises(Cancelled):
            next(gen)
        with self.assertRaises(BackendError):
            b.prepare_speech(SpeechRequest(model=b.model_id, input="123 ..."))
        with self.assertRaises(BackendError):
            b.prepare(None)

    def test_through_the_server(self):
        b = self.backend()
        LOG.clear()
        srv = RunningServer({b.model_id: b})
        try:
            st, hd, out = srv.request("POST", "/v1/audio/speech",
                                      {"model": "tts-1", "input": "Hello there.", "response_format": "pcm"})
            self.assertEqual(st, 200, out)
            self.assertEqual(int(hd["content-length"]), len(out))
            self.assertEqual(hd["x-sample-rate"], "22050")
            self.assertEqual(b.engine.opened, 1)
            st, hd, out = srv.request("GET", "/health")
            m = __import__("json").loads(out)["models"][0]
            self.assertEqual((m["id"], m["type"], m["requests"]), ("piper-lessac-medium", "speech", 1))
            self.assertTrue(any("model=piper-lessac-medium" in ln and "chunks=" in ln and "rtf=" in ln
                                for ln in LOG), LOG)
        finally:
            srv.close()


def _espeak():
    lib = os.environ.get("ESPEAKNG_LIB") or ctypes.util.find_library("espeak-ng")
    if not lib:
        return None
    from piper_phonemize import Espeak, PhonemizeError
    try:
        return Espeak("en-us", lib, os.environ.get("ESPEAKNG_DATA"))
    except PhonemizeError:
        return None


class TestPhonemize(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.es = _espeak()

    def setUp(self):
        if self.es is None:
            self.skipTest("needs libespeak-ng (ESPEAKNG_LIB / ESPEAKNG_DATA to point at one)")

    def test_clauses_and_sentences(self):
        from piper_phonemize import sentences
        cl = self.es.clauses("Hello! I am a small assistant, on a board. The answer? Yes.")
        self.assertEqual([t for _, t in cl], ["!", ",", ".", "?", "."])
        s = sentences(cl)
        self.assertEqual(len(s), 4)
        self.assertTrue(s[0].endswith("!"))
        self.assertIn(",", s[1])

    def test_ids(self):
        from piper_phonemize import to_ids, utterances
        m = {"^": [1], "_": [0], "$": [2], "a": [5], "b": [6]}
        self.assertEqual(to_ids("ab?", m), [1, 0, 5, 0, 6, 0, 2])
        long = " ".join(["ab"] * 100)
        u = utterances([long], {**m, " ": [3]}, max_ids=60)
        self.assertTrue(all(len(x) <= 60 for x in u))
        self.assertEqual(sum(x.count(5) for x in u), 100)


if __name__ == "__main__":
    unittest.main()
