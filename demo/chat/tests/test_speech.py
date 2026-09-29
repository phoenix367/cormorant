"""POST /v1/audio/speech of kv260_chat_server.py against a fake speech
backend: formats (wav / pcm with a Content-Length, mp3 / flac through ffmpeg,
chunked), SSE events, model aliases and defaults, OpenAI-style errors, the
shared FPGA queue and client disconnects."""

import array
import base64
import io
import json
import shutil
import threading
import time
import unittest
import wave

from _util import LOG, RunningServer, http_request_bytes, server_mod, sse_events, wait_until

from chat_backend import Backend, BackendError, Cancelled, Finish

PIECE = 1000                                   # samples per synthesized piece


def piece(i: int) -> bytes:
    return array.array("h", [(i * 37) % 30000 - 15000] * PIECE).tobytes()


class FakeSpeech(Backend):
    """One PCM piece per word of the input; `delay` seconds of cancellable
    "FPGA work" before each."""

    model_id = "fake-tts"
    speech = True
    sample_rate = 16000

    def __init__(self, delay=0.0):
        self.delay = delay
        self.stopped = 0                       # synthesize() left early (Cancelled or closed)
        self.jobs = []

    def prepare_speech(self, req):
        if req.input.strip() == "bad":
            raise BackendError("Invalid 'input': nothing to speak (no phonemes).", "input")
        words = req.input.split()
        self.jobs.append(req)
        return {"n": len(words), "samples": len(words) * PIECE}

    def synthesize(self, job, cancel):
        try:
            for i in range(job["n"]):
                end = time.monotonic() + self.delay
                while time.monotonic() < end:
                    cancel.check()
                    time.sleep(0.005)
                cancel.check()
                yield piece(i)
            yield Finish("stop", prompt_tokens=job["n"], completion_tokens=job["n"] * 4,
                         info={"log": {"chunks": job["n"]}})
        except (Cancelled, GeneratorExit):
            self.stopped += 1
            raise


class Job(dict):
    """prepare_speech() result with the attribute the server reads."""

    @property
    def samples(self):
        return self["samples"]


class FakeSpeechJob(FakeSpeech):
    def prepare_speech(self, req):
        return Job(super().prepare_speech(req))


def expected_pcm(n: int) -> bytes:
    return b"".join(piece(i) for i in range(n))


class SpeechServer(unittest.TestCase):
    ffmpeg = shutil.which("ffmpeg")

    def setUp(self):
        LOG.clear()
        self.tts = FakeSpeechJob()
        self.srv = RunningServer({"echo": server_mod.EchoBackend(), "fake-tts": self.tts},
                                 ffmpeg=self.ffmpeg)

    def tearDown(self):
        self.srv.close()

    def speech(self, body, **kw):
        return self.srv.request("POST", "/v1/audio/speech", body, **kw)


class TestSpeech(SpeechServer):
    def test_wav(self):
        st, hd, out = self.speech({"model": "fake-tts", "input": "one two three", "voice": "alloy",
                                   "response_format": "wav"})
        self.assertEqual(st, 200, out)
        self.assertEqual(hd["content-type"], "audio/wav")
        self.assertEqual(int(hd["content-length"]), 44 + 2 * 3 * PIECE)
        self.assertEqual(hd["x-sample-rate"], "16000")
        with wave.open(io.BytesIO(out)) as w:
            self.assertEqual((w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()),
                             (1, 2, 16000, 3 * PIECE))
            self.assertEqual(w.readframes(w.getnframes()), expected_pcm(3))
        self.assertTrue(any("POST /v1/audio/speech 200 model=fake-tts" in ln and "audio=0.19s" in ln
                            for ln in LOG), LOG)

    def test_default_format_is_wav_and_default_model_is_speech(self):
        st, hd, out = self.speech({"input": "a b"})
        self.assertEqual(st, 200, out)
        self.assertEqual(hd["content-type"], "audio/wav")
        self.assertEqual(out[:4], b"RIFF")
        # chat requests without a model still go to the chat backend
        st, _, out = self.srv.post({"messages": [{"role": "user", "content": "hi there"}]})
        self.assertEqual(st, 200, out)
        self.assertEqual(json.loads(out)["model"], "echo")

    def test_pcm(self):
        st, hd, out = self.speech({"model": "fake-tts", "input": "x y", "response_format": "pcm"})
        self.assertEqual(st, 200)
        self.assertEqual(hd["content-type"], "audio/pcm")
        self.assertEqual(out, expected_pcm(2))

    def test_aliases_speed_seed(self):
        for alias in ("tts-1", "tts-1-hd", "gpt-4o-mini-tts"):
            st, _, out = self.speech({"model": alias, "input": "a", "speed": 1.5, "seed": 7,
                                      "instructions": "cheerful"})
            self.assertEqual(st, 200, out)
        req = self.tts.jobs[-1]
        self.assertEqual((req.model, req.speed, req.seed), ("fake-tts", 1.5, 7))

    def test_sse(self):
        st, hd, out = self.speech({"model": "fake-tts", "input": "a b c", "stream_format": "sse",
                                   "response_format": "pcm"})
        self.assertEqual(st, 200)
        self.assertTrue(hd["content-type"].startswith("text/event-stream"))
        ev = sse_events(out)
        self.assertEqual([e["type"] for e in ev], ["speech.audio.delta"] * 3 + ["speech.audio.done"])
        self.assertEqual(b"".join(base64.b64decode(e["audio"]) for e in ev[:-1]), expected_pcm(3))
        self.assertEqual(ev[-1]["usage"], {"input_tokens": 3, "output_tokens": 12, "total_tokens": 15})

    def test_errors(self):
        cases = [({"model": "fake-tts"}, 400, "missing_required_parameter"),
                 ({"model": "fake-tts", "input": "  "}, 400, None),
                 ({"model": "fake-tts", "input": 5}, 400, "invalid_type"),
                 ({"model": "fake-tts", "input": "a" * 4097}, 400, "string_above_max_length"),
                 ({"model": "fake-tts", "input": "a", "response_format": "ogg"}, 400, None),
                 ({"model": "fake-tts", "input": "a", "stream_format": "ws"}, 400, None),
                 ({"model": "fake-tts", "input": "a", "speed": 5}, 400, "invalid_value"),
                 ({"model": "fake-tts", "input": "a", "voice": 3}, 400, None),
                 ({"model": "nope", "input": "a"}, 404, "model_not_found"),
                 ({"model": "echo", "input": "a"}, 400, "model_not_supported"),
                 ({"model": "fake-tts", "input": "bad"}, 400, None)]
        for body, status, code in cases:
            st, hd, out = self.speech(body)
            self.assertEqual(st, status, (body, out))
            err = json.loads(out)["error"]
            self.assertEqual(set(err), {"message", "type", "param", "code"})
            if code:
                self.assertEqual(err["code"], code, body)
        st, _, out = self.srv.post({"model": "fake-tts", "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(st, 400)
        self.assertIn("/v1/audio/speech", json.loads(out)["error"]["message"])
        st, _, out = self.srv.request("POST", "/v1/audio/speech", raw=b"{not json")
        self.assertEqual(st, 400)

    @unittest.skipUnless(shutil.which("ffmpeg"), "needs ffmpeg")
    def test_ffmpeg_formats(self):
        for fmt, ctype, magic in (("mp3", "audio/mpeg", (b"ID3", b"\xff\xfb", b"\xff\xf3", b"\xff\xe3")),
                                  ("flac", "audio/flac", (b"fLaC",)),
                                  ("opus", "audio/ogg", (b"OggS",)),
                                  ("aac", "audio/aac", (b"\xff\xf1", b"\xff\xf9"))):
            st, hd, out = self.speech({"model": "fake-tts", "input": "a b c d", "response_format": fmt})
            self.assertEqual(st, 200, (fmt, out[:200]))
            self.assertEqual(hd["content-type"], ctype)
            self.assertEqual(hd.get("transfer-encoding"), "chunked")
            self.assertTrue(out.startswith(magic), (fmt, out[:8]))
        # SSE around an encoded stream: the deltas concatenate to the file
        st, _, out = self.speech({"model": "fake-tts", "input": "a b", "response_format": "flac",
                                  "stream_format": "sse"})
        ev = sse_events(out)
        self.assertTrue(b"".join(base64.b64decode(e["audio"]) for e in ev[:-1]).startswith(b"fLaC"))

    def test_http10_chunkless(self):
        s = self.srv.raw_socket()
        req = http_request_bytes(self.srv.port, {"model": "fake-tts", "input": "a b",
                                                 "stream_format": "sse"}, "/v1/audio/speech")
        s.sendall(req.replace(b"HTTP/1.1", b"HTTP/1.0", 1))
        data = b""
        while True:
            b = s.recv(65536)
            if not b:
                break
            data += b
        s.close()
        head, _, body = data.partition(b"\r\n\r\n")
        self.assertIn(b"200", head.split(b"\r\n")[0])
        self.assertNotIn(b"chunked", head.lower())
        self.assertEqual([e["type"] for e in sse_events(body)][-1], "speech.audio.done")


class TestSpeechNoFfmpeg(unittest.TestCase):
    def test_encoded_format_needs_ffmpeg(self):
        srv = RunningServer({"fake-tts": FakeSpeechJob()}, ffmpeg=None)
        try:
            st, _, out = srv.request("POST", "/v1/audio/speech",
                                     {"model": "fake-tts", "input": "a", "response_format": "mp3"})
            self.assertEqual(st, 400)
            self.assertEqual(json.loads(out)["error"]["code"], "unsupported_format")
            st, _, _ = srv.request("POST", "/v1/audio/speech", {"input": "a", "response_format": "pcm"})
            self.assertEqual(st, 200)
        finally:
            srv.close()

    def test_unknown_length_wav_is_chunked(self):
        srv = RunningServer({"fake-tts": FakeSpeech()})          # a job without .samples
        try:
            st, hd, out = srv.request("POST", "/v1/audio/speech", {"input": "a b"})
            self.assertEqual(st, 200)
            self.assertEqual(hd.get("transfer-encoding"), "chunked")
            self.assertEqual(out[:4], b"RIFF")
            self.assertEqual(out[4:8], b"\xff\xff\xff\xff")
            self.assertEqual(out[44:], expected_pcm(2))
        finally:
            srv.close()


class TestSpeechQueueAndDisconnect(unittest.TestCase):
    def test_client_goes_away_mid_audio(self):
        tts = FakeSpeechJob(delay=0.1)
        srv = RunningServer({"fake-tts": tts})
        try:
            LOG.clear()
            s = srv.raw_socket()
            s.sendall(http_request_bytes(srv.port, {"model": "fake-tts", "input": "w " * 40},
                                         "/v1/audio/speech"))
            self.assertTrue(s.recv(100).startswith(b"HTTP/1.1 200"))
            s.close()
            self.assertTrue(wait_until(lambda: tts.stopped == 1, 10))
            self.assertTrue(wait_until(lambda: any(" 499 " in ln for ln in LOG), 5), LOG)
            self.assertFalse(srv.srv.fpga.busy)
        finally:
            srv.close()

    def test_speech_and_chat_share_the_fpga(self):
        tts = FakeSpeechJob(delay=0.05)
        srv = RunningServer({"echo": server_mod.EchoBackend(delay=0.02), "fake-tts": tts})
        spans, lock = [], threading.Lock()

        def run(fn):
            t0 = time.monotonic()
            st = fn()[0]
            with lock:
                spans.append((st, t0, time.monotonic()))
        try:
            ts = [threading.Thread(target=run, args=(lambda: srv.request(
                      "POST", "/v1/audio/speech", {"input": "a b c d"}),)),
                  threading.Thread(target=run, args=(lambda: srv.post(
                      {"model": "echo", "messages": [{"role": "user", "content": "one two three"}]}),))]
            for t in ts:
                t.start()
            for t in ts:
                t.join(30)
            self.assertEqual(sorted(s[0] for s in spans), [200, 200])
            self.assertFalse(srv.srv.fpga.busy)
            self.assertEqual(srv.srv.n_requests, 2)
        finally:
            srv.close()


if __name__ == "__main__":
    unittest.main()
