"""Speech to text: speech_input.py (WAV forms, the resampler, ffmpeg), the
tokenizer.json detokenizer and MoonshineBackend of moonshine_backend.py over a
fake library, and POST /v1/audio/transcriptions of kv260_chat_server.py
(multipart forms, the response formats, SSE, aliases, OpenAI-style errors).
The real library's transcripts are gated by demo/asr/scripts/asr_host_emu.py
--lib-check (host) and asr_board.py (board)."""

import json
import math
import os
import shutil
import struct
import subprocess
import tempfile
import time
import unittest

from _util import CHAT, LOG, RunningServer, server_mod, sse_events, wait_until

from chat_backend import Cancelled, CancelToken, Finish, Segment, TranscriptionRequest

try:
    import numpy as np
except ImportError:                                            # pragma: no cover
    np = None

if np is not None:
    import moonshine_backend as mb
    import speech_input as si

FFMPEG = shutil.which("ffmpeg")


def wav_bytes(samples, rate=16000, channels=1, width=2, tag=1, extensible=False):
    """A WAV file of `samples` ([frames][channels] or [frames]) in the given encoding."""
    x = np.asarray(samples)
    if x.ndim == 1:
        x = x[:, None]
    if tag == 3:
        data = x.astype("<f4" if width == 4 else "<f8").tobytes()
    elif width == 1:
        data = (np.asarray(x, np.int64) + 128).astype(np.uint8).tobytes()
    elif width == 2:
        data = x.astype("<i2").tobytes()
    elif width == 3:
        v = np.asarray(x, np.int64).reshape(-1) & 0xFFFFFF
        data = np.stack([v & 0xFF, (v >> 8) & 0xFF, (v >> 16) & 0xFF], 1).astype(np.uint8).tobytes()
    else:
        data = x.astype("<i4").tobytes()
    align = width * channels
    if extensible:
        guid = struct.pack("<H", tag) + b"\x00\x00\x00\x00\x10\x00\x80\x00\x00\xaa\x00\x38\x9b\x71"
        fmt = struct.pack("<HHIIHHHHI", 0xFFFE, channels, rate, rate * align, align, 8 * width, 22,
                          8 * width, 0) + guid
    else:
        fmt = struct.pack("<HHIIHH", tag, channels, rate, rate * align, align, 8 * width)
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"LIST" + struct.pack("<I", 4) + b"INFO" \
        + b"data" + struct.pack("<I", len(data)) + data
    return b"RIFF" + struct.pack("<I", len(body)) + body


def tone(seconds, rate, freq=440.0, amp=8000.0):
    t = np.arange(int(seconds * rate)) / rate
    return amp * np.sin(2 * math.pi * freq * t)


@unittest.skipIf(np is None, "needs numpy")
class TestSpeechInput(unittest.TestCase):
    def test_16k_mono_pcm_passes_through(self):
        x = np.random.default_rng(1).integers(-32768, 32768, 16000).astype(np.int16)
        pcm, info = si.decode_audio(wav_bytes(x))
        self.assertTrue(np.array_equal(pcm, x))
        self.assertEqual(pcm.dtype, np.int16)
        self.assertEqual(info, {"source": "wav", "rate": 16000, "channels": 1})

    def test_stereo_is_mixed(self):
        x = np.random.default_rng(2).integers(-20000, 20000, (8000, 2))
        pcm, info = si.decode_audio(wav_bytes(x, channels=2))
        self.assertEqual(info["channels"], 2)
        self.assertTrue(np.array_equal(pcm, np.rint(x.mean(1)).astype(np.int16)))

    def test_encodings_agree(self):
        """8 / 24 / 32-bit PCM, float32 / 64 and the extensible forms decode to the
        16-bit PCM version (within the 8-bit file's resolution)."""
        x = np.rint(tone(0.2, 16000)).astype(np.int64)
        ref, _ = si.decode_audio(wav_bytes(x))
        for kw, scale, tol in (({"width": 3}, 256, 0), ({"width": 4}, 65536, 0),
                               ({"width": 4, "tag": 3}, 1 / 32768, 0), ({"width": 8, "tag": 3}, 1 / 32768, 0),
                               ({"width": 3, "extensible": True}, 256, 0),
                               ({"width": 4, "tag": 3, "extensible": True}, 1 / 32768, 0),
                               ({"width": 1}, 1 / 256, 128)):
            data = wav_bytes(x * scale if kw.get("tag") == 3 or scale >= 1 else np.rint(x * scale), **kw)
            pcm, _ = si.decode_audio(data)
            self.assertEqual(len(pcm), len(ref), kw)
            self.assertLessEqual(int(np.abs(pcm.astype(np.int64) - ref).max()), tol, kw)

    def test_resample_44k_stereo(self):
        """A 1 kHz tone at 44.1 kHz, stereo: 16 kHz out, the length and the tone kept."""
        sec, rate = 1.5, 44100
        a = tone(sec, rate, 1000.0)
        pcm, info = si.decode_audio(wav_bytes(np.rint(np.stack([a, a], 1)), rate=rate, channels=2))
        self.assertEqual(info["rate"], 44100)
        self.assertEqual(len(pcm), round(len(a) * 16000 / rate))
        want = tone(sec, 16000, 1000.0)[:len(pcm)]
        mid = slice(800, len(pcm) - 800)                       # away from the edges
        err = np.abs(pcm[mid].astype(np.float64) - want[mid]).max()
        self.assertLess(err, 8000 * 2e-3)

    def test_resample_rejects_above_nyquist(self):
        """A 12 kHz tone at 48 kHz is above the 8 kHz Nyquist rate of 16 kHz: gone."""
        pcm, _ = si.decode_audio(wav_bytes(np.rint(tone(0.5, 48000, 12000.0)), rate=48000))
        self.assertLess(np.abs(pcm[400:-400]).max(), 8000 * 1e-3)

    def test_upsample_8k(self):
        pcm, _ = si.decode_audio(wav_bytes(np.rint(tone(0.5, 8000, 500.0)), rate=8000))
        self.assertEqual(len(pcm), 8000)
        want = tone(0.5, 16000, 500.0)
        self.assertLess(np.abs(pcm[400:-400] - want[400:-400]).max(), 8000 * 2e-3)

    def test_errors(self):
        for data, msg in ((b"", "empty"), (b"RIFF\x04\x00\x00\x00WAVE", "fmt"),
                          (wav_bytes(np.zeros(4), width=2, tag=7), "not supported")):
            with self.assertRaises(si.AudioError) as cm:
                si.decode_audio(data)
            self.assertIn(msg, str(cm.exception))
        with self.assertRaises(si.AudioError) as cm:
            si.decode_audio(b"ID3\x04 an mp3", None)
        self.assertIn("only WAV", str(cm.exception))

    @unittest.skipUnless(FFMPEG, "needs ffmpeg")
    def test_ffmpeg_formats(self):
        x = np.rint(tone(1.0, 22050, 300.0)).astype(np.int16)
        src = wav_bytes(x, rate=22050)
        for fmt in ("flac", "ogg"):
            r = subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-i", "pipe:0", "-f", fmt, "pipe:1"],
                               input=src, capture_output=True, check=True)
            pcm, info = si.decode_audio(r.stdout, FFMPEG)
            self.assertEqual(info["source"], "ffmpeg")
            self.assertLess(abs(len(pcm) - 16000), 400, fmt)
        with self.assertRaises(si.AudioError):
            si.decode_audio(b"not audio at all", FFMPEG)


def tokenizer_file(d):
    """A small tokenizer.json in Moonshine's form: specials, byte fallback, ▁ pieces."""
    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2}
    for b in range(256):
        vocab[f"<0x{b:02X}>"] = 3 + b
    for p in ("▁Hello", "▁wor", "ld", "!", "▁", "▁▁two"):
        vocab[p] = len(vocab)
    added = [{"id": i, "content": c, "special": True} for c, i in (("<unk>", 0), ("<s>", 1), ("</s>", 2))]
    added.append({"id": len(vocab), "content": "<<ST_0>>", "special": True})
    doc = {"model": {"type": "BPE", "vocab": vocab, "merges": []}, "added_tokens": added,
           "decoder": {"type": "Sequence", "decoders": [
               {"type": "Replace", "pattern": {"String": "▁"}, "content": " "}, {"type": "ByteFallback"},
               {"type": "Fuse"}, {"type": "Strip", "content": " ", "start": 1, "stop": 0}]}}
    path = os.path.join(d, "tokenizer.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f)
    return path, vocab


@unittest.skipIf(np is None, "needs numpy")
class TestDetokenizer(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.path, self.v = tokenizer_file(self.td.name)
        self.dt = mb.Detokenizer(self.path)

    def tearDown(self):
        self.td.cleanup()

    def ids(self, *pieces):
        return [self.v[p] if p in self.v else p for p in pieces]

    def test_pieces_bytes_specials(self):
        v = self.v
        self.assertEqual(self.dt.decode(self.ids("<s>", "▁Hello", "▁wor", "ld", "!", "</s>")),
                         "Hello world!")
        e_acute = [v["<0xC3>"], v["<0xA9>"]]                    # é in UTF-8
        self.assertEqual(self.dt.decode([v["▁Hello"], v["▁"]] + e_acute), "Hello é")
        self.assertEqual(self.dt.decode([v["▁Hello"], v["<0xFF>"], v["!"]]), "Hello�!")
        self.assertEqual(self.dt.decode([v["▁▁two"]]), " two")       # one leading space stripped
        self.assertEqual(self.dt.decode([len(v), 99999, -1, v["!"]]), "!")    # specials, unknown ids skipped

    def test_rejects_other_tokenizers(self):
        path = os.path.join(self.td.name, "other.json")
        with open(path, "w") as f:
            json.dump({"model": {"type": "WordPiece", "vocab": {}}}, f)
        with self.assertRaises(ValueError):
            mb.Detokenizer(path)

    def test_matches_tokenizers_on_moonshine(self):
        """The real tokenizer.json against the tokenizers library, where both exist."""
        real = os.path.join(CHAT, "..", "asr", "assets", "moonshine-base", "tokenizer.json")
        try:
            from tokenizers import Tokenizer
        except ImportError:
            self.skipTest("needs the tokenizers package")
        if not os.path.exists(real):
            self.skipTest("needs demo/asr/assets/moonshine-base (asr_study.py fetch)")
        tok, dt = Tokenizer.from_file(real), mb.Detokenizer(real)
        rng = np.random.default_rng(0)
        texts = ["Hello, world!", "It's 3:45 pm — naïve café ünïcödé ✓ 東京", "  leading spaces"]
        cases = [tok.encode(t).ids for t in texts] + [list(rng.integers(0, 32768, 40)) for _ in range(30)]
        for ids in cases:
            self.assertEqual(dt.decode(ids), tok.decode([int(i) for i in ids], skip_special_tokens=True))


class FakeAsrEngine:
    """Parts of at most `max_samples`; a part's ids: [len(part) // 100 + 3]; open
    / close counted."""

    lib_path = "/x/lib/libmoonshine_base.so"
    sample_rate = 16000

    def __init__(self, max_samples=16000, delay=0.0):
        self.max_samples, self.delay = max_samples, delay
        self.opened = self.closed = 0
        self.lib = None

    def open(self):
        self.opened += 1
        self.lib = True

    def close(self):
        self.closed += 1
        self.lib = None

    def chunks(self, pcm):
        n, out, a = len(pcm), [], 0
        while a < n:
            out.append((a, min(self.max_samples, n - a)))
            a += out[-1][1]
        return out

    def transcribe(self, pcm):
        time.sleep(self.delay)
        return [len(pcm) // 100 + 3], {"encode_ms": 1.0, "decode_ms": 2.0, "steps": 2}


class FakeDetok:
    def decode(self, ids):
        return " ".join(f"w{i}" for i in ids)


def asr_backend(**kw):
    eng = FakeAsrEngine(**{k: kw.pop(k) for k in ("max_samples", "delay") if k in kw})
    return mb.MoonshineBackend(eng, "unused", detokenizer=FakeDetok(), **kw)


@unittest.skipIf(np is None, "needs numpy")
class TestMoonshineBackend(unittest.TestCase):
    def req(self, audio, **kw):
        return TranscriptionRequest(model="moonshine-base", audio=audio, **kw)

    def test_parts_and_finish(self):
        b = asr_backend()
        self.assertEqual(b.model_id, "moonshine-base")
        b.load()
        job = b.prepare_transcription(self.req(wav_bytes(np.zeros(40000, np.int16))))
        out = list(b.transcribe(job, CancelToken()))
        segs, fin = out[:-1], out[-1]
        self.assertEqual([(s.start, s.end, s.tokens) for s in segs],
                         [(0.0, 1.0, [163]), (1.0, 2.0, [163]), (2.0, 2.5, [83])])
        self.assertEqual([s.text for s in segs], ["w163", "w163", "w83"])
        self.assertIsInstance(fin, Finish)
        self.assertEqual(fin.info["text"], "w163 w163 w83")
        self.assertEqual((fin.info["parts"], fin.info["tokens"], fin.info["audio_s"]), (3, 3, 2.5))
        self.assertEqual(b.health()["requests"], 1)

    def test_cancel_between_parts(self):
        b = asr_backend()
        b.load()
        job = b.prepare_transcription(self.req(wav_bytes(np.zeros(48000, np.int16))))
        tok = CancelToken()
        gen = b.transcribe(job, tok)
        self.assertIsInstance(next(gen), Segment)
        tok.cancel()
        with self.assertRaises(Cancelled):
            next(gen)

    def test_rejections(self):
        b = asr_backend(max_seconds=1.0)
        from chat_backend import BackendError
        for req, code in ((self.req(wav_bytes(np.zeros(16000, np.int16)), language="de"), "unsupported_language"),
                          (self.req(wav_bytes(np.zeros(20000, np.int16))), "audio_too_long"),
                          (self.req(wav_bytes(np.zeros(0, np.int16))), "invalid_audio"),
                          (self.req(b"garbage"), "invalid_audio")):
            with self.assertRaises(BackendError) as cm:
                b.prepare_transcription(req)
            self.assertEqual(cm.exception.code, code)
        for lang in ("en", "EN", "english", None, ""):
            b.prepare_transcription(self.req(wav_bytes(np.zeros(160, np.int16)), language=lang))
        with self.assertRaises(BackendError):
            b.prepare(None)

    def test_model_id_from_the_library(self):
        eng = FakeAsrEngine()
        eng.lib_path = "/root/kv260_chat/lib/libmoonshine_tiny.so"
        self.assertEqual(mb.MoonshineBackend(eng, "x", detokenizer=FakeDetok()).model_id, "moonshine-tiny")
        self.assertEqual(mb.MoonshineBackend(eng, "x", detokenizer=FakeDetok(), model_id="m").model_id, "m")


def form(fields, files=(), boundary="b0undary"):
    """(content type, body) of a multipart/form-data request."""
    out = b""
    for k, v in fields.items():
        out += f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
    for k, (name, data) in files:
        out += (f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"; filename="{name}"\r\n'
                f"Content-Type: application/octet-stream\r\n\r\n").encode() + data + b"\r\n"
    return f"multipart/form-data; boundary={boundary}", out + f"--{boundary}--\r\n".encode()


@unittest.skipIf(np is None, "needs numpy")
class TestTranscriptionServer(unittest.TestCase):
    def setUp(self):
        LOG.clear()
        self.asr = asr_backend()
        self.srv = RunningServer({"echo": server_mod.EchoBackend(), "moonshine-base": self.asr})
        self.audio = wav_bytes(np.zeros(40000, np.int16))       # 2.5 s: three parts of the fake

    def tearDown(self):
        self.srv.close()

    def post(self, fields, files=None, ctype=None):
        files = [("file", ("speech.wav", self.audio))] if files is None else files
        ct, body = form(fields, files)
        return self.srv.request("POST", "/v1/audio/transcriptions", raw=body,
                                headers={"Content-Type": ctype or ct})

    def test_json_default_and_aliases(self):
        for model in (None, "moonshine-base", "whisper-1", "gpt-4o-transcribe", "gpt-4o-mini-transcribe"):
            st, hd, out = self.post({} if model is None else {"model": model})
            self.assertEqual(st, 200, out)
            self.assertEqual(hd["content-type"], "application/json")
            obj = json.loads(out)
            self.assertEqual(obj["text"], "w163 w163 w83")
            self.assertEqual(obj["usage"], {"type": "duration", "seconds": 3})
            self.assertEqual(obj["kv260"]["parts"], 3)
        self.assertTrue(wait_until(lambda: any("POST /v1/audio/transcriptions 200 model=moonshine-base" in ln
                                               and "audio=2.50s" in ln for ln in LOG)), LOG)
        # chat requests without a model still go to the chat backend
        st, _, out = self.srv.post({"messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(json.loads(out)["model"], "echo")

    def test_text_srt_vtt_verbose(self):
        st, hd, out = self.post({"response_format": "text"})
        self.assertEqual((st, hd["content-type"], out), (200, "text/plain; charset=utf-8", b"w163 w163 w83\n"))
        st, hd, out = self.post({"response_format": "srt"})
        self.assertEqual(out.decode(), "1\n00:00:00,000 --> 00:00:01,000\nw163\n\n"
                                       "2\n00:00:01,000 --> 00:00:02,000\nw163\n\n"
                                       "3\n00:00:02,000 --> 00:00:02,500\nw83\n")
        st, hd, out = self.post({"response_format": "vtt"})
        self.assertEqual(hd["content-type"], "text/vtt; charset=utf-8")
        self.assertTrue(out.decode().startswith("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nw163\n"))
        st, hd, out = self.post({"response_format": "verbose_json", "language": "en"})
        obj = json.loads(out)
        self.assertEqual((obj["task"], obj["language"], obj["duration"], obj["text"]),
                         ("transcribe", "english", 2.5, "w163 w163 w83"))
        self.assertEqual([(s["id"], s["start"], s["end"], s["tokens"]) for s in obj["segments"]],
                         [(0, 0.0, 1.0, [163]), (1, 1.0, 2.0, [163]), (2, 2.0, 2.5, [83])])

    def test_stream(self):
        st, hd, out = self.post({"stream": "true"})
        self.assertEqual(st, 200)
        self.assertTrue(hd["content-type"].startswith("text/event-stream"))
        ev = sse_events(out)
        self.assertEqual([e["type"] for e in ev], ["transcript.text.delta"] * 3 + ["transcript.text.done"])
        self.assertEqual("".join(e["delta"] for e in ev[:-1]), "w163 w163 w83")
        self.assertEqual(ev[-1]["text"], "w163 w163 w83")
        self.assertEqual(ev[-1]["usage"], {"type": "duration", "seconds": 3})

    def test_errors(self):
        cases = [({"model": "moonshine-base"}, [], 400, "missing_required_parameter"),
                 ({"model": "moonshine-base", "file": "not a file"}, [], 400, None),
                 ({"model": "nope"}, None, 404, "model_not_found"),
                 ({"model": "echo"}, None, 400, "model_not_supported"),
                 ({"response_format": "mp3"}, None, 400, None),
                 ({"temperature": "2"}, None, 400, "invalid_value"),
                 ({"temperature": "warm"}, None, 400, "invalid_type"),
                 ({"stream": "maybe"}, None, 400, None),
                 ({"language": "fr"}, None, 400, "unsupported_language"),
                 ({}, [("file", ("x.mp3", b"ID3 not wav"))], 400, "invalid_audio")]
        for fields, files, status, code in cases:
            st, _, out = self.post(fields, files)
            self.assertEqual(st, status, (fields, out))
            err = json.loads(out)["error"]
            self.assertEqual(set(err), {"message", "type", "param", "code"})
            if code:
                self.assertEqual(err["code"], code, fields)
        st, _, out = self.srv.request("POST", "/v1/audio/transcriptions", {"model": "moonshine-base"})
        self.assertEqual(st, 400)
        self.assertIn("multipart/form-data", json.loads(out)["error"]["message"])
        st, _, out = self.srv.post({"model": "moonshine-base", "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(st, 400)
        self.assertIn("/v1/audio/transcriptions", json.loads(out)["error"]["message"])

    def test_no_transcription_model(self):
        srv = RunningServer({"echo": server_mod.EchoBackend()})
        try:
            ct, body = form({"model": "whisper-1"}, [("file", ("a.wav", self.audio))])
            st, _, out = srv.request("POST", "/v1/audio/transcriptions", raw=body, headers={"Content-Type": ct})
            self.assertEqual(st, 404, out)
        finally:
            srv.close()

    def test_upload_limit(self):
        self.assertEqual(server_mod.MAX_UPLOAD, 32 << 20)
        args = server_mod.parse_args(["--backend", "echo"])
        self.assertEqual(args.asr_max_seconds, 600.0)


@unittest.skipIf(np is None, "needs numpy")
class TestTranscriptionQueue(unittest.TestCase):
    def test_client_goes_away_between_parts(self):
        asr = asr_backend(delay=0.2)
        srv = RunningServer({"moonshine-base": asr})
        try:
            ct, body = form({}, [("file", ("a.wav", wav_bytes(np.zeros(16000 * 6, np.int16))))])
            s = srv.raw_socket()
            s.sendall((f"POST /v1/audio/transcriptions HTTP/1.1\r\nHost: x\r\nContent-Type: {ct}\r\n"
                       f"Content-Length: {len(body)}\r\n\r\n").encode() + body)
            time.sleep(0.3)
            s.close()
            self.assertTrue(wait_until(lambda: any("499" in ln and "transcriptions" in ln for ln in LOG), 10), LOG)
            self.assertLess(asr.stats["requests"], 1)
        finally:
            srv.close()


if __name__ == "__main__":
    unittest.main()
