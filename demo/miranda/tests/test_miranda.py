"""Miranda on the host: the RTP and control formats (miranda_proto.py), the VAD
and its segmenter (vad.py), the key phrase (wake.py), the answer's sentences,
and whole sessions — miranda_client.py streaming a recording to
miranda_server.py, which reaches the KV260 chat server's API served here with
fake backends (kv260_chat_server.py: a scripted speech-to-text model, the echo
chat model, a fake text-to-speech model).  No board, no FPGA."""

import io
import json
import os
import socket
import struct
import sys
import tempfile
import threading
import unittest
import unittest.mock
import wave
from contextlib import redirect_stdout

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MIRANDA = os.path.dirname(HERE)
CHAT_TESTS = os.path.join(MIRANDA, "..", "chat", "tests")
for _p in (MIRANDA, CHAT_TESTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import miranda_client as mc                                            # noqa: E402
import miranda_proto as mp                                             # noqa: E402
import miranda_server as ms                                            # noqa: E402
from vad import Segmenter, Vad                                         # noqa: E402
from wake import WakePhrase                                            # noqa: E402

from _util import RunningServer, server_mod                            # noqa: E402
from chat_backend import Backend, Finish, Segment                     # noqa: E402
from test_speech import PIECE, FakeSpeechJob                           # noqa: E402

ms.log = lambda msg: None                                              # quiet sessions


def speech_burst(seconds, seed, level=6000.0):
    """Speech-like sound: band-limited noise and a few tones under a syllable-rate envelope."""
    rng = np.random.default_rng(seed)
    n = int(seconds * 16000)
    t = np.arange(n) / 16000.0
    w = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / 16000)
    w[(f < 300) | (f > 3000)] = 0
    x = np.fft.irfft(w, n)
    x = x / np.abs(x).max() + sum(0.5 * np.sin(2 * np.pi * f0 * t) for f0 in rng.uniform(150, 900, 3))
    env = 0.55 + 0.45 * np.sin(2 * np.pi * 4.0 * t)
    return x / np.abs(x).max() * env * level


def recording(pieces, noise=20.0, seed=0):
    """[("speech", seconds) | ("pause", seconds)] -> int16, with a little background noise."""
    rng = np.random.default_rng(seed)
    out = [speech_burst(s, seed + i) if k == "speech" else np.zeros(int(s * 16000))
           for i, (k, s) in enumerate(pieces)]
    x = np.concatenate(out)
    return np.clip(np.rint(x + rng.standard_normal(len(x)) * noise), -32768, 32767).astype(np.int16)


class TestRtp(unittest.TestCase):
    def test_pack_unpack(self):
        payload = bytes(range(10))
        pkt = mp.rtp_pack(65535, 2 ** 32 - 1, 0xDEADBEEF, payload, marker=True)
        self.assertEqual(mp.rtp_unpack(pkt), (96, True, 65535, 2 ** 32 - 1, 0xDEADBEEF, payload))
        # CSRCs, an extension and padding are skipped
        hdr = struct.pack("!BBHII", 0x80 | 0x20 | 0x10 | 2, 96, 7, 9, 1) + b"\0" * 8
        ext = struct.pack("!HH", 0xBEDE, 1) + b"\1\2\3\4"
        self.assertEqual(mp.rtp_unpack(hdr + ext + payload + b"\0\0\3")[5], payload)
        self.assertIsNone(mp.rtp_unpack(b"\x40" + b"\0" * 20))   # version 1
        self.assertIsNone(mp.rtp_unpack(b"\x80\x60"))

    def test_l16_is_big_endian(self):
        le = struct.pack("<hh", 1, -2)
        self.assertEqual(mp.le_to_l16(le), struct.pack(">hh", 1, -2))
        self.assertEqual(mp.l16_to_le(mp.le_to_l16(le)), le)

    def test_receiver(self):
        r = mp.RtpReceiver(max_fill=3)
        f = lambda v: struct.pack(">h", v) * 2                          # noqa: E731
        self.assertEqual(r.feed(65534, True, f(1)), (struct.pack("<h", 1) * 2, True))
        self.assertEqual(r.feed(65535, False, f(2))[0], struct.pack("<h", 2) * 2)
        self.assertEqual(r.feed(1, False, f(3))[0], b"\0" * 4 + struct.pack("<h", 3) * 2)   # 0 lost, wrapped
        self.assertEqual(r.feed(0, False, f(9)), (b"", False))          # late
        self.assertEqual(r.feed(100, False, f(4))[1], True)            # a jump restarts
        self.assertEqual((r.lost, r.late), (1, 1))

    def test_channel(self):
        a, b = socket.socketpair()
        ca, cb = mp.Channel(a), mp.Channel(b)
        ca.send({"type": "hello", "x": [1, 2]})
        ca.send_audio(b"\1\2\3\4")
        self.assertEqual(cb.recv(), ("json", {"type": "hello", "x": [1, 2]}))
        self.assertEqual(cb.recv(), ("audio", b"\1\2\3\4"))
        ca.close()
        self.assertIsNone(cb.recv())
        cb.close()


class TestVad(unittest.TestCase):
    def segments(self, x, **kw):
        s = Segmenter(**kw)
        out = s.feed(x)
        last = s.flush()
        return out + ([last] if last else [])

    def test_silence_and_noise_are_not_speech(self):
        for noise in (0.0, 20.0, 300.0):
            x = recording([("pause", 6.0)], noise=noise)
            self.assertEqual(self.segments(x), [], noise)

    def test_bursts(self):
        x = recording([("pause", 1.0), ("speech", 1.5), ("pause", 1.2), ("speech", 0.8), ("pause", 1.0)])
        segs = self.segments(x)
        self.assertEqual(len(segs), 2)
        for seg, (t0, t1) in zip(segs, ((1.0, 2.5), (3.7, 4.5)), strict=True):
            self.assertAlmostEqual(seg.start / 16000, t0 - 0.3, delta=0.12)      # the pre-roll
            self.assertAlmostEqual((seg.start + len(seg.pcm)) / 16000, t1 + 0.2, delta=0.15)   # the tail

    def test_short_pause_does_not_split_and_clicks_are_dropped(self):
        x = recording([("pause", 1.0), ("speech", 1.0), ("pause", 0.4), ("speech", 1.0), ("pause", 1.0),
                       ("speech", 0.1), ("pause", 1.0)])
        segs = self.segments(x)
        self.assertEqual(len(segs), 1)
        self.assertGreater(segs[0].seconds, 2.3)

    def test_leading_digital_silence_then_room_noise(self):
        """A recorder's first frames are zeros: the noise is learnt from the room, not
        from them (2026-10-17: 10-12 s segments of room noise at the start)."""
        rng = np.random.default_rng(5)
        for lead, db in ((0.5, -50), (1.0, -40), (3.0, -35)):
            x = rng.standard_normal(16000 * 15) * 32768 * 10 ** (db / 20)
            x[:int(lead * 16000)] = 0
            self.assertEqual(self.segments(np.clip(np.rint(x), -32768, 32767).astype(np.int16)), [], (lead, db))

    def test_rising_noise_floor_is_learnt(self):
        rng = np.random.default_rng(6)
        x = np.concatenate([rng.standard_normal(16000 * 5) * 32768 * 10 ** (-55 / 20),
                            rng.standard_normal(16000 * 10) * 32768 * 10 ** (-38 / 20)])
        segs = self.segments(np.clip(np.rint(x), -32768, 32767).astype(np.int16))
        self.assertLessEqual(sum(s.seconds for s in segs), 2.5)

    def test_noise_kept_across_turns(self):
        """After an answer the microphone resumes; speech in its first frames is speech,
        not the noise to learn (a full reset lost the whole utterance)."""
        rng = np.random.default_rng(3)
        level = 32768 * 10 ** (-50 / 20)
        room = np.clip(np.rint(rng.standard_normal(16000 * 2) * level), -32768, 32767).astype(np.int16)
        s = Segmenter()
        s.feed(room)
        s.reset(keep_noise=True)
        segs = s.feed(recording([("speech", 1.2), ("pause", 1.0)], noise=level, seed=4))
        self.assertEqual(len(segs), 1)
        self.assertGreater(segs[0].seconds, 1.2)

    def test_max_length(self):
        segs = self.segments(recording([("pause", 0.5), ("speech", 7.0), ("pause", 1.0)]), max_s=3.0)
        self.assertEqual([s.reason for s in segs][:2], ["max", "max"])

    def test_threshold_scores(self):
        v = Vad()
        sil = recording([("pause", 1.0)], noise=30.0)
        sp = speech_burst(0.5, 3).astype(np.int16)
        scores = [v.score(sil[i:i + 320]) for i in range(0, 16000, 320)]
        self.assertTrue(all(s < v.threshold for s in scores))
        self.assertGreater(np.mean([v.score(sp[i:i + 320]) for i in range(0, 8000, 320)]), 5 * v.threshold)


class TestWakeAndSentences(unittest.TestCase):
    def test_wake(self):
        w = WakePhrase("hello miranda")
        for text, rest in (("Hello, Miranda. What is the capital of France?", "What is the capital of France?"),
                           ("hello miranda", ""), ("Hallo Maranda, tell me a joke.", "tell me a joke."),
                           ("Hello Mi randa what time is it", "what time is it"),
                           ("So, hello Miranda! How are you?", "How are you?"), ("Hello, Mirinda.", ""),
                           ("Haloma Randa, what is the capital of France?", "what is the capital of France?"),
                           ("Allo Miranda, hi", "hi"), ("hello my randa tell me", "tell me"),
                           # a speaker whose greeting Moonshine mangles (2026-10-17, the board's log)
                           ("How old, Miranda? what is the weather to-day?", "what is the weather to-day?"),
                           ("Oh, Mjolanda, what is the weather today?", "what is the weather today?"),
                           ("How old, Muganda, what is the weather to-day?", "what is the weather to-day?"),
                           ("How woe, meogandha!", ""), ("Miranda, what time is it?", "what time is it?"),
                           ("Hello there Miranda", "")):
            m = w.find(text)
            self.assertIsNotNone(m, text)
            self.assertEqual(m.rest, rest, text)
        for text in ("hello world", "Hello Rhonda", "Hello Miriam", "Hello Veranda",
                     "Hello Linda", "Halloween Rhonda", "A panda is a bear", "On the veranda it is warm",
                     "In Rwanda people", "Mandarin is a language", "Hello,", "What is the weather today?"):
            self.assertIsNone(w.find(text), text)

    def test_dialogue_phrases(self):
        for t in ("let's talk", "Let us chat for a bit", "start a dialogue", "Begin the conversation",
                  "dialogue mode please", "can we have a conversation?"):
            self.assertTrue(ms.wants_dialogue(t), t)
        for t in ("talk to me about cats", "what is a dialogue?"):
            self.assertFalse(ms.wants_dialogue(t), t)
        for t in ("Goodbye!", "Bye bye, Miranda.", "OK, that's all, thank you.", "stop the dialogue",
                  "End this conversation please", "see you later", "Good night, Miranda", "Talk to you soon."):
            self.assertTrue(ms.ends_dialogue(t), t)
        for t in ("say goodbye to my friend", "Thank you.", "Good.", "bye the way the moon", "Is that all you know?"):
            self.assertFalse(ms.ends_dialogue(t), t)

    def test_sentences(self):
        s = ms.Sentences()
        out = []
        for piece in ("The **capital** of", " France is Paris. It", " is lovely!\n\n- A list item", " here"):
            out += s.feed(piece)
        out.append(s.flush())
        self.assertEqual(out, ["The capital of France is Paris.", "It is lovely!", "A list item here"])


class ScriptedAsr(Backend):
    """A speech-to-text model whose transcripts come from a script, one per
    segment with sound (silence — the server's warm-up — gives "").  An entry
    (text, margin, forced) answers the phrase check (kv260_wake_phrase) as
    MoonshineBackend does: forced when margin >= kv260_wake_margin."""

    model_id = "fake-asr"
    transcription = True
    sample_rate = 16000

    def __init__(self, script):
        self.script, self.calls = list(script), []

    def prepare_transcription(self, req):
        import speech_input
        pcm, _ = speech_input.decode_audio(req.audio)
        return pcm, req.raw.get("kv260_wake_phrase"), float(req.raw.get("kv260_wake_margin", -10))

    def transcribe(self, job, cancel):
        pcm, phrase, min_margin = job
        loud = len(pcm) and float(np.sqrt(np.mean(pcm.astype(np.float64) ** 2))) > 500
        item = self.script.pop(0) if loud and self.script else ""
        text, info = item, {}
        if isinstance(item, tuple):
            text, margin, forced = item
            if phrase:
                info["wake"] = {"phrase": phrase, "margin": margin, "complete": True,
                                "passed": margin >= min_margin, "min_margin": min_margin}
                text = forced if margin >= min_margin else text
        self.calls.append((len(pcm) / 16000, text, phrase))
        yield Segment(0.0, len(pcm) / 16000, text, [])
        yield Finish("stop", info={"text": text, "audio_s": len(pcm) / 16000, **info})


class Session(unittest.TestCase):
    def run_session(self, pieces, script, turns=0, server_args=(), client_args=()):
        asr, tts = ScriptedAsr(script), FakeSpeechJob()
        chat = RunningServer({"echo": server_mod.EchoBackend(), "fake-tts": tts, "fake-asr": asr})
        args = ms.parse_args(["--host", "127.0.0.1", "--port", "0", "--chat-url",
                              f"http://127.0.0.1:{chat.port}/v1", "--chat-model", "echo",
                              "--vad-hangover-ms", "500", *server_args])
        srv = ms.MirandaServer(args, ms.ChatApi(args.chat_url))
        threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
        d = tempfile.mkdtemp(prefix="miranda_test_")
        wav = os.path.join(d, "in.wav")
        with wave.open(wav, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(recording(pieces).tobytes())
        out = io.StringIO()
        sessions = []
        orig = ms.Session.__init__

        def keep(self_, *a, **k):
            orig(self_, *a, **k)
            sessions.append(self_)
        try:
            with unittest.mock.patch.object(ms.Session, "__init__", keep), redirect_stdout(out):
                rc = mc.main(["--server", "127.0.0.1", "--port", str(srv.server_address[1]), "--input", wav,
                              "--speed", "2", "--player", "none", "--save", os.path.join(d, "out.wav"),
                              "--linger", "1.5", *(["--turns", str(turns)] if turns else []), *client_args])
        finally:
            srv.shutdown()
            srv.server_close()
            chat.close()
        saved = sorted(f for f in os.listdir(d) if f.startswith("out"))
        frames = []
        for f in saved:
            with wave.open(os.path.join(d, f)) as w:
                frames.append(w.getnframes())
        return rc, out.getvalue(), asr, tts, frames, sessions

    def test_wake_with_request_and_without(self):
        rc, text, asr, tts, frames, sessions = self.run_session(
            [("pause", 1.0), ("speech", 1.2), ("pause", 1.2), ("speech", 1.6), ("pause", 1.5),
             ("speech", 0.8), ("pause", 1.2), ("speech", 1.0), ("pause", 1.5)],
            ["good morning everyone", "Hello, Miranda. What is two plus two?", "Hello Miranda.",
             "Tell me a joke"])
        self.assertEqual(rc, 0, text)
        self.assertIn("· good morning everyone", text)
        self.assertIn("You: Hello, Miranda. What is two plus two?", text)
        self.assertIn("Miranda: What is two plus two?", text)                # the echo model
        self.assertIn("Miranda: yes? (listening for your request)", text)
        self.assertIn("You: Tell me a joke", text)
        self.assertIn("Miranda: Tell me a joke", text)
        self.assertEqual(frames, [5 * PIECE, 4 * PIECE])                    # one fake piece per word
        self.assertEqual([j.input for j in tts.jobs[-2:]], ["What is two plus two?", "Tell me a joke"])
        # the chat model saw the first turn when it answered the second
        self.assertEqual(len(asr.calls), 1 + 4)                             # the warm-up + four segments
        s = sessions[0]
        self.assertEqual(s.turns, 2)
        self.assertLessEqual(s.dropped, 3)                                  # the client muted itself
        self.assertEqual([m["content"] for m in s.history],
                         ["What is two plus two?", "What is two plus two?", "Tell me a joke", "Tell me a joke"])

    def test_phrase_check(self):
        """The recognizer writes the key phrase badly; checked against the audio it
        holds (or not, below --wake-margin).  While awake the phrase said again
        (2026-10-17, the board's log: "Homey granda" answered as a request) waits
        for the request anew.  --save-segments keeps every segment."""
        seg_dir = tempfile.mkdtemp(prefix="miranda_segs_")
        rc, text, asr, tts, frames, sessions = self.run_session(
            [("pause", 1.0), ("speech", 1.2), ("pause", 1.2), ("speech", 1.6), ("pause", 1.5),
             ("speech", 0.8), ("pause", 1.2), ("speech", 0.8), ("pause", 1.2), ("speech", 1.0), ("pause", 1.5)],
            [("It's the weather today", -15.0, "Hello, Miranda, it's the weather today"),
             ("A warm Yuranda, what is two plus two?", -3.2, "Hello, Miranda, what is two plus two?"),
             ("Harew.", -2.0, "Hello, Miranda."), ("Homey granda", -4.0, "Hello, Miranda."), "Tell me a joke"],
            server_args=["--save-segments", seg_dir])
        self.assertEqual(rc, 0, text)
        self.assertIn("· It's the weather today", text)                    # -15 < -10: not the phrase
        self.assertIn("You: Hello, Miranda, what is two plus two?\nMiranda: what is two plus two?", text)
        self.assertIn("Miranda: yes? (listening for your request)", text)
        self.assertNotIn("granda", text)
        self.assertIn("You: Tell me a joke\nMiranda: Tell me a joke", text)
        self.assertEqual(sessions[0].turns, 2)
        self.assertEqual([c[2] for c in asr.calls[1:]], ["Hello Miranda"] * 5)
        lines = [json.loads(ln) for ln in open(os.path.join(seg_dir, "segments.jsonl"))]
        self.assertEqual([(ln["state"], ln["text"]) for ln in lines],
                         [("listening", "It's the weather today"),
                          ("listening", "Hello, Miranda, what is two plus two?"),
                          ("listening", "Hello, Miranda."), ("awake", "Hello, Miranda."),
                          ("awake", "Tell me a joke")])
        self.assertEqual([ln["wake"] and ln["wake"]["margin"] for ln in lines], [-15.0, -3.2, -2.0, -4.0, None])
        for ln in lines:
            with wave.open(os.path.join(seg_dir, ln["file"])) as w:
                self.assertAlmostEqual(w.getnframes() / 16000, ln["seconds"], delta=0.01)
        rc, text, asr, *_ = self.run_session(
            [("pause", 1.0), ("speech", 1.6), ("pause", 1.5)],
            [("A warm Yuranda, what is two plus two?", -3.2, "Hello, Miranda, what is two plus two?")],
            server_args=["--wake-margin", "-3"])
        self.assertIn("· A warm Yuranda, what is two plus two?", text)      # -3.2 < -3
        rc, text, asr, *_ = self.run_session(
            [("pause", 1.0), ("speech", 1.6), ("pause", 1.5)],
            [("A warm Yuranda, what is two plus two?", -3.2, "Hello, Miranda, what is two plus two?")],
            server_args=["--no-wake-check"])
        self.assertEqual(asr.calls[-1][2], None)
        self.assertIn("· A warm Yuranda, what is two plus two?", text)

    def test_answer_stops_after_three_sentences(self):
        rc, text, asr, tts, frames, sessions = self.run_session(
            [("pause", 1.0), ("speech", 1.6), ("pause", 1.5)],
            ["Hello Miranda, one. Two. Three. Four. Five six"])
        self.assertEqual(rc, 0, text)
        # the echo, three sentences spoken (the first may go alone, the rest batched); job 0 is the warm-up
        self.assertEqual(" ".join(j.input for j in tts.jobs[1:]), "one. Two. Three.")
        self.assertEqual(frames, [3 * PIECE])
        self.assertEqual(sessions[0].history[-1]["content"], "one. Two. Three.")

    def test_dialogue_by_voice(self):
        rc, text, asr, tts, frames, sessions = self.run_session(
            [("pause", 1.0), ("speech", 1.4), ("pause", 1.2), ("speech", 1.2), ("pause", 1.2), ("speech", 0.8),
             ("pause", 1.2), ("speech", 0.7), ("pause", 1.2), ("speech", 1.0), ("pause", 1.5)],
            ["Hello Miranda, let's talk.", "What is two plus two?", "And three?", "Goodbye.", "this is not for you"])
        self.assertEqual(rc, 0, text)
        self.assertIn("(dialogue mode on: talk without “Hello Miranda”", text)
        self.assertIn("Miranda: Sure, let's talk. I'm listening.", text)
        self.assertIn("You: What is two plus two?\nMiranda: What is two plus two?", text)   # no key phrase needed
        self.assertIn("You: And three?\nMiranda: And three?", text)
        self.assertIn("Miranda: Goodbye!", text)
        self.assertIn("(dialogue mode off: you said so", text)
        self.assertIn("· this is not for you", text)                         # heard, not answered
        s = sessions[0]
        self.assertFalse(s.dialogue)
        self.assertEqual(s.turns, 2)                                        # two answers; the replies are fixed
        self.assertEqual(len(frames), 4)
        self.assertLessEqual(s.dropped, 3)

    def test_dialogue_auto_ends_after_silence(self):
        rc, text, asr, tts, frames, sessions = self.run_session(
            [("pause", 1.0), ("speech", 1.4), ("pause", 1.2), ("speech", 0.8), ("pause", 4.0), ("speech", 1.0),
             ("pause", 1.5)],
            ["Hello Miranda, one more thing.", "two", "three"], client_args=["--dialogue", "--dialogue-timeout", "2"])
        self.assertEqual(rc, 0, text)
        self.assertIn("Miranda: one more thing.", text)
        self.assertIn("You: two\nMiranda: two", text)
        self.assertIn("(dialogue mode off: after 2 s of silence", text)
        self.assertIn("· three", text)
        self.assertEqual(sessions[0].turns, 2)

    def test_dialogue_from_the_client(self):
        chat = RunningServer({"echo": server_mod.EchoBackend(), "fake-tts": FakeSpeechJob(),
                              "fake-asr": ScriptedAsr([])})
        args = ms.parse_args(["--host", "127.0.0.1", "--port", "0", "--chat-url",
                              f"http://127.0.0.1:{chat.port}/v1", "--chat-model", "echo"])
        srv = ms.MirandaServer(args, ms.ChatApi(args.chat_url))
        threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
        try:
            ch = mp.Channel(socket.create_connection(("127.0.0.1", srv.server_address[1])))
            ch.send({"type": "hello", "version": 1, "ssrc": 7, "codec": mp.CODEC, "dialogue_timeout": 30})

            def until(pred):
                seen = []
                while True:
                    m = ch.recv()
                    self.assertIsNotNone(m, seen)
                    seen.append(m[1])
                    if pred(m[1]):
                        return seen
            self.assertEqual(until(lambda m: m.get("type") == "welcome")[-1]["dialogue_timeout"], 30)
            until(lambda m: m == {"type": "state", "state": "listening"})
            ch.send({"type": "dialogue", "on": True})
            seen = until(lambda m: m == {"type": "state", "state": "dialogue"})
            self.assertIn({"type": "dialogue", "on": True, "reason": "client", "timeout": 30.0}, seen)
            ch.send({"type": "dialogue", "on": False})
            until(lambda m: m == {"type": "state", "state": "listening"})
            ch.send({"type": "bye"})
            ch.close()
        finally:
            srv.shutdown()
            srv.server_close()
            chat.close()

    def test_awake_timeout(self):
        rc, text, asr, tts, frames, _ = self.run_session(
            [("pause", 1.0), ("speech", 0.8), ("pause", 4.0), ("speech", 1.0), ("pause", 1.5)],
            ["Hello Miranda", "this is not for you"], server_args=["--awake-timeout", "2"])
        self.assertEqual(rc, 0, text)
        self.assertIn("yes? (listening for your request)", text)
        self.assertIn("· this is not for you", text)                         # heard, but asleep again
        self.assertEqual(frames, [])


if __name__ == "__main__":
    unittest.main()
