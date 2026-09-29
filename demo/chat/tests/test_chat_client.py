"""chat.py's audio: /v1/audio/speech streamed into a player (a recording
stand-in reading stdin, {rate} substituted), --say, -q --audio,
--audio-out, the REPL's /audio, /audio save and /say, the WAV fallback
without a player, and the text sent for speech."""

import contextlib
import io
import os
import shlex
import sys
import tempfile
import unittest
import wave
from unittest import mock

from _util import RunningServer, server_mod

import chat
from test_speech import FakeSpeechJob, expected_pcm

RECORDER = ("import sys, shutil; open(sys.argv[2], 'w').write(sys.argv[3]); "
            "shutil.copyfileobj(sys.stdin.buffer, open(sys.argv[1], 'wb'))")


def wav_frames(path):
    with wave.open(path) as w:
        return w.getframerate(), w.readframes(w.getnframes())


class ChatClientAudio(unittest.TestCase):
    def setUp(self):
        self.tts = FakeSpeechJob()
        self.srv = RunningServer({"echo": server_mod.EchoBackend(), "fake-tts": self.tts})
        self.url = f"http://127.0.0.1:{self.srv.port}/v1"
        self.dir = tempfile.TemporaryDirectory()
        self.out = os.path.join(self.dir.name, "played.pcm")
        self.rate = os.path.join(self.dir.name, "rate.txt")
        self.player = " ".join(shlex.quote(a) for a in (sys.executable, "-c", RECORDER, self.out, self.rate)) \
            + " {rate}"

    def tearDown(self):
        self.srv.close()
        self.dir.cleanup()

    def run_chat(self, argv, stdin=None):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), mock.patch.object(sys, "stdin", io.StringIO(stdin or "")):
            rc = chat.main(["--url", self.url] + argv)
        return rc, out.getvalue()

    def played(self):
        with open(self.out, "rb") as f:
            return f.read()

    def test_say_streams_into_the_player(self):
        rc, _ = self.run_chat(["--say", "one two three", "--player", self.player])
        self.assertEqual(rc, 0)
        self.assertEqual(self.played(), expected_pcm(3))
        with open(self.rate) as f:
            self.assertEqual(f.read(), "16000")
        req = self.tts.jobs[-1]
        self.assertEqual((req.model, req.response_format, req.input), ("fake-tts", "pcm", "one two three."))

    def test_audio_out_without_playing(self):
        wav = os.path.join(self.dir.name, "a.wav")
        rc, _ = self.run_chat(["--say", "a b", "--player", "none", "--audio-out", wav, "--speed", "1.5"])
        self.assertEqual(rc, 0)
        self.assertEqual(wav_frames(wav), (16000, expected_pcm(2)))
        self.assertFalse(os.path.exists(self.out))
        self.assertEqual(self.tts.jobs[-1].speed, 1.5)

    def test_question_read_aloud(self):
        rc, text = self.run_chat(["--model", "echo", "-q", "hello there world", "--audio",
                                  "--player", self.player])
        self.assertEqual(rc, 0)
        self.assertIn("hello there world", text)
        self.assertEqual(self.played(), expected_pcm(3))

    def test_repl_commands(self):
        wav = os.path.join(self.dir.name, "said.wav")
        script = (f"/say one two\n/audio save {wav}\n/audio\nfour five six seven\n/audio off\n"
                  "eight nine\n/say\n/audio bogus\n/quit\n")
        rc, text = self.run_chat(["--model", "echo", "--player", self.player, "-v"], script)
        self.assertEqual(rc, 0, text)
        self.assertEqual(wav_frames(wav), (16000, expected_pcm(2)))
        self.assertIn("answers read aloud: tts-1 via", text)
        self.assertIn("answers not read aloud", text)
        self.assertIn("usage: /audio", text)
        # spoken: "one two", the answer "four five six seven", then /say = the last answer "eight nine"
        self.assertEqual([r.input for r in self.tts.jobs], ["one two.", "four five six seven.", "eight nine."])
        self.assertEqual(self.played(), expected_pcm(2))
        self.assertEqual(text.count("[audio "), 3)

    def test_no_speech_model(self):
        srv = RunningServer({"echo": server_mod.EchoBackend()})
        try:
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                rc = chat.main(["--url", f"http://127.0.0.1:{srv.port}/v1", "--say", "hi", "--player", "none"])
            self.assertEqual(rc, 1)
            self.assertIn("404", err.getvalue())
        finally:
            srv.close()

    def test_without_a_player_a_wav_is_saved(self):
        with mock.patch.object(chat.shutil, "which", return_value=None), \
                mock.patch.object(chat.sys, "platform", "linux"):
            rc, text = self.run_chat(["--say", "a b c"])
        self.assertEqual(rc, 0)
        path = text.split("saved ")[1].split(")")[0]
        try:
            self.assertEqual(wav_frames(path), (16000, expected_pcm(3)))
        finally:
            os.unlink(path)


class SpeechText(unittest.TestCase):
    def test_markdown_and_pauses(self):
        t = "# Title\nHere is **bold** and `code`.\n- item one\n- item two\n```py\nx = 1\n```\nDone!"
        self.assertEqual(chat.speech_text(t), "Title. Here is bold and code. item one. item two. Done!")
        self.assertEqual(chat.speech_text("  \n "), "")

    def test_length_limit(self):
        s = chat.speech_text("A short sentence here. " * 400)
        self.assertLessEqual(len(s), chat.MAX_SPEECH)
        self.assertTrue(s.endswith("."))


if __name__ == "__main__":
    unittest.main()
