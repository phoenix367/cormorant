#!/usr/bin/env python3
"""
miranda_client.py — the console client of Miranda, the voice assistant on the
KV260 (Python 3 standard library only; runs on a laptop).

It connects to miranda_server.py on the board, streams the microphone to it
as RTP (UDP, 16 kHz L16), prints what the server hears and Miranda's answers
as they are written, and plays Miranda's speech.  Say "Hello Miranda" and
your request ("Hello Miranda, what is an FPGA?"), or "Hello Miranda", a pause,
then the request.

The dialogue mode: requests without "Hello Miranda", until a stretch of
silence (15 s by default, --dialogue-timeout) ends it.  Turn it on by voice —
"Hello Miranda, let's talk" (or "start a dialogue") — or by typing d and
Enter; turn it off by voice ("goodbye", "stop the dialogue") or with d again.
With --dialogue every "Hello Miranda" opens a dialogue.

The microphone is streamed only while the server listens: not while Miranda
thinks, and not while her speech plays (until the player has finished), so
she does not hear herself.

  miranda_client.py [--server HOST] [--port 8770] [--mic CMD] [--player CMD|none]
                    [--input FILE.wav] [--speed 1] [--save FILE.wav] [--turns N] [--quiet]
                    [--dialogue] [--dialogue-timeout S]

The microphone: the first recorder found — pw-record, parecord, arecord, or
ffmpeg from PulseAudio — writing raw s16le mono 16 kHz to stdout, or --mic
"CMD".  The speech: the first player found — pw-play, paplay, aplay, ffplay,
sox's play — reading raw s16le mono from stdin ({rate} in --player "CMD" is
replaced by the sample rate), or --player none.  --input FILE.wav streams a
recording instead of the microphone (at --speed times real time); --save
FILE.wav keeps Miranda's speech (FILE_2.wav for the second answer, ...);
--turns N exits after N answers.  Environment: MIRANDA_SERVER.
"""

from __future__ import annotations

import argparse
import os
import random
import shlex
import shutil
import socket
import subprocess
import sys
import threading
import time
import wave

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import miranda_proto as mp                                             # noqa: E402

COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
DIM, BOLD, RST = ("\033[2m", "\033[1m", "\033[0m") if COLOR else ("", "", "")

RECORDERS = (("pw-record", "pw-record --rate 16000 --channels 1 --format s16 -"),
             ("parecord", "parecord --raw --rate=16000 --channels=1 --format=s16le"),
             ("arecord", "arecord -q -t raw -f S16_LE -r 16000 -c 1 -"),
             ("ffmpeg", "ffmpeg -loglevel error -f pulse -i default -ac 1 -ar 16000 -f s16le -"))
PLAYERS = (("pw-play", "pw-play --rate {rate} --channels 1 --format s16 -"),
           ("paplay", "paplay --raw --rate={rate} --channels=1 --format=s16le"),
           ("aplay", "aplay -q -t raw -f S16_LE -r {rate} -c 1 -"),
           ("ffplay", "ffplay -nodisp -autoexit -loglevel quiet -f s16le -ar {rate} -ac 1 -i -"),
           ("play", "play -q -t raw -r {rate} -e signed -b 16 -c 1 -"))
FRAME_BYTES = 2 * mp.FRAME
TAIL_S = 0.25                       # after the speech: a moment before the microphone opens again


def first_command(table, override=None, rate=None):
    if override:
        return shlex.split(override.replace("{rate}", str(rate)))
    for name, tmpl in table:
        if shutil.which(name):
            return shlex.split(tmpl.format(rate=rate))
    return None


class Mic:
    """A recorder subprocess: 20 ms frames of s16le, in real time."""
    live = True

    def __init__(self, cmd):
        self.cmd = cmd
        self.p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def read(self):
        data = self.p.stdout.read(FRAME_BYTES)
        return data if data and len(data) == FRAME_BYTES else None

    def close(self):
        self.p.terminate()
        try:
            self.p.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.p.kill()


class FileSource:
    """A WAV file as the microphone: 16 kHz mono 16-bit (other formats through
    ffmpeg), 20 ms frames at `speed` times real time, then 1.5 s of silence
    (so the last segment ends)."""
    live = False

    def __init__(self, path, speed=1.0):
        self.data = self._load(path) + b"\0" * (mp.RATE * 3)
        self.pos, self.speed = 0, speed
        self.t0 = None

    @staticmethod
    def _load(path):
        try:
            with wave.open(path) as w:
                if (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, mp.RATE):
                    return w.readframes(w.getnframes())
        except (wave.Error, EOFError):
            pass
        if not shutil.which("ffmpeg"):
            raise SystemExit(f"{path}: not 16 kHz mono 16-bit WAV, and no ffmpeg to convert it")
        return subprocess.run(["ffmpeg", "-loglevel", "error", "-i", path, "-ac", "1", "-ar", str(mp.RATE),
                               "-f", "s16le", "-"], check=True, capture_output=True).stdout

    def read(self):
        if self.pos + FRAME_BYTES > len(self.data):
            return None
        if self.t0 is None:
            self.t0, self.sent = time.monotonic(), 0
        due = self.t0 + self.sent * mp.PTIME_MS / 1000.0 / self.speed
        delay = due - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        frame = self.data[self.pos:self.pos + FRAME_BYTES]
        self.pos += FRAME_BYTES
        self.sent += 1
        return frame

    def pause(self):
        """Muted: the file waits (its clock restarts on the next read)."""
        self.t0 = None

    def close(self):
        pass


class Client:
    def __init__(self, args):
        self.args = args
        self.state = "connecting"
        self.playing = False
        self.answers = 0
        self.stop = threading.Event()
        self.eof = threading.Event()
        self.last_event = time.monotonic()
        self.printed_query = None
        self.answer_open = False
        self.dialogue = False
        self.dialogue_timeout = None
        self.chan = None

    def can_send(self):
        return self.state in ("listening", "awake", "dialogue") and not self.playing

    def keyboard(self):
        """d + Enter toggles the dialogue mode, q + Enter quits (a terminal only)."""
        for line in sys.stdin:
            cmd = line.strip().lower()
            if cmd == "d" and self.chan is not None:
                try:
                    self.chan.send({"type": "dialogue", "on": not self.dialogue})
                except OSError:
                    return
            elif cmd in ("q", "quit", "exit"):
                self.stop.set()
                if self.chan is not None:
                    self.chan.close()
                return

    # ── the microphone -> RTP ──
    def sender(self, source, udp, dest, ssrc):
        seq, ts, marker = random.getrandbits(16), random.getrandbits(32), True
        while not self.stop.is_set():
            if not source.live and not self.can_send():
                source.pause()
                time.sleep(0.02)
                continue
            frame = source.read()
            if frame is None:
                self.eof.set()
                return
            if self.can_send():
                try:
                    udp.sendto(mp.rtp_pack(seq, ts, ssrc, mp.le_to_l16(frame), marker), dest)
                except OSError:
                    pass
                seq, marker = (seq + 1) & 0xFFFF, False
            else:
                marker = True                     # muted: the next packet opens a talkspurt
            ts = (ts + mp.FRAME) & 0xFFFFFFFF

    # ── output ──
    def say(self, text, style=""):
        if self.answer_open:
            print()
            self.answer_open = False
        print(f"{style}{text}{RST if style else ''}", flush=True)

    def run(self):
        a = self.args
        sock = socket.create_connection((a.server, a.port), timeout=15)
        sock.settimeout(None)
        chan = mp.Channel(sock)
        ssrc = random.getrandbits(32)
        hello = {"type": "hello", "version": mp.VERSION, "ssrc": ssrc, "codec": mp.CODEC,
                 "ptime": mp.PTIME_MS, "client": "miranda_client", "dialogue": "auto" if a.dialogue else "off"}
        if a.dialogue_timeout:
            hello["dialogue_timeout"] = a.dialogue_timeout
        chan.send(hello)
        self.chan = chan
        msg = chan.recv()
        if msg is None or msg[1].get("type") != "welcome":
            raise SystemExit(f"error: {msg[1].get('message') if msg else 'the server closed the connection'}")
        welcome = msg[1]
        wake = welcome.get("wake_phrase", "hello miranda")
        self.dialogue_timeout = welcome.get("dialogue_timeout")
        keys = sys.stdin.isatty()
        self.say(f"Miranda · {a.server}:{a.port} · say “{wake.title()}” and your request"
                 + (" · d + Enter: dialogue mode on / off" if keys else "") + " · Ctrl-C quits", DIM)
        if keys:
            threading.Thread(target=self.keyboard, daemon=True).start()
        if a.input:
            source = FileSource(a.input, a.speed)
        else:
            cmd = first_command(RECORDERS, a.mic)
            if not cmd:
                raise SystemExit("error: no recorder found (pw-record, parecord, arecord, ffmpeg): --mic \"CMD\"")
            source = Mic(cmd)
        udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        dest = (sock.getpeername()[0], int(welcome.get("rtp_port", a.port)))
        threading.Thread(target=self.sender, args=(source, udp, dest, ssrc), daemon=True).start()
        threading.Thread(target=self._watch_eof, args=(chan,), daemon=True).start()
        try:
            self.events(chan)
        finally:
            self.stop.set()
            source.close()
            try:
                chan.send({"type": "bye"})
            except OSError:
                pass
            chan.close()
            udp.close()
        return 0

    def _watch_eof(self, chan):
        """--input: after the file (and its silence), wait until the server is quiet, then leave."""
        while not self.stop.is_set():
            time.sleep(0.2)
            if self.eof.is_set() and self.state in ("listening", "awake", "dialogue") and not self.playing \
                    and time.monotonic() - self.last_event > self.args.linger:
                self.stop.set()
                chan.close()
                return

    def events(self, chan):
        a = self.args
        player, rate, pcm, saved = None, None, bytearray(), 0
        while not self.stop.is_set():
            try:
                msg = chan.recv()
            except (OSError, ValueError):
                break
            if msg is None:
                if not self.stop.is_set():
                    self.say("(the server closed the connection)", DIM)
                break
            self.last_event = time.monotonic()
            kind, obj = msg
            if kind == "audio":
                if player is not None:
                    try:
                        player.stdin.write(obj)
                    except (BrokenPipeError, OSError):
                        player = None
                pcm += obj
                continue
            t = obj.get("type")
            if t == "state":
                prev, self.state = self.state, obj["state"]
                if self.state == "warming":
                    self.say("(warming up the models on the board…)", DIM)
                elif self.state == "listening" and prev in ("warming", "speaking", "thinking", "awake", "dialogue"):
                    self.say("(listening)", DIM)
                elif self.state == "dialogue" and prev in ("speaking", "thinking", "listening", "awake"):
                    self.say("(your turn — dialogue mode)", DIM)
                elif self.state == "awake":
                    self.say("(listening for your request)", DIM)
            elif t == "heard":
                if obj.get("wake"):
                    self.say(f"You: {obj['text']}", BOLD)
                    self.printed_query = obj["text"]
                elif not a.quiet and obj.get("text"):
                    self.say(f"  · {obj['text']}", DIM)
            elif t == "query":
                if not (self.printed_query and obj["text"] in self.printed_query):
                    self.say(f"You: {obj['text']}", BOLD)
                self.printed_query = None
            elif t == "answer_delta":
                if not self.answer_open:
                    print(f"{BOLD}Miranda:{RST} ", end="", flush=True)
                    self.answer_open = True
                print(obj["text"], end="", flush=True)
            elif t == "answer_done":
                inf = obj.get("info", {})
                if inf.get("finish") == "fixed":                # Miranda's own reply ("Goodbye!"): no timings
                    if self.answer_open:
                        print(flush=True)
                        self.answer_open = False
                else:
                    self.say(f"  [first word {inf.get('first_token_ms', 0) / 1000:.1f} s · answer "
                             f"{inf.get('answer_ms', 0) / 1000:.1f} s · speech {inf.get('audio_s', 0):.1f} s "
                             f"from {inf.get('first_audio_ms', 0) / 1000:.1f} s]", DIM)
                if not self.playing:                  # no speech for this answer: the turn is over
                    self.answers += 1
                    if a.turns and self.answers >= a.turns:
                        break
            elif t == "audio_start":
                rate, pcm = int(obj["rate"]), bytearray()
                self.playing = True
                cmd = first_command(PLAYERS, a.player, rate) if a.player != "none" else None
                if cmd:
                    try:
                        player = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
                    except OSError as e:
                        self.say(f"(cannot start the player: {e})", DIM)
                        player = None
            elif t == "audio_end":
                if player is not None:
                    try:
                        player.stdin.close()
                    except OSError:
                        pass
                    player.wait()
                    player = None
                    time.sleep(TAIL_S)
                elif a.player != "none" and not a.save:
                    self.say("(no audio player found: pw-play, paplay, aplay, ffplay, play — or --save FILE.wav)",
                             DIM)
                if a.save and rate:
                    saved += 1
                    root, ext = os.path.splitext(a.save)
                    path = a.save if saved == 1 else f"{root}_{saved}{ext or '.wav'}"
                    with wave.open(path, "wb") as w:
                        w.setnchannels(1)
                        w.setsampwidth(2)
                        w.setframerate(rate)
                        w.writeframes(bytes(pcm))
                    self.say(f"  (saved {len(pcm) / 2 / rate:.1f} s to {path})", DIM)
                self.playing = False
                self.answers += 1
                chan.send({"type": "playback_done"})
                if a.turns and self.answers >= a.turns:
                    break
            elif t == "dialogue":
                self.dialogue = bool(obj.get("on"))
                timeout = obj.get("timeout") or self.dialogue_timeout or 15
                if self.dialogue:
                    self.say(f"(dialogue mode on: talk without “Hello Miranda”; it ends after {timeout:.0f} s "
                             f"of silence or when you say goodbye)", DIM)
                else:
                    why = {"silence": f"after {timeout:.0f} s of silence", "voice": "you said so",
                           "client": "from the keyboard"}.get(obj.get("reason"), obj.get("reason"))
                    self.say(f"(dialogue mode off: {why} — say “Hello Miranda” again)", DIM)
            elif t == "error":
                self.say(f"error: {obj.get('message')}")
        if player is not None:
            player.kill()


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default=os.environ.get("MIRANDA_SERVER", "localhost"),
                    help="the board running miranda_server.py (env MIRANDA_SERVER)")
    ap.add_argument("--port", type=int, default=mp.PORT)
    ap.add_argument("--mic", default=None, help="recorder command writing s16le mono 16 kHz to stdout")
    ap.add_argument("--player", default=os.environ.get("MIRANDA_PLAYER"),
                    help="player command reading s16le mono at {rate} from stdin, or none")
    ap.add_argument("--input", metavar="FILE.wav", help="stream a recording instead of the microphone")
    ap.add_argument("--speed", type=float, default=1.0, help="--input: times real time")
    ap.add_argument("--linger", type=float, default=4.0,
                    help="--input: seconds without news from the server before leaving")
    ap.add_argument("--save", metavar="FILE.wav", help="keep Miranda's speech")
    ap.add_argument("--turns", type=int, default=0, help="exit after N answers (0: never)")
    ap.add_argument("--quiet", action="store_true", help="do not print what was heard without the key phrase")
    ap.add_argument("--dialogue", action="store_true",
                    help="every “Hello Miranda” opens the dialogue mode (requests without the key phrase)")
    ap.add_argument("--dialogue-timeout", type=float, default=None,
                    help="seconds of silence that end the dialogue mode (default: the server's, 15)")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        return Client(args).run()
    except KeyboardInterrupt:
        print()
        return 0
    except OSError as e:
        print(f"error: cannot reach {args.server}:{args.port}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
