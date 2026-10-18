#!/usr/bin/env python3
# Copyright 2026 Ivan Gubochkin
# SPDX-License-Identifier: Apache-2.0

"""
miranda_server.py — Miranda, a voice assistant on the KV260 (runs on the
board; Python 3 standard library + numpy).

A client (miranda_client.py) opens a control connection (TCP) and streams its
microphone as RTP (UDP, L16 16 kHz; miranda_proto.py).  Per session:

  listening  the audio -> the VAD on the CPU (vad.py: speech segments) ->
             each segment transcribed (Moonshine on the FPGA) -> the key phrase
             (wake.py, "Hello Miranda"): the transcript after it is the request;
             the phrase alone gets a reply ("Hi! How can I help you?"; --greeting
             chat: the chat model's, off: none), then Miranda is awake: the next
             segment is the request (--awake-timeout seconds of audio).  Every transcription
             checks the phrase against the audio too (the chat server's
             kv260_wake_phrase: the decoder fed "Hello Miranda" and scored);
             when its margin is at least --wake-margin the transcript starts
             with the phrase — while awake or in a dialogue, the phrase said
             again is then not taken for the request
  dialogue   every segment is a request, no key phrase needed; entered by voice
             ("Hello Miranda, let's talk"), from the client (its "dialogue"
             message; with the hello's "dialogue": "auto" every key phrase opens
             one), left by voice ("goodbye", "stop the dialogue"), from the
             client, or after --dialogue-timeout seconds without speech
  thinking   the request -> the chat model, the answer streamed to the client
             as text; finished sentences -> text to speech (Piper on the FPGA).
             A weather question ("Will it rain tomorrow?", "What's the weather
             in Paris?" — WEATHER_INTENT; the chat models cannot tell one apart)
             runs the weather tool (weather.py: the board's public IP -> its
             position -> Open-Meteo's forecast, or a place it names) and its
             report goes into the answer's context
  speaking   the speech streamed to the client; the client mutes its
             microphone while it plays and says when it is done; then
             listening again (the audio meanwhile is dropped)

The FPGA work goes through the KV260 chat server (kv260_chat_server.py, its
OpenAI API at --chat-url): /v1/audio/transcriptions, /v1/chat/completions
(streamed) and /v1/audio/speech — the chat server owns the FPGA, queues the
requests and keeps the models resident.  It must serve a speech-to-text, a
chat and a text-to-speech model (moonshine, smollm2, piper).

usage: miranda_server.py [--host 0.0.0.0] [--port 8770] [--chat-url http://127.0.0.1:8000/v1]
                         [--api-key KEY | --api-key-file FILE] [--chat-model smollm2-360m-instruct]
                         [--wake "hello miranda"] [--wake-margin -10 | --no-wake-check]
                         [--greeting fixed|chat|off] [--no-weather] [--location PLACE]
                         [--save-segments DIR] [--max-tokens 100] [--max-sentences 3]
                         [--awake-timeout 8] [--dialogue-timeout 15]
                         [--vad-threshold 0.5] [--vad-hangover-ms 700] [--max-sessions 4]
"""

from __future__ import annotations

import argparse
import http.client
import io
import itertools
import json
import os
import queue
import re
import socket
import socketserver
import sys
import threading
import time
import traceback
import wave
from typing import Dict, Iterator, List, Optional, Tuple
from urllib.parse import urlsplit

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import miranda_proto as mp                                             # noqa: E402
from vad import Segmenter, Vad                                         # noqa: E402
from wake import WakePhrase                                            # noqa: E402
from weather import WEEKDAYS, Weather, WeatherError                    # noqa: E402

SYSTEM = ("You are Miranda, a friendly voice assistant. Your answers are spoken aloud, so answer "
          "briefly, in one to three short sentences of plain English, without lists, markdown or emojis.")


DIALOGUE_ON = re.compile(r"\b(?:let'?s|let us)\s+(?:talk|chat)\b"
                         r"|\b(?:start|begin|enter|open)\s+(?:a\s+|the\s+)?(?:dialog(?:ue)?|conversation)\b"
                         r"|\b(?:dialog(?:ue)?|conversation)\s+mode\b"
                         r"|\bhave\s+a\s+(?:conversation|chat|dialog(?:ue)?)\b", re.I)
DIALOGUE_OFF = re.compile(r"\b(?:stop|end|exit|finish|leave|close|quit)\s+(?:the\s+|this\s+|our\s+)?"
                          r"(?:dialog(?:ue)?|conversation|chat)\b", re.I)
GREETING = ("The user has only greeted you by name. Greet them back in one short sentence and ask "
            "how you can help.")         # --greeting chat: the chat model's note (SmolLM2-135M ignores it)
GREETINGS = ("Hi! How can I help you?", "Yes? What can I do for you?", "I'm here. What would you like to know?",
             "Hello! What can I do for you?")    # --greeting fixed (the default), in turn
TALK_PHRASE = "Let's talk"                # checked against the audio of a short request (_talk_check)
# The weather tool (weather.py): requests the harness takes for weather questions.  The chat models cannot
# decide it (2026-10-17, the board: asked "is this about the weather?" SmolLM2-360M said yes to "Write a song
# about snow" as much as to "Will it rain tomorrow?", SmolLM2-135M no to both), so these rules do.
_COND = r"(rain\w*|snow\w*|sunny|windy|cloudy|cold|hot|warm|chilly|freezing|storm\w*|thunder\w*|foggy|humid)"
_WHEN = (r"(?=\s*([?.!,]|$|(outside|out|today|tonight|tomorrow|now|later|again|soon|this|next|on|at|here|"
         r"day|days|morning|afternoon|evening|night|week|weekend|coming)\b|(?-i:in\s+[A-Z])))")   # not "a hot dog", "cold in space"
WEATHER_INTENT = re.compile(
    r"\b(weather|forecast)\b"
    r"|\b(will|is|was|are)\s+(it|there)\s+((going\s+to|gonna)\s+)?(be\s+)?((a|any|some|very|really|too|so)\s+)?"
    + _COND + _WHEN +
    r"|\bhow (hot|cold|warm|chilly)\b[^.?!]{0,20}\b(is it|will it|it is|outside|today|tonight|tomorrow)\b"
    r"|\btemperature\b[^.?!]{0,15}\b(outside|now|today|tonight|tomorrow|(?-i:in [A-Z]))"
    r"|\b(do|should|will|must)\s+(i|we|you)\s+(need|take|bring|wear|pack)\b[^.?!]{0,15}"
    r"\b(umbrella|coat|jacket|raincoat|sunscreen|sweater|scarf)\b",
    re.I)
COPY_SAMPLING = {"repetition_penalty": 1.0, "dry_multiplier": 0.0}   # an answer from tool data: greedy, and
#   no penalty on repeating the prompt's tokens (with the chat server's defaults SmolLM2-360M turned
#   "13 degrees (feels like 10)" into "10 degrees Celsius (34°F)", 2026-10-17)
YES_NO_WEATHER = re.compile(r"\b(rain\w*|umbrella|raincoat|snow\w*|drizzle)\b", re.I)   # Report.yes_no answers these
ANNOUNCE = ("You are Miranda, a voice assistant. To answer, you are about to look up {what}. Tell the user in one "
            "short sentence that you are checking it now. Do not guess the weather.")
ANNOUNCE_GUESS = re.compile(r"\d|\b(sunny|sun|clear|rain\w*|snow\w*|cloud\w*|storm\w*|thunder\w*|degree\w*|hot|cold|"
                            r"warm|chilly|cool|mild|breez\w*|wind\w*|beautiful|nice|lovely|likely|yes|no|not)\b", re.I)
DAY_SPELLINGS = re.compile(r"\bto-(morrow|day|night)\b", re.I)   # Moonshine writes "to-morrow"
GEAR = {"umbrella", "raincoat", "coat", "jacket", "sunscreen", "sweater", "scarf"}
NOT_WEATHER = re.compile(r"\b(joke|poem|song|story|haiku|rhyme|limerick)s?\b", re.I)
PLACE = re.compile(r"\b(?:in|for|at)\s+([A-Z][\w'-]+(?:\s+[A-Z][\w'-]+){0,2})")
NOT_PLACE = set(WEEKDAYS) | {"the", "today", "tomorrow", "tonight", "january", "february", "march", "april", "may",
                             "june", "july", "august", "september", "october", "november", "december"}
WAKE_MARGIN = -10.0                    # --wake-margin (nats; the chat server's kv260_wake_margin)
FILLERS = {"thank", "you", "thanks", "miranda", "ok", "okay", "well", "so", "then", "now", "for", "the", "help"}
BYE_WORDS = {"goodbye", "bye", "good", "see", "later", "thats", "that", "is", "all", "goodnight", "night",
             "talk", "to", "soon"}
BYE_KEYS = {"goodbye", "bye", "all", "later", "goodnight", "night", "soon"}


def weather_question(text: str) -> bool:
    """A request the weather tool answers (WEATHER_INTENT, not a joke / poem / song)."""
    return bool(WEATHER_INTENT.search(text)) and not NOT_WEATHER.search(text)


def weather_args(request: str) -> Tuple[str, str]:
    """(city, day) of a weather question: a capitalised place after "in" / "for" /
    "at" ("" — the board's own place), the day it names ("this weekend":
    Saturday), else "today" when it says today / tonight or asks what to take or
    wear ("Should I take my umbrella?"), "now" otherwise."""
    city = ""
    for m in PLACE.finditer(request):
        words = [w for w in m.group(1).split() if w.lower() not in NOT_PLACE]
        if words and words[0] == m.group(1).split()[0]:
            city = " ".join(words)
            break
    words = set(re.findall(r"[a-z]+", request.lower()))
    named = [d for d in ("tomorrow",) + WEEKDAYS if d in words]
    if named:
        return city, named[0]
    if "weekend" in words:
        return city, "saturday"
    if words & {"today", "tonight"} or words & GEAR:                   # what to take or wear: the day's forecast
        return city, "today"
    return city, "now"


def announcement(text: str, city: str) -> Tuple[str, bool]:
    """The chat model's announcement of a weather lookup, if it only announces:
    one sentence that says it checks / looks, without numbers or weather words
    (SmolLM2-360M goes on "It's a beautiful day with clear skies", "around 28
    degrees"); else the fixed one.  -> (the sentence, the model's?)"""
    t = " ".join(text.split()).strip(" .!?")
    if t and len(t.split()) <= 16 and re.search(r"\b(check\w*|look\w*)\b", t, re.I) and not ANNOUNCE_GUESS.search(t):
        return t + ".", True
    return f"Let me check the weather{f' in {city}' if city else ''}.", False


def wants_dialogue(text: str) -> bool:
    """A request to talk without the key phrase ("let's talk", "start a dialogue")."""
    return bool(DIALOGUE_ON.search(text))


def ends_dialogue(text: str) -> bool:
    """A request to stop: "stop the dialogue", or an utterance that is a goodbye
    ("Goodbye!", "OK, that's all, thank you", "Bye bye, Miranda")."""
    if DIALOGUE_OFF.search(text):
        return True
    words = {w for w in re.findall(r"[a-z]+", text.lower().replace("'", "")) if w not in FILLERS}
    return bool(words & BYE_KEYS) and words <= BYE_WORDS


def log(msg: str) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S ") + msg, file=sys.stderr, flush=True)


# ── the chat server's API ────────────────────────────────────────────────────

class ApiError(RuntimeError):
    pass


def wav_bytes(pcm) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(mp.RATE)
        w.writeframes(np.asarray(pcm, "<i2").tobytes())
    return buf.getvalue()


class ChatApi:
    """The KV260 chat server's OpenAI API (http.client: streamed responses)."""

    def __init__(self, url: str, api_key: Optional[str] = None, timeout: float = 300.0):
        u = urlsplit(url)
        self.host, self.port = u.hostname or "127.0.0.1", u.port or 80
        self.base = u.path.rstrip("/")
        self.api_key, self.timeout = api_key, timeout

    def _post(self, path: str, body: bytes, ctype: str) -> http.client.HTTPResponse:
        c = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
        h = {"Content-Type": ctype}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        try:
            c.request("POST", self.base + path, body=body, headers=h)
            r = c.getresponse()
        except OSError as e:
            c.close()
            raise ApiError(f"the chat server at {self.host}:{self.port} does not answer ({e})") from None
        if r.status != 200:
            data = r.read()
            c.close()
            try:
                msg = json.loads(data)["error"]["message"]
            except Exception:                                  # noqa: BLE001
                msg = data[:200].decode("utf-8", "replace")
            raise ApiError(f"{path}: HTTP {r.status}: {msg}")
        r._conn = c                                            # closed with the response
        return r

    def transcribe(self, pcm: np.ndarray, model: str, fields: Optional[Dict[str, str]] = None) -> Tuple[str, dict]:
        """The transcript of 16 kHz int16 samples (sent as WAV) and the response's
        "kv260" object; ``fields``: more form fields (kv260_wake_phrase ...)."""
        b = "miranda" + os.urandom(8).hex()
        body = "".join(f"--{b}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n"
                       for k, v in {"model": model, **(fields or {})}.items())
        body = (body + f"--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"segment.wav\"\r\n"
                f"Content-Type: audio/wav\r\n\r\n").encode() + wav_bytes(pcm) + f"\r\n--{b}--\r\n".encode()
        r = self._post("/audio/transcriptions", body, f"multipart/form-data; boundary={b}")
        try:
            obj = json.loads(r.read())
        finally:
            r._conn.close()
        return obj.get("text", ""), obj.get("kv260", {})

    def complete(self, messages: List[dict], model: str, max_tokens: int, **extra) -> str:
        """A non-streaming chat completion's text."""
        body = dict({"model": model, "messages": messages, "max_tokens": max_tokens}, **extra)
        r = self._post("/chat/completions", json.dumps(body).encode(), "application/json")
        try:
            return json.loads(r.read())["choices"][0]["message"].get("content") or ""
        finally:
            r._conn.close()

    def chat_stream(self, messages: List[dict], model: str, max_tokens: int,
                    temperature: Optional[float], meta: Optional[dict] = None,
                    extra: Optional[dict] = None) -> Iterator[str]:
        """The answer's text pieces as they are generated; meta["finish"]: the
        finish_reason; ``extra``: more request fields (sampling).  Closing the
        generator early closes the connection, and the chat server stops
        generating at the next token."""
        body = {"model": model, "messages": messages, "stream": True, "max_tokens": max_tokens}
        if temperature is not None:
            body["temperature"] = temperature
        body.update(extra or {})
        r = self._post("/chat/completions", json.dumps(body).encode(), "application/json")
        try:
            for line in r:
                line = line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                obj = json.loads(data)
                if "error" in obj:
                    raise ApiError(obj["error"].get("message", "error"))
                for ch in obj.get("choices") or []:
                    piece = (ch.get("delta") or {}).get("content")
                    if piece:
                        yield piece
                    if ch.get("finish_reason") and meta is not None:
                        meta["finish"] = ch["finish_reason"]
            try:
                r.read()                       # the chunked body's end: the server sees a clean close
            except (OSError, http.client.HTTPException):
                pass
        finally:
            r._conn.close()

    def speech_stream(self, text: str, model: str) -> Tuple[int, Iterator[bytes]]:
        """(sample rate, the s16le mono PCM pieces as they are synthesized)."""
        r = self._post("/audio/speech", json.dumps({"model": model, "input": text, "response_format": "pcm"}).encode(),
                       "application/json")
        rate = int(r.getheader("X-Sample-Rate") or 22050)

        def pieces():
            left = b""
            try:
                while True:
                    data = r.read1(16384) if hasattr(r, "read1") else r.read(16384)
                    if not data:
                        break
                    data = left + data
                    n = len(data) // 2 * 2
                    left = data[n:]
                    if n:
                        yield data[:n]
            finally:
                r._conn.close()
        return rate, pieces()


# ── the answer's sentences ───────────────────────────────────────────────────

class Sentences:
    """Streamed text -> whole sentences for text to speech (markdown marks out).

    A sentence ends at . ! ? … followed by space, or at a line break — not after
    a title or an initial ("Mr. Smith",
    "J. K. Rowling": 2026-10-17, "The owner of the house, Mr." was taken for the
    third sentence and the answer cut there), nor after an abbreviation before a
    lower-case word ("e.g. this", "5 p.m. today" — SmolLM2-360M starts sentences
    in lower case too: "...of our planet. it's home"), and a line that ends with a colon
    ("Here's my response:") runs on into the next.  The boundary is decided once
    the character after it has arrived; flush() gives what is left.

    feed_chunks(text, hurry): pieces of sentences too, at a clause boundary — a
    comma / semicolon / colon / dash, or before a joining word (and, but, which,
    who ... — once the word is whole), never at a comma between capitalised words
    ("Tbilisi, Georgia": 2026-10-18, a piece ended there and the next one came
    2.2 s later).  With ``hurry`` (the speech is about to run out) the first
    boundary after HURRY_WORDS / HURRY_JOIN words; without it only a sentence
    longer than LONG_WORDS / LONG_JOIN words is cut (Piper encodes a piece
    before it speaks: 0.9 s to the first audio for a 70-token sentence, 0.4 s
    for a short one)."""

    END = re.compile(r"(?<=[.!?…])[\"')\]]*\s+(?=\S)|\n+(?=\S)")
    MARKS = re.compile(r"[*_#`~]|^\s*(?:[-•]|\d+[.)])\s+", re.M)
    TITLES = {"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "mt", "messrs", "rev", "gen", "capt", "lt", "sgt"}
    ABBREV = {"etc", "vs", "approx", "incl", "esp", "ca"}   # a lower-case word after these goes on
    HURRY_WORDS, HURRY_JOIN = 3, 5
    LONG_WORDS, LONG_JOIN = 12, 16
    CLAUSE = re.compile(r"[,;:—–]\s+(?=\S)")
    JOIN = re.compile(r"\s+(?=(and|but|or|so|because|which|who|that|while|when|where|with|as|if|until)\s)")

    def __init__(self):
        self.buf = ""

    @classmethod
    def clean(cls, s: str) -> str:
        return re.sub(r"\s+", " ", cls.MARKS.sub("", s)).strip()

    def _ends(self, m) -> bool:
        before, after = self.buf[:m.start()], self.buf[m.end()]
        if "\n" in m.group():
            return not self.clean(before).endswith(":")
        word = re.search(r"([\w.]+)[.!?…][\"')\]]*$", before)
        w = word.group(1) if word else ""
        if before.rstrip("\"')]").endswith("."):
            if w.lower() in self.TITLES or re.fullmatch(r"[A-Z]", w):
                return False
            if "." in w or w.lower() in self.ABBREV:        # "e.g. this", "5 p.m. today", "etc. and"
                return not after.islower()
        return True

    def feed_chunks(self, text: str, hurry: bool = False) -> List[Tuple[str, bool]]:
        """(chunk, ends a sentence) of streamed text: sentences and pieces of them."""
        out = [(c, True) for c in self.feed(text)]
        limits = (self.HURRY_WORDS, self.HURRY_JOIN) if hurry else (self.LONG_WORDS, self.LONG_JOIN)
        for rx, words in zip((self.CLAUSE, self.JOIN), limits):
            m = next((c for c in rx.finditer(self.buf)
                      if len(self.buf[:c.start()].split()) >= words and not self._name_comma(c)), None)
            if m:
                out.append((self.clean(self.buf[:m.end()]), False))
                self.buf = self.buf[m.end():]
                break
        return out

    def _name_comma(self, m) -> bool:
        """A comma inside a name: "Tbilisi, Georgia", "Paris, France"."""
        before = re.search(r"(\w+)\W*$", self.buf[:m.start() + 1])
        after = re.match(r"\W*(\w+)", self.buf[m.end():])
        return bool(self.buf[m.start()] == "," and before and after and before.group(1)[0].isupper()
                    and after.group(1)[0].isupper())

    def feed(self, text: str) -> List[str]:
        self.buf += text
        out, pos = [], 0
        while True:
            m = self.END.search(self.buf, pos)
            if not m:
                break
            if not self._ends(m):
                pos = m.end()
                continue
            s = self.clean(self.buf[:m.end()])
            self.buf, pos = self.buf[m.end():], 0
            if s:
                out.append(s)
        return out

    def flush(self) -> Optional[str]:
        s = self.clean(self.buf)
        self.buf = ""
        return s or None


WORDS_PER_S = 4.0                      # Piper's speech before a turn has measured its own (~51 words in 12.6 s)
AHEAD_S = 2.0                          # less speech ahead than this: the next clause goes to speech at once
#   (Piper's first audio ~0.45 s + a clause written, ~1 s)


def speech_ahead(spoken: dict) -> float:
    """Seconds of speech still ahead of the client's player: audio sent and not
    played, the text being synthesised, the text not yet taken by the worker
    (text at the turn's own words per second once its first request has
    finished, else WORDS_PER_S)."""
    now = time.monotonic()
    play = max(0.0, spoken["playback_end"] - now) if spoken["playback_end"] else 0.0
    wps = spoken["wps"] or WORDS_PER_S
    return play + spoken["batch_left"] + (spoken["words_out"] - spoken["words_started"]) / wps


def new_spoken() -> dict:
    """A turn's speech bookkeeping: samples / rate / first_ms sent, and the gaps —
    playback_end is when the client's player runs dry, gaps the silences between
    pieces (a piece's first audio after it), words_out / words_started the text
    given to / taken by the speech worker (Session.respond's pacing)."""
    return {"samples": 0, "rate": None, "first_ms": None, "error": None, "playback_end": None, "gaps": [],
            "requests": 0, "words_out": 0, "words_started": 0, "batch_left": 0.0, "wps": None, "gap_at": [],
            "words_done": 0, "audio_done": 0.0}


# ── a session ────────────────────────────────────────────────────────────────

LISTENING_STATES = ("listening", "awake", "dialogue")


class Session:
    ids = itertools.count(1)

    def __init__(self, server: "MirandaServer", chan: mp.Channel, peer_ip: str, hello: dict):
        self.server, self.chan, self.peer_ip = server, chan, peer_ip
        self.args = server.args
        self.id = next(self.ids)
        self.ssrc = int(hello["ssrc"]) & 0xFFFFFFFF
        self.state = "warming"
        self.q: "queue.Queue[Optional[tuple]]" = queue.Queue(maxsize=1000)
        self.rtp = mp.RtpReceiver()
        self.seg = Segmenter(vad=Vad(threshold=self.args.vad_threshold),
                             hangover=max(1, self.args.vad_hangover_ms // mp.PTIME_MS))
        self.wake = WakePhrase(self.args.wake)
        self.history: List[dict] = []
        self.awake_until: Optional[int] = None
        self.greeted = False                    # the last turn answered the key phrase alone
        self.greetings = itertools.cycle(GREETINGS)
        self.dialogue = False                   # the dialogue mode: requests without the key phrase
        self.dialogue_auto = hello.get("dialogue") == "auto"
        self.dialogue_timeout = min(max(float(hello.get("dialogue_timeout") or self.args.dialogue_timeout), 2.0),
                                    600.0)
        self.dialogue_until = 0
        self.playback_done = threading.Event()
        self.closed = threading.Event()
        self.turns = 0
        self.epoch = 0                          # bumped when listening resumes: older segments are stale
        self.dropped = 0                        # RTP packets that arrived while not listening

    # control
    def send(self, obj: dict) -> None:
        if not self.chan.closed:
            try:
                self.chan.send(obj)
            except OSError:
                self.close()

    def set_state(self, state: str) -> None:
        self.state = state
        self.send({"type": "state", "state": state})

    def close(self) -> None:
        if not self.closed.is_set():
            self.closed.set()
            self.playback_done.set()
            try:
                self.q.put_nowait(None)
            except queue.Full:
                pass

    # audio in (the UDP thread)
    def on_rtp(self, seq: int, marker: bool, payload: bytes) -> None:
        if self.state in LISTENING_STATES:
            try:
                self.q.put_nowait((seq, marker, payload))
            except queue.Full:
                self.dropped += 1
        else:
            self.dropped += 1                  # the client should be muted now

    def _drain(self) -> None:
        while True:
            try:
                self.q.get_nowait()
            except queue.Empty:
                return

    def listen(self) -> None:
        """Back to listening (in the dialogue mode: for the next request): the
        audio received meanwhile and the open segment dropped; the VAD keeps
        the room's noise (the first frames after the pause may be speech)."""
        self._drain()
        self.seg.reset(keep_noise=True)
        self.rtp = mp.RtpReceiver()
        self.awake_until = None
        self.epoch += 1
        if self.dialogue:
            self.dialogue_until = self.seg.pos + int(self.dialogue_timeout * mp.RATE)
            self.set_state("dialogue")
        elif self.greeted:                     # she greeted back: the request comes without the key phrase
            self.awake_until = self.seg.pos + int(self.args.awake_timeout * mp.RATE)
            self.set_state("awake")
        else:
            self.set_state("listening")
        self.greeted = False

    def set_dialogue(self, on: bool, reason: str) -> None:
        """Switch the dialogue mode (any thread); the session's worker moves the
        state between listening and dialogue (sync_dialogue)."""
        if on != self.dialogue:
            self.dialogue = on
            self.send({"type": "dialogue", "on": on, "reason": reason, "timeout": self.dialogue_timeout})
            log(f"[{self.id}] dialogue {'on' if on else 'off'} ({reason})")

    def sync_dialogue(self) -> None:
        if self.state in ("listening", "awake") and self.dialogue:
            self.awake_until = None
            self.dialogue_until = self.seg.pos + int(self.dialogue_timeout * mp.RATE)
            self.set_state("dialogue")
        elif self.state == "dialogue" and not self.dialogue:
            self.set_state("listening")

    def run(self) -> None:
        """The session's worker: warm-up, then the audio -> segments -> turns loop."""
        try:
            self.warm_up()
            self.listen()
            while not self.closed.is_set():
                self.sync_dialogue()
                try:
                    item = self.q.get(timeout=0.5)
                except queue.Empty:
                    continue
                if item is None:
                    break
                if self.state not in LISTENING_STATES:
                    continue
                pcm, _restart = self.rtp.feed(*item)
                epoch = self.epoch
                for s in self.seg.feed(np.frombuffer(pcm, "<i2")):
                    self.on_segment(s)
                    if self.closed.is_set() or self.epoch != epoch:
                        break
                if self.state == "awake" and self.awake_until is not None and not self.seg.active \
                        and self.seg.pos > self.awake_until:
                    log(f"[{self.id}] awake timeout")
                    self.awake_until = None
                    self.set_state("listening")
                if self.state == "dialogue" and not self.seg.active and self.seg.pos > self.dialogue_until:
                    self.set_dialogue(False, "silence")
                    self.set_state("listening")
        except Exception as e:                                 # noqa: BLE001
            traceback.print_exc()
            self.send({"type": "error", "message": f"session failed: {e}"})
        finally:
            self.close()

    def warm_up(self) -> None:
        """One small request per model: each is loaded before the first turn."""
        self.set_state("warming")
        api, a = self.server.api, self.args
        t0 = time.monotonic()
        for what, call in (("speech to text", lambda: api.transcribe(np.zeros(8000, np.int16), a.asr_model)),
                           ("chat", lambda: list(api.chat_stream([{"role": "user", "content": "Hi"}],
                                                                 a.chat_model, 1, 0.0))),
                           ("text to speech", lambda: [None for _ in api.speech_stream("Hi.", a.tts_model)[1]])):
            try:
                call()
            except ApiError as e:
                self.send({"type": "error", "message": f"{what}: {e}"})
                log(f"[{self.id}] warm-up {what}: {e}")
        log(f"[{self.id}] warm-up {time.monotonic() - t0:.1f} s")

    # turns
    def _phrase(self) -> str:
        """The key phrase as the recognizer writes it: "Hello Miranda"."""
        return " ".join(w.capitalize() for w in self.args.wake.split())

    def _talk_check(self, s, text: str, req: str, prefix: str, note: str) -> Tuple[str, str]:
        """A short request that is no command may be "let's talk" misheard ("It is
        stock", "a little stoke", "red stalk": 2026-10-17, the board's log): the
        segment transcribed again with TALK_PHRASE (after the key phrase, when it
        was heard) checked against the audio.  -> (the transcript, the log note)."""
        a = self.args
        if not a.wake_check or not req or len(req.split()) > 3 or wants_dialogue(req) or ends_dialogue(req):
            return text, note
        phrase = f"{prefix} {TALK_PHRASE[0].lower()}{TALK_PHRASE[1:]}" if prefix else TALK_PHRASE
        try:
            t2, info = self.server.api.transcribe(s.pcm, a.asr_model, {"kv260_wake_phrase": phrase,
                                                                       "kv260_wake_margin": str(a.wake_margin)})
        except ApiError as e:
            log(f"[{self.id}] transcription failed: {e}")
            return text, note
        wk = info.get("wake") or {}
        if "margin" not in wk:
            return text, note
        note += f", {TALK_PHRASE!r} {wk['margin']:+.2f} {'kept' if wk.get('passed') else 'no'}"
        return (t2.strip(), note) if wk.get("passed") else (text, note)

    def on_segment(self, s) -> None:
        t0 = time.monotonic()
        a = self.args
        check = a.wake_check                   # awake / dialogue too: a repeated key phrase is not a request
        fields = {"kv260_wake_phrase": self._phrase(), "kv260_wake_margin": str(a.wake_margin)} if check else None
        try:
            text, info = self.server.api.transcribe(s.pcm, a.asr_model, fields)
        except ApiError as e:
            self.send({"type": "error", "message": str(e)})
            log(f"[{self.id}] transcription failed: {e}")
            return
        text = text.strip()
        asr_ms = (time.monotonic() - t0) * 1000.0
        wk = info.get("wake") if check else None
        note = ""
        if wk:
            note = (f", phrase {wk['margin']:+.2f} {'kept' if wk.get('passed') else 'no'}" if "margin" in wk
                    else f", phrase not checked: {wk.get('error')}")
        self.server.save_segment(self, s, text, wk)
        words = re.compile(r"[A-Za-z0-9]")
        if self.state in ("awake", "dialogue"):
            m = self.wake.find(text) if text else None
            req = m.rest if m is not None and m.start <= 3 else text       # a key phrase in front is not the request
            if self.state == "awake":
                text, note = self._talk_check(s, text, req, "" if m is None else self._phrase(), note)
                m = self.wake.find(text) if text else None
                req = m.rest if m is not None and m.start <= 3 else text
            log(f"[{self.id}] request ({s.seconds:.1f} s, asr {asr_ms:.0f} ms{note}, {self.state}): {text!r}")
            if not words.search(req):
                if self.state == "dialogue" and text:                          # the key phrase alone: still here
                    self.dialogue_until = self.seg.pos + int(self.dialogue_timeout * mp.RATE)
                elif self.state == "awake" and m is not None:                  # said again: wait for the request anew
                    self.awake_until = self.seg.pos + int(a.awake_timeout * mp.RATE)
                return
            if self.state == "dialogue" and ends_dialogue(req):
                self.say("Goodbye!", query=req, dialogue=False)
            elif wants_dialogue(req):
                self.say("Sure, let's talk. I'm listening.", query=req, dialogue=True)
            else:
                self.answer(req)
            return
        m = self.wake.find(text) if text else None
        if m is not None:
            text, note = self._talk_check(s, text, m.rest, self._phrase(), note)
            m = self.wake.find(text)
        self.send({"type": "heard", "text": text, "wake": m is not None})
        log(f"[{self.id}] heard ({s.seconds:.1f} s, asr {asr_ms:.0f} ms{note}){' WAKE' if m else ''}: {text!r}")
        if m is None:
            return
        if wants_dialogue(m.rest):
            self.say("Sure, let's talk. I'm listening.", query=m.rest, dialogue=True)
            return
        if self.dialogue_auto:
            self.set_dialogue(True, "auto")
        if words.search(m.rest):
            self.answer(m.rest)
        elif a.greeting == "fixed":            # the key phrase alone: she answers, then waits for the request
            self.say(next(self.greetings), query=text, greeting=True)
        elif a.greeting == "chat":
            self.respond(self._phrase() + "!", shown=text, greeting=True)
        elif self.dialogue:
            self.sync_dialogue()
        else:
            self.awake_until = self.seg.pos + int(a.awake_timeout * mp.RATE)
            self.set_state("awake")

    def answer(self, req: str) -> None:
        """A request: a weather question runs the weather tool (weather.py) — the chat
        model announces the lookup, then answers from its report (rain / umbrella /
        snow questions: a composed answer); anything else goes to the chat model alone."""
        heard, req = req, DAY_SPELLINGS.sub(lambda m: "to" + m.group(1).lower(), req)
        if not self.args.weather or not weather_question(req):
            return self.respond(req, shown=heard)
        self.respond(req, shown=heard, tool=lambda emit: self._weather(req, emit))

    def _weather(self, req: str, emit) -> Tuple[Optional[str], Optional[str]]:
        """The weather tool inside a turn: the lookup beside the announcement (emit()
        speaks it at once) -> (the answer's context, or a fixed answer)."""
        a, w = self.args, self.server.weather
        t0 = time.monotonic()
        city, day = weather_args(req)
        res: Dict[str, object] = {}

        def lookup():
            try:
                place = w.geocode(city) if city else self.server.home()
                res["place"] = place
                if place is not None:
                    res["report"] = w.report(place, day)
            except (WeatherError, OSError, ValueError, KeyError, TypeError) as e:
                res["error"] = e
        th = threading.Thread(target=lookup, daemon=True)
        th.start()
        if a.announce:
            what = ({"now": "the current weather", "today": "today's weather", "tomorrow": "tomorrow's weather"}
                    .get(day, f"the forecast for {day.capitalize()}") + (f" in {city}" if city else ""))
            try:
                said = self.server.api.complete(
                    [{"role": "system", "content": ANNOUNCE.format(what=what)}, {"role": "user", "content": req}],
                    a.chat_model, 24, temperature=0, stop=[".", "!", "?", "\n"], **COPY_SAMPLING)
            except ApiError as e:
                log(f"[{self.id}] announcement failed: {e}")
                said = ""
            text, own = announcement(said, city)
            log(f"[{self.id}] announce ({(time.monotonic() - t0) * 1000.0:.0f} ms, "
                f"{'the model' if own else f'fixed, the model said {said!r}'}): {text!r}")
            emit(text)
        th.join()
        if "error" in res:
            log(f"[{self.id}] weather: {res['error']}")
            return None, "I can't reach the weather service right now."
        if res.get("place") is None:
            return None, f"I could not find {city}."
        report, place = res["report"], res["place"]
        log(f"[{self.id}] weather(city={city!r}, day={day!r}) -> {place.label()} ({place.source}) "
            f"{(time.monotonic() - t0) * 1000.0:.0f} ms: {report.sentence()}")
        if YES_NO_WEATHER.search(req):         # the chat models answer these "yes" whatever the data say
            return None, report.yes_no(req)
        return (f"Weather data: {report.sentence()} Answer the question from these data in one or two short "
                f"sentences, with the numbers."), None

    def respond(self, query: str, shown: Optional[str] = None, greeting: bool = False,
                context: Optional[str] = None, tool=None) -> None:
        """A turn: the query (``shown``: as the client shows it) -> the chat model
        -> the answer as text and speech.  ``greeting``: the key phrase alone —
        the model greets back (GREETING, at most two sentences) and the next
        segment is the request.  ``tool(emit)``: a tool step first — emit(text)
        says a sentence at once; it returns (the answer's context, or a fixed
        answer that replaces the model's)."""
        a, api = self.args, self.server.api
        t0 = time.monotonic()
        self.turns += 1
        self.send({"type": "query", "text": shown or query})
        self.set_state("thinking")
        self.playback_done.clear()
        tts_q: "queue.Queue[Optional[str]]" = queue.Queue()
        spoken = new_spoken()
        worker = threading.Thread(target=self._speak, args=(tts_q, spoken, t0), daemon=True)
        worker.start()
        split, parts, said, first_ms, meta = Sentences(), [], [], None, {}
        n_sent = 0                              # whole sentences said (--max-sentences)
        pre: List[str] = []                     # the tool step's sentences (announcement, fixed answer)

        def emit(text: str) -> None:
            pre.append(text)
            tts_q.put(text)
            spoken["words_out"] += len(text.split())
            self.send({"type": "answer_delta", "text": text + " "})
        fixed = None
        if tool is not None:
            context, fixed = tool(emit)
            if fixed is not None:
                emit(fixed)
                meta["finish"] = "fixed"
        system = f"{a.system} {GREETING}" if greeting else a.system
        if context:
            system = f"{system} {context}"
        max_sentences = min(a.max_sentences or 2, 2) if greeting else a.max_sentences
        msgs = [{"role": "system", "content": system}] + self.history[-2 * a.history:] + \
            [{"role": "user", "content": query}]
        stream = (api.chat_stream(msgs, a.chat_model, a.max_tokens, 0.0 if context else a.temperature, meta,
                                  COPY_SAMPLING if context else None) if fixed is None else iter(()))
        try:
            for piece in stream:
                if first_ms is None:
                    first_ms = (time.monotonic() - t0) * 1000.0
                parts.append(piece)
                hurry = a.early_speech and speech_ahead(spoken) < AHEAD_S
                for chunk, whole in split.feed_chunks(piece, hurry):
                    if not max_sentences or n_sent < max_sentences:
                        said.append(chunk)
                        tts_q.put(chunk)
                        spoken["words_out"] += len(chunk.split())
                        n_sent += whole
                if max_sentences and n_sent >= max_sentences:
                    shown = piece[:len(piece) - min(len(split.buf), len(piece))]   # not the next sentence's start
                    if shown:
                        self.send({"type": "answer_delta", "text": shown})
                    meta["finish"] = "sentences"
                    break
                self.send({"type": "answer_delta", "text": piece})
        except ApiError as e:
            self.send({"type": "error", "message": str(e)})
            log(f"[{self.id}] chat failed: {e}")
        finally:
            if fixed is None:
                stream.close()                 # stopped early: the chat server stops generating
        rest = split.flush()
        if rest and (not n_sent or meta.get("finish") == "stop"):
            said.append(rest)                  # an answer cut at max_tokens keeps its whole sentences only
            tts_q.put(rest)
            spoken["words_out"] += len(rest.split())
        tts_q.put(None)
        gen_ms = (time.monotonic() - t0) * 1000.0
        worker.join()
        answer = " ".join(pre + said)
        log(f"[{self.id}] speech pieces: {' | '.join(pre + said)!r} ({spoken['wps'] or 0:.1f} words/s; gaps before "
            f"request {', '.join(f'{n}: {g * 1000:.0f} ms' for n, g in zip(spoken['gap_at'], spoken['gaps'])) or '-'})")
        if answer:
            self.history += [{"role": "user", "content": query}, {"role": "assistant", "content": answer}]
        self.greeted = greeting
        self._finish(query, answer, spoken, {"first_token_ms": round(first_ms or 0.0), "answer_ms": round(gen_ms),
                                             "finish": meta.get("finish")})

    def say(self, text: str, query: str = "", dialogue: Optional[bool] = None, greeting: bool = False) -> None:
        """A fixed reply, spoken (the dialogue mode's acknowledgements, the answer to
        the key phrase alone: ``greeting``, then the next segment is the request);
        `dialogue`: switch the mode (by voice) once the request is shown."""
        t0 = time.monotonic()
        self.send({"type": "query", "text": query})
        if dialogue is not None:
            self.set_dialogue(dialogue, "voice")
        self.set_state("thinking")
        self.playback_done.clear()
        self.send({"type": "answer_delta", "text": text})
        q: "queue.Queue[Optional[str]]" = queue.Queue()
        q.put(text)
        q.put(None)
        spoken = new_spoken()
        self._speak(q, spoken, t0)
        self.greeted = greeting
        self._finish(query, text, spoken, {"first_token_ms": 0, "answer_ms": 0, "finish": "fixed"})

    def _finish(self, query: str, answer: str, spoken: dict, info: dict) -> None:
        """The turn's end: answer_done, the speech's end, the client's playback, listening again."""
        audio_s = spoken["samples"] / spoken["rate"] if spoken["rate"] else 0.0
        info = dict(info, first_audio_ms=round(spoken["first_ms"] or 0.0), audio_s=round(audio_s, 2),
                    speech_requests=spoken["requests"], speech_gaps_ms=[round(g * 1000) for g in spoken["gaps"]],
                    dialogue=self.dialogue)
        self.send({"type": "answer_done", "text": answer, "info": info})
        log(f"[{self.id}] turn {self.turns}: {query!r} -> {answer!r} {info}")
        if spoken["rate"]:
            self.send({"type": "audio_end", "samples": spoken["samples"]})
            if not self.playback_done.wait(audio_s + self.args.playback_margin):
                log(f"[{self.id}] no playback_done from the client; listening again")
        if not self.closed.is_set():
            self.listen()

    def _speak(self, q: "queue.Queue[Optional[str]]", spoken: dict, t0: float) -> None:
        """Text to speech of the answer's sentences, in order; the sentences
        waiting when a request starts go in one request."""
        api, a = self.server.api, self.args
        done = False
        while not done:
            batch = [q.get()]
            if batch[0] is None:
                break
            while True:
                try:
                    s = q.get_nowait()
                except queue.Empty:
                    break
                if s is None:
                    done = True
                    break
                batch.append(s)
            text = " ".join(batch)
            spoken["requests"] += 1
            spoken["words_started"] += len(text.split())
            spoken["batch_left"] = len(text.split()) / (spoken["wps"] or WORDS_PER_S)   # its audio, until it arrives
            batch_audio = 0.0
            try:
                rate, pieces = api.speech_stream(text, a.tts_model)
                for pcm in pieces:
                    if self.closed.is_set():
                        return
                    now = time.monotonic()
                    if spoken["rate"] is None:
                        spoken["rate"], spoken["first_ms"] = rate, (now - t0) * 1000.0
                        spoken["playback_end"] = now
                        self.send({"type": "audio_start", "rate": rate})
                        self.set_state("speaking")
                    if now > spoken["playback_end"] + 0.05:          # the player ran dry: a gap
                        spoken["gaps"].append(now - spoken["playback_end"])
                        spoken["gap_at"].append(spoken["requests"])
                    spoken["playback_end"] = max(spoken["playback_end"], now) + len(pcm) / 2 / rate
                    spoken["batch_left"] = max(0.0, spoken["batch_left"] - len(pcm) / 2 / rate)
                    batch_audio += len(pcm) / 2 / rate
                    self.chan.send_audio(pcm)
                    spoken["samples"] += len(pcm) // 2
                spoken["words_done"] += len(text.split())
                spoken["audio_done"] += batch_audio
                if spoken["audio_done"] > 0:
                    spoken["wps"] = spoken["words_done"] / spoken["audio_done"]
                spoken["batch_left"] = 0.0
            except (ApiError, OSError) as e:
                spoken["batch_left"] = 0.0
                spoken["error"] = str(e)
                self.send({"type": "error", "message": f"text to speech: {e}"})
                log(f"[{self.id}] text to speech failed: {e}")


# ── the server ───────────────────────────────────────────────────────────────

class MirandaServer(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, args, api: ChatApi, weather: Optional[Weather] = None):
        super().__init__((args.host, args.port), ControlHandler)
        self.args, self.api = args, api
        self.weather = weather or Weather()
        self._home = None                       # --location, geocoded once
        self.sessions: Dict[int, Session] = {}
        self.lock = threading.Lock()
        self.udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        self.udp.bind((args.host, args.port))
        self.rtp_port = self.udp.getsockname()[1]
        self.udp_thread = threading.Thread(target=self._udp_loop, daemon=True)
        self.udp_thread.start()
        self.saved = itertools.count(1)
        if args.save_segments:
            os.makedirs(args.save_segments, exist_ok=True)

    def home(self):
        """Where "the weather" is: --location, else the board's position from its public IP."""
        if not self.args.location:
            return self.weather.here()
        if self._home is None:
            place = self.weather.geocode(self.args.location)
            if place is None:
                raise WeatherError(f"--location {self.args.location!r}: no such place")
            self._home = place
        return self._home

    def save_segment(self, sess: "Session", seg, text: str, wake: Optional[dict]) -> None:
        """--save-segments: the segment as WAV, a line in segments.jsonl."""
        d = self.args.save_segments
        if not d:
            return
        name = f"{time.strftime('%Y%m%d-%H%M%S')}-{sess.id}-{next(self.saved)}.wav"
        try:
            with open(os.path.join(d, name), "wb") as f:
                f.write(wav_bytes(seg.pcm))
            with self.lock, open(os.path.join(d, "segments.jsonl"), "a") as f:
                f.write(json.dumps({"file": name, "seconds": round(seg.seconds, 2), "state": sess.state,
                                    "text": text, "wake": wake}) + "\n")
        except OSError as e:
            log(f"[{sess.id}] --save-segments: {e}")

    def _udp_loop(self) -> None:
        while True:
            try:
                pkt, addr = self.udp.recvfrom(4096)
            except OSError:
                return
            r = mp.rtp_unpack(pkt)
            if r is None or r[0] != mp.PT_L16:
                continue
            _pt, marker, seq, _ts, ssrc, payload = r
            s = self.sessions.get(ssrc)
            if s is not None and addr[0] == s.peer_ip:
                s.on_rtp(seq, marker, payload)

    def server_close(self) -> None:
        super().server_close()
        self.udp.close()


class ControlHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        srv: MirandaServer = self.server
        chan = mp.Channel(self.request)
        peer = self.client_address[0]
        try:
            msg = chan.recv()
            if msg is None or msg[0] != "json" or msg[1].get("type") != "hello":
                return
            hello = msg[1]
            if hello.get("codec", mp.CODEC) != mp.CODEC:
                chan.send({"type": "error", "message": f"codec {hello.get('codec')} (expected {mp.CODEC})"})
                return
            with srv.lock:
                if len(srv.sessions) >= srv.args.max_sessions:
                    chan.send({"type": "error", "message": "too many sessions"})
                    return
                s = Session(srv, chan, peer, hello)
                if s.ssrc in srv.sessions:
                    chan.send({"type": "error", "message": "SSRC in use"})
                    return
                srv.sessions[s.ssrc] = s
            a = srv.args
            chan.send({"type": "welcome", "version": mp.VERSION, "session": s.id, "rtp_port": srv.rtp_port,
                       "wake_phrase": a.wake, "models": {"chat": a.chat_model, "asr": a.asr_model,
                                                         "tts": a.tts_model},
                       "dialogue": "auto" if s.dialogue_auto else "off", "dialogue_timeout": s.dialogue_timeout})
            log(f"[{s.id}] session from {peer} (ssrc {s.ssrc:08x})")
            worker = threading.Thread(target=s.run, daemon=True)
            worker.start()
            while not s.closed.is_set():
                msg = chan.recv()
                if msg is None:
                    break
                kind, obj = msg
                if kind == "json" and obj.get("type") == "playback_done":
                    s.playback_done.set()
                elif kind == "json" and obj.get("type") == "dialogue":
                    s.set_dialogue(bool(obj.get("on")), "client")
                elif kind == "json" and obj.get("type") == "bye":
                    break
            s.close()
            worker.join(timeout=5)
            log(f"[{s.id}] session ended ({s.turns} turns; rtp since the last turn: {s.rtp.packets} packets, "
                f"{s.rtp.lost} lost, {s.rtp.late} late; {s.dropped} dropped while not listening)")
        except (OSError, ValueError) as e:
            log(f"control connection from {peer}: {e}")
        finally:
            with srv.lock:
                for k, v in list(srv.sessions.items()):
                    if v.chan is chan:
                        del srv.sessions[k]
            chan.close()


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=mp.PORT, help="TCP control and UDP RTP port")
    ap.add_argument("--chat-url", default="http://127.0.0.1:8000/v1", help="the KV260 chat server's API")
    ap.add_argument("--api-key", default=os.environ.get("KV260_CHAT_API_KEY"))
    ap.add_argument("--api-key-file", default=None)
    ap.add_argument("--chat-model", default="smollm2-360m-instruct",
                    help="the chat model: SmolLM2-360M stays loaded beside Moonshine and Piper with cma=1400M "
                         "(smollm2-135m-instruct: about 2.5x faster, weaker answers)")
    ap.add_argument("--asr-model", default="whisper-1", help="speech to text (the chat server's alias)")
    ap.add_argument("--tts-model", default="tts-1", help="text to speech (the chat server's alias)")
    ap.add_argument("--wake", default="hello miranda", help="the key phrase")
    ap.add_argument("--wake-margin", type=float, default=WAKE_MARGIN,
                    help="the phrase check's margin (nats, <= 0): the decoder fed the key phrase keeps it at "
                         "the start of the transcript when it is at most this much less likely than the "
                         "decoder's own words")
    ap.add_argument("--no-weather", dest="weather", action="store_false",
                    help="no weather tool (weather.py: the board's public IP -> its position -> Open-Meteo)")
    ap.add_argument("--no-early-speech", dest="early_speech", action="store_false",
                    help="speak an answer by whole sentences only (by default, when less than 2 s of speech is "
                         "ahead, the next clause goes to speech as soon as it is written)")
    ap.add_argument("--no-announce", dest="announce", action="store_false",
                    help="no spoken announcement of a weather lookup (by default the chat model says it is "
                         "checking, a fixed sentence when it guesses the weather instead)")
    ap.add_argument("--location", default=None, metavar="PLACE",
                    help="the weather tool's place when the request names none (default: where the board's "
                         "public IP is, as ip-api.com / ipwho.is say)")
    ap.add_argument("--greeting", choices=("fixed", "chat", "off"), default="fixed",
                    help="the answer to the key phrase alone, before Miranda waits for the request: fixed "
                         "(the default: \"Hi! How can I help you?\" and others in turn), chat (the chat model "
                         "greets back — SmolLM2-135M does it badly) or off (none: she waits in silence)")
    ap.add_argument("--no-wake-check", dest="wake_check", action="store_false",
                    help="transcribe without the phrase check (the key phrase only from the transcript)")
    ap.add_argument("--save-segments", default=None, metavar="DIR",
                    help="keep every segment heard as DIR/<time>-<session>-<n>.wav, with its state, "
                         "transcript and phrase check in DIR/segments.jsonl (to tune --wake-margin; "
                         "deploy.py --fetch-segments downloads them)")
    ap.add_argument("--system", default=SYSTEM, help="the chat model's system message")
    ap.add_argument("--max-tokens", type=int, default=100)
    ap.add_argument("--max-sentences", type=int, default=3,
                    help="stop the answer after N sentences (0: at --max-tokens only)")
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--history", type=int, default=4, help="earlier turns the chat model sees")
    ap.add_argument("--awake-timeout", type=float, default=8.0,
                    help="seconds of audio to wait for the request after the key phrase alone")
    ap.add_argument("--dialogue-timeout", type=float, default=15.0,
                    help="the dialogue mode ends after this many seconds without speech (a client may ask for "
                         "another value in its hello)")
    ap.add_argument("--vad-threshold", type=float, default=0.5)
    ap.add_argument("--vad-hangover-ms", type=int, default=700, help="silence that ends a speech segment")
    ap.add_argument("--playback-margin", type=float, default=20.0,
                    help="seconds beyond the speech's length to wait for the client's playback_done")
    ap.add_argument("--max-sessions", type=int, default=4)
    args = ap.parse_args(argv)
    if args.api_key_file:
        with open(args.api_key_file) as f:
            args.api_key = f.read().strip()
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    srv = MirandaServer(args, ChatApi(args.chat_url, args.api_key))
    log(f"miranda listening on {args.host}:{args.port} (TCP control, UDP RTP); chat server {args.chat_url}; "
        f"models {args.asr_model} / {args.chat_model} / {args.tts_model}; key phrase {args.wake!r} "
        + (f"(checked against the audio, margin {args.wake_margin:+g})" if args.wake_check else "(text only)")
        + (f"; segments saved in {args.save_segments}" if args.save_segments else ""))
    try:
        srv.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
