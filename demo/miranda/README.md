# Miranda — a voice assistant on the KV260

Say "Hello Miranda" and a request; Miranda answers aloud.  The console client
runs on your laptop and streams the microphone to the board; everything else
— voice activity detection, speech recognition, the answer and its speech —
runs on the KV260: the VAD on its CPU, the three models on the FPGA.

```
$ python3 demo/miranda/miranda_client.py --server 192.168.100.8
Miranda · 192.168.100.8:8770 · say “Hello Miranda” and your request · Ctrl-C quits
(warming up the models on the board…)
(listening)
  · Good morning, everyone. this is just a conversation.
You: Haloma Randa, what is the capital of France?
Miranda: The capital of France is Paris. It's a beautiful city with many famous landmarks and historical sites.
  [first word 0.1 s · answer 1.1 s · speech 5.6 s from 1.3 s]
(listening)
You: Hello, Miranda.
Miranda: Hi! How can I help you?
(listening for your request)
You: What is the capital of Italy?
Miranda: The capital of Italy is Rome.
  [first word 0.1 s · answer 0.5 s · speech 1.7 s from 0.7 s]
(listening)
```

(Recordings spoken by Piper on the board, streamed with `--input`: in the
first the speech recognizer heard "Hello Miranda" as "Haloma Randa", and
Miranda still woke.  Lines with `·` were heard without the key phrase.
"Hello Miranda" alone gets a short answer, and the next sentence is the
request.)

## The dialogue mode

To talk without saying "Hello Miranda" before every phrase, switch to the
dialogue mode.  It ends by itself after 15 s without speech
(`--dialogue-timeout`).

```
You: Hello, Miranda. let's talk.
(dialogue mode on: talk without “Hello Miranda”; it ends after 15 s of silence or when you say goodbye)
Miranda: Sure, let's talk. I'm listening.
(your turn — dialogue mode)
You: What is the capital of Italy?
Miranda: The capital of Italy is Rome.
(your turn — dialogue mode)
You: And what about Spain?
Miranda: Spain is also known as Castile and León.
(your turn — dialogue mode)
You: Good-bye
(dialogue mode off: you said so — say “Hello Miranda” again)
Miranda: Goodbye!
(listening)
  · This should not be answered.
```

| | On | Off |
|---|---|---|
| **By voice** | "Hello Miranda, let's talk" (or "let's chat", "start a dialogue", "conversation mode", "can we have a conversation") | "Goodbye", "Bye bye", "That's all, thank you", "See you later", "Good night", "Stop the dialogue" |
| **From the client** | `d` + Enter in the console; with `--dialogue`, every "Hello Miranda" opens a dialogue | `d` + Enter |
| **By itself** | — | after `--dialogue-timeout` seconds without speech (the server's default 15 s; the client may ask for another value) |

- **Requests.**  In the dialogue mode every segment the VAD finds is a
  request.  A "Hello Miranda" in front of it is dropped, and the key phrase
  alone restarts the silence timer.
- **Silence.**  It is measured on the audio's own timeline from the end of
  Miranda's speech, so her own speaking time does not count.
- **Context.**  The chat model sees the earlier turns (`--history`, 4), so
  follow-ups such as "And what about Spain?" work.

## How it works

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 300}}}%%
flowchart LR
    client["<b>miranda_client.py</b> · laptop<br/>microphone (pw-record ...) · player (pw-play ...)<br/>console"]
    subgraph board["KV260"]
        server["<b>miranda_server.py</b> (CPU)<br/>RTP in · VAD (vad.py) · key phrase (wake.py)<br/>sentences · turns"]
        chat["<b>kv260_chat_server.py</b><br/>/v1/audio/transcriptions · /v1/chat/completions<br/>/v1/audio/speech · one request on the FPGA at a time<br/>(the answer lends it to speech between tokens)"]
        fpga["<b>FPGA</b><br/>Moonshine base · SmolLM2-360M · Piper"]
    end
    client -- "RTP over UDP: the microphone,<br/>L16 16 kHz, 20 ms packets" --> server
    client <-- "TCP: events, the answer's text,<br/>Miranda's speech (PCM)" --> server
    server -- "HTTP on the loopback" --> chat
    chat --> fpga
```

A session goes through four states:

| State | What happens | The microphone |
|---|---|---|
| `listening` | The VAD cuts the stream into speech segments.  Each segment is transcribed (Moonshine).  A transcript holding the key phrase wakes Miranda, and the text after the phrase is the request. | streamed |
| `awake` | The key phrase came alone ("Hello Miranda." then a pause).  Miranda answers it ("Hi! How can I help you?", see below), and the next segment is the request, or after `--awake-timeout` (8 s of audio) she listens again.  The key phrase said again restarts the wait. | streamed |
| `dialogue` | Every segment is a request; no key phrase needed (above).  After the dialogue timeout without speech she goes back to `listening`. | streamed |
| `thinking` | The request and the earlier turns go to the chat model (SmolLM2-360M), and the answer streams to the console as text. | muted |
| `speaking` | Finished sentences go to text to speech (Piper) and the speech streams to the client, which plays it.  The client reports when playback is over, and Miranda then listens again. | muted until the player has finished, plus 0.25 s |

- **Voice activity detection** (`vad.py`, numpy on the A53).  It is the
  likelihood-ratio detector of Sohn, Kim and Sung (1999), on 20 ms frames:
  - per frequency bin of the speech band (150 Hz–4 kHz), the log likelihood
    ratio of speech + noise against noise, with the decision-directed a
    priori SNR;
  - the noise spectrum is learnt from the first frames that are not
    digital silence (a recorder's leading zeros), then tracked in the frames
    judged silent;
  - when the last 1.5 s were stationary (the frames' power within 10 dB:
    a noise floor stays within 5 dB, speech moves by more than 12 dB), the
    noise is raised to their minimum (minimum statistics), so a louder room
    is not taken for speech for long;
  - it keeps the noise between turns, so speech in the first frames after
    the microphone resumes is heard;
  - frames below −62 dBFS are silence.

  A segment opens after 3 speech frames (it starts 0.3 s earlier) and
  closes after 0.7 s of silence (`--vad-hangover-ms`) or 24 s.  It costs
  2.5 % of one A53 core per stream.  On LibriSpeech utterances separated by
  pauses, with pink noise from none up to −35 dBFS, it found every
  utterance and made no segment from noise.
- **The key phrase, checked against the audio.**  Moonshine takes no
  prompt, but the library lets the caller choose each token the decoder is
  fed.  So every segment is transcribed with the
  chat server's phrase check (`kv260_wake_phrase`,
  [Speech to text](../chat/doc/SPEECH_TO_TEXT.md#the-phrase-check)):
  - the decoder is fed "Hello Miranda" as its first tokens;
  - the margin sums, over the phrase's tokens the decoder would not have
    written itself, how much less likely each is than its own choice (in
    nats; 0: it writes the phrase itself);
  - when the margin is at least `--wake-margin` (−10), the transcript is
    decoded on from the phrase, so it starts with "Hello, Miranda";
    otherwise it is the plain transcript.

  Awake or in a dialogue, a transcript that starts with the phrase is the
  phrase said again, not the request.

  "Let's talk" is checked the same way.  A real speaker's "Hello Miranda,
  let's talk" came out as "It is stock", "a little stoke" and "red stalk",
  and each was answered as a request.  So a short request (at most three
  words, not a command) is transcribed once more with "let's talk" checked
  against the audio, after the key phrase when there was one.  If it holds,
  the dialogue mode starts.  Piper saying "red stalk" or "it is stock"
  scores −13.6 / −14.6 there and stays a request.

  On the board a clear "Hello Miranda" scores 0, and ordinary speech scores
  below −15 (196 LibriSpeech utterances), and greetings to someone else
  below −10.  A speaker whose "Hello Miranda" Moonshine writes as "Hello,
  Meganda", "Homey granda" or "How old me, Randa?" scored −7.67, 0, −5.75,
  −2.68 and −2.15: five of five kept.  "Hello" with a name that rhymes ("Hello Veranda", "Hello
  Amanda") scores −3 to −11 and may wake her.  The check adds 3 ms to a
  segment on average.
- **The key phrase in the transcript** (`wake.py`).  The recognizer writes
  "Hello Miranda" in many ways, so a transcript holds the phrase in any of
  three cases:
  - **by words:** each word of the phrase is matched by letter similarity
    (≥ 0.8), and a recognizer word may also be two words run together;
  - **by sound:** one to four words carry the phrase's consonants in order.
    Vowels, h, w and y are dropped, and voiced / voiceless pairs are folded:
    "Haloma Randa" and "Hello Miranda" are both l-m-r-n-t, while "Hello
    Rhonda" (l-r-n-t) is not the phrase;
  - **by name:** the greeting is what the recognizer mangles most ("How
    old, Miranda", "Oh, Mjolanda", "How woe, meogandha"), so a word shaped
    like "Miranda" after at most two short words at the start is enough:
    similar letters, the same first and last consonant, at most one of its
    consonants missing ("Muganda").  "Hello Amanda" wakes her too; "Hello
    Rhonda", "Veranda" or "Rwanda" do not.

  The phrase can be changed with `--wake`.
- **The key phrase alone.**  "Hello Miranda" with nothing after it gets a
  short spoken answer — "Hi! How can I help you?", "Yes? What can I do for
  you?" and two more in turn — and the next segment is the request, without
  the key phrase.  `--greeting chat` lets the chat model answer instead (it
  is told to greet back and ask how it can help; SmolLM2-135M ignores that
  and rambles, so it is not the default), `--greeting off` waits in
  silence.
- **The weather.**  "What's the weather like?", "Will it rain tomorrow?",
  "Do I need an umbrella on Tuesday?", "What's the weather in Paris?" run
  the weather tool (`weather.py`, no API keys):
  - **Where:** the board's public IP (api.ipify.org), then its position
    (ip-api.com, else ipwho.is; an IP's city is approximate — for the board
    in Batumi ip-api.com said Batumi, ipwho.is Tbilisi), or the place the
    question names (Open-Meteo's geocoding), or `--location PLACE`.
  - **The forecast:** Open-Meteo, now or a day of the next week; cached
    for ten minutes, the position for an hour; 0.6 s when not cached.
  - **Which questions:** rules decide (`WEATHER_INTENT` in
    `miranda_server.py`), not the model.  Asked "is this about the
    weather?", SmolLM2-360M said yes to "Write a song about snow" as surely
    as to "Will it rain tomorrow?", and SmolLM2-135M said no to both (the
    chat server's `logprobs` measured it).  The rules take the place
    ("in Paris") and the day ("tomorrow", "on Friday", "this weekend") from
    the question; jokes, poems and songs about the weather stay with the
    chat model.
  - **The announcement:** as chat services do before a tool call, Miranda
    first says she is looking it up — "I'm currently checking the current
    weather in Paris." — while the lookup runs.  The chat model writes it,
    cut at its first sentence (1.0–1.6 s); it is spoken at once (the chat
    server lends the FPGA to the speech while the answer is written), so
    the speech starts 1.5–2.2 s after a weather question instead of 4–6 s.
    SmolLM2-360M goes on guessing the weather after that sentence ("It's a
    beautiful day with clear skies", "around 28 degrees"), so a sentence with
    a number or a weather word is replaced by "Let me check the weather in
    Paris."  `--no-announce` leaves it out.
  - **The answer:**
    - *Rain, umbrella and snow questions* get an answer composed from the
      numbers: "Probably not, rain is unlikely tomorrow in Tbilisi,
      Georgia: an 18 percent chance of rain, overcast, 15 to 26 degrees
      Celsius."  The model said "yes, it will likely rain" to these
      whatever the data.
    - *Other weather questions* go to the chat model with the forecast in
      its system message.  It answers greedily without the repetition and
      DRY penalties; with the chat server's defaults it turned "13 degrees
      (feels like 10)" into "10 degrees Celsius (34°F)".
  - **Options:** `--no-weather` turns the tool off, `--location PLACE`
    fixes its place.
- **The answer.**
  - The chat model gets the request and the last 4 turns, with a system
    message asking for one to three short spoken sentences.
  - The answer stops after 3 sentences (`--max-sentences`); the chat server
    stops generating when the stream closes.  An answer cut at
    `--max-tokens` (100) is spoken up to its last whole sentence.
  - SmolLM2-360M is the default: its answers are coherent and factual, at
    7.6 tokens/s — speech starts 2.4–3.5 s after the request and three
    sentences take about 10 s.  SmolLM2-135M (`deploy.py --chat-model
    smollm2-135m-instruct`) is about 2.5 times faster (0.5–1.9 s and 3–4 s)
    and wanders.  SmolLM2-360M needs the board at `cma=1400M`
    ([the repo README](../../README.md#5-prepare-the-board-and-load-the-bitstream)):
    there it stays loaded with Moonshine, Piper and SmolLM2-135M, while at
    1000M two of three turns failed (its 795 MB pool did not fit beside
    Moonshine and Piper).
- **The FPGA.**  Every model runs through the chat server's API on the
  board's loopback.  The chat server owns the FPGA, runs one request at a
  time and keeps the models loaded.
- **Speaking while the answer is written.**  The answer goes to text to
  speech in pieces, paced by the speech still ahead of the player (the
  audio sent and not played, plus the text not yet spoken, at the turn's
  measured rate, about 3 words/s):
  - while less than 2 s is ahead, the next clause goes out as soon as it is
    written — at a comma after 3 words, or before a joining word (and, but,
    which, who ...) after 5;
  - otherwise whole sentences, which sound better; a sentence longer than
    12 / 16 words is still cut at a clause (Piper encodes a piece before it
    speaks: 0.9 s to the first audio for a 70-token sentence, 0.4 s for a
    short one);
  - never at a comma between capitalised words ("Tbilisi, Georgia": a piece
    once ended there and the next came 2.2 s later), nor after Mr. / Dr. /
    an initial / an abbreviation before a lower-case word.

  The log shows each turn's pieces and its gaps (silences the player had,
  `speech_gaps_ms`).  2026-10-18, a weather question and four long answers,
  twice each with SmolLM2-360M: the long answers had no gap; the weather
  answer one of 0.42–0.44 s after the announcement (it had 2.1–2.2 s inside
  "Tbilisi, Georgia" with fixed word counts).  The speech starts 1.2–1.9 s
  after the request.  `--no-early-speech`: whole sentences only.
  - NLTK's Punkt splitter was tried for the sentence ends: on the 144
    answers in Miranda's log it splits exactly as `Sentences` does, it cuts
    "e.g. | the", and it would add NLTK and its data to the board.
- **Not hearing herself.**  The client streams the microphone only while
  the server listens.  It stops before Miranda thinks and stays silent until
  her speech has finished playing.  On the board, 0 packets arrived while
  the server was not listening (its log counts them).

## Install and run

Miranda needs the chat server on the board (`demo/chat`, [Deploying](../chat/doc/DEPLOY.md))
with a speech-to-text, a chat and a text-to-speech model: `moonshine`,
`smollm2` and `piper` in `server.backends` of `demo/chat/chat_config.json`.

```bash
PY=inference-scheduler/.venv/bin/python
$PY demo/miranda/deploy.py                       # checks the chat server, uploads, starts kv260-miranda
python3 demo/miranda/miranda_client.py --server 192.168.100.8
$PY demo/miranda/deploy.py --status              # | --stop
$PY demo/miranda/deploy.py --fetch-segments      # the recorded segments (-- --save-segments) -> demo/miranda/build/segments
```

- **`deploy.py`** uploads the server to `<chat dir>/miranda` and starts it as
  the transient systemd unit `kv260-miranda` (`journalctl -u kv260-miranda`)
  on port 8770: TCP for control, UDP for RTP.  It reads the board, ssh and
  the chat server's port and API key from `demo/chat/chat_config.json`.
  Arguments after `--` go to `miranda_server.py` (`--wake`,
  `--wake-margin`, `--no-wake-check`, `--save-segments DIR`, `--greeting`,
  `--max-sentences`, `--awake-timeout`, `--vad-*`, `--system` ...).
- **Tuning the key phrase for your voice.**  The server's log shows each
  segment's margin (`journalctl -u kv260-miranda`):

  ```
  [1] heard (1.5 s, asr 227 ms, phrase -11.61 no): 'It is the weather today.'
  [1] heard (3.1 s, asr 430 ms, phrase -2.88 kept) WAKE: 'Hello, Miranda. What is the capital of France?'
  ```

  (Piper's "It is the weather today." and "Haloma Randa, what is the
  capital of France?" in −50 dBFS noise; a margin that stopped below the
  limit is where the sum stopped, the whole one is lower.)  `-- --save-segments DIR` also keeps every
  segment as a WAV file on the board, with its state, transcript and margin
  in `DIR/segments.jsonl`; a later `deploy.py` without it stops the
  recording (`deploy.py --status` says whether the server records).  Say
  "Hello Miranda" a number of times and talk as usual; a `--wake-margin`
  between the margins of your key phrases and those of your other speech
  separates them.  `deploy.py --fetch-segments [DIR]` downloads the
  recordings (from the running server's `--save-segments` directory, else
  `<chat dir>/miranda/segments`) into `DIR`, by default
  `demo/miranda/build/segments`, and fetches only new or grown files when
  run again:

  ```bash
  $PY demo/miranda/deploy.py -- --save-segments /root/kv260_chat/miranda/segments
  $PY demo/miranda/deploy.py --fetch-segments
  ```
- **The client** needs Python 3 and nothing else.
  - **Microphone:** the first recorder found (`pw-record`, `parecord`,
    `arecord`, ffmpeg from PulseAudio), or `--mic "CMD"`.
  - **Speech:** the first player found (`pw-play`, `paplay`, `aplay`,
    `ffplay`, sox's `play`), or `--player "CMD"` (`{rate}` is replaced by
    the sample rate).
  - **Without a microphone:** `--input FILE.wav` streams a recording
    instead.  `--save FILE.wav` keeps Miranda's speech, `--player none`
    skips playing, and `--turns N` exits after N answers.
  - **Dialogue mode:** `--dialogue` opens it with every "Hello Miranda",
    `--dialogue-timeout S` sets its silence limit, and `d` + Enter toggles
    it.
- **The server's log** has a line per segment heard, with its length, the
  recognition time and the phrase check's margin (`phrase -3.91 kept`), and
  one per turn: the request, the answer and the timings.

## Protocol

[`miranda_proto.py`](miranda_proto.py) describes it in full.
- **Microphone:** RTP (RFC 3550) over UDP to port 8770.  Payload type 96 =
  L16/16000/1 (big-endian 16-bit, mono, 16 kHz), 20 ms per packet, the
  client's SSRC announced in its hello.  The marker bit opens a talkspurt:
  the first packet, and the first after the microphone was muted.  The
  server puts packets back in order, fills a small loss with silence and
  drops late packets.
- **Control:** one TCP connection to port 8770, carrying frames of kind (1
  byte), length (4 bytes) and payload: `J` for a JSON message, `A` for
  speech PCM.
  - The client sends `hello` (with `dialogue`: `auto` / `off` and
    `dialogue_timeout`), `dialogue` (`on`: switch the mode), `playback_done`
    and `bye`.
  - The server sends `welcome`, `state`, `heard`, `query`, `answer_delta`,
    `answer_done`, `audio_start` / `A` frames / `audio_end`, `dialogue`
    (`on`, `reason`: voice / client / auto / silence, `timeout`), and
    `error`.

## Measured

2026-10-17, the board's chat server with Moonshine base, the chat model and
Piper resident (the session above, SmolLM2-135M — then the default):
- **A segment's transcription:** 200–420 ms for 1.3–3.2 s of speech.
- **The answer:** first word after 0.1–0.2 s; with SmolLM2-135M 1.1 s for
  two sentences, 4.5 s for three long ones; with SmolLM2-360M 9.7–10.6 s
  for three (7.6 tokens/s).
- **Speech:** Piper makes it faster than it plays.  It starts 0.5–1.9 s
  after the request, while the answer is still being written.  Measured on
  three requests with three-sentence answers (a second run: 1.4 / 1.1 /
  1.9 s):

  | | speech starts | answer written |
  |---|---|---|
  | the FPGA lent between tokens | 1.5 / 0.5 / 1.2 s | 4.4 / 3.1 / 4.3 s |
  | the answer first (`--no-interleave`, before 2026-10-17) | 2.7 / 3.8 / 3.7 s | 2.3 / 3.5 / 3.3 s |

  The answer takes longer now by the speech's FPGA time.
- **Warm-up per session:** 0.4 s with the models loaded.  It takes longer
  when the chat server must load them first: Moonshine 12.7 s and
  SmolLM2-135M 22 s in the chat server's log.
- **The VAD on the board:** 2.5 % of one core per stream.
- **The phrase check** (`doc/plans/ASR_PLAN.md` §8): Piper's "Haloma Randa,
  what is the capital of France?" is transcribed "Helumor Renda, …" without
  it and "Hello, Miranda. What is …" with it (margin −2.37); +2.7 ms per
  segment on average.

## Limits

- **Half duplex:** no barge-in; while Miranda speaks she does not listen.
- **In the dialogue mode everything heard is a request**, including speech
  meant for someone else.  The silence limit and "goodbye" end it.
- **Answer quality** is SmolLM2-360M's.
- **The key phrase is checked at the start of a segment only.**  Said
  later in a sentence ("So, hello Miranda!"), only the transcript can hold
  it.  The phrase check also needs the segment to fit one encoder pass
  (24.6 s; the VAD cuts at 24 s).
- **The weather questions are the ones the rules know**: they were checked
  on 33 weather questions and 27 others about rain, snow, heat or cold
  that are not about the weather (all right), but other phrasings miss —
  the chat model then answers without data.
- **English only.**
- **Every word said near the microphone is transcribed** (on the board) to
  look for the key phrase.

## Tests

```bash
cd demo/miranda/tests && ../../../inference-scheduler/.venv/bin/python -m unittest -v   # 31 tests, ~80 s
```

They cover:
- RTP and its reordering, the control frames;
- the VAD on synthetic speech and noise: a recorder's leading zeros, a
  noise floor that rises, speech right after the microphone resumes;
- the key phrase in its spellings and near misses, among them the
  recognizer's renderings of a real speaker's "Hello Miranda";
- the phrase check: which segments are checked, the margin against
  `--wake-margin`, `--no-wake-check`, `--save-segments`;
- the sentences;
- whole sessions: the client streaming a recording to the server, which
  talks to the chat server running fake backends (a scripted recognizer, the
  echo chat model, a fake voice).  They check the request with the key
  phrase, the key phrase alone then the request, the awake timeout, the
  three-sentence cap, the history, and that no audio reached the server
  while it was not listening.
- the answer to the key phrase alone (`--greeting` fixed, chat, off);
- "let's talk" checked against the audio of a short request;
- the weather tool over canned service answers (no network): the place from
  the IP and its fallback, the caches, the reports and the composed answers,
  the questions the rules take and leave, and a voice session with weather
  questions;
- the dialogue mode: on and off by voice, `--dialogue` ending after
  silence, and the client's switch over the control protocol.
