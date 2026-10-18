# Copyright 2026 Ivan Gubochkin
# SPDX-License-Identifier: Apache-2.0

"""vad.py — voice activity detection on the board's CPU (numpy) for Miranda.

Vad: the statistical likelihood-ratio detector of Sohn, Kim and Sung (IEEE
Signal Processing Letters 6(1), 1999) on 20 ms frames of 16 kHz audio: per
frequency bin of the speech band (150 Hz - 4 kHz) the log likelihood ratio of
"speech + noise" against "noise" under Gaussian models, with the a priori
SNR from the decision-directed estimate; the frame's score is their mean.
The noise spectrum starts from the first frames that are not digital silence
(a recorder's leading zeros teach it nothing) and is tracked in the frames
judged silent.  When the last 1.5 s were stationary — the frames' power in the
band within `stat_range` (10 dB) of each other: a noise floor stays within
5 dB, speech moves by more than 12 dB with its syllables — it is raised to the
minimum of their smoothed spectra times a bias (minimum statistics, R. Martin,
IEEE Trans. Speech and Audio Processing 9(5), 2001): a noise floor that rose,
or an estimate that started too low, is corrected within 1.5 s instead of being
taken for speech.  Frames below the absolute level are silence whatever the
ratio.

Segmenter: frames -> speech segments.  A segment opens after `onset`
speech frames within the last onset + 1 (it starts `preroll` earlier, so the
first syllable is kept), and closes after `hangover` silent frames (`tail`
of them kept) or at `max_s` seconds; segments with fewer than `min_speech`
speech frames are dropped.  Times are in the audio's own timeline (samples),
not the wall clock.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

RATE = 16000
FRAME = 320                                  # 20 ms
NFFT = 512


class Vad:
    def __init__(self, rate: int = RATE, frame: int = FRAME, threshold: float = 0.5,
                 band=(150.0, 4000.0), init_frames: int = 10, dd_alpha: float = 0.98,
                 noise_beta: float = 0.92, noise_creep: float = 0.995, floor_dbfs: float = -62.0,
                 min_window: int = 75, min_bias: float = 1.5, min_smooth: float = 0.8,
                 stat_range: float = 10.0, dead_rms: float = 2.0):
        self.rate, self.frame, self.threshold = rate, frame, threshold
        self.win = np.hanning(frame).astype(np.float64)
        f = np.fft.rfftfreq(NFFT, 1.0 / rate)
        self.bins = np.nonzero((f >= band[0]) & (f <= band[1]))[0]
        self.init_frames, self.dd_alpha = init_frames, dd_alpha
        self.noise_beta, self.noise_creep = noise_beta, noise_creep
        self.min_window, self.min_bias, self.min_smooth = min_window, min_bias, min_smooth
        self.stat_range = stat_range
        self.dead_rms = dead_rms                  # digital silence: a recorder's leading zeros
        # the absolute gate: a frame's RMS below floor_dbfs (of int16 full scale) is silence
        self.floor_rms = 32768.0 * 10 ** (floor_dbfs / 20.0)
        # the noise spectrum's minimum: a sine at the gate level spread over the band
        self.min_noise = (self.floor_rms * frame / 4.0) ** 2 / len(self.bins) * 1e-2
        self.reset()

    def reset(self, keep_noise: bool = False) -> None:
        """Forget the audio; with keep_noise the noise estimate stays (the same
        room after a pause in the stream)."""
        if keep_noise and self.noise is not None:
            self.prev_a2 = np.zeros_like(self.noise)
            return
        self.n = 0
        self.noise: Optional[np.ndarray] = None
        self._acc = None
        self.prev_a2 = None
        self._smooth = None                     # minimum statistics: the smoothed power ...
        self._ring = np.full((self.min_window, len(self.bins)), np.inf)   # ... of the last frames
        self._tot = np.zeros(self.min_window)   # ... and each frame's power in the band
        self._k = 0

    def score(self, x) -> float:
        """The frame's mean log likelihood ratio (0 while the noise is learnt;
        -inf below the absolute gate)."""
        x = np.asarray(x, np.float64)
        rms = float(np.sqrt(np.mean(x * x))) if len(x) else 0.0
        p = np.abs(np.fft.rfft(x * self.win[:len(x)], NFFT))[self.bins] ** 2
        if rms < self.dead_rms and self.n < self.init_frames:
            return float("-inf")                               # digital silence: nothing to learn
        self.n += 1
        self._smooth = p if self._smooth is None else self.min_smooth * self._smooth + (1 - self.min_smooth) * p
        self._ring[self._k % self.min_window] = self._smooth
        self._tot[self._k % self.min_window] = p.sum()
        self._k += 1
        if self.n <= self.init_frames:                         # the first frames: the noise
            self._acc = p if self._acc is None else self._acc + p
            self.noise = np.maximum(self._acc / self.n, self.min_noise)
            self.prev_a2 = np.zeros_like(p)
            return 0.0
        if self._k >= self.min_window and self._tot.max() <= self.stat_range * self._tot.min():
            self.noise = np.maximum(self.noise, self.min_bias * self._ring.min(axis=0))   # stationary
        noise = self.noise
        gamma = np.minimum(p / noise, 1e4)
        xi = self.dd_alpha * self.prev_a2 / noise + (1.0 - self.dd_alpha) * np.maximum(gamma - 1.0, 0.0)
        xi = np.maximum(xi, 10 ** (-25.0 / 10.0))
        llr = float(np.mean(gamma * xi / (1.0 + xi) - np.log1p(xi)))
        self.prev_a2 = (xi / (1.0 + xi)) ** 2 * p
        if rms < self.floor_rms:
            llr = float("-inf")
        if llr < self.threshold:                               # silence: learn the noise
            self.noise = np.maximum(self.noise_beta * noise + (1.0 - self.noise_beta) * p, self.min_noise)
        else:                                                  # speech: creep towards it, slowly
            self.noise = np.maximum(self.noise_creep * noise + (1.0 - self.noise_creep) * np.minimum(p, 4 * noise),
                                    self.min_noise)
        return llr

    def is_speech(self, x) -> bool:
        return self.score(x) >= self.threshold


@dataclass
class Segment:
    start: int                                 # sample index in the stream
    pcm: np.ndarray                            # int16
    speech_frames: int
    reason: str = "end"                        # end | max

    @property
    def seconds(self) -> float:
        return len(self.pcm) / RATE


@dataclass
class Segmenter:
    vad: Vad = field(default_factory=Vad)
    onset: int = 3                             # speech frames to open (of the last onset + 1)
    hangover: int = 35                         # silent frames to close (0.7 s)
    tail: int = 10                             # silent frames kept at the end (0.2 s)
    preroll: int = 15                          # frames kept before the onset (0.3 s)
    min_speech: int = 12                       # speech frames a segment needs (0.24 s)
    max_s: float = 24.0

    def __post_init__(self):
        self.reset()

    def reset(self, keep_noise: bool = False) -> None:
        """Drop the open segment; with keep_noise the VAD keeps its noise estimate."""
        self.vad.reset(keep_noise)
        self.pos = 0                           # samples consumed
        self.ring: deque = deque(maxlen=self.preroll + self.onset + 1)   # (frame, speech) before a segment
        self.cur: Optional[List[np.ndarray]] = None
        self.cur_start = 0
        self.cur_speech = 0
        self.silent = 0
        self._buf = np.zeros(0, np.int16)

    @property
    def active(self) -> bool:
        return self.cur is not None

    def feed(self, pcm) -> List[Segment]:
        """int16 samples in (any count); the segments they complete out."""
        x = np.concatenate([self._buf, np.asarray(pcm, np.int16)])
        n = len(x) // FRAME * FRAME
        self._buf = x[n:]
        out = []
        for i in range(0, n, FRAME):
            seg = self._frame(x[i:i + FRAME])
            if seg is not None:
                out.append(seg)
        return out

    def _frame(self, fr) -> Optional[Segment]:
        sp = self.vad.is_speech(fr)
        start = self.pos
        self.pos += FRAME
        if self.cur is None:
            self.ring.append((fr, sp, start))
            recent = list(self.ring)[-(self.onset + 1):]
            if sp and sum(s for _, s, _ in recent) >= self.onset:
                frames = list(self.ring)
                self.cur = [f for f, _, _ in frames]
                self.cur_start = frames[0][2]
                self.cur_speech = sum(s for _, s, _ in frames)
                self.silent = 0
                self.ring.clear()
            return None
        self.cur.append(fr)
        if sp:
            self.cur_speech += 1
            self.silent = 0
        else:
            self.silent += 1
        if self.silent >= self.hangover:
            return self._close("end", drop=self.silent - self.tail)
        if len(self.cur) * FRAME >= self.max_s * RATE:
            return self._close("max", drop=0)
        return None

    def _close(self, reason: str, drop: int) -> Optional[Segment]:
        frames = self.cur[:len(self.cur) - drop] if drop > 0 else self.cur
        seg = Segment(self.cur_start, np.concatenate(frames), self.cur_speech, reason)
        self.cur = None
        self.silent = 0
        return seg if seg.speech_frames >= self.min_speech else None

    def flush(self) -> Optional[Segment]:
        """The open segment, if any (the stream ended)."""
        if self.cur is None:
            return None
        return self._close("end", drop=max(self.silent - self.tail, 0))
