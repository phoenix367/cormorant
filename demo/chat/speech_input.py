"""speech_input.py — uploaded audio -> 16 kHz mono int16 samples for the
speech-to-text backend (POST /v1/audio/transcriptions; doc/plans/ASR_PLAN.md
§7).  numpy; ffmpeg for anything but WAV.

  WAV       read here: PCM 8 / 16 / 24 / 32-bit, IEEE float 32 / 64, the
            WAVE_FORMAT_EXTENSIBLE forms of both; any rate, any channel
            count (mixed to mono by averaging)
  others    mp3, flac, ogg / opus, m4a / aac, webm, ... through ffmpeg
            (s16le, mono, 16 kHz out); without ffmpeg only WAV is accepted
  resample  a Kaiser-windowed sinc, polyphase (32 zero crossings, cut-off
            at 0.94 of the lower Nyquist rate); 16 kHz mono PCM 16-bit
            passes through unchanged, sample for sample
"""

from __future__ import annotations

import math
import struct
import subprocess
from typing import Optional, Tuple

RATE = 16000
ZEROS = 32                     # zero crossings of the sinc on each side
ROLLOFF = 0.94                 # cut-off, relative to the lower Nyquist rate
BETA = 9.6                     # the Kaiser window's shape (~ -100 dB side lobes)
MAX_PHASES = 4096              # polyphase table rows; finer rate ratios snap to this grid
BLOCK = 8192                   # output samples per vectorised block


class AudioError(ValueError):
    """Audio the server cannot read (the message is for the client)."""


def is_wav(data: bytes) -> bool:
    return len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE"


def read_wav(data: bytes):
    """(rate, samples [frames][channels] float64 in int16 units, exact16):
    exact16 when the file is 16-bit PCM (the samples are its integers)."""
    import numpy as np
    if not is_wav(data):
        raise AudioError("not a WAV file (RIFF / WAVE)")
    pos, fmt, body = 12, None, None
    while pos + 8 <= len(data):
        cid, size = data[pos:pos + 4], struct.unpack_from("<I", data, pos + 4)[0]
        start = pos + 8
        if cid == b"fmt ":
            fmt = data[start:start + size]
        elif cid == b"data":
            end = len(data) if size in (0, 0xFFFFFFFF) or start + size > len(data) else start + size
            body = data[start:end]
            if fmt is not None:
                break
        pos = start + size + (size & 1)
    if fmt is None or len(fmt) < 16:
        raise AudioError("WAV file without a valid 'fmt ' chunk")
    if body is None:
        raise AudioError("WAV file without a 'data' chunk")
    tag, ch, rate, _, align, bits = struct.unpack_from("<HHIIHH", fmt, 0)
    if tag == 0xFFFE and len(fmt) >= 26:                       # WAVE_FORMAT_EXTENSIBLE: the sub-format GUID
        tag = struct.unpack_from("<H", fmt, 24)[0]
    if ch < 1 or rate < 1 or align < 1:
        raise AudioError(f"WAV file with {ch} channels at {rate} Hz")
    width = align // ch
    n = len(body) // align
    raw = body[:n * align]
    if tag == 1 and width == 2:
        x = np.frombuffer(raw, "<i2").reshape(n, ch).astype(np.float64)
        return rate, x, True
    if tag == 1 and width == 1:
        x = (np.frombuffer(raw, np.uint8).astype(np.float64) - 128.0) * 256.0
    elif tag == 1 and width == 3:
        b = np.frombuffer(raw, np.uint8).reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        x = (np.where(v >= 1 << 23, v - (1 << 24), v)).astype(np.float64) / 256.0
    elif tag == 1 and width == 4:
        x = np.frombuffer(raw, "<i4").astype(np.float64) / 65536.0
    elif tag == 3 and width in (4, 8):
        x = np.frombuffer(raw, "<f4" if width == 4 else "<f8").astype(np.float64) * 32768.0
    else:
        raise AudioError(f"WAV encoding not supported (format {tag:#06x}, {bits} bits): send PCM or "
                         f"IEEE float, or another format for ffmpeg")
    return rate, x.reshape(n, ch), False


def _kernel(t, fc):
    """The windowed sinc at offsets t (input samples), cut-off fc (cycles per
    input sample x 2, i.e. relative to the input Nyquist rate)."""
    import numpy as np
    half = ZEROS / fc
    w = np.where(np.abs(t) < half, np.i0(BETA * np.sqrt(np.clip(1.0 - (t / half) ** 2, 0.0, 1.0))) / np.i0(BETA),
                 0.0)
    return fc * np.sinc(fc * t) * w


def resample(x, src: int, dst: int = RATE):
    """x [n] float64 at src Hz -> float64 at dst Hz (round(n * dst / src) samples)."""
    import numpy as np
    if src == dst or len(x) == 0:
        return np.asarray(x, np.float64)
    g = math.gcd(src, dst)
    up, down = dst // g, src // g                              # output n is at input n * down / up
    fc = min(1.0, dst / src) * ROLLOFF
    half = int(math.ceil(ZEROS / fc))
    taps = 2 * half + 1
    phases = min(up, MAX_PHASES)
    frac = np.arange(phases) / phases                          # the output's offset past its base sample
    table = _kernel(frac[:, None] + half - np.arange(taps)[None, :], fc)   # [phase][tap]
    n_out = (len(x) * dst + src // 2) // src
    xp = np.concatenate([np.zeros(half), np.asarray(x, np.float64), np.zeros(half + 1)])
    out = np.empty(n_out)
    j = np.arange(taps)
    for b0 in range(0, n_out, BLOCK):
        n = np.arange(b0, min(b0 + BLOCK, n_out), dtype=np.int64)
        num = n * down
        base = num // up                                       # the input sample at or before
        ph = ((num - base * up) * phases) // up                # which phase row
        out[b0:b0 + len(n)] = np.einsum("ij,ij->i", xp[base[:, None] + j[None, :]], table[ph])
    return out


def to_int16(x):
    import numpy as np
    return np.clip(np.rint(x), -32768, 32767).astype(np.int16)


def decode_ffmpeg(data: bytes, ffmpeg: str, timeout: float = 120.0):
    """Any format ffmpeg reads -> 16 kHz mono int16."""
    import numpy as np
    try:
        r = subprocess.run([ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
                            "-f", "s16le", "-ac", "1", "-ar", str(RATE), "pipe:1"],
                           input=data, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise AudioError(f"decoding the audio took longer than {timeout:.0f} s") from None
    if r.returncode != 0:
        msg = r.stderr.decode("utf-8", "replace").strip().splitlines()
        raise AudioError("ffmpeg could not decode the file" + (f": {msg[-1]}" if msg else ""))
    return np.frombuffer(r.stdout[:len(r.stdout) // 2 * 2], "<i2").copy()


def decode_audio(data: bytes, ffmpeg: Optional[str] = None) -> Tuple["object", dict]:
    """(16 kHz mono int16 samples, info {"source", "rate", "channels"}) of an
    uploaded file: WAV here, anything else through ffmpeg."""
    if not data:
        raise AudioError("the file is empty")
    if is_wav(data):
        try:
            rate, x, exact = read_wav(data)
        except AudioError:
            if not ffmpeg:
                raise
            return decode_ffmpeg(data, ffmpeg), {"source": "ffmpeg", "rate": None, "channels": None}
        ch = x.shape[1]
        if exact and ch == 1 and rate == RATE:
            return x[:, 0].astype("int16"), {"source": "wav", "rate": rate, "channels": ch}
        mono = x.mean(axis=1) if ch > 1 else x[:, 0]
        return to_int16(resample(mono, rate, RATE)), {"source": "wav", "rate": rate, "channels": ch}
    if not ffmpeg:
        raise AudioError("only WAV files can be read without ffmpeg on the server (send PCM WAV, or "
                         "install ffmpeg for mp3 / flac / ogg / m4a / webm)")
    return decode_ffmpeg(data, ffmpeg), {"source": "ffmpeg", "rate": None, "channels": None}
