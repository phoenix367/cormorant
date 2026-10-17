"""miranda_proto.py — the wire formats shared by the Miranda client and server
(Python standard library only).

Microphone audio: RTP (RFC 3550) over UDP, client -> server
  payload type 96 (dynamic) = L16/16000/1: 16-bit signed samples, big-endian
  (network order, RFC 3551), mono, 16 kHz, 20 ms (320 samples) per packet;
  the SSRC is the client's, announced in its hello; the marker bit opens a
  talkspurt (the first packet, and the first after the mic was paused — the
  timestamp then jumps over the pause, the sequence number does not).

Control: one TCP connection per session, both directions, frames of
  kind (1 byte) + length (4 bytes, big-endian) + payload:
    b"J"  a JSON object (UTF-8)
    b"A"  Miranda's speech: raw s16le mono PCM at the rate of the last
          audio_start message
  client -> server  {"type": "hello", "version", "ssrc", "codec", "ptime"},
                    {"type": "playback_done"}, {"type": "bye"}
  server -> client  {"type": "welcome", "session", "rtp_port", "wake_phrase", ...},
                    {"type": "state", "state": warming | listening | awake |
                     thinking | speaking}, {"type": "heard", "text", "wake"},
                    {"type": "query", "text"}, {"type": "answer_delta", "text"},
                    {"type": "answer_done", "text", "info"},
                    {"type": "audio_start", "rate"}, b"A" frames,
                    {"type": "audio_end", "samples"}, {"type": "error", "message"}
"""

from __future__ import annotations

import json
import socket
import struct
import threading
from typing import Any, Optional, Tuple

VERSION = 1
PORT = 8770                         # TCP control and UDP RTP, the same number
RATE = 16000                        # the microphone stream
PTIME_MS = 20
FRAME = RATE * PTIME_MS // 1000     # samples per RTP packet (320)
PT_L16 = 96
CODEC = "L16/16000/1"
RTP_HEADER = struct.Struct("!BBHII")
MAX_FRAME_BYTES = 8 << 20           # a control frame's payload at most


# ── RTP ──────────────────────────────────────────────────────────────────────

def rtp_pack(seq: int, ts: int, ssrc: int, payload: bytes, marker: bool = False, pt: int = PT_L16) -> bytes:
    """One RTP packet: V=2, no padding / extension / CSRCs."""
    return RTP_HEADER.pack(0x80, (0x80 if marker else 0) | pt, seq & 0xFFFF, ts & 0xFFFFFFFF,
                           ssrc & 0xFFFFFFFF) + payload


def rtp_unpack(pkt: bytes):
    """(pt, marker, seq, ts, ssrc, payload) of an RTP packet, or None when it
    is not one (version, length, padding, extension and CSRCs handled)."""
    if len(pkt) < RTP_HEADER.size:
        return None
    b0, b1, seq, ts, ssrc = RTP_HEADER.unpack_from(pkt)
    if b0 >> 6 != 2:
        return None
    off = RTP_HEADER.size + 4 * (b0 & 0x0F)
    if b0 & 0x10:                                              # header extension
        if len(pkt) < off + 4:
            return None
        off += 4 + 4 * struct.unpack_from("!H", pkt, off + 2)[0]
    end = len(pkt)
    if b0 & 0x20:                                              # padding
        end -= pkt[-1]
    if off > end:
        return None
    return b1 & 0x7F, bool(b1 & 0x80), seq, ts, ssrc, pkt[off:end]


def l16_to_le(payload: bytes) -> bytes:
    """Network-order L16 -> little-endian s16 (and back: the same swap)."""
    import array
    a = array.array("h")
    a.frombytes(payload[:len(payload) // 2 * 2])
    a.byteswap()
    return a.tobytes()


le_to_l16 = l16_to_le


class RtpReceiver:
    """Puts one SSRC's packets back in order: in-order packets pass, a small
    loss is filled with silence, late (older) packets are dropped, a large jump
    or a marker restarts the stream.  feed() returns (samples_le, restart)."""

    def __init__(self, max_fill: int = 10):
        self.next_seq: Optional[int] = None
        self.max_fill = max_fill
        self.lost = self.late = self.packets = 0

    def feed(self, seq: int, marker: bool, payload: bytes) -> Tuple[bytes, bool]:
        self.packets += 1
        pcm = l16_to_le(payload)
        if self.next_seq is None or marker:
            self.next_seq = (seq + 1) & 0xFFFF
            return pcm, True
        d = (seq - self.next_seq) & 0xFFFF
        if d == 0:
            self.next_seq = (seq + 1) & 0xFFFF
            return pcm, False
        if d >= 0x8000:                                        # behind: a late or duplicate packet
            self.late += 1
            return b"", False
        self.next_seq = (seq + 1) & 0xFFFF
        if d <= self.max_fill:                                 # d packets lost: silence in their place
            self.lost += d
            return b"\0" * (len(pcm) * d) + pcm, False
        return pcm, True


# ── the control channel ──────────────────────────────────────────────────────

class Channel:
    """Framed messages over a TCP socket; send() is thread-safe."""

    def __init__(self, sock: socket.socket):
        self.sock = sock
        self.rfile = sock.makefile("rb")
        self._lock = threading.Lock()
        self.closed = False

    def _send(self, kind: bytes, payload: bytes) -> None:
        with self._lock:
            self.sock.sendall(kind + struct.pack("!I", len(payload)) + payload)

    def send(self, obj: dict) -> None:
        self._send(b"J", json.dumps(obj, separators=(",", ":")).encode())

    def send_audio(self, pcm: bytes) -> None:
        if pcm:
            self._send(b"A", pcm)

    def recv(self) -> Optional[Tuple[str, Any]]:
        """("json", dict) | ("audio", bytes), or None at the end of the stream."""
        head = self.rfile.read(5)
        if len(head) < 5:
            return None
        kind, n = head[:1], struct.unpack("!I", head[1:])[0]
        if n > MAX_FRAME_BYTES:
            raise ValueError(f"control frame of {n} bytes")
        body = self.rfile.read(n)
        if len(body) < n:
            return None
        if kind == b"J":
            return "json", json.loads(body.decode())
        if kind == b"A":
            return "audio", body
        raise ValueError(f"unknown control frame kind {kind!r}")

    def close(self) -> None:
        self.closed = True
        for f in (lambda: self.sock.shutdown(socket.SHUT_RDWR), self.rfile.close, self.sock.close):
            try:
                f()
            except OSError:
                pass
