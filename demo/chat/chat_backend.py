"""chat_backend.py — the backend interface of the KV260 chat server
(kv260_chat_server.py).  Python standard library only.

A backend serves one model id.  The server owns HTTP, the OpenAI protocol,
authentication, the FPGA request queue and client-disconnect detection; a
backend only turns a validated ChatRequest into text:

    class MyBackend(Backend):
        model_id = "smollm2-135m-instruct"

        def load(self):                    # once, before the server listens
            ...                            # dlopen the library, init the model
        def prepare(self, req):            # outside the FPGA lock: tokenize,
            return job                     # apply the chat template, validate
        def generate(self, job, cancel):   # under the FPGA lock
            for step in ...:
                cancel.check()             # raises Cancelled if the client left
                yield Delta(text_piece)
            yield Finish("stop", prompt_tokens=..., completion_tokens=...)
        def close(self): ...

Contract:
  * prepare() may raise BackendError (-> an OpenAI-style 4xx JSON error);
    it must not touch the FPGA.
  * generate() yields zero or more Delta(text) and then exactly one Finish.
    It runs with the FPGA to itself (one request at a time, FIFO); it calls
    cancel.check() between steps — a window, a token — and lets Cancelled
    propagate (the server then releases the FPGA and logs the request as
    cancelled).  Raising BackendError before the first yield still gives
    the client a 4xx; after it, streaming clients get an SSE error event.
  * Finish.reason is "stop" (natural end / a stop string) or "length"
    (max_tokens); Finish.info is returned to the client under the
    non-standard "kv260" key and its "log" dict (if any) goes into the
    server's per-request log line.
  * Parameters the backend cannot honour must be rejected in prepare() or
    documented (e.g. temperature for an extractive model).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union


class BackendError(Exception):
    """A request the backend cannot serve.  Becomes an OpenAI-style error
    response: {"error": {"message", "type", "param", "code"}}."""

    def __init__(self, message: str, param: Optional[str] = None, status: int = 400,
                 code: Optional[str] = None, type_: str = "invalid_request_error"):
        super().__init__(message)
        self.message, self.param, self.status, self.code, self.type = \
            message, param, status, code, type_


class Cancelled(Exception):
    """The client disconnected; generation stops at the next step."""


class CancelToken:
    """Set by the server when the client goes away.  `probe` is polled by
    cancelled() (the server passes a non-blocking check of the client
    socket), so a backend that is between steps sees a disconnect without
    writing to the socket."""

    def __init__(self, probe: Optional[Callable[[], bool]] = None):
        self._ev = threading.Event()
        self._probe = probe

    def cancel(self) -> None:
        self._ev.set()

    def cancelled(self) -> bool:
        if not self._ev.is_set() and self._probe is not None:
            try:
                if self._probe():
                    self._ev.set()
            except Exception:                                  # noqa: BLE001
                self._ev.set()
        return self._ev.is_set()

    def check(self) -> None:
        if self.cancelled():
            raise Cancelled()


@dataclass
class ChatRequest:
    """A validated /v1/chat/completions request (the server fills it)."""
    model: str
    messages: List[Dict[str, str]]         # [{"role", "content": str}, ...]
    stream: bool = False
    max_tokens: Optional[int] = None       # max_tokens or max_completion_tokens
    temperature: Optional[float] = None    # 0 .. 2
    top_p: Optional[float] = None          # 0 .. 1
    stop: List[str] = field(default_factory=list)   # up to 4 strings
    seed: Optional[int] = None
    raw: Dict[str, Any] = field(default_factory=dict)   # the request JSON, for extras


@dataclass
class Delta:
    text: str


@dataclass
class Finish:
    reason: str                            # "stop" | "length"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    info: Dict[str, Any] = field(default_factory=dict)


Event = Union[Delta, Finish]


class Backend:
    """Base class; see the module docstring for the contract."""

    model_id: str = "?"
    owned_by: str = "kv260"
    fingerprint: Optional[str] = None      # -> "system_fingerprint"

    def load(self) -> None:
        pass

    def close(self) -> None:
        pass

    def health(self) -> Dict[str, Any]:
        """Extra fields for GET /health."""
        return {}

    def prepare(self, req: ChatRequest) -> Any:
        return req

    def generate(self, job: Any, cancel: CancelToken) -> Iterator[Event]:
        raise NotImplementedError


# ── helpers for backends ─────────────────────────────────────────────────────

def apply_stop(text: str, stops: List[str]) -> Tuple[str, bool]:
    """Cut text before the earliest stop string.  Returns (text, stopped)."""
    cut = min((i for i in (text.find(s) for s in stops if s) if i >= 0), default=-1)
    return (text[:cut], True) if cut >= 0 else (text, False)


class StopStream:
    """Incremental stop-string matcher for token streams: feed() returns the
    text that is safe to emit (a possible stop-string prefix at the end is
    held back) and whether a stop string was hit."""

    def __init__(self, stops: List[str]):
        self.stops = [s for s in stops if s]
        self.buf = ""
        self.hold = max((len(s) for s in self.stops), default=0) - 1

    def feed(self, piece: str) -> Tuple[str, bool]:
        self.buf += piece
        text, stopped = apply_stop(self.buf, self.stops)
        if stopped:
            self.buf = ""
            return text, True
        keep = 0
        for k in range(min(self.hold, len(self.buf)), 0, -1):
            if any(s.startswith(self.buf[-k:]) for s in self.stops):
                keep = k
                break
        out, self.buf = (self.buf[:-keep], self.buf[-keep:]) if keep else (self.buf, "")
        return out, False

    def flush(self) -> str:
        out, self.buf = self.buf, ""
        return out
