#!/usr/bin/env python3
"""
kv260_chat_server.py — OpenAI-compatible chat server for the KV260
(Python 3.8+ standard library only; runs on the board).

Endpoints
  GET  /health                  status, models, queue (no API key needed)
  GET  /v1/models               {"object": "list", "data": [model, ...]}
  GET  /v1/models/{id}          one model
  POST /v1/chat/completions     stream false -> one chat.completion JSON;
                                stream true  -> SSE "data: {chat.completion.chunk}"
                                lines, then "data: [DONE]"
  (the same paths without the /v1 prefix are accepted too)

Backends (chat_backend.Backend; one model id each)
  bert-squad   extractive question answering over a document with BERT-base
               SQuAD on the FPGA (bert_squad_backend.py, libbert_squad.so)
  smollm2      model id smollm2-135m-instruct: generative chat with
               SmolLM2-135M-Instruct on the FPGA (smollm2_backend.py,
               libsmollm2.so; tokenizer, template and sampling on the host)
  echo         repeats the last user message word by word; no FPGA — for
               trying clients against the protocol

Residency (--resident): which FPGA models are loaded (the CMA pool is tight:
BERT holds ~224 MB, SmolLM2 ~360 MB, idle CmaFree was 626-813 MB).
  auto (default)  the first backend loads at startup, the others when first
                  requested; before a load, models are evicted (least recently
                  used first) while CmaFree < the new model's cma_mb +
                  --cma-margin-mb, and once more if the load fails anyway —
                  so both stay resident when they fit, else they swap
  one             at most one FPGA model at a time: switching models unloads
                  the current one (a swap costs a load, seconds)
  all             everything loads at startup and stays (the old behaviour)
Loads and evictions happen under the FPGA lock, between requests.

Request parameters honoured: model, messages, stream, stream_options
.include_usage, max_tokens / max_completion_tokens, temperature, top_p, stop,
seed, n (= 1 only); other fields are ignored.  Invalid values get
OpenAI-style JSON errors ({"error": {"message", "type", "param", "code"}}):
400 bad request, 401 API key, 404 unknown model / URL, 413 body too large,
503 busy (queue full or timed out) or model not loaded.

One FPGA: requests run one at a time in arrival order; a request waits up to
--queue-timeout seconds (then 503).  A client that disconnects — while queued
or while its answer is computed — cancels its request at the next step (a
BERT window, a decoder token).  One log line per request on stderr.

usage:
  kv260_chat_server.py [--host 0.0.0.0] [--port 8000]
                       [--api-key KEY | --api-key-file FILE]
                       [--backend bert-squad] [--backend smollm2] [--backend echo] ...
                       [--resident auto|one|all] [--cma-margin-mb 32]
                       [--bert-lib lib/libbert_squad.so] [--bert-weights DIR]
                       [--vocab vocab.txt] [--max-windows 8] [--doc-stride 128]
                       [--llm-lib lib/libsmollm2.so] [--llm-weights DIR]
                       [--llm-tokenizer tokenizer.json] [--llm-* sampling defaults]
                       [--queue-timeout 120] [--max-queue 16]
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import hmac
import json
import os
import select
import signal
import socket
import sys
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional
from urllib.parse import urlsplit

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.join(HERE, "..", "bert_squad", "scripts")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from chat_backend import (Backend, BackendError, Cancelled, CancelToken,  # noqa: E402
                          ChatRequest, Delta, Finish, apply_stop)

VERSION = "1.0"
MAX_BODY = 4 << 20           # bytes; documents go into the messages
ROLES = ("system", "developer", "user", "assistant", "tool", "function")


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", file=sys.stderr, flush=True)


# ── request validation ───────────────────────────────────────────────────────

def _type_name(v: Any) -> str:
    return {bool: "a boolean", int: "an integer", float: "a number", str: "a string",
            list: "an array", dict: "an object", type(None): "null"}.get(type(v), type(v).__name__)


def _number(body: dict, key: str, lo: float, hi: Optional[float], integer: bool = False):
    v = body.get(key)
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)) or (integer and not isinstance(v, int)):
        raise BackendError(f"Invalid type for '{key}': expected {'an integer' if integer else 'a number'},"
                           f" but got {_type_name(v)} instead.", key, code="invalid_type")
    if v < lo:
        raise BackendError(f"Invalid '{key}': expected a value >= {lo}, but got {v} instead.", key,
                           code="invalid_value")
    if hi is not None and v > hi:
        raise BackendError(f"Invalid '{key}': expected a value <= {hi}, but got {v} instead.", key,
                           code="invalid_value")
    return v


def _content(m: dict, i: int) -> str:
    c = m.get("content")
    if c is None:
        return ""
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts = []
        for j, p in enumerate(c):
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict) and p.get("type") in ("text", "input_text"):
                parts.append(str(p.get("text") or ""))
            else:
                raise BackendError(f"Invalid content part in messages[{i}].content[{j}]: only text "
                                   f"parts are supported.", f"messages.[{i}].content.[{j}]")
        return "\n".join(parts)
    raise BackendError(f"Invalid type for 'messages[{i}].content': expected a string or an array "
                       f"of text parts, but got {_type_name(c)} instead.", f"messages.[{i}].content")


def parse_chat_request(body: Any, default_model: str) -> ChatRequest:
    """Validate a /v1/chat/completions body (unknown fields are ignored)."""
    if not isinstance(body, dict):
        raise BackendError("The request body must be a JSON object.")
    model = body.get("model", default_model)
    if not isinstance(model, str) or not model:
        raise BackendError("Invalid 'model': expected a non-empty string.", "model")
    msgs = body.get("messages")
    if msgs is None:
        raise BackendError("Missing required parameter: 'messages'.", "messages",
                           code="missing_required_parameter")
    if not isinstance(msgs, list) or not msgs:
        raise BackendError("Invalid 'messages': expected a non-empty array of messages.", "messages")
    messages = []
    for i, m in enumerate(msgs):
        if not isinstance(m, dict):
            raise BackendError(f"Invalid type for 'messages[{i}]': expected an object.", f"messages.[{i}]")
        role = m.get("role")
        if role not in ROLES:
            raise BackendError(f"Invalid value for 'messages[{i}].role': '{role}'. Supported values "
                               f"are: {', '.join(repr(r) for r in ROLES)}.", f"messages.[{i}].role")
        messages.append({"role": role, "content": _content(m, i)})
    stream = body.get("stream", False)
    if stream is None:
        stream = False
    if not isinstance(stream, bool):
        raise BackendError(f"Invalid type for 'stream': expected a boolean, but got "
                           f"{_type_name(stream)} instead.", "stream")
    n = _number(body, "n", 1, None, integer=True)
    if n not in (None, 1):
        raise BackendError("Invalid 'n': this server generates one choice per request (n = 1).", "n")
    max_tokens = _number(body, "max_completion_tokens", 1, None, integer=True)
    if max_tokens is None:
        max_tokens = _number(body, "max_tokens", 1, None, integer=True)
    stop = body.get("stop")
    if stop is None:
        stop = []
    elif isinstance(stop, str):
        stop = [stop]
    elif isinstance(stop, list) and all(isinstance(s, str) for s in stop):
        if len(stop) > 4:
            raise BackendError("Invalid 'stop': at most 4 stop sequences are allowed.", "stop")
    else:
        raise BackendError("Invalid 'stop': expected a string or an array of strings.", "stop")
    seed = _number(body, "seed", -2 ** 63, 2 ** 63 - 1, integer=True)
    return ChatRequest(model=model, messages=messages, stream=stream, max_tokens=max_tokens,
                       temperature=_number(body, "temperature", 0, 2),
                       top_p=_number(body, "top_p", 0, 1), stop=stop, seed=seed, raw=body)


# ── the FPGA queue ───────────────────────────────────────────────────────────

class FifoLock:
    """A mutex granted in arrival order.  acquire() gives up after `timeout`
    seconds, when `max_waiting` requests already wait, or when
    `cancelled()` turns true (polled every `poll` seconds)."""

    def __init__(self, max_waiting: int = 16):
        self.max_waiting = max_waiting
        self._cv = threading.Condition()
        self._queue: "collections.deque[object]" = collections.deque()
        self._busy = False

    @property
    def busy(self) -> bool:
        return self._busy

    @property
    def waiting(self) -> int:
        return len(self._queue)

    def acquire(self, timeout: float, cancelled=lambda: False, poll: float = 0.25) -> str:
        """'ok', 'full', 'timeout' or 'cancelled'."""
        with self._cv:
            if not self._busy and not self._queue:
                self._busy = True
                return "ok"
            if len(self._queue) >= self.max_waiting:
                return "full"
            ticket = object()
            self._queue.append(ticket)
            deadline = time.monotonic() + timeout
            try:
                while self._busy or self._queue[0] is not ticket:
                    left = deadline - time.monotonic()
                    if left <= 0:
                        return "timeout"
                    if cancelled():
                        return "cancelled"
                    self._cv.wait(min(left, poll))
                self._busy = True
                return "ok"
            finally:
                self._queue.remove(ticket)
                self._cv.notify_all()

    def release(self) -> None:
        with self._cv:
            self._busy = False
            self._cv.notify_all()


# ── the echo backend (no FPGA) ───────────────────────────────────────────────

class EchoBackend(Backend):
    """Repeats the last user message word by word (one Delta per word) —
    a stand-in model for trying clients and the protocol without the FPGA."""

    model_id = "echo"
    fingerprint = "kv260-echo"
    uses_fpga = False

    def __init__(self, delay: float = 0.0):
        self.delay = delay

    def generate(self, req: ChatRequest, cancel: CancelToken):
        last = next((m["content"] for m in reversed(req.messages) if m["role"] == "user"), "")
        words = last.split() or ["(nothing", "to", "echo)"]
        prompt = sum(len(m["content"].split()) for m in req.messages)
        text, n = "", 0
        for i, w in enumerate(words):
            if req.max_tokens is not None and n >= req.max_tokens:
                yield Finish("length", prompt, n)
                return
            cancel.check()
            if self.delay:
                time.sleep(self.delay)
            piece = ("" if i == 0 else " ") + w
            cut, stopped = apply_stop(text + piece, req.stop)
            if stopped:
                if len(cut) > len(text):
                    yield Delta(cut[len(text):])
                yield Finish("stop", prompt, n)
                return
            text += piece
            n += 1
            yield Delta(piece)
        yield Finish("stop", prompt, n)


# ── residency (CMA) ──────────────────────────────────────────────────────────

def read_cma_free_mb(path: str = "/proc/meminfo") -> Optional[float]:
    """CmaFree in MB (None where the kernel has no CMA, e.g. a host PC)."""
    try:
        with open(path) as f:
            for line in f:
                if line.startswith("CmaFree:"):
                    return int(line.split()[1]) / 1024.0
    except (OSError, ValueError, IndexError):
        pass
    return None


# ── HTTP ─────────────────────────────────────────────────────────────────────

class ChatServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 32

    def __init__(self, addr, backends: Dict[str, Backend], *, api_key: Optional[str] = None,
                 queue_timeout: float = 120.0, max_queue: int = 16, max_body: int = MAX_BODY,
                 resident: str = "all", cma_margin_mb: float = 32.0, cma_free=read_cma_free_mb):
        super().__init__(addr, ChatHandler)
        if resident not in ("all", "one", "auto"):
            raise ValueError(f"resident: {resident!r}")
        self.backends = backends
        self.default_model = next(iter(backends))
        self.api_key = api_key or None
        self.queue_timeout = queue_timeout
        self.max_body = max_body
        self.fpga = FifoLock(max_queue)
        self.started = time.time()
        self.n_requests = 0
        self.count_lock = threading.Lock()
        self.load_errors: Dict[str, str] = {}
        self.resident = resident
        self.cma_margin_mb = cma_margin_mb
        self.cma_free = cma_free
        self.loaded: Dict[str, bool] = {m: False for m in backends}
        self.last_used: Dict[str, float] = {}
        self.n_loads = self.n_unloads = 0

    # Loads and unloads run at startup / shutdown or under the FPGA lock.

    def load_model(self, mid: str, evict: bool = True) -> float:
        """Make `mid` resident (evicting others per --resident); returns the
        seconds spent (0 if it already was).  Raises what load() raises.
        With --resident auto a failed load is retried once after evicting the
        other FPGA models (CmaFree may be unknown or the estimate short) —
        not when the library itself cannot be opened (OSError from dlopen)."""
        b = self.backends[mid]
        if self.loaded[mid]:
            return 0.0
        t0 = time.monotonic()
        others = [m for m, on in self.loaded.items()
                  if on and m != mid and self.backends[m].uses_fpga]
        if b.uses_fpga and others and self.resident == "one":
            for m in others:
                self.unload_model(m, "one model at a time")
        elif b.uses_fpga and others and self.resident == "auto":
            need = b.cma_mb + self.cma_margin_mb
            for m in sorted(others, key=lambda m: self.last_used.get(m, 0.0)):
                free = self.cma_free()
                if free is None or free >= need:
                    break
                self.unload_model(m, f"CmaFree {free:.0f} MB < {need:.0f} MB for '{mid}'")
        try:
            b.load()
        except Exception as e:                                 # noqa: BLE001
            still = [m for m, on in self.loaded.items()
                     if on and m != mid and self.backends[m].uses_fpga]
            if not (evict and b.uses_fpga and still and self.resident == "auto") \
                    or isinstance(e, OSError):
                raise
            log(f"loading '{mid}' failed ({e}); unloading {', '.join(still)} and retrying")
            with contextlib.suppress(Exception):
                b.unload()
            for m in still:
                self.unload_model(m, f"'{mid}' did not load beside it")
            b.load()
        self.loaded[mid] = True
        self.load_errors.pop(mid, None)
        self.n_loads += 1
        dt = time.monotonic() - t0
        free = self.cma_free()
        log(f"loaded '{mid}' in {dt:.1f} s" + (f" (CmaFree {free:.0f} MB)" if free is not None else ""))
        return dt

    def unload_model(self, mid: str, why: str = "") -> None:
        if not self.loaded.get(mid):
            return
        try:
            self.backends[mid].unload()
        finally:
            self.loaded[mid] = False
            self.n_unloads += 1
        log(f"unloaded '{mid}'" + (f": {why}" if why else ""))

    def startup_models(self) -> list:
        """The models to load before listening: all (resident all), else the
        first FPGA model and the non-FPGA ones; with auto, further FPGA
        models while CmaFree allows (the rest load on first request)."""
        if self.resident == "all":
            return list(self.backends)
        first = next((m for m, b in self.backends.items() if b.uses_fpga), None)
        return [m for m, b in self.backends.items() if not b.uses_fpga or m == first]

    def load_startup(self) -> None:
        for mid in self.startup_models():
            self.load_model(mid)
        if self.resident == "auto":
            for mid, b in self.backends.items():
                if self.loaded[mid]:
                    continue
                free = self.cma_free()
                if free is not None and free >= b.cma_mb + self.cma_margin_mb:
                    try:
                        self.load_model(mid, evict=False)
                    except Exception as e:                     # noqa: BLE001
                        with contextlib.suppress(Exception):
                            b.unload()
                        log(f"'{mid}' did not load at startup ({e}); it loads on first request")

    def close_models(self) -> None:
        for mid, b in self.backends.items():
            try:
                b.close()
            except Exception as e:                             # noqa: BLE001
                log(f"closing '{mid}': {e}")
            self.loaded[mid] = False

    def model_obj(self, mid: str) -> dict:
        return {"id": mid, "object": "model", "created": int(self.started),
                "owned_by": self.backends[mid].owned_by}


class ChatHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "kv260-chat/" + VERSION
    sys_version = ""
    server: ChatServer

    timeout = 600                              # idle keep-alive connections

    def log_message(self, format, *args):      # noqa: A002 — per-request lines are ours
        pass

    def log_error(self, format, *args):        # noqa: A002
        log(f"{self.client_address[0]} " + (format % args))

    # ── plumbing ──

    def _path(self) -> str:
        p = urlsplit(self.path).path.rstrip("/") or "/"
        return p[3:] if p.startswith("/v1/") or p == "/v1" else p

    def _send_json(self, status: int, obj: Any, headers: Optional[Dict[str, str]] = None) -> None:
        data = (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def _error(self, status: int, message: str, type_: str = "invalid_request_error",
               param: Optional[str] = None, code: Optional[str] = None,
               headers: Optional[Dict[str, str]] = None) -> None:
        self._status = status
        self._send_json(status, {"error": {"message": message, "type": type_,
                                           "param": param, "code": code}}, headers)

    def _backend_error(self, e: BackendError) -> None:
        self._error(e.status, e.message, e.type, e.param, e.code)

    def _auth(self) -> bool:
        key = self.server.api_key
        if not key:
            return True
        h = self.headers.get("Authorization", "")
        tok = h[7:].strip() if h[:7].lower() == "bearer " else (
            self.headers.get("api-key") or self.headers.get("x-api-key"))
        if tok and hmac.compare_digest(tok.encode(), key.encode()):
            return True
        self._error(401, "Incorrect API key provided." if tok else
                    "You didn't provide an API key. Send it as 'Authorization: Bearer <key>'.",
                    param=None, code="invalid_api_key")
        return False

    def _read_body(self) -> bytes:
        limit = self.server.max_body
        if "chunked" in self.headers.get("Transfer-Encoding", "").lower():
            out = bytearray()
            while True:
                size = int(self.rfile.readline().split(b";")[0].strip() or b"0", 16)
                if size == 0:
                    while self.rfile.readline() not in (b"\r\n", b"\n", b""):
                        pass
                    return bytes(out)
                if len(out) + size > limit:
                    self.close_connection = True
                    raise BackendError(f"Request body too large (limit {limit} bytes).", status=413)
                out += self.rfile.read(size)
                self.rfile.readline()
        n = self.headers.get("Content-Length")
        if n is None:
            self.close_connection = True
            raise BackendError("Missing Content-Length.", status=411)
        n = int(n)
        if n > limit:
            self.close_connection = True
            raise BackendError(f"Request body too large ({n} bytes, limit {limit}).", status=413)
        return self.rfile.read(n)

    def _peer_gone(self) -> bool:
        """Non-blocking: has the client closed its end of the connection?"""
        sock = self.connection
        try:
            r, _, _ = select.select([sock], [], [], 0)
            if not r:
                return False
            return sock.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT) == b""
        except (BlockingIOError, InterruptedError):
            return False
        except OSError:
            return True

    # ── routes ──

    def do_GET(self):                                          # noqa: N802
        self._status = 200
        path = self._path()
        if path == "/health":
            return self._health()
        if path == "/models":
            if self._auth():
                self._send_json(200, {"object": "list", "data": [
                    self.server.model_obj(m) for m in self.server.backends]})
            return
        if path.startswith("/models/"):
            if self._auth():
                mid = path[len("/models/"):]
                if mid in self.server.backends:
                    self._send_json(200, self.server.model_obj(mid))
                else:
                    self._error(404, f"The model '{mid}' does not exist or you do not have "
                                f"access to it.", param="model", code="model_not_found")
            return
        self._error(404, f"Unknown request URL: GET {urlsplit(self.path).path}.",
                    code="unknown_url")

    def do_POST(self):                                         # noqa: N802
        self._status = 200
        if self._path() == "/chat/completions":
            return self._chat()
        self.close_connection = True                           # body left unread
        self._error(404, f"Unknown request URL: POST {urlsplit(self.path).path}.",
                    code="unknown_url")

    def _health(self) -> None:
        srv = self.server
        models = []
        for mid, b in srv.backends.items():
            m = {"id": mid, "ready": mid not in srv.load_errors, "loaded": srv.loaded.get(mid, False)}
            if mid in srv.load_errors:
                m["error"] = srv.load_errors[mid]
            try:
                m.update(b.health())
            except Exception as e:                             # noqa: BLE001
                m["health_error"] = str(e)
            models.append(m)
        ok = all(m["ready"] for m in models)
        free = srv.cma_free()
        self._send_json(200 if ok else 503, {
            "status": "ok" if ok else "error", "version": VERSION, "models": models,
            "resident": srv.resident, "cma_free_mb": None if free is None else round(free, 1),
            "loads": srv.n_loads, "unloads": srv.n_unloads,
            "busy": srv.fpga.busy, "waiting": srv.fpga.waiting,
            "requests": srv.n_requests, "uptime_s": round(time.time() - srv.started, 1)})

    # ── chat completions ──

    def _chat(self) -> None:
        srv = self.server
        t0 = time.monotonic()
        rec: Dict[str, Any] = {"model": "-", "stream": "-"}
        try:
            try:
                raw = self._read_body()
            except BackendError as e:
                return self._backend_error(e)
            if not self._auth():
                return
            try:
                body = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, ValueError):
                return self._error(400, "We could not parse the JSON body of your request. "
                                   "(HINT: This likely means you aren't using your HTTP library "
                                   "correctly. The API expects a JSON payload.)")
            try:
                req = parse_chat_request(body, srv.default_model)
            except BackendError as e:
                return self._backend_error(e)
            rec.update(model=req.model, stream=int(req.stream))
            backend = srv.backends.get(req.model)
            if backend is None:
                return self._error(404, f"The model '{req.model}' does not exist or you do not "
                                   f"have access to it.", param="model", code="model_not_found")
            try:
                job = backend.prepare(req)
            except BackendError as e:
                return self._backend_error(e)
            except Exception as e:                             # noqa: BLE001
                traceback.print_exc()
                return self._error(500, f"Internal error in the '{req.model}' backend: {e}",
                                   "server_error", code="backend_error")
            cancel = CancelToken(self._peer_gone)
            got = srv.fpga.acquire(srv.queue_timeout, cancel.cancelled)
            rec["queue_ms"] = (time.monotonic() - t0) * 1000.0
            if got == "cancelled":
                return self._cancelled(rec, "cancelled while queued")
            if got != "ok":
                why = ("the queue is full" if got == "full" else
                       f"the request waited {srv.queue_timeout:.0f} s")
                return self._error(503, f"The server is busy: the FPGA runs one request at a "
                                   f"time and {why}. Retry later.", "server_error",
                                   code="server_busy", headers={"Retry-After": "5"})
            try:
                srv.last_used[req.model] = time.monotonic()
                if not srv.loaded.get(req.model):
                    try:
                        rec["load_ms"] = srv.load_model(req.model) * 1000.0
                    except Exception as e:                     # noqa: BLE001
                        log(f"error: loading '{req.model}' failed: {e}")
                        srv.load_errors[req.model] = str(e)
                        return self._error(503, f"The model '{req.model}' is not available: {e}",
                                           "server_error", code="model_not_loaded")
                gen = backend.generate(job, cancel)
                if req.stream:
                    self._stream(req, backend, gen, cancel, rec, t0)
                else:
                    self._complete(req, backend, gen, rec, t0)
            finally:
                srv.fpga.release()
        except (BrokenPipeError, ConnectionResetError):
            self._cancelled(rec, "client disconnected")
        finally:
            with srv.count_lock:
                srv.n_requests += 1
            self._log_request(rec, t0)

    def _next(self, gen):
        try:
            return next(gen)
        except StopIteration:
            return Finish("stop")

    def _base(self, req: ChatRequest, backend: Backend, obj: str) -> Dict[str, Any]:
        return {"id": self._cid, "object": obj, "created": self._created, "model": req.model,
                "system_fingerprint": backend.fingerprint}

    @staticmethod
    def _usage(fin: Finish) -> Dict[str, int]:
        return {"prompt_tokens": fin.prompt_tokens, "completion_tokens": fin.completion_tokens,
                "total_tokens": fin.prompt_tokens + fin.completion_tokens}

    @staticmethod
    def _info(fin: Finish) -> Dict[str, Any]:
        return {k: v for k, v in fin.info.items() if k != "log"}

    def _new_ids(self) -> None:
        self._cid = "chatcmpl-" + uuid.uuid4().hex[:24]
        self._created = int(time.time())

    def _complete(self, req, backend, gen, rec, t0) -> None:
        self._new_ids()
        parts, fin = [], None
        try:
            while fin is None:
                ev = self._next(gen)
                if isinstance(ev, Delta):
                    if "ttft_ms" not in rec:
                        rec["ttft_ms"] = (time.monotonic() - t0) * 1000.0
                    parts.append(ev.text)
                else:
                    fin = ev
        except Cancelled:
            return self._cancelled(rec, "cancelled (client disconnected)")
        except BackendError as e:
            return self._backend_error(e)
        except Exception as e:                                 # noqa: BLE001
            traceback.print_exc()
            return self._error(500, f"Internal error in the '{req.model}' backend: {e}",
                               "server_error", code="backend_error")
        finally:
            gen.close()
        rec["fin"] = fin
        resp = self._base(req, backend, "chat.completion")
        resp["choices"] = [{"index": 0,
                            "message": {"role": "assistant", "content": "".join(parts),
                                        "refusal": None},
                            "logprobs": None, "finish_reason": fin.reason}]
        resp["usage"] = self._usage(fin)
        if fin.info:
            resp["kv260"] = self._info(fin)
        self._send_json(200, resp)

    def _write(self, data: bytes) -> None:
        if self._chunked:
            self.wfile.write(b"%X\r\n%s\r\n" % (len(data), data))
        else:
            self.wfile.write(data)
        self.wfile.flush()

    def _sse(self, obj: Any) -> None:
        self._write(b"data: " + json.dumps(obj, ensure_ascii=False).encode("utf-8") + b"\n\n")

    def _stream(self, req, backend, gen, cancel, rec, t0) -> None:
        self._new_ids()
        include_usage = bool(isinstance(req.raw.get("stream_options"), dict)
                             and req.raw["stream_options"].get("include_usage"))
        base = self._base(req, backend, "chat.completion.chunk")

        def chunk(delta, finish=None):
            c = dict(base, choices=[{"index": 0, "delta": delta, "logprobs": None,
                                     "finish_reason": finish}])
            if include_usage:
                c["usage"] = None
            return c

        # the first event decides between an HTTP error and a 200 stream
        try:
            ev = self._next(gen)
        except Cancelled:
            gen.close()
            return self._cancelled(rec, "cancelled (client disconnected)")
        except BackendError as e:
            gen.close()
            return self._backend_error(e)
        except Exception as e:                                 # noqa: BLE001
            gen.close()
            traceback.print_exc()
            return self._error(500, f"Internal error in the '{req.model}' backend: {e}",
                               "server_error", code="backend_error")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self._chunked = self.request_version == "HTTP/1.1"
        if self._chunked:
            self.send_header("Transfer-Encoding", "chunked")
        else:
            self.send_header("Connection", "close")
            self.close_connection = True
        self.end_headers()
        fin = None
        try:
            self._sse(chunk({"role": "assistant", "content": ""}))
            while True:
                if isinstance(ev, Finish):
                    fin = ev
                    break
                if ev.text:
                    if "ttft_ms" not in rec:
                        rec["ttft_ms"] = (time.monotonic() - t0) * 1000.0
                    self._sse(chunk({"content": ev.text}))
                ev = self._next(gen)
        except (BrokenPipeError, ConnectionResetError):
            cancel.cancel()
            return self._cancelled(rec, "cancelled (client disconnected)")
        except Cancelled:
            return self._cancelled(rec, "cancelled (client disconnected)")
        except Exception as e:                                 # noqa: BLE001
            traceback.print_exc()
            rec["result"] = f"error mid-stream: {e}"
            self._sse({"error": {"message": f"Internal error in the '{req.model}' backend: {e}",
                                 "type": "server_error", "param": None, "code": "backend_error"}})
            self._write(b"data: [DONE]\n\n")
            if self._chunked:
                self.wfile.write(b"0\r\n\r\n")
            return
        finally:
            gen.close()
        rec["fin"] = fin
        last = chunk({}, fin.reason)
        if fin.info:
            last["kv260"] = self._info(fin)
        self._sse(last)
        if include_usage:
            self._sse(dict(base, choices=[], usage=self._usage(fin)))
        self._write(b"data: [DONE]\n\n")
        if self._chunked:
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()

    def _log_request(self, rec: Dict[str, Any], t0: float) -> None:
        total = (time.monotonic() - t0) * 1000.0
        fin: Optional[Finish] = rec.get("fin")
        parts = [f"{self.client_address[0]} POST {urlsplit(self.path).path}",
                 str(getattr(self, "_status", 200)), f"model={rec['model']}", f"stream={rec['stream']}"]
        if fin is not None:
            parts += [f"prompt={fin.prompt_tokens}", f"completion={fin.completion_tokens}"]
            parts += [f"{k}={v}" for k, v in (fin.info.get("log") or {}).items()]
        if "queue_ms" in rec:
            parts.append(f"queue={rec['queue_ms']:.0f}ms")
        if "load_ms" in rec:
            parts.append(f"load={rec['load_ms']:.0f}ms")
        if "ttft_ms" in rec:
            parts.append(f"ttft={rec['ttft_ms']:.0f}ms")
            if fin is not None and fin.completion_tokens > 1 and total > rec["ttft_ms"]:
                rate = (fin.completion_tokens - 1) / ((total - rec["ttft_ms"]) / 1000.0)
                parts.append(f"tok/s={rate:.1f}")
        parts.append(f"total={total:.0f}ms")
        if fin is not None:
            parts.append(f"finish={fin.reason}")
        if rec.get("result"):
            parts.append(f"[{rec['result']}]")
        log(" ".join(parts))

    def _cancelled(self, rec: Dict[str, Any], what: str) -> None:
        rec["result"] = what
        self._status = 499                    # logged like nginx: client closed the request
        self.close_connection = True


# ── main ─────────────────────────────────────────────────────────────────────

def llm_engine(args):
    """libsmollm2.so, or with --llm-fake a stand-in from tests/fake_llm.py
    (development without the FPGA: 'float' = the float reference model,
    exact and slow, needs numpy + the safetensors weights; 'scripted' =
    a fixed reply)."""
    if args.llm_fake:
        sys.path.insert(0, os.path.join(HERE, "tests"))
        import fake_llm
        if args.llm_fake == "float":
            return fake_llm.FloatModelEngine(context_size=args.llm_context)
        from smollm2_tokenizer import Tokenizer
        reply = Tokenizer(args.llm_tokenizer).encode(
            "This is the scripted fake of libsmollm2.so (kv260_chat_server.py --llm-fake scripted).")
        return fake_llm.ScriptedEngine(reply, context_size=args.llm_context, delay=args.echo_delay)
    from smollm2_backend import LibLlmEngine
    return LibLlmEngine(args.llm_lib, args.llm_weights)


def build_backends(args) -> Dict[str, Backend]:
    out: Dict[str, Backend] = {}
    for name in args.backend or ["bert-squad"]:
        if name == "echo":
            b: Backend = EchoBackend(delay=args.echo_delay)
        elif name == "bert-squad":
            from bert_squad_backend import BertSquadBackend, LibBertEngine
            b = BertSquadBackend(LibBertEngine(args.bert_lib, args.bert_weights), args.vocab,
                                 max_windows=args.max_windows, doc_stride=args.doc_stride)
        elif name in ("smollm2", "smollm2-135m-instruct"):
            from sampler import SamplerParams
            from smollm2_backend import Smollm2Backend
            b = Smollm2Backend(
                llm_engine(args), args.llm_tokenizer, sampler_lib=args.llm_sampler_lib,
                defaults=SamplerParams(temperature=args.llm_temperature, top_p=args.llm_top_p,
                                       top_k=args.llm_top_k,
                                       repetition_penalty=args.llm_repetition_penalty,
                                       dry_multiplier=args.llm_dry_multiplier,
                                       dry_base=args.llm_dry_base,
                                       dry_allowed_length=args.llm_dry_allowed_length),
                dry_penalty_last_n=args.llm_dry_penalty_last_n,
                dry_sequence_breakers=tuple(json.loads(args.llm_dry_sequence_breakers)),
                loop_guard=not args.llm_no_loop_guard,
                context_size=args.llm_context, reserve=args.llm_reserve,
                repeat_last_n=args.llm_repeat_last_n, prefill_chunk=args.llm_prefill_chunk,
                cma_mb=args.llm_cma_mb)
        else:
            raise SystemExit(f"unknown backend '{name}' (known: bert-squad, smollm2, echo)")
        out[b.model_id] = b
    return out


def _first_existing(*paths: str) -> str:
    return next((p for p in paths if os.path.exists(p)), paths[0])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--api-key", default=os.environ.get("KV260_CHAT_API_KEY"),
                    help="require 'Authorization: Bearer KEY' on /v1/* (env KV260_CHAT_API_KEY)")
    ap.add_argument("--api-key-file", default=None, help="read the API key from a file")
    ap.add_argument("--backend", action="append",
                    choices=("bert-squad", "smollm2", "smollm2-135m-instruct", "echo"),
                    help="backend(s) to serve (default: bert-squad); the first is the default model")
    ap.add_argument("--resident", choices=("auto", "one", "all"), default="auto",
                    help="which FPGA models stay loaded (see above; default auto)")
    ap.add_argument("--cma-margin-mb", type=float, default=32.0,
                    help="--resident auto: CmaFree to keep beyond a model's own need")
    ap.add_argument("--bert-lib", default=os.path.join(HERE, "lib", "libbert_squad.so"))
    ap.add_argument("--bert-weights", default=None,
                    help="weights directory (holds weights/*.dat); default: the one the library was built for")
    ap.add_argument("--vocab", default=next((p for p in (
        os.path.join(HERE, "vocab.txt"),
        os.path.join(HERE, "..", "bert_squad", "assets", "vocab.txt")) if os.path.exists(p)),
        os.path.join(HERE, "vocab.txt")))
    ap.add_argument("--max-windows", type=int, default=8,
                    help="256-token windows per question at most (latency cap, ~1 s each)")
    ap.add_argument("--doc-stride", type=int, default=128)
    g = ap.add_argument_group("smollm2 (generative chat)")
    g.add_argument("--llm-lib", default=os.path.join(HERE, "lib", "libsmollm2.so"))
    g.add_argument("--llm-weights", default=None,
                   help="weights directory for llm_open(); default: the one the library was built for")
    g.add_argument("--llm-tokenizer", default=_first_existing(
        os.path.join(HERE, "smollm2", "tokenizer.json"),
        os.path.join(HERE, "assets", "smollm2-135m-instruct", "tokenizer.json")))
    g.add_argument("--llm-sampler-lib", default=None,
                   help="libsampler.so (default lib/libsampler.so; pure Python if it is missing)")
    g.add_argument("--llm-context", type=int, default=1024,
                   help="context size for trimming before the library is loaded (it reports its own)")
    g.add_argument("--llm-reserve", type=int, default=256,
                   help="answer tokens to keep free when trimming the history (max_tokens if smaller)")
    g.add_argument("--llm-temperature", type=float, default=0.2, help="default temperature (0: greedy)")
    g.add_argument("--llm-top-p", type=float, default=0.9, help="default top_p")
    g.add_argument("--llm-top-k", type=int, default=50, help="default top_k (0: off)")
    g.add_argument("--llm-repetition-penalty", type=float, default=1.1, help="default (1: off)")
    g.add_argument("--llm-dry-multiplier", type=float, default=0.8,
                   help="default DRY multiplier (0: off) — penalises extending an already repeated run")
    g.add_argument("--llm-dry-base", type=float, default=1.75, help="default DRY base (growth per matched token)")
    g.add_argument("--llm-dry-allowed-length", type=int, default=2,
                   help="default DRY allowed length (repeats shorter than this are free)")
    g.add_argument("--llm-dry-penalty-last-n", type=int, default=-1,
                   help="DRY window in tokens of prompt + answer (-1: the whole context, 0: off)")
    g.add_argument("--llm-dry-sequence-breakers", default=json.dumps([":", "\"", "*"]),
                   help="JSON list of strings; a token containing one cuts a DRY match")
    g.add_argument("--llm-no-loop-guard", action="store_true",
                   help="do not stop answers that end in a verbatim repeated block")
    g.add_argument("--llm-repeat-last-n", type=int, default=64,
                   help="penalty window in tokens (0: off, -1: the whole context)")
    g.add_argument("--llm-prefill-chunk", type=int, default=0,
                   help="split prefills into calls of at most N tokens (cancellable between them); 0: one call")
    g.add_argument("--llm-cma-mb", type=float, default=360.0,
                   help="CMA the loaded model holds (MB), for --resident auto")
    g.add_argument("--llm-fake", choices=("float", "scripted"), default=None,
                   help="development without the FPGA: tests/fake_llm.py instead of libsmollm2.so")
    ap.add_argument("--queue-timeout", type=float, default=120.0,
                    help="seconds a request may wait for the FPGA before a 503")
    ap.add_argument("--max-queue", type=int, default=16, help="requests waiting at most")
    ap.add_argument("--echo-delay", type=float, default=0.0, help="seconds per echo word")
    args = ap.parse_args(argv)
    if args.api_key_file:
        with open(args.api_key_file) as f:
            args.api_key = f.read().strip()

    backends = build_backends(args)
    srv = ChatServer((args.host, args.port), backends, api_key=args.api_key,
                     queue_timeout=args.queue_timeout, max_queue=args.max_queue,
                     resident=args.resident, cma_margin_mb=args.cma_margin_mb)
    try:
        for mid, b in backends.items():
            t0 = time.monotonic()
            b.load_host()
            if time.monotonic() - t0 > 0.05:
                log(f"'{mid}': host side ready in {time.monotonic() - t0:.1f} s")
        srv.load_startup()
    except Exception as e:                                     # noqa: BLE001
        traceback.print_exc()
        log(f"error: loading failed: {e}")
        srv.close_models()
        srv.server_close()
        return 1

    def stop(signum, _frame):
        log(f"signal {signum}: shutting down")
        threading.Thread(target=srv.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    log(f"kv260-chat {VERSION} listening on http://{args.host}:{args.port}/v1  "
        f"models: {', '.join(m + ('' if srv.loaded[m] else ' (loads on request)') for m in backends)}"
        f"  resident: {args.resident}  api key: {'required' if args.api_key else 'none'}")
    try:
        srv.serve_forever(poll_interval=0.5)
    finally:
        srv.server_close()
        # let a running request finish before the model goes away
        if srv.fpga.acquire(60.0) != "ok":
            log("warning: a request is still running; closing anyway")
        srv.close_models()
        log("stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
