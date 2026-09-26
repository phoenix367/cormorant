"""Shared helpers for the chat server tests (stdlib only)."""

import http.client
import json
import os
import socket
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CHAT = os.path.dirname(HERE)
DEMO = os.path.dirname(CHAT)
SCRIPTS = os.path.join(DEMO, "bert_squad", "scripts")
for _p in (CHAT, SCRIPTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import kv260_chat_server as server_mod  # noqa: E402

# Per-request log lines go here instead of stderr (tests assert on them).
LOG = []
server_mod.log = LOG.append


class RunningServer:
    """A ChatServer on 127.0.0.1:<free port> in a background thread."""

    def __init__(self, backends, **kw):
        self.srv = server_mod.ChatServer(("127.0.0.1", 0), backends, **kw)
        self.port = self.srv.server_address[1]
        self.thread = threading.Thread(target=self.srv.serve_forever, kwargs={"poll_interval": 0.05},
                                       daemon=True)
        self.thread.start()

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()

    def request(self, method, path, body=None, headers=None, raw=None, timeout=30):
        """(status, headers dict (lower-case), body bytes)."""
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        h = {"Content-Type": "application/json"}
        h.update(headers or {})
        data = raw if raw is not None else (None if body is None else json.dumps(body).encode())
        c.request(method, path, body=data, headers=h)
        r = c.getresponse()
        out = r.read()
        hd = {k.lower(): v for k, v in r.getheaders()}
        c.close()
        return r.status, hd, out

    def post(self, body, **kw):
        return self.request("POST", "/v1/chat/completions", body, **kw)

    def raw_socket(self):
        return socket.create_connection(("127.0.0.1", self.port), timeout=30)


def http_request_bytes(port, body, path="/v1/chat/completions"):
    data = json.dumps(body).encode()
    return (f"POST {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n"
            f"Content-Type: application/json\r\nContent-Length: {len(data)}\r\n\r\n").encode() + data


def sse_events(payload: bytes):
    """data: payloads of an SSE body, JSON-decoded ('[DONE]' kept as a string)."""
    out = []
    for block in payload.decode("utf-8").split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data:"):
                d = line[5:].strip()
                out.append(d if d == "[DONE]" else json.loads(d))
    return out


def wait_until(pred, timeout=5.0, step=0.01):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if pred():
            return True
        time.sleep(step)
    return pred()


# ── C helpers built on the host for the tests (libsampler.so, the fake libsmollm2) ──

_BUILD = {}


def build_c(src: str, name: str):
    """Compile src into a shared library in a temporary directory (once per
    process); None when there is no C compiler."""
    import atexit
    import shutil
    import subprocess
    import tempfile
    if name in _BUILD:
        return _BUILD[name]
    cc = os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc")
    out = None
    if cc:
        if "_dir" not in _BUILD:
            d = tempfile.mkdtemp(prefix="kv260_chat_tests_")
            atexit.register(shutil.rmtree, d, True)
            _BUILD["_dir"] = d
        out = os.path.join(_BUILD["_dir"], name)
        r = subprocess.run([cc, "-O2", "-Wall", "-shared", "-fPIC", "-o", out, src, "-lm"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"{cc} {src}: {r.stderr}")
    _BUILD[name] = out
    return out


def sampler_lib():
    return build_c(os.path.join(CHAT, "src", "sampler.c"), "libsampler.so")


def fake_llm_lib():
    return build_c(os.path.join(HERE, "fake_libsmollm2.c"), "libfakellm.so")
