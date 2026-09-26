"""Protocol layer of kv260_chat_server.py against a fake backend: the
OpenAI chat.completion / chat.completion.chunk schema field by field,
errors, the FPGA request queue, and client disconnects."""

import json
import socket
import threading
import time
import unittest

from _util import LOG, RunningServer, http_request_bytes, server_mod, sse_events, wait_until

from chat_backend import (Backend, BackendError, Cancelled, CancelToken, Delta, Finish,
                          StopStream, apply_stop)

FINISH_REASONS = {"stop", "length", "content_filter", "tool_calls", "function_call"}
COMPLETION_KEYS = {"id", "object", "created", "model", "system_fingerprint", "choices", "usage",
                   "service_tier", "kv260"}
CHUNK_KEYS = {"id", "object", "created", "model", "system_fingerprint", "choices", "usage",
              "service_tier", "kv260"}


class FakeBackend(Backend):
    """Emits `words` one Delta each after `first_delay` seconds of
    cancellable "FPGA work"; records concurrency and cancellations."""

    model_id = "fake"
    fingerprint = "fp-fake"

    def __init__(self, words=("The", " Denver", " Broncos"), delay=0.0, first_delay=0.0):
        self.words, self.delay, self.first_delay = list(words), delay, first_delay
        self.lock = threading.Lock()
        self.active = self.max_active = 0
        self.runs, self.cancelled, self.prepared = [], 0, 0

    def prepare(self, req):
        self.prepared += 1
        if req.messages[-1]["content"] == "bad":
            raise BackendError("This question is not acceptable.", "messages")
        if req.messages[-1]["content"] == "crash":
            raise RuntimeError("boom")
        return req

    def generate(self, req, cancel):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        t0 = time.monotonic()
        try:
            end = t0 + self.first_delay
            while time.monotonic() < end:
                cancel.check()
                time.sleep(0.01)
            n = 0
            for w in self.words:
                if req.max_tokens is not None and n >= req.max_tokens:
                    yield Finish("length", 7, n, {"windows": 1})
                    return
                cancel.check()
                if self.delay:
                    time.sleep(self.delay)
                yield Delta(w)
                n += 1
            yield Finish("stop", 7, n, {"windows": 1, "score": 1.5, "log": {"windows": "1/1"}})
        except Cancelled:
            with self.lock:
                self.cancelled += 1
            raise
        finally:
            with self.lock:
                self.active -= 1
                self.runs.append((t0, time.monotonic()))


def chat_body(content="Who won?", **kw):
    b = {"model": "fake", "messages": [{"role": "system", "content": "doc"},
                                       {"role": "user", "content": content}]}
    b.update(kw)
    return b


class SchemaMixin:
    def assert_completion(self, obj, model="fake"):
        self.assertTrue(set(obj) <= COMPLETION_KEYS, set(obj) - COMPLETION_KEYS)
        self.assertIsInstance(obj["id"], str)
        self.assertTrue(obj["id"].startswith("chatcmpl-"))
        self.assertEqual(obj["object"], "chat.completion")
        self.assertIsInstance(obj["created"], int)
        self.assertLess(abs(obj["created"] - time.time()), 60)
        self.assertEqual(obj["model"], model)
        self.assertIn(type(obj["system_fingerprint"]), (str, type(None)))
        self.assertIsInstance(obj["choices"], list)
        self.assertEqual(len(obj["choices"]), 1)
        ch = obj["choices"][0]
        self.assertEqual(set(ch), {"index", "message", "logprobs", "finish_reason"})
        self.assertEqual(ch["index"], 0)
        self.assertIsNone(ch["logprobs"])
        self.assertIn(ch["finish_reason"], FINISH_REASONS)
        msg = ch["message"]
        self.assertEqual(msg["role"], "assistant")
        self.assertIsInstance(msg["content"], str)
        self.assertIn("refusal", msg)
        self.assertIsNone(msg["refusal"])
        u = obj["usage"]
        for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
            self.assertIsInstance(u[k], int)
            self.assertGreaterEqual(u[k], 0)
        self.assertEqual(u["total_tokens"], u["prompt_tokens"] + u["completion_tokens"])

    def assert_stream(self, events, model="fake", include_usage=False):
        """Returns (content, finish_reason, usage or None, kv260 or None)."""
        self.assertEqual(events[-1], "[DONE]")
        self.assertEqual(events.count("[DONE]"), 1)
        chunks = events[:-1]
        self.assertGreaterEqual(len(chunks), 2)
        ids = {c["id"] for c in chunks}
        self.assertEqual(len(ids), 1)
        self.assertTrue(next(iter(ids)).startswith("chatcmpl-"))
        self.assertEqual(len({c["created"] for c in chunks}), 1)
        content, finish, usage, extra = "", None, None, None
        body = chunks[:-1] if include_usage else chunks
        for i, c in enumerate(chunks):
            self.assertTrue(set(c) <= CHUNK_KEYS, set(c) - CHUNK_KEYS)
            self.assertEqual(c["object"], "chat.completion.chunk")
            self.assertEqual(c["model"], model)
            self.assertIsInstance(c["created"], int)
            self.assertIn(type(c["system_fingerprint"]), (str, type(None)))
            if include_usage:
                self.assertIn("usage", c)
            else:
                self.assertNotIn("usage", c)
        for i, c in enumerate(body):
            self.assertEqual(len(c["choices"]), 1)
            ch = c["choices"][0]
            self.assertEqual(set(ch), {"index", "delta", "logprobs", "finish_reason"})
            self.assertEqual(ch["index"], 0)
            self.assertIsNone(ch["logprobs"])
            self.assertIsInstance(ch["delta"], dict)
            if include_usage:
                self.assertIsNone(c["usage"])
            if i == 0:
                self.assertEqual(ch["delta"].get("role"), "assistant")
            else:
                self.assertNotIn("role", ch["delta"])
            if i < len(body) - 1:
                self.assertIsNone(ch["finish_reason"])
            else:
                self.assertIn(ch["finish_reason"], FINISH_REASONS)
                self.assertEqual(ch["delta"], {})
                finish = ch["finish_reason"]
                extra = c.get("kv260")
            content += ch["delta"].get("content") or ""
        if include_usage:
            last = chunks[-1]
            self.assertEqual(last["choices"], [])
            usage = last["usage"]
            self.assertEqual(usage["total_tokens"], usage["prompt_tokens"] + usage["completion_tokens"])
        return content, finish, usage, extra


class TestProtocol(SchemaMixin, unittest.TestCase):

    def setUp(self):
        LOG.clear()
        self.fb = FakeBackend()
        self.s = RunningServer({"fake": self.fb, "echo": server_mod.EchoBackend()})

    def tearDown(self):
        self.s.close()

    def test_models(self):
        st, h, body = self.s.request("GET", "/v1/models")
        self.assertEqual(st, 200)
        self.assertTrue(h["content-type"].startswith("application/json"))
        obj = json.loads(body)
        self.assertEqual(obj["object"], "list")
        self.assertEqual([m["id"] for m in obj["data"]], ["fake", "echo"])
        for m in obj["data"]:
            self.assertEqual(set(m), {"id", "object", "created", "owned_by"})
            self.assertEqual(m["object"], "model")
            self.assertIsInstance(m["created"], int)
        st, _, body = self.s.request("GET", "/v1/models/echo")
        self.assertEqual((st, json.loads(body)["id"]), (200, "echo"))
        st, _, body = self.s.request("GET", "/v1/models/nope")
        self.assertEqual(st, 404)
        self.assertEqual(json.loads(body)["error"]["code"], "model_not_found")
        st, _, _ = self.s.request("GET", "/models")          # without the /v1 prefix
        self.assertEqual(st, 200)

    def test_health(self):
        st, _, body = self.s.request("GET", "/health")
        obj = json.loads(body)
        self.assertEqual((st, obj["status"]), (200, "ok"))
        self.assertEqual([m["id"] for m in obj["models"]], ["fake", "echo"])
        self.assertFalse(obj["busy"])

    def test_non_streaming(self):
        st, h, body = self.s.post(chat_body())
        self.assertEqual(st, 200, body)
        self.assertTrue(h["content-type"].startswith("application/json"))
        obj = json.loads(body)
        self.assert_completion(obj)
        self.assertEqual(obj["choices"][0]["message"]["content"], "The Denver Broncos")
        self.assertEqual(obj["choices"][0]["finish_reason"], "stop")
        self.assertEqual(obj["usage"], {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10})
        self.assertEqual(obj["system_fingerprint"], "fp-fake")
        self.assertEqual(obj["kv260"], {"windows": 1, "score": 1.5})    # "log" stays server-side

    def test_streaming(self):
        st, h, body = self.s.post(chat_body(stream=True))
        self.assertEqual(st, 200, body)
        self.assertTrue(h["content-type"].startswith("text/event-stream"))
        self.assertEqual(h.get("transfer-encoding"), "chunked")
        content, finish, usage, extra = self.assert_stream(sse_events(body))
        self.assertEqual((content, finish), ("The Denver Broncos", "stop"))
        self.assertEqual(extra, {"windows": 1, "score": 1.5})

    def test_streaming_include_usage(self):
        st, _, body = self.s.post(chat_body(stream=True, stream_options={"include_usage": True}))
        content, finish, usage, _ = self.assert_stream(sse_events(body), include_usage=True)
        self.assertEqual(usage, {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10})

    def test_stream_equals_non_stream(self):
        for body in (chat_body(), chat_body(max_tokens=2), {
                "model": "echo", "messages": [{"role": "user", "content": "one two three four"}],
                "stop": [" three"]}):
            _, _, a = self.s.post(dict(body, stream=False))
            _, _, b = self.s.post(dict(body, stream=True))
            a = json.loads(a)
            content, finish, _, _ = self.assert_stream(sse_events(b), model=body["model"])
            self.assertEqual(content, a["choices"][0]["message"]["content"])
            self.assertEqual(finish, a["choices"][0]["finish_reason"])

    def test_max_tokens_and_length(self):
        for key in ("max_tokens", "max_completion_tokens"):
            st, _, body = self.s.post(chat_body(**{key: 2}))
            obj = json.loads(body)
            self.assert_completion(obj)
            self.assertEqual(obj["choices"][0]["message"]["content"], "The Denver")
            self.assertEqual(obj["choices"][0]["finish_reason"], "length")
            self.assertEqual(obj["usage"]["completion_tokens"], 2)

    def test_stop_echo(self):
        st, _, body = self.s.post({"model": "echo", "stop": "c",
                                   "messages": [{"role": "user", "content": "a b c d"}]})
        obj = json.loads(body)
        self.assert_completion(obj, model="echo")
        self.assertEqual(obj["choices"][0]["message"]["content"], "a b ")
        self.assertEqual(obj["choices"][0]["finish_reason"], "stop")

    def test_ignores_unknown_fields_and_accepts_parts(self):
        body = chat_body(temperature=0.2, top_p=0.9, seed=42, n=1, user="me", tools=[],
                         response_format={"type": "text"}, logit_bias={}, frequency_penalty=0,
                         presence_penalty=0, some_future_field={"x": 1})
        body["messages"][1]["content"] = [{"type": "text", "text": "Who"}, {"type": "text", "text": "won?"}]
        st, _, out = self.s.post(body)
        self.assertEqual(st, 200, out)
        self.assert_completion(json.loads(out))

    def test_model_default(self):
        st, _, out = self.s.post({"messages": [{"role": "user", "content": "x"}]})
        self.assertEqual(st, 200)
        self.assertEqual(json.loads(out)["model"], "fake")

    def _err(self, st, out, status, param=None, code=None):
        self.assertEqual(st, status, out)
        obj = json.loads(out)
        self.assertEqual(set(obj), {"error"})
        e = obj["error"]
        self.assertEqual(set(e), {"message", "type", "param", "code"})
        self.assertIsInstance(e["message"], str)
        self.assertTrue(e["message"])
        self.assertIsInstance(e["type"], str)
        self.assertEqual(e["param"], param)
        if code is not None:
            self.assertEqual(e["code"], code)
        return e

    def test_errors_400(self):
        st, _, out = self.s.post(None, raw=b"{not json")
        e = self._err(st, out, 400)
        self.assertEqual(e["type"], "invalid_request_error")
        cases = [
            ({"model": "fake"}, "messages"),
            ({"model": "fake", "messages": []}, "messages"),
            ({"model": "fake", "messages": "hi"}, "messages"),
            ({"model": "fake", "messages": [{"role": "robot", "content": "x"}]}, "messages.[0].role"),
            ({"model": "fake", "messages": [{"role": "user", "content": 5}]}, "messages.[0].content"),
            ({"model": "fake", "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "x"}}]}]}, "messages.[0].content.[0]"),
            (chat_body(temperature=3), "temperature"),
            (chat_body(temperature="hot"), "temperature"),
            (chat_body(top_p=-0.1), "top_p"),
            (chat_body(max_tokens=0), "max_tokens"),
            (chat_body(max_tokens=1.5), "max_tokens"),
            (chat_body(max_completion_tokens=-1), "max_completion_tokens"),
            (chat_body(n=2), "n"),
            (chat_body(stop=["a", "b", "c", "d", "e"]), "stop"),
            (chat_body(stop=5), "stop"),
            (chat_body(stream="yes"), "stream"),
            (chat_body(seed="x"), "seed"),
            (chat_body(model=""), "model"),
            (chat_body("bad"), "messages"),                  # BackendError from prepare()
        ]
        for body, param in cases:
            with self.subTest(param=param, body=body):
                st, _, out = self.s.post(body)
                self._err(st, out, 400, param)
        st, _, out = self.s.post(None, raw=b"[1, 2]")
        self._err(st, out, 400)

    def test_errors_404(self):
        st, _, out = self.s.post(chat_body(model="gpt-4"))
        self._err(st, out, 404, "model", "model_not_found")
        st, _, out = self.s.request("GET", "/v1/nothing")
        self._err(st, out, 404, None, "unknown_url")
        st, _, out = self.s.request("POST", "/v1/completions", {"prompt": "x"})
        self._err(st, out, 404, None, "unknown_url")

    def test_backend_crash_is_500(self):
        st, _, out = self.s.post(chat_body("crash"))
        self._err(st, out, 500, None, "backend_error")
        st, _, out = self.s.post(chat_body())                # the server keeps serving
        self.assertEqual(st, 200)

    def test_body_limit(self):
        self.s.srv.max_body = 1000
        st, _, out = self.s.post(chat_body("x" * 2000))
        self._err(st, out, 413)

    def test_keep_alive(self):
        import http.client
        c = http.client.HTTPConnection("127.0.0.1", self.s.port, timeout=10)
        for stream in (False, True, False):
            c.request("POST", "/v1/chat/completions", json.dumps(chat_body(stream=stream)),
                      {"Content-Type": "application/json"})
            r = c.getresponse()
            data = r.read()
            self.assertEqual(r.status, 200)
            self.assertIn(b"Broncos", data)
        c.close()

    def test_http10_stream(self):
        sock = self.s.raw_socket()
        req = http_request_bytes(self.s.port, chat_body(stream=True)).replace(b"HTTP/1.1", b"HTTP/1.0", 1)
        sock.sendall(req)
        data = b""
        while True:
            b = sock.recv(65536)
            if not b:
                break
            data += b
        sock.close()
        head, _, body = data.partition(b"\r\n\r\n")
        self.assertIn(b"200", head.split(b"\r\n")[0])
        self.assertNotIn(b"chunked", head.lower())
        content, finish, _, _ = self.assert_stream(sse_events(body))
        self.assertEqual(content, "The Denver Broncos")


class TestAuth(unittest.TestCase):

    def setUp(self):
        self.s = RunningServer({"fake": FakeBackend()}, api_key="sekrit")

    def tearDown(self):
        self.s.close()

    def test_api_key(self):
        st, _, out = self.s.post(chat_body())
        self.assertEqual(st, 401)
        self.assertEqual(json.loads(out)["error"]["code"], "invalid_api_key")
        st, _, _ = self.s.post(chat_body(), headers={"Authorization": "Bearer wrong"})
        self.assertEqual(st, 401)
        st, _, _ = self.s.request("GET", "/v1/models")
        self.assertEqual(st, 401)
        st, _, out = self.s.post(chat_body(), headers={"Authorization": "Bearer sekrit"})
        self.assertEqual(st, 200, out)
        st, _, _ = self.s.request("GET", "/v1/models", headers={"Authorization": "Bearer sekrit"})
        self.assertEqual(st, 200)
        st, _, _ = self.s.request("GET", "/health")          # no key needed
        self.assertEqual(st, 200)


class TestQueue(unittest.TestCase):

    def test_one_at_a_time_fifo(self):
        fb = FakeBackend(first_delay=0.3)
        s = RunningServer({"fake": fb})
        try:
            results, order = {}, []

            def go(i):
                st, _, out = s.post(chat_body(f"q{i}"))
                results[i] = st
                order.append(i)
            ts = []
            for i in range(4):
                t = threading.Thread(target=go, args=(i,))
                t.start()
                ts.append(t)
                time.sleep(0.05)                            # arrival order 0, 1, 2, 3
            for t in ts:
                t.join()
            self.assertEqual(results, {0: 200, 1: 200, 2: 200, 3: 200})
            self.assertEqual(fb.max_active, 1)
            self.assertEqual(order, [0, 1, 2, 3])
            runs = sorted(fb.runs)
            for (a0, a1), (b0, b1) in zip(runs, runs[1:]):
                self.assertGreaterEqual(b0, a1 - 1e-3)
        finally:
            s.close()

    def test_queue_timeout_and_full(self):
        fb = FakeBackend(first_delay=1.5)
        s = RunningServer({"fake": fb}, queue_timeout=0.3, max_queue=1)
        try:
            t = threading.Thread(target=s.post, args=(chat_body(),))
            t.start()
            self.assertTrue(wait_until(lambda: s.srv.fpga.busy))
            t0 = time.monotonic()
            st, h, out = s.post(chat_body())                 # waits 0.3 s, then 503
            self.assertEqual(st, 503)
            self.assertGreaterEqual(time.monotonic() - t0, 0.25)
            e = json.loads(out)["error"]
            self.assertEqual((e["code"], e["type"]), ("server_busy", "server_error"))
            self.assertIn("retry-after", h)
            # max_queue 1: one waiter is allowed, the next is refused at once
            s.srv.queue_timeout = 5.0
            w = threading.Thread(target=s.post, args=(chat_body(),))
            w.start()
            self.assertTrue(wait_until(lambda: s.srv.fpga.waiting == 1))
            t0 = time.monotonic()
            st, _, out = s.post(chat_body())
            self.assertEqual(st, 503)
            self.assertLess(time.monotonic() - t0, 0.5)
            self.assertIn("queue is full", json.loads(out)["error"]["message"])
            t.join()
            w.join()
            self.assertEqual(fb.max_active, 1)
        finally:
            s.close()

    def test_fifo_lock_unit(self):
        lk = server_mod.FifoLock(max_waiting=2)
        self.assertEqual(lk.acquire(1.0), "ok")
        self.assertEqual(lk.acquire(0.05), "timeout")
        self.assertEqual(lk.waiting, 0)
        flag = threading.Event()
        res = {}
        t = threading.Thread(target=lambda: res.setdefault("r", lk.acquire(5.0, flag.is_set, poll=0.01)))
        t.start()
        self.assertTrue(wait_until(lambda: lk.waiting == 1))
        flag.set()
        t.join()
        self.assertEqual((res["r"], lk.waiting), ("cancelled", 0))
        lk.release()
        self.assertEqual(lk.acquire(0.1), "ok")
        lk.release()


class TestDisconnect(unittest.TestCase):

    def _drain_open(self, s, fb):
        """After a cancelled request the FPGA is free again at once."""
        self.assertTrue(wait_until(lambda: not s.srv.fpga.busy and fb.active == 0, 5.0))
        fb.first_delay, fb.delay = 0.0, 0.0
        st, _, _ = s.post(chat_body())
        self.assertEqual(st, 200)

    def test_streaming_client_goes_away(self):
        fb = FakeBackend(words=[f" w{i}" for i in range(200)], delay=0.02)
        s = RunningServer({"fake": fb})
        try:
            sock = s.raw_socket()
            sock.sendall(http_request_bytes(s.port, chat_body(stream=True)))
            got = b""
            while b"w1" not in got:
                got += sock.recv(4096)
            sock.close()
            self.assertTrue(wait_until(lambda: fb.cancelled == 1, 5.0))
            self.assertLess(len(fb.runs), 2)
            self.assertTrue(wait_until(lambda: any("stream=1" in x and "cancelled" in x for x in LOG)))
            self._drain_open(s, fb)
        finally:
            s.close()

    def test_non_streaming_client_goes_away(self):
        fb = FakeBackend(first_delay=3.0)
        s = RunningServer({"fake": fb})
        try:
            sock = s.raw_socket()
            sock.sendall(http_request_bytes(s.port, chat_body()))
            self.assertTrue(wait_until(lambda: fb.active == 1))
            t0 = time.monotonic()
            sock.close()
            self.assertTrue(wait_until(lambda: fb.cancelled == 1, 5.0))
            self.assertLess(time.monotonic() - t0, 1.0)     # not the 3 s of "work"
            self.assertTrue(wait_until(lambda: any("stream=0" in x and "[cancelled (client" in x
                                                   for x in LOG)))
            self._drain_open(s, fb)
        finally:
            s.close()

    def test_client_leaves_while_queued(self):
        fb = FakeBackend(first_delay=1.0)
        s = RunningServer({"fake": fb})
        try:
            t = threading.Thread(target=s.post, args=(chat_body(),))
            t.start()
            self.assertTrue(wait_until(lambda: fb.active == 1))
            sock = s.raw_socket()
            sock.sendall(http_request_bytes(s.port, chat_body()))
            self.assertTrue(wait_until(lambda: s.srv.fpga.waiting == 1))
            sock.close()
            self.assertTrue(wait_until(lambda: s.srv.fpga.waiting == 0, 2.0))
            t.join()
            time.sleep(0.2)
            self.assertEqual(len(fb.runs), 1)               # the queued request never ran
            self.assertEqual(fb.prepared, 2)
            self.assertTrue(any("[cancelled while queued]" in x for x in LOG))
        finally:
            s.close()


class TestHelpers(unittest.TestCase):

    def test_apply_stop(self):
        self.assertEqual(apply_stop("abc def", ["de", "c"]), ("ab", True))
        self.assertEqual(apply_stop("abc", ["x", ""]), ("abc", False))
        self.assertEqual(apply_stop("abc", []), ("abc", False))

    def test_stop_stream(self):
        ss = StopStream(["END", "\n\n"])
        out = []
        for piece in ["Hello", " E", "N", "x more", "\n", "\n", "tail"]:
            text, stopped = ss.feed(piece)
            out.append(text)
            if stopped:
                break
        self.assertEqual("".join(out), "Hello ENx more")
        self.assertTrue(stopped)
        ss = StopStream(["END"])
        self.assertEqual(ss.feed("abcEN"), ("abc", False))
        self.assertEqual(ss.flush(), "EN")

    def test_cancel_token_probe(self):
        calls = []
        tok = CancelToken(lambda: calls.append(1) or len(calls) >= 3)
        self.assertFalse(tok.cancelled())
        self.assertFalse(tok.cancelled())
        with self.assertRaises(Cancelled):
            tok.check()
        self.assertTrue(tok.cancelled())
        self.assertEqual(len(calls), 3)                     # sticky once set


if __name__ == "__main__":
    unittest.main()
