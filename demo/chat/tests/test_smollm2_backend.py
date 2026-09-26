"""The smollm2 backend (smollm2_backend.py) against fakes of libsmollm2.so
(doc/CHAT_PLAN.md §11): tests/fake_llm.py's ScriptedEngine (Python) and
tests/fake_libsmollm2.c through the real ctypes binding (LibLlmEngine).
Streaming schema, multi-turn prefix-cache reuse (only new tokens are
prefilled; the sink never is), stop strings split across tokens, max_tokens,
cancellation, context-full handling and trimming, UTF-8 across tokens,
sampling parameters and seeds, library errors, and lazy loading / eviction
between bert-squad and smollm2 (--resident)."""

import json
import os
import tempfile
import threading
import time
import unittest

from _util import CHAT, LOG, RunningServer, fake_llm_lib, http_request_bytes, sampler_lib, sse_events, wait_until

import chatml
from chat_backend import BackendError, Cancelled, CancelToken, ChatRequest, Delta, Finish
from fake_llm import GEN_PROMPT, IM_END, FakeLibraryError, ScriptedEngine
from sampler import SamplerParams
from smollm2_backend import DRY_BREAKERS, LibLlmEngine, LlmLibraryError, Smollm2Backend
from smollm2_tokenizer import Tokenizer
from test_protocol import SchemaMixin

TOKENIZER = os.path.join(CHAT, "assets", "smollm2-135m-instruct", "tokenizer.json")
HAVE = os.path.exists(TOKENIZER)
MID = "smollm2-135m-instruct"
_TOK = []


def tok():
    if not _TOK:
        _TOK.append(Tokenizer(TOKENIZER))
    return _TOK[0]


def backend(engine, **kw):
    kw.setdefault("defaults", SamplerParams(temperature=0.0))
    b = Smollm2Backend(engine, TOKENIZER, sampler_lib=sampler_lib() or "/nonexistent", **kw)
    b.load_host()
    b.load()
    return b


def req(messages, **kw):
    ms = [{"role": r, "content": c} for r, c in messages] if messages and isinstance(messages[0], tuple) \
        else messages
    return ChatRequest(model=MID, messages=ms, **kw)


def run(b, request, cancel=None):
    job = b.prepare(request)
    events = list(b.generate(job, cancel or CancelToken()))
    assert isinstance(events[-1], Finish) and all(isinstance(e, Delta) for e in events[:-1]), events
    return [e.text for e in events[:-1]], events[-1]


@unittest.skipUnless(HAVE, "demo/chat/assets/smollm2-135m-instruct/tokenizer.json not present")
class TestGenerate(unittest.TestCase):

    def test_basic(self):
        reply = "The capital of France is Paris."
        eng = ScriptedEngine(tok().encode(reply))
        b = backend(eng)
        r = req([("user", "What is the capital of France?")])
        pieces, fin = run(b, r)
        prompt = tok().encode(chatml.render(r.messages))
        self.assertEqual("".join(pieces), reply)
        self.assertEqual(len(pieces), len(tok().encode(reply)))            # one Delta per token
        self.assertEqual((fin.reason, fin.prompt_tokens, fin.completion_tokens),
                         ("stop", len(prompt), len(tok().encode(reply))))
        self.assertEqual(eng.prefills, [prompt[1:]])                       # the sink is never passed
        self.assertEqual(eng.decodes, tok().encode(reply))
        self.assertEqual(eng.truncates, [1])
        i = fin.info
        self.assertEqual((i["finish"], i["cached_tokens"], i["prefill_tokens"], i["decode_tokens"]),
                         ("eos", 1, len(prompt) - 1, len(tok().encode(reply))))
        for k in ("prefill_ms", "ttft_ms", "decode_tok_s", "seed", "sampler", "log"):
            self.assertIn(k, i)
        self.assertEqual(eng.position(), 1 + len(prompt) - 1 + len(tok().encode(reply)))

    def test_multi_turn_prefix_reuse(self):
        t = tok()
        a1, a2 = "Hello! How can I help?", "Pack warm clothes and water."
        eng = ScriptedEngine(t.encode(a1))
        b = backend(eng)
        m1 = [{"role": "user", "content": "Hi"}]
        run(b, req(m1))
        p1 = t.encode(chatml.render(m1))
        eng.reply = t.encode(a2)
        m2 = m1 + [{"role": "assistant", "content": a1}, {"role": "user", "content": "What should I pack?"}]
        pieces, fin = run(b, req(m2))
        p2 = t.encode(chatml.render(m2))
        self.assertEqual("".join(pieces), a2)
        cached = len(p1) + len(t.encode(a1))                                # sink + prompt + answer fed
        self.assertEqual(p2[:cached], p1 + t.encode(a1))
        self.assertEqual(eng.prefills[1], p2[cached:])                      # only the new tokens
        self.assertEqual(eng.truncates[-1], cached)
        self.assertEqual(fin.info["cached_tokens"], cached)
        self.assertEqual(fin.info["prefill_tokens"], len(p2) - cached)
        # the same request again: everything cached, the last token re-run for its logits
        pieces, fin = run(b, req(m2))
        self.assertEqual("".join(pieces), a2)
        self.assertEqual(eng.prefills[-1], p2[-1:])
        # an edited history: truncated to the common prefix
        m3 = [{"role": "user", "content": "Hi there"}]
        run(b, req(m3))
        p3 = t.encode(chatml.render(m3))
        common = next(i for i in range(len(p3)) if p3[i] != p2[i])
        self.assertEqual(eng.truncates[-1], common)
        self.assertEqual(eng.prefills[-1], p3[common:])
        self.assertEqual(b.health()["cached_tokens"], eng.position())

    def test_stop_string_across_tokens(self):
        t = tok()
        reply = "Hello world. The END of the story."
        ids = t.encode(reply)
        eng = ScriptedEngine(ids)
        b = backend(eng)
        stop = "ld. The EN"                                               # spans several tokens
        self.assertGreater(len(t.encode(stop)), 2)
        pieces, fin = run(b, req([("user", "x")], stop=[stop, "never"]))
        self.assertEqual("".join(pieces), "Hello wor")
        for p in pieces:
            self.assertNotIn("ld", p)                                     # held back, never sent
        self.assertEqual((fin.reason, fin.info["finish"]), ("stop", "stop_string"))
        n_through = next(k for k in range(1, len(ids) + 1) if stop in t.decode(ids[:k]))
        self.assertEqual(fin.completion_tokens, n_through)
        # a partial match at the very end is released at EOS
        eng.reply = t.encode("abc EN")
        pieces, fin = run(b, req([("user", "y")], stop=["END"]))
        self.assertEqual(("".join(pieces), fin.info["finish"]), ("abc EN", "eos"))

    def test_max_tokens(self):
        t = tok()
        ids = t.encode("one two three four five six")
        eng = ScriptedEngine(ids)
        b = backend(eng)
        pieces, fin = run(b, req([("user", "count")], max_tokens=3))
        self.assertEqual("".join(pieces), t.decode(ids[:3]))
        self.assertEqual((fin.reason, fin.completion_tokens, fin.info["finish"]), ("length", 3, "max_tokens"))
        self.assertEqual(len(eng.decodes), 2)                             # the 3rd token is not fed

    def test_context_full(self):
        t = tok()
        eng = ScriptedEngine(t.encode("word " * 200), context_size=64)
        b = backend(eng, reserve=16)
        self.assertEqual(b.ctx, 64)
        r = req([("user", "go")])
        prompt = t.encode(chatml.render(r.messages))
        pieces, fin = run(b, r)
        self.assertEqual((fin.reason, fin.info["finish"]), ("length", "context_full"))
        self.assertEqual(eng.position(), 64)
        self.assertEqual(fin.completion_tokens, 64 - len(prompt) + 1)
        # a message that cannot fit
        with self.assertRaises(BackendError) as cm:
            b.prepare(req([("user", "long " * 100)]))
        self.assertEqual((cm.exception.status, cm.exception.code), (400, "context_length_exceeded"))

    def test_trimming(self):
        t = tok()
        eng = ScriptedEngine(t.encode("ok"), context_size=160)
        b = backend(eng, reserve=64)
        m = [{"role": "system", "content": "Be brief."}]
        for i in range(6):
            m += [{"role": "user", "content": f"question {i} " + "blah " * 8},
                  {"role": "assistant", "content": f"answer {i} " + "blah " * 8}]
        m.append({"role": "user", "content": "last"})
        job = b.prepare(req(m))
        self.assertGreater(job.dropped, 0)
        self.assertLessEqual(len(job.ids), 160 - 64)
        pieces, fin = run(b, req(m))
        self.assertEqual(fin.info["trimmed_messages"], job.dropped)
        self.assertEqual("".join(pieces), "ok")
        # max_tokens smaller than the reserve leaves more history
        self.assertLess(b.prepare(req(m, max_tokens=4)).dropped, job.dropped)

    def test_utf8_across_tokens(self):
        t = tok()
        reply = "Emoji 😀👍🏽 and 中文 — naïve"
        eng = ScriptedEngine(t.encode(reply))
        b = backend(eng)
        pieces, fin = run(b, req([("user", "x")]))
        self.assertEqual("".join(pieces), reply)
        self.assertGreater(len(t.encode(reply)), len(pieces))              # some tokens held back
        for p in pieces:
            self.assertNotIn("�", p)

    def test_cancel_mid_generation(self):
        t = tok()
        eng = ScriptedEngine(t.encode("word " * 100), delay=0.002)
        b = backend(eng)
        cancel = CancelToken()
        job = b.prepare(req([("user", "go")]))
        got = []
        with self.assertRaises(Cancelled):
            for ev in b.generate(job, cancel):
                got.append(ev)
                if len(got) == 5:
                    cancel.cancel()
        self.assertEqual(len(got), 5)
        self.assertEqual(len(eng.decodes), 4)                              # stopped before the 5th decode
        self.assertEqual(eng.position(), 1 + len(job.ids) - 1 + 4)
        # the cache stays consistent: the next request reuses it
        eng.reply = t.encode("fine")
        pieces, fin = run(b, req([("user", "go")]))
        self.assertEqual("".join(pieces), "fine")
        self.assertEqual(fin.info["cached_tokens"], len(job.ids) - 1)

    def test_library_error_invalidates_cache(self):
        t = tok()
        fail = {"on": True}
        eng = ScriptedEngine(t.encode("a b c d e"),
                             fail_on=lambda seq, toks: fail["on"] and len(toks) == 1 and len(seq) > 30)
        b = backend(eng)
        job = b.prepare(req([("user", "q")]))
        with self.assertRaises(FakeLibraryError):
            list(b.generate(job, CancelToken()))
        self.assertIsNone(b._cached)
        fail["on"] = False
        pieces, fin = run(b, req([("user", "q")]))
        self.assertEqual("".join(pieces), "a b c d e")
        self.assertEqual((eng.truncates[-1], fin.info["cached_tokens"]), (1, 1))   # a full prefill

    def test_sampling_parameters_and_seed(self):
        t = tok()
        flat = [0.0] * 49152
        for i in range(1000, 1400):
            flat[i] = 1.0                                                 # 400 equally likely tokens
        eng = ScriptedEngine(logits_fn=lambda seq: flat)
        b = backend(eng)
        base = dict(max_tokens=8, temperature=1.0, top_p=1.0, raw={"top_k": 0})

        def gen(**kw):
            a = dict(base)
            a.update(kw)
            pieces, fin = run(b, req([("user", "x")], **a))
            return "".join(pieces), fin
        a, fa = gen(seed=7)
        b_, _ = gen(seed=7)
        c, _ = gen(seed=8)
        self.assertEqual(a, b_)
        self.assertNotEqual(a, c)
        self.assertEqual(fa.info["seed"], 7)
        txt, fr = gen()                                                  # a random seed, reported
        self.assertEqual(gen(seed=fr.info["seed"])[0], txt)
        # greedy: the first of the tied maxima every time
        g, _ = gen(temperature=0.0)
        self.assertEqual(g, t.decode([1000] * 8))
        # top_k = 1 is greedy as well; the request's extras reach the sampler
        _, fk = gen(raw={"top_k": 1, "repetition_penalty": 1.5, "repeat_last_n": 16})
        self.assertEqual((fk.info["sampler"]["top_k"], fk.info["sampler"]["repetition_penalty"],
                          fk.info["sampler"]["repeat_last_n"]), (1, 1.5, 16))
        # a repetition penalty spreads the greedy choice over the tied tokens
        gen(temperature=0.0, raw={"repetition_penalty": 2.0})
        self.assertEqual(len(set(eng.decodes[-7:])), 7)
        for bad in ({"top_k": -1}, {"top_k": 1.5}, {"repetition_penalty": "x"},
                    {"repetition_penalty": 0}, {"presence_penalty": 3}, {"repeat_last_n": -2}):
            with self.assertRaises(BackendError, msg=bad):
                b.prepare(req([("user", "x")], raw=bad))

    def test_defaults_apply(self):
        eng = ScriptedEngine(tok().encode("x"))
        b = backend(eng, defaults=SamplerParams(temperature=0.2, top_p=0.9, top_k=50))
        p = b.sampler_params(req([("user", "x")]))
        self.assertEqual((p.temperature, p.top_p, p.top_k, p.repetition_penalty), (0.2, 0.9, 50, 1.0))
        p = b.sampler_params(req([("user", "x")], temperature=0.0, top_p=0.5))
        self.assertEqual((p.temperature, p.top_p), (0.0, 0.5))

    def test_developer_role_is_system(self):
        eng = ScriptedEngine(tok().encode("x"))
        b = backend(eng)
        job = b.prepare(req([("developer", "Be terse."), ("user", "hi")]))
        self.assertEqual(job.ids, tok().encode(chatml.render([{"role": "system", "content": "Be terse."},
                                                              {"role": "user", "content": "hi"}])))


@unittest.skipUnless(HAVE, "demo/chat/assets/smollm2-135m-instruct/tokenizer.json not present")
class TestDry(unittest.TestCase):
    """DRY in the backend: request fields, breaker tokens from the tokenizer,
    the default, per-request breakers, and a scripted repetition loop."""

    CYCLE_TEXT = " the old man lives in a big house"

    def cycle_engine(self, margin=4.0):
        cyc = tok().encode(self.CYCLE_TEXT)
        V = tok().vocab_size

        def logits(seq):
            vals = [0.0] * V
            last = seq[-1]
            nxt = cyc[(cyc.index(last) + 1) % len(cyc)] if last in cyc else cyc[0]
            vals[nxt] = margin                               # never <|im_end|>: loops until max_tokens
            return vals
        return ScriptedEngine(logits_fn=logits), cyc

    def test_request_fields(self):
        b = backend(ScriptedEngine(tok().encode("ok")))
        good = b.prepare(req([("user", "hi")], raw={"dry_multiplier": 1.5, "dry_base": 2.0,
                                                    "dry_allowed_length": 3, "dry_penalty_last_n": 128,
                                                    "dry_sequence_breakers": ["\n", "."]}))
        self.assertEqual((good.params.dry_multiplier, good.params.dry_base, good.params.dry_allowed_length),
                         (1.5, 2.0, 3))
        self.assertEqual((good.dry_last_n, good.breakers), (128, ("\n", ".")))
        for bad in ({"dry_multiplier": -1}, {"dry_multiplier": "a"}, {"dry_base": 0.5},
                    {"dry_allowed_length": 0}, {"dry_allowed_length": 1.5}, {"dry_penalty_last_n": -2},
                    {"dry_sequence_breakers": "x"}, {"dry_sequence_breakers": [""]},
                    {"dry_sequence_breakers": ["a"] * 33}, {"dry_sequence_breakers": [1]}):
            with self.assertRaises(BackendError, msg=bad):
                b.prepare(req([("user", "hi")], raw=bad))

    def test_breaker_ids(self):
        b = backend(ScriptedEngine(tok().encode("ok")))
        t = tok()
        ids = set(b.breaker_ids(DRY_BREAKERS))
        for text in ("\n", "\n\n", ":", "*", "**", '"', ".\n", "):"):
            for i in t.encode(text, special=False):
                if any(x.encode() in t.token_bytes(i) for x in DRY_BREAKERS):
                    self.assertIn(i, ids, (text, i, t.token_bytes(i)))
        self.assertTrue(t.special_ids <= ids)
        for text in (" cat", " the", "Paris", "."):
            for i in t.encode(text, special=False):
                self.assertNotIn(i, ids, (text, i))
        self.assertEqual(ids, set(i for i in range(t.vocab_size)
                                  if t.is_special(i) or any(x.encode() in t.token_bytes(i)
                                                            for x in DRY_BREAKERS)))
        self.assertIn(t.encode("\n", special=False)[0], ids)

    def test_dry_breaks_the_loop(self):
        eng, cyc = self.cycle_engine()
        off = backend(eng, defaults=SamplerParams(temperature=0.0))
        text_off, fin_off = run(off, req([("user", "tell me")], max_tokens=60))
        self.assertEqual(fin_off.reason, "length")
        self.assertGreaterEqual("".join(text_off).count(self.CYCLE_TEXT.strip()), 4)     # a loop

        eng2, _ = self.cycle_engine()
        on = backend(eng2, defaults=SamplerParams(temperature=0.0, dry_multiplier=0.8))
        text_on, fin_on = run(on, req([("user", "tell me")], max_tokens=60))
        joined = "".join(text_on)
        self.assertLess(joined.count(self.CYCLE_TEXT.strip()), 2, joined)
        smp = fin_on.info["sampler"]
        self.assertEqual((smp["dry_multiplier"], smp["dry_penalty_last_n"]), (0.8, -1))
        self.assertEqual(smp["dry_sequence_breakers"], list(DRY_BREAKERS))
        # per request: DRY off again -> the loop
        eng3, _ = self.cycle_engine()
        on3 = backend(eng3, defaults=SamplerParams(temperature=0.0, dry_multiplier=0.8))
        text_req, _ = run(on3, req([("user", "tell me")], max_tokens=60, raw={"dry_multiplier": 0}))
        self.assertEqual(text_req, text_off)

    def test_per_request_breakers_and_restore(self):
        b = backend(ScriptedEngine(tok().encode("ok")), defaults=SamplerParams(temperature=0.0,
                                                                                dry_multiplier=0.8))
        run(b, req([("user", "hi")], raw={"dry_sequence_breakers": [" cat"]}))
        self.assertEqual(b._breakers_set, (" cat",))
        run(b, req([("user", "hi")]))
        self.assertEqual(b._breakers_set, DRY_BREAKERS)

    def test_server_default_is_on(self):
        b = Smollm2Backend(ScriptedEngine(tok().encode("ok")), TOKENIZER,
                           sampler_lib=sampler_lib() or "/nonexistent")
        self.assertEqual((b.defaults.dry_multiplier, b.defaults.dry_base, b.defaults.dry_allowed_length),
                         (0.8, 1.75, 2))
        self.assertEqual((b.dry_last_n, b.dry_breakers), (-1, DRY_BREAKERS))


@unittest.skipUnless(HAVE, "demo/chat/assets/smollm2-135m-instruct/tokenizer.json not present")
class TestHTTP(SchemaMixin, unittest.TestCase):

    def setUp(self):
        LOG.clear()
        self.reply = "Paris is the capital of France. 😀"
        self.eng = ScriptedEngine(tok().encode(self.reply))
        self.b = backend(self.eng)
        self.s = RunningServer({MID: self.b})

    def tearDown(self):
        self.s.close()

    def body(self, **kw):
        b = {"model": MID, "messages": [{"role": "user", "content": "What is the capital of France?"}]}
        b.update(kw)
        return b

    def test_non_stream(self):
        st, _, out = self.s.post(self.body())
        self.assertEqual(st, 200, out)
        obj = json.loads(out)
        self.assert_completion(obj, model=MID)
        self.assertEqual(obj["choices"][0]["message"]["content"], self.reply)
        self.assertEqual(obj["choices"][0]["finish_reason"], "stop")
        self.assertEqual(obj["usage"]["completion_tokens"], len(tok().encode(self.reply)))
        self.assertEqual(obj["kv260"]["finish"], "eos")
        self.assertNotIn("log", obj["kv260"])
        self.assertTrue(any("reuse=" in x and "prefill=" in x for x in LOG))

    def test_stream(self):
        st, h, out = self.s.post(self.body(stream=True, stream_options={"include_usage": True}))
        self.assertEqual(st, 200)
        ev = sse_events(out)
        content, finish, usage, extra = self.assert_stream(ev, model=MID, include_usage=True)
        self.assertEqual((content, finish), (self.reply, "stop"))
        self.assertEqual(usage["prompt_tokens"], len(tok().encode(chatml.render(self.body()["messages"]))))
        self.assertGreater(len(ev), len(tok().encode(self.reply)) // 2)      # streamed token by token
        self.assertEqual(extra["finish"], "eos")

    def test_stream_equals_non_stream_and_length(self):
        for kw in ({}, {"max_tokens": 4}, {"stop": ["capital"]}, {"stop": "😀"}):
            _, _, a = self.s.post(self.body(**kw))
            _, _, b = self.s.post(self.body(stream=True, **kw))
            a = json.loads(a)
            content, finish, _, _ = self.assert_stream(sse_events(b), model=MID)
            self.assertEqual((content, finish), (a["choices"][0]["message"]["content"],
                                                 a["choices"][0]["finish_reason"]), kw)
        _, _, a = self.s.post(self.body(max_tokens=4))
        self.assertEqual(json.loads(a)["choices"][0]["finish_reason"], "length")

    def test_context_errors(self):
        st, _, out = self.s.post(self.body(messages=[{"role": "user", "content": "word " * 3000}]))
        self.assertEqual(st, 400)
        e = json.loads(out)["error"]
        self.assertEqual((e["code"], e["param"]), ("context_length_exceeded", "messages"))
        st, _, out = self.s.post(self.body(top_k="many"))
        self.assertEqual((st, json.loads(out)["error"]["param"]), (400, "top_k"))

    def test_library_error_mid_stream(self):
        self.eng.fail_on = lambda seq, toks: len(toks) == 1 and seq[-1] == self.eng.reply[2]
        st, _, out = self.s.post(self.body(stream=True))
        ev = sse_events(out)
        self.assertEqual(st, 200)
        self.assertEqual(ev[-1], "[DONE]")
        self.assertIn("error", ev[-2])
        st, _, out = self.s.post(self.body())
        self.assertEqual(st, 500)
        self.eng.fail_on = None
        st, _, out = self.s.post(self.body())
        self.assertEqual((st, json.loads(out)["choices"][0]["message"]["content"]), (200, self.reply))

    def test_disconnect_cancels(self):
        self.eng.reply = tok().encode("word " * 300)
        self.eng.delay = 0.01
        sock = self.s.raw_socket()
        sock.sendall(http_request_bytes(self.s.port, self.body(stream=True)))
        got = b""
        while got.count(b"word") < 3:
            got += sock.recv(4096)
        sock.close()
        self.assertTrue(wait_until(lambda: not self.s.srv.fpga.busy, 5.0))
        self.assertLess(len(self.eng.decodes), 100)
        self.assertTrue(wait_until(lambda: any("cancelled" in x for x in LOG)))
        self.eng.delay = 0.0
        self.eng.reply = tok().encode("again")
        st, _, out = self.s.post(self.body())
        self.assertEqual(json.loads(out)["choices"][0]["message"]["content"], "again")


@unittest.skipUnless(HAVE and fake_llm_lib(), "tokenizer or C compiler missing")
class TestCtypesEngine(unittest.TestCase):
    """LibLlmEngine (the real ctypes binding) over tests/fake_libsmollm2.c."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="fakellm_")
        self.reply = "Hi from the C fake."

    def tearDown(self):
        import shutil
        shutil.rmtree(self.dir, True)

    def write(self, reply_ids, sizes=None):
        with open(os.path.join(self.dir, "script.txt"), "w") as f:
            f.write(" ".join(map(str, GEN_PROMPT)) + "\n" + " ".join(map(str, reply_ids)) + "\n")
        if sizes:
            with open(os.path.join(self.dir, "sizes.txt"), "w") as f:
                f.write("%d %d\n" % sizes)

    def test_generate_and_reuse(self):
        import ctypes
        t = tok()
        self.write(t.encode(self.reply))
        eng = LibLlmEngine(fake_llm_lib(), self.dir)
        b = backend(eng)
        try:
            eng.lib.fake_llm_counters.restype = ctypes.POINTER(ctypes.c_long)
            counters = eng.lib.fake_llm_counters()
            before = counters[1]
            m1 = [{"role": "user", "content": "hello"}]
            pieces, fin = run(b, req(m1))
            self.assertEqual("".join(pieces), self.reply)
            p1 = t.encode(chatml.render(m1))
            self.assertEqual(counters[1] - before, len(p1) - 1)
            m2 = m1 + [{"role": "assistant", "content": self.reply}, {"role": "user", "content": "more"}]
            before = counters[1]
            pieces, fin = run(b, req(m2))
            p2 = t.encode(chatml.render(m2))
            self.assertEqual(counters[1] - before, len(p2) - len(p1) - len(t.encode(self.reply)))
            self.assertEqual(eng.position(), len(p2) + len(t.encode(self.reply)))
            with self.assertRaises(LlmLibraryError) as cm:
                eng.truncate(0)
            self.assertIn("llm_truncate(0)", str(cm.exception))
        finally:
            b.close()

    def test_open_errors_and_context(self):
        eng = LibLlmEngine(fake_llm_lib(), os.path.join(self.dir, "missing"))
        with self.assertRaises(LlmLibraryError) as cm:
            eng.open()
        self.assertIn("cannot open", str(cm.exception))
        self.assertFalse(eng.is_open)
        self.write(tok().encode("x " * 200), sizes=(49152, 48))
        eng = LibLlmEngine(fake_llm_lib(), self.dir)
        b = backend(eng, reserve=8)
        try:
            self.assertEqual((eng.context_size, b.ctx), (48, 48))
            pieces, fin = run(b, req([("user", "go")]))
            self.assertEqual(fin.info["finish"], "context_full")
            self.assertEqual(eng.position(), 48)
        finally:
            b.close()


# ── residency: lazy loading and eviction between bert-squad and smollm2 ──────

class FakeBertEngine:
    seq = 256

    def __init__(self, cma):
        self.cma, self.opened, self.opens, self.closes = cma, False, 0, 0

    def open(self):
        if self.cma is not None:
            self.cma.alloc("bert", 224)
        self.opened = True
        self.opens += 1

    def close(self):
        if self.opened:
            if self.cma is not None:
                self.cma.free("bert")
            self.opened = False
            self.closes += 1

    def run(self, ids, seg, mask):
        return [-1280] * self.seq, [-1280] * self.seq


class FakeCma:
    """CmaFree bookkeeping; alloc beyond the total raises like a failed BO."""

    def __init__(self, total, report=True):
        self.total, self.used, self.report = total, {}, report

    def alloc(self, who, mb):
        if sum(self.used.values()) + mb > self.total:
            raise MemoryError(f"{who}: CMA exhausted")
        self.used[who] = mb

    def free(self, who):
        self.used.pop(who, None)

    def free_mb(self):
        return (self.total - sum(self.used.values())) if self.report else None


class CmaScripted(ScriptedEngine):
    def __init__(self, cma, *a, **kw):
        super().__init__(*a, **kw)
        self.cma = cma

    def open(self):
        if not self.opened and self.cma is not None:
            self.cma.alloc("llm", 360)
        super().open()

    def close(self):
        if self.opened and self.cma is not None:
            self.cma.free("llm")
        super().close()


VOCAB_TXT = os.path.join(CHAT, "..", "bert_squad", "assets", "vocab.txt")


@unittest.skipUnless(HAVE and os.path.exists(VOCAB_TXT), "tokenizer or BERT vocab.txt missing")
class TestResidency(unittest.TestCase):

    def make(self, resident, cma):
        from bert_squad_backend import BertSquadBackend
        self.bert_eng = FakeBertEngine(cma)
        bert = BertSquadBackend(self.bert_eng, VOCAB_TXT)
        self.llm_eng = CmaScripted(cma, tok().encode("Generated."))
        llm = Smollm2Backend(self.llm_eng, TOKENIZER, sampler_lib=sampler_lib() or "/nonexistent",
                             defaults=SamplerParams(temperature=0.0))
        backends = {"bert-squad": bert, MID: llm}
        for b in backends.values():
            b.load_host()
        s = RunningServer(backends, resident=resident, cma_free=cma.free_mb if cma else lambda: None)
        s.srv.load_startup()
        return s

    def ask(self, s, model):
        msgs = ([{"role": "system", "content": "Denver won the game."}, {"role": "user", "content": "Who won?"}]
                if model == "bert-squad" else [{"role": "user", "content": "hi"}])
        st, _, out = s.post({"model": model, "messages": msgs})
        return st, json.loads(out)

    def health(self, s):
        return {m["id"]: m["loaded"] for m in json.loads(s.request("GET", "/health")[2])["models"]}

    def test_one(self):
        LOG.clear()
        s = self.make("one", FakeCma(1000))
        try:
            self.assertEqual(self.health(s), {"bert-squad": True, MID: False})
            st, obj = self.ask(s, MID)
            self.assertEqual((st, obj["choices"][0]["message"]["content"]), (200, "Generated."))
            self.assertEqual(self.health(s), {"bert-squad": False, MID: True})
            self.assertEqual((self.bert_eng.opens, self.bert_eng.closes), (1, 1))
            self.assertTrue(any("load=" in x and MID in x for x in LOG))
            st, _ = self.ask(s, "bert-squad")
            self.assertEqual(st, 200)
            self.assertEqual(self.health(s), {"bert-squad": True, MID: False})
            self.assertEqual((self.llm_eng.open_count, self.llm_eng.close_count), (1, 1))
            # after the reload the prefix cache starts over (the library state is gone)
            st, obj = self.ask(s, MID)
            self.assertEqual(obj["kv260"]["cached_tokens"], 1)
            self.assertEqual(json.loads(s.request("GET", "/health")[2])["unloads"], 3)
        finally:
            s.close()

    def test_auto_fits_both(self):
        cma = FakeCma(1000)
        s = self.make("auto", cma)
        try:
            self.assertEqual(self.health(s), {"bert-squad": True, MID: True})   # 224 + 360 + margins fit
            self.assertEqual(self.ask(s, MID)[0], 200)
            self.assertEqual(self.ask(s, "bert-squad")[0], 200)
            self.assertEqual((self.bert_eng.closes, self.llm_eng.close_count), (0, 0))
        finally:
            s.close()

    def test_auto_evicts_by_cma(self):
        cma = FakeCma(600)                             # 224 + 360 = 584 fits the pool, not the margins
        s = self.make("auto", cma)
        try:
            self.assertEqual(self.health(s), {"bert-squad": True, MID: False})   # 376 free < 392
            st, obj = self.ask(s, MID)
            self.assertEqual((st, obj["choices"][0]["message"]["content"]), (200, "Generated."))
            self.assertEqual(self.health(s), {"bert-squad": False, MID: True})
            st, _ = self.ask(s, "bert-squad")                                  # 240 free < 256
            self.assertEqual(st, 200)
            self.assertEqual(self.health(s), {"bert-squad": True, MID: False})
        finally:
            s.close()

    def test_auto_retries_after_failed_load(self):
        cma = FakeCma(500, report=False)               # no CmaFree: find out by failing
        s = self.make("auto", cma)
        try:
            self.assertEqual(self.health(s), {"bert-squad": True, MID: False})
            st, obj = self.ask(s, MID)                 # 224 + 360 > 500: fails, evicts, retries
            self.assertEqual(st, 200)
            self.assertEqual(self.health(s), {"bert-squad": False, MID: True})
        finally:
            s.close()

    def test_load_failure_is_503_then_recovers(self):
        cma = FakeCma(300)                             # smollm2 (360) cannot load at all
        s = self.make("one", cma)
        try:
            st, obj = self.ask(s, MID)
            self.assertEqual((st, obj["error"]["code"]), (503, "model_not_loaded"))
            h = json.loads(s.request("GET", "/health")[2])
            self.assertEqual(h["status"], "error")
            cma.total = 1000
            st, obj = self.ask(s, MID)
            self.assertEqual(st, 200)
            self.assertEqual(json.loads(s.request("GET", "/health")[2])["status"], "ok")
        finally:
            s.close()

    def test_all(self):
        s = self.make("all", FakeCma(10_000))
        try:
            self.assertEqual(self.health(s), {"bert-squad": True, MID: True})
            for m in (MID, "bert-squad", MID):
                self.assertEqual(self.ask(s, m)[0], 200)
            self.assertEqual((self.bert_eng.closes, self.llm_eng.close_count), (0, 0))
        finally:
            s.close()


if __name__ == "__main__":
    unittest.main()
