# Copyright 2026 Ivan Gubochkin
# SPDX-License-Identifier: Apache-2.0

"""stories15m (doc/plans/CHAT_PLAN.md §27): llama_tokenizer.py against the
reference ids of `tokenizers` in data/stories15m_text_cases.json (written by
scripts/validate_text.py --model stories15m --write-fixture), how a chat is
read (a new story per message, "continue", "write a story about X"), the
stories backend (stories_backend.py) over
tests/fake_llm.py's ScriptedEngine and over tests/fake_libsmollm2.c through the
real ctypes binding, and the server's POST /v1/completions."""

import json
import os
import shutil
import tempfile
import unittest

from _util import CHAT, LOG, RunningServer, fake_llm_lib, sampler_lib, sse_events

from chat_backend import BackendError, CancelToken, ChatRequest, Delta, Finish
from fake_llm import ScriptedEngine
from llama_tokenizer import Tokenizer
from sampler import SamplerParams
from smollm2_backend import LibLlmEngine
from stories_backend import StoriesBackend, classify, plan_chat

TOKENIZER = os.path.join(CHAT, "assets", "stories15m", "tokenizer.json")
FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "stories15m_text_cases.json")
HAVE = os.path.exists(TOKENIZER)
MID = "stories15m"
BOS, EOS = 1, 2
_TOK = []


def tok():
    if not _TOK:
        _TOK.append(Tokenizer(TOKENIZER))
    return _TOK[0]


def plain(text):
    return tok().encode(text, add_special_tokens=False)


def cont(text):
    """The ids a model emits for the continuation " text ..." of a prompt: its
    first word is "▁text" (encoding the text alone prepends that "▁")."""
    assert text.startswith(" ")
    return plain(text[1:])


class Story(ScriptedEngine):
    """The scripted library of a story model: after any prefill or decode the
    next token is the reply's next one, then <s> (the next story)."""

    def __init__(self, reply, end=BOS, **kw):
        kw.setdefault("vocab_size", 32000)
        kw.setdefault("context_size", 256)
        super().__init__(next_token=self._next, **kw)
        self.reply, self.end, self.k = list(reply), end, 0

    def _next(self, seq):
        t = self.reply[self.k] if self.k < len(self.reply) else self.end
        self.k += 1
        return t

    def restart(self, reply=None):
        self.k = 0
        if reply is not None:
            self.reply = list(reply)


def backend(engine, **kw):
    kw.setdefault("defaults", SamplerParams(temperature=0.0))
    b = StoriesBackend(engine, TOKENIZER, sampler_lib=sampler_lib() or "/nonexistent", **kw)
    b.load_host()
    b.load()
    return b


def run(b, request):
    job = b.prepare(request)
    events = list(b.generate(job, CancelToken()))
    assert isinstance(events[-1], Finish) and all(isinstance(e, Delta) for e in events[:-1]), events
    return "".join(e.text for e in events[:-1]), events[-1]


def completion(prompt, **kw):
    return ChatRequest(model=MID, messages=[], prompt=prompt, **kw)


def chat(*msgs, **kw):
    return ChatRequest(model=MID, messages=[{"role": r, "content": c} for r, c in msgs], **kw)


@unittest.skipUnless(HAVE, "demo/chat/assets/stories15m/tokenizer.json not present")
class TestTokenizer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with open(FIXTURE, encoding="utf-8") as f:
            cls.fx = json.load(f)

    def test_fixture_ids_and_decode(self):
        self.assertGreaterEqual(len(self.fx["strings"]), 500)
        for c in self.fx["strings"]:
            with self.subTest(text=c["text"][:60]):
                self.assertEqual(tok().encode(c["text"]), c["ids"])

    def test_specials_and_spaces(self):
        t = tok()
        self.assertEqual((t.vocab_size, t.bos_id, t.eos_id), (32000, 1, 2))
        self.assertEqual(t.encode("Once upon a time"), [1, 9038, 2501, 263, 931])
        self.assertEqual(t.encode(""), [1])
        self.assertEqual(t.encode("", add_special_tokens=False), [])
        # an added token is "▁<s>": special after a space or at the start only
        self.assertEqual(plain("a <s>b"), [263, 1, 29890])
        self.assertEqual(plain("a<s>b")[1:4], [29966, 29879, 29958])        # "<", "s", ">"
        self.assertEqual(plain("<s>"), [BOS])
        self.assertNotIn(BOS, t.encode("a <s>b", special=False)[1:])        # tokenized as text
        self.assertEqual([t.id_to_token[i] for i in plain("a\nb")], ["▁a", "<0x0A>", "b"])

    def test_decode(self):
        t = tok()
        ids = t.encode("Once upon a time, a dog said: \"Hi!\"\nThe end. 😀")
        self.assertEqual(t.decode(ids), "Once upon a time, a dog said: \"Hi!\"\nThe end. 😀")
        self.assertEqual(t.decode([1, 9038, 2]), "Once")
        self.assertEqual(t.decode([1, 9038, 2], skip_special_tokens=False), " <s> Once </s>"[1:])
        self.assertEqual(t.decode(cont(" there was"), strip_first=False), " there was")
        self.assertEqual(t.decode([t.vocab["<0xE2>"], t.vocab["<0x82>"], 263]), "\ufffd\ufffd a")    # a broken run, then "▁a"

    def test_incremental_decoder(self):
        t = tok()
        ids = cont(" there was a café 😀 and\nmore")
        for strip in (True, False):
            want = t.decode(ids, strip_first=strip)
            for cut in range(len(ids) + 1):
                d = t.incremental_decoder(strip_first=strip)
                self.assertEqual(d.add_many(ids[:cut]) + d.add_many(ids[cut:]) + d.flush(), want)
        d = t.incremental_decoder()
        self.assertEqual(d.add(t.vocab["<0xF0>"]), "")
        self.assertTrue(d.pending)

    def test_token_bytes(self):
        t = tok()
        self.assertEqual(t.token_bytes(9038), b" Once")
        self.assertEqual(t.token_bytes(t.vocab["<0x0A>"]), b"\n")
        self.assertTrue(t.is_special(1) and not t.is_special(9038))


class TestChatPlan(unittest.TestCase):

    def test_classify(self):
        for text, want in (
                ("Hi!", ("story", "Hi!")),
                ("Lily found a red ball.", ("story", "Lily found a red ball.")),
                ("The story was long.", ("story", "The story was long.")),
                ("continue", ("continue", "")), ("Continue the story, please.", ("continue", "")),
                ("go on", ("continue", "")), ("What happened next?", ("continue", "")), ("More!", ("continue", "")),
                ("...and then a big dog came.", ("continue", "and then a big dog came.")),
                ("Write a short story about a cat", ("ask", "Once upon a time, there was a cat")),
                ("Tell me a story", ("ask", "Once upon a time,")),
                ("Can you tell me a bedtime story about two dragons, please?",
                 ("ask", "Once upon a time, there were two dragons")),
                ("Write me a funny story about an elephant.", ("ask", "Once upon a time, there was an elephant")),
                ("tell me a story about cats", ("ask", "Once upon a time, there were cats")),
                ("write a story about friendship", ("ask", "Once upon a time, there was a friendship")),
                ("Tell me a story about Lily", ("ask", "Once upon a time, there was Lily")),
                ("write a story where a dog learns to fly", ("ask", "Once upon a time, a dog learns to fly"))):
            with self.subTest(text=text):
                self.assertEqual(classify(text), want)

    def test_plan(self):
        U = lambda c: {"role": "user", "content": c}                       # noqa: E731
        A = lambda c: {"role": "assistant", "content": c}                  # noqa: E731
        S = {"role": "system", "content": "Be brief."}
        self.assertEqual(plan_chat([S, U("Hi!")]), ("Hi!", "", "new"))
        # every message starts a new story; a request's answer starts with its opening
        self.assertEqual(plan_chat([U("Hi!"), A("He was on a walk."), U("Write a short story about a cat")]),
                         ("Once upon a time, there was a cat", "Once upon a time, there was a cat", "new"))
        self.assertEqual(plan_chat([U("Hi!"), A("He was on a walk."), U("continue")]),
                         ("Hi! He was on a walk.", "", "continue"))
        conv = [U("Write a story about a cat"), A("Once upon a time, there was a cat named Tom."),
                U("continue"), A("Tom ran."), U("...Then a dog came.")]
        self.assertEqual(plan_chat(conv), ("Once upon a time, there was a cat named Tom. Tom ran. Then a dog came.",
                                           "", "continue"))
        self.assertEqual(plan_chat(conv + [A("It barked.")]),        # the assistant's turn last: it goes on
                         ("Once upon a time, there was a cat named Tom. Tom ran. Then a dog came. It barked.",
                          "", "continue"))
        self.assertEqual(plan_chat([S, U("continue")]), ("", "", "continue"))    # nothing to continue


@unittest.skipUnless(HAVE, "demo/chat/assets/stories15m/tokenizer.json not present")
class TestGenerate(unittest.TestCase):

    def test_completion(self):
        reply = " there was a dog. He liked to run."
        eng = Story(cont(reply))
        b = backend(eng)
        text, fin = run(b, completion("Once upon a time"))
        self.assertEqual(text, reply)                                   # the leading space is kept
        prompt = tok().encode("Once upon a time")
        self.assertEqual(eng.prefills, [prompt[1:]])                    # <s> is the sink
        self.assertEqual((fin.reason, fin.prompt_tokens, fin.completion_tokens),
                         ("stop", len(prompt), len(cont(reply))))
        self.assertEqual(fin.info["finish"], "eos")                     # <s>: the next story
        eng.restart()
        eng.end = EOS
        text2, fin2 = run(b, completion("Once upon a time"))
        self.assertEqual((text2, fin2.info["finish"], fin2.info["cached_tokens"]), (reply, "eos", len(prompt) - 1))

    def test_chat(self):
        a1 = " there was a dog."
        eng = Story(cont(a1))
        b = backend(eng)
        text, fin = run(b, chat(("system", "ignored"), ("user", "Once upon a time")))
        self.assertEqual((text, fin.info["story"]), (a1.lstrip(), "new"))   # a message of its own: no space
        # "continue": the same story goes on, its first turn reused from the cache
        eng.restart(cont(" He ran home."))
        text, fin = run(b, chat(("user", "Once upon a time"), ("assistant", text), ("user", "Go on, please.")))
        self.assertEqual((text, fin.info["story"]), ("He ran home.", "continue"))
        story = tok().encode("Once upon a time there was a dog.")
        self.assertEqual(fin.prompt_tokens, len(story))
        self.assertLess(fin.info["prefill_tokens"], len(story) - 1)
        # a request for a story: the answer starts with the opening made of it
        eng.restart(cont(" named Tom. Tom liked fish."))
        text, fin = run(b, chat(("user", "Once upon a time"), ("assistant", "there was a dog."),
                                ("user", "Write a short story about a cat")))
        self.assertEqual(text, "Once upon a time, there was a cat named Tom. Tom liked fish.")
        self.assertEqual((fin.info["story"], fin.info["opening"]), ("new", "Once upon a time, there was a cat"))
        self.assertEqual(fin.prompt_tokens, len(tok().encode("Once upon a time, there was a cat")))
        self.assertEqual(fin.completion_tokens, len(cont(" named Tom. Tom liked fish.")))

    def test_min_tokens(self):
        """After a finished story the model's next token is <s>: a chat "continue"
        still writes 16 tokens, a new chat story 1, a completion none (OpenAI's
        semantics) unless the request sets min_tokens."""
        the = plain("The")[0]

        def logits(seq):                                # <s> first, "▁The" second, always
            v = [0.0] * 32000
            v[BOS], v[the] = 30.0, 20.0
            return v
        b = backend(Story([], logits_fn=logits))
        story = (("user", "Lily found a ball."), ("assistant", "She played with it."))
        for request, n in ((chat(*story, ("user", "continue")), 16), (chat(("user", "Lily found a ball.")), 1),
                           (completion("Lily found a ball."), 0), (completion("Lily found a ball.", raw={"min_tokens": 3}), 3),
                           (chat(*story, ("user", "more"), raw={"min_tokens": 0}), 0)):
            with self.subTest(request=request.prompt or request.messages[-1]["content"], n=n):
                text, fin = run(b, request)
                self.assertEqual((fin.completion_tokens, fin.info["finish"]), (n, "eos"))
                self.assertEqual(text.split(), ["The"] * n)

    def test_empty_and_max_tokens(self):
        b = backend(Story(cont(" a b c d e f")))
        with self.assertRaises(BackendError) as cm:
            b.prepare(completion(""))
        self.assertEqual(cm.exception.param, "prompt")
        for c in (chat(("system", "only instructions")), chat(("user", "continue"))):
            with self.assertRaises(BackendError) as cm:
                b.prepare(c)
            self.assertEqual(cm.exception.param, "messages")
            self.assertIn("no story to continue", cm.exception.message)
        text, fin = run(b, completion("x", max_tokens=3))
        self.assertEqual((fin.reason, fin.completion_tokens, fin.info["finish"]), ("length", 3, "max_tokens"))

    def test_long_story_keeps_its_end(self):
        eng = Story(cont(" The end."))
        b = backend(eng)
        long = " ".join(f"word{i}" for i in range(400))
        ids = b.prepare(completion(long)).ids
        self.assertEqual(len(ids), 256 - 128)                           # context less the reserve
        self.assertEqual(ids[0], BOS)
        self.assertEqual(ids[1:], plain(long)[-(len(ids) - 1):])
        text, fin = run(b, completion(long, max_tokens=8))
        self.assertEqual(text, " The end.")
        self.assertEqual(fin.prompt_tokens, 256 - 8)
        self.assertEqual(fin.info["trimmed_tokens"], len(plain(long)) - (256 - 8 - 1))

    def test_defaults(self):
        b = StoriesBackend(Story([]), TOKENIZER, sampler_lib="/nonexistent")
        d = b.defaults
        self.assertEqual((d.temperature, d.top_p, d.top_k, d.repetition_penalty, d.dry_multiplier),
                         (0.8, 0.9, 0, 1.0, 0.0))
        self.assertTrue(b.completions)
        self.assertEqual((b.model_id, b.fingerprint, b.cma_mb), (MID, "kv260-stories15m-pow2+sink+p12", 50.0))


@unittest.skipUnless(HAVE, "demo/chat/assets/stories15m/tokenizer.json not present")
class TestCompletionsHTTP(unittest.TestCase):

    def setUp(self):
        LOG.clear()
        self.reply = " there was a little dog named Max."
        self.eng = Story(cont(self.reply))
        self.b = backend(self.eng)
        self.s = RunningServer({MID: self.b})

    def tearDown(self):
        self.s.close()

    def post(self, body):
        self.eng.restart()
        return self.s.request("POST", "/v1/completions", body)

    def check(self, obj):
        self.assertTrue(obj["id"].startswith("cmpl-"))
        self.assertEqual((obj["object"], obj["model"]), ("text_completion", MID))
        self.assertEqual(obj["system_fingerprint"], "kv260-stories15m-pow2+sink+p12")
        self.assertEqual(len(obj["choices"]), 1)
        self.assertEqual(set(obj["choices"][0]), {"text", "index", "logprobs", "finish_reason"})

    def test_non_stream(self):
        st, _, out = self.post({"model": MID, "prompt": "Once upon a time", "max_tokens": 64})
        self.assertEqual(st, 200, out)
        obj = json.loads(out)
        self.check(obj)
        self.assertEqual((obj["choices"][0]["text"], obj["choices"][0]["finish_reason"]), (self.reply, "stop"))
        n = len(cont(self.reply))
        self.assertEqual(obj["usage"], {"prompt_tokens": 5, "completion_tokens": n, "total_tokens": 5 + n})
        self.assertEqual(obj["kv260"]["finish"], "eos")

    def test_default_max_tokens_and_echo(self):
        st, _, out = self.post({"prompt": ["Once upon a time"], "echo": True})   # the default model
        obj = json.loads(out)
        self.assertEqual(st, 200, out)
        n = len(cont(self.reply))
        self.assertEqual(obj["choices"][0]["finish_reason"], "length" if n > 16 else "stop")
        self.assertEqual(obj["usage"]["completion_tokens"], min(n, 16))
        self.assertTrue(obj["choices"][0]["text"].startswith("Once upon a time there"))

    def test_stream(self):
        st, h, out = self.post({"model": MID, "prompt": "Once upon a time", "max_tokens": 64, "stream": True,
                                "echo": True, "stream_options": {"include_usage": True}})
        self.assertEqual(st, 200)
        self.assertTrue(h["content-type"].startswith("text/event-stream"))
        ev = sse_events(out)
        self.assertEqual(ev[-1], "[DONE]")
        chunks = ev[:-1]
        self.assertEqual({c["object"] for c in chunks}, {"text_completion"})
        self.assertEqual(len({c["id"] for c in chunks}), 1)
        self.assertTrue(chunks[0]["id"].startswith("cmpl-"))
        text = "".join(c["choices"][0]["text"] for c in chunks if c["choices"])
        self.assertEqual(text, "Once upon a time" + self.reply)
        self.assertEqual([c["choices"][0]["finish_reason"] for c in chunks if c["choices"]][-1], "stop")
        self.assertEqual(chunks[-1]["choices"], [])
        self.assertEqual(chunks[-1]["usage"]["completion_tokens"], len(cont(self.reply)))

    def test_chat_completions_too(self):
        self.eng.restart()
        st, _, out = self.s.post({"model": MID, "messages": [{"role": "user", "content": "Once upon a time"}]})
        self.assertEqual(st, 200, out)
        obj = json.loads(out)
        self.assertEqual(obj["choices"][0]["message"]["content"], self.reply.lstrip())
        self.assertEqual(obj["kv260"]["story"], "new")

    def test_errors(self):
        for body, param in (({"model": MID}, "prompt"), ({"model": MID, "prompt": 5}, "prompt"),
                            ({"model": MID, "prompt": ["a", "b"]}, "prompt"),
                            ({"model": MID, "prompt": "x", "logprobs": 2}, "logprobs"),
                            ({"model": MID, "prompt": "x", "suffix": "y"}, "suffix"),
                            ({"model": MID, "prompt": "x", "best_of": 2}, "best_of"),
                            ({"model": MID, "prompt": "x", "echo": "yes"}, "echo"),
                            ({"model": MID, "prompt": ""}, "prompt")):
            with self.subTest(body=body):
                st, _, out = self.post(body)
                self.assertEqual(st, 400, out)
                self.assertEqual(json.loads(out)["error"]["param"], param)

    def test_models_list(self):
        st, _, out = self.s.request("GET", "/v1/models")
        self.assertEqual(json.loads(out)["data"][0]["kv260"], {"type": "chat", "completions": True})

    def test_chat_model_refuses_completions(self):
        from _util import server_mod
        s = RunningServer({"echo": server_mod.EchoBackend(), MID: self.b})     # echo is the default model
        try:
            self.eng.restart()
            st, _, out = s.request("POST", "/v1/completions", {"prompt": "Once upon a time"})
            self.assertEqual((st, json.loads(out)["model"]), (200, MID))     # the first base model
            st, _, out = s.request("POST", "/v1/completions", {"model": "echo", "prompt": "hi"})
            self.assertEqual(st, 400)
            err = json.loads(out)["error"]
            self.assertEqual(err["code"], "model_not_supported")
            self.assertIn("/v1/completions serves stories15m", err["message"])
            st, _, out = s.request("GET", "/v1/models")
            self.assertEqual([m["kv260"] for m in json.loads(out)["data"]],
                             [{"type": "chat"}, {"type": "chat", "completions": True}])
        finally:
            s.close()


@unittest.skipUnless(HAVE, "demo/chat/assets/stories15m/tokenizer.json not present")
class TestCtypesEngine(unittest.TestCase):
    """The stories15m backend over tests/fake_libsmollm2.c built with the model's
    name, vocabulary and context: the server's build_backends, ctypes, the
    prefix reuse across requests."""

    def setUp(self):
        self.lib = fake_llm_lib(MID)
        if self.lib is None:
            self.skipTest("no C compiler")
        self.dir = tempfile.mkdtemp(prefix="fakestories_")
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.reply = " there was a big red ball."
        with open(os.path.join(self.dir, "script.txt"), "w") as f:      # after "▁time": the reply, then </s>
            f.write(f"{plain('time')[0]}\n" + " ".join(map(str, cont(self.reply))) + "\n")
        with open(os.path.join(self.dir, "sizes.txt"), "w") as f:
            f.write("32000 256\n")

    def test_server_builds_and_runs(self):
        from _util import server_mod
        args = server_mod.parse_args(["--backend", "stories15m", "--stories-lib", self.lib,
                                      "--stories-weights", self.dir, "--stories-tokenizer", TOKENIZER,
                                      "--stories-temperature", "0", "--llm-sampler-lib",
                                      sampler_lib() or "/nonexistent"])
        bs = server_mod.build_backends(args)
        self.assertEqual(list(bs), [MID])
        b = bs[MID]
        self.assertIsInstance(b.engine, LibLlmEngine)
        self.assertEqual((b.cma_mb, b.reserve, b.defaults.temperature), (50.0, 128, 0.0))
        b.load_host()
        b.load()
        try:
            self.assertEqual((b.ctx, b.engine.vocab_size), (256, 32000))
            text, fin = run(b, completion("Once upon a time"))
            self.assertEqual((text, fin.info["finish"]), (self.reply, "eos"))
            text, fin = run(b, completion("Once upon a time"))         # the same prompt: all but one reused
            self.assertEqual((text, fin.info["cached_tokens"], fin.info["prefill_tokens"]), (self.reply, 4, 1))
        finally:
            b.close()


if __name__ == "__main__":
    unittest.main()
