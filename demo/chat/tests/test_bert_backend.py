"""Backend A (bert_squad_backend.py) with fake engines: the protocol
mapping (document / question / follow-ups / hints), sliding windows and
their cap, max_tokens / stop, cancellation between windows, the HTTP path,
and — when the demo's board results are present — the same spans as the
demo for its SQuAD questions (replaying the board logits)."""

import json
import os
import random
import struct
import unittest

from _util import DEMO, LOG, RunningServer, sse_events

import squad_text as st
from bert_squad_backend import USAGE_HINT, BertSquadBackend, split_conversation
from chat_backend import Cancelled, CancelToken, ChatRequest, Delta, Finish

ASSETS = os.path.join(DEMO, "bert_squad", "assets")
VOCAB = os.path.join(ASSETS, "vocab.txt")
DEMO_BUILD = os.environ.get("KV260_CHAT_DEMO_BUILD", os.path.join(DEMO, "bert_squad", "build"))

SB50 = ("Super Bowl 50 was an American football game to determine the champion of the National "
        "Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion "
        "Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers "
        "24–10 to earn their third Super Bowl title. The game was played on February 7, 2016, at "
        "Levi's Stadium in the San Francisco Bay Area at Santa Clara, California.")


class PlantedEngine:
    """Start / end logits 8.0 on the first occurrence of `answer`'s word
    pieces inside a window's context, -5.0 elsewhere."""

    seq = st.SEQ

    def __init__(self, tok, answer):
        self.target = tok.ids(tok.tokenize(answer))
        self.calls = 0
        self.opened = self.closed = False

    def open(self):
        self.opened = True

    def close(self):
        self.closed = True

    def run(self, ids, seg, mask):
        self.calls += 1
        s, e = [-1280] * self.seq, [-1280] * self.seq
        n = len(self.target)
        for i in range(self.seq - n + 1):
            if seg[i] == 1 and mask[i] == 1 and ids[i:i + n] == self.target and seg[i + n - 1] == 1:
                s[i], e[i + n - 1] = 2048, 2048
                break
        return s, e


class RandomEngine:
    """Deterministic pseudo-random Q8.8 logit bits per window."""

    seq = st.SEQ

    def __init__(self):
        self.calls = 0

    def open(self):
        pass

    def close(self):
        pass

    def run(self, ids, seg, mask):
        self.calls += 1
        rng = random.Random(hash(tuple(ids)))
        return ([rng.randint(-1500, 1500) for _ in range(self.seq)],
                [rng.randint(-1500, 1500) for _ in range(self.seq)])


def req(messages, **kw):
    return ChatRequest(model="bert-squad", messages=[{"role": r, "content": c} for r, c in messages], **kw)


def run(backend, request, cancel=None):
    job = backend.prepare(request)
    events = list(backend.generate(job, cancel or CancelToken()))
    assert isinstance(events[-1], Finish) and all(isinstance(e, Delta) for e in events[:-1])
    return "".join(e.text for e in events[:-1]), events[-1]


@unittest.skipUnless(os.path.exists(VOCAB), "demo/bert_squad/assets/vocab.txt not present")
class TestBertBackend(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tok = st.Tokenizer(VOCAB)

    def backend(self, engine, **kw):
        b = BertSquadBackend(engine, VOCAB, **kw)
        b.load()
        return b

    def test_split_conversation(self):
        self.assertEqual(split_conversation([{"role": "user", "content": "hi"}]), (None, "hi"))
        self.assertEqual(split_conversation([{"role": "system", "content": "Context: D"},
                                             {"role": "user", "content": "q1"},
                                             {"role": "assistant", "content": "a1"},
                                             {"role": "user", "content": "q2"}]), ("D", "q2"))
        self.assertEqual(split_conversation([{"role": "user", "content": "context: The doc."}]),
                         ("The doc.", None))
        self.assertEqual(split_conversation([{"role": "user", "content": "Context: The doc.\n"
                                              "more\nQuestion: What?"}]), ("The doc.\nmore", "What?"))
        self.assertEqual(split_conversation([{"role": "system", "content": "old"},
                                             {"role": "user", "content": "Context: new"},
                                             {"role": "assistant", "content": "ok"},
                                             {"role": "user", "content": "q"}]), ("new", "q"))
        self.assertEqual(split_conversation([{"role": "developer", "content": "Document: X"},
                                             {"role": "user", "content": "  "}]), ("X", None))

    def test_hints(self):
        eng = PlantedEngine(self.tok, "Denver Broncos")
        b = self.backend(eng)
        text, fin = run(b, req([("user", "Who won Super Bowl 50?")]))
        self.assertEqual(text, USAGE_HINT)
        self.assertEqual(fin.reason, "stop")
        self.assertTrue(fin.info["hint"])
        text, fin = run(b, req([("user", "Context: " + SB50)]))
        self.assertTrue(text.startswith("Got the document ("), text)
        text, _ = run(b, req([("system", "   ")]))
        self.assertEqual(text, USAGE_HINT)
        self.assertEqual(eng.calls, 0)                            # no FPGA work for hints

    def test_answer_and_follow_up(self):
        eng = PlantedEngine(self.tok, "Denver Broncos")
        b = self.backend(eng)
        text, fin = run(b, req([("system", SB50), ("user", "Which NFL team won?")]))
        self.assertEqual(text, "Denver Broncos")
        self.assertEqual((fin.reason, fin.completion_tokens), ("stop", 2))
        self.assertEqual((fin.info["windows"], fin.info["windows_total"], fin.info["truncated"]),
                         (1, 1, False))
        self.assertGreater(fin.info["confidence"], 0.99)
        self.assertEqual(fin.prompt_tokens, len(self.tok.tokenize("Which NFL team won?"))
                         + len(self.tok.tokenize(SB50)) + 3)
        # follow-up in the same conversation reuses the document
        text, _ = run(b, req([("system", SB50), ("user", "Which NFL team won?"),
                              ("assistant", "Denver Broncos"), ("user", "Who won again?")]))
        self.assertEqual(text, "Denver Broncos")
        # Context: + Question: in one user message
        text, _ = run(b, req([("user", f"Context: {SB50}\nQuestion: Which team won?")]))
        self.assertEqual(text, "Denver Broncos")
        self.assertEqual(eng.calls, 3)

    def test_max_tokens_and_stop(self):
        b = self.backend(PlantedEngine(self.tok, "Levi's Stadium in the San Francisco Bay Area"))
        text, fin = run(b, req([("system", SB50), ("user", "Where?")]))
        self.assertEqual(text, "Levi's Stadium in the San Francisco Bay Area")
        text, fin = run(b, req([("system", SB50), ("user", "Where?")], max_tokens=4))
        self.assertEqual((text, fin.reason, fin.completion_tokens), ("Levi's Stadium", "length", 4))
        text, fin = run(b, req([("system", SB50), ("user", "Where?")], stop=[" in "]))
        self.assertEqual((text, fin.reason), ("Levi's Stadium", "stop"))

    def long_doc(self, n=40):
        rng = random.Random(5)
        teams = ["Broncos", "Panthers", "Patriots", "Steelers", "Cowboys", "Giants", "Jets", "Rams"]
        sents = [f"In week {i} the {rng.choice(teams)} beat the {rng.choice(teams)} by "
                 f"{rng.randint(1, 30)} points at home." for i in range(n)]
        sents[int(n * 0.8)] = "The championship trophy was carried by Marguerite Okonkwo-Lindqvist."
        return " ".join(sents)

    def test_long_document_windows(self):
        doc = self.long_doc()
        eng = PlantedEngine(self.tok, "Marguerite Okonkwo-Lindqvist")
        b = self.backend(eng)
        text, fin = run(b, req([("system", doc), ("user", "Who carried the championship trophy?")]))
        feats = st.build_features(self.tok, "Who carried the championship trophy?", doc)
        self.assertGreaterEqual(len(feats), 3)
        self.assertEqual(text, "Marguerite Okonkwo-Lindqvist.")
        self.assertEqual((fin.info["windows"], fin.info["windows_total"]), (len(feats), len(feats)))
        self.assertEqual(eng.calls, len(feats))
        self.assertGreaterEqual(fin.info["window"], 1)
        self.assertEqual(len(fin.info["window_ms"]), len(feats))
        self.assertEqual(fin.prompt_tokens, sum(f["n_tokens"] for f in feats))
        # capped: the answer lies beyond the first window
        eng.calls = 0
        b2 = self.backend(eng, max_windows=1)
        text, fin = run(b2, req([("system", doc), ("user", "Who carried the championship trophy?")]))
        self.assertEqual((fin.info["windows"], fin.info["windows_total"], fin.info["truncated"]),
                         (1, len(feats), True))
        self.assertEqual(eng.calls, 1)
        self.assertNotEqual(text, "Marguerite Okonkwo-Lindqvist.")

    def test_matches_squad_text(self):
        doc = self.long_doc(30)
        eng = RandomEngine()
        b = self.backend(eng)
        q = "How many points did the Jets win by?"
        text, fin = run(b, req([("system", doc), ("user", q)]))
        feats = st.build_features(self.tok, q, doc)
        logits = []
        for f in feats:
            s, e = RandomEngine().run(f["ids"], f["seg"], f["mask"])
            logits.append(([v / 256.0 for v in s], [v / 256.0 for v in e]))
        best = st.best_span_windows(feats, logits)
        self.assertEqual(text, best["text"])
        self.assertEqual((fin.info["window"], fin.info["span"]), (best["window"], [best["start"], best["end"]]))

    def test_cancel_between_windows(self):
        doc = self.long_doc()
        eng = PlantedEngine(self.tok, "Marguerite Okonkwo-Lindqvist")
        b = self.backend(eng)
        cancel = CancelToken(lambda: eng.calls >= 1)
        job = b.prepare(req([("system", doc), ("user", "Who carried it?")]))
        self.assertGreater(len(job.feats), 1)
        with self.assertRaises(Cancelled):
            list(b.generate(job, cancel))
        self.assertEqual(eng.calls, 1)

    def test_http(self):
        eng = PlantedEngine(self.tok, "Denver Broncos")
        b = self.backend(eng)
        s = RunningServer({"bert-squad": b})
        try:
            body = {"model": "bert-squad", "messages": [{"role": "system", "content": SB50},
                                                        {"role": "user", "content": "Who won?"}]}
            st_, _, out = s.post(body)
            obj = json.loads(out)
            self.assertEqual(st_, 200)
            self.assertEqual(obj["choices"][0]["message"]["content"], "Denver Broncos")
            self.assertEqual(obj["kv260"]["windows"], 1)
            self.assertNotIn("log", obj["kv260"])
            st_, _, out = s.post(dict(body, stream=True))
            ev = sse_events(out)
            self.assertEqual("".join((c["choices"][0]["delta"].get("content") or "")
                                     for c in ev[:-1] if c["choices"]), "Denver Broncos")
            self.assertEqual(ev[-2]["kv260"]["windows"], 1)
            self.assertTrue(any("model=bert-squad" in x and "windows=1/1" in x for x in LOG))
            st_, _, out = s.request("GET", "/health")
            self.assertEqual(json.loads(out)["models"][0]["windows_run"], 2)
        finally:
            s.close()


class ReplayEngine:
    """The demo's board logits, looked up by the input window."""

    seq = st.SEQ

    def __init__(self, inputs_bin, logits_bin):
        raw_in = open(inputs_bin, "rb").read()
        raw_lg = open(logits_bin, "rb").read()
        n_in, n_lg = len(raw_in) // (6 * self.seq), len(raw_lg) // (4 * self.seq)
        self.table = {}
        for i in range(min(n_in, n_lg)):
            rec = struct.unpack_from(f"<{3 * self.seq}h", raw_in, i * 6 * self.seq)
            lg = struct.unpack_from(f"<{2 * self.seq}h", raw_lg, i * 4 * self.seq)
            self.table[tuple(rec[:self.seq])] = (list(lg[:self.seq]), list(lg[self.seq:]))
        self.misses = 0

    def open(self):
        pass

    def close(self):
        pass

    def run(self, ids, seg, mask):
        hit = self.table.get(tuple(ids))
        if hit is None:
            self.misses += 1
            return [0] * self.seq, [0] * self.seq
        return hit


@unittest.skipUnless(all(os.path.exists(p) for p in (
    VOCAB, os.path.join(ASSETS, "dev-v1.1.json"), os.path.join(ASSETS, "preprocessed", "inputs.bin"),
    os.path.join(DEMO_BUILD, "logits.bin"), os.path.join(DEMO_BUILD, "results.json"))),
    "the demo's assets / board results are not present (run demo/bert_squad first)")
class TestDemoReplay(unittest.TestCase):
    """The backend builds the demo's exact input windows and decodes the
    demo's board logits to the demo's spans (the board run of the server
    itself is gate 2 in doc/CHAT_PLAN.md §6)."""

    def test_same_spans_as_demo(self):
        eng = ReplayEngine(os.path.join(ASSETS, "preprocessed", "inputs.bin"),
                           os.path.join(DEMO_BUILD, "logits.bin"))
        b = BertSquadBackend(eng, VOCAB)
        b.load()
        ctx = {}
        for art in json.load(open(os.path.join(ASSETS, "dev-v1.1.json"), encoding="utf-8"))["data"]:
            for para in art["paragraphs"]:
                for qa in para["qas"]:
                    ctx[qa["id"]] = para["context"]
        res = json.load(open(os.path.join(DEMO_BUILD, "results.json"), encoding="utf-8"))
        preds = res["predictions"]
        self.assertGreaterEqual(len(preds), 5)
        for p in preds:
            text, fin = run(b, req([("system", ctx[p["qid"]]), ("user", p["question"])]))
            self.assertEqual(text, p["text"], p["question"])
            self.assertEqual(fin.info["span"], p["span"])
            self.assertEqual(fin.info["windows"], 1)
        self.assertEqual(eng.misses, 0)


if __name__ == "__main__":
    unittest.main()
