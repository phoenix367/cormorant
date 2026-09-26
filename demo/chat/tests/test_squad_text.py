"""squad_text.py: tokenizer, sliding-window features and cross-window span
decoding against straightforward reference implementations.  Stdlib only;
uses demo/bert_squad/assets/vocab.txt when present, otherwise a vocabulary
built from the test texts."""

import heapq
import itertools
import os
import random
import tempfile
import unittest

from _util import DEMO

import squad_text as st

VOCAB = os.path.join(DEMO, "bert_squad", "assets", "vocab.txt")

DOC = ("Super Bowl 50 was an American football game to determine the champion of the National "
       "Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion "
       "Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers "
       "24-10 to earn their third Super Bowl title. The game was played on February 7, 2016, at "
       "Levi's Stadium in the San Francisco Bay Area at Santa Clara, California.")


def synthetic_tokenizer(texts):
    """A Tokenizer whose vocabulary holds every basic token of `texts` whole."""
    words = set()
    probe = st.Tokenizer.__new__(st.Tokenizer)
    for t in texts:
        words.update(probe.basic(t))
    fd, path = tempfile.mkstemp(suffix=".txt")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]"] + sorted(words)) + "\n")
    tok = st.Tokenizer(path)
    os.unlink(path)
    return tok


def long_doc(n_sentences=60, seed=0):
    rng = random.Random(seed)
    names = ["Denver", "Carolina", "Santa", "Clara", "Levi's", "Stadium", "Broncos", "Panthers",
             "Manning", "Newton", "Miller", "Kubiak", "Rivera", "Bay", "Area", "California"]
    verbs = ["played", "defeated", "earned", "won", "scored", "passed", "threw", "caught"]
    out = []
    for i in range(n_sentences):
        out.append(f"In game {i} {rng.choice(names)} {rng.choice(verbs)} {rng.randint(1, 99)} "
                   f"yards against {rng.choice(names)}, near {rng.choice(names)}.")
    return " ".join(out)


# ── references (written from the SQuAD recipe, independently of squad_text) ──

def ref_windows(n_doc, room, stride):
    """Windows start at 0, s, 2s, ... (s = min(stride, room)); each covers
    min(room, rest) tokens; stop at the first window that reaches the end."""
    step = min(stride, room)
    out = []
    for start in itertools.count(0, step):
        if start >= n_doc:
            break
        out.append((start, min(room, n_doc - start)))
        if start + room >= n_doc:
            break
    return out


def ref_max_context(spans, pos):
    """Index of the window where token pos has the largest
    min(left, right) + 0.01 * length (first on ties); brute force."""
    scores = []
    for i, (s, n) in enumerate(spans):
        if s <= pos < s + n:
            scores.append((min(pos - s, s + n - 1 - pos) + 0.01 * n, -i))
    return -max(scores)[1]


def ref_best(feats, logits, n_best, max_len):
    """(score, window, s, e) of the best span: per window the n_best
    highest start logits among max-context context positions and the n_best
    highest end logits among context positions (ties: the higher position),
    pairs with s <= e < s + max_len; best total, earliest-found on ties."""
    best = None
    for w, (f, (sl, el)) in enumerate(zip(feats, logits)):
        lo, n = f["ctx_off"], f["n_ctx"]
        starts = heapq.nlargest(n_best, [i for i in range(lo, lo + n)
                                         if f.get("max_ctx") is None or f["max_ctx"][i - lo]],
                                key=lambda i: (sl[i], i))
        ends = heapq.nlargest(n_best, range(lo, lo + n), key=lambda i: (el[i], i))
        for s in starts:
            for e in ends:
                if s <= e and e - s + 1 <= max_len:
                    sc = sl[s] + el[e]
                    if best is None or sc > best[0]:
                        best = (sc, w, s, e)
    return best


def exhaustive_best(feats, logits, max_len):
    """Best valid pair over every position (no n-best restriction)."""
    best = None
    for w, (f, (sl, el)) in enumerate(zip(feats, logits)):
        lo, n = f["ctx_off"], f["n_ctx"]
        for s in range(lo, lo + n):
            if f.get("max_ctx") is not None and not f["max_ctx"][s - lo]:
                continue
            for e in range(s, min(lo + n, s + max_len)):
                sc = sl[s] + el[e]
                if best is None or sc > best[0]:
                    best = (sc, w, s, e)
    return best


def random_logits(rng, seq, q=None):
    """Random logits; q = quantisation step (None: continuous, no ties)."""
    v = [rng.gauss(0, 3) for _ in range(seq)]
    return [round(x / q) * q for x in v] if q else v


class TestTokenizer(unittest.TestCase):

    def test_basic_and_wordpiece(self):
        tok = synthetic_tokenizer(["hello world", "caf"])
        tok.vocab["##e"] = len(tok.vocab)
        self.assertEqual(tok.basic("Hello, WORLD!"), ["hello", ",", "world", "!"])
        self.assertEqual(tok.basic("Café x\tq"), ["cafe", "x", "q"])       # accents, NBSP
        self.assertEqual(tok.tokenize("Café"), ["caf", "##e"])
        self.assertEqual(tok.tokenize("zzz"), ["[UNK]"])
        self.assertEqual(tok.ids(["[CLS]", "nope"]), [tok.vocab["[CLS]"], tok.vocab["[UNK]"]])

    @unittest.skipUnless(os.path.exists(VOCAB), "demo/bert_squad/assets/vocab.txt not present")
    def test_real_vocab(self):
        tok = st.Tokenizer(VOCAB)
        self.assertEqual(len(tok.vocab), 30522)
        self.assertEqual(tok.tokenize("Levi's Stadium, 2016"), ["levi", "'", "s", "stadium", ",", "2016"])
        self.assertEqual(tok.tokenize("unaffable"), ["una", "##ffa", "##ble"])   # greedy longest match

    def test_whitespace_tokens(self):
        self.assertEqual(st.whitespace_tokens("  a  b\tc\n d, "), ["a", "b", "c", "d,"])


class TestWindows(unittest.TestCase):

    def test_doc_windows_vs_reference(self):
        for n_doc in list(range(0, 40)) + [127, 128, 129, 200, 253, 254, 255, 256, 511, 1000, 2047]:
            for room in (1, 2, 5, 31, 64, 128, 189, 200, 253):
                for stride in (1, 3, 64, 128, 300):
                    with self.subTest(n_doc=n_doc, room=room, stride=stride):
                        w = st.doc_windows(n_doc, room, stride)
                        self.assertEqual(w, ref_windows(n_doc, room, stride))
                        covered = set()
                        for s, n in w:
                            covered.update(range(s, s + n))
                        self.assertEqual(covered, set(range(n_doc)))
        with self.assertRaises(ValueError):
            st.doc_windows(10, 0)

    def test_max_context_vs_brute_force(self):
        for n_doc, room, stride in ((1000, 189, 128), (300, 200, 128), (50, 20, 5), (97, 30, 30)):
            spans = st.doc_windows(n_doc, room, stride)
            for pos in range(n_doc):
                owner = ref_max_context(spans, pos)
                flags = [st.is_max_context(spans, i, pos) for i in range(len(spans))]
                self.assertEqual(flags.index(True), owner)
                self.assertEqual(sum(flags), 1)

    def test_single_window_equals_build_feature(self):
        tok = synthetic_tokenizer([DOC, "Who won Super Bowl 50?"])
        f1 = st.build_feature(tok, "Who won Super Bowl 50?", DOC)
        fs = st.build_features(tok, "Who won Super Bowl 50?", DOC)
        self.assertEqual(len(fs), 1)
        f = fs[0]
        for k in f1:
            self.assertEqual(f[k], f1[k], k)
        self.assertEqual((f["window"], f["n_windows"], f["doc_start"]), (0, 1, 0))
        self.assertTrue(all(f["max_ctx"]))
        self.assertEqual(f["n_tokens"], sum(f["mask"]))

    def test_long_document_windows(self):
        doc = long_doc(80)
        question = "Who threw 42 yards against Denver near Santa Clara?"
        tok = synthetic_tokenizer([doc, question])
        doc_words, sub_to_orig, ctx = st.tokenize_context(tok, doc)
        fs = st.build_features(tok, question, doc)
        q = tok.tokenize(question)
        room = st.SEQ - len(q) - 3
        self.assertGreater(len(ctx), 3 * room)
        self.assertEqual([(f["doc_start"], f["n_ctx"]) for f in fs],
                         ref_windows(len(ctx), room, st.DOC_STRIDE))
        self.assertIsNone(st.build_feature(tok, question, doc))           # does not fit one
        owners = [0] * len(ctx)
        for w, f in enumerate(fs):
            s, n = f["doc_start"], f["n_ctx"]
            toks = ["[CLS]"] + q + ["[SEP]"] + ctx[s:s + n] + ["[SEP]"]
            self.assertEqual(f["ids"], tok.ids(toks) + [0] * (st.SEQ - len(toks)))
            self.assertEqual(f["seg"], [0] * (len(q) + 2) + [1] * (n + 1) + [0] * (st.SEQ - len(toks)))
            self.assertEqual(f["mask"], [1] * len(toks) + [0] * (st.SEQ - len(toks)))
            self.assertEqual((f["ctx_off"], f["n_tokens"], f["window"], f["n_windows"]),
                             (len(q) + 2, len(toks), w, len(fs)))
            self.assertEqual(f["sub_to_orig"], sub_to_orig[s:s + n])
            self.assertTrue(f["doc"] == doc_words)
            for j, m in enumerate(f["max_ctx"]):
                owners[s + j] += m
        self.assertEqual(owners, [1] * len(ctx))                           # one owner per token
        # the question is capped at 64 tokens
        long_q = " ".join(["Denver"] * 100)
        fq = st.build_features(tok, long_q, doc)
        self.assertEqual(fq[0]["ctx_off"], st.MAX_QUERY + 2)


class TestSpans(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.doc = long_doc(70, seed=3)
        cls.tok = synthetic_tokenizer([cls.doc, "Who scored?"])
        cls.feats = st.build_features(cls.tok, "Who scored?", cls.doc)

    def test_one_window_equals_best_span(self):
        tok = synthetic_tokenizer([DOC, "Who won?"])
        f = st.build_features(tok, "Who won?", DOC)[0]
        rng = random.Random(7)
        for t in range(400):
            q = (None, 1.0, 0.25, 1 / 256)[t % 4]
            sl, el = random_logits(rng, st.SEQ, q), random_logits(rng, st.SEQ, q)
            (s, e), text = st.best_span(f, sl, el)
            b = st.best_span_windows([f], [(sl, el)])
            self.assertEqual((b["window"], b["start"], b["end"], b["text"]), (0, s, e, text))
            self.assertEqual(b["score"], sl[s] + el[e])

    def test_windows_vs_reference(self):
        rng = random.Random(11)
        feats = self.feats
        self.assertGreaterEqual(len(feats), 4)
        for t in range(150):
            q = (None, 0.5, 1 / 256)[t % 3]
            logits = [(random_logits(rng, st.SEQ, q), random_logits(rng, st.SEQ, q)) for _ in feats]
            for n_best, max_len in ((20, 30), (5, 10), (1, 30), (300, 30)):
                b = st.best_span_windows(feats, logits, n_best, max_len)
                r = ref_best(feats, logits, n_best, max_len)
                if r is None:                               # no qualifying pair (tiny n_best)
                    self.assertIsNone(b)
                    continue
                self.assertEqual((b["score"], b["window"], b["start"], b["end"]), r)
                self.assertEqual(b["text"], st.span_text(feats[r[1]], r[2], r[3]))
                if n_best >= st.SEQ:                        # unrestricted = exhaustive search
                    x = exhaustive_best(feats, logits, max_len)
                    # same best score; the same span unless equal scores tie (quantised logits)
                    self.assertEqual(r[0], x[0])
                    if q is None:
                        self.assertEqual(r, x)
                # n-best list: distinct texts, scores descending, probabilities sum to 1
                nb = b["n_best"]
                self.assertEqual(len({c["text"] for c in nb}), len(nb))
                self.assertEqual([c["score"] for c in nb], sorted((c["score"] for c in nb), reverse=True))
                self.assertAlmostEqual(sum(c["prob"] for c in nb), 1.0, places=9)
                self.assertEqual(nb[0]["text"], b["text"])

    def test_planted_answer(self):
        feats = self.feats
        ctx_len = feats[-1]["doc_start"] + feats[-1]["n_ctx"]
        target = (ctx_len * 2) // 3                      # a token in a middle / late window
        w_own = next(w for w, f in enumerate(feats)
                     if f["doc_start"] <= target < f["doc_start"] + f["n_ctx"]
                     and f["max_ctx"][target - f["doc_start"]])
        logits = []
        for w, f in enumerate(feats):
            sl, el = [-5.0] * st.SEQ, [-5.0] * st.SEQ
            if f["doc_start"] <= target < f["doc_start"] + f["n_ctx"]:
                p = f["ctx_off"] + target - f["doc_start"]
                sl[p] = 8.0 if w == w_own else 9.0      # higher, but not max-context there
                if p + 2 < f["ctx_off"] + f["n_ctx"]:
                    el[p + 2] = 8.0
            logits.append((sl, el))
        b = st.best_span_windows(feats, logits)
        self.assertEqual(b["window"], w_own)
        f = feats[w_own]
        self.assertEqual(b["start"], f["ctx_off"] + target - f["doc_start"])
        self.assertEqual(b["end"], b["start"] + 2)
        self.assertGreater(b["prob"], 0.99)
        words = st.whitespace_tokens(self.doc)
        o0 = f["sub_to_orig"][target - f["doc_start"]]
        self.assertTrue(b["text"].startswith(words[o0]))

    def test_max_answer_length(self):
        tok = synthetic_tokenizer([DOC, "Who won?"])
        f = st.build_features(tok, "Who won?", DOC)[0]
        sl, el = [0.0] * st.SEQ, [0.0] * st.SEQ
        lo = f["ctx_off"]
        sl[lo] = 10.0
        el[lo + 40] = 10.0                               # too long a span: not allowed
        el[lo + 3] = 1.0
        b = st.best_span_windows([f], [(sl, el)])
        self.assertEqual((b["start"], b["end"]), (lo, lo + 3))

    def test_metrics(self):
        self.assertEqual(st.em("The Denver Broncos!", "denver broncos"), 1.0)
        self.assertAlmostEqual(st.f1("Denver Broncos team", "the Denver Broncos"), 0.8)
        self.assertEqual(st.f1("x", "y"), 0.0)


class TestNumpyCompat(unittest.TestCase):
    """The study passes numpy arrays: identical spans to Python lists, and
    float32 arithmetic is kept (no conversion)."""

    def test_numpy_inputs(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy not available")
        tok = synthetic_tokenizer([DOC, "Who won?"])
        f = st.build_feature(tok, "Who won?", DOC)
        rng = np.random.default_rng(0)
        for t in range(200):
            sl = rng.normal(0, 3, st.SEQ).astype(np.float32)
            el = rng.normal(0, 3, st.SEQ).astype(np.float32)
            a = st.best_span(f, sl, el)
            b = st.best_span(f, sl.tolist(), el.tolist())
            self.assertEqual(a, b)
            # the old np.argsort-based code, with a stable sort
            lo, hi = f["ctx_off"], f["ctx_off"] + f["n_ctx"]
            s_idx = [i for i in np.argsort(sl, kind="stable")[::-1] if lo <= i < hi][:20]
            e_idx = [i for i in np.argsort(el, kind="stable")[::-1] if lo <= i < hi][:20]
            best, score = (lo, lo), -1e30
            for s in s_idx:
                for e in e_idx:
                    if s <= e < s + 30 and sl[s] + el[e] > score:
                        score, best = sl[s] + el[e], (s, e)
            self.assertEqual(a[0], (int(best[0]), int(best[1])))


if __name__ == "__main__":
    unittest.main()
