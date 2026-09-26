"""squad_text.py — the text side of BERT-SQuAD, Python standard library only.

Shared by the numeric study (bert_study.py, which re-imports everything
here), the demo scripts (prepare_inputs.py, deploy_and_run.py via
bert_study) and the KV260 chat server (demo/chat/bert_squad_backend.py),
which runs on the board without numpy:

  Tokenizer          bert-base-uncased WordPiece (basic + wordpiece split)
  build_feature      one 256-token window: [CLS] question [SEP] context [SEP];
                     None when question + context do not fit (the demo's and
                     the study's single-window example sets)
  build_features     sliding windows over a long context (standard SQuAD
                     practice: question <= 64 tokens, doc stride 128, the
                     "max context" flag per token)
  best_span          best answer span of one window (top-20 start / end
                     candidates inside the context, length <= 30)
  best_span_windows  the same across windows, with an n-best list and a
                     softmax "probability" of each candidate
  em / f1            official SQuAD 1.1 answer normalisation and metrics

Logits may be any indexable sequence of numbers (Python lists, numpy
arrays); the arithmetic is done on the elements as given, so numpy float32
input sums in float32 exactly as the study always did.

Tie order.  Candidates are ranked by descending logit, and equal logits by
descending position — the order of np.argsort(x, kind="stable")[::-1],
which is what the study's former np.argsort(x)[::-1] gave on its data (a
quicksort's tie order is implementation-defined; see
demo/chat/tests/test_squad_text.py for the check against it).
"""

from __future__ import annotations

import collections
import math
import re
import string
import unicodedata

SEQ = 256            # model sequence length (bertsquad-12 pinned to [1, 256])
MAX_QUERY = 64       # question tokens kept (standard SQuAD)
DOC_STRIDE = 128     # window stride over the context tokens
N_BEST = 20          # start / end candidates per window
MAX_ANSWER = 30      # answer length limit in WordPiece tokens


# ----------------------------------------------------------------- tokenizer
def _is_punct(ch):
    cp = ord(ch)
    if 33 <= cp <= 47 or 58 <= cp <= 64 or 91 <= cp <= 96 or 123 <= cp <= 126:
        return True
    return unicodedata.category(ch).startswith("P")


def _is_ws(ch):
    return ch in " \t\n\r" or unicodedata.category(ch) == "Zs"


def _is_ctrl(ch):
    if ch in "\t\n\r":
        return False
    return unicodedata.category(ch) in ("Cc", "Cf")


class Tokenizer:
    """bert-base-uncased WordPiece tokenizer (lower case, accents stripped,
    punctuation split, greedy longest-match-first word pieces)."""

    def __init__(self, vocab_path):
        self.vocab = {}
        with open(vocab_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                self.vocab[line.rstrip("\n")] = i

    def basic(self, text):
        text = "".join(" " if _is_ws(c) else c for c in text
                       if not (ord(c) == 0 or ord(c) == 0xFFFD or _is_ctrl(c)))
        out = []
        for tok in text.strip().split():
            tok = unicodedata.normalize("NFD", tok.lower())
            tok = "".join(c for c in tok if unicodedata.category(c) != "Mn")
            cur = ""
            for c in tok:
                if _is_punct(c):
                    if cur:
                        out.append(cur)
                        cur = ""
                    out.append(c)
                else:
                    cur += c
            if cur:
                out.append(cur)
        return out

    def wordpiece(self, token):
        if len(token) > 100:
            return ["[UNK]"]
        out, start = [], 0
        while start < len(token):
            end, cur = len(token), None
            while start < end:
                sub = token[start:end]
                if start > 0:
                    sub = "##" + sub
                if sub in self.vocab:
                    cur = sub
                    break
                end -= 1
            if cur is None:
                return ["[UNK]"]
            out.append(cur)
            start = end
        return out

    def tokenize(self, text):
        return [p for t in self.basic(text) for p in self.wordpiece(t)]

    def ids(self, tokens):
        unk = self.vocab["[UNK]"]
        return [self.vocab.get(t, unk) for t in tokens]


# ----------------------------------------------------------------- features
def whitespace_tokens(context):
    """SQuAD's whitespace split of the context: answers are reported as
    ' '.join() of these original words (punctuation stays attached)."""
    doc, prev_ws = [], True
    for c in context:
        if _is_ws(c):
            prev_ws = True
        else:
            if prev_ws:
                doc.append(c)
            else:
                doc[-1] += c
            prev_ws = False
    return doc


def tokenize_context(tok, context):
    """(doc words, sub_to_orig, context word pieces) — the part of feature
    building that depends only on the context (cacheable per document)."""
    doc = whitespace_tokens(context)
    sub_to_orig, ctx = [], []
    for i, w in enumerate(doc):
        for p in tok.tokenize(w):
            sub_to_orig.append(i)
            ctx.append(p)
    return doc, sub_to_orig, ctx


def _make_feature(tok, q, ctx, seq):
    """ids / segment ids / mask (Python lists, padded to seq) for
    [CLS] q [SEP] ctx [SEP]."""
    tokens = ["[CLS]"] + q + ["[SEP]"] + ctx + ["[SEP]"]
    seg = [0] * (len(q) + 2) + [1] * (len(ctx) + 1)
    ids = tok.ids(tokens)
    n = len(ids)
    ids += [0] * (seq - n)
    seg += [0] * (seq - n)
    mask = [1] * n + [0] * (seq - n)
    return ids, seg, mask


def build_feature(tok, question, context, seq=SEQ):
    """One window holding the whole context, or None when question +
    context do not fit.  ids / seg / mask are lists of length seq."""
    q = tok.tokenize(question)[:MAX_QUERY]
    doc, sub_to_orig, ctx = tokenize_context(tok, context)
    room = seq - len(q) - 3
    if len(ctx) > room:
        return None                      # single-window examples only
    ids, seg, mask = _make_feature(tok, q, ctx, seq)
    return dict(ids=ids, seg=seg, mask=mask, ctx_off=len(q) + 2, n_ctx=len(ctx),
                sub_to_orig=sub_to_orig, doc=doc)


def doc_windows(n_doc, room, stride=DOC_STRIDE):
    """(start, length) of each window over n_doc context tokens: windows of
    at most `room` tokens, each starting `stride` tokens after the previous
    one (or right after it when it is shorter than the stride), the last
    one ending at the last token."""
    if room <= 0:
        raise ValueError(f"no room for context tokens (room {room})")
    spans, start = [], 0
    while start < n_doc:
        length = min(n_doc - start, room)
        spans.append((start, length))
        if start + length == n_doc:
            break
        start += min(length, stride)
    return spans


def is_max_context(spans, cur, pos):
    """True when window `cur` is the window in which context token `pos`
    has the most context on its shorter side (ties: the longer window, then
    the earlier one) — so each token's start logit is used from exactly one
    window."""
    best_score, best = None, None
    for i, (start, length) in enumerate(spans):
        end = start + length - 1
        if pos < start or pos > end:
            continue
        score = min(pos - start, end - pos) + 0.01 * length
        if best_score is None or score > best_score:
            best_score, best = score, i
    return cur == best


def build_features(tok, question, context, seq=SEQ, stride=DOC_STRIDE,
                   max_query=MAX_QUERY, pre=None):
    """Sliding-window features over a context of any length.

    Returns a list (one dict per window, in document order) with the
    single-window keys (ids, seg, mask, ctx_off, n_ctx, sub_to_orig — local
    to the window —, doc) plus window, n_windows, doc_start (first context
    token of the window), max_ctx (per window token: this window is its
    max-context window) and n_tokens (real tokens incl. [CLS] / [SEP]).
    A context that fits one window gives exactly build_feature()'s feature.
    `pre` = tokenize_context(tok, context), to reuse a cached document."""
    q = tok.tokenize(question)[:max_query]
    doc, sub_all, ctx_all = pre if pre is not None else tokenize_context(tok, context)
    room = seq - len(q) - 3
    spans = doc_windows(len(ctx_all), room, stride)
    feats = []
    for w, (start, length) in enumerate(spans):
        ctx = ctx_all[start:start + length]
        ids, seg, mask = _make_feature(tok, q, ctx, seq)
        feats.append(dict(
            ids=ids, seg=seg, mask=mask, ctx_off=len(q) + 2, n_ctx=length,
            sub_to_orig=sub_all[start:start + length], doc=doc,
            window=w, n_windows=len(spans), doc_start=start,
            max_ctx=[is_max_context(spans, w, start + j) for j in range(length)],
            n_tokens=len(q) + length + 3))
    return feats


# ----------------------------------------------------------------- span decoding
def _top(logits, lo, hi, n_best, ok=None):
    """Positions lo <= i < hi (with ok[i - lo], if given) by descending
    logit, equal logits by descending position; the first n_best."""
    idx = [i for i in range(lo, hi) if ok is None or ok[i - lo]]
    idx.sort(key=lambda i: (logits[i], i), reverse=True)
    return idx[:n_best]


def span_text(f, s, e):
    """Answer text of window positions s..e: the original whitespace words
    that their word pieces come from."""
    lo = f["ctx_off"]
    o0, o1 = f["sub_to_orig"][s - lo], f["sub_to_orig"][e - lo]
    return " ".join(f["doc"][o0:o1 + 1])


def best_span(f, start, end, n_best=N_BEST, max_len=MAX_ANSWER):
    """Best (start, end) of one window: the top n_best start and end
    positions inside the context, e >= s, length <= max_len, maximal
    start + end logit (the first found on ties).  Returns ((s, e), text);
    (ctx_off, ctx_off) when no pair qualifies."""
    lo, hi = f["ctx_off"], f["ctx_off"] + f["n_ctx"]
    s_idx = _top(start, lo, hi, n_best)
    e_idx = _top(end, lo, hi, n_best)
    best, score = (lo, lo), -1e30
    for s in s_idx:
        for e in e_idx:
            if s <= e < s + max_len and start[s] + end[e] > score:
                score, best = start[s] + end[e], (s, e)
    return best, span_text(f, *best)


def best_span_windows(feats, logits, n_best=N_BEST, max_len=MAX_ANSWER):
    """Best answer across sliding windows.

    feats: build_features() output (or any subset, in order); logits: per
    window a (start, end) pair of sequences of length seq.  Per window the
    start candidates are the top n_best context positions whose window is
    their max-context window (features without max_ctx: every position),
    the end candidates the top n_best context positions; a pair qualifies
    when s <= e < s + max_len.  The answer maximises start + end logit
    over every window (the first found — earlier window, then higher start,
    then higher end logit — on ties), so for one window it equals
    best_span().

    Returns a dict: text, score, window, start, end (positions in that
    window), prob (softmax of the score over the n-best list), n_best (up
    to n_best distinct answer texts: text, score, prob, window, start, end)
    — or None when no window has a qualifying pair."""
    cands = []
    for w, (f, (st, en)) in enumerate(zip(feats, logits)):
        lo, hi = f["ctx_off"], f["ctx_off"] + f["n_ctx"]
        s_idx = _top(st, lo, hi, n_best, f.get("max_ctx"))
        e_idx = _top(en, lo, hi, n_best)
        for s in s_idx:
            for e in e_idx:
                if s <= e < s + max_len:
                    cands.append((st[s] + en[e], w, s, e))
    if not cands:
        return None
    cands.sort(key=lambda c: c[0], reverse=True)       # stable: ties keep discovery order
    nbest, seen = [], set()
    for score, w, s, e in cands:
        text = span_text(feats[w], s, e)
        if text in seen:
            continue
        seen.add(text)
        nbest.append({"text": text, "score": float(score), "window": w,
                      "start": s, "end": e})
        if len(nbest) == n_best:
            break
    top = nbest[0]["score"]
    ex = [math.exp(c["score"] - top) for c in nbest]
    tot = sum(ex)
    for c, x in zip(nbest, ex):
        c["prob"] = x / tot
    best = dict(nbest[0])
    best["n_best"] = nbest
    return best


# ----------------------------------------------------------------- SQuAD metrics
def _norm(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1(pred, gold):
    p, g = _norm(pred).split(), _norm(gold).split()
    common = collections.Counter(p) & collections.Counter(g)
    ns = sum(common.values())
    if ns == 0:
        return 0.0
    pr, rc = ns / len(p), ns / len(g)
    return 2 * pr * rc / (pr + rc)


def em(pred, gold):
    return float(_norm(pred) == _norm(gold))
