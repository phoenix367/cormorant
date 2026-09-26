"""bert_squad_backend.py — backend A of the KV260 chat server: extractive
question answering with BERT-base SQuAD (bertsquad-12) on the FPGA.
Python standard library only (ctypes); the text side is
demo/bert_squad/scripts/squad_text.py.

Protocol mapping (doc/CHAT_PLAN.md §3.1)
  document  the latest system (or developer) message, or the latest user
            message that starts with "Context:" — anywhere in the
            conversation, so follow-up questions reuse it; a leading
            "Context:" / "Document:" label is dropped.  A "Context:" user
            message may carry its question on a line starting "Question:".
  question  the last user message (if that is a pure "Context:" message,
            the reply just acknowledges the document).
  reply     the best answer span, quoted from the document (SQuAD style:
            the document's whitespace-separated words, punctuation kept).
  no document -> a short usage hint (no FPGA work).

Long documents are split into 256-token windows (question <= 64 tokens, doc
stride 128); every window is one FPGA inference (~1 s), the best span is
chosen across windows (squad_text.best_span_windows), and --max-windows
(default 8) caps the latency — later windows are skipped and "truncated"
is reported.

Parameters: max_tokens caps the answer in WordPiece tokens (finish_reason
"length"); stop cuts the answer before a stop string; temperature, top_p
and seed are accepted and have no effect (the model is extractive and
deterministic).  usage.prompt_tokens = real tokens over the windows run,
completion_tokens = WordPiece tokens of the answer.  The non-standard
"kv260" object in the response carries windows, the best span's window /
positions, its score (start + end logit) and confidence (softmax over the
n-best list), FPGA time and the top n-best answers.
"""

from __future__ import annotations

import collections
import ctypes
import hashlib
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import squad_text
from chat_backend import (Backend, BackendError, CancelToken, ChatRequest, Delta, Finish,
                          apply_stop)

USAGE_HINT = (
    "I answer questions about a document by quoting the passage of it that answers "
    "them (BERT-base fine-tuned on SQuAD, running on the KV260 FPGA). Send the document "
    "first - as the system message, or as a user message starting with \"Context:\" - and "
    "then ask questions about it.")
MAX_DOC_CHARS = 200_000

_CONTEXT = re.compile(r"^\s*(context|document)\s*:\s*", re.I)
_QUESTION = re.compile(r"^\s*question\s*:\s*", re.I | re.M)


class LibBertEngine:
    """libbert_squad.so through ctypes: one window in, raw Q8.8 logit bits
    out (see demo/bert_squad/src/bert_api.h)."""

    def __init__(self, lib_path: str, weights_dir: Optional[str] = None):
        self.lib_path = lib_path
        self.weights_dir = weights_dir
        self.lib = None
        self.seq = 0

    def open(self) -> None:
        lib = ctypes.CDLL(os.path.abspath(self.lib_path))
        p16 = ctypes.POINTER(ctypes.c_int16)
        lib.bert_open.argtypes = [ctypes.c_char_p]
        lib.bert_open.restype = ctypes.c_int
        lib.bert_run.argtypes = [p16, p16, p16, p16, p16]
        lib.bert_run.restype = ctypes.c_int
        lib.bert_close.argtypes = []
        lib.bert_close.restype = None
        lib.bert_seq_len.restype = ctypes.c_uint
        for f in (lib.bert_model_name, lib.bert_weights_dir, lib.bert_last_error):
            f.argtypes = []
            f.restype = ctypes.c_char_p
        rc = lib.bert_open(self.weights_dir.encode() if self.weights_dir else None)
        if rc != 0:
            raise RuntimeError(f"bert_open: {lib.bert_last_error().decode()} (rc {rc})")
        self.lib = lib
        self.seq = int(lib.bert_seq_len())
        self._bufs = [(ctypes.c_int16 * self.seq)() for _ in range(5)]

    def info(self) -> Dict[str, Any]:
        if self.lib is None:
            return {}
        return {"library": self.lib_path, "model_name": self.lib.bert_model_name().decode(),
                "weights_dir": (self.weights_dir or self.lib.bert_weights_dir().decode()),
                "seq_len": self.seq}

    def run(self, ids: List[int], seg: List[int], mask: List[int]) -> Tuple[List[int], List[int]]:
        a, b, c, s, e = self._bufs
        a[:], b[:], c[:] = ids, seg, mask
        rc = self.lib.bert_run(a, b, c, s, e)
        if rc != 0:
            raise RuntimeError(f"bert_run: {self.lib.bert_last_error().decode()} (rc {rc})")
        return list(s), list(e)

    def close(self) -> None:
        if self.lib is not None:
            self.lib.bert_close()
            self.lib = None


def split_conversation(messages: List[Dict[str, str]]) -> Tuple[Optional[str], Optional[str]]:
    """(document, question) per the protocol mapping in the module docstring."""
    doc: Optional[str] = None
    question: Optional[str] = None
    for m in messages:
        text = m["content"].strip()
        if m["role"] in ("system", "developer"):
            if text:
                doc = _CONTEXT.sub("", text, count=1).strip()
        elif m["role"] == "user":
            question = text
            if _CONTEXT.match(text):
                body = _CONTEXT.sub("", text, count=1)
                qs = list(_QUESTION.finditer(body))
                if qs:
                    doc, question = body[:qs[-1].start()].strip(), body[qs[-1].end():].strip()
                else:
                    doc, question = body.strip(), None
    return (doc or None), (question or None)


@dataclass
class Job:
    req: ChatRequest
    hint: Optional[str] = None
    feats: List[dict] = field(default_factory=list)
    n_windows: int = 0                      # windows the document needs
    question: str = ""


class BertSquadBackend(Backend):
    model_id = "bert-squad"
    owned_by = "kv260"
    fingerprint = "kv260-bertsquad12-q8.8"

    def __init__(self, engine, vocab_path: str, *, max_windows: int = 8,
                 doc_stride: int = squad_text.DOC_STRIDE, n_best: int = squad_text.N_BEST,
                 max_answer: int = squad_text.MAX_ANSWER, doc_cache: int = 8):
        self.engine = engine
        self.vocab_path = vocab_path
        self.max_windows = max_windows
        self.doc_stride = doc_stride
        self.n_best = n_best
        self.max_answer = max_answer
        self.tok: Optional[squad_text.Tokenizer] = None
        self._docs: "collections.OrderedDict[str, tuple]" = collections.OrderedDict()
        self._doc_cache = doc_cache
        self._docs_lock = threading.Lock()      # prepare() runs in concurrent handler threads
        self.seq = squad_text.SEQ
        self.windows_run = 0

    # ── lifecycle ──

    def load(self) -> None:
        self.tok = squad_text.Tokenizer(self.vocab_path)
        self.engine.open()
        seq = getattr(self.engine, "seq", 0) or squad_text.SEQ
        self.seq = seq

    def close(self) -> None:
        self.engine.close()

    def health(self) -> Dict[str, Any]:
        info = self.engine.info() if hasattr(self.engine, "info") else {}
        return dict(info, max_windows=self.max_windows, windows_run=self.windows_run)

    # ── request -> windows (no FPGA) ──

    def _context(self, doc: str):
        """tokenize_context() of the document, cached (LRU) across requests."""
        key = hashlib.sha1(doc.encode("utf-8")).hexdigest()
        with self._docs_lock:
            pre = self._docs.get(key)
            if pre is not None:
                self._docs.move_to_end(key)
                return pre
        pre = squad_text.tokenize_context(self.tok, doc)
        with self._docs_lock:
            self._docs[key] = pre
            while len(self._docs) > self._doc_cache:
                self._docs.popitem(last=False)
        return pre

    def prepare(self, req: ChatRequest) -> Job:
        doc, question = split_conversation(req.messages)
        if doc is None:
            return Job(req, hint=USAGE_HINT)
        if len(doc) > MAX_DOC_CHARS:
            raise BackendError(f"The document is too long ({len(doc)} characters, limit "
                               f"{MAX_DOC_CHARS}).", "messages")
        pre = self._context(doc)
        if not pre[2]:
            return Job(req, hint="The document is empty. " + USAGE_HINT)
        if question is None:
            n = len(squad_text.doc_windows(len(pre[2]), self.seq - 3 - squad_text.MAX_QUERY,
                                           self.doc_stride))
            return Job(req, hint=(f"Got the document ({len(pre[0])} words, {len(pre[2])} "
                                  f"tokens, up to {n} window{'s' if n != 1 else ''}). Ask me "
                                  f"a question about it."))
        feats = squad_text.build_features(self.tok, question, doc, seq=self.seq,
                                          stride=self.doc_stride, pre=pre)
        return Job(req, feats=feats[:self.max_windows], n_windows=len(feats), question=question)

    # ── windows -> answer (FPGA) ──

    def generate(self, job: Job, cancel: CancelToken):
        req = job.req
        if job.hint is not None:
            text, _ = apply_stop(job.hint, req.stop)
            yield Delta(text)
            yield Finish("stop", sum(len(self.tok.tokenize(m["content"])) for m in req.messages),
                         len(self.tok.tokenize(text)), {"backend": self.model_id, "hint": True,
                                                        "log": {"windows": 0}})
            return
        logits, ms = [], []
        for f in job.feats:
            cancel.check()
            t0 = time.monotonic()
            st, en = self.engine.run(f["ids"], f["seg"], f["mask"])
            ms.append((time.monotonic() - t0) * 1000.0)
            self.windows_run += 1
            logits.append(([v / 256.0 for v in st], [v / 256.0 for v in en]))
        best = squad_text.best_span_windows(job.feats, logits, self.n_best, self.max_answer)
        prompt = sum(f["n_tokens"] for f in job.feats)
        info: Dict[str, Any] = {
            "backend": self.model_id, "windows": len(job.feats), "windows_total": job.n_windows,
            "truncated": job.n_windows > len(job.feats), "fpga_ms": round(sum(ms), 1),
            "window_ms": [round(x, 1) for x in ms]}
        if best is None:                                       # no span qualifies (tiny context)
            info["log"] = {"windows": f"{len(job.feats)}/{job.n_windows}", "fpga_ms": f"{sum(ms):.0f}"}
            yield Delta("")
            yield Finish("stop", prompt, 0, info)
            return
        f = job.feats[best["window"]]
        s, e = best["start"], best["end"]
        n_tok, reason = e - s + 1, "stop"
        text = best["text"]
        if req.max_tokens is not None and n_tok > req.max_tokens:
            n_tok, reason = req.max_tokens, "length"
            text = squad_text.span_text(f, s, s + n_tok - 1)
        text, stopped = apply_stop(text, req.stop)
        if stopped:
            reason = "stop"
            n_tok = len(self.tok.tokenize(text))
        info.update({
            "window": best["window"], "span": [s, e], "score": round(best["score"], 4),
            "confidence": round(best["prob"], 4),
            "n_best": [{"text": c["text"], "score": round(c["score"], 4), "prob": round(c["prob"], 4)}
                       for c in best["n_best"][:5]],
            "log": {"windows": f"{len(job.feats)}/{job.n_windows}", "fpga_ms": f"{sum(ms):.0f}",
                    "conf": f"{best['prob']:.2f}"}})
        yield Delta(text)
        yield Finish(reason, prompt, n_tok, info)
