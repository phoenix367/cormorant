"""Fake decoder engines with the §11 contract of libsmollm2.so (doc/CHAT_PLAN.md),
at the level smollm2_backend.LibLlmEngine exposes it (open / close /
position / truncate / prefill / decode, vocab_size / context_size, logits as
a ctypes float array):

  ScriptedEngine   fast, for protocol tests: the next token comes from a
                   script (by default: after every "<|im_start|>assistant\\n"
                   the reply's tokens, then <|im_end|>); logits are a peak of
                   `peak` at the scripted token over zeros.  Records every call
                   (prefilled token lists, decodes, truncations).
  FloatModelEngine exact and slow: demo/chat/scripts/llm_study.py's numpy
                   float64 Llama forward from the safetensors weights with its
                   KV cache — the same next-token logits as transformers'
                   float model (needs numpy and the assets).  Position 0 holds
                   <|im_start|> (the sink), as in the real library.

tests/fake_libsmollm2.c is the same contract as a C shared library (scripted),
for the ctypes path.
"""

import ctypes
import os
import sys
import threading
import time
from typing import Callable, List, Optional, Sequence

HERE = os.path.dirname(os.path.abspath(__file__))
CHAT = os.path.dirname(HERE)
ASSETS = os.environ.get("SMOLLM_ASSETS", os.path.join(CHAT, "assets", "smollm2-135m-instruct"))

VOCAB = 49152
IM_START, IM_END, ENDOFTEXT = 1, 2, 0
GEN_PROMPT = (1, 520, 9531, 198)            # "<|im_start|>assistant\n"


class FakeLibraryError(RuntimeError):
    pass


class _Base:
    def __init__(self, vocab_size=VOCAB, context_size=1024):
        self.vocab_size = vocab_size
        self.context_size = context_size
        self.logits = (ctypes.c_float * vocab_size)()
        self.seq: List[int] = []
        self.opened = False
        self.open_count = self.close_count = 0
        self.prefills: List[List[int]] = []
        self.decodes: List[int] = []
        self.truncates: List[int] = []
        self.lock = threading.Lock()
        self.active = self.max_active = 0

    def open(self):
        if self.opened:
            return
        self.opened = True
        self.open_count += 1
        self.seq = [IM_START]                   # the sink

    def close(self):
        if self.opened:
            self.opened = False
            self.close_count += 1

    @property
    def is_open(self):
        return self.opened

    def info(self):
        return {"library": type(self).__name__, "vocab_size": self.vocab_size,
                "context_size": self.context_size}

    def _need_open(self):
        if not self.opened:
            raise FakeLibraryError("llm: not open")

    def position(self):
        self._need_open()
        return len(self.seq)

    def truncate(self, n):
        self._need_open()
        if n < 1 or n > len(self.seq):
            raise FakeLibraryError(f"llm_truncate({n}): position is {len(self.seq)}")
        self.truncates.append(n)
        del self.seq[n:]
        self._truncated(n)

    def _truncated(self, n):
        pass

    def _enter(self):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)

    def _leave(self):
        with self.lock:
            self.active -= 1

    def prefill(self, tokens: Sequence[int]):
        self._need_open()
        tokens = [int(t) for t in tokens]
        if not tokens:
            raise FakeLibraryError("llm_prefill: n must be >= 1")
        if len(self.seq) + len(tokens) > self.context_size:
            raise FakeLibraryError(f"llm_prefill: {len(self.seq)} + {len(tokens)} > context "
                                   f"{self.context_size}")
        if any(not 0 <= t < self.vocab_size for t in tokens):
            raise FakeLibraryError("llm_prefill: token out of range")
        self._enter()
        try:
            self.prefills.append(list(tokens))
            self._run(tokens)
            self.seq.extend(tokens)
            return self.logits
        finally:
            self._leave()

    def decode(self, token: int):
        self._need_open()
        if len(self.seq) >= self.context_size:
            raise FakeLibraryError("llm_decode: the context is full")
        if not 0 <= token < self.vocab_size:
            raise FakeLibraryError("llm_decode: token out of range")
        self._enter()
        try:
            self.decodes.append(int(token))
            self._run([int(token)])
            self.seq.append(int(token))
            return self.logits
        finally:
            self._leave()

    @property
    def prefilled_total(self):
        return sum(len(p) for p in self.prefills)


class ScriptedEngine(_Base):
    """next_token(seq) -> id decides the logits' peak; the default script
    answers every generation prompt with `reply` (token ids) + <|im_end|>.
    `delay`: seconds per decode (cancellation tests).  `logits_fn(seq)`, if
    given, returns the full logits (a sequence of vocab floats) instead."""

    def __init__(self, reply: Sequence[int] = (), next_token: Optional[Callable] = None,
                 logits_fn: Optional[Callable] = None, peak: float = 30.0, delay: float = 0.0,
                 fail_on: Optional[Callable] = None, **kw):
        super().__init__(**kw)
        self.reply = list(reply)
        self.next_token = next_token or self._script
        self.logits_fn = logits_fn
        self.peak = peak
        self.delay = delay
        self.fail_on = fail_on

    def _script(self, seq: List[int]) -> int:
        g = len(GEN_PROMPT)
        for i in range(len(seq) - g, -1, -1):
            if tuple(seq[i:i + g]) == GEN_PROMPT:
                k = len(seq) - (i + g)
                return self.reply[k] if k < len(self.reply) else IM_END
        return IM_END

    def _run(self, tokens):
        if self.fail_on is not None and self.fail_on(self.seq, tokens):
            raise FakeLibraryError("scripted failure")
        if self.delay:
            time.sleep(self.delay)
        seq = self.seq + list(tokens)
        if self.logits_fn is not None:
            vals = self.logits_fn(seq)
            for i, v in enumerate(vals):
                self.logits[i] = v
            return
        ctypes.memset(self.logits, 0, ctypes.sizeof(self.logits))
        self.logits[self.next_token(seq)] = self.peak


class FloatModelEngine(_Base):
    """llm_study.Model (numpy float64, pol=None) with a KV cache of
    context_size positions.  Every prefill / decode appends to the cache and
    returns the last position's logits (as float32)."""

    _shared = {}

    def __init__(self, assets: str = ASSETS, context_size: int = 1024):
        super().__init__(context_size=context_size)
        self.assets = assets
        self.model = None
        self.cache = None

    def open(self):
        if self.opened:
            return
        import numpy as np
        sys.path.insert(0, os.path.join(CHAT, "scripts"))
        import llm_study as ls
        key = os.path.abspath(self.assets)
        if key not in self._shared:                         # weights once per process
            cfg, W = ls.load_all(self.assets)
            self._shared[key] = (cfg, ls.Model(W, cfg, None))
        cfg, self.model = self._shared[key]
        self.np, self.ls = np, ls
        self.vocab_size = cfg.V
        self.logits = (ctypes.c_float * cfg.V)()
        self.cache = ls.Seq(cfg, self.context_size)
        self.model.forward([self.cache], [np.array([IM_START])])       # the sink, position 0
        super().open()

    def _truncated(self, n):
        self.cache.n = n

    def _run(self, tokens):
        np = self.np
        lg = self.model.forward([self.cache], [np.array(tokens, dtype=np.int64)])[0][-1]
        np.ctypeslib.as_array(self.logits)[:] = lg.astype(np.float32)
