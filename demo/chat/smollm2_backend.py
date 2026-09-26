"""smollm2_backend.py — backend B of the KV260 chat server: generative chat
with SmolLM2-135M-Instruct on the FPGA (libsmollm2.so, doc/CHAT_PLAN.md
§11).  Python standard library only (ctypes); text side in
smollm2_tokenizer.py and chatml.py, sampling in sampler.py (libsampler.so).

prepare()  (no FPGA)  messages -> ChatML prompt ids (SmolLM2's template: the
           default system prompt when the conversation has none; a
           "developer" message counts as "system"), history trimmed to leave
           `reserve` tokens of the context for the answer (the oldest turns
           go first; the system message and the last message always stay;
           400 context_length_exceeded if they alone do not fit), sampling
           parameters validated.
generate() (FPGA lock) prefix-cache reuse: the token list currently in the
           KV cache is kept between requests; the library is truncated to the
           common prefix with the new prompt and only the new tokens are
           prefilled (a multi-turn chat prefills just the new turn).  Position
           0 is the precomputed <|im_start|> sink: the prompt's leading
           <|im_start|> is never passed in.  Then decode + sample one token at
           a time until an end-of-generation token (<|im_end|>, <|endoftext|>,
           <|im_start|>), a stop string, max_tokens or a full context; every
           token's text is streamed as a Delta (UTF-8 completed across tokens,
           stop strings held back until they cannot match).

Parameters: temperature (0 = greedy), top_p, seed, stop, max_tokens,
presence_penalty, frequency_penalty (OpenAI fields), and the extra fields
top_k and repetition_penalty (HF semantics) and repeat_last_n (the penalty
window: 64 recent tokens of prompt + answer; 0 = off, -1 = everything), and
the DRY fields dry_multiplier, dry_base, dry_allowed_length,
dry_penalty_last_n (the DRY window over prompt + answer; -1 = the whole
context, 0 = off) and dry_sequence_breakers (strings; a token containing any
of them cuts a repeat match; special tokens always do).  Unset ones take the
server defaults (the model card's temperature 0.2, top_p 0.9; top_k 50 as HF
generate; DRY on at the usual 0.8 / 1.75 / 2 over the whole context with
breakers ":", "\"", "*"; repetition_penalty 1.1 over the last 64 tokens;
no presence / frequency penalty).  A loop guard (extra field `loop_guard`,
default true) ends the answer with finish_reason "stop" (kv260.finish
"loop") when the generated tokens end in a block repeated verbatim
(3 times, >= 24 tokens) — the last resort if sampling still loops.  Without a seed a random
one is drawn and returned in kv260.seed, so any answer can be reproduced.

usage.prompt_tokens = prompt ids incl. the leading <|im_start|>;
completion_tokens = generated tokens (the end-of-generation token not
counted).  The non-standard "kv260" object: cached / prefilled tokens,
prefill_ms, ttft_ms, decode tokens and tok/s, finish detail (eos,
stop_string, max_tokens, context_full), trimmed messages, sampler settings.
"""

from __future__ import annotations

import ctypes
import os
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import chatml
from chat_backend import (Backend, BackendError, Cancelled, CancelToken, ChatRequest, Delta,
                          Finish, StopStream)
from sampler import SamplerParams, make_sampler

# DRY sequence breakers.  text-generation-webui / llama.cpp also break at "\n",
# but then a loop of short lines ("The cat loves her adventures\n\nThis is a
# short story.\n\n" ...) never builds a run longer than a line and slips
# through (CHAT_PLAN §15); without it, lists and code were unchanged on the
# board because ':' / '"' / '*' still cut their structure.
DRY_BREAKERS = (":", "\"", "*")

# loop guard: stop when the last max(LOOP_REPEATS * p, LOOP_MIN_TOKENS)
# generated tokens repeat one block of p tokens (1 <= p <= LOOP_MAX_PERIOD)
LOOP_REPEATS = 3
LOOP_MIN_TOKENS = 24
LOOP_MAX_PERIOD = 128


def loop_period(gen: Sequence[int]) -> int:
    """The period p of a verbatim loop at the end of gen, or 0."""
    n = len(gen)
    if n < LOOP_MIN_TOKENS:
        return 0
    last = gen[-1]
    for p in range(1, min(LOOP_MAX_PERIOD, n // LOOP_REPEATS) + 1):
        if gen[-1 - p] != last:                            # cheap necessary condition
            continue
        span = max(LOOP_REPEATS * p, LOOP_MIN_TOKENS)
        if span > n:
            break
        tail = gen[-span:]
        if tail[p:] == tail[:-p]:
            return p
    return 0
from smollm2_tokenizer import Tokenizer

MODEL_ID = "smollm2-135m-instruct"
EOG_TOKENS = ("<|im_end|>", "<|endoftext|>", "<|im_start|>")


class LlmLibraryError(RuntimeError):
    pass


class LibLlmEngine:
    """libsmollm2.so through ctypes — exactly the §11 contract:

        int llm_open(const char *weights_dir);  void llm_close(void);
        const char *llm_last_error(void);
        int llm_vocab_size(void);  int llm_context_size(void);  int llm_position(void);
        int llm_truncate(int n);
        int llm_prefill(const int32_t *tokens, int n, float *logits);
        int llm_decode(int32_t token, float *logits);

    Every int function returns >= 0 on success; a negative rc raises
    LlmLibraryError with llm_last_error()."""

    def __init__(self, lib_path: str, weights_dir: Optional[str] = None):
        self.lib_path = lib_path
        self.weights_dir = weights_dir
        self.lib = None
        self.vocab_size = 0
        self.context_size = 0
        self.logits = None

    def _bind(self):
        lib = ctypes.CDLL(os.path.abspath(self.lib_path))
        i32p, fp = ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_float)
        sig = {"llm_open": ([ctypes.c_char_p], ctypes.c_int), "llm_close": ([], None),
               "llm_last_error": ([], ctypes.c_char_p), "llm_vocab_size": ([], ctypes.c_int),
               "llm_context_size": ([], ctypes.c_int), "llm_position": ([], ctypes.c_int),
               "llm_truncate": ([ctypes.c_int], ctypes.c_int),
               "llm_prefill": ([i32p, ctypes.c_int, fp], ctypes.c_int),
               "llm_decode": ([ctypes.c_int32, fp], ctypes.c_int)}
        for name, (args, res) in sig.items():
            f = getattr(lib, name)
            f.argtypes, f.restype = args, res
        return lib

    def _err(self, what: str, rc: int) -> LlmLibraryError:
        msg = self.lib.llm_last_error() if self.lib is not None else None
        return LlmLibraryError(f"{what}: {(msg or b'').decode('utf-8', 'replace') or 'error'} (rc {rc})")

    def open(self) -> None:
        if self.lib is not None:
            return
        lib = self._bind()
        self.lib = lib
        rc = lib.llm_open(self.weights_dir.encode() if self.weights_dir else None)
        if rc < 0:
            err = self._err("llm_open", rc)
            self.lib = None
            raise err
        self.vocab_size = int(lib.llm_vocab_size())
        self.context_size = int(lib.llm_context_size())
        self.logits = (ctypes.c_float * self.vocab_size)()

    def close(self) -> None:
        if self.lib is not None:
            self.lib.llm_close()
            self.lib = None

    @property
    def is_open(self) -> bool:
        return self.lib is not None

    def info(self) -> Dict[str, Any]:
        return {"library": self.lib_path, "weights_dir": self.weights_dir,
                "vocab_size": self.vocab_size, "context_size": self.context_size}

    def position(self) -> int:
        rc = self.lib.llm_position()
        if rc < 0:
            raise self._err("llm_position", rc)
        return rc

    def truncate(self, n: int) -> None:
        rc = self.lib.llm_truncate(n)
        if rc < 0:
            raise self._err(f"llm_truncate({n})", rc)

    def prefill(self, tokens: Sequence[int]):
        buf = (ctypes.c_int32 * len(tokens))(*tokens)
        rc = self.lib.llm_prefill(buf, len(tokens), self.logits)
        if rc < 0:
            raise self._err(f"llm_prefill({len(tokens)} tokens)", rc)
        return self.logits

    def decode(self, token: int):
        rc = self.lib.llm_decode(token, self.logits)
        if rc < 0:
            raise self._err(f"llm_decode({token})", rc)
        return self.logits


@dataclass
class Job:
    req: ChatRequest
    ids: List[int]
    params: SamplerParams
    seed: int
    max_tokens: Optional[int]
    stops: List[str]
    last_n: int
    dry_last_n: int = -1
    breakers: Optional[Tuple[str, ...]] = None       # None: the backend's defaults
    loop_guard: bool = True
    dropped: int = 0
    prepare_ms: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)


def _common_prefix(a: Sequence[int], b: Sequence[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


class Smollm2Backend(Backend):
    model_id = MODEL_ID
    owned_by = "kv260"
    fingerprint = "kv260-smollm2-135m-pow2+sink+p12"
    cma_mb = 360.0                      # CMA the loaded model holds (estimate, §10.6)

    def __init__(self, engine, tokenizer_path: str, *, sampler_lib: Optional[str] = None,
                 defaults: Optional[SamplerParams] = None, context_size: int = 1024,
                 reserve: int = 256, repeat_last_n: int = 64, prefill_chunk: int = 0,
                 cma_mb: Optional[float] = None, dry_penalty_last_n: int = -1,
                 dry_sequence_breakers: Sequence[str] = DRY_BREAKERS, loop_guard: bool = True):
        self.engine = engine
        self.tokenizer_path = tokenizer_path
        self.sampler_lib = sampler_lib
        self.defaults = defaults or SamplerParams(temperature=0.2, top_p=0.9, top_k=50,
                                                  repetition_penalty=1.1, dry_multiplier=0.8)
        self.loop_guard = loop_guard
        self.dry_last_n = dry_penalty_last_n
        self.dry_breakers = tuple(dry_sequence_breakers)
        self._breaker_ids: Dict[Tuple[str, ...], List[int]] = {}
        self._breakers_set: Optional[Tuple[str, ...]] = None      # what the sampler holds
        self.ctx = context_size
        self.reserve = reserve
        self.repeat_last_n = repeat_last_n
        self.prefill_chunk = prefill_chunk
        if cma_mb is not None:
            self.cma_mb = cma_mb
        self.tok: Optional[Tokenizer] = None
        self.builder: Optional[chatml.PromptBuilder] = None
        self.sampler = None
        self._prep_lock = threading.Lock()
        self._cached: Optional[List[int]] = None     # tokens at KV positions 1.. (after the sink)
        self.stats = {"requests": 0, "prompt_tokens": 0, "reused_tokens": 0, "prefilled_tokens": 0,
                      "generated_tokens": 0}

    # ── lifecycle ──

    def load_host(self) -> None:
        if self.tok is not None:
            return
        t0 = time.monotonic()
        self.tok = Tokenizer(self.tokenizer_path)
        self.builder = chatml.PromptBuilder(self.tok.encode)
        self.sampler = make_sampler(self.tok.vocab_size, self.sampler_lib)
        self.eog = frozenset(self.tok.special[t] for t in EOG_TOKENS if t in self.tok.special)
        self.bos = self.tok.special["<|im_start|>"]
        self._set_breakers(self.dry_breakers)
        self.host_load_s = time.monotonic() - t0

    def breaker_ids(self, breakers: Tuple[str, ...]) -> List[int]:
        """Token ids that cut a DRY match: every special token, and every token
        whose bytes contain one of the breaker strings (byte-level BPE merges
        "\n" into many tokens, so a per-token substring test, not one id)."""
        ids = self._breaker_ids.get(breakers)
        if ids is None:
            pats = [b.encode("utf-8") for b in breakers if b]
            ids = [i for i in range(self.tok.vocab_size)
                   if self.tok.is_special(i) or any(pt in self.tok.token_bytes(i) for pt in pats)]
            if len(self._breaker_ids) > 8:
                self._breaker_ids.clear()
            self._breaker_ids[breakers] = ids
        return ids

    def _set_breakers(self, breakers: Tuple[str, ...]) -> None:
        if breakers != self._breakers_set:
            self.sampler.set_breakers(self.breaker_ids(breakers))
            self._breakers_set = breakers

    def load(self) -> None:
        self.load_host()
        self.engine.open()
        if self.engine.vocab_size != self.tok.vocab_size:
            v = self.engine.vocab_size
            self.engine.close()
            raise RuntimeError(f"the library's vocabulary ({v}) is not the tokenizer's "
                               f"({self.tok.vocab_size})")
        self.ctx = self.engine.context_size
        self._cached = None

    def unload(self) -> None:
        self._cached = None
        self.engine.close()

    def close(self) -> None:
        self.unload()

    def health(self) -> Dict[str, Any]:
        info = self.engine.info() if hasattr(self.engine, "info") else {}
        return dict(info, context_size=self.ctx, reserve=self.reserve,
                    sampler=getattr(self.sampler, "kind", None),
                    cached_tokens=(len(self._cached) + 1) if self._cached is not None else 0,
                    defaults=self.defaults.as_dict(), **self.stats)

    # ── request -> prompt ids (no FPGA) ──

    @staticmethod
    def _extra(raw: dict, key: str, lo: float, hi: Optional[float], integer: bool = False):
        v = raw.get(key)
        if v is None:
            return None
        if isinstance(v, bool) or not isinstance(v, (int, float)) or (integer and not isinstance(v, int)):
            raise BackendError(f"Invalid type for '{key}': expected {'an integer' if integer else 'a number'}.",
                               key, code="invalid_type")
        if v < lo or (hi is not None and v > hi):
            raise BackendError(f"Invalid '{key}': expected a value in [{lo}, {hi if hi is not None else 'inf'}],"
                               f" but got {v} instead.", key, code="invalid_value")
        return v

    def sampler_params(self, req: ChatRequest) -> SamplerParams:
        d, raw = self.defaults, req.raw or {}

        def pick(v, default):
            return default if v is None else v
        top_k = self._extra(raw, "top_k", 0, None, integer=True)
        rep = self._extra(raw, "repetition_penalty", 0.01, 10.0)
        pres = self._extra(raw, "presence_penalty", -2.0, 2.0)
        freq = self._extra(raw, "frequency_penalty", -2.0, 2.0)
        dmul = self._extra(raw, "dry_multiplier", 0.0, 10.0)
        dbase = self._extra(raw, "dry_base", 1.0, 10.0)
        dlen = self._extra(raw, "dry_allowed_length", 1, 64, integer=True)
        return SamplerParams(temperature=float(pick(req.temperature, d.temperature)),
                             top_p=float(pick(req.top_p, d.top_p)),
                             top_k=int(pick(top_k, d.top_k)),
                             repetition_penalty=float(pick(rep, d.repetition_penalty)),
                             presence_penalty=float(pick(pres, d.presence_penalty)),
                             frequency_penalty=float(pick(freq, d.frequency_penalty)),
                             dry_multiplier=float(pick(dmul, d.dry_multiplier)),
                             dry_base=float(pick(dbase, d.dry_base)),
                             dry_allowed_length=int(pick(dlen, d.dry_allowed_length)),
                             dry_max_match=d.dry_max_match)

    @staticmethod
    def _breakers_field(raw: dict) -> Optional[Tuple[str, ...]]:
        v = raw.get("dry_sequence_breakers")
        if v is None:
            return None
        if (not isinstance(v, list) or len(v) > 32
                or any(not isinstance(x, str) or not x or len(x) > 32 for x in v)):
            raise BackendError("Invalid 'dry_sequence_breakers': expected a list of at most 32 "
                               "non-empty strings of at most 32 characters.", "dry_sequence_breakers",
                               code="invalid_value")
        return tuple(v)

    def prepare(self, req: ChatRequest) -> Job:
        t0 = time.monotonic()
        if self.tok is None:
            self.load_host()
        params = self.sampler_params(req)
        last_n = self._extra(req.raw or {}, "repeat_last_n", -1, None, integer=True)
        last_n = self.repeat_last_n if last_n is None else last_n
        dry_n = self._extra(req.raw or {}, "dry_penalty_last_n", -1, None, integer=True)
        dry_n = self.dry_last_n if dry_n is None else dry_n
        breakers = self._breakers_field(req.raw or {})
        lg = (req.raw or {}).get("loop_guard")
        if lg is not None and not isinstance(lg, bool):
            raise BackendError("Invalid type for 'loop_guard': expected a boolean.", "loop_guard",
                               code="invalid_type")
        loop_guard = self.loop_guard if lg is None else lg
        msgs = [{"role": "system" if m["role"] == "developer" else m["role"], "content": m["content"]}
                for m in req.messages]
        reserve = min(req.max_tokens, self.reserve) if req.max_tokens else self.reserve
        budget = self.ctx - reserve
        with self._prep_lock:
            try:
                ids, _, dropped = self.builder.fit(msgs, budget)
            except chatml.ContextTooLong as e:
                raise BackendError(f"{e} This model's context is {self.ctx} tokens.", "messages",
                                   code="context_length_exceeded") from None
        if not ids or ids[0] != self.bos or len(ids) < 2:
            raise BackendError("internal: the chat template must start with <|im_start|>", "messages")
        seed = req.seed if req.seed is not None else random.SystemRandom().getrandbits(63)
        return Job(req, ids, params, seed, req.max_tokens, list(req.stop), last_n, dry_n,
                   breakers, loop_guard, dropped, (time.monotonic() - t0) * 1000.0)

    # ── prompt -> tokens (FPGA) ──

    def _prefill(self, tokens: List[int], cancel: CancelToken):
        eng, chunk = self.engine, self.prefill_chunk
        logits = None
        pos = 0
        while pos < len(tokens):
            n = len(tokens) - pos if chunk <= 0 else min(chunk, len(tokens) - pos)
            if pos:
                cancel.check()
            logits = eng.prefill(tokens[pos:pos + n])
            self._cached.extend(tokens[pos:pos + n])
            pos += n
        return logits

    def generate(self, job: Job, cancel: CancelToken):
        eng, tok, smp, p = self.engine, self.tok, self.sampler, job.params
        t0 = time.monotonic()
        ctx = eng.context_size or self.ctx
        prompt = job.ids
        if len(prompt) > ctx:
            raise BackendError(f"The prompt needs {len(prompt)} tokens; this model's context is {ctx}.",
                               "messages", code="context_length_exceeded")
        body = prompt[1:]                                   # position 0 is the sink
        cancel.check()
        cached = self._cached
        try:
            if cached is None or eng.position() != 1 + len(cached):
                cached = []
            reuse = min(_common_prefix(cached, body), len(body) - 1)
            self._cached = None                             # unknown until the calls succeed
            eng.truncate(1 + reuse)
            self._cached = body[:reuse]
            new = body[reuse:]
            logits = self._prefill(new, cancel)
        except Cancelled:                                   # between calls: the cache is consistent
            raise
        except Exception:                                   # the library's state is unknown
            self._cached = None
            raise
        t_prefill = time.monotonic()
        prefill_ms = (t_prefill - t0) * 1000.0
        smp.seed(job.seed)
        self._set_breakers(job.breakers if job.breakers is not None else self.dry_breakers)
        history = list(prompt)
        last_n = job.last_n
        dry_n = job.dry_last_n
        dec = tok.incremental_decoder()
        ss = StopStream(job.stops)
        n_gen, n_decode, lib_ms, smp_ms = 0, 0, 0.0, 0.0
        loop_p = 0
        t_first: Optional[float] = None
        finish = None
        while True:
            ts = time.monotonic()
            recent = history if last_n < 0 else (history[-last_n:] if last_n > 0 else ())
            hist = history if dry_n < 0 else (history[-dry_n:] if dry_n > 0 else ())
            t = smp.sample(logits, recent, p, hist=hist)
            smp_ms += (time.monotonic() - ts) * 1000.0
            if t_first is None:
                t_first = time.monotonic()
            if t in self.eog:
                finish = "eos"
                break
            n_gen += 1
            history.append(t)
            piece = dec.add(t)
            if piece:
                out, stopped = ss.feed(piece)
                if out:
                    yield Delta(out)
                if stopped:
                    finish = "stop_string"
                    break
            if job.loop_guard:
                per = loop_period(history[len(prompt):])
                if per:
                    finish, loop_p = "loop", per
                    break
            if job.max_tokens is not None and n_gen >= job.max_tokens:
                finish = "max_tokens"
                break
            if 1 + len(self._cached) >= ctx:
                finish = "context_full"
                break
            cancel.check()
            tl = time.monotonic()
            cached_now, self._cached = self._cached, None
            logits = eng.decode(t)
            cached_now.append(t)
            self._cached = cached_now
            lib_ms += (time.monotonic() - tl) * 1000.0
            n_decode += 1
        t_end = time.monotonic()
        if finish != "stop_string":
            tail, stopped = ss.feed(dec.flush())
            if stopped:
                finish = "stop_string"
            else:
                tail += ss.flush()
            if tail:
                yield Delta(tail)
        reason = "length" if finish in ("max_tokens", "context_full") else "stop"
        dec_s = t_end - (t_first or t_end)
        rate = n_decode / dec_s if dec_s > 0 else 0.0
        st = self.stats
        st["requests"] += 1
        st["prompt_tokens"] += len(prompt)
        st["reused_tokens"] += reuse + 1
        st["prefilled_tokens"] += len(new)
        st["generated_tokens"] += n_gen
        info = {"backend": self.model_id, "finish": finish, "cached_tokens": reuse + 1,
                "prefill_tokens": len(new), "prefill_ms": round(prefill_ms, 1),
                "ttft_ms": round(((t_first or t_end) - t0) * 1000.0, 1),
                "decode_tokens": n_decode, "decode_ms": round(dec_s * 1000.0, 1),
                "decode_tok_s": round(rate, 2), "library_decode_ms": round(lib_ms, 1),
                "sampler_ms": round(smp_ms, 1), "prepare_ms": round(job.prepare_ms, 1),
                "trimmed_messages": job.dropped, "context_size": ctx, "seed": job.seed,
                "loop_period": loop_p, "loop_guard": job.loop_guard,
                "sampler": dict(p.as_dict(), repeat_last_n=last_n, dry_penalty_last_n=dry_n,
                                dry_sequence_breakers=list(self._breakers_set or ()),
                                impl=getattr(smp, "kind", "?")),
                "log": {"reuse": f"{reuse + 1}/{len(prompt)}", "prefill": f"{len(new)}tok/{prefill_ms:.0f}ms",
                        "decode": f"{rate:.1f}tok/s", "why": finish}}
        if job.dropped:
            info["log"]["trimmed"] = job.dropped
        yield Finish(reason, len(prompt), n_gen, info)
