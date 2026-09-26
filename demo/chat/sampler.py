"""sampler.py — next-token sampling for the KV260 chat server (standard
library only).

`CSampler` wraps libsampler.so (demo/chat/src/sampler.c, built on the board
by deploy.py: `cc -O2 -shared -fPIC -o lib/libsampler.so src/sampler.c -lm`);
`PySampler` is the same algorithm in pure Python — identical tokens for the
same logits, parameters and seed (both compute in double in the same order;
same libm), ~50x slower, for hosts without a compiler.  `make_sampler()`
picks the C one when the library loads.

The pipeline and its conventions are documented in src/sampler.h:
penalties over the recent tokens (repetition: HF / CTRL style; presence /
frequency: OpenAI style) -> DRY over the ordered history (sequence breakers
cut matches) -> greedy if temperature <= 0 or top_k == 1 -> temperature ->
top-k -> softmax -> top-p -> draw with a splitmix64 RNG.
"""

from __future__ import annotations

import ctypes
import math
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
MASK64 = (1 << 64) - 1


@dataclass
class SamplerParams:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    dry_multiplier: float = 0.0          # 0: DRY off
    dry_base: float = 1.75
    dry_allowed_length: int = 2
    dry_max_match: int = 50

    @property
    def greedy(self) -> bool:
        return not self.temperature > 0.0 or self.top_k == 1

    def as_dict(self) -> dict:
        return {"temperature": self.temperature, "top_p": self.top_p, "top_k": self.top_k,
                "repetition_penalty": self.repetition_penalty,
                "presence_penalty": self.presence_penalty,
                "frequency_penalty": self.frequency_penalty,
                "dry_multiplier": self.dry_multiplier, "dry_base": self.dry_base,
                "dry_allowed_length": self.dry_allowed_length}


class _CParams(ctypes.Structure):
    _fields_ = [("temperature", ctypes.c_double), ("top_p", ctypes.c_double),
                ("repetition_penalty", ctypes.c_double), ("presence_penalty", ctypes.c_double),
                ("frequency_penalty", ctypes.c_double), ("top_k", ctypes.c_int32),
                ("reserved", ctypes.c_int32), ("dry_multiplier", ctypes.c_double),
                ("dry_base", ctypes.c_double), ("dry_allowed_length", ctypes.c_int32),
                ("dry_max_match", ctypes.c_int32)]


def _cparams(p: SamplerParams) -> _CParams:
    return _CParams(float(p.temperature), float(p.top_p), float(p.repetition_penalty),
                    float(p.presence_penalty), float(p.frequency_penalty), int(p.top_k), 0,
                    float(p.dry_multiplier), float(p.dry_base), int(p.dry_allowed_length),
                    int(p.dry_max_match))


def _i32(seq: Sequence[int]):
    return (ctypes.c_int32 * len(seq))(*seq)


class CSampler:
    """libsampler.so through ctypes (one state: RNG + scratch buffers)."""

    kind = "c"

    def __init__(self, lib_path: str, n_vocab: int):
        lib = ctypes.CDLL(os.path.abspath(lib_path))
        pf, pi = ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int32)
        lib.smp_new.argtypes = [ctypes.c_int32]
        lib.smp_new.restype = ctypes.c_void_p
        lib.smp_free.argtypes = [ctypes.c_void_p]
        lib.smp_free.restype = None
        lib.smp_seed.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        lib.smp_seed.restype = None
        lib.smp_rng_next.argtypes = [ctypes.c_void_p]
        lib.smp_rng_next.restype = ctypes.c_uint64
        lib.smp_rng_uniform.argtypes = [ctypes.c_void_p]
        lib.smp_rng_uniform.restype = ctypes.c_double
        lib.smp_sample_ex.argtypes = [ctypes.c_void_p, pf, pi, ctypes.c_int32, pi, ctypes.c_int32,
                                      ctypes.POINTER(_CParams)]
        lib.smp_sample_ex.restype = ctypes.c_int32
        lib.smp_candidates_ex.argtypes = [ctypes.c_void_p, pf, pi, ctypes.c_int32, pi, ctypes.c_int32,
                                          ctypes.POINTER(_CParams), pi,
                                          ctypes.POINTER(ctypes.c_double), ctypes.c_int32]
        lib.smp_candidates_ex.restype = ctypes.c_int32
        lib.smp_set_breakers.argtypes = [ctypes.c_void_p, pi, ctypes.c_int32]
        lib.smp_set_breakers.restype = ctypes.c_int32
        lib.smp_version.restype = ctypes.c_char_p
        self.lib, self.n, self.path = lib, n_vocab, lib_path
        self.h = lib.smp_new(n_vocab)
        if not self.h:
            raise MemoryError("smp_new failed")
        self.version = lib.smp_version().decode()

    def __del__(self):
        h, self.h = getattr(self, "h", None), None
        if h:
            self.lib.smp_free(h)

    def _logits(self, logits):
        if isinstance(logits, ctypes.Array):
            if len(logits) != self.n:
                raise ValueError(f"expected {self.n} logits, got {len(logits)}")
            return ctypes.cast(logits, ctypes.POINTER(ctypes.c_float))
        try:                                            # array('f'), numpy float32, ...
            buf = (ctypes.c_float * self.n).from_buffer(logits)
        except (TypeError, ValueError):
            buf = (ctypes.c_float * self.n)(*logits)
        return ctypes.cast(buf, ctypes.POINTER(ctypes.c_float))

    def seed(self, seed: int) -> None:
        self.lib.smp_seed(self.h, seed & MASK64)

    def rng_next(self) -> int:
        return self.lib.smp_rng_next(self.h)

    def uniform(self) -> float:
        return self.lib.smp_rng_uniform(self.h)

    def set_breakers(self, ids: Sequence[int]) -> int:
        """DRY sequence-breaker token ids (replaces the previous set)."""
        k = self.lib.smp_set_breakers(self.h, _i32(ids), len(ids))
        if k < 0:
            raise ValueError("smp_set_breakers: bad arguments")
        return k

    def sample(self, logits, recent: Sequence[int], p: SamplerParams,
               hist: Optional[Sequence[int]] = None) -> int:
        """hist: the ordered token history for DRY (default: recent)."""
        rec = _i32(recent)
        h = rec if hist is None else _i32(hist)
        t = self.lib.smp_sample_ex(self.h, self._logits(logits), rec, len(recent), h,
                                   len(h), ctypes.byref(_cparams(p)))
        if t < 0:
            raise ValueError("smp_sample: bad arguments")
        return t

    def candidates(self, logits, recent: Sequence[int], p: SamplerParams,
                   max_out: Optional[int] = None,
                   hist: Optional[Sequence[int]] = None) -> List[Tuple[int, float]]:
        m = self.n if max_out is None else max_out
        ids, probs = (ctypes.c_int32 * m)(), (ctypes.c_double * m)()
        rec = _i32(recent)
        h = rec if hist is None else _i32(hist)
        k = self.lib.smp_candidates_ex(self.h, self._logits(logits), rec, len(recent), h,
                                       len(h), ctypes.byref(_cparams(p)), ids, probs, m)
        if k < 0:
            raise ValueError("smp_candidates: bad arguments")
        return [(ids[i], probs[i]) for i in range(min(k, m))]


class PySampler:
    """The same algorithm in pure Python (see src/sampler.h)."""

    kind = "python"
    version = "kv260-sampler 2 (dry) (python)"

    def __init__(self, n_vocab: int):
        self.n = n_vocab
        self.state = 0
        self.breakers = frozenset()

    def set_breakers(self, ids: Sequence[int]) -> int:
        self.breakers = frozenset(i for i in ids if 0 <= i < self.n)
        return len(self.breakers)

    def _dry(self, l: list, hist: Sequence[int], p: SamplerParams) -> None:
        """step 1b (src/sampler.h), in place."""
        n_h = len(hist)
        if not p.dry_multiplier > 0.0 or n_h < 2:
            return
        n, brk = self.n, self.breakers
        last = hist[-1]
        if not 0 <= last < n or last in brk:
            return
        maxm = p.dry_max_match if p.dry_max_match > 0 else 50
        al = p.dry_allowed_length if p.dry_allowed_length > 0 else 1
        ml = {}                                          # first-seen order, as in C
        for i in range(n_h - 1):
            if hist[i] != last:
                continue
            nx = hist[i + 1]
            if not 0 <= nx < n or nx in brk:
                continue
            ln = 1
            while ln < maxm:
                j = i - ln
                if j < 0:
                    break
                prev = hist[n_h - 1 - ln]
                if hist[j] != prev or not 0 <= prev < n or prev in brk:
                    break
                ln += 1
            if ln > ml.get(nx, 0):
                ml[nx] = ln
        for t, ln in ml.items():
            if ln >= al:
                l[t] = l[t] - p.dry_multiplier * math.pow(p.dry_base, float(ln - al))

    def seed(self, seed: int) -> None:
        self.state = seed & MASK64

    def rng_next(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return z ^ (z >> 31)

    def uniform(self) -> float:
        return (self.rng_next() >> 11) * (1.0 / 9007199254740992.0)

    def _filter(self, logits, recent: Sequence[int], p: SamplerParams,
                hist: Optional[Sequence[int]] = None):
        """-> ('greedy', id) or ('cands', [(id, e)], z)"""
        n = self.n
        try:
            l = memoryview(logits).cast("B").cast("f").tolist()     # ctypes / array('f') / numpy
        except (TypeError, ValueError):
            l = [float(v) for v in logits]
        if any(map(math.isnan, l)):
            l = [-math.inf if v != v else v for v in l]
        if len(l) != n:
            raise ValueError(f"expected {n} logits, got {len(l)}")
        rep = p.repetition_penalty
        use_rep = rep > 0.0 and rep != 1.0
        if use_rep or p.presence_penalty != 0.0 or p.frequency_penalty != 0.0:
            cnt, order = {}, []
            for t in recent:
                if 0 <= t < n:
                    if t not in cnt:
                        cnt[t] = 0
                        order.append(t)
                    cnt[t] += 1
            for t in order:
                v = l[t]
                if use_rep:
                    v = v / rep if v > 0.0 else v * rep
                v -= p.presence_penalty + p.frequency_penalty * float(cnt[t])
                l[t] = v
        self._dry(l, recent if hist is None else hist, p)
        if not p.temperature > 0.0 or p.top_k == 1:
            return "greedy", l.index(max(l))                   # the first maximum
        temp = float(p.temperature)
        l = [v / temp for v in l]
        k = p.top_k
        if 0 < k < n:
            import heapq
            ids = heapq.nlargest(k, range(n), key=l.__getitem__)     # (l desc, id asc)
            is_sorted = True
            m = l[ids[0]]
        else:
            ids = list(range(n))
            is_sorted = False
            m = max(l)
        exp = math.exp
        e = [exp(l[i] - m) for i in ids]
        z = 0.0
        for x in e:
            z += x
        tp = p.top_p
        if 0.0 < tp < 1.0:
            target = tp * z
            if not is_sorted:
                thr = (1.0 - tp) * z / float(n)
                kept = [i for i, x in zip(ids, e) if x >= thr]
                kept.sort(key=lambda i: (-l[i], i))
                e = [exp(l[i] - m) for i in kept]
                s = 0.0
                for x in e:
                    s += x
                if s < target:
                    kept = sorted(range(n), key=lambda i: (-l[i], i))
                    e = [exp(l[i] - m) for i in kept]
                ids = kept
            acc, cut = 0.0, len(ids)
            for j, x in enumerate(e):
                acc += x
                if acc >= target:
                    cut = j + 1
                    break
            ids, e, z = ids[:cut], e[:cut], acc
        return "cands", list(zip(ids, e)), z

    def sample(self, logits, recent: Sequence[int], p: SamplerParams,
               hist: Optional[Sequence[int]] = None) -> int:
        r = self._filter(logits, recent, p, hist)
        if r[0] == "greedy":
            return r[1]
        _, cands, z = r
        target = self.uniform() * z
        acc = 0.0
        for i, x in cands:
            acc += x
            if acc > target:
                return i
        return cands[-1][0]

    def candidates(self, logits, recent: Sequence[int], p: SamplerParams,
                   max_out: Optional[int] = None,
                   hist: Optional[Sequence[int]] = None) -> List[Tuple[int, float]]:
        r = self._filter(logits, recent, p, hist)
        if r[0] == "greedy":
            return [(r[1], 1.0)]
        _, cands, z = r
        return [(i, x / z) for i, x in (cands if max_out is None else cands[:max_out])]


def default_lib_paths() -> List[str]:
    return [os.path.join(HERE, "lib", "libsampler.so"), os.path.join(HERE, "libsampler.so")]


def make_sampler(n_vocab: int, lib_path: Optional[str] = None, allow_python: bool = True):
    """CSampler from lib_path (or lib/libsampler.so next to this file);
    PySampler if that does not load and allow_python."""
    paths = [lib_path] if lib_path else default_lib_paths()
    err = None
    for p in paths:
        if p and os.path.exists(p):
            try:
                return CSampler(p, n_vocab)
            except OSError as e:
                err = e
    if not allow_python:
        raise OSError(f"libsampler.so not loadable from {paths}: {err}")
    return PySampler(n_vocab)


def build_library(out_path: str, cc: str = "cc") -> str:
    """Compile src/sampler.c into out_path (host tests; deploy.py does it on the board)."""
    import subprocess
    src = os.path.join(HERE, "src", "sampler.c")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    subprocess.run([cc, "-O2", "-shared", "-fPIC", "-o", out_path, src, "-lm"], check=True)
    return out_path
