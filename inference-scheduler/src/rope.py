# Copyright 2026 Ivan Gubochkin
# SPDX-License-Identifier: Apache-2.0

"""
Rotary position embeddings of a Llama-family block: the inverse frequencies
of every scaling Hugging Face implements for these models (linear, llama3,
dynamic, yarn, longrope), partial rotation, and the tables the frontend
(src/llama.py) and the study (demo/chat/scripts/llm_study.py, which loads
this file by path) both use — numpy and the standard library only
(doc/plans/CHAT_PLAN.md §29).

The datapath rotates head dims (j, j + HD/2) of every head: rotate-half over
the whole head.  A model that rotates only its first ``rot`` dims (partial
rotation: rotate-half within them, the rest passed through) is mapped onto it
by permuting the head dims of q and k (``perm``: the same permutation on both
leaves q·k unchanged) and giving the pass-through pairs cos 1, sin 0; an
interleaved rotation (pairs (2j, 2j + 1), GPT-J style) likewise — src/moonshine.py
maps Moonshine's interleaved partial rotation so, its heads padded to a
multiple of 16 with more pass-through pairs (``half_tables`` of a wider head).  An
attention factor (yarn, longrope) scales cos and sin of the rotated dims,
i.e. the rotated part of q·k by factor²: llm_weights folds factor² into
those rows of W_q, so the tables keep |cos|, |sin| <= 1 (the row-ops unit's
int24 tables).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

SCALINGS = ("linear", "llama3", "dynamic", "yarn", "longrope")


@dataclass(frozen=True)
class RopeSpec:
    hd: int                          # head dim
    theta: float
    rot: int = 0                     # rotated dims per head (0: hd)
    scaling: Optional[dict] = field(default=None, compare=False)   # rope_type + its parameters
    max_pos: int = 0                 # max_position_embeddings
    interleaved: bool = False        # rotated pairs (2j, 2j + 1), not (j, j + rot/2)

    @property
    def dims(self) -> int:
        return self.rot or self.hd


def inv_freq(spec: RopeSpec) -> np.ndarray:
    """float32 [dims/2]: Hugging Face's inverse frequencies (computed in float64,
    rounded to float32; the unscaled ones exactly as src/llama.py always did)."""
    dim = spec.dims
    base = np.array([1.0 / (spec.theta ** (2 * j / dim)) for j in range(dim // 2)], np.float64)
    sc = spec.scaling
    kind = (sc or {}).get("rope_type") or (sc or {}).get("type")
    if kind in (None, "default", "dynamic"):
        # dynamic NTK changes nothing up to max_position_embeddings (the context is shorter)
        return base.astype(np.float32)
    if kind == "linear":
        return (base / float(sc["factor"])).astype(np.float32)
    if kind == "llama3":
        factor, lo, hi = float(sc["factor"]), float(sc["low_freq_factor"]), float(sc["high_freq_factor"])
        old = float(sc["original_max_position_embeddings"])
        wavelen = 2 * math.pi / base
        inv = np.where(wavelen > old / lo, base / factor, base)
        smooth = (old / wavelen - lo) / (hi - lo)
        smoothed = (1 - smooth) * inv / factor + smooth * inv
        medium = ~(wavelen < old / hi) & ~(wavelen > old / lo)
        return np.where(medium, smoothed, inv).astype(np.float32)
    if kind == "yarn":
        factor = sc.get("factor") or spec.max_pos / sc["original_max_position_embeddings"]
        orig = float(sc["original_max_position_embeddings"])
        beta_fast, beta_slow = sc.get("beta_fast") or 32, sc.get("beta_slow") or 1

        def corr(rot):
            return dim * math.log(orig / (rot * 2 * math.pi)) / (2 * math.log(spec.theta))
        lo, hi = corr(beta_fast), corr(beta_slow)
        if sc.get("truncate", True):
            lo, hi = math.floor(lo), math.ceil(hi)
        lo, hi = max(lo, 0), min(hi, dim - 1)
        if lo == hi:
            hi += 0.001
        ramp = np.clip((np.arange(dim // 2, dtype=np.float64) - lo) / (hi - lo), 0, 1)
        extra = 1 - ramp
        return (base / factor * (1 - extra) + base * extra).astype(np.float32)
    if kind == "longrope":
        # the short factors: the context is at most the original length
        f = np.asarray(sc["short_factor"], np.float64)
        return (base / f).astype(np.float32)
    raise ValueError(f"rope_scaling {kind!r}: not implemented ({', '.join(SCALINGS)})")


def attention_factor(spec: RopeSpec) -> float:
    """The factor yarn / longrope put on cos and sin of the rotated dims (1 for the others)."""
    sc = spec.scaling or {}
    kind = sc.get("rope_type") or sc.get("type")
    if kind not in ("yarn", "longrope"):
        return 1.0
    if sc.get("attention_factor") is not None:
        return float(sc["attention_factor"])
    orig = float(sc["original_max_position_embeddings"])
    factor = sc.get("factor") or spec.max_pos / orig
    if kind == "yarn":
        def mscale(scale, m=1.0):
            return 1.0 if scale <= 1 else 0.1 * m * math.log(scale) + 1.0
        if sc.get("mscale") and sc.get("mscale_all_dim"):
            return float(mscale(factor, sc["mscale"]) / mscale(factor, sc["mscale_all_dim"]))
        return mscale(factor)
    return 1.0 if factor <= 1.0 else math.sqrt(1 + math.log(factor) / math.log(orig))


def perm(spec: RopeSpec) -> np.ndarray:
    """[hd] head-dim order of q / k rows: rotated pair j (dims j, j + rot/2, or
    2j, 2j + 1 when interleaved) at (j, j + hd/2), the pass-through dims
    (rot..hd) in the remaining places (the identity for a full rotate-half)."""
    hd, rot = spec.hd, spec.dims
    half, npair = hd // 2, rot // 2
    order = np.empty(hd, np.int64)
    if spec.interleaved:
        order[:npair] = np.arange(0, rot, 2)
        order[half:half + npair] = np.arange(1, rot, 2)
    else:
        order[:npair] = np.arange(npair)
        order[half:half + npair] = np.arange(npair, rot)
    rest = np.arange(rot, hd)
    order[npair:half] = rest[:half - npair]
    order[half + npair:] = rest[half - npair:]
    return order


def half_tables(spec: RopeSpec, n: int):
    """float32 cos / sin [n][hd/2] of the pairs in perm's order:
    a = float32(float32(pos) * inv_j), cos = float32(cos(double(a))) (libm);
    the pass-through pairs cos 1, sin 0."""
    inv = inv_freq(spec)
    ang = (np.arange(n, dtype=np.float32)[:, None] * inv[None, :]).astype(np.float32)
    c = np.vectorize(math.cos, otypes=[np.float64])(ang.astype(np.float64)).astype(np.float32)
    s = np.vectorize(math.sin, otypes=[np.float64])(ang.astype(np.float64)).astype(np.float32)
    half = spec.hd // 2
    if c.shape[1] == half:
        return c, s
    cc, ss = np.ones((n, half), np.float32), np.zeros((n, half), np.float32)
    cc[:, :c.shape[1]], ss[:, :s.shape[1]] = c, s
    return cc, ss


__all__ = ("RopeSpec", "SCALINGS", "attention_factor", "half_tables", "inv_freq", "perm")
