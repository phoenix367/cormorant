"""
VectorOPKernel's softmax — the integer specification (doc/plans/SOFTMAX_PLAN.md
§2.1) that the scheduler's simulation, the model studies, the C++ reference
model and the RTL compute bit for bit.  numpy only (the studies import it).

Per softmax vector of raw int16 scores x (valid length v):

    m   = max_{j<v} x_j
    d_j = m - x_j                                      (0 .. 65535)
    y_j = (d_j * Cm) >> Cs                             Cm * 2^-Cs ~ log2(e) * sigma * 2^-f_s * 2^F
    e_j = TAB[y_j mod 2^F] >> (y_j div 2^F)            TAB[k] = round(2^E * 2^(-k / 2^F))
    S   = sum_{j<v} e_j
    R   = floor(2^RB / S)
    P_j = sat16((e_j * R + 2^(RB - f_p - 1)) >> (RB - f_p));   P_j = 0 for j >= v

Cm is a 24-bit mantissa (2^23 <= Cm < 2^24), Cs a right shift (0 .. 63).
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

F = 12                                   # table index bits (y's fraction)
E = 16                                   # e = 1.0 at 2^E
RB = 40                                  # R = floor(2^RB / S)
TAB = np.round(np.power(2.0, E) * np.power(2.0, -np.arange(1 << F) / (1 << F))).astype(np.int64)
MAX_ROW = 2048                           # row mode: elements buffered per row
MAX_KEYS = 1024                          # column mode: keys per query column


def scale_regs(f_s: float, sigma: float = 1.0) -> Tuple[int, int]:
    """(Cm, Cs) for scores at exponent ``f_s`` with logit scale ``sigma``:
    Cm * 2^-Cs = log2(e) * sigma * 2^-f_s * 2^F rounded to a 24-bit mantissa."""
    c = math.log2(math.e) * float(sigma) * 2.0 ** (-float(f_s)) * (1 << F)
    cs = 23 - math.floor(math.log2(c))
    cm = int(round(c * 2.0 ** cs))
    if cm >= 1 << 24:                    # rounding reached the next power of two
        cm, cs = cm >> 1, cs - 1
    if not (0 <= cs <= 63):
        raise ValueError(f"softmax scale out of range: f_s {f_s}, sigma {sigma} (Cs {cs})")
    return cm, cs


def softmax_raw(x: np.ndarray, valid, cm: int, cs: int, f_p: int) -> np.ndarray:
    """``x``: raw int16 scores [rows][n] (any integer dtype); ``valid``: the
    valid length per row (scalar or [rows]); returns raw P [rows][n] (int64)
    at 2^-f_p.  Rows of valid length 0 give zeros."""
    x = np.asarray(x, np.int64)
    rows, n = x.shape
    v = np.broadcast_to(np.minimum(np.asarray(valid, np.int64), n), (rows,))
    mask = np.arange(n)[None, :] < v[:, None]
    m = np.where(mask, x, np.iinfo(np.int64).min).max(axis=1, keepdims=True)
    d = np.where(mask, m - x, 0)
    y = (d * int(cm)) >> int(cs)
    sh = np.minimum(y >> F, 63)
    e = np.where(mask, TAB[y & ((1 << F) - 1)] >> sh, 0)
    S = e.sum(axis=1, keepdims=True)
    R = np.where(S > 0, (1 << RB) // np.maximum(S, 1), 0)
    P = (e * R + (1 << (RB - f_p - 1))) >> (RB - f_p)
    return np.minimum(P, 32767)


def softmax_cols(s: np.ndarray, valid, cm: int, cs: int, f_p: int) -> np.ndarray:
    """Column mode: ``s`` [keys][queries] raw scores (keys-major) -> P
    [queries][keys] — softmax_raw of the transpose."""
    return softmax_raw(np.asarray(s).T, valid, cm, cs, f_p)


def causal_valid(rows: int, valid0: int, period: int, n: int) -> np.ndarray:
    """The mask register's valid length per vector q: min(n, valid0 + (q mod
    period)), period 0: valid0."""
    q = np.arange(rows)
    return np.minimum(n, valid0 + (q % period if period else 0))


__all__ = ("F", "E", "RB", "TAB", "MAX_ROW", "MAX_KEYS", "scale_regs", "softmax_raw", "softmax_cols",
           "causal_valid")
