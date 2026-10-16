"""Integer specifications of the row ops a row-ops unit would run
(doc/plans/HOSTOPS_PLAN.md §5): what the hardware computes, bit for bit,
written as plain integer arithmetic so that the simulator, the C fallback
and the RTL can all be checked against one function — the pattern of
``vectorop_smx.py`` (the softmax unit).

Conventions: ``*_raw`` arrays are integers (numpy int64, or Python ints
where a product exceeds 63 bits); ``f`` is a power-of-two exponent
(value = raw * 2^-f); ``rne`` is round half to even; every output is
saturated to int16.  Nothing here uses floating point except to build the
constant tables (the rsqrt table, the SiLU table, the RoPE tables) — those
are computed once in double and rounded, as the activation and softmax
tables are.

The normalisations avoid every division: with S1 = sum x, S2 = sum x^2 and
V = n*S2 - S1^2 (exact integers, no cancellation),

    (x - mu) / sqrt(var + eps) = (n*x - S1) / sqrt(V + n^2 * eps * 2^(2f))

and for RMSNorm, h / sqrt(mean(h^2) + eps) = h * sqrt(n) / sqrt(sum h^2 + n*eps),
so sqrt(n) folds into gamma.  The one approximation is the reciprocal
square root: an 8 192-entry table on the leading 13 bits of the block-floated
denominator plus one Newton step in 32-bit integers (relative error below
2^-24), then the product with gamma at 2^-24 of the normalised value and a
single rounding at the output exponent.
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Optional, Tuple

import numpy as np

I16_LO, I16_HI = -32768, 32767
I32_LO, I32_HI = -2147483648, 2147483647
P1_BITS = 24          # the normalised value (x - mu) / sigma is kept at 2^-24
BQ_BITS = 20          # beta as an int22 at 2^-20 (the parameter block's Bq')
RM_BITS = 47          # rsqrt mantissa: 1/sqrt(Dm) * 2^47 for Dm in [2^30, 2^32)
ROPE_BITS = 23        # cos / sin as int24 at 2^-23
SILU_BITS = 20        # silu(g) as int32 at 2^-20


# ------------------------------------------------------------------ helpers
def rne_shr(v, k: int):
    """round half to even of v / 2^k (v integer array or int; k may be <= 0)."""
    if k <= 0:
        return v << (-k)
    q = v >> k
    r = v - (q << k)
    half = 1 << (k - 1)
    up = (r > half) | ((r == half) & ((q & 1) == 1))
    return q + up


def sat16(v):
    return np.clip(v, I16_LO, I16_HI)


def mul_shr_floor(a, m: int, k: int):
    """floor(a * m / 2^k) for an int64 array a with |a| < 2^62 and 0 < m < 2^32,
    exact without 128-bit integers: a is split into 20-bit limbs and the
    partial products are shifted and recombined with their remainders
    (floor((x * 2^20 + y) / 2^k) = floor(x / 2^(k-20)) + floor((rem * 2^20 + y) / 2^k))."""
    a = np.asarray(a, np.int64)
    amax = int(np.abs(a).max()) if a.size else 0
    if amax < (1 << 31):                          # |a * m| < 2^63: direct
        return ((a * m) << (-k)) if k <= 0 else ((a * m) >> k)
    if k >= 40:                                   # three 20-bit limbs, remainders carried down
        a0 = a & ((1 << 20) - 1)
        a1 = (a >> 20) & ((1 << 20) - 1)
        a2 = a >> 40                              # signed, |a2| < 2^22
        t2 = a2 * m                               # |t2| < 2^54
        q2 = t2 >> (k - 40)
        r2 = t2 - (q2 << (k - 40))                # 0 <= r2 < 2^(k-40)
        s1 = (r2 << 20) + a1 * m                  # < 2^(k-20) + 2^52
        q1 = s1 >> (k - 20)
        r1 = s1 - (q1 << (k - 20))                # 0 <= r1 < 2^(k-20)
        s0 = (r1 << 20) + a0 * m                  # < 2^k + 2^52 < 2^63 (k <= 62)
        return q2 + q1 + (s0 >> k)
    # wide a with a small shift: the result itself may be wide — exact Python integers
    out = np.array([(int(x) * m) >> k if k > 0 else (int(x) * m) << (-k) for x in a.reshape(-1)],
                   dtype=object).reshape(a.shape)
    return out.astype(np.int64)


# ------------------------------------------------------------------ rsqrt
@lru_cache(maxsize=None)
def rsqrt_table() -> np.ndarray:
    """y0[idx] = round(2^47 / sqrt((idx + 0.5) * 2^19)) for idx in [2048, 8192)
    (Dm = idx * 2^19 + ..., Dm in [2^30, 2^32)); entries below 2048 unused.
    A 13-bit index keeps the Newton step's residual below 2^-24 over the two
    octaves (a 12-bit one reaches 2^-23.4 at the bottom of the range)."""
    idx = np.arange(8192, dtype=np.float64)
    with np.errstate(divide="ignore"):
        y = np.rint(2.0 ** RM_BITS / np.sqrt((idx + 0.5) * 2.0 ** 19))
    y[:2048] = 0
    return np.minimum(y, 2.0 ** 32 - 1).astype(np.int64)


def rsqrt_bf(D: int) -> Tuple[int, int]:
    """1/sqrt(D) ~= Rm * 2^-sh for a positive integer D: Rm in [2^31, 2^32) from the
    table on the leading 13 bits of the block-floated D plus one Newton step in
    32-bit integers (every intermediate truncated as a right shift truncates)."""
    D = int(D)
    if D <= 0:
        raise ValueError("rsqrt of a non-positive denominator")
    e = D.bit_length()
    d = e - 32
    if d & 1:
        d += 1                                    # even exponent: Dm in [2^30, 2^32)
    Dm = D >> d if d >= 0 else D << (-d)
    y0 = int(rsqrt_table()[Dm >> 19])
    t = (Dm * y0) >> 32                           # ~ sqrt(Dm) * 2^15
    u = (t * y0) >> 32                            # ~ Dm * y0^2 * 2^-94 * 2^30 = 2^30 at exact
    y1 = y0 + ((y0 * ((1 << 30) - u)) >> 31)      # y0 * (1 + (1 - Dm y0^2) / 2)
    Rm = min(max(y1, 1 << 31), (1 << 32) - 1)
    return Rm, RM_BITS + d // 2


# ------------------------------------------------------------------ norms
def gamma_scale(g_real: np.ndarray) -> int:
    """the per-tensor exponent g with max |round(g_real * 2^g)| < 2^31."""
    m = float(np.abs(np.asarray(g_real, np.float64)).max())
    if m == 0.0:
        return 0
    return 30 - int(math.floor(math.log2(m)))


def beta_bits(beta: Optional[np.ndarray]) -> int:
    """beta's scale in the parameter block: Bq' = round(beta * 2^bb) must be an
    int22 — 2^-20 (BQ_BITS) while |beta| < 2, one bit coarser per doubling
    beyond (BERT's attention LayerNorms reach 4.2)."""
    if beta is None:
        return BQ_BITS
    m = float(np.abs(np.asarray(beta, np.float64)).max()) if np.size(beta) else 0.0
    bb = BQ_BITS
    while bb > 0 and abs(round(m * 2.0 ** bb)) >= (1 << 21):
        bb -= 1
    return bb


def bias_bits(bias: np.ndarray, f: int) -> int:
    """The scale of a residual add's bias in the parameter block: the sum's
    exponent f, coarser only when round(bias * 2^f) would exceed the int22."""
    bb = min(int(f), beta_bits(bias))
    return bb


def quant_bias(bias: np.ndarray, f: int) -> np.ndarray:
    """The bias as the unit adds it to a residual sum at 2^-f: rounded once at
    2^-bias_bits(bias, f) (values)."""
    bb = bias_bits(bias, f)
    return np.rint(np.asarray(bias, np.float64) * 2.0 ** bb) / 2.0 ** bb


BIAS_EXTRA = 8        # a COPY with a bias sums 8 bits below the finer of its input / output grids


def bias_exp(bias: np.ndarray, f_x, f_out, extra: int = BIAS_EXTRA) -> np.ndarray:
    """Per column, the exponent a COPY with a bias sums at (HOSTOPS_PLAN §10.4,
    the vision / TTS q, k, v bias adds and the TTS row prep): max(f_x, f_out)
    + extra, at most 24, lowered where round(bias * 2^fp) would leave the
    int22 parameter — never below max(f_x, f_out) (a ValueError then)."""
    bias = np.asarray(bias, np.float64).reshape(-1)
    n = bias.size
    f_x = np.broadcast_to(np.asarray(f_x, np.int64), (n,))
    f_out = np.broadcast_to(np.asarray(f_out, np.int64), (n,))
    lo = np.maximum(f_x, f_out)
    fp = np.minimum(lo + extra, 24)
    for _ in range(64):
        over = np.abs(np.rint(bias * np.power(2.0, fp.astype(np.float64)))) > (1 << 21) - 1
        if not over.any():
            break
        fp = np.where(over, fp - 1, fp)
    if (fp < lo).any():
        raise ValueError("bias_exp: a bias too large for the int22 parameter at the output's grid")
    return fp


def bias_params(bias: np.ndarray, f_x, f_out):
    """The parameter block of a COPY with a bias: (Bq' [n], sh_a [n], sh_c [n])
    at fp = bias_exp (bq_sh = 0)."""
    fp = bias_exp(bias, f_x, f_out)
    n = fp.size
    bq = np.rint(np.asarray(bias, np.float64).reshape(-1) * np.power(2.0, fp.astype(np.float64))).astype(np.int64)
    f_x = np.broadcast_to(np.asarray(f_x, np.int64), (n,))
    f_out = np.broadcast_to(np.asarray(f_out, np.int64), (n,))
    return bq, fp - f_x, fp - f_out


def bias_round(x_raw: np.ndarray, f_x, bias: np.ndarray, f_out, relu: bool = False) -> np.ndarray:
    """A COPY with a bias (ROWOPS_RTL_KERNEL.md §1.4 COPY): per column
    y = sat16(rne_shr((x << (fp - f_x)) + round(bias * 2^fp), fp - f_out)), then
    max(y, 0) with ``relu`` — the host's round((x * 2^-f_x + bias) * 2^f_out)
    with the bias rounded once at 2^-fp (bias_exp).  x_raw int16 [T][n]."""
    x_raw = np.asarray(x_raw, np.int64)
    bq, sa, sc = bias_params(bias, f_x, f_out)
    y = sat16(rne_shr_vec((x_raw << sa[None, :]) + bq[None, :], sc[None, :]))
    return np.maximum(y, 0) if relu else y


def norm_params(gamma: np.ndarray, n: int, eps: float, f_in: int, beta: Optional[np.ndarray] = None,
                rms: bool = False):
    """The job constants a norm needs: (G int32 [n], g, Bq [n] or None, E) —
    Bq at the 2^-(24 + g) scale of P1 * G: the unit's Bq' = round(beta * 2^bb)
    (bb = beta_bits(beta)) shifted by bq_sh = 24 + g - bb."""
    gamma = np.asarray(gamma, np.float64)
    gs = gamma * math.sqrt(n) if rms else gamma
    g = gamma_scale(gs)
    G = np.rint(gs * 2.0 ** g).astype(np.int64)
    if rms:
        E = int(round(eps * n * 2.0 ** (2 * f_in)))
        Bq = None
    else:
        E = int(round(eps * n * n * 2.0 ** (2 * f_in)))
        bb = beta_bits(beta)
        Bq = None if beta is None else (np.rint(np.asarray(beta, np.float64) * 2.0 ** bb).astype(np.int64)
                                        << (P1_BITS + g - bb))
    return G, g, Bq, E


def _out(P1: np.ndarray, G: np.ndarray, g: int, f_out, Bq: Optional[np.ndarray]) -> np.ndarray:
    """y_raw = rne((P1 * G + Bq) / 2^(24 + g - f_out[c])), saturated — P1 [rows, n]
    at 2^-24, G int32 [n], f_out scalar or [n]."""
    prod = P1 * G[None, :]                        # |P1| <= 2^31 (LN) / 2^24 (RMS), |G| < 2^31: < 2^62
    if Bq is not None:
        prod = prod + Bq[None, :]
    f_out = np.broadcast_to(np.asarray(f_out, np.int64), (P1.shape[1],))
    y = np.empty_like(prod)
    for k in np.unique(P1_BITS + g - f_out):
        cols = (P1_BITS + g - f_out) == k
        y[:, cols] = rne_shr(prod[:, cols], int(k))
    return sat16(y)


def rms_norm(h_raw: np.ndarray, f_h: int, gamma: np.ndarray, eps: float, f_out) -> np.ndarray:
    """RMSNorm rows: h_raw int [rows, n] at 2^-f_h (the int32 residual), gamma
    float [n] (folded with sqrt(n) into int32 G), output int16 at 2^-f_out[c].
    The sum of squares is exact (up to 2^74 for int32 rows of 4096: the
    unit's wide accumulators)."""
    h_raw = np.asarray(h_raw, np.int64)
    rows, n = h_raw.shape
    G, g, _b, E = norm_params(gamma, n, eps, f_h, rms=True)
    out = np.empty_like(h_raw)
    hobj = h_raw.astype(object)                   # sum h^2 can exceed 63 bits
    for r in range(rows):
        row = hobj[r]
        D = int(np.sum(row * row)) + E
        Rm, sh = rsqrt_bf(D)
        P1 = mul_shr_floor(h_raw[r], Rm, sh - P1_BITS)        # h_raw * R at 2^-24, |.| <= 2^24
        out[r] = _out(P1[None, :], G, g, f_out, None)[0]
    return out


def layer_norm(x_raw: np.ndarray, f_x: int, gamma: np.ndarray, beta: Optional[np.ndarray], eps: float,
               f_out) -> np.ndarray:
    """LayerNorm rows: x_raw int [rows, n] at 2^-f_x (int16, or the int32
    residual), gamma / beta float [n], output int16 at 2^-f_out[c].  S1, S2,
    V = n*S2 - S1^2 and A = n*x - S1 are exact integers (no cancellation: the
    unit's wide accumulators)."""
    x_raw = np.asarray(x_raw, np.int64)
    rows, n = x_raw.shape
    G, g, Bq, E = norm_params(gamma, n, eps, f_x, beta=beta)
    out = np.empty_like(x_raw)
    xobj = x_raw.astype(object)
    for r in range(rows):
        row = xobj[r]
        S1 = int(row.sum())
        S2 = int(np.sum(row * row))
        V = n * S2 - S1 * S1                               # exact, >= 0
        D = V + E
        if D <= 0:
            D = 1
        Rm, sh = rsqrt_bf(D)
        A = n * x_raw[r] - S1                              # |A| < 2^44 (int32 rows of 4096)
        P1 = mul_shr_floor(A, Rm, sh - P1_BITS)            # (x - mu)/sigma at 2^-24
        out[r] = _out(P1[None, :], G, g, f_out, Bq)[0]
    return out


# ------------------------------------------------------------------ RoPE
def rope_tables_raw(cos: np.ndarray, sin: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """int24 at 2^-23 of the (float32) cos / sin tables."""
    return (np.rint(np.asarray(cos, np.float64) * 2.0 ** ROPE_BITS).astype(np.int64),
            np.rint(np.asarray(sin, np.float64) * 2.0 ** ROPE_BITS).astype(np.int64))


def rne_shr_vec(v: np.ndarray, k: np.ndarray) -> np.ndarray:
    """rne_shr with a per-element shift k >= 0 (array broadcastable to v)."""
    k = np.broadcast_to(np.asarray(k, np.int64), v.shape)
    kk = np.maximum(k, 1)
    q = v >> kk
    r = v - (q << kk)
    half = np.int64(1) << (kk - 1)
    up = (r > half) | ((r == half) & ((q & 1) == 1))
    return np.where(k > 0, q + up, v)


def rope(x_raw: np.ndarray, f_x, c_raw: np.ndarray, s_raw: np.ndarray, f_out) -> np.ndarray:
    """rotate-half RoPE: x_raw int16 [T, nh, HD] at 2^-f_x (per head, or per
    channel [nh, HD] — the kernel output's exponents), c_raw / s_raw int24
    [T, HD] at 2^-23 -> int16 at 2^-f_out (per head or per channel), one
    rounding of the exact integer sum x*c + rot(x)*s.  When the two elements
    of a rotation pair sit at different exponents both products are shifted
    up to the finer one of the pair first (exact)."""
    x_raw = np.asarray(x_raw, np.int64)
    T, nh, HD = x_raw.shape
    half = HD // 2
    f_x = np.asarray(f_x, np.int64)
    f_x = np.broadcast_to(f_x[:, None] if f_x.ndim == 1 else f_x, (nh, HD))
    f_out = np.asarray(f_out, np.int64)
    f_out = np.broadcast_to(f_out[:, None] if f_out.ndim == 1 else f_out, (nh, HD))
    f_rot = np.concatenate([f_x[:, half:], f_x[:, :half]], axis=1)      # the partner's exponent
    fp = np.maximum(f_x, f_rot)
    rot = np.concatenate([-x_raw[..., half:], x_raw[..., :half]], axis=-1)
    num = (((x_raw * c_raw[:, None, :]) << (fp - f_x)[None])
           + ((rot * s_raw[:, None, :]) << (fp - f_rot)[None]))          # < 2^42
    return sat16(rne_shr_vec(num, (fp + ROPE_BITS - f_out)[None]))


# ------------------------------------------------------------------ SiLU * up
# silu(g) = g * sigma(g) from a 4 096-entry sigmoid table over |g| in [0, 16) at
# 2^-8 (entries at 2^-24) with linear interpolation on the 6 bits below the
# table step: g is first brought to 2^-14 (exact for any column exponent
# f_g <= 14, so per-channel exponents need no per-exponent table), the
# sigmoid's symmetry gives g < 0, and |g| >= 16 is the identity / zero.
G_BITS   = 14         # g at 2^-14 inside the unit
SIG_BITS = 24         # sigma at 2^-24
SIG_STEP = 8          # table step 2^-8: |g| at 2^-14 >> 6 indexes it
SIG_N    = 4096       # entries for |g| < 16 (+ 1 past the end)


@lru_cache(maxsize=None)
def sigmoid_table() -> np.ndarray:
    """T[i] = round(sigma(i / 256) * 2^24), i = 0 .. 4096 — with the C library's
    exp (math.exp), as the RTL's ROM generator and the C++ oracle compute it."""
    return np.array([round(2.0 ** SIG_BITS / (1.0 + math.exp(-i / 2.0 ** SIG_STEP)))
                     for i in range(SIG_N + 1)], dtype=np.int64)


def sig_q(gv: np.ndarray) -> np.ndarray:
    """sigma(g) at 2^-24 for g = gv * 2^-14 (gv int64): the interpolated table
    of |g|, mirrored for g < 0, 2^24 for |g| >= 16."""
    gv = np.asarray(gv, np.int64)
    a = np.abs(gv)
    big = a >= (16 << G_BITS)
    ai = np.minimum(a, (16 << G_BITS) - 1)
    idx = ai >> (G_BITS - SIG_STEP)
    frac = ai & ((1 << (G_BITS - SIG_STEP)) - 1)
    T = sigmoid_table()
    t0 = T[idx]
    sig = t0 + (((T[idx + 1] - t0) * frac) >> (G_BITS - SIG_STEP))
    sig = np.where(big, 1 << SIG_BITS, sig)
    return np.where(gv < 0, (1 << SIG_BITS) - sig, sig)          # sigma(-x) = 1 - sigma(x)


def silu_q(gv: np.ndarray) -> np.ndarray:
    """silu(g) at 2^-20 for g = gv * 2^-14 (gv int64): the interpolated sigmoid
    of |g|, mirrored for g < 0, the product gv * sigma rounded from 2^-38."""
    return rne_shr(gv * sig_q(gv), G_BITS + SIG_BITS - SILU_BITS)    # 2^-38 -> 2^-20


def silu_mul(g_raw: np.ndarray, f_g, u_raw: np.ndarray, f_u, f_out) -> np.ndarray:
    """a_raw = rne(silu_q(g_raw << (14 - f_g[c])) * u_raw / 2^(20 + f_u - f_out))
    per column (f_g <= 14, f_u, f_out scalars or per-column vectors), saturated int16."""
    g_raw = np.asarray(g_raw, np.int64)
    u_raw = np.asarray(u_raw, np.int64)
    n = g_raw.shape[1]
    f_g = np.broadcast_to(np.asarray(f_g, np.int64), (n,))
    f_u = np.broadcast_to(np.asarray(f_u, np.int64), (n,))
    f_out = np.broadcast_to(np.asarray(f_out, np.int64), (n,))
    if (f_g > G_BITS).any():
        raise ValueError("silu_mul: a gate exponent above 14")
    gv = g_raw << (G_BITS - f_g)[None, :]                               # exact, < 2^22
    prod = silu_q(gv) * u_raw                                           # < 2^28 * 2^15
    y = np.empty_like(prod)
    k = SILU_BITS + f_u - f_out
    for kk in np.unique(k):
        cols = k == kk
        y[:, cols] = rne_shr(prod[:, cols], int(kk))
    return sat16(y)


# ------------------------------------------------------------------ GATE (op 10)
# Piper's TtsGate, tanh(a) * sigma(b) (doc/plans/RPU_PLAN.md §14.2): tanh x =
# 2 sigma(2x) - 1 from the same sigmoid table, the product at 2^-48 rounded once.
# Not libm: one LSB on ~3-5e-5 of the outputs (a study policy, RPU_PLAN §14.8).
GATE_BITS = 2 * SIG_BITS      # th * sg at 2^-48


def gate_shifts(f_in: int, f_out: int) -> Tuple[int, int, int]:
    """(sh_a, sh_b, sh_c) of a GATE job: a and b at 2^-f_in, y at 2^-f_out."""
    if not 1 <= int(f_in) <= G_BITS:
        raise ValueError(f"gate: input exponent {f_in} outside 1 .. {G_BITS}")
    sh_c = GATE_BITS - int(f_out)
    if not 0 <= sh_c <= 63:
        raise ValueError(f"gate: output exponent {f_out} beyond the shift")
    return G_BITS + 1 - int(f_in), G_BITS - int(f_in), sh_c


def gate_op(a_raw: np.ndarray, b_raw: np.ndarray, sh_a, sh_b, sh_c) -> np.ndarray:
    """The GATE job per element (ROWOPS_RTL_KERNEL.md §1.4): th = 2 sig_q(a << sh_a) -
    2^24, sg = sig_q(b << sh_b), y = sat16(rne_shr(th * sg, sh_c)); the shifts
    broadcast (per column).  Contract: sh_a, sh_b <= 16 (a << sh_a within int32)."""
    a = np.asarray(a_raw, np.int64)
    b = np.asarray(b_raw, np.int64)
    th = 2 * sig_q(a << np.asarray(sh_a, np.int64)) - (1 << SIG_BITS)
    sg = sig_q(b << np.asarray(sh_b, np.int64))
    prod = th * sg
    return sat16(rne_shr_vec(prod, np.broadcast_to(np.asarray(sh_c, np.int64), prod.shape)))


def gate(a_raw: np.ndarray, b_raw: np.ndarray, f_in: int, f_out: int) -> np.ndarray:
    """y = sat16(rne(th * sg / 2^(48 - f_out))) with th = 2 sigma_q(2a) - 2^24 and
    sg = sigma_q(b) (a, b int16 at 2^-f_in; sigma_q = sig_q at 2^-24)."""
    return gate_op(a_raw, b_raw, *gate_shifts(f_in, f_out))


# ------------------------------------------------------------------ SCALE (op bit 17)
# COPY / ADDX with SCALE multiply the sum by the column's G before the rounding
# (RPU_PLAN §14.3): y = sat_C(rne_shr(S * G, sh_c)), S = (A << sh_a) [+ (B << sh_b)]
# + (Bq' << bq_sh).  Contract: |S| < 2^31, 0 <= G < 2^25.  The Piper stage average
# (y0 + y1 + y2) / 3 one exponent finer is S = T + y2 (T = y0 + y1, int32) with
# G = AVG3_G, sh_c = AVG3_SH: rne(2 S / 3) for every |S| <= 3 * 2^15 (no ties).
SCALE_S_BITS = 31
SCALE_G_BITS = 25
AVG3_SH = 18
AVG3_G = (2 ** (AVG3_SH + 1) + 1) // 3          # round(2^19 / 3) = 174 763


def scaled_round(s: np.ndarray, G, sh_c) -> np.ndarray:
    """rne_shr(S * G, sh_c) of the pre-rounding sums S (int64; no saturation)."""
    s = np.asarray(s, np.int64)
    if (np.abs(s) >= (1 << SCALE_S_BITS)).any():
        raise ValueError("SCALE: a sum outside +-2^31")
    G = np.asarray(G, np.int64)
    if ((G < 0) | (G >= (1 << SCALE_G_BITS))).any():
        raise ValueError("SCALE: G outside [0, 2^25)")
    prod = s * G
    return rne_shr_vec(prod, np.broadcast_to(np.asarray(sh_c, np.int64), prod.shape))


# ------------------------------------------------------------------ FOLD (op 9)
# Piper's TtsPrep (doc/plans/HOSTOPS_PLAN.md §11): rows of a [C][L] source cut into
# overlapping windows, LeakyReLU with alpha = 1/m and the re-rounding from f_in to
# f_out = f_in - s.  The C's double arithmetic (alpha * v rounded once, then the
# round half to even at f_out) equals the exact quotient below for the alphas
# fold_divisor() accepts — checked over every int16 for each call.
FOLD_K = 24           # the datapath's reciprocal: R = ceil(2^24 / m)
FOLD_M_MAX = 127      # aux[10:4]
FOLD_S_MAX = 15       # aux[3:0]
FOLD_W_MAX = 65535    # elements per output row (n)


def fold_elem(x_raw, m: int, s: int) -> np.ndarray:
    """rne(x / (m_x * 2^s)), m_x = m for x < 0 and 1 for x >= 0 (x int16, s >= 0):
    LeakyReLU with alpha = 1/m and the round half to even s bits coarser, exactly.
    No value saturates."""
    x = np.asarray(x_raw, np.int64)
    den = np.where(x < 0, np.int64(m), np.int64(1)) << np.int64(s)
    q, r = np.divmod(np.abs(x), den)
    up = (2 * r > den) | ((2 * r == den) & ((q & 1) == 1))
    return np.where(x < 0, -(q + up), q + up)


def fold_recip(x_raw, m: int, s: int) -> np.ndarray:
    """fold_elem as the datapath computes it, without a divider: u = 2|x|, R_x =
    ceil(2^24 / m) for x < 0 (2^24 for x >= 0), P = u * R_x; q1 = P >> 24 is
    floor(u / m_x) and (P mod 2^24) < R_x says its remainder is 0 (exact for
    |x| <= 32768 and m <= FOLD_M_MAX: the error of R_x stays below 1 / m); then
    the round at s + 1 bits — q = q1 >> (s + 1), low = q1 mod 2^(s + 1), up when
    low > 2^s, or low == 2^s and (the remainder is not 0 or q is odd)."""
    x = np.asarray(x_raw, np.int64)
    R = np.where(x < 0, np.int64(-(-(1 << FOLD_K) // m)), np.int64(1 << FOLD_K))
    P = (2 * np.abs(x)) * R
    q1 = P >> FOLD_K
    z = (P & ((1 << FOLD_K) - 1)) < R
    q = q1 >> (s + 1)
    low = q1 & ((1 << (s + 1)) - 1)
    half = 1 << s
    up = (low > half) | ((low == half) & (~z | ((q & 1) == 1)))
    return np.where(x < 0, -(q + up), q + up)


def fold_divisor(alpha: float, s: int) -> Optional[int]:
    """The m of a TtsPrep call: alpha = 1 gives 1; else m = round(1 / alpha) when it is
    in 2 .. FOLD_M_MAX and the C's arithmetic — nearbyint(fl(alpha * x) * 2^-s) —
    equals fold_elem(x, m, s) for every negative int16 x (the rest is the same rule
    for both); None when no m fits (the call stays on the host)."""
    if not 0 <= int(s) <= FOLD_S_MAX:
        return None
    if alpha == 1.0:
        return 1
    if not 0.0 < alpha < 1.0:
        return None
    m = int(round(1.0 / alpha))
    if not 2 <= m <= FOLD_M_MAX:
        return None
    x = np.arange(I16_LO, 0, dtype=np.int64)
    c = np.clip(np.rint(x.astype(np.float64) * alpha * 2.0 ** -int(s)), I16_LO, I16_HI).astype(np.int64)
    return m if np.array_equal(c, fold_elem(x, m, int(s))) else None


def fold(x_raw: np.ndarray, rows: int, w0: int, s0: int, W: int, slo: int, shi: int, m: int,
         s: int) -> np.ndarray:
    """FOLD: y[k][r][j] = fold_elem(x[k][c], m, s) with c = s0 + r * w0 + j when c lies
    in [slo, shi), else 0 — x [nch][L] int16 (the source rows in output channel order),
    y [nch][rows][W].  The window must lie inside the source row where it is used."""
    x = np.asarray(x_raw, np.int64)
    L = x.shape[1]
    col = int(s0) + np.arange(rows, dtype=np.int64)[:, None] * int(w0) + np.arange(W, dtype=np.int64)[None, :]
    ok = (col >= slo) & (col < shi)
    if ok.any() and (int(col[ok].min()) < 0 or int(col[ok].max()) >= L):
        raise ValueError("fold: the window [slo, shi) leaves the source row")
    v = fold_elem(x[:, np.clip(col, 0, L - 1)], m, s)
    return np.where(ok[None], v, 0)


def fold_aux(m: int, s: int) -> int:
    """FOLD's aux register: [3:0] s, [10:4] m."""
    if not (0 <= s <= FOLD_S_MAX and 1 <= m <= FOLD_M_MAX):
        raise ValueError(f"fold_aux: m {m} or s {s} outside the register fields")
    return int(s) | (int(m) << 4)


# ------------------------------------------------------------------ the unit's registers
# RowOpsKernel (kernels/rowops_rtl, doc/kernels/ROWOPS_RTL_KERNEL.md §1): the op
# register's fields, the kinds and the parameter block's 16-byte entry.
OP_COPY, OP_ADDX, OP_DEQUANT, OP_MULACT, OP_NORM, OP_ROPE, OP_XPOSE, OP_TLOAD = 1, 2, 3, 4, 5, 6, 7, 8
OP_FOLD = 9
OP_GATE = 10                  # RPU_PLAN §14.3
K_I16, K_I32, K_F32 = 0, 1, 2
F_B_EN, F_P_BCAST, F_P_KEEP, F_RELU, F_NORM_LN = 1 << 12, 1 << 13, 1 << 14, 1 << 15, 1 << 16
F_SCALE = 1 << 17             # COPY / ADDX: the sum times G before the rounding (RPU_PLAN §14.3)
N_MAX = 4096                  # elements per row
PARAM_BYTES = 16              # one column's entry: int32 G, int32 Bq' (int22), u8 sh_a, sh_b, sh_c
TABLE_BYTES = 65536           # the image OP_TLOAD reads once per bitstream
TABLE_WORDS = TABLE_BYTES // 16


def opword(op: int, ak: int = K_I16, bk: int = K_I16, ck: int = K_I16, b_en: bool = False,
           p_bcast: bool = False, p_keep: bool = False, relu: bool = False, ln: bool = False,
           scale: bool = False) -> int:
    """The op register: [3:0] op, [5:4] A kind, [7:6] B kind, [9:8] C kind, the flags."""
    return (int(op) | (int(ak) << 4) | (int(bk) << 6) | (int(ck) << 8) | (F_B_EN if b_en else 0)
            | (F_P_BCAST if p_bcast else 0) | (F_P_KEEP if p_keep else 0) | (F_RELU if relu else 0)
            | (F_NORM_LN if ln else 0) | (F_SCALE if scale else 0))


def param_block(G, Bq, sh_a, sh_b, sh_c) -> np.ndarray:
    """The parameter block P of n columns as raw int16 words (8 per entry): per
    column int32 G, int32 Bq' (an int22 sign-extended), u8 sh_a, sh_b, sh_c, the
    rest 0 (ROWOPS_RTL_KERNEL.md §1.3).  Scalars broadcast over the columns."""
    G = np.asarray(G, np.int64).reshape(-1)
    n = G.size
    Bq = np.broadcast_to(np.asarray(0 if Bq is None else Bq, np.int64), (n,))
    sa = np.broadcast_to(np.asarray(sh_a, np.int64), (n,))
    sb = np.broadcast_to(np.asarray(sh_b, np.int64), (n,))
    sc = np.broadcast_to(np.asarray(sh_c, np.int64), (n,))
    if (np.abs(G) > I32_HI).any() or (Bq < -(1 << 21)).any() or (Bq >= (1 << 21)).any():
        raise ValueError("param_block: G beyond int32 or Bq' beyond int22")
    if ((sa | sb | sc) < 0).any() or ((sa | sb | sc) > 63).any():
        raise ValueError("param_block: a shift outside 0..63")
    img = np.zeros((n, 16), np.uint8)
    img[:, 0:4] = (G & 0xFFFFFFFF).astype(np.uint32).view(np.uint8).reshape(n, 4)
    img[:, 4:8] = (Bq & 0xFFFFFFFF).astype(np.uint32).view(np.uint8).reshape(n, 4)
    img[:, 8] = sa
    img[:, 9] = sb
    img[:, 10] = sc
    return img.reshape(-1).view(np.int16).astype(np.int64)


@lru_cache(maxsize=None)
def table_image() -> bytes:
    """The 64 KiB image OP_TLOAD reads (ROWOPS_RTL_KERNEL.md §1.4): 2048 words of two
    sigmoid entries {D[i], T[i]} (49 bits in each 64-bit half), then 2048 words of four
    rsqrt entries (32 bits each) — the UltraRAM tables, which have no initial contents."""
    T = [int(v) for v in sigmoid_table()]
    R = [int(v) for v in rsqrt_table()]
    img = bytearray(TABLE_BYTES)
    for i in range(SIG_N):
        d = (T[i + 1] - T[i]) & 0xFFFFFF
        img[8 * i: 8 * i + 8] = ((d << 25) | (T[i] & 0x1FFFFFF)).to_bytes(8, "little")
    for i in range(8192):
        img[32768 + 4 * i: 32768 + 4 * i + 4] = (R[i] & 0xFFFFFFFF).to_bytes(4, "little")
    return bytes(img)


__all__ = ("rne_shr", "rne_shr_vec", "sat16", "mul_shr_floor", "rsqrt_bf", "rsqrt_table", "norm_params", "beta_bits",
           "rms_norm", "layer_norm", "rope_tables_raw", "rope", "sigmoid_table", "silu_q", "silu_mul",
           "bias_bits", "quant_bias", "BIAS_EXTRA", "bias_exp", "bias_params", "bias_round",
           "OP_COPY", "OP_ADDX", "OP_DEQUANT", "OP_MULACT", "OP_NORM", "OP_ROPE", "OP_XPOSE", "OP_TLOAD",
           "OP_FOLD", "FOLD_K", "FOLD_M_MAX", "FOLD_S_MAX", "FOLD_W_MAX", "fold_elem", "fold_recip",
           "fold_divisor", "fold", "fold_aux",
           "K_I16", "K_I32", "K_F32", "F_B_EN", "F_P_BCAST", "F_P_KEEP", "F_RELU", "F_NORM_LN", "N_MAX",
           "PARAM_BYTES", "TABLE_BYTES", "TABLE_WORDS", "opword", "param_block", "table_image")
