# Copyright 2026 Ivan Gubochkin
# SPDX-License-Identifier: Apache-2.0

"""Integer specification of the 8-bit configuration (doc/plans/INT8_PLAN.md §6):
what the 8-bit kernels compute, bit for bit, as plain integer arithmetic — the
oracle of the studies' ``i8`` policies, of the scheduler's simulator and of the
RTL testbenches, the pattern of ``rowops_spec.py``.

Conventions (those of ``numeric.py``): ``*_raw`` / ``q`` arrays are integers
(numpy int64); ``f`` is a power-of-two exponent, value = raw * 2^-f; ``rne`` is
round half to even.  numpy and the standard library only: the studies load this
file by path.

**Weights.**  int8 in [-127, 127] (symmetric: the packed DSP pair below needs
it), one exponent per output channel ``f_w[n]``, the error-minimal of three
candidates: the finest grid that does not clip and the two coarser ones.

**Blocked activations.**  A tensor read by ConvKernel / MatmulKernel is int8 in
[-128, 127] with a per-tensor exponent ``f`` and, per block of ``BLOCK`` values
along the reduction axis (the last axis here), a shift ``d`` in [0, DELTA_MAX]:
element value = q * 2^-(f + d).  ``d`` is the largest shift whose rounded block
still fits int8 — chosen by the kernel that writes the tensor
(``block_round``).  A static tensor has d = 0 everywhere (``dmax = 0``).

**Accumulation.**  Within a block the products are summed exactly; the block
sum enters the accumulator shifted left by ``DELTA_MAX - d`` (so the
accumulator is at exponent ``f_x + DELTA_MAX + f_w[n]``), which is a two's
complement integer of ``ACC_BITS`` bits that wraps, as today's kernels' int32
does.  Output: a per-channel shift, rne, saturation (``requant``), or a blocked
int8 output by the same rule as any producer (``requant_blocked``).

**Packed DSP pair.**  One DSP48E2 multiplies the 27-bit word
``w0 * 2^FIELD + w1`` by an int8 x: P = S0 * 2^FIELD + S1.  With weights in
[-127, 127] and x in [-128, 127] each product is within ±16 256, so CHAIN = 16
products keep |S1| <= 260 096 < 2^(FIELD-1) and both sums are recovered exactly
(``unpack``).

**W8A16.**  ``matmul_w8a16``: an int16 input times int8 weights, one product per DSP — for a
layer whose input needs 16 bits while its weight bytes matter (the LM head).

**Unsigned tensors.**  A static post-ReLU tensor holds u in [0, 255] stored as
q = u - 128; the consumer adds 128 * sum_k w[k][n] to its accumulator (in the
bias, ``offset_bias``), and the padding of such a tensor is -128.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

I8_LO, I8_HI = -128, 127      # activations
W8_LO, W8_HI = -127, 127      # weights: symmetric, for the packed pair
BLOCK = 32                    # block length along the reduction axis
DELTA_MAX = 7                 # the block shift's range: 3 bits
CHAIN = 16                    # products per DSP cascade
FIELD = 19                    # the packed word's low-field width
ACC_BITS = 40                 # the block-shifted accumulator (wraps)
U8_OFFSET = 128               # unsigned tensors are stored as u - 128
WEIGHT_CANDIDATES = 3


# ------------------------------------------------------------------ helpers
def rne_shr(v, k):
    """round half to even of v / 2^k, elementwise; v an integer array, k an
    integer or an integer array broadcastable to v (k <= 0: exact left shift)."""
    v = np.asarray(v, dtype=np.int64)
    k = np.broadcast_to(np.asarray(k, dtype=np.int64), v.shape)
    left = k <= 0
    kp = np.where(left, 1, k)
    q = v >> kp
    r = v - (q << kp)
    half = np.int64(1) << (kp - 1)
    up = (r > half) | ((r == half) & ((q & 1) == 1))
    return np.where(left, v << np.where(left, -k, 0), q + up)


def rne_float(x) -> np.ndarray:
    """round half to even of a float64 array to int64 (exact for x * 2^k)."""
    return np.round(np.asarray(x, dtype=np.float64)).astype(np.int64)


def wrap(v, bits: int = ACC_BITS) -> np.ndarray:
    """two's complement wrap to ``bits`` bits; bits >= 63: no wrap (int64 itself)."""
    v = np.asarray(v, dtype=np.int64)
    if bits >= 63:
        return v
    m = np.int64(1) << bits
    return ((v + (m >> 1)) & (m - 1)) - (m >> 1)


# ------------------------------------------------------------------ weights
def encode_weights(w, axis: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """float weights -> (q int8 in [-127, 127], f_w per output channel).

    ``axis`` is the output-channel axis (0 for a Conv [O, I, kh, kw] or a
    torch Linear [out, in]; 1 for a MatMul B [K, N]).  Per channel the
    candidates are f0, the finest exponent at which max |w| rounds into
    [-127, 127], and f0 + 1, f0 + 2 (finer grids that clip the largest
    weights); the one with the least squared error wins, ties to the coarser.
    An all-zero channel gets f = 0."""
    w = np.asarray(w, dtype=np.float64)
    wm = np.moveaxis(w, axis, 0).reshape(w.shape[axis], -1)
    amax = np.abs(wm).max(axis=1)
    f_w = np.zeros(len(amax), dtype=np.int64)
    q = np.zeros(wm.shape, dtype=np.int64)
    for n, a in enumerate(amax):
        if a == 0.0:
            continue
        f0 = int(np.floor(np.log2(W8_HI / a)))
        while abs(round(a * 2.0 ** f0)) > W8_HI:       # rne(a * 2^f0) must fit
            f0 -= 1
        while abs(round(a * 2.0 ** (f0 + 1))) <= W8_HI:
            f0 += 1
        best = None
        for f in range(f0, f0 + WEIGHT_CANDIDATES):
            qn = np.clip(rne_float(wm[n] * 2.0 ** f), W8_LO, W8_HI)
            err = float(np.sum((qn * 2.0 ** -f - wm[n]) ** 2))
            if best is None or err < best[0]:
                best = (err, f, qn)
        f_w[n], q[n] = best[1], best[2]
    qm = q.reshape((w.shape[axis],) + tuple(np.delete(w.shape, axis)))
    return np.moveaxis(qm, 0, axis), f_w


# ------------------------------------------------------------------ the block rule
def _blocks(a: np.ndarray, block: int) -> np.ndarray:
    n = a.shape[-1]
    if n % block:
        raise ValueError(f"last axis {n} is not a multiple of the block {block}")
    return a.reshape(a.shape[:-1] + (n // block, block))


def block_round(A, k, block: int = BLOCK, dmax: int = DELTA_MAX,
                lo: int = I8_LO, hi: int = I8_HI) -> Tuple[np.ndarray, np.ndarray]:
    """The producer's rule.  A: integers [..., N] (the value A * 2^-g); k: the
    shift g - f to the tensor's base grid, an integer or a per-element array
    [..., N] (per-channel exponents).  Per block of ``block`` along the last
    axis, d = the largest shift in [0, dmax] at which every element of the
    block, rne_shr(A, k - d), lies in [lo, hi]; none fits -> d = 0 and the
    elements saturate.  Returns (q [..., N], d [..., N / block])."""
    A = np.asarray(A, dtype=np.int64)
    kk = np.broadcast_to(np.asarray(k, dtype=np.int64), A.shape)
    if dmax == 0:
        d_best = np.zeros(A.shape[:-1] + (A.shape[-1] // block,), dtype=np.int64)
    elif A.size == 0 or (A.max() < 2 ** 53 and A.min() > -(2 ** 53)):
        d_best = _block_shifts(np.ldexp(A.astype(np.float64), -kk), block, dmax, lo, hi)   # A * 2^-k: exact
    else:                                              # beyond float64's integers: each shift in turn
        d_best = np.zeros(A.shape[:-1] + (A.shape[-1] // block,), dtype=np.int64)
        for d in range(1, dmax + 1):
            r = _blocks(rne_shr(A, kk - d), block)
            fits = (r.min(axis=-1) >= lo) & (r.max(axis=-1) <= hi)
            d_best = np.where(fits, d, d_best)
    dk = np.repeat(d_best, block, axis=-1)
    q = np.clip(rne_shr(A, kk - dk), lo, hi)
    return q, d_best


def _block_shifts(y, block: int, dmax: int, lo: int, hi: int) -> np.ndarray:
    """d per block of y [..., N] (float64): the largest d in [1, dmax] at which every
    rne(y * 2^d) of the block lies in [lo, hi], else 0.  rne is monotone, so a block fits
    where its extremes do, and rne(t) <= hi exactly when t < hi + 1/2 (or t = hi + 1/2 for
    an even hi, the tie rounding down to it), rne(t) >= lo when t > lo - 1/2 (or = for an
    even lo); y * 2^d is exact — the rule of block_round's definition, one pass over y
    instead of one per shift."""
    yb = _blocks(y, block)
    ymax, ymin = yb.max(axis=-1), yb.min(axis=-1)
    up, dn = hi + 0.5, lo - 0.5
    d_best = np.zeros(ymax.shape, dtype=np.int64)
    for d in range(1, dmax + 1):
        t_hi, t_lo = np.ldexp(ymax, d), np.ldexp(ymin, d)
        fits = (((t_hi < up) | ((t_hi == up) & (hi % 2 == 0)))
                & ((t_lo > dn) | ((t_lo == dn) & (lo % 2 == 0))))
        d_best = np.where(fits, d, d_best)
    return d_best


def block_round_float(x, f: int, block: int = BLOCK, dmax: int = DELTA_MAX,
                      lo: int = I8_LO, hi: int = I8_HI) -> Tuple[np.ndarray, np.ndarray]:
    """The same rule for a producer that computes in double (a host op, the
    studies' calibration): rne(x * 2^(f + d))."""
    x = np.asarray(x, dtype=np.float64)
    if dmax == 0:
        d_best = np.zeros(x.shape[:-1] + (x.shape[-1] // block,), dtype=np.int64)
    else:
        d_best = _block_shifts(x * 2.0 ** f, block, dmax, lo, hi)
    dk = np.repeat(d_best, block, axis=-1)
    q = np.clip(rne_float(x * np.exp2(f + dk)), lo, hi)
    return q, d_best


def block_value(q, f: int, d, block: int = BLOCK) -> np.ndarray:
    """the float64 value of a blocked tensor (exact)."""
    dk = np.repeat(np.asarray(d, dtype=np.int64), block, axis=-1)
    return np.asarray(q, dtype=np.float64) * np.exp2(-(f + dk))


def base_exponent(amax: float, margin: int = 0) -> int:
    """the per-tensor exponent E of a calibrated tensor: the finest f at which
    amax rounds into [-128, 127], ``margin`` bits coarser (headroom)."""
    if amax <= 0.0:
        return 0
    f = int(np.floor(np.log2(I8_HI / amax)))
    while abs(round(amax * 2.0 ** f)) > I8_HI:
        f -= 1
    while abs(round(amax * 2.0 ** (f + 1))) <= I8_HI:
        f += 1
    return f - margin


# ------------------------------------------------------------------ the MAC side
def matmul(xq, f_x: int, xd, wq, f_w, block: int = BLOCK, dmax: int = DELTA_MAX,
           bias_acc=None, acc_bits: int = ACC_BITS) -> Tuple[np.ndarray, np.ndarray]:
    """y[..., n] = sum_k x[..., k] w[k, n] as the 8-bit kernels compute it.

    xq: int8 [..., K] blocked (f_x, xd [..., K / block]; a static tensor passes
    the scalar xd = 0, and then K is free); wq: int8 [K, N] at the per-column
    exponents f_w [N].  Each block's
    products are summed exactly and shifted left by dmax - d; bias_acc (raw at
    the accumulator's exponent, [N]) seeds the sum; the accumulator wraps at
    acc_bits.  Returns (acc [..., N], g [N]): value = acc * 2^-g,
    g = f_x + dmax + f_w."""
    xq = np.asarray(xq, dtype=np.int64)
    wq = np.asarray(wq, dtype=np.int64)
    K = xq.shape[-1]
    if wq.shape[0] != K:
        raise ValueError(f"K mismatch: x {K}, w {wq.shape[0]}")
    if K * 128 * 128 << dmax >= 1 << 53:
        raise ValueError(f"K = {K}: the exact float64 evaluation needs K * 2^(14 + dmax) < 2^53")
    # every block's elements times 2^(dmax - d): |x'| < 2^(7 + dmax), and every partial sum of
    # x' @ w is below K * 2^(14 + dmax) < 2^53 — exact in float64, whatever order BLAS sums in
    if np.ndim(xd) == 0:                         # a static input: any K
        xs = xq << (dmax - int(xd))
    else:
        if K % block:
            raise ValueError(f"K = {K} is not a multiple of the block {block}")
        xd = np.broadcast_to(np.asarray(xd, dtype=np.int64), xq.shape[:-1] + (K // block,))
        xs = xq * (np.int64(1) << np.repeat(dmax - xd, block, axis=-1))
    acc = np.rint(xs.astype(np.float64) @ wq.astype(np.float64)).astype(np.int64)
    if bias_acc is not None:
        acc = acc + np.asarray(bias_acc, dtype=np.int64)
    g = f_x + dmax + np.asarray(f_w, dtype=np.int64)
    return wrap(acc, acc_bits), g


def matmul_w8a16(x16, f_x: int, wq, f_w, bias_acc=None,
                 acc_bits: int = ACC_BITS) -> Tuple[np.ndarray, np.ndarray]:
    """The W8A16 mode (INT8_PLAN §9.5): an int16 input at one exponent f_x times int8 weights at
    the per-column exponents f_w — one product per DSP48E2 (int16 on the 18-bit port, int8 on the
    27-bit one), no blocks, no packing.  For the layers whose input 8 bits cost too much while
    their weights' bytes are what matters (the LM head: a bandwidth-bound GEMV in decode).
    Returns (acc, g): value = acc * 2^-g, g = f_x + f_w; exact (|x * w| < 2^22, K < 2^31)."""
    x16 = np.asarray(x16, dtype=np.int64)
    wq = np.asarray(wq, dtype=np.int64)
    if x16.shape[-1] != wq.shape[0]:
        raise ValueError(f"K mismatch: x {x16.shape[-1]}, w {wq.shape[0]}")
    acc = np.rint(x16.astype(np.float64) @ wq.astype(np.float64)).astype(np.int64)
    if bias_acc is not None:
        acc = acc + np.asarray(bias_acc, dtype=np.int64)
    return wrap(acc, acc_bits), f_x + np.asarray(f_w, dtype=np.int64)


def bias_raw(bias, g) -> np.ndarray:
    """a float bias at the accumulator's per-channel exponents g (rne)."""
    return rne_float(np.asarray(bias, dtype=np.float64) * np.exp2(np.asarray(g, dtype=np.float64)))


def requant(acc, g, f_out, lo: int = I8_LO, hi: int = I8_HI) -> np.ndarray:
    """a static output: rne_shr(acc, g - f_out) saturated to [lo, hi]
    (int8 by default; int16 -32768 / 32767 where a row op reads it)."""
    return np.clip(rne_shr(acc, np.asarray(g) - np.asarray(f_out)), lo, hi)


def requant_blocked(acc, g, f_out: int, block: int = BLOCK, dmax: int = DELTA_MAX):
    """a blocked int8 output along the output channels (the next linear's
    reduction axis): ``block_round`` straight from the accumulator, one
    rounding per element.  Returns (q, d)."""
    return block_round(acc, np.asarray(g) - f_out, block, dmax)


# ------------------------------------------------------------------ the packed DSP pair
def pack_pair(w0, w1) -> np.ndarray:
    """the 27-bit word of two weights sharing an activation: w0 * 2^FIELD + w1."""
    w0 = np.asarray(w0, dtype=np.int64)
    w1 = np.asarray(w1, dtype=np.int64)
    if np.any((w0 < W8_LO) | (w0 > W8_HI) | (w1 < W8_LO) | (w1 > W8_HI)):
        raise ValueError("packed weights must lie in [-127, 127]")
    return (w0 << FIELD) + w1


def unpack(P) -> Tuple[np.ndarray, np.ndarray]:
    """(S0, S1) from the cascade's P = S0 * 2^FIELD + S1: S1 = sext(P[FIELD-1:0]),
    S0 = (P - S1) >> FIELD; exact while |S1| < 2^(FIELD-1)."""
    P = np.asarray(P, dtype=np.int64)
    S1 = wrap(P, FIELD)
    return (P - S1) >> FIELD, S1


def chain_bound(n: int = CHAIN) -> int:
    """the largest |S1| n products can reach (weights ±127, x in [-128, 127])."""
    return n * W8_HI * -I8_LO


# ------------------------------------------------------------------ unsigned tensors
def to_offset(u) -> np.ndarray:
    """an unsigned tensor u in [0, 255] as stored: u - 128."""
    return np.asarray(u, dtype=np.int64) - U8_OFFSET


def offset_bias(wq, axis: int = 0, dmax: int = DELTA_MAX) -> np.ndarray:
    """what the consumer of an offset (static, d = 0) tensor adds to its
    accumulator, at the accumulator's exponent: (128 * sum_k w[k][n]) << dmax;
    ``axis`` is the reduction axis of wq."""
    return (U8_OFFSET * np.asarray(wq, dtype=np.int64).sum(axis=axis)) << dmax
