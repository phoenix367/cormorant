"""Moonshine on the KV260 int16 datapath: a bit-level emulation of the planned
partition (doc/plans/ASR_PLAN.md) — the specification the scheduler's
frontend will have to match — and the float64 reference with the same
structure (pol None) that records the calibration maxima.

Conventions (llm_study.py / vlm_study.py): a value x is stored as the int16
raw round(x * 2^f) (host and row-ops writes: round half even, saturate) or
floor(acc / 2^8) (kernel MatMuls / convs: exact int64 sums of raw int16
products; the weight of input row i / output column j at f_out[j] + 8 -
f_in[i], rounded once).  Exponents per tensor class and layer, per channel
(per head where a tensor is the operand of q.K^T).  The residual stream is
an int32 DMA tensor at 2^-15 (the row-ops unit's), every add exact.

Encoder (audio -> [T, d]):
  pcm      int16 samples, exponent 15 (exact)
  conv1    127 taps, stride 64, no bias: a kernel MatMul on the polyphase
           frame image [t1][2 x 64] (row-ops FOLD) -> c1
  tanh     host table on c1's raw int16 -> t at 2^-15
  GN       GroupNorm(1 group): host, the global mean / variance of t (exact
           integer sums), per-channel affine -> gn (group_norm)
  conv2/3  ConvKernel convs (the bias in the accumulator: kconv), GELU (erf) on
           the host from an int32 table at 2^-16 -> g2 (int16), and conv3's
           straight into the int32 residual h0 (stem_gelu)
  layers   x = LN(h) (row-ops NORM, no beta), q0 k0 v kernel; RoPE on q0 / k0
           (row-ops ROPE: the first 32 of 36 dims in interleaved pairs,
           permuted into rotate-half order) -> q, k per head; s = q.K^T per
           head (ConvKernel); P = VectorOPKernel's softmax (vectorop_smx,
           scale hd^-1/2) at 2^-p_bits; pv = P.V; o kernel; h += o (exact);
           x2 = LN(h); f = fc1 kernel; a = GELU(f + b1); d = fc2; h += d +
           b2 (the bias rounded once at 2^-15)
  xf       final LN -> int16 (the cross-attention K / V input)
Decoder (one token per step):
  embed    host float32 table -> int32 residual at 2^-15
  self     y = LN(h); q0 k0 v kernel (GEMV); RoPE at the position; K / V
           caches at k / vc; the host p12 softmax (LlmAttnSoftmax) over the
           pos + 1 keys
  cross    y2 = LN(h); cq kernel (per head); s = cq.CK^T over the encoder
           frames (CK / CV kernel outputs from xf, once per utterance), the
           host p12 softmax
  mlp      y3 = LN(h); fc1 -> u (x half) / g (gate half) + bias (row-ops COPY
           with bias); a = SiLU(g) * u (row-ops MULACT); fc2; h += d + b2
  head     yf = LN(h); logits = yf . E^T (tied, kernel GEMV); argmax
Audio longer than MAX_FRAMES encoder frames (24.6 s) is transcribed in equal
chunks (transcribe_long).
"""

from __future__ import annotations

import math
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "chat", "scripts"))
import llm_study as ls  # noqa: E402
import moonshine_np as mn  # noqa: E402
from whisper_np import _erf  # noqa: E402

I16_LO, I16_HI = -32768, 32767
RES_BITS = 15
MARGIN = 2.0


def p2(e):
    return np.power(2.0, np.asarray(e, np.float64))


def gelu(x):
    """GELU (erf) for the float paths (torch's erf, vectorised)."""
    return 0.5 * x * (1.0 + _erf(np.asarray(x, np.float64) / math.sqrt(2.0)))


_ERF = np.vectorize(math.erf, otypes=[np.float64])
_TANH = np.vectorize(math.tanh, otypes=[np.float64])


def gelu_libm(x):
    """GELU (erf) through libm, in the operation order of vectorop_act.act_values (the
    unit's specification) — every table of the int16 path."""
    x = np.asarray(x, np.float64)
    return x * (0.5 * (1.0 + _ERF(x / math.sqrt(2.0))))


_GT = {}


def gelu_table(f):
    """GELU (erf) of every raw int16 at 2^-f (index raw + 32768); f = 8: the
    activation unit's input grid (rounded at 2^-8 it is vectorop_act.kernel_table)."""
    if f not in _GT:
        _GT[f] = gelu_libm((np.arange(65536) - 32768) / 2.0 ** int(f))
    return _GT[f]


_GT16 = {}
GELU_BITS = 16                       # the stem GELU table's fixed point (int32 at 2^-16)


def gelu16_table(f):
    """The stem's GELU (erf) of every raw int16 at 2^-f as int32 at 2^-16 (round half
    even, saturated) — the host op's table (computed here, shipped as data: no libm
    erf on the board)."""
    if f not in _GT16:
        v = np.rint(gelu_libm((np.arange(65536) - 32768) / 2.0 ** int(f)) * 2.0 ** GELU_BITS)
        _GT16[f] = np.clip(v, -2.0 ** 31, 2.0 ** 31 - 1).astype(np.int64)
    return _GT16[f]


def rne_shr(v, k):
    """round half to even of v / 2^k (int64 v; k may be <= 0), elementwise k."""
    v = np.asarray(v, np.int64)
    k = np.broadcast_to(np.asarray(k, np.int64), v.shape)
    up = np.where(k <= 0, v << np.maximum(-k, 0), 0)
    kk = np.maximum(k, 1)
    q = v >> kk
    r = v - (q << kk)
    half = np.int64(1) << (kk - 1)
    rd = q + ((r > half) | ((r == half) & ((q & 1) == 1)))
    return np.where(k <= 0, up, rd)


_TT = {}


def tanh_table(f):
    if f not in _TT:
        _TT[f] = _TANH((np.arange(65536) - 32768) / 2.0 ** int(f))
    return _TT[f]


def rne(x):
    return np.rint(x)                                  # numpy rounds half to even


_SEXP = {}


def sexp_table(f, scale):
    """exp(-k 2^-f scale) for k in [0, 65535] — llm_nodes.sexp_table (libm exp, the
    same operation order)."""
    key = (int(f), float(scale))
    if key not in _SEXP:
        d = 2.0 ** int(f)
        _SEXP[key] = np.array([math.exp(-k / d * scale) for k in range(65536)])
    return _SEXP[key]


def layer_norm_fast(x_raw, f_x, gamma, beta, eps, f_out):
    """rowops_spec.layer_norm, bit for bit, without Python integers per element: S2 from
    16-bit halves summed in int64 (exact), V and rsqrt per row, A and P1 per row in int64
    (checked against the spec in asr_study.py selftest)."""
    R = ls.rowops()
    x = np.asarray(x_raw, np.int64)
    rows, n = x.shape
    G, g, Bq, E = R.norm_params(gamma, n, eps, f_x, beta=beta)
    hi, lo = x >> 16, x & 0xFFFF
    s_hh, s_hl, s_ll = (hi * hi).sum(1), (hi * lo).sum(1), (lo * lo).sum(1)
    S1 = x.sum(1)
    P1 = np.empty_like(x)
    for r in range(rows):
        S2 = (int(s_hh[r]) << 32) + (int(s_hl[r]) << 17) + int(s_ll[r])
        D = n * S2 - int(S1[r]) * int(S1[r]) + E
        Rm, sh = R.rsqrt_bf(D if D > 0 else 1)
        P1[r] = R.mul_shr_floor(n * x[r] - S1[r], Rm, sh - R.P1_BITS)
    return R._out(P1, G, g, f_out, Bq)


# ------------------------------------------------------------------ RoPE layout
def rope_perm(hd, rot):
    """Head dims in the unit's rotate-half order: pair j of the interleaved RoPE
    (dims 2j, 2j + 1) at (j, j + hd / 2); the pass-through dims pair with each other."""
    half = hd // 2
    npair = rot // 2
    order = np.empty(hd, np.int64)
    order[:npair] = np.arange(0, rot, 2)
    order[half: half + npair] = np.arange(1, rot, 2)
    rest = np.arange(rot, hd)
    order[npair:half] = rest[: half - npair]
    order[half + npair:] = rest[half - npair:]
    return order


def rope_half_tables(cfg, heads, positions):
    """float32 cos / sin [T][hd / 2] of the rotate-half pairs in the permuted order
    (llama.rope_tables' scheme: inv_j = float32(1 / theta^(2j / rot)), a =
    float32(float32(pos) * inv_j), cos = float32(cos(double(a))) libm); the
    pass-through pairs cos 1, sin 0 — src/moonshine.py builds the same tables."""
    hd = cfg["hidden_size"] // heads
    rot = int(hd * cfg["partial_rotary_factor"])
    th = float(cfg["rope_theta"])
    inv = np.array([1.0 / (th ** (2 * j / rot)) for j in range(rot // 2)], np.float64).astype(np.float32)
    ang = (np.asarray(positions, np.float32)[:, None] * inv[None, :]).astype(np.float32)
    cs = np.vectorize(math.cos, otypes=[np.float64])(ang.astype(np.float64)).astype(np.float32)
    sn = np.vectorize(math.sin, otypes=[np.float64])(ang.astype(np.float64)).astype(np.float32)
    T = len(positions)
    c = np.ones((T, hd // 2), np.float32)
    s = np.zeros((T, hd // 2), np.float32)
    c[:, :rot // 2], s[:, :rot // 2] = cs, sn
    return c, s


def rope_raw_tables(cfg, heads, positions):
    """int24 cos / sin at 2^-23 [T][hd] (both halves) — the unit's B rows."""
    c, s = rope_half_tables(cfg, heads, positions)
    c2, s2 = np.concatenate([c, c], 1).astype(np.float64), np.concatenate([s, s], 1).astype(np.float64)
    return ls.rowops().rope_tables_raw(c2, s2)


# ------------------------------------------------------------------ the model
LIN_ENC = {"q": ("self_attn.q_proj", "x", "q0"), "k": ("self_attn.k_proj", "x", "k0"),
           "v": ("self_attn.v_proj", "x", "v"), "o": ("self_attn.o_proj", "pv", "o"),
           "f1": ("mlp.fc1", "x2", "f"), "f2": ("mlp.fc2", "a", "d")}
LIN_DEC = {"q": ("self_attn.q_proj", "y", "q0"), "k": ("self_attn.k_proj", "y", "k0"),
           "v": ("self_attn.v_proj", "y", "v"), "o": ("self_attn.o_proj", "pv", "o"),
           "cq": ("encoder_attn.q_proj", "y2", "cq"), "co": ("encoder_attn.o_proj", "cpv", "co"),
           "f1": ("mlp.fc1", "y3", "u"), "f2": ("mlp.fc2", "a", "d")}
CROSS = {"ck": ("encoder_attn.k_proj", "xf", "ck"), "cv": ("encoder_attn.v_proj", "xf", "cv")}
HEAD_CLASSES = {"q", "k", "s", "p", "cq", "ck", "cs", "cp"}   # per head (q.K^T operands, scores, P)


class QM:
    """pol None: float64 (record=True collects the per-channel maxima of every class);
    pol {"bf16": True}: float math, bf16 at every boundary; else the int16 emulation
    with fmt the exponents (make_formats)."""

    def __init__(self, cfg, W, pol=None, fmt=None, record=False):
        self.cfg, self.W, self.pol = cfg, W, pol or {}
        self.bf = bool(self.pol.get("bf16"))
        self.q = pol is not None and not self.bf
        self.fmt = fmt or {}
        self.record = record
        self.exact_cls = set(self.pol.get("exact", ()))      # ablation: classes left unquantised
        self.ch = {}
        self.ms = {}
        self.sat = defaultdict(int)
        self.cnt = defaultdict(int)
        self.accmax = defaultdict(float)
        self.wsat = 0
        d = cfg["hidden_size"]
        self.d = d
        self.H = cfg["encoder_num_attention_heads"]
        self.Hd = cfg["decoder_num_attention_heads"]
        self.hd = d // self.H
        self.ff = cfg["intermediate_size"]
        self.L, self.Ld = cfg["encoder_num_hidden_layers"], cfg["decoder_num_hidden_layers"]
        g = lambda n: np.asarray(W[n], np.float64)  # noqa: E731
        self.w, self.b, self.wlo = {}, {}, {}
        for l in range(self.L):
            for key, (name, ci, co) in LIN_ENC.items():
                self._lin(("e", l, key), g(f"encoder.layers.{l}.{name}.weight").T, ci, co, ("e", l))
                if key in ("f1", "f2"):
                    self.b[("e", l, key)] = g(f"encoder.layers.{l}.{name}.bias")
        for l in range(self.Ld):
            for key, (name, ci, co) in list(LIN_DEC.items()) + list(CROSS.items()):
                self._lin(("d", l, key), g(f"decoder.layers.{l}.{name}.weight").T, ci, co, ("d", l))
                if key in ("f1", "f2"):
                    self.b[("d", l, key)] = g(f"decoder.layers.{l}.{name}.bias")
        w1 = g("encoder.conv1.weight")[:, 0, :]                       # [d, 127]
        w1 = np.concatenate([w1, np.zeros((d, 1))], 1)                 # 128 taps = 2 x 64 phases
        self._lin(("s", 0, "c1"), w1.T, "pcm", "c1", ("s", 0))
        w2 = g("encoder.conv2.weight")                                 # [2d, d, 7]
        self._lin(("s", 0, "c2"), w2.transpose(2, 1, 0).reshape(-1, 2 * d), "gn7", "c2", ("s", 0))
        w3 = g("encoder.conv3.weight")                                 # [d, 2d, 3]
        self._lin(("s", 0, "c3"), w3.transpose(2, 1, 0).reshape(-1, d), "g2x3", "c3", ("s", 0))
        self.b[("s", 0, "c2")], self.b[("s", 0, "c3")] = g("encoder.conv2.bias"), g("encoder.conv3.bias")
        self.emb = g("decoder.embed_tokens.weight")
        self._lin(("h", 0, "lm"), self.emb.T, "yf", "logits", ("h", 0))
        self.perm = rope_perm(self.hd, int(self.hd * cfg["partial_rotary_factor"]))

    # ---- exponents
    def nheads(self, scope):
        return self.H if scope[0] == "e" else self.Hd

    def E(self, cls, scope, n=None):
        """The exponent(s) of class cls in scope (part, layer): an int, or a vector of n
        (a per-head exponent repeated over the head's channels where n = heads * hd)."""
        if cls == "xf":                                           # written by the encoder, read by
            scope = ("e", self.L)                                 # every decoder layer's cross K / V
        key = (cls,) + tuple(scope)
        if cls == "pcm":
            e = 15
        elif cls == "gn7":
            e = np.tile(np.broadcast_to(self.E("gn", ("s", 0)), (self.d,)), 7)
        elif cls == "g2x3":
            e = np.tile(np.broadcast_to(self.E("g2", ("s", 0)), (2 * self.d,)), 3)
        elif cls in ("s", "cs") and key not in self.fmt:          # q.K^T: f_q + f_k - 8 per head
            a, b = ("q", "k") if cls == "s" else ("cq", "ck")
            H = self.nheads(scope)
            e = self.E(a, scope, H) + self.E(b, scope, H) - 8
        elif cls in ("pv", "cpv") and key not in self.fmt:        # P.V: f_p + f_v - 8 per channel
            p, v = ("p", "vc") if cls == "pv" else ("cp", "cv")
            e = np.repeat(self.E(p, scope, self.nheads(scope)), self.hd) + self.E(v, scope, self.d) - 8
        else:
            e = self.fmt.get(key, 8)
        if n is None:
            return e
        e = np.asarray(e, np.int64)
        if e.ndim == 0:
            return np.full(n, int(e))
        if len(e) == n:
            return e
        assert n % len(e) == 0, (cls, scope, len(e), n)
        return np.repeat(e, n // len(e))

    def _lin(self, key, Wt, ci, co, scope):
        Wt = np.ascontiguousarray(Wt, np.float64)
        if not self.q:
            self.w[key] = Wt
            return
        K, N = Wt.shape
        fi, fo = self.E(ci, scope, K), self.E(co, scope, N)
        sc = p2(fo + 8)[None, :] / p2(fi)[:, None]
        if self.pol.get("float_weights"):
            self.w[key] = Wt * sc
            return
        r = np.rint(Wt * sc)
        self.wsat += int(((r < I16_LO) | (r > I16_HI)).sum())
        self.w[key] = np.clip(r, I16_LO, I16_HI)
        k = self.pol.get("wsplit", {}).get(key[2]) if key[0] == "s" else None
        if k:                                        # the remainder at an output 2^k finer: a 2nd call
            self.wlo[key] = (np.rint((Wt * sc - self.w[key]) * 2.0 ** k), k)

    # ---- primitives
    def _rec(self, cls, scope, v):
        """Calibration: per channel max |x| (ch) and sum of squares / rows (ms)."""
        if self.record and v.size:
            k = (cls,) + tuple(scope)
            v2 = v.reshape(-1, v.shape[-1])
            m = np.abs(v2).max(0)
            self.ch[k] = m if k not in self.ch else np.maximum(self.ch[k], m)
            sq = (v2 * v2).sum(0)
            if k in self.ms:
                self.ms[k][0] += sq
                self.ms[k][1] += v2.shape[0]
            else:
                self.ms[k] = [sq, v2.shape[0]]

    def _count(self, cls, r):
        self.cnt[cls] += r.size
        self.sat[cls] += int(((r <= I16_LO) | (r >= I16_HI)).sum())

    def exact(self, cls):
        return not self.q or cls in self.exact_cls

    def host(self, v, cls, scope, e=None):
        """A host / row-ops write: round half even + saturate at 2^-e (per channel)."""
        self._rec(cls, scope, v)
        if self.bf:
            return ls.bf16(v)
        if self.exact(cls):
            return v
        e = self.E(cls, scope, v.shape[-1]) if e is None else e
        r = np.clip(rne(v * p2(e)), I16_LO, I16_HI)
        self._count(cls, r)
        return r / p2(e)

    def kmm(self, x, key, ci, co, scope):
        """A kernel MatMul with a constant weight: x (values at E(ci)) times the encoded weight."""
        W = self.w[key]
        if not self.q:
            y = x @ W
            self._rec(co, scope, y)
            return ls.bf16(y) if self.bf else y
        fi, fo = self.E(ci, scope, W.shape[0]), self.E(co, scope, W.shape[1])
        xr = rne(x * p2(fi))
        y = self._floor(xr @ W, fo, co)
        if key in self.wlo:                          # weight split: + the remainder's call, summed
            Wl, k = self.wlo[key]                    # exactly by the host op that reads both
            y = y + self._floor(xr @ Wl, fo + k, co + "_lo")
        return y

    def kconv(self, x, key, ci, co, scope, bias):
        """A ConvKernel conv with a bias: the kernel seeds the accumulator with the bias
        rounded at the output's exponent (an int16, numeric.encode_conv_weights), one
        floor and one saturation: clip(floor(acc / 2^8) + round(b * 2^f_out))."""
        if not self.q or self.exact(co):
            y = x @ self.w[key] + bias if not self.q else self.kmm(x, key, ci, co, scope) + bias
            if not self.q:
                self._rec(co, scope, y)                 # calibrated with the bias in
            return ls.bf16(y) if self.bf else y
        W = self.w[key]
        fi, fo = self.E(ci, scope, W.shape[0]), self.E(co, scope, W.shape[1])
        acc = rne(x * p2(fi)) @ W
        self.accmax[co] = max(self.accmax[co], float(np.abs(acc).max()) / 65536.0 if acc.size else 0.0)
        braw = np.clip(rne(bias * p2(fo)), I16_LO, I16_HI)
        r = np.clip(np.floor(acc / 256.0) + braw[None, :], I16_LO, I16_HI)
        self._count(co, r)
        return r / p2(fo)

    def group_norm(self, t):
        """GroupNorm(1 group) of the tanh output t [t1][d] (raw int16 at 2^-15): the host
        op's arithmetic — S1, S2 exact integer sums over every valid element, in raw units
        mean = S1 / N, var = S2 / N - mean^2 (double), then per element
        ((raw - mean) / sqrt(var + eps * 2^30)) * gamma[c] + beta[c]."""
        g, b = self.W["encoder.groupnorm.weight"], self.W["encoder.groupnorm.bias"]
        if not self.q or self.exact("t"):
            mu, var = t.mean(), t.var()
            return (t - mu) / np.sqrt(var + 1e-5) * g + b
        raw = np.rint(t * 32768.0).astype(np.int64)
        N = raw.size
        S1, S2 = int(raw.sum()), int((raw * raw).sum())
        mean = float(S1) / N
        var = float(S2) / N - mean * mean
        den = math.sqrt(var + 1e-5 * 2.0 ** 30)
        return ((raw.astype(np.float64) - mean) / den) * g[None, :] + b[None, :]

    def stem_gelu(self, c, ci, co, scope, residual=False):
        """GELU of a stem conv output (bias included) on the host: the int32 table at
        2^-16 of c's raw value (per channel exponent), then round half even to co's
        exponent (int16) or, residual=True, to the int32 residual at 2^-15."""
        if not self.q or self.exact(ci):
            y = gelu(c)
            return self.res(y) if residual else self.host(y, co, scope)
        fc = self.E(ci, scope, c.shape[-1])
        t16 = np.empty(c.shape, np.int64)
        for f in np.unique(fc):
            cols = fc == f
            t16[:, cols] = gelu16_table(int(f))[np.rint(c[:, cols] * 2.0 ** f).astype(np.int64) + 32768]
        if residual:
            self._rec(co, scope, t16 / 2.0 ** GELU_BITS)
            return rne_shr(t16, GELU_BITS - RES_BITS) / 2.0 ** RES_BITS
        fo = self.E(co, scope, c.shape[-1])
        self._rec(co, scope, t16 / 2.0 ** GELU_BITS)
        r = np.clip(rne_shr(t16, GELU_BITS - fo[None, :]), I16_LO, I16_HI)
        self._count(co, r)
        return r / p2(fo)

    def amm(self, a, ea, b, eb, co, scope, eo):
        """A kernel MatMul of two activations (q.K^T, P.V): raw operands at ea / eb
        (the float path records nothing: attention() records per head / whole)."""
        if not self.q:
            y = a @ b
            return ls.bf16(y) if self.bf else y
        return self._floor(rne(a * p2(ea)) @ rne(b * p2(eb)), eo, co)

    def _floor(self, acc, fo, co):
        self.accmax[co] = max(self.accmax[co], float(np.abs(acc).max()) / 65536.0 if acc.size else 0.0)
        if self.exact(co):
            return acc / p2(np.asarray(fo) + 8)
        if acc.size and np.abs(acc).max() >= 2.0 ** 31:            # ap_fixed<32,16> wraps
            acc = np.mod(acc + 2.0 ** 31, 2.0 ** 32) - 2.0 ** 31
        r = np.clip(np.floor(acc / 256.0), I16_LO, I16_HI)
        self._count(co, r)
        return r / p2(fo)

    def res(self, h):
        """The int32 residual at 2^-15 (float32 in the bf16 yardstick, float64 in the reference)."""
        if self.bf:
            return h.astype(np.float32).astype(np.float64)
        if not self.q:
            return h
        return np.clip(rne(h * 2.0 ** RES_BITS), -2.0 ** 31, 2.0 ** 31 - 1) / 2.0 ** RES_BITS

    def bias_res(self, b):
        return ls.rowops().quant_bias(b, RES_BITS) if self.q else b

    def ln(self, h, gamma, cls, scope):
        if not self.q or self.exact(cls):
            mu = h.mean(-1, keepdims=True)
            var = ((h - mu) ** 2).mean(-1, keepdims=True)
            return self.host((h - mu) / np.sqrt(var + 1e-5) * gamma, cls, scope)
        e = self.E(cls, scope, h.shape[-1])
        self._rec(cls, scope, h)
        y = layer_norm_fast(np.rint(h * 2.0 ** RES_BITS).astype(np.int64), RES_BITS, gamma, None, 1e-5, e)
        self._count(cls, y)
        return y / p2(e)

    def gelu_bias(self, f, b, ci, co, scope):
        """GELU(f + b): VectorOPKernel (ADD, MUL, act GELU) where co is at 2^-8 on every channel
        and the policy allows it, else the host (bias added as an integer at f's exponent, table)."""
        if not self.q or self.exact(ci):
            return self.host(gelu(f + b), co, scope)
        ff, fa = self.E(ci, scope, f.shape[-1]), self.E(co, scope, f.shape[-1])
        raw = np.rint(f * p2(ff))
        if self.pol.get("vop_gelu", True) and np.all(fa == 8) and np.all((ff >= 2) & (ff <= 16)):
            half = np.where(ff > 8, np.power(2.0, ff - 9), 0.0)
            ba, sc = np.rint(b * p2(ff)) + half, np.power(2.0, 16 - ff)
            s = np.clip(raw + ba, I16_LO, I16_HI)
            qq = np.clip(np.floor(s * sc / 256.0), I16_LO, I16_HI).astype(np.int64)
            return self.host(gelu_table(8)[qq + 32768], co, scope)
        rb = np.clip(raw + np.rint(b * p2(ff)), I16_LO, I16_HI).astype(np.int64)
        y = np.empty(f.shape)
        for e in np.unique(ff):
            cols = ff == e
            y[..., cols] = gelu_table(int(e))[rb[..., cols] + 32768]
        return self.host(y, co, scope)

    def rope(self, x, positions, H, cls_in, cls, scope):
        """RoPE of a [T, H * hd] kernel output: the row-ops unit's ROPE on the permuted head dims."""
        T = x.shape[0]
        if not self.q or self.exact(cls):
            y = mn.rope(x, H, mn.rope_tables(self.cfg, H, positions))
            return self.host(y, cls, scope)
        pm = self.perm
        fx = self.E(cls_in, scope, H * self.hd).reshape(H, self.hd)[:, pm]
        fo = self.E(cls, scope, H)
        xr = np.rint(x.reshape(T, H, self.hd)[:, :, pm] * p2(fx)[None]).astype(np.int64)
        c, s = rope_raw_tables(self.cfg, H, positions)
        self._rec(cls, scope, mn.rope(x, H, mn.rope_tables(self.cfg, H, positions)))
        y = ls.rowops().rope(xr, fx, c, s, fo)
        self._count(cls, y)
        out = np.empty((T, H, self.hd))
        out[:, :, pm] = y / p2(fo)[None, :, None]
        return out.reshape(T, H * self.hd)

    def softmax(self, s, fs, cls, scope):
        """One head's scores s [Tq, S] -> P at 2^-p_bits: the encoder's on VectorOPKernel's
        unit (vectorop_smx, column mode), a decoder step's on the host — LlmAttnSoftmax's
        p12 softmax: k = raw_max - raw, e = exp(-k 2^-f_s scale) (sexp_table), the sum
        left to right, P = round_half_even(e / sum 2^f_p).  scale = hd^-1/2 (36, not the
        padded head)."""
        scale = self.hd ** -0.5
        if not self.q or self.exact("s"):
            x = s * scale
            e = np.exp(x - x.max(-1, keepdims=True))
            P = e / e.sum(-1, keepdims=True)                   # (P's exponent is p_bits: not recorded)
            return ls.bf16(P) if self.bf else P
        fp = int(self.pol.get("p_bits", 12))
        raw = np.rint(s * p2(fs)).astype(np.int64)
        if scope[0] == "e":
            cm, cs = ls.vsmx().scale_regs(int(fs), scale)
            P = ls.vsmx().softmax_raw(raw, s.shape[-1], cm, cs, fp)
            return P / 2.0 ** fp
        k = raw.max(-1, keepdims=True) - raw
        e = sexp_table(int(fs), scale)[k]
        P = np.clip(np.round(e / np.cumsum(e, -1)[..., -1:] * 2.0 ** fp), I16_LO, I16_HI)
        return P / 2.0 ** fp

    def attention(self, q, k, v, cls_q, cls_k, cls_v, cls_s, cls_p, cls_pv, scope, H):
        hd = self.hd
        out = np.empty((q.shape[0], H * hd))
        fq, fk, fs = self.E(cls_q, scope, H), self.E(cls_k, scope, H), self.E(cls_s, scope, H)
        fv, fpv = self.E(cls_v, scope, H * hd), self.E(cls_pv, scope, H * hd)
        fp = int(self.pol.get("p_bits", 12))
        for h in range(H):
            sl = slice(h * hd, (h + 1) * hd)
            s = self.amm(q[:, sl], fq[h], k[:, sl].T, fk[h], cls_s, scope, fs[h])
            if not self.q:
                self._rec(cls_s, scope + (h,), np.abs(s).max(keepdims=True).reshape(1, 1))
            p = self.softmax(s, fs[h], cls_p, scope)
            out[:, sl] = self.amm(p, fp, v[:, sl], fv[sl], cls_pv, scope, fpv[sl])
        if not self.q:
            self._rec(cls_pv, scope, out)
        return out

    # ---- encoder
    def encode(self, pcm):
        d = self.d
        n = len(pcm)
        x = pcm.astype(np.float64) / 32768.0
        t1 = (n - 127) // 64 + 1
        fr = np.zeros((t1 + 1) * 64)
        fr[:min(n, (t1 + 1) * 64)] = x[:(t1 + 1) * 64]
        fr = fr.reshape(t1 + 1, 64)
        cols = np.concatenate([fr[:-1], fr[1:]], 1)                      # [t1, 128]: the polyphase image
        S = ("s", 0)
        c1 = self.kmm(cols, ("s", 0, "c1"), "pcm", "c1", S)
        if self.q and not self.exact("c1"):
            e = self.E("c1", S, d)
            t = np.empty_like(c1)
            for f in np.unique(e):
                cc = e == f
                t[:, cc] = tanh_table(int(f))[np.rint(c1[:, cc] * 2.0 ** f).astype(np.int64) + 32768]
        else:
            t = np.tanh(c1)
        t = self.host(t, "t", S)
        gn = self.host(self.group_norm(t), "gn", S)
        t2 = (t1 - 7) // 3 + 1
        im = np.concatenate([gn[k: k + 3 * (t2 - 1) + 1: 3] for k in range(7)], 1)
        c2 = self.kconv(im, ("s", 0, "c2"), "gn7", "c2", S, self.b[("s", 0, "c2")])
        g2 = self.stem_gelu(c2, "c2", "g2", S)
        T = (t2 - 3) // 2 + 1
        im = np.concatenate([g2[k: k + 2 * (T - 1) + 1: 2] for k in range(3)], 1)
        c3 = self.kconv(im, ("s", 0, "c3"), "g2x3", "c3", S, self.b[("s", 0, "c3")])
        h = self.stem_gelu(c3, "c3", "g3", S, residual=True)
        pos = np.arange(T)
        W = self.W
        for l in range(self.L):
            sc = ("e", l)
            p = f"encoder.layers.{l}."
            xa = self.ln(h, W[p + "input_layernorm.weight"], "x", sc)
            q = self.rope(self.kmm(xa, ("e", l, "q"), "x", "q0", sc), pos, self.H, "q0", "q", sc)
            k = self.rope(self.kmm(xa, ("e", l, "k"), "x", "k0", sc), pos, self.H, "k0", "k", sc)
            v = self.kmm(xa, ("e", l, "v"), "x", "v", sc)
            v = self.host(v, "vc", sc) if self.q and not np.array_equal(self.E("vc", sc, d), self.E("v", sc, d)) else v
            att = self.attention(q, k, v, "q", "k", "vc", "s", "p", "pv", sc, self.H)
            h = self.res(h + self.kmm(att, ("e", l, "o"), "pv", "o", sc))
            x2 = self.ln(h, W[p + "post_attention_layernorm.weight"], "x2", sc)
            f = self.kmm(x2, ("e", l, "f1"), "x2", "f", sc)
            a = self.gelu_bias(f, self.b[("e", l, "f1")], "f", "a", sc)
            h = self.res(h + self.kmm(a, ("e", l, "f2"), "a", "d", sc) + self.bias_res(self.b[("e", l, "f2")]))
        return self.ln(h, W["encoder.layer_norm.weight"], "xf", ("e", self.L))

    # ---- decoder
    def start(self, xf):
        """Cross K / V of every layer from the encoder output; empty self caches."""
        self.ck = [self.kmm(xf, ("d", l, "ck"), "xf", "ck", ("d", l)) for l in range(self.Ld)]
        self.cv = [self.kmm(xf, ("d", l, "cv"), "xf", "cv", ("d", l)) for l in range(self.Ld)]
        self.kc = [np.zeros((0, self.d)) for _ in range(self.Ld)]
        self.vc = [np.zeros((0, self.d)) for _ in range(self.Ld)]
        self.pos = 0

    def step(self, tok):
        W, d = self.W, self.d
        h = self.res(self.emb[[tok]])
        pos = np.array([self.pos])
        for l in range(self.Ld):
            sc = ("d", l)
            p = f"decoder.layers.{l}."
            y = self.ln(h, W[p + "input_layernorm.weight"], "y", sc)
            q = self.rope(self.kmm(y, ("d", l, "q"), "y", "q0", sc), pos, self.Hd, "q0", "q", sc)
            k = self.rope(self.kmm(y, ("d", l, "k"), "y", "k0", sc), pos, self.Hd, "k0", "k", sc)
            v = self.kmm(y, ("d", l, "v"), "y", "v", sc)
            v = self.host(v, "vc", sc) if self.q and not np.array_equal(self.E("vc", sc, d), self.E("v", sc, d)) else v
            self.kc[l] = np.concatenate([self.kc[l], k])
            self.vc[l] = np.concatenate([self.vc[l], v])
            att = self.attention(q, self.kc[l], self.vc[l], "q", "k", "vc", "s", "p", "pv", sc, self.Hd)
            h = self.res(h + self.kmm(att, ("d", l, "o"), "pv", "o", sc))
            y2 = self.ln(h, W[p + "post_attention_layernorm.weight"], "y2", sc)
            cq = self.kmm(y2, ("d", l, "cq"), "y2", "cq", sc)
            att = self.attention(cq, self.ck[l], self.cv[l], "cq", "ck", "cv", "cs", "cp", "cpv", sc, self.Hd)
            h = self.res(h + self.kmm(att, ("d", l, "co"), "cpv", "co", sc))
            y3 = self.ln(h, W[p + "final_layernorm.weight"], "y3", sc)
            u0 = self.kmm(y3, ("d", l, "f1"), "y3", "u", sc)
            b1 = self.b[("d", l, "f1")]
            a = self.silu_mul(u0, b1, sc)
            h = self.res(h + self.kmm(a, ("d", l, "f2"), "a", "d", sc) + self.bias_res(self.b[("d", l, "f2")]))
        self.pos += 1
        yf = self.ln(h, W["decoder.norm.weight"], "yf", ("h", 0))
        return self.kmm(yf, ("h", 0, "lm"), "yf", "logits", ("h", 0))[0]

    def silu_mul(self, u0, b1, sc):
        """fc1's halves + bias (row-ops COPY with bias -> ub / gb), then SiLU(gate) * x (MULACT)."""
        ff = self.ff
        if not self.q or self.exact("g"):
            z = u0 + b1
            ub, gb = self.host(z[:, :ff], "ub", sc), self.host(z[:, ff:], "gb", sc)
            return self.host(gb / (1.0 + np.exp(-gb)) * ub, "a", sc)
        R = ls.rowops()
        fu = self.E("u", sc, 2 * ff)
        raw = np.rint(u0 * p2(fu)).astype(np.int64)
        fub, fgb, fa = self.E("ub", sc, ff), self.E("gb", sc, ff), self.E("a", sc, ff)
        ub = R.bias_round(raw[:, :ff], fu[:ff], b1[:ff], fub)
        gb = R.bias_round(raw[:, ff:], fu[ff:], b1[ff:], fgb)
        self._count("ub", ub)
        self._count("gb", gb)
        ar = R.silu_mul(gb, fgb, ub, fub, fa)
        self._count("a", ar)
        return ar / p2(fa)


MAX_FRAMES = 1024                     # the largest encode bucket (24.6 s): the softmax unit's
                                      # column mode takes 1024 keys


def max_samples(frames=MAX_FRAMES):
    """The most samples whose encoder output fits `frames` frames (384 per frame + 511)."""
    return 384 * frames + 511


SPLIT_SEARCH = 2 * 16000               # a boundary within +-2 s of the equal split
SPLIT_WIN = 1600                       # ... at the centre of the quietest 0.1 s window


def chunks(pcm):
    """[(start, length)]: audio longer than max_samples() split into the fewest parts,
    each boundary at the centre of the 0.1 s window of least energy (the sum of the
    squared int16 samples, exact) within +-2 s of the equal split — the first such
    window; every part at most max_samples().  Splitting mid-word made greedy decoding
    loop (2026-10-17: 'the principles of' to the token limit).  The library's policy;
    the transcripts are joined."""
    x = np.asarray(pcm).astype(np.int64)
    n, M = len(x), max_samples()
    if n <= M:
        return [(0, n)]
    k = -(-n // M)
    e = np.concatenate([[0], np.cumsum(x * x)])
    bounds = [0]
    for i in range(1, k):
        ideal = (i * n) // k
        lo = max(bounds[-1] + 1, ideal - SPLIT_SEARCH, n - (k - i) * M)
        hi = min(ideal + SPLIT_SEARCH, bounds[-1] + M)
        s0, s1 = max(lo - SPLIT_WIN // 2, 0), max(hi - SPLIT_WIN // 2, 0)
        starts = np.arange(s0, min(s1, n - SPLIT_WIN) + 1)
        if starts.size == 0:
            bounds.append(min(max(ideal, lo), hi))
            continue
        energy = e[starts + SPLIT_WIN] - e[starts]
        b = int(starts[int(np.argmin(energy))]) + SPLIT_WIN // 2
        bounds.append(min(max(b, lo), hi))
    bounds.append(n)
    return [(a, b - a) for a, b in zip(bounds[:-1], bounds[1:])]


def transcribe_long(m, cfg, pcm):
    """Greedy transcript of any length: transcribe() per chunk, the ids joined."""
    ids = []
    for a, L in chunks(pcm):
        ids += transcribe(m, cfg, pcm[a:a + L])[0]
    return ids


def transcribe(m, cfg, pcm, forced=None, xf=None):
    """Greedy (or teacher-forced on `forced`): (ids, per-step argmax); xf: the encoder
    output when already computed."""
    m.start(m.encode(pcm) if xf is None else xf)
    tok = cfg["decoder_start_token_id"]
    ids, am = [], []
    limit = mn.max_tokens(len(pcm))
    steps = (len(forced) + 1) if forced is not None else limit - 1
    for i in range(steps):
        lg = m.step(tok)
        t = int(np.argmax(lg))
        am.append(t)
        if forced is not None:
            if i < len(forced):
                tok = forced[i]
            continue
        if t == cfg["eos_token_id"]:
            break
        ids.append(t)
        tok = t
    return ids, am


# ------------------------------------------------------------------ formats
def fbits(mx, margin=MARGIN, lo=-8, hi=15):
    return ls.fbits(mx, margin, lo, hi)


def kernel_inputs(m):
    """{input tensor key: [(weight key, taps)]}: the constant-weight MatMuls each
    host / row-ops written kernel input feeds (taps: the input's channels repeat
    taps times down the weight's rows — the conv im2col images)."""
    cons = defaultdict(list)
    for l in range(m.L):
        for key, (_, ci, _) in LIN_ENC.items():
            if ci != "pv":
                cons[(ci, "e", l)].append((("e", l, key), 1))
    for l in range(m.Ld):
        for key, (_, ci, _) in LIN_DEC.items():
            if ci not in ("pv", "cpv"):
                cons[(ci, "d", l)].append((("d", l, key), 1))
        for key in CROSS:
            cons[("xf", "e", m.L)].append((("d", l, key), 1))
    cons[("gn", "s", 0)].append((("s", 0, "c2"), 7))
    cons[("g2", "s", 0)].append((("s", 0, "c3"), 3))
    cons[("yf", "h", 0)].append((("h", 0, "lm"), 1))
    return cons


def balanced(m_float, fmt, ch, ms, keys):
    """Per channel of each kernel input in keys, the exponent f that minimises the
    rounding error of its consumers' outputs, a * 4^-f + b * 4^f: a = sum W_ij^2
    (the input's rounding), b = E[x_i^2] * sum_j 4^-(f_out_j + 8) (the weight's rounding
    at f_out_j + 8 - f); at most the saturation bound fbits(max |x_i|)."""
    cons = kernel_inputs(m_float)
    saved, m_float.fmt = m_float.fmt, fmt
    try:
        for key in keys:
            n = len(ch[key])
            a, b = np.zeros(n), np.zeros(n)
            for wkey, taps in cons[key]:
                Wt = m_float.w[wkey].reshape(taps, n, -1)
                part, l, name = wkey
                co = {"e": LIN_ENC, "d": {**LIN_DEC, **CROSS}}.get(part, {}).get(name, (None, None, None))[2] or \
                    {"c2": "c2", "c3": "c3", "lm": "logits"}[name]
                fo = m_float.E(co, (part, l), Wt.shape[2])
                a += (Wt ** 2).sum((0, 2))
                b += taps * np.sum(np.power(4.0, -(fo + 8.0)))
            b *= np.maximum(ms[key], 1e-30)
            f = np.rint(np.log2(np.maximum(a, 1e-30) / b) / 4.0).astype(np.int64)
            fmt[key] = np.clip(f, -8, fbits(ch[key]))
    finally:
        m_float.fmt = saved


def make_formats(m_float, ch, pol, ms=None):
    """Exponents from the float calibration maxima ch[(cls, part, layer)] (per channel):
    one bit of headroom; host-written kernel inputs capped at in_cap; the score, P.V and
    weight-fit constraints of llm_study.make_formats; q / k / cq / ck per head."""
    pol = pol or {}
    in_cap = pol.get("in_cap", 8)
    tensor = pol.get("per_tensor", False)
    fp = int(pol.get("p_bits", 12))
    hd = m_float.hd
    fmt = {}

    margins = pol.get("margins", {})                 # per-class headroom (default MARGIN: one bit)

    def rng(key, cap=None):
        mx = ch[key]
        if tensor:
            mx = np.full_like(mx, mx.max())
        f = fbits(mx, margins.get(key[0], MARGIN))
        return np.minimum(f, cap) if cap is not None else f

    def heads(key, cap, nh):
        mx = ch[key].reshape(nh, -1).max(1)
        f = fbits(np.full_like(mx, mx.max()) if tensor else mx, margins.get(key[0], MARGIN))
        return np.minimum(f, cap) if cap is not None else f

    INPUTS = {"x", "x2", "a", "xf", "y", "y2", "y3", "yf", "gn", "g2", "ub", "gb"}
    for key in ch:
        if key[0] in HEAD_CLASSES or key[0] in ("pv", "cpv") or len(key) > 3:
            continue
        fmt[key] = rng(key, in_cap if key[0] in INPUTS else None)
    fmt[("t", "s", 0)] = 15                                             # tanh: |t| < 1
    for cls, cap in pol.get("caps", {}).items():   # per-class caps: "x", or (class, part) as ("a", "d")
        for key in list(fmt):
            if key[0] == cls or key[:2] == cls:
                fmt[key] = np.minimum(fmt[key], cap)
    for key in list(fmt):
        if key[0] == "gb":
            fmt[key] = np.minimum(fmt[key], 14)                        # MULACT: a gate exponent <= 14
    if pol.get("bal"):
        keys = [k for k in kernel_inputs(m_float) if k in ch]
        if pol.get("bal") == "novop":                                  # VectorOP's GELU outputs stay Q8.8
            keys = [k for k in keys if not (k[0] in ("a", "g2") and k[1] in ("e", "s"))]
        balanced(m_float, fmt, ch, ms, keys)
    if pol.get("vop_gelu", True):
        for key in list(fmt):
            if key[0] in ("a", "g2", "g3") and key[1] in ("e", "s") and np.all(np.asarray(fmt[key]) >= 8):
                fmt[key] = np.full(np.shape(fmt[key]), 8)              # VectorOP GELU writes Q8.8
    for part, n_l in (("e", m_float.L), ("d", m_float.Ld)):
        nh = m_float.nheads((part, 0))
        for l in range(n_l):
            sc = (part, l)
            groups = [("q", "k", "s", "p", "vc", "pv", "v")]
            if part == "d":
                groups.append(("cq", "ck", "cs", "cp", "cv", "cpv", "cv"))
            for cq_, ck_, cs_, cp_, cv_, cpv_, vsrc in groups:
                kernel_q = cq_ == "cq"                                 # cq / ck: kernel outputs, no cap
                fq = heads((cq_,) + sc, None if kernel_q else in_cap, nh)
                fk = heads((ck_,) + sc, None if kernel_q else in_cap, nh)
                smax = np.array([ch[(cs_,) + sc + (h,)].max() for h in range(nh)])
                fmt[(cq_,) + sc] = np.minimum(fq, fbits(smax) + 8 - fk)
                fmt[(ck_,) + sc] = fk
                fmt[(cp_,) + sc] = np.full(nh, fp)
                fpv_max = fbits(ch[(cpv_,) + sc]).reshape(-1)
                fmt[(cv_,) + sc] = np.minimum(rng((vsrc,) + sc, None if kernel_q else in_cap),
                                              fpv_max + 8 - fp)
    # every weight must fit int16 at f_out[j] + 8 - f_in[i]: lower f_out[j] where not
    saved, m_float.fmt = m_float.fmt, fmt
    try:
        names = {"e": LIN_ENC, "d": {**LIN_DEC, **CROSS},
                 "s": {"c1": (None, "pcm", "c1"), "c2": (None, "gn7", "c2"), "c3": (None, "g2x3", "c3")},
                 "h": {"lm": (None, "yf", "logits")}}
        for (part, l, name), Wt in m_float.w.items():
            ci, co = names[part][name][1:]
            sc = (part, l)
            fi = m_float.E(ci, sc, Wt.shape[0])
            with np.errstate(divide="ignore"):
                lim = np.floor(np.log2(32767.0 / np.maximum(np.abs(Wt), 1e-30)))
            fo_max = (lim + fi[:, None] - 8).min(0).astype(np.int64)
            fmt[(co,) + sc] = np.minimum(m_float.E(co, sc, Wt.shape[1]), fo_max)
    finally:
        m_float.fmt = saved
    for co, bname in (("c2", "encoder.conv2.bias"), ("c3", "encoder.conv3.bias")):   # the conv bias is an
        b = np.abs(np.asarray(m_float.W[bname], np.float64))                         # int16 at f_out
        with np.errstate(divide="ignore"):
            lim = np.floor(np.log2(32767.0 / np.maximum(b, 1e-30))).astype(np.int64)
        fmt[(co, "s", 0)] = np.minimum(np.broadcast_to(np.asarray(fmt[(co, "s", 0)]), lim.shape), lim)
    fmt[("logits", "h", 0)] = int(np.min(fmt[("logits", "h", 0)]))  # one exponent: argmax on raw values
    for l in range(m_float.Ld):                                        # cq / ck back to per head
        for c in ("cq", "ck"):
            fmt[(c, "d", l)] = np.asarray(fmt[(c, "d", l)]).reshape(m_float.Hd, hd).min(1)
    return _vec(fmt)


def load_formats(path):
    """A formats JSON (asr_study.py formats; src/moonshine.py reads the same file) ->
    the fmt dict of QM: {(class, part, layer): int or vector}."""
    import json
    return load_formats_dict(json.load(open(path)))


def load_formats_dict(d):
    """load_formats of the parsed JSON ({"exponents": {class@<part><layer>: ...}})."""
    import re
    fmt = {}
    for k, v in d["exponents"].items():
        cls, rest = k.split("@")
        m = re.fullmatch(r"([a-z])(\d+)", rest)
        fmt[(cls, m.group(1), int(m.group(2)))] = v
    return _vec(fmt)


def _vec(fmt):
    out = {}
    for k, v in fmt.items():
        v = np.asarray(v, np.int64)
        out[k] = int(v.flat[0]) if v.size and (v == v.flat[0]).all() else v
    return out
