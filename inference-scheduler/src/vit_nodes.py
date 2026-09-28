"""
Host-CPU ops of a SigLIP-style vision encoder — SmolVLM-256M-Instruct's
(doc/plans/CHAT_PLAN.md §22-§23).  ONNX nodes of the domain ``axi.llm``
emitted by src/vit.py; the emulation of demo/chat/scripts/vlm_study.py
(``VisionModel``, policy pow2+p12) is the specification.  The numeric
contract is src/llm_nodes.py's: an int16 element is read as raw * 2^-f[c]
(exact), every result is written back with nearbyint(v * 2^f[c]) (round half
to even) saturated to int16, float32 host tensors hold float32 values, all
arithmetic is IEEE double without FMA contraction, sums run left to right.

  VitEmbedAdd      pe (int16) -> h (f32) = float32(pe + b0[t][c]); b0 a float32
                   host table: the patch bias with the pixel normalisation folded
                   in, plus the position embedding
  VitResAdd        h (f32), d (int16) -> float32(h + (d + b[c]))
  VitLayerNorm     h (f32) -> y (int16): mu = sum(h) / n, var = sum((h - mu)^2) / n
                   (left to right), y = ((h - mu) / sqrt(var + eps)) * gamma + beta
  VitAttnPrep      q0, k0, v (int16 kernel outputs), the K / V caches (DMA states,
                   group-major [H][C][HD], V interleaved, RAW integers — exponent 0:
                   one pair serves every layer) -> qx: K = k0 + bk rounded at the
                   per-head k exponents, V = v + bv at the per-channel v exponents
                   (attributes k_exp / v_exp; rows 0 .. T-1, T == C), q = q0 + bq
                   rounded at the per-head q exponents and written as every head's
                   q.K^T conv input image (LlmAttnPrep's layout, one head per
                   group); both caches flushed
  VitAttnSoftmax   s_h [C][T] (raw scores at f_s) -> P_h [T][C] (raw at f_p): every
                   key (no mask), k = raw_max - raw, e = T_hi[k >> 8] * T_lo[k & 255]
                   (two 256-entry libm exp tables per score exponent, L1-resident),
                   sum left to right, P = round_half_even(e / sum * 2^f_p)
  VitGelu          f (int16) -> a (int16): r = sat16(raw + round_half_even(b[c] *
                   2^f[c])), a = round_half_even(gelu(r * 2^-f[c]) * 2^f_a[c]), GELU's
                   tanh form through libm exp: y = x - x / (exp(2u) + 1),
                   u = 0.7978845608028654 * (x + 0.044715 * ((x * x) * x)); in C one
                   int16 table of the rounded outputs per (f, f_a) pair
  VitPixelShuffle  xf [n*n][D] -> columns [k0, k0 + K) of the pixel shuffle by s
                   (raw copy): out[I*(n/s) + J][(b*s + a)*D + c] = xf[(I*s + b)*n
                   + J*s + a][c]
  VitSumDequant    p_0 .. p_{m-1} (int16, equal exponents) -> y (f32) =
                   float32((p_0 + p_1) + ...): the connector's K-chunk outputs
                   summed into the image features

The q.K^T / P.V calls are LlmAttnConvNode (src/llm_nodes.py) with a static
key count (no pos / n inputs: keys = C).
"""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, field
from typing import ClassVar, Dict, Optional, Tuple

import numpy as np

from .host_nodes import HostContext, _attrs, _c_float, _resolve
from .llm_nodes import (LLM_DOMAIN, SEXP_EMIN, SEXP_NE, HostTable, LlmNode, RuntimeItem,
                        _attr_ints, _const_array, _f32, _llm_inputs, _require, _st, exp_tag,
                        scale_item)
from .nodes import SchedulerError

GELU_C = 0.7978845608028654          # sqrt(2 / pi), the C helper's double literal
GELU_EMIN, GELU_NE = -32, 80         # _vit_gelu_tab[f - GELU_EMIN] for f in [-32, 48)


def gelu_exp(x: float) -> float:
    """GELU (tanh form) through libm exp in the C helper's operation order."""
    u = GELU_C * (x + 0.044715 * ((x * x) * x))
    try:
        e = math.exp(2.0 * u)
    except OverflowError:
        e = math.inf
    return x - x / (e + 1.0)


_GELU_CACHE: Dict[int, np.ndarray] = {}


def gelu_table(f: int) -> np.ndarray:
    """gelu_exp(r * 2^-f) for every raw int16 r (index r + 32768)."""
    f = int(f)
    if f not in _GELU_CACHE:
        d = 2.0 ** f
        _GELU_CACHE[f] = np.array([gelu_exp((i - 32768) / d) for i in range(65536)])
    return _GELU_CACHE[f]


_SEXP2_CACHE: Dict[Tuple[int, float], np.ndarray] = {}


def sexp2_table(f: int, scale: float) -> np.ndarray:
    """e(k) = T_hi[k >> 8] * T_lo[k & 255] for k in [0, 65535] (vlm_study.sexp2_table):
    T_hi[i] = exp(-(i * 256) / 2^f * scale), T_lo[j] = exp(-j / 2^f * scale)."""
    key = (int(f), float(scale))
    if key not in _SEXP2_CACHE:
        d = 2.0 ** int(f)
        hi = np.array([math.exp(-(i * 256) / d * scale) for i in range(256)])
        lo = np.array([math.exp(-j / d * scale) for j in range(256)])
        k = np.arange(65536)
        _SEXP2_CACHE[key] = hi[k >> 8] * lo[k & 255]
    return _SEXP2_CACHE[key]


def sexp2_item(f: int, scale: float) -> RuntimeItem:
    if not SEXP_EMIN <= int(f) < SEXP_EMIN + SEXP_NE:
        raise SchedulerError(f"score exponent {f} outside [{SEXP_EMIN}, {SEXP_EMIN + SEXP_NE})")
    return RuntimeItem(key=f"vsexp:{int(f)}", decl="", group="vsexp",
                       row=f"{{ {int(f)}, {scale!r} }}")


def gelu_item(fx: int, fa: int) -> RuntimeItem:
    """The int16 GELU output table of input exponent fx and output exponent fa."""
    for f in (fx, fa):
        if not GELU_EMIN <= int(f) < GELU_EMIN + GELU_NE:
            raise SchedulerError(f"GELU exponent {f} outside [{GELU_EMIN}, {GELU_EMIN + GELU_NE})")
    return RuntimeItem(key=f"gelu:{int(fx)}:{int(fa)}", decl="", group="gelu",
                       row=f"{{ {int(fx)}, {int(fa)} }}")


def pixel_shuffle_rows(n: int, s: int) -> np.ndarray:
    """src[r][q] = the xf row feeding output row r, column block q = b*s + a."""
    m = n // s
    src = np.empty((m * m, s * s), np.int64)
    for r in range(m * m):
        ri, rj = divmod(r, m)
        for b in range(s):
            for a in range(s):
                src[r, b * s + a] = (ri * s + b) * n + rj * s + a
    return src


def _vec_name(prefix: str, v: np.ndarray, ctype: str) -> str:
    return f"{prefix}_{zlib.crc32(np.ascontiguousarray(v).tobytes()):08x}_{v.size}"


def float_array_item(prefix: str, v: np.ndarray) -> Tuple[str, RuntimeItem]:
    """A float32 constant vector as a C array (bias, gamma, beta)."""
    v = np.asarray(v, np.float32).reshape(-1)
    name = _vec_name(prefix, v, "float")
    lits = [_c_float(float(x)) for x in v]
    rows = ",\n".join("    " + ", ".join(lits[i:i + 6]) for i in range(0, len(lits), 6))
    return name, RuntimeItem(key=f"vf:{name}",
                             decl=f"static const float {name}[{v.size}] = {{\n{rows}\n}};")


def int_array_item(prefix: str, v: np.ndarray) -> Tuple[str, RuntimeItem]:
    v = np.asarray(v, np.int64).reshape(-1)
    name = _vec_name(prefix, v.astype(np.int32), "int32_t")
    lits = [str(int(x)) for x in v]
    rows = ",\n".join("    " + ", ".join(lits[i:i + 12]) for i in range(0, len(lits), 12))
    return name, RuntimeItem(key=f"vi:{name}",
                             decl=f"static const int32_t {name}[{v.size}] = {{\n{rows}\n}};")


@dataclass
class VitNode(LlmNode):
    helpers: ClassVar[Tuple[str, ...]] = ("llm", "vit")


def _bias(ctx, tensors, node, idx, n, what):
    b = _const_array(ctx, tensors, node.input[idx], node, what).astype(np.float32).reshape(-1)
    _require(b.size == n, node, f"{what} size {b.size} != {n}")
    return b.copy()


# ------------------------------------------------------------------ #
# VitEmbedAdd / VitResAdd / VitLayerNorm                               #
# ------------------------------------------------------------------ #

@dataclass
class VitEmbedAddNode(VitNode):
    """h = float32(pe + b0[t][c]) — b0 a float32 host table [rows][n]."""
    rows:  int = 1
    n:     int = 1
    table: Optional[HostTable] = field(default=None, repr=False)

    @classmethod
    def from_onnx_node(cls, node, tensors, index, align_elems, ctx: HostContext):
        pe = _resolve(tensors, node.input[0], node)
        y = _resolve(tensors, node.output[0], node)
        b0 = _const_array(ctx, tensors, node.input[1], node, "table").astype(np.float32)
        n = int(pe.shape[-1])
        rows = pe.numel // n
        _require(b0.size == rows * n, node, f"table must hold {rows} x {n} values")
        sn = cls(onnx_node=node, inputs=[pe], output=y, index=index, align_elems=align_elems,
                 rows=rows, n=n, F=ctx.frac_bits,
                 table=HostTable("_vit_tab_" + tensors[node.input[1]].c_name, "f32", b0.reshape(-1)))
        sn._want(pe, None, "input")
        sn._want(y, "f32", "output")
        _require(y.numel == pe.numel, node, "shapes")
        return sn

    def tables(self):
        return [self.table]

    def c_runtime(self):
        return super().c_runtime() + [self._scales(self.inputs[0])[2]]

    def describe(self):
        return f"rows={self.rows} n={self.n} + a float32 [{self.rows}][{self.n}] table"

    def c_call(self, ins, out, scratch, direct, dtype):
        s = self._scales(self.inputs[0])[0]
        return [f"vit_embed_add({ins[0]}, {s}, {self.table.name}, {self.rows}u, {self.n}u, {out});"]

    def reference(self, ins, dtype):
        pe = np.asarray(ins[0], np.float64).reshape(self.rows, self.n)
        b0 = np.asarray(self.table.data, np.float32).astype(np.float64).reshape(self.rows, self.n)
        return _f32(pe + b0).reshape(self.output.shape)


@dataclass
class VitResAddNode(VitNode):
    """y = float32(h + (d + b[c]))."""
    rows: int = 1
    n:    int = 1
    bias: Optional[np.ndarray] = field(default=None, repr=False)

    @classmethod
    def from_onnx_node(cls, node, tensors, index, align_elems, ctx: HostContext):
        h = _resolve(tensors, node.input[0], node)
        d = _resolve(tensors, node.input[1], node)
        y = _resolve(tensors, node.output[0], node)
        n = int(h.shape[-1])
        sn = cls(onnx_node=node, inputs=[h, d], output=y, index=index, align_elems=align_elems,
                 rows=h.numel // n, n=n, bias=_bias(ctx, tensors, node, 2, n, "bias"),
                 F=ctx.frac_bits)
        sn._want(h, "f32", "h")
        sn._want(d, None, "delta")
        sn._want(y, "f32", "output")
        _require(d.numel == h.numel == y.numel and d.shape[-1] == n, node, "shapes")
        return sn

    def c_runtime(self):
        return [self._scales(self.inputs[1])[2], float_array_item("_vit_b", self.bias)[1]]

    def describe(self):
        return f"rows={self.rows} n={self.n} (float residual + bias)"

    def c_call(self, ins, out, scratch, direct, dtype):
        s = self._scales(self.inputs[1])[0]
        b = float_array_item("_vit_b", self.bias)[0]
        return [f"vit_resadd({ins[0]}, {ins[1]}, {s}, {b}, {self.rows}u, {self.n}u, {out});"]

    def reference(self, ins, dtype):
        h = np.asarray(ins[0], np.float64).reshape(self.rows, self.n)
        d = np.asarray(ins[1], np.float64).reshape(self.rows, self.n)
        return _f32(h + (d + self.bias.astype(np.float64)[None, :])).reshape(self.output.shape)


@dataclass
class VitLayerNormNode(VitNode):
    """y = ((h - mu) / sqrt(var + eps)) * gamma + beta per row (sums left to right)."""
    rows:  int = 1
    n:     int = 1
    eps:   float = 1e-6
    gamma: Optional[np.ndarray] = field(default=None, repr=False)
    beta:  Optional[np.ndarray] = field(default=None, repr=False)

    @classmethod
    def from_onnx_node(cls, node, tensors, index, align_elems, ctx: HostContext):
        h = _resolve(tensors, node.input[0], node)
        y = _resolve(tensors, node.output[0], node)
        n = int(h.shape[-1])
        a = _attrs(node)
        eps = a.get("axi_eps", 1e-6)
        eps = float(eps.decode() if isinstance(eps, bytes) else eps)
        sn = cls(onnx_node=node, inputs=[h], output=y, index=index, align_elems=align_elems,
                 rows=h.numel // n, n=n, eps=eps,
                 gamma=_bias(ctx, tensors, node, 1, n, "gamma"),
                 beta=_bias(ctx, tensors, node, 2, n, "beta"), F=ctx.frac_bits)
        sn._want(h, "f32", "h")
        sn._want(y, None, "output")
        _require(y.numel == h.numel, node, "shapes")
        return sn

    def c_runtime(self):
        return [float_array_item("_vit_g", self.gamma)[1], float_array_item("_vit_b", self.beta)[1],
                self._scales(self.output)[2]]

    def describe(self):
        return f"rows={self.rows} n={self.n} eps={self.eps:.3g}"

    def c_call(self, ins, out, scratch, direct, dtype):
        g = float_array_item("_vit_g", self.gamma)[0]
        b = float_array_item("_vit_b", self.beta)[0]
        iy = self._scales(self.output)[1]
        return [f"vit_layernorm({ins[0]}, {self.rows}u, {self.n}u, {g}, {b}, "
                f"{self.eps!r}, {iy}, {out});"]

    def reference(self, ins, dtype):
        h = np.asarray(ins[0], np.float64).reshape(self.rows, self.n)
        mu = np.cumsum(h, axis=-1)[:, -1] / float(self.n)
        d = h - mu[:, None]
        var = np.cumsum(d * d, axis=-1)[:, -1] / float(self.n)
        y = d / np.sqrt(var + self.eps)[:, None] * self.gamma.astype(np.float64)[None, :] \
            + self.beta.astype(np.float64)[None, :]
        return _st(dtype, y, self.output.exp_channels(self.F)[None, :]).reshape(self.output.shape)


# ------------------------------------------------------------------ #
# VitAttnPrep / VitAttnSoftmax                                         #
# ------------------------------------------------------------------ #

@dataclass
class VitAttnPrepNode(VitNode):
    """inputs [q0, k0, v, cache_k, cache_v] (+ the constant biases bq, bk, bv);
    output qx: the q.K^T conv input image of every head (logically
    B_h[d][t] = round_half_even((q0[t][h][d] + bq) * 2^f_q[h]))."""
    T:  int = 1
    H:  int = 1
    HD: int = 16
    C:  int = 16
    kw: int = 1
    fq: Optional[np.ndarray] = field(default=None, repr=False)     # [H]
    fk: Optional[np.ndarray] = field(default=None, repr=False)     # [H]
    fv: Optional[np.ndarray] = field(default=None, repr=False)     # [H*HD]
    bq: Optional[np.ndarray] = field(default=None, repr=False)
    bk: Optional[np.ndarray] = field(default=None, repr=False)
    bv: Optional[np.ndarray] = field(default=None, repr=False)

    @classmethod
    def from_onnx_node(cls, node, tensors, index, align_elems, ctx: HostContext):
        ins = _llm_inputs(node, tensors)
        _require(len(ins) == 8, node, "inputs: q0, k0, v, cache_k, cache_v, bq, bk, bv")
        q0, k0, v, ck, cv = ins[:5]
        y = _resolve(tensors, node.output[0], node)
        a = _attrs(node)
        H, HD = int(a["num_heads"]), int(a["head_dim"])
        kw = int(a.get("qk_kw", 1))
        C = int(ck.shape[0])
        D = H * HD
        T = q0.numel // D
        _require(HD % (16 * kw) == 0, node, f"head_dim {HD} % (16 * qk_kw {kw})")
        _require(T == C, node, f"rows {T} must equal the cache rows {C} (every key is valid)")
        _require(q0.numel == k0.numel == v.numel == T * D, node, "q / k / v shapes")
        _require(list(y.shape) == [H, HD, T], node, f"qx must be [{H}][{HD}][{T}]")
        for t, w in ((ck, "cache_k"), (cv, "cache_v")):
            _require(t.is_state and not t.is_host and t.group_layout == (H, HD), node,
                     f"{w} must be a DMA state stored group-major [{H}][{C}][{HD}]")
            _require(not t.exp_channels(ctx.frac_bits).any(), node,
                     f"{w} holds raw integers (exponent 0); the exponents are attributes")
        _require(ck.group_kw == 1, node, "the K cache rows must not be interleaved")
        sn = cls(onnx_node=node, inputs=[q0, k0, v, ck, cv], output=y, index=index,
                 align_elems=align_elems, T=T, H=H, HD=HD, C=C, kw=kw,
                 fq=np.asarray(_attr_ints(a, "q_exp", node, H), np.int64),
                 fk=np.asarray(_attr_ints(a, "k_exp", node, H), np.int64),
                 fv=np.asarray(_attr_ints(a, "v_exp", node, D), np.int64),
                 bq=_bias(ctx, tensors, node, 5, D, "bq"), bk=_bias(ctx, tensors, node, 6, D, "bk"),
                 bv=_bias(ctx, tensors, node, 7, D, "bv"), F=ctx.frac_bits)
        for t, w in ((q0, "q"), (k0, "k"), (v, "v"), (y, "output")):
            sn._want(t, None, w)
        return sn

    @property
    def ck(self):
        return self.inputs[3]

    @property
    def cv(self):
        return self.inputs[4]

    def state_writes(self):
        return [self.ck, self.cv]

    @property
    def fk_channels(self) -> np.ndarray:
        return np.repeat(self.fk, self.HD)

    def c_runtime(self):
        items = [self._scales(t)[2] for t in self.inputs[:3]]
        items += [scale_item(self.fq)[2], scale_item(self.fk_channels)[2], scale_item(self.fv)[2]]
        items += [float_array_item("_vit_b", b)[1] for b in (self.bq, self.bk, self.bv)]
        return items

    def describe(self):
        return (f"T={self.T} heads {self.H} head_dim {self.HD}: K / V caches + biases, q.K^T input "
                f"image kw={self.kw} (FPGA attention)")

    def c_call(self, ins, out, scratch, direct, dtype):
        s = {k: self._scales(t) for k, t in (("q", self.inputs[0]), ("k", self.inputs[1]),
                                             ("v", self.inputs[2]))}
        s["ck"], s["cv"] = scale_item(self.fk_channels), scale_item(self.fv)
        b = [float_array_item("_vit_b", x)[0] for x in (self.bq, self.bk, self.bv)]
        return [
            "{",
            "    vit_prep_t _a;",
            f"    _a.q0 = {ins[0]}; _a.k0 = {ins[1]}; _a.v = {ins[2]};",
            f"    _a.sq = {s['q'][0]}; _a.sk = {s['k'][0]}; _a.sv = {s['v'][0]};",
            f"    _a.bq = {b[0]}; _a.bk = {b[1]}; _a.bv = {b[2]};",
            f"    _a.ck = {ins[3]}; _a.cv = {ins[4]};",
            f"    _a.ick = {s['ck'][1]}; _a.icv = {s['cv'][1]}; _a.iq = {scale_item(self.fq)[1]};",
            f"    _a.qx = {out};",
            f"    _a.T = {self.T}u; _a.H = {self.H}u; _a.HD = {self.HD}u; _a.C = {self.C}u;"
            f" _a.kw = {self.kw}u; _a.VK = {self.cv.group_kw}u;",
            "    vit_attn_prep(&_a);",
            f"    llm_cache_flush({self.ck.c_name}, {self.C}u, {self.H}u, {self.C}u, {self.HD}u);",
            f"    llm_cache_flush({self.cv.c_name}, {self.C}u, {self.H}u, {self.C}u, {self.HD}u);",
            "}",
        ]

    def reference(self, ins, dtype):
        T, H, HD = self.T, self.H, self.HD
        q0 = np.asarray(ins[0], np.float64).reshape(T, H * HD)
        k0 = np.asarray(ins[1], np.float64).reshape(T, H * HD)
        v = np.asarray(ins[2], np.float64).reshape(T, H * HD)
        ck, cv = ins[3], ins[4]                      # logical state arrays (raw), in place
        sk = np.power(2.0, self.fk_channels.astype(np.float64))[None, :]
        sv = np.power(2.0, self.fv.astype(np.float64))[None, :]
        ck[:T] = np.clip(np.round((k0 + self.bk.astype(np.float64)[None, :]) * sk), -32768, 32767)
        cv[:T] = np.clip(np.round((v + self.bv.astype(np.float64)[None, :]) * sv), -32768, 32767)
        s = np.power(2.0, np.repeat(self.fq, HD).astype(np.float64))
        raw = np.clip(np.round((q0 + self.bq.astype(np.float64)[None, :]) * s[None, :]), -32768, 32767)
        return raw.reshape(T, H, HD).transpose(1, 2, 0).copy()        # [H][HD][T]


@dataclass
class VitAttnSoftmaxNode(VitNode):
    """s [C][T] raw scores (at f_s) -> P [T][C] raw at 2^-f_p, every key."""
    T:     int = 1
    C:     int = 16
    scale: float = 1.0
    fs:    int = 8
    fp:    int = 12

    @classmethod
    def from_onnx_node(cls, node, tensors, index, align_elems, ctx: HostContext):
        s = _resolve(tensors, node.input[0], node)
        y = _resolve(tensors, node.output[0], node)
        a = _attrs(node)
        C, T = int(s.shape[0]), int(s.shape[1])
        _require(len(s.shape) == 2 and list(y.shape) == [T, C], node, f"scores [C][T] -> P [{T}][{C}]")
        sn = cls(onnx_node=node, inputs=[s], output=y, index=index, align_elems=align_elems,
                 T=T, C=C, scale=1.0 / math.sqrt(int(a["head_dim"])),
                 fs=_attr_ints(a, "s_exp", node, 1)[0], fp=_attr_ints(a, "p_exp", node, 1)[0],
                 F=ctx.frac_bits)
        sn._want(s, None, "scores")
        sn._want(y, None, "output")
        return sn

    def c_runtime(self):
        return [sexp2_item(self.fs, self.scale)]

    def describe(self):
        return f"{self.T} query columns x {self.C} keys, score exponent {self.fs}, P at 2^-{self.fp}"

    def c_call(self, ins, out, scratch, direct, dtype):
        return [f"vit_attn_softmax({ins[0]}, {self.T}u, {self.C}u, {self.fs}, "
                f"{float(2.0 ** self.fp)!r}, {out});"]

    def reference(self, ins, dtype):
        raw = np.asarray(ins[0], np.float64).T.astype(np.int64)          # [T][C]
        m = raw.max(-1, keepdims=True)
        e = sexp2_table(self.fs, self.scale)[m - raw]
        p = e / np.cumsum(e, -1)[..., -1:]
        return np.clip(np.round(p * 2.0 ** self.fp), -32768, 32767)


# ------------------------------------------------------------------ #
# VitGelu / VitPixelShuffle / VitSumDequant                            #
# ------------------------------------------------------------------ #

@dataclass
class VitGeluNode(VitNode):
    """a = gelu_{f[c]}[sat16(raw + braw[c])], braw = round_half_even(b * 2^f)."""
    rows: int = 1
    n:    int = 1
    braw: Optional[np.ndarray] = field(default=None, repr=False)

    @classmethod
    def from_onnx_node(cls, node, tensors, index, align_elems, ctx: HostContext):
        f = _resolve(tensors, node.input[0], node)
        y = _resolve(tensors, node.output[0], node)
        n = int(f.shape[-1])
        b = _bias(ctx, tensors, node, 1, n, "bias")
        ff = f.exp_channels(ctx.frac_bits)
        braw = np.round(b.astype(np.float64) * np.power(2.0, ff.astype(np.float64))).astype(np.int64)
        _require(np.abs(braw).max(initial=0) < 2 ** 31, node, "bias beyond int32 at the input exponent")
        sn = cls(onnx_node=node, inputs=[f], output=y, index=index, align_elems=align_elems,
                 rows=f.numel // n, n=n, braw=braw, F=ctx.frac_bits)
        sn._want(f, None, "input")
        sn._want(y, None, "output")
        _require(y.numel == f.numel, node, "shapes")
        return sn

    def _pairs(self):
        ff = self.inputs[0].exp_channels(self.F)
        fa = self.output.exp_channels(self.F)
        return sorted(set(zip((int(v) for v in ff), (int(v) for v in fa), strict=True)))

    def c_runtime(self):
        items = [gelu_item(fx, fa) for fx, fa in self._pairs()]
        items.append(self._scales(self.inputs[0])[2])             # _llm_e_<tag> of the input
        items.append(self._scales(self.output)[2])                # ... and of the output
        items.append(int_array_item("_vit_bi", self.braw)[1])
        return items

    def describe(self):
        return (f"rows={self.rows} n={self.n} int16 GELU tables for (input, output) exponents "
                f"{self._pairs()}")

    def c_call(self, ins, out, scratch, direct, dtype):
        ef = "_llm_e_" + exp_tag(self.inputs[0].exp_channels(self.F))
        ea = "_llm_e_" + exp_tag(self.output.exp_channels(self.F))
        bi = int_array_item("_vit_bi", self.braw)[0]
        return [f"vit_gelu({ins[0]}, {ef}, {ea}, {bi}, {self.rows}u, {self.n}u, {out});"]

    def reference(self, ins, dtype):
        ff = self.inputs[0].exp_channels(self.F)
        f = np.asarray(ins[0], np.float64).reshape(self.rows, self.n)
        raw = np.clip(np.rint(f * np.power(2.0, ff.astype(np.float64))[None, :]) + self.braw[None, :],
                      -32768, 32767).astype(np.int64)
        y = np.empty((self.rows, self.n))
        for e in np.unique(ff):
            cols = ff == e
            y[:, cols] = gelu_table(int(e))[raw[:, cols] + 32768]
        return _st(dtype, y, self.output.exp_channels(self.F)[None, :]).reshape(self.output.shape)


@dataclass
class VitPixelShuffleNode(VitNode):
    """Columns [k0, k0 + K) of the pixel shuffle of xf [n*n][D] by s (raw copy)."""
    n:  int = 1
    D:  int = 1
    s:  int = 1
    k0: int = 0
    K:  int = 1

    @classmethod
    def from_onnx_node(cls, node, tensors, index, align_elems, ctx: HostContext):
        x = _resolve(tensors, node.input[0], node)
        y = _resolve(tensors, node.output[0], node)
        a = _attrs(node)
        s, k0 = int(a["scale"]), int(a.get("col0", 0))
        D = int(x.shape[-1])
        n = int(round((x.numel // D) ** 0.5))
        rows, K = y.numel // int(y.shape[-1]), int(y.shape[-1])
        _require(n * n * D == x.numel and n % s == 0, node, "input must be [n*n][D], n % scale == 0")
        _require(rows == (n // s) ** 2 and 0 <= k0 and k0 + K <= D * s * s, node,
                 f"output must be [{(n // s) ** 2}][K] with columns within {D * s * s}")
        fx = np.tile(x.exp_channels(ctx.frac_bits), s * s)[k0:k0 + K]
        _require(np.array_equal(fx, y.exp_channels(ctx.frac_bits)), node,
                 "output exponents must be the input's, tiled over the shuffle")
        sn = cls(onnx_node=node, inputs=[x], output=y, index=index, align_elems=align_elems,
                 n=n, D=D, s=s, k0=k0, K=K, F=ctx.frac_bits)
        sn._want(x, None, "input")
        sn._want(y, None, "output")
        return sn

    def describe(self):
        return f"[{self.n * self.n}][{self.D}] -> columns {self.k0}..{self.k0 + self.K - 1} of the x{self.s} shuffle"

    def c_call(self, ins, out, scratch, direct, dtype):
        return [f"vit_pixel_shuffle({ins[0]}, {self.n}u, {self.D}u, {self.s}u, {self.k0}u, "
                f"{self.K}u, {out});"]

    def reference(self, ins, dtype):
        x = np.asarray(ins[0], np.float64).reshape(self.n * self.n, self.D)
        src = pixel_shuffle_rows(self.n, self.s)                       # [m*m][s*s]
        full = x[src].reshape(src.shape[0], self.s * self.s * self.D)
        return full[:, self.k0:self.k0 + self.K].reshape(self.output.shape).copy()


@dataclass
class VitSumDequantNode(VitNode):
    """y = float32(((p_0 + p_1) + ...) as values), equal exponents."""
    count: int = 1

    @classmethod
    def from_onnx_node(cls, node, tensors, index, align_elems, ctx: HostContext):
        ins = _llm_inputs(node, tensors)
        y = _resolve(tensors, node.output[0], node)
        _require(len(ins) >= 1, node, "at least one input")
        e0 = ins[0].exp_channels(ctx.frac_bits)
        for t in ins:
            _require(t.numel == ins[0].numel and t.shape[-1] == ins[0].shape[-1], node, "shapes")
            _require(np.array_equal(t.exp_channels(ctx.frac_bits), e0), node, "equal exponents")
        sn = cls(onnx_node=node, inputs=list(ins), output=y, index=index, align_elems=align_elems,
                 count=ins[0].numel, F=ctx.frac_bits)
        for t in ins:
            sn._want(t, None, "input")
        sn._want(y, "f32", "output")
        _require(y.numel == ins[0].numel, node, "output numel")
        return sn

    def c_runtime(self):
        return [self._scales(self.inputs[0])[2]]

    def describe(self):
        return f"{len(self.inputs)} x {self.count} values summed"

    def c_call(self, ins, out, scratch, direct, dtype):
        s = self._scales(self.inputs[0])[0]
        n = int(self.inputs[0].shape[-1])
        return ["{",
                f"    const Data_t *_p[{len(ins)}] = {{ {', '.join(ins)} }};",
                f"    vit_sum_dequant(_p, {len(ins)}u, {s}, {self.count}u, {n}u, {out});",
                "}"]

    def reference(self, ins, dtype):
        y = np.asarray(ins[0], np.float64)
        for p in ins[1:]:
            y = y + np.asarray(p, np.float64)
        return _f32(y).reshape(self.output.shape)


VIT_OP_FACTORIES = {
    "VitEmbedAdd":     VitEmbedAddNode.from_onnx_node,
    "VitResAdd":       VitResAddNode.from_onnx_node,
    "VitLayerNorm":    VitLayerNormNode.from_onnx_node,
    "VitAttnPrep":     VitAttnPrepNode.from_onnx_node,
    "VitAttnSoftmax":  VitAttnSoftmaxNode.from_onnx_node,
    "VitGelu":         VitGeluNode.from_onnx_node,
    "VitPixelShuffle": VitPixelShuffleNode.from_onnx_node,
    "VitSumDequant":   VitSumDequantNode.from_onnx_node,
}


VIT_C = r"""/*
 * Vision-encoder host ops (src/vit_nodes.py; vlm_study.py VisionModel, policy
 * pow2+p12): the numeric contract of the llm_* helpers above.
 */
#define VIT_MAX_HD 256u

/* ---- LlmEmbed with image rows: ids rows .. rows + R - 1 -> img[id - rows] ---- */
static void llm_embed_img(const int32_t *ids, unsigned n, const uint16_t *t16, const float *t32,
                          unsigned rows, unsigned d, const float *img, unsigned R, float *h)
{
    unsigned t, i;
    for (t = 0u; t < n; t++) {
        long   k = (long)ids[t];
        float *o = h + (size_t)t * d;
        if (k < 0) k = 0;
        if (k >= (long)(rows + R)) k = (long)(rows + R) - 1;
        if (k >= (long)rows) {
            memcpy(o, img + (size_t)(k - (long)rows) * d, (size_t)d * sizeof(float));
        } else if (t16) {
            const uint16_t *r = t16 + (size_t)k * d;
            for (i = 0u; i < d; i++) {
                uint32_t u = (uint32_t)r[i] << 16;
                memcpy(&o[i], &u, sizeof u);
            }
        } else {
            memcpy(o, t32 + (size_t)k * d, (size_t)d * sizeof(float));
        }
    }
}

/* ---- VitEmbedAdd: h = float32(pe + b0[t][c]) ---- */
typedef struct {
    const Data_t *x;
    const double *sx;
    const float  *b0;
    unsigned      n;
    float        *h;
} vit_emb_t;

static void vit_embed_rows(void *p, unsigned r0, unsigned r1)
{
    const vit_emb_t *a = (const vit_emb_t *)p;
    unsigned r, c;
    for (r = r0; r < r1; r++) {
        const Data_t *x = a->x + (size_t)r * a->n;
        const float  *b = a->b0 + (size_t)r * a->n;
        float        *h = a->h + (size_t)r * a->n;
        for (c = 0u; c < a->n; c++)
            h[c] = (float)(llm_ld(x[c], a->sx[c]) + (double)b[c]);
    }
}

static void vit_embed_add(const Data_t *x, const double *sx, const float *b0, unsigned rows,
                          unsigned n, float *h)
{
    vit_emb_t a;
    a.x = x; a.sx = sx; a.b0 = b0; a.n = n; a.h = h;
    host_parallel(vit_embed_rows, &a, rows, host_row_grain(n), 1u);
}

/* ---- VitResAdd: y = float32(h + (d + b)) ---- */
typedef struct {
    const float  *h;
    const Data_t *x;
    const double *sx;
    const float  *b;
    unsigned      n;
    float        *y;
} vit_res_t;

/* The per-element host loops below go 4 elements per step with no branch
 * in between: the in-order A53 otherwise waits out every convert, multiply
 * and add of one element before the next starts (2.5-3x on the board). */
static inline int16_t vit_st16f(double v, double si)       /* llm_st16, v finite */
{
    return (int16_t)fmin(fmax(nearbyint(v * si), -32768.0), 32767.0);
}

static void vit_resadd_rows(void *p, unsigned r0, unsigned r1)
{
    const vit_res_t *a = (const vit_res_t *)p;
    const double    *sx = a->sx;
    const float     *b = a->b;
    unsigned         r, c;
    for (r = r0; r < r1; r++) {
        const float  *h = a->h + (size_t)r * a->n;
        const Data_t *x = a->x + (size_t)r * a->n;
        float        *y = a->y + (size_t)r * a->n;
        for (c = 0u; c + 4u <= a->n; c += 4u) {
            const double y0 = (double)h[c] + (llm_ld(x[c], sx[c]) + (double)b[c]);
            const double y1 = (double)h[c + 1u] + (llm_ld(x[c + 1u], sx[c + 1u]) + (double)b[c + 1u]);
            const double y2 = (double)h[c + 2u] + (llm_ld(x[c + 2u], sx[c + 2u]) + (double)b[c + 2u]);
            const double y3 = (double)h[c + 3u] + (llm_ld(x[c + 3u], sx[c + 3u]) + (double)b[c + 3u]);
            y[c] = (float)y0; y[c + 1u] = (float)y1; y[c + 2u] = (float)y2; y[c + 3u] = (float)y3;
        }
        for (; c < a->n; c++)
            y[c] = (float)((double)h[c] + (llm_ld(x[c], sx[c]) + (double)b[c]));
    }
}

static void vit_resadd(const float *h, const Data_t *x, const double *sx, const float *b,
                       unsigned rows, unsigned n, float *y)
{
    vit_res_t a;
    a.h = h; a.x = x; a.sx = sx; a.b = b; a.n = n; a.y = y;
    host_parallel(vit_resadd_rows, &a, rows, host_row_grain(n), 1u);
}

/* ---- VitLayerNorm: mu = sum h / n, var = sum (h - mu)^2 / n (left to right),
 *      y = ((h - mu) / sqrt(var + eps)) * gamma + beta.  The quotient is
 *      taken as (h - mu) * (1 / sd), within 3 ulp of the division: the
 *      scaled value moves by < (5 |gamma q| + 2 |y|) 2^(f_y - 53), far
 *      inside 2^-20 for int16 outputs, and an element within 2^-20 of a
 *      rounding tie is redone with the division — the same int16 as the
 *      division everywhere, without a divide per element.  Rows go 4 at a
 *      time (4 independent sum chains, each row still summed left to
 *      right: the in-order A53 otherwise waits out every add). ---- */
typedef struct {
    const float  *h;
    unsigned      n;
    const float  *g, *b;
    double        eps;
    const double *iy;
    Data_t       *y;
} vit_ln_t;

#define VIT_LN_TIE (0.5 - 0x1p-20)

static inline Data_t vit_ln_st(double x, double q, double d, double sd, double g, double b, double iy)
{
    if (!(fabs(x - q) <= VIT_LN_TIE))                   /* near a tie (or NaN): exact */
        return llm_st((d / sd) * g + b, iy);
    return (Data_t)(int16_t)(q > 32767.0 ? 32767.0 : q < -32768.0 ? -32768.0 : q);
}

static void vit_layernorm_rows(void *p, unsigned r0, unsigned r1)
{
    const vit_ln_t *a = (const vit_ln_t *)p;
    const unsigned  n = a->n;
    const double    dn = (double)n;
    unsigned        r = r0, c;
    for (; r + 4u <= r1; r += 4u) {
        const float *h0 = a->h + (size_t)r * n, *h1 = h0 + n, *h2 = h1 + n, *h3 = h2 + n;
        Data_t      *y0 = a->y + (size_t)r * n, *y1 = y0 + n, *y2 = y1 + n, *y3 = y2 + n;
        double       s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0, v0 = 0.0, v1 = 0.0, v2 = 0.0, v3 = 0.0;
        double       m0, m1, m2, m3, sd0, sd1, sd2, sd3, i0, i1, i2, i3;
        for (c = 0u; c < n; c++) {
            s0 += (double)h0[c]; s1 += (double)h1[c]; s2 += (double)h2[c]; s3 += (double)h3[c];
        }
        m0 = s0 / dn; m1 = s1 / dn; m2 = s2 / dn; m3 = s3 / dn;
        for (c = 0u; c < n; c++) {
            const double d0 = (double)h0[c] - m0, d1 = (double)h1[c] - m1;
            const double d2 = (double)h2[c] - m2, d3 = (double)h3[c] - m3;
            v0 += d0 * d0; v1 += d1 * d1; v2 += d2 * d2; v3 += d3 * d3;
        }
        sd0 = sqrt(v0 / dn + a->eps); sd1 = sqrt(v1 / dn + a->eps);
        sd2 = sqrt(v2 / dn + a->eps); sd3 = sqrt(v3 / dn + a->eps);
        i0 = 1.0 / sd0; i1 = 1.0 / sd1; i2 = 1.0 / sd2; i3 = 1.0 / sd3;
        for (c = 0u; c < n; c++) {
            const double g = (double)a->g[c], b = (double)a->b[c], iy = a->iy[c];
            const double d0 = (double)h0[c] - m0, d1 = (double)h1[c] - m1;
            const double d2 = (double)h2[c] - m2, d3 = (double)h3[c] - m3;
            const double x0 = ((d0 * i0) * g + b) * iy, x1 = ((d1 * i1) * g + b) * iy;
            const double x2 = ((d2 * i2) * g + b) * iy, x3 = ((d3 * i3) * g + b) * iy;
            const double q0 = nearbyint(x0), q1 = nearbyint(x1), q2 = nearbyint(x2), q3 = nearbyint(x3);
            if ((fabs(x0 - q0) <= VIT_LN_TIE) & (fabs(x1 - q1) <= VIT_LN_TIE)
                & (fabs(x2 - q2) <= VIT_LN_TIE) & (fabs(x3 - q3) <= VIT_LN_TIE)
                & (fabs(q0) <= 32767.0) & (fabs(q1) <= 32767.0)
                & (fabs(q2) <= 32767.0) & (fabs(q3) <= 32767.0)) {
                y0[c] = (Data_t)(int16_t)q0; y1[c] = (Data_t)(int16_t)q1;
                y2[c] = (Data_t)(int16_t)q2; y3[c] = (Data_t)(int16_t)q3;
            } else {
                y0[c] = vit_ln_st(x0, q0, d0, sd0, g, b, iy);
                y1[c] = vit_ln_st(x1, q1, d1, sd1, g, b, iy);
                y2[c] = vit_ln_st(x2, q2, d2, sd2, g, b, iy);
                y3[c] = vit_ln_st(x3, q3, d3, sd3, g, b, iy);
            }
        }
    }
    for (; r < r1; r++) {
        const float *h = a->h + (size_t)r * n;
        Data_t      *y = a->y + (size_t)r * n;
        double       sum = 0.0, var = 0.0, mu, sd, rsd;
        for (c = 0u; c < n; c++)
            sum += (double)h[c];
        mu = sum / dn;
        for (c = 0u; c < n; c++) {
            const double d = (double)h[c] - mu;
            var += d * d;
        }
        sd = sqrt(var / dn + a->eps);
        rsd = 1.0 / sd;
        for (c = 0u; c < n; c++) {
            const double g = (double)a->g[c], b = (double)a->b[c], iy = a->iy[c];
            const double d = (double)h[c] - mu, x = ((d * rsd) * g + b) * iy;
            y[c] = vit_ln_st(x, nearbyint(x), d, sd, g, b, iy);
        }
    }
}

static void vit_layernorm(const float *h, unsigned rows, unsigned n, const float *g,
                          const float *b, double eps, const double *iy, Data_t *y)
{
    vit_ln_t a;
    a.h = h; a.n = n; a.g = g; a.b = b; a.eps = eps; a.iy = iy; a.y = y;
    host_parallel(vit_layernorm_rows, &a, rows, host_row_grain(n), 1u);
}

/* ---- VitAttnPrep: K / V cache rows t (k0 + bk, v + bv at the cache
 * exponents), q0 + bq at the per-head q exponent -> the q.K^T conv input
 * image of head h, qx_h[c][kw*t + j] = q[t][h][(c/16)*16kw + j*16 + c%16] ---- */
typedef struct {
    const Data_t *q0, *k0, *v;
    const double *sq, *sk, *sv;
    const float  *bq, *bk, *bv;
    int16_t      *ck, *cv;
    const double *ick, *icv, *iq;
    Data_t       *qx;
    unsigned      T, H, HD, C, kw, VK;
} vit_prep_t;

static void vit_prep_kv_rows(void *p, unsigned t0, unsigned t1)
{
    const vit_prep_t *a = (const vit_prep_t *)p;
    const unsigned    HD = a->HD, D = a->H * HD;
    unsigned          t, g, d;
    for (t = t0; t < t1; t++)
        for (g = 0u; g < a->H; g++) {
            const size_t o = (size_t)t * D + (size_t)g * HD;
            int16_t     *kc = a->ck + ((size_t)g * a->C + t) * HD;
            int16_t     *vc = a->cv + llm_vrow(a->C, HD, a->VK, g, t);
            const size_t c0 = (size_t)g * HD;
            for (d = 0u; d + 2u <= HD; d += 2u) {
                const size_t  c = c0 + d;
                const int16_t k0 = vit_st16f(llm_ld(a->k0[o + d], a->sk[c]) + (double)a->bk[c], a->ick[c]);
                const int16_t k1 = vit_st16f(llm_ld(a->k0[o + d + 1u], a->sk[c + 1u]) + (double)a->bk[c + 1u],
                                             a->ick[c + 1u]);
                const int16_t v0 = vit_st16f(llm_ld(a->v[o + d], a->sv[c]) + (double)a->bv[c], a->icv[c]);
                const int16_t v1 = vit_st16f(llm_ld(a->v[o + d + 1u], a->sv[c + 1u]) + (double)a->bv[c + 1u],
                                             a->icv[c + 1u]);
                kc[d] = k0; kc[d + 1u] = k1;
                vc[(size_t)d * a->VK] = v0; vc[(size_t)(d + 1u) * a->VK] = v1;
            }
            for (; d < HD; d++) {
                const size_t c = c0 + d;
                kc[d] = vit_st16f(llm_ld(a->k0[o + d], a->sk[c]) + (double)a->bk[c], a->ick[c]);
                vc[(size_t)d * a->VK] = vit_st16f(llm_ld(a->v[o + d], a->sv[c]) + (double)a->bv[c], a->icv[c]);
            }
        }
}

#define VIT_PREP_TB 16u
static void vit_prep_q_items(void *p, unsigned i0, unsigned i1)
{
    const vit_prep_t *a = (const vit_prep_t *)p;
    const unsigned    HD = a->HD, T = a->T, D = a->H * HD, kw = a->kw;
    const unsigned    plane = T * kw, nb = HD / (16u * kw), ntb = (T + VIT_PREP_TB - 1u) / VIT_PREP_TB;
    unsigned          it, d, b, j, l, u;
    for (it = i0; it < i1; it++) {
        const unsigned h = it / ntb, t0 = (it % ntb) * VIT_PREP_TB;
        const unsigned nt = t0 + VIT_PREP_TB < T ? VIT_PREP_TB : T - t0;
        int16_t        q[VIT_PREP_TB][VIT_MAX_HD];
        Data_t        *xg = a->qx + (size_t)h * HD * T + (size_t)kw * t0;
        for (u = 0u; u < nt; u++) {
            const size_t o = (size_t)(t0 + u) * D + (size_t)h * HD;
            const size_t  c0 = (size_t)h * HD;
            const double  iq = a->iq[h];
            for (d = 0u; d + 4u <= HD; d += 4u) {
                const size_t  c = c0 + d;
                const int16_t y0 = vit_st16f(llm_ld(a->q0[o + d], a->sq[c]) + (double)a->bq[c], iq);
                const int16_t y1 = vit_st16f(llm_ld(a->q0[o + d + 1u], a->sq[c + 1u]) + (double)a->bq[c + 1u], iq);
                const int16_t y2 = vit_st16f(llm_ld(a->q0[o + d + 2u], a->sq[c + 2u]) + (double)a->bq[c + 2u], iq);
                const int16_t y3 = vit_st16f(llm_ld(a->q0[o + d + 3u], a->sq[c + 3u]) + (double)a->bq[c + 3u], iq);
                q[u][d] = y0; q[u][d + 1u] = y1; q[u][d + 2u] = y2; q[u][d + 3u] = y3;
            }
            for (; d < HD; d++)
                q[u][d] = vit_st16f(llm_ld(a->q0[o + d], a->sq[c0 + d]) + (double)a->bq[c0 + d], iq);
        }
        for (b = 0u; b < nb; b++)
            for (l = 0u; l < 16u; l++) {
                Data_t *row = xg + (size_t)(16u * b + l) * plane;
                for (u = 0u; u < nt; u++)
                    for (j = 0u; j < kw; j++)
                        row[kw * u + j] = (Data_t)q[u][b * 16u * kw + j * 16u + l];
            }
    }
}

static void vit_attn_prep(vit_prep_t *a)
{
    host_parallel(vit_prep_kv_rows, a, a->T, 8u, 1u);
    host_parallel(vit_prep_q_items, a, a->H * ((a->T + VIT_PREP_TB - 1u) / VIT_PREP_TB), 1u, 1u);
}

/* ---- VitAttnSoftmax: s [C][T] raw scores (row stride T) -> P [T][C] raw at
 * 2^-f_p (the P.V weight), every key: m = max raw, k = m - raw_j,
 * e_j = T_hi[k >> 8] * T_lo[k & 255] (two 256-entry libm exp tables per score
 * exponent: they stay in L1, where the 65 536-entry table of the text softmax
 * misses), sum left to right, P = round_half_even(e / sum * 2^f_p)
 * (llm_attn_softmax's arithmetic: e * (2^f_p / sum) unless within 1e-7 of a
 * rounding tie).  Work items are blocks of LLM_SMX_CB query columns, read
 * once (the rows ahead prefetched) and transposed into a tile whose row
 * stride is padded (at 2 KB its 32 rows share 4 L1 sets); then 4 columns
 * at a time — 4 independent sum chains, each column still summed left to
 * right.  AArch64: NEON 8 x 8 transposes, max and rounding (frinti; the
 * saturating narrow is the clamp; a block of 8 with a near-tie is redone
 * by the scalar code). ---- */
#if defined(__aarch64__) && defined(__ARM_NEON)
#  include <arm_neon.h>
#  define VIT_SMX_NEON 1
#endif
#if defined(__GNUC__)
#  define VIT_PREFETCH(p) __builtin_prefetch(p)
#else
#  define VIT_PREFETCH(p) ((void)(p))
#endif
#define VIT_SMX_TP     40u      /* tile row padding (int16) */
#define VIT_SEXP_EMIN  (SEXP_EMIN_VALUE)
#define VIT_SEXP_NE    (SEXP_NE_VALUE)
static double *_vit_sexp_tab[VIT_SEXP_NE];      /* T_hi[256] then T_lo[256] */

static int vit_sexp_table(int f, double scale)
{
    double **t = &_vit_sexp_tab[f - VIT_SEXP_EMIN];
    double   d = ldexp(1.0, f);
    unsigned i;
    if (*t) return 0;
    *t = (double *)malloc(512u * sizeof(double));
    if (!*t) return -1;
    for (i = 0u; i < 256u; i++) {
        (*t)[i] = exp(-(double)(i * 256u) / d * scale);
        (*t)[256u + i] = exp(-(double)i / d * scale);
    }
    return 0;
}

static void vit_sexp_free(int f)
{
    free(_vit_sexp_tab[f - VIT_SEXP_EMIN]);
    _vit_sexp_tab[f - VIT_SEXP_EMIN] = NULL;
}

typedef struct {
    const Data_t *s;
    Data_t       *p;
    const double *tab;              /* T_hi[256], T_lo[256] */
    double        ip;
    unsigned      T, C, nblk;
} vit_smx_t;

static inline Data_t vit_smx_rnd(double e, double rinv, double sum, double ip)
{
    const double q = e * rinv, r = nearbyint(q);
    if (fabs(q - r) > 0.5 - 1e-7)
        return llm_st(e / sum, ip);                     /* near a tie: exact */
    return (Data_t)(int16_t)(r > 32767.0 ? 32767.0 : r);
}

static inline int vit_smx_max(const int16_t *r, unsigned n)
{
    unsigned j = 1u;
    int      m = r[0];
#ifdef VIT_SMX_NEON
    if (n >= 8u) {
        int16x8_t v = vld1q_s16(r);
        for (j = 8u; j + 8u <= n; j += 8u)
            v = vmaxq_s16(v, vld1q_s16(r + j));
        m = vmaxvq_s16(v);
    }
#endif
    for (; j < n; j++)
        m = r[j] > m ? r[j] : m;
    return m;
}

static void vit_smx_round(const double *e, double sum, double ip, unsigned n, Data_t *p)
{
    const double rinv = ip / sum;
    unsigned     j = 0u, u;
#ifdef VIT_SMX_NEON
    const float64x2_t ri = vdupq_n_f64(rinv), th = vdupq_n_f64(0.5 - 1e-7);
    for (; j + 8u <= n; j += 8u) {
        const float64x2_t q0 = vmulq_f64(vld1q_f64(e + j), ri), q1 = vmulq_f64(vld1q_f64(e + j + 2u), ri);
        const float64x2_t q2 = vmulq_f64(vld1q_f64(e + j + 4u), ri), q3 = vmulq_f64(vld1q_f64(e + j + 6u), ri);
        const float64x2_t r0 = vrndiq_f64(q0), r1 = vrndiq_f64(q1), r2 = vrndiq_f64(q2), r3 = vrndiq_f64(q3);
        const uint64x2_t  t = vorrq_u64(vorrq_u64(vcgtq_f64(vabdq_f64(q0, r0), th), vcgtq_f64(vabdq_f64(q1, r1), th)),
                                        vorrq_u64(vcgtq_f64(vabdq_f64(q2, r2), th), vcgtq_f64(vabdq_f64(q3, r3), th)));
        if (vmaxvq_u32(vreinterpretq_u32_u64(t))) {
            for (u = 0u; u < 8u; u++)
                p[j + u] = vit_smx_rnd(e[j + u], rinv, sum, ip);
            continue;
        }
        vst1q_s16((int16_t *)(void *)(p + j),
                  vcombine_s16(vqmovn_s32(vcombine_s32(vqmovn_s64(vcvtq_s64_f64(r0)), vqmovn_s64(vcvtq_s64_f64(r1)))),
                               vqmovn_s32(vcombine_s32(vqmovn_s64(vcvtq_s64_f64(r2)), vqmovn_s64(vcvtq_s64_f64(r3))))));
    }
#endif
    (void)u;
    for (; j < n; j++)
        p[j] = vit_smx_rnd(e[j], rinv, sum, ip);
}

#ifdef VIT_SMX_NEON
/* 8 rows of 8 int16 (row stride ss) -> 8 rows of 8 (stride ds), transposed */
static inline void vit_tr8(const int16_t *s, size_t ss, int16_t *d, size_t ds)
{
    const int16x8x2_t a0 = vtrnq_s16(vld1q_s16(s), vld1q_s16(s + ss));
    const int16x8x2_t a1 = vtrnq_s16(vld1q_s16(s + 2u * ss), vld1q_s16(s + 3u * ss));
    const int16x8x2_t a2 = vtrnq_s16(vld1q_s16(s + 4u * ss), vld1q_s16(s + 5u * ss));
    const int16x8x2_t a3 = vtrnq_s16(vld1q_s16(s + 6u * ss), vld1q_s16(s + 7u * ss));
    const int32x4x2_t b0 = vtrnq_s32(vreinterpretq_s32_s16(a0.val[0]), vreinterpretq_s32_s16(a1.val[0]));
    const int32x4x2_t b1 = vtrnq_s32(vreinterpretq_s32_s16(a0.val[1]), vreinterpretq_s32_s16(a1.val[1]));
    const int32x4x2_t b2 = vtrnq_s32(vreinterpretq_s32_s16(a2.val[0]), vreinterpretq_s32_s16(a3.val[0]));
    const int32x4x2_t b3 = vtrnq_s32(vreinterpretq_s32_s16(a2.val[1]), vreinterpretq_s32_s16(a3.val[1]));
    const int64x2_t   c0 = vreinterpretq_s64_s32(b0.val[0]), c1 = vreinterpretq_s64_s32(b1.val[0]);
    const int64x2_t   c2 = vreinterpretq_s64_s32(b0.val[1]), c3 = vreinterpretq_s64_s32(b1.val[1]);
    const int64x2_t   c4 = vreinterpretq_s64_s32(b2.val[0]), c5 = vreinterpretq_s64_s32(b3.val[0]);
    const int64x2_t   c6 = vreinterpretq_s64_s32(b2.val[1]), c7 = vreinterpretq_s64_s32(b3.val[1]);
    vst1q_s16(d,          vreinterpretq_s16_s64(vzip1q_s64(c0, c4)));
    vst1q_s16(d + ds,     vreinterpretq_s16_s64(vzip1q_s64(c1, c5)));
    vst1q_s16(d + 2u * ds, vreinterpretq_s16_s64(vzip1q_s64(c2, c6)));
    vst1q_s16(d + 3u * ds, vreinterpretq_s16_s64(vzip1q_s64(c3, c7)));
    vst1q_s16(d + 4u * ds, vreinterpretq_s16_s64(vzip2q_s64(c0, c4)));
    vst1q_s16(d + 5u * ds, vreinterpretq_s16_s64(vzip2q_s64(c1, c5)));
    vst1q_s16(d + 6u * ds, vreinterpretq_s16_s64(vzip2q_s64(c2, c6)));
    vst1q_s16(d + 7u * ds, vreinterpretq_s16_s64(vzip2q_s64(c3, c7)));
}
#endif

static void vit_smx_items(void *pp, unsigned i0, unsigned i1)
{
    const vit_smx_t *a = (const vit_smx_t *)pp;
    const unsigned   T = a->T, C = a->C, TS = C + VIT_SMX_TP;
    const int16_t   *s = (const int16_t *)(const void *)a->s;
    const double    *hi = a->tab, *lo = a->tab + 256;
    unsigned         it;
    int16_t         *tl = (int16_t *)malloc((size_t)LLM_SMX_CB * TS * sizeof(int16_t));
    double          *eb = (double *)malloc((size_t)4u * C * sizeof(double));
    if (!tl || !eb) {
        free(tl); free(eb);
        return;
    }
    for (it = i0; it < i1; it++) {
        const unsigned c0 = it * LLM_SMX_CB, nc = c0 + LLM_SMX_CB < T ? LLM_SMX_CB : T - c0;
        unsigned       k, j = 0u;
#ifdef VIT_SMX_NEON
        if (nc == LLM_SMX_CB)
            for (; j + 8u <= C; j += 8u) {
                const int16_t *row = s + (size_t)j * T + c0;
                unsigned       u;
                if (j + 32u <= C)
                    for (u = 0u; u < 8u; u++)
                        VIT_PREFETCH(row + (size_t)(24u + u) * T);
                for (u = 0u; u < LLM_SMX_CB; u += 8u)
                    vit_tr8(row + u, T, tl + (size_t)u * TS + j, TS);
            }
#endif
        for (; j < C; j++) {
            const int16_t *row = s + (size_t)j * T + c0;
            if (j + 16u < C)
                VIT_PREFETCH(row + (size_t)16u * T);
            for (k = 0u; k < nc; k++)
                tl[(size_t)k * TS + j] = row[k];
        }
        for (k = 0u; k + 4u <= nc; k += 4u) {
            const int16_t *r0 = tl + (size_t)k * TS, *r1 = r0 + TS, *r2 = r1 + TS, *r3 = r2 + TS;
            double        *e0 = eb, *e1 = eb + C, *e2 = eb + 2u * C, *e3 = eb + 3u * C;
            const int      m0 = vit_smx_max(r0, C), m1 = vit_smx_max(r1, C);
            const int      m2 = vit_smx_max(r2, C), m3 = vit_smx_max(r3, C);
            double         s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
            Data_t        *p0 = a->p + (size_t)(c0 + k) * C;
            for (j = 0u; j < C; j++) {
                const unsigned k0 = (unsigned)(m0 - r0[j]), k1 = (unsigned)(m1 - r1[j]);
                const unsigned k2 = (unsigned)(m2 - r2[j]), k3 = (unsigned)(m3 - r3[j]);
                e0[j] = hi[k0 >> 8] * lo[k0 & 255u];
                e1[j] = hi[k1 >> 8] * lo[k1 & 255u];
                e2[j] = hi[k2 >> 8] * lo[k2 & 255u];
                e3[j] = hi[k3 >> 8] * lo[k3 & 255u];
                s0 += e0[j]; s1 += e1[j]; s2 += e2[j]; s3 += e3[j];
            }
            vit_smx_round(e0, s0, a->ip, C, p0);
            vit_smx_round(e1, s1, a->ip, C, p0 + C);
            vit_smx_round(e2, s2, a->ip, C, p0 + 2u * C);
            vit_smx_round(e3, s3, a->ip, C, p0 + 3u * C);
        }
        for (; k < nc; k++) {
            const int16_t *r = tl + (size_t)k * TS;
            const int      m = vit_smx_max(r, C);
            double         sum = 0.0;
            for (j = 0u; j < C; j++) {
                const unsigned kk = (unsigned)(m - r[j]);
                eb[j] = hi[kk >> 8] * lo[kk & 255u];
                sum += eb[j];
            }
            vit_smx_round(eb, sum, a->ip, C, a->p + (size_t)(c0 + k) * C);
        }
    }
    free(tl);
    free(eb);
}

static void vit_attn_softmax(const Data_t *s, unsigned T, unsigned C, int fs, double ip, Data_t *p)
{
    vit_smx_t a;
    a.s = s; a.p = p; a.tab = _vit_sexp_tab[fs - VIT_SEXP_EMIN]; a.ip = ip; a.T = T; a.C = C;
    a.nblk = (T + LLM_SMX_CB - 1u) / LLM_SMX_CB;
    host_parallel(vit_smx_items, &a, a.nblk, 1u, 1u);
}

/* ---- VitGelu: a = round_half_even(gelu(r * 2^-fx) * 2^fa), r = sat16(raw +
 * braw[c]) (GELU tanh form via libm exp: y = x - x / (exp(2u) + 1)).  One
 * int16 table of the ROUNDED outputs per (fx, fa) exponent pair (128 KB,
 * filled at init with the same double arithmetic and llm_st rounding): a
 * lookup per element, the tables of a layer stay in L2. ---- */
#define VIT_GELU_EMIN  (GELU_EMIN_VALUE)
#define VIT_GELU_NE    (GELU_NE_VALUE)
static int16_t *_vit_gelu_tab[VIT_GELU_NE][VIT_GELU_NE];

typedef struct {
    int16_t *t;
    double   d, ia;
} vit_gelu_fill_t;

static void vit_gelu_fill(void *p, unsigned i0, unsigned i1)
{
    const vit_gelu_fill_t *a = (const vit_gelu_fill_t *)p;
    unsigned i;
    for (i = i0; i < i1; i++) {
        const double x = (double)((int)i - 32768) / a->d;
        const double u = 0.7978845608028654 * (x + 0.044715 * ((x * x) * x));
        a->t[i] = llm_st16(x - x / (exp(2.0 * u) + 1.0), a->ia);
    }
}

static int vit_gelu_table(int fx, int fa)
{
    vit_gelu_fill_t a;
    int16_t       **t = &_vit_gelu_tab[fx - VIT_GELU_EMIN][fa - VIT_GELU_EMIN];
    if (*t) return 0;
    *t = (int16_t *)malloc(65536u * sizeof(int16_t));
    if (!*t) return -1;
    a.t = *t; a.d = ldexp(1.0, fx); a.ia = ldexp(1.0, fa);
    host_parallel(vit_gelu_fill, &a, 65536u, 1024u, 8u);
    return 0;
}

static void vit_gelu_free(int fx, int fa)
{
    free(_vit_gelu_tab[fx - VIT_GELU_EMIN][fa - VIT_GELU_EMIN]);
    _vit_gelu_tab[fx - VIT_GELU_EMIN][fa - VIT_GELU_EMIN] = NULL;
}

typedef struct {
    const Data_t          *x;
    const int32_t         *bi;
    const int16_t *const  *tab;     /* per channel: the table of (fx[c], fa[c]) */
    unsigned               n;
    Data_t                *y;
} vit_gelu_t;

static inline int32_t vit_gelu_ix(Data_t x, int32_t bi)
{
    const int32_t v = (int32_t)(int16_t)x + bi;
    return (v > 32767 ? 32767 : v < -32768 ? -32768 : v) + 32768;
}

/* 8 lookups issued before their stores: the table reads (L2) overlap */
static void vit_gelu_rows(void *p, unsigned r0, unsigned r1)
{
    const vit_gelu_t     *a = (const vit_gelu_t *)p;
    const int16_t *const *t = a->tab;
    const int32_t        *bi = a->bi;
    unsigned              r, c;
    for (r = r0; r < r1; r++) {
        const Data_t *x = a->x + (size_t)r * a->n;
        Data_t       *y = a->y + (size_t)r * a->n;
        for (c = 0u; c + 8u <= a->n; c += 8u) {
            const int16_t y0 = t[c][vit_gelu_ix(x[c], bi[c])];
            const int16_t y1 = t[c + 1u][vit_gelu_ix(x[c + 1u], bi[c + 1u])];
            const int16_t y2 = t[c + 2u][vit_gelu_ix(x[c + 2u], bi[c + 2u])];
            const int16_t y3 = t[c + 3u][vit_gelu_ix(x[c + 3u], bi[c + 3u])];
            const int16_t y4 = t[c + 4u][vit_gelu_ix(x[c + 4u], bi[c + 4u])];
            const int16_t y5 = t[c + 5u][vit_gelu_ix(x[c + 5u], bi[c + 5u])];
            const int16_t y6 = t[c + 6u][vit_gelu_ix(x[c + 6u], bi[c + 6u])];
            const int16_t y7 = t[c + 7u][vit_gelu_ix(x[c + 7u], bi[c + 7u])];
            y[c] = (Data_t)y0; y[c + 1u] = (Data_t)y1; y[c + 2u] = (Data_t)y2; y[c + 3u] = (Data_t)y3;
            y[c + 4u] = (Data_t)y4; y[c + 5u] = (Data_t)y5; y[c + 6u] = (Data_t)y6; y[c + 7u] = (Data_t)y7;
        }
        for (; c < a->n; c++)
            y[c] = (Data_t)t[c][vit_gelu_ix(x[c], bi[c])];
    }
}

static void vit_gelu(const Data_t *x, const signed char *fx, const signed char *fa,
                     const int32_t *bi, unsigned rows, unsigned n, Data_t *y)
{
    vit_gelu_t      a;
    const int16_t **tab = (const int16_t **)malloc((size_t)n * sizeof(*tab));
    unsigned        c;
    if (!tab)
        return;
    for (c = 0u; c < n; c++)
        tab[c] = _vit_gelu_tab[fx[c] - VIT_GELU_EMIN][fa[c] - VIT_GELU_EMIN];
    a.x = x; a.bi = bi; a.tab = tab; a.n = n; a.y = y;
    host_parallel(vit_gelu_rows, &a, rows, host_row_grain(n), 1u);
    free(tab);
}

/* ---- VitPixelShuffle: columns [k0, k0 + K) of out[I*m + J][(b*s + a)*D + c]
 * = x[(I*s + b)*n + J*s + a][c], m = n / s (raw copy) ---- */
static void vit_pixel_shuffle(const Data_t *x, unsigned n, unsigned D, unsigned s, unsigned k0,
                              unsigned K, Data_t *y)
{
    const unsigned m = n / s;
    unsigned       r, col;
    for (r = 0u; r < m * m; r++) {
        const unsigned I = r / m, J = r % m;
        for (col = k0; col < k0 + K; ) {
            const unsigned q = col / D, c = col % D, b = q / s, a = q % s;
            const unsigned len = (D - c) < (k0 + K - col) ? (D - c) : (k0 + K - col);
            memcpy(y + (size_t)r * K + (col - k0),
                   x + ((size_t)(I * s + b) * n + J * s + a) * D + c, (size_t)len * sizeof(Data_t));
            col += len;
        }
    }
}

/* ---- VitSumDequant: y = float32(((p0 + p1) + ...) as values) ---- */
static void vit_sum_dequant(const Data_t *const *p, unsigned m, const double *sx, unsigned count,
                            unsigned n, float *y)
{
    unsigned i, k;
    for (i = 0u; i < count; i++) {
        double v = llm_ld(p[0][i], sx[i % n]);
        for (k = 1u; k < m; k++)
            v += llm_ld(p[k][i], sx[i % n]);
        y[i] = (float)v;
    }
}
"""


def vit_c_helpers() -> str:
    return (VIT_C.replace("GELU_EMIN_VALUE", str(GELU_EMIN)).replace("GELU_NE_VALUE", str(GELU_NE))
            .replace("SEXP_EMIN_VALUE", str(SEXP_EMIN)).replace("SEXP_NE_VALUE", str(SEXP_NE)))


__all__ = ("VIT_OP_FACTORIES", "VitNode", "VitEmbedAddNode", "VitResAddNode", "VitLayerNormNode",
           "VitAttnPrepNode", "VitAttnSoftmaxNode", "VitGeluNode", "VitPixelShuffleNode",
           "VitSumDequantNode", "vit_c_helpers", "gelu_table", "gelu_exp", "pixel_shuffle_rows",
           "LLM_DOMAIN", "SEXP_EMIN", "SEXP_NE")
