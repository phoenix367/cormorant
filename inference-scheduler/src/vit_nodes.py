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
                   key (no mask), k = raw_max - raw, e = sexp_{f_s}[k], sum left to
                   right, P = round_half_even(e / sum * 2^f_p)
  VitGelu          f (int16) -> a (int16): r = sat16(raw + round_half_even(b[c] *
                   2^f[c])), a = gelu_{f[c]}[r] — a 65 536-entry table per exponent,
                   GELU's tanh form through libm exp: y = x - x / (exp(2u) + 1),
                   u = 0.7978845608028654 * (x + 0.044715 * ((x * x) * x))
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
                        scale_item, sexp_item, sexp_table)
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


def gelu_item(f: int) -> RuntimeItem:
    if not GELU_EMIN <= int(f) < GELU_EMIN + GELU_NE:
        raise SchedulerError(f"GELU input exponent {f} outside [{GELU_EMIN}, {GELU_EMIN + GELU_NE})")
    return RuntimeItem(key=f"gelu:{int(f)}", decl="", group="gelu", row=f"{{ {int(f)} }}")


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
        return [sexp_item(self.fs, self.scale)]

    def describe(self):
        return f"{self.T} query columns x {self.C} keys, score exponent {self.fs}, P at 2^-{self.fp}"

    def c_call(self, ins, out, scratch, direct, dtype):
        return [f"vit_attn_softmax({ins[0]}, {self.T}u, {self.C}u, {self.fs}, "
                f"{float(2.0 ** self.fp)!r}, {out});"]

    def reference(self, ins, dtype):
        raw = np.asarray(ins[0], np.float64).T.astype(np.int64)          # [T][C]
        m = raw.max(-1, keepdims=True)
        e = sexp_table(self.fs, self.scale)[m - raw]
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

    def c_runtime(self):
        ff = self.inputs[0].exp_channels(self.F)
        items = [gelu_item(int(e)) for e in sorted(set(int(v) for v in ff))]
        items.append(self._scales(self.inputs[0])[2])             # _llm_e_<tag> of the input
        items.append(self._scales(self.output)[2])
        items.append(int_array_item("_vit_bi", self.braw)[1])
        return items

    def describe(self):
        ff = sorted(set(int(v) for v in self.inputs[0].exp_channels(self.F)))
        return f"rows={self.rows} n={self.n} GELU tables for input exponents {ff}"

    def c_call(self, ins, out, scratch, direct, dtype):
        ef = "_llm_e_" + exp_tag(self.inputs[0].exp_channels(self.F))
        ia = self._scales(self.output)[1]
        bi = int_array_item("_vit_bi", self.braw)[0]
        return [f"vit_gelu({ins[0]}, {ef}, {bi}, {ia}, {self.rows}u, {self.n}u, {out});"]

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

static void vit_resadd_rows(void *p, unsigned r0, unsigned r1)
{
    const vit_res_t *a = (const vit_res_t *)p;
    unsigned r, c;
    for (r = r0; r < r1; r++) {
        const float  *h = a->h + (size_t)r * a->n;
        const Data_t *x = a->x + (size_t)r * a->n;
        float        *y = a->y + (size_t)r * a->n;
        for (c = 0u; c < a->n; c++)
            y[c] = (float)((double)h[c] + (llm_ld(x[c], a->sx[c]) + (double)a->b[c]));
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
 *      y = ((h - mu) / sqrt(var + eps)) * gamma + beta ---- */
typedef struct {
    const float  *h;
    unsigned      n;
    const float  *g, *b;
    double        eps;
    const double *iy;
    Data_t       *y;
} vit_ln_t;

static void vit_layernorm_rows(void *p, unsigned r0, unsigned r1)
{
    const vit_ln_t *a = (const vit_ln_t *)p;
    unsigned r, c;
    for (r = r0; r < r1; r++) {
        const float *h = a->h + (size_t)r * a->n;
        Data_t      *y = a->y + (size_t)r * a->n;
        double       sum = 0.0, var = 0.0, mu, sd;
        for (c = 0u; c < a->n; c++)
            sum += (double)h[c];
        mu = sum / (double)a->n;
        for (c = 0u; c < a->n; c++) {
            const double d = (double)h[c] - mu;
            var += d * d;
        }
        sd = sqrt(var / (double)a->n + a->eps);
        for (c = 0u; c < a->n; c++)
            y[c] = llm_st((((double)h[c] - mu) / sd) * (double)a->g[c] + (double)a->b[c], a->iy[c]);
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
            for (d = 0u; d < HD; d++) {
                const size_t c = (size_t)g * HD + d;
                kc[d] = llm_st16(llm_ld(a->k0[o + d], a->sk[c]) + (double)a->bk[c], a->ick[c]);
                vc[(size_t)d * a->VK] = llm_st16(llm_ld(a->v[o + d], a->sv[c]) + (double)a->bv[c],
                                                 a->icv[c]);
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
            for (d = 0u; d < HD; d++) {
                const size_t c = (size_t)h * HD + d;
                q[u][d] = llm_st16(llm_ld(a->q0[o + d], a->sq[c]) + (double)a->bq[c], a->iq[h]);
            }
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
 * 2^-f_p (the P.V weight), every key: m = max raw, e_j = sexp_{f_s}[m - raw_j],
 * sum left to right, P = round_half_even(e / sum * 2^f_p) (llm_attn_softmax's
 * arithmetic: e * (2^f_p / sum) unless within 1e-7 of a rounding tie).  Work
 * items are blocks of LLM_SMX_CB columns read once, transposed. ---- */
typedef struct {
    const Data_t *s;
    Data_t       *p;
    const double *tab;
    double        ip;
    unsigned      T, C, nblk;
} vit_smx_t;

static void vit_smx_items(void *pp, unsigned i0, unsigned i1)
{
    const vit_smx_t *a = (const vit_smx_t *)pp;
    const unsigned   T = a->T, C = a->C;
    unsigned         it;
    int16_t         *tl = (int16_t *)malloc((size_t)LLM_SMX_CB * C * sizeof(int16_t));
    double          *eb = (double *)malloc((size_t)C * sizeof(double));
    if (!tl || !eb) {
        free(tl); free(eb);
        return;
    }
    for (it = i0; it < i1; it++) {
        const unsigned c0 = it * LLM_SMX_CB, nc = c0 + LLM_SMX_CB < T ? LLM_SMX_CB : T - c0;
        unsigned       k, j;
        for (j = 0u; j < C; j++) {
            const int16_t *row = (const int16_t *)a->s + (size_t)j * T + c0;
            for (k = 0u; k < nc; k++)
                tl[(size_t)k * C + j] = row[k];
        }
        for (k = 0u; k < nc; k++) {
            Data_t        *pr = a->p + (size_t)(c0 + k) * C;
            const int16_t *r = tl + (size_t)k * C;
            int            m = r[0];
            double         sum = 0.0, rinv;
            for (j = 1u; j < C; j++)
                m = r[j] > m ? r[j] : m;
            for (j = 0u; j < C; j++) {
                eb[j] = a->tab[m - r[j]];
                sum += eb[j];
            }
            rinv = a->ip / sum;
            for (j = 0u; j < C; j++) {
                const double q = eb[j] * rinv, f = q - floor(q);
                double       rq;
                if (f > 0.5 - 1e-7 && f < 0.5 + 1e-7) {
                    pr[j] = llm_st(eb[j] / sum, a->ip);        /* near a tie: exact */
                    continue;
                }
                rq = nearbyint(q);
                pr[j] = (Data_t)(int16_t)(rq > 32767.0 ? 32767.0 : rq);
            }
        }
    }
    free(tl);
    free(eb);
}

static void vit_attn_softmax(const Data_t *s, unsigned T, unsigned C, int fs, double ip, Data_t *p)
{
    vit_smx_t a;
    a.s = s; a.p = p; a.tab = _llm_sexp_tab[fs - LLM_SEXP_EMIN]; a.ip = ip; a.T = T; a.C = C;
    a.nblk = (T + LLM_SMX_CB - 1u) / LLM_SMX_CB;
    host_parallel(vit_smx_items, &a, a.nblk, 1u, 1u);
}

/* ---- VitGelu: a = gelu_f[sat16(raw + braw[c])] (tables per input exponent,
 * GELU tanh form via libm exp: y = x - x / (exp(2u) + 1)) ---- */
#define VIT_GELU_EMIN  (GELU_EMIN_VALUE)
#define VIT_GELU_NE    (GELU_NE_VALUE)
static double *_vit_gelu_tab[VIT_GELU_NE];

typedef struct {
    double *t;
    double  d;
} vit_gelu_fill_t;

static void vit_gelu_fill(void *p, unsigned i0, unsigned i1)
{
    const vit_gelu_fill_t *a = (const vit_gelu_fill_t *)p;
    unsigned i;
    for (i = i0; i < i1; i++) {
        const double x = (double)((int)i - 32768) / a->d;
        const double u = 0.7978845608028654 * (x + 0.044715 * ((x * x) * x));
        a->t[i] = x - x / (exp(2.0 * u) + 1.0);
    }
}

static int vit_gelu_table(int f)
{
    vit_gelu_fill_t a;
    double        **t = &_vit_gelu_tab[f - VIT_GELU_EMIN];
    if (*t) return 0;
    *t = (double *)malloc(65536u * sizeof(double));
    if (!*t) return -1;
    a.t = *t; a.d = ldexp(1.0, f);
    host_parallel(vit_gelu_fill, &a, 65536u, 1024u, 8u);
    return 0;
}

static void vit_gelu_free(int f)
{
    free(_vit_gelu_tab[f - VIT_GELU_EMIN]);
    _vit_gelu_tab[f - VIT_GELU_EMIN] = NULL;
}

typedef struct {
    const Data_t      *x;
    const signed char *fx;
    const int32_t     *bi;
    const double      *ia;
    unsigned           n;
    Data_t            *y;
} vit_gelu_t;

static void vit_gelu_rows(void *p, unsigned r0, unsigned r1)
{
    const vit_gelu_t *a = (const vit_gelu_t *)p;
    unsigned r, c;
    for (r = r0; r < r1; r++) {
        const Data_t *x = a->x + (size_t)r * a->n;
        Data_t       *y = a->y + (size_t)r * a->n;
        for (c = 0u; c < a->n; c++) {
            int32_t v = (int32_t)(int16_t)x[c] + a->bi[c];
            if (v > 32767) v = 32767;
            if (v < -32768) v = -32768;
            y[c] = llm_st(_vit_gelu_tab[a->fx[c] - VIT_GELU_EMIN][v + 32768], a->ia[c]);
        }
    }
}

static void vit_gelu(const Data_t *x, const signed char *fx, const int32_t *bi, const double *ia,
                     unsigned rows, unsigned n, Data_t *y)
{
    vit_gelu_t a;
    a.x = x; a.fx = fx; a.bi = bi; a.ia = ia; a.n = n; a.y = y;
    host_parallel(vit_gelu_rows, &a, rows, host_row_grain(n), 1u);
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
    return VIT_C.replace("GELU_EMIN_VALUE", str(GELU_EMIN)).replace("GELU_NE_VALUE", str(GELU_NE))


__all__ = ("VIT_OP_FACTORIES", "VitNode", "VitEmbedAddNode", "VitResAddNode", "VitLayerNormNode",
           "VitAttnPrepNode", "VitAttnSoftmaxNode", "VitGeluNode", "VitPixelShuffleNode",
           "VitSumDequantNode", "vit_c_helpers", "gelu_table", "gelu_exp", "pixel_shuffle_rows",
           "LLM_DOMAIN", "SEXP_EMIN", "SEXP_NE")
