"""
Host ops of the Piper (VITS) text-to-speech chunk (doc/plans/TTS_PLAN.md
§4; frontend src/piper.py).  ONNX nodes of the domain ``axi.llm``:

  TtsPrep       a conv input: channels [ch0, ch0 + nch) of a DMA int16 or
                float32 host tensor (optionally reversed), LeakyReLU (alpha),
                zero outside the utterance ([lo, hi) frames of the chunk: the
                i32 inputs lo / hi), written at the output's exponent in the
                folded layout [nch][rows][w0 + hl + hr]: row r, column j
                holds source time r*w0 + j - hl + t0 (zero outside the
                source window [wlo, whi))
  TtsGate       y = tanh(x[:n]) * sigmoid(x[n:])  (WaveNet gate)
  TtsSum        y = ((x0[c0..] + x1[c1..]) + ...) / div, optionally masked
  TtsFlowOut    the flow's coupling update on its float32 state:
                z' = flip(z); z'[n:] -= m; masked outside the utterance
  TtsInterleave y[o][m*s + r] = x[r*O + o][m]  (polyphase transposed conv)
  TtsPcm        pcm = round(tanh(x) * 32767)  (int16 host samples)

The text encoder's ops (the encode_<T> entries; rows >= n, the utterance's
ids, are padding: written as zeros, never read by a row < n):

  TtsEmbed      x[t] = table[ids[t]] * scale
  TtsRowPrep    a MatMul input from [T][C]: taps 1 (a copy at another
                exponent) or 3 (row t holds rows t-1, t, t+1: a kernel-3
                'same' conv as one MatMul), optionally + bias and ReLU
  TtsAttnSoftmax  the scores of one head (q.K^T from ConvKernel, keys-major)
                + the relative-key terms q[i] . ek[j - i + W] for |j - i| <= W
                (q recomputed as the q / k / v prep writes it), softmax over
                the keys < n -> P at 2^-f_p
  TtsAttnMerge  the heads' P.V (ConvKernel) + the relative-value terms
                sum_k P[i][i + k] * ev[k + W] -> [T][H * HD]
  TtsResNorm    LayerNorm(x + (y + b)) over the channels of each row
  TtsEncOut     int16 (+ bias) -> float32 host rows

Every DMA tensor carries one power-of-two exponent (scalar); host ops
compute in double, write int16 with round-half-even + saturate (llm_st16).
tanh / exp come from libm on both sides (Python ``math`` in the
references, C in the helpers): numpy's own transcendental kernels may
differ in the last bit.  A 2-D DMA view [C][L] of a tensor is its
[C][numel / C] (a conv output [1][C][rows][w0] is time-contiguous per
channel).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import ClassVar, Optional, Tuple

import numpy as np

from .host_nodes import HostContext, _c_double, _resolve
from .llm_nodes import LLM_DOMAIN, HostTable, LlmNode, _attr_ints, _const_array, _require, table_kind
from .vit_nodes import _bias, float_array_item

__all__ = ("TTS_OP_FACTORIES", "TtsNode", "TtsPrepNode", "TtsGateNode", "TtsSumNode",
           "TtsFlowOutNode", "TtsInterleaveNode", "TtsPcmNode", "TtsEmbedNode", "TtsRowPrepNode",
           "TtsAttnSoftmaxNode", "TtsAttnMergeNode", "TtsResNormNode", "TtsEncOutNode",
           "tts_c_helpers", "LLM_DOMAIN", "libm_map")


def libm_map(fn, x: np.ndarray) -> np.ndarray:
    """fn (a ``math`` function) element-wise through libm, evaluated once per
    distinct value (int16 grids: at most 65 536)."""
    x = np.asarray(x, np.float64)
    u, inv = np.unique(x, return_inverse=True)
    return np.array([fn(float(v)) for v in u], np.float64)[inv].reshape(x.shape)


def _sig(v: float) -> float:
    return 1.0 / (1.0 + math.exp(-v))


def _st16(v: np.ndarray, f: int) -> np.ndarray:
    """llm_st16 in values: round-half-even at 2^-f, saturate."""
    return np.clip(np.round(np.asarray(v, np.float64) * 2.0 ** f), -32768, 32767) * 2.0 ** -f


def _cl(t) -> Tuple[int, int]:
    """(channels, length) of a tensor's 2-D view: channels = shape[1] for a
    4-D [1][C][H][W] tensor, else shape[0]."""
    c = int(t.shape[1]) if len(t.shape) == 4 else int(t.shape[0])
    return c, t.numel // c


def _exp(t, node) -> int:
    _require(t.exp is not None and np.ndim(t.exp) == 0, node,
             f"'{t.onnx_name}' needs one power-of-two exponent")
    return int(np.asarray(t.exp))


def _attrs(node) -> dict:
    import onnx.helper as oh
    return {a.name: oh.get_attribute_value(a) for a in node.attribute}


def _double_attr(v) -> float:
    """A double carried as a string attribute (ONNX float attributes are
    float32: 0.1 would not be the double 0.1 the specification uses)."""
    return float(v.decode() if isinstance(v, bytes) else v)


def _p2(f: int) -> str:
    """2^f as an exact C double literal."""
    return f"ldexp(1.0, {int(f)})"


def _mask_cols(L: int, off: int, rate: int, lo: int, hi: int) -> np.ndarray:
    """Source positions s in [0, L) inside the utterance: lo*rate <= s + off*rate < hi*rate."""
    s = np.arange(L) + off * rate
    return (s >= lo * rate) & (s < hi * rate)


@dataclass
class TtsNode(LlmNode):
    helpers: ClassVar[Tuple[str, ...]] = ("llm", "tts")


# ------------------------------------------------------------------ #
# TtsPrep                                                              #
# ------------------------------------------------------------------ #

@dataclass
class TtsPrepNode(TtsNode):
    C_src: int = 1
    L_src: int = 1
    ch0: int = 0
    nch: int = 1
    reverse: int = 0
    t0: int = 0
    rows: int = 1
    w0: int = 1
    hl: int = 0
    hr: int = 0
    rate: int = 1
    frame_off: int = 0
    alpha: float = 1.0
    wlo: int = 0
    whi: int = 0
    src_f32: bool = False
    f_in: int = 0
    f_out: int = 0

    @classmethod
    def from_onnx_node(cls, node, tensors, index, align_elems, ctx: HostContext):
        src = _resolve(tensors, node.input[0], node)
        lo = _resolve(tensors, node.input[1], node)
        hi = _resolve(tensors, node.input[2], node)
        y = _resolve(tensors, node.output[0], node)
        a = _attrs(node)
        C, L = _cl(src)
        sn = cls(onnx_node=node, inputs=[src, lo, hi], output=y, index=index,
                 align_elems=align_elems, F=ctx.frac_bits, C_src=C, L_src=L,
                 ch0=int(a.get("ch0", 0)), nch=int(a["nch"]), reverse=int(a.get("reverse", 0)),
                 t0=int(a.get("t0", 0)), rows=int(a["rows"]), w0=int(a["w0"]),
                 hl=int(a.get("hl", 0)), hr=int(a.get("hr", 0)), rate=int(a.get("rate", 1)),
                 frame_off=int(a.get("frame_off", 0)), alpha=_double_attr(a.get("alpha", b"1.0")),
                 wlo=int(a.get("wlo", 0)), whi=int(a.get("whi", L)), src_f32=src.host == "f32")
        if not sn.src_f32:
            sn._want(src, None, "source")
            sn.f_in = _exp(src, node)
        sn._want(lo, "i32", "lo")
        sn._want(hi, "i32", "hi")
        sn._want(y, None, "output")
        sn.f_out = _exp(y, node)
        _require(sn.ch0 + sn.nch <= C, node, "channels beyond the source")
        _require(y.numel == sn.nch * sn.rows * (sn.w0 + sn.hl + sn.hr), node, "output size")
        return sn

    def describe(self):
        return (f"{self.nch} ch from [{self.C_src}][{self.L_src}] -> [{self.rows}][{self.w0}+{self.hl}"
                f"+{self.hr}] t0 {self.t0} alpha {self.alpha:g} rate {self.rate}")

    def c_call(self, ins, out, scratch, direct, dtype):
        s_in = "1.0" if self.src_f32 else _p2(-self.f_in)
        return [f"tts_prep({'NULL' if self.src_f32 else ins[0]}, "
                f"{ins[0] if self.src_f32 else 'NULL'}, {s_in}, {self.C_src}u, {self.L_src}u, "
                f"{self.ch0}u, {self.nch}u, {self.reverse}, {self.t0}, {self.rows}u, {self.w0}u, "
                f"{self.hl}u, {self.hr}u, {self.wlo}, {self.whi}, {ins[1]}[0], {ins[2]}[0], "
                f"{self.rate}u, {self.frame_off}, {_c_double(self.alpha)}, {_p2(self.f_out)}, {out});"]

    def reference(self, ins, dtype):
        x = np.asarray(ins[0], np.float64).reshape(self.C_src, self.L_src)
        lo, hi = int(np.asarray(ins[1]).reshape(-1)[0]), int(np.asarray(ins[2]).reshape(-1)[0])
        ch = np.arange(self.nch)
        ch = self.ch0 + (self.nch - 1 - ch if self.reverse else ch)
        x = x[ch]
        if self.alpha != 1.0:
            x = np.where(x > 0, x, self.alpha * x)
        x = np.where(_mask_cols(self.L_src, self.frame_off, self.rate, lo, hi)[None, :], x, 0.0)
        W = self.w0 + self.hl + self.hr
        s = (np.arange(self.rows)[:, None] * self.w0 + np.arange(W)[None, :] - self.hl + self.t0)
        ok = (s >= self.wlo) & (s < self.whi) & (s >= 0) & (s < self.L_src)
        y = np.where(ok[None], x[:, np.clip(s, 0, self.L_src - 1)], 0.0)
        return _st16(y, self.f_out).reshape(self.output.shape)


# ------------------------------------------------------------------ #
# TtsGate / TtsSum / TtsFlowOut / TtsInterleave / TtsPcm               #
# ------------------------------------------------------------------ #

@dataclass
class TtsGateNode(TtsNode):
    n: int = 1
    L: int = 1
    f_in: int = 0
    f_out: int = 0

    @classmethod
    def from_onnx_node(cls, node, tensors, index, align_elems, ctx: HostContext):
        x = _resolve(tensors, node.input[0], node)
        y = _resolve(tensors, node.output[0], node)
        C, L = _cl(x)
        sn = cls(onnx_node=node, inputs=[x], output=y, index=index, align_elems=align_elems,
                 F=ctx.frac_bits, n=C // 2, L=L)
        sn._want(x, None, "input")
        sn._want(y, None, "output")
        sn.f_in, sn.f_out = _exp(x, node), _exp(y, node)
        _require(C % 2 == 0 and y.numel == sn.n * L, node, "shapes")
        return sn

    def describe(self):
        return f"tanh * sigmoid on [{2 * self.n}][{self.L}]"

    def c_call(self, ins, out, scratch, direct, dtype):
        return [f"tts_gate({ins[0]}, {_p2(-self.f_in)}, {self.n}u, {self.L}u, {_p2(self.f_out)}, {out});"]

    def reference(self, ins, dtype):
        x = np.asarray(ins[0], np.float64).reshape(2 * self.n, self.L)
        y = libm_map(math.tanh, x[:self.n]) * libm_map(_sig, x[self.n:])
        return _st16(y, self.f_out).reshape(self.output.shape)


@dataclass
class TtsSumNode(TtsNode):
    nch: int = 1
    L: int = 1
    div: float = 1.0
    ch0s: Tuple[int, ...] = ()
    Cs: Tuple[int, ...] = ()
    f_ins: Tuple[int, ...] = ()
    f_out: int = 0
    masked: bool = False
    rate: int = 1
    frame_off: int = 0

    @classmethod
    def from_onnx_node(cls, node, tensors, index, align_elems, ctx: HostContext):
        a = _attrs(node)
        masked = bool(a.get("masked", 0))
        names = list(node.input)
        bounds = [_resolve(tensors, n, node) for n in names[-2:]] if masked else []
        if masked:
            names = names[:-2]
        xs = [_resolve(tensors, n, node) for n in names]
        y = _resolve(tensors, node.output[0], node)
        ch0s = tuple(int(v) for v in a.get("ch0s", [0] * len(xs)))
        nch = int(a["nch"])
        cls_ = [_cl(x) for x in xs]
        sn = cls(onnx_node=node, inputs=xs + bounds, output=y, index=index,
                 align_elems=align_elems, F=ctx.frac_bits, nch=nch, L=cls_[0][1],
                 div=float(a.get("div", 1.0)), ch0s=ch0s, Cs=tuple(c for c, _ in cls_),
                 masked=masked, rate=int(a.get("rate", 1)), frame_off=int(a.get("frame_off", 0)))
        for x in xs:
            sn._want(x, None, "input")
        for b in bounds:
            sn._want(b, "i32", "lo / hi")
        sn._want(y, None, "output")
        sn.f_ins = tuple(_exp(x, node) for x in xs)
        sn.f_out = _exp(y, node)
        _require(2 <= len(xs) <= 4, node, "2 to 4 inputs")
        _require(all(L == sn.L and c0 + nch <= c for (c, L), c0 in zip(cls_, ch0s, strict=True)),
                 node, "input shapes")
        _require(y.numel == nch * sn.L, node, "output size")
        return sn

    def describe(self):
        return (f"{len(self.ch0s)} inputs [{self.nch}][{self.L}]" + (f" / {self.div:g}" if self.div != 1 else "")
                + (" masked" if self.masked else ""))

    def c_call(self, ins, out, scratch, direct, dtype):
        k = len(self.ch0s)
        ptrs = ", ".join(f"{ins[i]} + (size_t){self.ch0s[i]}u * {self.L}u" for i in range(k))
        scs = ", ".join(_p2(-f) for f in self.f_ins)
        lo, hi = (f"{ins[k]}[0]", f"{ins[k + 1]}[0]") if self.masked else ("0", "0")
        return ["{",
                f"    const Data_t *xs_[{k}] = {{ {ptrs} }};",
                f"    const double  sx_[{k}] = {{ {scs} }};",
                f"    tts_sum(xs_, sx_, {k}u, {self.nch}u, {self.L}u, {_c_double(self.div)}, "
                f"{int(self.masked)}, {lo}, {hi}, {self.rate}u, {self.frame_off}, {_p2(self.f_out)}, "
                f"{out});",
                "}"]

    def reference(self, ins, dtype):
        k = len(self.ch0s)
        v = None
        for i in range(k):
            x = np.asarray(ins[i], np.float64).reshape(self.Cs[i], self.L)[self.ch0s[i]:self.ch0s[i] + self.nch]
            v = x if v is None else v + x
        if self.div != 1.0:
            v = v / self.div
        if self.masked:
            lo, hi = int(np.asarray(ins[k]).reshape(-1)[0]), int(np.asarray(ins[k + 1]).reshape(-1)[0])
            v = np.where(_mask_cols(self.L, self.frame_off, self.rate, lo, hi)[None, :], v, 0.0)
        return _st16(v, self.f_out).reshape(self.output.shape)


@dataclass
class TtsFlowOutNode(TtsNode):
    n: int = 1
    L: int = 1
    f_m: int = 0

    @classmethod
    def from_onnx_node(cls, node, tensors, index, align_elems, ctx: HostContext):
        z = _resolve(tensors, node.input[0], node)
        m = _resolve(tensors, node.input[1], node)
        lo = _resolve(tensors, node.input[2], node)
        hi = _resolve(tensors, node.input[3], node)
        y = _resolve(tensors, node.output[0], node)
        C, L = _cl(z)
        sn = cls(onnx_node=node, inputs=[z, m, lo, hi], output=y, index=index,
                 align_elems=align_elems, F=ctx.frac_bits, n=C // 2, L=L)
        sn._want(z, "f32", "state")
        sn._want(m, None, "m")
        sn._want(lo, "i32", "lo")
        sn._want(hi, "i32", "hi")
        sn._want(y, "f32", "output")
        sn.f_m = _exp(m, node)
        _require(m.numel == sn.n * L and y.numel == z.numel, node, "shapes")
        return sn

    def describe(self):
        return f"flip, x1 -= m on [{2 * self.n}][{self.L}] (float32 state)"

    def c_call(self, ins, out, scratch, direct, dtype):
        return [f"tts_flow_out({ins[0]}, {ins[1]}, {_p2(-self.f_m)}, {self.n}u, {self.L}u, "
                f"{ins[2]}[0], {ins[3]}[0], {out});"]

    def reference(self, ins, dtype):
        z = np.asarray(ins[0], np.float32).reshape(2 * self.n, self.L)[::-1]
        m = np.asarray(ins[1], np.float64).reshape(self.n, self.L)
        lo, hi = int(np.asarray(ins[2]).reshape(-1)[0]), int(np.asarray(ins[3]).reshape(-1)[0])
        ok = _mask_cols(self.L, 0, 1, lo, hi)[None, :]
        x0 = np.where(ok, z[:self.n], np.float32(0))
        x1 = np.where(ok, (z[self.n:].astype(np.float64) - m).astype(np.float32), np.float32(0))
        return np.concatenate([x0, x1]).astype(np.float32).astype(np.float64).reshape(self.output.shape)


@dataclass
class TtsInterleaveNode(TtsNode):
    s: int = 1
    O: int = 1          # noqa: E741  (output channels)
    L: int = 1

    @classmethod
    def from_onnx_node(cls, node, tensors, index, align_elems, ctx: HostContext):
        x = _resolve(tensors, node.input[0], node)
        y = _resolve(tensors, node.output[0], node)
        s = int(_attrs(node)["s"])
        C, L = _cl(x)
        sn = cls(onnx_node=node, inputs=[x], output=y, index=index, align_elems=align_elems,
                 F=ctx.frac_bits, s=s, O=C // s, L=L)
        sn._want(x, None, "input")
        sn._want(y, None, "output")
        _require(C % s == 0 and y.numel == x.numel and _exp(x, node) == _exp(y, node), node,
                 "shapes / exponents")
        return sn

    def describe(self):
        return f"interleave {self.s} phases of [{self.O}][{self.L}]"

    def c_call(self, ins, out, scratch, direct, dtype):
        return [f"tts_interleave({ins[0]}, {self.s}u, {self.O}u, {self.L}u, {out});"]

    def reference(self, ins, dtype):
        x = np.asarray(ins[0], np.float64).reshape(self.s, self.O, self.L)
        return x.transpose(1, 2, 0).reshape(self.output.shape)


@dataclass
class TtsPcmNode(TtsNode):
    L: int = 1
    f_in: int = 0

    @classmethod
    def from_onnx_node(cls, node, tensors, index, align_elems, ctx: HostContext):
        x = _resolve(tensors, node.input[0], node)
        y = _resolve(tensors, node.output[0], node)
        sn = cls(onnx_node=node, inputs=[x], output=y, index=index, align_elems=align_elems,
                 F=ctx.frac_bits, L=x.numel)
        sn._want(x, None, "input")
        sn._want(y, "i16", "output")
        sn.f_in = _exp(x, node)
        _require(y.numel == x.numel, node, "shapes")
        return sn

    def describe(self):
        return f"tanh -> int16 PCM, {self.L} samples"

    def c_call(self, ins, out, scratch, direct, dtype):
        return [f"tts_pcm({ins[0]}, {_p2(-self.f_in)}, {self.L}u, {out});"]

    def reference(self, ins, dtype):
        x = np.asarray(ins[0], np.float64).reshape(-1)
        return np.clip(np.round(libm_map(math.tanh, x) * 32767.0), -32768, 32767).reshape(self.output.shape)


# ------------------------------------------------------------------ #
# The text encoder's ops (encode_<T> entries, src/piper.py)            #
# ------------------------------------------------------------------ #

@dataclass
class TtsEmbedNode(TtsNode):
    """inputs [ids, table (constant [rows][D]), n] -> x [T][D] int16 at f."""
    T: int = 1
    D: int = 1
    rows: int = 1
    scale: float = 1.0
    f_out: int = 0
    table: Optional[HostTable] = field(default=None, repr=False)

    @classmethod
    def from_onnx_node(cls, node, tensors, index, align_elems, ctx: HostContext):
        ids = _resolve(tensors, node.input[0], node)
        n = _resolve(tensors, node.input[2], node)
        y = _resolve(tensors, node.output[0], node)
        tab = _const_array(ctx, tensors, node.input[1], node, "table")
        _require(tab.ndim == 2, node, "table must be [rows][D]")
        rows, D = tab.shape
        name = "_tts_tab_" + tensors[node.input[1]].c_name
        sn = cls(onnx_node=node, inputs=[ids, n], output=y, index=index, align_elems=align_elems,
                 F=ctx.frac_bits, T=ids.numel, D=int(D), rows=int(rows),
                 scale=_double_attr(_attrs(node).get("scale", b"1.0")),
                 table=HostTable(name, table_kind(tab), np.asarray(tab, np.float32)))
        sn._want(ids, "i32", "ids")
        sn._want(n, "i32", "n")
        sn._want(y, None, "output")
        sn.f_out = _exp(y, node)
        _require(y.numel == sn.T * sn.D, node, "output [T][D]")
        return sn

    def tables(self):
        return [self.table]

    def describe(self):
        return f"{self.T} rows of {self.D} x {self.scale:.6g} from a [{self.rows}] table"

    def c_call(self, ins, out, scratch, direct, dtype):
        bf = self.table.kind == "bf16"
        return [f"tts_embed({ins[0]}, {self.T}u, {ins[1]}[0], {'NULL' if bf else self.table.name}, "
                f"{self.table.name if bf else 'NULL'}, {self.rows}u, {self.D}u, "
                f"{_c_double(self.scale)}, {_p2(self.f_out)}, {out});"]

    def reference(self, ins, dtype):
        ids = np.asarray(ins[0]).reshape(-1).astype(np.int64)
        n = int(np.asarray(ins[1]).reshape(-1)[0])
        tab = np.asarray(self.table.data, np.float32).reshape(self.rows, self.D).astype(np.float64)
        v = tab[np.clip(ids, 0, self.rows - 1)] * self.scale
        v[max(n, 0):] = 0.0
        return _st16(v, self.f_out).reshape(self.output.shape)


@dataclass
class TtsRowPrepNode(TtsNode):
    """inputs [x [T][C], n (, bias [C])] -> [T][taps * C]: row t, tap k reads
    row t + k - (taps - 1) / 2 (zero outside [0, n)), + bias, ReLU."""
    T: int = 1
    C: int = 1
    taps: int = 1
    relu: int = 0
    f_in: int = 0
    f_out: int = 0
    bias: Optional[np.ndarray] = field(default=None, repr=False)

    @classmethod
    def from_onnx_node(cls, node, tensors, index, align_elems, ctx: HostContext):
        x = _resolve(tensors, node.input[0], node)
        n = _resolve(tensors, node.input[1], node)
        y = _resolve(tensors, node.output[0], node)
        a = _attrs(node)
        C = int(x.shape[-1])
        sn = cls(onnx_node=node, inputs=[x, n], output=y, index=index, align_elems=align_elems,
                 F=ctx.frac_bits, T=x.numel // C, C=C, taps=int(a.get("taps", 1)),
                 relu=int(a.get("relu", 0)),
                 bias=_bias(ctx, tensors, node, 2, C, "bias") if len(node.input) > 2 and node.input[2] else None)
        sn._want(x, None, "input")
        sn._want(n, "i32", "n")
        sn._want(y, None, "output")
        sn.f_in, sn.f_out = _exp(x, node), _exp(y, node)
        _require(sn.taps in (1, 3), node, "taps 1 or 3")
        _require(y.numel == sn.T * sn.taps * C, node, "output [T][taps * C]")
        return sn

    def c_runtime(self):
        return [] if self.bias is None else [float_array_item("_tts_b", self.bias)[1]]

    def describe(self):
        return (f"[{self.T}][{self.C}] -> taps {self.taps}" + (" + bias" if self.bias is not None else "")
                + (" ReLU" if self.relu else ""))

    def c_call(self, ins, out, scratch, direct, dtype):
        b = "NULL" if self.bias is None else float_array_item("_tts_b", self.bias)[0]
        return [f"tts_rowprep({ins[0]}, {self.T}u, {self.C}u, {ins[1]}[0], {b}, {self.relu}, "
                f"{self.taps}u, {_p2(-self.f_in)}, {_p2(self.f_out)}, {out});"]

    def reference(self, ins, dtype):
        x = np.asarray(ins[0], np.float64).reshape(self.T, self.C)
        n = int(np.asarray(ins[1]).reshape(-1)[0])
        if self.bias is not None:
            x = x + self.bias.astype(np.float64)[None, :]
        if self.relu:
            x = np.maximum(x, 0.0)
        x[max(n, 0):] = 0.0
        h = (self.taps - 1) // 2
        p = np.pad(x, ((h, h), (0, 0)))
        y = np.concatenate([p[k:k + self.T] for k in range(self.taps)], axis=1)
        y[max(n, 0):] = 0.0
        return _st16(y, self.f_out).reshape(self.output.shape)


@dataclass
class TtsAttnSoftmaxNode(TtsNode):
    """inputs [s [C][T] (raw, keys-major), q0 [T][H * HD] (the q projection),
    bq (constant), ek (constant [2W + 1][HD]), n] -> P [T][C] raw at 2^-f_p."""
    T: int = 1
    C: int = 1
    HD: int = 16
    D: int = 16
    g: int = 0
    W: int = 4
    fs: int = 0
    fq: int = 0
    fq0: int = 0
    fp: int = 15
    bq: Optional[np.ndarray] = field(default=None, repr=False)
    ek: Optional[np.ndarray] = field(default=None, repr=False)

    @classmethod
    def from_onnx_node(cls, node, tensors, index, align_elems, ctx: HostContext):
        s = _resolve(tensors, node.input[0], node)
        q0 = _resolve(tensors, node.input[1], node)
        n = _resolve(tensors, node.input[4], node)
        y = _resolve(tensors, node.output[0], node)
        a = _attrs(node)
        HD, W = int(a["head_dim"]), int(a.get("window", 4))
        C, T = int(s.shape[0]), int(s.shape[1])
        D = int(q0.shape[-1])
        sn = cls(onnx_node=node, inputs=[s, q0, n], output=y, index=index, align_elems=align_elems,
                 F=ctx.frac_bits, T=T, C=C, HD=HD, D=D, g=int(a["head"]), W=W,
                 fs=_attr_ints(a, "s_exp", node, 1)[0], fq=_attr_ints(a, "q_exp", node, 1)[0],
                 fp=_attr_ints(a, "p_exp", node, 1)[0], bq=_bias(ctx, tensors, node, 2, D, "bq"),
                 ek=_bias(ctx, tensors, node, 3, (2 * W + 1) * HD, "ek"))
        sn._want(s, None, "scores")
        sn._want(q0, None, "q0")
        sn._want(n, "i32", "n")
        sn._want(y, None, "output")
        sn.fq0 = _exp(q0, node)
        _require(q0.numel == T * D and (sn.g + 1) * HD <= D, node, "q0 [T][H * HD]")
        _require(list(y.shape) == [T, C] and C == T, node, "P [T][C], C == T")
        return sn

    def c_runtime(self):
        return [float_array_item("_tts_b", self.bq)[1], float_array_item("_tts_e", self.ek)[1]]

    def describe(self):
        return (f"head {self.g}: [{self.C}] keys x [{self.T}] queries, relative window {self.W}, "
                f"P at 2^-{self.fp}")

    def c_call(self, ins, out, scratch, direct, dtype):
        bq = float_array_item("_tts_b", self.bq)[0]
        ek = float_array_item("_tts_e", self.ek)[0]
        return [f"tts_attn_softmax({ins[0]}, {ins[1]}, {bq}, {ek}, {self.T}u, {self.C}u, {self.D}u, "
                f"{self.HD}u, {self.g}u, {self.W}u, {ins[2]}[0], {_p2(-self.fs)}, {_p2(-self.fq0)}, "
                f"{_p2(self.fq)}, {_p2(self.fp)}, {out});"]

    def reference(self, ins, dtype):
        s = np.asarray(ins[0], np.float64).reshape(self.C, self.T).T            # [queries][keys] raw
        q0 = np.asarray(ins[1], np.float64).reshape(self.T, self.D)
        n = max(int(np.asarray(ins[2]).reshape(-1)[0]), 0)
        sl = slice(self.g * self.HD, (self.g + 1) * self.HD)
        q = np.clip(np.round((q0[:n, sl] + self.bq[sl].astype(np.float64)) * 2.0 ** self.fq), -32768, 32767)
        qv = q * 2.0 ** -self.fq
        ek = self.ek.astype(np.float64).reshape(2 * self.W + 1, self.HD)
        rel = np.cumsum(qv[:, None, :] * ek[None, :, :], axis=-1)[..., -1]      # [n][2W + 1]
        val = s[:n, :n] * 2.0 ** -self.fs
        for d in range(-self.W, self.W + 1):
            idx = np.arange(max(0, -d), min(n, n - d))
            val[idx, idx + d] = val[idx, idx + d] + rel[idx, d + self.W]
        p = np.zeros((self.T, self.C))
        if n:
            e = libm_map(math.exp, val - val.max(1, keepdims=True))
            p[:n, :n] = np.clip(np.round(e / np.cumsum(e, axis=1)[:, -1:] * 2.0 ** self.fp), 0, 32767)
        return p


@dataclass
class TtsAttnMergeNode(TtsNode):
    """inputs [o_0 .. o_{H-1} [T][HD] (raw P.V), P_0 .. P_{H-1} [T][C] (raw),
    ev (constant [2W + 1][HD]), n] -> [T][H * HD] int16 at f."""
    T: int = 1
    C: int = 1
    H: int = 1
    HD: int = 16
    W: int = 4
    fo: Tuple[int, ...] = ()
    fp: int = 15
    f_out: int = 0
    ev: Optional[np.ndarray] = field(default=None, repr=False)

    @classmethod
    def from_onnx_node(cls, node, tensors, index, align_elems, ctx: HostContext):
        a = _attrs(node)
        H, HD, W = int(a["num_heads"]), int(a["head_dim"]), int(a.get("window", 4))
        os_ = [_resolve(tensors, node.input[h], node) for h in range(H)]
        ps = [_resolve(tensors, node.input[H + h], node) for h in range(H)]
        n = _resolve(tensors, node.input[2 * H + 1], node)
        y = _resolve(tensors, node.output[0], node)
        T, C = int(ps[0].shape[0]), int(ps[0].shape[1])
        sn = cls(onnx_node=node, inputs=os_ + ps + [n], output=y, index=index, align_elems=align_elems,
                 F=ctx.frac_bits, T=T, C=C, H=H, HD=HD, W=W,
                 fo=tuple(_attr_ints(a, "o_exp", node, H)), fp=_attr_ints(a, "p_exp", node, 1)[0],
                 ev=_bias(ctx, tensors, node, 2 * H, (2 * W + 1) * HD, "ev"))
        for t in os_ + ps:
            sn._want(t, None, "P.V / P")
        sn._want(n, "i32", "n")
        sn._want(y, None, "output")
        sn.f_out = _exp(y, node)
        _require(all(o.numel == T * HD for o in os_), node, "o [T][HD]")
        _require(y.numel == T * H * HD, node, "output [T][H * HD]")
        return sn

    def c_runtime(self):
        return [float_array_item("_tts_e", self.ev)[1]]

    def describe(self):
        return f"{self.H} heads of [{self.T}][{self.HD}] + relative values (window {self.W})"

    def c_call(self, ins, out, scratch, direct, dtype):
        H = self.H
        ev = float_array_item("_tts_e", self.ev)[0]
        return ["{",
                f"    const Data_t *o_[{H}] = {{ {', '.join(ins[:H])} }};",
                f"    const Data_t *p_[{H}] = {{ {', '.join(ins[H:2 * H])} }};",
                f"    const double  so_[{H}] = {{ {', '.join(_p2(-f) for f in self.fo)} }};",
                f"    tts_attn_merge(o_, p_, so_, {ev}, {self.T}u, {self.C}u, {H}u, {self.HD}u, {self.W}u, "
                f"{ins[2 * H]}[0], {_p2(-self.fp)}, {_p2(self.f_out)}, {out});",
                "}"]

    def reference(self, ins, dtype):
        H, HD, T = self.H, self.HD, self.T
        n = max(int(np.asarray(ins[2 * H]).reshape(-1)[0]), 0)
        ev = self.ev.astype(np.float64).reshape(2 * self.W + 1, HD)
        out = np.zeros((T, H * HD))
        for h in range(H):
            o = np.asarray(ins[h], np.float64).reshape(T, HD)[:n] * 2.0 ** -self.fo[h]
            p = np.asarray(ins[H + h], np.float64).reshape(T, self.C)
            for d in range(-self.W, self.W + 1):
                idx = np.arange(max(0, -d), min(n, n - d))
                o[idx] = o[idx] + (p[idx, idx + d] * 2.0 ** -self.fp)[:, None] * ev[d + self.W][None, :]
            out[:n, h * HD:(h + 1) * HD] = o
        return _st16(out, self.f_out).reshape(self.output.shape)


@dataclass
class TtsResNormNode(TtsNode):
    """inputs [x, y, by, gamma, beta (constants), n] -> LayerNorm(x + (y + by))
    over the D channels of every row < n (sums left to right)."""
    T: int = 1
    D: int = 1
    eps: float = 1e-5
    fx: int = 0
    fy: int = 0
    f_out: int = 0
    by: Optional[np.ndarray] = field(default=None, repr=False)
    gamma: Optional[np.ndarray] = field(default=None, repr=False)
    beta: Optional[np.ndarray] = field(default=None, repr=False)

    @classmethod
    def from_onnx_node(cls, node, tensors, index, align_elems, ctx: HostContext):
        x = _resolve(tensors, node.input[0], node)
        yy = _resolve(tensors, node.input[1], node)
        n = _resolve(tensors, node.input[5], node)
        out = _resolve(tensors, node.output[0], node)
        D = int(x.shape[-1])
        sn = cls(onnx_node=node, inputs=[x, yy, n], output=out, index=index, align_elems=align_elems,
                 F=ctx.frac_bits, T=x.numel // D, D=D, eps=_double_attr(_attrs(node).get("eps", b"1e-5")),
                 by=_bias(ctx, tensors, node, 2, D, "bias"), gamma=_bias(ctx, tensors, node, 3, D, "gamma"),
                 beta=_bias(ctx, tensors, node, 4, D, "beta"))
        for t, w in ((x, "x"), (yy, "y"), (out, "output")):
            sn._want(t, None, w)
        sn._want(n, "i32", "n")
        sn.fx, sn.fy, sn.f_out = _exp(x, node), _exp(yy, node), _exp(out, node)
        _require(yy.numel == x.numel == out.numel, node, "shapes")
        return sn

    def c_runtime(self):
        return [float_array_item("_tts_b", v)[1] for v in (self.by, self.gamma, self.beta)]

    def describe(self):
        return f"[{self.T}][{self.D}] eps {self.eps:g}"

    def c_call(self, ins, out, scratch, direct, dtype):
        by, g, b = (float_array_item("_tts_b", v)[0] for v in (self.by, self.gamma, self.beta))
        return [f"tts_resnorm({ins[0]}, {ins[1]}, {by}, {g}, {b}, {self.T}u, {self.D}u, {ins[2]}[0], "
                f"{_p2(-self.fx)}, {_p2(-self.fy)}, {_c_double(self.eps)}, {_p2(self.f_out)}, {out});"]

    def reference(self, ins, dtype):
        n = max(int(np.asarray(ins[2]).reshape(-1)[0]), 0)
        x = np.asarray(ins[0], np.float64).reshape(self.T, self.D)[:n]
        y = np.asarray(ins[1], np.float64).reshape(self.T, self.D)[:n]
        s = x + (y + self.by.astype(np.float64)[None, :])
        mean = np.cumsum(s, axis=1)[:, -1] / self.D
        d = s - mean[:, None]
        var = np.cumsum(d * d, axis=1)[:, -1] / self.D
        v = np.zeros((self.T, self.D))
        v[:n] = d / np.sqrt(var + self.eps)[:, None] * self.gamma.astype(np.float64)[None, :] \
            + self.beta.astype(np.float64)[None, :]
        return _st16(v, self.f_out).reshape(self.output.shape)


@dataclass
class TtsEncOutNode(TtsNode):
    """inputs [x [T][C], n (, bias)] -> float32 host [T][C] = x (+ bias), rows >= n zero."""
    T: int = 1
    C: int = 1
    f_in: int = 0
    bias: Optional[np.ndarray] = field(default=None, repr=False)

    @classmethod
    def from_onnx_node(cls, node, tensors, index, align_elems, ctx: HostContext):
        x = _resolve(tensors, node.input[0], node)
        n = _resolve(tensors, node.input[1], node)
        y = _resolve(tensors, node.output[0], node)
        C = int(x.shape[-1])
        sn = cls(onnx_node=node, inputs=[x, n], output=y, index=index, align_elems=align_elems,
                 F=ctx.frac_bits, T=x.numel // C, C=C,
                 bias=_bias(ctx, tensors, node, 2, C, "bias") if len(node.input) > 2 and node.input[2] else None)
        sn._want(x, None, "input")
        sn._want(n, "i32", "n")
        sn._want(y, "f32", "output")
        sn.f_in = _exp(x, node)
        _require(y.numel == x.numel, node, "shapes")
        return sn

    def c_runtime(self):
        return [] if self.bias is None else [float_array_item("_tts_b", self.bias)[1]]

    def describe(self):
        return f"[{self.T}][{self.C}] -> float32" + (" + bias" if self.bias is not None else "")

    def c_call(self, ins, out, scratch, direct, dtype):
        b = "NULL" if self.bias is None else float_array_item("_tts_b", self.bias)[0]
        return [f"tts_enc_out({ins[0]}, {b}, {self.T}u, {self.C}u, {ins[1]}[0], {_p2(-self.f_in)}, {out});"]

    def reference(self, ins, dtype):
        n = max(int(np.asarray(ins[1]).reshape(-1)[0]), 0)
        x = np.asarray(ins[0], np.float64).reshape(self.T, self.C)
        if self.bias is not None:
            x = x + self.bias.astype(np.float64)[None, :]
        y = x.astype(np.float32).astype(np.float64)
        y[n:] = 0.0
        return y.reshape(self.output.shape)


TTS_OP_FACTORIES = {
    "TtsPrep":       TtsPrepNode.from_onnx_node,
    "TtsGate":       TtsGateNode.from_onnx_node,
    "TtsSum":        TtsSumNode.from_onnx_node,
    "TtsFlowOut":    TtsFlowOutNode.from_onnx_node,
    "TtsInterleave": TtsInterleaveNode.from_onnx_node,
    "TtsPcm":        TtsPcmNode.from_onnx_node,
    "TtsEmbed":      TtsEmbedNode.from_onnx_node,
    "TtsRowPrep":    TtsRowPrepNode.from_onnx_node,
    "TtsAttnSoftmax": TtsAttnSoftmaxNode.from_onnx_node,
    "TtsAttnMerge":  TtsAttnMergeNode.from_onnx_node,
    "TtsResNorm":    TtsResNormNode.from_onnx_node,
    "TtsEncOut":     TtsEncOutNode.from_onnx_node,
}


TTS_C = r"""
/* =============== TTS host ops (axi.llm domain, src/tts_nodes.py) =============== */

/* The int16 write-back of these loops: llm_st() branch-free (round half to
 * even, saturate, NaN -> 0), bit-identical. */
static inline Data_t tts_st(double v, double si)
{
    double r = nearbyint(v * si);
    r = r > 32767.0 ? 32767.0 : r;
    r = r < -32768.0 ? -32768.0 : r;
    r = r == r ? r : 0.0;
    return (Data_t)(int16_t)r;
}

/* The same for a finite v (from int16 data: never NaN) — fminnm / fmaxnm. */
static inline Data_t tts_stf(double v, double si)
{
    return (Data_t)(int16_t)fmax(fmin(nearbyint(v * si), 32767.0), -32768.0);
}

static inline long tts_clamp(long v, long lo, long hi) { return v < lo ? lo : v > hi ? hi : v; }

/* s == 2^e exactly? */
static inline int tts_pow2(double s, int *e)
{
    int k;
    if (!(s > 0.0) || frexp(s, &k) != 0.5)
        return 0;
    *e = k - 1;
    return 1;
}

/* acc / 2^m rounded half to even (m >= 1), saturated to int16: what
 * nearbyint + saturation give on the exact double acc * 2^-m. */
static inline Data_t tts_rshift(int64_t acc, int m, int64_t half, int64_t mask)
{
    int64_t q = acc >> m;
    const int64_t r = acc & mask;
    q += (r > half) | ((r == half) & (q & 1));
    q = q > 32767 ? 32767 : q;
    q = q < -32768 ? -32768 : q;
    return (Data_t)(int16_t)q;
}

static inline Data_t tts_sat(int64_t q)
{
    q = q > 32767 ? 32767 : q;
    q = q < -32768 ? -32768 : q;
    return (Data_t)(int16_t)q;
}

typedef struct {
    const Data_t  *x16;
    const float   *x32;
    double         sx, alpha, si;
    unsigned       L, ch0, nch, rows, w0, hl, hr, rate;
    int            reverse, t0, wlo, whi, lo, hi, off;
    Data_t        *y;
} tts_prep_t;

/* Row r of the folded layout, column j reads source s = r*w0 + j - hl + t0;
 * the columns whose s lies in the window, the tensor and the mask are one
 * run [jlo, jhi) — zeros around it, no test per element. */
static void tts_prep_rows(void *p, unsigned c0, unsigned c1)
{
    const tts_prep_t *a = (const tts_prep_t *)p;
    const unsigned    W = a->w0 + a->hl + a->hr, rows = a->rows;
    const long        L = (long)a->L, sh = (long)a->off * a->rate;
    const long        slo = tts_clamp((long)a->lo * a->rate - sh, (long)a->wlo > 0 ? a->wlo : 0, L);
    const long        shi = tts_clamp((long)a->hi * a->rate - sh, slo, (long)a->whi < L ? a->whi : L);
    const double      sx = a->sx, si = a->si, alpha = a->alpha;
    unsigned          c, r;
    for (c = c0; c < c1; c++) {
        const unsigned sc = a->ch0 + (a->reverse ? a->nch - 1u - c : c);
        for (r = 0u; r < rows; r++) {
            Data_t *restrict yr   = a->y + ((size_t)c * rows + r) * W;
            const long       base = (long)r * a->w0 - (long)a->hl + a->t0;      /* s of column 0 */
            const long       jlo = tts_clamp(slo - base, 0, W), jhi = tts_clamp(shi - base, jlo, W);
            long             j;
            memset(yr, 0, (size_t)jlo * sizeof(Data_t));
            memset(yr + jhi, 0, (size_t)(W - jhi) * sizeof(Data_t));
            if (a->x32) {
                const float *restrict x = a->x32 + (size_t)sc * a->L + base;
                for (j = jlo; j < jhi; j++) {
                    const double v = (double)x[j];
                    yr[j] = tts_st(v > 0.0 ? v : alpha * v, si);
                }
            } else if (alpha == 1.0) {
                const Data_t *restrict x = a->x16 + (size_t)sc * a->L + base;
                for (j = jlo; j < jhi; j++)
                    yr[j] = tts_stf((double)(int16_t)x[j] * sx, si);
            } else {                                   /* four independent chains */
                const Data_t *restrict x = a->x16 + (size_t)sc * a->L + base;
                for (j = jlo; j + 4 <= jhi; j += 4) {
                    const double v0 = (double)(int16_t)x[j] * sx, v1 = (double)(int16_t)x[j + 1] * sx;
                    const double v2 = (double)(int16_t)x[j + 2] * sx, v3 = (double)(int16_t)x[j + 3] * sx;
                    yr[j]     = tts_stf(v0 > 0.0 ? v0 : alpha * v0, si);
                    yr[j + 1] = tts_stf(v1 > 0.0 ? v1 : alpha * v1, si);
                    yr[j + 2] = tts_stf(v2 > 0.0 ? v2 : alpha * v2, si);
                    yr[j + 3] = tts_stf(v3 > 0.0 ? v3 : alpha * v3, si);
                }
                for (; j < jhi; j++) {
                    const double v = (double)(int16_t)x[j] * sx;
                    yr[j] = tts_stf(v > 0.0 ? v : alpha * v, si);
                }
            }
        }
    }
}

static void tts_prep(const Data_t *x16, const float *x32, double sx, unsigned C, unsigned L,
                     unsigned ch0, unsigned nch, int reverse, int t0, unsigned rows, unsigned w0,
                     unsigned hl, unsigned hr, int wlo, int whi, int lo, int hi, unsigned rate,
                     int off, double alpha, double si, Data_t *y)
{
    tts_prep_t a;
    (void)C;
    a.x16 = x16; a.x32 = x32; a.sx = sx; a.alpha = alpha; a.si = si;
    a.L = L; a.ch0 = ch0; a.nch = nch; a.rows = rows; a.w0 = w0; a.hl = hl; a.hr = hr;
    a.rate = rate; a.reverse = reverse; a.t0 = t0; a.wlo = wlo; a.whi = whi; a.lo = lo;
    a.hi = hi; a.off = off;
    a.y = y;
    host_parallel(tts_prep_rows, &a, nch, 1u, 1u);
}

typedef struct {
    const Data_t *x;
    double        sx, si;
    unsigned      n, L;
    Data_t       *y;
    const double *th, *sg;                     /* tanh / sigmoid of every int16 at sx */
} tts_gate_t;

/* tanh(x * sx) and 1 / (1 + exp(-x * sx)) for all 65536 int16 x — the same
 * libm calls as per element, once per exponent (a few per model; kept until
 * the process exits). */
#define TTS_GATE_TABLES 8
static struct { double sx; double *th, *sg; } s_tts_gate_tab[TTS_GATE_TABLES];

static int tts_gate_tables(double sx, const double **th, const double **sg)
{
    unsigned i;
    int      x;
    for (i = 0u; i < TTS_GATE_TABLES && s_tts_gate_tab[i].th; i++)
        if (s_tts_gate_tab[i].sx == sx) {
            *th = s_tts_gate_tab[i].th + 32768;
            *sg = s_tts_gate_tab[i].sg + 32768;
            return 1;
        }
    if (i == TTS_GATE_TABLES)
        return 0;
    s_tts_gate_tab[i].th = (double *)malloc(65536u * sizeof(double));
    s_tts_gate_tab[i].sg = (double *)malloc(65536u * sizeof(double));
    if (!s_tts_gate_tab[i].th || !s_tts_gate_tab[i].sg) {
        free(s_tts_gate_tab[i].th);
        free(s_tts_gate_tab[i].sg);
        s_tts_gate_tab[i].th = s_tts_gate_tab[i].sg = NULL;
        return 0;
    }
    for (x = -32768; x < 32768; x++) {
        s_tts_gate_tab[i].th[x + 32768] = tanh((double)x * sx);
        s_tts_gate_tab[i].sg[x + 32768] = 1.0 / (1.0 + exp(-((double)x * sx)));
    }
    s_tts_gate_tab[i].sx = sx;
    *th = s_tts_gate_tab[i].th + 32768;
    *sg = s_tts_gate_tab[i].sg + 32768;
    return 1;
}

static void tts_gate_rows(void *p, unsigned c0, unsigned c1)
{
    const tts_gate_t *a = (const tts_gate_t *)p;
    const unsigned    L = a->L;
    const double      sx = a->sx, si = a->si;
    unsigned          c, t;
    for (c = c0; c < c1; c++) {
        const Data_t *restrict xa = a->x + (size_t)c * L;
        const Data_t *restrict xb = a->x + (size_t)(c + a->n) * L;
        Data_t *restrict       y  = a->y + (size_t)c * L;
        if (a->th)
            for (t = 0u; t < L; t++)
                y[t] = tts_st(a->th[(int16_t)xa[t]] * a->sg[(int16_t)xb[t]], si);
        else
            for (t = 0u; t < L; t++) {
                const double g = tanh((double)(int16_t)xa[t] * sx);
                const double s = 1.0 / (1.0 + exp(-((double)(int16_t)xb[t] * sx)));
                y[t] = tts_st(g * s, si);
            }
    }
}

static void tts_gate(const Data_t *x, double sx, unsigned n, unsigned L, double si, Data_t *y)
{
    tts_gate_t a;
    a.x = x; a.sx = sx; a.si = si; a.n = n; a.L = L; a.y = y;
    if (!tts_gate_tables(sx, &a.th, &a.sg))
        a.th = a.sg = NULL;
    host_parallel(tts_gate_rows, &a, n, 1u, 1u);
}

typedef struct {
    const Data_t *const *xs;
    const double        *sx;
    unsigned             k, nch, L, rate;
    double               div, si;
    int                  masked, lo, hi, off;
    Data_t              *y;
} tts_sum_t;

/* Sequential double adds (as the reference), then / div; the masked-out
 * columns are one run on each side, written as zeros. */
static void tts_sum_rows(void *p, unsigned c0, unsigned c1)
{
    const tts_sum_t *a = (const tts_sum_t *)p;
    const unsigned   L = a->L, k = a->k;
    const double     si = a->si, div = a->div;
    long             tlo = 0, thi = (long)L, t;
    unsigned         c, i;
    /* two inputs, no division, power-of-two scales: sum x_i * 2^(e_i + m) in
     * int64, then round half to even by 2^m (m = 0: exact) */
    int              e0, e1, e2, fo, m = 0, ishift = 0, a0 = 0, a1 = 0, a2 = 0, j3 = 0, idiv3 = 0;
    int64_t          half = 0, mask = 0;
    if (k == 2u && div == 1.0 && tts_pow2(a->sx[0], &e0) && tts_pow2(a->sx[1], &e1) &&
        tts_pow2(si, &fo)) {
        e0 += fo;
        e1 += fo;
        m = -(e0 < e1 ? e0 : e1);
        m = m > 0 ? m : 0;
        if (e0 + m <= 40 && e1 + m <= 40 && m <= 40) {
            ishift = 1;
            a0 = e0 + m;
            a1 = e1 + m;
            half = m ? (int64_t)1 << (m - 1) : 0;
            mask = ((int64_t)1 << m) - 1;
        }
    }
    /* three inputs / 3 (the resblock average), power-of-two scales: the exact
     * sum acc at 2^-F (F = the finest input exponent), then round half to
     * even of acc / (3 * 2^j), j = F - fo — what the double route gives:
     * a tie is exact, any other value is >= 1 / (6 * 2^j) from a boundary,
     * far beyond the rounding of v / 3 */
    if (k == 3u && div == 3.0 && tts_pow2(a->sx[0], &e0) && tts_pow2(a->sx[1], &e1) &&
        tts_pow2(a->sx[2], &e2) && tts_pow2(si, &fo)) {
        const int lo3 = e0 < e1 ? (e0 < e2 ? e0 : e2) : (e1 < e2 ? e1 : e2);   /* -F */
        a0 = e0 - lo3;
        a1 = e1 - lo3;
        a2 = e2 - lo3;
        j3 = -lo3 - fo;
        if (a0 <= 40 && a1 <= 40 && a2 <= 40 && j3 >= 0 && j3 <= 40)
            idiv3 = a0 <= 13 && a1 <= 13 && a2 <= 13 && j3 <= 27 ? 2 : 1;   /* 2: |acc| < 2^30 */
    }
    if (a->masked) {
        const long sh = (long)a->off * a->rate;
        tlo = tts_clamp((long)a->lo * a->rate - sh, 0, L);
        thi = tts_clamp((long)a->hi * a->rate - sh, tlo, L);
    }
    for (c = c0; c < c1; c++) {
        const size_t     o = (size_t)c * L;
        Data_t *restrict y = a->y + o;
        memset(y, 0, (size_t)tlo * sizeof(Data_t));
        memset(y + thi, 0, (size_t)(L - thi) * sizeof(Data_t));
        if (k == 2u && ishift) {                    /* exact in integers */
            const Data_t *restrict x0 = a->xs[0] + o, *restrict x1 = a->xs[1] + o;
            /* signed << is two's complement in GCC (documented extension) */
            if (m == 0)
                for (t = tlo; t < thi; t++)
                    y[t] = tts_sat(((int64_t)(int16_t)x0[t] << a0) + ((int64_t)(int16_t)x1[t] << a1));
            else
                for (t = tlo; t < thi; t++)
                    y[t] = tts_rshift(((int64_t)(int16_t)x0[t] << a0) + ((int64_t)(int16_t)x1[t] << a1),
                                      m, half, mask);
        } else if (k == 2u && div == 1.0) {
            const Data_t *restrict x0 = a->xs[0] + o, *restrict x1 = a->xs[1] + o;
            const double           s0 = a->sx[0], s1 = a->sx[1];
            for (t = tlo; t < thi; t++)
                y[t] = tts_st((double)(int16_t)x0[t] * s0 + (double)(int16_t)x1[t] * s1, si);
        } else if (k == 3u && idiv3 == 2) {           /* the same in int32 */
            const Data_t *restrict x0 = a->xs[0] + o, *restrict x1 = a->xs[1] + o,
                         *restrict x2 = a->xs[2] + o;
            for (t = tlo; t < thi; t++) {
                const int32_t acc = ((int32_t)(int16_t)x0[t] << a0) + ((int32_t)(int16_t)x1[t] << a1) +
                                    ((int32_t)(int16_t)x2[t] << a2);
                int32_t q3 = acc / 3;                       /* toward zero -> floor */
                q3 -= (acc - (q3 + (q3 << 1))) < 0;
                {
                    int32_t       Q = q3 >> j3;             /* floor(acc / (3 << j3)) */
                    const int32_t Qd = Q << j3;
                    const int32_t R2 = (acc - (Qd + (Qd << 1))) << 1;
                    const int32_t D = 3 << j3;
                    Q += (R2 > D) | ((R2 == D) & (Q & 1));
                    y[t] = tts_sat(Q);
                }
            }
        } else if (k == 3u && idiv3) {
            const Data_t *restrict x0 = a->xs[0] + o, *restrict x1 = a->xs[1] + o,
                         *restrict x2 = a->xs[2] + o;
            const int64_t          D = (int64_t)3 << j3;
            for (t = tlo; t < thi; t++) {
                const int64_t acc = ((int64_t)(int16_t)x0[t] << a0) + ((int64_t)(int16_t)x1[t] << a1) +
                                    ((int64_t)(int16_t)x2[t] << a2);
                int64_t q3 = acc / 3;                       /* toward zero -> floor */
                q3 -= (acc - 3 * q3) < 0;
                {
                    int64_t       Q = q3 >> j3;             /* floor(acc / D) */
                    const int64_t R2 = 2 * (acc - Q * D);   /* 2 * remainder, in [0, 2D) */
                    Q += (R2 > D) | ((R2 == D) & (Q & 1));
                    y[t] = tts_sat(Q);
                }
            }
        } else if (k == 3u) {                          /* four independent chains */
            const Data_t *restrict x0 = a->xs[0] + o, *restrict x1 = a->xs[1] + o,
                         *restrict x2 = a->xs[2] + o;
            const double           s0 = a->sx[0], s1 = a->sx[1], s2 = a->sx[2];
            for (t = tlo; t + 4 <= thi; t += 4) {
                double v0 = (double)(int16_t)x0[t] * s0 + (double)(int16_t)x1[t] * s1;
                double v1 = (double)(int16_t)x0[t + 1] * s0 + (double)(int16_t)x1[t + 1] * s1;
                double v2 = (double)(int16_t)x0[t + 2] * s0 + (double)(int16_t)x1[t + 2] * s1;
                double v3 = (double)(int16_t)x0[t + 3] * s0 + (double)(int16_t)x1[t + 3] * s1;
                v0 = v0 + (double)(int16_t)x2[t] * s2;
                v1 = v1 + (double)(int16_t)x2[t + 1] * s2;
                v2 = v2 + (double)(int16_t)x2[t + 2] * s2;
                v3 = v3 + (double)(int16_t)x2[t + 3] * s2;
                y[t]     = tts_st(v0 / div, si);
                y[t + 1] = tts_st(v1 / div, si);
                y[t + 2] = tts_st(v2 / div, si);
                y[t + 3] = tts_st(v3 / div, si);
            }
            for (; t < thi; t++) {
                double v = (double)(int16_t)x0[t] * s0 + (double)(int16_t)x1[t] * s1;
                v = v + (double)(int16_t)x2[t] * s2;
                y[t] = tts_st(v / div, si);
            }
        } else {
            for (t = tlo; t < thi; t++) {
                double v = (double)(int16_t)a->xs[0][o + t] * a->sx[0];
                for (i = 1u; i < k; i++)
                    v = v + (double)(int16_t)a->xs[i][o + t] * a->sx[i];
                y[t] = tts_st(div != 1.0 ? v / div : v, si);
            }
        }
    }
}

static void tts_sum(const Data_t *const *xs, const double *sx, unsigned k, unsigned nch, unsigned L,
                    double div, int masked, int lo, int hi, unsigned rate, int off, double si,
                    Data_t *y)
{
    tts_sum_t a;
    a.xs = xs; a.sx = sx; a.k = k; a.nch = nch; a.L = L; a.rate = rate; a.div = div; a.si = si;
    a.masked = masked; a.lo = lo; a.hi = hi; a.off = off; a.y = y;
    host_parallel(tts_sum_rows, &a, nch, 1u, 1u);
}

static void tts_flow_out(const float *z, const Data_t *m, double sm, unsigned n, unsigned L,
                         int lo, int hi, float *y)
{
    unsigned c, t;
    for (c = 0u; c < 2u * n; c++) {
        const float *zf = z + (size_t)(2u * n - 1u - c) * L;           /* flip */
        float       *yc = y + (size_t)c * L;
        for (t = 0u; t < L; t++) {
            const int in = (long)t >= lo && (long)t < hi;
            if (!in)
                yc[t] = 0.0f;
            else if (c < n)
                yc[t] = zf[t];
            else
                yc[t] = (float)((double)zf[t] - llm_ld(m[(size_t)(c - n) * L + t], sm));
        }
    }
}

typedef struct {
    const Data_t *x;
    unsigned      s, O, L;
    Data_t       *y;
} tts_interleave_t;

/* y[o][m*s + r] = x[r*O + o][m]: rows of x read in order, written with stride s. */
static void tts_interleave_rows(void *p, unsigned o0, unsigned o1)
{
    const tts_interleave_t *a = (const tts_interleave_t *)p;
    const unsigned          s = a->s, L = a->L;
    unsigned                o, r, m;
    for (o = o0; o < o1; o++)
        for (r = 0u; r < s; r++) {
            const Data_t *restrict x = a->x + ((size_t)r * a->O + o) * L;
            Data_t *restrict       y = a->y + (size_t)o * L * s + r;
            for (m = 0u; m < L; m++)
                y[(size_t)m * s] = x[m];
        }
}

static void tts_interleave(const Data_t *x, unsigned s, unsigned O, unsigned L, Data_t *y)
{
    tts_interleave_t a;
    a.x = x; a.s = s; a.O = O; a.L = L; a.y = y;
    host_parallel(tts_interleave_rows, &a, O, 1u, 1u);
}

static void tts_pcm(const Data_t *x, double sx, unsigned L, int16_t *y)
{
    unsigned t;
    for (t = 0u; t < L; t++) {
        double r = nearbyint(tanh(llm_ld(x[t], sx)) * 32767.0);
        if (r > 32767.0) r = 32767.0;
        if (r < -32768.0) r = -32768.0;
        y[t] = (int16_t)r;
    }
}
"""


TTS_ENC_C = r"""
/* ---- the text encoder's ops (TtsEmbed ... TtsEncOut; encode_<T> entries) ---- */

#define TTS_MAX_HD   256u
#define TTS_MAX_W    8u
#define TTS_MAX_KEYS 1024u
#define TTS_MAX_D    1024u

static void tts_embed(const int32_t *ids, unsigned T, int n, const float *t32, const uint16_t *t16,
                      unsigned rows, unsigned D, double scale, double si, Data_t *y)
{
    unsigned t, c;
    for (t = 0u; t < T; t++) {
        Data_t *yr = y + (size_t)t * D;
        long    k;
        if ((int)t >= n) {
            memset(yr, 0, (size_t)D * sizeof(Data_t));
            continue;
        }
        k = ids[t];
        k = k < 0 ? 0 : (k >= (long)rows ? (long)rows - 1 : k);
        for (c = 0u; c < D; c++) {
            double v;
            if (t32) {
                v = (double)t32[(size_t)k * D + c];
            } else {
                const uint32_t u = (uint32_t)t16[(size_t)k * D + c] << 16;
                float          f;
                memcpy(&f, &u, sizeof f);
                v = (double)f;
            }
            yr[c] = tts_st(v * scale, si);
        }
    }
}

typedef struct {
    const Data_t *x;
    const float  *b;
    unsigned      T, C, taps;
    int           n, relu;
    double        sx, si;
    Data_t       *y;
} tts_rowprep_t;

static void tts_rowprep_rows(void *p, unsigned r0, unsigned r1)
{
    const tts_rowprep_t *a = (const tts_rowprep_t *)p;
    const unsigned       C = a->C, taps = a->taps;
    const int            h = (int)(taps - 1u) / 2;
    unsigned             i, k, c;
    for (i = r0; i < r1; i++) {
        Data_t *yr = a->y + (size_t)i * taps * C;
        for (k = 0u; k < taps; k++) {
            Data_t *restrict yk = yr + (size_t)k * C;
            const int        r = (int)i + (int)k - h;
            const Data_t *restrict xr;
            if ((int)i >= a->n || r < 0 || r >= a->n) {
                memset(yk, 0, (size_t)C * sizeof(Data_t));
                continue;
            }
            xr = a->x + (size_t)r * C;
            for (c = 0u; c < C; c++) {
                double v = (double)(int16_t)xr[c] * a->sx;
                if (a->b)
                    v = v + (double)a->b[c];
                if (a->relu)
                    v = v > 0.0 ? v : 0.0;
                yk[c] = tts_stf(v, a->si);
            }
        }
    }
}

static void tts_rowprep(const Data_t *x, unsigned T, unsigned C, int n, const float *b, int relu,
                        unsigned taps, double sx, double si, Data_t *y)
{
    tts_rowprep_t a;
    a.x = x; a.b = b; a.T = T; a.C = C; a.taps = taps; a.n = n; a.relu = relu; a.sx = sx; a.si = si;
    a.y = y;
    host_parallel(tts_rowprep_rows, &a, T, 1u, 1u);
}

typedef struct {
    const Data_t *s, *q0;
    const float  *bq, *ek;
    unsigned      T, C, D, HD, g, W;
    int           n;
    double        ss, sq0, iq, ip;
    Data_t       *p;
} tts_asoft_t;

static void tts_attn_softmax_rows(void *pp, unsigned r0, unsigned r1)
{
    const tts_asoft_t *a = (const tts_asoft_t *)pp;
    const unsigned     T = a->T, C = a->C, HD = a->HD, W = a->W, NW = 2u * a->W + 1u;
    const int          n = a->n;
    double             qv[TTS_MAX_HD], rel[2u * TTS_MAX_W + 1u], v[TTS_MAX_KEYS];
    unsigned           i, d, k;
    int                j;
    for (i = r0; i < r1; i++) {
        Data_t       *pr = a->p + (size_t)i * C;
        const Data_t *qr = a->q0 + (size_t)i * a->D + (size_t)a->g * HD;
        const float  *bq = a->bq + (size_t)a->g * HD;
        double        m, sum;
        memset(pr, 0, (size_t)C * sizeof(Data_t));
        if ((int)i >= n)
            continue;
        for (d = 0u; d < HD; d++) {                   /* q as the q / k / v prep writes it */
            double r = nearbyint(((double)(int16_t)qr[d] * a->sq0 + (double)bq[d]) * a->iq);
            r = r > 32767.0 ? 32767.0 : (r < -32768.0 ? -32768.0 : r);
            qv[d] = r / a->iq;
        }
        for (k = 0u; k < NW; k++) {                   /* relative keys, sums left to right */
            const float *e   = a->ek + (size_t)k * HD;
            double       acc = 0.0;
            for (d = 0u; d < HD; d++)
                acc += qv[d] * (double)e[d];
            rel[k] = acc;
        }
        m = -INFINITY;
        for (j = 0; j < n; j++) {
            const int dk = j - (int)i;
            double    x  = (double)(int16_t)a->s[(size_t)j * T + i] * a->ss;
            if (dk >= -(int)W && dk <= (int)W)
                x = x + rel[dk + (int)W];
            v[j] = x;
            m = x > m ? x : m;
        }
        sum = 0.0;
        for (j = 0; j < n; j++) {
            v[j] = exp(v[j] - m);
            sum += v[j];
        }
        for (j = 0; j < n; j++) {
            double r = nearbyint(v[j] / sum * a->ip);
            pr[j] = (Data_t)(int16_t)(r > 32767.0 ? 32767.0 : r);
        }
    }
}

static void tts_attn_softmax(const Data_t *s, const Data_t *q0, const float *bq, const float *ek,
                             unsigned T, unsigned C, unsigned D, unsigned HD, unsigned g, unsigned W,
                             int n, double ss, double sq0, double iq, double ip, Data_t *p)
{
    tts_asoft_t a;
    if (HD > TTS_MAX_HD || W > TTS_MAX_W || C > TTS_MAX_KEYS) {
        fprintf(stderr, "tts_attn_softmax: head_dim %u / window %u / keys %u beyond the limits\n",
                HD, W, C);
        abort();
    }
    a.s = s; a.q0 = q0; a.bq = bq; a.ek = ek; a.T = T; a.C = C; a.D = D; a.HD = HD; a.g = g; a.W = W;
    a.n = n < (int)C ? n : (int)C; a.ss = ss; a.sq0 = sq0; a.iq = iq; a.ip = ip; a.p = p;
    host_parallel(tts_attn_softmax_rows, &a, T, 1u, 1u);
}

typedef struct {
    const Data_t *const *o, *const *p;
    const double        *so;
    const float         *ev;
    unsigned             T, C, H, HD, W;
    int                  n;
    double               sp, si;
    Data_t              *y;
} tts_amerge_t;

static void tts_attn_merge_rows(void *pp, unsigned r0, unsigned r1)
{
    const tts_amerge_t *a = (const tts_amerge_t *)pp;
    const unsigned      HD = a->HD, W = a->W, C = a->C;
    unsigned            i, h, d;
    int                 k;
    for (i = r0; i < r1; i++) {
        Data_t *yr = a->y + (size_t)i * a->H * HD;
        if ((int)i >= a->n) {
            memset(yr, 0, (size_t)a->H * HD * sizeof(Data_t));
            continue;
        }
        for (h = 0u; h < a->H; h++) {
            const Data_t *orow = a->o[h] + (size_t)i * HD;
            const Data_t *prow = a->p[h] + (size_t)i * C;
            for (d = 0u; d < HD; d++) {
                double v = (double)(int16_t)orow[d] * a->so[h];
                for (k = -(int)W; k <= (int)W; k++) {       /* relative values */
                    const int j = (int)i + k;
                    if (j >= 0 && j < a->n)
                        v = v + ((double)(int16_t)prow[j] * a->sp) * (double)a->ev[(size_t)(k + (int)W) * HD + d];
                }
                yr[h * HD + d] = tts_stf(v, a->si);
            }
        }
    }
}

static void tts_attn_merge(const Data_t *const *o, const Data_t *const *p, const double *so,
                           const float *ev, unsigned T, unsigned C, unsigned H, unsigned HD, unsigned W,
                           int n, double sp, double si, Data_t *y)
{
    tts_amerge_t a;
    a.o = o; a.p = p; a.so = so; a.ev = ev; a.T = T; a.C = C; a.H = H; a.HD = HD; a.W = W;
    a.n = n < (int)C ? n : (int)C; a.sp = sp; a.si = si; a.y = y;
    host_parallel(tts_attn_merge_rows, &a, T, 1u, 1u);
}

typedef struct {
    const Data_t *x, *y;
    const float  *by, *g, *b;
    unsigned      T, D;
    int           n;
    double        sx, sy, eps, si;
    Data_t       *out;
} tts_resnorm_t;

static void tts_resnorm_rows(void *pp, unsigned r0, unsigned r1)
{
    const tts_resnorm_t *a = (const tts_resnorm_t *)pp;
    const unsigned       D = a->D;
    double               sv[TTS_MAX_D];
    unsigned             i, c;
    for (i = r0; i < r1; i++) {
        const Data_t *xr = a->x + (size_t)i * D, *yr = a->y + (size_t)i * D;
        Data_t       *orow = a->out + (size_t)i * D;
        double        sum = 0.0, mean, acc = 0.0, sd;
        if ((int)i >= a->n) {
            memset(orow, 0, (size_t)D * sizeof(Data_t));
            continue;
        }
        for (c = 0u; c < D; c++) {
            sv[c] = (double)(int16_t)xr[c] * a->sx + ((double)(int16_t)yr[c] * a->sy + (double)a->by[c]);
            sum += sv[c];
        }
        mean = sum / (double)D;
        for (c = 0u; c < D; c++) {
            const double d = sv[c] - mean;
            acc += d * d;
        }
        sd = sqrt(acc / (double)D + a->eps);
        for (c = 0u; c < D; c++)
            orow[c] = tts_stf((sv[c] - mean) / sd * (double)a->g[c] + (double)a->b[c], a->si);
    }
}

static void tts_resnorm(const Data_t *x, const Data_t *y, const float *by, const float *g, const float *b,
                        unsigned T, unsigned D, int n, double sx, double sy, double eps, double si,
                        Data_t *out)
{
    tts_resnorm_t a;
    if (D > TTS_MAX_D) {
        fprintf(stderr, "tts_resnorm: %u channels beyond %u\n", D, TTS_MAX_D);
        abort();
    }
    a.x = x; a.y = y; a.by = by; a.g = g; a.b = b; a.T = T; a.D = D; a.n = n; a.sx = sx; a.sy = sy;
    a.eps = eps; a.si = si; a.out = out;
    host_parallel(tts_resnorm_rows, &a, T, 1u, 1u);
}

static void tts_enc_out(const Data_t *x, const float *b, unsigned T, unsigned C, int n, double sx, float *y)
{
    unsigned i, c;
    for (i = 0u; i < T; i++)
        for (c = 0u; c < C; c++) {
            double v = (double)(int16_t)x[(size_t)i * C + c] * sx;
            if (b)
                v = v + (double)b[c];
            y[(size_t)i * C + c] = (int)i < n ? (float)v : 0.0f;
        }
}
"""


def tts_c_helpers() -> str:
    return TTS_C + TTS_ENC_C
