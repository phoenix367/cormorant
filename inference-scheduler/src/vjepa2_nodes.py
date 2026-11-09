# Copyright 2026 Ivan Gubochkin
# SPDX-License-Identifier: Apache-2.0

"""
V-JEPA 2 ops (the video classifier facebook/vjepa2-vitl-fpc16-256-ssv2,
doc/plans/VJEPA2_PLAN.md; the frontend src/vjepa2.py).  The specification is
demo/video_classification/scripts/vjepa2_q.py (vjepa2_q.VJEPA2 with the
formats): the generated project computes what it computes, bit for bit.

Row-ops node:

  VjResCopy    x [rows][n] -> y, one COPY with a per-column bias, the sum at
               2^-fp (the int32 side's exponent): into the int32 residual (x
               int16 at per-column exponents: the patch embedding plus its
               folded bias; x int16: the encoder's output, the pooler's
               residual) or out of it (x the int32 residual -> int16 at
               per-column exponents: the pooled row the classifier reads) —
               y = sat(rne(((x << (fp - f_x)) + (round(b 2^bb) << (fp - bb)))
               >> (fp - f_y))), the bias rounded once at 2^-bb
               (rowops_spec.bias_bits: fp, coarser only where the int22 would
               overflow) — vjepa2_q's ibias / res32 / host write-back

Host ops (numpy reference + the C helpers of VJ_C):

  VjCrossAttn  the attentive pooler's cross attention, one query row: k0 / v0
               [N][D] (the cross layer's k / v MatMuls) -> r [1][D], the int32
               residual at 2^-fr: k = k0 + bk and v = v0 + bv rounded at the
               per-head k / per-channel v exponents (rowops_spec.bias_round),
               per head s = floor(q . k^T / 2^8) saturated (ConvKernel's
               ap_fixed<32,16> sum), P = the softmax unit's integer
               specification over the N keys (vectorop_smx.softmax_raw), o =
               floor(P . v / 2^8) saturated at f_p + f_v - 8, then r =
               round_half_even((query + o) 2^fr) — query the learned query
               token (float32), q its projection rounded at f_q (a constant of
               the model)
  VjLogits     x [rows][n] (int16) -> float32(x 2^-f + b) (the classifier's
               output with its bias, float32 host)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import ClassVar, Optional, Tuple

import numpy as np

from . import rowops_spec as rs
from . import vectorop_smx as smx
from .host_nodes import HostContext, _attrs, _label
from .llm_nodes import LlmNode, _attr_ints, _const_array, _llm_inputs, _resolve
from .nodes import SchedulerError
from .rowops_nodes import (NO_WINDOW, RowOpsNode, _exps, _need_unit, _pblock, row_stride_bytes,
                           tables_tensor)


def _require(cond, node, msg):
    if not cond:
        raise SchedulerError(f"{node.op_type} node '{_label(node)}': {msg}")


def _wrap_floor(acc: np.ndarray) -> np.ndarray:
    """ConvKernel's output of exact integer sums: ap_fixed<32,16> wraps, then
    floor(acc / 2^8), saturated to int16 (vjepa2_q's kmm)."""
    acc = np.asarray(acc, np.int64)
    acc = ((acc + (1 << 31)) & 0xFFFFFFFF) - (1 << 31)
    return np.clip(acc >> 8, rs.I16_LO, rs.I16_HI)


# ------------------------------------------------------------------ #
# VjResCopy: a COPY with a bias into / out of the int32 residual       #
# ------------------------------------------------------------------ #

@dataclass
class ResCopyRoNode(RowOpsNode):
    """x [rows][n] -> y [rows][n]: one COPY (A i16 / i32, C i32 / i16) with the
    bias in the parameter block — the sum exact at 2^-fp, one rounding at y's
    exponents.  inputs [x, P, tables]."""
    rows:  int = 1
    n:     int = 1
    fp:    int = 15
    bb:    int = 15
    a32:   bool = False
    c32:   bool = True
    bcast: bool = True
    sa:    Optional[np.ndarray] = field(default=None, repr=False)     # [n] fp - f_x
    sc:    Optional[np.ndarray] = field(default=None, repr=False)     # [n] fp - f_y
    bq:    Optional[np.ndarray] = field(default=None, repr=False)     # [n] round(bias 2^bb)

    @property
    def opword(self) -> int:
        return rs.opword(rs.OP_COPY, rs.K_I32 if self.a32 else rs.K_I16, rs.K_I16,
                         rs.K_I32 if self.c32 else rs.K_I16, p_bcast=self.bcast)

    def _strides(self, layouts):
        nd, al = self.onnx_node, self.align_elems
        return (row_stride_bytes(self.inputs[0], layouts, self.n, 4 if self.a32 else 2, nd, al),
                row_stride_bytes(self.output, layouts, self.n, 4 if self.c32 else 2, nd, al))

    def reference(self, ins, dtype):  # noqa: ARG002
        fx = _exps(self.inputs[0], self.F).astype(np.float64)
        fy = _exps(self.output, self.F)
        x = np.rint(np.asarray(ins[0], np.float64).reshape(self.rows, self.n) * np.power(2.0, fx)[None, :])
        s = (x.astype(np.int64) << self.sa[None, :]) + (self.bq[None, :] << (self.fp - self.bb))
        y = rs.rne_shr_vec(s, self.sc[None, :])
        y = np.clip(y, rs.I32_LO, rs.I32_HI) if self.c32 else rs.sat16(y)
        return (y / np.power(2.0, fy.astype(np.float64))[None, :]).reshape(self.output.shape)

    def kernel_calls(self, layouts: dict) -> list:
        from .perf_calls import KernelCall
        a_s1, c_s1 = self._strides(layouts)
        return [KernelCall.of("RowOpsKernel", op=self.opword, n=self.n, rows=self.rows, blocks=1,
                              a_s1=a_s1, c_s1=c_s1, win_hi=NO_WINDOW)]

    def describe(self) -> str:
        return (f"rows={self.rows} n={self.n} on RowOpsKernel (COPY {'int32' if self.a32 else 'int16'} + bias at "
                f"2^-{self.bb} -> {'int32' if self.c32 else 'int16'}, the sum at 2^-{self.fp})")

    def emit_comment(self) -> str:
        return (f"    /* [{self.index}] VjResCopy({self.inputs[0].onnx_name}) -> {self.output.onnx_name}  "
                f"[{self.rows}][{self.n}] on RowOpsKernel (COPY with a bias, "
                f"{'into' if self.c32 else 'out of'} the int32 residual) */")

    def emit_call(self, layouts: dict) -> str:
        a_s1, c_s1 = self._strides(layouts)
        return self.call(self.inputs[0].c_name, 0, "NULL", 0, self.output.c_name, 0, self.inputs[1].c_name,
                         self.opword, self.n, self.rows, 1, a_s1, 0, 0, 0, c_s1, 0, 0, 0, self.fp - self.bb, 0,
                         NO_WINDOW)


def from_vj_res_copy(node, tensors, index, align_elems, ctx: HostContext):
    """VjResCopy(x, bias) -> y: x or y the int32 DMA residual (one exponent), the
    other an int16 DMA tensor (per-column exponents)."""
    _need_unit(node, "VjResCopy")
    F = ctx.frac_bits
    ins = _llm_inputs(node, tensors)
    _require(len(ins) == 2, node, "inputs: x, bias")
    x = ins[0]
    y = _resolve(tensors, node.output[0], node)
    n = int(x.shape[-1])
    _require(x.numel == y.numel and int(y.shape[-1]) == n, node, "x and y must have the same shape")
    _require(x.host is None and not x.is_state and y.host is None and not y.is_state, node, "DMA tensors")
    _require(x.dma in (None, "i32") and y.dma in (None, "i32") and (x.dma == "i32") != (y.dma == "i32"),
             node, "one side the int32 residual, the other an int16 DMA tensor")
    fx, fy = _exps(x, F), _exps(y, F)
    a32 = x.dma == "i32"
    side = fx if a32 else fy
    _require((side == side[0]).all(), node, "the int32 side at one exponent")
    fp = int(side[0])
    bias = np.asarray(_const_array(ctx, tensors, node.input[1], node, "bias"), np.float64).reshape(-1)
    _require(bias.size == n and np.isfinite(bias).all(), node, f"bias [{n}]")
    bb = int(rs.bias_bits(bias, fp))
    bq = np.rint(bias * 2.0 ** bb).astype(np.int64)
    sa, sc = fp - fx, fp - fy
    _require((sa >= 0).all() and (sa <= 63).all() and (sc >= 0).all() and (sc <= 63).all()
             and 0 <= fp - bb <= 63, node, f"exponents beyond the sum's 2^-{fp}")
    _require(n % align_elems == 0, node, f"row length {n} % {align_elems}")
    p, bc = _pblock(tensors, f"{y.onnx_name}__rowops_p", 0, bq, sa, 0, sc, n)
    return ResCopyRoNode(onnx_node=node, inputs=[x, p, tables_tensor(tensors)], output=y, index=index,
                         align_elems=align_elems, F=F, rows=x.numel // n, n=n, fp=fp, bb=bb, a32=a32,
                         c32=not a32, bcast=bc, sa=sa, sc=sc, bq=bq)


# ------------------------------------------------------------------ #
# VjCrossAttn: the attentive pooler's one-row cross attention          #
# ------------------------------------------------------------------ #

def cross_attention(k0, fk0, bk, fk, v0, fv0, bv, fv, q_raw, fq, fp, cm, cs, query, fr, H, HD):
    """The specification of VjCrossAttn on raw integers (int64 arrays): k0 / v0
    [N][D] at fk0 / fv0 [D]; fk / fq / fp / cm / cs [H]; fv [D]; q_raw [D];
    query [D] float; -> r [D], the raw int32 residual at 2^-fr."""
    D = H * HD
    k = rs.bias_round(k0, fk0, bk, np.repeat(fk, HD))
    v = rs.bias_round(v0, fv0, bv, fv)
    o = np.zeros(D)
    fpv = np.zeros(D, np.int64)
    for h in range(H):
        sl = slice(h * HD, (h + 1) * HD)
        s = _wrap_floor(k[:, sl] @ q_raw[sl])                        # [N]
        P = smx.softmax_raw(s[None, :], s.size, int(cm[h]), int(cs[h]), int(fp[h]))[0]
        o[sl] = _wrap_floor(P @ v[:, sl])
        fpv[sl] = int(fp[h]) + fv[sl] - 8
    att = o / np.power(2.0, fpv.astype(np.float64))
    raw = np.clip(np.rint((np.asarray(query, np.float64) + att) * 2.0 ** fr), rs.I32_LO, rs.I32_HI) + 0.0
    return raw


@dataclass
class VjCrossAttnNode(LlmNode):
    """k0, v0 [N][D] (int16 DMA) -> r [1][D] (the int32 DMA residual)."""
    helpers: ClassVar[Tuple[str, ...]] = ("llm", "vj")
    N:     int = 1
    H:     int = 1
    HD:    int = 16
    fr:    int = 15
    fk:    Optional[np.ndarray] = field(default=None, repr=False)      # [H]
    fv:    Optional[np.ndarray] = field(default=None, repr=False)      # [D]
    fq:    Optional[np.ndarray] = field(default=None, repr=False)      # [H]
    fp:    Optional[np.ndarray] = field(default=None, repr=False)      # [H]
    cm:    Optional[np.ndarray] = field(default=None, repr=False)      # [H]
    cs:    Optional[np.ndarray] = field(default=None, repr=False)      # [H]
    bk:    Optional[np.ndarray] = field(default=None, repr=False)
    bv:    Optional[np.ndarray] = field(default=None, repr=False)
    q_raw: Optional[np.ndarray] = field(default=None, repr=False)      # [D]
    query: Optional[np.ndarray] = field(default=None, repr=False)      # [D] float32 values

    @classmethod
    def from_onnx_node(cls, node, tensors, index, align_elems, ctx: HostContext):
        ins = _llm_inputs(node, tensors)
        _require(len(ins) == 6, node, "inputs: k0, v0, bk, bv, q, query")
        k0, v0 = ins[0], ins[1]
        y = _resolve(tensors, node.output[0], node)
        a = _attrs(node)
        H, HD = int(a["num_heads"]), int(a["head_dim"])
        D = H * HD
        N = k0.numel // D
        cst = lambda i, w: np.asarray(_const_array(ctx, tensors, node.input[i], node, w),   # noqa: E731
                                      np.float64).reshape(-1)
        bk, bv, q, query = cst(2, "bk"), cst(3, "bv"), cst(4, "q"), cst(5, "query")
        fk = np.asarray(_attr_ints(a, "k_exp", node, H), np.int64)
        fv = np.asarray(_attr_ints(a, "v_exp", node, D), np.int64)
        fq = np.asarray(_attr_ints(a, "q_exp", node, H), np.int64)
        fp = np.asarray(_attr_ints(a, "p_exp", node, H), np.int64)
        _require(all(t.size == D for t in (bk, bv, q, query)), node, f"bk / bv / q / query: [{D}]")
        q_raw = np.rint(q * np.power(2.0, np.repeat(fq, HD).astype(np.float64)))
        _require((np.abs(q_raw) <= 32767).all() and (q_raw * np.power(2.0, -np.repeat(fq, HD).astype(
            np.float64)) == q).all(), node, "q: int16 values at q_exp")
        fs = fq + fk - 8
        regs = [smx.scale_regs(int(f), 1.0 / math.sqrt(HD)) for f in fs]
        _require(N <= smx.MAX_ROW and N >= 1, node, f"{N} keys (at most {smx.MAX_ROW})")
        sn = cls(onnx_node=node, inputs=[k0, v0], output=y, index=index, align_elems=align_elems,
                 F=ctx.frac_bits, N=N, H=H, HD=HD, fk=fk, fv=fv, fq=fq, fp=fp,
                 cm=np.array([r[0] for r in regs], np.int64), cs=np.array([r[1] for r in regs], np.int64),
                 bk=bk, bv=bv, q_raw=q_raw.astype(np.int64), query=query)
        sn._want(k0, None, "k0")
        sn._want(v0, None, "v0")
        _require(sn._want_residual(y, "output") and y.numel == D, node, f"output: the int32 residual [1][{D}]")
        _require(k0.numel == v0.numel == N * D and int(k0.shape[-1]) == D and int(v0.shape[-1]) == D, node,
                 f"k0 / v0: [{N}][{D}]")
        fo = _exps(y, ctx.frac_bits)
        _require((fo == fo[0]).all(), node, "the residual at one exponent")
        sn.fr = int(fo[0])
        for t, b, f, w in ((k0, bk, np.repeat(fk, HD), "k"), (v0, bv, fv, "v")):
            try:
                rs.bias_params(b, _exps(t, sn.F), f)
            except ValueError as e:
                raise SchedulerError(f"{node.op_type} node '{_label(node)}': {w}: {e}") from None
        fpv = np.repeat(fp, HD) + fv - 8
        _require((fpv <= sn.fr).all(), node, f"the attention output's exponents beyond 2^-{sn.fr}")
        return sn

    def _params(self):
        D = self.H * self.HD
        F = self.F
        kb, ksa, ksc = rs.bias_params(self.bk, _exps(self.inputs[0], F), np.repeat(self.fk, self.HD))
        vb, vsa, vsc = rs.bias_params(self.bv, _exps(self.inputs[1], F), self.fv)
        osh = self.fr - (np.repeat(self.fp, self.HD) + self.fv - 8)
        return D, (kb, ksa, ksc), (vb, vsa, vsc), osh

    def describe(self):
        return (f"{self.H} heads x {self.N} keys, one query row (the unit's softmax spec), + the query -> "
                f"int32 at 2^-{self.fr}")

    def host_features(self):
        D = self.H * self.HD
        return {"in_elems": float(2 * self.N * D), "out_elems": float(D), "rows": float(self.N)}

    def c_call(self, ins, out, scratch, direct, dtype):
        D, (kb, ksa, ksc), (vb, vsa, vsc), osh = self._params()
        H = self.H
        arr = lambda ctype, name, v, n: (f"    static const {ctype} {name}[{n}] = {{ "   # noqa: E731
                                         + ", ".join(str(int(x)) for x in np.asarray(v).reshape(-1)) + " };")
        return ["{",
                arr("int32_t", "_kb", kb, D), arr("unsigned char", "_ksa", ksa, D),
                arr("unsigned char", "_ksc", ksc, D),
                arr("int32_t", "_vb", vb, D), arr("unsigned char", "_vsa", vsa, D),
                arr("unsigned char", "_vsc", vsc, D),
                arr("int16_t", "_q", self.q_raw, D), arr("unsigned char", "_osh", osh, D),
                f"    static const uint32_t _cm[{H}] = {{ " + ", ".join(f"{int(v)}u" for v in self.cm) + " };",
                arr("int", "_cs", self.cs, H), arr("int", "_fp", self.fp, H),
                f"    static const float _query[{D}] = {{ "
                + ", ".join(_c_float(float(v)) for v in self.query) + " };",
                f"    vj_cross_attn({ins[0]}, {ins[1]}, {self.N}u, {H}u, {self.HD}u, _kb, _ksa, _ksc, _vb, _vsa, "
                f"_vsc, _q, _cm, _cs, _fp, _osh, _query, {self.fr}, {out});",
                "}"]

    def reference(self, ins, dtype):  # noqa: ARG002
        F, N, D = self.F, self.N, self.H * self.HD
        raw = lambda t, x: np.rint(np.asarray(x, np.float64).reshape(N, D)                 # noqa: E731
                                   * np.power(2.0, _exps(t, F).astype(np.float64))[None, :]).astype(np.int64)
        r = cross_attention(raw(self.inputs[0], ins[0]), _exps(self.inputs[0], F), self.bk, self.fk,
                            raw(self.inputs[1], ins[1]), _exps(self.inputs[1], F), self.bv, self.fv,
                            self.q_raw, self.fq, self.fp, self.cm, self.cs, self.query, self.fr, self.H, self.HD)
        return (r / 2.0 ** self.fr).reshape(self.output.shape)


def _c_float(v: float) -> str:
    """A float32 value as an exact C literal."""
    return f"{float(np.float32(v))!r}f"


# ------------------------------------------------------------------ #
# VjLogits                                                             #
# ------------------------------------------------------------------ #

@dataclass
class VjLogitsNode(LlmNode):
    """x [rows][n] (int16 at per-column exponents) -> float32(x 2^-f + b) (host f32)."""
    helpers: ClassVar[Tuple[str, ...]] = ("llm", "vj")
    rows: int = 1
    n:    int = 1
    b:    Optional[np.ndarray] = field(default=None, repr=False)

    @classmethod
    def from_onnx_node(cls, node, tensors, index, align_elems, ctx: HostContext):
        ins = _llm_inputs(node, tensors)
        _require(len(ins) == 2, node, "inputs: x, bias")
        x = ins[0]
        y = _resolve(tensors, node.output[0], node)
        n = int(x.shape[-1])
        b = np.asarray(_const_array(ctx, tensors, node.input[1], node, "bias"), np.float32).reshape(-1)
        _require(b.size == n, node, f"bias [{n}]")
        sn = cls(onnx_node=node, inputs=[x], output=y, index=index, align_elems=align_elems, F=ctx.frac_bits,
                 rows=x.numel // n, n=n, b=b)
        sn._want(x, None, "x")
        sn._want(y, "f32", "output")
        _require(y.numel == x.numel, node, "shapes")
        return sn

    def c_runtime(self):
        return super().c_runtime() + [self._scales(self.inputs[0])[2]]

    def describe(self):
        return f"rows={self.rows} n={self.n} -> float32 + bias"

    def c_call(self, ins, out, scratch, direct, dtype):
        s = self._scales(self.inputs[0])[0]
        return ["{",
                f"    static const float _b[{self.n}] = {{ " + ", ".join(_c_float(float(v)) for v in self.b) + " };",
                f"    vj_logits({ins[0]}, {s}, _b, {self.rows}u, {self.n}u, {out});",
                "}"]

    def reference(self, ins, dtype):  # noqa: ARG002
        x = np.asarray(ins[0], np.float64).reshape(self.rows, self.n)
        return (x + self.b.astype(np.float64)[None, :]).astype(np.float32).astype(np.float64).reshape(
            self.output.shape)


VJ_OP_FACTORIES = {
    "VjResCopy":   from_vj_res_copy,
    "VjCrossAttn": VjCrossAttnNode.from_onnx_node,
    "VjLogits":    VjLogitsNode.from_onnx_node,
}


# ------------------------------------------------------------------ #
# C helpers                                                            #
# ------------------------------------------------------------------ #

def _tab_literal() -> str:
    t = [int(v) for v in smx.TAB]
    rows = ",\n".join("    " + ", ".join(f"{v}u" for v in t[i:i + 12]) for i in range(0, len(t), 12))
    return f"static const uint32_t vj_smx_tab[{len(t)}] = {{\n{rows}\n}};\n"


VJ_C = r"""
/* ---- V-JEPA 2 host ops (src/vjepa2_nodes.py; doc/plans/VJEPA2_PLAN.md) ---- */

/* round half to even of v / 2^k, k >= 0 (rowops_spec.rne_shr) */
static int64_t vj_rne_shr(int64_t v, unsigned k)
{
    int64_t q, r, half;
    if (k == 0u) return v;
    q = v >> k;
    r = v - q * ((int64_t)1 << k);
    half = (int64_t)1 << (k - 1u);
    return q + ((r > half || (r == half && (q & 1))) ? 1 : 0);
}

static int64_t vj_sat16(int64_t v)
{
    return v > 32767 ? 32767 : (v < -32768 ? -32768 : v);
}

/* A COPY with a bias (rowops_spec.bias_round): sat16(rne((x << sa) + bq, sc)). */
static int64_t vj_bias_round(int16_t x, int32_t bq, unsigned char sa, unsigned char sc)
{
    return vj_sat16(vj_rne_shr((int64_t)x * ((int64_t)1 << sa) + (int64_t)bq, sc));
}

/* ConvKernel's output of an exact sum: ap_fixed<32,16> wraps, floor(acc / 2^8), saturate. */
static int64_t vj_kout(int64_t acc)
{
    const int64_t w = (int64_t)(int32_t)(uint32_t)(uint64_t)acc;
    return vj_sat16(w >> 8);
}

/* The attentive pooler's cross attention, one query row (VjCrossAttn): per head h,
 * s_j = kout(sum_d q[d] k[j][d]) over the N keys (k = k0 + bk rounded at the head's
 * exponent), P = the softmax unit's integer specification (vectorop_smx.softmax_raw,
 * every key valid), o[d] = kout(sum_j P_j v[j][d]) (v = v0 + bv rounded); then
 * r[c] = round_half_even((query[c] + o[c] 2^-fpv[c]) 2^fr), saturated; osh = fr - fpv. */
static void vj_cross_attn(const Data_t *k0, const Data_t *v0, unsigned N, unsigned H, unsigned HD,
                          const int32_t *kb, const unsigned char *ksa, const unsigned char *ksc,
                          const int32_t *vb, const unsigned char *vsa, const unsigned char *vsc,
                          const int16_t *q, const uint32_t *cm, const int *cs, const int *fp,
                          const unsigned char *osh, const float *query, int fr, int32_t *r)
{
    const unsigned D = H * HD;
    const double   sc = ldexp(1.0, fr);
    uint64_t      *e = (uint64_t *)malloc((size_t)N * sizeof(uint64_t));
    int64_t       *s = (int64_t *)malloc((size_t)N * sizeof(int64_t));
    unsigned       h, j, d;
    if (!e || !s) {
        free(e);
        free(s);
        memset(r, 0, (size_t)D * sizeof(int32_t));
        return;
    }
    for (h = 0u; h < H; h++) {
        const unsigned c0 = h * HD;
        int64_t        m = INT64_MIN;
        uint64_t       S = 0u, R;
        const int      sh = 40 - fp[h];
        for (j = 0u; j < N; j++) {
            const Data_t *kr = k0 + (size_t)j * D + c0;
            int64_t       acc = 0;
            for (d = 0u; d < HD; d++)
                acc += (int64_t)q[c0 + d] * vj_bias_round((int16_t)kr[d], kb[c0 + d], ksa[c0 + d], ksc[c0 + d]);
            s[j] = vj_kout(acc);
            if (s[j] > m) m = s[j];
        }
        for (j = 0u; j < N; j++) {
            const uint64_t y = ((uint64_t)(m - s[j]) * cm[h]) >> cs[h];
            const uint64_t k = y >> 12;
            e[j] = (uint64_t)vj_smx_tab[y & 4095u] >> (k > 63u ? 63u : (unsigned)k);
            S += e[j];
        }
        R = S ? (1ull << 40) / S : 0u;
        for (j = 0u; j < N; j++) {
            const uint64_t p = (e[j] * R + (1ull << (sh - 1))) >> sh;
            e[j] = p > 32767u ? 32767u : p;
        }
        for (d = 0u; d < HD; d++) {
            const unsigned c = c0 + d;
            int64_t        acc = 0;
            for (j = 0u; j < N; j++)
                acc += (int64_t)e[j] * vj_bias_round((int16_t)v0[(size_t)j * D + c], vb[c], vsa[c], vsc[c]);
            r[c] = llm_st32((double)query[c] * sc + (double)(vj_kout(acc) * ((int64_t)1 << osh[c])));
        }
    }
    free(e);
    free(s);
}

/* y[r][c] = float32(x[r][c] 2^-f[c] + b[c]) (VjLogits). */
static void vj_logits(const Data_t *x, const double *sx, const float *b, unsigned rows, unsigned n, float *y)
{
    unsigned r, c;
    for (r = 0u; r < rows; r++)
        for (c = 0u; c < n; c++)
            y[(size_t)r * n + c] = (float)(llm_ld(x[(size_t)r * n + c], sx[c]) + (double)b[c]);
}
"""


def vj_c_helpers() -> str:
    return _tab_literal() + VJ_C


__all__ = ("VJ_OP_FACTORIES", "vj_c_helpers", "cross_attention", "ResCopyRoNode", "VjCrossAttnNode",
           "VjLogitsNode", "from_vj_res_copy")
