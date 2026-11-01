# Copyright 2026 Ivan Gubochkin
# SPDX-License-Identifier: Apache-2.0

"""
SmolVLA ops (vision-language-action, doc/plans/VLA_PLAN.md; the frontend
src/smolvla.py).  The specification is demo/vla/scripts/smolvla_q.py (SmolVLA
with the formats): the generated project computes what it computes, bit for
bit.

Host ops (numpy reference + the C helpers of VLA_C):

  VlaPrefix     ids [L] (i32), n (i32: the prefix's valid rows), state [S]
                (f32), the cameras' image features [I][D] (f32 host states) ->
                the residual [T][D], the int32 DMA tensor at 2^-fr: rows
                [c I, (c + 1) I) camera c's features, then nl = n - C I - 1
                language rows table[id] * scale, then the state row (sum_i
                state[i] W[i][d] in double, left to right, + b[d]), zeros past
                it; round half even at 2^fr, saturate
  VlaActIn      x [R][A] (f32 host) -> [T][A] int16 at the output's exponents
                (round half even, saturate), rows >= R zero
  VlaTimeSilu   a0 [T][N] (int16), step (i32) -> silu(a0 + b[step]): the bias
                an integer at a0's exponent (bq, rounded by the frontend), the
                sum saturated, SiLU from the 65 536-entry table of a0's
                exponent (llm_silu_table), written at the output's exponents
  VlaVelocity   v0 [T][A] (int16) -> v [R][A] (f32 host) = float32(v0 + b)
  VlaStateAttn  the prefix's last valid row (the state token, which alone
                attends its own key — the prefix-LM mask): per head of KV group
                g, from the q.K^T output s_g [C][G T] (column h' T + n - 1) and
                the V cache, P over the keys < n by the softmax unit's integer
                specification (vectorop_smx.softmax_raw) and P.V as ConvKernel
                computes it (exact sum, floor(/ 2^8), saturate) -> [G][HD]
                (int16 host)
  VlaAttnMerge  LlmAttnMerge, then row n - 1 replaced by the VlaStateAttn rows

Row-ops node:

  VlaAttnPrep   LlmAttnPrep's jobs (rowops_nodes.AttnPrepRoNode) with parts left
                out: ``parts = "q"`` (the expert's cross layers: RoPE(q0) at
                positions pos + t -> the q staging rows, the q.K^T image; no
                cache), ``parts = "k"`` (the VLM's odd layers: RoPE(k0) of all T
                rows at the K cache's exponents -> the output, which the cross
                K' MatMul reads), ``parts = "v"`` (v0 of all T rows re-rounded at
                the output's exponents — the V' MatMul's input, its exponent
                capped like a host-written input's: one COPY)

The prefix-LM softmax itself is LlmAttnSoftmax with ``causal = 2``
(src/smx_nodes.py: every row over the keys < pos + n - 1; the last valid
row is VlaStateAttn's).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import ClassVar, Optional, Tuple

import numpy as np

from . import rowops_spec as rs
from . import vectorop_smx as smx
from .host_nodes import HostContext, _attrs, _c_double, _label, _resolve
from .llm_nodes import (HostTable, LlmNode, _const_array, _i32, _llm_inputs, _res32, _silu, _st, exp_tag,
                        silu_item, table_kind)
from .nodes import SchedulerError
from .rowops_nodes import (QIMAGE_R, AttnPrepRoNode, NO_WINDOW, _exps, _need_unit, _pblock, rope_raw_tensor,
                           tables_tensor, zero_param_tensor)


def _require(cond, node, msg):
    if not cond:
        raise SchedulerError(f"{node.op_type} node '{_label(node)}': {msg}")


class VlaNode(LlmNode):
    helpers: ClassVar[Tuple[str, ...]] = ("llm", "vla")


# ------------------------------------------------------------------ #
# VlaPrefix                                                            #
# ------------------------------------------------------------------ #

@dataclass
class VlaPrefixNode(VlaNode):
    """The prefix rows: image features, language embeddings, the state row."""
    cams:  int = 1
    img:   int = 1                                   # image rows per camera
    d:     int = 1
    L:     int = 1                                   # language ids (the most)
    S:     int = 1                                   # state values
    T:     int = 1
    rows:  int = 1                                   # the embedding table's rows
    scale: float = 1.0
    fr:    int = 15
    table: Optional[HostTable] = field(default=None, repr=False)
    ws:    Optional[HostTable] = field(default=None, repr=False)
    bs:    Optional[HostTable] = field(default=None, repr=False)

    @classmethod
    def from_onnx_node(cls, node, tensors, index, align_elems, ctx: HostContext):
        ins = _llm_inputs(node, tensors)
        _require(len(ins) >= 7, node, "inputs: ids, n, state, table, W_state, b_state, image rows (>= 1)")
        ids, n, st = ins[:3]
        imgs = ins[6:]
        y = _resolve(tensors, node.output[0], node)
        tab = _const_array(ctx, tensors, node.input[3], node, "table")
        ws = _const_array(ctx, tensors, node.input[4], node, "W_state").astype(np.float32)
        bs = _const_array(ctx, tensors, node.input[5], node, "b_state").astype(np.float32).reshape(-1)
        _require(tab.ndim == 2 and ws.ndim == 2, node, "table [rows][d], W_state [S][d]")
        rows, d = tab.shape
        S = int(ws.shape[0])
        _require(int(ws.shape[1]) == d and bs.size == d, node, f"W_state [S][{d}], b_state [{d}]")
        img = imgs[0].numel // d
        T = y.numel // d
        a = _attrs(node)
        sc = a.get("scale")
        scale = float(sc.decode() if isinstance(sc, bytes) else sc)
        tag = f"{zlib_crc(ws)}_{S}x{d}"
        sn = cls(onnx_node=node, inputs=[ids, n, st] + list(imgs), output=y, index=index, align_elems=align_elems,
                 F=ctx.frac_bits, cams=len(imgs), img=img, d=int(d), L=ids.numel, S=S, T=T, rows=int(rows),
                 scale=scale, table=HostTable("_llm_tab_" + tensors[node.input[3]].c_name, table_kind(tab),
                                              np.asarray(tab, np.float32)),
                 ws=HostTable(f"_vla_ws_{tag}", "f32", ws.reshape(-1)), bs=HostTable(f"_vla_bs_{tag}", "f32", bs))
        sn._want(ids, "i32", "ids")
        sn._want(n, "i32", "n")
        sn._want(st, "f32", "state")
        _require(st.numel == S, node, f"state [{S}]")
        for t in imgs:
            sn._want(t, "f32", "image rows")
            _require(t.is_state and t.numel == img * d, node, f"image rows: a float32 host state [{img}][{d}]")
        _require(y.dma == "i32", node, "output: the int32 DMA residual")
        fr = y.exp_channels(ctx.frac_bits)
        _require((fr == fr[0]).all(), node, "the residual at one exponent")
        sn.fr = int(fr[0])
        _require(T * d == y.numel and T >= len(imgs) * img + 1, node, "rows")
        return sn

    def tables(self):
        return [self.table, self.ws, self.bs]

    def describe(self):
        return (f"{self.cams} camera(s) x {self.img} image rows + <= {self.L} language rows (x {self.scale:.6g}) "
                f"+ the state row -> [{self.T}][{self.d}] int32 at 2^-{self.fr}")

    def c_call(self, ins, out, scratch, direct, dtype):
        bf = self.table.kind == "bf16"
        t16, t32 = (self.table.name, "NULL") if bf else ("NULL", self.table.name)
        imgs = ", ".join(ins[3:])
        return ["{",
                f"    const float *_img[{self.cams}] = {{ {imgs} }};",
                f"    vla_prefix({ins[0]}, {ins[1]}[0], {ins[2]}, _img, {self.cams}u, {self.img}u, {self.d}u,"
                f" {self.L}u, {self.S}u, {t16}, {t32}, {self.rows}u, {_c_double(self.scale)},",
                f"               {self.ws.name}, {self.bs.name}, {self.T}u, {self.fr}, {out});",
                "}"]

    def reference(self, ins, dtype):  # noqa: ARG002
        C, R, d, T = self.cams, self.img, self.d, self.T
        ids = np.asarray(ins[0], np.float64).reshape(-1).astype(np.int64)
        n = _i32(ins[1])
        st = np.asarray(ins[2], np.float32).astype(np.float64).reshape(-1)
        out = np.zeros((T, d))
        for c in range(C):
            out[c * R:(c + 1) * R] = np.asarray(ins[3 + c], np.float32).astype(np.float64).reshape(R, d)
        nl = min(max(n - C * R - 1, 0), self.L)
        tab = np.asarray(self.table.data, np.float32).reshape(self.rows, d)
        k = np.clip(ids[:nl], 0, self.rows - 1)
        out[C * R:C * R + nl] = tab[k].astype(np.float64) * self.scale
        r = C * R + nl
        if n >= C * R + 1 and r < T:
            ws = np.asarray(self.ws.data, np.float32).astype(np.float64).reshape(self.S, d)
            bs = np.asarray(self.bs.data, np.float32).astype(np.float64)
            out[r] = np.cumsum(st[:, None] * ws, axis=0)[-1] + bs
        return _res32(out, self.fr).reshape(self.output.shape)


def zlib_crc(a: np.ndarray) -> str:
    import zlib
    return f"{zlib.crc32(np.ascontiguousarray(a).tobytes()):08x}"


# ------------------------------------------------------------------ #
# The suffix: VlaActIn, VlaTimeSilu, VlaVelocity                       #
# ------------------------------------------------------------------ #

@dataclass
class VlaActInNode(VlaNode):
    """x [R][A] float32 -> int16 [T][A] at the output's exponents, rows >= R zero."""
    R: int = 1
    T: int = 1
    A: int = 1

    @classmethod
    def from_onnx_node(cls, node, tensors, index, align_elems, ctx: HostContext):
        (x,) = _llm_inputs(node, tensors)
        y = _resolve(tensors, node.output[0], node)
        A = int(y.shape[-1])
        sn = cls(onnx_node=node, inputs=[x], output=y, index=index, align_elems=align_elems, F=ctx.frac_bits,
                 R=x.numel // A, T=y.numel // A, A=A)
        sn._want(x, "f32", "x")
        sn._want(y, None, "output")
        _require(int(x.shape[-1]) == A and sn.R <= sn.T, node, f"x [R][{A}] -> [T >= R][{A}]")
        return sn

    def c_runtime(self):
        return [self._scales(self.output)[2]]

    def describe(self):
        return f"{self.R} x {self.A} float32 -> [{self.T}][{self.A}] int16"

    def c_call(self, ins, out, scratch, direct, dtype):
        ia = self._scales(self.output)[1]
        return [f"vla_act_in({ins[0]}, {self.R}u, {self.T}u, {self.A}u, {ia}, {out});"]

    def reference(self, ins, dtype):
        x = np.asarray(ins[0], np.float32).astype(np.float64).reshape(self.R, self.A)
        out = np.zeros((self.T, self.A))
        out[:self.R] = _st(dtype, x, self.output.exp_channels(self.F)[None, :])
        return out.reshape(self.output.shape)


@dataclass
class VlaTimeSiluNode(VlaNode):
    """silu(a0 + b[step]) with the bias an integer at a0's exponent."""
    T:     int = 1
    N:     int = 1
    steps: int = 1
    bq:    Optional[HostTable] = field(default=None, repr=False)

    @classmethod
    def from_onnx_node(cls, node, tensors, index, align_elems, ctx: HostContext):
        a0, step = _llm_inputs(node, tensors)[:2]
        y = _resolve(tensors, node.output[0], node)
        bq = _const_array(ctx, tensors, node.input[2], node, "bq")
        N = int(a0.shape[-1])
        _require(bq.ndim == 2 and int(bq.shape[1]) == N, node, f"bq [steps][{N}]")
        _require(np.array_equal(bq, np.rint(bq)) and np.abs(bq).max() < 2 ** 31, node, "bq: integers")
        sn = cls(onnx_node=node, inputs=[a0, step], output=y, index=index, align_elems=align_elems,
                 F=ctx.frac_bits, T=a0.numel // N, N=N, steps=int(bq.shape[0]),
                 bq=HostTable(f"_vla_bq_{zlib_crc(np.asarray(bq, np.int32))}_{bq.shape[0]}x{N}", "i32",
                              np.asarray(bq, np.int64).reshape(-1)))
        sn._want(a0, None, "a0")
        sn._want(step, "i32", "step")
        sn._want(y, None, "output")
        _require(y.numel == a0.numel, node, "shapes")
        return sn

    def tables(self):
        return [self.bq]

    def c_runtime(self):
        fa = self.inputs[0].exp_channels(self.F)
        items = [silu_item(int(f)) for f in sorted(set(int(v) for v in fa))]
        items.append(self._scales(self.inputs[0])[2])
        items.append(self._scales(self.output)[2])
        return items + [tb.runtime_item() for tb in self.tables()]

    def describe(self):
        return f"rows={self.T} n={self.N}: SiLU(a0 + the bias of the step; {self.steps} steps)"

    def c_call(self, ins, out, scratch, direct, dtype):
        ea = "_llm_e_" + exp_tag(self.inputs[0].exp_channels(self.F))
        ia = self._scales(self.output)[1]
        return [f"vla_time_silu({ins[0]}, {ins[1]}[0], {self.bq.name}, {self.steps}u, {ea}, {ia}, "
                f"{self.T}u, {self.N}u, {out});"]

    def reference(self, ins, dtype):
        fa = self.inputs[0].exp_channels(self.F)
        a0 = np.asarray(ins[0], np.float64).reshape(self.T, self.N)
        s = min(max(_i32(ins[1]), 0), self.steps - 1)
        bq = np.asarray(self.bq.data, np.int64).reshape(self.steps, self.N)[s]
        raw = np.clip(np.rint(a0 * np.power(2.0, fa)[None, :]) + bq[None, :], -32768, 32767).astype(np.int64)
        y = np.empty_like(a0)
        for f in np.unique(fa):
            cols = fa == f
            y[:, cols] = _silu(int(f))[raw[:, cols] + 32768]
        return _st(dtype, y, self.output.exp_channels(self.F)[None, :]).reshape(self.output.shape)


@dataclass
class VlaVelocityNode(VlaNode):
    """v [R][A] (f32 host) = float32(v0 + b), the first R rows."""
    R: int = 1
    A: int = 1
    b: Optional[np.ndarray] = field(default=None, repr=False)

    @classmethod
    def from_onnx_node(cls, node, tensors, index, align_elems, ctx: HostContext):
        v0 = _llm_inputs(node, tensors)[0]
        y = _resolve(tensors, node.output[0], node)
        A = int(v0.shape[-1])
        b = _const_array(ctx, tensors, node.input[1], node, "bias").astype(np.float32).reshape(-1)
        _require(b.size == A, node, f"bias [{A}]")
        sn = cls(onnx_node=node, inputs=[v0], output=y, index=index, align_elems=align_elems, F=ctx.frac_bits,
                 R=y.numel // A, A=A, b=b.copy())
        sn._want(v0, None, "v0")
        sn._want(y, "f32", "output")
        _require(sn.R * A <= v0.numel and int(y.shape[-1]) == A, node, "shapes")
        return sn

    @property
    def b_name(self) -> str:
        return f"_vla_vb_{zlib_crc(self.b)}_{self.A}"

    def c_runtime(self):
        return [self._scales(self.inputs[0])[2], HostTable(self.b_name, "f32", self.b).runtime_item()]

    def describe(self):
        return f"{self.R} x {self.A} int16 + bias -> float32"

    def c_call(self, ins, out, scratch, direct, dtype):
        sx = self._scales(self.inputs[0])[0]
        return [f"vla_velocity({ins[0]}, {sx}, {self.b_name}, {self.R}u, {self.A}u, {out});"]

    def reference(self, ins, dtype):  # noqa: ARG002
        v0 = np.asarray(ins[0], np.float64).reshape(-1, self.A)[:self.R]
        return (v0 + self.b.astype(np.float64)[None, :]).astype(np.float32).astype(np.float64).reshape(
            self.output.shape)


# ------------------------------------------------------------------ #
# The prefix's state row: VlaStateAttn, VlaAttnMerge                   #
# ------------------------------------------------------------------ #

def state_row_attention(s, V, n, T, G, cm, cs, fp):
    """Per head h' < G of one KV group: P over the keys < n of score column h' T + n - 1
    (the softmax unit's integer specification), then floor(sum_j P_j V[j][d] / 2^8)
    saturated (ConvKernel's P.V).  s [C][G T] raw, V [n][HD] raw; -> [G][HD] raw."""
    HD = V.shape[1]
    out = np.zeros((G, HD))
    if n < 1:
        return out
    for h in range(G):
        x = np.asarray(s[:n, h * T + n - 1], np.int64)[None, :]
        P = smx.softmax_raw(x, n, int(cm[h]), int(cs[h]), int(fp[h]))[0]
        acc = P @ np.asarray(V[:n], np.int64)
        out[h] = np.clip(np.floor(acc / 256.0), -32768, 32767)
    return out


@dataclass
class VlaStateAttnNode(VlaNode):
    """The state row's attention of KV group ``group`` (the prefix-LM mask's one
    row that attends its own key): -> [G][HD] raw (int16 host)."""
    T:     int = 1
    G:     int = 1
    HD:    int = 16
    C:     int = 16
    VK:    int = 1
    group: int = 0
    cm:    Optional[np.ndarray] = field(default=None, repr=False)     # [G]
    cs:    Optional[np.ndarray] = field(default=None, repr=False)
    fp:    Optional[np.ndarray] = field(default=None, repr=False)

    @classmethod
    def from_onnx_node(cls, node, tensors, index, align_elems, ctx: HostContext):
        s, cv, n = _llm_inputs(node, tensors)
        y = _resolve(tensors, node.output[0], node)
        a = _attrs(node)
        H, KV, HD = int(a["num_heads"]), int(a["num_kv_heads"]), int(a["head_dim"])
        G = H // KV
        C = int(s.shape[0])
        T = s.numel // (C * G)
        ls_ = a.get("logit_scale")
        scale = float(ls_.decode() if isinstance(ls_, bytes) else ls_) if ls_ is not None else 1.0 / math.sqrt(HD)
        fs = [int(v) for v in a["s_exp"]]
        fp = [int(v) for v in a["p_exp"]]
        _require(len(fs) == G and len(fp) == G, node, f"s_exp / p_exp: {G} values")
        regs = [smx.scale_regs(f, scale) for f in fs]
        sn = cls(onnx_node=node, inputs=[s, cv, n], output=y, index=index, align_elems=align_elems,
                 F=ctx.frac_bits, T=T, G=G, HD=HD, C=C, VK=int(cv.group_kw), group=int(a["group"]),
                 cm=np.array([r[0] for r in regs], np.int64), cs=np.array([r[1] for r in regs], np.int64),
                 fp=np.array(fp, np.int64))
        sn._want(s, None, "scores")
        sn._want(n, "i32", "n")
        sn._want(y, "i16", "output")
        _require(list(s.shape) == [C, G * T], node, f"scores [{C}][{G * T}]")
        _require(cv.is_state and not cv.is_host and cv.group_layout == (KV, HD) and int(cv.shape[0]) == C, node,
                 f"the V cache: a DMA state, group-major [{KV}][{C}][{HD}]")
        _require(y.numel == G * HD, node, f"output [{G}][{HD}]")
        return sn

    def describe(self):
        return f"group {self.group}: the state row of {self.G} heads over the keys < n (the unit's softmax)"

    def c_call(self, ins, out, scratch, direct, dtype):
        cv = self.inputs[1]
        cms = ", ".join(f"{int(v)}u" for v in self.cm)
        css = ", ".join(str(int(v)) for v in self.cs)
        fps = ", ".join(str(int(v)) for v in self.fp)
        return ["{",
                f"    static const uint32_t _cm[{self.G}] = {{ {cms} }};",
                f"    static const int _cs[{self.G}] = {{ {css} }}, _fp[{self.G}] = {{ {fps} }};",
                f"    const unsigned _n = vla_clamp_n({ins[2]}[0], {self.T}u, {self.C}u);",
                f"    llm_cache_inval_rows({cv.c_name}, 0u, _n, {self.inputs[1].group_layout[0]}u, {self.C}u,"
                f" {self.HD}u, {self.VK}u);",
                f"    vla_state_attn({ins[0]}, {ins[1]}, _n, {self.T}u, {self.G}u, {self.C}u, {self.HD}u, "
                f"{self.VK}u, {self.group}u, _cm, _cs, _fp, {out});",
                "}"]

    def reference(self, ins, dtype):  # noqa: ARG002
        T, G, HD, C = self.T, self.G, self.HD, self.C
        s = np.asarray(ins[0], np.float64).reshape(C, G * T)
        n = min(max(_i32(ins[2]), 0), T, C)
        fv = self.inputs[1].exp_channels(self.F)
        cv = np.asarray(ins[1], np.float64).reshape(C, -1)
        KVHD = cv.shape[1]
        V = np.rint(cv[:, self.group * HD:(self.group + 1) * HD]
                    * np.power(2.0, fv[self.group * HD:(self.group + 1) * HD])[None, :])
        del KVHD
        return state_row_attention(s, V, n, T, G, self.cm, self.cs, self.fp).reshape(self.output.shape)


@dataclass
class VlaAttnMergeNode(VlaNode):
    """LlmAttnMerge of the P.V outputs o_g [G T][HD], row n - 1 from the
    VlaStateAttn rows st_g [G][HD] -> pv [T][H HD] (raw at pv's exponents)."""
    T:  int = 1
    H:  int = 1
    KV: int = 1
    HD: int = 16

    @classmethod
    def from_onnx_node(cls, node, tensors, index, align_elems, ctx: HostContext):
        ins = _llm_inputs(node, tensors)
        y = _resolve(tensors, node.output[0], node)
        a = _attrs(node)
        H, KV, HD = int(a["num_heads"]), int(a["num_kv_heads"]), int(a["head_dim"])
        G = H // KV
        _require(len(ins) == 2 * KV + 1, node, f"inputs: {KV} P.V outputs, {KV} state rows, n")
        T = y.numel // (H * HD)
        sn = cls(onnx_node=node, inputs=list(ins), output=y, index=index, align_elems=align_elems,
                 F=ctx.frac_bits, T=T, H=H, KV=KV, HD=HD)
        for t in ins[:KV]:
            sn._want(t, None, "P.V output")
            _require(list(t.shape) == [G * T, HD], node, f"P.V outputs [{G * T}][{HD}]")
        for t in ins[KV:2 * KV]:
            sn._want(t, "i16", "state rows")
            _require(t.numel == G * HD, node, f"state rows [{G}][{HD}]")
        sn._want(ins[-1], "i32", "n")
        sn._want(y, None, "output")
        return sn

    def describe(self):
        return (f"{self.KV} groups x {self.H // self.KV} heads x {self.T} rows -> [{self.T}][{self.H * self.HD}], "
                f"row n - 1 the state row's")

    def c_call(self, ins, out, scratch, direct, dtype):
        KV = self.KV
        return ["{",
                f"    const Data_t *_o[{KV}] = {{ {', '.join(ins[:KV])} }};",
                f"    const int16_t *_st[{KV}] = {{ {', '.join(ins[KV:2 * KV])} }};",
                f"    llm_attn_merge(_o, {KV}u, {self.H // KV}u, {self.T}u, {self.HD}u, {out});",
                f"    vla_merge_state(_st, {ins[-1]}[0], {KV}u, {self.H // KV}u, {self.T}u, {self.HD}u, {out});",
                "}"]

    def reference(self, ins, dtype):  # noqa: ARG002
        T, H, KV, HD = self.T, self.H, self.KV, self.HD
        G = H // KV
        raw = np.zeros((T, H, HD))
        for g in range(KV):
            o = np.asarray(ins[g], np.float64).reshape(G, T, HD)
            for hh in range(G):
                raw[:, g * G + hh] = o[hh]
        n = _i32(ins[-1])
        if 1 <= n <= T:
            for g in range(KV):
                st = np.asarray(ins[KV + g], np.float64).reshape(G, HD)
                raw[n - 1, g * G:(g + 1) * G] = st
        f = self.output.exp_channels(self.F).astype(np.float64)
        return raw.reshape(T, H * HD) / np.power(2.0, f)[None, :]


# ------------------------------------------------------------------ #
# VlaAttnPrep: the attention prep's q part or k part on the unit       #
# ------------------------------------------------------------------ #

@dataclass
class VlaPrepRoNode(AttnPrepRoNode):
    """parts "q": [zero the q staging rows (first)], RoPE(q0) -> qs (rows < n), the
    q.K^T image of every group; parts "k": RoPE(k0) of all T rows -> the output at
    the K cache's exponents (per head); parts "v": v0 of all T rows re-rounded at the
    output's exponents (COPY: shifted left / right with round half even, saturated)."""
    parts: str = "q"

    def state_writes(self):
        return [self.qs] if self.parts == "q" else []

    def state_updates(self):
        return self.state_writes()

    def _jobs(self, layouts, pos: int, n):
        from .rowops_nodes import row_stride_bytes
        T, HD, KV, G = self.T, self.HD, self.KV, self.G
        nd, al = self.onnx_node, self.align_elems
        NW = NO_WINDOW
        bo = f"{pos} * {8 * HD}u" if isinstance(pos, str) else pos * 8 * HD
        jobs = []
        if self.parts == "v":
            sv = row_stride_bytes(self.v, layouts, self.nk, 2, nd, al)
            so = row_stride_bytes(self.output, layouts, self.nk, 2, nd, al)
            jobs.append(("v0 re-rounded at the output's exponents (all T rows)", self.v, 0, None, 0, self.output, 0,
                         self.pvb, rs.opword(rs.OP_COPY, p_bcast=self.pv_bc), self.nk, T, 1, sv, 0, 0, 0, so, 0,
                         0, 0, NW, False))
            return jobs, None, None
        if self.parts == "k":
            sk = row_stride_bytes(self.k0, layouts, self.nk, 2, nd, al)
            so = row_stride_bytes(self.output, layouts, self.nk, 2, nd, al)
            jobs.append(("RoPE(k0) at f_k[g] -> the output (all T rows)", self.k0, 0, self.raw, bo, self.output,
                         0, self.pk, self.op_rope | (rs.F_P_BCAST if self.pk_bc else 0), self.nk, T, 1, sk, 0,
                         8 * HD, 0, so, 0, HD, 0, NW, False))
            return jobs, None, None
        sq = row_stride_bytes(self.q0, layouts, self.nq, 2, nd, al)
        sqs = self.nq * 2
        if self.first:
            jobs.append(("the q staging zeroed: rows past n stay 0", self.qs, 0, None, 0, self.qs, 0,
                         self.pz, self.op_zero, self.nq, T, 1, sqs, 0, 0, 0, sqs, 0, 0, 0, 0, False))
        jobs.append(("RoPE(q0) at f_q[h] -> qs (rows < n; B = cos / sin of pos + t)", self.q0, 0,
                     self.raw, bo, self.qs, 0, self.pq, self.op_rope | (rs.F_P_BCAST if self.pq_bc else 0),
                     self.nq, n, 1, sq, 0, 8 * HD, 0, sqs, 0, HD, 0, NW, False))
        for g in range(KV):
            jobs.append((f"the q.K^T x image of group {g}: qs -> qx (8-row tiles)", self.qs, g * G * HD * 2,
                         None, 0, self.output, g * HD * G * T * 2, None, self.op_xpose, HD, T // QIMAGE_R, G,
                         QIMAGE_R * sqs, HD * 2, sqs, 2 * G * T, 2 * QIMAGE_R, 2 * T, QIMAGE_R, 0, NW, False))
        return jobs, None, None

    def chain_jobs(self, layouts: dict) -> int:
        jobs, _, _ = self._jobs(layouts, 0, self.T)
        return len(jobs)

    def hidden_calls(self, layouts: dict) -> list:  # noqa: ARG002
        return []

    def describe(self) -> str:
        what = {"q": "RoPE(q) + the q image (no cache)", "k": "RoPE(k) of all rows (no cache)",
                "v": "v re-rounded (all rows)"}[self.parts]
        return f"T={self.T} heads {self.H}/{self.KV} head_dim {self.HD} on RowOpsKernel: {what}"

    def emit_comment(self) -> str:
        src = {"q": self.q0, "k": self.k0, "v": self.v}[self.parts]
        return (f"    /* [{self.index}] VlaAttnPrep({src.onnx_name}) -> {self.output.onnx_name}  "
                f"{self.describe()} */")

    def emit_call(self, layouts: dict) -> str:
        jobs, _, _ = self._jobs(layouts, "_pos", "_n")
        cn = lambda t: "NULL" if t is None else t.c_name   # noqa: E731
        lines = ["    {",
                 f"        const unsigned _pos = (unsigned){self.pos.c_name}[0];",
                 f"        const unsigned _n = llm_ro_clamp_n(_pos, (unsigned){self.n_t.c_name}[0], {self.T}u, "
                 f"{self.C}u);",
                 "        (void)_n;"]
        for i, j in enumerate(jobs):
            if i:
                lines.append("        kernel_wait(KERNEL_ROWOPS);")
            lines.append(f"        /* {j[0]} */")
            lines.append(self.call(cn(j[1]), j[2], cn(j[3]), j[4], cn(j[5]), j[6], cn(j[7]), j[8], j[9], j[10],
                                   j[11], j[12], j[13], j[14], j[15], j[16], j[17], 0, j[18], 0, j[19], j[20],
                                   indent="        "))
        lines.append("    }")
        return "\n".join(lines)

    def reference(self, ins, dtype):  # noqa: ARG002
        from .llm_nodes import attn_keys
        a = {t.onnx_name: x for t, x in zip(self.inputs, ins, strict=True)}
        T, H, KV, HD, F = self.T, self.H, self.KV, self.HD, self.F
        G = H // KV
        pos = _i32(a[self.pos.onnx_name])
        cr, sr = self._raw_tables()
        p2 = lambda e: np.power(2.0, np.asarray(e, np.float64))   # noqa: E731
        if self.parts == "v":
            fv0, fo = _exps(self.v, F), _exps(self.output, F)
            vr = np.rint(np.asarray(a[self.v.onnx_name], np.float64).reshape(T, KV * HD) * p2(fv0)[None, :])
            sa, sc = np.maximum(fo - fv0, 0), np.maximum(fv0 - fo, 0)
            y = rs.sat16(rs.rne_shr_vec(vr.astype(np.int64) << sa[None, :], sc[None, :]))
            return (y / p2(fo)[None, :]).reshape(self.output.shape)
        if self.parts == "k":
            fk0 = _exps(self.k0, F)
            fo = _exps(self.output, F)
            xk = np.rint(np.asarray(a[self.k0.onnx_name], np.float64).reshape(T, KV * HD)
                         * p2(fk0)[None, :]).astype(np.int64).reshape(T, KV, HD)
            rows = np.minimum(pos + np.arange(T), self.C - 1)
            yk = rs.rope(xk, fk0.reshape(KV, HD), cr[rows], sr[rows], fo.reshape(KV, HD)[:, 0])
            return (yk.reshape(T, -1) / p2(fo)[None, :]).reshape(self.output.shape)
        n, _keys = attn_keys(pos, _i32(a[self.n_t.onnx_name]), T, self.C, self.Q)
        qx = np.zeros((KV, HD, G * T))
        if n > 0:
            fq0 = _exps(self.q0, F)
            q0 = np.asarray(a[self.q0.onnx_name], np.float64).reshape(T, H * HD)[:n]
            xq = np.rint(q0 * p2(fq0)[None, :]).astype(np.int64).reshape(n, H, HD)
            yq = rs.rope(xq, fq0.reshape(H, HD), cr[pos:pos + n], sr[pos:pos + n], self.fq)
            for h in range(H):
                g, hh = divmod(h, G)
                qx[g, :, hh * T: hh * T + n] = yq[:, h, :].T
        return qx


def from_vla_attn_prep(node, tensors, index, align_elems, ctx: HostContext):
    """VlaAttnPrep(x0, pos, n, cos, sin[, qs]) -> qx (parts "q") or the rotated k
    (parts "k"); the attributes of LlmAttnPrep (num_heads, num_kv_heads, head_dim,
    key_quantum, q_exp / k_exp, first)."""
    from .llm_nodes import KEY_QUANTUM, _attr_ints, _heads_attrs
    _need_unit(node, "VlaAttnPrep")
    F = ctx.frac_bits
    ins = _llm_inputs(node, tensors)
    a, H, KV, HD = _heads_attrs(node)
    parts = a.get("parts", b"q")
    parts = parts.decode() if isinstance(parts, bytes) else str(parts)
    _require(parts in ("q", "k", "v"), node, "parts: 'q', 'k' or 'v'")
    x0, pos, n = ins[:3]
    y = _resolve(tensors, node.output[0], node)
    G = H // KV
    nh = H if parts == "q" else KV
    T = x0.numel // (nh * HD)
    Q = int(a.get("key_quantum", KEY_QUANTUM))
    cos = _const_array(ctx, tensors, node.input[3], node, "cos").astype(np.float32)
    sin = _const_array(ctx, tensors, node.input[4], node, "sin").astype(np.float32)
    C = int(cos.shape[0])
    _require(cos.shape == (C, HD // 2) and sin.shape == cos.shape and C % Q == 0, node, "cos / sin [C][HD / 2]")
    _require(HD % 16 == 0 and 16 <= HD <= 128 and T % QIMAGE_R == 0, node, "head_dim / rows")
    _require(x0.dma is None and x0.host is None and not x0.is_state and pos.host == "i32" and n.host == "i32",
             node, "x0: an int16 DMA tensor; pos / n: int32 host tensors")

    def rope_params(t, nh_, f_out):
        fx = _exps(t, F).reshape(nh_, HD)
        half = HD // 2
        frot = np.concatenate([fx[:, half:], fx[:, :half]], axis=1)
        fp = np.maximum(fx, frot)
        fo = np.broadcast_to(np.asarray(f_out, np.int64).reshape(nh_, -1), (nh_, HD))
        sa, sb, sc = fp - fx, fp - frot, fp + rs.ROPE_BITS - fo
        _require((sa >= 0).all() and (sa <= 63).all() and (sb <= 63).all() and (sc >= 0).all()
                 and (sc <= 63).all(), node, "RoPE exponents beyond the shifts")
        return sa.reshape(-1), sb.reshape(-1), sc.reshape(-1)

    raw = rope_raw_tensor(tensors, cos, sin)
    pz = zero_param_tensor(tensors)
    tb = tables_tensor(tensors)
    common = dict(onnx_node=node, output=y, index=index, align_elems=align_elems, F=F, T=T, H=H, KV=KV, HD=HD,
                  C=C, Q=Q, K=1, first=int(a.get("first", 0)) != 0, has_n=True, v_direct=True,
                  cos=cos, sin=sin, pos=pos, n_t=n, raw=raw, pz=pz, parts=parts)
    if parts == "v":
        _require(y.dma is None and y.host is None and not y.is_state and list(y.shape) == [T, KV * HD], node,
                 f"the output: an int16 DMA tensor [{T}][{KV * HD}]")
        fv0, fo = _exps(x0, F), _exps(y, F)
        sa, sc = np.maximum(fo - fv0, 0), np.maximum(fv0 - fo, 0)
        _require((sa <= 63).all() and (sc <= 63).all(), node, "v / output exponents beyond the shifts")
        pvb, pv_bc = _pblock(tensors, f"{y.onnx_name}__rowops_pv", 0, 0, sa, 0, sc, KV * HD)
        return VlaPrepRoNode(inputs=[x0, pos, n, pvb, pz, tb], v=x0, pvb=pvb, pv_bc=pv_bc,
                             fq=np.zeros(H, np.int64), **common)
    if parts == "q":
        qs = ins[5]
        _require(qs is not None and qs.is_state and not qs.is_host and qs.dma is None
                 and list(qs.shape) == [T, H * HD], node, f"qs: an int16 DMA state [{T}][{H * HD}]")
        _require(list(y.shape) == [KV, HD, G * T], node, f"qx must be [{KV}][{HD}][{G * T}]")
        fq = np.asarray(_attr_ints(a, "q_exp", node, H), np.int64)
        sa, sb, sc = rope_params(x0, H, fq)
        pq, pq_bc = _pblock(tensors, f"{y.onnx_name}__rowops_pq", 0, 0, sa, sb, sc, H * HD)
        return VlaPrepRoNode(inputs=[x0, pos, n, qs, raw, pq, pz, tb], q0=x0, qs=qs, pq=pq, pq_bc=pq_bc, fq=fq,
                             **common)
    _require(y.dma is None and y.host is None and not y.is_state and list(y.shape) == [T, KV * HD], node,
             f"the output: an int16 DMA tensor [{T}][{KV * HD}]")
    fo = _exps(y, F).reshape(KV, HD)
    _require((fo == fo[:, :1]).all(), node, "the output's exponent must be one per KV head")
    sa, sb, sc = rope_params(x0, KV, fo[:, 0])
    pk, pk_bc = _pblock(tensors, f"{y.onnx_name}__rowops_pk", 0, 0, sa, sb, sc, KV * HD)
    return VlaPrepRoNode(inputs=[x0, pos, n, raw, pk, pz, tb], k0=x0, pk=pk, pk_bc=pk_bc,
                         fq=np.zeros(H, np.int64), **common)


VLA_OP_FACTORIES = {
    "VlaPrefix":    VlaPrefixNode.from_onnx_node,
    "VlaActIn":     VlaActInNode.from_onnx_node,
    "VlaTimeSilu":  VlaTimeSiluNode.from_onnx_node,
    "VlaVelocity":  VlaVelocityNode.from_onnx_node,
    "VlaStateAttn": VlaStateAttnNode.from_onnx_node,
    "VlaAttnMerge": VlaAttnMergeNode.from_onnx_node,
    "VlaAttnPrep":  from_vla_attn_prep,
}


# ------------------------------------------------------------------ #
# C helpers                                                            #
# ------------------------------------------------------------------ #

def _tab_literal() -> str:
    t = [int(v) for v in smx.TAB]
    rows = ",\n".join("    " + ", ".join(f"{v}u" for v in t[i:i + 12]) for i in range(0, len(t), 12))
    return f"static const uint32_t vla_smx_tab[{len(t)}] = {{\n{rows}\n}};\n"


VLA_C_HEAD = r"""
/* ---- SmolVLA host ops (src/vla_nodes.py; doc/plans/VLA_PLAN.md) ---- */

/* The prefix rows (VlaPrefix): image features, language embeddings x scale, the
 * state row (sum_i state[i] ws[i][c] left to right + bs[c]); int32 at 2^-fr. */
static void vla_prefix(const int32_t *ids, int32_t n, const float *state, const float *const *img,
                       unsigned C, unsigned I, unsigned D, unsigned L, unsigned S, const uint16_t *t16,
                       const float *t32, unsigned rows, double scale, const float *ws, const float *bs,
                       unsigned T, int fr, int32_t *h)
{
    const double sc = ldexp(1.0, fr);
    unsigned     c, r, i, d, nl;
    long         m = (long)n - (long)(C * I) - 1L;
    nl = m < 0 ? 0u : ((unsigned long)m > L ? L : (unsigned)m);
    for (c = 0u; c < C; c++)
        for (r = 0u; r < I; r++)
            for (d = 0u; d < D; d++)
                h[((size_t)c * I + r) * D + d] = llm_st32((double)img[c][(size_t)r * D + d] * sc);
    for (r = 0u; r < nl; r++) {
        long     k = (long)ids[r];
        int32_t *o = h + ((size_t)C * I + r) * D;
        if (k < 0) k = 0;
        if (k >= (long)rows) k = (long)rows - 1;
        for (d = 0u; d < D; d++) {
            float v;
            if (t16) {
                uint32_t u = (uint32_t)t16[(size_t)k * D + d] << 16;
                memcpy(&v, &u, sizeof v);
            } else {
                v = t32[(size_t)k * D + d];
            }
            o[d] = llm_st32((double)v * scale * sc);
        }
    }
    r = C * I + nl;
    if (n >= (int32_t)(C * I + 1u) && r < T) {
        int32_t *o = h + (size_t)r * D;
        for (d = 0u; d < D; d++) {
            double acc = 0.0;
            for (i = 0u; i < S; i++)
                acc += (double)state[i] * (double)ws[(size_t)i * D + d];
            o[d] = llm_st32((acc + (double)bs[d]) * sc);
        }
        r++;
    }
    for (; r < T; r++)
        memset(h + (size_t)r * D, 0, (size_t)D * sizeof(int32_t));
}

/* x [R][A] float32 -> [T][A] int16 at 2^f (ia = 2^f per channel), rows >= R zero. */
static void vla_act_in(const float *x, unsigned R, unsigned T, unsigned A, const double *ia, Data_t *y)
{
    unsigned r, c;
    for (r = 0u; r < T; r++)
        for (c = 0u; c < A; c++)
            y[(size_t)r * A + c] = r < R ? llm_st((double)x[(size_t)r * A + c], ia[c]) : (Data_t)0;
}

/* silu(a0 + bq[step]) — the bias an integer at a0's exponent, the sum saturated. */
static void vla_time_silu(const Data_t *a0, int32_t step, const int32_t *bq, unsigned steps,
                          const signed char *fa, const double *ia, unsigned T, unsigned N, Data_t *y)
{
    const int32_t *b;
    unsigned       r, c;
    if (step < 0) step = 0;
    if ((unsigned)step >= steps) step = (int32_t)steps - 1;
    b = bq + (size_t)step * N;
    for (r = 0u; r < T; r++)
        for (c = 0u; c < N; c++) {
            int32_t v = (int32_t)(int16_t)a0[(size_t)r * N + c] + b[c];
            if (v > 32767) v = 32767;
            if (v < -32768) v = -32768;
            y[(size_t)r * N + c] = llm_st(_llm_silu_tab[fa[c] - LLM_SILU_EMIN][v + 32768], ia[c]);
        }
}

/* v [R][A] = float32(v0 * sx + b). */
static void vla_velocity(const Data_t *v0, const double *sx, const float *b, unsigned R, unsigned A, float *v)
{
    unsigned r, c;
    for (r = 0u; r < R; r++)
        for (c = 0u; c < A; c++)
            v[(size_t)r * A + c] = (float)(llm_ld(v0[(size_t)r * A + c], sx[c]) + (double)b[c]);
}

static unsigned vla_clamp_n(int32_t n, unsigned T, unsigned C)
{
    if (n < 0) return 0u;
    if ((unsigned)n > T) n = (int32_t)T;
    if ((unsigned)n > C) n = (int32_t)C;
    return (unsigned)n;
}
"""

VLA_C_TAIL = r"""
/* The prefix's state row (VlaStateAttn): per head h' of KV group g, P over the
 * keys < n of score column h' T + n - 1 by the softmax unit's integer
 * specification (vectorop_smx.softmax_raw), then floor(sum_j P_j V_j / 2^8)
 * saturated (ConvKernel's P.V). */
static void vla_state_attn(const Data_t *s, const int16_t *cv, unsigned n, unsigned T, unsigned G,
                           unsigned C, unsigned HD, unsigned VK, unsigned g, const uint32_t *cm,
                           const int *cs, const int *fp, int16_t *o)
{
    static uint64_t e[LLM_MAX_KEYS];
    unsigned        h, j, d;
    if (n == 0u) {
        memset(o, 0, (size_t)G * HD * sizeof(int16_t));
        return;
    }
    for (h = 0u; h < G; h++) {
        const size_t col = (size_t)h * T + n - 1u;
        int64_t      m = INT64_MIN;
        uint64_t     S = 0u, R;
        const int    sh = 40 - fp[h];
        for (j = 0u; j < n; j++) {
            int64_t x = (int16_t)s[(size_t)j * G * T + col];
            if (x > m) m = x;
        }
        for (j = 0u; j < n; j++) {
            const int64_t  dlt = m - (int64_t)(int16_t)s[(size_t)j * G * T + col];
            const uint64_t y = ((uint64_t)dlt * cm[h]) >> cs[h];
            const uint64_t k = y >> 12;
            e[j] = (uint64_t)vla_smx_tab[y & 4095u] >> (k > 63u ? 63u : (unsigned)k);
            S += e[j];
        }
        R = S ? (1ull << 40) / S : 0u;
        for (j = 0u; j < n; j++) {
            uint64_t p = (e[j] * R + (1ull << (sh - 1))) >> sh;
            e[j] = p > 32767u ? 32767u : p;
        }
        for (d = 0u; d < HD; d++) {
            int64_t acc = 0;
            for (j = 0u; j < n; j++)
                acc += (int64_t)e[j] * (int64_t)cv[llm_vrow(C, HD, VK, g, j) + (size_t)d * VK];
            acc >>= 8;
            if (acc > 32767) acc = 32767;
            if (acc < -32768) acc = -32768;
            o[(size_t)h * HD + d] = (int16_t)acc;
        }
    }
}

/* Row n - 1 of the merged attention output from the state rows st_g [G][HD]. */
static void vla_merge_state(const int16_t *const *st, int32_t n, unsigned KV, unsigned G, unsigned T,
                            unsigned HD, Data_t *pv)
{
    const unsigned H = KV * G;
    unsigned       g;
    if (n < 1 || (unsigned)n > T) return;
    for (g = 0u; g < KV; g++)
        memcpy(pv + ((size_t)(n - 1) * H + g * G) * HD, st[g], (size_t)G * HD * sizeof(Data_t));
}
"""


def vla_c_helpers() -> str:
    return VLA_C_HEAD + _tab_literal() + VLA_C_TAIL


__all__ = ("VLA_OP_FACTORIES", "vla_c_helpers", "state_row_attention", "VlaPrefixNode", "VlaActInNode",
           "VlaTimeSiluNode", "VlaVelocityNode", "VlaStateAttnNode", "VlaAttnMergeNode", "VlaPrepRoNode",
           "from_vla_attn_prep")
