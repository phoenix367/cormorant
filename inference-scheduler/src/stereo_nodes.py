"""
Stereo depth ops (doc/plans/STEREO_PLAN.md; frontend src/stereo.py): ONNX
nodes of the domain ``axi.llm`` on tensors with power-of-two exponents
(``axi.numeric``; a tensor's value is raw * 2^-f).  The simulator holds
values; every node below computes in raw integers, exactly as its kernel
call or its C helper does.

Kernel calls (``LlmKernelNode``):

  * ``StereoVop`` — one VectorOPKernel call over n raw int16 (attribute
    ``op``): ADD (``act`` 1: ReLU after it), MUL, RELU, RELU6, LEAKY_RELU
    (``alpha``: the slope, quantised to 2^-16 as the alpha register holds
    it).  The kernel computes on the raw bits as Q8.8 values, so the
    exponents follow its arithmetic: ADD f_a = f_b = f_c; MUL f_c = f_a + f_b
    - 8 (the truncated product); RELU / LEAKY_RELU f_c = f_a; RELU6 clamps at
    raw 6.0 in Q8.8, so f_a = f_c = 8.
  * ``StereoSoftmax`` — the softmax unit's column mode over the channels of
    [1][C][H][W] logits: C keys (the first ``valid`` of them), H*W query
    columns -> P [H*W][C] raw at 2^-f_p (``p_exp``), the integer softmax of
    src/vectorop_smx.py.

Host ops (``LlmNode``; C helpers in STEREO_C):

  * ``StereoInstanceNorm`` — per channel of [1][C][H][W]: (x - mean) /
    sqrt(var + eps) over H*W (the exact integer sums, then double).
  * ``StereoPadEdge`` — replicate padding by ``pad`` on every side.
  * ``StereoCorrelation`` — the cost volume of left / right features
    [1][C][H][W]: vol[d][h][w] = (1 / C) sum_c L[c][h][w] R[c][h][w - d] for
    w >= d, else 0; d < ``disp`` (an exact int64 dot product, then double).
  * ``StereoUpsample`` — the disparity regression and LightStereo's context
    upsampling from the two softmaxes: d = sum_j Pd[.][j] * j at 1/s
    resolution (exact), then disp[y][x] = s * sum_k Pu[y,x][k] * d[y/s +
    dy_k][x/s + dx_k] over the 3 x 3 neighbourhood (zero outside; Pu row-major
    or by the four output phases of a polyphase deconv), a float32 host
    tensor [H][W].

Every C helper and its numpy ``reference`` perform the same integer sums
and the same double operations in the same order (no fused multiply-add:
the generated file turns fp-contract off), so they agree bit for bit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import ClassVar, Tuple

import numpy as np

from . import vectorop_smx as smx
from .host_nodes import HostContext, _attrs, _resolve
from .llm_nodes import LlmKernelNode, LlmNode, _require
from .nodes import (ACT_NONE, ACT_RELU, ACT_NAMES, OP_ADD, OP_LEAKY_RELU, OP_MUL, OP_NAMES, OP_RELU,
                    OP_RELU6)
from .smx_nodes import COL_BLOCK, OP_SOFTMAX_T, SmxVopNode, _unpack, enabled, mask_reg, regs

ALPHA_ONE = 1 << 16          # VectorOP's alpha register: slope = alpha / 2^16


def _exp(t, node) -> int:
    _require(t.exp is not None and np.ndim(t.exp) == 0, node,
             f"'{t.onnx_name}' needs one power-of-two exponent (axi.numeric exp)")
    return int(np.asarray(t.exp))


def _raw(v, f: int) -> np.ndarray:
    """Values on the 2^-f grid -> their raw integers (int64)."""
    r = np.asarray(v, np.float64) * 2.0 ** f
    q = np.rint(r)
    if not np.array_equal(q, r):
        raise ValueError(f"values not on the 2^-{f} grid")
    return q.astype(np.int64)


def _sat(r: np.ndarray) -> np.ndarray:
    return np.clip(r, -32768, 32767)


def _chw(t, node) -> Tuple[int, int, int]:
    _require(len(t.shape) == 4 and t.shape[0] == 1, node, f"'{t.onnx_name}' must be [1][C][H][W]")
    return int(t.shape[1]), int(t.shape[2]), int(t.shape[3])


def _int_attr(a: dict, name: str, node, default=None) -> int:
    if name not in a:
        _require(default is not None, node, f"attribute '{name}' is required")
        return int(default)
    return int(a[name])


def _double_attr(a: dict, name: str, default: float) -> float:
    """A double carried as a string attribute (ONNX float attributes are float32)."""
    v = a.get(name)
    if v is None:
        return float(default)
    return float(v.decode() if isinstance(v, bytes) else v)


def _c_double(v: float) -> str:
    return repr(float(v)) if math.isfinite(v) else "0.0"


# ------------------------------------------------------------------ #
# StereoVop: one VectorOPKernel call on exponent tensors               #
# ------------------------------------------------------------------ #

VOP_OPS = {OP_ADD: 2, OP_MUL: 2, OP_RELU: 1, OP_RELU6: 1, OP_LEAKY_RELU: 1}   # op -> arity


@dataclass
class StereoVopNode(LlmKernelNode):
    """One VectorOPKernel call over ``n`` raw int16 (see the module docstring)."""

    kernel_name: ClassVar[str] = "VectorOPKernel"

    onnx_node:   object
    inputs:      list
    output:      object
    index:       int = 0
    align_elems: int = 8
    F:           int = 8
    op:          int = OP_ADD
    act:         int = ACT_NONE
    alpha:       int = 0
    n:           int = 1
    f_out:       int = 8

    # Compatibility shims (read by the layout / header passes)
    outer_count:        int  = field(default=1,    init=False)
    chunk_size:         int  = field(default=0,    init=False)
    aligned_chunk_size: int  = field(default=0,    init=False)
    a_advances:         bool = field(default=True, init=False)
    b_advances:         bool = field(default=True, init=False)
    arity:              int  = field(default=2,    init=False)

    def __post_init__(self):
        self.arity = VOP_OPS[self.op]
        # LeakyReLU runs on the activation unit (run_op_act with the alpha register)
        self.uses_activation_unit = self.op == OP_LEAKY_RELU

    @classmethod
    def from_onnx_node(cls, node, tensors, index, align_elems, ctx: HostContext):
        a = _attrs(node)
        op = _int_attr(a, "op", node)
        _require(op in VOP_OPS, node, f"op {op}: StereoVop runs ADD, MUL, RELU, RELU6, LEAKY_RELU")
        act = _int_attr(a, "act", node, ACT_NONE)
        _require(act in (ACT_NONE, ACT_RELU) and (act == ACT_NONE or op == OP_ADD), node,
                 "act: none, or ReLU after an ADD")
        xs = [_resolve(tensors, n, node) for n in node.input]
        _require(len(xs) == VOP_OPS[op], node, f"{VOP_OPS[op]} input(s)")
        y = _resolve(tensors, node.output[0], node)
        F = ctx.frac_bits
        fs = [_exp(x, node) for x in xs]
        fy = _exp(y, node)
        _require(all(x.numel == y.numel for x in xs), node, "inputs and output of one size")
        for t in xs + [y]:
            _require(t.host is None and not t.is_state, node, f"'{t.onnx_name}' must be a DMA tensor")
        if op == OP_ADD:
            _require(fs[0] == fs[1] == fy, node, f"ADD: one exponent (got {fs} -> {fy})")
        elif op == OP_MUL:
            _require(fy == fs[0] + fs[1] - F, node, f"MUL: f_c = f_a + f_b - {F} (got {fs} -> {fy})")
        elif op == OP_RELU6:
            _require(fs[0] == fy == F, node, f"RELU6 clamps at raw 6.0 in Q8.{F}: the input and "
                                             f"output at 2^-{F} (got {fs[0]} -> {fy})")
        else:
            _require(fs[0] == fy, node, f"the activation keeps the exponent (got {fs[0]} -> {fy})")
        alpha = 0
        if op == OP_LEAKY_RELU:
            slope = _double_attr(a, "alpha", 0.01)
            alpha = int(round(slope * ALPHA_ONE))
            _require(0 <= alpha < ALPHA_ONE, node, f"alpha {slope}: 0 <= alpha < 1")
        return cls(onnx_node=node, inputs=xs, output=y, index=index, align_elems=align_elems, F=F,
                   op=op, act=act, alpha=alpha, n=y.numel, f_out=fy)

    def reference(self, ins, dtype):  # noqa: ARG002
        r = [_raw(v, int(np.asarray(t.exp))).reshape(-1) for v, t in zip(ins, self.inputs, strict=True)]
        if self.op == OP_ADD:
            y = _sat(r[0] + r[1])
            if self.act == ACT_RELU:
                y = np.maximum(y, 0)
        elif self.op == OP_MUL:
            y = _sat((r[0] * r[1]) >> self.F)                 # floor (AP_TRN)
        elif self.op == OP_RELU:
            y = np.maximum(r[0], 0)
        elif self.op == OP_RELU6:
            y = np.clip(r[0], 0, 6 << self.F)
        else:                                                 # LEAKY_RELU, rounded half to even
            neg = np.rint(r[0].astype(np.float64) * (self.alpha / ALPHA_ONE)).astype(np.int64)
            y = np.where(r[0] >= 0, r[0], neg)
        return (y.astype(np.float64) * 2.0 ** -self.f_out).reshape(self.output.shape)

    def kernel_calls(self, layouts: dict) -> list:  # noqa: ARG002
        from .perf_calls import KernelCall
        return [KernelCall.of("VectorOPKernel", op=self.op, size=self.n, outer=1, act=self.act)]

    def describe(self) -> str:
        act = f" + {ACT_NAMES[self.act]}" if self.act else ""
        return f"{self.n} elements at 2^-{self.f_out} on VectorOPKernel ({OP_NAMES[self.op]}{act})"

    def emit_comment(self) -> str:
        ins = ", ".join(t.onnx_name for t in self.inputs)
        return (f"    /* [{self.index}] StereoVop({ins}) -> {self.output.onnx_name}  {self.n} elements "
                f"at 2^-{self.f_out}: VectorOPKernel {OP_NAMES[self.op]} */")

    def emit_call(self, layouts: dict) -> str:  # noqa: ARG002
        a = self.inputs[0].c_name
        b = self.inputs[1].c_name if self.arity == 2 else "(inference_buf_t *)0"
        c = self.output.c_name
        if self.op == OP_LEAKY_RELU or self.act != ACT_NONE:
            return (f"    run_op_act({a}, {b}, {c}, {self.n}u, {OP_NAMES[self.op]}, 1u, 0u, 0u, "
                    f"{ACT_NAMES[self.act]}, {self.alpha}u);")
        return f"    run_op({a}, {b}, {c}, {self.n}u, {OP_NAMES[self.op]}, 1u, 0u, 0u);"


# ------------------------------------------------------------------ #
# StereoSoftmax: the softmax unit's column mode over channels          #
# ------------------------------------------------------------------ #

@dataclass
class StereoSoftmaxNode(SmxVopNode):
    """logits [1][G*C][H][W] raw at 2^-f_s (``groups`` G of C keys each, T =
    H*W query columns, the first ``valid`` keys of each) -> P [G*T][C] raw at
    2^-f_p: one column-mode call per group."""
    T:     int = 16
    C:     int = 8
    G:     int = 1
    valid: int = 8
    f_s:   int = 8
    cm:    int = 0
    cfg:   int = 0

    @classmethod
    def from_onnx_node(cls, node, tensors, index: int, align_elems: int, ctx: HostContext):
        x = _resolve(tensors, node.input[0], node)
        y = _resolve(tensors, node.output[0], node)
        a = _attrs(node)
        GC, H, W = _chw(x, node)
        T = H * W
        G = _int_attr(a, "groups", node, 1)
        _require(G >= 1 and GC % G == 0, node, f"{GC} channels in {G} groups")
        C = GC // G
        valid = _int_attr(a, "valid", node, C)
        fp = _int_attr(a, "p_exp", node)
        fs = _exp(x, node)
        _require(enabled(), node, "StereoSoftmax needs VectorOPKernel's softmax unit "
                                  "(kernels.vectorop.softmax, AXI_VECTOROP_SOFTMAX)")
        _require(C <= smx.MAX_KEYS and 1 <= valid <= C, node,
                 f"{C} keys, {valid} valid (the unit takes {smx.MAX_KEYS})")
        _require(T % COL_BLOCK == 0 and C % 8 == 0, node,
                 f"{T} query columns, {C} keys: the unit takes blocks of {COL_BLOCK} columns at "
                 f"16-byte aligned rows")
        _require(list(y.shape) == [G * T, C], node, f"P [{G * T}][{C}]")
        _require(_exp(y, node) == fp, node, f"P's exponent must be p_exp = {fp}")
        for t in (x, y):
            _require(t.host is None and not t.is_state, node, f"'{t.onnx_name}' must be a DMA tensor")
        cm, cfg = regs(fs, 1.0, fp)
        return cls(onnx_node=node, inputs=[x], output=y, index=index, align_elems=align_elems,
                   F=ctx.frac_bits, T=T, C=C, G=G, valid=valid, f_s=fs, cm=cm, cfg=cfg)

    def reference(self, ins, dtype):  # noqa: ARG002
        s = _raw(ins[0], self.f_s).reshape(self.G, self.C, self.T)
        cs, fp = _unpack(self.cfg)
        p = np.concatenate([smx.softmax_cols(s[g], self.valid, self.cm, cs, fp) for g in range(self.G)])
        return (p.astype(np.float64) * 2.0 ** -fp).reshape(self.output.shape)

    def kernel_calls(self, layouts: dict) -> list:  # noqa: ARG002
        from .perf_calls import KernelCall
        return [KernelCall.of("VectorOPKernel", count=self.G, op=OP_SOFTMAX_T, size=self.C, outer=self.T,
                              a_inc=self.T, b_inc=self.C, act=0)]

    def describe(self) -> str:
        cs, fp = _unpack(self.cfg)
        grp = f"{self.G} x " if self.G > 1 else ""
        return (f"{grp}{self.T} query columns x {self.C} keys ({self.valid} valid) on VectorOPKernel's "
                f"softmax unit (column mode; Cm {self.cm} >> {cs}, P at 2^-{fp})")

    def emit_comment(self) -> str:
        grp = f"{self.G} x " if self.G > 1 else ""
        return (f"    /* [{self.index}] StereoSoftmax({self.inputs[0].onnx_name}) -> "
                f"{self.output.onnx_name}  {grp}[{self.C}][{self.T}] -> P [{self.T}][{self.C}] on "
                f"VectorOPKernel's softmax unit (column mode) */")

    def emit_call(self, layouts: dict) -> str:  # noqa: ARG002
        lines = []
        for g in range(self.G):
            if g:
                lines.append("    kernel_wait(KERNEL_VECTOROP);")
            lines.append(self.call(self.inputs[0].c_name, g * self.C * self.T, self.output.c_name,
                                   g * self.T * self.C, self.C, OP_SOFTMAX_T, self.T, self.T, self.C, self.cm,
                                   self.cfg, mask_reg(self.valid)))
        return "\n".join(lines)


# ------------------------------------------------------------------ #
# Host ops                                                             #
# ------------------------------------------------------------------ #

@dataclass
class StereoNode(LlmNode):
    helpers: ClassVar[Tuple[str, ...]] = ("llm", "stereo")


@dataclass
class StereoInstanceNormNode(StereoNode):
    C:    int = 1
    n:    int = 1
    f_x:  int = 8
    f_y:  int = 8
    eps:  float = 1e-5

    @classmethod
    def from_onnx_node(cls, node, tensors, index, align_elems, ctx: HostContext):
        x = _resolve(tensors, node.input[0], node)
        y = _resolve(tensors, node.output[0], node)
        C, H, W = _chw(x, node)
        _require(list(y.shape) == list(x.shape), node, "the output has the input's shape")
        sn = cls(onnx_node=node, inputs=[x], output=y, index=index, align_elems=align_elems,
                 F=ctx.frac_bits, C=C, n=H * W, f_x=_exp(x, node), f_y=_exp(y, node),
                 eps=_double_attr(_attrs(node), "eps", 1e-5))
        sn._want(x, None, "input")
        sn._want(y, None, "output")
        return sn

    def _eps_raw(self) -> float:
        return self.eps * 2.0 ** (2 * self.f_x)                # eps in raw^2 units (exact)

    def describe(self):
        return f"instance norm of [{self.C}][{self.n}] (2^-{self.f_x} -> 2^-{self.f_y})"

    def c_call(self, ins, out, scratch, direct, dtype):
        return [f"stereo_instnorm({ins[0]}, {self.C}u, {self.n}u, {_c_double(self._eps_raw())}, "
                f"{self.f_y}, {out});"]

    def reference(self, ins, dtype):  # noqa: ARG002
        x = _raw(ins[0], self.f_x).reshape(self.C, self.n)
        out = np.empty((self.C, self.n), np.float64)
        n = self.n
        sc = 2.0 ** self.f_y
        for c in range(self.C):
            s1 = int(x[c].sum())
            s2 = int((x[c] * x[c]).sum())
            m = float(s1) / float(n)
            var = float(n * s2 - s1 * s1) / (float(n) * float(n))
            k = sc / math.sqrt(var + self._eps_raw())
            q = np.rint((x[c].astype(np.float64) - m) * k)
            out[c] = np.clip(q, -32768, 32767)
        return (out * 2.0 ** -self.f_y).reshape(self.output.shape)


@dataclass
class StereoPadEdgeNode(StereoNode):
    C:   int = 1
    H:   int = 1
    W:   int = 1
    pad: int = 1

    @classmethod
    def from_onnx_node(cls, node, tensors, index, align_elems, ctx: HostContext):
        x = _resolve(tensors, node.input[0], node)
        y = _resolve(tensors, node.output[0], node)
        C, H, W = _chw(x, node)
        p = _int_attr(_attrs(node), "pad", node, 1)
        _require(list(y.shape) == [1, C, H + 2 * p, W + 2 * p], node, "output [1][C][H+2p][W+2p]")
        _require(_exp(x, node) == _exp(y, node), node, "a copy keeps the exponent")
        sn = cls(onnx_node=node, inputs=[x], output=y, index=index, align_elems=align_elems,
                 F=ctx.frac_bits, C=C, H=H, W=W, pad=p)
        sn._want(x, None, "input")
        sn._want(y, None, "output")
        return sn

    def describe(self):
        return f"replicate pad of [{self.C}][{self.H}][{self.W}] by {self.pad}"

    def c_call(self, ins, out, scratch, direct, dtype):
        return [f"stereo_pad_edge({ins[0]}, {self.C}u, {self.H}u, {self.W}u, {self.pad}u, {out});"]

    def reference(self, ins, dtype):  # noqa: ARG002
        x = np.asarray(ins[0], np.float64).reshape(1, self.C, self.H, self.W)
        p = self.pad
        return np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode="edge").reshape(self.output.shape)


@dataclass
class StereoCorrelationNode(StereoNode):
    C:    int = 1
    H:    int = 1
    W:    int = 1
    D:    int = 1
    f_l:  int = 8
    f_r:  int = 8
    f_v:  int = 8

    @classmethod
    def from_onnx_node(cls, node, tensors, index, align_elems, ctx: HostContext):
        lt = _resolve(tensors, node.input[0], node)
        rt = _resolve(tensors, node.input[1], node)
        y = _resolve(tensors, node.output[0], node)
        C, H, W = _chw(lt, node)
        D = _int_attr(_attrs(node), "disp", node)
        _require(list(rt.shape) == list(lt.shape), node, "left and right features of one shape")
        _require(list(y.shape) == [1, D, H, W] and 1 <= D <= W, node, f"volume [1][{D}][{H}][{W}]")
        sn = cls(onnx_node=node, inputs=[lt, rt], output=y, index=index, align_elems=align_elems,
                 F=ctx.frac_bits, C=C, H=H, W=W, D=D, f_l=_exp(lt, node), f_r=_exp(rt, node),
                 f_v=_exp(y, node))
        for t, what in ((lt, "left"), (rt, "right"), (y, "output")):
            sn._want(t, None, what)
        return sn

    def scratch_bytes(self) -> int:
        return 0

    def _scale(self) -> float:
        return 2.0 ** (self.f_v - self.f_l - self.f_r)

    def describe(self):
        return (f"correlation volume of [{self.C}][{self.H}][{self.W}] over {self.D} disparities "
                f"(2^-{self.f_l} x 2^-{self.f_r} -> 2^-{self.f_v})")

    def c_call(self, ins, out, scratch, direct, dtype):
        return [f"stereo_corr({ins[0]}, {ins[1]}, {self.C}u, {self.H}u, {self.W}u, {self.D}u, "
                f"{_c_double(self._scale())}, {out});"]

    def reference(self, ins, dtype):  # noqa: ARG002
        lr = _raw(ins[0], self.f_l).reshape(self.C, self.H, self.W)
        rr = _raw(ins[1], self.f_r).reshape(self.C, self.H, self.W)
        vol = np.zeros((self.D, self.H, self.W), np.float64)
        sc = self._scale()
        for d in range(self.D):
            s = (lr[:, :, d:] * rr[:, :, :self.W - d]).sum(axis=0)       # exact int64
            q = np.rint(s.astype(np.float64) / float(self.C) * sc)
            vol[d, :, d:] = np.clip(q, -32768, 32767)
        return (vol * 2.0 ** -self.f_v).reshape(self.output.shape)


@dataclass
class StereoUpsampleNode(StereoNode):
    """The disparity regression and LightStereo's context upsampling, from
    the two softmaxes: Pd [h*w][D] (the disparity probabilities at 1/s
    resolution) and Pu (the 3 x 3 upsampling weights, K keys of which the
    first 9 count: [H*W][K], or with ``phases`` 2 the four output phases of a
    polyphase deconv, [4][(H/2)*(W/2)][K], phase 2*(y%2) + x%2) ->
    disp[y][x] = s * sum_k Pu[y,x][k] * d[y/s + dy_k][x/s + dx_k] (zero
    outside), d = sum_j Pd[.][j] * j; exact int64 sums, float32 out."""
    h:      int = 1
    w:      int = 1
    s:      int = 4
    D:      int = 48
    K:      int = 16
    phases: int = 1
    f_d:    int = 15
    f_p:    int = 15

    @classmethod
    def from_onnx_node(cls, node, tensors, index, align_elems, ctx: HostContext):
        pd = _resolve(tensors, node.input[0], node)
        pu = _resolve(tensors, node.input[1], node)
        y = _resolve(tensors, node.output[0], node)
        a = _attrs(node)
        h, w = _int_attr(a, "h", node), _int_attr(a, "w", node)
        s = _int_attr(a, "scale", node, 4)
        ph = _int_attr(a, "phases", node, 1)
        H, W = h * s, w * s
        _require(ph in (1, 2) and H % ph == 0 and W % ph == 0, node, "phases: 1, or 2 (H, W even)")
        _require(len(pd.shape) == 2 and pd.shape[0] == h * w, node, f"disparity probabilities [{h * w}][D]")
        _require(len(pu.shape) == 2 and pu.shape[0] == H * W and 9 <= pu.shape[1], node,
                 f"weights [{H * W}][K >= 9]")
        _require(list(y.shape) == [H, W], node, f"output [{H}][{W}]")
        sn = cls(onnx_node=node, inputs=[pd, pu], output=y, index=index, align_elems=align_elems,
                 F=ctx.frac_bits, h=h, w=w, s=s, D=int(pd.shape[1]), K=int(pu.shape[1]), phases=ph,
                 f_d=_exp(pd, node), f_p=_exp(pu, node))
        sn._want(pd, None, "disparity probabilities")
        sn._want(pu, None, "weights")
        sn._want(y, "f32", "output")
        return sn

    def _scale(self) -> float:
        return float(self.s) * 2.0 ** -(self.f_d + self.f_p)

    def describe(self):
        ph = ", 4 phases" if self.phases == 2 else ""
        return (f"disparity regression over {self.D} + context upsampling [{self.h}][{self.w}] x {self.s} "
                f"-> [{self.h * self.s}][{self.w * self.s}] float32 (3 x 3 weights of {self.K}{ph})")

    def c_call(self, ins, out, scratch, direct, dtype):
        return [f"stereo_upsample({ins[0]}, {ins[1]}, {self.h}u, {self.w}u, {self.s}u, {self.D}u, {self.K}u, "
                f"{self.phases}u, {_c_double(self._scale())}, {out});"]

    def _rows(self) -> np.ndarray:
        """Row of Pu for every output pixel [H][W]."""
        H, W = self.h * self.s, self.w * self.s
        yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        if self.phases == 1:
            return yy * W + xx
        q = (yy % 2) * 2 + xx % 2
        return q * (H // 2) * (W // 2) + (yy // 2) * (W // 2) + xx // 2

    def reference(self, ins, dtype):  # noqa: ARG002
        pd = _raw(ins[0], self.f_d).reshape(self.h * self.w, self.D)
        d = (pd * np.arange(self.D, dtype=np.int64)[None, :]).sum(axis=1).reshape(self.h, self.w)
        P = _raw(ins[1], self.f_p).reshape(-1, self.K)[self._rows()]            # [H][W][K]
        dp = np.pad(d, 1)
        H, W = self.h * self.s, self.w * self.s
        yy, xx = np.arange(H) // self.s, np.arange(W) // self.s
        acc = np.zeros((H, W), np.int64)
        for k in range(9):
            dy, dx = k // 3, k % 3
            acc += P[:, :, k] * dp[yy + dy][:, xx + dx]                   # d[y/s + dy - 1][x/s + dx - 1]
        out = (acc.astype(np.float64) * self._scale()).astype(np.float32)
        return out.astype(np.float64).reshape(self.output.shape)


STEREO_OP_FACTORIES = {
    "StereoVop":          StereoVopNode.from_onnx_node,
    "StereoSoftmax":      StereoSoftmaxNode.from_onnx_node,
    "StereoInstanceNorm": StereoInstanceNormNode.from_onnx_node,
    "StereoPadEdge":      StereoPadEdgeNode.from_onnx_node,
    "StereoCorrelation":  StereoCorrelationNode.from_onnx_node,
    "StereoUpsample":     StereoUpsampleNode.from_onnx_node,
}


STEREO_C = r"""
/* =============== Stereo host ops (axi.llm domain, src/stereo_nodes.py) =============== */

static inline Data_t stereo_st(double v)
{
    double r = nearbyint(v);
    r = r > 32767.0 ? 32767.0 : r;
    r = r < -32768.0 ? -32768.0 : r;
    return (Data_t)(int16_t)r;
}

/* Instance norm, one channel per task item: the exact sums of x and x^2
 * (int64; n * s2 - s1^2 in 128 bits), then mean, variance and the scale in
 * double; y = st((x - mean) * 2^f_y / sqrt(var + eps)). */
typedef struct { const Data_t *x; unsigned n; double eps; int fy; Data_t *y; } stereo_in_t;

static void stereo_instnorm_part(void *p, unsigned c0, unsigned c1)
{
    const stereo_in_t *a = (const stereo_in_t *)p;
    unsigned c, i;
    for (c = c0; c < c1; c++) {
        const Data_t *x = a->x + (size_t)c * a->n;
        Data_t       *y = a->y + (size_t)c * a->n;
        int64_t s1 = 0, s2 = 0;
        for (i = 0u; i < a->n; i++) {
            const int64_t v = (int16_t)x[i];
            s1 += v;
            s2 += v * v;
        }
        {
            const double    n   = (double)a->n;
            const double    m   = (double)s1 / n;
            const __int128  num = (__int128)a->n * s2 - (__int128)s1 * s1;
            const double    var = (double)num / (n * n);
            const double    k   = ldexp(1.0, a->fy) / sqrt(var + a->eps);
            for (i = 0u; i < a->n; i++)
                y[i] = stereo_st(((double)(int16_t)x[i] - m) * k);
        }
    }
}

static void stereo_instnorm(const Data_t *x, unsigned C, unsigned n, double eps, int fy, Data_t *y)
{
    stereo_in_t a;
    a.x = x; a.n = n; a.eps = eps; a.fy = fy; a.y = y;
    host_parallel(stereo_instnorm_part, &a, C, 1u, 1u);
}

/* Replicate padding by p, one output row per task item. */
typedef struct { const Data_t *x; unsigned H, W, p; Data_t *y; } stereo_pad_t;

static void stereo_pad_part(void *q, unsigned r0, unsigned r1)
{
    const stereo_pad_t *a = (const stereo_pad_t *)q;
    const unsigned Ho = a->H + 2u * a->p, Wo = a->W + 2u * a->p;
    unsigned r, j;
    for (r = r0; r < r1; r++) {
        const unsigned c = r / Ho, oy = r % Ho;
        const unsigned iy = oy < a->p ? 0u : (oy - a->p >= a->H ? a->H - 1u : oy - a->p);
        const Data_t  *src = a->x + ((size_t)c * a->H + iy) * a->W;
        Data_t        *dst = a->y + (size_t)r * Wo;
        for (j = 0u; j < a->p; j++) {
            dst[j] = src[0];
            dst[a->p + a->W + j] = src[a->W - 1u];
        }
        memcpy(dst + a->p, src, (size_t)a->W * sizeof(Data_t));
    }
}

static void stereo_pad_edge(const Data_t *x, unsigned C, unsigned H, unsigned W, unsigned p, Data_t *y)
{
    stereo_pad_t a;
    a.x = x; a.H = H; a.W = W; a.p = p; a.y = y;
    host_parallel(stereo_pad_part, &a, C * (H + 2u * p), 1u, 1u);
}

/* The correlation volume, one row h per task item: the row's left and right
 * features gathered channel-last (lr[w][c]), then per disparity d and column
 * w >= d the exact int64 dot product, divided by C and scaled in double. */
typedef struct { const Data_t *l, *r; unsigned C, H, W, D; double sc; Data_t *vol; } stereo_corr_t;

static void stereo_corr_part(void *p, unsigned h0, unsigned h1)
{
    const stereo_corr_t *a = (const stereo_corr_t *)p;
    const unsigned C = a->C, H = a->H, W = a->W;
    const size_t   plane = (size_t)H * W;
    int16_t *lr = (int16_t *)malloc((size_t)2u * W * C * sizeof(int16_t));
    int16_t *rr = lr + (size_t)W * C;
    unsigned h, c, w, d;
    if (!lr)
        return;
    for (h = h0; h < h1; h++) {
        for (c = 0u; c < C; c++)
            for (w = 0u; w < W; w++) {
                lr[(size_t)w * C + c] = (int16_t)a->l[c * plane + (size_t)h * W + w];
                rr[(size_t)w * C + c] = (int16_t)a->r[c * plane + (size_t)h * W + w];
            }
        for (d = 0u; d < a->D; d++) {
            Data_t *out = a->vol + (size_t)d * plane + (size_t)h * W;
            for (w = 0u; w < d && w < W; w++)
                out[w] = 0;
            for (w = d; w < W; w++) {
                const int16_t *x = lr + (size_t)w * C, *y = rr + (size_t)(w - d) * C;
                int64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
                for (c = 0u; c + 4u <= C; c += 4u) {
                    s0 += (int32_t)x[c] * y[c];
                    s1 += (int32_t)x[c + 1u] * y[c + 1u];
                    s2 += (int32_t)x[c + 2u] * y[c + 2u];
                    s3 += (int32_t)x[c + 3u] * y[c + 3u];
                }
                for (; c < C; c++)
                    s0 += (int32_t)x[c] * y[c];
                out[w] = stereo_st((double)(s0 + s1 + s2 + s3) / (double)C * a->sc);
            }
        }
    }
    free(lr);
}

static void stereo_corr(const Data_t *l, const Data_t *r, unsigned C, unsigned H, unsigned W, unsigned D,
                        double sc, Data_t *vol)
{
    stereo_corr_t a;
    a.l = l; a.r = r; a.C = C; a.H = H; a.W = W; a.D = D; a.sc = sc; a.vol = vol;
    host_parallel(stereo_corr_part, &a, H, 1u, 1u);
}

/* The disparity regression, one low-resolution row per task item: d = the
 * exact int64 sum of Pd[j] * j. */
typedef struct { const Data_t *pd; unsigned w, D; int64_t *d; } stereo_reg_t;

static void stereo_regress_part(void *q, unsigned r0, unsigned r1)
{
    const stereo_reg_t *a = (const stereo_reg_t *)q;
    unsigned t, j;
    for (t = r0 * a->w; t < r1 * a->w; t++) {
        const Data_t *p = a->pd + (size_t)t * a->D;
        int64_t acc = 0;
        for (j = 1u; j < a->D; j++)
            acc += (int64_t)(int16_t)p[j] * (int64_t)j;
        a->d[t] = acc;
    }
}

/* The context upsampling, one output row per task item: the exact int64 sum
 * of Pu[k] * d over the 3 x 3 neighbourhood (zero outside), scaled in double,
 * stored as float32.  Pu's row of pixel (y, x): y*W + x, or with 2 phases
 * (2*(y%2) + x%2)*(H/2)*(W/2) + (y/2)*(W/2) + x/2. */
typedef struct { const int64_t *d; const Data_t *p; unsigned h, w, s, K, phases; double sc; float *out; } stereo_up_t;

static void stereo_upsample_part(void *q, unsigned y0, unsigned y1)
{
    const stereo_up_t *a = (const stereo_up_t *)q;
    const unsigned H = a->h * a->s, W = a->w * a->s, H2 = H / 2u, W2 = W / 2u;
    unsigned y, x, k;
    for (y = y0; y < y1; y++) {
        const int yl = (int)(y / a->s);
        for (x = 0u; x < W; x++) {
            const int      xl = (int)(x / a->s);
            const size_t   row = a->phases == 2u
                ? ((size_t)((y % 2u) * 2u + x % 2u) * H2 + y / 2u) * W2 + x / 2u
                : (size_t)y * W + x;
            const Data_t  *pw = a->p + row * a->K;
            int64_t        acc = 0;
            for (k = 0u; k < 9u; k++) {
                const int yy = yl + (int)(k / 3u) - 1, xx = xl + (int)(k % 3u) - 1;
                if (yy >= 0 && yy < (int)a->h && xx >= 0 && xx < (int)a->w)
                    acc += (int64_t)(int16_t)pw[k] * a->d[(size_t)yy * a->w + (unsigned)xx];
            }
            a->out[(size_t)y * W + x] = (float)((double)acc * a->sc);
        }
    }
}

static void stereo_upsample(const Data_t *pd, const Data_t *p, unsigned h, unsigned w, unsigned s, unsigned D,
                            unsigned K, unsigned phases, double sc, float *out)
{
    stereo_reg_t r;
    stereo_up_t  a;
    int64_t     *d = (int64_t *)malloc((size_t)h * w * sizeof(int64_t));
    if (!d)
        return;
    r.pd = pd; r.w = w; r.D = D; r.d = d;
    host_parallel(stereo_regress_part, &r, h, 1u, 1u);
    a.d = d; a.p = p; a.h = h; a.w = w; a.s = s; a.K = K; a.phases = phases; a.sc = sc; a.out = out;
    host_parallel(stereo_upsample_part, &a, h * s, 1u, 1u);
    free(d);
}
"""


def stereo_c_helpers() -> str:
    return STEREO_C


__all__ = ("STEREO_OP_FACTORIES", "StereoVopNode", "StereoSoftmaxNode", "StereoInstanceNormNode",
           "StereoPadEdgeNode", "StereoCorrelationNode", "StereoUpsampleNode", "stereo_c_helpers")
