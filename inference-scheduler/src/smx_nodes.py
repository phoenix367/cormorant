"""
VectorOPKernel's softmax in the scheduler (doc/plans/SOFTMAX_PLAN.md §2.3).

The kernel's softmax unit (``kernels/vectorop_rtl/rtl/vo_smx.sv``; ops 10 / 11,
registers ``smx_cm`` / ``smx_cfg`` / ``smx_mask``) computes the integer
specification of ``vectorop_smx.py`` bit for bit — an approximation of the
exact softmax within one LSB of P.  Three node kinds issue it:

  * ``SoftmaxVopNode`` — an ONNX ``Softmax`` (BERT) on Q8.8 DMA tensors, the
    last axis of at most 2048 elements: one ``OP_SOFTMAX`` call over the rows
    (scores at 2^-8, logit scale 1, P at 2^-8).  Chosen by ``from_onnx`` where
    the platform has the unit (``_vectorop_hw_config.VECTOROP_SOFTMAX``), else
    the host op ``SoftmaxNode`` (double precision) as before.
  * ``VitAttnSoftmaxVopNode`` — a ``VitAttnSoftmax`` with the attribute
    ``vsmx = 1`` (the vision frontend's ``vsmx`` option, the study policy
    ``pow2+p12+vgelu+vsmx``): one ``OP_SOFTMAX_T`` call — the keys-major
    scores s [C][T] -> P [nc][C], every key valid.
  * ``LlmAttnSoftmaxVopNode`` — a prefill ``LlmAttnSoftmax`` with ``vsmx = 1``
    (the Llama frontend's ``vsmx`` option, ``pow2+sink+p12+vsmx``): per head
    of the group one ``OP_SOFTMAX_T`` call over the runtime key count
    keys = roundup(pos + n, Q), query row t valid over keys
    j < min(keys, pos + 1 + t) (smx_mask valid0 = pos + 1, period = T) —
    the padded rows t >= n too (they only feed padded rows).

A node with ``vsmx = 1`` needs the unit: its numbers are the kernel's, and the
host has no copy of them (the frontends set it only where the platform has the
unit).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, List, Optional

import numpy as np

from . import _vectorop_hw_config
from . import vectorop_smx as smx
from .host_nodes import HostContext, SoftmaxNode, _attrs, _label
from .llm_nodes import KEY_QUANTUM, LlmKernelNode, _i32, _require, attn_keys
from .nodes import SchedulerError

OP_SOFTMAX = 10              # row mode (VectorOP.h)
OP_SOFTMAX_T = 11            # column mode
COL_BLOCK = 16               # column mode: query columns per block (outer % 16 == 0)


def enabled() -> bool:
    """The platform's VectorOPKernel has the softmax unit (read at call time,
    so a test can patch ``_vectorop_hw_config.VECTOROP_SOFTMAX``)."""
    return _vectorop_hw_config.VECTOROP_SOFTMAX


def cfg_reg(cs: int, fp: int) -> int:
    """smx_cfg: Cs [5:0], f_p [12:8]."""
    return (int(cs) & 63) | ((int(fp) & 31) << 8)


def mask_reg(valid0: int, period: int = 0) -> int:
    """smx_mask: valid0 [15:0], period [31:16]."""
    return (int(valid0) & 0xFFFF) | ((int(period) & 0xFFFF) << 16)


def regs(f_s: int, sigma: float, f_p: int):
    """(smx_cm, smx_cfg) of scores at 2^-f_s, logit scale ``sigma``, P at 2^-f_p."""
    cm, cs = smx.scale_regs(f_s, sigma)
    return cm, cfg_reg(cs, f_p)


def _unpack(cfg: int):
    return int(cfg) & 63, (int(cfg) >> 8) & 31


@dataclass
class SmxVopNode(LlmKernelNode):
    """A node of VectorOPKernel softmax calls (run_softmax: the registers of
    run_op plus smx_cm / smx_cfg / smx_mask; element offsets into a and c)."""

    kernel_name: ClassVar[str] = "VectorOPKernel"
    uses_softmax_unit: ClassVar[bool] = True

    onnx_node:   object
    inputs:      list
    output:      object
    index:       int = 0
    align_elems: int = 8
    F:           int = 8

    # Compatibility shims (read by the layout / header passes)
    outer_count:        int  = field(default=1,    init=False)
    chunk_size:         int  = field(default=0,    init=False)
    aligned_chunk_size: int  = field(default=0,    init=False)
    a_advances:         bool = field(default=True, init=False)
    b_advances:         bool = field(default=True, init=False)
    arity:              int  = field(default=1,    init=False)

    @staticmethod
    def call(a: str, a_off, c: str, c_off, size, op: int, outer, a_inc, b_inc, cm: int, cfg: int,
             mask, indent: str = "    ") -> str:
        name = "VECTOROP_SOFTMAX_T" if op == OP_SOFTMAX_T else "VECTOROP_SOFTMAX"
        u = lambda v: f"{v}u" if isinstance(v, int) else str(v)   # noqa: E731
        return (f"{indent}run_softmax({a}, {u(a_off)}, {c}, {u(c_off)}, {u(size)}, {name}, "
                f"{u(outer)}, {u(a_inc)}, {u(b_inc)},\n"
                f"{indent}            {cm}u, 0x{cfg:04X}u, {u(mask)});")


# ------------------------------------------------------------------ #
# ONNX Softmax (row mode)                                              #
# ------------------------------------------------------------------ #

@dataclass
class SoftmaxVopNode(SmxVopNode):
    """ONNX Softmax on VectorOPKernel (row mode): ``rows`` rows of ``n``
    elements, Q8.8 in and out — scores at 2^-8 with logit scale 1, P at 2^-8,
    every element valid."""
    rows: int = 1
    n:    int = 1
    cm:   int = 0
    cfg:  int = 0

    def reference(self, ins, dtype):  # noqa: ARG002
        raw = np.rint(np.asarray(ins[0], np.float64).reshape(self.rows, self.n) * 2.0 ** self.F)
        cs, fp = _unpack(self.cfg)
        p = smx.softmax_raw(raw.astype(np.int64), self.n, self.cm, cs, fp)
        return (p / 2.0 ** fp).reshape(self.output.shape)

    def _strides(self, layouts: dict):
        """(a_inc, b_inc): the rows' strides in the input / output layouts —
        n where the buffer has no gaps (flat, or chunks without alignment
        padding), a chunked layout's stride where its chunks are the rows."""
        out = []
        for t in (self.inputs[0], self.output):
            lay = layouts.get(t.onnx_name) if layouts else None
            if lay is None or lay.n_chunks <= 1 or not lay.is_strided:
                out.append(self.n)
                continue
            if lay.chunk != self.n or lay.stride % self.align_elems:
                raise SchedulerError(
                    f"Softmax node '{_label(self.onnx_node)}': tensor '{t.onnx_name}' is laid "
                    f"out in chunks of {lay.chunk} at stride {lay.stride}, not rows of {self.n}")
            out.append(lay.stride)
        return tuple(out)

    def kernel_calls(self, layouts: dict) -> list:
        from .perf_calls import KernelCall
        a_inc, b_inc = self._strides(layouts)
        return [KernelCall.of("VectorOPKernel", op=OP_SOFTMAX, size=self.n, outer=self.rows,
                              a_inc=a_inc, b_inc=b_inc, act=0)]

    def describe(self) -> str:
        return (f"rows={self.rows} n={self.n} on VectorOPKernel's softmax unit (row mode; "
                f"Q8.8 in and out)")

    def emit_comment(self) -> str:
        return (f"    /* [{self.index}] Softmax({self.inputs[0].onnx_name}) -> {self.output.onnx_name}"
                f"  [{self.rows}][{self.n}] on VectorOPKernel's softmax unit (row mode) */")

    def emit_call(self, layouts: dict) -> str:
        a_inc, b_inc = self._strides(layouts)
        return self.call(self.inputs[0].c_name, 0, self.output.c_name, 0, self.n, OP_SOFTMAX,
                         self.rows, a_inc, b_inc, self.cm, self.cfg, mask_reg(self.n))


def softmax_ineligible(host: SoftmaxNode, tensors, dtype) -> Optional[str]:
    """Why the ONNX Softmax ``host`` cannot run on the unit, or None."""
    if not enabled():
        return ("the platform's VectorOPKernel has no softmax unit "
                "(platforms/<platform>.json kernels.vectorop.softmax, AXI_VECTOROP_SOFTMAX)")
    if getattr(dtype, "name", "") != "ap_fixed<16,8>":
        return f"the softmax unit reads ap_fixed<16,8>, not {getattr(dtype, 'name', dtype)}"
    for t in (host.inputs[0], host.output):
        if t.exp is not None or t.host is not None or t.is_state:
            return f"tensor '{t.onnx_name}' is not a plain Q8.8 DMA tensor"
    if host.n > smx.MAX_ROW:
        return f"rows of {host.n} elements (the unit buffers {smx.MAX_ROW})"
    if host.rows > 1 and host.n % host.align_elems:
        return f"rows of {host.n} elements are not 16-byte aligned"
    return None


def from_onnx_softmax(node, tensors, index: int, align_elems: int, ctx: HostContext, dtype):
    """The scheduled node of an ONNX Softmax: SoftmaxVopNode where the unit can
    run it, else the host SoftmaxNode."""
    host = SoftmaxNode.from_onnx_node(node, tensors, index, align_elems, ctx)
    if softmax_ineligible(host, tensors, dtype) is not None:
        return host
    cm, cfg = regs(ctx.frac_bits, 1.0, ctx.frac_bits)
    return SoftmaxVopNode(onnx_node=node, inputs=list(host.inputs), output=host.output,
                          index=index, align_elems=align_elems, F=ctx.frac_bits,
                          rows=host.rows, n=host.n, cm=cm, cfg=cfg)


# ------------------------------------------------------------------ #
# VitAttnSoftmax (column mode, static keys)                            #
# ------------------------------------------------------------------ #

@dataclass
class VitAttnSoftmaxVopNode(SmxVopNode):
    """VitAttnSoftmax on VectorOPKernel (column mode): s [C][T] raw scores at
    2^-f_s -> P [nc][C] raw at 2^-f_p, query columns c0 .. c0 + nc - 1,
    every key valid."""
    T:   int = 1
    C:   int = 16
    c0:  int = 0
    nc:  int = 0
    cm:  int = 0
    cfg: int = 0

    @classmethod
    def from_host(cls, host, node):
        """``host``: the VitAttnSoftmaxNode the attribute vsmx = 1 replaces."""
        T, C, c0, nc = host.T, host.C, host.c0, host.nc
        _require(enabled(), node, "vsmx = 1 needs VectorOPKernel's softmax unit "
                                  "(kernels.vectorop.softmax, AXI_VECTOROP_SOFTMAX)")
        _require(C <= smx.MAX_KEYS, node, f"{C} keys (the unit takes {smx.MAX_KEYS})")
        _require(nc % COL_BLOCK == 0 and c0 % 8 == 0 and T % 8 == 0 and C % 8 == 0, node,
                 f"columns [{c0}, {nc}] of {T}, {C} keys: the unit takes blocks of "
                 f"{COL_BLOCK} columns at 16-byte aligned rows")
        cm, cfg = regs(host.fs, host.scale, host.fp)
        return cls(onnx_node=node, inputs=list(host.inputs), output=host.output, index=host.index,
                   align_elems=host.align_elems, F=host.F, T=T, C=C, c0=c0, nc=nc, cm=cm, cfg=cfg)

    def reference(self, ins, dtype):  # noqa: ARG002
        s = np.asarray(ins[0], np.float64).astype(np.int64)[:, self.c0:self.c0 + self.nc]  # [C][nc]
        cs, fp = _unpack(self.cfg)
        return smx.softmax_cols(s, self.C, self.cm, cs, fp).astype(np.float64)

    def kernel_calls(self, layouts: dict) -> list:  # noqa: ARG002
        from .perf_calls import KernelCall
        return [KernelCall.of("VectorOPKernel", op=OP_SOFTMAX_T, size=self.C, outer=self.nc,
                              a_inc=self.T, b_inc=self.C, act=0)]

    def describe(self) -> str:
        cols = f" (columns {self.c0}..{self.c0 + self.nc - 1})" if (self.c0, self.nc) != (0, self.T) else ""
        cs, fp = _unpack(self.cfg)
        return (f"{self.T} query columns{cols} x {self.C} keys on VectorOPKernel's softmax unit "
                f"(column mode; Cm {self.cm} >> {cs}, P at 2^-{fp})")

    def emit_comment(self) -> str:
        return (f"    /* [{self.index}] VitAttnSoftmax({self.inputs[0].onnx_name}) -> "
                f"{self.output.onnx_name}  s [{self.C}][{self.T}] -> P [{self.nc}][{self.C}] on "
                f"VectorOPKernel's softmax unit (column mode) */")

    def emit_call(self, layouts: dict) -> str:  # noqa: ARG002
        return self.call(self.inputs[0].c_name, self.c0, self.output.c_name, 0, self.C,
                         OP_SOFTMAX_T, self.nc, self.T, self.C, self.cm, self.cfg,
                         mask_reg(self.C))


# ------------------------------------------------------------------ #
# LlmAttnSoftmax, prefill (column mode, runtime keys, causal)          #
# ------------------------------------------------------------------ #

@dataclass
class LlmAttnSoftmaxVopNode(SmxVopNode):
    """The prefill LlmAttnSoftmax of KV group ``group`` on VectorOPKernel: per
    head h' < G one column-mode call over s_g [C][G*T] (columns h'*T .. +T) ->
    P_g rows h'*T .. +T at the runtime row stride keys = roundup(pos + n, Q);
    row t valid over keys j < min(keys, pos + 1 + t)."""
    # a runtime key count (as LlmAttnConvNode's): the timing / calibration price
    # its calls at keys_of(sn) keys, kernel_calls(layouts, keys=...)
    static: ClassVar[bool] = False

    T:     int = 1
    G:     int = 1
    C:     int = 16
    Q:     int = KEY_QUANTUM
    group: int = 0
    has_n: bool = True
    cm:    List[int] = field(default_factory=list)                 # [G]
    cfg:   List[int] = field(default_factory=list)                 # [G]

    @classmethod
    def from_host(cls, host, node):
        """``host``: the LlmAttnSoftmaxNode the attribute vsmx = 1 replaces."""
        _require(enabled(), node, "vsmx = 1 needs VectorOPKernel's softmax unit "
                                  "(kernels.vectorop.softmax, AXI_VECTOROP_SOFTMAX)")
        _require(host.C <= smx.MAX_KEYS, node, f"{host.C} cache rows (the unit takes "
                                              f"{smx.MAX_KEYS} keys)")
        _require(host.T % COL_BLOCK == 0, node, f"{host.T} query rows: the unit takes blocks of "
                                                f"{COL_BLOCK}")
        _require(host.Q % 8 == 0 and host.C < 1 << 16, node, "key quantum / cache rows")
        cm, cfg = [], []
        for h in range(host.G):
            a, b = regs(int(host.fs[h]), host.scale, int(host.fp[h]))
            cm.append(a)
            cfg.append(b)
        return cls(onnx_node=node, inputs=list(host.inputs), output=host.output, index=host.index,
                   align_elems=host.align_elems, F=host.F, T=host.T, G=host.G, C=host.C, Q=host.Q,
                   group=host.group, has_n=host.has_n, cm=cm, cfg=cfg)

    def reference(self, ins, dtype):  # noqa: ARG002
        T, G, C = self.T, self.G, self.C
        s = np.asarray(ins[0], np.float64).astype(np.int64)                # [C][G*T]
        pos = _i32(ins[1])
        _, keys = attn_keys(pos, _i32(ins[2]) if self.has_n else T, T, C, self.Q)
        valid = smx.causal_valid(T, pos + 1, T, keys)                     # [T]
        P = np.zeros((G * T, C))
        for h in range(G):
            cs, fp = _unpack(self.cfg[h])
            P[h * T:(h + 1) * T, :keys] = smx.softmax_cols(s[:keys, h * T:(h + 1) * T], valid,
                                                           self.cm[h], cs, fp)
        return P

    def kernel_calls(self, layouts: dict, keys: int = None) -> list:  # noqa: ARG002
        """The G calls emit_call() issues at ``keys`` keys (default C, the most
        a call reads)."""
        from .perf_calls import KernelCall
        keys = self.C if keys is None else int(keys)
        return [KernelCall.of("VectorOPKernel", op=OP_SOFTMAX_T, size=keys, outer=self.T,
                              a_inc=self.G * self.T, b_inc=keys, act=0)] * self.G

    def describe(self) -> str:
        return (f"group {self.group}: {self.G} heads x {self.T} query rows on VectorOPKernel's "
                f"softmax unit (column mode; keys = roundup(pos + n, {self.Q}) at run time, "
                f"causal: row t over keys <= pos + t)")

    def emit_comment(self) -> str:
        return (f"    /* [{self.index}] LlmAttnSoftmax({self.inputs[0].onnx_name}) -> "
                f"{self.output.onnx_name}  group {self.group}: s [keys][{self.G * self.T}] -> "
                f"P [{self.G * self.T}][keys] on VectorOPKernel's softmax unit, {self.G} call(s); "
                f"keys = roundup(pos + n, {self.Q}), row t valid to pos + t */")

    def emit_call(self, layouts: dict) -> str:  # noqa: ARG002
        s, pos = self.inputs[0].c_name, self.inputs[1].c_name
        n = f"(unsigned){self.inputs[2].c_name}[0]" if self.has_n else f"{self.T}u"
        lines = ["    {",
                 f"        const unsigned _pos = (unsigned){pos}[0];",
                 f"        const unsigned _keys = llm_keys(_pos, {n}, {self.T}u, {self.C}u, {self.Q}u);",
                 f"        const unsigned _mask = ((_pos + 1u) & 0xFFFFu) | ({self.T}u << 16);"]
        for h in range(self.G):
            if h:
                lines.append("        kernel_wait(KERNEL_VECTOROP);")
            lines.append(self.call(s, h * self.T, self.output.c_name, f"{h * self.T}u * _keys",
                                   "_keys", OP_SOFTMAX_T, self.T, self.G * self.T, "_keys",
                                   self.cm[h], self.cfg[h], "_mask", indent="        "))
        lines.append("    }")
        return "\n".join(lines)


def vsmx_attr(node) -> bool:
    """The node asks for the softmax unit (attribute vsmx = 1)."""
    return int(_attrs(node).get("vsmx", 0)) != 0


__all__ = ("OP_SOFTMAX", "OP_SOFTMAX_T", "COL_BLOCK", "enabled", "cfg_reg", "mask_reg", "regs",
           "SmxVopNode", "SoftmaxVopNode", "VitAttnSoftmaxVopNode", "LlmAttnSoftmaxVopNode",
           "softmax_ineligible", "from_onnx_softmax", "vsmx_attr")
