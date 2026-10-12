"""
The row-ops unit in the scheduler (doc/plans/HOSTOPS_PLAN.md §6; the kernel:
kernels/rowops_rtl, doc/kernels/ROWOPS_RTL_KERNEL.md).

RowOpsKernel is the fifth kernel lane: one read+write m_axi port, jobs of
``rows`` rows of ``n`` elements (times ``blocks``), every column's parameters
(G, Bq', three shifts) in a 16-byte parameter block P the job reads first,
the sigmoid and rsqrt tables in UltraRAM loaded once per bitstream from a
64 KiB image (OP_TLOAD, which ``inference_init()`` issues).  What a job
computes is the integer specification of ``src/rowops_spec.py`` bit for bit;
the nodes here compute their results with those functions, emit the
``run_rowops()`` call and list it for the performance models.

The nodes are chosen only where the platform has the unit
(``_rowops_hw_config.ROWOPS_UNIT``, ``AXI_ROWOPS``); otherwise the host op
stays as before.  Constant operands of a node that are not ONNX initializers
— the parameter block, the table image — are registered as weight tensors
(raw int16 words carried as the float values whose Q8.8 encoding is those
words, so every weight path emits the bits unchanged).

Node kinds:

  * ``LayerNormRoNode`` — an ONNX / fused ``LayerNormalization`` on Q8.8 DMA
    tensors (BERT): one NORM job with NORM_LN, ``rowops_spec.layer_norm``
    (gamma at 2^-g, beta as Bq' at 2^-bb: 2^-20, coarser where |beta| >= 2).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Optional

import numpy as np

from . import _rowops_hw_config
from . import rowops_spec as rs
from .host_nodes import HostContext, LayerNormNode, _label
from .llm_nodes import LlmKernelNode
from .nodes import SchedulerError
from .tensor import TensorInfo

TABLES_NAME = "rowops_tables"            # the table image: one weight per project
NO_WINDOW = 0xFFFFFFFF


def enabled() -> bool:
    """The platform's bitstream has the row-ops unit (read at call time, so a
    test can patch ``_rowops_hw_config.ROWOPS_UNIT``)."""
    return _rowops_hw_config.ROWOPS_UNIT


def raw_as_data(raw) -> np.ndarray:
    """Raw int16 words as the float32 values whose Q8.8 encoding is those
    words (exact: every int16 / 256 is a float32)."""
    return (np.asarray(raw, np.float64).reshape(-1) / 256.0).astype(np.float32)


def tables_tensor(tensors: dict) -> TensorInfo:
    """The project's table-image weight (64 KiB as 32 768 int16 words),
    created on first use."""
    t = tensors.get(TABLES_NAME)
    if t is None:
        raw = np.frombuffer(rs.table_image(), np.int16)
        t = TensorInfo(onnx_name=TABLES_NAME, shape=[int(raw.size)], dtype="float32",
                       data=raw_as_data(raw))
        tensors[TABLES_NAME] = t
    return t


def param_tensor(tensors: dict, name: str, words: np.ndarray) -> TensorInfo:
    """A parameter block (``rowops_spec.param_block`` words) as a weight tensor
    named ``name``."""
    t = TensorInfo(onnx_name=name, shape=[int(np.asarray(words).size)], dtype="float32",
                   data=raw_as_data(words))
    tensors[name] = t
    return t


def row_stride_bytes(t, layouts: dict, n: int, elem_bytes: int, node, align_elems: int) -> int:
    """The byte stride between the rows of ``t`` (rows of ``n`` elements): a
    chunked layout's stride where its chunks are the rows, else n elements."""
    lay = layouts.get(t.onnx_name) if layouts else None
    if lay is None or lay.n_chunks <= 1 or not lay.is_strided:
        return n * elem_bytes
    if lay.chunk != n or lay.stride % align_elems:
        raise SchedulerError(
            f"{node.op_type} node '{_label(node)}': tensor '{t.onnx_name}' is laid out in chunks "
            f"of {lay.chunk} at stride {lay.stride}, not rows of {n}")
    return int(lay.stride) * elem_bytes


@dataclass
class RowOpsNode(LlmKernelNode):
    """A node of one or more RowOpsKernel jobs (run_rowops: every register of
    the kernel; byte offsets into the buffers)."""

    kernel_name: ClassVar[str] = "RowOpsKernel"

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
    def call(a, a_off, b, b_off, c, c_off, p, op: int, n, rows, blocks, a_s1, a_s2, b_s1, b_s2,
             c_s1, c_s2, eps: int, aux, bq_sh, win_lo, win_hi, indent: str = "    ") -> str:
        u = lambda v: f"{v}u" if isinstance(v, int) else str(v)   # noqa: E731
        return (f"{indent}run_rowops({a}, {u(a_off)}, {b}, {u(b_off)}, {c}, {u(c_off)}, {p},\n"
                f"{indent}           0x{op:05X}u, {u(n)}, {u(rows)}, {u(blocks)}, "
                f"{u(a_s1)}, {u(a_s2)}, {u(b_s1)}, {u(b_s2)}, {u(c_s1)}, {u(c_s2)},\n"
                f"{indent}           {int(eps)}ull, {u(aux)}, {u(bq_sh)}, {u(win_lo)}, {u(win_hi)});")


# ------------------------------------------------------------------ #
# LayerNormalization (BERT): NORM with NORM_LN on Q8.8 tensors         #
# ------------------------------------------------------------------ #

@dataclass
class LayerNormRoNode(RowOpsNode):
    """ONNX / fused LayerNormalization on the unit: ``rows`` rows of ``n``
    Q8.8 elements -> Q8.8, ``rowops_spec.layer_norm`` (the exact integer
    sums, the table + Newton rsqrt, gamma at 2^-g, beta as Bq' at 2^-20,
    one rounding at the output).  inputs [x, P, tables]."""
    rows:  int = 1
    n:     int = 1
    eps:   float = 1e-5
    g:     int = 0                                  # gamma's exponent
    bb:    int = rs.BQ_BITS                         # beta's scale (Bq' = round(beta 2^-bb))
    E:     int = 0                                  # the eps register
    gamma: Optional[np.ndarray] = field(default=None, repr=False)
    beta:  Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def opword(self) -> int:
        return rs.opword(rs.OP_NORM, rs.K_I16, rs.K_I16, rs.K_I16, ln=True)

    def _strides(self, layouts: dict):
        return (row_stride_bytes(self.inputs[0], layouts, self.n, 2, self.onnx_node, self.align_elems),
                row_stride_bytes(self.output, layouts, self.n, 2, self.onnx_node, self.align_elems))

    def reference(self, ins, dtype):  # noqa: ARG002
        x = np.rint(np.asarray(ins[0], np.float64).reshape(self.rows, self.n) * 2.0 ** self.F)
        y = rs.layer_norm(x.astype(np.int64), self.F, self.gamma, self.beta, self.eps, self.F)
        return (y / 2.0 ** self.F).reshape(self.output.shape)

    def kernel_calls(self, layouts: dict) -> list:
        from .perf_calls import KernelCall
        a_s1, c_s1 = self._strides(layouts)
        return [KernelCall.of("RowOpsKernel", op=self.opword, n=self.n, rows=self.rows, blocks=1,
                              a_s1=a_s1, c_s1=c_s1, win_hi=NO_WINDOW)]

    def describe(self) -> str:
        return (f"rows={self.rows} n={self.n} eps={self.eps:.3g} on RowOpsKernel (NORM_LN; gamma at "
                f"2^-{self.g}, beta as Bq' at 2^-{self.bb}, Q8.8 in and out)")

    def emit_comment(self) -> str:
        return (f"    /* [{self.index}] LayerNormalization({self.inputs[0].onnx_name}) -> "
                f"{self.output.onnx_name}  [{self.rows}][{self.n}] on RowOpsKernel (NORM_LN) */")

    def emit_call(self, layouts: dict) -> str:
        a_s1, c_s1 = self._strides(layouts)
        return self.call(self.inputs[0].c_name, 0, "NULL", 0, self.output.c_name, 0, self.inputs[1].c_name,
                         self.opword, self.n, self.rows, 1, a_s1, 0, 0, 0, c_s1, 0, self.E, 0,
                         rs.P1_BITS + self.g - self.bb, 0, NO_WINDOW)


def layernorm_ineligible(host: LayerNormNode, dtype) -> Optional[str]:
    """Why the LayerNormalization ``host`` cannot run on the unit, or None."""
    if not enabled():
        return "the platform has no row-ops unit (platforms/<platform>.json kernels.rowops.unit, AXI_ROWOPS)"
    if getattr(dtype, "name", "") != "ap_fixed<16,8>":
        return f"the unit's LayerNorm reads ap_fixed<16,8>, not {getattr(dtype, 'name', dtype)}"
    for t in (host.inputs[0], host.output):
        if t.exp is not None or t.host is not None or t.is_state or t.is_int:
            return f"tensor '{t.onnx_name}' is not a plain Q8.8 DMA tensor"
    if host.n > rs.N_MAX:
        return f"rows of {host.n} elements (the unit takes {rs.N_MAX})"
    if host.rows > 1 and host.n % host.align_elems:
        return f"rows of {host.n} elements are not 16-byte aligned"
    n = host.n
    gamma = np.ones(n) if host.gamma is None else np.asarray(host.gamma, np.float64)
    beta = np.zeros(n) if host.beta is None else np.asarray(host.beta, np.float64)
    if not (np.isfinite(gamma).all() and np.isfinite(beta).all()):
        return "gamma / beta are not finite"
    g = rs.gamma_scale(gamma)
    if g + 4 > 63 or rs.P1_BITS + g - 8 > 63:
        return f"gamma's exponent 2^-{g} is beyond the unit's shifts"
    bb = rs.beta_bits(beta)
    if (np.abs(np.rint(beta * 2.0 ** bb)) >= 2 ** 21).any() or rs.P1_BITS + g - bb > 63:
        return "beta beyond the parameter block's int22"
    e = host.eps * n * n * 2.0 ** 16
    if not 0 <= e < 2.0 ** 63:
        return f"eps {host.eps} (E = eps n^2 2^16 beyond 63 bits)"
    return None


def from_onnx_layernorm(node, tensors, index: int, align_elems: int, ctx: HostContext, dtype):
    """The scheduled node of a LayerNormalization: LayerNormRoNode where the unit
    can run it, else the host LayerNormNode."""
    host = LayerNormNode.from_onnx_node(node, tensors, index, align_elems, ctx)
    if layernorm_ineligible(host, dtype) is not None:
        return host
    n, F = host.n, ctx.frac_bits
    gamma = np.ones(n) if host.gamma is None else np.asarray(host.gamma, np.float64)
    beta = np.zeros(n) if host.beta is None else np.asarray(host.beta, np.float64)
    G, g, _bq, E = rs.norm_params(gamma, n, host.eps, F, beta=beta)
    bb = rs.beta_bits(beta)
    bqp = np.rint(beta * 2.0 ** bb).astype(np.int64)                  # Bq' (int22), added << (24 + g - bb)
    words = rs.param_block(G, bqp, 0, 0, rs.P1_BITS + g - F)
    p = param_tensor(tensors, f"{host.output.onnx_name}__rowops_p", words)
    tb = tables_tensor(tensors)
    return LayerNormRoNode(onnx_node=node, inputs=[host.inputs[0], p, tb], output=host.output,
                           index=index, align_elems=align_elems, F=F, rows=host.rows, n=n,
                           eps=host.eps, g=int(g), bb=int(bb), E=int(E), gamma=gamma, beta=beta)


# ------------------------------------------------------------------ #
# Transpose (BERT's head splits / merges and K^T; RPU_PLAN §3.1)      #
# ------------------------------------------------------------------ #

@dataclass
class TransposeRoNode(RowOpsNode):
    """An ONNX Transpose of an int16 DMA tensor as one unit job (a bit copy,
    the reference is numpy's transpose).  After the host node's collapse of
    the output dims (``host_nodes._collapse``: adjacent dims merged, size-1
    dropped) two shapes map:

      * **[D2][D1][n] from source strides [n, D2·n, 1]** — the two outer axes
        swapped, the rows kept (a head split / merge): one COPY of n elements,
        ``rows = D1``, ``blocks = D2`` (A: a_s1 = D2·n, a_s2 = n; C: c_s1 = n,
        c_s2 = D1·n — in bytes);
      * **[B][L][M] from source strides [sB, 1, sM]** — the innermost output
        axis gathers source elements sM apart (K^T): one TRANSPOSE of tiles
        of 8 source rows (m) x L into granules of 8 at the output's row
        stride M, ``rows = M / 8`` tiles, ``blocks = B``.

    Everything else (odd strides, a last dim not a multiple of 8, other
    element sizes) stays the host op."""
    kind: str = "copy"            # "copy" | "xpose"
    n: int = 0                    # copy: elements per row; xpose: L
    rows: int = 0
    blocks: int = 1
    a_off: int = 0
    strides: tuple = (0, 0, 0, 0, 0, 0)       # a_s1, a_s2, b_s1, b_s2, c_s1, c_s2 (bytes)
    perm: list = field(default_factory=list)
    pz: object = field(default=None, repr=False)

    @property
    def opword(self) -> int:
        return rs.opword(rs.OP_COPY, p_bcast=True) if self.kind == "copy" else rs.opword(rs.OP_XPOSE)

    @property
    def aux(self) -> int:
        return 0 if self.kind == "copy" else QIMAGE_R

    def reference(self, ins, dtype):  # noqa: ARG002
        x = np.asarray(ins[0]).reshape(self.inputs[0].shape)
        return np.ascontiguousarray(np.transpose(x, self.perm)).reshape(self.output.shape)

    def kernel_calls(self, layouts: dict) -> list:  # noqa: ARG002
        from .perf_calls import KernelCall
        s = self.strides
        return [KernelCall.of("RowOpsKernel", op=self.opword, n=self.n, rows=self.rows, blocks=self.blocks,
                              a_s1=s[0], a_s2=s[1], b_s1=s[2], b_s2=s[3], c_s1=s[4], c_s2=s[5], aux=self.aux,
                              win_lo=0, win_hi=NO_WINDOW)]

    def describe(self) -> str:
        return (f"perm={''.join(str(p) for p in self.perm)} on RowOpsKernel ("
                + ("COPY" if self.kind == "copy" else "TRANSPOSE, 8-row tiles")
                + f": n={self.n} rows={self.rows} blocks={self.blocks})")

    def emit_comment(self) -> str:
        return (f"    /* [{self.index}] Transpose({self.inputs[0].onnx_name}) -> {self.output.onnx_name}  "
                f"{self.describe()} */")

    def emit_call(self, layouts: dict) -> str:  # noqa: ARG002
        s = self.strides
        return self.call(self.inputs[0].c_name, self.a_off, "NULL", 0, self.output.c_name, 0,
                         self.pz.c_name if self.kind == "copy" else "NULL", self.opword, self.n, self.rows,
                         self.blocks, s[0], s[1], s[2], s[3], s[4], s[5], 0, self.aux, 0, 0, NO_WINDOW)


def transpose_job(dims: list, sstr: list, elem_bytes: int = 2):
    """The unit job of a collapsed transpose (``host_nodes._collapse`` output:
    5 dims left-padded with 1s and their source strides), or None where the
    unit cannot do it.  Returns (kind, n, rows, blocks, strides)."""
    d = [(int(a), int(b)) for a, b in zip(dims, sstr, strict=True) if a != 1]
    if elem_bytes != 2 or not d or len(d) > 3:
        return None
    al = 16 // elem_bytes
    if d[-1][1] == 1:                                  # the rows are kept: a COPY
        if len(d) == 1:
            return None                                # a plain copy: a Reshape, not a transpose
        if len(d) == 2:
            d = [(1, 0)] + d
        (D2, s2), (D1, s1), (n, _) = d
        if n % al or s1 % al or s2 % al or n > rs.N_MAX:
            return None
        return ("copy", n, D1, D2, (s1 * elem_bytes, s2 * elem_bytes, 0, 0,
                                    n * elem_bytes, D1 * n * elem_bytes))
    if len(d) == 2:
        d = [(1, 0)] + d
    (B, sB), (L, sL), (M, sM) = d
    if sL != 1 or M % QIMAGE_R or sM % al or sB % al or L > 1024:
        return None
    return ("xpose", L, M // QIMAGE_R, B, (QIMAGE_R * sM * elem_bytes, sB * elem_bytes, sM * elem_bytes,
                                           M * elem_bytes, QIMAGE_R * elem_bytes, L * M * elem_bytes))


def from_onnx_transpose(node, tensors, index: int, align_elems: int, ctx: HostContext, dtype):
    """The scheduled node of a Transpose: TransposeRoNode where the unit can run
    it (an int16 DMA tensor, one of the two shapes above), else the host
    TransposeNode."""
    from .host_nodes import TransposeNode
    host = TransposeNode.from_onnx_node(node, tensors, index, align_elems, ctx)
    x, y = host.inputs[0], host.output
    if not enabled() or dtype.bytes_per_elem != 2 or x.is_host or y.is_host or x.dma or y.dma \
            or x.is_state or y.is_state or host.offset % align_elems:
        return host
    job = transpose_job(host.dims, host.sstr, dtype.bytes_per_elem)
    if job is None:
        return host
    kind, n, rows, blocks, strides = job
    return TransposeRoNode(onnx_node=node, inputs=[x, zero_param_tensor(tensors), tables_tensor(tensors)],
                           output=y, index=index, align_elems=align_elems, F=ctx.frac_bits, kind=kind, n=n,
                           rows=rows, blocks=blocks, a_off=host.offset * dtype.bytes_per_elem, strides=strides,
                           perm=list(host.perm), pz=zero_param_tensor(tensors))


# ------------------------------------------------------------------ #
# The Llama ops (src/llama.py rowops=True; llm_study policy ...+rowops)  #
# ------------------------------------------------------------------ #

def _pblock(tensors: dict, name: str, G, Bq, sh_a, sh_b, sh_c, n: int):
    """The parameter block of n columns as a weight tensor — one entry with
    P_BCAST when every column's fields are equal.  Returns (tensor, bcast)."""
    cols = [np.broadcast_to(np.asarray(v, np.int64), (n,)) for v in (G, Bq, sh_a, sh_b, sh_c)]
    bcast = all((c == c[0]).all() for c in cols)
    if bcast:
        cols = [c[:1] for c in cols]
    return param_tensor(tensors, name, rs.param_block(*cols)), bcast


def _exps(t, F: int) -> np.ndarray:
    return t.exp_channels(F).astype(np.int64)


def _require(cond, node, msg):
    if not cond:
        raise SchedulerError(f"{node.op_type} node '{_label(node)}': {msg}")


def _need_unit(node, what: str):
    _require(enabled(), node, f"{what} needs the row-ops unit (platforms/<platform>.json "
                              f"kernels.rowops.unit, AXI_ROWOPS)")


@dataclass
class ResAddRoNode(RowOpsNode):
    """LlmResAdd on the unit: h1 = sat32(h + d << (fr - f_d)) — the int32 DMA
    residual at 2^-fr plus the int16 delta at its per-channel exponents, exact
    (f_d <= fr), one ADDX job.  inputs [h, d, P, tables]."""
    rows: int = 1
    n:    int = 1
    fr:   int = 15
    bcast: bool = True

    @property
    def opword(self) -> int:
        return rs.opword(rs.OP_ADDX, rs.K_I32, rs.K_I16, rs.K_I32, b_en=True, p_bcast=self.bcast)

    def _strides(self, layouts):
        nd, out = self.onnx_node, self.output
        return (row_stride_bytes(self.inputs[0], layouts, self.n, 4, nd, self.align_elems),
                row_stride_bytes(self.inputs[1], layouts, self.n, 2, nd, self.align_elems),
                row_stride_bytes(out, layouts, self.n, 4, nd, self.align_elems))

    def reference(self, ins, dtype):  # noqa: ARG002
        h = np.asarray(ins[0], np.float64).reshape(self.rows, self.n)
        d = np.asarray(ins[1], np.float64).reshape(self.rows, self.n)
        raw = np.clip(np.rint((h + d) * 2.0 ** self.fr), -2147483648.0, 2147483647.0)
        return (raw / 2.0 ** self.fr).reshape(self.output.shape)

    def kernel_calls(self, layouts: dict) -> list:
        from .perf_calls import KernelCall
        a_s1, b_s1, c_s1 = self._strides(layouts)
        return [KernelCall.of("RowOpsKernel", op=self.opword, n=self.n, rows=self.rows, blocks=1,
                              a_s1=a_s1, b_s1=b_s1, c_s1=c_s1, win_hi=NO_WINDOW)]

    def describe(self) -> str:
        return f"rows={self.rows} n={self.n} on RowOpsKernel (ADDX: int32 residual at 2^-{self.fr} + delta)"

    def emit_comment(self) -> str:
        return (f"    /* [{self.index}] LlmResAdd({self.inputs[0].onnx_name}, {self.inputs[1].onnx_name}) -> "
                f"{self.output.onnx_name}  [{self.rows}][{self.n}] on RowOpsKernel (ADDX, int32 at "
                f"2^-{self.fr}) */")

    def emit_call(self, layouts: dict) -> str:
        a_s1, b_s1, c_s1 = self._strides(layouts)
        return self.call(self.inputs[0].c_name, 0, self.inputs[1].c_name, 0, self.output.c_name, 0,
                         self.inputs[2].c_name, self.opword, self.n, self.rows, 1, a_s1, 0, b_s1, 0,
                         c_s1, 0, 0, 0, 0, 0, NO_WINDOW)


@dataclass
class RMSNormRoNode(RowOpsNode):
    """LlmRMSNorm on the unit: the int32 DMA residual at 2^-fr -> int16 at the
    output's per-channel exponents, ``rowops_spec.rms_norm`` (gamma with sqrt(n)
    folded at 2^-g; the exact sum of squares, the table + Newton rsqrt, one
    rounding).  inputs [h, P, tables]."""
    rows:  int = 1
    n:     int = 1
    fr:    int = 15
    eps:   float = 1e-5
    g:     int = 0
    E:     int = 0
    bcast: bool = False
    gamma: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def opword(self) -> int:
        return rs.opword(rs.OP_NORM, rs.K_I32, rs.K_I16, rs.K_I16, p_bcast=self.bcast)

    def _strides(self, layouts):
        nd = self.onnx_node
        return (row_stride_bytes(self.inputs[0], layouts, self.n, 4, nd, self.align_elems),
                row_stride_bytes(self.output, layouts, self.n, 2, nd, self.align_elems))

    def reference(self, ins, dtype):  # noqa: ARG002
        h = np.rint(np.asarray(ins[0], np.float64).reshape(self.rows, self.n) * 2.0 ** self.fr)
        fo = _exps(self.output, self.F)
        y = rs.rms_norm(h.astype(np.int64), self.fr, self.gamma, self.eps, fo)
        return (y / np.power(2.0, fo.astype(np.float64))[None, :]).reshape(self.output.shape)

    def kernel_calls(self, layouts: dict) -> list:
        from .perf_calls import KernelCall
        a_s1, c_s1 = self._strides(layouts)
        return [KernelCall.of("RowOpsKernel", op=self.opword, n=self.n, rows=self.rows, blocks=1,
                              a_s1=a_s1, c_s1=c_s1, win_hi=NO_WINDOW)]

    def describe(self) -> str:
        return (f"rows={self.rows} n={self.n} eps={self.eps:.3g} on RowOpsKernel (NORM rms; int32 residual at "
                f"2^-{self.fr}, gamma sqrt(n) at 2^-{self.g})")

    def emit_comment(self) -> str:
        return (f"    /* [{self.index}] LlmRMSNorm({self.inputs[0].onnx_name}) -> {self.output.onnx_name}  "
                f"[{self.rows}][{self.n}] on RowOpsKernel (NORM rms, int32 in) */")

    def emit_call(self, layouts: dict) -> str:
        a_s1, c_s1 = self._strides(layouts)
        return self.call(self.inputs[0].c_name, 0, "NULL", 0, self.output.c_name, 0, self.inputs[1].c_name,
                         self.opword, self.n, self.rows, 1, a_s1, 0, 0, 0, c_s1, 0, self.E, 0, 0, 0, NO_WINDOW)


@dataclass
class SiluMulRoNode(RowOpsNode):
    """LlmSiluMul on the unit: a = silu(g) * u with the sigmoid table at 2^-24
    (``rowops_spec.silu_mul``): g int16 at per-channel exponents <= 14, u and a
    int16 at theirs, one MULACT job.  inputs [g, u, P, tables]."""
    rows:  int = 1
    n:     int = 1
    bcast: bool = False

    @property
    def opword(self) -> int:
        return rs.opword(rs.OP_MULACT, rs.K_I16, rs.K_I16, rs.K_I16, b_en=True, p_bcast=self.bcast)

    def _strides(self, layouts):
        nd = self.onnx_node
        return tuple(row_stride_bytes(t, layouts, self.n, 2, nd, self.align_elems)
                     for t in (self.inputs[0], self.inputs[1], self.output))

    def reference(self, ins, dtype):  # noqa: ARG002
        fg, fu, fa = (_exps(t, self.F) for t in (self.inputs[0], self.inputs[1], self.output))
        g = np.rint(np.asarray(ins[0], np.float64).reshape(self.rows, self.n) * np.power(2.0, fg)[None, :])
        u = np.rint(np.asarray(ins[1], np.float64).reshape(self.rows, self.n) * np.power(2.0, fu)[None, :])
        y = rs.silu_mul(g.astype(np.int64), fg, u.astype(np.int64), fu, fa)
        return (y / np.power(2.0, fa.astype(np.float64))[None, :]).reshape(self.output.shape)

    def kernel_calls(self, layouts: dict) -> list:
        from .perf_calls import KernelCall
        a_s1, b_s1, c_s1 = self._strides(layouts)
        return [KernelCall.of("RowOpsKernel", op=self.opword, n=self.n, rows=self.rows, blocks=1,
                              a_s1=a_s1, b_s1=b_s1, c_s1=c_s1, win_hi=NO_WINDOW)]

    def describe(self) -> str:
        fg = sorted(set(int(v) for v in _exps(self.inputs[0], self.F)))
        return f"rows={self.rows} n={self.n} on RowOpsKernel (MULACT: SiLU * up, gate exponents {fg})"

    def emit_comment(self) -> str:
        return (f"    /* [{self.index}] LlmSiluMul({self.inputs[0].onnx_name}, {self.inputs[1].onnx_name}) -> "
                f"{self.output.onnx_name}  [{self.rows}][{self.n}] on RowOpsKernel (MULACT) */")

    def emit_call(self, layouts: dict) -> str:
        a_s1, b_s1, c_s1 = self._strides(layouts)
        return self.call(self.inputs[0].c_name, 0, self.inputs[1].c_name, 0, self.output.c_name, 0,
                         self.inputs[2].c_name, self.opword, self.n, self.rows, 1, a_s1, 0, b_s1, 0,
                         c_s1, 0, 0, 0, 0, 0, NO_WINDOW)


def _rowops_attr(node) -> bool:
    from .host_nodes import _attrs
    return int(_attrs(node).get("rowops", 0)) != 0


def from_llm_resadd(node, tensors, index, align_elems, ctx: HostContext):
    """LlmResAdd: ResAddRoNode when h is the int32 DMA residual, else the host op."""
    from .llm_nodes import LlmResAddNode
    h = tensors.get(node.input[0])
    if h is None or h.dma != "i32":
        return LlmResAddNode.from_onnx_node(node, tensors, index, align_elems, ctx)
    _need_unit(node, "the int32 DMA residual")
    F = ctx.frac_bits
    d, y = tensors[node.input[1]], tensors[node.output[0]]
    n = int(h.shape[-1])
    fr = _exps(h, F)
    _require((fr == fr[0]).all() and y.dma == "i32" and (_exps(y, F) == fr[0]).all(), node,
             "h and the output: the int32 residual at one exponent")
    _require(d.dma is None and d.host is None and not d.is_state and d.shape[-1] == n, node,
             "delta: an int16 DMA tensor of the residual's width")
    _require(d.numel == h.numel == y.numel, node, "shapes")
    fd = _exps(d, F)
    _require((fd <= fr[0]).all() and (fr[0] - fd <= 63).all(), node,
             f"delta exponents {sorted(set(int(v) for v in fd))} beyond the residual's 2^-{int(fr[0])}")
    _require(n % align_elems == 0, node, f"row length {n} % {align_elems}")
    p, bc = _pblock(tensors, f"{y.onnx_name}__rowops_p", 0, 0, 0, fr[0] - fd, 0, n)
    return ResAddRoNode(onnx_node=node, inputs=[h, d, p, tables_tensor(tensors)], output=y, index=index,
                        align_elems=align_elems, F=F, rows=h.numel // n, n=n, fr=int(fr[0]), bcast=bc)


def from_llm_rmsnorm(node, tensors, index, align_elems, ctx: HostContext):
    """LlmRMSNorm: RMSNormRoNode when h is the int32 DMA residual, else the host op."""
    from .host_nodes import _attrs
    from .llm_nodes import LlmRMSNormNode, _const_array
    h = tensors.get(node.input[0])
    if h is None or h.dma != "i32":
        return LlmRMSNormNode.from_onnx_node(node, tensors, index, align_elems, ctx)
    _need_unit(node, "the int32 DMA residual")
    F = ctx.frac_bits
    y = tensors[node.output[0]]
    n = int(h.shape[-1])
    fr = _exps(h, F)
    _require((fr == fr[0]).all(), node, "h: the int32 residual at one exponent")
    _require(y.dma is None and y.host is None and y.numel == h.numel and y.shape[-1] == n, node,
             "output: an int16 DMA tensor of h's shape")
    gamma = _const_array(ctx, tensors, node.input[1], node, "gamma").astype(np.float64).reshape(-1)
    _require(gamma.size == n, node, "gamma size")
    a = _attrs(node)
    eps = float(a["axi_eps"].decode() if isinstance(a.get("axi_eps"), bytes)
                else a.get("axi_eps", a.get("epsilon", 1e-5)))
    G, g, _bq, E = rs.norm_params(gamma, n, eps, int(fr[0]), rms=True)
    fo = _exps(y, F)
    sh_c = rs.P1_BITS + g - fo
    _require((sh_c >= 0).all() and (sh_c <= 63).all(), node, f"gamma's exponent 2^-{g} beyond the shifts")
    _require(0 <= E < 2 ** 63, node, f"eps {eps} beyond the eps register")
    _require(n % align_elems == 0, node, f"row length {n} % {align_elems}")
    p, bc = _pblock(tensors, f"{y.onnx_name}__rowops_p", G, 0, 0, 0, sh_c, n)
    return RMSNormRoNode(onnx_node=node, inputs=[h, p, tables_tensor(tensors)], output=y, index=index,
                         align_elems=align_elems, F=F, rows=h.numel // n, n=n, fr=int(fr[0]), eps=eps,
                         g=int(g), E=int(E), bcast=bc, gamma=gamma)


def from_llm_silu_mul(node, tensors, index, align_elems, ctx: HostContext):
    """LlmSiluMul: SiluMulRoNode with the attribute rowops = 1, else the host op."""
    from .llm_nodes import LlmSiluMulNode
    if not _rowops_attr(node):
        return LlmSiluMulNode.from_onnx_node(node, tensors, index, align_elems, ctx)
    _need_unit(node, "rowops = 1")
    F = ctx.frac_bits
    g, u = tensors[node.input[0]], tensors[node.input[1]]
    y = tensors[node.output[0]]
    n = int(g.shape[-1])
    for t, what in ((g, "gate"), (u, "up"), (y, "output")):
        _require(t.dma is None and t.host is None and not t.is_state and t.shape[-1] == n, node,
                 f"{what}: an int16 DMA tensor of the gate's width")
    _require(g.numel == u.numel == y.numel, node, "shapes")
    fg, fu, fa = (_exps(t, F) for t in (g, u, y))
    _require((fg <= rs.G_BITS).all() and (rs.G_BITS - fg <= 63).all(), node,
             f"gate exponents {sorted(set(int(v) for v in fg))} above {rs.G_BITS}")
    sh_c = rs.SILU_BITS + fu - fa
    _require((sh_c >= 0).all() and (sh_c <= 63).all(), node, "up / output exponents beyond the shifts")
    _require(n % align_elems == 0, node, f"row length {n} % {align_elems}")
    p, bc = _pblock(tensors, f"{y.onnx_name}__rowops_p", 0, 0, rs.G_BITS - fg, 0, sh_c, n)
    return SiluMulRoNode(onnx_node=node, inputs=[g, u, p, tables_tensor(tensors)], output=y, index=index,
                         align_elems=align_elems, F=F, rows=g.numel // n, n=n, bcast=bc)


# ------------------------------------------------------------------ #
# LlmAttnPrep of the prefill entries (src/llama.py rowops=True; the      #
# policy's irope): RoPE, the caches, the q image — HOSTOPS_PLAN §10      #
# ------------------------------------------------------------------ #

ROPE_TABLE_PREFIX = "rowops_rope"
ZERO_P_NAME = "rowops_p_zero"
QIMAGE_R = 8                      # the q image's TRANSPOSE tiles: 8 query rows -> 16-byte granules


def rope_raw_tensor(tensors: dict, cos: np.ndarray, sin: np.ndarray) -> TensorInfo:
    """The cos / sin table the ROPE jobs stream as B: one row per position of
    2·HD int32 — cos[0..HD) then sin[0..HD) as int24 at 2^-23
    (``rowops_spec.rope_tables_raw`` of the float32 [C][HD/2] tables repeated
    over both halves) — as one weight tensor per table (the int32 words as
    little-endian int16 pairs), created on first use."""
    import zlib
    cos = np.asarray(cos, np.float32)
    sin = np.asarray(sin, np.float32)
    tag = zlib.crc32(np.ascontiguousarray(cos).tobytes() + np.ascontiguousarray(sin).tobytes())
    name = f"{ROPE_TABLE_PREFIX}_{tag:08x}_{cos.shape[0]}x{cos.shape[1]}"
    t = tensors.get(name)
    if t is None:
        cr, sr = rs.rope_tables_raw(np.concatenate([cos, cos], axis=1).astype(np.float64),
                                    np.concatenate([sin, sin], axis=1).astype(np.float64))
        rows = np.ascontiguousarray(np.concatenate([cr, sr], axis=1).astype(np.int32))   # [C][2 HD]
        words = rows.reshape(-1).view(np.int16).astype(np.int64)
        t = TensorInfo(onnx_name=name, shape=[int(words.size)], dtype="float32",
                       data=raw_as_data(words))
        tensors[name] = t
    return t


def zero_param_tensor(tensors: dict) -> TensorInfo:
    """One all-zero P_BCAST entry: the plain copies and the zero fill."""
    t = tensors.get(ZERO_P_NAME)
    if t is None:
        t = param_tensor(tensors, ZERO_P_NAME, rs.param_block(0, 0, 0, 0, 0))
    return t


def _vjob(pos: int, k: int, rows: int, blocks: int, HD: int, K: int, stride: int) -> tuple:
    c = (((k // (16 * K)) * 16 + k % 16) * HD * K + (k // 16) % K) * 2
    return ((k - pos) * stride, c, rows, blocks, stride, 16 * stride, 0, 2 * K, HD * K * 2, 2, 1)


def vimage_jobs(pos: int, n: int, HD: int, K: int, stride: int) -> list:
    """The TRANSPOSE jobs that write the V rows [pos, pos + n) of one KV group
    into the P.V image interleaved by K (``llm_nodes.v_row_base``) from staging
    rows ``stride`` bytes apart (row t = cache row pos + t): the 16K-row blocks
    wholly inside [pos, pos + n) as R = K tiles with contiguous 2K-byte
    granules, the rows outside them (the two boundary blocks) as R = 1
    granules — 2 bytes, byte-strobed, so the block's other keys keep their
    values — one job per run of complete 16-row groups and per partial group.
    The C twin is ``llm_ro_vimage``; this one prices the calls.  Returns
    [(a_off, c_off, rows, blocks, a_s1, a_s2, b_s1, b_s2, c_s1, c_s2, R)],
    the byte offsets relative to the group's staging / cache base."""
    B = 16 * K
    end = pos + n
    jobs: list = []
    if n <= 0:
        return jobs

    def edge(k0, k1):
        k = k0
        while k < k1:
            bend = min(k // B * B + B, k1)
            r = k
            if r % 16:
                cnt = min(16 - r % 16, bend - r)
                jobs.append(_vjob(pos, r, cnt, 1, HD, K, stride))
                r += cnt
            cg = (bend - r) // 16
            if cg:
                jobs.append(_vjob(pos, r, 16, cg, HD, K, stride))
                r += 16 * cg
            if r < bend:
                jobs.append(_vjob(pos, r, bend - r, 1, HD, K, stride))
            k = bend
    b0 = -(-pos // B) * B
    b1 = end // B * B
    if b1 > b0:
        jobs.append(((b0 - pos) * stride, b0 * HD * 2, 16, (b1 - b0) // B, stride, B * stride,
                     16 * stride, 2 * K, HD * K * 2, B * HD * 2, K))
        edge(pos, b0)
        edge(b1, end)
    else:
        edge(pos, end)
    return jobs


# The C twin of vimage_jobs (emitted after run_rowops when a project has an
# AttnPrepRoNode, src/codegen/_source.py): every job but the first waits for
# the one before it (one engine), the last is left in flight for the
# schedule's own kernel_wait.
ATTN_PREP_C = r"""
/* ---- LlmAttnPrep on the row-ops unit (src/rowops_nodes.py AttnPrepRoNode,
 * doc/plans/HOSTOPS_PLAN.md §10): the V rows [pos, pos + n) of every KV group
 * into the P.V image interleaved by K (llm_vrow) — the 16K-row blocks wholly
 * inside as R = K TRANSPOSE tiles (contiguous granules), the rows of the two
 * boundary blocks as R = 1 granules (byte-strobed: the block's other keys keep
 * their values), per run of complete 16-row groups / partial group.  The
 * Python twin, rowops_nodes.vimage_jobs, prices the calls. */
static unsigned llm_ro_clamp_n(unsigned pos, unsigned n, unsigned T, unsigned C)
{
    if (n > T) n = T;
    if (pos >= C) return 0u;
    return n > C - pos ? C - pos : n;
}

static void llm_ro_vjob(inference_buf_t *src, unsigned stride, inference_buf_t *cache, unsigned a_g,
                        unsigned c_g, unsigned pos, unsigned k, unsigned rows, unsigned blocks,
                        unsigned HD, unsigned K, int *issued)
{
    const unsigned c = (((k / (16u * K)) * 16u + k % 16u) * HD * K + (k / 16u) % K) * 2u;
    if (*issued) kernel_wait(KERNEL_ROWOPS);
    *issued = 1;
    run_rowops(src, a_g + (k - pos) * stride, NULL, 0u, cache, c_g + c, NULL,
               0x00007u, HD, rows, blocks, stride, 16u * stride, 0u, 2u * K, HD * K * 2u, 2u,
               0ull, 1u, 0u, 0u, 0xFFFFFFFFu);
}

static void llm_ro_vedge(inference_buf_t *src, unsigned stride, inference_buf_t *cache, unsigned a_g,
                         unsigned c_g, unsigned pos, unsigned k0, unsigned k1, unsigned HD,
                         unsigned K, int *issued)
{
    const unsigned B = 16u * K;
    unsigned k = k0;
    while (k < k1) {
        unsigned bend = k / B * B + B, r = k, cg;
        if (bend > k1) bend = k1;
        if (r % 16u) {
            unsigned cnt = 16u - r % 16u;
            if (cnt > bend - r) cnt = bend - r;
            llm_ro_vjob(src, stride, cache, a_g, c_g, pos, r, cnt, 1u, HD, K, issued);
            r += cnt;
        }
        cg = (bend - r) / 16u;
        if (cg) {
            llm_ro_vjob(src, stride, cache, a_g, c_g, pos, r, 16u, cg, HD, K, issued);
            r += 16u * cg;
        }
        if (r < bend)
            llm_ro_vjob(src, stride, cache, a_g, c_g, pos, r, bend - r, 1u, HD, K, issued);
        k = bend;
    }
}

static void llm_ro_vimage(inference_buf_t *src, unsigned stride, inference_buf_t *cache, unsigned pos,
                          unsigned n, unsigned KV, unsigned C, unsigned HD, unsigned K)
{
    const unsigned B = 16u * K, end = pos + n, b0 = (pos + B - 1u) / B * B, b1 = end / B * B;
    unsigned g;
    int      issued = 0;
    if (!n) return;
    for (g = 0u; g < KV; g++) {
        const unsigned a_g = g * HD * 2u, c_g = g * C * HD * 2u;
        if (b1 > b0) {
            if (issued) kernel_wait(KERNEL_ROWOPS);
            issued = 1;
            run_rowops(src, a_g + (b0 - pos) * stride, NULL, 0u, cache, c_g + b0 * HD * 2u, NULL,
                       0x00007u, HD, 16u, (b1 - b0) / B, stride, B * stride, 16u * stride, 2u * K,
                       HD * K * 2u, B * HD * 2u, 0ull, K, 0u, 0u, 0xFFFFFFFFu);
            llm_ro_vedge(src, stride, cache, a_g, c_g, pos, pos, b0, HD, K, &issued);
            llm_ro_vedge(src, stride, cache, a_g, c_g, pos, b1, end, HD, K, &issued);
        } else {
            llm_ro_vedge(src, stride, cache, a_g, c_g, pos, pos, end, HD, K, &issued);
        }
    }
}
"""


@dataclass
class AttnPrepRoNode(RowOpsNode):
    """LlmAttnPrep of a prefill entry on the unit (doc/plans/HOSTOPS_PLAN.md
    §10): the jobs, in order —

      * (the first layer only) a COPY with the window [0, 0) zeroes the q
        staging rows ``qs`` [T][H·HD], so the rows t >= n of this run stay 0;
      * ROPE of q0 -> ``qs`` (rows < n), rounded at the per-head q exponent
        (``rowops_spec.rope``: the policy's irope), the cos / sin row of
        position pos + t streamed as B from ``rope_raw_tensor``;
      * ROPE of k0 -> ``ks`` [T][KV·HD] at the K cache's per-head exponent;
      * (where the V cache exponent differs from v's) a COPY re-rounds v ->
        ``vs`` [T][KV·HD] — the host's round-half-even, bit for bit;
      * per KV group a TRANSPOSE of ``qs`` (tiles of 8 query rows x HD) into
        the group's q.K^T x image qx (kw = 1: x_g[d][h'·T + t]), all T rows;
      * a COPY of ``ks`` into the K cache rows [pos, pos + n), group-major
        (``blocks = KV``);
      * the V image: K = 1 a COPY like the K cache's, else ``vimage_jobs`` /
        ``llm_ro_vimage`` (the boundary blocks as R = 1 granules).

    ``rows = n`` (clamped as LlmAttention) and the B offset pos·8·HD are C
    expressions; the calls are priced at pos = 0, n = T.  The staging rows
    are DMA states of the entry.  The reference writes the same bytes as the
    host node (the irope numerics): the cache rows [pos, pos + n), the image
    with zero columns past n.  Decode steps (T = 1) keep the host node."""
    T:   int = 1
    H:   int = 1
    KV:  int = 1
    HD:  int = 16
    C:   int = 16
    Q:   int = 16
    K:   int = 1
    first: bool = False
    v_direct: bool = False
    pq_bc: bool = False
    pk_bc: bool = False
    pv_bc: bool = True
    fq:   Optional[np.ndarray] = field(default=None, repr=False)      # [H]
    cos:  Optional[np.ndarray] = field(default=None, repr=False)      # float32 [C][HD/2]
    sin:  Optional[np.ndarray] = field(default=None, repr=False)
    q0:   object = field(default=None, repr=False)
    k0:   object = field(default=None, repr=False)
    v:    object = field(default=None, repr=False)
    pos:  object = field(default=None, repr=False)
    n_t:  object = field(default=None, repr=False)
    ck:   object = field(default=None, repr=False)
    cv:   object = field(default=None, repr=False)
    qs:   object = field(default=None, repr=False)
    ks:   object = field(default=None, repr=False)
    vs:   object = field(default=None, repr=False)
    raw:  object = field(default=None, repr=False)
    pq:   object = field(default=None, repr=False)
    pk:   object = field(default=None, repr=False)
    pvb:  object = field(default=None, repr=False)
    pz:   object = field(default=None, repr=False)

    # ---- geometry ------------------------------------------------------ #
    @property
    def G(self) -> int:
        return self.H // self.KV

    @property
    def nq(self) -> int:
        return self.H * self.HD

    @property
    def nk(self) -> int:
        return self.KV * self.HD

    def state_writes(self):
        return [self.ck, self.cv, self.qs, self.ks] + ([self.vs] if self.vs is not None else [])

    def state_updates(self):
        return self.state_writes()

    @property
    def calls_env(self) -> dict:
        """The runtime names of the emitted calls at the priced geometry."""
        return {"_pos": 0, "_n": self.T}

    def _strides(self, layouts):
        nd, al = self.onnx_node, self.align_elems
        sq = row_stride_bytes(self.q0, layouts, self.nq, 2, nd, al)
        sk = row_stride_bytes(self.k0, layouts, self.nk, 2, nd, al)
        sv = row_stride_bytes(self.v, layouts, self.nk, 2, nd, al)
        return sq, sk, sv

    @property
    def op_zero(self) -> int:
        return rs.opword(rs.OP_COPY, p_bcast=True)

    @property
    def op_rope(self) -> int:
        return rs.opword(rs.OP_ROPE, rs.K_I16, rs.K_I32, rs.K_I16, b_en=True)

    @property
    def op_xpose(self) -> int:
        return rs.opword(rs.OP_XPOSE)

    def _jobs(self, layouts, pos: int, n):
        """Every job as (comment, a, a_off, b, b_off, c, c_off, p, op, n, rows, blocks,
        a_s1, a_s2, b_s1, b_s2, c_s1, c_s2, aux, win_lo, win_hi, hidden) with ints or C
        expressions; ``hidden`` marks the V-image jobs llm_ro_vimage issues."""
        sq, sk, sv = self._strides(layouts)
        T, KV, HD, C, K, G = self.T, self.KV, self.HD, self.C, self.K, self.G
        sqs, sks = self.nq * 2, self.nk * 2
        NW = NO_WINDOW
        bo = f"{pos} * {8 * HD}u" if isinstance(pos, str) else pos * 8 * HD
        jobs = []
        if self.first:
            jobs.append(("the q staging zeroed: rows past n stay 0", self.qs, 0, None, 0, self.qs, 0,
                         self.pz, self.op_zero, self.nq, T, 1, sqs, 0, 0, 0, sqs, 0, 0, 0, 0, False))
        jobs.append(("RoPE(q0) at f_q[h] -> qs (rows < n; B = cos / sin of pos + t)", self.q0, 0,
                     self.raw, bo, self.qs, 0, self.pq, self.op_rope | (rs.F_P_BCAST if self.pq_bc else 0),
                     self.nq, n, 1, sq, 0, 8 * HD, 0, sqs, 0, HD, 0, NW, False))
        jobs.append(("RoPE(k0) at f_k[g] -> ks", self.k0, 0, self.raw, bo, self.ks, 0, self.pk,
                     self.op_rope | (rs.F_P_BCAST if self.pk_bc else 0), self.nk, n, 1, sk, 0, 8 * HD, 0,
                     sks, 0, HD, 0, NW, False))
        if self.v_direct:
            vsrc, svs = self.v, sv
        else:
            vsrc, svs = self.vs, sks
            jobs.append(("v re-rounded at the V cache's exponents -> vs", self.v, 0, None, 0, self.vs, 0,
                         self.pvb, rs.opword(rs.OP_COPY, p_bcast=self.pv_bc), self.nk, n, 1, sv, 0, 0, 0,
                         sks, 0, 0, 0, NW, False))
        for g in range(KV):
            jobs.append((f"the q.K^T x image of group {g}: qs -> qx (8-row tiles)", self.qs, g * G * HD * 2,
                         None, 0, self.output, g * HD * G * T * 2, None, self.op_xpose, HD, T // QIMAGE_R, G,
                         QIMAGE_R * sqs, HD * 2, sqs, 2 * G * T, 2 * QIMAGE_R, 2 * T, QIMAGE_R, 0, NW, False))
        co = f"{pos} * {HD * 2}u" if isinstance(pos, str) else pos * HD * 2
        jobs.append(("ks -> the K cache rows [pos, pos + n), group-major", self.ks, 0, None, 0, self.ck, co,
                     self.pz, self.op_zero, HD, n, KV, sks, HD * 2, 0, 0, HD * 2, C * HD * 2, 0, 0, NW, False))
        if K == 1:
            jobs.append(("v -> the V cache rows [pos, pos + n), group-major", vsrc, 0, None, 0, self.cv, co,
                         self.pz, self.op_zero, HD, n, KV, svs, HD * 2, 0, 0, HD * 2, C * HD * 2, 0, 0, NW,
                         False))
        else:
            for a_off, c_off, rows, blocks, a_s1, a_s2, b_s1, b_s2, c_s1, c_s2, R in \
                    vimage_jobs(int(pos) if not isinstance(pos, str) else 0,
                                int(n) if not isinstance(n, str) else self.T, HD, K, svs):
                for g in range(KV):
                    jobs.append(("the V image (llm_ro_vimage)", vsrc, g * HD * 2 + a_off, None, 0, self.cv,
                                 g * C * HD * 2 + c_off, None, self.op_xpose, HD, rows, blocks, a_s1, a_s2,
                                 b_s1, b_s2, c_s1, c_s2, R, 0, NW, True))
        return jobs, vsrc, svs

    def kernel_calls(self, layouts: dict) -> list:
        from .perf_calls import KernelCall, merge
        jobs, _, _ = self._jobs(layouts, 0, self.T)
        return merge([KernelCall.of("RowOpsKernel", op=j[8], n=j[9], rows=j[10], blocks=j[11], a_s1=j[12],
                                    a_s2=j[13], b_s1=j[14], b_s2=j[15], c_s1=j[16], c_s2=j[17], aux=j[18],
                                    win_lo=j[19], win_hi=j[20]) for j in jobs])

    def hidden_calls(self, layouts: dict) -> list:
        """The calls a C helper issues (not in emit_call's text): the V image's."""
        from .perf_calls import KernelCall
        jobs, _, _ = self._jobs(layouts, 0, self.T)
        return [KernelCall.of("RowOpsKernel", op=j[8], n=j[9], rows=j[10], blocks=j[11], a_s1=j[12],
                              a_s2=j[13], b_s1=j[14], b_s2=j[15], c_s1=j[16], c_s2=j[17], aux=j[18],
                              win_lo=j[19], win_hi=j[20]) for j in jobs if j[21]]

    def describe(self) -> str:
        return (f"T={self.T} heads {self.H}/{self.KV} head_dim {self.HD}, cache {self.C} rows on RowOpsKernel "
                f"(ROPE q / k, the q image kw=1, the caches; V interleave {self.K}"
                f"{', v copied as is' if self.v_direct else ''})")

    def emit_comment(self) -> str:
        return (f"    /* [{self.index}] LlmAttnPrep({self.q0.onnx_name}, {self.k0.onnx_name}, {self.v.onnx_name}) "
                f"-> {self.output.onnx_name} + the caches  on RowOpsKernel: {self.describe()} */")

    def emit_call(self, layouts: dict) -> str:
        jobs, vsrc, svs = self._jobs(layouts, "_pos", "_n")
        cn = lambda t: "NULL" if t is None else t.c_name   # noqa: E731
        lines = ["    {",
                 f"        const unsigned _pos = (unsigned){self.pos.c_name}[0];",
                 f"        const unsigned _n = llm_ro_clamp_n(_pos, (unsigned){self.n_t.c_name}[0], "
                 f"{self.T}u, {self.C}u);"]
        issued = False
        for j in jobs:
            if j[21]:
                continue
            if issued:
                lines.append("        kernel_wait(KERNEL_ROWOPS);")
            issued = True
            lines.append(f"        /* {j[0]} */")
            lines.append(self.call(cn(j[1]), j[2], cn(j[3]), j[4], cn(j[5]), j[6], cn(j[7]), j[8], j[9], j[10],
                                   j[11], j[12], j[13], j[14], j[15], j[16], j[17], 0, j[18], 0, j[19], j[20],
                                   indent="        "))
        if self.K > 1:
            lines.append("        kernel_wait(KERNEL_ROWOPS);")
            lines.append("        /* the V image: the whole 16K-row blocks as R = K tiles, the boundary rows as "
                         "R = 1 granules (llm_ro_vimage) */")
            lines.append(f"        llm_ro_vimage({vsrc.c_name}, {svs}u, {self.cv.c_name}, _pos, _n, {self.KV}u, "
                         f"{self.C}u, {self.HD}u, {self.K}u);")
        lines.append("    }")
        return "\n".join(lines)

    # ---- simulation ---------------------------------------------------- #
    def _raw_tables(self):
        if not hasattr(self, "_raw_cs"):
            cr, sr = rs.rope_tables_raw(np.concatenate([self.cos, self.cos], 1).astype(np.float64),
                                        np.concatenate([self.sin, self.sin], 1).astype(np.float64))
            self._raw_cs = (cr, sr)
        return self._raw_cs

    def reference(self, ins, dtype):  # noqa: ARG002
        from .llm_nodes import _i32, attn_keys
        a = {t.onnx_name: x for t, x in zip(self.inputs, ins, strict=True)}
        T, H, KV, HD, C, F = self.T, self.H, self.KV, self.HD, self.C, self.F
        G = H // KV
        pos = _i32(a[self.pos.onnx_name])
        n, _keys = attn_keys(pos, _i32(a[self.n_t.onnx_name]), T, C, self.Q)
        ck, cv = a[self.ck.onnx_name], a[self.cv.onnx_name]            # logical state arrays, in place
        qx = np.zeros((KV, HD, G * T))
        if n > 0:
            cr, sr = self._raw_tables()
            fq0, fk0, fv = _exps(self.q0, F), _exps(self.k0, F), _exps(self.v, F)
            fkc, fvc = _exps(self.ck, F), _exps(self.cv, F)
            p2 = lambda e: np.power(2.0, np.asarray(e, np.float64))   # noqa: E731
            q0 = np.asarray(a[self.q0.onnx_name], np.float64).reshape(T, H * HD)[:n]
            k0 = np.asarray(a[self.k0.onnx_name], np.float64).reshape(T, KV * HD)[:n]
            v = np.asarray(a[self.v.onnx_name], np.float64).reshape(T, KV * HD)[:n]
            xq = np.rint(q0 * p2(fq0)[None, :]).astype(np.int64).reshape(n, H, HD)
            xk = np.rint(k0 * p2(fk0)[None, :]).astype(np.int64).reshape(n, KV, HD)
            yq = rs.rope(xq, fq0.reshape(H, HD), cr[pos:pos + n], sr[pos:pos + n], self.fq)
            yk = rs.rope(xk, fk0.reshape(KV, HD), cr[pos:pos + n], sr[pos:pos + n], fkc.reshape(KV, HD))
            ck[pos:pos + n] = yk.reshape(n, -1) / p2(fkc)[None, :]
            vr = np.rint(v * p2(fv)[None, :]).astype(np.int64)
            sa = np.maximum(fvc - fv, 0)
            sc = np.maximum(fv - fvc, 0)
            cv[pos:pos + n] = rs.sat16(rs.rne_shr_vec(vr << sa[None, :], sc[None, :])) / p2(fvc)[None, :]
            for h in range(H):
                g, hh = divmod(h, G)
                qx[g, :, hh * T: hh * T + n] = yq[:, h, :].T
        return qx


def from_llm_attn_prep(node, tensors, index, align_elems, ctx: HostContext):
    """LlmAttnPrep: AttnPrepRoNode with the attribute rowops = 1 (the prefill
    entries of src/llama.py rowops=True), else the host op."""
    from .llm_nodes import (KEY_QUANTUM, LlmAttnPrepNode, _attr_ints, _const_array, _heads_attrs,
                            _llm_inputs, _resolve)
    if not _rowops_attr(node):
        return LlmAttnPrepNode.from_onnx_node(node, tensors, index, align_elems, ctx)
    _need_unit(node, "rowops = 1")
    F = ctx.frac_bits
    ins = _llm_inputs(node, tensors)
    _require(len(ins) == 12, node, "inputs: q0, k0, v, pos, n, cache_k, cache_v, cos, sin, qs, ks, vs "
                                   "(may be empty)")
    q0, k0, v, pos, n, ck, cv, _c, _s, qs, ks, vs = ins
    _require(n is not None, node, "the unit's attention prep needs n (a prefill entry)")
    y = _resolve(tensors, node.output[0], node)
    a, H, KV, HD = _heads_attrs(node)
    G = H // KV
    T = q0.numel // (H * HD)
    C = int(ck.shape[0])
    K = int(cv.group_kw)
    Q = int(a.get("key_quantum", KEY_QUANTUM))
    _require(int(a.get("qk_kw", 1)) == 1, node, "the unit writes the q image with qk_kw = 1")
    _require(HD % 16 == 0 and 16 <= HD <= 128, node, f"head_dim {HD}: 16 .. 128, a multiple of 16")
    _require(H * HD <= rs.N_MAX and KV * HD <= rs.N_MAX, node, f"rows of {H * HD} / {KV * HD} beyond {rs.N_MAX}")
    _require(T % QIMAGE_R == 0, node, f"prefill rows {T} % {QIMAGE_R} (the q image's tiles)")
    _require(C % Q == 0 and Q % (16 * K) == 0, node, "cache rows % key_quantum, key_quantum % 16 x the V interleave")
    _require(q0.numel == T * H * HD and k0.numel == T * KV * HD and v.numel == k0.numel, node, "q / k / v shapes")
    _require(list(y.shape) == [KV, HD, G * T], node, f"qx must be [{KV}][{HD}][{G * T}]")
    for t, w in ((q0, "q0"), (k0, "k0"), (v, "v"), (y, "qx")):
        _require(t.dma is None and t.host is None and not t.is_state, node, f"{w}: an int16 DMA tensor")
    _require(pos.host == "i32" and n.host == "i32", node, "pos / n: int32 host tensors")
    for t, w in ((ck, "cache_k"), (cv, "cache_v")):
        _require(t.is_state and not t.is_host and t.group_layout == (KV, HD), node,
                 f"{w} must be a DMA state stored group-major [{KV}][{C}][{HD}]")
    _require(ck.group_kw == 1, node, "the K cache rows must not be interleaved")
    for t, w, cols in ((qs, "qs", H * HD), (ks, "ks", KV * HD)):
        _require(t is not None and t.is_state and not t.is_host and t.dma is None
                 and list(t.shape) == [T, cols], node, f"{w}: an int16 DMA state [{T}][{cols}]")
    cos = _const_array(ctx, tensors, node.input[7], node, "cos").astype(np.float32)
    sin = _const_array(ctx, tensors, node.input[8], node, "sin").astype(np.float32)
    _require(cos.shape == (C, HD // 2) and sin.shape == cos.shape, node, "cos / sin shape")
    fq = np.asarray(_attr_ints(a, "q_exp", node, H), np.int64)
    # the parameter blocks: ROPE's shifts per column (rowops_spec.rope), v's re-round
    def rope_params(t, nh, f_out):
        fx = _exps(t, F).reshape(nh, HD)
        half = HD // 2
        frot = np.concatenate([fx[:, half:], fx[:, :half]], axis=1)
        fp = np.maximum(fx, frot)
        fo = np.broadcast_to(np.asarray(f_out, np.int64).reshape(nh, -1), (nh, HD))
        sa, sb, sc = fp - fx, fp - frot, fp + rs.ROPE_BITS - fo
        _require((sa >= 0).all() and (sa <= 63).all() and (sb <= 63).all() and (sc >= 0).all()
                 and (sc <= 63).all(), node, "RoPE exponents beyond the shifts")
        return sa.reshape(-1), sb.reshape(-1), sc.reshape(-1)
    fkc = _exps(ck, F).reshape(KV, HD)
    _require((fkc == fkc[:, :1]).all(), node, "the K cache's exponent must be one per KV head")
    sa, sb, sc = rope_params(q0, H, fq)
    pq, pq_bc = _pblock(tensors, f"{y.onnx_name}__rowops_pq", 0, 0, sa, sb, sc, H * HD)
    sa, sb, sc = rope_params(k0, KV, fkc[:, 0])
    pk, pk_bc = _pblock(tensors, f"{y.onnx_name}__rowops_pk", 0, 0, sa, sb, sc, KV * HD)
    fv, fvc = _exps(v, F), _exps(cv, F)
    v_direct = bool((fv == fvc).all())
    pvb, pv_bc = None, True
    if not v_direct:
        _require(vs is not None and vs.is_state and not vs.is_host and vs.dma is None
                 and list(vs.shape) == [T, KV * HD], node, f"vs: an int16 DMA state [{T}][{KV * HD}] "
                 "(the V cache exponents differ from v's)")
        sa_v, sc_v = np.maximum(fvc - fv, 0), np.maximum(fv - fvc, 0)
        _require((sa_v <= 63).all() and (sc_v <= 63).all(), node, "v / V cache exponents beyond the shifts")
        pvb, pv_bc = _pblock(tensors, f"{y.onnx_name}__rowops_pv", 0, 0, sa_v, 0, sc_v, KV * HD)
    else:
        vs = None
    raw = rope_raw_tensor(tensors, cos, sin)
    pz = zero_param_tensor(tensors)
    tb = tables_tensor(tensors)
    inputs = [q0, k0, v, pos, n, ck, cv, qs, ks] + ([vs] if vs is not None else []) + [raw, pq, pk] \
        + ([pvb] if pvb is not None else []) + [pz, tb]
    return AttnPrepRoNode(onnx_node=node, inputs=inputs, output=y, index=index, align_elems=align_elems, F=F,
                          T=T, H=H, KV=KV, HD=HD, C=C, Q=Q, K=K, first=int(a.get("first", 0)) != 0,
                          v_direct=v_direct, pq_bc=pq_bc, pk_bc=pk_bc, pv_bc=pv_bc, fq=fq, cos=cos, sin=sin,
                          q0=q0, k0=k0, v=v, pos=pos, n_t=n, ck=ck, cv=cv, qs=qs, ks=ks, vs=vs, raw=raw,
                          pq=pq, pk=pk, pvb=pvb, pz=pz)


# ------------------------------------------------------------------ #
# The vision encoder's ops (src/vit.py rowops=True; vlm_study ...+rowops)   #
# ------------------------------------------------------------------ #

@dataclass
class ResAddBiasRoNode(ResAddRoNode):
    """A residual add with a per-channel bias (VitResAdd; the TTS encoder's
    sum): h1 = sat32(h + (d << (fr - f_d)) + Bq' << (fr - bb)) with Bq' =
    round(bias * 2^bb), bb = rowops_spec.bias_bits(bias, fr) — the bias rounded
    once.  A i16 or i32 at fr (sh_a = fr - f_a)."""
    bb:    int = 15
    fa:    int = 15                                 # A's exponent (i16 A: fr - fa shifts it up)
    a32:   bool = True
    bias:  Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def opword(self) -> int:
        return rs.opword(rs.OP_ADDX, rs.K_I32 if self.a32 else rs.K_I16, rs.K_I16, rs.K_I32, b_en=True,
                         p_bcast=self.bcast)

    def _strides(self, layouts):
        nd, out = self.onnx_node, self.output
        return (row_stride_bytes(self.inputs[0], layouts, self.n, 4 if self.a32 else 2, nd, self.align_elems),
                row_stride_bytes(self.inputs[1], layouts, self.n, 2, nd, self.align_elems),
                row_stride_bytes(out, layouts, self.n, 4, nd, self.align_elems))

    def reference(self, ins, dtype):  # noqa: ARG002
        h = np.asarray(ins[0], np.float64).reshape(self.rows, self.n)
        d = np.asarray(ins[1], np.float64).reshape(self.rows, self.n)
        qb = rs.quant_bias(self.bias, self.fr)[None, :]
        raw = np.clip(np.rint((h + d + qb) * 2.0 ** self.fr), -2147483648.0, 2147483647.0)
        return (raw / 2.0 ** self.fr).reshape(self.output.shape)

    def describe(self) -> str:
        return (f"rows={self.rows} n={self.n} on RowOpsKernel (ADDX: {'int32' if self.a32 else 'int16'} "
                f"+ delta + bias at 2^-{self.bb} -> int32 at 2^-{self.fr})")

    def emit_comment(self) -> str:
        return (f"    /* [{self.index}] {self.onnx_node.op_type}({self.inputs[0].onnx_name}, "
                f"{self.inputs[1].onnx_name}) -> {self.output.onnx_name}  [{self.rows}][{self.n}] on "
                f"RowOpsKernel (ADDX + bias, int32 at 2^-{self.fr}) */")

    def emit_call(self, layouts: dict) -> str:
        a_s1, b_s1, c_s1 = self._strides(layouts)
        return self.call(self.inputs[0].c_name, 0, self.inputs[1].c_name, 0, self.output.c_name, 0,
                         self.inputs[2].c_name, self.opword, self.n, self.rows, 1, a_s1, 0, b_s1, 0,
                         c_s1, 0, 0, 0, self.fr - self.bb, 0, NO_WINDOW)


@dataclass
class LayerNorm32RoNode(RowOpsNode):
    """A LayerNorm of the int32 DMA residual at 2^-fr -> int16 at the output's
    per-channel exponents (VitLayerNorm, the TTS encoder's): rowops_spec.layer_norm
    with gamma at 2^-g and beta as Bq' at 2^-bb.  inputs [h, P, tables]."""
    rows:  int = 1
    n:     int = 1
    fr:    int = 15
    eps:   float = 1e-6
    g:     int = 0
    bb:    int = rs.BQ_BITS
    E:     int = 0
    bcast: bool = False
    gamma: Optional[np.ndarray] = field(default=None, repr=False)
    beta:  Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def opword(self) -> int:
        return rs.opword(rs.OP_NORM, rs.K_I32, rs.K_I16, rs.K_I16, p_bcast=self.bcast, ln=True)

    def _strides(self, layouts):
        nd = self.onnx_node
        return (row_stride_bytes(self.inputs[0], layouts, self.n, 4, nd, self.align_elems),
                row_stride_bytes(self.output, layouts, self.n, 2, nd, self.align_elems))

    def reference(self, ins, dtype):  # noqa: ARG002
        h = np.rint(np.asarray(ins[0], np.float64).reshape(self.rows, self.n) * 2.0 ** self.fr)
        fo = _exps(self.output, self.F)
        y = rs.layer_norm(h.astype(np.int64), self.fr, self.gamma, self.beta, self.eps, fo)
        return (y / np.power(2.0, fo.astype(np.float64))[None, :]).reshape(self.output.shape)

    def kernel_calls(self, layouts: dict) -> list:
        from .perf_calls import KernelCall
        a_s1, c_s1 = self._strides(layouts)
        return [KernelCall.of("RowOpsKernel", op=self.opword, n=self.n, rows=self.rows, blocks=1,
                              a_s1=a_s1, c_s1=c_s1, win_hi=NO_WINDOW)]

    def describe(self) -> str:
        return (f"rows={self.rows} n={self.n} eps={self.eps:.3g} on RowOpsKernel (NORM_LN; int32 residual at "
                f"2^-{self.fr}, gamma at 2^-{self.g}, beta at 2^-{self.bb})")

    def emit_comment(self) -> str:
        return (f"    /* [{self.index}] {self.onnx_node.op_type}({self.inputs[0].onnx_name}) -> "
                f"{self.output.onnx_name}  [{self.rows}][{self.n}] on RowOpsKernel (NORM_LN, int32 in) */")

    def emit_call(self, layouts: dict) -> str:
        a_s1, c_s1 = self._strides(layouts)
        return self.call(self.inputs[0].c_name, 0, "NULL", 0, self.output.c_name, 0, self.inputs[1].c_name,
                         self.opword, self.n, self.rows, 1, a_s1, 0, 0, 0, c_s1, 0, self.E, 0,
                         rs.P1_BITS + self.g - self.bb, 0, NO_WINDOW)


def _resadd_bias(cls, node, tensors, index, align_elems, ctx, h, d, y, bias, fr, name):
    """A ResAddBiasRoNode of h (int16 / int32 at one exponent) + d + bias -> y (int32 at fr)."""
    F = ctx.frac_bits
    n = int(h.shape[-1])
    fa = _exps(h, F)                                 # an int16 h may be at per-channel exponents
    _require((h.dma is None or (fa == fa[0]).all()) and y.dma == "i32" and (_exps(y, F) == fr).all(), node,
             "an int32 h at one exponent, the output the int32 residual")
    _require(d.dma is None and d.host is None and not d.is_state and d.shape[-1] == n, node,
             "delta: an int16 DMA tensor of the residual's width")
    _require(d.numel == h.numel == y.numel, node, "shapes")
    fd = _exps(d, F)
    _require((fa <= fr).all() and (fr - fa <= 63).all() and (fd <= fr).all() and (fr - fd <= 63).all(), node,
             f"exponents beyond the residual's 2^-{fr}")
    _require(n % align_elems == 0, node, f"row length {n} % {align_elems}")
    bias = np.asarray(bias, np.float64).reshape(-1)
    _require(bias.size == n and np.isfinite(bias).all(), node, "bias [n]")
    bb = rs.bias_bits(bias, fr)
    bq = np.rint(bias * 2.0 ** bb).astype(np.int64)
    p, bc = _pblock(tensors, name, 0, bq, fr - fa, fr - fd, 0, n)
    return cls(onnx_node=node, inputs=[h, d, p, tables_tensor(tensors)], output=y, index=index,
               align_elems=align_elems, F=F, rows=h.numel // n, n=n, fr=int(fr), bcast=bc, bb=int(bb),
               fa=int(fa[0]), a32=h.dma == "i32", bias=bias)


def _layernorm32(node, tensors, index, align_elems, ctx, h, y, gamma, beta, eps):
    F = ctx.frac_bits
    n = int(h.shape[-1])
    fr = _exps(h, F)
    _require(h.dma == "i32" and (fr == fr[0]).all(), node, "h: the int32 residual at one exponent")
    _require(y.dma is None and y.host is None and y.numel == h.numel and y.shape[-1] == n, node,
             "output: an int16 DMA tensor of h's shape")
    gamma = np.asarray(gamma, np.float64).reshape(-1)
    beta = np.asarray(beta, np.float64).reshape(-1)
    _require(gamma.size == n and beta.size == n and np.isfinite(gamma).all() and np.isfinite(beta).all(),
             node, "gamma / beta [n]")
    G, g, _bq, E = rs.norm_params(gamma, n, eps, int(fr[0]), beta=beta)
    bb = rs.beta_bits(beta)
    bq = np.rint(beta * 2.0 ** bb).astype(np.int64)
    fo = _exps(y, F)
    sh_c = rs.P1_BITS + g - fo
    _require((sh_c >= 0).all() and (sh_c <= 63).all() and 0 <= rs.P1_BITS + g - bb <= 63, node,
             f"gamma's exponent 2^-{g} beyond the shifts")
    _require(0 <= E < 2 ** 63, node, f"eps {eps} beyond the eps register")
    _require(n % align_elems == 0, node, f"row length {n} % {align_elems}")
    p, bc = _pblock(tensors, f"{y.onnx_name}__rowops_p", G, bq, 0, 0, sh_c, n)
    return LayerNorm32RoNode(onnx_node=node, inputs=[h, p, tables_tensor(tensors)], output=y, index=index,
                             align_elems=align_elems, F=F, rows=h.numel // n, n=n, fr=int(fr[0]), eps=eps,
                             g=int(g), bb=int(bb), E=int(E), bcast=bc, gamma=gamma, beta=beta)


def from_vit_resadd(node, tensors, index, align_elems, ctx: HostContext):
    """VitResAdd: ResAddBiasRoNode when h is the int32 DMA residual, else the host op."""
    from .vit_nodes import VitResAddNode, _bias
    h, y = tensors.get(node.input[0]), tensors.get(node.output[0])
    if h is None or y is None or (h.dma != "i32" and y.dma != "i32"):
        return VitResAddNode.from_onnx_node(node, tensors, index, align_elems, ctx)
    _need_unit(node, "the int32 DMA residual")
    _require(h.host is None and not h.is_state and h.dma in (None, "i32"), node, "h: a DMA tensor")
    d = tensors[node.input[1]]
    n = int(h.shape[-1])
    fr = int(_exps(y, ctx.frac_bits)[0])             # the int32 sum's exponent (h's when h is the residual)
    return _resadd_bias(ResAddBiasRoNode, node, tensors, index, align_elems, ctx, h, d, y,
                        _bias(ctx, tensors, node, 2, n, "bias"), fr, f"{y.onnx_name}__rowops_p")


def from_vit_layernorm(node, tensors, index, align_elems, ctx: HostContext):
    """VitLayerNorm: LayerNorm32RoNode when h is the int32 DMA residual, else the host op."""
    from .host_nodes import _attrs
    from .vit_nodes import VitLayerNormNode, _bias
    h = tensors.get(node.input[0])
    if h is None or h.dma != "i32":
        return VitLayerNormNode.from_onnx_node(node, tensors, index, align_elems, ctx)
    _need_unit(node, "the int32 DMA residual")
    y = tensors[node.output[0]]
    n = int(h.shape[-1])
    eps = _attrs(node).get("axi_eps", 1e-6)
    eps = float(eps.decode() if isinstance(eps, bytes) else eps)
    return _layernorm32(node, tensors, index, align_elems, ctx, h, y, _bias(ctx, tensors, node, 1, n, "gamma"),
                        _bias(ctx, tensors, node, 2, n, "beta"), eps)


@dataclass
class VitAttnPrepRoNode(RowOpsNode):
    """VitAttnPrep on the unit (the vision encoder and the Piper encoder: T = C
    rows, every key valid, pos 0; HOSTOPS_PLAN §10, phase 4b), the jobs —

      * q0 + bq rounded at the per-head q exponent -> the staging rows ``qs``
        (a COPY with a bias: ``rowops_spec.bias_round``, the policy's iqkv),
        then one TRANSPOSE (tiles of 8 query rows x HD, the heads as blocks)
        into the q.K^T x images (kw = 1);
      * per head a COPY with the bias of k0 straight into the K cache rows
        (its own parameter block: the exponents and the bias are per head);
      * v + bv rounded at the V cache's exponents -> ``vs``, then the V image:
        group_kw 1 a COPY with the heads as blocks, else per head the
        whole-block TRANSPOSEs of ``vimage_jobs`` (T % 16K == 0: no boundary
        rows).

    Every job is static (no runtime n)."""
    T:  int = 1
    H:  int = 1
    HD: int = 16
    C:  int = 16
    K:  int = 1
    pq_bc: bool = False
    pv_bc: bool = False
    fq: Optional[np.ndarray] = field(default=None, repr=False)     # [H]
    fk: Optional[np.ndarray] = field(default=None, repr=False)     # [H]
    fv: Optional[np.ndarray] = field(default=None, repr=False)     # [D]
    bq: Optional[np.ndarray] = field(default=None, repr=False)
    bk: Optional[np.ndarray] = field(default=None, repr=False)
    bv: Optional[np.ndarray] = field(default=None, repr=False)
    q0: object = field(default=None, repr=False)
    k0: object = field(default=None, repr=False)
    v:  object = field(default=None, repr=False)
    ck: object = field(default=None, repr=False)
    cv: object = field(default=None, repr=False)
    qs: object = field(default=None, repr=False)
    vs: object = field(default=None, repr=False)
    pq: object = field(default=None, repr=False)
    pk: list = field(default_factory=list, repr=False)
    pvb: object = field(default=None, repr=False)
    pz: object = field(default=None, repr=False)
    ekr: object = field(default=None, repr=False)   # Piper rel_kernel: raw key rows [CR][D] -> cache rows T ..
    CR: int = 0

    @property
    def D(self) -> int:
        return self.H * self.HD

    def state_writes(self):
        return [self.ck, self.cv, self.qs, self.vs]

    def state_updates(self):
        return self.state_writes()

    def _strides(self, layouts):
        nd, al = self.onnx_node, self.align_elems
        return (row_stride_bytes(self.q0, layouts, self.D, 2, nd, al),
                row_stride_bytes(self.k0, layouts, self.D, 2, nd, al),
                row_stride_bytes(self.v, layouts, self.D, 2, nd, al))

    def _jobs(self, layouts):
        sq, sk, sv = self._strides(layouts)
        T, H, HD, C, K, D = self.T, self.H, self.HD, self.C, self.K, self.D
        SQ = SV = D * 2
        NW = NO_WINDOW
        xp = rs.opword(rs.OP_XPOSE)
        jobs = [("q0 + bq at f_q[h] -> qs", self.q0, 0, None, 0, self.qs, 0, self.pq,
                 rs.opword(rs.OP_COPY, p_bcast=self.pq_bc), D, T, 1, sq, 0, 0, 0, SQ, 0, 0, 0, NW),
                ("qs -> the q.K^T x image of every head (8-row tiles, the heads as blocks)", self.qs, 0,
                 None, 0, self.output, 0, None, xp, HD, T // QIMAGE_R, H, QIMAGE_R * SQ, HD * 2, SQ, 2 * T,
                 2 * QIMAGE_R, HD * T * 2, QIMAGE_R, 0, NW)]
        for h in range(H):
            jobs.append((f"k0 + bk of head {h} at f_k -> the K cache", self.k0, h * HD * 2, None, 0, self.ck,
                         h * C * HD * 2, self.pk[h], rs.opword(rs.OP_COPY), HD, T, 1, sk, 0, 0, 0, HD * 2, 0,
                         0, 0, NW))
        if self.CR:
            jobs.append((f"the relative keys -> K cache rows {T} .. {T + self.CR - 1} of every head", self.ekr, 0,
                         None, 0, self.ck, T * HD * 2, self.pz, rs.opword(rs.OP_COPY, p_bcast=True), HD, self.CR,
                         H, D * 2, HD * 2, 0, 0, HD * 2, C * HD * 2, 0, 0, NW))
        jobs.append(("v + bv at the V cache's exponents -> vs", self.v, 0, None, 0, self.vs, 0, self.pvb,
                     rs.opword(rs.OP_COPY, p_bcast=self.pv_bc), D, T, 1, sv, 0, 0, 0, SV, 0, 0, 0, NW))
        if K == 1:                                       # the V cache has T rows per head (C: the K cache's)
            jobs.append(("vs -> the V cache rows (the heads as blocks)", self.vs, 0, None, 0, self.cv, 0,
                         self.pz, rs.opword(rs.OP_COPY, p_bcast=True), HD, T, H, SV, HD * 2, 0, 0, HD * 2,
                         T * HD * 2, 0, 0, NW))
        else:
            for h in range(H):
                for a_off, c_off, rows, blocks, a_s1, a_s2, b_s1, b_s2, c_s1, c_s2, R in \
                        vimage_jobs(0, T, HD, K, SV):
                    jobs.append((f"the V image of head {h}", self.vs, h * HD * 2 + a_off, None, 0, self.cv,
                                 h * T * HD * 2 + c_off, None, xp, HD, rows, blocks, a_s1, a_s2, b_s1, b_s2,
                                 c_s1, c_s2, R, 0, NW))
        return jobs

    def kernel_calls(self, layouts: dict) -> list:
        from .perf_calls import KernelCall, merge
        return merge([KernelCall.of("RowOpsKernel", op=j[8], n=j[9], rows=j[10], blocks=j[11], a_s1=j[12],
                                    a_s2=j[13], b_s1=j[14], b_s2=j[15], c_s1=j[16], c_s2=j[17], aux=j[18],
                                    win_lo=j[19], win_hi=j[20]) for j in self._jobs(layouts)])

    def describe(self) -> str:
        return (f"T={self.T} heads {self.H} head_dim {self.HD} on RowOpsKernel (the biases in the parameter "
                f"blocks, the q image kw=1, the caches; V interleave {self.K})")

    def emit_comment(self) -> str:
        return (f"    /* [{self.index}] VitAttnPrep({self.q0.onnx_name}, {self.k0.onnx_name}, {self.v.onnx_name}) "
                f"-> {self.output.onnx_name} + the caches  on RowOpsKernel: {self.describe()} */")

    def emit_call(self, layouts: dict) -> str:
        cn = lambda t: "NULL" if t is None else t.c_name   # noqa: E731
        lines = ["    {"]
        for i, j in enumerate(self._jobs(layouts)):
            if i:
                lines.append("        kernel_wait(KERNEL_ROWOPS);")
            lines.append(f"        /* {j[0]} */")
            lines.append(self.call(cn(j[1]), j[2], cn(j[3]), j[4], cn(j[5]), j[6], cn(j[7]), j[8], j[9], j[10],
                                   j[11], j[12], j[13], j[14], j[15], j[16], j[17], 0, j[18], 0, j[19], j[20],
                                   indent="        "))
        lines.append("    }")
        return "\n".join(lines)

    def reference(self, ins, dtype):  # noqa: ARG002
        a = {t.onnx_name: x for t, x in zip(self.inputs, ins, strict=True)}
        T, H, HD, D, F = self.T, self.H, self.HD, self.D, self.F
        p2 = lambda e: np.power(2.0, np.asarray(e, np.float64))   # noqa: E731
        raw = lambda t: np.rint(np.asarray(a[t.onnx_name], np.float64).reshape(T, D)   # noqa: E731
                                * p2(_exps(t, F))[None, :]).astype(np.int64)
        ck, cv = a[self.ck.onnx_name], a[self.cv.onnx_name]       # logical state arrays (raw), in place
        ck[:T] = rs.bias_round(raw(self.k0), _exps(self.k0, F), self.bk, np.repeat(self.fk, HD))
        if self.CR:                                              # the key rows' Q8.8 image, raw
            ck[T:T + self.CR] = np.rint(np.asarray(a[self.ekr.onnx_name], np.float64).reshape(self.CR, D)
                                        * p2(_exps(self.ekr, F))[None, :])
        cv[:T] = rs.bias_round(raw(self.v), _exps(self.v, F), self.bv, self.fv)
        yq = rs.bias_round(raw(self.q0), _exps(self.q0, F), self.bq, np.repeat(self.fq, HD))
        return yq.astype(np.float64).reshape(T, H, HD).transpose(1, 2, 0).copy()      # [H][HD][T]


def from_vit_attn_prep(node, tensors, index, align_elems, ctx: HostContext):
    """VitAttnPrep: VitAttnPrepRoNode with the attribute rowops = 1 (src/vit.py /
    src/piper.py rowops=True), else the host op."""
    from .host_nodes import _attrs
    from .llm_nodes import _attr_ints, _llm_inputs, _resolve
    from .vit_nodes import VitAttnPrepNode, _bias
    if not _rowops_attr(node):
        return VitAttnPrepNode.from_onnx_node(node, tensors, index, align_elems, ctx)
    _need_unit(node, "rowops = 1")
    F = ctx.frac_bits
    ins = _llm_inputs(node, tensors)
    _require(len(ins) in (10, 11), node, "inputs: q0, k0, v, cache_k, cache_v, bq, bk, bv, qs, vs[, key_rows]")
    q0, k0, v, ck, cv, _bq, _bk, _bv, qs, vs = ins[:10]
    ekr = ins[10] if len(ins) == 11 else None
    y = _resolve(tensors, node.output[0], node)
    a = _attrs(node)
    H, HD = int(a["num_heads"]), int(a["head_dim"])
    C = int(ck.shape[0])
    D = H * HD
    T = q0.numel // D
    K = int(cv.group_kw)
    CR = 0
    _require(int(a.get("qk_kw", 1)) == 1, node, "the unit writes the q image with qk_kw = 1")
    _require(HD % 16 == 0 and HD <= 1024, node, f"head_dim {HD}: a multiple of 16")
    _require(int(cv.shape[0]) == T, node, f"rows {T} must equal the V cache rows {cv.shape[0]}")
    if ekr is None:
        _require(T == C, node, f"rows {T} must equal the cache rows {C} (every key is valid)")
    else:                       # Piper rel_kernel: constant key rows [CR][D] (raw) behind the T keys of every head
        CR = int(ekr.shape[0])
        _require(len(ekr.shape) == 2 and int(ekr.shape[1]) == D and ekr.is_weight and ekr.dma is None
                 and (ekr.exp_channels(F) == F).all(), node,
                 f"key_rows: a constant [rows][{D}] at 2^-{F} (its image the raw cache rows)")
        _require(T + CR == C, node, f"the cache rows {C} must be {T} + the {CR} key rows")
    _require(T % QIMAGE_R == 0, node, f"rows {T} % {QIMAGE_R} (the q image's tiles)")
    _require(K == 1 or T % (16 * K) == 0, node, f"rows {T} % the V image's {16 * K}-row blocks")
    _require(q0.numel == k0.numel == v.numel == T * D, node, "q / k / v shapes")
    _require(list(y.shape) == [H, HD, T], node, f"qx must be [{H}][{HD}][{T}]")
    for t, w in ((q0, "q0"), (k0, "k0"), (v, "v"), (y, "qx")):
        _require(t.dma is None and t.host is None and not t.is_state, node, f"{w}: an int16 DMA tensor")
    for t, w in ((ck, "cache_k"), (cv, "cache_v")):
        _require(t.is_state and not t.is_host and t.group_layout == (H, HD), node,
                 f"{w} must be a DMA state stored group-major [{H}][{C}][{HD}]")
        _require(not t.exp_channels(F).any(), node, f"{w} holds raw integers (exponent 0)")
    _require(ck.group_kw == 1, node, "the K cache rows must not be interleaved")
    for t, w in ((qs, "qs"), (vs, "vs")):
        _require(t is not None and t.is_state and not t.is_host and t.dma is None
                 and list(t.shape) == [T, D], node, f"{w}: an int16 DMA state [{T}][{D}]")
    fq = np.asarray(_attr_ints(a, "q_exp", node, H), np.int64)
    fk = np.asarray(_attr_ints(a, "k_exp", node, H), np.int64)
    fv = np.asarray(_attr_ints(a, "v_exp", node, D), np.int64)
    bq, bk, bv = (_bias(ctx, tensors, node, i, D, w) for i, w in ((5, "bq"), (6, "bk"), (7, "bv")))
    try:
        bqq, saq, scq = rs.bias_params(bq, _exps(q0, F), np.repeat(fq, HD))
        bkq, sak, sck = rs.bias_params(bk, _exps(k0, F), np.repeat(fk, HD))
        bvq, sav, scv = rs.bias_params(bv, _exps(v, F), fv)
    except ValueError as e:
        raise SchedulerError(f"{node.op_type} node '{_label(node)}': {e}") from None
    _require((saq <= 63).all() and (sak <= 63).all() and (sav <= 63).all(), node, "exponents beyond the shifts")
    pq, pq_bc = _pblock(tensors, f"{y.onnx_name}__rowops_pq", 0, bqq, saq, 0, scq, D)
    pk = [param_tensor(tensors, f"{y.onnx_name}__rowops_pk{h}",
                       rs.param_block(np.zeros(HD, np.int64), bkq[h * HD:(h + 1) * HD], sak[h * HD:(h + 1) * HD],
                                      0, sck[h * HD:(h + 1) * HD])) for h in range(H)]
    pvb, pv_bc = _pblock(tensors, f"{y.onnx_name}__rowops_pv", 0, bvq, sav, 0, scv, D)
    pz = zero_param_tensor(tensors)
    tb = tables_tensor(tensors)
    return VitAttnPrepRoNode(onnx_node=node, inputs=[q0, k0, v, ck, cv, qs, vs, pq, *pk, pvb, pz, tb]
                             + ([ekr] if ekr is not None else []), output=y,
                             index=index, align_elems=align_elems, F=F, T=T, H=H, HD=HD, C=C, K=K,
                             pq_bc=pq_bc, pv_bc=pv_bc, fq=fq, fk=fk, fv=fv, bq=bq, bk=bk, bv=bv, q0=q0, k0=k0,
                             v=v, ck=ck, cv=cv, qs=qs, vs=vs, pq=pq, pk=pk, pvb=pvb, pz=pz, ekr=ekr, CR=CR)


VIT_OVERRIDES = {
    "VitResAdd":    from_vit_resadd,
    "VitLayerNorm": from_vit_layernorm,
    "VitAttnPrep":  from_vit_attn_prep,
}


# ------------------------------------------------------------------ #
# The TTS encoder's row prep (src/piper.py rowops=True)                #
# ------------------------------------------------------------------ #

@dataclass
class TtsRowPrepRoNode(RowOpsNode):
    """TtsRowPrep on the unit: x [T][C] at f_in -> y [T][taps·C] at f_out, row
    t's tap k reading row t + k - h (h = (taps - 1) / 2; zero outside [0, n)),
    + bias, ReLU, rows t >= n zero.  Per tap a COPY of the rows it reads (with
    the bias and RELU in the parameter block: ``rowops_spec.bias_round``; a
    plain re-round without one, exact) at the tap's column block; the rows the
    host zeroes — the first tap's row 0, the last tap's row n - 1, every row
    t >= n — as COPYs with the window [0, 0).  ``n`` is a runtime value
    (``_n``, ``_n1`` = n - 1, ``_nz`` = n > 0 in the emitted C); the calls are
    priced at n = T."""
    T: int = 1
    C: int = 1
    taps: int = 1
    relu: int = 0
    f_in: int = 0
    f_out: int = 0
    pb_bc: bool = False
    bias: Optional[np.ndarray] = field(default=None, repr=False)
    x:   object = field(default=None, repr=False)
    n_t: object = field(default=None, repr=False)
    pb:  object = field(default=None, repr=False)
    pz:  object = field(default=None, repr=False)

    @property
    def calls_env(self) -> dict:
        return {"_n": self.T, "_n1": self.T - 1, "_nz": 1}

    def _jobs(self, layouts, n, n1, nz):
        nd, al = self.onnx_node, self.align_elems
        T, C, taps = self.T, self.C, self.taps
        SX = row_stride_bytes(self.x, layouts, C, 2, nd, al)
        SY = row_stride_bytes(self.output, layouts, taps * C, 2, nd, al)
        NW = NO_WINDOW
        opc = rs.opword(rs.OP_COPY, p_bcast=self.pb_bc, relu=bool(self.relu))
        opz = rs.opword(rs.OP_COPY, p_bcast=True)
        h = (taps - 1) // 2
        sy = lambda r: f"{r} * {SY}u" if isinstance(r, str) else r * SY   # noqa: E731
        jobs = []
        for k in range(taps):
            d = k - h
            if d == 0:
                jobs.append((f"tap {k}: x rows [0, n) -> y columns [{k * C}, {(k + 1) * C})", self.x, 0, None, 0,
                             self.output, k * C * 2, self.pb, opc, C, n, 1, SX, 0, 0, 0, SY, 0, 0, 0, NW))
            elif d < 0:
                jobs.append((f"tap {k}: x rows [0, n - 1) -> y rows [1, n)", self.x, 0, None, 0, self.output,
                             k * C * 2 + SY, self.pb, opc, C, n1, 1, SX, 0, 0, 0, SY, 0, 0, 0, NW))
                jobs.append((f"tap {k}: y row 0 zero", self.output, 0, None, 0, self.output, k * C * 2,
                             self.pz, opz, C, nz, 1, SY, 0, 0, 0, SY, 0, 0, 0, 0))
            else:
                jobs.append((f"tap {k}: x rows [1, n) -> y rows [0, n - 1)", self.x, SX, None, 0, self.output,
                             k * C * 2, self.pb, opc, C, n1, 1, SX, 0, 0, 0, SY, 0, 0, 0, NW))
                jobs.append((f"tap {k}: y row n - 1 zero", self.output, 0, None, 0, self.output,
                             f"{k * C * 2}u + {sy(n1)}" if isinstance(n1, str) else k * C * 2 + sy(n1),
                             self.pz, opz, C, nz, 1, SY, 0, 0, 0, SY, 0, 0, 0, 0))
        jobs.append(("rows [n, T) zero", self.output, 0, None, 0, self.output, sy(n), self.pz, opz, taps * C,
                     f"{T}u - {n}" if isinstance(n, str) else T - n, 1, SY, 0, 0, 0, SY, 0, 0, 0, 0))
        return jobs

    def kernel_calls(self, layouts: dict) -> list:
        from .perf_calls import KernelCall, merge
        return merge([KernelCall.of("RowOpsKernel", op=j[8], n=j[9], rows=j[10], blocks=j[11], a_s1=j[12],
                                    a_s2=j[13], b_s1=j[14], b_s2=j[15], c_s1=j[16], c_s2=j[17], aux=j[18],
                                    win_lo=j[19], win_hi=j[20])
                      for j in self._jobs(layouts, self.T, self.T - 1, 1)])

    def describe(self) -> str:
        return (f"[{self.T}][{self.C}] -> taps {self.taps}" + (" + bias" if self.bias is not None else "")
                + (" ReLU" if self.relu else "") + " on RowOpsKernel (COPY per tap, the window for the zero rows)")

    def emit_comment(self) -> str:
        return (f"    /* [{self.index}] TtsRowPrep({self.x.onnx_name}) -> {self.output.onnx_name}  "
                f"on RowOpsKernel: {self.describe()} */")

    def emit_call(self, layouts: dict) -> str:
        cn = lambda t: "NULL" if t is None else t.c_name   # noqa: E731
        lines = ["    {",
                 f"        const unsigned _n = (unsigned){self.n_t.c_name}[0] > {self.T}u ? {self.T}u : "
                 f"(unsigned){self.n_t.c_name}[0];",
                 ] + (["        const unsigned _n1 = _n ? _n - 1u : 0u, _nz = _n ? 1u : 0u;"]
                      if self.taps > 1 else [])
        for i, j in enumerate(self._jobs(layouts, "_n", "_n1", "_nz")):
            if i:
                lines.append("        kernel_wait(KERNEL_ROWOPS);")
            lines.append(f"        /* {j[0]} */")
            lines.append(self.call(cn(j[1]), j[2], cn(j[3]), j[4], cn(j[5]), j[6], cn(j[7]), j[8], j[9], j[10],
                                   j[11], j[12], j[13], j[14], j[15], j[16], j[17], 0, j[18], 0, j[19], j[20],
                                   indent="        "))
        lines.append("    }")
        return "\n".join(lines)

    def reference(self, ins, dtype):  # noqa: ARG002
        T, C, taps = self.T, self.C, self.taps
        x = np.rint(np.asarray(ins[0], np.float64).reshape(T, C) * 2.0 ** self.f_in).astype(np.int64)
        n = max(0, min(int(np.asarray(ins[1]).reshape(-1)[0]), T))
        bias = np.zeros(C) if self.bias is None else self.bias
        core = rs.bias_round(x, self.f_in, bias, self.f_out, relu=bool(self.relu)).astype(np.float64)
        core[n:] = 0.0
        h = (taps - 1) // 2
        p = np.pad(core, ((h, h), (0, 0)))
        y = np.concatenate([p[k:k + T] for k in range(taps)], axis=1)
        y[n:] = 0.0
        return (y / 2.0 ** self.f_out).reshape(self.output.shape)


def from_tts_rowprep(node, tensors, index, align_elems, ctx: HostContext):
    """TtsRowPrep: TtsRowPrepRoNode with the attribute rowops = 1, else the host op."""
    from .host_nodes import _attrs
    from .tts_nodes import TtsRowPrepNode, _exp
    from .vit_nodes import _bias
    if not _rowops_attr(node):
        return TtsRowPrepNode.from_onnx_node(node, tensors, index, align_elems, ctx)
    _need_unit(node, "rowops = 1")
    F = ctx.frac_bits
    x, n, y = tensors[node.input[0]], tensors[node.input[1]], tensors[node.output[0]]
    a = _attrs(node)
    C = int(x.shape[-1])
    T = x.numel // C
    taps, relu = int(a.get("taps", 1)), int(a.get("relu", 0))
    _require(taps in (1, 3), node, "taps 1 or 3")
    _require(x.dma is None and x.host is None and not x.is_state and y.dma is None and y.host is None, node,
             "x / y: int16 DMA tensors")
    _require(n.host == "i32", node, "n: an int32 host tensor")
    _require(y.numel == T * taps * C, node, "output [T][taps * C]")
    _require(C % align_elems == 0, node, f"row length {C} % {align_elems}")
    f_in, f_out = _exp(x, node), _exp(y, node)
    bias = _bias(ctx, tensors, node, 2, C, "bias") if len(node.input) > 2 and node.input[2] else None
    try:
        bq, sa, sc = rs.bias_params(np.zeros(C) if bias is None else bias, f_in, f_out)
    except ValueError as e:
        raise SchedulerError(f"{node.op_type} node '{_label(node)}': {e}") from None
    _require((sa <= 63).all() and (sc <= 63).all(), node, "exponents beyond the shifts")
    pb, pb_bc = _pblock(tensors, f"{y.onnx_name}__rowops_p", 0, bq, sa, 0, sc, C)
    pz = zero_param_tensor(tensors)
    return TtsRowPrepRoNode(onnx_node=node, inputs=[x, n, pb, pz, tables_tensor(tensors)], output=y, index=index,
                            align_elems=align_elems, F=F, T=T, C=C, taps=taps, relu=relu, f_in=f_in, f_out=f_out,
                            pb_bc=pb_bc, bias=bias, x=x, n_t=n, pb=pb, pz=pz)


# ------------------------------------------------------------------ #
# The TTS encoder's residual LayerNorm (src/piper.py rowops=True)      #
# ------------------------------------------------------------------ #

def from_tts_resadd(node, tensors, index, align_elems, ctx: HostContext):
    """TtsResAdd (the frontend's rowops form of TtsResNorm's sum): x (int16 at
    f_x) + y (int16 at f_y) + bias -> s, the int32 DMA tensor at max(f_x, f_y)
    (piper_vits._enc_resnorm with the unit: the bias rounded once at that scale)."""
    from .vit_nodes import _bias
    _need_unit(node, "TtsResAdd")
    x, y, s = tensors[node.input[0]], tensors[node.input[1]], tensors[node.output[0]]
    n = int(x.shape[-1])
    fs = int(_exps(s, ctx.frac_bits)[0])
    _require(x.dma is None and x.host is None, node, "x: an int16 DMA tensor")
    return _resadd_bias(ResAddBiasRoNode, node, tensors, index, align_elems, ctx, x, y, s,
                        _bias(ctx, tensors, node, 2, n, "bias"), fs, f"{s.onnx_name}__rowops_p")


def from_tts_layernorm(node, tensors, index, align_elems, ctx: HostContext):
    """TtsLayerNorm: the int32 sum -> int16 rows at the output's exponent."""
    from .host_nodes import _attrs
    from .tts_nodes import _double_attr
    from .vit_nodes import _bias
    _need_unit(node, "TtsLayerNorm")
    h, y = tensors[node.input[0]], tensors[node.output[0]]
    n = int(h.shape[-1])
    eps = _double_attr(_attrs(node).get("eps", b"1e-5"))
    return _layernorm32(node, tensors, index, align_elems, ctx, h, y, _bias(ctx, tensors, node, 1, n, "gamma"),
                        _bias(ctx, tensors, node, 2, n, "beta"), eps)


@dataclass
class TtsSumRoNode(RowOpsNode):
    """A two-input TtsSum of the Piper flows (h + the res-skip half, RPU_PLAN
    §3.1) on the unit: one ADDX over the ``nch`` channel rows of L — A and B
    the channel slices [ch0, ch0 + nch) of their [C][L] tensors, the shifts
    fp - f_in0 / fp - f_in1 / fp - f_out at fp = max of the three exponents:
    the host's integer path (round half to even, saturate), bit for bit."""
    nch: int = 1
    L: int = 1
    ch0s: tuple = (0, 0)
    f_ins: tuple = (0, 0)
    f_out: int = 0
    pb: object = field(default=None, repr=False)

    @property
    def opword(self) -> int:
        return rs.opword(rs.OP_ADDX, rs.K_I16, rs.K_I16, rs.K_I16, b_en=True, p_bcast=True)

    def _regs(self):
        LB = self.L * 2
        return dict(op=self.opword, n=self.L, rows=self.nch, blocks=1, a_s1=LB, a_s2=0, b_s1=LB, b_s2=0,
                    c_s1=LB, c_s2=0, aux=0, win_lo=0, win_hi=NO_WINDOW)

    def kernel_calls(self, layouts: dict) -> list:  # noqa: ARG002
        from .perf_calls import KernelCall
        return [KernelCall.of("RowOpsKernel", **self._regs())]

    def describe(self) -> str:
        return (f"2 inputs [{self.nch}][{self.L}] at 2^-{self.f_ins[0]} / 2^-{self.f_ins[1]} -> 2^-{self.f_out} "
                f"on RowOpsKernel (ADDX)")

    def emit_comment(self) -> str:
        return (f"    /* [{self.index}] TtsSum({self.inputs[0].onnx_name}, {self.inputs[1].onnx_name}) -> "
                f"{self.output.onnx_name}  {self.describe()} */")

    def emit_call(self, layouts: dict) -> str:  # noqa: ARG002
        r = self._regs()
        return self.call(self.inputs[0].c_name, self.ch0s[0] * self.L * 2, self.inputs[1].c_name,
                         self.ch0s[1] * self.L * 2, self.output.c_name, 0, self.pb.c_name, r["op"], r["n"],
                         r["rows"], 1, r["a_s1"], 0, r["b_s1"], 0, r["c_s1"], 0, 0, 0, 0, 0, NO_WINDOW)

    def reference(self, ins, dtype):  # noqa: ARG002
        fp = max(*self.f_ins, self.f_out)
        acc = None
        for x, c0, f in zip(ins[:2], self.ch0s, self.f_ins, strict=True):
            raw = np.rint(np.asarray(x, np.float64).reshape(-1, self.L)[c0:c0 + self.nch] * 2.0 ** f).astype(np.int64)
            acc = (raw << (fp - f)) if acc is None else acc + (raw << (fp - f))
        y = rs.sat16(rs.rne_shr_vec(acc, np.full(acc.shape, fp - self.f_out)))
        return (y / 2.0 ** self.f_out).reshape(self.output.shape)


def from_tts_sum(node, tensors, index, align_elems, ctx: HostContext):
    """TtsSum: the VectorOP ADD or the host op as before; a two-input unmasked
    sum with differing exponents or channel offsets becomes TtsSumRoNode."""
    from .tts_nodes import TtsAddVopNode, TtsSumNode
    host = TtsSumNode.from_onnx_node(node, tensors, index, align_elems, ctx)
    if isinstance(host, TtsAddVopNode) or not enabled():
        return host
    xs = host.inputs[:2]
    if (len(host.ch0s) != 2 or host.masked or host.div != 1.0 or host.L % align_elems
            or any(t.dma or t.is_host or t.is_state for t in xs + [host.output])
            or any(c0 * host.L % align_elems for c0 in host.ch0s) or host.L > rs.N_MAX):
        return host
    fp = max(*host.f_ins, host.f_out)
    if fp - min(host.f_ins) > 63:
        return host
    pb, _bc = _pblock(tensors, f"{host.output.onnx_name}__rowops_p", 0, 0, fp - host.f_ins[0], fp - host.f_ins[1],
                      fp - host.f_out, 1)
    return TtsSumRoNode(onnx_node=node, inputs=[xs[0], xs[1], pb, tables_tensor(tensors)], output=host.output,
                        index=index, align_elems=align_elems, F=ctx.frac_bits, nch=host.nch, L=host.L,
                        ch0s=tuple(host.ch0s), f_ins=tuple(host.f_ins), f_out=host.f_out, pb=pb)


@dataclass
class TtsInterleaveRoNode(RowOpsNode):
    """TtsInterleave (the polyphase phases of a ConvTranspose into one row,
    y[o][m·s + r] = x[r·O + o][m]) on the unit: one TRANSPOSE of tiles of
    R = s phase rows x L into contiguous granules of s, one tile per output
    channel o — a bit copy."""
    s: int = 1
    O: int = 1      # noqa: E741
    L: int = 1
    Lc: int = 1     # columns per tile (<= 1024): L / Lc column chunks as the job's blocks

    def _regs(self):
        return dict(op=rs.opword(rs.OP_XPOSE), n=self.Lc, rows=self.O, blocks=self.L // self.Lc, a_s1=self.L * 2,
                    a_s2=self.Lc * 2, b_s1=self.O * self.L * 2, b_s2=2 * self.s, c_s1=self.L * self.s * 2,
                    c_s2=self.Lc * self.s * 2, aux=self.s, win_lo=0, win_hi=NO_WINDOW)

    def kernel_calls(self, layouts: dict) -> list:  # noqa: ARG002
        from .perf_calls import KernelCall
        return [KernelCall.of("RowOpsKernel", **self._regs())]

    def describe(self) -> str:
        return (f"interleave {self.s} phases of [{self.O}][{self.L}] on RowOpsKernel (TRANSPOSE, R = {self.s}, "
                f"{self.L // self.Lc} column chunks of {self.Lc})")

    def emit_comment(self) -> str:
        return (f"    /* [{self.index}] TtsInterleave({self.inputs[0].onnx_name}) -> {self.output.onnx_name}  "
                f"{self.describe()} */")

    def emit_call(self, layouts: dict) -> str:  # noqa: ARG002
        r = self._regs()
        return self.call(self.inputs[0].c_name, 0, "NULL", 0, self.output.c_name, 0, "NULL", r["op"], r["n"],
                         r["rows"], r["blocks"], r["a_s1"], r["a_s2"], r["b_s1"], r["b_s2"], r["c_s1"], r["c_s2"],
                         0, r["aux"], 0, 0, NO_WINDOW)

    def reference(self, ins, dtype):  # noqa: ARG002
        x = np.asarray(ins[0], np.float64).reshape(self.s, self.O, self.L)
        return x.transpose(1, 2, 0).reshape(self.output.shape)


def from_tts_interleave(node, tensors, index, align_elems, ctx: HostContext):
    """TtsInterleave: TtsInterleaveRoNode where the unit can run it, else the host op."""
    from .tts_nodes import TtsInterleaveNode
    host = TtsInterleaveNode.from_onnx_node(node, tensors, index, align_elems, ctx)
    x, y = host.inputs[0], host.output
    Lc = next((c for c in range(min(host.L, 1024), 0, -1) if host.L % c == 0 and c % align_elems == 0), 0)
    if (not enabled() or host.s > 8 or host.L % align_elems or not Lc
            or any(t.dma or t.is_host or t.is_state for t in (x, y))):
        return host
    return TtsInterleaveRoNode(onnx_node=node, inputs=[x, tables_tensor(tensors)], output=y, index=index,
                               align_elems=align_elems, F=ctx.frac_bits, s=host.s, O=host.O, L=host.L, Lc=Lc)


@dataclass
class TtsRelBandRoNode(RowOpsNode):
    """The Piper encoder's relative-key band on the unit (RPU_PLAN §3.1,
    ``PiperEncoderFrontend(rel_kernel=True)``): the scores ``s`` [T + CR][T]
    (keys-major) carry, as rows T + k, the band terms q.ek_k^T of key offset
    d = k - W (the relative keys are K cache rows, VitAttnPrepRoNode).  Per
    offset one TRANSPOSE of R = 1 (byte-strobed granules 2(T + 1) bytes apart)
    writes row T + k from its (16-byte-aligned) start onto the diagonal
    (i + d, i) of the band state [W + T][T] — its first W rows are guard rows
    that take the granules of i < -d (a stream A must start aligned, so a
    negative offset's row is read whole); the band proper, rows W .., is zero
    off the diagonals (they are rewritten every run).  Then one ADDX adds the
    band to the first T rows: lg[j][i] = sat16(s[j][i] + band[W + j][i]) —
    the host op's band add on every row and column (the softmax masks keys
    and queries >= n).  inputs [s, band (state), P0, tables]."""
    T: int = 1
    CR: int = 16
    W: int = 4
    band: object = field(default=None, repr=False)
    pz: object = field(default=None, repr=False)

    def state_writes(self):
        return [self.band]

    def state_updates(self):
        return self.state_writes()

    @property
    def opword(self) -> int:
        return rs.opword(rs.OP_ADDX, rs.K_I16, rs.K_I16, rs.K_I16, b_en=True, p_bcast=True)

    def _jobs(self):
        T, W = self.T, self.W
        xp, NW = rs.opword(rs.OP_XPOSE), NO_WINDOW
        s, band = self.inputs[0], self.band
        jobs = []
        for d in range(-W, W + 1):
            jobs.append((f"key offset {d:+d}: row {T + d + W} of the scores -> the diagonal (i {d:+d}, i)", s,
                         (T + d + W) * T * 2, None, 0, band, (d + W) * T * 2, None, xp, T - max(d, 0),
                         1, 1, 0, 0, 0, 2 * (T + 1), 0, 0, 1, 0, NW))
        jobs.append(("the scores + the band", s, 0, band, W * T * 2, self.output, 0, self.pz, self.opword, T, T, 1,
                     T * 2, 0, T * 2, 0, T * 2, 0, 0, 0, NW))
        return jobs

    def kernel_calls(self, layouts: dict) -> list:  # noqa: ARG002
        from .perf_calls import KernelCall, merge
        return merge([KernelCall.of("RowOpsKernel", op=j[8], n=j[9], rows=j[10], blocks=j[11], a_s1=j[12],
                                    a_s2=j[13], b_s1=j[14], b_s2=j[15], c_s1=j[16], c_s2=j[17], aux=j[18],
                                    win_lo=j[19], win_hi=j[20]) for j in self._jobs()])

    def describe(self) -> str:
        return (f"[{self.T}] keys x [{self.T}] queries + the relative-key band (window {self.W}) on RowOpsKernel: "
                f"{2 * self.W + 1} diagonal TRANSPOSEs (R = 1) onto the band state + one ADDX")

    def emit_comment(self) -> str:
        return (f"    /* [{self.index}] TtsRelBand({self.inputs[0].onnx_name}) -> {self.output.onnx_name}  "
                f"{self.describe()} */")

    def emit_call(self, layouts: dict) -> str:  # noqa: ARG002
        cn = lambda t: "NULL" if t is None else t.c_name   # noqa: E731
        lines = ["    {"]
        for i, j in enumerate(self._jobs()):
            if i:
                lines.append("        kernel_wait(KERNEL_ROWOPS);")
            lines.append(f"        /* {j[0]} */")
            lines.append(self.call(cn(j[1]), j[2], cn(j[3]), j[4], cn(j[5]), j[6], cn(j[7]), j[8], j[9], j[10],
                                   j[11], j[12], j[13], j[14], j[15], j[16], j[17], 0, j[18], 0, j[19], j[20],
                                   indent="        "))
        lines.append("    }")
        return "\n".join(lines)

    def reference(self, ins, dtype):  # noqa: ARG002
        T, W = self.T, self.W
        s = np.rint(np.asarray(ins[0], np.float64).reshape(T + self.CR, T)).astype(np.int64)
        band = ins[1]                                           # the logical state array [W + T][T], in place
        for d in range(-W, W + 1):
            i = np.arange(0, T - max(d, 0))
            band[i + d + W, i] = s[T + d + W, i]
        y = rs.sat16(s[:T] + np.rint(band[W:]).astype(np.int64))
        return y.astype(np.float64).reshape(self.output.shape) + 0.0


def from_tts_rel_band(node, tensors, index, align_elems, ctx: HostContext):
    """TtsRelBand (src/piper.py rel_kernel=True) -> TtsRelBandRoNode."""
    from .host_nodes import _attrs
    from .llm_nodes import _llm_inputs, _resolve
    _need_unit(node, "TtsRelBand")
    s, band = _llm_inputs(node, tensors)
    y = _resolve(tensors, node.output[0], node)
    W = int(_attrs(node).get("window", 4))
    T = int(y.shape[1])
    CR = int(s.shape[0]) - T
    F = ctx.frac_bits
    _require(list(y.shape) == [T, T] and T <= rs.N_MAX and T % align_elems == 0, node,
             f"logits [T][T], T <= {rs.N_MAX}, a multiple of {align_elems}")
    _require(len(s.shape) == 2 and int(s.shape[1]) == T and CR >= 2 * W + 1, node,
             f"scores [T + >= {2 * W + 1}][{T}] (the band terms as the rows behind the keys)")
    _require(band.is_state and not band.is_host and band.dma is None and list(band.shape) == [W + T, T]
             and band.group_layout is None, node, f"band: an int16 DMA state [{W + T}][{T}] (W guard rows)")
    for t, w in ((s, "scores"), (y, "logits")):
        _require(t.dma is None and t.host is None and not t.is_state and not t.exp_channels(F).any(), node,
                 f"{w}: a raw int16 DMA tensor")
    _require(not band.exp_channels(F).any(), node, "band: raw int16")
    pz = zero_param_tensor(tensors)
    return TtsRelBandRoNode(onnx_node=node, inputs=[s, band, pz, tables_tensor(tensors)], output=y, index=index,
                            align_elems=align_elems, F=F, T=T, CR=CR, W=W, band=band, pz=pz)


TTS_OVERRIDES = {
    "TtsResAdd":    from_tts_resadd,
    "TtsLayerNorm": from_tts_layernorm,
    "TtsRowPrep":   from_tts_rowprep,
    "TtsSum":       from_tts_sum,
    "TtsInterleave": from_tts_interleave,
    "TtsRelBand":   from_tts_rel_band,
}


# the axi.llm ops the unit takes over (merged after LLM_OP_FACTORIES in src/graph.py)
@dataclass
class DequantRoNode(RowOpsNode):
    """LlmDequant on the unit: x (int16 at per-channel exponents, or the int32
    residual) -> float32 DMA rows, ``y = float32(raw · 2^-f[c])`` (DEQUANT:
    round to nearest even — exact for |raw| < 2^24, so the host op's value bit
    for bit).  One job of ``rows`` rows of ``n``; a single row longer than the
    unit's N_MAX (the logits) as one job per contiguous chunk of ``n``, each
    with its own parameter block (the exponents of its columns).  inputs
    [x, P..., tables]; the output a DMA tensor or state."""
    rows: int = 1
    n:    int = 1
    a32:  bool = False
    bcast: bool = True
    rerow: bool = False
    pbs: list = field(default_factory=list, repr=False)

    @property
    def opword(self) -> int:
        return rs.opword(rs.OP_DEQUANT, rs.K_I32 if self.a32 else rs.K_I16, rs.K_I16, rs.K_F32,
                         p_bcast=self.bcast)

    def _strides(self, layouts):
        nd = self.onnx_node
        ab = 4 if self.a32 else 2
        if self.rerow:
            return (self.n * ab, self.n * 4)
        return (row_stride_bytes(self.inputs[0], layouts, self.n, ab, nd, self.align_elems),
                row_stride_bytes(self.output, layouts, self.n, 4, nd, self.align_elems))

    def _jobs(self, layouts):
        """[(a_off, c_off, rows, a_s1, c_s1, P, opword)]"""
        a_s1, c_s1 = self._strides(layouts)
        if not self.rerow:
            return [(0, 0, self.rows, a_s1, c_s1, self.pbs[0], self.opword)]
        ab = 4 if self.a32 else 2
        return [(k * self.n * ab, k * self.n * 4, 1, a_s1, c_s1, pb,
                 rs.opword(rs.OP_DEQUANT, rs.K_I32 if self.a32 else rs.K_I16, rs.K_I16, rs.K_F32, p_bcast=bc))
                for k, (pb, bc) in enumerate(self.pbs)]

    def reference(self, ins, dtype):  # noqa: ARG002
        from .llm_nodes import _f32
        return _f32(np.asarray(ins[0], np.float64)).reshape(self.output.shape)

    def kernel_calls(self, layouts: dict) -> list:
        from .perf_calls import KernelCall, merge
        return merge([KernelCall.of("RowOpsKernel", op=j[6], n=self.n, rows=j[2], blocks=1, a_s1=j[3], c_s1=j[4],
                                    win_hi=NO_WINDOW) for j in self._jobs(layouts)])

    def describe(self) -> str:
        return (f"rows={self.rows} n={self.n} on RowOpsKernel (DEQUANT: {'int32' if self.a32 else 'int16'} "
                f"-> float32 DMA" + (f", {len(self.pbs)} chunks of {self.n}" if self.rerow else "") + ")")

    def emit_comment(self) -> str:
        return (f"    /* [{self.index}] LlmDequant({self.inputs[0].onnx_name}) -> {self.output.onnx_name}  "
                f"[{self.rows}][{self.n}] on RowOpsKernel (DEQUANT -> float32"
                + (f"; {len(self.pbs)} jobs" if self.rerow else "") + ") */")

    def emit_call(self, layouts: dict) -> str:
        jobs = self._jobs(layouts)
        if not self.rerow:
            a_off, c_off, rows, a_s1, c_s1, pb, op = jobs[0]
            return self.call(self.inputs[0].c_name, 0, "NULL", 0, self.output.c_name, 0, pb.c_name, op, self.n,
                             rows, 1, a_s1, 0, 0, 0, c_s1, 0, 0, 0, 0, 0, NO_WINDOW)
        lines = ["    {"]
        for i, (a_off, c_off, rows, a_s1, c_s1, pb, op) in enumerate(jobs):
            if i:
                lines.append("        kernel_wait(KERNEL_ROWOPS);")
            lines.append(self.call(self.inputs[0].c_name, a_off, "NULL", 0, self.output.c_name, c_off, pb.c_name,
                                   op, self.n, rows, 1, a_s1, 0, 0, 0, c_s1, 0, 0, 0, 0, 0, NO_WINDOW,
                                   indent="        "))
        lines.append("    }")
        return "\n".join(lines)


def from_llm_dequant(node, tensors, index, align_elems, ctx: HostContext):
    """LlmDequant: DequantRoNode when the output is a float32 DMA tensor (the
    frontends' rowops=True), else the host op (a float32 host output)."""
    import hashlib
    from .llm_nodes import LlmDequantNode, _resolve
    y = _resolve(tensors, node.output[0], node)
    if y.dma != "f32":
        return LlmDequantNode.from_onnx_node(node, tensors, index, align_elems, ctx)
    _need_unit(node, "a float32 DMA output")
    x = _resolve(tensors, node.input[0], node)
    F = ctx.frac_bits
    n = int(x.shape[-1])
    _require(x.host is None and not x.is_state and x.dma in (None, "i32"), node, "input: a DMA tensor")
    _require(y.numel == x.numel and int(y.shape[-1]) == n and not y.is_host, node, "shapes")
    f = _exps(x, F)
    _require((0 <= f).all() and (f <= 63).all(), node, "exponents beyond the shifts")
    a32 = x.dma == "i32"
    if n <= rs.N_MAX:
        _require(n % align_elems == 0 or x.numel == n, node, f"rows of {n}: a multiple of {align_elems} (or one row)")
        p, bc = _pblock(tensors, f"{y.onnx_name}__rowops_p", 0, 0, 0, 0, f, n)
        return DequantRoNode(onnx_node=node, inputs=[x, p, tables_tensor(tensors)], output=y, index=index,
                             align_elems=align_elems, F=F, rows=x.numel // n, n=n, a32=a32, bcast=bc, pbs=[p])
    # one long row (the logits): contiguous chunks of <= N_MAX, a parameter block per chunk named by
    # its exponents (so the entries of a project share them)
    _require(x.numel == n, node, f"a row of {n} (> {rs.N_MAX}): one row")
    nc = next((d for d in range(rs.N_MAX, 0, -1) if n % d == 0 and d % align_elems == 0), 0)
    _require(nc > 0, node, f"a row of {n} has no divisor <= {rs.N_MAX} that is a multiple of {align_elems}")
    pbs = []
    for k in range(n // nc):
        fk = f[k * nc:(k + 1) * nc]
        h = hashlib.sha1(fk.astype(np.int64).tobytes()).hexdigest()[:12]
        pbs.append(_pblock(tensors, f"rowops_p_dq_{nc}_{h}", 0, 0, 0, 0, fk, nc))
    return DequantRoNode(onnx_node=node, inputs=[x, *[pb for pb, _ in pbs], tables_tensor(tensors)], output=y,
                         index=index, align_elems=align_elems, F=F, rows=1, n=nc, a32=a32, bcast=False, rerow=True,
                         pbs=pbs)


@dataclass
class SelectRowRoNode(RowOpsNode):
    """LlmSelectRow on the unit: y = h[n - 1] (n clamped to [1, rows]) — one COPY
    of one int32 row into the int32 DMA state (``_r`` = the row, from the
    runtime n; the call priced at row rows - 1).  inputs [h, n, P0, tables]."""
    rows: int = 1
    n:    int = 1
    pz:   object = field(default=None, repr=False)

    @property
    def opword(self) -> int:
        return rs.opword(rs.OP_COPY, rs.K_I32, rs.K_I16, rs.K_I32, p_bcast=True)

    def state_writes(self):
        return [self.output] if self.output.is_state else []

    def state_updates(self):
        return self.state_writes()

    def _stride(self, layouts):
        return row_stride_bytes(self.inputs[0], layouts, self.n, 4, self.onnx_node, self.align_elems)

    def reference(self, ins, dtype):  # noqa: ARG002
        h = np.asarray(ins[0], np.float64).reshape(self.rows, self.n)
        r = min(max(int(np.asarray(ins[1]).reshape(-1)[0]), 1), self.rows) - 1
        return h[r].reshape(self.output.shape).copy()

    def kernel_calls(self, layouts: dict) -> list:  # noqa: ARG002
        from .perf_calls import KernelCall
        return [KernelCall.of("RowOpsKernel", op=self.opword, n=self.n, rows=1, blocks=1, a_s1=0, c_s1=0,
                              win_hi=NO_WINDOW)]

    def describe(self) -> str:
        return f"row n-1 of [{self.rows}][{self.n}] (int32) on RowOpsKernel (COPY)"

    def emit_comment(self) -> str:
        return (f"    /* [{self.index}] LlmSelectRow({self.inputs[0].onnx_name}) -> {self.output.onnx_name}  "
                f"{self.describe()} */")

    def emit_call(self, layouts: dict) -> str:
        nv = f"{self.inputs[1].c_name}[0]"
        return "\n".join([
            "    {",
            f"        const unsigned _r = {nv} < 1 ? 0u : ((unsigned){nv} > {self.rows}u ? {self.rows - 1}u : "
            f"(unsigned){nv} - 1u);",
            self.call(self.inputs[0].c_name, f"_r * {self._stride(layouts)}u", "NULL", 0, self.output.c_name, 0,
                      self.pz.c_name, self.opword, self.n, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, NO_WINDOW,
                      indent="        "),
            "    }"])


def from_llm_select_row(node, tensors, index, align_elems, ctx: HostContext):
    """LlmSelectRow: SelectRowRoNode for the int32 DMA residual (rowops), else the host op."""
    from .llm_nodes import LlmSelectRowNode, _llm_inputs, _resolve
    h, nt = _llm_inputs(node, tensors)
    y = _resolve(tensors, node.output[0], node)
    if h.dma != "i32" or y.dma != "i32" or not enabled():
        return LlmSelectRowNode.from_onnx_node(node, tensors, index, align_elems, ctx)
    F = ctx.frac_bits
    n = int(h.shape[-1])
    _require(nt.host == "i32", node, "n must be an i32 host tensor")
    _require(y.numel == n and np.array_equal(_exps(y, F), _exps(h, F)), node,
             "output: one int32 row at h's exponent")
    _require(n % align_elems == 0 and n <= rs.N_MAX, node, f"row length {n}: a multiple of {align_elems}")
    return SelectRowRoNode(onnx_node=node, inputs=[h, nt, zero_param_tensor(tensors), tables_tensor(tensors)],
                           output=y, index=index, align_elems=align_elems, F=F, rows=h.numel // n, n=n,
                           pz=zero_param_tensor(tensors))


@dataclass
class PixelShuffleRoNode(RowOpsNode):
    """VitPixelShuffle on the unit: columns [k0, k0 + K) of the x s shuffle of xf
    [n·n][D] — output row (i, j) (m = n / s per side) concatenates the source
    rows (i·s + b)·n + j·s + a, column block q = b·s + a.  Per source block q
    that the column window touches, one COPY of its (partial) row with the
    output rows as a 2-level pattern (j: rows, i: blocks) — a bit copy.
    inputs [xf, P0, tables]."""
    n: int = 1
    D: int = 1
    s: int = 1
    k0: int = 0
    K: int = 1
    pz: object = field(default=None, repr=False)

    @property
    def opword(self) -> int:
        return rs.opword(rs.OP_COPY, p_bcast=True)

    def _jobs(self, layouts):
        n, D, s, k0, K = self.n, self.D, self.s, self.k0, self.K
        m = n // s
        sx = row_stride_bytes(self.inputs[0], layouts, D, 2, self.onnx_node, self.align_elems)
        sy = row_stride_bytes(self.output, layouts, K, 2, self.onnx_node, self.align_elems)
        jobs = []
        for q in range(s * s):
            lo, hi = max(k0, q * D), min(k0 + K, (q + 1) * D)
            if lo >= hi:
                continue
            b, a = divmod(q, s)
            jobs.append((f"source block {q} (rows (i s + {b}) n + j s + {a}), columns [{lo}, {hi})",
                         (lo - q * D) * 2 + (b * n + a) * sx, (lo - k0) * 2, hi - lo, m, m, s * sx, s * n * sx,
                         sy, m * sy))
        return jobs

    def reference(self, ins, dtype):  # noqa: ARG002
        from .vit_nodes import pixel_shuffle_rows
        x = np.asarray(ins[0], np.float64).reshape(self.n * self.n, self.D)
        src = pixel_shuffle_rows(self.n, self.s)
        full = x[src].reshape(src.shape[0], self.s * self.s * self.D)
        return full[:, self.k0:self.k0 + self.K].reshape(self.output.shape).copy()

    def kernel_calls(self, layouts: dict) -> list:
        from .perf_calls import KernelCall, merge
        return merge([KernelCall.of("RowOpsKernel", op=self.opword, n=j[3], rows=j[4], blocks=j[5], a_s1=j[6],
                                    a_s2=j[7], c_s1=j[8], c_s2=j[9], win_hi=NO_WINDOW) for j in self._jobs(layouts)])

    def describe(self) -> str:
        return (f"[{self.n * self.n}][{self.D}] -> columns {self.k0}..{self.k0 + self.K - 1} of the x{self.s} "
                f"shuffle on RowOpsKernel ({len(self._jobs({}))} COPYs)")

    def emit_comment(self) -> str:
        return (f"    /* [{self.index}] VitPixelShuffle({self.inputs[0].onnx_name}) -> {self.output.onnx_name}  "
                f"{self.describe()} */")

    def emit_call(self, layouts: dict) -> str:
        lines = ["    {"]
        for i, j in enumerate(self._jobs(layouts)):
            if i:
                lines.append("        kernel_wait(KERNEL_ROWOPS);")
            lines.append(f"        /* {j[0]} */")
            lines.append(self.call(self.inputs[0].c_name, j[1], "NULL", 0, self.output.c_name, j[2], self.pz.c_name,
                                   self.opword, j[3], j[4], j[5], j[6], j[7], 0, 0, j[8], j[9], 0, 0, 0, 0,
                                   NO_WINDOW, indent="        "))
        lines.append("    }")
        return "\n".join(lines)


def from_vit_pixel_shuffle(node, tensors, index, align_elems, ctx: HostContext):
    """VitPixelShuffle: PixelShuffleRoNode with the attribute rowops = 1, else the host op."""
    from .vit_nodes import VitPixelShuffleNode
    host = VitPixelShuffleNode.from_onnx_node(node, tensors, index, align_elems, ctx)
    if not _rowops_attr(node):
        return host
    _need_unit(node, "rowops = 1")
    _require(host.D % align_elems == 0 and host.k0 % align_elems == 0 and host.K % align_elems == 0, node,
             f"D, col0 and the column count: multiples of {align_elems}")
    _require(host.D <= rs.N_MAX, node, f"D {host.D} beyond the unit's rows of {rs.N_MAX}")
    return PixelShuffleRoNode(onnx_node=node, inputs=[host.inputs[0], zero_param_tensor(tensors),
                                                      tables_tensor(tensors)],
                              output=host.output, index=index, align_elems=align_elems, F=ctx.frac_bits,
                              n=host.n, D=host.D, s=host.s, k0=host.k0, K=host.K, pz=zero_param_tensor(tensors))


@dataclass
class EmbedAddRoNode(RowOpsNode):
    """VitEmbedAdd on the unit (RPU_PLAN §3.1, the vision study's ``iembed``):
    h = sat32(pe << (fr - f_pe[c]) + B) — the patch embedding (int16 at its
    per-channel exponents) plus the position / bias table B as an int32 DMA
    stream at 2^-fr (the table rounded once; shipped as a Q8.8 weight whose
    image is the int32 words), into the int32 residual: one ADDX.  inputs
    [pe, table, P, tables]."""
    rows: int = 1
    n:    int = 1
    fr:   int = 15
    bcast: bool = True
    table: object = field(default=None, repr=False)

    @property
    def opword(self) -> int:
        return rs.opword(rs.OP_ADDX, rs.K_I16, rs.K_I32, rs.K_I32, b_en=True, p_bcast=self.bcast)

    def _strides(self, layouts):
        nd = self.onnx_node
        return (row_stride_bytes(self.inputs[0], layouts, self.n, 2, nd, self.align_elems), self.n * 4,
                row_stride_bytes(self.output, layouts, self.n, 4, nd, self.align_elems))

    @staticmethod
    def table_words(b0q: np.ndarray) -> np.ndarray:
        """The int32 table [rows][n] as the int16 words of a Q8.8 weight [rows][2n]."""
        w = np.asarray(b0q, np.int64).astype(np.int32).reshape(-1).view(np.int16)
        return raw_as_data(w).reshape(b0q.shape[0], 2 * b0q.shape[1])

    def reference(self, ins, dtype):  # noqa: ARG002
        pe = np.asarray(ins[0], np.float64).reshape(self.rows, self.n)
        w = np.rint(np.asarray(ins[1], np.float64).reshape(self.rows, 2 * self.n) * 256.0).astype(np.int64)
        b0 = ((w[:, 1::2] << 16) + (w[:, 0::2] & 0xFFFF)).astype(np.int64)
        b0 = np.where(b0 >= 2 ** 31, b0 - 2 ** 32, b0)
        raw = np.clip(np.rint(pe * 2.0 ** self.fr) + b0, -2147483648.0, 2147483647.0)
        return (raw / 2.0 ** self.fr).reshape(self.output.shape)

    def kernel_calls(self, layouts: dict) -> list:
        from .perf_calls import KernelCall
        a_s1, b_s1, c_s1 = self._strides(layouts)
        return [KernelCall.of("RowOpsKernel", op=self.opword, n=self.n, rows=self.rows, blocks=1,
                              a_s1=a_s1, b_s1=b_s1, c_s1=c_s1, win_hi=NO_WINDOW)]

    def describe(self) -> str:
        return (f"rows={self.rows} n={self.n} on RowOpsKernel (ADDX: the patch embedding + the int32 "
                f"position table at 2^-{self.fr} -> the int32 residual)")

    def emit_comment(self) -> str:
        return (f"    /* [{self.index}] VitEmbedAdd({self.inputs[0].onnx_name}) -> {self.output.onnx_name}  "
                f"{self.describe()} */")

    def emit_call(self, layouts: dict) -> str:
        a_s1, b_s1, c_s1 = self._strides(layouts)
        return self.call(self.inputs[0].c_name, 0, self.inputs[1].c_name, 0, self.output.c_name, 0,
                         self.inputs[2].c_name, self.opword, self.n, self.rows, 1, a_s1, 0, b_s1, 0, c_s1, 0,
                         0, 0, 0, 0, NO_WINDOW)


def from_vit_embed_add(node, tensors, index, align_elems, ctx: HostContext):
    """VitEmbedAdd: EmbedAddRoNode with the attribute rowops = 1 (src/vit.py
    iembed: the table an int32 image), else the host op."""
    from .llm_nodes import _resolve
    from .vit_nodes import VitEmbedAddNode
    if not _rowops_attr(node):
        return VitEmbedAddNode.from_onnx_node(node, tensors, index, align_elems, ctx)
    _need_unit(node, "rowops = 1")
    F = ctx.frac_bits
    pe = _resolve(tensors, node.input[0], node)
    tb = _resolve(tensors, node.input[1], node)
    y = _resolve(tensors, node.output[0], node)
    n = int(pe.shape[-1])
    rows = pe.numel // n
    _require(pe.dma is None and pe.host is None and not pe.is_state, node, "pe: an int16 DMA tensor")
    _require(y.dma == "i32" and y.numel == pe.numel and int(y.shape[-1]) == n, node,
             "output: the int32 residual of pe's shape")
    fr = _exps(y, F)
    _require((fr == fr[0]).all(), node, "the residual at one exponent")
    _require(tb.is_weight and list(tb.shape) == [rows, 2 * n] and (tb.exp_channels(F) == F).all(), node,
             f"table: a Q8.8 weight [{rows}][{2 * n}] holding the int32 words")
    fp = _exps(pe, F)
    _require((fp <= fr[0]).all() and (fr[0] - fp <= 63).all(), node, "pe exponents beyond the residual's")
    _require(n % align_elems == 0 and n <= rs.N_MAX, node, f"row length {n}")
    p, bc = _pblock(tensors, f"{y.onnx_name}__rowops_p", 0, 0, fr[0] - fp, 0, 0, n)
    return EmbedAddRoNode(onnx_node=node, inputs=[pe, tb, p, tables_tensor(tensors)], output=y, index=index,
                          align_elems=align_elems, F=F, rows=rows, n=n, fr=int(fr[0]), bcast=bc, table=tb)


VIT_OVERRIDES["VitPixelShuffle"] = from_vit_pixel_shuffle
VIT_OVERRIDES["VitEmbedAdd"] = from_vit_embed_add

LLM_OVERRIDES = {
    "LlmDequant":  from_llm_dequant,
    "LlmSelectRow": from_llm_select_row,
    "LlmResAdd":   from_llm_resadd,
    "LlmRMSNorm":  from_llm_rmsnorm,
    "LlmSiluMul":  from_llm_silu_mul,
    "LlmAttnPrep": from_llm_attn_prep,
}


__all__ = ("TABLES_NAME", "NO_WINDOW", "enabled", "raw_as_data", "tables_tensor", "param_tensor",
           "row_stride_bytes", "RowOpsNode", "LayerNormRoNode", "layernorm_ineligible",
           "from_onnx_layernorm", "ResAddRoNode", "RMSNormRoNode", "SiluMulRoNode", "LLM_OVERRIDES",
           "AttnPrepRoNode", "from_llm_attn_prep", "rope_raw_tensor", "zero_param_tensor", "vimage_jobs",
           "ATTN_PREP_C", "QIMAGE_R", "VitAttnPrepRoNode", "from_vit_attn_prep", "TtsRowPrepRoNode",
           "from_tts_rowprep", "TransposeRoNode", "transpose_job", "from_onnx_transpose",
           "TtsSumRoNode", "from_tts_sum", "TtsInterleaveRoNode", "from_tts_interleave",
           "TtsRelBandRoNode", "from_tts_rel_band", "DequantRoNode", "from_llm_dequant", "SelectRowRoNode", "from_llm_select_row",
           "PixelShuffleRoNode", "from_vit_pixel_shuffle", "EmbedAddRoNode", "from_vit_embed_add",
           "ResAddBiasRoNode", "LayerNorm32RoNode", "VIT_OVERRIDES", "TTS_OVERRIDES")
