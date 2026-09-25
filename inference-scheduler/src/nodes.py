"""
Supported ONNX operators and their mapping to VectorOPKernel op codes,
plus MatmulNode for the separate MatmulKernel IP, ConvNode for ConvKernel,
and PoolNode for PoolingKernel.

Each ScheduledNode wraps one onnx.NodeProto, holds resolved TensorInfo
references for its inputs/outputs, and knows how to emit the corresponding
run_op() call in the generated C file.

MatmulNode wraps a MatMul ONNX op and emits run_matmul() calls for
the XMatmulkernel driver API.

ConvNode wraps a Conv ONNX op and emits run_conv() calls for
the XConvkernel driver API.  Only 2-D convolution (4-D input) is supported.

PoolNode wraps MaxPool / AveragePool / LpPool / GlobalMaxPool /
GlobalAveragePool / GlobalLpPool ONNX ops and emits run_pool() calls for
the XPoolingkernel driver API.  Only 2-D pooling (4-D NCHW input) is supported.
Global* variants are handled by the node itself (pool_h=in_h, pool_w=in_w).

Supported ONNX ops (VectorOPKernel)
------------------------------------
  Add                 → OP_ADD   (binary)
  Sub                 → OP_SUB   (binary)
  Mul                 → OP_MUL   (binary)
  Div                 → OP_DIV   (binary)
  Relu                → OP_RELU  (unary)
  Clip(min=0, max=6)  → OP_RELU6 (unary)

Supported ONNX ops (MatmulKernel)
----------------------------------
  MatMul              → run_matmul()  (2-D and batched 3-D)

Supported ONNX ops (ConvKernel)
--------------------------------
  Conv (2-D, NCHW)    → run_conv()

Supported ONNX ops (PoolingKernel)
-----------------------------------
  MaxPool             → run_pool()  (pool_type=0)
  AveragePool         → run_pool()  (pool_type=1)
  LpPool (p=1 or 2)  → run_pool()  (pool_type=2)
  GlobalMaxPool       → run_pool()  (pool_type=0, pool_h/w = in_h/w)
  GlobalAveragePool   → run_pool()  (pool_type=1, pool_h/w = in_h/w)
  GlobalLpPool        → run_pool()  (pool_type=2, pool_h/w = in_h/w)

All other ops raise SchedulerError.

Multidirectional broadcasting
------------------------------
Binary ops support partial ONNX multidirectional broadcasting: one input
may be smaller than the output, provided its shape right-aligns to the
output and any broadcasted (size-1 or absent) dimensions form a contiguous
leading block (no interleaved matching and broadcast dims).

When broadcasting is detected the node emits a single run_op() call with
outer, a_inc, and b_inc arguments so the hardware kernel handles the outer
loop internally.  The chunk stride is rounded up to INFERENCE_ALIGN_BYTES
so every inner burst starts at a 16-byte-aligned physical address.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import ClassVar, List, Optional, Tuple

import onnx

import numpy as np

from .tensor import TensorInfo


class SchedulerError(Exception):
    """Raised when the ONNX graph cannot be compiled to VectorOPKernel calls."""


# ------------------------------------------------------------------ #
# Op-code constants (must match VectorOP.h)                          #
# ------------------------------------------------------------------ #

OP_ADD   = 0
OP_SUB   = 1
OP_MUL   = 2
OP_DIV   = 3
OP_RELU  = 4
OP_RELU6 = 5

# Human-readable names for comments
OP_NAMES = {
    OP_ADD:   "VECTOROP_ADD",
    OP_SUB:   "VECTOROP_SUB",
    OP_MUL:   "VECTOROP_MUL",
    OP_DIV:   "VECTOROP_DIV",
    OP_RELU:  "VECTOROP_RELU",
    OP_RELU6: "VECTOROP_RELU6",
}

# Fused-activation codes for the kernel's `act` register (must match the
# Act enum in VectorOP.h).  Applied by the kernel after the op, so
# Add -> Relu / Clip(0,6) collapses into one run_op() call.
ACT_NONE  = 0
ACT_RELU  = 1
ACT_RELU6 = 2

ACT_NAMES = {
    ACT_NONE:  "VECTOROP_ACT_NONE",
    ACT_RELU:  "VECTOROP_ACT_RELU",
    ACT_RELU6: "VECTOROP_ACT_RELU6",
}

# op_code of a unary activation node -> act code it fuses into
_ACT_FOR_OP = {
    OP_RELU:  ACT_RELU,
    OP_RELU6: ACT_RELU6,
}

# ONNX op_type → (op_code, arity)
# arity 2 = binary (reads a and b), arity 1 = unary (reads a only)
_ONNX_OP_MAP = {
    "Add":  (OP_ADD,   2),
    "Sub":  (OP_SUB,   2),
    "Mul":  (OP_MUL,   2),
    "Div":  (OP_DIV,   2),
    "Relu": (OP_RELU,  1),
    # Clip is matched by name but validated for (0,6) attributes below
    "Clip": (OP_RELU6, 1),
}

# Public sets used by graph.py to build a comprehensive "supported ops" list.
VECTOROP_OP_TYPES: frozenset = frozenset(_ONNX_OP_MAP)
RESHAPE_OP_TYPES: frozenset = frozenset({
    "Reshape",
    "Squeeze",
    "Unsqueeze",
    "Dropout",
    "Flatten",   # axis-N split is irrelevant for a buffer alias — same numel
})

# ------------------------------------------------------------------ #
# Alignment constants                                                 #
#                                                                     #
# _ALIGN_BYTES is a hardware requirement (AXI burst boundary).       #
# _BYTES_PER_ELEM must match INFERENCE_BYTES_PER_ELEM in the         #
# generated inference.h — change both together when the data type    #
# changes (e.g. ap_fixed<16,8> → float32 means 2 → 4).              #
# _ALIGN_ELEMS is derived and must not be hardcoded elsewhere.       #
# ------------------------------------------------------------------ #

_ALIGN_BYTES = 16   # AXI burst alignment requirement (hardware-fixed)
# _BYTES_PER_ELEM and _ALIGN_ELEMS are data-type-dependent.
# They are injected per-node via ScheduledNode.from_onnx_node(align_elems=...)
# rather than kept as module-level constants.


# ------------------------------------------------------------------ #
# Clip attribute extraction                                           #
# ------------------------------------------------------------------ #

def _get_clip_bounds(node: onnx.NodeProto) -> Tuple[Optional[float], Optional[float]]:
    """
    Extract Clip min/max from either:
      - Opset < 11: stored as attributes 'min' and 'max'
      - Opset >= 11: stored as optional input tensors (indices 1 and 2)
    Returns (min_val, max_val); None means the bound is not present.
    """
    attrs = {a.name: a for a in node.attribute}
    if "min" in attrs or "max" in attrs:
        mn = attrs["min"].f if "min" in attrs else None
        mx = attrs["max"].f if "max" in attrs else None
        return mn, mx
    return None, None


# ------------------------------------------------------------------ #
# Broadcasting helper                                                 #
# ------------------------------------------------------------------ #

def _broadcast_info(
    t: TensorInfo, output: TensorInfo, align_elems: int = 8
) -> Tuple[int, int, int, bool]:
    """
    Compute how tensor *t* broadcasts to *output*.

    Returns (outer_count, chunk_size, aligned_chunk_size, advances):

      outer_count         Number of loop iterations.  1 = no broadcast
                          (t covers the whole output in one call).
      chunk_size          t.numel — data elements in one chunk.
      aligned_chunk_size  chunk_size rounded up to align_elems so every
                          loop iteration starts at an INFERENCE_ALIGN_BYTES-
                          aligned physical address.  May be > chunk_size,
                          producing gap elements between data blocks.
      advances            True  → t strides through the output (outer_count=1,
                                  or t.numel == output.numel).
                          False → t repeats at offset 0 each iteration
                                  (t.numel < output.numel).

    Raises SchedulerError if t's shape is incompatible with broadcasting
    to output under the trailing-contiguous constraint.
    """
    if t.numel == output.numel:
        # Exact match — no broadcasting, single call covers everything.
        return (1, output.numel, output.numel, True)

    if output.numel % t.numel != 0:
        raise SchedulerError(
            f"Tensor '{t.onnx_name}' (numel={t.numel}) is not a factor of "
            f"output '{output.onnx_name}' (numel={output.numel}); "
            f"cannot broadcast."
        )

    outer_count = output.numel // t.numel
    chunk_size  = t.numel

    # Right-align t.shape to output.shape, padding missing leading dims with 1.
    pad       = len(output.shape) - len(t.shape)
    t_aligned = [1] * pad + list(t.shape)

    # Trailing-contiguous constraint:
    # After right-alignment each t dim must be either:
    #   - Equal to the corresponding output dim  (matching)
    #   - 1                                      (broadcast)
    # AND all broadcast dims (1s) must form a contiguous LEADING block —
    # no matching dim may appear before a broadcast dim.
    found_match = False
    for td, od in zip(t_aligned, output.shape, strict=False):
        if td == od:
            found_match = True
        elif td == 1:
            if found_match:
                raise SchedulerError(
                    f"Tensor '{t.onnx_name}' shape={t.shape} cannot be broadcast "
                    f"to output '{output.onnx_name}' shape={output.shape}: "
                    f"broadcast dimensions must form a contiguous leading block "
                    f"(found a broadcast dim after a matching dim)."
                )
        else:
            raise SchedulerError(
                f"Tensor '{t.onnx_name}' shape={t.shape} cannot be broadcast "
                f"to output '{output.onnx_name}' shape={output.shape}: "
                f"dimension {td} does not match output dimension {od} and is not 1."
            )

    # Round chunk up to align_elems so each iteration's physical start address
    # is INFERENCE_ALIGN_BYTES-aligned.  The extra elements are gap/padding and
    # are never read or written by VectorOPKernel.
    aligned_chunk_size = (chunk_size + align_elems - 1) & ~(align_elems - 1)

    return (outer_count, chunk_size, aligned_chunk_size, False)


# ------------------------------------------------------------------ #
# ScheduledNode                                                       #
# ------------------------------------------------------------------ #

@dataclass
class ScheduledNode:
    """One ONNX operator mapped to one or more VectorOPKernel invocations."""

    kernel_name: ClassVar[str] = "VectorOPKernel"

    onnx_node:   onnx.NodeProto
    op_code:     int                   # OP_ADD … OP_RELU6
    arity:       int                   # 1 = unary, 2 = binary
    inputs:      List[TensorInfo]      # resolved input tensors
    output:      TensorInfo            # single output tensor
    index:       int = 0               # sequential index in graph
    align_elems: int = 8               # elements per AXI alignment boundary

    # ------------------------------------------------------------------ #
    # Broadcast fields — populated by validate()                          #
    # ------------------------------------------------------------------ #

    # outer_count > 1 means broadcasting: the kernel is invoked outer_count
    # times, each processing chunk_size elements at an aligned stride.
    outer_count:        int  = field(default=1,    init=False)
    chunk_size:         int  = field(default=0,    init=False)
    aligned_chunk_size: int  = field(default=0,    init=False)
    # Which inputs stride through the output (True) vs repeat at offset 0 (False).
    a_advances:         bool = field(default=True, init=False)
    b_advances:         bool = field(default=True, init=False)

    # Fused activation (ACT_*) applied by the kernel after the op — set by
    # OnnxGraph._fuse_activations() when a following Relu / Clip(0,6) was
    # folded into this node; the folded ONNX nodes are kept for the report.
    act:                int  = field(default=ACT_NONE, init=False)
    fused_nodes:        List[onnx.NodeProto] = field(default_factory=list, init=False)

    @property
    def fusable_act(self) -> Optional[int]:
        """The act code this node would contribute if fused into its
        producer (Relu -> ACT_RELU, Clip(0,6) -> ACT_RELU6), else None."""
        if self.arity != 1:
            return None
        return _ACT_FOR_OP.get(self.op_code)

    # ------------------------------------------------------------------ #
    # Factory                                                              #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_onnx_node(
        cls,
        node:        onnx.NodeProto,
        tensors:     dict,           # name → TensorInfo
        index:       int,
        align_elems: int = 8,
    ) -> "ScheduledNode":
        op_type = node.op_type

        if op_type not in _ONNX_OP_MAP:
            raise SchedulerError(
                f"Node '{node.name or op_type}' (op_type='{op_type}') is not "
                f"supported by VectorOPKernel.\n"
                f"Supported ops: {sorted(_ONNX_OP_MAP)}"
            )

        op_code, arity = _ONNX_OP_MAP[op_type]

        # Resolve input tensors (skip empty optional inputs)
        inputs = []
        for inp_name in node.input:
            if inp_name == "":
                continue
            if inp_name not in tensors:
                raise SchedulerError(
                    f"Tensor '{inp_name}' referenced by node "
                    f"'{node.name or op_type}' was not found in the graph."
                )
            inputs.append(tensors[inp_name])

        # Resolve output tensor
        if not node.output or node.output[0] == "":
            raise SchedulerError(
                f"Node '{node.name or op_type}' has no output tensor."
            )
        out_name = node.output[0]
        if out_name not in tensors:
            raise SchedulerError(
                f"Output tensor '{out_name}' of node '{node.name or op_type}' "
                f"was not found in the graph."
            )
        output = tensors[out_name]

        sn = cls(
            onnx_node=node,
            op_code=op_code,
            arity=arity,
            inputs=inputs,
            output=output,
            index=index,
            align_elems=align_elems,
        )
        sn.validate()
        return sn

    # ------------------------------------------------------------------ #
    # Validation                                                           #
    # ------------------------------------------------------------------ #

    def validate(self) -> None:
        op_type = self.onnx_node.op_type

        # Clip: only allow (min=0, max=6)
        if op_type == "Clip":
            mn, mx = _get_clip_bounds(self.onnx_node)
            if mn is None and len(self.inputs) > 1:
                mn_tensor = self.inputs[1]
                if mn_tensor.is_weight and mn_tensor.data is not None:
                    mn = float(mn_tensor.data.flat[0])
            if mx is None and len(self.inputs) > 2:
                mx_tensor = self.inputs[2]
                if mx_tensor.is_weight and mx_tensor.data is not None:
                    mx = float(mx_tensor.data.flat[0])

            if mn != 0.0 or mx != 6.0:
                raise SchedulerError(
                    f"Clip node '{self.onnx_node.name}' has bounds "
                    f"(min={mn}, max={mx}). Only Clip(0, 6) maps to "
                    f"VECTOROP_RELU6."
                )
            self.inputs = [self.inputs[0]]
            self.arity  = 1

        # Binary ops need exactly 2 inputs; unary need exactly 1
        n_data = len(self.inputs)
        if self.arity == 2 and n_data != 2:
            raise SchedulerError(
                f"Binary op '{op_type}' in node '{self.onnx_node.name}' "
                f"requires 2 inputs, got {n_data}."
            )
        if self.arity == 1 and n_data != 1:
            raise SchedulerError(
                f"Unary op '{op_type}' in node '{self.onnx_node.name}' "
                f"requires 1 input, got {n_data}."
            )

        if self.arity == 1:
            # Unary ops: input must match output exactly — no broadcasting.
            t = self.inputs[0]
            if t.numel != self.output.numel:
                raise SchedulerError(
                    f"Shape mismatch in unary node "
                    f"'{self.onnx_node.name or op_type}': "
                    f"input '{t.onnx_name}' has {t.numel} elements but output "
                    f"'{self.output.onnx_name}' has {self.output.numel} elements."
                )
            self.outer_count        = 1
            self.chunk_size         = self.output.numel
            self.aligned_chunk_size = self.output.numel
            self.a_advances         = True
            self.b_advances         = True

        else:
            # Binary ops: check each input for trailing-contiguous broadcast.
            ae = self.align_elems
            a_outer, _, _, a_advances = _broadcast_info(self.inputs[0], self.output, ae)
            b_outer, _, _, b_advances = _broadcast_info(self.inputs[1], self.output, ae)

            if not a_advances and not b_advances:
                raise SchedulerError(
                    f"Binary node '{self.onnx_node.name or op_type}': "
                    f"both inputs are smaller than the output — "
                    f"only one input may broadcast per operation."
                )

            outer_count = max(a_outer, b_outer)
            chunk_size  = self.output.numel // outer_count
            aligned_chunk_size = (
                (chunk_size + ae - 1) & ~(ae - 1)
            )

            self.outer_count        = outer_count
            self.chunk_size         = chunk_size
            self.aligned_chunk_size = aligned_chunk_size
            self.a_advances         = a_advances
            self.b_advances         = b_advances

    # ------------------------------------------------------------------ #
    # Code emission                                                        #
    # ------------------------------------------------------------------ #

    def emit_comment(self) -> str:
        in_names  = ", ".join(t.onnx_name for t in self.inputs)
        broadcast = (
            f"  (broadcast \u00d7{self.outer_count})"
            if self.outer_count > 1 else ""
        )
        fused = "".join(f" + {n.op_type}" for n in self.fused_nodes)
        return (
            f"    /* [{self.index}] {self.onnx_node.op_type}{fused}"
            f"({in_names}) -> {self.output.onnx_name}"
            f"  shape={self.output.shape}{broadcast} */"
        )

    def emit_call(self, layouts: dict) -> str:
        """
        Emit the C kernel call(s) for this node.

        Cache sync is handled entirely by the caller (inference_run):
          - sync_to_device for graph inputs at the top of inference_run()
          - sync_from_device for graph outputs at the bottom of inference_run()
          - weights were flushed once in inference_init()
          - internal (kernel-to-kernel) buffers need no sync

        layouts is {onnx_name: TensorLayout}.  For non-broadcast nodes the
        output's alloc_size is used as op_size so run_op() covers the full
        padded buffer when the output inherited stride from upstream.
        """
        a  = self.inputs[0].c_name
        b  = self.inputs[1].c_name if self.arity == 2 else "NULL"
        c  = self.output.c_name
        op = OP_NAMES[self.op_code]
        # run_op() programs act = VECTOROP_ACT_NONE; a node with a fused
        # activation goes through run_op_act() with the act code appended.
        fn  = "run_op" if self.act == ACT_NONE else "run_op_act"
        act = "" if self.act == ACT_NONE else f", {ACT_NAMES[self.act]}"

        if self.outer_count == 1:
            y_lay = layouts.get(self.output.onnx_name)
            size  = y_lay.alloc if y_lay is not None else self.chunk_size
            return f"    {fn}({a}, {b}, {c}, {size}u, {op}, 1u, 0u, 0u{act});"

        # Broadcasting: single run_op() call with outer, a_inc, b_inc.
        # VectorOPKernel handles the outer loop internally, advancing a/b/c
        # by a_inc/b_inc/c_inc (c_inc = a_inc+b_inc) per outer iteration.
        # The chunk stride is INFERENCE_ALIGN_UP(CHUNK) so every inner burst
        # starts at an INFERENCE_ALIGN_BYTES-aligned physical address.
        chunk_macro  = f"INFERENCE_{c.upper()}_CHUNK"
        stride_macro = f"INFERENCE_{c.upper()}_CHUNK_STRIDE"
        n     = self.outer_count
        a_inc = stride_macro if self.a_advances else "0u"
        b_inc = (stride_macro if self.b_advances else "0u") \
                if self.arity == 2 else "0u"
        return (
            f"    {fn}({a}, {b}, {c},"
            f" {chunk_macro}, {op},"
            f" {n}u, {a_inc}, {b_inc}{act});"
        )


# ------------------------------------------------------------------ #
# MatmulNode                                                           #
# ------------------------------------------------------------------ #

# ---------------------------------------------------------------------------
# Hardware-side bound — single source of truth is the platform JSON
# (platforms/<AXI_PLATFORM>.json, ``kernels.matmul`` object — see
# doc/MATMUL_KERNEL.md §3 for the field reference).  The resolver in
# ``_matmul_hw_config`` reads the same file the C++ CMake build consumes
# via ``matmul_load_constants()`` in kernels/matmul/CMakeLists.txt.
# ---------------------------------------------------------------------------
from ._matmul_hw_config import MATMUL_MAX_K, MATMUL_TILE_M  # noqa: E402


def matmul_packed_m(m: int) -> int:
    """m padded to a multiple of the kernel's kTileM (packed-B layout)."""
    return -(-m // MATMUL_TILE_M) * MATMUL_TILE_M


def _pack_matmul_b(t: TensorInfo, k: int, m: int) -> None:
    """Emit a constant B in MatmulKernel's tile-major packed layout
    (MatmulKernel.h "Packed (tile-major) B layout", MATMUL_OPTIMISATION §3b):

        packed[s][(mt * k + kk) * kTileM + m1] = B[s][kk][mt * kTileM + m1]

    for every leading batch slice s, with m zero-padded to matmul_packed_m(m),
    so a (m_tile, k_tile) block is one contiguous DDR run.  `data` / `shape`
    stay logical for the simulator; the ROM / .dat / DMA size follow the
    packed image (TensorInfo.numel / emit_data).
    """
    tm  = MATMUL_TILE_M
    pm  = matmul_packed_m(m)
    b   = np.asarray(t.data).reshape(-1, k, m)          # [slices][k][m]
    sl  = b.shape[0]
    mt  = pm // tm
    packed = np.zeros((sl, mt, k, tm), dtype=b.dtype)
    for i in range(mt):
        c0, c1 = i * tm, min(m, (i + 1) * tm)
        packed[:, i, :, :c1 - c0] = b[:, :, c0:c1]
    t.packed_data = np.ascontiguousarray(packed).reshape(-1)
    t.packed_note = (f"MatmulKernel tile-major B [slices][ceil(M/{tm})][K][{tm}]"
                     f" = [{sl}][{mt}][{k}][{tm}] (M {m} -> {pm})")


@dataclass
class MatmulNode:
    """One ONNX MatMul operator mapped to one or more XMatmulkernel invocations.

    kernel_name identifies this node's hardware kernel for the registry lookup.

    Supports four shapes:

      2-D   A[N,K]        @ B[K,M]           → Y[N,M]
      3-D   A[b,N,K]      @ B[b or 1,K,M]    → Y[b,N,M]   (B may broadcast)
      4-D   A[b1,b2,N,K]  @ B[b1,b2,K,M]     → Y[b1,b2,N,M]   (flat batch)
      4D×3D A[b1,b2,N,K]  @ B[b2,K,M]        → Y[b1,b2,N,M]   (outer loop)

    The 4D×3D case uses an outer loop of b1 calls to XMatmulkernel (each with
    batch=b2), emitting run_matmul_at() with element offsets per iteration.

    Fields
    ------
    n, k, m, batch          inner matrix dimensions and inner batch count
    a/b/c_batch_stride      element stride inside each XMatmulkernel call
    outer_count             outer loop iterations (1 for 2-D/3-D/flat-4-D)
    a/b/c_outer_stride      element offset advance per outer iteration
                            (b_outer_stride == 0 when B has no outer dim)

    Compatibility fields (always fixed):
    chunk_size, aligned_chunk_size, a_advances, b_advances, arity
    """

    kernel_name: ClassVar[str] = "MatmulKernel"

    onnx_node:      onnx.NodeProto
    inputs:         List[TensorInfo]   # [A, B]
    output:         TensorInfo
    index:          int = 0
    align_elems:    int = 8

    # Inner matrix dimensions (used in every kernel call)
    n:              int = 0   # output rows        (A.shape[-2])
    k:              int = 0   # inner dimension    (A.shape[-1] == B.shape[-2])
    m:              int = 0   # output columns     (B.shape[-1])
    batch:          int = 1   # inner batch count  (1 for 2-D; b for 3-D/4-D)
    a_batch_stride: int = 0   # A elements between consecutive inner-batch slices
    b_batch_stride: int = 0   # B elements between consecutive inner-batch slices
    c_batch_stride: int = 0   # Y elements between consecutive inner-batch slices

    # Outer loop parameters (set when A has a leading batch dim absent from B)
    outer_count:    int = 1   # outer loop iterations; 1 means a single call
    a_outer_stride: int = 0   # A elements advanced per outer iteration
    b_outer_stride: int = 0   # B elements advanced per outer iteration (0 if B broadcasts)
    c_outer_stride: int = 0   # Y elements advanced per outer iteration
    # B layout (MATMUL_OPTIMISATION §3b): True when this node's B was emitted
    # in the kernel's tile-major packed layout (OnnxGraph packs a constant B
    # after node creation, see pack_b()); b_batch_stride / b_outer_stride are
    # then counts in the PACKED image.
    b_packed:       bool = False

    # Compatibility shims for _compute_alloc_sizes / _broadcast_io_map.
    # These are always derived constants — never set by callers.
    chunk_size:         int  = field(default=0,    init=False)
    aligned_chunk_size: int  = field(default=0,    init=False)
    a_advances:         bool = field(default=True, init=False)
    b_advances:         bool = field(default=True, init=False)
    arity:              int  = field(default=2,    init=False)

    # ------------------------------------------------------------------ #
    # Factory                                                              #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_onnx_node(
        cls,
        node:        onnx.NodeProto,
        tensors:     dict,
        index:       int,
        align_elems: int = 8,
    ) -> "MatmulNode":
        if node.op_type != "MatMul":
            raise SchedulerError(
                f"MatmulNode.from_onnx_node() called with op_type='{node.op_type}'"
            )

        # Resolve inputs
        if len(node.input) != 2:
            raise SchedulerError(
                f"MatMul node '{node.name}' must have exactly 2 inputs, "
                f"got {len(node.input)}."
            )
        a_name, b_name = node.input[0], node.input[1]
        for tname in (a_name, b_name):
            if tname not in tensors:
                raise SchedulerError(
                    f"Tensor '{tname}' referenced by MatMul node "
                    f"'{node.name}' not found in graph."
                )
        a_info = tensors[a_name]
        b_info = tensors[b_name]

        # Resolve output
        if not node.output or node.output[0] == "":
            raise SchedulerError(f"MatMul node '{node.name}' has no output.")
        out_name = node.output[0]
        if out_name not in tensors:
            raise SchedulerError(
                f"Output tensor '{out_name}' of MatMul node '{node.name}' "
                f"not found in graph."
            )
        out_info = tensors[out_name]

        # Derive n, k, m, batch, and strides from shapes using ONNX
        # numpy-style batched matmul broadcasting.
        a_shape = tuple(a_info.shape)
        b_shape = tuple(b_info.shape)

        a_rank = len(a_shape)
        b_rank = len(b_shape)

        if a_rank < 2 or b_rank < 2 or a_rank > 5 or b_rank > 5:
            raise SchedulerError(
                f"MatMul node '{node.name}': rank 2–5 supported, "
                f"got A{list(a_shape)} @ B{list(b_shape)}."
            )

        n_val = a_shape[-2]
        k_val = a_shape[-1]
        m_val = b_shape[-1]
        if b_shape[-2] != k_val:
            raise SchedulerError(
                f"MatMul shape mismatch in node '{node.name}': "
                f"A K={k_val} != B K={b_shape[-2]} "
                f"in A{list(a_shape)} @ B{list(b_shape)}."
            )

        # ------------------------------------------------------------------
        # Hardware-bound validation — see _matmul_hw_config for where this
        # constant comes from (platforms/<AXI_PLATFORM>.json `kernels.matmul`).
        # MatmulKernel sizes its row-staging buffer ``a_buf[kTileN][kMaxK]``
        # at compile time, so a matmul with inner-dim k > kMaxK has no
        # runtime fallback — reject at parse time rather than emit code the
        # kernel can't service.  n / m / batch stay unbounded — the kernel
        # tiles them naturally.  Reference: doc/MATMUL_KERNEL.md §3.
        # ------------------------------------------------------------------
        if k_val > MATMUL_MAX_K:
            raise SchedulerError(
                f"MatMul node '{node.name or 'MatMul'}': inner-dim k="
                f"{k_val} exceeds MatmulKernel's compile-time limit "
                f"kMaxK={MATMUL_MAX_K}.  Raise 'kernels.matmul.max_k' in "
                f"the platform JSON (platforms/<AXI_PLATFORM>.json) and "
                f"rebuild."
            )

        a_batch = a_shape[:-2]   # leading batch dims of A; () when A is 2-D
        b_batch = b_shape[:-2]   # leading batch dims of B; () when B is 2-D

        # ---- Special cases: one input is 2-D (no batch dims) -------------
        # The 2-D side broadcasts across all batch dimensions of the other.
        # The kernel's batch loop handles this with a batch stride of 0 for
        # the 2-D operand; no outer loop is needed.

        if len(a_batch) == 0 and len(b_batch) == 0:
            # A[N,K] @ B[K,M] — plain matrix multiply
            return cls(
                onnx_node=node, inputs=[a_info, b_info], output=out_info,
                index=index, align_elems=align_elems,
                n=n_val, k=k_val, m=m_val,
                batch=1, a_batch_stride=0, b_batch_stride=0, c_batch_stride=0,
            )

        if len(b_batch) == 0:
            # A[*batch,N,K] @ B[K,M] — B broadcasts across all of A's batch dims
            batch = 1
            for d in a_batch:
                batch *= d
            return cls(
                onnx_node=node, inputs=[a_info, b_info], output=out_info,
                index=index, align_elems=align_elems,
                n=n_val, k=k_val, m=m_val,
                batch=batch,
                a_batch_stride=n_val * k_val,
                b_batch_stride=0,
                c_batch_stride=n_val * m_val,
            )

        if len(a_batch) == 0:
            # A[N,K] @ B[*batch,K,M] — A broadcasts across all of B's batch dims
            batch = 1
            for d in b_batch:
                batch *= d
            return cls(
                onnx_node=node, inputs=[a_info, b_info], output=out_info,
                index=index, align_elems=align_elems,
                n=n_val, k=k_val, m=m_val,
                batch=batch,
                a_batch_stride=0,
                b_batch_stride=k_val * m_val,
                c_batch_stride=n_val * m_val,
            )

        # ---- General case: both inputs have batch dims -------------------
        # Right-align batch dimensions and apply ONNX broadcast rules.
        #
        # Batch dims are split into two contiguous blocks:
        #
        #   outer block (leading)  — exactly one input has size 1; the other
        #                advances element-by-element, driving an outer loop.
        #                All outer dims must be from the SAME input.
        #
        #   shared block (trailing) — both inputs have the same (non-1) size.
        #                The kernel's built-in batch loop iterates this block.
        #
        # outer_count = product of leading (outer) output dims
        # batch       = product of trailing (shared) output dims
        #
        # outer_count == 1 → emit run_matmul()     (no outer loop)
        # outer_count >  1 → emit a for-loop calling run_matmul_at()

        max_len = max(len(a_batch), len(b_batch))
        a_ext   = (1,) * (max_len - len(a_batch)) + a_batch
        b_ext   = (1,) * (max_len - len(b_batch)) + b_batch

        # Validate broadcastability and compute the output batch shape.
        out_batch: list = []
        for ad, bd in zip(a_ext, b_ext, strict=True):
            if ad == bd:
                out_batch.append(ad)
            elif ad == 1:
                out_batch.append(bd)
            elif bd == 1:
                out_batch.append(ad)
            else:
                raise SchedulerError(
                    f"MatMul '{node.name}': batch dims {list(a_batch)} and "
                    f"{list(b_batch)} are not broadcastable."
                )

        # Find the split between the outer block and the shared block.
        # outer_a_advances: True  → A is the advancing side (b_ext has 1s)
        #                   False → B is the advancing side (a_ext has 1s)
        outer_a_advances: Optional[bool] = None
        split = max_len   # tentative: all dims are outer

        for i, (ad, bd) in enumerate(zip(a_ext, b_ext, strict=True)):
            if ad > 1 and bd > 1:
                # Both present → shared block starts here
                split = i
                break
            if ad == 1 and bd == 1:
                # Degenerate size-1 on both sides → treat as shared
                split = i
                break
            # Exactly one is 1 → outer broadcast dim
            advances = (ad > 1)
            if outer_a_advances is None:
                outer_a_advances = advances
            elif outer_a_advances != advances:
                raise SchedulerError(
                    f"MatMul '{node.name}': mixed-direction outer broadcast "
                    f"in batch dims {list(a_batch)} vs {list(b_batch)} "
                    f"is not supported."
                )

        outer_count = 1
        for d in out_batch[:split]:
            outer_count *= d

        batch = 1
        for d in out_batch[split:]:
            batch *= d

        if batch <= 1:
            a_batch_stride = 0
            b_batch_stride = 0
            c_batch_stride = 0
        else:
            a_batch_stride = (n_val * k_val
                              if any(a_ext[i] > 1 for i in range(split, max_len))
                              else 0)
            b_batch_stride = (k_val * m_val
                              if any(b_ext[i] > 1 for i in range(split, max_len))
                              else 0)
            c_batch_stride = n_val * m_val

        if outer_count <= 1:
            return cls(
                onnx_node=node, inputs=[a_info, b_info], output=out_info,
                index=index, align_elems=align_elems,
                n=n_val, k=k_val, m=m_val,
                batch=batch,
                a_batch_stride=a_batch_stride,
                b_batch_stride=b_batch_stride,
                c_batch_stride=c_batch_stride,
            )

        # Outer loop — the advancing side steps through outer-count blocks;
        # the other side repeats at offset 0 every iteration.
        if outer_a_advances:
            a_outer_stride = batch * n_val * k_val
            b_outer_stride = 0
        else:
            a_outer_stride = 0
            b_outer_stride = batch * k_val * m_val
        c_outer_stride = batch * n_val * m_val

        return cls(
            onnx_node=node, inputs=[a_info, b_info], output=out_info,
            index=index, align_elems=align_elems,
            n=n_val, k=k_val, m=m_val,
            batch=batch,
            a_batch_stride=a_batch_stride,
            b_batch_stride=b_batch_stride,
            c_batch_stride=c_batch_stride,
            outer_count=outer_count,
            a_outer_stride=a_outer_stride,
            b_outer_stride=b_outer_stride,
            c_outer_stride=c_outer_stride,
        )

    # ------------------------------------------------------------------ #
    # Code emission                                                        #
    # ------------------------------------------------------------------ #

    def emit_comment(self) -> str:
        a = self.inputs[0].onnx_name
        b = self.inputs[1].onnx_name
        batch_str = f", batch={self.batch}" if self.batch > 1 else ""
        outer_str = (
            f"  (outer_loop\u00d7{self.outer_count})"
            if self.outer_count > 1 else ""
        )
        return (
            f"    /* [{self.index}] MatMul({a}, {b}) -> {self.output.onnx_name}"
            f"  [{self.n},{self.k}]x[{self.k},{self.m}]->[{self.n},{self.m}]"
            f"{batch_str}{outer_str} */"
        )

    def pack_b(self) -> None:
        """Switch this node to the packed B layout (the tensor image itself is
        packed once by OnnxGraph; every node sharing it calls this)."""
        if self.b_packed:
            return
        scale   = matmul_packed_m(self.m) * self.k       # packed slice size
        logical = self.k * self.m
        # B strides are whole (k x m) slices; rescale them to packed slices.
        if self.b_batch_stride:
            assert self.b_batch_stride % logical == 0
            self.b_batch_stride = self.b_batch_stride // logical * scale
        if self.b_outer_stride:
            assert self.b_outer_stride % logical == 0
            self.b_outer_stride = self.b_outer_stride // logical * scale
        self.b_packed = True

    def emit_call(self, layouts: dict) -> str:
        """Emit the run_matmul() or run_matmul_at() call for this node.

        layouts is {onnx_name: TensorLayout}.

        When the A input or Y output DMA buffer has alignment gaps
        (layout.gap > 0), the kernel is called with batch=N, n=1 so the
        batch stride acts as a per-row stride, allowing MatmulKernel to skip
        alignment gaps between rows.
        """
        a = self.inputs[0].c_name
        b = self.inputs[1].c_name
        c = self.output.c_name

        if self.outer_count == 1:
            a_lay = layouts.get(self.inputs[0].onnx_name)
            y_lay = layouts.get(self.output.onnx_name)
            a_row_stride = a_lay.stride if (a_lay and a_lay.gap > 0) else 0
            c_row_stride = y_lay.stride if (y_lay and y_lay.gap > 0) else 0

            if a_row_stride != 0 or c_row_stride != 0:
                # Row-strided decomposition: N calls of (1 × K) × (K × M).
                # a_batch_stride = a_row_stride (or natural K if only c is strided)
                # c_batch_stride = c_row_stride (or natural M if only a is strided)
                eff_a = a_row_stride if a_row_stride != 0 else self.k
                eff_c = c_row_stride if c_row_stride != 0 else self.m
                return (
                    f"    run_matmul({a}, {b}, {c},\n"
                    f"               1u, {self.k}u, {self.m}u, {self.n}u,\n"
                    f"               {eff_a}u, {self.b_batch_stride}u, {eff_c}u,"
                    f" {int(self.b_packed)}u);"
                )
            return (
                f"    run_matmul({a}, {b}, {c},\n"
                f"               {self.n}u, {self.k}u, {self.m}u, {self.batch}u,\n"
                f"               {self.a_batch_stride}u,"
                f" {self.b_batch_stride}u,"
                f" {self.c_batch_stride}u, {int(self.b_packed)}u);"
            )

        # 4D×3D broadcasting: outer loop over the b1 dimension.
        # A and Y advance by one b2-block per iteration; B repeats at offset 0.
        return "\n".join([
            f"    for (unsigned _i = 0u; _i < {self.outer_count}u; _i++) {{",
            f"        run_matmul_at({a}, _i * {self.a_outer_stride}u,",
            f"                      {b}, _i * {self.b_outer_stride}u,",
            f"                      {c}, _i * {self.c_outer_stride}u,",
            f"                      {self.n}u, {self.k}u, {self.m}u, {self.batch}u,",
            f"                      {self.a_batch_stride}u, {self.b_batch_stride}u,"
            f" {self.c_batch_stride}u, {int(self.b_packed)}u);",
            "    }",
        ])


# ------------------------------------------------------------------ #
# ConvNode                                                             #
# ------------------------------------------------------------------ #

# ---------------------------------------------------------------------------
# Hardware-side bounds — single source of truth is the platform JSON
# (platforms/<AXI_PLATFORM>.json, ``kernels.conv`` object — see
# doc/CONV_KERNEL.md §3 for the field reference).  The resolver in
# ``_conv_hw_config`` reads the same file the C++ CMake build consumes via
# ``conv_load_constants()`` in kernels/conv/CMakeLists.txt, so a CLI
# invocation targeting a non-default board can stay in sync with
# ``cmake -DAXI_PLATFORM=<name>``.
# ---------------------------------------------------------------------------
from ._conv_hw_config import (  # noqa: E402
    CONV_MAX_IN_CH,
    CONV_MAX_OUT_CH,
    CONV_MAX_LINE_BUF_ROWS,
    CONV_MAX_LINE_BUF_COLS,
    CONV_MAX_ACC_PERSIST_ENTRIES,
    CONV_TILE_M,
    CONV_TILE_IC,
    CONV_WEIGHT_PORT_ELEMS,
)


# ---------------------------------------------------------------------------
# ConvKernel packed DDR layouts (ConvKernel.h, "Weight / bias port width and
# DDR layout", CONV_OPTIMISATION.md §2.32).  The kernel's 128-bit weight port
# fills one 16-input-channel-lane cache word per beat pair, so the lanes of
# a kernel position must be adjacent in DDR:
#
#   standard : weight[M][ceil(C/16)][kH][kW][lanes]  — 16 lanes per tile,
#              except the LAST tile is a half tile of 8 lanes when it has
#              <= 8 valid channels (§2.34); lanes >= C are zero
#   depthwise: weight[M][roundup(kH*kW, 8)]
#   bias     : bias[roundup(M, 8)]
#
# The logical ONNX arrays stay on TensorInfo.data (the simulator uses them);
# only the emitted ROM / .dat image and the DMA buffer size are packed.
# ---------------------------------------------------------------------------
def _pack_conv_weight(t: TensorInfo, out_ch: int, in_ch: int,
                      kh: int, kw: int, is_depthwise: bool) -> None:
    w = np.asarray(t.data)
    lanes = CONV_WEIGHT_PORT_ELEMS
    if is_depthwise:
        stride = -(-(kh * kw) // lanes) * lanes
        packed = np.zeros((out_ch, stride), dtype=w.dtype)
        packed[:, :kh * kw] = w.reshape(out_ch, kh * kw)
        t.packed_note = (f"ConvKernel depthwise [M][roundup(kH*kW,{lanes})]"
                         f" = [{out_ch}][{stride}]")
    else:
        tile     = CONV_TILE_IC
        ic_tiles = -(-in_ch // tile)
        last_rem   = in_ch - (ic_tiles - 1) * tile          # 1..tile
        last_lanes = lanes if last_rem <= lanes else tile   # §2.34 half tile
        w4 = w.reshape(out_ch, in_ch, kh, kw)
        per_m = kh * kw * ((ic_tiles - 1) * tile + last_lanes)
        packed = np.zeros((out_ch, per_m), dtype=w.dtype)
        for ict in range(ic_tiles):
            t_lanes = last_lanes if ict == ic_tiles - 1 else tile
            c0, c1 = ict * tile, min(in_ch, (ict + 1) * tile)
            blk = np.zeros((out_ch, kh, kw, t_lanes), dtype=w.dtype)
            # [M][c][kH][kW] -> [M][kH][kW][ic_l]
            blk[:, :, :, :c1 - c0] = w4[:, c0:c1].transpose(0, 2, 3, 1)
            base = ict * kh * kw * tile
            packed[:, base:base + kh * kw * t_lanes] = blk.reshape(out_ch, -1)
        t.packed_note = (f"ConvKernel tile-major [M][ceil(C/{tile})][kH][kW][lanes]"
                         f" = [{out_ch}][{ic_tiles}][{kh}][{kw}][{tile}|{last_lanes} last]")
    t.packed_data = np.ascontiguousarray(packed).reshape(-1)


def _pad_conv_bias(t: TensorInfo, out_ch: int) -> None:
    lanes = CONV_WEIGHT_PORT_ELEMS
    b = np.asarray(t.data).reshape(-1)
    n = -(-out_ch // lanes) * lanes
    packed = np.zeros(n, dtype=b.dtype)
    packed[:out_ch] = b[:out_ch]
    t.packed_data = packed
    t.packed_note = f"ConvKernel bias [roundup(M,{lanes})] = [{n}]"


@dataclass
class ConvNode:
    """One ONNX Conv operator mapped to one XConvkernel invocation.

    Supports 2-D convolution with NCHW layout:
      x      [N, C, H, W]  — input feature map
      weight [M, C, kH, kW] — filters (M output channels)
      bias   [M]            — optional per-channel bias
      y      [N, M, oH, oW] — output feature map

    auto_pad modes supported: NOTSET, VALID, SAME_UPPER, SAME_LOWER.
    Only dilations == [1,1] are strictly needed (ConvKernel supports arbitrary
    dilation, so we pass whatever ONNX specifies).

    The output is always NCHW-flat.  ConvNode is excluded from phases 2 and 3
    of _compute_tensor_layouts() so that broadcast VectorOP nodes downstream
    of a Conv never see advancing-strided buffers that ConvKernel didn't fill.
    Any graph that requires broadcasting immediately after a Conv (other than
    the Conv's own bias) must use the Conv's built-in bias input, not a
    separate Add node.

    Compatibility shims (init=False)
    ---------------------------------
    outer_count, chunk_size, aligned_chunk_size, a_advances, b_advances, arity
    — set to neutral values so that _compute_tensor_layouts() phase 2/3 exclusion
    guards (isinstance checks) are the only gating logic needed.
    """

    kernel_name: ClassVar[str] = "ConvKernel"

    onnx_node:   onnx.NodeProto
    inputs:      List[TensorInfo]   # [x, weight] or [x, weight, bias]
    output:      TensorInfo
    index:       int = 0
    align_elems: int = 8

    # Conv geometry (derived from ONNX attributes + shape inference)
    batch:       int = 1
    in_ch:       int = 0
    in_h:        int = 0
    in_w:        int = 0
    out_ch:      int = 0
    out_h:       int = 0
    out_w:       int = 0
    kh:          int = 1
    kw:          int = 1
    stride_h:    int = 1
    stride_w:    int = 1
    dilation_h:  int = 1
    dilation_w:  int = 1
    pad_top:     int = 0
    pad_left:    int = 0
    has_bias:    bool = False
    is_depthwise: bool = False

    # Compatibility shims — never set by callers.
    outer_count:        int  = field(default=1,    init=False)
    chunk_size:         int  = field(default=0,    init=False)
    aligned_chunk_size: int  = field(default=0,    init=False)
    a_advances:         bool = field(default=True, init=False)
    b_advances:         bool = field(default=True, init=False)
    arity:              int  = field(default=2,    init=False)

    # ------------------------------------------------------------------ #
    # Factory                                                              #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_onnx_node(
        cls,
        node:        onnx.NodeProto,
        tensors:     dict,
        index:       int,
        align_elems: int = 8,
    ) -> "ConvNode":
        if node.op_type != "Conv":
            raise SchedulerError(
                f"ConvNode.from_onnx_node() called with op_type='{node.op_type}'"
            )

        # ONNX Conv has 2 required inputs (x, W) and 1 optional input (B).
        n_inputs = len([nm for nm in node.input if nm != ""])
        if n_inputs not in (2, 3):
            raise SchedulerError(
                f"Conv node '{node.name}' must have 2 or 3 inputs, got {n_inputs}."
            )

        for tname in node.input[:n_inputs]:
            if tname == "":
                continue
            if tname not in tensors:
                raise SchedulerError(
                    f"Tensor '{tname}' referenced by Conv node "
                    f"'{node.name}' not found in graph."
                )

        x_info  = tensors[node.input[0]]
        w_info  = tensors[node.input[1]]
        has_b   = n_inputs == 3 and node.input[2] != ""
        b_info  = tensors[node.input[2]] if has_b else None

        if not node.output or node.output[0] == "":
            raise SchedulerError(f"Conv node '{node.name}' has no output.")
        out_name = node.output[0]
        if out_name not in tensors:
            raise SchedulerError(
                f"Output tensor '{out_name}' of Conv node '{node.name}' "
                f"not found in graph."
            )
        y_info = tensors[out_name]

        # --- Validate input/weight shapes ---
        if len(x_info.shape) != 4:
            raise SchedulerError(
                f"Conv node '{node.name}': input must be 4-D (NCHW), "
                f"got shape {x_info.shape}."
            )
        if len(w_info.shape) != 4:
            raise SchedulerError(
                f"Conv node '{node.name}': weight must be 4-D (M,C/g,kH,kW), "
                f"got shape {w_info.shape}."
            )
        if len(y_info.shape) != 4:
            raise SchedulerError(
                f"Conv node '{node.name}': output must be 4-D (NCHW), "
                f"got shape {y_info.shape}."
            )

        # --- Parse ONNX attributes ---
        attrs = {a.name: a for a in node.attribute}

        def _int_list(name: str, default: List[int]) -> List[int]:
            return (list(attrs[name].ints) if name in attrs else default)

        n_val, c_in, h_in, w_in = x_info.shape
        m_val, c_w, kh_val, kw_val = w_info.shape
        _, _, h_out, w_out = y_info.shape

        group = attrs["group"].i if "group" in attrs else 1

        # Validate group and determine mode.
        # group=1      → standard conv: weight[M, C, kH, kW], c_w == c_in.
        # group=in_ch  → depthwise conv: weight[C, 1, kH, kW], c_w == 1, M == C.
        # other values → not supported.
        if group == 1:
            is_dw = False
            if c_w != c_in:
                raise SchedulerError(
                    f"Conv node '{node.name}': weight in_channels={c_w} "
                    f"!= input in_channels={c_in}."
                )
        elif group == c_in:
            is_dw = True
            if c_w != 1:
                raise SchedulerError(
                    f"Conv node '{node.name}': depthwise conv (group={group}) "
                    f"weight must have 1 channel per filter (C/group=1), "
                    f"got {c_w}."
                )
            if m_val != c_in:
                raise SchedulerError(
                    f"Conv node '{node.name}': depthwise conv (group={group}) "
                    f"requires out_ch ({m_val}) == in_ch ({c_in})."
                )
        else:
            raise SchedulerError(
                f"Conv node '{node.name}': grouped convolution (group={group}) "
                f"is not supported by ConvKernel. "
                f"Only standard (group=1) and depthwise (group=in_ch) "
                f"convolutions are supported."
            )

        if m_val != y_info.shape[1]:
            raise SchedulerError(
                f"Conv node '{node.name}': weight out_channels={m_val} "
                f"!= output channels={y_info.shape[1]}."
            )

        strides    = _int_list("strides",   [1, 1])
        dilations  = _int_list("dilations", [1, 1])
        pads_attr  = _int_list("pads",      [0, 0, 0, 0])  # [top,left,bottom,right]
        auto_pad   = (attrs["auto_pad"].s.decode("utf-8")
                      if "auto_pad" in attrs else "NOTSET")

        if len(strides) != 2:
            raise SchedulerError(
                f"Conv node '{node.name}': only 2-D strides supported, got {strides}."
            )
        if len(dilations) != 2:
            raise SchedulerError(
                f"Conv node '{node.name}': only 2-D dilations supported, "
                f"got {dilations}."
            )

        sh, sw = strides
        dh, dw = dilations

        # --- Derive padding ---
        if auto_pad == "NOTSET":
            pad_top_val  = pads_attr[0]
            pad_left_val = pads_attr[1]
        elif auto_pad == "VALID":
            pad_top_val  = 0
            pad_left_val = 0
        elif auto_pad in ("SAME_UPPER", "SAME_LOWER"):
            # Effective kernel size
            eff_kh = dh * (kh_val - 1) + 1
            eff_kw = dw * (kw_val - 1) + 1
            pad_h = max(0, (h_out - 1) * sh + eff_kh - h_in)
            pad_w = max(0, (w_out - 1) * sw + eff_kw - w_in)
            if auto_pad == "SAME_UPPER":
                pad_top_val  = pad_h // 2
                pad_left_val = pad_w // 2
            else:
                pad_top_val  = (pad_h + 1) // 2
                pad_left_val = (pad_w + 1) // 2
        else:
            raise SchedulerError(
                f"Conv node '{node.name}': unsupported auto_pad='{auto_pad}'."
            )

        inputs_list: List[TensorInfo] = [x_info, w_info]
        if has_b and b_info is not None:
            inputs_list.append(b_info)

        # ------------------------------------------------------------------
        # Hardware-bound validation — see _conv_hw_config for where these
        # constants come from (platforms/<AXI_PLATFORM>.json `kernels.conv`).
        # ConvKernel sizes its bias buffer, line buffer and persistent
        # accumulator at compile time, so a layer violating any of these
        # bounds has no runtime fallback — reject at parse time rather than
        # emit code the kernel can't service.  Reference: doc/CONV_KERNEL.md
        # §3 "Runtime constraints validated by the inference scheduler".
        #
        # in_h, in_w, and out_h are NOT capped — the kernel handles them
        # via transparent tiling (ow-tiling, oh-chunking, M-grouping).
        # ------------------------------------------------------------------
        node_label = node.name or "Conv"
        if c_in > CONV_MAX_IN_CH:
            raise SchedulerError(
                f"Conv node '{node_label}': in_ch={c_in} exceeds "
                f"ConvKernel's compile-time limit "
                f"kMaxInCh={CONV_MAX_IN_CH}.  Raise "
                f"'kernels.conv.max_in_ch' in the platform JSON "
                f"(platforms/<AXI_PLATFORM>.json) and rebuild."
            )
        if m_val > CONV_MAX_OUT_CH:
            raise SchedulerError(
                f"Conv node '{node_label}': out_ch={m_val} exceeds "
                f"ConvKernel's compile-time limit "
                f"kMaxOutCh={CONV_MAX_OUT_CH}.  Raise "
                f"'kernels.conv.max_out_ch' in the platform JSON "
                f"(platforms/<AXI_PLATFORM>.json) and rebuild."
            )
        v_span = (kh_val - 1) * dh + 1
        if v_span > CONV_MAX_LINE_BUF_ROWS:
            raise SchedulerError(
                f"Conv node '{node_label}': dilated vertical span "
                f"(kh-1)*dilation_h + 1 = ({kh_val}-1)*{dh} + 1 = "
                f"{v_span} exceeds line-buffer row capacity "
                f"kMaxLineBufRows={CONV_MAX_LINE_BUF_ROWS} (one "
                f"kernel-height window must fit).  Reduce dilation_h "
                f"or raise 'kernels.conv.max_line_buf_rows' in the "
                f"platform JSON (must remain a power of 2)."
            )
        h_span = (kw_val - 1) * dw + 1
        if h_span > CONV_MAX_LINE_BUF_COLS:
            raise SchedulerError(
                f"Conv node '{node_label}': dilated horizontal span "
                f"(kw-1)*dilation_w + 1 = ({kw_val}-1)*{dw} + 1 = "
                f"{h_span} exceeds line-buffer column capacity "
                f"kMaxLineBufCols={CONV_MAX_LINE_BUF_COLS} (one "
                f"kernel-width window must fit; in_w wider than this "
                f"is auto-tiled along ow).  Reduce dilation_w or "
                f"raise 'kernels.conv.max_line_buf_cols' in the "
                f"platform JSON (must remain a power of 2)."
            )
        # The kernel's persistent accumulator stores out_ch padded up to a
        # multiple of kTileM (one aligned word per m-tile), so the row
        # that must fit is out_w * ceil(out_ch/kTileM)*kTileM, not
        # out_w * out_ch.  Mirrors compute_oh_chunking() in ConvKernel.cpp.
        m_padded    = -(-m_val // CONV_TILE_M) * CONV_TILE_M
        row_entries = w_out * m_padded
        if row_entries > CONV_MAX_ACC_PERSIST_ENTRIES:
            raise SchedulerError(
                f"Conv node '{node_label}': out_w*out_ch (padded to the "
                f"kTileM={CONV_TILE_M} tile) = {w_out}*{m_padded} = "
                f"{row_entries} exceeds persistent accumulator capacity "
                f"kMaxAccPersistEntries={CONV_MAX_ACC_PERSIST_ENTRIES} "
                f"(one padded output row must fit; taller out_h is "
                f"auto-chunked along oh).  Raise "
                f"'kernels.conv.max_acc_persist_entries' in the platform "
                f"JSON (each 4096 entries spends one URAM block)."
            )

        # §2.32: emit the weight / bias in the kernel's packed DDR layout.
        _pack_conv_weight(w_info, m_val, c_in, kh_val, kw_val, is_dw)
        if has_b:
            _pad_conv_bias(b_info, m_val)

        return cls(
            onnx_node=node,
            inputs=inputs_list,
            output=y_info,
            index=index,
            align_elems=align_elems,
            batch=n_val,
            in_ch=c_in,
            in_h=h_in,
            in_w=w_in,
            out_ch=m_val,
            out_h=h_out,
            out_w=w_out,
            kh=kh_val,
            kw=kw_val,
            stride_h=sh,
            stride_w=sw,
            dilation_h=dh,
            dilation_w=dw,
            pad_top=pad_top_val,
            pad_left=pad_left_val,
            has_bias=has_b,
            is_depthwise=is_dw,
        )

    # ------------------------------------------------------------------ #
    # Code emission                                                        #
    # ------------------------------------------------------------------ #

    def emit_comment(self) -> str:
        x_name = self.inputs[0].onnx_name
        w_name = self.inputs[1].onnx_name
        bias_str = (
            f", bias={self.inputs[2].onnx_name}" if self.has_bias else ""
        )
        k_str = (
            f"{self.kh}x{self.kw}"
            if self.kh != self.kw else f"{self.kh}"
        )
        dw_str = " dw" if self.is_depthwise else ""
        return (
            f"    /* [{self.index}] Conv({x_name}, {w_name}{bias_str})"
            f" -> {self.output.onnx_name}"
            f"  [{self.batch},{self.in_ch},{self.in_h},{self.in_w}]"
            f"→[{self.batch},{self.out_ch},{self.out_h},{self.out_w}]"
            f"  k={k_str}"
            f" s={self.stride_h}x{self.stride_w}"
            f" p={self.pad_top},{self.pad_left}{dw_str} */"
        )

    def emit_call(self, layouts: dict) -> str:  # noqa: ARG002
        x      = self.inputs[0].c_name
        weight = self.inputs[1].c_name
        bias   = self.inputs[2].c_name if self.has_bias else "NULL"
        y      = self.output.c_name
        hb     = "1u" if self.has_bias else "0u"
        idw    = "1u" if self.is_depthwise else "0u"
        return (
            f"    run_conv({x}, {weight}, {bias}, {y},\n"
            f"             {self.batch}u, {self.in_ch}u,"
            f" {self.in_h}u, {self.in_w}u,\n"
            f"             {self.out_ch}u, {self.out_h}u, {self.out_w}u,\n"
            f"             {self.kh}u, {self.kw}u,"
            f" {self.stride_h}u, {self.stride_w}u,\n"
            f"             {self.dilation_h}u, {self.dilation_w}u,"
            f" {self.pad_top}u, {self.pad_left}u, {hb}, {idw});"
        )


# ------------------------------------------------------------------ #
# PoolNode                                                             #
# ------------------------------------------------------------------ #

# Pool-type codes — must match kPoolMax / kPoolAvg / kPoolLp in
# pool/include/Config.h.in.
POOL_MAX = 0
POOL_AVG = 1
POOL_LP  = 2

_POOL_NAMES = {POOL_MAX: "MAX", POOL_AVG: "AVG", POOL_LP: "LP"}

_GLOBAL_POOL_OP_TYPES = frozenset({
    "GlobalMaxPool", "GlobalAveragePool", "GlobalLpPool",
})

POOL_OP_TYPES = frozenset({
    "MaxPool", "AveragePool", "LpPool",
}) | _GLOBAL_POOL_OP_TYPES


# ---------------------------------------------------------------------------
# Hardware-side bounds — re-exported here for callers that import them from
# `nodes`, but the source of truth is the C++ CMake configuration.  The
# resolver in `_pool_hw_config` parses kernels/pool/CMakeLists.txt for the
# `set(POOL_<NAME> <N> CACHE ...)` defaults and overlays any active
# build/CMakeCache.txt overrides, so `cmake -DPOOL_MAX_KH=...` flows through
# to this validator without manual edits.  See doc/POOL_OPTIMIZATION.md §4.
# ---------------------------------------------------------------------------
from ._pool_hw_config import (  # noqa: E402 (deferred until POOL_OP_TYPES is defined above)
    POOL_MAX_KH,
    POOL_MAX_KW,
    POOL_MAX_LINE_BUF_ROWS,
    POOL_MAX_LINE_BUF_COLS,
)


@dataclass
class PoolNode:
    """One ONNX pooling operator mapped to one XPoolingkernel invocation.

    Supports 2-D pooling with NCHW layout:
      x  [N, C, H, W]         — input feature map
      y  [N, C, out_h, out_w] — output feature map

    Pool types (runtime AXI-Lite register):
      POOL_MAX (0) — MaxPool / GlobalMaxPool
      POOL_AVG (1) — AveragePool / GlobalAveragePool
      POOL_LP  (2) — LpPool / GlobalLpPool (p=1 or 2)

    Global* variants are normalised at parse time: pool_h=in_h, pool_w=in_w,
    stride=1, pad=0.  No special hardware path is needed.

    Compatibility shims (init=False) satisfy the isinstance() exclusion guards
    in _compute_tensor_layouts() so that no broadcast layout logic is applied
    to PoolNode inputs or outputs.
    """

    kernel_name: ClassVar[str] = "PoolKernel"

    onnx_node:   onnx.NodeProto
    inputs:      List[TensorInfo]   # [x]
    output:      TensorInfo
    index:       int = 0
    align_elems: int = 8

    # Geometry (derived from ONNX attributes + shape inference)
    batch:     int = 1
    channels:  int = 0
    in_h:      int = 0
    in_w:      int = 0
    out_h:     int = 0
    out_w:     int = 0
    pool_h:    int = 1
    pool_w:    int = 1
    stride_h:  int = 1
    stride_w:  int = 1
    pad_top:   int = 0
    pad_left:  int = 0
    dil_h:     int = 1
    dil_w:     int = 1

    pool_type:         int = POOL_MAX  # POOL_MAX / POOL_AVG / POOL_LP
    lp_order:          int = 2         # 1 or 2 (LP only)
    count_include_pad: int = 0         # 0 or 1 (AVG only)

    # Compatibility shims — never set by callers.
    outer_count:        int  = field(default=1,    init=False)
    chunk_size:         int  = field(default=0,    init=False)
    aligned_chunk_size: int  = field(default=0,    init=False)
    a_advances:         bool = field(default=True, init=False)
    b_advances:         bool = field(default=True, init=False)
    arity:              int  = field(default=1,    init=False)

    # ------------------------------------------------------------------ #
    # Factory                                                              #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_onnx_node(
        cls,
        node:        onnx.NodeProto,
        tensors:     dict,
        index:       int,
        align_elems: int = 8,
    ) -> "PoolNode":
        op_type = node.op_type
        if op_type not in POOL_OP_TYPES:
            raise SchedulerError(
                f"PoolNode.from_onnx_node() called with op_type='{op_type}'"
            )

        # Single required input: x
        if not node.input or node.input[0] == "":
            raise SchedulerError(
                f"Pool node '{node.name or op_type}' has no input tensor."
            )
        x_name = node.input[0]
        if x_name not in tensors:
            raise SchedulerError(
                f"Tensor '{x_name}' referenced by Pool node "
                f"'{node.name or op_type}' not found in graph."
            )
        x_info = tensors[x_name]

        if not node.output or node.output[0] == "":
            raise SchedulerError(
                f"Pool node '{node.name or op_type}' has no output tensor."
            )
        y_name = node.output[0]
        if y_name not in tensors:
            raise SchedulerError(
                f"Output tensor '{y_name}' of Pool node "
                f"'{node.name or op_type}' not found in graph."
            )
        y_info = tensors[y_name]

        # Validate 4-D NCHW layout
        if len(x_info.shape) != 4:
            raise SchedulerError(
                f"Pool node '{node.name or op_type}': input must be 4-D "
                f"(NCHW), got shape {x_info.shape}."
            )
        if len(y_info.shape) != 4:
            raise SchedulerError(
                f"Pool node '{node.name or op_type}': output must be 4-D "
                f"(NCHW), got shape {y_info.shape}."
            )

        n_val, c_val, h_in, w_in = x_info.shape
        _, _, h_out, w_out = y_info.shape

        # Pool type
        if op_type in ("MaxPool", "GlobalMaxPool"):
            pool_type_val = POOL_MAX
        elif op_type in ("AveragePool", "GlobalAveragePool"):
            pool_type_val = POOL_AVG
        else:
            pool_type_val = POOL_LP

        attrs = {a.name: a for a in node.attribute}

        def _ints(name: str, default: List[int]) -> List[int]:
            return list(attrs[name].ints) if name in attrs else default

        def _int(name: str, default: int) -> int:
            return attrs[name].i if name in attrs else default

        # Global* ops: pool window = full spatial extent, no stride/pad needed
        is_global = op_type in _GLOBAL_POOL_OP_TYPES
        if is_global:
            pool_h_val   = h_in
            pool_w_val   = w_in
            stride_h_val = 1
            stride_w_val = 1
            pad_top_val  = 0
            pad_left_val = 0
            dil_h_val    = 1
            dil_w_val    = 1
        else:
            ks = _ints("kernel_shape", [])
            if len(ks) != 2:
                raise SchedulerError(
                    f"Pool node '{node.name or op_type}': kernel_shape must "
                    f"have exactly 2 values (2-D pooling only), got {ks!r}."
                )
            pool_h_val, pool_w_val = int(ks[0]), int(ks[1])

            strides = _ints("strides", [1, 1])
            if len(strides) != 2:
                raise SchedulerError(
                    f"Pool node '{node.name or op_type}': only 2-D strides "
                    f"supported, got {strides}."
                )
            stride_h_val, stride_w_val = int(strides[0]), int(strides[1])

            dilations     = _ints("dilations", [1, 1])
            dil_h_val     = int(dilations[0]) if len(dilations) >= 1 else 1
            dil_w_val     = int(dilations[1]) if len(dilations) >= 2 else 1

            auto_pad  = (attrs["auto_pad"].s.decode("utf-8")
                         if "auto_pad" in attrs else "NOTSET")
            pads_attr = _ints("pads", [0, 0, 0, 0])

            if auto_pad == "NOTSET":
                pad_top_val  = int(pads_attr[0]) if len(pads_attr) >= 1 else 0
                pad_left_val = int(pads_attr[1]) if len(pads_attr) >= 2 else 0
            elif auto_pad == "VALID":
                pad_top_val  = 0
                pad_left_val = 0
            elif auto_pad in ("SAME_UPPER", "SAME_LOWER"):
                eff_kh = dil_h_val * (pool_h_val - 1) + 1
                eff_kw = dil_w_val * (pool_w_val - 1) + 1
                pad_h  = max(0, (h_out - 1) * stride_h_val + eff_kh - h_in)
                pad_w  = max(0, (w_out - 1) * stride_w_val + eff_kw - w_in)
                if auto_pad == "SAME_UPPER":
                    pad_top_val  = pad_h // 2
                    pad_left_val = pad_w // 2
                else:
                    pad_top_val  = (pad_h + 1) // 2
                    pad_left_val = (pad_w + 1) // 2
            else:
                raise SchedulerError(
                    f"Pool node '{node.name or op_type}': "
                    f"unsupported auto_pad='{auto_pad}'."
                )

            if _int("ceil_mode", 0) != 0:
                raise SchedulerError(
                    f"Pool node '{node.name or op_type}': ceil_mode=1 is not "
                    f"supported; only floor output dimensions are implemented."
                )

        # LpPool / GlobalLpPool: parse p attribute (default 2)
        lp_order_val = _int("p", 2) if pool_type_val == POOL_LP else 2
        if pool_type_val == POOL_LP and lp_order_val not in (1, 2):
            raise SchedulerError(
                f"Pool node '{node.name or op_type}': p={lp_order_val} is not "
                f"supported. PoolingKernel implements p=1 and p=2 only."
            )

        # AveragePool / GlobalAveragePool: count_include_pad (default 0)
        cip_val = _int("count_include_pad", 0) if pool_type_val == POOL_AVG else 0

        # ------------------------------------------------------------------
        # Hardware-bound validation — see _pool_hw_config for where these
        # constants come from (kernels/pool/CMakeLists.txt + optional
        # build/CMakeCache.txt overrides).  PoolingKernel sizes line_buf
        # and the unrolled per-position adders at compile time, so a
        # window violating any of these bounds has no runtime fallback —
        # reject at parse time rather than emit code the kernel can't
        # service.  See doc/POOL_OPTIMIZATION.md §4 for the constraint
        # rationale (vertical span / horizontal span fit the line buffer;
        # pool window fits the bounded reducer adder tree).
        # ------------------------------------------------------------------
        node_label = node.name or op_type
        if pool_h_val > POOL_MAX_KH:
            raise SchedulerError(
                f"Pool node '{node_label}' ({op_type}): pool_h={pool_h_val} "
                f"exceeds PoolingKernel's compile-time limit "
                f"kMaxPoolH={POOL_MAX_KH}.  Increase POOL_MAX_KH in "
                f"kernels/pool/CMakeLists.txt (and rebuild) to support a "
                f"taller window."
            )
        if pool_w_val > POOL_MAX_KW:
            raise SchedulerError(
                f"Pool node '{node_label}' ({op_type}): pool_w={pool_w_val} "
                f"exceeds PoolingKernel's compile-time limit "
                f"kMaxPoolW={POOL_MAX_KW}.  Increase POOL_MAX_KW in "
                f"kernels/pool/CMakeLists.txt (and rebuild) to support a "
                f"wider window."
            )
        v_span = (pool_h_val - 1) * dil_h_val + 1
        if v_span > POOL_MAX_LINE_BUF_ROWS:
            raise SchedulerError(
                f"Pool node '{node_label}' ({op_type}): dilated vertical "
                f"span (pool_h-1)*dil_h + 1 = ({pool_h_val}-1)*{dil_h_val} "
                f"+ 1 = {v_span} exceeds line-buffer row capacity "
                f"kMaxLineBufRows={POOL_MAX_LINE_BUF_ROWS}.  Reduce dil_h "
                f"or raise POOL_MAX_LINE_BUF_ROWS (must remain a power of 2)."
            )
        h_span = (pool_w_val - 1) * dil_w_val + 1
        if h_span > POOL_MAX_LINE_BUF_COLS:
            raise SchedulerError(
                f"Pool node '{node_label}' ({op_type}): dilated horizontal "
                f"span (pool_w-1)*dil_w + 1 = ({pool_w_val}-1)*{dil_w_val} "
                f"+ 1 = {h_span} exceeds line-buffer column capacity "
                f"kMaxLineBufCols={POOL_MAX_LINE_BUF_COLS} (a single pool "
                f"window must fit horizontally in the line buffer).  Reduce "
                f"dil_w or raise POOL_MAX_LINE_BUF_COLS."
            )

        return cls(
            onnx_node=node,
            inputs=[x_info],
            output=y_info,
            index=index,
            align_elems=align_elems,
            batch=n_val,
            channels=c_val,
            in_h=h_in,
            in_w=w_in,
            out_h=h_out,
            out_w=w_out,
            pool_h=pool_h_val,
            pool_w=pool_w_val,
            stride_h=stride_h_val,
            stride_w=stride_w_val,
            pad_top=pad_top_val,
            pad_left=pad_left_val,
            dil_h=dil_h_val,
            dil_w=dil_w_val,
            pool_type=pool_type_val,
            lp_order=lp_order_val,
            count_include_pad=cip_val,
        )

    # ------------------------------------------------------------------ #
    # Code emission                                                        #
    # ------------------------------------------------------------------ #

    def emit_comment(self) -> str:
        x_name  = self.inputs[0].onnx_name
        op_type = self.onnx_node.op_type
        is_global = op_type in _GLOBAL_POOL_OP_TYPES
        if is_global:
            geo = "  global"
        else:
            k_str = (
                f"{self.pool_h}x{self.pool_w}"
                if self.pool_h != self.pool_w else str(self.pool_h)
            )
            geo = (
                f"  k={k_str} s={self.stride_h}x{self.stride_w}"
                f" p={self.pad_top},{self.pad_left}"
            )
        if self.pool_type == POOL_LP:
            geo += f" norm={self.lp_order}"
        return (
            f"    /* [{self.index}] {op_type}({x_name})"
            f" -> {self.output.onnx_name}"
            f"  [{self.batch},{self.channels},{self.in_h},{self.in_w}]"
            f"→[{self.batch},{self.channels},{self.out_h},{self.out_w}]"
            f"{geo} */"
        )

    def emit_call(self, layouts: dict) -> str:  # noqa: ARG002
        x = self.inputs[0].c_name
        y = self.output.c_name
        return (
            f"    run_pool({x}, {y},\n"
            f"             {self.batch}u, {self.channels}u,\n"
            f"             {self.in_h}u, {self.in_w}u,\n"
            f"             {self.out_h}u, {self.out_w}u,\n"
            f"             {self.pool_h}u, {self.pool_w}u,\n"
            f"             {self.stride_h}u, {self.stride_w}u,\n"
            f"             {self.pad_top}u, {self.pad_left}u,\n"
            f"             {self.dil_h}u, {self.dil_w}u,\n"
            f"             {self.pool_type}u, {self.lp_order}u,"
            f" {self.count_include_pad}u);"
        )


# ------------------------------------------------------------------ #
# ReshapeNode                                                          #
# ------------------------------------------------------------------ #

@dataclass
class ReshapeNode:
    """ONNX Reshape / Squeeze / Unsqueeze / Dropout — a pure memory view, no hardware call.

    The output buffer is an alias for the input buffer (same physical
    address, different shape metadata).  emit_call() returns an empty
    string; the alias assignment is emitted once in inference_init().

    Dropout is the identity in inference mode (training_mode=False), so it
    costs nothing at runtime.  The ratio and mask outputs are ignored.
    """

    kernel_name: ClassVar[str] = ""  # no hardware kernel

    onnx_node:   onnx.NodeProto
    inputs:      List[TensorInfo]   # [source]
    output:      TensorInfo
    index:       int = 0
    align_elems: int = 8

    # Compatibility shims
    outer_count:        int  = field(default=1,    init=False)
    chunk_size:         int  = field(default=0,    init=False)
    aligned_chunk_size: int  = field(default=0,    init=False)
    a_advances:         bool = field(default=True, init=False)
    b_advances:         bool = field(default=True, init=False)
    arity:              int  = field(default=1,    init=False)

    @classmethod
    def from_onnx_node(
        cls,
        node:        onnx.NodeProto,
        tensors:     dict,
        index:       int,
        align_elems: int = 8,
    ) -> "ReshapeNode":
        op_type  = node.op_type
        src_name = node.input[0]
        out_name = node.output[0]
        if src_name not in tensors:
            raise SchedulerError(
                f"{op_type} node '{node.name}': source tensor '{src_name}' not found."
            )
        if out_name not in tensors:
            raise SchedulerError(
                f"{op_type} node '{node.name}': output tensor '{out_name}' not found."
            )
        src = tensors[src_name]
        out = tensors[out_name]
        if src.numel != out.numel:
            raise SchedulerError(
                f"{op_type} node '{node.name or op_type}': "
                f"source numel={src.numel} (shape={src.shape}) != "
                f"output numel={out.numel} (shape={out.shape}). "
                f"Only view-reshapes (same number of elements) are supported."
            )
        return cls(onnx_node=node, inputs=[src], output=out,
                   index=index, align_elems=align_elems)

    def emit_comment(self) -> str:
        return (
            f"    /* [{self.index}] {self.onnx_node.op_type}({self.inputs[0].onnx_name})"
            f" -> {self.output.onnx_name}"
            f"  {self.inputs[0].shape} → {self.output.shape}"
            f"  (buffer alias, no hardware call) */"
        )

    def emit_call(self, layouts: dict) -> str:  # noqa: ARG002
        return ""


# ------------------------------------------------------------------ #
# SpaceToDepthNode — host-side block reorder (space-to-depth stem)     #
# ------------------------------------------------------------------ #

SPACE_TO_DEPTH_OP_TYPES: frozenset = frozenset({"SpaceToDepth"})


def _s2d_stem_geometry(k: int, pad: int) -> Tuple[int, int, int]:
    """One axis of the stride-2 → space-to-depth rewrite.

    A stride-2 conv tap ``t`` (0..k-1) on the original axis reads input row
    ``2o + t - pad``.  After a block-2 space-to-depth the same row is
    ``(r, ph)`` with ``2r + ph = 2o + t - pad``; a stride-1 conv over the
    reordered tensor with pad ``P`` reads rows ``o + R - P``, so
    ``t = 2R + ph + pad - 2P``.  ``P = ceil(pad/2)`` makes ``t >= 0`` for
    every ``(R, ph)`` reach ``t = 0``; the last tap then fixes the new
    kernel extent.  Returns ``(K, P, off)`` with ``off = pad - 2P`` (0 or
    -1) so that ``t = 2R + ph + off``; taps outside ``0..k-1`` are zero.
    """
    P   = -(-pad // 2)
    off = pad - 2 * P
    K   = (k - 1 - off) // 2 + 1
    return K, P, off


def _s2d_stem_weight(w: np.ndarray, pad_top: int, pad_left: int) -> np.ndarray:
    """Re-index a ``[M, C, kh, kw]`` stride-2 filter for the block-2
    space-to-depth input (ONNX SpaceToDepth channel order: output channel
    ``(ph*2 + pw)*C + c`` holds ``x[c][2r+ph][2cc+pw]``).

    ``w'[m][(ph*2+pw)*C + c][R][Cc] = w[m][c][2R+ph+off_h][2Cc+pw+off_w]``
    with ``off`` from :func:`_s2d_stem_geometry`; source taps outside the
    original window are zero.
    """
    M, C, kh, kw = w.shape
    Kh, _, off_h = _s2d_stem_geometry(kh, pad_top)
    Kw, _, off_w = _s2d_stem_geometry(kw, pad_left)
    out = np.zeros((M, 4 * C, Kh, Kw), dtype=w.dtype)
    for ph in range(2):
        for pw in range(2):
            for R in range(Kh):
                a = 2 * R + ph + off_h
                if not 0 <= a < kh:
                    continue
                for Cc in range(Kw):
                    b = 2 * Cc + pw + off_w
                    if not 0 <= b < kw:
                        continue
                    out[:, (ph * 2 + pw) * C:(ph * 2 + pw + 1) * C, R, Cc] = w[:, :, a, b]
    return out


@dataclass
class SpaceToDepthNode:
    """ONNX SpaceToDepth — a block reorder run on the host CPU, no hardware call.

    ``y[n][(ph*bs + pw)*C + c][r][cc] = x[n][c][bs*r + ph][bs*cc + pw]``
    (the ONNX channel order).  Emitted as a C loop inside inference_run():
    it reads the source buffer after the producing lane has drained
    (invalidating the CPU cache first when the source is a kernel output),
    writes the reordered tensor into its own DMA buffer and flushes it so
    the consuming kernel sees it.  The node is synchronous — it occupies
    no lane and its output is complete when the loop returns — so the
    event stream emits it as a ``('cpu', idx)`` event and its liveness
    interval starts and ends at that event.

    The loop never touches the DMA buffers element-wise: on the board the
    DMA pool is an XRT BO with a NON-CACHEABLE CPU mapping, so the strided
    2-byte source loads of the reorder were ~100 ns each (16 ms for the
    ResNet-18 stem, measured).  inference_init() therefore mallocs a cached
    staging block per node (``stage_c_name``, 2 x source numel); the run
    does memcpy(BO -> stage_in), reorder stage_in -> stage_out in cached
    memory, memcpy(stage_out -> BO): the only BO traffic is two wide
    sequential copies.

    Created by ``OnnxGraph._space_to_depth_stems`` (the ``s2d_stem`` graph
    transform) in front of a re-indexed stride-1 Conv; models that carry a
    native SpaceToDepth are accepted the same way.
    """

    kernel_name: ClassVar[str] = ""  # no hardware kernel

    onnx_node:   onnx.NodeProto
    inputs:      List[TensorInfo]   # [source]
    output:      TensorInfo
    index:       int = 0
    align_elems: int = 8

    batch:     int = 1
    in_ch:     int = 0
    in_h:      int = 0
    in_w:      int = 0
    blocksize: int = 2
    # Set by OnnxGraph: True when the source is a graph input (already
    # flushed by the caller, nothing to invalidate before the CPU reads it).
    src_is_graph_input: bool = field(default=False, init=False)

    # Compatibility shims
    outer_count:        int  = field(default=1,    init=False)
    chunk_size:         int  = field(default=0,    init=False)
    aligned_chunk_size: int  = field(default=0,    init=False)
    a_advances:         bool = field(default=True, init=False)
    b_advances:         bool = field(default=True, init=False)
    arity:              int  = field(default=1,    init=False)

    @classmethod
    def from_onnx_node(
        cls,
        node:        onnx.NodeProto,
        tensors:     dict,
        index:       int,
        align_elems: int = 8,
    ) -> "SpaceToDepthNode":
        label = node.name or node.op_type
        if node.op_type not in SPACE_TO_DEPTH_OP_TYPES:
            raise SchedulerError(
                f"SpaceToDepthNode.from_onnx_node() called with op_type='{node.op_type}'"
            )
        src_name = node.input[0]
        out_name = node.output[0]
        if src_name not in tensors:
            raise SchedulerError(
                f"SpaceToDepth node '{label}': source tensor '{src_name}' not found."
            )
        if out_name not in tensors:
            raise SchedulerError(
                f"SpaceToDepth node '{label}': output tensor '{out_name}' not found."
            )
        src = tensors[src_name]
        out = tensors[out_name]
        attrs = {a.name: a for a in node.attribute}
        if "blocksize" not in attrs:
            raise SchedulerError(
                f"SpaceToDepth node '{label}': 'blocksize' attribute is required."
            )
        bs = int(attrs["blocksize"].i)
        if bs < 1:
            raise SchedulerError(
                f"SpaceToDepth node '{label}': blocksize={bs} must be >= 1."
            )
        if len(src.shape) != 4:
            raise SchedulerError(
                f"SpaceToDepth node '{label}': input must be 4-D (NCHW), "
                f"got shape {src.shape}."
            )
        n, c, h, w = src.shape
        if h % bs != 0 or w % bs != 0:
            raise SchedulerError(
                f"SpaceToDepth node '{label}': input H={h}, W={w} must be "
                f"multiples of blocksize={bs}."
            )
        expect = [n, c * bs * bs, h // bs, w // bs]
        if list(out.shape) != expect:
            raise SchedulerError(
                f"SpaceToDepth node '{label}': output shape {out.shape} != "
                f"expected {expect} for input {src.shape}, blocksize={bs}."
            )
        return cls(onnx_node=node, inputs=[src], output=out,
                   index=index, align_elems=align_elems,
                   batch=n, in_ch=c, in_h=h, in_w=w, blocksize=bs)

    @property
    def numel(self) -> int:
        """Elements of the source (== of the output)."""
        return self.batch * self.in_ch * self.in_h * self.in_w

    @property
    def stage_c_name(self) -> str:
        """C identifier of the node's cached host staging block
        (allocated in inference_init(), freed in inference_deinit())."""
        return f"_s2d_stage_{self.output.c_name}"

    def emit_comment(self) -> str:
        return (
            f"    /* [{self.index}] SpaceToDepth({self.inputs[0].onnx_name})"
            f" -> {self.output.onnx_name}"
            f"  {self.inputs[0].shape} → {self.output.shape}"
            f"  blocksize={self.blocksize}"
            f"  (host CPU reorder, no hardware call) */"
        )

    def emit_call(self, layouts: dict) -> str:  # noqa: ARG002
        x, y = self.inputs[0].c_name, self.output.c_name
        bs = self.blocksize
        oh, ow = self.in_h // bs, self.in_w // bs
        sync_src = (
            "" if self.src_is_graph_input else
            f"        /* '{x}' was written by a kernel: drop stale CPU cache lines. */\n"
            f"        inference_buf_sync_from_device({x});\n"
        )
        n_el  = self.numel
        stage = self.stage_c_name
        return (
            "    {\n"
            f"        /* y[n][(ph*{bs}+pw)*C + c][r][cc] = x[n][c][{bs}*r+ph][{bs}*cc+pw]"
            f"  (N={self.batch}, C={self.in_ch}, H={self.in_h}, W={self.in_w});\n"
            "         * dst advances in output order, so it is written sequentially.\n"
            "         * Staged through cached host memory: the DMA buffers are mapped\n"
            "         * non-cacheable (XRT BO), so the strided source loads of the reorder\n"
            "         * must not hit them — only two wide sequential memcpys do. */\n"
            f"{sync_src}"
            f"        const Data_t *src = {stage};\n"
            f"        Data_t       *dst = {stage} + {n_el}u;\n"
            "        unsigned n, ph, pw, c, r, cc;\n"
            f"        memcpy({stage}, inference_buf_ptr({x}),"
            f" {n_el}u * INFERENCE_BYTES_PER_ELEM);\n"
            f"        for (n = 0u; n < {self.batch}u; n++)\n"
            f"        for (ph = 0u; ph < {bs}u; ph++)\n"
            f"        for (pw = 0u; pw < {bs}u; pw++)\n"
            f"        for (c = 0u; c < {self.in_ch}u; c++)\n"
            f"        for (r = 0u; r < {oh}u; r++) {{\n"
            f"            const Data_t *s = src + ((n * {self.in_ch}u + c) * {self.in_h}u"
            f" + r * {bs}u + ph) * {self.in_w}u + pw;\n"
            f"            for (cc = 0u; cc < {ow}u; cc++)\n"
            f"                *dst++ = s[cc * {bs}u];\n"
            "        }\n"
            f"        memcpy(inference_buf_ptr({y}), {stage} + {n_el}u,"
            f" {n_el}u * INFERENCE_BYTES_PER_ELEM);\n"
            f"        inference_buf_sync_to_device({y});\n"
            "    }"
        )
