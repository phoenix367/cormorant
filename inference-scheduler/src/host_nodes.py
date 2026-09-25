"""
Host-CPU operator nodes (BERT_PLAN phase 1b).

Ops the PL kernels cannot run are executed on the A53 inside
``inference_run()``: Softmax, LayerNormalization, Gelu, Transpose and Slice
(lowered ``Split``).  ``SpaceToDepthNode`` (nodes.py) was the first host op; these nodes
generalise it and share its event-stream treatment: no lane
(``kernel_name == ""``), one synchronous ``('cpu', idx)`` event that waits
on in-flight producers first, liveness interval that starts and ends at
that event.

Numeric contract (identical in the generated C and in ``reference()``,
which the scheduler's fixed-point simulator runs):

  * inputs are read as double: ``Data_t`` elements through ``host_ld``
    (``(int16_t)bits / 256.0`` for ap_fixed<16,8>, exact);
  * all arithmetic is IEEE double, in the operation order written in the
    helper (reductions accumulate left to right; the generated project is
    compiled with ``-ffp-contract=off`` so no FMA contraction); exp / tanh /
    erf come from libm — the simulator calls Python's ``math`` module (the
    same glibc), never numpy's SIMD variants (``libm()`` below);
  * outputs are written back with ``host_st``: round half to even
    (``nearbyint`` under the default FE_TONEAREST mode == ``np.round``) and
    saturation to the Data_t range, NaN -> 0.

Data movement never touches the DMA buffers element-wise: on the KV260 the
buffer pool is an XRT BO mapped NON-CACHEABLE (a strided 2-byte read costs
~100 ns; the SpaceToDepth stem measured 16 ms for 300 KB when it read the
BO directly).  Every host op therefore does one wide ``memcpy`` BO -> a
malloc'd cached staging arena per input (``host_load``), computes
stage -> stage, and one ``memcpy`` back (``host_store``, followed by a
cache flush for the consuming kernel).  The arena is shared by all host
ops (they run one at a time on the CPU) and sized to the largest one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import ClassVar, Dict, List, Optional, Tuple

import numpy as np
import onnx
from .nodes import SchedulerError
from .tensor import TensorInfo


# ------------------------------------------------------------------ #
# Context handed to the factories                                      #
# ------------------------------------------------------------------ #

@dataclass
class HostContext:
    """Model-level facts a host-node factory needs besides the tensors."""
    opset: int                                   # default-domain opset
    consts: Dict[str, np.ndarray]                # raw initializer arrays (original dtype)


def _attrs(node: onnx.NodeProto) -> dict:
    return {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}


def _label(node: onnx.NodeProto) -> str:
    return node.name or node.op_type


def _prod(xs) -> int:
    n = 1
    for x in xs:
        n *= int(x)
    return n


def _row_major_strides(shape) -> List[int]:
    st, acc = [], 1
    for d in reversed(list(shape)):
        st.append(acc)
        acc *= int(d)
    return list(reversed(st))


def _resolve(tensors: dict, name: str, node: onnx.NodeProto) -> TensorInfo:
    if name not in tensors:
        raise SchedulerError(
            f"{node.op_type} node '{_label(node)}': tensor '{name}' not found.")
    return tensors[name]


def _const(ctx: HostContext, tensors: dict, name: str, node: onnx.NodeProto,
           what: str) -> np.ndarray:
    """Raw value of a constant input (initializer), else SchedulerError."""
    if name in ctx.consts:
        return np.asarray(ctx.consts[name])
    t = tensors.get(name)
    if t is not None and t.data is not None:
        return np.asarray(t.data)
    raise SchedulerError(
        f"{node.op_type} node '{_label(node)}': {what} ('{name}') must be a "
        f"constant initializer.")


def _collapse(dims: List[int], sstr: List[int]) -> Tuple[List[int], List[int]]:
    """Merge adjacent output dims that are also adjacent in the source, drop
    size-1 dims, and left-pad to 5 dims for ``host_copy_nd``."""
    d, s = [], []
    for di, si in zip(dims, sstr, strict=True):
        if di == 1:
            continue
        if d and s[-1] == si * di:
            d[-1] *= di
            s[-1] = si
        else:
            d.append(int(di))
            s.append(int(si))
    if not d:
        d, s = [1], [1]
    if len(d) > 5:
        raise SchedulerError(
            f"host copy: {len(d)} non-mergeable dimensions (max 5): dims={dims}")
    pad = 5 - len(d)
    return [1] * pad + d, [0] * pad + s


# Transcendentals exactly as the generated C gets them: Python's math module
# is the platform libm (glibc), the library the host ops link on the board.
# numpy's own SIMD kernels are NOT used: np.tanh differs from glibc's tanh by
# up to 3 ulp on ~26 % of inputs (numpy 2.x, x86-64), and np.exp switches
# to SVML on AVX-512 hosts.
_LIBM = {name: np.vectorize(getattr(math, name), otypes=[np.float64])
         for name in ("exp", "tanh", "erf")}


def libm(name: str, x) -> np.ndarray:
    """Elementwise libm ``exp`` / ``tanh`` / ``erf`` of a float64 array."""
    x = np.asarray(x, np.float64)
    return _LIBM[name](x) if x.size else x.copy()


def _c_float(v: float) -> str:
    """float32 literal that round-trips exactly."""
    v = float(np.float32(v))
    if not math.isfinite(v):
        raise SchedulerError(f"non-finite host-op constant {v}")
    s = "%.9g" % v
    if not any(c in s for c in ".eE"):
        s += ".0"
    return s + "f"


def _c_double(v: float) -> str:
    """double literal that round-trips exactly."""
    v = float(v)
    if not math.isfinite(v):
        raise SchedulerError(f"non-finite host-op constant {v}")
    s = repr(v)
    if not any(c in s for c in ".eE"):
        s += ".0"
    return s


# ------------------------------------------------------------------ #
# C helper library (emitted once per project, only the kinds in use)   #
# ------------------------------------------------------------------ #

HOST_C_COMMON = r"""/*
 * host_load  — one wide memcpy DMA buffer -> cached stage.  A buffer with an
 *              advancing-strided layout (n_chunks blocks of `stride`
 *              elements, the first `chunk` valid) is compacted in place.
 * host_store — the inverse (expand back to front, zero the gaps), one wide
 *              memcpy stage -> DMA buffer, then flush the CPU cache so the
 *              consuming kernel reads the new data.
 * The DMA buffers are an XRT BO mapped non-cacheable on the KV260: these two
 * sequential copies are the only accesses a host op makes to them.
 */
static void host_load(Data_t *dst, inference_buf_t *src,
                      unsigned n_chunks, unsigned chunk, unsigned stride)
{
    unsigned c;
    memcpy(dst, inference_buf_ptr(src),
           ((size_t)(n_chunks - 1u) * stride + chunk) * INFERENCE_BYTES_PER_ELEM);
    for (c = 1u; c < n_chunks && stride != chunk; c++)
        memmove(dst + (size_t)c * chunk, dst + (size_t)c * stride,
                (size_t)chunk * INFERENCE_BYTES_PER_ELEM);
}

static void host_store(inference_buf_t *dst, Data_t *src,
                       unsigned n_chunks, unsigned chunk, unsigned stride)
{
    unsigned c;
    size_t   n = (n_chunks > 1u) ? (size_t)n_chunks * stride : (size_t)chunk;
    if (n_chunks > 1u && stride != chunk) {
        for (c = n_chunks - 1u; c > 0u; c--)
            memmove(src + (size_t)c * stride, src + (size_t)c * chunk,
                    (size_t)chunk * INFERENCE_BYTES_PER_ELEM);
        for (c = 0u; c < n_chunks; c++)
            memset(src + (size_t)c * stride + chunk, 0,
                   (size_t)(stride - chunk) * INFERENCE_BYTES_PER_ELEM);
    }
    memcpy(inference_buf_ptr(dst), src, n * INFERENCE_BYTES_PER_ELEM);
    inference_buf_sync_to_device(dst);
}
"""

HOST_C_HELPERS: Dict[str, str] = {
    "softmax": r"""/* Softmax over rows of n elements (ONNX Softmax, last axis; opset < 13
 * "coerce to 2-D" with n = prod(shape[axis:])):
 *   y[j] = exp(x[j] - max) / sum_k exp(x[k] - max),  sum left to right.
 * e: n doubles of scratch. */
static void host_softmax(const Data_t *x, Data_t *y, unsigned rows, unsigned n,
                         double *e)
{
    unsigned r, j;
    for (r = 0u; r < rows; r++, x += n, y += n) {
        double m = host_ld(x[0]), s = 0.0;
        for (j = 1u; j < n; j++) {
            double v = host_ld(x[j]);
            if (v > m) m = v;
        }
        for (j = 0u; j < n; j++) {
            e[j] = exp(host_ld(x[j]) - m);
            s += e[j];
        }
        for (j = 0u; j < n; j++)
            y[j] = host_st(e[j] / s);
    }
}
""",
    "layernorm": r"""/* LayerNormalization over rows of n elements, gamma / beta float32
 * (NULL = 1 / 0):
 *   mean = sum(x) / n,  var = sum((x - mean)^2) / n   (sums left to right)
 *   inv  = 1 / sqrt(var + eps)
 *   tf_form 1 (TensorFlow / BERT graph):  g = inv * gamma;
 *                                          y = x * g + (beta - mean * g)
 *   tf_form 0 (ONNX op):                   y = (x - mean) * inv * gamma + beta */
static void host_layernorm(const Data_t *x, Data_t *y, unsigned rows, unsigned n,
                           const float *gamma, const float *beta, double eps,
                           int tf_form)
{
    unsigned r, j;
    for (r = 0u; r < rows; r++, x += n, y += n) {
        double sum = 0.0, var = 0.0, mean, inv;
        for (j = 0u; j < n; j++)
            sum += host_ld(x[j]);
        mean = sum / (double)n;
        for (j = 0u; j < n; j++) {
            double d = host_ld(x[j]) - mean;
            var += d * d;
        }
        var = var / (double)n;
        inv = 1.0 / sqrt(var + eps);
        for (j = 0u; j < n; j++) {
            double xv = host_ld(x[j]);
            double ga = gamma ? (double)gamma[j] : 1.0;
            double be = beta  ? (double)beta[j]  : 0.0;
            if (tf_form) {
                double g = inv * ga;
                y[j] = host_st(xv * g + (be - mean * g));
            } else {
                y[j] = host_st((xv - mean) * inv * ga + be);
            }
        }
    }
}
""",
    "gelu_tanh": r"""/* GELU, tanh approximation (BERT's graph form, constants as found):
 *   y = x * (0.5 * (1 + tanh(c2 * (x + c1 * x^3)))) */
static void host_gelu_tanh(const Data_t *x, Data_t *y, unsigned n,
                           double c1, double c2)
{
    unsigned i;
    for (i = 0u; i < n; i++) {
        double v = host_ld(x[i]);
        double u = c2 * (v + c1 * (v * v * v));
        y[i] = host_st(v * (0.5 * (1.0 + tanh(u))));
    }
}
""",
    "gelu_erf": r"""/* GELU, exact form:  y = x * (0.5 * (1 + erf(u))),  u = x / k (div) or x * k */
static void host_gelu_erf(const Data_t *x, Data_t *y, unsigned n,
                          double k, int div)
{
    unsigned i;
    for (i = 0u; i < n; i++) {
        double v = host_ld(x[i]);
        double u = div ? v / k : v * k;
        y[i] = host_st(v * (0.5 * (1.0 + erf(u))));
    }
}
""",
    "copy_nd": r"""/* Strided gather-copy (Transpose / Slice / non-contiguous Split): dst is
 * written sequentially in output order, dst[i0..i4] = src[sum ik * s[k]]. */
static void host_copy_nd(const Data_t *src, Data_t *dst,
                         const unsigned d[5], const unsigned s[5])
{
    unsigned i0, i1, i2, i3, i4;
    for (i0 = 0u; i0 < d[0]; i0++)
    for (i1 = 0u; i1 < d[1]; i1++)
    for (i2 = 0u; i2 < d[2]; i2++)
    for (i3 = 0u; i3 < d[3]; i3++) {
        const Data_t *p = src + (size_t)i0 * s[0] + (size_t)i1 * s[1]
                              + (size_t)i2 * s[2] + (size_t)i3 * s[3];
        if (s[4] == 1u) {
            memcpy(dst, p, (size_t)d[4] * sizeof(Data_t));
            dst += d[4];
        } else {
            for (i4 = 0u; i4 < d[4]; i4++)
                *dst++ = p[(size_t)i4 * s[4]];
        }
    }
}
""",
}

# Helpers that need <math.h> (exp / sqrt / tanh / erf); host_st uses
# nearbyint, so math.h is included whenever any host node exists.
HOST_C_HELPER_ORDER = ("softmax", "layernorm", "gelu_tanh", "gelu_erf", "copy_nd")


# ------------------------------------------------------------------ #
# Base class                                                           #
# ------------------------------------------------------------------ #

@dataclass
class HostNode:
    """Common part of every host-CPU node (see module docstring).

    ``inputs`` are the RUNTIME tensors the node reads (DMA buffers: graph
    inputs, kernel / host outputs, or a weight read at run time).  Constant
    operands (LayerNorm gamma / beta, Slice starts / ends, ...) are baked
    into the node at
    construction and emitted as C literals, never as DMA weights.
    """

    kernel_name: ClassVar[str] = ""        # no hardware lane
    helpers:     ClassVar[Tuple[str, ...]] = ()

    onnx_node:   onnx.NodeProto
    inputs:      List[TensorInfo]
    output:      TensorInfo
    index:       int = 0
    align_elems: int = 8

    # Compatibility shims (read by the layout / header passes)
    outer_count:        int  = field(default=1,    init=False)
    chunk_size:         int  = field(default=0,    init=False)
    aligned_chunk_size: int  = field(default=0,    init=False)
    a_advances:         bool = field(default=True, init=False)
    b_advances:         bool = field(default=True, init=False)
    arity:              int  = field(default=1,    init=False)

    # ---- codegen interface ---------------------------------------- #

    def staged_inputs(self) -> List[TensorInfo]:
        """Inputs copied into the cached staging arena before compute."""
        return list(self.inputs)

    def direct_inputs(self) -> List[TensorInfo]:
        """Inputs read in place from their DMA buffer (e.g. whole rows)."""
        return []

    def scratch_bytes(self) -> int:
        return 0

    def c_file_consts(self, dtype) -> List[str]:  # noqa: ARG002
        """File-scope C declarations (per-node constant tables)."""
        return []

    def c_call(self, ins: List[str], out: str, scratch: str,
               direct: List[str], dtype) -> List[str]:
        raise NotImplementedError

    def reference(self, ins: List[np.ndarray], dtype) -> np.ndarray:
        """Output (logical shape, float64) exactly as the C helper computes it."""
        raise NotImplementedError

    def describe(self) -> str:
        return ""

    def c_helpers(self) -> Tuple[str, ...]:
        """Names of the HOST_C_HELPERS this node calls."""
        return type(self).helpers

    @property
    def c_prefix(self) -> str:
        """Prefix for this node's file-scope C identifiers."""
        return f"_host_{self.output.c_name}"

    def emit_comment(self) -> str:
        ins = ", ".join(t.onnx_name for t in self.inputs)
        shp = " x ".join(str(t.shape) for t in self.inputs)
        det = self.describe()
        return (f"    /* [{self.index}] {self.onnx_node.op_type}({ins}) -> "
                f"{self.output.onnx_name}  {shp} → {self.output.shape}"
                f"{'  ' + det if det else ''}  (host CPU, no hardware call) */")

    def emit_call(self, layouts: dict) -> str:  # noqa: ARG002
        raise RuntimeError("host nodes are emitted by CodeGenerator._emit_host_block")



# ------------------------------------------------------------------ #
# Softmax                                                              #
# ------------------------------------------------------------------ #

@dataclass
class SoftmaxNode(HostNode):
    """ONNX Softmax on the host (double precision).

    opset >= 13: the axis must be the last one.  opset < 13: the input is
    coerced to 2-D ``[prod(shape[:axis]), prod(shape[axis:])]`` and the
    softmax runs over the second dimension (ONNX semantics) — for BERT's
    ``axis = 3`` on a rank-4 tensor that is the last axis.
    """
    helpers: ClassVar[Tuple[str, ...]] = ("softmax",)
    rows: int = 1
    n:    int = 1

    @classmethod
    def from_onnx_node(cls, node, tensors, index, align_elems, ctx: HostContext):
        x = _resolve(tensors, node.input[0], node)
        y = _resolve(tensors, node.output[0], node)
        a = _attrs(node)
        rank = len(x.shape)
        axis = int(a.get("axis", 1 if ctx.opset < 13 else -1))
        if axis < 0:
            axis += rank
        if not 0 <= axis < max(rank, 1):
            raise SchedulerError(f"Softmax node '{_label(node)}': axis {a.get('axis')} "
                                 f"out of range for rank {rank}.")
        if ctx.opset >= 13 and axis != rank - 1:
            raise SchedulerError(
                f"Softmax node '{_label(node)}': axis={axis} on a rank-{rank} "
                f"tensor (opset {ctx.opset}); only the last axis is supported.")
        n = _prod(x.shape[axis:]) if rank else 1
        sn = cls(onnx_node=node, inputs=[x], output=y, index=index,
                 align_elems=align_elems, rows=max(x.numel // max(n, 1), 1), n=n)
        if y.numel != x.numel:
            raise SchedulerError(f"Softmax node '{_label(node)}': output numel "
                                 f"{y.numel} != input numel {x.numel}.")
        return sn

    def scratch_bytes(self) -> int:
        return 8 * self.n

    def describe(self) -> str:
        return f"rows={self.rows} n={self.n}"

    def c_call(self, ins, out, scratch, direct, dtype):
        return [f"host_softmax({ins[0]}, {out}, {self.rows}u, {self.n}u, (double *){scratch});"]

    def reference(self, ins, dtype):
        x = np.asarray(ins[0], np.float64).reshape(self.rows, self.n)
        e = libm("exp", x - x.max(axis=1, keepdims=True))
        s = np.cumsum(e, axis=1)[:, -1:]                   # left-to-right sum
        return dtype.host_quantize(e / s).reshape(self.output.shape)


# ------------------------------------------------------------------ #
# LayerNormalization (native op or fused TF pattern)                    #
# ------------------------------------------------------------------ #

@dataclass
class LayerNormNode(HostNode):
    """LayerNormalization over the trailing ``prod(shape[axis:])`` elements.

    Created from a native ONNX ``LayerNormalization`` (``tf_form = 0``) or
    from the TensorFlow / BERT subgraph fused by ``fusion.fuse_patterns``
    (attribute ``axi_tf_form = 1``, which selects the graph's own
    ``x * g + (beta - mean * g)`` arrangement).  Scale / bias must be
    constants; they stay float32 (host-side C arrays), exactly the values
    in the model.
    """
    helpers: ClassVar[Tuple[str, ...]] = ("layernorm",)
    rows:    int = 1
    n:       int = 1
    eps:     float = 1e-5
    tf_form: int = 0
    gamma:   Optional[np.ndarray] = field(default=None, repr=False)
    beta:    Optional[np.ndarray] = field(default=None, repr=False)

    @classmethod
    def from_onnx_node(cls, node, tensors, index, align_elems, ctx: HostContext):
        x = _resolve(tensors, node.input[0], node)
        y = _resolve(tensors, node.output[0], node)
        a = _attrs(node)
        rank = len(x.shape)
        axis = int(a.get("axis", -1))
        if axis < 0:
            axis += rank
        if not 0 <= axis < rank:
            raise SchedulerError(f"LayerNormalization node '{_label(node)}': axis "
                                 f"{a.get('axis')} out of range for rank {rank}.")
        n = _prod(x.shape[axis:])

        def _param(i, what):
            if len(node.input) <= i or not node.input[i]:
                return None
            v = _const(ctx, tensors, node.input[i], node, what).astype(np.float32)
            if v.size != n:
                raise SchedulerError(
                    f"LayerNormalization node '{_label(node)}': {what} has {v.size} "
                    f"elements, expected {n} (= prod(shape[axis:])).")
            return v.reshape(-1).copy()

        sn = cls(onnx_node=node, inputs=[x], output=y, index=index,
                 align_elems=align_elems, rows=x.numel // n, n=n,
                 eps=float(a.get("epsilon", 1e-5)), tf_form=int(a.get("axi_tf_form", 0)),
                 gamma=_param(1, "scale (gamma)"), beta=_param(2, "bias (beta)"))
        if len(node.output) > 1 and any(node.output[1:]):
            raise SchedulerError(f"LayerNormalization node '{_label(node)}': only "
                                 f"the Y output is supported (Mean / InvStdDev requested).")
        return sn

    def describe(self) -> str:
        form = "TF form" if self.tf_form else "ONNX form"
        return f"rows={self.rows} n={self.n} eps={self.eps:.3g} {form}"

    def c_file_consts(self, dtype):
        out = []
        for nm, arr in (("gamma", self.gamma), ("beta", self.beta)):
            if arr is None:
                continue
            lits = [_c_float(v) for v in arr]
            rows = ",\n".join("    " + ", ".join(lits[i:i + 6]) for i in range(0, len(lits), 6))
            out.append(f"/* [{self.index}] LayerNorm {nm} (float32, host-side constant) */\n"
                       f"static const float {self.c_prefix}_{nm}[{len(lits)}] = {{\n{rows}\n}};")
        return out

    def c_call(self, ins, out, scratch, direct, dtype):
        g = f"{self.c_prefix}_gamma" if self.gamma is not None else "NULL"
        b = f"{self.c_prefix}_beta" if self.beta is not None else "NULL"
        return [f"host_layernorm({ins[0]}, {out}, {self.rows}u, {self.n}u,",
                f"               {g}, {b}, {_c_double(self.eps)}, {self.tf_form});"]

    def reference(self, ins, dtype):
        x = np.asarray(ins[0], np.float64).reshape(self.rows, self.n)
        ga = (np.ones(self.n) if self.gamma is None else self.gamma.astype(np.float64))[None, :]
        be = (np.zeros(self.n) if self.beta is None else self.beta.astype(np.float64))[None, :]
        n = float(self.n)
        mean = np.cumsum(x, axis=1)[:, -1:] / n
        d = x - mean
        var = np.cumsum(d * d, axis=1)[:, -1:] / n
        inv = 1.0 / np.sqrt(var + self.eps)
        if self.tf_form:
            g = inv * ga
            y = x * g + (be - mean * g)
        else:
            y = d * inv * ga + be
        return dtype.host_quantize(y).reshape(self.output.shape)


# ------------------------------------------------------------------ #
# Gelu (native op or fused tanh / erf pattern)                          #
# ------------------------------------------------------------------ #

GELU_C1 = 0.044715
GELU_C2 = math.sqrt(2.0 / math.pi)
SQRT2 = math.sqrt(2.0)


@dataclass
class GeluNode(HostNode):
    """GELU on the host.  ``approximate == 'tanh'``:
    ``x * (0.5 * (1 + tanh(c2 * (x + c1 * x^3))))``; ``'none'``:
    ``x * (0.5 * (1 + erf(x / k)))`` (or ``x * k`` when the fused graph
    multiplies by 1/sqrt(2)).  A fused pattern carries the constants found
    in the graph (``axi_c1`` / ``axi_c2`` / ``axi_k`` / ``axi_div``, float32
    values as the reference implementation used them); a native ``Gelu``
    uses the exact double constants.
    """
    helpers: ClassVar[Tuple[str, ...]] = ("gelu_tanh", "gelu_erf")
    n:           int = 1
    approximate: str = "none"
    c1:          float = GELU_C1
    c2:          float = GELU_C2
    k:           float = SQRT2
    div:         int = 1

    @classmethod
    def from_onnx_node(cls, node, tensors, index, align_elems, ctx: HostContext):
        x = _resolve(tensors, node.input[0], node)
        y = _resolve(tensors, node.output[0], node)
        a = _attrs(node)
        approx = a.get("approximate", "none")
        if isinstance(approx, bytes):
            approx = approx.decode()
        if approx not in ("none", "tanh"):
            raise SchedulerError(f"Gelu node '{_label(node)}': approximate='{approx}'.")
        sn = cls(onnx_node=node, inputs=[x], output=y, index=index,
                 align_elems=align_elems, n=x.numel, approximate=approx,
                 c1=float(a.get("axi_c1", GELU_C1)), c2=float(a.get("axi_c2", GELU_C2)),
                 k=float(a.get("axi_k", SQRT2)), div=int(a.get("axi_div", 1)))
        return sn

    @property
    def helper(self) -> str:
        return "gelu_tanh" if self.approximate == "tanh" else "gelu_erf"

    def c_helpers(self) -> Tuple[str, ...]:
        return (self.helper,)

    def describe(self) -> str:
        return ("tanh approximation" if self.approximate == "tanh" else "erf form") + f" n={self.n}"

    def c_call(self, ins, out, scratch, direct, dtype):
        if self.approximate == "tanh":
            return [f"host_gelu_tanh({ins[0]}, {out}, {self.n}u, "
                    f"{_c_double(self.c1)}, {_c_double(self.c2)});"]
        return [f"host_gelu_erf({ins[0]}, {out}, {self.n}u, {_c_double(self.k)}, {self.div});"]

    def reference(self, ins, dtype):
        v = np.asarray(ins[0], np.float64).reshape(-1)
        if self.approximate == "tanh":
            y = v * (0.5 * (1.0 + libm("tanh", self.c2 * (v + self.c1 * (v * v * v)))))
        else:
            u = v / self.k if self.div else v * self.k
            y = v * (0.5 * (1.0 + libm("erf", u)))
        return dtype.host_quantize(y).reshape(self.output.shape)


# ------------------------------------------------------------------ #
# Strided copies: Transpose and Slice (Split is lowered to Slices)      #
# ------------------------------------------------------------------ #

@dataclass
class _CopyNode(HostNode):
    helpers: ClassVar[Tuple[str, ...]] = ("copy_nd",)
    dims:   List[int] = field(default_factory=list)     # collapsed, 5 entries
    sstr:   List[int] = field(default_factory=list)     # source strides, 5 entries
    offset: int = 0                                      # source element offset

    def c_file_consts(self, dtype):
        d = ", ".join(f"{v}u" for v in self.dims)
        s = ", ".join(f"{v}u" for v in self.sstr)
        return [f"static const unsigned {self.c_prefix}_d[5] = {{{d}}};"
                f"  /* [{self.index}] {self.onnx_node.op_type} */\n"
                f"static const unsigned {self.c_prefix}_s[5] = {{{s}}};"]

    def c_call(self, ins, out, scratch, direct, dtype):
        src = f"{ins[0]} + {self.offset}u" if self.offset else ins[0]
        return [f"host_copy_nd({src}, {out}, {self.c_prefix}_d, {self.c_prefix}_s);"]


@dataclass
class TransposeNode(_CopyNode):
    """ONNX Transpose (any perm) as a strided host copy; dtype-agnostic
    (integer tensors are copied bit for bit)."""
    perm: List[int] = field(default_factory=list)

    @classmethod
    def from_onnx_node(cls, node, tensors, index, align_elems, ctx: HostContext):
        x = _resolve(tensors, node.input[0], node)
        y = _resolve(tensors, node.output[0], node)
        rank = len(x.shape)
        perm = [int(p) for p in _attrs(node).get("perm", list(range(rank))[::-1])]
        if sorted(perm) != list(range(rank)):
            raise SchedulerError(f"Transpose node '{_label(node)}': bad perm {perm}.")
        out_shape = [x.shape[p] for p in perm]
        if list(y.shape) != out_shape:
            raise SchedulerError(f"Transpose node '{_label(node)}': output shape "
                                 f"{y.shape} != {out_shape}.")
        st = _row_major_strides(x.shape)
        dims, sstr = _collapse(out_shape, [st[p] for p in perm])
        return cls(onnx_node=node, inputs=[x], output=y, index=index,
                   align_elems=align_elems, dims=dims, sstr=sstr, perm=perm)

    def describe(self) -> str:
        return "perm=" + "".join(str(p) for p in self.perm)

    def reference(self, ins, dtype):
        return np.ascontiguousarray(np.transpose(np.asarray(ins[0]), self.perm)).reshape(
            self.output.shape)


@dataclass
class SliceNode(_CopyNode):
    """ONNX Slice with constant starts / ends / axes / positive steps; also
    every output of a ``Split`` (lowered to one Slice per output in
    ``fusion.lower_split``).

    ``is_view`` (decided by ``OnnxGraph._choose_slice_views``): a
    contiguous, 64-byte-aligned piece of an internal buffer becomes a
    zero-cost sub-buffer view (``inference_buf_init_view``) instead of a
    host copy.
    """
    is_view: bool = False
    starts:  List[int] = field(default_factory=list)
    steps:   List[int] = field(default_factory=list)
    out_dims: List[int] = field(default_factory=list)

    @classmethod
    def from_onnx_node(cls, node, tensors, index, align_elems, ctx: HostContext):
        x = _resolve(tensors, node.input[0], node)
        y = _resolve(tensors, node.output[0], node)
        a = _attrs(node)
        rank = len(x.shape)
        if "starts" in a:                                    # opset < 10
            starts, ends = list(a["starts"]), list(a["ends"])
            axes = list(a.get("axes", range(len(starts))))
            steps = [1] * len(starts)
        else:
            def _ints(i, what, default=None):
                if len(node.input) <= i or not node.input[i]:
                    return default
                return [int(v) for v in _const(ctx, tensors, node.input[i], node, what).reshape(-1)]
            starts = _ints(1, "starts")
            ends = _ints(2, "ends")
            axes = _ints(3, "axes", list(range(len(starts))))
            steps = _ints(4, "steps", [1] * len(starts))
        begin = [0] * rank
        step = [1] * rank
        size = list(x.shape)
        for s, e, ax, stp in zip(starts, ends, axes, steps, strict=True):
            ax = ax + rank if ax < 0 else ax
            dim = int(x.shape[ax])
            if stp <= 0:
                raise SchedulerError(f"Slice node '{_label(node)}': step {stp} "
                                     f"(only positive steps are supported).")
            s = s + dim if s < 0 else s
            e = e + dim if e < 0 else e
            s = min(max(s, 0), dim)
            e = min(max(e, 0), dim)
            begin[ax], step[ax] = s, stp
            size[ax] = max(0, -(-(e - s) // stp))
        if list(y.shape) != size:
            raise SchedulerError(f"Slice node '{_label(node)}': output shape {y.shape} "
                                 f"!= computed {size}.")
        st = _row_major_strides(x.shape)
        offset = sum(b * s for b, s in zip(begin, st, strict=True))
        dims, sstr = _collapse(size, [s * k for s, k in zip(st, step, strict=True)])
        return cls(onnx_node=node, inputs=[x], output=y, index=index,
                   align_elems=align_elems, dims=dims, sstr=sstr, offset=offset,
                   starts=begin, steps=step, out_dims=size)

    @property
    def is_contiguous(self) -> bool:
        """The piece is one contiguous run of the source buffer."""
        return (all(k == 1 for k in self.steps)
                and self.dims[:4] == [1, 1, 1, 1] and self.sstr[4] == 1)

    def describe(self) -> str:
        kind = "zero-cost view" if self.is_view else "host copy"
        return f"offset={self.offset} ({kind})"

    def emit_comment(self) -> str:
        if not self.is_view:
            return super().emit_comment()
        return (f"    /* [{self.index}] {self.onnx_node.op_type}({self.inputs[0].onnx_name})"
                f" -> {self.output.onnx_name}  {self.inputs[0].shape} → {self.output.shape}"
                f"  (sub-buffer view at element {self.offset}, no copy) */")

    def reference(self, ins, dtype):
        x = np.asarray(ins[0])
        sl = tuple(slice(b, b + n * k, k) for b, n, k in
                   zip(self.starts, self.out_dims, self.steps, strict=True))
        return np.ascontiguousarray(x[sl]).reshape(self.output.shape)


# ------------------------------------------------------------------ #
# Registry                                                             #
# ------------------------------------------------------------------ #

HOST_OP_FACTORIES = {
    "Softmax":            SoftmaxNode.from_onnx_node,
    "LayerNormalization": LayerNormNode.from_onnx_node,
    "Gelu":               GeluNode.from_onnx_node,
    "Transpose":          TransposeNode.from_onnx_node,
    "Slice":              SliceNode.from_onnx_node,
}

HOST_OP_TYPES: frozenset = frozenset(HOST_OP_FACTORIES)
