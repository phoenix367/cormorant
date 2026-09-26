"""
Mixin that forward-simulates the ONNX graph with fixed-point arithmetic.

Every operation is computed in float64 and then truncated/clipped to the
representable grid of ``self._dtype`` after each node — identical to what
the hardware kernel does after each element-wise op.

Quantization model
------------------
Two distinct rounding behaviours are used:

* **dtype.quantize()** — round-to-nearest (used when seeding weight values).
  Weights are written to the .dat / C-array via float_to_storage(), which
  also rounds to nearest; quantize() must match so the simulation uses the
  same value the hardware reads from the DMA buffer.

* **dtype.truncate()** — floor toward −∞ (used after each arithmetic node).
  HLS ap_fixed defaults to AP_TRN (truncation toward −∞) when narrowing an
  arithmetic result back to the element type.  For ADD/SUB/RELU/RELU6 this
  is a no-op because the inputs are already on the representable grid and
  the result inherits the same precision.  For MUL/DIV the intermediate
  result has more fractional bits and truncation matters.

The ramp input used by ``_simulate()`` mirrors the pattern written by the
generated test harness, so the expected output can be embedded verbatim in
test_inference.c for on-device verification.

Extending to a new data type
-----------------------------
Pass a different ``dtype`` to ``CodeGenerator.__init__``.  No changes to this
file are needed: all type-specific logic is encapsulated in ``DataType``.
"""

from __future__ import annotations
from collections import namedtuple
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..nodes  import (
    OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_RELU, OP_RELU6,
    ACT_RELU, ACT_RELU6,
    MatmulNode, MatmulConvNode, ConvNode, PoolNode, ReshapeNode, SpaceToDepthNode,
    POOL_MAX, POOL_AVG,
)
from ..tensor import TensorInfo
from ..host_nodes import GatherNode, HostNode, OneHotNode
from ..llm_nodes import LlmEmbedNode

# Expected GT arrays larger than this threshold are written to external


ResidualStats = namedtuple("ResidualStats", ("abs_max", "nrmse", "sqnr_db"))


def _residual_stats(full: np.ndarray, quant: np.ndarray,
                    *, eps: float = 1e-12) -> ResidualStats:
    """Tensor-level summary of the residual ``r = full − quant``.

    Returns three numbers:

    * ``abs_max``  — ``max |r|`` over all elements; the tightest bound on
      any single-element error.  Useful for verifying the dtype's
      theoretical LSB ceiling.
    * ``nrmse``    — normalised RMS error, ``‖r‖₂ / ‖full‖₂``.  Treats
      the residual as noise and the original as signal, weighted by
      magnitude — so a single near-zero element cannot dominate, unlike
      max-elementwise relative error.
    * ``sqnr_db``  — signal-to-quantisation-noise ratio in decibels,
      ``20·log₁₀(‖full‖₂ / ‖r‖₂)``.  Higher is better.  This is the
      standard quality metric for fixed-point quantisation.

    Edge cases:
      * All-zero ``full`` and zero residual: ``nrmse = 0``, ``sqnr_db = inf``.
      * All-zero ``full`` with non-zero residual (impossible for true
        quantisation, defensive only): ``nrmse = inf``, ``sqnr_db = -inf``.
    """
    full  = np.asarray(full,  dtype=np.float64)
    quant = np.asarray(quant, dtype=np.float64).reshape(full.shape)
    r = full - quant

    abs_max = float(np.abs(r).max()) if r.size else 0.0
    norm_r  = float(np.linalg.norm(r.flat))
    norm_w  = float(np.linalg.norm(full.flat))

    if norm_w <= eps:
        # Tensor is essentially zero — relative metrics aren't meaningful.
        if norm_r <= eps:
            return ResidualStats(abs_max, 0.0, float("inf"))
        return ResidualStats(abs_max, float("inf"), float("-inf"))

    nrmse = norm_r / norm_w
    if norm_r <= eps:
        sqnr_db = float("inf")
    else:
        sqnr_db = 20.0 * np.log10(norm_w / norm_r)
    return ResidualStats(abs_max, nrmse, float(sqnr_db))


def _conv2d_ref(
    x: np.ndarray,        # float64, shape [N, C, H, W]
    w: np.ndarray,        # float64, shape [M, C, kH, kW]
    bias,                 # float64 shape [M] or None
    stride_h: int,
    stride_w: int,
    pad_top: int,
    pad_left: int,
    dilation_h: int,
    dilation_w: int,
    out_h: int,
    out_w: int,
) -> np.ndarray:
    """Reference 2-D standard convolution (group=1) in float64."""
    N, C, H, W = x.shape
    M, _, kH, kW = w.shape
    y = np.zeros((N, M, out_h, out_w), dtype=np.float64)

    for oh in range(out_h):
        for ow in range(out_w):
            for khi in range(kH):
                ih = oh * stride_h + khi * dilation_h - pad_top
                if ih < 0 or ih >= H:
                    continue
                for kwi in range(kW):
                    iw = ow * stride_w + kwi * dilation_w - pad_left
                    if iw < 0 or iw >= W:
                        continue
                    # x[:, :, ih, iw]  shape [N, C]
                    # w[:, :, khi, kwi] shape [M, C]
                    # einsum('nc,mc->nm') → [N, M]
                    y[:, :, oh, ow] += np.einsum(
                        "nc,mc->nm", x[:, :, ih, iw], w[:, :, khi, kwi]
                    )

    if bias is not None:
        y += bias.reshape(1, M, 1, 1)

    return y


def _depthwise_conv2d_ref(
    x: np.ndarray,        # float64, shape [N, C, H, W]
    w: np.ndarray,        # float64, shape [C, 1, kH, kW]  (depthwise layout)
    bias,                 # float64 shape [C] or None
    stride_h: int,
    stride_w: int,
    pad_top: int,
    pad_left: int,
    dilation_h: int,
    dilation_w: int,
    out_h: int,
    out_w: int,
) -> np.ndarray:
    """Reference depthwise 2-D convolution (group=in_ch) in float64."""
    N, C, H, W = x.shape
    kH, kW = w.shape[2], w.shape[3]
    y = np.zeros((N, C, out_h, out_w), dtype=np.float64)

    for oh in range(out_h):
        for ow in range(out_w):
            for khi in range(kH):
                ih = oh * stride_h + khi * dilation_h - pad_top
                if ih < 0 or ih >= H:
                    continue
                for kwi in range(kW):
                    iw = ow * stride_w + kwi * dilation_w - pad_left
                    if iw < 0 or iw >= W:
                        continue
                    # x[:, :, ih, iw]  shape [N, C]
                    # w[:, 0, khi, kwi] shape [C]  — one filter per channel
                    y[:, :, oh, ow] += x[:, :, ih, iw] * w[:, 0, khi, kwi].reshape(1, C)

    if bias is not None:
        y += bias.reshape(1, C, 1, 1)

    return y
def _quantize_trn(v: np.ndarray | float, frac_bits: int) -> np.ndarray | float:
    """AP_TRN-style truncation: floor toward -∞ at the given fractional bit
    width.  Mirrors HLS ap_fixed<W, I, AP_TRN> narrowing semantics."""
    scale = float(1 << frac_bits) if frac_bits >= 0 else 1.0 / (1 << -frac_bits)
    return np.floor(v * scale) / scale


def _pool_poly_sqrt(acc: np.ndarray) -> np.ndarray:
    """Bit-accurate float64 mirror of PoolingKernel.cpp's poly_sqrt().

    The hardware kernel replaces sqrtf() with a fully fixed-point 3rd-order
    polynomial approximation under range reduction (x = m × 4^k, m ∈ [1, 4)).
    This helper applies the IDENTICAL coefficients, range reduction, AND
    intermediate ap_fixed truncations so the generated expected-output
    fixtures match the kernel's RTL output without any 1-LSB drift.
    """
    out = np.zeros_like(acc, dtype=np.float64)
    pos = acc > 0.0
    if not np.any(pos):
        return out
    a = acc[pos]

    # Range-reduce: m = a / 4^k, m ∈ [1, 4).  np.frexp returns m, e where
    # a = m * 2^e with m ∈ [0.5, 1).  Convert to a = m' * 4^k:
    m, e = np.frexp(a)
    k = np.floor_divide(e, 2)
    m = m * np.exp2(e - 2 * k)
    bump = m < 1.0
    m = np.where(bump, m * 4.0, m)
    k = np.where(bump, k - 1, k)

    # Mirror the kernel's ap_ufixed<18,2> on m (16 frac bits).
    m = _quantize_trn(m, 16)

    # Coefficients in ap_fixed<16,1> (15 frac bits).
    c0 = _quantize_trn( 0.4434, 15)
    c1 = _quantize_trn( 0.6432, 15)
    c2 = _quantize_trn(-0.0943, 15)
    c3 = _quantize_trn( 0.0077, 15)

    # Horner's scheme — each intermediate truncated to ap_fixed<24,4>
    # (20 frac bits) to mirror the kernel exactly.
    t1     = _quantize_trn(c3 * m + c2, 20)
    t2     = _quantize_trn(t1 * m + c1, 20)
    sqrt_m = _quantize_trn(t2 * m + c0, 20)

    # Apply 2^k scaling, then quantize to AccData_t = ap_fixed<32,16>
    # (16 frac bits).
    out[pos] = _quantize_trn(np.ldexp(sqrt_m, k), 16)
    return out


def _pool2d_ref(
    x: np.ndarray,       # float64, shape [N, C, H, W]
    pool_h: int,
    pool_w: int,
    stride_h: int,
    stride_w: int,
    pad_top: int,
    pad_left: int,
    dil_h: int,
    dil_w: int,
    out_h: int,
    out_w: int,
    pool_type: int,
    lp_order: int,
    count_include_pad: int,
) -> np.ndarray:
    """Reference 2-D pooling in float64 matching PoolingKernel's NCHW output.

    LP-Pool p=2 finalize uses _pool_poly_sqrt (mirrors the kernel's
    polynomial sqrt approximation) so generated expected-output fixtures
    line up with the kernel's actual output rather than with an idealised
    np.sqrt that the hardware no longer computes.
    """
    N, C, H, W = x.shape
    y = np.zeros((N, C, out_h, out_w), dtype=np.float64)

    for oh in range(out_h):
        for ow in range(out_w):
            valid_count = 0
            for khi in range(pool_h):
                for kwi in range(pool_w):
                    ih = oh * stride_h + khi * dil_h - pad_top
                    iw = ow * stride_w + kwi * dil_w - pad_left
                    if 0 <= ih < H and 0 <= iw < W:
                        valid_count += 1
            denom = (pool_h * pool_w) if count_include_pad else valid_count
            denom = max(denom, 1)

            acc = (
                np.full((N, C), -np.inf, dtype=np.float64)
                if pool_type == POOL_MAX
                else np.zeros((N, C), dtype=np.float64)
            )

            for khi in range(pool_h):
                for kwi in range(pool_w):
                    ih = oh * stride_h + khi * dil_h - pad_top
                    iw = ow * stride_w + kwi * dil_w - pad_left
                    if ih < 0 or ih >= H or iw < 0 or iw >= W:
                        continue
                    v = x[:, :, ih, iw]
                    if pool_type == POOL_MAX:
                        acc = np.maximum(acc, v)
                    elif pool_type == POOL_AVG:
                        acc += v
                    else:
                        acc += np.abs(v) if lp_order == 1 else v * v

            if pool_type == POOL_MAX:
                y[:, :, oh, ow] = acc
            elif pool_type == POOL_AVG:
                # Mirror PoolingKernel.cpp's fixed-point AVG-pool divide:
                # the kernel substitutes `acc / d` with `acc *
                # inv_denom_lut[d]` where the LUT entry is
                #   raw = round(2^23 / d)  (computed via the +d/2 integer
                #                           round-to-nearest trick)
                #   inv_denom_q = raw / 2^23   as ap_ufixed<24,1>.
                # That introduces a ≤ 1-LSB-of-Data_t deviation from a true
                # float divide on ~2.5% of inputs.  test_inference.c
                # compares output bytes for strict equality against the
                # expected fixtures generated here, so we must replicate
                # the exact kernel arithmetic — otherwise legitimate
                # kernel output bytes would be flagged as mismatches.
                # See PoolingKernel.cpp::inv_denom_lookup.
                raw = ((1 << 23) + denom // 2) // denom
                inv_denom_q = raw / float(1 << 23)
                y[:, :, oh, ow] = acc * inv_denom_q
            else:
                y[:, :, oh, ow] = (
                    acc if lp_order == 1
                    else _pool_poly_sqrt(np.maximum(acc, 0.0))
                )

    return y


# expected/<c_name>.dat files and loaded at runtime by fread(), matching
# the same approach used for large weight tensors.
LARGE_EXPECTED_THRESHOLD = 4096


class _SimulateMixin:
    """
    Fixed-point graph simulation and expected-output helpers.

    Public API
    ----------
    simulate(inputs)   — user-supplied float64 arrays → output dict
    _simulate()        — ramp inputs (matching test harness) → all arrays
    _expected_storage  — convert simulated logical array to raw storage array
    _emit_expected_c   — produce the static C array declaration for embedding
    """

    # ------------------------------------------------------------------ #
    # Public: simulate with caller-supplied inputs                        #
    # ------------------------------------------------------------------ #

    def simulate(
        self,
        inputs: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """
        Forward-pass the graph with the given float inputs.

        Parameters
        ----------
        inputs : {onnx_name: ndarray}
            Float64 arrays in each tensor's logical shape.  Values should be
            representable by the active data type; the simulation quantizes at
            each node boundary exactly as the hardware does, but assumes
            inputs are already on the representable grid.

        Returns
        -------
        {onnx_name: ndarray}  for each graph output (float64, logical shape).
        """
        arrays = self._forward_pass(inputs)
        return {t.onnx_name: arrays[t.onnx_name]
                for t in self._graph.output_tensors}

    # ------------------------------------------------------------------ #
    # Internal: ramp-input simulation (used by generate_test)             #
    # ------------------------------------------------------------------ #

    def compute_quant_errors(self) -> Dict[str, Tuple[float, float]]:
        """Run the ramp-input simulation and return ``{output_name:
        (abs_max, rel_max)}`` for every kernel-bearing node.

        ReshapeNodes have no truncation (pure buffer alias) and are
        omitted.  Used by the report generator to surface per-layer
        quantization residuals when the active dtype is fixed-point.
        Always runs in float64; the figures are independent of any
        runtime numpy quirks on the deployment platform.
        """
        ramp_inputs = self._build_ramp_inputs()
        errors: Dict[str, Tuple[float, float]] = {}
        self._forward_pass(ramp_inputs, errors_out=errors)
        return errors

    def _build_ramp_inputs(self) -> Dict[str, np.ndarray]:
        """Construct the same ramp inputs ``_simulate()`` uses, factored
        out so ``compute_quant_errors()`` can share the seeding logic
        without duplicating it."""
        dtype = self._dtype
        ramp_inputs: Dict[str, np.ndarray] = {}
        for t in self._graph.input_tensors:
            idx = np.arange(t.numel, dtype=np.int64)
            if t.is_host:
                ramp_inputs[t.onnx_name] = self._host_fill_values(t).reshape(t.shape)
                continue
            if t.exp is not None:
                ramp_inputs[t.onnx_name] = dtype.ramp_to_float_exp(
                    idx, t.exp_full(dtype.frac_bits).reshape(-1)).reshape(t.shape)
                continue
            if t.is_int:
                # Integer inputs (ids, masks): p[i] = i % R, a valid index
                # for every Gather / OneHot that reads them.
                r = self._int_fill_range(t)
                ramp_inputs[t.onnx_name] = (idx % r).astype(np.float64).reshape(t.shape)
                continue
            lay = self._layouts.get(t.onnx_name)
            if lay and lay.n_chunks > 1:
                positions = (idx // lay.chunk) * lay.stride + (idx % lay.chunk)
            else:
                positions = idx
            ramp_inputs[t.onnx_name] = dtype.ramp_to_float(positions).reshape(t.shape)
        return ramp_inputs

    def _host_fill_values(self, t: TensorInfo) -> np.ndarray:
        """Values the test harness writes into a host-memory graph input
        (C: ``_host_fill_c``): i32 — the ``test_fill`` constant of the
        model's numeric metadata, else ``i % R`` (R as for integer tensors);
        f32 — ``(i % 17 - 8) * 0.25``; i16 — raw ``i`` at the exponent."""
        i = np.arange(t.numel, dtype=np.int64)
        if t.host == "i32":
            fill = self._graph.numeric.get("test_fill", {}).get(t.onnx_name)
            if fill is not None:
                return np.full(t.numel, float(int(fill)))
            return (i % self._int_fill_range(t)).astype(np.float64)
        if t.host == "f32":
            return ((i % 17) - 8).astype(np.float64) * 0.25
        return self._dtype.ramp_to_float_exp(i, t.exp_full(self._dtype.frac_bits).reshape(-1))

    def _host_fill_c(self, t: TensorInfo, ptr: str) -> List[str]:
        """C lines filling host graph input ``t`` (see _host_fill_values)."""
        n = t.numel
        if t.host == "i32":
            fill = self._graph.numeric.get("test_fill", {}).get(t.onnx_name)
            rhs = f"(int32_t){int(fill)}" if fill is not None else \
                f"(int32_t)(i % {self._int_fill_range(t)}u)"
            return [f"    for (i = 0u; i < {n}u; i++) {ptr}[i] = {rhs};"]
        if t.host == "f32":
            return [f"    for (i = 0u; i < {n}u; i++) {ptr}[i] = "
                    f"(float)((int)(i % 17u) - 8) * 0.25f;"]
        return [f"    for (i = 0u; i < {n}u; i++) {ptr}[i] = (int16_t)(uint16_t)(i & 0xFFFFu);"]

    def _int_fill_range(self, t: TensorInfo) -> int:
        """R of the test-harness fill ``p[i] = i % R`` for integer graph input
        ``t``: the smallest table size (Gather rows / OneHot depth) among the
        host ops that read it through Reshape aliases, else 2 (a 0/1 pattern,
        right for masks); capped to the positive range of the storage."""
        names, frontier = {t.onnx_name}, [t.onnx_name]
        children: Dict[str, List[str]] = {}
        for sn in self._graph.nodes:
            if self._is_alias_node(sn):
                children.setdefault(sn.inputs[0].onnx_name, []).append(sn.output.onnx_name)
        while frontier:
            for c in children.get(frontier.pop(), []):
                if c not in names:
                    names.add(c)
                    frontier.append(c)
        bounds = []
        for sn in self._graph.nodes:
            if isinstance(sn, GatherNode) and sn.inputs[1].onnx_name in names:
                bounds.append(sn.rows)
            elif isinstance(sn, LlmEmbedNode) and sn.inputs[0].onnx_name in names:
                bounds.append(sn.rows)
            elif isinstance(sn, OneHotNode) and sn.inputs[0].onnx_name in names:
                bounds.append(sn.depth)
        r = min(bounds) if bounds else 2
        cap = (1 << (8 * self._dtype.bytes_per_elem - 1)) - 1
        return max(1, min(r, cap))

    def _simulate(self) -> Dict[str, np.ndarray]:
        """
        Forward-pass with the ramp inputs that generate_test() writes into
        the DMA buffers:

            p[pos] = (Data_t)(pos & mask)     (C test harness fill)

        The DataType converts these bit patterns to float64 values that match
        what the hardware reads from the buffer.  For broadcast tensors, the
        logical element at (chunk c, offset j) occupies buffer position
        ``c * aligned_chunk + j``.

        Returns
        -------
        {onnx_name: ndarray}  for every tensor visited (inputs, weights,
        intermediates, outputs), float64 in logical shape.
        """
        return self._forward_pass(self._build_ramp_inputs())

    # ------------------------------------------------------------------ #
    # Core: topological forward pass                                      #
    # ------------------------------------------------------------------ #

    def _forward_pass(
        self,
        input_arrays: Dict[str, np.ndarray],
        errors_out: Optional[Dict[str, "tuple"]] = None,
        states: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Run every ScheduledNode in topological order, quantizing outputs with
        ``self._dtype.quantize()`` at each step.  Numpy broadcasting handles
        the same trailing-contiguous rules as the hardware broadcast loop.

        If ``errors_out`` is provided, each kernel-bearing node's output is
        accompanied by an ``(abs_max, rel_max)`` tuple measuring the
        truncation residual at that node — i.e. ``full_precision_result -
        dtype.truncate(...)``.  ReshapeNodes (no truncation) are skipped.

        ``states`` ({name: float64 array}) holds the persistent state
        tensors (src/numeric.py); they are updated IN PLACE, so a caller can
        run several passes (and several entries of a multi-entry project)
        over the same states.  Default: fresh copies of the initial values.

        Returns {onnx_name: float64 ndarray} for every tensor visited.
        """
        dtype   = self._dtype
        arrays: Dict[str, np.ndarray] = {}
        if states is None:
            states = self.initial_states()

        def _store_quant(name, full, truncate_fn=dtype.truncate, shape=None):
            quant = truncate_fn(full)
            if shape is not None:
                quant = quant.reshape(shape)
            arrays[name] = quant
            if errors_out is not None:
                full_r = full.reshape(shape) if shape is not None else full
                errors_out[name] = _residual_stats(full_r, quant)
            return quant

        # Seed with quantized weights (mirrors the ROM encoding written to C);
        # a weight encoded at a rank-1 exponent holds raw / 2^F in `data`
        for t in self._graph.weight_tensors:
            if t.data is not None:
                arrays[t.onnx_name] = dtype.quantize(
                    t.data.reshape(t.shape).astype(np.float64)
                )
                if t.wexp is not None:
                    arrays[t.onnx_name] = arrays[t.onnx_name] * np.power(
                        2.0, (dtype.frac_bits - t.wexp).astype(np.float64))

        # Persistent states (updated in place by the host ops)
        for t in self._graph.state_tensors:
            arrays[t.onnx_name] = states[t.onnx_name]

        # Seed with caller-supplied inputs (already quantized by convention)
        arrays.update(input_arrays)

        # Node-by-node forward pass
        for sn in self._graph.nodes:
            if isinstance(sn, (MatmulNode, MatmulConvNode)):
                # A MatMul lowered onto ConvKernel (MatmulConvNode) computes
                # exactly what MatmulKernel computes: exact Q8.8 products,
                # an ap_fixed<32,16> sum, floor + saturate — one model for
                # both engines (test_matmul_on_conv.py checks the conv view).
                a = arrays[sn.inputs[0].onnx_name]
                b = arrays[sn.inputs[1].onnx_name]
                if any(t.exp is not None or t.wexp is not None
                       for t in (sn.inputs[0], sn.inputs[1], sn.output)):
                    self._matmul_exp(sn, a, b, arrays, errors_out)
                    continue
                _store_quant(sn.output.onnx_name, np.matmul(a, b),
                             shape=sn.output.shape)
                continue

            if isinstance(sn, ConvNode):
                conv_fn = _depthwise_conv2d_ref if sn.is_depthwise else _conv2d_ref
                full = conv_fn(
                    x=arrays[sn.inputs[0].onnx_name],
                    w=arrays[sn.inputs[1].onnx_name],
                    bias=(arrays[sn.inputs[2].onnx_name]
                          if sn.has_bias else None),
                    stride_h=sn.stride_h,
                    stride_w=sn.stride_w,
                    pad_top=sn.pad_top,
                    pad_left=sn.pad_left,
                    dilation_h=sn.dilation_h,
                    dilation_w=sn.dilation_w,
                    out_h=sn.out_h,
                    out_w=sn.out_w,
                )
                _store_quant(sn.output.onnx_name, full, shape=sn.output.shape)
                continue

            if isinstance(sn, ReshapeNode):
                src = arrays[sn.inputs[0].onnx_name]
                arrays[sn.output.onnx_name] = src.reshape(sn.output.shape)
                continue

            if isinstance(sn, HostNode):
                # Host-CPU op: double arithmetic, round-half-even + saturate
                # on write-back — the same operations, in the same order, as
                # the generated C helper (host_nodes.py / llm_nodes.py).
                y = sn.reference([arrays[t.onnx_name] for t in sn.inputs], dtype)
                if sn.output.is_state:                 # written in place
                    arrays[sn.output.onnx_name][...] = np.asarray(y).reshape(
                        arrays[sn.output.onnx_name].shape)
                else:
                    arrays[sn.output.onnx_name] = y
                continue

            if isinstance(sn, SpaceToDepthNode):
                # Pure reorder (ONNX SpaceToDepth channel order), no truncation.
                bs = sn.blocksize
                src = arrays[sn.inputs[0].onnx_name].reshape(
                    sn.batch, sn.in_ch, sn.in_h // bs, bs, sn.in_w // bs, bs)
                arrays[sn.output.onnx_name] = np.ascontiguousarray(
                    src.transpose(0, 3, 5, 1, 2, 4)).reshape(sn.output.shape)
                continue

            if isinstance(sn, PoolNode):
                full = _pool2d_ref(
                    x=arrays[sn.inputs[0].onnx_name],
                    pool_h=sn.pool_h,
                    pool_w=sn.pool_w,
                    stride_h=sn.stride_h,
                    stride_w=sn.stride_w,
                    pad_top=sn.pad_top,
                    pad_left=sn.pad_left,
                    dil_h=sn.dil_h,
                    dil_w=sn.dil_w,
                    out_h=sn.out_h,
                    out_w=sn.out_w,
                    pool_type=sn.pool_type,
                    lp_order=sn.lp_order,
                    count_include_pad=sn.count_include_pad,
                )
                _store_quant(sn.output.onnx_name, full, shape=sn.output.shape)
                continue

            a = arrays[sn.inputs[0].onnx_name]

            # DIV: HLS computes a_int / b_int (C integer division =
            # truncation toward zero), so use truncate_div instead of
            # the default truncate (floor toward −∞).
            truncate_fn = dtype.truncate
            if sn.op_code == OP_ADD:
                result = a + arrays[sn.inputs[1].onnx_name]
            elif sn.op_code == OP_SUB:
                result = a - arrays[sn.inputs[1].onnx_name]
            elif sn.op_code == OP_MUL:
                result = a * arrays[sn.inputs[1].onnx_name]
            elif sn.op_code == OP_DIV:
                result = a / arrays[sn.inputs[1].onnx_name]
                truncate_fn = dtype.truncate_div
            elif sn.op_code == OP_RELU:
                result = np.maximum(a, 0.0)
            elif sn.op_code == OP_RELU6:
                result = np.minimum(np.maximum(a, 0.0), 6.0)
            else:
                raise ValueError(
                    f"_forward_pass: unknown op_code {sn.op_code} "
                    f"in node '{sn.onnx_node.name or sn.onnx_node.op_type}'"
                )

            # Fused activation (kernel `act` register).  Clipping commutes
            # with the saturating truncation (0 and 6 are representable),
            # so applying it before quantisation matches the hardware.
            if sn.act == ACT_RELU:
                result = np.maximum(result, 0.0)
            elif sn.act == ACT_RELU6:
                result = np.minimum(np.maximum(result, 0.0), 6.0)

            _store_quant(sn.output.onnx_name, result,
                         truncate_fn=truncate_fn, shape=sn.output.shape)

        return arrays

    # ------------------------------------------------------------------ #
    # Power-of-two exponents and states (src/numeric.py)                   #
    # ------------------------------------------------------------------ #

    def initial_states(self) -> Dict[str, np.ndarray]:
        """Fresh float64 copies of every state tensor's initial value."""
        out = {}
        for t in self._graph.state_tensors:
            init = t.init_data
            out[t.onnx_name] = (np.zeros(t.shape) if init is None
                                else np.array(init, np.float64).reshape(t.shape))
        return out

    def _matmul_exp(self, sn, a, b, arrays, errors_out) -> None:
        """A MatMul whose tensors carry power-of-two exponents: the kernel
        sums raw products exactly in ap_fixed<32,16> — an int32 that wraps —
        and writes floor(acc / 2^F) saturated; the value is raw * 2^-f_out.
        Every column's products share the scale 2^-(f_out[j] + F) (the
        rank-1 weight exponent), so the float64 value sum is exact."""
        dtype = self._dtype
        F = dtype.frac_bits
        full = np.matmul(a, b)
        fo = sn.output.exp_full(F).astype(np.float64)
        acc = full * np.power(2.0, fo + F)                     # raw Q16.16-like units
        if not np.array_equal(acc, np.round(acc)):
            raise ValueError(f"MatMul '{sn.onnx_node.name}': exponents are inconsistent "
                             f"(accumulator not on the 2^-(f_out+F) grid)")
        big = np.abs(acc) >= 2.0 ** 31
        if big.any():                                          # ap_fixed<32,16> wraps
            acc = np.where(big, np.mod(acc + 2.0 ** 31, 2.0 ** 32) - 2.0 ** 31, acc)
        lo, hi = dtype.raw_range
        q = np.clip(np.floor(acc / float(1 << F)), lo, hi) / np.power(2.0, fo)
        arrays[sn.output.onnx_name] = q.reshape(sn.output.shape)
        if errors_out is not None:
            errors_out[sn.output.onnx_name] = _residual_stats(full.reshape(q.shape), q)

    # ------------------------------------------------------------------ #
    # Helpers: convert simulated output → C array                         #
    # ------------------------------------------------------------------ #

    def _expected_storage(
        self,
        name: str,
        logical: np.ndarray,
    ) -> np.ndarray:
        """
        Convert a logical float64 array to the strided DMA-buffer layout that
        the hardware writes, encoded as the active data type's storage dtype.

        For broadcast tensors data elements occupy strided positions; gap slots
        (alignment padding) are zero.  For flat tensors the array is contiguous.

        Parameters
        ----------
        name    : onnx_name of the tensor
        logical : float64 ndarray in logical (non-strided) shape

        Returns
        -------
        ndarray with dtype == self._dtype.np_storage, length _alloc_sizes[name]
        """
        dtype = self._dtype
        t = self._graph._tensors.get(name)
        flat  = logical.flatten()
        if t is not None and t.is_host:
            if t.host == "f32":
                return flat.astype(np.float32)
            if t.host == "i32":
                return flat.astype(np.int32)
            return dtype.exp_to_storage(flat, t.exp_full(dtype.frac_bits).reshape(-1)) \
                .view(np.int16)
        lay   = self._layouts.get(name)
        alloc = lay.alloc if lay is not None else len(logical.flatten())
        buf   = np.zeros(alloc, dtype=dtype.np_storage)
        if t is not None and t.is_int:
            encoded = dtype.int_to_storage(flat.astype(np.float64))   # raw integers
        elif t is not None and t.exp is not None:
            encoded = dtype.exp_to_storage(flat, t.exp_full(dtype.frac_bits).reshape(-1))
        else:
            encoded = dtype.float_to_storage(flat.astype(np.float64))

        if lay and lay.n_chunks > 1:
            idx      = np.arange(len(flat), dtype=np.int64)
            pos      = (idx // lay.chunk) * lay.stride + (idx % lay.chunk)
            buf[pos] = encoded
        else:
            buf[:len(flat)] = encoded

        return buf

    def _emit_expected_c(
        self,
        c_name: str,
        storage_buf: np.ndarray,
    ) -> str:
        """
        Emit a ``static const <type> expected_<c_name>[N]`` array suitable
        for embedding in test_inference.c.

        Eight elements per row, same style as the weight ROM arrays.
        The array type and literal format are determined by the active DataType.
        """
        dtype = self._dtype
        n     = len(storage_buf)
        rows  = []
        for i in range(0, n, 8):
            chunk = storage_buf[i : i + 8]
            rows.append("    " + ", ".join(dtype.format_literal(v) for v in chunk))
        inner = ",\n".join(rows)
        return (
            f"static const {dtype.c_array_type} expected_{c_name}[{n}] = {{\n"
            f"{inner}\n"
            f"}};\n"
        )

    def _emit_expected_host_c(self, t, values) -> str:
        """``static const <ctype> expected_<c>[N]`` for a host-memory output
        (float32 literals that round-trip exactly, or integers)."""
        from ..host_nodes import _c_float
        ctype = self._host_c_type(t)
        v = np.asarray(values).reshape(-1)
        if t.host == "f32":
            lits = [_c_float(float(x)) for x in v.astype(np.float32)]
        else:
            lits = [str(int(x)) for x in v]
        rows = ",\n".join("    " + ", ".join(lits[i:i + 8]) for i in range(0, len(lits), 8))
        return f"static const {ctype} expected_{t.c_name}[{len(lits)}] = {{\n{rows}\n}};\n"

    # ------------------------------------------------------------------ #
    # Large expected: external .dat files                                  #
    # ------------------------------------------------------------------ #

    @property
    def large_expected_tensors(self) -> List[TensorInfo]:
        """
        Output tensors whose GT storage buffer exceeds LARGE_EXPECTED_THRESHOLD
        elements.  These are loaded from expected/<c_name>.dat at runtime
        instead of being embedded as static C arrays.

        Returns an empty list when embed_large_expected=True (all arrays inlined).
        """
        if self._embed_large_expected:
            return []
        result = []
        for t in self._graph.output_tensors:
            if self._alloc_sizes[t.onnx_name] > LARGE_EXPECTED_THRESHOLD:
                result.append(t)
        return result

    def generate_expected_dat(self, tensor: TensorInfo) -> bytes:
        """
        Return raw little-endian bytes for expected/<c_name>.dat.

        Serialises the storage buffer produced by ``_expected_storage()``
        using the same byte order as the weight .dat files, so the C
        ``_load_expected()`` helper can fread() it directly into a Data_t array.
        """
        sim_arrays  = self._simulate()
        storage_buf = self._expected_storage(
            tensor.onnx_name, sim_arrays[tensor.onnx_name]
        )
        return storage_buf.flatten().astype(
            storage_buf.dtype.newbyteorder("<")
        ).tobytes()
