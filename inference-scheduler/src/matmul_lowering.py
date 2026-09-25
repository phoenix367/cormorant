"""
MatMul on ConvKernel — the lowering pass of doc/BERT_PLAN.md §2 2A.

``lower_matmuls(graph_nodes, ...)`` replaces MatmulNodes by
``MatmulConvNode``s (nodes.py) where ConvKernel is estimated to be faster
than MatmulKernel.  For ``C[N][M] = A[N][K] · B[K][M]`` the conv has
``out_ch = N``, ``in_ch = K/kw``, a ``1 x kw`` kernel with stride
``(1, kw)`` and an ``out_h x out_w = M`` output; A (row-major) is the conv
weight, B the conv input (see MatmulConvNode for the full contract).

Eligibility (every rule is a hardware or layout fact, the rest is cost):

  * the graph's element type is ap_fixed<16,8> (the bit-exactness argument
    is about the two kernels' ap_fixed<32,16> accumulators);
  * ``outer_count == 1`` (no 4D x 3D outer loop) and a batch pattern
    ConvKernel can express (``_batch_mode``);
  * ``N > 1`` — batch-1 FC layers use one of ConvKernel's 16 output-channel
    lanes and are B-bandwidth bound on MatmulKernel already;
  * ``K % 16 == 0`` (the weight tile has no pad lanes, so A's rows ARE the
    packed layout) and ``M % 8 == 0`` (every row / batch slice of B and C
    starts on a 16-byte word, and no broadcast consumer can give B, C or A
    an alignment-gapped layout);
  * ConvKernel's compile-time bounds: ``out_ch <= max_out_ch``,
    ``in_ch = K/kw <= max_in_ch``, ``out_w · ceil(N/16)·16 <=
    max_acc_persist_entries``, ``kw <= max_kw``.

Kernel width: an activation B (or a constant B that something else also
reads) is used as is, so ``kw = 1``; a constant B read only by this MatMul
is re-laid out at codegen (``conv_lowered_b_image``) and any ``kw`` with
``K % (16 kw) == 0`` is a candidate.  ``kw >= 2`` halves the sweep of a
``kw = 1`` conv: the §2.42 sweep spends ``max(kh·kw, 2)`` cycles per pixel
pair, so a 1x1 pays a dummy second position.

Engine choice (``mode``):
  "auto"   — MatMuls whose conv has at least one full output-channel tile
             (``out_ch >= kTileM`` = 16 rows; below that the conv leaves
             MAC columns idle and both kernels are dominated by fixed
             per-call costs the models only approximate) and whose cheapest
             (kw, out_w) by ``cost_model.conv_cycles`` (plus
             ``CALL_OVERHEAD`` per call) is below ``LOWER_MARGIN`` x
             ``cost_model.matmul_cycles``;
  "always" — every eligible MatMul, cheapest geometry (tests, experiments);
  "off"    — none (CLI ``--no-matmul-on-conv``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from ._conv_hw_config import (
    CONV_MAX_ACC_PERSIST_ENTRIES,
    CONV_MAX_IN_CH,
    CONV_MAX_KW,
    CONV_MAX_LINE_BUF_COLS,
    CONV_MAX_OUT_CH,
    CONV_TILE_IC,
    CONV_TILE_M,
)
from .cost_model import CALL_OVERHEAD, conv_batch_cycles, matmul_cycles
from .nodes import MatmulConvNode, MatmulNode, SchedulerError, conv_lowered_b_image

MODES = ("auto", "always", "off")

# Lower only when the conv estimate is at least 10 % below MatmulKernel's:
# the cycle model is within ~5 % of the RTL on large layers, so a smaller
# predicted gain is not a reliable one.
LOWER_MARGIN = 0.9

# out_w candidates below this are only used when M has no larger divisor
# that fits: narrow rows cost a pipeline ramp per (ict, ow-tile, M-group)
# and waste pair slots, and they make the cost search slow.
_MIN_OUT_W = 8


@dataclass(frozen=True)
class ConvPlan:
    kw:            int
    out_h:         int
    out_w:         int
    conv_n:        int
    conv_batch:    int
    calls:         int
    a_call_stride: int
    b_call_stride: int
    c_call_stride: int
    cycles:        float     # all calls, CALL_OVERHEAD included


def normalize_mode(mode) -> str:
    """``True`` / ``None`` -> "auto", ``False`` -> "off"; strings checked."""
    if mode is True or mode is None:
        return "auto"
    if mode is False:
        return "off"
    if mode not in MODES:
        raise ValueError(f"matmul_on_conv must be one of {MODES}, True or False; got {mode!r}")
    return mode


def _batch_mode(mm: MatmulNode) -> Optional[Tuple[int, int, int, int, int, int]]:
    """How the MatMul's batch maps onto ConvKernel calls:
    ``(conv_n, conv_batch, calls, a_call_stride, b_call_stride,
    c_call_stride)``, or None when the pattern is not expressible."""
    n, k, m, b = mm.n, mm.k, mm.m, mm.batch
    if mm.outer_count != 1:
        return None
    if b <= 1:
        return (n, 1, 1, 0, 0, 0)
    a_s, b_s, c_s = mm.a_batch_stride, mm.b_batch_stride, mm.c_batch_stride
    if c_s != n * m:
        return None
    if a_s == 0 and b_s == k * m:
        return (n, b, 1, 0, 0, 0)                 # shared weights = conv batch
    if b_s == 0 and a_s == n * k:
        if b * n <= CONV_MAX_OUT_CH:
            return (b * n, 1, 1, 0, 0, 0)         # batch folds into the rows
        return (n, 1, b, a_s, 0, c_s)
    if a_s == n * k and b_s == k * m:
        return (n, 1, b, a_s, b_s, c_s)           # one call per batch item
    return None


def ineligible_reason(mm: MatmulNode, is_ap_fixed_16_8: bool = True) -> Optional[str]:
    """Why ``mm`` cannot run on ConvKernel (None when it can)."""
    if not is_ap_fixed_16_8:
        return "element type is not ap_fixed<16,8>"
    if mm.outer_count != 1:
        return "4D x 3D outer loop"
    if mm.n < 2:
        return "N = 1 (FC layer)"
    if mm.k % CONV_TILE_IC:
        return f"K = {mm.k} is not a multiple of {CONV_TILE_IC}"
    if mm.m % 8:
        return f"M = {mm.m} is not a multiple of 8"
    for t in (mm.inputs[0], mm.inputs[1]):
        if t.is_int:
            return f"integer operand {t.onnx_name}"
    bm = _batch_mode(mm)
    if bm is None:
        return "batch pattern"
    conv_n = bm[0]
    if conv_n > CONV_MAX_OUT_CH:
        return f"N = {conv_n} > max_out_ch {CONV_MAX_OUT_CH}"
    if mm.k > CONV_MAX_IN_CH * CONV_MAX_KW:
        return f"K = {mm.k} > max_in_ch x max_kw"
    return None


def _divisors(v: int) -> List[int]:
    small = [d for d in range(1, int(v ** 0.5) + 1) if v % d == 0]
    return sorted(set(small + [v // d for d in small]))


def conv_plans(mm: MatmulNode, kw_options: Sequence[int]) -> List[ConvPlan]:
    """Every admissible (kw, out_w) geometry for ``mm``, cheapest first."""
    bm = _batch_mode(mm)
    if bm is None:
        return []
    conv_n, conv_batch, calls, a_cs, b_cs, c_cs = bm
    n_pad = -(-conv_n // CONV_TILE_M) * CONV_TILE_M
    plans: List[ConvPlan] = []
    for kw in kw_options:
        if kw < 1 or kw > CONV_MAX_KW or kw > CONV_MAX_LINE_BUF_COLS:
            continue
        if mm.k % (CONV_TILE_IC * kw):
            continue
        in_ch = mm.k // kw
        if in_ch > CONV_MAX_IN_CH:
            continue
        widths = [d for d in _divisors(mm.m) if d * n_pad <= CONV_MAX_ACC_PERSIST_ENTRIES]
        wide = [d for d in widths if d >= _MIN_OUT_W]
        for out_w in (wide or widths[-1:]):
            out_h = mm.m // out_w
            per_call = conv_batch_cycles(
                conv_batch, in_ch=in_ch, out_ch=conv_n, in_h=out_h, in_w=kw * out_w,
                oh=out_h, ow=out_w, kh=1, kw=kw, sw=kw)
            plans.append(ConvPlan(kw, out_h, out_w, conv_n, conv_batch, calls,
                                  a_cs, b_cs, c_cs, calls * (per_call + CALL_OVERHEAD)))
    plans.sort(key=lambda p: (p.cycles, p.kw, -p.out_w))
    return plans


def matmul_plan_cycles(mm: MatmulNode) -> float:
    return matmul_cycles(mm.n, mm.k, mm.m, mm.batch * mm.outer_count) + CALL_OVERHEAD


def _readers(nodes) -> Dict[str, list]:
    readers: Dict[str, list] = {}
    for sn in nodes:
        for t in getattr(sn, "inputs", []):
            readers.setdefault(t.onnx_name, []).append(sn)
    return readers


def lower_matmuls(nodes: list, *, mode: str = "auto", is_ap_fixed_16_8: bool = True,
                  graph_io: Sequence[str] = ()) -> Tuple[list, dict]:
    """Return ``(new_nodes, stats)``: ``nodes`` with every MatmulNode that
    should run on ConvKernel replaced by a MatmulConvNode (same index,
    inputs and output), and the constant Bs that use a ``kw > 1`` layout
    re-imaged (``TensorInfo.packed_data``).  ``stats`` =
    ``{"lowered", "kept", "conv_calls", "conv_cycles", "matmul_cycles"}``
    (cycles of the lowered nodes on either engine)."""
    mode = normalize_mode(mode)
    stats = {"lowered": 0, "kept": 0, "conv_calls": 0,
             "conv_cycles": 0.0, "matmul_cycles": 0.0}
    if mode == "off":
        stats["kept"] = sum(isinstance(sn, MatmulNode) for sn in nodes)
        return nodes, stats
    readers = _readers(nodes)
    io = set(graph_io)
    out = []
    for sn in nodes:
        if not isinstance(sn, MatmulNode):
            out.append(sn)
            continue
        if ineligible_reason(sn, is_ap_fixed_16_8) is not None:
            stats["kept"] += 1
            out.append(sn)
            continue
        if mode == "auto" and _batch_mode(sn)[0] < CONV_TILE_M:
            stats["kept"] += 1                        # "N tiny", see the docstring
            out.append(sn)
            continue
        b = sn.inputs[1]
        relayout_ok = (b.data is not None and b.packed_data is None
                       and b.onnx_name not in io
                       and len(readers.get(b.onnx_name, [])) == 1)
        plans = conv_plans(sn, range(1, CONV_MAX_KW + 1) if relayout_ok else (1,))
        mm_cyc = matmul_plan_cycles(sn)
        if not plans or (mode == "auto" and plans[0].cycles >= LOWER_MARGIN * mm_cyc):
            stats["kept"] += 1
            out.append(sn)
            continue
        p = plans[0]
        if p.kw > 1:
            if not relayout_ok:                      # pragma: no cover — kw_options
                raise SchedulerError("internal: kw > 1 for a B that cannot be re-laid out")
            b.packed_data = conv_lowered_b_image(b.data, sn.k, sn.m, p.kw)
            b.packed_note = (f"ConvKernel x image of a MatMul-on-ConvKernel B, kw={p.kw}:"
                             f" x[c][{p.kw}*p + j] = B[(c/16)*{16 * p.kw} + j*16 + c%16][p]"
                             f" = [{sn.k // p.kw}][{sn.m * p.kw}]"
                             + (f" x {b.data.size // (sn.k * sn.m)} slices"
                                if b.data.size != sn.k * sn.m else ""))
        node = MatmulConvNode(
            onnx_node=sn.onnx_node, inputs=list(sn.inputs), output=sn.output,
            index=sn.index, align_elems=sn.align_elems,
            n=sn.n, k=sn.k, m=sn.m, batch=sn.batch,
            a_batch_stride=sn.a_batch_stride, b_batch_stride=sn.b_batch_stride,
            c_batch_stride=sn.c_batch_stride,
            kw=p.kw, out_h=p.out_h, out_w=p.out_w, conv_n=p.conv_n,
            conv_batch=p.conv_batch, calls=p.calls,
            a_call_stride=p.a_call_stride, b_call_stride=p.b_call_stride,
            c_call_stride=p.c_call_stride, b_relayout=p.kw > 1,
            est_conv_cycles=p.cycles, est_matmul_cycles=mm_cyc,
        )
        stats["lowered"] += 1
        stats["conv_calls"] += p.calls
        stats["conv_cycles"] += p.cycles
        stats["matmul_cycles"] += mm_cyc
        out.append(node)
    return out, stats


__all__ = (
    "MODES",
    "LOWER_MARGIN",
    "ConvPlan",
    "normalize_mode",
    "ineligible_reason",
    "conv_plans",
    "matmul_plan_cycles",
    "lower_matmuls",
)
