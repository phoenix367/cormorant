"""
MatMul on ConvKernel — the lowering pass of doc/plans/BERT_PLAN.md §2 2A.

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

Row split: the accumulator holds ``max_acc_persist_entries`` outputs, so a
call with many rows (out_ch) sweeps fewer output rows per chunk than the
line buffer holds (``max_line_buf_rows``) — at 1024 rows and ``out_w = 8``
only 8 of 16.  Such a plan re-fetches its weight slabs twice as often and
the board runs it 1.3x slower than the cycle model predicts (SmolVLM's
1024-token vision linears: 25.2 / 100.6 / 96.6 ms vs 2 x 6.9 / 2 x 29.3 /
2 x 25.3 ms in two 512-row calls, doc/plans/CHAT_PLAN.md §24).  When EVERY
single-call plan of a MatMul with contiguous rows is limited that way, the
rows are split over several calls (call i reads A rows and writes C rows
``[i·r, (i+1)·r)``, B shared) and the cheapest split plan that is not
limited is taken; no other MatMul changes.

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
from typing import Dict, List, Optional, Sequence, Tuple, Union

from ._conv_hw_config import (
    CONV_MAX_ACC_PERSIST_ENTRIES,
    CONV_MAX_IN_CH,
    CONV_MAX_KW,
    CONV_MAX_LINE_BUF_COLS,
    CONV_MAX_LINE_BUF_ROWS,
    CONV_MAX_OUT_CH,
    CONV_TILE_IC,
    CONV_TILE_M,
)
from . import cost_model
from .cost_model import CALL_OVERHEAD, conv_batch_cycles, gemv_cycles, matmul_cycles
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
    acc_limited:   bool = False   # see "Row split" in the module docstring


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


def _acc_limited(conv_n: int, out_h: int, out_w: int) -> bool:
    """The accumulator caps a chunk below the rows the line buffer holds
    (cost_model._conv_geom's ``per`` for a 1 x kw kernel)."""
    n_pad = -(-conv_n // CONV_TILE_M) * CONV_TILE_M
    per = max(1, min(out_h, CONV_MAX_ACC_PERSIST_ENTRIES // (out_w * n_pad)))
    return per < min(out_h, CONV_MAX_LINE_BUF_ROWS)


def _geometry_plans(mm: MatmulNode, kw_options: Sequence[int], conv_n: int,
                    conv_batch: int, calls: int, a_cs: int, b_cs: int,
                    c_cs: int) -> List[ConvPlan]:
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
                                  a_cs, b_cs, c_cs, calls * (per_call + CALL_OVERHEAD),
                                  _acc_limited(conv_n, out_h, out_w)))
    plans.sort(key=lambda p: (p.cycles, p.kw, -p.out_w))
    return plans


def conv_plans(mm: MatmulNode, kw_options: Sequence[int],
               splits: str = "auto") -> List[ConvPlan]:
    """Every admissible (kw, out_w) geometry for ``mm``, cheapest first —
    or, when every one of them is accumulator-limited and the rows are
    contiguous (one call, ConvKernel batch 1), the row-split plans that
    are not, cheapest first, ahead of them ("Row split" above).

    ``splits="all"`` (the planner's candidates, src/tactics.py): every
    one-call plan and every row-split plan of a contiguous-rows MatMul,
    accumulator-limited or not, cheapest first."""
    bm = _batch_mode(mm)
    if bm is None:
        return []
    conv_n, conv_batch, calls, a_cs, b_cs, c_cs = bm
    plans = _geometry_plans(mm, kw_options, *bm)
    contiguous = calls == 1 and conv_batch == 1
    if splits == "all":
        if not contiguous:
            return plans
        out = list(plans)
        for rows in _divisors(conv_n):
            if rows == conv_n or rows % CONV_TILE_M:
                continue
            out += _geometry_plans(mm, kw_options, rows, 1, conv_n // rows,
                                   rows * mm.k, 0, rows * mm.m)
        out.sort(key=lambda p: (p.cycles, p.kw, -p.out_w, -p.conv_n))
        return out
    if not plans or not all(p.acc_limited for p in plans) or not contiguous:
        return plans
    split: List[ConvPlan] = []
    for rows in _divisors(conv_n):
        if rows == conv_n or rows % CONV_TILE_M:
            continue
        split += [p for p in _geometry_plans(mm, kw_options, rows, 1, conv_n // rows,
                                             rows * mm.k, 0, rows * mm.m)
                  if not p.acc_limited]
    split.sort(key=lambda p: (p.cycles, p.kw, -p.out_w))
    return split + plans


def matmul_plan_cycles(mm: MatmulNode, kw_pin: Optional[int] = None) -> float:
    """MatmulKernel's cycles for ``mm`` (plus the call).  On the RTL kernel a
    weight pinned by another entry (``kw_pin``: an image width, or 0 for the
    tiled path's layout) is read in that layout, so it is priced in it."""
    batch = mm.batch * mm.outer_count
    if kw_pin and cost_model.MATMUL_IMPL == "rtl":
        from .matmul_gemv import gemv_shape_reason
        if gemv_shape_reason(mm, kw_pin) is None:
            return gemv_cycles(mm.n, mm.k, mm.m, batch, kw_pin) + CALL_OVERHEAD
    return matmul_cycles(mm.n, mm.k, mm.m, batch,
                         b_packed=bool(mm.b_packed) or kw_pin == 0) + CALL_OVERHEAD


# Image widths MatmulKernel reads (its GEMV / image path, matmul_gemv.GEMV_KWS).
_MM_IMAGE_KWS = (1, 2, 4, 8)


def _image_ok(sn, kw: int, is_ap_fixed_16_8: bool) -> bool:
    from .matmul_gemv import ineligible_reason as gemv_ineligible
    return gemv_ineligible(sn, kw, is_ap_fixed_16_8) is None


def _unplanned_mm_layout(sn, users: list, io, gemv_mode: str, gemv_hint: Dict[str, int],
                         is_ap_fixed_16_8: bool) -> int:
    """The MatmulKernel layout the GEMV pass gives ``sn`` without planning
    (0 = tiled, kw = the image at kw)."""
    from .matmul_gemv import image_choice
    if gemv_mode == "off":
        return 0
    ch = image_choice(sn, users, io, gemv_hint, gemv_mode, is_ap_fixed_16_8)
    return ch[0] if ch is not None and ch[2] else 0


def shared_weight_layouts(nodes: list) -> Dict[str, int]:
    """{constant B name: layout} of a graph's MatMul weights, for a
    multi-entry project's other graphs to pin (``matmul_conv_kw``), so a
    shared weight stays one buffer: the kernel width of every weight read by
    ConvKernel (MatmulConvNode), and on the RTL kernel also of every weight
    MatmulKernel reads — its image width (``gemv_kw``), or 0 for the tiled
    path's layout (packed or row-major)."""
    out: Dict[str, int] = {}
    for sn in nodes:
        if isinstance(sn, MatmulConvNode) and sn.inputs[1].is_weight:
            out[sn.inputs[1].onnx_name] = sn.kw
        elif (cost_model.MATMUL_IMPL == "rtl" and type(sn) is MatmulNode
              and sn.inputs[1].is_weight):
            out.setdefault(sn.inputs[1].onnx_name, sn.gemv_kw)
    return out


def _readers(nodes) -> Dict[str, list]:
    readers: Dict[str, list] = {}
    for sn in nodes:
        for t in getattr(sn, "inputs", []):
            readers.setdefault(t.onnx_name, []).append(sn)
    return readers


# Planning (--plan): a tactic replaces the baseline only when the model
# prices it this much below (TACTICS_PLAN §4.1: no churn on noise).
PLAN_MIN_GAIN = 0.03


def plan_conv_calls(mm, p: ConvPlan) -> list:
    """The ConvKernel calls of plan ``p`` for MatMul ``mm`` (perf_calls.py)."""
    from .perf_calls import KernelCall
    return [KernelCall.of("ConvKernel", count=p.calls, batch=p.conv_batch, in_ch=mm.k // p.kw,
                          in_h=p.out_h, in_w=p.kw * p.out_w, out_ch=p.conv_n, out_h=p.out_h,
                          out_w=p.out_w, kh=1, kw=p.kw, stride_h=1, stride_w=p.kw,
                          dilation_h=1, dilation_w=1)]


def plan_tiled_calls(mm, packed: bool) -> list:
    """MatmulKernel's tiled call for ``mm`` (B packed when ``packed``)."""
    from .nodes import matmul_packed_m
    from .perf_calls import KernelCall
    b_stride = (mm.b_batch_stride // (mm.k * mm.m) * matmul_packed_m(mm.m) * mm.k
                if packed and mm.b_batch_stride else mm.b_batch_stride)
    return [KernelCall.of("MatmulKernel", n=mm.n, k=mm.k, m=mm.m, batch=mm.batch,
                          a_stride=mm.a_batch_stride, b_stride=b_stride,
                          c_stride=mm.c_batch_stride, b_packed=int(packed), gemv_kw=0)]


def plan_mm_calls(mm, layout: int, b_constant: bool) -> list:
    """MatmulKernel's call for ``mm`` in ``layout``: 0 = the tiled path (B
    packed when it is a constant), kw = the GEMV / image path at kw."""
    if layout == 0:
        return plan_tiled_calls(mm, b_constant)
    from .perf_calls import KernelCall
    return [KernelCall.of("MatmulKernel", n=mm.n, k=mm.k, m=mm.m, batch=mm.batch,
                          a_stride=mm.a_batch_stride, b_stride=mm.b_batch_stride,
                          c_stride=mm.c_batch_stride, b_packed=0, gemv_kw=layout)]


# A planned choice: a ConvPlan (ConvKernel), or MatmulKernel's layout as an
# int — 0 the tiled path, kw the GEMV / image path at kw.
Choice = Union[ConvPlan, int]


def _plan_label(p: Choice) -> str:
    if isinstance(p, int):
        return "MatmulKernel tiled" if p == 0 else f"MatmulKernel image kw={p}"
    return (f"ConvKernel kw={p.kw} out_w={p.out_w}"
            + (f" {p.calls}x{p.conv_n} rows" if p.calls > 1 else ""))


def _tie(c: Choice) -> tuple:
    """Tie-break among equally priced choices: MatmulKernel first, then the
    narrower kernel width, then the wider output tile."""
    return (0, c, 0) if isinstance(c, int) else (1, c.kw, -c.out_w)


def _plan_matmul(sn, baseline: Choice, kws, mm_layouts: Sequence[int], perf_model,
                 plan_log: Optional[list]) -> Choice:
    """The planned choice for ``sn`` among its ConvKernel plans (kernel
    widths ``kws``), MatmulKernel in each layout of ``mm_layouts`` and the
    ``baseline`` (always a candidate)."""
    from .perf_model import clearly_faster
    const = sn.inputs[1].data is not None

    def band(c: Choice):
        return perf_model.calls_band(plan_mm_calls(sn, c, const) if isinstance(c, int)
                                     else plan_conv_calls(sn, c))
    choices: List[Choice] = list(conv_plans(sn, kws, splits="all")) + list(mm_layouts)
    if baseline not in choices:
        choices.append(baseline)
    cands = [(band(c), c) for c in choices]
    base = band(baseline)
    priced = [(b, c) for b, c in cands if b is not None]
    chosen, why = baseline, "baseline"
    best = min(priced, key=lambda bc: (bc[0][0],) + _tie(bc[1])) if priced else None
    if base is None:
        why = "baseline not priced"
    elif best is not None and best[1] != baseline:
        # the fastest candidate that may replace the baseline (perf_model.clearly_faster)
        ok = [bc for bc in priced if bc[1] != baseline and clearly_faster(bc[0], base, PLAN_MIN_GAIN)]
        if ok:
            best = min(ok, key=lambda bc: (bc[0][0],) + _tie(bc[1]))
            chosen, why = best[1], "planned"
        elif best[0][0] < base[0]:
            from .perf_model import MAX_MODEL_ERROR
            why = ("gain below the minimum" if best[0][0] >= base[0] * (1.0 - PLAN_MIN_GAIN)
                   else "predicted by a model outside its trust" if best[0][1] > MAX_MODEL_ERROR
                   else "gain within the error band")
    if plan_log is not None:
        plan_log.append({"node": sn.index, "name": sn.onnx_node.name or f"MatMul_{sn.index}",
                         "baseline": _plan_label(baseline), "chosen": _plan_label(chosen),
                         "baseline_us": base[0] if base else None,
                         "best_us": best[0][0] if best else None,
                         "chosen_us": (best[0][0] if best is not None and chosen == best[1]
                                       else base[0] if base else None),
                         "decision": why, "candidates": len(cands), "priced": len(priced)})
    return chosen


def lower_matmuls(nodes: list, *, mode: str = "auto", is_ap_fixed_16_8: bool = True,
                  graph_io: Sequence[str] = (),
                  kw_override: Optional[Dict[str, int]] = None,
                  kw_choices: Optional[Sequence[int]] = None,
                  gemv_mode: str = "auto", gemv_hint: Optional[Dict[str, int]] = None,
                  perf_model=None, plan_log: Optional[list] = None) -> Tuple[list, dict]:
    """Return ``(new_nodes, stats)``: ``nodes`` with every MatmulNode that
    should run on ConvKernel replaced by a MatmulConvNode (same index,
    inputs and output), and the constant Bs that use a ``kw > 1`` layout
    re-imaged (``TensorInfo.packed_data``).  ``stats`` =
    ``{"lowered", "kept", "conv_calls", "conv_cycles", "matmul_cycles"}``
    (cycles of the lowered nodes on either engine).

    ``kw_override`` ({constant B name: kw}) pins the kernel width of the
    listed weights — a multi-entry project's graphs must re-lay out a
    shared weight identically for it to stay one buffer.  ``kw_choices``
    limits the widths a re-laid-out B may take (the LLM projects pass the
    ones MatmulKernel's GEMV decode can read as well, matmul_gemv.py).

    ``perf_model`` (planning, src/perf_model.py): every MatMul's choice
    above is its baseline — on MatmulKernel in the layout the GEMV pass's
    unplanned rules give it (``gemv_mode``, ``gemv_hint``: matmul_gemv
    .image_choice) — and the planner prices the baseline and every other
    tactic: each conv geometry and row split with the allowed kernel widths,
    and unless ``mode == "always"`` MatmulKernel in each layout it may take
    (its tiled path; its image path at kw 1, and at kw 2 / 4 / 8 for a B it
    may re-image — on the HLS kernel only for one row; a weight pinned by
    ``kw_override`` only in its pinned layout, so it keeps one copy).  It
    takes the cheapest one when it is priced at least ``PLAN_MIN_GAIN``
    below the baseline; a baseline the model cannot price is kept.  A
    MatMul kept on MatmulKernel carries the layout in ``plan_kw`` for the
    GEMV pass.  Each decision is appended to ``plan_log``."""
    from .matmul_gemv import normalize_mode as gemv_normalize
    mode = normalize_mode(mode)
    gemv_mode = gemv_normalize(gemv_mode)
    kw_override = kw_override or {}
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
        kws = range(1, CONV_MAX_KW + 1) if relayout_ok else (1,)
        if kw_choices is not None:
            kws = [k for k in kws if k in kw_choices]
        if b.onnx_name in kw_override:
            kws = [k for k in kws if k == kw_override[b.onnx_name]]
        plans = conv_plans(sn, kws)
        pin = kw_override.get(b.onnx_name)
        mm_cyc = matmul_plan_cycles(sn, pin)
        keep = not plans or (mode == "auto" and plans[0].cycles >= LOWER_MARGIN * mm_cyc)
        if keep and plans and pin and cost_model.MATMUL_IMPL == "rtl" and pin not in _MM_IMAGE_KWS:
            keep = False      # pinned to a ConvKernel-only image: MatmulKernel would need a copy
        p = None if keep else plans[0]
        if perf_model is not None:
            users = readers.get(b.onnx_name, [])
            base_mm = _unplanned_mm_layout(sn, users, io, gemv_mode, gemv_hint or {},
                                           is_ap_fixed_16_8)
            mm_layouts: List[int] = []
            if b.onnx_name in kw_override:
                # a weight shared with other entries keeps one image: MatmulKernel
                # only when it reads exactly the pinned layout (the RTL kernel, any
                # n); the baseline stays a candidate whatever it is
                if (mode != "always" and cost_model.MATMUL_IMPL == "rtl"
                        and base_mm == kw_override[b.onnx_name]):
                    mm_layouts = [base_mm]
            elif mode != "always":
                mm_layouts = [0] + [kw for kw in _MM_IMAGE_KWS
                                    if gemv_mode != "off" and (kw == 1 or relayout_ok)
                                    and (kw_choices is None or kw == 1 or kw in kw_choices)
                                    and _image_ok(sn, kw, is_ap_fixed_16_8)]
            choice = _plan_matmul(sn, base_mm if p is None else p, kws, mm_layouts,
                                  perf_model, plan_log)
            keep = isinstance(choice, int)
            if keep:
                sn.plan_kw = choice
            else:
                p = choice
        if keep:
            stats["kept"] += 1
            out.append(sn)
            continue
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
    "shared_weight_layouts",
    "lower_matmuls",
    "PLAN_MIN_GAIN",
    "plan_conv_calls",
    "plan_tiled_calls",
)
