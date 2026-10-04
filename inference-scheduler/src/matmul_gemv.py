"""
MatmulKernel GEMV streaming mode — the selection pass (doc/kernels/
MATMUL_OPTIMISATION.md §8b, MatmulKernel.h "GEMV streaming mode").

``choose_gemv(nodes, ...)`` switches MatmulNodes to the kernel's GEMV / image
path (``MatmulNode.gemv_kw``): single-row ones (``n == 1``: the batch-1 FC
layers of the CNNs, the LLM decode linears and LM head) on the HLS kernel,
any on the RTL kernel (below).  B is streamed once per A row (on the RTL
kernel once per panel of 8 rows), half through each of the kernel's two
read ports, in the ConvKernel x image
of kernel width kw (``nodes.conv_lowered_b_image``; kw = 1 is plain
row-major B, so an activation B works too).

kw > 1 comes from ``kw_hint`` ({constant B name: kw}): a multi-entry project
passes the kernel widths its prefill graphs chose for their MatMul-on-
ConvKernel weights, so the decode graph reads the very same image and the
project keeps ONE copy of each weight (multi.py deduplicates by name and
image).  A B that this graph's ConvKernel lowering already re-imaged is read
in that image.  Everything else uses kw = 1.

Eligibility (every rule is a kernel requirement):

  * the platform's kernel has the path (``kernels.matmul.gemv_max_m > 0``)
    and the graph's element type is ap_fixed<16,8> (8 lanes per word);
  * a plain MatmulNode (``outer_count == 1``) with ``n == 1`` on the HLS
    kernel — it streams all of B per A row, so several rows are the tiled
    path's job.  The RTL kernel (``kernels.matmul.impl == "rtl"``) streams B
    once per panel of 8 rows on either path, so any ``n`` is eligible and the
    image is simply a B layout: a constant weight shared with other entries
    (``kw_hint``) is always read in its image there, keeping one copy;
  * ``k % 8 == 0`` and ``k <= max_k`` (``k % (16 kw) == 0`` for kw > 1),
    ``m % 8 == 0`` and ``m * kw >= 64``, A and B batch strides multiples of
    8 elements.

Mode: "auto" (default) — eligible nodes whose ``cost_model.gemv_cycles`` is
below the tiled path's ``matmul_cycles`` (on the RTL kernel also every hinted
weight); "always" — every eligible node; "off" (CLI ``--matmul-gemv off``).
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

from . import cost_model
from ._conv_hw_config import CONV_TILE_IC
from ._matmul_hw_config import MATMUL_GEMV_MAX_M, MATMUL_MAX_K
from .cost_model import gemv_cycles, matmul_cycles
from .nodes import MatmulConvNode, MatmulNode, SchedulerError, conv_lowered_b_image

MODES = ("auto", "always", "off")
GEMV_KWS = (1, 2, 4, 8)
LANES = 8              # elements per 128-bit word (ap_fixed<16,8>)
MIN_PLANE = 64         # m * kw: elements of one image plane (MatmulKernel.h kGemvMinPlane)


def normalize_mode(mode) -> str:
    if mode is True:
        return "auto"
    if mode is False or mode is None:
        return "off"
    if mode not in MODES:
        raise SchedulerError(f"matmul_gemv must be one of {MODES}, got {mode!r}")
    return mode


def ineligible_reason(sn, kw: int = 1, is_ap_fixed_16_8: bool = True) -> Optional[str]:
    """Why ``sn`` cannot run as GEMV with kernel width ``kw`` (None: it can)."""
    if MATMUL_GEMV_MAX_M <= 0:
        return "the platform's MatmulKernel has no GEMV path (gemv_max_m = 0)"
    if not is_ap_fixed_16_8:
        return "element type is not ap_fixed<16,8>"
    if type(sn) is not MatmulNode:
        return "not a MatmulKernel node"
    return gemv_shape_reason(sn, kw)


def gemv_shape_reason(sn, kw: int = 1) -> Optional[str]:
    """ineligible_reason()'s shape rules alone, for any object with the
    MatMul attributes (n, k, m, outer_count, b_packed, batch strides) — the
    planner's candidates of a MatMul already lowered elsewhere
    (src/tactics.py)."""
    if sn.outer_count != 1:
        return "4D x 3D outer loop"
    if sn.n != 1 and cost_model.MATMUL_IMPL != "rtl":
        return f"n = {sn.n} rows (the HLS kernel's GEMV streams B once per row)"
    if sn.b_packed:
        return "B is in the packed tile-major layout"
    if kw not in GEMV_KWS:
        return f"kernel width {kw} not in {GEMV_KWS}"
    if sn.k % LANES or sn.k > MATMUL_MAX_K:
        return f"k = {sn.k} (needs k % {LANES} == 0 and k <= {MATMUL_MAX_K})"
    if kw > 1 and (sn.k % (CONV_TILE_IC * kw) or CONV_TILE_IC != 16):
        return f"k = {sn.k} with kw = {kw} (needs k % (16 kw) == 0)"
    if sn.m % LANES or sn.m * kw < MIN_PLANE:
        return f"m = {sn.m} (needs m % {LANES} == 0 and m * kw >= {MIN_PLANE})"
    if sn.a_batch_stride % LANES or sn.b_batch_stride % LANES:
        return "batch strides not multiples of 8 elements"
    return None


def _plan_gemv(sn, kw: int, baseline: bool, perf_model, plan_log) -> bool:
    """Planned GEMV (True) or tiled (False) for a MatMul the ConvKernel
    lowering's planner did not price (one row; on the RTL kernel also the
    MatMuls too small for ConvKernel)."""
    from .matmul_lowering import PLAN_MIN_GAIN, plan_tiled_calls
    from .perf_calls import KernelCall
    from .perf_model import clearly_faster
    b_gemv = perf_model.calls_band([KernelCall.of(
        "MatmulKernel", n=sn.n, k=sn.k, m=sn.m, batch=sn.batch, a_stride=sn.a_batch_stride,
        b_stride=sn.b_batch_stride, c_stride=sn.c_batch_stride, b_packed=0, gemv_kw=kw)])
    b_tiled = perf_model.calls_band(plan_tiled_calls(sn, sn.inputs[1].data is not None))
    t_gemv = b_gemv[0] if b_gemv else None
    t_tiled = b_tiled[0] if b_tiled else None
    chosen, why = baseline, "baseline"
    if b_gemv is None or b_tiled is None:
        why = "not priced"
    else:
        base, other = (b_gemv, b_tiled) if baseline else (b_tiled, b_gemv)
        if clearly_faster(other, base, PLAN_MIN_GAIN):
            chosen, why = not baseline, "planned"
        elif other[0] < base[0]:
            why = ("gain below the minimum" if other[0] >= base[0] * (1.0 - PLAN_MIN_GAIN)
                   else "gain within the error band")
    if plan_log is not None:
        lab = lambda g: f"MatmulKernel GEMV kw={kw}" if g else "MatmulKernel tiled"  # noqa: E731
        plan_log.append({"node": sn.index, "name": sn.onnx_node.name or f"MatMul_{sn.index}",
                         "baseline": lab(baseline), "chosen": lab(chosen),
                         "baseline_us": t_gemv if baseline else t_tiled,
                         "best_us": min(t for t in (t_gemv, t_tiled) if t is not None)
                         if (t_gemv is not None or t_tiled is not None) else None,
                         "chosen_us": t_gemv if chosen else t_tiled,
                         "decision": why, "candidates": 2,
                         "priced": sum(t is not None for t in (t_gemv, t_tiled))})
    return chosen


def image_choice(sn, users: list, io, kw_hint: Dict[str, int], mode: str,
                 is_ap_fixed_16_8: bool = True) -> Optional[tuple]:
    """The unplanned rules for MatmulNode ``sn`` (``users``: every node
    reading its B): ``(kw, relayout, use, tiled_cycles, gemv_cycles)`` — the
    image it would read (``relayout``: a constant B re-imaged for it) and
    whether it takes the path (``use``); None when no image fits.  Shared by
    :func:`choose_gemv` and the planner's baseline (matmul_lowering)."""
    b = sn.inputs[1]
    relayout = False
    if b.packed_data is not None:
        # Re-imaged by this graph's ConvKernel lowering: read that image.
        kws = {u.kw for u in users if isinstance(u, MatmulConvNode) and u.inputs[1] is b}
        if len(kws) != 1:
            return None
        kw = kws.pop()
    else:
        kw = kw_hint.get(b.onnx_name, 1)
        relayout = kw > 1
        if relayout and not (b.data is not None and b.onnx_name not in io
                             and len(users) == 1):
            kw, relayout = 1, False
    if ineligible_reason(sn, kw, is_ap_fixed_16_8) is not None:
        if not relayout or ineligible_reason(sn, 1, is_ap_fixed_16_8) is not None:
            return None
        kw, relayout = 1, False          # the hint's image does not fit: plain B
    tiled = matmul_cycles(sn.n, sn.k, sn.m, sn.batch, b_packed=bool(sn.b_packed))
    gemv = gemv_cycles(sn.n, sn.k, sn.m, sn.batch, kw)
    # the RTL kernel reads a shared weight in its image whatever n is:
    # the tiled path would need the packed layout as a second copy
    shared = cost_model.MATMUL_IMPL == "rtl" and b.onnx_name in kw_hint
    return kw, relayout, mode != "auto" or gemv < tiled or shared, tiled, gemv


def choose_gemv(nodes: list, *, mode: str = "auto", is_ap_fixed_16_8: bool = True,
                graph_io: Sequence[str] = (),
                kw_hint: Optional[Dict[str, int]] = None,
                perf_model=None, plan_log: Optional[list] = None) -> dict:
    """Set ``gemv_kw`` on the MatmulNodes of ``nodes`` that should run as
    GEMV, re-imaging a constant B for kw > 1 (``TensorInfo.packed_data``).
    Runs after the ConvKernel lowering and before the packed-B pass.
    Returns ``{"gemv", "kw>1", "tiled_cycles", "gemv_cycles"}``.

    ``perf_model`` (planning): in "auto" mode the choice between the tiled
    and the GEMV path is priced with the performance model instead of the
    cost model (the other rules unchanged: the image a B is read in); a
    choice changes only when priced PLAN_MIN_GAIN below the cost model's,
    and when either path cannot be priced the cost model decides.  A node
    the ConvKernel lowering's planner already placed on MatmulKernel
    (``MatmulNode.plan_kw``: 0 = tiled, kw = the image at kw) takes that
    layout as is."""
    mode = normalize_mode(mode)
    kw_hint = kw_hint or {}
    stats = {"gemv": 0, "kw>1": 0, "tiled_cycles": 0.0, "gemv_cycles": 0.0}
    if mode == "off":
        return stats
    readers: Dict[str, list] = {}
    for sn in nodes:
        for t in getattr(sn, "inputs", []):
            readers.setdefault(t.onnx_name, []).append(sn)
    io = set(graph_io)
    for sn in nodes:
        if type(sn) is not MatmulNode:
            continue
        b = sn.inputs[1]
        if sn.plan_kw is not None:
            if sn.plan_kw == 0 or ineligible_reason(sn, sn.plan_kw, is_ap_fixed_16_8) is not None:
                continue                     # planned: the tiled path
            kw = sn.plan_kw
            relayout = kw > 1 and b.packed_data is None
            tiled = matmul_cycles(sn.n, sn.k, sn.m, sn.batch, b_packed=bool(sn.b_packed))
            gemv = gemv_cycles(sn.n, sn.k, sn.m, sn.batch, kw)
        else:
            ch = image_choice(sn, readers.get(b.onnx_name, []), io, kw_hint, mode,
                              is_ap_fixed_16_8)
            if ch is None:
                continue
            kw, relayout, use, tiled, gemv = ch
            if perf_model is not None and mode == "auto" and b.onnx_name not in kw_hint:
                # (a weight in a shared image keeps the unplanned choice: switching
                # it would need a second copy in another layout)
                use = _plan_gemv(sn, kw, use, perf_model, plan_log)
            if not use:
                continue
        if relayout:
            b.packed_data = conv_lowered_b_image(b.data, sn.k, sn.m, kw)
            b.packed_note = (f"MatmulKernel GEMV / ConvKernel x image, kw={kw}:"
                             f" x[c][{kw}*p + j] = B[(c/16)*{16 * kw} + j*16 + c%16][p]"
                             f" = [{sn.k // kw}][{sn.m * kw}]"
                             + (f" x {b.data.size // (sn.k * sn.m)} slices"
                                if b.data.size != sn.k * sn.m else ""))
        sn.gemv_kw = kw
        stats["gemv"] += 1
        stats["kw>1"] += kw > 1
        stats["tiled_cycles"] += tiled
        stats["gemv_cycles"] += gemv
    return stats


__all__ = (
    "MODES",
    "GEMV_KWS",
    "normalize_mode",
    "ineligible_reason",
    "image_choice",
    "choose_gemv",
)
