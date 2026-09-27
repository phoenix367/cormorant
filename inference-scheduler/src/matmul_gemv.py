"""
MatmulKernel GEMV streaming mode — the selection pass (doc/kernels/
MATMUL_OPTIMISATION.md §8b, MatmulKernel.h "GEMV streaming mode").

``choose_gemv(nodes, ...)`` switches single-row MatmulNodes (``n == 1``: the
batch-1 FC layers of the CNNs, the LLM decode linears and LM head) to the
kernel's GEMV path (``MatmulNode.gemv_kw``).  B is streamed once per A row,
half through each of the kernel's two read ports, in the ConvKernel x image
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
  * a plain MatmulNode (``outer_count == 1``) with ``n == 1`` — each A row
    streams all of B, so several rows are the tiled path's job;
  * ``k % 8 == 0`` and ``k <= max_k`` (``k % (16 kw) == 0`` for kw > 1),
    ``m % 8 == 0`` and ``m * kw >= 64``, A and B batch strides multiples of
    8 elements.

Mode: "auto" (default) — eligible nodes whose ``cost_model.gemv_cycles`` is
below the tiled path's ``matmul_cycles``; "always" — every eligible node;
"off" (CLI ``--matmul-gemv off``).
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

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
    if sn.outer_count != 1:
        return "4D x 3D outer loop"
    if sn.n != 1:
        return f"n = {sn.n} rows (GEMV streams B once per row)"
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


def choose_gemv(nodes: list, *, mode: str = "auto", is_ap_fixed_16_8: bool = True,
                graph_io: Sequence[str] = (),
                kw_hint: Optional[Dict[str, int]] = None) -> dict:
    """Set ``gemv_kw`` on the MatmulNodes of ``nodes`` that should run as
    GEMV, re-imaging a constant B for kw > 1 (``TensorInfo.packed_data``).
    Runs after the ConvKernel lowering and before the packed-B pass.
    Returns ``{"gemv", "kw>1", "tiled_cycles", "gemv_cycles"}``."""
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
        users = readers.get(b.onnx_name, [])
        relayout = False
        if b.packed_data is not None:
            # Re-imaged by this graph's ConvKernel lowering: read that image.
            kws = {u.kw for u in users if isinstance(u, MatmulConvNode) and u.inputs[1] is b}
            if len(kws) != 1:
                continue
            kw = kws.pop()
        else:
            kw = kw_hint.get(b.onnx_name, 1)
            relayout = kw > 1
            if relayout and not (b.data is not None and b.onnx_name not in io
                                 and len(users) == 1):
                kw, relayout = 1, False
        if ineligible_reason(sn, kw, is_ap_fixed_16_8) is not None:
            if not relayout or ineligible_reason(sn, 1, is_ap_fixed_16_8) is not None:
                continue
            kw, relayout = 1, False          # the hint's image does not fit: plain B
        tiled = matmul_cycles(sn.n, sn.k, sn.m, sn.batch)
        gemv = gemv_cycles(sn.n, sn.k, sn.m, sn.batch, kw)
        if mode == "auto" and gemv >= tiled:
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
    "choose_gemv",
)
