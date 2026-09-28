"""
Tactics — the implementations a node can take, each with the kernel calls
it would issue (doc/plans/TACTICS_PLAN.md §4.1).

A tactic never changes the results: the two kernels compute a MatMul bit
for bit alike, so the planner (``--plan``) chooses among them on speed
only.  The calibration campaign (``perf_calibrate.py``) measures the calls
of the tactics of the shipped models; the planner prices each tactic with
the performance model and picks the fastest.

MatMul tactics (:func:`matmul_tactics`):

  conv   ConvKernel with swapped operand roles (matmul_lowering.py): every
         (kw, out_w) geometry, and for a contiguous-rows MatMul every row
         split; kw > 1 needs B re-imaged, allowed only for a constant B
         read by this MatMul alone and not a graph input / output;
  tiled  MatmulKernel's tiled path, B packed when it is a constant;
  gemv   MatmulKernel's GEMV path (one A row), B read in kw's image.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from ._conv_hw_config import CONV_MAX_KW
from ._matmul_hw_config import MATMUL_GEMV_MAX_M
from .cost_model import CALL_OVERHEAD, gemv_cycles, matmul_cycles
from .matmul_gemv import GEMV_KWS
from .matmul_gemv import gemv_shape_reason
from .matmul_lowering import conv_plans
from .matmul_lowering import ineligible_reason as conv_ineligible
from .nodes import MatmulConvNode, MatmulNode, matmul_packed_m
from .perf_calls import KernelCall


@dataclass(frozen=True)
class Tactic:
    kind:       str                         # "conv" | "tiled" | "gemv"
    params:     Tuple[Tuple[str, int], ...]
    calls:      Tuple[KernelCall, ...]
    est_cycles: float                       # the analytic cost model, all calls
    b_image:    Tuple[str, int]             # the B layout it reads: ("row", 1) /
                                            # ("conv", kw) / ("packed", 1)

    @property
    def p(self) -> dict:
        return dict(self.params)

    def label(self) -> str:
        return self.kind + "(" + ",".join(f"{k}={v}" for k, v in self.params) + ")"


@dataclass
class _MM:
    """The MatMul semantics of a MatmulNode / MatmulConvNode, for the
    lowering's plan search (which reads these attributes)."""
    n: int
    k: int
    m: int
    batch: int
    a_batch_stride: int
    b_batch_stride: int
    c_batch_stride: int
    inputs: list
    outer_count: int = 1
    b_packed: bool = False


def matmul_of(sn) -> Optional[_MM]:
    if isinstance(sn, _MM):
        return sn
    if isinstance(sn, (MatmulNode, MatmulConvNode)):
        return _MM(sn.n, sn.k, sn.m, sn.batch, sn.a_batch_stride, sn.b_batch_stride,
                   sn.c_batch_stride, list(sn.inputs), getattr(sn, "outer_count", 1))
    return None


def _conv_calls(mm: _MM, p) -> Tuple[KernelCall, ...]:
    return (KernelCall.of("ConvKernel", count=p.calls, batch=p.conv_batch,
                          in_ch=mm.k // p.kw, in_h=p.out_h, in_w=p.kw * p.out_w,
                          out_ch=p.conv_n, out_h=p.out_h, out_w=p.out_w, kh=1, kw=p.kw,
                          stride_h=1, stride_w=p.kw, dilation_h=1, dilation_w=1),)


def matmul_tactics(sn, *, relayout_ok: bool, b_constant: bool,
                   kw_choices: Optional[Sequence[int]] = None,
                   top_conv: Optional[int] = None,
                   layouts: Optional[dict] = None) -> List[Tactic]:
    """Every tactic of the MatMul node ``sn``, cheapest (by the analytic
    model) first within each kind.  ``relayout_ok``: B may be re-imaged
    (kw > 1); ``b_constant``: B is a weight (the tiled path packs it);
    ``kw_choices`` limits the kernel widths of a re-imaged B;
    ``top_conv`` keeps only the cheapest conv plans; ``layouts`` (the code
    generator's tensor layouts) gives a tiled MatmulNode its own calls — an
    A / C with alignment gaps runs row by row (MatmulNode.emit_call)."""
    mm = matmul_of(sn)
    if mm is None:
        return []
    out: List[Tactic] = []
    # conv
    if conv_ineligible(mm) is None:
        kws = list(range(1, CONV_MAX_KW + 1)) if relayout_ok else [1]
        if kw_choices is not None:
            kws = [k for k in kws if k in kw_choices]
        plans = conv_plans(mm, kws, splits="all")
        for p in plans[:top_conv] if top_conv else plans:
            out.append(Tactic("conv", (("kw", p.kw), ("out_w", p.out_w), ("rows", p.conv_n),
                                       ("calls", p.calls)),
                              _conv_calls(mm, p), float(p.cycles), ("conv", p.kw)))
    # tiled (an outer-loop MatmulNode has this tactic only: its own calls)
    own = isinstance(sn, MatmulNode) and not sn.gemv_kw and layouts is not None
    if mm.outer_count == 1 or own:
        packed = b_constant
        # a packed B's batch stride counts whole packed slices (MatmulNode.pack_b)
        b_stride = (mm.b_batch_stride // (mm.k * mm.m) * matmul_packed_m(mm.m) * mm.k
                    if packed and mm.b_batch_stride else mm.b_batch_stride)
        calls = (KernelCall.of("MatmulKernel", n=mm.n, k=mm.k, m=mm.m, batch=mm.batch,
                               a_stride=mm.a_batch_stride, b_stride=b_stride,
                               c_stride=mm.c_batch_stride, b_packed=int(packed), gemv_kw=0),)
        if own:
            calls = tuple(sn.kernel_calls(layouts))
            packed = bool(sn.b_packed)
        out.append(Tactic("tiled", (("packed", int(packed)),), calls,
                          float(matmul_cycles(mm.n, mm.k, mm.m, mm.batch) + CALL_OVERHEAD),
                          ("packed", 1) if packed else ("row", 1)))
    # gemv
    if MATMUL_GEMV_MAX_M > 0 and mm.outer_count == 1 and mm.n == 1:
        for kw in GEMV_KWS:
            if kw > 1 and not relayout_ok:
                continue
            if kw_choices is not None and kw > 1 and kw not in kw_choices:
                continue
            if gemv_shape_reason(mm, kw) is not None:
                continue
            call = KernelCall.of("MatmulKernel", n=1, k=mm.k, m=mm.m, batch=mm.batch,
                                 a_stride=mm.a_batch_stride, b_stride=mm.b_batch_stride,
                                 c_stride=mm.c_batch_stride, b_packed=0, gemv_kw=kw)
            out.append(Tactic("gemv", (("kw", kw),), (call,),
                              float(gemv_cycles(1, mm.k, mm.m, mm.batch, kw) + CALL_OVERHEAD),
                              ("conv", kw) if kw > 1 else ("row", 1)))
    return out


def graph_matmul_tactics(g, top_conv: Optional[int] = None,
                         layouts: Optional[dict] = None) -> List[Tuple[object, List[Tactic]]]:
    """``[(node, tactics)]`` for every MatMul of the graph ``g``, with the
    re-image and packing rules the graph applies (a constant B read by this
    node alone, not a graph input / output); ``layouts`` as in
    :func:`matmul_tactics`."""
    readers: dict = {}
    for sn in g.nodes:
        for t in getattr(sn, "inputs", []):
            readers.setdefault(t.onnx_name, []).append(sn)
    io = {t.onnx_name for t in g.input_tensors} | {t.onnx_name for t in g.output_tensors}
    out = []
    for sn in g.nodes:
        if not isinstance(sn, (MatmulNode, MatmulConvNode)):
            continue
        b = sn.inputs[1]
        const = b.data is not None
        alone = len(readers.get(b.onnx_name, [])) == 1
        out.append((sn, matmul_tactics(sn, relayout_ok=const and alone and b.onnx_name not in io,
                                       b_constant=const, top_conv=top_conv,
                                       layouts=layouts)))
    return out


__all__ = ("Tactic", "matmul_of", "matmul_tactics", "graph_matmul_tactics")
