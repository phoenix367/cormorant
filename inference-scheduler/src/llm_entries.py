"""
The graphs of a Llama multi-entry project (src/llama.py entries: decode,
prefill_<T>, head) scheduled so that the project keeps ONE copy of every
weight (doc/plans/CHAT_PLAN.md §13.1, MATMUL_OPTIMISATION.md §8b).

The prefill linears run on ConvKernel, which reads a weight in its x image
of some kernel width kw (``nodes.conv_lowered_b_image``); the decode and head
linears (one row) run on MatmulKernel.  With the kernel's GEMV streaming
path (``kernels.matmul.gemv_max_m > 0``) MatmulKernel reads that very image:

  1. the largest prefill bucket plans the kernel widths, limited to the
     widths the GEMV path reads (1 / 2 / 4 / 8); the smaller buckets reuse
     them (``matmul_conv_kw``) so all buckets share one image;
  2. decode and head read every weight in that image (``matmul_gemv_kw``).

multi.py then deduplicates each weight by name and image.  Without the GEMV
path (or where a decode MatMul cannot use it) decode keeps MatmulKernel's
packed tile-major copy and multi.py renames the prefill copy ``<name>@1``.

With planning (``plan``, doc/plans/TACTICS_PLAN.md §4.1) step 1 becomes a
joint choice: each shared weight takes the kernel width whose predicted
time, summed over the prefill buckets and decode / head weighted by
``planning.entry_weight``, is least (:func:`plan_shared_kw`); every bucket
then plans its own geometry at that width.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import gc
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import onnx

from ._matmul_hw_config import MATMUL_GEMV_MAX_M
from .graph import OnnxGraph
from .matmul_gemv import GEMV_KWS
from .nodes import MatmulConvNode


def _release_memory() -> None:
    """Collect garbage and hand freed heap back to the OS (glibc)."""
    gc.collect()
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c"))
        libc.malloc_trim(0)
    except (OSError, AttributeError):
        pass


def _share_arrays(g: OnnxGraph, pool: Dict[Tuple[str, str], np.ndarray]) -> None:
    """Point g's constant tensors at the arrays an earlier graph holds for
    the same name when their contents are equal (the entries of one model
    share every weight): one copy per weight in memory instead of one per
    entry.  Arrays are only read after the graphs are built."""
    for t in g._tensors.values():                          # noqa: SLF001
        for attr in ("data", "packed_data"):
            a = getattr(t, attr, None)
            if not isinstance(a, np.ndarray) or a.nbytes < 4096:
                continue
            key = (t.onnx_name, attr)
            b = pool.get(key)
            if b is None:
                pool[key] = a
            elif b is not a and b.dtype == a.dtype and b.shape == a.shape and np.array_equal(a, b):
                setattr(t, attr, b)


def _matmul_weights(model: onnx.ModelProto) -> Dict[str, Tuple[int, int]]:
    """{constant B name: (k, m)} of the model's 2-D-weight MatMuls."""
    inits = {t.name: tuple(t.dims) for t in model.graph.initializer}
    out = {}
    for node in model.graph.node:
        if node.op_type == "MatMul" and len(node.input) > 1 and node.input[1] in inits:
            dims = inits[node.input[1]]
            if len(dims) == 2:
                out[node.input[1]] = (int(dims[0]), int(dims[1]))
    return out


def plan_shared_kw(models: Dict[str, onnx.ModelProto], prefills: List[str], kws, perf_model,
                   weight: Callable[[str], float],
                   log: Optional[Callable[[str], None]] = None) -> Dict[str, int]:
    """{weight name: kernel width} for the constant MatMul weights of the
    prefill entries.  Per width: the best priced prefill tactic of every
    bucket (a ConvKernel geometry reading that image, or at kw 1 also
    MatmulKernel's tiled path) and GEMV in decode / head at that width, each
    times its entry weight.  The least total replaces the unplanned width
    (the cost model's choice for the largest bucket) only when it is faster
    beyond the model's error band (perf_model.clearly_faster); a weight
    whose unplanned width cannot be priced is left out (the unplanned rule
    then decides it)."""
    from .matmul_gemv import gemv_shape_reason
    from .matmul_lowering import PLAN_MIN_GAIN, conv_plans, plan_conv_calls, plan_tiled_calls
    from .perf_calls import KernelCall
    from .perf_model import clearly_faster
    from .tactics import _MM

    class _T:
        is_int, data, onnx_name = False, None, ""

    one_row = [n for n in ("decode", "head") if n in models]
    one_row_w = {n: _matmul_weights(models[n]) for n in one_row}
    shapes: Dict[str, Tuple[int, int]] = {}
    for name in prefills:
        shapes.update(_matmul_weights(models[name]))
    largest = max(int(n.split("_")[1]) for n in prefills)

    def cost(b, k, m, kw):
        tot = err = 0.0
        for name in prefills:
            mm = _MM(int(name.split("_")[1]), k, m, 1, 0, 0, 0, [_T(), _T()])
            bands = [perf_model.calls_band(plan_conv_calls(mm, p))
                     for p in conv_plans(mm, [kw], splits="all")]
            if kw == 1:
                bands.append(perf_model.calls_band(plan_tiled_calls(mm, False)))
            bands = [x for x in bands if x is not None]
            if not bands:
                return None
            t, e = min(bands)
            tot += weight(name) * t
            err += weight(name) * t * e
        for name in one_row:
            if b not in one_row_w[name]:
                continue
            if gemv_shape_reason(_MM(1, k, m, 1, 0, 0, 0, []), kw) is not None:
                return None
            x = perf_model.calls_band([KernelCall.of("MatmulKernel", n=1, k=k, m=m, batch=1,
                                                     gemv_kw=kw)])
            if x is None:
                return None
            tot += weight(name) * x[0]
            err += weight(name) * x[0] * x[1]
        return tot, (err / tot if tot > 0 else 0.0)

    out: Dict[str, int] = {}
    changed = 0
    for b, (k, m) in sorted(shapes.items()):
        plans = conv_plans(_MM(largest, k, m, 1, 0, 0, 0, [_T(), _T()]), list(kws))
        if not plans:
            continue
        base_kw = plans[0].kw
        base = cost(b, k, m, base_kw)
        if base is None:
            continue
        best_kw, best = base_kw, base
        for kw in kws:
            c = cost(b, k, m, kw) if kw != base_kw else None
            if c is not None and c[0] < best[0]:
                best_kw, best = kw, c
        if best_kw != base_kw and clearly_faster(best, base, PLAN_MIN_GAIN):
            out[b] = best_kw
            changed += 1
        else:
            out[b] = base_kw
    if log:
        from collections import Counter
        log(f"  planned kernel widths of {len(out)} / {len(shapes)} shared weights "
            f"({changed} changed): "
            + ", ".join(f"kw {k}: {n}" for k, n in sorted(Counter(out.values()).items())))
    return out


def entry_graphs(models: Dict[str, onnx.ModelProto], *, prefill_engine: str = "conv",
                 matmul_on_conv: Optional[str] = None,
                 log: Optional[Callable[[str], None]] = None,
                 plan=None) -> List[Tuple[str, OnnxGraph]]:
    """``[(name, OnnxGraph)]`` in the order decode, prefill buckets
    (ascending), head, then any other entry (a VLM's ``vision``, src/vit.py:
    its own weights, the default lowering), for ``models`` = {entry name:
    ModelProto} with names ``decode``, ``prefill_<T>``, ``head``, ....  ``prefill_engine`` is "conv"
    (MatMul-on-ConvKernel where the cost model prefers it) or "matmul";
    ``matmul_on_conv`` overrides the prefill lowering mode (tests).  ``plan``
    (src/planning.py) goes to every entry's graph.

    ``models`` is consumed: each entry's ModelProto is removed from it once
    its graph is built (with the graphs' weight arrays shared, a Llama
    project needs about one copy of the weights plus one model at a time —
    SmolLM2-360M's five entries otherwise exceed a 46 GB host)."""
    log = log or (lambda _msg: None)
    pool: Dict[Tuple[str, str], np.ndarray] = {}
    mode = matmul_on_conv or ("auto" if prefill_engine == "conv" else "off")
    kws = GEMV_KWS if MATMUL_GEMV_MAX_M > 0 else None
    graphs: Dict[str, OnnxGraph] = {}
    prefills = sorted((n for n in models if n.startswith("prefill_")),
                      key=lambda n: -int(n.split("_")[1]))
    kw = None
    if plan is not None and plan.enabled and prefills and mode != "off":
        from .planning import PlanError, entry_weight, resolve_perf_model
        from .nodes import SchedulerError
        try:
            pm = resolve_perf_model(plan)
        except PlanError as e:
            raise SchedulerError(str(e)) from None
        kw = plan_shared_kw(models, prefills, kws or GEMV_KWS, pm,
                            lambda n: entry_weight(plan, n), log) or None
    for name in prefills:
        t0 = time.time()
        g = OnnxGraph(models.pop(name), fuse_act=True, s2d_stem=True,
                      matmul_on_conv=mode, matmul_conv_kw=kw, matmul_conv_kws=kws, plan=plan)
        _share_arrays(g, pool)
        _release_memory()
        if kw is None or name == prefills[0]:
            # the first bucket's widths pin the others; a planned width wins
            first = {sn.inputs[1].onnx_name: sn.kw for sn in g.nodes
                     if isinstance(sn, MatmulConvNode) and sn.inputs[1].is_weight}
            kw = first if kw is None else {**first, **kw}
        graphs[name] = g
        st = g.matmul_conv_stats
        est = (f", est. {st['conv_cycles'] / 1e6:.1f} M cycles vs {st['matmul_cycles'] / 1e6:.1f} M "
               f"on MatmulKernel" if st["lowered"] else "")
        log(f"  {name}: {len(g.nodes)} nodes, MatMul on ConvKernel "
            f"{st['lowered']} / kept {st['kept']}{est} ({time.time() - t0:.0f} s)")
    for name in ("decode", "head"):
        if name not in models:
            continue
        t0 = time.time()
        g = OnnxGraph(models.pop(name), fuse_act=True, s2d_stem=True, matmul_gemv_kw=kw,
                      plan=plan)
        _share_arrays(g, pool)
        _release_memory()
        graphs[name] = g
        st = g.matmul_gemv_stats
        log(f"  {name}: {len(g.nodes)} nodes, MatMul GEMV {st['gemv']} "
            f"(conv image {st['kw>1']}) ({time.time() - t0:.0f} s)")
    others = sorted(models)
    for name in others:
        t0 = time.time()
        g = OnnxGraph(models.pop(name), fuse_act=True, s2d_stem=True, matmul_on_conv=mode,
                      plan=plan)
        _share_arrays(g, pool)
        _release_memory()
        graphs[name] = g
        st = g.matmul_conv_stats
        log(f"  {name}: {len(g.nodes)} nodes, MatMul on ConvKernel {st['lowered']} / kept "
            f"{st['kept']} ({time.time() - t0:.0f} s)")
    order = (["decode"] + sorted(prefills, key=lambda n: int(n.split("_")[1])) + ["head"]
             + others)
    return [(n, graphs[n]) for n in order if n in graphs]


__all__ = ("entry_graphs", "plan_shared_kw")
