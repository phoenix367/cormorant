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
"""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional, Tuple

import onnx

from ._matmul_hw_config import MATMUL_GEMV_MAX_M
from .graph import OnnxGraph
from .matmul_gemv import GEMV_KWS
from .nodes import MatmulConvNode


def entry_graphs(models: Dict[str, onnx.ModelProto], *, prefill_engine: str = "conv",
                 matmul_on_conv: Optional[str] = None,
                 log: Optional[Callable[[str], None]] = None) -> List[Tuple[str, OnnxGraph]]:
    """``[(name, OnnxGraph)]`` in the order decode, prefill buckets
    (ascending), head, for ``models`` = {entry name: ModelProto} with names
    ``decode``, ``prefill_<T>``, ``head``.  ``prefill_engine`` is "conv"
    (MatMul-on-ConvKernel where the cost model prefers it) or "matmul";
    ``matmul_on_conv`` overrides the prefill lowering mode (tests)."""
    log = log or (lambda _msg: None)
    mode = matmul_on_conv or ("auto" if prefill_engine == "conv" else "off")
    kws = GEMV_KWS if MATMUL_GEMV_MAX_M > 0 else None
    graphs: Dict[str, OnnxGraph] = {}
    prefills = sorted((n for n in models if n.startswith("prefill_")),
                      key=lambda n: -int(n.split("_")[1]))
    kw = None
    for name in prefills:
        t0 = time.time()
        g = OnnxGraph(models[name], fuse_act=True, s2d_stem=True,
                      matmul_on_conv=mode, matmul_conv_kw=kw, matmul_conv_kws=kws)
        if kw is None:
            kw = {sn.inputs[1].onnx_name: sn.kw for sn in g.nodes
                  if isinstance(sn, MatmulConvNode) and sn.inputs[1].is_weight}
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
        g = OnnxGraph(models[name], fuse_act=True, s2d_stem=True, matmul_gemv_kw=kw)
        graphs[name] = g
        st = g.matmul_gemv_stats
        log(f"  {name}: {len(g.nodes)} nodes, MatMul GEMV {st['gemv']} "
            f"(conv image {st['kw>1']}) ({time.time() - t0:.0f} s)")
    order = (["decode"] + sorted(prefills, key=lambda n: int(n.split("_")[1])) + ["head"])
    return [(n, graphs[n]) for n in order if n in graphs]


__all__ = ("entry_graphs",)
