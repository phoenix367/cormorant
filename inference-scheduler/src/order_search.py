"""
Order search — the issue order from a timed simulation (doc/plans/
TACTICS_PLAN.md §4.4).

The generated ``inference_run()`` issues the nodes in list order on one
CPU: a kernel start does not block, a host op does, and a kernel lane
takes one call at a time.  :func:`search_order` looks for a list order with
a smaller simulated total (src/codegen/timing.py): local search from the
current order, moving one node earlier past neighbours it does not depend
on (the DAG of the ORIGINAL order, whose state edges pin every state
access, src/schedule.py), keeping a move when the simulated total drops.

:func:`plan_order` applies it to an OnnxGraph under ``--plan`` when every
node can be priced (the kernels by the performance model, the host ops by
the host model), and keeps the new order only if the DMA pool stays within
the budget (default: the intermediates' pool of the original order, with
slot reuse).
"""

from __future__ import annotations

import time
from typing import Callable, List, Optional, Tuple

MIN_GAIN = 0.005          # keep an order only if it is this much faster in total


def search_order(cg, kernel_us: Callable, host_us: Callable, *, window: int = 10,
                 passes: int = 3, keys_of: Optional[Callable] = None,
                 time_budget_s: float = 60.0) -> Tuple[List[int], float, float, int]:
    """``(order, total_before_us, total_after_us, moves)``.  Only kernel
    starts are moved (earlier): issuing a kernel sooner is what overlaps it
    with host work; moving a host op later is the same move."""
    from .codegen.timing import simulate
    from .schedule import Dag
    graph = cg._graph
    dag = Dag.from_graph(graph)
    order = [sn.index for sn in graph.nodes]
    preds = {i: dag.predecessors(i) for i in order}
    kinds = {sn.index: bool(cg._kernel_id_of(sn)) for sn in graph.nodes}

    def kernel_node(i):
        return kinds[i]

    def total(o):
        return simulate(cg, kernel_us, host_us, events=cg._compute_event_stream(o, dag)).total_us

    before = best = total(order)
    moves = 0
    t_end = time.monotonic() + time_budget_s
    for _ in range(passes):
        improved = False
        i = 1
        while i < len(order):
            if time.monotonic() > t_end:
                break
            v = order[i]
            if not kernel_node(v):
                i += 1
                continue
            j = i
            moved = False
            while j > 0 and i - j < window and order[j - 1] not in preds[v]:
                j -= 1
                cand = order[:j] + [v] + order[j:i] + order[i + 1:]
                t = total(cand)
                if t < best - 1e-6:
                    best, order, moved = t, cand, True
                    moves += 1
                    break
            improved |= moved
            i += 1
        if not improved or time.monotonic() > t_end:
            break
    return order, before, best, moves


def apply_order(graph, order: List[int]) -> None:
    """Put ``graph``'s nodes in ``order`` (node indices) and renumber them
    0 .. n-1 in that order: the profiler's layer-name table is written in
    list order and its brackets use ``sn.index``, and the tools that read
    profiles map layer i to the i-th node."""
    by_index = {sn.index: sn for sn in graph._nodes}
    graph._nodes = [by_index[i] for i in order]
    for i, sn in enumerate(graph._nodes):
        sn.index = i


def plan_order(graph, log: Optional[Callable[[str], None]] = None) -> Optional[dict]:
    """Search and apply a faster order to ``graph`` (planning on).  Returns
    ``{"before_us", "after_us", "moves", "applied", "why"}`` or None when
    the graph cannot be priced."""
    from .codegen import CodeGenerator
    from .codegen.timing import kernel_duration_fn
    from .host_model import HostModel, default_host_model_path, is_host_op
    from .llm_nodes import attn_keys
    pm = getattr(graph, "perf_model", None)
    hp = default_host_model_path()
    if pm is None or not hp.exists():
        return None
    hm = HostModel.load(hp)
    cg = CodeGenerator(graph, model_path="plan.onnx")

    def keys_of(sn):
        return attn_keys(1, sn.T, sn.T, sn.C, sn.Q)[1]
    kus = kernel_duration_fn(pm, cg._layouts, keys_of=keys_of)
    missing = [sn for sn in graph.nodes
               if (is_host_op(sn) and hm.us(sn) is None)
               or (getattr(type(sn), "kernel_name", "") and hasattr(sn, "kernel_calls")
                   and kus(sn) is None)]
    if missing:
        info = {"applied": False, "why": f"{len(missing)} nodes not priced "
                f"(e.g. {type(missing[0]).__name__} {missing[0].onnx_node.name})"}
        graph.order_log = info
        return info
    order, before, after, moves = search_order(cg, kus, hm.us)
    info = {"before_us": before, "after_us": after, "moves": moves, "applied": False, "why": ""}
    if after > before * (1.0 - MIN_GAIN):
        info["why"] = "no order faster by 0.5 %"
        graph.order_log = info
        return info
    budget = getattr(graph.plan, "pool_budget_mib", None)
    elem = cg._dtype.bytes_per_elem
    # the intermediates' pool with slot reuse (the order changes the liveness)
    pool0 = cg._compute_intermediate_layout()[1] * elem
    old = [(sn, sn.index) for sn in graph._nodes]
    apply_order(graph, order)
    pool1 = CodeGenerator(graph, model_path="plan.onnx")._compute_intermediate_layout()[1] * elem
    limit = budget * 2 ** 20 if budget is not None else pool0
    if pool1 > limit:
        graph._nodes = [sn for sn, _ in old]
        for sn, idx in old:
            sn.index = idx
        info["why"] = f"pool {pool1 / 2**20:.1f} MiB over the budget {limit / 2**20:.1f} MiB"
    else:
        info.update(applied=True, why="planned", pool_before=pool0, pool_after=pool1)
    graph.order_log = info
    if log:
        log(f"  order: {before / 1e3:.1f} -> {after / 1e3:.1f} ms simulated, {moves} moves, "
            f"{info['why']}")
    return info


__all__ = ("search_order", "apply_order", "plan_order", "MIN_GAIN")
