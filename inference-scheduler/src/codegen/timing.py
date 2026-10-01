"""
Timed replay of the event stream (doc/plans/TACTICS_PLAN.md §4.4).

``simulate(cg, kernel_us, host_us)`` walks ``cg._compute_event_stream()``
the way the generated ``inference_run()`` executes it on the one CPU:

  start       the CPU writes the registers and starts the kernel
              (``issue_us``) and goes on; the kernel runs ``kernel_us(sn)``
              on its lane.  A MatmulConvNode with several calls waits for
              each call but the last inside its loop (the CPU is busy
              until the last one is issued);
  start_sync  the CPU waits for the whole call (the 4D x 3D MatMul loop);
  wait/drain  the CPU waits until that node's kernel is done;
  cpu         a host op runs ``host_us(sn)`` on the CPU.

It returns the total time, the busy time of the CPU and of every lane, and
the CPU's waits: which node's lane it waited for, how long — the "where the
CPU waits" of the planning report.  ``Timeline.spans`` is the whole replay,
one ``(lane, node, start µs, end µs, kind)`` per interval, for the timeline
view (src/timeline_html.py):

  kernel lanes   "kernel": the node's kernel running on its lane
  "CPU"          "host": a host op; "issue": register writes + Start (for a
                 multi-call MatmulConvNode, the loop that issues every call);
                 "wait": blocked on the node's kernel; "sync": blocked on a
                 start_sync call
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

ISSUE_US = 3.0          # register writes + Start through the UIO mapping


@dataclass
class Timeline:
    total_us: float
    cpu_us: float                                   # host ops + issue
    lane_us: Dict[str, float]
    waits: List[Tuple[int, str, float]]             # (node waited for, lane, µs)
    node_end: Dict[int, float] = field(default_factory=dict)
    unpriced: List[int] = field(default_factory=list)
    spans: List[Tuple[str, int, float, float, str]] = field(default_factory=list)  # (lane, node, t0, t1, kind)

    @property
    def wait_us(self) -> float:
        return sum(w for _, _, w in self.waits)

    def top_waits(self, n: int = 10) -> List[Tuple[int, str, float, int]]:
        """[(node, lane, total µs, times)] of the longest waits, per node."""
        agg: Dict[Tuple[int, str], List[float]] = defaultdict(list)
        for idx, lane, w in self.waits:
            agg[(idx, lane)].append(w)
        rows = [(i, lane, sum(v), len(v)) for (i, lane), v in agg.items()]
        return sorted(rows, key=lambda r: -r[2])[:n]


def simulate(cg, kernel_us: Callable[[object], Optional[float]],
             host_us: Callable[[object], Optional[float]],
             issue_us: float = ISSUE_US, events: Optional[list] = None) -> Timeline:
    """Replay ``cg``'s event stream (or ``events``) with these durations; a
    duration of ``None`` counts as 0 and the node is listed in
    ``Timeline.unpriced``."""
    events = events if events is not None else cg._compute_event_stream()
    by_index = {sn.index: sn for sn in cg._graph.nodes}
    t = cpu = 0.0
    lane_us: Dict[str, float] = defaultdict(float)
    node_end: Dict[int, float] = {}
    waits: List[Tuple[int, str, float]] = []
    unpriced: List[int] = []
    spans: List[Tuple[str, int, float, float, str]] = []

    def dur(fn, sn) -> float:
        d = fn(sn)
        if d is None:
            unpriced.append(sn.index)
            return 0.0
        return float(d)

    for ev in events:
        kind = ev[0]
        if kind in ("wait", "drain"):
            _, lane, idx = ev
            end = node_end.get(idx, t)
            if end > t:
                waits.append((idx, lane, end - t))
                spans.append(("CPU", idx, t, end, "wait"))
                t = end
        elif kind == "start":
            sn = by_index[ev[1]]
            d = dur(kernel_us, sn)
            calls = getattr(sn, "calls", 1) if hasattr(sn, "conv_n") else 1
            lane = cg._kernel_id_of(sn) or "?"
            lane_us[lane] += d
            spans.append((lane, sn.index, t, t + d, "kernel"))
            node_end[sn.index] = t + d
            t0 = t
            if calls > 1:
                per = d / calls
                t += (calls - 1) * per + issue_us
            else:
                t += issue_us
            spans.append(("CPU", sn.index, t0, t, "issue"))
            cpu += issue_us
        elif kind == "start_sync":
            sn = by_index[ev[1]]
            d = dur(kernel_us, sn)
            lane = cg._kernel_id_of(sn) or "?"
            lane_us[lane] += d
            spans.append((lane, sn.index, t, t + d, "kernel"))
            spans.append(("CPU", sn.index, t, t + d, "sync"))
            t += d
            node_end[sn.index] = t
        elif kind == "cpu":
            sn = by_index[ev[1]]
            d = dur(host_us, sn)
            spans.append(("CPU", sn.index, t, t + d, "host"))
            t += d
            cpu += d
            node_end[sn.index] = t
    return Timeline(t, cpu, dict(lane_us), waits, node_end, unpriced, spans)


def kernel_duration_fn(perf_model, layouts: dict, keys_of: Optional[Callable] = None,
                       fallback: Optional[Callable] = None):
    """``sn -> µs`` of a kernel node: the performance model's price of its
    kernel_calls() (a runtime-keys attention call at ``keys_of(sn)`` keys);
    ``fallback(sn)`` (e.g. the analytic model) when it cannot be priced."""
    def f(sn):
        if not hasattr(sn, "kernel_calls"):
            return 0.0
        if keys_of is not None and getattr(sn, "static", True) is False:
            calls = sn.kernel_calls(layouts, keys=keys_of(sn))
        else:
            calls = sn.kernel_calls(layouts)
        t = perf_model.calls_us(calls)
        if t is None and fallback is not None:
            t = fallback(sn)
        return t
    return f


__all__ = ("ISSUE_US", "Timeline", "simulate", "kernel_duration_fn")
