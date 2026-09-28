"""
Host-op timing models (doc/plans/TACTICS_PLAN.md §4.2 "host-op models").

A host op (``kernel_name == ""``: HostNode, SpaceToDepthNode, the LLM /
vision host ops) runs on the board's CPU with ``INFERENCE_HOST_THREADS``
threads.  Its time comes from per-layer profiles of generated projects on
the board (the profiler's host-op brackets are the op's run time):

  * exact entries: the mean µs per signature (the op kind plus the shapes
    of its inputs and output) — an op measured once is known;
  * per kind a non-negative linear fit over the element and row counts.

The file is per platform (the CPU does not change with the bitstream):
``perf_models/<platform>/host.json``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

FEATURES = ("in_elems", "out_elems", "rows", "one")


def is_host_op(sn) -> bool:
    return getattr(type(sn), "kernel_name", "x") == "" and type(sn).__name__ != "ReshapeNode" \
        and not getattr(sn, "is_view", False)


def signature(sn) -> str:
    ins = ";".join("x".join(map(str, t.shape)) for t in sn.inputs if not t.is_weight)
    return f"{type(sn).__name__}:{ins}->{'x'.join(map(str, sn.output.shape))}"


def features(sn) -> Dict[str, float]:
    """in_elems, out_elems, rows, one — or the node's own ``host_features()``
    when it touches only part of its inputs (a softmax over some columns)."""
    own = getattr(sn, "host_features", None)
    if own is not None:
        return {**{k: 0.0 for k in FEATURES}, **own(), "one": 1.0}
    ins = [t for t in sn.inputs if not t.is_weight]
    shape = list(sn.output.shape) or [1]
    return {"in_elems": float(sum(math.prod(t.shape or [1]) for t in ins)),
            "out_elems": float(math.prod(shape)),
            "rows": float(shape[0] if len(shape) > 1 else 1), "one": 1.0}


class HostModel:
    def __init__(self, d: Optional[dict] = None):
        d = d or {"exact": {}, "kinds": {}}
        self.exact: Dict[str, float] = {k: float(v["us"]) for k, v in d["exact"].items()}
        self.kinds: Dict[str, dict] = d["kinds"]
        self.meta = {k: v for k, v in d.items() if k not in ("exact", "kinds")}

    @classmethod
    def load(cls, path) -> "HostModel":
        return cls(json.loads(Path(path).read_text()))

    def to_dict(self) -> dict:
        return {**self.meta, "exact": {k: {"us": round(v, 3)} for k, v in sorted(self.exact.items())},
                "kinds": self.kinds}

    def predict(self, sn) -> Optional[Tuple[float, str]]:
        sig = signature(sn)
        if sig in self.exact:
            return self.exact[sig], "exact"
        k = self.kinds.get(type(sn).__name__)
        if not k:
            return None
        f = features(sn)
        return max(0.0, sum(c * f[n] for c, n in zip(k["coef"], FEATURES, strict=True))), "model"

    def us(self, sn) -> Optional[float]:
        p = self.predict(sn)
        return None if p is None else p[0]

    @staticmethod
    def fit(observations: Iterable[Tuple[object, float]], meta: Optional[dict] = None) -> "HostModel":
        """``observations``: (host node, measured µs) pairs from profiles."""
        from .perf_fit import nnls
        by_sig: Dict[str, List[float]] = {}
        by_kind: Dict[str, List[Tuple[Dict[str, float], float]]] = {}
        for sn, us in observations:
            by_sig.setdefault(signature(sn), []).append(float(us))
            by_kind.setdefault(type(sn).__name__, []).append((features(sn), float(us)))
        kinds = {}
        for kind, rows in by_kind.items():
            X = np.array([[f[n] for n in FEATURES] for f, _ in rows])
            y = np.array([u for _, u in rows])
            ok = y > 0
            if ok.sum() < 1:
                continue
            w = 1.0 / y[ok]
            coef = nnls(X[ok] * w[:, None], np.ones(int(ok.sum())))
            rel = np.abs(X[ok] @ coef - y[ok]) / y[ok]
            kinds[kind] = {"coef": [float(c) for c in coef], "n": int(ok.sum()),
                           "median_rel": float(np.median(rel)), "max_rel": float(rel.max())}
        return HostModel({**(meta or {}),
                          "exact": {k: {"us": float(np.mean(v))} for k, v in by_sig.items()},
                          "kinds": kinds})


def observations_from_profile(graph, layers: Iterable[dict]) -> List[Tuple[object, float]]:
    """(host node, mean µs) of one graph from a profiler phase's layers
    (``{"name", "calls", "mean_us"}``), matched by the node's ONNX name."""
    by_name = {sn.onnx_node.name: sn for sn in graph.nodes if is_host_op(sn) and sn.onnx_node.name}
    out = []
    for ly in layers:
        sn = by_name.get(ly.get("name"))
        if sn is not None and ly.get("calls"):
            out.append((sn, float(ly["mean_us"])))
    return out


def default_host_model_path(platform: str = "kv260") -> Path:
    return Path(__file__).resolve().parent.parent / "perf_models" / platform / "host.json"


__all__ = ("FEATURES", "is_host_op", "signature", "features", "HostModel",
           "observations_from_profile", "default_host_model_path")
