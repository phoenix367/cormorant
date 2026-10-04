"""
Performance models of one bitstream (doc/plans/TACTICS_PLAN.md §4.2).

A model file (``perf_models/<platform>/<bitstream-id>.json``, written by
``perf_calibrate.py fit``) holds

  * exact entries: the measured µs of every calibrated call signature (the
    kernels are deterministic: a measured call is known);
  * kernel families: a linear model per family over the analytic cost
    model's terms (:func:`features`), fitted to the measurements, with the
    range of every register it was calibrated on;
  * the validation of each family on held-out calls.

:meth:`PerfModel.predict` prices a :class:`~src.perf_calls.KernelCall`:
the exact entry when there is one, else the family model inside its
calibrated range, else ``None`` (the planner then keeps today's choice).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .perf_calls import KernelCall

MHZ = 100.0


# ---------------------------------------------------------------------------
# Families and features
# ---------------------------------------------------------------------------

def family(c: KernelCall) -> str:
    f = c.fields
    if c.kernel == "ConvKernel":
        if f["is_dw"]:
            return "conv-dw"
        if (f["kh"] == 1 and f["stride_h"] == 1 and f["stride_w"] == f["kw"]
                and f["pad_top"] == 0 and f["pad_left"] == 0):
            return "conv-mm"
        return "conv"
    if c.kernel == "MatmulKernel":
        return "mm-gemv" if f["gemv_kw"] else "mm-tiled"
    if c.kernel == "VectorOPKernel":
        return "vecop-div" if f["op"] == 3 else "vecop"
    return "pool"


def conv_segment_terms(f) -> Dict[str, float]:
    """conv_cycles()'s walk (cost_model.py) with every overlap assumption
    kept apart: per (chunk, input-channel tile, width tile, M-group) segment
    the sweep s and the weight fill w, summed as s, w, max(s, w) (fill fully
    hidden), max(0, w - s / 2) (the analytic model's half-hidden fill) and
    w itself (not hidden); the row loader likewise; plus the segment and
    chunk counts for fixed per-segment costs."""
    from .cost_model import (CONV_TILE_IC, CONV_TILE_M, CONV_WEIGHT_PORT_ELEMS, DRAIN_SEG,
                             DRAIN_STEP_RAMP, PIXEL_OVERHEAD, ROW_FILL_LATENCY, _conv_geom,
                             _loader_row_cycles, _tile_pixel_units)
    in_ch, out_ch, in_h, in_w = f["in_ch"], f["out_ch"], f["in_h"], f["in_w"]
    oh, ow, kh, kw = f["out_h"], f["out_w"], f["kh"], f["kw"]
    sh, sw, dh, dw = f["stride_h"], f["stride_w"], f["dilation_h"], f["dilation_w"]
    pt = f["pad_top"]
    m_tiles, ic_tiles, mtg, groups, per, chunks, owpt, owt = _conv_geom(
        in_ch, out_ch, oh, ow, kh, kw, sh, sw, dh, dw)
    E, T = CONV_WEIGHT_PORT_ELEMS, CONV_TILE_IC
    fill_steps = sum(-(-min(CONV_TILE_M, out_ch - t * CONV_TILE_M) // E) for t in range(m_tiles))
    acc = dict(sweep=0.0, wfill=0.0, smax=0.0, half=0.0, ph1=0.0, ph3=0.0, rowfill=0.0,
               loader=0.0, ldr_exposed=0.0, segments=0.0, blocks=0.0, chunks=float(chunks))
    for c in range(chunks):
        rows = min(per, oh - c * per)
        acc["ph1"] += rows * ow * m_tiles
        L = rows * ow
        nseg = -(-L // DRAIN_SEG)
        acc["ph3"] += fill_steps * L + min(L, DRAIN_SEG) + DRAIN_STEP_RAMP * (m_tiles * nseg + 1)
        r0 = c * per * sh - pt
        r1 = (c * per + rows - 1) * sh + (kh - 1) * dh - pt
        in_rows = max(0, min(r1, in_h - 1) - max(r0, 0) + 1)
        for ict in range(ic_tiles):
            icv = min(T, in_ch - ict * T)
            lanes = E if (ict == ic_tiles - 1 and icv <= E) else T
            for t in range(owt):
                tw = min(owpt, ow - t * owpt)
                cols = min(in_w, (tw - 1) * sw + (kw - 1) * dw + 1)
                units = _tile_pixel_units(ow, owpt, t, kh, kw)
                fill_c = in_rows * (cols + ROW_FILL_LATENCY)
                ldr = in_rows * _loader_row_cycles(icv, cols)
                blk = 0.0
                for g in range(groups):
                    G = min(mtg, m_tiles - g * mtg)
                    mv = sum(min(CONV_TILE_M, out_ch - (g * mtg + i) * CONV_TILE_M) for i in range(G))
                    w = mv * kh * kw * lanes / E
                    sw_ = rows * units * G + PIXEL_OVERHEAD
                    acc["sweep"] += sw_
                    acc["wfill"] += w
                    acc["smax"] += max(sw_, w)
                    acc["half"] += max(0.0, w - sw_ / 2)
                    acc["segments"] += 1
                    blk += sw_
                acc["rowfill"] += fill_c
                acc["loader"] += ldr
                acc["ldr_exposed"] += max(0.0, ldr - (blk + fill_c))
                acc["blocks"] += 1
    b = f["batch"]
    out = {k: b * v for k, v in acc.items()}
    out["one"] = 1.0
    return out


def _conv_terms(f) -> Dict[str, float]:
    from .cost_model import conv_cycles
    r = conv_cycles(in_ch=f["in_ch"], out_ch=f["out_ch"], in_h=f["in_h"], in_w=f["in_w"],
                    oh=f["out_h"], ow=f["out_w"], kh=f["kh"], kw=f["kw"],
                    sh=f["stride_h"], sw=f["stride_w"], dh=f["dilation_h"], dw=f["dilation_w"],
                    pt=f["pad_top"], pl=f["pad_left"])
    b = f["batch"]
    ict, owt, chunks = r["ic_tiles"], r["owt"], r["chunks"]
    # every weight slab fetched once per (chunk, input-channel tile, output-width tile)
    wfetch = chunks * ict * owt * f["out_ch"] * f["kh"] * f["kw"] * 2.0
    return {"sweep": b * r["sweep"], "fill": b * r["fill"], "ph1": b * r["ph1"],
            "ph3": b * r["ph3"], "loads": b * r["loads"], "wfetch": b * wfetch,
            "segments": b * chunks * ict * owt * r["groups"],
            "out_words": b * f["out_ch"] * f["out_h"] * f["out_w"] / 8.0,
            "one": 1.0}


# MatmulKernel exists as the Vitis HLS kernel and as the SystemVerilog one
# (kernels/matmul_rtl): the families carry the terms of both structures,
# the HLS tile model above and the RTL job walk (cost_model.rtl_matmul_terms,
# prefixed "rtl_"), and the fit keeps the set that fits the bitstream's
# measurements (perf_fit.FEATURE_SETS).
RTL_PREFIX = "rtl_"


def _rtl_terms(f) -> Dict[str, float]:
    from .cost_model import rtl_matmul_terms
    t = rtl_matmul_terms(f["n"], f["k"], f["m"], f["batch"], bool(f["b_packed"]), f["gemv_kw"])
    return {RTL_PREFIX + name: v for name, v in t.items() if name != "one"}


def features(c: KernelCall) -> Dict[str, float]:
    """The regressors of the call's family, in cycles-like units."""
    f = c.fields
    fam = family(c)
    if fam in ("conv", "conv-mm"):
        return _conv_terms(f)
    if fam == "conv-dw":
        pix = f["batch"] * f["out_ch"] * f["out_h"] * f["out_w"]
        return {"taps": pix * f["kh"] * f["kw"] / 16.0, "out_words": pix / 8.0,
                "in_words": f["batch"] * f["in_ch"] * f["in_h"] * f["in_w"] / 8.0,
                "rows": f["batch"] * (f["out_ch"] / 16.0) * f["out_h"], "one": 1.0}
    if fam == "mm-tiled":
        from ._matmul_hw_config import MATMUL_TILE_M, MATMUL_TILE_N
        n, k, m, bt = f["n"], f["k"], f["m"], f["batch"]
        blocks = bt * math.ceil(n / MATMUL_TILE_N) * math.ceil(m / MATMUL_TILE_M)
        return {"kloop": blocks * MATMUL_TILE_N * k, "blocks": blocks,
                "b_words": bt * k * m / 8.0 * (0.0 if f["b_packed"] else 1.0),
                "c_words": bt * n * m / 8.0, "one": 1.0, **_rtl_terms(f)}
    if fam == "mm-gemv":
        n, k, m, bt = f["n"], f["k"], f["m"], f["batch"]
        return {"b_words": bt * n * k * m / 8.0, "rows": bt * n, "k_words": bt * n * k / 8.0,
                "m": bt * n * m, "one": 1.0, **_rtl_terms(f)}
    if fam in ("vecop", "vecop-div"):
        words = f["outer"] * math.ceil(f["size"] / 8)
        unary = f["op"] in (4, 5)
        return {"words": words, "runs": f["outer"], "b_words": 0.0 if unary else words,
                "one": 1.0}
    pix = f["batch"] * f["channels"] * f["out_h"] * f["out_w"]
    return {"taps": pix * f["pool_h"] * f["pool_w"] / 8.0, "out_words": pix / 8.0,
            "in_words": f["batch"] * f["channels"] * f["in_h"] * f["in_w"] / 8.0,
            "planes": f["batch"] * f["channels"], "one": 1.0}


def feature_vector(c: KernelCall, names: List[str]) -> List[float]:
    d = features(c)
    return [float(d.get(n, 0.0)) for n in names]


# ---------------------------------------------------------------------------
# The model file
# ---------------------------------------------------------------------------

@dataclass
class Family:
    name:     str
    features: List[str]
    coef:     List[float]
    range:    Dict[str, Tuple[int, int]]
    holdout:  Dict[str, float]
    knn:      Optional[dict] = None

    def in_range(self, c: KernelCall) -> bool:
        f = c.fields
        return all(lo <= f[k] <= hi for k, (lo, hi) in self.range.items())

    def predict_us(self, c: KernelCall) -> float:
        """The linear base times the k-nearest-neighbour correction
        (perf_fit.fit_family)."""
        x = feature_vector(c, self.features)
        base = max(sum(a * b for a, b in zip(self.coef, x, strict=True)), 1e-3)
        kn = self.knn or {}
        k = int(kn.get("k", 0))
        if not k or not kn.get("z"):
            return base
        z = [(math.log1p(max(v, 0.0)) - m) / s for v, m, s in zip(x, kn["mu"], kn["sd"], strict=True)]
        d = [math.sqrt(sum((a - b) ** 2 for a, b in zip(row, z, strict=True))) for row in kn["z"]]
        nn = sorted(range(len(d)), key=d.__getitem__)[:k]
        w = [1.0 / (d[i] + 1e-3) for i in nn]
        return base * math.exp(sum(kn["r"][i] * wi for i, wi in zip(nn, w, strict=True)) / sum(w))

    @property
    def p90_error(self) -> float:
        """The held-out 90th-percentile relative error (1.0 if unknown)."""
        return float(self.holdout.get("p90_rel", 1.0)) if self.holdout.get("n") else 1.0


class PerfModel:
    def __init__(self, d: dict, path: Optional[str] = None):
        self.path = path
        self.platform = d["platform"]
        self.bitstream = d["bitstream"]
        self.exact: Dict[str, float] = {k: float(v["us"]) for k, v in d["exact"].items()}
        self.families = {n: Family(n, v["features"], v["coef"],
                                   {k: tuple(r) for k, r in v["range"].items()}, v.get("holdout", {}),
                                   v.get("knn"))
                         for n, v in d["families"].items()}
        self.call_overhead_us = float(d.get("call_overhead_us", 0.0))

    @classmethod
    def load(cls, path) -> "PerfModel":
        return cls(json.loads(Path(path).read_text()), str(path))

    def predict(self, c: KernelCall) -> Optional[Tuple[float, str]]:
        """``(µs of one call, "exact" | "model")`` or None (out of range)."""
        k = KernelCall(c.kernel, c.regs, 1).key()
        if k in self.exact:
            return self.exact[k], "exact"
        fam = self.families.get(family(c))
        if fam is None or not fam.in_range(c):
            return None
        return max(fam.predict_us(c), 0.0), "model"

    def calls_us(self, calls) -> Optional[float]:
        """Total µs of ``calls`` (counts included), None if any is unpriced."""
        b = self.calls_band(calls)
        return None if b is None else b[0]

    def calls_band(self, calls) -> Optional[Tuple[float, float]]:
        """``(total µs, relative error)`` of ``calls``: an exact entry has
        error 0, a family prediction its family's held-out p90 error; the
        total's error is the time-weighted mean.  None if any is unpriced."""
        tot = err = 0.0
        for c in calls:
            p = self.predict(c)
            if p is None:
                return None
            t = p[0] * c.count
            tot += t
            if p[1] != "exact":
                err += t * self.families[family(c)].p90_error
        return tot, (err / tot if tot > 0 else 0.0)


# A tactic priced by a family model may replace a measured choice only when
# the family predicts this well (held-out p90): the families' error tails
# are heavy, and one over-optimistic prediction slipped past a p90 band in
# the first campaign (SmolLM2 prefill 16, TACTICS_PLAN §5 T2).
MAX_MODEL_ERROR = 0.05


def clearly_faster(cand: Tuple[float, float], base: Tuple[float, float], min_gain: float) -> bool:
    """The candidate beats the baseline even with each at the unfavourable
    end of its error band, by ``min_gain`` more — and the candidate is
    measured or priced by a family within ``MAX_MODEL_ERROR``."""
    (tc, ec), (tb, eb) = cand, base
    if ec > MAX_MODEL_ERROR:
        return False
    return tc * (1.0 + ec) < tb * (1.0 - eb) * (1.0 - min_gain)


def default_model_path(bitstream_id: Optional[str], platform: str = "kv260") -> Optional[Path]:
    if not bitstream_id:
        return None
    p = Path(__file__).resolve().parent.parent / "perf_models" / platform / f"{bitstream_id}.json"
    return p if p.exists() else None


__all__ = ("MHZ", "family", "features", "feature_vector", "Family", "PerfModel",
           "clearly_faster", "MAX_MODEL_ERROR", "default_model_path")
