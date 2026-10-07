#!/usr/bin/env python3
"""
fit_cost_model.py — refit the engine cost model's board terms to a
performance-model calibration campaign (doc/plans/OFFLOAD_PLAN.md §2.3).

``src/cost_model.py`` prices the scheduler's engine choices (MatMul on
ConvKernel vs MatmulKernel, GEMV or tiled, the lowered conv's kw / row split,
``--fc-conv``) in kernel cycles with three board-fitted terms:

  RTL_CONV_BOARD  the parameters of ConvKernel's pipeline recurrence
                  (``rtl_conv_walk(..., board=True)``, ``conv_board_cycles``);
  RTL_COEF        MatmulKernel's term weights (``rtl_matmul_terms``);
  CALL_OVERHEAD   the host's cost of one call, charged per call to both engines.

A campaign (``perf_models/<platform>/<id>.{cases,calib,json}``, written by
``perf_calibrate.py``) times every call alone with ``bench_src/calib_runner.c``:
the register writes, Start and the IsDone poll — what the generated code pays
per call, the host's part included.  So a measured call, in cycles at the
model's ``clock_mhz``, is ``kernel model + CALL_OVERHEAD``, and that is how
the callers combine them (matmul_lowering, tactics, llm_entries:
``conv_board_cycles(...)["total"] + CALL_OVERHEAD`` against
``matmul_cycles(...) + CALL_OVERHEAD``).  The fit therefore takes
CALL_OVERHEAD from the call floor — the cheapest measured call of the
campaign, a 10-element VectorOP Add, which is ``call_overhead_us`` of the
model file — and fits the kernels' fixed costs (``RTL_CONV_BOARD["ONE"]``,
``RTL_COEF["one"]``) net of it.  Only the sums ONE + CALL_OVERHEAD and
one + CALL_OVERHEAD reach a decision, so the split does not move any choice.

``fit [ID]`` (default: the local production bitstream,
``perf_calls.local_bitstream_id()``):

  * reports the CURRENT constants' relative error per kernel and family
    (ConvKernel conv / conv-dw / conv-mm, MatmulKernel mm-tiled / mm-gemv):
    "as used" (+ the current CALL_OVERHEAD) and "kernel" (+ the call floor,
    how the current constants were fitted);
  * fits RTL_COEF by non-negative least squares on the MatmulKernel calls
    (weights 1 / measured: relative error; leave-one-out reported);
  * fits RTL_CONV_BOARD by a coordinate search over its parameters (integers
    and three per-beat floats, bounded) from the current values and from
    their clock-scaled copy, minimising median + p90 (+ 0.1 x mean) of the
    relative error over the ConvKernel calls (``--objective log``: of
    |log ratio|).  The search runs the recurrence for every campaign call and
    many parameter sets at once (ConvTrace, a vectorised copy of
    ``rtl_conv_walk`` checked against it at the start and the result);
  * ``--shipped-weight W``: the conv objective is (1 - W) x all calls + W x
    the cases' "shipped" set (the calls the shipped models issue or any of
    their MatMul tactics would — the shapes the engine choices compare);
  * ``--decisions DIR`` (the ``decisions`` command's files): the shipped
    models' MatMuls with both engines priced by the bitstream's performance
    model (Decisions); the conv objective adds ``--decision-weight`` x the
    sum over the distinct options of (chosen engine's time / faster one's - 1),
    and the fit reports the wrong choices before / after;
  * leaves GEMV_WORD_CYCLE / GEMV_JOB_OVERHEAD (and the MM_* / CONV_* HLS
    terms) alone: ``gemv_cycles`` uses them only for the HLS MatmulKernel
    (``kernels.matmul.impl == "hls"``); the RTL kernel's GEMV path is
    ``rtl_matmul_cycles`` with RTL_COEF;
  * prints before / after errors and writes the proposed constants, the
    current ones and the error tables to a JSON file (``--out``).

``--eval ID ...`` also scores the current and proposed constants on other
campaigns; ``--holdout`` fits without the campaign's held-out calls and
reports them; ``--perf-check`` refits the bitstream's performance model in
memory with the proposed constants and compares the held-out errors.  The
performance models' ConvKernel families use the walk with the frozen
``RTL_CONV_FEATURES`` as features (``perf_model._rtl_conv_terms``), so a new
RTL_CONV_BOARD leaves the stored models' predictions alone (no
``perf_calibrate.py fit`` needed; ``--perf-check`` then shows no change).

``decisions`` builds the shipped models' graphs (``perf_calibrate.SHIPPED`` /
``shipped_graphs``; several GB of RAM and minutes for the LLMs — one
subprocess per model) with the module's constants and records the inputs of
every unplanned engine choice (``<model>.base.json``).  ``diff PROPOSED.json``
builds them again with the proposed constants (``<model>.proposed.json``) and
lists every MatMul whose engine, kernel width / output tile / row split or
GEMV / tiled path changes and every fully-connected Conv whose lowering
changes, with both options' estimates under either constant set, the change
priced by the bitstream's performance model, and per model how many choices
each constant set gets wrong against that model.  ``--base before`` compares
the JSON's "before" constants instead of the module's (after the proposal is
applied).

Outputs go to ``$COST_FIT_DIR`` (default ``<tmp>/cost_fit``).  Recommended run
(the constants of OFFLOAD_PLAN §2.3), from inference-scheduler/:
  .venv/bin/python tools/fit_cost_model.py decisions --out-dir D
  .venv/bin/python tools/fit_cost_model.py fit --shipped-weight 0.75 --decisions D \
      --perf-check --out D/cost_model.json
  .venv/bin/python tools/fit_cost_model.py diff D/cost_model.json --out-dir D

usage:
  .venv/bin/python tools/fit_cost_model.py fit [ID] [--out JSON] [--eval ID ...] [--holdout]
        [--perf-check] [--shipped-weight W] [--decisions DIR [--decision-weight L]]
        [--objective rel|log] [--start JSON ...] [--rounds N]
  .venv/bin/python tools/fit_cost_model.py decisions [--models NAME ...] [--out-dir DIR] [--rebuild]
  .venv/bin/python tools/fit_cost_model.py diff JSON [--models NAME ...] [--base module|before]
        [--out-dir DIR] [--rebuild] [--rebuild-base]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent.parent          # inference-scheduler/
sys.path.insert(0, str(HERE))

from src import cost_model as cm                         # noqa: E402
from src.perf_calls import KernelCall, local_bitstream_id  # noqa: E402
from src.perf_fit import _measured, kernel_clock_mhz, nnls  # noqa: E402
from src.perf_model import PerfModel, family             # noqa: E402

PLATFORM = "kv260"
MODELS_DIR = HERE / "perf_models" / PLATFORM
OUT_DIR = Path(os.environ.get("COST_FIT_DIR", Path(tempfile.gettempdir()) / "cost_fit"))
FAMILIES = {"ConvKernel": ("conv", "conv-dw", "conv-mm"), "MatmulKernel": ("mm-tiled", "mm-gemv")}
TERMS = ("stream", "drain", "aload", "steps", "runs", "one")    # cost_model.rtl_matmul_terms

# RTL_CONV_BOARD's search space: (kind, low, high).  CAP and HAZ are kernel
# structure (the weight FIFO's run-ahead, the short-sweep hazard), the rest
# DDR / host latencies in cycles and per-beat costs.
CONV_SPACE = {
    "CAP": ("int", 32, 4096), "XL": ("int", 0, 4000), "PL": ("int", 0, 4000),
    "FL": ("int", 0, 4000), "H": ("int", 0, 1000), "HAZ": ("int", 0, 1000),
    "HZW": ("int", 0, 1000), "LAT": ("int", 0, 4000), "ROW": ("int", 0, 64),
    "ONE": ("int", 0, 20000), "XBEAT": ("float", 0.25, 8.0), "XROW": ("int", 0, 1000),
    "RUN": ("int", 0, 64), "XLAT": ("int", 0, 4000), "WLAT": ("int", 0, 4000),
    "WBEAT": ("float", 0.25, 8.0), "DPX": ("float", 0.25, 8.0),
}
# the parameters that are a fixed time (DDR / pipeline latency): scaled by the
# clock ratio for the second starting point of the search
CONV_LATENCIES = ("XL", "PL", "FL", "H", "HZW", "LAT", "ONE", "XROW", "XLAT", "WLAT")
FIT_MHZ = 100.0                    # the clock the current constants were fitted at


def log(msg: str = "") -> None:
    print(msg, flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# The cost model's constants, patched everywhere they are read
# ─────────────────────────────────────────────────────────────────────────────

def current_constants() -> dict:
    return {"CALL_OVERHEAD": cm.CALL_OVERHEAD, "RTL_COEF": dict(cm.RTL_COEF),
            "RTL_CONV_BOARD": dict(cm.RTL_CONV_BOARD)}


def _importers() -> List[object]:
    """Every loaded module of the scheduler that holds its own CALL_OVERHEAD
    (``from .cost_model import CALL_OVERHEAD`` at module level copies it)."""
    import src.matmul_lowering  # noqa: F401 — load the module-level importers first
    import src.tactics  # noqa: F401
    out = []
    for name, mod in list(sys.modules.items()):
        if mod is None or mod is cm or not hasattr(mod, "CALL_OVERHEAD"):
            continue
        if name.startswith("src.") or name in ("perf_calibrate", "bert_schedule_stats"):
            out.append(mod)
    return out


@contextmanager
def patched(consts: Optional[dict]):
    """Run with the cost model's CALL_OVERHEAD / RTL_COEF / RTL_CONV_BOARD
    set to ``consts`` (missing keys unchanged): the dicts are updated in
    place (every alias sees them), CALL_OVERHEAD is set on cost_model and on
    every module that imported its value, rtl_conv_walk's cache is cleared;
    all restored on exit."""
    if not consts:
        yield
        return
    old = current_constants()
    mods = _importers()

    def apply(c):
        if "RTL_COEF" in c:
            cm.RTL_COEF.clear()
            cm.RTL_COEF.update(c["RTL_COEF"])
        if "RTL_CONV_BOARD" in c:
            cm.RTL_CONV_BOARD.clear()
            cm.RTL_CONV_BOARD.update(c["RTL_CONV_BOARD"])
        if "CALL_OVERHEAD" in c:
            cm.CALL_OVERHEAD = c["CALL_OVERHEAD"]
            for m in mods + [m for m in _importers() if m not in mods]:
                m.CALL_OVERHEAD = c["CALL_OVERHEAD"]
        cm.rtl_conv_walk.cache_clear()

    apply(consts)
    try:
        yield
    finally:
        apply(old)


# ─────────────────────────────────────────────────────────────────────────────
# The campaign
# ─────────────────────────────────────────────────────────────────────────────

class Campaign:
    """The measured ConvKernel / MatmulKernel calls of one campaign, in
    cycles at its kernel clock."""

    def __init__(self, bid: str, models_dir: Path = MODELS_DIR):
        self.bid = bid
        data = json.loads((models_dir / f"{bid}.calib.json").read_text())
        cases_p = models_dir / f"{bid}.cases.json"
        cases = json.loads(cases_p.read_text()) if cases_p.exists() else {"cases": []}
        self.info = {e["key"]: e for e in cases["cases"]}
        self.mhz = float(kernel_clock_mhz(bid, models_dir / f"{bid}.json"))
        meas = _measured(data)
        self.floor_key = min(meas, key=lambda k: meas[k]["us"])
        self.floor_us = meas[self.floor_key]["us"]
        self.floor = self.floor_us * self.mhz
        self.conv: List[dict] = []
        self.mm: List[dict] = []
        for k, m in meas.items():
            c = KernelCall.from_key(k)
            row = {"key": k, "call": c, "f": c.fields, "fam": family(c), "y": m["us"] * self.mhz,
                   "holdout": bool(self.info.get(k, {}).get("holdout"))}
            if c.kernel == "ConvKernel":
                self.conv.append(row)
            elif c.kernel == "MatmulKernel":
                self.mm.append(row)
        self.conv_y = np.array([r["y"] for r in self.conv])
        self.mm_y = np.array([r["y"] for r in self.mm])
        self.mm_X = np.array([[cm.rtl_matmul_terms(r["f"]["n"], r["f"]["k"], r["f"]["m"],
                                                   r["f"]["batch"], bool(r["f"]["b_packed"]),
                                                   r["f"]["gemv_kw"])[t] for t in TERMS]
                              for r in self.mm])
        # the distinct walks (batch and has_bias do not enter rtl_conv_walk)
        geoms: Dict[tuple, int] = {}
        self.conv_g = []
        for r in self.conv:
            f = r["f"]
            g = (f["in_ch"], f["out_ch"], f["in_h"], f["in_w"], f["out_h"], f["out_w"], f["kh"],
                 f["kw"], f["stride_h"], f["stride_w"], f["dilation_h"], f["dilation_w"],
                 f["pad_top"], f["pad_left"], bool(f["is_dw"]))
            self.conv_g.append(geoms.setdefault(g, len(geoms)))
        self.geoms = list(geoms)
        self.conv_g = np.array(self.conv_g)
        self.conv_b = np.array([r["f"]["batch"] for r in self.conv], dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# Model predictions (kernel cycles, the host's call excluded)
# ─────────────────────────────────────────────────────────────────────────────

# The walk's structure per geometry — every sweep's parameter-free
# quantities, in rtl_conv_walk's loop order — so that the recurrence can run
# for many parameter sets and all geometries at once (numpy, one step per
# sweep index).  ConvTrace.check compares it with cost_model.rtl_conv_walk.
_SWEEP = ("b", "mv", "w", "s", "npn", "rows", "loads", "icv", "cw", "inr", "cols", "hz",
          "prev", "f2", "end", "dru")


def _trace(g: tuple) -> List[tuple]:
    """rtl_conv_walk's loops for geometry ``g`` (its 14 geometry arguments
    and ``dwise``) without the parameters: one tuple per sweep (_SWEEP)."""
    in_ch, out_ch, in_h, in_w, oh, ow, kh, kw, sh, sw, dh, dw, pt, pl, dwise = g
    m_tiles, ic_tiles, mtg, groups, per, chunks, owpt, owt = cm._conv_geom(
        in_ch, out_ch, oh, ow, kh, kw, sh, sw, dh, dw)
    if dwise:
        per = max(1, min(oh, cm.CONV_MAX_ACC_PERSIST_ENTRIES // (ow * m_tiles * cm.CONV_TILE_M)))
        chunks, groups, mtg = -(-oh // per), 1, 1
    E, T, TM = cm.CONV_WEIGHT_PORT_ELEMS, cm.CONV_TILE_IC, cm.CONV_TILE_M
    npos = kh * kw
    nwin = max(npos, 2)
    out: List[list] = []
    prev_s = 0
    for c in range(chunks):
        rows = min(per, oh - c * per)
        r0 = c * per * sh - pt
        r1 = (c * per + rows - 1) * sh + (kh - 1) * dh - pt
        in_rows = max(0, min(r1, in_h - 1) - max(r0, 0) + 1)
        if sh > (kh - 1) * dh + 1:
            in_rows = sum(1 for r in range(max(r0, 0), min(r1, in_h - 1) + 1)
                          if (r + pt) % sh <= (kh - 1) * dh)
        first = True
        for ct in range(m_tiles if dwise else ic_tiles):
            icv = min(T, (out_ch if dwise else in_ch) - ct * T)
            half = not dwise and ct == ic_tiles - 1 and icv <= E
            for t in range(owt):
                cols = min(in_w, (min(owpt, ow - t * owpt) - 1) * sw + (kw - 1) * dw + 1)
                ow0 = t * owpt
                npairs = ((min(ow, ow0 + owpt) - 1) >> 1) - (ow0 >> 1) + 1
                for gi in range(groups):
                    if dwise:
                        G, mv = 1, icv
                        b = mv * -(-npos // E)
                    else:
                        G = min(mtg, m_tiles - gi * mtg)
                        mv = sum(min(TM, out_ch - (gi * mtg + i) * TM) for i in range(G))
                        b = mv * npos * (1 if half else 2)
                    out.append([b, mv, mv * npos, rows * npairs * nwin * G, npairs * npos, rows,
                                int(dwise or gi == 0), icv, -(-cols // E), in_rows, cols,
                                int(not dwise and ct > 0), prev_s, int(first and c >= 2), 0, 0])
                    prev_s = rows * npairs * nwin * G
                    first = False
        out[-1][14] = 1
        out[-1][15] = sum(rows * ow * (2 if min(TM, out_ch - mt * TM) > E else 1)
                          for mt in range(m_tiles))
    return [tuple(r) for r in out]


class ConvTrace:
    """rtl_conv_walk(board=True)'s ``total - ONE`` for every geometry of a
    campaign and K parameter sets at once."""

    def __init__(self, geoms: List[tuple]):
        traces = [_trace(g) for g in geoms]
        nsw = np.array([len(t) for t in traces])
        order = np.argsort(-nsw, kind="stable")          # longest first: step i's active
        self.n = len(geoms)                               # geometries are a prefix
        self.inv = np.argsort(order)
        ns = nsw[order]
        self.maxsw = int(ns.max()) if len(ns) else 0
        self.active = np.array([int((ns > i).sum()) for i in range(self.maxsw)])
        self.start = np.concatenate([[0], np.cumsum(self.active)])
        flat = np.empty((int(ns.sum()), len(_SWEEP)))
        for j, gi in enumerate(order):               # step-major: sweep i of the j-th longest
            flat[self.start[np.arange(len(traces[gi]))] + j] = traces[gi]
        self.cols = {name: np.ascontiguousarray(flat[:, k:k + 1]) for k, name in enumerate(_SWEEP)}
        self.geoms = geoms

    def run(self, P: Dict[str, np.ndarray]) -> np.ndarray:
        """``total - ONE`` per geometry (rows, campaign order) and parameter
        set (columns) for P = {name: array of K values}."""
        K = len(P["CAP"])
        p = {k: np.asarray(v, float)[None, :] for k, v in P.items()}
        L, F, Pd, S, S2, D0, D1 = (np.zeros((self.n, K)) for _ in range(7))
        c = self.cols
        for i in range(self.maxsw):
            na, lo = int(self.active[i]), int(self.start[i])
            q = {name: a[lo:lo + na] for name, a in c.items()}
            b, mv = q["b"], q["mv"]
            bb = np.maximum(b, (p["WLAT"] + b / mv) * mv / 8) * p["WBEAT"]
            pk = q["rows"] * (q["npn"] + p["ROW"])
            ld = q["loads"] > 0
            if ld.any():
                icv, inr = q["icv"], q["inr"]
                words = q["cw"] + p["RUN"]
                xrow = np.maximum(icv * words * p["XBEAT"] + p["XROW"], (p["XLAT"] + words) * icv / 16)
                pk = np.where(ld, np.maximum(pk + inr * q["cols"], inr * xrow) + p["XL"], pk)
            Ls, Fs, Ss = L[:na], F[:na], S[:na]
            Ln = np.maximum(Ls, Fs - p["CAP"]) + bb
            Fn = np.maximum(np.maximum(Fs, S2[:na]) + q["w"], Ln)
            Pd[:na] += pk
            hz = (q["hz"] > 0) & (q["prev"] < p["HAZ"])
            h = p["H"] + np.where(hz, p["HZW"], 0.0)
            S0 = np.maximum(Ss + h, Fn + p["FL"])
            f2 = q["f2"] > 0
            if f2.any():
                S0 = np.where(f2, np.maximum(S0, D0[:na]), S0)
            Sn = np.maximum(S0 + q["s"], Pd[:na] + p["PL"])
            S2[:na] = Ss
            S[:na] = Sn
            F[:na] = Fn
            L[:na] = Ln
            end = q["end"] > 0
            if end.any():
                d1 = D1[:na].copy()
                D1[:na] = np.where(end, np.maximum(d1, Sn + p["LAT"]) + p["DPX"] * q["dru"], d1)
                D0[:na] = np.where(end, d1, D0[:na])
        return D1[self.inv]

    def check(self, p: dict, tol: float = 1e-9) -> float:
        """The largest relative difference to cost_model.rtl_conv_walk (board
        parameters ``p``); raises when above ``tol``."""
        mine = self.run({k: [v] for k, v in p.items()})[:, 0]
        old = cm.RTL_CONV_BOARD
        cm.RTL_CONV_BOARD = p
        try:
            ref = np.array([cm.rtl_conv_walk.__wrapped__(*g[:14], g[14], True)["total"] - p["ONE"]
                            for g in self.geoms])
        finally:
            cm.RTL_CONV_BOARD = old
        d = float(np.max(np.abs(mine - ref) / np.maximum(np.abs(ref), 1.0))) if len(ref) else 0.0
        if d > tol:
            raise RuntimeError(f"ConvTrace differs from cost_model.rtl_conv_walk by {d:.3g}: "
                               "update _trace / ConvTrace.run to the walk")
        return d


class ConvEvaluator:
    """Kernel-cycle predictions of the campaign's ConvKernel calls for
    parameter sets: ``batch x (walk - ONE) + ONE`` (conv_batch_cycles)."""

    def __init__(self, camp: Campaign):
        self.camp = camp
        self.trace = ConvTrace(camp.geoms)

    def predict_many(self, ps: List[dict]) -> List[np.ndarray]:
        c = self.camp
        W = self.trace.run({k: [p[k] for p in ps] for k in ps[0]})
        return [c.conv_b * W[c.conv_g, j] + p["ONE"] for j, p in enumerate(ps)]

    def predict(self, p: dict) -> np.ndarray:
        return self.predict_many([p])[0]


def mm_predict(camp: Campaign, coef: dict) -> np.ndarray:
    return camp.mm_X @ np.array([coef[t] for t in TERMS])


# ─────────────────────────────────────────────────────────────────────────────
# Errors
# ─────────────────────────────────────────────────────────────────────────────

def rel_err(pred: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.abs(pred - y) / y


def err_stats(r: np.ndarray) -> dict:
    if not len(r):
        return {"n": 0}
    return {"n": int(len(r)), "median": float(np.median(r)), "p90": float(np.quantile(r, 0.9)),
            "max": float(r.max()), "mean": float(r.mean()),
            "bias": None}


def error_table(camp: Campaign, conv_pred: np.ndarray, mm_pred: np.ndarray, host: float) -> dict:
    """{group: stats} for measured vs model + host; ``bias`` is the median
    log(predicted / measured) (negative: the model is optimistic)."""
    out = {}
    for kernel, rows, y, pred in (("ConvKernel", camp.conv, camp.conv_y, conv_pred),
                                  ("MatmulKernel", camp.mm, camp.mm_y, mm_pred)):
        p = pred + host
        r = rel_err(p, y)
        lr = np.log(np.maximum(p, 1.0) / y)
        fams = np.array([x["fam"] for x in rows])
        for fam in FAMILIES[kernel]:
            m = fams == fam
            out[fam] = err_stats(r[m])
            if m.any():
                out[fam]["bias"] = float(np.median(lr[m]))
        out[kernel] = err_stats(r)
        out[kernel]["bias"] = float(np.median(lr))
    return out


def print_tables(title: str, tables: List[Tuple[str, dict]]) -> None:
    log(title)
    head = f"  {'group':13s} {'n':>5s}"
    for name, _ in tables:
        head += f" | {name:^31s}"
    log(head)
    sub = f"  {'':13s} {'':>5s}" + " | {:>6s} {:>6s} {:>7s} {:>8s}".format(
        "med", "p90", "max", "bias") * len(tables)
    log(sub)
    for g in ("conv", "conv-dw", "conv-mm", "ConvKernel", "mm-tiled", "mm-gemv", "MatmulKernel"):
        n = tables[0][1][g]["n"]
        line = f"  {g:13s} {n:5d}"
        for _, t in tables:
            s = t[g]
            line += (f" | {s['median'] * 100:5.1f}% {s['p90'] * 100:5.1f}% {s['max'] * 100:6.1f}%"
                     f" {s['bias']:+8.3f}") if s["n"] else " | " + " " * 31
        log(line)


# ─────────────────────────────────────────────────────────────────────────────
# Fits
# ─────────────────────────────────────────────────────────────────────────────

def _sig(v: float, digits: int = 4) -> float:
    return float(f"{v:.{digits}g}") if v else 0.0


def fit_rtl_coef(camp: Campaign, host: float, mask: Optional[np.ndarray] = None) -> Tuple[dict, dict]:
    """RTL_COEF by NNLS of (measured - host) on rtl_matmul_terms, weighted by
    1 / measured; returns (coef, {"loo": stats of the leave-one-out error})."""
    y = camp.mm_y
    X = camp.mm_X
    m = np.ones(len(y), bool) if mask is None else mask
    w = 1.0 / y
    coef = nnls(X[m] * w[m, None], (y[m] - host) * w[m])
    idx = np.where(m)[0]
    loo = np.empty(len(idx))
    for j, i in enumerate(idx):
        keep = idx[idx != i]
        c = nnls(X[keep] * w[keep, None], (y[keep] - host) * w[keep])
        loo[j] = abs(X[i] @ c + host - y[i]) / y[i]
    return {t: _sig(c) for t, c in zip(TERMS, coef, strict=True)}, {"loo": err_stats(loo)}


def _conv_objective(pred: np.ndarray, y: np.ndarray, host: float, kind: str = "rel") -> float:
    """median + p90 + 0.1 x mean of the relative error ("rel"), or of
    |log(predicted / measured)| ("log": an optimistic miss counts as much as
    a pessimistic one of the same ratio)."""
    p = pred + host
    r = rel_err(p, y) if kind == "rel" else np.abs(np.log(np.maximum(p, 1.0) / y))
    return float(np.median(r) + np.quantile(r, 0.9) + 0.1 * r.mean())


def _round(name: str, v: float) -> float:
    kind, lo, hi = CONV_SPACE[name]
    v = min(hi, max(lo, v))
    return int(round(float(v))) if kind == "int" else round(float(v), 2)


def _candidates(name: str, v: float, fine: bool) -> List[float]:
    kind, lo, hi = CONV_SPACE[name]
    facs = (0.93, 0.97, 1.03, 1.07) if fine else (0.5, 0.7, 0.85, 1.15, 1.4, 2.0)
    c = {v * f for f in facs}
    if kind == "int":
        c |= {v - 1, v + 1} if fine else {v - 2, v - 1, v + 1, v + 2, lo}
        if v == 0 and not fine:
            c |= {1, 2, 5, 10, 20, 50, 100}
    else:
        c |= {v - 0.01, v + 0.01} if fine else {v - 0.05, v + 0.05}
    out = sorted({_round(name, x) for x in c} - {_round(name, v)})
    return out


def fit_rtl_conv_board(camp: Campaign, ev: ConvEvaluator, host: float, start: dict,
                       mask: Optional[np.ndarray] = None, max_rounds: int = 12,
                       params: Iterable[str] = tuple(CONV_SPACE),
                       objective: str = "rel",
                       ship_weight: float = 0.0, dec: Optional["Decisions"] = None,
                       dec_weight: float = 0.0, em: Optional[np.ndarray] = None,
                       starts: Iterable[dict] = ()) -> Tuple[dict, List[str]]:
    """Coordinate search of RTL_CONV_BOARD from ``start`` (and from its
    clock-scaled copy and ``starts``): per parameter every candidate step at
    once, the best taken when it lowers the objective; coarse steps until a
    round changes nothing, then fine steps.  With ``dec`` (the shipped models'
    engine options) the objective adds ``dec_weight`` x Decisions.penalty
    (MatmulKernel's estimates ``em`` fixed)."""
    m = np.ones(len(camp.conv_y), bool) if mask is None else mask
    ship = m & np.array([camp.info.get(r["key"], {}).get("set") == "shipped" for r in camp.conv])
    if not ship.any():
        ship_weight = 0.0
    hist: List[str] = []

    def obj_many(ps):
        out = []
        pen = dec.penalty(ps, em, host) if dec is not None and dec_weight else None
        for j, pr in enumerate(ev.predict_many(ps)):
            o = _conv_objective(pr[m], camp.conv_y[m], host, objective)
            if ship_weight:
                o = ((1.0 - ship_weight) * o
                     + ship_weight * _conv_objective(pr[ship], camp.conv_y[ship], host, objective))
            if pen is not None:
                o += dec_weight * pen[j]
            out.append(o)
        return out

    scale = camp.mhz / FIT_MHZ
    scaled = {k: (_round(k, v * scale) if k in CONV_LATENCIES else v) for k, v in start.items()}
    cand = [start, scaled] + [dict(x) for x in starts]
    objs = obj_many(cand)
    i = int(np.argmin(objs))
    p, best = dict(cand[i]), objs[i]
    hist.append(f"start: current {objs[0]:.4f}, clock-scaled (x{scale:g} on the latencies) "
                f"{objs[1]:.4f}" + "".join(f", --start {o:.4f}" for o in objs[2:])
                + f" -> from {['the current', 'the scaled', 'a --start'][min(i, 2)]} set")
    log("    " + hist[-1])
    fine = False
    for rnd in range(max_rounds):
        t0 = time.time()
        changed = []
        for name in params:
            cands = _candidates(name, p[name], fine)
            ps = [{**p, name: v} for v in cands]
            objs = obj_many(ps)
            i = int(np.argmin(objs))
            if objs[i] < best - 1e-6:
                changed.append(f"{name} {p[name]} -> {cands[i]}")
                p, best = ps[i], objs[i]
        hist.append(f"round {rnd + 1} ({'fine' if fine else 'coarse'}, {time.time() - t0:.0f} s): "
                    f"objective {best:.4f}; " + (", ".join(changed) or "no change"))
        log("    " + hist[-1])
        if not changed:
            if fine:
                break
            fine = True
    return p, hist


# ─────────────────────────────────────────────────────────────────────────────
# fit
# ─────────────────────────────────────────────────────────────────────────────

class Decisions:
    """The shipped models' MatMul engine options — per distinct MatMul the
    cheapest conv plan and MatmulKernel in the layout it would run, as the
    unplanned lowering compares them (``matmul_lowering.lower_matmuls``), both
    priced by the performance model — read from the decision files of
    ``decisions`` / ``diff`` (``<model>.base.json``), so that a fit can count
    the engine choices it gets wrong.  A MatMul whose weight is pinned to an
    image only ConvKernel reads has no choice and is left out."""

    def __init__(self, files: Iterable[Path], pm: PerfModel, tie: float = 0.03):
        from src.matmul_gemv import GEMV_KWS
        opts: Dict[tuple, dict] = {}
        for path in files:
            d = json.loads(Path(path).read_text())
            for entry, e in d["entries"].items():
                for m in e["matmuls"]:
                    cp, pin = m.get("conv_plan"), m.get("kw_pin")
                    if not cp or "mm_cycles" not in m or (pin and pin not in GEMV_KWS):
                        continue
                    t = option_times(m, pm)
                    if t is None:
                        continue
                    key = (m["n"], m["k"], m["m"], m["batch"], m["outer"], pin, cp["kw"],
                           cp["out_w"], cp["conv_n"], cp["calls"], cp["conv_batch"])
                    o = opts.setdefault(key, {"m": m, "cp": cp, "pin": pin, "tc": t[0],
                                              "tm": t[1], "count": 0, "where": set()})
                    o["count"] += 1
                    o["where"].add(f"{d['model']}/{entry}")
        self.opts = list(opts.values())
        self.tie = tie
        self.tc = np.array([o["tc"] for o in self.opts])
        self.tm = np.array([o["tm"] for o in self.opts])
        self.best = np.minimum(self.tc, self.tm)
        self.count = np.array([o["count"] for o in self.opts], dtype=float)
        self.calls = np.array([o["cp"]["calls"] for o in self.opts], dtype=float)
        self.cb = np.array([o["cp"]["conv_batch"] for o in self.opts], dtype=float)
        self.trace = ConvTrace([(o["m"]["k"] // o["cp"]["kw"], o["cp"]["conv_n"], o["cp"]["out_h"],
                                 o["cp"]["kw"] * o["cp"]["out_w"], o["cp"]["out_h"],
                                 o["cp"]["out_w"], 1, o["cp"]["kw"], 1, o["cp"]["kw"], 1, 1, 0, 0,
                                 False) for o in self.opts]) if self.opts else None

    def mm_estimates(self, coef: dict, host: float) -> np.ndarray:
        """MatmulKernel's cycles per option as matmul_plan_cycles prices them
        (a pinned image width on the GEMV path, else the tiled path with B
        row-major — packed only when pinned to it), plus the call."""
        from src.matmul_gemv import gemv_shape_reason
        out = []
        for o in self.opts:
            m, pin = o["m"], o["pin"]
            n, k, mm, b = m["n"], m["k"], m["m"], m["batch"] * m["outer"]
            if pin and gemv_shape_reason(_MMShape(m), pin) is None:
                t = cm.rtl_matmul_terms(n, k, mm, b, False, max(1, pin))
            else:
                t = cm.rtl_matmul_terms(n, k, mm, b, pin == 0, 0)
            out.append(sum(coef[name] * v for name, v in t.items()) + host)
        return np.array(out)

    def choices(self, ps: List[dict], em: np.ndarray, host: float) -> np.ndarray:
        """ConvKernel (True) or MatmulKernel per option (rows) and board
        parameter set (columns): conv_batch_cycles + the call, per call,
        against LOWER_MARGIN x MatmulKernel's estimate."""
        from src.matmul_lowering import LOWER_MARGIN
        W = self.trace.run({k: [p[k] for p in ps] for k in ps[0]})
        one = np.array([p["ONE"] for p in ps])[None, :]
        ec = self.calls[:, None] * (W * self.cb[:, None] + one + host)
        return ec < LOWER_MARGIN * em[:, None]

    def penalty(self, ps: List[dict], em: np.ndarray, host: float) -> np.ndarray:
        """Per parameter set: the sum over the distinct options of the
        chosen engine's measured time over the faster one's, minus 1."""
        if not self.opts:
            return np.zeros(len(ps))
        conv = self.choices(ps, em, host)
        t = np.where(conv, self.tc[:, None], self.tm[:, None])
        return (t / self.best[:, None] - 1.0).sum(0)

    def quality(self, p: dict, em: np.ndarray, host: float) -> dict:
        """Wrong choices (the other engine more than ``tie`` faster) and the
        µs they lose, counting every MatMul (one run of each entry)."""
        if not self.opts:
            return {"options": 0, "matmuls": 0, "wrong": 0, "lost_us": 0.0, "wrong_list": []}
        conv = self.choices([p], em, host)[:, 0]
        t = np.where(conv, self.tc, self.tm)
        bad = t > self.best * (1.0 + self.tie)
        return {"options": len(self.opts), "matmuls": int(self.count.sum()),
                "wrong": int(self.count[bad].sum()),
                "lost_us": float(((t - self.best) * self.count)[bad].sum()),
                "wrong_list": [f"{sorted(o['where'])[0]} {o['m']['n']}x{o['m']['k']}x{o['m']['m']}"
                               f"{' b' + str(o['m']['batch']) if o['m']['batch'] > 1 else ''} "
                               f"x{o['count']}: {'ConvKernel' if c else 'MatmulKernel'} "
                               f"(measured {o['tc']:.1f} / {o['tm']:.1f} us)"
                               for o, c, b in zip(self.opts, conv, bad, strict=True) if b]}


def score(camp: Campaign, ev: ConvEvaluator, consts: dict, host: float) -> dict:
    return error_table(camp, ev.predict(consts["RTL_CONV_BOARD"]),
                       mm_predict(camp, consts["RTL_COEF"]), host)


def perf_check(bid: str, proposed: dict) -> dict:
    """Refit bitstream ``bid``'s performance model in memory with the
    proposed constants and compare every kernel family's held-out error
    with the stored model file."""
    from src import perf_fit
    cases = json.loads((MODELS_DIR / f"{bid}.cases.json").read_text())
    data = json.loads((MODELS_DIR / f"{bid}.calib.json").read_text())
    stored = json.loads((MODELS_DIR / f"{bid}.json").read_text())
    with patched(proposed):
        new = perf_fit.fit(cases, data, stored["clock_mhz"])
    out = {}
    for n, f in stored["families"].items():
        g = new["families"].get(n)
        if g is None:
            continue
        out[n] = {"features_stored": f["features"], "features_refit": g["features"],
                  "holdout_stored": f["holdout"], "holdout_refit": g["holdout"],
                  "train_stored": f["train"], "train_refit": g["train"]}
    return out


def cmd_fit(args) -> int:
    bid = args.bitstream or local_bitstream_id()
    if not bid:
        log("no bitstream id given and no local bitstream config")
        return 1
    t0 = time.time()
    camp = Campaign(bid)
    host = round(camp.floor)
    cur = current_constants()
    log(f"campaign {bid} @ {camp.mhz:g} MHz: {len(camp.conv)} ConvKernel and {len(camp.mm)} "
        f"MatmulKernel calls ({len(camp.geoms)} distinct conv walks)")
    log(f"call floor {camp.floor_us:.3f} us = {camp.floor:.0f} cycles ({camp.floor_key}); "
        f"current CALL_OVERHEAD {cur['CALL_OVERHEAD']}")
    t1 = time.time()
    ev = ConvEvaluator(camp)
    d0 = ev.trace.check(cur["RTL_CONV_BOARD"])
    log(f"conv trace: {ev.trace.maxsw} steps, matches rtl_conv_walk within {d0:.1e} "
        f"({time.time() - t1:.0f} s)")
    conv_cur = ev.predict(cur["RTL_CONV_BOARD"])
    mm_cur = mm_predict(camp, cur["RTL_COEF"])
    before_used = error_table(camp, conv_cur, mm_cur, cur["CALL_OVERHEAD"])
    before_kernel = error_table(camp, conv_cur, mm_cur, host)
    mask_conv = mask_mm = None
    if args.holdout:
        mask_conv = ~np.array([r["holdout"] for r in camp.conv])
        mask_mm = ~np.array([r["holdout"] for r in camp.mm])
    log("\nRTL_COEF: NNLS on the MatmulKernel calls (measured - CALL_OVERHEAD, weights 1/measured)")
    coef, coef_info = fit_rtl_coef(camp, host, mask_mm)
    lo = coef_info["loo"]
    log(f"  {coef}\n  leave-one-out: median {lo['median'] * 100:.2f} %, p90 "
        f"{lo['p90'] * 100:.2f} %, max {lo['max'] * 100:.1f} %")
    dec = em_new = None
    dq = {}
    if args.decisions:
        files = sorted(Path(args.decisions).glob("*.base.json"))
        dec = Decisions(files, PerfModel.load(MODELS_DIR / f"{bid}.json"))
        em_cur = dec.mm_estimates(cur["RTL_COEF"], cur["CALL_OVERHEAD"])
        em_new = dec.mm_estimates(coef, host)
        dq["current"] = dec.quality(cur["RTL_CONV_BOARD"], em_cur, cur["CALL_OVERHEAD"])
        log(f"\nengine choices of the shipped models ({len(files)} decision files): "
            f"{dq['current']['options']} distinct options, {dq['current']['matmuls']} MatMuls; "
            f"current constants: {dq['current']['wrong']} wrong, "
            f"{dq['current']['lost_us']:.0f} us lost")
    log(f"\nRTL_CONV_BOARD: coordinate search (median + p90 + 0.1 mean of the "
        f"{'relative error' if args.objective == 'rel' else '|log ratio|'}"
        + (f"; shipped calls weighted {args.shipped_weight:g}" if args.shipped_weight else "")
        + (f"; + {args.decision_weight:g} x wrong-choice penalty" if dec is not None else "") + ")")
    starts = [json.loads(Path(x).read_text()).get("proposed", {}).get("RTL_CONV_BOARD")
              or json.loads(Path(x).read_text()) for x in (args.start or [])]
    board, hist = fit_rtl_conv_board(camp, ev, host, cur["RTL_CONV_BOARD"], mask_conv,
                                     max_rounds=args.rounds, objective=args.objective,
                                     ship_weight=args.shipped_weight, dec=dec,
                                     dec_weight=args.decision_weight, em=em_new, starts=starts)
    if dec is not None:
        dq["proposed"] = dec.quality(board, em_new, host)
        for name in ("current", "proposed"):
            q = dq[name]
            log(f"  engine choices, {name}: {q['wrong']} wrong, {q['lost_us']:.0f} us lost"
                + "".join(f"\n    {w}" for w in q["wrong_list"]))
    d1 = ev.trace.check(board)
    log(f"  {board}\n  (the trace matches rtl_conv_walk at the result within {d1:.1e})")
    proposed = {"CALL_OVERHEAD": host, "RTL_COEF": coef, "RTL_CONV_BOARD": board}
    after = score(camp, ev, proposed, host)
    print_tables(f"\nrelative error, {bid} (bias: median log(predicted / measured))",
                 [(f"current +{cur['CALL_OVERHEAD']} (as used)", before_used),
                  (f"current +{host} (floor)", before_kernel),
                  (f"proposed +{host}", after)])
    holdout = None
    if args.holdout:
        hm_c = ~mask_conv
        hm_m = ~mask_mm
        hc = rel_err(ev.predict(board)[hm_c] + host, camp.conv_y[hm_c])
        hmm = rel_err(mm_predict(camp, coef)[hm_m] + host, camp.mm_y[hm_m])
        holdout = {"ConvKernel": err_stats(hc), "MatmulKernel": err_stats(hmm)}
        log(f"  held out: ConvKernel {hc.size} calls median {np.median(hc) * 100:.1f} % p90 "
            f"{np.quantile(hc, .9) * 100:.1f} %; MatmulKernel {hmm.size} median "
            f"{np.median(hmm) * 100:.2f} % p90 {np.quantile(hmm, .9) * 100:.2f} %")
    evals = {}
    for other in args.eval or []:
        oc = Campaign(other)
        oev = ConvEvaluator(oc)
        oh = round(oc.floor)
        t_cur = score(oc, oev, cur, cur["CALL_OVERHEAD"])
        t_new = score(oc, oev, proposed, host)
        print_tables(f"\nrelative error, {other} @ {oc.mhz:g} MHz (floor {oc.floor:.0f} cycles)",
                     [(f"current +{cur['CALL_OVERHEAD']} (as used)", t_cur),
                      (f"current +{oh} (its floor)", score(oc, oev, cur, oh)),
                      (f"proposed +{host}", t_new)])
        evals[other] = {"clock_mhz": oc.mhz, "floor_cycles": oc.floor,
                        "current_as_used": t_cur, "proposed": t_new}
    pc = None
    if args.perf_check:
        log(f"\nperformance model {bid} refitted with the proposed constants (held-out errors):")
        pc = perf_check(bid, proposed)
        for n, d in pc.items():
            hs, hr = d["holdout_stored"], d["holdout_refit"]
            if not hs.get("n"):
                continue
            log(f"  {n:9s} stored {hs['median_rel'] * 100:5.2f} / {hs['p90_rel'] * 100:5.2f} %  "
                f"refit {hr['median_rel'] * 100:5.2f} / {hr['p90_rel'] * 100:5.2f} %  (median / p90; "
                f"features {'rtl_*' if d['features_refit'][0].startswith('rtl_') else 'hls'}"
                f"{'' if d['features_stored'] == d['features_refit'] else ', changed'})")
    out = Path(args.out) if args.out else OUT_DIR / f"cost_model_{bid}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "bitstream": bid, "clock_mhz": camp.mhz, "date": time.strftime("%Y-%m-%d"),
        "calls": {"ConvKernel": len(camp.conv), "MatmulKernel": len(camp.mm)},
        "call_floor": {"key": camp.floor_key, "us": camp.floor_us, "cycles": camp.floor},
        "holdout_excluded": bool(args.holdout), "objective": args.objective,
        "shipped_weight": args.shipped_weight,
        "decisions": args.decisions, "decision_weight": args.decision_weight if args.decisions else None,
        "before": cur, "proposed": proposed,
        "errors": {"before_as_used": before_used, "before_floor": before_kernel,
                   "proposed": after},
        "rtl_coef_loo": coef_info["loo"], "conv_search": hist, "holdout": holdout,
        "engine_choices": dq,
        "eval": evals, "perf_check": pc,
        "not_fitted": {
            "GEMV_WORD_CYCLE / GEMV_JOB_OVERHEAD": "used by gemv_cycles only with "
            "kernels.matmul.impl == 'hls'; the RTL kernel's GEMV path is rtl_matmul_cycles (RTL_COEF)",
            "RTL_CONV_SIM": "the Verilator testbench's (ideal memory, kernel cycles): clock-independent",
            "MM_* / CONV_WEIGHT_REQ_CYCLES / CONV_PREFETCH_HIDE": "the HLS kernels' models"},
    }
    out.write_text(json.dumps(doc, indent=1) + "\n")
    log(f"\nproposed constants -> {out}  ({time.time() - t0:.0f} s)")
    log(json.dumps(proposed, indent=None))
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# diff: the shipped models' engine choices under two constant sets
# ─────────────────────────────────────────────────────────────────────────────

def _summ_plan(p) -> Optional[dict]:
    if p is None:
        return None
    return {"kw": p.kw, "out_w": p.out_w, "out_h": p.out_h, "conv_n": p.conv_n,
            "conv_batch": p.conv_batch, "calls": p.calls, "cycles": p.cycles,
            "board_cycles": p.board_cycles, "acc_limited": p.acc_limited}


@contextmanager
def recording():
    """Record the inputs of every unplanned engine decision while graphs
    are built: the cheapest conv plan and MatmulKernel's estimate of each
    MatMul (matmul_lowering), the GEMV pass's image choice (matmul_gemv),
    the fully-connected Conv estimates (fc_conv); attached to each
    OnnxGraph as ``_costfit``."""
    from src import fc_conv, matmul_gemv, matmul_lowering
    from src.graph import OnnxGraph
    stack: List[list] = []
    saved = (OnnxGraph.__init__, matmul_lowering.conv_plans, matmul_lowering.matmul_plan_cycles,
             matmul_gemv.image_choice, fc_conv.estimate)
    o_init, o_plans, o_mm, o_img, o_est = saved

    def init(self, *a, **kw):
        stack.append([])
        try:
            o_init(self, *a, **kw)
        finally:
            self._costfit = stack.pop()

    def rec(item):
        if stack:
            stack[-1].append(item)

    def plans(mm, kws, splits="auto"):
        r = o_plans(mm, kws, splits)
        if splits == "auto":
            rec(("conv", mm.onnx_node, list(kws), _summ_plan(r[0] if r else None), len(r)))
        return r

    def mmc(mm, kw_pin=None):
        r = o_mm(mm, kw_pin)
        rec(("mm", mm.onnx_node, kw_pin, float(r)))
        return r

    def img(sn, *a, **kw):
        r = o_img(sn, *a, **kw)
        rec(("gemv", sn.onnx_node, None if r is None else
             {"kw": r[0], "relayout": r[1], "use": bool(r[2]), "tiled": float(r[3]),
              "gemv": float(r[4])}))
        return r

    def est(geo):
        r = o_est(geo)
        rec(("fc", geo["y"], {k: geo[k] for k in ("n", "c", "h", "wd", "m")}, bool(geo["b"]), r))
        return r

    OnnxGraph.__init__ = init
    matmul_lowering.conv_plans = plans
    matmul_lowering.matmul_plan_cycles = mmc
    matmul_gemv.image_choice = img
    fc_conv.estimate = est
    try:
        yield
    finally:
        (OnnxGraph.__init__, matmul_lowering.conv_plans, matmul_lowering.matmul_plan_cycles,
         matmul_gemv.image_choice, fc_conv.estimate) = saved


def graph_decisions(g) -> dict:
    """The engine decisions of one built graph."""
    from src.nodes import MatmulConvNode, MatmulNode
    recs = getattr(g, "_costfit", [])
    by_node: Dict[int, dict] = {}
    for r in recs:
        if r[0] == "fc":
            continue
        d = by_node.setdefault(id(r[1]), {})
        if r[0] == "conv":
            d["kws"], d["conv_plan"], d["conv_plans"] = r[2], r[3], r[4]
        elif r[0] == "mm":
            d["kw_pin"], d["mm_cycles"] = r[2], r[3]
        else:
            d["gemv_choice"] = r[2]
    mms = []
    for sn in g.nodes:
        if not isinstance(sn, (MatmulNode, MatmulConvNode)):
            continue
        d = {"index": sn.index, "name": sn.onnx_node.name or f"MatMul_{sn.index}",
             "n": sn.n, "k": sn.k, "m": sn.m, "batch": sn.batch,
             "outer": getattr(sn, "outer_count", 1), "b_const": sn.inputs[1].data is not None,
             "a_stride": sn.a_batch_stride, "b_stride": sn.b_batch_stride,
             "c_stride": sn.c_batch_stride,
             **by_node.get(id(sn.onnx_node), {})}
        if isinstance(sn, MatmulConvNode):
            d.update(engine="conv", kw=sn.kw, out_w=sn.out_w, conv_n=sn.conv_n, calls=sn.calls,
                     conv_batch=sn.conv_batch)
        else:
            d.update(engine="mm", gemv_kw=sn.gemv_kw, b_packed=bool(sn.b_packed))
        d["kernel_calls"] = [[c.key(), c.count] for c in sn.kernel_calls({})]
        mms.append(d)
    fcs = [{"y": r[1], "geo": r[2], "bias": r[3], "est": r[4]} for r in recs if r[0] == "fc"]
    return {"matmuls": mms, "fc": fcs, "fc_stats": getattr(g, "fc_conv_stats", None),
            "matmul_conv_stats": getattr(g, "matmul_conv_stats", None),
            "gemv_stats": getattr(g, "matmul_gemv_stats", None)}


def cmd_decisions(args) -> int:
    """(internal, one subprocess per model and constant set) Build a shipped
    model's graphs and write its engine decisions."""
    consts = json.loads(Path(args.constants).read_text()) if args.constants else None
    import perf_calibrate as pc
    out = {"model": args.model, "constants": consts or current_constants(), "entries": {}}
    t0 = time.time()
    with patched(consts), recording():
        for entry, g in pc.shipped_graphs(args.model):
            out["entries"][entry] = graph_decisions(g)
            del g
    out["seconds"] = time.time() - t0
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    return 0


def _choice(d: dict) -> tuple:
    if d["engine"] == "conv":
        return ("conv", d["kw"], d["out_w"], d["conv_n"], d["calls"], d["conv_batch"])
    return ("mm", d["gemv_kw"], d["b_packed"])


def _label(d: dict) -> str:
    if d["engine"] == "conv":
        s = f"ConvKernel kw={d['kw']} out_w={d['out_w']}"
        if d["calls"] > 1:
            s += f" {d['calls']}x{d['conv_n']} rows"
        if d["conv_batch"] > 1:
            s += f" batch {d['conv_batch']}"
        return s
    if d["gemv_kw"]:
        return f"MatmulKernel GEMV kw={d['gemv_kw']}"
    return "MatmulKernel tiled" + (" packed" if d["b_packed"] else "")


def _price(pm: Optional[PerfModel], calls: List[list]) -> Optional[float]:
    if pm is None:
        return None
    return pm.calls_us([KernelCall.from_key(k, n) for k, n in calls])


def _fc_calls(fc: dict, lowered: bool) -> List[list]:
    g = fc["geo"]
    k = g["c"] * g["h"] * g["wd"]
    if not lowered:
        return [[KernelCall.of("ConvKernel", batch=g["n"], in_ch=g["c"], in_h=g["h"], in_w=g["wd"],
                               out_ch=g["m"], out_h=1, out_w=1, kh=g["h"], kw=g["wd"],
                               stride_h=1, stride_w=1, dilation_h=1, dilation_w=1,
                               has_bias=int(fc["bias"])).key(), 1]]
    gemv = fc["est"]["gemv"]
    out = [[KernelCall.of("MatmulKernel", n=1 if gemv else g["n"], k=k, m=g["m"], batch=1,
                          b_packed=0 if gemv else 1, gemv_kw=1 if gemv else 0).key(), 1]]
    if fc["bias"]:
        out.append([KernelCall.of("VectorOPKernel", op=0, size=g["n"] * g["m"], outer=1).key(), 1])
    return out


class _MMShape:
    """The MatMul attributes matmul_lowering's call builders read."""

    def __init__(self, d: dict):
        self.n, self.k, self.m, self.batch = d["n"], d["k"], d["m"], d["batch"]
        self.a_batch_stride, self.b_batch_stride = d["a_stride"], d["b_stride"]
        self.c_batch_stride = d["c_stride"]
        self.outer_count, self.b_packed = d["outer"], False


def option_times(d: dict, pm: Optional[PerfModel]) -> Optional[Tuple[float, float]]:
    """(µs on ConvKernel, µs on MatmulKernel) of the two options the lowering
    compared for MatMul ``d`` — its cheapest conv plan, and MatmulKernel in
    the layout it would run (a pinned image, else the tiled path, packed for
    a constant B) — priced by the performance model; None when either is
    missing or unpriced."""
    cp = d.get("conv_plan")
    if pm is None or not cp or "mm_cycles" not in d or "a_stride" not in d:
        return None
    from src.matmul_gemv import gemv_shape_reason
    from src.matmul_lowering import plan_mm_calls
    mm = _MMShape(d)
    conv = [KernelCall.of("ConvKernel", count=cp["calls"], batch=cp["conv_batch"],
                          in_ch=d["k"] // cp["kw"], in_h=cp["out_h"], in_w=cp["kw"] * cp["out_w"],
                          out_ch=cp["conv_n"], out_h=cp["out_h"], out_w=cp["out_w"], kh=1,
                          kw=cp["kw"], stride_h=1, stride_w=cp["kw"], dilation_h=1, dilation_w=1)]
    pin = d.get("kw_pin")
    layout = pin if pin and gemv_shape_reason(mm, pin) is None else 0
    tc, tm = pm.calls_us(conv), pm.calls_us(plan_mm_calls(mm, layout, d["b_const"]))
    return None if tc is None or tm is None else (tc, tm)


def decision_quality(dec: dict, pm: Optional[PerfModel], tie: float = 0.03) -> dict:
    """How the cost model's ConvKernel / MatmulKernel choices of one decision
    file compare with the performance model's prices of both options:
    MatMuls priced, wrong choices (the other option more than ``tie``
    faster), and the µs lost to them per run of each entry (summed)."""
    from src.matmul_gemv import GEMV_KWS
    out = {"priced": 0, "wrong": 0, "lost_us": 0.0, "total_us": 0.0, "wrong_list": []}
    for entry, e in dec["entries"].items():
        for d in e["matmuls"]:
            if d.get("kw_pin") and d["kw_pin"] not in GEMV_KWS:
                continue                 # pinned to a ConvKernel-only image: no choice
            t = option_times(d, pm)
            if t is None:
                continue
            tc, tm = t
            chose_conv = d["engine"] == "conv"
            mine, other = (tc, tm) if chose_conv else (tm, tc)
            out["priced"] += 1
            out["total_us"] += mine
            if other < mine * (1.0 - tie):
                out["wrong"] += 1
                out["lost_us"] += mine - other
                out["wrong_list"].append((entry, d["name"], "conv" if chose_conv else "mm",
                                          round(tc, 1), round(tm, 1)))
    return out


def diff_model(a: dict, b: dict, pm: Optional[PerfModel]) -> List[str]:
    """Lines describing every changed decision between two decision files."""
    from src.fc_conv import MIN_GAIN
    lines = []
    for entry, ea in a["entries"].items():
        eb = b["entries"].get(entry)
        if eb is None:
            lines.append(f"  {entry}: missing in the proposed run")
            continue
        mb = {d["name"]: d for d in eb["matmuls"]}
        n_changed, dt = 0, 0.0
        for d in ea["matmuls"]:
            e = mb.get(d["name"])
            if e is None or _choice(d) == _choice(e):
                continue
            n_changed += 1
            ta, tb = _price(pm, d["kernel_calls"]), _price(pm, e["kernel_calls"])
            if ta is not None and tb is not None:
                dt += tb - ta

            def est(x):
                cp = x.get("conv_plan")
                conv = f"{cp['board_cycles']:.0f}" if cp else "-"
                mm = f"{x['mm_cycles']:.0f}" if "mm_cycles" in x else "-"
                g = x.get("gemv_choice")
                gv = (f", tiled {g['tiled']:.0f} / gemv kw{g['kw']} {g['gemv']:.0f}" if g else "")
                return f"conv {conv} / mm {mm}{gv}"
            us = (f"; perf model {ta:.1f} -> {tb:.1f} us ({tb - ta:+.1f})"
                  if ta is not None and tb is not None else "; perf model: not priced")
            lines.append(f"  {entry} {d['name']} [{d['n']}x{d['k']}x{d['m']}"
                         f"{' b' + str(d['batch']) if d['batch'] > 1 else ''}]: {_label(d)} -> "
                         f"{_label(e)}  (cycles before: {est(d)}; after: {est(e)}{us})")
        fb = {f["y"]: f for f in eb["fc"]}
        for f in ea["fc"]:
            g = fb.get(f["y"])
            if g is None:
                continue
            la = f["est"]["matmul"] <= f["est"]["conv"] * (1.0 - MIN_GAIN)
            lb = g["est"]["matmul"] <= g["est"]["conv"] * (1.0 - MIN_GAIN)
            if la == lb:
                continue
            n_changed += 1
            ta, tb = _price(pm, _fc_calls(f, la)), _price(pm, _fc_calls(f, lb))
            us = (f"; perf model {ta:.1f} -> {tb:.1f} us" if ta is not None and tb is not None
                  else "")
            lines.append(f"  {entry} fc_conv {f['y']} {f['geo']}: "
                         f"{'MatMul' if la else 'Conv'} -> {'MatMul' if lb else 'Conv'} "
                         f"(conv {f['est']['conv']:.0f} / mm {f['est']['matmul']:.0f} -> "
                         f"conv {g['est']['conv']:.0f} / mm {g['est']['matmul']:.0f}){us}")
        if n_changed:
            lines.append(f"  {entry}: {n_changed} change(s), perf model total {dt:+.1f} us per run")
    return lines


def build_decisions(model: str, out: Path, constants: Optional[str] = None) -> bool:
    """One model's decision file, built in a subprocess (memory)."""
    cmd = [sys.executable, str(Path(__file__).resolve()), "_decisions", model, "--out", str(out)]
    if constants:
        cmd += ["--constants", constants]
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(HERE), capture_output=True, text=True)
    if r.returncode:
        log(f"{model}: FAILED\n{r.stdout[-2000:]}{r.stderr[-4000:]}")
        return False
    log(f"{model}: {out.name} built in {time.time() - t0:.0f} s")
    return True


def cmd_build(args) -> int:
    """The base decision files of the shipped models (the module's constants)."""
    import perf_calibrate as pc
    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR / "diff"
    out_dir.mkdir(parents=True, exist_ok=True)
    ok = True
    for model in args.models or pc.SHIPPED:
        f = out_dir / f"{model}.base.json"
        if f.exists() and not args.rebuild:
            continue
        ok &= build_decisions(model, f)
    return 0 if ok else 1


def cmd_diff(args) -> int:
    import perf_calibrate as pc
    prop = json.loads(Path(args.json).read_text())
    proposed = prop["proposed"]
    base = prop["before"] if args.base == "before" else None
    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR / "diff"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfiles = {}
    for name, c in (("base", base), ("proposed", proposed)):
        if c is not None:
            p = out_dir / f"constants_{name}.json"
            p.write_text(json.dumps(c, indent=1) + "\n")
            cfiles[name] = str(p)
    bid = prop.get("bitstream")
    mp = MODELS_DIR / f"{bid}.json"
    pm = PerfModel.load(mp) if mp.exists() else None
    total_lines = []
    for model in args.models or pc.SHIPPED:
        files = {}
        for side in ("base", "proposed"):
            f = out_dir / f"{model}.{side}.json"
            files[side] = f
            if f.exists() and not (args.rebuild_base if side == "base" else args.rebuild):
                continue
            if not build_decisions(model, f, cfiles.get(side)):
                files = None
                break
        if files is None:
            total_lines.append(f"{model}: build failed")
            continue
        a = json.loads(files["base"].read_text())
        b = json.loads(files["proposed"].read_text())
        lines = diff_model(a, b, pm)
        qa, qb = decision_quality(a, pm), decision_quality(b, pm)
        qual = (f"   [engine choices vs the performance model: {qa['priced']} MatMuls priced; "
                f"wrong {qa['wrong']} -> {qb['wrong']}, lost {qa['lost_us']:.0f} -> "
                f"{qb['lost_us']:.0f} us of {qa['total_us']:.0f} -> {qb['total_us']:.0f} us "
                f"(all entries, one run each)]")
        total_lines.append(f"{model}: " + ("no change" if not lines else f"{len(lines)} line(s)")
                           + (qual if qa["priced"] else ""))
        total_lines += lines
        log(total_lines[-1 - len(lines)])
        for ln in lines:
            log(ln)
    report = out_dir / "diff.txt"
    report.write_text("\n".join(total_lines) + "\n")
    log(f"\nreport -> {report}")
    return 0


# ─────────────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit", help="refit the board terms to a campaign")
    f.add_argument("bitstream", nargs="?", help="bitstream id (default: the local production one)")
    f.add_argument("--out", help=f"proposed constants JSON (default {OUT_DIR}/cost_model_<id>.json)")
    f.add_argument("--rounds", type=int, default=12, help="coordinate-search rounds (max)")
    f.add_argument("--shipped-weight", type=float, default=0.0, metavar="W",
                   help="RTL_CONV_BOARD's objective: (1 - W) x all ConvKernel calls + W x the "
                   "shipped models' calls and tactics (the cases' set 'shipped')")
    f.add_argument("--decisions", metavar="DIR",
                   help="the shipped models' base decision files (the 'decisions' command): add "
                   "a penalty for the engine choices the conv parameters get wrong")
    f.add_argument("--decision-weight", type=float, default=1.0, metavar="L",
                   help="--decisions: weight of the penalty (sum over the distinct options of "
                   "chosen / faster measured time - 1)")
    f.add_argument("--start", nargs="*", metavar="JSON",
                   help="further starting points of the search (fit outputs or board dicts)")
    f.add_argument("--objective", choices=("rel", "log"), default="rel",
                   help="RTL_CONV_BOARD's error measure: relative error or |log ratio|")
    f.add_argument("--eval", nargs="*", metavar="ID", help="also score on these campaigns")
    f.add_argument("--holdout", action="store_true",
                   help="fit without the campaign's held-out calls and report them")
    f.add_argument("--perf-check", action="store_true",
                   help="refit the performance model with the proposed constants (in memory)")
    d = sub.add_parser("diff", help="engine choices of the shipped models, current vs proposed")
    d.add_argument("json", help="the fit's output")
    d.add_argument("--models", nargs="*", help="shipped models (default perf_calibrate.SHIPPED)")
    d.add_argument("--base", choices=("module", "before"), default="module",
                   help="compare against the module's constants or the JSON's 'before'")
    d.add_argument("--out-dir", help=f"decision files and report (default {OUT_DIR}/diff)")
    d.add_argument("--rebuild", action="store_true", help="rebuild the proposed decision files")
    d.add_argument("--rebuild-base", action="store_true", help="rebuild the base decision files")
    b = sub.add_parser("decisions", help="build the shipped models' base decision files")
    b.add_argument("--models", nargs="*", help="shipped models (default perf_calibrate.SHIPPED)")
    b.add_argument("--out-dir", help=f"decision files (default {OUT_DIR}/diff)")
    b.add_argument("--rebuild", action="store_true", help="rebuild existing decision files")
    x = sub.add_parser("_decisions", help="(internal) one model's decision file")
    x.add_argument("model")
    x.add_argument("--constants")
    x.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    return {"fit": cmd_fit, "diff": cmd_diff, "decisions": cmd_build,
            "_decisions": cmd_decisions}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
