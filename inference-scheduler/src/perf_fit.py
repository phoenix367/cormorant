"""
Fitting the performance models (``perf_calibrate.py fit``, doc/plans/
TACTICS_PLAN.md §4.2 / §4.3): from the measurements of a calibration
campaign (``<id>.calib.json``) to the model file (``<id>.json``).

Per family (perf_model.family) a non-negative least-squares fit of the
measured µs on the family's features, weighted by 1 / µs so the fit
minimises relative error; the held-out quarter of the space-filling set is
never fitted on and gives the validation.  The exact entries are the mean
of the two passes of every measured call.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from .perf_calls import KernelCall
from .perf_model import family, feature_vector, features

TARGET_KERNEL = 0.03          # held-out relative error, kernels (TACTICS_PLAN §4.3)
NOISY = 0.005                 # pass-to-pass spread above which a point is "noisy"


def nnls(A: np.ndarray, b: np.ndarray, iters: int = 500) -> np.ndarray:
    """Lawson-Hanson non-negative least squares: argmin |Ax - b|, x >= 0."""
    m, n = A.shape
    x = np.zeros(n)
    P = np.zeros(n, dtype=bool)
    w = A.T @ (b - A @ x)
    for _ in range(iters):
        if P.all() or w[~P].max(initial=-np.inf) <= 1e-12 * max(1.0, np.abs(w).max()):
            break
        j = int(np.argmax(np.where(P, -np.inf, w)))
        P[j] = True
        while True:
            z = np.zeros(n)
            z[P] = np.linalg.lstsq(A[:, P], b, rcond=None)[0]
            if (z[P] > 0).all():
                x = z
                break
            neg = P & (z <= 0)
            alpha = np.min(x[neg] / (x[neg] - z[neg]))
            x = x + alpha * (z - x)
            P &= x > 1e-15
            x[~P] = 0.0
        w = A.T @ (b - A @ x)
    return x


def _measured(data: dict) -> Dict[str, dict]:
    """{key: {"us", "spread", "passes"}} from both passes."""
    p1 = data["passes"].get("1", {})
    p2 = data["passes"].get("2", {})
    out = {}
    for k, r in p1.items():
        if not r.get("ok"):
            continue
        vals = [r["mean_us"]]
        r2 = p2.get(k)
        if r2 and r2.get("ok"):
            vals.append(r2["mean_us"])
        mean = float(np.mean(vals))
        out[k] = {"us": mean, "spread": (max(vals) - min(vals)) / mean if len(vals) > 1 else None,
                  "passes": len(vals), "sd_us": r.get("sd_us")}
    return out


KNN_KS = (1, 3, 5, 8)


def _knn_correct(Z: np.ndarray, r: np.ndarray, zq: np.ndarray, k: int,
                 exclude: int = -1) -> float:
    """Distance-weighted mean of the k nearest residuals (log ratios)."""
    d = np.sqrt(((Z - zq) ** 2).sum(1))
    if exclude >= 0:
        d[exclude] = np.inf
    nn = np.argsort(d)[:k]
    w = 1.0 / (d[nn] + 1e-3)
    return float((r[nn] * w).sum() / w.sum())


def fit_family(rows: List[dict]) -> dict:
    """rows: {"call", "us", "holdout"}; returns the family's model entry:
    a non-negative linear fit (the base) times exp of the distance-weighted
    mean log residual of the k nearest training calls in the standardised
    log-feature space (the board's timing has local structure no global
    formula of the analytic terms captures; k by leave-one-out on the
    training calls only)."""
    names = sorted({k for r in rows for k in features(r["call"])}, key=lambda n: (n == "one", n))
    X = np.array([feature_vector(r["call"], names) for r in rows])
    y = np.array([r["us"] for r in rows])
    train = np.array([not r["holdout"] for r in rows])
    w = 1.0 / y
    coef = nnls(X[train] * w[train, None], np.ones(int(train.sum())))
    base = np.maximum(X @ coef, 1e-3)
    L = np.log1p(np.maximum(X, 0.0))
    mu, sd = L[train].mean(0), L[train].std(0) + 1e-9
    Z = (L - mu) / sd
    Zt, rt = Z[train], np.log(y[train] / base[train])
    tidx = np.where(train)[0]
    best_k, best_err = 0, float(np.median(np.abs(base[train] - y[train]) / y[train]))
    for k in KNN_KS:
        if k >= len(tidx):
            continue
        loo = np.array([base[i] * np.exp(_knn_correct(Zt, rt, Z[i], k, exclude=j))
                        for j, i in enumerate(tidx)])
        err = float(np.median(np.abs(loo - y[train]) / y[train]))
        if err < best_err:
            best_k, best_err = k, err
    pred = base.copy()
    if best_k:
        for i in range(len(y)):
            ex = int(np.where(tidx == i)[0][0]) if train[i] else -1
            pred[i] = base[i] * np.exp(_knn_correct(Zt, rt, Z[i], best_k, exclude=ex))
    rel = np.abs(pred - y) / y

    def stats(mask):
        if not mask.any():
            return {"n": 0}
        r = rel[mask]
        return {"n": int(mask.sum()), "median_rel": float(np.median(r)),
                "p90_rel": float(np.quantile(r, 0.9)), "max_rel": float(r.max())}
    fields = rows[0]["call"].fields.keys()
    rng = {f: [int(min(r["call"].fields[f] for r in rows)), int(max(r["call"].fields[f] for r in rows))]
           for f in fields}
    worst = sorted(range(len(rows)), key=lambda i: -rel[i])[:5]
    knn = {"k": best_k, "mu": [float(v) for v in mu], "sd": [float(v) for v in sd],
           "z": [[round(float(v), 4) for v in row] for row in Zt] if best_k else [],
           "r": [round(float(v), 5) for v in rt] if best_k else []}
    return {"kernel": rows[0]["call"].kernel, "features": names, "coef": [float(c) for c in coef],
            "knn": knn, "range": rng, "train": stats(train), "holdout": stats(~train),
            "worst": [{"key": rows[i]["call"].key(), "us": float(y[i]), "pred": float(pred[i]),
                       "holdout": bool(rows[i]["holdout"])} for i in worst]}


def fit(cases: dict, data: dict) -> dict:
    meas = _measured(data)
    info = {e["key"]: e for e in cases["cases"]}
    fams: Dict[str, List[dict]] = {}
    for k, m in meas.items():
        e = info.get(k)
        c = KernelCall.from_key(k)
        fams.setdefault(family(c), []).append(
            {"call": c, "us": m["us"], "holdout": bool(e and e.get("holdout"))})
    families = {n: fit_family(rows) for n, rows in sorted(fams.items()) if len(rows) >= 6}
    spreads = [m["spread"] for m in meas.values() if m["spread"] is not None]
    return {
        "platform": data["platform"], "bitstream": data["bitstream"], "clock_mhz": 100,
        "axi_bus_width": 128, "date": time.strftime("%Y-%m-%d"),
        "calibration_date": data.get("date"), "runner": data.get("runner"),
        "exact": {k: {"us": round(m["us"], 3)} for k, m in sorted(meas.items())},
        "families": families,
        "determinism": {"points": len(spreads),
                        "median_spread": float(np.median(spreads)) if spreads else None,
                        "max_spread": float(max(spreads)) if spreads else None,
                        "noisy": sum(s > NOISY for s in spreads)},
        "call_overhead_us": float(min((m["us"] for m in meas.values()), default=0.0)),
    }


def report(model: dict) -> str:
    lines = [f"performance model {model['platform']} / {model['bitstream']}: "
             f"{len(model['exact'])} exact calls"]
    d = model["determinism"]
    if d["points"]:
        lines.append(f"determinism: {d['points']} calls measured twice, median spread "
                     f"{d['median_spread'] * 100:.3f} %, max {d['max_spread'] * 100:.2f} %, "
                     f"{d['noisy']} above {NOISY * 100:.1f} %")
    lines.append(f"{'family':10s} {'k':>2s} {'n':>4s} {'train med':>9s} {'max':>7s} | {'held-out n':>10s} "
                 f"{'med':>7s} {'p90':>7s} {'max':>7s}  (train: leave-one-out)")
    for n, f in model["families"].items():
        t, h = f["train"], f["holdout"]
        hs = (f"{h['n']:10d} {h['median_rel'] * 100:6.2f}% {h['p90_rel'] * 100:6.2f}% "
              f"{h['max_rel'] * 100:6.2f}%") if h["n"] else f"{0:10d}"
        lines.append(f"{n:10s} {f['knn']['k']:2d} {t['n']:4d} {t['median_rel'] * 100:8.2f}% "
                     f"{t['max_rel'] * 100:6.2f}% | {hs}")
    return "\n".join(lines)


def cmd_fit(models_dir: Path, bid: str) -> int:
    cases = json.loads((models_dir / f"{bid}.cases.json").read_text())
    data = json.loads((models_dir / f"{bid}.calib.json").read_text())
    model = fit(cases, data)
    out = models_dir / f"{bid}.json"
    out.write_text(json.dumps(model, indent=1) + "\n")
    print(report(model))
    print(f"model -> {out}")
    bad = [n for n, f in model["families"].items()
           if f["holdout"].get("n") and f["holdout"]["p90_rel"] > TARGET_KERNEL]
    if bad:
        print(f"note: held-out p90 above {TARGET_KERNEL * 100:.0f} % in {bad}")
    return 0


__all__ = ("nnls", "fit", "fit_family", "report", "cmd_fit")
