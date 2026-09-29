#!/usr/bin/env python3
"""onnx_study.py — the go / no-go numbers of an ONNX model on the KV260 stack
(model-study skill, route B: CNNs, BERT-like encoders, any graph the
inference scheduler takes as is).

  1. coverage   OnnxGraph with the CLI defaults (pattern fusion, activation
                fusion, stride-2 stem rewrite); an unsupported op or a shape
                beyond a kernel bound stops here with the scheduler's error.
                Otherwise: nodes per engine (Conv / Matmul / VectorOP / Pool
                kernels, host CPU, zero-cost views).
  2. memory     the DMA pool: weights + intermediates (slot reuse), against
                cma=1000M; host-memory tensors.
  3. numerics   the scheduler's bit-level simulation (CodeGenerator, what the
                generated C computes) against onnxruntime float32 on the
                SAME grid-quantised inputs: per output rel. L2 / cosine /
                top-1 agreement; per intermediate tensor the drift from float
                and the float value range against the tensor's representable
                range (saturated / below-resolution fractions) — where the
                datapath loses the model.
  4. latency    the timed event-stream replay (codegen/timing.py) priced by
                the bitstream's performance model (perf_models/kv260/) and the
                host-op model; the kernel time's error band and the unpriced
                nodes.

Inputs: --inputs X.npz ({graph input name: array of the input's shape, or
[N, *shape] for N samples}) — real data is the meaningful study; without it
seeded random data (normal * --input-std for float inputs, valid indices for
integer ones), which measures drift but not task accuracy.

usage: inference-scheduler/.venv/bin/python .claude/skills/model-study/scripts/onnx_study.py
           MODEL.onnx [--inputs X.npz] [--samples 2] [--seed 0] [--input-std 1.0]
           [--threshold 0.05] [--top 12] [--no-numerics] [--no-latency] [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]          # .claude/skills/model-study/scripts
SCHED = REPO / "inference-scheduler"
if str(SCHED) not in sys.path:
    sys.path.insert(0, str(SCHED))

CMA_MB = 1000.0            # cma=1000M on the board's kernel command line (idle CmaFree ~1011 MB)


def mib(n_bytes: float) -> float:
    return n_bytes / 2 ** 20


def size(m: float) -> str:
    """MiB as '12.3 MiB' / '45.6 KiB'."""
    return f"{m:.1f} MiB" if m >= 1 else f"{m * 1024:.1f} KiB"


def op_of(sn) -> str:
    op = getattr(getattr(sn, "onnx_node", None), "op_type", "")
    return op or type(sn).__name__


# ── 1. coverage ─────────────────────────────────────────────────────────────

def census(model: str) -> dict:
    """ONNX op counts of the whole graph, each marked supported, pattern-only
    (fine inside a fused LayerNorm / GELU, rejected alone) or missing — the
    scheduler itself stops at the first missing one."""
    import onnx
    from src.fusion import PATTERN_ONLY_OPS
    from src.graph import _ALL_SUPPORTED_OP_TYPES
    m = onnx.load(model, load_external_data=False)
    ops = Counter(f"{n.domain}:{n.op_type}" if n.domain not in ("", "ai.onnx") else n.op_type
                  for n in m.graph.node)
    status = {op: ("supported" if op in _ALL_SUPPORTED_OP_TYPES else
                   "pattern-only" if op in PATTERN_ONLY_OPS else "missing") for op in ops}
    return {"ops": dict(ops.most_common()), "status": status,
            "missing": {op: ops[op] for op in ops if status[op] == "missing"},
            "pattern_only": {op: ops[op] for op in ops if status[op] == "pattern-only"}}


def build(model: str):
    from src.graph import OnnxGraph
    from src.nodes import SchedulerError
    t0 = time.time()
    try:
        g = OnnxGraph(model, fuse_act=True, s2d_stem=True, fuse_patterns=True)
    except SchedulerError as e:
        return None, f"{type(e).__name__}: {e}", time.time() - t0
    return g, None, time.time() - t0


def partition(g) -> dict:
    lanes: Counter = Counter()
    kinds: dict = defaultdict(Counter)
    for sn in g.nodes:
        k = getattr(type(sn), "kernel_name", "")
        if type(sn).__name__ == "ReshapeNode" or getattr(sn, "is_view", False):
            lane = "zero-cost"
        else:
            lane = k or "host"
        lanes[lane] += 1
        kinds[lane][op_of(sn)] += 1
    return {"nodes": len(g.nodes), "lanes": dict(lanes),
            "kinds": {k: dict(v) for k, v in kinds.items()}}


# ── 2. memory ───────────────────────────────────────────────────────────────

def memory(g, cg) -> dict:
    bpe = cg._dtype.bytes_per_elem
    layout, total = cg._compute_pool_layout()
    weights = {t.onnx_name for t in g.weight_tensors}
    w = sum(a for n, _, a in layout if n in weights)
    host = sum(int(np.prod(t.shape or [1])) * 4 for t in g._tensors.values()
               if getattr(t, "is_host", False))
    return {"pool_mib": mib(total * bpe), "weights_mib": mib(w * bpe),
            "intermediates_mib": mib((total - w) * bpe), "host_tensors_mib": mib(host),
            "cma_share": total * bpe / (CMA_MB * 1e6)}


# ── 3. numerics ─────────────────────────────────────────────────────────────

def n_samples(a, shape):
    """Samples in array ``a`` for an input of ``shape``: 0 for one sample of
    exactly that shape, N for [N, *shape] or — a batch-1 input — N stacked
    along the batch axis ([N, *shape[1:]]); None if it fits neither."""
    shape = tuple(shape)
    if tuple(a.shape) == shape:
        return 0
    if tuple(a.shape[1:]) == shape or (shape and shape[0] == 1 and tuple(a.shape[1:]) == shape[1:]):
        return a.shape[0]
    return None


def make_inputs(g, cg, path, samples, seed, std) -> list:
    """[{name: float64 array on the datapath's grid}] per sample: every
    sample of the --inputs file (at most ``samples`` if given), else
    ``samples`` (default 2) random ones."""
    dt = cg._dtype
    rng = np.random.default_rng(seed)
    data = dict(np.load(path)) if path else {}
    counts = []
    for t in g.input_tensors:
        a = data.get(t.onnx_name)
        if a is None:
            continue
        k = n_samples(a, t.shape)
        if k is None:
            raise SystemExit(f"--inputs {t.onnx_name}: shape {a.shape}, the model wants "
                             f"{tuple(t.shape)} or [N, *shape]")
        counts.append(max(k, 1))
    n = min(counts) if counts else (samples or 2)
    if samples:
        n = min(n, samples)
    out = []
    for i in range(n):
        s = {}
        for t in g.input_tensors:
            a = data.get(t.onnx_name)
            if a is not None:
                x = np.asarray(a if n_samples(a, t.shape) == 0 else a[i], np.float64).reshape(t.shape)
            elif t.is_int:
                x = rng.integers(0, cg._int_fill_range(t), size=t.shape).astype(np.float64)
            else:
                x = rng.normal(0.0, std, size=t.shape)
            if t.is_int:
                s[t.onnx_name] = np.rint(x)
            elif getattr(t, "is_host", False):
                s[t.onnx_name] = x.astype(np.float32).astype(np.float64)
            elif t.exp is not None:
                s[t.onnx_name] = dt.quantize_exp(x, t.exp_full(dt.frac_bits))
            else:
                s[t.onnx_name] = dt.quantize(x)
        out.append(s)
    return out


def ort_session(model: str):
    import onnx
    import onnxruntime as ort
    m = onnx.load(model)
    m = onnx.shape_inference.infer_shapes(m)
    have = {o.name for o in m.graph.output}
    for vi in m.graph.value_info:
        if vi.type.tensor_type.elem_type == onnx.TensorProto.FLOAT and vi.name not in have:
            m.graph.output.append(vi)
            have.add(vi.name)
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    so.log_severity_level = 3
    sess = ort.InferenceSession(m.SerializeToString(), so, providers=["CPUExecutionProvider"])
    types = {i.name: i.type for i in sess.get_inputs()}
    return sess, types


def _feed(types, sample):
    np_of = {"tensor(float)": np.float32, "tensor(int64)": np.int64, "tensor(int32)": np.int32,
             "tensor(double)": np.float64, "tensor(float16)": np.float16}
    return {k: v.astype(np_of.get(types[k], np.float32)) for k, v in sample.items() if k in types}


def limits(t, dt):
    """(largest representable |x|, resolution) per element of tensor t,
    broadcast along its last axis for per-channel exponents."""
    bits = 8 * dt.bytes_per_elem
    f = np.asarray(t.exp_full(dt.frac_bits) if t.exp is not None else dt.frac_bits, np.float64)
    return 2.0 ** (bits - 1 - f), 2.0 ** (-f)


def numerics(g, cg, model, samples, threshold, top) -> dict:
    sess, types = ort_session(model)
    names = [o.name for o in sess.get_outputs()]
    dt = cg._dtype
    tensors = g._tensors
    order = [sn.output.onnx_name for sn in g.nodes]
    outs = [t.onnx_name for t in g.output_tensors]
    per_t = defaultdict(lambda: {"rel": [], "sat": [], "tiny": [], "maxabs": 0.0})
    per_o = defaultdict(lambda: {"rel": [], "cos": [], "top1": []})
    for s in samples:
        ref = dict(zip(names, sess.run(names, _feed(types, s)), strict=True))
        sim = cg._forward_pass(s)
        for name in order:
            t = tensors.get(name)
            if name not in ref or name not in sim or t is None or t.is_int:
                continue
            r = np.asarray(ref[name], np.float64)
            x = np.asarray(sim[name], np.float64)
            if r.size != x.size:
                continue
            x = x.reshape(r.shape)
            nr = float(np.linalg.norm(r))
            rel = float(np.linalg.norm(x - r)) / nr if nr > 0 else float(np.linalg.norm(x))
            d = per_t[name]
            d["rel"].append(rel)
            d["maxabs"] = max(d["maxabs"], float(np.abs(r).max()) if r.size else 0.0)
            if not getattr(t, "is_host", False):
                hi, lo = limits(t, dt)
                try:
                    a = np.abs(r).reshape(x.shape)
                    d["sat"].append(float(np.mean(a >= hi)))
                    nz = a > 0
                    d["tiny"].append(float(np.mean(a[nz] < np.broadcast_to(lo, a.shape)[nz] / 2))
                                     if nz.any() else 0.0)
                except ValueError:
                    pass
        for name in outs:
            if name not in ref or name not in sim:
                continue
            r = np.asarray(ref[name], np.float64)
            x = np.asarray(sim[name], np.float64).reshape(r.shape)
            nr, nx = float(np.linalg.norm(r)), float(np.linalg.norm(x))
            o = per_o[name]
            o["rel"].append(float(np.linalg.norm(x - r)) / nr if nr else float(nx))
            o["cos"].append(float(np.dot(x.ravel(), r.ravel()) / (nr * nx)) if nr and nx else 0.0)
            if r.ndim >= 1 and r.shape[-1] >= 2:
                a, b = r.reshape(-1, r.shape[-1]), x.reshape(-1, r.shape[-1])
                o["top1"].append(float(np.mean(a.argmax(-1) == b.argmax(-1))))
    rows = []
    for name in order:
        if name in per_t:
            d = per_t[name]
            sn = next((n for n in g.nodes if n.output.onnx_name == name), None)
            rows.append({"tensor": name, "node": op_of(sn) if sn else "?",
                         "rel_l2": max(d["rel"]), "max_abs_float": d["maxabs"],
                         "saturated": max(d["sat"]) if d["sat"] else 0.0,
                         "below_resolution": max(d["tiny"]) if d["tiny"] else 0.0})
    first = next((r for r in rows if r["rel_l2"] > threshold), None)
    return {
        "samples": len(samples), "compared_tensors": len(rows),
        "outputs": {k: {"rel_l2_mean": float(np.mean(v["rel"])), "rel_l2_max": float(np.max(v["rel"])),
                        "cosine_min": float(np.min(v["cos"])),
                        "top1_agreement": float(np.mean(v["top1"])) if v["top1"] else None}
                    for k, v in per_o.items()},
        "first_over_threshold": first,
        "worst": sorted(rows, key=lambda r: -r["rel_l2"])[:top],
        "saturating": [r for r in rows if r["saturated"] > 0][:top],
        "tensors": rows,
    }


# ── 4. latency ──────────────────────────────────────────────────────────────

def latency(g, cg) -> dict:
    from src.codegen.timing import kernel_duration_fn, simulate
    from src.host_model import HostModel, default_host_model_path
    from src.planning import PlanError, PlanOptions, resolve_perf_model
    try:
        pm = resolve_perf_model(PlanOptions())
    except PlanError as e:
        return {"error": str(e)}
    hm = HostModel.load(default_host_model_path())
    tl = simulate(cg, kernel_duration_fn(pm, cg._layouts), hm.us)
    exact = model = err_w = 0.0
    for sn in g.nodes:
        if not hasattr(sn, "kernel_calls"):
            continue
        calls = sn.kernel_calls(cg._layouts)
        b = pm.calls_band(calls)
        if b is None:
            continue
        kinds = [pm.predict(c) for c in calls]
        if all(k is not None and k[1] == "exact" for k in kinds):
            exact += b[0]
        else:
            model += b[0]
        err_w += b[0] * b[1]
    by_index = {sn.index: sn for sn in g.nodes}
    unpriced = [f"{op_of(by_index[i])} {by_index[i].onnx_node.name}" for i in tl.unpriced]
    ktot = exact + model
    return {"perf_model": str(pm.path), "total_ms": tl.total_us / 1e3, "cpu_ms": tl.cpu_us / 1e3,
            "lanes_ms": {k: v / 1e3 for k, v in sorted(tl.lane_us.items())},
            "cpu_waits_ms": tl.wait_us / 1e3,
            "kernel_ms_exact": exact / 1e3, "kernel_ms_modelled": model / 1e3,
            "kernel_error_band": (err_w / ktot) if ktot else 0.0,
            "unpriced": unpriced}


# ── report ──────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--inputs", default=None, help="npz: {input name: array} (or [N, *shape])")
    ap.add_argument("--samples", type=int, default=None,
                    help="samples to run (default: all of --inputs, else 2 random)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--input-std", type=float, default=1.0)
    ap.add_argument("--threshold", type=float, default=0.05,
                    help="rel. L2 drift that marks the first failing tensor (default 5 %%)")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--no-numerics", action="store_true")
    ap.add_argument("--no-latency", action="store_true")
    ap.add_argument("--json", default=None, help="write every figure here")
    args = ap.parse_args(argv)

    res: dict = {"model": args.model}
    print(f"== {args.model}")
    c = census(args.model)
    res["census"] = c
    print(f"ops        {sum(c['ops'].values())} ONNX nodes: "
          + ", ".join(f"{op} {n}" for op, n in c["ops"].items()))
    if c["missing"]:
        print("  MISSING (no kernel or host op): "
              + ", ".join(f"{op} x{n}" for op, n in c["missing"].items()))
    if c["pattern_only"]:
        print("  pattern-only (supported inside a fused LayerNorm / GELU only): "
              + ", ".join(f"{op} x{n}" for op, n in c["pattern_only"].items()))
    g, err, dt = build(args.model)
    if g is None:
        res["coverage"] = {"supported": False, "error": err}
        print(f"coverage   NOT SUPPORTED ({dt:.1f} s)\n  {err.splitlines()[0]}")
        if args.json:
            Path(args.json).write_text(json.dumps(res, indent=1))
        return 2
    from src.codegen import CodeGenerator
    cg = CodeGenerator(g, model_path=args.model)
    p = partition(g)
    res["coverage"] = {"supported": True, **p}
    print(f"coverage   supported: {p['nodes']} nodes ({dt:.1f} s)")
    for lane, n in sorted(p["lanes"].items(), key=lambda x: -x[1]):
        kinds = ", ".join(f"{k} {v}" for k, v in sorted(p["kinds"][lane].items(), key=lambda x: -x[1]))
        print(f"  {lane:<16} {n:>5}   {kinds}")

    m = memory(g, cg)
    res["memory"] = m
    print(f"memory     DMA pool {size(m['pool_mib'])} = weights {size(m['weights_mib'])} + "
          f"intermediates {size(m['intermediates_mib'])} ({100 * m['cma_share']:.1f} % of cma=1000M); "
          f"host tensors {size(m['host_tensors_mib'])}")

    if not args.no_numerics:
        t0 = time.time()
        samples = make_inputs(g, cg, args.inputs, args.samples, args.seed, args.input_std)
        n = numerics(g, cg, args.model, samples, args.threshold, args.top)
        res["numerics"] = n
        src = f"--inputs {args.inputs}" if args.inputs else f"random (std {args.input_std}, seed {args.seed})"
        print(f"numerics   {n['samples']} samples, {src}; {n['compared_tensors']} tensors compared "
              f"({time.time() - t0:.1f} s)")
        for k, o in n["outputs"].items():
            t1 = "" if o["top1_agreement"] is None else f"  top-1 agreement {100 * o['top1_agreement']:.1f} %"
            print(f"  output {k}: rel L2 {100 * o['rel_l2_mean']:.2f} % (max {100 * o['rel_l2_max']:.2f} %), "
                  f"cosine >= {o['cosine_min']:.5f}{t1}")
        f = n["first_over_threshold"]
        print(f"  first tensor over {100 * args.threshold:.0f} % drift: "
              + (f"{f['tensor']} ({f['node']}, {100 * f['rel_l2']:.1f} %)" if f else "none"))
        print(f"  worst drift:  {'tensor':<40} {'node':<16} {'rel L2':>8} {'max|x|':>9} {'sat':>7} {'<res':>7}")
        for r in n["worst"]:
            print(f"                {r['tensor'][:40]:<40} {r['node'][:16]:<16} {100 * r['rel_l2']:7.2f}% "
                  f"{r['max_abs_float']:9.3g} {100 * r['saturated']:6.2f}% {100 * r['below_resolution']:6.1f}%")
        if n["saturating"]:
            print("  saturating (float value beyond the tensor's range): "
                  + ", ".join(f"{r['tensor']} {100 * r['saturated']:.2f} %" for r in n["saturating"]))
        else:
            print("  saturating: none")

    if not args.no_latency:
        lat = latency(g, cg)
        res["latency"] = lat
        if "error" in lat:
            print(f"latency    no performance model: {lat['error']}")
        else:
            lanes = ", ".join(f"{k} {v:.2f}" for k, v in lat["lanes_ms"].items())
            print(f"latency    {lat['total_ms']:.2f} ms predicted (kernels busy: {lanes} ms; "
                  f"CPU {lat['cpu_ms']:.2f} ms, waiting {lat['cpu_waits_ms']:.2f} ms)")
            print(f"  kernel time: {lat['kernel_ms_exact']:.2f} ms from measured calls, "
                  f"{lat['kernel_ms_modelled']:.2f} ms from family models "
                  f"(band ±{100 * lat['kernel_error_band']:.0f} %)")
            if lat["unpriced"]:
                print(f"  UNPRICED ({len(lat['unpriced'])}, counted as 0): "
                      + ", ".join(lat["unpriced"][:8]) + (" ..." if len(lat["unpriced"]) > 8 else ""))
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1, default=float))
        print(f"json       {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
