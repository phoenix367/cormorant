#!/usr/bin/env python3
"""
perf_calibrate.py — the calibration campaign of the performance models
(doc/plans/TACTICS_PLAN.md §4.3).  Once per bitstream: measure kernel calls
on the board, fit the models, write perf_models/<platform>/<bitstream-id>.json.

  cases   the case list: every kernel call the shipped models issue or any
          of their MatMul tactics would issue (exact entries), plus a
          space-filling set per kernel from synthesised ONNX graphs (legal by
          construction; a quarter held out for validation)
  run     measure the cases on the board, twice (a second pass in another
          order and with other buffer contents: the determinism check);
          the chat server must be stopped (--stop-server does it and
          restarts it afterwards)
  fit     fit the kernel models and write the model file + validation report
  host    fit the host-op model (perf_models/<platform>/host.json) from board
          profiles: llm_board.py --profile --out (chat models), tts_board.py
          --profile --out (piper-lessac-medium), a demo's results.json
          (--profile-layers); --merge keeps every op kind fitted before
  simulate  predicted vs measured totals per phase (prefill, decode,
          llm_image, inference, piper's chunk and encode buckets)

The shipped models (SHIPPED) are the demos' CNNs, BERT, SmolLM2-135M /
360M, SmolVLM-256M and Piper (the chunk and encode_<T> entries of
libpiper_tts.so, built by demo/tts/scripts/generate_tts_project.py).

usage:
  .venv/bin/python perf_calibrate.py cases [--models ...]
  .venv/bin/python perf_calibrate.py run --config remote_config.json [--stop-server]
  .venv/bin/python perf_calibrate.py fit
  .venv/bin/python perf_calibrate.py all --config remote_config.json --stop-server
  .venv/bin/python perf_calibrate.py host --profile MODEL=RESULTS.json ... [--merge]
  .venv/bin/python perf_calibrate.py simulate --profile MODEL=RESULTS.json ...

Files (perf_models/<platform>/, versioned): <id>.cases.json (the case
list), <id>.calib.json (the measurements), <id>.json (the model).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

from src.perf_calls import (KernelCall, local_bitstream_id, local_board_bin,  # noqa: E402
                            local_kernel_clock_mhz)

PLATFORM = "kv260"
MODELS_DIR = HERE / "perf_models" / PLATFORM
CHAT = REPO / "demo" / "chat"
TTS = REPO / "demo" / "tts"
RUNNER_VERSION = 1
KLETTER = {"VectorOPKernel": "V", "MatmulKernel": "M", "ConvKernel": "C", "PoolKernel": "P"}
# the kernels' clock of the local bitstream (its HWH): turns the cost model's
# cycles into the per-call time estimates that size iterations and timeouts
MHZ = local_kernel_clock_mhz() or 100.0

SHIPPED = ("mnist_convnet", "mnist_lenet", "mobilenet_v1", "mobilenet_v2", "resnet18", "bert",
           "smollm2-135m-instruct", "smolvlm-256m-instruct", "smollm2-360m-instruct",
           "piper-lessac-medium")


def log(msg: str) -> None:
    print(msg, flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Shipped models
# ─────────────────────────────────────────────────────────────────────────────

def _cnn_path(demo: str, name: str) -> Path:
    cfg = json.loads((REPO / "demo" / demo / f"{demo}_config.json").read_text())
    for m in cfg["models"]:
        if m["name"] == name:
            return REPO / "demo" / demo / "assets" / "models" / m["drive_filename"]
    raise KeyError(f"{name} not in demo/{demo}")


def _model_path(name: str) -> str:
    if name.startswith("mnist_"):
        return str(_cnn_path("mnist", name))
    if name == "bert":
        return str(REPO / "demo/bert_squad/assets/models/bertsquad-12-simplified.onnx")
    return str(_cnn_path("image_classification", name))


def piper_graphs() -> List[Tuple[str, object]]:
    """Piper's entries (chunk, encode_<T>) exactly as libpiper_tts.so is generated
    (demo/tts/scripts/generate_tts_project.py: the encoder buckets share the
    largest bucket's kernel widths)."""
    if str(TTS / "scripts") not in sys.path:
        sys.path.insert(0, str(TTS / "scripts"))
    import generate_tts_project as gtp
    return gtp.entry_graphs(gtp.DEFAULT_ASSETS)


def shipped_graphs(name: str) -> Iterator[Tuple[str, object]]:
    """``(entry, OnnxGraph)`` of a shipped model, scheduled as its demo does."""
    from src.graph import OnnxGraph
    if name.startswith("mnist_"):
        yield name, OnnxGraph(str(_cnn_path("mnist", name)), fuse_act=True, s2d_stem=True)
    elif name in ("mobilenet_v1", "mobilenet_v2", "resnet18"):
        yield name, OnnxGraph(str(_cnn_path("image_classification", name)), fuse_act=True,
                              s2d_stem=True)
    elif name == "bert":
        yield name, OnnxGraph(str(REPO / "demo/bert_squad/assets/models/bertsquad-12-simplified.onnx"),
                              fuse_act=True, s2d_stem=True, fuse_patterns=True)
    elif name.startswith("smollm2-") or name.startswith("smolvlm-"):
        sys.path.insert(0, str(CHAT / "scripts"))
        from src.llm_entries import entry_graphs
        assets = str(CHAT / "assets" / name)
        if name.startswith("smolvlm-"):
            import vlm_project as vp
            m = vp.load(assets)
            fe_t, fe_v = vp.frontends(m)
            models = vp.entry_models(fe_t, fe_v)
        else:
            import llm_project as lp
            cfg, W, fmt, _fd = lp.load_model(assets)
            models = lp.entry_models(lp.frontend(cfg, W, fmt))
            del W
        for e, g in entry_graphs(models):
            yield e, g
    elif name.startswith("piper-"):
        yield from piper_graphs()
    else:
        raise KeyError(f"unknown shipped model {name!r} (choose from {SHIPPED})")


def graph_calls(g, top_conv_per_rows: int = 3) -> Dict[str, Tuple[KernelCall, str]]:
    """``{key: (call, node label)}`` of one graph: every kernel node's own
    calls (a runtime-keys attention call at every key count), and every
    MatMul tactic's calls (conv plans: the model's top few per row count)."""
    from src.codegen import CodeGenerator
    from src.llm_nodes import LlmAttnConvNode
    from src.tactics import graph_matmul_tactics
    cg = CodeGenerator(g, model_path="calib.onnx")
    lay = cg._layouts
    out: Dict[str, Tuple[KernelCall, str]] = {}

    def add(calls, label):
        for c in calls:
            c = KernelCall(c.kernel, c.regs, 1)
            out.setdefault(c.key(), (c, label))

    for sn in g.nodes:
        if not (getattr(type(sn), "kernel_name", "") and hasattr(sn, "kernel_calls")):
            continue
        label = sn.onnx_node.name or f"{sn.onnx_node.op_type}_{sn.index}"
        if isinstance(sn, LlmAttnConvNode) and not sn.static:
            for keys in range(sn.Q, sn.C + 1, sn.Q):
                add(sn.kernel_calls(lay, keys=keys), f"{label}@{keys}")
        else:
            add(sn.kernel_calls(lay), label)
    for sn, ts in graph_matmul_tactics(g, layouts=lay):
        label = sn.onnx_node.name or f"MatMul_{sn.index}"
        per_rows: Dict[int, int] = {}
        for t in ts:
            if t.kind == "conv":
                r = t.p["rows"]
                if per_rows.get(r, 0) >= top_conv_per_rows:
                    continue
                per_rows[r] = per_rows.get(r, 0) + 1
            add(t.calls, f"{label}:{t.label()}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# The space-filling set: synthesised graphs, legal by construction
# ─────────────────────────────────────────────────────────────────────────────

def _model(nodes, inputs, outputs, inits):
    from onnx import TensorProto
    from onnx import helper as oh
    from onnx import numpy_helper as nph
    vi = lambda n, s: oh.make_tensor_value_info(n, TensorProto.FLOAT, s)  # noqa: E731
    g = oh.make_graph(nodes, "calib", [vi(n, s) for n, s in inputs], [vi(n, s) for n, s in outputs],
                      initializer=[nph.from_array(np.asarray(v, np.float32), name=k)
                                   for k, v in inits.items()])
    m = oh.make_model(g, opset_imports=[oh.make_opsetid("", 13)])
    m.ir_version = 8
    return m


def _calls_of(model, **graph_kw) -> List[KernelCall]:
    from src.codegen import CodeGenerator
    from src.graph import OnnxGraph
    from src.nodes import SchedulerError
    try:
        g = OnnxGraph(model, fuse_act=True, **graph_kw)
        cg = CodeGenerator(g, model_path="calib.onnx")
    except (SchedulerError, ValueError, NotImplementedError):
        return []
    out = []
    for sn in g.nodes:
        if getattr(type(sn), "kernel_name", "") and hasattr(sn, "kernel_calls"):
            out += [KernelCall(c.kernel, c.regs, 1) for c in sn.kernel_calls(cg._layouts)]
    return out


def _choice(rng, xs):
    return xs[int(rng.integers(len(xs)))]


def grid_calls(seed: int = 2026) -> List[Tuple[KernelCall, str]]:
    """The space-filling set: ``[(call, family)]``."""
    from onnx import helper as oh
    from src.tactics import matmul_tactics, _MM
    rng = np.random.default_rng(seed)
    out: List[Tuple[KernelCall, str]] = []
    seen = set()

    def add(calls, fam):
        for c in calls:
            if c.key() not in seen:
                seen.add(c.key())
                out.append((c, fam))

    def loguni(lo, hi, mult):
        v = math.exp(rng.uniform(math.log(lo), math.log(hi)))
        return max(mult, int(round(v / mult)) * mult)

    # MatMul on ConvKernel: random MatMuls, random plans (not only the model's best)
    for _ in range(90):
        n, k, m = loguni(16, 1024, 16), loguni(64, 4096, 64), loguni(64, 4096, 8)

        class _B:
            onnx_name, is_int, data = "B", False, None
        mm = _MM(n, k, m, 1, 0, 0, 0, [_B(), _B()])
        ts = [t for t in matmul_tactics(mm, relayout_ok=True, b_constant=True) if t.kind == "conv"]
        if not ts:
            continue
        for i in rng.choice(len(ts), size=min(3, len(ts)), replace=False):
            add(ts[int(i)].calls, "conv-mm")
    # standard convolutions
    for _ in range(110):
        k = _choice(rng, (1, 1, 3, 3, 3, 5, 7))
        s = _choice(rng, (1, 1, 2))
        dw = rng.random() < 0.25
        hw = _choice(rng, (7, 14, 28, 56, 112))
        cin = _choice(rng, (3, 16, 32, 64, 96, 128, 192, 256, 384, 512)) if not dw else \
            _choice(rng, (16, 32, 64, 128, 256, 512))
        cout = cin if dw else _choice(rng, (16, 32, 64, 96, 128, 192, 256, 384, 512, 1024))
        pad = k // 2
        w = np.zeros((cout, 1 if dw else cin, k, k), np.float32)
        oh_ = (hw + 2 * pad - k) // s + 1
        node = oh.make_node("Conv", ["X", "W"] + (["Bb"] if rng.random() < 0.5 else []), ["Y"],
                            kernel_shape=[k, k], strides=[s, s], pads=[pad] * 4,
                            group=cin if dw else 1)
        inits = {"W": w}
        if len(node.input) == 3:
            inits["Bb"] = np.zeros(cout, np.float32)
        add(_calls_of(_model([node], [("X", [1, cin, hw, hw])], [("Y", [1, cout, oh_, oh_])], inits),
                      s2d_stem=False), "conv")
    # MatmulKernel tiled (constant and activation B, batched)
    for _ in range(70):
        n, k, m = loguni(1, 512, 1), loguni(16, 4096, 16), loguni(8, 3072, 8)
        b = _choice(rng, (1, 1, 1, 4, 12))
        const = rng.random() < 0.7
        a_shape = [b, n, k] if b > 1 else [n, k]
        b_shape = [k, m] if const else ([b, k, m] if b > 1 else [k, m])
        y_shape = [b, n, m] if b > 1 else [n, m]
        node = oh.make_node("MatMul", ["A", "B"], ["Y"])
        ins = [("A", a_shape)] + ([] if const else [("B", b_shape)])
        inits = {"B": np.zeros(b_shape, np.float32)} if const else {}
        add(_calls_of(_model([node], ins, [("Y", y_shape)], inits),
                      matmul_on_conv="off", matmul_gemv="off"), "mm-tiled")
    # MatmulKernel GEMV (kernel widths 1 / 2 / 4 / 8)
    from src.matmul_gemv import gemv_shape_reason
    for _ in range(60):
        k, m = loguni(64, 4096, 64), loguni(64, 4096, 8)
        kw = _choice(rng, (1, 2, 4, 8))
        mm = _MM(1, k, m, 1, 0, 0, 0, [])
        if gemv_shape_reason(mm, kw) is None:
            add([KernelCall.of("MatmulKernel", n=1, k=k, m=m, batch=1, gemv_kw=kw)], "mm-gemv")
    # VectorOP: elementwise and broadcast, every op, fused activations
    for _ in range(70):
        op = _choice(rng, ("Add", "Sub", "Mul", "Div", "Relu", "Add+Relu"))
        c, h, w = _choice(rng, (8, 16, 64, 128, 256, 512)), _choice(rng, (1, 7, 14, 28, 56)), \
            _choice(rng, (7, 14, 28, 56, 64, 768))
        shape = [1, c, h, w]
        bshape = _choice(rng, (shape, [1, c, 1, 1], [1, 1, 1, w]))
        if op == "Relu":
            nodes, ins = [oh.make_node("Relu", ["A"], ["Y"])], [("A", shape)]
        elif op == "Add+Relu":
            nodes = [oh.make_node("Add", ["A", "B"], ["T"]), oh.make_node("Relu", ["T"], ["Y"])]
            ins = [("A", shape), ("B", bshape)]
        else:
            nodes, ins = [oh.make_node(op, ["A", "B"], ["Y"])], [("A", shape), ("B", bshape)]
        add(_calls_of(_model(nodes, ins, [("Y", shape)], {})), "vecop")
    # Pooling
    for _ in range(50):
        kind = _choice(rng, ("MaxPool", "AveragePool", "GlobalAveragePool", "GlobalMaxPool"))
        c, hw = _choice(rng, (16, 32, 64, 128, 256, 512, 1024)), _choice(rng, (7, 14, 28, 56, 112))
        if kind.startswith("Global"):
            node, ohw = oh.make_node(kind, ["X"], ["Y"]), 1
        else:
            k, s = _choice(rng, ((2, 2), (3, 2), (3, 1))), None
            k, s = k
            pad = 1 if (k == 3 and s == 2 and rng.random() < 0.5) else 0
            node = oh.make_node(kind, ["X"], ["Y"], kernel_shape=[k, k], strides=[s, s],
                                pads=[pad] * 4)
            ohw = (hw + 2 * pad - k) // s + 1
        add(_calls_of(_model([node], [("X", [1, c, hw, hw])], [("Y", [1, c, ohw, ohw])], {})),
            "pool")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Buffers, estimates, case lines
# ─────────────────────────────────────────────────────────────────────────────

def _up(v: int, a: int) -> int:
    return -(-v // a) * a


def buffer_sizes(c: KernelCall) -> List[int]:
    """Element counts of the call's buffers, with margins (the kernels read
    and write whole 16-byte words)."""
    from src._matmul_hw_config import MATMUL_TILE_M
    f = c.fields
    pad = 256
    if c.kernel == "VectorOPKernel":
        span = _up(f["size"], 8)
        a = (f["outer"] - 1) * f["a_inc"] + span
        b = (f["outer"] - 1) * f["b_inc"] + span
        cc = (f["outer"] - 1) * (f["a_inc"] + f["b_inc"]) + span
        return [a + pad, b + pad, cc + pad]
    if c.kernel == "MatmulKernel":
        n, k, m, bt = f["n"], f["k"], f["m"], f["batch"]
        a = (bt - 1) * f["a_stride"] + n * k
        slab = k * _up(m, MATMUL_TILE_M) if f["b_packed"] else k * m
        b = (bt - 1) * f["b_stride"] + slab
        cc = (bt - 1) * f["c_stride"] + n * m
        return [a + pad, b + pad, cc + pad]
    if c.kernel == "ConvKernel":
        cin = 1 if f["is_dw"] else _up(f["in_ch"], 16)
        x = f["batch"] * _up(f["in_ch"], 16) * f["in_h"] * _up(f["in_w"], 8)
        w = _up(f["out_ch"], 16) * cin * f["kh"] * f["kw"]
        y = f["batch"] * _up(f["out_ch"], 16) * f["out_h"] * _up(f["out_w"], 8)
        return [x + pad, w + pad, (_up(f["out_ch"], 16) if f["has_bias"] else 8) + pad, y + pad]
    x = f["batch"] * f["channels"] * f["in_h"] * _up(f["in_w"], 8)
    y = f["batch"] * f["channels"] * f["out_h"] * _up(f["out_w"], 8)
    return [x + pad, y + pad]


def estimate_us(c: KernelCall) -> float:
    """The analytic cost model's time of one call (µs), to size iterations."""
    from src.cost_model import CALL_OVERHEAD, conv_batch_cycles, gemv_cycles, matmul_cycles
    f = c.fields
    if c.kernel == "ConvKernel":
        try:
            cyc = conv_batch_cycles(f["batch"], in_ch=f["in_ch"], out_ch=f["out_ch"],
                                    in_h=f["in_h"], in_w=f["in_w"], oh=f["out_h"], ow=f["out_w"],
                                    kh=f["kh"], kw=f["kw"], sh=f["stride_h"], sw=f["stride_w"],
                                    dh=f["dilation_h"], dw=f["dilation_w"],
                                    pt=f["pad_top"], pl=f["pad_left"])
            if f["is_dw"]:
                cyc /= max(1, f["in_ch"] // 16)
        except Exception:                                    # noqa: BLE001 — sizing only
            cyc = 1e6
    elif c.kernel == "MatmulKernel":
        cyc = (gemv_cycles(f["n"], f["k"], f["m"], f["batch"], f["gemv_kw"]) if f["gemv_kw"]
               else matmul_cycles(f["n"], f["k"], f["m"], f["batch"]))
    elif c.kernel == "VectorOPKernel":
        cyc = f["outer"] * math.ceil(f["size"] / 8) * (8 if f["op"] == 3 else 1) + 500
    else:
        cyc = f["batch"] * f["channels"] * (f["out_h"] * f["out_w"] * f["pool_h"] * f["pool_w"]
                                            + f["in_h"] * f["in_w"]) / 8 + 500
    return (cyc + CALL_OVERHEAD) / MHZ


def iterations(est_us: float) -> Tuple[int, int]:
    """(iters, warmup): about 30 ms of timed calls, 3 .. 200 iterations."""
    return max(3, min(200, int(30000.0 / max(est_us, 1.0)))), 2


def case_line(cid: str, c: KernelCall, iters: int, warmup: int) -> str:
    sizes = buffer_sizes(c)
    return " ".join([cid, KLETTER[c.kernel], str(iters), str(warmup), str(len(sizes)),
                     *map(str, sizes), *map(str, c.regs)])


# ─────────────────────────────────────────────────────────────────────────────
# cases
# ─────────────────────────────────────────────────────────────────────────────

def _holdout(key: str) -> bool:
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) % 4 == 0


def build_cases(models: Iterable[str], seed: int) -> dict:
    cases: Dict[str, dict] = {}
    for name in models:
        t0 = time.time()
        n0 = len(cases)
        for entry, g in shipped_graphs(name):
            for key, (c, label) in graph_calls(g).items():
                e = cases.setdefault(key, {"key": key, "kernel": c.kernel, "set": "shipped",
                                           "sources": []})
                if len(e["sources"]) < 4:
                    e["sources"].append(f"{name}/{entry}/{label}")
            del g
        log(f"  {name}: {len(cases) - n0} new calls ({time.time() - t0:.0f} s)")
    t0 = time.time()
    n0 = len(cases)
    for c, fam in grid_calls(seed):
        e = cases.get(c.key())
        if e is None:
            cases[c.key()] = {"key": c.key(), "kernel": c.kernel, "set": "grid", "family": fam,
                              "holdout": _holdout(c.key()), "sources": []}
    log(f"  space-filling set: {len(cases) - n0} new calls ({time.time() - t0:.0f} s)")
    for e in cases.values():
        c = KernelCall.from_key(e["key"])
        e["est_us"] = round(estimate_us(c), 3)
        e["iters"], e["warmup"] = iterations(e["est_us"])
    return {"platform": PLATFORM, "seed": seed, "models": list(models),
            "cases": sorted(cases.values(), key=lambda e: e["key"])}


def default_id(args) -> str:
    bid = getattr(args, "bitstream_id", None) or local_bitstream_id()
    if not bid:
        raise SystemExit("error: no bitstream id: pass --bitstream-id, or build the bitstream "
                         "named in bitstream_config_kv260.json")
    return bid


def refine_cases(cases: dict, pm, models: Iterable[str], per_matmul: int = 6,
                 within: float = 0.15) -> int:
    """Append to ``cases`` the calls of the tactics the fitted model ranks
    near the best of every shipped MatMul (within ``within`` of its best
    prediction, at most ``per_matmul``) that were never measured: the planner
    can only take a tactic from an untrusted family once it is measured
    (perf_model.MAX_MODEL_ERROR).  Returns the number added."""
    from src.codegen import CodeGenerator
    from src.tactics import graph_matmul_tactics
    have = {e["key"] for e in cases["cases"]}
    added = 0
    for name in models:
        t0, n0 = time.time(), added
        for entry, g in shipped_graphs(name):
            lay = CodeGenerator(g, model_path="calib.onnx")._layouts
            for sn, ts in graph_matmul_tactics(g, layouts=lay):
                priced = [(pm.calls_us(t.calls), t) for t in ts]
                priced = sorted((p for p in priced if p[0] is not None), key=lambda p: p[0])
                if not priced:
                    continue
                best = priced[0][0]
                for t_us, t in priced[:per_matmul]:
                    if t_us > best * (1.0 + within):
                        break
                    for c in t.calls:
                        k = KernelCall(c.kernel, c.regs, 1).key()
                        if k in have:
                            continue
                        have.add(k)
                        e = {"key": k, "kernel": c.kernel, "set": "refine",
                             "sources": [f"{name}/{entry}/{sn.onnx_node.name}:{t.label()}"]}
                        e["est_us"] = round(max(t_us / c.count, 1.0), 3)
                        e["iters"], e["warmup"] = iterations(e["est_us"])
                        cases["cases"].append(e)
                        added += 1
            del g
        log(f"  {name}: {added - n0} calls to measure ({time.time() - t0:.0f} s)")
    return added


def cmd_cases(args) -> int:
    bid = default_id(args)
    if args.refine:
        from src.perf_model import PerfModel
        cpath = MODELS_DIR / f"{bid}.cases.json"
        cases = json.loads(cpath.read_text())
        pm = PerfModel.load(MODELS_DIR / f"{bid}.json")
        n = refine_cases(cases, pm, args.models or list(SHIPPED))
        cpath.write_text(json.dumps(cases, indent=0) + "\n")
        log(f"{n} refinement calls added -> {cpath} (measure them: run --resume)")
        return 0
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    models = args.models or list(SHIPPED)
    log(f"case list for bitstream {bid}: models {', '.join(models)}")
    d = build_cases(models, args.seed)
    d["bitstream"] = bid
    path = MODELS_DIR / f"{bid}.cases.json"
    path.write_text(json.dumps(d, indent=0) + "\n")
    by = {}
    for e in d["cases"]:
        by[(e["kernel"], e["set"])] = by.get((e["kernel"], e["set"]), 0) + 1
    est = sum(e["est_us"] * (e["iters"] + e["warmup"]) for e in d["cases"]) / 1e6
    log(f"{len(d['cases'])} cases -> {path} ({', '.join(f'{k[0]} {k[1]} {v}' for k, v in sorted(by.items()))});"
        f" about {est / 60:.0f} min of calls per pass")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# run (board)
# ─────────────────────────────────────────────────────────────────────────────

def _server_active(session) -> bool:
    out, _, _ = session.exec("systemctl is-active kv260-chat 2>/dev/null || true", timeout=15)
    return out.strip() == "active"


def _deploy(*argv) -> int:
    py = str(HERE / ".venv" / "bin" / "python")
    return subprocess.call([py, str(CHAT / "deploy.py"), *argv])


def _drop_caches(session) -> None:
    session.exec("sync; echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory",
                 timeout=120)


def board_bitstream_id(session, board_bin: str) -> Optional[str]:
    out, _, rc = session.exec(f"sha256sum {board_bin} 2>/dev/null", timeout=60)
    return out.split()[0][:12] if rc == 0 and out.strip() else None


def _chunks(cases: List[dict], seconds: float) -> Iterator[List[dict]]:
    cur, t = [], 0.0
    for e in cases:
        dt = e["est_us"] * (e["iters"] + e["warmup"]) / 1e6 + 0.02
        if cur and t + dt > seconds:
            yield cur
            cur, t = [], 0.0
        cur.append(e)
        t += dt
    if cur:
        yield cur


def cmd_run(args) -> int:
    import run_remote_perf as rp
    from src.remote import RemoteSession, hold_board_lock, load_config, uio_devices_from_cfg
    bid = default_id(args)
    cpath = MODELS_DIR / f"{bid}.cases.json"
    if not cpath.exists():
        log(f"error: {cpath} missing — run 'perf_calibrate.py cases' first")
        return 1
    cases = json.loads(cpath.read_text())["cases"]
    dpath = MODELS_DIR / f"{bid}.calib.json"
    data = json.loads(dpath.read_text()) if dpath.exists() and args.resume else \
        {"platform": PLATFORM, "bitstream": bid, "runner": RUNNER_VERSION, "passes": {}}
    cfg = load_config(args.config, rp._EXTRA_DEFAULTS)
    uio = uio_devices_from_cfg(cfg)
    inst = [uio.get(k, "-") for k in ("VectorOPKernel", "MatmulKernel", "ConvKernel", "PoolingKernel")]
    hold_board_lock(cfg)                  # one job per board (src/remote/lock.py)
    session = RemoteSession(cfg["ssh"])
    session.connect()
    stopped = False
    try:
        board_bin = args.board_bin or local_board_bin()
        bb = board_bitstream_id(session, board_bin)
        if bb != bid:
            log(f"error: the board runs bitstream {bb} ({board_bin}), the cases are for {bid}")
            return 1
        if _server_active(session):
            if not args.stop_server:
                log("error: the chat server is running (it owns the kernels); pass --stop-server")
                return 1
            log("stopping the chat server")
            if _deploy("--stop") != 0:
                return 1
            stopped = True
        session.exec(f"mkdir -p {cfg['remote']['work_dir']}", timeout=10)
        with tempfile.TemporaryDirectory() as td:
            proj = rp._generate_local_project(Path(td), cfg)
            ok, build_dir, blog = rp.build_benchmark_project(session, cfg, proj)
        if not ok:
            log(blog[-3000:])
            return 1
        out, _, _ = session.exec(f"ls {build_dir}/calib_runner", timeout=10)
        if "calib_runner" not in out:
            log("error: calib_runner was not built (all four kernels' drivers are needed)")
            return 1
        _drop_caches(session)
        rng = np.random.default_rng(7)
        for p in (1, 2):
            order = list(cases) if p == 1 else [cases[i] for i in rng.permutation(len(cases))]
            done = data["passes"].setdefault(str(p), {})
            todo = [e for e in order if e["key"] not in done]
            log(f"pass {p}: {len(todo)} cases ({len(done)} done)")
            t0 = time.time()
            fill = 1 if p == 1 else 0x5A
            for chunk in _chunks(todo, args.chunk_seconds):
                ids = {f"c{i}": e for i, e in enumerate(chunk)}
                lines = [case_line(cid, KernelCall.from_key(e["key"]), e["iters"], e["warmup"])
                         for cid, e in ids.items()]
                remote = f"{build_dir}/cases_{p}.txt"
                session.exec(f"cat > {remote} <<'EOF'\n" + "\n".join(lines) + "\nEOF", timeout=60)
                cmd = f"{build_dir}/calib_runner {remote} {' '.join(inst)} {fill}"
                budget = sum(e["est_us"] * (e["iters"] + e["warmup"]) for e in chunk) / 1e6
                out, err, rc = session.exec(cmd, timeout=max(120, int(10 * budget) + 60))
                for ln in out.splitlines():
                    try:
                        r = json.loads(ln)
                    except json.JSONDecodeError:
                        continue
                    e = ids.get(r.get("id"))
                    if e is not None:
                        done[e["key"]] = {k: r[k] for k in r if k != "id"}
                dpath.write_text(json.dumps(data, indent=0) + "\n")
                if rc != 0:
                    log(f"error: calib_runner rc={rc}: {err.strip()[:500]}")
                    return 1
                log(f"  pass {p}: {len(done)}/{len(cases)} ({time.time() - t0:.0f} s)")
        data["date"] = time.strftime("%Y-%m-%d")
        dpath.write_text(json.dumps(data, indent=0) + "\n")
        log(f"measurements -> {dpath}")
        return 0
    finally:
        if stopped:
            log("restarting the chat server")
            _drop_caches(session)
            _deploy()
        session.close()


# ─────────────────────────────────────────────────────────────────────────────
# host (host-op models) and simulate (validation against board totals)
# ─────────────────────────────────────────────────────────────────────────────

def _profile_specs(specs: List[str]) -> List[Tuple[str, bool, dict]]:
    """``MODEL[:plan]=RESULTS.json`` (llm_board.py --profile --out) ->
    [(model, planned, results)]."""
    out = []
    for spec in specs:
        lhs, sep, path = spec.partition("=")
        if not sep:
            raise SystemExit(f"error: --profile {spec!r}: expected MODEL[:plan]=RESULTS.json")
        model, _, flag = lhs.partition(":")
        if model not in SHIPPED:
            raise SystemExit(f"error: --profile: unknown model {model!r}")
        out.append((model, flag == "plan", json.loads(Path(path).read_text())))
    return out


def _single(model: str, res) -> Optional[Tuple[list, float]]:
    """(per-layer stats, measured ms) of a single-graph model from the BERT
    demo's results.json or the image demo's results list, else None."""
    if isinstance(res, list):
        item = next((x for x in res if x.get("name") == model), None)
        if item is None:
            return None
        m = item["metrics"]
    elif "metrics" in res:
        m = res["metrics"]
    else:
        return None
    layers = [ly for ly in (m.get("layer_stats") or {}).get("layers", []) if ly.get("calls")]
    return layers, float(m.get("p50_ms") or m["mean_ms"])


def _entry_graphs(model: str, planned: bool) -> Dict[str, object]:
    """The entry graphs of a shipped chat model, planned or not."""
    from src.planning import PlanOptions
    if model.startswith("piper-"):
        if planned:
            raise SystemExit(f"error: {model}: generate_tts_project.py has no planned build")
        return dict(piper_graphs())
    if not (model.startswith("smollm2-") or model.startswith("smolvlm-")):
        from src.graph import OnnxGraph
        (name, g), = shipped_graphs(model)
        if planned:
            g = OnnxGraph(g._source_path if hasattr(g, "_source_path") else _model_path(model),
                          fuse_act=True, s2d_stem=True, fuse_patterns=(model == "bert"),
                          plan=PlanOptions(enabled=True))
        return {model: g}
    sys.path.insert(0, str(CHAT / "scripts"))
    from src.llm_entries import entry_graphs
    assets = str(CHAT / "assets" / model)
    if model.startswith("smolvlm-"):
        import vlm_project as vp
        m = vp.load(assets)
        models = vp.entry_models(*vp.frontends(m))
    else:
        import llm_project as lp
        cfg, W, fmt, _fd = lp.load_model(assets)
        models = lp.entry_models(lp.frontend(cfg, W, fmt))
        del W
    return dict(entry_graphs(models, plan=PlanOptions(enabled=planned) if planned else None))


def _split_by_entry(graphs: Dict[str, object], phase: str, layers: List[dict]) -> Dict[str, List[dict]]:
    """A profiler phase that ran several entries (piper's "encode": every
    encode_<T> bucket) -> {entry: its layers}.  A multi-entry project numbers
    the layers globally, entry after entry (MultiEntryGenerator), and the
    buckets share node names, so the layer index picks the entry."""
    base, ranges = 0, []
    for name, g in graphs.items():
        ranges.append((name, base, base + len(g.nodes)))
        base += len(g.nodes)
    out: Dict[str, List[dict]] = {}
    for ly in layers:
        i = ly.get("i")
        hit = next((n for n, lo, hi in ranges if i is not None and lo <= i < hi), None)
        if hit is not None and hit.startswith(phase + "_"):
            out.setdefault(hit, []).append(ly)
    return out


def cmd_host(args) -> int:
    from src.host_model import HostModel, default_host_model_path, observations_from_profile
    obs = []
    for model, planned, res in _profile_specs(args.profile):
        one = _single(model, res)
        if one is not None:
            graphs = _entry_graphs(model, planned)
            n0 = len(obs)
            obs += observations_from_profile(graphs[model], one[0])
            log(f"  {model}: {len(obs) - n0} host-op timings")
            continue
        layers = res.get("profile_layers") or {}
        if not layers:
            log(f"  {model}: no per-layer profile in the results (llm_board.py --profile --out)")
            continue
        graphs = _entry_graphs(model, planned)
        n0 = len(obs)
        for phase, ly in layers.items():
            if phase in graphs:
                obs += observations_from_profile(graphs[phase], ly)
            else:                       # one phase over several entries (piper's "encode")
                for entry, part in _split_by_entry(graphs, phase, ly).items():
                    obs += observations_from_profile(graphs[entry], part)
        log(f"  {model}{' (planned)' if planned else ''}: {len(obs) - n0} host-op timings")
    path = Path(args.host_model) if args.host_model else default_host_model_path()
    old = HostModel.load(path) if path.exists() and not args.fresh else None
    hm = HostModel.fit(obs, {"platform": PLATFORM, "date": time.strftime("%Y-%m-%d"),
                             "threads": 4})
    if old is not None:                       # keep earlier exact entries not re-measured
        for k, v in old.exact.items():
            hm.exact.setdefault(k, v)
    if old is not None and args.merge:        # keep every kind fitted before; add the new ones
        new_kinds = sorted(set(hm.kinds) - set(old.kinds))
        kept = sorted(set(hm.kinds) & set(old.kinds))
        hm.kinds = {**hm.kinds, **old.kinds}
        log(f"merge: {len(new_kinds)} new kinds {new_kinds}; {len(kept)} kept as fitted before "
            f"{kept} (their new signatures are exact entries)")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(hm.to_dict(), indent=1) + "\n")
    log(f"host model: {len(hm.exact)} exact signatures, {len(hm.kinds)} kinds -> {path}")
    for k, v in sorted(hm.kinds.items()):
        log(f"  {k:26s} n {v['n']:4d}  fit median {v['median_rel'] * 100:5.1f} %  max {v['max_rel'] * 100:6.1f} %")
    return 0


def cmd_simulate(args) -> int:
    from src.codegen import CodeGenerator
    from src.codegen.timing import kernel_duration_fn, simulate
    from src.host_model import HostModel, default_host_model_path
    from src.llm_nodes import attn_keys
    from src.perf_model import PerfModel
    pm = PerfModel.load(MODELS_DIR / f"{default_id(args)}.json")
    hm = HostModel.load(Path(args.host_model) if args.host_model else default_host_model_path())
    rows = []
    for model, planned, res in _profile_specs(args.profile):
        one = _single(model, res)
        if one is not None:
            g = _entry_graphs(model, planned)[model]
            cg = CodeGenerator(g, model_path=f"{model}.onnx")
            tl = simulate(cg, kernel_duration_fn(pm, cg._layouts), hm.us)
            rows.append((f"{model}{' plan' if planned else ''}", "inference", tl.total_us / 1e3,
                         one[1], tl, cg, one[0]))
            continue
        b = res.get("bench", res)
        graphs = _entry_graphs(model, planned)
        cgs = {n: CodeGenerator(g, model_path=f"{n}.onnx") for n, g in graphs.items()}
        prof = dict(res.get("profile_layers") or {})          # per-layer times, for --html
        for phase in [ph for ph in prof if ph not in graphs]:  # piper's old "encode": every bucket
            prof.update(_split_by_entry(graphs, phase, prof.pop(phase)))
        if model.startswith("piper-"):          # tts_board.py --out: chunks and encoder buckets
            tag = model
            ms = [m for u in b.get("utts", []) for m in u.get("chunk_ms", [])]
            if ms and "chunk" in cgs:
                tl = simulate(cgs["chunk"], kernel_duration_fn(pm, cgs["chunk"]._layouts), hm.us)
                rows.append((tag, "chunk", tl.total_us / 1e3, sorted(ms)[len(ms) // 2], tl, cgs["chunk"],
                             prof.get("chunk")))
            # per bucket, the sequence nearest a full bucket: the graph prices T rows,
            # and some host ops scale with the real id count n (tts_bench profiles
            # each bucket's sequences; the largest n is the last one per bucket)
            fullest = {}
            for e in b.get("encs", []):
                if e["n"] >= fullest.get(e["bucket"], {"n": -1})["n"]:
                    fullest[e["bucket"]] = e
            for T, e in sorted(fullest.items()):
                entry = f"encode_{T}"
                if entry in cgs:
                    tl = simulate(cgs[entry], kernel_duration_fn(pm, cgs[entry]._layouts), hm.us)
                    rows.append((tag, f"encode {T} ({e['n']} ids)", tl.total_us / 1e3, e["best_ms"], tl,
                                 cgs[entry], prof.get(entry)))
            continue

        def run(entry, pos0=1, n=None, cgs=cgs):
            cg = cgs[entry]

            def keys_of(sn):
                return attn_keys(pos0, n if n is not None else sn.T, sn.T, sn.C, sn.Q)[1]
            tl = simulate(cg, kernel_duration_fn(pm, cg._layouts, keys_of=keys_of), hm.us)
            tl.cg, tl.keys_of = cg, keys_of            # for --html: the calls as priced
            return tl
        tag = f"{model}{' plan' if planned else ''}"
        if "vision" in cgs and b.get("images"):
            tl = run("vision")
            meas = min(x["ms"] for x in b["images"])
            rows.append((tag, "llm_image", tl.total_us / 1e3, meas, tl, tl.cg, prof.get("vision")))
        for p in b.get("prefill", []):
            e = f"prefill_{p['n']}"
            if e in cgs:
                tl, th = run(e, 1, p["n"]), run("head")
                rows.append((tag, f"prefill {p['n']}", (tl.total_us + th.total_us) / 1e3,
                             p["best_ms"], tl, tl.cg, prof.get(e)))
        if b.get("llm_summary", {}).get("decode_ms_mean") and "decode" in cgs:
            tl = run("decode")                  # the decode entry includes the LM head
            rows.append((tag, "decode step", tl.total_us / 1e3,
                         b["llm_summary"]["decode_ms_mean"], tl, tl.cg, prof.get("decode")))
    log(f"{'project':28s} {'phase':20s} {'predicted':>10s} {'measured':>9s} {'error':>7s}  "
        f"{'CPU waits':>9s}  unpriced")
    for tag, ph, pred, meas, tl, _cg, _layers in rows:
        log(f"{tag:28s} {ph:20s} {pred:9.1f}ms {meas:8.1f}ms {(pred - meas) / meas * 100:+6.1f}%  "
            f"{tl.wait_us / 1e3:8.1f}ms  {len(set(tl.unpriced))}")
    if args.html:
        from src.timeline_html import timeline_entry, write_html
        entries = [timeline_entry(f"{tag} · {ph}", cg, tl, pm, hm, measured_us=meas * 1e3,
                                  note=("prefill: the entry only, the LM head not drawn" if ph.startswith("prefill")
                                        else ""), layers=layers)
                   for tag, ph, pred, meas, tl, cg, layers in rows]
        write_html(args.html, entries, "Predicted execution timeline",
                   f"performance model {pm.platform}/{pm.bitstream} · host model "
                   f"{hm.meta.get('date', '?')} · perf_calibrate.py simulate")
        log(f"timeline -> {args.html} ({len(entries)} phases, "
            f"{sum(e['measured_nodes'] for e in entries)} nodes with measured times)")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("cases", "run", "fit", "all", "host", "simulate"))
    ap.add_argument("--bitstream-id", default=None,
                    help="default: the id of the bitstream in bitstream_config_kv260.json")
    ap.add_argument("--models", nargs="+", default=None, choices=SHIPPED,
                    help="cases: the shipped models (default: all)")
    ap.add_argument("--seed", type=int, default=2026, help="cases: the space-filling set's seed")
    ap.add_argument("--refine", action="store_true",
                    help="cases: add the unmeasured tactics the fitted model ranks near each "
                         "shipped MatMul's best (then: run --resume, fit)")
    ap.add_argument("--config", default=None, help="run: the board config (run_remote_perf.py format)")
    ap.add_argument("--stop-server", action="store_true",
                    help="run: stop the chat server for the campaign and restart it afterwards")
    ap.add_argument("--board-bin", default=None,
                    help="run: the flat bitstream the board loaded (its id is checked; default: "
                         "/lib/firmware/<overlay>.bin of bitstream_config_kv260.json, the name "
                         "upload_bitstream.py gives it: overlay_name, else the .dtbo stem)")
    ap.add_argument("--chunk-seconds", type=float, default=60.0)
    ap.add_argument("--resume", action="store_true", help="run: keep measurements already taken")
    ap.add_argument("--profile", nargs="+", default=[], metavar="MODEL[:plan]=RESULTS",
                    help="host / simulate: llm_board.py --profile --out results of a shipped chat "
                         "model's project (':plan' when it was generated with --plan), "
                         "tts_board.py --profile --out results for piper-lessac-medium, or a demo's "
                         "results.json")
    ap.add_argument("--host-model", default=None,
                    help="host / simulate: the host-op model file (default perf_models/kv260/host.json)")
    ap.add_argument("--fresh", action="store_true", help="host: drop the earlier exact entries")
    ap.add_argument("--html", default=None, metavar="FILE",
                    help="simulate: also write the predicted timelines as one HTML page (src/timeline_html.py), "
                         "with the measured per-layer times of the results that carry a profile")
    ap.add_argument("--merge", action="store_true",
                    help="host: keep every op kind fitted before (fit only kinds the earlier model "
                         "lacks), so one new model's profile can be added without re-profiling the rest")
    args = ap.parse_args(argv)
    if args.cmd == "host":
        return cmd_host(args)
    if args.cmd == "simulate":
        return cmd_simulate(args)
    if args.cmd in ("cases", "all") and cmd_cases(args):
        return 1
    if args.cmd in ("run", "all"):
        if not args.config:
            ap.error("run needs --config")
        if cmd_run(args):
            return 1
    if args.cmd in ("fit", "all"):
        from src.perf_fit import cmd_fit
        return cmd_fit(MODELS_DIR, default_id(args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
