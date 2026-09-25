#!/usr/bin/env python3
"""
generate_project.py — schedule bertsquad-12 into a KV260 inference project
for the BERT-SQuAD demo.

  1. inference-scheduler (Python API, CLI defaults: pattern fusion,
     activation fusion) -> CMake C project in build/project/, with the
     76 large weight tensors as build/project/weights/*.dat (208 MB).
  2. driver/ populated from local.driver_dirs for the active kernels
     (VectorOPKernel, ConvKernel — the MatMuls, BERT_PLAN 2A — and
     MatmulKernel for the two MatMuls that stay there).
  3. src/squad_bench.c copied to test/, plus a generated test/bench_glue.h:
     the buffer order of inference_run(), each buffer's numel macro and the
     index of each role (input_ids, segment_ids, input_mask, unique_ids,
     start / end logits), all read from OnnxGraph — no hard-coded C names.
  4. CMakeLists.txt patched: squad_bench target added, the generated
     test_inference smoke test kept but off by default
     (-DINFERENCE_BUILD_TEST=ON builds it).
  5. build/project/layers.json: every scheduled node (profile index, name,
     op, engine, kind) for the per-kind latency breakdown, and
     build/project.json: the summary deploy_and_run.py reads.

usage:  generate_project.py [--config CFG] [--check-only]
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List

from _common import (DEMO_DIR, INPUT_ROLES, OPTIONAL_ROLES, OUTPUT_ROLES,
                     PROJECT_DIR, PROJECT_SUMMARY, demo_path, load_config, log,
                     model_path)

from src.codegen import CodeGenerator          # noqa: E402
from src.graph import OnnxGraph                # noqa: E402
from src.host_nodes import (GeluNode, HostNode, LayerNormNode,  # noqa: E402
                            SoftmaxNode, TransposeNode)
from src.kernels import KERNEL_REGISTRY        # noqa: E402
from src.nodes import MatmulConvNode, MatmulNode, ScheduledNode  # noqa: E402

SRC_DIR = DEMO_DIR / "src"

# Kinds of the per-layer breakdown, in report order.
KINDS = ("MatMul linear", "MatMul attention", "VectorOP", "LayerNorm", "GELU",
         "Softmax", "Transpose", "other")


def node_kind(sn) -> str:
    if isinstance(sn, (MatmulNode, MatmulConvNode)):
        return "MatMul linear" if sn.inputs[1].is_weight else "MatMul attention"
    if isinstance(sn, ScheduledNode):
        return "VectorOP"
    for cls, kind in ((LayerNormNode, "LayerNorm"), (GeluNode, "GELU"),
                      (SoftmaxNode, "Softmax"), (TransposeNode, "Transpose")):
        if isinstance(sn, cls):
            return kind
    return "other"


def node_engine(sn) -> str:
    if isinstance(sn, HostNode):
        return "host"
    return sn.kernel_name or "alias"


def build_graph(model: Path):
    """The scheduler's graph and code generator with the CLI's defaults."""
    g = OnnxGraph(str(model), fuse_act=True, s2d_stem=True, fuse_patterns=True)
    return g, CodeGenerator(graph=g, model_path=str(model))


def run_scheduler(model: Path, out_dir: Path) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    from inference_scheduler import main as sched_main
    rc = sched_main(["--out-dir", str(out_dir), str(model)])
    if rc != 0:
        raise RuntimeError(f"inference-scheduler failed for {model} (rc={rc})")


def populate_drivers(project_dir: Path, driver_dirs: Dict[str, str],
                     active: List[str]) -> List[str]:
    dst = project_dir / "driver"
    dst.mkdir(parents=True, exist_ok=True)
    missing = []
    for kernel in active:
        files = KERNEL_REGISTRY[kernel].driver_files
        src = driver_dirs.get(kernel)
        src = demo_path(src) if src else None
        for f in files:
            if src is not None and (src / f).exists():
                shutil.copy2(src / f, dst / f)
            else:
                missing.append(f"{kernel}/{f}" + (f"  (not under {src})" if src else ""))
    return missing


def patch_cmake(project_dir: Path, gen: CodeGenerator) -> None:
    cmake = project_dir / "CMakeLists.txt"
    text = cmake.read_text()
    text, n = re.subn(r'option\(INFERENCE_BUILD_TEST\s+"([^"]+)"\s+ON\)',
                      r'option(INFERENCE_BUILD_TEST "\1" OFF)', text, count=1)
    if n != 1:
        raise RuntimeError("CMakeLists.txt: INFERENCE_BUILD_TEST option not found")
    macros = "\n".join(f"            {kd.instance_macro}" for kd in gen._active_kernels)
    text += (
        "\n"
        "# -----------------------------------------------------------------------\n"
        "# squad_bench — BERT-SQuAD demo runner (added by\n"
        "# demo/bert_squad/scripts/generate_project.py).  UIO instance macros take\n"
        "# the same quoted form as test_inference: -DINFERENCE_..._INSTANCE=\\\"name\\\"\n"
        "# -----------------------------------------------------------------------\n"
        "if(INFERENCE_TARGET STREQUAL \"LINUX\")\n"
        "    add_executable(squad_bench test/squad_bench.c)\n"
        "    target_link_libraries(squad_bench PRIVATE inference m)\n"
        "    target_include_directories(squad_bench PRIVATE test)\n"
        "    target_compile_options(squad_bench PRIVATE -Wall -Wextra -O2)\n"
        "    foreach(_macro IN ITEMS\n"
        f"{macros})\n"
        "        if(DEFINED ${_macro})\n"
        "            target_compile_definitions(squad_bench PRIVATE ${_macro}=${${_macro}})\n"
        "        endif()\n"
        "    endforeach()\n"
        "endif()\n")
    cmake.write_text(text)


def resolve_io(g: OnnxGraph, io: Dict[str, str]) -> Dict[str, object]:
    """Map each role to its position in inference_run()'s argument list."""
    bufs = list(g.input_tensors) + list(g.output_tensors)
    by_name = {t.onnx_name: i for i, t in enumerate(bufs)}
    n_in = len(g.input_tensors)
    idx = {}
    for role in INPUT_ROLES + OUTPUT_ROLES:
        name = io.get(role)
        if not name:
            if role in OPTIONAL_ROLES:
                idx[role] = -1
                continue
            raise RuntimeError(f"io.{role} is not set")
        if name not in by_name:
            if role in OPTIONAL_ROLES:
                idx[role] = -1
                continue
            raise RuntimeError(f"io.{role} = '{name}' is not a graph input / output; "
                               f"graph has {sorted(by_name)}")
        i = by_name[name]
        if (role in INPUT_ROLES) != (i < n_in):
            raise RuntimeError(f"io.{role} = '{name}' is a graph "
                               f"{'output' if i >= n_in else 'input'}")
        idx[role] = i
    seq = bufs[idx["input_ids"]].numel
    for role in ("input_ids", "segment_ids", "input_mask", "unique_ids", "unique_ids_out"):
        if idx[role] >= 0 and not bufs[idx[role]].is_int:
            raise RuntimeError(f"io.{role}: '{bufs[idx[role]].onnx_name}' is not an integer tensor")
    for role in ("segment_ids", "input_mask", "start_logits", "end_logits"):
        t = bufs[idx[role]]
        if t.numel != seq:
            raise RuntimeError(f"io.{role}: '{t.onnx_name}' has {t.numel} elements, expected {seq}")
    for role in ("start_logits", "end_logits"):
        if bufs[idx[role]].is_int:
            raise RuntimeError(f"io.{role}: '{bufs[idx[role]].onnx_name}' is an integer tensor")
    unused_inputs = [bufs[i].onnx_name for i in range(n_in) if i not in idx.values()]
    if unused_inputs:
        raise RuntimeError(f"graph inputs without a role (squad_bench cannot fill them): {unused_inputs}")
    return {"index": idx, "seq_len": seq, "bufs": bufs, "n_inputs": n_in}


def emit_glue(project_dir: Path, *, model_name: str, g: OnnxGraph,
              gen: CodeGenerator, io_map: dict) -> None:
    bufs, idx, seq = io_map["bufs"], io_map["index"], io_map["seq_len"]
    active = [kd for kd in gen._active_kernels]
    numel = ",\n".join(f"    INFERENCE_{t.c_name.upper()}_SIZE" for t in bufs)
    names = ",\n".join(f"    \"{t.onnx_name}\"" for t in bufs)
    is_int = ", ".join("1" if t.is_int else "0" for t in bufs)
    args = ", ".join(f"b[{i}]" for i in range(len(bufs)))
    listing = "\n".join(
        f" *   [{i}] {'in ' if i < io_map['n_inputs'] else 'out'} {t.c_name:<28s} "
        f"'{t.onnx_name}' {list(t.shape)}{' int16' if t.is_int else ' Q8.8'}"
        for i, t in enumerate(bufs))
    roles = "\n".join(f"#define BENCH_IDX_{r.upper():<16s} {idx[r]:2d}"
                      f"{'   /* -1: absent */' if r in OPTIONAL_ROLES else ''}"
                      for r in INPUT_ROLES + OUTPUT_ROLES)
    glue = f"""/*
 * bench_glue.h — generated by demo/bert_squad/scripts/generate_project.py:
 *                bridge between squad_bench.c and this model's inference API.
 *
 * Model          : {model_name}
 * Active kernels : {", ".join(kd.name for kd in active)}
 * inference_run() buffers, in argument order (graph inputs, then outputs):
{listing}
 */
#pragma once

#include "inference.h"

#define BENCH_MODEL_NAME  "{model_name}"
#define BENCH_SEQ_LEN     {seq}u
#define BENCH_N_BUFS      {len(bufs)}u
#define BENCH_N_INPUTS    {io_map['n_inputs']}u

/* Position of each role in the buffer list. */
{roles}

static const unsigned bench_buf_numel[BENCH_N_BUFS] = {{
{numel}
}};
static const char *const bench_buf_name[BENCH_N_BUFS] = {{
{names}
}};
/* 1 = integer tensor (raw int16 in Data_t), 0 = ap_fixed<16,8> */
static const unsigned char bench_buf_is_int[BENCH_N_BUFS] = {{ {is_int} }};

static inline int bench_inference_init(void)
{{
    return inference_init({", ".join(kd.instance_macro for kd in active)});
}}

static inline void bench_inference_run(inference_buf_t *const *b)
{{
    inference_run({args});
}}
"""
    (project_dir / "test" / "bench_glue.h").write_text(glue)


def write_layers(project_dir: Path, g: OnnxGraph) -> List[dict]:
    layers = []
    for sn in g.nodes:
        rec = {"i": sn.index, "name": sn.onnx_node.name or sn.output.onnx_name,
               "op": sn.onnx_node.op_type, "engine": node_engine(sn),
               "kind": node_kind(sn), "out_numel": sn.output.numel}
        if isinstance(sn, (MatmulNode, MatmulConvNode)):
            rec["macs"] = sn.n * sn.k * sn.m * sn.batch * sn.outer_count
            rec["nkm"] = [sn.n, sn.k, sn.m, sn.batch * sn.outer_count]
        if isinstance(sn, MatmulConvNode):
            # the engine cost model's estimate, for model-vs-board per kind
            rec["conv"] = {"kw": sn.kw, "out_h": sn.out_h, "out_w": sn.out_w,
                           "calls": sn.calls, "est_cycles": sn.est_conv_cycles,
                           "est_matmul_cycles": sn.est_matmul_cycles}
        layers.append(rec)
    (project_dir / "layers.json").write_text(json.dumps(layers, indent=0))
    return layers


def preflight(cfg: dict) -> bool:
    ok = True
    m = model_path(cfg)
    if not m.exists():
        log(f"error: model {m} not found — see README.md 'Assets'")
        ok = False
    if not (SRC_DIR / "squad_bench.c").exists():
        log(f"error: {SRC_DIR / 'squad_bench.c'} missing")
        ok = False
    for k, p in (cfg.get("local", {}).get("driver_dirs") or {}).items():
        if k.startswith("_"):
            continue
        if not demo_path(p).is_dir():
            log(f"warning: local.driver_dirs.{k}: {demo_path(p)} not found "
                f"(make synthesize_{ {'VectorOPKernel': 'vectorop', 'ConvKernel': 'conv'}.get(k, 'matmul')}_kv260)")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out-dir", default=str(PROJECT_DIR))
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if not preflight(cfg):
        return 1
    if args.check_only:
        log("preflight: ok")
        return 0

    model = model_path(cfg)
    out = Path(args.out_dir)
    t0 = time.time()
    log(f"scheduling {model.name} -> {out}")
    run_scheduler(model, out)
    g, gen = build_graph(model)
    active = [kd.name for kd in gen._active_kernels]
    log(f"active kernels: {', '.join(active)}   nodes: {len(g.nodes)}   "
        f"({time.time() - t0:.0f} s)")

    io_map = resolve_io(g, cfg["io"])
    missing = populate_drivers(out, {k: v for k, v in
                                     (cfg.get("local", {}).get("driver_dirs") or {}).items()
                                     if not k.startswith("_")}, active)
    if missing:
        log("warning: missing driver files (the on-board cmake will fail):")
        for m in missing:
            log(f"    {m}")

    patch_cmake(out, gen)
    shutil.copy2(SRC_DIR / "squad_bench.c", out / "test" / "squad_bench.c")
    emit_glue(out, model_name=model.stem, g=g, gen=gen, io_map=io_map)
    layers = write_layers(out, g)

    weights = sorted((out / "weights").glob("*.dat"))
    summary = {
        "model":         str(model),
        "model_name":    model.stem,
        "project_dir":   str(out),
        "active":        active,
        "missing":       missing,
        "nodes":         len(g.nodes),
        "seq_len":       io_map["seq_len"],
        "io": {role: ({"index": i, "onnx_name": io_map["bufs"][i].onnx_name,
                       "c_name": io_map["bufs"][i].c_name} if i >= 0 else None)
               for role, i in io_map["index"].items()},
        "weights":       {"files": len(weights),
                          "bytes": sum(p.stat().st_size for p in weights)},
        "kinds":         {k: sum(1 for L in layers if L["kind"] == k) for k in KINDS},
        "generated_s":   round(time.time() - t0, 1),
    }
    PROJECT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    PROJECT_SUMMARY.write_text(json.dumps(summary, indent=2))
    log(f"project ready: {out}  ({summary['weights']['files']} weight files, "
        f"{summary['weights']['bytes'] / 1e6:.0f} MB; summary {PROJECT_SUMMARY})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
