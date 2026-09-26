#!/usr/bin/env python3
"""
generate_llm_project.py — schedule SmolLM2-135M-Instruct (any Llama-family
checkpoint with a formats JSON) into the multi-entry KV260 project behind
libsmollm2.so (doc/CHAT_PLAN.md phase 3).

  1. src/llama.py frontend: config.json + model.safetensors + the calibrated
     formats (llm_study.py formats, policy pow2+sink+p12; the shipped
     attention is xattn) -> entry graphs decode, prefill_<P> per bucket,
     head.
  2. inference-scheduler: OnnxGraph per entry (the decode step and the head
     are N = 1 MatMuls -> MatmulKernel with packed B; the prefill buckets use
     the MatMul-on-ConvKernel lowering where the cost model says it wins,
     every bucket with the same kernel width per weight so they share one
     re-laid-out copy; --prefill-engine matmul keeps prefill on MatmulKernel
     and the decode copy: one weight copy, slower prefill).
  3. src/codegen/multi.py: ONE project, weights deduplicated, the KV cache /
     sink and h_last as shared states -> <out>/ (CMake project, weights/*.dat
     incl. the host tables: the bf16 embedding, RoPE cos / sin).
  4. driver/ for the active kernels, test/llm_api.{c,h}, test/llm_bench.c,
     a generated test/llm_glue.h (entry functions, vocabulary, context,
     buckets), CMake targets llm_bench and smollm2 (libsmollm2.so: only the
     llm_* symbols exported), layers.json (per-layer kinds for the profile)
     and project.json (summary).

usage: inference-scheduler/.venv/bin/python demo/chat/scripts/generate_llm_project.py
           [--out-dir demo/chat/build/llm_project] [--buckets 16,64,256]
           [--prefill-engine conv|matmul] [--no-weights] [--driver-dirs JSON]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import llm_project as lp                                           # noqa: E402
from src.codegen.multi import MultiEntryGenerator                  # noqa: E402
from src.graph import OnnxGraph                                    # noqa: E402
from src.host_nodes import HostNode                                # noqa: E402
from src.kernels import KERNEL_REGISTRY                            # noqa: E402
from src.llm_nodes import (LlmAttentionNode, LlmDequantNode,       # noqa: E402
                           LlmEmbedNode, LlmResAddNode, LlmRMSNormNode,
                           LlmSelectRowNode, LlmSiluMulNode)
from src.nodes import MatmulConvNode, MatmulNode                   # noqa: E402

CHAT = os.path.dirname(HERE)
SRC = os.path.join(CHAT, "src")
C_SOURCES = ("llm_api.c", "llm_api.h", "llm_bench.c")
DEFAULT_OUT = os.path.join(CHAT, "build", "llm_project")

KINDS = ("MatMul linear", "LM head", "attention (host)", "RMSNorm", "residual add",
         "SiLU*up", "embedding", "other host")


def node_kind(sn) -> str:
    if isinstance(sn, (MatmulNode, MatmulConvNode)):
        return "LM head" if sn.m >= 4096 and sn.inputs[1].is_weight and \
            "lm_head" in sn.inputs[1].onnx_name else "MatMul linear"
    for cls, kind in ((LlmAttentionNode, "attention (host)"), (LlmRMSNormNode, "RMSNorm"),
                      (LlmResAddNode, "residual add"), (LlmSiluMulNode, "SiLU*up"),
                      (LlmEmbedNode, "embedding")):
        if isinstance(sn, cls):
            return kind
    return "other host" if isinstance(sn, HostNode) else "other"


def build_entries(models: dict, prefill_engine: str, log=print):
    """[(name, OnnxGraph)]: decode, the prefill buckets (largest first plans
    the conv kernel widths the smaller ones reuse), head."""
    graphs = {}
    t0 = time.time()
    graphs["decode"] = OnnxGraph(models["decode"], fuse_act=True, s2d_stem=True)
    log(f"  decode: {len(graphs['decode'].nodes)} nodes ({time.time() - t0:.0f} s)")
    prefills = sorted((n for n in models if n.startswith("prefill_")),
                      key=lambda n: -int(n.split("_")[1]))
    kw = None
    for name in prefills:
        t0 = time.time()
        g = OnnxGraph(models[name], fuse_act=True, s2d_stem=True,
                      matmul_on_conv="auto" if prefill_engine == "conv" else "off",
                      matmul_conv_kw=kw)
        if kw is None:
            kw = {sn.inputs[1].onnx_name: sn.kw for sn in g.nodes
                  if isinstance(sn, MatmulConvNode) and sn.inputs[1].is_weight}
        graphs[name] = g
        log(f"  {name}: {len(g.nodes)} nodes, MatMul on ConvKernel "
            f"{g.matmul_conv_stats['lowered']} / kept {g.matmul_conv_stats['kept']} "
            f"({time.time() - t0:.0f} s)")
    graphs["head"] = OnnxGraph(models["head"], fuse_act=True, s2d_stem=True)
    order = ["decode"] + sorted(prefills, key=lambda n: int(n.split("_")[1])) + ["head"]
    return [(n, graphs[n]) for n in order]


def populate_drivers(out: str, driver_dirs: dict, active) -> list:
    dst = os.path.join(out, "driver")
    os.makedirs(dst, exist_ok=True)
    missing = []
    for kd in active:
        src = driver_dirs.get(kd.name)
        for f in kd.driver_files:
            p = os.path.join(src, f) if src else None
            if p and os.path.exists(p):
                shutil.copy2(p, os.path.join(dst, f))
            else:
                missing.append(f"{kd.name}/{f}")
    return missing


def emit_glue(out: str, mg: MultiEntryGenerator, model_name: str, cfg, ctx, buckets) -> None:
    active = mg._active_kernels
    cases = "\n".join(
        f"    case {b}u: inference_run_prefill_{b}(ids, pos, n); break;" for b in buckets)
    glue = f"""/*
 * llm_glue.h — generated by demo/chat/scripts/generate_llm_project.py: the
 * bridge between llm_api.c and this project's multi-entry inference API.
 *
 * Model   : {model_name} (layers {cfg.L}, hidden {cfg.D}, heads {cfg.H}/{cfg.KV},
 *           head_dim {cfg.HD}, FFN {cfg.FF}, vocab {cfg.V})
 * Kernels : {", ".join(kd.name for kd in active)}
 * Entries : {", ".join(n for n, _ in mg.entries)}
 */
#pragma once

#include "inference.h"

#define LLM_MODEL_NAME  "{model_name}"
#define LLM_VOCAB       {cfg.V}
#define LLM_CONTEXT     {ctx}
#define LLM_N_BUCKETS   {len(buckets)}u
#define LLM_MAX_BUCKET  {max(buckets)}u

static const unsigned llm_buckets[LLM_N_BUCKETS] = {{ {", ".join(f"{b}u" for b in buckets)} }};

static inline int llm_glue_init(void)
{{
    return inference_init({", ".join(kd.instance_macro for kd in active)});
}}

/* One prefill call: rows [0, *n) of ids (padded to the bucket) at cache
 * positions *pos .. *pos + *n - 1; leaves the last valid row in h_last. */
static inline void llm_glue_prefill(unsigned bucket, const int32_t *ids, const int32_t *pos,
                                    const int32_t *n)
{{
    switch (bucket) {{
{cases}
    default: break;
    }}
}}

static inline void llm_glue_head(float *logits)
{{
    inference_run_head(logits);
}}

static inline void llm_glue_decode(const int32_t *id, const int32_t *pos, float *logits)
{{
    inference_run_decode(id, pos, logits);
}}
"""
    with open(os.path.join(out, "test", "llm_glue.h"), "w") as f:
        f.write(glue)


def patch_cmake(out: str, mg: MultiEntryGenerator) -> None:
    path = os.path.join(out, "CMakeLists.txt")
    text = open(path).read()
    text, n = re.subn(r'option\(INFERENCE_BUILD_TEST\s+"([^"]+)"\s+ON\)',
                      r'option(INFERENCE_BUILD_TEST "\1" OFF)', text, count=1)
    if n != 1:
        raise RuntimeError("CMakeLists.txt: INFERENCE_BUILD_TEST option not found")
    macros = "\n".join(f"            {kd.instance_macro}" for kd in mg._active_kernels)
    text += (
        "\n"
        "# -----------------------------------------------------------------------\n"
        "# llm_bench — the phase-3 board runner, and smollm2 — libsmollm2.so for\n"
        "# the chat server (demo/chat/), both over test/llm_api.c (added by\n"
        "# demo/chat/scripts/generate_llm_project.py).  Only the llm_* symbols of\n"
        "# the shared library are exported.  UIO instance macros take the quoted\n"
        "# form of test_inference: -DINFERENCE_..._INSTANCE=\\\"name\\\"\n"
        "# -----------------------------------------------------------------------\n"
        "if(INFERENCE_TARGET STREQUAL \"LINUX\")\n"
        "    set_property(TARGET inference PROPERTY POSITION_INDEPENDENT_CODE ON)\n"
        "    add_executable(llm_bench test/llm_bench.c test/llm_api.c)\n"
        "    target_link_libraries(llm_bench PRIVATE inference m)\n"
        "    add_library(smollm2 SHARED test/llm_api.c)\n"
        "    target_link_libraries(smollm2 PRIVATE inference m)\n"
        "    target_link_options(smollm2 PRIVATE -Wl,--exclude-libs,ALL -Wl,--no-undefined)\n"
        "    foreach(_tgt IN ITEMS llm_bench smollm2)\n"
        "        target_include_directories(${_tgt} PRIVATE test)\n"
        "        target_compile_options(${_tgt} PRIVATE -Wall -Wextra -O2)\n"
        "        target_compile_definitions(${_tgt} PRIVATE\n"
        "            LLM_API_WEIGHTS_DIR=\"${INFERENCE_WEIGHTS_DIR}\")\n"
        "        foreach(_macro IN ITEMS\n"
        f"{macros})\n"
        "            if(DEFINED ${_macro})\n"
        "                target_compile_definitions(${_tgt} PRIVATE ${_macro}=${${_macro}})\n"
        "            endif()\n"
        "        endforeach()\n"
        "    endforeach()\n"
        "endif()\n")
    with open(path, "w") as f:
        f.write(text)


def write_layers(out: str, mg: MultiEntryGenerator) -> list:
    layers = []
    for name, g in mg.entries:
        for sn in g.nodes:
            rec = {"i": sn.index, "entry": name, "name": sn.onnx_node.name,
                   "op": sn.onnx_node.op_type, "kind": node_kind(sn),
                   "engine": "host" if isinstance(sn, HostNode) else (sn.kernel_name or "alias")}
            if isinstance(sn, (MatmulNode, MatmulConvNode)):
                rec["macs"] = sn.n * sn.k * sn.m * sn.batch * sn.outer_count
                rec["weight_bytes"] = sn.inputs[1].numel * 2 if sn.inputs[1].is_weight else 0
            layers.append(rec)
    with open(os.path.join(out, "layers.json"), "w") as f:
        json.dump(layers, f, indent=0)
    return layers


def driver_dirs_from_config(path: str) -> dict:
    """local.driver_dirs of demo/bert_squad/bert_squad_config.json (the same
    HLS driver sources; relative paths are relative to demo/bert_squad/)."""
    if not path or not os.path.exists(path):
        return {}
    with open(path) as f:
        cfg = json.load(f)
    base = os.path.dirname(os.path.abspath(path))
    out = {}
    for k, v in (cfg.get("local", {}).get("driver_dirs") or {}).items():
        if k.startswith("_") or not v:
            continue
        v = os.path.expanduser(v)
        out[k] = v if os.path.isabs(v) else os.path.normpath(os.path.join(base, v))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--assets", default=None)
    ap.add_argument("--formats", default=None)
    ap.add_argument("--buckets", default=",".join(map(str, lp.BUCKETS)))
    ap.add_argument("--context", type=int, default=lp.CONTEXT)
    ap.add_argument("--prefill-engine", choices=("conv", "matmul"), default="conv")
    ap.add_argument("--model-name", default="smollm2-135m-instruct")
    ap.add_argument("--no-weights", action="store_true", help="skip writing weights/*.dat")
    ap.add_argument("--driver-dirs", default=None,
                    help="JSON {kernel: dir}; default: local.driver_dirs of "
                         "demo/bert_squad/bert_squad_config.json")
    args = ap.parse_args(argv)
    t0 = time.time()
    buckets = sorted(int(b) for b in args.buckets.split(","))
    cfg, W, fmt, _fd = lp.load_model(args.assets, args.formats)
    fe = lp.frontend(cfg, W, fmt, ctx=args.context, name=args.model_name)
    print(f"frontend: {cfg.L} layers, hidden {cfg.D}, heads {cfg.H}/{cfg.KV}, vocab {cfg.V}, "
          f"context {args.context}, buckets {buckets}, prefill on {args.prefill_engine}",
          flush=True)
    models = lp.entry_models(fe, buckets)
    del W
    entries = build_entries(models, args.prefill_engine,
                            log=lambda m: print(m, flush=True))
    del models
    mg = MultiEntryGenerator(entries, args.model_name.replace("-", "_").replace(".", "_"))
    out = os.path.abspath(args.out_dir)
    if os.path.exists(out):
        keep = os.path.join(out, "weights") if args.no_weights else None
        for item in os.listdir(out):
            p = os.path.join(out, item)
            if keep and p == keep:
                continue
            shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
    os.makedirs(out, exist_ok=True)
    summary = mg.write_project(out, weights=not args.no_weights)
    dd = json.loads(args.driver_dirs) if args.driver_dirs else driver_dirs_from_config(
        os.path.join(lp.REPO, "demo", "bert_squad", "bert_squad_config.json")
        if os.path.exists(os.path.join(lp.REPO, "demo", "bert_squad", "bert_squad_config.json"))
        else os.path.join(lp._main_checkout(), "demo", "bert_squad", "bert_squad_config.json"))
    missing = populate_drivers(out, dd, mg._active_kernels)
    for name in C_SOURCES:
        shutil.copy2(os.path.join(SRC, name), os.path.join(out, "test", name))
    emit_glue(out, mg, args.model_name, cfg, args.context, buckets)
    patch_cmake(out, mg)
    layers = write_layers(out, mg)
    summary.update({
        "model": args.model_name, "context": args.context, "buckets": buckets,
        "prefill_engine": args.prefill_engine, "policy": lp.POLICY,
        "formats": os.path.abspath(args.formats or lp.default_formats(args.assets)),
        "config": {"layers": cfg.L, "hidden": cfg.D, "heads": cfg.H, "kv_heads": cfg.KV,
                   "head_dim": cfg.HD, "ffn": cfg.FF, "vocab": cfg.V},
        "kinds": {k: sum(1 for L in layers if L["kind"] == k) for k in KINDS},
        "missing_drivers": missing, "targets": ["llm_bench", "smollm2"],
        "generated_s": round(time.time() - t0, 1)})
    with open(os.path.join(out, "project.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"project: {out}")
    print(f"  pool {summary['pool_bytes'] / 2**20:.1f} MiB (weights {summary['weights_bytes'] / 2**20:.1f},"
          f" intermediates {summary['intermediate_region_bytes'] / 2**20:.2f}), host tables "
          f"{summary['host_table_bytes'] / 2**20:.1f} MiB, states {summary['state_bytes'] / 2**20:.1f} MiB,"
          f" host arena {summary['host_arena_bytes'] / 2**20:.2f} MiB")
    print(f"  weight files {summary['weight_file_bytes'] / 1e6:.0f} MB; renamed (second layout) "
          f"{len(summary['renamed_weights'])} weights; {time.time() - t0:.0f} s")
    if missing:
        print(f"  warning: missing driver files: {missing}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
