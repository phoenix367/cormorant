#!/usr/bin/env python3
"""
generate_llm_project.py — schedule a Llama-family checkpoint with a formats
JSON (SmolLM2-135M / 360M-Instruct; SmolVLM-256M-Instruct with its vision
entry) into the multi-entry KV260 project behind the chat library
(libsmollm2.so, doc/plans/CHAT_PLAN.md phase 3, §20, §23).

  1. src/llama.py frontend: config.json + model.safetensors + the calibrated
     formats (llm_study.py formats, policy pow2+sink+p12) -> entry graphs
     decode, prefill_<P> per bucket, head.  --prefill-attn fpga (default,
     policy pow2+sink+p12+mix, CHAT_PLAN §16): prefill attention q.K^T / P.V
     on ConvKernel over the runtime key count with the p12 host softmax,
     decode attention the xattn host region, the KV caches DMA states in the
     CMA pool; --prefill-attn host: phase 3's xattn everywhere
     (pow2+sink+p12+xattn), host-memory caches.
  2. inference-scheduler: OnnxGraph per entry (src/llm_entries.py: the
     prefill buckets use the MatMul-on-ConvKernel lowering where the cost
     model says it wins, every bucket with the same kernel width per weight;
     the decode step and the head are N = 1 MatMuls on MatmulKernel's GEMV
     path reading that same image — one copy of every weight, CHAT_PLAN §19;
     --prefill-engine matmul keeps prefill on MatmulKernel.  A VLM
     (--assets of SmolVLM) adds the `vision` entry, src/vit.py.  --plan and
     the other planning options: doc/plans/TACTICS_PLAN.md).
  3. src/codegen/multi.py: ONE project, weights deduplicated, the KV cache /
     sink and h_last as shared states -> <out>/ (CMake project, weights/*.dat
     incl. the host tables: the bf16 embedding, RoPE cos / sin).
  4. driver/ for the active kernels, test/llm_api.{c,h}, test/llm_bench.c,
     a generated test/llm_glue.h (entry functions, vocabulary, context,
     buckets), CMake targets llm_bench and smollm2 (libsmollm2.so: only the
     llm_* symbols exported), layers.json (per-layer kinds for the profile)
     and project.json (summary).

usage: inference-scheduler/.venv/bin/python demo/chat/scripts/generate_llm_project.py
           [--assets DIR] [--model-name NAME] [--formats JSON]
           [--out-dir demo/chat/build/llm_project] [--buckets 16,64,256]
           [--prefill-engine conv|matmul] [--prefill-attn fpga|host] [--no-weights]
           [--driver-dirs JSON] [--force] [--plan ...]

The model name is the checkpoint directory's name (assets/<model name>): it
names the output project, the library and — through llm_board.py — the
board's library and weights paths, so --model-name must agree with --assets
(--allow-name-mismatch overrides) and a model name alone looks for
assets/<name>.  Without either the default is SmolLM2-135M.  An output
directory holding another model's project, or anything that is not a
generated project, is not wiped without --force.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from typing import Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import llm_project as lp                                           # noqa: E402
import vlm_project as vp                                           # noqa: E402
from src.codegen.multi import MultiEntryGenerator                  # noqa: E402
from src.llm_entries import entry_graphs                         # noqa: E402
from src.host_nodes import HostNode                                # noqa: E402
from src.kernels import KERNEL_REGISTRY                            # noqa: E402
from src.llm_nodes import (LlmAttentionNode, LlmAttnConvNode,      # noqa: E402
                           LlmAttnMergeNode, LlmAttnPrepNode, LlmAttnSoftmaxNode,
                           LlmDequantNode, LlmEmbedNode, LlmResAddNode, LlmRMSNormNode,
                           LlmSelectRowNode, LlmSiluMulNode)
from src.nodes import MatmulConvNode, MatmulNode                   # noqa: E402
from src.planning import add_plan_args, plan_options_from_args      # noqa: E402
from src.vit_nodes import (VitAttnPrepNode, VitAttnSoftmaxNode,    # noqa: E402
                           VitEmbedAddNode, VitGeluNode, VitLayerNormNode, VitPixelShuffleNode,
                           VitResAddNode, VitSumDequantNode)

CHAT = os.path.dirname(HERE)
SRC = os.path.join(CHAT, "src")
C_SOURCES = ("llm_api.c", "llm_api.h", "llm_bench.c")
DEFAULT_OUT = os.path.join(CHAT, "build", "llm_project")
DEFAULT_MODEL = "smollm2-135m-instruct"

KINDS = ("MatMul linear", "LM head", "attention (host)", "attention q.K^T (FPGA)",
         "attention P.V (FPGA)", "attention softmax (host)", "attention prep (host)",
         "attention merge (host)", "RMSNorm", "residual add", "SiLU*up", "embedding",
         "other host", "vision MatMul", "vision attention q.K^T (FPGA)",
         "vision attention P.V (FPGA)", "vision attention softmax (host)",
         "vision attention prep (host)", "vision attention merge (host)", "vision LayerNorm",
         "vision residual add", "vision GELU", "vision patch embedding add",
         "vision pixel shuffle", "vision connector sum")


VISION_KINDS = ((VitAttnSoftmaxNode, "vision attention softmax (host)"),
                (VitAttnPrepNode, "vision attention prep (host)"),
                (VitLayerNormNode, "vision LayerNorm"), (VitResAddNode, "vision residual add"),
                (VitGeluNode, "vision GELU"), (VitEmbedAddNode, "vision patch embedding add"),
                (VitPixelShuffleNode, "vision pixel shuffle"),
                (VitSumDequantNode, "vision connector sum"))


def node_kind(sn) -> str:
    if sn.onnx_node.name.startswith("vision."):
        if isinstance(sn, (MatmulNode, MatmulConvNode)):
            return "vision MatMul"
        if isinstance(sn, LlmAttnConvNode):
            return ("vision attention q.K^T (FPGA)" if sn.kind == "qk"
                    else "vision attention P.V (FPGA)")
        if isinstance(sn, LlmAttnMergeNode):
            return "vision attention merge (host)"
        for cls, kind in VISION_KINDS:
            if isinstance(sn, cls):
                return kind
    if isinstance(sn, (MatmulNode, MatmulConvNode)):
        return "LM head" if sn.m >= 4096 and sn.inputs[1].is_weight and \
            "lm_head" in sn.inputs[1].onnx_name else "MatMul linear"
    if isinstance(sn, LlmAttnConvNode):
        return "attention q.K^T (FPGA)" if sn.kind == "qk" else "attention P.V (FPGA)"
    for cls, kind in ((LlmAttentionNode, "attention (host)"),
                      (LlmAttnSoftmaxNode, "attention softmax (host)"),
                      (LlmAttnPrepNode, "attention prep (host)"),
                      (LlmAttnMergeNode, "attention merge (host)"), (LlmRMSNormNode, "RMSNorm"),
                      (LlmResAddNode, "residual add"), (LlmSiluMulNode, "SiLU*up"),
                      (LlmEmbedNode, "embedding")):
        if isinstance(sn, cls):
            return kind
    return "other host" if isinstance(sn, HostNode) else "other"


def build_entries(models: dict, prefill_engine: str, log=print, plan=None):
    """[(name, OnnxGraph)]: decode, the prefill buckets, head — one copy of
    every weight where MatmulKernel's GEMV path can read the prefill image
    (src/llm_entries.py); ``plan`` = the planning options (src/planning.py)."""
    return entry_graphs(models, prefill_engine=prefill_engine, log=log, plan=plan)


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


def vision_glue(vcfg) -> str:
    """llm_glue.h's image part (LLM_IMAGE_TOKENS 0 for a text-only model)."""
    if vcfg is None:
        return "\n#define LLM_IMAGE_TOKENS 0\n"
    return f"""
/* The vision entry (src/vit.py): an image of LLM_IMAGE_SIZE^2 RGB pixels in
 * LLM_PATCH^2 patches -> LLM_IMAGE_TOKENS image-feature rows (the state the
 * prefill entries read for ids LLM_VOCAB .. LLM_VOCAB + LLM_IMAGE_TOKENS - 1). */
#define LLM_IMAGE_TOKENS {vcfg.n_img}
#define LLM_IMAGE_SIZE   {vcfg.S}
#define LLM_PATCH        {vcfg.P}
#define LLM_N_PATCHES    {vcfg.N}u

static inline void llm_glue_vision(inference_buf_t *patches)
{{
    inference_run_vision(patches);
}}
"""


def emit_glue(out: str, mg: MultiEntryGenerator, model_name: str, cfg, ctx, buckets,
              vcfg=None) -> None:
    active = mg._active_kernels
    buckets = sorted(buckets)
    costs = lp.bucket_costs(buckets)
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
/* Cost of one call per bucket (ms on the board without the head,
 * llm_project.py BUCKET_COST_MS): llm_prefill's least-cost split. */
static const unsigned llm_bucket_cost[LLM_N_BUCKETS] = {{ {", ".join(f"{c}u" for c in costs)} }};

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
""" + vision_glue(vcfg)
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


def resolve_model(model_name: Optional[str], assets: Optional[str],
                  allow_mismatch: bool = False) -> Tuple[str, str]:
    """(model name, checkpoint directory).  The name is the directory's
    name — study_dir() and llm_board.board_paths() key on it too — so a
    --model-name that disagrees with --assets is refused (it would write
    another model's project, library and board weights under this name)."""
    if assets is None:
        if model_name in (None, DEFAULT_MODEL):
            assets = lp.default_assets()          # SMOLLM_ASSETS or the checkout's 135M
        elif model_name == vp.MODEL:
            assets = vp.default_assets()
        else:
            assets = os.path.join(os.path.dirname(lp.default_assets()), model_name)
    assets = os.path.normpath(os.path.abspath(assets))
    if not os.path.isfile(os.path.join(assets, "config.json")):
        raise SystemExit(f"no checkpoint at {assets} (config.json missing): pass --assets DIR "
                         f"(fetch it with scripts/llm_calibrate.py fetch <model>)")
    name = os.path.basename(assets)
    if model_name is None:
        model_name = name
    elif model_name != name and not allow_mismatch:
        raise SystemExit(f"--model-name {model_name} but the checkpoint is {assets} ({name}): the "
                         f"model name names the project, the library and the board's weights "
                         f"directory — drop --model-name, or pass --allow-name-mismatch")
    return model_name, assets


def check_out_dir(out: str, model_name: str, force: bool = False) -> None:
    """Refuse to wipe an output directory that holds another model's project
    or something that is not a generated project (no project.json)."""
    if force or not os.path.isdir(out) or not os.listdir(out):
        return
    pj = os.path.join(out, "project.json")
    if not os.path.isfile(pj):
        raise SystemExit(f"{out} exists and is not a generated project (no project.json); "
                         f"generating would delete its contents — pass another --out-dir, or "
                         f"--force")
    with open(pj) as f:
        other = json.load(f).get("model")
    if other != model_name:
        raise SystemExit(f"{out} holds the project of {other!r}, not {model_name!r} — pass "
                         f"--out-dir, or --force to replace it")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=None,
                    help="default: build/llm_project (SmolLM2-135M), "
                         "build/llm_project_<model> for another --model-name")
    ap.add_argument("--assets", default=None,
                    help="checkpoint directory (default: assets/<--model-name>, else SmolLM2-135M)")
    ap.add_argument("--formats", default=None)
    ap.add_argument("--vision-formats", default=None,
                    help="a VLM's vision formats (default: its study dir)")
    ap.add_argument("--buckets", default=",".join(map(str, lp.BUCKETS)))
    ap.add_argument("--context", type=int, default=lp.CONTEXT)
    ap.add_argument("--prefill-engine", choices=("conv", "matmul"), default="conv")
    ap.add_argument("--prefill-attn", choices=("fpga", "host"), default=lp.PREFILL_ATTN,
                    help="prefill attention on ConvKernel (fpga) or the host xattn region")
    ap.add_argument("--model-name", default=None,
                    help="default: the --assets directory's name (it must match)")
    ap.add_argument("--allow-name-mismatch", action="store_true",
                    help="accept a --model-name that differs from the --assets directory's name")
    ap.add_argument("--force", action="store_true",
                    help="replace an --out-dir that holds another model's project or other files")
    ap.add_argument("--no-weights", action="store_true", help="skip writing weights/*.dat")
    ap.add_argument("--driver-dirs", default=None,
                    help="JSON {kernel: dir}; default: local.driver_dirs of "
                         "demo/bert_squad/bert_squad_config.json")
    add_plan_args(ap)
    args = ap.parse_args(argv)
    args.model_name, args.assets = resolve_model(args.model_name, args.assets,
                                                 args.allow_name_mismatch)
    if args.out_dir is None:
        tag = args.model_name.removesuffix("-instruct").replace("-", "_").replace(".", "_")
        args.out_dir = (DEFAULT_OUT if args.model_name == DEFAULT_MODEL
                        else f"{DEFAULT_OUT}_{tag}")
    check_out_dir(os.path.abspath(args.out_dir), args.model_name, args.force)
    t0 = time.time()
    buckets = sorted(int(b) for b in args.buckets.split(","))
    vcfg = None
    if args.assets and vp.is_vlm(args.assets):
        if args.prefill_attn != "fpga":
            raise SystemExit("a VLM needs --prefill-attn fpga")
        m = vp.load(args.assets, args.formats, args.vision_formats)
        cfg, vcfg = m.tcfg, m.vcfg
        fe, fe_v = vp.frontends(m, ctx=args.context, name=args.model_name)
        formats_paths = m.formats_paths
        print(f"frontend: VLM, vision {vcfg.L} layers, hidden {vcfg.D}, {vcfg.N} patches -> "
              f"{vcfg.n_img} image tokens; text {cfg.L} layers, hidden {cfg.D}, heads "
              f"{cfg.H}/{cfg.KV}, vocab {cfg.V}, context {args.context}, buckets {buckets}",
              flush=True)
        models = vp.entry_models(fe, fe_v, buckets)
        del m
    else:
        cfg, W, fmt, _fd = lp.load_model(args.assets, args.formats)
        fe = lp.frontend(cfg, W, fmt, ctx=args.context, name=args.model_name,
                         prefill_attn=args.prefill_attn)
        formats_paths = (os.path.abspath(args.formats or lp.default_formats(args.assets)),)
        print(f"frontend: {cfg.L} layers, hidden {cfg.D}, heads {cfg.H}/{cfg.KV}, vocab {cfg.V}, "
              f"context {args.context}, buckets {buckets}, prefill on {args.prefill_engine}, "
              f"prefill attention {args.prefill_attn}", flush=True)
        models = lp.entry_models(fe, buckets)
        del W
    entries = build_entries(models, args.prefill_engine,
                            log=lambda m: print(m, flush=True), plan=plan_options_from_args(args))
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
    emit_glue(out, mg, args.model_name, cfg, args.context, buckets, vcfg)
    patch_cmake(out, mg)
    layers = write_layers(out, mg)
    summary.update({
        "model": args.model_name, "context": args.context, "buckets": buckets,
        "prefill_engine": args.prefill_engine, "prefill_attn": args.prefill_attn,
        "policy": lp.POLICIES[args.prefill_attn],
        "assets": os.path.abspath(args.assets or lp.default_assets()),
        "formats": [os.path.abspath(p) for p in formats_paths] if vcfg else formats_paths[0],
        "config": {"layers": cfg.L, "hidden": cfg.D, "heads": cfg.H, "kv_heads": cfg.KV,
                   "head_dim": cfg.HD, "ffn": cfg.FF, "vocab": cfg.V},
        "vision": ({"layers": vcfg.L, "hidden": vcfg.D, "heads": vcfg.H, "image": vcfg.S,
                    "patch": vcfg.P, "image_tokens": vcfg.n_img} if vcfg else None),
        "kinds": {k: sum(1 for L in layers if L["kind"] == k) for k in KINDS},
        "missing_drivers": missing, "targets": ["llm_bench", "smollm2"],
        "generated_s": round(time.time() - t0, 1)})
    with open(os.path.join(out, "project.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"project: {out}")
    print(f"  pool {summary['pool_bytes'] / 2**20:.1f} MiB (weights {summary['weights_bytes'] / 2**20:.1f},"
          f" KV caches {summary['dma_state_bytes'] / 2**20:.1f},"
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
