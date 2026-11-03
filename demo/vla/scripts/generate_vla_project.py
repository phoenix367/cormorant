#!/usr/bin/env python3
# Copyright 2026 Ivan Gubochkin
# SPDX-License-Identifier: Apache-2.0

"""generate_vla_project.py — a SmolVLA checkpoint + its calibrated formats ->
the KV260 project behind libsmolvla.so (doc/plans/VLA_PLAN.md).

  1. src/smolvla.py: the entries ``vision``, ``image_<c>`` (c < the config's
     cameras), ``prefix_<C>`` (C = 1 .. cameras) and ``expert``, every
     exponent from the formats of vla_study.py formats
  2. inference-scheduler: OnnxGraph + MultiEntryGenerator -> <out>/ (CMake
     project, weights/*.dat); the prefix entry of the most cameras plans the
     VLM's MatMul weight layouts and the others reuse them (one copy of every
     weight)
  3. driver/ for the kernels the entries drive, test/vla_api.{c,h} +
     test/vla_bench.c + test/vla_glue.h (inference_init with the project's
     kernels, the entries' run functions, the model's constants), CMake targets
     vla_bench and smolvla (the shared library: only the vla_* symbols
     exported), project.json, layers.json
  4. weights/tokenizer.json (SmolVLM2's) for the caller's tokenizer

usage: inference-scheduler/.venv/bin/python demo/vla/scripts/generate_vla_project.py
           [--assets demo/vla/assets] [--formats JSON] [--cameras N]
           [--out-dir demo/vla/build/smolvla_base_project] [--driver-dirs JSON]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(REPO, "inference-scheduler"))

from src.codegen.multi import MultiEntryGenerator       # noqa: E402
from src.graph import OnnxGraph                         # noqa: E402
from src.checkpoint import load_safetensors             # noqa: E402
from src.matmul_lowering import shared_weight_layouts    # noqa: E402
from src.smolvla import SmolVLAConfig, SmolVLAFormats, SmolVLAFrontend   # noqa: E402

ASSETS = os.path.join(REPO, "demo", "vla", "assets")
SRC = os.path.join(REPO, "demo", "vla", "src")
C_SOURCES = ("vla_api.c", "vla_api.h", "vla_bench.c")    # + the generated test/vla_glue.h
POLICY = "pow2+p12+vgelu+vsmx+rowops"                    # the formats' policy (smolvla_q.POLICIES)


def driver_dirs_from_config(path: str) -> dict:
    """local.driver_dirs of demo/bert_squad/bert_squad_config.json."""
    if not os.path.exists(path):
        return {}
    cfg = json.load(open(path))
    base = os.path.dirname(os.path.abspath(path))
    out = {}
    for k, v in (cfg.get("local", {}).get("driver_dirs") or {}).items():
        if k.startswith("_") or not v:
            continue
        v = os.path.expanduser(v)
        out[k] = v if os.path.isabs(v) else os.path.normpath(os.path.join(base, v))
    return out


def populate_drivers(out: str, driver_dirs: dict, active) -> list:
    """driver/: the active kernels' driver files; returns the missing ones."""
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


def default_formats() -> str:
    return os.path.join(REPO, "demo", "vla", f"smolvla_base_{POLICY}_formats.json")


def load(assets: str, formats: str):
    """(SmolVLAConfig, the checkpoint's tensors, SmolVLAFormats)."""
    cfg = SmolVLAConfig.from_files(os.path.join(assets, "smolvla_base", "config.json"),
                                   os.path.join(assets, "smolvlm2-500m-video-instruct", "config.json"))
    W = load_safetensors(os.path.join(assets, "smolvla_base", "model.safetensors"))
    return cfg, W, SmolVLAFormats.from_file(formats, cfg)


def entry_graphs(cfg: SmolVLAConfig, W: dict, fmt: SmolVLAFormats, cameras: int, name: str = "smolvla") -> list:
    """[(entry, OnnxGraph)] in project order: vision, image_<c>, prefix_<C> (ascending),
    expert.  The prefix of the most cameras plans the VLM's MatMul layouts."""
    fe = SmolVLAFrontend(cfg, W, fmt, name=name)
    pre, kw = {}, None
    for C in range(cameras, 0, -1):
        g = OnnxGraph(fe.prefix_entry(C), matmul_conv_kw=kw)
        if kw is None:
            kw = shared_weight_layouts(g.nodes)
        pre[C] = g
    out = [("vision", OnnxGraph(fe.vision_entry()))]
    out += [(f"image_{c}", OnnxGraph(fe.image_entry(c))) for c in range(cameras)]
    out += [(f"prefix_{C}", pre[C]) for C in range(1, cameras + 1)]
    out.append(("expert", OnnxGraph(fe.expert_entry())))
    return out


def write_glue(out: str, cfg: SmolVLAConfig, cameras: int, active) -> None:
    """test/vla_glue.h: inference_init() with the project's kernels, the entries'
    run functions behind dispatchers, the model's constants."""
    v = cfg.vision["vision_config"]
    P, S = int(v["patch_size"]), int(v["image_size"])
    rows = ", ".join(str(cfg.prefix_rows(C)) for C in range(1, cameras + 1))
    img = "\n".join(f"    case {c}: inference_run_image_{c}(); return 0;" for c in range(cameras))
    pre = "\n".join(f"    case {C}: inference_run_prefix_{C}(ids, n, state, zero); return 0;"
                    for C in range(1, cameras + 1))
    text = (
        "/* vla_glue.h - generated by demo/vla/scripts/generate_vla_project.py: the entries vision,\n"
        " * image_<c>, prefix_<C> and expert of src/smolvla.py. */\n"
        "#ifndef VLA_GLUE_H\n#define VLA_GLUE_H\n\n"
        f"#define VLA_CAMERAS     {cameras}\n"
        f"#define VLA_IMAGE       {S}      /* the vision encoder's square input */\n"
        f"#define VLA_PATCH       {P}\n"
        f"#define VLA_PATCHES     {(S // P) ** 2}\n"
        f"#define VLA_IMG_TOKENS  {cfg.img}\n"
        f"#define VLA_TOK_MAX     {cfg.tok_max}      /* language ids */\n"
        f"#define VLA_STATE       {cfg.max_state}      /* the state vector (zero padded) */\n"
        f"#define VLA_ACTION      {cfg.max_action}      /* the action vector */\n"
        f"#define VLA_STATE_DIM   {cfg.state_dim}      /* the config's (the robot's) state values */\n"
        f"#define VLA_ACTION_DIM  {cfg.action_dim}      /* the config's (the robot's) action values */\n"
        f"#define VLA_CHUNK       {cfg.chunk}      /* actions per chunk */\n"
        f"#define VLA_STEPS       {cfg.steps}      /* Euler steps */\n"
        f"#define VLA_VOCAB       {cfg.V}\n"
        f"static const int vla_prefix_rows[VLA_CAMERAS] = {{ {rows} }};\n\n"
        "/* inference_init() with every kernel the project drives (" + ", ".join(kd.name for kd in active)
        + ") */\nstatic int vla_glue_init(void)\n{\n"
        f"    return inference_init({', '.join(kd.instance_macro for kd in active)});\n}}\n\n"
        "static void vla_glue_vision(inference_buf_t *patches)\n{\n    inference_run_vision(patches);\n}\n\n"
        "static int vla_glue_image(int c)\n{\n    switch (c) {\n" + img + "\n    default: return -1;\n    }\n}\n\n"
        "static int vla_glue_prefix(int C, const int32_t *ids, const int32_t *n, const float *state, "
        "const int32_t *zero)\n{\n    switch (C) {\n" + pre + "\n    default: return -1;\n    }\n}\n\n"
        "static void vla_glue_expert(const float *x, const int32_t *step, const int32_t *pos, const int32_t *n,\n"
        "                            const int32_t *xpos, const int32_t *zero, float *v)\n{\n"
        "    inference_run_expert(x, step, pos, n, xpos, zero, v);\n}\n\n#endif\n")
    with open(os.path.join(out, "test", "vla_glue.h"), "w") as f:
        f.write(text)


def patch_cmake(out: str, cg, lib: str) -> None:
    path = os.path.join(out, "CMakeLists.txt")
    text = open(path).read()
    macros = "\n".join(f"            {kd.instance_macro}" for kd in cg._active_kernels)
    text += (
        "\n"
        "# -----------------------------------------------------------------------\n"
        f"# vla_bench — the board gate runner, and {lib} — lib{lib}.so, both over\n"
        "# test/vla_api.c (added by demo/vla/scripts/generate_vla_project.py).  Only\n"
        "# the vla_* symbols of the shared library are exported.\n"
        "# -----------------------------------------------------------------------\n"
        "if(INFERENCE_TARGET STREQUAL \"LINUX\")\n"
        "    set_property(TARGET inference PROPERTY POSITION_INDEPENDENT_CODE ON)\n"
        "    add_executable(vla_bench test/vla_bench.c test/vla_api.c)\n"
        "    target_link_libraries(vla_bench PRIVATE inference m pthread)\n"
        f"    add_library({lib} SHARED test/vla_api.c)\n"
        f"    target_link_libraries({lib} PRIVATE inference m pthread)\n"
        f"    target_link_options({lib} PRIVATE -Wl,--exclude-libs,ALL -Wl,--no-undefined)\n"
        f"    foreach(_tgt IN ITEMS vla_bench {lib})\n"
        "        target_include_directories(${_tgt} PRIVATE test)\n"
        "        target_compile_options(${_tgt} PRIVATE -Wall -Wextra -O2)\n"
        "        target_compile_definitions(${_tgt} PRIVATE\n"
        "            VLA_API_WEIGHTS_DIR=\"${INFERENCE_WEIGHTS_DIR}\")\n"
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


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def generate(out: str, cfg: SmolVLAConfig, W: dict, fmt: SmolVLAFormats, cameras: int = None, lib: str = "smolvla",
             driver_dirs: dict = None, tokenizer: str = None, extra: dict = None) -> dict:
    """Write the project into ``out`` (replaced); returns project.json's contents."""
    t0 = time.time()
    cameras = cfg.cameras if cameras is None else int(cameras)
    entries = entry_graphs(cfg, W, fmt, cameras, name=lib)
    cg = MultiEntryGenerator(entries, lib)
    out = os.path.abspath(out)
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)
    summary = cg.write_project(out)
    missing = populate_drivers(out, driver_dirs or {}, cg._active_kernels)
    for name in C_SOURCES:
        shutil.copy2(os.path.join(SRC, name), os.path.join(out, "test", name))
    write_glue(out, cfg, cameras, cg._active_kernels)
    patch_cmake(out, cg, lib)
    if tokenizer:
        shutil.copy2(tokenizer, os.path.join(out, "weights", "tokenizer.json"))
    layers, base = [], 0                  # MultiEntryGenerator numbers the nodes globally
    for ename, eg in entries:
        if eg.nodes[0].index != base:
            raise RuntimeError(f"entry {ename}: first layer {eg.nodes[0].index}, expected {base}")
        layers += [{"i": sn.index, "entry": ename, "name": sn.onnx_node.name, "op": sn.onnx_node.op_type,
                    "kind": "conv" if sn.kernel_name == "ConvKernel" else sn.onnx_node.op_type}
                   for sn in eg.nodes]
        base += len(eg.nodes)
    json.dump(layers, open(os.path.join(out, "layers.json"), "w"), indent=0)
    summary.update({"model": "smolvla_base", "library": f"lib{lib}.so", "cameras": cameras,
                    "chunk": cfg.chunk, "steps": cfg.steps, "max_action": cfg.max_action,
                    "max_state": cfg.max_state, "tok_max": cfg.tok_max, "image": cfg.image_size,
                    "nodes": {n: len(eg.nodes) for n, eg in entries},
                    "missing_drivers": missing, "generated_s": round(time.time() - t0, 1)})
    summary.update(extra or {})
    json.dump(summary, open(os.path.join(out, "project.json"), "w"), indent=2)
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    ap.add_argument("--assets", default=ASSETS, help="smolvla_base/ and smolvlm2-500m-video-instruct/")
    ap.add_argument("--formats", default=None, help=f"the calibrated exponents (default demo/vla/smolvla_base_"
                                                    f"{POLICY}_formats.json)")
    ap.add_argument("--cameras", type=int, default=None, help="cameras the library takes (default: the config's)")
    ap.add_argument("--out-dir", default=os.path.join(REPO, "demo", "vla", "build", "smolvla_base_project"))
    ap.add_argument("--driver-dirs", default=None,
                    help="JSON {kernel: dir}; default: local.driver_dirs of demo/bert_squad/bert_squad_config.json")
    a = ap.parse_args(argv)
    formats = a.formats or default_formats()
    cfg, W, fmt = load(a.assets, formats)
    dd = json.loads(a.driver_dirs) if a.driver_dirs else driver_dirs_from_config(
        os.path.join(REPO, "demo", "bert_squad", "bert_squad_config.json"))
    tok = os.path.join(a.assets, "smolvlm2-500m-video-instruct", "tokenizer.json")
    summary = generate(a.out_dir, cfg, W, fmt, a.cameras, driver_dirs=dd, tokenizer=tok if os.path.exists(tok) else None,
                       extra={"assets": os.path.abspath(a.assets), "formats": os.path.abspath(formats),
                              "formats_sha256": _sha256(formats)})
    print(f"project: {os.path.abspath(a.out_dir)} ({sum(summary['nodes'].values())} nodes in "
          f"{len(summary['nodes'])} entries, pool {summary['pool_bytes'] / 2**20:.1f} MiB, {summary['generated_s']} s)"
          + (f"; missing drivers: {summary['missing_drivers']}" if summary["missing_drivers"] else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
