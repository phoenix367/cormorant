#!/usr/bin/env python3
"""
generate_project.py — produce a model-specific KV260 inference project for
the image-classification demo.

Pipeline per model:
  1. Run inference-scheduler on the ONNX model → CMake C project under
     <out_dir>/projects/<model_name>/.
  2. Drop in driver/ files from the local Vivado-HLS build directories
     listed in image_classification_config.json.
  3. Disable the auto-generated test/test_inference.c smoke test.
  4. Copy demo/image_classification/src/classify_images.c into
     test/classify_images.c and emit a model-specific test/bench_glue.h
     that supplies:
       - BENCH_INPUT_NUMEL / BENCH_OUTPUT_NUMEL / BENCH_NUM_CLASSES
       - BENCH_MODEL_NAME
       - bench_inference_init()  (forwards to inference_init with the right
         number of UIO instance args for this model's active kernels)
       - bench_inference_run()
  5. Append an `add_executable(classify_images test/classify_images.c)` block
     to CMakeLists.txt.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

DEMO_DIR    = Path(__file__).resolve().parent.parent
REPO_ROOT   = DEMO_DIR.parent.parent
SCHED_DIR   = REPO_ROOT / "inference-scheduler"
ASSETS_DIR  = DEMO_DIR / "assets"
SRC_DIR     = DEMO_DIR / "src"

sys.path.insert(0, str(SCHED_DIR))

from src.graph   import OnnxGraph                 # noqa: E402
from src.codegen import CodeGenerator             # noqa: E402
from src.kernels import KERNEL_REGISTRY           # noqa: E402


_KERNEL_INIT_MACRO = {
    "VectorOPKernel": "INFERENCE_VECTOROPKERNEL_INSTANCE",
    "MatmulKernel":   "INFERENCE_MATMULKERNEL_INSTANCE",
    "ConvKernel":     "INFERENCE_CONVKERNEL_INSTANCE",
    "PoolKernel":     "INFERENCE_POOLKERNEL_INSTANCE",
}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _run_scheduler(model_path: Path, out_dir: Path) -> Tuple[OnnxGraph,
                                                              CodeGenerator]:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    graph = OnnxGraph(str(model_path), fuse_act=True,   # Relu/Clip(0,6) fused into the producing VectorOP call (VECTOROP_OPTIMISATION §3)
                      s2d_stem=True)                    # stride-2 RGB stem -> host SpaceToDepth + 4x4 s1 Conv (RESNET18_15FPS_PLAN step 1)
    gen   = CodeGenerator(graph=graph, model_path=str(model_path))

    from inference_scheduler import main as sched_main
    rc = sched_main(["--out-dir", str(out_dir), str(model_path)])
    if rc != 0:
        raise RuntimeError(f"inference-scheduler failed for {model_path} (rc={rc})")
    return graph, gen


def _populate_drivers(project_dir: Path, driver_dirs: Dict[str, str],
                      active: List[str]) -> List[str]:
    driver_dst = project_dir / "driver"
    driver_dst.mkdir(parents=True, exist_ok=True)
    missing: List[str] = []

    for kernel in active:
        kd  = KERNEL_REGISTRY[kernel]
        src_dir = driver_dirs.get(kernel)
        if not src_dir:
            for f in kd.driver_files:
                missing.append(f"{kernel}/{f}")
            continue
        src = (DEMO_DIR / src_dir).resolve() if not Path(src_dir).is_absolute() \
              else Path(src_dir)
        if not src.is_dir():
            for f in kd.driver_files:
                missing.append(f"{kernel}/{f}  (not under {src})")
            continue
        for f in kd.driver_files:
            sp = src / f
            if sp.exists():
                shutil.copy2(sp, driver_dst / f)
            else:
                missing.append(f"{kernel}/{f}")
    return missing


def _disable_default_test(project_dir: Path) -> None:
    test_file = project_dir / "test" / "test_inference.c"
    if test_file.exists():
        test_file.unlink()


def _patch_cmake_for_classifier(project_dir: Path) -> None:
    """Append a classify_images target to the generated CMakeLists.txt and
    disable the default smoke test."""
    cmake = project_dir / "CMakeLists.txt"
    text  = cmake.read_text()

    text = re.sub(
        r'option\(INFERENCE_BUILD_TEST\s+"[^"]+"\s+ON\)',
        'option(INFERENCE_BUILD_TEST "Build test_inference smoke test" OFF)',
        text,
        count=1,
    )

    classifier_block = (
        "\n"
        "# ──────────────────────────────────────────────────────────────────\n"
        "# classify_images — image-classification host (added by the\n"
        "# demo/image_classification project generator).\n"
        "# ──────────────────────────────────────────────────────────────────\n"
        "if(INFERENCE_TARGET STREQUAL \"LINUX\")\n"
        "    add_executable(classify_images test/classify_images.c)\n"
        "    target_link_libraries(classify_images PRIVATE inference m)\n"
        "    target_include_directories(classify_images PRIVATE test)\n"
        "    target_compile_options(classify_images PRIVATE -Wall -Wextra -O2)\n"
        "    if(DEFINED BENCH_DATA_DIR)\n"
        "        target_compile_definitions(classify_images PRIVATE\n"
        "            BENCH_DATA_DIR=\"${BENCH_DATA_DIR}\")\n"
        "    endif()\n"
        "    if(DEFINED BENCH_TOP_K)\n"
        "        target_compile_definitions(classify_images PRIVATE\n"
        "            BENCH_TOP_K=${BENCH_TOP_K})\n"
        "    endif()\n"
        "    if(DEFINED BENCH_LABEL_OFFSET)\n"
        "        target_compile_definitions(classify_images PRIVATE\n"
        "            BENCH_LABEL_OFFSET=${BENCH_LABEL_OFFSET})\n"
        "    endif()\n"
        "    foreach(_macro IN ITEMS\n"
        "            INFERENCE_VECTOROPKERNEL_INSTANCE\n"
        "            INFERENCE_MATMULKERNEL_INSTANCE\n"
        "            INFERENCE_CONVKERNEL_INSTANCE\n"
        "            INFERENCE_POOLKERNEL_INSTANCE)\n"
        "        if(DEFINED ${_macro})\n"
        "            target_compile_definitions(classify_images PRIVATE\n"
        "                ${_macro}=\"${${_macro}}\")\n"
        "        endif()\n"
        "    endforeach()\n"
        "endif()\n"
    )
    cmake.write_text(text + classifier_block)


def _emit_glue(project_dir: Path, *, model_name: str,
               graph: OnnxGraph, gen: CodeGenerator) -> None:
    inputs  = graph.input_tensors
    outputs = graph.output_tensors
    if len(inputs) != 1 or len(outputs) != 1:
        raise RuntimeError(
            f"classify_images expects exactly one input and one output tensor; "
            f"model has {len(inputs)} input(s) / {len(outputs)} output(s)")

    in_t,  out_t  = inputs[0], outputs[0]
    in_size_macro  = f"INFERENCE_{in_t.c_name.upper()}_SIZE"
    out_size_macro = f"INFERENCE_{out_t.c_name.upper()}_SIZE"

    if out_t.numel not in (1000, 1001):
        _log(f"  warning: output numel={out_t.numel}, expected 1000 "
             "(ONNX Model Zoo MobileNetV2 / ResNet — set "
             "labels.skip_background_class=true) or 1001 (TF "
             "MobileNetV1 with the synthetic 'background' slot)")

    active = [kd.name for kd in gen._active_kernels]
    init_args = ", ".join(_KERNEL_INIT_MACRO[k] for k in active)

    macro_defaults = "\n".join(
        f'#ifndef {_KERNEL_INIT_MACRO[k]}\n'
        f'#  define {_KERNEL_INIT_MACRO[k]} "{k}_0"\n'
        f'#endif'
        for k in active
    )

    glue = f"""/*
 * bench_glue.h — auto-generated bridge between classify_images.c and the
 *                model-specific inference API.
 *
 * Model       : {model_name}
 * Input  '{in_t.onnx_name}'  numel={in_t.numel}  shape={list(in_t.shape)}
 * Output '{out_t.onnx_name}'  numel={out_t.numel}  shape={list(out_t.shape)}
 * Active kernels: {", ".join(active) if active else "(none)"}
 */
#pragma once

#include "inference.h"

#define BENCH_MODEL_NAME    "{model_name}"
#define BENCH_INPUT_NUMEL   {in_size_macro}
#define BENCH_OUTPUT_NUMEL  {out_size_macro}
#define BENCH_NUM_CLASSES   {out_t.numel}u

{macro_defaults}

static inline int bench_inference_init(void) {{
    return inference_init({init_args});
}}

static inline void bench_inference_run(inference_buf_t *in,
                                       inference_buf_t *out) {{
    inference_run(in, out);
}}
"""
    (project_dir / "test" / "bench_glue.h").write_text(glue)


# ──────────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────────

def generate_for_model(*, model_name: str, model_path: Path,
                       project_dir: Path,
                       driver_dirs: Dict[str, str]) -> Dict[str, object]:
    _log(f"[{model_name}] scheduling {model_path.name}")
    graph, gen = _run_scheduler(model_path, project_dir)
    active = [kd.name for kd in gen._active_kernels]
    _log(f"[{model_name}] active kernels: {', '.join(active) or '(none)'}")

    missing = _populate_drivers(project_dir, driver_dirs, active)
    if missing:
        _log(f"[{model_name}] warning: missing driver files:")
        for m in missing[:8]:
            _log(f"    {m}")
        if len(missing) > 8:
            _log(f"    … and {len(missing) - 8} more")

    _disable_default_test(project_dir)
    _patch_cmake_for_classifier(project_dir)

    src_host = SRC_DIR / "classify_images.c"
    if not src_host.exists():
        raise FileNotFoundError(f"missing {src_host}")
    shutil.copy2(src_host, project_dir / "test" / "classify_images.c")

    _emit_glue(project_dir, model_name=model_name, graph=graph, gen=gen)

    return {
        "model_name":   model_name,
        "project_dir":  str(project_dir),
        "active":       active,
        "missing":      missing,
        "input_name":   graph.input_tensors[0].onnx_name,
        "output_name":  graph.output_tensors[0].onnx_name,
        "input_numel":  graph.input_tensors[0].numel,
        "output_numel": graph.output_tensors[0].numel,
    }


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _load_config(path: Path) -> dict:
    if not path.exists():
        from _config_help import missing_config_die
        missing_config_die(path)
    with open(path) as f:
        return json.load(f)


def _preflight(cfg: dict, *, models_filter: List[str] = None) -> bool:
    ok = True
    drivers = cfg.get("local", {}).get("driver_dirs", {})
    selected = set(models_filter or [])
    src_host = SRC_DIR / "classify_images.c"
    if not src_host.exists():
        _log(f"error: {src_host} missing")
        ok = False

    requested = [m for m in cfg.get("models", [])
                 if not selected or m["name"] in selected]
    if not requested:
        _log("error: no models selected/configured")
        return False

    for m in requested:
        onnx = ASSETS_DIR / "models" / m["drive_filename"]
        if not onnx.exists():
            _log(f"error: {onnx} not found — run download_assets.py first")
            ok = False

    if not drivers:
        _log("warning: local.driver_dirs is empty — generated projects will "
             "ship with empty driver/ folders and the on-board cmake will fail")
    else:
        for kernel, path in drivers.items():
            src = (DEMO_DIR / path).resolve() if not Path(path).is_absolute() \
                  else Path(path)
            if not src.is_dir():
                _log(f"warning: driver_dirs.{kernel}: {src} not found "
                     "(run `make synthesize_kv260` from the repo root)")
    return ok


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config",
                   default=str(DEMO_DIR / "image_classification_config.json"))
    p.add_argument("--out-dir", default=str(DEMO_DIR / "build" / "projects"),
                   help="root for generated projects (one subdir per model)")
    p.add_argument("--models", nargs="+",
                   help="restrict to a subset of models from the config (by name)")
    p.add_argument("--check-only", action="store_true",
                   help="validate config + assets and exit without generating")
    args = p.parse_args(argv)

    cfg = _load_config(Path(args.config))
    if not _preflight(cfg, models_filter=args.models):
        return 1
    if args.check_only:
        _log("preflight: ok")
        return 0

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    drivers = cfg.get("local", {}).get("driver_dirs", {})
    summaries = []

    selected = args.models
    for m in cfg.get("models", []):
        name = m["name"]
        if selected and name not in selected:
            continue
        onnx = (ASSETS_DIR / "models" / m["drive_filename"])
        if not onnx.exists():
            _log(f"[{name}] error: {onnx} not found — run download_assets.py first")
            return 1
        proj = out_root / name
        summaries.append(
            generate_for_model(
                model_name=name,
                model_path=onnx,
                project_dir=proj,
                driver_dirs=drivers,
            )
        )

    if not summaries:
        _log("error: no models selected/configured")
        return 1

    (out_root / "projects.json").write_text(json.dumps(summaries, indent=2))
    _log(f"wrote {len(summaries)} project(s) under {out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
