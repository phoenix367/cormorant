#!/usr/bin/env python3
"""generate_tts_project.py — the Piper voice + calibrated exponents -> the KV260
project behind libpiper_tts.so (doc/plans/TTS_PLAN.md §4).

  1. src/piper.py: the ``chunk`` entry (flow + HiFi-GAN decoder of 128
     frames, 1.49 s of audio) with every exponent of piper_study.py
     calibrate
  2. inference-scheduler: OnnxGraph + CodeGenerator -> <out>/ (CMake
     project, weights/*.dat)
  3. driver/ for ConvKernel, test/tts_api.{c,h} + test/tts_bench.c, CMake
     targets tts_bench and piper_tts (libpiper_tts.so: only the tts_*
     symbols exported), project.json, layers.json
  4. weights/frontend.npz + weights/voice.json: the host front end of the
     chat server (demo/chat/piper_backend.py)

usage: inference-scheduler/.venv/bin/python demo/tts/scripts/generate_tts_project.py
           [--assets DIR] [--out-dir demo/tts/build/piper_project] [--driver-dirs JSON]
"""

from __future__ import annotations

import argparse
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
from src.piper import PiperChunkFrontend, entry_info, load_weights   # noqa: E402

DEFAULT_ASSETS = os.path.join(REPO, "demo", "tts", "assets", "piper-lessac-medium")
DEFAULT_OUT = os.path.join(REPO, "demo", "tts", "build", "piper_project")
SRC = os.path.join(REPO, "demo", "tts", "src")
C_SOURCES = ("tts_api.c", "tts_api.h", "tts_bench.c")


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


def write_frontend(out: str, W: dict, voice_json: str) -> None:
    """weights/frontend.npz (the text encoder's and the duration predictor's
    weights, float32 as in the voice) and weights/voice.json (phoneme ids,
    espeak voice, sampling defaults) for the chat server's host front end
    (piper_vits.front_end; demo/chat/piper_backend.py)."""
    import numpy as np
    keys = sorted(k for k in W if k == "emb" or k.startswith(("enc_p.", "dp.")))
    np.savez(os.path.join(out, "weights", "frontend.npz"),
             **{k: np.asarray(W[k], np.float32) for k in keys})
    cfg = json.load(open(voice_json))
    voice = {k: cfg[k] for k in ("audio", "espeak", "inference", "phoneme_id_map", "language", "dataset")
             if k in cfg}
    voice["model"] = "piper-lessac-medium"
    with open(os.path.join(out, "weights", "voice.json"), "w") as f:
        json.dump(voice, f, indent=1, ensure_ascii=False)


def patch_cmake(out: str, cg, info: dict) -> None:
    path = os.path.join(out, "CMakeLists.txt")
    text = open(path).read()
    macros = "\n".join(f"            {kd.instance_macro}" for kd in cg._active_kernels)
    text += (
        "\n"
        "# -----------------------------------------------------------------------\n"
        "# tts_bench — the board gate runner, and piper_tts — libpiper_tts.so for\n"
        "# the chat server, both over test/tts_api.c (added by\n"
        "# demo/tts/scripts/generate_tts_project.py).  Only the tts_* symbols of\n"
        "# the shared library are exported.\n"
        "# -----------------------------------------------------------------------\n"
        "if(INFERENCE_TARGET STREQUAL \"LINUX\")\n"
        "    set_property(TARGET inference PROPERTY POSITION_INDEPENDENT_CODE ON)\n"
        "    add_executable(tts_bench test/tts_bench.c test/tts_api.c)\n"
        "    target_link_libraries(tts_bench PRIVATE inference m)\n"
        "    add_library(piper_tts SHARED test/tts_api.c)\n"
        "    target_link_libraries(piper_tts PRIVATE inference m)\n"
        "    target_link_options(piper_tts PRIVATE -Wl,--exclude-libs,ALL -Wl,--no-undefined)\n"
        "    foreach(_tgt IN ITEMS tts_bench piper_tts)\n"
        "        target_include_directories(${_tgt} PRIVATE test)\n"
        "        target_compile_options(${_tgt} PRIVATE -Wall -Wextra -O2)\n"
        "        target_compile_definitions(${_tgt} PRIVATE\n"
        "            TTS_API_WEIGHTS_DIR=\"${INFERENCE_WEIGHTS_DIR}\"\n"
        f"            TTS_MODEL_NAME=\"{info['model_name']}\" TTS_SAMPLE_RATE={info['sample_rate']})\n"
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    ap.add_argument("--assets", default=DEFAULT_ASSETS)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--driver-dirs", default=None,
                    help="JSON {kernel: dir}; default: local.driver_dirs of "
                         "demo/bert_squad/bert_squad_config.json")
    a = ap.parse_args(argv)
    t0 = time.time()
    W = load_weights(os.path.join(a.assets, "en_US-lessac-medium.onnx"))
    E = json.load(open(os.path.join(a.assets, "exponents.json")))["exponents"]
    model = PiperChunkFrontend(W, E, name="piper_lessac_medium").entry()
    g = OnnxGraph(model, fuse_act=True, s2d_stem=True)
    cg = MultiEntryGenerator([("chunk", g)], "piper_lessac_medium")
    out = os.path.abspath(a.out_dir)
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)
    summary = cg.write_project(out)
    dd = json.loads(a.driver_dirs) if a.driver_dirs else driver_dirs_from_config(
        os.path.join(REPO, "demo", "bert_squad", "bert_squad_config.json"))
    missing = populate_drivers(out, dd, cg._active_kernels)
    for name in C_SOURCES:
        shutil.copy2(os.path.join(SRC, name), os.path.join(out, "test", name))
    info = entry_info(model)
    write_frontend(out, W, os.path.join(a.assets, "en_US-lessac-medium.onnx.json"))
    patch_cmake(out, cg, {"model_name": "piper-lessac-medium", "sample_rate": info["sample_rate"]})
    layers = [{"i": sn.index, "name": sn.onnx_node.name, "op": sn.onnx_node.op_type,
               "kind": "conv" if sn.kernel_name == "ConvKernel" else sn.onnx_node.op_type}
              for sn in g.nodes]
    json.dump(layers, open(os.path.join(out, "layers.json"), "w"), indent=0)
    summary.update({"model": "piper-lessac-medium", "entry": info, "assets": os.path.abspath(a.assets),
                    "nodes": len(g.nodes),
                    "missing_drivers": missing, "generated_s": round(time.time() - t0, 1)})
    json.dump(summary, open(os.path.join(out, "project.json"), "w"), indent=2)
    print(f"project: {out} ({len(g.nodes)} nodes, pool {summary['pool_bytes'] / 2**20:.1f} MiB, "
          f"{summary['generated_s']} s)" + (f"; missing drivers: {missing}" if missing else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
