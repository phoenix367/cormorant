#!/usr/bin/env python3
"""generate_asr_project.py — a Moonshine checkpoint + its calibrated formats ->
the KV260 project behind libmoonshine_<size>.so (doc/plans/ASR_PLAN.md §5).

  1. src/moonshine.py: an ``encode_<T>`` entry per bucket (T encoder frames,
     384 T + 511 samples; the audio's valid length a runtime input) and the
     ``decode`` entry (one greedy step), every exponent from the formats of
     asr_study.py formats
  2. inference-scheduler: OnnxGraph + MultiEntryGenerator -> <out>/ (CMake
     project, weights/*.dat); the largest bucket plans the encoder's MatMul
     weight layouts and the others reuse them, so the project keeps one copy
     of every weight
  3. driver/ for the four kernels the entries drive, test/asr_api.{c,h} +
     test/asr_bench.c + test/asr_glue.h (inference_init with the project's
     kernels, the buckets, their run functions, the model's constants),
     CMake targets asr_bench and moonshine_<size> (the shared library: only
     the asr_* symbols exported), project.json, layers.json
  4. weights/tokenizer.json (the checkpoint's) for the caller's detokenizer

usage: inference-scheduler/.venv/bin/python demo/asr/scripts/generate_asr_project.py
           [--model moonshine-tiny] [--assets DIR] [--formats JSON]
           [--buckets 128,256,512,1024] [--out-dir demo/asr/build/<model>_project]
           [--driver-dirs JSON]
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
from src.llama import load_safetensors                  # noqa: E402
from src.matmul_lowering import shared_weight_layouts    # noqa: E402
from src.moonshine import BUCKETS, MoonshineConfig, MoonshineFormats, MoonshineFrontend, samples   # noqa: E402

ASSETS = os.path.join(REPO, "demo", "asr", "assets")
SRC = os.path.join(REPO, "demo", "asr", "src")
C_SOURCES = ("asr_api.c", "asr_api.h", "asr_bench.c")    # + the generated test/asr_glue.h
SAMPLE_RATE = 16000


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


def lib_name(model: str) -> str:
    """The C name of a model: moonshine-tiny -> moonshine_tiny (libmoonshine_tiny.so)."""
    return model.replace("-", "_").replace(".", "_")


def default_formats(model: str) -> str:
    return os.path.join(REPO, "demo", "asr", f"{model}_opt_formats.json")


def load(assets_dir: str, formats: str):
    """(MoonshineConfig, weights by Hugging Face name, MoonshineFormats)."""
    cfg = MoonshineConfig.from_file(os.path.join(assets_dir, "config.json"))
    W = load_safetensors(os.path.join(assets_dir, "model.safetensors"))
    return cfg, W, MoonshineFormats.from_file(formats, cfg)


def entry_graphs(cfg: MoonshineConfig, W: dict, fmt: MoonshineFormats, buckets=BUCKETS,
                 name: str = "moonshine") -> list:
    """[(entry, OnnxGraph)] of the library in project order: ``encode_<T>`` for every
    bucket (ascending), then ``decode``.  The largest bucket plans the encoder's
    MatMul weight layouts (ConvKernel's kernel widths), the others reuse them (one
    copy of every weight, as src/llm_entries.py); the decoder's weights are its own."""
    fe = MoonshineFrontend(cfg, W, fmt, name=name)
    enc, kw = {}, None
    for T in sorted(buckets, reverse=True):
        g = OnnxGraph(fe.encode_entry(T), matmul_conv_kw=kw)
        if kw is None:
            kw = shared_weight_layouts(g.nodes)
        enc[T] = g
    return [(f"encode_{T}", enc[T]) for T in sorted(buckets)] + [("decode", OnnxGraph(fe.decode_entry()))]


def write_glue(out: str, cfg: MoonshineConfig, buckets, active) -> None:
    """test/asr_glue.h: inference_init() with the project's kernels (``active``:
    the generator's active kernels), the buckets, a dispatcher to their run
    functions and the model's constants."""
    bs = sorted(buckets)
    cases = "\n".join(f"    case {i}: inference_run_encode_{T}(pcm, n, pos); return;" for i, T in enumerate(bs))
    text = (
        "/* asr_glue.h - generated by demo/asr/scripts/generate_asr_project.py: the encoder's\n"
        " * encode_<T> entries (T frames, 384 T + 511 samples) and the decode step. */\n"
        "#ifndef ASR_GLUE_H\n#define ASR_GLUE_H\n\n"
        f"#define ASR_N_BUCKETS  {len(bs)}\n"
        f"#define ASR_MAX_FRAMES {max(bs)}\n"
        f"#define ASR_MAX_SAMPLES {samples(max(bs))}   /* 384 x ASR_MAX_FRAMES + 511 */\n"
        f"#define ASR_BOS        {cfg.bos}\n"
        f"#define ASR_EOS        {cfg.eos}\n"
        f"#define ASR_MAX_POS    {cfg.max_pos}   /* the decoder's positions */\n"
        f"#define ASR_VOCAB      {cfg.V}\n"
        f"static const int asr_buckets[ASR_N_BUCKETS] = {{ {', '.join(str(b) for b in bs)} }};\n\n"
        "/* inference_init() with every kernel the project drives (" + ", ".join(kd.name for kd in active)
        + ") */\nstatic int asr_glue_init(void)\n{\n"
        f"    return inference_init({', '.join(kd.instance_macro for kd in active)});\n}}\n\n"
        "/* pcm: 384 x asr_buckets[b] + 511 samples, zero past n */\n"
        "static void asr_glue_encode(int b, const int16_t *pcm, const int32_t *n, const int32_t *pos)\n"
        "{\n    switch (b) {\n" + cases + "\n    default: return;\n    }\n}\n\n"
        "static void asr_glue_decode(const int32_t *ids, const int32_t *pos, const int32_t *xpos, int32_t *id)\n"
        "{\n    inference_run_decode(ids, pos, xpos, id);\n}\n\n#endif\n")
    with open(os.path.join(out, "test", "asr_glue.h"), "w") as f:
        f.write(text)


def patch_cmake(out: str, cg, model: str, lib: str) -> None:
    path = os.path.join(out, "CMakeLists.txt")
    text = open(path).read()
    macros = "\n".join(f"            {kd.instance_macro}" for kd in cg._active_kernels)
    text += (
        "\n"
        "# -----------------------------------------------------------------------\n"
        f"# asr_bench — the board gate runner, and {lib} — lib{lib}.so for the chat\n"
        "# server, both over test/asr_api.c (added by\n"
        "# demo/asr/scripts/generate_asr_project.py).  Only the asr_* symbols of the\n"
        "# shared library are exported.\n"
        "# -----------------------------------------------------------------------\n"
        "if(INFERENCE_TARGET STREQUAL \"LINUX\")\n"
        "    set_property(TARGET inference PROPERTY POSITION_INDEPENDENT_CODE ON)\n"
        "    add_executable(asr_bench test/asr_bench.c test/asr_api.c)\n"
        "    target_link_libraries(asr_bench PRIVATE inference m pthread)\n"
        f"    add_library({lib} SHARED test/asr_api.c)\n"
        f"    target_link_libraries({lib} PRIVATE inference m pthread)\n"
        f"    target_link_options({lib} PRIVATE -Wl,--exclude-libs,ALL -Wl,--no-undefined)\n"
        f"    foreach(_tgt IN ITEMS asr_bench {lib})\n"
        "        target_include_directories(${_tgt} PRIVATE test)\n"
        "        target_compile_options(${_tgt} PRIVATE -Wall -Wextra -O2)\n"
        "        target_compile_definitions(${_tgt} PRIVATE\n"
        "            ASR_API_WEIGHTS_DIR=\"${INFERENCE_WEIGHTS_DIR}\"\n"
        f"            ASR_MODEL_NAME=\"{model}\")\n"
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


def generate(out: str, cfg: MoonshineConfig, W: dict, fmt: MoonshineFormats, model: str, buckets=BUCKETS,
             driver_dirs: dict = None, tokenizer: str = None, extra: dict = None) -> dict:
    """Write the project into ``out`` (replaced); returns project.json's contents."""
    t0 = time.time()
    lib = lib_name(model)
    entries = entry_graphs(cfg, W, fmt, buckets, name=lib)
    cg = MultiEntryGenerator(entries, lib)
    out = os.path.abspath(out)
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)
    summary = cg.write_project(out)
    missing = populate_drivers(out, driver_dirs or {}, cg._active_kernels)
    for name in C_SOURCES:
        shutil.copy2(os.path.join(SRC, name), os.path.join(out, "test", name))
    write_glue(out, cfg, buckets, cg._active_kernels)
    patch_cmake(out, cg, model, lib)
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
    summary.update({"model": model, "library": f"lib{lib}.so", "buckets": sorted(buckets),
                    "max_samples": samples(max(buckets)), "sample_rate": SAMPLE_RATE,
                    "bos": cfg.bos, "eos": cfg.eos, "max_pos": cfg.max_pos, "vocab": cfg.V,
                    "nodes": {n: len(eg.nodes) for n, eg in entries},
                    "missing_drivers": missing, "generated_s": round(time.time() - t0, 1)})
    summary.update(extra or {})
    json.dump(summary, open(os.path.join(out, "project.json"), "w"), indent=2)
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    ap.add_argument("--model", default="moonshine-tiny", help="moonshine-tiny or moonshine-base")
    ap.add_argument("--assets", default=None, help="the checkpoint's directory (default demo/asr/assets/<model>)")
    ap.add_argument("--formats", default=None,
                    help="the calibrated exponents (default demo/asr/<model>_opt_formats.json)")
    ap.add_argument("--buckets", default=",".join(str(b) for b in BUCKETS),
                    help="encoder frames per encode entry: multiples of 64, at most 1024")
    ap.add_argument("--out-dir", default=None, help="default demo/asr/build/<model>_project")
    ap.add_argument("--driver-dirs", default=None,
                    help="JSON {kernel: dir}; default: local.driver_dirs of "
                         "demo/bert_squad/bert_squad_config.json")
    a = ap.parse_args(argv)
    assets = a.assets or os.path.join(ASSETS, a.model)
    formats = a.formats or default_formats(a.model)
    out = a.out_dir or os.path.join(REPO, "demo", "asr", "build", f"{a.model}_project")
    buckets = tuple(sorted(int(b) for b in a.buckets.split(",") if b))
    cfg, W, fmt = load(assets, formats)
    dd = json.loads(a.driver_dirs) if a.driver_dirs else driver_dirs_from_config(
        os.path.join(REPO, "demo", "bert_squad", "bert_squad_config.json"))
    tok = os.path.join(assets, "tokenizer.json")
    summary = generate(out, cfg, W, fmt, a.model, buckets, dd, tok if os.path.exists(tok) else None,
                       {"assets": os.path.abspath(assets), "formats": os.path.abspath(formats),
                        "formats_sha256": _sha256(formats)})
    print(f"project: {os.path.abspath(out)} ({sum(summary['nodes'].values())} nodes in {len(summary['nodes'])} "
          f"entries, pool {summary['pool_bytes'] / 2**20:.1f} MiB, {summary['generated_s']} s)"
          + (f"; missing drivers: {summary['missing_drivers']}" if summary["missing_drivers"] else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
