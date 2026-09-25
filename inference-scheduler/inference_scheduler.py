#!/usr/bin/env python3
"""
inference_scheduler.py — ONNX-to-VectorOPKernel project generator

Parses an ONNX model and writes a CMake project that compiles a static
library implementing the full inference loop on the Xilinx KV260, using
VectorOPKernel invocations via the XVectoropkernel driver API.

Supported ONNX operators:
  Add, Sub, Mul, Div, Relu, Clip(min=0, max=6)

Output project layout:
  <out_dir>/
  ├── CMakeLists.txt              INFERENCE_TARGET=BARE_METAL|LINUX
  ├── include/
  │   └── inference.h             public API: Data_t, sizes, init/run
  ├── src/
  │   └── inference.c             generated inference loop (dual-target)
  └── driver/                     XVectoropkernel driver sources
      ├── xvectoropkernel.h       ┐
      ├── xvectoropkernel_hw.h    │  copied from --driver-dir,
      ├── xvectoropkernel.c       │  or left empty with a README
      ├── xvectoropkernel_sinit.c │ (bare-metal)
      └── xvectoropkernel_linux.c ┘ (Linux)

Usage:
  python inference_scheduler.py model.onnx
  python inference_scheduler.py model.onnx --out-dir ./my_project
  python inference_scheduler.py model.onnx --out-dir ./my_project \\
      --driver-dir ../build/kv260/vadd_kv260/solution1/impl/ip/\\
                   drivers/VectorOPKernel_v1_0/src
"""

import argparse
import os
import shutil
import sys

# Allow running from the project root without installing the package
sys.path.insert(0, os.path.dirname(__file__))

from src.graph            import OnnxGraph
from src.codegen          import CodeGenerator
from src.codegen._simulate import LARGE_EXPECTED_THRESHOLD
from src.kernels           import KERNEL_REGISTRY, mixed_driver_readme
from src.nodes            import SchedulerError
from src.report           import ReportGenerator
from src.tensor           import LARGE_WEIGHT_THRESHOLD


# ---------------------------------------------------------------------------
# Backward-compat aliases (used by some existing tests)
# ---------------------------------------------------------------------------
_VECTOROP_DRIVER_FILES = list(KERNEL_REGISTRY["VectorOPKernel"].driver_files)
_MATMUL_DRIVER_FILES   = list(KERNEL_REGISTRY["MatmulKernel"].driver_files)
_DRIVER_FILES          = _VECTOROP_DRIVER_FILES
_VECTOROP_DRIVER_README = KERNEL_REGISTRY["VectorOPKernel"].driver_readme
_MATMUL_DRIVER_README   = KERNEL_REGISTRY["MatmulKernel"].driver_readme
_DRIVER_README          = _VECTOROP_DRIVER_README


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Generate a VectorOPKernel inference project from an ONNX model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "model",
        metavar="model.onnx",
        help="Input ONNX model file",
    )
    p.add_argument(
        "--out-dir",
        metavar="DIR",
        default=None,
        help=(
            "Output project directory (default: ./<model_stem>_inference/). "
            "Created if absent; existing files are overwritten."
        ),
    )
    p.add_argument(
        "--driver-dir",
        metavar="DIR",
        default=None,
        help=(
            "Path to the XVectoropkernel driver source directory "
            "(contains xvectoropkernel.c, .h, _hw.h, _sinit.c, _linux.c). "
            "If omitted, driver/ is left empty with a README."
        ),
    )
    p.add_argument(
        "--embed-large-weights",
        action="store_true",
        default=False,
        help=(
            f"Embed all weight tensors as C arrays in inference.c, even those "
            f"exceeding the {LARGE_WEIGHT_THRESHOLD}-element threshold that would "
            f"normally be written to external weights/*.dat files. "
            f"Produces a larger but fully self-contained C file."
        ),
    )
    p.add_argument(
        "--embed-large-expected",
        action="store_true",
        default=False,
        help=(
            f"Embed all GT expected arrays as C arrays in test_inference.c, even "
            f"those exceeding the {LARGE_EXPECTED_THRESHOLD}-element threshold that "
            f"would normally be written to external expected/*.dat files. "
            f"Produces a larger but fully self-contained test file."
        ),
    )
    p.add_argument(
        "--no-report",
        action="store_true",
        default=False,
        help="Skip writing report.md (the human-readable model summary).",
    )
    p.add_argument(
        "--no-fuse-act",
        dest="fuse_act",
        action="store_false",
        default=True,
        help=(
            "Do not fold Relu / Clip(0,6) nodes into the VectorOP call that "
            "produces their input (the kernel's `act` register).  Fusion is "
            "on by default: it halves the traffic of every Add->Relu pair."
        ),
    )
    p.add_argument(
        "--no-s2d-stem",
        dest="s2d_stem",
        action="store_false",
        default=True,
        help=(
            "Do not rewrite stride-2 Conv layers with <= 4 input channels "
            "(e.g. an RGB 7x7 s2 stem) as a host-side SpaceToDepth(2) plus a "
            "stride-1 Conv over 4x the channels.  The rewrite is on by "
            "default: it fills 4x more ConvKernel input-channel lanes and "
            "cuts the taps to a quarter."
        ),
    )
    p.add_argument(
        "--no-fuse-patterns",
        dest="fuse_patterns",
        action="store_false",
        default=True,
        help=(
            "Do not fuse transformer subgraphs (TensorFlow-style LayerNorm, "
            "GELU tanh / erf) into single host-CPU ops, and do not reshape "
            "constant VectorOP operands for the kernel's broadcast.  Fusion is "
            "on by default; it only changes graphs that contain these patterns."
        ),
    )
    p.add_argument(
        "--matmul-on-conv",
        dest="matmul_on_conv",
        choices=("auto", "always", "off"),
        default="auto",
        help=(
            "Run MatMuls on ConvKernel with swapped operand roles (A = conv "
            "weight, B = conv input, 1 x kw kernel with stride (1, kw); "
            "doc/BERT_PLAN.md 2A): 'auto' (default) wherever the engine cost "
            "model estimates ConvKernel faster than MatmulKernel, 'always' for "
            "every eligible MatMul, 'off' for none.  Batch-1 FC layers never "
            "qualify.  Bit-identical results either way."
        ),
    )
    p.add_argument(
        "--no-matmul-on-conv",
        dest="matmul_on_conv",
        action="store_const",
        const="off",
        help="Same as --matmul-on-conv off: every MatMul stays on MatmulKernel.",
    )
    return p.parse_args(argv)


def _write(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


def _copy_driver(src_dir: str, dst_dir: str, files: list = None) -> list:
    """
    Copy driver files from src_dir into dst_dir.
    Returns list of filenames that were not found (warnings).
    files defaults to _VECTOROP_DRIVER_FILES for backwards compatibility.
    """
    if files is None:
        files = _VECTOROP_DRIVER_FILES
    os.makedirs(dst_dir, exist_ok=True)
    missing = []
    for fname in files:
        src = os.path.join(src_dir, fname)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(dst_dir, fname))
        else:
            missing.append(fname)
    return missing


def main(argv=None):
    args = parse_args(argv)

    # ---------------------------------------------------------------- #
    # 1. Resolve output directory                                       #
    # ---------------------------------------------------------------- #
    if args.out_dir is None:
        stem    = os.path.splitext(os.path.basename(args.model))[0]
        out_dir = os.path.join(os.getcwd(), f"{stem}_inference")
    else:
        out_dir = os.path.abspath(args.out_dir)

    # ---------------------------------------------------------------- #
    # 2. Parse and validate the ONNX model                             #
    # ---------------------------------------------------------------- #
    try:
        graph = OnnxGraph(args.model, fuse_act=args.fuse_act, s2d_stem=args.s2d_stem,
                          fuse_patterns=args.fuse_patterns,
                          matmul_on_conv=args.matmul_on_conv)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except SchedulerError as e:
        print(f"error: unsupported graph — {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"error: failed to load ONNX model: {e}", file=sys.stderr)
        return 1

    # ---------------------------------------------------------------- #
    # 3. Print graph summary                                            #
    # ---------------------------------------------------------------- #
    inputs  = graph.input_tensors
    outputs = graph.output_tensors
    nodes   = graph.nodes

    print(f"Model      : {args.model}", file=sys.stderr)
    print(f"Inputs     : {[f'{t.onnx_name}{t.shape}' for t in inputs]}", file=sys.stderr)
    print(f"Outputs    : {[f'{t.onnx_name}{t.shape}' for t in outputs]}", file=sys.stderr)
    print(f"Nodes      : {len(nodes)}", file=sys.stderr)
    for sn in nodes:
        in_shapes = [str(t.shape) for t in sn.inputs]
        print(
            f"  [{sn.index:3d}] {sn.onnx_node.op_type:<12}"
            f" {' x '.join(in_shapes)} -> {sn.output.shape}",
            file=sys.stderr,
        )
    mc = graph.matmul_conv_stats
    if mc["lowered"]:
        print(f"MatMul->Conv: {mc['lowered']} MatMul(s) on ConvKernel "
              f"({mc['conv_calls']} calls, est {mc['conv_cycles'] / 1e5:.1f} ms vs "
              f"{mc['matmul_cycles'] / 1e5:.1f} ms on MatmulKernel @100 MHz), "
              f"{mc['kept']} on MatmulKernel", file=sys.stderr)
    print(f"Output dir : {out_dir}", file=sys.stderr)

    # ---------------------------------------------------------------- #
    # 4. Generate code                                                  #
    # ---------------------------------------------------------------- #
    try:
        gen        = CodeGenerator(graph=graph, model_path=args.model,
                                   embed_large_weights=args.embed_large_weights,
                                   embed_large_expected=args.embed_large_expected)
        header     = gen.generate_header()
        source     = gen.generate_source()
        buf_impl   = gen.generate_buf_impl()
        cmake      = gen.generate_cmake()
        setup_sh   = gen.generate_setup_script()
        test_src   = gen.generate_test()
    except SchedulerError as e:
        print(f"error: code generation failed — {e}", file=sys.stderr)
        return 1

    # ---------------------------------------------------------------- #
    # 5. Write project files                                            #
    # ---------------------------------------------------------------- #
    _write(os.path.join(out_dir, "CMakeLists.txt"),              cmake)
    _write(os.path.join(out_dir, "include", "inference.h"),      header)
    _write(os.path.join(out_dir, "src",     "inference.c"),      source)
    _write(os.path.join(out_dir, "src",     "inference_buf.c"),  buf_impl)
    _write(os.path.join(out_dir, "scripts", "check_inference_setup.sh"), setup_sh)
    _write(os.path.join(out_dir, "test",    "test_inference.c"), test_src)

    # ---------------------------------------------------------------- #
    # 5a. Runtime helper sources (static templates copied verbatim)     #
    # ---------------------------------------------------------------- #
    runtime_src = os.path.join(os.path.dirname(__file__), "runtime")
    runtime_files = [
        ("inference_prof.h",         os.path.join("include", "inference_prof.h")),
        ("inference_prof.c",         os.path.join("src",     "inference_prof.c")),
        ("inference_ddr.h",          os.path.join("include", "inference_ddr.h")),
        ("inference_ddr.c",          os.path.join("src",     "inference_ddr.c")),
        ("inference_ddr_backend.h",  os.path.join("src",     "inference_ddr_backend.h")),
        # DDR backends — one .c per platform under src/ddr/.
        ("ddr/zuplus_apm.c",         os.path.join("src", "ddr", "zuplus_apm.c")),
    ]
    for src_name, rel_dst in runtime_files:
        sp = os.path.join(runtime_src, src_name)
        dp = os.path.join(out_dir,    rel_dst)
        os.makedirs(os.path.dirname(dp), exist_ok=True)
        shutil.copy2(sp, dp)

    # ---------------------------------------------------------------- #
    # 5b. External weight .dat files (large tensors)                    #
    # ---------------------------------------------------------------- #
    large_weights = gen.large_weight_tensors
    if large_weights:
        weights_dir = os.path.join(out_dir, "weights")
        os.makedirs(weights_dir, exist_ok=True)
        for t in large_weights:
            dat_path = os.path.join(weights_dir, f"{t.c_name}.dat")
            with open(dat_path, "wb") as f:
                f.write(gen.generate_weight_dat(t))
        print(f"Weights    : {len(large_weights)} large weight(s) written to"
              f" {weights_dir}/", file=sys.stderr)

    # ---------------------------------------------------------------- #
    # 5c. External expected GT .dat files (large outputs)               #
    # ---------------------------------------------------------------- #
    large_expected = gen.large_expected_tensors
    if large_expected:
        expected_dir = os.path.join(out_dir, "expected")
        os.makedirs(expected_dir, exist_ok=True)
        for t in large_expected:
            dat_path = os.path.join(expected_dir, f"{t.c_name}.dat")
            with open(dat_path, "wb") as f:
                f.write(gen.generate_expected_dat(t))
        print(f"Expected   : {len(large_expected)} large GT array(s) written to"
              f" {expected_dir}/", file=sys.stderr)

    # ---------------------------------------------------------------- #
    # 6. Driver sources                                                 #
    # ---------------------------------------------------------------- #
    # Collect all driver files needed for the active kernels.
    active_kernels = gen._active_kernels
    all_driver_files = []
    seen_files = set()
    for kd in active_kernels:
        for f in kd.driver_files:
            if f not in seen_files:
                seen_files.add(f)
                all_driver_files.append(f)

    if len(active_kernels) == 1:
        driver_readme = active_kernels[0].driver_readme
    else:
        driver_readme = mixed_driver_readme([kd.name for kd in active_kernels])

    driver_dir = os.path.join(out_dir, "driver")
    if args.driver_dir:
        src_dir = os.path.abspath(args.driver_dir)
        if not os.path.isdir(src_dir):
            print(
                f"error: --driver-dir '{src_dir}' is not a directory",
                file=sys.stderr,
            )
            return 1
        missing = _copy_driver(src_dir, driver_dir, all_driver_files)
        copied  = len(all_driver_files) - len(missing)
        print(f"Driver     : copied {copied}/{len(all_driver_files)} files from {src_dir}",
              file=sys.stderr)
        if missing:
            print("  warning: missing files:", file=sys.stderr)
            for f in missing:
                print(f"    {f}", file=sys.stderr)
            print("  Project will not build until driver/ is complete.",
                  file=sys.stderr)
    else:
        os.makedirs(driver_dir, exist_ok=True)
        with open(os.path.join(driver_dir, "README.md"), "w") as f:
            f.write(driver_readme)
        print("Driver     : driver/ is empty — see driver/README.md",
              file=sys.stderr)

    # ---------------------------------------------------------------- #
    # 7. Generated-files summary on stderr                              #
    # ---------------------------------------------------------------- #
    print("", file=sys.stderr)
    print("Generated project:", file=sys.stderr)
    report_items = [
        "CMakeLists.txt",
        "include/inference.h",
        "include/inference_prof.h",
        "include/inference_ddr.h",
        "src/inference.c",
        "src/inference_buf.c",
        "src/inference_prof.c",
        "src/inference_ddr.c",
        "src/ddr/",
        "test/test_inference.c",
        "scripts/check_inference_setup.sh",
        "driver/",
    ]
    if large_weights:
        report_items.append("weights/")
    if large_expected:
        report_items.append("expected/")

    # ---------------------------------------------------------------- #
    # 8. Markdown report (model summary, transformations, layers)      #
    # ---------------------------------------------------------------- #
    if not args.no_report:
        report_items.append("report.md")
        try:
            md = ReportGenerator(
                graph=graph, codegen=gen,
                model_path=args.model, out_dir=out_dir,
                generated_files=report_items,
            ).render_markdown()
            _write(os.path.join(out_dir, "report.md"), md)
        except Exception as e:
            # The report is informational; never let a formatting bug
            # break a successful project generation.
            print(f"warning: report.md not written ({e})", file=sys.stderr)
            report_items.remove("report.md")

    for rel in report_items:
        print(f"  {os.path.join(out_dir, rel)}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
