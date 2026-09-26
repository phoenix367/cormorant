#!/usr/bin/env python3
"""Generate all test ONNX models to a user-specified directory.

Usage
-----
  python test/gen_all_models.py                      # default: test/models/
  python test/gen_all_models.py --out-dir /tmp/models
"""

import argparse
import importlib.util
import inspect
import os
import sys
import types

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_OUT = os.path.join(_TEST_DIR, "models")


def _load(module_name: str) -> types.ModuleType:
    """Import a generator module from the test/ directory by filename stem."""
    path = os.path.join(_TEST_DIR, module_name + ".py")
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run_argparse_main(mod: types.ModuleType, out_dir: str) -> None:
    """Call main() for modules that read --out-dir via argparse."""
    old_argv = sys.argv[:]
    sys.argv = [mod.__spec__.name, "--out-dir", out_dir]
    try:
        mod.main()
    finally:
        sys.argv = old_argv


def _run_global_out_dir_main(mod: types.ModuleType, out_dir: str) -> None:
    """Call main() for modules that use a global OUT_DIR variable."""
    mod.OUT_DIR = out_dir
    mod.main()


def _run_gen_funcs(mod: types.ModuleType, out_dir: str) -> None:
    """Call every gen_*() function for modules that have no main()."""
    mod.OUT_DIR = out_dir
    os.makedirs(out_dir, exist_ok=True)
    for name, fn in inspect.getmembers(mod, inspect.isfunction):
        if name.startswith("gen_"):
            fn()


# Ordered list: (module_stem, runner_function)
# Pattern A — main() uses argparse --out-dir
_ARGPARSE_MAIN = [
    "gen_test_models",
    "gen_matmul_models",
    "gen_bert_models",
]
# Pattern B — main() uses global OUT_DIR
_GLOBAL_MAIN = [
    "gen_pool_models",
    "gen_reshape_gemm_models",
    "gen_parallel_models",
    "gen_llama_models",
]
# Pattern C/D — no main(); global OUT_DIR + gen_* functions
_GEN_FUNCS = [
    "gen_conv_models",
    "gen_mixed_kernel_models",
    "gen_mixed_all_kernels_models",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate all test ONNX models into a single directory."
    )
    parser.add_argument(
        "--out-dir",
        default=_DEFAULT_OUT,
        help=f"Destination directory for .onnx files (default: {_DEFAULT_OUT})",
    )
    args = parser.parse_args()
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Writing all test models to: {out_dir}\n")

    for stem in _ARGPARSE_MAIN:
        print(f"--- {stem} ---")
        mod = _load(stem)
        _run_argparse_main(mod, out_dir)

    for stem in _GLOBAL_MAIN:
        print(f"--- {stem} ---")
        mod = _load(stem)
        _run_global_out_dir_main(mod, out_dir)

    for stem in _GEN_FUNCS:
        print(f"--- {stem} ---")
        mod = _load(stem)
        _run_gen_funcs(mod, out_dir)

    print("\nAll models generated.")


if __name__ == "__main__":
    main()
