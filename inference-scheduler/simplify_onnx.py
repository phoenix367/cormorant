#!/usr/bin/env python3
# Copyright 2025 GradeBuilder SL
# SPDX-License-Identifier: Apache-2.0

"""
simplify_onnx.py — Simplify an ONNX model.

Pins any dynamic input dimensions, runs ``onnxsim`` (constant folding, shape
inference, dead-node removal), then runs ``onnxoptimizer.fuse_bn_into_conv``
as a safety-net for BatchNormalization layers that onnxsim didn't fold.

The default output path is ``<stem>-simplified.onnx`` next to the input.

Usage:
  python simplify_onnx.py model.onnx                       # uses model's existing shapes
  python simplify_onnx.py model.onnx --batch 1             # pin all input batch dims to 1
  python simplify_onnx.py model.onnx --input-shape data=1,3,224,224
  python simplify_onnx.py model.onnx -o out.onnx --no-fuse-bn
  python simplify_onnx.py model.onnx --check               # also run an ORT smoke test
"""

import argparse
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.remote import _bold, _cyan, _dim, _green, _red, _yellow  # noqa: E402


def _parse_input_shape(spec: str) -> tuple[str, List[int]]:
    """Parse ``NAME=D1,D2,...`` into ``(NAME, [D1, D2, ...])``."""
    if "=" not in spec:
        raise argparse.ArgumentTypeError(
            f"--input-shape '{spec}' must be NAME=D1,D2,... (no '=' found)"
        )
    name, dims_str = spec.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError(f"--input-shape '{spec}': empty NAME")
    try:
        dims = [int(d.strip()) for d in dims_str.split(",") if d.strip()]
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"--input-shape '{spec}': non-integer dimension ({e})"
        ) from e
    if not dims or any(d <= 0 for d in dims):
        raise argparse.ArgumentTypeError(
            f"--input-shape '{spec}': dims must be positive integers"
        )
    return name, dims


def _input_shapes_from_args(model, batch: Optional[int],
                            explicit: List[tuple[str, List[int]]]) -> Dict[str, List[int]]:
    """Build the ``overwrite_input_shapes`` dict for ``onnxsim.simplify``.

    ``--input-shape`` entries take precedence over ``--batch``.  ``--batch``
    pins the first dim of any input whose first dim is currently dynamic
    (dim_param set) and that isn't already covered by ``--input-shape``.
    """
    out: Dict[str, List[int]] = {name: list(dims) for name, dims in explicit}
    if batch is None:
        return out
    for vi in model.graph.input:
        if vi.name in out:
            continue
        dims = vi.type.tensor_type.shape.dim
        if not dims:
            continue
        # Pin first dim to batch only if it is dynamic; concrete first dims
        # stay as authored to avoid surprising shape rewrites.
        first = dims[0]
        if first.dim_value > 0:
            continue
        resolved: List[int] = [batch]
        for d in dims[1:]:
            if d.dim_value > 0:
                resolved.append(d.dim_value)
            else:
                # Can't infer non-batch dynamic dims from --batch alone.
                raise SystemExit(
                    f"input '{vi.name}' has a non-batch dynamic dim "
                    f"(dim_param={d.dim_param!r}); use --input-shape "
                    f"{vi.name}=N,..."
                )
        out[vi.name] = resolved
    return out


def _op_counts(model) -> Counter:
    return Counter(n.op_type for n in model.graph.node)


def _print_op_diff(before: Counter, after: Counter) -> None:
    all_ops = sorted(set(before) | set(after))
    for op in all_ops:
        b, a = before.get(op, 0), after.get(op, 0)
        if b == a:
            line = f"  {op:<25s} {a:>5d}"
        else:
            delta = a - b
            sign = "-" if delta < 0 else "+"
            line = f"  {op:<25s} {a:>5d}   ({sign}{abs(delta)} from {b})"
            line = _yellow(line)
        print(line)


def _smoke_test(path: Path) -> None:
    """Run a random-input inference via onnxruntime to confirm the model loads."""
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    feeds = {}
    np.random.seed(0)
    for inp in sess.get_inputs():
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
        feeds[inp.name] = np.random.randn(*shape).astype(np.float32)
    outs = sess.run(None, feeds)
    for o, val in zip(sess.get_outputs(), outs, strict=True):
        print(f"  {_dim('out')}  {o.name}: shape={list(val.shape)} "
              f"min={float(val.min()):.3f} max={float(val.max()):.3f}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Simplify an ONNX model (onnxsim + fuse_bn_into_conv).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("input", help="Input .onnx file")
    p.add_argument("-o", "--output", metavar="PATH",
                   help="Output .onnx path (default: <stem>-simplified.onnx)")
    p.add_argument("--batch", type=int, metavar="N",
                   help="Pin every input's dynamic first dim to N "
                        "(non-batch dynamic dims must use --input-shape).")
    p.add_argument("--input-shape", metavar="NAME=D1,D2,...",
                   action="append", default=[], type=_parse_input_shape,
                   help="Override the shape of a single input. Repeatable.")
    p.add_argument("--no-fuse-bn", action="store_true",
                   help="Skip the explicit fuse_bn_into_conv pass "
                        "(onnxsim usually folds BN itself, so this is mostly "
                        "useful when you want to keep BN nodes for inspection).")
    p.add_argument("--check", action="store_true",
                   help="After saving, run a random-input inference via "
                        "onnxruntime to confirm the simplified model loads "
                        "and produces an output.")
    args = p.parse_args(argv)

    # Defer heavy imports until args parse so --help is snappy.
    import onnx
    import onnxsim

    src = Path(args.input)
    if not src.is_file():
        print(_red(f"error: input not found: {src}"), file=sys.stderr)
        return 1

    dst = Path(args.output) if args.output \
          else src.with_name(f"{src.stem}-simplified.onnx")

    print(f"{_bold('Loading')} {src}")
    model = onnx.load(src)
    before = _op_counts(model)
    n_before = len(model.graph.node)
    opset = ", ".join(f"{(o.domain or 'ai.onnx')}:{o.version}"
                      for o in model.opset_import)
    print(f"  {n_before} nodes, opset {opset}")

    try:
        shapes = _input_shapes_from_args(model, args.batch, args.input_shape)
    except SystemExit as exc:
        print(_red(f"error: {exc}"), file=sys.stderr)
        return 1
    if shapes:
        print(f"  {_cyan('pin')} input shapes: " +
              ", ".join(f"{k}={v}" for k, v in shapes.items()))

    print(f"\n{_bold('onnxsim.simplify')}")
    simp, ok = onnxsim.simplify(
        model,
        overwrite_input_shapes=shapes if shapes else None,
    )
    if not ok:
        print(_red("  onnxsim returned ok=False — output may be incorrect"),
              file=sys.stderr)
        return 1

    if not args.no_fuse_bn:
        print(f"\n{_bold('onnxoptimizer.fuse_bn_into_conv')}")
        import onnxoptimizer
        simp = onnxoptimizer.optimize(simp, ["fuse_bn_into_conv"])

    onnx.checker.check_model(simp)

    after = _op_counts(simp)
    n_after = len(simp.graph.node)
    print(f"\n{_bold('Result')}: {n_before} → {n_after} nodes "
          f"({_green(f'-{n_before - n_after}') if n_after < n_before else f'+{n_after - n_before}'})")
    _print_op_diff(before, after)

    onnx.save(simp, dst)
    size_mb = dst.stat().st_size / 1024 / 1024
    print(f"\n{_green('wrote')} {dst}  ({size_mb:.1f} MB)")

    if args.check:
        print(f"\n{_bold('onnxruntime smoke test')}")
        try:
            _smoke_test(dst)
            print(_green("  OK"))
        except Exception as exc:
            print(_red(f"  FAILED: {exc}"), file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
