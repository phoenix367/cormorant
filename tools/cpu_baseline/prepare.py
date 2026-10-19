#!/usr/bin/env python3
# Copyright 2026 Ivan Gubochkin
# SPDX-License-Identifier: Apache-2.0

"""
prepare.py — the CPU baseline's models (cpu_baseline.py runs it in its own
venv: onnxruntime 1.23.2 = the board's, onnx, sympy).

  int8 SRC DST
      onnxruntime's dynamic quantisation of SRC: QInt8 weights, the
      activations quantised per call.
  last-row SRC DST
      a SmolLM2 export with its head on the last position only: a Slice
      (axis 1, the last element) before the MatMul that makes "logits", as
      the FPGA library computes a prefill's logits (LlmSelectRow).  Checked:
      for a prompt of a few tokens the edited model's logits equal the last
      row of SRC's exactly, or the script fails.
"""

import argparse
import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper


def int8(src: str, dst: str) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic
    quantize_dynamic(src, dst, weight_type=QuantType.QInt8)


def last_row(src: str, dst: str) -> None:
    model = onnx.load(src)
    g = model.graph
    head = next(n for n in g.node if "logits" in n.output)                 # /lm_head/MatMul
    g.initializer.extend(numpy_helper.from_array(np.array([v], np.int64), k) for k, v in
                         (("last_row_starts", -1), ("last_row_ends", np.iinfo(np.int64).max),
                          ("last_row_axes", 1)))
    g.node.insert(list(g.node).index(head),
                  helper.make_node("Slice", [head.input[0], "last_row_starts", "last_row_ends", "last_row_axes"],
                                   [head.input[0] + "_last_row"], name="/lm_head/last_row"))
    head.input[0] += "_last_row"
    next(o for o in g.output if o.name == "logits").type.tensor_type.shape.dim[1].dim_value = 1
    # (onnx.checker rejects the export itself: SimplifiedLayerNormalization in the default domain)
    onnx.save(model, dst)
    a, b = logits(src), logits(dst)
    d = float(np.abs(a[0, -1] - b[0, -1]).max())
    if b.shape[1] != 1 or d != 0.0:
        sys.exit(f"{dst}: the last-row logits differ from {src}'s (shape {b.shape}, max |diff| {d})")
    print(f"  {dst}: logits [1, 1, {b.shape[2]}] = the last row of the export's [1, {a.shape[1]}, "
          f"{a.shape[2]}] (max |diff| 0)", flush=True)


def logits(model: str) -> np.ndarray:
    import onnxruntime as ort
    s = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    ids = [1, 9690, 198, 2683, 359, 253, 5356, 5646, 11173, 3365]              # a chat prompt's start
    f = {"input_ids": np.array([ids], np.int64), "attention_mask": np.ones((1, len(ids)), np.int64),
         "position_ids": np.arange(len(ids), dtype=np.int64)[None, :]}
    f.update({i.name: np.zeros((1, i.shape[1], 0, i.shape[3]), np.float32)
              for i in s.get_inputs() if i.name.startswith("past_key_values.")})
    return s.run(["logits"], f)[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    ap.add_argument("what", choices=("int8", "last-row"))
    ap.add_argument("src")
    ap.add_argument("dst")
    a = ap.parse_args()
    (int8 if a.what == "int8" else last_row)(a.src, a.dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
