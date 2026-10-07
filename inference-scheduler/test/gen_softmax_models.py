"""Generate ONNX test models for VectorOPKernel's softmax unit
(doc/plans/SOFTMAX_PLAN.md, src/smx_nodes.py): ONNX Softmax on the last axis.

Where the platform has the unit (``kernels.vectorop.softmax``,
``AXI_VECTOROP_SOFTMAX=1``) each Softmax is one OP_SOFTMAX call — row mode, an
integer softmax within one LSB of the exact one — and the board run checks it
against the scheduler's simulation; without it the scheduler keeps the host
op.  The test harness fills inputs with the ramp ``p[pos] = (Data_t)pos``;
the models scale it with a constant first so the rows' score ranges vary.
"""

import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OUT_DIR = os.path.join(os.path.dirname(__file__), "models")
OPSET = 13


def _vi(name, shape):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)


def _init(name, arr):
    return numpy_helper.from_array(np.asarray(arr, np.float32), name=name)


def _save(nodes, inputs, outputs, inits, name):
    graph = helper.make_graph(nodes, name.replace(".onnx", ""), inputs, outputs, initializer=inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    out = os.path.join(OUT_DIR, name)
    onnx.save(model, out)
    print(f"  {out}")


def _scaled_softmax(shape, seed, lo, hi, name):
    """Mul by a constant [shape] (scores of different ranges per row), Softmax."""
    rng = np.random.default_rng(seed)
    scale = np.round(rng.uniform(lo, hi, shape) * 256) / 256
    _save([helper.make_node("Mul", ["X", "scale"], ["S"]),
           helper.make_node("Softmax", ["S"], ["Y"], axis=-1)],
          [_vi("X", shape)], [_vi("Y", shape)], [_init("scale", scale)], name)


def gen_rows_64():
    """32 rows of 64 (scores within about +-16)."""
    _scaled_softmax([32, 64], 30, -2.0, 2.0, "smx_rows_64.onnx")


def gen_rows_2048():
    """2 rows of 2048, the row mode's longest."""
    _scaled_softmax([2, 2048], 31, -0.25, 0.25, "smx_rows_2048.onnx")


def gen_attention():
    """BERT's shape: scores [1, heads, q, k] plus a broadcast key mask (-10000:
    masked keys saturate to -128), softmax over the keys."""
    rng = np.random.default_rng(32)
    mask = np.zeros((1, 1, 1, 48), np.float32)
    mask[..., 40:] = -10000.0
    scale = np.round(rng.uniform(-1.0, 1.0, (1, 4, 24, 48)) * 256) / 256
    _save([helper.make_node("Mul", ["X", "scale"], ["S"]),
           helper.make_node("Add", ["S", "mask"], ["M"]),
           helper.make_node("Softmax", ["M"], ["Y"], axis=-1)],
          [_vi("X", [1, 4, 24, 48])], [_vi("Y", [1, 4, 24, 48])],
          [_init("scale", scale), _init("mask", mask)], "smx_attention.onnx")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for fn in (gen_rows_64, gen_rows_2048, gen_attention):
        fn()


if __name__ == "__main__":
    main()
