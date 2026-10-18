# Copyright 2025 GradeBuilder SL
# SPDX-License-Identifier: Apache-2.0

"""Generate ONNX test models that combine all three kernel types:
VectorOPKernel (element-wise), MatmulKernel, and ConvKernel.

Each model exercises at least one node from every kernel class.
The Conv kernel requires 4-D NCHW tensors [N, C, H, W].
MatMul is applied to the last two dimensions of its 4-D input, treating
the leading dims as a batch — valid ONNX batched matmul semantics.
VectorOP is element-wise on the full flat array regardless of shape.

Models
------
1  mixed_all_relu_conv_matmul      VectorOP → Conv → VectorOP → MatMul
2  mixed_all_conv_matmul_relu      Conv(+bias) → VectorOP → MatMul → VectorOP
3  mixed_all_norm_conv_project     VectorOP×2 → Conv → VectorOP → MatMul
4  mixed_all_matmul_relu_conv      MatMul → VectorOP → Conv → VectorOP
5  mixed_all_skip_conv_matmul      Conv → VectorOP → VectorOP(skip) → MatMul
6  mixed_all_two_conv_matmul_relu6 Conv(+bias) → VectorOP → Conv → VectorOP → MatMul → VectorOP
7  mixed_all_sub_matmul_conv       VectorOP → MatMul → VectorOP → Conv → VectorOP
"""

import os
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

OUT_DIR = os.path.join(os.path.dirname(__file__), "models")
RNG = np.random.default_rng(0)


def _save(model, name: str) -> None:
    onnx.checker.check_model(model)
    out = os.path.join(OUT_DIR, name)
    onnx.save(model, out)
    print(f"  {out}")


def _vi(name: str, shape) -> onnx.TensorProto:
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)


def _init(data: np.ndarray, name: str):
    return numpy_helper.from_array(data.astype(np.float32), name=name)


def _conv(inputs, output, kernel_shape, pads=None, **kw):
    attrs = {"kernel_shape": kernel_shape}
    if pads:
        attrs["pads"] = pads
    return helper.make_node("Conv", inputs=inputs, outputs=[output], **attrs)


# ---------------------------------------------------------------------------
# 1. mixed_all_relu_conv_matmul
#
#   X[1,4,8,8]
#   → Relu(X)                      [VectorOP]   → Z[1,4,8,8]
#   → Conv(Z, Wc[8,4,1,1])        [ConvKernel] → A[1,8,8,8]
#   → Relu(A)                      [VectorOP]   → B[1,8,8,8]
#   → MatMul(B[1,8,8,8], Wm[8,4]) [MatMul]     → Y[1,8,8,4]
#
#   MatMul: a_batch=[1,8], n=8, k=8, m=4, batch=8, a_stride=64, c_stride=32.
# ---------------------------------------------------------------------------
def gen_relu_conv_matmul() -> None:
    wc = _init(RNG.standard_normal((8, 4, 1, 1)) * 0.25, "Wc")
    wm = _init(RNG.standard_normal((8, 4)) * 0.25,       "Wm")

    nodes = [
        helper.make_node("Relu",   ["X",  ],       ["Z"]),
        _conv(                     ["Z",  "Wc"],   "A", [1, 1]),
        helper.make_node("Relu",   ["A",  ],       ["B"]),
        helper.make_node("MatMul", ["B",  "Wm"],   ["Y"]),
    ]
    graph = helper.make_graph(
        nodes, "relu_conv_matmul",
        inputs=[_vi("X", [1, 4, 8, 8])],
        outputs=[_vi("Y", [1, 8, 8, 4])],
        initializer=[wc, wm],
    )
    _save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)]),
          "mixed_all_relu_conv_matmul.onnx")


# ---------------------------------------------------------------------------
# 2. mixed_all_conv_matmul_relu
#
#   X[1,3,8,8]
#   → Conv(X, Wc[8,3,3,3], Bc[8], pad=1) [ConvKernel] → Z[1,8,8,8]
#   → Relu(Z)                              [VectorOP]   → A[1,8,8,8]
#   → MatMul(A[1,8,8,8], Wm[8,4])         [MatMul]     → B[1,8,8,4]
#   → Relu(B)                              [VectorOP]   → Y[1,8,8,4]
#
#   Conv uses its built-in bias Bc[8] (per-output-channel).
#   Bias addition is handled by ConvKernel, not a separate VectorOP.
#   The trailing Relu is flat (same shape as B, non-broadcast).
# ---------------------------------------------------------------------------
def gen_conv_matmul_relu() -> None:
    wc = _init(RNG.standard_normal((8, 3, 3, 3)) * 0.25, "Wc")
    bc = _init(np.zeros(8),                               "Bc")
    wm = _init(RNG.standard_normal((8, 4)) * 0.25,       "Wm")

    nodes = [
        _conv(                     ["X", "Wc", "Bc"], "Z", [3, 3], pads=[1, 1, 1, 1]),
        helper.make_node("Relu",   ["Z"],              ["A"]),
        helper.make_node("MatMul", ["A", "Wm"],        ["B"]),
        helper.make_node("Relu",   ["B"],              ["Y"]),
    ]
    graph = helper.make_graph(
        nodes, "conv_matmul_relu",
        inputs=[_vi("X", [1, 3, 8, 8])],
        outputs=[_vi("Y", [1, 8, 8, 4])],
        initializer=[wc, bc, wm],
    )
    _save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)]),
          "mixed_all_conv_matmul_relu.onnx")


# ---------------------------------------------------------------------------
# 3. mixed_all_norm_conv_project
#
#   X[1,4,8,8]
#   → Sub(X, offset[1,4,8,8])      [VectorOP]   → Z1[1,4,8,8]  flat, non-broadcast
#   → Mul(Z1, scale[1,4,8,8])      [VectorOP]   → Z2[1,4,8,8]  flat, non-broadcast
#   → Conv(Z2, Wc[8,4,3,3], pad=1) [ConvKernel] → A[1,8,8,8]
#   → Relu(A)                       [VectorOP]   → B[1,8,8,8]
#   → MatMul(B[1,8,8,8], Wm[8,4])  [MatMul]     → Y[1,8,8,4]
#
#   Simulates per-element normalisation (subtract mean, multiply inv-std)
#   before convolution and a spatial linear projection afterward.
# ---------------------------------------------------------------------------
def gen_norm_conv_project() -> None:
    offset = _init(np.zeros((1, 4, 8, 8)),                     "offset")
    scale  = _init(np.ones((1, 4, 8, 8)),                      "scale")
    wc     = _init(RNG.standard_normal((8, 4, 3, 3)) * 0.25,   "Wc")
    wm     = _init(RNG.standard_normal((8, 4)) * 0.25,         "Wm")

    nodes = [
        helper.make_node("Sub",    ["X",  "offset"], ["Z1"]),
        helper.make_node("Mul",    ["Z1", "scale"],  ["Z2"]),
        _conv(                     ["Z2", "Wc"],     "A", [3, 3], pads=[1, 1, 1, 1]),
        helper.make_node("Relu",   ["A",  ],         ["B"]),
        helper.make_node("MatMul", ["B",  "Wm"],     ["Y"]),
    ]
    graph = helper.make_graph(
        nodes, "norm_conv_project",
        inputs=[_vi("X", [1, 4, 8, 8])],
        outputs=[_vi("Y", [1, 8, 8, 4])],
        initializer=[offset, scale, wc, wm],
    )
    _save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)]),
          "mixed_all_norm_conv_project.onnx")


# ---------------------------------------------------------------------------
# 4. mixed_all_matmul_relu_conv
#
#   X[1,4,8,8]
#   → MatMul(X[1,4,8,8], Wm[8,8]) [MatMul]     → Z[1,4,8,8]
#   → Relu(Z)                      [VectorOP]   → A[1,4,8,8]
#   → Conv(A, Wc[8,4,1,1])        [ConvKernel] → B[1,8,8,8]
#   → Relu6(B)                     [VectorOP]   → Y[1,8,8,8]
#
#   MatMul: a_batch=[1,4], n=8, k=8, m=8, batch=4, a_stride=64, c_stride=64.
#   Relu6 modelled as Clip(min=0, max=6).
# ---------------------------------------------------------------------------
def gen_matmul_relu_conv() -> None:
    wm = _init(RNG.standard_normal((8, 8)) * 0.25, "Wm")
    wc = _init(RNG.standard_normal((8, 4, 1, 1)) * 0.25, "Wc")

    nodes = [
        helper.make_node("MatMul", ["X", "Wm"],         ["Z"]),
        helper.make_node("Relu",   ["Z"],                ["A"]),
        _conv(                     ["A", "Wc"],          "B", [1, 1]),
        helper.make_node("Clip",   ["B", "clip_min", "clip_max"], ["Y"]),
    ]
    clip_min = numpy_helper.from_array(np.array(0.0, dtype=np.float32), name="clip_min")
    clip_max = numpy_helper.from_array(np.array(6.0, dtype=np.float32), name="clip_max")

    graph = helper.make_graph(
        nodes, "matmul_relu_conv",
        inputs=[_vi("X", [1, 4, 8, 8])],
        outputs=[_vi("Y", [1, 8, 8, 8])],
        initializer=[wm, wc, clip_min, clip_max],
    )
    _save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)]),
          "mixed_all_matmul_relu_conv.onnx")


# ---------------------------------------------------------------------------
# 5. mixed_all_skip_conv_matmul
#
#   X[1,8,4,4]
#   → Conv(X, Wc[8,8,1,1])        [ConvKernel] → Z[1,8,4,4]
#   → Relu(Z)                      [VectorOP]   → A[1,8,4,4]
#   → Add(X, A)                    [VectorOP]   → B[1,8,4,4]  skip connection
#   → MatMul(B[1,8,4,4], Wm[4,4]) [MatMul]     → Y[1,8,4,4]
#
#   All Add inputs have the same shape → flat, non-broadcast.
#   MatMul: a_batch=[1,8], n=4, k=4, m=4, batch=8, a_stride=16, c_stride=16.
# ---------------------------------------------------------------------------
def gen_skip_conv_matmul() -> None:
    wc = _init(RNG.standard_normal((8, 8, 1, 1)) * 0.25, "Wc")
    wm = _init(RNG.standard_normal((4, 4)) * 0.25,       "Wm")

    nodes = [
        _conv(                     ["X", "Wc"],  "Z", [1, 1]),
        helper.make_node("Relu",   ["Z"],        ["A"]),
        helper.make_node("Add",    ["X", "A"],   ["B"]),
        helper.make_node("MatMul", ["B", "Wm"],  ["Y"]),
    ]
    graph = helper.make_graph(
        nodes, "skip_conv_matmul",
        inputs=[_vi("X", [1, 8, 4, 4])],
        outputs=[_vi("Y", [1, 8, 4, 4])],
        initializer=[wc, wm],
    )
    _save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)]),
          "mixed_all_skip_conv_matmul.onnx")


# ---------------------------------------------------------------------------
# 6. mixed_all_two_conv_matmul_relu6
#
#   X[1,3,8,8]
#   → Conv(X,  Wc1[8,3,3,3], Bc1[8], pad=1) [ConvKernel] → Z1[1,8,8,8]
#   → Relu(Z1)                                [VectorOP]   → A1[1,8,8,8]
#   → Conv(A1, Wc2[8,8,1,1])                 [ConvKernel] → Z2[1,8,8,8]
#   → Relu(Z2)                                [VectorOP]   → A2[1,8,8,8]
#   → MatMul(A2[1,8,8,8], Wm[8,4])           [MatMul]     → B[1,8,8,4]
#   → Relu6(B)                                [VectorOP]   → Y[1,8,8,4]
#
#   Two conv layers: first with bias (ConvKernel bias path), second without.
#   Relu6 (Clip 0…6) as final activation exercises OP_RELU6 code path.
# ---------------------------------------------------------------------------
def gen_two_conv_matmul_relu6() -> None:
    wc1  = _init(RNG.standard_normal((8, 3, 3, 3)) * 0.25, "Wc1")
    bc1  = _init(np.zeros(8),                               "Bc1")
    wc2  = _init(RNG.standard_normal((8, 8, 1, 1)) * 0.25, "Wc2")
    wm   = _init(RNG.standard_normal((8, 4)) * 0.25,       "Wm")

    clip_min = numpy_helper.from_array(np.array(0.0, dtype=np.float32), name="clip_min2")
    clip_max = numpy_helper.from_array(np.array(6.0, dtype=np.float32), name="clip_max2")

    nodes = [
        _conv(                     ["X",  "Wc1", "Bc1"],           "Z1", [3, 3], pads=[1, 1, 1, 1]),
        helper.make_node("Relu",   ["Z1"],                          ["A1"]),
        _conv(                     ["A1", "Wc2"],                   "Z2", [1, 1]),
        helper.make_node("Relu",   ["Z2"],                          ["A2"]),
        helper.make_node("MatMul", ["A2", "Wm"],                    ["B"]),
        helper.make_node("Clip",   ["B", "clip_min2", "clip_max2"], ["Y"]),
    ]
    graph = helper.make_graph(
        nodes, "two_conv_matmul_relu6",
        inputs=[_vi("X", [1, 3, 8, 8])],
        outputs=[_vi("Y", [1, 8, 8, 4])],
        initializer=[wc1, bc1, wc2, wm, clip_min, clip_max],
    )
    _save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)]),
          "mixed_all_two_conv_matmul_relu6.onnx")


# ---------------------------------------------------------------------------
# 7. mixed_all_sub_matmul_conv
#
#   X[1,4,4,8]
#   → Sub(X, norm_w[1,4,4,8])      [VectorOP]   → Z1[1,4,4,8]  flat, non-broadcast
#   → MatMul(Z1[1,4,4,8], Wm[8,8]) [MatMul]     → Z2[1,4,4,8]
#   → Relu(Z2)                      [VectorOP]   → A[1,4,4,8]
#   → Conv(A[1,4,4,8], Wc[8,4,1,1]) [ConvKernel] → B[1,8,4,8]
#   → Relu6(B)                      [VectorOP]   → Y[1,8,4,8]
#
#   MatMul: a_batch=[1,4], n=4, k=8, m=8, batch=4, a_stride=32, c_stride=32.
#   Conv treats A as [N=1, IC=4, H=4, W=8] with a 1×1 kernel.
# ---------------------------------------------------------------------------
def gen_sub_matmul_conv() -> None:
    norm_w = _init(np.zeros((1, 4, 4, 8)),                    "norm_w")
    wm     = _init(RNG.standard_normal((8, 8)) * 0.25,        "Wm")
    wc     = _init(RNG.standard_normal((8, 4, 1, 1)) * 0.25,  "Wc")

    clip_min = numpy_helper.from_array(np.array(0.0, dtype=np.float32), name="clip_min")
    clip_max = numpy_helper.from_array(np.array(6.0, dtype=np.float32), name="clip_max")

    nodes = [
        helper.make_node("Sub",    ["X",  "norm_w"],             ["Z1"]),
        helper.make_node("MatMul", ["Z1", "Wm"],                 ["Z2"]),
        helper.make_node("Relu",   ["Z2"],                       ["A"]),
        _conv(                     ["A",  "Wc"],                 "B", [1, 1]),
        helper.make_node("Clip",   ["B", "clip_min", "clip_max"], ["Y"]),
    ]
    graph = helper.make_graph(
        nodes, "sub_matmul_conv",
        inputs=[_vi("X", [1, 4, 4, 8])],
        outputs=[_vi("Y", [1, 8, 4, 8])],
        initializer=[norm_w, wm, wc, clip_min, clip_max],
    )
    _save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)]),
          "mixed_all_sub_matmul_conv.onnx")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Generating mixed all-kernel test models...")
    gen_relu_conv_matmul()
    gen_conv_matmul_relu()
    gen_norm_conv_project()
    gen_matmul_relu_conv()
    gen_skip_conv_matmul()
    gen_two_conv_matmul_relu6()
    gen_sub_matmul_conv()
    print("Done.")
