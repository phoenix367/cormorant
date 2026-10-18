# Copyright 2025 GradeBuilder SL
# SPDX-License-Identifier: Apache-2.0

"""Generate ONNX test models that combine MatmulKernel and VectorOPKernel nodes."""

import os
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

OUT_DIR = os.path.join(os.path.dirname(__file__), "models")


def _save(model, name: str) -> None:
    onnx.checker.check_model(model)
    out = os.path.join(OUT_DIR, name)
    onnx.save(model, out)
    print(f"  {out}")


def gen_matmul_then_relu(n=4, k=8, m=4) -> None:
    """MatMul(X, W) → Relu(Z) — uses both kernels in sequence."""
    w_data = (np.random.randn(k, m) * 0.5).astype(np.float32)
    w_init = numpy_helper.from_array(w_data, name="W")

    mm = helper.make_node("MatMul", inputs=["X", "W"], outputs=["Z"])
    relu = helper.make_node("Relu", inputs=["Z"], outputs=["Y"])

    graph = helper.make_graph(
        [mm, relu],
        "matmul_relu",
        inputs=[helper.make_tensor_value_info("X", TensorProto.FLOAT, [n, k])],
        outputs=[helper.make_tensor_value_info("Y", TensorProto.FLOAT, [n, m])],
        initializer=[w_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    _save(model, "mixed_matmul_relu.onnx")


def gen_add_then_matmul(n=4, k=8, m=4) -> None:
    """Add(X, bias) → MatMul(Z, W) — VectorOP feeds into MatMul."""
    bias_data = np.zeros(k, dtype=np.float32)
    w_data    = (np.random.randn(k, m) * 0.5).astype(np.float32)
    bias_init = numpy_helper.from_array(bias_data, name="bias")
    w_init    = numpy_helper.from_array(w_data,    name="W")

    add = helper.make_node("Add",    inputs=["X", "bias"], outputs=["Z"])
    mm  = helper.make_node("MatMul", inputs=["Z", "W"],    outputs=["Y"])

    graph = helper.make_graph(
        [add, mm],
        "add_matmul",
        inputs=[helper.make_tensor_value_info("X", TensorProto.FLOAT, [n, k])],
        outputs=[helper.make_tensor_value_info("Y", TensorProto.FLOAT, [n, m])],
        initializer=[bias_init, w_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    _save(model, "mixed_add_matmul.onnx")


def gen_matmul_add_relu(n=4, k=8, m=4) -> None:
    """MatMul(X, W) → Add(Z, bias) → Relu(A) — classic linear layer."""
    w_data    = (np.random.randn(k, m) * 0.5).astype(np.float32)
    bias_data = np.zeros(m, dtype=np.float32)
    w_init    = numpy_helper.from_array(w_data,    name="W")
    bias_init = numpy_helper.from_array(bias_data, name="bias")

    mm   = helper.make_node("MatMul", inputs=["X", "W"],      outputs=["Z"])
    add  = helper.make_node("Add",    inputs=["Z", "bias"],    outputs=["A"])
    relu = helper.make_node("Relu",   inputs=["A"],            outputs=["Y"])

    graph = helper.make_graph(
        [mm, add, relu],
        "matmul_add_relu",
        inputs=[helper.make_tensor_value_info("X", TensorProto.FLOAT, [n, k])],
        outputs=[helper.make_tensor_value_info("Y", TensorProto.FLOAT, [n, m])],
        initializer=[w_init, bias_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    _save(model, "mixed_matmul_add_relu.onnx")


def gen_two_layer_mlp(n=4, k=8, h=6, m=4) -> None:
    """Two linear layers: MatMul+Add+Relu → MatMul+Add — a minimal MLP."""
    w1_data  = (np.random.randn(k, h) * 0.5).astype(np.float32)
    b1_data  = np.zeros(h, dtype=np.float32)
    w2_data  = (np.random.randn(h, m) * 0.5).astype(np.float32)
    b2_data  = np.zeros(m, dtype=np.float32)

    inits = [
        numpy_helper.from_array(w1_data, name="W1"),
        numpy_helper.from_array(b1_data, name="b1"),
        numpy_helper.from_array(w2_data, name="W2"),
        numpy_helper.from_array(b2_data, name="b2"),
    ]

    nodes = [
        helper.make_node("MatMul", ["X",  "W1"], ["z1"]),
        helper.make_node("Add",    ["z1", "b1"], ["a1"]),
        helper.make_node("Relu",   ["a1"],        ["h1"]),
        helper.make_node("MatMul", ["h1", "W2"], ["z2"]),
        helper.make_node("Add",    ["z2", "b2"], ["Y"]),
    ]

    graph = helper.make_graph(
        nodes, "two_layer_mlp",
        inputs=[helper.make_tensor_value_info("X",  TensorProto.FLOAT, [n, k])],
        outputs=[helper.make_tensor_value_info("Y", TensorProto.FLOAT, [n, m])],
        initializer=inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    _save(model, "mixed_two_layer_mlp.onnx")


def gen_add_matmul_unaligned(n=4, k=6, m=4) -> None:
    """Add(X[4,6], bias[6]) → MatMul(Z[4,6], W[6,4]) → Y[4,4].
    K=6 is unaligned (aligned_chunk=8), producing a gap=2 in Z's layout.
    """
    bias_data = np.zeros(k, dtype=np.float32)
    w_data    = (np.random.randn(k, m) * 0.5).astype(np.float32)
    bias_init = numpy_helper.from_array(bias_data, name="bias")
    w_init    = numpy_helper.from_array(w_data,    name="W")

    add = helper.make_node("Add",    inputs=["X", "bias"], outputs=["Z"])
    mm  = helper.make_node("MatMul", inputs=["Z", "W"],    outputs=["Y"])

    graph = helper.make_graph(
        [add, mm],
        "add_matmul_unaligned",
        inputs=[helper.make_tensor_value_info("X", TensorProto.FLOAT, [n, k])],
        outputs=[helper.make_tensor_value_info("Y", TensorProto.FLOAT, [n, m])],
        initializer=[bias_init, w_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    _save(model, "mixed_add_matmul_unaligned.onnx")


def gen_matmul_scale_bias(n=4, k=8, m=4) -> None:
    """MatMul(X[4,8], W[8,4]) → Mul(Z, scale[4]) → Add(S, bias[4]) → Y[4,4].
    Mul and Add both broadcast (outer_count=4, chunk=4, aligned=8), producing
    gap=4 in Z, S, and Y layouts.
    """
    w_data     = (np.random.randn(k, m) * 0.5).astype(np.float32)
    scale_data = np.zeros(m, dtype=np.float32)
    bias_data  = np.zeros(m, dtype=np.float32)
    w_init     = numpy_helper.from_array(w_data,     name="W")
    scale_init = numpy_helper.from_array(scale_data, name="scale")
    bias_init  = numpy_helper.from_array(bias_data,  name="bias")

    mm  = helper.make_node("MatMul", inputs=["X", "W"],       outputs=["Z"])
    mul = helper.make_node("Mul",    inputs=["Z", "scale"],    outputs=["S"])
    add = helper.make_node("Add",    inputs=["S", "bias"],     outputs=["Y"])

    graph = helper.make_graph(
        [mm, mul, add],
        "matmul_scale_bias",
        inputs=[helper.make_tensor_value_info("X", TensorProto.FLOAT, [n, k])],
        outputs=[helper.make_tensor_value_info("Y", TensorProto.FLOAT, [n, m])],
        initializer=[w_init, scale_init, bias_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    _save(model, "mixed_matmul_scale_bias.onnx")


def gen_outer_matmul_relu(b1=2, b2=3, n=4, k=6, m=4) -> None:
    """A[2,3,4,6] @ B[3,6,4] → Z[2,3,4,4], Relu(Z) → Y[2,3,4,4].
    Outer loop (outer_count=2), inner batch=3.
    B is a weight initializer (not a graph input).
    """
    b_data = (np.random.randn(b2, k, m) * 0.5).astype(np.float32)
    b_init = numpy_helper.from_array(b_data, name="B")

    mm   = helper.make_node("MatMul", inputs=["A", "B"], outputs=["Z"])
    relu = helper.make_node("Relu",   inputs=["Z"],       outputs=["Y"])

    graph = helper.make_graph(
        [mm, relu],
        "outer_matmul_relu",
        inputs=[helper.make_tensor_value_info("A", TensorProto.FLOAT, [b1, b2, n, k])],
        outputs=[helper.make_tensor_value_info("Y", TensorProto.FLOAT, [b1, b2, n, m])],
        initializer=[b_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    _save(model, "mixed_outer_matmul_relu.onnx")


def gen_relu6_matmul_add(n=4, k=8, h=6) -> None:
    """Relu6(X[4,8]) → MatMul(Z[4,8], W[8,6]) → Add(A[4,6], bias[6]) → Y[4,6].
    Add broadcasts (outer_count=4, chunk=6, aligned=8) → A.alloc=32 (gap=2).
    """
    w_data    = (np.random.randn(k, h) * 0.5).astype(np.float32)
    bias_data = np.zeros(h, dtype=np.float32)
    w_init    = numpy_helper.from_array(w_data,    name="W")
    bias_init = numpy_helper.from_array(bias_data, name="bias")

    relu6 = helper.make_node("Relu",   inputs=["X"],          outputs=["Z"])
    mm    = helper.make_node("MatMul", inputs=["Z", "W"],      outputs=["A"])
    add   = helper.make_node("Add",    inputs=["A", "bias"],   outputs=["Y"])

    graph = helper.make_graph(
        [relu6, mm, add],
        "relu6_matmul_add",
        inputs=[helper.make_tensor_value_info("X", TensorProto.FLOAT, [n, k])],
        outputs=[helper.make_tensor_value_info("Y", TensorProto.FLOAT, [n, h])],
        initializer=[w_init, bias_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    _save(model, "mixed_relu6_matmul_add.onnx")


def gen_residual(n=4, k=8, m=4) -> None:
    """MatMul(X[4,8], W[8,4]) → Add(Z[4,4], res[4,4]) → Y[4,4].
    res is a weight initializer with shape [4,4] (same as Z) — non-broadcast Add.
    Z.alloc=16 (flat), Y.alloc=16 (flat).
    """
    w_data   = (np.random.randn(k, m) * 0.5).astype(np.float32)
    res_data = np.zeros((n, m), dtype=np.float32)
    w_init   = numpy_helper.from_array(w_data,   name="W")
    res_init = numpy_helper.from_array(res_data, name="res")

    mm  = helper.make_node("MatMul", inputs=["X", "W"],    outputs=["Z"])
    add = helper.make_node("Add",    inputs=["Z", "res"],   outputs=["Y"])

    graph = helper.make_graph(
        [mm, add],
        "residual",
        inputs=[helper.make_tensor_value_info("X", TensorProto.FLOAT, [n, k])],
        outputs=[helper.make_tensor_value_info("Y", TensorProto.FLOAT, [n, m])],
        initializer=[w_init, res_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    _save(model, "mixed_residual.onnx")


def gen_batch_matmul_relu(b=2, n=4, k=6, m=4) -> None:
    """A[2,4,6] @ B[6,4] → Z[2,4,4], Relu(Z) → Y[2,4,4].
    B is 2-D (broadcasts across A's batch). outer_count=1, batch=2.
    """
    b_data = (np.random.randn(k, m) * 0.5).astype(np.float32)
    b_init = numpy_helper.from_array(b_data, name="B")

    mm   = helper.make_node("MatMul", inputs=["A", "B"], outputs=["Z"])
    relu = helper.make_node("Relu",   inputs=["Z"],       outputs=["Y"])

    graph = helper.make_graph(
        [mm, relu],
        "batch_matmul_relu",
        inputs=[helper.make_tensor_value_info("A", TensorProto.FLOAT, [b, n, k])],
        outputs=[helper.make_tensor_value_info("Y", TensorProto.FLOAT, [b, n, m])],
        initializer=[b_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    _save(model, "mixed_batch_matmul_relu.onnx")


def gen_sub_div_matmul(n=4, k=6, m=4) -> None:
    """Sub(X[4,6], offset[6]) → Div(S[4,6], scale[6]) → MatMul(D[4,6], W[6,4]) → Y[4,4].
    Sub and Div both broadcast (outer_count=4, chunk=6, aligned=8), producing gap=2.
    Y: flat(16) — MatmulNode output (Fix A).
    """
    offset_data = np.ones(k, dtype=np.float32) * 0.5
    scale_data  = np.ones(k, dtype=np.float32) * 0.5
    w_data      = (np.random.randn(k, m) * 0.5).astype(np.float32)
    offset_init = numpy_helper.from_array(offset_data, name="offset")
    scale_init  = numpy_helper.from_array(scale_data,  name="scale")
    w_init      = numpy_helper.from_array(w_data,      name="W")

    sub = helper.make_node("Sub",    inputs=["X", "offset"], outputs=["S"])
    div = helper.make_node("Div",    inputs=["S", "scale"],  outputs=["D"])
    mm  = helper.make_node("MatMul", inputs=["D", "W"],      outputs=["Y"])

    graph = helper.make_graph(
        [sub, div, mm],
        "sub_div_matmul",
        inputs=[helper.make_tensor_value_info("X", TensorProto.FLOAT, [n, k])],
        outputs=[helper.make_tensor_value_info("Y", TensorProto.FLOAT, [n, m])],
        initializer=[offset_init, scale_init, w_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    _save(model, "mixed_sub_div_matmul.onnx")


def gen_two_input_matmul(n=4, k=8, m=4) -> None:
    """Add(X1[4,8], X2[4,8]) → Z[4,8] (same-shape, non-broadcast),
    MatMul(Z[4,8], W[8,4]) → Y[4,4].
    Two graph inputs (X1, X2), one output (Y).  Z and Y are both flat.
    """
    w_data = (np.random.randn(k, m) * 0.5).astype(np.float32)
    w_init = numpy_helper.from_array(w_data, name="W")

    add = helper.make_node("Add",    inputs=["X1", "X2"], outputs=["Z"])
    mm  = helper.make_node("MatMul", inputs=["Z",  "W"],  outputs=["Y"])

    graph = helper.make_graph(
        [add, mm],
        "two_input_matmul",
        inputs=[
            helper.make_tensor_value_info("X1", TensorProto.FLOAT, [n, k]),
            helper.make_tensor_value_info("X2", TensorProto.FLOAT, [n, k]),
        ],
        outputs=[helper.make_tensor_value_info("Y", TensorProto.FLOAT, [n, m])],
        initializer=[w_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    _save(model, "mixed_two_input_matmul.onnx")


def gen_two_output(n=4, k=8, m=4) -> None:
    """X[4,8] → MatMul(X, W[8,4]) → Z[4,4],
    Add(Z, bias[4]) → Yadd[4,4]  (broadcast, outer=4, chunk=4, gap=4),
    Relu(Z)         → Yrelu[4,4] (phase-3 propagation from Z).
    One input (X), two outputs (Yadd, Yrelu).
    Z.alloc=32 (advancing, gap=4).  Yrelu inherits Z's layout.
    """
    w_data    = (np.random.randn(k, m) * 0.5).astype(np.float32)
    bias_data = np.zeros(m, dtype=np.float32)
    w_init    = numpy_helper.from_array(w_data,    name="W")
    bias_init = numpy_helper.from_array(bias_data, name="bias")

    mm   = helper.make_node("MatMul", inputs=["X",  "W"],    outputs=["Z"])
    add  = helper.make_node("Add",    inputs=["Z",  "bias"], outputs=["Yadd"])
    relu = helper.make_node("Relu",   inputs=["Z"],           outputs=["Yrelu"])

    graph = helper.make_graph(
        [mm, add, relu],
        "two_output",
        inputs=[helper.make_tensor_value_info("X", TensorProto.FLOAT, [n, k])],
        outputs=[
            helper.make_tensor_value_info("Yadd",  TensorProto.FLOAT, [n, m]),
            helper.make_tensor_value_info("Yrelu", TensorProto.FLOAT, [n, m]),
        ],
        initializer=[w_init, bias_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    _save(model, "mixed_two_output.onnx")


def gen_two_input_two_output(n=4, k=8, m=4) -> None:
    """X1[4,8] → MatMul(X1, W[8,4]) → Z[4,4],
    Add(Z, X2[4]) → Yadd[4,4]  (broadcast, outer=4, chunk=4; X2 is a graph input),
    Relu(Z)       → Yrelu[4,4] (phase-3 propagation from Z).
    Two inputs (X1, X2), two outputs (Yadd, Yrelu).
    Z.alloc=32 (advancing, gap=4).  X2.alloc=8 (repeating, n_chunks=1).
    INFERENCE_X2_SIZE = 4u (numel, not alloc).
    """
    w_data = (np.random.randn(k, m) * 0.5).astype(np.float32)
    w_init = numpy_helper.from_array(w_data, name="W")

    mm   = helper.make_node("MatMul", inputs=["X1", "W"],  outputs=["Z"])
    add  = helper.make_node("Add",    inputs=["Z",  "X2"], outputs=["Yadd"])
    relu = helper.make_node("Relu",   inputs=["Z"],         outputs=["Yrelu"])

    graph = helper.make_graph(
        [mm, add, relu],
        "two_input_two_output",
        inputs=[
            helper.make_tensor_value_info("X1", TensorProto.FLOAT, [n, k]),
            helper.make_tensor_value_info("X2", TensorProto.FLOAT, [m]),
        ],
        outputs=[
            helper.make_tensor_value_info("Yadd",  TensorProto.FLOAT, [n, m]),
            helper.make_tensor_value_info("Yrelu", TensorProto.FLOAT, [n, m]),
        ],
        initializer=[w_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    _save(model, "mixed_two_input_two_output.onnx")


def gen_spatial_matmul_relu() -> None:
    """X[1,3,224,224] @ W[224,10] → Z[1,3,224,10] → Relu → Y[1,3,224,10].

    4-D × 2-D batched MatMul: the last two dims of X are the matrix dims
    [n=224, k=224]; leading dims [1,3] are the batch → batch = 1*3 = 3.
    W (2-D) broadcasts across all leading dims of X.
    MatmulKernel: n=224, k=224, m=10, batch=3,
                  a_batch_stride=50176, b_batch_stride=0, c_batch_stride=2240.
    Relu is a plain non-broadcast VectorOPKernel call on Z → Y (6720 elements).
    """
    w_data = (np.random.randn(224, 10) * 0.01).astype(np.float32)
    w_init = numpy_helper.from_array(w_data, name="W")

    mm   = helper.make_node("MatMul", inputs=["X", "W"], outputs=["Z"])
    relu = helper.make_node("Relu",   inputs=["Z"],      outputs=["Y"])

    graph = helper.make_graph(
        [mm, relu],
        "spatial_matmul_relu",
        inputs=[helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 224, 224])],
        outputs=[helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3, 224, 10])],
        initializer=[w_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    _save(model, "mixed_spatial_matmul_relu.onnx")


def gen_skip_connection(n=4, k=8) -> None:
    """X[4,8] → Z=MatMul(X, W[8,8]) → R=Relu(Z) → Y=Add(X, R): Y = X + Relu(X @ W).

    True skip/residual connection: X flows through both the MatMul path and the
    identity path, then the two paths are summed.  W is square so the MatMul
    output has the same shape as X, making the Add non-broadcast.

    MatmulKernel: n=4, k=8, m=8, batch=1, all strides=0 (plain 2-D).
    VectorOPKernel (Relu): run_op(Z, NULL, R, 32u, VECTOROP_RELU).
    VectorOPKernel (Add):  run_op(X, R, Y, 32u, VECTOROP_ADD).
    All layouts flat (n_chunks=1, gap=0).
    """
    w_data = (np.random.randn(k, k) * 0.5).astype(np.float32)
    w_init = numpy_helper.from_array(w_data, name="W")

    mm   = helper.make_node("MatMul", inputs=["X", "W"], outputs=["Z"])
    relu = helper.make_node("Relu",   inputs=["Z"],      outputs=["R"])
    add  = helper.make_node("Add",    inputs=["X", "R"], outputs=["Y"])

    graph = helper.make_graph(
        [mm, relu, add],
        "skip_connection",
        inputs=[helper.make_tensor_value_info("X", TensorProto.FLOAT, [n, k])],
        outputs=[helper.make_tensor_value_info("Y", TensorProto.FLOAT, [n, k])],
        initializer=[w_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    _save(model, "mixed_skip_connection.onnx")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Generating mixed-kernel test models...")
    gen_matmul_then_relu()
    gen_add_then_matmul()
    gen_matmul_add_relu()
    gen_two_layer_mlp()
    gen_add_matmul_unaligned()
    gen_matmul_scale_bias()
    gen_outer_matmul_relu()
    gen_relu6_matmul_add()
    gen_residual()
    gen_batch_matmul_relu()
    gen_sub_div_matmul()
    gen_two_input_matmul()
    gen_two_output()
    gen_two_input_two_output()
    gen_spatial_matmul_relu()
    gen_skip_connection()
    print("Done.")
