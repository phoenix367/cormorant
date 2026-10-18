# Copyright 2025 GradeBuilder SL
# SPDX-License-Identifier: Apache-2.0

"""Generate ONNX test models covering Reshape (buffer-alias) and Gemm (→MatMul+Add)."""

import os
import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as nph
from onnx import TensorProto

OUT_DIR = os.path.join(os.path.dirname(__file__), "models")


def _save(model, name: str) -> None:
    onnx.checker.check_model(model)
    out = os.path.join(OUT_DIR, name)
    onnx.save(model, out)
    print(f"  {out}")


def _vi(name: str, shape) -> onnx.ValueInfoProto:
    return oh.make_tensor_value_info(name, TensorProto.FLOAT, shape)


def _init(name: str, data: np.ndarray) -> onnx.TensorProto:
    return nph.from_array(data.astype(np.float32), name=name)


def _opset(v: int = 13):
    return [oh.make_opsetid("", v)]


# ---------------------------------------------------------------------------
# reshape_flatten: pure Reshape, [1,4,4,4] → [1,64]
# Tests that Reshape produces a ReshapeNode (buffer alias).
# ---------------------------------------------------------------------------
def gen_reshape_flatten() -> None:
    shape_data = np.array([1, 64], dtype=np.int64)
    shape_init = nph.from_array(shape_data, name="shape")
    node = oh.make_node("Reshape", inputs=["X", "shape"], outputs=["Z"])
    graph = oh.make_graph(
        [node], "reshape_flatten",
        inputs=[_vi("X", [1, 4, 4, 4])],
        outputs=[_vi("Z", [1, 64])],
        initializer=[shape_init],
    )
    _save(oh.make_model(graph, opset_imports=_opset()), "reshape_flatten.onnx")


# ---------------------------------------------------------------------------
# reshape_then_matmul: Reshape → MatMul
# X[1,4,4,4] → Z[1,64] (alias) → Y[1,8] (MatMul with W[64,8])
# ---------------------------------------------------------------------------
def gen_reshape_then_matmul() -> None:
    w_data  = np.full((64, 8), 1.0 / 64, dtype=np.float32)
    shape_data = np.array([1, 64], dtype=np.int64)
    w_init     = _init("W", w_data)
    shape_init = nph.from_array(shape_data, name="shape")

    reshape = oh.make_node("Reshape", inputs=["X", "shape"], outputs=["Z"])
    matmul  = oh.make_node("MatMul",  inputs=["Z", "W"],     outputs=["Y"])
    graph = oh.make_graph(
        [reshape, matmul], "reshape_then_matmul",
        inputs=[_vi("X", [1, 4, 4, 4])],
        outputs=[_vi("Y", [1, 8])],
        initializer=[w_init, shape_init],
    )
    _save(oh.make_model(graph, opset_imports=_opset()), "reshape_then_matmul.onnx")


# ---------------------------------------------------------------------------
# gemm_no_bias: Gemm without C operand → decomposes to MatMul only
# X[1,64] @ W[64,8] → Y[1,8]
# ---------------------------------------------------------------------------
def gen_gemm_no_bias() -> None:
    w_data = np.full((64, 8), 1.0 / 64, dtype=np.float32)
    w_init = _init("W", w_data)
    node = oh.make_node("Gemm", inputs=["X", "W"], outputs=["Y"])
    graph = oh.make_graph(
        [node], "gemm_no_bias",
        inputs=[_vi("X", [1, 64])],
        outputs=[_vi("Y", [1, 8])],
        initializer=[w_init],
    )
    _save(oh.make_model(graph, opset_imports=_opset()), "gemm_no_bias.onnx")


# ---------------------------------------------------------------------------
# gemm_with_bias: Gemm(X, W, B) → decomposes to MatMul + Add
# X[1,64] @ W[64,8] + B[1,8] → Y[1,8]
# ---------------------------------------------------------------------------
def gen_gemm_with_bias() -> None:
    w_data = np.full((64, 8), 1.0 / 64, dtype=np.float32)
    b_data = np.full((1, 8), 0.5, dtype=np.float32)
    w_init = _init("W", w_data)
    b_init = _init("B", b_data)
    node = oh.make_node("Gemm", inputs=["X", "W", "B"], outputs=["Y"])
    graph = oh.make_graph(
        [node], "gemm_with_bias",
        inputs=[_vi("X", [1, 64])],
        outputs=[_vi("Y", [1, 8])],
        initializer=[w_init, b_init],
    )
    _save(oh.make_model(graph, opset_imports=_opset()), "gemm_with_bias.onnx")


# ---------------------------------------------------------------------------
# gemm_chain: two Gemm(with bias) nodes in sequence
# X[1,32] → H[1,16] → Y[1,8]
# ---------------------------------------------------------------------------
def gen_gemm_chain() -> None:
    w1 = np.full((32, 16), 1.0 / 32, dtype=np.float32)
    b1 = np.zeros((1, 16), dtype=np.float32)
    w2 = np.full((16, 8), 1.0 / 16, dtype=np.float32)
    b2 = np.zeros((1, 8), dtype=np.float32)
    inits = [_init("W1", w1), _init("B1", b1), _init("W2", w2), _init("B2", b2)]

    g1 = oh.make_node("Gemm", inputs=["X", "W1", "B1"], outputs=["H"])
    g2 = oh.make_node("Gemm", inputs=["H", "W2", "B2"], outputs=["Y"])
    graph = oh.make_graph(
        [g1, g2], "gemm_chain",
        inputs=[_vi("X", [1, 32])],
        outputs=[_vi("Y", [1, 8])],
        initializer=inits,
    )
    _save(oh.make_model(graph, opset_imports=_opset()), "gemm_chain.onnx")


# ---------------------------------------------------------------------------
# reshape_gemm_pipeline: Relu → Reshape → Gemm (mini CNN head)
# X[1,8,4,4] → (Relu) R[1,8,4,4] → (Reshape) F[1,128] → (Gemm+bias) Y[1,4]
# ---------------------------------------------------------------------------
def gen_reshape_gemm_pipeline() -> None:
    w_data     = np.full((128, 4), 1.0 / 128, dtype=np.float32)
    b_data     = np.zeros((1, 4), dtype=np.float32)
    shape_data = np.array([1, 128], dtype=np.int64)
    inits = [_init("W", w_data), _init("B", b_data),
             nph.from_array(shape_data, name="shape")]

    relu    = oh.make_node("Relu",    inputs=["X"],         outputs=["R"])
    reshape = oh.make_node("Reshape", inputs=["R", "shape"], outputs=["F"])
    gemm    = oh.make_node("Gemm",    inputs=["F", "W", "B"], outputs=["Y"])
    graph = oh.make_graph(
        [relu, reshape, gemm], "reshape_gemm_pipeline",
        inputs=[_vi("X", [1, 8, 4, 4])],
        outputs=[_vi("Y", [1, 4])],
        initializer=inits,
    )
    _save(oh.make_model(graph, opset_imports=_opset()), "reshape_gemm_pipeline.onnx")


# ---------------------------------------------------------------------------
# conv_then_squeeze: Conv → Squeeze → graph output (terminal alias bug)
# X[1,4,1,1] → Conv(W[2,4,1,1], B[2]) → C[1,2,1,1] → Squeeze(axes=[2,3]) → Y[1,2]
# The graph output Y is the terminal reshape alias of the Conv output C.
# This model exercises the output-alias redirect code path in inference_run().
# ---------------------------------------------------------------------------
def gen_conv_then_squeeze() -> None:
    w_data = np.eye(2, 4, dtype=np.float32)  # [2, 4] as [2, 4, 1, 1]
    w_data = w_data.reshape(2, 4, 1, 1)
    b_data = np.zeros(2, dtype=np.float32)
    w_init = _init("W", w_data)
    b_init = _init("B", b_data)

    conv    = oh.make_node("Conv",    inputs=["X", "W", "B"], outputs=["C"],
                           kernel_shape=[1, 1])
    squeeze = oh.make_node("Squeeze", inputs=["C"],           outputs=["Y"],
                           axes=[2, 3])
    graph = oh.make_graph(
        [conv, squeeze], "conv_then_squeeze",
        inputs=[_vi("X", [1, 4, 1, 1])],
        outputs=[_vi("Y", [1, 2])],
        initializer=[w_init, b_init],
    )
    _save(oh.make_model(graph, opset_imports=_opset(7)), "conv_then_squeeze.onnx")


# ---------------------------------------------------------------------------
# squeeze_then_matmul: GlobalAveragePool → Squeeze → MatMul (MobileNet head)
# X[1,4,4,4] → Pool → P[1,4,1,1] → Squeeze(axes=[2,3]) → F[1,4] → Y[1,8]
# ---------------------------------------------------------------------------
def gen_squeeze_then_matmul() -> None:
    w_data = np.full((4, 8), 0.25, dtype=np.float32)
    w_init = _init("W", w_data)

    pool    = oh.make_node("GlobalAveragePool", inputs=["X"],       outputs=["P"])
    squeeze = oh.make_node("Squeeze",           inputs=["P"],       outputs=["F"],
                           axes=[2, 3])
    matmul  = oh.make_node("MatMul",            inputs=["F", "W"],  outputs=["Y"])
    graph = oh.make_graph(
        [pool, squeeze, matmul], "squeeze_then_matmul",
        inputs=[_vi("X", [1, 4, 4, 4])],
        outputs=[_vi("Y", [1, 8])],
        initializer=[w_init],
    )
    _save(oh.make_model(graph, opset_imports=_opset(7)), "squeeze_then_matmul.onnx")


# ---------------------------------------------------------------------------
# unsqueeze_then_relu: Unsqueeze → Relu  (adds size-1 dims, then activates)
# X[1,4] → Unsqueeze(axes=[2,3]) → E[1,4,1,1] → Relu → Y[1,4,1,1]
# ---------------------------------------------------------------------------
def gen_unsqueeze_then_relu() -> None:
    unsqueeze = oh.make_node("Unsqueeze", inputs=["X"],  outputs=["E"], axes=[2, 3])
    relu      = oh.make_node("Relu",      inputs=["E"],  outputs=["Y"])
    graph = oh.make_graph(
        [unsqueeze, relu], "unsqueeze_then_relu",
        inputs=[_vi("X", [1, 4])],
        outputs=[_vi("Y", [1, 4, 1, 1])],
    )
    _save(oh.make_model(graph, opset_imports=_opset(7)), "unsqueeze_then_relu.onnx")


# ---------------------------------------------------------------------------
# relu_dropout_relu: Relu → Dropout → Relu  (inference passthrough)
# X[1,8] → Relu → D[1,8] → Dropout(ratio=0.5) → Z[1,8] → Relu → Y[1,8]
# In inference mode Dropout is identity; Y == Relu(X).
# ---------------------------------------------------------------------------
def gen_relu_dropout_relu() -> None:
    relu1   = oh.make_node("Relu",    inputs=["X"],  outputs=["R"])
    dropout = oh.make_node("Dropout", inputs=["R"],  outputs=["D"], ratio=0.5)
    relu2   = oh.make_node("Relu",    inputs=["D"],  outputs=["Y"])
    graph = oh.make_graph(
        [relu1, dropout, relu2], "relu_dropout_relu",
        inputs=[_vi("X", [1, 8])],
        outputs=[_vi("Y", [1, 8])],
    )
    _save(oh.make_model(graph, opset_imports=_opset(11)), "relu_dropout_relu.onnx")


# ---------------------------------------------------------------------------
# gemm_transA_unsupported: transA=1 → must raise SchedulerError
# ---------------------------------------------------------------------------
def gen_gemm_transA_unsupported() -> None:
    w_data = np.eye(8, dtype=np.float32)
    w_init = _init("W", w_data)
    # transA=1: A must be [8,1] so that A^T = [1,8] @ W[8,8] = [1,8]
    node = oh.make_node("Gemm", inputs=["X", "W"], outputs=["Y"], transA=1)
    graph = oh.make_graph(
        [node], "gemm_transA_unsupported",
        inputs=[_vi("X", [8, 1])],
        outputs=[_vi("Y", [1, 8])],
        initializer=[w_init],
    )
    _save(oh.make_model(graph, opset_imports=_opset()), "gemm_transA_unsupported.onnx")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

ALL_GENERATORS = [
    gen_reshape_flatten,
    gen_reshape_then_matmul,
    gen_gemm_no_bias,
    gen_gemm_with_bias,
    gen_gemm_chain,
    gen_reshape_gemm_pipeline,
    gen_conv_then_squeeze,
    gen_squeeze_then_matmul,
    gen_unsqueeze_then_relu,
    gen_relu_dropout_relu,
    gen_gemm_transA_unsupported,
]


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Generating reshape/Gemm test models in {OUT_DIR}/")
    for gen in ALL_GENERATORS:
        gen()
    print("Done.")


if __name__ == "__main__":
    main()
