"""Generate ONNX test models that use ConvKernel nodes."""

import os
import sys

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

OUT_DIR = os.path.join(os.path.dirname(__file__), "models")

# Pull bounds from the platform JSON so violator/at-limit models always
# size against the active ConvKernel configuration.  Avoids the trap of a
# hard-coded out_ch=1025 silently becoming a legal value the moment the
# JSON max_out_ch is bumped (which would make the corresponding "must
# raise" test pass even though the validator never fired).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src._conv_hw_config import (   # noqa: E402
    CONV_MAX_IN_CH,
    CONV_MAX_OUT_CH,
    CONV_MAX_LINE_BUF_ROWS,
    CONV_MAX_LINE_BUF_COLS,
    CONV_MAX_ACC_PERSIST_ENTRIES,
)


def _save(model, name: str) -> None:
    onnx.checker.check_model(model)
    out = os.path.join(OUT_DIR, name)
    onnx.save(model, out)
    print(f"  {out}")


def _vi(name: str, shape) -> onnx.TensorProto:
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)


# ---------------------------------------------------------------------------
# conv_simple: 1x1 conv, no bias, batch=1
# X[1,4,8,8] * W[8,4,1,1] → Y[1,8,8,8]
# ---------------------------------------------------------------------------
def gen_conv_simple() -> None:
    w_data = (np.random.randn(8, 4, 1, 1) * 0.25).astype(np.float32)
    w_init = numpy_helper.from_array(w_data, name="W")

    conv = helper.make_node(
        "Conv",
        inputs=["X", "W"],
        outputs=["Y"],
        kernel_shape=[1, 1],
    )
    graph = helper.make_graph(
        [conv], "conv_simple",
        inputs=[_vi("X", [1, 4, 8, 8])],
        outputs=[_vi("Y", [1, 8, 8, 8])],
        initializer=[w_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    _save(model, "conv_simple.onnx")


# ---------------------------------------------------------------------------
# conv_with_bias: 3x3 conv with bias, stride=1, no padding
# X[1,4,8,8] * W[6,4,3,3] + B[6] → Y[1,6,6,6]
# ---------------------------------------------------------------------------
def gen_conv_with_bias() -> None:
    w_data = (np.random.randn(6, 4, 3, 3) * 0.25).astype(np.float32)
    b_data = np.zeros(6, dtype=np.float32)
    w_init = numpy_helper.from_array(w_data, name="W")
    b_init = numpy_helper.from_array(b_data, name="B")

    conv = helper.make_node(
        "Conv",
        inputs=["X", "W", "B"],
        outputs=["Y"],
        kernel_shape=[3, 3],
        strides=[1, 1],
        pads=[0, 0, 0, 0],
    )
    graph = helper.make_graph(
        [conv], "conv_with_bias",
        inputs=[_vi("X", [1, 4, 8, 8])],
        outputs=[_vi("Y", [1, 6, 6, 6])],
        initializer=[w_init, b_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    _save(model, "conv_with_bias.onnx")


# ---------------------------------------------------------------------------
# conv_stride2: 3x3 conv, stride=2, no padding
# X[1,4,8,8] * W[8,4,3,3] → Y[1,8,3,3]
# ---------------------------------------------------------------------------
def gen_conv_stride2() -> None:
    w_data = (np.random.randn(8, 4, 3, 3) * 0.25).astype(np.float32)
    w_init = numpy_helper.from_array(w_data, name="W")

    conv = helper.make_node(
        "Conv",
        inputs=["X", "W"],
        outputs=["Y"],
        kernel_shape=[3, 3],
        strides=[2, 2],
        pads=[0, 0, 0, 0],
    )
    graph = helper.make_graph(
        [conv], "conv_stride2",
        inputs=[_vi("X", [1, 4, 8, 8])],
        outputs=[_vi("Y", [1, 8, 3, 3])],
        initializer=[w_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    _save(model, "conv_stride2.onnx")


# ---------------------------------------------------------------------------
# conv_padded: 3x3 conv, stride=1, SAME_UPPER padding
# X[1,4,8,8] * W[8,4,3,3] → Y[1,8,8,8]
# ---------------------------------------------------------------------------
def gen_conv_padded() -> None:
    w_data = (np.random.randn(8, 4, 3, 3) * 0.25).astype(np.float32)
    w_init = numpy_helper.from_array(w_data, name="W")

    conv = helper.make_node(
        "Conv",
        inputs=["X", "W"],
        outputs=["Y"],
        kernel_shape=[3, 3],
        strides=[1, 1],
        pads=[1, 1, 1, 1],
    )
    graph = helper.make_graph(
        [conv], "conv_padded",
        inputs=[_vi("X", [1, 4, 8, 8])],
        outputs=[_vi("Y", [1, 8, 8, 8])],
        initializer=[w_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    _save(model, "conv_padded.onnx")


# ---------------------------------------------------------------------------
# conv_batch2: 1x1 conv, batch=2
# X[2,4,4,4] * W[8,4,1,1] → Y[2,8,4,4]
# ---------------------------------------------------------------------------
def gen_conv_batch2() -> None:
    w_data = (np.random.randn(8, 4, 1, 1) * 0.25).astype(np.float32)
    w_init = numpy_helper.from_array(w_data, name="W")

    conv = helper.make_node(
        "Conv",
        inputs=["X", "W"],
        outputs=["Y"],
        kernel_shape=[1, 1],
    )
    graph = helper.make_graph(
        [conv], "conv_batch2",
        inputs=[_vi("X", [2, 4, 4, 4])],
        outputs=[_vi("Y", [2, 8, 4, 4])],
        initializer=[w_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    _save(model, "conv_batch2.onnx")


# ---------------------------------------------------------------------------
# conv_then_relu: Conv → Relu (ConvKernel then VectorOPKernel)
# X[1,4,8,8] * W[8,4,1,1] → Z[1,8,8,8] → Relu → Y[1,8,8,8]
# ---------------------------------------------------------------------------
def gen_conv_then_relu() -> None:
    w_data = (np.random.randn(8, 4, 1, 1) * 0.25).astype(np.float32)
    w_init = numpy_helper.from_array(w_data, name="W")

    conv = helper.make_node("Conv",  inputs=["X", "W"], outputs=["Z"], kernel_shape=[1, 1])
    relu = helper.make_node("Relu",  inputs=["Z"],      outputs=["Y"])

    graph = helper.make_graph(
        [conv, relu], "conv_then_relu",
        inputs=[_vi("X", [1, 4, 8, 8])],
        outputs=[_vi("Y", [1, 8, 8, 8])],
        initializer=[w_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    _save(model, "conv_then_relu.onnx")


# ---------------------------------------------------------------------------
# conv_mnist_first_layer: Conv(+bias) → Relu  (MNIST-sized input)
# X[1,1,28,28] * W[8,1,5,5] + B[8] → Z[1,8,24,24] → Relu → Y[1,8,24,24]
# Classic first conv layer: 5×5 kernel, no padding, stride 1.
# ---------------------------------------------------------------------------
def gen_conv_mnist_first_layer() -> None:
    w_data = (np.random.randn(8, 1, 5, 5) * 0.1).astype(np.float32)
    b_data = np.zeros(8, dtype=np.float32)
    w_init = numpy_helper.from_array(w_data, name="W")
    b_init = numpy_helper.from_array(b_data, name="B")

    conv = helper.make_node("Conv",  inputs=["X", "W", "B"], outputs=["Z"], kernel_shape=[5, 5])
    relu = helper.make_node("Relu",  inputs=["Z"],           outputs=["Y"])

    graph = helper.make_graph(
        [conv, relu], "conv_mnist_first_layer",
        inputs=[_vi("X", [1, 1, 28, 28])],
        outputs=[_vi("Y", [1, 8, 24, 24])],
        initializer=[w_init, b_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    _save(model, "conv_mnist_first_layer.onnx")


# ---------------------------------------------------------------------------
# conv_mnist_second_layer: Conv(+bias) → Relu  (MNIST-sized second conv layer)
# X[1,8,14,14] * W[16,8,5,5] + B[16] → Z[1,16,10,10] → Relu → Y[1,16,10,10]
# Follows conv_mnist_first_layer (after 2×2 max-pool on its output).
# ---------------------------------------------------------------------------
def gen_conv_mnist_second_layer() -> None:
    w_data = (np.random.randn(16, 8, 5, 5) * 0.1).astype(np.float32)
    b_data = np.zeros(16, dtype=np.float32)
    w_init = numpy_helper.from_array(w_data, name="W")
    b_init = numpy_helper.from_array(b_data, name="B")

    conv = helper.make_node("Conv",  inputs=["X", "W", "B"], outputs=["Z"], kernel_shape=[5, 5])
    relu = helper.make_node("Relu",  inputs=["Z"],           outputs=["Y"])

    graph = helper.make_graph(
        [conv, relu], "conv_mnist_second_layer",
        inputs=[_vi("X", [1, 8, 14, 14])],
        outputs=[_vi("Y", [1, 16, 10, 10])],
        initializer=[w_init, b_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    _save(model, "conv_mnist_second_layer.onnx")


# ---------------------------------------------------------------------------
# conv_then_add_flat: Conv → Add with same-shape weight (non-broadcast)
# Z[1,8,8,8] + scale[1,8,8,8] → Y  (no broadcasting, flat layout)
# ---------------------------------------------------------------------------
def gen_conv_then_add_flat() -> None:
    w_data     = (np.random.randn(8, 4, 1, 1) * 0.25).astype(np.float32)
    scale_data = np.ones((1, 8, 8, 8), dtype=np.float32)
    w_init     = numpy_helper.from_array(w_data,     name="W")
    scale_init = numpy_helper.from_array(scale_data, name="scale")

    conv = helper.make_node("Conv", inputs=["X", "W"],       outputs=["Z"], kernel_shape=[1, 1])
    add  = helper.make_node("Add",  inputs=["Z", "scale"],   outputs=["Y"])

    graph = helper.make_graph(
        [conv, add], "conv_then_add_flat",
        inputs=[_vi("X", [1, 4, 8, 8])],
        outputs=[_vi("Y", [1, 8, 8, 8])],
        initializer=[w_init, scale_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    _save(model, "conv_then_add_flat.onnx")


# ---------------------------------------------------------------------------
# conv_relu_chain: Conv (no bias) → Relu → Conv (with bias) → Relu
# Tests two-stage pipeline with both kernel types
# X[1,4,8,8] → [1,8,8,8] → [1,4,8,8]
# ---------------------------------------------------------------------------
def gen_conv_relu_chain() -> None:
    w1_data = (np.random.randn(8, 4, 1, 1) * 0.25).astype(np.float32)
    w2_data = (np.random.randn(4, 8, 1, 1) * 0.25).astype(np.float32)
    b2_data = np.zeros(4, dtype=np.float32)
    w1_init = numpy_helper.from_array(w1_data, name="W1")
    w2_init = numpy_helper.from_array(w2_data, name="W2")
    b2_init = numpy_helper.from_array(b2_data, name="B2")

    conv1 = helper.make_node("Conv", inputs=["X", "W1"],      outputs=["Z1"], kernel_shape=[1, 1])
    relu1 = helper.make_node("Relu", inputs=["Z1"],            outputs=["Z2"])
    conv2 = helper.make_node("Conv", inputs=["Z2", "W2", "B2"], outputs=["Z3"], kernel_shape=[1, 1])
    relu2 = helper.make_node("Relu", inputs=["Z3"],            outputs=["Y"])

    graph = helper.make_graph(
        [conv1, relu1, conv2, relu2], "conv_relu_chain",
        inputs=[_vi("X", [1, 4, 8, 8])],
        outputs=[_vi("Y", [1, 4, 8, 8])],
        initializer=[w1_init, w2_init, b2_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    _save(model, "conv_relu_chain.onnx")


# ---------------------------------------------------------------------------
# conv_depthwise: depthwise convolution (group == in_channels == 4)
# Supported by ConvKernel (is_depthwise=1 path).
# X[1,4,8,8] * W[4,1,3,3] → Y[1,4,6,6]  groups=4
# ---------------------------------------------------------------------------
def gen_conv_depthwise() -> None:
    w_data = (np.random.randn(4, 1, 3, 3) * 0.25).astype(np.float32)
    w_init = numpy_helper.from_array(w_data, name="W")

    conv = helper.make_node(
        "Conv",
        inputs=["X", "W"],
        outputs=["Y"],
        kernel_shape=[3, 3],
        group=4,
    )
    graph = helper.make_graph(
        [conv], "conv_depthwise",
        inputs=[_vi("X", [1, 4, 8, 8])],
        outputs=[_vi("Y", [1, 4, 6, 6])],
        initializer=[w_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    _save(model, "conv_depthwise.onnx")


# ---------------------------------------------------------------------------
# conv_depthwise_bias: depthwise convolution with bias
# X[1,8,8,8] * W[8,1,3,3] + B[8] → Y[1,8,6,6]  groups=8
# ---------------------------------------------------------------------------
def gen_conv_depthwise_bias() -> None:
    w_data = (np.random.randn(8, 1, 3, 3) * 0.25).astype(np.float32)
    b_data = np.zeros(8, dtype=np.float32)
    w_init = numpy_helper.from_array(w_data, name="W")
    b_init = numpy_helper.from_array(b_data, name="B")

    conv = helper.make_node(
        "Conv",
        inputs=["X", "W", "B"],
        outputs=["Y"],
        kernel_shape=[3, 3],
        group=8,
    )
    graph = helper.make_graph(
        [conv], "conv_depthwise_bias",
        inputs=[_vi("X", [1, 8, 8, 8])],
        outputs=[_vi("Y", [1, 8, 6, 6])],
        initializer=[w_init, b_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    _save(model, "conv_depthwise_bias.onnx")


# ---------------------------------------------------------------------------
# conv_grouped_invalid: grouped conv with group != 1 and group != in_ch
# Not supported by ConvKernel — must raise SchedulerError.
# X[1,4,8,8] * W[4,2,3,3] → Y[1,4,6,6]  groups=2 (partial grouping)
# ---------------------------------------------------------------------------
def gen_conv_grouped_invalid() -> None:
    w_data = (np.random.randn(4, 2, 3, 3) * 0.25).astype(np.float32)
    w_init = numpy_helper.from_array(w_data, name="W")

    conv = helper.make_node(
        "Conv",
        inputs=["X", "W"],
        outputs=["Y"],
        kernel_shape=[3, 3],
        group=2,
    )
    graph = helper.make_graph(
        [conv], "conv_grouped_invalid",
        inputs=[_vi("X", [1, 4, 8, 8])],
        outputs=[_vi("Y", [1, 4, 6, 6])],
        initializer=[w_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    _save(model, "conv_grouped_invalid.onnx")


# ---------------------------------------------------------------------------
# conv_dilation: 3x3 conv with dilation=2, no padding
# X[1,4,8,8] * W[8,4,3,3] → Y[1,8,4,4]  (eff. kernel 5x5 → 4 output pixels)
# ---------------------------------------------------------------------------
def gen_conv_dilation() -> None:
    w_data = (np.random.randn(8, 4, 3, 3) * 0.25).astype(np.float32)
    w_init = numpy_helper.from_array(w_data, name="W")

    conv = helper.make_node(
        "Conv",
        inputs=["X", "W"],
        outputs=["Y"],
        kernel_shape=[3, 3],
        strides=[1, 1],
        dilations=[2, 2],
        pads=[0, 0, 0, 0],
    )
    graph = helper.make_graph(
        [conv], "conv_dilation",
        inputs=[_vi("X", [1, 4, 8, 8])],
        outputs=[_vi("Y", [1, 8, 4, 4])],
        initializer=[w_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    _save(model, "conv_dilation.onnx")


# ---------------------------------------------------------------------------
# conv_auto_pad_valid: 3x3 conv with auto_pad=VALID
# Equivalent to pads=[0,0,0,0]
# X[1,4,8,8] → Y[1,8,6,6]
# ---------------------------------------------------------------------------
def gen_conv_auto_pad_valid() -> None:
    w_data = (np.random.randn(8, 4, 3, 3) * 0.25).astype(np.float32)
    w_init = numpy_helper.from_array(w_data, name="W")

    conv = helper.make_node(
        "Conv",
        inputs=["X", "W"],
        outputs=["Y"],
        kernel_shape=[3, 3],
        auto_pad="VALID",
    )
    graph = helper.make_graph(
        [conv], "conv_auto_pad_valid",
        inputs=[_vi("X", [1, 4, 8, 8])],
        outputs=[_vi("Y", [1, 8, 6, 6])],
        initializer=[w_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    _save(model, "conv_auto_pad_valid.onnx")


# ---------------------------------------------------------------------------
# conv_two_layer_vgg: VGG-style two-layer conv block, 224×224 input.
#
# Mirrors the first two convolutional layers of VGG-16:
#   Layer 1: X[1,3,224,224]   * W1[64,3,3,3]   → Z[1,64,222,222]  (no pad, stride=1)
#   Layer 2: Z[1,64,222,222]  * W2[64,64,3,3]  → Y[1,64,220,220]  (no pad, stride=1)
#
# The 64×64×3×3 second-layer weight shape is what motivates the first layer:
# the input has in_ch=3, so a 64→64 conv requires a preceding 3→64 lift.
#
# Weight sizes:
#   W1: 64×3×3×3   = 1 728 elements  (<  4096 threshold → embedded as C array)
#   W2: 64×64×3×3  = 36 864 elements (>  4096 threshold → external weights/W2.dat)
# ---------------------------------------------------------------------------
def gen_conv_two_layer_vgg() -> None:
    w1_data = (np.random.randn(64, 3,  3, 3) * 0.1).astype(np.float32)
    w2_data = (np.random.randn(64, 64, 3, 3) * 0.1).astype(np.float32)
    w1_init = numpy_helper.from_array(w1_data, name="W1")
    w2_init = numpy_helper.from_array(w2_data, name="W2")

    conv1 = helper.make_node(
        "Conv",
        inputs=["X", "W1"],
        outputs=["Z"],
        kernel_shape=[3, 3],
        strides=[1, 1],
        pads=[0, 0, 0, 0],
    )
    conv2 = helper.make_node(
        "Conv",
        inputs=["Z", "W2"],
        outputs=["Y"],
        kernel_shape=[3, 3],
        strides=[1, 1],
        pads=[0, 0, 0, 0],
    )
    graph = helper.make_graph(
        [conv1, conv2], "conv_two_layer_vgg",
        inputs=[_vi("X", [1, 3, 224, 224])],
        outputs=[_vi("Y", [1, 64, 220, 220])],
        initializer=[w1_init, w2_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    _save(model, "conv_two_layer_vgg.onnx")


# ---------------------------------------------------------------------------
# Hardware-bound violation models — must each raise SchedulerError when
# loaded by OnnxGraph.  The bounds come from
# platforms/kv260.json (kernels.conv): max_in_ch=1024, max_out_ch=1024,
# max_line_buf_rows=16, max_line_buf_cols=64, max_acc_persist_entries=65536.
#
# Each model violates exactly one bound by exactly one unit so the test
# can identify which constraint fired.  Geometries are otherwise minimal
# to keep ONNX shape-inference fast and the model files tiny.  Two
# boundary-ok models confirm the inequality is `≤` (limit value passes).
# ---------------------------------------------------------------------------
def gen_unsupported_in_ch_too_large() -> None:
    """in_ch one above kMaxInCh — must raise.

    Uses a 3x3 kernel (not pointwise) so the planned 1x1→MatMul
    transform won't intercept this fixture before constraint validation.
    """
    in_ch = CONV_MAX_IN_CH + 1
    w_data = np.zeros((4, in_ch, 3, 3), dtype=np.float32)
    w_init = numpy_helper.from_array(w_data, name="W")
    conv = helper.make_node(
        "Conv", inputs=["X", "W"], outputs=["Y"],
        kernel_shape=[3, 3],
    )
    graph = helper.make_graph(
        [conv], "conv_unsupported_in_ch",
        inputs=[_vi("X", [1, in_ch, 3, 3])],
        outputs=[_vi("Y", [1, 4, 1, 1])],
        initializer=[w_init],
    )
    _save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)]),
          "conv_unsupported_in_ch.onnx")


def gen_unsupported_out_ch_too_large() -> None:
    """out_ch one above kMaxOutCh — must raise.

    Uses a 3x3 kernel (not pointwise) so the planned 1x1→MatMul
    transform won't intercept this fixture before constraint validation.
    """
    out_ch = CONV_MAX_OUT_CH + 1
    w_data = np.zeros((out_ch, 4, 3, 3), dtype=np.float32)
    w_init = numpy_helper.from_array(w_data, name="W")
    conv = helper.make_node(
        "Conv", inputs=["X", "W"], outputs=["Y"],
        kernel_shape=[3, 3],
    )
    graph = helper.make_graph(
        [conv], "conv_unsupported_out_ch",
        inputs=[_vi("X", [1, 4, 3, 3])],
        outputs=[_vi("Y", [1, out_ch, 1, 1])],
        initializer=[w_init],
    )
    _save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)]),
          "conv_unsupported_out_ch.onnx")


def gen_unsupported_dil_h_overflows_line_buf() -> None:
    """Vertical kernel span one above kMaxLineBufRows — must raise.

    Uses kh=2 with dilation_h = MAX_ROWS, so span = (kh-1)*dh + 1
    = MAX_ROWS + 1, exactly one element over the line-buffer-row capacity.
    kh stays within max_kh=7 so the violation isolates the line-buffer-row
    constraint, not a (non-existent) kh constraint.
    """
    dh = CONV_MAX_LINE_BUF_ROWS
    span = dh + 1                       # (2-1)*dh + 1
    in_h = span                         # out_h = in_h - span + 1 = 1
    w_data = np.zeros((4, 4, 2, 1), dtype=np.float32)
    w_init = numpy_helper.from_array(w_data, name="W")
    conv = helper.make_node(
        "Conv", inputs=["X", "W"], outputs=["Y"],
        kernel_shape=[2, 1], dilations=[dh, 1],
    )
    graph = helper.make_graph(
        [conv], "conv_unsupported_dil_h",
        inputs=[_vi("X", [1, 4, in_h, 8])],
        outputs=[_vi("Y", [1, 4, 1, 8])],
        initializer=[w_init],
    )
    _save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)]),
          "conv_unsupported_dil_h.onnx")


def gen_unsupported_dil_w_overflows_line_buf() -> None:
    """Horizontal kernel span one above kMaxLineBufCols — must raise.

    kw=2 with dilation_w = MAX_COLS so span = MAX_COLS + 1.
    """
    dw = CONV_MAX_LINE_BUF_COLS
    span = dw + 1
    in_w = span                         # out_w = 1
    w_data = np.zeros((4, 4, 1, 2), dtype=np.float32)
    w_init = numpy_helper.from_array(w_data, name="W")
    conv = helper.make_node(
        "Conv", inputs=["X", "W"], outputs=["Y"],
        kernel_shape=[1, 2], dilations=[1, dw],
    )
    graph = helper.make_graph(
        [conv], "conv_unsupported_dil_w",
        inputs=[_vi("X", [1, 4, 8, in_w])],
        outputs=[_vi("Y", [1, 4, 8, 1])],
        initializer=[w_init],
    )
    _save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)]),
          "conv_unsupported_dil_w.onnx")


def gen_unsupported_acc_persist() -> None:
    """out_w * out_ch one above kMaxAccPersistEntries — must raise.

    Pads=1 with a 3x3 kernel preserves the spatial size, so picking
    out_w=256 and out_ch = floor(MAX/256) + 1 lands at exactly MAX+stride
    above the limit while keeping the weight tensor small.  Non-pointwise
    so the planned 1x1→MatMul transform won't intercept it.
    """
    out_w = 256
    # Smallest out_ch such that out_w * out_ch > MAX.
    out_ch = CONV_MAX_ACC_PERSIST_ENTRIES // out_w + 1
    w_data = np.zeros((out_ch, 1, 3, 3), dtype=np.float32)
    w_init = numpy_helper.from_array(w_data, name="W")
    conv = helper.make_node(
        "Conv", inputs=["X", "W"], outputs=["Y"],
        kernel_shape=[3, 3], pads=[1, 1, 1, 1],
    )
    graph = helper.make_graph(
        [conv], "conv_unsupported_acc_persist",
        inputs=[_vi("X", [1, 1, out_w, out_w])],
        outputs=[_vi("Y", [1, out_ch, out_w, out_w])],
        initializer=[w_init],
    )
    _save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)]),
          "conv_unsupported_acc_persist.onnx")


def gen_in_ch_at_limit() -> None:
    """Boundary-case: in_ch == kMaxInCh; must parse OK.

    Uses a 3x3 kernel (not pointwise) so the planned 1x1→MatMul
    transform won't intercept this fixture before constraint validation.
    """
    in_ch = CONV_MAX_IN_CH
    w_data = np.zeros((4, in_ch, 3, 3), dtype=np.float32)
    w_init = numpy_helper.from_array(w_data, name="W")
    conv = helper.make_node(
        "Conv", inputs=["X", "W"], outputs=["Y"],
        kernel_shape=[3, 3],
    )
    graph = helper.make_graph(
        [conv], "conv_in_ch_at_limit",
        inputs=[_vi("X", [1, in_ch, 3, 3])],
        outputs=[_vi("Y", [1, 4, 1, 1])],
        initializer=[w_init],
    )
    _save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)]),
          "conv_in_ch_at_limit.onnx")


def gen_dil_h_at_line_buf_limit() -> None:
    """Boundary-case: vertical kernel span == kMaxLineBufRows; must parse OK.

    kh=2 with dilation_h = MAX_ROWS - 1 gives span = MAX_ROWS exactly.
    """
    dh = CONV_MAX_LINE_BUF_ROWS - 1
    span = dh + 1                       # exactly MAX_ROWS
    in_h = span + 4                     # gives out_h = 5, mild non-degenerate
    out_h = in_h - span + 1
    w_data = np.zeros((4, 4, 2, 1), dtype=np.float32)
    w_init = numpy_helper.from_array(w_data, name="W")
    conv = helper.make_node(
        "Conv", inputs=["X", "W"], outputs=["Y"],
        kernel_shape=[2, 1], dilations=[dh, 1],
    )
    graph = helper.make_graph(
        [conv], "conv_dil_h_at_limit",
        inputs=[_vi("X", [1, 4, in_h, 8])],
        outputs=[_vi("Y", [1, 4, out_h, 8])],
        initializer=[w_init],
    )
    _save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)]),
          "conv_dil_h_at_limit.onnx")


def gen_acc_persist_at_limit() -> None:
    """Boundary-case: out_w * out_ch == kMaxAccPersistEntries; must parse OK.

    Uses a 3x3 kernel with pads=1 (preserves spatial size) so the
    planned 1x1→MatMul transform won't intercept this fixture.  out_w
    is chosen as the largest divisor of MAX ≤ 256 so the spatial extent
    stays small while the equality lands on the boundary exactly.
    """
    # Pick out_w as the largest divisor of MAX that is ≤ 256 — this
    # keeps the input tensor compact while letting out_ch = MAX / out_w
    # be an integer that hits the boundary exactly.
    out_w = max(d for d in range(1, 257) if CONV_MAX_ACC_PERSIST_ENTRIES % d == 0)
    out_ch = CONV_MAX_ACC_PERSIST_ENTRIES // out_w
    w_data = np.zeros((out_ch, 1, 3, 3), dtype=np.float32)
    w_init = numpy_helper.from_array(w_data, name="W")
    conv = helper.make_node(
        "Conv", inputs=["X", "W"], outputs=["Y"],
        kernel_shape=[3, 3], pads=[1, 1, 1, 1],
    )
    graph = helper.make_graph(
        [conv], "conv_acc_persist_at_limit",
        inputs=[_vi("X", [1, 1, out_w, out_w])],
        outputs=[_vi("Y", [1, out_ch, out_w, out_w])],
        initializer=[w_init],
    )
    _save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)]),
          "conv_acc_persist_at_limit.onnx")



# ---------------------------------------------------------------------------
# Weight-path / M-group coverage at scale (CONV_OPTIMISATION §2.32–§2.36):
# one-pixel "fully-connected" convolutions whose time is the weight stream,
# and a many-M-group layer that exercises the §2.35 w_cache prefetch across
# group and ic-tile boundaries.  Small weights keep ramp inputs unsaturated.
# ---------------------------------------------------------------------------
def _conv_model(name, x_shape, w_shape, y_shape, scale, seed, bias=False, **attrs):
    w = (np.random.RandomState(seed).uniform(-1.0, 1.0, size=w_shape) * scale).astype(np.float32)
    inits = [numpy_helper.from_array(w, name="W")]
    inputs = ["X", "W"]
    if bias:
        b = (np.random.RandomState(seed + 1).uniform(-1.0, 1.0, size=(w_shape[0],)) * 0.1).astype(np.float32)
        inits.append(numpy_helper.from_array(b, name="B")); inputs.append("B")
    node = helper.make_node("Conv", inputs=inputs, outputs=["Y"], **attrs)
    graph = helper.make_graph([node], name, inputs=[_vi("X", x_shape)], outputs=[_vi("Y", y_shape)], initializer=inits)
    _save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)]), name + ".onnx")


def gen_conv_fc_7x7_64to256() -> None:
    """X[1,64,7,7] * W[256,64,7,7] -> Y[1,256,1,1]: LeNet conv3-style
    fully-connected conv (1.6 MB of packed weights, 4 ic-tiles x 8 M-groups,
    one output pixel per slab)."""
    _conv_model("conv_fc_7x7_64to256", [1, 64, 7, 7], [256, 64, 7, 7], [1, 256, 1, 1],
                0.004, 21, bias=True, kernel_shape=[7, 7])


def gen_conv_1x1_classifier_1024() -> None:
    """X[1,128,1,1] * W[1024,128,1,1] -> Y[1,1024,1,1]: 1x1 classifier
    (8 ic-tiles x 32 M-groups, one pixel)."""
    _conv_model("conv_1x1_classifier_1024", [1, 128, 1, 1], [1024, 128, 1, 1], [1, 1024, 1, 1],
                0.02, 22, bias=True, kernel_shape=[1, 1])


def gen_conv_mgroups_prefetch() -> None:
    """X[1,24,12,12] * W[80,24,3,3] pad1 -> Y[1,80,12,12]: 2 ic-tiles
    (16 + 8, half-tile last) x 3 M-groups so the w_cache prefetch crosses
    every (ict, mg) boundary with a partial last group."""
    _conv_model("conv_mgroups_prefetch", [1, 24, 12, 12], [80, 24, 3, 3], [1, 80, 12, 12],
                0.02, 23, bias=False, kernel_shape=[3, 3], pads=[1, 1, 1, 1])


    print("Done.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=OUT_DIR)
    args = parser.parse_args()
    OUT_DIR = args.out_dir
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Generating Conv test models...")
    gen_conv_simple()
    gen_conv_with_bias()
    gen_conv_stride2()
    gen_conv_padded()
    gen_conv_batch2()
    gen_conv_mnist_first_layer()
    gen_conv_mnist_second_layer()
    gen_conv_then_relu()
    gen_conv_then_add_flat()
    gen_conv_relu_chain()
    gen_conv_depthwise()
    gen_conv_depthwise_bias()
    gen_conv_grouped_invalid()
    gen_conv_dilation()
    gen_conv_auto_pad_valid()
    gen_conv_two_layer_vgg()
    gen_unsupported_in_ch_too_large()
    gen_unsupported_out_ch_too_large()
    gen_unsupported_dil_h_overflows_line_buf()
    gen_unsupported_dil_w_overflows_line_buf()
    gen_unsupported_acc_persist()
    gen_in_ch_at_limit()
    gen_dil_h_at_line_buf_limit()
    gen_acc_persist_at_limit()
    gen_conv_fc_7x7_64to256,
    gen_conv_1x1_classifier_1024,
    gen_conv_mgroups_prefetch,
