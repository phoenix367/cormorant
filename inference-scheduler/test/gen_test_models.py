#!/usr/bin/env python3
# Copyright 2025 GradeBuilder SL
# SPDX-License-Identifier: Apache-2.0

"""
gen_test_models.py — create minimal ONNX models for testing inference_scheduler.

Models produced:
  single_add.onnx           one Add node with a constant bias (one runtime input)
  relu_chain.onnx           Add followed by Relu (one runtime input)
  mixed_ops.onnx            Add -> Mul -> Relu6 (Clip 0,6) chain (one runtime input)
  two_input_add.onnx        Add with two runtime inputs: A + B -> Y
  two_input_chain.onnx      Two runtime inputs: A + B -> Relu -> Y
  two_output_tap.onnx       One input, two outputs: X+bias -> add_Y[out1] -> Relu -> relu_Y[out2]
  two_output_chain.onnx     One input, two outputs: X+b1 -> add_Y[out1] -> *scale -> Relu6 -> clip_Y[out2]
  batch_relu.onnx           2D input [4, 64]: X+bias -> Relu -> Y  (batch of 4 rows)
  matrix_ops.onnx           2D input [8, 32]: A*B -> Relu6 -> Y  (two runtime inputs, matrix shape)
  large_tensor.onnx         4D input [64, 64, 16, 16] (1M elements): X+bias -> add_Y[out1] -> Relu -> Y[out2]
  residual_add.onnx         minimal skip connection: Relu(X) -> relu_X, then X + relu_X -> Y
  residual_chain.onnx       full residual block: X+bias -> Relu -> *scale -> result, then X+result -> Y
  sub_bias.onnx             single Sub with a constant offset: X - mean -> Y
  div_scale.onnx            single Div with a constant scale: X / std -> Y
  normalize.onnx            Sub then Div: (X - mean) / std -> Y  (mean/std normalisation)
  broadcast_aligned.onnx    X[8,16] + bias[16] -> Y[8,16]  (chunk=16 elems, naturally aligned)
  broadcast_unaligned.onnx  X[4,6]  + bias[6]  -> Y[4,6]   (chunk=6 elems, needs padding to 8)
  broadcast_mixed.onnx      X[4,6]  + bias[6]  -> Relu -> Y[4,6]  (broadcast Add + plain Relu)
  broadcast_chain.onnx      X[4,6]  + bias[6]  -> Relu -> +X -> Y[4,6]  (stride propagates 2 levels)
  broadcast_chain_bias.onnx X[4,6]  + bias[6]  -> Relu -> +bias2[4,6] -> Y[4,6]  (bias2 in strided ROM)
  chunk_one.onnx            X[16,1] + bias[1]  -> Y[16,1]  (chunk=1, max gap: 7 zeros per block)
  broadcast_3d.onnx         X[2,3,4]+ bias[4]  -> Y[2,3,4] (3D input, outer_count=6)
  broadcast_tapped.onnx     X[4,6]  + bias[6]  -> add_Y[out1] -> Relu -> Y[out2] (broadcast output tapped)
  clip_opset10.onnx         Clip(min=0,max=6) via opset-10 attributes (not input tensors)
  sat_add_pos.onnx          X[1,256] + 127.5  -> Y; elements [128..255] saturate to max (+127.996)
  sat_add_neg.onnx          X[1,256] + (-63) + (-65.5) -> Y; elements [0..127] saturate to min (-128)
  sat_mul_pos.onnx          (X+1)*90 -> Y; elements [109..255] saturate to max (+127.996)
  sat_mul_neg.onnx          (X-2)*90 -> Y; elements [0..147] saturate to min (-128)
  sat_sub_pos.onnx          X[1,256] - (-127.5) -> Y; elements [128..255] saturate to max (+127.996)
  sat_sub_neg.onnx          X[1,256] - 63 - 65.5 -> Y; elements [0..128] saturate to min (-128)
  sat_div_pos.onnx          X[1,256] / (1/256) -> Y=i; elements [128..255] saturate to max (+127.996)
  sat_div_neg.onnx          X[1,256] / (-1/256) -> Y=-i; elements [128..255] saturate to min (-128)
  unsupported.onnx          Tanh node — must trigger a SchedulerError

Run from the inference-scheduler directory:
  python test/gen_test_models.py [--out-dir /path/to/models]
"""

import argparse
import os
import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as nph
from onnx import TensorProto


# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #

def _save(model: onnx.ModelProto, path: str) -> None:
    onnx.checker.check_model(model)
    onnx.save(model, path)
    print(f"  saved: {path}")


def _float32(name: str, shape: list[int]) -> onnx.TypeProto:
    return oh.make_tensor_value_info(name, TensorProto.FLOAT, shape)


def _initializer(name: str, data: np.ndarray) -> onnx.TensorProto:
    return nph.from_array(data.astype(np.float32), name=name)


# ------------------------------------------------------------------ #
# Model 1: single Add with a constant bias                           #
# ------------------------------------------------------------------ #

def make_single_add(out_dir: str) -> None:
    """input + bias  (both [1, 256])"""
    N = 256
    bias = np.ones((1, N), dtype=np.float32) * 0.5

    X    = _float32("X",    [1, N])
    Y    = _float32("Y",    [1, N])
    bias_init = _initializer("bias", bias)

    add_node = oh.make_node("Add", inputs=["X", "bias"], outputs=["Y"])

    graph = oh.make_graph(
        nodes=[add_node],
        name="single_add",
        inputs=[X],
        outputs=[Y],
        initializer=[bias_init],
    )
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
    model.ir_version = 8
    _save(model, os.path.join(out_dir, "single_add.onnx"))


# ------------------------------------------------------------------ #
# Model 2: Add -> Relu                                               #
# ------------------------------------------------------------------ #

def make_relu_chain(out_dir: str) -> None:
    """input + bias  then  Relu"""
    N = 128
    bias = np.random.randn(1, N).astype(np.float32) * 0.5

    X      = _float32("X",      [1, N])
    add_Y  = _float32("add_Y",  [1, N])
    relu_Y = _float32("relu_Y", [1, N])
    bias_init = _initializer("bias", bias)

    add_node  = oh.make_node("Add",  inputs=["X", "bias"], outputs=["add_Y"])
    relu_node = oh.make_node("Relu", inputs=["add_Y"],     outputs=["relu_Y"])

    graph = oh.make_graph(
        nodes=[add_node, relu_node],
        name="relu_chain",
        inputs=[X],
        outputs=[relu_Y],
        initializer=[bias_init],
    )
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
    model.ir_version = 8
    _save(model, os.path.join(out_dir, "relu_chain.onnx"))


# ------------------------------------------------------------------ #
# Model 3: Add -> Mul -> Clip(0,6)                                   #
# ------------------------------------------------------------------ #

def make_mixed_ops(out_dir: str) -> None:
    """
    input + bias1  →  * scale  →  Clip(0, 6)

    Clip uses input-based bounds (opset >= 11 style).
    """
    N     = 64
    bias1 = np.random.randn(1, N).astype(np.float32)
    scale = np.ones((1, N), dtype=np.float32) * 2.0
    mn    = np.array([0.0], dtype=np.float32)
    mx    = np.array([6.0], dtype=np.float32)

    X      = _float32("X",       [1, N])
    add_Y  = _float32("add_Y",   [1, N])
    mul_Y  = _float32("mul_Y",   [1, N])
    clip_Y = _float32("clip_Y",  [1, N])

    inits = [
        _initializer("bias1", bias1),
        _initializer("scale", scale),
        _initializer("clip_min", mn),
        _initializer("clip_max", mx),
    ]

    add_node  = oh.make_node("Add",  ["X",     "bias1"], ["add_Y"])
    mul_node  = oh.make_node("Mul",  ["add_Y", "scale"], ["mul_Y"])
    clip_node = oh.make_node("Clip", ["mul_Y", "clip_min", "clip_max"], ["clip_Y"])

    graph = oh.make_graph(
        nodes=[add_node, mul_node, clip_node],
        name="mixed_ops",
        inputs=[X],
        outputs=[clip_Y],
        initializer=inits,
    )
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
    model.ir_version = 8
    _save(model, os.path.join(out_dir, "mixed_ops.onnx"))


# ------------------------------------------------------------------ #
# Model 4: two runtime inputs — Add(A, B) -> Y                       #
# ------------------------------------------------------------------ #

def make_two_input_add(out_dir: str) -> None:
    """A + B  (both [1, 256], both runtime inputs — no initializers)"""
    N = 256

    A = _float32("A", [1, N])
    B = _float32("B", [1, N])
    Y = _float32("Y", [1, N])

    add_node = oh.make_node("Add", inputs=["A", "B"], outputs=["Y"])

    graph = oh.make_graph(
        nodes=[add_node],
        name="two_input_add",
        inputs=[A, B],
        outputs=[Y],
    )
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
    model.ir_version = 8
    _save(model, os.path.join(out_dir, "two_input_add.onnx"))


# ------------------------------------------------------------------ #
# Model 5: two runtime inputs — Add(A, B) -> Relu -> Y               #
# ------------------------------------------------------------------ #

def make_two_input_chain(out_dir: str) -> None:
    """A + B  then  Relu  (both inputs [1, 128], no initializers)"""
    N = 128

    A      = _float32("A",      [1, N])
    B      = _float32("B",      [1, N])
    add_Y  = _float32("add_Y",  [1, N])
    relu_Y = _float32("relu_Y", [1, N])

    add_node  = oh.make_node("Add",  inputs=["A", "B"],    outputs=["add_Y"])
    relu_node = oh.make_node("Relu", inputs=["add_Y"],     outputs=["relu_Y"])

    graph = oh.make_graph(
        nodes=[add_node, relu_node],
        name="two_input_chain",
        inputs=[A, B],
        outputs=[relu_Y],
    )
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
    model.ir_version = 8
    _save(model, os.path.join(out_dir, "two_input_chain.onnx"))


# ------------------------------------------------------------------ #
# Model 6: one input, two outputs — tap an intermediate result       #
#   X + bias -> add_Y [output 1] -> Relu -> relu_Y [output 2]        #
# ------------------------------------------------------------------ #

def make_two_output_tap(out_dir: str) -> None:
    """
    add_Y is the output of the Add node AND a graph output, so the
    caller receives both the pre-activation (add_Y) and the rectified
    (relu_Y) result in separate caller-supplied buffers.
    """
    N    = 128
    bias = np.ones((1, N), dtype=np.float32) * 0.5

    X      = _float32("X",      [1, N])
    add_Y  = _float32("add_Y",  [1, N])
    relu_Y = _float32("relu_Y", [1, N])
    bias_init = _initializer("bias", bias)

    add_node  = oh.make_node("Add",  inputs=["X", "bias"], outputs=["add_Y"])
    relu_node = oh.make_node("Relu", inputs=["add_Y"],     outputs=["relu_Y"])

    graph = oh.make_graph(
        nodes=[add_node, relu_node],
        name="two_output_tap",
        inputs=[X],
        outputs=[add_Y, relu_Y],   # intermediate add_Y is also exposed
        initializer=[bias_init],
    )
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
    model.ir_version = 8
    _save(model, os.path.join(out_dir, "two_output_tap.onnx"))


# ------------------------------------------------------------------ #
# Model 7: one input, two outputs — longer chain with tapped middle  #
#   X + b1 -> add_Y [output 1] -> * scale -> Relu6 -> clip_Y [out2] #
# ------------------------------------------------------------------ #

def make_two_output_chain(out_dir: str) -> None:
    """
    add_Y (post-Add) is tapped as output 1; the full chain result
    clip_Y (post-Relu6) is output 2.  scale is a constant weight.
    """
    N     = 64
    b1    = np.random.randn(1, N).astype(np.float32) * 0.5
    scale = np.ones((1, N), dtype=np.float32) * 2.0
    mn    = np.array([0.0], dtype=np.float32)
    mx    = np.array([6.0], dtype=np.float32)

    X      = _float32("X",      [1, N])
    add_Y  = _float32("add_Y",  [1, N])
    mul_Y  = _float32("mul_Y",  [1, N])
    clip_Y = _float32("clip_Y", [1, N])

    inits = [
        _initializer("b1",       b1),
        _initializer("scale",    scale),
        _initializer("clip_min", mn),
        _initializer("clip_max", mx),
    ]

    add_node  = oh.make_node("Add",  ["X",     "b1"],              ["add_Y"])
    mul_node  = oh.make_node("Mul",  ["add_Y", "scale"],            ["mul_Y"])
    clip_node = oh.make_node("Clip", ["mul_Y", "clip_min", "clip_max"], ["clip_Y"])

    graph = oh.make_graph(
        nodes=[add_node, mul_node, clip_node],
        name="two_output_chain",
        inputs=[X],
        outputs=[add_Y, clip_Y],   # tap after Add; final after Relu6
        initializer=inits,
    )
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
    model.ir_version = 8
    _save(model, os.path.join(out_dir, "two_output_chain.onnx"))


# ------------------------------------------------------------------ #
# Model 8: 2D input [4, 64] — batch of rows, one runtime input       #
#   X[4,64] + bias[4,64] -> add_Y -> Relu -> Y[4,64]                 #
# ------------------------------------------------------------------ #

def make_batch_relu(out_dir: str) -> None:
    """
    X has shape [4, 64] — 4 independent rows processed as a flat
    vector of 256 elements by VectorOPKernel.
    """
    B, N = 4, 64
    bias = np.ones((B, N), dtype=np.float32) * 0.25

    X     = _float32("X",     [B, N])
    add_Y = _float32("add_Y", [B, N])
    Y     = _float32("Y",     [B, N])
    bias_init = _initializer("bias", bias)

    add_node  = oh.make_node("Add",  inputs=["X", "bias"], outputs=["add_Y"])
    relu_node = oh.make_node("Relu", inputs=["add_Y"],     outputs=["Y"])

    graph = oh.make_graph(
        nodes=[add_node, relu_node],
        name="batch_relu",
        inputs=[X],
        outputs=[Y],
        initializer=[bias_init],
    )
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
    model.ir_version = 8
    _save(model, os.path.join(out_dir, "batch_relu.onnx"))


# ------------------------------------------------------------------ #
# Model 9: 2D input [8, 32] — two runtime inputs, matrix shape       #
#   A[8,32] * B[8,32] -> mul_Y -> Relu6 -> Y[8,32]                   #
# ------------------------------------------------------------------ #

def make_matrix_ops(out_dir: str) -> None:
    """
    A and B both have shape [8, 32] — a matrix of 256 elements each.
    Both are runtime inputs (no initializers).
    """
    M, N = 8, 32

    A     = _float32("A",     [M, N])
    B     = _float32("B",     [M, N])
    mul_Y = _float32("mul_Y", [M, N])
    Y     = _float32("Y",     [M, N])
    mn    = np.array([0.0], dtype=np.float32)
    mx    = np.array([6.0], dtype=np.float32)

    mul_node  = oh.make_node("Mul",  ["A", "B"],              ["mul_Y"])
    clip_node = oh.make_node("Clip", ["mul_Y", "clip_min", "clip_max"], ["Y"])

    graph = oh.make_graph(
        nodes=[mul_node, clip_node],
        name="matrix_ops",
        inputs=[A, B],
        outputs=[Y],
        initializer=[
            _initializer("clip_min", mn),
            _initializer("clip_max", mx),
        ],
    )
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
    model.ir_version = 8
    _save(model, os.path.join(out_dir, "matrix_ops.onnx"))


# ------------------------------------------------------------------ #
# Model 10: large 4-D tensor [64, 64, 16, 16] — 1 048 576 elements  #
#   X[64,64,16,16] + bias -> add_Y[out1], Relu(add_Y) -> Y[out2]    #
# ------------------------------------------------------------------ #

def make_large_tensor(out_dir: str) -> None:
    """
    Feature-map-shaped tensor typical of a mid-network conv layer:
      batch=64, channels=64, height=16, width=16  => 1 048 576 elements.

    The intermediate add_Y is exposed as a second output so the caller
    can inspect pre-activation values.  VectorOPKernel sees a flat
    vector of 1 048 576 elements for each op.
    """
    shape = [64, 64, 16, 16]
    numel = 64 * 64 * 16 * 16   # 1 048 576
    bias  = np.full(shape, -0.01, dtype=np.float32)

    X      = _float32("X",     shape)
    add_Y  = _float32("add_Y", shape)
    Y      = _float32("Y",     shape)
    bias_init = _initializer("bias", bias)

    add_node  = oh.make_node("Add",  inputs=["X", "bias"], outputs=["add_Y"])
    relu_node = oh.make_node("Relu", inputs=["add_Y"],     outputs=["Y"])

    graph = oh.make_graph(
        nodes=[add_node, relu_node],
        name="large_tensor",
        inputs=[X],
        outputs=[add_Y, Y],     # tap pre-activation as first output
        initializer=[bias_init],
    )
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
    model.ir_version = 8
    _save(model, os.path.join(out_dir, "large_tensor.onnx"))


# ------------------------------------------------------------------ #
# Model 11: minimal residual connection                              #
#   Relu(X) -> relu_X                                                #
#   X + relu_X -> Y          (skip: X bypasses the Relu branch)     #
# ------------------------------------------------------------------ #

def make_residual_add(out_dir: str) -> None:
    """
    Simplest residual block: one unary transform on the skip path,
    then the input is added back.  X is used as input to two nodes:
    the Relu and the final Add.
    """
    N = 128

    X      = _float32("X",      [1, N])
    relu_X = _float32("relu_X", [1, N])
    Y      = _float32("Y",      [1, N])

    relu_node = oh.make_node("Relu", inputs=["X"],          outputs=["relu_X"])
    add_node  = oh.make_node("Add",  inputs=["relu_X", "X"], outputs=["Y"])

    graph = oh.make_graph(
        nodes=[relu_node, add_node],
        name="residual_add",
        inputs=[X],
        outputs=[Y],
    )
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
    model.ir_version = 8
    _save(model, os.path.join(out_dir, "residual_add.onnx"))


# ------------------------------------------------------------------ #
# Model 12: full residual block                                      #
#   X + bias  -> add_X                                               #
#   Relu(add_X) -> relu_X                                            #
#   relu_X * scale -> mul_X                                          #
#   X + mul_X -> Y           (skip: X bypasses the whole block)     #
# ------------------------------------------------------------------ #

def make_residual_chain(out_dir: str) -> None:
    """
    Residual block with bias shift, activation, and scaling on the
    transform branch; the original X is added back at the end.
    X is used as input to two nodes: the first Add and the last Add.
    """
    N     = 64
    bias  = np.random.randn(1, N).astype(np.float32) * 0.1
    scale = np.ones((1, N), dtype=np.float32) * 0.5

    X     = _float32("X",     [1, N])
    add_X = _float32("add_X", [1, N])
    relu_X= _float32("relu_X",[1, N])
    mul_X = _float32("mul_X", [1, N])
    Y     = _float32("Y",     [1, N])

    inits = [
        _initializer("bias",  bias),
        _initializer("scale", scale),
    ]

    nodes = [
        oh.make_node("Add",  ["X",      "bias"],  ["add_X"]),
        oh.make_node("Relu", ["add_X"],            ["relu_X"]),
        oh.make_node("Mul",  ["relu_X", "scale"],  ["mul_X"]),
        oh.make_node("Add",  ["X",      "mul_X"],  ["Y"]),
    ]

    graph = oh.make_graph(
        nodes=nodes,
        name="residual_chain",
        inputs=[X],
        outputs=[Y],
        initializer=inits,
    )
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
    model.ir_version = 8
    _save(model, os.path.join(out_dir, "residual_chain.onnx"))


# ------------------------------------------------------------------ #
# Model 13: single Sub — X - mean -> Y                               #
# ------------------------------------------------------------------ #

def make_sub_bias(out_dir: str) -> None:
    """Element-wise subtraction of a constant mean vector."""
    N    = 256
    mean = np.random.randn(1, N).astype(np.float32) * 0.1

    X = _float32("X", [1, N])
    Y = _float32("Y", [1, N])

    graph = oh.make_graph(
        nodes=[oh.make_node("Sub", ["X", "mean"], ["Y"])],
        name="sub_bias",
        inputs=[X],
        outputs=[Y],
        initializer=[_initializer("mean", mean)],
    )
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
    model.ir_version = 8
    _save(model, os.path.join(out_dir, "sub_bias.onnx"))


# ------------------------------------------------------------------ #
# Model 14: single Div — X / std -> Y                                #
# ------------------------------------------------------------------ #

def make_div_scale(out_dir: str) -> None:
    """Element-wise division by a constant standard-deviation vector."""
    N   = 256
    std = np.abs(np.random.randn(1, N).astype(np.float32)) + 0.1  # keep > 0

    X = _float32("X", [1, N])
    Y = _float32("Y", [1, N])

    graph = oh.make_graph(
        nodes=[oh.make_node("Div", ["X", "std"], ["Y"])],
        name="div_scale",
        inputs=[X],
        outputs=[Y],
        initializer=[_initializer("std", std)],
    )
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
    model.ir_version = 8
    _save(model, os.path.join(out_dir, "div_scale.onnx"))


# ------------------------------------------------------------------ #
# Model 15: Sub then Div — (X - mean) / std -> Y                    #
#   Classic mean/std normalisation as a two-op chain.                #
# ------------------------------------------------------------------ #

def make_normalize(out_dir: str) -> None:
    """Mean-centre then scale: two VectorOPKernel calls."""
    N    = 256
    mean = np.random.randn(1, N).astype(np.float32) * 0.1
    std  = np.abs(np.random.randn(1, N).astype(np.float32)) + 0.1

    X     = _float32("X",     [1, N])
    sub_Y = _float32("sub_Y", [1, N])
    Y     = _float32("Y",     [1, N])

    graph = oh.make_graph(
        nodes=[
            oh.make_node("Sub", ["X",     "mean"], ["sub_Y"]),
            oh.make_node("Div", ["sub_Y", "std"],  ["Y"]),
        ],
        name="normalize",
        inputs=[X],
        outputs=[Y],
        initializer=[
            _initializer("mean", mean),
            _initializer("std",  std),
        ],
    )
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
    model.ir_version = 8
    _save(model, os.path.join(out_dir, "normalize.onnx"))


# ------------------------------------------------------------------ #
# Model 16: broadcasting — chunk naturally 16-byte aligned           #
#   X[8, 16] + bias[16] -> Y[8, 16]                                  #
#   chunk = 16 elements × 2 bytes = 32 bytes (multiple of 16 → no gap)
# ------------------------------------------------------------------ #

def make_broadcast_aligned(out_dir: str) -> None:
    """
    Bias [16] broadcasts over the leading dimension of X [8, 16].
    chunk_size = 16 elements; with 2-byte elements chunk_bytes = 32,
    which is a multiple of INFERENCE_ALIGN_BYTES (16) so aligned_chunk == chunk.
    outer_count = 8.
    """
    ROWS, COLS = 8, 16
    bias = np.ones(COLS, dtype=np.float32) * 0.5

    X = _float32("X", [ROWS, COLS])
    Y = _float32("Y", [ROWS, COLS])

    graph = oh.make_graph(
        nodes=[oh.make_node("Add", ["X", "bias"], ["Y"])],
        name="broadcast_aligned",
        inputs=[X],
        outputs=[Y],
        initializer=[_initializer("bias", bias)],
    )
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
    model.ir_version = 8
    _save(model, os.path.join(out_dir, "broadcast_aligned.onnx"))


# ------------------------------------------------------------------ #
# Model 17: broadcasting — chunk NOT 16-byte aligned (needs padding) #
#   X[4, 6] + bias[6] -> Y[4, 6]                                     #
#   chunk = 6 elements × 2 bytes = 12 bytes → padded to 8 elems (16 B)
# ------------------------------------------------------------------ #

def make_broadcast_unaligned(out_dir: str) -> None:
    """
    Bias [6] broadcasts over the leading dimension of X [4, 6].
    chunk_size = 6 elements; with 2-byte elements chunk_bytes = 12,
    NOT a multiple of INFERENCE_ALIGN_BYTES (16), so aligned_chunk = 8
    (gap of 2 unused elements per block).
    outer_count = 4.
    """
    ROWS, COLS = 4, 6
    bias = np.ones(COLS, dtype=np.float32) * 0.25

    X = _float32("X", [ROWS, COLS])
    Y = _float32("Y", [ROWS, COLS])

    graph = oh.make_graph(
        nodes=[oh.make_node("Add", ["X", "bias"], ["Y"])],
        name="broadcast_unaligned",
        inputs=[X],
        outputs=[Y],
        initializer=[_initializer("bias", bias)],
    )
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
    model.ir_version = 8
    _save(model, os.path.join(out_dir, "broadcast_unaligned.onnx"))


# ------------------------------------------------------------------ #
# Model 18: mixed broadcast + non-broadcast nodes                    #
#   X[4,6] + bias[6] -> add_Y[4,6] (broadcast, chunk=6, padded to 8)#
#   Relu(add_Y) -> Y[4,6]          (non-broadcast, plain run_op)     #
# ------------------------------------------------------------------ #

def make_broadcast_mixed(out_dir: str) -> None:
    """
    Bias [6] broadcasts over the leading dimension of X [4, 6] (same
    unaligned chunk as broadcast_unaligned), then a Relu is applied.

    Node 0 (Add):  outer_count=4, chunk=6, aligned_chunk=8 — emits run_op_at loop.
    Node 1 (Relu): outer_count=1, chunk=24              — emits plain run_op call.

    Both run_op() and run_op_at() must be emitted.
    add_Y (intermediate) is allocated with the padded size (4*8=32 elements).
    Y (output of Relu) uses plain numel (24 elements) — not padded.
    """
    ROWS, COLS = 4, 6
    bias = np.ones(COLS, dtype=np.float32) * 0.25

    X     = _float32("X",     [ROWS, COLS])
    add_Y = _float32("add_Y", [ROWS, COLS])
    Y     = _float32("Y",     [ROWS, COLS])

    graph = oh.make_graph(
        nodes=[
            oh.make_node("Add",  ["X", "bias"], ["add_Y"]),
            oh.make_node("Relu", ["add_Y"],     ["Y"]),
        ],
        name="broadcast_mixed",
        inputs=[X],
        outputs=[Y],
        initializer=[_initializer("bias", bias)],
    )
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
    model.ir_version = 8
    _save(model, os.path.join(out_dir, "broadcast_mixed.onnx"))


# ------------------------------------------------------------------ #
# Model 19: broadcast chain — stride propagates through two levels   #
#   X[4,6] + bias[6] -> add_Y[4,6]  (broadcast, chunk=6, padded=8)  #
#   Relu(add_Y)       -> relu_Y[4,6] (non-broadcast, level-1 prop)   #
#   relu_Y + X        -> Y[4,6]      (non-broadcast Add, level-2)    #
# ------------------------------------------------------------------ #

def make_broadcast_chain(out_dir: str) -> None:
    """
    Extends broadcast_mixed with a second non-broadcast layer to verify
    that stride propagation works through multiple sequential layers.

    Node 0 (Add):   broadcast, outer_count=4, chunk=6, aligned_chunk=8
                    → emits run_op_at loop; add_Y alloc = 4*8 = 32
    Node 1 (Relu):  non-broadcast, inherits add_Y's padded alloc (32)
                    → emits run_op(add_Y, NULL, relu_Y, 32u, …)
    Node 2 (Add):   non-broadcast, both inputs (relu_Y, X) have alloc=32
                    → emits run_op(relu_Y, X, Y, 32u, …)

    All four buffers (X, add_Y, relu_Y, Y) must have alloc=32.
    INFERENCE_Y_SIZE = 4u * INFERENCE_ADD_Y_CHUNK_STRIDE.
    """
    ROWS, COLS = 4, 6
    bias = np.ones(COLS, dtype=np.float32) * 0.25

    X      = _float32("X",      [ROWS, COLS])
    add_Y  = _float32("add_Y",  [ROWS, COLS])
    relu_Y = _float32("relu_Y", [ROWS, COLS])
    Y      = _float32("Y",      [ROWS, COLS])

    graph = oh.make_graph(
        nodes=[
            oh.make_node("Add",  ["X", "bias"],     ["add_Y"]),
            oh.make_node("Relu", ["add_Y"],          ["relu_Y"]),
            oh.make_node("Add",  ["relu_Y", "X"],   ["Y"]),
        ],
        name="broadcast_chain",
        inputs=[X],
        outputs=[Y],
        initializer=[_initializer("bias", bias)],
    )
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
    model.ir_version = 8
    _save(model, os.path.join(out_dir, "broadcast_chain.onnx"))


# ------------------------------------------------------------------ #
# Model 20: two broadcast adds separated by Relu                     #
#   X[4,6] + bias[6]  -> add_Y[4,6]  (broadcast, chunk=6, padded=8) #
#   Relu(add_Y)        -> relu_Y[4,6] (non-broadcast, bridge)        #
#   relu_Y + bias2[6]  -> Y[4,6]      (broadcast, own chunk macros)  #
# ------------------------------------------------------------------ #

def make_broadcast_chain_bias(out_dir: str) -> None:
    """
    Same broadcast-Add → Relu chain as broadcast_chain, but the last layer
    is a non-broadcast Add with a full-shape bias weight (bias2[4,6], 24 elements)
    instead of a residual connection.

    Node 0 (Add):   broadcast, outer_count=4, chunk=6, aligned_chunk=8
                    → run_op_at loop; add_Y alloc = 4*8 = 32
    Node 1 (Relu):  non-broadcast — run_op(add_Y, NULL, relu_Y, 32u, …)
    Node 2 (Add):   non-broadcast — run_op(relu_Y, bias2, Y, 32u, …)

    Because relu_Y has alloc=32 (strided layout with 2-element gaps per row),
    bias2 must be stored in the same strided layout so run_op() can add
    element-wise at every position.  The code generator emits bias2's ROM
    array as 4 blocks of 8 uint16 values: 6 data elements followed by 2
    zero-padded gap elements per block (32 elements total).

    All four data buffers (X, add_Y, relu_Y, Y) and bias2 have alloc=32.
    """
    ROWS, COLS = 4, 6
    bias  = np.ones(COLS,          dtype=np.float32) * 0.25
    bias2 = np.ones((ROWS, COLS),  dtype=np.float32) * 0.5

    X      = _float32("X",      [ROWS, COLS])
    add_Y  = _float32("add_Y",  [ROWS, COLS])
    relu_Y = _float32("relu_Y", [ROWS, COLS])
    Y      = _float32("Y",      [ROWS, COLS])

    graph = oh.make_graph(
        nodes=[
            oh.make_node("Add",  ["X", "bias"],      ["add_Y"]),
            oh.make_node("Relu", ["add_Y"],           ["relu_Y"]),
            oh.make_node("Add",  ["relu_Y", "bias2"], ["Y"]),
        ],
        name="broadcast_chain_bias",
        inputs=[X],
        outputs=[Y],
        initializer=[
            _initializer("bias",  bias),
            _initializer("bias2", bias2),
        ],
    )
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
    model.ir_version = 8
    _save(model, os.path.join(out_dir, "broadcast_chain_bias.onnx"))


# ------------------------------------------------------------------ #
# Model 21: maximum alignment gap — chunk=1, gap=7 elements per row  #
#   X[16,1] + bias[1] -> Y[16,1]                                     #
# ------------------------------------------------------------------ #

def make_chunk_one(out_dir: str) -> None:
    """
    Bias [1] broadcasts over the leading dimension of X [16, 1].
    chunk_size = 1 element; aligned_chunk = 8 (gap = 7 per block).
    This is the worst-case alignment ratio: 7 out of every 8 slots
    in each buffer block are gap padding.

    outer_count  = 16
    chunk_size   = 1
    aligned_chunk= 8  (= _ALIGN_ELEMS)
    alloc(X)     = 16 * 8 = 128 elements
    alloc(bias)  = 8 elements (one aligned block; only position 0 is data)
    alloc(Y)     = 128 elements
    """
    ROWS, COLS = 16, 1
    bias = np.array([0.25], dtype=np.float32)

    X = _float32("X", [ROWS, COLS])
    Y = _float32("Y", [ROWS, COLS])

    graph = oh.make_graph(
        nodes=[oh.make_node("Add", ["X", "bias"], ["Y"])],
        name="chunk_one",
        inputs=[X],
        outputs=[Y],
        initializer=[_initializer("bias", bias)],
    )
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
    model.ir_version = 8
    _save(model, os.path.join(out_dir, "chunk_one.onnx"))


# ------------------------------------------------------------------ #
# Model 22: 3-D input shape                                          #
#   X[2,3,4] + bias[4] -> Y[2,3,4]                                  #
# ------------------------------------------------------------------ #

def make_broadcast_3d(out_dir: str) -> None:
    """
    Bias [4] broadcasts over the first two dimensions of X [2, 3, 4].
    Right-aligns to [1, 1, 4] → dims 0 and 1 are broadcast dims (contiguous
    leading block), dim 2 matches.

    outer_count   = 2 * 3 = 6
    chunk_size    = 4
    aligned_chunk = 8  (4 → next multiple of _ALIGN_ELEMS)
    alloc(X)      = 6 * 8 = 48
    alloc(bias)   = 8 (one aligned block)
    alloc(Y)      = 48
    """
    bias = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)

    X = _float32("X", [2, 3, 4])
    Y = _float32("Y", [2, 3, 4])

    graph = oh.make_graph(
        nodes=[oh.make_node("Add", ["X", "bias"], ["Y"])],
        name="broadcast_3d",
        inputs=[X],
        outputs=[Y],
        initializer=[_initializer("bias", bias)],
    )
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
    model.ir_version = 8
    _save(model, os.path.join(out_dir, "broadcast_3d.onnx"))


# ------------------------------------------------------------------ #
# Model 23: broadcast output tapped as a graph output                #
#   X[4,6] + bias[6] -> add_Y[4,6] [output 1] -> Relu -> Y[out 2]  #
# ------------------------------------------------------------------ #

def make_broadcast_tapped(out_dir: str) -> None:
    """
    The broadcast Add node's output (add_Y) is exposed as a graph output
    so the caller can read the pre-activation values.  Relu then produces
    the second graph output Y.

    Unlike broadcast_mixed (where add_Y is an internal intermediate buffer),
    here add_Y is caller-supplied.  Unique properties:
      - add_Y is an output_tensor, NOT an intermediate_tensor
      - inference_run signature includes add_Y and Y (two output parameters)
      - INFERENCE_ADD_Y_SIZE and INFERENCE_Y_SIZE both use strided SIZE macros
      - test_inference.c prints BOTH add_Y and Y with chunk loops

    Node 0 (Add):   broadcast, outer_count=4, chunk=6, aligned_chunk=8
    Node 1 (Relu):  non-broadcast (run_op with padded size 32u)
    """
    ROWS, COLS = 4, 6
    bias = np.ones(COLS, dtype=np.float32) * 0.25

    X     = _float32("X",     [ROWS, COLS])
    add_Y = _float32("add_Y", [ROWS, COLS])
    Y     = _float32("Y",     [ROWS, COLS])

    graph = oh.make_graph(
        nodes=[
            oh.make_node("Add",  ["X", "bias"], ["add_Y"]),
            oh.make_node("Relu", ["add_Y"],      ["Y"]),
        ],
        name="broadcast_tapped",
        inputs=[X],
        outputs=[add_Y, Y],      # add_Y exposed — no intermediate buffer
        initializer=[_initializer("bias", bias)],
    )
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
    model.ir_version = 8
    _save(model, os.path.join(out_dir, "broadcast_tapped.onnx"))


# ------------------------------------------------------------------ #
# Model 24: Clip(0, 6) with opset-10 attribute-style bounds          #
#   X[1,64] -> Clip(min=0, max=6) -> Y[1,64]                        #
# ------------------------------------------------------------------ #

def make_clip_opset10(out_dir: str) -> None:
    """
    Clip node with min/max stored as node *attributes* (opset 10 style),
    not as input tensors (opset >= 11 style).

    This exercises the attribute-reading branch in _get_clip_bounds():
        attrs = {a.name: a for a in node.attribute}
        if "min" in attrs or "max" in attrs:
            return attrs["min"].f, attrs["max"].f   # ← this path

    Expected: op_code = OP_RELU6, arity = 1, VECTOROP_RELU6 emitted.
    """
    N = 64

    X = _float32("X", [1, N])
    Y = _float32("Y", [1, N])

    clip_node = oh.make_node(
        "Clip",
        inputs=["X"],
        outputs=["Y"],
        min=0.0,
        max=6.0,
    )
    graph = oh.make_graph(
        nodes=[clip_node],
        name="clip_opset10",
        inputs=[X],
        outputs=[Y],
    )
    # opset 10: Clip uses float attributes for min/max
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 10)])
    model.ir_version = 8
    _save(model, os.path.join(out_dir, "clip_opset10.onnx"))


# ------------------------------------------------------------------ #
# Saturation models                                                  #
#                                                                    #
# ap_fixed<16,8> range: [-128.0,  127.99609375]  (= [-32768, 32767] #
#                        scaled by 1/256)                            #
#                                                                    #
# The ramp input fills buffer position p with (Data_t)(p & 0xFFFF), #
# so element i of a [1,N] tensor has value i/256.                   #
# All four models use N=256, giving input range [0/256 .. 255/256]. #
# ------------------------------------------------------------------ #

# Model 26: Add positive saturation
# X[1,256] + 127.5  -> Y
#   i=0..127:  i/256 + 127.5 in [127.5, 127.996]  — no saturation
#   i=128..255: i/256 + 127.5 >= 128.0             — saturate to 127.996
def make_sat_add_pos(out_dir: str) -> None:
    """Single Add with a large positive bias; upper half saturates."""
    N    = 256
    bias = np.full((1, N), 127.5, dtype=np.float32)

    X = _float32("X", [1, N])
    Y = _float32("Y", [1, N])

    graph = oh.make_graph(
        nodes=[oh.make_node("Add", ["X", "bias"], ["Y"])],
        name="sat_add_pos",
        inputs=[X],
        outputs=[Y],
        initializer=[_initializer("bias", bias)],
    )
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
    model.ir_version = 8
    _save(model, os.path.join(out_dir, "sat_add_pos.onnx"))


# Model 27: Add negative saturation (two-stage chain)
# X[1,256] + (-63) -> Z,  Z + (-65.5) -> Y
#   Effective: Y[i] = i/256 - 128.5
#   i=0..127:   Y[i] in [-128.5, -128.004]  — saturate to -128.0
#   i=128..255: Y[i] in [-128.0, -127.504]  — no saturation
def make_sat_add_neg(out_dir: str) -> None:
    """Two-stage Add chain with large negative biases; lower half saturates."""
    N     = 256
    bias1 = np.full((1, N), -63.0, dtype=np.float32)
    bias2 = np.full((1, N), -65.5, dtype=np.float32)

    X = _float32("X", [1, N])
    Z = _float32("Z", [1, N])
    Y = _float32("Y", [1, N])

    graph = oh.make_graph(
        nodes=[
            oh.make_node("Add", ["X",  "bias1"], ["Z"]),
            oh.make_node("Add", ["Z",  "bias2"], ["Y"]),
        ],
        name="sat_add_neg",
        inputs=[X],
        outputs=[Y],
        initializer=[
            _initializer("bias1", bias1),
            _initializer("bias2", bias2),
        ],
    )
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
    model.ir_version = 8
    _save(model, os.path.join(out_dir, "sat_add_neg.onnx"))


# Model 28: Mul positive saturation
# X[1,256] + 1.0 -> Z,  Z * 90 -> Y
#   Z[i] = i/256 + 1,  range [1.0, 1.996]
#   i=0..108:   Z[i] * 90 <= 127.969  — no saturation
#   i=109..255: Z[i] * 90 >= 128.320  — saturate to 127.996
def make_sat_mul_pos(out_dir: str) -> None:
    """Add offset then Mul by large scale; upper portion saturates."""
    N      = 256
    offset = np.full((1, N), 1.0,  dtype=np.float32)
    scale  = np.full((1, N), 90.0, dtype=np.float32)

    X = _float32("X", [1, N])
    Z = _float32("Z", [1, N])
    Y = _float32("Y", [1, N])

    graph = oh.make_graph(
        nodes=[
            oh.make_node("Add", ["X", "offset"], ["Z"]),
            oh.make_node("Mul", ["Z", "scale"],  ["Y"]),
        ],
        name="sat_mul_pos",
        inputs=[X],
        outputs=[Y],
        initializer=[
            _initializer("offset", offset),
            _initializer("scale",  scale),
        ],
    )
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
    model.ir_version = 8
    _save(model, os.path.join(out_dir, "sat_mul_pos.onnx"))


# Model 29: Mul negative saturation
# X[1,256] - 2.0 -> Z,  Z * 90 -> Y
#   Z[i] = i/256 - 2,  range [-2.0, -1.004]
#   i=0..147:   Z[i] * 90 <= -128.320  — saturate to -128.0
#   i=148..255: Z[i] * 90 >= -127.969  — no saturation
def make_sat_mul_neg(out_dir: str) -> None:
    """Sub mean then Mul by large scale; lower portion saturates."""
    N     = 256
    mean  = np.full((1, N), 2.0,  dtype=np.float32)
    scale = np.full((1, N), 90.0, dtype=np.float32)

    X = _float32("X", [1, N])
    Z = _float32("Z", [1, N])
    Y = _float32("Y", [1, N])

    graph = oh.make_graph(
        nodes=[
            oh.make_node("Sub", ["X", "mean"],  ["Z"]),
            oh.make_node("Mul", ["Z", "scale"], ["Y"]),
        ],
        name="sat_mul_neg",
        inputs=[X],
        outputs=[Y],
        initializer=[
            _initializer("mean",  mean),
            _initializer("scale", scale),
        ],
    )
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
    model.ir_version = 8
    _save(model, os.path.join(out_dir, "sat_mul_neg.onnx"))


# Model 30: Sub positive saturation
# X[1,256] - (-127.5) -> Y    (equivalent to X + 127.5)
#   Y[i] = i/256 + 127.5
#   i=0..126:   Y[i] in [127.5, 127.992]  — no saturation
#   i=127:      Y[127] = 127.996 = AP_MAX  — exact, no overflow
#   i=128..255: Y[i] >= 128.0             — saturate to AP_MAX
def make_sat_sub_pos(out_dir: str) -> None:
    """Single Sub with a large negative bias; upper portion saturates to max."""
    N    = 256
    bias = np.full((1, N), -127.5, dtype=np.float32)

    X = _float32("X", [1, N])
    Y = _float32("Y", [1, N])

    graph = oh.make_graph(
        nodes=[oh.make_node("Sub", ["X", "bias"], ["Y"])],
        name="sat_sub_pos",
        inputs=[X],
        outputs=[Y],
        initializer=[_initializer("bias", bias)],
    )
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
    model.ir_version = 8
    _save(model, os.path.join(out_dir, "sat_sub_pos.onnx"))


# Model 31: Sub negative saturation (two-stage chain)
# X[1,256] - 63.0 -> Z,  Z - 65.5 -> Y
#   Effective: Y[i] = i/256 - 128.5
#   i=0..127:   Y[i] in [-128.5, -128.004]  — saturate to AP_MIN
#   i=128:      Y[128] = -128.0 = AP_MIN     — exact, representable
#   i=129..255: Y[i] in [-127.996, -127.504] — no saturation
def make_sat_sub_neg(out_dir: str) -> None:
    """Two-stage Sub chain with large positive biases; lower portion saturates to min."""
    N     = 256
    bias1 = np.full((1, N), 63.0, dtype=np.float32)
    bias2 = np.full((1, N), 65.5, dtype=np.float32)

    X = _float32("X", [1, N])
    Z = _float32("Z", [1, N])
    Y = _float32("Y", [1, N])

    graph = oh.make_graph(
        nodes=[
            oh.make_node("Sub", ["X", "bias1"], ["Z"]),
            oh.make_node("Sub", ["Z", "bias2"], ["Y"]),
        ],
        name="sat_sub_neg",
        inputs=[X],
        outputs=[Y],
        initializer=[
            _initializer("bias1", bias1),
            _initializer("bias2", bias2),
        ],
    )
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
    model.ir_version = 8
    _save(model, os.path.join(out_dir, "sat_sub_neg.onnx"))


# Model 32: Div positive saturation
# X[1,256] / (1/256) -> Y
#   Y[i] = i/256 / (1/256) = float(i)
#   i=0..127:   Y[i] in [0, 127]  — no saturation (127 < AP_MAX=127.996)
#   i=128..255: Y[i] >= 128       — saturate to AP_MAX
def make_sat_div_pos(out_dir: str) -> None:
    """Single Div by smallest positive unit; upper half saturates to max."""
    N       = 256
    divisor = np.full((1, N), 1.0 / 256, dtype=np.float32)   # 1 LSB = 0.00390625

    X = _float32("X", [1, N])
    Y = _float32("Y", [1, N])

    graph = oh.make_graph(
        nodes=[oh.make_node("Div", ["X", "divisor"], ["Y"])],
        name="sat_div_pos",
        inputs=[X],
        outputs=[Y],
        initializer=[_initializer("divisor", divisor)],
    )
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
    model.ir_version = 8
    _save(model, os.path.join(out_dir, "sat_div_pos.onnx"))


# Model 33: Div negative saturation
# X[1,256] / (-1/256) -> Y
#   Y[i] = i/256 / (-1/256) = float(-i)
#   i=0..127:   Y[i] in [0, -127]    — no saturation (-127 > AP_MIN=-128)
#   i=128:      Y[128] = -128 = AP_MIN — exact, representable
#   i=129..255: Y[i] <= -129          — saturate to AP_MIN
def make_sat_div_neg(out_dir: str) -> None:
    """Single Div by smallest negative unit; lower half saturates to min."""
    N       = 256
    divisor = np.full((1, N), -1.0 / 256, dtype=np.float32)  # -1 LSB = -0.00390625

    X = _float32("X", [1, N])
    Y = _float32("Y", [1, N])

    graph = oh.make_graph(
        nodes=[oh.make_node("Div", ["X", "divisor"], ["Y"])],
        name="sat_div_neg",
        inputs=[X],
        outputs=[Y],
        initializer=[_initializer("divisor", divisor)],
    )
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
    model.ir_version = 8
    _save(model, os.path.join(out_dir, "sat_div_neg.onnx"))


# ------------------------------------------------------------------ #
# Model 34: unsupported op (Tanh)                                    #
# ------------------------------------------------------------------ #

def make_unsupported(out_dir: str) -> None:
    """A single Tanh node — must cause SchedulerError."""
    X = _float32("X", [1, 8])
    Y = _float32("Y", [1, 8])

    tanh_node = oh.make_node("Tanh", inputs=["X"], outputs=["Y"])
    graph = oh.make_graph(
        nodes=[tanh_node],
        name="unsupported",
        inputs=[X],
        outputs=[Y],
    )
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
    model.ir_version = 8
    _save(model, os.path.join(out_dir, "unsupported.onnx"))


# ------------------------------------------------------------------ #
# Entry point                                                         #
# ------------------------------------------------------------------ #

def main():
    p = argparse.ArgumentParser(description="Generate test ONNX models")
    p.add_argument(
        "--out-dir",
        default=os.path.join(os.path.dirname(__file__), "models"),
        help="Directory to write .onnx files (created if absent)",
    )
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Generating test ONNX models …")
    make_single_add(args.out_dir)
    make_relu_chain(args.out_dir)
    make_mixed_ops(args.out_dir)
    make_two_input_add(args.out_dir)
    make_two_input_chain(args.out_dir)
    make_two_output_tap(args.out_dir)
    make_two_output_chain(args.out_dir)
    make_batch_relu(args.out_dir)
    make_matrix_ops(args.out_dir)
    make_large_tensor(args.out_dir)
    make_residual_add(args.out_dir)
    make_residual_chain(args.out_dir)
    make_sub_bias(args.out_dir)
    make_div_scale(args.out_dir)
    make_normalize(args.out_dir)
    make_broadcast_aligned(args.out_dir)
    make_broadcast_unaligned(args.out_dir)
    make_broadcast_mixed(args.out_dir)
    make_broadcast_chain(args.out_dir)
    make_broadcast_chain_bias(args.out_dir)
    make_chunk_one(args.out_dir)
    make_broadcast_3d(args.out_dir)
    make_broadcast_tapped(args.out_dir)
    make_clip_opset10(args.out_dir)
    make_sat_add_pos(args.out_dir)
    make_sat_add_neg(args.out_dir)
    make_sat_mul_pos(args.out_dir)
    make_sat_mul_neg(args.out_dir)
    make_sat_sub_pos(args.out_dir)
    make_sat_sub_neg(args.out_dir)
    make_sat_div_pos(args.out_dir)
    make_sat_div_neg(args.out_dir)
    make_unsupported(args.out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
