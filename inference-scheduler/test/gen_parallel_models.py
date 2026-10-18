# Copyright 2025 GradeBuilder SL
# SPDX-License-Identifier: Apache-2.0

"""Generate ONNX test models that exercise parallel-kernel scheduling.

Each fixture produces a graph whose dependency DAG has at least two
concurrency-independent nodes that the parallel scheduler should be able
to overlap on different hardware kernel lanes (VectorOPKernel /
MatmulKernel / ConvKernel / PoolKernel).

Shape choice keeps every join's operands shape-equal so the joining
``Add`` does not need broadcasting:

  * Conv:    1×1 same-channels, stride=1, pad=0   → preserves NCHW shape
  * MaxPool: kernel=3, stride=1, pads=[1,1,1,1]   → preserves NCHW shape
  * MatMul:  square weight [K, K]                 → preserves [N, K]
"""

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


def _vi(name: str, shape):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)


def _conv1x1_init(name: str, c: int, seed: int) -> onnx.TensorProto:
    """Identity-ish 1×1 conv weight [c, c, 1, 1]."""
    rng = np.random.default_rng(seed)
    w = (rng.standard_normal((c, c, 1, 1)) * 0.25).astype(np.float32)
    return numpy_helper.from_array(w, name=name)


def _conv3x3_same_init(name: str, c: int, seed: int) -> onnx.TensorProto:
    """3×3 conv weight [c, c, 3, 3] — needs pad=1 to preserve shape."""
    rng = np.random.default_rng(seed)
    w = (rng.standard_normal((c, c, 3, 3)) * 0.10).astype(np.float32)
    return numpy_helper.from_array(w, name=name)


# ------------------------------------------------------------------ #
# Two parallel branches, single kernel type                           #
# ------------------------------------------------------------------ #

def gen_parallel_relu_mul_add() -> None:
    """X → Relu → a, Mul(X,X) → b, Add(a,b) → Y.

    Pure VectorOP; serializes on a single kernel lane but the DAG has
    two independent roots — useful as a sanity baseline for the
    scheduler's "must serialize on same kernel" path.
    """
    n = 16
    nodes = [
        helper.make_node("Relu", ["X"],      ["a"], name="relu"),
        helper.make_node("Mul",  ["X", "X"], ["b"], name="mul"),
        helper.make_node("Add",  ["a", "b"], ["Y"], name="add"),
    ]
    graph = helper.make_graph(
        nodes, "parallel_relu_mul_add",
        inputs=[_vi("X", [n])], outputs=[_vi("Y", [n])],
        value_info=[_vi("a", [n]), _vi("b", [n])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    _save(model, "parallel_relu_mul_add.onnx")


# ------------------------------------------------------------------ #
# Two parallel branches, different kernel types                       #
# ------------------------------------------------------------------ #

def gen_parallel_conv_pool_join(c=4, h=8, w=8) -> None:
    """X → Conv(1×1) → cv, X → MaxPool(3,p=1) → pl, Add(cv,pl) → Y.

    Conv and Pool live on different hardware lanes; the parallel
    scheduler should run them concurrently.
    """
    inits = [_conv1x1_init("Wc", c, seed=1)]
    nodes = [
        helper.make_node("Conv", ["X", "Wc"], ["cv"], name="conv",
                         kernel_shape=[1, 1], pads=[0, 0, 0, 0],
                         strides=[1, 1]),
        helper.make_node("MaxPool", ["X"], ["pl"], name="pool",
                         kernel_shape=[3, 3], pads=[1, 1, 1, 1],
                         strides=[1, 1]),
        helper.make_node("Add", ["cv", "pl"], ["Y"], name="add"),
    ]
    graph = helper.make_graph(
        nodes, "parallel_conv_pool_join",
        inputs=[_vi("X", [1, c, h, w])],
        outputs=[_vi("Y", [1, c, h, w])],
        value_info=[_vi("cv", [1, c, h, w]), _vi("pl", [1, c, h, w])],
        initializer=inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    _save(model, "parallel_conv_pool_join.onnx")


def gen_parallel_matmul_relu_join(n=4, k=8) -> None:
    """X → MatMul(X,W[k,k]) → mm, Relu(X) → rl, Add(mm,rl) → Y.

    MatmulKernel and VectorOPKernel run in parallel.  Square weight
    keeps shapes equal so the Add needs no broadcasting.
    """
    rng = np.random.default_rng(2)
    w = (rng.standard_normal((k, k)) * 0.25).astype(np.float32)
    inits = [numpy_helper.from_array(w, name="W")]
    nodes = [
        helper.make_node("MatMul", ["X", "W"], ["mm"], name="mm"),
        helper.make_node("Relu",   ["X"],      ["rl"], name="rl"),
        helper.make_node("Add",    ["mm", "rl"], ["Y"], name="add"),
    ]
    graph = helper.make_graph(
        nodes, "parallel_matmul_relu_join",
        inputs=[_vi("X", [n, k])],
        outputs=[_vi("Y", [n, k])],
        value_info=[_vi("mm", [n, k]), _vi("rl", [n, k])],
        initializer=inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    _save(model, "parallel_matmul_relu_join.onnx")


# ------------------------------------------------------------------ #
# Three parallel branches, three different kernel types               #
# ------------------------------------------------------------------ #

def gen_parallel_three_kernel_branches(c=4, h=8, w=8) -> None:
    """X → Conv → c0, X → MaxPool → c1, X → Relu → c2,
       Add(c0, c1) → cp, Add(cp, c2) → Y.

    Three branches, three different kernels.  Two chained joins;
    inception-style fan-out to a tree of Adds (no Concat — not supported).
    """
    inits = [_conv1x1_init("Wc", c, seed=3)]
    nodes = [
        helper.make_node("Conv", ["X", "Wc"], ["c0"], name="conv",
                         kernel_shape=[1, 1], pads=[0, 0, 0, 0],
                         strides=[1, 1]),
        helper.make_node("MaxPool", ["X"], ["c1"], name="pool",
                         kernel_shape=[3, 3], pads=[1, 1, 1, 1],
                         strides=[1, 1]),
        helper.make_node("Relu", ["X"], ["c2"], name="relu"),
        helper.make_node("Add",  ["c0", "c1"], ["cp"], name="join1"),
        helper.make_node("Add",  ["cp", "c2"], ["Y"],  name="join2"),
    ]
    graph = helper.make_graph(
        nodes, "parallel_three_kernel_branches",
        inputs=[_vi("X", [1, c, h, w])],
        outputs=[_vi("Y", [1, c, h, w])],
        value_info=[
            _vi("c0", [1, c, h, w]),
            _vi("c1", [1, c, h, w]),
            _vi("c2", [1, c, h, w]),
            _vi("cp", [1, c, h, w]),
        ],
        initializer=inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    _save(model, "parallel_three_kernel_branches.onnx")


# ------------------------------------------------------------------ #
# Two longer chains running in parallel                                #
# ------------------------------------------------------------------ #

def gen_parallel_two_chains(c=4, h=8, w=8) -> None:
    """Two chains rooted at X, joined at the end:

      A: Conv(X, Wa) → ca0, Relu(ca0) → ca1, MaxPool(ca1) → ca2
      B: Conv(X, Wb) → cb0, Relu(cb0) → cb1
      Y = Add(ca2, cb1)

    The two Convs at the root must serialize on ConvKernel, but the
    downstream Relu/Pool of chain A can overlap chain B's Conv/Relu.
    Good fixture for a list-scheduler that mixes lanes mid-graph.
    """
    inits = [_conv1x1_init("Wa", c, seed=4),
             _conv1x1_init("Wb", c, seed=5)]
    nodes = [
        helper.make_node("Conv", ["X", "Wa"], ["ca0"], name="convA",
                         kernel_shape=[1, 1], pads=[0, 0, 0, 0],
                         strides=[1, 1]),
        helper.make_node("Relu", ["ca0"], ["ca1"], name="reluA"),
        helper.make_node("MaxPool", ["ca1"], ["ca2"], name="poolA",
                         kernel_shape=[3, 3], pads=[1, 1, 1, 1],
                         strides=[1, 1]),
        helper.make_node("Conv", ["X", "Wb"], ["cb0"], name="convB",
                         kernel_shape=[1, 1], pads=[0, 0, 0, 0],
                         strides=[1, 1]),
        helper.make_node("Relu", ["cb0"], ["cb1"], name="reluB"),
        helper.make_node("Add",  ["ca2", "cb1"], ["Y"], name="join"),
    ]
    graph = helper.make_graph(
        nodes, "parallel_two_chains",
        inputs=[_vi("X", [1, c, h, w])],
        outputs=[_vi("Y", [1, c, h, w])],
        value_info=[
            _vi("ca0", [1, c, h, w]),
            _vi("ca1", [1, c, h, w]),
            _vi("ca2", [1, c, h, w]),
            _vi("cb0", [1, c, h, w]),
            _vi("cb1", [1, c, h, w]),
        ],
        initializer=inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    _save(model, "parallel_two_chains.onnx")


# ------------------------------------------------------------------ #
# Two parallel branches that target the SAME hardware kernel           #
# ------------------------------------------------------------------ #

def gen_parallel_same_kernel_branches(c=4, h=8, w=8) -> None:
    """X → Conv(Wa) → a, X → Conv(Wb) → b, Add(a,b) → Y.

    The DAG marks Conv_a and Conv_b as concurrency-independent, but the
    scheduler must serialize them because there is exactly one
    ConvKernel.  Negative-test fixture for the kernel-lane resource
    constraint.
    """
    inits = [_conv1x1_init("Wa", c, seed=6),
             _conv1x1_init("Wb", c, seed=7)]
    nodes = [
        helper.make_node("Conv", ["X", "Wa"], ["a"], name="convA",
                         kernel_shape=[1, 1], pads=[0, 0, 0, 0],
                         strides=[1, 1]),
        helper.make_node("Conv", ["X", "Wb"], ["b"], name="convB",
                         kernel_shape=[1, 1], pads=[0, 0, 0, 0],
                         strides=[1, 1]),
        helper.make_node("Add",  ["a", "b"], ["Y"], name="add"),
    ]
    graph = helper.make_graph(
        nodes, "parallel_same_kernel_branches",
        inputs=[_vi("X", [1, c, h, w])],
        outputs=[_vi("Y", [1, c, h, w])],
        value_info=[_vi("a", [1, c, h, w]), _vi("b", [1, c, h, w])],
        initializer=inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    _save(model, "parallel_same_kernel_branches.onnx")


# ------------------------------------------------------------------ #
# Realistic ResNet-style residual block                                #
# ------------------------------------------------------------------ #

def gen_resnet_block(c=4, h=8, w=8) -> None:
    """X → Conv(W1) → Relu → Conv(W2) → Add(skip=X) → Y.

    Classic skip-connection pattern.  The skip-Add depends on the second
    Conv's output and on X (graph input → external).  Inside the main
    chain the Relu can overlap the next Conv's setup; the skip-Add
    can be issued on VectorOP as soon as the second Conv finishes.
    """
    inits = [_conv3x3_same_init("W1", c, seed=8),
             _conv3x3_same_init("W2", c, seed=9)]
    nodes = [
        helper.make_node("Conv", ["X", "W1"], ["m0"], name="conv1",
                         kernel_shape=[3, 3], pads=[1, 1, 1, 1],
                         strides=[1, 1]),
        helper.make_node("Relu", ["m0"], ["m1"], name="relu1"),
        helper.make_node("Conv", ["m1", "W2"], ["m2"], name="conv2",
                         kernel_shape=[3, 3], pads=[1, 1, 1, 1],
                         strides=[1, 1]),
        helper.make_node("Add",  ["X", "m2"], ["Y"], name="skip_add"),
    ]
    graph = helper.make_graph(
        nodes, "resnet_block",
        inputs=[_vi("X", [1, c, h, w])],
        outputs=[_vi("Y", [1, c, h, w])],
        value_info=[
            _vi("m0", [1, c, h, w]),
            _vi("m1", [1, c, h, w]),
            _vi("m2", [1, c, h, w]),
        ],
        initializer=inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    _save(model, "resnet_block.onnx")


# ------------------------------------------------------------------ #
# Asymmetric branches with nested sub-branches                         #
# ------------------------------------------------------------------ #

def gen_asymmetric_nested_branches(c=4, h=8, w=8) -> None:
    """Two top-level branches of different depth, plus a graph-input skip;
    the shallow branch fans out internally into two sub-branches that
    rejoin before the top-level join.

    Shape: every intermediate is [1, c, h, w] so all Adds are exact-match
    binary ops with no broadcasting.

    Branch A (deep, 4 nodes — Conv/VectorOP chain):

        X --Conv(W_a1)--> ar0 --Relu--> ar1 --Conv(W_a2)--> ar2 --Relu--> ar3

    Branch B (shallow with two sub-branches — Pool ‖ {Mul→Relu}):

        X --MaxPool------>  bp  --\\
                                   +--Add--> b_out
        X --Mul(X,X)--> bm --Relu--> br --/

    Final tree (two chained joins, with a graph-input skip):

        ab  = Add(ar3, b_out)        # join branch A and branch B
        Y   = Add(ab,  X)            # skip-add the original input

    DAG node order (sched.index):
        0 convA1, 1 reluA1, 2 convA2, 3 reluA2,
        4 poolB,  5 mulB,   6 reluB,
        7 joinB,  8 finalJoin1, 9 finalJoin2

    Three roots (0, 4, 5); branch A is fully independent of branch B
    until node 8; inside branch B the sub-branches (4) ‖ (5→6) are
    independent until node 7.  Demonstrates that the DAG handles nested,
    asymmetric branching uniformly — no special case for sub-branches.
    """
    inits = [
        _conv1x1_init("W_a1", c, seed=10),
        _conv1x1_init("W_a2", c, seed=11),
    ]
    nodes = [
        # branch A — deep Conv/VectorOP chain
        helper.make_node("Conv", ["X", "W_a1"], ["ar0"], name="convA1",
                         kernel_shape=[1, 1], pads=[0, 0, 0, 0],
                         strides=[1, 1]),
        helper.make_node("Relu", ["ar0"], ["ar1"], name="reluA1"),
        helper.make_node("Conv", ["ar1", "W_a2"], ["ar2"], name="convA2",
                         kernel_shape=[1, 1], pads=[0, 0, 0, 0],
                         strides=[1, 1]),
        helper.make_node("Relu", ["ar2"], ["ar3"], name="reluA2"),
        # branch B — sub-branch 1 (Pool)
        helper.make_node("MaxPool", ["X"], ["bp"], name="poolB",
                         kernel_shape=[3, 3], pads=[1, 1, 1, 1],
                         strides=[1, 1]),
        # branch B — sub-branch 2 (Mul → Relu)
        helper.make_node("Mul",  ["X", "X"], ["bm"], name="mulB"),
        helper.make_node("Relu", ["bm"],     ["br"], name="reluB"),
        # branch B — sub-branch join
        helper.make_node("Add",  ["bp", "br"], ["b_out"], name="joinB"),
        # top-level join A ⊕ B
        helper.make_node("Add",  ["ar3", "b_out"], ["ab"], name="joinAB"),
        # skip-add the original input
        helper.make_node("Add",  ["ab", "X"], ["Y"], name="skipAdd"),
    ]
    graph = helper.make_graph(
        nodes, "asymmetric_nested_branches",
        inputs=[_vi("X", [1, c, h, w])],
        outputs=[_vi("Y", [1, c, h, w])],
        value_info=[
            _vi("ar0",   [1, c, h, w]),
            _vi("ar1",   [1, c, h, w]),
            _vi("ar2",   [1, c, h, w]),
            _vi("ar3",   [1, c, h, w]),
            _vi("bp",    [1, c, h, w]),
            _vi("bm",    [1, c, h, w]),
            _vi("br",    [1, c, h, w]),
            _vi("b_out", [1, c, h, w]),
            _vi("ab",    [1, c, h, w]),
        ],
        initializer=inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    _save(model, "asymmetric_nested_branches.onnx")


# ------------------------------------------------------------------ #
# NOP-layer corner cases (Dropout, Reshape, Squeeze, ...) under         #
# parallel kernel execution.                                            #
#                                                                       #
# All five fixtures exercise interactions between buffer-aliasing       #
# layers and the cross-lane wait scheme:                                #
#   * predecessor-wait analysis must walk through ReshapeNodes to       #
#     reach the real producing kernel,                                  #
#   * pool-slot colouring must keep an aliased buffer live for as long  #
#     as ANY consumer (reached via the alias chain) is in flight on its #
#     lane.                                                             #
# ------------------------------------------------------------------ #


def gen_nop_dropout_fork(c=4, h=8, w=8) -> None:
    """Conv → Dropout → fork to (Pool ‖ Mul) → Add.

    Dropout is a single NOP at the branching point.  Its two consumers
    target different hardware lanes (Pool and VectorOP) so they should
    run in parallel after Conv drains.
    """
    inits = [_conv1x1_init("Wc", c, seed=20)]
    nodes = [
        helper.make_node("Conv", ["X", "Wc"], ["C"], name="conv",
                         kernel_shape=[1, 1], pads=[0, 0, 0, 0],
                         strides=[1, 1]),
        helper.make_node("Dropout", ["C"], ["D"], name="drop",
                         ratio=0.5),
        helper.make_node("MaxPool", ["D"], ["P"], name="pool",
                         kernel_shape=[3, 3], pads=[1, 1, 1, 1],
                         strides=[1, 1]),
        helper.make_node("Mul",  ["D", "D"], ["M"], name="mul"),
        helper.make_node("Add",  ["P", "M"], ["Y"], name="join"),
    ]
    graph = helper.make_graph(
        nodes, "nop_dropout_fork",
        inputs=[_vi("X", [1, c, h, w])],
        outputs=[_vi("Y", [1, c, h, w])],
        value_info=[
            _vi("C", [1, c, h, w]),
            _vi("D", [1, c, h, w]),
            _vi("P", [1, c, h, w]),
            _vi("M", [1, c, h, w]),
        ],
        initializer=inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    _save(model, "nop_dropout_fork.onnx")


def gen_nop_chain_dropout_fork(c=4, h=8, w=8) -> None:
    """Conv → Dropout → Dropout → Dropout → fork to (Pool ‖ Mul) → Add.

    Three chained NOPs.  Each successive consumer must trace the alias
    chain back to Conv when computing predecessor waits, and Conv's
    output buffer must be kept live until both Pool and Mul have
    drained on their respective lanes.
    """
    inits = [_conv1x1_init("Wc", c, seed=21)]
    nodes = [
        helper.make_node("Conv", ["X", "Wc"], ["C"], name="conv",
                         kernel_shape=[1, 1], pads=[0, 0, 0, 0],
                         strides=[1, 1]),
        helper.make_node("Dropout", ["C"],  ["D1"], name="drop1", ratio=0.1),
        helper.make_node("Dropout", ["D1"], ["D2"], name="drop2", ratio=0.1),
        helper.make_node("Dropout", ["D2"], ["D3"], name="drop3", ratio=0.1),
        helper.make_node("MaxPool", ["D3"], ["P"], name="pool",
                         kernel_shape=[3, 3], pads=[1, 1, 1, 1],
                         strides=[1, 1]),
        helper.make_node("Mul",  ["D3", "D3"], ["M"], name="mul"),
        helper.make_node("Add",  ["P", "M"],   ["Y"], name="join"),
    ]
    graph = helper.make_graph(
        nodes, "nop_chain_dropout_fork",
        inputs=[_vi("X", [1, c, h, w])],
        outputs=[_vi("Y", [1, c, h, w])],
        value_info=[
            _vi("C",  [1, c, h, w]),
            _vi("D1", [1, c, h, w]),
            _vi("D2", [1, c, h, w]),
            _vi("D3", [1, c, h, w]),
            _vi("P",  [1, c, h, w]),
            _vi("M",  [1, c, h, w]),
        ],
        initializer=inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    _save(model, "nop_chain_dropout_fork.onnx")


def gen_nop_graph_input_fork(c=4, h=8, w=8) -> None:
    """X (graph input) → Dropout → fork to (Pool ‖ Mul) → Add.

    The aliased tensor is rooted at a graph input, exercising
    `_run_reshape_aliases` (the X1 = X pointer assignment that lives at
    the top of inference_run).  No producer wait is needed for graph
    inputs, but Pool and Mul must still both safely read the same
    buffer in parallel.
    """
    nodes = [
        helper.make_node("Dropout", ["X"], ["X1"], name="drop", ratio=0.0),
        helper.make_node("MaxPool", ["X1"], ["P"], name="pool",
                         kernel_shape=[3, 3], pads=[1, 1, 1, 1],
                         strides=[1, 1]),
        helper.make_node("Mul",  ["X1", "X1"], ["M"], name="mul"),
        helper.make_node("Add",  ["P", "M"],   ["Y"], name="join"),
    ]
    graph = helper.make_graph(
        nodes, "nop_graph_input_fork",
        inputs=[_vi("X", [1, c, h, w])],
        outputs=[_vi("Y", [1, c, h, w])],
        value_info=[
            _vi("X1", [1, c, h, w]),
            _vi("P",  [1, c, h, w]),
            _vi("M",  [1, c, h, w]),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    _save(model, "nop_graph_input_fork.onnx")


def gen_nop_asymmetric_branch(c=4, h=8, w=8) -> None:
    """Asymmetric branches: one direct, one through a NOP chain.

      X → Conv → C
              ├── Pool(C) → P                   (direct, branch A)
              └── Drop(C) → D → Drop(D) → D2 → Mul(D2, D2) → M  (NOP-padded B)
              Add(P, M) → Y

    Tests that the consumer at the end of branch B's NOP chain still
    waits on Conv's lane, and that branch A reads the unaliased buffer
    in parallel without spurious waits.
    """
    inits = [_conv1x1_init("Wc", c, seed=22)]
    nodes = [
        helper.make_node("Conv", ["X", "Wc"], ["C"], name="conv",
                         kernel_shape=[1, 1], pads=[0, 0, 0, 0],
                         strides=[1, 1]),
        helper.make_node("MaxPool", ["C"], ["P"], name="pool",
                         kernel_shape=[3, 3], pads=[1, 1, 1, 1],
                         strides=[1, 1]),
        helper.make_node("Dropout", ["C"],  ["D"],  name="drop1", ratio=0.0),
        helper.make_node("Dropout", ["D"],  ["D2"], name="drop2", ratio=0.0),
        helper.make_node("Mul",  ["D2", "D2"], ["M"], name="mul"),
        helper.make_node("Add",  ["P", "M"],   ["Y"], name="join"),
    ]
    graph = helper.make_graph(
        nodes, "nop_asymmetric_branch",
        inputs=[_vi("X", [1, c, h, w])],
        outputs=[_vi("Y", [1, c, h, w])],
        value_info=[
            _vi("C",  [1, c, h, w]),
            _vi("D",  [1, c, h, w]),
            _vi("D2", [1, c, h, w]),
            _vi("P",  [1, c, h, w]),
            _vi("M",  [1, c, h, w]),
        ],
        initializer=inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    _save(model, "nop_asymmetric_branch.onnx")


def gen_nop_nested_inner_branch(c=4, h=8, w=8) -> None:
    """Outer fork; one branch has a real op followed by a NOP chain
    feeding a second real op, and joins the outer pair.

      X → Conv(W_outer) → outer_C
        ├── Pool(outer_C) → A_p                              (branch A)
        └── Conv(outer_C, W_b) → b_c
            → Drop → Drop → Mul(., .) → b_m                   (branch B)
        Add(A_p, b_m) → Y

    outer_C is consumed concurrently by Pool (branch A) and the inner
    Conv (branch B), exercising shared-buffer parallelism across
    different lanes.  The branch-B Mul reads b_c through two Dropouts
    so its predecessor-wait must trace back to the inner Conv.
    """
    inits = [
        _conv1x1_init("W_outer", c, seed=23),
        _conv1x1_init("W_b",     c, seed=24),
    ]
    nodes = [
        helper.make_node("Conv", ["X", "W_outer"], ["outer_C"], name="conv_outer",
                         kernel_shape=[1, 1], pads=[0, 0, 0, 0],
                         strides=[1, 1]),
        helper.make_node("MaxPool", ["outer_C"], ["A_p"], name="pool_A",
                         kernel_shape=[3, 3], pads=[1, 1, 1, 1],
                         strides=[1, 1]),
        helper.make_node("Conv", ["outer_C", "W_b"], ["b_c"], name="conv_B",
                         kernel_shape=[1, 1], pads=[0, 0, 0, 0],
                         strides=[1, 1]),
        helper.make_node("Dropout", ["b_c"], ["b_d"],  name="drop_b1", ratio=0.0),
        helper.make_node("Dropout", ["b_d"], ["b_d2"], name="drop_b2", ratio=0.0),
        helper.make_node("Mul",  ["b_d2", "b_d2"], ["b_m"], name="mul_B"),
        helper.make_node("Add",  ["A_p", "b_m"],   ["Y"],   name="join"),
    ]
    graph = helper.make_graph(
        nodes, "nop_nested_inner_branch",
        inputs=[_vi("X", [1, c, h, w])],
        outputs=[_vi("Y", [1, c, h, w])],
        value_info=[
            _vi("outer_C", [1, c, h, w]),
            _vi("A_p",     [1, c, h, w]),
            _vi("b_c",     [1, c, h, w]),
            _vi("b_d",     [1, c, h, w]),
            _vi("b_d2",    [1, c, h, w]),
            _vi("b_m",     [1, c, h, w]),
        ],
        initializer=inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    _save(model, "nop_nested_inner_branch.onnx")


# ------------------------------------------------------------------ #
# Entry point                                                          #
# ------------------------------------------------------------------ #

def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Generating parallel-scheduling fixture models in {OUT_DIR}/")
    gen_parallel_relu_mul_add()
    gen_parallel_conv_pool_join()
    gen_parallel_matmul_relu_join()
    gen_parallel_three_kernel_branches()
    gen_parallel_two_chains()
    gen_parallel_same_kernel_branches()
    gen_resnet_block()
    gen_asymmetric_nested_branches()
    gen_nop_dropout_fork()
    gen_nop_chain_dropout_fork()
    gen_nop_graph_input_fork()
    gen_nop_asymmetric_branch()
    gen_nop_nested_inner_branch()


if __name__ == "__main__":
    main()
