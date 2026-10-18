# Copyright 2025 GradeBuilder SL
# SPDX-License-Identifier: Apache-2.0

"""Tests for src.schedule.Dag — data-flow DAG over scheduled nodes."""

import os
import tempfile
import unittest

import numpy as np
import onnx
import onnx.helper as oh
from onnx import TensorProto

from helpers import _model, _models_exist
from src.graph    import OnnxGraph
from src.schedule import Dag


# ---------------------------------------------------------------- #
# Tiny in-test ONNX-graph builders                                  #
# ---------------------------------------------------------------- #

def _save(graph: onnx.GraphProto) -> str:
    """Wrap a GraphProto in a Model, write to a tmp file, return the path."""
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
    model.ir_version = 7
    fd, path = tempfile.mkstemp(suffix=".onnx")
    os.close(fd)
    onnx.save(model, path)
    return path


def _vi(name: str, shape):
    return oh.make_tensor_value_info(name, TensorProto.FLOAT, shape)


def _two_branch_relu_mul_add() -> str:
    """X -> Relu -> a, Mul(X,X) -> b, Add(a,b) -> Y.

    a and b are independent: both consume X; neither feeds the other.
    Add consumes both.
    """
    X = _vi("X", [8])
    Y = _vi("Y", [8])
    a = _vi("a", [8])
    b = _vi("b", [8])
    nodes = [
        oh.make_node("Relu", inputs=["X"],      outputs=["a"], name="n_relu"),
        oh.make_node("Mul",  inputs=["X", "X"], outputs=["b"], name="n_mul"),
        oh.make_node("Add",  inputs=["a", "b"], outputs=["Y"], name="n_add"),
    ]
    graph = oh.make_graph(nodes, "two_branch", [X], [Y], value_info=[a, b])
    return _save(graph)


def _linear_three_relu() -> str:
    """X -> Relu -> a -> Relu -> b -> Relu -> Y (pure chain)."""
    X = _vi("X", [4])
    Y = _vi("Y", [4])
    a = _vi("a", [4])
    b = _vi("b", [4])
    nodes = [
        oh.make_node("Relu", inputs=["X"], outputs=["a"], name="r1"),
        oh.make_node("Relu", inputs=["a"], outputs=["b"], name="r2"),
        oh.make_node("Relu", inputs=["b"], outputs=["Y"], name="r3"),
    ]
    return _save(oh.make_graph(nodes, "lin3", [X], [Y], value_info=[a, b]))


def _add_with_constant_weight() -> str:
    """Add(X, W) -> Y where W is a constant initializer.

    The DAG must NOT create an edge for W: it is external (a weight),
    so the Add node has zero predecessors.
    """
    X = _vi("X", [4])
    Y = _vi("Y", [4])
    w_data = np.ones(4, dtype=np.float32)
    W_init = oh.make_tensor("W", TensorProto.FLOAT, [4], w_data.tolist())
    nodes = [oh.make_node("Add", inputs=["X", "W"], outputs=["Y"], name="add")]
    graph = oh.make_graph(nodes, "add_weight", [X], [Y], initializer=[W_init])
    return _save(graph)


def _diamond() -> str:
    """X -> Relu -> a, X -> Relu -> b, Add(a,b) -> c, Mul(c,X) -> Y.

    Tests that:
      * a and b are independent (parallel branches)
      * c is the join (depends on both a and b)
      * Y depends on c AND on X (skip)
    """
    X = _vi("X", [4])
    Y = _vi("Y", [4])
    a = _vi("a", [4])
    b = _vi("b", [4])
    c = _vi("c", [4])
    nodes = [
        oh.make_node("Relu", inputs=["X"], outputs=["a"], name="ra"),
        oh.make_node("Relu", inputs=["X"], outputs=["b"], name="rb"),
        oh.make_node("Add",  inputs=["a", "b"], outputs=["c"], name="ab"),
        oh.make_node("Mul",  inputs=["c", "X"], outputs=["Y"], name="cx"),
    ]
    return _save(oh.make_graph(
        nodes, "diamond", [X], [Y], value_info=[a, b, c]
    ))


# ---------------------------------------------------------------- #
# Tests                                                             #
# ---------------------------------------------------------------- #


class TestDagLinear(unittest.TestCase):
    """A pure chain becomes a chain DAG with no parallelism."""

    def setUp(self):
        self.path = _linear_three_relu()
        self.dag  = Dag.from_graph(OnnxGraph(self.path))

    def tearDown(self):
        os.unlink(self.path)

    def test_three_nodes(self):
        self.assertEqual(len(self.dag.nodes), 3)

    def test_chain_edges(self):
        self.assertEqual(self.dag.predecessors(0), set())
        self.assertEqual(self.dag.predecessors(1), {0})
        self.assertEqual(self.dag.predecessors(2), {1})
        self.assertEqual(self.dag.successors(0), {1})
        self.assertEqual(self.dag.successors(1), {2})
        self.assertEqual(self.dag.successors(2), set())

    def test_topo_matches_graph_order(self):
        self.assertEqual(self.dag.topological_order(), [0, 1, 2])

    def test_no_independent_pairs(self):
        # In a strict chain, every pair has an ancestor relationship.
        self.assertEqual(self.dag.independent_pairs(), [])

    def test_roots_and_leaves(self):
        self.assertEqual(self.dag.roots(),  [0])
        self.assertEqual(self.dag.leaves(), [2])


class TestDagTwoBranches(unittest.TestCase):
    """Relu and Mul both fan out from X; Add joins them."""

    def setUp(self):
        self.path = _two_branch_relu_mul_add()
        self.dag  = Dag.from_graph(OnnxGraph(self.path))

    def tearDown(self):
        os.unlink(self.path)

    def test_branches_are_independent(self):
        # Nodes 0 (Relu) and 1 (Clip) both consume X (a graph input);
        # neither produces the other's input.
        self.assertEqual(self.dag.predecessors(0), set())
        self.assertEqual(self.dag.predecessors(1), set())
        self.assertIn((0, 1), self.dag.independent_pairs())

    def test_join_depends_on_both(self):
        self.assertEqual(self.dag.predecessors(2), {0, 1})

    def test_two_roots(self):
        self.assertEqual(self.dag.roots(), [0, 1])

    def test_topo_keeps_join_last(self):
        order = self.dag.topological_order()
        self.assertEqual(order[-1], 2)
        self.assertEqual(set(order[:2]), {0, 1})


class TestDagWeightsAreExternal(unittest.TestCase):
    """Constant initializers MUST NOT create dependency edges."""

    def setUp(self):
        self.path = _add_with_constant_weight()
        self.dag  = Dag.from_graph(OnnxGraph(self.path))

    def tearDown(self):
        os.unlink(self.path)

    def test_add_has_no_predecessors(self):
        self.assertEqual(self.dag.predecessors(0), set())

    def test_weight_in_externals(self):
        self.assertIn("W", self.dag.external_tensors)
        self.assertIn("X", self.dag.external_tensors)


class TestDagDiamond(unittest.TestCase):
    """X fans out to two Relus, joined by Add, then Mul with X (skip)."""

    def setUp(self):
        self.path = _diamond()
        self.dag  = Dag.from_graph(OnnxGraph(self.path))

    def tearDown(self):
        os.unlink(self.path)

    def test_branch_relus_are_independent(self):
        self.assertIn((0, 1), self.dag.independent_pairs())

    def test_join_node_has_two_preds(self):
        self.assertEqual(self.dag.predecessors(2), {0, 1})

    def test_skip_into_mul_is_external_not_edge(self):
        # Mul reads X (graph input) and c (intermediate from Add).
        # Only c contributes a DAG edge; X is external.
        self.assertEqual(self.dag.predecessors(3), {2})

    def test_topo_respects_partial_order(self):
        order = self.dag.topological_order()
        pos = {idx: p for p, idx in enumerate(order)}
        self.assertLess(pos[0], pos[2])   # Relu_a before Add
        self.assertLess(pos[1], pos[2])   # Relu_b before Add
        self.assertLess(pos[2], pos[3])   # Add    before Mul


@unittest.skipUnless(_models_exist(), "Run test/gen_test_models.py first")
class TestDagOnRealModels(unittest.TestCase):
    """Spot-check the DAG against pre-generated fixture models."""

    def test_residual_add_join(self):
        # residual_add: Relu(X) -> relu_X, then Add(X, relu_X) -> Y.
        # Two nodes; the Add depends on the Relu; X is external.
        dag = Dag.from_graph(OnnxGraph(_model("residual_add.onnx")))
        self.assertEqual(len(dag.nodes), 2)
        self.assertEqual(dag.predecessors(0), set())
        self.assertEqual(dag.predecessors(1), {0})

    def test_residual_chain_is_linear(self):
        # residual_chain: X+bias -> Relu -> *scale -> X+result -> Y.
        # X is external; the four nodes form a chain in dependency terms.
        dag = Dag.from_graph(OnnxGraph(_model("residual_chain.onnx")))
        order = dag.topological_order()
        self.assertEqual(order, sorted(order))

    def test_no_cycles_anywhere(self):
        for name in ("residual_add.onnx", "residual_chain.onnx",
                     "mixed_ops.onnx", "relu_chain.onnx"):
            with self.subTest(model=name):
                dag = Dag.from_graph(OnnxGraph(_model(name)))
                # topological_order() raises on cycle
                self.assertEqual(len(dag.topological_order()), len(dag.nodes))


# ------------------------------------------------------------------ #
# Parallel-scheduling fixture models (test/gen_parallel_models.py)    #
# ------------------------------------------------------------------ #


def _parallel_models_exist() -> bool:
    required = [
        "parallel_conv_pool_join.onnx",
        "parallel_matmul_relu_join.onnx",
        "parallel_three_kernel_branches.onnx",
        "parallel_two_chains.onnx",
        "parallel_same_kernel_branches.onnx",
        "resnet_block.onnx",
        "asymmetric_nested_branches.onnx",
    ]
    return all(os.path.isfile(_model(m)) for m in required)


@unittest.skipUnless(_parallel_models_exist(),
                     "Run test/gen_parallel_models.py first")
class TestDagOnParallelModels(unittest.TestCase):
    """DAG structure of fixtures produced by gen_parallel_models.py."""

    def _kernels(self, dag):
        return [(n.index, n.kernel_name) for n in dag.nodes]

    # ---- two parallel branches, single kernel type --------------------

    def test_relu_mul_add_two_independent_roots(self):
        dag = Dag.from_graph(OnnxGraph(_model("parallel_relu_mul_add.onnx")))
        self.assertEqual(len(dag.nodes), 3)
        self.assertEqual(dag.roots(), [0, 1])
        self.assertEqual(dag.predecessors(2), {0, 1})
        self.assertIn((0, 1), dag.independent_pairs())
        # Both branches land on the same lane — kernel-lane resource will
        # force serialization in the future scheduler.
        self.assertTrue(
            all(name == "VectorOPKernel" for _, name in self._kernels(dag))
        )

    # ---- two parallel branches, different kernel types ----------------

    def test_conv_pool_join_branches_target_different_kernels(self):
        dag = Dag.from_graph(OnnxGraph(_model("parallel_conv_pool_join.onnx")))
        self.assertEqual(len(dag.nodes), 3)
        self.assertEqual(dag.predecessors(0), set())
        self.assertEqual(dag.predecessors(1), set())
        self.assertEqual(dag.predecessors(2), {0, 1})
        self.assertIn((0, 1), dag.independent_pairs())
        kernels = dict(self._kernels(dag))
        self.assertEqual(kernels[0], "ConvKernel")
        self.assertEqual(kernels[1], "PoolKernel")
        self.assertEqual(kernels[2], "VectorOPKernel")

    def test_matmul_relu_join_uses_two_kernels(self):
        dag = Dag.from_graph(OnnxGraph(
            _model("parallel_matmul_relu_join.onnx")))
        self.assertIn((0, 1), dag.independent_pairs())
        kernels = dict(self._kernels(dag))
        self.assertEqual(kernels[0], "MatmulKernel")
        self.assertEqual(kernels[1], "VectorOPKernel")

    # ---- three parallel branches across three kernels -----------------

    def test_three_kernel_branches_all_independent(self):
        dag = Dag.from_graph(OnnxGraph(
            _model("parallel_three_kernel_branches.onnx")))
        self.assertEqual(len(dag.nodes), 5)
        # Branches 0, 1, 2 are pairwise independent.
        indep = set(dag.independent_pairs())
        self.assertIn((0, 1), indep)
        self.assertIn((0, 2), indep)
        self.assertIn((1, 2), indep)
        # First join consumes 0 and 1; second join consumes the first
        # join and 2.
        self.assertEqual(dag.predecessors(3), {0, 1})
        self.assertEqual(dag.predecessors(4), {3, 2})

    # ---- two longer chains running in parallel ------------------------

    def test_two_chains_independent_until_join(self):
        dag = Dag.from_graph(OnnxGraph(_model("parallel_two_chains.onnx")))
        # Six nodes: convA(0), reluA(1), poolA(2), convB(3), reluB(4),
        # join(5).  Chain A and chain B are independent until join.
        self.assertEqual(len(dag.nodes), 6)
        indep = set(dag.independent_pairs())
        # Cross-chain pairs all independent (any (chain-A node, chain-B node))
        for a in (0, 1, 2):
            for b in (3, 4):
                self.assertIn((min(a, b), max(a, b)), indep,
                              f"chain pair ({a},{b}) should be independent")
        # Within each chain, nodes are strictly ordered.
        self.assertNotIn((0, 1), indep)
        self.assertNotIn((1, 2), indep)
        self.assertNotIn((3, 4), indep)
        # join depends on the tails of both chains
        self.assertEqual(dag.predecessors(5), {2, 4})

    # ---- same-kernel branches: scheduler must serialize ---------------

    def test_same_kernel_branches_are_independent_in_dag(self):
        # The DAG only encodes data dependencies; it does NOT encode
        # the kernel-lane resource constraint.  The scheduler will
        # serialize them later because both target ConvKernel.
        dag = Dag.from_graph(OnnxGraph(
            _model("parallel_same_kernel_branches.onnx")))
        self.assertIn((0, 1), dag.independent_pairs())
        kernels = dict(self._kernels(dag))
        self.assertEqual(kernels[0], "ConvKernel")
        self.assertEqual(kernels[1], "ConvKernel")

    # ---- ResNet residual block ----------------------------------------

    # ---- asymmetric branches with nested sub-branches -----------------

    def test_asymmetric_nested_branches_structure(self):
        """Validates the DAG handles asymmetric, nested branching:
          * 10 nodes, 3 roots (convA1, poolB, mulB)
          * branch A (deep) is fully independent of branch B until joinAB
          * inside branch B, sub-branch (poolB) ‖ sub-chain (mulB → reluB)
          * the skipAdd reads X (graph input) — no spurious DAG edge
        """
        dag = Dag.from_graph(OnnxGraph(
            _model("asymmetric_nested_branches.onnx")))

        # Layout: 0 convA1, 1 reluA1, 2 convA2, 3 reluA2,
        #         4 poolB,  5 mulB,   6 reluB,  7 joinB,
        #         8 joinAB, 9 skipAdd
        self.assertEqual(len(dag.nodes), 10)
        self.assertEqual(dag.roots(), [0, 4, 5])

        # Branch A is a strict chain.
        self.assertEqual(dag.predecessors(0), set())
        self.assertEqual(dag.predecessors(1), {0})
        self.assertEqual(dag.predecessors(2), {1})
        self.assertEqual(dag.predecessors(3), {2})

        # Branch B sub-branches converge at joinB.
        self.assertEqual(dag.predecessors(4), set())     # poolB root
        self.assertEqual(dag.predecessors(5), set())     # mulB  root
        self.assertEqual(dag.predecessors(6), {5})       # reluB ← mulB
        self.assertEqual(dag.predecessors(7), {4, 6})    # joinB ← poolB, reluB

        # Top-level join consumes the deepest A node and the B-sub-join.
        self.assertEqual(dag.predecessors(8), {3, 7})

        # skipAdd reads X (external) and ab (intermediate) — only one edge.
        self.assertEqual(dag.predecessors(9), {8})

    def test_asymmetric_branch_a_independent_of_branch_b(self):
        """Every node in branch A is concurrency-independent of every
        node in branch B's sub-branches (and the sub-join), since the
        two top-level branches do not share any data dependency until
        joinAB at index 8."""
        dag = Dag.from_graph(OnnxGraph(
            _model("asymmetric_nested_branches.onnx")))
        indep = set(dag.independent_pairs())
        for a in (0, 1, 2, 3):           # branch A
            for b in (4, 5, 6, 7):       # branch B and its sub-join
                pair = (min(a, b), max(a, b))
                self.assertIn(
                    pair, indep,
                    f"branch-A node {a} and branch-B node {b} must be "
                    f"concurrency-independent",
                )

    def test_asymmetric_sub_branch_independence(self):
        """Inside branch B, the Pool sub-branch is independent of the
        Mul→Relu sub-chain; and within the Mul→Relu chain itself the two
        nodes are NOT independent (data dependency)."""
        dag = Dag.from_graph(OnnxGraph(
            _model("asymmetric_nested_branches.onnx")))
        indep = set(dag.independent_pairs())
        # Pool ‖ Mul (both roots of branch B)
        self.assertIn((4, 5), indep)
        # Pool ‖ Relu_b (Pool doesn't depend on Mul; Relu_b depends only on Mul)
        self.assertIn((4, 6), indep)
        # Mul → Relu_b is a hard data dependency.
        self.assertNotIn((5, 6), indep)

    def test_asymmetric_kernel_lane_diversity(self):
        """The fixture spans three of the four hardware kernels — Conv,
        Pool, VectorOP — so the eventual scheduler will exercise
        non-trivial multi-lane dispatch."""
        dag = Dag.from_graph(OnnxGraph(
            _model("asymmetric_nested_branches.onnx")))
        kernels = {n.kernel_name for n in dag.nodes if n.kernel_name}
        self.assertEqual(
            kernels, {"ConvKernel", "PoolKernel", "VectorOPKernel"}
        )

    def test_resnet_block_skip_is_external(self):
        dag = Dag.from_graph(OnnxGraph(_model("resnet_block.onnx")))
        # Four nodes: conv1(0), relu1(1), conv2(2), skip_add(3).
        # The skip_add reads X (external) and m2 → only one DAG edge.
        self.assertEqual(len(dag.nodes), 4)
        self.assertEqual(dag.predecessors(0), set())
        self.assertEqual(dag.predecessors(1), {0})
        self.assertEqual(dag.predecessors(2), {1})
        self.assertEqual(dag.predecessors(3), {2})
        # The chain is strictly linear in dep terms; no parallelism gain
        # from the DAG alone.  The scheduler can still overlap conv2's
        # start with relu1's wait — but that's a kernel-lane optimization,
        # not a dependency one.
        self.assertEqual(dag.independent_pairs(), [])


if __name__ == "__main__":
    unittest.main()
