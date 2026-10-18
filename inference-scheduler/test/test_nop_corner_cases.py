# Copyright 2025 GradeBuilder SL
# SPDX-License-Identifier: Apache-2.0

"""Corner-case tests for NOP layers (Dropout / Squeeze / Reshape / ...) under
the parallel-wait code generator.

Each fixture model exercises one specific interaction between the buffer
aliasing machinery (ReshapeNode emits no kernel work but its output
shares the source's DMA buffer) and the cross-lane wait scheme:

  * predecessor-wait analysis must walk *through* a ReshapeNode chain to
    reach the real producing kernel,
  * pool-slot colouring must keep an aliased buffer live for as long as
    ANY consumer reached via the alias chain is still in flight on its
    lane.

These tests were added after on-device runs surfaced two bugs that were
masked by sequential execution: a missing wait when MatMul read a Pool
output through a Squeeze (`squeeze_then_matmul.onnx`), and slot
aliasing across overlapping intervals (`parallel_two_chains.onnx`).
"""

import os
import re
import unittest

from helpers import _model
from src.graph    import OnnxGraph
from src.codegen  import CodeGenerator
from src.schedule import Dag


def _gen_source(model_name: str) -> str:
    g  = OnnxGraph(_model(model_name))
    cg = CodeGenerator(g, model_path=_model(model_name))
    return cg.generate_source()


def _has_model(name: str) -> bool:
    return os.path.isfile(_model(name))


def _inference_run_body(src: str) -> str:
    m = re.search(r"void inference_run\([^)]*\)\s*\{(.*?)\n\}", src, re.S)
    assert m is not None, "inference_run() not found"
    return m.group(1)


def _slot_offset(src: str, c_name: str) -> int:
    """Pool offset for a buffer view, or raise."""
    m = re.search(
        rf"inference_buf_init_view\(&_s_buf_{c_name},"
        rf"\s*s_alloc_pool,\s*(\d+)u,",
        src,
    )
    assert m is not None, f"buffer view for {c_name} not found"
    return int(m.group(1))


def _wait_count(body: str, kid: str) -> int:
    return len(re.findall(rf"kernel_wait\({kid}\);", body))


# Required fixtures (from gen_parallel_models.py)
_REQUIRED = [
    "nop_dropout_fork.onnx",
    "nop_chain_dropout_fork.onnx",
    "nop_graph_input_fork.onnx",
    "nop_asymmetric_branch.onnx",
    "nop_nested_inner_branch.onnx",
]


def _all_models_present() -> bool:
    return all(_has_model(n) for n in _REQUIRED)


@unittest.skipUnless(_all_models_present(),
                     "Run test/gen_parallel_models.py first")
class TestNopDropoutFork(unittest.TestCase):
    """Conv → Dropout → fork to (Pool ‖ Mul) → Add.

    Single Dropout NOP at the branch fork.  Pool and Mul target different
    lanes; they must run in parallel after Conv drains.  C must stay
    alive until BOTH branches' lanes have drained.
    """

    def setUp(self):
        self.src  = _gen_source("nop_dropout_fork.onnx")
        self.body = _inference_run_body(self.src)

    def test_dag_dropout_traversed_for_predecessors(self):
        # In the DAG the Dropout is a real node, so MaxPool's direct
        # predecessor is the Dropout (idx 1) — but for wait emission
        # we walk through it to the Conv (idx 0).
        dag = Dag.from_graph(OnnxGraph(_model("nop_dropout_fork.onnx")))
        self.assertEqual(dag.predecessors(2), {1})    # Pool ← Dropout
        self.assertEqual(dag.predecessors(3), {1})    # Mul  ← Dropout
        self.assertEqual(dag.predecessors(1), {0})    # Dropout ← Conv

    def test_conv_wait_emitted_once(self):
        # Both Pool and Mul effectively depend on Conv (through Dropout),
        # but only ONE kernel_wait(KERNEL_CONV) should fire — once Conv
        # is drained for the first consumer, the second sees pending
        # empty and emits no extra wait.
        self.assertEqual(_wait_count(self.body, "KERNEL_CONV"), 1)

    def test_pool_and_mul_overlap(self):
        m_pool = re.search(r"run_pool\([^;]*\);", self.body, re.S)
        m_mul  = re.search(
            r"run_op\([^;]*VECTOROP_MUL[^;]*\);", self.body, re.S)
        self.assertIsNotNone(m_pool)
        self.assertIsNotNone(m_mul)
        self.assertLess(m_pool.start(), m_mul.start())
        between = self.body[m_pool.end():m_mul.start()]
        self.assertNotIn("kernel_wait(KERNEL_POOL)", between,
                         "Pool wait between Pool start and Mul start kills overlap")
        self.assertNotIn("kernel_wait(KERNEL_VECTOROP)", between,
                         "VOP wait inserted before Mul start")

    def test_C_buffer_held_until_both_consumers_drain(self):
        # C, P, M must each occupy their own slot — at the time Pool
        # is reading C the Mul lane also reads C in parallel, and Pool
        # is still in flight when Mul writes M.
        c_off = _slot_offset(self.src, "C")
        p_off = _slot_offset(self.src, "P")
        m_off = _slot_offset(self.src, "M")
        self.assertEqual(len({c_off, p_off, m_off}), 3,
                         f"C/P/M aliased: C={c_off} P={p_off} M={m_off}")


@unittest.skipUnless(_all_models_present(),
                     "Run test/gen_parallel_models.py first")
class TestNopChainDropoutFork(unittest.TestCase):
    """Conv → Drop → Drop → Drop → fork to (Pool ‖ Mul) → Add.

    Three chained NOPs.  The alias walker must trace through all three
    Dropouts to reach Conv when computing predecessor waits, and the
    underlying buffer must stay live until both terminal consumers
    drain.
    """

    def setUp(self):
        self.src  = _gen_source("nop_chain_dropout_fork.onnx")
        self.body = _inference_run_body(self.src)

    def test_three_chained_reshape_nodes_in_dag(self):
        dag = Dag.from_graph(OnnxGraph(
            _model("nop_chain_dropout_fork.onnx")))
        # Conv(0) → Drop(1) → Drop(2) → Drop(3) → Pool(4) ‖ Mul(5) → Add(6).
        self.assertEqual(dag.predecessors(1), {0})
        self.assertEqual(dag.predecessors(2), {1})
        self.assertEqual(dag.predecessors(3), {2})
        self.assertEqual(dag.predecessors(4), {3})
        self.assertEqual(dag.predecessors(5), {3})

    def test_walk_traverses_full_chain(self):
        # Despite the 3-deep alias chain, exactly one Conv wait fires.
        self.assertEqual(_wait_count(self.body, "KERNEL_CONV"), 1)
        # Pool wait fires only once at the join.
        self.assertEqual(_wait_count(self.body, "KERNEL_POOL"), 1)

    def test_pool_and_mul_still_overlap(self):
        m_pool = re.search(r"run_pool\([^;]*\);", self.body, re.S)
        m_mul  = re.search(
            r"run_op\([^;]*VECTOROP_MUL[^;]*\);", self.body, re.S)
        between = self.body[m_pool.end():m_mul.start()]
        self.assertNotIn("kernel_wait(KERNEL_POOL)", between)


@unittest.skipUnless(_all_models_present(),
                     "Run test/gen_parallel_models.py first")
class TestNopGraphInputFork(unittest.TestCase):
    """X (graph input) → Dropout → fork to (Pool ‖ Mul) → Add.

    The aliased tensor is rooted at a graph input.  No producer wait is
    needed (graph inputs have no producing node), but two lanes still
    read the same buffer concurrently.
    """

    def setUp(self):
        self.src  = _gen_source("nop_graph_input_fork.onnx")
        self.body = _inference_run_body(self.src)

    def test_input_alias_pointer_assignment(self):
        # X1 = X must be emitted at the top of inference_run before any
        # kernel call uses X1.
        m_assign = re.search(r"X1\s*=\s*X;", self.body)
        m_pool   = re.search(r"run_pool\(X1", self.body)
        self.assertIsNotNone(m_assign, "X1 = X alias assignment missing")
        self.assertIsNotNone(m_pool)
        self.assertLess(m_assign.start(), m_pool.start())

    def test_no_producer_waits(self):
        # X has no producer → no Conv/Matmul/Pool wait should be emitted
        # before the parallel branches fire.  Only the join's lane waits
        # and the final drain are expected.
        m_pool = re.search(r"run_pool\(X1", self.body)
        prefix = self.body[:m_pool.start()]
        self.assertEqual(_wait_count(prefix, "KERNEL_CONV"),     0)
        self.assertEqual(_wait_count(prefix, "KERNEL_POOL"),     0)
        self.assertEqual(_wait_count(prefix, "KERNEL_VECTOROP"), 0)
        self.assertEqual(_wait_count(prefix, "KERNEL_MATMUL"),   0)

    def test_pool_and_mul_overlap_on_graph_input_alias(self):
        m_pool = re.search(r"run_pool\(X1", self.body)
        m_mul  = re.search(
            r"run_op\(X1, X1[^;]*VECTOROP_MUL[^;]*\);", self.body, re.S)
        self.assertIsNotNone(m_pool)
        self.assertIsNotNone(m_mul)
        between = self.body[m_pool.end():m_mul.start()]
        self.assertNotIn("kernel_wait", between,
                         "any wait between graph-input parallel branches "
                         "is unnecessary and kills overlap")


@unittest.skipUnless(_all_models_present(),
                     "Run test/gen_parallel_models.py first")
class TestNopAsymmetricBranch(unittest.TestCase):
    """Asymmetric branches — A reads C directly, B goes through 2 Dropouts.

      X → Conv → C
              ├── Pool(C) → P                                 (branch A)
              └── Drop(C) → D → Drop(D) → D2 → Mul(D2,D2) → M (branch B)
              Add(P, M) → Y
    """

    def setUp(self):
        self.src  = _gen_source("nop_asymmetric_branch.onnx")
        self.body = _inference_run_body(self.src)

    def test_dag_structure(self):
        dag = Dag.from_graph(OnnxGraph(
            _model("nop_asymmetric_branch.onnx")))
        # 0 conv, 1 pool, 2 drop1, 3 drop2, 4 mul, 5 add
        self.assertEqual(dag.predecessors(1), {0})  # Pool ← Conv (direct)
        self.assertEqual(dag.predecessors(2), {0})  # Drop ← Conv
        self.assertEqual(dag.predecessors(3), {2})  # Drop ← Drop
        self.assertEqual(dag.predecessors(4), {3})  # Mul ← Drop
        self.assertEqual(dag.predecessors(5), {1, 4})  # Add ← Pool, Mul

    def test_single_conv_wait_serves_both_branches(self):
        # Branch A is direct, branch B is via 2 Dropouts — but they share
        # the same producer (Conv).  Conv's lane must drain exactly once.
        self.assertEqual(_wait_count(self.body, "KERNEL_CONV"), 1)

    def test_C_buffer_distinct_from_M(self):
        # C must outlive both Pool's and Mul's reads.  Mul writes M,
        # which must NOT alias C (that would clobber the Pool read).
        c_off = _slot_offset(self.src, "C")
        m_off = _slot_offset(self.src, "M")
        self.assertNotEqual(c_off, m_off)


@unittest.skipUnless(_all_models_present(),
                     "Run test/gen_parallel_models.py first")
class TestNopNestedInnerBranch(unittest.TestCase):
    """Outer fork; one branch contains a NOP chain feeding a second op.

      X → Conv(W_outer) → outer_C
        ├── Pool(outer_C) → A_p                               (branch A)
        └── Conv(outer_C, W_b) → b_c
            → Drop → Drop → Mul(., .) → b_m                   (branch B)
        Add(A_p, b_m) → Y
    """

    def setUp(self):
        self.src  = _gen_source("nop_nested_inner_branch.onnx")
        self.body = _inference_run_body(self.src)

    def test_outer_C_consumed_by_two_lanes(self):
        # Both Pool and the inner Conv read outer_C in parallel.  outer_C
        # must NOT alias either of their write targets (A_p, b_c).
        outer = _slot_offset(self.src, "outer_C")
        a_p   = _slot_offset(self.src, "A_p")
        b_c   = _slot_offset(self.src, "b_c")
        self.assertEqual(len({outer, a_p, b_c}), 3,
                         f"outer_C/A_p/b_c aliased: outer={outer} "
                         f"A_p={a_p} b_c={b_c}")

    def test_pool_overlaps_inner_conv(self):
        # Pool must be issued before inner Conv with NO Pool wait between.
        m_pool = re.search(
            r"run_pool\(outer_C", self.body)
        m_conv = re.search(
            r"run_conv\(outer_C, W_b", self.body)
        self.assertIsNotNone(m_pool)
        self.assertIsNotNone(m_conv)
        self.assertLess(m_pool.start(), m_conv.start())
        between = self.body[m_pool.end():m_conv.start()]
        self.assertNotIn("kernel_wait(KERNEL_POOL)", between,
                         "Pool wait kills outer-fork overlap")

    def test_mul_waits_for_inner_conv_through_two_dropouts(self):
        # The Mul reads b_d2 which aliases b_c through two Dropouts.
        # The wait emitter must trace the chain back to the inner Conv.
        m_mul = re.search(
            r"run_op\(b_d2, b_d2[^;]*\);", self.body, re.S)
        self.assertIsNotNone(m_mul)
        prefix = self.body[:m_mul.start()]
        # At least one Conv wait must fire before the Mul (covers inner
        # Conv's drain through the alias chain).
        self.assertGreaterEqual(_wait_count(prefix, "KERNEL_CONV"), 1)

    def test_b_c_kept_alive_through_alias_chain(self):
        # b_c is consumed by Mul reading b_d2 (2-deep alias).  The Mul
        # writes b_m on the VectorOP lane while the slot allocator must
        # treat b_c as live until the VectorOP wait for Mul.  b_c MUST
        # NOT share a slot with b_m.
        b_c = _slot_offset(self.src, "b_c")
        b_m = _slot_offset(self.src, "b_m")
        self.assertNotEqual(b_c, b_m,
                            "b_c slot reused by b_m — alias chain liveness "
                            "is broken")


# ----------------------------------------------------------------- #
# Cross-cutting safety check: every fixture preserves the invariant  #
# "no two tensors with overlapping event-stream intervals share a    #
# pool slot".  This protects against silent regressions in the slot  #
# allocator that wouldn't show up in the per-fixture asserts.        #
# ----------------------------------------------------------------- #


@unittest.skipUnless(_all_models_present(),
                     "Run test/gen_parallel_models.py first")
class TestNopFixturesNoSlotAliasing(unittest.TestCase):
    """For every NOP corner-case fixture, walk the schedule and assert
    that no two tensors with overlapping event-stream intervals end up
    in the same pool slot."""

    MODELS = _REQUIRED

    def test_no_slot_aliasing_with_overlapping_intervals(self):
        for name in self.MODELS:
            with self.subTest(model=name):
                g  = OnnxGraph(_model(name))
                cg = CodeGenerator(g, model_path=_model(name))
                src = cg.generate_source()
                intervals = cg._compute_live_intervals()
                # Map tensor -> pool offset.
                offsets = {}
                for ti in g.intermediate_tensors:
                    m = re.search(
                        rf"inference_buf_init_view\(&_s_buf_{ti.c_name},"
                        rf"\s*s_alloc_pool,\s*(\d+)u,",
                        src,
                    )
                    if m is not None:
                        offsets[ti.onnx_name] = int(m.group(1))

                names = [n for n in offsets if n in intervals]
                for i, a in enumerate(names):
                    for b in names[i + 1:]:
                        if offsets[a] != offsets[b]:
                            continue
                        # Same slot — intervals must be disjoint.
                        sa, ea = intervals[a]
                        sb, eb = intervals[b]
                        self.assertTrue(
                            ea < sb or eb < sa,
                            f"{name}: {a} and {b} share pool slot "
                            f"{offsets[a]} but their event intervals "
                            f"overlap: {a}=[{sa},{ea}], {b}=[{sb},{eb}]"
                        )


if __name__ == "__main__":
    unittest.main()
