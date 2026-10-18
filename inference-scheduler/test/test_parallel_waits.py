# Copyright 2025 GradeBuilder SL
# Modifications copyright 2026 Ivan Gubochkin
# SPDX-License-Identifier: Apache-2.0

"""Tests for the split start / wait emission introduced to enable
cross-lane kernel overlap.

The codegen drops the trailing `while (!XKernel_IsDone) {}` poll from each
``run_*()`` helper and emits explicit ``kernel_wait(KERNEL_*)`` calls in
``inference_run()`` only where a data dependency or lane re-use forces a
drain.  These tests assert the resulting structure on fixture models.
"""

import os
import re
import unittest

from helpers import _model
from src.graph   import OnnxGraph
from src.codegen import CodeGenerator


def _gen_source(model_name: str) -> str:
    g  = OnnxGraph(_model(model_name))
    cg = CodeGenerator(g, model_path=_model(model_name))
    return cg.generate_source()


def _has_model(name: str) -> bool:
    return os.path.isfile(_model(name))


# ---------------------------------------------------------------- #
# Helper-level structure                                            #
# ---------------------------------------------------------------- #


@unittest.skipUnless(_has_model("single_add.onnx"),
                     "Run test/gen_test_models.py first")
class TestNonBlockingHelpers(unittest.TestCase):
    """The run_*() helpers must Start the kernel and return immediately."""

    def test_run_op_helper_does_not_poll_isdone(self):
        s = _gen_source("single_add.onnx")
        # Locate the run_op() definition body.
        m = re.search(r"static void run_op\([^)]*\)\s*\{(.*?)\n\}", s, re.S)
        self.assertIsNotNone(m, "run_op() helper not found")
        body = m.group(1)
        self.assertIn("XVectoropkernel_Start", body)
        self.assertNotIn("XVectoropkernel_IsDone", body)


@unittest.skipUnless(_has_model("conv_simple.onnx"),
                     "Run test/gen_conv_models.py first")
class TestNonBlockingConvHelper(unittest.TestCase):

    def test_run_conv_helper_does_not_poll_isdone(self):
        s = _gen_source("conv_simple.onnx")
        m = re.search(r"static void run_conv\([^)]*\)\s*\{(.*?)\n\}", s, re.S)
        self.assertIsNotNone(m, "run_conv() helper not found")
        body = m.group(1)
        self.assertIn("XConvkernel_Start", body)
        self.assertNotIn("XConvkernel_IsDone", body)


@unittest.skipUnless(_has_model("pool_maxpool_simple.onnx"),
                     "Run test/gen_pool_models.py first")
class TestNonBlockingPoolHelper(unittest.TestCase):

    def test_run_pool_helper_does_not_poll_isdone(self):
        s = _gen_source("pool_maxpool_simple.onnx")
        m = re.search(r"static void run_pool\([^)]*\)\s*\{(.*?)\n\}", s, re.S)
        self.assertIsNotNone(m, "run_pool() helper not found")
        body = m.group(1)
        self.assertIn("XPoolingkernel_Start", body)
        self.assertNotIn("XPoolingkernel_IsDone", body)


# ---------------------------------------------------------------- #
# kernel_wait() primitive                                           #
# ---------------------------------------------------------------- #


@unittest.skipUnless(_has_model("conv_simple.onnx"),
                     "Run test/gen_conv_models.py first")
class TestKernelWaitPrimitive(unittest.TestCase):

    def test_kernel_id_enum_emitted(self):
        # the enum lives in the header since RPU_PLAN B1 (the firmware's
        # inference_irq.c names the lanes); inference.c includes it
        g  = OnnxGraph(_model("conv_simple.onnx"))
        s = CodeGenerator(g, model_path=_model("conv_simple.onnx")).generate_header()
        self.assertIn("typedef enum {", s)
        self.assertIn("} kernel_id_t;", s)
        # Conv-only model: only KERNEL_CONV is enumerated.
        self.assertIn("KERNEL_CONV", s)
        self.assertNotIn("KERNEL_MATMUL", s)

    def test_weak_kernel_wait_is_emitted(self):
        s = _gen_source("conv_simple.onnx")
        self.assertIn("__attribute__((weak))", s)
        self.assertIn("void kernel_wait(kernel_id_t k)", s)
        # Polling default still uses IsDone — under the hood.
        self.assertIn("XConvkernel_IsDone", s)


# ---------------------------------------------------------------- #
# Cross-lane overlap on parallel_two_chains                         #
# ---------------------------------------------------------------- #


def _inference_run_body(src: str) -> str:
    m = re.search(r"void inference_run\([^)]*\)\s*\{(.*?)\n\}", src, re.S)
    if m is None:
        raise AssertionError("inference_run() not found in source")
    return m.group(1)


@unittest.skipUnless(_has_model("parallel_two_chains.onnx"),
                     "Run test/gen_parallel_models.py first")
class TestParallelTwoChainsOverlap(unittest.TestCase):
    """parallel_two_chains has two Conv→Relu→… chains rooted at X.
    Convs serialize on the Conv lane, but the second Conv (chain B) can
    start while chain A's Pool is still running."""

    def setUp(self):
        self.body = _inference_run_body(_gen_source("parallel_two_chains.onnx"))

    def test_pool_start_precedes_second_conv_with_no_pool_wait_between(self):
        """run_pool(... ca2 ...) must appear before run_conv(... cb0 ...),
        and there must NOT be a kernel_wait(KERNEL_POOL) between them — i.e.
        chain B's Conv launches while chain A's Pool is still in flight."""
        m_pool = re.search(r"run_pool\([^;]*ca2[^;]*\);", self.body, re.S)
        m_conv = re.search(r"run_conv\([^;]*Wb[^;]*\);", self.body, re.S)
        self.assertIsNotNone(m_pool, "ca2 pool start not found")
        self.assertIsNotNone(m_conv, "convB (Wb) start not found")
        self.assertLess(m_pool.start(), m_conv.start(),
                        "Pool must start before second Conv")
        between = self.body[m_pool.end():m_conv.start()]
        self.assertNotIn("kernel_wait(KERNEL_POOL)", between,
                         "Pool wait inserted prematurely — kills overlap")

    def test_pool_wait_only_at_join(self):
        """The single kernel_wait(KERNEL_POOL) must appear right before the
        join Add that consumes ca2 (the Pool output)."""
        pool_waits = [m.start() for m in re.finditer(
            r"kernel_wait\(KERNEL_POOL\);", self.body)]
        self.assertEqual(len(pool_waits), 1,
                         f"expected exactly 1 Pool wait, got {len(pool_waits)}")
        join_pos = re.search(r"run_op\([^;]*ca2[^;]*cb1[^;]*Y[^;]*\);",
                             self.body, re.S)
        self.assertIsNotNone(join_pos, "join Add not found")
        self.assertLess(pool_waits[0], join_pos.start(),
                        "Pool wait must precede the join")

    def test_no_blocking_polls_in_inference_run(self):
        """No raw IsDone polls leak into inference_run — all sync funnels
        through kernel_wait()."""
        self.assertNotIn("XVectoropkernel_IsDone", self.body)
        self.assertNotIn("XConvkernel_IsDone", self.body)
        self.assertNotIn("XPoolingkernel_IsDone", self.body)


# ---------------------------------------------------------------- #
# Sequential chains still drain at the end                          #
# ---------------------------------------------------------------- #


@unittest.skipUnless(_has_model("squeeze_then_matmul.onnx"),
                     "Run test/gen_reshape_gemm_models.py first")
class TestReshapeAliasWaitPassThrough(unittest.TestCase):
    """When a Reshape sits between a producer and a consumer, the consumer
    must still wait on the producer's hardware lane.  ReshapeNode has
    kernel_name="" so a naive predecessor walk would skip it and emit no
    wait; the event-stream walker must traverse Reshape chains."""

    def test_matmul_waits_for_pool_through_squeeze(self):
        body = _inference_run_body(_gen_source("squeeze_then_matmul.onnx"))
        m_pool = re.search(r"run_pool\([^;]*\);", body, re.S)
        m_wait = re.search(r"kernel_wait\(KERNEL_POOL\);", body)
        m_mm   = re.search(r"run_matmul\([^;]*\);", body, re.S)
        self.assertIsNotNone(m_pool, "Pool start not found")
        self.assertIsNotNone(m_wait, "POOL wait not emitted before MatMul "
                                     "— Reshape pass-through is broken")
        self.assertIsNotNone(m_mm, "MatMul start not found")
        # Order: Pool start  →  Pool wait  →  MatMul start.
        self.assertLess(m_pool.start(), m_wait.start())
        self.assertLess(m_wait.start(), m_mm.start())


@unittest.skipUnless(_has_model("parallel_two_chains.onnx"),
                     "Run test/gen_parallel_models.py first")
class TestEventTimelineLiveness(unittest.TestCase):
    """Buffer-slot colouring must use event-stream intervals, not
    node-index intervals.  In parallel_two_chains, ca1 (consumed by Pool)
    and cb0 (written by Conv-B) have OVERLAPPING execution intervals even
    though their node indices look disjoint — they MUST land in different
    pool slots."""

    def setUp(self):
        self.src = _gen_source("parallel_two_chains.onnx")

    def _slot_offset(self, c_name: str) -> int:
        m = re.search(
            rf"inference_buf_init_view\(&_s_buf_{c_name},"
            rf"\s*s_alloc_pool,\s*(\d+)u,",
            self.src,
        )
        self.assertIsNotNone(m, f"buffer view for {c_name} not found")
        return int(m.group(1))

    def test_ca1_and_cb0_have_distinct_slots(self):
        """ca1 is read by Pool which stays in flight while Conv-B writes
        cb0 in parallel — sharing a slot would corrupt Pool's input."""
        self.assertNotEqual(
            self._slot_offset("ca1"),
            self._slot_offset("cb0"),
            "ca1 and cb0 still alias the same pool slot — "
            "parallel-execution buffer-slot bug regressed",
        )

    def test_ca0_and_ca2_can_share_slot(self):
        """Sanity check: where the timelines genuinely don't overlap,
        coloring should still pack tensors.  ca0 ends before ca2 starts
        in event-stream space, so they share."""
        self.assertEqual(
            self._slot_offset("ca0"),
            self._slot_offset("ca2"),
            "non-overlapping intervals should still share slots",
        )


@unittest.skipUnless(_has_model("relu_chain.onnx"),
                     "Run test/gen_test_models.py first")
class TestSequentialChainDrain(unittest.TestCase):
    """For a strict chain, the last node's wait is the final drain
    (emitted right before sync_from_device)."""

    def test_final_drain_before_output_sync(self):
        body = _inference_run_body(_gen_source("relu_chain.onnx"))
        m_wait = list(re.finditer(r"kernel_wait\([^)]*\);", body))
        self.assertGreater(len(m_wait), 0, "expected at least one wait")
        m_sync = re.search(r"inference_buf_sync_from_device\(", body)
        self.assertIsNotNone(m_sync, "sync_from_device not found")
        # Final wait is before the final-output sync.
        self.assertLess(m_wait[-1].start(), m_sync.start())


if __name__ == "__main__":
    unittest.main()
