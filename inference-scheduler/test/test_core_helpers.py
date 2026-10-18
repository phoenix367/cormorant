# Copyright 2025 GradeBuilder SL
# SPDX-License-Identifier: Apache-2.0

"""Coverage tests for src.codegen._core layout phases, validation
errors, and helper-property edge cases.

Targets uncovered lines 147-149, 157-158, 196, 228, 241, 744-745, 788
in `_CoreMixin`.  Programmatic ONNX construction is used so the tests
don't depend on pre-generated fixture models.
"""

import os
import tempfile
import unittest

import numpy as np
import onnx
import onnx.helper as oh
from onnx import TensorProto, numpy_helper

from src.codegen import CodeGenerator
from src.graph   import OnnxGraph, SchedulerError


# ---------------------------------------------------------------- #
# Helper                                                            #
# ---------------------------------------------------------------- #

def _save_model(graph, opset=13):
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", opset)])
    model.ir_version = 7
    fd, path = tempfile.mkstemp(suffix=".onnx")
    os.close(fd)
    onnx.save(model, path)
    return path


# ================================================================ #
# Layout Phase 2 — repeating-input replace branch                   #
# ================================================================ #

class TestLayoutBroadcastReplace(unittest.TestCase):
    """A=[1, 4] broadcasts across B=[8, 4].  outer_count=8, chunk=4,
    stride=align_up(4, 8)=8, so the broadcaster's alloc gets bumped
    from numel=4 → 8 via the repeating-layout replace branch."""

    def _gen(self, a_shape, b_shape, y_shape):
        A = oh.make_tensor_value_info("A", TensorProto.FLOAT, list(a_shape))
        B = oh.make_tensor_value_info("B", TensorProto.FLOAT, list(b_shape))
        Y = oh.make_tensor_value_info("Y", TensorProto.FLOAT, list(y_shape))
        node  = oh.make_node("Add", ["A", "B"], ["Y"], name="add")
        graph = oh.make_graph([node], "g", [A, B], [Y])
        path  = _save_model(graph)
        try:
            g = OnnxGraph(path)
            return CodeGenerator(g, model_path=path)
        finally:
            os.unlink(path)

    def test_a_broadcast_input_alloc_bumped(self):
        # Hits lines 147–149.
        cg = self._gen([1, 4], [8, 4], [8, 4])
        self.assertEqual(cg._layouts["A"].alloc,    8)
        self.assertEqual(cg._layouts["A"].n_chunks, 1)

    def test_b_broadcast_input_alloc_bumped(self):
        # Hits lines 157–158 (mirror of the above for input b).
        cg = self._gen([8, 4], [1, 4], [8, 4])
        self.assertEqual(cg._layouts["B"].alloc,    8)
        self.assertEqual(cg._layouts["B"].n_chunks, 1)


# ================================================================ #
# Layout Phase 3 — output bumped to flat alloc (no dominant)        #
# ================================================================ #

class TestLayoutPhase3NoDominantFallback(unittest.TestCase):
    """Graph:
       Add(A[1,4], B[8,4]) → AB[8,4]   - A becomes repeating, alloc=8
       Relu(A)              → C[1,4]    - inherits A's larger alloc as flat

    Phase 3 sees Relu's input A with alloc=8 > C.numel=4, but A is
    repeating (n_chunks=1) so there is no dominant input → fallback
    branch at line 196 fires.
    """

    def test_relu_of_repeating_input_inherits_flat_alloc(self):
        A  = oh.make_tensor_value_info("A",  TensorProto.FLOAT, [1, 4])
        B  = oh.make_tensor_value_info("B",  TensorProto.FLOAT, [8, 4])
        AB = oh.make_tensor_value_info("AB", TensorProto.FLOAT, [8, 4])
        C  = oh.make_tensor_value_info("C",  TensorProto.FLOAT, [1, 4])

        add_node  = oh.make_node("Add",  ["A", "B"], ["AB"], name="add")
        relu_node = oh.make_node("Relu", ["A"],      ["C"],  name="relu")
        graph = oh.make_graph(
            [add_node, relu_node], "g",
            [A, B], [AB, C],
        )
        path = _save_model(graph)
        try:
            g  = OnnxGraph(path)
            cg = CodeGenerator(g, model_path=path)
            self.assertEqual(cg._layouts["A"].alloc, 8)
            self.assertEqual(cg._layouts["C"].alloc, 8)
            self.assertEqual(cg._layouts["C"].n_chunks, 1)
        finally:
            os.unlink(path)


# ================================================================ #
# Validation — Conv / Pool feeding a broadcast VectorOP             #
# ================================================================ #

class TestConvIntoBroadcastRaises(unittest.TestCase):
    def test_conv_followed_by_broadcast_add_raises(self):
        # batch=2 lets the bias broadcast over the leading batch dim
        # (a [8,6,6] bias right-aligns to [1,8,6,6] inside [2,8,6,6]).
        # This satisfies the contiguous-leading-block rule and produces
        # outer_count=2 for Add → the Conv → broadcast validation fires.
        X = oh.make_tensor_value_info("X", TensorProto.FLOAT, [2, 4, 8, 8])
        Y = oh.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 8, 6, 6])
        W = numpy_helper.from_array(
            np.zeros((8, 4, 3, 3), dtype=np.float32), name="W")
        bias = numpy_helper.from_array(
            np.zeros((8, 6, 6), dtype=np.float32), name="bias")
        conv_out = oh.make_tensor_value_info(
            "conv_out", TensorProto.FLOAT, [2, 8, 6, 6])

        conv_node = oh.make_node("Conv", ["X", "W"], ["conv_out"],
                                  name="conv", kernel_shape=[3, 3])
        add_node  = oh.make_node("Add", ["conv_out", "bias"], ["Y"],
                                  name="add")
        graph = oh.make_graph(
            [conv_node, add_node], "g", [X], [Y],
            initializer=[W, bias], value_info=[conv_out],
        )
        path = _save_model(graph)
        try:
            with self.assertRaisesRegex(SchedulerError,
                                        "Conv node.*broadcast VectorOP"):
                g = OnnxGraph(path)
                CodeGenerator(g, model_path=path)
        finally:
            os.unlink(path)


class TestPoolIntoBroadcastRaises(unittest.TestCase):
    def test_pool_followed_by_broadcast_add_raises(self):
        # batch=2 — same trick as the Conv test.  bias=[4,4,4] right-aligns
        # to [1,4,4,4] inside [2,4,4,4]; outer_count=2 for the Add.
        X = oh.make_tensor_value_info("X", TensorProto.FLOAT, [2, 4, 8, 8])
        Y = oh.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4, 4, 4])
        bias = numpy_helper.from_array(
            np.zeros((4, 4, 4), dtype=np.float32), name="bias")
        pool_out = oh.make_tensor_value_info(
            "pool_out", TensorProto.FLOAT, [2, 4, 4, 4])

        pool_node = oh.make_node("MaxPool", ["X"], ["pool_out"], name="pool",
                                  kernel_shape=[2, 2], strides=[2, 2])
        add_node  = oh.make_node("Add", ["pool_out", "bias"], ["Y"],
                                  name="add")
        graph = oh.make_graph(
            [pool_node, add_node], "g", [X], [Y],
            initializer=[bias], value_info=[pool_out],
        )
        path = _save_model(graph)
        try:
            with self.assertRaisesRegex(SchedulerError,
                                        "Pool node.*broadcast VectorOP"):
                g = OnnxGraph(path)
                CodeGenerator(g, model_path=path)
        finally:
            os.unlink(path)


# ================================================================ #
# _driver_prefix fallback when no kernel nodes are present          #
# ================================================================ #

class TestDriverPrefixFallback(unittest.TestCase):
    def test_dropout_only_graph_returns_default_prefix(self):
        # Dropout maps to ReshapeNode (no kernel) → _active_kernels=[].
        X    = oh.make_tensor_value_info("X", TensorProto.FLOAT, [8])
        Y    = oh.make_tensor_value_info("Y", TensorProto.FLOAT, [8])
        node = oh.make_node("Dropout", ["X"], ["Y"], name="d")
        graph = oh.make_graph([node], "g", [X], [Y])
        path  = _save_model(graph)
        try:
            g  = OnnxGraph(path)
            cg = CodeGenerator(g, model_path=path)
            self.assertEqual(cg._driver_prefix, "xvectoropkernel")
            self.assertEqual(cg._active_kernels, [])
        finally:
            os.unlink(path)


# ================================================================ #
# _broadcast_io_map — b_advances=True branch                        #
# ================================================================ #

class TestBroadcastIoMapBAdvances(unittest.TestCase):
    """A=[1,4], B=[8,4] → b_advances=True, a_advances=False.  Line 788
    sets the canonical entry for input B."""

    def test_b_advances_canonical_set(self):
        A = oh.make_tensor_value_info("A", TensorProto.FLOAT, [1, 4])
        B = oh.make_tensor_value_info("B", TensorProto.FLOAT, [8, 4])
        Y = oh.make_tensor_value_info("Y", TensorProto.FLOAT, [8, 4])
        node  = oh.make_node("Add", ["A", "B"], ["Y"], name="add")
        graph = oh.make_graph([node], "g", [A, B], [Y])
        path  = _save_model(graph)
        try:
            g  = OnnxGraph(path)
            cg = CodeGenerator(g, model_path=path)
            iomap = cg._broadcast_io_map()
            # Y (output) and B (advancing) appear with n_chunks > 1.
            self.assertIn("Y", iomap)
            self.assertIn("B", iomap)
            # A (broadcaster, n_chunks=1) is excluded from the result map.
            self.assertNotIn("A", iomap)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
