"""Activation fusion (VectorOPKernel `act` register) and the 128-bit port
alignment contract.

Models are built inline (onnx.helper) so the tests need no generator run.
"""

import os
import re
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as nph
from onnx import TensorProto

from src.graph   import OnnxGraph
from src.codegen import CodeGenerator
from src.nodes   import (ScheduledNode, ConvNode,
                         ACT_NONE, ACT_RELU, ACT_RELU6, OP_ADD, OP_MUL)
from src.report  import ReportGenerator

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)


def _f32(name, shape):
    return oh.make_tensor_value_info(name, TensorProto.FLOAT, shape)


def _init(name, arr):
    return nph.from_array(np.asarray(arr, dtype=np.float32), name=name)


def _save(nodes, inputs, outputs, initializers, name, path, value_info=()):
    graph = oh.make_graph(nodes, name, inputs, outputs,
                          initializer=initializers, value_info=list(value_info))
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return path


class _ModelDir(unittest.TestCase):
    """Builds the fixture models once per class into a temp dir."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        d = cls._tmp.name
        rng = np.random.default_rng(7)

        # X[1,128] + bias -> Relu -> * scale -> Clip(0,6) -> Y : two fusions
        N = 128
        cls.chain = _save(
            [oh.make_node("Add",  ["X", "bias"],  ["add_Y"]),
             oh.make_node("Relu", ["add_Y"],      ["relu_Y"]),
             oh.make_node("Mul",  ["relu_Y", "scale"], ["mul_Y"]),
             oh.make_node("Clip", ["mul_Y", "clip_min", "clip_max"], ["Y"])],
            [_f32("X", [1, N])], [_f32("Y", [1, N])],
            [_init("bias", rng.standard_normal((1, N)) * 0.5),
             _init("scale", rng.uniform(0.5, 2.0, (1, N))),
             _init("clip_min", np.float32(0.0)), _init("clip_max", np.float32(6.0))],
            "act_fusion_chain", os.path.join(d, "chain.onnx"))

        # add_Y is ALSO a graph output -> the Relu must stay a separate call
        cls.tap = _save(
            [oh.make_node("Add",  ["X", "bias"], ["add_Y"]),
             oh.make_node("Relu", ["add_Y"],     ["Y"])],
            [_f32("X", [1, N])], [_f32("add_Y", [1, N]), _f32("Y", [1, N])],
            [_init("bias", rng.standard_normal((1, N)) * 0.5)],
            "act_fusion_tap", os.path.join(d, "tap.onnx"))

        # add_Y has two consumers (Relu and a second Add) -> no fusion
        cls.shared = _save(
            [oh.make_node("Add",  ["X", "bias"],     ["add_Y"]),
             oh.make_node("Relu", ["add_Y"],         ["relu_Y"]),
             oh.make_node("Add",  ["add_Y", "relu_Y"], ["Y"])],
            [_f32("X", [1, N])], [_f32("Y", [1, N])],
            [_init("bias", rng.standard_normal((1, N)) * 0.5)],
            "act_fusion_shared", os.path.join(d, "shared.onnx"))

        # Relu on a graph input (no producer) then Add -> nothing to fuse into
        cls.relu_first = _save(
            [oh.make_node("Relu", ["X"],             ["relu_X"]),
             oh.make_node("Add",  ["relu_X", "bias"], ["Y"])],
            [_f32("X", [1, N])], [_f32("Y", [1, N])],
            [_init("bias", rng.standard_normal((1, N)) * 0.5)],
            "act_fusion_relu_first", os.path.join(d, "relu_first.onnx"))

        # Broadcast Add (chunk 12 over 5 rows -> stride 16) then Relu
        cls.bcast = _save(
            [oh.make_node("Add",  ["X", "bias"], ["add_Y"]),
             oh.make_node("Relu", ["add_Y"],     ["Y"])],
            [_f32("X", [5, 12])], [_f32("Y", [5, 12])],
            [_init("bias", rng.standard_normal((1, 12)) * 0.5)],
            "act_fusion_bcast", os.path.join(d, "bcast.onnx"))

        # Conv -> Relu: the producer is a ConvNode, which has no act register
        w = rng.standard_normal((4, 3, 3, 3)) * 0.1
        cls.conv = _save(
            [oh.make_node("Conv", ["X", "W"], ["conv_Y"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
             oh.make_node("Relu", ["conv_Y"], ["Y"])],
            [_f32("X", [1, 3, 8, 8])], [_f32("Y", [1, 4, 8, 8])],
            [_init("W", w)],
            "act_fusion_conv", os.path.join(d, "conv.onnx"))

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    @staticmethod
    def _gen(path, fuse_act):
        g  = OnnxGraph(path, fuse_act=fuse_act)
        cg = CodeGenerator(g, model_path=path)
        return g, cg

    @staticmethod
    def _run_op_calls(src):
        return [l.strip() for l in src.splitlines()
                if re.match(r"\s*run_op(_act)?\(", l)]


class TestActFusionChain(_ModelDir):

    def test_default_is_unfused(self):
        g = OnnxGraph(self.chain)
        self.assertEqual(len(g.nodes), 4)
        self.assertEqual(g.act_fused_count, 0)
        self.assertTrue(all(sn.act == ACT_NONE for sn in g.nodes))

    def test_two_activations_fold(self):
        g, _ = self._gen(self.chain, True)
        self.assertEqual(g.act_fused_count, 2)
        self.assertEqual([sn.onnx_node.op_type for sn in g.nodes], ["Add", "Mul"])
        self.assertEqual([sn.index for sn in g.nodes], [0, 1])
        add, mul = g.nodes
        self.assertEqual(add.act, ACT_RELU)
        self.assertEqual(mul.act, ACT_RELU6)
        # the fused node writes the activation's output tensor
        self.assertEqual(add.output.onnx_name, "relu_Y")
        self.assertEqual(mul.output.onnx_name, "Y")
        self.assertEqual(mul.inputs[0].onnx_name, "relu_Y")
        self.assertEqual([n.op_type for n in add.fused_nodes], ["Relu"])
        self.assertEqual([n.op_type for n in mul.fused_nodes], ["Clip"])

    def test_dropped_tensors_are_not_allocated(self):
        g, cg = self._gen(self.chain, True)
        names = {t.onnx_name for t in g.intermediate_tensors}
        self.assertNotIn("add_Y", names)
        self.assertNotIn("mul_Y", names)
        self.assertIn("relu_Y", names)
        self.assertNotIn("add_Y", cg.generate_header())

    def test_emitted_calls(self):
        _, cg = self._gen(self.chain, True)
        src = cg.generate_source()
        calls = self._run_op_calls(src)
        self.assertEqual(calls, [
            "run_op_act(X, bias, relu_Y, 128u, VECTOROP_ADD, 1u, 0u, 0u, VECTOROP_ACT_RELU);",
            "run_op_act(relu_Y, scale, Y, 128u, VECTOROP_MUL, 1u, 0u, 0u, VECTOROP_ACT_RELU6);",
        ])
        self.assertIn("#define VECTOROP_ACT_NONE    0u", src)
        self.assertIn("#define VECTOROP_ACT_RELU    1u", src)
        self.assertIn("#define VECTOROP_ACT_RELU6   2u", src)
        self.assertIn("static void run_op_act(", src)
        self.assertIn("XVectoropkernel_Set_act(&s_vectoropkernel, act);", src)
        # plain run_op() still writes the register (it keeps its last value)
        m = re.search(r"static void run_op\([^)]*\)\s*\{(.*?)\n\}", src, re.S)
        self.assertIsNotNone(m)
        self.assertIn("XVectoropkernel_Set_act(&s_vectoropkernel, VECTOROP_ACT_NONE);", m.group(1))
        self.assertIn("/* [0] Add + Relu(X, bias) -> relu_Y", src)

    def test_unfused_source_has_no_run_op_act(self):
        _, cg = self._gen(self.chain, False)
        src = cg.generate_source()
        self.assertNotIn("static void run_op_act(", src)
        calls = self._run_op_calls(src)
        self.assertEqual(len(calls), 4)
        self.assertTrue(all(c.startswith("run_op(") for c in calls))
        self.assertTrue(all(c.endswith("0u);") for c in calls))
        self.assertIn("XVectoropkernel_Set_act(&s_vectoropkernel, VECTOROP_ACT_NONE);", src)

    def test_simulation_matches_unfused(self):
        g0, cg0 = self._gen(self.chain, False)
        g1, cg1 = self._gen(self.chain, True)
        rng = np.random.default_rng(3)
        x = np.round(rng.uniform(-8, 8, (1, 128)) * 256) / 256
        y0 = cg0.simulate({"X": x})["Y"]
        y1 = cg1.simulate({"X": x})["Y"]
        np.testing.assert_array_equal(y0, y1)
        self.assertGreaterEqual(y1.min(), 0.0)
        self.assertLessEqual(y1.max(), 6.0)
        # the generated self-test carries the same expected values
        self.assertEqual(cg0.generate_test().count("expected"), cg1.generate_test().count("expected"))

    def test_report_lists_fusion(self):
        g, cg = self._gen(self.chain, True)
        md = ReportGenerator(graph=g, codegen=cg, model_path=self.chain,
                             out_dir="/tmp", generated_files=[]).render_markdown()
        self.assertIn("Activation fusion", md)
        self.assertIn("2 `Relu` / `Clip(0,6)` nodes folded", md)
        self.assertIn("+ Relu (act)", md)
        self.assertIn("+ Clip (act)", md)


class TestActFusionBlocked(_ModelDir):

    def test_graph_output_between_blocks_fusion(self):
        g, cg = self._gen(self.tap, True)
        self.assertEqual(g.act_fused_count, 0)
        self.assertEqual(len(g.nodes), 2)
        self.assertNotIn("static void run_op_act(", cg.generate_source())

    def test_second_consumer_blocks_fusion(self):
        g, cg = self._gen(self.shared, True)
        self.assertEqual(g.act_fused_count, 0)
        self.assertEqual([sn.onnx_node.op_type for sn in g.nodes], ["Add", "Relu", "Add"])
        self.assertNotIn("static void run_op_act(", cg.generate_source())

    def test_relu_on_graph_input_stays(self):
        g, cg = self._gen(self.relu_first, True)
        self.assertEqual(g.act_fused_count, 0)
        src = cg.generate_source()
        self.assertIn("run_op(X, NULL, relu_X, 128u, VECTOROP_RELU, 1u, 0u, 0u);", src)

    def test_conv_producer_is_not_fused(self):
        g, cg = self._gen(self.conv, True)
        self.assertEqual(g.act_fused_count, 0)
        self.assertIsInstance(g.nodes[0], ConvNode)
        self.assertIsInstance(g.nodes[1], ScheduledNode)
        self.assertIn("VECTOROP_RELU", cg.generate_source())


class TestActFusionBroadcast(_ModelDir):

    def test_broadcast_add_fuses_relu(self):
        g, cg = self._gen(self.bcast, True)
        self.assertEqual(g.act_fused_count, 1)
        sn = g.nodes[0]
        self.assertEqual(sn.op_code, OP_ADD)
        self.assertEqual(sn.act, ACT_RELU)
        self.assertEqual(sn.outer_count, 5)
        self.assertEqual(sn.chunk_size, 12)
        self.assertEqual(sn.aligned_chunk_size, 16)
        src = cg.generate_source()
        self.assertIn(
            "run_op_act(X, bias, Y, INFERENCE_Y_CHUNK, VECTOROP_ADD, 5u, "
            "INFERENCE_Y_CHUNK_STRIDE, 0u, VECTOROP_ACT_RELU);", src)
        # the stride macro follows the (renamed) output tensor
        hdr = cg.generate_header()
        self.assertIn("#define INFERENCE_Y_CHUNK", hdr)
        self.assertNotIn("INFERENCE_ADD_Y_CHUNK", hdr)

    def test_broadcast_simulation_matches(self):
        _, cg0 = self._gen(self.bcast, False)
        _, cg1 = self._gen(self.bcast, True)
        x = np.round(np.random.default_rng(5).uniform(-4, 4, (5, 12)) * 256) / 256
        np.testing.assert_array_equal(cg0.simulate({"X": x})["Y"], cg1.simulate({"X": x})["Y"])


class TestAlignmentContract(_ModelDir):
    """What the 128-bit VectorOPKernel ports rely on (VectorOP.h)."""

    def test_chunk_stride_is_a_whole_number_of_words(self):
        for fuse in (False, True):
            g, cg = self._gen(self.bcast, fuse)
            for sn in g.nodes:
                if isinstance(sn, ScheduledNode) and sn.outer_count > 1:
                    self.assertEqual(sn.aligned_chunk_size % 8, 0)
                    self.assertGreaterEqual(sn.aligned_chunk_size, sn.chunk_size)
            hdr = cg.generate_header()
            self.assertIn("#define INFERENCE_ALIGN_BYTES  16u", hdr)
            self.assertRegex(
                hdr, r"INFERENCE_\w+_CHUNK_STRIDE\s+INFERENCE_ALIGN_UP\(INFERENCE_\w+_CHUNK\)")
            for lay in cg._layouts.values():
                if lay.n_chunks > 1:
                    self.assertEqual(lay.stride % 8, 0)
                    self.assertEqual(lay.alloc, lay.n_chunks * lay.stride)

    def test_buffers_are_allocated_in_64_byte_multiples(self):
        # The kernel writes the last 16-byte word of every run whole, so an
        # allocation must extend to the next word past its element count;
        # both allocators round to 64 bytes (also the base alignment).
        _, cg = self._gen(self.chain, True)
        impl = cg.generate_buf_impl()
        self.assertEqual(
            impl.count("INFERENCE_BYTES_PER_ELEM + 63u) & ~(size_t)63u"), 2)

    def test_non_broadcast_size_covers_strided_buffer(self):
        # A non-broadcast consumer of a strided buffer runs over alloc
        # (n_chunks * stride), a whole number of words.
        g, cg = self._gen(self.bcast, False)
        relu = g.nodes[1]
        lay = cg._layouts[relu.output.onnx_name]
        self.assertEqual(lay.alloc, 5 * 16)
        self.assertIn("run_op(add_Y, NULL, Y, 80u, VECTOROP_RELU, 1u, 0u, 0u);",
                      cg.generate_source())


class TestActFusionCli(_ModelDir):

    def _cli(self, model, extra):
        with tempfile.TemporaryDirectory() as td:
            cmd = [sys.executable, os.path.join(_ROOT, "inference_scheduler.py"),
                   model, "--out-dir", td, "--no-report"] + extra
            r = subprocess.run(cmd, capture_output=True, text=True, cwd=_ROOT)
            self.assertEqual(r.returncode, 0, r.stderr)
            with open(os.path.join(td, "src", "inference.c")) as f:
                return f.read()

    def test_cli_fuses_by_default(self):
        src = self._cli(self.chain, [])
        self.assertIn("run_op_act(X, bias, relu_Y, 128u, VECTOROP_ADD, 1u, 0u, 0u, VECTOROP_ACT_RELU);", src)

    def test_cli_no_fuse_act(self):
        src = self._cli(self.chain, ["--no-fuse-act"])
        self.assertNotIn("static void run_op_act(", src)
        self.assertIn("run_op(add_Y, NULL, relu_Y, 128u, VECTOROP_RELU, 1u, 0u, 0u);", src)


if __name__ == "__main__":
    unittest.main()
