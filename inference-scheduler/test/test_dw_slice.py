"""Depthwise channel slices (OnnxGraph(dw_slice=...), cost_model.dw_slice,
doc/scheduler/INFERENCE_SCHEDULER.md §Depthwise channel slices): a batch-1
depthwise Conv issued as several ConvKernel calls of 16 / 32 / 64 channels
through run_conv_dw_at() — the choice, the emitted calls, kernel_calls(),
and the project compiled on the host, bit-exact with the simulation."""

import re
import shutil
import tempfile
import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

import host_emu
from src import cost_model
from src.codegen import CodeGenerator
from src.graph import OnnxGraph
from src.nodes import ConvNode


def dw_model(c=40, h=24, w=30, batch=1, seed=0):
    """x -> dw 3x3 (pad 1, bias) -> dw 3x3 stride 2 dilation 1 (pad 1, no bias) -> y."""
    rng = np.random.default_rng(seed)
    w1 = (rng.standard_normal((c, 1, 3, 3)) * 0.3).astype(np.float32)
    b1 = (rng.standard_normal(c) * 0.5).astype(np.float32)
    w2 = (rng.standard_normal((c, 1, 3, 3)) * 0.3).astype(np.float32)
    oh, ow = (h - 1) // 2 + 1, (w - 1) // 2 + 1
    nodes = [
        helper.make_node("Conv", ["x", "w1", "b1"], ["t"], group=c, kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        helper.make_node("Conv", ["t", "w2"], ["y"], group=c, kernel_shape=[3, 3], pads=[1, 1, 1, 1],
                         strides=[2, 2]),
    ]
    g = helper.make_graph(
        nodes, "dw_slices",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [batch, c, h, w])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [batch, c, oh, ow])],
        initializer=[numpy_helper.from_array(w1, "w1"), numpy_helper.from_array(b1, "b1"),
                     numpy_helper.from_array(w2, "w2")])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    onnx.checker.check_model(m)
    return m


def convs(g):
    return [sn for sn in g.nodes if isinstance(sn, ConvNode)]


class TestChoice(unittest.TestCase):
    """cost_model.dw_slice on shapes measured on the board (8599aa7a5f12)."""

    def test_tall_windows_slice(self):
        # 144 ch on 120 x 160: 10.26 ms in one call, 9 x 0.635 ms as 16-channel calls
        self.assertEqual(cost_model.dw_slice(144, 120, 160, 120, 160, 3, 3, 1, 1, 1, 1, 1, 1), 16)
        # a 2x1 stripe piece dilated by 13: 1.69 ms against 6 x 0.150
        self.assertEqual(cost_model.dw_slice(96, 60, 80, 60, 80, 2, 1, 1, 1, 13, 1, 9, 0), 16)

    def test_no_row_halo_keeps_one_call(self):
        # a 1x7 stripe: 0.715 ms in one call, 6 x 0.142 as 16-channel calls
        self.assertEqual(cost_model.dw_slice(96, 60, 80, 60, 80, 1, 7, 1, 1, 1, 1, 0, 3), 96)

    def test_few_channels(self):
        self.assertEqual(cost_model.dw_slice(16, 120, 160, 120, 160, 3, 3, 1, 1, 1, 1, 1, 1), 16)

    def test_hls_kernel_never_sliced(self):
        old = cost_model.CONV_IMPL
        try:
            cost_model.CONV_IMPL = "hls"
            cost_model.dw_slice.cache_clear()
            self.assertEqual(cost_model.dw_slice(144, 120, 160, 120, 160, 3, 3, 1, 1, 1, 1, 1, 1), 144)
        finally:
            cost_model.CONV_IMPL = old
            cost_model.dw_slice.cache_clear()


class TestGraph(unittest.TestCase):

    def test_forced_slices(self):
        g = OnnxGraph(dw_model(), dw_slice=16)
        a, b = convs(g)
        self.assertTrue(a.sliced and b.sliced)
        self.assertEqual((a.dw_slice, a.slice_calls()), (16, 3))
        self.assertEqual(g.dw_slice_stats, {"sliced": 2, "kept": 0, "calls": 6})

    def test_off(self):
        g = OnnxGraph(dw_model(), dw_slice="off")
        self.assertFalse(any(sn.sliced for sn in convs(g)))
        self.assertEqual(g.dw_slice_stats, {"sliced": 0, "kept": 2, "calls": 0})
        src = CodeGenerator(g, model_path="m.onnx").generate_source()
        self.assertIn("run_conv(", src)
        self.assertNotIn("run_conv_dw_at", src)

    def test_auto_small_maps_keep_one_call(self):
        g = OnnxGraph(dw_model())                    # 24 x 30: one chunk either way
        self.assertFalse(any(sn.sliced for sn in convs(g)))

    def test_auto_large_map_slices(self):
        g = OnnxGraph(dw_model(c=144, h=120, w=160))      # the shape measured above
        self.assertEqual(convs(g)[0].dw_slice, 16)

    def test_batch_two_not_sliced(self):
        g = OnnxGraph(dw_model(batch=2), dw_slice=16)
        self.assertFalse(any(sn.sliced for sn in convs(g)))

    def test_bad_value(self):
        for v in (8, 24, "always", -16):
            with self.assertRaises(ValueError):
                OnnxGraph(dw_model(), dw_slice=v)


class TestEmit(unittest.TestCase):

    def setUp(self):
        self.g = OnnxGraph(dw_model(), dw_slice=16)
        self.src = CodeGenerator(self.g, model_path="m.onnx").generate_source()

    def test_helper_and_loop(self):
        self.assertIn("static void run_conv_dw_at(", self.src)
        self.assertNotIn("static void run_conv(", self.src)      # every conv is sliced
        self.assertEqual(self.src.count("for (unsigned _c = 0u; _c < 40u; _c += 16u)"), 2)
        self.assertIn("if (_c) kernel_wait(KERNEL_CONV);", self.src)
        # offsets: x per channel in_h*in_w, weights roundup(9, 8) = 16, y out_h*out_w
        self.assertIn("_c * 720u,", self.src)                    # 24 x 30
        self.assertIn("_c * 16u,", self.src)
        self.assertIn("_c * 180u,", self.src)                    # 12 x 15
        self.assertIn("(40u - _c < 16u) ? 40u - _c : 16u", self.src)

    def test_kernel_calls_match_the_loop(self):
        a = convs(self.g)[0]
        calls = a.kernel_calls({})
        self.assertEqual([(c.count, c.fields["in_ch"], c.fields["out_ch"]) for c in calls],
                         [(2, 16, 16), (1, 8, 8)])
        self.assertTrue(all(c.fields["is_dw"] == 1 and c.fields["batch"] == 1 for c in calls))

    def test_comment(self):
        self.assertTrue(re.search(r"dw, 3 calls of 16 channels \*/", self.src))


class TestHostEmu(unittest.TestCase):
    """The generated project on the host's software ConvKernel, bit-exact with
    the simulation (test_inference compares every element)."""

    def run_emu(self, g):
        if host_emu.which_cc() is None:
            self.fail("no C compiler")
        cg = CodeGenerator(g, model_path="dw.onnx")
        d = tempfile.mkdtemp(prefix="dw_slice_emu_")
        try:
            rc, out = host_emu.build_and_run(cg, d)
            self.assertEqual(rc, 0, out[-3000:])
            self.assertEqual(host_emu.failures(out), [], out[-3000:])
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_sliced_16(self):
        self.run_emu(OnnxGraph(dw_model(), dw_slice=16))

    def test_sliced_32(self):
        self.run_emu(OnnxGraph(dw_model(c=72, seed=1), dw_slice=32))

    def test_auto_large_map(self):
        g = OnnxGraph(dw_model(c=96, h=64, w=160, seed=2))   # 32-channel calls
        self.assertEqual(convs(g)[0].dw_slice, 32)
        self.run_emu(g)


if __name__ == "__main__":
    unittest.main()
