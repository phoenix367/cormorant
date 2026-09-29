"""Fully-connected Convs -> MatMul (src/fc_conv.py, doc/plans/LENET_PLAN.md).

A Conv whose kernel covers its whole unpadded input (one output pixel) is
rewritten as Flatten + MatMul + Reshape (+ bias Add); the fixed-point result
is bit-identical to the Conv path.  Models are built inline (onnx.helper).
"""

import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as nph
from onnx import TensorProto

from src.codegen import CodeGenerator
from src.fc_conv import candidate, estimate, lower_fc_convs
from src.graph import OnnxGraph
from src.nodes import ConvNode, MatmulNode, SchedulerError

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _f32(name, shape):
    return oh.make_tensor_value_info(name, TensorProto.FLOAT, shape)


def _grid(rng, shape, scale):
    """Random values on the ap_fixed<16,8> grid (multiples of 1/256)."""
    return np.round(rng.uniform(-scale, scale, shape) * 256) / 256


def _save(path, nodes, inputs, outputs, inits):
    g = oh.make_graph(nodes, os.path.basename(path), inputs, outputs,
                      initializer=[nph.from_array(np.asarray(v, np.float32), name=k)
                                   for k, v in inits.items()])
    m = oh.make_model(g, opset_imports=[oh.make_opsetid("", 13)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    onnx.save(m, path)
    return path


def _fc_model(path, C, H, W, M, *, bias=True, relu=True, head=None, pads=None,
              group=1, dilation=1, kernel=None, seed=1, wscale=0.05):
    """X[1,C,H,W] -> Conv(kernel H x W) [-> Relu] [-> Conv 1x1 to `head`]."""
    rng = np.random.default_rng(seed)
    kh, kw = kernel or (H, W)
    inits = {"W": _grid(rng, (M, C // group, kh, kw), wscale)}
    conv_in = ["X", "W"]
    if bias:
        inits["B"] = _grid(rng, (M,), 0.5)
        conv_in.append("B")
    attrs = dict(kernel_shape=[kh, kw], group=group, dilations=[dilation, dilation])
    if pads:
        attrs["pads"] = pads
    oh_ = H + (pads[0] + pads[2] if pads else 0) - dilation * (kh - 1)
    ow_ = W + (pads[1] + pads[3] if pads else 0) - dilation * (kw - 1)
    nodes = [oh.make_node("Conv", conv_in, ["c1"], name="fc1", **attrs)]
    last, shape = "c1", [1, M, oh_, ow_]
    if relu:
        nodes.append(oh.make_node("Relu", ["c1"], ["r1"]))
        last = "r1"
    if head:
        inits["W2"] = _grid(rng, (head, M, 1, 1), wscale)
        inits["B2"] = _grid(rng, (head,), 0.5)
        nodes.append(oh.make_node("Conv", [last, "W2", "B2"], ["c2"], name="fc2", kernel_shape=[1, 1]))
        last, shape = "c2", [1, head, 1, 1]
    nodes.append(oh.make_node("Identity", [last], ["Y"]))
    return _save(path, nodes, [_f32("X", [1, C, H, W])], [_f32("Y", shape)], inits)


class _Models(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        j = lambda n: os.path.join(cls._tmp.name, n)  # noqa: E731
        # LeNet's conv3 + conv4last geometry, scaled to 16 channels
        cls.small = _fc_model(j("small.onnx"), 16, 7, 7, 64, head=16)
        cls.nobias = _fc_model(j("nobias.onnx"), 8, 4, 4, 32, bias=False, relu=False)
        # LeNet's conv3 itself: auto lowers it (GEMV)
        cls.lenet3 = _fc_model(j("lenet3.onnx"), 64, 7, 7, 1024, relu=True)
        # MobileNet v1's classifier (1x1 on 1x1, 1001 outputs): not GEMV-eligible,
        # the tiled MatMul is not faster -> auto keeps the Conv
        cls.mnv1 = _fc_model(j("mnv1.onnx"), 1024, 1, 1, 1001, relu=False)
        # not fully connected / not eligible
        cls.padded = _fc_model(j("padded.onnx"), 8, 4, 4, 16, pads=[1, 1, 1, 1], kernel=(6, 6))
        cls.two_px = _fc_model(j("two_px.onnx"), 8, 4, 5, 16, kernel=(4, 4))
        cls.grouped = _fc_model(j("grouped.onnx"), 8, 4, 4, 8, group=8)    # depthwise
        cls.big_k = _fc_model(j("big_k.onnx"), 128, 6, 6, 16)        # K = 4608 > max_k

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    @staticmethod
    def graph(path, mode="always", **kw):
        return OnnxGraph(path, fc_conv=mode, **kw)

    @staticmethod
    def kinds(g):
        return [type(sn).__name__ for sn in g.nodes]


class TestRewrite(_Models):
    def test_nodes(self):
        g = self.graph(self.small)
        self.assertEqual(g.fc_conv_stats["lowered"], 2)
        self.assertFalse(any(isinstance(sn, ConvNode) for sn in g.nodes))
        self.assertEqual(sum(isinstance(sn, MatmulNode) for sn in g.nodes), 2)
        ops = [sn.onnx_node.op_type for sn in g.nodes]
        self.assertEqual(ops.count("Flatten"), 2)
        self.assertEqual(ops.count("Add"), 2)          # the two biases
        mm = [sn for sn in g.nodes if isinstance(sn, MatmulNode)]
        self.assertEqual([(sn.n, sn.k, sn.m) for sn in mm], [(1, 16 * 49, 64), (1, 64, 16)])

    def test_no_bias(self):
        g = self.graph(self.nobias)
        ops = [sn.onnx_node.op_type for sn in g.nodes]
        self.assertIn("MatMul", ops)
        self.assertNotIn("Add", ops)

    def test_original_weight_dropped(self):
        g = self.graph(self.small)
        names = {t.onnx_name for t in g.weight_tensors}
        self.assertNotIn("W", names)
        self.assertIn("W_fc", names)

    def test_relu_fuses_into_bias_add(self):
        g = self.graph(self.small, fuse_act=True)
        self.assertEqual(g.act_fused_count, 1)
        self.assertNotIn("Relu", [sn.onnx_node.op_type for sn in g.nodes])

    def test_not_applicable(self):
        for path in (self.padded, self.two_px, self.grouped, self.big_k):
            g = self.graph(path)
            self.assertEqual(g.fc_conv_stats["lowered"], 0, path)
            self.assertTrue(any(isinstance(sn, ConvNode) for sn in g.nodes), path)

    def test_off(self):
        g = self.graph(self.small, "off")
        self.assertEqual(sum(isinstance(sn, ConvNode) for sn in g.nodes), 2)
        self.assertEqual(g.fc_conv_stats["lowered"], 0)

    def test_bad_mode(self):
        with self.assertRaises(SchedulerError):
            self.graph(self.small, "sometimes")


class TestAuto(_Models):
    def test_lenet_conv3_lowered_to_gemv(self):
        g = self.graph(self.lenet3, "auto")
        self.assertEqual(g.fc_conv_stats["lowered"], 1)
        self.assertLess(g.fc_conv_stats["matmul_cycles"], 0.6 * g.fc_conv_stats["conv_cycles"])
        self.assertEqual(g.matmul_gemv_stats["gemv"], 1)

    def test_mobilenet_classifier_kept(self):
        g = self.graph(self.mnv1, "auto")
        self.assertEqual((g.fc_conv_stats["lowered"], g.fc_conv_stats["kept"]), (0, 1))
        m = onnx.shape_inference.infer_shapes(onnx.load(self.mnv1))
        shapes = {vi.name: [d.dim_value for d in vi.type.tensor_type.shape.dim]
                  for vi in list(m.graph.input) + list(m.graph.value_info) + list(m.graph.output)}
        inits = {i.name: i for i in m.graph.initializer}
        est = estimate(candidate(m.graph.node[0], shapes, inits))
        self.assertFalse(est["gemv"])

    def test_library_default_is_auto(self):
        self.assertEqual(OnnxGraph(self.lenet3).fc_conv_stats["lowered"], 1)


class TestNumerics(_Models):
    def _check(self, path, shape, **kw):
        x = _grid(np.random.default_rng(7), shape, 1.0)
        y0 = CodeGenerator(g0 := self.graph(path, "off", **kw), model_path=path).simulate({"X": x})["Y"]
        y1 = CodeGenerator(g1 := self.graph(path, "always", **kw), model_path=path).simulate({"X": x})["Y"]
        self.assertGreater(g1.fc_conv_stats["lowered"], 0)
        self.assertEqual(g0.fc_conv_stats["lowered"], 0)
        np.testing.assert_array_equal(y1, y0)
        self.assertGreater(np.count_nonzero(y1), 0)

    def test_bit_identical(self):
        self._check(self.small, (1, 16, 7, 7))

    def test_bit_identical_fused_act(self):
        self._check(self.small, (1, 16, 7, 7), fuse_act=True)

    def test_bit_identical_no_bias(self):
        self._check(self.nobias, (1, 8, 4, 4))

    def test_lenet3_gemv(self):
        self._check(self.lenet3, (1, 64, 7, 7))

    def test_float_semantics(self):
        """The rewritten ONNX graph computes the Conv (onnxruntime-free check)."""
        m, st = lower_fc_convs(onnx.shape_inference.infer_shapes(onnx.load(self.nobias)), "always")
        self.assertEqual(st["lowered"], 1)
        w = nph.to_array(next(i for i in m.graph.initializer if i.name == "W_fc"))
        w0 = nph.to_array(next(i for i in onnx.load(self.nobias).graph.initializer if i.name == "W"))
        x = np.random.default_rng(3).normal(size=(1, 8, 4, 4))
        conv = np.einsum("nchw,mchw->nm", x, w0)
        np.testing.assert_allclose(x.reshape(1, -1) @ w, conv, rtol=1e-6)


class TestCli(_Models):
    def _src(self, extra):
        with tempfile.TemporaryDirectory() as td:
            r = subprocess.run([sys.executable, os.path.join(_ROOT, "inference_scheduler.py"),
                                self.lenet3, "--out-dir", td, "--no-report"] + extra,
                               capture_output=True, text=True, cwd=_ROOT)
            self.assertEqual(r.returncode, 0, r.stderr)
            with open(os.path.join(td, "src", "inference.c")) as f:
                return f.read()

    def test_default_auto(self):
        src = self._src([])
        self.assertIn("run_matmul", src)
        self.assertNotIn("run_conv", src)

    def test_off(self):
        src = self._src(["--fc-conv", "off"])
        self.assertIn("run_conv", src)
        self.assertNotIn("run_matmul", src)


if __name__ == "__main__":
    unittest.main()
