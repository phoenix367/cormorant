# Copyright 2026 Ivan Gubochkin
# SPDX-License-Identifier: Apache-2.0

"""The 8-bit configuration of an ONNX CNN (demo/image_classification/scripts/onnx_i8.py;
doc/plans/INT8_PLAN.md §9.7) on a small random graph — a 3 x 3 Conv + Relu, a 1 x 1 Conv + Relu,
a depthwise stride-2 Conv + Clip(0, 6), a 1 x 1 Conv, the Add, GlobalAveragePool, Flatten, Gemm:
the float64 evaluator equals onnxruntime; the calibration gives every tensor a format (unsigned
after the activations, blocks where only Convs read a Conv output, per channel where only a depthwise
Conv reads it); every kernel output of the
8-bit run equals the exact dyadic value of its operands (the dequantised int8 weights, the bias
on the accumulator's grid, the fused activation) rounded into its format — the static rule, and
the block rule for the blocked tensors."""

import os
import sys
import unittest

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "demo", "image_classification", "scripts"))
import onnx_i8 as oi  # noqa: E402
from src import int8_spec as s8  # noqa: E402


def build(path):
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    rng = np.random.default_rng(3)

    def init(name, shape, scale):
        return numpy_helper.from_array((rng.normal(size=shape) * scale).astype(np.float32), name)
    inits = [init("w1", (8, 3, 3, 3), 0.4), init("b1", (8,), 0.1),
             init("w2", (40, 8, 1, 1), 0.3), init("b2", (40,), 0.1),
             init("w3", (40, 1, 3, 3), 0.5), init("b3", (40,), 0.1),
             init("w4", (40, 40, 1, 1), 0.2), init("b4", (40,), 0.05),
             init("w5", (10, 40), 0.3), init("b5", (10,), 0.1),
             numpy_helper.from_array(np.array(0.0, np.float32), "lo"),
             numpy_helper.from_array(np.array(6.0, np.float32), "hi")]
    N = helper.make_node
    nodes = [N("Conv", ["x", "w1", "b1"], ["c1"], pads=[1, 1, 1, 1]), N("Relu", ["c1"], ["r1"]),
             N("Conv", ["r1", "w2", "b2"], ["c2"]), N("Relu", ["c2"], ["r2"]),
             N("Conv", ["r2", "w3", "b3"], ["c3"], group=40, pads=[1, 1, 1, 1], strides=[2, 2]),
             N("Clip", ["c3", "lo", "hi"], ["r3"]),
             N("Conv", ["r3", "w4", "b4"], ["c4"]), N("Add", ["r3", "c4"], ["s"]),
             N("GlobalAveragePool", ["s"], ["gp"]), N("Flatten", ["gp"], ["fl"]),
             N("Gemm", ["fl", "w5", "b5"], ["y"], transB=1)]
    g = helper.make_graph(nodes, "i8cnn", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, 12, 12])],
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 10])], inits)
    onnx.save(helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)]), path)


class TestOnnxI8(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import tempfile
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = os.path.join(cls.tmp.name, "i8cnn.onnx")
        build(cls.path)
        cls.g = oi.Graph(cls.path)
        rng = np.random.default_rng(4)
        cls.cal = [{"x": rng.normal(size=(2, 3, 12, 12))} for _ in range(3)]
        cls.x = rng.normal(size=(2, 3, 12, 12))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_float_equals_onnxruntime(self):
        try:
            import onnxruntime as ort
        except ImportError:
            self.skipTest("onnxruntime")
        y = self.g.run_float({"x": self.x})[0]
        ref = ort.InferenceSession(self.path).run(None, {"x": self.x.astype(np.float32)})[0]
        np.testing.assert_allclose(y, ref, rtol=1e-4, atol=1e-5)

    def test_fusion_and_formats(self):
        self.assertEqual([n.op for n in self.g.nodes],
                         ["Conv", "Conv", "Conv", "Conv", "Add", "GlobalAveragePool", "Flatten", "Gemm"])
        fm = oi.calibrate(self.g, self.cal, blocks=True)
        self.assertEqual({t for t, f in fm.items() if f.kind == "b"}, {"r1", "r2"})
        self.assertEqual(fm["y"].kind, "o16")
        self.assertTrue(fm["r3"].unsigned and fm["r3"].kind == "s")
        self.assertFalse(fm["x"].unsigned)
        self.assertEqual(fm["gp"].f, fm["fl"].f)                  # a view shares the exponent
        fs = oi.calibrate(self.g, self.cal, blocks=None)
        self.assertFalse(any(f.kind == "b" for f in fs.values()))

    def test_policy_options(self):
        self.assertEqual(oi.policy_options("i8"), dict(blocks=None, act16=False, in16=False, dwc=False))
        self.assertEqual(oi.policy_options("i8ba+a16+in16"), dict(blocks="any", act16=True, in16=True, dwc=False))
        fm = oi.calibrate(self.g, self.cal, **oi.policy_options("i8+in16"))
        self.assertEqual(fm["x"].kind, "o16")                     # the first Conv in the W8A16 mode
        y = self.g.run_i8({"x": self.x}, fm)[0]
        self.assertLess(np.linalg.norm(y - self.g.run_float({"x": self.x})[0]) / np.linalg.norm(y), 0.1)

    def check_kernels(self, blocks, dwc=False):
        g = self.g
        fm = oi.calibrate(g, self.cal, blocks=blocks, dwc=dwc)
        tr = {}
        g.run_i8({"x": self.x}, fm, trace=tr)
        for n in g.nodes:
            if n.op not in ("Conv", "Gemm"):
                continue
            x = tr[n.inputs[0]].v
            Wd = g.dequant_weights(n)
            _, fw = g.wq[n.outputs[0]]
            fin = fm[n.inputs[0]].f
            gacc = fin + s8.DELTA_MAX + np.asarray(fw, np.int64)
            b = g.init[n.inputs[2]] * (n.attrs.get("beta", 1.0) if n.op == "Gemm" else 1.0)
            bq = s8.bias_raw(b, gacc) * np.exp2(-gacc.astype(np.float64))      # the accumulator's bias
            if n.op == "Conv":
                v = oi.conv_float(x, Wd, None, n.attrs) + bq[None, :, None, None]
            else:
                v = x @ (Wd.T if n.attrs.get("transB") else Wd) + bq[None, :]
            v = oi.apply_act(v, n.act)
            f = fm[n.outputs[0]]
            got = tr[n.outputs[0]]
            if f.kind == "b":                                     # the block rule on the exact value
                q, d = s8.block_round_float(oi.nhwc_pad(v), f.f)
                ref = s8.block_value(q, f.f, d)[..., :v.shape[1]].transpose(0, 3, 1, 2)
                np.testing.assert_array_equal(got.v, ref, err_msg=n.outputs[0])
                np.testing.assert_array_equal(got.d, d.transpose(0, 3, 1, 2), err_msg=n.outputs[0])
            else:
                lo, hi = f.range()
                sc = f.scale(v.ndim)
                ref = np.clip(np.round(v * sc), lo, hi) / sc
                np.testing.assert_array_equal(got.v, ref, err_msg=n.outputs[0])
        return tr, fm

    def test_kernels_static(self):
        self.check_kernels(False)

    def test_kernels_blocked(self):
        tr, _ = self.check_kernels(True)
        self.assertTrue(np.any(tr["r1"].d > 0))                  # the blocks do shift

    def test_kernels_depthwise_per_channel(self):
        _, fm = self.check_kernels(True, dwc=True)
        self.assertEqual(fm["r2"].kind, "c")                     # only the depthwise Conv reads it
        self.assertEqual(fm["r2"].f.shape, (40,))
        self.assertTrue(fm["r2"].unsigned)

    def test_close_to_float(self):
        fm = oi.calibrate(self.g, self.cal, blocks=True)
        y8 = self.g.run_i8({"x": self.x}, fm)[0]
        y = self.g.run_float({"x": self.x})[0]
        self.assertLess(np.linalg.norm(y8 - y) / np.linalg.norm(y), 0.1)


if __name__ == "__main__":
    unittest.main()
