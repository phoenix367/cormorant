"""Convs on tensors with power-of-two exponents (src/numeric.py
encode_conv_weights, the simulator's _conv_exp): ConvKernel sums raw int16
products exactly and writes floor(acc / 2^8) saturated, so the weight sits
at f_w = f_y + 8 - f_x and the bias at f_y.  Checked against an independent
integer loop, and the generated C against the simulation (host_emu)."""

import os
import shutil
import tempfile
import unittest

import numpy as np
from onnx import TensorProto as TP, helper as oh, numpy_helper as nph

import host_emu
from src import numeric
from src.codegen import CodeGenerator
from src.graph import OnnxGraph


def conv_model(C, M, H, W, k, dil=1, pad=0, fx=11, fy=10, bias=True, depthwise=False, seed=0):
    rng = np.random.default_rng(seed)
    w = (rng.standard_normal((M, 1 if depthwise else C, 1, k)) * 0.05).astype(np.float32)
    b = (rng.standard_normal(M) * 0.3).astype(np.float32)
    ow = W + 2 * pad - dil * (k - 1)
    inits = [nph.from_array(w, "w")] + ([nph.from_array(b, "b")] if bias else [])
    node = oh.make_node("Conv", ["x", "w"] + (["b"] if bias else []), ["y"], name="conv",
                        kernel_shape=[1, k], dilations=[1, dil], pads=[0, pad, 0, pad],
                        group=C if depthwise else 1)
    g = oh.make_graph([node], "c", [oh.make_tensor_value_info("x", TP.FLOAT, [1, C, H, W])],
                      [oh.make_tensor_value_info("y", TP.FLOAT, [1, M, H, ow])], initializer=inits)
    m = oh.make_model(g, opset_imports=[oh.make_opsetid("", 17)])
    m.ir_version = 8
    meta = numeric.empty()
    meta["exp"]["x"], meta["exp"]["y"] = fx, fy
    p = m.metadata_props.add()
    p.key, p.value = numeric.METADATA_KEY, numeric.to_metadata(meta)
    return m, w, b


def int_conv(x_raw, w_raw, b_raw, k, dil, pad, depthwise):
    """The kernel in integers: sum raw products, seed with bias << 8, floor(>> 8), saturate."""
    N, C, H, W = x_raw.shape
    M = w_raw.shape[0]
    xp = np.pad(x_raw, ((0, 0), (0, 0), (0, 0), (pad, pad))).astype(np.int64)
    ow = W + 2 * pad - dil * (k - 1)
    acc = np.zeros((N, M, H, ow), np.int64)
    for m in range(M):
        for t in range(k):
            seg = xp[:, :, :, t * dil: t * dil + ow]
            if depthwise:
                acc[:, m] += seg[:, m] * int(w_raw[m, 0, 0, t])
            else:
                acc[:, m] += np.einsum("nchw,c->nhw", seg, w_raw[m, :, 0, t].astype(np.int64))
        if b_raw is not None:
            acc[:, m] += int(b_raw[m]) << 8
    return np.clip(np.floor_divide(acc, 256), -32768, 32767)


class TestConvExp(unittest.TestCase):
    def check(self, depthwise=False, **kw):
        C = kw.get("C", 24)
        m, w, b = conv_model(depthwise=depthwise, **kw)
        g = OnnxGraph(m)
        sn = g.nodes[0]
        fx, fy = kw.get("fx", 11), kw.get("fy", 10)
        fw = fy + 8 - fx
        wt, bt = sn.inputs[1], sn.inputs[2] if sn.has_bias else None
        w_raw = np.clip(np.round(w.astype(np.float64) * 2.0 ** fw), -32768, 32767)
        np.testing.assert_array_equal(np.round(wt.data.astype(np.float64) * 256).reshape(w.shape), w_raw)
        self.assertEqual(int(wt.wexp), fw)
        b_raw = None
        if bt is not None:
            b_raw = np.clip(np.round(b.astype(np.float64) * 2.0 ** fy), -32768, 32767)
            np.testing.assert_array_equal(np.round(bt.data.astype(np.float64) * 256), b_raw)
        rng = np.random.default_rng(1)
        x_raw = rng.integers(-3000, 3000, (1, C, kw.get("H", 3), kw.get("W", 40))).astype(np.float64)
        cg = CodeGenerator(g, model_path="c.onnx")
        y = cg._forward_pass({"x": x_raw * 2.0 ** -fx})["y"]
        want = int_conv(x_raw, w_raw, b_raw, kw.get("k", 5), kw.get("dil", 1), kw.get("pad", 0), depthwise)
        np.testing.assert_array_equal(np.round(y * 2.0 ** fy), want)
        return g, cg

    def test_plain(self):
        self.check(C=24, M=40, H=3, W=40, k=5, pad=2)

    def test_dilated_no_bias(self):
        self.check(C=16, M=16, H=2, W=64, k=7, dil=3, pad=9, bias=False, fx=9, fy=12)

    def test_depthwise(self):
        self.check(C=32, M=32, H=2, W=48, k=3, pad=1, depthwise=True)

    def test_saturation_and_encoding(self):
        # a coarse input exponent pushes the weight exponent up: fx 4, fy 12 -> fw 16
        g, _ = self.check(C=8, M=8, H=1, W=32, k=3, pad=1, fx=4, fy=12)
        self.assertIn("w", g.weights_saturated)

    def test_legacy_conv_unchanged(self):
        m, w, _ = conv_model(C=8, M=8, H=2, W=16, k=3, pad=1)
        del m.metadata_props[:]
        g = OnnxGraph(m)
        self.assertIsNone(g.nodes[0].inputs[1].wexp)
        np.testing.assert_array_equal(g.nodes[0].inputs[1].data, w)

    def test_exponent_rules(self):
        m, _, _ = conv_model(C=8, M=8, H=1, W=16, k=3, pad=1)
        meta = numeric.empty()
        meta["exp"]["x"], meta["exp"]["y"] = [9] * 16, 10          # per last axis: rejected
        m.metadata_props[0].value = numeric.to_metadata(meta)
        with self.assertRaises(numeric.NumericError):
            OnnxGraph(m)

    @unittest.skipUnless(shutil.which("cc") and shutil.which("cmake"), "needs cc and cmake")
    def test_generated_c_matches_simulation(self):
        m, _, _ = conv_model(C=24, M=40, H=3, W=40, k=5, pad=2)
        cg = CodeGenerator(OnnxGraph(m), model_path="c.onnx")
        with tempfile.TemporaryDirectory() as td:
            rc, out = host_emu.build_and_run(cg, os.path.join(td, "p"))
        self.assertEqual(rc, 0, out[-3000:])
        self.assertIn("PASSED", out)


def per_channel_model(C=6, M=5, H=3, W=20, fx=11, fy=9, fc=(9, 12, 14, 10, 16), seed=2):
    """conv (per-output-channel exponents fc, chexp) -> identity depthwise 1 x 1
    conv (the rescale) -> y at fy."""
    rng = np.random.default_rng(seed)
    w = (rng.standard_normal((M, C, 3, 3)) * np.array([1, 0.05, 0.01, 0.3, 0.002])[:M, None, None, None]
         ).astype(np.float32)
    b = (rng.standard_normal(M) * 0.01).astype(np.float32)
    ones = np.ones((M, 1, 1, 1), np.float32)
    n1 = oh.make_node("Conv", ["x", "w", "b"], ["t"], name="conv", kernel_shape=[3, 3], pads=[1, 1, 1, 1])
    n2 = oh.make_node("Conv", ["t", "ones"], ["y"], name="rescale", kernel_shape=[1, 1], group=M)
    g = oh.make_graph([n1, n2], "pc", [oh.make_tensor_value_info("x", TP.FLOAT, [1, C, H, W])],
                      [oh.make_tensor_value_info("y", TP.FLOAT, [1, M, H, W])],
                      initializer=[nph.from_array(w, "w"), nph.from_array(b, "b"), nph.from_array(ones, "ones")],
                      value_info=[oh.make_tensor_value_info("t", TP.FLOAT, [1, M, H, W])])
    m = oh.make_model(g, opset_imports=[oh.make_opsetid("", 17)])
    m.ir_version = 8
    meta = numeric.empty()
    meta["exp"]["x"], meta["exp"]["y"] = fx, fy
    meta["chexp"]["t"] = list(fc[:M])
    p = m.metadata_props.add()
    p.key, p.value = numeric.METADATA_KEY, numeric.to_metadata(meta)
    return m, w, b


class TestPerChannel(unittest.TestCase):
    """Per-output-channel weight exponents (chexp) and the depthwise 1 x 1
    rescale: equal to one floor at f_y of the per-channel-encoded sum."""

    def test_encoding_and_rescale(self):
        fx, fy, fc = 11, 9, (9, 12, 14, 10, 16)
        m, w, b = per_channel_model(fx=fx, fy=fy, fc=fc)
        g = OnnxGraph(m)
        conv, resc = g.nodes
        fw = np.array(fc)[:, None] + 8 - fx                       # [M][C] (rank-1, C equal)
        self.assertEqual(conv.inputs[1].wexp.shape, (5, 6, 1, 1))
        np.testing.assert_array_equal(conv.inputs[1].wexp[:, :, 0, 0], np.broadcast_to(fw, (5, 6)))
        np.testing.assert_array_equal(resc.inputs[1].wexp.reshape(-1), fy + 8 - np.array(fc))
        rng = np.random.default_rng(3)
        x_raw = rng.integers(-4000, 4000, (1, 6, 3, 20)).astype(np.float64)
        cg = CodeGenerator(g, model_path="pc.onnx")
        arr = cg._forward_pass({"x": x_raw * 2.0 ** -fx})
        # the integer reference: per-channel weights, floor at f_c, then floor at f_y
        w_raw = np.clip(np.round(w.astype(np.float64) * 2.0 ** fw[:, :, None, None]), -32768, 32767)
        b_raw = np.clip(np.round(b.astype(np.float64) * 2.0 ** np.array(fc)), -32768, 32767)
        xp = np.pad(x_raw, ((0, 0), (0, 0), (1, 1), (1, 1))).astype(np.int64)
        acc = np.zeros((5, 3, 20), np.int64)
        for a in range(3):
            for c in range(3):
                acc += np.einsum("chw,mc->mhw", xp[0, :, a:a + 3, c:c + 20], w_raw[:, :, a, c].astype(np.int64))
        acc += (b_raw.astype(np.int64) << 8)[:, None, None]
        t_raw = np.clip(np.floor_divide(acc, 256), -32768, 32767)
        y_raw = np.clip(np.floor_divide(t_raw * (2 ** (fy + 8 - np.array(fc)))[:, None, None], 256), -32768, 32767)
        np.testing.assert_array_equal(arr["y"].reshape(5, 3, 20) * 2.0 ** fy, y_raw)

    def test_chexp_only_on_convs(self):
        m, _, _ = per_channel_model()
        meta = numeric.empty()
        meta["exp"]["x"], meta["exp"]["y"] = 11, 9
        meta["chexp"]["x"] = [11] * 6
        m.metadata_props[0].value = numeric.to_metadata(meta)
        with self.assertRaises(numeric.NumericError):              # x has exp and chexp
            OnnxGraph(m)

    def test_generated_c_matches_simulation(self):
        m, _, _ = per_channel_model(H=4, W=40)
        cg = CodeGenerator(OnnxGraph(m), model_path="pc.onnx")
        with tempfile.TemporaryDirectory() as td:
            rc, out = host_emu.build_and_run(cg, os.path.join(td, "p"))
        self.assertEqual(rc, 0, out[-3000:])
        self.assertIn("PASSED", out)


if __name__ == "__main__":
    unittest.main()
