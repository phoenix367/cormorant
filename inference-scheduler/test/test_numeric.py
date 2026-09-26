"""Numeric annotations (src/numeric.py, doc/CHAT_PLAN.md §10.5): power-of-two
exponents, host-memory tensors, states — metadata parsing and validation,
the element-type helpers, the rank-1 weight encoding and the simulator's
MatMul exponent path (exact products, ap_fixed<32,16> wrap, floor at f_out),
checked against explicit integer arithmetic; the generated C against the
software kernels (host emulation)."""

import json
import os
import tempfile
import unittest

import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as nph
from onnx import TensorProto

import host_emu
from src import numeric
from src.codegen import CodeGenerator
from src.dtype import AP_FIXED_16_8 as Q
from src.graph import OnnxGraph
from src.nodes import SchedulerError

vi = oh.make_tensor_value_info


def _model(nodes, inputs, outputs, inits=(), meta=None, value_info=(), domains=()):
    g = oh.make_graph(nodes, "m", inputs, outputs, initializer=list(inits),
                      value_info=list(value_info))
    m = oh.make_model(g, opset_imports=[oh.make_opsetid("", 17)]
                      + [oh.make_opsetid(d, 1) for d in domains])
    m.ir_version = 8
    if meta is not None:
        p = m.metadata_props.add()
        p.key = numeric.METADATA_KEY
        p.value = json.dumps(meta)
    return m


def _mm_model(K=32, M=24, N=3, f_in=None, f_out=None, seed=0, scale=1.0):
    rng = np.random.default_rng(seed)
    W = (rng.normal(0, scale / np.sqrt(K), (K, M))).astype(np.float32)
    meta = {"exp": {}}
    if f_in is not None:
        meta["exp"]["X"] = f_in
    if f_out is not None:
        meta["exp"]["Y"] = f_out
    return _model([oh.make_node("MatMul", ["X", "W"], ["Y"], name="mm")],
                  [vi("X", TensorProto.FLOAT, [N, K])], [vi("Y", TensorProto.FLOAT, [N, M])],
                  [nph.from_array(W, "W")], meta), W


class TestMetadata(unittest.TestCase):

    def test_round_trip(self):
        meta = numeric.empty()
        meta["exp"]["a"] = [8, 9]
        meta["host"]["h"] = "f32"
        meta["state"].append("s")
        d = json.loads(numeric.to_metadata(meta))
        self.assertEqual(d["exp"], {"a": [8, 9]})
        self.assertEqual(d["host"], {"h": "f32"})
        self.assertEqual(d["state"], ["s"])
        self.assertNotIn("test_fill", d)

    def test_unknown_key(self):
        m = _model([oh.make_node("Relu", ["X"], ["Y"])], [vi("X", TensorProto.FLOAT, [8])],
                   [vi("Y", TensorProto.FLOAT, [8])], meta={"expo": {}})
        with self.assertRaises(numeric.NumericError):
            numeric.parse(m)

    def test_no_metadata_is_inactive(self):
        m, _ = _mm_model()
        del m.metadata_props[:]
        g = OnnxGraph(m)
        self.assertFalse(numeric.is_active(g.numeric))
        self.assertTrue(all(t.exp is None and t.host is None and not t.is_state
                            for t in g._tensors.values()))
        self.assertEqual(g.weights_saturated, {})

    def test_bad_exponent_shape(self):
        m, _ = _mm_model(f_out=[8, 9, 10])               # M = 24 channels
        with self.assertRaises(numeric.NumericError):
            OnnxGraph(m)

    def test_exponent_on_f32_host(self):
        m = _model([oh.make_node("Relu", ["X"], ["Y"])], [vi("X", TensorProto.FLOAT, [8])],
                   [vi("Y", TensorProto.FLOAT, [8])], meta={"host": {"X": "f32"}, "exp": {"X": 9}})
        with self.assertRaises(numeric.NumericError):
            OnnxGraph(m)

    def test_host_tensor_into_a_kernel_rejected(self):
        m = _model([oh.make_node("Relu", ["X"], ["Y"])], [vi("X", TensorProto.FLOAT, [8])],
                   [vi("Y", TensorProto.FLOAT, [8])], meta={"host": {"X": "f32"}})
        with self.assertRaisesRegex(SchedulerError, "host / state tensors"):
            OnnxGraph(m)

    def test_exponent_on_vectorop_rejected(self):
        m = _model([oh.make_node("Add", ["X", "X"], ["Y"])], [vi("X", TensorProto.FLOAT, [8])],
                   [vi("Y", TensorProto.FLOAT, [8])], meta={"exp": {"X": 10}})
        with self.assertRaisesRegex(SchedulerError, "power-of-two exponents"):
            OnnxGraph(m)

    def test_state_initializer_becomes_state(self):
        m, _W = _mm_model()
        init = nph.from_array(np.arange(6, dtype=np.float32).reshape(2, 3), "S")
        m.graph.initializer.append(init)
        p = m.metadata_props[0]
        d = json.loads(p.value)
        d["state"] = ["S"]
        d["host"] = {"S": "f32"}
        p.value = json.dumps(d)
        g = OnnxGraph(m)
        t = g.get_tensor("S")
        self.assertTrue(t.is_state and t.is_host and t.data is None)
        np.testing.assert_array_equal(t.init_data, np.arange(6).reshape(2, 3))


class TestDtypeHelpers(unittest.TestCase):

    def test_quantize_exp_half_even_and_saturation(self):
        x = np.array([[0.5 / 256, 1.5 / 1024], [1e9, -1e9], [np.nan, 2.5 / 256]])
        q = Q.quantize_exp(x, np.array([8, 10]))                            # per channel
        np.testing.assert_array_equal(q[0], [0.0, 2.0 / 1024])            # half to even
        np.testing.assert_array_equal(q[1], [32767 / 256, -32768 / 1024])  # saturated
        np.testing.assert_array_equal(q[2], [0.0, 10.0 / 1024])           # NaN -> 0

    def test_truncate_exp_floor(self):
        q = Q.truncate_exp(np.array([-0.1 / 512, 0.9 / 512, 1e6]), 9)
        np.testing.assert_array_equal(q, [-1 / 512, 0.0, 32767 / 512])

    def test_storage_and_ramp(self):
        v = np.array([-1.0, 0.25, 127.0])
        raw = Q.exp_to_storage(v, 7).view(np.int16)
        np.testing.assert_array_equal(raw, [-128, 32, 16256])
        pos = np.array([0, 1, 0x8000, 0xFFFF], np.int64)
        np.testing.assert_array_equal(Q.ramp_to_float_exp(pos, 12),
                                      np.array([0, 1, -32768, -1]) / 4096.0)
        self.assertEqual(Q.frac_bits, 8)
        self.assertEqual(Q.raw_range, (-32768.0, 32767.0))


class TestRankOneWeights(unittest.TestCase):

    def test_encoding_and_simulation(self):
        K, M, N = 32, 24, 3
        rng = np.random.default_rng(3)
        f_in = rng.integers(6, 11, K).tolist()
        f_out = rng.integers(9, 14, M).tolist()
        m, W = _mm_model(K, M, N, f_in=f_in, f_out=f_out)
        g = OnnxGraph(m)
        b = g.get_tensor("W")
        fw = np.asarray(f_out)[None, :] + 8 - np.asarray(f_in)[:, None]
        np.testing.assert_array_equal(b.wexp, fw)
        raw = np.clip(np.round(W.astype(np.float64) * 2.0 ** fw), -32768, 32767)
        np.testing.assert_array_equal(b.data.astype(np.float64), raw / 256.0)
        # the packed image / storage carries the raw bits unchanged
        self.assertTrue(g.nodes[0].b_packed)
        # simulation vs explicit integers: raw inputs, exact products, floor(acc / 256)
        cg = CodeGenerator(g, model_path="mm.onnx")
        xr = rng.integers(-3000, 3000, (N, K))
        x = xr / 2.0 ** np.asarray(f_in)[None, :]
        y = cg.simulate({"X": x})["Y"]
        acc = xr.astype(np.int64) @ raw.astype(np.int64)
        yr = np.clip(np.floor_divide(acc, 256), -32768, 32767)
        np.testing.assert_array_equal(y, yr / 2.0 ** np.asarray(f_out)[None, :])

    def test_accumulator_wraps(self):
        """ap_fixed<32,16>: the raw int32 sum wraps (the study emulates it too)."""
        K, M = 16, 8
        m, _ = _mm_model(K, M, 1, f_in=8, f_out=8, scale=0.0)
        g = OnnxGraph(m)
        b = g.get_tensor("W")
        b.data = np.full((K, M), 32767 / 256.0, np.float32)          # raw 32767
        cg = CodeGenerator(g, model_path="mm.onnx")
        x = np.full((1, K), 32767 / 256.0)
        y = cg.simulate({"X": x})["Y"]
        acc = 16 * 32767 * 32767                                       # > 2^31
        wrapped = ((acc + 2 ** 31) % 2 ** 32) - 2 ** 31
        self.assertNotEqual(acc, wrapped)
        np.testing.assert_array_equal(y, np.full((1, M), max(-32768, min(32767, wrapped // 256)) / 256))

    def test_weight_shared_with_different_exponents_rejected(self):
        K, M = 16, 8
        W = np.ones((K, M), np.float32) * 0.01
        m = _model([oh.make_node("MatMul", ["X", "W"], ["A"]),
                    oh.make_node("MatMul", ["X", "W"], ["B"])],
                   [vi("X", TensorProto.FLOAT, [2, K])],
                   [vi("A", TensorProto.FLOAT, [2, M]), vi("B", TensorProto.FLOAT, [2, M])],
                   [nph.from_array(W, "W")], {"exp": {"A": 9, "B": 10}})
        with self.assertRaisesRegex(SchedulerError, "different exponents"):
            OnnxGraph(m)

    def test_host_emulation(self):
        """Generated C against the software MatmulKernel: raw bits of the
        exponent output equal the simulation (ramp inputs at the per-channel
        input exponents)."""
        rng = np.random.default_rng(5)
        K, M, N = 48, 40, 5
        m, _ = _mm_model(K, M, N, f_in=rng.integers(7, 10, K).tolist(),
                         f_out=rng.integers(10, 13, M).tolist(), scale=0.2)
        cg = CodeGenerator(OnnxGraph(m), model_path="mm.onnx")
        with tempfile.TemporaryDirectory() as td:
            rc, out = host_emu.build_and_run(cg, td)
        self.assertEqual(rc, 0, out[-2000:])
        self.assertIn("test_inference PASSED", out)


if __name__ == "__main__":
    unittest.main()
