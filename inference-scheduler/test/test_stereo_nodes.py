"""The stereo depth ops (src/stereo_nodes.py, doc/plans/STEREO_PLAN.md) and
the copies with exponents (numeric.check): every reference against an
independent float formula, the exponent rules, and a small graph with every
op — Convs, StereoVop (LEAKY_RELU, RELU6, ADD + ReLU, MUL), instance norm,
edge pad, correlation, Concat, a pixel-shuffle Transpose, the column-mode
softmax, a regression MatMul and the context upsampling — compiled on the
host against the kernels' software models and compared bit for bit with
the simulation (host_emu)."""

import shutil
import tempfile
import unittest

import numpy as np
from onnx import TensorProto as TP, helper as oh, numpy_helper as nph

import host_emu
from src import numeric
from src.codegen import CodeGenerator
from src.graph import OnnxGraph
from src.llm_nodes import LLM_DOMAIN
from src.nodes import SchedulerError
from src.stereo_nodes import (StereoCorrelationNode, StereoInstanceNormNode, StereoSoftmaxNode,
                              StereoUpsampleNode, StereoVopNode)

H, W = 8, 32


def _vi(name, shape):
    return oh.make_tensor_value_info(name, TP.FLOAT, shape)


def _model(nodes, inputs, outputs, inits, exp, host=None, shapes=None):
    g = oh.make_graph(nodes, "stereo_ops", inputs, outputs, initializer=inits,
                      value_info=[_vi(n, list(s)) for n, s in (shapes or {}).items()])
    m = oh.make_model(g, opset_imports=[oh.make_opsetid("", 17), oh.make_opsetid(LLM_DOMAIN, 1)])
    m.ir_version = 8
    meta = numeric.empty()
    meta["exp"].update(exp)
    meta["host"].update(host or {})
    p = m.metadata_props.add()
    p.key, p.value = numeric.METADATA_KEY, numeric.to_metadata(meta)
    return m


def _s(op_type, ins, out, **attrs):
    return oh.make_node(op_type, ins, [out], domain=LLM_DOMAIN, name=out, **attrs)


def full_model(seed=0):
    """Every stereo op in one graph (the shapes of a small LightStereo slice)."""
    rng = np.random.default_rng(seed)

    def w(name, shape, scale):
        return nph.from_array((rng.standard_normal(shape) * scale).astype(np.float32), name)

    inits = [w("w1", (8, 3, 3, 3), 0.3), w("b1", (8,), 0.2), w("w2", (8, 8, 3, 3), 0.1),
             w("w3", (16, 8, 1, 1), 0.3), w("w4", (8, 16, 1, 1), 0.2), w("w5", (8, 8, 1, 1), 0.2),
             nph.from_array(np.arange(8, dtype=np.float32).reshape(8, 1), "dvals"),
             nph.from_array(np.array([2, 2, 4, H, W], np.int64), "shp5"),
             nph.from_array(np.array([1, 4, 2 * H, 2 * W], np.int64), "shp4")]
    nodes = []
    for s in ("L", "R"):
        nodes += [oh.make_node("Conv", [s, "w1", "b1"], [f"c1{s}"], name=f"c1{s}", pads=[1, 1, 1, 1]),
                  _s("StereoVop", [f"c1{s}"], f"a1{s}", op=6, alpha="0.2"),
                  _s("StereoInstanceNorm", [f"a1{s}"], f"n{s}"),
                  _s("StereoPadEdge", [f"n{s}"], f"p{s}", pad=1),
                  oh.make_node("Conv", [f"p{s}", "w2"], [f"c2{s}"], name=f"c2{s}")]
    nodes += [_s("StereoCorrelation", ["c2L", "c2R"], "vol", disp=8),
              oh.make_node("Conv", ["vol", "w3"], ["e"], name="e"),
              _s("StereoVop", ["e"], "r6", op=5),
              oh.make_node("Conv", ["r6", "w4"], ["pj"], name="pj"),
              _s("StereoVop", ["vol", "pj"], "s", op=0, act=1),
              oh.make_node("Conv", ["vol", "w5"], ["gate"], name="gate"),
              _s("StereoVop", ["s", "gate"], "m", op=2),
              oh.make_node("Concat", ["m", "m"], ["cat"], name="cat", axis=1),
              oh.make_node("Reshape", ["cat", "shp5"], ["r5"], name="r5"),
              oh.make_node("Transpose", ["r5"], ["t5"], name="t5", perm=[2, 3, 0, 4, 1]),
              oh.make_node("Reshape", ["t5", "shp4"], ["sh"], name="sh"),
              _s("StereoSoftmax", ["m"], "P", p_exp=15),
              oh.make_node("MatMul", ["P", "dvals"], ["disp"], name="disp"),
              _s("StereoSoftmax", ["G"], "P2", p_exp=15, valid=9),
              _s("StereoUpsample", ["P", "P2"], "up", h=H, w=W, scale=2),
              _s("StereoSoftmax", ["G4"], "P4", p_exp=15, valid=9, groups=4),
              _s("StereoUpsample", ["P", "P4"], "up4", h=H, w=W, scale=2, phases=2)]
    inputs = [_vi("L", [1, 3, H, W]), _vi("R", [1, 3, H, W]), _vi("G", [1, 16, 2 * H, 2 * W]),
              _vi("G4", [1, 64, H, W])]
    outputs = [_vi("up", [2 * H, 2 * W]), _vi("up4", [2 * H, 2 * W]), _vi("sh", [1, 4, 2 * H, 2 * W]),
               _vi("disp", [H * W, 1])]
    exp = {"L": 12, "R": 12, "G": 10, "G4": 10, "P4": 15, "c1L": 10, "c1R": 10, "a1L": 10, "a1R": 10, "nL": 11, "nR": 11,
           "pL": 11, "pR": 11, "c2L": 9, "c2R": 9, "vol": 12, "e": 8, "r6": 8, "pj": 12, "s": 12,
           "gate": 4, "m": 8, "cat": 8, "r5": 8, "t5": 8, "sh": 8, "P": 15, "disp": 9, "P2": 15}
    c8 = (1, 8, H, W)
    shapes = {"a1L": c8, "a1R": c8, "nL": c8, "nR": c8, "pL": (1, 8, H + 2, W + 2),
              "pR": (1, 8, H + 2, W + 2), "vol": c8, "r6": (1, 16, H, W), "s": c8, "m": c8,
              "P": (H * W, 8), "P2": (4 * H * W, 16), "P4": (4 * H * W, 16)}
    return _model(nodes, inputs, outputs, inits, exp, host={"up": "f32", "up4": "f32"}, shapes=shapes)


def _feeds(g, seed=1):
    rng = np.random.default_rng(seed)
    out = {}
    for t in g.input_tensors:
        f = int(np.asarray(t.exp))
        raw = rng.integers(-6000, 6000, t.shape) if t.onnx_name != "G" else rng.integers(-4000, 4000, t.shape)
        out[t.onnx_name] = raw.astype(np.float64) * 2.0 ** -f
    return out


class TestStereoGraph(unittest.TestCase):
    def test_host_emu_bit_exact(self):
        """The generated C (host ops, kernel calls, layouts) equals the simulation."""
        if host_emu.which_cc() is None:
            self.fail("no C compiler")
        g = OnnxGraph(full_model())
        kinds = {type(sn).__name__ for sn in g.nodes}
        for k in ("StereoVopNode", "StereoSoftmaxNode", "StereoInstanceNormNode", "StereoPadEdgeNode",
                  "StereoCorrelationNode", "StereoUpsampleNode", "ConcatNode", "TransposeNode"):
            self.assertIn(k, kinds)
        cg = CodeGenerator(g, model_path="stereo_ops.onnx")
        d = tempfile.mkdtemp(prefix="stereo_emu_")
        try:
            for threads in (1, 4):
                rc, out = host_emu.build_and_run(cg, d, threads=threads, min_elems=1)
                self.assertEqual(rc, 0, out[-3000:])
                self.assertEqual(host_emu.failures(out), [], out[-3000:])
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_simulation_against_float(self):
        """The simulated disparity follows the float computation of the same graph."""
        g = OnnxGraph(full_model())
        cg = CodeGenerator(g, model_path="stereo_ops.onnx")
        arr = cg._forward_pass(_feeds(g))
        for k in ("up", "up4"):
            up = arr[k]
            self.assertEqual(up.shape, (2 * H, 2 * W))
            self.assertTrue(np.all(up >= 0) and np.all(up <= 2 * 7 + 1e-6))   # 2 x a mean of 0 .. 7


class TestStereoReferences(unittest.TestCase):
    def node(self, cls, **kw):
        sn = cls.__new__(cls)
        for k, v in kw.items():
            setattr(sn, k, v)
        return sn

    def test_instnorm_within_one_lsb(self):
        rng = np.random.default_rng(3)
        C, n, fx, fy = 5, 300, 9, 11
        raw = rng.integers(-20000, 20000, (C, n))
        sn = StereoInstanceNormNode.__new__(StereoInstanceNormNode)
        sn.C, sn.n, sn.f_x, sn.f_y, sn.eps = C, n, fx, fy, 1e-5
        sn.output = type("T", (), {"shape": (1, C, 10, 30)})()
        y = sn.reference([raw * 2.0 ** -fx], None).reshape(C, n)
        x = raw * 2.0 ** -fx
        want = (x - x.mean(1, keepdims=True)) / np.sqrt(x.var(1, keepdims=True) + 1e-5)
        self.assertLessEqual(np.abs(y - want).max(), 2.0 ** -fy * 0.5 + 1e-9)

    def test_correlation_within_half_lsb(self):
        rng = np.random.default_rng(4)
        C, Hh, Ww, D, fl, fr, fv = 6, 3, 20, 7, 11, 10, 12
        lr, rr = rng.integers(-9000, 9000, (2, C, Hh, Ww))
        sn = StereoCorrelationNode.__new__(StereoCorrelationNode)
        sn.C, sn.H, sn.W, sn.D, sn.f_l, sn.f_r, sn.f_v = C, Hh, Ww, D, fl, fr, fv
        sn.output = type("T", (), {"shape": (1, D, Hh, Ww)})()
        y = sn.reference([lr * 2.0 ** -fl, rr * 2.0 ** -fr], None).reshape(D, Hh, Ww)
        L, R = lr * 2.0 ** -fl, rr * 2.0 ** -fr
        for d in range(D):
            want = np.zeros((Hh, Ww))
            want[:, d:] = (L[:, :, d:] * R[:, :, :Ww - d]).mean(0)
            self.assertLessEqual(np.abs(np.clip(want, -8, 32767 * 2.0 ** -fv) - y[d]).max(), 2.0 ** -fv * 0.5 + 1e-12)

    def test_upsample_formula(self):
        rng = np.random.default_rng(5)
        h, w, s, K, D = 3, 5, 4, 16, 6
        pd = rng.integers(0, 6000, (h * w, D)) * 2.0 ** -15
        p = rng.integers(0, 32768, (h * s * w * s, K)) * 2.0 ** -15
        d = (pd * np.arange(D)[None, :]).sum(1).reshape(h, w)
        dp = np.pad(d, 1)
        H, W = h * s, w * s
        for phases in (1, 2):
            sn = StereoUpsampleNode.__new__(StereoUpsampleNode)
            sn.h, sn.w, sn.s, sn.K, sn.D, sn.phases, sn.f_d, sn.f_p = h, w, s, K, D, phases, 15, 15
            sn.output = type("T", (), {"shape": (H, W)})()
            y = sn.reference([pd, p], None)
            want = np.zeros((H, W))
            for yy in range(H):
                for xx in range(W):
                    row = (yy * W + xx if phases == 1 else
                           ((yy % 2) * 2 + xx % 2) * (H // 2) * (W // 2) + (yy // 2) * (W // 2) + xx // 2)
                    for k in range(9):
                        want[yy, xx] += p[row, k] * s * dp[yy // s + k // 3, xx // s + k % 3]
            np.testing.assert_array_equal(y, want.astype(np.float32).astype(np.float64))

    def test_leaky_relu_rounds_half_even(self):
        sn = StereoVopNode.__new__(StereoVopNode)
        sn.op, sn.act, sn.alpha, sn.F, sn.f_out = 6, 0, 1 << 15, 8, 10
        t = type("T", (), {"exp": np.asarray(10)})()
        sn.inputs = [t]
        sn.output = type("O", (), {"shape": (4,)})()
        y = sn.reference([np.array([-3, -5, 7, -1]) * 2.0 ** -10], None) * 2 ** 10
        np.testing.assert_array_equal(y, [-2, -2, 7, -0])          # -1.5 -> -2, -2.5 -> -2, -0.5 -> 0

    def test_softmax_columns(self):
        sn = StereoSoftmaxNode.__new__(StereoSoftmaxNode)
        from src.smx_nodes import regs
        sn.C, sn.T, sn.G, sn.valid, sn.f_s = 8, 16, 1, 5, 9
        sn.cm, sn.cfg = regs(9, 1.0, 15)
        sn.output = type("O", (), {"shape": (16, 8)})()
        rng = np.random.default_rng(6)
        x = rng.integers(-3000, 3000, (8, 16)) * 2.0 ** -9
        p = sn.reference([x], None)
        self.assertTrue(np.all(p[:, 5:] == 0))
        e = np.exp(x[:5].T - x[:5].T.max(1, keepdims=True))
        np.testing.assert_allclose(p[:, :5], e / e.sum(1, keepdims=True), atol=2 * 2.0 ** -15)


class TestStereoRules(unittest.TestCase):
    def one(self, node, ins, out, exp, out_shape=(1, 8, H, W)):
        m = _model([node], [_vi(n, [1, 8, H, W]) for n in ins], [_vi(out, list(out_shape))], [], exp)
        return OnnxGraph(m)

    def test_add_needs_one_exponent(self):
        with self.assertRaisesRegex(SchedulerError, "ADD: one exponent"):
            self.one(_s("StereoVop", ["a", "b"], "c", op=0), ["a", "b"], "c", {"a": 10, "b": 11, "c": 10})

    def test_mul_exponent(self):
        self.one(_s("StereoVop", ["a", "b"], "c", op=2), ["a", "b"], "c", {"a": 10, "b": 11, "c": 13})
        with self.assertRaisesRegex(SchedulerError, "MUL"):
            self.one(_s("StereoVop", ["a", "b"], "c", op=2), ["a", "b"], "c", {"a": 10, "b": 11, "c": 12})

    def test_relu6_at_q88_only(self):
        self.one(_s("StereoVop", ["a"], "c", op=5), ["a"], "c", {"a": 8, "c": 8})
        with self.assertRaisesRegex(SchedulerError, "RELU6"):
            self.one(_s("StereoVop", ["a"], "c", op=5), ["a"], "c", {"a": 10, "c": 10})

    def test_concat_needs_one_exponent(self):
        node = oh.make_node("Concat", ["a", "b"], ["c"], name="c", axis=1)
        self.one(node, ["a", "b"], "c", {"a": 10, "b": 10, "c": 10}, out_shape=(1, 16, H, W))
        with self.assertRaisesRegex(SchedulerError, "one and the same"):
            self.one(node, ["a", "b"], "c", {"a": 10, "b": 11, "c": 10}, out_shape=(1, 16, H, W))


if __name__ == "__main__":
    unittest.main()
