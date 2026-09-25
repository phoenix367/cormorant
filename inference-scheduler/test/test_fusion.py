"""Pattern-fusion pre-pass (BERT_PLAN 1c, src/fusion.py).

Matched patterns become one host node and compute exactly what the original
subgraph computes op by op in float64 (then one round-half-even write-back);
near misses are left alone, so their ReduceMean / Tanh / Erf / ... fail
dispatch with the pattern hint.  Also: Split lowering, Constant folding,
constant-broadcast normalisation for the VectorOP kernel.
"""

import math
import os
import tempfile
import unittest

import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as nph
from onnx import TensorProto

from src.codegen import CodeGenerator
from src.dtype import AP_FIXED_16_8 as DT
from src.graph import OnnxGraph
from src.host_nodes import GeluNode, LayerNormNode, SliceNode
from src.nodes import ScheduledNode, SchedulerError


def _vi(name, shape, et=TensorProto.FLOAT):
    return oh.make_tensor_value_info(name, et, shape)


def _c(name, v, dt=np.float32):
    return nph.from_array(np.asarray(v, dt), name)


def _grid(rng, shape, lo, hi):
    return np.round(rng.uniform(lo, hi, shape) * 256) / 256


class _G:
    """Graph under construction: nodes, initializers, outputs."""

    def __init__(self):
        self.nodes, self.inits, self.extra_outputs = [], [], []

    def n(self, op, ins, out, **kw):
        self.nodes.append(oh.make_node(op, ins, [out], name=f"blk/{out}", **kw))
        return out

    def c(self, name, v, dt=np.float32):
        self.inits.append(_c(name, v, dt))
        return name


def _tf_layernorm(g, x, n, axis, *, eps=1e-6, swap=False, pow2=False, div_recip=False,
                  gamma_runtime=False, eps_name=None, rng=None):
    """bertsquad-12's LayerNorm subgraph; ``swap`` flips every commutative
    operand order, the other flags select accepted variants / near misses."""
    rng = rng or np.random.default_rng(0)
    gamma = "gamma" if gamma_runtime else g.c("gamma", rng.normal(1, 0.2, n))
    beta = g.c("beta", rng.normal(0, 0.2, n))
    o = (lambda a, b: [b, a]) if swap else (lambda a, b: [a, b])
    mean = g.n("ReduceMean", [x], "mean", axes=[axis], keepdims=1)
    d = g.n("Sub", [x, mean], "sqdiff")
    dd = (g.n("Pow", [d, g.c("two", 2.0)], "dd") if pow2 else g.n("Mul", [d, d], "dd"))
    var = g.n("ReduceMean", [dd], "var", axes=[axis], keepdims=1)
    ve = g.n("Add", o(var, eps_name or g.c("eps", eps)), "ve")
    sd = g.n("Sqrt", [ve], "sd")
    inv = (g.n("Div", [g.c("one", 1.0), sd], "inv") if div_recip else
           g.n("Reciprocal", [sd], "inv"))
    gg = g.n("Mul", o(inv, gamma), "g")
    mg = g.n("Mul", o(mean, gg), "mg")
    b = g.n("Sub", [beta, mg], "b")
    xg = g.n("Mul", o(x, gg), "xg")
    return g.n("Add", o(xg, b), "Y")


def _tf_ref(x, gamma, beta, eps):
    """The TF subgraph op by op in float64 (float32 params), numpy's own
    reductions — an independent reference for the fused node."""
    mean = np.mean(x, -1, keepdims=True)
    var = np.mean((x - mean) * (x - mean), -1, keepdims=True)
    g = (1.0 / np.sqrt(var + np.float64(np.float32(eps)))) * gamma.astype(np.float64)
    return x * g + (beta.astype(np.float64) - mean * g)


def _gelu_tanh(g, x, *, c1=0.044715, tail="bert", swap=False, cube="pow"):
    o = (lambda a, b: [b, a]) if swap else (lambda a, b: [a, b])
    if cube == "pow":
        p = g.n("Pow", [x, g.c("three", 3.0)], "p")
    else:
        sq = g.n("Mul", [x, x], "sq")
        p = g.n("Mul", o(sq, x), "p")
    m = g.n("Mul", o(g.c("c1", c1), p), "m")
    a = g.n("Add", o(x, m), "a")
    m1 = g.n("Mul", o(g.c("c2", math.sqrt(2 / math.pi)), a), "m1")
    t = g.n("Tanh", [m1], "t")
    a1 = g.n("Add", o(g.c("one", 1.0), t), "a1")
    return _gelu_tail(g, x, a1, tail, o)


def _gelu_tail(g, x, a1, tail, o):
    half = g.c("half", 0.5)
    if tail == "bert":                               # x * (0.5 * a1)
        return g.n("Mul", o(x, g.n("Mul", o(half, a1), "m2")), "Y")
    if tail == "xa_half":                            # (x * a1) * 0.5
        return g.n("Mul", o(g.n("Mul", o(x, a1), "q"), half), "Y")
    if tail == "halfx_a":                            # (x * 0.5) * a1
        return g.n("Mul", o(g.n("Mul", o(x, half), "h"), a1), "Y")
    if tail == "none":                               # near miss: no 0.5
        return g.n("Mul", o(x, a1), "Y")
    raise ValueError(tail)


def _gelu_erf(g, x, *, mul=False, tail="xa_half", k=None):
    o = lambda a, b: [a, b]  # noqa: E731
    if mul:
        u = g.n("Mul", [x, g.c("k", k or 1 / math.sqrt(2))], "u")
    else:
        u = g.n("Div", [x, g.c("k", k or math.sqrt(2))], "u")
    e = g.n("Erf", [u], "e")
    a1 = g.n("Add", [e, g.c("one", 1.0)], "a1")
    return _gelu_tail(g, x, a1, tail, o)


class _Base(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.k = 0

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def save(self, g, shape, inputs=None, opset=13, out_shape=None):
        type(self).k += 1
        ins = inputs or [_vi("X", shape)]
        outs = [_vi("Y", out_shape or shape)] + g.extra_outputs
        graph = oh.make_graph(g.nodes, f"f{self.k}", ins, outs, initializer=g.inits)
        m = oh.make_model(graph, opset_imports=[oh.make_opsetid("", opset)])
        m.ir_version = 8
        onnx.checker.check_model(m)
        p = os.path.join(self._tmp.name, f"f{self.k}.onnx")
        onnx.save(m, p)
        return p

    def sim(self, p, feeds, **kw):
        g = OnnxGraph(p, **kw)
        cg = CodeGenerator(g, model_path=p)
        return g, cg.simulate(feeds)


class TestLayerNormFusion(_Base):

    def _check_fused(self, shape, axis, **kw):
        rng = np.random.default_rng(len(shape) * 10 + axis)
        g = _G()
        _tf_layernorm(g, "X", shape[-1], axis, rng=rng, **kw)
        p = self.save(g, shape)
        x = _grid(np.random.default_rng(1), shape, -8, 8)
        graph, out = self.sim(p, {"X": x})
        self.assertEqual(graph.fusion_counts["layernorm"], 1)
        self.assertEqual(len(graph.nodes), 1)
        sn = graph.nodes[0]
        self.assertIsInstance(sn, LayerNormNode)
        self.assertEqual((sn.tf_form, sn.n, sn.rows), (1, shape[-1], x.size // shape[-1]))
        self.assertEqual(sn.onnx_node.name, "blk")               # common name prefix
        gamma = nph.to_array(next(i for i in g.inits if i.name == "gamma"))
        beta = nph.to_array(next(i for i in g.inits if i.name == "beta"))
        self.assertEqual(sn.eps, float(np.float32(kw.get("eps", 1e-6))))
        # bit-exact vs the original subgraph evaluated op by op in float64
        np.testing.assert_array_equal(out["Y"], DT.host_quantize(
            _tf_ref(x, gamma, beta, kw.get("eps", 1e-6))))

    def test_bert_arrangement_3d(self):
        self._check_fused([2, 6, 32], 2)

    def test_bert_arrangement_2d_axis1(self):
        self._check_fused([6, 32], 1)

    def test_commuted_operands_and_variants(self):
        self._check_fused([4, 16], -1, swap=True)
        self._check_fused([4, 16], 1, pow2=True, div_recip=True, eps=1e-12)

    def _near_miss(self, g, shape, why, **kw):
        p = self.save(g, shape, **kw)
        with self.assertRaisesRegex(SchedulerError, "only supported inside a fused"):
            OnnxGraph(p)
        return p

    def test_near_miss_extra_consumer(self):
        g = _G()
        _tf_layernorm(g, "X", 16, 1)
        g.extra_outputs.append(_vi("mean", [4, 1]))            # mean escapes
        self._near_miss(g, [4, 16], "mean is a graph output")

    def test_near_miss_wrong_axis(self):
        g = _G()
        _tf_layernorm(g, "X", 16, 0)
        self._near_miss(g, [4, 16], "reduces axis 0")

    def test_near_miss_runtime_gamma_and_eps(self):
        g = _G()
        _tf_layernorm(g, "X", 16, 1, gamma_runtime=True)
        self._near_miss(g, [4, 16], "gamma", inputs=[_vi("X", [4, 16]), _vi("gamma", [16])])
        g = _G()
        _tf_layernorm(g, "X", 16, 1, eps_name="eps_in")
        self._near_miss(g, [4, 16], "eps", inputs=[_vi("X", [4, 16]), _vi("eps_in", [])])

    def test_disabled_flag(self):
        g = _G()
        _tf_layernorm(g, "X", 16, 1)
        p = self.save(g, [4, 16])
        with self.assertRaisesRegex(SchedulerError, "ReduceMean"):
            OnnxGraph(p, fuse_patterns=False)
        self.assertEqual(OnnxGraph(p).fusion_counts["layernorm"], 1)   # library default ON


class TestGeluFusion(_Base):

    def _run(self, g, shape=(4, 64), **kw):
        p = self.save(g, list(shape))
        x = _grid(np.random.default_rng(2), shape, -9, 9)
        graph, out = self.sim(p, {"X": x}, **kw)
        return graph, x, out["Y"]

    def test_tanh_all_tails_bit_exact(self):
        for tail in ("bert", "xa_half", "halfx_a"):
            for swap in (False, True):
                for cube in ("pow", "mul"):
                    g = _G()
                    _gelu_tanh(g, "X", tail=tail, swap=swap, cube=cube)
                    graph, x, y = self._run(g)
                    self.assertEqual([type(sn) for sn in graph.nodes], [GeluNode], (tail, swap))
                    sn = graph.nodes[0]
                    self.assertEqual(sn.approximate, "tanh")
                    self.assertEqual(sn.c1, float(np.float32(0.044715)))     # graph constant
                    self.assertEqual(sn.c2, float(np.float32(math.sqrt(2 / math.pi))))
                    c1, c2 = np.float64(np.float32(0.044715)), np.float64(
                        np.float32(math.sqrt(2 / math.pi)))
                    # the BERT graph op by op (pow, mul, add, mul, tanh, add, mul, mul),
                    # tanh from libm like the generated C
                    tanh = np.vectorize(math.tanh, otypes=[np.float64])
                    ref = x * (0.5 * (1.0 + tanh(c2 * (x + c1 * np.power(x, 3.0)))))
                    np.testing.assert_array_equal(y, DT.host_quantize(ref), (tail, swap, cube))

    def test_erf_forms(self):
        erf = np.vectorize(math.erf)
        for mul in (False, True):
            for tail in ("xa_half", "halfx_a", "bert"):
                g = _G()
                _gelu_erf(g, "X", mul=mul, tail=tail)
                graph, x, y = self._run(g)
                self.assertEqual([type(sn) for sn in graph.nodes], [GeluNode])
                sn = graph.nodes[0]
                self.assertEqual((sn.approximate, sn.div), ("none", 0 if mul else 1))
                k = np.float64(np.float32(1 / math.sqrt(2) if mul else math.sqrt(2)))
                u = x * k if mul else x / k
                np.testing.assert_array_equal(y, DT.host_quantize((x * (erf(u) + 1.0)) * 0.5))

    def test_near_misses(self):
        for why, build in [
            ("c1 = 0.05", lambda g: _gelu_tanh(g, "X", c1=0.05)),
            ("no 0.5", lambda g: _gelu_tanh(g, "X", tail="none")),
            ("erf k = 2", lambda g: _gelu_erf(g, "X", k=2.0)),
        ]:
            g = _G()
            build(g)
            p = self.save(g, [4, 64])
            with self.assertRaisesRegex(SchedulerError, "only supported inside a fused",
                                        msg=why):
                OnnxGraph(p)

    def test_intermediate_graph_output_blocks_fusion(self):
        g = _G()
        _gelu_tanh(g, "X")
        g.extra_outputs.append(_vi("t", [4, 64]))
        p = self.save(g, [4, 64])
        with self.assertRaisesRegex(SchedulerError, "'Pow' is only supported inside a fused"):
            OnnxGraph(p)


class TestConstBroadcast(_Base):

    def test_scalar_becomes_last_dim_vector(self):
        g = _G()
        g.n("Mul", ["X", g.c("s", 0.125)], "Y")
        p = self.save(g, [1, 3, 4, 16])
        graph = OnnxGraph(p)
        sn = graph.nodes[0]
        self.assertIsInstance(sn, ScheduledNode)
        self.assertEqual(graph.fusion_counts["const_bcast"], 1)
        self.assertEqual(sn.inputs[1].shape, [16])
        self.assertEqual((sn.outer_count, sn.chunk_size), (12, 16))
        x = _grid(np.random.default_rng(3), (1, 3, 4, 16), -8, 8)
        _, out = self.sim(p, {"X": x})
        np.testing.assert_array_equal(out["Y"], DT.truncate(x * 0.125))
        # without the flag: one-element chunks at stride 8 (the old behaviour)
        self.assertEqual(OnnxGraph(p, fuse_patterns=False).nodes[0].chunk_size, 1)

    def test_unaligned_last_dim_left_alone(self):
        g = _G()
        g.n("Add", ["X", g.c("s", 1.0)], "Y")
        p = self.save(g, [4, 6])
        self.assertEqual(OnnxGraph(p).fusion_counts["const_bcast"], 0)

    def test_ones_times_mask_prebroadcast(self):
        g = _G()
        g.n("Mul", [g.c("ones", np.ones((1, 8, 1))), "M"], "Y")
        p = self.save(g, None, inputs=[_vi("M", [1, 1, 8])], out_shape=[1, 8, 8])
        with self.assertRaisesRegex(SchedulerError, "broadcast"):
            OnnxGraph(p, fuse_patterns=False)
        graph = OnnxGraph(p)
        sn = graph.nodes[0]
        self.assertEqual(sn.inputs[0].shape, [1, 8, 8])
        self.assertEqual((sn.outer_count, sn.a_advances, sn.b_advances), (8, True, False))
        m = _grid(np.random.default_rng(4), (1, 1, 8), -4, 4)
        _, out = self.sim(p, {"M": m})
        np.testing.assert_array_equal(out["Y"], np.broadcast_to(m, (1, 8, 8)))


class TestLowerings(_Base):

    def test_constant_nodes_folded(self):
        g = _G()
        g.nodes.append(oh.make_node("Constant", [], ["k"], value=_c("kv", [2.0] * 8)))
        g.n("Mul", ["X", "k"], "Y")
        p = self.save(g, [3, 8])
        graph = OnnxGraph(p)
        self.assertEqual(graph.constant_nodes_folded, 1)
        self.assertEqual([sn.onnx_node.op_type for sn in graph.nodes], ["Mul"])
        self.assertTrue(graph.nodes[0].inputs[1].is_weight)

    def test_split_lowering_sizes(self):
        for opset, extra, attrs in [
                (12, [], {"split": [3, 5]}),
                (13, ["sp"], {}),
                (18, [], {"num_outputs": 2})]:
            g = _G()
            ins = ["X"] + extra
            if extra:
                g.c("sp", [3, 5], np.int64)
            if opset == 18:
                shape_b = [2, 4]
                g.nodes.append(oh.make_node("Split", ins, ["A", "B"], axis=1, **attrs))
            else:
                shape_b = [2, 5]
                g.nodes.append(oh.make_node("Split", ins, ["A", "B"], axis=1, **attrs))
            g.n("Relu", ["A"], "Y")
            g.extra_outputs.append(_vi("B", shape_b))
            p = self.save(g, [2, 8], opset=opset,
                          out_shape=[2, 4] if opset == 18 else [2, 3])
            graph = OnnxGraph(p)
            self.assertEqual(graph.split_lowered_count, 1)
            slices = [sn for sn in graph.nodes if isinstance(sn, SliceNode)]
            self.assertEqual([sn.output.shape for sn in slices],
                             [[2, 4], [2, 4]] if opset == 18 else [[2, 3], [2, 5]])


if __name__ == "__main__":
    unittest.main()
