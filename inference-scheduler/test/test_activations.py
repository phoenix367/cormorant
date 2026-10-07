"""VectorOPKernel's activation unit in the scheduler (doc/plans/ACTIVATIONS_PLAN.md):
ONNX Gelu / LeakyRelu / x * Sigmoid(x) -> VectorOP ops 6-9 and acts 3-6.

The models are test/gen_activation_models.py's, written into a temporary
directory (no generator run needed).
"""

import importlib.util
import os
import re
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import onnx
import onnx.helper as oh
from onnx import TensorProto

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_REPO = os.path.dirname(_ROOT)
sys.path.insert(0, _HERE)

import host_emu                                                       # noqa: E402
from src import _vectorop_hw_config, vectorop_act                     # noqa: E402
from src.codegen import CodeGenerator                                 # noqa: E402
from src.dtype import AP_FIXED_16_8, FLOAT32                          # noqa: E402
from src.graph import OnnxGraph                                       # noqa: E402
from src.host_nodes import GeluNode                                   # noqa: E402
from src.nodes import (ACT_GELU, ACT_GELU_TANH, ACT_LEAKY_RELU, ACT_NONE,  # noqa: E402
                       ACT_SILU, OP_ADD, OP_DIV, OP_GELU, OP_GELU_TANH, OP_LEAKY_RELU,
                       OP_MUL, OP_SILU, ScheduledNode, SchedulerError, job_act)
from src.tensor import TensorInfo                                     # noqa: E402

_ROM = os.path.join(_REPO, "kernels", "vectorop_rtl", "rtl", "vo_act_rom.sv")


def _unit(on: bool):
    """Patch whether the platform's VectorOPKernel has the activation unit."""
    return mock.patch.object(_vectorop_hw_config, "VECTOROP_ACTIVATIONS", on)


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "gen_activation_models", os.path.join(_HERE, "gen_activation_models.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _model(path, nodes, inputs, outputs, opset=20):
    g = oh.make_graph(nodes, "m", inputs, outputs)
    m = oh.make_model(g, opset_imports=[oh.make_opsetid("", opset)])
    m.ir_version = 9
    onnx.save(m, path)
    return path


def _vi(name, shape):
    return oh.make_tensor_value_info(name, TensorProto.FLOAT, shape)


class _Models(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        gen = _load_generator()
        gen.OUT_DIR = cls._tmp.name
        with open(os.devnull, "w") as dn, mock.patch("sys.stdout", dn):
            gen.main()
        cls.path = {f[:-5]: os.path.join(cls._tmp.name, f)
                    for f in os.listdir(cls._tmp.name) if f.endswith(".onnx")}

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def gen(self, name_or_path, on=True, **kw):
        path = self.path.get(name_or_path, name_or_path)
        kw.setdefault("fuse_act", True)
        with _unit(on):
            g = OnnxGraph(path, **kw)
            return g, CodeGenerator(g, model_path=path)


class TestTables(unittest.TestCase):
    """The scheduler's activation semantics against the RTL kernel's table."""

    def test_tables_equal_the_rtl_rom(self):
        text = open(_ROM).read()
        rom = np.zeros(4096, np.int64)
        for a, v in re.findall(r"rom\[\s*(\d+)\] = 7'd(\d+)", text):
            rom[int(a)] = int(v)
        seg = {name: (int(b), int(n)) for name, b, n in re.findall(
            r"(\w+)_BASE\s*=\s*12'd(\d+),\s*\w+_LEN\s*=\s*12'd(\d+);", text)}
        raw = np.arange(-32768, 32768, dtype=np.int64)
        for name, act in (("SILU", ACT_SILU), ("GELU", ACT_GELU), ("GELU_TANH", ACT_GELU_TANH)):
            base, n = seg[name]
            t = np.abs(raw)
            m = np.where(t < n, rom[base + np.minimum(t, n - 1)], 0)
            rtl = np.maximum(raw, 0) - m
            got = np.round(vectorop_act.kernel_table(act) * 256).astype(np.int64)
            np.testing.assert_array_equal(got, rtl, err_msg=name)

    def test_leaky_relu_rounds_half_to_even(self):
        x = np.array([-1, -3, -5, -32768, 7], np.float64) / 256
        y = vectorop_act.apply(ACT_LEAKY_RELU, x, AP_FIXED_16_8, alpha=0x8000)
        np.testing.assert_array_equal(y * 256, [0, -2, -2, -16384, 7])
        y = vectorop_act.apply(ACT_LEAKY_RELU, x[3:4], AP_FIXED_16_8, alpha=0xFFFF)
        np.testing.assert_array_equal(y * 256, [-32768])      # -32767.5 -> even

    def test_leaky_alpha_quantisation(self):
        self.assertEqual(vectorop_act.leaky_alpha(0.01, "n"), 655)
        self.assertEqual(vectorop_act.leaky_alpha(0.1, "n"), 6554)
        self.assertEqual(vectorop_act.leaky_alpha(0.0, "n"), 0)
        for bad in (1.0, 1.5, -0.1):
            with self.assertRaises(SchedulerError):
                vectorop_act.leaky_alpha(bad, "n")

    def test_job_act(self):
        self.assertEqual(job_act(OP_GELU, ACT_LEAKY_RELU), ACT_GELU)
        self.assertEqual(job_act(OP_LEAKY_RELU, ACT_NONE), ACT_LEAKY_RELU)
        self.assertEqual(job_act(OP_ADD, ACT_SILU), ACT_SILU)

    def test_gelu_with_other_constants_stays_on_the_host(self):
        node = oh.make_node("Gelu", ["x"], ["y"], approximate="none", axi_k="1.0", axi_div=1)
        tensors = {"x": TensorInfo("x", [1, 64], "float32"), "y": TensorInfo("y", [1, 64], "float32")}
        gelu = GeluNode.from_onnx_node(node, tensors, 0, 8, None)
        self.assertIsNone(vectorop_act.gelu_act(gelu, AP_FIXED_16_8))
        self.assertIsNone(vectorop_act.gelu_act(gelu, FLOAT32))
        node = oh.make_node("Gelu", ["x"], ["y"], approximate="none")
        self.assertEqual(vectorop_act.gelu_act(
            GeluNode.from_onnx_node(node, tensors, 0, 8, None), AP_FIXED_16_8), ACT_GELU)

    def test_exponent_tensors_are_not_eligible(self):
        node = oh.make_node("LeakyRelu", ["x"], ["y"])
        tensors = {"x": TensorInfo("x", [8], "float32", exp=np.array(10)),
                   "y": TensorInfo("y", [8], "float32")}
        with _unit(True):
            self.assertIn("exponent", vectorop_act._eligible(node, tensors, AP_FIXED_16_8))
            tensors["x"].exp = None
            self.assertIsNone(vectorop_act._eligible(node, tensors, AP_FIXED_16_8))
            self.assertIn("ap_fixed<16,8>", vectorop_act._eligible(node, tensors, FLOAT32))
        with _unit(False):
            self.assertIn("activation unit", vectorop_act._eligible(node, tensors, AP_FIXED_16_8))


class TestMapping(_Models):

    def _vop(self, g):
        return [(sn.op_code, sn.act, sn.alpha) for sn in g.nodes]

    def test_ops(self):
        cases = {"act_gelu": [(OP_GELU, ACT_NONE, 0)],
                 "act_gelu_tanh": [(OP_GELU_TANH, ACT_NONE, 0)],
                 "act_leaky_relu": [(OP_LEAKY_RELU, ACT_NONE, 6554)],
                 "act_silu": [(OP_SILU, ACT_NONE, 0)]}
        for name, want in cases.items():
            g, _ = self.gen(name)
            self.assertTrue(all(isinstance(sn, ScheduledNode) for sn in g.nodes), name)
            self.assertEqual(self._vop(g), want, name)
        g, _ = self.gen("act_silu")
        self.assertEqual(g.fusion_counts["silu"], 1)

    def test_fused_after_an_op(self):
        for name, want in {"act_bias_gelu": [(OP_ADD, ACT_GELU, 0)],
                           "act_mul_leaky_relu": [(OP_MUL, ACT_LEAKY_RELU, 655)],
                           "act_div_silu": [(OP_DIV, ACT_SILU, 0)]}.items():
            g, _ = self.gen(name)
            self.assertEqual(self._vop(g), want, name)
            self.assertEqual(g.act_fused_count, 1, name)
            g, _ = self.gen(name, fuse_act=False)
            self.assertEqual(len(g.nodes), 2, name)

    def test_nothing_fused_into_an_activation_op(self):
        g, _ = self.gen("act_gelu_leaky_chain")
        self.assertEqual(self._vop(g), [(OP_GELU_TANH, ACT_NONE, 0),
                                        (OP_LEAKY_RELU, ACT_NONE, 13107)])

    def test_without_the_unit(self):
        g, _ = self.gen("act_gelu", on=False)
        self.assertEqual([type(sn) for sn in g.nodes], [GeluNode])
        g, _ = self.gen("act_bias_gelu", on=False)
        self.assertEqual([type(sn).__name__ for sn in g.nodes], ["ScheduledNode", "GeluNode"])
        with self.assertRaisesRegex(SchedulerError, "activation unit"):
            self.gen("act_leaky_relu", on=False)
        with self.assertRaisesRegex(SchedulerError, "Sigmoid.*SiLU"):
            self.gen("act_silu", on=False)

    def test_env_override(self):
        env = dict(os.environ, AXI_VECTOROP_ACTIVATIONS="0")
        r = subprocess.run([sys.executable, "-c",
                            "from src import _vectorop_hw_config as c; print(c.VECTOROP_ACTIVATIONS)"],
                           cwd=_ROOT, env=env, capture_output=True, text=True)
        self.assertEqual(r.stdout.strip(), "False", r.stderr)
        self.assertTrue(_vectorop_hw_config.resolve()["VECTOROP_ACTIVATIONS"] or
                        os.environ.get("AXI_VECTOROP_ACTIVATIONS") == "0")

    def test_sigmoid_with_another_consumer_is_not_silu(self):
        with tempfile.TemporaryDirectory() as td:
            p = _model(os.path.join(td, "m.onnx"),
                       [oh.make_node("Sigmoid", ["X"], ["S"]),
                        oh.make_node("Mul", ["X", "S"], ["Y"]),
                        oh.make_node("Add", ["S", "X"], ["Z"])],
                       [_vi("X", [1, 64])], [_vi("Y", [1, 64]), _vi("Z", [1, 64])])
            with self.assertRaisesRegex(SchedulerError, "Sigmoid"):
                self.gen(p)
            p = _model(os.path.join(td, "n.onnx"),                # Mul(S, X) order
                       [oh.make_node("Sigmoid", ["X"], ["S"]),
                        oh.make_node("Mul", ["S", "X"], ["Y"])],
                       [_vi("X", [1, 64])], [_vi("Y", [1, 64])])
            g, _ = self.gen(p)
            self.assertEqual(self._vop(g), [(OP_SILU, ACT_NONE, 0)])


class TestCode(_Models):

    def test_calls_and_alpha(self):
        _, cg = self.gen("act_mul_leaky_relu")
        src = cg.generate_source()
        self.assertIn("VECTOROP_MUL, 1u, 0u, 0u, VECTOROP_ACT_LEAKY_RELU, 655u);", src)
        self.assertIn("XVectoropkernel_Set_alpha(&s_vectoropkernel, alpha);", src)
        _, cg = self.gen("act_gelu")
        src = cg.generate_source()
        self.assertIn("run_op(X, NULL, Y, 65536u, VECTOROP_GELU, 1u, 0u, 0u);", src)
        self.assertIn("#define VECTOROP_GELU        8u", src)

    def test_init_checks_the_unit(self):
        _, cg = self.gen("act_silu")
        src = cg.generate_source()
        init = src[src.index("int inference_init("):src.index("void inference_deinit(void)")]
        self.assertIn("XVectoropkernel_Set_alpha(&s_vectoropkernel, 0x5A5Au);", init)
        self.assertIn("XVectoropkernel_Get_alpha(&s_vectoropkernel) != 0x5A5Au", init)
        self.assertIn("#include <stdio.h>", src)
        # run_op() writes alpha 0 guarded; run_op_act() is not emitted here
        self.assertIn("#ifdef XVECTOROPKERNEL_CTRL_ADDR_ALPHA_DATA\n"
                      "    XVectoropkernel_Set_alpha(&s_vectoropkernel, 0u);\n#endif", src)

    def test_projects_without_the_unit_need_no_alpha_register(self):
        _, cg = self.gen("act_bias_gelu", on=False)
        src = cg.generate_source()
        self.assertNotIn("Get_alpha", src)
        self.assertNotIn("0x5A5Au", src)
        self.assertIsNone(re.search(r"^    XVectoropkernel_Set_alpha", src.replace(
            "#ifdef XVECTOROPKERNEL_CTRL_ADDR_ALPHA_DATA\n    XVectoropkernel_Set_alpha", ""), re.M))


class TestSimulation(_Models):

    def test_gelu_unit_equals_host_op(self):
        x = np.arange(-32768, 32768, dtype=np.float64).reshape(1, 65536) / 256
        for name in ("act_gelu", "act_gelu_tanh"):
            _, cg1 = self.gen(name)
            _, cg0 = self.gen(name, on=False)
            np.testing.assert_array_equal(cg1.simulate({"X": x})["Y"],
                                          cg0.simulate({"X": x})["Y"], err_msg=name)

    def test_activations_on_every_input(self):
        x = np.arange(-32768, 32768, dtype=np.float64).reshape(1, 65536) / 256
        for name, act, alpha in (("act_gelu", ACT_GELU, 0), ("act_gelu_tanh", ACT_GELU_TANH, 0),
                                 ("act_silu", ACT_SILU, 0), ("act_leaky_relu", ACT_LEAKY_RELU, 6554)):
            _, cg = self.gen(name)
            want = vectorop_act.apply(act, x, AP_FIXED_16_8, alpha)
            np.testing.assert_array_equal(cg.simulate({"X": x})["Y"], want, err_msg=name)

    def test_fused_equals_unfused(self):
        rng = np.random.default_rng(5)
        for name, shape in (("act_bias_gelu", (1, 64, 256)), ("act_mul_leaky_relu", (1, 4096)),
                            ("act_div_silu", (1, 1024))):
            x = np.round(rng.uniform(-6, 6, shape) * 256) / 256
            _, cg1 = self.gen(name)
            _, cg0 = self.gen(name, fuse_act=False)
            np.testing.assert_array_equal(cg1.simulate({"X": x})["Y"],
                                          cg0.simulate({"X": x})["Y"], err_msg=name)


@unittest.skipUnless(host_emu.which_cc(), "C compiler not available on host")
class TestGeneratedC(_Models):

    def test_host_emulated_run_matches_simulation(self):
        """inference.c compiled against the software VectorOP model: every
        output bit-identical to the simulator (the single-op models cover all
        65 536 Q8.8 inputs through the harness's ramp fill)."""
        for name in sorted(self.path):
            _, cg = self.gen(name)
            with self.subTest(model=name), tempfile.TemporaryDirectory() as td:
                rc, out = host_emu.build_and_run(cg, td)
                self.assertEqual(rc, 0, f"{name}\n{out}")
                self.assertIn("test_inference PASSED", out)
                self.assertEqual(host_emu.failures(out), [])


if __name__ == "__main__":
    unittest.main()
