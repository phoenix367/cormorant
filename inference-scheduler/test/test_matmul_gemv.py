"""MatmulKernel GEMV streaming mode in the scheduler (src/matmul_gemv.py,
MatmulNode.gemv_kw, MATMUL_OPTIMISATION.md §8b)."""

import os
import shutil
import sys
import tempfile
import unittest

import numpy as np
import onnx
import onnx.helper as oh
from onnx import TensorProto as TP

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import host_emu  # noqa: E402
from src._matmul_hw_config import MATMUL_GEMV_MAX_M  # noqa: E402
from src.codegen import CodeGenerator  # noqa: E402
from src.graph import OnnxGraph  # noqa: E402
from src.matmul_gemv import choose_gemv, ineligible_reason  # noqa: E402
from src.nodes import MatmulNode, conv_lowered_b_image  # noqa: E402


def _model(nodes, inputs, outputs, inits):
    g = oh.make_graph(nodes, "t", inputs, outputs, initializer=inits)
    m = oh.make_model(g, opset_imports=[oh.make_opsetid("", 13)])
    m.ir_version = 8
    d = tempfile.mkdtemp()
    path = os.path.join(d, "m.onnx")
    onnx.save(m, path)
    return path


def _fc(n, k, m, seed=0, extra_reader=False):
    """A[n,k] @ W[k,m] (+ a second MatMul reading W when extra_reader)."""
    rng = np.random.default_rng(seed)
    w = (rng.integers(-64, 64, (k, m)) / 64.0).astype(np.float32)
    nodes = [oh.make_node("MatMul", ["A", "W"], ["Y"])]
    outs = [oh.make_tensor_value_info("Y", TP.FLOAT, [n, m])]
    ins = [oh.make_tensor_value_info("A", TP.FLOAT, [n, k])]
    if extra_reader:
        nodes.append(oh.make_node("MatMul", ["A2", "W"], ["Y2"]))
        ins.append(oh.make_tensor_value_info("A2", TP.FLOAT, [n, k]))
        outs.append(oh.make_tensor_value_info("Y2", TP.FLOAT, [n, m]))
    return _model(nodes, ins, outs, [oh.make_tensor("W", TP.FLOAT, w.shape, w.flatten())])


def _mm(g):
    return [sn for sn in g.nodes if isinstance(sn, MatmulNode)]


@unittest.skipUnless(MATMUL_GEMV_MAX_M > 0, "platform kernel has no GEMV path")
class TestGemvSelection(unittest.TestCase):

    def test_batch1_fc_runs_as_gemv(self):
        g = OnnxGraph(_fc(1, 256, 96))
        sn, = _mm(g)
        self.assertEqual(sn.gemv_kw, 1)
        self.assertFalse(sn.b_packed)
        self.assertIsNone(sn.inputs[1].packed_data)          # plain row-major B
        self.assertEqual(g.matmul_gemv_stats["gemv"], 1)
        self.assertTrue(sn.emit_call({}).rstrip().endswith("0u, 1u);"))
        self.assertIn("GEMV kw=1", sn.emit_comment())

    def test_ineligible_shapes_stay_tiled(self):
        for n, k, m, why in ((1, 20, 10, "m"), (2, 256, 96, "n"), (1, 36, 64, "k"),
                             (1, 64, 32, "m")):
            with self.subTest(n=n, k=k, m=m):
                sn, = _mm(OnnxGraph(_fc(n, k, m)))
                self.assertEqual(sn.gemv_kw, 0)
                self.assertIn(why, ineligible_reason(sn, 1))

    def test_mode_off_and_always(self):
        sn, = _mm(OnnxGraph(_fc(1, 256, 96), matmul_gemv="off"))
        self.assertEqual(sn.gemv_kw, 0)
        self.assertTrue(sn.b_packed)
        # m * kw = 64 with the smallest eligible sizes: "always" takes it
        sn, = _mm(OnnxGraph(_fc(1, 8, 64), matmul_gemv="always"))
        self.assertEqual(sn.gemv_kw, 1)

    def test_kw_hint_reimages_b(self):
        k, m = 128, 48
        g = OnnxGraph(_fc(1, k, m), matmul_gemv_kw={"W": 4})
        sn, = _mm(g)
        self.assertEqual(sn.gemv_kw, 4)
        b = sn.inputs[1]
        np.testing.assert_array_equal(b.packed_data, conv_lowered_b_image(b.data, k, m, 4))
        self.assertEqual(g.matmul_gemv_stats["kw>1"], 1)

    def test_kw_hint_falls_back_to_plain_b(self):
        # k % (16 kw) != 0 for kw = 8: the hint cannot apply -> kw = 1
        sn, = _mm(OnnxGraph(_fc(1, 64, 64), matmul_gemv_kw={"W": 8}))
        self.assertEqual(sn.gemv_kw, 1)
        self.assertIsNone(sn.inputs[1].packed_data)
        # B read by two MatMuls: never re-imaged
        g = OnnxGraph(_fc(1, 128, 64, extra_reader=True), matmul_gemv_kw={"W": 4})
        self.assertEqual([sn.gemv_kw for sn in _mm(g)], [1, 1])
        self.assertIsNone(_mm(g)[0].inputs[1].packed_data)

    def test_gemv_b_is_never_packed(self):
        # one GEMV reader keeps the shared constant row-major for every reader
        g = OnnxGraph(_fc(1, 128, 64, extra_reader=True))
        self.assertTrue(all(sn.gemv_kw == 1 and not sn.b_packed for sn in _mm(g)))

    def test_choose_gemv_off_is_a_noop(self):
        g = OnnxGraph(_fc(1, 256, 96), matmul_gemv="off")
        st = choose_gemv(g.nodes, mode="off")
        self.assertEqual(st["gemv"], 0)

    def test_helper_sets_the_gemv_registers(self):
        cg = CodeGenerator(OnnxGraph(_fc(1, 256, 96)), model_path="fc.onnx")
        src = cg.generate_source()
        self.assertIn("uint32_t b_packed, uint32_t gemv_kw)", src)
        self.assertIn("XMatmulkernel_Set_gemv_kw(&", src)
        self.assertIn("XMatmulkernel_Set_a_to_b(&", src)
        self.assertIn("(uint64_t)((inference_buf_phys(b)) - (inference_buf_phys(a)))", src)


@unittest.skipUnless(MATMUL_GEMV_MAX_M > 0 and shutil.which("cc"), "needs GEMV and cc")
class TestGemvHostEmulation(unittest.TestCase):
    """Generated C against the software MatmulKernel (GEMV image, a_to_b
    checked) == the scheduler's simulation."""

    def _run(self, g, name):
        cg = CodeGenerator(g, model_path=f"{name}.onnx")
        with tempfile.TemporaryDirectory() as td:
            rc, out = host_emu.build_and_run(cg, td, incoherent=True)
        self.assertEqual(rc, 0, out[-3000:])
        self.assertIn("test_inference PASSED", out)

    def test_plain_b(self):
        self._run(OnnxGraph(_fc(1, 256, 96, seed=1)), "gemv_plain")

    def test_conv_image_kw4(self):
        g = OnnxGraph(_fc(1, 128, 48, seed=2), matmul_gemv_kw={"W": 4})
        self.assertEqual(_mm(g)[0].gemv_kw, 4)
        self._run(g, "gemv_kw4")


if __name__ == "__main__":
    unittest.main(verbosity=2)
