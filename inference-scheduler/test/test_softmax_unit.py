"""VectorOPKernel's softmax unit in the scheduler (doc/plans/SOFTMAX_PLAN.md §2.3,
src/smx_nodes.py):

  * the specification (src/vectorop_smx.py): register packing, the table
    against the RTL's ROM, the error against the exact softmax;
  * ONNX Softmax -> SoftmaxVopNode (row mode) where the platform has the unit,
    the host op otherwise; the simulation equals the specification; the
    generated C programs run_softmax and checks the IP;
  * the tiny ViT with vsmx: VitAttnSoftmaxVopNode per head, the simulation
    equals vlm_study's pow2+p12+vgelu+vsmx bit for bit;
  * the tiny Llama with vsmx: LlmAttnSoftmaxVopNode per KV group in prefill
    (decode keeps the host softmax), the simulation equals llm_study's
    pow2+sink+p12+vsmx bit for bit over a chat-like call sequence;
  * every generated program, compiled against the software kernels
    (test/host_emu.py), reproduces its simulation.
"""

import os
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

import gen_llama_models as G                                          # noqa: E402
import host_emu                                                       # noqa: E402
from src import _vectorop_hw_config, smx_nodes, vectorop_smx as smx   # noqa: E402
from src.codegen import CodeGenerator                                 # noqa: E402
from src.graph import OnnxGraph                                       # noqa: E402
from src.host_nodes import SoftmaxNode                                # noqa: E402
from src.llm_nodes import LlmAttnSoftmaxNode                          # noqa: E402
from src.nodes import SchedulerError                                  # noqa: E402
from src.vit import VitFrontend, patches                              # noqa: E402

_ROM = os.path.join(_REPO, "kernels", "vectorop_rtl", "rtl", "vo_smx_rom.sv")
VSMX_POLICY = "pow2+sink+p12+vsmx"


def _unit(on: bool):
    """Patch whether the platform's VectorOPKernel has the softmax unit."""
    return mock.patch.object(_vectorop_hw_config, "VECTOROP_SOFTMAX", on)


def _exact(x, valid, sigma, f_s, f_p):
    """The exact softmax of raw scores, rounded to P's grid."""
    x = np.asarray(x, np.float64) * sigma * 2.0 ** -f_s
    mask = np.arange(x.shape[1])[None, :] < np.asarray(valid)[:, None]
    e = np.where(mask, np.exp(x - np.where(mask, x, -np.inf).max(1, keepdims=True)), 0.0)
    return np.minimum(np.round(e / e.sum(1, keepdims=True) * 2.0 ** f_p), 32767)


class TestSpec(unittest.TestCase):
    def test_registers(self):
        self.assertEqual(smx_nodes.cfg_reg(19, 8), 0x0813)
        self.assertEqual(smx_nodes.mask_reg(37, 16), 37 | (16 << 16))
        cm, cfg = smx_nodes.regs(8, 1.0, 8)
        self.assertEqual((cm, cfg), (12102203, 0x0813))
        for f_s, sigma in ((8, 1.0), (13, 0.125), (4, 0.25), (0, 1.0)):
            cm, cs = smx.scale_regs(f_s, sigma)
            self.assertTrue(1 << 23 <= cm < 1 << 24)
            c = np.log2(np.e) * sigma * 2.0 ** -f_s * 4096
            self.assertAlmostEqual(cm * 2.0 ** -cs / c, 1.0, delta=2.0 ** -23)

    def test_table_is_the_rtl_rom(self):
        rom = {}
        with open(_ROM) as f:
            for line in f:
                for part in line.split(";"):
                    part = part.strip()
                    if part.startswith("rom["):
                        k, v = part[4:].split("] = 17'd")
                        rom[int(k)] = int(v)
        self.assertEqual([rom[k] for k in range(4096)], [int(v) for v in smx.TAB])

    def test_within_one_lsb_of_exact(self):
        rng = np.random.default_rng(3)
        for f_s, sigma, f_p, n in ((8, 1.0, 8, 256), (12, 0.125, 12, 64), (10, 0.25, 15, 1000)):
            x = np.clip(np.round(rng.normal(0, 3, (40, n)) * 2.0 ** f_s), -32768, 32767)
            valid = rng.integers(1, n + 1, 40)
            cm, cs = smx.scale_regs(f_s, sigma)
            got = smx.softmax_raw(x.astype(np.int64), valid, cm, cs, f_p)
            want = _exact(x, valid, sigma, f_s, f_p)
            self.assertLessEqual(np.abs(got - want).max(), 1, (f_s, sigma, f_p))
            np.testing.assert_array_equal(got[np.arange(n)[None, :] >= valid[:, None]], 0)


def _softmax_model(path, shape, opset=13):
    g = oh.make_graph([oh.make_node("Softmax", ["X"], ["Y"], axis=-1)], "smx",
                      [oh.make_tensor_value_info("X", TensorProto.FLOAT, shape)],
                      [oh.make_tensor_value_info("Y", TensorProto.FLOAT, shape)])
    m = oh.make_model(g, opset_imports=[oh.make_opsetid("", opset)])
    m.ir_version = 9
    onnx.save(m, path)
    return path


class TestOnnxSoftmax(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.path = {name: _softmax_model(os.path.join(cls._tmp.name, f"{name}.onnx"), shape)
                    for name, shape in (("bert", [1, 2, 16, 64]), ("odd", [3, 12]),
                                        ("one_row", [1, 13]), ("long", [2, 2056]))}

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_mapping(self):
        with _unit(True):
            kinds = {n: type(OnnxGraph(p).nodes[0]) for n, p in self.path.items()}
        self.assertEqual(kinds, {"bert": smx_nodes.SoftmaxVopNode, "odd": SoftmaxNode,
                                 "one_row": smx_nodes.SoftmaxVopNode, "long": SoftmaxNode})
        with _unit(False):
            self.assertIsInstance(OnnxGraph(self.path["bert"]).nodes[0], SoftmaxNode)

    def test_simulation_is_the_spec(self):
        rng = np.random.default_rng(5)
        with _unit(True):
            g = OnnxGraph(self.path["bert"])
            cg = CodeGenerator(g, model_path="smx.onnx")
            sn = g.nodes[0]
            calls = sn.kernel_calls(cg._compute_tensor_layouts())
        self.assertEqual([(c.fields["op"], c.fields["size"], c.fields["outer"], c.fields["a_inc"],
                           c.fields["b_inc"]) for c in calls], [(10, 64, 32, 64, 64)])
        x = np.round(rng.normal(0, 3, (1, 2, 16, 64)) * 256) / 256
        y = cg.simulate({"X": x})["Y"]
        want = smx.softmax_raw(np.rint(x.reshape(32, 64) * 256).astype(np.int64), 64,
                               12102203, 19, 8) / 256.0
        np.testing.assert_array_equal(y.reshape(32, 64), want)

    def test_generated_source(self):
        with _unit(True):
            src = CodeGenerator(OnnxGraph(self.path["bert"]), model_path="smx.onnx").generate_source()
        self.assertIn("static void run_softmax(", src)
        self.assertIn("run_softmax(X, 0u, Y, 0u, 64u, VECTOROP_SOFTMAX, 32u, 64u, 64u,", src)
        self.assertIn("#define VECTOROP_SOFTMAX_T", src)
        self.assertIn("has no softmax", src)
        with _unit(False):
            src = CodeGenerator(OnnxGraph(self.path["bert"]), model_path="smx.onnx").generate_source()
        self.assertNotIn("VECTOROP_SOFTMAX", src)

    @unittest.skipUnless(host_emu.which_cc(), "C compiler not available on host")
    def test_host_emulation(self):
        for name in ("bert", "one_row"):
            with _unit(True):
                cg = CodeGenerator(OnnxGraph(self.path[name]), model_path=f"{name}.onnx")
            with self.subTest(model=name), tempfile.TemporaryDirectory() as td:
                rc, out = host_emu.build_and_run(cg, td, incoherent=True)
                self.assertEqual(rc, 0, out[-3000:])
                self.assertIn("test_inference PASSED", out)


@unittest.skipUnless(host_emu.which_cc(), "C compiler not available on host")
class TestSuiteModels(unittest.TestCase):
    """test/gen_softmax_models.py's models (the board suite's smx_*): with the
    unit, the generated C in the emulator equals the simulation."""

    def test_host_emulation(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "gen_softmax_models", os.path.join(_HERE, "gen_softmax_models.py"))
        gen = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gen)
        with tempfile.TemporaryDirectory() as md:
            gen.OUT_DIR = md
            with open(os.devnull, "w") as dn, mock.patch("sys.stdout", dn):
                gen.main()
            for f in sorted(os.listdir(md)):
                with _unit(True):
                    g = OnnxGraph(os.path.join(md, f), fuse_act=True)
                    cg = CodeGenerator(g, model_path=f)
                self.assertTrue(any(isinstance(sn, smx_nodes.SoftmaxVopNode) for sn in g.nodes), f)
                with self.subTest(model=f), tempfile.TemporaryDirectory() as td:
                    rc, out = host_emu.build_and_run(cg, td)
                    self.assertEqual(rc, 0, out[-3000:])
                    self.assertIn("test_inference PASSED", out)


# ------------------------------------------------------------------ #
# The tiny ViT                                                          #
# ------------------------------------------------------------------ #

class TestVision(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import test_vit as TV
        cls.TV = TV
        cfg, W, _vm, fe = TV.Tiny.get()
        cls.cfg = cfg
        cls.fe = VitFrontend(fe.cfg, fe.W, fe.fmt, name="vit_tiny", conn_k=TV.CONN_K, vsmx=True)
        vs = TV.vs
        Wv = {k[len("model."):]: v for k, v in W.items()}
        vc = vs.VCfg(TV.TINY_CFG["vision_config"])
        KW = vs.kernel_weights(Wv, vc)
        host = TV.Tiny.host_study()
        cls.vm = vs.VisionModel(Wv, KW, vc, TV.TINY_CFG["scale_factor"],
                                vs.VPOLICIES["pow2+p12+vgelu+vsmx"], host.fmt)
        cls.vm.conn_k = TV.CONN_K

    def test_simulation_equals_study(self):
        with _unit(True):
            g = OnnxGraph(self.fe.entry(output=True), fuse_act=True, s2d_stem=True)
            cg = CodeGenerator(g, model_path="vit_tiny_vsmx.onnx")
        vop = [sn for sn in g.nodes if isinstance(sn, smx_nodes.VitAttnSoftmaxVopNode)]
        self.assertEqual(len(vop), 2 * 4)
        self.assertEqual([(c.fields["op"], c.fields["size"], c.fields["outer"])
                          for c in vop[0].kernel_calls({})], [(11, 64, 64)])
        for im in self.TV.images(2, 7):
            out = cg._forward_pass({"vision.patches": patches(im, self.cfg.P).astype(np.float64)},
                                   states=cg.initial_states())
            np.testing.assert_array_equal(out["vision.image"], self.vm.forward(im))
        self.assertFalse(np.array_equal(out["vision.image"], self.TV.Tiny.get()[2].forward(im)))
        if host_emu.which_cc():
            with tempfile.TemporaryDirectory() as td:
                rc, log = host_emu.build_and_run(cg, td, cached=True, threads=3, min_elems=1,
                                                 incoherent=True)
                self.assertEqual(rc, 0, log[-3000:])
                self.assertIn("test_inference PASSED", log)

    def test_needs_the_unit(self):
        with _unit(False), self.assertRaises(SchedulerError):
            OnnxGraph(self.fe.entry(), fuse_act=True, s2d_stem=True)


# ------------------------------------------------------------------ #
# The tiny Llama                                                        #
# ------------------------------------------------------------------ #

class TestLlamaPrefill(unittest.TestCase):
    """Policy pow2+sink+p12+vsmx: the shipped FPGA attention (decode included)
    with the prefill softmax on the unit; buckets of 16 rows."""

    @classmethod
    def setUpClass(cls):
        import test_llama as TL
        cls.TL = TL
        cfg, W, formats, _fe = TL.tiny()
        cls.fe = G.frontend(cfg, W, formats, decode_attn="fpga", vsmx=True)
        with _unit(True):
            gs = TL.graphs(cls.fe, {"decode": ("decode", 1, False), "prefill_16": ("prefill", 16, False),
                                    "head": ("head", 1, False)})
        cls.gs = gs
        cls.cgs = {n: CodeGenerator(g, model_path=n) for n, g in gs.items()}
        cls.sm, cls.sc = G.study_model(cfg, W, formats, policy=VSMX_POLICY)

    def test_structure(self):
        pre = [type(sn) for sn in self.gs["prefill_16"].nodes]
        dec = [type(sn) for sn in self.gs["decode"].nodes]
        self.assertEqual(pre.count(smx_nodes.LlmAttnSoftmaxVopNode), 2 * 2)   # layers x KV groups
        self.assertNotIn(LlmAttnSoftmaxNode, pre)
        self.assertEqual(dec.count(LlmAttnSoftmaxNode), 2 * 2)
        self.assertNotIn(smx_nodes.LlmAttnSoftmaxVopNode, dec)
        sn = next(sn for sn in self.gs["prefill_16"].nodes
                  if isinstance(sn, smx_nodes.LlmAttnSoftmaxVopNode))
        self.assertEqual(len(sn.kernel_calls({})), 2)                         # one per head of the group
        src = self.cgs["prefill_16"].generate_source()
        self.assertIn("_mask = ((_pos + 1u) & 0xFFFFu) | (16u << 16)", src)
        self.assertIn("VECTOROP_SOFTMAX_T", src)
        with self.assertRaises(ValueError):
            self.fe.entry("prefill", 8)

    def test_chat_sequence_equals_study(self):
        rng = np.random.default_rng(21)
        p1, p2 = rng.integers(2, 256, 13), rng.integers(2, 256, 5)
        steps = [("prefill", [1] + list(p1)), ("decode", 7), ("decode", 9),
                 ("prefill", list(p2)), ("decode", 3), ("truncate", 6), ("prefill", list(p2)),
                 ("decode", 4)]
        b = self.TL.study_steps(self.sm, self.sc, steps)
        a = self.TL.sched_steps(self.TL.Session(self.cgs, [16]), steps)
        self.assertEqual(len(a), len(b))
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            np.testing.assert_array_equal(x, y, err_msg=f"step {i} {steps[i]}")

    @unittest.skipUnless(host_emu.which_cc(), "C compiler not available on host")
    def test_host_emulation(self):
        for name in ("prefill_16", "decode"):
            with self.subTest(entry=name), tempfile.TemporaryDirectory() as td:
                rc, out = host_emu.build_and_run(self.cgs[name], td, cached=True, threads=2,
                                                 min_elems=1, incoherent=True)
                self.assertEqual(rc, 0, out[-3000:])
                self.assertIn("test_inference PASSED", out)


if __name__ == "__main__":
    unittest.main()
