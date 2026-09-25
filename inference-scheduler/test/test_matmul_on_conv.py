"""MatMul on ConvKernel with swapped operand roles (doc/BERT_PLAN.md §2 2A,
src/matmul_lowering.py, nodes.MatmulConvNode).

For C[N][M] = A[N][K] · B[K][M] the scheduler issues a ConvKernel call with
out_ch = N, in_ch = K/kw, a 1 x kw kernel with stride (1, kw) and an
out_h x out_w = M output, A (row-major) as the conv weight and B — as is for
kw = 1, re-laid out at codegen for a constant B and kw > 1 — as the conv
input.  Checked here:

  * the layout function against an explicit loop, and the whole mapping
    against the conv reference: A read through ConvKernel.h's packed-weight
    formula and B's image read as NCHW x give exactly A · B;
  * the engine choice (BERT-shaped linears and attention lowered, FC /
    misaligned / tiny MatMuls kept, modes auto / always / off);
  * the simulator is the same for both engines (bit-identical outputs with
    the lowering on and off), and so are the emitted weight images;
  * the generated C: run_conv / run_conv_at emission, -Werror compile, and a
    host run against software kernel models that reproduces the simulation
    bit for bit;
  * the scheduler's cost model equals the conv-cycle-model skill script.

Models are built inline (onnx.helper) so no generator run is needed.
"""

import importlib.util
import os
import random
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as nph
from onnx import TensorProto

import host_emu
from src.codegen import CodeGenerator
from src.codegen._simulate import _conv2d_ref
from src.cost_model import conv_cycles, matmul_cycles
from src.graph import OnnxGraph
from src.matmul_lowering import (LOWER_MARGIN, conv_plans, ineligible_reason,
                                 normalize_mode)
from src.nodes import (MatmulConvNode, MatmulNode, ScheduledNode, _pack_conv_weight,
                       conv_lowered_b_image)
from src.report import ReportGenerator
from src.tensor import TensorInfo
from src._conv_hw_config import CONV_TILE_IC

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_SKILL = os.path.join(os.path.dirname(_ROOT), ".claude", "skills", "conv-cycle-model",
                      "scripts", "conv_cycle_model.py")


def _grid(rng, shape, scale):
    """Random values on the ap_fixed<16,8> grid (multiples of 1/256)."""
    return np.round(rng.uniform(-scale, scale, shape) * 256) / 256


def _f32(name, shape):
    return oh.make_tensor_value_info(name, TensorProto.FLOAT, shape)


def _save(path, nodes, inputs, outputs, inits):
    graph = oh.make_graph(nodes, os.path.basename(path), inputs, outputs,
                          initializer=[nph.from_array(np.asarray(v, np.float32), name=k)
                                       for k, v in inits.items()])
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return path


def _matmul_model(path, a_shape, b_shape, *, a_const=False, b_const=True, seed=0,
                  scale=0.5, relu_after=False):
    """Y = MatMul(A, B) (optionally -> Relu); constants are initializers."""
    rng = np.random.default_rng(seed)
    out_shape = list(np.broadcast_shapes(tuple(a_shape[:-2]), tuple(b_shape[:-2]))) \
        + [a_shape[-2], b_shape[-1]]
    inputs, inits = [], {}
    for name, shape, const in (("A", a_shape, a_const), ("B", b_shape, b_const)):
        if const:
            inits[name] = _grid(rng, shape, scale)
        else:
            inputs.append(_f32(name, shape))
    nodes = [oh.make_node("MatMul", ["A", "B"], ["MM" if relu_after else "Y"], name="mm")]
    if relu_after:
        nodes.append(oh.make_node("Relu", ["MM"], ["Y"], name="relu"))
    return _save(path, nodes, inputs, [_f32("Y", out_shape)], inits)


def _gemm_model(path, n, k, m, seed=0):
    """Y = Gemm(X, W^T, b) as exported by TF / PyTorch (transB = 1)."""
    rng = np.random.default_rng(seed)
    inits = {"W": _grid(rng, (m, k), 0.25), "bias": _grid(rng, (m,), 0.25)}
    nodes = [oh.make_node("Gemm", ["X", "W", "bias"], ["Y"], name="gemm", transB=1)]
    return _save(path, nodes, [_f32("X", [n, k])], [_f32("Y", [n, m])], inits)


def _attention_model(path, heads, s, d):
    """scores = Q·K^T, ctx = softmax-free P·V stand-in: two batched MatMuls of
    activations, both operands advancing per head (BERT's attention)."""
    nodes = [oh.make_node("MatMul", ["Q", "KT"], ["S"], name="qk"),
             oh.make_node("MatMul", ["S", "V"], ["Y"], name="pv")]
    return _save(path, nodes,
                 [_f32("Q", [1, heads, s, d]), _f32("KT", [1, heads, d, s]),
                  _f32("V", [1, heads, s, d])],
                 [_f32("Y", [1, heads, s, d])], {})


def _gen(path, mode="auto"):
    g = OnnxGraph(path, fuse_act=True, s2d_stem=True, matmul_on_conv=mode)
    return g, CodeGenerator(g, model_path=path)


def _lowered(g):
    return [sn for sn in g.nodes if isinstance(sn, MatmulConvNode)]


# --------------------------------------------------------------------------- #
# Layout: B's x image and the whole mapping                                     #
# --------------------------------------------------------------------------- #

class TestLayout(unittest.TestCase):

    def test_b_image_matches_explicit_loop(self):
        """x[s][c][kw·p + j] = B[s][(c/16)·16·kw + j·16 + c%16][p]."""
        rng = np.random.default_rng(1)
        for kw in (1, 2, 3, 4, 6):
            for slices, k, m in ((1, 16 * kw, 5), (3, 32 * kw, 12)):
                b = rng.integers(-1000, 1000, (slices, k, m)).astype(np.float64)
                img = conv_lowered_b_image(b, k, m, kw)
                ref = np.zeros((slices, k // kw, m * kw))
                for s in range(slices):
                    for c in range(k // kw):
                        for p in range(m):
                            for j in range(kw):
                                kk = (c // 16) * 16 * kw + j * 16 + c % 16
                                ref[s, c, kw * p + j] = b[s, kk, p]
                np.testing.assert_array_equal(img, ref.reshape(-1), (kw, slices, k, m))
        b = rng.standard_normal((32, 24))
        np.testing.assert_array_equal(conv_lowered_b_image(b, 32, 24, 1), b.reshape(-1))

    @staticmethod
    def _weight_from_packed(a_flat, n, in_ch, kw):
        """Read a flat buffer the way ConvKernel reads its weight port
        (ConvKernel.h conv_weight_index, kh = 1, in_ch % 16 == 0 so every
        ic-tile is a full 16-lane tile): w[m][c][0][j]."""
        t = CONV_TILE_IC
        per_m = kw * in_ch
        w = np.zeros((n, in_ch, 1, kw))
        for m in range(n):
            for c in range(in_ch):
                for j in range(kw):
                    w[m, c, 0, j] = a_flat[m * per_m + (c // t) * kw * t + j * t + c % t]
        return w

    def test_lowered_conv_equals_matmul(self):
        """A as ConvKernel reads its weight buffer, B's image as NCHW x, the
        conv reference with a 1 x kw kernel and stride (1, kw) = A · B
        exactly — and packing that filter the ConvNode way gives back A
        itself (no weight packing for the lowered MatMul)."""
        rng = np.random.default_rng(2)
        for n, k, m, kw, out_w in ((20, 64, 48, 2, 16), (16, 96, 40, 3, 20),
                                   (33, 128, 24, 4, 8), (40, 32, 30, 1, 10),
                                   (8, 192, 12, 6, 12)):
            a = _grid(rng, (n, k), 1.0)
            b = _grid(rng, (k, m), 1.0)
            in_ch, out_h = k // kw, m // out_w
            x = conv_lowered_b_image(b, k, m, kw).reshape(1, in_ch, out_h, kw * out_w)
            w = self._weight_from_packed(a.reshape(-1), n, in_ch, kw)
            y = _conv2d_ref(x, w, None, 1, kw, 0, 0, 1, 1, out_h, out_w)
            np.testing.assert_array_equal(y.reshape(n, m), a @ b, (n, k, m, kw))
            t = TensorInfo(onnx_name="w", shape=list(w.shape), dtype="float32",
                           data=w.astype(np.float32))
            _pack_conv_weight(t, n, in_ch, 1, kw, False)
            np.testing.assert_array_equal(t.packed_data, a.reshape(-1).astype(np.float32))


# --------------------------------------------------------------------------- #
# Engine choice                                                                 #
# --------------------------------------------------------------------------- #

class _Models(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        d = cls._tmp.name
        cls.m = {
            # BERT-base Q/K/V/out linear and FFN, constant B
            "linear": _matmul_model(os.path.join(d, "linear.onnx"), [1, 256, 768], [768, 768]),
            "ffn_up": _matmul_model(os.path.join(d, "ffn_up.onnx"), [256, 768], [768, 3072], seed=1),
            # small linear with a fused Relu consumer, constant B
            "small": _matmul_model(os.path.join(d, "small.onnx"), [64, 256], [256, 96],
                                   seed=2, relu_after=True),
            "gemm": _gemm_model(os.path.join(d, "gemm.onnx"), 48, 128, 64, seed=3),
            # attention: both operands activations, one call per head
            "attn": _attention_model(os.path.join(d, "attn.onnx"), 3, 32, 32),
            # shared constant A, batched activation B -> ConvKernel batch
            "shared_a": _matmul_model(os.path.join(d, "shared_a.onnx"), [32, 64], [3, 64, 40],
                                      a_const=True, b_const=False, seed=4),
            # batched activation A, shared constant B -> rows fold into out_ch
            "fold": _matmul_model(os.path.join(d, "fold.onnx"), [3, 16, 64], [64, 32], seed=5),
            # stays on MatmulKernel
            "fc": _matmul_model(os.path.join(d, "fc.onnx"), [1, 512], [512, 1000], seed=6),
            "k24": _matmul_model(os.path.join(d, "k24.onnx"), [32, 24], [24, 64], seed=7),
            "m12": _matmul_model(os.path.join(d, "m12.onnx"), [32, 64], [64, 12], seed=8),
            "n8": _matmul_model(os.path.join(d, "n8.onnx"), [8, 64], [64, 64], seed=9),
        }

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()


class TestEngineChoice(_Models):

    def test_bert_shapes_lowered(self):
        for name, (n, k, m) in (("linear", (256, 768, 768)), ("ffn_up", (256, 768, 3072))):
            g, _ = _gen(self.m[name])
            (sn,) = _lowered(g)
            self.assertEqual((sn.n, sn.k, sn.m, sn.calls, sn.conv_n), (n, k, m, 1, n))
            self.assertGreater(sn.kw, 1)                  # constant B: re-laid out
            self.assertTrue(sn.b_relayout)
            self.assertEqual(sn.out_h * sn.out_w, m)
            self.assertEqual(sn.k % (CONV_TILE_IC * sn.kw), 0)
            # the cycle model puts the conv > 10x ahead of MatmulKernel
            self.assertLess(sn.est_conv_cycles * 10, sn.est_matmul_cycles)
            self.assertIn("ConvKernel x image", sn.inputs[1].packed_note)

    def test_attention_one_call_per_head(self):
        g, _ = _gen(self.m["attn"])
        low = _lowered(g)
        self.assertEqual(len(low), 2)
        for sn in low:
            self.assertEqual((sn.kw, sn.calls, sn.conv_batch, sn.batch), (1, 3, 1, 3))
            self.assertFalse(sn.b_relayout)
            self.assertEqual((sn.a_call_stride, sn.b_call_stride, sn.c_call_stride),
                             (sn.n * sn.k, sn.k * sn.m, sn.n * sn.m))

    def test_batch_patterns(self):
        (sn,) = _lowered(_gen(self.m["shared_a"])[0])
        self.assertEqual((sn.conv_batch, sn.calls, sn.conv_n, sn.kw), (3, 1, 32, 1))
        (sn,) = _lowered(_gen(self.m["fold"])[0])
        self.assertEqual((sn.conv_batch, sn.calls, sn.conv_n), (1, 1, 48))

    def test_kept_on_matmul_kernel(self):
        for name, why in (("fc", "N = 1"), ("k24", "K = 24"), ("m12", "M = 12")):
            g, _ = _gen(self.m[name], "always")
            (sn,) = [s for s in g.nodes if isinstance(s, MatmulNode)]
            self.assertIn(why, ineligible_reason(sn))
            self.assertEqual(_lowered(g), [])
        # N = 8 < one 16-row output-channel tile: kept by "auto", not by "always"
        self.assertEqual(_lowered(_gen(self.m["n8"])[0]), [])
        self.assertEqual(len(_lowered(_gen(self.m["n8"], "always")[0])), 1)

    def test_modes(self):
        self.assertEqual(normalize_mode(True), "auto")
        self.assertEqual(normalize_mode(False), "off")
        with self.assertRaises(ValueError):
            normalize_mode("sometimes")
        g, _ = _gen(self.m["linear"], "off")
        self.assertEqual(_lowered(g), [])
        self.assertEqual(g.matmul_conv_stats["lowered"], 0)
        self.assertTrue(g.nodes[0].b_packed)              # MatmulKernel packed B as before
        g, _ = _gen(self.m["linear"], False)
        self.assertEqual(_lowered(g), [])

    def test_cheapest_plan_is_used(self):
        (sn,) = _lowered(_gen(self.m["small"])[0])
        plans = conv_plans(sn, range(1, 8))
        self.assertEqual((plans[0].kw, plans[0].out_w), (sn.kw, sn.out_w))
        self.assertTrue(all(p.cycles >= plans[0].cycles for p in plans))
        self.assertLess(sn.est_conv_cycles, LOWER_MARGIN * sn.est_matmul_cycles)

    def test_shared_constant_b_keeps_row_major(self):
        """A constant B read by two MatMuls is not re-laid out (kw = 1)."""
        with tempfile.TemporaryDirectory() as td:
            rng = np.random.default_rng(3)
            p = _save(os.path.join(td, "shared_b.onnx"),
                      [oh.make_node("MatMul", ["X1", "W"], ["Y1"]),
                       oh.make_node("MatMul", ["X2", "W"], ["Y2"])],
                      [_f32("X1", [64, 256]), _f32("X2", [32, 256])],
                      [_f32("Y1", [64, 64]), _f32("Y2", [32, 64])],
                      {"W": _grid(rng, (256, 64), 0.25)})
            g, cg = _gen(p)
            low = _lowered(g)
            self.assertEqual(len(low), 2)
            self.assertTrue(all(sn.kw == 1 and not sn.b_relayout for sn in low))
            w = g.get_tensor("W")
            self.assertIsNone(w.packed_data)                   # emitted row-major
            self.assertIs(w.emit_data, w.data)
            del cg


# --------------------------------------------------------------------------- #
# Simulation and weight images                                                  #
# --------------------------------------------------------------------------- #

class TestSimulation(_Models):

    def test_outputs_identical_with_lowering_on_and_off(self):
        """The lowered op has no model of its own: _simulate gives the same
        bits whichever engine runs the MatMul (both kernels: exact products,
        ap_fixed<32,16> sum, floor + saturate)."""
        for name in ("linear", "small", "gemm", "attn", "shared_a", "fold"):
            _, on = _gen(self.m[name])
            _, off = _gen(self.m[name], "off")
            a, b = on._simulate(), off._simulate()
            for t in on._graph.output_tensors:
                np.testing.assert_array_equal(a[t.onnx_name], b[t.onnx_name], name)

    def test_saturation_identical(self):
        with tempfile.TemporaryDirectory() as td:
            p = _matmul_model(os.path.join(td, "sat.onnx"), [32, 64], [64, 32],
                              seed=11, scale=40.0)
            _, on = _gen(p)
            _, off = _gen(p, "off")
            self.assertEqual(len(_lowered(on._graph)), 1)
            a, b = on._simulate()["Y"], off._simulate()["Y"]
            np.testing.assert_array_equal(a, b)
            self.assertTrue((np.abs(a) >= 127.99).any())      # some outputs saturate

    def test_weight_image_is_the_relayout(self):
        """The emitted DMA image of a re-laid-out B is conv_lowered_b_image
        of the rounded weights (the simulator's view stays logical)."""
        g, cg = _gen(self.m["small"])
        (sn,) = _lowered(g)
        b = sn.inputs[1]
        self.assertEqual(b.numel, sn.k * sn.m)
        self.assertEqual(list(b.shape), [sn.k, sn.m])
        raw = np.frombuffer(cg.generate_weight_dat(b), dtype="<i2") if \
            b in cg.large_weight_tensors else None
        q = np.round(b.data.astype(np.float64) * 256)
        img = conv_lowered_b_image(q, sn.k, sn.m, sn.kw)
        if raw is not None:
            np.testing.assert_array_equal(raw, img)
        np.testing.assert_array_equal(np.round(b.emit_data.astype(np.float64) * 256), img)


# --------------------------------------------------------------------------- #
# Generated C                                                                   #
# --------------------------------------------------------------------------- #

class TestCodegen(_Models):

    def test_single_call_emission(self):
        g, cg = _gen(self.m["linear"])
        (sn,) = _lowered(g)
        src = cg.generate_source()
        self.assertIn("static void run_conv(", src)
        self.assertNotIn("static void run_conv_at(", src)
        self.assertNotIn("run_matmul", src)
        call = src[src.index("    run_conv(B, A, NULL, Y,"):]
        call = call[:call.index(";") + 1]
        self.assertIn(f"1u, {sn.k // sn.kw}u, {sn.out_h}u, {sn.kw * sn.out_w}u,", call)
        self.assertIn(f"256u, {sn.out_h}u, {sn.out_w}u,", call)
        self.assertIn(f"1u, {sn.kw}u, 1u, {sn.kw}u,", call)
        self.assertIn("1u, 1u, 0u, 0u, 0u, 0u);", call)
        self.assertIn("on ConvKernel: weight=A x=B (kw layout)", src)
        h = cg.generate_header()
        self.assertIn("convkernel_instance", h)
        self.assertNotIn("matmulkernel_instance", h)

    def test_per_head_loop_emission(self):
        _, cg = _gen(self.m["attn"])
        src = cg.generate_source()
        self.assertIn("static void run_conv_at(", src)
        self.assertNotIn("static void run_conv(", src)       # every conv call is per head
        self.assertIn("for (unsigned _i = 0u; _i < 3u; _i++) {", src)
        self.assertIn("if (_i) kernel_wait(KERNEL_CONV);", src)
        self.assertIn("run_conv_at(KT, _i * 1024u, Q, _i * 1024u,", src)
        self.assertIn("S, _i * 1024u,", src)
        # the second MatMul waits for the first one's last head
        body = src[src.index("run_conv_at(KT"):]
        self.assertLess(body.index("kernel_wait(KERNEL_CONV);\n    INFERENCE_PROF_END(0u);"),
                        body.index("run_conv_at(V"))

    def test_event_stream_lane(self):
        g, cg = _gen(self.m["gemm"])
        (sn,) = _lowered(g)
        add = next(s for s in g.nodes if isinstance(s, ScheduledNode))
        ev = cg._compute_event_stream()
        self.assertIn(("start", sn.index), ev)
        i = ev.index(("start", add.index))
        self.assertIn(("wait", "KERNEL_CONV", sn.index), ev[:i])

    def test_werror_compile(self):
        from test_s2d_stem import _host_compile, _which_cc
        if not _which_cc():
            self.skipTest("no C compiler")
        for name in ("linear", "attn", "shared_a", "gemm", "small"):
            _, cg = _gen(self.m[name])
            with tempfile.TemporaryDirectory() as td:
                rc, log = _host_compile(cg, td)
            self.assertEqual(rc, 0, f"{name}\n{log}")

    @unittest.skipUnless(host_emu.which_cc(), "C compiler not available on host")
    def test_host_emulated_run_matches_simulation(self):
        """inference.c + test_inference.c against the software ConvKernel /
        MatmulKernel / VectorOPKernel models (test/host_emu.py) reproduce the
        simulation bit for bit: the call arguments, per-head offsets and the
        re-laid-out weight images are right."""
        for name, mode in (("small", "auto"), ("gemm", "auto"), ("attn", "auto"),
                           ("shared_a", "auto"), ("fold", "auto"), ("n8", "always"),
                           ("small", "off")):
            _, cg = _gen(self.m[name], mode)
            with tempfile.TemporaryDirectory() as td:
                rc, out = host_emu.build_and_run(cg, td)
            self.assertEqual(rc, 0, f"{name} {mode}\n{out[-2000:]}")
            self.assertIn("test_inference PASSED", out)

    def test_report_and_cli(self):
        g, cg = _gen(self.m["gemm"])
        md = ReportGenerator(graph=g, codegen=cg, model_path=self.m["gemm"], out_dir="/tmp",
                             generated_files=[]).render_markdown()
        self.assertIn("**MatMul on ConvKernel** — 1 `MatMul` node run as ConvKernel calls", md)
        self.assertIn("on ConvKernel 1×", md)
        with tempfile.TemporaryDirectory() as td:
            cli = os.path.join(_ROOT, "inference_scheduler.py")
            r = subprocess.run([sys.executable, cli, self.m["gemm"], "--out-dir", td],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("MatMul->Conv: 1 MatMul(s) on ConvKernel", r.stderr)
            with open(os.path.join(td, "src", "inference.c")) as f:
                self.assertIn("run_conv(", f.read())
            r = subprocess.run([sys.executable, cli, self.m["gemm"], "--out-dir", td,
                                "--no-matmul-on-conv"], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertNotIn("MatMul->Conv", r.stderr)
            with open(os.path.join(td, "src", "inference.c")) as f:
                src = f.read()
            self.assertIn("run_matmul(", src)
            self.assertNotIn("run_conv(", src)


# --------------------------------------------------------------------------- #
# Cost model                                                                    #
# --------------------------------------------------------------------------- #

class TestCostModel(unittest.TestCase):

    @unittest.skipUnless(os.path.isfile(_SKILL), "conv-cycle-model skill script not found")
    def test_conv_model_equals_skill_script(self):
        spec = importlib.util.spec_from_file_location("conv_cycle_model", _SKILL)
        ccm = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ccm)
        P = ccm.load_platform(os.environ.get("AXI_PLATFORM", "kv260"))
        rnd = random.Random(5)
        n = 0
        while n < 300:
            c, m = rnd.randint(1, 200), rnd.randint(1, 300)
            kh, kw = rnd.randint(1, 7), rnd.randint(1, 7)
            sh, sw, dh, dw = rnd.randint(1, 3), rnd.randint(1, 4), rnd.randint(1, 2), rnd.randint(1, 2)
            pt, pl = rnd.randint(0, 2), rnd.randint(0, 2)
            h, w = rnd.randint(kh * dh, 60), rnd.randint(kw * dw, 140)
            oh = (h + 2 * pt - (dh * (kh - 1) + 1)) // sh + 1
            ow = (w + 2 * pl - (dw * (kw - 1) + 1)) // sw + 1
            if oh < 1 or ow < 1 or ow * (-(-m // 16) * 16) > 65536:
                continue
            ref = ccm.model_layer(P, c, m, h, w, oh, ow, kh, kw, sh, sw, dh, dw, pt, pl, False)
            got = conv_cycles(c, m, h, w, oh, ow, kh, kw, sh, sw, dh, dw, pt, pl)
            for key in ("total", "sweep", "fill", "ph1", "ph3", "loads"):
                self.assertEqual(got[key], ref[key], (key, c, m, h, w, kh, kw, sh, sw))
            n += 1

    def test_matmul_model_board_calibration(self):
        """MatmulKernel block model against the board (100 MHz): 256^3 7.24 ms,
        BERT 768^2 linear 54.3 ms, 3072x768 203 ms, QK^T head 3.28 ms."""
        for (n, k, m), ms in (((256, 256, 256), 7.24), ((256, 768, 768), 54.3),
                              ((256, 3072, 768), 203.0), ((256, 64, 256), 3.28)):
            self.assertAlmostEqual(matmul_cycles(n, k, m) / 1e5 / ms, 1.0, delta=0.05)


if __name__ == "__main__":
    unittest.main()
