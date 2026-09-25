"""Space-to-depth stem transform (RESNET18_15FPS_PLAN step 1).

A stride-2 Conv with 4*C <= kTileIC input channels is rewritten as a
host-side SpaceToDepth(2) + stride-1 Conv over 4*C channels with
re-indexed weights (``OnnxGraph._space_to_depth_stems``).  Models are
built inline (onnx.helper) so the tests need no generator run.
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as nph
from onnx import TensorProto

from src.graph   import OnnxGraph
from src.codegen import CodeGenerator
from src.codegen._simulate import _conv2d_ref
from src.nodes   import (ConvNode, ScheduledNode, SpaceToDepthNode,
                         _s2d_stem_geometry, _s2d_stem_weight)
from src.report  import ReportGenerator
from src._conv_hw_config import CONV_TILE_IC

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)


def _f32(name, shape):
    return oh.make_tensor_value_info(name, TensorProto.FLOAT, shape)


def _init(name, arr):
    return nph.from_array(np.asarray(arr, dtype=np.float32), name=name)


def _save(nodes, inputs, outputs, initializers, name, path):
    graph = oh.make_graph(nodes, name, inputs, outputs, initializer=initializers)
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return path


def _grid(rng, shape, scale):
    """Random values on the ap_fixed<16,8> grid (multiples of 1/256)."""
    return np.round(rng.uniform(-scale, scale, shape) * 256) / 256


def _conv_out(h, k, s, p):
    return (h + 2 * p - k) // s + 1


def _stem_model(path, name, C, H, W, k, pad, stride=2, M=8, bias=True,
                group=1, pre_relu=False, rng=None):
    """[Relu ->] Conv kxk stride pad -> Relu, C input channels."""
    rng = rng or np.random.default_rng(11)
    cw = C // group
    w = _grid(rng, (M, cw, k, k), 0.5)
    inits = [_init("W", w)]
    conv_in = ["X", "W"]
    if bias:
        inits.append(_init("B", _grid(rng, (M,), 1.0)))
        conv_in.append("B")
    nodes = []
    src = "X"
    if pre_relu:
        nodes.append(oh.make_node("Relu", ["X"], ["relu_X"]))
        src = "relu_X"
    nodes.append(oh.make_node("Conv", [src] + conv_in[1:], ["conv_Y"],
                              kernel_shape=[k, k], strides=[stride, stride],
                              pads=[pad] * 4, group=group))
    nodes.append(oh.make_node("Relu", ["conv_Y"], ["Y"]))
    oh_, ow_ = _conv_out(H, k, stride, pad), _conv_out(W, k, stride, pad)
    return _save(nodes, [_f32("X", [1, C, H, W])], [_f32("Y", [1, M, oh_, ow_])],
                 inits, name, path)


def _which_cc():
    return shutil.which("cc") or shutil.which("gcc")


class _ModelDir(unittest.TestCase):
    """Builds the fixture models once per class into a temp dir."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        d = cls._tmp.name
        j = lambda n: os.path.join(d, n)  # noqa: E731
        # ResNet-18 stem geometry, scaled down: 7x7 s2 p3, 3 -> 8, 16x16 -> 8x8
        cls.stem7      = _stem_model(j("stem7.onnx"), "stem7", 3, 16, 16, 7, 3)
        cls.stem5_ic4  = _stem_model(j("stem5.onnx"), "stem5", 4, 12, 14, 5, 2)
        cls.stem3_ic1  = _stem_model(j("stem3.onnx"), "stem3", 1, 8, 8, 3, 1, bias=False)
        cls.stem3_p0   = _stem_model(j("stem3p0.onnx"), "stem3p0", 2, 10, 8, 3, 0)
        cls.stem7_pre  = _stem_model(j("stem7_pre.onnx"), "stem7_pre", 3, 16, 16, 7, 3,
                                     pre_relu=True)
        # must NOT transform
        cls.ic5        = _stem_model(j("ic5.onnx"), "ic5", 5, 16, 16, 7, 3)
        cls.stride1    = _stem_model(j("stride1.onnx"), "stride1", 3, 16, 16, 7, 3, stride=1)
        cls.odd_h      = _stem_model(j("odd_h.onnx"), "odd_h", 3, 15, 16, 7, 3)
        cls.odd_w      = _stem_model(j("odd_w.onnx"), "odd_w", 3, 16, 17, 7, 3)
        cls.depthwise  = _stem_model(j("dw.onnx"), "dw", 3, 16, 16, 3, 1, M=3, group=3)

        # one input feeding two stems and a plain Relu (shared SpaceToDepth output)
        rng = np.random.default_rng(5)
        cls.shared = _save(
            [oh.make_node("Conv", ["X", "W1"], ["c1"], kernel_shape=[7, 7],
                          strides=[2, 2], pads=[3, 3, 3, 3]),
             oh.make_node("Conv", ["X", "W2"], ["c2"], kernel_shape=[7, 7],
                          strides=[2, 2], pads=[3, 3, 3, 3]),
             oh.make_node("Add",  ["c1", "c2"], ["Y"]),
             oh.make_node("Relu", ["X"], ["Y2"])],
            [_f32("X", [1, 3, 16, 16])], [_f32("Y", [1, 8, 8, 8]), _f32("Y2", [1, 3, 16, 16])],
            [_init("W1", _grid(rng, (8, 3, 7, 7), 0.5)), _init("W2", _grid(rng, (8, 3, 7, 7), 0.5))],
            "shared", j("shared.onnx"))

        # a model that carries a native SpaceToDepth op
        cls.native = _save(
            [oh.make_node("SpaceToDepth", ["X"], ["s"], blocksize=2),
             oh.make_node("Relu", ["s"], ["Y"])],
            [_f32("X", [1, 2, 6, 4])], [_f32("Y", [1, 8, 3, 2])], [],
            "native_s2d", j("native.onnx"))

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    @staticmethod
    def _gen(path, s2d):
        g  = OnnxGraph(path, s2d_stem=s2d)
        cg = CodeGenerator(g, model_path=path)
        return g, cg

    @staticmethod
    def _ops(g):
        return [sn.onnx_node.op_type for sn in g.nodes]


# --------------------------------------------------------------------------- #
# Pure transform maths                                                          #
# --------------------------------------------------------------------------- #

class TestGeometry(unittest.TestCase):

    def test_axis_geometry(self):
        # (k, pad) -> (K', P', off): t = 2R + ph + off
        self.assertEqual(_s2d_stem_geometry(7, 3), (4, 2, -1))
        self.assertEqual(_s2d_stem_geometry(5, 2), (3, 1, 0))
        self.assertEqual(_s2d_stem_geometry(3, 1), (2, 1, -1))
        self.assertEqual(_s2d_stem_geometry(3, 0), (2, 0, 0))
        self.assertEqual(_s2d_stem_geometry(1, 0), (1, 0, 0))
        self.assertEqual(_s2d_stem_geometry(7, 2), (4, 1, 0))

    def test_every_tap_lands_exactly_once(self):
        # w' is a permutation of w's taps plus zeros: same multiset of values
        rng = np.random.default_rng(0)
        for k, pad in [(7, 3), (5, 2), (3, 1), (3, 0), (4, 1), (7, 2)]:
            w = rng.standard_normal((2, 3, k, k))
            w2 = _s2d_stem_weight(w, pad, pad)
            self.assertEqual(w2.shape[1], 12)
            self.assertEqual(np.count_nonzero(w2), w.size)
            np.testing.assert_array_equal(np.sort(w2[w2 != 0]), np.sort(w.ravel()))

    def test_weight_reindex_matches_float_conv(self):
        """Original stride-2 conv == stride-1 conv over the space-to-depth
        input with the re-indexed weights (float64, generic values)."""
        rng = np.random.default_rng(1)
        cases = [(7, 3, 224, 224, 3), (7, 3, 16, 18, 3), (5, 2, 14, 12, 4),
                 (3, 1, 8, 8, 1), (3, 0, 10, 8, 2), (4, 1, 12, 12, 2), (1, 0, 8, 8, 4)]
        for k, pad, H, W, C in cases:
            x = rng.standard_normal((1, C, H, W))
            w = rng.standard_normal((4, C, k, k))
            oh_, ow_ = _conv_out(H, k, 2, pad), _conv_out(W, k, 2, pad)
            y0 = _conv2d_ref(x, w, None, 2, 2, pad, pad, 1, 1, oh_, ow_)
            x2 = (x.reshape(1, C, H // 2, 2, W // 2, 2).transpose(0, 3, 5, 1, 2, 4)
                    .reshape(1, 4 * C, H // 2, W // 2))
            w2 = _s2d_stem_weight(w, pad, pad)
            _, P, _ = _s2d_stem_geometry(k, pad)
            y1 = _conv2d_ref(x2, w2, None, 1, 1, P, P, 1, 1, oh_, ow_)
            np.testing.assert_allclose(y1, y0, rtol=0, atol=1e-9,
                                       err_msg=f"k={k} pad={pad} {H}x{W} C={C}")


# --------------------------------------------------------------------------- #
# Graph rewrite                                                                 #
# --------------------------------------------------------------------------- #

class TestRewrite(_ModelDir):

    def test_library_default_is_off(self):
        g = OnnxGraph(self.stem7)
        self.assertEqual(g.s2d_stem_count, 0)
        self.assertEqual(self._ops(g), ["Conv", "Relu"])
        self.assertEqual(g.nodes[0].kh, 7)

    def test_stem7_rewrite(self):
        g, _ = self._gen(self.stem7, True)
        self.assertEqual(g.s2d_stem_count, 1)
        self.assertEqual(self._ops(g), ["SpaceToDepth", "Conv", "Relu"])
        self.assertEqual([sn.index for sn in g.nodes], [0, 1, 2])
        s2d, conv = g.nodes[0], g.nodes[1]
        self.assertIsInstance(s2d, SpaceToDepthNode)
        self.assertIsInstance(conv, ConvNode)
        self.assertEqual((s2d.batch, s2d.in_ch, s2d.in_h, s2d.in_w, s2d.blocksize), (1, 3, 16, 16, 2))
        self.assertTrue(s2d.src_is_graph_input)
        self.assertEqual(s2d.inputs[0].onnx_name, "X")
        self.assertEqual(s2d.output.onnx_name, "X_s2d")
        self.assertEqual(s2d.output.shape, [1, 12, 8, 8])
        self.assertEqual(conv.inputs[0].onnx_name, "X_s2d")
        self.assertEqual(conv.inputs[1].onnx_name, "W_s2d")
        self.assertEqual(conv.inputs[1].shape, [8, 12, 4, 4])
        self.assertEqual(conv.inputs[2].onnx_name, "B")           # bias unchanged
        self.assertEqual((conv.in_ch, conv.in_h, conv.in_w), (12, 8, 8))
        self.assertEqual((conv.kh, conv.kw, conv.stride_h, conv.stride_w), (4, 4, 1, 1))
        self.assertEqual((conv.pad_top, conv.pad_left), (2, 2))
        self.assertEqual((conv.out_ch, conv.out_h, conv.out_w), (8, 8, 8))
        self.assertEqual(conv.output.onnx_name, "conv_Y")
        # the rewritten Conv keeps the ONNX pads that reproduce out_h / out_w
        pads = {a.name: list(a.ints) for a in conv.onnx_node.attribute}["pads"]
        self.assertEqual(pads, [2, 2, 1, 1])
        # the packed weight follows the normal ConvNode path
        self.assertIsNotNone(conv.inputs[1].packed_data)
        self.assertIn("ConvKernel tile-major", conv.inputs[1].packed_note)
        # the original 7x7 weight is no longer referenced / emitted
        self.assertNotIn("W", [t.onnx_name for t in g.weight_tensors])
        self.assertIn("W_s2d", [t.onnx_name for t in g.weight_tensors])

    def test_other_geometries(self):
        for path, kk, P, C4, pads in [
            (self.stem5_ic4, 3, 1, 16, [1, 1, 1, 1]),
            (self.stem3_ic1, 2, 1, 4,  [1, 1, 0, 0]),
            (self.stem3_p0,  2, 0, 8,  [0, 0, 0, 0]),
        ]:
            g, _ = self._gen(path, True)
            self.assertEqual(g.s2d_stem_count, 1, path)
            conv = g.nodes[1]
            self.assertEqual((conv.kh, conv.kw, conv.pad_top, conv.pad_left, conv.in_ch),
                             (kk, kk, P, P, C4), path)
            attr = {a.name: list(a.ints) for a in conv.onnx_node.attribute}
            self.assertEqual(attr["pads"], pads, path)

    def test_no_bias(self):
        g, _ = self._gen(self.stem3_ic1, True)
        conv = g.nodes[1]
        self.assertFalse(conv.has_bias)
        self.assertEqual(len(conv.inputs), 2)

    def test_not_applicable(self):
        for path, why in [(self.ic5, "5 channels"), (self.stride1, "stride 1"),
                          (self.odd_h, "odd H"), (self.odd_w, "odd W"),
                          (self.depthwise, "depthwise")]:
            g, _ = self._gen(path, True)
            self.assertEqual(g.s2d_stem_count, 0, why)
            self.assertEqual(self._ops(g), ["Conv", "Relu"], why)

    def test_channel_bound_follows_tile(self):
        self.assertEqual(CONV_TILE_IC, 16)   # ic <= 4 on this platform

    def test_input_from_kernel(self):
        g, _ = self._gen(self.stem7_pre, True)
        self.assertEqual(self._ops(g), ["Relu", "SpaceToDepth", "Conv", "Relu"])
        s2d = g.nodes[1]
        self.assertFalse(s2d.src_is_graph_input)
        self.assertEqual(s2d.inputs[0].onnx_name, "relu_X")
        self.assertEqual(s2d.output.onnx_name, "relu_X_s2d")

    def test_shared_input(self):
        g, cg = self._gen(self.shared, True)
        self.assertEqual(g.s2d_stem_count, 2)
        self.assertEqual(self._ops(g), ["SpaceToDepth", "Conv", "Conv", "Add", "Relu"])
        c1, c2 = g.nodes[1], g.nodes[2]
        self.assertEqual(c1.inputs[0].onnx_name, "X_s2d")
        self.assertEqual(c2.inputs[0].onnx_name, "X_s2d")
        self.assertEqual({c1.inputs[1].onnx_name, c2.inputs[1].onnx_name}, {"W1_s2d", "W2_s2d"})
        self.assertEqual(g.nodes[4].inputs[0].onnx_name, "X")  # the Relu still reads X
        src = cg.generate_source()
        self.assertEqual(src.count("*dst++ = s[cc * 2u];"), 1)
        self.assertEqual(src.count("= (Data_t *)malloc("), 1)

    def test_native_space_to_depth(self):
        g, cg = self._gen(self.native, False)
        self.assertEqual(self._ops(g), ["SpaceToDepth", "Relu"])
        x = _grid(np.random.default_rng(3), (1, 2, 6, 4), 4.0)
        y = cg.simulate({"X": x})["Y"]
        ref = np.maximum(x.reshape(1, 2, 3, 2, 2, 2).transpose(0, 3, 5, 1, 2, 4)
                          .reshape(1, 8, 3, 2), 0.0)
        np.testing.assert_array_equal(y, ref)


# --------------------------------------------------------------------------- #
# Numerics                                                                      #
# --------------------------------------------------------------------------- #

class TestNumerics(_ModelDir):

    def _check(self, path, shape):
        g0, cg0 = self._gen(path, False)
        g1, cg1 = self._gen(path, True)
        self.assertEqual(g1.s2d_stem_count, 1)
        x = _grid(np.random.default_rng(9), shape, 4.0)
        y0 = cg0.simulate({"X": x})["Y"]
        y1 = cg1.simulate({"X": x})["Y"]
        self.assertEqual(y0.shape, y1.shape)
        # every partial sum is a multiple of 2^-16 far below 2^37, so the
        # float64 accumulation is exact in both orders: bit-identical outputs
        np.testing.assert_array_equal(y1, y0)
        self.assertGreater(np.count_nonzero(y1), 0)
        # ... and so are the generated self-test's expected bytes
        self.assertEqual(cg0.generate_expected_dat(g0.output_tensors[0]),
                         cg1.generate_expected_dat(g1.output_tensors[0]))

    def test_stem7(self):
        self._check(self.stem7, (1, 3, 16, 16))

    def test_stem5_ic4(self):
        self._check(self.stem5_ic4, (1, 4, 12, 14))

    def test_stem3_ic1(self):
        self._check(self.stem3_ic1, (1, 1, 8, 8))

    def test_stem3_p0(self):
        self._check(self.stem3_p0, (1, 2, 10, 8))

    def test_stem7_after_kernel(self):
        self._check(self.stem7_pre, (1, 3, 16, 16))

    def test_shared(self):
        g0, cg0 = self._gen(self.shared, False)
        g1, cg1 = self._gen(self.shared, True)
        x = _grid(np.random.default_rng(2), (1, 3, 16, 16), 4.0)
        r0, r1 = cg0.simulate({"X": x}), cg1.simulate({"X": x})
        for k in ("Y", "Y2"):
            np.testing.assert_array_equal(r1[k], r0[k])


# --------------------------------------------------------------------------- #
# Generated C                                                                   #
# --------------------------------------------------------------------------- #

def _host_compile(cg, workdir):
    """Compile the generated inference.c on the host with stub driver
    headers (every X<Kernel>_* call becomes a no-op).  Returns (rc, log)."""
    cc = _which_cc()
    os.makedirs(os.path.join(workdir, "include"), exist_ok=True)
    os.makedirs(os.path.join(workdir, "src"), exist_ok=True)
    os.makedirs(os.path.join(workdir, "stub"), exist_ok=True)
    src = cg.generate_source()
    with open(os.path.join(workdir, "include", "inference.h"), "w") as f:
        f.write(cg.generate_header())
    with open(os.path.join(workdir, "src", "inference.c"), "w") as f:
        f.write(src)
    shutil.copy(os.path.join(_ROOT, "runtime", "inference_prof.h"),
                os.path.join(workdir, "include", "inference_prof.h"))
    for kd in cg._active_kernels:
        funcs = sorted(set(re.findall(rf"\b({kd.c_type}_[A-Za-z_]+)\s*\(", src)))
        lines = ["#pragma once", "#include <stdint.h>",
                 "typedef uint64_t u64;", f"typedef struct {{ int dummy; }} {kd.c_type};"]
        for fn in funcs:
            if fn.endswith("_IsDone") or fn.endswith("_Initialize"):
                lines.append(f"static inline int {fn}({kd.c_type} *p, ...) {{ (void)p; return 1; }}")
            else:
                lines.append(f"static inline void {fn}({kd.c_type} *p, ...) {{ (void)p; }}")
        with open(os.path.join(workdir, "stub", f"{kd.driver_prefix}.h"), "w") as f:
            f.write("\n".join(lines) + "\n")
    cmd = [cc, "-std=c99", "-Wall", "-Wextra", "-Werror", "-fsyntax-only",
           "-I", os.path.join(workdir, "include"), "-I", os.path.join(workdir, "stub"),
           os.path.join(workdir, "src", "inference.c")]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


class TestCodegen(_ModelDir):

    def test_emitted_source(self):
        g, cg = self._gen(self.stem7, True)
        src = cg.generate_source()
        self.assertIn("/* [0] SpaceToDepth(X) -> X_s2d  [1, 3, 16, 16] → [1, 12, 8, 8]"
                      "  blocksize=2  (host CPU reorder, no hardware call) */", src)
        loop = src[src.index("INFERENCE_PROF_BEGIN(0u);"):src.index("INFERENCE_PROF_END(0u);")]
        # a cacheable DMA buffer is reordered in place; a non-cacheable one is
        # staged through cached host memory (only two sequential memcpys
        # touch it, never the strided loads)
        self.assertIn("if (inference_buf_is_cached(X)) {\n            src = inference_buf_ptr(X);", loop)
        self.assertIn("memcpy(_s2d_stage_X_s2d, inference_buf_ptr(X), 768u * INFERENCE_BYTES_PER_ELEM);\n"
                      "            src = _s2d_stage_X_s2d;", loop)
        self.assertIn("dst0 = inference_buf_is_cached(X_s2d) ? inference_buf_ptr(X_s2d)"
                      " : _s2d_stage_X_s2d + 768u;", loop)
        self.assertIn("if (dst0 != inference_buf_ptr(X_s2d))\n"
                      "            memcpy(inference_buf_ptr(X_s2d), dst0, 768u * INFERENCE_BYTES_PER_ELEM);",
                      loop)
        self.assertLess(loop.index("memcpy(_s2d_stage_X_s2d,"), loop.index("*dst++"))
        self.assertLess(loop.index("*dst++"), loop.index("memcpy(inference_buf_ptr(X_s2d)"))
        self.assertLess(loop.index("memcpy(inference_buf_ptr(X_s2d)"), loop.index("inference_buf_sync_to_device(X_s2d);"))
        self.assertNotIn("inference_buf_ptr(X)[", loop)
        self.assertIn("non-cacheable", loop)
        # the staging block is malloc'd in init and freed in deinit
        self.assertIn("#include <stdlib.h>", src)
        self.assertIn("static Data_t *_s2d_stage_X_s2d = NULL;", src)
        init = src[src.index("int inference_init("):src.index("void inference_deinit(void)")]
        self.assertIn("_s2d_stage_X_s2d = (Data_t *)malloc(2u * 768u * INFERENCE_BYTES_PER_ELEM);", init)
        self.assertIn("if (!_s2d_stage_X_s2d) { rc = -1; goto fail; }", init)
        deinit = src[src.index("void inference_deinit(void)"):]
        self.assertIn("free(_s2d_stage_X_s2d); _s2d_stage_X_s2d = NULL;", deinit)
        self.assertIn("for (c = 0u; c < 3u; c++)", loop)
        self.assertIn("for (r = 0u; r < 8u; r++) {", loop)
        self.assertIn("const Data_t *s = src + ((n * 3u + c) * 16u + r * 2u + ph) * 16u + pw;", loop)
        self.assertIn("for (cc = 0u; cc < 8u; cc++)", loop)
        self.assertIn("*dst++ = s[cc * 2u];", loop)
        self.assertIn("inference_buf_sync_to_device(X_s2d);", loop)
        self.assertNotIn("sync_from_device", loop)            # graph input: already flushed
        self.assertNotIn("kernel_wait", loop)
        # the conv reads the reordered buffer with the new geometry
        m = re.search(r"run_conv\(X_s2d, W_s2d, B, conv_Y,\s*1u, 12u, 8u, 8u,\s*8u, 8u, 8u,"
                      r"\s*4u, 4u, 1u, 1u,\s*1u, 1u, 2u, 2u, 1u, 0u\);", src)
        self.assertIsNotNone(m)
        # X_s2d is a real pool buffer, not an alias
        self.assertIn("static inference_buf_t _s_buf_X_s2d;", src)
        self.assertRegex(src, r"inference_buf_init_view\(&_s_buf_X_s2d, s_alloc_pool, \d+u, 768u\);")
        self.assertIn("/* External weight 'W_s2d'", src) if g.nodes[1].inputs[1].is_large_weight \
            else self.assertIn("_rom_W_s2d[", src)

    def test_public_api_unchanged(self):
        _, cg0 = self._gen(self.stem7, False)
        _, cg1 = self._gen(self.stem7, True)
        h0, h1 = cg0.generate_header(), cg1.generate_header()
        self.assertNotIn("<stdlib.h>", cg0.generate_source())   # no host op: no staging
        for h in (h0, h1):
            self.assertIn("#define INFERENCE_X_SIZE", h)
            self.assertIn("768u  /* shape=[1, 3, 16, 16] */", h)
            self.assertRegex(h, r"void inference_run\(\s*inference_buf_t \*X,\s*inference_buf_t \*Y\);")
        self.assertNotIn("X_s2d", h1)
        self.assertNotIn("INFERENCE_X_S2D", h1)
        self.assertEqual(h1.count("inference_init("), h0.count("inference_init("))
        self.assertIn("#define INFERENCE_NUM_LAYERS  3u", h1)

    def test_event_stream_and_liveness_after_kernel(self):
        g, cg = self._gen(self.stem7_pre, True)
        events = cg._compute_event_stream()
        kinds = [(e[0], e[1]) for e in events if e[0] != 'comment']
        # Relu starts on the VectorOP lane; the host op must wait for it,
        # runs inline, and the conv then starts without waiting on anything.
        self.assertEqual(kinds[:4], [('start', 0), ('wait', 'KERNEL_VECTOROP'),
                                     ('cpu', 1), ('start', 2)])
        self.assertEqual(events[[e[0] for e in events].index('wait')][2], 0)
        src = cg.generate_source()
        loop = src[src.index("INFERENCE_PROF_BEGIN(1u);"):src.index("INFERENCE_PROF_END(1u);")]
        self.assertIn("inference_buf_sync_from_device(relu_X);", loop)
        self.assertLess(src.index("kernel_wait(KERNEL_VECTOROP);"), src.index("INFERENCE_PROF_BEGIN(1u);"))
        # relu_X is live until the host op reads it; relu_X_s2d from that
        # same event until the conv drains -> they must not share a slot
        iv = cg._compute_live_intervals()
        cpu_ei = next(i for i, e in enumerate(events) if e[0] == 'cpu')
        self.assertEqual(iv["relu_X"][1], cpu_ei)
        self.assertEqual(iv["relu_X_s2d"][0], cpu_ei)
        self.assertGreater(iv["relu_X_s2d"][1], cpu_ei)
        layout, _ = cg._compute_pool_layout()
        off = {name: o for name, o, _a in layout}
        self.assertNotEqual(off["relu_X"], off["relu_X_s2d"])

    def test_report(self):
        g, cg = self._gen(self.stem7, True)
        md = ReportGenerator(graph=g, codegen=cg, model_path=self.stem7,
                             out_dir="/tmp", generated_files=[]).render_markdown()
        self.assertIn("Space-to-depth stem", md)
        self.assertIn("1 stride-2 `Conv` rewritten", md)
        self.assertIn("host CPU reorder · blocksize=2 · no kernel call", md)
        self.assertIn("| Reshape / host op (no lane) | – | 1 |", md)

    @unittest.skipUnless(_which_cc(), "C compiler not available on host")
    def test_generated_source_compiles_on_host(self):
        for path in (self.stem7, self.stem7_pre, self.shared):
            _, cg = self._gen(path, True)
            with tempfile.TemporaryDirectory() as td:
                rc, log = _host_compile(cg, td)
            self.assertEqual(rc, 0, log)


class TestCli(_ModelDir):

    def _cli(self, model, extra):
        with tempfile.TemporaryDirectory() as td:
            cmd = [sys.executable, os.path.join(_ROOT, "inference_scheduler.py"),
                   model, "--out-dir", td, "--no-report"] + extra
            r = subprocess.run(cmd, capture_output=True, text=True, cwd=_ROOT)
            self.assertEqual(r.returncode, 0, r.stderr)
            with open(os.path.join(td, "src", "inference.c")) as f:
                return f.read(), r.stderr

    def test_cli_default_on(self):
        src, log = self._cli(self.stem7, [])
        self.assertIn("SpaceToDepth(X) -> X_s2d", src)
        self.assertIn("[  0] SpaceToDepth", log)

    def test_cli_no_s2d_stem(self):
        src, _ = self._cli(self.stem7, ["--no-s2d-stem"])
        self.assertNotIn("SpaceToDepth", src)
        self.assertIn("7u, 7u, 2u, 2u,", src)


if __name__ == "__main__":
    unittest.main()
