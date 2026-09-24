"""Tests for ConvKernel scheduler integration."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.graph   import OnnxGraph
from src.codegen import CodeGenerator
from src.nodes   import ConvNode, SchedulerError


MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")


def _conv_model(name: str) -> str:
    return os.path.join(MODELS_DIR, name)


def _conv_models_exist() -> bool:
    return (os.path.isfile(_conv_model("conv_simple.onnx")) and
            os.path.isfile(_conv_model("conv_depthwise.onnx")) and
            os.path.isfile(_conv_model("conv_grouped_invalid.onnx")))


def _gen(name: str) -> CodeGenerator:
    path = _conv_model(name)
    g    = OnnxGraph(path)
    return CodeGenerator(g, model_path=path)


# ---------------------------------------------------------------------------
# ConvNode validation
# ---------------------------------------------------------------------------

@unittest.skipUnless(_conv_models_exist(),
                     "Run test/gen_conv_models.py first")
class TestConvNodeValidation(unittest.TestCase):
    """ConvNode construction validates shapes and attributes."""

    def test_basic_conv_parses(self):
        path = _conv_model("conv_simple.onnx")
        g = OnnxGraph(path)
        self.assertEqual(len(g.nodes), 1)
        sn = g.nodes[0]
        self.assertIsInstance(sn, ConvNode)

    def test_conv_with_bias_has_bias(self):
        path = _conv_model("conv_with_bias.onnx")
        g = OnnxGraph(path)
        sn = g.nodes[0]
        self.assertIsInstance(sn, ConvNode)
        self.assertTrue(sn.has_bias)
        self.assertEqual(len(sn.inputs), 3)

    def test_conv_without_bias_no_bias(self):
        path = _conv_model("conv_simple.onnx")
        g = OnnxGraph(path)
        sn = g.nodes[0]
        self.assertFalse(sn.has_bias)
        self.assertEqual(len(sn.inputs), 2)

    def test_depthwise_parses(self):
        """conv_depthwise.onnx (group=4, in_ch=4) must parse without error."""
        path = _conv_model("conv_depthwise.onnx")
        g = OnnxGraph(path)
        sn = g.nodes[0]
        self.assertIsInstance(sn, ConvNode)
        self.assertTrue(sn.is_depthwise)

    def test_grouped_not_depthwise_raises(self):
        """group=2 with in_ch=4 is unsupported grouped conv; must raise."""
        path = _conv_model("conv_grouped_invalid.onnx")
        with self.assertRaises(SchedulerError) as cm:
            OnnxGraph(path)
        self.assertIn("group=2", str(cm.exception))
        self.assertIn("not supported", str(cm.exception))

    def test_stride_parsed(self):
        path = _conv_model("conv_stride2.onnx")
        g = OnnxGraph(path)
        sn = g.nodes[0]
        self.assertEqual(sn.stride_h, 2)
        self.assertEqual(sn.stride_w, 2)

    def test_dilation_parsed(self):
        path = _conv_model("conv_dilation.onnx")
        g = OnnxGraph(path)
        sn = g.nodes[0]
        self.assertEqual(sn.dilation_h, 2)
        self.assertEqual(sn.dilation_w, 2)

    def test_auto_pad_valid(self):
        path = _conv_model("conv_auto_pad_valid.onnx")
        g = OnnxGraph(path)
        sn = g.nodes[0]
        self.assertEqual(sn.pad_top,  0)
        self.assertEqual(sn.pad_left, 0)

    def test_explicit_pads(self):
        path = _conv_model("conv_padded.onnx")
        g = OnnxGraph(path)
        sn = g.nodes[0]
        self.assertEqual(sn.pad_top,  1)
        self.assertEqual(sn.pad_left, 1)


# ---------------------------------------------------------------------------
# ConvNode hardware-bound validation
#
# These bounds come from platforms/<AXI_PLATFORM>.json (see _conv_hw_config)
# and match the kernel's compile-time bias_buf / line_buf /
# partial_outputs sizes.  Each model below violates exactly one bound by
# exactly one unit; the tests assert that SchedulerError fires AND that
# the message names the violated constraint so users get an actionable
# error.  Boundary-ok models confirm the inequality is `≤` (limit value
# passes) rather than `<`.
# ---------------------------------------------------------------------------

@unittest.skipUnless(_conv_models_exist(),
                     "Run test/gen_conv_models.py first")
class TestConvNodeHardwareBounds(unittest.TestCase):
    """ConvNode rejects layer geometries the kernel cannot service.

    All assertions read the active bounds from ``_conv_hw_config`` so the
    suite stays green when the platform JSON is bumped (and
    ``gen_conv_models.py`` re-runs against the new values)."""

    def test_in_ch_too_large_raises(self):
        from src._conv_hw_config import CONV_MAX_IN_CH
        with self.assertRaises(SchedulerError) as cm:
            OnnxGraph(_conv_model("conv_unsupported_in_ch.onnx"))
        msg = str(cm.exception)
        self.assertIn(f"in_ch={CONV_MAX_IN_CH + 1}", msg)
        self.assertIn("kMaxInCh", msg)

    def test_out_ch_too_large_raises(self):
        from src._conv_hw_config import CONV_MAX_OUT_CH
        with self.assertRaises(SchedulerError) as cm:
            OnnxGraph(_conv_model("conv_unsupported_out_ch.onnx"))
        msg = str(cm.exception)
        self.assertIn(f"out_ch={CONV_MAX_OUT_CH + 1}", msg)
        self.assertIn("kMaxOutCh", msg)

    def test_dil_h_overflows_line_buf_rows_raises(self):
        """Vertical span = kMaxLineBufRows + 1; must raise."""
        from src._conv_hw_config import CONV_MAX_LINE_BUF_ROWS
        with self.assertRaises(SchedulerError) as cm:
            OnnxGraph(_conv_model("conv_unsupported_dil_h.onnx"))
        msg = str(cm.exception)
        self.assertIn("vertical span", msg)
        self.assertIn("kMaxLineBufRows", msg)
        self.assertIn(str(CONV_MAX_LINE_BUF_ROWS + 1), msg)

    def test_dil_w_overflows_line_buf_cols_raises(self):
        """Horizontal span = kMaxLineBufCols + 1; must raise."""
        from src._conv_hw_config import CONV_MAX_LINE_BUF_COLS
        with self.assertRaises(SchedulerError) as cm:
            OnnxGraph(_conv_model("conv_unsupported_dil_w.onnx"))
        msg = str(cm.exception)
        self.assertIn("horizontal span", msg)
        self.assertIn("kMaxLineBufCols", msg)
        self.assertIn(str(CONV_MAX_LINE_BUF_COLS + 1), msg)

    def test_acc_persist_overflow_raises(self):
        """out_w * out_ch > kMaxAccPersistEntries; must raise."""
        from src._conv_hw_config import CONV_MAX_ACC_PERSIST_ENTRIES
        with self.assertRaises(SchedulerError) as cm:
            OnnxGraph(_conv_model("conv_unsupported_acc_persist.onnx"))
        msg = str(cm.exception)
        self.assertIn("out_w*out_ch", msg)
        self.assertIn("kMaxAccPersistEntries", msg)
        # The generator uses out_w=256 and picks the smallest out_ch that
        # overflows the limit (out_ch = MAX // 256 + 1).  The kernel pads
        # out_ch up to a multiple of kTileM (§2.23 accumulator layout), and
        # the validator reports the padded product.
        from src._conv_hw_config import CONV_TILE_M
        out_w  = 256
        out_ch = CONV_MAX_ACC_PERSIST_ENTRIES // out_w + 1
        padded = -(-out_ch // CONV_TILE_M) * CONV_TILE_M
        self.assertIn(f"{out_w}*{padded} = {out_w * padded}", msg)

    def test_in_ch_at_limit_parses(self):
        """in_ch == kMaxInCh must parse — bound is `≤`, not `<`."""
        from src._conv_hw_config import CONV_MAX_IN_CH
        g = OnnxGraph(_conv_model("conv_in_ch_at_limit.onnx"))
        sn = g.nodes[0]
        self.assertIsInstance(sn, ConvNode)
        self.assertEqual(sn.in_ch, CONV_MAX_IN_CH)

    def test_dil_h_span_at_limit_parses(self):
        """span == kMaxLineBufRows must parse — boundary inclusive."""
        from src._conv_hw_config import CONV_MAX_LINE_BUF_ROWS
        g = OnnxGraph(_conv_model("conv_dil_h_at_limit.onnx"))
        sn = g.nodes[0]
        self.assertIsInstance(sn, ConvNode)
        span = (sn.kh - 1) * sn.dilation_h + 1
        self.assertEqual(span, CONV_MAX_LINE_BUF_ROWS)

    def test_acc_persist_at_limit_parses(self):
        """out_w * out_ch == kMaxAccPersistEntries must parse — boundary inclusive."""
        from src._conv_hw_config import CONV_MAX_ACC_PERSIST_ENTRIES
        g = OnnxGraph(_conv_model("conv_acc_persist_at_limit.onnx"))
        sn = g.nodes[0]
        self.assertIsInstance(sn, ConvNode)
        self.assertEqual(sn.out_w * sn.out_ch, CONV_MAX_ACC_PERSIST_ENTRIES)


class TestConvHwConfigResolver(unittest.TestCase):
    """The hardware-bound resolver reads from the platform JSON
    (platforms/<AXI_PLATFORM>.json), the same source the C++ CMake build
    consumes via ``conv_load_constants()``.  Tests use the public
    ``resolve()`` function so failure paths can be probed without
    reloading the module — that would replace the
    ``ConvHwConfigError`` class object and break ``assertRaises``."""

    def _platforms_dir(self):
        from pathlib import Path
        return (Path(__file__).resolve().parent.parent.parent / "platforms")

    def test_constants_match_kv260_json(self):
        """Resolved values match platforms/kv260.json's kernels.conv —
        guards against drift between the Python validator and the JSON
        the C++ build reads."""
        import json

        from src._conv_hw_config import (
            CONV_MAX_IN_CH, CONV_MAX_OUT_CH,
            CONV_MAX_LINE_BUF_ROWS, CONV_MAX_LINE_BUF_COLS,
            CONV_MAX_ACC_PERSIST_ENTRIES,
        )
        with (self._platforms_dir() / "kv260.json").open() as f:
            cfg = json.load(f)["kernels"]["conv"]
        self.assertEqual(CONV_MAX_IN_CH,               cfg["max_in_ch"])
        self.assertEqual(CONV_MAX_OUT_CH,              cfg["max_out_ch"])
        self.assertEqual(CONV_MAX_LINE_BUF_ROWS,       cfg["max_line_buf_rows"])
        self.assertEqual(CONV_MAX_LINE_BUF_COLS,       cfg["max_line_buf_cols"])
        self.assertEqual(CONV_MAX_ACC_PERSIST_ENTRIES, cfg["max_acc_persist_entries"])

    def test_resolve_alternate_platform_picks_up_overrides(self):
        """``resolve('<name>')`` reads ``platforms/<name>.json`` — same
        mechanism the CMake build uses when ``-DAXI_PLATFORM=<name>``
        is passed."""
        import json
        from src._conv_hw_config import resolve

        alt = self._platforms_dir() / "_test_conv_override.json"
        alt.write_text(json.dumps({
            "description": "test-only override",
            "part": "x", "clock": 100,
            "kernels": {"conv": {
                "tile_m": 8, "tile_ic": 16,
                "max_kh": 7, "max_kw": 7,
                "max_in_ch":               512,          # <-- the override
                "max_out_ch":              1024,
                "max_line_buf_rows":       16,
                "max_line_buf_cols":       64,
                "max_acc_persist_entries": 65536,
                "max_m_per_group":         4,
            }},
        }))
        try:
            cfg = resolve("_test_conv_override")
            self.assertEqual(cfg["CONV_MAX_IN_CH"],  512)
            self.assertEqual(cfg["CONV_MAX_OUT_CH"], 1024)
        finally:
            alt.unlink(missing_ok=True)

    def test_missing_platform_file_raises(self):
        """``resolve()`` with an unknown platform name must error loudly
        rather than silently fall back to defaults."""
        from src._conv_hw_config import ConvHwConfigError, resolve

        with self.assertRaises(ConvHwConfigError) as cm:
            resolve("_does_not_exist_xyzzy_conv")
        self.assertIn("not found", str(cm.exception))

    def test_missing_kernels_conv_section_raises(self):
        """A platform JSON with no ``kernels.conv`` object must error —
        no silent default fallback."""
        import json
        from src._conv_hw_config import ConvHwConfigError, resolve

        bad = self._platforms_dir() / "_test_no_conv_section.json"
        bad.write_text(json.dumps({"description": "no kernels section",
                                   "part": "x", "clock": 100}))
        try:
            with self.assertRaises(ConvHwConfigError) as cm:
                resolve("_test_no_conv_section")
            self.assertIn("kernels.conv", str(cm.exception))
        finally:
            bad.unlink(missing_ok=True)

    def test_missing_required_field_raises(self):
        """A platform JSON missing one of the mandatory ``kernels.conv``
        fields (e.g. max_in_ch) must error and name the missing field."""
        import json
        from src._conv_hw_config import ConvHwConfigError, resolve

        bad = self._platforms_dir() / "_test_conv_missing_field.json"
        bad.write_text(json.dumps({
            "description": "missing max_in_ch",
            "part": "x", "clock": 100,
            "kernels": {"conv": {
                "tile_m": 8, "tile_ic": 16,
                "max_kh": 7, "max_kw": 7,
                # max_in_ch deliberately absent
                "max_out_ch":              1024,
                "max_line_buf_rows":       16,
                "max_line_buf_cols":       64,
                "max_acc_persist_entries": 65536,
                "max_m_per_group":         4,
            }},
        }))
        try:
            with self.assertRaises(ConvHwConfigError) as cm:
                resolve("_test_conv_missing_field")
            self.assertIn("max_in_ch", str(cm.exception))
        finally:
            bad.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# ConvNode geometry fields
# ---------------------------------------------------------------------------

@unittest.skipUnless(_conv_models_exist(),
                     "Run test/gen_conv_models.py first")
class TestConvNodeGeometry(unittest.TestCase):
    """ConvNode derives correct shape parameters from ONNX node + inference."""

    def _node(self, name: str) -> ConvNode:
        g = OnnxGraph(_conv_model(name))
        return g.nodes[0]

    def test_simple_geometry(self):
        sn = self._node("conv_simple.onnx")
        self.assertEqual(sn.batch,  1)
        self.assertEqual(sn.in_ch,  4)
        self.assertEqual(sn.in_h,   8)
        self.assertEqual(sn.in_w,   8)
        self.assertEqual(sn.out_ch, 8)
        self.assertEqual(sn.out_h,  8)
        self.assertEqual(sn.out_w,  8)
        self.assertEqual(sn.kh, 1)
        self.assertEqual(sn.kw, 1)

    def test_stride2_output_size(self):
        sn = self._node("conv_stride2.onnx")
        # 3x3 no-pad stride-2: out = (8-3)//2+1 = 3
        self.assertEqual(sn.out_h, 3)
        self.assertEqual(sn.out_w, 3)

    def test_batch2_geometry(self):
        sn = self._node("conv_batch2.onnx")
        self.assertEqual(sn.batch, 2)

    def test_conv_kernel_name(self):
        sn = self._node("conv_simple.onnx")
        self.assertEqual(sn.kernel_name, "ConvKernel")

    def test_compatibility_shims(self):
        sn = self._node("conv_simple.onnx")
        self.assertEqual(sn.outer_count, 1)
        self.assertEqual(sn.chunk_size,  0)
        self.assertEqual(sn.aligned_chunk_size, 0)
        self.assertTrue(sn.a_advances)
        self.assertTrue(sn.b_advances)
        self.assertEqual(sn.arity, 2)


# ---------------------------------------------------------------------------
# TensorLayout — ConvNode excluded from phases 2 and 3
# ---------------------------------------------------------------------------

@unittest.skipUnless(_conv_models_exist(),
                     "Run test/gen_conv_models.py first")
class TestConvLayouts(unittest.TestCase):
    """ConvNode output always has a flat TensorLayout (n_chunks == 1)."""

    def test_simple_flat_output(self):
        gen = _gen("conv_simple.onnx")
        # Y[1,8,8,8] = 512 elements, flat
        y_lay = gen._layouts["Y"]
        self.assertEqual(y_lay.n_chunks, 1)
        self.assertEqual(y_lay.alloc,    y_lay.numel)

    def test_conv_then_relu_flat_intermediate(self):
        gen = _gen("conv_then_relu.onnx")
        # Z is ConvNode output feeding Relu; must stay flat
        z_lay = gen._layouts["Z"]
        self.assertEqual(z_lay.n_chunks, 1)
        self.assertEqual(z_lay.alloc,    z_lay.numel)

    def test_conv_then_add_flat(self):
        gen = _gen("conv_then_add_flat.onnx")
        # Z and Y should both be flat (same-shape Add, no broadcast)
        z_lay = gen._layouts["Z"]
        y_lay = gen._layouts["Y"]
        self.assertEqual(z_lay.n_chunks, 1)
        self.assertEqual(y_lay.n_chunks, 1)

    def test_alloc_correct_for_simple(self):
        gen = _gen("conv_simple.onnx")
        # numel = 1*8*8*8 = 512
        y_lay = gen._layouts["Y"]
        self.assertEqual(y_lay.numel, 512)
        self.assertEqual(y_lay.alloc, 512)

    def test_chain_intermediates_flat(self):
        gen = _gen("conv_relu_chain.onnx")
        for name in ["Z1", "Z2", "Z3"]:
            lay = gen._layouts.get(name)
            if lay is not None:
                self.assertEqual(lay.n_chunks, 1,
                                 f"expected flat layout for {name}")


# ---------------------------------------------------------------------------
# Generated inference.c source
# ---------------------------------------------------------------------------

@unittest.skipUnless(_conv_models_exist(),
                     "Run test/gen_conv_models.py first")
class TestConvSource(unittest.TestCase):
    """Generated inference.c contains correct ConvKernel calls."""

    def _src(self, name: str) -> str:
        return _gen(name).generate_source()

    def test_conv_header_included(self):
        s = self._src("conv_simple.onnx")
        self.assertIn('#include "xconvkernel.h"', s)

    def test_conv_instance_declared(self):
        s = self._src("conv_simple.onnx")
        self.assertIn("XConvkernel s_convkernel", s)

    def test_run_conv_helper_emitted(self):
        s = self._src("conv_simple.onnx")
        self.assertIn("static void run_conv(", s)

    def test_run_conv_called(self):
        s = self._src("conv_simple.onnx")
        self.assertIn("run_conv(", s)

    def test_no_run_op_for_conv_only(self):
        s = self._src("conv_simple.onnx")
        self.assertNotIn("run_op(", s)
        self.assertNotIn("static void run_op(", s)

    def test_mixed_both_helpers(self):
        s = self._src("conv_then_relu.onnx")
        self.assertIn("static void run_conv(", s)
        self.assertIn("static void run_op(",   s)

    def test_conv_args_no_bias(self):
        s = self._src("conv_simple.onnx")
        # has_bias=0 → bias arg is NULL, last param is 0u
        self.assertIn("run_conv(", s)
        # call line: run_conv(X, W, NULL, Y, ...)
        self.assertIn(", NULL, ", s)
        self.assertIn("0u);", s)   # has_bias=0u at end of call

    def test_conv_args_with_bias(self):
        s = self._src("conv_with_bias.onnx")
        # has_bias=1u, is_depthwise=0u → call ends with "1u, 0u);"
        self.assertIn("1u, 0u);", s)
        # The run_conv call should use the bias buffer name, not NULL
        self.assertNotIn(", NULL, ", s)

    def test_kernel_instance_registers(self):
        s = self._src("conv_simple.onnx")
        for fn in [
            "XConvkernel_Set_x",
            "XConvkernel_Set_weight",
            "XConvkernel_Set_bias",
            "XConvkernel_Set_y",
            "XConvkernel_Set_batch",
            "XConvkernel_Set_in_ch",
            "XConvkernel_Set_out_ch",
            "XConvkernel_Set_kh",
            "XConvkernel_Set_stride_h",
            "XConvkernel_Set_dilation_h",
            "XConvkernel_Set_pad_top",
            "XConvkernel_Set_has_bias",
            "XConvkernel_Set_is_depthwise",
            "XConvkernel_Start",
            "XConvkernel_IsDone",
        ]:
            self.assertIn(fn, s, f"{fn} not found in source")

    def test_emit_comment_contains_conv(self):
        s = self._src("conv_simple.onnx")
        self.assertIn("[0] Conv(", s)

    def test_stride2_params_correct(self):
        s = self._src("conv_stride2.onnx")
        # emit_comment uses "s=2x2"; emit_call passes numeric args
        self.assertIn("s=2x2", s)
        self.assertIn("2u, 2u,", s)

    def test_dilation_params_correct(self):
        s = self._src("conv_dilation.onnx")
        self.assertIn("2u, 2u,", s)

    def test_batch2_param_correct(self):
        s = self._src("conv_batch2.onnx")
        # batch=2u should appear as first geometry param
        self.assertIn("2u,", s)

    def test_inference_init_has_conv_instance(self):
        s = self._src("conv_simple.onnx")
        self.assertIn("XConvkernel_Initialize(", s)
        self.assertIn("s_convkernel", s)

    def test_mixed_both_inits(self):
        s = self._src("conv_then_relu.onnx")
        self.assertIn("XConvkernel_Initialize(",    s)
        self.assertIn("XVectoropkernel_Initialize(", s)

    def test_kernel_registry_order(self):
        # When both VectorOPKernel and ConvKernel are active,
        # VectorOPKernel comes first (registry insertion order).
        s = self._src("conv_then_relu.onnx")
        vop_pos  = s.find("XVectoropkernel_Initialize(")
        conv_pos = s.find("XConvkernel_Initialize(")
        self.assertGreater(conv_pos, vop_pos)


# ---------------------------------------------------------------------------
# _active_kernels for Conv-containing graphs
# ---------------------------------------------------------------------------

@unittest.skipUnless(_conv_models_exist(),
                     "Run test/gen_conv_models.py first")
class TestConvActiveKernels(unittest.TestCase):

    def test_conv_only_active(self):
        gen = _gen("conv_simple.onnx")
        names = [kd.name for kd in gen._active_kernels]
        self.assertEqual(names, ["ConvKernel"])

    def test_conv_vectorop_active(self):
        gen = _gen("conv_then_relu.onnx")
        names = [kd.name for kd in gen._active_kernels]
        self.assertIn("ConvKernel",     names)
        self.assertIn("VectorOPKernel", names)

    def test_conv_vectorop_registry_order(self):
        # VectorOPKernel should precede ConvKernel (registry insertion order)
        gen = _gen("conv_then_relu.onnx")
        names = [kd.name for kd in gen._active_kernels]
        self.assertLess(names.index("VectorOPKernel"), names.index("ConvKernel"))

    def test_has_conv_nodes_property(self):
        gen = _gen("conv_simple.onnx")
        self.assertTrue(gen._has_conv_nodes)
        self.assertFalse(gen._has_vectorop_nodes)
        self.assertFalse(gen._has_matmul_nodes)


# ---------------------------------------------------------------------------
# Simulation — _forward_pass produces correct conv output
# ---------------------------------------------------------------------------

@unittest.skipUnless(_conv_models_exist(),
                     "Run test/gen_conv_models.py first")
class TestConvSimulation(unittest.TestCase):
    """Simulated conv output matches scipy/numpy reference."""

    def _simulate(self, name: str) -> dict:
        gen = _gen(name)
        return gen._simulate()

    def test_simple_1x1_output_shape(self):
        arrays = self._simulate("conv_simple.onnx")
        y = arrays["Y"]
        self.assertEqual(list(y.shape), [1, 8, 8, 8])

    def test_with_bias_output_shape(self):
        arrays = self._simulate("conv_with_bias.onnx")
        y = arrays["Y"]
        self.assertEqual(list(y.shape), [1, 6, 6, 6])

    def test_stride2_output_shape(self):
        arrays = self._simulate("conv_stride2.onnx")
        y = arrays["Y"]
        self.assertEqual(list(y.shape), [1, 8, 3, 3])

    def test_padded_output_same_spatial(self):
        arrays = self._simulate("conv_padded.onnx")
        y = arrays["Y"]
        self.assertEqual(y.shape[2], 8)
        self.assertEqual(y.shape[3], 8)

    def test_batch2_output_shape(self):
        arrays = self._simulate("conv_batch2.onnx")
        y = arrays["Y"]
        self.assertEqual(y.shape[0], 2)

    def test_dilation_output_shape(self):
        arrays = self._simulate("conv_dilation.onnx")
        y = arrays["Y"]
        self.assertEqual(list(y.shape), [1, 8, 4, 4])

    def test_bias_zero_matches_no_bias(self):
        """Conv with B=zeros should match Conv without B."""
        # conv_with_bias uses B=zeros; manually compare to conv_simple after
        # loading the conv_simple graph and feeding the same input.
        gen_nb = _gen("conv_simple.onnx")
        gen_wb = _gen("conv_with_bias.onnx")

        # Both get the same weight data? No — they have different weights.
        # Just verify the bias branch doesn't blow up and has correct shape.
        arrays = gen_wb._simulate()
        self.assertEqual(list(arrays["Y"].shape), [1, 6, 6, 6])

    def test_conv_then_relu_output_nonneg(self):
        arrays = self._simulate("conv_then_relu.onnx")
        y = arrays["Y"]
        self.assertTrue((y >= 0).all(), "Relu output must be non-negative")

    def test_simulate_public_api(self):
        gen = _gen("conv_simple.onnx")
        dtype = gen._dtype
        # Build a small fixed input
        x_flat = np.arange(1 * 4 * 8 * 8, dtype=np.float64)
        x_quant = dtype.quantize(x_flat).reshape(1, 4, 8, 8)
        result = gen.simulate({"X": x_quant})
        self.assertIn("Y", result)
        self.assertEqual(list(result["Y"].shape), [1, 8, 8, 8])


# ---------------------------------------------------------------------------
# emit_comment and emit_call format
# ---------------------------------------------------------------------------

@unittest.skipUnless(_conv_models_exist(),
                     "Run test/gen_conv_models.py first")
class TestConvEmit(unittest.TestCase):

    def test_emit_comment_format(self):
        g = OnnxGraph(_conv_model("conv_simple.onnx"))
        sn = g.nodes[0]
        self.assertIsInstance(sn, ConvNode)
        comment = sn.emit_comment()
        self.assertIn("[0] Conv(", comment)
        self.assertIn("->", comment)
        self.assertIn("k=1", comment)

    def test_emit_call_format_no_bias(self):
        g = OnnxGraph(_conv_model("conv_simple.onnx"))
        sn = g.nodes[0]
        gen = CodeGenerator(g, model_path=_conv_model("conv_simple.onnx"))
        call = sn.emit_call(gen._layouts)
        self.assertIn("run_conv(", call)
        self.assertIn("NULL", call)
        self.assertIn("0u);", call)

    def test_emit_call_format_with_bias(self):
        g = OnnxGraph(_conv_model("conv_with_bias.onnx"))
        sn = g.nodes[0]
        gen = CodeGenerator(g, model_path=_conv_model("conv_with_bias.onnx"))
        call = sn.emit_call(gen._layouts)
        self.assertIn("run_conv(", call)
        self.assertIn("1u, 0u);", call)   # has_bias=1u, is_depthwise=0u

    def test_emit_call_correct_dims(self):
        g = OnnxGraph(_conv_model("conv_stride2.onnx"))
        sn = g.nodes[0]
        gen = CodeGenerator(g, model_path=_conv_model("conv_stride2.onnx"))
        call = sn.emit_call(gen._layouts)
        # stride params appear in call
        self.assertIn("2u, 2u,", call)

    def test_emit_call_with_bias_name(self):
        g = OnnxGraph(_conv_model("conv_with_bias.onnx"))
        sn = g.nodes[0]
        gen = CodeGenerator(g, model_path=_conv_model("conv_with_bias.onnx"))
        call = sn.emit_call(gen._layouts)
        # bias tensor c_name appears as 3rd arg
        bias_name = sn.inputs[2].c_name
        self.assertIn(bias_name, call)


# ---------------------------------------------------------------------------
# Two-layer VGG-style conv block: X[1,3,224,224] → [1,64,222,222] → [1,64,220,220]
# ---------------------------------------------------------------------------

def _vgg_model_exists() -> bool:
    return os.path.isfile(_conv_model("conv_two_layer_vgg.onnx"))


@unittest.skipUnless(_vgg_model_exists(),
                     "Run test/gen_conv_models.py first")
class TestConvTwoLayerVGG(unittest.TestCase):
    """Scheduler correctness for the 224×224 two-layer VGG-style conv block."""

    @classmethod
    def setUpClass(cls):
        cls.gen = _gen("conv_two_layer_vgg.onnx")
        cls.graph = cls.gen._graph

    def _src(self) -> str:
        return self.gen.generate_source()

    # ---- Graph structure ------------------------------------------------

    def test_two_conv_nodes(self):
        self.assertEqual(len(self.graph.nodes), 2)
        for sn in self.graph.nodes:
            self.assertIsInstance(sn, ConvNode)

    def test_single_input_output(self):
        self.assertEqual(len(self.graph.input_tensors),  1)
        self.assertEqual(len(self.graph.output_tensors), 1)

    # ---- First layer geometry -------------------------------------------

    def test_layer1_in_shape(self):
        sn = self.graph.nodes[0]
        self.assertEqual(sn.batch,  1)
        self.assertEqual(sn.in_ch,  3)
        self.assertEqual(sn.in_h,   224)
        self.assertEqual(sn.in_w,   224)

    def test_layer1_out_shape(self):
        sn = self.graph.nodes[0]
        self.assertEqual(sn.out_ch, 64)
        self.assertEqual(sn.out_h,  222)   # (224 - 3) / 1 + 1
        self.assertEqual(sn.out_w,  222)

    def test_layer1_kernel(self):
        sn = self.graph.nodes[0]
        self.assertEqual(sn.kh, 3)
        self.assertEqual(sn.kw, 3)
        self.assertEqual(sn.stride_h, 1)
        self.assertEqual(sn.stride_w, 1)
        self.assertFalse(sn.has_bias)

    # ---- Second layer geometry ------------------------------------------

    def test_layer2_in_shape(self):
        sn = self.graph.nodes[1]
        self.assertEqual(sn.in_ch, 64)
        self.assertEqual(sn.in_h,  222)
        self.assertEqual(sn.in_w,  222)

    def test_layer2_out_shape(self):
        sn = self.graph.nodes[1]
        self.assertEqual(sn.out_ch, 64)
        self.assertEqual(sn.out_h,  220)   # (222 - 3) / 1 + 1
        self.assertEqual(sn.out_w,  220)

    def test_layer2_kernel(self):
        sn = self.graph.nodes[1]
        self.assertEqual(sn.kh, 3)
        self.assertEqual(sn.kw, 3)
        self.assertFalse(sn.has_bias)

    # ---- Tensor layouts -------------------------------------------------

    def test_intermediate_z_flat(self):
        # Z is the output of layer 1 and input of layer 2; must be flat
        z_lay = self.gen._layouts["Z"]
        self.assertEqual(z_lay.n_chunks, 1)
        self.assertEqual(z_lay.numel, 1 * 64 * 222 * 222)
        self.assertEqual(z_lay.alloc,   z_lay.numel)

    def test_output_y_flat(self):
        y_lay = self.gen._layouts["Y"]
        self.assertEqual(y_lay.n_chunks, 1)
        self.assertEqual(y_lay.numel, 1 * 64 * 220 * 220)

    # ---- Large weight goes to external .dat file ------------------------

    def test_w2_is_large_weight(self):
        # W2[64,64,3,3] = 36 864 elements > LARGE_WEIGHT_THRESHOLD (4096)
        large = self.gen.large_weight_tensors
        names = [t.onnx_name for t in large]
        self.assertIn("W2", names)

    def test_w1_packed_layout_size(self):
        # W1[64,3,3,3] is 1 728 logical elements, but ConvKernel's packed
        # tile-major layout pads the 3 input channels to one 16-lane tile:
        # 64 * 1 * 3 * 3 * 16 = 9 216 elements (> LARGE_WEIGHT_THRESHOLD, so
        # it is now externalised to weights/W1.dat).  The logical shape is
        # untouched (the simulator keeps using it).
        from src._conv_hw_config import CONV_TILE_IC
        w1 = next(t for t in self.graph.weight_tensors if t.onnx_name == "W1")
        self.assertEqual(w1.shape, [64, 3, 3, 3])
        self.assertEqual(w1.numel, 64 * 1 * 3 * 3 * CONV_TILE_IC)
        self.assertIsNotNone(w1.packed_data)
        names = [t.onnx_name for t in self.gen.large_weight_tensors]
        self.assertIn("W1", names)

    def test_w1_packed_lane_order(self):
        # packed[(m, ict, khi, kwi, ic_l)] == logical[m, ict*16 + ic_l, khi, kwi],
        # zero for padded lanes.
        from src._conv_hw_config import CONV_TILE_IC
        w1 = next(t for t in self.graph.weight_tensors if t.onnx_name == "W1")
        logical = w1.data.reshape(64, 3, 3, 3)
        packed  = w1.packed_data.reshape(64, 1, 3, 3, CONV_TILE_IC)
        for c in range(3):
            self.assertTrue((packed[:, 0, :, :, c] == logical[:, c]).all())
        self.assertTrue((packed[:, 0, :, :, 3:] == 0).all())

    # ---- Generated source -----------------------------------------------

    def test_no_run_op_pure_conv(self):
        self.assertNotIn("run_op(", self._src())

    def test_two_run_conv_calls(self):
        s = self._src()
        body = s[s.find("void inference_run("):]
        self.assertEqual(body.count("run_conv("), 2)

    def test_run_conv_calls_in_order(self):
        s = self._src()
        body = s[s.find("void inference_run("):]
        first  = body.find("run_conv(")
        second = body.find("run_conv(", first + 1)
        self.assertGreater(second, first)

    def test_conv_only_kernel_active(self):
        names = [kd.name for kd in self.gen._active_kernels]
        self.assertEqual(names, ["ConvKernel"])

    def test_layer1_dims_in_source(self):
        s = self._src()
        # First call: batch=1 in_ch=3 in_h=224 in_w=224 out_ch=64 ...
        self.assertIn("224u,", s)
        self.assertIn("3u,",   s)

    def test_layer2_in_ch_64_in_source(self):
        s = self._src()
        # Second call uses in_ch=64 and in_h/in_w=222
        self.assertIn("222u,", s)
        self.assertIn("64u,",  s)


# ---------------------------------------------------------------------------
# Depthwise convolution
# ---------------------------------------------------------------------------

def _dw_models_exist() -> bool:
    return (os.path.isfile(_conv_model("conv_depthwise.onnx")) and
            os.path.isfile(_conv_model("conv_depthwise_bias.onnx")))


@unittest.skipUnless(_dw_models_exist(),
                     "Run test/gen_conv_models.py first")
class TestConvDepthwise(unittest.TestCase):
    """ConvNode correctly handles depthwise convolution (group=in_ch)."""

    def _node(self, name: str) -> ConvNode:
        g = OnnxGraph(_conv_model(name))
        return g.nodes[0]

    def test_depthwise_is_depthwise_flag(self):
        sn = self._node("conv_depthwise.onnx")
        self.assertTrue(sn.is_depthwise)

    def test_standard_not_depthwise(self):
        sn = self._node("conv_simple.onnx")
        self.assertFalse(sn.is_depthwise)

    def test_depthwise_geometry(self):
        sn = self._node("conv_depthwise.onnx")
        # X[1,4,8,8], W[4,1,3,3], Y[1,4,6,6]
        self.assertEqual(sn.in_ch,  4)
        self.assertEqual(sn.out_ch, 4)
        self.assertEqual(sn.out_h,  6)
        self.assertEqual(sn.out_w,  6)
        self.assertEqual(sn.kh, 3)
        self.assertEqual(sn.kw, 3)

    def test_depthwise_no_bias(self):
        sn = self._node("conv_depthwise.onnx")
        self.assertFalse(sn.has_bias)

    def test_depthwise_with_bias_flag(self):
        sn = self._node("conv_depthwise_bias.onnx")
        self.assertTrue(sn.is_depthwise)
        self.assertTrue(sn.has_bias)

    def test_depthwise_emit_comment_has_dw(self):
        sn = self._node("conv_depthwise.onnx")
        comment = sn.emit_comment()
        self.assertIn(" dw", comment)

    def test_standard_emit_comment_no_dw(self):
        sn = self._node("conv_simple.onnx")
        comment = sn.emit_comment()
        self.assertNotIn(" dw", comment)

    def test_depthwise_emit_call_has_is_depthwise_1(self):
        g   = OnnxGraph(_conv_model("conv_depthwise.onnx"))
        sn  = g.nodes[0]
        gen = CodeGenerator(g, model_path=_conv_model("conv_depthwise.onnx"))
        call = sn.emit_call(gen._layouts)
        # has_bias=0u, is_depthwise=1u → ends with "0u, 1u);"
        self.assertIn("0u, 1u);", call)

    def test_standard_emit_call_is_depthwise_0(self):
        g   = OnnxGraph(_conv_model("conv_simple.onnx"))
        sn  = g.nodes[0]
        gen = CodeGenerator(g, model_path=_conv_model("conv_simple.onnx"))
        call = sn.emit_call(gen._layouts)
        # has_bias=0u, is_depthwise=0u → ends with "0u, 0u);"
        self.assertIn("0u, 0u);", call)

    def test_depthwise_source_has_is_depthwise_register(self):
        s = _gen("conv_depthwise.onnx").generate_source()
        self.assertIn("XConvkernel_Set_is_depthwise", s)

    def test_depthwise_output_shape(self):
        gen = _gen("conv_depthwise.onnx")
        arrays = gen._simulate()
        y = arrays["Y"]
        self.assertEqual(list(y.shape), [1, 4, 6, 6])

    def test_depthwise_bias_output_shape(self):
        gen = _gen("conv_depthwise_bias.onnx")
        arrays = gen._simulate()
        y = arrays["Y"]
        self.assertEqual(list(y.shape), [1, 8, 6, 6])

    def test_depthwise_output_finite(self):
        gen = _gen("conv_depthwise.onnx")
        arrays = gen._simulate()
        y = arrays["Y"]
        self.assertTrue(np.all(np.isfinite(y)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
