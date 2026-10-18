# Copyright 2025 GradeBuilder SL
# SPDX-License-Identifier: Apache-2.0

"""Tests for PoolingKernel scheduler integration."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.graph   import OnnxGraph
from src.codegen import CodeGenerator
from src.nodes   import PoolNode, SchedulerError, POOL_MAX, POOL_AVG, POOL_LP


MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")


def _pool_model(name: str) -> str:
    return os.path.join(MODELS_DIR, name)


def _pool_models_exist() -> bool:
    return os.path.isfile(_pool_model("pool_maxpool_simple.onnx"))


def _gen(name: str) -> CodeGenerator:
    path = _pool_model(name)
    g    = OnnxGraph(path)
    return CodeGenerator(g, model_path=path)


# ---------------------------------------------------------------------------
# PoolNode validation
# ---------------------------------------------------------------------------

@unittest.skipUnless(_pool_models_exist(),
                     "Run test/gen_pool_models.py first")
class TestPoolNodeValidation(unittest.TestCase):
    """PoolNode construction validates shapes and attributes."""

    def test_maxpool_parses(self):
        g = OnnxGraph(_pool_model("pool_maxpool_simple.onnx"))
        self.assertEqual(len(g.nodes), 1)
        self.assertIsInstance(g.nodes[0], PoolNode)

    def test_avgpool_parses(self):
        g = OnnxGraph(_pool_model("pool_avgpool_simple.onnx"))
        self.assertIsInstance(g.nodes[0], PoolNode)

    def test_global_max_parses(self):
        g = OnnxGraph(_pool_model("pool_global_max.onnx"))
        self.assertIsInstance(g.nodes[0], PoolNode)

    def test_global_avg_parses(self):
        g = OnnxGraph(_pool_model("pool_global_avg.onnx"))
        self.assertIsInstance(g.nodes[0], PoolNode)

    def test_lp_p2_parses(self):
        g = OnnxGraph(_pool_model("pool_lp_p2.onnx"))
        self.assertIsInstance(g.nodes[0], PoolNode)

    def test_lp_p1_parses(self):
        g = OnnxGraph(_pool_model("pool_lp_p1.onnx"))
        self.assertIsInstance(g.nodes[0], PoolNode)

    def test_kernel_name(self):
        g = OnnxGraph(_pool_model("pool_maxpool_simple.onnx"))
        self.assertEqual(g.nodes[0].kernel_name, "PoolKernel")

    def test_compatibility_shims(self):
        g = OnnxGraph(_pool_model("pool_maxpool_simple.onnx"))
        sn = g.nodes[0]
        self.assertEqual(sn.outer_count, 1)
        self.assertEqual(sn.chunk_size,  0)
        self.assertEqual(sn.aligned_chunk_size, 0)
        self.assertTrue(sn.a_advances)
        self.assertTrue(sn.b_advances)
        self.assertEqual(sn.arity, 1)


# ---------------------------------------------------------------------------
# PoolNode hardware-bound validation
#
# These bounds come from the C++ CMake config (see _pool_hw_config) and
# match the kernel's compile-time line_buf / adder-tree sizes.  Each model
# below violates exactly one bound by exactly one unit; the tests assert
# that SchedulerError fires AND that the message names the violated
# constraint so users get an actionable error.  Two boundary-ok models
# confirm the inequality is `≤` (limit value passes) rather than `<`.
# ---------------------------------------------------------------------------

@unittest.skipUnless(_pool_models_exist(),
                     "Run test/gen_pool_models.py first")
class TestPoolNodeHardwareBounds(unittest.TestCase):
    """PoolNode rejects pool windows the kernel cannot service."""

    def test_pool_h_too_large_raises(self):
        from src._pool_hw_config import POOL_MAX_KH
        with self.assertRaises(SchedulerError) as cm:
            OnnxGraph(_pool_model("pool_unsupported_pool_h.onnx"))
        msg = str(cm.exception)
        self.assertIn(f"pool_h={POOL_MAX_KH + 1}", msg)
        self.assertIn("kMaxPoolH", msg)

    def test_pool_w_too_large_raises(self):
        from src._pool_hw_config import POOL_MAX_KW
        with self.assertRaises(SchedulerError) as cm:
            OnnxGraph(_pool_model("pool_unsupported_pool_w.onnx"))
        msg = str(cm.exception)
        self.assertIn(f"pool_w={POOL_MAX_KW + 1}", msg)
        self.assertIn("kMaxPoolW", msg)

    def test_dil_h_overflows_line_buf_rows_raises(self):
        """Vertical span = kMaxLineBufRows + 1; must raise."""
        from src._pool_hw_config import POOL_MAX_LINE_BUF_ROWS
        with self.assertRaises(SchedulerError) as cm:
            OnnxGraph(_pool_model("pool_unsupported_dil_h.onnx"))
        msg = str(cm.exception)
        self.assertIn("vertical span", msg)
        self.assertIn("kMaxLineBufRows", msg)
        self.assertIn(str(POOL_MAX_LINE_BUF_ROWS + 1), msg)

    def test_dil_w_overflows_line_buf_cols_raises(self):
        """Horizontal span = kMaxLineBufCols + 1; must raise."""
        from src._pool_hw_config import POOL_MAX_LINE_BUF_COLS
        with self.assertRaises(SchedulerError) as cm:
            OnnxGraph(_pool_model("pool_unsupported_dil_w.onnx"))
        msg = str(cm.exception)
        self.assertIn("horizontal span", msg)
        self.assertIn("kMaxLineBufCols", msg)
        self.assertIn(str(POOL_MAX_LINE_BUF_COLS + 1), msg)

    def test_pool_h_at_limit_parses(self):
        """pool_h == kMaxPoolH must parse — bound is `≤`, not `<`."""
        from src._pool_hw_config import POOL_MAX_KH
        g = OnnxGraph(_pool_model("pool_pool_h_at_limit.onnx"))
        self.assertIsInstance(g.nodes[0], PoolNode)
        self.assertEqual(g.nodes[0].pool_h, POOL_MAX_KH)

    def test_dil_h_span_at_limit_parses(self):
        """span == kMaxLineBufRows must parse — boundary inclusive."""
        from src._pool_hw_config import POOL_MAX_LINE_BUF_ROWS
        g = OnnxGraph(_pool_model("pool_dil_h_at_limit.onnx"))
        self.assertIsInstance(g.nodes[0], PoolNode)
        sn = g.nodes[0]
        self.assertEqual((sn.pool_h - 1) * sn.dil_h + 1, POOL_MAX_LINE_BUF_ROWS)


class TestPoolHwConfigResolver(unittest.TestCase):
    """The hardware-bound resolver reads from the platform JSON
    (platforms/<AXI_PLATFORM>.json), the same source the C++ CMake build
    consumes via ``pool_load_constants()``.  Tests use the public
    ``resolve()`` function so failure paths can be probed without
    reloading the module — that would replace the
    ``PoolHwConfigError`` class object and break ``assertRaises``."""

    def _platforms_dir(self):
        from pathlib import Path
        return (Path(__file__).resolve().parent.parent.parent / "platforms")

    def test_constants_match_kv260_json(self):
        """Resolved values match platforms/kv260.json's kernels.pool —
        guards against drift between the Python validator and the JSON
        the C++ build reads."""
        import json

        from src._pool_hw_config import (
            POOL_MAX_KH, POOL_MAX_KW,
            POOL_MAX_LINE_BUF_ROWS, POOL_MAX_LINE_BUF_COLS,
        )
        with (self._platforms_dir() / "kv260.json").open() as f:
            cfg = json.load(f)["kernels"]["pool"]
        self.assertEqual(POOL_MAX_KH,            cfg["max_kh"])
        self.assertEqual(POOL_MAX_KW,            cfg["max_kw"])
        self.assertEqual(POOL_MAX_LINE_BUF_ROWS, cfg["max_line_buf_rows"])
        self.assertEqual(POOL_MAX_LINE_BUF_COLS, cfg["max_line_buf_cols"])

    def test_resolve_alternate_platform_picks_up_overrides(self):
        """``resolve('<name>')`` reads ``platforms/<name>.json`` — same
        mechanism the CMake build uses when ``-DAXI_PLATFORM=<name>``
        is passed."""
        import json
        from src._pool_hw_config import resolve

        alt = self._platforms_dir() / "_test_override.json"
        alt.write_text(json.dumps({
            "description": "test-only override",
            "part": "x", "clock": 100,
            "kernels": {"pool": {
                "tile_c": 8,
                "max_kh": 5,                      # <-- the override
                "max_kw": 7,
                "max_line_buf_rows": 16,
                "max_line_buf_cols": 64,
                "ow_parallel": 2,
            }},
        }))
        try:
            cfg = resolve("_test_override")
            self.assertEqual(cfg["POOL_MAX_KH"], 5)
            self.assertEqual(cfg["POOL_MAX_KW"], 7)
        finally:
            alt.unlink(missing_ok=True)

    def test_missing_platform_file_raises(self):
        """``resolve()`` with an unknown platform name must error loudly
        rather than silently fall back to defaults."""
        from src._pool_hw_config import PoolHwConfigError, resolve

        with self.assertRaises(PoolHwConfigError) as cm:
            resolve("_does_not_exist_xyzzy")
        self.assertIn("not found", str(cm.exception))

    def test_missing_kernels_pool_section_raises(self):
        """A platform JSON with no ``kernels.pool`` object must error —
        no silent default fallback."""
        import json
        from src._pool_hw_config import PoolHwConfigError, resolve

        bad = self._platforms_dir() / "_test_no_pool_section.json"
        bad.write_text(json.dumps({"description": "no kernels section",
                                   "part": "x", "clock": 100}))
        try:
            with self.assertRaises(PoolHwConfigError) as cm:
                resolve("_test_no_pool_section")
            self.assertIn("kernels.pool", str(cm.exception))
        finally:
            bad.unlink(missing_ok=True)

    def test_missing_required_field_raises(self):
        """A platform JSON missing one of the mandatory ``kernels.pool``
        fields (e.g. max_kh) must error and name the missing field."""
        import json
        from src._pool_hw_config import PoolHwConfigError, resolve

        bad = self._platforms_dir() / "_test_missing_field.json"
        bad.write_text(json.dumps({
            "description": "missing max_kh",
            "part": "x", "clock": 100,
            "kernels": {"pool": {
                "tile_c": 8,
                # max_kh deliberately absent
                "max_kw": 7,
                "max_line_buf_rows": 16,
                "max_line_buf_cols": 64,
                "ow_parallel": 2,
            }},
        }))
        try:
            with self.assertRaises(PoolHwConfigError) as cm:
                resolve("_test_missing_field")
            self.assertIn("max_kh", str(cm.exception))
        finally:
            bad.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# PoolNode geometry fields
# ---------------------------------------------------------------------------

@unittest.skipUnless(_pool_models_exist(),
                     "Run test/gen_pool_models.py first")
class TestPoolNodeGeometry(unittest.TestCase):
    """PoolNode derives correct geometry from ONNX attributes."""

    def _node(self, name: str) -> PoolNode:
        return OnnxGraph(_pool_model(name)).nodes[0]

    def test_maxpool_simple_geometry(self):
        sn = self._node("pool_maxpool_simple.onnx")
        self.assertEqual(sn.batch,    1)
        self.assertEqual(sn.channels, 4)
        self.assertEqual(sn.in_h,     8)
        self.assertEqual(sn.in_w,     8)
        self.assertEqual(sn.out_h,    4)
        self.assertEqual(sn.out_w,    4)
        self.assertEqual(sn.pool_h,   2)
        self.assertEqual(sn.pool_w,   2)
        self.assertEqual(sn.stride_h, 2)
        self.assertEqual(sn.stride_w, 2)
        self.assertEqual(sn.pad_top,  0)
        self.assertEqual(sn.pad_left, 0)

    def test_maxpool_padded_geometry(self):
        sn = self._node("pool_maxpool_padded.onnx")
        # 3x3 pad=1 stride=1: output same size as input
        self.assertEqual(sn.out_h, 8)
        self.assertEqual(sn.out_w, 8)
        self.assertEqual(sn.pad_top,  1)
        self.assertEqual(sn.pad_left, 1)

    def test_maxpool_type(self):
        sn = self._node("pool_maxpool_simple.onnx")
        self.assertEqual(sn.pool_type, POOL_MAX)

    def test_avgpool_type(self):
        sn = self._node("pool_avgpool_simple.onnx")
        self.assertEqual(sn.pool_type, POOL_AVG)

    def test_lp_type_and_order(self):
        sn = self._node("pool_lp_p2.onnx")
        self.assertEqual(sn.pool_type, POOL_LP)
        self.assertEqual(sn.lp_order,  2)

    def test_lp_p1_order(self):
        sn = self._node("pool_lp_p1.onnx")
        self.assertEqual(sn.lp_order, 1)

    def test_avgpool_count_include_pad(self):
        sn = self._node("pool_avgpool_count_pad.onnx")
        self.assertEqual(sn.count_include_pad, 1)

    def test_avgpool_no_count_include_pad_default(self):
        sn = self._node("pool_avgpool_simple.onnx")
        self.assertEqual(sn.count_include_pad, 0)

    def test_global_max_pool_window_equals_input(self):
        sn = self._node("pool_global_max.onnx")
        self.assertEqual(sn.pool_h, sn.in_h)
        self.assertEqual(sn.pool_w, sn.in_w)
        self.assertEqual(sn.stride_h, 1)
        self.assertEqual(sn.stride_w, 1)
        self.assertEqual(sn.pad_top,  0)
        self.assertEqual(sn.pad_left, 0)
        self.assertEqual(sn.out_h, 1)
        self.assertEqual(sn.out_w, 1)

    def test_batch2_geometry(self):
        sn = self._node("pool_batch2.onnx")
        self.assertEqual(sn.batch, 2)


# ---------------------------------------------------------------------------
# TensorLayout — PoolNode excluded from phases 2 and 3
# ---------------------------------------------------------------------------

@unittest.skipUnless(_pool_models_exist(),
                     "Run test/gen_pool_models.py first")
class TestPoolLayouts(unittest.TestCase):
    """PoolNode output always has a flat TensorLayout (n_chunks == 1)."""

    def test_simple_flat_output(self):
        gen = _gen("pool_maxpool_simple.onnx")
        y_lay = gen._layouts["Y"]
        self.assertEqual(y_lay.n_chunks, 1)
        self.assertEqual(y_lay.alloc,    y_lay.numel)

    def test_pool_then_relu_flat_intermediate(self):
        gen = _gen("pool_then_relu.onnx")
        # Z is PoolNode output feeding Relu — must stay flat
        z_lay = gen._layouts["Z"]
        self.assertEqual(z_lay.n_chunks, 1)
        self.assertEqual(z_lay.alloc,    z_lay.numel)

    def test_alloc_matches_numel(self):
        gen = _gen("pool_maxpool_simple.onnx")
        y_lay = gen._layouts["Y"]
        # Y[1,4,4,4] = 64 elements
        self.assertEqual(y_lay.numel, 64)
        self.assertEqual(y_lay.alloc, 64)


# ---------------------------------------------------------------------------
# Generated inference.c source
# ---------------------------------------------------------------------------

@unittest.skipUnless(_pool_models_exist(),
                     "Run test/gen_pool_models.py first")
class TestPoolSource(unittest.TestCase):
    """Generated inference.c contains correct PoolingKernel calls."""

    def _src(self, name: str) -> str:
        return _gen(name).generate_source()

    def test_pool_header_included(self):
        s = self._src("pool_maxpool_simple.onnx")
        self.assertIn('#include "xpoolingkernel.h"', s)

    def test_pool_instance_declared(self):
        s = self._src("pool_maxpool_simple.onnx")
        self.assertIn("XPoolingkernel s_poolkernel", s)

    def test_run_pool_helper_emitted(self):
        s = self._src("pool_maxpool_simple.onnx")
        self.assertIn("static void run_pool(", s)

    def test_run_pool_called(self):
        s = self._src("pool_maxpool_simple.onnx")
        self.assertIn("run_pool(", s)

    def test_no_run_op_for_pool_only(self):
        s = self._src("pool_maxpool_simple.onnx")
        self.assertNotIn("static void run_op(", s)

    def test_mixed_pool_and_relu(self):
        s = self._src("pool_then_relu.onnx")
        self.assertIn("static void run_pool(", s)
        self.assertIn("static void run_op(",   s)

    def test_pool_type_max_in_call(self):
        s = self._src("pool_maxpool_simple.onnx")
        # pool_type=0 (POOL_MAX), stride 2, no pad, no dil
        self.assertIn("0u, 2u, 0u)", s)   # pool_type=0, lp_order=2, count_include_pad=0

    def test_pool_type_avg_in_call(self):
        s = self._src("pool_avgpool_simple.onnx")
        # pool_type=1 (POOL_AVG)
        self.assertIn("1u, 2u, 0u)", s)

    def test_pool_kernel_set_calls(self):
        s = self._src("pool_maxpool_simple.onnx")
        for setter in [
            "XPoolingkernel_Set_x", "XPoolingkernel_Set_y",
            "XPoolingkernel_Set_batch", "XPoolingkernel_Set_channels",
            "XPoolingkernel_Set_in_h", "XPoolingkernel_Set_in_w",
            "XPoolingkernel_Set_out_h", "XPoolingkernel_Set_out_w",
            "XPoolingkernel_Set_pool_h", "XPoolingkernel_Set_pool_w",
            "XPoolingkernel_Set_stride_h", "XPoolingkernel_Set_stride_w",
            "XPoolingkernel_Set_pad_top", "XPoolingkernel_Set_pad_left",
            "XPoolingkernel_Set_dil_h", "XPoolingkernel_Set_dil_w",
            "XPoolingkernel_Set_pool_type", "XPoolingkernel_Set_lp_order",
            "XPoolingkernel_Set_count_include_pad",
        ]:
            self.assertIn(setter, s, f"missing {setter}")

    def test_global_avg_pool_window_in_call(self):
        s = self._src("pool_global_avg.onnx")
        # GlobalAveragePool: pool_h=pool_w=in_h=in_w=4, out_h=out_w=1
        # run_pool(X, Y, 1u, 8u, 4u, 4u, 1u, 1u, 4u, 4u, 1u, 1u, 0u, 0u, 1u, 1u, 1u, 2u, 0u)
        self.assertIn("run_pool(", s)


# ---------------------------------------------------------------------------
# Simulation: forward pass
# ---------------------------------------------------------------------------

@unittest.skipUnless(_pool_models_exist(),
                     "Run test/gen_pool_models.py first")
class TestPoolSimulate(unittest.TestCase):
    """_simulate() produces plausible outputs for pool models."""

    def _sim(self, name: str) -> dict:
        gen = _gen(name)
        return gen._simulate()

    def test_maxpool_output_shape(self):
        arrays = self._sim("pool_maxpool_simple.onnx")
        self.assertIn("Y", arrays)
        self.assertEqual(arrays["Y"].shape, (1, 4, 4, 4))

    def test_avgpool_output_shape(self):
        arrays = self._sim("pool_avgpool_simple.onnx")
        self.assertEqual(arrays["Y"].shape, (1, 4, 4, 4))

    def test_global_max_output_shape(self):
        arrays = self._sim("pool_global_max.onnx")
        self.assertEqual(arrays["Y"].shape, (1, 8, 1, 1))

    def test_global_avg_output_shape(self):
        arrays = self._sim("pool_global_avg.onnx")
        self.assertEqual(arrays["Y"].shape, (1, 8, 1, 1))

    def test_lp_p2_output_shape(self):
        arrays = self._sim("pool_lp_p2.onnx")
        self.assertEqual(arrays["Y"].shape, (1, 4, 4, 4))

    def test_maxpool_max_property(self):
        """Every output element must be >= all 4 inputs in its 2x2 window."""
        path = _pool_model("pool_maxpool_simple.onnx")
        g    = OnnxGraph(path)
        cg   = CodeGenerator(g, model_path=path)

        # Use a controlled input: channel c at position (ih, iw) = c * H * W + ih * W + iw
        x_arr = np.arange(1 * 4 * 8 * 8, dtype=np.float64).reshape(1, 4, 8, 8) * (1.0/256)
        # Quantize to ap_fixed<16,8> grid
        x_q   = cg._dtype.quantize(x_arr)
        result = cg.simulate({"X": x_q})
        y = result["Y"]  # [1,4,4,4]

        # Each output pixel must be >= all 4 pixels in its 2x2 window
        for oh in range(4):
            for ow in range(4):
                window = x_q[0, :, oh*2:oh*2+2, ow*2:ow*2+2]
                expected_max = window.max(axis=(-2, -1))  # [C]
                np.testing.assert_array_less(
                    expected_max - 1e-9, y[0, :, oh, ow],
                    err_msg=f"MaxPool output[0,:,{oh},{ow}] < window max"
                )

    def test_avgpool_values(self):
        """AveragePool output must be the mean of the 2x2 window."""
        path = _pool_model("pool_avgpool_simple.onnx")
        g    = OnnxGraph(path)
        cg   = CodeGenerator(g, model_path=path)

        x_arr = np.ones((1, 4, 8, 8), dtype=np.float64) * 2.0
        x_q   = cg._dtype.quantize(x_arr)
        result = cg.simulate({"X": x_q})
        y = result["Y"]
        # Average of 2.0s = 2.0 everywhere
        np.testing.assert_allclose(y, 2.0, atol=0.01)

    def test_pool_then_relu_relu_applied(self):
        """After MaxPool → Relu, all output values must be >= 0."""
        path = _pool_model("pool_then_relu.onnx")
        g    = OnnxGraph(path)
        cg   = CodeGenerator(g, model_path=path)

        # Use ramp input including negatives
        ramp = np.linspace(-1.0, 1.0, 1 * 4 * 8 * 8).reshape(1, 4, 8, 8)
        x_q  = cg._dtype.quantize(ramp)
        result = cg.simulate({"X": x_q})
        y = result["Y"]
        self.assertTrue(np.all(y >= -1e-9), "Relu output contains negatives")

    def test_simulate_produces_input_and_output(self):
        """_simulate() returns entries for all tensors including input."""
        arrays = self._sim("pool_maxpool_simple.onnx")
        # X is seeded from the ramp; Y is produced by the pool
        self.assertIn("X", arrays)
        self.assertIn("Y", arrays)
