# Copyright 2026 Ivan Gubochkin
# SPDX-License-Identifier: Apache-2.0

"""The Cortex-R5 policy (src/rpu_policy.py, RPU_PLAN §9.3): small models stay on
the A53, large ones go to the R5 — the generated header's numbers, the runtime
hooks of every entry, and the decision itself with a fixed prediction."""

import unittest
from unittest import mock

from helpers import _model, _models_exist
from src import rpu_policy
from src.codegen import CodeGenerator
from src.graph import OnnxGraph


def _cg(name):
    g = OnnxGraph(_model(name))
    return CodeGenerator(g, model_path=_model(name))


class TestDecision(unittest.TestCase):
    """The per-call / per-batch fit on the protocol v4 board numbers (2026-10-13,
    RPU_PLAN §9.4; every run back to back, the R5 never asleep)."""

    CASES = [   # (calls, batches, A53 µs, R5 - A53 µs measured)
        ("chain1", 1, 1, 9.0, 3.0),
        ("chain16", 16, 1, 33.0, 21.0),
        ("chain128", 128, 1, 220.0, 146.0),
        ("mnist_convnet", 8, 1, 112.0, 17.0),
        ("mnist_lenet", 10, 1, 1283.0, 11.0),
        ("resnet18", 41, 1, 19982.0, 36.0),
        ("mobilenet_v1", 57, 1, 20461.0, 47.0),
        ("mobilenet_v2", 100, 1, 17462.0, 111.0),
        ("yolov5n", 127, 14, 46955.0, 276.0),
    ]

    def test_overhead_fit(self):
        for name, calls, batches, _us, extra in self.CASES:
            over = calls * rpu_policy.CALL_US + batches * rpu_policy.BATCH_US
            self.assertLess(abs(over - extra), 25.0, name)


class TestDecisionV5(unittest.TestCase):
    """The rule on the protocol v5 board numbers (2026-10-13 evening, RPU_PLAN
    §11.2; rpu_board.py --mode r5, the A53 reference run between the R5 runs, so
    the R5 sleeps before each): the per-inference wake costs keep the
    sub-millisecond MNIST convnet and LeNet on the A53 and put the large models on
    the R5."""

    CASES = [   # (calls, batches, R5 wakes per run, A53 µs, R5 - A53 µs measured, R5?)
        ("mnist_convnet", 8, 1, 0, 114.0, 42.0, False),       # runs back to back: no R5 wake
        ("mnist_lenet", 10, 1, 1, 1258.0, 82.0, False),
        ("resnet18", 41, 1, 1, 19986.0, 112.0, True),
        ("mobilenet_v1", 57, 1, 1, 20464.0, 151.0, True),
        ("mobilenet_v2", 100, 1, 1, 17470.0, 245.0, True),
        ("yolov5n", 127, 14, 3.4, 47292.0, 825.0, True),
    ]

    @staticmethod
    def _model(calls, batches, wakes):
        """The policy's overhead for that many R5 wakes (0: the A53's share of
        RUN_US only)."""
        return (calls * rpu_policy.CALL_US + batches * rpu_policy.BATCH_US + rpu_policy.RUN_US
                + (wakes - 1) * rpu_policy.WAKE_US)

    def test_measured_models(self):
        for name, calls, batches, _w, us, extra, r5 in self.CASES:
            over = self._model(calls, batches, 1)                  # the policy assumes one wake
            self.assertEqual(over <= rpu_policy.MAX_OVERHEAD * us, r5, name)
            self.assertEqual(extra <= rpu_policy.MAX_OVERHEAD * us, r5, name)

    def test_small_models_fit(self):
        """Within 25 µs where the calls are few; the larger models' calls cost
        more than CALL_US (the model's share only grows: the decision holds)."""
        for name, calls, batches, wakes, _us, extra, _r5 in self.CASES:
            over = self._model(calls, batches, wakes)
            if calls <= 41:
                self.assertLess(abs(over - extra), 25.0, name)
            else:
                self.assertLess(over, extra, name)

    def test_r5_wakes_from_the_replay(self):
        """One wake at the start, one after each host interval longer than
        R5_IDLE_US that kernel work follows (back-to-back host ops merged)."""
        long_gap = rpu_policy.R5_IDLE_US + 100.0
        spans = [("Conv", 1, 0.0, 50.0, "kernel"),
                 ("CPU", 2, 60.0, 60.0 + long_gap / 2, "host"),
                 ("CPU", 3, 60.0 + long_gap / 2, 60.0 + long_gap, "host"),    # merged with the one before
                 ("Conv", 4, 70.0 + long_gap, 120.0 + long_gap, "kernel"),
                 ("CPU", 5, 130.0 + long_gap, 140.0 + long_gap, "host"),     # short
                 ("Conv", 6, 150.0 + long_gap, 160.0 + long_gap, "kernel"),
                 ("CPU", 7, 170.0 + long_gap, 170.0 + 2 * long_gap, "host")]  # long, no kernel after it
        tl = mock.Mock(spans=spans, total_us=170.0 + 2 * long_gap)
        self.assertEqual(rpu_policy._r5_wakes(tl), 2)


@unittest.skipUnless(_models_exist(), "Run test/gen_test_models.py first")
class TestEstimate(unittest.TestCase):

    def test_small_model_stays_on_the_a53(self):
        est = rpu_policy.rpu_estimate(_cg("mixed_matmul_relu.onnx"))
        self.assertEqual((est["calls"], est["batches"]), (2, 1))
        self.assertEqual(est["r5_wakes"], 1)
        self.assertAlmostEqual(est["overhead_us"], 2 * rpu_policy.CALL_US + rpu_policy.BATCH_US + rpu_policy.RUN_US)
        self.assertFalse(est["recommended"])

    def test_large_prediction_goes_to_the_r5(self):
        cg = _cg("mixed_matmul_relu.onnx")
        with mock.patch.object(rpu_policy, "_default_perf_model", return_value=None), \
             mock.patch.object(rpu_policy, "estimate_us", return_value=10000.0):
            est = rpu_policy.rpu_estimate(cg)
        self.assertEqual(est["basis"], "analytic cost model")
        self.assertGreater(est["predicted_us"], 10000.0)
        self.assertTrue(est["recommended"])

    def test_header_macros(self):
        h = _cg("single_add.onnx").generate_header()
        for m in ("INFERENCE_RPU_CALLS ", "INFERENCE_RPU_BATCHES ", "INFERENCE_RPU_PREDICTED_US ",
                  "INFERENCE_RPU_OVERHEAD_US ", "INFERENCE_RPU_RECOMMENDED "):
            self.assertIn("#define " + m, h)
        self.assertIn("#define INFERENCE_RPU_CALLS         1u", h)
        self.assertIn("#define INFERENCE_RPU_RECOMMENDED   0   /* the A53 drives the kernels", h)

    def test_runtime_hooks(self):
        s = _cg("single_add.onnx").generate_source()
        init = s[s.index("int inference_init("):s.index("void inference_deinit(void)")]
        self.assertIn("inference_rpu_select(0u);", init)
        deinit = s[s.index("void inference_deinit(void)"):]
        self.assertLess(deinit.index("inference_rpu_detach();"), deinit.index("inference_buf_pool_deinit();"))
        run = s[s.index("void inference_run("):]
        self.assertIn("INFERENCE_RPU_ENTER(INFERENCE_RPU_RECOMMENDED);", run)

    def test_no_kernel_no_policy(self):
        est = rpu_policy.rpu_estimate(_cg("reshape_flatten.onnx"))
        self.assertEqual(est["calls"], 0)
        self.assertFalse(est["recommended"])


@unittest.skipUnless(_models_exist(), "Run test/gen_test_models.py first")
class TestMultiEntry(unittest.TestCase):

    def test_per_entry_recommendation(self):
        import os
        if not os.path.exists(_model("llama_tiny_decode.onnx")):
            self.skipTest("run test/gen_llama_models.py")
        from src.codegen.multi import MultiEntryGenerator
        entries = [(n, OnnxGraph(_model(f"llama_tiny_{n}.onnx"))) for n in ("decode", "head")]
        mg = MultiEntryGenerator(entries, "llama_tiny")
        h, s = mg.generate_header(), mg.generate_source()
        for n in ("DECODE", "HEAD"):
            self.assertIn(f"#define INFERENCE_RPU_RECOMMENDED_{n}", h)
            self.assertIn(f"INFERENCE_RPU_ENTER(INFERENCE_RPU_RECOMMENDED_{n});", s)

    def test_written_project_carries_every_cmake_source(self):
        """Both project writers copy the runtime files (src/codegen/runtime_files.py):
        every src/ file the generated CMakeLists.txt names is there — the chat
        libraries' build broke on src/inference_rpu.c once (2026-10-13)."""
        import os
        import re
        import tempfile
        if not os.path.exists(_model("llama_tiny_decode.onnx")):
            self.skipTest("run test/gen_llama_models.py")
        from src.codegen.multi import MultiEntryGenerator
        entries = [(n, OnnxGraph(_model(f"llama_tiny_{n}.onnx"))) for n in ("decode", "head")]
        mg = MultiEntryGenerator(entries, "llama_tiny")
        with tempfile.TemporaryDirectory() as td:
            mg.write_project(td, weights=False)
            cm = open(os.path.join(td, "CMakeLists.txt")).read()
            for rel in sorted(set(re.findall(r"\b(src/[A-Za-z0-9_/]+\.c)\b", cm))):
                self.assertTrue(os.path.isfile(os.path.join(td, rel)), rel)
            for rel in ("include/inference_rpu.h", "include/rpu_ctrl.h", "rpu/rpu_host.c", "include/inference_irq.h"):
                self.assertTrue(os.path.isfile(os.path.join(td, rel)), rel)
