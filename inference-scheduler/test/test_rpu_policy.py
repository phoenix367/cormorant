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
    """The rule on the measured board numbers (2026-10-13): the overhead model
    and the 5 % limit put the MNIST models and MobileNet v2 on the A53 and
    ResNet-18, MobileNet v1 and YOLOv5n on the R5."""

    CASES = [   # (calls, batches, predicted / measured A53 µs, R5?)
        ("mnist_convnet", 8, 1, 111.0, False),
        ("mnist_lenet", 10, 1, 1259.0, False),
        ("resnet18", 41, 1, 20004.0, True),
        ("mobilenet_v1", 57, 1, 20472.0, True),
        ("mobilenet_v2", 100, 1, 17475.0, False),
        ("yolov5n", 127, 14, 46998.0, True),
    ]

    def test_measured_models(self):
        for name, calls, batches, us, r5 in self.CASES:
            over = calls * rpu_policy.CALL_US + batches * rpu_policy.BATCH_US
            self.assertEqual(over <= rpu_policy.MAX_OVERHEAD * us, r5, name)

    def test_overhead_fit(self):
        measured = {"mnist_convnet": 138, "mnist_lenet": 159, "resnet18": 410,
                    "mobilenet_v1": 530, "mobilenet_v2": 900, "yolov5n": 1910}
        for name, calls, batches, _us, _r5 in self.CASES:
            over = calls * rpu_policy.CALL_US + batches * rpu_policy.BATCH_US
            self.assertLess(abs(over - measured[name]) / measured[name], 0.10, name)


@unittest.skipUnless(_models_exist(), "Run test/gen_test_models.py first")
class TestEstimate(unittest.TestCase):

    def test_small_model_stays_on_the_a53(self):
        est = rpu_policy.rpu_estimate(_cg("mixed_matmul_relu.onnx"))
        self.assertEqual((est["calls"], est["batches"]), (2, 1))
        self.assertAlmostEqual(est["overhead_us"], 2 * rpu_policy.CALL_US + rpu_policy.BATCH_US)
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
