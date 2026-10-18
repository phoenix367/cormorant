# Copyright 2025 GradeBuilder SL
# SPDX-License-Identifier: Apache-2.0

"""Tests for test/test_inference.c generation (TestTestGen)."""

import unittest

from helpers import _model, _models_exist
from src.graph   import OnnxGraph
from src.codegen import CodeGenerator


@unittest.skipUnless(_models_exist(), "Run test/gen_test_models.py first")
class TestTestGen(unittest.TestCase):

    def _test_src(self, model_name: str) -> str:
        g  = OnnxGraph(_model(model_name))
        cg = CodeGenerator(g, model_path=_model(model_name))
        return cg.generate_test()

    def test_includes_stdio(self):
        t = self._test_src("single_add.onnx")
        self.assertIn("#include <stdio.h>", t)

    def test_includes_inference_h(self):
        t = self._test_src("single_add.onnx")
        self.assertIn('#include "inference.h"', t)

    def test_main_signature(self):
        t = self._test_src("single_add.onnx")
        self.assertIn("int main(int argc, char **argv)", t)

    def test_inference_init_called(self):
        t = self._test_src("single_add.onnx")
        self.assertIn("inference_init(", t)

    def test_inference_run_called(self):
        t = self._test_src("single_add.onnx")
        self.assertIn("inference_run(", t)

    def test_inference_deinit_called(self):
        t = self._test_src("single_add.onnx")
        self.assertIn("inference_deinit(", t)

    def test_input_buf_alloc(self):
        t = self._test_src("single_add.onnx")
        self.assertIn("inference_buf_alloc(INFERENCE_X_SIZE)", t)

    def test_output_buf_alloc(self):
        t = self._test_src("single_add.onnx")
        self.assertIn("inference_buf_alloc(INFERENCE_Y_SIZE)", t)

    def test_ramp_fill_pattern(self):
        t = self._test_src("single_add.onnx")
        self.assertIn("i & 0xFFFF", t)

    def test_printf_output(self):
        t = self._test_src("single_add.onnx")
        self.assertIn("printf(", t)

    def test_linux_default_instance(self):
        # The UIO sysfs name is the DT node label from the DTBO overlay.
        # XVectoropkernel_Initialize() scans /sys/class/uio/*/name for the match.
        t = self._test_src("single_add.onnx")
        self.assertIn("VectorOPKernel_0", t)
        self.assertNotIn("/dev/uio0", t)

    def test_baremetal_default_instance(self):
        t = self._test_src("single_add.onnx")
        self.assertIn("VectorOPKernel_0", t)

    def test_passed_message(self):
        t = self._test_src("single_add.onnx")
        self.assertIn("test_inference PASSED", t)

    def test_multi_op_allocs(self):
        # relu_chain: input X, output relu_Y — both must be allocated
        t = self._test_src("relu_chain.onnx")
        self.assertIn("inference_buf_alloc(INFERENCE_X_SIZE)", t)
        # output is relu_Y → macro is INFERENCE_RELU_Y_SIZE
        self.assertIn("inference_buf_alloc(INFERENCE_RELU_Y_SIZE)", t)
        self.assertIn("inference_run(", t)


if __name__ == "__main__":
    unittest.main(verbosity=2)
