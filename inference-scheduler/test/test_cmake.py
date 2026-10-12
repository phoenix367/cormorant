"""Tests for CMakeLists.txt generation (TestCMakeGen)."""

import unittest

from helpers import _model, _models_exist
from src.graph   import OnnxGraph
from src.codegen import CodeGenerator


@unittest.skipUnless(_models_exist(), "Run test/gen_test_models.py first")
class TestCMakeGen(unittest.TestCase):

    def _cmake(self, model_name: str) -> str:
        g  = OnnxGraph(_model(model_name))
        cg = CodeGenerator(g, model_path=_model(model_name))
        return cg.generate_cmake()

    def test_add_library_static(self):
        cm = self._cmake("single_add.onnx")
        self.assertIn("add_library(inference STATIC", cm)

    def test_inference_target_option(self):
        cm = self._cmake("single_add.onnx")
        self.assertIn("INFERENCE_TARGET", cm)
        self.assertIn("BARE_METAL", cm)
        self.assertIn("LINUX", cm)

    def test_bare_metal_driver_source(self):
        # bare metal (the R5 firmware) initialises the kernels from their base
        # addresses: no _sinit.c (device-id lookup), no _linux.c
        cm = self._cmake("single_add.onnx")
        self.assertIn("driver/xvectoropkernel.c", cm)
        self.assertNotIn("xvectoropkernel_sinit.c", cm.split("# External weight files")[0].split("LINUX")[0])

    def test_linux_driver_source(self):
        cm = self._cmake("single_add.onnx")
        self.assertIn("xvectoropkernel_linux.c", cm)

    def test_inference_source_listed(self):
        cm = self._cmake("single_add.onnx")
        self.assertIn("src/inference.c", cm)

    def test_buf_impl_source_listed(self):
        cm = self._cmake("single_add.onnx")
        self.assertIn("src/inference_buf.c", cm)

    def test_test_executable_in_cmake(self):
        cm = self._cmake("single_add.onnx")
        self.assertIn("test_inference", cm)
        self.assertIn("INFERENCE_BUILD_TEST", cm)
        self.assertIn("test/test_inference.c", cm)

    def test_include_dirs(self):
        cm = self._cmake("single_add.onnx")
        self.assertIn("include", cm)
        self.assertIn("driver", cm)
        self.assertIn("BSP_INCLUDE_DIR", cm)

    def test_install_targets(self):
        cm = self._cmake("single_add.onnx")
        self.assertIn("install(TARGETS inference", cm)
        self.assertIn("install(FILES include/inference.h", cm)

    def test_driver_existence_check(self):
        cm = self._cmake("single_add.onnx")
        self.assertIn("FATAL_ERROR", cm)

    def test_xrt_dir_for_linux(self):
        cm = self._cmake("single_add.onnx")
        # pkg-config primary path
        self.assertIn("pkg_check_modules", cm)
        self.assertIn("PkgConfig", cm)
        # manual fallback still present
        self.assertIn("XRT_DIR", cm)
        self.assertIn("libxrt_core.so", cm)


if __name__ == "__main__":
    unittest.main(verbosity=2)
