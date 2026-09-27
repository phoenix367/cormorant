"""Tests for inference_buf.c buffer implementation generation (TestBufImpl)."""

import os
import re
import shutil
import subprocess
import tempfile
import unittest

from helpers import _model, _models_exist
from src.graph   import OnnxGraph
from src.codegen import CodeGenerator
from src.dtype   import AP_FIXED_16_8, FLOAT32


@unittest.skipUnless(_models_exist(), "Run test/gen_test_models.py first")
class TestBufImpl(unittest.TestCase):

    def _buf(self, model_name: str) -> str:
        g  = OnnxGraph(_model(model_name))
        cg = CodeGenerator(g, model_path=_model(model_name))
        return cg.generate_buf_impl()

    def test_struct_definition(self):
        """struct inference_buf is defined in inference.h (not inference_buf.c)
        so that inference.c can declare static view instances."""
        g  = OnnxGraph(_model("single_add.onnx"))
        cg = CodeGenerator(g, model_path=_model("single_add.onnx"))
        h = cg.generate_header()
        b = cg.generate_buf_impl()
        self.assertIn("struct inference_buf", h)
        self.assertNotIn("struct inference_buf {", b)

    def test_linux_xrt_alloc(self):
        b = self._buf("single_add.onnx")
        self.assertIn("xclAllocBO(", b)

    def test_linux_xrt_map(self):
        b = self._buf("single_add.onnx")
        self.assertIn("xclMapBO(", b)

    def test_linux_xrt_sync(self):
        b = self._buf("single_add.onnx")
        self.assertIn("xclSyncBO(", b)
        self.assertIn("XCL_BO_SYNC_BO_TO_DEVICE", b)
        self.assertIn("XCL_BO_SYNC_BO_FROM_DEVICE", b)

    def test_linux_xrt_paddr(self):
        b = self._buf("single_add.onnx")
        self.assertIn("xclGetBOProperties(", b)
        self.assertIn("props.paddr", b)

    def test_alloc_function(self):
        b = self._buf("single_add.onnx")
        self.assertIn("inference_buf_alloc(", b)

    def test_phys_equals_virt_on_baremetal(self):
        # Bare-metal: physical address = cast of virtual pointer
        b = self._buf("single_add.onnx")
        self.assertIn("(uint64_t)(uintptr_t)", b)

    def test_pool_init_function(self):
        b = self._buf("single_add.onnx")
        self.assertIn("inference_buf_pool_init(", b)

    def test_pool_deinit_function(self):
        b = self._buf("single_add.onnx")
        self.assertIn("inference_buf_pool_deinit(", b)

    def test_accessors_present(self):
        b = self._buf("single_add.onnx")
        self.assertIn("inference_buf_ptr(", b)
        self.assertIn("inference_buf_phys(", b)
        self.assertIn("inference_buf_count(", b)

    def test_sync_functions_present(self):
        b = self._buf("single_add.onnx")
        self.assertIn("inference_buf_sync_to_device(", b)
        self.assertIn("inference_buf_sync_from_device(", b)

    def test_bare_metal_cache_api_in_buf(self):
        b = self._buf("single_add.onnx")
        self.assertIn("Xil_DCacheFlushRange", b)
        self.assertIn("Xil_DCacheInvalidateRange", b)

    def test_float_cast_fill_present(self):
        b = self._buf("single_add.onnx")
        self.assertIn("inference_buf_fill_float(", b)

    def test_float_cast_read_present(self):
        b = self._buf("single_add.onnx")
        self.assertIn("inference_buf_read_float(", b)

    def test_fill_float_converts_via_dtype(self):
        # Values are converted to Data_t's number format by the data type's
        # buf_from_float (fixed point: scaled, rounded, saturated).
        b = self._buf("single_add.onnx")
        self.assertIn("dst[i] = buf_from_float(src[i]);", b)
        self.assertIn(AP_FIXED_16_8.c_buf_float_conversions(), b)

    def test_read_float_converts_via_dtype(self):
        b = self._buf("single_add.onnx")
        self.assertIn("dst[i] = buf_to_float(src[i]);", b)

    def test_float_conversions_follow_dtype(self):
        # ap_fixed<16,8>: scale 2^8, round half to even, saturate at int16;
        # float32: a plain cast, no fixed-point constants.
        fx = self._buf("single_add.onnx")
        self.assertIn("(double)v * 256.0", fx)
        self.assertIn("if (r > 32767.0) r = 32767.0;", fx)
        g = OnnxGraph(_model("single_add.onnx"), dtype=FLOAT32)
        fl = CodeGenerator(g, model_path=_model("single_add.onnx"),
                           dtype=FLOAT32).generate_buf_impl()
        self.assertIn("return (Data_t)v;", fl)
        self.assertNotIn("256.0", fl)



class TestSmmuBackend(unittest.TestCase):
    """The /dev/fpga_smmu_mem backend of the Linux inference_buf.c."""

    _DRIVER_H = os.path.join(os.path.dirname(__file__), "..", "..", "board", "kv260",
                             "fpga-smmu-mem", "fpga_smmu_mem.h")

    def _buf(self):
        g = OnnxGraph(_model("single_add.onnx"))
        return CodeGenerator(g, model_path=_model("single_add.onnx")).generate_buf_impl()

    @staticmethod
    def _abi(text):
        """#define FSM_* values and struct fsm_* field lists, types normalised."""
        text = text.replace("__u64", "uint64_t").replace("__u32", "uint32_t")
        defines = dict(re.findall(r"#define\s+(FSM_\w+)\s+(.+?)\s*(?:/\*.*)?$", text, re.M))
        structs = {name: re.findall(r"(uint\d+_t)\s+(\w+);", body)
                   for name, body in re.findall(r"struct (fsm_\w+) \{(.*?)\};", text, re.S)}
        return defines, structs

    @unittest.skipUnless(_models_exist(), "run test/gen_test_models.py first")
    def test_abi_matches_the_driver_header(self):
        with open(self._DRIVER_H) as f:
            want = self._abi(f.read())
        got = self._abi(self._buf())
        self.assertEqual(got, want)
        self.assertEqual(set(want[1]), {"fsm_alloc", "fsm_sync"})
        self.assertLessEqual({"FSM_DEVICE_PATH", "FSM_ALLOC_WC", "FSM_SYNC_TO_DEVICE",
                              "FSM_SYNC_FROM_DEVICE", "FSM_IOC_MAGIC", "FSM_IOC_ALLOC",
                              "FSM_IOC_SYNC"}, set(want[0]))

    @unittest.skipUnless(_models_exist(), "run test/gen_test_models.py first")
    def test_backend_selection(self):
        b = self._buf()
        self.assertIn('getenv("INFERENCE_BUF_BACKEND")', b)
        self.assertIn("access(FSM_DEVICE_PATH, R_OK | W_OK)", b)
        self.assertIn("ioctl(fd, FSM_IOC_ALLOC, &a)", b)
        self.assertIn("ioctl((int)buf->bo, FSM_IOC_SYNC, &s)", b)

    @unittest.skipUnless(_models_exist() and shutil.which("cc"), "needs the models and cc")
    def test_linux_backend_compiles(self):
        """Compile the generated inference_buf.c (both backends) against a stub xrt.h."""
        g = OnnxGraph(_model("single_add.onnx"))
        cg = CodeGenerator(g, model_path=_model("single_add.onnx"))
        stub = (
            "#pragma once\n#include <stddef.h>\n#include <stdbool.h>\n"
            "typedef void *xclDeviceHandle;\ntypedef unsigned int xclBufferHandle;\n"
            "#define NULLBO 0xffffffffu\n"
            "enum xclVerbosityLevel { XCL_QUIET = 0 };\n"
            "enum xclBOSyncDirection { XCL_BO_SYNC_BO_TO_DEVICE = 0, XCL_BO_SYNC_BO_FROM_DEVICE };\n"
            "struct xclBOProperties { unsigned flags; unsigned long long size; unsigned long long paddr; };\n"
            "xclDeviceHandle xclOpen(unsigned, const char *, enum xclVerbosityLevel);\n"
            "void xclClose(xclDeviceHandle);\n"
            "xclBufferHandle xclAllocBO(xclDeviceHandle, size_t, int, unsigned);\n"
            "void *xclMapBO(xclDeviceHandle, xclBufferHandle, bool);\n"
            "int xclUnmapBO(xclDeviceHandle, xclBufferHandle, void *);\n"
            "void xclFreeBO(xclDeviceHandle, xclBufferHandle);\n"
            "int xclSyncBO(xclDeviceHandle, xclBufferHandle, enum xclBOSyncDirection, size_t, size_t);\n"
            "int xclGetBOProperties(xclDeviceHandle, xclBufferHandle, struct xclBOProperties *);\n")
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "xrt.h"), "w") as f:
                f.write(stub)
            with open(os.path.join(td, "inference.h"), "w") as f:
                f.write(cg.generate_header())
            with open(os.path.join(td, "inference_buf.c"), "w") as f:
                f.write(cg.generate_buf_impl())
            r = subprocess.run(["cc", "-std=gnu99", "-Wall", "-Wextra", "-Werror", "-c",
                                "-I", td, "-o", os.path.join(td, "b.o"),
                                os.path.join(td, "inference_buf.c")],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr[-3000:])


if __name__ == "__main__":
    unittest.main(verbosity=2)
