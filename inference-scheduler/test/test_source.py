"""Tests for inference.c source file generation (TestSourceGen)."""

import unittest

from helpers import _model, _models_exist
from src.graph   import OnnxGraph
from src.codegen import CodeGenerator


@unittest.skipUnless(_models_exist(), "Run test/gen_test_models.py first")
class TestSourceGen(unittest.TestCase):

    def _source(self, model_name: str) -> str:
        g  = OnnxGraph(_model(model_name))
        cg = CodeGenerator(g, model_path=_model(model_name))
        return cg.generate_source()

    def test_includes_own_header(self):
        s = self._source("single_add.onnx")
        self.assertIn('#include "inference.h"', s)

    def test_includes_driver(self):
        s = self._source("single_add.onnx")
        self.assertIn("xvectoropkernel.h", s)

    def test_no_data_t_typedef_in_source(self):
        # Data_t is defined in inference.h; source must not redefine it
        s = self._source("single_add.onnx")
        self.assertNotIn("typedef uint16_t Data_t", s)

    def test_bytes_per_elem_not_redefined(self):
        s = self._source("single_add.onnx")
        # BYTES_PER_ELEM in run_op body references the header's INFERENCE_BYTES_PER_ELEM
        self.assertNotIn("#define BYTES_PER_ELEM ", s)

    def test_op_defines_in_source(self):
        s = self._source("single_add.onnx")
        self.assertIn("#define VECTOROP_ADD", s)

    def test_weight_array_present(self):
        s = self._source("single_add.onnx")
        self.assertIn("static const uint16_t", s)

    def test_run_op_helper(self):
        s = self._source("single_add.onnx")
        self.assertIn("static void run_op(", s)

    def test_sync_forward_decls_in_source(self):
        s = self._source("single_add.onnx")
        # inference_buf.c sync API must be forward-declared before run_op()
        self.assertIn("inference_buf_sync_to_device", s)
        self.assertIn("inference_buf_sync_from_device", s)

    def test_run_op_uses_sync_functions(self):
        s = self._source("single_add.onnx")
        # run_op() must NOT sync anything — all sync is the caller's responsibility
        self.assertNotIn("inference_buf_sync_to_device(a)", s)
        self.assertNotIn("inference_buf_sync_from_device(c)", s)
        # inference_run() flushes graph inputs and invalidates graph outputs
        self.assertIn("inference_buf_sync_to_device(X)", s)
        self.assertIn("inference_buf_sync_from_device(Y)", s)
        # inference_init() syncs the entire weight pool once after loading —
        # the single pool covers all weight and intermediate buffers.
        self.assertIn("inference_buf_sync_to_device(s_alloc_pool)", s)
        self.assertNotIn("inference_buf_sync_to_device(bias)", s)

    def test_run_op_call_add(self):
        s = self._source("single_add.onnx")
        self.assertIn("VECTOROP_ADD", s)
        self.assertIn("run_op(", s)

    def test_relu_null_b(self):
        s = self._source("relu_chain.onnx")
        self.assertIn("VECTOROP_RELU", s)
        self.assertIn("NULL", s)

    def test_mixed_ops_relu6(self):
        s = self._source("mixed_ops.onnx")
        self.assertIn("VECTOROP_RELU6", s)

    def test_inference_run_body(self):
        s = self._source("single_add.onnx")
        self.assertIn("void inference_run(", s)

    def test_inference_init_body(self):
        s = self._source("single_add.onnx")
        self.assertIn("int inference_init(", s)
        self.assertIn("XVectoropkernel_Initialize", s)

    def test_weight_rom_prefix(self):
        # ROM arrays must use _rom_ prefix; buf pointer is separate
        s = self._source("single_add.onnx")
        self.assertIn("_rom_", s)

    def test_inference_buf_alloc_in_source(self):
        s = self._source("single_add.onnx")
        self.assertIn("inference_buf_alloc(", s)

    def test_inference_buf_phys_in_source(self):
        s = self._source("single_add.onnx")
        self.assertIn("inference_buf_phys(", s)

    def test_memcpy_in_source(self):
        # Weight ROM data must be copied into DMA buffers at init
        s = self._source("single_add.onnx")
        self.assertIn("memcpy(", s)

    def test_inference_deinit_in_source(self):
        s = self._source("single_add.onnx")
        self.assertIn("inference_deinit(", s)

    def test_pool_init_in_source(self):
        s = self._source("single_add.onnx")
        self.assertIn("inference_buf_pool_init(", s)

    def test_run_op_uses_physical_address(self):
        # run_op must use inference_buf_phys not a raw pointer cast
        s = self._source("single_add.onnx")
        self.assertIn("inference_buf_phys(a)", s)
        self.assertNotIn("(u64)(UINTPTR)", s)

    def test_source_includes_string_h(self):
        # memcpy requires <string.h>
        s = self._source("single_add.onnx")
        self.assertIn("<string.h>", s)


if __name__ == "__main__":
    unittest.main(verbosity=2)


@unittest.skipUnless(_models_exist(), "Run test/gen_test_models.py first")
class TestRpuRecorderHooks(unittest.TestCase):
    """The hooks that let the Cortex-R5 drive the kernels (inference_rpu.h,
    RPU_PLAN B1): every Start goes through the recorder while attached, the
    default kernel_wait records, the recorder's kernel table names each
    kernel's lane and register count, and a batch is flushed before every
    host op and before the output sync."""

    def _source(self, model_name: str) -> str:
        g  = OnnxGraph(_model(model_name))
        cg = CodeGenerator(g, model_path=_model(model_name))
        return cg.generate_source()

    def test_start_intercept_and_table(self):
        s = self._source("mixed_matmul_relu.onnx")
        self.assertIn("#define XVectoropkernel_Start(inst)  (inference_rpu_active() ? "
                      "inference_rpu_start(KERNEL_VECTOROP) : "
                      "(inference_irq_start(KERNEL_VECTOROP), (XVectoropkernel_Start)(inst)))", s)
        self.assertIn("#define XMatmulkernel_Start(inst)  (inference_rpu_active() ? "
                      "inference_rpu_start(KERNEL_MATMUL) : "
                      "(inference_irq_start(KERNEL_MATMUL), (XMatmulkernel_Start)(inst)))", s)
        self.assertIn("{ 0u, 28u, &s_vectoropkernel.Ctrl_BaseAddress },   /* KERNEL_VECTOROP */", s)
        self.assertIn("{ 1u, 29u, &s_matmulkernel.Ctrl_BaseAddress },   /* KERNEL_MATMUL */", s)
        self.assertIn("if (inference_rpu_active()) { inference_rpu_wait(k); return; }", s)
        self.assertIn("#define INFERENCE_RPU_FLUSH()  ((void)0)", s)

    def test_irq_waits(self):
        """While the A53 drives, a Start arms the kernel's interrupt and
        kernel_wait() sleeps on it (inference_irq.h): after the R5's recorded
        wait, before the IsDone() polling; inference_init() opens each kernel's
        UIO interrupt after its driver, inference_deinit() closes them."""
        s = self._source("mixed_matmul_relu.onnx")
        wait = s[s.index("void kernel_wait(kernel_id_t k)"):]
        self.assertLess(wait.index("inference_rpu_wait(k)"), wait.index("if (inference_irq_wait(k)) return;"))
        self.assertLess(wait.index("if (inference_irq_wait(k)) return;"), wait.index("_IsDone("))
        init = s[s.index("int inference_init("):s.index("void inference_deinit(void)")]
        for kid, inst, var in (("KERNEL_VECTOROP", "vectoropkernel_instance", "s_vectoropkernel"),
                               ("KERNEL_MATMUL", "matmulkernel_instance", "s_matmulkernel")):
            self.assertIn(f"(void)inference_irq_open({kid}, {inst},\n"
                          f"                             (void *)(uintptr_t){var}.Ctrl_BaseAddress);", init)
        self.assertLess(init.index("XMatmulkernel_Initialize("), init.index("inference_irq_open(KERNEL_MATMUL"))
        deinit = s[s.index("void inference_deinit(void)"):]
        self.assertIn("inference_irq_close();", deinit[:deinit.index("}")])
        self.assertIn('#include "inference_irq.h"', s)

    def test_flush_before_output_sync(self):
        s = self._source("single_add.onnx")
        body = s[s.index("void inference_run("):]
        self.assertIn("INFERENCE_RPU_FLUSH();", body)
        self.assertLess(body.index("INFERENCE_RPU_FLUSH();"), body.index("Invalidate graph outputs"))
