"""Tests for multi-kernel graph support (MatmulKernel + VectorOPKernel)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from helpers import _model, _models_exist
from src.graph   import OnnxGraph
from src.codegen import CodeGenerator
from src.kernels  import KERNEL_REGISTRY


def _mixed_model(name: str) -> str:
    return os.path.join(os.path.dirname(__file__), "models", name)


def _mixed_models_exist() -> bool:
    return os.path.isfile(_mixed_model("mixed_matmul_relu.onnx"))


def _gen(name: str) -> CodeGenerator:
    path = _mixed_model(name)
    g    = OnnxGraph(path)
    return CodeGenerator(g, model_path=path)


@unittest.skipUnless(_mixed_models_exist(),
                     "Run test/gen_mixed_kernel_models.py first")
class TestMixedKernelActive(unittest.TestCase):
    """_active_kernels returns the right KernelDesc objects."""

    def test_vectorop_only_active(self):
        path = _model("single_add.onnx") if _models_exist() else None
        if path is None:
            self.skipTest("VOP-only models not generated")
        gen = CodeGenerator(OnnxGraph(path), model_path=path)
        names = [kd.name for kd in gen._active_kernels]
        self.assertEqual(names, ["VectorOPKernel"])

    def test_matmul_only_active(self):
        path = _model("mm_batch.onnx") if _models_exist() else None
        if path is None:
            self.skipTest("MatMul-only models not generated")
        gen = CodeGenerator(OnnxGraph(path), model_path=path)
        names = [kd.name for kd in gen._active_kernels]
        self.assertEqual(names, ["MatmulKernel"])

    def test_mixed_matmul_relu_both_active(self):
        gen = _gen("mixed_matmul_relu.onnx")
        names = [kd.name for kd in gen._active_kernels]
        self.assertIn("VectorOPKernel", names)
        self.assertIn("MatmulKernel",   names)

    def test_active_kernel_order(self):
        """VectorOPKernel always precedes MatmulKernel (registry insertion order)."""
        gen = _gen("mixed_matmul_relu.onnx")
        names = [kd.name for kd in gen._active_kernels]
        self.assertLess(names.index("VectorOPKernel"), names.index("MatmulKernel"))


@unittest.skipUnless(_mixed_models_exist(),
                     "Run test/gen_mixed_kernel_models.py first")
class TestMixedKernelSource(unittest.TestCase):
    """Generated inference.c is correct for mixed-kernel models."""

    def _src(self, name: str) -> str:
        return _gen(name).generate_source()

    def test_both_headers_included(self):
        s = self._src("mixed_matmul_relu.onnx")
        self.assertIn('#include "xvectoropkernel.h"', s)
        self.assertIn('#include "xmatmulkernel.h"',   s)

    def test_both_static_instances(self):
        s = self._src("mixed_matmul_relu.onnx")
        self.assertIn("XVectoropkernel s_vectoropkernel", s)
        self.assertIn("XMatmulkernel s_matmulkernel",     s)

    def test_no_single_s_kernel(self):
        """Old generic 's_kernel' variable must not appear in multi-kernel output."""
        s = self._src("mixed_matmul_relu.onnx")
        self.assertNotIn("s_kernel;", s)

    def test_init_has_both_params(self):
        s = self._src("mixed_matmul_relu.onnx")
        self.assertIn("vectoropkernel_instance", s)
        self.assertIn("matmulkernel_instance",   s)

    def test_init_calls_both_initialize(self):
        s = self._src("mixed_matmul_relu.onnx")
        self.assertIn("XVectoropkernel_Initialize(&s_vectoropkernel", s)
        self.assertIn("XMatmulkernel_Initialize(&s_matmulkernel",     s)

    def test_run_op_uses_named_var(self):
        s = self._src("mixed_matmul_relu.onnx")
        self.assertIn("s_vectoropkernel", s)

    def test_run_matmul_uses_named_var(self):
        s = self._src("mixed_matmul_relu.onnx")
        self.assertIn("s_matmulkernel", s)

    def test_vectorop_only_uses_named_var(self):
        """VectorOP-only model also uses the named variable."""
        if not _models_exist():
            self.skipTest("VOP-only models not generated")
        path = _model("single_add.onnx")
        s = CodeGenerator(OnnxGraph(path), model_path=path).generate_source()
        self.assertIn("s_vectoropkernel", s)
        self.assertNotIn("s_kernel;", s)


@unittest.skipUnless(_mixed_models_exist(),
                     "Run test/gen_mixed_kernel_models.py first")
class TestMixedKernelHeader(unittest.TestCase):
    """Generated inference.h is correct for mixed-kernel models."""

    def _hdr(self, name: str) -> str:
        return _gen(name).generate_header()

    def test_both_uio_macros(self):
        h = self._hdr("mixed_matmul_relu.onnx")
        self.assertIn("INFERENCE_VECTOROPKERNEL_INSTANCE", h)
        self.assertIn("INFERENCE_MATMULKERNEL_INSTANCE",   h)

    def test_uio_defaults_match_dtsi(self):
        h = self._hdr("mixed_matmul_relu.onnx")
        self.assertIn('"VectorOPKernel_0"', h)
        self.assertIn('"MatmulKernel_0"',   h)

    def test_init_signature_has_both_params(self):
        h = self._hdr("mixed_matmul_relu.onnx")
        self.assertIn("vectoropkernel_instance", h)
        self.assertIn("matmulkernel_instance",   h)

    def test_single_kernel_init_single_param(self):
        """VectorOP-only model has exactly one init param."""
        if not _models_exist():
            self.skipTest("VOP-only models not generated")
        path = _model("single_add.onnx")
        h = CodeGenerator(OnnxGraph(path), model_path=path).generate_header()
        self.assertIn("vectoropkernel_instance", h)
        self.assertNotIn("matmulkernel_instance", h)


@unittest.skipUnless(_mixed_models_exist(),
                     "Run test/gen_mixed_kernel_models.py first")
class TestMixedKernelCMake(unittest.TestCase):
    """Generated CMakeLists.txt handles both kernel driver sets."""

    def _cmake(self, name: str) -> str:
        return _gen(name).generate_cmake()

    def test_both_driver_common_files(self):
        cm = self._cmake("mixed_matmul_relu.onnx")
        self.assertIn("driver/xvectoropkernel.c", cm)
        self.assertIn("driver/xmatmulkernel.c",   cm)

    def test_both_sinit_files(self):
        cm = self._cmake("mixed_matmul_relu.onnx")
        self.assertIn("xvectoropkernel_sinit.c", cm)
        self.assertIn("xmatmulkernel_sinit.c",   cm)

    def test_both_linux_files(self):
        cm = self._cmake("mixed_matmul_relu.onnx")
        self.assertIn("xvectoropkernel_linux.c", cm)
        self.assertIn("xmatmulkernel_linux.c",   cm)

    def test_axi_addresses_in_comment(self):
        cm = self._cmake("mixed_matmul_relu.onnx")
        self.assertIn("0xA0000000", cm)   # VectorOPKernel AXI base
        self.assertIn("0xA0010000", cm)   # MatmulKernel AXI base


@unittest.skipUnless(_mixed_models_exist(),
                     "Run test/gen_mixed_kernel_models.py first")
class TestMixedKernelTestGen(unittest.TestCase):
    """Generated test_inference.c calls inference_init() with per-kernel args."""

    def _test(self, name: str) -> str:
        return _gen(name).generate_test()

    def test_both_instance_macros(self):
        t = self._test("mixed_matmul_relu.onnx")
        self.assertIn("INFERENCE_VECTOROPKERNEL_INSTANCE", t)
        self.assertIn("INFERENCE_MATMULKERNEL_INSTANCE",   t)

    def test_both_uio_defaults(self):
        t = self._test("mixed_matmul_relu.onnx")
        self.assertIn('"VectorOPKernel_0"', t)
        self.assertIn('"MatmulKernel_0"',   t)

    def test_inference_init_call_has_both_args(self):
        t = self._test("mixed_matmul_relu.onnx")
        self.assertIn("INFERENCE_VECTOROPKERNEL_INSTANCE", t)
        self.assertIn("INFERENCE_MATMULKERNEL_INSTANCE",   t)
        # Both must appear inside the inference_init() call
        init_pos  = t.index("inference_init(")
        paren_end = t.index(");", init_pos)
        call_text = t[init_pos:paren_end]
        self.assertIn("INFERENCE_VECTOROPKERNEL_INSTANCE", call_text)
        self.assertIn("INFERENCE_MATMULKERNEL_INSTANCE",   call_text)


@unittest.skipUnless(_mixed_models_exist(),
                     "Run test/gen_mixed_kernel_models.py first")
class TestKernelRegistry(unittest.TestCase):
    """Sanity checks on KERNEL_REGISTRY contents."""

    def test_vectorop_entry(self):
        kd = KERNEL_REGISTRY["VectorOPKernel"]
        self.assertEqual(kd.driver_prefix, "xvectoropkernel")
        self.assertEqual(kd.c_type, "XVectoropkernel")
        self.assertEqual(kd.uio_default, "VectorOPKernel_0")
        self.assertEqual(kd.axi_base, 0xA000_0000)
        self.assertIn("xvectoropkernel.h", kd.driver_files)

    def test_matmul_entry(self):
        kd = KERNEL_REGISTRY["MatmulKernel"]
        self.assertEqual(kd.driver_prefix, "xmatmulkernel")
        self.assertEqual(kd.c_type, "XMatmulkernel")
        self.assertEqual(kd.uio_default, "MatmulKernel_0")
        self.assertEqual(kd.axi_base, 0xA001_0000)
        self.assertIn("xmatmulkernel.h", kd.driver_files)

    def test_c_var_names(self):
        self.assertEqual(KERNEL_REGISTRY["VectorOPKernel"].c_var, "s_vectoropkernel")
        self.assertEqual(KERNEL_REGISTRY["MatmulKernel"].c_var,   "s_matmulkernel")

    def test_init_param_names(self):
        self.assertEqual(KERNEL_REGISTRY["VectorOPKernel"].init_param, "vectoropkernel_instance")
        self.assertEqual(KERNEL_REGISTRY["MatmulKernel"].init_param,   "matmulkernel_instance")

    def test_instance_macros(self):
        self.assertEqual(
            KERNEL_REGISTRY["VectorOPKernel"].instance_macro,
            "INFERENCE_VECTOROPKERNEL_INSTANCE"
        )
        self.assertEqual(
            KERNEL_REGISTRY["MatmulKernel"].instance_macro,
            "INFERENCE_MATMULKERNEL_INSTANCE"
        )


# ======================================================================
# Tests for the three runtime-failure fixes:
#   Fix A — _broadcast_io_map pass 2 excludes MatmulNode outputs
#   Fix B — _compute_alloc_sizes pads MatmulNode output when it feeds
#             a downstream broadcast VectorOP
#   Fix C — _get_matmul_strides() returns correct (a_row_stride, c_row_stride)
#   Fix D — MatmulNode.emit_call() generates strided run_matmul() correctly
#   Fix E — _source.py uses _get_matmul_strides when emitting MatmulNode calls
# ======================================================================

@unittest.skipUnless(_mixed_models_exist(),
                     "Run test/gen_mixed_kernel_models.py first")
class TestBroadcastIoMapExcludesMatmul(unittest.TestCase):
    """
    Fix A: _broadcast_io_map() pass 2 must not propagate a bcast entry
    from a strided input into a MatmulNode's output.

    mixed_add_matmul:     Add(X,bias)->Z  MatMul(Z,W)->Y
      Z is an advancing input of the broadcast Add → Z is in the map.
      Y is MatMul's output — must NOT inherit Z's entry (Fix A).

    mixed_matmul_add_relu: MatMul(X,W)->Z  Add(Z,bias)->A  Relu(A)->Y
      Z is a MatmulNode output — must NOT appear in the map (Fix A).
      A is the broadcast Add output → A is in the map.
      Y is the Relu output, which propagates from A → Y is in the map.
    """

    def _bmap(self, name: str) -> dict:
        return _gen(name)._broadcast_io_map()

    def test_add_matmul_z_in_map(self):
        """Z is an advancing input of the broadcast Add — must be in the map."""
        bmap = self._bmap("mixed_add_matmul.onnx")
        self.assertIn("Z", bmap)

    def test_add_matmul_y_not_in_map(self):
        """Y is a MatmulNode output — must NOT be in the map (Fix A)."""
        bmap = self._bmap("mixed_add_matmul.onnx")
        self.assertNotIn("Y", bmap)

    def test_matmul_add_relu_z_in_map(self):
        """Z is an advancing input of the broadcast Add(Z, bias) → A — pass 1
        correctly adds it to the map so VectorOPKernel reads it at strided offsets."""
        bmap = self._bmap("mixed_matmul_add_relu.onnx")
        self.assertIn("Z", bmap)

    def test_matmul_add_relu_a_in_map(self):
        """A is the broadcast Add output — must be in the map."""
        bmap = self._bmap("mixed_matmul_add_relu.onnx")
        self.assertIn("A", bmap)

    def test_matmul_add_relu_y_in_map(self):
        """Y is Relu(A) — stride propagates from A through the non-broadcast
        Relu, so Y must be in the map."""
        bmap = self._bmap("mixed_matmul_add_relu.onnx")
        self.assertIn("Y", bmap)

    def test_two_layer_mlp_z1_in_map(self):
        """z1 is an advancing input of the first broadcast Add(z1,b1) — pass 1
        correctly adds it so VectorOPKernel reads it at strided offsets."""
        bmap = self._bmap("mixed_two_layer_mlp.onnx")
        self.assertIn("z1", bmap)

    def test_two_layer_mlp_z2_in_map(self):
        """z2 is an advancing input of the second broadcast Add(z2,b2) — pass 1
        correctly adds it so VectorOPKernel reads it at strided offsets."""
        bmap = self._bmap("mixed_two_layer_mlp.onnx")
        self.assertIn("z2", bmap)

    def test_two_layer_mlp_a1_in_map(self):
        """a1 is the first broadcast Add output — must be in the map."""
        bmap = self._bmap("mixed_two_layer_mlp.onnx")
        self.assertIn("a1", bmap)

    def test_two_layer_mlp_h1_in_map(self):
        """h1 is Relu(a1) — stride propagates from a1, so h1 must be in the map."""
        bmap = self._bmap("mixed_two_layer_mlp.onnx")
        self.assertIn("h1", bmap)


@unittest.skipUnless(_mixed_models_exist(),
                     "Run test/gen_mixed_kernel_models.py first")
class TestAllocSizesForMixedModels(unittest.TestCase):
    """
    Fix B: _compute_alloc_sizes() must pad a MatmulNode output when it is
    used as an advancing input of a downstream broadcast VectorOP.

    All models use n=4, aligned_chunk_size=8.
    Padded alloc = outer_count(4) × aligned_chunk(8) = 32.

    mixed_add_matmul (n=4, k=8, m=4):
      Z[4,8] numel=32 — advancing input of broadcast Add, aligned_chunk=8.
      Z.alloc must equal numel (32) — no extra padding needed since K==aligned_chunk.
      Y[4,4] numel=16 — MatMul output with no downstream broadcast.
      Y.alloc must equal numel (16) — not inflated by Fix A.

    mixed_matmul_add_relu (n=4, k=8, m=4):
      Z[4,4] numel=16 — MatMul output that feeds broadcast Add as advancing input.
      Z.alloc must be 32 (padded to outer×aligned_chunk).
      Y[4,4] numel=16 — Relu output, inherits Z→A→Y padding.
      Y.alloc must be 32.

    mixed_two_layer_mlp (n=4, k=8, h=6, m=4):
      z1[4,6] numel=24 — first MatMul output, feeds broadcast Add (chunk=6, aligned=8).
      z1.alloc must be 32.
      h1[4,6] numel=24 — Relu output, inherits padding from z1→a1→h1.
      h1.alloc must be 32.
      z2[4,4] numel=16 — second MatMul output, feeds broadcast Add (chunk=4, aligned=8).
      z2.alloc must be 32.
    """

    def _alloc(self, name: str) -> dict:
        return _gen(name)._alloc_sizes

    # ---- mixed_add_matmul ------------------------------------------ #

    def test_add_matmul_z_alloc_not_inflated(self):
        """Z.numel==32 and aligned_chunk==8==K — no extra padding needed."""
        alloc = self._alloc("mixed_add_matmul.onnx")
        self.assertEqual(alloc["Z"], 32)

    def test_add_matmul_y_alloc_equals_numel(self):
        """Y is the MatMul output with no downstream broadcast — alloc must
        equal numel (16), not the inflated 32 that the old bug produced."""
        alloc = self._alloc("mixed_add_matmul.onnx")
        self.assertEqual(alloc["Y"], 16)

    # ---- mixed_matmul_add_relu ------------------------------------- #

    def test_matmul_add_relu_z_alloc_padded(self):
        """Z[4,4] numel=16 feeds broadcast Add — alloc must be 4×8=32."""
        alloc = self._alloc("mixed_matmul_add_relu.onnx")
        self.assertEqual(alloc["Z"], 32)

    def test_matmul_add_relu_a_alloc_padded(self):
        """A is the broadcast Add output — alloc must be 32."""
        alloc = self._alloc("mixed_matmul_add_relu.onnx")
        self.assertEqual(alloc["A"], 32)

    def test_matmul_add_relu_y_alloc_padded(self):
        """Y=Relu(A) inherits A's padded alloc — must be 32."""
        alloc = self._alloc("mixed_matmul_add_relu.onnx")
        self.assertEqual(alloc["Y"], 32)

    # ---- mixed_two_layer_mlp --------------------------------------- #

    def test_two_layer_mlp_z1_alloc_padded(self):
        """z1[4,6] numel=24 feeds broadcast Add (aligned_chunk=8) — alloc=32."""
        alloc = self._alloc("mixed_two_layer_mlp.onnx")
        self.assertEqual(alloc["z1"], 32)

    def test_two_layer_mlp_h1_alloc_padded(self):
        """h1 is Relu(a1), inherits padding — alloc=32."""
        alloc = self._alloc("mixed_two_layer_mlp.onnx")
        self.assertEqual(alloc["h1"], 32)

    def test_two_layer_mlp_z2_alloc_padded(self):
        """z2[4,4] numel=16 feeds broadcast Add — alloc=32."""
        alloc = self._alloc("mixed_two_layer_mlp.onnx")
        self.assertEqual(alloc["z2"], 32)


@unittest.skipUnless(_mixed_models_exist(),
                     "Run test/gen_mixed_kernel_models.py first")
class TestMatmulLayoutStrides(unittest.TestCase):
    """
    Fix C: MatmulNode row strides are now derived from TensorLayout.

    _layouts[name].gap > 0 (stride > chunk) indicates alignment padding;
    the emitted run_matmul() then uses the row-strided decomposition.

    mixed_matmul_relu:     MatMul(X,W)->Z, Relu(Z)->Y — no broadcast VectorOP,
      no padding → no MatmulNode needs row strides.

    mixed_add_matmul:      Add(X,bias)->Z, MatMul(Z,W)->Y — Z.numel==32==Z.alloc
      (K==aligned_chunk, gap=0), Y.alloc==Y.numel → no strides needed.

    mixed_matmul_add_relu: MatMul(X,W)->Z, Add(Z,bias)->A, Relu(A)->Y
      Z.layout: n_chunks=4, stride=8, chunk=4, gap=4 → c_row_stride=8.
      X.layout: flat (gap=0) → a_row_stride=0.

    mixed_two_layer_mlp:   MatMul(X,W1)->z1, …, MatMul(h1,W2)->z2, …
      z1.layout: n_chunks=4, stride=8, chunk=6, gap=2 → c_row_stride=8.
      h1.layout: n_chunks=4, stride=8, chunk=6, gap=2 → a_row_stride=8.
      z2.layout: n_chunks=4, stride=8, chunk=4, gap=4 → c_row_stride=8.
    """

    def _matmul_strides(self, name: str) -> dict:
        """Return {output_name: (a_row_stride, c_row_stride)} for MatmulNodes
        that need row-strided decomposition (gap > 0 on A input or Y output)."""
        from src.nodes import MatmulNode
        gen = _gen(name)
        result = {}
        for sn in gen._graph.nodes:
            if not isinstance(sn, MatmulNode):
                continue
            if sn.outer_count > 1:
                continue
            a_lay = gen._layouts.get(sn.inputs[0].onnx_name)
            y_lay = gen._layouts.get(sn.output.onnx_name)
            a_row = a_lay.stride if (a_lay and a_lay.gap > 0) else 0
            c_row = y_lay.stride if (y_lay and y_lay.gap > 0) else 0
            if a_row != 0 or c_row != 0:
                result[sn.output.onnx_name] = (a_row, c_row)
        return result

    def test_matmul_relu_no_strides_needed(self):
        """No broadcast VectorOP in graph → no MatmulNode needs strides."""
        strides = self._matmul_strides("mixed_matmul_relu.onnx")
        self.assertEqual(strides, {})

    def test_add_matmul_no_strides_needed(self):
        """Z.gap==0 (K equals aligned_chunk) and Y.gap==0 → no strides needed."""
        strides = self._matmul_strides("mixed_add_matmul.onnx")
        self.assertEqual(strides, {})

    def test_matmul_add_relu_a_row_stride_zero(self):
        """X is the graph input with flat layout — a_row_stride must be 0."""
        strides = self._matmul_strides("mixed_matmul_add_relu.onnx")
        a_row, _ = strides["Z"]
        self.assertEqual(a_row, 0)

    def test_matmul_add_relu_c_row_stride(self):
        """Z.layout.stride=8, gap=4 → c_row_stride=8."""
        strides = self._matmul_strides("mixed_matmul_add_relu.onnx")
        _, c_row = strides["Z"]
        self.assertEqual(c_row, 8)

    def test_two_layer_mlp_z1_a_row_stride_zero(self):
        """First MatMul reads X which has flat layout — a_row_stride=0."""
        strides = self._matmul_strides("mixed_two_layer_mlp.onnx")
        a_row, _ = strides["z1"]
        self.assertEqual(a_row, 0)

    def test_two_layer_mlp_z1_c_row_stride(self):
        """z1.layout.stride=8, gap>0 → c_row_stride=8."""
        strides = self._matmul_strides("mixed_two_layer_mlp.onnx")
        _, c_row = strides["z1"]
        self.assertEqual(c_row, 8)

    def test_two_layer_mlp_z2_a_row_stride(self):
        """h1.layout.stride=8, gap>0 → a_row_stride=8 (reading strided h1)."""
        strides = self._matmul_strides("mixed_two_layer_mlp.onnx")
        a_row, _ = strides["z2"]
        self.assertEqual(a_row, 8)

    def test_two_layer_mlp_z2_c_row_stride(self):
        """z2.layout.stride=8, gap>0 → c_row_stride=8."""
        strides = self._matmul_strides("mixed_two_layer_mlp.onnx")
        _, c_row = strides["z2"]
        self.assertEqual(c_row, 8)


@unittest.skipUnless(_mixed_models_exist(),
                     "Run test/gen_mixed_kernel_models.py first")
class TestRunMatmulCallStrided(unittest.TestCase):
    """
    Fixes D+E: generated run_matmul() calls must use the row-strided
    decomposition (batch=N, n=1) whenever a_row_stride or c_row_stride
    is non-zero, and the standard layout (batch=1, n=N) otherwise.

    All models use n=4.  The strided form is:
        run_matmul(a, b, c,
                   1u, <k>u, <m>u, 4u,
                   <eff_a>u, 0u, <eff_c>u);

    The natural form is:
        run_matmul(a, b, c,
                   <n>u, <k>u, <m>u, 1u,
                   0u, 0u, 0u);
    """

    def _src(self, name: str) -> str:
        return _gen(name).generate_source()

    def test_matmul_relu_natural_call(self):
        """No striding needed — MatMul(X,W)->Z followed by Relu."""
        s = self._src("mixed_matmul_relu.onnx")
        self.assertIn("4u, 8u, 4u, 1u", s)   # n=4, k=8, m=4, batch=1
        self.assertIn("0u, 0u, 0u", s)         # all strides zero

    def test_add_matmul_natural_call(self):
        """Z.numel==Z.alloc (K=aligned_chunk=8) — no striding needed."""
        s = self._src("mixed_add_matmul.onnx")
        self.assertIn("4u, 8u, 4u, 1u", s)   # n=4, k=8, m=4, batch=1
        self.assertIn("0u, 0u, 0u", s)

    def test_matmul_add_relu_strided_call_dimensions(self):
        """c_row_stride=8 → decomposed as batch=4, n=1."""
        s = self._src("mixed_matmul_add_relu.onnx")
        self.assertIn("1u, 8u, 4u, 4u", s)   # n=1, k=8, m=4, batch=4

    def test_matmul_add_relu_strided_call_strides(self):
        """eff_a=K=8 (natural), eff_c=8 (aligned_chunk)."""
        s = self._src("mixed_matmul_add_relu.onnx")
        self.assertIn("8u, 0u, 8u", s)         # a_stride=8, b_stride=0, c_stride=8

    def test_two_layer_mlp_first_matmul_dimensions(self):
        """MatMul(X[4,8], W1[8,6]): c_row_stride=8 → n=1, k=8, m=6, batch=4."""
        s = self._src("mixed_two_layer_mlp.onnx")
        self.assertIn("1u, 8u, 6u, 4u", s)

    def test_two_layer_mlp_second_matmul_dimensions(self):
        """MatMul(h1[4,6], W2[6,4]): a_row_stride=8, c_row_stride=8 → n=1, k=6, m=4, batch=4."""
        s = self._src("mixed_two_layer_mlp.onnx")
        self.assertIn("1u, 6u, 4u, 4u", s)

    def test_two_layer_mlp_both_matmul_strides(self):
        """Both MatMul calls use a_stride=8, b_stride=0, c_stride=8."""
        s = self._src("mixed_two_layer_mlp.onnx")
        # count occurrences: there must be exactly two
        self.assertEqual(s.count("8u, 0u, 8u"), 2)


@unittest.skipUnless(_mixed_models_exist(),
                     "Run test/gen_mixed_kernel_models.py first")
class TestHeaderSizeMacrosForMixedModels(unittest.TestCase):
    """
    Fix A (header side): when a MatmulNode output is no longer added to
    _broadcast_io_map, the header must not emit a CHUNK macro for it and
    must size it at numel, not outer_count*aligned_chunk.

    Before Fix A, mixed_add_matmul emitted:
        INFERENCE_Y_SIZE   (4u * INFERENCE_Y_CHUNK_STRIDE)  /* 32 instead of 16 */
    and the test binary verified 32 elements — positions 16..31 were DMA-zero,
    causing "FAIL Y[chunk=2,j=0]: got 0.0000 expected ..." on hardware.
    """

    def _hdr(self, name: str) -> str:
        return _gen(name).generate_header()

    def test_add_matmul_y_size_is_numel(self):
        """INFERENCE_Y_SIZE must be the bare numel (16), not 32."""
        h = self._hdr("mixed_add_matmul.onnx")
        self.assertIn("INFERENCE_Y_SIZE", h)
        self.assertIn("16u", h)
        # Must not reference the (non-existent) CHUNK_STRIDE macro for Y
        self.assertNotIn("INFERENCE_Y_CHUNK_STRIDE", h)

    def test_add_matmul_no_y_chunk_macro(self):
        """No CHUNK macro should be emitted for Y (it is a MatmulNode output)."""
        h = self._hdr("mixed_add_matmul.onnx")
        self.assertNotIn("INFERENCE_Y_CHUNK", h)

    def test_matmul_add_relu_z_no_chunk_macro(self):
        """Z is a MatmulNode output — no CHUNK macro must be emitted for it."""
        h = self._hdr("mixed_matmul_add_relu.onnx")
        self.assertNotIn("INFERENCE_Z_CHUNK", h)

    def test_matmul_add_relu_y_size_uses_stride_macro(self):
        """Y is the final graph output and inherits broadcast stride from A.
        INFERENCE_Y_SIZE must reference INFERENCE_A_CHUNK_STRIDE so the
        caller allocates enough space for all aligned rows.  There is no
        INFERENCE_Y_CHUNK macro — Y borrows the broadcast Add output's macro."""
        h = self._hdr("mixed_matmul_add_relu.onnx")
        self.assertIn("INFERENCE_Y_SIZE", h)
        self.assertIn("INFERENCE_A_CHUNK_STRIDE", h)
        # The SIZE expression must multiply outer_count by the stride macro
        self.assertIn("4u * INFERENCE_A_CHUNK_STRIDE", h)

    def test_two_layer_mlp_z1_no_chunk_macro(self):
        """z1 is the first MatMul output — no CHUNK macro emitted."""
        h = self._hdr("mixed_two_layer_mlp.onnx")
        self.assertNotIn("INFERENCE_Z1_CHUNK", h)

    def test_two_layer_mlp_z2_no_chunk_macro(self):
        """z2 is the second MatMul output — no CHUNK macro emitted."""
        h = self._hdr("mixed_two_layer_mlp.onnx")
        self.assertNotIn("INFERENCE_Z2_CHUNK", h)


def _new_mixed_models_exist() -> bool:
    return os.path.isfile(_mixed_model("mixed_add_matmul_unaligned.onnx"))


def _matmul_strides(gen) -> dict:
    """Return {output_name: (a_row_stride, c_row_stride)} for MatmulNodes
    that need row-strided decomposition (gap > 0 on A input or Y output)
    and have outer_count == 1."""
    from src.nodes import MatmulNode
    result = {}
    for sn in gen._graph.nodes:
        if not isinstance(sn, MatmulNode) or sn.outer_count > 1:
            continue
        a_lay = gen._layouts.get(sn.inputs[0].onnx_name)
        y_lay = gen._layouts.get(sn.output.onnx_name)
        a_row = a_lay.stride if (a_lay and a_lay.gap > 0) else 0
        c_row = y_lay.stride if (y_lay and y_lay.gap > 0) else 0
        if a_row or c_row:
            result[sn.output.onnx_name] = (a_row, c_row)
    return result


@unittest.skipUnless(_new_mixed_models_exist(),
                     "Run test/gen_mixed_kernel_models.py first")
class TestAddMatmulUnaligned(unittest.TestCase):
    """mixed_add_matmul_unaligned: Add(X[4,6], bias[6])→Z[4,6], MatMul(Z,W[6,4])→Y[4,4].
    K=6 unaligned → aligned_chunk=8, gap=2 in Z's layout.
    Y is a MatmulNode output (Fix A: not in bcast_map).
    """

    def setUp(self):
        self.gen = _gen("mixed_add_matmul_unaligned.onnx")

    def test_z_alloc(self):
        """Z[4,6] numel=24, outer=4, aligned_chunk=8 → alloc=32."""
        self.assertEqual(self.gen._alloc_sizes["Z"], 32)

    def test_z_layout(self):
        """Z: n_chunks=4, chunk=6, stride=8, gap=2."""
        lay = self.gen._layouts["Z"]
        self.assertEqual(lay.n_chunks, 4)
        self.assertEqual(lay.chunk, 6)
        self.assertEqual(lay.stride, 8)
        self.assertEqual(lay.gap, 2)

    def test_y_alloc(self):
        """Y is MatmulNode output (Fix A) — alloc must equal numel (16), not 32."""
        self.assertEqual(self.gen._alloc_sizes["Y"], 16)

    def test_y_not_in_bcast_map(self):
        """Y is a MatmulNode output — must NOT be in _broadcast_io_map (Fix A)."""
        bmap = self.gen._broadcast_io_map()
        self.assertNotIn("Y", bmap)

    def test_z_in_bcast_map(self):
        """Z is an advancing input of broadcast Add — must be in the map."""
        bmap = self.gen._broadcast_io_map()
        self.assertIn("Z", bmap)

    def test_matmul_strides(self):
        """a_row_stride=8 (Z has gap=2), c_row_stride=0 (Y flat)."""
        strides = _matmul_strides(self.gen)
        self.assertIn("Y", strides)
        a_row, c_row = strides["Y"]
        self.assertEqual(a_row, 8)
        self.assertEqual(c_row, 0)

    def test_run_matmul_call(self):
        """run_matmul(Z, W, Y, 1u, 6u, 4u, 4u, 8u, 0u, 4u)."""
        src = self.gen.generate_source()
        self.assertIn("1u, 6u, 4u, 4u", src)
        self.assertIn("8u, 0u, 4u", src)


@unittest.skipUnless(_new_mixed_models_exist(),
                     "Run test/gen_mixed_kernel_models.py first")
class TestMatmulScaleBias(unittest.TestCase):
    """mixed_matmul_scale_bias: MatMul(X[4,8],W[8,4])→Z, Mul(Z,scale[4])→S, Add(S,bias[4])→Y.
    Mul and Add both broadcast (outer=4, chunk=4, aligned=8) → gap=4.
    MatMul: c_row_stride=8 (Z has gap), a_row_stride=0 (X flat).
    """

    def setUp(self):
        self.gen = _gen("mixed_matmul_scale_bias.onnx")

    def test_z_alloc(self):
        """Z[4,4] numel=16, downstream broadcast → alloc=32."""
        self.assertEqual(self.gen._alloc_sizes["Z"], 32)

    def test_s_alloc(self):
        """S[4,4] is Mul output with outer=4, aligned=8 → alloc=32."""
        self.assertEqual(self.gen._alloc_sizes["S"], 32)

    def test_y_alloc(self):
        """Y[4,4] is Add output with outer=4, aligned=8 → alloc=32."""
        self.assertEqual(self.gen._alloc_sizes["Y"], 32)

    def test_z_layout_gap(self):
        """Z: n_chunks=4, stride=8, chunk=4, gap=4."""
        lay = self.gen._layouts["Z"]
        self.assertEqual(lay.n_chunks, 4)
        self.assertEqual(lay.stride, 8)
        self.assertEqual(lay.gap, 4)

    def test_matmul_c_row_stride(self):
        """c_row_stride=8 (Z gap=4), a_row_stride=0 (X flat)."""
        strides = _matmul_strides(self.gen)
        self.assertIn("Z", strides)
        a_row, c_row = strides["Z"]
        self.assertEqual(a_row, 0)
        self.assertEqual(c_row, 8)

    def test_run_matmul_call(self):
        """Strided form: run_matmul(X, W, Z, 1u, 8u, 4u, 4u, 8u, 0u, 8u)."""
        src = self.gen.generate_source()
        self.assertIn("1u, 8u, 4u, 4u", src)
        self.assertIn("8u, 0u, 8u", src)

    def test_header_s_chunk_defined(self):
        """INFERENCE_S_CHUNK must be emitted (S is Mul output)."""
        hdr = self.gen.generate_header()
        self.assertIn("INFERENCE_S_CHUNK", hdr)

    def test_header_y_chunk_defined(self):
        """INFERENCE_Y_CHUNK must be emitted (Y is Add output)."""
        hdr = self.gen.generate_header()
        self.assertIn("INFERENCE_Y_CHUNK", hdr)


@unittest.skipUnless(_new_mixed_models_exist(),
                     "Run test/gen_mixed_kernel_models.py first")
class TestOuterMatmulRelu(unittest.TestCase):
    """mixed_outer_matmul_relu: A[2,3,4,6] @ B[3,6,4] → Z[2,3,4,4], Relu(Z) → Y[2,3,4,4].
    outer_count=2, batch=3, n=4, k=6, m=4.
    No VectorOP broadcast → Z and Y are flat.
    """

    def setUp(self):
        self.gen = _gen("mixed_outer_matmul_relu.onnx")

    def test_z_alloc(self):
        """Z[2,3,4,4] numel=96, no broadcast VectorOP → alloc=96 (flat)."""
        self.assertEqual(self.gen._alloc_sizes["Z"], 96)

    def test_y_alloc(self):
        """Y[2,3,4,4] numel=96 → alloc=96 (flat)."""
        self.assertEqual(self.gen._alloc_sizes["Y"], 96)

    def test_z_layout_flat(self):
        """Z: flat layout (n_chunks=1, gap=0)."""
        lay = self.gen._layouts["Z"]
        self.assertEqual(lay.n_chunks, 1)
        self.assertEqual(lay.gap, 0)

    def test_y_not_in_bcast_map(self):
        """Y is MatmulNode output — not in bcast_map (Fix A)."""
        bmap = self.gen._broadcast_io_map()
        self.assertNotIn("Y", bmap)

    def test_outer_loop_emitted(self):
        """run_matmul_at() inside a for-loop over 2 outer iterations."""
        src = self.gen.generate_source()
        self.assertIn("run_matmul_at(", src)
        self.assertIn("_i < 2u", src)

    def test_outer_strides_in_source(self):
        """A outer stride=72, B outer stride=0, C outer stride=48."""
        src = self.gen.generate_source()
        self.assertIn("_i * 72u", src)
        self.assertIn("_i * 0u", src)
        self.assertIn("_i * 48u", src)

    def test_inner_dimensions(self):
        """Inner call: n=4, k=6, m=4, batch=3."""
        src = self.gen.generate_source()
        self.assertIn("4u, 6u, 4u, 3u", src)

    def test_relu_runs_on_full_z(self):
        """Relu: run_op(Z, NULL, Y, 96u, VECTOROP_RELU)."""
        src = self.gen.generate_source()
        self.assertIn("run_op(", src)
        self.assertIn("96u", src)
        self.assertIn("VECTOROP_RELU", src)


@unittest.skipUnless(_new_mixed_models_exist(),
                     "Run test/gen_mixed_kernel_models.py first")
class TestRelu6MatmulAdd(unittest.TestCase):
    """mixed_relu6_matmul_add: Relu6(X[4,8])→Z[4,8], MatMul(Z,W[8,6])→A[4,6], Add(A,bias[6])→Y[4,6].
    Add broadcasts (outer=4, chunk=6, aligned=8) → A.alloc=32 (gap=2).
    MatMul: a_row_stride=0 (Z flat, K=8 aligned), c_row_stride=8 (A gap=2).
    """

    def setUp(self):
        self.gen = _gen("mixed_relu6_matmul_add.onnx")

    def test_z_alloc(self):
        """Z[4,8] numel=32, K=8 aligned — no gap → alloc=32 (flat)."""
        self.assertEqual(self.gen._alloc_sizes["Z"], 32)

    def test_z_layout_flat(self):
        """Z: flat (n_chunks=1, gap=0)."""
        lay = self.gen._layouts["Z"]
        self.assertEqual(lay.n_chunks, 1)
        self.assertEqual(lay.gap, 0)

    def test_a_alloc(self):
        """A[4,6] numel=24, outer=4, aligned_chunk=8 → alloc=32 (gap=2)."""
        self.assertEqual(self.gen._alloc_sizes["A"], 32)

    def test_a_layout(self):
        """A: n_chunks=4, chunk=6, stride=8, gap=2."""
        lay = self.gen._layouts["A"]
        self.assertEqual(lay.n_chunks, 4)
        self.assertEqual(lay.chunk, 6)
        self.assertEqual(lay.stride, 8)
        self.assertEqual(lay.gap, 2)

    def test_a_not_in_bcast_map_as_matmul_output(self):
        """A is a MatmulNode output — Fix A: must NOT be in bcast_map via pass 2
        (it IS in bcast_map via pass 1 as the broadcast Add output)."""
        bmap = self.gen._broadcast_io_map()
        # A must be in the map because it's the advancing input of Add (pass 1).
        self.assertIn("A", bmap)

    def test_matmul_strides(self):
        """a_row_stride=0 (Z flat, K=8), c_row_stride=8 (A gap=2)."""
        strides = _matmul_strides(self.gen)
        self.assertIn("A", strides)
        a_row, c_row = strides["A"]
        self.assertEqual(a_row, 0)
        self.assertEqual(c_row, 8)

    def test_run_matmul_call(self):
        """Strided form: run_matmul(Z, W, A, 1u, 8u, 6u, 4u, 8u, 0u, 8u)."""
        src = self.gen.generate_source()
        self.assertIn("1u, 8u, 6u, 4u", src)
        self.assertIn("8u, 0u, 8u", src)


@unittest.skipUnless(_new_mixed_models_exist(),
                     "Run test/gen_mixed_kernel_models.py first")
class TestResidual(unittest.TestCase):
    """mixed_residual: MatMul(X[4,8], W[8,4])→Z[4,4], Add(Z, res[4,4])→Y[4,4].
    res is a weight with shape [4,4] == Z shape — non-broadcast Add.
    Z.alloc=16 (flat), Y.alloc=16 (flat).
    """

    def setUp(self):
        self.gen = _gen("mixed_residual.onnx")

    def test_z_alloc(self):
        """Z[4,4] numel=16, non-broadcast downstream → alloc=16 (flat)."""
        self.assertEqual(self.gen._alloc_sizes["Z"], 16)

    def test_y_alloc(self):
        """Y[4,4] numel=16, non-broadcast Add → alloc=16 (flat)."""
        self.assertEqual(self.gen._alloc_sizes["Y"], 16)

    def test_z_layout_flat(self):
        """Z: flat (n_chunks=1, gap=0)."""
        lay = self.gen._layouts["Z"]
        self.assertEqual(lay.n_chunks, 1)
        self.assertEqual(lay.gap, 0)

    def test_y_not_in_bcast_map(self):
        """No broadcast nodes → Y not in bcast_map."""
        bmap = self.gen._broadcast_io_map()
        self.assertNotIn("Y", bmap)

    def test_matmul_natural_call(self):
        """Natural form: run_matmul(X, W, Z, 4u, 8u, 4u, 1u, 0u, 0u, 0u)."""
        src = self.gen.generate_source()
        self.assertIn("4u, 8u, 4u, 1u", src)
        self.assertIn("0u, 0u, 0u", src)

    def test_add_non_broadcast(self):
        """Non-broadcast Add: run_op(Z, res, Y, 16u, VECTOROP_ADD)."""
        src = self.gen.generate_source()
        self.assertIn("run_op(", src)
        self.assertIn("16u", src)
        self.assertIn("VECTOROP_ADD", src)

    def test_no_chunk_macros(self):
        """No CHUNK macros for Z or Y (both flat, no broadcast)."""
        hdr = self.gen.generate_header()
        self.assertNotIn("INFERENCE_Z_CHUNK", hdr)
        self.assertNotIn("INFERENCE_Y_CHUNK", hdr)


@unittest.skipUnless(_mixed_models_exist(),
                     "Run test/gen_mixed_kernel_models.py first")
class TestSkipConnection(unittest.TestCase):
    """mixed_skip_connection: X[4,8] → Z=MatMul(X,W[8,8]) → R=Relu(Z) → Y=Add(X,R).

    True skip/residual connection: Y = X + Relu(X @ W).
    X participates in both the MatMul path (as operand A) and the Add (as skip input).
    All tensors flat (n_chunks=1); Add is non-broadcast (same-shape inputs).
    """

    @classmethod
    def setUpClass(cls):
        cls.path  = _mixed_model("mixed_skip_connection.onnx")
        cls.graph = OnnxGraph(cls.path)
        cls.gen   = CodeGenerator(cls.graph, model_path=cls.path)

    # ---- Graph structure ----

    def test_three_nodes(self):
        from src.nodes import MatmulNode, ScheduledNode
        nodes = self.graph.nodes
        self.assertEqual(len(nodes), 3)
        self.assertIsInstance(nodes[0], MatmulNode)   # MatMul
        self.assertIsInstance(nodes[1], ScheduledNode) # Relu
        self.assertIsInstance(nodes[2], ScheduledNode) # Add

    def test_single_input_output(self):
        self.assertEqual(len(self.graph.input_tensors),  1)
        self.assertEqual(len(self.graph.output_tensors), 1)
        self.assertEqual(self.graph.input_tensors[0].shape,  [4, 8])
        self.assertEqual(self.graph.output_tensors[0].shape, [4, 8])

    def test_x_is_both_matmul_input_and_add_input(self):
        """X participates in MatMul (operand A) and in Add (skip path)."""
        mm  = self.graph.nodes[0]
        add = self.graph.nodes[2]
        self.assertEqual(mm.inputs[0].onnx_name, "X")
        input_names = {t.onnx_name for t in add.inputs}
        self.assertIn("X", input_names)

    # ---- MatmulNode geometry ----

    def test_matmul_dims(self):
        mm = self.graph.nodes[0]
        self.assertEqual(mm.n, 4)
        self.assertEqual(mm.k, 8)
        self.assertEqual(mm.m, 8)   # W is square [8,8]

    def test_matmul_plain_2d(self):
        """W is 2-D: batch=1, all strides zero (no batch loop)."""
        mm = self.graph.nodes[0]
        self.assertEqual(mm.batch, 1)
        self.assertEqual(mm.a_batch_stride, 0)
        self.assertEqual(mm.b_batch_stride, 0)
        self.assertEqual(mm.c_batch_stride, 0)
        self.assertEqual(mm.outer_count,    1)

    # ---- TensorLayouts ----

    def test_all_tensors_flat(self):
        for name in ("Z", "R", "Y"):
            lay = self.gen._layouts[name]
            self.assertEqual(lay.n_chunks, 1, f"{name} should be flat")
            self.assertEqual(lay.gap,      0, f"{name} should have no gap")

    def test_x_layout_flat(self):
        lay = self.gen._layouts["X"]
        self.assertEqual(lay.n_chunks, 1)
        self.assertEqual(lay.numel, 32)
        self.assertEqual(lay.alloc, 32)

    def test_z_numel(self):
        self.assertEqual(self.gen._layouts["Z"].numel, 32)  # 4*8

    def test_no_broadcast_in_io_map(self):
        self.assertEqual(self.gen._broadcast_io_map(), {})

    # ---- Generated source ----

    def _src(self):
        return self.gen.generate_source()

    def test_matmul_call_plain(self):
        """run_matmul(X, W, Z, 4u, 8u, 8u, 1u, 0u, 0u, 0u)."""
        s = self._src()
        self.assertIn("4u, 8u, 8u, 1u,", s)
        self.assertIn("0u, 0u, 0u", s)

    def test_relu_call(self):
        """run_op(Z, NULL, R, 32u, VECTOROP_RELU)."""
        s = self._src()
        self.assertIn("32u, VECTOROP_RELU", s)

    def test_add_call_with_x_as_skip(self):
        """Add uses X (skip path) and R (transformed path) → Y."""
        s = self._src()
        # run_op(X, R, Y, 32u, VECTOROP_ADD) — X is the skip input
        self.assertIn("32u, VECTOROP_ADD", s)
        self.assertIn("run_op(X, R, Y", s)

    def test_three_kernel_calls_in_order(self):
        s = self._src()
        # Search within the inference_run body to avoid matching the file banner
        body_start = s.find("void inference_run(")
        body = s[body_start:]
        mm_pos   = body.find("run_matmul(")
        relu_pos = body.find("VECTOROP_RELU")
        add_pos  = body.find("VECTOROP_ADD")
        self.assertLess(mm_pos,   relu_pos)
        self.assertLess(relu_pos, add_pos)

    def test_no_run_op_at(self):
        """No broadcast loops — run_op_at must not appear."""
        self.assertNotIn("run_op_at(", self._src())

    def test_both_kernels_active(self):
        names = [kd.name for kd in self.gen._active_kernels]
        self.assertIn("VectorOPKernel", names)
        self.assertIn("MatmulKernel",   names)

    def test_no_chunk_macros_in_header(self):
        hdr = self.gen.generate_header()
        for tensor in ("X", "Z", "R", "Y"):
            self.assertNotIn(f"INFERENCE_{tensor}_CHUNK", hdr)

    # ---- Simulation ----

    def test_simulate_output_shape(self):
        arrays = self.gen._simulate()
        self.assertEqual(list(arrays["Y"].shape), [4, 8])

    def test_simulate_residual_identity(self):
        """When W=0, Relu(X@0)=0 and Y=X+0=X: output equals input."""
        from src.graph   import OnnxGraph
        from src.codegen import CodeGenerator
        import onnx
        import numpy as np
        from onnx import helper, TensorProto, numpy_helper

        w_zero = numpy_helper.from_array(
            np.zeros((8, 8), dtype=np.float32), name="W"
        )
        mm   = helper.make_node("MatMul", inputs=["X", "W"], outputs=["Z"])
        relu = helper.make_node("Relu",   inputs=["Z"],      outputs=["R"])
        add  = helper.make_node("Add",    inputs=["X", "R"], outputs=["Y"])
        graph = helper.make_graph(
            [mm, relu, add], "skip_zero",
            inputs=[helper.make_tensor_value_info("X", TensorProto.FLOAT, [4, 8])],
            outputs=[helper.make_tensor_value_info("Y", TensorProto.FLOAT, [4, 8])],
            initializer=[w_zero],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

        import tempfile
        import os
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            tmp = f.name
        try:
            gen = CodeGenerator(OnnxGraph(tmp), model_path=tmp)
            x_in = np.ones((4, 8), dtype=np.float64) * 0.5
            result = gen.simulate({"X": x_in})
            np.testing.assert_array_equal(result["Y"], x_in)
        finally:
            os.unlink(tmp)


@unittest.skipUnless(_new_mixed_models_exist(),
                     "Run test/gen_mixed_kernel_models.py first")
class TestBatchMatmulRelu(unittest.TestCase):
    """mixed_batch_matmul_relu: A[2,4,6] @ B[6,4] → Z[2,4,4], Relu(Z) → Y[2,4,4].
    B is 2-D (broadcasts across A's batch). outer_count=1, batch=2.
    Z.alloc=32 (flat), Y.alloc=32 (flat).
    """

    def setUp(self):
        self.gen = _gen("mixed_batch_matmul_relu.onnx")

    def test_z_alloc(self):
        """Z[2,4,4] numel=32, no broadcast VectorOP → alloc=32 (flat)."""
        self.assertEqual(self.gen._alloc_sizes["Z"], 32)

    def test_y_alloc(self):
        """Y[2,4,4] numel=32 → alloc=32 (flat)."""
        self.assertEqual(self.gen._alloc_sizes["Y"], 32)

    def test_z_layout_flat(self):
        """Z: flat (n_chunks=1, gap=0)."""
        lay = self.gen._layouts["Z"]
        self.assertEqual(lay.n_chunks, 1)
        self.assertEqual(lay.gap, 0)

    def test_y_not_in_bcast_map(self):
        """Y is MatmulNode output — not in bcast_map (Fix A)."""
        bmap = self.gen._broadcast_io_map()
        self.assertNotIn("Y", bmap)

    def test_no_outer_loop(self):
        """outer_count=1 → no run_matmul_at(), uses plain run_matmul()."""
        src = self.gen.generate_source()
        self.assertNotIn("run_matmul_at(", src)

    def test_run_matmul_batched(self):
        """Natural batched form: run_matmul(A, B, Z, 4u, 6u, 4u, 2u, 24u, 0u, 16u)."""
        src = self.gen.generate_source()
        self.assertIn("4u, 6u, 4u, 2u", src)
        self.assertIn("24u, 0u, 16u", src)

    def test_relu_on_full_z(self):
        """Relu: run_op(Z, NULL, Y, 32u, VECTOROP_RELU)."""
        src = self.gen.generate_source()
        self.assertIn("run_op(", src)
        self.assertIn("32u", src)
        self.assertIn("VECTOROP_RELU", src)


@unittest.skipUnless(_new_mixed_models_exist(),
                     "Run test/gen_mixed_kernel_models.py first")
class TestSubDivMatmul(unittest.TestCase):
    """mixed_sub_div_matmul: Sub(X[4,6],offset[6])→S, Div(S,scale[6])→D, MatMul(D,W[6,4])→Y.
    Sub and Div both broadcast (outer=4, chunk=6, aligned=8) → gap=2.
    Y: flat(16) — MatmulNode output (Fix A).
    MatMul: a_row_stride=8 (D gap=2), c_row_stride=0 (Y flat).
    """

    def setUp(self):
        self.gen = _gen("mixed_sub_div_matmul.onnx")

    def test_s_alloc(self):
        """S[4,6] numel=24, outer=4, aligned=8 → alloc=32 (gap=2)."""
        self.assertEqual(self.gen._alloc_sizes["S"], 32)

    def test_d_alloc(self):
        """D[4,6] numel=24, outer=4, aligned=8 → alloc=32 (gap=2)."""
        self.assertEqual(self.gen._alloc_sizes["D"], 32)

    def test_y_alloc(self):
        """Y is MatmulNode output (Fix A) — alloc=16 (flat, numel=16)."""
        self.assertEqual(self.gen._alloc_sizes["Y"], 16)

    def test_d_layout(self):
        """D: n_chunks=4, chunk=6, stride=8, gap=2."""
        lay = self.gen._layouts["D"]
        self.assertEqual(lay.n_chunks, 4)
        self.assertEqual(lay.chunk, 6)
        self.assertEqual(lay.stride, 8)
        self.assertEqual(lay.gap, 2)

    def test_y_not_in_bcast_map(self):
        """Y is MatmulNode output — Fix A: not in bcast_map."""
        bmap = self.gen._broadcast_io_map()
        self.assertNotIn("Y", bmap)

    def test_s_in_bcast_map(self):
        """S is advancing input of Sub broadcast → in bcast_map."""
        bmap = self.gen._broadcast_io_map()
        self.assertIn("S", bmap)

    def test_matmul_strides(self):
        """a_row_stride=8 (D gap=2), c_row_stride=0 (Y flat)."""
        strides = _matmul_strides(self.gen)
        self.assertIn("Y", strides)
        a_row, c_row = strides["Y"]
        self.assertEqual(a_row, 8)
        self.assertEqual(c_row, 0)

    def test_run_matmul_call(self):
        """Strided form: run_matmul(D, W, Y, 1u, 6u, 4u, 4u, 8u, 0u, 4u)."""
        src = self.gen.generate_source()
        self.assertIn("1u, 6u, 4u, 4u", src)
        self.assertIn("8u, 0u, 4u", src)


def _multi_io_models_exist() -> bool:
    return os.path.isfile(_mixed_model("mixed_two_input_two_output.onnx"))


@unittest.skipUnless(_multi_io_models_exist(),
                     "Run test/gen_mixed_kernel_models.py first")
class TestTwoInputMatmul(unittest.TestCase):
    """mixed_two_input_matmul: Add(X1[4,8], X2[4,8])→Z[4,8] (same-shape, non-broadcast),
    MatMul(Z[4,8], W[8,4])→Y[4,4].
    Two graph inputs (X1, X2), one output (Y).  Z and Y are flat.
    """

    def setUp(self):
        self.gen = _gen("mixed_two_input_matmul.onnx")

    def test_x1_alloc(self):
        """X1[4,8] numel=32, flat → alloc=32."""
        self.assertEqual(self.gen._alloc_sizes["X1"], 32)

    def test_x2_alloc(self):
        """X2[4,8] numel=32, flat → alloc=32."""
        self.assertEqual(self.gen._alloc_sizes["X2"], 32)

    def test_z_alloc(self):
        """Z[4,8] non-broadcast Add output → alloc=32 (flat, numel=32)."""
        self.assertEqual(self.gen._alloc_sizes["Z"], 32)

    def test_y_alloc(self):
        """Y[4,4] MatmulNode output, flat → alloc=16."""
        self.assertEqual(self.gen._alloc_sizes["Y"], 16)

    def test_z_layout_flat(self):
        """Z: flat (n_chunks=1, gap=0)."""
        lay = self.gen._layouts["Z"]
        self.assertEqual(lay.n_chunks, 1)
        self.assertEqual(lay.gap, 0)

    def test_y_layout_flat(self):
        """Y: flat (n_chunks=1, gap=0)."""
        lay = self.gen._layouts["Y"]
        self.assertEqual(lay.n_chunks, 1)
        self.assertEqual(lay.gap, 0)

    def test_no_broadcast_chunk_macros(self):
        """No broadcast VectorOP nodes → bcast_map is empty."""
        bmap = self.gen._broadcast_io_map()
        self.assertEqual(bmap, {})

    def test_no_chunk_macros_in_header(self):
        """No CHUNK macros should appear (no broadcast nodes)."""
        hdr = self.gen.generate_header()
        self.assertNotIn("INFERENCE_Z_CHUNK", hdr)
        self.assertNotIn("INFERENCE_Y_CHUNK", hdr)

    def test_natural_matmul_call(self):
        """Natural form: run_matmul(Z, W, Y, 4u, 8u, 4u, 1u, 0u, 0u, 0u)."""
        src = self.gen.generate_source()
        self.assertIn("4u, 8u, 4u, 1u", src)
        self.assertIn("0u, 0u, 0u", src)

    def test_two_inputs_in_run_signature(self):
        """inference_run() must accept both X1 and X2."""
        src = self.gen.generate_source()
        self.assertIn("inference_buf_t *X1", src)
        self.assertIn("inference_buf_t *X2", src)

    def test_sync_to_device_for_both_inputs(self):
        """Both inputs are flushed before the first kernel call."""
        src = self.gen.generate_source()
        self.assertIn("sync_to_device(X1)", src)
        self.assertIn("sync_to_device(X2)", src)

    def test_one_sync_from_device(self):
        """Only Y is invalidated after the kernel."""
        src = self.gen.generate_source()
        self.assertIn("sync_from_device(Y)", src)
        self.assertNotIn("sync_from_device(X1)", src)
        self.assertNotIn("sync_from_device(X2)", src)

    def test_header_inference_run_two_inputs(self):
        """Header declaration lists X1 and X2 as parameters."""
        hdr = self.gen.generate_header()
        self.assertIn("inference_buf_t *X1", hdr)
        self.assertIn("inference_buf_t *X2", hdr)


@unittest.skipUnless(_multi_io_models_exist(),
                     "Run test/gen_mixed_kernel_models.py first")
class TestTwoOutput(unittest.TestCase):
    """mixed_two_output: X[4,8]→MatMul→Z[4,4], Add(Z,bias[4])→Yadd[4,4] (broadcast),
    Relu(Z)→Yrelu[4,4] (phase-3 propagation from Z).
    One input (X), two outputs (Yadd, Yrelu).
    Z.alloc=32 (advancing, gap=4).  Yrelu inherits Z's advancing layout.
    """

    def setUp(self):
        self.gen = _gen("mixed_two_output.onnx")

    def test_z_alloc(self):
        """Z[4,4] numel=16, advancing input of broadcast Add → alloc=32."""
        self.assertEqual(self.gen._alloc_sizes["Z"], 32)

    def test_z_layout(self):
        """Z: n_chunks=4, chunk=4, stride=8, gap=4."""
        lay = self.gen._layouts["Z"]
        self.assertEqual(lay.n_chunks, 4)
        self.assertEqual(lay.chunk, 4)
        self.assertEqual(lay.stride, 8)
        self.assertEqual(lay.gap, 4)

    def test_yadd_alloc(self):
        """Yadd[4,4] output of broadcast Add → alloc=32."""
        self.assertEqual(self.gen._alloc_sizes["Yadd"], 32)

    def test_yrelu_alloc(self):
        """Yrelu[4,4] inherits Z's advancing layout via phase 3 → alloc=32."""
        self.assertEqual(self.gen._alloc_sizes["Yrelu"], 32)

    def test_yrelu_layout(self):
        """Yrelu: n_chunks=4, stride=8 (propagated from Z through Relu)."""
        lay = self.gen._layouts["Yrelu"]
        self.assertEqual(lay.n_chunks, 4)
        self.assertEqual(lay.stride, 8)

    def test_z_in_bcast_map(self):
        """Z is the advancing input of broadcast Add → in bcast_map."""
        bmap = self.gen._broadcast_io_map()
        self.assertIn("Z", bmap)

    def test_yadd_in_bcast_map(self):
        """Yadd is the direct output of broadcast Add → in bcast_map."""
        bmap = self.gen._broadcast_io_map()
        self.assertIn("Yadd", bmap)

    def test_yrelu_in_bcast_map(self):
        """Yrelu inherits Z's canonical prefix via phase-2 propagation → in bcast_map."""
        bmap = self.gen._broadcast_io_map()
        self.assertIn("Yrelu", bmap)

    def test_yrelu_uses_yadd_macros(self):
        """Yrelu's bcast_map entry references INFERENCE_YADD_CHUNK macros."""
        bmap = self.gen._broadcast_io_map()
        _, chunk_macro, stride_macro = bmap["Yrelu"]
        self.assertEqual(chunk_macro,  "INFERENCE_YADD_CHUNK")
        self.assertEqual(stride_macro, "INFERENCE_YADD_CHUNK_STRIDE")

    def test_matmul_c_row_stride(self):
        """Z has gap=4 → c_row_stride=8; X is flat → a_row_stride=0."""
        strides = _matmul_strides(self.gen)
        self.assertIn("Z", strides)
        a_row, c_row = strides["Z"]
        self.assertEqual(a_row, 0)
        self.assertEqual(c_row, 8)

    def test_run_matmul_strided_call(self):
        """Strided form: run_matmul(X, W, Z, 1u, 8u, 4u, 4u, 8u, 0u, 8u)."""
        src = self.gen.generate_source()
        self.assertIn("1u, 8u, 4u, 4u", src)
        self.assertIn("8u, 0u, 8u", src)

    def test_two_outputs_in_run_signature(self):
        """inference_run() must accept Yadd and Yrelu."""
        src = self.gen.generate_source()
        self.assertIn("inference_buf_t *Yadd", src)
        self.assertIn("inference_buf_t *Yrelu", src)

    def test_sync_from_device_for_both_outputs(self):
        """Both outputs are invalidated after all kernel ops."""
        src = self.gen.generate_source()
        self.assertIn("sync_from_device(Yadd)", src)
        self.assertIn("sync_from_device(Yrelu)", src)

    def test_one_sync_to_device(self):
        """X is flushed and the two outputs are cleaned (no CPU-dirty line may
        be evicted over a kernel's result) once, before the first kernel
        call; nothing is flushed between kernels."""
        src = self.gen.generate_source()
        body = src[src.index("void inference_run("):]
        first_run = min(body.index(k) for k in ("run_op(", "run_op_act(", "run_matmul(")
                        if k in body)
        for b in ("X", "Yadd", "Yrelu"):
            self.assertEqual(body.count(f"sync_to_device({b})"), 1, b)
            self.assertLess(body.index(f"sync_to_device({b})"), first_run, b)
        self.assertEqual(body.count("sync_to_device("), 3)

    def test_header_yrelu_size_uses_yadd_stride_macro(self):
        """INFERENCE_YRELU_SIZE must reference INFERENCE_YADD_CHUNK_STRIDE."""
        hdr = self.gen.generate_header()
        self.assertIn("INFERENCE_YRELU_SIZE", hdr)
        self.assertIn("INFERENCE_YADD_CHUNK_STRIDE", hdr)
        self.assertIn("4u * INFERENCE_YADD_CHUNK_STRIDE", hdr)

    def test_header_two_outputs_in_run_declaration(self):
        """Header inference_run() declaration lists Yadd and Yrelu."""
        hdr = self.gen.generate_header()
        self.assertIn("inference_buf_t *Yadd", hdr)
        self.assertIn("inference_buf_t *Yrelu", hdr)


@unittest.skipUnless(_multi_io_models_exist(),
                     "Run test/gen_mixed_kernel_models.py first")
class TestTwoInputTwoOutput(unittest.TestCase):
    """mixed_two_input_two_output:
    X1[4,8] → MatMul(X1, W[8,4]) → Z[4,4],
    Add(Z, X2[4]) → Yadd[4,4]  (broadcast, outer=4, chunk=4; X2 is a graph input),
    Relu(Z)       → Yrelu[4,4] (phase-3 propagation from Z).
    Two inputs (X1, X2), two outputs (Yadd, Yrelu).
    Z.alloc=32 (advancing, gap=4).  X2.alloc=8 (repeating, n_chunks=1).
    INFERENCE_X2_SIZE = 4u (numel, not alloc=8).
    """

    def setUp(self):
        self.gen = _gen("mixed_two_input_two_output.onnx")

    def test_z_alloc(self):
        """Z[4,4] advancing input of broadcast Add → alloc=32."""
        self.assertEqual(self.gen._alloc_sizes["Z"], 32)

    def test_z_layout(self):
        """Z: n_chunks=4, stride=8, gap=4."""
        lay = self.gen._layouts["Z"]
        self.assertEqual(lay.n_chunks, 4)
        self.assertEqual(lay.stride, 8)
        self.assertEqual(lay.gap, 4)

    def test_x2_alloc(self):
        """X2[4] repeating graph input (co-input to broadcast Add) → alloc=8."""
        self.assertEqual(self.gen._alloc_sizes["X2"], 8)

    def test_x2_layout(self):
        """X2: n_chunks=1 (repeating), alloc=8, chunk=4, stride=8."""
        lay = self.gen._layouts["X2"]
        self.assertEqual(lay.n_chunks, 1)
        self.assertEqual(lay.alloc, 8)
        self.assertEqual(lay.chunk, 4)
        self.assertEqual(lay.stride, 8)

    def test_yadd_alloc(self):
        """Yadd[4,4] output of broadcast Add → alloc=32."""
        self.assertEqual(self.gen._alloc_sizes["Yadd"], 32)

    def test_yrelu_alloc(self):
        """Yrelu[4,4] inherits Z's advancing layout → alloc=32."""
        self.assertEqual(self.gen._alloc_sizes["Yrelu"], 32)

    def test_x2_not_in_bcast_map(self):
        """X2 has n_chunks=1 (repeating) → excluded from bcast_map result."""
        bmap = self.gen._broadcast_io_map()
        self.assertNotIn("X2", bmap)

    def test_z_in_bcast_map(self):
        """Z is the advancing input of broadcast Add → in bcast_map."""
        bmap = self.gen._broadcast_io_map()
        self.assertIn("Z", bmap)

    def test_yrelu_in_bcast_map(self):
        """Yrelu inherits Z's canonical prefix → in bcast_map."""
        bmap = self.gen._broadcast_io_map()
        self.assertIn("Yrelu", bmap)

    def test_header_x2_size_is_numel(self):
        """INFERENCE_X2_SIZE must be t.numel=4 (not alloc=8)."""
        hdr = self.gen.generate_header()
        self.assertIn("INFERENCE_X2_SIZE", hdr)
        self.assertIn("4u", hdr)
        self.assertNotIn("8u * INFERENCE", hdr)

    def test_matmul_c_row_stride(self):
        """Z has gap=4 → c_row_stride=8; X1 is flat → a_row_stride=0."""
        strides = _matmul_strides(self.gen)
        self.assertIn("Z", strides)
        a_row, c_row = strides["Z"]
        self.assertEqual(a_row, 0)
        self.assertEqual(c_row, 8)

    def test_run_matmul_strided_call(self):
        """Strided form: run_matmul(X1, W, Z, 1u, 8u, 4u, 4u, 8u, 0u, 8u)."""
        src = self.gen.generate_source()
        self.assertIn("1u, 8u, 4u, 4u", src)
        self.assertIn("8u, 0u, 8u", src)

    def test_two_inputs_in_run_signature(self):
        """inference_run() must accept X1 and X2."""
        src = self.gen.generate_source()
        self.assertIn("inference_buf_t *X1", src)
        self.assertIn("inference_buf_t *X2", src)

    def test_two_outputs_in_run_signature(self):
        """inference_run() must accept Yadd and Yrelu."""
        src = self.gen.generate_source()
        self.assertIn("inference_buf_t *Yadd", src)
        self.assertIn("inference_buf_t *Yrelu", src)

    def test_sync_to_device_for_both_inputs(self):
        """Both X1 and X2 are flushed before the first kernel call."""
        src = self.gen.generate_source()
        self.assertIn("sync_to_device(X1)", src)
        self.assertIn("sync_to_device(X2)", src)

    def test_sync_from_device_for_both_outputs(self):
        """Both Yadd and Yrelu are invalidated after all kernel ops."""
        src = self.gen.generate_source()
        self.assertIn("sync_from_device(Yadd)", src)
        self.assertIn("sync_from_device(Yrelu)", src)

    def test_header_two_inputs_in_run_declaration(self):
        """Header inference_run() declaration lists X1 and X2."""
        hdr = self.gen.generate_header()
        self.assertIn("inference_buf_t *X1", hdr)
        self.assertIn("inference_buf_t *X2", hdr)

    def test_header_two_outputs_in_run_declaration(self):
        """Header inference_run() declaration lists Yadd and Yrelu."""
        hdr = self.gen.generate_header()
        self.assertIn("inference_buf_t *Yadd", hdr)
        self.assertIn("inference_buf_t *Yrelu", hdr)


@unittest.skipUnless(_mixed_models_exist(),
                     "Run test/gen_mixed_kernel_models.py first")
class TestSpatialMatmulRelu(unittest.TestCase):
    """X[1,3,224,224] @ W[224,10] → Z[1,3,224,10] → Relu → Y[1,3,224,10].

    4-D × 2-D batched MatMul: W broadcasts across all leading dims of X.
    The last two dims of X are the matrix dims [n=224, k=224]; the leading
    dims [1, 3] are the batch → batch = 1*3 = 3.
    MatmulKernel: n=224, k=224, m=10, batch=3,
                  a_batch_stride=50176, b_batch_stride=0, c_batch_stride=2240.
    VectorOPKernel: Relu on Z (6720 elements), flat layout.
    """

    @classmethod
    def setUpClass(cls):
        cls.path = _mixed_model("mixed_spatial_matmul_relu.onnx")
        cls.graph = OnnxGraph(cls.path)
        cls.gen   = CodeGenerator(cls.graph, model_path=cls.path)

    # ---- Graph structure ----

    def test_two_nodes(self):
        from src.nodes import MatmulNode, ScheduledNode
        self.assertEqual(len(self.graph.nodes), 2)
        self.assertIsInstance(self.graph.nodes[0], MatmulNode)
        self.assertIsInstance(self.graph.nodes[1], ScheduledNode)

    def test_input_shape(self):
        inputs = self.graph.input_tensors
        self.assertEqual(len(inputs), 1)
        self.assertEqual(inputs[0].shape, [1, 3, 224, 224])

    def test_output_shape(self):
        outputs = self.graph.output_tensors
        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0].shape, [1, 3, 224, 10])

    # ---- MatmulNode geometry ----

    def test_matmul_dims(self):
        from src.nodes import MatmulNode
        mm = self.graph.nodes[0]
        self.assertIsInstance(mm, MatmulNode)
        self.assertEqual(mm.n, 224)
        self.assertEqual(mm.k, 224)
        self.assertEqual(mm.m,  10)

    def test_matmul_batch(self):
        mm = self.graph.nodes[0]
        # a_batch = A.shape[:-2] = [1, 3]; batch = 1 * 3 = 3
        # The last two dims [224, 224] are the n×k matrix dims, not batch.
        self.assertEqual(mm.batch, 3)

    def test_matmul_strides(self):
        mm = self.graph.nodes[0]
        self.assertEqual(mm.a_batch_stride, 224 * 224)   # 50176
        self.assertEqual(mm.b_batch_stride, 0)            # W broadcasts
        self.assertEqual(mm.c_batch_stride, 224 * 10)    # 2240

    def test_no_outer_loop(self):
        mm = self.graph.nodes[0]
        self.assertEqual(mm.outer_count, 1)

    # ---- TensorLayouts ----

    def test_z_flat(self):
        z_lay = self.gen._layouts["Z"]
        self.assertEqual(z_lay.n_chunks, 1)
        self.assertEqual(z_lay.numel,   1 * 3 * 224 * 10)  # 6720
        self.assertEqual(z_lay.alloc,   6720)

    def test_y_flat(self):
        y_lay = self.gen._layouts["Y"]
        self.assertEqual(y_lay.n_chunks, 1)
        self.assertEqual(y_lay.numel,   6720)
        self.assertEqual(y_lay.alloc,   6720)

    def test_w_flat(self):
        # W[224,10] is a constant MatMul B: the scheduler emits it in
        # MatmulKernel's tile-major packed layout, m padded 10 -> 16
        # (MATMUL_OPTIMISATION §3b), so the DMA buffer is 224 * 16 elements.
        from src._matmul_hw_config import MATMUL_TILE_M
        packed_m = -(-10 // MATMUL_TILE_M) * MATMUL_TILE_M
        w_lay = self.gen._layouts["W"]
        self.assertEqual(w_lay.n_chunks, 1)
        self.assertEqual(w_lay.numel,   224 * packed_m)  # packed to the kernel's tile width
        self.assertEqual(w_lay.alloc,   224 * packed_m)

    # ---- Generated source ----

    def _src(self):
        return self.gen.generate_source()

    def test_run_matmul_call_correct_dims(self):
        s = self._src()
        # n=224, k=224, m=10, batch=3 (leading dims [1,3] of A)
        self.assertIn("224u, 224u, 10u, 3u,", s)

    def test_run_matmul_strides(self):
        s = self._src()
        # a_batch_stride=50176, b_batch_stride=0, c_batch_stride=2240
        self.assertIn("50176u, 0u, 2240u", s)

    def test_relu_call_correct_size(self):
        s = self._src()
        # run_op(Z, NULL, Y, 6720u, VECTOROP_RELU)
        self.assertIn("6720u, VECTOROP_RELU", s)

    def test_both_kernels_active(self):
        names = [kd.name for kd in self.gen._active_kernels]
        self.assertIn("VectorOPKernel", names)
        self.assertIn("MatmulKernel",   names)

    # ---- Simulation ----

    def test_simulate_output_shape(self):
        arrays = self.gen._simulate()
        self.assertEqual(list(arrays["Y"].shape), [1, 3, 224, 10])

    def test_simulate_output_nonneg(self):
        arrays = self.gen._simulate()
        self.assertTrue((arrays["Y"] >= 0).all(), "Relu output must be non-negative")

    def test_intermediate_z_shape(self):
        arrays = self.gen._simulate()
        self.assertEqual(list(arrays["Z"].shape), [1, 3, 224, 10])


if __name__ == "__main__":
    unittest.main(verbosity=2)
