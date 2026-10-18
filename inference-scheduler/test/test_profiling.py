# Copyright 2025 GradeBuilder SL
# SPDX-License-Identifier: Apache-2.0

"""Tests for the per-layer profiling additions.

Covers:
  - inference.c includes inference_prof.h.
  - The static layer-name table and accessors are emitted with one entry
    per scheduled node, in graph order.
  - Each kernel-backed node call is wrapped with INFERENCE_PROF_BEGIN /
    INFERENCE_PROF_END using the node's index.
  - inference.h exposes INFERENCE_NUM_LAYERS, inference_num_layers(),
    and inference_layer_names_ptr().
  - The CMake gate (option + PUBLIC compile def) is emitted.
  - _layer_display_names() prefers the ONNX node name when present and
    resolves collisions by suffixing with the node index.
"""

import os
import re
import sys
import tempfile
import unittest

import onnx
from onnx import helper as oh, TensorProto

from helpers    import _model, _models_exist
from src.graph   import OnnxGraph
from src.codegen import CodeGenerator
from src.nodes   import ReshapeNode


def _build_named_model(node_names):
    """Return an in-memory ONNX model with a Relu chain whose nodes carry
    the supplied .name strings.  Used for collision / fallback tests."""
    inp = oh.make_tensor_value_info("X", TensorProto.FLOAT, [1, 8])
    cur = "X"
    nodes = []
    for i, name in enumerate(node_names):
        out = f"t{i}" if i + 1 < len(node_names) else "Y"
        nodes.append(oh.make_node("Relu", inputs=[cur], outputs=[out],
                                  name=name))
        cur = out
    out_v = oh.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 8])
    graph = oh.make_graph(nodes, "namedmodel", [inp], [out_v])
    model = oh.make_model(graph,
                           opset_imports=[oh.make_opsetid("", 13)],
                           ir_version=7)
    return model


@unittest.skipUnless(_models_exist(), "Run test/gen_test_models.py first")
class TestProfilingEmissions(unittest.TestCase):

    def _gen(self, model_name):
        path = _model(model_name)
        g    = OnnxGraph(path)
        cg   = CodeGenerator(g, model_path=path)
        return cg

    # -- inference.c ---------------------------------------------------- #

    def test_source_includes_prof_header(self):
        s = self._gen("single_add.onnx").generate_source()
        self.assertIn('#include "inference_prof.h"', s)

    def test_source_emits_layer_name_table(self):
        s = self._gen("single_add.onnx").generate_source()
        self.assertIn("static const char *const inference_layer_names[1]", s)
        self.assertIn('"Add_0"', s)

    def test_source_emits_accessors(self):
        s = self._gen("single_add.onnx").generate_source()
        self.assertIn("unsigned inference_num_layers(void)", s)
        self.assertIn("const char *const *inference_layer_names_ptr(void)", s)
        self.assertIn("return 1u;", s)

    def test_kernel_call_wrapped_with_prof_macros(self):
        s = self._gen("single_add.onnx").generate_source()
        # PROF_BEGIN(0) precedes the (now non-blocking) Start; PROF_END(0)
        # follows the matching kernel_wait that drains lane 0.  The drain
        # may include a comment line and the wait call between Start and END.
        m = re.search(
            r"INFERENCE_PROF_BEGIN\(0u\);\s*"
            r"run_op\([^)]*\);"
            r"(?:\s*/\*[^*]*\*/)?"             # optional drain comment
            r"\s*kernel_wait\(KERNEL_\w+\);\s*"
            r"INFERENCE_PROF_END\(0u\);",
            s,
        )
        self.assertIsNotNone(m, f"profile wrap not found:\n{s}")

    def test_each_node_gets_distinct_index(self):
        cg = self._gen("relu_chain.onnx")
        s  = cg.generate_source()
        kernel_nodes = [
            sn for sn in cg._graph.nodes if not isinstance(sn, ReshapeNode)
        ]
        for sn in kernel_nodes:
            self.assertIn(f"INFERENCE_PROF_BEGIN({sn.index}u);", s)
            self.assertIn(f"INFERENCE_PROF_END({sn.index}u);",   s)

    # -- inference.h ---------------------------------------------------- #

    def test_header_exposes_num_layers_macro(self):
        h = self._gen("single_add.onnx").generate_header()
        self.assertIn("#define INFERENCE_NUM_LAYERS  1u", h)

    def test_header_declares_accessors(self):
        h = self._gen("single_add.onnx").generate_header()
        self.assertIn("inference_num_layers(void);",        h)
        self.assertIn("inference_layer_names_ptr(void);",   h)

    # -- CMakeLists.txt ------------------------------------------------- #

    def test_cmake_emits_profiling_option(self):
        c = self._gen("single_add.onnx").generate_cmake()
        self.assertIn("option(INFERENCE_PROFILING", c)
        self.assertIn(
            "target_compile_definitions(inference PUBLIC INFERENCE_PROFILING=1)",
            c,
        )
        self.assertIn("src/inference_prof.c", c)


class TestLayerDisplayNames(unittest.TestCase):
    """In-memory tests that don't require the gen_test_models.py output."""

    def _gen_from_model(self, model):
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            path = f.name
        try:
            g  = OnnxGraph(path)
            cg = CodeGenerator(g, model_path=path)
            return cg._layer_display_names()
        finally:
            os.unlink(path)

    def test_uses_onnx_name_when_present(self):
        names = self._gen_from_model(_build_named_model(["alpha", "beta"]))
        self.assertEqual(names, ["alpha", "beta"])

    def test_falls_back_to_op_type_index_when_empty(self):
        names = self._gen_from_model(_build_named_model(["", ""]))
        # Both nodes are Relu; fallback yields op-type-with-index, which
        # is already unique without further suffixing.
        self.assertEqual(names, ["Relu_0", "Relu_1"])

    def test_resolves_collisions_with_index_suffix(self):
        names = self._gen_from_model(_build_named_model(["dup", "dup"]))
        self.assertEqual(names, ["dup_0", "dup_1"])

    def test_mixes_named_and_unnamed(self):
        names = self._gen_from_model(_build_named_model(["", "tail"]))
        self.assertEqual(names, ["Relu_0", "tail"])


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(__file__))
    unittest.main()
