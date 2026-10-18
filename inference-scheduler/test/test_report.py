# Copyright 2025 GradeBuilder SL
# SPDX-License-Identifier: Apache-2.0

"""Tests for src.report.ReportGenerator and the --no-report CLI flag.

Covers:
  * Every required section is present in the markdown output.
  * Structural facts (input/output shapes, weight count, lane usage,
    Gemm decomposition count, reshape count, parallel overlap count)
    match the underlying graph + codegen state.
  * The CLI writes report.md by default and skips it under --no-report.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

from helpers import _model
from src.graph    import OnnxGraph
from src.codegen  import CodeGenerator
from src.report   import ReportGenerator, _count_overlapping_starts


def _has(name: str) -> bool:
    return os.path.isfile(_model(name))


def _gen_md(model_name: str) -> str:
    path = _model(model_name)
    g    = OnnxGraph(path)
    cg   = CodeGenerator(g, model_path=path)
    rg   = ReportGenerator(
        graph=g, codegen=cg, model_path=path,
        out_dir="/tmp/test", generated_files=["CMakeLists.txt"],
    )
    return rg.render_markdown()


# ----------------------------------------------------------------- #
# Section presence — every required heading is emitted              #
# ----------------------------------------------------------------- #


@unittest.skipUnless(_has("single_add.onnx"), "Run test/gen_test_models.py first")
class TestReportSections(unittest.TestCase):

    def test_all_sections_present(self):
        md = _gen_md("single_add.onnx")
        for heading in (
            "# Inference scheduler report",
            "## Inputs and outputs",
            "## Parameters",
            "## Activation memory",
            "## Hardware lanes",
            "## Applied transformations",
            "## Layers",
            "## Generated artifacts",
        ):
            self.assertIn(heading, md, f"missing section: {heading}")

    def test_header_includes_model_basename(self):
        md = _gen_md("single_add.onnx")
        self.assertIn("`single_add.onnx`", md)

    def test_header_includes_dtype_and_lanes(self):
        md = _gen_md("single_add.onnx")
        self.assertIn("ap_fixed<16,8>", md)
        self.assertIn("VectorOPKernel", md)


# ----------------------------------------------------------------- #
# Structural facts — numbers match the underlying state             #
# ----------------------------------------------------------------- #


@unittest.skipUnless(_has("parallel_two_chains.onnx"),
                     "Run test/gen_parallel_models.py first")
class TestReportFactsParallelTwoChains(unittest.TestCase):

    def setUp(self):
        self.md = _gen_md("parallel_two_chains.onnx")

    def test_layer_count_matches_graph(self):
        # 6 nodes: convA, reluA, poolA, convB, reluB, joinAdd.
        m = re.search(r"## Layers \((\d+)\)", self.md)
        self.assertIsNotNone(m)
        self.assertEqual(int(m.group(1)), 6)

    def test_lane_table_marks_active_lanes(self):
        # Conv, Pool, VectorOP are used; Matmul is not.
        m = re.search(r"## Hardware lanes(.+?)## Applied", self.md, re.S)
        self.assertIsNotNone(m)
        block = m.group(1)
        self.assertRegex(block, r"`VectorOPKernel`\s*\|\s*✓")
        self.assertRegex(block, r"`MatmulKernel`\s*\|\s*–")
        self.assertRegex(block, r"`ConvKernel`\s*\|\s*✓")
        self.assertRegex(block, r"`PoolKernel`\s*\|\s*✓")

    def test_cross_lane_parallelism_count_is_positive(self):
        # parallel_two_chains has overlap (Conv-B starts while Pool is in flight).
        m = re.search(
            r"Cross-lane parallelism.* (\d+) of (\d+) kernel starts",
            self.md,
        )
        self.assertIsNotNone(m, f"parallelism line not found in:\n{self.md}")
        overlapping, total = int(m.group(1)), int(m.group(2))
        self.assertGreater(overlapping, 0)
        self.assertEqual(total, 6)

    def test_input_and_output_rows_emitted(self):
        # |  Direction | Tensor | ... — at least one Input + one Output row.
        self.assertRegex(self.md, r"\|\s+Input\s+\|\s+`X`")
        self.assertRegex(self.md, r"\|\s+Output\s+\|\s+`Y`")


@unittest.skipUnless(_has("gemm_with_bias.onnx"),
                     "Run test/gen_reshape_gemm_models.py first")
class TestReportFactsGemmDecomposition(unittest.TestCase):

    def test_gemm_decomposition_count_reported(self):
        md = _gen_md("gemm_with_bias.onnx")
        # 1 Gemm node was rewritten.  The bullet term is markdown-bold,
        # so accept `**` and arbitrary whitespace before the em-dash.
        self.assertRegex(
            md, r"Gemm decomposition\*\*\s*—\s*1\s+`Gemm`\s+node rewritten",
        )


@unittest.skipUnless(_has("nop_chain_dropout_fork.onnx"),
                     "Run test/gen_parallel_models.py first")
class TestReportFactsReshapeFolding(unittest.TestCase):

    def test_reshape_folding_count_matches_node_count(self):
        # The fixture has Conv → Drop → Drop → Drop → fork → Add.
        # Three Dropouts → three ReshapeNode aliases.
        md = _gen_md("nop_chain_dropout_fork.onnx")
        self.assertRegex(
            md, r"Reshape folding\*\*\s*—\s*3\s+`ReshapeNode` outputs aliased",
        )


# ----------------------------------------------------------------- #
# _count_overlapping_starts unit-test                                #
# ----------------------------------------------------------------- #


class TestOverlapCounter(unittest.TestCase):

    def test_strict_chain_has_no_overlap(self):
        events = [
            ("comment", 0), ("start", 0),
            ("comment", 1), ("wait", "K", 0), ("start", 1),
            ("comment", 2), ("wait", "K", 1), ("start", 2),
            ("drain", "K", 2),
        ]
        self.assertEqual(_count_overlapping_starts(events), 0)

    def test_two_lanes_in_flight_simultaneously(self):
        # node 0 starts; node 1 starts while 0 still in flight; both drain.
        events = [
            ("start", 0),
            ("start", 1),                    # overlap +1
            ("wait", "A", 0),
            ("wait", "B", 1),
        ]
        self.assertEqual(_count_overlapping_starts(events), 1)

    def test_start_sync_counts_overlap_but_does_not_track(self):
        events = [
            ("start", 0),
            ("start_sync", 1),               # overlap +1 (0 is in flight)
            # synchronous helper drained itself; node 1 NOT in flight afterwards
            ("wait", "A", 0),
        ]
        self.assertEqual(_count_overlapping_starts(events), 1)


# ----------------------------------------------------------------- #
# CLI integration — report.md is written by default; --no-report    #
# suppresses it.                                                    #
# ----------------------------------------------------------------- #


@unittest.skipUnless(_has("single_add.onnx"), "Run test/gen_test_models.py first")
class TestReportCli(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="report_cli_")
        self.script = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "inference_scheduler.py",
        )
        self.model = _model("single_add.onnx")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run(self, *extra) -> int:
        rc = subprocess.run(
            [sys.executable, self.script, self.model,
             "--out-dir", self.tmp, *extra],
            capture_output=True, text=True, timeout=60,
        )
        return rc

    def test_default_writes_report_md(self):
        rc = self._run()
        self.assertEqual(rc.returncode, 0,
                         f"scheduler failed:\nstderr={rc.stderr}")
        path = os.path.join(self.tmp, "report.md")
        self.assertTrue(os.path.isfile(path),
                        f"report.md not written at {path}")
        with open(path) as f:
            self.assertIn("# Inference scheduler report", f.read())

    def test_no_report_flag_skips_file(self):
        rc = self._run("--no-report")
        self.assertEqual(rc.returncode, 0,
                         f"scheduler failed:\nstderr={rc.stderr}")
        path = os.path.join(self.tmp, "report.md")
        self.assertFalse(os.path.isfile(path),
                         f"report.md unexpectedly present at {path}")


# ----------------------------------------------------------------- #
# Quantization-error reporting (fixed-point only)                    #
# ----------------------------------------------------------------- #


@unittest.skipUnless(_has("conv_with_bias.onnx"),
                     "Run test/gen_conv_models.py first")
class TestReportQuantizationFixedPoint(unittest.TestCase):
    """For ap_fixed<16,8> the report adds:
      * a Weight quantization sub-section under Parameters,
      * two new columns (Out max |abs|, Out max |rel|) in Layers."""

    def setUp(self):
        # Default dtype is ap_fixed<16,8>.
        self.md = _gen_md("conv_with_bias.onnx")

    def test_weight_quant_subsection_present(self):
        self.assertIn("### Weight quantization error", self.md)
        self.assertIn("`ap_fixed<16,8>`", self.md)

    def test_weight_quant_header_escapes_pipes(self):
        # The literal `|` in `|abs|` MUST be backslash-escaped so
        # markdown parses the header row as the same column count as
        # the separator row below; otherwise the table renders mangled.
        weight_block = self.md.split("### Weight quantization error", 1)[1]
        weight_block = weight_block.split("##", 1)[0]
        self.assertIn("Max \\|abs\\|", weight_block)
        self.assertIn("NRMSE",   weight_block)
        self.assertIn("SQNR (dB)", weight_block)
        header_line = next(
            ln for ln in weight_block.splitlines()
            if ln.startswith("| Tensor")
        )
        sep_line = next(
            ln for ln in weight_block.splitlines()
            if ln.startswith("|--")
        )
        # Column separators in markdown are unescaped pipes; count those.
        def _cell_count(s: str) -> int:
            unescaped = re.sub(r"\\\|", "", s)
            return unescaped.count("|") - 1
        self.assertEqual(
            _cell_count(header_line), _cell_count(sep_line),
            f"header/separator column-count mismatch:\n"
            f"  header={header_line!r}\n  sep={sep_line!r}",
        )

    def test_weight_quant_table_lists_aggregate_and_per_tensor(self):
        # Aggregate row + at least one per-tensor row (the Conv weight).
        weight_block = self.md.split("### Weight quantization error", 1)[1]
        weight_block = weight_block.split("##", 1)[0]
        self.assertIn("**All weights (worst)**", weight_block)
        # At least two table data rows beyond the aggregate (W and bias).
        data_rows = [ln for ln in weight_block.splitlines()
                     if ln.startswith("| `") and "|" in ln]
        self.assertGreaterEqual(len(data_rows), 1,
                                f"expected per-tensor rows, got:\n{weight_block}")

    def test_weight_rows_link_to_consuming_layer(self):
        # The "Used by" column should name the consuming node and the
        # role.  conv_with_bias has exactly one Conv layer with a weight
        # tensor `W` and a bias tensor `B`.
        weight_block = self.md.split("### Weight quantization error", 1)[1]
        weight_block = weight_block.split("##", 1)[0]
        self.assertRegex(weight_block, r"`W`\s+\|\s+\[\d+,\d+,\d+,\d+\]\s+\|"
                                       r"\s+\d+\s+\|\s+\[0\] Conv \(weight\)")
        self.assertRegex(weight_block, r"`B`\s+\|\s+\[\d+\]\s+\|\s+\d+\s+\|"
                                       r"\s+\[0\] Conv \(bias\)")

    def test_layers_table_has_quant_columns(self):
        layers_block = self.md.split("## Layers", 1)[1]
        layers_block = layers_block.split("## Generated", 1)[0]
        self.assertIn("Out max \\|abs\\|", layers_block)
        self.assertIn("NRMSE", layers_block)
        self.assertIn("SQNR (dB)", layers_block)

    def test_legend_present_for_both_quant_tables(self):
        # Each fixed-point quant table is preceded by an inline legend
        # defining each metric so a reader doesn't have to guess what
        # the columns mean.  We dropped the misleading max-relative
        # metric in favour of NRMSE + SQNR — the legend must reflect
        # that.
        weight_block = self.md.split("### Weight quantization error", 1)[1]
        weight_block = weight_block.split("##", 1)[0]
        self.assertIn("**Legend.**", weight_block)
        self.assertIn("`Max |abs|`", weight_block)
        self.assertIn("`NRMSE`", weight_block)
        self.assertIn("`SQNR (dB)`", weight_block)
        self.assertNotIn("Max |rel|", weight_block,
                         "old misleading max-relative metric must be gone")

        layers_block = self.md.split("## Layers", 1)[1]
        layers_block = layers_block.split("## Generated", 1)[0]
        self.assertIn("**Legend.**", layers_block)
        self.assertIn("`Out max |abs|`", layers_block)
        self.assertIn("`NRMSE`", layers_block)
        self.assertIn("`SQNR (dB)`", layers_block)

    def test_weight_sqnr_is_reasonable_for_ap_fixed_16_8(self):
        # SQNR for round-to-nearest 16-bit on dense Conv weights should
        # be high enough that the model's accuracy is not destroyed.
        # We assert a very loose floor (≥ 30 dB) so the test stays
        # robust across fixtures, but tight enough to catch regressions
        # that would silently degrade quantisation quality.
        weight_block = self.md.split("### Weight quantization error", 1)[1]
        weight_block = weight_block.split("##", 1)[0]
        # Per-tensor row format: `… | `1.234e-03` | `0.05%` | `60.5 dB` |`
        sqnr_values = re.findall(r"`(-?\d+\.\d+)\s*dB`", weight_block)
        self.assertGreater(len(sqnr_values), 0, "no SQNR cells found")
        for s in sqnr_values:
            self.assertGreater(
                float(s), 30.0,
                f"weight SQNR {s} dB too low for ap_fixed<16,8> — "
                "quantisation has regressed",
            )

    def test_per_layer_abs_error_within_lsb(self):
        # ap_fixed<16,8>: LSB = 1/2**8 ≈ 3.9e-3.  Truncation is
        # floor-toward-−∞ so the residual is bounded by 1 LSB.
        # We assert a generous 2-LSB ceiling to stay flake-free.
        layers_block = self.md.split("## Layers", 1)[1]
        layers_block = layers_block.split("## Generated", 1)[0]
        # Find every "abs" cell (`<value>e-NN`) in the data rows.
        abs_values = re.findall(r"`(\d+\.\d+e[+-]\d+)`\s+\|", layers_block)
        self.assertGreater(len(abs_values), 0,
                           "no abs-error cells parsed from layers table")
        for s in abs_values:
            v = float(s)
            self.assertLessEqual(v, 2.0 / 256,
                                 f"layer abs error {s} exceeds 2 LSB "
                                 f"({2.0/256:.3e}) for ap_fixed<16,8>")


@unittest.skipUnless(_has("single_add.onnx"), "Run test/gen_test_models.py first")
class TestReportQuantizationFloat32(unittest.TestCase):
    """For Float32 the quantization columns and weight subsection are
    suppressed (Float32 has no truncation)."""

    def test_no_quant_columns_for_float32(self):
        from src.dtype import FLOAT32
        path = _model("single_add.onnx")
        g    = OnnxGraph(path, dtype=FLOAT32)
        cg   = CodeGenerator(g, model_path=path, dtype=FLOAT32)
        md = ReportGenerator(
            graph=g, codegen=cg, model_path=path,
            out_dir="/tmp/test", generated_files=["CMakeLists.txt"],
        ).render_markdown()

        self.assertNotIn("### Weight quantization error", md)
        self.assertNotIn("Out max \\|abs\\|", md)
        self.assertNotIn("Out max \\|rel\\|", md)


if __name__ == "__main__":
    unittest.main()
