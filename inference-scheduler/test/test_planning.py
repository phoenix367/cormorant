"""
Planning mode, phase T0 (doc/plans/TACTICS_PLAN.md §5): the options
(src/planning.py), the persistent states' DAG edges (src/schedule.py) and
the ``--plan`` switch of the generators.
"""

import argparse
import filecmp
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from src.codegen import CodeGenerator  # noqa: E402
from src.graph import OnnxGraph  # noqa: E402
from src.llm_nodes import LlmAttnConvNode, LlmAttnPrepNode  # noqa: E402
from src.planning import (PlanOptions, add_plan_args, format_entry_weights,  # noqa: E402
                          parse_entry_weights, plan_options_from_args)
from src.schedule import Dag  # noqa: E402

MODELS = os.path.join(HERE, "models")


class TestPlanOptions(unittest.TestCase):
    def _parse(self, argv):
        p = argparse.ArgumentParser()
        add_plan_args(p)
        return plan_options_from_args(p.parse_args(argv))

    def test_default_is_off(self):
        o = self._parse([])
        self.assertEqual(o, PlanOptions())
        self.assertFalse(o.active)
        self.assertEqual(o.argv(), [])

    def test_cli_roundtrip(self):
        argv = ["--plan", "--plan-report", "--perf-model", "m.json", "--pool-budget-mib", "512",
                "--entry-weights", "decode=64,prefill_256=1"]
        o = self._parse(argv)
        self.assertTrue(o.enabled and o.report and o.active)
        self.assertEqual(o.entry_weights, {"decode": 64.0, "prefill_256": 1.0})
        self.assertEqual(self._parse(o.argv()), o)

    def test_entry_weights(self):
        self.assertEqual(parse_entry_weights(""), {})
        self.assertEqual(format_entry_weights({"a": 2.0, "b": 0.5}), "a=2,b=0.5")
        for bad in ("decode", "=3", "decode=-1", "decode=x"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_entry_weights(bad)

    def test_from_config(self):
        self.assertEqual(PlanOptions.from_config({}), PlanOptions())
        self.assertEqual(PlanOptions.from_config({"plan": True}), PlanOptions(enabled=True))
        o = PlanOptions.from_config({"plan": {"perf_model": "x.json", "entry_weights": "d=2"}})
        self.assertEqual((o.enabled, o.perf_model, o.entry_weights), (True, "x.json", {"d": 2.0}))
        with self.assertRaises(ValueError):
            PlanOptions.from_config({"plan": {"bogus": 1}})
        with self.assertRaises(ValueError):
            PlanOptions.from_config({"plan": "yes"})

    def test_graph_keeps_options(self):
        path = os.path.join(MODELS, "mixed_ops.onnx")
        if not os.path.exists(path):
            self.skipTest("test models not generated")
        self.assertEqual(OnnxGraph(path).plan, PlanOptions())
        # an explicit (empty) model: the test must not depend on the local
        # bitstream_config_kv260.json naming a calibrated bitstream
        with tempfile.TemporaryDirectory() as td:
            o = PlanOptions(enabled=True, perf_model=_model_file(td))
            self.assertIs(OnnxGraph(path, plan=o).plan, o)


def _model_file(td, exact=None, families=None):
    """A performance model file with the given exact entries {key: us}."""
    import json
    d = {"platform": "kv260", "bitstream": "test00000000", "exact":
         {k: {"us": v} for k, v in (exact or {}).items()}, "families": families or {}}
    p = os.path.join(td, "model.json")
    with open(p, "w") as f:
        json.dump(d, f)
    return p


def _strip_plan(lines, out_dir):
    """File lines without the timestamp, the banner's "Planned:" block, the
    report's planning bullet and section; the output directory normalised."""
    out, cont, section = [], False, False
    for ln in lines:
        if ln.startswith("## "):
            section = ln.startswith("## Planning")
        if section:
            continue
        if "Planned:" in ln or "**Planning**" in ln:
            cont = True
            continue
        if cont and ln.startswith(" *          "):
            continue
        cont = False
        if "Generated at" not in ln:
            out.append(ln.replace(out_dir, "OUT"))
    return out


class TestPlanSwitch(unittest.TestCase):
    """--plan on a graph with nothing to plan changes only the banner."""

    def test_cli_output_identical(self):
        import inference_scheduler
        path = os.path.join(MODELS, "mixed_ops.onnx")
        if not os.path.exists(path):
            self.skipTest("test models not generated")
        with tempfile.TemporaryDirectory() as td:
            a, b = os.path.join(td, "a"), os.path.join(td, "b")
            pm = _model_file(td)
            self.assertEqual(inference_scheduler.main(["--out-dir", a, path]), 0)
            self.assertEqual(inference_scheduler.main(["--out-dir", b, "--plan", "--perf-model", pm,
                                                       path]), 0)
            diff = []
            for root, _dirs, files in os.walk(a):
                for f in files:
                    pa = os.path.join(root, f)
                    pb = os.path.join(b, os.path.relpath(pa, a))
                    if filecmp.cmp(pa, pb, shallow=False):
                        continue
                    la = _strip_plan(open(pa, errors="replace"), a)
                    lb = _strip_plan(open(pb, errors="replace"), b)
                    if la != lb:
                        diff.append(os.path.relpath(pa, a))
            self.assertEqual(diff, [])


class TestPlannedTactics(unittest.TestCase):
    """T2: the planner's choices from a performance model."""

    def _rows_model(self, td):
        from test_matmul_on_conv import _matmul_model
        return _matmul_model(os.path.join(td, "rows.onnx"), [1024, 64], [64, 128], seed=12)

    def _calls(self, g):
        from src.nodes import MatmulConvNode
        (sn,) = [s for s in g.nodes if isinstance(s, MatmulConvNode)]
        return sn

    def test_clearly_faster_plan_is_taken(self):
        from src.matmul_lowering import conv_plans, plan_conv_calls
        from src.tactics import matmul_of
        with tempfile.TemporaryDirectory() as td:
            path = self._rows_model(td)
            base = self._calls(OnnxGraph(path, fuse_act=True, s2d_stem=True))
            mm = matmul_of(base)
            plans = conv_plans(mm, range(1, 8), splits="all")
            other = next(p for p in plans if p.conv_n == 256)
            exact = {plan_conv_calls(mm, p)[0].key(): 1000.0 for p in plans}
            exact[plan_conv_calls(mm, other)[0].key()] = 100.0     # 4 x 100 us beats 2 x 1000
            pm = _model_file(td, exact)
            g = OnnxGraph(path, fuse_act=True, s2d_stem=True,
                          plan=PlanOptions(enabled=True, perf_model=pm))
            sn = self._calls(g)
            self.assertEqual((sn.conv_n, sn.calls, sn.kw, sn.out_w),
                             (256, 4, other.kw, other.out_w))
            (e,) = g.plan_log
            self.assertEqual(e["decision"], "planned")
            self.assertAlmostEqual(e["chosen_us"], 400.0)
            # the planned project is still bit-exact on the host emulation
            import host_emu
            if host_emu.which_cc():
                cg = CodeGenerator(g, model_path=path)
                rc, out = host_emu.build_and_run(cg, os.path.join(td, "emu"))
                self.assertEqual(rc, 0, out[-2000:])
                self.assertIn("test_inference PASSED", out)

    def test_small_gain_keeps_the_baseline(self):
        from src.matmul_lowering import conv_plans, plan_conv_calls
        from src.tactics import matmul_of
        with tempfile.TemporaryDirectory() as td:
            path = self._rows_model(td)
            base = self._calls(OnnxGraph(path, fuse_act=True, s2d_stem=True))
            mm = matmul_of(base)
            plans = conv_plans(mm, range(1, 8), splits="all")
            # every plan 1000 us in all, one 1 % faster
            exact = {plan_conv_calls(mm, p)[0].key(): 1000.0 / p.calls for p in plans}
            other = next(p for p in plans
                         if (p.conv_n, p.kw, p.out_w) != (base.conv_n, base.kw, base.out_w))
            exact[plan_conv_calls(mm, other)[0].key()] = 990.0 / other.calls
            g = OnnxGraph(path, fuse_act=True, s2d_stem=True,
                          plan=PlanOptions(enabled=True, perf_model=_model_file(td, exact)))
            sn = self._calls(g)
            self.assertEqual((sn.conv_n, sn.kw, sn.out_w), (base.conv_n, base.kw, base.out_w))
            self.assertEqual(g.plan_log[0]["decision"], "gain below the minimum")

    def test_error_band(self):
        from src.perf_model import clearly_faster
        self.assertTrue(clearly_faster((90.0, 0.0), (100.0, 0.0), 0.03))
        self.assertFalse(clearly_faster((98.0, 0.0), (100.0, 0.0), 0.03))
        self.assertTrue(clearly_faster((90.0, 0.04), (100.0, 0.0), 0.03))   # 93.6 < 97
        self.assertFalse(clearly_faster((94.0, 0.04), (100.0, 0.0), 0.03))  # 97.8 > 97
        self.assertTrue(clearly_faster((50.0, 0.0), (100.0, 0.3), 0.03))    # measured vs model
        # a candidate from a family predicting worse than MAX_MODEL_ERROR never wins
        self.assertFalse(clearly_faster((10.0, 0.2), (100.0, 0.0), 0.03))

    def test_unpriced_baseline_is_kept(self):
        with tempfile.TemporaryDirectory() as td:
            path = self._rows_model(td)
            base = self._calls(OnnxGraph(path, fuse_act=True, s2d_stem=True))
            g = OnnxGraph(path, fuse_act=True, s2d_stem=True,
                          plan=PlanOptions(enabled=True, perf_model=_model_file(td)))
            sn = self._calls(g)
            self.assertEqual((sn.conv_n, sn.kw, sn.out_w), (base.conv_n, base.kw, base.out_w))
            self.assertEqual(g.plan_log[0]["decision"], "baseline not priced")

    def test_no_model_is_an_error(self):
        from src.nodes import SchedulerError
        from src import planning
        with tempfile.TemporaryDirectory() as td:
            path = self._rows_model(td)
            orig = planning.resolve_perf_model

            def missing(opts):
                raise planning.PlanError("--plan: no performance model (test)")
            planning.resolve_perf_model = missing
            try:
                with self.assertRaises(SchedulerError):
                    OnnxGraph(path, plan=PlanOptions(enabled=True))
            finally:
                planning.resolve_perf_model = orig

    def test_banner_and_report(self):
        with tempfile.TemporaryDirectory() as td:
            path = self._rows_model(td)
            g = OnnxGraph(path, fuse_act=True, s2d_stem=True,
                          plan=PlanOptions(enabled=True, perf_model=_model_file(td)))
            src = CodeGenerator(g, model_path=path).generate_source()
            self.assertIn(" * Planned: performance model kv260/test00000000: 0 of 1 MatMul", src)

    def test_shared_kw_joint_choice(self):
        """plan_shared_kw: a width clearly cheaper over prefill + decode wins,
        a marginal one does not."""
        from onnx import TensorProto
        from onnx import helper as oh
        from onnx import numpy_helper as nph
        import numpy as np
        from src.llm_entries import plan_shared_kw
        from src.matmul_lowering import conv_plans, plan_conv_calls
        from src.perf_calls import KernelCall
        from src.perf_model import PerfModel
        from src.tactics import _MM
        K, M = 768, 768

        def entry(n):
            g = oh.make_graph([oh.make_node("MatMul", ["X", "W"], ["Y"])], "e",
                              [oh.make_tensor_value_info("X", TensorProto.FLOAT, [n, K])],
                              [oh.make_tensor_value_info("Y", TensorProto.FLOAT, [n, M])],
                              initializer=[nph.from_array(np.zeros((K, M), np.float32), "W")])
            return oh.make_model(g)

        class _T:
            is_int, data, onnx_name = False, None, ""
        models = {"prefill_64": entry(64), "decode": entry(1)}
        mm = _MM(64, K, M, 1, 0, 0, 0, [_T(), _T()])
        base_kw = conv_plans(mm, [1, 2, 4, 8])[0].kw
        other = 2 if base_kw != 2 else 4
        exact = {}
        for kw in (1, 2, 4, 8):
            for p in conv_plans(mm, [kw], splits="all"):
                exact[plan_conv_calls(mm, p)[0].key()] = 500.0
            exact[KernelCall.of("MatmulKernel", n=1, k=K, m=M, batch=1, gemv_kw=kw).key()] = 100.0
        with tempfile.TemporaryDirectory() as td:
            pm = PerfModel.load(_model_file(td, exact))
            self.assertEqual(plan_shared_kw(models, ["prefill_64"], (1, 2, 4, 8), pm,
                                            lambda n: 1.0), {"W": base_kw})
            exact[KernelCall.of("MatmulKernel", n=1, k=K, m=M, batch=1, gemv_kw=other).key()] = 10.0
            pm = PerfModel.load(_model_file(td, exact))
            self.assertEqual(plan_shared_kw(models, ["prefill_64"], (1, 2, 4, 8), pm,
                                            lambda n: 64.0 if n == "decode" else 1.0),
                             {"W": other})


class TestPlannedImageTactic(unittest.TestCase):
    """The RTL MatmulKernel's image (GEMV) path for several rows as a planned
    tactic (MATMUL_RTL_PLAN phase 4 follow-up): offered by src/tactics.py,
    chosen by the lowering's planner when clearly faster (a B it may re-image
    at kw > 1), for a pinned weight only in its pinned layout, and priced in
    the shared kernel widths of a multi-entry project (plus the tiled
    layout, 0); never on the HLS kernel."""

    N, K, M = 64, 256, 96

    def _model(self, td):
        from test_matmul_on_conv import _matmul_model
        return _matmul_model(os.path.join(td, "mm.onnx"), [self.N, self.K], [self.K, self.M],
                             seed=21)

    def _exact(self, fast: dict):
        """{key: us}: every ConvKernel plan, the tiled path and each image
        width at 1000 us, except the {layout: us} of ``fast`` (0 = tiled)."""
        from src.matmul_lowering import conv_plans, plan_conv_calls, plan_mm_calls
        from src.tactics import _MM
        mm = _MM(self.N, self.K, self.M, 1, 0, 0, 0, [])
        exact = {plan_conv_calls(mm, p)[0].key(): 1000.0
                 for p in conv_plans(mm, range(1, 8), splits="all")}
        for lay in (0, 1, 2, 4, 8):
            exact[plan_mm_calls(mm, lay, True)[0].key()] = fast.get(lay, 1000.0)
        return exact

    def _graph(self, td, fast, **kw):
        pm = _model_file(td, self._exact(fast))
        return OnnxGraph(self._model(td), fuse_act=True, s2d_stem=True,
                         plan=PlanOptions(enabled=True, perf_model=pm), **kw)

    @staticmethod
    def _mm(g):
        from src.nodes import MatmulNode
        return [s for s in g.nodes if type(s) is MatmulNode]

    class _T:                                     # a MatMul operand stub (src/tactics._MM)
        is_int, data, onnx_name = False, None, ""

    def test_tactics_offer_the_image_for_several_rows_on_rtl_only(self):
        from helpers import matmul_impl
        from src.tactics import _MM, matmul_tactics
        mm = _MM(16, 256, 96, 1, 0, 0, 0, [self._T(), self._T()])
        with matmul_impl("rtl"):
            ts = matmul_tactics(mm, relayout_ok=True, b_constant=True)
        self.assertEqual(sorted(t.p["kw"] for t in ts if t.kind == "gemv"), [1, 2, 4, 8])
        self.assertEqual({t.calls[0].fields["n"] for t in ts if t.kind == "gemv"}, {16})
        with matmul_impl("hls"):
            ts = matmul_tactics(mm, relayout_ok=True, b_constant=True)
        self.assertEqual([t for t in ts if t.kind == "gemv"], [])

    def test_image_chosen_when_clearly_faster(self):
        from helpers import matmul_impl
        from src.nodes import conv_lowered_b_image
        for kw in (1, 4):
            with self.subTest(kw=kw), tempfile.TemporaryDirectory() as td, matmul_impl("rtl"):
                g = self._graph(td, {kw: 100.0})
                (sn,) = self._mm(g)
                self.assertEqual((sn.gemv_kw, sn.plan_kw, bool(sn.b_packed)), (kw, kw, False))
                self.assertEqual(g.plan_log[0]["chosen"], f"MatmulKernel image kw={kw}")
                b = sn.inputs[1]
                if kw > 1:                      # re-imaged for MatmulKernel alone
                    import numpy as np
                    np.testing.assert_array_equal(
                        b.packed_data, conv_lowered_b_image(b.data, self.K, self.M, kw))
                else:
                    self.assertIsNone(b.packed_data)
                import host_emu
                if host_emu.which_cc():
                    cg = CodeGenerator(g, model_path="mm.onnx")
                    rc, out = host_emu.build_and_run(cg, os.path.join(td, "emu"))
                    self.assertEqual(rc, 0, out[-2000:])
                    self.assertIn("test_inference PASSED", out)

    def test_tiled_chosen_when_clearly_faster(self):
        from helpers import matmul_impl
        with tempfile.TemporaryDirectory() as td, matmul_impl("rtl"):
            g = self._graph(td, {0: 100.0})
            (sn,) = self._mm(g)
            self.assertEqual((sn.gemv_kw, sn.plan_kw, bool(sn.b_packed)), (0, 0, True))

    def test_pinned_weight_reads_its_layout_only(self):
        from helpers import matmul_impl
        with tempfile.TemporaryDirectory() as td, matmul_impl("rtl"):
            # the tiled path and kw 1 are cheapest, but the weight is pinned to
            # ConvKernel's kw-4 image: MatmulKernel may read only that one
            g = self._graph(td, {0: 10.0, 1: 10.0, 4: 100.0}, matmul_conv_kw={"B": 4})
            (sn,) = self._mm(g)
            self.assertEqual((sn.gemv_kw, bool(sn.b_packed)), (4, False))
            self.assertEqual(g.plan_log[0]["chosen"], "MatmulKernel image kw=4")

    def test_no_image_on_hls(self):
        from helpers import matmul_impl
        with tempfile.TemporaryDirectory() as td, matmul_impl("hls"):
            g = self._graph(td, {1: 100.0, 4: 100.0})
            self.assertNotIn("image", g.plan_log[0]["chosen"])
            self.assertTrue(all(sn.gemv_kw == 0 for sn in self._mm(g)))

    def test_shared_kw_counts_the_image_and_the_tiled_layout(self):
        from onnx import TensorProto
        from onnx import helper as oh
        from onnx import numpy_helper as nph
        import numpy as np
        from helpers import matmul_impl
        from src.llm_entries import plan_shared_kw
        from src.matmul_lowering import conv_plans, plan_conv_calls, plan_mm_calls
        from src.perf_model import PerfModel
        from src.tactics import _MM
        K, M = 768, 768

        def entry(n):
            g = oh.make_graph([oh.make_node("MatMul", ["X", "W"], ["Y"])], "e",
                              [oh.make_tensor_value_info("X", TensorProto.FLOAT, [n, K])],
                              [oh.make_tensor_value_info("Y", TensorProto.FLOAT, [n, M])],
                              initializer=[nph.from_array(np.zeros((K, M), np.float32), "W")])
            return oh.make_model(g)

        pre = _MM(64, K, M, 1, 0, 0, 0, [self._T(), self._T()])
        one = _MM(1, K, M, 1, 0, 0, 0, [self._T(), self._T()])
        hls_base = conv_plans(pre, [1, 2, 4, 8])[0].kw   # the unplanned width (ConvKernel here)
        fast = next(kw for kw in (2, 4, 8) if kw != hls_base)
        exact = {plan_conv_calls(pre, p)[0].key(): 500.0
                 for kw in (1, 2, 4, 8) for p in conv_plans(pre, [kw], splits="all")}
        for kw in (1, 2, 4, 8):
            exact[plan_mm_calls(one, kw, True)[0].key()] = 100.0
        exact[plan_mm_calls(pre, fast, True)[0].key()] = 50.0   # the bucket's image at `fast`
        exact[plan_mm_calls(pre, 0, True)[0].key()] = 400.0     # tiled, packed
        exact[plan_mm_calls(one, 0, True)[0].key()] = 100.0
        args = (["prefill_64"], (1, 2, 4, 8))
        with tempfile.TemporaryDirectory() as td:
            pm = PerfModel.load(_model_file(td, exact))
            with matmul_impl("rtl"):
                self.assertEqual(plan_shared_kw({"prefill_64": entry(64), "decode": entry(1)},
                                                *args, pm, lambda n: 1.0), {"W": fast})
            with matmul_impl("hls"):          # no image path for 64 rows: all widths tie
                self.assertEqual(plan_shared_kw({"prefill_64": entry(64), "decode": entry(1)},
                                                *args, pm, lambda n: 1.0), {"W": hls_base})
            exact[plan_mm_calls(pre, 0, True)[0].key()] = 10.0       # the tiled layout wins
            pm = PerfModel.load(_model_file(td, exact))
            with matmul_impl("rtl"):
                self.assertEqual(plan_shared_kw({"prefill_64": entry(64), "decode": entry(1)},
                                                *args, pm, lambda n: 1.0), {"W": 0})


class TestStateEdges(unittest.TestCase):
    """RAW / WAR / WAW edges of the persistent states."""

    def test_vision_caches_shared_by_layers(self):
        try:
            from test_vit import Tiny
        except ImportError as e:                              # pragma: no cover
            self.skipTest(f"tiny ViT unavailable: {e}")
        fe = Tiny.get()[3]
        g = OnnxGraph(fe.entry(), fuse_act=True, s2d_stem=True)
        dag, bare = Dag.from_graph(g), Dag.from_graph(g, state_edges=False)
        preps = [sn for sn in g.nodes if type(sn).__name__ == "VitAttnPrepNode"]
        attn = [sn for sn in g.nodes if isinstance(sn, LlmAttnConvNode)]
        self.assertEqual(len(preps), 2)
        p0, p1 = preps
        layer0 = [sn for sn in attn if p0.index < sn.index < p1.index]
        layer1 = [sn for sn in attn if sn.index > p1.index]
        self.assertTrue(layer0 and layer1)
        for sn in layer0:                                     # RAW: reads the caches p0 wrote
            self.assertIn(p0.index, dag.predecessors(sn.index))
        for sn in layer1:
            self.assertIn(p1.index, dag.predecessors(sn.index))
        for sn in layer0:                                     # WAR: p1 overwrites what they read
            self.assertIn(sn.index, dag.predecessors(p1.index))
            self.assertNotIn(sn.index, bare.predecessors(p1.index))
        self.assertIn(p0.index, dag.predecessors(p1.index))  # WAW
        order = {i: k for k, i in enumerate(dag.topological_order())}
        self.assertTrue(all(order[sn.index] < order[p1.index] for sn in layer0))

    def test_llm_prefill_caches(self):
        path = os.path.join(MODELS, "llama_tiny_prefill8.onnx")
        if not os.path.exists(path):
            self.skipTest("llama tiny models not generated")
        g = OnnxGraph(path, fuse_act=True, s2d_stem=True)
        dag = Dag.from_graph(g)
        preps = {sn.index: sn for sn in g.nodes if isinstance(sn, LlmAttnPrepNode)}
        self.assertTrue(preps)
        for sn in g.nodes:
            if not isinstance(sn, LlmAttnConvNode):
                continue
            states = {t.onnx_name for t in sn.inputs if t.is_state}
            writers = [i for i, p in preps.items() if i < sn.index and
                       states & {t.onnx_name for t in p.state_updates()}]
            if writers:
                self.assertIn(max(writers), dag.predecessors(sn.index), sn.index)

    def test_event_stream_unchanged(self):
        """Today's list order already respects every state edge: the edges
        add no wait to the generated code."""
        import src.schedule as S
        paths = [os.path.join(MODELS, n) for n in ("llama_tiny_prefill8.onnx",
                                                    "llama_tiny_decode.onnx")]
        paths = [p for p in paths if os.path.exists(p)]
        if not paths:
            self.skipTest("llama tiny models not generated")
        orig = S.Dag.from_graph.__func__
        for p in paths:
            g = OnnxGraph(p, fuse_act=True, s2d_stem=True)
            cg = CodeGenerator(g, model_path=p)
            with_edges = cg._compute_event_stream()
            try:
                S.Dag.from_graph = classmethod(
                    lambda cls, gr, state_edges=True: orig(cls, gr, state_edges=False))
                without = cg._compute_event_stream()
            finally:
                S.Dag.from_graph = classmethod(orig)
            self.assertEqual(with_edges, without, os.path.basename(p))


if __name__ == "__main__":
    unittest.main()


class TestReorderedCode(unittest.TestCase):
    """Any order the DAG allows (state edges included) generates C that
    computes the same bits: random valid orders of mixed models, host
    emulation against the simulation."""

    def _random_order(self, g, seed):
        import random
        rnd = random.Random(seed)
        dag = Dag.from_graph(g)
        indeg = {n.index: len(n.preds) for n in dag.nodes}
        ready = sorted(i for i, d in indeg.items() if d == 0)
        order = []
        while ready:
            v = ready.pop(rnd.randrange(len(ready)))
            order.append(v)
            for s in sorted(dag.successors(v)):
                indeg[s] -= 1
                if indeg[s] == 0:
                    ready.append(s)
        return order

    def test_random_orders_bit_exact(self):
        import host_emu
        if not host_emu.which_cc():
            self.skipTest("no C compiler")
        names = ("mixed_all_skip_conv_matmul.onnx", "mixed_all_norm_conv_project.onnx",
                 "mixed_matmul_add_relu.onnx", "bert_tiny_h32_l1.onnx",
                 "llama_tiny_prefill8.onnx")
        done = 0
        for n in names:
            path = os.path.join(MODELS, n)
            if not os.path.exists(path):
                continue
            for seed in (1, 2):
                from src.order_search import apply_order
                g = OnnxGraph(path, fuse_act=True, s2d_stem=True, fuse_patterns=True)
                order = self._random_order(g, seed)
                names = {sn.index: sn.onnx_node.name for sn in g.nodes}
                apply_order(g, order)
                self.assertEqual([sn.index for sn in g.nodes], list(range(len(order))))
                cg = CodeGenerator(g, model_path=path)
                # the profiler's name table matches the brackets (PROF_BEGIN(sn.index))
                table = cg._layer_display_names()
                for sn in g.nodes:
                    if sn.onnx_node.name and table.count(sn.onnx_node.name) == 1:
                        self.assertEqual(table[sn.index], sn.onnx_node.name)
                self.assertEqual(sorted(names.values()), sorted(sn.onnx_node.name for sn in g.nodes))
                with self.subTest(model=n, seed=seed), tempfile.TemporaryDirectory() as td:
                    rc, out = host_emu.build_and_run(cg, td)
                    self.assertEqual(rc, 0, out[-2000:])
                    self.assertIn("test_inference PASSED", out)
                done += 1
        if not done:
            self.skipTest("test models not generated")
