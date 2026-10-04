"""The execution timeline: the spans of src/codegen/timing.simulate (the CPU
tiles the run, a kernel lane never overlaps itself, every node ends where
node_end says) and its profiler windows, the page data of src/timeline_html.py
(lanes, node table, dependencies, escaping, measured per-layer times matched
to nodes, the profile formats), its JavaScript (node --check when Node.js is
installed) and timeline.html from the CLI with --plan-report and
--timeline-profile."""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers import _model, _models_exist  # noqa: E402

from src.codegen import CodeGenerator  # noqa: E402
from src.codegen.timing import simulate  # noqa: E402
from src.graph import OnnxGraph  # noqa: E402
from src.timeline_html import KINDS, load_profile, match_layers, render_html, timeline_entry  # noqa: E402

MODELS = ("parallel_conv_pool_join.onnx", "parallel_matmul_relu_join.onnx", "asymmetric_nested_branches.onnx",
          "mixed_all_skip_conv_matmul.onnx", "mixed_all_norm_conv_project.onnx", "mixed_ops.onnx")


def _sim(name):
    g = OnnxGraph(_model(name), fuse_act=True)
    cg = CodeGenerator(g, model_path=_model(name))
    tl = simulate(cg, lambda sn: 10.0 + sn.index % 7, lambda sn: 4.0)   # µs per kernel / host node
    return g, cg, tl


def _page_data(text: str) -> dict:
    m = re.search(r'<script id="data" type="application/json">(.*?)</script>', text, re.S)
    return json.loads(m.group(1))


@unittest.skipUnless(_models_exist(), "Run test/gen_all_models.py first")
class TestSpans(unittest.TestCase):

    def test_invariants(self):
        for name in MODELS:
            with self.subTest(model=name):
                g, cg, tl = _sim(name)
                cpu = sorted((s for s in tl.spans if s[0] == "CPU"), key=lambda s: s[2])
                self.assertTrue(cpu)
                self.assertAlmostEqual(cpu[0][2], 0.0)
                self.assertAlmostEqual(cpu[-1][3], tl.total_us)
                for a, b in zip(cpu, cpu[1:], strict=False):          # the CPU's time is one chain
                    self.assertAlmostEqual(a[3], b[2], places=6)
                lanes = defaultdict(list)
                for s in tl.spans:
                    self.assertIn(s[4], KINDS)
                    self.assertGreaterEqual(s[3], s[2])
                    if s[0] != "CPU":
                        self.assertEqual(s[4], "kernel")
                        lanes[s[0]].append(s)
                for spans in lanes.values():                          # one call at a time per lane
                    spans.sort(key=lambda s: s[2])
                    for a, b in zip(spans, spans[1:], strict=False):
                        self.assertLessEqual(a[3], b[2] + 1e-9)
                ends = defaultdict(float)
                for s in tl.spans:
                    ends[s[1]] = max(ends[s[1]], s[3])
                for idx, end in tl.node_end.items():
                    self.assertAlmostEqual(ends[idx], end, places=6)
                cpu_busy = sum(s[3] - s[2] for s in cpu if s[4] in ("host", "issue"))
                self.assertAlmostEqual(cpu_busy, tl.cpu_us, places=6)
                waits = sum(s[3] - s[2] for s in cpu if s[4] == "wait")
                self.assertAlmostEqual(waits, tl.wait_us, places=6)

    def test_windows(self):
        """The profiler's bracket per node: a started kernel from its issue to the
        CPU passing its wait (covering the kernel), a host op / sync call its span."""
        for name in MODELS:
            with self.subTest(model=name):
                g, cg, tl = _sim(name)
                own = {}
                for lane, idx, t0, t1, kind in tl.spans:
                    own.setdefault(idx, []).append((lane, t0, t1, kind))
                self.assertEqual(set(tl.windows), {i for i, sp in own.items()
                                                   if any(k in ("kernel", "host") for *_, k in sp)})
                for idx, (w0, w1) in tl.windows.items():
                    kinds = {k for *_, k in own[idx]}
                    main = next(sp for sp in own[idx] if sp[3] in ("kernel", "host"))
                    self.assertAlmostEqual(w0, main[1])
                    if "issue" in kinds:
                        self.assertGreaterEqual(w1 + 1e-9, tl.node_end[idx])
                        waits = [t1 for lane, t0, t1, k in own[idx] if k == "wait"]
                        if waits:
                            self.assertAlmostEqual(w1, max(waits))
                    else:
                        self.assertAlmostEqual(w1, main[2])

    def test_lanes_run_in_parallel(self):
        g, cg, tl = _sim("parallel_conv_pool_join.onnx")
        k = sorted((s for s in tl.spans if s[0] != "CPU"), key=lambda s: s[2])
        self.assertTrue(any(b[2] < a[3] and a[0] != b[0] for a, b in zip(k, k[1:], strict=False)),
                        "two kernel lanes should overlap in time")


@unittest.skipUnless(_models_exist(), "Run test/gen_all_models.py first")
class TestPageData(unittest.TestCase):

    def test_entry(self):
        g, cg, tl = _sim("mixed_all_skip_conv_matmul.onnx")
        e = timeline_entry("inference", cg, tl)
        self.assertEqual(e["lanes"][0]["id"], "CPU")
        self.assertAlmostEqual(e["total_us"], round(tl.total_us, 3))
        n = len(e["nodes"])
        self.assertEqual(n, len({s[1] for s in tl.spans}))
        start = {}
        for L in e["lanes"]:
            self.assertEqual(len(L["s"]) % 4, 0)
            for i in range(0, len(L["s"]), 4):
                t0, t1, ref, kind = L["s"][i:i + 4]
                self.assertTrue(0 <= ref < n and 0 <= kind < len(KINDS) and t1 >= t0)
                start[ref] = min(start.get(ref, t0), t0)
        end = {}
        for L in e["lanes"]:
            for i in range(0, len(L["s"]), 4):
                end[L["s"][i + 2]] = max(end.get(L["s"][i + 2], 0.0), L["s"][i + 1])
        for k, node in enumerate(e["nodes"]):
            idx, name, op, cls, lane, dur, src, calls, shape, preds, inputs, call_rows, meas, win = node
            self.assertTrue(name and cls and lane)
            self.assertIsNone(meas)
            self.assertEqual(win, [round(v, 3) for v in tl.windows[idx]] if idx in tl.windows else None)
            sn = next(s for s in g.nodes if s.index == idx)
            self.assertEqual([i[0] for i in inputs], ["x".join(map(str, t.shape)) for t in sn.inputs if t is not None])
            self.assertTrue(all(len(i) == 3 and i[2] in (0, 1) for i in inputs))
            from src.perf_calls import FIELDS
            want = list(sn.kernel_calls(cg._layouts)) if hasattr(sn, "kernel_calls") else []
            self.assertEqual([(r[0], r[1], tuple(r[2])) for r in call_rows],
                             [(c.kernel, c.count, c.regs) for c in want])
            self.assertEqual(calls, sum(c.count for c in want))
            self.assertTrue(all(len(r[2]) == len(FIELDS[r[0]]) for r in call_rows))
            for p in preds:                       # an input is ready before its consumer starts
                self.assertTrue(0 <= p < n and p != k)
                self.assertLessEqual(end[p], start[k] + 1e-6)

    def test_measured_layers(self):
        """Profile layers match nodes by the generated layer names; a name made
        unique with a (global) index matches by position; unprofiled layers
        (0 calls) and foreign names do not match."""
        g, cg, tl = _sim("mixed_all_skip_conv_matmul.onnx")
        names = cg._layer_display_names()
        base = 500                                         # a later entry of a multi-entry project
        layers = [{"i": base + k, "name": n, "calls": 3, "mean_us": 10.0 + k, "min_us": 9.0 + k}
                  for k, n in enumerate(names)]
        layers[1]["name"] = names[1] + "_" + str(base + 1)    # deduplicated with the global index
        layers[2]["calls"] = 0
        layers.append({"i": base + len(names), "name": "another_entry_node", "calls": 1, "mean_us": 1.0})
        m = match_layers(cg, layers)
        want = {sn.index: [10.0 + k, 9.0 + k, 3] for k, sn in enumerate(g.nodes) if k != 2}
        self.assertEqual(m, want)
        e = timeline_entry("inference", cg, tl, layers=layers)
        drawn = {n[0] for n in e["nodes"]}
        self.assertEqual(e["measured_nodes"], len(drawn & set(want)))
        self.assertTrue(e["profiled"])
        for n in e["nodes"]:
            self.assertEqual(n[12], want.get(n[0]))
        self.assertFalse(timeline_entry("inference", cg, tl)["profiled"])

    def test_profile_formats(self):
        ly = [{"i": 0, "name": "a", "calls": 1, "mean_us": 2.0, "min_us": 2.0}]
        cases = [
            ("PROFILE_PHASE: decode\nnoise\nLAYERS_JSON: " + json.dumps({"layers": ly}) + "\n"
             "LAYERS_JSON: " + json.dumps({"layers": ly[:0]}), {"decode": ly[:0]}),     # the last one wins
            ("LAYERS_JSON: " + json.dumps({"layers": ly}), {"": ly}),
            (json.dumps({"layers": ly}), {"": ly}),
            (json.dumps({"metrics": {"p50_ms": 1, "layer_stats": {"layers": ly}}}), {"": ly}),
            (json.dumps([{"name": "resnet18", "metrics": {"layer_stats": {"layers": ly}}}]), {"resnet18": ly}),
            (json.dumps({"bench": {}, "profile_layers": {"chunk": ly}}), {"chunk": ly}),
        ]
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "prof.txt")
            for text, want in cases:
                with open(p, "w") as f:
                    f.write(text)
                self.assertEqual(load_profile(p), want)
            for bad in ("no profile here", json.dumps({"metrics": {"p50_ms": 1}})):
                with open(p, "w") as f:
                    f.write(bad)
                with self.assertRaises(ValueError):
                    load_profile(p)

    def test_page_round_trip_and_escaping(self):
        g, cg, tl = _sim("mixed_ops.onnx")
        e = timeline_entry("inference", cg, tl)
        e["nodes"][0][1] = "</script><b>x</b>"
        text = render_html([e], "T <&>", "sub")
        self.assertEqual(text.count("</script>"), 2)            # the data block and the viewer only
        d = _page_data(text)
        self.assertEqual(d["entries"][0]["nodes"][0][1], "</script><b>x</b>")
        self.assertEqual(d["title"], "T <&>")
        self.assertIn("<title>T &lt;&amp;&gt;</title>", text)
        self.assertEqual(d["kinds"], list(KINDS))
        from src.perf_calls import FIELDS
        self.assertEqual(d["fields"], {k: list(v) for k, v in FIELDS.items()})

    @unittest.skipUnless(shutil.which("node"), "Node.js not installed")
    def test_javascript_parses(self):
        g, cg, tl = _sim("mixed_ops.onnx")
        text = render_html([timeline_entry("inference", cg, tl)], "t")
        js = re.findall(r"<script>(.*?)</script>", text, re.S)
        self.assertEqual(len(js), 1)
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "viewer.js")
            with open(p, "w") as f:
                f.write(js[0])
            r = subprocess.run(["node", "--check", p], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)


@unittest.skipUnless(_models_exist(), "Run test/gen_all_models.py first")
class TestCli(unittest.TestCase):

    def _model_file(self, td):
        p = os.path.join(td, "model.json")
        with open(p, "w") as f:
            json.dump({"platform": "kv260", "bitstream": "test00000000", "exact": {}, "families": {}}, f)
        return p

    def test_plan_report_writes_timeline(self):
        import inference_scheduler
        with tempfile.TemporaryDirectory() as td:
            out, plain = os.path.join(td, "p"), os.path.join(td, "u")
            self.assertEqual(inference_scheduler.main(["--out-dir", out, "--plan-report", "--perf-model",
                                                       self._model_file(td), _model("mixed_ops.onnx")]), 0)
            page = os.path.join(out, "timeline.html")
            self.assertTrue(os.path.isfile(page))
            with open(page) as f:
                d = _page_data(f.read())
            self.assertEqual([e["name"] for e in d["entries"]], ["inference"])
            self.assertIn("test00000000", d["subtitle"])
            self.assertEqual(inference_scheduler.main(["--out-dir", plain, _model("mixed_ops.onnx")]), 0)
            self.assertFalse(os.path.exists(os.path.join(plain, "timeline.html")))   # only when planning
            # --no-report skips report.md only
            quiet = os.path.join(td, "q")
            self.assertEqual(inference_scheduler.main(["--out-dir", quiet, "--no-report", "--plan-report",
                                                       "--perf-model", self._model_file(td),
                                                       _model("mixed_ops.onnx")]), 0)
            self.assertTrue(os.path.isfile(os.path.join(quiet, "timeline.html")))
            self.assertFalse(os.path.exists(os.path.join(quiet, "report.md")))

    def test_timeline_profile(self):
        """--timeline-profile: the profiler's LAYERS_JSON output, or one PHASE of a
        results file, annotates timeline.html; an unreadable one only warns."""
        import inference_scheduler
        model = _model("mixed_ops.onnx")
        g = OnnxGraph(model, fuse_act=True)
        names = CodeGenerator(g, model_path=model)._layer_display_names()
        ly = [{"i": k, "name": n, "calls": 2, "mean_us": 5.0, "min_us": 4.0} for k, n in enumerate(names)]
        with tempfile.TemporaryDirectory() as td:
            raw, res = os.path.join(td, "run.log"), os.path.join(td, "results.json")
            with open(raw, "w") as f:
                f.write("test_inference: PASS\nLAYERS_JSON: " + json.dumps({"layers": ly}) + "\n")
            with open(res, "w") as f:
                json.dump([{"name": "other", "metrics": {}}, {"name": "mine", "metrics": {"layer_stats": {"layers": ly}}}], f)
            for k, prof in enumerate((raw, "mine=" + res, os.path.join(td, "missing.json"))):
                out = os.path.join(td, f"p{k}")
                self.assertEqual(inference_scheduler.main(["--out-dir", out, "--plan-report", "--perf-model",
                                                           self._model_file(td), "--timeline-profile", prof,
                                                           model]), 0)
                with open(os.path.join(out, "timeline.html")) as f:
                    e = _page_data(f.read())["entries"][0]
                if k < 2:
                    self.assertTrue(e["profiled"])
                    self.assertEqual(e["measured_nodes"], len(e["nodes"]))
                    self.assertTrue(all(n[12] == [5.0, 4.0, 2] for n in e["nodes"]))
                else:
                    self.assertFalse(e["profiled"])


if __name__ == "__main__":
    unittest.main()
