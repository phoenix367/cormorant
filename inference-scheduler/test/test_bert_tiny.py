"""Tiny BERT fixtures end to end (BERT_PLAN phase 1f; test/gen_bert_models.py).

The fixtures reproduce bertsquad-12's node arrangement at toy size.  Checks:
the op -> engine partition; the scheduler simulation against onnxruntime
(float) within a quantisation tolerance and — for the opset-12 variants —
bit-exactly against the independent numpy emulation of the same partition
(demo/bert_squad/scripts/bert_study.py, policy "sched"); the generated C
compiles with -Werror and, run on the host against software models of the
kernels (test/host_emu.py), reproduces the simulation bit for bit.
"""

import collections
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

import gen_bert_models as gbm
import host_emu
from src.codegen import CodeGenerator
from src.graph import OnnxGraph
from src.host_nodes import (CastNode, GatherNode, GeluNode, HostNode, LayerNormNode,
                            OneHotNode, SliceNode, SoftmaxNode, TransposeNode)
from src.nodes import MatmulNode, ScheduledNode, SchedulerError
from src.report import ReportGenerator

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_STUDY = os.path.join(os.path.dirname(_ROOT), "demo", "bert_squad", "scripts")

try:
    import onnxruntime as ort
except ImportError:                                    # pragma: no cover
    ort = None


def _study():
    if _STUDY not in sys.path:
        sys.path.insert(0, _STUDY)
    import bert_study
    return bert_study


class _Tiny(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.models = {}
        for i, (fname, h, nh, nl, s, gelu, style, opset) in enumerate(gbm.VARIANTS):
            cls.models[fname] = dict(
                path=gbm.make_bert(os.path.join(cls._tmp.name, fname), h, nh, nl, s,
                                   gelu, style, opset, seed=i),
                hidden=h, heads=nh, layers=nl, seq=s, style=style, opset=opset)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    @staticmethod
    def gen(path, **kw):
        kw.setdefault("fuse_act", True)
        kw.setdefault("s2d_stem", True)
        g = OnnxGraph(path, **kw)
        return g, CodeGenerator(g, model_path=path)


class TestPartition(_Tiny):

    def test_node_kinds(self):
        for fname, m in self.models.items():
            g, _ = self.gen(m["path"])
            L = m["layers"]
            kinds = collections.Counter(type(sn) for sn in g.nodes)
            self.assertEqual(kinds[LayerNormNode], 1 + 2 * L, fname)
            self.assertEqual(kinds[GeluNode], L, fname)
            self.assertEqual(kinds[SoftmaxNode], L, fname)
            self.assertEqual(kinds[TransposeNode], 4 * L + 1, fname)
            self.assertEqual(kinds[GatherNode], 1, fname)
            self.assertEqual(kinds[OneHotNode], 1, fname)
            self.assertEqual(kinds[CastNode], 1, fname)
            self.assertEqual(kinds[SliceNode], 2, fname)
            # 6 Gemm per layer + head Gemm, QK^T and PV, OneHot x token-type
            self.assertEqual(kinds[MatmulNode], 6 * L + 1 + 2 * L + 1, fname)
            # Gemm biases, residual / embedding / mask Adds, mask Sub / Muls, scale Mul
            self.assertEqual(kinds[ScheduledNode], (6 * L + 1) + 2 * L + 2 + 3 + L + L, fname)
            fused = m["style"] != "native"
            self.assertEqual(g.fusion_counts["layernorm"], (1 + 2 * L) if fused else 0)
            self.assertEqual(g.fusion_counts["gelu"], L if fused else 0)
            # scalar -1e4 / 1 / scale -> [S]; ones[1,S,1] pre-broadcast
            self.assertEqual(g.fusion_counts["const_bcast"], 3 + L)
            self.assertEqual(g.split_lowered_count, 1)
            for sn in g.nodes:
                self.assertNotIn(sn.onnx_node.op_type,
                                 ("ReduceMean", "Pow", "Sqrt", "Reciprocal", "Tanh", "Erf"))

    def test_attention_matmuls_batched(self):
        m = self.models["bert_tiny_h64_l2.onnx"]
        g, _ = self.gen(m["path"])
        att = [sn for sn in g.nodes if isinstance(sn, MatmulNode) and sn.batch > 1]
        self.assertEqual(len(att), 2 * m["layers"])
        S, dh = m["seq"], m["hidden"] // m["heads"]
        self.assertEqual({(sn.n, sn.k, sn.m, sn.batch) for sn in att},
                         {(S, dh, S, m["heads"]), (S, S, dh, m["heads"])})

    def test_fusion_off_is_rejected(self):
        # without the pre-pass the first failure is the ones x mask broadcast
        g_path = self.models["bert_tiny_h32_l1.onnx"]["path"]
        with self.assertRaisesRegex(SchedulerError, "broadcast"):
            OnnxGraph(g_path, fuse_patterns=False)


@unittest.skipUnless(ort is not None, "onnxruntime not installed")
class TestNumerics(_Tiny):

    def test_vs_onnxruntime(self):
        """Q8.8 datapath vs float: every kernel op floors (<= 1 LSB = 1/256
        each, on average a -0.5 LSB bias), host ops round (<= 0.5 LSB),
        weights are rounded to 1/256; ~15-25 such steps lie on a logit's
        path in these models.  Observed over 4 models x 3 inputs: max error
        13 LSB (0.049) on logits up to 2.8, mean error <= 4.1 LSB.  Bounds:
        max 32 LSB (0.125), mean 8 LSB — ~2x margin, still far below the
        logit spread that decides the answer span."""
        for fname, m in self.models.items():
            _, cg = self.gen(m["path"])
            sess = ort.InferenceSession(m["path"], providers=["CPUExecutionProvider"])
            names = [o.name for o in sess.get_outputs()]
            for seed in range(3):
                feeds = gbm.random_feeds(m["path"], seed)
                ref = dict(zip(names, sess.run(None, feeds), strict=True))
                sim = cg.simulate({k: v.astype(np.float64) for k, v in feeds.items()})
                for o in ("unstack:0", "unstack:1"):
                    err = np.abs(sim[o] - ref[o])
                    self.assertLessEqual(err.max(), 0.125, (fname, seed, o))
                    self.assertLessEqual(err.mean(), 8 / 256, (fname, seed, o))
                np.testing.assert_array_equal(sim["unique_ids:0"], [7])

    def test_bit_exact_vs_study_emulation(self):
        """The study's op-by-op numpy emulation of the scheduler partition
        (policy "sched": LayerNorm / GELU as their 12 / 8 separate nodes in
        float64) equals the scheduler simulation bit for bit, on every DDR
        tensor both materialise."""
        bs = _study()
        for fname, m in self.models.items():
            if m["opset"] != 12:
                continue                 # the study interpreter speaks opset 12 (bertsquad-12)
            _, cg = self.gen(m["path"])
            bert = bs.Bert(m["path"], bs.POLS["sched"])
            for seed in range(2):
                feeds = gbm.random_feeds(m["path"], seed)
                sim = cg._forward_pass({k: v.astype(np.float64) for k, v in feeds.items()})
                env = bert.run(feeds, "q88")
                common = [sn.output.onnx_name for sn in cg._graph.nodes
                          if sn.output.onnx_name in env]
                self.assertGreater(len(common), 20)
                for n in common:
                    np.testing.assert_array_equal(
                        np.asarray(sim[n], np.float64).reshape(-1),
                        np.asarray(env[n], np.float64).reshape(-1), (fname, seed, n))


@unittest.skipUnless(host_emu.which_cc(), "C compiler not available on host")
class TestGeneratedC(_Tiny):

    def test_host_emulated_run_matches_simulation(self):
        """inference.c + test_inference.c compiled unchanged against software
        kernel models: every output bit-identical to the simulator — host ops
        in place (cacheable buffers, the default) and staged (non-cacheable)."""
        configs = [dict(), dict(cached=False)]
        for fname, m in self.models.items():
            _, cg = self.gen(m["path"])
            for cfg in configs:
                with self.subTest(model=fname, **cfg), tempfile.TemporaryDirectory() as td:
                    rc, out = host_emu.build_and_run(cg, td, **cfg)
                    self.assertEqual(rc, 0, f"{fname}\n{out}")
                    self.assertIn("test_inference PASSED", out)
                    self.assertEqual(host_emu.failures(out), [])

    def test_source_structure(self):
        m = self.models["bert_tiny_h32_l1.onnx"]
        g, cg = self.gen(m["path"])
        src = cg.generate_source()
        self.assertIn("#include <math.h>", src)
        self.assertIn('#  pragma GCC optimize ("fp-contract=off")', src)
        for fn in ("host_softmax", "host_layernorm", "host_lut_map", "host_copy_nd",
                   "host_gather_rows", "host_onehot", "host_cast", "host_load", "host_store",
                   "host_out_done", "host_parallel"):
            self.assertIn(f"static void {fn}(", src)
        self.assertIn("static int host_gelu_tanh_lut(", src)
        self.assertNotIn("host_gelu_erf", src)                 # only the kinds in use
        self.assertIn("static Data_t *s_host_stage = NULL;", src)
        # tables + threads set up in init, torn down in deinit
        init = src[src.index("int inference_init("):src.index("void inference_deinit(void)")]
        self.assertIn("if (host_runtime_init() != 0) { rc = -1; goto fail; }", init)
        self.assertIn("host_runtime_deinit();", src[src.index("void inference_deinit(void)"):])
        rt = src[src.index("static int host_runtime_init(void)"):]
        self.assertIn("if (host_exp_lut_init() != 0) return -1;", rt)
        self.assertRegex(rt, r"if \(host_gelu_tanh_lut\(&s_host_gelu_tanh_lut_[0-9a-f]{8}, "
                             r"0\.04471499\d*, 0\.79788\d*\) != 0\) return -1;")
        self.assertRegex(src, r"s_host_stage = \(Data_t \*\)malloc\(\d+u \* INFERENCE_BYTES_PER_ELEM\);")
        self.assertIn("free(s_host_stage); s_host_stage = NULL;", src)
        # LN parameters stay float32 host constants, not DMA weights
        self.assertRegex(src, r"static const float _host_\w+_gamma\[32\] = \{")
        self.assertNotIn("gamma", " ".join(t.onnx_name for t in g.weight_tensors))
        # a kernel-written source is invalidated before the host op reads it
        ln = src[src.index("host_layernorm(in0"):]
        blk = src[:src.index("host_layernorm(in0")]
        blk = blk[blk.rindex("    {"):]
        self.assertIn("inference_buf_sync_from_device(", blk)
        self.assertLess(blk.index("inference_buf_sync_from_device("), blk.index("in0 = host_in("))
        self.assertIn("host_out_done(", ln[:600])
        cmake = cg.generate_cmake()
        self.assertIn("target_compile_options(inference PRIVATE -ffp-contract=off)", cmake)
        self.assertIn("target_link_libraries(inference PUBLIC m)", cmake)


    def test_event_stream_and_liveness(self):
        for m in self.models.values():
            g, cg = self.gen(m["path"])
            events = cg._compute_event_stream()
            cpu = {e[1] for e in events if e[0] == "cpu"}
            self.assertEqual(cpu, {sn.index for sn in g.nodes
                                   if isinstance(sn, HostNode) and not
                                   (isinstance(sn, SliceNode) and sn.is_view)})
            # every host op runs after its kernel producers have been drained
            pending = set()
            by_idx = {sn.index: sn for sn in g.nodes}
            producer = {sn.output.onnx_name: sn for sn in g.nodes}
            for e in events:
                if e[0] == "start":
                    pending.add(e[1])
                elif e[0] in ("wait", "drain"):
                    pending.discard(e[2])
                elif e[0] == "cpu":
                    for t in by_idx[e[1]].inputs:
                        root = cg._alias_root(t.onnx_name)
                        p = producer.get(root)
                        if p is not None:
                            self.assertNotIn(p.index, pending)
            # tensors sharing a pool slot have disjoint live intervals
            iv = cg._compute_live_intervals()
            layout, _ = cg._compute_pool_layout()
            by_off = collections.defaultdict(list)
            for name, off, _a in layout:
                if name in iv:
                    by_off[off].append(iv[name])
            for ivs in by_off.values():
                ivs.sort()
                for (_s0, e0), (s1, _e1) in zip(ivs, ivs[1:], strict=False):
                    self.assertLess(e0, s1)

    def test_integer_io(self):
        m = self.models["bert_tiny_h32_l1.onnx"]
        g, cg = self.gen(m["path"])
        h = cg.generate_header()
        self.assertIn("Integer tensors (token ids, segment ids, masks, ...) are NOT stored", h)
        self.assertIn("input_ids_0 (int64 [1, 8])", h)
        self.assertIn("(8 elem, shape=[1, 8], raw int64 values)", h)
        t = cg.generate_test()
        self.assertIn(f"p[i] = (Data_t)(i % {gbm.VOCAB}u);", t)   # ids: Gather table rows
        self.assertIn("p[i] = (Data_t)(i % 2u);", t)              # segments: OneHot depth 2
        self.assertIn("got %d expected %d", t)
        self.assertIn("static const uint16_t expected_unique_ids_0[1] = {\n    0x0000\n};", t)
        src = cg.generate_source()
        self.assertIn("memcpy(inference_buf_ptr(unique_ids_0), "
                      "inference_buf_ptr(unique_ids_raw_output_9_0), 1u * INFERENCE_BYTES_PER_ELEM);",
                      src)

    def test_report(self):
        m = self.models["bert_tiny_h64_l2.onnx"]
        g, cg = self.gen(m["path"])
        md = ReportGenerator(graph=g, codegen=cg, model_path=m["path"], out_dir="/tmp",
                             generated_files=[]).render_markdown()
        self.assertIn("**Pattern fusion** — 5 LayerNorm and 2 GELU subgraph(s)", md)
        self.assertIn("**Host-CPU ops**", md)
        self.assertIn("host CPU · rows=64 n=16 · no kernel call", md)     # a Softmax row

    def test_cli(self):
        m = self.models["bert_tiny_erf.onnx"]
        with tempfile.TemporaryDirectory() as td:
            r = subprocess.run([sys.executable, os.path.join(_ROOT, "inference_scheduler.py"),
                                m["path"], "--out-dir", td], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("Gelu", r.stderr)
            with open(os.path.join(td, "src", "inference.c")) as f:
                self.assertIn("host_gelu_erf_lut(", f.read())
            r = subprocess.run([sys.executable, os.path.join(_ROOT, "inference_scheduler.py"),
                                m["path"], "--out-dir", td, "--no-fuse-patterns"],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 1)
            self.assertIn("unsupported graph", r.stderr)


if __name__ == "__main__":
    unittest.main()
