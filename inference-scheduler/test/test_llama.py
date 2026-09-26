"""Llama-family decoders (doc/CHAT_PLAN.md phase 3) on the tiny random Llama
fixture of test/gen_llama_models.py (hidden 64, 2 layers, 4 / 2 heads
(GQA), head_dim 16, FFN 128, vocab 256, context 32):

  * the frontend's entry graphs (src/llama.py): structure, numerics metadata;
  * the scheduler's simulation equals the study's emulation of the shipped
    policy (demo/chat/scripts/llm_study.py pow2+sink+p12+xattn) bit for bit
    — prefill in one call or split over padded bucket calls, greedy decode,
    truncate and re-prefill (the library's use);
  * the generated C of every entry and of the multi-entry project, compiled
    with -Werror against the software kernels, reproduces the simulation
    (cacheable / staged buffers, 1 / 4 host threads), incl. the
    inference_deinit() / inference_init() cycle;
  * the multi-entry project shares weights (packed MatmulKernel copy for
    decode + head, one ConvKernel copy for all prefill buckets) and states;
  * the cache-coherency audit is clean and still catches a dropped sync.
"""

import os
import re
import sys
import tempfile
import unittest

import numpy as np

import gen_llama_models as G
import host_emu
from src.codegen import CodeGenerator
from src.codegen.multi import MultiEntryGenerator
from src.graph import OnnxGraph
from src.llm_nodes import (LlmAttentionNode, LlmDequantNode, LlmEmbedNode, LlmResAddNode,
                           LlmRMSNormNode, LlmSelectRowNode, LlmSiluMulNode)
from src.nodes import MatmulConvNode, MatmulNode
from test_cache_coherency import CoherencyChecker

study = G.study

_TINY = None


def tiny():
    global _TINY
    if _TINY is None:
        _TINY = G.tiny(0)
    return _TINY


def graphs(fe, kinds, **kw):
    out = {}
    for name, (kind, T, wh) in kinds.items():
        out[name] = OnnxGraph(fe.entry(kind, T, with_head=wh), fuse_act=True, s2d_stem=True, **kw)
    return out


class Session:
    """The library's calling convention over simulations of the entries."""

    def __init__(self, cgs, buckets):
        self.cgs = cgs
        self.buckets = sorted(buckets)
        self.states = {}
        for cg in cgs.values():
            for k, v in cg.initial_states().items():
                self.states.setdefault(k, v)
        self.pos = 1

    def _run(self, e, feeds):
        return self.cgs[e]._forward_pass({k: np.asarray(v, np.float64) for k, v in feeds.items()},
                                         states=self.states)

    def prefill(self, toks, split=None):
        i = 0
        chunks = split or []
        if not chunks:
            n = len(toks)
            while n:
                fit = [b for b in self.buckets if b <= n]
                B = fit[-1] if fit else self.buckets[0]
                chunks.append(min(n, B))
                n -= chunks[-1]
        for k in chunks:
            B = min(b for b in self.buckets if b >= k)
            ids = np.zeros(B)
            ids[:k] = toks[i:i + k]
            e = f"prefill_{B}"
            self._run(e, {f"{e}.ids": ids, f"{e}.pos": [self.pos], f"{e}.n": [k]})
            self.pos += k
            i += k
        return self._run("head", {})["head.logits"].reshape(-1)

    def decode(self, tok):
        out = self._run("decode", {"decode.ids": [tok], "decode.pos": [self.pos]})
        self.pos += 1
        return out["decode.logits"].reshape(-1)


class TestFrontend(unittest.TestCase):

    def test_decode_graph(self):
        cfg, W, formats, fe = tiny()
        g = OnnxGraph(fe.entry("decode"))
        kinds = [type(sn).__name__ for sn in g.nodes]
        L = cfg["num_hidden_layers"]
        self.assertEqual(kinds.count("MatmulNode"), 7 * L + 1)
        self.assertEqual(kinds.count("LlmAttentionNode"), L)
        self.assertEqual(kinds.count("LlmRMSNormNode"), 2 * L + 1)
        self.assertEqual(kinds.count("LlmResAddNode"), 2 * L)
        self.assertEqual(kinds.count("LlmSiluMulNode"), L)
        self.assertEqual(kinds.count("LlmEmbedNode"), 1)
        self.assertEqual(kinds.count("LlmDequantNode"), 1)
        self.assertEqual([t.host for t in g.input_tensors], ["i32", "i32"])
        self.assertEqual([t.host for t in g.output_tensors], ["f32"])
        self.assertEqual(len(g.state_tensors), 2 * L)
        self.assertTrue(all(t.host == "i16" and t.init_data is not None for t in g.state_tensors))
        # decode linears: N = 1 -> MatmulKernel, packed constant B
        self.assertTrue(all(sn.b_packed for sn in g.nodes if isinstance(sn, MatmulNode)))
        # every MatMul weight encoded at its rank-1 exponent, none saturated
        self.assertTrue(all(sn.inputs[1].wexp is not None for sn in g.nodes
                            if isinstance(sn, MatmulNode)))
        self.assertEqual(sum(g.weights_saturated.values()), 0)

    def test_residual_stream_is_float_host(self):
        _cfg, _W, _f, fe = tiny()
        g = OnnxGraph(fe.entry("decode"))
        for sn in g.nodes:
            if isinstance(sn, LlmResAddNode):
                self.assertEqual(sn.output.host, "f32")
                self.assertEqual(sn.inputs[0].host, "f32")
            if isinstance(sn, LlmRMSNormNode):
                self.assertEqual(sn.inputs[0].host, "f32")
                self.assertIsNone(sn.output.host)
        self.assertTrue(all(t.host == "f32" for t in g.host_tensors))

    def test_prefill_uses_conv_and_head_state(self):
        _cfg, _W, _f, fe = tiny()
        g = OnnxGraph(fe.entry("prefill", 16))
        self.assertTrue(any(isinstance(sn, MatmulConvNode) for sn in g.nodes))
        last = g.nodes[-1]
        self.assertIsInstance(last, LlmSelectRowNode)
        self.assertTrue(last.output.is_state)
        self.assertEqual([t.onnx_name for t in g.output_tensors], [])


class TestSimVsStudy(unittest.TestCase):
    """Bit-exact against llm_study.Model (pow2+sink+p12+xattn)."""

    @classmethod
    def setUpClass(cls):
        cfg, W, formats, fe = tiny()
        gs = graphs(fe, {"decode": ("decode", 1, False), "prefill_8": ("prefill", 8, False),
                         "prefill_16": ("prefill", 16, False), "head": ("head", 1, False)})
        cls.cgs = {n: CodeGenerator(g, model_path=n) for n, g in gs.items()}
        cls.sm, cls.sc = G.study_model(cfg, W, formats)

    def _study_run(self, prompt, steps):
        s = study.Seq(self.sc, 64)
        lg = [self.sm.forward([s], [np.concatenate([[1], prompt])])[0][-1]]
        for _ in range(steps):
            lg.append(self.sm.forward([s], [np.array([int(np.argmax(lg[-1]))])])[0][-1])
        return lg

    def _sched_run(self, prompt, steps, split=None):
        ses = Session(self.cgs, [8, 16])
        lg = [ses.prefill(list(prompt), split)]
        for _ in range(steps):
            lg.append(ses.decode(int(np.argmax(lg[-1]))))
        return lg

    def test_prefill_and_decode(self):
        rng = np.random.default_rng(11)
        for n, split in ((11, None), (16, [16]), (5, [5]), (13, [3, 8, 2])):
            prompt = rng.integers(2, 256, n)
            a = self._sched_run(prompt, 5, split)
            b = self._study_run(prompt, 5)
            with self.subTest(n=n, split=split):
                for i, (x, y) in enumerate(zip(a, b)):
                    np.testing.assert_array_equal(x, y, err_msg=f"step {i}")

    def test_truncate_and_reprefill(self):
        """A second turn: truncate to the common prefix, prefill the new
        tokens — equal to prefilling the whole conversation at once."""
        rng = np.random.default_rng(12)
        p1, p2 = rng.integers(2, 256, 9), rng.integers(2, 256, 6)
        ses = Session(self.cgs, [8, 16])
        ses.prefill(list(p1))
        for _ in range(3):
            ses.decode(7)
        ses.pos = 1 + len(p1)                     # llm_truncate(1 + len(p1))
        a = ses.prefill(list(p2))
        b = self._study_run(np.concatenate([p1, p2]), 0)[0]
        np.testing.assert_array_equal(a, b)


class TestHostEmulation(unittest.TestCase):
    """Generated C (-Werror) against the software kernels == simulation."""

    def test_entries(self):
        _cfg, _W, _f, fe = tiny()
        gs = graphs(fe, {"decode": ("decode", 1, False), "prefill8": ("prefill", 8, True),
                         "prefill16": ("prefill", 16, True), "head": ("head", 1, False)})
        for name, g in gs.items():
            cg = CodeGenerator(g, model_path=f"llama_tiny_{name}.onnx")
            for cached, threads in ((True, 4), (False, 1), (True, 3)):
                with self.subTest(entry=name, cached=cached, threads=threads), \
                        tempfile.TemporaryDirectory() as td:
                    rc, out = host_emu.build_and_run(cg, td, cached=cached, threads=threads,
                                                     min_elems=1)
                    self.assertEqual(rc, 0, out[-3000:])
                    self.assertIn("test_inference PASSED", out)

    def test_multi_entry_project(self):
        _cfg, _W, _f, fe = tiny()
        gs = graphs(fe, {"decode": ("decode", 1, False), "prefill_8": ("prefill", 8, False),
                         "prefill_16": ("prefill", 16, False), "head": ("head", 1, False)},
                    matmul_on_conv="always")
        mg = MultiEntryGenerator(list(gs.items()), "llama_tiny")
        s = mg.summary()
        # decode + head share the packed copy; both prefill buckets one conv copy
        n_lin = 7 * 2
        self.assertEqual(len(s["renamed_weights"]), n_lin)
        self.assertTrue(all(v == [k + "@1"] for k, v in s["renamed_weights"].items()))
        self.assertEqual(s["weights"], 2 * n_lin + 1)
        self.assertEqual(sorted(t.onnx_name for t in mg.combined.state_tensors),
                         sorted([f"kv.{w}.l{l}" for w in "kv" for l in range(2)] + ["h_last"]))
        src = mg.generate_source()
        for e in ("decode", "prefill_8", "prefill_16", "head"):
            self.assertIn(f"\nvoid inference_run_{e}(", src)
        self.assertIn("XMatmulkernel_Release(&s_matmulkernel)", src)
        for cached in (True, False):
            with self.subTest(cached=cached), tempfile.TemporaryDirectory() as td:
                rc, out = host_emu.build_and_run(mg, td, cached=cached, threads=4, min_elems=1)
                self.assertEqual(rc, 0, out[-3000:])
                self.assertIn("re-open: ok", out)
                self.assertIn("test_inference PASSED", out)

    def test_kw_unification(self):
        """Pinned kernel widths make the prefill buckets' conv images equal."""
        _cfg, _W, _f, fe = tiny()
        big = OnnxGraph(fe.entry("prefill", 16), matmul_on_conv="always")
        kw = {sn.inputs[1].onnx_name: sn.kw for sn in big.nodes if isinstance(sn, MatmulConvNode)}
        forced = {k: 2 for k in kw}
        small = OnnxGraph(fe.entry("prefill", 8), matmul_on_conv="always", matmul_conv_kw=forced)
        self.assertTrue(all(sn.kw == 2 for sn in small.nodes if isinstance(sn, MatmulConvNode)))


class TestCoherency(unittest.TestCase):

    def test_entries_clean_and_mutations_caught(self):
        _cfg, _W, _f, fe = tiny()
        for name, (kind, T, wh) in {"decode": ("decode", 1, False),
                                    "prefill": ("prefill", 16, True),
                                    "head": ("head", 1, False)}.items():
            cg = CodeGenerator(OnnxGraph(fe.entry(kind, T, with_head=wh), fuse_act=True),
                               model_path=name)
            src = cg.generate_source()
            with self.subTest(entry=name):
                self.assertEqual(CoherencyChecker(cg, src).check(), [])
                m = re.search(r"\n *inference_buf_sync_from_device\((\w+)\);  /\* written by a "
                              r"kernel \*/", src)
                errs = CoherencyChecker(cg, src.replace(m.group(0), "", 1)).check()
                self.assertTrue(any(f"CPU reads '{m.group(1)}'" in e for e in errs), errs)
                m = re.search(r"\n *host_out_done\((\w+),[^\n]*", src)
                errs = CoherencyChecker(cg, src.replace(m.group(0), "", 1)).check()
                self.assertTrue(any("CPU-dirty" in e for e in errs), errs)
                # host-memory tensors never reach a sync or a kernel call
                host = {t.c_name for t in cg._graph.host_tensors + cg._graph.state_tensors
                        + cg._graph.input_tensors + cg._graph.output_tensors if t.is_host}
                for m in re.finditer(r"(inference_buf_sync_\w+|run_\w+|host_in|host_out)\(\s*(\w+)",
                                     src):
                    self.assertNotIn(m.group(2), host, m.group(0))


if __name__ == "__main__":
    unittest.main()
