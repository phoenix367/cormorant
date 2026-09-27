"""Llama-family decoders (doc/plans/CHAT_PLAN.md phase 3) on the tiny random Llama
fixture of test/gen_llama_models.py (hidden 64, 2 layers, 4 / 2 heads
(GQA), head_dim 16, FFN 128, vocab 256, context 32):

  * the frontend's entry graphs (src/llama.py): structure, numerics metadata;
  * the scheduler's simulation equals the study's emulation of the shipped
    policy bit for bit — demo/chat/scripts/llm_study.py pow2+sink+p12+mix
    (FPGA prefill attention, xattn decode; the frontend's default) and
    pow2+sink+p12+xattn (prefill_attn="host") — prefill in one call or split
    over padded bucket calls, greedy decode, truncate and re-prefill, a second
    turn prefilled after decode steps (the library's use);
  * the generated C of every entry and of the multi-entry project, compiled
    with -Werror against the software kernels, reproduces the simulation
    (cacheable / staged buffers, 1 / 4 host threads, and with separate CPU /
    DDR copies so that every cache sync — the KV caches' row flushes
    included — matters), incl. the inference_deinit() / inference_init()
    cycle, and over a library-style call sequence (prefill split over
    buckets, decode, a second turn, truncate);
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

import shutil
import subprocess

import gen_llama_models as G
import host_emu
from src.codegen import CodeGenerator
from src.codegen.multi import MultiEntryGenerator
from src.graph import OnnxGraph
from src.llm_entries import entry_graphs
from src.llm_nodes import (LlmAttentionNode, LlmAttnConvNode, LlmAttnMergeNode,
                           LlmAttnPrepNode, LlmAttnSoftmaxNode, LlmDequantNode, LlmEmbedNode,
                           LlmResAddNode, LlmRMSNormNode, LlmSelectRowNode, LlmSiluMulNode)
from src.nodes import MatmulConvNode, MatmulNode
from test_cache_coherency import CoherencyChecker

study = G.study

_TINY = None
_TINY_HOST = None


def tiny():
    global _TINY
    if _TINY is None:
        _TINY = G.tiny(0)
    return _TINY


def tiny_host():
    """The tiny fixture with prefill_attn="host" (phase 3's xattn everywhere)."""
    global _TINY_HOST
    if _TINY_HOST is None:
        cfg, W, formats, _fe = tiny()
        _TINY_HOST = (cfg, W, formats, G.frontend(cfg, W, formats, prefill_attn="host"))
    return _TINY_HOST


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


def study_steps(sm, sc, script, cap=64):
    """The study's emulation of a library call script: ("prefill", tokens)
    (the first one starts with the sink token 1), ("decode", token),
    ("truncate", n) -> the logits after every prefill / decode."""
    seq = study.Seq(sc, cap)
    out = []
    for op, arg in script:
        if op == "prefill":
            out.append(sm.forward([seq], [np.asarray(arg, np.int64)])[0][-1])
        elif op == "decode":
            out.append(sm.forward([seq], [np.array([arg])], phase="decode")[0][-1])
        else:
            seq.n = arg
    return out


def sched_steps(ses, script):
    """The same script over the scheduler simulation (Session)."""
    out = []
    for op, arg in script:
        if op == "prefill":
            toks = list(arg[1:]) if ses.pos == 1 and arg[0] == 1 else list(arg)
            out.append(ses.prefill(toks))
        elif op == "decode":
            out.append(ses.decode(arg))
        else:
            ses.pos = arg
    return out


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
        # the KV caches: DMA states in the pool (the FPGA prefill attention reads
        # them), group-major [KV][C][HD], the sink row as their initial value
        self.assertTrue(all(t.host is None and t.init_data is not None
                            and t.group_layout == (cfg["num_key_value_heads"], cfg["head_dim"])
                            for t in g.state_tensors))
        gh = OnnxGraph(tiny_host()[3].entry("decode"))
        self.assertTrue(all(t.host == "i16" and t.group_layout for t in gh.state_tensors))
        # decode linears: N = 1 -> MatmulKernel: the GEMV streaming path where
        # it applies (matmul_gemv.py; the k / v projections of this tiny model
        # have m = 32 < 64 in the plain image), else the packed constant B
        from src.matmul_gemv import ineligible_reason
        mms = [sn for sn in g.nodes if isinstance(sn, MatmulNode)]
        for sn in mms:
            self.assertNotEqual(bool(sn.gemv_kw), bool(sn.b_packed), sn.onnx_node.name)
            self.assertEqual(bool(sn.gemv_kw), ineligible_reason(sn, 1) is None,
                             sn.onnx_node.name)
        self.assertTrue(any(sn.gemv_kw for sn in mms))
        self.assertTrue(any(sn.b_packed for sn in mms))
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

    def test_prefill_fpga_attention(self):
        """Per layer: LlmAttnPrep, per KV group q.K^T / softmax / P.V, each
        softmax issued behind a ConvKernel call it overlaps, LlmAttnMerge; no
        host attention in prefill."""
        cfg, _W, _f, fe = tiny()
        g = OnnxGraph(fe.entry("prefill", 16))
        L, KV = cfg["num_hidden_layers"], cfg["num_key_value_heads"]
        kinds = [type(sn).__name__ for sn in g.nodes]
        self.assertNotIn("LlmAttentionNode", kinds)
        self.assertEqual(kinds.count("LlmAttnPrepNode"), L)
        self.assertEqual(kinds.count("LlmAttnConvNode"), 2 * KV * L)
        self.assertEqual(kinds.count("LlmAttnSoftmaxNode"), KV * L)
        self.assertEqual(kinds.count("LlmAttnMergeNode"), L)
        i = kinds.index("LlmAttnPrepNode")
        seq = [(type(sn).__name__, getattr(sn, "kind", ""), getattr(sn, "group", None))
               for sn in g.nodes[i:i + 2 + 3 * KV]]
        self.assertEqual(seq, [("LlmAttnPrepNode", "", None), ("LlmAttnConvNode", "qk", 0),
                               ("LlmAttnConvNode", "qk", 1), ("LlmAttnSoftmaxNode", "", 0),
                               ("LlmAttnConvNode", "pv", 0), ("LlmAttnSoftmaxNode", "", 1),
                               ("LlmAttnConvNode", "pv", 1), ("LlmAttnMergeNode", "", None)])
        cg = CodeGenerator(g, model_path="p.onnx")
        ev = [e for e in cg._compute_event_stream() if e[0] in ("start", "cpu", "wait")]
        idx = {sn.index: sn for sn in g.nodes}
        sm0 = next(sn.index for sn in g.nodes if isinstance(sn, LlmAttnSoftmaxNode))
        qk1 = sm0 - 1
        # softmax 0 runs while q.K^T 1 is in flight: no wait between its start and it
        a = ev.index(("start", qk1))
        b = ev.index(("cpu", sm0))
        self.assertTrue(all(e[0] != "wait" for e in ev[a:b]), ev[a:b + 1])
        self.assertEqual(idx[qk1].kind, "qk")
        # the host mode keeps LlmAttention (phase 3)
        gh = OnnxGraph(tiny_host()[3].entry("prefill", 16))
        self.assertIn("LlmAttentionNode", [type(sn).__name__ for sn in gh.nodes])


class TestSimVsStudy(unittest.TestCase):
    """Bit-exact against llm_study.Model (pow2+sink+p12+mix: FPGA prefill
    attention, xattn decode — the frontend's default)."""
    TINY = staticmethod(tiny)
    POLICY = G.MIX_POLICY

    @classmethod
    def setUpClass(cls):
        cfg, W, formats, fe = cls.TINY()
        gs = graphs(fe, {"decode": ("decode", 1, False), "prefill_8": ("prefill", 8, False),
                         "prefill_16": ("prefill", 16, False), "head": ("head", 1, False)})
        cls.cgs = {n: CodeGenerator(g, model_path=n) for n, g in gs.items()}
        cls.sm, cls.sc = G.study_model(cfg, W, formats, policy=cls.POLICY)

    def _study_run(self, prompt, steps):
        s = study.Seq(self.sc, 64)
        lg = [self.sm.forward([s], [np.concatenate([[1], prompt])])[0][-1]]
        for _ in range(steps):
            lg.append(self.sm.forward([s], [np.array([int(np.argmax(lg[-1]))])],
                                      phase="decode")[0][-1])
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

    def test_second_turn(self):
        """A chat's second turn: prefill, greedy decode steps (their cache rows
        come from decode — xattn under the mixed policy), then the next turn's
        tokens prefilled at position > 1 on top of them, decode again; and a
        truncate into the answer followed by a re-prefill."""
        rng = np.random.default_rng(13)
        p1, p2, p3 = (rng.integers(2, 256, k) for k in (10, 7, 4))
        ses = Session(self.cgs, [8, 16])
        steps, toks = [], []
        a = ses.prefill(list(p1))
        steps.append(("prefill", [1] + list(p1)))
        for _ in range(4):
            t = int(np.argmax(a))
            toks.append(t)
            steps.append(("decode", t))
            a = ses.decode(t)
        steps.append(("prefill", list(p2)))
        a = ses.prefill(list(p2))
        for _ in range(3):
            t = int(np.argmax(a))
            steps.append(("decode", t))
            a = ses.decode(t)
        steps += [("truncate", 1 + len(p1) + 2), ("prefill", list(p3)), ("decode", 5)]
        b = study_steps(self.sm, self.sc, steps)
        a = sched_steps(Session(self.cgs, [8, 16]), steps)
        self.assertEqual(len(a), len(b))
        for i, (x, y) in enumerate(zip(a, b)):
            np.testing.assert_array_equal(x, y, err_msg=f"step {i} {steps[i]}")


class TestSimVsStudyHost(TestSimVsStudy):
    """prefill_attn="host": bit-exact against pow2+sink+p12+xattn (phase 3)."""
    TINY = staticmethod(tiny_host)
    POLICY = G.POLICY


class TestOtherShapes(unittest.TestCase):
    """Llama-generic: the SmolLM2-360M layer shape (hidden 960, 15 / 5 heads,
    head_dim 64, FFN 2560 — its K = 2560 down-projection needs kw >= 3 on
    ConvKernel) and plain multi-head attention, 2 random layers each."""

    def _run(self, cfg, seed):
        W = G.tiny_weights(cfg, seed)
        formats = G.calibrate(cfg, W, n_seq=2, length=20)
        fe = G.frontend(cfg, W, formats, ctx=48, name="shape")
        gs = graphs(fe, {"decode": ("decode", 1, False), "prefill_16": ("prefill", 16, False),
                         "head": ("head", 1, False)})
        cgs = {n: CodeGenerator(g, model_path=n) for n, g in gs.items()}
        sm, sc = G.study_model(cfg, W, formats, policy=G.MIX_POLICY)
        rng = np.random.default_rng(seed)
        prompt = rng.integers(2, cfg["vocab_size"], 12)
        ses = Session(cgs, [16])
        a = [ses.prefill(list(prompt))]
        s = study.Seq(sc, 32)
        b = [sm.forward([s], [np.concatenate([[1], prompt])])[0][-1]]
        for _ in range(3):
            tok = int(np.argmax(a[-1]))
            a.append(ses.decode(tok))
            b.append(sm.forward([s], [np.array([tok])], phase="decode")[0][-1])
        for i, (x, y) in enumerate(zip(a, b)):
            np.testing.assert_array_equal(x, y, err_msg=f"step {i}")
        return gs

    def test_smollm2_360m_layer_shape(self):
        cfg = dict(G.TINY, hidden_size=960, num_attention_heads=15, num_key_value_heads=5,
                   head_dim=64, intermediate_size=2560, vocab_size=1024, rope_theta=100000.0)
        gs = self._run(cfg, 3)
        down = [sn for sn in gs["prefill_16"].nodes
                if isinstance(sn, MatmulConvNode) and sn.k == 2560]
        self.assertEqual(len(down), 2)
        self.assertTrue(all(sn.kw >= 3 and sn.in_ch <= 1024 for sn in down))
        attn = [sn for sn in gs["prefill_16"].nodes if isinstance(sn, LlmAttnConvNode)]
        self.assertEqual(len(attn), 2 * 5 * 2)
        # head_dim 64: the q image may use kw > 1 (the cost model's choice)
        self.assertTrue(all(sn.kw in (1, 2, 4) for sn in attn))
        cg = CodeGenerator(gs["prefill_16"], model_path="shape_prefill_16.onnx")
        with tempfile.TemporaryDirectory() as td:
            rc, out = host_emu.build_and_run(cg, td, threads=4, incoherent=True)
            self.assertEqual(rc, 0, out[-3000:])
            self.assertIn("test_inference PASSED", out)

    def test_multi_head_attention(self):
        self._run(dict(G.TINY, num_key_value_heads=4), 4)


class TestHostEmulation(unittest.TestCase):
    """Generated C (-Werror) against the software kernels == simulation."""

    def test_entries(self):
        for fx in (tiny, tiny_host):
            _cfg, _W, _f, fe = fx()
            gs = graphs(fe, {"decode": ("decode", 1, False), "prefill8": ("prefill", 8, True),
                             "prefill16": ("prefill", 16, True), "head": ("head", 1, False)})
            for name, g in gs.items():
                cg = CodeGenerator(g, model_path=f"llama_tiny_{name}.onnx")
                for cached, threads, inc in ((True, 4, False), (False, 1, False),
                                             (True, 3, True)):
                    with self.subTest(fixture=fx.__name__, entry=name, cached=cached,
                                      threads=threads, incoherent=inc), \
                            tempfile.TemporaryDirectory() as td:
                        rc, out = host_emu.build_and_run(cg, td, cached=cached, threads=threads,
                                                         min_elems=1, incoherent=inc)
                        self.assertEqual(rc, 0, out[-3000:])
                        self.assertIn("test_inference PASSED", out)

    def test_multi_entry_project(self):
        _cfg, _W, _f, fe = tiny()
        kinds = {"decode": ("decode", 1, False), "prefill_8": ("prefill", 8, False),
                 "prefill_16": ("prefill", 16, False), "head": ("head", 1, False)}
        models = {n: fe.entry(kind, T, with_head=wh) for n, (kind, T, wh) in kinds.items()}
        entries = entry_graphs(models, matmul_on_conv="always")
        gs = dict(entries)
        # entry_graphs consumes the models and shares the entries' weight
        # arrays: one copy per weight in memory
        self.assertEqual(models, {})
        w_dec = {t.onnx_name: t for t in gs["decode"].weight_tensors}
        w_pre = {t.onnx_name: t for t in gs["prefill_16"].weight_tensors}
        shared = [n for n in w_dec if n in w_pre and w_dec[n].data is not None
                  and w_dec[n].data.nbytes >= 4096]
        self.assertTrue(shared)
        self.assertTrue(all(w_dec[n].data is w_pre[n].data for n in shared))
        self.assertEqual([n for n, _ in entries], ["decode", "prefill_8", "prefill_16", "head"])
        mg = MultiEntryGenerator(entries, "llama_tiny")
        s = mg.summary()
        # Both prefill buckets read one conv image of each linear; decode reads
        # it too through MatmulKernel's GEMV path (src/llm_entries.py) — one
        # copy.  A decode linear GEMV cannot take keeps the packed image, and
        # the prefill copy is renamed <name>@1.  decode and head share the LM
        # head.
        n_lin = 7 * 2
        conv_w = {sn.inputs[1].onnx_name: sn.kw for sn in gs["prefill_16"].nodes
                  if isinstance(sn, MatmulConvNode)}
        dec = {sn.inputs[1].onnx_name: sn for sn in gs["decode"].nodes
               if isinstance(sn, MatmulNode)}
        self.assertEqual(len(conv_w), n_lin)
        self.assertTrue(all(kw in (1, 2, 4, 8) for kw in conv_w.values()))
        for w, sn in dec.items():
            if w in conv_w and sn.gemv_kw:
                self.assertEqual(sn.gemv_kw, conv_w[w], w)
        packed = sorted(w for w, sn in dec.items() if w in conv_w and not sn.gemv_kw)
        self.assertLess(len(packed), n_lin // 2)
        self.assertEqual(sorted(s["renamed_weights"]), packed)
        self.assertTrue(all(v == [k + "@1"] for k, v in s["renamed_weights"].items()))
        self.assertEqual(s["weights"], n_lin + 1 + len(packed))
        self.assertEqual(sorted(t.onnx_name for t in mg.combined.state_tensors),
                         sorted([f"kv.{w}.l{l}" for w in "kv" for l in range(2)] + ["h_last"]))
        src = mg.generate_source()
        for e in ("decode", "prefill_8", "prefill_16", "head"):
            self.assertIn(f"\nvoid inference_run_{e}(", src)
        self.assertIn("XMatmulkernel_Release(&s_matmulkernel)", src)
        # the KV caches: DMA states in the pool, between the weights and the
        # intermediates region
        self.assertEqual(s["dma_state_bytes"], 2 * 2 * 32 * 2 * 16 * 2)
        for cached, inc in ((True, False), (False, False), (True, True)):
            with self.subTest(cached=cached, incoherent=inc), tempfile.TemporaryDirectory() as td:
                rc, out = host_emu.build_and_run(mg, td, cached=cached, threads=4, min_elems=1,
                                                 incoherent=inc)
                self.assertEqual(rc, 0, out[-3000:])
                self.assertIn("re-open: ok", out)
                self.assertIn("test_inference PASSED", out)

    def test_library_sequence(self):
        """The multi-entry project driven like llm_api.c — prefill split over
        the buckets, head, decode steps, a second turn on top of the decoded
        rows, a truncate — compiled against the software kernels with
        separate CPU / DDR copies (every cache flush matters): every logits
        vector equals the simulation of the same calls."""
        for fx in (tiny, tiny_host):
            _cfg, _W, _f, fe = fx()
            gs = graphs(fe, {"decode": ("decode", 1, False), "prefill_8": ("prefill", 8, False),
                             "prefill_16": ("prefill", 16, False), "head": ("head", 1, False)})
            mg = MultiEntryGenerator(list(gs.items()), "llama_tiny")
            rng = np.random.default_rng(21)
            calls = []                    # (entry, ids, pos, n)
            pos = 1

            def prefill(toks):
                nonlocal pos
                i = 0
                for k, B in ((min(8, len(toks) - j), 8 if len(toks) - j <= 8 else 16)
                             for j in range(0, len(toks), 16)):
                    k = min(len(toks) - i, B)
                    calls.append((f"prefill_{B}", list(toks[i:i + k]) + [0] * (B - k), pos, k))
                    pos += k
                    i += k
                calls.append(("head", [], 0, 0))
            prefill(rng.integers(2, 256, 11))
            for _ in range(3):
                calls.append(("decode", [int(rng.integers(2, 256))], pos, 1))
                pos += 1
            prefill(rng.integers(2, 256, 20))          # second turn: 16 + 4 rows
            calls.append(("decode", [7], pos, 1))
            pos = 6                                     # truncate
            prefill(rng.integers(2, 256, 5))
            calls.append(("decode", [9], pos, 1))
            states = mg.initial_states()
            want = []
            for e, ids, p, n in calls:
                cg = mg.cgs[e]
                feeds = {} if e == "head" else {f"{e}.ids": ids, f"{e}.pos": [p]}
                if e.startswith("prefill"):
                    feeds[f"{e}.n"] = [n]
                out = cg._forward_pass({k: np.asarray(v, np.float64) for k, v in feeds.items()},
                                       states=states)
                if e in ("head", "decode"):
                    want.append(np.asarray(out[f"{e}.logits"], np.float32).reshape(-1))
            with self.subTest(fixture=fx.__name__), tempfile.TemporaryDirectory() as td:
                got = run_calls(mg, td, calls, incoherent=True)
                self.assertEqual(len(got), len(want))
                for i, (x, y) in enumerate(zip(got, want)):
                    np.testing.assert_array_equal(x.view(np.uint32), y.view(np.uint32),
                                                  err_msg=f"logits {i}")

    def test_kw_unification(self):
        """Pinned kernel widths make the prefill buckets' conv images equal."""
        _cfg, _W, _f, fe = tiny()
        big = OnnxGraph(fe.entry("prefill", 16), matmul_on_conv="always")
        kw = {sn.inputs[1].onnx_name: sn.kw for sn in big.nodes if isinstance(sn, MatmulConvNode)}
        forced = {k: 2 for k in kw}
        small = OnnxGraph(fe.entry("prefill", 8), matmul_on_conv="always", matmul_conv_kw=forced)
        self.assertTrue(all(sn.kw == 2 for sn in small.nodes if isinstance(sn, MatmulConvNode)))


def run_calls(mg, workdir, calls, incoherent=False):
    """Build mg's project against the software kernels with a driver that
    runs ``calls`` [(entry, ids, pos, n)] in order; returns the logits of
    every head / decode call (float32)."""
    from src._conv_hw_config import CONV_TILE_IC
    from src._matmul_hw_config import MATMUL_TILE_M
    inc, src, emu = (os.path.join(workdir, d) for d in ("include", "src", "emu"))
    for d in (inc, src, emu):
        os.makedirs(d, exist_ok=True)
    files = {os.path.join(inc, "inference.h"): mg.generate_header(),
             os.path.join(src, "inference.c"): mg.generate_source(),
             os.path.join(emu, "inference_buf_emu.c"): host_emu.buf_emu_source(),
             os.path.join(emu, "emu_common.h"): host_emu._COMMON,
             os.path.join(emu, "xvectoropkernel.h"): host_emu._VOP,
             os.path.join(emu, "xmatmulkernel.h"): host_emu._MM,
             os.path.join(emu, "xconvkernel.h"): host_emu._CONV}
    V = mg.entries[0][1].output_tensors[0].numel
    body = []
    for k, (e, ids, p, n) in enumerate(calls):
        if e == "head":
            body.append("    inference_run_head(lg); dump(lg);")
            continue
        body.append(f"    {{ static const int32_t ids{k}[] = {{ {', '.join(str(int(x)) for x in ids)} }};"
                    f" const int32_t p = {p}, n = {n};")
        if e == "decode":
            body.append(f"      inference_run_decode(ids{k}, &p, lg); dump(lg); (void)n; }}")
        else:
            body.append(f"      inference_run_{e}(ids{k}, &p, &n); }}")
    files[os.path.join(src, "drive.c")] = (
        "#include <stdio.h>\n#include <stdint.h>\n#include \"inference.h\"\n"
        f"static float lg[{V}];\n"
        f"static void dump(const float *x) {{ fwrite(x, sizeof(float), {V}u, stdout); }}\n"
        "int main(void)\n{\n    if (inference_init(" + ", ".join(
            f'"{kd.uio_default}"' for kd in mg._active_kernels) + ") != 0) return 1;\n"
        + "\n".join(body) + "\n    inference_deinit();\n    return 0;\n}\n")
    for path, text in files.items():
        with open(path, "w") as f:
            f.write(text)
    shutil.copy(os.path.join(host_emu._ROOT, "runtime", "inference_prof.h"), inc)
    for name, tb in mg.host_table_files():
        os.makedirs(os.path.join(workdir, "weights"), exist_ok=True)
        with open(os.path.join(workdir, "weights", f"{name}.dat"), "wb") as f:
            f.write(tb.dat_bytes())
    for t in mg.large_weight_tensors:
        os.makedirs(os.path.join(workdir, "weights"), exist_ok=True)
        with open(os.path.join(workdir, "weights", f"{t.c_name}.dat"), "wb") as f:
            f.write(mg.generate_weight_dat(t))
    exe = os.path.join(workdir, "drive")
    r = subprocess.run([host_emu.which_cc(), "-std=gnu99", "-O2", "-Wall", "-Wextra", "-Werror",
                        "-Wno-unused-function", "-pthread", f"-DEMU_TILE_M={MATMUL_TILE_M}",
                        f"-DEMU_CONV_TILE_IC={CONV_TILE_IC}",
                        *(["-DEMU_INCOHERENT"] if incoherent else []),
                        f'-DINFERENCE_WEIGHTS_DIR="{workdir}"', "-I", inc, "-I", emu,
                        os.path.join(src, "inference.c"), os.path.join(emu, "inference_buf_emu.c"),
                        os.path.join(src, "drive.c"), "-lm", "-o", exe],
                       capture_output=True, text=True)
    if r.returncode:
        raise AssertionError("compile failed:\n" + r.stderr[-4000:])
    r = subprocess.run([exe], capture_output=True, timeout=600, cwd=workdir)
    if r.returncode:
        raise AssertionError(f"drive failed: {r.stderr[-2000:]}")
    return list(np.frombuffer(r.stdout, np.float32).reshape(-1, V))


class TestCli(unittest.TestCase):

    def test_multi_entry_cli(self):
        """inference_scheduler.py --entry NAME=MODEL.onnx ... writes one project
        with one run function per entry; a model plus --entry is an error."""
        from inference_scheduler import main
        _cfg, _W, _f, fe = tiny()
        import onnx
        with tempfile.TemporaryDirectory() as td:
            paths = {}
            for name, (kind, T) in {"decode": ("decode", 1), "head": ("head", 1)}.items():
                paths[name] = os.path.join(td, f"{name}.onnx")
                onnx.save(fe.entry(kind, T), paths[name])
            out = os.path.join(td, "proj")
            rc = main(["--entry", f"decode={paths['decode']}", "--entry", f"head={paths['head']}",
                       "--out-dir", out])
            self.assertEqual(rc, 0)
            with open(os.path.join(out, "include", "inference.h")) as f:
                h = f.read()
            self.assertIn("void inference_run_decode(", h)
            self.assertIn("void inference_run_head(", h)
            self.assertIn("#define INFERENCE_NUM_ENTRIES  2u", h)
            for rel in ("CMakeLists.txt", "src/inference.c", "src/inference_buf.c",
                        "test/test_inference.c", "driver/README.md"):
                self.assertTrue(os.path.exists(os.path.join(out, rel)), rel)
            self.assertEqual(main([paths["decode"], "--entry", f"head={paths['head']}"]), 1)
            self.assertEqual(main(["--entry", "nonsense"]), 1)


class TestCoherency(unittest.TestCase):

    def test_multi_entry_project(self):
        """Every entry of the multi-entry project (compact codegen: the run
        functions' noinline parts inlined) is clean; the KV caches (DMA
        states) start dirty, so a prefill without its prep op's
        llm_cache_flush is caught."""
        _cfg, _W, _f, fe = tiny()
        gs = graphs(fe, {"decode": ("decode", 1, False), "prefill_8": ("prefill", 8, False),
                         "prefill_16": ("prefill", 16, False), "head": ("head", 1, False)})
        mg = MultiEntryGenerator(list(gs.items()), "llama_tiny")
        src = mg.generate_source()
        for name in mg.cgs:
            with self.subTest(entry=name):
                chk = CoherencyChecker(mg.cgs[name], src, run_name=f"inference_run_{name}")
                self.assertEqual(chk.check(), [])
        self.assertIn("inference_run_prefill_16_part1(", src)
        start = src.index("static void __attribute__((noinline)) inference_run_prefill_16_part0(")
        for cache in ("kv_k_l1", "kv_v_l0"):
            m = re.compile(rf"\n *llm_cache_flush\({cache},[^\n]*").search(src, start)
            bad = src[:m.start()] + src[m.end():]
            errs = CoherencyChecker(mg.cgs["prefill_16"], bad,
                                    run_name="inference_run_prefill_16").check()
            self.assertTrue(any(f"reads '{cache}' with CPU-dirty lines" in e for e in errs), errs)

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
                for m in re.finditer(r"\b(inference_buf_sync_\w+|run_\w+|host_in|host_out)\(\s*(\w+)",
                                     src):
                    self.assertNotIn(m.group(2), host, m.group(0))


if __name__ == "__main__":
    unittest.main()
