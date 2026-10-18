# Copyright 2026 Ivan Gubochkin
# SPDX-License-Identifier: Apache-2.0

"""RowOpsKernel descriptor chains in the generated code (doc/plans/RPU_PLAN.md
§13.3, A2.4): with ``kernels.rowops.chains`` a row-ops node of two or more jobs
collects them as 128-byte descriptors in its region at the end of the pool and
starts them as chains of at most 16 — the generated C on the software kernels
(test/host_emu.py runs a chain's descriptors from emulated DDR) against the
simulation, with separate CPU / DDR copies so a descriptor not cleaned before
its Start reads stale."""
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import onnx

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import host_emu                                                  # noqa: E402
from src import _rowops_hw_config, _vectorop_hw_config           # noqa: E402
from src.codegen import CodeGenerator                            # noqa: E402
from src.codegen.multi import MultiEntryGenerator                # noqa: E402
from src.graph import OnnxGraph                                  # noqa: E402
from src.rowops_nodes import RowOpsNode                          # noqa: E402


def _flags(chains: bool):
    """The unit, and whether it runs chains."""
    return mock.patch.multiple(_rowops_hw_config, ROWOPS_UNIT=True, ROWOPS_CHAINS=chains)


_LLAMA = None


def llama_fe():
    """The tiny Llama with rowops=True (test_rowops_unit.TestLlamaOnUnit's)."""
    global _LLAMA
    if _LLAMA is None:
        import gen_llama_models as G
        W = G.tiny_weights(G.TINY, 0)
        _LLAMA = G.frontend(G.TINY, W, G.calibrate(G.TINY, W), decode_attn="fpga", rowops=True)
    return _LLAMA


def chained(g, cg):
    """{type name: [descriptors]} of the chained nodes."""
    out = {}
    for sn in g.nodes:
        if sn.index in cg._ro_chains:
            out.setdefault(type(sn).__name__, []).append(cg._ro_chains[sn.index])
    return out


@unittest.skipUnless(host_emu.which_cc(), "C compiler not available on host")
class TestChains(unittest.TestCase):

    def _run(self, cg, td, **kw):
        rc, out = host_emu.build_and_run(cg, td, **kw)
        self.assertEqual(rc, 0, out[-3000:])
        self.assertEqual(host_emu.failures(out), [])
        self.assertIn("PASSED", out)

    def test_llama_prefill(self):
        """The prefill attention prep (AttnPrepRoNode: ROPE, the q image, the
        caches and the V image's runtime jobs — 20 / 21 descriptors, a chain of
        16 and one of the rest); with the head, so the logits are compared.
        Without the flag nothing is chained and the source has no chain code."""
        fe = llama_fe()
        for chains in (True, False):
            with tempfile.TemporaryDirectory() as td, _flags(chains), self.subTest(chains=chains):
                path = os.path.join(td, "p.onnx")
                onnx.save(fe.entry("prefill", 16, with_head=True), path)
                g = OnnxGraph(path)
                cg = CodeGenerator(g, model_path=path)
                src = cg.generate_source()
                if not chains:
                    self.assertEqual(cg._ro_chains, {})
                    self.assertNotIn("ro_chain", src)
                    continue
                ch = chained(g, cg)
                self.assertEqual(list(ch), ["AttnPrepRoNode"])
                self.assertTrue(all(n > 16 for n in ch["AttnPrepRoNode"]))
                for sn in g.nodes:
                    if isinstance(sn, RowOpsNode) and sn.index not in cg._ro_chains:
                        self.assertEqual(sn.chain_jobs(cg._layouts), 1, type(sn).__name__)
                self.assertEqual(src.count("    ro_chain_begin(&s_ro_ch_"), len(cg._ro_chains))
                self.assertIn("#ifndef XROWOPSKERNEL_CTRL_ADDR_CHAIN_N_DATA", src)
                self.assertIn("if (k == KERNEL_ROWOPS && s_ro_chain) return;", src)
                # the descriptor area at the end of the pool, in the advertised size
                self.assertGreaterEqual(cg._compute_pool_bytes() - cg._compute_pool_layout()[1] * 2,
                                        128 * sum(cg._ro_chains.values()))
                # the early wake: a chained node's prediction covers its shortest chain, not one job
                ex = cg._kernel_expect_us()
                for sn in g.nodes:
                    if sn.index in cg._ro_chains:
                        from src.rpu_policy import estimate_us
                        cs = sn.kernel_calls(cg._layouts)
                        self.assertGreater(ex[sn.index], int(min(estimate_us(c) for c in cs)))
                self._run(cg, td, incoherent=True)
                self._run(cg, td, cached=False)

    def test_dequant_rows(self):
        """A row longer than the unit's 4 096 as three DEQUANT jobs: one chain."""
        from test_rowops_unit import _dequant_model
        with tempfile.TemporaryDirectory() as td, _flags(True):
            path = _dequant_model(os.path.join(td, "m.onnx"), 12000, exp=8)
            g = OnnxGraph(path)
            cg = CodeGenerator(g, model_path=path)
            self.assertEqual(chained(g, cg), {"DequantRoNode": [3]})
            self._run(cg, td, incoherent=True)

    def test_vision_encoder(self):
        """The vision encoder on the unit (VitAttnPrepRoNode: the bias adds, the
        q image, the caches and the V image per head)."""
        from test_vit import CONN_K, Tiny
        from src.vit import VitFrontend
        cfg, W, _vm, fe = Tiny.get()
        with mock.patch.object(_vectorop_hw_config, "VECTOROP_SOFTMAX", True), _flags(True):
            fe2 = VitFrontend(cfg, W, fe.fmt, name="vit_tiny", conn_k=CONN_K, vsmx=True, rowops=True)
            g = OnnxGraph(fe2.entry(output=True), fuse_act=True, s2d_stem=True)
            cg = CodeGenerator(g, model_path="vit_tiny_chains.onnx")
        ch = chained(g, cg)
        self.assertIn("VitAttnPrepRoNode", ch)
        with tempfile.TemporaryDirectory() as td:
            self._run(cg, td, cached=True, threads=3, min_elems=1, incoherent=True)

    @unittest.skipUnless(shutil.which("cc"), "needs cc")
    def test_piper_encoder(self):
        """Piper's text encoder on the unit: the attention prep, the row preps
        (a COPY per tap) and the relative-key band (9 diagonal TRANSPOSEs + the
        ADDX) as chains."""
        from test_piper import _unit, encoder_setup
        from src.piper import PiperEncoderFrontend
        W, _, E = encoder_setup()
        with _unit(), _flags(True):
            g = OnnxGraph(PiperEncoderFrontend(W, E, 32, vsmx=True, rowops=True).entry())
            cg = CodeGenerator(g, model_path="enc_chains.onnx")
        ch = chained(g, cg)
        self.assertTrue({"VitAttnPrepRoNode", "TtsRowPrepRoNode", "TtsRelBandRoNode"} <= set(ch), ch)
        self.assertEqual(ch["TtsRelBandRoNode"], [10] * len(ch["TtsRelBandRoNode"]))
        with tempfile.TemporaryDirectory() as td:
            self._run(cg, os.path.join(td, "p"), timeout=1800)

    def test_multi_entry_project(self):
        """Prefill buckets, decode and the head in one library: every chained
        node of every entry has its own region (the nodes are numbered across
        the entries)."""
        from src.llm_entries import entry_graphs
        fe = llama_fe()
        kinds = {"decode": ("decode", 1), "prefill_16": ("prefill", 16), "head": ("head", 1)}
        with _flags(True):
            models = {n: fe.entry(k, T) for n, (k, T) in kinds.items()}
            mg = MultiEntryGenerator(entry_graphs(models), "llama_tiny_chains")
        caps = mg.cg._ro_chains
        self.assertTrue(caps)
        for name, cg in mg.cgs.items():
            self.assertTrue(set(cg._ro_chains) <= set(caps), name)
        src = mg.generate_source()
        self.assertEqual(src.count("static ro_chain_t s_ro_ch_"), len(caps))
        with tempfile.TemporaryDirectory() as td:
            rc, out = host_emu.build_and_run(mg, td, cached=True, threads=4, min_elems=1, incoherent=True)
            self.assertEqual(rc, 0, out[-3000:])
            self.assertIn("test_inference PASSED", out)


class TestChainPricing(unittest.TestCase):
    """The chain term of the performance model (perf_fit.fit_chain,
    PerfModel.chain_adjust_us) and its uses: chained cases kept out of the
    exact entries, a chained node priced as its jobs plus the term, one R5
    call per chain."""

    @staticmethod
    def _rowops_key(n):
        from src.perf_calls import KernelCall
        return KernelCall.of("RowOpsKernel", op=1, n=n, rows=1, blocks=1, win_hi=0xFFFFFFFF).key()

    def test_fit_recovers_the_term(self):
        import perf_calibrate as pc
        from src.perf_fit import fit
        base = {self._rowops_key(n): 3.0 + n / 1000 for n in (64, 128, 256, 512, 1024, 2048, 4096)}
        cases = {k: {"key": k, "kernel": "RowOpsKernel", "set": "shipped", "est_us": us, "sources": []}
                 for k, us in base.items()}
        chains = pc.chain_cases(cases)
        self.assertEqual(len(chains), pc.CHAIN_CALLS * len(pc.CHAIN_KS))
        p1 = {k: {"ok": 1, "mean_us": us} for k, us in base.items()}
        for e in chains:
            p1[e["key"]] = {"ok": 1, "mean_us": e["chain"] * (base[e["base"]] - 0.8) + 2.5}
        model = fit({"cases": list(cases.values()) + chains},
                    {"platform": "kv260", "bitstream": "x", "passes": {"1": p1}})
        self.assertEqual(sorted(model["exact"]), sorted(base))            # no chain key priced alone
        ch = model["rowops_chain"]
        self.assertAlmostEqual(ch["per_job_us"], 0.8, places=3)
        self.assertAlmostEqual(ch["per_chain_us"], 2.5, places=3)
        self.assertEqual((ch["points"], ch["calls"]), (len(chains), pc.CHAIN_CALLS))
        # the runner's line: the R call's buffers + the descriptors, its registers + k
        e = next(c for c in chains if c["chain"] == 4)
        line = pc.entry_line("c0", e).split()
        self.assertEqual(line[1], "Q")
        self.assertEqual(line[-1], "4")
        self.assertEqual(int(line[4 + int(line[4])]), 64 * 4)

    def test_adjust_and_duration(self):
        from src.codegen.timing import kernel_duration_fn
        from src.perf_calls import KernelCall
        from src.perf_model import PerfModel
        k = self._rowops_key(64)
        d = {"platform": "kv260", "bitstream": "x", "exact": {k: {"us": 5.0}}, "families": {}}
        pm0 = PerfModel(dict(d))
        self.assertEqual(pm0.chain_adjust_us(20), 0.0)                    # not measured: unchanged
        pm = PerfModel(dict(d, rowops_chain={"per_job_us": 1.0, "per_chain_us": 3.0}))
        self.assertAlmostEqual(pm.chain_adjust_us(20), 2 * 3.0 - 20 * 1.0)

        class Node:
            index = 7

            def kernel_calls(self, layouts):
                return [KernelCall.from_key(k, 20)]
        self.assertAlmostEqual(kernel_duration_fn(pm, {})(Node()), 100.0)
        self.assertAlmostEqual(kernel_duration_fn(pm, {}, chains={7: 20})(Node()), 100.0 - 14.0)

    def test_rpu_calls_per_chain(self):
        from src.rpu_policy import rpu_estimate
        fe = llama_fe()
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "p.onnx")
            onnx.save(fe.entry("prefill", 16), path)
            est = {}
            for chains in (False, True):
                with _flags(chains):
                    g = OnnxGraph(path)
                    cg = CodeGenerator(g, model_path=path)
                    est[chains] = rpu_estimate(cg)["calls"]
                    if chains:
                        merged = sum(len(sn.kernel_calls(cg._layouts)) for sn in g.nodes
                                     if sn.index in cg._ro_chains)
                        n_chains = sum(-(-sum(c.count for c in sn.kernel_calls(cg._layouts)) // 16)
                                       for sn in g.nodes if sn.index in cg._ro_chains)
            self.assertEqual(est[True], est[False] - merged + n_chains)


if __name__ == "__main__":
    unittest.main()
