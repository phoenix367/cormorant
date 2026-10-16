"""The row-ops unit in the scheduler (doc/plans/HOSTOPS_PLAN.md §6,
src/rowops_nodes.py): the fifth kernel lane, the table load at init, the
LayerNorm node — its result is rowops_spec.layer_norm, its C call carries the
registers kernel_calls() lists, and the generated project runs on the host
against the kernel's software model (test/host_emu.py) exactly as the
simulation predicts.
"""
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import onnx
import onnx.helper as oh
from onnx import TensorProto

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_REPO = os.path.dirname(_ROOT)
sys.path.insert(0, _HERE)
import host_emu                                                       # noqa: E402
from src import _rowops_hw_config, rowops_nodes, rowops_spec as rs    # noqa: E402
from src.codegen import CodeGenerator                                 # noqa: E402
from src.graph import OnnxGraph                                       # noqa: E402
from src.host_nodes import LayerNormNode                              # noqa: E402
from src.kernels import KERNEL_REGISTRY                               # noqa: E402
from src.perf_calls import FIELDS, KernelCall, merge                  # noqa: E402
from src.rowops_nodes import LayerNormRoNode, ResAddRoNode, RowOpsNode      # noqa: E402

_FIXTURES = os.path.join(_REPO, "hw", "test_data", "rowops_test_data")


def _unit(on: bool):
    """Patch whether the platform has the row-ops unit."""
    return mock.patch.object(_rowops_hw_config, "ROWOPS_UNIT", on)


def _ln_model(path, rows, n, seed=1, beta=True, eps=1e-5, batch=1):
    rng = np.random.default_rng(seed)
    inits = [oh.make_tensor("gamma", TensorProto.FLOAT, [n],
                            rng.uniform(0.5, 1.5, n).astype(np.float32).tolist())]
    ins = ["x", "gamma"]
    if beta:
        inits.append(oh.make_tensor("beta", TensorProto.FLOAT, [n],
                                    rng.uniform(-0.5, 0.5, n).astype(np.float32).tolist()))
        ins.append("beta")
    g = oh.make_graph([oh.make_node("LayerNormalization", ins, ["y"], axis=-1, epsilon=eps)], "m",
                      [oh.make_tensor_value_info("x", TensorProto.FLOAT, [batch, rows, n])],
                      [oh.make_tensor_value_info("y", TensorProto.FLOAT, [batch, rows, n])], inits)
    m = oh.make_model(g, opset_imports=[oh.make_opsetid("", 17)])
    m.ir_version = 9
    onnx.save(m, path)
    return path


class TestSpecHelpers(unittest.TestCase):
    def test_table_image_is_the_fixtures(self):
        """The image the generated project loads is the RTL testbench's tables.hex."""
        img = rs.table_image()
        self.assertEqual(len(img), rs.TABLE_BYTES)
        path = os.path.join(_FIXTURES, "tables.hex")
        if os.path.exists(path):
            ref = bytes.fromhex("".join(line.strip() for line in open(path)))
            self.assertEqual(img, ref)

    def test_param_block(self):
        w = rs.param_block([7, -3], [5, -(1 << 21)], 1, 2, [24, 63])
        self.assertEqual(w.size, 16)
        e = np.asarray(w, np.int64).astype(np.int16).view(np.uint8).reshape(2, 16)
        self.assertEqual(int(np.frombuffer(e[0, 0:4].tobytes(), "<i4")[0]), 7)
        self.assertEqual(int(np.frombuffer(e[1, 0:4].tobytes(), "<i4")[0]), -3)
        self.assertEqual(int(np.frombuffer(e[0, 4:8].tobytes(), "<i4")[0]), 5)
        self.assertEqual(int(np.frombuffer(e[1, 4:8].tobytes(), "<i4")[0]), -(1 << 21))
        self.assertEqual(list(e[0, 8:11]), [1, 2, 24])
        self.assertEqual(list(e[1, 8:11]), [1, 2, 63])
        self.assertEqual(int(e[:, 11:].sum()), 0)
        with self.assertRaises(ValueError):
            rs.param_block([1], [1 << 21], 0, 0, 0)

    def test_opword(self):
        self.assertEqual(rs.opword(rs.OP_NORM, ln=True), 5 | (1 << 16))
        self.assertEqual(rs.opword(rs.OP_ADDX, rs.K_I32, rs.K_I16, rs.K_I32, b_en=True, p_bcast=True),
                         2 | (1 << 4) | (1 << 8) | (1 << 12) | (1 << 13))

    def test_raw_as_data_roundtrip(self):
        from src.dtype import AP_FIXED_16_8
        raw = np.arange(-32768, 32768, 977, dtype=np.int64)
        back = AP_FIXED_16_8.float_to_storage(rowops_nodes.raw_as_data(raw)).astype(np.int64)
        self.assertTrue(np.array_equal(back.astype(np.int16).astype(np.int64), raw))


class TestRegistry(unittest.TestCase):
    def test_lane(self):
        kd = KERNEL_REGISTRY["RowOpsKernel"]
        self.assertEqual(kd.c_type, "XRowopskernel")
        self.assertEqual(kd.driver_prefix, "xrowopskernel")
        self.assertIn("xrowopskernel_hw.h", kd.driver_files)
        self.assertIn("RowOpsKernel", FIELDS)
        self.assertEqual(KernelCall.of("RowOpsKernel", op=5, n=8).fields["n"], 8)


class TestLayerNormChoice(unittest.TestCase):
    def test_host_without_the_unit(self):
        with tempfile.TemporaryDirectory() as td, _unit(False):
            g = OnnxGraph(_ln_model(os.path.join(td, "m.onnx"), 4, 32))
            self.assertIsInstance(g.nodes[0], LayerNormNode)

    def test_unit_node(self):
        with tempfile.TemporaryDirectory() as td, _unit(True):
            g = OnnxGraph(_ln_model(os.path.join(td, "m.onnx"), 4, 32))
            sn = g.nodes[0]
            self.assertIsInstance(sn, LayerNormRoNode)
            self.assertEqual(sn.kernel_name, "RowOpsKernel")
            self.assertEqual([t.onnx_name for t in sn.inputs][1:], ["y__rowops_p", rowops_nodes.TABLES_NAME])
            self.assertEqual(sn.inputs[1].numel, 8 * 32)
            self.assertEqual(sn.inputs[2].numel, rs.TABLE_BYTES // 2)
            self.assertTrue(all(t.is_weight for t in sn.inputs[1:]))
            calls = sn.kernel_calls({})
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0].fields["op"], rs.opword(rs.OP_NORM, ln=True))
            self.assertEqual(calls[0].fields["rows"], 4)
            self.assertEqual(calls[0].fields["a_s1"], 64)

    def test_ineligible_rows(self):
        with tempfile.TemporaryDirectory() as td, _unit(True):
            # rows of 12 elements are not 16-byte aligned: the host op
            g = OnnxGraph(_ln_model(os.path.join(td, "m.onnx"), 4, 12))
            self.assertIsInstance(g.nodes[0], LayerNormNode)
            # one row of 12 is a flat buffer: fine
            g = OnnxGraph(_ln_model(os.path.join(td, "m1.onnx"), 1, 12))
            self.assertIsInstance(g.nodes[0], LayerNormRoNode)

    def test_reference_is_the_spec(self):
        with tempfile.TemporaryDirectory() as td, _unit(True):
            path = _ln_model(os.path.join(td, "m.onnx"), 5, 40, seed=3)
            g = OnnxGraph(path)
            sn = g.nodes[0]
            rng = np.random.default_rng(7)
            x = np.round(rng.normal(0, 2, (1, 5, 40)) * 256) / 256
            y = sn.reference([x], None)
            raw = np.rint(x.reshape(5, 40) * 256).astype(np.int64)
            ref = rs.layer_norm(raw, 8, sn.gamma, sn.beta, sn.eps, 8) / 256.0
            self.assertTrue(np.array_equal(y.reshape(5, 40), ref))
            # within a Q8.8 LSB or two of the exact LayerNorm
            ex = (x.reshape(5, 40) - x.reshape(5, 40).mean(1, keepdims=True))
            ex = ex / np.sqrt(x.reshape(5, 40).var(1, keepdims=True) + sn.eps) * sn.gamma + sn.beta
            self.assertLess(np.abs(y.reshape(5, 40) - ex).max(), 2.5 / 256)


class TestGeneratedCode(unittest.TestCase):
    def _cg(self, td, **kw):
        path = _ln_model(os.path.join(td, "m.onnx"), **kw)
        g = OnnxGraph(path)
        return g, CodeGenerator(g, model_path=path)

    def test_source(self):
        with tempfile.TemporaryDirectory() as td, _unit(True):
            g, cg = self._cg(td, rows=4, n=32)
            src = cg.generate_source()
            hdr = cg.generate_header()
            self.assertIn("KERNEL_ROWOPS", src)
            self.assertIn("static void run_rowops(", src)
            # the table load once at init, then the job
            init = src[src.index("int inference_init("):src.index("void inference_deinit(")]
            self.assertIn("run_rowops(rowops_tables", init)
            self.assertIn("kernel_wait(KERNEL_ROWOPS)", init)
            self.assertIn("0x00158u, 4096u, 4u, 1u, 16384u, 65536u", init)
            run = src[src.index("void inference_run("):]
            self.assertIn("run_rowops(x, 0u, NULL, 0u, y, 0u, y_rowops_p", run)
            self.assertIn("INFERENCE_ROWOPSKERNEL_INSTANCE", hdr)
            self.assertIn("rowopskernel_instance", src)

    def test_emitted_call_matches_kernel_calls(self):
        from test_perf_calls import _Eval, emitted_calls
        with tempfile.TemporaryDirectory() as td, _unit(True):
            g, cg = self._cg(td, rows=3, n=24)
            ev = _Eval(cg.generate_header())
            for sn in g.nodes:
                self.assertEqual(emitted_calls(sn.emit_call(cg._layouts), ev), sn.kernel_calls(cg._layouts))

    @unittest.skipUnless(host_emu.which_cc(), "C compiler not available on host")
    def test_host_emulated_run_matches_simulation(self):
        for rows, n, beta, eps, incoherent in ((6, 48, True, 1e-5, False), (3, 8, False, 1e-12, True),
                                               (1, 20, True, 1e-3, False)):
            with tempfile.TemporaryDirectory() as td, _unit(True), self.subTest(rows=rows, n=n):
                g, cg = self._cg(td, rows=rows, n=n, beta=beta, eps=eps)
                self.assertIsInstance(g.nodes[0], LayerNormRoNode)
                rc, out = host_emu.build_and_run(cg, td, incoherent=incoherent)
                self.assertEqual(rc, 0, out[-3000:])
                self.assertEqual(host_emu.failures(out), [])


def _dequant_model(path, n, exp=10, i32=False):
    """x [1][n] (int16 at 2^-exp, or the int32 residual) -> LlmDequant -> yd (float32 DMA)
    -> LlmToHost -> y (float32 host output): RPU_PLAN A1's logits path."""
    from src import numeric
    from src.llm_nodes import LLM_DOMAIN
    meta = numeric.empty()
    meta["exp"]["x"] = exp
    if i32:
        meta["dma"]["x"] = "i32"
    meta["dma"]["yd"] = "f32"
    meta["host"]["y"] = "f32"
    g = oh.make_graph([oh.make_node("LlmDequant", ["x"], ["yd"], name="dq", domain=LLM_DOMAIN),
                       oh.make_node("LlmToHost", ["yd"], ["y"], name="out", domain=LLM_DOMAIN)], "m",
                      [oh.make_tensor_value_info("x", TensorProto.FLOAT, [1, n])],
                      [oh.make_tensor_value_info("y", TensorProto.FLOAT, [1, n])],
                      value_info=[oh.make_tensor_value_info("yd", TensorProto.FLOAT, [1, n])])
    m = oh.make_model(g, opset_imports=[oh.make_opsetid("", 17), oh.make_opsetid(LLM_DOMAIN, 1)])
    m.ir_version = 9
    md = m.metadata_props.add()
    md.key, md.value = numeric.METADATA_KEY, numeric.to_metadata(meta)
    onnx.save(m, path)
    return path


class TestDequantOnUnit(unittest.TestCase):
    """LlmDequant into a float32 DMA tensor (DequantRoNode: one row, or a row
    longer than the unit's 4 096 cut into contiguous rows at one exponent) and
    the LlmToHost copy — the generated C against the simulation."""

    def test_nodes(self):
        from src.rowops_nodes import DequantRoNode
        with tempfile.TemporaryDirectory() as td, _unit(True):
            for n, jobs, nn in ((24, 1, 24), (8192, 2, 4096), (12000, 3, 4000)):
                g = OnnxGraph(_dequant_model(os.path.join(td, "m.onnx"), n))
                dq = g.nodes[0]
                self.assertIsInstance(dq, DequantRoNode)
                self.assertEqual((dq.rows, dq.n, dq.rerow, len(dq.pbs)), (1, nn, n > 4096, jobs), n)
                self.assertEqual(sum(c.count for c in dq.kernel_calls({})), jobs)
                self.assertEqual(type(g.nodes[1]).__name__, "LlmToHostNode")
                self.assertEqual(dq.output.dma, "f32")
                self.assertEqual(dq.output.alloc_numel, 2 * n)
            g = OnnxGraph(_dequant_model(os.path.join(td, "m.onnx"), 64, exp=15, i32=True))
            self.assertTrue(g.nodes[0].a32)

    def test_host_without_the_unit(self):
        with tempfile.TemporaryDirectory() as td, _unit(False), \
                self.assertRaisesRegex(Exception, "row-ops unit"):
            OnnxGraph(_dequant_model(os.path.join(td, "m.onnx"), 24))

    @unittest.skipUnless(host_emu.which_cc(), "C compiler not available on host")
    def test_host_emulated_run_matches_simulation(self):
        # (an int32 graph input has no consistent test-harness fill; the int32 path is the
        # vision connector sum's, test_vit TestVisionRowops)
        for n, exp in ((24, 10), (8192, 12), (12000, 8)):
            with tempfile.TemporaryDirectory() as td, _unit(True), self.subTest(n=n):
                path = _dequant_model(os.path.join(td, "m.onnx"), n, exp=exp)
                g = OnnxGraph(path)
                cg = CodeGenerator(g, model_path=path)
                rc, out = host_emu.build_and_run(cg, td, incoherent=True)
                self.assertEqual(rc, 0, out[-3000:])
                self.assertEqual(host_emu.failures(out), [])


class TestVImageJobs(unittest.TestCase):
    """rowops_nodes.vimage_jobs: the TRANSPOSE jobs that write the V rows
    [pos, pos + n) of one KV group into the P.V image (HOSTOPS_PLAN §10.3)
    touch exactly the host's elements — every (row, d) of those rows once, at
    llm_nodes.v_row_base, and nothing else — for every K and many (pos, n)."""

    def _written(self, pos, n, HD, K, C):
        from src.rowops_nodes import vimage_jobs
        stride = 2 * HD                                            # one group per staging row
        hits = {}
        for a_off, c_off, rows, blocks, a_s1, a_s2, b_s1, b_s2, c_s1, c_s2, R in \
                vimage_jobs(pos, n, HD, K, stride):
            self.assertEqual((a_s1 % 16, a_s2 % 16, b_s1 % 16), (0, 0, 0))
            self.assertTrue(1 <= R <= 8)
            for i2 in range(blocks):
                for i1 in range(rows):
                    for r in range(R):
                        src = a_off + i2 * a_s2 + i1 * a_s1 + r * b_s1
                        self.assertEqual(src % stride, 0, "a tile row starts a staging row")
                        t = src // stride
                        for d in range(HD):
                            dst = c_off + i2 * c_s2 + i1 * c_s1 + d * b_s2 + 2 * r
                            self.assertNotIn(dst, hits, "a byte written twice")
                            hits[dst] = (t, d)
        return hits

    def test_matches_the_host_layout(self):
        rng = np.random.default_rng(7)
        from src.llm_nodes import v_row_base
        for K in (1, 2, 4):
            if K == 1:
                continue                                             # a plain COPY, not these jobs
            HD, C = 16, 256
            for pos, n in [(0, 16), (0, 64), (0, 256), (3, 29), (61, 7), (17, 100), (130, 126),
                           (255, 1), (64, 64), (0, 1)] + [tuple(map(int, (rng.integers(0, C),
                                                                        rng.integers(1, 64))))
                                                          for _ in range(30)]:
                n = min(n, C - pos)
                with self.subTest(K=K, pos=pos, n=n):
                    hits = self._written(pos, n, HD, K, C)
                    want = {2 * (v_row_base(C, HD, K, 0, pos + t) + d * K): (t, d)
                            for t in range(n) for d in range(HD)}
                    self.assertEqual(hits, want)

    def test_whole_blocks_use_contiguous_granules(self):
        from src.rowops_nodes import vimage_jobs
        jobs = vimage_jobs(64, 256, 64, 4, 384)
        whole = [j for j in jobs if j[10] == 4]
        self.assertEqual(len(whole), 1)
        self.assertEqual((whole[0][2], whole[0][3], whole[0][7]), (16, 4, 8))   # 16 tiles x 4 blocks, 2R
        self.assertEqual([j[10] for j in jobs if j[10] == 1], [])            # aligned: no edges
        edges = [j for j in vimage_jobs(3, 100, 64, 4, 384) if j[10] == 1]
        self.assertEqual([(j[2], j[3]) for j in edges], [(13, 1), (16, 3), (16, 2), (7, 1)])


class TestLlamaOnUnit(unittest.TestCase):
    """The tiny Llama (test/gen_llama_models.py) with rowops=True: the residual
    stream as the int32 DMA tensor, the residual adds / RMSNorms / SiLU * up on
    the unit, the attention as before."""

    @classmethod
    def setUpClass(cls):
        import gen_llama_models as G
        cls.G = G
        W = G.tiny_weights(G.TINY, 0)
        cls.formats = G.calibrate(G.TINY, W)
        cls.fe = G.frontend(G.TINY, W, cls.formats, decode_attn="fpga", rowops=True)

    def _graph(self, td, kind, T=1, with_head=False):
        import onnx
        path = os.path.join(td, f"{kind}.onnx")
        onnx.save(self.fe.entry(kind, T, with_head=with_head), path)
        g = OnnxGraph(path)
        return g, CodeGenerator(g, model_path=path)

    def test_nodes_and_tensors(self):
        with tempfile.TemporaryDirectory() as td, _unit(True):
            g, cg = self._graph(td, "prefill", 16)
            kinds = {type(sn).__name__ for sn in g.nodes}
            self.assertTrue({"RMSNormRoNode", "ResAddRoNode", "SiluMulRoNode"} <= kinds)
            self.assertNotIn("LlmRMSNormNode", kinds)
            h = next(sn for sn in g.nodes if isinstance(sn, ResAddRoNode)).output
            self.assertEqual(h.dma, "i32")
            self.assertEqual(h.alloc_numel, 2 * h.numel)
            self.assertEqual(int(h.exp_channels(8)[0]), 15)
            hl = next(t for t in g.state_tensors if t.onnx_name == "h_last")
            self.assertEqual((hl.dma, hl.is_host), ("i32", False))
            src = cg.generate_source()
            self.assertIn("llm_embed32(", src)
            self.assertIn("llm_select_row32(", src)
            self.assertIn("(int32_t *)inference_buf_ptr(", src)

    def test_needs_the_unit(self):
        from src.nodes import SchedulerError
        with tempfile.TemporaryDirectory() as td, _unit(False):
            with self.assertRaises(SchedulerError):
                self._graph(td, "decode")

    def test_emitted_calls_match(self):
        from test_perf_calls import _Eval, kernel_nodes, node_emitted_calls
        with tempfile.TemporaryDirectory() as td, _unit(True):
            for kind, T in (("decode", 1), ("prefill", 16)):
                g, cg = self._graph(td, kind, T)
                ev = _Eval(cg.generate_header())
                for sn in kernel_nodes(g):
                    if isinstance(sn, RowOpsNode):
                        self.assertEqual(node_emitted_calls(sn, cg, ev), merge(sn.kernel_calls(cg._layouts)),
                                         f"{kind}: {type(sn).__name__}")

    def test_coherency_audit(self):
        from test_cache_coherency import CoherencyChecker
        with tempfile.TemporaryDirectory() as td, _unit(True):
            for kind, T in (("decode", 1), ("prefill", 16)):
                g, cg = self._graph(td, kind, T)
                self.assertEqual(CoherencyChecker(cg).check(), [], kind)

    @unittest.skipUnless(host_emu.which_cc(), "C compiler not available on host")
    def test_host_emulated_run_matches_simulation(self):
        # the prefill with the head: its logits are the output compared (a prefill entry alone
        # writes only states)
        for kind, T, incoherent in (("decode", 1, False), ("prefill", 16, True), ("head", 1, True)):
            with tempfile.TemporaryDirectory() as td, _unit(True), self.subTest(kind=kind):
                g, cg = self._graph(td, kind, T, with_head=kind == "prefill")
                self.assertTrue(g.output_tensors, kind)
                rc, out = host_emu.build_and_run(cg, td, incoherent=incoherent)
                self.assertEqual(rc, 0, out[-3000:])
                self.assertEqual(host_emu.failures(out), [])


class TestDecodePrepOnUnit(unittest.TestCase):
    """RPU_PLAN §13.4 (A2.4): the frontend's drope — the decode step's
    attention prep on the unit too (AttnPrepRoNode with one row: ROPE of q / k,
    the q image as one TRANSPOSE with R = G, the cache rows at the runtime pos)
    — bit-exact against the study's ...+rowops+drope and different from the
    host prep's double RoPE; the generated C equals the simulation, with and
    without descriptor chains."""

    @classmethod
    def setUpClass(cls):
        import gen_llama_models as G
        cls.G = G
        cls.W = G.tiny_weights(G.TINY, 0)
        cls.formats = G.calibrate(G.TINY, cls.W)

    def _fe(self, drope):
        return self.G.frontend(self.G.TINY, self.W, self.formats, decode_attn="fpga", rowops=True,
                               drope=drope)

    def test_simulation_matches_study(self):
        import test_llama as TL
        from src.codegen import CodeGenerator as CG
        G = self.G
        runs = {}
        for drope in (False, True):
            with _unit(True):
                gs = TL.graphs(self._fe(drope), {"decode": ("decode", 1, False), "prefill_8": ("prefill", 8, False),
                                                 "prefill_16": ("prefill", 16, False), "head": ("head", 1, False)})
                cgs = {n: CG(g, model_path=n) for n, g in gs.items()}
            kinds = [type(sn).__name__ for sn in gs["decode"].nodes]
            self.assertEqual("AttnPrepRoNode" in kinds, drope)
            self.assertEqual("LlmAttnPrepNode" in kinds, not drope)
            ses = TL.Session(cgs, [8, 16])
            prompt = np.random.default_rng(11).integers(2, 256, 13)
            a = [ses.prefill(list(prompt))]
            for _ in range(6):
                a.append(ses.decode(int(np.argmax(a[-1]))))
            runs[drope] = a
            sm, sc = G.study_model(G.TINY, self.W, self.formats,
                                   policy="pow2+sink+p12+rowops" + ("+drope" if drope else ""))
            st = TL.study.Seq(sc, 64)
            b = [sm.forward([st], [np.concatenate([[1], prompt])])[0][-1]]
            for _ in range(6):
                b.append(sm.forward([st], [np.array([int(np.argmax(b[-1]))])], phase="decode")[0][-1])
            for i, (x, y) in enumerate(zip(a, b, strict=True)):
                np.testing.assert_array_equal(x, y, err_msg=f"drope={drope} step {i}")
        # the prefill is the same (on this model the decode steps' integer RoPE rounds as the
        # host's double one did; the real models' differ — RPU_PLAN §13.8, the drope study)
        np.testing.assert_array_equal(runs[False][0], runs[True][0])

    def test_graph_and_source(self):
        import onnx
        from src.rowops_nodes import AttnPrepRoNode
        with tempfile.TemporaryDirectory() as td, _unit(True):
            path = os.path.join(td, "decode.onnx")
            onnx.save(self._fe(True).entry("decode", 1), path)
            g = OnnxGraph(path)
            cg = CodeGenerator(g, model_path=path)
            preps = [sn for sn in g.nodes if isinstance(sn, AttnPrepRoNode)]
            self.assertEqual(len(preps), 2)
            G_ = preps[0].H // preps[0].KV
            for sn in preps:
                self.assertTrue(sn.T == 1 and not sn.has_n and not sn.first)
                xp = [c for c in sn.kernel_calls(cg._layouts) if c.fields["op"] == rs.opword(rs.OP_XPOSE)
                      and c.fields["aux"] == G_]
                self.assertEqual(len(xp), 1)                  # the q image: one job, R = G
            src = cg.generate_source()
            self.assertEqual(src.count("llm_cache_flush_rows("),       # no CPU write into the caches:
                             src.count("static void llm_cache_flush_rows("))   # the helper, never called
            self.assertEqual(src.count("llm_attn_prep("), src.count("static void llm_attn_prep("))

    @unittest.skipUnless(host_emu.which_cc(), "C compiler not available on host")
    def test_host_emulated_run_matches_simulation(self):
        import onnx
        for chains in (False, True):
            with tempfile.TemporaryDirectory() as td, _unit(True), self.subTest(chains=chains), \
                    mock.patch.object(_rowops_hw_config, "ROWOPS_CHAINS", chains):
                path = os.path.join(td, "decode.onnx")
                onnx.save(self._fe(True).entry("decode", 1), path)
                g = OnnxGraph(path)
                cg = CodeGenerator(g, model_path=path)
                self.assertEqual(bool(cg._ro_chains), chains)
                self.assertTrue(g.output_tensors)
                rc, out = host_emu.build_and_run(cg, td, incoherent=True)
                self.assertEqual(rc, 0, out[-3000:])
                self.assertEqual(host_emu.failures(out), [])


class TestRowopsHwConfig(unittest.TestCase):
    """kernels.rowops: the five flags, required bools; fold, chains, gate and scale
    need the unit; AXI_ROWOPS_CHAINS / _GATE / _SCALE override them."""

    def test_chains_flag(self):
        import json
        plats = _rowops_hw_config._PLATFORMS_DIR
        p = plats / "_test_rowops_chains.json"
        try:
            for unit, chains, env, want in ((True, False, None, False), (True, True, None, True),
                                            (False, True, None, False), (True, False, "1", True),
                                            (True, True, "0", False)):
                p.write_text(json.dumps({"kernels": {"rowops": {"unit": unit, "fold": False,
                                                                "chains": chains, "gate": False,
                                                                "scale": False}}}))
                with mock.patch.dict(os.environ, {"AXI_ROWOPS_CHAINS": env} if env else {}):
                    if not env:
                        os.environ.pop("AXI_ROWOPS_CHAINS", None)
                    os.environ.pop("AXI_ROWOPS", None)
                    self.assertEqual(_rowops_hw_config.resolve("_test_rowops_chains")["ROWOPS_CHAINS"], want)
            p.write_text(json.dumps({"kernels": {"rowops": {"unit": True, "fold": False}}}))
            with self.assertRaises(_rowops_hw_config.RowopsHwConfigError) as cm:
                _rowops_hw_config.resolve("_test_rowops_chains")
            self.assertIn("chains", str(cm.exception))
        finally:
            p.unlink(missing_ok=True)

    def test_gate_scale_flags(self):
        """GATE and SCALE (RPU_PLAN §14.3): each its own flag, off without the unit,
        its environment override; a platform file without them is refused."""
        import json
        p = _rowops_hw_config._PLATFORMS_DIR / "_test_rowops_gate.json"
        envs = ("AXI_ROWOPS", "AXI_ROWOPS_GATE", "AXI_ROWOPS_SCALE")
        try:
            for unit, gate, scale, env, want in ((True, True, False, {}, (True, False)),
                                                 (True, False, True, {}, (False, True)),
                                                 (False, True, True, {}, (False, False)),
                                                 (True, False, False, {"AXI_ROWOPS_GATE": "1"}, (True, False)),
                                                 (True, True, True, {"AXI_ROWOPS_SCALE": "0"}, (True, False))):
                p.write_text(json.dumps({"kernels": {"rowops": {"unit": unit, "fold": True, "chains": True,
                                                                "gate": gate, "scale": scale}}}))
                with mock.patch.dict(os.environ, env):
                    for e in envs:
                        if e not in env:
                            os.environ.pop(e, None)
                    r = _rowops_hw_config.resolve("_test_rowops_gate")
                    self.assertEqual((r["ROWOPS_GATE"], r["ROWOPS_SCALE"]), want, (unit, gate, scale, env))
            p.write_text(json.dumps({"kernels": {"rowops": {"unit": True, "fold": True, "chains": True}}}))
            with self.assertRaises(_rowops_hw_config.RowopsHwConfigError) as cm:
                _rowops_hw_config.resolve("_test_rowops_gate")
            self.assertIn("gate", str(cm.exception))
        finally:
            p.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
