"""The axi.llm host ops (src/llm_nodes.py) one at a time.

Each op is a single-node graph (host-memory and DMA inputs / outputs with
per-channel power-of-two exponents) scheduled, compiled with -Werror and run
against the software kernels (test/host_emu.py): the C helper must equal
``reference()`` — the simulator's — bit for bit, for cacheable and staged
buffers and 1 / 4 host threads.  Plus: the SiLU table of every relevant gate
exponent exhaustively (C vs Python, all 65 536 inputs); the references
against independent formulas; the attention op's causality, padded rows
and context clamp.  The FPGA prefill attention (LlmAttnPrep -> q.K^T /
P.V ConvKernel calls with a runtime key count -> the p12 softmax ->
LlmAttnMerge) as one chain over DMA-state caches, for several pos / n /
kernel widths, on the host emulation (also with separate CPU / DDR copies,
so every sync matters), its softmax exp tables exhaustively, and keys16.
"""

import json
import math
import os
import subprocess
import tempfile
import unittest

import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as nph
from onnx import TensorProto

import host_emu
from src import numeric
from src.codegen import CodeGenerator
from src.dtype import AP_FIXED_16_8 as Q
from src.graph import OnnxGraph
from src.host_nodes import HOST_C_POOL
from src.llm_nodes import (KEY_QUANTUM, LLM_DOMAIN, LlmAttnConvNode, dot8, keys16,
                           llm_c_helpers, rope, sexp_table, silu_table)
from src.nodes import SchedulerError

vi = oh.make_tensor_value_info
F32, I32 = TensorProto.FLOAT, TensorProto.INT32
RNG = np.random.default_rng(2026)


def model(nodes, inputs, outputs, inits=(), meta=None, value_info=()):
    g = oh.make_graph(nodes, "llm_op", inputs, outputs, initializer=list(inits),
                      value_info=list(value_info))
    m = oh.make_model(g, opset_imports=[oh.make_opsetid("", 17), oh.make_opsetid(LLM_DOMAIN, 1)])
    m.ir_version = 8
    p = m.metadata_props.add()
    p.key = numeric.METADATA_KEY
    p.value = json.dumps(meta or {})
    return m


def node(op, ins, outs, **attrs):
    return oh.make_node(op, ins, outs, name=op.lower(), domain=LLM_DOMAIN, **attrs)


def emulate(testcase, m, name, incoherent=False):
    cg = CodeGenerator(OnnxGraph(m, fuse_act=True), model_path=name + ".onnx")
    runs = ((True, 4, False), (False, 1, False)) + (((True, 3, True),) if incoherent else ())
    for cached, threads, inc in runs:
        with testcase.subTest(cached=cached, threads=threads, incoherent=inc), \
                tempfile.TemporaryDirectory() as td:
            rc, out = host_emu.build_and_run(cg, td, cached=cached, threads=threads, min_elems=1,
                                             incoherent=inc)
            testcase.assertEqual(rc, 0, out[-3000:])
            testcase.assertIn("test_inference PASSED", out)
    return cg


def exps(n, lo, hi):
    return RNG.integers(lo, hi + 1, n).tolist()


class TestOpsOnHost(unittest.TestCase):

    def test_rmsnorm(self):
        T, D = 3, 40
        g = (1 + RNG.normal(0, 0.3, D)).astype(np.float32)
        m = model([node("LlmRMSNorm", ["h", "g"], ["y"], axi_eps="1e-05")],
                  [vi("h", F32, [T, D])], [vi("y", F32, [T, D])], [nph.from_array(g, "g")],
                  {"host": {"h": "f32"}, "exp": {"y": exps(D, 6, 12)}})
        cg = emulate(self, m, "rmsnorm")
        sn = cg._graph.nodes[0]
        h = RNG.normal(0, 3, (T, D)).astype(np.float32).astype(np.float64)
        y = sn.reference([h], Q)
        ss = np.array([sum(float(v) * float(v) for v in row) for row in h])   # left to right
        ref = (h / np.sqrt(ss / D + 1e-5)[:, None]) * g
        f = np.asarray(json.loads(m.metadata_props[0].value)["exp"]["y"])
        self.assertLessEqual(np.abs(y - ref).max(), float(np.max(2.0 ** -f)))

    def test_resadd(self):
        T, D = 2, 24
        m = model([node("LlmResAdd", ["h", "d"], ["y"])],
                  [vi("h", F32, [T, D]), vi("d", F32, [T, D])], [vi("y", F32, [T, D])], (),
                  {"host": {"h": "f32", "y": "f32"}, "exp": {"d": exps(D, 3, 13)}})
        cg = emulate(self, m, "resadd")
        y = cg._graph.nodes[0].reference([np.full((T, D), 0.1), np.full((T, D), 2.0 ** -13)], Q)
        self.assertTrue(np.all(y == np.float32(0.1 + 2.0 ** -13)))

    def test_silu_mul(self):
        T, N = 2, 48
        m = model([node("LlmSiluMul", ["g", "u"], ["a"])],
                  [vi("g", F32, [T, N]), vi("u", F32, [T, N])], [vi("a", F32, [T, N])], (),
                  {"exp": {"g": exps(N, 9, 13), "u": exps(N, 9, 13), "a": exps(N, 6, 9)}})
        emulate(self, m, "silu_mul")

    def test_embed_bf16_and_f32(self):
        for kind, tab in (("bf16", (RNG.normal(0, 1, (50, 16)).astype(np.float32)
                                    .view(np.uint32) & np.uint32(0xFFFF0000)).view(np.float32)),
                          ("f32", RNG.normal(0, 1, (50, 16)).astype(np.float32))):
            m = model([node("LlmEmbed", ["ids", "E"], ["h"])], [vi("ids", I32, [5])],
                      [vi("h", F32, [5, 16])], [nph.from_array(tab, "E")],
                      {"host": {"ids": "i32", "h": "f32"}})
            with self.subTest(kind=kind):
                cg = emulate(self, m, "embed_" + kind)
                self.assertEqual(cg._graph.nodes[0].table.kind, kind)
                y = cg._graph.nodes[0].reference([np.array([0, 49, 60, -3, 7])], Q)
                np.testing.assert_array_equal(y, tab[[0, 49, 49, 0, 7]])        # clamped

    def test_dequant_and_select_row(self):
        m = model([node("LlmDequant", ["x"], ["y"])], [vi("x", F32, [1, 40])],
                  [vi("y", F32, [1, 40])], (), {"host": {"y": "f32"}, "exp": {"x": exps(40, 7, 11)}})
        emulate(self, m, "dequant")
        m = model([node("LlmSelectRow", ["h", "n"], ["y"])],
                  [vi("h", F32, [6, 8]), vi("n", I32, [1])], [vi("y", F32, [1, 8])], (),
                  {"host": {"h": "f32", "n": "i32", "y": "f32"}, "test_fill": {"n": 4}})
        cg = emulate(self, m, "select_row")
        h = np.arange(48.0).reshape(6, 8)
        sn = cg._graph.nodes[0]
        np.testing.assert_array_equal(sn.reference([h, [4]], Q), h[3:4])
        np.testing.assert_array_equal(sn.reference([h, [0]], Q), h[0:1])
        np.testing.assert_array_equal(sn.reference([h, [99]], Q), h[5:6])

    def _attention(self, T=3, H=4, KV=2, HD=16, C=12, pos=4, n=None, seed=0, dma=False):
        rng = np.random.default_rng(seed)
        half = HD // 2
        cos = rng.uniform(-1, 1, (C, half)).astype(np.float32)
        sin = rng.uniform(-1, 1, (C, half)).astype(np.float32)
        fk = np.repeat(rng.integers(7, 10, KV), HD)
        fv = rng.integers(7, 10, KV * HD)
        ck0 = np.round(rng.normal(0, 800, (C, KV * HD))) / 2.0 ** fk
        cv0 = np.round(rng.normal(0, 800, (C, KV * HD))) / 2.0 ** fv
        grp = np.arange(H) // (H // KV)
        fp = 12
        fpv = (fp + fv.reshape(KV, HD)[grp] - 8).reshape(-1)
        meta = {"host": {"pos": "i32", "ck": "i16", "cv": "i16"}, "state": ["ck", "cv"],
                "exp": {"q": exps(H * HD, 10, 12), "k": exps(KV * HD, 10, 12),
                        "v": exps(KV * HD, 10, 12), "ck": fk.tolist(), "cv": fv.tolist(),
                        "pv": fpv.tolist()},
                "layout": {"ck": [KV, HD], "cv": [KV, HD]},
                "test_fill": {"pos": pos}}
        if dma:                            # the caches as DMA states in the pool
            del meta["host"]["ck"], meta["host"]["cv"]
        ins = [vi("q", F32, [T, H * HD]), vi("k", F32, [T, KV * HD]), vi("v", F32, [T, KV * HD]),
               vi("pos", I32, [1])]
        nin = ""
        if n is not None:
            ins.append(vi("n", I32, [1]))
            meta["host"]["n"] = "i32"
            meta["test_fill"]["n"] = n
            nin = "n"
        inits = [nph.from_array(ck0.astype(np.float32), "ck"),
                 nph.from_array(cv0.astype(np.float32), "cv"),
                 nph.from_array(cos, "cos"), nph.from_array(sin, "sin")]
        m = model([node("LlmAttention", ["q", "k", "v", "pos", nin, "ck", "cv", "cos", "sin"],
                        ["pv"], num_heads=H, num_kv_heads=KV, head_dim=HD)],
                  ins, [vi("pv", F32, [T, H * HD])], inits, meta)
        return m, dict(cos=cos, sin=sin, ck0=ck0, cv0=cv0, grp=grp)

    def test_attention_on_host(self):
        for kw in ({}, {"n": 2}, {"pos": 10, "T": 4}, {"H": 3, "KV": 3}, {"HD": 8, "H": 2, "KV": 1},
                   {"dma": True}, {"dma": True, "n": 2}):
            m, _ = self._attention(**kw)
            with self.subTest(**kw):
                emulate(self, m, "attention", incoherent=kw.get("dma", False))

    def test_attention_semantics(self):
        m, d = self._attention(T=3, pos=4)
        g = OnnxGraph(m)
        sn = g.nodes[0]
        C, KVHD = d["ck0"].shape
        q = RNG.integers(-3000, 3000, (3, 64)) / 2048.0
        k = RNG.integers(-3000, 3000, (3, 32)) / 2048.0
        v = RNG.integers(-3000, 3000, (3, 32)) / 2048.0
        ck, cv = d["ck0"].copy(), d["cv0"].copy()
        out = sn.reference([q, k, v, [4], ck, cv], Q)
        # rows 4..6 written, others untouched
        np.testing.assert_array_equal(ck[:4], d["ck0"][:4])
        np.testing.assert_array_equal(ck[7:], d["ck0"][7:])
        self.assertFalse(np.array_equal(ck[4:7], d["ck0"][4:7]))
        # causality: row t reads keys <= 4 + t only
        ck2, cv2 = d["ck0"].copy(), d["cv0"].copy()
        ck2[7:] += 1.0
        cv2[7:] -= 1.0
        np.testing.assert_array_equal(sn.reference([q, k, v, [4], ck2, cv2], Q), out)
        # an independent computation of row 0, head 1 (group 0)
        HD, t, h, gq = 16, 0, 1, 0
        fq = np.asarray(json.loads(m.metadata_props[0].value)["exp"]["q"])
        qr = rope(q[t].reshape(4, HD)[h], d["cos"][4], d["sin"][4])
        K = ck[:5].reshape(5, 2, HD)[:, gq]
        V = cv[:5].reshape(5, 2, HD)[:, gq]
        s = dot8(qr[None, :] * K) * 0.25
        e = np.array([math.exp(x) for x in s - s.max()])
        p = e / np.cumsum(e)[-1]
        o = np.cumsum(p[:, None] * V, axis=0)[-1]
        f_pv = np.asarray(json.loads(m.metadata_props[0].value)["exp"]["pv"]).reshape(4, HD)[h]
        np.testing.assert_array_equal(out[t].reshape(4, HD)[h], Q.quantize_exp(o, f_pv))
        del fq
        # padded rows: n = 1 -> rows 1, 2 zero, only cache row 4 written
        ck3, cv3 = d["ck0"].copy(), d["cv0"].copy()
        m2, _ = self._attention(T=3, pos=4, n=1)
        sn2 = OnnxGraph(m2).nodes[0]
        out2 = sn2.reference([q, k, v, [4], [1], ck3, cv3], Q)
        self.assertTrue((out2[1:] == 0).all())
        np.testing.assert_array_equal(ck3[5:], d["ck0"][5:])
        # context clamp: pos = C - 1 writes one row, never beyond C
        ck4, cv4 = d["ck0"].copy(), d["cv0"].copy()
        out4 = sn.reference([q, k, v, [C - 1], ck4, cv4], Q)
        self.assertTrue((out4[1:] == 0).all())

    def test_attention_validation(self):
        m, _ = self._attention()
        d = json.loads(m.metadata_props[0].value)
        d["state"] = ["cv"]                                     # cache_k not a state
        del d["layout"]["ck"]
        m.metadata_props[0].value = json.dumps(d)
        with self.assertRaisesRegex(SchedulerError, "must be a state"):
            OnnxGraph(m)
        m, _ = self._attention()
        d = json.loads(m.metadata_props[0].value)
        del d["layout"]                                         # row-major caches
        m.metadata_props[0].value = json.dumps(d)
        with self.assertRaisesRegex(SchedulerError, "group-major"):
            OnnxGraph(m)


def fpga_attention_model(T=8, H=6, KV=2, HD=64, C=48, pos=5, n=None, kw=1, seed=0,
                         order="pipelined"):
    """LlmAttnPrep -> per group LlmAttnScores / LlmAttnSoftmax / LlmAttnPV ->
    LlmAttnMerge over DMA-state caches (group-major) holding random rows,
    output pv (DMA) — one layer of the frontend's FPGA prefill attention."""
    rng = np.random.default_rng(seed)
    G = H // KV
    half = HD // 2
    cos = rng.uniform(-1, 1, (C, half)).astype(np.float32)
    sin = rng.uniform(-1, 1, (C, half)).astype(np.float32)
    fq = rng.integers(5, 8, H)
    fk = rng.integers(7, 9, KV)
    fvc = rng.integers(6, 9, KV * HD)
    grp = np.arange(H) // G
    fs = fq + fk[grp] - 8
    fp = np.full(H, 12)
    fpv = (fp[:, None] + fvc.reshape(KV, HD)[grp] - 8).reshape(-1)
    ck0 = np.round(rng.normal(0, 900, (C, KV * HD))) / 2.0 ** np.repeat(fk, HD)
    cv0 = np.round(rng.normal(0, 900, (C, KV * HD))) / 2.0 ** fvc
    common = dict(num_heads=H, num_kv_heads=KV, head_dim=HD)
    nodes = [node("LlmAttnPrep", ["q", "k", "v", "pos", "n", "ck", "cv", "cos", "sin"], ["qx"],
                  qk_kw=kw, q_exp=[int(x) for x in fq], **common)]
    nodes[0].name = "prep"

    def qk(g):
        nd = node("LlmAttnScores", ["ck", "qx", "pos", "n"], [f"s{g}"], group=g, qk_kw=kw, **common)
        nd.name = f"qk{g}"
        nodes.append(nd)

    def sm_pv(g):
        hs = slice(g * G, (g + 1) * G)
        a = node("LlmAttnSoftmax", [f"s{g}", "pos", "n"], [f"p{g}"], group=g,
                 s_exp=[int(x) for x in fs[hs]], p_exp=[int(x) for x in fp[hs]], **common)
        b = node("LlmAttnPV", [f"p{g}", "cv", "pos", "n"], [f"o{g}"], group=g, **common)
        a.name, b.name = f"sm{g}", f"pv{g}"
        nodes.extend([a, b])
    if order == "pipelined":
        qk(0)
        if KV > 1:
            qk(1)
        for g in range(KV):
            sm_pv(g)
            if g + 2 < KV:
                qk(g + 2)
    else:
        for g in range(KV):
            qk(g)
            sm_pv(g)
    nodes.append(node("LlmAttnMerge", [f"o{g}" for g in range(KV)], ["pv"], **common))
    meta = {"host": {"pos": "i32", "n": "i32"}, "state": ["ck", "cv"],
            "layout": {"ck": [KV, HD], "cv": [KV, HD]},
            "exp": {"q": exps(H * HD, 10, 12), "k": exps(KV * HD, 10, 12),
                    "v": exps(KV * HD, 10, 12), "ck": np.repeat(fk, HD).tolist(),
                    "cv": fvc.tolist(), "pv": fpv.tolist(), "qx": 0,
                    **{f"{x}{g}": 0 for x in "spo" for g in range(KV)}},
            "test_fill": {"pos": pos, "n": T - 1 if n is None else n}}
    vis = ([vi("qx", F32, [KV, HD, G * T])]
           + [vi(f"s{g}", F32, [C, G * T]) for g in range(KV)]
           + [vi(f"p{g}", F32, [G * T, C]) for g in range(KV)]
           + [vi(f"o{g}", F32, [G * T, HD]) for g in range(KV)])
    inits = [nph.from_array(ck0.astype(np.float32), "ck"),
             nph.from_array(cv0.astype(np.float32), "cv"),
             nph.from_array(cos, "cos"), nph.from_array(sin, "sin")]
    ins = [vi("q", F32, [T, H * HD]), vi("k", F32, [T, KV * HD]), vi("v", F32, [T, KV * HD]),
           vi("pos", I32, [1]), vi("n", I32, [1])]
    return model(nodes, ins, [vi("pv", F32, [T, H * HD])], inits, meta, vis)


class TestFpgaAttention(unittest.TestCase):
    """The FPGA prefill attention chain: C (software ConvKernel for the two
    runtime-dimension calls) == simulation, the simulation == an independent
    float computation of the p12 formulas, keys16, the exp tables."""

    def test_chain_on_host(self):
        cases = ({}, {"kw": 2}, {"kw": 4}, {"pos": 1, "n": 8}, {"pos": 40, "n": 3},
                 {"pos": 44, "n": 8},                     # context clamp: 4 rows, keys16 = C
                 {"n": 0}, {"H": 4, "KV": 4, "HD": 32, "kw": 2},
                 {"H": 3, "KV": 1, "T": 16, "order": "serial"})
        for kw in cases:
            with self.subTest(**kw):
                emulate(self, fpga_attention_model(**kw), "fpga_attn", incoherent=True)

    def test_chain_semantics(self):
        """sim of the chain == the p12 attention written independently (the
        llm_study.Model policy pow2+sink+p12 formulas) on random rows."""
        T, H, KV, HD, C, pos, n = 8, 6, 2, 64, 48, 5, 6
        m = fpga_attention_model(T=T, H=H, KV=KV, HD=HD, C=C, pos=pos, n=n, kw=2)
        cg = CodeGenerator(OnnxGraph(m), model_path="c.onnx")
        g = cg._graph
        meta = json.loads(m.metadata_props[0].value)
        prep = g.nodes[0]
        states = cg.initial_states()
        ck0, cv0 = states["ck"].copy(), states["cv"].copy()
        q = RNG.integers(-2000, 2000, (T, H * HD)) / 2.0 ** np.asarray(meta["exp"]["q"])
        k = RNG.integers(-2000, 2000, (T, KV * HD)) / 2.0 ** np.asarray(meta["exp"]["k"])
        v = RNG.integers(-2000, 2000, (T, KV * HD)) / 2.0 ** np.asarray(meta["exp"]["v"])
        out = cg._forward_pass({"q": q, "k": k, "v": v, "pos": np.array([pos]),
                                "n": np.array([n])}, states=states)["pv"]
        # independent: caches, then per head q.K^T (floor), softmax p12, P.V (floor)
        ck, cv = states["ck"], states["cv"]
        fk = np.asarray(meta["exp"]["ck"])
        fvc = np.asarray(meta["exp"]["cv"])
        for t in range(n):
            kr = rope(k[t].reshape(KV, HD), prep.cos[pos + t], prep.sin[pos + t]).reshape(-1)
            np.testing.assert_array_equal(ck[pos + t], Q.quantize_exp(kr, fk))
            np.testing.assert_array_equal(cv[pos + t], Q.quantize_exp(v[t], fvc))
        np.testing.assert_array_equal(ck[:pos], ck0[:pos])
        np.testing.assert_array_equal(ck[pos + n:], ck0[pos + n:])
        G, S = H // KV, pos + n
        fq = prep.fq
        fpv = np.asarray(meta["exp"]["pv"]).reshape(H, HD)
        for t in range(n):
            for h in range(H):
                gq = h // G
                qr = rope(q[t].reshape(H, HD)[h], prep.cos[pos + t], prep.sin[pos + t])
                qraw = np.clip(np.round(qr * 2.0 ** fq[h]), -32768, 32767)
                kraw = ck[:pos + t + 1, gq * HD:(gq + 1) * HD] * 2.0 ** fk[gq * HD]
                sraw = np.floor(kraw @ qraw / 256.0)
                fs = int(fq[h] + fk[gq * HD] - 8)
                kk = (sraw.max() - sraw).astype(int)
                e = np.array([math.exp(-x / 2.0 ** fs * (1 / math.sqrt(HD))) for x in kk])
                praw = np.round(e / np.cumsum(e)[-1] * 4096.0)
                vraw = cv[:pos + t + 1, gq * HD:(gq + 1) * HD] * 2.0 ** fvc[gq * HD:(gq + 1) * HD]
                o = np.clip(np.floor(praw @ vraw / 256.0), -32768, 32767)
                np.testing.assert_array_equal(out[t].reshape(H, HD)[h], o / 2.0 ** fpv[h])
        self.assertTrue((out[n:] == 0).all())
        del S

    def test_runtime_geometry(self):
        """The q.K^T / P.V calls' registers: keys16 goes into out_ch / in_ch,
        the rest is fixed at codegen; the plan's estimates favour ConvKernel."""
        cg = CodeGenerator(OnnxGraph(fpga_attention_model(T=16, kw=2)), model_path="g.onnx")
        convs = [sn for sn in cg._graph.nodes if isinstance(sn, LlmAttnConvNode)]
        self.assertEqual([sn.kind for sn in convs], ["qk", "qk", "pv", "pv"])
        src = cg.generate_source()
        for sn in convs:
            call = sn.emit_call({})
            self.assertIn("llm_keys16((unsigned)pos[0], (unsigned)n[0], 16u, 48u)", call)
            self.assertIn(call, src)
            for keys, conv, mm in sn.est_cycles:
                self.assertLess(conv, mm, (sn.kind, keys))
        qk, pv = convs[0], convs[2]
        self.assertIn("1u, 32u, ", qk.conv_regs("_keys"))                 # in_ch = HD / kw
        self.assertIn("_keys, ", qk.conv_regs("_keys").split("\n")[1])    # out_ch
        self.assertTrue(pv.conv_regs("_keys").startswith("1u, _keys, 1u, 64u"))
        self.assertEqual((pv.out_h, pv.out_w), (1, 64))

    def test_keys16_c_equals_python(self):
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "k.c")
            with open(src, "w") as f:
                f.write("#include <stdint.h>\n#include <stdio.h>\n#include <stdlib.h>\n"
                        "#include <string.h>\n#include <math.h>\ntypedef uint16_t Data_t;\n"
                        + HOST_C_POOL + llm_c_helpers() +
                        "int main(void) {\n    unsigned p, n, T;\n"
                        "    for (T = 1; T <= 40; T += 13) for (p = 0; p <= 70; p++)"
                        " for (n = 0; n <= 45; n++)\n"
                        "        printf(\"%u\\n\", llm_keys16(p, n, T, 64u));\n    return 0;\n}\n")
            exe = os.path.join(td, "k")
            r = subprocess.run([host_emu.which_cc(), "-std=gnu99", "-O2", "-Wall", "-Werror",
                                "-Wno-unused-function", "-pthread", src, "-lm", "-o", exe],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            c = [int(x) for x in subprocess.run([exe], capture_output=True,
                                                text=True).stdout.split()]
            py = [keys16(p, n, T, 64)[1] for T in range(1, 41, 13) for p in range(71)
                  for n in range(46)]
            self.assertEqual(c, py)
            self.assertTrue(all(KEY_QUANTUM <= k <= 64 and k % KEY_QUANTUM == 0 for k in py))

    def test_sexp_tables(self):
        """The softmax exp tables (C fill over 4 threads == Python), all
        65 536 entries, for the score exponents and head dims in use."""
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "sexp.c")
            with open(src, "w") as f:
                f.write("#include <stdint.h>\n#include <stdio.h>\n#include <stdlib.h>\n"
                        "#include <string.h>\n#include <math.h>\ntypedef uint16_t Data_t;\n"
                        + HOST_C_POOL + llm_c_helpers() +
                        "int main(int argc, char **argv) {\n"
                        "    int f = atoi(argv[1]); double sc = atof(argv[2]);\n"
                        "    if (host_pool_init() != 0 || llm_sexp_table(f, sc) != 0) return 1;\n"
                        "    fwrite(_llm_sexp_tab[f - LLM_SEXP_EMIN], sizeof(double), 65536u, stdout);\n"
                        "    llm_sexp_free(f); host_pool_deinit(); return 0;\n}\n")
            exe = os.path.join(td, "sexp")
            r = subprocess.run([host_emu.which_cc(), "-std=gnu99", "-O2", "-Wall", "-Werror",
                                "-Wno-unused-function", "-pthread", "-ffp-contract=off", src,
                                "-lm", "-o", exe], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            for f, hd in ((5, 64), (6, 64), (8, 64), (9, 16), (7, 80)):
                sc = 1.0 / math.sqrt(hd)
                with self.subTest(f=f, hd=hd):
                    out = subprocess.run([exe, str(f), repr(sc)], capture_output=True,
                                         env=dict(os.environ, INFERENCE_HOST_THREADS="4"))
                    c = np.frombuffer(out.stdout, np.float64)
                    np.testing.assert_array_equal(c.view(np.uint64),
                                                  sexp_table(f, sc).view(np.uint64))


class TestSiluTableExhaustive(unittest.TestCase):
    """The C table fill (libm exp, host_parallel over 4 threads) equals the
    simulator's table for all 65 536 raw inputs of every gate exponent."""

    def test_tables(self):
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "silu.c")
            with open(src, "w") as f:
                f.write("#include <stdint.h>\n#include <stdio.h>\n#include <stdlib.h>\n"
                        "#include <string.h>\n#include <math.h>\ntypedef uint16_t Data_t;\n"
                        + HOST_C_POOL + llm_c_helpers() +
                        "int main(int argc, char **argv) {\n"
                        "    int f = argc > 1 ? atoi(argv[1]) : 8;\n"
                        "    if (host_pool_init() != 0 || llm_silu_table(f) != 0) return 1;\n"
                        "    fwrite(_llm_silu_tab[f - LLM_SILU_EMIN], sizeof(double), 65536u, stdout);\n"
                        "    llm_silu_free(f); host_pool_deinit(); return 0;\n}\n")
            exe = os.path.join(td, "silu")
            r = subprocess.run([host_emu.which_cc(), "-std=gnu99", "-O2", "-Wall", "-Wextra",
                                "-Werror", "-Wno-unused-function", "-pthread",
                                "-ffp-contract=off", src, "-lm", "-o", exe],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            for f in (7, 8, 10, 13, 15):
                with self.subTest(f=f):
                    out = subprocess.run([exe, str(f)], capture_output=True,
                                         env=dict(os.environ, INFERENCE_HOST_THREADS="4"))
                    c = np.frombuffer(out.stdout, np.float64)
                    np.testing.assert_array_equal(c.view(np.uint64),
                                                  silu_table(f).view(np.uint64))


if __name__ == "__main__":
    unittest.main()
