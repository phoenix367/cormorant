"""The axi.llm host ops (src/llm_nodes.py) one at a time.

Each op is a single-node graph (host-memory and DMA inputs / outputs with
per-channel power-of-two exponents) scheduled, compiled with -Werror and run
against the software kernels (test/host_emu.py): the C helper must equal
``reference()`` — the simulator's — bit for bit, for cacheable and staged
buffers and 1 / 4 host threads.  Plus: the SiLU table of every relevant gate
exponent exhaustively (C vs Python, all 65 536 inputs); the references
against independent formulas; the attention op's causality, padded rows
and context clamp.
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
from src.llm_nodes import LLM_DOMAIN, dot8, llm_c_helpers, rope, silu_table
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


def emulate(testcase, m, name):
    cg = CodeGenerator(OnnxGraph(m, fuse_act=True), model_path=name + ".onnx")
    for cached, threads in ((True, 4), (False, 1)):
        with testcase.subTest(cached=cached, threads=threads), tempfile.TemporaryDirectory() as td:
            rc, out = host_emu.build_and_run(cg, td, cached=cached, threads=threads, min_elems=1)
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

    def _attention(self, T=3, H=4, KV=2, HD=16, C=12, pos=4, n=None, seed=0):
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
                "test_fill": {"pos": pos}}
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
        for kw in ({}, {"n": 2}, {"pos": 10, "T": 4}, {"H": 3, "KV": 3}, {"HD": 8, "H": 2, "KV": 1}):
            m, _ = self._attention(**kw)
            with self.subTest(**kw):
                emulate(self, m, "attention")

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
        m.metadata_props[0].value = json.dumps(d)
        with self.assertRaisesRegex(SchedulerError, "must be a state"):
            OnnxGraph(m)


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
