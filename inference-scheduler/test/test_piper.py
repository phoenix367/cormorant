"""Piper (VITS) text to speech (doc/plans/TTS_PLAN.md §4, §6): the chunk entry
of src/piper.py through the scheduler (Conv exponents, the TTS host ops
of src/tts_nodes.py) against the specification demo/tts/scripts/piper_vits.py
``chunk_forward``, and the encode_<T> entries against ``encoder_forward``
(every bucket and length: the padding never reaches a valid row), bit for
bit; the generated C against the simulation (host_emu).  Random weights at
Piper medium's shapes (the voice itself is an untracked asset).  And the
library's C duration predictor (demo/tts/src/tts_dp.c) against
duration_predictor_seq."""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest

from unittest import mock

import numpy as np

import host_emu
from src import _vectorop_hw_config
from src.codegen import CodeGenerator
from src.graph import OnnxGraph
from src.piper import (FLOW_FRAMES, PiperChunkFrontend, PiperEncoderFrontend, encoder_weights, exponent_keys,
                       vop_exponents)

_SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        "demo", "tts", "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
import piper_vits as pv  # noqa: E402


def random_weights(seed=0):
    """The flow's and the decoder's weights at Piper medium's shapes."""
    rng = np.random.default_rng(seed)
    W = {}

    def w(name, *shape, scale=0.05):
        W[name] = rng.standard_normal(shape) * scale

    for fi in (0, 2, 4, 6):
        p = f"flow.flows.{fi}"
        w(f"{p}.pre.weight", 192, 96, 1)
        w(f"{p}.pre.bias", 192)
        w(f"{p}.post.weight", 96, 192, 1)
        w(f"{p}.post.bias", 96)
        for i in range(4):
            w(f"{p}.enc.in_layers.{i}.weight", 384, 192, 5, scale=0.03)
            w(f"{p}.enc.in_layers.{i}.bias", 384)
            o = 384 if i < 3 else 192
            w(f"{p}.enc.res_skip_layers.{i}.weight", o, 192, 1)
            w(f"{p}.enc.res_skip_layers.{i}.bias", o)
    w("dec.conv_pre.weight", 256, 192, 7, scale=0.02)
    w("dec.conv_pre.bias", 256)
    c = 256
    for i, (s, k) in enumerate(((8, 16), (8, 16), (4, 8))):
        w(f"dec.ups.{i}.weight", c, c // 2, k, scale=0.03)
        w(f"dec.ups.{i}.bias", c // 2)
        c //= 2
        for j, kk in enumerate((3, 5, 7)):
            for cc in range(2):
                w(f"dec.resblocks.{3 * i + j}.convs.{cc}.weight", c, c, kk, scale=0.04)
                w(f"dec.resblocks.{3 * i + j}.convs.{cc}.bias", c)
    w("dec.conv_post.weight", 1, c, 7, scale=0.1)
    # the text encoder
    w("emb", 256, 192, scale=0.3)
    for i in range(6):
        p = f"enc_p.encoder.attn_layers.{i}"
        for t in "qkvo":
            w(f"{p}.conv_{t}.weight", 192, 192, 1, scale=0.07)
            w(f"{p}.conv_{t}.bias", 192)
        w(f"{p}.emb_rel_k", 1, 9, 96, scale=0.1)
        w(f"{p}.emb_rel_v", 1, 9, 96, scale=0.1)
        f = f"enc_p.encoder.ffn_layers.{i}"
        w(f"{f}.conv_1.weight", 768, 192, 3, scale=0.04)
        w(f"{f}.conv_1.bias", 768)
        w(f"{f}.conv_2.weight", 192, 768, 3, scale=0.02)
        w(f"{f}.conv_2.bias", 192)
        for j in (1, 2):
            W[f"enc_p.encoder.norm_layers_{j}.{i}.gamma"] = 1.0 + rng.standard_normal(192) * 0.1
            w(f"enc_p.encoder.norm_layers_{j}.{i}.beta", 192)
    w("enc_p.proj.weight", 384, 192, 1, scale=0.07)
    w("enc_p.proj.bias", 384)
    # the stochastic duration predictor

    def dds(p):
        for i in range(3):
            w(f"{p}.convs_sep.{i}.weight", 192, 1, 3, scale=0.3)
            w(f"{p}.convs_sep.{i}.bias", 192)
            w(f"{p}.convs_1x1.{i}.weight", 192, 192, 1, scale=0.07)
            w(f"{p}.convs_1x1.{i}.bias", 192)
            for j in (1, 2):
                W[f"{p}.norms_{j}.{i}.gamma"] = 1.0 + rng.standard_normal(192) * 0.1
                w(f"{p}.norms_{j}.{i}.beta", 192)
    w("dp.pre.weight", 192, 192, 1, scale=0.07)
    w("dp.pre.bias", 192)
    dds("dp.convs")
    w("dp.proj.weight", 192, 192, 1, scale=0.07)
    w("dp.proj.bias", 192)
    for fi in (7, 5, 3):
        w(f"dp.flows.{fi}.pre.weight", 192, 1, 1, scale=0.5)
        w(f"dp.flows.{fi}.pre.bias", 192)
        dds(f"dp.flows.{fi}.convs")
        w(f"dp.flows.{fi}.proj.weight", 29, 192, 1, scale=0.05)
        w(f"dp.flows.{fi}.proj.bias", 29)
    w("dp.flows.0.m", 2, 1)
    W["dp.flows.0.exp_neg_logs"] = np.exp(-rng.standard_normal(2) * 0.1)
    return W


def exponents(seed=0):
    rng = np.random.default_rng(seed)
    return {k: int(rng.integers(8, 12)) for k in exponent_keys()}


_CG = None


def chunk_cg():
    global _CG
    if _CG is None:
        W, E = random_weights(), exponents()
        _CG = (W, E, CodeGenerator(OnnxGraph(PiperChunkFrontend(W, E).entry(), fuse_act=True,
                                             s2d_stem=True), model_path="piper_chunk.onnx"))
    return _CG


_ENC = None


def encoder_setup():
    """(W, EW, exponents): the spec calibrated on random ids (no input search: fast)."""
    global _ENC
    if _ENC is None:
        W = random_weights()
        EW = pv.encoder_weights(W)
        rng = np.random.default_rng(3)
        rec = {}
        for n in (40, 90):
            pv.encoder_float(EW, rng.integers(0, 256, n), rec)
        _ENC = (W, EW, pv.encoder_exponents(EW, rec, search=False))
    return _ENC


class TestPiperEncoder(unittest.TestCase):
    def test_weights_equal_the_specification(self):
        W, EW, _ = encoder_setup()
        mine = encoder_weights(W)
        self.assertEqual(set(mine), set(EW))
        for k in EW:
            np.testing.assert_array_equal(mine[k].astype(np.float64), EW[k], err_msg=k)

    def test_simulation_matches_specification(self):
        W, EW, E = encoder_setup()
        rng = np.random.default_rng(4)
        for T in (32, 64):
            cg = CodeGenerator(OnnxGraph(PiperEncoderFrontend(W, E, T).entry()), model_path="enc.onnx")
            for n in (T, T - 5, 17, 1):
                ids = rng.integers(0, 256, n)
                pad = np.zeros(T, np.int64)
                pad[:n] = ids
                out = cg._forward_pass({f"enc{T}.ids": pad, f"enc{T}.n": np.array([n])},
                                       keep=[f"enc{T}.x", f"enc{T}.stats"])
                x = np.asarray(out[f"enc{T}.x"]).reshape(T, 192)
                st = np.asarray(out[f"enc{T}.stats"]).reshape(T, 384)
                xs, ss = pv.encoder_forward(EW, E, ids)
                np.testing.assert_array_equal(x[:n].astype(np.float32), xs, err_msg=f"x T {T} n {n}")
                np.testing.assert_array_equal(st[:n].astype(np.float32), ss, err_msg=f"stats T {T} n {n}")
                self.assertFalse(x[n:].any() or st[n:].any(), f"padding rows T {T} n {n}")

    def test_graph(self):
        W, _, E = encoder_setup()
        g = OnnxGraph(PiperEncoderFrontend(W, E, 64).entry())
        kinds = {}
        for sn in g.nodes:
            kinds[type(sn).__name__] = kinds.get(type(sn).__name__, 0) + 1
        self.assertEqual(kinds, {"TtsEmbedNode": 1, "TtsRowPrepNode": 19, "MatmulConvNode": 37,
                                 "VitAttnPrepNode": 6, "LlmAttnConvNode": 24, "TtsAttnSoftmaxNode": 12,
                                 "TtsAttnMergeNode": 6, "TtsResNormNode": 12, "TtsEncOutNode": 2})

    @unittest.skipUnless(shutil.which("cc"), "needs cc")
    def test_generated_c_matches_simulation(self):
        W, _, E = encoder_setup()
        cg = CodeGenerator(OnnxGraph(PiperEncoderFrontend(W, E, 64).entry()), model_path="enc.onnx")
        with tempfile.TemporaryDirectory() as td:
            rc, out = host_emu.build_and_run(cg, os.path.join(td, "p"), timeout=1800)
        self.assertEqual(rc, 0, out[-3000:])
        self.assertIn("PASSED", out)


def _unit(on=True):
    """The platform gate of VectorOPKernel's softmax unit, patched."""
    return mock.patch.object(_vectorop_hw_config, "VECTOROP_SOFTMAX", on)


class TestPiperEncoderSoftmaxUnit(unittest.TestCase):
    """PiperEncoderFrontend(vsmx=True) (doc/plans/SOFTMAX_PLAN.md §5): the host
    adds the relative-key band (TtsAttnRelAdd), VectorOPKernel's softmax unit
    takes the softmax — against encoder_forward(vsmx_unit=True)."""

    def test_simulation_matches_specification(self):
        W, EW, E = encoder_setup()
        rng = np.random.default_rng(5)
        with _unit():
            for T in (32, 64):
                cg = CodeGenerator(OnnxGraph(PiperEncoderFrontend(W, E, T, vsmx=True).entry()),
                                   model_path="enc.onnx")
                for n in (T, T - 5, 17, 1):
                    ids = rng.integers(0, 256, n)
                    pad = np.zeros(T, np.int64)
                    pad[:n] = ids
                    out = cg._forward_pass({f"enc{T}.ids": pad, f"enc{T}.n": np.array([n])},
                                           keep=[f"enc{T}.x", f"enc{T}.stats"])
                    x = np.asarray(out[f"enc{T}.x"]).reshape(T, 192)
                    st = np.asarray(out[f"enc{T}.stats"]).reshape(T, 384)
                    xs, ss = pv.encoder_forward(EW, E, ids, vsmx_unit=True)
                    np.testing.assert_array_equal(x[:n].astype(np.float32), xs, err_msg=f"x T {T} n {n}")
                    np.testing.assert_array_equal(st[:n].astype(np.float32), ss, err_msg=f"stats T {T} n {n}")
                    self.assertFalse(x[n:].any() or st[n:].any(), f"padding rows T {T} n {n}")

    def test_differs_from_the_host_softmax(self):
        # the unit's P is not the host's (within one LSB): the policy really changes the numbers
        W, EW, E = encoder_setup()
        ids = np.random.default_rng(6).integers(0, 256, 40)
        self.assertFalse(np.array_equal(pv.encoder_forward(EW, E, ids)[0],
                                        pv.encoder_forward(EW, E, ids, vsmx_unit=True)[0]))

    def test_graph(self):
        W, _, E = encoder_setup()
        with _unit():
            g = OnnxGraph(PiperEncoderFrontend(W, E, 64, vsmx=True).entry())
        kinds = {}
        for sn in g.nodes:
            kinds[type(sn).__name__] = kinds.get(type(sn).__name__, 0) + 1
        self.assertEqual(kinds, {"TtsEmbedNode": 1, "TtsRowPrepNode": 19, "MatmulConvNode": 37,
                                 "VitAttnPrepNode": 6, "LlmAttnConvNode": 24, "TtsAttnRelAddNode": 12,
                                 "TtsAttnSoftmaxVopNode": 12, "TtsAttnMergeNode": 6, "TtsResNormNode": 12,
                                 "TtsEncOutNode": 2})

    def test_needs_the_unit(self):
        W, _, E = encoder_setup()
        with _unit(False), self.assertRaisesRegex(Exception, "softmax unit"):
            OnnxGraph(PiperEncoderFrontend(W, E, 32, vsmx=True).entry())

    @unittest.skipUnless(shutil.which("cc"), "needs cc")
    def test_generated_c_matches_simulation(self):
        W, _, E = encoder_setup()
        with _unit():
            cg = CodeGenerator(OnnxGraph(PiperEncoderFrontend(W, E, 64, vsmx=True).entry()),
                               model_path="enc.onnx")
            with tempfile.TemporaryDirectory() as td:
                rc, out = host_emu.build_and_run(cg, os.path.join(td, "p"), timeout=1800)
        self.assertEqual(rc, 0, out[-3000:])
        self.assertIn("PASSED", out)
        src = cg.generate_source()
        self.assertIn("tts_attn_rel_add(", src)
        self.assertIn("VECTOROP_SOFTMAX_T, 64u, 64u, 64u,", src)
        self.assertLessEqual(src.count("tts_attn_softmax("), 1)          # the helper's definition, no call


_DP_DRIVER = r"""
#include <stdio.h>
#include <stdlib.h>
#include "tts_dp.h"
int main(int argc, char **argv)
{
    int n = atoi(argv[2]);
    (void)argc;
    float *x = malloc(sizeof(float) * 192 * n);
    double *z = malloc(sizeof(double) * 2 * n), *lw = malloc(sizeof(double) * n);
    FILE *f = fopen(argv[3], "rb");
    if (fread(x, 4, (size_t)192 * n, f) != (size_t)192 * n) return 2;
    fclose(f);
    f = fopen(argv[4], "rb");
    if (fread(z, 8, (size_t)2 * n, f) != (size_t)2 * n) return 2;
    fclose(f);
    if (tts_dp_load(argv[1], (size_t)atol(argv[6])) != 0) return 3;
    if (tts_dp_run(x, n, z, lw, atoi(argv[7])) != 0) return 4;
    f = fopen(argv[5], "wb");
    fwrite(lw, 8, n, f);
    fclose(f);
    tts_dp_free();
    return 0;
}
"""


class TestPiperDuration(unittest.TestCase):
    """demo/tts/src/tts_dp.c (the library's duration predictor, TTS_PLAN §7)
    against piper_vits.duration_predictor_seq, bit for bit; the latter
    against the float duration_predictor."""

    def test_specification_is_the_float_predictor(self):
        W = random_weights()
        rng = np.random.default_rng(5)
        x = (rng.standard_normal((192, 50)) * 0.5).astype(np.float32).astype(np.float64)
        z = np.random.default_rng(6).standard_normal((2, 50)) * 0.8
        ref = pv.duration_predictor(W, x, 0.8, np.random.default_rng(6), True)
        got = pv.duration_predictor_seq(W, x, z)
        # the same model: ULP-level differences (sum order, x 1/sd, libm exp)
        # that random flows amplify (the voice's weights: ~1e-14)
        np.testing.assert_allclose(got, ref, rtol=0, atol=1e-6)

    @unittest.skipUnless(shutil.which("cc"), "needs cc")
    def test_c_matches_specification(self):
        W = random_weights()
        src = os.path.join(os.path.dirname(_SCRIPTS), "src")
        flat = pv.dp_flat(W)
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "drv.c"), "w") as f:
                f.write(_DP_DRIVER)
            exe = os.path.join(td, "drv")
            r = subprocess.run(["cc", "-O2", "-ffp-contract=off", "-std=gnu99", "-Wall", "-Wextra", "-Werror", "-pthread",
                                "-I", src, os.path.join(src, "tts_dp.c"), os.path.join(td, "drv.c"), "-lm", "-o", exe],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            flat.astype("<f4").tofile(os.path.join(td, "dp.dat"))
            rng = np.random.default_rng(7)
            for n, threads in ((1, 1), (7, 4), (61, 4), (61, 1)):
                x = (rng.standard_normal((192, n)) * 0.5).astype(np.float32)
                z = rng.standard_normal((2, n)) * 0.8
                x.astype("<f4").tofile(os.path.join(td, "x.bin"))
                z.astype("<f8").tofile(os.path.join(td, "z.bin"))
                r = subprocess.run([exe, os.path.join(td, "dp.dat"), str(n), os.path.join(td, "x.bin"),
                                    os.path.join(td, "z.bin"), os.path.join(td, "lw.bin"), str(flat.size), str(threads)],
                                   capture_output=True, text=True)
                self.assertEqual(r.returncode, 0, r.stderr)
                got = np.fromfile(os.path.join(td, "lw.bin"), "<f8")
                want = pv.duration_predictor_seq(W, x, z)
                self.assertTrue(np.array_equal(got.view(np.uint64), want.view(np.uint64)), f"n {n} threads {threads}")


class TestPiperChunk(unittest.TestCase):
    def test_simulation_matches_specification(self):
        W, E, cg = chunk_cg()
        rng = np.random.default_rng(1)
        zp = (rng.standard_normal((192, FLOW_FRAMES)) * 0.8).astype(np.float32)
        for lo, hi in ((64, 256), (0, 256), (-64, 121), (70, 90)):
            z = zp.copy()
            z[:, :max(lo, 0)] = 0
            z[:, max(min(hi, FLOW_FRAMES), 0):] = 0
            sim = cg._forward_pass({"zp": z.astype(np.float64), "lo": np.array([lo]),
                                    "hi": np.array([hi])}, keep=["pcm"])["pcm"]
            spec = pv.chunk_forward(W, E, z, lo, hi)
            np.testing.assert_array_equal(np.asarray(sim).astype(np.int64), spec.astype(np.int64),
                                          err_msg=f"lo {lo} hi {hi}")

    def test_host_sums_variant(self):
        """vop_sums=False (the library before OFFLOAD_PLAN §2.2: calibrated
        decoder exponents, three-input sums on the host) against its spec."""
        W, E, _ = chunk_cg()
        g = OnnxGraph(PiperChunkFrontend(W, E, vop_sums=False).entry(), fuse_act=True, s2d_stem=True)
        cg = CodeGenerator(g, model_path="piper_chunk_host.onnx")
        rng = np.random.default_rng(4)
        z = (rng.standard_normal((192, FLOW_FRAMES)) * 0.8).astype(np.float32)
        sim = cg._forward_pass({"zp": z.astype(np.float64), "lo": np.array([0]), "hi": np.array([256])},
                               keep=["pcm"])["pcm"]
        np.testing.assert_array_equal(np.asarray(sim).astype(np.int64),
                                      pv.chunk_forward(W, E, z, 0, 256, vop_sums=False).astype(np.int64))
        self.assertFalse(np.array_equal(pv.chunk_forward(W, E, z, 0, 256, vop_sums=False),
                                        pv.chunk_forward(W, E, z, 0, 256)))

    def test_vop_exponents(self):
        """One exponent per decoder stage, the smallest; the rest unchanged;
        the scheduler's copy equals the specification's."""
        _, E, _ = chunk_cg()
        E2 = pv.vop_exponents(E)
        self.assertEqual(E2, vop_exponents(E))
        for i in range(3):
            keys = [f"dec.up{i}"] + [f"dec.rb{i}{j}.{k}{c}" for j in range(3) for k in "cy" for c in range(2)]
            self.assertEqual({E2[k] for k in keys}, {min(E[k] for k in keys)})
        self.assertEqual({k: v for k, v in E2.items() if "dec.up" not in k and "dec.rb" not in k},
                         {k: v for k, v in E.items() if "dec.up" not in k and "dec.rb" not in k})

    def test_stitched_chunks_equal_one_chunk(self):
        W, E, _ = chunk_cg()
        rng = np.random.default_rng(2)
        T = 300
        zp = (rng.standard_normal((192, T)) * 0.8).astype(np.float32)
        st = pv.synthesize_chunked(W, E, zp)
        big = np.zeros((192, T + 128), np.float32)
        big[:, 64:64 + T] = zp
        one = pv.chunk_forward(W, E, big, 64, 64 + T, dec_off=32)[32 * 256: 32 * 256 + T * 256]
        np.testing.assert_array_equal(st, one)

    def test_graph_census(self):
        _, _, cg = chunk_cg()
        kinds = {}
        for sn in cg._graph.nodes:
            kinds[type(sn).__name__] = kinds.get(type(sn).__name__, 0) + 1
        # 37 sums + the 3 split convs' tap-group sums; the decoder's 21 on VectorOP
        # (and any flow sum whose random exponents happen to be equal)
        sums = kinds.pop("TtsSumNode", 0) + kinds.get("TtsAddVopNode", 0)
        self.assertEqual(sums, 40)
        self.assertGreaterEqual(kinds.pop("TtsAddVopNode"), 21)
        self.assertEqual(kinds, {"ConvNode": 66, "TtsPrepNode": 46, "TtsGateNode": 16,
                                 "TtsFlowOutNode": 4, "TtsInterleaveNode": 3, "TtsPcmNode": 1})
        dec = [sn for sn in cg._graph.nodes if sn.onnx_node.name.startswith("dec.rb")
               and sn.onnx_node.op_type == "TtsSum"]
        self.assertTrue(dec and all(type(sn).__name__ == "TtsAddVopNode" for sn in dec))
        for sn in cg._graph.nodes:
            if type(sn).__name__ == "ConvNode":
                self.assertLessEqual((sn.kw - 1) * sn.dilation_w + 1, 64)
                self.assertLessEqual(sn.out_w * (-(-sn.out_ch // 16) * 16), 65536)

    @unittest.skipUnless(shutil.which("cc"), "needs cc")
    def test_generated_c_matches_simulation(self):
        _, _, cg = chunk_cg()
        with tempfile.TemporaryDirectory() as td:
            rc, out = host_emu.build_and_run(cg, os.path.join(td, "p"), timeout=1800)
        self.assertEqual(rc, 0, out[-3000:])
        self.assertIn("PASSED", out)


if __name__ == "__main__":
    unittest.main()
