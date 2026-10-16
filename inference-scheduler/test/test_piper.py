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
from src.rowops_nodes import chains_enabled as rowops_chains
from src.rowops_nodes import enabled as rowops_unit

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
    """The chunk with TtsPrep as host ops and the gates in libm (the unit's FOLD,
    GATE and SCALE off whatever the platform says; fold_chunk_cg is the FOLD
    variant, a3_chunk_cg the GATE / SCALE one)."""
    global _CG
    if _CG is None:
        from src import _rowops_hw_config
        W, E = random_weights(), exponents()
        with mock.patch.multiple(_rowops_hw_config, ROWOPS_FOLD=False, ROWOPS_GATE=False, ROWOPS_SCALE=False):
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


class TestPiperEncoderRowops(unittest.TestCase):
    """PiperEncoderFrontend(vsmx=True, rowops=True) (doc/plans/HOSTOPS_PLAN.md §6, RPU_PLAN
    §3.1): the residual LayerNorms as TtsResAdd (the int32 sum on the row-ops unit) +
    TtsLayerNorm, the attention prep and the relative-key band on the unit (rel_kernel,
    the default with both) — against encoder_forward(vsmx_unit=True, rowops_unit=True,
    rel_kernel=True)."""

    @staticmethod
    def _units(on=True):
        from src import _rowops_hw_config
        return mock.patch.object(_rowops_hw_config, "ROWOPS_UNIT", on)

    def test_simulation_matches_specification(self):
        W, EW, E = encoder_setup()
        rng = np.random.default_rng(7)
        with _unit(), self._units():
            for T in (32, 64):
                cg = CodeGenerator(OnnxGraph(PiperEncoderFrontend(W, E, T, vsmx=True, rowops=True).entry()),
                                   model_path="enc.onnx")
                for n in (T, T - 5, 17, 1):
                    ids = rng.integers(0, 256, n)
                    pad = np.zeros(T, np.int64)
                    pad[:n] = ids
                    out = cg._forward_pass({f"enc{T}.ids": pad, f"enc{T}.n": np.array([n])},
                                           keep=[f"enc{T}.x", f"enc{T}.stats"])
                    x = np.asarray(out[f"enc{T}.x"]).reshape(T, 192)
                    st = np.asarray(out[f"enc{T}.stats"]).reshape(T, 384)
                    xs, ss = pv.encoder_forward(EW, E, ids, vsmx_unit=True, rowops_unit=True, rel_kernel=True)
                    np.testing.assert_array_equal(x[:n].astype(np.float32), xs, err_msg=f"x T {T} n {n}")
                    np.testing.assert_array_equal(st[:n].astype(np.float32), ss, err_msg=f"stats T {T} n {n}")
                    self.assertFalse(x[n:].any() or st[n:].any(), f"padding rows T {T} n {n}")

    def test_host_band_matches_its_specification(self):
        """rel_kernel=False: the band stays the host op (TtsAttnRelAdd) — against
        encoder_forward(rowops_unit=True) without rel_kernel."""
        W, EW, E = encoder_setup()
        rng = np.random.default_rng(9)
        with _unit(), self._units():
            T = 32
            cg = CodeGenerator(OnnxGraph(PiperEncoderFrontend(W, E, T, vsmx=True, rowops=True,
                                                              rel_kernel=False).entry()), model_path="enc.onnx")
            self.assertEqual(sum(type(sn).__name__ == "TtsAttnRelAddNode" for sn in cg._graph.nodes), 12)
            self.assertFalse(any(type(sn).__name__ == "TtsRelBandRoNode" for sn in cg._graph.nodes))
            for n in (T, 17):
                ids = rng.integers(0, 256, n)
                pad = np.zeros(T, np.int64)
                pad[:n] = ids
                out = cg._forward_pass({f"enc{T}.ids": pad, f"enc{T}.n": np.array([n])}, keep=[f"enc{T}.x"])
                x = np.asarray(out[f"enc{T}.x"]).reshape(T, 192)
                xs, _ = pv.encoder_forward(EW, E, ids, vsmx_unit=True, rowops_unit=True)
                np.testing.assert_array_equal(x[:n].astype(np.float32), xs, err_msg=f"n {n}")

    def test_differs_from_the_host_norm(self):
        W, EW, E = encoder_setup()
        ids = np.random.default_rng(8).integers(0, 256, 40)
        host = pv.encoder_forward(EW, E, ids, vsmx_unit=True)[0]
        unit = pv.encoder_forward(EW, E, ids, vsmx_unit=True, rowops_unit=True)[0]
        rel = pv.encoder_forward(EW, E, ids, vsmx_unit=True, rowops_unit=True, rel_kernel=True)[0]
        self.assertFalse(np.array_equal(host, unit))
        self.assertFalse(np.array_equal(unit, rel))

    def test_rel_kernel_needs_both(self):
        W, _, E = encoder_setup()
        for kw in (dict(vsmx=False, rowops=True), dict(vsmx=True, rowops=False)):
            with self.assertRaisesRegex(ValueError, "rel_kernel needs vsmx and rowops"):
                PiperEncoderFrontend(W, E, 32, rel_kernel=True, **kw)

    def test_graph(self):
        W, _, E = encoder_setup()
        with _unit(), self._units():
            g = OnnxGraph(PiperEncoderFrontend(W, E, 64, vsmx=True, rowops=True).entry())
        kinds = {}
        for sn in g.nodes:
            kinds[type(sn).__name__] = kinds.get(type(sn).__name__, 0) + 1
        self.assertEqual(kinds, {"TtsEmbedNode": 1, "TtsRowPrepRoNode": 19, "MatmulConvNode": 37,
                                 "VitAttnPrepRoNode": 6, "LlmAttnConvNode": 24, "TtsRelBandRoNode": 12,
                                 "TtsAttnSoftmaxVopNode": 12, "TtsAttnMergeNode": 6, "ResAddBiasRoNode": 12,
                                 "LayerNorm32RoNode": 12, "TtsEncOutNode": 2})
        for sn in g.nodes:
            if type(sn).__name__ == "ResAddBiasRoNode":
                self.assertFalse(sn.a32)                     # x int16 at f_x, the sum int32 at max(f_x, f_y)
                self.assertEqual(sn.output.dma, "i32")
            elif type(sn).__name__ == "VitAttnPrepRoNode":
                self.assertEqual((sn.T, sn.C, sn.CR), (64, 80, 16))   # the relative keys behind the 64 keys
            elif type(sn).__name__ == "TtsRelBandRoNode":
                calls = sn.kernel_calls({})
                self.assertEqual(sum(c.count for c in calls), 10)     # 9 diagonals + the add
                self.assertEqual((sn.T, sn.CR, sn.W), (64, 16, 4))
            elif type(sn).__name__ == "LlmAttnConvNode" and sn.kind == "scores":
                self.assertEqual(list(sn.output.shape), [80, 64])

    def test_needs_the_unit(self):
        W, _, E = encoder_setup()
        with _unit(), self._units(False), self.assertRaisesRegex(Exception, "row-ops unit"):
            OnnxGraph(PiperEncoderFrontend(W, E, 32, vsmx=True, rowops=True).entry())

    @unittest.skipUnless(shutil.which("cc"), "needs cc")
    def test_generated_c_matches_simulation(self):
        W, _, E = encoder_setup()
        with _unit(), self._units():
            cg = CodeGenerator(OnnxGraph(PiperEncoderFrontend(W, E, 64, vsmx=True, rowops=True).entry()),
                               model_path="enc.onnx")
            with tempfile.TemporaryDirectory() as td:
                rc, out = host_emu.build_and_run(cg, os.path.join(td, "p"), timeout=1800)
        self.assertEqual(rc, 0, out[-3000:])
        self.assertIn("PASSED", out)
        src = cg.generate_source()
        # 12 adds + 12 norms + per layer the prep's 7 jobs reading entry tensors (q, the q image, k per
        # head x 2, v, the V image per head x 2; the relative-key rows read a weight), the row preps' 14
        # (xin 2, c1 6, h 6) and the band's 2 x 10 (9 diagonals + the add) + the 2 of the stats' input
        self.assertEqual(src.count("run_rowops(enc64_"), 24 + 6 * (7 + 14 + 20) + 2)
        self.assertEqual(src.count("run_rowops(v_enc_l"), 6)                      # the relative-key rows
        self.assertNotIn("tts_attn_rel_add(", src.split("inference_run_")[-1])
        self.assertEqual(src.count("run_rowops(rowops_tables,"), 1)  # TLOAD once, in inference_init
        self.assertNotIn("tts_resnorm(", src.split("inference_run_")[-1])


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
        from src import _rowops_hw_config
        W, E, _ = chunk_cg()
        with mock.patch.multiple(_rowops_hw_config, ROWOPS_GATE=False, ROWOPS_SCALE=False):
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
        # (with the row-ops unit the flow sums — h + res and the four-input post sums —
        # are TtsSumRoNodes and the interleaves TtsInterleaveRoNodes; the /3 stage sums stay host ops)
        sums = kinds.pop("TtsSumNode", 0) + kinds.pop("TtsSumRoNode", 0) + kinds.get("TtsAddVopNode", 0)
        self.assertEqual(sums, 40)
        self.assertGreaterEqual(kinds.pop("TtsAddVopNode"), 21)
        il = "TtsInterleaveRoNode" if rowops_unit() else "TtsInterleaveNode"
        self.assertEqual(kinds, {"ConvNode": 66, "TtsPrepNode": 46, "TtsGateNode": 16,
                                 "TtsFlowOutNode": 4, il: 3, "TtsPcmNode": 1})
        dec = [sn for sn in cg._graph.nodes if sn.onnx_node.name.startswith("dec.rb")
               and sn.onnx_node.op_type == "TtsSum"]
        self.assertTrue(dec and all(type(sn).__name__ == "TtsAddVopNode" for sn in dec))
        for sn in cg._graph.nodes:
            if type(sn).__name__ == "ConvNode":
                self.assertLessEqual((sn.kw - 1) * sn.dilation_w + 1, 64)
                self.assertLessEqual(sn.out_w * (-(-sn.out_ch // 16) * 16), 65536)

    def test_post_sums_on_the_unit(self):
        """The flows' post sums (RPU_PLAN §14.4): unmasked four-input TtsSums on the
        unit, one chain of three ADDX through one shared int32 scratch state, the
        emitted calls the priced ones.  The mask could go because the post conv is
        1 x 1 and its output only reaches TtsFlowOut (which masks x1 itself)."""
        from test_perf_calls import _Eval, node_emitted_calls
        from src.rowops_nodes import TtsSumRoNode
        from src.tts_nodes import TtsFlowOutNode
        _, _, cg = chunk_cg()
        if not rowops_unit():
            self.skipTest("the platform has no row-ops unit")
        nodes = cg._graph.nodes
        post = [sn for sn in nodes if isinstance(sn, TtsSumRoNode) and len(sn.ch0s) == 4]
        self.assertEqual(len(post), 4)
        self.assertEqual({sn.tmp.onnx_name for sn in post}, {"rowops_sum_scratch_49152"})
        self.assertTrue(all(sn.tmp.is_state and sn.tmp.dma == "i32" for sn in post))
        ev = _Eval(cg.generate_header())
        for sn in post:
            self.assertEqual(sn.chain_jobs(cg._layouts), 3)
            self.assertEqual(sn.kernel_calls(cg._layouts), node_emitted_calls(sn, cg, ev), sn.describe())
            conv = [c for c in nodes if sn.output in c.inputs]
            self.assertEqual(len(conv), 1)
            self.assertEqual((type(conv[0]).__name__, conv[0].kh, conv[0].kw), ("ConvNode", 1, 1))
            users = [u for u in nodes if conv[0].output in u.inputs]
            self.assertEqual([type(u) for u in users], [TtsFlowOutNode])
        if rowops_chains():
            self.assertEqual(sorted(n for i, n in cg._ro_chains.items() if nodes[i] in post), [3] * 4)

    @unittest.skipUnless(shutil.which("cc"), "needs cc")
    def test_generated_c_matches_simulation(self):
        _, _, cg = chunk_cg()
        with tempfile.TemporaryDirectory() as td:
            rc, out = host_emu.build_and_run(cg, os.path.join(td, "p"), timeout=1800)
        self.assertEqual(rc, 0, out[-3000:])
        self.assertIn("PASSED", out)


_FOLD_CG = None


def _fold(on=True):
    """The platform gates of the row-ops unit and its FOLD op, patched (GATE and
    SCALE off: the library with libm's gates)."""
    from src import _rowops_hw_config
    return mock.patch.multiple(_rowops_hw_config, ROWOPS_UNIT=on, ROWOPS_FOLD=on, ROWOPS_GATE=False,
                               ROWOPS_SCALE=False)


def fold_chunk_cg():
    """chunk_cg's graph with the row-ops unit's FOLD (HOSTOPS_PLAN §11)."""
    global _FOLD_CG
    if _FOLD_CG is None:
        W, E, _ = chunk_cg()
        with _fold():
            _FOLD_CG = (W, E, CodeGenerator(OnnxGraph(PiperChunkFrontend(W, E).entry(), fuse_act=True,
                                                      s2d_stem=True), model_path="piper_chunk_fold.onnx"))
    return _FOLD_CG


class TestPiperChunkFold(unittest.TestCase):
    """TtsPrep as the row-ops unit's FOLD (rowops_nodes.FoldRoNode): the chunk still
    equals chunk_forward bit for bit, the generated C its simulation."""

    def test_simulation_matches_specification(self):
        W, E, cg = fold_chunk_cg()
        rng = np.random.default_rng(5)
        zp = (rng.standard_normal((192, FLOW_FRAMES)) * 0.8).astype(np.float32)
        for lo, hi in ((64, 256), (0, 256), (-64, 121), (70, 90), (300, 400)):
            z = zp.copy()
            z[:, :max(min(lo, FLOW_FRAMES), 0)] = 0
            z[:, max(min(hi, FLOW_FRAMES), 0):] = 0
            with _fold():
                sim = cg._forward_pass({"zp": z.astype(np.float64), "lo": np.array([lo]),
                                        "hi": np.array([hi])}, keep=["pcm"])["pcm"]
            np.testing.assert_array_equal(np.asarray(sim).astype(np.int64),
                                          pv.chunk_forward(W, E, z, lo, hi).astype(np.int64),
                                          err_msg=f"lo {lo} hi {hi}")

    def test_census(self):
        """Every int16 TtsPrep whose rule is an exact 1/m with a coarser output is a
        FOLD; the float32 sources (and, with these random exponents, the finer outputs)
        stay host ops."""
        from src.rowops_nodes import FoldRoNode, fold_ineligible
        from src.tts_nodes import TtsPrepNode
        _, _, cg = fold_chunk_cg()
        folds = [sn for sn in cg._graph.nodes if isinstance(sn, FoldRoNode)]
        hosts = [sn for sn in cg._graph.nodes if isinstance(sn, TtsPrepNode)]
        self.assertEqual(len(folds) + len(hosts), 46)
        self.assertGreaterEqual(len(folds), 10)
        with _fold():
            self.assertEqual({fold_ineligible(sn) for sn in hosts}, {"a float32 source", "a finer output exponent"})
        self.assertEqual(sum(1 for sn in hosts if sn.src_f32), 5)
        self.assertTrue(any(sn.m == 10 for sn in folds) and any(sn.m == 1 for sn in folds))
        self.assertTrue(any(sn.W % 8 for sn in folds) and any(sn.s0 < 0 for sn in folds))

    def test_kernel_calls_are_the_emitted_ones(self):
        from test_perf_calls import _Eval, node_emitted_calls
        from src.rowops_nodes import FoldRoNode
        _, _, cg = fold_chunk_cg()
        ev = _Eval(cg.generate_header())
        for sn in cg._graph.nodes:
            if isinstance(sn, FoldRoNode):
                self.assertEqual(sn.kernel_calls(cg._layouts), node_emitted_calls(sn, cg, ev), sn.describe())

    def test_off_without_the_op(self):
        from src.rowops_nodes import FoldRoNode
        _, _, cg = chunk_cg()
        self.assertFalse(any(isinstance(sn, FoldRoNode) for sn in cg._graph.nodes))

    @unittest.skipUnless(shutil.which("cc"), "needs cc")
    def test_generated_c_matches_simulation(self):
        _, _, cg = fold_chunk_cg()
        with tempfile.TemporaryDirectory() as td, _fold():
            rc, out = host_emu.build_and_run(cg, os.path.join(td, "p"), timeout=1800)
        self.assertEqual(rc, 0, out[-3000:])
        self.assertIn("PASSED", out)


_A3_CG = None


def _a3(on=True):
    """The row-ops unit with FOLD, chains, GATE and SCALE (RPU_PLAN §14), patched."""
    from src import _rowops_hw_config
    return mock.patch.multiple(_rowops_hw_config, ROWOPS_UNIT=on, ROWOPS_FOLD=on, ROWOPS_CHAINS=on,
                               ROWOPS_GATE=on, ROWOPS_SCALE=on)


def a3_chunk_cg():
    """The chunk on the unit with GATE and SCALE: the gates as GateRoNodes, the stage
    averages split (PiperChunkFrontend(split_avg=True))."""
    global _A3_CG
    if _A3_CG is None:
        W, E, _ = chunk_cg()
        with _a3():
            _A3_CG = (W, E, CodeGenerator(OnnxGraph(PiperChunkFrontend(W, E, split_avg=True).entry(), fuse_act=True,
                                                    s2d_stem=True), model_path="piper_chunk_a3.onnx"))
    return _A3_CG


class TestPiperChunkGateScale(unittest.TestCase):
    """RPU_PLAN §14 (A3a.3): the gates on the unit's GATE equal chunk_forward(gate_q=True),
    the split stage averages the unsplit ones (exact sums), bit for bit."""

    def test_simulation_matches_specification(self):
        W, E, cg = a3_chunk_cg()
        rng = np.random.default_rng(14)
        zp = (rng.standard_normal((192, FLOW_FRAMES)) * 0.8).astype(np.float32)
        for lo, hi in ((64, 256), (0, 256), (-64, 121), (70, 90)):
            z = zp.copy()
            z[:, :max(lo, 0)] = 0
            z[:, max(min(hi, FLOW_FRAMES), 0):] = 0
            with _a3():
                sim = cg._forward_pass({"zp": z.astype(np.float64), "lo": np.array([lo]),
                                        "hi": np.array([hi])}, keep=["pcm"])["pcm"]
            np.testing.assert_array_equal(np.asarray(sim).astype(np.int64),
                                          pv.chunk_forward(W, E, z, lo, hi, gate_q=True).astype(np.int64),
                                          err_msg=f"lo {lo} hi {hi}")
        self.assertFalse(np.array_equal(pv.chunk_forward(W, E, zp, 0, 256, gate_q=True),
                                        pv.chunk_forward(W, E, zp, 0, 256)))

    def test_census(self):
        """16 GATE jobs, no host gate; the three averages as T (int16 + int16 -> int32)
        and the scaled ADDX ((T + y2) * G >> sh_c, rowops_spec.div_scale); every TtsSum
        on a kernel."""
        from src import rowops_spec as rs
        from src.rowops_nodes import GateRoNode, TtsSumRoNode
        _, _, cg = a3_chunk_cg()
        kinds = {}
        for sn in cg._graph.nodes:
            kinds[type(sn).__name__] = kinds.get(type(sn).__name__, 0) + 1
        self.assertEqual(kinds.get("GateRoNode"), 16)
        self.assertNotIn("TtsGateNode", kinds)
        self.assertNotIn("TtsSumNode", kinds)
        sums = [sn for sn in cg._graph.nodes if isinstance(sn, TtsSumRoNode)]
        t = [sn for sn in sums if sn.kinds[2] == rs.K_I32]
        avg = [sn for sn in sums if sn.G]
        self.assertEqual((len(t), len(avg)), (3, 3))
        for sn in avg:
            self.assertEqual(sn.kinds, (rs.K_I32, rs.K_I16, rs.K_I16))
            self.assertEqual((sn.G, sn.sh_c), rs.div_scale(3, sn.f_out - max(sn.f_ins), 17))
            self.assertLessEqual(sn.n, rs.N_MAX)
        self.assertTrue(all(isinstance(sn, GateRoNode) and sn.nr <= rs.N_MAX
                            for sn in cg._graph.nodes if sn.onnx_node.op_type == "TtsGate"))

    def test_t_runs_under_resblock_2(self):
        """T = y0 + y1 comes right after resblock 2's first prep and before its conv,
        so the event stream runs it beside that conv."""
        _, _, cg = a3_chunk_cg()
        names = [sn.onnx_node.name for sn in cg._graph.nodes]
        for i in range(3):
            ti = next(k for k, n in enumerate(names) if n.startswith(f"dec.st{i}.t.sum"))
            prep = next(k for k, n in enumerate(names) if n.startswith(f"dec.rb{i}2.c0.prep"))
            conv = next(k for k, n in enumerate(names) if n.startswith(f"dec.rb{i}2.c0#"))
            self.assertEqual((prep + 1, conv), (ti, ti + 1), names[prep:conv + 1])

    def test_kernel_calls_are_the_emitted_ones(self):
        from test_perf_calls import _Eval, node_emitted_calls
        from src.rowops_nodes import GateRoNode, TtsSumRoNode
        _, _, cg = a3_chunk_cg()
        ev = _Eval(cg.generate_header())
        for sn in cg._graph.nodes:
            if isinstance(sn, (GateRoNode, TtsSumRoNode)):
                self.assertEqual(sn.kernel_calls(cg._layouts), node_emitted_calls(sn, cg, ev), sn.describe())

    def test_split_needs_scale(self):
        """The split averages without the unit's SCALE are refused (the host op takes no
        int32); without split_avg nothing changes."""
        from src.nodes import SchedulerError
        W, E, _ = chunk_cg()
        from src import _rowops_hw_config
        with _a3(), mock.patch.object(_rowops_hw_config, "ROWOPS_SCALE", False):
            with self.assertRaises(SchedulerError) as cm:
                OnnxGraph(PiperChunkFrontend(W, E, split_avg=True).entry(), fuse_act=True, s2d_stem=True)
            self.assertIn("SCALE", str(cm.exception))

    @unittest.skipUnless(shutil.which("cc"), "needs cc")
    def test_generated_c_matches_simulation(self):
        _, _, cg = a3_chunk_cg()
        with tempfile.TemporaryDirectory() as td, _a3():
            rc, out = host_emu.build_and_run(cg, os.path.join(td, "p"), timeout=1800)
        self.assertEqual(rc, 0, out[-3000:])
        self.assertIn("PASSED", out)


if __name__ == "__main__":
    unittest.main()
