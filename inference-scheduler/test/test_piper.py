"""Piper (VITS) text to speech (doc/plans/TTS_PLAN.md §4): the chunk entry of
src/piper.py through the scheduler (Conv exponents, the axi.tts host ops of
src/tts_nodes.py) against the specification demo/tts/scripts/piper_vits.py
``chunk_forward``, bit for bit; the generated C against the simulation
(host_emu).  Random weights at Piper medium's shapes (the voice itself is
an untracked asset)."""

import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

import host_emu
from src.codegen import CodeGenerator
from src.graph import OnnxGraph
from src.piper import FLOW_FRAMES, PiperChunkFrontend, exponent_keys

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
        self.assertEqual(kinds, {"ConvNode": 66, "TtsPrepNode": 46, "TtsSumNode": 37, "TtsGateNode": 16,
                                 "TtsFlowOutNode": 4, "TtsInterleaveNode": 3, "TtsPcmNode": 1})
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
