# Copyright 2026 Ivan Gubochkin
# SPDX-License-Identifier: Apache-2.0

"""V-JEPA 2 (doc/plans/VJEPA2_PLAN.md): the ``clip`` entry of src/vjepa2.py through
the scheduler (VitAttnPrep with the 3-D RoPE on the row-ops unit, the query-major
attention in query parts — LlmAttnScores ``rows`` — the row-mode softmax and GELU
erf on VectorOPKernel, the attentive pooler's VjCrossAttn on the host, VjResCopy
into and out of the int32 residual, VjLogits) against the specification
demo/video_classification/scripts/vjepa2_q.py (vjepa2_q.VJEPA2 with the same
formats), bit for bit.  A small random V-JEPA 2 — 2 encoder layers 64 wide (2
heads of 32: RoPE segments of 10 dims), a pooler of one self-attention layer and
the cross layer, 10 classes, 4 frames of 128 x 128 (128 tokens in two row pieces
and two query parts) — with formats the specification calibrates on synthetic
clips (the checkpoint is an untracked asset); the generated C against the
simulation (host_emu)."""

import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

import host_emu
from src.codegen import CodeGenerator
from src.graph import OnnxGraph
from src.rowops_nodes import enabled as rowops_unit
from src.smx_nodes import enabled as softmax_unit
from src.vjepa2 import VJepa2Config, VJepa2Formats, VJepa2Frontend, patches, rope_raw

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SCRIPTS = os.path.join(_REPO, "demo", "video_classification", "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
import vjepa2_q as vq  # noqa: E402

CFG = {
    "hidden_act": "gelu", "qkv_bias": True, "use_SiLU": False, "in_chans": 3, "hidden_size": 64,
    "num_hidden_layers": 2, "num_attention_heads": 2, "mlp_ratio": 2.0, "layer_norm_eps": 1e-6,
    "patch_size": 16, "tubelet_size": 2, "frames_per_clip": 4, "crop_size": 128, "num_pooler_layers": 1,
    "id2label": {str(i): f"class {i}" for i in range(10)},
}
PREP = {"size": {"shortest_edge": 146}, "crop_size": {"height": 128, "width": 128}, "do_center_crop": True,
        "resample": 2, "do_rescale": True, "do_normalize": True, "image_mean": [0.485, 0.456, 0.406],
        "image_std": [0.229, 0.224, 0.225]}
POLICY = "pow2+p12+vgelu+vsmx+rowops"
PIECES = ((64, "conv"), (64, "conv"))


def random_weights(seed=0):
    """The checkpoint's tensors (Hugging Face names) at the config's shapes, scaled by fan-in."""
    rng = np.random.default_rng(seed)
    D, F, P, C = 64, 128, 16, 10
    W = {}

    def w(name, *shape, fan=None, scale=1.0):
        W[name] = (rng.standard_normal(shape) * scale / np.sqrt(fan or shape[-1])).astype(np.float32)

    def b(name, n, scale=0.05):
        W[name] = (scale * rng.standard_normal(n)).astype(np.float32)

    def g(name, n):
        W[name] = (1.0 + 0.1 * rng.standard_normal(n)).astype(np.float32)

    def block(pre, names, lin, cross=False):
        for nm in names:
            g(pre + nm + ".weight", D)
            b(pre + nm + ".bias", D)
        for key in ("k", "v") if cross else ("q", "k", "v", "o"):
            w(pre + lin[key] + ".weight", D, D)
            b(pre + lin[key] + ".bias", D)
        w(pre + lin["f1"] + ".weight", F, D)
        b(pre + lin["f1"] + ".bias", F)
        w(pre + lin["f2"] + ".weight", D, F)
        b(pre + lin["f2"] + ".bias", D)

    w(vq.ENC + "embeddings.patch_embeddings.proj.weight", D, 3, 2, P, P, fan=3 * 2 * P * P, scale=2.0)
    b(vq.ENC + "embeddings.patch_embeddings.proj.bias", D)
    for li in range(2):
        block(f"{vq.ENC}layer.{li}.", ("norm1", "norm2"), vq.ENC_LIN)
    g(vq.ENC + "layernorm.weight", D)
    b(vq.ENC + "layernorm.bias", D)
    block(f"{vq.POOL}self_attention_layers.0.", ("layer_norm1", "layer_norm2"), vq.POOL_LIN)
    W[vq.POOL + "query_tokens"] = rng.standard_normal((1, 1, D)).astype(np.float32)
    pre = vq.POOL + "cross_attention_layer."
    block(pre, ("layer_norm1", "layer_norm2"), vq.CROSS_LIN, cross=True)
    w(pre + "cross_attn.q_proj.weight", D, D)
    b(pre + "cross_attn.q_proj.bias", D)
    w("classifier.weight", C, D)
    b("classifier.bias", C)
    return W


def clip(seed, n=4, s=128):
    """A synthetic clip (uint8 [n][s][s][3]): gradients, blocks that move, noise."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:s, 0:s]
    base = np.stack([x * 255.0 / s, y * 255.0 / s, (x + y) * 127.0 / s], -1)
    blocks = [(rng.integers(0, s - 24), rng.integers(0, s - 24), rng.uniform(0, 255, 3)) for _ in range(5)]
    out = []
    for t in range(n):
        img = base.copy()
        for k, (r0, c0, col) in enumerate(blocks):
            r, c = (r0 + 3 * t * (k % 3)) % (s - 16), (c0 + 5 * t) % (s - 16)
            img[r:r + 16, c:c + 16] = col
        out.append(np.clip(img + rng.normal(0, 8, img.shape), 0, 255))
    return np.stack(out).astype(np.uint8)


_SETUP = None


def setup():
    """(vjepa2_q.VCfg, W, KW, the spec, its formats JSON): calibrated in float on synthetic clips."""
    global _SETUP
    if _SETUP is None:
        td = tempfile.mkdtemp()
        try:
            with open(os.path.join(td, "config.json"), "w") as f:
                json.dump(CFG, f)
            with open(os.path.join(td, "video_preprocessor_config.json"), "w") as f:
                json.dump(PREP, f)
            cfg = vq.VCfg(td)
        finally:
            shutil.rmtree(td)
        W = random_weights()
        KW = vq.kernel_weights(W, cfg)
        rec = vq.VJEPA2(cfg, W, record_ch=True, KW=KW)
        for k in range(3):
            rec.forward(clip(10 + k))
        pol = vq.POLICIES[POLICY]
        fmt = vq.make_formats(pol, rec.stats.ch, KW, cfg)
        _SETUP = (cfg, W, KW, vq.VJEPA2(cfg, W, pol, fmt, KW=KW), vq.formats_json(fmt, POLICY, pol))
    return _SETUP


def frontend():
    _, W, _, _, js = setup()
    cfg = VJepa2Config.from_dicts(CFG, PREP)
    return VJepa2Frontend(cfg, W, VJepa2Formats(js, cfg), name="vjepa2_test", pieces=PIECES, parts=2)


def run_entry(cg, cfg, x):
    """The entry's simulation of clip x -> (logits [C], the kept tensors)."""
    pt = patches(x, cfg).astype(np.float64)
    o = cg._forward_pass({"clip.patches.p0": pt[:64], "clip.patches.p1": pt[64:]}, states=cg.initial_states(),
                         keep=["vj.xf.p0", "vj.xf.p1", "clip.logits"])
    return np.asarray(o["clip.logits"]).reshape(-1), o


@unittest.skipUnless(rowops_unit() and softmax_unit(), "needs the row-ops unit and the softmax unit")
class TestVJepa2Clip(unittest.TestCase):

    def test_config(self):
        c = VJepa2Config.from_dicts(CFG, PREP)
        self.assertEqual((c.D, c.L, c.H, c.HD, c.FF, c.N, c.rot, c.C, c.patch_k), (64, 2, 2, 32, 128, 128, 10, 10,
                                                                                  1536))
        self.assertEqual((c.lf, c.lp, c.lc, c.lk), (2, [3], 4, 5))
        cfg = setup()[0]
        self.assertEqual((cfg.N, cfg.rot, cfg.lk), (c.N, c.rot, c.lk))

    def test_rope_tables_match_specification(self):
        cfg, W, KW, spec, _ = setup()
        cr, sr = rope_raw(VJepa2Config.from_dicts(CFG, PREP))
        np.testing.assert_array_equal(cr, spec.CR)
        np.testing.assert_array_equal(sr, spec.SR)

    def test_simulation_matches_specification(self):
        cfg, _, _, spec, _ = setup()
        c = VJepa2Config.from_dicts(CFG, PREP)
        cg = CodeGenerator(OnnxGraph(frontend().entry()), model_path="clip.onnx")
        for seed in (1, 2):
            x = clip(seed)
            tr = {}
            want = spec.forward(x, trace=tr)
            got, o = run_entry(cg, c, x)
            xf = np.concatenate([np.asarray(o["vj.xf.p0"]).reshape(64, -1), np.asarray(o["vj.xf.p1"]).reshape(64, -1)])
            np.testing.assert_array_equal(xf, tr["xf"], err_msg=f"encoder output, clip {seed}")
            np.testing.assert_array_equal(got, np.asarray(want, np.float64).astype(np.float32).astype(np.float64),
                                          err_msg=f"logits, clip {seed}")
            self.assertEqual(int(np.argmax(got)), int(np.argmax(want)))

    def test_graph(self):
        kinds = {}
        for sn in OnnxGraph(frontend().entry()).nodes:
            kinds[type(sn).__name__] = kinds.get(type(sn).__name__, 0) + 1
        L, P, H, R = 2, 1, 2, 2                        # encoder layers, pooler self layers, heads, query parts
        self.assertEqual(kinds.get("VitAttnPrepRoNode"), L + P)
        self.assertEqual(kinds.get("LlmAttnConvNode"), (L + P) * H * R * 2)
        self.assertEqual(kinds.get("VitAttnSoftmaxVopNode"), (L + P) * H * R)
        self.assertEqual(kinds.get("ResCopyRoNode"), 2 + 2 + 1)     # the embedding, the pooler's input, xc
        self.assertEqual(kinds.get("VjCrossAttnNode"), 1)
        self.assertEqual(kinds.get("VjLogitsNode"), 1)
        self.assertEqual(kinds.get("VitGeluVopNode"), (L + P) * 2 + 1)
        self.assertEqual(kinds.get("LayerNorm32RoNode"), (L + P) * 2 * 2 + 2 + 2 + 1)
        for host in ("VitGeluNode", "VitAttnSoftmaxNode", "VitLayerNormNode", "VitResAddNode", "VitAttnPrepNode"):
            self.assertNotIn(host, kinds)
        rope = [sn for sn in OnnxGraph(frontend().entry()).nodes if type(sn).__name__ == "VitAttnPrepRoNode"]
        self.assertEqual([sn.rope for sn in rope], [True] * L + [False] * P)

    @unittest.skipUnless(shutil.which("cc"), "needs cc")
    def test_generated_c_matches_simulation(self):
        m = frontend().entry()
        for incoherent in (False, True):        # True: the syncs move the data (VjCrossAttn's reads)
            cg = CodeGenerator(OnnxGraph(m), model_path="m.onnx")
            with tempfile.TemporaryDirectory() as td:
                rc, out = host_emu.build_and_run(cg, os.path.join(td, "p"), timeout=1800, incoherent=incoherent)
            self.assertEqual(rc, 0, out[-3000:])
            self.assertIn("PASSED", out)


def video(seed, n, h, w):
    """A synthetic decoded video (uint8 [n][h][w][3]): the clip() pattern at another size."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w]
    base = np.stack([x * 255.0 / w, y * 255.0 / h, (x + y) * 127.0 / (w + h)], -1)
    out = []
    for t in range(n):
        img = base.copy()
        r, c = (5 * t) % (h - 12), (7 * t) % (w - 12)
        img[r:r + 12, c:c + 12] = rng.uniform(0, 255, 3)
        out.append(np.clip(img + rng.normal(0, 6, img.shape), 0, 255))
    return np.stack(out).astype(np.uint8)


@unittest.skipUnless(rowops_unit() and softmax_unit(), "needs the row-ops unit and the softmax unit")
@unittest.skipUnless(shutil.which("cc"), "needs cc")
class TestVJepa2Library(unittest.TestCase):
    """The library (demo/video_classification: generate_vjepa2_project.py, vjepa2_api.c,
    vjepa2_bench.c, vjepa2_lib.py) on the host emulation: decoded videos of both frame
    samplers' branches (fewer and more than 64 frames), upscaled and downscaled
    (antialiased) frames — every logit equal to the specification's VJEPA2.forward of
    make_clip, the repetitions and the re-open identical; the frame sampler and the
    resize through ctypes equal to vjepa2_q's."""

    def test_library_matches_specification(self):
        import generate_vjepa2_project as gvp
        import vjepa2_host_emu
        cfg, W, _, spec, js = setup()
        c = VJepa2Config.from_dicts(CFG, PREP)
        videos = [video(1, 7, 100, 150), video(2, 70, 96, 64), video(3, 3, 300, 200)]
        with tempfile.TemporaryDirectory() as td:
            proj = os.path.join(td, "p")
            s = gvp.generate(proj, c, W, VJepa2Formats(js, c), lib="vjepa2_test", pieces=PIECES, parts=2)
            self.assertEqual(s["nodes"]["clip"] > 0, True)
            glue = open(os.path.join(proj, "test", "vjepa2_glue.h")).read()
            self.assertIn("#define VJ_CLASSES  10", glue)
            self.assertIn('"class 9"', glue)
            exe = vjepa2_host_emu.build(proj, td)
            res, lg = vjepa2_host_emu.run_bench(exe, td, videos, c.C)
            lib = vjepa2_host_emu.build(proj, td, shared=True)
            sys.path.insert(0, os.path.join(_REPO, "demo", "video_classification"))
            from vjepa2_lib import VJepa2
            vj = VJepa2(lib)
            got = vj.classify(videos[2])
            for n in (1, 2, 5, 17, 63, 64, 65, 300):
                # (one frame: the specification's index -1 is numpy's last frame, the library's n - 1)
                np.testing.assert_array_equal(vj.sample_frames(n), np.mod(vq.sample_frames(n, c.frames), n),
                                              err_msg=f"n {n}")
            for v in videos:
                np.testing.assert_array_equal(vj.frame(v[0]), vq.center_crop(vq.resize_short(v[0], c.short), c.S))
            self.assertEqual((vj.frames, vj.crop, vj.classes, vj.labels[3]), (4, 128, 10, "class 3"))
            vj.close()
        self.assertTrue(res["reopen"]["identical"])
        np.testing.assert_array_equal(got, lg[2])
        for k, v in enumerate(videos):
            self.assertTrue(res["videos"][k]["reps_identical"])
            want = spec.forward(vq.make_clip(v, cfg)[0])
            np.testing.assert_array_equal(lg[k], np.asarray(want, np.float64).astype(np.float32), err_msg=f"video {k}")


if __name__ == "__main__":
    unittest.main()
