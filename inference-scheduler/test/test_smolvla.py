# Copyright 2026 Ivan Gubochkin
# SPDX-License-Identifier: Apache-2.0

"""SmolVLA (doc/plans/VLA_PLAN.md): the vision, image_<c>, prefix_<C> and expert
entries of src/smolvla.py through the scheduler (the VLA host ops and row-ops
nodes of src/vla_nodes.py, the prefix-LM softmax on VectorOPKernel's unit with
the state row on the host) against the specification
demo/vla/scripts/smolvla_q.py (SmolVLA with the same formats), bit for bit: the
velocity of every denoising step over the caches the prefix entry wrote, for two
cameras and a short task.  A small random SmolVLA — a vision encoder of 2
layers 64 wide on 128 x 128 images (16 image tokens per camera), a VLM and an
expert of 4 layers (64 / 48 wide, 4 / 2 heads of 16; the expert's odd layers
cross-attend), 10 actions, 3 steps — with formats the specification calibrates
on synthetic frames (the checkpoint is an untracked asset); the generated C
against the simulation (host_emu)."""

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
from src.smolvla import SmolVLAConfig, SmolVLAFormats, SmolVLAFrontend
from src.smx_nodes import enabled as softmax_unit
from src.vit import patches

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SCRIPTS = os.path.join(_REPO, "demo", "vla", "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
import smolvla_q as sq  # noqa: E402

POLICY_CFG = {
    "type": "smolvla", "chunk_size": 10, "n_action_steps": 10, "num_steps": 3, "max_state_dim": 8,
    "max_action_dim": 8, "resize_imgs_with_padding": [128, 128], "tokenizer_max_length": 8,
    "attention_mode": "cross_attn", "add_image_special_tokens": False, "prefix_length": 0, "empty_cameras": 0,
    "adapt_to_pi_aloha": False, "num_expert_layers": 0, "num_vlm_layers": 4, "self_attn_every_n_layers": 2,
    "expert_width_multiplier": 0.75, "min_period": 0.004, "max_period": 4.0,
    "input_features": {"observation.state": {"type": "STATE", "shape": [6]},
                       "observation.images.camera1": {"type": "VISUAL", "shape": [3, 128, 128]},
                       "observation.images.camera2": {"type": "VISUAL", "shape": [3, 128, 128]}},
    "output_features": {"action": {"type": "ACTION", "shape": [6]}},
}
VLM_CFG = {
    "scale_factor": 2,
    "text_config": {"hidden_size": 64, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
                    "intermediate_size": 128, "num_hidden_layers": 6, "rms_norm_eps": 1e-5, "rope_theta": 100000,
                    "vocab_size": 100},
    "vision_config": {"hidden_size": 64, "num_attention_heads": 4, "num_hidden_layers": 2,
                      "intermediate_size": 128, "patch_size": 16, "image_size": 128, "layer_norm_eps": 1e-6},
}
POLICY = "pow2+p12+vgelu+vsmx+rowops"
VP = sq.VLM_PFX + "vision_model."


def random_weights(seed=0):
    """The checkpoint's tensors (lerobot names) at the configs' shapes, scaled by fan-in."""
    rng = np.random.default_rng(seed)
    cfg = sq.VLACfg(POLICY_CFG, VLM_CFG)
    v, t, e = cfg.vision, cfg.text, cfg.expert
    W = {}

    def w(name, *shape, fan=None, scale=1.0):
        W[name] = (rng.standard_normal(shape) * scale / np.sqrt(fan or shape[-1])).astype(np.float32)

    def b(name, n, scale=0.05):
        W[name] = (scale * rng.standard_normal(n)).astype(np.float32)

    def g(name, n):
        W[name] = (1.0 + 0.1 * rng.standard_normal(n)).astype(np.float32)

    w(VP + "embeddings.patch_embedding.weight", v.D, 3, v.P, v.P, fan=3 * v.P * v.P, scale=2.0)
    b(VP + "embeddings.patch_embedding.bias", v.D)
    w(VP + "embeddings.position_embedding.weight", v.N, v.D, scale=3.0)
    for li in range(v.L):
        p = f"{VP}encoder.layers.{li}."
        for k in ("q", "k", "v", "out"):
            w(p + f"self_attn.{k}_proj.weight", v.D, v.D)
            b(p + f"self_attn.{k}_proj.bias", v.D)
        for nm in ("layer_norm1", "layer_norm2"):
            g(p + nm + ".weight", v.D)
            b(p + nm + ".bias", v.D)
        w(p + "mlp.fc1.weight", v.F, v.D)
        b(p + "mlp.fc1.bias", v.F)
        w(p + "mlp.fc2.weight", v.D, v.F)
        b(p + "mlp.fc2.bias", v.D)
    g(VP + "post_layernorm.weight", v.D)
    b(VP + "post_layernorm.bias", v.D)
    w(sq.VLM_PFX + "connector.modality_projection.proj.weight", t.D, v.D * cfg.scale ** 2, scale=0.1)
    w(sq.TXT_PFX + "embed_tokens.weight", 100, t.D, scale=0.05)
    g(sq.TXT_PFX + "norm.weight", t.D)
    kv = t.KV * t.HD
    for m, pfx, D, FF in (("text", sq.TXT_PFX, t.D, t.F), ("expert", sq.EXP_PFX, e.D, e.F)):
        for li in range(t.L):
            p = f"{pfx}layers.{li}."
            g(p + "input_layernorm.weight", D)
            g(p + "post_attention_layernorm.weight", D)
            w(p + "self_attn.q_proj.weight", t.H * t.HD, D)
            cross = m == "expert" and not cfg.self_layer(li)
            w(p + "self_attn.k_proj.weight", kv, kv if cross else D)
            w(p + "self_attn.v_proj.weight", kv, kv if cross else D)
            w(p + "self_attn.o_proj.weight", D, t.H * t.HD)
            w(p + "mlp.gate_proj.weight", FF, D)
            w(p + "mlp.up_proj.weight", FF, D)
            w(p + "mlp.down_proj.weight", D, FF)
    g(sq.EXP_PFX + "norm.weight", e.D)
    w("model.state_proj.weight", t.D, cfg.max_state)
    b("model.state_proj.bias", t.D)
    w("model.action_in_proj.weight", e.D, cfg.max_action)
    b("model.action_in_proj.bias", e.D)
    w("model.action_time_mlp_in.weight", e.D, 2 * e.D)
    b("model.action_time_mlp_in.bias", e.D)
    w("model.action_time_mlp_out.weight", e.D, e.D)
    b("model.action_time_mlp_out.bias", e.D)
    w("model.action_out_proj.weight", cfg.max_action, e.D)
    b("model.action_out_proj.bias", cfg.max_action)
    return W


def frame(seed, h=48, w=80):
    """A synthetic camera frame (uint8 [h][w][3]): gradients, a few blocks, noise —
    not square, so resize_with_pad pads."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w]
    img = np.stack([x * 255.0 / w, y * 255.0 / h, (x + y) * 127.0 / (w + h)], -1)
    for _ in range(4):
        r0, c0 = rng.integers(0, h - 8), rng.integers(0, w - 8)
        img[r0:r0 + 8, c0:c0 + 8] = rng.uniform(0, 255, 3)
    return np.clip(img + rng.normal(0, 6, img.shape), 0, 255).astype(np.uint8)


def task(seed, n=5):
    return np.random.default_rng(seed).integers(3, 100, n)


def state(seed):
    return np.random.default_rng(seed).standard_normal(6).astype(np.float32)


def noise(seed, cfg):
    return np.random.default_rng(seed).standard_normal((cfg.chunk, cfg.max_action)).astype(np.float32)


_SETUP = None


def setup():
    """(VLACfg, W, the spec, its formats JSON): calibrated in float on synthetic frames."""
    global _SETUP
    if _SETUP is None:
        W = random_weights()
        cfg = sq.VLACfg(POLICY_CFG, VLM_CFG)
        rec = sq.SmolVLA(cfg, W, record_ch=True)
        for k in range(3):
            rec.sample([frame(10 + 2 * k), frame(11 + 2 * k)], task(20 + k), state(30 + k), noise(40 + k, cfg))
        pol = sq.POLICIES[POLICY]
        fmt = sq.make_formats(pol, rec.vision.stats.ch, rec.text.stats.ch, rec.expert.stats.ch, cfg, rec.KW, W)
        spec = sq.SmolVLA(cfg, W, pol, fmt)
        _SETUP = (cfg, W, spec, sq.formats_json(fmt, POLICY, pol))
    return _SETUP


def frontend():
    _, W, _, js = setup()
    cfg = SmolVLAConfig.from_dicts(POLICY_CFG, VLM_CFG)
    return SmolVLAFrontend(cfg, W, SmolVLAFormats(js, cfg), name="smolvla_test")


@unittest.skipUnless(rowops_unit() and softmax_unit(), "needs the row-ops unit and the softmax unit")
class TestSmolVLAEntries(unittest.TestCase):

    def test_config(self):
        c = SmolVLAConfig.from_dicts(POLICY_CFG, VLM_CFG)
        self.assertEqual((c.L, c.D, c.De, c.H, c.KV, c.HD, c.Fe, c.img), (4, 64, 48, 4, 2, 16, 256, 16))
        self.assertEqual((c.prefix_rows(1), c.prefix_rows(2), c.cache_rows, c.expert_rows), (64, 64, 64, 16))
        self.assertEqual([c.self_layer(i) for i in range(4)], [True, False, True, False])

    def test_simulation_matches_specification(self):
        cfg, _, spec, _ = setup()
        fe = frontend()
        es = fe.entries()
        cgs = {k: CodeGenerator(OnnxGraph(m), model_path=f"{k}.onnx") for k, m in es.items()}
        states = {}
        for cg in cgs.values():
            for k, v in cg.initial_states().items():
                states.setdefault(k, v)
        frames = [frame(1), frame(2)]
        ids, st, x = task(3), state(4), noise(5, cfg).astype(np.float64)
        feats = spec.image_features(frames)
        for cam, f in enumerate(frames):
            pix = np.asarray(sq.pixels(f, cfg, 0), np.uint8)
            cgs["vision"]._forward_pass({"vision.patches": patches(pix, cfg.vision.P).astype(np.float64)},
                                        states=states, keep=[])
            cgs[f"image_{cam}"]._forward_pass({}, states=states, keep=[])
            np.testing.assert_array_equal(np.asarray(states[f"vla.img{cam}"]).reshape(-1, cfg.text.D), feats[cam],
                                          err_msg=f"image features of camera {cam}")
        n = 2 * cfg.img_tokens + len(ids) + 1
        idp = np.zeros(cfg.tok_max)
        idp[:len(ids)] = ids
        stp = np.zeros(cfg.max_state)
        stp[:6] = st
        cgp = CodeGenerator(OnnxGraph(fe.prefix_entry(2, output=True)), model_path="p.onnx")
        o = cgp._forward_pass({"prefix_2.ids": idp, "prefix_2.n": np.array([n]),
                               "prefix_2.state": stp, "prefix_2.zero": np.array([0])},
                              states=states, keep=["prefix_2.kx"])
        K, V, V0 = spec.prefix(spec.prefix_rows(feats, ids, st))
        kx = np.asarray(o["prefix_2.kx"]).reshape(-1, cfg.text.KV * cfg.text.HD)[:n]
        for li in range(cfg.text.L - 1):
            kc = np.asarray(states[f"kv.k.l{li}"]).reshape(-1, cfg.text.KV * cfg.text.HD)[:n]
            np.testing.assert_array_equal(kc, K[li].transpose(1, 0, 2).reshape(n, -1), err_msg=f"K cache {li}")
        Kx, Vx = spec.cross_kv(K, V0)
        np.testing.assert_array_equal(kx, Kx[max(Kx)].transpose(1, 0, 2).reshape(n, -1), err_msg="the kx output")
        for li in Kx:
            ck = np.asarray(states[f"vla.xk.l{li}"]).reshape(-1, cfg.text.KV * cfg.text.HD)[:n]
            np.testing.assert_array_equal(ck, Kx[li].transpose(1, 0, 2).reshape(n, -1), err_msg=f"K' {li}")
        _, dt = sq.step_times(cfg)
        for step in range(cfg.steps):
            o = cgs["expert"]._forward_pass({"expert.x": x, "expert.step": np.array([step]),
                                             "expert.pos": np.array([n]), "expert.n": np.array([cfg.chunk]),
                                             "expert.xpos": np.array([n - cfg.chunk]),
                                             "expert.zero": np.array([0])}, states=states, keep=["expert.v"])
            v = spec.velocity(x, step, K, V, Kx, Vx)
            np.testing.assert_array_equal(np.asarray(o["expert.v"]).reshape(cfg.chunk, -1), v,
                                          err_msg=f"velocity of step {step}")
            x = (x + dt * v).astype(np.float32).astype(np.float64)
        a = spec.sample(frames, ids, st, noise(5, cfg))
        np.testing.assert_array_equal(x, a)

    def test_graph(self):
        fe = frontend()
        kinds = {}
        for k, m in fe.entries().items():
            if k == "vision":
                continue
            for sn in OnnxGraph(m).nodes:
                kinds[type(sn).__name__] = kinds.get(type(sn).__name__, 0) + 1
        L, KV = 4, 2
        # prefix_1 and prefix_2: layers 0..2 attend (state rows per group), the odd layers' K' / V'
        self.assertEqual(kinds.get("VlaStateAttnNode"), 2 * 3 * KV)
        self.assertEqual(kinds.get("VlaAttnMergeNode"), 2 * 3)
        self.assertEqual(kinds.get("VlaPrepRoNode"), 2 * 2 * 2 + 2)     # k, v (2 + 2 per prefix) + q (expert)
        self.assertEqual(kinds.get("AsrCrossPrepRoNode"), 2 * 2)
        self.assertEqual(kinds.get("VlaPrefixNode"), 2)
        self.assertEqual(kinds.get("VlaTimeSiluNode"), 1)
        self.assertEqual(kinds.get("LlmAttnSoftmaxVopNode"), (2 * 3 + L) * KV)
        self.assertNotIn("LlmAttnSoftmaxNode", kinds)

    @unittest.skipUnless(shutil.which("cc"), "needs cc")
    def test_generated_c_matches_simulation(self):
        fe = frontend()
        for m in (fe.prefix_entry(2, output=True), fe.expert_entry()):
            for incoherent in (False, True):        # True: the caches' syncs move the data (VlaStateAttn's reads)
                cg = CodeGenerator(OnnxGraph(m), model_path="m.onnx")
                with tempfile.TemporaryDirectory() as td:
                    rc, out = host_emu.build_and_run(cg, os.path.join(td, "p"), timeout=1800,
                                                     incoherent=incoherent)
                self.assertEqual(rc, 0, out[-3000:])
                self.assertIn("PASSED", out)


@unittest.skipUnless(rowops_unit() and softmax_unit(), "needs the row-ops unit and the softmax unit")
@unittest.skipUnless(shutil.which("cc"), "needs cc")
class TestSmolVLALibrary(unittest.TestCase):
    """The library (demo/vla: generate_vla_project.py, vla_api.c, vla_bench.c) on the host
    emulation: one sample of one camera (a frame already square: no resize), one of two
    cameras (non-square frames: resized with padding) — every action equal to the
    specification's SmolVLA.sample, the repetitions and the re-open identical."""

    def test_library_matches_specification(self):
        import generate_vla_project as gvp
        import vla_host_emu
        cfg, W, spec, js = setup()
        c = SmolVLAConfig.from_dicts(POLICY_CFG, VLM_CFG)
        square = np.asarray(sq.pixels(frame(7, 128, 128), cfg, 0), np.uint8)
        samples = [([square], task(8, 3), state(9), noise(10, cfg)),
                   ([frame(11), frame(12, 60, 40)], task(13, 8), state(14), noise(15, cfg))]
        with tempfile.TemporaryDirectory() as td:
            proj = os.path.join(td, "p")
            s = gvp.generate(proj, c, W, SmolVLAFormats(js, c), lib="smolvla_test")
            self.assertEqual(list(s["nodes"]), ["vision", "image_0", "image_1", "prefix_1", "prefix_2", "expert"])
            glue = open(os.path.join(proj, "test", "vla_glue.h")).read()
            self.assertIn("#define VLA_CHUNK       10", glue)
            self.assertIn("add_library(smolvla_test SHARED", open(os.path.join(proj, "CMakeLists.txt")).read())
            exe = vla_host_emu.build(proj, td)
            res, acts = vla_host_emu.run_bench(exe, td, samples, (cfg.chunk, cfg.max_action))
            # the same library through ctypes (demo/vla/vla_lib.py): the second sample, its resize
            lib = vla_host_emu.build(proj, td, shared=True)
            sys.path.insert(0, os.path.join(_REPO, "demo", "vla"))
            from vla_lib import SmolVLA
            vla = SmolVLA(lib)
            frames, ids, st, nz = samples[1]
            for cam, f in enumerate(frames):
                vla.set_image(cam, f)
            got = vla.act(2, ids, st, noise=nz)
            np.testing.assert_array_equal(vla.resize(frames[1]), sq.pixels(frames[1], cfg, 0))
            self.assertEqual((vla.chunk, vla.action_size, vla.cameras), (cfg.chunk, cfg.max_action, 2))
            vla.close()
        self.assertTrue(res["reopen"]["identical"])
        np.testing.assert_array_equal(got, acts[1])
        for k, (frames, ids, st, nz) in enumerate(samples):
            self.assertTrue(res["samples"][k]["reps_identical"])
            np.testing.assert_array_equal(acts[k].astype(np.float64), spec.sample(frames, ids, st, nz),
                                          err_msg=f"sample {k}")


if __name__ == "__main__":
    unittest.main()
