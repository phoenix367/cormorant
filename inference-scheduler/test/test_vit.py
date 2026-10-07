"""The SmolVLM vision encoder (doc/plans/CHAT_PLAN.md §22-§23) on a tiny random
SigLIP-style ViT (hidden 64, 2 layers, 4 heads of 16, FFN 128, 32 x 32 images
in 4 x 4 patches = 64 tokens, pixel shuffle x2 -> 16 image tokens of a
32-wide text model, the connector in two K chunks):

  * the frontend's vision entry (src/vit.py): structure, numerics;
  * the scheduler's simulation equals the study's emulation
    (demo/chat/scripts/vlm_study.py VisionModel) bit for bit: policy
    pow2+p12+vgelu (the GELU on VectorOPKernel's activation unit,
    doc/plans/OFFLOAD_PLAN.md §2.1), pow2+p12 (host GELU) without the unit;
  * the generated C (-Werror) against the software kernels reproduces the
    simulation (cacheable / staged buffers, 1 / 4 host threads, separate CPU /
    DDR copies), and so does a multi-entry project of the vision entry and a
    tiny Llama whose prefill reads the image rows;
  * the text model's image rows (LlmEmbed with the image state, ids V + k)
    against vlm_study.TextModel; the GELU tables against the study's.
"""

import tempfile
import unittest
from unittest import mock

import numpy as np

import gen_llama_models as G
import host_emu
import vlm_study as vs
from src import _vectorop_hw_config
from src.codegen import CodeGenerator
from src.codegen.multi import MultiEntryGenerator
from src.graph import OnnxGraph
from src.llm_nodes import LlmAttnConvNode, LlmEmbedNode
from src.nodes import MatmulConvNode, MatmulNode
from src.vit import VisionFormats, VitConfig, VitFrontend, patches
from src.vit_nodes import VitGeluNode, VitGeluVopNode, VitSumDequantNode, gelu_table

_host_gelu = mock.patch.object(_vectorop_hw_config, "VECTOROP_ACTIVATIONS", False)

TINY_CFG = {"scale_factor": 2,
            "vision_config": dict(hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                                  intermediate_size=128, layer_norm_eps=1e-6, patch_size=4,
                                  image_size=32),
            "text_config": dict(hidden_size=32)}
CONN_K = 128


def bf16(x):
    return G.bf16(np.asarray(x, np.float32))


def tiny_weights(seed=0):
    """HF-named vision + connector weights, bf16-representable."""
    c = TINY_CFG["vision_config"]
    D, FF, P = c["hidden_size"], c["intermediate_size"], c["patch_size"]
    N = (c["image_size"] // P) ** 2
    Dt = TINY_CFG["text_config"]["hidden_size"]
    K = D * TINY_CFG["scale_factor"] ** 2
    r = np.random.default_rng(seed)

    def n(*s, sc=0.08):
        return bf16(r.normal(0, sc, s))
    W = {"model.vision_model.embeddings.patch_embedding.weight": n(D, 3, P, P, sc=0.05),
         "model.vision_model.embeddings.patch_embedding.bias": n(D, sc=0.02),
         "model.vision_model.embeddings.position_embedding.weight": n(N, D, sc=0.2),
         "model.vision_model.post_layernorm.weight": bf16(1 + r.normal(0, 0.1, D)),
         "model.vision_model.post_layernorm.bias": n(D, sc=0.05),
         "model.connector.modality_projection.proj.weight": n(Dt, K, sc=0.05)}
    for l in range(c["num_hidden_layers"]):
        p = f"model.vision_model.encoder.layers.{l}."
        for m in ("q_proj", "k_proj", "v_proj", "out_proj"):
            W[p + f"self_attn.{m}.weight"] = n(D, D, sc=0.15)
            W[p + f"self_attn.{m}.bias"] = n(D, sc=0.05)
        W[p + "mlp.fc1.weight"], W[p + "mlp.fc1.bias"] = n(FF, D, sc=0.12), n(FF, sc=0.1)
        W[p + "mlp.fc2.weight"], W[p + "mlp.fc2.bias"] = n(D, FF, sc=0.1), n(D, sc=0.05)
        for ln in ("layer_norm1", "layer_norm2"):
            W[p + f"{ln}.weight"] = bf16(1 + r.normal(0, 0.1, D))
            W[p + f"{ln}.bias"] = n(D, sc=0.05)
    return W


def images(n, seed):
    s = TINY_CFG["vision_config"]["image_size"]
    return [np.random.default_rng(seed + i).integers(0, 256, (s, s, 3)).astype(np.uint8)
            for i in range(n)]


class Tiny:
    """(config, weights, study model, frontend) of the tiny ViT; the study model
    computes the scheduler's policy (vlm_study.vision_policy), ``host_study()``
    the host-GELU one."""
    _cache = None
    _host = None

    @classmethod
    def get(cls):
        if cls._cache is None:
            W = tiny_weights(0)
            Wv = {k[len("model."):]: v for k, v in W.items()}
            vc = vs.VCfg(TINY_CFG["vision_config"])
            scale = TINY_CFG["scale_factor"]
            KW = vs.kernel_weights(Wv, vc)
            vf = vs.VisionModel(Wv, KW, vc, scale, record_ch=True)
            vf.conn_k = CONN_K
            for im in images(4, 100):
                vf.forward(im)
            pol = vs.VPOLICIES[vs.VISION_SHIPPED]
            fmt = vs.make_vformats(pol, vf.stats.ch, KW, vc, scale)
            vm, cls._host = (vs.VisionModel(Wv, KW, vc, scale, vs.VPOLICIES[p], fmt)
                             for p in (vs.vision_policy(True), vs.VISION_SHIPPED))
            vm.conn_k = cls._host.conn_k = CONN_K
            cfg = VitConfig.from_dict(TINY_CFG)
            fd = vs.formats_json(fmt, pol, vs.VISION_SHIPPED)
            fe = VitFrontend(cfg, W, VisionFormats(fd, cfg), name="vit_tiny", conn_k=CONN_K)
            cls._cache = (cfg, W, vm, fe)
        return cls._cache

    @classmethod
    def host_study(cls):
        cls.get()
        return cls._host


class TestGeluTables(unittest.TestCase):
    def test_equal_to_study(self):
        for f in (4, 9, 12):
            np.testing.assert_array_equal(gelu_table(f), vs.gelu_table(f))

    def test_values(self):
        t = gelu_table(8)
        for r in (-32768, -1000, -1, 0, 1, 300, 32767):
            x = r / 256.0
            self.assertAlmostEqual(t[r + 32768], vs.gelu_tanh(np.array([x]))[0], places=12)


class TestVisionEntry(unittest.TestCase):
    def setUp(self):
        self.cfg, self.W, self.vm, self.fe = Tiny.get()

    def test_structure(self):
        g = OnnxGraph(self.fe.entry(), fuse_act=True, s2d_stem=True)
        kinds = [type(sn).__name__ for sn in g.nodes]
        L, H = 2, 4
        self.assertEqual(kinds.count("VitLayerNormNode"), 2 * L + 1)
        self.assertEqual(kinds.count("VitAttnPrepNode"), L)
        self.assertEqual(kinds.count("VitAttnSoftmaxNode"), L * H)
        self.assertEqual(kinds.count("LlmAttnConvNode"), 2 * L * H)
        self.assertEqual(kinds.count("VitGeluVopNode"), L)
        self.assertEqual(kinds.count("VitPixelShuffleNode"), 2)
        self.assertEqual(kinds.count("VitSumDequantNode"), 1)
        mm = [sn for sn in g.nodes if isinstance(sn, (MatmulNode, MatmulConvNode))]
        self.assertEqual(len(mm), 1 + 6 * L + 2)
        attn = [sn for sn in g.nodes if isinstance(sn, LlmAttnConvNode)]
        self.assertTrue(all(sn.static and sn.C == 64 for sn in attn))
        out = [sn for sn in g.nodes if isinstance(sn, VitSumDequantNode)][0].output
        self.assertTrue(out.is_state and out.host == "f32")
        self.assertEqual(sum(g.weights_saturated.values()), 0)

    def test_simulation_equals_study(self):
        g = OnnxGraph(self.fe.entry(), fuse_act=True, s2d_stem=True)
        cg = CodeGenerator(g, model_path="vit_tiny.onnx")
        for im in images(3, 7):
            states = cg.initial_states()
            cg._forward_pass({"vision.patches": patches(im, self.cfg.P).astype(np.float64)},
                             states=states)
            np.testing.assert_array_equal(states["vlm.img"], self.vm.forward(im))

    def test_gelu_on_vectorop(self):
        """Two VectorOP calls per layer (ADD bias, MUL by 2^(8-f) + GELU_TANH), the
        rows' raw values those of vlm_study.vop_gelu_b; the generated C programs
        the activation unit."""
        g = OnnxGraph(self.fe.entry(), fuse_act=True, s2d_stem=True)
        for sn in (sn for sn in g.nodes if isinstance(sn, VitGeluVopNode)):
            f, ba, sc = sn.inputs
            l = int(sn.onnx_node.name.split(".")[1][1:])
            b = self.W[f"model.vision_model.encoder.layers.{l}.mlp.fc1.bias"]
            want_ba, want_sc = vs.vop_gelu_b(b.astype(np.float32), f.exp_channels(8))
            np.testing.assert_array_equal(sn._raw(ba), want_ba)
            np.testing.assert_array_equal(sn._raw(sc), want_sc)
            self.assertEqual([(c.fields["op"], c.fields["act"], c.fields["outer"], c.fields["b_inc"])
                              for c in sn.kernel_calls({})], [(0, 0, sn.rows, 0), (2, 6, sn.rows, 0)])
        src = CodeGenerator(g, model_path="vit_tiny.onnx").generate_source()
        self.assertIn("VECTOROP_ACT_GELU_TANH", src)
        self.assertIn("static void run_op_act(", src)

    @_host_gelu
    def test_host_gelu_without_the_unit(self):
        """No activation unit (AXI_VECTOROP_ACTIVATIONS=0): the host VitGelu, the
        policy pow2+p12, in simulation and in the generated C."""
        g = OnnxGraph(self.fe.entry(output=True), fuse_act=True, s2d_stem=True)
        kinds = [type(sn) for sn in g.nodes]
        self.assertEqual(kinds.count(VitGeluNode), 2)
        self.assertNotIn(VitGeluVopNode, kinds)
        cg = CodeGenerator(g, model_path="vit_tiny_host.onnx")
        im = images(1, 7)[0]
        out = cg._forward_pass({"vision.patches": patches(im, self.cfg.P).astype(np.float64)},
                               states=cg.initial_states())
        np.testing.assert_array_equal(out["vision.image"], Tiny.host_study().forward(im))
        self.assertFalse(np.array_equal(out["vision.image"], self.vm.forward(im)))
        with tempfile.TemporaryDirectory() as td:
            rc, log = host_emu.build_and_run(cg, td, cached=True, threads=3, min_elems=1)
            self.assertEqual(rc, 0, log[-3000:])
            self.assertIn("test_inference PASSED", log)

    def test_engines(self):
        for mode in ("off", "always"):
            g = OnnxGraph(self.fe.entry(output=True), fuse_act=True, s2d_stem=True,
                          matmul_on_conv=mode)
            cg = CodeGenerator(g, model_path="vit_tiny.onnx")
            im = images(1, 11)[0]
            out = cg._forward_pass({"vision.patches": patches(im, self.cfg.P).astype(np.float64)},
                                   states=cg.initial_states())
            np.testing.assert_array_equal(out["vision.image"], self.vm.forward(im), err_msg=mode)


class TestHostEmulation(unittest.TestCase):
    """Generated C (-Werror) against the software kernels == simulation."""

    def test_vision_entry(self):
        _cfg, _W, _vm, fe = Tiny.get()
        g = OnnxGraph(fe.entry(output=True), fuse_act=True, s2d_stem=True)
        cg = CodeGenerator(g, model_path="vit_tiny_vision.onnx")
        for cached, threads, inc in ((True, 4, False), (False, 1, False), (True, 3, True)):
            with self.subTest(cached=cached, threads=threads, incoherent=inc), \
                    tempfile.TemporaryDirectory() as td:
                rc, out = host_emu.build_and_run(cg, td, cached=cached, threads=threads,
                                                 min_elems=1, incoherent=inc)
                self.assertEqual(rc, 0, out[-3000:])
                self.assertIn("test_inference PASSED", out)


class TestAttentionSplit(unittest.TestCase):
    """attn_split: every head's softmax and P.V in query-row parts — the same
    bits as the unsplit entry, in simulation and in the generated C."""

    def test_split_equals_unsplit(self):
        cfg, W, vm, fe = Tiny.get()
        fe2 = VitFrontend(fe.cfg, fe.W, fe.fmt, name="vit_tiny", conn_k=CONN_K, attn_split=2)
        g = OnnxGraph(fe2.entry(output=True), fuse_act=True, s2d_stem=True)
        names = [sn.onnx_node.name for sn in g.nodes]
        self.assertIn("vision.l0.softmax0r1", names)
        self.assertIn("vision.l0.pv0r1", names)
        cg = CodeGenerator(g, model_path="vit_tiny_split.onnx")
        for im in images(2, 13):
            out = cg._forward_pass({"vision.patches": patches(im, cfg.P).astype(np.float64)},
                                   states=cg.initial_states())
            np.testing.assert_array_equal(out["vision.image"], vm.forward(im))
        with tempfile.TemporaryDirectory() as td:
            rc, out = host_emu.build_and_run(cg, td, cached=True, threads=3, min_elems=1)
            self.assertEqual(rc, 0, out[-3000:])
            self.assertIn("test_inference PASSED", out)

    def test_bad_split(self):
        _cfg, _W, _vm, fe = Tiny.get()
        with self.assertRaises(ValueError):
            VitFrontend(fe.cfg, fe.W, fe.fmt, attn_split=3)


class TestImageRows(unittest.TestCase):
    """The text model's prefill reads image rows: ids V .. V + R - 1."""
    R = 4

    def setUp(self):
        cfg = G.TINY
        self.cfg = cfg
        self.W = G.tiny_weights(cfg, 5)
        self.formats = G.calibrate(cfg, self.W, n_seq=2, length=20)
        from src.llama import Formats, LlamaConfig, LlamaFrontend
        lc = LlamaConfig.from_dict(cfg)
        self.fe = LlamaFrontend(lc, self.W, Formats(self.formats, lc), ctx=G.TINY_CTX,
                                name="llama_img", image_rows=self.R, image_state="vlm.img")

    def test_prefill_with_image_rows(self):
        g = OnnxGraph(self.fe.entry("prefill", 16, with_head=True), fuse_act=True, s2d_stem=True)
        emb = [sn for sn in g.nodes if isinstance(sn, LlmEmbedNode)][0]
        self.assertEqual(emb.image_rows, self.R)
        cg = CodeGenerator(g, model_path="llama_img_prefill.onnx")
        V = self.cfg["vocab_size"]
        rng = np.random.default_rng(3)
        img = bf16(rng.normal(0, 2.0, (self.R, self.cfg["hidden_size"])))
        # study: TextModel with image token 7 and the image rows
        sm, sc = G.study_model(self.cfg, self.W, self.formats, policy=G.MIX_POLICY)
        tm = vs.TextModel.__new__(vs.TextModel)
        tm.__dict__.update(sm.__dict__)
        tm.image_tok = 7
        prompt = [3, 9] + [7] * self.R + [11, 12, 13]
        seq = G.study.Seq(sc, 32)
        tm.img = img.astype(np.float64)
        want = tm.forward([seq], [np.array([1] + prompt, np.int64)])[0][-1]
        # scheduler: ids with the image rows V + k after the sink
        ids = np.zeros(16)
        sim = [3, 9] + [V + k for k in range(self.R)] + [11, 12, 13]
        ids[:len(sim)] = sim
        states = cg.initial_states()
        states["vlm.img"][...] = img
        out = cg._forward_pass({"prefill_16_logits.ids": ids, "prefill_16_logits.pos": [1],
                                "prefill_16_logits.n": [len(sim)]}, states=states)
        np.testing.assert_array_equal(out["prefill_16_logits.logits"].reshape(-1), want)
        with tempfile.TemporaryDirectory() as td:
            rc, log = host_emu.build_and_run(cg, td, threads=4, min_elems=1, incoherent=True)
            self.assertEqual(rc, 0, log[-3000:])
            self.assertIn("test_inference PASSED", log)

    def test_multi_entry_project(self):
        """vision + prefill (image rows) + head in one project: the image state
        is shared; the generated C reproduces the simulation."""
        _cfg, _W, _vm, vfe = Tiny.get()
        from src.llama import Formats, LlamaConfig, LlamaFrontend
        cfg = dict(G.TINY, hidden_size=32, num_attention_heads=2, num_key_value_heads=1,
                   head_dim=16, intermediate_size=64)
        W = G.tiny_weights(cfg, 6)
        fmts = G.calibrate(cfg, W, n_seq=2, length=20)
        lc = LlamaConfig.from_dict(cfg)
        tfe = LlamaFrontend(lc, W, Formats(fmts, lc), ctx=G.TINY_CTX, name="vlm_tiny",
                            image_rows=vfe.cfg.n_img, image_state=vfe.image_state)
        entries = [("vision", OnnxGraph(vfe.entry(), fuse_act=True, s2d_stem=True)),
                   ("prefill_16", OnnxGraph(tfe.entry("prefill", 16), fuse_act=True, s2d_stem=True)),
                   ("head", OnnxGraph(tfe.entry("head"), fuse_act=True, s2d_stem=True))]
        mg = MultiEntryGenerator(entries, "vlm_tiny")
        with tempfile.TemporaryDirectory() as td:
            rc, out = host_emu.build_and_run(mg, td, threads=4, min_elems=1, incoherent=True)
            self.assertEqual(rc, 0, out[-3000:])
            self.assertIn("PASSED", out)


if __name__ == "__main__":
    unittest.main()
