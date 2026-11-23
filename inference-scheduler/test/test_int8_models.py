# Copyright 2026 Ivan Gubochkin
# SPDX-License-Identifier: Apache-2.0

"""The 8-bit configuration (doc/plans/INT8_PLAN.md §9.6–§9.11) in the vision, video, VLA and
speech studies, on their tests' tiny random models: the 8-bit calibration's formats cover every
attention layer; the first kernel of each model equals the integer specification — the patch
embedding on uint8 pixels (the -128 offset folded into the bias, src/int8_spec.py), Moonshine's
conv1 on 16-bit audio (the W8A16 mode); the whole model runs at 8 bits close to its float
reference."""

import unittest

import numpy as np

import test_moonshine as TM
import test_smolvla as TS
import test_vit as TV
import test_vjepa2 as TJ
vs = TV.vs
I8 = "+i8"          # the o input static: the tiny heads (16 channels) are no whole blocks


def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def check_static(test, fmt, layers, H, KV):
    """q8 per head, k8 / vc8 per KV head, pv8 per tensor and the block base pvb, each layer."""
    for l in layers:
        test.assertEqual(np.shape(fmt[("q8", l)]), (H,), l)
        for c in ("k8", "vc8"):
            test.assertEqual(np.shape(fmt[(c, l)]), (KV,), (c, l))
        test.assertEqual(np.shape(fmt[("pv8", l)]), (1,), l)
        test.assertIn(("pvb", l), fmt)


class TestVisionI8(unittest.TestCase):
    """SmolVLM's vision encoder (vlm_study.VisionModel)."""

    @classmethod
    def setUpClass(cls):
        W = TV.tiny_weights(0)
        cls.Wv = Wv = {k[len("model."):]: v for k, v in W.items()}
        cls.vc = vc = vs.VCfg(TV.TINY_CFG["vision_config"])
        cls.scale = scale = TV.TINY_CFG["scale_factor"]
        cls.KW = KW = vs.kernel_weights(Wv, vc)
        vf = vs.VisionModel(Wv, KW, vc, scale, record_ch=True)
        vf.conn_k = TV.CONN_K
        cal = TV.images(4, 100)
        for im in cal:
            vf.forward(im)
        base = vs.vision_policy(True, True, True, iembed=False)
        p16, p8 = vs.vpolicy_spec(base), vs.vpolicy_spec(base + I8)
        cls.i8fmt = vs.vision_i8cal(Wv, KW, vc, scale, vf.stats.ch, cal)
        fmt8 = {**vs.make_vformats(p8, vf.stats.ch, KW, vc, scale), **cls.i8fmt}
        cls.m8 = vs.VisionModel(Wv, KW, vc, scale, p8, fmt8)
        cls.m16 = vs.VisionModel(Wv, KW, vc, scale, p16, vs.make_vformats(p16, vf.stats.ch, KW, vc, scale))
        cls.ref = vs.VisionModel(Wv, KW, vc, scale)
        for m in (cls.m8, cls.m16, cls.ref):
            m.conn_k = TV.CONN_K
        cls.im = TV.images(2, 7)

    def test_formats(self):
        vc = self.vc
        check_static(self, self.i8fmt, range(vc.L), vc.H, vc.KV)
        for l in range(vc.L):
            for c in ("x", "x2", "a"):
                self.assertIsInstance(self.i8fmt[(c, l)], int)
        self.assertEqual(self.i8fmt[("xs", vc.L)], self.i8fmt[("xf", vc.L)])

    def test_patch_embedding_equals_spec(self):
        vc, m = self.vc, self.m8
        n, P = vc.side, vc.P
        x = self.im[0].astype(np.float64).reshape(n, P, n, P, 3).transpose(0, 2, 4, 1, 3).reshape(n * n, -1)
        wq, fw = m.w[(0, "pe")]
        fo = m.Eh("pe", 0, wq.shape[1])
        # the uint8 pixels times the int8 weights, exactly; rne at the output's exponents
        v = x @ (wq * np.exp2(-fw.astype(np.float64))[None, :])
        ref = np.clip(np.round(v * np.exp2(fo)[None, :]), -32768, 32767) * np.exp2(-fo)[None, :]
        np.testing.assert_array_equal(m.lin(x, 0, "pe"), ref)

    def test_close_to_float(self):
        for im in self.im:
            r = self.ref.forward(im)
            e8, e16 = rel(self.m8.forward(im), r), rel(self.m16.forward(im), r)
            self.assertLess(e8, 0.12, e16)


class TestVJepa2I8(unittest.TestCase):
    """V-JEPA 2 (vjepa2_q.VJEPA2): the encoder, the pooler's self layer, the cross layer."""

    @classmethod
    def setUpClass(cls):
        vq = TJ.vq
        cls.cfg, W, KW, cls.m16, _ = TJ.setup()
        cfg = cls.cfg
        rec = vq.VJEPA2(cfg, W, record_ch=True, KW=KW)
        clips = [TJ.clip(10 + k) for k in range(3)]
        for c in clips:
            rec.forward(c)
        cal = vq.VJEPA2(cfg, W, KW=KW)
        cal.i8cal = vq.i8_calibrator(cfg, rec.stats.ch)
        for c in clips:
            cal.forward(c)
        cls.i8fmt = cal.i8cal.formats()
        pol = vq.policy_spec(TJ.POLICY + I8)
        cls.m8 = vq.VJEPA2(cfg, W, pol, {**vq.make_formats(pol, rec.stats.ch, KW, cfg), **cls.i8fmt}, KW=KW)
        cls.ref = vq.VJEPA2(cfg, W, KW=KW)
        cls.clip = TJ.clip(3)

    def test_formats(self):
        cfg = self.cfg
        check_static(self, self.i8fmt, cfg.attn_layers(), cfg.H, cfg.KV)
        self.assertIn(("xc", cfg.lk), self.i8fmt)

    def test_close_to_float(self):
        r = self.ref.forward(self.clip)
        e8, e16 = rel(self.m8.forward(self.clip), r), rel(self.m16.forward(self.clip), r)
        self.assertLess(e8, 0.08, e16)


class TestSmolVLAI8(unittest.TestCase):
    """SmolVLA (smolvla_q.SmolVLA): the vision encoder, the VLM prefix, the action expert with
    its self layers over the VLM's caches, its cross layers and its width of 48 (not whole blocks)."""

    @classmethod
    def setUpClass(cls):
        sq = TS.sq
        cls.cfg, W, cls.m16, _ = TS.setup()
        cfg = cls.cfg
        rec = sq.SmolVLA(cfg, W, record_ch=True)
        args = [([TS.frame(10 + 2 * k), TS.frame(11 + 2 * k)], TS.task(20 + k), TS.state(30 + k),
                 TS.noise(40 + k, cfg)) for k in range(3)]
        for a in args:
            rec.sample(*a)
        ch = {p: getattr(rec, p).stats.ch for p in ("vision", "text", "expert")}
        cal = sq.SmolVLA(cfg, W)
        sq.i8_calibrators(cal, ch)
        for a in args:
            cal.sample(*a)
        cls.i8fmt = sq.i8_formats(cal)
        pol = sq.policy_spec(TS.POLICY + I8)
        fmt = sq.make_formats(pol, ch["vision"], ch["text"], ch["expert"], cfg, rec.KW, W)
        for p in fmt:
            fmt[p].update(cls.i8fmt[p])
        cls.m8 = sq.SmolVLA(cfg, W, pol, fmt)
        cls.ref = sq.SmolVLA(cfg, W)
        cls.args = ([TS.frame(3), TS.frame(4)], TS.task(5), TS.state(6), TS.noise(7, cfg))

    def test_formats(self):
        cfg, f = self.cfg, self.i8fmt
        check_static(self, f["text"], range(cfg.text.L - 1), cfg.text.H, cfg.text.KV)
        check_static(self, f["expert"], range(cfg.expert.L), cfg.expert.H, cfg.expert.KV)
        for l in range(cfg.expert.L):
            if cfg.self_layer(l):                   # the VLM's caches: its exponents
                for c in ("k8", "vc8"):
                    np.testing.assert_array_equal(f["expert"][(c, l)], f["text"][(c, l)])
        for c in ("xt", "at", "xf"):
            self.assertIn((c, cfg.expert.L), f["expert"])

    def test_close_to_float(self):
        r = self.ref.sample(*self.args)
        e8, e16 = rel(self.m8.sample(*self.args), r), rel(self.m16.sample(*self.args), r)
        self.assertLess(e8, 0.2, e16)


class TestMoonshineI8(unittest.TestCase):
    """Moonshine (moonshine_q.QM): the conv stem, the encoder, the decoder's self and cross
    attention, its width of 80 (not whole blocks)."""

    @classmethod
    def setUpClass(cls):
        mq = TM.mq
        W = TM.random_weights()
        cls.W64 = W64 = {k.removeprefix("model."): v.astype(np.float64) for k, v in W.items()}
        rec = mq.QM(TM.CFG, W64, record=True)
        cal = [(TM.audio(s, seed), [5, 9, 17, 33]) for s, seed in ((1.1, 11), (0.7, 12))]
        for a, f in cal:
            mq.transcribe(rec, TM.CFG, a, forced=f)
        mf = mq.QM(TM.CFG, W64)
        mf.i8cal = mq.i8_calibrator(mf, rec.ch)
        for a, f in cal:
            mq.transcribe(mf, TM.CFG, a, forced=f)
        cls.i8fmt = mq.i8_formats(mf, rec.ch, mf.i8cal)
        pol = dict(TM.POLICY, i8=True)
        fmt = mq.make_formats(mq.QM(TM.CFG, W64), rec.ch, pol, ms={k: sq / n for k, (sq, n) in rec.ms.items()})
        fmt.update(cls.i8fmt)
        cls.m8 = mq.QM(TM.CFG, W64, pol=pol, fmt=fmt)
        cls.m16 = TM.setup()[2]
        cls.ref = mq.QM(TM.CFG, W64)
        cls.pcm = TM.audio(0.9, 5)
        cls.mq = mq

    def test_formats(self):
        m = self.m8
        for part, L, H in (("e", m.L, m.H), ("d", m.Ld, m.Hd)):
            for l in range(L):
                for c in ("q8", "k8", "vc8") + (("cq8", "ck8", "cv8") if part == "d" else ()):
                    self.assertEqual(np.shape(self.i8fmt[(c, part, l)]), (H,), (c, part, l))
                self.assertIn(("pv8", part, l), self.i8fmt)
        self.assertIn(("xf", "e", m.L), self.i8fmt)

    def test_conv1_w8a16_equals_spec(self):
        m = self.m8
        x = np.random.default_rng(3).integers(-20000, 20000, (6, 128)) / 32768.0
        wq, fw = m.w[("s", 0, "c1")]
        fo = m.E("c1", ("s", 0), wq.shape[1])
        v = x @ (wq * np.exp2(-fw.astype(np.float64))[None, :])          # 16-bit samples, exact
        ref = np.clip(np.round(v * np.exp2(fo)[None, :]), -32768, 32767) * np.exp2(-fo)[None, :]
        np.testing.assert_array_equal(m.kmm(x, ("s", 0, "c1"), "pcm", "c1", ("s", 0)), ref)

    def test_close_to_float(self):
        r = self.ref.encode(self.pcm)
        e8, e16 = rel(self.m8.encode(self.pcm), r), rel(self.m16.encode(self.pcm), r)
        self.assertLess(e8, 0.25, e16)
        forced = [5, 9, 17, 33]
        _, a8 = self.mq.transcribe(self.m8, TM.CFG, self.pcm, forced=forced)
        _, af = self.mq.transcribe(self.ref, TM.CFG, self.pcm, forced=forced)
        self.assertGreaterEqual(sum(a == b for a, b in zip(a8, af, strict=True)), len(af) - 1)


if __name__ == "__main__":
    unittest.main()
