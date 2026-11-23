# Copyright 2026 Ivan Gubochkin
# SPDX-License-Identifier: Apache-2.0

"""The 8-bit configuration in the LLM study (demo/chat/scripts/llm_study.py, policy
modifier ``i8``; doc/plans/INT8_PLAN.md §6) on the tiny random Llama of
gen_llama_models.py: a linear and an attention head equal the integer specification
(src/int8_spec.py) evaluated independently from the weights and the formats, the
8-bit calibration's formats cover every layer within their candidate range, and the
logits stay close to the int16 policy's."""

import unittest

import numpy as np

import gen_llama_models as G
from src import int8_spec as s8

study = G.study
POLICY16 = "pow2+sink+p12"
POLICY8 = POLICY16 + "+i8"


def build():
    cfg = G.TINY
    W = G.tiny_weights(cfg, 3)
    sc = G.study_cfg(cfg)
    rng = np.random.default_rng(1)
    calib = [np.concatenate([[1], rng.integers(2, sc.V, 23)]) for _ in range(4)]
    fm = study.Model(W, sc, None)
    cm = study.Model(W, sc, None, base=fm, sink=True, sink_model=fm, record_ch=True)
    for c in calib:
        study.teacher_forced(cm, c)
    ch = cm.stats.ch
    i8m = study.Model(W, sc, None, base=fm, sink=True, sink_model=fm)
    i8m.i8cal = study.I8Cal(ch, sc)
    for c in calib:
        study.teacher_forced(i8m, c)
    i8fmt = i8m.i8cal.formats()
    models = {}
    for name in (POLICY16, POLICY8):
        pol = study.policy_spec(name)
        fmt = study.make_formats(pol, ch, W, sc)
        if pol.get("i8"):
            fmt.update(i8fmt)
        models[name] = study.Model(W, sc, pol, fmt, sink_model=fm)
    return sc, W, fm, models, i8m.i8cal, i8fmt


class TestI8Policy(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sc, cls.W, cls.fm, cls.models, cls.cal, cls.i8fmt = build()
        cls.m8 = cls.models[POLICY8]

    def test_formats(self):
        sc = self.sc
        for l in range(sc.L):
            for cls_, n in (("q", sc.H), ("k", sc.KV), ("vc", sc.KV), ("pv", 1)):
                f = np.asarray(self.i8fmt[(cls_ + "8", l)])
                f0 = self.cal.f8max[(cls_, l)]
                self.assertEqual(f.shape, (n,))
                self.assertTrue(np.all((f >= f0) & (f < f0 + study.I8_CAND)), (cls_, l, f, f0))

    def test_linear_equals_spec(self):
        m8, sc = self.m8, self.sc
        rng = np.random.default_rng(5)
        l, key = 1, "g"
        name, ci, co = study.LINEARS[key]
        Wt = self.W[f"model.layers.{l}.{name}.weight"].astype(np.float64).T          # [K, N]
        wq, fw = s8.encode_weights(Wt, axis=1)
        np.testing.assert_array_equal(m8.w[(l, key)][0], wq)
        e16 = m8.Eh(ci, l, Wt.shape[0])
        x = np.rint(rng.normal(size=(5, Wt.shape[0])) * 0.3 * 2.0 ** e16) / 2.0 ** e16
        y = m8.lin(x, l, key)
        f8 = int(e16.min()) - 8
        xq, xd = s8.block_round(np.rint(x * 2.0 ** e16).astype(np.int64), e16 - f8)
        acc, g = s8.matmul(xq, f8, xd, wq, fw)
        eo = m8.Eh(co, l, Wt.shape[1])
        np.testing.assert_array_equal(y, s8.requant(acc, g, eo, -32768, 32767) / 2.0 ** eo)

    def test_attention_head_equals_spec(self):
        m8, sc = self.m8, self.sc
        rng = np.random.default_rng(6)
        l, h = 0, 1
        g = int(m8.grp[h])
        T, S, HD = 3, 37, sc.HD
        q = rng.normal(size=(T, HD))
        K = rng.normal(size=(S, HD)) * 0.5
        V = rng.normal(size=(S, HD)) * 0.5
        mask = np.arange(S)[None, :] <= (S - T + np.arange(T))[:, None]
        fq, fk, fs = m8.Eh("q", l, sc.H), m8.Eh("k", l, sc.KV), m8.Eh("s", l, sc.H)
        fv = m8.Ev("vc", l, sc.KV * HD)[g * HD:(g + 1) * HD]
        fpv = m8.Ev("pv", l, sc.H * HD)[h * HD:(h + 1) * HD]
        y = m8.attn8_head(q, K, V, mask, fq[h], fk[g], fs[h], fv, fpv, l, True, h, g)
        # independently: the calibrated static exponents, P from the host softmax at 2^-14
        fq8, fk8 = int(self.i8fmt[("q8", l)][h]), int(self.i8fmt[("k8", l)][g])
        fv8, fo = int(self.i8fmt[("vc8", l)][g]), int(self.i8fmt[("pv8", l)][0])
        q8 = np.clip(np.rint(q * 2.0 ** fq8), -128, 127).astype(np.int64)
        k8 = np.clip(np.rint(K * 2.0 ** fk8), -127, 127).astype(np.int64)
        acc, gg = s8.matmul(q8, fq8, 0, k8.T, np.full(S, fk8))
        s = s8.requant(acc, gg, int(fs[h]), -32768, 32767) / 2.0 ** fs[h]
        p = m8.softmax(s, mask, fs[h], m8.P8_SRC, l, unit=True)
        pad = (-S) % s8.BLOCK
        P8, Pd = s8.block_round(np.rint(np.pad(p, ((0, 0), (0, pad))) * 2.0 ** m8.P8_SRC).astype(np.int64),
                                m8.P8_SRC - m8.P8_BASE)
        v8 = np.pad(np.clip(np.rint(V * 2.0 ** fv8), -127, 127).astype(np.int64), ((0, pad), (0, 0)))
        acc, gg = s8.matmul(P8, m8.P8_BASE, Pd, v8, np.full(HD, fv8))
        np.testing.assert_array_equal(y, s8.requant(acc, gg, fo) / 2.0 ** fo)
        # and the head is close to the exact attention
        x = np.where(mask, (q @ K.T) / np.sqrt(HD), -np.inf)
        e = np.exp(x - x.max(-1, keepdims=True))
        ref = (e / e.sum(-1, keepdims=True)) @ V
        self.assertLess(np.abs(y - ref).max(), 0.1 * np.abs(ref).max())

    def test_logits_close_to_int16(self):
        sc = self.sc
        rng = np.random.default_rng(7)
        seq = np.concatenate([[1], rng.integers(2, sc.V, 20)])
        a, b = (study.teacher_forced(self.models[n], seq) for n in (POLICY16, POLICY8))
        self.assertEqual(a.shape, b.shape)
        self.assertLess(np.abs(a - b).max(), 0.2 * np.abs(a).max())


if __name__ == "__main__":
    unittest.main()
