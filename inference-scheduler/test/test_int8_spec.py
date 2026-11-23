# Copyright 2026 Ivan Gubochkin
# SPDX-License-Identifier: Apache-2.0

"""src/int8_spec.py — the 8-bit configuration's integer specification
(doc/plans/INT8_PLAN.md §6) against exact arithmetic: the rounding helpers, the
weight encoding, the producers' block rule (maximal shift, single rounding),
the blocked MatMul equal to the exact dyadic product, the requantization, the
packed DSP pair over whole 16-term chains, the unsigned offset."""

import unittest
from fractions import Fraction

import numpy as np

from src import int8_spec as s8


class TestHelpers(unittest.TestCase):

    def test_rne_shr(self):
        v = np.array([5, 6, 7, -5, -6, -7, 4, -4, 12, 20], dtype=np.int64)
        self.assertEqual(list(s8.rne_shr(v, 2)), [1, 2, 2, -1, -2, -2, 1, -1, 3, 5])
        self.assertEqual(int(s8.rne_shr(np.int64(3), 0)), 3)
        self.assertEqual(int(s8.rne_shr(np.int64(3), -2)), 12)

    def test_rne_shr_per_element_shifts(self):
        rng = np.random.default_rng(1)
        v = rng.integers(-(2 ** 40), 2 ** 40, size=5000, dtype=np.int64)
        k = rng.integers(-3, 30, size=5000, dtype=np.int64)
        got = s8.rne_shr(v, k)
        ref = [round(Fraction(int(a), 2 ** int(b))) if b > 0 else int(a) << -int(b) for a, b in zip(v, k, strict=True)]
        np.testing.assert_array_equal(got, np.array(ref, dtype=np.int64))

    def test_wrap(self):
        self.assertEqual(int(s8.wrap(2 ** 39, 40)), -(2 ** 39))
        self.assertEqual(int(s8.wrap(2 ** 39 - 1, 40)), 2 ** 39 - 1)
        self.assertEqual(int(s8.wrap(-(2 ** 39) - 1, 40)), 2 ** 39 - 1)
        self.assertEqual(int(s8.wrap(-5, 40)), -5)
        big = np.array([2 ** 50, -(2 ** 50)], np.int64)
        np.testing.assert_array_equal(s8.wrap(big, 63), big)          # no wrap: the studies' peak recording
        np.testing.assert_array_equal(s8.wrap(big, 64), big)


class TestWeights(unittest.TestCase):

    def test_range_and_error(self):
        rng = np.random.default_rng(2)
        w = rng.standard_t(4, size=(64, 96)) * 0.05          # heavy tails, as trained weights
        w[7] = 0.0
        q, f = s8.encode_weights(w, axis=0)
        self.assertEqual(q.shape, w.shape)
        self.assertTrue(np.all((q >= -127) & (q <= 127)))
        self.assertEqual(int(f[7]), 0)
        self.assertTrue(np.all(q[7] == 0))
        for n in range(64):
            if n == 7:
                continue
            a = np.abs(w[n]).max()
            f0 = max(ff for ff in range(-10, 40) if round(a * 2.0 ** ff) <= 127)
            self.assertIn(int(f[n]), (f0, f0 + 1, f0 + 2))
            errs = {ff: np.sum((np.clip(np.round(w[n] * 2.0 ** ff), -127, 127) * 2.0 ** -ff - w[n]) ** 2)
                    for ff in (f0, f0 + 1, f0 + 2)}
            self.assertAlmostEqual(errs[int(f[n])], min(errs.values()), places=15)
            np.testing.assert_array_equal(q[n], np.clip(np.round(w[n] * 2.0 ** f[n]), -127, 127))

    def test_axis(self):
        rng = np.random.default_rng(3)
        w = rng.normal(size=(40, 24))
        q0, f0 = s8.encode_weights(w, axis=0)
        q1, f1 = s8.encode_weights(w.T, axis=1)
        np.testing.assert_array_equal(q0, q1.T)
        np.testing.assert_array_equal(f0, f1)
        wc = rng.normal(size=(8, 4, 3, 3))                    # a Conv weight [O, I, kh, kw]
        qc, fc = s8.encode_weights(wc, axis=0)
        self.assertEqual(qc.shape, wc.shape)
        self.assertEqual(fc.shape, (8,))


class TestBlockRule(unittest.TestCase):

    def _check(self, A, k, q, d, block, dmax):
        A = np.asarray(A, np.int64)
        kk = np.broadcast_to(np.asarray(k, np.int64), A.shape)
        nb = A.shape[-1] // block
        for idx in np.ndindex(A.shape[:-1] + (nb,)):
            *row, b = idx
            sl = tuple(row) + (slice(b * block, (b + 1) * block),)
            db = int(d[idx])
            self.assertTrue(0 <= db <= dmax)
            r = s8.rne_shr(A[sl], kk[sl] - db)
            np.testing.assert_array_equal(q[sl], np.clip(r, -128, 127))
            if db > 0:                                       # fits at d
                self.assertTrue(r.min() >= -128 and r.max() <= 127)
            if db < dmax:                                    # and not at d + 1
                r1 = s8.rne_shr(A[sl], kk[sl] - db - 1)
                self.assertFalse(r1.min() >= -128 and r1.max() <= 127)

    def test_maximal_shift(self):
        rng = np.random.default_rng(4)
        scale = np.exp2(rng.integers(0, 14, size=(6, 8, 1)))       # blocks of very different sizes
        A = (rng.normal(size=(6, 8, 32)) * scale).astype(np.int64).reshape(6, 256)
        A[0, :32] = 0                                                # an all-zero block
        A[1, 32:64] = 0
        A[1, 40] = 2 ** 20                                           # one outlier: saturates at d = 0
        q, d = s8.block_round(A, 8)
        self.assertEqual(d.shape, (6, 8))
        self._check(A, 8, q, d, 32, 7)
        self.assertEqual(int(d[0, 0]), 7)
        self.assertEqual(int(d[1, 1]), 0)
        self.assertEqual(int(q[1, 40]), 127)

    def test_rounding_edges(self):
        # rne(127.5) = 128 overflows, rne(-128.5) = -128 fits: the rule follows the rounded values
        A = np.zeros(32, np.int64)
        A[0] = 255                        # 127.5 at k = 1 rounds to 128: no shift fits, d = 0 saturates
        q, d = s8.block_round(A, 1, dmax=1)
        self.assertEqual(int(d[0]), 0)
        self.assertEqual(int(q[0]), 127)
        A[0] = -257                       # -128.5 -> -128 (even): fits at k = 1
        q, d = s8.block_round(A, 2, dmax=1)
        self.assertEqual(int(d[0]), 1)
        self.assertEqual(int(q[0]), -128)

    def test_per_channel_shifts(self):
        rng = np.random.default_rng(5)
        A = rng.integers(-(2 ** 22), 2 ** 22, size=(3, 64), dtype=np.int64)
        k = rng.integers(10, 18, size=64)
        q, d = s8.block_round(A, k)
        self._check(A, k, q, d, 32, 7)

    def test_static(self):
        rng = np.random.default_rng(6)
        A = rng.integers(-5000, 5000, size=(4, 64), dtype=np.int64)
        q, d = s8.block_round(A, 5, dmax=0)
        self.assertTrue(np.all(d == 0))
        np.testing.assert_array_equal(q, np.clip(s8.rne_shr(A, 5), -128, 127))

    def test_float_rule_matches_integer_rule(self):
        rng = np.random.default_rng(7)
        A = (rng.normal(size=(5, 96)) * np.exp2(rng.integers(4, 16, size=(5, 1)))).astype(np.int64)
        g, f = 20, 6
        qi, di = s8.block_round(A, g - f)
        qf, df = s8.block_round_float(A * 2.0 ** -g, f)
        np.testing.assert_array_equal(qi, qf)
        np.testing.assert_array_equal(di, df)

    def test_rule_is_idempotent_on_values(self):
        # a blocked tensor's values, rounded again by the rule, are the same values (the shift may
        # differ only where -128 * 2^-(f+d+1) = -64 * 2^-(f+d)): a consumer may re-derive blocks
        rng = np.random.default_rng(13)
        A = (rng.normal(size=(40, 128)) * np.exp2(rng.integers(2, 18, size=(40, 4)).repeat(32, 1))).astype(np.int64)
        A[0, :32] = -257 * 2 ** 6                          # the -128.5 edge
        q, d = s8.block_round(A, 9)
        v = s8.block_value(q, 0, d)                      # A at 2^-9, base 0
        q2, d2 = s8.block_round_float(v, 0)
        np.testing.assert_array_equal(s8.block_value(q2, 0, d2), v)

    def test_base_exponent(self):
        for amax in (0.3, 1.0, 1.9921875, 127.0, 127.6, 1e-4):
            f = s8.base_exponent(amax)
            self.assertLessEqual(abs(round(amax * 2.0 ** f)), 127)
            self.assertGreater(abs(round(amax * 2.0 ** (f + 1))), 127)
        self.assertEqual(s8.base_exponent(1.0, margin=1), s8.base_exponent(1.0) - 1)


class TestMatmul(unittest.TestCase):

    def test_exact_dyadic_product(self):
        rng = np.random.default_rng(8)
        x = rng.normal(size=(5, 128)) * np.exp2(rng.integers(-6, 3, size=(5, 128 // 32)).repeat(32, 1))
        w = rng.normal(size=(128, 24)) * 0.05
        f_x = s8.base_exponent(np.abs(x).max())
        xq, xd = s8.block_round_float(x, f_x)
        wq, f_w = s8.encode_weights(w, axis=1)
        bias = rng.normal(size=24)
        g = f_x + s8.DELTA_MAX + f_w
        acc, g2 = s8.matmul(xq, f_x, xd, wq, f_w, bias_acc=s8.bias_raw(bias, g))
        np.testing.assert_array_equal(g, g2)
        xv = s8.block_value(xq, f_x, xd)
        wv = wq * np.exp2(-f_w)
        ref = xv @ wv + s8.bias_raw(bias, g) * np.exp2(-g)          # exact in double (< 2^53)
        np.testing.assert_array_equal(acc * np.exp2(-g), ref)
        # the blocked product is close to float: dominated by the activations' rounding
        rel = np.abs(xv @ wv - x @ w).max() / np.abs(x @ w).max()
        self.assertLess(rel, 0.03)

    def test_static_input_any_k(self):
        rng = np.random.default_rng(12)
        xq = rng.integers(-128, 128, size=(3, 16))           # K = 16 < BLOCK: a head of 16
        wq = rng.integers(-127, 128, size=(16, 5))
        acc, g = s8.matmul(xq, 3, 0, wq, np.arange(5))
        np.testing.assert_array_equal(acc, (xq @ wq) << s8.DELTA_MAX)
        np.testing.assert_array_equal(g, 3 + s8.DELTA_MAX + np.arange(5))
        with self.assertRaises(ValueError):                  # blocked inputs need whole blocks
            s8.matmul(xq, 3, np.zeros((3, 1), np.int64), wq, np.arange(5))

    def test_static_input_and_wrap(self):
        xq = np.full((1, 64), -128, np.int64)
        wq = np.full((64, 2), -127, np.int64)
        acc, g = s8.matmul(xq, 0, 0, wq, [0, 1])
        self.assertEqual(int(acc[0, 0]), (64 * 128 * 127) << s8.DELTA_MAX)
        np.testing.assert_array_equal(g, [7, 8])
        acc, _ = s8.matmul(xq, 0, 0, wq, [0, 0], acc_bits=20)
        self.assertEqual(int(acc[0, 0]), int(s8.wrap((64 * 128 * 127) << 7, 20)))

    def test_requant(self):
        rng = np.random.default_rng(9)
        acc = rng.integers(-(2 ** 30), 2 ** 30, size=(4, 64), dtype=np.int64)
        g = rng.integers(18, 24, size=64)
        q = s8.requant(acc, g, 2)
        exact = np.array([[round(Fraction(int(a), 2 ** int(gg - 2))) for a, gg in zip(row, g, strict=True)] for row in acc])
        np.testing.assert_array_equal(q, np.clip(exact, -128, 127))
        q16 = s8.requant(acc, g, 8, lo=-32768, hi=32767)
        self.assertTrue(np.all(np.abs(q16) <= 32768))
        qb, d = s8.requant_blocked(acc, g, 2)
        qb2, d2 = s8.block_round(acc, g - 2)
        np.testing.assert_array_equal(qb, qb2)
        np.testing.assert_array_equal(d, d2)


class TestW8A16(unittest.TestCase):

    def test_exact(self):
        rng = np.random.default_rng(14)
        x = rng.integers(-32768, 32768, size=(3, 576))
        wq = rng.integers(-127, 128, size=(576, 40))
        f_w = rng.integers(4, 9, size=40)
        acc, g = s8.matmul_w8a16(x, 9, wq, f_w)
        ref = np.array([[sum(int(a) * int(b) for a, b in zip(row, col, strict=True)) for col in wq.T] for row in x])
        np.testing.assert_array_equal(acc, ref)
        np.testing.assert_array_equal(g, 9 + f_w)
        self.assertLess(np.abs(acc).max(), 2 ** (s8.ACC_BITS - 1))


class TestPackedPair(unittest.TestCase):

    def test_chain_bound(self):
        self.assertEqual(s8.chain_bound(), 16 * 127 * 128)
        self.assertLess(s8.chain_bound(), 2 ** (s8.FIELD - 1))
        self.assertLess(abs(int(s8.pack_pair(-127, -127))), 2 ** 26)      # 27-bit signed word
        self.assertLess(int(s8.pack_pair(127, 127)), 2 ** 26)
        with self.assertRaises(ValueError):
            s8.pack_pair(-128, 0)

    def test_chains_exact(self):
        rng = np.random.default_rng(10)
        for trial in range(2000):
            if trial < 4:                                    # the extremes
                x = np.full(16, (-128, 127, -128, 127)[trial], np.int64)
                w0 = np.full(16, (127, -127, -127, 127)[trial], np.int64)
                w1 = np.full(16, (-127, 127, -127, 127)[trial], np.int64)
            else:
                x = rng.integers(-128, 128, 16)
                w0 = rng.integers(-127, 128, 16)
                w1 = rng.integers(-127, 128, 16)
            P = int(np.sum(s8.pack_pair(w0, w1) * x))
            self.assertLess(abs(P), 2 ** 47)                 # inside the 48-bit P register
            S0, S1 = s8.unpack(P)
            self.assertEqual(int(S0), int(np.dot(w0, x)))
            self.assertEqual(int(S1), int(np.dot(w1, x)))

    def test_minus_128_weights_break_the_chain(self):
        # why the weights are symmetric: with -128 the low field overflows within 16 terms
        x = np.full(16, -128, np.int64)
        w1 = np.full(16, -128, np.int64)
        S1 = int(np.dot(w1, x))
        self.assertGreaterEqual(S1, 2 ** (s8.FIELD - 1))


class TestUnsignedOffset(unittest.TestCase):

    def test_offset_identity(self):
        rng = np.random.default_rng(11)
        u = rng.integers(0, 256, size=(3, 64))
        wq = rng.integers(-127, 128, size=(64, 5))
        f_w = np.zeros(5, np.int64)
        acc_u = (u @ wq) << s8.DELTA_MAX
        acc_s, _ = s8.matmul(s8.to_offset(u), 0, 0, wq, f_w, bias_acc=s8.offset_bias(wq, axis=0))
        np.testing.assert_array_equal(acc_s, acc_u)
        self.assertTrue(np.all((s8.to_offset(u) >= -128) & (s8.to_offset(u) <= 127)))


if __name__ == "__main__":
    unittest.main()
