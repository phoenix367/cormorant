"""src/rowops_spec.py — the integer row-op specifications against the exact
functions: every output within 1 LSB of the exactly rounded value, the
reciprocal square root within 2^-24, the helpers exact."""

import math
import unittest

import numpy as np

from src import rowops_spec as rs


def exact_rne(v, f):
    return np.clip(np.rint(np.asarray(v, np.float64) * 2.0 ** f), -32768, 32767)


class TestHelpers(unittest.TestCase):

    def test_rne_shr(self):
        v = np.array([5, 6, 7, -5, -6, -7, 4, -4, 12, 20], dtype=np.int64)
        self.assertEqual(list(rs.rne_shr(v, 2)), [1, 2, 2, -1, -2, -2, 1, -1, 3, 5])
        self.assertEqual(int(rs.rne_shr(np.int64(3), 0)), 3)
        self.assertEqual(int(rs.rne_shr(np.int64(3), -2)), 12)

    def test_mul_shr_floor_exact(self):
        rng = np.random.default_rng(1)
        a = rng.integers(-(2 ** 43) + 1, 2 ** 43, size=20000, dtype=np.int64)
        for m, k in ((2 ** 32 - 1, 37), (3_000_000_001, 25), (2 ** 31 + 12345, 60), (77777, 21)):
            got = rs.mul_shr_floor(a, m, k)
            ref = np.array([(int(x) * m) >> k for x in a], dtype=np.int64)
            np.testing.assert_array_equal(got, ref)

    def test_rsqrt(self):
        rng = np.random.default_rng(2)
        worst = 0.0
        for _ in range(5000):
            e = int(rng.integers(1, 90))
            D = int(rng.integers(1, 2 ** 20)) << max(0, e - 20) | 1
            Rm, sh = rs.rsqrt_bf(D)
            rel = abs(Rm * 2.0 ** -sh * math.sqrt(D) - 1.0)
            worst = max(worst, rel)
        self.assertLess(worst, 2.0 ** -24, worst)


class TestNorms(unittest.TestCase):

    def test_rms_norm_within_1_lsb(self):
        rng = np.random.default_rng(3)
        n, rows, fr = 576, 64, 15
        gamma = rng.uniform(0.05, 4.0, n)
        eps = 1e-5
        # rows of very different scale, one with an outlier channel, one near-constant
        h = rng.standard_normal((rows, n)) * rng.choice([0.01, 1.0, 40.0], rows)[:, None]
        h[3, 7] = 25000.0
        h[4] = 0.5 + 1e-3 * rng.standard_normal(n)
        h_raw = np.rint(h * 2.0 ** fr).astype(np.int64)
        hq = h_raw / 2.0 ** fr
        f_out = rng.integers(8, 13, n)
        got = rs.rms_norm(h_raw, fr, gamma, eps, f_out)
        r = 1.0 / np.sqrt((hq * hq).sum(1) / n + eps)
        ref = exact_rne(hq * r[:, None] * gamma[None, :], f_out[None, :])
        diff = np.abs(got - ref)
        self.assertLessEqual(diff.max(), 1)
        self.assertLess((diff > 0).mean(), 0.01, (diff > 0).mean())

    def test_layer_norm_int16_within_1_lsb(self):
        rng = np.random.default_rng(4)
        n, rows = 768, 48
        gamma = rng.uniform(0.2, 2.0, n)
        beta = rng.uniform(-0.5, 0.5, n)
        x = rng.standard_normal((rows, n)) * rng.choice([0.05, 1.0, 8.0], rows)[:, None] + 3.0
        x_raw = np.clip(np.rint(x * 256.0), -32768, 32767).astype(np.int64)
        xq = x_raw / 256.0
        got = rs.layer_norm(x_raw, 8, gamma, beta, 1e-12, 8)
        mu = xq.mean(1, keepdims=True)
        var = ((xq - mu) ** 2).mean(1, keepdims=True)
        ref = exact_rne((xq - mu) / np.sqrt(var + 1e-12) * gamma + beta, 8)
        diff = np.abs(got - ref)
        self.assertLessEqual(diff.max(), 1)
        self.assertLess((diff > 0).mean(), 0.01)

    def test_layer_norm_int32_rows(self):
        rng = np.random.default_rng(5)
        n, rows, fr = 768, 32, 15
        gamma = rng.uniform(0.2, 2.0, n)
        beta = rng.uniform(-0.5, 0.5, n)
        h = rng.standard_normal((rows, n)) * rng.choice([0.3, 5.0, 300.0], rows)[:, None]
        h[2, 11] = 750.0                                  # the vision residual's outlier
        h_raw = np.rint(h * 2.0 ** fr).astype(np.int64)
        hq = h_raw / 2.0 ** fr
        got = rs.layer_norm(h_raw, fr, gamma, beta, 1e-6, 8)
        mu = hq.mean(1, keepdims=True)
        var = ((hq - mu) ** 2).mean(1, keepdims=True)
        ref = exact_rne((hq - mu) / np.sqrt(var + 1e-6) * gamma + beta, 8)
        diff = np.abs(got - ref)
        self.assertLessEqual(diff.max(), 1)
        self.assertLess((diff > 0).mean(), 0.01)


class TestRopeSilu(unittest.TestCase):

    def test_rope_within_1_lsb(self):
        rng = np.random.default_rng(6)
        T, nh, HD = 50, 3, 64
        ang = rng.uniform(-math.pi, math.pi, (T, HD // 2)).astype(np.float32)
        cos = np.concatenate([np.cos(ang), np.cos(ang)], 1).astype(np.float32).astype(np.float64)
        sin = np.concatenate([np.sin(ang), np.sin(ang)], 1).astype(np.float32).astype(np.float64)
        f_x = rng.choice([12, 13, 14], (nh, HD))          # per channel, as the pow2 formats
        f_out = np.array([7, 6, 7])
        x_raw = rng.integers(-32768, 32768, (T, nh, HD), dtype=np.int64)
        c_raw, s_raw = rs.rope_tables_raw(cos, sin)
        got = rs.rope(x_raw, f_x, c_raw, s_raw, f_out)
        x = x_raw / 2.0 ** f_x[None]
        half = HD // 2
        rot = np.concatenate([-x[..., half:], x[..., :half]], -1)
        ref = exact_rne(x * cos[:, None, :] + rot * sin[:, None, :], f_out[None, :, None])
        diff = np.abs(got - ref)
        self.assertLessEqual(diff.max(), 1)
        self.assertLess((diff > 0).mean(), 0.01)

    def test_silu_mul_within_1_lsb(self):
        rng = np.random.default_rng(7)
        rows, n = 40, 1536
        f_g = rng.choice([9, 11, 12, 13, 14], n)       # per column (9: |g| up to 64)
        f_u = rng.choice([12, 13, 14], n)
        f_out = np.full(n, 8)
        g_raw = rng.integers(-32768, 32768, (rows, n), dtype=np.int64)
        u_raw = rng.integers(-32768, 32768, (rows, n), dtype=np.int64)
        got = rs.silu_mul(g_raw, f_g, u_raw, f_u, f_out)
        g = g_raw / 2.0 ** f_g[None, :]
        u = u_raw / 2.0 ** f_u[None, :]
        ref = exact_rne(g / (1.0 + np.exp(-g)) * u, f_out[None, :])
        diff = np.abs(got - ref)
        self.assertLessEqual(diff.max(), 1)
        self.assertLess((diff > 0).mean(), 0.01)


if __name__ == "__main__":
    unittest.main()
