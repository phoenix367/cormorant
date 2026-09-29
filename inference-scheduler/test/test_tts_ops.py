"""The axi.tts C helpers (src/tts_nodes.py tts_c_helpers) op by op against
an independent numpy model of their contract (double arithmetic, int16
write-back with round-half-even + saturation, libm tanh / exp): every
branch of the fast paths — the integer sums with exact and rounding shifts
(ties included), the int32 / int64 three-input average, the double
fallbacks (non-power-of-two scales, wide exponent spreads, / 2, four
inputs), masks, the folded prep windows, LeakyReLU, float32 input, the gate
tables, the interleave."""

import math
import os
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

from src.tts_nodes import libm_map, tts_c_helpers

PRELUDE = r"""
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef uint16_t Data_t;
typedef void (*host_task_fn)(void *, unsigned, unsigned);
/* two workers' ranges, one after the other: row splits must not matter */
static void host_parallel(host_task_fn fn, void *arg, unsigned n, unsigned grain, unsigned align)
{ (void)grain; (void)align; if (n) { fn(arg, 0u, n / 2u); fn(arg, n / 2u, n); } }
static inline double llm_ld(Data_t b, double sc) { return (double)(int16_t)b * sc; }
static inline int16_t llm_st16(double v, double si)
{
    double r;
    if (v != v) return 0;
    r = nearbyint(v * si);
    if (r > 32767.0) r = 32767.0;
    if (r < -32768.0) r = -32768.0;
    return (int16_t)r;
}
static inline Data_t llm_st(double v, double si) { return (Data_t)llm_st16(v, si); }
"""


def st(v, si):
    return np.clip(np.round(np.asarray(v, np.float64) * si), -32768, 32767).astype(np.int16)


def arr(name, a, ctype):
    body = ",".join(str(int(v)) if ctype != "float" else repr(float(v)) + "f" for v in np.ravel(a))
    return f"static {ctype} {name}[] = {{{body}}};\n"


@unittest.skipUnless(shutil.which("cc"), "needs cc")
class TestTtsOps(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp(prefix="tts_ops_")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)

    def run_c(self, decls, call, n_out, name):
        src = os.path.join(self.dir, name + ".c")
        exe = os.path.join(self.dir, name)
        with open(src, "w") as f:
            f.write(PRELUDE + tts_c_helpers() + decls +
                    f"static Data_t out_[{max(n_out, 1)}];\nint main(void)\n{{\n    {call}\n"
                    f"    fwrite(out_, 2, {n_out}, stdout);\n    return 0;\n}}\n")
        r = subprocess.run(["cc", "-O2", "-ffp-contract=off", "-Wall", "-Wextra", "-Werror",
                            "-Wno-unused-function", src, "-lm", "-o", exe], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])
        out = subprocess.run([exe], capture_output=True, check=True).stdout
        return np.frombuffer(out, "<i2")

    # ── TtsSum ──

    def sum_case(self, name, xs, sx, div=1.0, si=2.0 ** 10, masked=None):
        k, (nch, L) = len(xs), xs[0].shape
        decls = "".join(arr(f"x{i}", x.view(np.uint16), "Data_t") for i, x in enumerate(xs))
        decls += f"static const Data_t *xs_[] = {{{', '.join(f'x{i}' for i in range(k))}}};\n"
        decls += f"static const double sx_[] = {{{', '.join(repr(float(s)) for s in sx)}}};\n"
        lo, hi, rate, off = masked or (0, 0, 1, 0)
        call = (f"tts_sum(xs_, sx_, {k}u, {nch}u, {L}u, {div!r}, {int(masked is not None)}, {lo}, {hi}, "
                f"{rate}u, {off}, {si!r}, out_);")
        got = self.run_c(decls, call, nch * L, name).reshape(nch, L)
        v = xs[0].astype(np.float64) * sx[0]
        for x, s in zip(xs[1:], sx[1:], strict=True):
            v = v + x.astype(np.float64) * s
        if div != 1.0:
            v = v / div
        if masked is not None:
            t = np.arange(L) + off * rate
            v = np.where((t >= lo * rate) & (t < hi * rate), v, 0.0)
        np.testing.assert_array_equal(got, st(v, si), err_msg=name)

    def x16(self, shape, seed, lim=32768):
        return np.random.default_rng(seed).integers(-lim, lim, shape).astype(np.int16)

    def test_sum_two_inputs(self):
        a, b = self.x16((5, 300), 1), self.x16((5, 300), 2)
        self.sum_case("s2_exact", [a, b], [2.0 ** -10, 2.0 ** -11], si=2.0 ** 12)          # m = 0
        self.sum_case("s2_round", [a, b], [2.0 ** -10, 2.0 ** -11], si=2.0 ** 10)          # m = 1: ties
        self.sum_case("s2_round3", [a, b], [2.0 ** -12, 2.0 ** -9], si=2.0 ** 8)           # m = 4
        self.sum_case("s2_masked", [a, b], [2.0 ** -9, 2.0 ** -9], si=2.0 ** 9, masked=(3, 9, 32, 2))
        self.sum_case("s2_masked_out", [a, b], [2.0 ** -9, 2.0 ** -9], si=2.0 ** 9, masked=(-4, 1, 64, 2))
        self.sum_case("s2_masked_all", [a, b], [2.0 ** -9, 2.0 ** -9], si=2.0 ** 9, masked=(-4, 90, 16, 0))
        self.sum_case("s2_not_pow2", [a, b], [0.3 * 2.0 ** -9, 2.0 ** -9], si=2.0 ** 9)   # double path
        self.sum_case("s2_wide", [a, b], [2.0 ** -50, 2.0 ** -2], si=2.0 ** 2)             # spread > 40
        self.sum_case("s2_div", [a, b], [2.0 ** -9, 2.0 ** -10], div=2.0, si=2.0 ** 10)

    def test_sum_average_of_three(self):
        a, b, c = self.x16((4, 257), 3), self.x16((4, 257), 4), self.x16((4, 257), 5)
        self.sum_case("s3_int32", [a, b, c], [2.0 ** -9, 2.0 ** -11, 2.0 ** -12], div=3.0, si=2.0 ** 12)
        self.sum_case("s3_int32_j", [a, b, c], [2.0 ** -9, 2.0 ** -11, 2.0 ** -12], div=3.0, si=2.0 ** 9)
        self.sum_case("s3_int64", [a, b, c], [2.0 ** -2, 2.0 ** -18, 2.0 ** -20], div=3.0, si=2.0 ** 14)
        self.sum_case("s3_fo_above", [a, b, c], [2.0 ** -9, 2.0 ** -10, 2.0 ** -9], div=3.0, si=2.0 ** 13)
        self.sum_case("s3_masked", [a, b, c], [2.0 ** -9, 2.0 ** -11, 2.0 ** -12], div=3.0, si=2.0 ** 11,
                      masked=(2, 7, 32, 1))
        self.sum_case("s3_div2", [a, b, c], [2.0 ** -9, 2.0 ** -11, 2.0 ** -12], div=2.0, si=2.0 ** 11)
        self.sum_case("s3_plain", [a, b, c], [2.0 ** -9, 2.0 ** -11, 2.0 ** -12], si=2.0 ** 11)
        small = [self.x16((4, 257), s, 64) for s in (6, 7, 8)]                          # many ties / 3
        self.sum_case("s3_small", small, [2.0 ** -1, 2.0 ** -1, 2.0 ** -1], div=3.0, si=2.0 ** 0)
        self.sum_case("s4", [a, b, c, a], [2.0 ** -9, 2.0 ** -11, 2.0 ** -12, 2.0 ** -8], si=2.0 ** 8)

    # ── TtsPrep ──

    def prep_case(self, name, C, L, ch0, nch, reverse, t0, rows, w0, hl, hr, wlo, whi, lo, hi, rate, off,
                  alpha, sx, si, f32=False):
        rng = np.random.default_rng(len(name))
        if f32:
            x = (rng.standard_normal((C, L)) * 3).astype(np.float32)
            decls = arr("xf", x, "float")
            src = "NULL, xf, 1.0"
            xv = x.astype(np.float64)
        else:
            x = self.x16((C, L), len(name))
            decls = arr("xi", x.view(np.uint16), "Data_t")
            src = f"xi, NULL, {sx!r}"
            xv = x.astype(np.float64) * sx
        W = w0 + hl + hr
        call = (f"tts_prep({src}, {C}u, {L}u, {ch0}u, {nch}u, {int(reverse)}, {t0}, {rows}u, {w0}u, {hl}u, "
                f"{hr}u, {wlo}, {whi}, {lo}, {hi}, {rate}u, {off}, {alpha!r}, {si!r}, out_);")
        got = self.run_c(decls, call, nch * rows * W, name).reshape(nch, rows, W)
        want = np.zeros((nch, rows, W))
        for c in range(nch):
            sc = ch0 + (nch - 1 - c if reverse else c)
            for r in range(rows):
                s = r * w0 + np.arange(W) - hl + t0
                ok = (s >= wlo) & (s < whi) & (s >= 0) & (s < L) & (s + off * rate >= lo * rate) & \
                    (s + off * rate < hi * rate)
                v = np.where(ok, xv[sc, np.clip(s, 0, L - 1)], 0.0)
                want[c, r] = np.where(v > 0, v, alpha * v) if alpha != 1.0 else v
        np.testing.assert_array_equal(got, st(want, si), err_msg=name)

    def test_prep(self):
        P = dict(C=6, L=512, ch0=1, nch=4, reverse=False, t0=0, rows=4, w0=128, hl=2, hr=2, wlo=0, whi=512,
                 lo=0, hi=64, rate=8, off=0, alpha=1.0, sx=2.0 ** -10, si=2.0 ** 11)
        self.prep_case("p_plain", **P)
        self.prep_case("p_leaky", **{**P, "alpha": 0.1, "si": 2.0 ** 9})
        self.prep_case("p_masked", **{**P, "alpha": 0.1, "lo": 5, "hi": 40, "off": 3})
        self.prep_case("p_window", **{**P, "t0": -3, "hl": 9, "hr": 9, "wlo": 100, "whi": 470})
        self.prep_case("p_rounding", **{**P, "si": 2.0 ** 7, "reverse": True})
        self.prep_case("p_f32", **{**P, "C": 8, "L": 64, "ch0": 4, "nch": 4, "reverse": True, "rows": 1,
                                   "w0": 64, "hl": 0, "hr": 0, "whi": 64, "lo": 3, "hi": 50, "rate": 1,
                                   "sx": 1.0, "si": 2.0 ** 7}, f32=True)
        self.prep_case("p_f32_leaky", **{**P, "C": 3, "L": 64, "ch0": 0, "nch": 3, "rows": 2, "w0": 32,
                                         "whi": 64, "rate": 1, "alpha": 0.2, "sx": 1.0}, f32=True)

    # ── TtsGate / TtsInterleave ──

    def test_gate(self):
        n, L = 3, 200
        x = self.x16((2 * n, L), 9)
        for i, (sx, si) in enumerate(((2.0 ** -10, 2.0 ** 9), (2.0 ** -9, 2.0 ** 12), (2.0 ** -10, 2.0 ** 8))):
            decls = arr("x", x.view(np.uint16), "Data_t")
            call = (f"tts_gate(x, {sx!r}, {n}u, {L}u, {si!r}, out_); "      # twice: the table is reused
                    f"tts_gate(x, {sx!r}, {n}u, {L}u, {si!r}, out_);")
            got = self.run_c(decls, call, n * L, f"gate{i}").reshape(n, L)
            xv = x.astype(np.float64) * sx
            g = libm_map(math.tanh, xv[:n]) * libm_map(lambda v: 1.0 / (1.0 + math.exp(-v)), xv[n:])
            np.testing.assert_array_equal(got, st(g, si))

    def test_interleave(self):
        for s, O, L in ((4, 3, 50), (8, 5, 17)):
            x = self.x16((s * O, L), s)
            got = self.run_c(arr("x", x.view(np.uint16), "Data_t"), f"tts_interleave(x, {s}u, {O}u, {L}u, out_);",
                             s * O * L, f"il{s}").reshape(O, L * s)
            want = x.reshape(s, O, L).transpose(1, 2, 0).reshape(O, L * s)
            np.testing.assert_array_equal(got, want)


if __name__ == "__main__":
    unittest.main()
