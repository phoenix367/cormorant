"""sampler.py / src/sampler.c: greedy = argmax, penalties, top-k and top-p
masks against independent references, the sampled distribution with fixed
seeds, determinism, and the C library == the Python fallback token for
token.  The C half is skipped when there is no C compiler."""

import array
import ctypes
import math
import random
import unittest

from _util import sampler_lib

from sampler import CSampler, PySampler, SamplerParams, make_sampler

V = 49152


def rand_logits(rng, n=V, scale=3.0):
    return (ctypes.c_float * n)(*[rng.gauss(0.0, scale) for _ in range(n)])


def softmax(vals):
    m = max(vals)
    e = [math.exp(v - m) for v in vals]
    s = sum(e)
    return [x / s for x in e]


class SamplerCases:
    """Mixed into one TestCase per implementation (self.make(n) -> sampler)."""

    def test_rng_vector(self):
        s = self.make(8)
        s.seed(0)
        self.assertEqual(s.rng_next(), 0xE220A8397B1DCDAF)        # splitmix64 reference output
        self.assertEqual(s.rng_next(), 0x6E789E6AA1B965F4)
        s.seed(-1)                                                # negative seeds wrap to uint64
        a = s.uniform()
        self.assertTrue(0.0 <= a < 1.0)

    def test_greedy_is_argmax(self):
        rng = random.Random(1)
        s = self.make(V)
        for trial in range(20):
            lg = rand_logits(rng)
            if trial % 4 == 0:
                m = max(lg)
                lg[100] = lg[7] = m + 1.0                         # a tie: the first index wins
            best = max(range(V), key=lambda i: (lg[i], -i))
            for p in (SamplerParams(temperature=0.0), SamplerParams(temperature=0.7, top_k=1)):
                self.assertEqual(s.sample(lg, [], p), best)
                self.assertEqual(s.candidates(lg, [], p), [(best, 1.0)])

    def test_penalties(self):
        n = 6
        s = self.make(n)
        lg = (ctypes.c_float * n)(2.0, 1.5, -1.0, 0.5, 1.9, -3.0)
        # repetition 2: token 0 (positive) 2.0 -> 1.0; token 2 (negative) -1 -> -2
        p = SamplerParams(temperature=0.0, repetition_penalty=2.0)
        self.assertEqual(s.sample(lg, [0, 0, 2], p), 4)           # 1.9 is now the max
        self.assertEqual(s.sample(lg, [], p), 0)
        # presence 0.3 + frequency 0.1 * count: token 0 seen 3x -> 2.0 - 0.6 = 1.4 < 1.5
        p = SamplerParams(temperature=0.0, presence_penalty=0.3, frequency_penalty=0.1)
        self.assertEqual(s.sample(lg, [0, 0, 0, 4], p), 1)
        self.assertEqual(s.sample(lg, [0, 4], p), 0)              # 2.0 - 0.4 = 1.6 > 1.5, 1.9 - 0.4
        # out-of-range ids in the window are ignored
        self.assertEqual(s.sample(lg, [-5, 99], SamplerParams(temperature=0.0, repetition_penalty=5)), 0)
        # penalised probabilities: repetition 2 on token 0, temperature 1
        c = dict(s.candidates(lg, [0], SamplerParams(temperature=1.0, repetition_penalty=2.0)))
        vals = list(lg)                                           # float32 values
        vals[0] /= 2.0
        ref = softmax(vals)
        for i in range(n):
            self.assertAlmostEqual(c[i], ref[i], places=12)

    def test_top_k_mask(self):
        rng = random.Random(2)
        s = self.make(V)
        for k in (1, 2, 5, 40, 50, 1000):
            lg = rand_logits(rng)
            lg[11] = lg[12] = lg[13] = 50.0                        # ties at the top
            p = SamplerParams(temperature=0.8, top_k=k)
            got = s.candidates(lg, [], p)
            order = sorted(range(V), key=lambda i: (-lg[i], i))[:k]
            self.assertEqual([i for i, _ in got], order)
            if k > 1:
                ref = softmax([lg[i] / 0.8 for i in order])
                for (_, pr), r in zip(got, ref):
                    self.assertAlmostEqual(pr, r, places=12)
            self.assertAlmostEqual(sum(pr for _, pr in got), 1.0, places=12)
            s.seed(k)
            for _ in range(50):
                self.assertIn(s.sample(lg, [], p), order)

    def test_top_p_mask(self):
        rng = random.Random(3)
        s = self.make(V)
        for tp in (0.05, 0.3, 0.5, 0.9, 0.95, 0.999):
            for temp in (0.5, 1.0, 1.7):
                lg = rand_logits(rng, scale=2.5)
                p = SamplerParams(temperature=temp, top_p=tp)
                got = [i for i, _ in s.candidates(lg, [], p)]
                probs = softmax([v / temp for v in lg])
                order = sorted(range(V), key=lambda i: (-lg[i], i))
                acc, ref = 0.0, []
                for i in order:                                    # the shortest prefix reaching tp
                    ref.append(i)
                    acc += probs[i]
                    if acc >= tp:
                        break
                self.assertEqual(got, ref, (tp, temp))
                mass = sum(probs[i] for i in got)
                self.assertGreaterEqual(mass, tp - 1e-9)
                self.assertLess(mass - probs[got[-1]], tp + 1e-9)

    def test_top_k_then_top_p(self):
        rng = random.Random(4)
        s = self.make(V)
        lg = rand_logits(rng, scale=1.0)
        p = SamplerParams(temperature=1.0, top_k=20, top_p=0.5)
        got = [i for i, _ in s.candidates(lg, [], p)]
        top = sorted(range(V), key=lambda i: (-lg[i], i))[:20]
        probs = softmax([lg[i] for i in top])
        acc, ref = 0.0, []
        for i, pr in zip(top, probs):
            ref.append(i)
            acc += pr
            if acc >= 0.5:
                break
        self.assertEqual(got, ref)

    def test_distribution(self):
        n = 64
        s = self.make(n)
        vals = [-math.inf] * n
        for i, v in {3: 1.0, 10: 0.5, 17: 0.0, 40: -1.0, 63: 2.0}.items():
            vals[i] = v
        lg = (ctypes.c_float * n)(*vals)
        for temp in (1.0, 0.5, 2.0):
            ref = {i: q for i, q in zip(range(n), softmax([v / temp for v in vals])) if q > 0}
            s.seed(12345)
            N = 20000 if isinstance(s, CSampler) else 4000
            counts = {}
            p = SamplerParams(temperature=temp)
            for _ in range(N):
                t = s.sample(lg, [], p)
                counts[t] = counts.get(t, 0) + 1
            self.assertEqual(set(counts) - set(ref), set())       # -inf is never drawn
            for i, q in ref.items():
                f = counts.get(i, 0) / N
                self.assertLess(abs(f - q), 5 * math.sqrt(q * (1 - q) / N) + 1e-3, (temp, i, f, q))

    def test_seeded_determinism(self):
        rng = random.Random(5)
        s = self.make(V)
        lg = rand_logits(rng, scale=1.0)
        p = SamplerParams(temperature=1.0, top_p=0.95)
        runs = []
        for seed in (42, 42, 43):
            s.seed(seed)
            runs.append([s.sample(lg, [], p) for _ in range(8)])
        self.assertEqual(runs[0], runs[1])
        self.assertNotEqual(runs[0], runs[2])

    def test_nan_and_inputs(self):
        s = self.make(8)
        lg = (ctypes.c_float * 8)(0.0, float("nan"), 1.0, -math.inf, 0.5, 0.2, 0.1, 0.0)
        self.assertEqual(s.sample(lg, [], SamplerParams(temperature=0.0)), 2)
        s.seed(1)
        for _ in range(200):
            self.assertNotIn(s.sample(lg, [], SamplerParams(temperature=1.0)), (1, 3))
        a = array.array("f", list(lg))                            # other buffer types
        self.assertEqual(s.sample(a, [], SamplerParams(temperature=0.0)), 2)
        self.assertEqual(s.sample(list(lg), [], SamplerParams(temperature=0.0)), 2)


class TestPySampler(SamplerCases, unittest.TestCase):
    def make(self, n):
        return PySampler(n)


@unittest.skipUnless(sampler_lib(), "no C compiler for src/sampler.c")
class TestCSampler(SamplerCases, unittest.TestCase):
    def make(self, n):
        return CSampler(sampler_lib(), n)

    def test_c_equals_python(self):
        rng = random.Random(6)
        c, py = CSampler(sampler_lib(), V), PySampler(V)
        params = [SamplerParams(0.0), SamplerParams(0.0, repetition_penalty=1.3),
                  SamplerParams(0.2, 0.9, 50), SamplerParams(0.7, 0.9, 0, 1.1),
                  SamplerParams(1.0, 0.95), SamplerParams(1.3, 1.0, 0, 1.0, 0.5, 0.2),
                  SamplerParams(0.8, 0.5, 40), SamplerParams(2.0, 0.999), SamplerParams(1.0, 1.0, 0)]
        n = 0
        for trial in range(6):
            lg = rand_logits(rng)
            if trial % 3 == 0:
                for i in range(0, V, 7):
                    lg[i] = lg[3]                                  # many ties
            recent = [rng.randrange(V) for _ in range(rng.randint(0, 64))] + [3, 3]
            for p in params:
                seed = rng.getrandbits(64)
                c.seed(seed)
                py.seed(seed)
                self.assertEqual([c.sample(lg, recent, p) for _ in range(4)],
                                 [py.sample(lg, recent, p) for _ in range(4)], p)
                self.assertEqual(c.candidates(lg, recent, p), py.candidates(lg, recent, p), p)
                n += 1
        self.assertEqual(n, 6 * len(params))

    def test_make_sampler(self):
        self.assertIsInstance(make_sampler(V, sampler_lib()), CSampler)
        self.assertIsInstance(make_sampler(V, "/nonexistent/libsampler.so"), PySampler)
        with self.assertRaises(OSError):
            make_sampler(V, "/nonexistent/libsampler.so", allow_python=False)


if __name__ == "__main__":
    unittest.main()
