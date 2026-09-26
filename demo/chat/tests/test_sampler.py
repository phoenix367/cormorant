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


def dry_reference(hist, n, breakers, mult, base, allowed, max_match=50):
    """text-generation-webui's DRY processor (p-e-w, 2024), transcribed:
    token -> penalty subtracted from its logit."""
    if mult <= 0 or len(hist) < 2:
        return {}
    last = hist[-1]
    if last in breakers or not 0 <= last < n:
        return {}
    match_lengths = {}
    for i in [k for k in range(len(hist) - 1) if hist[k] == last]:
        nxt = hist[i + 1]
        if nxt in breakers or not 0 <= nxt < n:
            continue
        ml = 1
        while ml < max_match:
            j = i - ml
            if j < 0:
                break
            prev = hist[-(ml + 1)]
            if hist[j] != prev or prev in breakers or not 0 <= prev < n:
                break
            ml += 1
        match_lengths[nxt] = max(ml, match_lengths.get(nxt, 0))
    return {t: mult * base ** (ml - allowed) for t, ml in match_lengths.items() if ml >= allowed}


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

    def test_dry_matches_reference(self):
        """Penalties recovered from the candidate distribution (zero logits,
        temperature 1: log p_i - log p_j = pen_j - pen_i) equal the reference."""
        rng = random.Random(11)
        n = 40
        s = self.make(n)
        zeros = (ctypes.c_float * n)(*([0.0] * n))
        for trial in range(200):
            alphabet = rng.randint(2, 8)
            hist = [rng.randrange(alphabet) for _ in range(rng.randint(2, 120))]
            if trial % 3 == 0:                       # plant a long repeat
                k = rng.randint(3, 20)
                seg = hist[-k:] if len(hist) >= k else hist
                hist = hist + [rng.randrange(alphabet)] + seg
            brk = set(rng.sample(range(alphabet), rng.randint(0, 2)))
            s.set_breakers(sorted(brk) + [n + 5, -1])            # out-of-range ids ignored
            mult, base, allowed = rng.choice([0.8, 0.5, 2.0]), rng.choice([1.75, 1.1, 2.0]), rng.randint(1, 4)
            p = SamplerParams(temperature=1.0, dry_multiplier=mult, dry_base=base, dry_allowed_length=allowed)
            want = dry_reference(hist, n, brk, mult, base, allowed)
            cands = dict(s.candidates(zeros, [], p, hist=hist))
            ref = softmax([-want.get(t, 0.0) for t in range(n)])       # zero logits, temperature 1
            for t in range(n):
                self.assertAlmostEqual(cands.get(t, 0.0), ref[t], delta=1e-12 + 1e-9 * ref[t],
                                       msg=(trial, t, hist[-12:], brk))

    def test_dry_semantics(self):
        n = 10
        s = self.make(n)
        s.set_breakers([9])
        lg = (ctypes.c_float * n)(*([0.0] * n))
        p = SamplerParams(temperature=0.0, dry_multiplier=1.0, dry_base=2.0, dry_allowed_length=2)
        # 1 2 3 4 ... 1 2 3: token 4 would extend a run of 3 -> 1 * 2^(3-2) = 2
        hist = [1, 2, 3, 4, 5, 6, 1, 2, 3]
        c = dict(s.candidates(lg, [], SamplerParams(temperature=1.0, dry_multiplier=1.0, dry_base=2.0,
                                                    dry_allowed_length=2), hist=hist))
        self.assertAlmostEqual(math.log(c[4]) - math.log(c[0]), -2.0, places=12)
        self.assertNotEqual(s.sample(lg, [], p, hist=hist), 4)      # greedy avoids extending it
        # a run of 1 (< allowed_length) is free
        c = dict(s.candidates(lg, [], SamplerParams(temperature=1.0, dry_multiplier=1.0, dry_allowed_length=2),
                              hist=[1, 4, 5, 1]))
        self.assertAlmostEqual(c[4], c[0], places=15)
        # a breaker inside the earlier occurrence cuts the run: 1 9 3 ... 1 9 3 -> only "3" matches
        c = dict(s.candidates(lg, [], SamplerParams(temperature=1.0, dry_multiplier=1.0, dry_base=2.0,
                                                    dry_allowed_length=1), hist=[1, 9, 3, 4, 1, 9, 3]))
        self.assertAlmostEqual(math.log(c[4]) - math.log(c[0]), -1.0, places=12)   # L = 1: 2^0
        # the last token is a breaker: no penalty at all
        c = dict(s.candidates(lg, [], SamplerParams(temperature=1.0, dry_multiplier=1.0, dry_allowed_length=1),
                              hist=[9, 4, 9]))
        self.assertAlmostEqual(c[4], c[0], places=15)
        # dry_multiplier 0 and n_hist < 2: off
        for h, m in (([1, 2, 1], 0.0), ([1], 1.0), ([], 1.0)):
            c = dict(s.candidates(lg, [], SamplerParams(temperature=1.0, dry_multiplier=m, dry_allowed_length=1),
                                  hist=h))
            self.assertAlmostEqual(c[2], c[0], places=15)
        # hist defaults to recent; an explicit hist is used for DRY only
        rp = SamplerParams(temperature=1.0, dry_multiplier=1.0, dry_base=2.0, dry_allowed_length=1)
        c1 = dict(s.candidates(lg, [1, 2, 1], rp))
        c2 = dict(s.candidates(lg, [5], rp, hist=[1, 2, 1]))
        self.assertEqual(c1, c2)

    def test_dry_breaks_a_greedy_loop(self):
        """A model that always prefers continuing the cycle 0 1 2 3 4 0 1 2 ...
        by a margin of 4: greedy loops forever; with DRY the cycle breaks."""
        n = 16
        s = self.make(n)
        s.set_breakers([])

        def run(p, steps=60):
            hist = [0]
            for _ in range(steps):
                lg = (ctypes.c_float * n)(*([0.0] * n))
                lg[(hist[-1] + 1) % 5] = 4.0
                hist.append(s.sample(lg, [], p, hist=hist))
            return hist
        loop = run(SamplerParams(temperature=0.0))
        self.assertEqual(loop, [i % 5 for i in range(61)])
        broke = run(SamplerParams(temperature=0.0, dry_multiplier=0.8, dry_base=1.75, dry_allowed_length=2))
        self.assertNotEqual(broke, loop)
        # after the first repeat grows past allowed_length + log_1.75(4 / 0.8) ~ 5 tokens it deviates
        first_dev = next(i for i, (a, b) in enumerate(zip(broke, loop)) if a != b)
        self.assertLess(first_dev, 15)


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
                  SamplerParams(0.8, 0.5, 40), SamplerParams(2.0, 0.999), SamplerParams(1.0, 1.0, 0),
                  SamplerParams(0.0, dry_multiplier=0.8), SamplerParams(0.2, 0.9, 50, dry_multiplier=0.8),
                  SamplerParams(1.0, 0.95, 0, 1.1, dry_multiplier=2.0, dry_base=1.3, dry_allowed_length=1)]
        brk = [rng.randrange(V) for _ in range(40)] + [3]
        c.set_breakers(brk)
        py.set_breakers(brk)
        n = 0
        for trial in range(6):
            lg = rand_logits(rng)
            if trial % 3 == 0:
                for i in range(0, V, 7):
                    lg[i] = lg[3]                                  # many ties
            recent = [rng.randrange(V) for _ in range(rng.randint(0, 64))] + [3, 3]
            base = [rng.randrange(60) for _ in range(rng.randint(10, 300))]
            hist = base + base[: rng.randint(2, 40)]                     # a planted repeat
            for p in params:
                seed = rng.getrandbits(64)
                c.seed(seed)
                py.seed(seed)
                self.assertEqual([c.sample(lg, recent, p, hist=hist) for _ in range(4)],
                                 [py.sample(lg, recent, p, hist=hist) for _ in range(4)], p)
                self.assertEqual(c.candidates(lg, recent, p, hist=hist),
                                 py.candidates(lg, recent, p, hist=hist), p)
                n += 1
        self.assertEqual(n, 6 * len(params))

    def test_make_sampler(self):
        self.assertIsInstance(make_sampler(V, sampler_lib()), CSampler)
        self.assertIsInstance(make_sampler(V, "/nonexistent/libsampler.so"), PySampler)
        with self.assertRaises(OSError):
            make_sampler(V, "/nonexistent/libsampler.so", allow_python=False)


if __name__ == "__main__":
    unittest.main()
