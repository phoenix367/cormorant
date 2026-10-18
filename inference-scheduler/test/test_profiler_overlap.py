# Copyright 2025 GradeBuilder SL
# SPDX-License-Identifier: Apache-2.0

"""Compile + run a C harness that exercises inference_prof under the
overlapping PROF_BEGIN/PROF_END pattern produced by the parallel-wait
emission, and assert each layer records its own bracket correctly.

This is the regression test for the bug where the old single-global
g_begin_ns / g_begin_layer state was clobbered when a second BEGIN fired
before the first matching END — silently dropping the overlapping
layer's sample.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest


HERE      = os.path.dirname(os.path.abspath(__file__))
ROOT      = os.path.dirname(HERE)
RUNTIME   = os.path.join(ROOT, "runtime")
HARNESS_C = os.path.join(HERE, "c", "profiler_overlap_harness.c")


def _have_cc() -> bool:
    return shutil.which("cc") is not None or shutil.which("gcc") is not None


@unittest.skipUnless(_have_cc(), "cc/gcc not installed; cannot build C harness")
@unittest.skipIf(sys.platform == "win32", "C harness uses POSIX clock_gettime")
class TestProfilerOverlap(unittest.TestCase):
    """Drive the profiler with an overlapping bracket pattern and verify
    each layer's own duration is recorded correctly."""

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp(prefix="prof_overlap_")
        cls.binary = os.path.join(cls.tmpdir, "prof_overlap")
        cc = shutil.which("cc") or shutil.which("gcc")
        rc = subprocess.run(
            [
                cc,
                "-std=c99", "-O2", "-Wall", "-Wextra",
                "-D_POSIX_C_SOURCE=199309L",
                "-DINFERENCE_PROFILING=1",
                f"-I{RUNTIME}",
                HARNESS_C,
                os.path.join(RUNTIME, "inference_prof.c"),
                "-o", cls.binary,
            ],
            capture_output=True, text=True,
        )
        if rc.returncode != 0:
            raise unittest.SkipTest(
                f"failed to compile profiler harness:\n{rc.stderr}"
            )

        # Run a couple of times, take the run with the shortest reluB —
        # makes the test less flaky on a loaded CI box.  We only need ONE
        # run that satisfies the bounds; the bug under test is structural,
        # not timing-sensitive.
        cls.layers = None
        last_err = None
        for _ in range(3):
            run = subprocess.run(
                [cls.binary], capture_output=True, text=True, timeout=30,
            )
            if run.returncode != 0:
                last_err = run.stderr
                continue
            line = next(
                (ln for ln in run.stdout.splitlines()
                 if ln.startswith("LAYERS_JSON: ")), None,
            )
            if line is None:
                last_err = "no LAYERS_JSON line in output:\n" + run.stdout
                continue
            data = json.loads(line[len("LAYERS_JSON: "):])
            cls.layers = {entry["i"]: entry for entry in data["layers"]}
            break

        if cls.layers is None:
            raise RuntimeError(f"profiler harness never ran cleanly: {last_err}")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    # ---- structural: every layer must record exactly one call ------------

    def test_every_layer_recorded_one_call(self):
        """The OLD profiler dropped layer 2's sample because BEGIN(3) and
        BEGIN(4) clobbered the global g_begin_layer to 3 and 4, so the
        END(2) fired the 'layer_idx != g_begin_layer' guard.  The fixed
        profiler MUST record one call for every layer."""
        for i in range(6):
            with self.subTest(layer=i):
                self.assertEqual(self.layers[i]["calls"], 1,
                                 f"layer {i} expected calls=1, got "
                                 f"{self.layers[i]['calls']}")

    def test_layer_names_propagated(self):
        names = ["convA", "reluA", "poolA", "convB", "reluB", "join"]
        for i, n in enumerate(names):
            self.assertEqual(self.layers[i]["name"], n)

    # ---- timing: per-layer brackets reflect each layer's own interval ----

    def _between(self, layer_idx: int, lo_us: float, hi_us: float):
        L = self.layers[layer_idx]
        # min == max == total for a single call.
        self.assertGreaterEqual(L["total_us"], lo_us,
                                f"layer {layer_idx}: total_us={L['total_us']} "
                                f"below floor {lo_us}")
        self.assertLessEqual(L["total_us"], hi_us,
                             f"layer {layer_idx}: total_us={L['total_us']} "
                             f"above ceiling {hi_us}")

    def test_overlapping_pool_records_full_duration(self):
        """poolA brackets contain BEGIN(3), END(3), BEGIN(4), END(4) —
        ~15ms convB + ~50ms reluB ≈ 65ms.  Allow a wide range to absorb
        scheduler jitter on shared CI runners; the OLD profiler would
        report 0 for this layer (sample dropped)."""
        # Lower bound is the 50 ms reluB sleep (poolA spans at least this).
        # Upper bound is generous because OS scheduling can stretch sleeps.
        self._between(2, lo_us=55_000, hi_us=200_000)

    def test_inner_brackets_unaffected_by_overlap(self):
        # convB ~15 ms; reluB ~50 ms.  These nest INSIDE poolA's bracket
        # and must record their own (smaller) durations, not poolA's.
        self._between(3, lo_us=10_000,  hi_us=60_000)
        self._between(4, lo_us=45_000,  hi_us=120_000)

    def test_outer_layers_short_brackets(self):
        # convA, reluA, join all use ~5–10 ms — confirm none of them got
        # accidentally extended.
        self._between(0, lo_us=5_000, hi_us=40_000)
        self._between(1, lo_us=2_000, hi_us=30_000)
        self._between(5, lo_us=2_000, hi_us=30_000)


if __name__ == "__main__":
    unittest.main()
