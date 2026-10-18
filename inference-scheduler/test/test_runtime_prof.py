# Copyright 2025 GradeBuilder SL
# SPDX-License-Identifier: Apache-2.0

"""Tests for the runtime/ inference_prof.{h,c} module.

Compiles inference_prof.c together with runtime/test/test_inference_prof.c
once per session, then invokes the resulting binary with a scenario name
and asserts on the LAYERS_JSON line it prints.

Skipped on hosts without a C compiler — keeps the pure-Python pytest
workflow viable when only static analysis is wanted.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT       = Path(__file__).resolve().parents[1]
RUNTIME    = ROOT / "runtime"
DRIVER_SRC = RUNTIME / "test" / "test_inference_prof.c"
PROF_SRC   = RUNTIME / "inference_prof.c"


def _which_cc() -> str:
    for cc in ("cc", "gcc", "clang"):
        p = shutil.which(cc)
        if p:
            return p
    return ""


@unittest.skipUnless(_which_cc(), "C compiler not available on host")
class TestRuntimeProf(unittest.TestCase):
    """Each test invokes a single C scenario from test_inference_prof.c
    and parses its LAYERS_JSON output."""

    binary: str = ""
    tmp:    str = ""

    @classmethod
    def setUpClass(cls):
        cls.tmp    = tempfile.mkdtemp(prefix="prof_runtime_test_")
        cls.binary = os.path.join(cls.tmp, "prof_test")
        cmd = [
            _which_cc(),
            "-std=c99", "-Wall", "-Wextra", "-Werror",
            "-DINFERENCE_PROFILING=1",
            "-I", str(RUNTIME),
            str(PROF_SRC),
            str(DRIVER_SRC),
            "-o", cls.binary,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise unittest.SkipTest(
                f"compile failed:\nCMD: {' '.join(cmd)}\n"
                f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
            )

    @classmethod
    def tearDownClass(cls):
        if cls.tmp:
            shutil.rmtree(cls.tmp, ignore_errors=True)

    def _run(self, scenario: str) -> dict:
        proc = subprocess.run(
            [self.binary, scenario],
            capture_output=True, text=True, check=True, timeout=15,
        )
        for line in proc.stdout.splitlines():
            if line.startswith("LAYERS_JSON:"):
                return json.loads(line[len("LAYERS_JSON:"):].strip())
        self.fail(
            f"no LAYERS_JSON line in scenario '{scenario}':\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )

    # ---- scenarios ---------------------------------------------------- #

    def test_basic_counters(self):
        d = self._run("basic")
        self.assertEqual(len(d["layers"]), 2)
        L0, L1 = d["layers"]
        self.assertEqual(L0["name"], "a")
        self.assertEqual(L1["name"], "b")
        self.assertEqual(L0["i"],    0)
        self.assertEqual(L1["i"],    1)
        self.assertEqual(L0["calls"], 3)
        self.assertEqual(L1["calls"], 3)
        # min ≤ mean ≤ max for both layers, and all > 0 since we slept.
        for L in (L0, L1):
            self.assertGreater(L["min_us"], 0.0)
            self.assertGreaterEqual(L["max_us"], L["min_us"])
            self.assertGreaterEqual(L["max_us"], L["mean_us"])
            self.assertGreaterEqual(L["mean_us"], L["min_us"])
            # total ≈ mean * calls (within float rounding)
            self.assertAlmostEqual(L["total_us"],
                                    L["mean_us"] * L["calls"],
                                    places=2)

    def test_reset_zeroes_counters(self):
        d = self._run("reset")
        L = d["layers"][0]
        self.assertEqual(L["calls"],    0)
        self.assertEqual(L["mean_us"],  0.0)
        self.assertEqual(L["min_us"],   0.0)   # not UINT64_MAX/1000
        self.assertEqual(L["max_us"],   0.0)
        self.assertEqual(L["total_us"], 0.0)
        self.assertEqual(L["name"],     "only")

    def test_first_sample_initialises_min(self):
        d = self._run("min_first_sample")
        L = d["layers"][0]
        self.assertEqual(L["calls"], 1)
        # If the UINT64_MAX initial value leaked through it would land
        # somewhere around 1.8e16 microseconds — sanity-check well below
        # that and well above zero.
        self.assertGreater(L["min_us"], 0.0)
        self.assertLess(L["min_us"], 1.0e9)
        self.assertEqual(L["min_us"], L["max_us"])
        self.assertEqual(L["mean_us"], L["min_us"])

    def test_json_escape(self):
        d = self._run("escape")
        names = [L["name"] for L in d["layers"]]
        # json.loads has already un-escaped the values, so each name
        # round-trips back to the original Python string we passed in C.
        self.assertEqual(names, [
            "quote\"X",
            "back\\Y",
            "tab\tZ",
            "newline\nW",
            "control\x01end",
        ])

    def test_zero_layers_dumps_empty_array(self):
        d = self._run("empty")
        self.assertEqual(d["layers"], [])

    def test_out_of_bounds_index_is_ignored(self):
        d = self._run("oob")
        L = d["layers"][0]
        self.assertEqual(L["calls"], 1)   # only the valid begin/end recorded

    def test_begin_end_layer_mismatch_is_ignored(self):
        d = self._run("mismatch")
        # begin(0) + end(1) — neither layer should record a sample.
        # The follow-up clean begin(1)/end(1) must still work, so layer 1
        # ends up with one call.
        self.assertEqual(d["layers"][0]["calls"], 0)
        self.assertEqual(d["layers"][1]["calls"], 1)

    def test_double_init_replaces_state(self):
        d = self._run("double_init")
        self.assertEqual(len(d["layers"]), 2)
        names = [L["name"] for L in d["layers"]]
        self.assertEqual(names, ["new0", "new1"])
        self.assertEqual(d["layers"][0]["calls"], 1)
        self.assertEqual(d["layers"][1]["calls"], 0)


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(__file__))
    unittest.main()
