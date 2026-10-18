# Copyright 2025 GradeBuilder SL
# SPDX-License-Identifier: Apache-2.0

"""Tests for the runtime/inference_ddr module.

The success path requires an actual Zynq UltraScale+ DDR PMU (or any
PMU exposed under /sys/bus/event_source/devices/), so this suite only
exercises the graceful-failure path: every CI host should fall through
the candidate list, emit a single warning, and return a parseable
"available":false JSON line.
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

ROOT        = Path(__file__).resolve().parents[1]
RUNTIME     = ROOT / "runtime"
DRIVER_SRC  = RUNTIME / "test" / "test_inference_ddr.c"
DDR_SRC     = RUNTIME / "inference_ddr.c"
# Compile every backend under runtime/ddr/ — keeps the test honest as
# new backends are added.
BACKEND_SRCS = sorted((RUNTIME / "ddr").glob("*.c"))


def _which_cc() -> str:
    for cc in ("cc", "gcc", "clang"):
        p = shutil.which(cc)
        if p:
            return p
    return ""


@unittest.skipUnless(_which_cc(), "C compiler not available on host")
@unittest.skipUnless(sys.platform.startswith("linux"),
                      "inference_ddr is Linux-only (perf_event_open)")
class TestRuntimeDdr(unittest.TestCase):
    binary: str = ""
    tmp:    str = ""

    @classmethod
    def setUpClass(cls):
        cls.tmp    = tempfile.mkdtemp(prefix="ddr_runtime_test_")
        cls.binary = os.path.join(cls.tmp, "ddr_test")
        cmd = [
            _which_cc(),
            "-std=c99", "-Wall", "-Wextra", "-Werror",
            "-DINFERENCE_PROFILING=1",
            "-I", str(RUNTIME),
            str(DDR_SRC),
            *[str(s) for s in BACKEND_SRCS],
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

    def _run(self, scenario: str):
        proc = subprocess.run(
            [self.binary, scenario],
            capture_output=True, text=True, check=True, timeout=15,
        )
        ddr = None
        init_rc = None
        for line in proc.stdout.splitlines():
            line = line.strip()
            if line.startswith("DDR_JSON:"):
                ddr = json.loads(line[len("DDR_JSON:"):].strip())
            elif line.startswith("INIT_RC:"):
                init_rc = int(line[len("INIT_RC:"):].strip())
        if ddr is None:
            self.fail(
                f"no DDR_JSON line in scenario '{scenario}':\n"
                f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
            )
        return ddr, proc.stderr, init_rc

    # ---- scenarios ---------------------------------------------------- #

    def test_init_failure_emits_warning_and_unavailable_json(self):
        ddr, stderr, init_rc = self._run("init_failure_dump")
        self.assertFalse(ddr["available"])
        self.assertIn("reason", ddr)
        self.assertTrue(ddr["reason"], "non-empty reason expected")
        self.assertEqual(init_rc, -1)
        # The warning is written to stderr — exactly once.
        warnings = [
            ln for ln in stderr.splitlines()
            if "inference_ddr" in ln and "warning" in ln
        ]
        self.assertEqual(len(warnings), 1, stderr)

    def test_dump_without_init_reports_unavailable(self):
        ddr, _, _ = self._run("no_init_then_dump")
        self.assertFalse(ddr["available"])
        self.assertIn("reason", ddr)

    def test_start_stop_after_failed_init_is_safe(self):
        ddr, _, _ = self._run("start_stop_after_failed_init")
        self.assertFalse(ddr["available"])

    def test_double_init_is_safe(self):
        ddr, stderr, _ = self._run("double_init")
        self.assertFalse(ddr["available"])
        # Each init prints its own warning (we ran init twice).
        warnings = [
            ln for ln in stderr.splitlines()
            if "inference_ddr" in ln and "warning" in ln
        ]
        self.assertEqual(len(warnings), 2, stderr)

    def test_explicit_disable_silences_warning(self):
        """INFERENCE_DDR_BACKEND=disabled must skip the probe AND
        suppress the warning — the user opted out."""
        ddr, stderr, init_rc = self._run("disabled_via_env")
        self.assertFalse(ddr["available"])
        self.assertIn("disabled", ddr.get("reason", ""))
        self.assertEqual(init_rc, -1)
        warnings = [
            ln for ln in stderr.splitlines()
            if "inference_ddr" in ln and "warning" in ln
        ]
        self.assertEqual(len(warnings), 0, stderr)

    def test_unknown_backend_is_reported(self):
        """An unrecognised backend name surfaces clearly in both the
        warning and the JSON reason field."""
        ddr, stderr, init_rc = self._run("unknown_backend")
        self.assertFalse(ddr["available"])
        self.assertIn("no_such_backend", ddr.get("reason", ""))
        self.assertEqual(init_rc, -1)
        warnings = [
            ln for ln in stderr.splitlines()
            if "inference_ddr" in ln and "warning" in ln
        ]
        self.assertEqual(len(warnings), 1, stderr)


if __name__ == "__main__":
    unittest.main()
