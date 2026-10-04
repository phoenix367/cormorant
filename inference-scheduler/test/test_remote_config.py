"""Tests for src.remote.config — SHARED_DEFAULTS, deep_merge, load_config, UIO helpers."""

import json
import os
import tempfile
import unittest
from pathlib import Path

from src.remote.config import (
    SHARED_DEFAULTS,
    deep_merge,
    load_config,
    uio_devices_from_cfg,
)


# ---------------------------------------------------------------- #
# SHARED_DEFAULTS shape                                             #
# ---------------------------------------------------------------- #

class TestSharedDefaults(unittest.TestCase):
    def test_required_top_level_sections_present(self):
        for section in ("ssh", "remote", "build", "run", "local"):
            self.assertIn(section, SHARED_DEFAULTS)
        self.assertIn("cleanup", SHARED_DEFAULTS)

    def test_ssh_defaults_empty_host(self):
        # Empty host is the sentinel that load_config() validates against.
        self.assertEqual(SHARED_DEFAULTS["ssh"]["host"], "")
        self.assertEqual(SHARED_DEFAULTS["ssh"]["user"], "root")
        self.assertEqual(SHARED_DEFAULTS["ssh"]["port"], 22)


# ---------------------------------------------------------------- #
# deep_merge — recursive dict merge                                 #
# ---------------------------------------------------------------- #

class TestDeepMerge(unittest.TestCase):
    def test_top_level_override_and_add(self):
        base     = {"a": 1, "b": 2}
        override = {"b": 99, "c": 3}
        self.assertEqual(deep_merge(base, override),
                         {"a": 1, "b": 99, "c": 3})

    def test_nested_dict_merges_recursively(self):
        base     = {"ssh": {"host": "", "port": 22, "user": "root"}}
        override = {"ssh": {"host": "kv260.local", "port": 2222}}
        out      = deep_merge(base, override)
        self.assertEqual(out["ssh"],
                         {"host": "kv260.local", "port": 2222, "user": "root"})

    def test_non_dict_override_replaces_dict(self):
        # When the override's value is not a dict, it wholesale replaces.
        base     = {"x": {"nested": 1}}
        override = {"x": "no longer dict"}
        self.assertEqual(deep_merge(base, override), {"x": "no longer dict"})

    def test_does_not_mutate_inputs(self):
        base     = {"a": {"b": 1}}
        override = {"a": {"b": 2}}
        deep_merge(base, override)
        self.assertEqual(base["a"]["b"], 1, "deep_merge mutated `base`")


# ---------------------------------------------------------------- #
# load_config                                                       #
# ---------------------------------------------------------------- #

class TestLoadConfig(unittest.TestCase):
    def _write_json(self, payload: dict) -> Path:
        fd, p = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        path = Path(p)
        path.write_text(json.dumps(payload))
        return path

    def test_happy_path_preserves_defaults(self):
        path = self._write_json({"ssh": {"host": "kv260.local"}})
        try:
            cfg = load_config(str(path))
        finally:
            path.unlink(missing_ok=True)
        self.assertEqual(cfg["ssh"]["host"], "kv260.local")
        self.assertEqual(cfg["ssh"]["user"], "root")    # SHARED default preserved
        self.assertEqual(cfg["build"]["jobs"], 4)       # SHARED default preserved
        self.assertEqual(cfg["remote"]["work_dir"], "/tmp/inference_hw_tests")

    def test_missing_ssh_host_raises(self):
        path = self._write_json({"ssh": {"user": "alice"}})
        try:
            with self.assertRaisesRegex(ValueError, "ssh.host is required"):
                load_config(str(path))
        finally:
            path.unlink(missing_ok=True)

    def test_extra_defaults_applied_before_user_config(self):
        path = self._write_json({"ssh": {"host": "x"}})
        try:
            cfg = load_config(str(path), extra_defaults={"build": {"jobs": 8}})
        finally:
            path.unlink(missing_ok=True)
        self.assertEqual(cfg["build"]["jobs"],    8)
        self.assertEqual(cfg["build"]["timeout"], 180)  # SHARED default preserved


# ---------------------------------------------------------------- #
# uio_devices_from_cfg                                              #
# ---------------------------------------------------------------- #

class TestUioDevicesFromCfg(unittest.TestCase):
    def test_per_kernel_map_takes_priority_over_legacy_string(self):
        cfg = {"remote": {
            "uio_devices": {"VectorOPKernel": "vop_0", "ConvKernel": "cnv_0"},
            "uio_device":  "should_be_ignored",
        }}
        self.assertEqual(
            uio_devices_from_cfg(cfg),
            {"VectorOPKernel": "vop_0", "ConvKernel": "cnv_0"},
        )

    def test_legacy_string_wraps_as_vectoropkernel(self):
        cfg = {"remote": {"uio_devices": {}, "uio_device": "legacy_name"}}
        self.assertEqual(uio_devices_from_cfg(cfg),
                         {"VectorOPKernel": "legacy_name"})

    def test_returns_empty_when_no_uio_configured(self):
        cfg = {"remote": {"uio_devices": {}, "uio_device": None}}
        self.assertEqual(uio_devices_from_cfg(cfg), {})


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------- #
# run_remote_perf: a bad benchmark case fails before any SSH work   #
# ---------------------------------------------------------------- #

class TestPerfCaseValidationBeforeConnect(unittest.TestCase):
    def test_unsupported_op_is_a_config_error_before_connecting(self):
        import contextlib
        import io
        from unittest import mock

        import run_remote_perf

        here = Path(__file__).resolve().parent.parent
        cfg = json.loads((here / "perf_config.json.example").read_text())
        cfg["benchmarks"]["VectorOPKernel"]["cases"][0]["op"] = 99
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "perf.json")
            with open(path, "w") as f:
                json.dump(cfg, f)
            err = io.StringIO()
            with mock.patch.object(run_remote_perf, "RemoteSession",
                                   side_effect=AssertionError("connected")), \
                    contextlib.redirect_stderr(err):
                rc = run_remote_perf.main(["--config", path])
        self.assertEqual(rc, 1)
        self.assertIn("config error", err.getvalue())
        self.assertIn("unsupported op=99", err.getvalue())

    def test_pool_window_out_of_bounds_is_a_config_error_before_connecting(self):
        """The 14x14 GlobalMaxPool case that hung the kernel (POOL_OPTIMISATION §2.14)."""
        import contextlib
        import io
        from unittest import mock

        import run_remote_perf

        here = Path(__file__).resolve().parent.parent
        cfg = json.loads((here / "perf_config.json.example").read_text())
        cfg["benchmarks"]["PoolingKernel"]["cases"][0].update(
            label="GlobalMaxPool-14x14", in_h=14, in_w=14, pool_h=14, pool_w=14, stride_h=1, stride_w=1)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "perf.json")
            with open(path, "w") as f:
                json.dump(cfg, f)
            err = io.StringIO()
            with mock.patch.object(run_remote_perf, "RemoteSession",
                                   side_effect=AssertionError("connected")), \
                    contextlib.redirect_stderr(err):
                rc = run_remote_perf.main(["--config", path])
        self.assertEqual(rc, 1)
        self.assertIn("config error", err.getvalue())
        self.assertIn("pool_h=14 (1..kMaxPoolH=7)", err.getvalue())
