# Copyright 2025 GradeBuilder SL
# SPDX-License-Identifier: Apache-2.0

"""SSH / remote config: shared defaults, merge, load, and UIO helpers."""

import json
from typing import Optional

SHARED_DEFAULTS: dict = {
    "ssh": {
        "host":            "",
        "user":            "root",
        "port":            22,
        "key_file":        None,
        "password":        None,
        "connect_timeout": 15,
    },
    "remote": {
        "work_dir":    "/tmp/inference_hw_tests",
        "driver_dir":  None,
        "driver_dirs": {},
        "uio_devices": {},
        "uio_device":  None,
        "cmake_args":  [],
    },
    "build": {
        "jobs":    4,
        "timeout": 180,
    },
    "run": {
        "timeout":  120,
        "use_sudo": True,
    },
    "local": {
        "driver_dir":  None,
        "driver_dirs": {},
    },
    "cleanup": True,
}


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into a deep copy of *base*."""
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def load_config(path: str, extra_defaults: Optional[dict] = None) -> dict:
    """Load JSON config, merge with SHARED_DEFAULTS (and optional extra_defaults)."""
    with open(path) as fh:
        raw = json.load(fh)
    base = deep_merge(SHARED_DEFAULTS, extra_defaults or {})
    cfg  = deep_merge(base, raw)
    if not cfg["ssh"]["host"]:
        raise ValueError("config: ssh.host is required")
    return cfg


def uio_devices_from_cfg(cfg: dict) -> dict:
    """Return per-kernel UIO sysfs name map (handles legacy uio_device string)."""
    remote = cfg["remote"]
    if remote.get("uio_devices"):
        return dict(remote["uio_devices"])
    legacy = remote.get("uio_device")
    if legacy:
        return {"VectorOPKernel": legacy}
    return {}
