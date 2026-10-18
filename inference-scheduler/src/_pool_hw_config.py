# Copyright 2025 GradeBuilder SL
# Modifications copyright 2026 Ivan Gubochkin
# SPDX-License-Identifier: Apache-2.0

"""
PoolingKernel hardware-bound constants resolved from the platform JSON.

The PoolingKernel sizes its line buffer and unrolled per-position adders at
compile time, so a model whose pool window violates those bounds has no
fallback path.  The Python scheduler must reject such models *before*
codegen, and the bound values it checks against MUST match whatever the
actual C++ build was configured with — otherwise generated C code would
reference geometries the kernel can't service.

Source of truth: ``platforms/<AXI_PLATFORM>.json`` ``kernels.pool`` object,
the same file the C++ CMake build reads (see ``pool_load_constants()`` in
``kernels/pool_rtl/model/CMakeLists.txt``).  Keeping a single JSON file as the single
source means the Python validator and the C++ kernel build cannot drift
apart — no parsing of ``CMakeLists.txt`` defaults, no dependency on
``CMakeCache.txt`` having been generated.

JSON shape (only the fields this module reads)::

    {
      "kernels": {
        "pool": {
          "tile_c":            8,
          "max_kh":            7,
          "max_kw":            7,
          "max_line_buf_rows": 16,
          "max_line_buf_cols": 64,
          "ow_parallel":       2
        }
      }
    }

Platform selection (lower entries override higher ones):

1. **Default** — ``platforms/kv260.json`` in the repo (the only platform
   shipped today).
2. **``AXI_PLATFORM`` environment variable** — picks
   ``platforms/<value>.json``.  Mirrors the CMake cache var of the same
   name so a user who configured C++ with ``cmake -DAXI_PLATFORM=foo``
   can run the Python tools against the same platform with
   ``AXI_PLATFORM=foo .venv/bin/python ...``.

Missing / malformed fields raise ``PoolHwConfigError`` rather than
silently falling back to defaults — silent defaults defeat the whole
"single source of truth" property.

``kTileC`` and ``kOwParallel`` are read from the JSON but not exported
to the validator: any C runs (channel-tiled) and any out_w runs (residual
padding), so models cannot violate them.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Mapping, Tuple


# Repo layout:
#   <repo>/inference-scheduler/src/_pool_hw_config.py   ← this file
#   <repo>/platforms/<name>.json
_SCHEDULER_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT     = _SCHEDULER_DIR.parent
_PLATFORMS_DIR = _REPO_ROOT / "platforms"

_DEFAULT_PLATFORM = "kv260"

# Required fields and the Python attribute they get exported as.
_REQUIRED: Tuple[Tuple[str, str], ...] = (
    ("max_kh",            "POOL_MAX_KH"),
    ("max_kw",            "POOL_MAX_KW"),
    ("max_line_buf_rows", "POOL_MAX_LINE_BUF_ROWS"),
    ("max_line_buf_cols", "POOL_MAX_LINE_BUF_COLS"),
)


class PoolHwConfigError(RuntimeError):
    """Platform JSON is missing / malformed for the pool kernel."""


def _resolve_platform_path(platform_name: str) -> Path:
    path = _PLATFORMS_DIR / f"{platform_name}.json"
    if not path.is_file():
        raise PoolHwConfigError(
            f"Platform file not found: {path}.  Set AXI_PLATFORM to a "
            f"<name> for which platforms/<name>.json exists."
        )
    return path


def _load_pool_section(path: Path) -> Mapping[str, int]:
    try:
        with path.open("r") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise PoolHwConfigError(
            f"Platform JSON {path} is not valid JSON: {e}"
        ) from e
    try:
        section = data["kernels"]["pool"]
    except (KeyError, TypeError) as e:
        raise PoolHwConfigError(
            f"Platform JSON {path} is missing the 'kernels.pool' object."
        ) from e
    if not isinstance(section, Mapping):
        raise PoolHwConfigError(
            f"Platform JSON {path}: 'kernels.pool' must be an object, "
            f"got {type(section).__name__}."
        )
    return section


def resolve(platform_name: str = None) -> Dict[str, int]:
    """Resolve PoolingKernel constants for the given platform.

    ``platform_name=None`` falls back to ``AXI_PLATFORM`` from the env, then
    to the built-in default ``kv260``.  Function-based so tests can probe
    arbitrary platforms without re-importing this module — module-level
    constants below freeze the default-platform values at import time.
    """
    if platform_name is None:
        platform_name = os.environ.get("AXI_PLATFORM", _DEFAULT_PLATFORM)
    path    = _resolve_platform_path(platform_name)
    section = _load_pool_section(path)
    out: Dict[str, int] = {}
    for json_key, py_attr in _REQUIRED:
        if json_key not in section:
            raise PoolHwConfigError(
                f"Platform JSON {path}: 'kernels.pool.{json_key}' is "
                f"required but missing.  Add it (or pick a different "
                f"AXI_PLATFORM) so the Python validator and C++ kernel "
                f"build agree on bounds."
            )
        val = section[json_key]
        if not isinstance(val, int) or isinstance(val, bool):
            raise PoolHwConfigError(
                f"Platform JSON {path}: 'kernels.pool.{json_key}' must "
                f"be an integer, got {type(val).__name__} ({val!r})."
            )
        out[py_attr] = val
    return out


_CFG = resolve()

# Exported constants — these are what `nodes.py` validates against.
POOL_MAX_KH            : int = _CFG["POOL_MAX_KH"]
POOL_MAX_KW            : int = _CFG["POOL_MAX_KW"]
POOL_MAX_LINE_BUF_ROWS : int = _CFG["POOL_MAX_LINE_BUF_ROWS"]
POOL_MAX_LINE_BUF_COLS : int = _CFG["POOL_MAX_LINE_BUF_COLS"]


__all__ = (
    "POOL_MAX_KH",
    "POOL_MAX_KW",
    "POOL_MAX_LINE_BUF_ROWS",
    "POOL_MAX_LINE_BUF_COLS",
    "PoolHwConfigError",
    "resolve",
)
