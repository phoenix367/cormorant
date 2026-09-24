"""
MatmulKernel hardware-bound constants resolved from the platform JSON.

The MatmulKernel sizes its row-staging buffer ``a_buf[kTileN][kMaxK]``
at compile time, so a model whose inner dimension ``k`` exceeds
``kMaxK`` has no fallback path.  The Python scheduler must reject such
models *before* codegen, and the bound it checks against MUST match
whatever the actual C++ build was configured with — otherwise generated
C code would reference geometries the kernel can't service.

Source of truth: ``platforms/<AXI_PLATFORM>.json`` ``kernels.matmul``
object, the same file the C++ CMake build reads (see
``matmul_load_constants()`` in ``kernels/matmul/CMakeLists.txt``).
Keeping a single JSON file as the single source means the Python
validator and the C++ kernel build cannot drift apart.

JSON shape (only the fields this module reads)::

    {
      "kernels": {
        "matmul": {
          "tile_n":   4,
          "tile_m":  16,
          "tile_k": 256,
          "max_k": 2048
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

Missing / malformed fields raise ``MatmulHwConfigError`` rather than
silently falling back to defaults.

``tile_n``, ``tile_k`` are read but not exported to the validator: any
``n`` / ``m`` / ``k`` runs (residual-tile padded inside the kernel), so
models cannot violate them.  ``tile_m`` is exported because the packed
tile-major B layout the scheduler emits for constant weights
(``MatmulNode.b_packed``) pads ``m`` to a multiple of it.  Only ``max_k`` is the
hard upper bound — see ``doc/MATMUL_KERNEL.md`` §3 "Runtime constraint
validated by the scheduler".
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Mapping, Tuple


# Repo layout:
#   <repo>/inference-scheduler/src/_matmul_hw_config.py   ← this file
#   <repo>/platforms/<name>.json
_SCHEDULER_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT     = _SCHEDULER_DIR.parent
_PLATFORMS_DIR = _REPO_ROOT / "platforms"

_DEFAULT_PLATFORM = "kv260"

# Required fields and the Python attribute they get exported as.
_REQUIRED: Tuple[Tuple[str, str], ...] = (
    ("max_k",  "MATMUL_MAX_K"),
    ("tile_m", "MATMUL_TILE_M"),   # packed-B tile width (MATMUL_OPTIMISATION §3b)
)


class MatmulHwConfigError(RuntimeError):
    """Platform JSON is missing / malformed for the matmul kernel."""


def _resolve_platform_path(platform_name: str) -> Path:
    path = _PLATFORMS_DIR / f"{platform_name}.json"
    if not path.is_file():
        raise MatmulHwConfigError(
            f"Platform file not found: {path}.  Set AXI_PLATFORM to a "
            f"<name> for which platforms/<name>.json exists."
        )
    return path


def _load_matmul_section(path: Path) -> Mapping[str, int]:
    try:
        with path.open("r") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise MatmulHwConfigError(
            f"Platform JSON {path} is not valid JSON: {e}"
        ) from e
    try:
        section = data["kernels"]["matmul"]
    except (KeyError, TypeError) as e:
        raise MatmulHwConfigError(
            f"Platform JSON {path} is missing the 'kernels.matmul' object."
        ) from e
    if not isinstance(section, Mapping):
        raise MatmulHwConfigError(
            f"Platform JSON {path}: 'kernels.matmul' must be an object, "
            f"got {type(section).__name__}."
        )
    return section


def resolve(platform_name: str = None) -> Dict[str, int]:
    """Resolve MatmulKernel constants for the given platform.

    ``platform_name=None`` falls back to ``AXI_PLATFORM`` from the env, then
    to the built-in default ``kv260``.  Function-based so tests can probe
    arbitrary platforms without re-importing this module — module-level
    constants below freeze the default-platform values at import time.
    """
    if platform_name is None:
        platform_name = os.environ.get("AXI_PLATFORM", _DEFAULT_PLATFORM)
    path    = _resolve_platform_path(platform_name)
    section = _load_matmul_section(path)
    out: Dict[str, int] = {}
    for json_key, py_attr in _REQUIRED:
        if json_key not in section:
            raise MatmulHwConfigError(
                f"Platform JSON {path}: 'kernels.matmul.{json_key}' is "
                f"required but missing.  Add it (or pick a different "
                f"AXI_PLATFORM) so the Python validator and C++ kernel "
                f"build agree on bounds."
            )
        val = section[json_key]
        if not isinstance(val, int) or isinstance(val, bool):
            raise MatmulHwConfigError(
                f"Platform JSON {path}: 'kernels.matmul.{json_key}' must "
                f"be an integer, got {type(val).__name__} ({val!r})."
            )
        out[py_attr] = val
    return out


_CFG = resolve()

# Exported constants — these are what `nodes.py` validates against.
MATMUL_MAX_K : int = _CFG["MATMUL_MAX_K"]
MATMUL_TILE_M: int = _CFG["MATMUL_TILE_M"]


__all__ = (
    "MATMUL_MAX_K",
    "MATMUL_TILE_M",
    "MatmulHwConfigError",
    "resolve",
)
