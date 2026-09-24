"""
ConvKernel hardware-bound constants resolved from the platform JSON.

The ConvKernel sizes its line buffer, bias buffer, weight cache, and
persistent-accumulator buffer at compile time, so a model whose geometry
violates those bounds has no fallback path.  The Python scheduler must
reject such models *before* codegen, and the bound values it checks
against MUST match whatever the actual C++ build was configured with —
otherwise generated C code would reference geometries the kernel can't
service.

Source of truth: ``platforms/<AXI_PLATFORM>.json`` ``kernels.conv``
object, the same file the C++ CMake build reads (see
``conv_load_constants()`` in ``kernels/conv/CMakeLists.txt``).  Keeping a
single JSON file as the single source means the Python validator and
the C++ kernel build cannot drift apart.

JSON shape (only the fields this module reads)::

    {
      "kernels": {
        "conv": {
          "tile_m":                  8,
          "tile_ic":                 16,
          "max_kh":                  7,
          "max_kw":                  7,
          "max_in_ch":               1024,
          "max_out_ch":              1024,
          "max_line_buf_cols":       64,
          "max_line_buf_rows":       16,
          "max_acc_persist_entries": 65536,
          "max_m_per_group":         4
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

Missing / malformed fields raise ``ConvHwConfigError`` rather than
silently falling back to defaults.

``tile_ic``, ``max_kh``, ``max_kw`` are read but not exported to the
validator: ``kTileIC`` is a pure unrolling factor (any in_ch is
residual-padded), and the kernel-size bounds are already validated
against weight tensor rank earlier in ConvNode.  ``tile_m`` IS exported
(``CONV_TILE_M``) because the kernel's persistent accumulator pads
out_ch up to a multiple of kTileM (ConvKernel.cpp §2.23 layout), so the
capacity rule the scheduler must enforce is
``out_w * ceil(out_ch / kTileM) * kTileM <= kMaxAccPersistEntries``.
The bounds exported here are the ones the scheduler must enforce ahead
of codegen — see ``doc/CONV_KERNEL.md`` §3 "Runtime constraints
validated by the inference scheduler".
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Mapping, Tuple


# Repo layout:
#   <repo>/inference-scheduler/src/_conv_hw_config.py   ← this file
#   <repo>/platforms/<name>.json
_SCHEDULER_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT     = _SCHEDULER_DIR.parent
_PLATFORMS_DIR = _REPO_ROOT / "platforms"

_DEFAULT_PLATFORM = "kv260"

# Required fields and the Python attribute they get exported as.
_REQUIRED: Tuple[Tuple[str, str], ...] = (
    ("max_in_ch",               "CONV_MAX_IN_CH"),
    ("max_out_ch",              "CONV_MAX_OUT_CH"),
    ("max_line_buf_rows",       "CONV_MAX_LINE_BUF_ROWS"),
    ("max_line_buf_cols",       "CONV_MAX_LINE_BUF_COLS"),
    ("max_acc_persist_entries", "CONV_MAX_ACC_PERSIST_ENTRIES"),
    ("tile_m",                  "CONV_TILE_M"),
    ("tile_ic",                 "CONV_TILE_IC"),
)


class ConvHwConfigError(RuntimeError):
    """Platform JSON is missing / malformed for the conv kernel."""


def _resolve_platform_path(platform_name: str) -> Path:
    path = _PLATFORMS_DIR / f"{platform_name}.json"
    if not path.is_file():
        raise ConvHwConfigError(
            f"Platform file not found: {path}.  Set AXI_PLATFORM to a "
            f"<name> for which platforms/<name>.json exists."
        )
    return path


def _load_conv_section(path: Path) -> Mapping[str, int]:
    try:
        with path.open("r") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ConvHwConfigError(
            f"Platform JSON {path} is not valid JSON: {e}"
        ) from e
    try:
        section = data["kernels"]["conv"]
    except (KeyError, TypeError) as e:
        raise ConvHwConfigError(
            f"Platform JSON {path} is missing the 'kernels.conv' object."
        ) from e
    if not isinstance(section, Mapping):
        raise ConvHwConfigError(
            f"Platform JSON {path}: 'kernels.conv' must be an object, "
            f"got {type(section).__name__}."
        )
    return section


def resolve(platform_name: str = None) -> Dict[str, int]:
    """Resolve ConvKernel constants for the given platform.

    ``platform_name=None`` falls back to ``AXI_PLATFORM`` from the env, then
    to the built-in default ``kv260``.  Function-based so tests can probe
    arbitrary platforms without re-importing this module — module-level
    constants below freeze the default-platform values at import time.
    """
    if platform_name is None:
        platform_name = os.environ.get("AXI_PLATFORM", _DEFAULT_PLATFORM)
    path    = _resolve_platform_path(platform_name)
    section = _load_conv_section(path)
    out: Dict[str, int] = {}
    for json_key, py_attr in _REQUIRED:
        if json_key not in section:
            raise ConvHwConfigError(
                f"Platform JSON {path}: 'kernels.conv.{json_key}' is "
                f"required but missing.  Add it (or pick a different "
                f"AXI_PLATFORM) so the Python validator and C++ kernel "
                f"build agree on bounds."
            )
        val = section[json_key]
        if not isinstance(val, int) or isinstance(val, bool):
            raise ConvHwConfigError(
                f"Platform JSON {path}: 'kernels.conv.{json_key}' must "
                f"be an integer, got {type(val).__name__} ({val!r})."
            )
        out[py_attr] = val
    return out


_CFG = resolve()

# Exported constants — these are what `nodes.py` validates against.
CONV_MAX_IN_CH               : int = _CFG["CONV_MAX_IN_CH"]
CONV_MAX_OUT_CH              : int = _CFG["CONV_MAX_OUT_CH"]
CONV_MAX_LINE_BUF_ROWS       : int = _CFG["CONV_MAX_LINE_BUF_ROWS"]
CONV_MAX_LINE_BUF_COLS       : int = _CFG["CONV_MAX_LINE_BUF_COLS"]
CONV_MAX_ACC_PERSIST_ENTRIES : int = _CFG["CONV_MAX_ACC_PERSIST_ENTRIES"]
CONV_TILE_M                  : int = _CFG["CONV_TILE_M"]
CONV_TILE_IC                 : int = _CFG["CONV_TILE_IC"]

# ConvKernel's weight / bias ports are hls::burst_maxi<ap_uint<128>>:
# kWeightPortBits / kDataBits = 128 / 16 Data_t lanes per beat
# (ConvKernel.h, "Weight / bias port width and DDR layout").  The packed
# weight layout pads each input-channel tile to CONV_TILE_IC lanes and each
# depthwise channel / the bias to a multiple of this many elements.
CONV_WEIGHT_PORT_ELEMS       : int = 128 // 16


__all__ = (
    "CONV_MAX_IN_CH",
    "CONV_MAX_OUT_CH",
    "CONV_MAX_LINE_BUF_ROWS",
    "CONV_MAX_LINE_BUF_COLS",
    "CONV_MAX_ACC_PERSIST_ENTRIES",
    "CONV_TILE_M",
    "CONV_TILE_IC",
    "CONV_WEIGHT_PORT_ELEMS",
    "ConvHwConfigError",
    "resolve",
)
