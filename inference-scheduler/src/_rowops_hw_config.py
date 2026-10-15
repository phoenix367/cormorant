"""
RowOpsKernel capability resolved from the platform JSON.

``unit`` says whether the platform's bitstream carries the row-ops unit
(doc/plans/HOSTOPS_PLAN.md, doc/kernels/ROWOPS_RTL_KERNEL.md): the fifth
kernel lane that runs the transformer's row ops — LayerNorm / RMSNorm, the
residual add, SiLU · up, dequantisation, RoPE, transposes and copies — in
fixed point.  Where it is false the scheduler keeps those ops on the host
(``src/rowops_nodes.py``: the kernel nodes are chosen only with the unit).
``fold`` says whether its unit has op 9, FOLD (doc/plans/HOSTOPS_PLAN.md §11:
Piper's TtsPrep — windowed rows, LeakyReLU, the re-rounding); bitstreams up to
``1cedaf2ed816`` lack it, ``de968ccc45da`` has it.  It needs ``unit``.

JSON shape (only the fields this module reads)::

    {
      "kernels": {
        "rowops": {
          "unit": false,
          "fold": false
        }
      }
    }

Platform selection as for the other kernels (``_matmul_hw_config``): the
``AXI_PLATFORM`` environment variable picks ``platforms/<name>.json``,
default ``kv260``.  ``AXI_ROWOPS`` (``0`` / ``1``) overrides the JSON for
the scheduler — ``1`` to generate for a bitstream with the unit before the
platform file says so, ``0`` for one without it; ``AXI_ROWOPS_FOLD`` likewise
for ``fold`` (without the unit it is off whatever it says).  A missing or
malformed field raises ``RowopsHwConfigError``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict

_SCHEDULER_DIR = Path(__file__).resolve().parent.parent
_PLATFORMS_DIR = _SCHEDULER_DIR.parent / "platforms"
_DEFAULT_PLATFORM = "kv260"


class RowopsHwConfigError(RuntimeError):
    """Platform JSON is missing / malformed for the rowops kernel."""


def _flag(section, key: str, path, what: str, env_name: str) -> bool:
    try:
        v = section[key]
    except (KeyError, TypeError) as e:
        raise RowopsHwConfigError(
            f"Platform JSON {path}: 'kernels.rowops.{key}' is required (true when the "
            f"bitstream carries {what}).") from e
    if not isinstance(v, bool):
        raise RowopsHwConfigError(
            f"Platform JSON {path}: 'kernels.rowops.{key}' must be true or false, got {v!r}.")
    env = os.environ.get(env_name)
    if env:
        if env not in ("0", "1"):
            raise RowopsHwConfigError(f"{env_name} must be 0 or 1, got {env!r}.")
        v = env == "1"
    return v


def resolve(platform_name: str = None) -> Dict[str, object]:
    """``{"ROWOPS_UNIT": bool, "ROWOPS_FOLD": bool}`` for the given platform
    (``None``: ``AXI_PLATFORM``, else ``kv260``), ``AXI_ROWOPS`` /
    ``AXI_ROWOPS_FOLD`` applied."""
    if platform_name is None:
        platform_name = os.environ.get("AXI_PLATFORM", _DEFAULT_PLATFORM)
    path = _PLATFORMS_DIR / f"{platform_name}.json"
    if not path.is_file():
        raise RowopsHwConfigError(
            f"Platform file not found: {path}.  Set AXI_PLATFORM to a "
            f"<name> for which platforms/<name>.json exists.")
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise RowopsHwConfigError(f"Platform JSON {path} is not valid JSON: {e}") from e
    try:
        section = data["kernels"]["rowops"]
    except (KeyError, TypeError) as e:
        raise RowopsHwConfigError(f"Platform JSON {path}: 'kernels.rowops' is required.") from e
    unit = _flag(section, "unit", path, "the row-ops unit", "AXI_ROWOPS")
    fold = _flag(section, "fold", path, "the unit's op 9, FOLD", "AXI_ROWOPS_FOLD")
    return {"ROWOPS_UNIT": unit, "ROWOPS_FOLD": unit and fold}


_CFG = resolve()

# Exported: whether the platform's bitstream has the row-ops unit, and its FOLD op.
ROWOPS_UNIT: bool = _CFG["ROWOPS_UNIT"]
ROWOPS_FOLD: bool = _CFG["ROWOPS_FOLD"]

__all__ = ("ROWOPS_UNIT", "ROWOPS_FOLD", "RowopsHwConfigError", "resolve")
