"""
VectorOPKernel capabilities resolved from the platform JSON.

``activations`` says whether the platform's bitstream carries VectorOPKernel's
activation unit (doc/plans/ACTIVATIONS_PLAN.md): LeakyReLU, SiLU, GELU and
GELU tanh as ops 6-9 and acts 3-6, and the ``alpha`` register.  An IP without
it passes ops 6-9 through unchanged and ignores acts 3-6, so the scheduler
maps ONNX ``Gelu`` / ``LeakyRelu`` / ``x * Sigmoid(x)`` onto the kernel only
when it is true (a generated program also checks for the unit at
``inference_init()``).

JSON shape (only the field this module reads)::

    {
      "kernels": {
        "vectorop": {
          "activations": true
        }
      }
    }

Platform selection as for the other kernels (``_matmul_hw_config``): the
``AXI_PLATFORM`` environment variable picks ``platforms/<name>.json``,
default ``kv260``.  ``AXI_VECTOROP_ACTIVATIONS`` (``0`` / ``1``) overrides
the JSON for the scheduler — ``0`` for a project on a bitstream built before
the activation unit (``986cef4866a0`` and older).  A missing or malformed
field raises ``VectoropHwConfigError``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict

_SCHEDULER_DIR = Path(__file__).resolve().parent.parent
_PLATFORMS_DIR = _SCHEDULER_DIR.parent / "platforms"
_DEFAULT_PLATFORM = "kv260"


class VectoropHwConfigError(RuntimeError):
    """Platform JSON is missing / malformed for the vectorop kernel."""


def resolve(platform_name: str = None) -> Dict[str, object]:
    """``{"VECTOROP_ACTIVATIONS": bool}`` for the given platform (``None``:
    ``AXI_PLATFORM``, else ``kv260``), ``AXI_VECTOROP_ACTIVATIONS`` applied."""
    if platform_name is None:
        platform_name = os.environ.get("AXI_PLATFORM", _DEFAULT_PLATFORM)
    path = _PLATFORMS_DIR / f"{platform_name}.json"
    if not path.is_file():
        raise VectoropHwConfigError(
            f"Platform file not found: {path}.  Set AXI_PLATFORM to a "
            f"<name> for which platforms/<name>.json exists.")
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise VectoropHwConfigError(f"Platform JSON {path} is not valid JSON: {e}") from e
    try:
        section = data["kernels"]["vectorop"]
        act = section["activations"]
    except (KeyError, TypeError) as e:
        raise VectoropHwConfigError(
            f"Platform JSON {path}: 'kernels.vectorop.activations' is required "
            f"(true when the bitstream's VectorOPKernel has the activation unit).") from e
    if not isinstance(act, bool):
        raise VectoropHwConfigError(
            f"Platform JSON {path}: 'kernels.vectorop.activations' must be true or "
            f"false, got {act!r}.")
    env = os.environ.get("AXI_VECTOROP_ACTIVATIONS")
    if env:
        if env not in ("0", "1"):
            raise VectoropHwConfigError(
                f"AXI_VECTOROP_ACTIVATIONS must be 0 or 1, got {env!r}.")
        act = env == "1"
    return {"VECTOROP_ACTIVATIONS": act}


_CFG = resolve()

# Exported: whether the platform's VectorOPKernel has the activation unit.
VECTOROP_ACTIVATIONS: bool = _CFG["VECTOROP_ACTIVATIONS"]

__all__ = ("VECTOROP_ACTIVATIONS", "VectoropHwConfigError", "resolve")
