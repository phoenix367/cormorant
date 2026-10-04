"""
Kernel call signatures — the register values that set a kernel call's run
time (doc/plans/TACTICS_PLAN.md §3).

Every node that drives a hardware kernel describes the calls it issues with
``kernel_calls(layouts)`` next to its ``emit_call()`` (nodes.py,
llm_nodes.py); ``test/test_perf_calls.py`` parses the generated C and checks
the two agree.  The performance models (``perf_models/``), the calibration
runner and the planner key everything on :meth:`KernelCall.key`.

Buffer addresses and call offsets are not part of a signature: every call
start is 16-byte aligned, and the kernels' run time depends on the geometry
registers only (the calibration campaign checks both claims).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# Register fields per kernel, in key order.  ConvKernel / PoolKernel carry
# out_h / out_w explicitly (asymmetric padding: not derivable from in_h /
# in_w); MatmulKernel's c stride is set per call.
FIELDS: Dict[str, Tuple[str, ...]] = {
    "VectorOPKernel": ("op", "size", "outer", "a_inc", "b_inc", "act"),
    "MatmulKernel":   ("n", "k", "m", "batch", "a_stride", "b_stride", "c_stride",
                       "b_packed", "gemv_kw"),
    "ConvKernel":     ("batch", "in_ch", "in_h", "in_w", "out_ch", "out_h", "out_w",
                       "kh", "kw", "stride_h", "stride_w", "dilation_h", "dilation_w",
                       "pad_top", "pad_left", "has_bias", "is_dw"),
    "PoolKernel":     ("batch", "channels", "in_h", "in_w", "out_h", "out_w",
                       "pool_h", "pool_w", "stride_h", "stride_w", "pad_top", "pad_left",
                       "dil_h", "dil_w", "pool_type", "lp_order", "count_include_pad"),
}
KERNELS = tuple(FIELDS)


@dataclass(frozen=True)
class KernelCall:
    """``count`` calls of one kernel with one set of register values."""

    kernel: str
    regs:   Tuple[int, ...]
    count:  int = 1

    def __post_init__(self):
        if self.kernel not in FIELDS:
            raise ValueError(f"unknown kernel {self.kernel!r}")
        if len(self.regs) != len(FIELDS[self.kernel]):
            raise ValueError(f"{self.kernel}: {len(self.regs)} register values, "
                             f"expected {len(FIELDS[self.kernel])}")
        if self.count < 1:
            raise ValueError("count must be >= 1")

    @classmethod
    def of(cls, kernel: str, count: int = 1, **regs) -> "KernelCall":
        f = FIELDS[kernel]
        extra = set(regs) - set(f)
        if extra:
            raise ValueError(f"{kernel}: unknown fields {sorted(extra)}")
        return cls(kernel, tuple(int(regs.get(k, 0)) for k in f), count)

    @property
    def fields(self) -> Dict[str, int]:
        return dict(zip(FIELDS[self.kernel], self.regs, strict=True))

    def key(self) -> str:
        """``"ConvKernel:1,128,96,48,..."`` — the performance models' key."""
        return f"{self.kernel}:" + ",".join(str(v) for v in self.regs)

    @staticmethod
    def from_key(key: str, count: int = 1) -> "KernelCall":
        kernel, _, vals = key.partition(":")
        return KernelCall(kernel, tuple(int(v) for v in vals.split(",")), count)


def merge(calls: Iterable[KernelCall]) -> List[KernelCall]:
    """Equal signatures summed into one entry, first-seen order."""
    out: Dict[Tuple[str, Tuple[int, ...]], int] = {}
    for c in calls:
        k = (c.kernel, c.regs)
        out[k] = out.get(k, 0) + c.count
    return [KernelCall(k, r, n) for (k, r), n in out.items()]


# ---------------------------------------------------------------------------
# Bitstream identity
# ---------------------------------------------------------------------------

BITSTREAM_ID_LEN = 12


def bitstream_id_of_bin(data: bytes) -> str:
    """The id of a flat bitstream (the .bin the board loads from
    /lib/firmware): the first 12 hex digits of its SHA-256."""
    return hashlib.sha256(data).hexdigest()[:BITSTREAM_ID_LEN]


def bitstream_id(bit_path: Path) -> str:
    """The id of a Vivado .bit, computed on the image the board runs
    (``bitstream.convert.bit_to_bin``, as upload_bitstream.py writes it)."""
    from .bitstream.convert import bit_to_bin
    return bitstream_id_of_bin(bit_to_bin(Path(bit_path)))


def _bitstream_config(config: Optional[Path]) -> Tuple[Path, dict]:
    """The ``bitstream`` section of ``bitstream_config_<platform>.json``
    (default: the KV260 one next to this package); {} when it is missing."""
    import json
    cfg = Path(config) if config else Path(__file__).resolve().parent.parent / "bitstream_config_kv260.json"
    try:
        return cfg, json.loads(cfg.read_text())["bitstream"]
    except (OSError, KeyError, ValueError):
        return cfg, {}


def local_bitstream_id(config: Optional[Path] = None) -> Optional[str]:
    """The id of the bitstream named by ``bitstream_config_<platform>.json``
    (default: the KV260 one next to this package), or None when the config
    or its .bit is missing."""
    cfg, bs = _bitstream_config(config)
    if not bs.get("bit"):
        return None
    p = (cfg.parent / bs["bit"]).resolve()
    return bitstream_id(p) if p.exists() else None


def local_board_bin(config: Optional[Path] = None) -> str:
    """Where the board keeps the flat bitstream upload_bitstream.py loads for
    that config: ``/lib/firmware/<overlay_name>.bin``, the overlay name
    defaulting to the .dtbo stem as there (``pl`` without a config)."""
    _, bs = _bitstream_config(config)
    name = bs.get("overlay_name") or (Path(bs["dtbo"]).stem if bs.get("dtbo") else "pl")
    return f"/lib/firmware/{name}.bin"


__all__ = ("FIELDS", "KERNELS", "KernelCall", "merge", "BITSTREAM_ID_LEN",
           "bitstream_id_of_bin", "bitstream_id", "local_bitstream_id", "local_board_bin")
