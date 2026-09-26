"""
Tensor metadata and code-emission helpers.

TensorInfo wraps an ONNX tensor (weight, input, intermediate, or output)
and knows how to emit:
  - a static weight array (for constants/initializers)
  - a buffer declaration (for mutable intermediate tensors)
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional
import numpy as np

if TYPE_CHECKING:
    from .dtype import DataType


def _sanitize_c_name(name: str) -> str:
    """Turn any ONNX tensor name into a valid C identifier."""
    # Replace anything that isn't alphanumeric or underscore
    s = re.sub(r"[^A-Za-z0-9_]", "_", name)
    # Identifiers must not start with a digit
    if s and s[0].isdigit():
        s = "t_" + s
    # Collapse runs of underscores for readability
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "tensor"


# Weight tensors with more elements than this are written to external .dat
# files and loaded at runtime via fread(), rather than embedded as C arrays.
# 4096 elements = 8 KB in ap_fixed<16,8>; keeps generated C files small.
LARGE_WEIGHT_THRESHOLD = 4096

# ONNX element types (as numpy dtype names, see graph._ONNX_DTYPE_MAP) that
# are integers: stored as raw integers in Data_t-sized DMA elements.
INT_DTYPE_NAMES = frozenset({
    "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64", "bool",
})


@dataclass
class TensorInfo:
    onnx_name: str
    shape:     List[int]       # [] = scalar
    dtype:     str             # 'float32', 'int8', …  (ONNX dtype string)
    data:      Optional[np.ndarray] = field(default=None, repr=False)

    # Kernel-specific packed image of `data` (float, flat), when the hardware
    # wants a layout other than the ONNX row-major one — e.g. ConvKernel's
    # tile-major weights and word-padded bias (ConvNode.from_onnx sets it).
    # `data`/`shape` stay logical for the simulator; `numel`, the ROM array,
    # the .dat file and the DMA buffer size follow the packed image.
    packed_data: Optional[np.ndarray] = field(default=None, repr=False)
    packed_note: str = ""

    # ---- Numerics beyond the element type (doc/CHAT_PLAN.md §10.5, set from
    # the model's "axi.numeric" metadata, see src/numeric.py) -------------
    # exp: power-of-two exponent f of a fixed-point DMA / host-int16 tensor
    #      (value = raw * 2^-f) — None = the element type's own (8 for
    #      ap_fixed<16,8>), else an int64 array broadcastable to `shape`
    #      (a scalar, or one exponent per last-axis channel).
    exp:       Optional[np.ndarray] = field(default=None, repr=False)
    # wexp: a constant MatMul weight encoded at the rank-1 exponent
    #      f_w[i][j] = f_out[j] + F - f_in[i] (F = the kernels' output shift);
    #      `data` then holds raw / 2^F so every existing encode / pack path
    #      emits the raw bits unchanged, and the simulator's value is
    #      data * 2^(F - wexp).
    wexp:      Optional[np.ndarray] = field(default=None, repr=False)
    # host: tensor lives in host memory, never in a DMA buffer: "f32"
    #      (float32, e.g. a transformer's residual stream), "i32" (int32
    #      token ids / positions) or "i16" (raw int16 at `exp`, e.g. a KV
    #      cache only host ops touch).  None = a DMA buffer (the default).
    host:      Optional[str] = None
    # is_state: persistent across inference_run() calls (and shared by the
    #      entries of a multi-entry project); init_data = its initial VALUE
    #      (None = zeros).  Excluded from buffer reuse.
    is_state:  bool = False
    init_data: Optional[np.ndarray] = field(default=None, repr=False)

    # ------------------------------------------------------------------ #
    # Derived properties                                                   #
    # ------------------------------------------------------------------ #

    @property
    def is_host(self) -> bool:
        """True for a host-memory tensor (``host`` set): no DMA buffer."""
        return self.host is not None

    def exp_full(self, default: int) -> np.ndarray:
        """Exponent of every element in the logical shape (int64)."""
        e = np.asarray(default if self.exp is None else self.exp, np.int64)
        return np.broadcast_to(e, tuple(self.shape) if self.shape else ())

    def exp_channels(self, default: int) -> np.ndarray:
        """Exponent per last-axis channel (length shape[-1]).  ``exp`` is a
        scalar or a vector over the last axis (src/numeric.py validates)."""
        n = int(self.shape[-1]) if self.shape else 1
        e = np.asarray(default if self.exp is None else self.exp, np.int64)
        return np.broadcast_to(e, (n,)).copy()

    @property
    def numel(self) -> int:
        if self.packed_data is not None:
            return max(int(self.packed_data.size), 1)
        n = 1
        for d in self.shape:
            n *= d
        return max(n, 1)

    @property
    def emit_data(self) -> Optional[np.ndarray]:
        """The array actually written to the ROM / .dat (packed if present)."""
        return self.packed_data if self.packed_data is not None else self.data

    @property
    def c_name(self) -> str:
        return _sanitize_c_name(self.onnx_name)

    @property
    def is_weight(self) -> bool:
        return self.data is not None

    @property
    def is_int(self) -> bool:
        """True for integer / bool ONNX tensors (token ids, masks, ...).

        Integer tensors are stored in DMA buffers as RAW signed integers of
        the element width (int16 for ap_fixed<16,8>), not in the fixed-point
        encoding; only host ops (Gather / OneHot / Cast / data movement)
        read or write them."""
        return self.dtype in INT_DTYPE_NAMES

    @property
    def is_large_weight(self) -> bool:
        """True when the weight should be stored in an external .dat file."""
        return self.is_weight and self.numel > LARGE_WEIGHT_THRESHOLD

    # ------------------------------------------------------------------ #
    # Code emission                                                        #
    # ------------------------------------------------------------------ #

    def emit_weight_decl(self, dtype: "DataType") -> str:
        """
        Emit a ROM array (prefixed _rom_) plus a DMA buffer pointer initialised
        to NULL.  inference_init() allocates the DMA buffer and copies the ROM
        data into it so the kernel can access the weights via physical addresses.

        Example:
            /* ROM data for weight 'bias' ... */
            static const uint16_t _rom_bias[64] = { 0x0100, ... };
            /* DMA buffer pointer — allocated at inference_init() */
            static inference_buf_t *bias = NULL;
        """
        if not self.is_weight:
            raise ValueError(f"Tensor '{self.onnx_name}' has no data")

        literals = dtype.encode_weight(self.emit_data)
        n        = len(literals)
        c_type   = dtype.c_array_type

        # Format 8 values per row for readability
        rows = []
        for i in range(0, n, 8):
            rows.append("    " + ", ".join(literals[i:i+8]))
        inner = ",\n".join(rows)

        packed = f"  packed: {self.packed_note}\n" if self.packed_data is not None else ""
        return (
            f"/* ROM data for weight '{self.onnx_name}'"
            f"  shape={self.shape}  dtype={self.dtype}\n{packed}"
            f" * Copied into a DMA-capable buffer at inference_init(). */\n"
            f"static const {c_type} _rom_{self.c_name}[{n}] = {{\n"
            f"{inner}\n"
            f"}};\n"
            f"/* DMA buffer pointer for '{self.onnx_name}' */\n"
            f"static inference_buf_t *{self.c_name} = NULL;"
        )

    def emit_weight_decl_strided(self,
                                 outer_count: int,
                                 aligned_chunk_size: int,
                                 dtype: "DataType") -> str:
        """
        Emit a ROM array in strided layout for a weight that is used alongside
        a strided buffer in a non-broadcast run_op() call.

        The data is split into outer_count blocks; each block occupies
        aligned_chunk_size slots in the array.  The first (numel // outer_count)
        slots contain the encoded weight values; the remaining
        (aligned_chunk_size - numel // outer_count) slots are zero-padded so
        every block starts at an INFERENCE_ALIGN_BYTES-aligned offset.

        Example for bias shape=[4,6], outer_count=4, aligned_chunk_size=8:
          [row0_e0..e5, 0x0000, 0x0000,   <- block 0: 6 data + 2 gap
           row1_e0..e5, 0x0000, 0x0000,   <- block 1
           row2_e0..e5, 0x0000, 0x0000,   <- block 2
           row3_e0..e5, 0x0000, 0x0000]   <- block 3
        """
        if not self.is_weight:
            raise ValueError(f"Tensor '{self.onnx_name}' has no data")
        if self.packed_data is not None:
            raise ValueError(
                f"Tensor '{self.onnx_name}' has a kernel-packed layout and "
                f"cannot also be emitted in strided broadcast layout")

        chunk_size = self.numel // outer_count
        gap        = aligned_chunk_size - chunk_size
        total      = outer_count * aligned_chunk_size
        c_type     = dtype.c_array_type

        literals  = dtype.encode_weight(self.data)
        zero_lit  = dtype.format_literal(0)

        padded: list = []
        for i in range(outer_count):
            padded.extend(literals[i * chunk_size : (i + 1) * chunk_size])
            padded.extend([zero_lit] * gap)

        rows = []
        for i in range(0, len(padded), 8):
            rows.append("    " + ", ".join(padded[i:i+8]))
        inner = ",\n".join(rows)

        return (
            f"/* ROM data for weight '{self.onnx_name}'"
            f"  shape={self.shape}  dtype={self.dtype}\n"
            f" * Strided layout: {outer_count} blocks × {aligned_chunk_size} elements"
            f" ({chunk_size} data + {gap} zero-pad per block).\n"
            f" * Co-input to a non-broadcast run_op() that processes a strided buffer;\n"
            f" * gap zeros ensure both operands are aligned at every block boundary. */\n"
            f"static const {c_type} _rom_{self.c_name}[{total}] = {{\n"
            f"{inner}\n"
            f"}};\n"
            f"/* DMA buffer pointer for '{self.onnx_name}' */\n"
            f"static inference_buf_t *{self.c_name} = NULL;"
        )

    def emit_large_weight_ptr_decl(self) -> str:
        """
        For large weights: emit only the DMA buffer pointer (no ROM array).
        The data is loaded at runtime from a .dat file.

        Example:
            /* External weight 'bias'  shape=[64,64,16,16]  numel=1048576
             * Loaded at inference_init() from weights/bias.dat */
            static inference_buf_t *bias = NULL;
        """
        return (
            f"/* External weight '{self.onnx_name}'"
            f"  shape={self.shape}  numel={self.numel}\n"
            f" * Loaded at inference_init() from weights/{self.c_name}.dat */\n"
            f"static inference_buf_t *{self.c_name} = NULL;"
        )

    def to_dat_bytes(self, dtype: "DataType") -> bytes:
        """
        Serialise the weight tensor to raw little-endian bytes for external
        .dat files.  Written to weights/<c_name>.dat by the scheduler and
        loaded at runtime by fread().
        """
        if not self.is_weight:
            raise ValueError(f"Tensor '{self.onnx_name}' has no data")
        return dtype.dat_bytes(self.emit_data)

    def emit_buffer_decl(self) -> str:
        """
        Emit a DMA buffer pointer for an intermediate tensor.
        Allocated at inference_init(); NULL until then.

        Example:
            static inference_buf_t *add_Y = NULL;  /* 'add_Y' shape=[1,64] */
        """
        return (
            f"static inference_buf_t *{self.c_name} = NULL;"
            f"  /* '{self.onnx_name}'  shape={self.shape} */"
        )
