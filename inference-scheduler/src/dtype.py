"""
Data-type abstraction for the inference scheduler.

A DataType object captures everything the Python code generator and simulator
need to know about the element type used in DMA buffers:

  - How many bytes each element occupies and the derived AXI alignment
  - How to quantize float64 simulation values to the nearest representable value
  - How the C test harness fills ramp inputs (matching the C cast semantics)
  - How to encode float64 values back into the raw storage dtype (for expected
    arrays embedded in test_inference.c, and for external .dat weight files)
  - What C type declarations and display expressions to emit

Supported types
---------------
  AP_FIXED_16_8   ap_fixed<16,8>   — 16-bit signed fixed-point, 8 fractional bits
  FLOAT32         float            — IEEE 754 single precision

Adding a new type
-----------------
Subclass DataType and implement the abstract methods.  Then pass the instance
to OnnxGraph and CodeGenerator:

    g  = OnnxGraph("model.onnx", dtype=MY_DTYPE)
    cg = CodeGenerator(g, "model.onnx", dtype=MY_DTYPE)
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List
import numpy as np

# AXI burst-alignment requirement: every broadcast-chunk start address must be
# a multiple of this many bytes.  Hardware-fixed; does not change with dtype.
ALIGN_BYTES: int = 16


class DataType(ABC):
    """Abstract base for element data types used in inference buffers."""

    # ------------------------------------------------------------------ #
    # Properties every subclass must expose                                #
    # ------------------------------------------------------------------ #

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name, e.g. 'ap_fixed<16,8>'."""

    @property
    @abstractmethod
    def bytes_per_elem(self) -> int:
        """Bytes per element in DMA buffers."""

    @property
    @abstractmethod
    def c_type(self) -> str:
        """C typedef target, e.g. 'uint16_t' or 'float'."""

    @property
    @abstractmethod
    def c_array_type(self) -> str:
        """C type for static ROM / expected arrays, e.g. 'uint16_t' or 'float'."""

    @property
    @abstractmethod
    def np_storage(self) -> np.dtype:
        """Numpy dtype for raw DMA buffer contents."""

    @property
    def align_elems(self) -> int:
        """Number of elements per AXI alignment boundary (ALIGN_BYTES / bytes_per_elem)."""
        return ALIGN_BYTES // self.bytes_per_elem

    # ------------------------------------------------------------------ #
    # Simulation operations                                                #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def quantize(self, x: np.ndarray) -> np.ndarray:
        """
        Round float64 array x to the nearest value representable in this
        type (saturation + round-to-nearest).  Returns float64.

        Used for encoding weights and inputs: the result matches what
        float_to_storage() + decode produces, i.e. the value the hardware
        actually reads from the DMA buffer.

        For fixed-point types: clip to [min, max] and round to nearest
        fractional grid point.  For floating-point types: float32 round-trip.
        """

    def truncate(self, x: np.ndarray) -> np.ndarray:
        """
        Truncate float64 array x toward −∞ to the nearest representable
        value (saturation + floor).  Returns float64.

        Used to simulate AP_TRN quantization of MUL results.  For
        ap_fixed<16,8> × ap_fixed<16,8> HLS produces ap_fixed<32,16>; the
        AP_TRN cast back to ap_fixed<16,8> discards the lower 8 fractional
        bits with an arithmetic right-shift (= floor toward −∞).

        The default implementation delegates to quantize() (round-to-nearest),
        which is correct for types where arithmetic results carry no
        sub-representable-precision remainder (e.g. float32).  Fixed-point
        subclasses override this with floor.
        """
        return self.quantize(x)

    def truncate_div(self, x: np.ndarray) -> np.ndarray:
        """
        Truncate float64 array x toward zero to the nearest representable
        value (saturation + trunc).  Returns float64.

        Used to simulate DIV results.  HLS implements ap_fixed division as
        integer division of the raw representations (a_int / b_int), which
        in C truncates toward zero.  For positive quotients this equals
        floor; for negative quotients it equals ceiling (less negative).

        The default delegates to truncate() (floor), which is correct for
        float32 and for types where all div results are non-negative.
        Fixed-point subclasses override with np.trunc.
        """
        return self.truncate(x)

    @abstractmethod
    def ramp_to_float(self, positions: np.ndarray) -> np.ndarray:
        """
        Convert int64 DMA buffer positions to float64 values, matching
        the pattern written by the C test harness:

            p[pos] = (Data_t)(pos & mask);

        where mask depends on the storage width.  Returns float64.
        """

    @abstractmethod
    def float_to_storage(self, x: np.ndarray) -> np.ndarray:
        """
        Encode float64 values to the raw storage dtype (np_storage), applying
        saturation/rounding as needed.  Returns an array with
        dtype == self.np_storage.
        """

    # ------------------------------------------------------------------ #
    # Weight encoding                                                      #
    # ------------------------------------------------------------------ #

    def encode_weight(self, data: np.ndarray) -> List[str]:
        """
        Encode a weight array as a list of C literal strings suitable for
        embedding in a static ROM array.  The order matches data.flatten().

        Example (ap_fixed<16,8>):  ["0x0100", "0x0080", ...]
        Example (float32):         ["0.25000000f", "0.50000000f", ...]
        """
        storage = self.float_to_storage(data.flatten().astype(np.float64))
        return [self.format_literal(v) for v in storage]

    @abstractmethod
    def dat_bytes(self, data: np.ndarray) -> bytes:
        """
        Serialise a weight array to raw little-endian bytes for external
        .dat files.  Loaded at runtime via fread() into a DMA buffer.
        """

    # ------------------------------------------------------------------ #
    # Code-generation helpers                                              #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def c_display(self, ptr: str, idx: str) -> str:
        """
        C expression that evaluates to a double suitable for printf("%.4f").

        Examples:
          ap_fixed<16,8>  →  "(double)(int16_t)ptr[idx] / 256.0"
          float32         →  "(double)ptr[idx]"
        """

    @abstractmethod
    def c_fill_rhs(self, pos_expr: str) -> str:
        """
        Right-hand side of the C ramp-fill assignment:

            p[pos] = <c_fill_rhs(pos_expr)>;

        Examples:
          ap_fixed<16,8>  →  "(Data_t)(pos & 0xFFFFu)  /* pos/256.0 */"
          float32         →  "(Data_t)(pos & 0xFFFFu)"
        """

    def format_literal(self, storage_val) -> str:
        """
        Format a single storage value as a C array literal string.
        Default: hex for integer storage types.
        """
        nbytes = self.bytes_per_elem
        width  = nbytes * 2   # hex digits
        return f"0x{int(storage_val) & ((1 << nbytes * 8) - 1):0{width}X}"

    def c_typedef_comment(self) -> str:
        """One-line comment for the Data_t typedef in the generated header."""
        return f"/* {self.name} */"

    # ------------------------------------------------------------------ #
    # Host-CPU ops and integer tensors                                     #
    # ------------------------------------------------------------------ #

    def host_quantize(self, x: np.ndarray) -> np.ndarray:
        """Value a host op writes back for the double result ``x``: round
        half to even + saturate (``quantize``), NaN -> 0.  Mirrors the
        generated C ``host_st()`` (``nearbyint`` under FE_TONEAREST)."""
        x = np.asarray(x, dtype=np.float64)
        return self.quantize(np.where(np.isnan(x), 0.0, x))

    def int_quantize(self, x: np.ndarray) -> np.ndarray:
        """Integer value a host op stores for the double ``x`` into an
        integer tensor: truncate toward zero, saturate to the storage range,
        NaN -> 0.  Mirrors the generated C ``host_st_int()``."""
        raise NotImplementedError(f"{self.name}: host ops / integer tensors not supported")

    def int_to_storage(self, x: np.ndarray) -> np.ndarray:
        """Encode integer VALUES (float64 holding integers) as the raw
        storage of an integer tensor (np_storage dtype)."""
        raise NotImplementedError(f"{self.name}: host ops / integer tensors not supported")

    def c_int_display(self, ptr: str, idx: str) -> str:
        """C expression (an int) for printing element ``ptr[idx]`` of an
        integer tensor with ``%d``."""
        raise NotImplementedError(f"{self.name}: host ops / integer tensors not supported")

    def c_host_conversions(self) -> str:
        """C source of the ``host_ld`` / ``host_st`` (Data_t <-> double) and
        ``host_ld_int`` / ``host_st_int`` (raw integer element <-> double)
        helpers every host op is written in terms of."""
        raise NotImplementedError(f"{self.name}: host ops / integer tensors not supported")

    @property
    def host_lut_bits(self) -> int:
        """Bits of a Data_t element when host ops may tabulate a function of
        one element over every bit pattern (GELU, Softmax's exp), else 0."""
        return 0

    def c_host_lut_defs(self) -> str:
        """C definitions the table-based host helpers use (``host_lut_bits``
        != 0): HOST_LUT_SIZE, HOST_LUT_SCALE, ``host_sint_t``."""
        return ""


# ------------------------------------------------------------------ #
# ap_fixed<W, I>                                                      #
# ------------------------------------------------------------------ #

class ApFixed(DataType):
    """
    ap_fixed<W, I> — Xilinx HLS arbitrary-precision signed fixed-point.

    W  total bits (must be 8, 16, or 32)
    I  integer bits, including sign bit; fractional bits F = W - I

    Representable range : [-2^(I-1),  2^(I-1) - 2^(-F)]
    Quantization step   : 2^(-F) = 1 / 2^F

    Storage format in DMA buffers: W-bit two's-complement integer
    (same as C ap_fixed<W,I> in-memory layout).
    """

    _SUPPORTED_WIDTHS = {8: np.uint8, 16: np.uint16, 32: np.uint32}
    _SIGNED_NP        = {8: np.int8,  16: np.int16,  32: np.int32}
    _C_TYPES          = {8: "uint8_t", 16: "uint16_t", 32: "uint32_t"}

    def __init__(self, W: int, I: int) -> None:  # noqa: E741 (I = integer bits per ap_fixed convention)
        if W not in self._SUPPORTED_WIDTHS:
            raise ValueError(f"ApFixed: W={W} not supported; choose from {sorted(self._SUPPORTED_WIDTHS)}")
        if I < 1 or I >= W:
            raise ValueError(f"ApFixed: I={I} out of range for W={W}; need 1 <= I < W")
        self._W = W
        self._I = I
        self._F = W - I
        self._scale   = float(1 << self._F)   # 2^F
        self._min_val = -float(1 << (I - 1))
        self._max_val =  float((1 << (I - 1))) - 1.0 / self._scale
        self._mask    = (1 << W) - 1           # e.g. 0xFFFF for W=16
        self._np_uint = self._SUPPORTED_WIDTHS[W]
        self._np_int  = self._SIGNED_NP[W]

    # Properties
    @property
    def name(self) -> str:
        return f"ap_fixed<{self._W},{self._I}>"

    @property
    def bytes_per_elem(self) -> int:
        return self._W // 8

    @property
    def c_type(self) -> str:
        return self._C_TYPES[self._W]

    @property
    def c_array_type(self) -> str:
        return self._C_TYPES[self._W]

    @property
    def np_storage(self) -> np.dtype:
        return self._np_uint

    # Simulation
    def quantize(self, x: np.ndarray) -> np.ndarray:
        clipped = np.clip(x.astype(np.float64), self._min_val, self._max_val)
        return np.round(clipped * self._scale) / self._scale

    def truncate(self, x: np.ndarray) -> np.ndarray:
        # AP_TRN: floor toward −∞.  Only differs from quantize when the
        # arithmetic result lands between two representable values
        # (e.g. after a multiply that produces 16 fractional bits narrowed
        # to 8).  ADD/SUB inputs are always on-grid so floor is a no-op.
        clipped = np.clip(x.astype(np.float64), self._min_val, self._max_val)
        return np.floor(clipped * self._scale) / self._scale

    def truncate_div(self, x: np.ndarray) -> np.ndarray:
        # HLS division is a_int / b_int (C integer division = truncation
        # toward zero).  Differs from floor only for negative non-grid values.
        clipped = np.clip(x.astype(np.float64), self._min_val, self._max_val)
        return np.trunc(clipped * self._scale) / self._scale

    def ramp_to_float(self, positions: np.ndarray) -> np.ndarray:
        # p[pos] = (Data_t)(pos & mask)  →  interpret bit pattern as signed int
        uint_vals = (positions & self._mask).astype(self._np_uint)
        int_vals  = uint_vals.view(self._np_int)
        return int_vals.astype(np.float64) / self._scale

    def float_to_storage(self, x: np.ndarray) -> np.ndarray:
        # Clip before rounding so out-of-range weight values saturate correctly.
        clipped  = np.clip(x.astype(np.float64), self._min_val, self._max_val)
        int_vals = np.round(clipped * self._scale).astype(self._np_int)
        return int_vals.view(self._np_uint)

    def dat_bytes(self, data: np.ndarray) -> bytes:
        storage = self.float_to_storage(data.flatten().astype(np.float64))
        # Force little-endian for cross-platform .dat files
        return storage.astype(storage.dtype.newbyteorder('<')).tobytes()

    # Code generation
    def c_display(self, ptr: str, idx: str) -> str:
        int_t = f"int{self._W}_t"
        scale = f"{self._scale:.1f}"
        return f"(double)({int_t}){ptr}[{idx}] / {scale}"

    def c_fill_rhs(self, pos_expr: str) -> str:
        mask  = f"0x{self._mask:0{self._W // 4}X}u"
        scale = f"{self._scale:.1f}"
        return f"(Data_t)({pos_expr} & {mask})  /* {pos_expr}/{scale} */"

    def format_literal(self, storage_val: int) -> str:
        width = self._W // 4   # hex digits
        return f"0x{int(storage_val) & self._mask:0{width}X}"

    def c_typedef_comment(self) -> str:
        int_t = f"int{self._W}_t"
        scale = f"{self._scale:.1f}"
        return (
            f"/* {self.name}: {self._W}-bit two's-complement,"
            f" value = ({int_t})bits / {scale} */"
        )

    # Host ops / integer tensors: raw two's-complement integers of width W.
    @property
    def _int_lo(self) -> float:
        return -float(1 << (self._W - 1))

    @property
    def _int_hi(self) -> float:
        return float((1 << (self._W - 1)) - 1)

    def int_quantize(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        x = np.trunc(np.where(np.isnan(x), 0.0, x))
        return np.clip(x, self._int_lo, self._int_hi)

    def int_to_storage(self, x: np.ndarray) -> np.ndarray:
        v = self.int_quantize(x)
        return v.astype(self._np_int).view(self._np_uint)

    def c_int_display(self, ptr: str, idx: str) -> str:
        return f"(int)(int{self._W}_t){ptr}[{idx}]"

    @property
    def host_lut_bits(self) -> int:
        return self._W if self._W <= 16 else 0

    def c_host_lut_defs(self) -> str:
        if not self.host_lut_bits:
            return ""
        return (
            f"/* Lookup tables over every {self.name} bit pattern (value = bits / HOST_LUT_SCALE). */\n"
            f"#define HOST_LUT_SIZE   {1 << self._W}u\n"
            f"#define HOST_LUT_SCALE  {self._scale:.1f}\n"
            f"typedef int{self._W}_t host_sint_t;   /* signed view of the Data_t bits */\n"
        )

    def c_host_conversions(self) -> str:
        int_t = f"int{self._W}_t"
        scale = f"{self._scale:.1f}"
        lo, hi = f"{self._int_lo:.1f}", f"{self._int_hi:.1f}"
        return (
            f"/* Element conversions ({self.name}: value = ({int_t})bits / {scale}).\n"
            " * host_st rounds half to even (nearbyint under the default FE_TONEAREST\n"
            " * mode, identical to numpy's np.round) and saturates; NaN -> 0.\n"
            f" * Integer tensors (ids, masks) hold raw {int_t} values instead. */\n"
            f"static inline double host_ld(Data_t b) {{ return (double)({int_t})b / {scale}; }}\n"
            "static inline Data_t host_st(double v)\n"
            "{\n"
            "    double r;\n"
            "    if (v != v) return (Data_t)0u;\n"
            f"    r = nearbyint(v * {scale});\n"
            f"    if (r > {hi}) r = {hi};\n"
            f"    if (r < {lo}) r = {lo};\n"
            f"    return (Data_t)({int_t})r;\n"
            "}\n"
            f"static inline double host_ld_int(Data_t b) {{ return (double)({int_t})b; }}\n"
            "static inline Data_t host_st_int(double v)\n"
            "{\n"
            "    if (v != v) return (Data_t)0u;\n"
            "    v = trunc(v);\n"
            f"    if (v > {hi}) v = {hi};\n"
            f"    if (v < {lo}) v = {lo};\n"
            f"    return (Data_t)({int_t})v;\n"
            "}\n"
        )


# ------------------------------------------------------------------ #
# float32                                                             #
# ------------------------------------------------------------------ #

class Float32(DataType):
    """
    IEEE 754 single-precision floating-point.

    The test ramp fill uses the same (Data_t)(pos & 0xFFFFu) pattern as for
    ap_fixed, but the C cast means the integer value 0..65535 is converted
    to the float 0.0..65535.0 (not a fractional encoding).  The quantize
    operation is a float32 round-trip (to match hardware float32 precision).
    """

    @property
    def name(self) -> str:
        return "float32"

    @property
    def bytes_per_elem(self) -> int:
        return 4

    @property
    def c_type(self) -> str:
        return "float"

    @property
    def c_array_type(self) -> str:
        return "float"

    @property
    def np_storage(self) -> np.dtype:
        return np.float32

    def quantize(self, x: np.ndarray) -> np.ndarray:
        # Round-trip through float32 to match hardware precision
        return x.astype(np.float32).astype(np.float64)

    def ramp_to_float(self, positions: np.ndarray) -> np.ndarray:
        # (float)(pos & 0xFFFFu) → small non-negative integer
        return (positions & 0xFFFF).astype(np.float64)

    def float_to_storage(self, x: np.ndarray) -> np.ndarray:
        return x.astype(np.float32)

    def dat_bytes(self, data: np.ndarray) -> bytes:
        storage = self.float_to_storage(data.flatten().astype(np.float64))
        return storage.astype('<f4').tobytes()

    def c_display(self, ptr: str, idx: str) -> str:
        return f"(double){ptr}[{idx}]"

    def c_fill_rhs(self, pos_expr: str) -> str:
        return f"(Data_t)({pos_expr} & 0xFFFFu)"

    def format_literal(self, storage_val: float) -> str:
        return f"{float(storage_val):.8g}f"

    def c_typedef_comment(self) -> str:
        return "/* float32: IEEE 754 single precision */"

    # Host ops / integer tensors: integers are stored as float values.
    def int_quantize(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return np.trunc(np.where(np.isnan(x), 0.0, x)).astype(np.float32).astype(np.float64)

    def int_to_storage(self, x: np.ndarray) -> np.ndarray:
        return self.int_quantize(x).astype(np.float32)

    def c_int_display(self, ptr: str, idx: str) -> str:
        return f"(int){ptr}[{idx}]"

    def c_host_conversions(self) -> str:
        return (
            "/* Element conversions (float32).  Integer tensors hold integer values. */\n"
            "static inline double host_ld(Data_t b) { return (double)b; }\n"
            "static inline Data_t host_st(double v) { return (v != v) ? 0.0f : (Data_t)v; }\n"
            "static inline double host_ld_int(Data_t b) { return (double)b; }\n"
            "static inline Data_t host_st_int(double v) { return (v != v) ? 0.0f : (Data_t)trunc(v); }\n"
        )


# ------------------------------------------------------------------ #
# Pre-built singletons                                                #
# ------------------------------------------------------------------ #

AP_FIXED_16_8: DataType = ApFixed(16, 8)
"""Default element type — ap_fixed<16,8>, 2 bytes, scale 256."""

FLOAT32: DataType = Float32()
"""IEEE 754 single precision, 4 bytes."""
