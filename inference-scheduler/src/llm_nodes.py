"""
Host-CPU ops of a Llama-family decoder (doc/plans/CHAT_PLAN.md §3.2 B2, §10.5).

ONNX nodes of the custom domain ``axi.llm`` (emitted by src/llama.py), run on
the A53 inside inference_run() like the other host ops (``HostNode``: no
lane, one synchronous ``('cpu', idx)`` event, liveness starting and ending at
it).  They are the float regions of the numeric policy pow2+sink+p12+xattn of
demo/chat/scripts/llm_study.py, whose ``Model`` is the specification:

  LlmEmbed     ids (i32 host) -> h (f32 host) = table[id] (a host-memory
               bf16 / f32 table, not a DMA weight)
  LlmResAdd    h (f32), d (int16 at its exponents) -> float32(h + d)
  LlmRMSNorm   h (f32) -> y (int16):  ss = sum h_i^2 left to right,
               r = 1 / sqrt(ss / n + eps), y_i = (h_i * r) * gamma_i
  LlmSiluMul   g, u (int16) -> a (int16) = silu(g) * u, silu from a
               65 536-entry double table per gate exponent (libm exp)
  LlmAttention q0, k0, v (int16 kernel outputs), pos, n (i32) + the KV cache
               states (i16 host, [C][KV*HD]): RoPE(k) written to the K cache,
               v re-rounded into the V cache, RoPE(q) in double, then per
               query row and head the xattn order (dot8 scores * 1/sqrt(HD),
               libm exp, left-to-right sums, P.V) -> pv (int16)
  LlmSelectRow h (f32) [T][D], n (i32) -> row n-1 (f32 [1][D], a state)
  LlmDequant   x (int16) -> float32(raw * 2^-f)

FPGA prefill attention (policy pow2+sink+p12+mix, doc/plans/CHAT_PLAN.md §16): per
layer one host op, then per KV group g two ConvKernel calls and a host
softmax, then a host merge — the K / V caches are DMA states in the CMA pool
(group-major [KV][C][HD]) that the host writes and the kernels read:

  LlmAttnPrep    q0, k0, v, pos, n, caches -> qx: RoPE(k0) / v written into
                 the cache rows pos .. pos+n-1 (as LlmAttention), the cache
                 rows [0, keys) flushed for the kernels; RoPE(q0) rounded
                 at the per-head q exponents, written as the q.K^T conv
                 input image of every group (raw int16)
  LlmAttnScores  (ConvKernel) s_g[j][p] = floor(sum_d K_g[j][d] q_g[d][p] / 2^8)
                 over the keys j < keys = roundup(pos + n, Q) — a RUNTIME
                 dimension (out_ch = keys from pos / n at run time)
  LlmAttnSoftmax s_g -> P_g[p][j] (raw at 2^-f_p): per query column p =
                 (head, row t < n), keys j <= pos + t: k = raw_max - raw,
                 e = exp(-k * 2^-f_s * scale) (a 65 536-entry table per score
                 exponent), sum left to right, p = e / sum, round half even;
                 masked keys and rows t >= n get 0; row stride keys
  LlmAttnPV      (ConvKernel) o_g[p][d] = floor(sum_j P_g[p][j] V_g[j][d] / 2^8),
                 in_ch = keys / K (runtime), 1 x K over the V cache stored as
                 the conv's x image (TensorInfo.group_kw = K)
The key quantum Q (attribute key_quantum, 16 * K) keeps every row of P / K /
V the kernels read a whole input-channel tile.
  LlmAttnMerge   o_0 .. o_{KV-1} -> pv [T][H*HD] (raw copy into head order)

Numeric contract (C and ``reference()``, which the simulator runs): an
int16 element is read as ``(double)raw * 2^-f[c]`` (exact), every result is
written back with ``nearbyint(v * 2^f[c])`` (round half to even, the default
FE_TONEAREST mode == np.round) saturated to int16, NaN -> 0; float32 host
tensors hold float32 values; all arithmetic is IEEE double without FMA
contraction (the host section is compiled with fp-contract off), sums run
left to right; exp comes from libm (the simulator calls Python's math
module = the platform glibc, like host_nodes.libm).
"""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, field
from typing import ClassVar, Dict, List, Optional, Tuple

import numpy as np

from .host_nodes import (HostContext, HostNode, _attrs, _c_double, _c_float, _label,
                         _prod, _resolve, libm)
from .nodes import SchedulerError

LLM_DOMAIN = "axi.llm"

# Host tables above this many bytes are written to weights/<name>.dat and
# loaded into malloc'd memory at init; smaller ones are C arrays.
TABLE_FILE_BYTES = 64 * 1024

# The FPGA prefill attention runs over keys = roundup(pos + n, Q) keys, Q =
# 16 * K (the V cache's interleave, TensorInfo.group_kw): every row of P / K
# / V the kernels read is a whole ConvKernel input-channel tile of the 1 x K
# P.V conv, no pad lanes.  KEY_QUANTUM is the tile (K = 1).
KEY_QUANTUM = 16


# ------------------------------------------------------------------ #
# Runtime items: file-scope declarations with an init / free statement #
# ------------------------------------------------------------------ #

@dataclass(frozen=True)
class RuntimeItem:
    """A file-scope object a host op needs at run time, created in
    host_runtime_init() and released in host_runtime_deinit().  Items with
    the same ``key`` are emitted once (their text must then be equal).

    ``group`` / ``row``: instead of per-item init / free statements, the item
    is one row of a descriptor table of that group (RUNTIME_GROUPS) set up by
    one loop — hundreds of straight-line calls taking addresses of statics
    make the compiler's register allocator explode (2.6 GB, aarch64 gcc)."""
    key:   str
    decl:  str
    init:  str = ""
    free:  str = ""
    group: str = ""
    row:   str = ""


# group -> (row struct, table name, init loop body, free loop body); in the
# loop bodies T is the row pointer
RUNTIME_GROUPS = {
    "scale": ("typedef struct {\n    double           **s, **inv;   /* 2^-f, 2^f per channel */\n"
              "    const signed char *e;\n    unsigned           n;\n} llm_scale_slot_t;",
              "_llm_scale_slots", "llm_scale_slot_t",
              "if (llm_scales(T->s, T->inv, T->e, T->n) != 0) return -1;",
              "free(*T->s); free(*T->inv); *T->s = NULL; *T->inv = NULL;"),
    "silu":  ("typedef struct {\n    int f;                          /* a gate exponent */\n"
              "} llm_silu_slot_t;",
              "_llm_silu_slots", "llm_silu_slot_t",
              "if (llm_silu_table(T->f) != 0) return -1;",
              "llm_silu_free(T->f);"),
    "attn":  ("typedef struct {\n    unsigned n;                     /* cache values of one layer */\n"
              "} llm_attn_slot_t;",
              "_llm_attn_slots", "llm_attn_slot_t",
              "if (llm_attn_reserve(T->n) != 0) return -1;",
              "(void)T; llm_attn_release();"),
    "sexp":  ("typedef struct {\n    int    f;                       /* a score exponent */\n"
              "    double scale;                   /* 1 / sqrt(head_dim) */\n"
              "} llm_sexp_slot_t;",
              "_llm_sexp_slots", "llm_sexp_slot_t",
              "if (llm_sexp_table(T->f, T->scale) != 0) return -1;",
              "llm_sexp_free(T->f);"),
}


@dataclass
class HostTable:
    """A constant table in host memory (never a DMA buffer)."""
    name:  str               # C identifier (and weights/<name>.dat)
    kind:  str               # "bf16" | "f32"
    data:  np.ndarray        # the values (float64 / float32), flat

    @property
    def c_type(self) -> str:
        return "uint16_t" if self.kind == "bf16" else "float"

    @property
    def storage(self) -> np.ndarray:
        v = np.asarray(self.data, np.float32).reshape(-1)
        if self.kind == "bf16":
            return (v.view(np.uint32) >> np.uint32(16)).astype(np.uint16)
        return v

    @property
    def nbytes(self) -> int:
        return int(self.storage.nbytes)

    @property
    def is_file(self) -> bool:
        return self.nbytes > TABLE_FILE_BYTES

    def dat_bytes(self) -> bytes:
        s = self.storage
        return s.astype(s.dtype.newbyteorder("<")).tobytes()

    def runtime_item(self) -> RuntimeItem:
        n = int(self.storage.size)
        if self.is_file:
            return RuntimeItem(
                key=f"table:{self.name}",
                decl=(f"/* host table '{self.name}' ({self.kind}, {n} values, "
                      f"weights/{self.name}.dat) */\n"
                      f"static {self.c_type} *{self.name} = NULL;"),
                init=(f"    if (llm_load_table((void **)&{self.name}, \"{self.name}\", "
                      f"{n}u * sizeof({self.c_type})) != 0) return -1;"),
                free=f"    free({self.name}); {self.name} = NULL;")
        s = self.storage
        if self.kind == "bf16":
            lits = [f"0x{int(v):04X}" for v in s]
        else:
            lits = [_c_float(float(v)) for v in s]
        rows = ",\n".join("    " + ", ".join(lits[i:i + 8]) for i in range(0, len(lits), 8))
        return RuntimeItem(
            key=f"table:{self.name}",
            decl=(f"/* host table '{self.name}' ({self.kind}, {n} values) */\n"
                  f"static const {self.c_type} {self.name}[{n}] = {{\n{rows}\n}};"))


def table_kind(values: np.ndarray) -> str:
    """bf16 when every value is exactly a bfloat16 (a bf16 checkpoint's
    embedding), else f32."""
    v = np.asarray(values, np.float32).reshape(-1)
    return "bf16" if not (v.view(np.uint32) & np.uint32(0xFFFF)).any() else "f32"


def scale_item(exps: np.ndarray) -> Tuple[str, str, RuntimeItem]:
    """(scale name, inverse-scale name, runtime item) for a per-channel
    exponent vector: two double arrays 2^-f and 2^f built at init with ldexp
    (exact) from an int8 exponent array ``_llm_e_<tag>`` in the source."""
    e = np.asarray(exps, np.int64).reshape(-1)
    if (e < -127).any() or (e > 127).any():
        raise SchedulerError("exponent out of the int8 range")
    tag = exp_tag(e)
    s, i = f"_llm_s_{tag}", f"_llm_i_{tag}"
    lits = [str(int(v)) for v in e]
    rows = ",\n".join("    " + ", ".join(lits[k:k + 24]) for k in range(0, len(lits), 24))
    decl = (f"static const signed char _llm_e_{tag}[{e.size}] = {{\n{rows}\n}};\n"
            f"static double *{s} = NULL, *{i} = NULL;   /* 2^-f, 2^f per channel */")
    return s, i, RuntimeItem(key=f"scale:{tag}", decl=decl, group="scale",
                             row=f"{{ &{s}, &{i}, _llm_e_{tag}, {e.size}u }}")


def exp_tag(e: np.ndarray) -> str:
    e = np.asarray(e, np.int64).reshape(-1)
    return f"{zlib.crc32(e.astype(np.int8).tobytes()):08x}_{e.size}"


SILU_EMIN, SILU_NE = -32, 80          # _llm_silu_tab[f - SILU_EMIN] for f in [-32, 48)
SEXP_EMIN, SEXP_NE = -32, 80          # _llm_sexp_tab[f - SEXP_EMIN]: softmax exp per score exponent


def sexp_item(f: int, scale: float) -> RuntimeItem:
    """The softmax exp table of score exponent f (LlmAttnSoftmax)."""
    if not SEXP_EMIN <= int(f) < SEXP_EMIN + SEXP_NE:
        raise SchedulerError(f"score exponent {f} outside [{SEXP_EMIN}, {SEXP_EMIN + SEXP_NE})")
    return RuntimeItem(key=f"sexp:{int(f)}", decl="", group="sexp",
                       row=f"{{ {int(f)}, {_c_double(scale)} }}")


def silu_item(f: int) -> RuntimeItem:
    if not SILU_EMIN <= int(f) < SILU_EMIN + SILU_NE:
        raise SchedulerError(f"gate exponent {f} outside [{SILU_EMIN}, {SILU_EMIN + SILU_NE})")
    return RuntimeItem(key=f"silu:{f}", decl="", group="silu", row=f"{{ {int(f)} }}")


# ------------------------------------------------------------------ #
# Numerics shared by reference() implementations                       #
# ------------------------------------------------------------------ #

def _st(dtype, v, f):
    """Host write-back at exponent f (round half even, saturate, NaN -> 0)."""
    return dtype.quantize_exp(v, f)


def _f32(v):
    return np.asarray(v, np.float64).astype(np.float32).astype(np.float64)


def dot8(prod: np.ndarray) -> np.ndarray:
    """Row sums of prod [n, HD] in the xattn order (see module docstring of
    demo/chat/scripts/llm_study.py): 8 lanes acc_l = sum_a prod[:, 8a + l]
    (a ascending, from the first product), combined
    ((acc0 + acc1) + (acc2 + acc3)) + ((acc4 + acc5) + (acc6 + acc7))."""
    n, hd = prod.shape
    acc = np.cumsum(prod.reshape(n, hd // 8, 8), axis=1)[:, -1, :]
    return (((acc[:, 0] + acc[:, 1]) + (acc[:, 2] + acc[:, 3]))
            + ((acc[:, 4] + acc[:, 5]) + (acc[:, 6] + acc[:, 7])))


def rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """RoPE rotate-half of x [..., HD] with the float32 tables' rows cos / sin
    [HD/2] (as double): y = x*c + rot*s, rot = [-x[half:], x[:half]]."""
    half = x.shape[-1] // 2
    c = np.concatenate([cos, cos]).astype(np.float64)
    s = np.concatenate([sin, sin]).astype(np.float64)
    rot = np.concatenate([-x[..., half:], x[..., :half]], axis=-1)
    return x * c + rot * s


def silu_table(f: int) -> np.ndarray:
    """silu(g) = g / (1 + exp(-g)) for every raw int16 r, g = (r + 0.0) / 2^f
    (index r + 32768; 0 for g <= -700), libm exp — llm_study.silu_table."""
    t = np.empty(65536)
    d = 2.0 ** f
    for i in range(65536):
        g = (i - 32768 + 0.0) / d
        t[i] = g / (1.0 + math.exp(-g)) if g > -700 else 0.0
    return t


_SILU_CACHE: Dict[int, np.ndarray] = {}
_SEXP_CACHE: Dict[Tuple[int, float], np.ndarray] = {}


def sexp_table(f: int, scale: float) -> np.ndarray:
    """e[k] = exp(-k * 2^-f * scale) for k = raw_max - raw in [0, 65535]
    (libm exp, the operation order of llm_study.exp_table and the C fill)."""
    key = (int(f), float(scale))
    if key not in _SEXP_CACHE:
        d = 2.0 ** int(f)
        _SEXP_CACHE[key] = np.array([math.exp(-k / d * scale) for k in range(65536)])
    return _SEXP_CACHE[key]


def _silu(f: int) -> np.ndarray:
    if f not in _SILU_CACHE:
        _SILU_CACHE[f] = silu_table(f)
    return _SILU_CACHE[f]


# ------------------------------------------------------------------ #
# Base class                                                           #
# ------------------------------------------------------------------ #

@dataclass
class LlmNode(HostNode):
    """Common part of the ``axi.llm`` host ops.  ``inputs`` are every
    runtime tensor the node reads — DMA buffers, host tensors and the
    states it updates in place; constant parameters (gamma, RoPE / embedding
    tables) are baked into the node (``c_runtime`` / ``tables``)."""
    helpers: ClassVar[Tuple[str, ...]] = ("llm",)
    F:       int = 8                              # kernel output shift

    def dma_inputs(self) -> List:
        """DMA inputs read through host_in (a DMA state is passed as a
        pointer into its buffer instead)."""
        return [t for t in self.inputs if not t.is_host and not t.is_state]

    def staged_inputs(self):
        return self.dma_inputs()

    def state_inputs(self) -> List:
        return [t for t in self.inputs if t.is_state]

    def state_writes(self) -> List:
        """DMA states this op writes in place (the coherency audit marks them
        CPU-dirty: a kernel may read them only after a flush)."""
        return []

    def tables(self) -> List[HostTable]:
        return []

    def c_runtime(self) -> List[RuntimeItem]:
        """Scale arrays, tables, ... this node needs (deduplicated by key)."""
        return [tb.runtime_item() for tb in self.tables()]

    def _scales(self, t) -> Tuple[str, str, RuntimeItem]:
        return scale_item(t.exp_channels(self.F))

    def _want(self, t, kind: Optional[str], what: str):
        """Check that tensor t is stored as ``kind`` (None = DMA int16)."""
        if t.host != kind:
            raise SchedulerError(
                f"{self.onnx_node.op_type} node '{_label(self.onnx_node)}': {what} "
                f"'{t.onnx_name}' must be {'a DMA tensor' if kind is None else kind + ' host'}"
                f" (is {'DMA' if t.host is None else t.host})")

    def describe(self) -> str:
        return ""


def _llm_inputs(node, tensors):
    return [_resolve(tensors, n, node) if n else None for n in node.input]


def _require(cond, node, msg):
    if not cond:
        raise SchedulerError(f"{node.op_type} node '{_label(node)}': {msg}")


def _const_array(ctx: HostContext, tensors, name, node, what) -> np.ndarray:
    if name in ctx.consts:
        return np.asarray(ctx.consts[name])
    t = tensors.get(name)
    _require(t is not None and t.data is not None, node, f"{what} ('{name}') must be a constant")
    return np.asarray(t.data)


# ------------------------------------------------------------------ #
# LlmEmbed                                                             #
# ------------------------------------------------------------------ #

@dataclass
class LlmEmbedNode(LlmNode):
    """h[t] = table[ids[t]] (float32 host rows); the index is clamped to
    [0, rows) like Gather."""
    rows:  int = 1
    d:     int = 1
    n:     int = 1
    table: Optional[HostTable] = field(default=None, repr=False)

    @classmethod
    def from_onnx_node(cls, node, tensors, index, align_elems, ctx: HostContext):
        ids, _tab = _llm_inputs(node, tensors)
        y = _resolve(tensors, node.output[0], node)
        tab = _const_array(ctx, tensors, node.input[1], node, "table")
        _require(tab.ndim == 2, node, "table must be 2-D [rows][d]")
        rows, d = tab.shape
        name = "_llm_tab_" + tensors[node.input[1]].c_name
        sn = cls(onnx_node=node, inputs=[ids], output=y, index=index,
                 align_elems=align_elems, rows=int(rows), d=int(d), n=ids.numel,
                 table=HostTable(name, table_kind(tab), np.asarray(tab, np.float32)),
                 F=ctx.frac_bits)
        sn._want(ids, "i32", "ids")
        sn._want(y, "f32", "output")
        _require(y.numel == ids.numel * d, node, "output numel")
        return sn

    def tables(self):
        return [self.table]

    def describe(self):
        return f"{self.n} rows of {self.d} from a [{self.rows}] {self.table.kind} host table"

    def c_call(self, ins, out, scratch, direct, dtype):
        bf = self.table.kind == "bf16"
        return [f"llm_embed({ins[0]}, {self.n}u, {self.table.name if bf else 'NULL'}, "
                f"{'NULL' if bf else self.table.name}, {self.rows}u, {self.d}u, {out});"]

    def reference(self, ins, dtype):
        k = np.asarray(ins[0], np.float64).reshape(-1).astype(np.int64)
        k = np.clip(k, 0, self.rows - 1)
        tab = np.asarray(self.table.data, np.float32).reshape(self.rows, self.d)
        return tab[k].astype(np.float64).reshape(self.output.shape)


# ------------------------------------------------------------------ #
# LlmResAdd                                                            #
# ------------------------------------------------------------------ #

@dataclass
class LlmResAddNode(LlmNode):
    """y = float32((double)h + d) — the float residual stream's add."""
    rows: int = 1
    n:    int = 1

    @classmethod
    def from_onnx_node(cls, node, tensors, index, align_elems, ctx: HostContext):
        h, d = _llm_inputs(node, tensors)
        y = _resolve(tensors, node.output[0], node)
        n = int(h.shape[-1])
        sn = cls(onnx_node=node, inputs=[h, d], output=y, index=index,
                 align_elems=align_elems, rows=h.numel // n, n=n, F=ctx.frac_bits)
        sn._want(h, "f32", "h")
        sn._want(d, None, "delta")
        sn._want(y, "f32", "output")
        _require(d.numel == h.numel == y.numel and d.shape[-1] == n, node, "shapes")
        return sn

    def c_runtime(self):
        return [self._scales(self.inputs[1])[2]]

    def describe(self):
        return f"rows={self.rows} n={self.n} (float residual)"

    def c_call(self, ins, out, scratch, direct, dtype):
        s, _i, _ = self._scales(self.inputs[1])
        return [f"llm_resadd({ins[0]}, {ins[1]}, {s}, {self.rows}u, {self.n}u, {out});"]

    def reference(self, ins, dtype):
        h = np.asarray(ins[0], np.float64)
        d = np.asarray(ins[1], np.float64)
        return _f32(h + d).reshape(self.output.shape)


# ------------------------------------------------------------------ #
# LlmRMSNorm                                                           #
# ------------------------------------------------------------------ #

@dataclass
class LlmRMSNormNode(LlmNode):
    """y = (h * r) * gamma, r = 1 / sqrt(sum(h^2) / n + eps) per row;
    gamma float32 (a host-side constant)."""
    rows:  int = 1
    n:     int = 1
    eps:   float = 1e-5
    gamma: Optional[np.ndarray] = field(default=None, repr=False)

    @classmethod
    def from_onnx_node(cls, node, tensors, index, align_elems, ctx: HostContext):
        h = _resolve(tensors, node.input[0], node)
        y = _resolve(tensors, node.output[0], node)
        n = int(h.shape[-1])
        g = _const_array(ctx, tensors, node.input[1], node, "gamma").astype(np.float32).reshape(-1)
        _require(g.size == n, node, "gamma size")
        a = _attrs(node)
        eps = float(a["axi_eps"].decode() if isinstance(a.get("axi_eps"), bytes)
                    else a.get("axi_eps", a.get("epsilon", 1e-5)))
        sn = cls(onnx_node=node, inputs=[h], output=y, index=index, align_elems=align_elems,
                 rows=h.numel // n, n=n, eps=eps, gamma=g.copy(),
                 F=ctx.frac_bits)
        sn._want(h, "f32", "h")
        sn._want(y, None, "output")
        _require(y.numel == h.numel, node, "shapes")
        return sn

    @property
    def gamma_name(self) -> str:
        return f"_llm_g_{zlib.crc32(self.gamma.tobytes()):08x}_{self.gamma.size}"

    def c_runtime(self):
        lits = [_c_float(float(v)) for v in self.gamma]
        rows = ",\n".join("    " + ", ".join(lits[i:i + 6]) for i in range(0, len(lits), 6))
        return [RuntimeItem(key=f"gamma:{self.gamma_name}",
                            decl=(f"static const float {self.gamma_name}[{self.gamma.size}] = "
                                  f"{{\n{rows}\n}};")),
                self._scales(self.output)[2]]

    def describe(self):
        return f"rows={self.rows} n={self.n} eps={self.eps:.3g}"

    def c_call(self, ins, out, scratch, direct, dtype):
        _s, i, _ = self._scales(self.output)
        return [f"llm_rmsnorm({ins[0]}, {self.rows}u, {self.n}u, {self.gamma_name}, "
                f"{_c_double(self.eps)}, {i}, {out});"]

    def reference(self, ins, dtype):
        h = np.asarray(ins[0], np.float64).reshape(self.rows, self.n)
        ss = np.cumsum(h * h, axis=-1)[:, -1]
        r = 1.0 / np.sqrt(ss / float(self.n) + self.eps)
        y = (h * r[:, None]) * self.gamma.astype(np.float64)[None, :]
        return _st(dtype, y, self.output.exp_channels(self.F)[None, :]).reshape(self.output.shape)


# ------------------------------------------------------------------ #
# LlmSiluMul                                                           #
# ------------------------------------------------------------------ #

@dataclass
class LlmSiluMulNode(LlmNode):
    """a = silu(g) * u with silu from a table over g's raw int16 per gate
    exponent (llm_study: SiLU*up)."""
    rows: int = 1
    n:    int = 1

    @classmethod
    def from_onnx_node(cls, node, tensors, index, align_elems, ctx: HostContext):
        g, u = _llm_inputs(node, tensors)
        y = _resolve(tensors, node.output[0], node)
        n = int(g.shape[-1])
        sn = cls(onnx_node=node, inputs=[g, u], output=y, index=index,
                 align_elems=align_elems, rows=g.numel // n, n=n,
                 F=ctx.frac_bits)
        for t, w in ((g, "gate"), (u, "up"), (y, "output")):
            sn._want(t, None, w)
        _require(g.shape == u.shape and y.numel == g.numel, node, "shapes")
        return sn

    def c_runtime(self):
        fg = self.inputs[0].exp_channels(self.F)
        items = [silu_item(int(f)) for f in sorted(set(int(v) for v in fg))]
        items.append(self._scales(self.inputs[0])[2])        # _llm_e_<tag> of the gate
        items.append(self._scales(self.inputs[1])[2])
        items.append(self._scales(self.output)[2])
        return items

    def describe(self):
        fg = sorted(set(int(v) for v in self.inputs[0].exp_channels(self.F)))
        return f"rows={self.rows} n={self.n} silu tables for gate exponents {fg}"

    def c_call(self, ins, out, scratch, direct, dtype):
        su = self._scales(self.inputs[1])[0]
        ia = self._scales(self.output)[1]
        eg = "_llm_e_" + exp_tag(self.inputs[0].exp_channels(self.F))
        return [f"llm_silu_mul({ins[0]}, {ins[1]}, {eg}, {su}, {ia}, "
                f"{self.rows}u, {self.n}u, {out});"]

    def reference(self, ins, dtype):
        g = np.asarray(ins[0], np.float64).reshape(self.rows, self.n)
        u = np.asarray(ins[1], np.float64).reshape(self.rows, self.n)
        fg = self.inputs[0].exp_channels(self.F)
        sil = np.empty_like(g)
        for f in np.unique(fg):
            cols = fg == f
            idx = np.floor(g[:, cols] * 2.0 ** int(f)).astype(np.int64) + 32768
            sil[:, cols] = _silu(int(f))[idx]
        return _st(dtype, sil * u, self.output.exp_channels(self.F)[None, :]).reshape(
            self.output.shape)


# ------------------------------------------------------------------ #
# LlmAttention (xattn)                                                 #
# ------------------------------------------------------------------ #

@dataclass
class LlmAttentionNode(LlmNode):
    """Causal GQA attention over the KV cache, one float host region
    (policy xattn).  inputs = [q0, k0, v, pos, n?, cache_k, cache_v]; the
    caches (logical [C][KV*HD] raw int16 states stored group-major
    [KV][C][HD]: i16 host states, or DMA states in the pool when the FPGA
    prefill attention reads them) receive the new rows."""
    T:        int = 1
    H:        int = 1
    KV:       int = 1
    HD:       int = 8
    C:        int = 1
    has_n:    bool = False
    scale:    float = 1.0
    cos:      Optional[np.ndarray] = field(default=None, repr=False)   # float32 [C][HD/2]
    sin:      Optional[np.ndarray] = field(default=None, repr=False)

    @classmethod
    def from_onnx_node(cls, node, tensors, index, align_elems, ctx: HostContext):
        ins = _llm_inputs(node, tensors)
        _require(len(ins) == 9, node, "inputs: q0, k0, v, pos, n (may be empty), cache_k, "
                                      "cache_v, cos, sin")
        q0, k0, v, pos, n, ck, cv = ins[:7]
        y = _resolve(tensors, node.output[0], node)
        a = _attrs(node)
        H, KV, HD = int(a["num_heads"]), int(a["num_kv_heads"]), int(a["head_dim"])
        cos = _const_array(ctx, tensors, node.input[7], node, "cos").astype(np.float32)
        sin = _const_array(ctx, tensors, node.input[8], node, "sin").astype(np.float32)
        C = int(ck.shape[0])
        _require(HD % 8 == 0 and H % KV == 0, node, "head_dim % 8 and heads % kv_heads")
        _require(HD <= 256 and int(ck.shape[0]) <= 4096, node,
                 "head_dim <= 256 and a cache of <= 4096 rows (LLM_MAX_HD / LLM_MAX_KEYS)")
        _require(cos.shape == (C, HD // 2) and sin.shape == cos.shape, node,
                 f"cos / sin must be [{C}][{HD // 2}]")
        T = q0.numel // (H * HD)
        _require(q0.numel == T * H * HD and k0.numel == T * KV * HD and v.numel == k0.numel,
                 node, "q / k / v shapes")
        _require(list(ck.shape) == [C, KV * HD] and list(cv.shape) == [C, KV * HD], node,
                 f"caches must be [{C}][{KV * HD}]")
        _require(y.numel == q0.numel, node, "output numel")
        rt = [q0, k0, v, pos] + ([n] if n is not None else []) + [ck, cv]
        sn = cls(onnx_node=node, inputs=rt, output=y, index=index, align_elems=align_elems,
                 T=T, H=H, KV=KV, HD=HD, C=C, has_n=n is not None,
                 scale=1.0 / math.sqrt(HD), cos=cos, sin=sin,
                 F=ctx.frac_bits)
        for t, w in ((q0, "q"), (k0, "k"), (v, "v"), (y, "output")):
            sn._want(t, None, w)
        sn._want(pos, "i32", "pos")
        if n is not None:
            sn._want(n, "i32", "n")
        for t, w in ((ck, "cache_k"), (cv, "cache_v")):
            _require(t.host in ("i16", None) and t.is_state, node,
                     f"{w} '{t.onnx_name}' must be a state (i16 host or DMA)")
            _require(t.group_layout == (KV, HD), node,
                     f"{w} '{t.onnx_name}' must be stored group-major [{KV}][{C}][{HD}]")
        _require(ck.group_kw == 1, node, "the K cache rows must not be interleaved")
        return sn

    def state_writes(self):
        return [t for t in (self.ck, self.cv) if not t.is_host]

    @property
    def q0(self):
        return self.inputs[0]

    @property
    def k0(self):
        return self.inputs[1]

    @property
    def v(self):
        return self.inputs[2]

    @property
    def ck(self):
        return self.inputs[-2]

    @property
    def cv(self):
        return self.inputs[-1]

    @property
    def table_prefix(self) -> str:
        tag = zlib.crc32(self.cos.tobytes() + self.sin.tobytes())
        return f"_llm_rope_{tag:08x}_{self.C}x{self.HD // 2}"

    def tables(self):
        p = self.table_prefix
        return [HostTable(p + "_cos", "f32", self.cos.reshape(-1)),
                HostTable(p + "_sin", "f32", self.sin.reshape(-1))]

    def c_runtime(self):
        items = super().c_runtime()
        for t in (self.q0, self.k0, self.v, self.ck, self.cv, self.output):
            items.append(self._scales(t)[2])
        n = self.C * self.KV * self.HD
        items.append(RuntimeItem(key=f"attn:{n}", decl="", group="attn", row=f"{{ {n}u }}"))
        return items

    def describe(self):
        return (f"T={self.T} heads {self.H}/{self.KV} head_dim {self.HD}, cache {self.C} rows"
                f"{', n valid rows' if self.has_n else ''} (xattn)")

    def c_call(self, ins, out, scratch, direct, dtype):
        s = {k: self._scales(t) for k, t in (("q", self.q0), ("k", self.k0), ("v", self.v),
                                             ("ck", self.ck), ("cv", self.cv),
                                             ("pv", self.output))}
        n_expr = f"(unsigned){ins[4]}[0]" if self.has_n else f"{self.T}u"
        p = self.table_prefix
        return [
            "{",
            "    llm_attn_t _a;",
            f"    _a.q0 = {ins[0]}; _a.k0 = {ins[1]}; _a.v = {ins[2]};",
            f"    _a.sq = {s['q'][0]}; _a.sk = {s['k'][0]}; _a.sv = {s['v'][0]};",
            f"    _a.ck = {ins[-2]}; _a.cv = {ins[-1]};",
            f"    _a.sck = {s['ck'][0]}; _a.ick = {s['ck'][1]};"
            f" _a.scv = {s['cv'][0]}; _a.icv = {s['cv'][1]};",
            f"    _a.ipv = {s['pv'][1]}; _a.pv = {out};",
            f"    _a.cos = {p}_cos; _a.sin = {p}_sin;",
            f"    _a.pos = (unsigned){ins[3]}[0]; _a.n = {n_expr}; _a.T = {self.T}u;",
            f"    _a.H = {self.H}u; _a.KV = {self.KV}u; _a.HD = {self.HD}u; _a.C = {self.C}u;"
            f" _a.VK = {self.cv.group_kw}u;",
            f"    _a.scale = {_c_double(self.scale)};",
            "    llm_attention(&_a);",
            "}",
        ]

    def reference(self, ins, dtype):
        q0 = np.asarray(ins[0], np.float64).reshape(self.T, self.H, self.HD)
        k0 = np.asarray(ins[1], np.float64).reshape(self.T, self.KV, self.HD)
        v = np.asarray(ins[2], np.float64).reshape(self.T, self.KV * self.HD)
        pos = int(np.asarray(ins[3]).reshape(-1)[0])
        n = int(np.asarray(ins[4]).reshape(-1)[0]) if self.has_n else self.T
        ck, cv = ins[-2], ins[-1]                       # state arrays, updated in place
        n = max(0, min(n, self.T, self.C - pos))
        fk = self.ck.exp_channels(self.F)
        fvc = self.cv.exp_channels(self.F)
        grp = np.arange(self.H) // (self.H // self.KV)
        out = np.zeros((self.T, self.H, self.HD))
        for t in range(n):
            p = pos + t
            kr = rope(k0[t], self.cos[p], self.sin[p]).reshape(-1)
            ck[p] = _st(dtype, kr, fk)
            cv[p] = _st(dtype, v[t], fvc)
        for t in range(n):
            p = pos + t
            qr = rope(q0[t], self.cos[p], self.sin[p])                  # [H][HD], unquantised
            K = ck[:p + 1].reshape(p + 1, self.KV, self.HD)
            V = cv[:p + 1].reshape(p + 1, self.KV, self.HD)
            for h in range(self.H):
                g = grp[h]
                s = dot8(qr[h][None, :] * K[:, g]) * self.scale
                e = libm("exp", s - s.max())
                pr = e / np.cumsum(e)[-1]
                out[t, h] = np.cumsum(pr[:, None] * V[:, g], axis=0)[-1]
        return _st(dtype, out.reshape(self.T, -1),
                   self.output.exp_channels(self.F)[None, :]).reshape(self.output.shape)


# ------------------------------------------------------------------ #
# FPGA prefill attention: LlmAttnPrep / LlmAttnConvNode (scores, P.V)  #
# / LlmAttnSoftmax / LlmAttnMerge                                      #
# ------------------------------------------------------------------ #

def attn_keys(pos: int, n: int, T: int, C: int, q: int = KEY_QUANTUM) -> Tuple[int, int]:
    """(valid rows n, keys) of an FPGA prefill-attention call: n clamped
    like LlmAttention (0 <= n <= T, pos + n <= C); the kernels run over
    keys = roundup(pos + n, q) keys (>= q, <= C; C % q == 0) — the runtime
    dimension (C: llm_keys)."""
    n = max(0, min(int(n), T, C - int(pos))) if int(pos) < C else 0
    k = -(-(int(pos) + n) // q) * q
    return n, max(q, min(k, C))


def v_row_base(C: int, HD: int, K: int, g: int, k: int) -> int:
    """Element offset of V cache row k (element d at + d * K) of KV head g in
    the group-major layout interleaved by K (TensorInfo.group_kw; C
    llm_vrow): ((k / 16K) * 16 + k % 16) * HD*K + (k / 16) % K in group g."""
    return g * C * HD + ((k // (16 * K)) * 16 + k % 16) * HD * K + (k // 16) % K


def _i32(a) -> int:
    return int(np.asarray(a).reshape(-1)[0])


def _attr_ints(a, name, node, n=None):
    v = a.get(name)
    _require(v is not None, node, f"attribute '{name}' missing")
    v = [int(x) for x in v]
    if n is not None:
        _require(len(v) == n, node, f"attribute '{name}': {n} values expected, got {len(v)}")
    return v


def _raw(t, arr, F):
    """Raw integers of a tensor's simulator values (value * 2^f per channel)."""
    return np.asarray(arr, np.float64) * np.power(2.0, t.exp_channels(F).astype(np.float64))


def _kernel_out(acc: np.ndarray, F: int) -> np.ndarray:
    """ConvKernel / MatmulKernel output of exact raw sums: the ap_fixed<32,16>
    accumulator wraps (int32), then floor(acc / 2^F) saturated to int16."""
    big = np.abs(acc) >= 2.0 ** 31
    if big.any():
        acc = np.where(big, np.mod(acc + 2.0 ** 31, 2.0 ** 32) - 2.0 ** 31, acc)
    return np.clip(np.floor(acc / float(1 << F)), -32768, 32767)


def _heads_attrs(node):
    a = _attrs(node)
    H, KV, HD = int(a["num_heads"]), int(a["num_kv_heads"]), int(a["head_dim"])
    _require(H % KV == 0 and HD % KEY_QUANTUM == 0, node, "heads % kv_heads, head_dim % 16")
    return a, H, KV, HD


@dataclass
class LlmAttnPrepNode(LlmNode):
    """FPGA prefill attention, host part 1.  inputs = [q0, k0, v, pos, n,
    cache_k, cache_v] (the caches DMA states, group-major [KV][C][HD]);
    output qx = the q.K^T conv input image of every KV group (raw int16):
    logically B_g[d][p] = round_half_even(RoPE(q0)[t][h][d] * 2^f_q[h]),
    p = h' * T + t for head h = g * G + h' (G = H / KV), rows t >= n zero;
    in memory the ConvKernel x image of a 1 x kw lowered MatMul
    (nodes.conv_lowered_b_image): x_g[c][kw*p + j] = B_g[(c/16)*16kw + j*16 + c%16][p].
    Rows t < n: RoPE(k0) and v go into the cache rows pos + t (as
    LlmAttention; the V cache interleaved by its group_kw); then the rows
    [0, keys) of both caches are flushed — the kernels read them (decode
    steps write cache rows without flushing)."""
    T:   int = 1
    H:   int = 1
    KV:  int = 1
    HD:  int = 16
    C:   int = 16
    kw:  int = 1
    Q:   int = KEY_QUANTUM
    fq:  Optional[np.ndarray] = field(default=None, repr=False)     # [H]
    cos: Optional[np.ndarray] = field(default=None, repr=False)
    sin: Optional[np.ndarray] = field(default=None, repr=False)

    @classmethod
    def from_onnx_node(cls, node, tensors, index, align_elems, ctx: HostContext):
        ins = _llm_inputs(node, tensors)
        _require(len(ins) == 9, node, "inputs: q0, k0, v, pos, n, cache_k, cache_v, cos, sin")
        q0, k0, v, pos, n, ck, cv = ins[:7]
        y = _resolve(tensors, node.output[0], node)
        a, H, KV, HD = _heads_attrs(node)
        kw = int(a.get("qk_kw", 1))
        _require(HD % (16 * kw) == 0, node, f"head_dim {HD} % (16 * qk_kw {kw})")
        cos = _const_array(ctx, tensors, node.input[7], node, "cos").astype(np.float32)
        sin = _const_array(ctx, tensors, node.input[8], node, "sin").astype(np.float32)
        C = int(ck.shape[0])
        T = q0.numel // (H * HD)
        G = H // KV
        Q = int(a.get("key_quantum", KEY_QUANTUM))
        _require(C % Q == 0 and Q % (16 * cv.group_kw) == 0, node,
                 f"cache rows {C} % key_quantum {Q}, key_quantum % 16 x the V interleave")
        _require(q0.numel == T * H * HD and k0.numel == T * KV * HD and v.numel == k0.numel,
                 node, "q / k / v shapes")
        _require(cos.shape == (C, HD // 2) and sin.shape == cos.shape, node, "cos / sin shape")
        _require(list(y.shape) == [KV, HD, G * T], node, f"qx must be [{KV}][{HD}][{G * T}]")
        fq = np.asarray(_attr_ints(a, "q_exp", node, H), np.int64)
        sn = cls(onnx_node=node, inputs=[q0, k0, v, pos, n, ck, cv], output=y, index=index,
                 align_elems=align_elems, T=T, H=H, KV=KV, HD=HD, C=C, kw=kw, Q=Q, fq=fq,
                 cos=cos, sin=sin, F=ctx.frac_bits)
        for t, w in ((q0, "q"), (k0, "k"), (v, "v"), (y, "output")):
            sn._want(t, None, w)
        sn._want(pos, "i32", "pos")
        sn._want(n, "i32", "n")
        for t, w in ((ck, "cache_k"), (cv, "cache_v")):
            sn._want(t, None, w)
            _require(t.is_state and t.group_layout == (KV, HD), node,
                     f"{w} must be a DMA state stored group-major [{KV}][{C}][{HD}]")
        _require(ck.group_kw == 1, node, "the K cache rows must not be interleaved")
        return sn

    @property
    def ck(self):
        return self.inputs[5]

    @property
    def cv(self):
        return self.inputs[6]

    def state_writes(self):
        return [self.ck, self.cv]

    @property
    def table_prefix(self) -> str:
        tag = zlib.crc32(self.cos.tobytes() + self.sin.tobytes())
        return f"_llm_rope_{tag:08x}_{self.C}x{self.HD // 2}"

    def tables(self):
        p = self.table_prefix
        return [HostTable(p + "_cos", "f32", self.cos.reshape(-1)),
                HostTable(p + "_sin", "f32", self.sin.reshape(-1))]

    def c_runtime(self):
        items = super().c_runtime()
        for t in self.inputs[:3] + [self.ck, self.cv]:
            items.append(self._scales(t)[2])
        items.append(scale_item(self.fq)[2])
        return items

    def describe(self):
        return (f"T={self.T} heads {self.H}/{self.KV} head_dim {self.HD}, cache {self.C} rows, "
                f"q.K^T input image kw={self.kw} (FPGA prefill attention)")

    def c_call(self, ins, out, scratch, direct, dtype):
        s = {k: self._scales(t) for k, t in (("q", self.inputs[0]), ("k", self.inputs[1]),
                                             ("v", self.inputs[2]), ("ck", self.ck),
                                             ("cv", self.cv))}
        iq = scale_item(self.fq)[1]
        p = self.table_prefix
        return [
            "{",
            "    llm_prep_t _a;",
            f"    _a.q0 = {ins[0]}; _a.k0 = {ins[1]}; _a.v = {ins[2]};",
            f"    _a.sq = {s['q'][0]}; _a.sk = {s['k'][0]}; _a.sv = {s['v'][0]};",
            f"    _a.ck = {ins[5]}; _a.cv = {ins[6]};",
            f"    _a.ick = {s['ck'][1]}; _a.icv = {s['cv'][1]}; _a.iq = {iq};",
            f"    _a.qx = {out}; _a.cos = {p}_cos; _a.sin = {p}_sin;",
            f"    _a.pos = (unsigned){ins[3]}[0]; _a.n = (unsigned){ins[4]}[0]; _a.T = {self.T}u;",
            f"    _a.H = {self.H}u; _a.KV = {self.KV}u; _a.HD = {self.HD}u; _a.C = {self.C}u;"
            f" _a.kw = {self.kw}u; _a.VK = {self.cv.group_kw}u; _a.Q = {self.Q}u;",
            "    llm_attn_prep(&_a);",
            f"    llm_cache_flush({self.ck.c_name}, _a.keys, {self.KV}u, {self.C}u, {self.HD}u);",
            f"    llm_cache_flush({self.cv.c_name}, _a.keys, {self.KV}u, {self.C}u, {self.HD}u);",
            "}",
        ]

    def reference(self, ins, dtype):
        H, KV, HD, T = self.H, self.KV, self.HD, self.T
        G = H // KV
        q0 = np.asarray(ins[0], np.float64).reshape(T, H, HD)
        k0 = np.asarray(ins[1], np.float64).reshape(T, KV, HD)
        v = np.asarray(ins[2], np.float64).reshape(T, KV * HD)
        pos = _i32(ins[3])
        n, _k = attn_keys(pos, _i32(ins[4]), T, self.C, self.Q)
        ck, cv = ins[5], ins[6]                         # logical state arrays, in place
        fk = self.ck.exp_channels(self.F)
        fvc = self.cv.exp_channels(self.F)
        for t in range(n):
            p = pos + t
            ck[p] = _st(dtype, rope(k0[t], self.cos[p], self.sin[p]).reshape(-1), fk)
            cv[p] = _st(dtype, v[t], fvc)
        qx = np.zeros((KV, HD, G * T))
        s = np.power(2.0, self.fq.astype(np.float64))
        for t in range(n):
            qr = rope(q0[t], self.cos[pos + t], self.sin[pos + t])          # [H][HD]
            raw = np.clip(np.round(qr * s[:, None]), -32768, 32767)
            for h in range(H):
                g, hh = divmod(h, G)
                qx[g, :, hh * T + t] = raw[h]
        return qx


@dataclass
class LlmAttnConvNode:
    """One ConvKernel call of the FPGA prefill attention (KV group ``group``)
    whose key count is a RUNTIME dimension: keys = roundup(pos + n, Q),
    computed in the run function from the entry's pos / n inputs (C
    llm_keys) and programmed into the AXI-Lite registers — out_ch of q.K^T,
    in_ch of P.V — so the work follows the real conversation length, not the
    C-row cache (buffers are sized for C).

      kind "qk":  s_g[j][p] = sum_d K_g[j][d] * B_g[d][p]   (MatMul on
                  ConvKernel, doc/plans/BERT_PLAN.md §2 2A: weight = A = the K cache
                  rows [keys][HD] of group g, x = qx_g in the 1 x kw image,
                  out_ch = keys, output p = G*T pixels = out_h x out_w)
      kind "pv":  o_g[p][d] = sum_j P_g[p][j] * V_g[j][d]   (weight = P_g
                  [G*T][keys] rows of stride keys, x = the V cache of group g
                  stored as the 1 x K image of its rows (group_kw = K),
                  in_ch = keys / K, kernel 1 x K, stride (1, K), out HD pixels;
                  K = 4 moves each weight row of a slab in one 8-beat request
                  instead of four 2-beat ones — 2.3-2.6x faster on the board)

    Both write floor(sum / 2^8) saturated (ap_fixed<32,16> accumulator), raw
    integers.  inputs = [weight, x, pos, n]; the geometry (kw, out_h, out_w)
    is fixed at codegen by the cost model."""

    kernel_name: ClassVar[str] = "ConvKernel"
    is_llm_op:   ClassVar[bool] = True

    onnx_node:   object
    inputs:      List
    output:      object
    index:       int = 0
    align_elems: int = 8
    kind:        str = "qk"
    group:       int = 0
    T:           int = 1
    H:           int = 1
    KV:          int = 1
    HD:          int = 16
    C:           int = 16
    kw:          int = 1
    out_h:       int = 1
    out_w:       int = 1
    Q:           int = KEY_QUANTUM    # the runtime key count's quantum
    F:           int = 8
    est_cycles:  Tuple = ()           # ((keys, conv cycles, MatmulKernel cycles), ...)

    # Compatibility shims (read by the layout / header passes)
    outer_count:        int  = field(default=1,    init=False)
    chunk_size:         int  = field(default=0,    init=False)
    aligned_chunk_size: int  = field(default=0,    init=False)
    a_advances:         bool = field(default=True, init=False)
    b_advances:         bool = field(default=True, init=False)
    arity:              int  = field(default=2,    init=False)

    @property
    def G(self) -> int:
        return self.H // self.KV

    @classmethod
    def from_onnx_node(cls, node, tensors, index, align_elems, ctx: HostContext):
        from ._conv_hw_config import (CONV_MAX_ACC_PERSIST_ENTRIES, CONV_MAX_IN_CH,
                                      CONV_MAX_OUT_CH)
        ins = _llm_inputs(node, tensors)
        _require(len(ins) == 4, node, "inputs: weight, x, pos, n")
        y = _resolve(tensors, node.output[0], node)
        a, H, KV, HD = _heads_attrs(node)
        kind = "qk" if node.op_type == "LlmAttnScores" else "pv"
        g = int(a["group"])
        G = H // KV
        _require(0 <= g < KV, node, f"group {g} of {KV}")
        w, x, pos, n = ins
        cache = w if kind == "qk" else x
        C = int(cache.shape[0])
        _require(cache.is_state and not cache.is_host and cache.group_layout == (KV, HD), node,
                 "the cache must be a DMA state stored group-major")
        Q = int(a.get("key_quantum", KEY_QUANTUM))
        _require(C % Q == 0 and Q % KEY_QUANTUM == 0, node, f"cache rows {C} % key_quantum {Q}")
        if kind == "qk":
            T = x.numel // (KV * HD * G)
            kw = int(a.get("qk_kw", 1))
            _require(list(x.shape) == [KV, HD, G * T], node, "qx shape")
            _require(list(y.shape) == [C, G * T], node, f"scores must be [{C}][{G * T}]")
            M = G * T
        else:
            T = w.numel // (G * C)
            kw = x.group_kw                          # the V cache's interleave
            _require(Q % (16 * kw) == 0, node, f"key_quantum {Q} % 16 x the V interleave {kw}")
            _require(list(w.shape) == [G * T, C], node, f"P must be [{G * T}][{C}]")
            _require(list(y.shape) == [G * T, HD], node, f"output must be [{G * T}][{HD}]")
            M = HD
        if kind == "qk":
            _require(HD % (16 * kw) == 0, node, f"head_dim % (16 * kw {kw})")
            _require(w.group_kw == 1, node, "the K cache rows must not be interleaved")
        # ConvKernel bounds at the largest runtime extent (keys = C)
        out_ch_max = C if kind == "qk" else G * T
        in_ch_max = HD // kw if kind == "qk" else C // kw
        _require(out_ch_max <= CONV_MAX_OUT_CH and in_ch_max <= CONV_MAX_IN_CH, node,
                 f"out_ch {out_ch_max} / in_ch {in_ch_max} beyond ConvKernel's bounds")
        m_pad = -(-out_ch_max // 16) * 16
        oh, ow, est = cls._plan(kind, T, G, HD, C, kw, M, m_pad, CONV_MAX_ACC_PERSIST_ENTRIES, Q)
        sn = cls(onnx_node=node, inputs=[w, x, pos, n], output=y, index=index,
                 align_elems=align_elems, kind=kind, group=g, T=T, H=H, KV=KV, HD=HD, C=C,
                 kw=kw, out_h=oh, out_w=ow, Q=Q, F=ctx.frac_bits, est_cycles=est)
        for t, what in ((pos, "pos"), (n, "n")):
            _require(t.host == "i32", node, f"{what} must be an i32 host tensor")
        for t in ((x, y) if kind == "qk" else (w, y)):
            _require(t.host is None, node, f"'{t.onnx_name}' must be a DMA tensor")
        return sn

    @staticmethod
    def _plan(kind, T, G, HD, C, kw, M, m_pad, max_acc, Q=KEY_QUANTUM):
        """(out_h, out_w, estimates): the output split out_h x out_w = M of the
        cheapest conv by cost_model.conv_cycles at keys = C / 2 (the geometry
        is fixed at codegen; keys varies at run time), and the conv /
        MatmulKernel estimates at T + 1 and C keys (report)."""
        from .cost_model import CALL_OVERHEAD, conv_cycles, matmul_cycles

        def cyc(keys, oh, ow):
            if kind == "qk":
                return conv_cycles(in_ch=HD // kw, out_ch=keys, in_h=oh, in_w=kw * ow,
                                   oh=oh, ow=ow, kh=1, kw=kw, sw=kw)["total"] + CALL_OVERHEAD
            return conv_cycles(in_ch=keys // kw, out_ch=G * T, in_h=oh, in_w=kw * ow, oh=oh,
                               ow=ow, kh=1, kw=kw, sw=kw)["total"] + CALL_OVERHEAD
        cands = [d for d in range(1, M + 1) if M % d == 0 and d <= 64 and d * m_pad <= max_acc]
        wide = [d for d in cands if d >= 8] or cands
        ref = max(Q, (C // 2) // Q * Q)
        ow = min(wide, key=lambda d: (cyc(ref, M // d, d), -d))
        oh = M // ow
        est = []
        for keys in sorted({max(Q, -(-(T + 1) // Q) * Q), C}):
            mm = (matmul_cycles(keys, HD, G * T) if kind == "qk"
                  else matmul_cycles(G * T, keys, HD)) + CALL_OVERHEAD
            est.append((keys, float(cyc(keys, oh, ow)), float(mm)))
        return oh, ow, tuple(est)

    # ---- geometry ------------------------------------------------------ #
    def conv_regs(self, keys: str) -> str:
        """run_conv_at() geometry arguments with the runtime key count
        ``keys`` (a C expression)."""
        if self.kind == "qk":
            return (f"1u, {self.HD // self.kw}u, {self.out_h}u, {self.kw * self.out_w}u,\n"
                    f"                    {keys}, {self.out_h}u, {self.out_w}u,\n"
                    f"                    1u, {self.kw}u, 1u, {self.kw}u, 1u, 1u, 0u, 0u, 0u, 0u")
        in_ch = keys if self.kw == 1 else f"{keys} / {self.kw}u"
        return (f"1u, {in_ch}, {self.out_h}u, {self.kw * self.out_w}u,\n"
                f"                    {self.G * self.T}u, {self.out_h}u, {self.out_w}u,\n"
                f"                    1u, {self.kw}u, 1u, {self.kw}u, 1u, 1u, 0u, 0u, 0u, 0u")

    def emit_comment(self) -> str:
        w, x = self.inputs[0].onnx_name, self.inputs[1].onnx_name
        if self.kind == "qk":
            what = (f"q.K^T group {self.group}: [keys][{self.HD}]x[{self.HD}][{self.G * self.T}]"
                    f" on ConvKernel: weight=K cache rows x=q image (kw={self.kw})"
                    f" out_ch=keys in_ch={self.HD // self.kw} 1x{self.kw}"
                    f" out {self.out_h}x{self.out_w}")
        else:
            what = (f"P.V group {self.group}: [{self.G * self.T}][keys]x[keys][{self.HD}]"
                    f" on ConvKernel: weight=P x=V cache image out_ch={self.G * self.T}"
                    f" in_ch=keys/{self.kw} 1x{self.kw} out {self.out_h}x{self.out_w}")
        return (f"    /* [{self.index}] {self.onnx_node.op_type}({w}, {x}) -> "
                f"{self.output.onnx_name}  {what}; keys = roundup(pos + n, {self.Q}) at run"
                f" time */")

    def emit_call(self, layouts: dict) -> str:  # noqa: ARG002
        w, x, pos, n = self.inputs
        GTHD = self.G * self.T * self.HD
        if self.kind == "qk":
            xo, wo = self.group * GTHD, self.group * self.C * self.HD
        else:
            xo, wo = self.group * self.C * self.HD, 0
        return "\n".join([
            "    {",
            f"        const unsigned _keys = llm_keys((unsigned){pos.c_name}[0],"
            f" (unsigned){n.c_name}[0], {self.T}u, {self.C}u, {self.Q}u);",
            f"        run_conv_at({x.c_name}, {xo}u, {w.c_name}, {wo}u, {self.output.c_name}, 0u,",
            f"                    {self.conv_regs('_keys')});",
            "    }",
        ])

    # ---- simulation ------------------------------------------------------ #
    def reference(self, ins, dtype):
        w, x = ins[0], ins[1]
        _n, keys = attn_keys(_i32(ins[2]), _i32(ins[3]), self.T, self.C, self.Q)
        g, HD, F = self.group, self.HD, self.F
        if self.kind == "qk":
            kc = _raw(self.inputs[0], w, F)[:keys, g * HD:(g + 1) * HD]     # [keys][HD]
            B = np.asarray(x, np.float64)[g]                                 # [HD][G*T] raw
            out = np.zeros((self.C, self.G * self.T))
            out[:keys] = _kernel_out(kc @ B, F)
            return out
        P = np.asarray(w, np.float64)[:, :keys]                              # [G*T][keys] raw
        vc = _raw(self.inputs[1], x, F)[:keys, g * HD:(g + 1) * HD]         # [keys][HD]
        return _kernel_out(P @ vc, F)


@dataclass
class LlmAttnSoftmaxNode(LlmNode):
    """FPGA prefill attention, host part 2 (group g): the p12 softmax of
    pow2+sink+p12 over the q.K^T output s_g [C][G*T] (raw at the score
    exponent f_s[h] = f_q[h] + f_k[g] - 8, rows j < keys valid) into P_g,
    the P.V call's weight [G*T][keys] (raw at f_p, row stride keys — a
    runtime stride; the tensor is sized [G*T][C]).  Column p = h'*T + t,
    rows t < n: keys j <= pos + t, k = raw_max - raw, e = sexp_f_s[k], sum
    left to right, P = round_half_even(e / sum * 2^f_p); other entries 0."""
    T:     int = 1
    G:     int = 1
    C:     int = 16
    Q:     int = KEY_QUANTUM
    group: int = 0
    scale: float = 1.0
    fs:    Optional[np.ndarray] = field(default=None, repr=False)   # [G]
    fp:    Optional[np.ndarray] = field(default=None, repr=False)   # [G]

    @classmethod
    def from_onnx_node(cls, node, tensors, index, align_elems, ctx: HostContext):
        ins = _llm_inputs(node, tensors)
        _require(len(ins) == 3, node, "inputs: scores, pos, n")
        s, pos, n = ins
        y = _resolve(tensors, node.output[0], node)
        a, H, KV, HD = _heads_attrs(node)
        G = H // KV
        C = int(s.shape[0])
        T = s.numel // (C * G)
        _require(list(s.shape) == [C, G * T] and list(y.shape) == [G * T, C], node,
                 f"scores [{C}][{G * T}] -> P [{G * T}][{C}]")
        Q = int(a.get("key_quantum", KEY_QUANTUM))
        _require(C % Q == 0 and Q % KEY_QUANTUM == 0, node, f"cache rows {C} % key_quantum {Q}")
        sn = cls(onnx_node=node, inputs=[s, pos, n], output=y, index=index,
                 align_elems=align_elems, T=T, G=G, C=C, Q=Q, group=int(a["group"]),
                 scale=1.0 / math.sqrt(HD),
                 fs=np.asarray(_attr_ints(a, "s_exp", node, G), np.int64),
                 fp=np.asarray(_attr_ints(a, "p_exp", node, G), np.int64), F=ctx.frac_bits)
        sn._want(s, None, "scores")
        sn._want(y, None, "output")
        sn._want(pos, "i32", "pos")
        sn._want(n, "i32", "n")
        return sn

    def c_runtime(self):
        items = [sexp_item(int(f), self.scale) for f in sorted(set(int(v) for v in self.fs))]
        items.append(scale_item(self.fp)[2])
        items.append(scale_item(self.fs)[2])          # _llm_e_<tag>: f_s per head
        return items

    def describe(self):
        return (f"group {self.group}: {self.G}x{self.T} query columns, score exponents "
                f"{[int(v) for v in self.fs]}, P at 2^-{[int(v) for v in self.fp]}")

    def c_call(self, ins, out, scratch, direct, dtype):
        ip = scale_item(self.fp)[1]
        return [
            "{",
            "    llm_smx_t _a;",
            f"    _a.s = {ins[0]}; _a.p = {out}; _a.fs = _llm_e_{exp_tag(self.fs)}; _a.ip = {ip};",
            f"    _a.pos = (unsigned){ins[1]}[0]; _a.n = (unsigned){ins[2]}[0];",
            f"    _a.T = {self.T}u; _a.G = {self.G}u; _a.C = {self.C}u; _a.Q = {self.Q}u;",
            "    llm_attn_softmax(&_a);",
            "}",
        ]

    def reference(self, ins, dtype):
        T, G, C = self.T, self.G, self.C
        s = np.asarray(ins[0], np.float64)
        pos = _i32(ins[1])
        n, keys = attn_keys(pos, _i32(ins[2]), T, C, self.Q)
        P = np.zeros((G * T, C))
        if n == 0:
            return P
        S = pos + n
        j = np.arange(S)[None, :]
        t = np.arange(n)[:, None]
        mask = j <= pos + t                                     # [n][S]
        for hh in range(G):
            raw = s[:S, hh * T:hh * T + n].T.astype(np.int64)   # [n][S]
            m = np.where(mask, raw, -(1 << 20)).max(-1, keepdims=True)
            k = np.where(mask, m - raw, 0)
            e = np.where(mask, sexp_table(int(self.fs[hh]), self.scale)[k], 0.0)
            p = e / np.cumsum(e, -1)[..., -1:]
            P[hh * T:hh * T + n, :S] = np.clip(np.round(p * 2.0 ** int(self.fp[hh])),
                                               -32768, 32767)
        return P


@dataclass
class LlmAttnMergeNode(LlmNode):
    """FPGA prefill attention, host part 3: the P.V outputs o_g [G*T][HD]
    (raw) -> pv [T][H*HD] in head order, pv[t][(g*G + h')*HD + d] =
    o_g[h'*T + t][d] (raw, at pv's exponent f_p + f_vc - 8)."""
    T:  int = 1
    H:  int = 1
    KV: int = 1
    HD: int = 16

    @classmethod
    def from_onnx_node(cls, node, tensors, index, align_elems, ctx: HostContext):
        ins = _llm_inputs(node, tensors)
        y = _resolve(tensors, node.output[0], node)
        a, H, KV, HD = _heads_attrs(node)
        G = H // KV
        _require(len(ins) == KV, node, f"{KV} inputs (one per KV group)")
        T = y.numel // (H * HD)
        for t in ins:
            _require(list(t.shape) == [G * T, HD], node, f"inputs must be [{G * T}][{HD}]")
        _require(list(y.shape) == [T, H * HD], node, f"output must be [{T}][{H * HD}]")
        sn = cls(onnx_node=node, inputs=list(ins), output=y, index=index,
                 align_elems=align_elems, T=T, H=H, KV=KV, HD=HD, F=ctx.frac_bits)
        for t in ins + [y]:
            sn._want(t, None, "tensor")
        return sn

    def describe(self):
        return f"{self.KV} groups x {self.H // self.KV} heads x {self.T} rows -> [{self.T}][{self.H * self.HD}]"

    def c_call(self, ins, out, scratch, direct, dtype):
        return [
            "{",
            f"    const Data_t *_o[{self.KV}] = {{ {', '.join(ins)} }};",
            f"    llm_attn_merge(_o, {self.KV}u, {self.H // self.KV}u, {self.T}u, {self.HD}u, {out});",
            "}",
        ]

    def reference(self, ins, dtype):
        T, H, KV, HD = self.T, self.H, self.KV, self.HD
        G = H // KV
        raw = np.zeros((T, H, HD))
        for g in range(KV):
            o = np.asarray(ins[g], np.float64).reshape(G, T, HD)
            for hh in range(G):
                raw[:, g * G + hh] = o[hh]
        f = self.output.exp_channels(self.F).astype(np.float64)
        return (raw.reshape(T, H * HD) / np.power(2.0, f)[None, :])


# ------------------------------------------------------------------ #
# LlmSelectRow / LlmDequant                                            #
# ------------------------------------------------------------------ #

@dataclass
class LlmSelectRowNode(LlmNode):
    """y = h[n - 1] (n clamped to [1, rows]) — the last valid row of a
    padded prefill, handed to the head entry through a state."""
    rows: int = 1
    n:    int = 1

    @classmethod
    def from_onnx_node(cls, node, tensors, index, align_elems, ctx: HostContext):
        h, idx = _llm_inputs(node, tensors)
        y = _resolve(tensors, node.output[0], node)
        n = int(h.shape[-1])
        sn = cls(onnx_node=node, inputs=[h, idx], output=y, index=index,
                 align_elems=align_elems, rows=h.numel // n, n=n, F=ctx.frac_bits)
        sn._want(h, "f32", "h")
        sn._want(idx, "i32", "n")
        sn._want(y, "f32", "output")
        _require(y.numel == n, node, "output must be one row")
        return sn

    def describe(self):
        return f"row n-1 of [{self.rows}][{self.n}]"

    def c_call(self, ins, out, scratch, direct, dtype):
        return [f"llm_select_row({ins[0]}, {self.rows}u, {self.n}u, {ins[1]}[0], {out});"]

    def reference(self, ins, dtype):
        h = np.asarray(ins[0], np.float64).reshape(self.rows, self.n)
        r = int(np.asarray(ins[1]).reshape(-1)[0])
        r = min(max(r, 1), self.rows) - 1
        return h[r].reshape(self.output.shape).copy()


@dataclass
class LlmDequantNode(LlmNode):
    """y = float32(raw * 2^-f[c]) — e.g. the logits for the caller."""
    count: int = 1

    @classmethod
    def from_onnx_node(cls, node, tensors, index, align_elems, ctx: HostContext):
        x = _resolve(tensors, node.input[0], node)
        y = _resolve(tensors, node.output[0], node)
        sn = cls(onnx_node=node, inputs=[x], output=y, index=index,
                 align_elems=align_elems, count=x.numel, F=ctx.frac_bits)
        sn._want(x, None, "input")
        sn._want(y, "f32", "output")
        _require(y.numel == x.numel and x.shape[-1] == y.shape[-1], node, "shapes")
        return sn

    def c_runtime(self):
        return [self._scales(self.inputs[0])[2]]

    def describe(self):
        return f"{self.count} values"

    def c_call(self, ins, out, scratch, direct, dtype):
        s = self._scales(self.inputs[0])[0]
        n = int(self.inputs[0].shape[-1])
        return [f"llm_dequant({ins[0]}, {s}, {self.count}u, {n}u, {out});"]

    def reference(self, ins, dtype):
        return _f32(np.asarray(ins[0], np.float64)).reshape(self.output.shape)


LLM_OP_FACTORIES = {
    "LlmEmbed":     LlmEmbedNode.from_onnx_node,
    "LlmResAdd":    LlmResAddNode.from_onnx_node,
    "LlmRMSNorm":   LlmRMSNormNode.from_onnx_node,
    "LlmSiluMul":   LlmSiluMulNode.from_onnx_node,
    "LlmAttention": LlmAttentionNode.from_onnx_node,
    "LlmSelectRow": LlmSelectRowNode.from_onnx_node,
    "LlmDequant":   LlmDequantNode.from_onnx_node,
    "LlmAttnPrep":    LlmAttnPrepNode.from_onnx_node,
    "LlmAttnScores":  LlmAttnConvNode.from_onnx_node,
    "LlmAttnSoftmax": LlmAttnSoftmaxNode.from_onnx_node,
    "LlmAttnPV":      LlmAttnConvNode.from_onnx_node,
    "LlmAttnMerge":   LlmAttnMergeNode.from_onnx_node,
}


# ------------------------------------------------------------------ #
# C helper library                                                     #
# ------------------------------------------------------------------ #

LLM_C = r"""/*
 * Llama-family host ops (src/llm_nodes.py; policy pow2+sink+p12+xattn of
 * demo/chat/scripts/llm_study.py).  int16 tensors carry a power-of-two
 * exponent per channel: value = raw * sc[c] (sc = 2^-f, exact), written back
 * with llm_st (round half to even at 2^f, saturate, NaN -> 0).
 */
static int llm_scales(double **s, double **inv, const signed char *e, unsigned n)
{
    unsigned c;
    *s = (double *)malloc((size_t)n * sizeof(double));
    *inv = (double *)malloc((size_t)n * sizeof(double));
    if (!*s || !*inv) return -1;
    for (c = 0u; c < n; c++) {
        (*s)[c] = ldexp(1.0, -(int)e[c]);
        (*inv)[c] = ldexp(1.0, (int)e[c]);
    }
    return 0;
}

static inline double llm_ld(Data_t b, double sc) { return (double)(int16_t)b * sc; }

static inline int16_t llm_st16(double v, double si)
{
    double r;
    if (v != v) return 0;
    r = nearbyint(v * si);
    if (r > 32767.0) r = 32767.0;
    if (r < -32768.0) r = -32768.0;
    return (int16_t)r;
}

static inline Data_t llm_st(double v, double si) { return (Data_t)llm_st16(v, si); }

#ifndef INFERENCE_WEIGHTS_DIR
#  define INFERENCE_WEIGHTS_DIR  "."
#endif

/* A host table from INFERENCE_WEIGHTS_DIR/weights/<name>.dat into malloc'd
 * (cached, non-DMA) memory. */
static int llm_load_table(void **dst, const char *name, size_t bytes)
{
    char   path[512];
    FILE  *f;
    size_t got;
    snprintf(path, sizeof(path), INFERENCE_WEIGHTS_DIR "/weights/%s.dat", name);
    *dst = malloc(bytes ? bytes : 1u);
    if (!*dst) return -1;
    f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "inference: cannot open host table '%s'\n", path);
        return -1;
    }
    got = fread(*dst, 1u, bytes, f);
    fclose(f);
    if (got != bytes) {
        fprintf(stderr, "inference: short read from '%s': %zu of %zu bytes\n", path, got, bytes);
        return -1;
    }
    return 0;
}

/* ---- LlmEmbed: h[t] = table[ids[t]] (bf16 or f32 rows; index clamped) ---- */
typedef struct {
    const int32_t  *ids;
    const uint16_t *t16;
    const float    *t32;
    unsigned        rows, d;
    float          *h;
} llm_embed_t;

static void llm_embed_range(void *p, unsigned t0, unsigned t1)
{
    const llm_embed_t *a = (const llm_embed_t *)p;
    unsigned t, i;
    for (t = t0; t < t1; t++) {
        long   k = (long)a->ids[t];
        float *h = a->h + (size_t)t * a->d;
        if (k < 0) k = 0;
        if (k >= (long)a->rows) k = (long)a->rows - 1;
        if (a->t16) {
            const uint16_t *r = a->t16 + (size_t)k * a->d;
            for (i = 0u; i < a->d; i++) {
                uint32_t u = (uint32_t)r[i] << 16;
                memcpy(&h[i], &u, sizeof u);
            }
        } else {
            memcpy(h, a->t32 + (size_t)k * a->d, (size_t)a->d * sizeof(float));
        }
    }
}

static void llm_embed(const int32_t *ids, unsigned n, const uint16_t *t16, const float *t32,
                      unsigned rows, unsigned d, float *h)
{
    llm_embed_t a;
    a.ids = ids; a.t16 = t16; a.t32 = t32; a.rows = rows; a.d = d; a.h = h;
    host_parallel(llm_embed_range, &a, n, host_row_grain(d), 1u);
}

/* ---- LlmResAdd: y = float32((double)h + x·sx) ---- */
typedef struct {
    const float  *h;
    const Data_t *x;
    const double *sx;
    unsigned      n;
    float        *y;
} llm_resadd_t;

static void llm_resadd_rows(void *p, unsigned r0, unsigned r1)
{
    const llm_resadd_t *a = (const llm_resadd_t *)p;
    unsigned r, c;
    for (r = r0; r < r1; r++) {
        const float  *h = a->h + (size_t)r * a->n;
        const Data_t *x = a->x + (size_t)r * a->n;
        float        *y = a->y + (size_t)r * a->n;
        for (c = 0u; c < a->n; c++)
            y[c] = (float)((double)h[c] + llm_ld(x[c], a->sx[c]));
    }
}

static void llm_resadd(const float *h, const Data_t *x, const double *sx,
                       unsigned rows, unsigned n, float *y)
{
    llm_resadd_t a;
    a.h = h; a.x = x; a.sx = sx; a.n = n; a.y = y;
    host_parallel(llm_resadd_rows, &a, rows, host_row_grain(n), 1u);
}

/* ---- LlmRMSNorm: ss = sum h^2 (left to right), r = 1/sqrt(ss/n + eps),
 *      y = (h * r) * gamma ---- */
typedef struct {
    const float  *h;
    unsigned      n;
    const float  *gamma;
    double        eps;
    const double *iy;
    Data_t       *y;
} llm_rms_t;

static void llm_rmsnorm_rows(void *p, unsigned r0, unsigned r1)
{
    const llm_rms_t *a = (const llm_rms_t *)p;
    unsigned r, c;
    for (r = r0; r < r1; r++) {
        const float *h = a->h + (size_t)r * a->n;
        Data_t      *y = a->y + (size_t)r * a->n;
        double       ss = 0.0, rr;
        for (c = 0u; c < a->n; c++)
            ss += (double)h[c] * (double)h[c];
        rr = 1.0 / sqrt(ss / (double)a->n + a->eps);
        for (c = 0u; c < a->n; c++)
            y[c] = llm_st(((double)h[c] * rr) * (double)a->gamma[c], a->iy[c]);
    }
}

static void llm_rmsnorm(const float *h, unsigned rows, unsigned n, const float *gamma,
                        double eps, const double *iy, Data_t *y)
{
    llm_rms_t a;
    a.h = h; a.n = n; a.gamma = gamma; a.eps = eps; a.iy = iy; a.y = y;
    host_parallel(llm_rmsnorm_rows, &a, rows, host_row_grain(n), 1u);
}

/* ---- LlmSiluMul: a = silu(g) * u, silu tabulated over g's raw int16 ---- */
typedef struct {
    double *t;
    double  d;
} llm_silu_fill_t;

static void llm_silu_fill(void *p, unsigned i0, unsigned i1)
{
    const llm_silu_fill_t *a = (const llm_silu_fill_t *)p;
    unsigned i;
    for (i = i0; i < i1; i++) {
        double g = ((double)((int)i - 32768) + 0.0) / a->d;
        a->t[i] = g > -700.0 ? g / (1.0 + exp(-g)) : 0.0;
    }
}

/* One silu table per gate exponent f, _llm_silu_tab[f - LLM_SILU_EMIN]. */
#define LLM_SILU_EMIN  (SILU_EMIN_VALUE)
#define LLM_SILU_NE    (SILU_NE_VALUE)
static double *_llm_silu_tab[LLM_SILU_NE];

static int llm_silu_table(int f)
{
    llm_silu_fill_t a;
    double        **t = &_llm_silu_tab[f - LLM_SILU_EMIN];
    if (*t) return 0;
    *t = (double *)malloc(65536u * sizeof(double));
    if (!*t) return -1;
    a.t = *t; a.d = ldexp(1.0, f);
    host_parallel(llm_silu_fill, &a, 65536u, 1024u, 8u);
    return 0;
}

static void llm_silu_free(int f)
{
    free(_llm_silu_tab[f - LLM_SILU_EMIN]);
    _llm_silu_tab[f - LLM_SILU_EMIN] = NULL;
}

typedef struct {
    const Data_t        *g, *u;
    const signed char   *fg;            /* gate exponent per channel */
    const double        *su, *ia;
    unsigned             n;
    Data_t              *y;
} llm_silu_t;

static void llm_silu_rows(void *p, unsigned r0, unsigned r1)
{
    const llm_silu_t *a = (const llm_silu_t *)p;
    unsigned r, c;
    for (r = r0; r < r1; r++) {
        const Data_t *g = a->g + (size_t)r * a->n;
        const Data_t *u = a->u + (size_t)r * a->n;
        Data_t       *y = a->y + (size_t)r * a->n;
        for (c = 0u; c < a->n; c++)
            y[c] = llm_st(_llm_silu_tab[a->fg[c] - LLM_SILU_EMIN][(int)(int16_t)g[c] + 32768]
                          * llm_ld(u[c], a->su[c]), a->ia[c]);
    }
}

static void llm_silu_mul(const Data_t *g, const Data_t *u, const signed char *fg,
                         const double *su, const double *ia, unsigned rows, unsigned n,
                         Data_t *y)
{
    llm_silu_t a;
    a.g = g; a.u = u; a.fg = fg; a.su = su; a.ia = ia; a.n = n; a.y = y;
    host_parallel(llm_silu_rows, &a, rows, host_row_grain(n), 1u);
}

/* ---- LlmSelectRow / LlmDequant ---- */
static void llm_select_row(const float *h, unsigned rows, unsigned n, int32_t nv, float *y)
{
    unsigned r = nv < 1 ? 0u : ((unsigned)nv > rows ? rows - 1u : (unsigned)nv - 1u);
    memcpy(y, h + (size_t)r * n, (size_t)n * sizeof(float));
}

static void llm_dequant(const Data_t *x, const double *sx, unsigned count, unsigned n, float *y)
{
    unsigned i;
    for (i = 0u; i < count; i++)
        y[i] = (float)llm_ld(x[i], sx[i % n]);
}

/* ---- LlmAttention (xattn) ----
 * The caches are stored group-major, [KV][C][HD] (row j of KV head g at
 * (g * C + j) * HD): host states, or DMA states the FPGA prefill attention
 * reads (written here without a flush — LlmAttnPrep flushes the rows the
 * kernels read).
 * Rows t < n (pos + t < C): RoPE(k0[t]) -> K cache row pos + t at its
 * exponent, v[t] re-rounded -> V cache row.  The cache rows the queries read
 * (0 .. pos + n - 1) are converted once to their exact double values
 * (raw * 2^-f, the same bits the per-pair expression gives) into a scratch
 * when n > 1 (prefill; a decode step, n == 1, runs llm_attn_decode below —
 * the same values, split across all host threads).
 * Then per (t, head h), over the keys j <= pos + t: RoPE(q0[t,h]) in double,
 * s_j = dot8(q, k_j) * scale, e_j = exp(s_j - max), sum left to right,
 * p_j = e_j / sum, o[d] = sum_j p_j v_j[d] -> pv[t][h*HD + d].  Rows t >= n
 * get pv = 0.  Work items are ordered head-major with the rows interleaved
 * (item k: h = k / n, t = k % n), so the host threads' contiguous ranges get
 * equal shares of the causal work.  Every sum keeps its order: the result
 * does not depend on the thread count or on vectorisation (lanes only). */
typedef struct {
    const Data_t *q0, *k0, *v;
    const double *sq, *sk, *sv;
    int16_t      *ck, *cv;
    const double *sck, *ick, *scv, *icv, *ipv;
    Data_t       *pv;
    const float  *cos, *sin;
    unsigned      pos, n, T, H, KV, HD, C, VK;
    double        scale;
} llm_attn_t;

/* V cache row k of KV head g: group-major, rows interleaved by VK as the
 * 1 x VK conv input image of the P.V call (TensorInfo.group_kw); element d
 * at llm_vrow(...) + d * VK.  VK = 1: (g * C + k) * HD. */
static inline size_t llm_vrow(unsigned C, unsigned HD, unsigned VK, unsigned g, unsigned k)
{
    return (size_t)g * C * HD + ((size_t)(k / (16u * VK)) * 16u + k % 16u) * HD * VK
         + (k / 16u) % VK;
}

#ifndef LLM_MAX_KEYS
#  define LLM_MAX_KEYS 4096u
#endif
#ifndef LLM_MAX_HD
#  define LLM_MAX_HD 256u
#endif

static double  *_llm_attn_kd = NULL, *_llm_attn_vd = NULL;   /* exact K / V values */
static unsigned _llm_attn_cap = 0u;

static int llm_attn_reserve(unsigned n)
{
    if (n <= _llm_attn_cap) return 0;
    free(_llm_attn_kd);
    free(_llm_attn_vd);
    _llm_attn_kd = (double *)malloc((size_t)n * sizeof(double));
    _llm_attn_vd = (double *)malloc((size_t)n * sizeof(double));
    _llm_attn_cap = (_llm_attn_kd && _llm_attn_vd) ? n : 0u;
    return _llm_attn_cap ? 0 : -1;
}

static void llm_dec_release(void);

static void llm_attn_release(void)
{
    free(_llm_attn_kd);
    free(_llm_attn_vd);
    _llm_attn_kd = _llm_attn_vd = NULL;
    _llm_attn_cap = 0u;
    llm_dec_release();
}

#if defined(__GNUC__) && !defined(__clang__)
#  pragma GCC push_options
#  pragma GCC optimize ("tree-vectorize")    /* lane-wise only: bits unchanged */
#endif
static void llm_attn_convert(void *p, unsigned r0, unsigned r1)
{
    const llm_attn_t *a = (const llm_attn_t *)p;
    const unsigned    HD = a->HD, nk = a->pos + a->n;
    unsigned          r, d;
    for (r = r0; r < r1; r++) {              /* item r: KV head r / nk, key row r % nk */
        const unsigned g = r / nk, j = r % nk;
        const size_t   o = ((size_t)g * a->C + j) * HD;
        const int16_t *kr = a->ck + o, *vr = a->cv + llm_vrow(a->C, HD, a->VK, g, j);
        double        *kd = _llm_attn_kd + o, *vd = _llm_attn_vd + o;
        const double  *sk = a->sck + (size_t)g * HD, *sv = a->scv + (size_t)g * HD;
        for (d = 0u; d < HD; d++) {
            kd[d] = (double)kr[d] * sk[d];
            vd[d] = (double)vr[(size_t)d * a->VK] * sv[d];
        }
    }
}

static void llm_attn_items(void *p, unsigned i0, unsigned i1)
{
    const llm_attn_t *a = (const llm_attn_t *)p;
    const unsigned    HD = a->HD, half = HD / 2u, grp = a->H / a->KV;
    double            q[LLM_MAX_HD], o[LLM_MAX_HD], e[LLM_MAX_KEYS];
    unsigned          it;
    for (it = i0; it < i1; it++) {
        const unsigned t = it % a->n, h = it / a->n, g = h / grp, pp = a->pos + t;
        const Data_t  *qr = a->q0 + (size_t)t * a->H * HD + (size_t)h * HD;
        const double  *sq = a->sq + (size_t)h * HD;
        const float   *cs = a->cos + (size_t)pp * half, *sn = a->sin + (size_t)pp * half;
        const unsigned nk = pp + 1u;
        double         m = 0.0, sum = 0.0;
        unsigned       d, j, l;
        for (d = 0u; d < HD; d++) {
            double x = llm_ld(qr[d], sq[d]);
            double r = d < half ? -llm_ld(qr[d + half], sq[d + half])
                                :  llm_ld(qr[d - half], sq[d - half]);
            q[d] = x * (double)cs[d % half] + r * (double)sn[d % half];
        }
        for (j = 0u; j < nk; j++) {
            double acc[8], s;
            if (a->n > 1u) {                        /* prefill: converted rows */
                const double *kr = _llm_attn_kd + ((size_t)g * a->C + j) * HD;
                for (l = 0u; l < 8u; l++)
                    acc[l] = q[l] * kr[l];
                for (d = 8u; d < HD; d += 8u)
                    for (l = 0u; l < 8u; l++)
                        acc[l] += q[d + l] * kr[d + l];
            } else {                                /* decode: the int16 cache row */
                const int16_t *kr = a->ck + ((size_t)g * a->C + j) * HD;
                const double  *sk = a->sck + (size_t)g * HD;
                for (l = 0u; l < 8u; l++)
                    acc[l] = q[l] * ((double)kr[l] * sk[l]);
                for (d = 8u; d < HD; d += 8u)
                    for (l = 0u; l < 8u; l++)
                        acc[l] += q[d + l] * ((double)kr[d + l] * sk[d + l]);
            }
            s = (((acc[0] + acc[1]) + (acc[2] + acc[3]))
                 + ((acc[4] + acc[5]) + (acc[6] + acc[7]))) * a->scale;
            e[j] = s;
            if (j == 0u || s > m) m = s;
        }
        for (j = 0u; j < nk; j++) {
            e[j] = exp(e[j] - m);
            sum += e[j];
        }
        for (j = 0u; j < nk; j++) {
            const double pj = e[j] / sum;
            if (a->n > 1u) {
                const double *vr = _llm_attn_vd + ((size_t)g * a->C + j) * HD;
                if (j == 0u)
                    for (d = 0u; d < HD; d++)
                        o[d] = pj * vr[d];
                else
                    for (d = 0u; d < HD; d++)
                        o[d] += pj * vr[d];
            } else {
                const int16_t *vr = a->cv + llm_vrow(a->C, HD, a->VK, g, j);
                const double  *sv = a->scv + (size_t)g * HD;
                const unsigned vk = a->VK;
                if (j == 0u)
                    for (d = 0u; d < HD; d++)
                        o[d] = pj * ((double)vr[(size_t)d * vk] * sv[d]);
                else
                    for (d = 0u; d < HD; d++)
                        o[d] += pj * ((double)vr[(size_t)d * vk] * sv[d]);
            }
        }
        {
            Data_t       *y = a->pv + (size_t)t * a->H * HD + (size_t)h * HD;
            const double *iy = a->ipv + (size_t)h * HD;
            for (d = 0u; d < HD; d++)
                y[d] = llm_st(o[d], iy[d]);
        }
    }
}
#if defined(__GNUC__) && !defined(__clang__)
#  pragma GCC pop_options
#endif

/* ---- LlmAttention decode step (n == 1) on all host threads ----
 * One pool dispatch; the phases are split into units and separated by spin
 * barriers (a thread-pool wake-up per phase would cost more than it saves):
 *   A  scores, unit = (KV group, key quarter): the group's G heads share
 *      each K row load and conversion; a partial max per quarter
 *   B  e = exp(s - max), unit = (group, key quarter)
 *   C  sum of e left to right, unit = head
 *   D  p = e / sum, unit = (group, key quarter)
 *   E  P.V, unit = (group, lane quarter)
 * The power-of-two cache scales are folded out exactly: the scores use
 * (q * 2^-f_k[d]) against the raw K row (the same product, one rounding),
 * and P.V sums p * raw v and scales once — 2^-f commutes with every
 * rounding while all terms stay normal, i.e. while p >= 2^-900; a head with
 * a smaller non-zero p keeps the per-term scale.  Every value is therefore
 * the per-head code's (llm_attn_items) bit for bit, for any thread count;
 * the NEON kernels use separate multiplies and adds (no FMA). */
#define LLM_DEC_NQ 4u
#ifndef LLM_MAX_G
#  define LLM_MAX_G 8u
#endif
#if HOST_POOL
#  include <sched.h>
#endif

static double  *_llm_dec_p = NULL, *_llm_dec_qs = NULL, *_llm_dec_u = NULL;
static unsigned _llm_dec_pcap = 0u, _llm_dec_hcap = 0u;

static int llm_dec_reserve(unsigned n_p, unsigned n_h)
{
    if (n_p > _llm_dec_pcap) {
        free(_llm_dec_p);
        _llm_dec_p = (double *)malloc((size_t)n_p * sizeof(double));
        _llm_dec_pcap = _llm_dec_p ? n_p : 0u;
    }
    if (n_h > _llm_dec_hcap) {
        free(_llm_dec_qs);
        free(_llm_dec_u);
        _llm_dec_qs = (double *)malloc((size_t)n_h * sizeof(double));
        _llm_dec_u = (double *)malloc((size_t)n_h * sizeof(double));
        _llm_dec_hcap = (_llm_dec_qs && _llm_dec_u) ? n_h : 0u;
    }
    return (n_p <= _llm_dec_pcap && n_h <= _llm_dec_hcap) ? 0 : -1;
}

static void llm_dec_release(void)
{
    free(_llm_dec_p);
    free(_llm_dec_qs);
    free(_llm_dec_u);
    _llm_dec_p = _llm_dec_qs = _llm_dec_u = NULL;
    _llm_dec_pcap = _llm_dec_hcap = 0u;
}

typedef struct {
    const llm_attn_t  *a;
    double             mq[64][LLM_DEC_NQ];   /* partial max per head and key quarter */
    double             sum[64];
    unsigned long long tiny;                 /* heads with 0 < p < 2^-900 */
    unsigned           nt, G, nk;
    unsigned           bar_count, bar_gen;
} llm_dec_t;

static void llm_dec_barrier(llm_dec_t *s)
{
#if HOST_POOL
    const unsigned gen = __atomic_load_n(&s->bar_gen, __ATOMIC_ACQUIRE);
    if (__atomic_add_fetch(&s->bar_count, 1u, __ATOMIC_ACQ_REL) == s->nt) {
        __atomic_store_n(&s->bar_count, 0u, __ATOMIC_RELAXED);
        __atomic_store_n(&s->bar_gen, gen + 1u, __ATOMIC_RELEASE);
    } else {
        unsigned spins = 0u;
        while (__atomic_load_n(&s->bar_gen, __ATOMIC_ACQUIRE) == gen)
            if (++spins == 2048u) {
                sched_yield();
                spins = 0u;
            }
    }
#else
    (void)s;
#endif
}

static inline unsigned llm_dec_part(unsigned n, unsigned q)
{
    return (unsigned)((unsigned long long)n * q / LLM_DEC_NQ);
}

#if defined(__GNUC__) && !defined(__clang__)
#  pragma GCC push_options
#  pragma GCC optimize ("O3")                /* lane-wise only: bits unchanged */
#endif
/* Scores of G heads against K rows [j0, j1): S[i * C + j], max into m[i].
 * Lane l of a head sums d = l, l + 8, ... in order (dot8). */
static inline __attribute__((always_inline))
void llm_dec_scores(const double *qs, const int16_t *K, unsigned j0, unsigned j1, unsigned HD,
                    double scale, double *S, unsigned C, double *m, const unsigned G)
{
    unsigned j, d, l, i;
    for (j = j0; j < j1; j++) {
        const int16_t *kr = K + (size_t)j * HD;
        double         acc[LLM_MAX_G][8];
        for (i = 0u; i < G; i++)
            for (l = 0u; l < 8u; l++)
                acc[i][l] = qs[i * HD + l] * (double)kr[l];
        for (d = 8u; d < HD; d += 8u)
            for (l = 0u; l < 8u; l++) {
                const double x = (double)kr[d + l];
                for (i = 0u; i < G; i++)
                    acc[i][l] += qs[i * HD + d + l] * x;
            }
        for (i = 0u; i < G; i++) {
            const double *c = acc[i];
            const double  s = (((c[0] + c[1]) + (c[2] + c[3]))
                               + ((c[4] + c[5]) + (c[6] + c[7]))) * scale;
            S[(size_t)i * C + j] = s;
            if (s > m[i]) m[i] = s;
        }
    }
}

/* o[i * HD + d] = sum_j p[i][j] * raw v_j[d] (j ascending), d in [d0, d0 + 8).
 * Key j = (b * VK + ln) * 16 + r is element ln of image row b * 16 + r. */
static inline __attribute__((always_inline))
void llm_dec_pv8(const double *P, unsigned C, const int16_t *V, unsigned nk, unsigned HD,
                 unsigned VK, unsigned d0, double *o, const unsigned G)
{
    double         acc[LLM_MAX_G][8] = { { 0.0 } };
    const size_t   rs = (size_t)HD * VK;
    const int16_t *blk = V + (size_t)d0 * VK;
    unsigned       j = 0u, l, i;
    for (; j < nk; blk += 16u * rs) {
        unsigned ln;
        for (ln = 0u; ln < VK && j < nk; ln++) {
            const int16_t *vr = blk + ln;
            unsigned       r;
            for (r = 0u; r < 16u && j < nk; r++, j++, vr += rs) {
                double x[8];
                for (l = 0u; l < 8u; l++)
                    x[l] = (double)vr[(size_t)l * VK];
                for (i = 0u; i < G; i++) {
                    const double p = P[(size_t)i * C + j];
                    if (j == 0u)
                        for (l = 0u; l < 8u; l++) acc[i][l] = p * x[l];
                    else
                        for (l = 0u; l < 8u; l++) acc[i][l] += p * x[l];
                }
            }
        }
    }
    for (i = 0u; i < G; i++)
        for (l = 0u; l < 8u; l++)
            o[i * HD + d0 + l] = acc[i][l];
}

/* P.V with the per-term scale (llm_attn_items' expression), lanes d0 .. d0 + 7 */
static void llm_dec_pv8_scaled(const double *P, unsigned C, const int16_t *V, unsigned nk,
                               unsigned HD, unsigned VK, unsigned d0, const double *sv,
                               double *o, unsigned G)
{
    unsigned i, j, l;
    for (i = 0u; i < G; i++) {
        double acc[8] = { 0.0 };
        for (j = 0u; j < nk; j++) {
            const int16_t *vr = V + llm_vrow(0u, HD, VK, 0u, j) + (size_t)d0 * VK;
            const double   p = P[(size_t)i * C + j];
            for (l = 0u; l < 8u; l++) {
                const double t = p * ((double)vr[(size_t)l * VK] * sv[d0 + l]);
                acc[l] = j == 0u ? t : acc[l] + t;
            }
        }
        for (l = 0u; l < 8u; l++)
            o[i * HD + d0 + l] = acc[l];
    }
}

#if defined(__aarch64__) && defined(__ARM_NEON)
#  include <arm_neon.h>
#  define LLM_DEC_NEON 1
/* int16 x 8 -> 4 x float64x2 (exact: int16 -> f32 -> f64) */
static inline __attribute__((always_inline)) void llm_dec_cvt8(int16x8_t k, float64x2_t x[4])
{
    const float32x4_t lo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(k)));
    const float32x4_t hi = vcvtq_f32_s32(vmovl_high_s16(k));
    x[0] = vcvt_f64_f32(vget_low_f32(lo));
    x[1] = vcvt_high_f64_f32(lo);
    x[2] = vcvt_f64_f32(vget_low_f32(hi));
    x[3] = vcvt_high_f64_f32(hi);
}

/* llm_dec_scores for G = 3: lanes as 4 x float64x2 per head */
static void llm_dec_scores3(const double *qs, const int16_t *K, unsigned j0, unsigned j1,
                            unsigned HD, double scale, double *S, unsigned C, double *m)
{
    unsigned j, d, i, v;
    for (j = j0; j < j1; j++) {
        const int16_t *kr = K + (size_t)j * HD;
        float64x2_t    a[3][4], x[4];
        llm_dec_cvt8(vld1q_s16(kr), x);
        for (i = 0u; i < 3u; i++)
            for (v = 0u; v < 4u; v++)
                a[i][v] = vmulq_f64(vld1q_f64(qs + i * HD + 2u * v), x[v]);
        for (d = 8u; d < HD; d += 8u) {
            llm_dec_cvt8(vld1q_s16(kr + d), x);
            for (i = 0u; i < 3u; i++)
                for (v = 0u; v < 4u; v++)
                    a[i][v] = vaddq_f64(a[i][v],
                                        vmulq_f64(vld1q_f64(qs + i * HD + d + 2u * v), x[v]));
        }
        for (i = 0u; i < 3u; i++) {
            /* ((c0 + c1) + (c2 + c3)) + ((c4 + c5) + (c6 + c7)) */
            const double lo = vaddvq_f64(vpaddq_f64(a[i][0], a[i][1]));
            const double hi = vaddvq_f64(vpaddq_f64(a[i][2], a[i][3]));
            const double s = (lo + hi) * scale;
            S[(size_t)i * C + j] = s;
            if (s > m[i]) m[i] = s;
        }
    }
}

/* llm_dec_pv8 for G = 3 and VK = 1 or 4 */
static void llm_dec_pv8_3(const double *P, unsigned C, const int16_t *V, unsigned nk,
                          unsigned HD, unsigned VK, unsigned d0, double *o)
{
    float64x2_t    a[3][4], x[4];
    const size_t   rs = (size_t)HD * VK;
    const int16_t *blk = V + (size_t)d0 * VK;
    unsigned       j = 0u, i, v;
    for (i = 0u; i < 3u; i++)
        for (v = 0u; v < 4u; v++)
            a[i][v] = vdupq_n_f64(0.0);
    for (; j < nk; blk += 16u * rs) {
        unsigned ln;
        for (ln = 0u; ln < VK && j < nk; ln++) {
            const int16_t *vr = blk;
            unsigned       r;
            for (r = 0u; r < 16u && j < nk; r++, j++, vr += rs) {
                int16x8_t k8;
                if (VK == 1u) {
                    k8 = vld1q_s16(vr);
                } else {
                    const int16x8x4_t q4 = vld4q_s16(vr);     /* one 64-byte image row piece */
                    k8 = ln == 0u ? q4.val[0] : ln == 1u ? q4.val[1]
                       : ln == 2u ? q4.val[2] : q4.val[3];
                }
                llm_dec_cvt8(k8, x);
                for (i = 0u; i < 3u; i++) {
                    const float64x2_t p = vld1q_dup_f64(P + (size_t)i * C + j);
                    if (j == 0u)
                        for (v = 0u; v < 4u; v++) a[i][v] = vmulq_f64(p, x[v]);
                    else
                        for (v = 0u; v < 4u; v++) a[i][v] = vaddq_f64(a[i][v], vmulq_f64(p, x[v]));
                }
            }
        }
    }
    for (i = 0u; i < 3u; i++)
        for (v = 0u; v < 4u; v++)
            vst1q_f64(o + i * HD + d0 + 2u * v, a[i][v]);
}
#else
#  define LLM_DEC_NEON 0
#endif

static void llm_dec_phase_scores(llm_dec_t *s, unsigned t, unsigned step)
{
    const llm_attn_t *a = s->a;
    const unsigned    G = s->G, HD = a->HD;
    unsigned          w, i;
    for (w = t; w < a->KV * LLM_DEC_NQ; w += step) {
        const unsigned g = w / LLM_DEC_NQ, q = w % LLM_DEC_NQ;
        const unsigned j0 = llm_dec_part(s->nk, q), j1 = llm_dec_part(s->nk, q + 1u);
        const int16_t *K = a->ck + (size_t)g * a->C * HD;
        double        *S = _llm_dec_p + (size_t)g * G * a->C;
        const double  *qs = _llm_dec_qs + (size_t)g * G * HD;
        double         m[LLM_MAX_G];
        for (i = 0u; i < G; i++) m[i] = -INFINITY;
#if LLM_DEC_NEON
        if (G == 3u) llm_dec_scores3(qs, K, j0, j1, HD, a->scale, S, a->C, m);
        else
#endif
        if (G == 3u) llm_dec_scores(qs, K, j0, j1, HD, a->scale, S, a->C, m, 3u);
        else if (G == 1u) llm_dec_scores(qs, K, j0, j1, HD, a->scale, S, a->C, m, 1u);
        else llm_dec_scores(qs, K, j0, j1, HD, a->scale, S, a->C, m, G);
        for (i = 0u; i < G; i++) s->mq[g * G + i][q] = m[i];
    }
}

static void llm_dec_phase_exp(llm_dec_t *s, unsigned t, unsigned step)
{
    const llm_attn_t *a = s->a;
    unsigned          w, i, j, q;
    for (w = t; w < a->KV * LLM_DEC_NQ; w += step) {
        const unsigned g = w / LLM_DEC_NQ, qq = w % LLM_DEC_NQ;
        const unsigned j0 = llm_dec_part(s->nk, qq), j1 = llm_dec_part(s->nk, qq + 1u);
        for (i = 0u; i < s->G; i++) {
            const unsigned h = g * s->G + i;
            double        *e = _llm_dec_p + (size_t)h * a->C, m = -INFINITY;
            for (q = 0u; q < LLM_DEC_NQ; q++)
                if (s->mq[h][q] > m) m = s->mq[h][q];
            for (j = j0; j < j1; j++)
                e[j] = exp(e[j] - m);
        }
    }
}

static void llm_dec_phase_sum(llm_dec_t *s, unsigned t, unsigned step)
{
    unsigned h, j;
    for (h = t; h < s->a->H; h += step) {
        const double *e = _llm_dec_p + (size_t)h * s->a->C;
        double        sum = 0.0;
        for (j = 0u; j < s->nk; j++)
            sum += e[j];
        s->sum[h] = sum;
    }
}

static void llm_dec_phase_div(llm_dec_t *s, unsigned t, unsigned step)
{
    const llm_attn_t *a = s->a;
    unsigned          w, i, j;
    for (w = t; w < a->KV * LLM_DEC_NQ; w += step) {
        const unsigned g = w / LLM_DEC_NQ, q = w % LLM_DEC_NQ;
        const unsigned j0 = llm_dec_part(s->nk, q), j1 = llm_dec_part(s->nk, q + 1u);
        for (i = 0u; i < s->G; i++) {
            const unsigned h = g * s->G + i;
            double        *e = _llm_dec_p + (size_t)h * a->C;
            const double   sum = s->sum[h];
            unsigned       tiny = 0u;
            for (j = j0; j < j1; j++) {
                e[j] = e[j] / sum;
                tiny |= e[j] != 0.0 && e[j] < 0x1p-900;
            }
            if (tiny) __atomic_fetch_or(&s->tiny, 1ull << h, __ATOMIC_RELAXED);
        }
    }
}

static void llm_dec_phase_pv(llm_dec_t *s, unsigned t, unsigned step)
{
    const llm_attn_t *a = s->a;
    const unsigned    G = s->G, HD = a->HD, nb = HD / 8u;
    unsigned          w, i, b, l;
    for (w = t; w < a->KV * LLM_DEC_NQ; w += step) {
        const unsigned g = w / LLM_DEC_NQ, q = w % LLM_DEC_NQ;
        const unsigned b0 = llm_dec_part(nb, q), b1 = llm_dec_part(nb, q + 1u);
        const int16_t *V = a->cv + (size_t)g * a->C * HD;
        const double  *P = _llm_dec_p + (size_t)g * G * a->C, *sv = a->scv + (size_t)g * HD;
        double        *u = _llm_dec_u + (size_t)g * G * HD;
        const unsigned long long gm = ((1ull << G) - 1ull) << (g * G);
        const unsigned scaled = (__atomic_load_n(&s->tiny, __ATOMIC_RELAXED) & gm) != 0ull;
        for (b = b0; b < b1; b++) {
            const unsigned d0 = 8u * b;
            if (scaled)
                llm_dec_pv8_scaled(P, a->C, V, s->nk, HD, a->VK, d0, sv, u, G);
#if LLM_DEC_NEON
            else if (G == 3u && (a->VK == 1u || a->VK == 4u))
                llm_dec_pv8_3(P, a->C, V, s->nk, HD, a->VK, d0, u);
#endif
            else if (G == 3u) llm_dec_pv8(P, a->C, V, s->nk, HD, a->VK, d0, u, 3u);
            else if (G == 1u) llm_dec_pv8(P, a->C, V, s->nk, HD, a->VK, d0, u, 1u);
            else llm_dec_pv8(P, a->C, V, s->nk, HD, a->VK, d0, u, G);
            for (i = 0u; i < G; i++) {
                const unsigned h = g * G + i;
                Data_t        *y = a->pv + (size_t)h * HD;
                const double  *iy = a->ipv + (size_t)h * HD;
                for (l = d0; l < d0 + 8u; l++)
                    y[l] = llm_st(scaled ? u[i * HD + l] : u[i * HD + l] * sv[l], iy[l]);
            }
        }
    }
}
#if defined(__GNUC__) && !defined(__clang__)
#  pragma GCC pop_options
#endif

/* One range per pool thread (n = nt items): phases with barriers; a single
 * range [0, nt) (no pool) runs them in turn. */
static void llm_dec_run(void *p, unsigned b, unsigned e)
{
    llm_dec_t     *s = (llm_dec_t *)p;
    const unsigned par = e - b < s->nt, t = par ? b : 0u, step = par ? s->nt : 1u;
    llm_dec_phase_scores(s, t, step);
    if (par) llm_dec_barrier(s);
    llm_dec_phase_exp(s, t, step);
    if (par) llm_dec_barrier(s);
    llm_dec_phase_sum(s, t, step);
    if (par) llm_dec_barrier(s);
    llm_dec_phase_div(s, t, step);
    if (par) llm_dec_barrier(s);
    llm_dec_phase_pv(s, t, step);
}

/* Env INFERENCE_LLM_DECODE_PAR (read once): 0 = the per-head code
 * (llm_attn_items), 1 = the phases on all host threads (default), 2 = the
 * phases on the calling thread only (no pool dispatch, no spin barriers).
 * The results are the same bits in every mode. */
static int llm_dec_mode = -1;

/* The decode step's attention (query row 0 at a->pos; the cache rows are
 * already written).  0 = not handled here (mode 0, geometry, scratch, or a
 * q * 2^-f below the normal range): the caller runs the per-head code. */
static int llm_attn_decode(const llm_attn_t *a)
{
    const unsigned HD = a->HD, half = HD / 2u, G = a->KV ? a->H / a->KV : 0u, pp = a->pos;
    const float   *cs = a->cos + (size_t)pp * half, *sn = a->sin + (size_t)pp * half;
    llm_dec_t      s;
    unsigned       h, d;
    if (llm_dec_mode < 0) {
        const char *e = getenv("INFERENCE_LLM_DECODE_PAR");
        llm_dec_mode = (e && (e[0] == '0' || e[0] == '2')) ? e[0] - '0' : 1;
    }
    if (llm_dec_mode == 0 || G == 0u || G > LLM_MAX_G || a->H > 64u || a->H != G * a->KV || HD % 8u
            || HD > LLM_MAX_HD || llm_dec_reserve(a->H * a->C, a->H * HD) != 0)
        return 0;
    for (h = 0u; h < a->H; h++) {                /* RoPE(q) * 2^-f_k (exact) */
        const Data_t *qr = a->q0 + (size_t)h * HD;
        const double *sq = a->sq + (size_t)h * HD, *sk = a->sck + (size_t)(h / G) * HD;
        double       *qs = _llm_dec_qs + (size_t)h * HD;
        for (d = 0u; d < HD; d++) {
            double x = llm_ld(qr[d], sq[d]);
            double r = d < half ? -llm_ld(qr[d + half], sq[d + half])
                                :  llm_ld(qr[d - half], sq[d - half]);
            double q = x * (double)cs[d % half] + r * (double)sn[d % half];
            qs[d] = q * sk[d];
            if (q != 0.0 && fabs(qs[d]) < 0x1p-1000) return 0;
        }
    }
    s.a = a;
    s.tiny = 0ull;
    s.G = G;
    s.nk = pp + 1u;
    s.nt = llm_dec_mode == 2 ? 1u : s_host_nthreads;
    s.bar_count = 0u;
    s.bar_gen = 0u;
    if (s.nt > 1u)
        host_parallel(llm_dec_run, &s, s.nt, 1u, 1u);
    else
        llm_dec_run(&s, 0u, 1u);
    return 1;
}

static void llm_attention(llm_attn_t *a)
{
    const unsigned HD = a->HD, half = HD / 2u, kvhd = a->KV * HD;
    unsigned       t, c, n = a->n;
    if (n > a->T) n = a->T;
    if (a->pos >= a->C) n = 0u;
    else if (n > a->C - a->pos) n = a->C - a->pos;
    if (a->pos + n > LLM_MAX_KEYS || HD > LLM_MAX_HD
            || (size_t)(a->pos + n) * kvhd > _llm_attn_cap) {
        fprintf(stderr, "llm_attention: %u keys / head_dim %u above LLM_MAX_KEYS / LLM_MAX_HD"
                " or the scratch\n", a->pos + n, HD);
        n = 0u;
    }
    for (t = 0u; t < n; t++) {
        const unsigned pp = a->pos + t;
        const Data_t  *k = a->k0 + (size_t)t * kvhd;
        const Data_t  *v = a->v + (size_t)t * kvhd;
        const float   *cs = a->cos + (size_t)pp * half, *sn = a->sin + (size_t)pp * half;
        for (c = 0u; c < kvhd; c++) {
            const unsigned d = c % HD, b = c - d, g = c / HD;
            const size_t   o = ((size_t)g * a->C + pp) * HD + d;          /* group-major */
            double x = llm_ld(k[c], a->sk[c]);
            double r = d < half ? -llm_ld(k[b + d + half], a->sk[b + d + half])
                                :  llm_ld(k[b + d - half], a->sk[b + d - half]);
            a->ck[o] = llm_st16(x * (double)cs[d % half] + r * (double)sn[d % half], a->ick[c]);
            a->cv[llm_vrow(a->C, HD, a->VK, g, pp) + (size_t)d * a->VK] =
                llm_st16(llm_ld(v[c], a->sv[c]), a->icv[c]);
        }
    }
    a->n = n;
    if (n == 1u && llm_attn_decode(a)) {
        /* decode step: all host threads, bit-identical to the per-head code */
    } else if (n) {
        if (n > 1u)                  /* prefill: each key row serves n queries */
            host_parallel(llm_attn_convert, a, a->KV * (a->pos + n), 32u, 1u);
        host_parallel(llm_attn_items, a, n * a->H, 1u, 1u);
    }
    if (n < a->T)
        memset(a->pv + (size_t)n * a->H * HD, 0, (size_t)(a->T - n) * a->H * HD * sizeof(Data_t));
}

/* ---- FPGA prefill attention (policy pow2+sink+p12+mix) ----
 * keys = roundup(pos + n, q) keys (n clamped as in llm_attention; q = 16 x
 * the V cache interleave, C % q == 0): the runtime dimension of the q.K^T /
 * P.V ConvKernel calls (out_ch / in_ch). */
static unsigned llm_keys(unsigned pos, unsigned n, unsigned T, unsigned C, unsigned q)
{
    unsigned k;
    if (n > T) n = T;
    if (pos >= C) n = 0u;
    else if (n > C - pos) n = C - pos;
    k = (pos + n + q - 1u) / q * q;
    if (k > C) k = C;
    if (k < q) k = q;
    return k;
}

/* LlmAttnPrep: rows t < n -> the cache rows pos + t (RoPE(k0), v re-rounded,
 * group-major); RoPE(q0) rounded at the per-head q exponent -> the q.K^T conv
 * input image of every KV group, qx_g[c][kw*p + j] = q[t][h][(c/16)*16kw +
 * j*16 + c%16], p = (h % G) * T + t (0 for t >= n). */
typedef struct {
    const Data_t *q0, *k0, *v;
    const double *sq, *sk, *sv;
    int16_t      *ck, *cv;
    const double *ick, *icv, *iq;
    Data_t       *qx;
    const float  *cos, *sin;
    unsigned      pos, n, T, H, KV, HD, C, kw, VK, Q, keys;
} llm_prep_t;

static void llm_prep_kv_rows(void *p, unsigned t0, unsigned t1)
{
    const llm_prep_t *a = (const llm_prep_t *)p;
    const unsigned    HD = a->HD, half = HD / 2u, KV = a->KV;
    unsigned          t, g, d;
    for (t = t0; t < t1; t++) {
        const unsigned pp = a->pos + t;
        const float   *cs = a->cos + (size_t)pp * half, *sn = a->sin + (size_t)pp * half;
        for (g = 0u; g < KV; g++) {
            const size_t  o = (size_t)t * KV * HD + (size_t)g * HD;
            const Data_t *k = a->k0 + o, *v = a->v + o;
            const double *sk = a->sk + (size_t)g * HD, *sv = a->sv + (size_t)g * HD;
            const double *ick = a->ick + (size_t)g * HD, *icv = a->icv + (size_t)g * HD;
            int16_t      *kc = a->ck + ((size_t)g * a->C + pp) * HD;
            int16_t      *vc = a->cv + llm_vrow(a->C, HD, a->VK, g, pp);
            const unsigned vk = a->VK;
            double        x[LLM_MAX_HD];
            for (d = 0u; d < HD; d++)
                x[d] = llm_ld(k[d], sk[d]);
            for (d = 0u; d < half; d++)            /* rotate_half: -x[d + half], then x[d - half] */
                kc[d] = llm_st16(x[d] * (double)cs[d] + -x[d + half] * (double)sn[d], ick[d]);
            for (d = half; d < HD; d++)
                kc[d] = llm_st16(x[d] * (double)cs[d - half] + x[d - half] * (double)sn[d - half],
                                 ick[d]);
            for (d = 0u; d < HD; d++)
                vc[(size_t)d * vk] = llm_st16(llm_ld(v[d], sv[d]), icv[d]);
        }
    }
}

/* Work item: rows [tb * LLM_PREP_TB, +LLM_PREP_TB) of one head; the item's
 * rotated rows are written to the image as one contiguous run per image row
 * (nt * kw elements: whole cache lines, not scattered halfwords). */
#define LLM_PREP_TB 16u
static void llm_prep_q_items(void *p, unsigned i0, unsigned i1)
{
    const llm_prep_t *a = (const llm_prep_t *)p;
    const unsigned    HD = a->HD, half = HD / 2u, G = a->H / a->KV, T = a->T, GT = G * T;
    const unsigned    kw = a->kw, plane = GT * kw, nb = HD / (16u * kw);
    const unsigned    ntb = (T + LLM_PREP_TB - 1u) / LLM_PREP_TB;
    unsigned          it, d, b, j, l, u;
    for (it = i0; it < i1; it++) {
        const unsigned h = it / ntb, t0 = (it % ntb) * LLM_PREP_TB;
        const unsigned nt = t0 + LLM_PREP_TB < T ? LLM_PREP_TB : T - t0;
        int16_t        q[LLM_PREP_TB][LLM_MAX_HD];
        Data_t        *xg = a->qx + (size_t)(h / G) * HD * GT + (size_t)kw * ((h % G) * T + t0);
        for (u = 0u; u < nt; u++) {
            const unsigned t = t0 + u;
            if (t < a->n) {
                const unsigned pp = a->pos + t;
                const Data_t  *qr = a->q0 + (size_t)t * a->H * HD + (size_t)h * HD;
                const double  *sq = a->sq + (size_t)h * HD, iq = a->iq[h];
                const float   *cs = a->cos + (size_t)pp * half, *sn = a->sin + (size_t)pp * half;
                double         x[LLM_MAX_HD];
                for (d = 0u; d < HD; d++)
                    x[d] = llm_ld(qr[d], sq[d]);
                for (d = 0u; d < half; d++)
                    q[u][d] = llm_st16(x[d] * (double)cs[d] + -x[d + half] * (double)sn[d], iq);
                for (d = half; d < HD; d++)
                    q[u][d] = llm_st16(x[d] * (double)cs[d - half]
                                       + x[d - half] * (double)sn[d - half], iq);
            } else {
                for (d = 0u; d < HD; d++)
                    q[u][d] = 0;
            }
        }
        /* d = b*16kw + j*16 + l  ->  image row 16b + l, element kw*(p0 + u) + j */
        for (b = 0u; b < nb; b++)
            for (l = 0u; l < 16u; l++) {
                Data_t *row = xg + (size_t)(16u * b + l) * plane;
                for (u = 0u; u < nt; u++)
                    for (j = 0u; j < kw; j++)
                        row[kw * u + j] = (Data_t)q[u][b * 16u * kw + j * 16u + l];
            }
    }
}

static void llm_attn_prep(llm_prep_t *a)
{
    unsigned n = a->n;
    if (n > a->T) n = a->T;
    if (a->pos >= a->C) n = 0u;
    else if (n > a->C - a->pos) n = a->C - a->pos;
    a->n = n;
    host_parallel(llm_prep_kv_rows, a, n, 8u, 1u);
    host_parallel(llm_prep_q_items, a, a->H * ((a->T + LLM_PREP_TB - 1u) / LLM_PREP_TB), 1u, 1u);
    a->keys = llm_keys(a->pos, n, a->T, a->C, a->Q);
}

/* LlmAttnSoftmax: s [keys][G*T] raw scores (row stride G*T) -> P [G*T][keys]
 * raw at 2^-f_p (row stride keys, the P.V weight).  Column c = h' * T + t,
 * rows t < n: keys j <= pos + t, m = max raw, e_j = sexp_{f_s[h']}[m - raw_j],
 * sum left to right, P = round_half_even(e / sum * 2^f_p); other entries 0.
 * Work items are blocks of LLM_SMX_CB columns (one 64-byte line of a score
 * row), zig-zag ordered (light and heavy causal rows alternate) so that the
 * threads' ranges balance.  An item reads its columns once, transposed into
 * a stack buffer, then works column by column in cache.  e * (2^f_p / sum)
 * replaces the division unless the product lies within 1e-7 of a rounding
 * tie (both are within ~2^-40 of the exact quotient at |P| <= 2^f_p, so the
 * rounded integers agree elsewhere); then the exact expression is used.
 * Every column's sums keep their order: bits independent of the threads. */
#define LLM_SEXP_EMIN  (SEXP_EMIN_VALUE)
#define LLM_SEXP_NE    (SEXP_NE_VALUE)
#define LLM_SMX_CB     32u
static double *_llm_sexp_tab[LLM_SEXP_NE];

typedef struct {
    double *t;
    double  d, scale;
} llm_sexp_fill_t;

static void llm_sexp_fill(void *p, unsigned i0, unsigned i1)
{
    const llm_sexp_fill_t *a = (const llm_sexp_fill_t *)p;
    unsigned i;
    for (i = i0; i < i1; i++)
        a->t[i] = exp(-(double)i / a->d * a->scale);
}

static int llm_sexp_table(int f, double scale)
{
    llm_sexp_fill_t a;
    double        **t = &_llm_sexp_tab[f - LLM_SEXP_EMIN];
    if (*t) return 0;
    *t = (double *)malloc(65536u * sizeof(double));
    if (!*t) return -1;
    a.t = *t; a.d = ldexp(1.0, f); a.scale = scale;
    host_parallel(llm_sexp_fill, &a, 65536u, 1024u, 8u);
    return 0;
}

static void llm_sexp_free(int f)
{
    free(_llm_sexp_tab[f - LLM_SEXP_EMIN]);
    _llm_sexp_tab[f - LLM_SEXP_EMIN] = NULL;
}

typedef struct {
    const Data_t      *s;
    Data_t            *p;
    const signed char *fs;          /* score exponent per head of the group */
    const double      *ip;          /* 2^f_p per head of the group */
    unsigned           pos, n, T, G, C, Q, keys, nblk;
} llm_smx_t;

static void llm_smx_items(void *pp, unsigned i0, unsigned i1)
{
    const llm_smx_t *a = (const llm_smx_t *)pp;
    const unsigned   GT = a->G * a->T, keys = a->keys, nb = a->nblk;
    unsigned         it;
    for (it = i0; it < i1; it++) {
        const unsigned b = (it & 1u) ? nb - 1u - it / 2u : it / 2u;
        const unsigned c0 = b * LLM_SMX_CB, nc = c0 + LLM_SMX_CB < GT ? LLM_SMX_CB : GT - c0;
        unsigned       k, j, jend = 0u;
        for (k = 0u; k < nc; k++) {
            const unsigned t = (c0 + k) % a->T;
            if (t < a->n && a->pos + t + 1u > jend) jend = a->pos + t + 1u;
        }
        {
            int16_t tl[LLM_SMX_CB * (jend ? jend : 1u)];   /* the columns, transposed */
            double  eb[jend ? jend : 1u];
            for (j = 0u; j < jend; j++) {
                const int16_t *row = (const int16_t *)a->s + (size_t)j * GT + c0;
                for (k = 0u; k < nc; k++)
                    tl[(size_t)k * jend + j] = row[k];
            }
            for (k = 0u; k < nc; k++) {
                const unsigned c = c0 + k, t = c % a->T, hh = c / a->T;
                Data_t        *pr = a->p + (size_t)c * keys;
                const int16_t *r = tl + (size_t)k * jend;
                const double  *tab = _llm_sexp_tab[a->fs[hh] - LLM_SEXP_EMIN];
                const double   ip = a->ip[hh];
                unsigned       nk;
                int            m;
                double         sum = 0.0, rinv;
                if (t >= a->n) {
                    memset(pr, 0, (size_t)keys * sizeof(Data_t));
                    continue;
                }
                nk = a->pos + t + 1u;
                m = r[0];
                for (j = 1u; j < nk; j++)
                    m = r[j] > m ? r[j] : m;
                for (j = 0u; j < nk; j++) {
                    eb[j] = tab[m - r[j]];
                    sum += eb[j];
                }
                rinv = ip / sum;
                for (j = 0u; j < nk; j++) {
                    const double q = eb[j] * rinv, f = q - floor(q);
                    double       rq;
                    if (f > 0.5 - 1e-7 && f < 0.5 + 1e-7) {
                        pr[j] = llm_st(eb[j] / sum, ip);         /* near a tie: exact */
                        continue;
                    }
                    rq = nearbyint(q);
                    pr[j] = (Data_t)(int16_t)(rq > 32767.0 ? 32767.0 : rq);
                }
                memset(pr + nk, 0, (size_t)(keys - nk) * sizeof(Data_t));
            }
        }
    }
}

static void llm_attn_softmax(llm_smx_t *a)
{
    unsigned n = a->n;
    if (n > a->T) n = a->T;
    if (a->pos >= a->C) n = 0u;
    else if (n > a->C - a->pos) n = a->C - a->pos;
    a->n = n;
    a->keys = llm_keys(a->pos, n, a->T, a->C, a->Q);
    a->nblk = (a->G * a->T + LLM_SMX_CB - 1u) / LLM_SMX_CB;
    host_parallel(llm_smx_items, a, a->nblk, 1u, 1u);
}

/* LlmAttnMerge: o_g [G*T][HD] (row h' * T + t) -> pv [T][H*HD] (raw copy). */
static void llm_attn_merge(const Data_t *const *o, unsigned KV, unsigned G, unsigned T,
                           unsigned HD, Data_t *pv)
{
    const unsigned H = KV * G;
    unsigned       g, hh, t;
    for (g = 0u; g < KV; g++)
        for (hh = 0u; hh < G; hh++)
            for (t = 0u; t < T; t++)
                memcpy(pv + ((size_t)t * H + g * G + hh) * HD,
                       o[g] + ((size_t)hh * T + t) * HD, (size_t)HD * sizeof(Data_t));
}
"""


# DMA-state helpers (need inference_buf_t): emitted when a project has DMA states.
LLM_C_DMA = r"""
/* Flush (clean) rows [0, rows) of every group of a group-major [G][C][D]
 * DMA state — for a V cache interleaved by K the same prefix of rows * D
 * elements holds them when rows % 16K == 0 — the rows the next kernels
 * read (host writes of earlier calls, e.g. decode steps, stay dirty until
 * then). */
static void llm_cache_flush(inference_buf_t *cache, unsigned rows, unsigned G, unsigned C,
                            unsigned D)
{
    inference_buf_t v;
    unsigned        g;
    if (rows > C) rows = C;
    if (!rows) return;
    if (rows == C) {
        inference_buf_sync_to_device(cache);
        return;
    }
    for (g = 0u; g < G; g++) {
        inference_buf_init_view(&v, cache, g * C * D, rows * D);
        inference_buf_sync_to_device(&v);
    }
}
"""


def llm_c_helpers() -> str:
    return (LLM_C.replace("SILU_EMIN_VALUE", str(SILU_EMIN)).replace("SILU_NE_VALUE", str(SILU_NE))
            .replace("SEXP_EMIN_VALUE", str(SEXP_EMIN)).replace("SEXP_NE_VALUE", str(SEXP_NE)))


__all__ = ("LLM_DOMAIN", "LLM_OP_FACTORIES", "LlmNode", "LlmEmbedNode", "LlmResAddNode",
           "LlmRMSNormNode", "LlmSiluMulNode", "LlmAttentionNode", "LlmSelectRowNode",
           "LlmDequantNode", "LlmAttnPrepNode", "LlmAttnConvNode", "LlmAttnSoftmaxNode",
           "LlmAttnMergeNode", "RuntimeItem", "HostTable", "llm_c_helpers", "dot8", "rope",
           "silu_table", "sexp_table", "attn_keys", "v_row_base", "KEY_QUANTUM",
           "TABLE_FILE_BYTES",
           "LLM_C_DMA")
