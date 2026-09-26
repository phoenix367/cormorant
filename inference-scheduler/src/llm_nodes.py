"""
Host-CPU ops of a Llama-family decoder (doc/CHAT_PLAN.md §3.2 B2, §10.5).

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
        return [t for t in self.inputs if not t.is_host]

    def staged_inputs(self):
        return self.dma_inputs()

    def state_inputs(self) -> List:
        return [t for t in self.inputs if t.is_state]

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
    """Causal GQA attention over a host KV cache, one float host region
    (policy xattn).  inputs = [q0, k0, v, pos, n?, cache_k, cache_v]; the
    caches ([C][KV*HD] raw int16, host states) receive the new rows."""
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
            sn._want(t, "i16", w)
            _require(t.is_state, node, f"{w} '{t.onnx_name}' must be a state")
        return sn

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
            f"    _a.H = {self.H}u; _a.KV = {self.KV}u; _a.HD = {self.HD}u; _a.C = {self.C}u;",
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
 * Rows t < n (pos + t < C): RoPE(k0[t]) -> K cache row pos + t at its
 * exponent, v[t] re-rounded -> V cache row.  The cache rows the queries read
 * (0 .. pos + n - 1) are converted once to their exact double values
 * (raw * 2^-f, the same bits the per-pair expression gives) into a scratch
 * when n > 1 (prefill; a decode step converts inline — one query per row).
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
    unsigned      pos, n, T, H, KV, HD, C;
    double        scale;
} llm_attn_t;

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

static void llm_attn_release(void)
{
    free(_llm_attn_kd);
    free(_llm_attn_vd);
    _llm_attn_kd = _llm_attn_vd = NULL;
    _llm_attn_cap = 0u;
}

#if defined(__GNUC__) && !defined(__clang__)
#  pragma GCC push_options
#  pragma GCC optimize ("tree-vectorize")    /* lane-wise only: bits unchanged */
#endif
static void llm_attn_convert(void *p, unsigned r0, unsigned r1)
{
    const llm_attn_t *a = (const llm_attn_t *)p;
    const unsigned    kvhd = a->KV * a->HD;
    unsigned          r, c;
    for (r = r0; r < r1; r++) {
        const int16_t *kr = a->ck + (size_t)r * kvhd, *vr = a->cv + (size_t)r * kvhd;
        double        *kd = _llm_attn_kd + (size_t)r * kvhd, *vd = _llm_attn_vd + (size_t)r * kvhd;
        for (c = 0u; c < kvhd; c++) {
            kd[c] = (double)kr[c] * a->sck[c];
            vd[c] = (double)vr[c] * a->scv[c];
        }
    }
}

static void llm_attn_items(void *p, unsigned i0, unsigned i1)
{
    const llm_attn_t *a = (const llm_attn_t *)p;
    const unsigned    HD = a->HD, half = HD / 2u, kvhd = a->KV * HD, grp = a->H / a->KV;
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
                const double *kr = _llm_attn_kd + (size_t)j * kvhd + (size_t)g * HD;
                for (l = 0u; l < 8u; l++)
                    acc[l] = q[l] * kr[l];
                for (d = 8u; d < HD; d += 8u)
                    for (l = 0u; l < 8u; l++)
                        acc[l] += q[d + l] * kr[d + l];
            } else {                                /* decode: the int16 cache row */
                const int16_t *kr = a->ck + (size_t)j * kvhd + (size_t)g * HD;
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
                const double *vr = _llm_attn_vd + (size_t)j * kvhd + (size_t)g * HD;
                if (j == 0u)
                    for (d = 0u; d < HD; d++)
                        o[d] = pj * vr[d];
                else
                    for (d = 0u; d < HD; d++)
                        o[d] += pj * vr[d];
            } else {
                const int16_t *vr = a->cv + (size_t)j * kvhd + (size_t)g * HD;
                const double  *sv = a->scv + (size_t)g * HD;
                if (j == 0u)
                    for (d = 0u; d < HD; d++)
                        o[d] = pj * ((double)vr[d] * sv[d]);
                else
                    for (d = 0u; d < HD; d++)
                        o[d] += pj * ((double)vr[d] * sv[d]);
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
        int16_t       *kc = a->ck + (size_t)pp * kvhd, *vc = a->cv + (size_t)pp * kvhd;
        for (c = 0u; c < kvhd; c++) {
            const unsigned d = c % HD, b = c - d;
            double x = llm_ld(k[c], a->sk[c]);
            double r = d < half ? -llm_ld(k[b + d + half], a->sk[b + d + half])
                                :  llm_ld(k[b + d - half], a->sk[b + d - half]);
            kc[c] = llm_st16(x * (double)cs[d % half] + r * (double)sn[d % half], a->ick[c]);
            vc[c] = llm_st16(llm_ld(v[c], a->sv[c]), a->icv[c]);
        }
    }
    a->n = n;
    if (n) {
        if (n > 1u)                  /* prefill: each key row serves n queries */
            host_parallel(llm_attn_convert, a, a->pos + n, 32u, 1u);
        host_parallel(llm_attn_items, a, n * a->H, 1u, 1u);
    }
    if (n < a->T)
        memset(a->pv + (size_t)n * a->H * HD, 0, (size_t)(a->T - n) * a->H * HD * sizeof(Data_t));
}
"""


def llm_c_helpers() -> str:
    return LLM_C.replace("SILU_EMIN_VALUE", str(SILU_EMIN)).replace("SILU_NE_VALUE", str(SILU_NE))


__all__ = ("LLM_DOMAIN", "LLM_OP_FACTORIES", "LlmNode", "LlmEmbedNode", "LlmResAddNode",
           "LlmRMSNormNode", "LlmSiluMulNode", "LlmAttentionNode", "LlmSelectRowNode",
           "LlmDequantNode", "RuntimeItem", "HostTable", "llm_c_helpers", "dot8", "rope",
           "silu_table", "TABLE_FILE_BYTES")
