"""
Numeric annotations beyond the element type (doc/plans/CHAT_PLAN.md §10.5).

A model may carry, in its ``metadata_props`` under the key ``axi.numeric``,
a JSON object

    {"exp":       {tensor: f | [f per last-axis channel]},
     "chexp":     {tensor: [f per channel (axis 1) of a 4-D Conv tensor]},
     "host":      {tensor: "f32" | "i32" | "i16"},
     "state":     [tensor, ...],
     "layout":    {tensor: [G, D]},
     "test_fill": {tensor: int}}

* ``exp`` — power-of-two exponents: a fixed-point tensor stores raw int16
  with value ``raw * 2^-f`` (f = 8 is ap_fixed<16,8> itself).  The kernels
  never see f: they multiply raw operands, sum exactly in ap_fixed<32,16>
  (an int32 raw sum that wraps) and write ``floor(acc / 2^F)`` saturated, F
  the element type's fractional bits.  So a MatMul keeps
  ``f_out[j] = f_in[i] + f_w[i][j] - F`` for every i, and a constant weight
  is encoded — round half to even, saturate — at the rank-1 exponent
  ``f_w[i][j] = f_out[j] + F - f_in[i]`` (``encode_matmul_weights``).
  Unannotated tensors keep the element type's exponent (F).
* ``chexp`` — one exponent per channel (axis 1) of a 4-D Conv tensor, which
  only Convs may read or write: a Conv writing it takes per-output-channel
  weight exponents ``f_w[m][c] = f_y[m] + F - f_x[c]`` (the rank-1 rule of a
  MatMul), and a Conv reading it per-input-channel ones — a depthwise 1 x 1
  Conv of weight 1.0 is then an exact per-channel rescale (floor) to one
  exponent (the stereo frontend's per-channel weights, STEREO_PLAN).
* ``host`` — the tensor lives in host memory, never in a DMA buffer:
  ``f32`` (float32; a transformer's residual stream), ``i32`` (token ids,
  positions: graph inputs become ``const int32_t *``), ``i16`` (raw int16 at
  its exponent, e.g. a KV cache only host ops read).  Only host ops
  (src/llm_nodes.py) read or write host tensors.
* ``state`` — persistent across inference_run() calls and shared by all
  entries of a multi-entry project (a KV cache, the last hidden row
  handed from a prefill entry to the head entry).  A state is an
  initializer (its initial VALUE; the C init image is the raw encoding of
  its non-zero prefix) or a node output (written in place by that node);
  host ops also update states in place (the KV cache rows).
* ``test_fill`` — the constant the generated test harness writes into an
  integer host input (e.g. a start position); ids default to ``i % rows``.
* ``layout`` — ``[G, D]`` or ``[G, D, K]``: a state of logical shape
  ``[R][G*D]`` stored group-major, ``[G][R][D]`` (``TensorInfo.group_layout``):
  a KV cache whose KV heads' rows must each be contiguous for the FPGA
  prefill attention; K > 1 interleaves each group's rows as a 1 x K conv
  input image (``TensorInfo.group_kw``, R % 16K == 0).  A state
  without a host kind is a DMA state: a persistent buffer in the CMA pool
  that host ops write (flushing what a kernel will read) and kernels read.

Models without the key are unaffected (every generated project of a model
without it is byte-identical to before).
"""

from __future__ import annotations

import json
from typing import Dict

import numpy as np

from .nodes import SchedulerError

METADATA_KEY = "axi.numeric"
HOST_KINDS = ("f32", "i32", "i16")


class NumericError(SchedulerError):
    pass


def empty() -> dict:
    return {"exp": {}, "chexp": {}, "host": {}, "state": [], "test_fill": {}, "layout": {}}


def parse(model) -> dict:
    """The ``axi.numeric`` annotations of an onnx.ModelProto (empty if none)."""
    meta = empty()
    for p in model.metadata_props:
        if p.key == METADATA_KEY:
            d = json.loads(p.value)
            for k in meta:
                if k in d:
                    meta[k] = d[k]
            unknown = set(d) - set(meta) - {"version", "note"}
            if unknown:
                raise NumericError(f"{METADATA_KEY}: unknown keys {sorted(unknown)}")
    return meta


def to_metadata(meta: dict) -> str:
    """JSON text for model.metadata_props[METADATA_KEY]."""
    out = {"version": 1}
    for k, v in meta.items():
        if v:
            out[k] = v
    return json.dumps(out, separators=(",", ":"))


def is_active(meta: dict) -> bool:
    return any(meta[k] for k in ("exp", "chexp", "host", "state"))


def apply(tensors: dict, meta: dict, dtype) -> None:
    """Set exp / host / is_state / init_data on the TensorInfos."""
    if is_active(meta):
        if getattr(dtype, "bytes_per_elem", 0) != 2:
            raise NumericError(f"{METADATA_KEY} requires a 16-bit fixed-point element type "
                               f"(got {dtype.name})")
        dtype.frac_bits  # noqa: B018 — raises for non-fixed-point types
    for name, kind in meta["host"].items():
        t = _get(tensors, name, "host")
        if kind not in HOST_KINDS:
            raise NumericError(f"host tensor '{name}': kind {kind!r} not in {HOST_KINDS}")
        t.host = kind
    for name, e in meta["exp"].items():
        t = _get(tensors, name, "exp")
        if t.host in ("f32", "i32"):
            raise NumericError(f"tensor '{name}': an exponent on a {t.host} host tensor")
        arr = np.asarray(e, np.int64)
        if arr.ndim > 1 or (arr.ndim == 1 and (not t.shape or arr.size != t.shape[-1])):
            raise NumericError(f"tensor '{name}': exponent must be an int or one int per "
                               f"last-axis channel ({t.shape[-1] if t.shape else 1}), "
                               f"got shape {list(arr.shape)}")
        if (arr < -30).any() or (arr > 40).any():
            raise NumericError(f"tensor '{name}': exponent out of range")
        t.exp = arr.copy()
    for name, e in meta.get("chexp", {}).items():
        t = _get(tensors, name, "chexp")
        arr = np.asarray(e, np.int64)
        if len(t.shape) != 4 or arr.ndim != 1 or arr.size != t.shape[1] or t.exp is not None or t.host:
            raise NumericError(f"tensor '{name}': chexp is one exponent per channel of a 4-D DMA "
                               f"tensor without exp (shape {t.shape}, got {list(arr.shape)})")
        if (arr < -30).any() or (arr > 40).any():
            raise NumericError(f"tensor '{name}': exponent out of range")
        t.chexp = arr.copy()
    for name in meta["state"]:
        t = _get(tensors, name, "state")
        t.is_state = True
        if t.data is not None:
            t.init_data = np.asarray(t.data, np.float64).reshape(t.shape)
            t.data = None
            t.packed_data = None
    for name in meta["test_fill"]:
        _get(tensors, name, "test_fill")
    for name, gd in meta.get("layout", {}).items():
        t = _get(tensors, name, "layout")
        if len(gd) not in (2, 3):
            raise NumericError(f"layout of '{name}': [G, D] or [G, D, K], got {gd}")
        g, d = int(gd[0]), int(gd[1])
        k = int(gd[2]) if len(gd) == 3 else 1
        if not t.is_state or len(t.shape) != 2 or t.shape[1] != g * d or g < 1 or d < 1:
            raise NumericError(f"layout of '{name}': a group-major layout [G={g}][R][D={d}] "
                               f"needs a 2-D state [R][G*D] (got shape {t.shape}, "
                               f"state {t.is_state})")
        if k < 1 or (k > 1 and t.shape[0] % (16 * k)):
            raise NumericError(f"layout of '{name}': interleave K={k} needs rows % 16K == 0 "
                               f"(rows {t.shape[0]})")
        t.group_layout = (g, d)
        t.group_kw = k


def _get(tensors, name, what):
    if name not in tensors:
        raise NumericError(f"{METADATA_KEY}.{what}: tensor '{name}' not in the graph")
    return tensors[name]


def has_numeric(t) -> bool:
    return (t.exp is not None or t.chexp is not None or t.host is not None or t.is_state
            or t.wexp is not None)


_ENCODE_BLOCK = 1 << 20          # weight elements per block of encode_matmul_weights


def encode_matmul_weights(nodes, dtype) -> Dict[str, int]:
    """Encode the constant B of every MatMul whose A or C has an exponent at
    the rank-1 weight exponent.  Returns {weight name: saturated count}."""
    from .nodes import MatmulNode
    F = dtype.frac_bits
    lo, hi = dtype.raw_range
    done: Dict[str, np.ndarray] = {}
    sat: Dict[str, int] = {}
    for sn in nodes:
        if not isinstance(sn, MatmulNode):
            continue
        a, b = sn.inputs
        c = sn.output
        if a.exp is None and c.exp is None and b.exp is None:
            continue
        if b.data is None:
            raise NumericError(
                f"MatMul '{sn.onnx_node.name}': exponents with a runtime B "
                f"('{b.onnx_name}') are not supported")
        if b.exp is not None:
            raise NumericError(f"weight '{b.onnx_name}': weight exponents are derived "
                               f"(f_w = f_out + {F} - f_in), not annotated")
        fa = a.exp_channels(F)                              # [K]
        fo = c.exp_channels(F)                              # [M]
        if fa.size != sn.k or fo.size != sn.m:
            raise NumericError(f"MatMul '{sn.onnx_node.name}': exponent sizes")
        rows, cols = F - fa.astype(np.int64), fo.astype(np.int64)
        if b.onnx_name in done:
            if not np.array_equal(done[b.onnx_name], np.add.outer(rows, cols)):
                raise NumericError(f"weight '{b.onnx_name}' read by MatMuls with different "
                                   f"exponents")
            continue
        # wexp[i][j] = f_out[j] + F - f_in[i], stored in the narrowest integer
        # type that holds it; the encoding runs in row blocks (float64
        # temporaries of a whole LM head would take GBs) — element-wise the
        # same arithmetic
        e_lo, e_hi = int(rows.min() + cols.min()), int(rows.max() + cols.max())
        etype = next(t for t in (np.int8, np.int16, np.int32, np.int64)
                     if np.iinfo(t).min <= e_lo and e_hi <= np.iinfo(t).max)
        wexp = np.empty((sn.k, sn.m), etype)                # [K][M]
        w = b.data.reshape(-1, sn.k, sn.m)
        out = np.empty(w.shape, np.float32)
        n_sat = 0
        step = max(1, _ENCODE_BLOCK // max(1, w.shape[0] * sn.m))
        for r0 in range(0, sn.k, step):
            e = np.add.outer(rows[r0:r0 + step], cols)
            wexp[r0:r0 + step] = e
            r = np.round(w[:, r0:r0 + step].astype(np.float64)
                         * np.power(2.0, e.astype(np.float64))[None])
            n_sat += int(((r < lo) | (r > hi)).sum())
            r = np.clip(r, lo, hi)
            out[:, r0:r0 + step] = (r / float(1 << F)).astype(np.float32)
        sat[b.onnx_name] = n_sat
        b.data = out.reshape(b.data.shape)
        b.wexp = wexp
        done[b.onnx_name] = wexp
    return sat


def encode_conv_weights(nodes, dtype) -> Dict[str, int]:
    """Encode the weight and bias of every Conv whose input or output has an
    exponent: ConvKernel sums raw int16 products exactly and writes
    floor(acc / 2^F) saturated, so the weight sits at f_w = f_y + F - f_x
    and the bias at f_y (the kernel seeds the accumulator with bias << F).
    ``data`` then holds raw / 2^F, so the packing and ROM paths emit the raw
    bits unchanged; ``wexp`` records the exponent (the simulator's value is
    data * 2^(F - wexp)).  The packed images are rebuilt.  Returns
    {weight / bias name: saturated count}."""
    from .nodes import ConvNode, _pack_conv_weight, _pad_conv_bias
    F = dtype.frac_bits
    lo, hi = dtype.raw_range
    sat: Dict[str, int] = {}
    done: Dict[str, int] = {}
    for sn in nodes:
        if not isinstance(sn, ConvNode):
            continue
        x, w, y = sn.inputs[0], sn.inputs[1], sn.output
        if x.exp is None and y.exp is None and x.chexp is None and y.chexp is None:
            continue
        if x.chexp is None and y.chexp is None:
            fx = int(np.asarray(F if x.exp is None else x.exp))
            fy = int(np.asarray(F if y.exp is None else y.exp))
            fw = fy + F - fx
            encoded = [(w, fw)] + ([(sn.inputs[2], fy)] if sn.has_bias else [])
        else:
            # per-channel exponents (chexp, axis 1): f_w[m][c] = f_y[m] + F - f_x[c]
            # (depthwise: f_w[m] = f_y[m] + F - f_x[m]), the bias at f_y[m]
            fx = (x.chexp if x.chexp is not None else
                  np.full(sn.in_ch, F if x.exp is None else int(np.asarray(x.exp)))).astype(np.int64)
            fy = (y.chexp if y.chexp is not None else
                  np.full(sn.out_ch, F if y.exp is None else int(np.asarray(y.exp)))).astype(np.int64)
            fw = (fy + F - fx)[:, None] if sn.is_depthwise else np.add.outer(fy + F, -fx)
            encoded = [(w, fw.reshape(fw.shape + (1, 1)))] + ([(sn.inputs[2], fy)] if sn.has_bias else [])
        for t, f in encoded:
            if t.onnx_name in done:
                if not np.array_equal(done[t.onnx_name], f):
                    raise NumericError(f"'{t.onnx_name}' read by Convs with different exponents")
                continue
            fa = np.asarray(f, np.int64)
            r = np.round(np.asarray(t.data, np.float64).reshape(t.shape) * np.power(2.0, fa))
            sat[t.onnx_name] = int(((r < lo) | (r > hi)).sum())
            t.data = (np.clip(r, lo, hi) / float(1 << F)).astype(np.float32).reshape(t.data.shape)
            t.wexp = (np.asarray(f, np.int8 if -128 <= f <= 127 else np.int64) if fa.ndim == 0
                      else fa.astype(np.int8))
            done[t.onnx_name] = f
        _pack_conv_weight(w, sn.out_ch, sn.in_ch, sn.kh, sn.kw, sn.is_depthwise)
        if sn.has_bias:
            _pad_conv_bias(sn.inputs[2], sn.out_ch)
    return sat


def check(nodes, dtype) -> None:
    """Only MatMuls (constant B), Convs (per-tensor exponents; constant
    weight and bias, encoded by encode_conv_weights), Concat / Transpose
    (raw copies: one exponent on every input and the output) and the LLM /
    TTS / stereo ops (host ops and the FPGA kernel calls) may touch tensors
    with exponents; only those host ops touch host tensors and states."""
    from .nodes import ConvNode, MatmulNode, ReshapeNode
    from .host_nodes import ConcatNode, TransposeNode
    from .llm_nodes import LlmNode
    for sn in nodes:
        label = f"{sn.onnx_node.op_type} node '{sn.onnx_node.name or sn.onnx_node.op_type}'"
        ts = list(sn.inputs) + [sn.output]
        if isinstance(sn, LlmNode) or getattr(sn, "is_llm_op", False):
            continue
        if any(t.is_host or t.is_state for t in ts):
            raise NumericError(f"{label}: host / state tensors are only read or written "
                               f"by the LLM host ops")
        if any(t.chexp is not None for t in ts) and not isinstance(sn, ConvNode):
            raise NumericError(f"{label}: per-channel (chexp) tensors are only read or written by Convs")
        if isinstance(sn, MatmulNode):
            continue
        if isinstance(sn, ConvNode):
            x, w = sn.inputs[0], sn.inputs[1]
            b = sn.inputs[2] if sn.has_bias else None
            if w.exp is not None or (b is not None and b.exp is not None):
                raise NumericError(f"{label}: weight / bias exponents are derived "
                                   f"(f_w = f_y + {dtype.frac_bits} - f_x, bias at f_y), "
                                   f"not annotated")
            if (x.exp is not None or sn.output.exp is not None or x.chexp is not None
                    or sn.output.chexp is not None) and (w.data is None or (b is not None and b.data is None)):
                raise NumericError(f"{label}: exponents need a constant weight and bias")
            for t in (x, sn.output):
                if t.exp is not None and np.ndim(t.exp) != 0:
                    raise NumericError(f"{label}: '{t.onnx_name}': a Conv tensor takes one "
                                       f"exponent (per-channel exponents are along the last "
                                       f"axis, which is not a Conv's channel axis)")
            continue
        if isinstance(sn, (ConcatNode, TransposeNode)) and any(t.exp is not None for t in ts):
            # raw copies: right for any exponent the inputs and the output share
            es = {None if t.exp is None or np.ndim(t.exp) else int(np.asarray(t.exp)) for t in ts}
            if len(es) != 1 or None in es:
                raise NumericError(f"{label}: a copy's inputs and output must carry one and the same "
                                   f"per-tensor exponent")
            continue
        if isinstance(sn, ReshapeNode):
            src, out = sn.inputs[0], sn.output
            if src.exp is not None or out.exp is not None:
                se = src.exp_channels(dtype.frac_bits) if src.shape else None
                oe = out.exp_channels(dtype.frac_bits) if out.shape else None
                same_last = src.shape and out.shape and src.shape[-1] == out.shape[-1]
                if not ((np.ndim(src.exp) == 0 and np.ndim(out.exp) == 0
                         and np.array_equal(np.asarray(src.exp), np.asarray(out.exp)))
                        or (same_last and np.array_equal(se, oe))):
                    raise NumericError(f"{label}: '{src.onnx_name}' and '{out.onnx_name}' "
                                       f"must carry the same exponents")
            continue
        if any(t.exp is not None for t in ts):
            raise NumericError(f"{label}: tensors with power-of-two exponents are only "
                               f"supported on MatMul and the LLM host ops")


__all__ = ("METADATA_KEY", "HOST_KINDS", "NumericError", "empty", "parse", "to_metadata",
           "is_active", "apply", "has_numeric", "encode_matmul_weights", "encode_conv_weights",
           "check")
