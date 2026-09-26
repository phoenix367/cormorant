"""
Numeric annotations beyond the element type (doc/CHAT_PLAN.md §10.5).

A model may carry, in its ``metadata_props`` under the key ``axi.numeric``,
a JSON object

    {"exp":       {tensor: f | [f per last-axis channel]},
     "host":      {tensor: "f32" | "i32" | "i16"},
     "state":     [tensor, ...],
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

Models without the key are unaffected (every generated project of a model
without it is byte-identical to before).
"""

from __future__ import annotations

import json
from typing import Dict, List

import numpy as np

from .nodes import SchedulerError

METADATA_KEY = "axi.numeric"
HOST_KINDS = ("f32", "i32", "i16")


class NumericError(SchedulerError):
    pass


def empty() -> dict:
    return {"exp": {}, "host": {}, "state": [], "test_fill": {}}


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
    return any(meta[k] for k in ("exp", "host", "state"))


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
    for name in meta["state"]:
        t = _get(tensors, name, "state")
        t.is_state = True
        if t.data is not None:
            t.init_data = np.asarray(t.data, np.float64).reshape(t.shape)
            t.data = None
            t.packed_data = None
    for name in meta["test_fill"]:
        _get(tensors, name, "test_fill")


def _get(tensors, name, what):
    if name not in tensors:
        raise NumericError(f"{METADATA_KEY}.{what}: tensor '{name}' not in the graph")
    return tensors[name]


def has_numeric(t) -> bool:
    return t.exp is not None or t.host is not None or t.is_state or t.wexp is not None


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
        wexp = fo[None, :] + F - fa[:, None]                # [K][M]
        if b.onnx_name in done:
            if not np.array_equal(done[b.onnx_name], wexp):
                raise NumericError(f"weight '{b.onnx_name}' read by MatMuls with different "
                                   f"exponents")
            continue
        w = np.asarray(b.data, np.float64).reshape(-1, sn.k, sn.m)
        r = np.round(w * np.power(2.0, wexp)[None])
        sat[b.onnx_name] = int(((r < lo) | (r > hi)).sum())
        r = np.clip(r, lo, hi)
        b.data = (r / float(1 << F)).astype(np.float32).reshape(b.data.shape)
        b.wexp = wexp.copy()
        done[b.onnx_name] = wexp
    return sat


def check(nodes, dtype) -> None:
    """Only MatMuls (constant B) and the LLM host ops may touch tensors with
    exponents; only the LLM host ops touch host tensors and states."""
    from .nodes import MatmulNode, ReshapeNode
    from .llm_nodes import LlmNode
    for sn in nodes:
        label = f"{sn.onnx_node.op_type} node '{sn.onnx_node.name or sn.onnx_node.op_type}'"
        ts = list(sn.inputs) + [sn.output]
        if isinstance(sn, LlmNode):
            continue
        if any(t.is_host or t.is_state for t in ts):
            raise NumericError(f"{label}: host / state tensors are only read or written "
                               f"by the LLM host ops")
        if isinstance(sn, MatmulNode):
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
           "is_active", "apply", "has_numeric", "encode_matmul_weights", "check")
