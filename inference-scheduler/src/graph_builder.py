# Copyright 2026 Ivan Gubochkin
# SPDX-License-Identifier: Apache-2.0

"""
The ONNX plumbing the model frontends share (src/llama.py, vit.py, smolvla.py,
moonshine.py, piper.py — doc/plans/CHAT_PLAN.md §30): an entry graph's
tensors with their ``axi.numeric`` metadata (src/numeric.py: exponents,
host / DMA storage, persistent states), its float32 initializers, its nodes,
and the ModelProto with the entry's info record — plus the helpers of the
FPGA attention every frontend emits (the q image's kernel width, the order
of the per-group calls) and the exponent tables the calibrations write.

A frontend subclasses ``GraphBuilder``, calls ``_new()`` per entry, builds the
graph through ``_t`` / ``_init`` / ``_node`` / ``_matmul`` and returns
``_make_model()``.  The order of those calls is the graph's: tensors (value_info),
initializers (the weight pool's layout) and nodes keep the order they were
first named in.
"""

from __future__ import annotations

import json
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as nph
from onnx import TensorProto

from . import numeric
from .llm_nodes import LLM_DOMAIN

F = 8               # the kernels' output shift (floor(acc / 2^F)); ap_fixed<16,8>
RES_BITS = 15       # the int32 DMA residual's exponent (rowops; doc/plans/HOSTOPS_PLAN.md §5.1: r15)


def compact_exp(e):
    """An int when every channel has the same exponent, else a list."""
    e = np.asarray(e, np.int64).reshape(-1)
    return int(e[0]) if (e == e[0]).all() else [int(v) for v in e]


def graph_with_initializers(nodes, name, inputs, outputs, inits: Dict[str, np.ndarray],
                            value_info) -> onnx.GraphProto:
    """``oh.make_graph`` with ``inits`` ({name: array}) converted to float32
    TensorProtos one at a time, straight into the graph: one copy of each
    weight, not a second set of TensorProtos beside the graph's."""
    g = oh.make_graph(nodes, name, inputs, outputs, value_info=value_info)
    for n, arr in inits.items():
        g.initializer.append(nph.from_array(np.ascontiguousarray(arr, np.float32), n))
    return g


def entry_info(model: onnx.ModelProto, key: str = "axi.llm.entry") -> Optional[dict]:
    """The entry's info record (``GraphBuilder._make_model``'s ``info``), or None."""
    for p in model.metadata_props:
        if p.key == key:
            return json.loads(p.value)
    return None


class GraphBuilder:
    """One entry graph at a time.  ``NODE_DOMAIN``: the domain of ``_node``
    when none is given; ``PRODUCER`` / ``INFO_KEY``: the model's producer name
    and the metadata key of its info record."""

    NODE_DOMAIN = ""
    PRODUCER = "inference-scheduler/src/graph_builder.py"
    INFO_KEY = "axi.llm.entry"

    def _new(self, ename: Optional[str] = None):
        self._nodes: List[onnx.NodeProto] = []
        self._vi: Dict[str, onnx.ValueInfoProto] = {}
        self._inits: Dict[str, np.ndarray] = {}      # float32 when the graph is made (_make_model)
        self._int_inits: Dict[str, np.ndarray] = {}  # int64 (a Slice's starts / ends)
        self._meta = numeric.empty()
        self._e = ename

    def _t(self, name, shape, elem=TensorProto.FLOAT, exp=None, chexp=None, host=None, state=False,
           dma=None):
        """A tensor (its first shape wins) and its metadata: ``exp`` its exponents
        (scalar or per channel), ``chexp`` a Conv weight's per-channel ones, ``host``
        / ``dma`` its storage kind, ``state`` persistent across runs."""
        if name not in self._vi:
            self._vi[name] = oh.make_tensor_value_info(name, elem, list(shape))
        if exp is not None:
            self._meta["exp"][name] = compact_exp(exp)
        if chexp is not None:
            self._meta["chexp"][name] = [int(v) for v in np.asarray(chexp).reshape(-1)]
        if host is not None:
            self._meta["host"][name] = host
        if dma is not None:
            self._meta["dma"][name] = dma
        if state and name not in self._meta["state"]:
            self._meta["state"].append(name)
        return name

    def _init(self, name, arr):
        """A float32 initializer (the first one of a name wins)."""
        if name not in self._inits:
            self._inits[name] = arr
        return name

    def _int_init(self, name, arr):
        """An int64 initializer (after the float32 ones in the graph)."""
        if name not in self._int_inits:
            self._int_inits[name] = arr
        return name

    def _node(self, op, ins, outs, name, domain=None, **attrs):
        self._nodes.append(oh.make_node(op, ins, outs, name=name,
                                        domain=self.NODE_DOMAIN if domain is None else domain, **attrs))

    def _residual(self, name, rows, width, dma=True, state=False):
        """A residual-stream tensor [rows][width]: the int32 DMA tensor at
        2^-RES_BITS (the row-ops unit's adds, exact), else float32 host memory."""
        if dma:
            return self._t(name, [rows, width], exp=RES_BITS, dma="i32", state=state)
        return self._t(name, [rows, width], host="f32", state=state)

    def _matmul(self, x, wname, W, T, y, exp, name):
        """y [T][M] = x . W (W [K][M] the initializer ``wname``) at exponents ``exp``."""
        self._init(wname, W)
        self._t(y, [T, W.shape[1]], exp=exp)
        self._node("MatMul", [x, wname], [y], name, domain="")
        return y

    def _make_model(self, graph_name: str, inputs: Iterable[str], outputs: Iterable[str],
                    info: dict) -> onnx.ModelProto:
        """The entry's ModelProto: the graph, the numeric metadata, ``info`` under INFO_KEY."""
        inputs, outputs = list(inputs), list(outputs)
        io = set(inputs) | set(outputs)
        g = graph_with_initializers(
            self._nodes, graph_name, [self._vi[n] for n in inputs], [self._vi[n] for n in outputs],
            self._inits, value_info=[vi for n, vi in self._vi.items() if n not in io])
        for n, arr in self._int_inits.items():
            g.initializer.append(nph.from_array(np.asarray(arr, np.int64), n))
        self._inits, self._int_inits = {}, {}
        m = oh.make_model(g, opset_imports=[oh.make_opsetid("", 17), oh.make_opsetid(LLM_DOMAIN, 1)],
                          producer_name=self.PRODUCER)
        m.ir_version = 8
        p = m.metadata_props.add()
        p.key, p.value = numeric.METADATA_KEY, numeric.to_metadata(self._meta)
        p = m.metadata_props.add()
        p.key, p.value = self.INFO_KEY, json.dumps(info)
        return m


# ------------------------------------------------------------------ #
# Exponent tables                                                      #
# ------------------------------------------------------------------ #

def expand_exp(v, n: int, what: str = "exponents") -> np.ndarray:
    """int64 [n] of a calibrated value: a scalar broadcast, n values as they are,
    a per-head vector repeated over each head's n / heads channels."""
    v = np.asarray(v, np.int64).reshape(-1)
    if v.size == 1:
        return np.full(n, int(v[0]), np.int64)
    if v.size == n:
        return v.copy()
    if n % v.size:
        raise ValueError(f"formats: {what} has {v.size} values, {n} wanted")
    return np.repeat(v, n // v.size)


class ExponentTable:
    """A calibration's exponents per ``class@<layer key>`` (the formats JSON's
    "exponents"), its policy.  A class the calibration did not write is an
    error, not a default: a new tensor class (or a misspelled one) would
    otherwise run at 2^-F silently."""

    def __init__(self, d: dict):
        self.exp = dict(d["exponents"])
        self.policy = d.get("policy", "")

    @classmethod
    def from_file(cls, path: str, *args):
        """The table of a formats JSON file (``args``: the subclass's, e.g. the config)."""
        with open(path) as f:
            return cls(json.load(f), *args)

    def has(self, key: str) -> bool:
        return key in self.exp

    def lookup(self, key: str, n: int) -> np.ndarray:
        if key not in self.exp:
            raise KeyError(f"formats: no exponent for {key} (policy {self.policy!r})")
        return expand_exp(self.exp[key], n, key)


# ------------------------------------------------------------------ #
# FPGA attention                                                       #
# ------------------------------------------------------------------ #

def score_exp(fq: np.ndarray, fk: np.ndarray, groups: int) -> np.ndarray:
    """Per query head: the q.K^T scores' exponent f_q + f_k(its KV head) - F."""
    fq = np.asarray(fq, np.int64)
    return fq + np.asarray(fk, np.int64)[np.arange(fq.size) // (fq.size // groups)] - F


def pv_exp(fp: np.ndarray, fv: np.ndarray, groups: int, hd: int) -> np.ndarray:
    """[H * hd]: the P.V output's exponent per channel, f_p(head) + f_v(channel of
    its KV head) - F."""
    fp = np.asarray(fp, np.int64)
    grp = np.arange(fp.size) // (fp.size // groups)
    return (fp[:, None] + np.asarray(fv, np.int64).reshape(groups, hd)[grp] - F).reshape(-1)


def cheapest_qk_kw(hd: int, rows: int, keys: int) -> int:
    """The q image's kernel width for the q.K^T calls (the frontend writes the
    image, so any kw with hd % (16 kw) == 0 is free): the cheapest by
    cost_model.conv_cycles over ``rows`` query rows (heads x tokens of a group)
    and ``keys`` keys."""
    from .cost_model import conv_cycles
    best = None
    for kw in (1, 2, 4):
        if hd % (16 * kw):
            continue
        for ow in [d for d in range(8, 65) if rows % d == 0] or [rows]:
            cyc = conv_cycles(in_ch=hd // kw, out_ch=keys, in_h=rows // ow, in_w=kw * ow,
                              oh=rows // ow, ow=ow, kh=1, kw=kw, sw=kw)["total"]
            if best is None or cyc < best[0]:
                best = (cyc, kw)
    return best[1]


# The issue order of the per-group calls (q.K^T and P.V on ConvKernel, the
# softmax between them on the host or VectorOPKernel's unit).  ConvKernel runs
# one call at a time and the host issues them in order, so the order decides
# what each softmax hides behind:
#   "ahead1" (src/llama.py) — q.K^T 0, q.K^T 1, softmax 0, P.V 0, then per
#            group g >= 1 softmax g, q.K^T g+1, P.V g: softmax 0 hides behind the
#            short q.K^T 1, every later one behind the long P.V of the group before;
#   "ahead2" (src/vit.py) — q.K^T 0, q.K^T 1, then per group softmax g, P.V g,
#            q.K^T g+2: the CPU waits only for the short P.V to free the lane.
ORDERS = ("ahead1", "ahead2")


def group_order(groups: int, order: str) -> List[Tuple[str, int]]:
    """[(stage, group)], stage "qk" / "softmax" / "pv", in the issue order ``order``."""
    if order == "ahead1":
        out = [("qk", 0)] + ([("qk", 1)] if groups > 1 else []) + [("softmax", 0), ("pv", 0)]
        for g in range(1, groups):
            out.append(("softmax", g))
            if g + 1 < groups:
                out.append(("qk", g + 1))
            out.append(("pv", g))
        return out
    if order == "ahead2":
        out = [("qk", 0)] + ([("qk", 1)] if groups > 1 else [])
        for g in range(groups):
            out += [("softmax", g), ("pv", g)]
            if g + 2 < groups:
                out.append(("qk", g + 2))
        return out
    raise ValueError(f"attention order {order!r}: one of {', '.join(ORDERS)}")


__all__ = ("LLM_DOMAIN", "F", "RES_BITS", "GraphBuilder", "ExponentTable", "compact_exp", "expand_exp",
           "graph_with_initializers", "entry_info", "score_exp", "pv_exp", "cheapest_qk_kw",
           "group_order", "ORDERS")
