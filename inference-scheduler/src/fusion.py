"""
ONNX-level lowering and pattern-fusion passes (BERT_PLAN phase 1b / 1c).

All passes mutate ``model.graph`` in place (the initializer list is never
copied, so a 435 MB BERT stays cheap) and return how many sites they
rewrote.  They run in ``OnnxGraph.__init__`` after shape inference and the
Gemm decomposition, before tensor registration, so every later stage sees
an ordinary graph.

Always on (they only touch ops that were unsupported before):

  * ``fold_constant_nodes`` — ``Constant`` nodes become initializers.
  * ``lower_split``         — ``Split`` becomes one ``Slice`` per output
                              (host copy, or a zero-cost view when the
                              piece is contiguous, see ``SliceNode``).

``fuse_patterns`` (``OnnxGraph(fuse_patterns=True)``, the library and CLI
default; ``--no-fuse-patterns``) — changes only graphs that contain these
patterns:

  * TensorFlow / BERT LayerNorm subgraph  -> ``LayerNormalization``
    (host op; its x^2 intermediates would saturate Q8.8 op by op).
  * GELU, tanh approximation (BERT) and erf form (PyTorch export)
                                          -> ``Gelu`` (host op; x^3 would
    saturate).
  * Constant-operand broadcast normalisation for the VectorOP kernel:
    a scalar constant on a tensor whose last dim L is a multiple of the
    AXI alignment (<= 2048) becomes an [L] vector (a repeating chunk of L
    instead of L one-element chunks at stride 8); when the runtime operand
    is itself broadcast (the kernel repeats one side only) a constant the
    kernel cannot broadcast is pre-broadcast to the output shape (BERT's
    ``ones[1,S,1] * mask[1,1,S]``).
    Values are unchanged, so results are identical.

Matching is structural — producer / consumer links, op types and constant
values (with a tolerance for the transcendental constants) — never node
names.  Every intermediate of a matched pattern must have no consumer
outside the pattern and must not be a graph output; a near miss is left
alone, and its ReduceMean / Pow / Sqrt / Reciprocal / Tanh / Erf then fail
node dispatch with a pointer to this module (lowering them op by op is
numerically wrong in Q8.8).
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as nph

from .nodes import SchedulerError, _broadcast_info
from .tensor import TensorInfo

# Ops that exist only inside a fusable pattern — the dispatch error for an
# unmatched instance points here.
PATTERN_ONLY_OPS = frozenset({"ReduceMean", "Pow", "Sqrt", "Reciprocal", "Tanh", "Erf"})

GELU_C1 = 0.044715
GELU_C2 = math.sqrt(2.0 / math.pi)
_RTOL = 1e-5                      # tolerance on c1 / c2 / sqrt(2) constants

# Largest constant the broadcast normalisation materialises (elements).
_MAX_PREBROADCAST = 1 << 22
# VectorOP replays a stride-0 operand of up to this many elements on chip.
_VECTOROP_REPLAY_MAX = 2048


def default_opset(model: onnx.ModelProto) -> int:
    for o in model.opset_import:
        if o.domain in ("", "ai.onnx"):
            return int(o.version)
    return 1


# ------------------------------------------------------------------ #
# Graph index                                                          #
# ------------------------------------------------------------------ #

class _Index:
    """Producer / consumer / constant lookups over a GraphProto."""

    def __init__(self, graph: onnx.GraphProto) -> None:
        self.graph = graph
        self.nodes: List[onnx.NodeProto] = list(graph.node)
        self.producer: Dict[str, onnx.NodeProto] = {}
        self.consumers: Dict[str, List[onnx.NodeProto]] = defaultdict(list)
        for n in self.nodes:
            for o in n.output:
                if o:
                    self.producer[o] = n
            seen = set()
            for i in n.input:
                if i and i not in seen:
                    seen.add(i)
                    self.consumers[i].append(n)
        self.inits = {t.name: t for t in graph.initializer}
        self.outputs = {o.name for o in graph.output}
        self.shapes: Dict[str, List[int]] = {}
        for t in graph.initializer:
            self.shapes[t.name] = [int(d) for d in t.dims]
        for vi in list(graph.input) + list(graph.value_info) + list(graph.output):
            dims = []
            for d in vi.type.tensor_type.shape.dim:
                dims.append(int(d.dim_value) if d.HasField("dim_value") else 0)
            if vi.type.tensor_type.HasField("shape"):
                self.shapes[vi.name] = dims
        self._cache: Dict[str, np.ndarray] = {}

    def const(self, name: str, max_numel: int = 1 << 16) -> Optional[np.ndarray]:
        if name in self._cache:
            return self._cache[name]
        t = self.inits.get(name)
        if t is None or int(np.prod(t.dims, dtype=np.int64)) > max_numel:
            return None
        arr = nph.to_array(t)
        self._cache[name] = arr
        return arr

    def scalar(self, name: str) -> Optional[float]:
        arr = self.const(name)
        if arr is None or arr.size != 1 or arr.dtype.kind not in "fiu":
            return None
        return float(arr.reshape(-1)[0])

    def sole(self, name: str) -> Optional[onnx.NodeProto]:
        """The only consumer of an internal (non-output) tensor, else None."""
        if name in self.outputs:
            return None
        cons = self.consumers.get(name, [])
        return cons[0] if len(cons) == 1 else None


def _other(node: onnx.NodeProto, name: str) -> Optional[str]:
    """The operand of a binary node that is not ``name`` (None if absent)."""
    ins = list(node.input[:2])
    if len(ins) != 2 or name not in ins:
        return None
    return ins[1] if ins[0] == name else ins[0]


def _is(node: Optional[onnx.NodeProto], op: str) -> bool:
    return node is not None and node.op_type == op


def _close(v: Optional[float], ref: float, rtol: float = _RTOL) -> bool:
    return v is not None and abs(v - ref) <= rtol * abs(ref)


def _mul_by_const(ix: _Index, node: Optional[onnx.NodeProto], value: float,
                  name: str) -> bool:
    """``node`` is Mul(name, c) / Mul(c, name) with scalar constant c == value."""
    if not _is(node, "Mul"):
        return False
    o = _other(node, name)
    return o is not None and ix.scalar(o) == value


def _reduce_last_axis(ix: _Index, node: onnx.NodeProto, rank: int) -> bool:
    attrs = {a.name: oh.get_attribute_value(a) for a in node.attribute}
    if int(attrs.get("keepdims", 1)) != 1:
        return False
    axes = attrs.get("axes")
    if axes is None and len(node.input) > 1 and node.input[1]:
        arr = ix.const(node.input[1])
        axes = None if arr is None else [int(v) for v in arr.reshape(-1)]
    if axes is None or len(axes) != 1:
        return False
    return axes[0] in (-1, rank - 1)


def _common_prefix(names: List[str], fallback: str) -> str:
    names = [n for n in names if n]
    if not names:
        return fallback
    pre = names[0]
    for n in names[1:]:
        while not n.startswith(pre):
            pre = pre[:-1]
    pre = pre[:pre.rfind("/")] if "/" in pre else ""
    return pre or fallback


def _apply(ix: _Index, matches: List[dict]) -> None:
    """Replace every match's nodes by its fused node, placed where the
    pattern's final node was."""
    removed = {id(n) for m in matches for n in m["nodes"]}
    anchor = {id(m["anchor"]): m["fused"] for m in matches}
    new_nodes = []
    for n in ix.nodes:
        if id(n) in anchor:
            new_nodes.append(anchor[id(n)])
        elif id(n) not in removed:
            new_nodes.append(n)
    del ix.graph.node[:]
    ix.graph.node.extend(new_nodes)


# ------------------------------------------------------------------ #
# Always-on lowerings                                                  #
# ------------------------------------------------------------------ #

def fold_constant_nodes(model: onnx.ModelProto) -> int:
    """``Constant`` nodes -> initializers (value / value_float(s) / value_int(s))."""
    g = model.graph
    keep, count = [], 0
    for n in g.node:
        if n.op_type != "Constant" or len(n.output) != 1:
            keep.append(n)
            continue
        a = {x.name: x for x in n.attribute}
        if "value" in a:
            t = onnx.TensorProto()
            t.CopyFrom(a["value"].t)
            t.name = n.output[0]
        elif "value_float" in a:
            t = nph.from_array(np.array(a["value_float"].f, np.float32), n.output[0])
        elif "value_floats" in a:
            t = nph.from_array(np.array(list(a["value_floats"].floats), np.float32), n.output[0])
        elif "value_int" in a:
            t = nph.from_array(np.array(a["value_int"].i, np.int64), n.output[0])
        elif "value_ints" in a:
            t = nph.from_array(np.array(list(a["value_ints"].ints), np.int64), n.output[0])
        else:
            keep.append(n)
            continue
        g.initializer.append(t)
        count += 1
    if count:
        del g.node[:]
        g.node.extend(keep)
    return count


def lower_split(model: onnx.ModelProto) -> int:
    """``Split`` -> one ``Slice`` per (non-empty) output, in place.

    Split sizes come from the ``split`` attribute (opset < 13), the
    ``split`` input (opset >= 13), ``num_outputs`` (opset 18: ceil-sized
    pieces, the last one shorter) or an equal split.
    """
    g = model.graph
    ix = _Index(g)
    new_nodes, count = [], 0
    for n in ix.nodes:
        if n.op_type != "Split":
            new_nodes.append(n)
            continue
        x = n.input[0]
        shape = ix.shapes.get(x)
        a = {x_.name: oh.get_attribute_value(x_) for x_ in n.attribute}
        if not shape or min(shape) <= 0:
            raise SchedulerError(f"Split node '{n.name or 'Split'}': input shape unknown.")
        rank = len(shape)
        axis = int(a.get("axis", 0))
        axis = axis + rank if axis < 0 else axis
        dim = shape[axis]
        n_out = len(n.output)
        sizes = a.get("split")
        if sizes is None and len(n.input) > 1 and n.input[1]:
            arr = ix.const(n.input[1])
            sizes = None if arr is None else [int(v) for v in arr.reshape(-1)]
        if sizes is None:
            if "num_outputs" in a:
                k = -(-dim // n_out)
                sizes = [k] * (n_out - 1) + [dim - k * (n_out - 1)]
            else:
                sizes = [dim // n_out] * n_out
        sizes = [int(s) for s in sizes]
        if len(sizes) != n_out or sum(sizes) != dim:
            raise SchedulerError(f"Split node '{n.name or 'Split'}': sizes {sizes} do not "
                                 f"cover dim {dim} with {n_out} outputs.")
        begin = 0
        for i, (out, sz) in enumerate(zip(n.output, sizes, strict=True)):
            if out:
                base = f"{out}__slice"
                for suffix, val in (("starts", begin), ("ends", begin + sz), ("axes", axis)):
                    g.initializer.append(nph.from_array(np.array([val], np.int64),
                                                        f"{base}_{suffix}"))
                new_nodes.append(oh.make_node(
                    "Slice", [x, f"{base}_starts", f"{base}_ends", f"{base}_axes"], [out],
                    name=f"{n.name or 'Split'}/slice{i}"))
            begin += sz
        count += 1
    if count:
        del g.node[:]
        g.node.extend(new_nodes)
    return count


# ------------------------------------------------------------------ #
# LayerNorm (TensorFlow / BERT form)                                   #
# ------------------------------------------------------------------ #

def _match_tf_layernorm(ix: _Index, mean: onnx.NodeProto) -> Optional[dict]:
    """
        mean = ReduceMean(x, last axis, keepdims)
        d    = Sub(x, mean)              ("SquaredDifference")
        dd   = Mul(d, d) | Pow(d, 2)
        var  = ReduceMean(dd, last axis, keepdims)
        ve   = Add(var, eps)
        sd   = Sqrt(ve)
        inv  = Reciprocal(sd) | Div(1, sd)
        g    = Mul(inv, gamma)
        mg   = Mul(mean, g)
        b    = Sub(beta, mg)
        xg   = Mul(x, g)
        y    = Add(xg, b)                (commutative operands in any order)
    """
    if mean.op_type != "ReduceMean" or not mean.input:
        return None
    x, mo = mean.input[0], mean.output[0]
    shape = ix.shapes.get(x) or []
    rank = len(shape)
    if rank == 0 or shape[-1] <= 0 or not _reduce_last_axis(ix, mean, rank):
        return None
    n = shape[-1]
    if mo in ix.outputs:
        return None
    cons = ix.consumers.get(mo, [])
    if len(cons) != 2:
        return None
    d = next((c for c in cons if c.op_type == "Sub" and list(c.input[:2]) == [x, mo]), None)
    mg = next((c for c in cons if c is not d and c.op_type == "Mul"), None)
    if d is None or mg is None:
        return None
    dd = ix.sole(d.output[0])
    if not (_is(dd, "Mul") and list(dd.input[:2]) == [d.output[0]] * 2) and \
       not (_is(dd, "Pow") and dd.input[0] == d.output[0] and ix.scalar(dd.input[1]) == 2.0):
        return None
    var = ix.sole(dd.output[0])
    if not _is(var, "ReduceMean") or not _reduce_last_axis(ix, var, rank):
        return None
    ve = ix.sole(var.output[0])
    if not _is(ve, "Add"):
        return None
    eps = ix.scalar(_other(ve, var.output[0]) or "")
    if eps is None or eps < 0:
        return None
    sd = ix.sole(ve.output[0])
    if not _is(sd, "Sqrt"):
        return None
    inv = ix.sole(sd.output[0])
    if not (_is(inv, "Reciprocal") or
            (_is(inv, "Div") and list(inv.input[1:2]) == [sd.output[0]]
             and ix.scalar(inv.input[0]) == 1.0)):
        return None
    g = ix.sole(inv.output[0])
    if not _is(g, "Mul"):
        return None
    gamma = _other(g, inv.output[0])
    go = g.output[0]
    if go in ix.outputs or len(ix.consumers.get(go, [])) != 2:
        return None
    if mg not in ix.consumers[go] or _other(mg, mo) != go:
        return None
    xg = next(c for c in ix.consumers[go] if c is not mg)
    if not _is(xg, "Mul") or _other(xg, go) != x:
        return None
    b = ix.sole(mg.output[0])
    if not _is(b, "Sub") or list(b.input[1:2]) != [mg.output[0]]:
        return None
    beta = b.input[0]
    y = ix.sole(b.output[0])
    if not _is(y, "Add") or ix.sole(xg.output[0]) is not y or _other(y, xg.output[0]) != b.output[0]:
        return None
    for name in (gamma, beta):
        arr = ix.const(name)
        if arr is None or arr.dtype.kind != "f" or arr.size != n or \
           any(s != 1 for s in arr.shape[:-1]):
            return None
    nodes = [mean, d, dd, var, ve, sd, inv, g, mg, b, xg, y]
    pre = _common_prefix([m.name for m in nodes], "")
    fused = oh.make_node(
        "LayerNormalization", [x, gamma, beta], [y.output[0]],
        name=pre or f"{y.name or 'LayerNorm'}/fused",
        axis=-1, epsilon=eps, axi_tf_form=1, axi_epsilon=repr(eps))
    return dict(nodes=nodes, anchor=y, fused=fused)


# ------------------------------------------------------------------ #
# GELU                                                                 #
# ------------------------------------------------------------------ #

def _match_gelu_tail(ix: _Index, x: str, a1: onnx.NodeProto):
    """``y = x * 0.5 * a1`` in one of the three association orders.
    Returns (y_node, tail_nodes) or None.  (Scaling by 0.5 is exact, so all
    three orders give the same double result.)"""
    c = ix.sole(a1.output[0])
    if not _is(c, "Mul"):
        return None
    u = _other(c, a1.output[0])
    if u is None:
        return None
    if ix.scalar(u) == 0.5:                              # m2 = 0.5 * a1; y = x * m2
        y = ix.sole(c.output[0])
        if _is(y, "Mul") and _other(y, c.output[0]) == x:
            return y, [c, y]
        return None
    if u == x:                                           # q = x * a1; y = q * 0.5
        y = ix.sole(c.output[0])
        if _mul_by_const(ix, y, 0.5, c.output[0]):
            return y, [c, y]
        return None
    h = ix.producer.get(u)                               # h = x * 0.5; y = h * a1
    if _mul_by_const(ix, h, 0.5, x) and ix.sole(u) is c:
        return c, [h, c]
    return None


def _match_gelu_tanh(ix: _Index, t: onnx.NodeProto) -> Optional[dict]:
    """
        p  = Pow(x, 3) | Mul(x, Mul(x, x)) | Mul(Mul(x, x), x)
        m  = Mul(c1 ~ 0.044715, p)
        a  = Add(x, m)
        m1 = Mul(c2 ~ sqrt(2/pi), a)
        t  = Tanh(m1)
        a1 = Add(1, t)
        y  = x * 0.5 * a1                (see _match_gelu_tail)
    """
    m1 = ix.producer.get(t.input[0])
    if not _is(m1, "Mul") or ix.sole(m1.output[0]) is not t:
        return None
    a_name = next((i for i in m1.input[:2] if ix.scalar(i) is None), None)
    c2 = ix.scalar(_other(m1, a_name) or "") if a_name else None
    if not _close(c2, GELU_C2):
        return None
    a = ix.producer.get(a_name)
    if not _is(a, "Add") or ix.sole(a.output[0]) is not m1:
        return None
    for x, m_name in ((a.input[0], a.input[1]), (a.input[1], a.input[0])):
        m = ix.producer.get(m_name)
        if not _is(m, "Mul") or ix.sole(m_name) is not a:
            continue
        p_name = next((i for i in m.input[:2] if ix.scalar(i) is None), None)
        c1 = ix.scalar(_other(m, p_name) or "") if p_name else None
        if not _close(c1, GELU_C1):
            continue
        p = ix.producer.get(p_name)
        if ix.sole(p_name) is not m:
            continue
        cube: List[onnx.NodeProto] = []
        if _is(p, "Pow") and p.input[0] == x and ix.scalar(p.input[1]) == 3.0:
            cube = [p]
        elif _is(p, "Mul") and x in p.input[:2]:
            sq = ix.producer.get(_other(p, x) or "")
            if _is(sq, "Mul") and list(sq.input[:2]) == [x, x] and ix.sole(sq.output[0]) is p:
                cube = [sq, p]
        if not cube:
            continue
        a1 = ix.sole(t.output[0])
        if not _is(a1, "Add") or ix.scalar(_other(a1, t.output[0]) or "") != 1.0:
            return None
        tail = _match_gelu_tail(ix, x, a1)
        if tail is None:
            return None
        y, tail_nodes = tail
        nodes = cube + [m, a, m1, t, a1] + tail_nodes
        pre = _common_prefix([k.name for k in nodes], "")
        fused = oh.make_node("Gelu", [x], [y.output[0]],
                             name=pre or f"{y.name or 'Gelu'}/fused",
                             approximate="tanh", axi_c1=repr(c1), axi_c2=repr(c2))
        return dict(nodes=nodes, anchor=y, fused=fused)
    return None


def _match_gelu_erf(ix: _Index, e: onnx.NodeProto) -> Optional[dict]:
    """
        u  = Div(x, k ~ sqrt(2)) | Mul(x, k ~ 1/sqrt(2))
        e  = Erf(u)
        a1 = Add(e, 1)
        y  = x * 0.5 * a1                (see _match_gelu_tail)
    """
    u = ix.producer.get(e.input[0])
    if u is None or ix.sole(u.output[0]) is not e:
        return None
    if _is(u, "Div") and _close(ix.scalar(u.input[1]), math.sqrt(2.0)):
        x, k, div = u.input[0], ix.scalar(u.input[1]), 1
    elif _is(u, "Mul"):
        x = next((i for i in u.input[:2] if ix.scalar(i) is None), None)
        k = ix.scalar(_other(u, x) or "") if x else None
        if not _close(k, 1.0 / math.sqrt(2.0)):
            return None
        div = 0
    else:
        return None
    a1 = ix.sole(e.output[0])
    if not _is(a1, "Add") or ix.scalar(_other(a1, e.output[0]) or "") != 1.0:
        return None
    tail = _match_gelu_tail(ix, x, a1)
    if tail is None:
        return None
    y, tail_nodes = tail
    nodes = [u, e, a1] + tail_nodes
    pre = _common_prefix([k_.name for k_ in nodes], "")
    fused = oh.make_node("Gelu", [x], [y.output[0]], name=pre or f"{y.name or 'Gelu'}/fused",
                         approximate="none", axi_k=repr(k), axi_div=div)
    return dict(nodes=nodes, anchor=y, fused=fused)


# ------------------------------------------------------------------ #
# Constant broadcast normalisation (VectorOP kernel)                    #
# ------------------------------------------------------------------ #

def _vectorop_can_broadcast(a_shape, b_shape, out_shape, align_elems) -> bool:
    out = TensorInfo("out", list(out_shape), "float32")
    try:
        _, _, _, a_adv = _broadcast_info(TensorInfo("a", list(a_shape), "float32"), out, align_elems)
        _, _, _, b_adv = _broadcast_info(TensorInfo("b", list(b_shape), "float32"), out, align_elems)
    except SchedulerError:
        return False
    return a_adv or b_adv


def _normalise_const_broadcast(ix: _Index, align_elems: int) -> int:
    count = 0
    made: Dict[tuple, str] = {}
    for n in ix.nodes:
        if n.op_type not in ("Add", "Sub", "Mul", "Div") or len(n.input) != 2:
            continue
        out = ix.shapes.get(n.output[0])
        if not out or min(out) <= 0:
            continue
        is_c = [i in ix.inits for i in n.input]
        if is_c.count(True) != 1:
            continue
        ci = is_c.index(True)
        cname = n.input[ci]
        other_shape = ix.shapes.get(n.input[1 - ci])
        if other_shape is None:
            continue
        cshape = [int(d) for d in ix.inits[cname].dims]
        numel = int(np.prod(out, dtype=np.int64))
        csize = int(np.prod(cshape, dtype=np.int64))
        L = out[-1]
        if csize == 1 and L % align_elems == 0 and L <= _VECTOROP_REPLAY_MAX and numel > L:
            new_shape = [L]
        else:
            # Pre-broadcast only when the RUNTIME operand is itself smaller
            # than the output (the kernel lets one side repeat, never both):
            # BERT's ones[1,S,1] * mask[1,1,S].  A single interleaved
            # broadcast against a full-size tensor stays an error.
            if int(np.prod(other_shape, dtype=np.int64)) >= numel:
                continue
            shapes = [None, None]
            shapes[ci], shapes[1 - ci] = cshape, other_shape
            if _vectorop_can_broadcast(shapes[0], shapes[1], out, align_elems):
                continue
            if numel > _MAX_PREBROADCAST:
                continue
            shapes[ci] = list(out)
            if not _vectorop_can_broadcast(shapes[0], shapes[1], out, align_elems):
                continue                              # still unsupported: leave the error
            new_shape = list(out)
        key = (cname, tuple(new_shape))
        new_name = made.get(key)
        if new_name is None:
            arr = ix.const(cname, max_numel=_MAX_PREBROADCAST)
            if arr is None:
                continue
            try:
                arr2 = np.ascontiguousarray(np.broadcast_to(
                    arr.reshape(cshape) if csize > 1 else arr.reshape(-1)[0], new_shape))
            except ValueError:
                continue
            new_name = f"{cname}__bcast{'x'.join(str(d) for d in new_shape)}"
            while new_name in ix.inits:
                new_name += "_"
            t = nph.from_array(arr2.astype(arr.dtype), new_name)
            ix.graph.initializer.append(t)
            ix.inits[new_name] = t
            ix.shapes[new_name] = list(new_shape)
            made[key] = new_name
        n.input[ci] = new_name
        count += 1
    return count


# ------------------------------------------------------------------ #
# Entry point                                                          #
# ------------------------------------------------------------------ #

def fuse_patterns(model: onnx.ModelProto, align_elems: int = 8) -> Dict[str, int]:
    """Run the pattern fusions in place.  Returns
    ``{"layernorm": n, "gelu": n, "const_bcast": n}``."""
    counts = {"layernorm": 0, "gelu": 0, "const_bcast": 0}
    ix = _Index(model.graph)
    matches: List[dict] = []
    used: set = set()

    def _take(m):
        if m is None or any(id(k) in used for k in m["nodes"]):
            return False
        used.update(id(k) for k in m["nodes"])
        matches.append(m)
        return True

    for n in ix.nodes:
        if n.op_type == "ReduceMean" and _take(_match_tf_layernorm(ix, n)):
            counts["layernorm"] += 1
        elif n.op_type == "Tanh" and _take(_match_gelu_tanh(ix, n)):
            counts["gelu"] += 1
        elif n.op_type == "Erf" and _take(_match_gelu_erf(ix, n)):
            counts["gelu"] += 1
    if matches:
        _apply(ix, matches)
        ix = _Index(model.graph)
    counts["const_bcast"] = _normalise_const_broadcast(ix, align_elems)
    return counts


def pattern_hint(op_type: str) -> str:
    """Extra text for the 'not supported' error of a pattern-only op."""
    if op_type not in PATTERN_ONLY_OPS:
        return ""
    return (f"\n'{op_type}' is only supported inside a fused LayerNorm / GELU pattern "
            f"(src/fusion.py: TensorFlow-style LayerNorm, GELU tanh / erf forms, or "
            f"native LayerNormalization / Gelu ops); this instance did not match "
            f"(or pattern fusion is disabled).  Lowering it op by op would saturate "
            f"ap_fixed<16,8> intermediates.")


__all__ = ["fold_constant_nodes", "lower_split", "fuse_patterns", "pattern_hint",
           "PATTERN_ONLY_OPS", "default_opset"]
