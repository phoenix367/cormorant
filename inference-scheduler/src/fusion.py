"""
ONNX-level lowering passes (BERT_PLAN phase 1b).

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
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as nph

from .nodes import SchedulerError


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


__all__ = ["fold_constant_nodes", "lower_split", "default_opset"]
