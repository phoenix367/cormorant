"""
Fully-connected Convs -> MatMul (doc/plans/LENET_PLAN.md).

A Conv whose kernel covers its whole unpadded input computes ONE output
pixel per image: it is a fully-connected layer.  LeNet's conv3 (7x7 over a
64 x 7 x 7 map, 1024 outputs) and a 1x1 conv on a 1x1 map (conv4last,
MobileNet's classifier) have that shape.  ConvKernel streams such a weight
through one 128-bit port for a single output pixel (LeNet conv3: 4.6 ms of
its 5.4); as a MatMul the weight is B of a one-row product, and
MatmulKernel's GEMV path reads it through both ports (2.0 ms).

  y = Conv(x[N, C, H, W], W[M, C, H, W], b[M])          (kernel H x W, pads 0)
    -> Reshape(MatMul(Flatten(x)[N, C*H*W], W'[C*H*W, M]), [N, M, 1, 1]) (+ b)

W'[(c*H + h)*W + w][m] = W[m][c][h][w] is Flatten's element order, so the
MatMul reads the same products; the bias becomes a VectorOP Add after the
Reshape (a following Relu still fuses into it).  In fixed point the result
is bit-identical: the Conv adds the bias inside its accumulator and floors
once — floor((acc + b_raw * 2^F) / 2^F) = floor(acc / 2^F) + b_raw — unless
the sum before the bias saturates (the MatMul saturates it first).

``mode``: "auto" rewrites where the engine cost model (src/cost_model.py:
the ConvKernel cycle model against the GEMV / tiled MatmulKernel model, plus
one VectorOP call for the bias) estimates the MatMul at least
``MIN_GAIN`` faster; "always" every eligible Conv; "off" none.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import onnx
from onnx import TensorProto
from onnx import helper as onnx_helper
from onnx import numpy_helper as nph

from ._matmul_hw_config import MATMUL_GEMV_MAX_M, MATMUL_MAX_K
from .cost_model import conv_board_cycles, gemv_cycles, matmul_cycles
from .nodes import SchedulerError

MODES = ("auto", "always", "off")
MIN_GAIN = 0.2              # "auto": the MatMul estimate at most 80 % of the Conv's
BIAS_ADD_CYCLES = 600       # the bias Add's own VectorOP call (issue + start-up)
LANES = 8                   # elements per 128-bit word (ap_fixed<16,8>)


def normalize_mode(mode) -> str:
    if mode is True:
        return "auto"
    if mode is False or mode is None:
        return "off"
    if mode not in MODES:
        raise SchedulerError(f"fc_conv must be one of {MODES}, got {mode!r}")
    return mode


def _shape(tp) -> List[int]:
    return [d.dim_value if d.HasField("dim_value") else -1 for d in tp.tensor_type.shape.dim]


def candidate(node: onnx.NodeProto, shape_map: dict, inits: dict) -> Optional[dict]:
    """The FC geometry of ``node`` — ``{x, w, b, y, n, c, h, wd, m}`` — or
    None when it is not a fully-connected Conv the MatMul path can take."""
    if node.op_type != "Conv" or len(node.input) < 2 or not node.output:
        return None
    attrs = {a.name: a for a in node.attribute}
    if (attrs["group"].i if "group" in attrs else 1) != 1:
        return None
    if "auto_pad" in attrs and attrs["auto_pad"].s.decode("utf-8") not in ("NOTSET", "VALID"):
        return None
    if any(d != 1 for d in (list(attrs["dilations"].ints) if "dilations" in attrs else [])):
        return None
    if any(p != 0 for p in (list(attrs["pads"].ints) if "pads" in attrs else [])):
        return None
    x, w = node.input[0], node.input[1]
    b = node.input[2] if len(node.input) > 2 and node.input[2] else None
    y = node.output[0]
    if w not in inits or (b is not None and b not in inits):
        return None
    xs, ws, ys = shape_map.get(x), list(inits[w].dims), shape_map.get(y)
    if not xs or not ys or len(xs) != 4 or len(ws) != 4 or len(ys) != 4 or min(xs + ys) <= 0:
        return None
    n, c, h, wd = xs
    m = ws[0]
    if ws[1:] != [c, h, wd] or ys != [n, m, 1, 1]:
        return None                                   # the kernel must cover the whole input
    if "kernel_shape" in attrs and list(attrs["kernel_shape"].ints) != [h, wd]:
        return None
    if b is not None and list(inits[b].dims) != [m]:
        return None
    if c * h * wd > MATMUL_MAX_K:
        return None                                   # MatmulKernel's K bound
    return dict(x=x, w=w, b=b, y=y, n=n, c=c, h=h, wd=wd, m=m)


def estimate(geo: dict) -> Dict[str, float]:
    """Cycles of the Conv and of its MatMul (GEMV where eligible) + bias."""
    n, c, h, wd, m = geo["n"], geo["c"], geo["h"], geo["wd"], geo["m"]
    k = c * h * wd
    conv = n * conv_board_cycles(in_ch=c, out_ch=m, in_h=h, in_w=wd, oh=1, ow=1, kh=h, kw=wd)["total"]
    gemv = (n == 1 and MATMUL_GEMV_MAX_M > 0 and m <= MATMUL_GEMV_MAX_M
            and k % LANES == 0 and m % LANES == 0 and m >= 64)
    mm = gemv_cycles(1, k, m) if gemv else matmul_cycles(n, k, m)
    return {"conv": float(conv), "matmul": float(mm) + (BIAS_ADD_CYCLES if geo["b"] else 0.0),
            "gemv": gemv}


def lower_fc_convs(model: onnx.ModelProto, mode="auto") -> Tuple[onnx.ModelProto, dict]:
    """Rewrite the fully-connected Convs of ``model`` (see the module doc).
    Returns ``(model, stats)`` with ``stats = {"lowered", "kept",
    "conv_cycles", "matmul_cycles"}`` (the estimates of the lowered ones);
    the model is returned unmodified when nothing is rewritten."""
    mode = normalize_mode(mode)
    stats = {"lowered": 0, "kept": 0, "conv_cycles": 0.0, "matmul_cycles": 0.0}
    if mode == "off":
        return model, stats
    graph = model.graph
    shape_map: Dict[str, List[int]] = {i.name: list(i.dims) for i in graph.initializer}
    for vi in list(graph.input) + list(graph.value_info) + list(graph.output):
        shape_map[vi.name] = _shape(vi.type)
    inits = {i.name: i for i in graph.initializer}
    names = set(shape_map) | {o for nd in graph.node for o in nd.output}

    def fresh(base: str) -> str:
        s = base
        while s in names:
            s += "_"
        names.add(s)
        return s

    new_nodes: List[onnx.NodeProto] = []
    new_vi: List[onnx.ValueInfoProto] = []
    new_inits: List[onnx.TensorProto] = []
    flat_of: Dict[str, str] = {}
    w_of: Dict[str, str] = {}
    for node in graph.node:
        geo = candidate(node, shape_map, inits)
        if geo is None:
            new_nodes.append(node)
            continue
        est = estimate(geo)
        if mode == "auto" and est["matmul"] > est["conv"] * (1.0 - MIN_GAIN):
            stats["kept"] += 1
            new_nodes.append(node)
            continue
        stats["lowered"] += 1
        stats["conv_cycles"] += est["conv"]
        stats["matmul_cycles"] += est["matmul"]
        n, m, k = geo["n"], geo["m"], geo["c"] * geo["h"] * geo["wd"]
        tag = node.name or geo["y"]
        flat = flat_of.get(geo["x"])
        if flat is None:
            flat = flat_of[geo["x"]] = fresh(f"{geo['x']}_fc_flat")
            new_nodes.append(onnx_helper.make_node("Flatten", [geo["x"]], [flat],
                                                   name=f"{tag}_fc_flatten", axis=1))
            new_vi.append(onnx_helper.make_tensor_value_info(flat, TensorProto.FLOAT, [n, k]))
        wt = w_of.get(geo["w"])
        if wt is None:
            wt = w_of[geo["w"]] = fresh(f"{geo['w']}_fc")
            arr = nph.to_array(inits[geo["w"]]).astype(np.float32).reshape(m, k)
            new_inits.append(nph.from_array(np.ascontiguousarray(arr.T), name=wt))
        mm = fresh(f"{geo['y']}_fc_mm")
        new_nodes.append(onnx_helper.make_node("MatMul", [flat, wt], [mm], name=f"{tag}_fc_matmul"))
        new_vi.append(onnx_helper.make_tensor_value_info(mm, TensorProto.FLOAT, [n, m]))
        shp = fresh(f"{geo['y']}_fc_shape")
        new_inits.append(nph.from_array(np.array([n, m, 1, 1], np.int64), name=shp))
        if geo["b"] is None:
            new_nodes.append(onnx_helper.make_node("Reshape", [mm, shp], [geo["y"]],
                                                   name=f"{tag}_fc_reshape"))
            continue
        r4 = fresh(f"{geo['y']}_fc_4d")
        new_nodes.append(onnx_helper.make_node("Reshape", [mm, shp], [r4], name=f"{tag}_fc_reshape"))
        new_vi.append(onnx_helper.make_tensor_value_info(r4, TensorProto.FLOAT, [n, m, 1, 1]))
        b4 = fresh(f"{geo['b']}_fc")
        new_inits.append(nph.from_array(
            nph.to_array(inits[geo["b"]]).astype(np.float32).reshape(1, m, 1, 1), name=b4))
        new_nodes.append(onnx_helper.make_node("Add", [r4, b4], [geo["y"]], name=f"{tag}_fc_bias"))
    if not stats["lowered"]:
        return model, stats
    # drop the original weights / biases nothing reads any more (LeNet also
    # lists them as graph inputs)
    used = {i for nd in new_nodes for i in nd.input}
    dead = {nm for nm in inits if nm not in used}
    new_graph = onnx_helper.make_graph(
        new_nodes, graph.name,
        [i for i in graph.input if i.name not in dead], list(graph.output),
        initializer=[i for i in graph.initializer if i.name not in dead] + new_inits,
        value_info=list(graph.value_info) + new_vi)
    new_model = onnx_helper.make_model(new_graph, opset_imports=list(model.opset_import))
    new_model.ir_version = model.ir_version
    for p in model.metadata_props:
        new_model.metadata_props.add(key=p.key, value=p.value)
    return new_model, stats


__all__ = ("MODES", "MIN_GAIN", "normalize_mode", "candidate", "estimate", "lower_fc_convs")
