#!/usr/bin/env python3
"""stereo_proxy.py — a latency proxy of LightStereo-S for the scheduler (doc/plans/STEREO_PLAN.md).

Builds an ONNX graph with LightStereo-S's layers at a resolution, written
with the ops the scheduler takes today, so that
`.claude/skills/model-study/scripts/onnx_study.py --no-numerics` prices it with
the bitstream's performance model.  Random weights (the price does not
depend on them).  Stand-ins, each priced as the implementation would run:

  ConvTranspose k4 s2 p1   Conv 3x3 (C -> 4C, the four output phases) + a
                           pixel shuffle (Reshape / Transpose / Reshape: a host copy)
  ConvTranspose k3 s2 op1  Conv 2x2 (C -> 4C) + the same pixel shuffle
  1 x 11, 1 x 21 stripes   1 x 7 (7 x 1) depthwise pieces summed by Adds
  InstanceNorm             LayerNormalization over H*W (the same host work)
  replicate pad + conv     the conv's zero padding
  channel softmax          Transpose + Softmax (VectorOP's unit; the real one
                           runs in column mode without the transpose)

Not in the graph (priced by hand in the plan): the correlation volume and the
context upsampling (host ops).  The graph's two inputs are the images; its
outputs are the disparity-softmax regression (1/4 resolution) and the 9
upsampling weights (full resolution).

usage: inference-scheduler/.venv/bin/python demo/stereo_depth/scripts/stereo_proxy.py [--height 480 --width 640] [--price]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

DEMO = Path(__file__).resolve().parent.parent
D = 48   # MAX_DISP / 4

MOBILENETV2 = [(1, 1, 16, 1), (2, 6, 24, 2), (3, 6, 32, 2), (4, 6, 64, 2), (3, 6, 96, 1), (3, 6, 160, 2)]


class G:
    def __init__(self, seed=0):
        self.nodes, self.inits, self.n = [], [], 0
        self.rng = np.random.default_rng(seed)

    def name(self, p):
        self.n += 1
        return f"{p}_{self.n}"

    def init(self, arr, p="w"):
        n = self.name(p)
        arr = np.asarray(arr)
        self.inits.append(numpy_helper.from_array(arr.astype(np.int64 if p == "shape" else np.float32), n))
        return n

    def node(self, op, ins, p=None, **attrs):
        out = self.name(p or op.lower())
        self.nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    def conv(self, x, cin, cout, k=(3, 3), s=1, pads=None, group=1, act=None):
        kh, kw = k
        w = self.rng.standard_normal((cout, cin // group, kh, kw)) * (1.0 / np.sqrt(cin // group * kh * kw))
        b = np.zeros(cout)
        if pads is None:
            pads = [kh // 2, kw // 2, kh // 2, kw // 2]
        y = self.node("Conv", [x, self.init(w), self.init(b, "b")], "conv", kernel_shape=[kh, kw],
                      strides=[s, s], pads=pads, group=group)
        return self.act(y, act)

    def act(self, y, act):
        if act == "relu6":
            return self.node("Clip", [y, self.init(np.array(0.0), "lo"), self.init(np.array(6.0), "hi")])
        if act == "relu":
            return self.node("Relu", [y])
        if act and act.startswith("leaky"):
            return self.node("LeakyRelu", [y], alpha=float(act[5:] or 0.01))
        return y

    def shuffle(self, x, c, h, w):
        """[1, 4c, h, w] (phase-major) -> [1, c, 2h, 2w]: a host copy."""
        x = self.node("Reshape", [x, self.init(np.array([2, 2, c, h, w]), "shape")])
        x = self.node("Transpose", [x], perm=[2, 3, 0, 4, 1])
        return self.node("Reshape", [x, self.init(np.array([1, c, 2 * h, 2 * w]), "shape")])

    def deconv4(self, x, cin, cout, h, w, act=None):
        y = self.conv(x, cin, 4 * cout, (3, 3))
        return self.act(self.shuffle(y, cout, h, w), act)

    def deconv3(self, x, cin, cout, h, w):
        y = self.conv(x, cin, 4 * cout, (2, 2), pads=[0, 0, 1, 1])
        return self.shuffle(y, cout, h, w)

    def add(self, a, b):
        return self.node("Add", [a, b])

    def instnorm(self, x, c, h, w):
        y = self.node("Reshape", [x, self.init(np.array([1, c, h * w]), "shape")])
        y = self.node("LayerNormalization", [y, self.init(np.ones(h * w), "g"), self.init(np.zeros(h * w), "be")],
                      axis=-1, epsilon=1e-5)
        return self.node("Reshape", [y, self.init(np.array([1, c, h, w]), "shape")])


def mbv2(g, x, cin, cout, e, s):
    if e == 1:
        h = g.conv(x, cin, cin, group=cin, s=s, act="relu6")
        h = g.conv(h, cin, cout, (1, 1))
    else:
        h = g.conv(x, cin, cin * e, (1, 1), act="relu6")
        h = g.conv(h, cin * e, cin * e, s=s, group=cin * e, act="relu6")
        h = g.conv(h, cin * e, cout, (1, 1))
    return g.add(x, h) if s == 1 and cin == cout else h


def backbone(g, img, H, W):
    x = g.conv(img, 3, 32, s=2, act="relu6")
    c, feats = 32, []
    for nb, e, out, s in MOBILENETV2:
        for j in range(nb):
            x = mbv2(g, x, c, out, e, s if j == 0 else 1)
            c = out
        feats.append(x)
    c2, c3, c4, c5 = feats[1], feats[2], feats[4], feats[5]

    def fpn(low, high, cl, ch, h, w):   # low at h x w, high at 2h x 2w
        y = g.deconv4(low, cl, ch, h, w, act="leaky0.2")
        y = g.node("Concat", [high, y], axis=1)
        return g.conv(y, 2 * ch, ch, act="leaky0.2")
    p4 = fpn(c5, c4, 160, 96, H // 32, W // 32)
    p3 = fpn(p4, c3, 96, 32, H // 16, W // 16)
    p2 = fpn(p3, c2, 32, 24, H // 8, W // 8)
    p2 = g.instnorm(g.conv(p2, 24, 24), 24, H // 4, W // 4)
    return p2, p3, p4


def stripes(g, a, c, h, w):
    out = a
    for k in (7, 11, 21):
        y = a
        for horizontal in (True, False):
            pieces = None
            for s in range(0, k, 7):
                kk = min(7, k - s)
                p = g.conv(y, c, c, (1, kk) if horizontal else (kk, 1), group=c,
                           pads=[0, kk // 2, 0, kk - 1 - kk // 2] if horizontal else
                           [kk // 2, 0, kk - 1 - kk // 2, 0])
                pieces = p if pieces is None else g.add(pieces, p)
            y = pieces
        out = g.add(out, y)
    return out


def attention(g, cost, feat, c, cf, h, w):
    a = g.conv(feat, cf, c, (1, 1))
    a = stripes(g, a, c, h, w)
    a = g.conv(a, c, c, (1, 1))
    return g.node("Mul", [a, cost])


def mv2res(g, x, cin, cout, s):
    return mbv2(g, x, cin, cout, 4, s)


def build(H: int, W: int) -> onnx.ModelProto:
    g = G()
    fl = backbone(g, "left", H, W)
    backbone(g, "right", H, W)      # its p2 is what the host correlation reads
    h4, w4 = H // 4, W // 4
    # correlation volume: a host op, not in the graph — the volume enters as an input
    vol = "volume"
    x = mv2res(g, vol, 48, 48, 1)
    x = attention(g, x, fl[0], 48, 24, h4, w4)
    c1 = mv2res(g, x, 48, 96, 2)
    c2 = mv2res(g, c1, 96, 96, 1)
    c2 = attention(g, c2, fl[1], 96, 32, h4 // 2, w4 // 2)
    c3 = mv2res(g, c2, 96, 192, 2)
    c4 = c3
    for _ in range(3):
        c4 = mv2res(g, c4, 192, 192, 1)
    c4 = attention(g, c4, fl[2], 192, 96, h4 // 4, w4 // 4)
    c5 = g.deconv3(c4, 192, 96, h4 // 4, w4 // 4)
    c5 = g.node("Relu", [g.add(c5, mv2res(g, c2, 96, 96, 1))])
    c6 = g.deconv3(c5, 96, 48, h4 // 2, w4 // 2)
    c6 = g.node("Relu", [g.add(c6, mv2res(g, x, 48, 48, 1))])
    # disparity softmax + regression
    t = g.node("Reshape", [c6, g.init(np.array([D, h4 * w4]), "shape")])
    t = g.node("Transpose", [t], perm=[1, 0])
    prob = g.node("Softmax", [t], axis=-1)
    disp = g.node("MatMul", [prob, g.init(np.arange(D).reshape(D, 1) * 1.0, "dvals")])
    # refinement: the 9 upsampling weights at full resolution
    xs = g.conv(fl[0], 24, 24)
    xs = g.act(g.instnorm(xs, 24, h4, w4), "leaky")
    xs = g.conv(xs, 24, 24)
    xs = g.node("Relu", [g.instnorm(xs, 24, h4, w4)])
    st = g.conv("left", 3, 16, s=2, act="leaky")
    st = g.conv(st, 16, 16, act="relu")
    y = g.deconv4(xs, 24, 16, h4, w4, act="leaky0.2")
    y = g.node("Concat", [st, y], axis=1)
    y = g.conv(y, 32, 16, act="leaky0.2")
    y = g.deconv4(y, 16, 9, H // 2, W // 2)
    t = g.node("Reshape", [y, g.init(np.array([9, H * W]), "shape")])
    t = g.node("Transpose", [t], perm=[1, 0])
    spx = g.node("Softmax", [t], axis=-1)
    inputs = [helper.make_tensor_value_info(n, TensorProto.FLOAT, [1, 3, H, W]) for n in ("left", "right")]
    inputs.append(helper.make_tensor_value_info("volume", TensorProto.FLOAT, [1, D, h4, w4]))
    outputs = [helper.make_tensor_value_info(disp, TensorProto.FLOAT, [h4 * w4, 1]),
               helper.make_tensor_value_info(spx, TensorProto.FLOAT, [H * W, 9])]
    graph = helper.make_graph(g.nodes, f"lightstereo_s_proxy_{W}x{H}", inputs, outputs, g.inits)
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    return onnx.shape_inference.infer_shapes(m)


def price(path: Path) -> dict:
    """The scheduler's event stream replayed with the bitstream's performance
    model (perf_models/kv260/, src/codegen/timing.py), as onnx_study.py does,
    except that a kernel call outside its family's fitted range is
    extrapolated by the family model (reported apart) and the proxy-only host
    nodes (the softmax transposes; the 9-wide host softmax) are dropped."""
    import collections
    import sys
    sys.path.insert(0, str(DEMO.parent.parent / "inference-scheduler"))
    from src.codegen import CodeGenerator
    from src.codegen.timing import kernel_duration_fn, simulate
    from src.graph import OnnxGraph
    from src.host_model import HostModel, default_host_model_path
    from src.perf_model import family
    from src.planning import PlanOptions, resolve_perf_model
    g = OnnxGraph(str(path), fuse_act=True, s2d_stem=True, fuse_patterns=True)
    cg = CodeGenerator(g, model_path=str(path))
    pm = resolve_perf_model(PlanOptions())
    hm = HostModel.load(default_host_model_path())
    kfn = kernel_duration_fn(pm, cg._layouts)
    extrap: dict = collections.Counter()

    def kus(sn):
        d = kfn(sn)
        if d is not None:
            return d
        tot = 0.0
        for c in sn.kernel_calls(cg._layouts):
            us = max(pm.families[family(c)].predict_us(c), 0.0) * c.count
            extrap[family(c)] += us
            tot += us
        return tot

    dropped = []

    def hus(sn):
        op, shp = sn.onnx_node.op_type, tuple(sn.inputs[0].shape)
        if (op == "Transpose" and len(shp) == 2) or (op == "Softmax" and shp[-1] == 9):
            dropped.append(f"{op} {shp}")
            return 0.0
        return hm.us(sn)

    tl = simulate(cg, kus, hus)
    by = {sn.index: sn for sn in g.nodes}
    host: dict = collections.defaultdict(float)
    for _lane, idx, t0, t1, kind in tl.spans:
        if kind == "host":
            host[by[idx].onnx_node.op_type] += t1 - t0
    return {"perf_model": str(pm.path), "total_ms": tl.total_us / 1e3, "cpu_ms": tl.cpu_us / 1e3,
            "lanes_ms": {k: v / 1e3 for k, v in sorted(tl.lane_us.items())},
            "extrapolated_ms": {k: v / 1e3 for k, v in extrap.items()},
            "host_ms": {k: v / 1e3 for k, v in host.items()}, "dropped": dropped,
            "unpriced": sorted({by[i].onnx_node.op_type for i in tl.unpriced})}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--out", default=None)
    ap.add_argument("--price", action="store_true", help="price it with the bitstream's performance model")
    a = ap.parse_args(argv)
    m = build(a.height, a.width)
    out = Path(a.out) if a.out else DEMO / "assets" / "study" / f"proxy_{a.width}x{a.height}.onnx"
    out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(m, out)
    print(f"{out}: {len(m.graph.node)} nodes")
    if a.price:
        import json
        r = price(out)
        out.with_suffix(".price.json").write_text(json.dumps(r, indent=1))
        print(f"predicted {r['total_ms']:.1f} ms (CPU {r['cpu_ms']:.1f}); lanes "
              + ", ".join(f"{k} {v:.1f}" for k, v in r["lanes_ms"].items()))
        print("  extrapolated beyond the measured range: "
              + ", ".join(f"{k} {v:.1f} ms" for k, v in r["extrapolated_ms"].items()))
        print("  host: " + ", ".join(f"{k} {v:.1f} ms" for k, v in sorted(r["host_ms"].items(), key=lambda kv: -kv[1]))
              + f"; unpriced {r['unpriced']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
