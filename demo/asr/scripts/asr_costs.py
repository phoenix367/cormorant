#!/usr/bin/env python3
"""asr_costs.py — the ASR candidates priced on the KV260 (doc/plans/ASR_PLAN.md):
proxy ONNX graphs of each model's encoder (for an audio length) and of one
decoder step (for a context), with the real shapes and standard ops, scheduled
as BERT is (MatMuls on ConvKernel / MatmulKernel, Softmax on VectorOPKernel's
unit, LayerNormalization on the row-ops unit, Gelu on VectorOPKernel's
activation unit, head transposes as row-ops jobs) and replayed with the
bitstream's performance model (the model-study skill's onnx_study.py latency).

The conv stems are priced as their im2col MatMuls (the same MACs on ConvKernel);
the weights are random (pricing does not read them).  A decoder step reads its
self-attention K / V and the encoder's cross K / V as graph inputs (the caches).

usage: inference-scheduler/.venv/bin/python demo/asr/scripts/asr_costs.py
           [--seconds 2 5 10 30] [--models ...] [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

import numpy as np
import onnx
from onnx import TensorProto, helper as oh, numpy_helper as onh

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
SCHED = os.path.join(REPO, "inference-scheduler")
sys.path.insert(0, SCHED)
sys.path.insert(0, os.path.join(REPO, ".claude", "skills", "model-study", "scripts"))
ASSETS = os.path.normpath(os.path.join(HERE, "..", "assets"))
SR = 16000


class G:
    """A tiny ONNX graph builder."""

    def __init__(self, seed=0):
        self.nodes, self.inits, self.inputs, self.n = [], [], [], 0
        self.rng = np.random.default_rng(seed)

    def name(self, p):
        self.n += 1
        return f"{p}_{self.n}"

    def input(self, name, shape):
        self.inputs.append(oh.make_tensor_value_info(name, TensorProto.FLOAT, list(shape)))
        return name

    def const(self, shape, scale=0.02, kind=None):
        n = self.name("c")
        if kind == "i64":
            self.inits.append(onh.from_array(np.asarray(shape, np.int64), n))
        else:
            self.inits.append(onh.from_array((self.rng.standard_normal(shape) * scale).astype(np.float32), n))
        return n

    def op(self, t, ins, **kw):
        o = self.name(t.lower())
        self.nodes.append(oh.make_node(t, ins, [o], name=o, **kw))
        return o

    def linear(self, x, k, n, bias=True):
        y = self.op("MatMul", [x, self.const((k, n))])
        return self.op("Add", [y, self.const((n,))]) if bias else y

    def ln(self, x, d):
        return self.op("LayerNormalization", [x, self.const((d,), 1.0), self.const((d,))], axis=-1, epsilon=1e-5)

    def heads(self, x, T, H, hd):                       # [T, H*hd] -> [H, T, hd]
        r = self.op("Reshape", [x, self.const([T, H, hd], kind="i64")])
        return self.op("Transpose", [r], perm=[1, 0, 2])

    def merge(self, x, T, H, hd):                       # [H, T, hd] -> [T, H*hd]
        t = self.op("Transpose", [x], perm=[1, 0, 2])
        return self.op("Reshape", [t, self.const([T, H * hd], kind="i64")])

    def attention(self, q, k, v, Tq, Tk, H, hd, kt=None):
        """q [Tq, H*hd]; k, v [Tk, H*hd] (or K^T / V head tensors given as kt / v)."""
        qh = self.heads(q, Tq, H, hd)
        if kt is None:
            kt = self.op("Transpose", [self.heads(k, Tk, H, hd)], perm=[0, 2, 1])
            v = self.heads(v, Tk, H, hd)
        s = self.op("MatMul", [qh, kt])
        p = self.op("Softmax", [s], axis=-1)
        return self.merge(self.op("MatMul", [p, v]), Tq, H, hd)

    def save(self, outputs, path):
        g = oh.make_graph(self.nodes, "proxy", self.inputs, [], self.inits)   # the outputs added once inferred
        m = onnx.shape_inference.infer_shapes(oh.make_model(g, opset_imports=[oh.make_opsetid("", 20)]))
        vi = {v.name: v for v in m.graph.value_info}
        del m.graph.output[:]
        m.graph.output.extend([vi[o] for o in outputs])          # the outputs with their inferred shapes
        onnx.save(m, path)
        return path


# ------------------------------------------------------------------ shapes
def arch(name):
    c = json.load(open(os.path.join(ASSETS, name, "config.json")))
    if c["model_type"] == "whisper":
        return dict(kind="whisper", d=c["d_model"], H=c["encoder_attention_heads"], ff=c["encoder_ffn_dim"],
                    L=c["encoder_layers"], Ld=c["decoder_layers"], V=c["vocab_size"], glu=False, hd_pad=None)
    d, H = c["hidden_size"], c["encoder_num_attention_heads"]
    hd = d // H
    return dict(kind="moonshine", d=d, H=H, ff=c["intermediate_size"], L=c["encoder_num_hidden_layers"],
                Ld=c["decoder_num_hidden_layers"], V=c["vocab_size"], glu=True,
                hd_pad=-(-hd // 16) * 16)                       # head dim padded for the kernels' K % 16


def enc_frames(a, seconds, window):
    n = int(seconds * SR)
    if a["kind"] == "whisper":
        return 1500 if window is None else min(1500, -(-(-(-n // 320)) // window) * window)
    t1 = (n - 127) // 64 + 1
    t2 = (t1 - 7) // 3 + 1
    return (t2 - 3) // 2 + 1


def pad16(n):
    return -(-n // 16) * 16


MAX_ROWS = 1280                      # ConvKernel max_out_ch: rows per MatMul-on-ConvKernel call


def row_parts(Tp):
    """(P, R): P row parts of R rows (R a multiple of 16, P * R >= Tp, R <= MAX_ROWS)."""
    P = -(-Tp // MAX_ROWS)
    return P, pad16(-(-Tp // P))


class EncG(G):
    """An encoder over P x R rows: every linear one batched MatMul (P ConvKernel calls of
    R rows: the batch pattern a_s = R * K, b_s = 0), the attention per part of R query
    rows (q laid out [P, H, R, hd]: each part a contiguous view) against all keys."""

    def __init__(self, P, R):
        super().__init__()
        self.P, self.R, self.T = P, R, P * R

    def rlin(self, x, k, n, bias=True):
        if self.P == 1:
            return self.linear(x, k, n, bias)
        x3 = self.op("Reshape", [x, self.const([self.P, self.R, k], kind="i64")])
        y = self.op("Reshape", [self.op("MatMul", [x3, self.const((k, n))]), self.const([self.T, n], kind="i64")])
        return self.op("Add", [y, self.const((n,))]) if bias else y

    def attn(self, q, k, v, H, hd):
        P, R, T = self.P, self.R, self.T
        kt = self.op("Transpose", [self.heads(k, T, H, hd)], perm=[0, 2, 1])
        vh = self.heads(v, T, H, hd)
        q4 = self.op("Transpose", [self.op("Reshape", [q, self.const([P, R, H, hd], kind="i64")])],
                     perm=[0, 2, 1, 3])
        outs = []
        for p in range(P):
            qp = q4 if P == 1 else self.op("Slice", [q4, self.const([p], kind="i64"), self.const([p + 1], kind="i64"),
                                                     self.const([0], kind="i64")])
            qp = self.op("Reshape", [qp, self.const([H, R, hd], kind="i64")])
            pr = self.op("Softmax", [self.op("MatMul", [qp, kt])], axis=-1)
            outs.append(self.merge(self.op("MatMul", [pr, vh]), R, H, hd))
        return outs[0] if P == 1 else self.op("Concat", outs, axis=0)


def encoder_graph(a, T, path):
    """The encoder for T frames: stem MatMuls, L layers, final LN."""
    d, H, ff = a["d"], a["H"], a["ff"]
    hd = a["hd_pad"] or d // H
    da = H * hd                                         # attention width (padded heads)
    P, R = row_parts(pad16(T))
    g = EncG(P, R)
    Tp = g.T
    if a["kind"] == "whisper":
        P1, R1 = row_parts(2 * Tp)
        g1 = EncG(P1, R1)                               # conv1 (stride 1) over twice the rows
        g1.nodes, g1.inits, g1.inputs, g1.n = g.nodes, g.inits, g.inputs, g.n
        g1.op("Gelu", [g1.rlin(g1.input("conv1_cols", (g1.T, 80 * 3)), 240, d)])
        g.n = g1.n
        h = g.op("Gelu", [g.rlin(g.input("conv2_cols", (Tp, d * 3)), 3 * d, d)])     # conv2 (stride 2)
        h = g.op("Add", [h, g.const((Tp, d))])          # + positions
    else:
        P1, R1 = row_parts(pad16(6 * Tp + 4))
        g1 = EncG(P1, R1)                               # conv1: 127 taps, stride 64 -> polyphase [64 x 2]
        g1.nodes, g1.inits, g1.inputs, g1.n = g.nodes, g.inits, g.inputs, g.n
        g1.rlin(g1.input("conv1_cols", (g1.T, 128)), 128, d, bias=False)   # (tanh, GroupNorm: host)
        P2, R2 = row_parts(pad16(2 * Tp + 1))
        g2 = EncG(P2, R2)
        g2.nodes, g2.inits, g2.inputs, g2.n = g.nodes, g.inits, g.inputs, g1.n
        g2.op("Gelu", [g2.rlin(g2.input("conv2_cols", (g2.T, d * 7)), 7 * d, 2 * d)])
        g.n = g2.n
        h = g.op("Gelu", [g.rlin(g.input("conv3_cols", (Tp, 2 * d * 3)), 6 * d, d)])
    for _ in range(a["L"]):
        x = g.ln(h, d)
        att = g.attn(g.rlin(x, d, da), g.rlin(x, d, da, bias=False), g.rlin(x, d, da), H, hd)
        h = g.op("Add", [h, g.rlin(att, da, d)])
        x = g.ln(h, d)
        h = g.op("Add", [h, g.rlin(g.op("Gelu", [g.rlin(x, d, ff)]), ff, d)])
    return g.save([g.ln(h, d)], path)


def decoder_graph(a, T, n_ctx, path, head=True):
    """One decoder step at context n_ctx (cached self K / V and the cross K / V as inputs)."""
    g = G()
    d, H, ff, V = a["d"], a["H"], a["ff"], a["V"]
    hd = a["hd_pad"] or d // H
    da = H * hd
    Tp, nc = pad16(T), pad16(n_ctx)
    h = g.input("x", (1, d))
    for i in range(a["Ld"]):
        x = g.ln(h, d)
        q = g.heads(g.linear(x, d, da), 1, H, hd)
        g.linear(x, d, da, bias=False)                  # k, v of the new row (into the cache)
        g.linear(x, d, da)
        kt, v = g.input(f"self_kt{i}", (H, hd, nc)), g.input(f"self_v{i}", (H, nc, hd))
        p = g.op("Softmax", [g.op("MatMul", [q, kt])], axis=-1)
        h = g.op("Add", [h, g.linear(g.merge(g.op("MatMul", [p, v]), 1, H, hd), da, d)])
        x = g.ln(h, d)
        q = g.heads(g.linear(x, d, da), 1, H, hd)
        kt, v = g.input(f"cross_kt{i}", (H, hd, Tp)), g.input(f"cross_v{i}", (H, Tp, hd))
        p = g.op("Softmax", [g.op("MatMul", [q, kt])], axis=-1)
        h = g.op("Add", [h, g.linear(g.merge(g.op("MatMul", [p, v]), 1, H, hd), da, d)])
        x = g.ln(h, d)
        if a["glu"]:
            u = g.linear(x, d, 2 * ff)
            f = g.op("Mul", [g.op("Sigmoid", [u]), u])      # stands in for silu(gate) * x (one VectorOP pass)
            f = g.op("Slice", [f, g.const([0], kind="i64"), g.const([ff], kind="i64"), g.const([1], kind="i64")])
        else:
            f = g.op("Gelu", [g.linear(x, d, ff)])
        h = g.op("Add", [h, g.linear(f, ff, d)])
    out = g.ln(h, d)
    if head:
        out = g.op("MatMul", [out, g.const((d, pad16(V)))])
    return g.save([out], path)


def cross_graph(a, T, path):
    """The cross-attention K / V of every decoder layer, once per utterance."""
    d, H = a["d"], a["H"]
    da = H * (a["hd_pad"] or d // H)
    P, R = row_parts(pad16(T))
    g = EncG(P, R)
    e = g.input("enc", (g.T, d))
    outs = []
    for _ in range(a["Ld"]):
        outs += [g.rlin(e, d, da, bias=False), g.rlin(e, d, da)]
    return g.save(outs, path)


def price(path):
    """Predicted time of the proxy graph: the performance model's price per kernel call,
    the analytic cost model (rpu_policy.estimate_us) where it has none — the share
    reported as fallback_ms — and the host model for host ops."""
    import onnx_study as os_
    from src.codegen import CodeGenerator
    from src.codegen.timing import kernel_duration_fn, simulate
    from src.host_model import HostModel, default_host_model_path
    from src.planning import PlanOptions, resolve_perf_model
    from src.rpu_policy import estimate_us
    g, err, _ = os_.build(path)
    if g is None:
        return {"error": err}
    cg = CodeGenerator(g, model_path=path)
    pm = resolve_perf_model(PlanOptions())
    hm = HostModel.load(default_host_model_path())
    fb = {"ms": 0.0, "nodes": 0}

    def fallback(sn):
        t = sum(estimate_us(c) * c.count for c in sn.kernel_calls(cg._layouts))
        fb["ms"] += t / 1e3
        fb["nodes"] += 1
        return t
    kfn = kernel_duration_fn(pm, cg._layouts, fallback=fallback, chains=cg._ro_chains)
    by_op = {}

    def kus(sn):
        t = kfn(sn)
        if t:
            key = f"{sn.kernel_name}:{os_.op_of(sn)}"
            by_op[key] = by_op.get(key, 0.0) + t / 1e3
        return t
    tl = simulate(cg, kus, hm.us)
    by_index = {sn.index: sn for sn in g.nodes}
    return {"total_ms": tl.total_us / 1e3, "cpu_ms": tl.cpu_us / 1e3, "cpu_waits_ms": tl.wait_us / 1e3,
            "lanes_ms": {k: v / 1e3 for k, v in sorted(tl.lane_us.items())},
            "fallback_ms": fb["ms"], "fallback_nodes": fb["nodes"],
            "unpriced": [f"{os_.op_of(by_index[i])} {by_index[i].onnx_node.name}" for i in tl.unpriced],
            "by_op_ms": dict(sorted(by_op.items(), key=lambda kv: -kv[1])),
            "partition": os_.partition(g)["lanes"], "pool_mib": os_.memory(g, cg)["pool_mib"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", default=["whisper-tiny.en", "whisper-base.en", "moonshine-tiny",
                                                     "moonshine-base"])
    ap.add_argument("--seconds", type=float, nargs="*", default=[2.0, 5.0, 10.0, 20.0])
    ap.add_argument("--window", type=int, default=None,
                    help="Whisper: encoder window rounded up to buckets of N frames (default: 30 s)")
    ap.add_argument("--ctx", type=int, nargs="*", default=[16, 64],
                    help="decoder self-attention contexts to price")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    res = {}
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as td:
        for name in args.models:
            a = arch(name)
            r = res.setdefault(name, {"encoder": {}, "cross": {}, "decoder": {}})
            for s in args.seconds:
                T = enc_frames(a, s, args.window)
                key = f"{s:g}s"
                if a["kind"] == "whisper" and args.window is None and r["encoder"]:
                    r["encoder"][key] = next(iter(r["encoder"].values()))
                    r["cross"][key] = next(iter(r["cross"].values()))
                    continue
                e = price(encoder_graph(a, T, os.path.join(td, f"{name}_enc_{T}.onnx")))
                c = price(cross_graph(a, T, os.path.join(td, f"{name}_cross_{T}.onnx")))
                e["frames"] = c["frames"] = T
                r["encoder"][key], r["cross"][key] = e, c
                print(f"{name:16s} {key:>4s}: encoder {T:5d} frames {e.get('total_ms', float('nan')):8.1f} ms "
                      f"(CPU {e.get('cpu_ms', 0):6.1f}, analytic {e.get('fallback_ms', 0):5.1f} ms in "
                      f"{e.get('fallback_nodes', 0)} nodes, unpriced {len(e.get('unpriced', []))}), "
                      f"cross K/V {c.get('total_ms', float('nan')):6.1f} ms  {e.get('error', '')}", flush=True)
            T = enc_frames(a, max(args.seconds), args.window)
            for n in args.ctx:
                dct = price(decoder_graph(a, T, n, os.path.join(td, f"{name}_dec_{n}.onnx")))
                r["decoder"][f"ctx{n}_frames{T}"] = dct
                print(f"{name:16s} decoder step ctx {n:3d} over {T} frames: {dct.get('total_ms', float('nan')):6.2f} ms "
                      f"(CPU {dct.get('cpu_ms', 0):5.2f}, analytic {dct.get('fallback_ms', 0):5.2f} ms, "
                      f"pool {dct.get('pool_mib', 0):.0f} MiB, "
                      f"unpriced {len(dct.get('unpriced', []))})  {dct.get('error', '')}", flush=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
