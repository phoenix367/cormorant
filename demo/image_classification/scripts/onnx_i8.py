# Copyright 2026 Ivan Gubochkin
# SPDX-License-Identifier: Apache-2.0

"""onnx_i8.py — the 8-bit configuration (doc/plans/INT8_PLAN.md §6, §9.7) of an ONNX CNN: a numpy
evaluator of the graph in float64 (the reference) and on the 8-bit kernels' integer arithmetic
(inference-scheduler/src/int8_spec.py) — the CNN studies' engine (cnn_i8_study.py,
demo/object_detection/scripts/yolo_study.py).

The 8-bit execution (Graph.run with the formats of calibrate()):
  weights   Conv / Gemm: int8 [-127, 127], a power-of-two exponent per output channel
            (int8_spec.encode_weights); the bias in the accumulator (bias_raw)
  tensors   int8 at a calibrated power-of-two exponent per tensor ("static"); unsigned [0, 255]
            where the calibration saw no negative value (stored with the -128 offset, which the
            consumer's bias absorbs — exact, so the emulation sums the unsigned values; padding
            takes the offset, value 0).  With ``blocks``, a Conv output that only Convs read is
            int8 in blocks of 32 channels at every pixel, each with a 3-bit shift below the
            calibrated base (the drain's block maximum: int8_spec.requant_blocked)
  Conv      the exact sums of the int8 products (block-shifted), the bias, a fused Relu / Clip
            (the drain), rne + saturation into the output's format; depthwise per channel
  VectorOP  Add (+ a fused Relu / Clip), Relu, Clip, SiLU (x * Sigmoid(x), fused): the exact
            function of the exact operands, rne into the output's format
  Pool      MaxPool exact; AveragePool / GlobalAveragePool the exact mean, rne
  outputs   the graph's outputs int16 at their calibrated exponent + 8
Exponents (calibrate): two float passes over the calibration inputs — the maxima, then the squared
error of rounding at the int8 exponent of the maximum and the 7 finer ones (llm_study.I8Cal's rule);
the least-error candidate per tensor.  Tensors that must share an exponent (MaxPool / Resize /
Reshape / Flatten in and out, a Concat's inputs and output) form one group.  A blocked tensor's base
is the exponent of its maximum (the blocks go finer below it).
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCHED = os.path.abspath(os.path.join(HERE, "..", "..", "..", "inference-scheduler"))
if SCHED not in sys.path:
    sys.path.insert(0, SCHED)
from src import int8_spec as s8    # noqa: E402

VIEW = {"Reshape", "Flatten", "Squeeze", "Unsqueeze", "Identity", "Dropout"}
SAME = VIEW | {"MaxPool", "Resize"}          # the output keeps the input's exponent (exact)
CAND = 8                                     # calibration candidates (llm_study.I8_CAND)
I16_LO, I16_HI = -32768, 32767


class Node:
    def __init__(self, op, inputs, outputs, attrs, name=""):
        self.op, self.inputs, self.outputs, self.attrs, self.name = op, list(inputs), list(outputs), attrs, name
        self.act = None                          # a fused activation: ("relu",) / ("clip", lo, hi)


def _attrs(n):
    import onnx
    return {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute}


class Fmt:
    """A tensor's 8-bit format: kind "s" (static int8 at f; unsigned [0, 255] or signed),
    "c" (static int8 at an exponent per channel f [C]: a depthwise conv's input), "b" (int8
    blocks of 32 channels, base f) or "o16" (int16 at f: a graph output)."""
    def __init__(self, kind, f, unsigned=False):
        self.kind, self.unsigned = kind, unsigned
        self.f = np.asarray(f, np.int64) if kind == "c" else int(f)

    def range(self):
        if self.kind == "o16":
            return I16_LO, I16_HI
        return (0, 255) if self.unsigned else (s8.I8_LO, s8.I8_HI)

    def __repr__(self):
        f = f"{int(self.f.min())}..{int(self.f.max())}" if self.kind == "c" else self.f
        return f"{self.kind}{'u' if self.unsigned else ''}{f}"

    def scale(self, ndim):
        """2^f broadcast over an NCHW (or [N, C]) tensor."""
        if self.kind != "c":
            return 2.0 ** self.f
        return np.exp2(self.f.astype(np.float64)).reshape((1, -1) + (1,) * (ndim - 2))


class T:
    """A tensor in the 8-bit run: its exact value v (float64), its format and, blocked, q / d."""
    def __init__(self, v, fmt=None, q=None, d=None):
        self.v, self.fmt, self.q, self.d = v, fmt, q, d


class Graph:
    def __init__(self, path):
        import onnx
        from onnx import numpy_helper
        m = onnx.load(path)
        self.init = {t.name: numpy_helper.to_array(t).astype(np.float64) for t in m.graph.initializer}
        nodes = []
        for n in m.graph.node:
            a = _attrs(n)
            if n.op_type == "Constant":
                self.init[n.output[0]] = numpy_helper.to_array(a["value"]).astype(np.float64)
                continue
            nodes.append(Node(n.op_type, n.input, n.output, a, n.name))
        self.inputs = [i.name for i in m.graph.input if i.name not in self.init]
        self.outputs = [o.name for o in m.graph.output]
        self.nodes = nodes
        self._fuse()
        self.wq = {}                              # Conv / Gemm weights: (int8 [K, O] or depthwise, f_w)

    # ---- graph rewrites
    def _consumers(self):
        cons = defaultdict(list)
        for n in self.nodes:
            for i in n.inputs:
                if i:
                    cons[i].append(n)
        return cons

    def _act_of(self, n):
        if n.op == "Relu":
            return ("relu",)
        if n.op == "Clip":
            lo = n.attrs.get("min", -np.inf)
            hi = n.attrs.get("max", np.inf)
            if len(n.inputs) > 1 and n.inputs[1]:
                lo = float(np.ravel(self.init[n.inputs[1]])[0])
            if len(n.inputs) > 2 and n.inputs[2]:
                hi = float(np.ravel(self.init[n.inputs[2]])[0])
            return ("clip", float(lo), float(hi))
        return None

    def _fuse(self):
        """x * Sigmoid(x) -> SiLU; a Relu / Clip that is the only reader of a Conv / Add output
        -> that node's act (the drain / VectorOP's act register)."""
        cons = self._consumers()
        prod = {o: n for n in self.nodes for o in n.outputs}
        drop = set()
        for n in self.nodes:
            if n.op != "Mul":
                continue
            a, b = n.inputs
            for x, s in ((a, b), (b, a)):
                sg = prod.get(s)
                if sg is not None and sg.op == "Sigmoid" and sg.inputs[0] == x and len(cons[s]) == 1:
                    n.op, n.inputs, n.attrs = "SiLU", [x], {}
                    drop.add(id(sg))
                    break
        self.nodes = [n for n in self.nodes if id(n) not in drop]
        cons = self._consumers()
        drop = set()
        out = set(self.outputs)
        for n in self.nodes:
            if n.op not in ("Conv", "Add") or n.outputs[0] in out:
                continue
            c = cons[n.outputs[0]]
            if len(c) == 1 and c[0].op in ("Relu", "Clip"):
                n.act = self._act_of(c[0])
                n.outputs = [c[0].outputs[0]]
                drop.add(id(c[0]))
        self.nodes = [n for n in self.nodes if id(n) not in drop]
        self.cons = self._consumers()

    # ---- the float64 reference
    def run_float(self, feeds, hook=None):
        env = dict(self.init)
        env.update({k: np.asarray(v, np.float64) for k, v in feeds.items()})
        for n in self.nodes:
            y = self._float_op(n, env)
            env[n.outputs[0]] = y
            if hook:
                hook(n.outputs[0], y)
        return [env[o] for o in self.outputs]

    def _float_op(self, n, env):
        x = [env[i] if i else None for i in n.inputs]
        op, a = n.op, n.attrs
        if op == "Conv":
            y = conv_float(x[0], x[1], x[2] if len(x) > 2 else None, a)
        elif op == "Gemm":
            A = x[0].T if a.get("transA") else x[0]
            B = x[1].T if a.get("transB") else x[1]
            y = a.get("alpha", 1.0) * (A @ B)
            if len(x) > 2 and x[2] is not None:
                y = y + a.get("beta", 1.0) * x[2]
        elif op == "MatMul":
            y = x[0] @ x[1]
        elif op == "Add":
            y = x[0] + x[1]
        elif op == "Relu":
            return np.maximum(x[0], 0.0)
        elif op == "Clip":
            _, lo, hi = self._act_of(n)
            return np.clip(x[0], lo, hi)
        elif op == "SiLU":
            return silu(x[0])
        elif op == "Sigmoid":
            return 1.0 / (1.0 + np.exp(-x[0]))
        elif op == "Mul":
            return x[0] * x[1]
        elif op == "MaxPool":
            return pool_float(x[0], a, "max")
        elif op == "AveragePool":
            return pool_float(x[0], a, "avg")
        elif op == "GlobalAveragePool":
            return x[0].mean(axis=(2, 3), keepdims=True)
        elif op in VIEW:
            return view(x, n, a)
        elif op == "Concat":
            return np.concatenate(x, axis=a["axis"])
        elif op == "Resize":
            return resize_nearest(x[0], n, env)
        else:
            raise NotImplementedError(f"onnx_i8: op {op} ({n.name})")
        return apply_act(y, n.act)

    # ---- formats
    def groups(self, tie_concat=True):
        """Union-find over the tensors that share an exponent: {tensor: root} (``tie_concat``
        False: a Concat is a host copy that re-rounds, its inputs keep their own exponents)."""
        par = {}

        def find(t):
            par.setdefault(t, t)
            while par[t] != t:
                par[t] = par[par[t]]
                t = par[t]
            return t

        def union(a, b):
            par[find(a)] = find(b)
        for n in self.nodes:
            if n.op in SAME:
                union(n.inputs[0], n.outputs[0])
            elif n.op == "Concat" and tie_concat:
                for i in n.inputs:
                    union(i, n.outputs[0])
        return find

    BLOCK_WRITERS = ("Conv",)                     # blocks="conv": ConvKernel's drain
    BLOCK_WRITERS_ANY = ("Conv", "SiLU", "Add", "Relu", "Clip")   # blocks="any": + VectorOP's writer
    BLOCK_WRITERS_HOST = BLOCK_WRITERS_ANY + ("Concat",)           # blocks="host": + the Concat copy

    BLOCK_READERS = ("Conv",)
    BLOCK_READERS_VOP = ("Conv", "Add", "SiLU", "Relu", "Clip")   # VectorOP with per-block pre-shifts

    def blocked_tensors(self, writers=BLOCK_WRITERS, readers=BLOCK_READERS):
        """The outputs of ``writers`` that only ``readers`` read (not a graph output): in blocks."""
        out = set(self.outputs)
        return {n.outputs[0] for n in self.nodes
                if n.op in writers and n.outputs[0] not in out
                and self.cons[n.outputs[0]] and all(c.op in readers for c in self.cons[n.outputs[0]])}

    def dw_inputs(self):
        """Tensors that only depthwise Convs read (not a graph input or output)."""
        out = set(self.outputs) | set(self.inputs)
        return {n.outputs[0] for n in self.nodes
                if n.outputs[0] not in out and self.cons[n.outputs[0]]
                and all(c.op == "Conv" and is_depthwise(c, self.init[c.inputs[1]]) for c in self.cons[n.outputs[0]])}

    def act_inputs(self):
        """Conv outputs that only an activation (VectorOP: SiLU, Relu, Clip) reads."""
        return {n.outputs[0] for n in self.nodes
                if n.op == "Conv" and self.cons[n.outputs[0]]
                and all(c.op in ("SiLU", "Relu", "Clip") for c in self.cons[n.outputs[0]])}

    def encode(self):
        """The int8 weights of every Conv / Gemm, per output channel."""
        for n in self.nodes:
            if n.op == "Conv":
                W = self.init[n.inputs[1]]
                O, Cg, kh, kw = W.shape
                if is_depthwise(n, W):
                    q, f = s8.encode_weights(W.reshape(O, kh * kw), axis=0)
                    self.wq[n.outputs[0]] = (q.reshape(O, kh, kw), f)
                else:
                    assert n.attrs.get("group", 1) == 1, f"grouped conv {n.name}"
                    q, f = s8.encode_weights(W.transpose(2, 3, 1, 0).reshape(kh * kw * Cg, O), axis=1)
                    self.wq[n.outputs[0]] = (q.reshape(kh, kw, Cg, O), f)
            elif n.op == "Gemm":
                B = self.init[n.inputs[1]]
                B = B.T if n.attrs.get("transB") else B
                self.wq[n.outputs[0]] = s8.encode_weights(B * n.attrs.get("alpha", 1.0), axis=1)

    # ---- the 8-bit run
    def run_i8(self, feeds, fmts, trace=None):
        """The 8-bit run: the outputs' values; ``trace`` (a dict) receives every tensor's T."""
        if not self.wq:
            self.encode()
        env = {k: T(v) for k, v in self.init.items()}
        for k, v in feeds.items():
            env[k] = self.write_float(np.asarray(v, np.float64), fmts[k])
        for n in self.nodes:
            env[n.outputs[0]] = self._i8_op(n, env, fmts[n.outputs[0]])
        if trace is not None:
            trace.update(env)
        return [env[o].v for o in self.outputs]

    def dequant_weights(self, n):
        """A Conv / Gemm node's int8 weights as float64 in the ONNX layout (exact)."""
        q, f = self.wq[n.outputs[0]]
        sc = np.exp2(-np.asarray(f, np.float64))
        if n.op == "Gemm":
            B = q * sc[None, :]
            return B.T if n.attrs.get("transB") else B
        if q.ndim == 3:                                   # depthwise [O, kh, kw]
            return (q * sc[:, None, None])[:, None]
        return (q * sc).transpose(3, 2, 0, 1)             # [kh, kw, C, O] -> [O, C, kh, kw]

    def write_float(self, v, fmt):
        """A producer that computes the exact value (VectorOP, Pool, the input): rne into fmt."""
        if fmt.kind == "b":
            q, d = s8.block_round_float(nhwc_pad(v), fmt.f)
            return blocked_t(q, d, fmt, v.shape)
        lo, hi = fmt.range()
        sc = fmt.scale(v.ndim)
        q = np.clip(s8.rne_float(v * sc), lo, hi)
        return T(q / sc, fmt)

    def write_acc(self, acc, g, fmt, act, shape):
        """A kernel's accumulator [M, O] (value acc * 2^-g) -> the output tensor (NCHW shape):
        the fused activation on the exact sums, then rne + saturation (or the block rule)."""
        acc = act_acc(acc, g, act)
        if fmt.kind == "b":
            O = acc.shape[-1]
            q, d = s8.requant_blocked(np.pad(acc, ((0, 0), (0, (-O) % s8.BLOCK))),
                                      np.pad(g, (0, (-O) % s8.BLOCK)), fmt.f)
            return blocked_t(q, d, fmt, shape)
        lo, hi = fmt.range()
        f = fmt.f[None, :] if fmt.kind == "c" else fmt.f
        r = s8.requant(acc, g, f, lo, hi)
        return T(nhwc_to(r * np.exp2(-np.asarray(f, np.float64)), shape), fmt)

    def _i8_op(self, n, env, fmt):
        op, a = n.op, n.attrs
        x = [env[i] if i else None for i in n.inputs]
        if op == "Conv":
            W = self.init[n.inputs[1]]
            b = x[2].v if len(x) > 2 and x[2] is not None else None
            acc, g, shape = conv_i8(x[0], W, b, a, self.wq[n.outputs[0]], is_depthwise(n, W))
            return self.write_acc(acc, g, fmt, n.act, shape)
        if op == "Gemm":
            assert not a.get("transA")
            xi = x[0]
            assert xi.fmt.kind in ("s", "o16"), "Gemm reads a static tensor"
            wq, fw = self.wq[n.outputs[0]]
            xq = s8.rne_float(xi.v * 2.0 ** xi.fmt.f)
            bias = None
            if len(x) > 2 and x[2] is not None:
                bias = x[2].v * a.get("beta", 1.0)
            g = xi.fmt.f + s8.DELTA_MAX + fw
            acc, g = s8.matmul(xq, xi.fmt.f, 0, wq, fw, acc_bits=63,
                               bias_acc=None if bias is None else s8.bias_raw(bias, g))
            return self.write_acc(acc, g, fmt, n.act, acc.shape)
        if op in ("Add", "Relu", "Clip", "SiLU", "AveragePool", "GlobalAveragePool"):
            return self.write_float(self._float_op(n, {i: env[i].v for i in n.inputs if i}), fmt)
        if op in SAME or op == "Concat":                  # exact where the group shares the exponent
            y = self._float_op(n, {i: env[i].v for i in n.inputs if i})
            if op in VIEW:
                return T(y, fmt)
            return self.write_float(y, fmt)               # (a Concat copy re-rounds: blocks="host")
        raise NotImplementedError(f"onnx_i8: op {op} ({n.name})")


# ------------------------------------------------------------------ calibration
def policy_options(name):
    """A policy name -> calibrate()'s options: "i8" every tensor static; "i8b" + the Conv outputs
    only Convs read in blocks; "i8ba" + any writer's (VectorOP's too, with a block maximum);
    modifiers "+a16": a Conv output only an activation reads stays int16 (the activation reads 16
    bits, writes 8); "+in16": the graph's inputs int16 (the first Conv in the W8A16 mode); "+dwc": a
    tensor only depthwise Convs read static per channel (its exponent folds into each output
    channel's shift: each reads one input channel); "i8bc":
    "i8ba" + a Concat that only Convs read is the host's copy, written in blocks; "i8bv": "i8ba"
    with VectorOP reading blocks too (per-block pre-shifts), so a residual stream is blocked."""
    base, *mods = name.split("+")
    blocks = {"i8": None, "i8b": "conv", "i8ba": "any", "i8bc": "host", "i8bv": "vop"}[base]
    for m in mods:
        if m not in ("a16", "in16", "dwc"):
            raise KeyError(f"policy {name!r}: unknown modifier {m!r}")
    return dict(blocks=blocks, act16="a16" in mods, in16="in16" in mods, dwc="dwc" in mods)


class Calib:
    """The two calibration passes (module docstring): ``maxima`` then ``errors`` hooks for
    Graph.run_float; ``formats`` -> {tensor: Fmt} (options: policy_options)."""

    def __init__(self, g, blocks=None, act16=False, in16=False, dwc=False):
        self.g = g
        self.find = g.groups(tie_concat=blocks != "host")
        if blocks is True:
            blocks = "conv"
        writers = {"conv": g.BLOCK_WRITERS, "any": g.BLOCK_WRITERS_ANY, "host": g.BLOCK_WRITERS_HOST,
                   "vop": g.BLOCK_WRITERS_ANY}
        self.blocked = (g.blocked_tensors(writers[blocks], g.BLOCK_READERS_VOP if blocks == "vop"
                                          else g.BLOCK_READERS) if blocks else set())
        self.wide = (g.act_inputs() if act16 else set()) | (set(g.inputs) if in16 else set())
        self.chan = g.dw_inputs() - self.wide if dwc else set()
        self.blocked -= self.chan
        self.cmax, self.cf0, self.csse = {}, {}, {}
        self.amax, self.amin = defaultdict(float), defaultdict(float)
        self.sse = defaultdict(lambda: np.zeros(CAND))
        self.f0 = {}

    def maxima(self, name, v):
        if name in self.chan:
            m = np.abs(v).max(axis=tuple(a for a in range(v.ndim) if a != 1))
            self.cmax[name] = m if name not in self.cmax else np.maximum(self.cmax[name], m)
        r = self.find(name)
        self.amax[r] = max(self.amax[r], float(np.abs(v).max()))
        self.amin[r] = min(self.amin[r], float(v.min()))

    def unsigned(self, r):
        return self.amin[r] >= 0.0

    def base(self, r):
        a = self.amax[r]
        if self.unsigned(r):                          # [0, 255]: one bit finer than the signed range
            return s8.base_exponent(a / 2.0) if a > 0 else 0
        return s8.base_exponent(a)

    def errors(self, name, v):
        if name in self.chan:                         # per channel: the least error of its own
            self.chan_errors(name, v)
            return
        r = self.find(name)
        if r not in self.f0:
            self.f0[r] = self.base(r)
        lo, hi = (0, 255) if self.unsigned(r) else (s8.I8_LO, s8.I8_HI)
        for c in range(CAND):
            sc = 2.0 ** (self.f0[r] + c)
            self.sse[r][c] += float(((np.clip(np.rint(v * sc), lo, hi) / sc - v) ** 2).sum())

    def chan_errors(self, name, v):
        r = self.find(name)
        u = self.unsigned(r)
        if name not in self.cf0:
            self.cf0[name] = np.array([s8.base_exponent(m / 2.0 if u else m) for m in self.cmax[name]], np.int64)
            self.csse[name] = np.zeros((len(self.cf0[name]), CAND))
        lo, hi = (0, 255) if u else (s8.I8_LO, s8.I8_HI)
        ax = tuple(a for a in range(v.ndim) if a != 1)
        for c in range(CAND):
            sc = np.exp2((self.cf0[name] + c).astype(np.float64)).reshape((1, -1) + (1,) * (v.ndim - 2))
            self.csse[name][:, c] += ((np.clip(np.rint(v * sc), lo, hi) / sc - v) ** 2).sum(axis=ax)

    def formats(self):
        fm = {}
        outs = set(self.g.outputs)
        out_groups = {self.find(o) for o in outs}             # an output's views: the same int16
        names = list(self.g.inputs) + [n.outputs[0] for n in self.g.nodes]
        for t in names:
            r = self.find(t)
            if t in outs or r in out_groups or t in self.wide:
                fm[t] = Fmt("o16", s8.base_exponent(self.amax[r]) + 8)
            elif t in self.chan:
                fm[t] = Fmt("c", self.cf0[t] + self.csse[t].argmin(axis=1), self.unsigned(r))
            elif t in self.blocked:
                fm[t] = Fmt("b", s8.base_exponent(self.amax[r]))
            else:
                f0 = self.f0.get(r, self.base(r))
                fm[t] = Fmt("s", f0 + int(np.argmin(self.sse[r])), self.unsigned(r))
        return fm


def calibrate(g, samples, blocks=None, act16=False, in16=False, dwc=False):
    """samples: a list of feed dicts.  Returns {tensor: Fmt} (options: policy_options)."""
    c = Calib(g, blocks, act16, in16, dwc)
    for s in samples:
        for k, v in s.items():
            c.maxima(k, np.asarray(v, np.float64))
        g.run_float(s, c.maxima)
    for s in samples:
        for k, v in s.items():
            c.errors(k, np.asarray(v, np.float64))
        g.run_float(s, c.errors)
    return c.formats()


# ------------------------------------------------------------------ ops
def silu(x):
    return x / (1.0 + np.exp(-x))


def apply_act(y, act):
    if act is None:
        return y
    if act[0] == "relu":
        return np.maximum(y, 0.0)
    return np.clip(y, act[1], act[2])


def act_acc(acc, g, act):
    """The fused activation on the exact accumulator (value acc * 2^-g, g per column)."""
    if act is None:
        return acc
    if act[0] == "relu":
        return np.maximum(acc, 0)
    lo, hi = act[1], act[2]
    sc = np.exp2(np.asarray(g, np.float64))
    lo_a = np.where(np.isfinite(lo), np.ceil(lo * sc), np.iinfo(np.int64).min).astype(np.int64)
    hi_a = np.where(np.isfinite(hi), np.floor(hi * sc), np.iinfo(np.int64).max).astype(np.int64)
    return np.clip(acc, lo_a, hi_a)


def is_depthwise(n, W):
    gr = n.attrs.get("group", 1)
    return gr > 1 and gr == W.shape[0] and W.shape[1] == 1


def conv_geom(x_shape, W_shape, a):
    N, C, H, Wd = x_shape
    kh, kw = W_shape[2], W_shape[3]
    sh, sw = a.get("strides", [1, 1])
    dh, dw = a.get("dilations", [1, 1])
    pt, pl, pb, pr = auto_pads(a, H, Wd, kh, kw, sh, sw, dh, dw)
    eh, ew = (kh - 1) * dh + 1, (kw - 1) * dw + 1
    OH = (H + pt + pb - eh) // sh + 1
    OW = (Wd + pl + pr - ew) // sw + 1
    return kh, kw, sh, sw, dh, dw, (pt, pl, pb, pr), OH, OW


def auto_pads(a, H, W, kh, kw, sh, sw, dh=1, dw=1):
    """ONNX pads [top, left, bottom, right], auto_pad SAME_UPPER / SAME_LOWER resolved."""
    ap = a.get("auto_pad", b"NOTSET")
    ap = ap.decode() if isinstance(ap, bytes) else ap
    if ap in ("NOTSET", "VALID"):
        return tuple(a.get("pads", [0, 0, 0, 0])) if ap == "NOTSET" else (0, 0, 0, 0)
    out = []
    for n, k, st, dl in ((H, kh, sh, dh), (W, kw, sw, dw)):
        tot = max((-(-n // st) - 1) * st + (k - 1) * dl + 1 - n, 0)
        out.append((tot // 2, tot - tot // 2) if ap == "SAME_UPPER" else (tot - tot // 2, tot // 2))
    return out[0][0], out[1][0], out[0][1], out[1][1]


def windows(x, kh, kw, sh, sw, dh, dw, pads, OH, OW, pad_value=0.0):
    """x [N, C, H, W] -> [N, C, OH, OW, kh, kw] (a strided view of the padded input)."""
    pt, pl, pb, pr = pads
    xp = np.pad(x, ((0, 0), (0, 0), (pt, pb), (pl, pr)), constant_values=pad_value)
    eh, ew = (kh - 1) * dh + 1, (kw - 1) * dw + 1
    v = np.lib.stride_tricks.sliding_window_view(xp, (eh, ew), axis=(2, 3))
    return v[:, :, ::sh, ::sw][:, :, :OH, :OW, ::dh, ::dw]


def conv_float(x, W, b, a):
    O = W.shape[0]
    kh, kw, sh, sw, dh, dw, pads, OH, OW = conv_geom(x.shape, W.shape, a)
    N = x.shape[0]
    gr = a.get("group", 1)
    v = windows(x, kh, kw, sh, sw, dh, dw, pads, OH, OW)
    if gr == 1:
        cols = v.transpose(0, 2, 3, 4, 5, 1).reshape(N * OH * OW, -1)
        y = cols @ W.transpose(2, 3, 1, 0).reshape(-1, O)
        y = y.reshape(N, OH, OW, O).transpose(0, 3, 1, 2)
    elif gr == x.shape[1] == O and W.shape[1] == 1:
        y = np.einsum("nchwij,cij->nchw", v, W[:, 0], optimize=True)
    else:
        Cg, Og = x.shape[1] // gr, O // gr
        y = np.concatenate([conv_float(x[:, i * Cg:(i + 1) * Cg], W[i * Og:(i + 1) * Og], None,
                                       dict(a, group=1)) for i in range(gr)], axis=1)
    if b is not None:
        y = y + b[None, :, None, None]
    return y


def conv_i8(xt, W, b, a, wenc, depthwise):
    """The 8-bit Conv of the tensor xt (T): (acc [N*OH*OW, O], g [O], output shape)."""
    O = W.shape[0]
    N, C = xt.v.shape[:2]
    kh, kw, sh, sw, dh, dw, pads, OH, OW = conv_geom(xt.v.shape, W.shape, a)
    wq, fw = wenc
    dmax = s8.DELTA_MAX
    f = xt.fmt.f
    if xt.fmt.kind == "b":
        q, d = xt.q, xt.d                                  # [N, Cp, H, W], [N, Cb, H, W]
    else:
        assert xt.fmt.kind != "c" or depthwise, "a per-channel input feeds depthwise Convs only"
        q = s8.rne_float(xt.v * xt.fmt.scale(xt.v.ndim))
        d = None
    g = f + dmax + np.asarray(fw, np.int64)
    if depthwise:
        if d is None:
            xs = q << dmax
        else:
            xs = (q * (np.int64(1) << np.repeat(dmax - d, s8.BLOCK, axis=1)))[:, :C]
        v = windows(xs.astype(np.float64), kh, kw, sh, sw, dh, dw, pads, OH, OW)
        acc = np.einsum("nchwij,cij->nhwc", v, wq.astype(np.float64), optimize=True)
        acc = np.rint(acc).reshape(N * OH * OW, O).astype(np.int64)
    else:
        Cg = wq.shape[2]
        if d is None:
            v = windows(q.astype(np.float64), kh, kw, sh, sw, dh, dw, pads, OH, OW)
            cols = v.transpose(0, 2, 3, 4, 5, 1).reshape(N * OH * OW, kh * kw * C)
            acc, g = s8.matmul(cols.astype(np.int64), f, 0, wq.reshape(-1, O), fw, acc_bits=63)
        else:
            Cp = q.shape[1]
            wp = np.zeros((kh, kw, Cp, O), np.int64)
            wp[:, :, :Cg] = wq
            v = windows(q.astype(np.float64), kh, kw, sh, sw, dh, dw, pads, OH, OW)
            cols = v.transpose(0, 2, 3, 4, 5, 1).reshape(N * OH * OW, kh * kw * Cp)
            vd = windows(d.astype(np.float64), kh, kw, sh, sw, dh, dw, pads, OH, OW, pad_value=dmax)
            xd = vd.transpose(0, 2, 3, 4, 5, 1).reshape(N * OH * OW, -1)
            acc, g = s8.matmul(cols.astype(np.int64), f, xd.astype(np.int64), wp.reshape(-1, O), fw,
                               acc_bits=63)
    if b is not None:
        acc = acc + s8.bias_raw(b, g)[None, :]
    return acc, g, (N, O, OH, OW)


def nhwc_to(y, shape):
    """[M, O] rows (N, OH, OW) -> NCHW; a 2-D [N, O] stays."""
    if len(shape) == 2:
        return y
    N, O, OH, OW = shape
    return y.reshape(N, OH, OW, O).transpose(0, 3, 1, 2)


def nhwc_pad(v):
    """NCHW (or [N, C]) v -> channels last, padded to a multiple of the block."""
    if v.ndim == 4:
        v = v.transpose(0, 2, 3, 1)
    return np.pad(v, [(0, 0)] * (v.ndim - 1) + [(0, (-v.shape[-1]) % s8.BLOCK)])


def blocked_t(q, d, fmt, shape):
    """q [M or N, H, W, Cp] / [M, Cp] and d from the block rule -> a T with q [N, Cp, H, W],
    d [N, Cb, H, W] and the exact value v (NCHW, unpadded)."""
    if len(shape) == 2:
        N, O = shape
        v = s8.block_value(q, fmt.f, d)[:, :O]
        return T(v, fmt, q, d)
    N, O, OH, OW = shape
    Cp = q.shape[-1]
    q4 = q.reshape(N, OH, OW, Cp)
    d4 = d.reshape(N, OH, OW, Cp // s8.BLOCK)
    v = s8.block_value(q4, fmt.f, d4)[..., :O].transpose(0, 3, 1, 2)
    return T(v, fmt, q4.transpose(0, 3, 1, 2), d4.transpose(0, 3, 1, 2))


def pool_float(x, a, kind):
    kh, kw = a["kernel_shape"]
    sh, sw = a.get("strides", [1, 1])
    if a.get("ceil_mode"):
        raise NotImplementedError("ceil_mode")
    N, C, H, W = x.shape
    pt, pl, pb, pr = auto_pads(a, H, W, kh, kw, sh, sw)
    OH = (H + pt + pb - kh) // sh + 1
    OW = (W + pl + pr - kw) // sw + 1
    fill = -np.inf if kind == "max" else 0.0
    v = windows(x, kh, kw, sh, sw, 1, 1, (pt, pl, pb, pr), OH, OW, pad_value=fill)
    if kind == "max":
        return v.max(axis=(4, 5))
    if a.get("count_include_pad") or not any((pt, pl, pb, pr)):
        return v.mean(axis=(4, 5))
    ones = windows(np.ones((1, 1, H, W)), kh, kw, sh, sw, 1, 1, (pt, pl, pb, pr), OH, OW)
    return v.sum(axis=(4, 5)) / ones.sum(axis=(4, 5))


def view(x, n, a):
    op = n.op
    if op in ("Identity", "Dropout"):
        return x[0]
    if op == "Flatten":
        ax = a.get("axis", 1)
        return x[0].reshape(int(np.prod(x[0].shape[:ax])), -1)
    if op == "Reshape":
        shape = [int(s) for s in x[1]]
        shape = [x[0].shape[i] if s == 0 else s for i, s in enumerate(shape)]
        if shape and shape[0] == 1 and x[0].shape[0] > 1:
            shape[0] = x[0].shape[0]                      # a batch-1 export run on a batch
        return x[0].reshape(shape)
    if op == "Squeeze":
        axes = a.get("axes") if "axes" in a else ([int(s) for s in x[1]] if len(x) > 1 and x[1] is not None else None)
        return np.squeeze(x[0], axis=tuple(axes) if axes is not None else None)
    if op == "Unsqueeze":
        axes = a.get("axes") if "axes" in a else [int(s) for s in x[1]]
        y = x[0]
        for ax in sorted(axes):
            y = np.expand_dims(y, ax)
        return y
    raise NotImplementedError(op)


def resize_nearest(x, n, env):
    mode = n.attrs.get("mode", b"nearest")
    assert mode in (b"nearest", "nearest"), mode
    scales = env[n.inputs[2]] if len(n.inputs) > 2 and n.inputs[2] and env[n.inputs[2]].size else None
    if scales is None:
        sizes = env[n.inputs[3]]
        scales = np.asarray(sizes, np.float64) / np.asarray(x.shape, np.float64)
    sh, sw = int(round(scales[2])), int(round(scales[3]))
    assert scales[0] == 1 and scales[1] == 1 and sh == scales[2] and sw == scales[3], scales
    return x.repeat(sh, axis=2).repeat(sw, axis=3)
