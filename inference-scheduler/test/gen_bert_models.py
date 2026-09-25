#!/usr/bin/env python3
"""Tiny BERT-like ONNX fixtures (BERT_PLAN phase 1f).

Every model reproduces the structure of bertsquad-12 (onnx model zoo,
opset 12, as simplified by onnx-simplifier) at toy size with random small
weights, so the scheduler exercises the exact node arrangement of the real
model — host ops, pattern fusion, integer tensors, attention batching:

  embeddings  Gather(word) + OneHot(segment) · MatMul(token type) + position
              Add, TensorFlow-style LayerNorm (12 nodes)
  mask        Reshape -> Cast -> Mul(ones[1,S,1]) -> Reshape -> Sub(1, .)
              -> Mul(-10000)
  per layer   Q / K / V Gemm -> Reshape -> Transpose 0213 (K: 0231),
              MatMul, Mul 1/sqrt(d), mask Add, Softmax(axis 3), MatMul,
              Transpose 0213, Reshape, Gemm, residual Add, TF LayerNorm,
              GELU-tanh FFN (8 nodes), residual Add, TF LayerNorm
  head        Gemm -> Reshape -> Transpose 201 -> Split -> Squeeze x 2
  passthrough unique_ids int64 -> Identity -> output

Variants:
  bert_tiny_h32_l1.onnx      hidden 32, 2 heads, 1 layer,  seq 8,  opset 12
  bert_tiny_h64_l2.onnx      hidden 64, 4 heads, 2 layers, seq 16, opset 12
  bert_tiny_erf.onnx         hidden 32, 2 heads, 1 layer,  seq 8,  GELU erf
                             form (PyTorch export: Div sqrt2 -> Erf -> Add 1
                             -> Mul x -> Mul 0.5), opset 13 (Squeeze axes as
                             an input, Split sizes as an input)
  bert_tiny_native.onnx      hidden 32, 2 heads, 1 layer,  seq 8,  native
                             LayerNormalization / Gelu(tanh) / Softmax(-1),
                             opset 20 (Split num_outputs)

Usage:  .venv/bin/python test/gen_bert_models.py [--out-dir test/models]
"""

import argparse
import math
import os

import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as nph
from onnx import TensorProto

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_OUT = os.path.join(_HERE, "models")

# (file, hidden, heads, layers, seq, gelu, style, opset)
VARIANTS = [
    ("bert_tiny_h32_l1.onnx", 32, 2, 1, 8, "tanh", "tf", 12),
    ("bert_tiny_h64_l2.onnx", 64, 4, 2, 16, "tanh", "tf", 12),
    ("bert_tiny_erf.onnx", 32, 2, 1, 8, "erf", "tf", 13),
    ("bert_tiny_native.onnx", 32, 2, 1, 8, "tanh", "native", 20),
]
VOCAB = 50
EPS = np.float32(9.999999960041972e-13)          # bertsquad-12's epsilon


class _B:
    """Tiny graph builder (node list + initializers + unique names)."""

    def __init__(self, rng, opset):
        self.rng, self.opset = rng, opset
        self.nodes, self.inits, self.n = [], [], 0

    def name(self, base):
        self.n += 1
        return f"{base}_{self.n}"

    def const(self, arr, base, dtype=None):
        """Initializer; integer arrays keep their dtype, floats become float32."""
        nm = self.name(base)
        a = np.asarray(arr)
        a = a.astype(dtype) if dtype is not None else (
            a if a.dtype.kind in "iu" else a.astype(np.float32))
        self.inits.append(nph.from_array(a, nm))
        return nm

    def w(self, shape, scale, base):
        return self.const(self.rng.normal(0.0, scale, shape).astype(np.float32), base)

    def op(self, op_type, ins, base, n_out=1, **attrs):
        outs = [self.name(base + ":0") for _ in range(n_out)]
        self.nodes.append(oh.make_node(op_type, ins, outs, name=self.name(base), **attrs))
        return outs[0] if n_out == 1 else outs

    def reshape(self, x, shape, base):
        return self.op("Reshape", [x, self.const(np.array(shape, np.int64), base + "/shape")],
                       base + "/Reshape")

    def squeeze(self, x, axes, out, base):
        if self.opset >= 13:
            self.nodes.append(oh.make_node(
                "Squeeze", [x, self.const(np.array(axes, np.int64), base + "/axes")], [out],
                name=base))
        else:
            self.nodes.append(oh.make_node("Squeeze", [x], [out], name=base, axes=axes))

    def reduce_mean(self, x, axis, base):
        return self.op("ReduceMean", [x], base, axes=[axis], keepdims=1)

    # ---- TensorFlow-style LayerNorm, exactly as in bertsquad-12 -------- #
    def layernorm_tf(self, x, axis, n, base):
        gamma = self.const(1.0 + self.rng.normal(0, 0.1, n), base + "/gamma")
        beta = self.const(self.rng.normal(0, 0.1, n), base + "/beta")
        if self.style == "native":
            return self.op("LayerNormalization", [x, gamma, beta], base + "/LayerNorm",
                           axis=-1, epsilon=float(EPS))
        mean = self.reduce_mean(x, axis, base + "/moments/mean")
        sqd = self.op("Sub", [x, mean], base + "/moments/SquaredDifference")
        sq = self.op("Mul", [sqd, sqd], base + "/moments/SquaredDifference__sq")
        var = self.reduce_mean(sq, axis, base + "/moments/variance")
        add = self.op("Add", [var, self.const(EPS, base + "/eps")], base + "/batchnorm/add")
        rs = self.op("Sqrt", [add], base + "/batchnorm/Rsqrt")
        rr = self.op("Reciprocal", [rs], base + "/batchnorm/Rsqrt__rec")
        mul = self.op("Mul", [rr, gamma], base + "/batchnorm/mul")
        mul2 = self.op("Mul", [mean, mul], base + "/batchnorm/mul_2")
        sub = self.op("Sub", [beta, mul2], base + "/batchnorm/sub")
        mul1 = self.op("Mul", [x, mul], base + "/batchnorm/mul_1")
        return self.op("Add", [mul1, sub], base + "/batchnorm/add_1")

    # ---- GELU ---------------------------------------------------------- #
    def gelu(self, x, base):
        if self.style == "native":
            return self.op("Gelu", [x], base + "/Gelu", approximate="tanh")
        if self.gelu_form == "erf":                    # PyTorch export
            d = self.op("Div", [x, self.const(np.float32(math.sqrt(2.0)), base + "/sqrt2")],
                        base + "/Div")
            e = self.op("Erf", [d], base + "/Erf")
            a = self.op("Add", [e, self.const(np.float32(1.0), base + "/one")], base + "/Add")
            m = self.op("Mul", [x, a], base + "/Mul")
            return self.op("Mul", [m, self.const(np.float32(0.5), base + "/half")], base + "/Mul_1")
        p = self.op("Pow", [x, self.const(np.float32(3.0), base + "/three")], base + "/Pow")
        m = self.op("Mul", [self.const(np.float32(0.044715), base + "/c1"), p], base + "/mul")
        a = self.op("Add", [x, m], base + "/add")
        m1 = self.op("Mul", [self.const(np.float32(math.sqrt(2.0 / math.pi)), base + "/c2"), a],
                     base + "/mul_1")
        t = self.op("Tanh", [m1], base + "/Tanh")
        a1 = self.op("Add", [self.const(np.float32(1.0), base + "/one"), t], base + "/add_1")
        m2 = self.op("Mul", [self.const(np.float32(0.5), base + "/half"), a1], base + "/mul_2")
        return self.op("Mul", [x, m2], base + "/mul_3")

    def gemm(self, x, k, m, base):
        return self.op("Gemm", [x, self.w((k, m), 1.0 / math.sqrt(k), base + "/kernel"),
                                self.w((m,), 0.05, base + "/bias")], base + "/dense",
                       alpha=1.0, beta=1.0, transA=0, transB=0)


def make_bert(path, hidden, heads, layers, seq, gelu="tanh", style="tf", opset=12, seed=0):
    rng = np.random.default_rng(seed)
    b = _B(rng, opset)
    b.style, b.gelu_form = style, gelu
    H, S, nh = hidden, seq, heads
    dh = H // nh
    F = 4 * H
    i64 = TensorProto.INT64

    inputs = [oh.make_tensor_value_info("unique_ids_raw_output___9:0", i64, [1]),
              oh.make_tensor_value_info("segment_ids:0", i64, [1, S]),
              oh.make_tensor_value_info("input_mask:0", i64, [1, S]),
              oh.make_tensor_value_info("input_ids:0", i64, [1, S])]
    outputs = [oh.make_tensor_value_info("unstack:1", TensorProto.FLOAT, [1, S]),
               oh.make_tensor_value_info("unstack:0", TensorProto.FLOAT, [1, S]),
               oh.make_tensor_value_info("unique_ids:0", i64, [1])]

    b.nodes.append(oh.make_node("Identity", ["unique_ids_raw_output___9:0"], ["unique_ids:0"],
                                name="unique_ids_graph_outputs_Identity"))
    # attention mask: [1,S] int -> additive [1,1,S,S]
    m = b.reshape("input_mask:0", [1, 1, S], "bert/encoder/Reshape")
    m = b.op("Cast", [m], "bert/encoder/Cast", to=TensorProto.FLOAT)
    m = b.op("Mul", [b.const(np.ones((1, S, 1)), "bert/encoder/ones"), m], "bert/encoder/mul")
    m = b.reshape(m, [-1, 1, S, S], "bert/encoder/ExpandDims")
    m = b.op("Sub", [b.const(np.float32(1.0), "bert/encoder/one"), m], "bert/encoder/sub")
    mask = b.op("Mul", [m, b.const(np.float32(-10000.0), "bert/encoder/neg")], "bert/encoder/mul_1")
    # embeddings
    seg = b.reshape("segment_ids:0", [-1], "bert/embeddings/Reshape_2")
    ids = b.reshape("input_ids:0", [-1], "bert/embeddings/Reshape")
    e = b.op("Gather", [b.w((VOCAB, H), 0.5, "bert/embeddings/word_embeddings"), ids],
             "bert/embeddings/GatherV2", axis=0)
    e = b.reshape(e, [1, S, H], "bert/embeddings/Reshape_1")
    oh_ = b.op("OneHot", [seg, b.const(np.array([2], np.int32), "depth", np.int32),
                          b.const(np.array([0.0, 1.0]), "values")],
               "bert/embeddings/one_hot", axis=-1)
    tt = b.op("MatMul", [oh_, b.w((2, H), 0.5, "bert/embeddings/token_type_embeddings")],
              "bert/embeddings/MatMul")
    tt = b.reshape(tt, [-1, S, H], "bert/embeddings/Reshape_3")
    e = b.op("Add", [e, tt], "bert/embeddings/add")
    e = b.op("Add", [e, b.w((1, S, H), 0.5, "bert/embeddings/position")], "bert/embeddings/add_1")
    e = b.layernorm_tf(e, 2, H, "bert/embeddings/LayerNorm")
    x = b.reshape(e, [-1, H], "bert/encoder/Reshape_1")

    scale = np.float32(1.0 / math.sqrt(dh))
    for li in range(layers):
        p = f"bert/encoder/layer_{li}"
        v = b.gemm(x, H, H, p + "/attention/self/value")
        v = b.op("Transpose", [b.reshape(v, [-1, S, nh, dh], p + "/attention/self/Reshape_2")],
                 p + "/attention/self/transpose_2", perm=[0, 2, 1, 3])
        q = b.gemm(x, H, H, p + "/attention/self/query")
        q = b.op("Transpose", [b.reshape(q, [-1, S, nh, dh], p + "/attention/self/Reshape")],
                 p + "/attention/self/transpose", perm=[0, 2, 1, 3])
        k = b.gemm(x, H, H, p + "/attention/self/key")
        k = b.op("Transpose", [b.reshape(k, [-1, S, nh, dh], p + "/attention/self/Reshape_1")],
                 p + "/attention/self/MatMul__T", perm=[0, 2, 3, 1])
        sc = b.op("MatMul", [q, k], p + "/attention/self/MatMul")
        sc = b.op("Mul", [sc, b.const(scale, p + "/attention/self/Mul/y")], p + "/attention/self/Mul")
        sc = b.op("Add", [sc, mask], p + "/attention/self/add")
        pr = b.op("Softmax", [sc], p + "/attention/self/Softmax",
                  axis=-1 if style == "native" else 3)
        c = b.op("MatMul", [pr, v], p + "/attention/self/MatMul_1")
        c = b.op("Transpose", [c], p + "/attention/self/transpose_3", perm=[0, 2, 1, 3])
        c = b.reshape(c, [-1, H], p + "/attention/self/Reshape_3")
        a = b.gemm(c, H, H, p + "/attention/output")
        a = b.op("Add", [a, x], p + "/attention/output/add")
        x1 = b.layernorm_tf(a, 1, H, p + "/attention/output/LayerNorm")
        h = b.gemm(x1, H, F, p + "/intermediate")
        h = b.gelu(h, p + "/intermediate/dense")
        o = b.gemm(h, F, H, p + "/output")
        o = b.op("Add", [o, x1], p + "/output/add")
        x = b.layernorm_tf(o, 1, H, p + "/output/LayerNorm")

    lg = b.gemm(x, H, 2, "cls/squad")
    lg = b.reshape(lg, [-1, S, 2], "Reshape_1")
    lg = b.op("Transpose", [lg], "transpose", perm=[2, 0, 1])
    u6, u3 = "unstack_raw_output___6:0", "unstack_raw_output___3:0"
    if opset >= 18:
        b.nodes.append(oh.make_node("Split", [lg], [u6, u3], name="unstack", axis=0,
                                    num_outputs=2))
    elif opset >= 13:
        b.nodes.append(oh.make_node("Split", [lg, b.const(np.array([1, 1], np.int64), "split")],
                                    [u6, u3], name="unstack", axis=0))
    else:
        b.nodes.append(oh.make_node("Split", [lg], [u6, u3], name="unstack", axis=0))
    b.squeeze(u3, [0], "unstack:1", "unstack__490")
    b.squeeze(u6, [0], "unstack:0", "unstack__488")

    graph = oh.make_graph(b.nodes, os.path.splitext(os.path.basename(path))[0],
                          inputs, outputs, initializer=b.inits)
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", opset)])
    model.ir_version = 8 if opset < 20 else 9
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return path


def random_feeds(model_path, seed=0, n_pad=None):
    """Plausible BERT inputs for a tiny model: [CLS] q [SEP] ctx [SEP] + padding."""
    m = onnx.load(model_path)
    S = [d.dim_value for d in m.graph.input[1].type.tensor_type.shape.dim][1]
    rng = np.random.default_rng(seed)
    n = S - (n_pad if n_pad is not None else S // 4)
    ids = np.zeros((1, S), np.int64)
    ids[0, :n] = rng.integers(1, VOCAB, n)
    seg = np.zeros((1, S), np.int64)
    seg[0, n // 2:n] = 1
    mask = np.zeros((1, S), np.int64)
    mask[0, :n] = 1
    return {"unique_ids_raw_output___9:0": np.array([7], np.int64), "segment_ids:0": seg,
            "input_mask:0": mask, "input_ids:0": ids}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=_DEFAULT_OUT)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    for i, (fname, h, nh, nl, s, gelu, style, opset) in enumerate(VARIANTS):
        p = make_bert(os.path.join(args.out_dir, fname), h, nh, nl, s, gelu, style, opset, seed=i)
        print(f"  {p}")


if __name__ == "__main__":
    main()
