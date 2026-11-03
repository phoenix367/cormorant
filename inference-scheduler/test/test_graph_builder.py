# Copyright 2026 Ivan Gubochkin
# SPDX-License-Identifier: Apache-2.0

"""The frontends' shared plumbing, src/graph_builder.py (doc/plans/CHAT_PLAN.md
§30): the entry graph and its metadata, the exponent tables, the attention
helpers — and the interleaved RoPE permutation of src/rope.py."""

import unittest

import numpy as np
from onnx import TensorProto

from src import numeric
from src.graph_builder import (F, ExponentTable, GraphBuilder, entry_info, expand_exp, group_order,
                               pv_exp, score_exp)
from src.rope import RopeSpec, half_tables, perm


class Tiny(GraphBuilder):
    PRODUCER = "test"

    def build(self):
        self._new("e")
        x = self._t("x", [2, 4], host="f32")
        self._t("x", [9, 9], exp=3)                    # the first shape wins, the metadata adds up
        self._init("w", np.ones((4, 3), np.float64))
        self._init("w", np.zeros((4, 3)))              # the first initializer of a name wins
        self._int_init("axis", np.array([0]))
        y = self._matmul(x, "w", np.ones((4, 3), np.float32), 2, "y", [5, 5, 6], "mm")
        s = self._t("s", [1], state=True, dma="i32")
        self._t("s", [1], state=True)
        self._node("LlmToHost", [y], [s], "copy")
        return self._make_model("tiny_e", [x], [s], {"entry": "e", "rows": 2})


class TestGraphBuilder(unittest.TestCase):

    def test_model(self):
        m = Tiny().build()
        g = m.graph
        self.assertEqual(m.producer_name, "test")
        self.assertEqual([i.name for i in g.input], ["x"])
        self.assertEqual([d.dim_value for d in g.input[0].type.tensor_type.shape.dim], [2, 4])
        self.assertEqual([v.name for v in g.value_info], ["y"])
        self.assertEqual([(t.name, t.data_type) for t in g.initializer],
                         [("w", TensorProto.FLOAT), ("axis", TensorProto.INT64)])
        self.assertEqual(g.initializer[0].float_data or list(np.frombuffer(g.initializer[0].raw_data, "<f4")),
                         [1.0] * 12)
        self.assertEqual([(n.op_type, n.domain) for n in g.node], [("MatMul", ""), ("LlmToHost", "")])
        meta = numeric.parse(m)
        self.assertEqual(meta["exp"], {"x": 3, "y": [5, 5, 6]})
        self.assertEqual(meta["state"], ["s"])
        self.assertEqual((meta["host"], meta["dma"]), ({"x": "f32"}, {"s": "i32"}))
        self.assertEqual(entry_info(m), {"entry": "e", "rows": 2})
        self.assertIsNone(entry_info(m, "axi.tts.entry"))


class TestExponents(unittest.TestCase):

    def test_expand(self):
        self.assertEqual(expand_exp(7, 3).tolist(), [7, 7, 7])
        self.assertEqual(expand_exp([1, 2, 3], 3).tolist(), [1, 2, 3])
        self.assertEqual(expand_exp([1, 2], 4).tolist(), [1, 1, 2, 2])          # per head
        with self.assertRaises(ValueError):
            expand_exp([1, 2], 3)

    def test_table(self):
        t = ExponentTable({"exponents": {"q@0": 9}, "policy": "p"})
        self.assertEqual(t.lookup("q@0", 2).tolist(), [9, 9])
        with self.assertRaises(KeyError):
            t.lookup("k@0", 2)

    def test_attention_exponents(self):
        fq, fk = np.array([10, 11, 12, 13]), np.array([7, 9])            # 4 heads over 2 KV heads
        self.assertEqual(score_exp(fq, fk, 2).tolist(), [10 + 7 - F, 11 + 7 - F, 12 + 9 - F, 13 + 9 - F])
        fp, fv = np.array([12, 12, 13, 13]), np.arange(4)                  # head dim 2
        self.assertEqual(pv_exp(fp, fv, 2, 2).tolist(),
                         [12 + 0 - F, 12 + 1 - F, 12 + 0 - F, 12 + 1 - F,
                          13 + 2 - F, 13 + 3 - F, 13 + 2 - F, 13 + 3 - F])


class TestGroupOrder(unittest.TestCase):

    def test_orders(self):
        def seq(n, order):
            return " ".join(f"{s[0]}{g}" for s, g in group_order(n, order))
        self.assertEqual(seq(3, "ahead1"), "q0 q1 s0 p0 s1 q2 p1 s2 p2")
        self.assertEqual(seq(4, "ahead2"), "q0 q1 s0 p0 q2 s1 p1 q3 s2 p2 s3 p3")
        for order in ("ahead1", "ahead2"):
            self.assertEqual(seq(1, order), "q0 s0 p0")
            for n in (1, 2, 5):
                steps = group_order(n, order)
                self.assertEqual(sorted(steps), sorted((s, g) for g in range(n) for s in ("qk", "softmax", "pv")))
                for g in range(n):        # every group: scores, then its softmax, then P.V
                    self.assertLess(steps.index(("qk", g)), steps.index(("softmax", g)))
                    self.assertLess(steps.index(("softmax", g)), steps.index(("pv", g)))
        with self.assertRaises(ValueError):
            group_order(2, "sideways")


class TestRopePerm(unittest.TestCase):

    def test_interleaved(self):
        """Pair j = dims (2j, 2j + 1) to (j, j + hd/2); the pass-through dims fill the rest."""
        o = perm(RopeSpec(8, 1e4, rot=4, interleaved=True))
        self.assertEqual(o.tolist(), [0, 2, 4, 5, 1, 3, 6, 7])
        self.assertEqual(perm(RopeSpec(8, 1e4, rot=4)).tolist(), [0, 1, 4, 5, 2, 3, 6, 7])
        self.assertEqual(perm(RopeSpec(8, 1e4)).tolist(), list(range(8)))

    def test_padded_tables(self):
        """A head padded beyond its rotated dims: the extra pairs cos 1, sin 0."""
        c, s = half_tables(RopeSpec(16, 1e4, rot=4), 3)
        self.assertEqual(c.shape, (3, 8))
        self.assertTrue((c[:, 2:] == 1).all() and (s[:, 2:] == 0).all())
        c0, s0 = half_tables(RopeSpec(4, 1e4), 3)
        np.testing.assert_array_equal(c[:, :2], c0)
        np.testing.assert_array_equal(s[:, :2], s0)


if __name__ == "__main__":
    unittest.main()
