"""Split / Slice (zero-cost views vs host copies) and integer tensors
(BERT_PLAN 1b / 1d).  Every generated project is also run on the host
against the software kernel models (test/host_emu.py)."""

import os
import tempfile
import unittest

import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as nph
from onnx import TensorProto

import host_emu
from src.codegen import CodeGenerator
from src.graph import OnnxGraph
from src.host_nodes import SliceNode
from src.nodes import SchedulerError


def _vi(name, shape, et=TensorProto.FLOAT):
    return oh.make_tensor_value_info(name, et, shape)


class _Base(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.k = 0

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def save(self, nodes, inputs, outputs, inits=(), opset=13):
        type(self).k += 1
        g = oh.make_graph(nodes, f"s{self.k}", inputs, outputs, initializer=list(inits))
        m = oh.make_model(g, opset_imports=[oh.make_opsetid("", opset)])
        m.ir_version = 8
        onnx.checker.check_model(m)
        p = os.path.join(self._tmp.name, f"s{self.k}.onnx")
        onnx.save(m, p)
        return p

    def gen(self, p):
        g = OnnxGraph(p, fuse_act=True)
        return g, CodeGenerator(g, model_path=p)

    @unittest.skipUnless(host_emu.which_cc(), "C compiler not available on host")
    def emu_pass(self, cg):
        with tempfile.TemporaryDirectory() as td:
            rc, out = host_emu.build_and_run(cg, td)
        self.assertEqual(rc, 0, out)
        self.assertIn("test_inference PASSED", out)

    @staticmethod
    def w(name, shape, seed):
        return nph.from_array(np.random.default_rng(seed).normal(0, 0.3, shape)
                              .astype(np.float32), name)


class TestSplitViews(_Base):

    def _mm_split(self, rows, axis, sizes, to_outputs=False):
        """X[rows,16] @ W -> T, Split(T, axis) -> A, B -> Relu / Add."""
        n_split = oh.make_node("Split", ["T", "sp"], ["A", "B"], axis=axis)
        sp = nph.from_array(np.array(sizes, np.int64), "sp")
        shape_a = [sizes[0], 16] if axis == 0 else [rows, sizes[0]]
        shape_b = [sizes[1], 16] if axis == 0 else [rows, sizes[1]]
        nodes = [oh.make_node("MatMul", ["X", "W"], ["T"]), n_split]
        if to_outputs:
            outs = [_vi("A", shape_a), _vi("B", shape_b)]
        else:
            nodes += [oh.make_node("Relu", ["A"], ["Y1"]),
                      oh.make_node("Add", ["B", "B"], ["Y2"])]
            outs = [_vi("Y1", shape_a), _vi("Y2", shape_b)]
        return self.save(nodes, [_vi("X", [rows, 16])], outs, [self.w("W", [16, 16], 1), sp])

    def test_contiguous_aligned_pieces_are_views(self):
        p = self._mm_split(8, 0, [4, 4])            # offsets 0 / 64 elem = 128 B
        g, cg = self.gen(p)
        sl = [sn for sn in g.nodes if isinstance(sn, SliceNode)]
        self.assertEqual([sn.is_view for sn in sl], [True, True])
        src = cg.generate_source()
        self.assertIn("inference_buf_init_view(&_s_buf_A, T, 0u, 64u);", src)
        self.assertIn("inference_buf_init_view(&_s_buf_B, T, 64u, 64u);", src)
        self.assertNotIn("host_copy_nd(", src)
        layout, _ = cg._compute_pool_layout()
        self.assertEqual({n for n, _o, _a in layout} & {"A", "B"}, set())
        # the views' consumers keep T alive
        events = cg._compute_event_stream()
        iv = cg._compute_live_intervals()
        add_idx = next(sn.index for sn in g.nodes if sn.onnx_node.op_type == "Add")
        drain = max(i for i, e in enumerate(events) if e[0] in ("wait", "drain") and e[2] == add_idx)
        self.assertGreaterEqual(iv["T"][1], drain)
        self.assertEqual([e[0] for e in events if e[0] == "cpu"], [])
        self.emu_pass(cg)

    def test_unaligned_piece_is_copied(self):
        p = self._mm_split(8, 0, [3, 5])            # second piece at 48 elem = 96 B
        g, cg = self.gen(p)
        sl = [sn for sn in g.nodes if isinstance(sn, SliceNode)]
        self.assertEqual([sn.is_view for sn in sl], [True, False])
        self.emu_pass(cg)

    def test_non_contiguous_axis1_copies(self):
        p = self._mm_split(4, 1, [8, 8])
        g, cg = self.gen(p)
        self.assertEqual([sn.is_view for sn in g.nodes if isinstance(sn, SliceNode)],
                         [False, False])
        self.assertIn("host_copy_nd(", cg.generate_source())
        x = np.random.default_rng(3).normal(0, 1, (4, 16))
        x = np.round(x * 256) / 256
        out = cg.simulate({"X": x})
        t = cg._forward_pass({"X": x})["T"]
        np.testing.assert_array_equal(out["Y1"], np.maximum(t[:, :8], 0))
        self.emu_pass(cg)

    def test_pieces_that_are_graph_outputs_are_copied(self):
        p = self._mm_split(8, 0, [4, 4], to_outputs=True)
        g, cg = self.gen(p)
        self.assertEqual([sn.is_view for sn in g.nodes if isinstance(sn, SliceNode)],
                         [False, False])
        self.emu_pass(cg)

    def test_view_demoted_when_consumer_needs_strided_layout(self):
        # piece [4,6] + bias[6]: chunk 6 -> stride 8 layout, cannot alias T
        bias = nph.from_array(np.arange(6, dtype=np.float32) / 8, "bias")
        sp = nph.from_array(np.array([4, 4], np.int64), "sp")
        p = self.save([oh.make_node("MatMul", ["X", "W"], ["T"]),
                       oh.make_node("Split", ["T", "sp"], ["A", "B"], axis=0),
                       oh.make_node("Add", ["A", "bias"], ["Y1"]),
                       oh.make_node("Relu", ["B"], ["Y2"])],
                      [_vi("X", [8, 16])], [_vi("Y1", [4, 6]), _vi("Y2", [4, 6])],
                      [nph.from_array(np.random.default_rng(2).normal(0, .3, (16, 6))
                                      .astype(np.float32), "W"), bias, sp])
        g, cg = self.gen(p)
        a = next(sn for sn in g.nodes if isinstance(sn, SliceNode) and sn.output.onnx_name == "A")
        self.assertFalse(a.is_view)
        self.assertIn("host_store(A, out, 4u, 6u, 8u);", cg.generate_source())
        self.emu_pass(cg)


class TestIntegerTensors(_Base):

    def test_passthrough_and_cast(self):
        p = self.save([oh.make_node("Identity", ["ids"], ["ids_out"]),
                       oh.make_node("Cast", ["ids"], ["f"], to=TensorProto.FLOAT),
                       oh.make_node("Relu", ["f"], ["Y"])],
                      [_vi("ids", [2, 8], TensorProto.INT64)],
                      [_vi("ids_out", [2, 8], TensorProto.INT64), _vi("Y", [2, 8])])
        g, cg = self.gen(p)
        self.assertTrue(g.input_tensors[0].is_int)
        h = cg.generate_header()
        self.assertIn("ids (int64 [2, 8])", h)
        t = cg.generate_test()
        self.assertIn("p[i] = (Data_t)(i % 2u);", t)
        sim = cg._simulate()
        np.testing.assert_array_equal(sim["ids_out"].reshape(-1), np.arange(16) % 2)
        st = cg._expected_storage("ids_out", sim["ids_out"])
        np.testing.assert_array_equal(st, np.arange(16) % 2)          # raw ints, not x256
        st = cg._expected_storage("Y", sim["Y"])
        np.testing.assert_array_equal(st, (np.arange(16) % 2) * 256)  # Q8.8
        self.emu_pass(cg)

    def test_kernel_rejects_integer_input(self):
        p = self.save([oh.make_node("Add", ["ids", "ids"], ["Y"])],
                      [_vi("ids", [8], TensorProto.INT64)], [_vi("Y", [8], TensorProto.INT64)])
        with self.assertRaisesRegex(SchedulerError, "must go through a Cast"):
            OnnxGraph(p)

    def test_negative_values_and_gather_range(self):
        # the harness fill for ids reaching a Gather is i % rows
        tbl = nph.from_array(np.random.default_rng(5).normal(0, 1, (13, 8)).astype(np.float32), "E")
        p = self.save([oh.make_node("Gather", ["E", "ids"], ["Y"], axis=0)],
                      [_vi("ids", [40], TensorProto.INT64)], [_vi("Y", [40, 8])], [tbl])
        g, cg = self.gen(p)
        self.assertEqual(cg._int_fill_range(g.input_tensors[0]), 13)
        self.assertIn("p[i] = (Data_t)(i % 13u);", cg.generate_test())
        feeds = {"ids": np.array([-1, -13, 12, 0] * 10, np.float64)}
        y = cg.simulate(feeds)["Y"]
        e = cg._forward_pass(feeds)["E"]
        np.testing.assert_array_equal(y[:4], e[[12, 0, 12, 0]])
        self.emu_pass(cg)


if __name__ == "__main__":
    unittest.main()
