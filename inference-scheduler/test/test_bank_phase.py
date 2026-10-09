"""DRAM bank phases in the DMA pool layout (src/bank_phase.py,
doc/plans/PS_PORTS_PLAN.md §11): the operand streams of a binary VectorOP
start in different DDR banks (byte address bits 14–15), the other slots keep
their 64-byte packing, `bank_phase=False` is the old layout, and the project
compiled on the host is bit-exact with the simulation."""

import shutil
import tempfile
import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

import host_emu
from src.bank_phase import BANK_PERIOD, BANK_STRIDE, bank_phase, place_slots
from src.codegen import CodeGenerator
from src.codegen.multi import MultiEntryGenerator
from src.graph import OnnxGraph


def ew_model(n=64 * 1024, seed=0, prefix=""):
    """x -> Add(x, w1) -> t -> Mul(t, x) -> u -> Relu(u) -> v -> Add(v, t) -> y:
    four VectorOP calls on n-element (128 KB) tensors, two binary ones with
    three streams each, one unary, and a binary one whose streams are two
    intermediates and the output."""
    rng = np.random.default_rng(seed)
    w1 = (rng.standard_normal(n) * 0.5).astype(np.float32)
    p = prefix
    nodes = [
        helper.make_node("Add", [p + "x", p + "w1"], [p + "t"]),
        helper.make_node("Mul", [p + "t", p + "x"], [p + "u"]),
        helper.make_node("Relu", [p + "u"], [p + "v"]),
        helper.make_node("Add", [p + "v", p + "t"], [p + "y"]),
    ]
    g = helper.make_graph(
        nodes, "bank_phase",
        [helper.make_tensor_value_info(p + "x", TensorProto.FLOAT, [n])],
        [helper.make_tensor_value_info(p + "y", TensorProto.FLOAT, [n])],
        initializer=[numpy_helper.from_array(w1, p + "w1")])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    onnx.checker.check_model(m)
    return m


def phases(cg):
    """{name: bank phase of its pool start}, the pool total in elements, and
    the generator's operand groups."""
    layout, total = cg._compute_pool_layout()
    bpe = cg._dtype.bytes_per_elem
    return {n: bank_phase(off * bpe) for n, off, _a in layout}, total, cg._bank_groups


class TestPlaceSlots(unittest.TestCase):

    def test_unconstrained_slots_pack(self):
        slots = [(100, ["a"]), (200, ["b"]), (300, ["c"])]
        self.assertEqual(place_slots(slots, [], 2), [0, 100, 300])

    def test_three_streams_get_three_phases(self):
        k = 64 * 1024                     # 128 KB slots: multiples of the bank period
        slots = [(k, ["a"]), (k, ["b"]), (k, ["c"])]
        starts = place_slots(slots, [[("a", 0), ("b", 0), ("c", 0)]], 2)
        ph = [bank_phase(s * 2) for s in starts]
        self.assertEqual(len(set(ph)), 3)
        # each padded by at most 48 KB
        for i, s in enumerate(starts):
            self.assertLessEqual(s * 2 - sum(a for a, _n in slots[:i]) * 2, BANK_PERIOD - BANK_STRIDE)

    def test_view_shift_counts(self):
        # b is a view 16 KB into its root: the phase of root + 16 KB must differ from a's
        k = 64 * 1024
        slots = [(k, ["a"]), (k, ["root"])]
        starts = place_slots(slots, [[("a", 0), ("root", BANK_STRIDE)]], 2)
        self.assertNotEqual(bank_phase(starts[0] * 2), bank_phase(starts[1] * 2 + BANK_STRIDE))

    def test_fixed_partner(self):
        k = 64 * 1024
        starts = place_slots([(k, ["t"])], [[("t", 0), ("w", 0)]], 2, base_elems=0, fixed={"w": 0})
        self.assertNotEqual(bank_phase(starts[0] * 2), 0)


class TestLayout(unittest.TestCase):

    def setUp(self):
        self.g = OnnxGraph(ew_model())

    def test_groups(self):
        groups = CodeGenerator(self.g, model_path="m.onnx")._bank_groups
        # Add(x, w1): x is the caller's -> (w1, t); Mul(t, x) -> (t, u); Relu -> (u, v); Add(v, t) -> (v, t): y is the caller's too
        names = [sorted(n for n, _s in g) for g in groups]
        self.assertIn(["t", "w1"], names)
        self.assertIn(["t", "u"], names)
        self.assertIn(["u", "v"], names)
        self.assertIn(["t", "v"], names)

    def test_phases_distinct_per_group(self):
        ph, _total, groups = phases(CodeGenerator(self.g, model_path="m.onnx"))
        for g in groups:
            got = [ph[n] for n, _s in g]
            self.assertEqual(len(set(got)), len(g), (g, got))

    def test_off_is_the_old_packing(self):
        cg = CodeGenerator(self.g, model_path="m.onnx", bank_phase=False)
        self.assertEqual(cg._bank_groups, [])
        layout, total = cg._compute_pool_layout()
        bpe = cg._dtype.bytes_per_elem
        self.assertEqual(sum(((a * bpe + 63) & ~63) for _n, _o, a in layout) // bpe, total)

    def test_growth_bounded(self):
        on = CodeGenerator(self.g, model_path="m.onnx")
        off = CodeGenerator(self.g, model_path="m.onnx", bank_phase=False)
        _l, t_on = on._compute_pool_layout()
        _l, t_off = off._compute_pool_layout()
        slots = len(_l)
        self.assertLessEqual((t_on - t_off) * 2, slots * (BANK_PERIOD - BANK_STRIDE))
        self.assertGreaterEqual(on._compute_pool_bytes(), t_on * 2)

    def test_small_streams_ignored(self):
        g = OnnxGraph(ew_model(n=1024))            # 2 KB tensors: no groups, no padding
        cg = CodeGenerator(g, model_path="m.onnx")
        self.assertEqual(cg._bank_groups, [])

    def test_multi_entry(self):
        entries = [("e1", OnnxGraph(ew_model(prefix="e1_"))), ("e2", OnnxGraph(ew_model(seed=1, prefix="e2_")))]
        gen = MultiEntryGenerator(entries, "bank_multi")
        layout, _total = gen.pool_layout()
        bpe = gen._dtype.bytes_per_elem
        ph = {n: bank_phase(off * bpe) for n, off, _a in layout}
        for name, _g in entries:
            for grp in gen.cgs[name]._bank_groups:
                got = [ph[n] for n, _s in grp]
                self.assertEqual(len(set(got)), len(grp), (name, grp, got))


class TestHostEmu(unittest.TestCase):
    """The generated project on the host, bit-exact with the simulation
    (test_inference compares every element) — the phases only move data."""

    def test_bit_exact(self):
        if host_emu.which_cc() is None:
            self.fail("no C compiler")
        cg = CodeGenerator(OnnxGraph(ew_model(n=32 * 1024, seed=3)), model_path="bank.onnx")
        self.assertTrue(cg._bank_groups)
        d = tempfile.mkdtemp(prefix="bank_phase_emu_")
        try:
            rc, out = host_emu.build_and_run(cg, d)
            self.assertEqual(rc, 0, out[-3000:])
            self.assertEqual(host_emu.failures(out), [], out[-3000:])
        finally:
            shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
