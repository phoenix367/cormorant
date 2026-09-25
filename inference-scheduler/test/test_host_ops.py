"""Host-CPU ops (BERT_PLAN 1b / 1e): the generated C helpers against the
simulator's reference, bit for bit.

Every case builds a one-op ONNX model, lets the scheduler generate its
host-op section (``CodeGenerator._host_ops_section`` — exactly the C that
lands in inference.c), compiles it into a tiny harness that runs the node's
``c_call`` on random Data_t inputs, and compares the raw output with
``HostNode.reference`` (what ``_simulate`` uses).  The write-back rounding
(round half to even + saturation) is tested on exact ties.
"""

import os
import shutil
import subprocess
import tempfile
import unittest

import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as nph
from onnx import TensorProto

from src.codegen import CodeGenerator
from src.dtype import AP_FIXED_16_8
from src.graph import OnnxGraph
from src.host_nodes import (CastNode, GatherNode, GeluNode, HostNode, LayerNormNode,
                            OneHotNode, SliceNode, TransposeNode)
from src.nodes import ReshapeNode, SchedulerError

DT = AP_FIXED_16_8
_CC = shutil.which("cc") or shutil.which("gcc")

_HARNESS_HEAD = r"""
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
typedef uint16_t Data_t;
#define INFERENCE_BYTES_PER_ELEM 2u
typedef struct { void *virt; unsigned count; uint8_t cached; } inference_buf_t;
static Data_t *inference_buf_ptr(inference_buf_t *b) { return (Data_t *)b->virt; }
static inline int inference_buf_is_cached(const inference_buf_t *b) { return b->cached; }
static void inference_buf_sync_to_device(inference_buf_t *b) { (void)b; }
"""


def _vi(name, shape, et=TensorProto.FLOAT):
    return oh.make_tensor_value_info(name, et, shape)


def _save(tmp, nodes, inputs, outputs, inits=(), opset=13, name="m"):
    g = oh.make_graph(nodes, name, inputs, outputs, initializer=list(inits))
    m = oh.make_model(g, opset_imports=[oh.make_opsetid("", opset)])
    m.ir_version = 8 if opset < 20 else 9
    onnx.checker.check_model(m)
    p = os.path.join(tmp, f"{name}.onnx")
    onnx.save(m, p)
    return p


def _grid(rng, shape, lo, hi):
    return np.round(rng.uniform(lo, hi, shape) * 256) / 256


def _storage(t, v):
    return DT.int_to_storage(v) if t.is_int else DT.float_to_storage(v)


def _run_c(cg, sn, ins):
    """Compile the generated host section + a harness around ``sn.c_call``;
    returns the raw output storage."""
    staged = sn.staged_inputs()
    direct = sn.direct_inputs()
    by_name = {t.onnx_name: v for t, v in zip(sn.inputs, ins, strict=True)}
    lines = [_HARNESS_HEAD, cg._host_ops_section(), "int main(void)", "{"]
    for i, t in enumerate(staged):
        lines.append(f"    static Data_t in{i}[{max(t.numel, 1)}];")
    for i, t in enumerate(direct):
        lines.append(f"    static Data_t dd{i}[{max(t.numel, 1)}];")
        lines.append(f"    inference_buf_t db{i} = {{ dd{i}, {t.numel}u, 1u }};")
    lines.append(f"    static Data_t out[{max(sn.output.numel, 1)}];")
    lines.append(f"    static double tmp_d[{max(sn.scratch_bytes() // 8, 1)}];")
    lines.append("    void *tmp = tmp_d;")
    for i, t in enumerate(staged):
        lines.append(f"    if (fread(in{i}, 2, {t.numel}, stdin) != {t.numel}u) return 2;")
    for i, t in enumerate(direct):
        lines.append(f"    if (fread(dd{i}, 2, {t.numel}, stdin) != {t.numel}u) return 2;")
    lines += ["    " + ln for ln in sn.c_call([f"in{i}" for i in range(len(staged))], "out",
                                              "tmp", [f"(&db{i})" for i in range(len(direct))],
                                              DT)]
    lines += ["    (void)tmp;",
              f"    fwrite(out, 2, {sn.output.numel}, stdout);", "    return 0;", "}"]
    blob = b"".join(_storage(t, by_name[t.onnx_name]).astype("<u2").tobytes()
                    for t in staged + direct)
    with tempfile.TemporaryDirectory() as td:
        c = os.path.join(td, "h.c")
        with open(c, "w") as f:
            f.write("\n".join(lines))
        exe = os.path.join(td, "h")
        r = subprocess.run([_CC, "-std=gnu99", "-O2", "-Wall", "-Wextra", "-Werror",
                            "-Wno-unused-function", c, "-lm", "-o", exe],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        r = subprocess.run([exe], input=blob, capture_output=True)
        assert r.returncode == 0, r.stderr
    return np.frombuffer(r.stdout, dtype="<u2")


@unittest.skipUnless(_CC, "C compiler not available on host")
class _Base(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.d = cls._tmp.name

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def check(self, path, feeds, kind=HostNode, **graph_kw):
        """Every host node of the model: C output == reference, bitwise."""
        g = OnnxGraph(path, **graph_kw)
        cg = CodeGenerator(g, model_path=path)
        arrays = cg._forward_pass({k: np.asarray(v, np.float64) for k, v in feeds.items()})
        nodes = [sn for sn in g.nodes if isinstance(sn, kind)]
        self.assertTrue(nodes)
        for sn in nodes:
            ins = [arrays[t.onnx_name] for t in sn.inputs]
            ref = _storage(sn.output, arrays[sn.output.onnx_name].reshape(-1))
            got = _run_c(cg, sn, ins)
            np.testing.assert_array_equal(got, ref, err_msg=f"{sn.onnx_node.op_type} [{sn.index}]")
        return g, cg, arrays


class TestRounding(_Base):
    """host_st: round half to even + saturation, NaN -> 0; host_st_int:
    truncate toward zero + saturation."""

    def test_host_st_matches_host_quantize(self):
        vals = np.array([0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 3.49, 3.51,
                         255.0 * 256 / 256 + 0.5, 32767.4, 32767.5, 32768.0, -32768.5,
                         -40000.0, 1e300, -1e300, np.nan, 0.0, -0.0]) / 256.0
        prog = [_HARNESS_HEAD, DT.c_host_conversions(), "int main(void)", "{",
                "    double v; Data_t d;",
                "    while (fread(&v, 8, 1, stdin) == 1) { d = host_st(v); fwrite(&d, 2, 1, stdout);",
                "        d = host_st_int(v * 256.0); fwrite(&d, 2, 1, stdout); }",
                "    return 0;", "}"]
        with tempfile.TemporaryDirectory() as td:
            c, exe = os.path.join(td, "r.c"), os.path.join(td, "r")
            with open(c, "w") as f:
                f.write("\n".join(prog))
            subprocess.run([_CC, "-O2", c, "-lm", "-o", exe], check=True)
            out = subprocess.run([exe], input=vals.astype("<f8").tobytes(),
                                 capture_output=True, check=True).stdout
        got = np.frombuffer(out, "<u2").reshape(-1, 2)
        np.testing.assert_array_equal(got[:, 0], DT.float_to_storage(DT.host_quantize(vals)))
        np.testing.assert_array_equal(got[:, 1], DT.int_to_storage(DT.int_quantize(vals * 256)))
        # ties go to even, not away from zero
        q = DT.host_quantize(np.array([0.5, 1.5, 2.5, -0.5, -1.5]) / 256) * 256
        np.testing.assert_array_equal(q, [0, 2, 2, 0, -2])


class TestSoftmax(_Base):

    def test_last_axis_random(self):
        p = _save(self.d, [oh.make_node("Softmax", ["X"], ["Y"], axis=-1)],
                  [_vi("X", [3, 5, 40])], [_vi("Y", [3, 5, 40])], name="sm")
        x = _grid(np.random.default_rng(0), (3, 5, 40), -8, 8)
        _, _, a = self.check(p, {"X": x})
        ref = np.exp(x - x.max(-1, keepdims=True))
        ref /= ref.sum(-1, keepdims=True)
        self.assertLessEqual(np.abs(a["Y"] - ref).max(), 0.5 / 256 + 1e-12)

    def test_ties_round_to_even(self):
        # 512 equal logits -> p = 1/512 = 0.5 LSB: a tie, rounds to 0 (even)
        p = _save(self.d, [oh.make_node("Softmax", ["X"], ["Y"], axis=-1)],
                  [_vi("X", [2, 512])], [_vi("Y", [2, 512])], name="sm_tie")
        x = np.zeros((2, 512))
        x[1, :] = 3.0
        _, _, a = self.check(p, {"X": x})
        self.assertTrue(np.all(a["Y"] == 0.0))

    def test_opset11_coerce_2d(self):
        # opset < 13: axis=1 on [2,3,4] softmaxes over the 12 trailing elements
        p = _save(self.d, [oh.make_node("Softmax", ["X"], ["Y"], axis=1)],
                  [_vi("X", [2, 3, 4])], [_vi("Y", [2, 3, 4])], opset=11, name="sm11")
        x = _grid(np.random.default_rng(1), (2, 3, 4), -4, 4)
        g, _, a = self.check(p, {"X": x})
        sn = g.nodes[0]
        self.assertEqual((sn.rows, sn.n), (2, 12))
        e = np.exp(x.reshape(2, 12) - x.reshape(2, 12).max(1, keepdims=True))
        self.assertLessEqual(np.abs(a["Y"].reshape(2, 12) - e / e.sum(1, keepdims=True)).max(),
                             0.5 / 256 + 1e-12)

    def test_opset13_non_last_axis_rejected(self):
        p = _save(self.d, [oh.make_node("Softmax", ["X"], ["Y"], axis=1)],
                  [_vi("X", [2, 3, 4])], [_vi("Y", [2, 3, 4])], name="sm13bad")
        with self.assertRaisesRegex(SchedulerError, "only the last axis"):
            OnnxGraph(p)


class TestLayerNormGelu(_Base):

    def test_native_layernorm(self):
        rng = np.random.default_rng(2)
        g_ = nph.from_array(rng.normal(1, 0.2, 24).astype(np.float32), "g")
        b_ = nph.from_array(rng.normal(0, 0.2, 24).astype(np.float32), "b")
        p = _save(self.d, [oh.make_node("LayerNormalization", ["X", "g", "b"], ["Y"],
                                        axis=-1, epsilon=1e-5)],
                  [_vi("X", [2, 5, 24])], [_vi("Y", [2, 5, 24])], [g_, b_], opset=17,
                  name="ln")
        x = _grid(rng, (2, 5, 24), -6, 6)
        g, _, a = self.check(p, {"X": x})
        self.assertIsInstance(g.nodes[0], LayerNormNode)
        self.assertEqual(g.nodes[0].tf_form, 0)
        m, v = x.mean(-1, keepdims=True), x.var(-1, keepdims=True)
        ref = (x - m) / np.sqrt(v + 1e-5) * nph.to_array(g_) + nph.to_array(b_)
        self.assertLessEqual(np.abs(a["Y"] - ref).max(), 0.5 / 256 + 1e-9)

    def test_native_layernorm_no_bias_axis1(self):
        rng = np.random.default_rng(3)
        g_ = nph.from_array(rng.normal(1, 0.2, (4, 6)).astype(np.float32), "g")
        p = _save(self.d, [oh.make_node("LayerNormalization", ["X", "g"], ["Y"], axis=1)],
                  [_vi("X", [3, 4, 6])], [_vi("Y", [3, 4, 6])], [g_], opset=17, name="ln2")
        g, _, _ = self.check(p, {"X": _grid(rng, (3, 4, 6), -3, 3)})
        self.assertEqual((g.nodes[0].rows, g.nodes[0].n), (3, 24))
        self.assertIsNone(g.nodes[0].beta)

    def test_native_gelu_both_forms(self):
        for approx in ("tanh", "none"):
            p = _save(self.d, [oh.make_node("Gelu", ["X"], ["Y"], approximate=approx)],
                      [_vi("X", [4, 64])], [_vi("Y", [4, 64])], opset=20, name=f"gelu_{approx}")
            x = _grid(np.random.default_rng(4), (4, 64), -10, 10)
            g, _, a = self.check(p, {"X": x})
            self.assertIsInstance(g.nodes[0], GeluNode)
            import math
            erf = np.vectorize(math.erf)
            ref = (0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))
                   if approx == "tanh" else 0.5 * x * (1 + erf(x / np.sqrt(2))))
            self.assertLessEqual(np.abs(a["Y"] - ref).max(), 0.5 / 256 + 1e-9)


class TestDataMovement(_Base):

    def test_transpose_perms(self):
        rng = np.random.default_rng(5)
        for perm, shape in [([0, 2, 1, 3], [1, 6, 3, 8]), ([0, 2, 3, 1], [1, 6, 3, 8]),
                            ([2, 0, 1], [1, 6, 2]), ([4, 2, 0, 3, 1], [2, 3, 1, 4, 5]),
                            ([1, 0], [5, 7])]:
            out = [shape[i] for i in perm]
            p = _save(self.d, [oh.make_node("Transpose", ["X"], ["Y"], perm=perm)],
                      [_vi("X", shape)], [_vi("Y", out)], name="tr" + "".join(map(str, perm)))
            x = _grid(rng, shape, -100, 100)
            g, _, a = self.check(p, {"X": x})
            self.assertIsInstance(g.nodes[0], TransposeNode)
            np.testing.assert_array_equal(a["Y"], np.transpose(x, perm))

    def test_slice_copy_steps(self):
        st = [nph.from_array(np.array(v, np.int64), n) for n, v in
              [("s", [1, 0]), ("e", [4, 100]), ("ax", [0, 2]), ("sp", [2, 3])]]
        p = _save(self.d, [oh.make_node("Slice", ["X", "s", "e", "ax", "sp"], ["Y"])],
                  [_vi("X", [5, 3, 8])], [_vi("Y", [2, 3, 3])], st, name="slice")
        x = _grid(np.random.default_rng(6), (5, 3, 8), -50, 50)
        g, _, a = self.check(p, {"X": x})
        self.assertIsInstance(g.nodes[0], SliceNode)
        self.assertFalse(g.nodes[0].is_view)
        np.testing.assert_array_equal(a["Y"], x[1:4:2, :, 0:8:3])

    def test_gather_clamps_indices(self):
        tbl = nph.from_array(np.random.default_rng(7).normal(0, 2, (10, 3, 4)).astype(np.float32), "T")
        p = _save(self.d, [oh.make_node("Gather", ["T", "I"], ["Y"], axis=0)],
                  [_vi("I", [2, 6], TensorProto.INT64)], [_vi("Y", [2, 6, 3, 4])], [tbl],
                  name="gather")
        idx = np.array([[0, 9, -1, -10, 12, -11], [3, 3, 5, 1, 0, 2]])
        g, _, a = self.check(p, {"I": idx})
        sn = g.nodes[0]
        self.assertIsInstance(sn, GatherNode)
        self.assertEqual(sn.direct_inputs()[0].onnx_name, "T")
        t = DT.quantize(nph.to_array(tbl).astype(np.float64))
        k = np.clip(np.where(idx < 0, idx + 10, idx), 0, 9)       # -1 -> 9, 12 -> 9, -11 -> 0
        np.testing.assert_array_equal(a["Y"], t[k])

    def test_onehot(self):
        inits = [nph.from_array(np.array([5], np.int32), "depth"),
                 nph.from_array(np.array([-0.5, 2.25], np.float32), "vals")]
        p = _save(self.d, [oh.make_node("OneHot", ["I", "depth", "vals"], ["Y"], axis=-1)],
                  [_vi("I", [7], TensorProto.INT64)], [_vi("Y", [7, 5])], inits, name="onehot")
        idx = np.array([0, 4, -1, -5, 5, -6, 2])
        g, _, a = self.check(p, {"I": idx})
        self.assertIsInstance(g.nodes[0], OneHotNode)
        ref = np.full((7, 5), -0.5)
        for i, k in enumerate(idx):
            k = k + 5 if k < 0 else k
            if 0 <= k < 5:
                ref[i, k] = 2.25
        np.testing.assert_array_equal(a["Y"], ref)

    def test_cast_modes(self):
        cases = [(TensorProto.INT64, TensorProto.FLOAT, np.array([0, 1, -3, 127, 128, -129, 30000])),
                 (TensorProto.FLOAT, TensorProto.INT64, np.array([0.0, 1.5, -1.5, -0.99, 100.25])),
                 (TensorProto.FLOAT, TensorProto.BOOL, np.array([0.0, 0.00390625, -2.0])),
                 (TensorProto.INT32, TensorProto.BOOL, np.array([0, 5, -1]))]
        for i, (src, dst, v) in enumerate(cases):
            p = _save(self.d, [oh.make_node("Cast", ["X"], ["Y"], to=dst)],
                      [_vi("X", [len(v)], src)], [_vi("Y", [len(v)], dst)], name=f"cast{i}")
            g, _, a = self.check(p, {"X": v})
            self.assertIsInstance(g.nodes[0], CastNode)
            if dst == TensorProto.FLOAT:
                np.testing.assert_array_equal(a["Y"], np.clip(v, -128, 32767 / 256))
            elif dst == TensorProto.INT64:
                np.testing.assert_array_equal(a["Y"], np.trunc(v))
            else:
                np.testing.assert_array_equal(a["Y"], (v != 0).astype(float))

    def test_same_kind_cast_is_alias(self):
        p = _save(self.d, [oh.make_node("Cast", ["X"], ["Y"], to=TensorProto.INT32),
                           oh.make_node("Cast", ["Y"], ["Z"], to=TensorProto.FLOAT)],
                  [_vi("X", [4], TensorProto.INT64)], [_vi("Z", [4])], name="cast_alias")
        g = OnnxGraph(p)
        self.assertIsInstance(g.nodes[0], ReshapeNode)
        self.assertIsInstance(g.nodes[1], CastNode)

    def test_float_host_ops_reject_integer_input(self):
        p = _save(self.d, [oh.make_node("Softmax", ["X"], ["Y"], axis=-1)],
                  [_vi("X", [2, 4], TensorProto.INT64)], [_vi("Y", [2, 4])], name="sm_int")
        with self.assertRaisesRegex(SchedulerError, "integer tensor"):
            OnnxGraph(p)


class TestStagingHelpers(_Base):
    """host_load / host_store compact and re-expand advancing-strided layouts;
    host_in / host_out / host_out_done work in place on a cacheable flat
    buffer and stage otherwise."""

    def test_strided_round_trip(self):
        g = OnnxGraph(_save(self.d, [oh.make_node("Softmax", ["X"], ["Y"], axis=-1)],
                            [_vi("X", [2, 4])], [_vi("Y", [2, 4])], name="stage"))
        cg = CodeGenerator(g, model_path="stage")
        prog = [_HARNESS_HEAD, cg._host_ops_section(), "int main(void)", "{",
                "    static Data_t buf[40], st[40], out[40]; unsigned i;",
                "    inference_buf_t b = { buf }, o = { out };",
                "    for (i = 0; i < 40u; i++) { buf[i] = (Data_t)(i + 1u); out[i] = 0xAAAAu; }",
                "    host_load(st, &b, 5u, 3u, 8u);        /* 5 chunks of 3 at stride 8 */",
                "    fwrite(st, 2, 15, stdout);",
                "    host_store(&o, st, 5u, 3u, 8u);",
                "    fwrite(out, 2, 40, stdout);", "    return 0;", "}"]
        with tempfile.TemporaryDirectory() as td:
            c, exe = os.path.join(td, "s.c"), os.path.join(td, "s")
            with open(c, "w") as f:
                f.write("\n".join(prog))
            subprocess.run([_CC, "-O2", "-Wall", "-Wno-unused-function", c, "-lm", "-o", exe],
                           check=True)
            out = np.frombuffer(subprocess.run([exe], capture_output=True, check=True).stdout, "<u2")
        compact = np.array([c * 8 + j + 1 for c in range(5) for j in range(3)])
        np.testing.assert_array_equal(out[:15], compact)
        expanded = np.zeros(40, np.uint16)
        for c in range(5):
            expanded[c * 8:c * 8 + 3] = compact[c * 3:c * 3 + 3]
        np.testing.assert_array_equal(out[15:], expanded)      # gaps zeroed

    def test_direct_or_staged(self):
        g = OnnxGraph(_save(self.d, [oh.make_node("Softmax", ["X"], ["Y"], axis=-1)],
                            [_vi("X", [2, 4])], [_vi("Y", [2, 4])], name="stage2"))
        cg = CodeGenerator(g, model_path="stage2")
        prog = [_HARNESS_HEAD.replace("DATA_T", "uint16_t"), cg._host_ops_section(),
                "int main(void)", "{",
                "    static Data_t buf[40], st[40], out[40]; unsigned i, c;",
                "    for (c = 0; c < 2u; c++) {",
                "        inference_buf_t b = { buf, 40u, (uint8_t)c }, o = { out, 40u, (uint8_t)c };",
                "        const Data_t *in; Data_t *y;",
                "        for (i = 0; i < 40u; i++) { buf[i] = (Data_t)(i + 1u); out[i] = 0xAAAAu; }",
                "        memset(st, 0, sizeof st);",
                "        in = host_in(&b, st, 1u, 40u, 40u);            /* flat */",
                "        printf(\"%d \", in == buf);",
                "        in = host_in(&b, st, 5u, 3u, 8u);              /* strided: always staged */",
                "        printf(\"%d %u \", in == st, (unsigned)in[3]);",
                "        y = host_out(&o, st, 1u, 40u, 40u);",
                "        printf(\"%d \", y == out);",
                "        for (i = 0; i < 40u; i++) y[i] = (Data_t)(100u + i);",
                "        host_out_done(&o, y, 1u, 40u, 40u);",
                "        printf(\"%u %u\\n\", (unsigned)out[0], (unsigned)out[39]);",
                "    }",
                "    return 0;", "}"]
        with tempfile.TemporaryDirectory() as td:
            c, exe = os.path.join(td, "s.c"), os.path.join(td, "s")
            with open(c, "w") as f:
                f.write("\n".join(prog))
            subprocess.run([_CC, "-O2", "-Wall", "-Wno-unused-function", c, "-lm",
                            "-o", exe], check=True)
            out = subprocess.run([exe], capture_output=True, check=True, text=True).stdout
        # cached = 0: staged in / out (copied back); cached = 1: in place
        self.assertEqual(out.splitlines(), ["0 1 9 0 100 139", "1 1 9 1 100 139"])

    def test_strided_round_trip(self):
        g = OnnxGraph(_save(self.d, [oh.make_node("Softmax", ["X"], ["Y"], axis=-1)],
                            [_vi("X", [2, 4])], [_vi("Y", [2, 4])], name="stage"))
        cg = CodeGenerator(g, model_path="stage")
        prog = [_HARNESS_HEAD.replace("DATA_T", "uint16_t"), cg._host_ops_section(),
                "int main(void)", "{",
                "    static Data_t buf[40], st[40], out[40]; unsigned i;",
                "    inference_buf_t b = { buf }, o = { out };",
                "    for (i = 0; i < 40u; i++) { buf[i] = (Data_t)(i + 1u); out[i] = 0xAAAAu; }",
                "    host_load(st, &b, 5u, 3u, 8u);        /* 5 chunks of 3 at stride 8 */",
                "    fwrite(st, 2, 15, stdout);",
                "    host_store(&o, st, 5u, 3u, 8u);",
                "    fwrite(out, 2, 40, stdout);", "    return 0;", "}"]
        with tempfile.TemporaryDirectory() as td:
            c, exe = os.path.join(td, "s.c"), os.path.join(td, "s")
            with open(c, "w") as f:
                f.write("\n".join(prog))
            subprocess.run([_CC, "-O2", "-Wall", "-Wno-unused-function", c, "-lm",
                            "-o", exe], check=True)
            out = np.frombuffer(subprocess.run([exe], capture_output=True, check=True).stdout, "<u2")
        compact = np.array([c * 8 + j + 1 for c in range(5) for j in range(3)])
        np.testing.assert_array_equal(out[:15], compact)
        expanded = np.zeros(40, np.uint16)
        for c in range(5):
            expanded[c * 8:c * 8 + 3] = compact[c * 3:c * 3 + 3]
        np.testing.assert_array_equal(out[15:], expanded)      # gaps zeroed



if __name__ == "__main__":
    unittest.main()
