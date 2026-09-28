"""
kernel_calls() == the emitted C (src/perf_calls.py, doc/plans/TACTICS_PLAN.md
§3): the performance models, the calibration and the planner key on a
node's kernel_calls(), so for every kernel node of the test models its
calls must be exactly the run_*() calls emit_call() writes — register
values (the C arguments evaluated, header macros resolved) and counts (the
per-call for-loops).
"""

import glob
import os
import re
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from src.codegen import CodeGenerator  # noqa: E402
from src.graph import OnnxGraph  # noqa: E402
from src.nodes import ACT_NAMES, OP_NAMES, SchedulerError  # noqa: E402
from src.perf_calls import (FIELDS, KernelCall, bitstream_id_of_bin,  # noqa: E402
                            local_bitstream_id, merge)

# C helper -> (kernel, leading pointer / offset arguments, register order of
# the remaining arguments)
_FN = {
    "run_op":        ("VectorOPKernel", 3, ("size", "op", "outer", "a_inc", "b_inc")),
    "run_op_act":    ("VectorOPKernel", 3, ("size", "op", "outer", "a_inc", "b_inc", "act")),
    "run_matmul":    ("MatmulKernel", 3, FIELDS["MatmulKernel"]),
    "run_matmul_at": ("MatmulKernel", 6, FIELDS["MatmulKernel"]),
    "run_conv":      ("ConvKernel", 4, FIELDS["ConvKernel"]),
    "run_conv_at":   ("ConvKernel", 6, FIELDS["ConvKernel"]),
    "run_pool":      ("PoolKernel", 2, FIELDS["PoolKernel"]),
}
_CALL = re.compile(r"\b(" + "|".join(sorted(_FN, key=len, reverse=True)) + r")\(")
_LOOP = re.compile(r"for \(unsigned _i = 0u; _i < (\d+)u; _i\+\+\)")
_DEFINE = re.compile(r"^#define\s+(\w+)(\([^)]*\))?\s+(.+?)\s*(?:/\*.*)?$")


def _args(text, start):
    """The top-level comma-separated arguments of the call opened at start."""
    depth, cur, out, i = 1, "", [], start
    while depth:
        ch = text[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if not depth:
                break
        if ch == "," and depth == 1:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
        i += 1
    out.append(cur.strip())
    return out


class _Eval:
    """Evaluates the numeric C expressions of the generated calls."""

    def __init__(self, header):
        self.macros = {}
        for line in header.splitlines():
            m = _DEFINE.match(line.strip())
            if m and not m.group(2):
                self.macros[m.group(1)] = m.group(3)
        self.names = {v: k for k, v in OP_NAMES.items()}
        self.names.update({v: k for k, v in ACT_NAMES.items()})

    def __call__(self, expr, env=None):
        py = re.sub(r"\b(\d+)u\b", r"\1", expr).replace("/", "//")
        py = re.sub(r"\(unsigned\)", "", py)

        def align_up(n):
            e = self("INFERENCE_ALIGN_ELEMS")
            return (n + e - 1) & ~(e - 1)
        scope = {"INFERENCE_ALIGN_UP": align_up, **(env or {})}
        for name in set(re.findall(r"\b[A-Za-z_]\w*\b", py)):
            if name in scope:
                continue
            if name in self.names:
                scope[name] = self.names[name]
            elif name in self.macros:
                scope[name] = self(self.macros[name], env)
            elif name == "INFERENCE_BYTES_PER_ELEM":
                scope[name] = 2
        return int(eval(py, {"__builtins__": {}}, scope))    # noqa: S307


def emitted_calls(snippet, ev, env=None):
    """KernelCalls of one node's emit_call() text."""
    loop = _LOOP.search(snippet)
    count = int(loop.group(1)) if loop else 1
    calls = []
    for m in _CALL.finditer(snippet):
        kernel, skip, order = _FN[m.group(1)]
        vals = [ev(a, env) for a in _args(snippet, m.end())[skip:]]
        regs = dict(zip(order, vals, strict=True))
        calls.append(KernelCall.of(kernel, count=count, **regs))
    return merge(calls)


def kernel_nodes(g):
    return [sn for sn in g.nodes if getattr(type(sn), "kernel_name", "")
            and hasattr(sn, "kernel_calls")]


class TestKernelCall(unittest.TestCase):
    def test_key_roundtrip(self):
        c = KernelCall.of("ConvKernel", batch=1, in_ch=128, in_h=96, in_w=48, out_ch=512,
                          out_h=96, out_w=8, kh=1, kw=6, stride_h=1, stride_w=6,
                          dilation_h=1, dilation_w=1)
        self.assertEqual(c.key(), "ConvKernel:1,128,96,48,512,96,8,1,6,1,6,1,1,0,0,0,0")
        self.assertEqual(KernelCall.from_key(c.key()), c)
        self.assertEqual(c.fields["out_ch"], 512)

    def test_validation(self):
        with self.assertRaises(ValueError):
            KernelCall("NoKernel", (1,))
        with self.assertRaises(ValueError):
            KernelCall("MatmulKernel", (1, 2))
        with self.assertRaises(ValueError):
            KernelCall.of("MatmulKernel", bogus=1)
        with self.assertRaises(ValueError):
            KernelCall.of("MatmulKernel", count=0)

    def test_merge(self):
        a = KernelCall.of("VectorOPKernel", op=0, size=64)
        b = KernelCall.of("VectorOPKernel", op=1, size=64)
        self.assertEqual(merge([a, b, a]), [KernelCall(a.kernel, a.regs, 2), b])

    def test_bitstream_id(self):
        self.assertEqual(bitstream_id_of_bin(b"abc"), "ba7816bf8f01")
        self.assertIsNone(local_bitstream_id("/nonexistent/bitstream_config.json"))


class TestEmittedCalls(unittest.TestCase):
    """Every kernel node of every test model, under all three MatMul
    lowerings."""

    def _check(self, g, cg, label):
        ev = _Eval(cg.generate_header())
        n = 0
        for sn in kernel_nodes(g):
            got = merge(sn.kernel_calls(cg._layouts))
            env = {"_keys": sn.C} if hasattr(sn, "static") and not sn.static else None
            want = emitted_calls(sn.emit_call(cg._layouts), ev, env)
            self.assertEqual(got, want, f"{label}: node {sn.index} {type(sn).__name__}")
            n += 1
        return n

    def test_model_zoo(self):
        models = sorted(glob.glob(os.path.join(HERE, "models", "*.onnx")))
        if not models:
            self.skipTest("test models not generated")
        checked = 0
        for path in models:
            for mode in ("auto", "off", "always"):
                try:
                    g = OnnxGraph(path, fuse_act=True, s2d_stem=True, matmul_on_conv=mode)
                    cg = CodeGenerator(g, model_path=path)
                except (SchedulerError, ValueError, NotImplementedError):
                    continue
                with self.subTest(model=os.path.basename(path), mode=mode):
                    checked += self._check(g, cg, f"{os.path.basename(path)} {mode}")
        self.assertGreater(checked, 500)

    def test_row_split_and_vision(self):
        from test_matmul_on_conv import _matmul_model
        with tempfile.TemporaryDirectory() as td:
            p = _matmul_model(os.path.join(td, "rows.onnx"), [1024, 64], [64, 128], seed=12)
            g = OnnxGraph(p, fuse_act=True, s2d_stem=True)
            cg = CodeGenerator(g, model_path=p)
            (sn,) = kernel_nodes(g)
            self.assertEqual(sn.kernel_calls(cg._layouts)[0].count, 2)
            self._check(g, cg, "rows")
        try:
            from test_vit import Tiny
        except ImportError as e:                              # pragma: no cover
            self.skipTest(f"tiny ViT unavailable: {e}")
        fe = Tiny.get()[3]
        g = OnnxGraph(fe.entry(), fuse_act=True, s2d_stem=True)
        cg = CodeGenerator(g, model_path="vit_tiny.onnx")
        self.assertGreater(self._check(g, cg, "vit_tiny"), 20)

    def test_runtime_keys(self):
        path = os.path.join(HERE, "models", "llama_tiny_prefill8.onnx")
        if not os.path.exists(path):
            self.skipTest("llama tiny models not generated")
        g = OnnxGraph(path, fuse_act=True, s2d_stem=True)
        cg = CodeGenerator(g, model_path=path)
        attn = [sn for sn in kernel_nodes(g) if hasattr(sn, "static") and not sn.static]
        self.assertTrue(attn)
        ev = _Eval(cg.generate_header())
        for sn in attn:
            for keys in (sn.Q, sn.C):
                got = merge(sn.kernel_calls(cg._layouts, keys=keys))
                want = emitted_calls(sn.emit_call(cg._layouts), ev, {"_keys": keys})
                self.assertEqual(got, want, f"node {sn.index} keys {keys}")


if __name__ == "__main__":
    unittest.main()
