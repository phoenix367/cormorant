"""The generator's memory savings change no value (doc/plans/CHAT_PLAN.md §25):
the forward pass's lean mode, OnnxGraph's weightless validation / shape
inference, the lazy entry models and checkpoints, and the constant arrays
shared between the entries of one model.  On the tiny random Llama
(gen_llama_models.py)."""

import json
import os
import struct
import tempfile
import unittest

import numpy as np
import onnx

import gen_llama_models as G
import src.graph as graph_mod
from src.codegen import CodeGenerator
from src.graph import OnnxGraph
from src.llama import LazySafetensors, load_safetensors
from src.llm_entries import EntryModels, entry_graphs

_FE = None


def fe():
    global _FE
    if _FE is None:
        _FE = G.tiny(0)[3]
    return _FE


def _code(model) -> str:
    cg = CodeGenerator(OnnxGraph(model, fuse_act=True, s2d_stem=True), model_path="m.onnx")
    text = cg.generate_header() + cg.generate_source()
    return "\n".join(line for line in text.splitlines() if "Generated at" not in line)


class TestLeanForwardPass(unittest.TestCase):
    def test_same_outputs_and_states(self):
        cg = CodeGenerator(OnnxGraph(fe().entry("prefill", 16, with_head=True), fuse_act=True,
                                     s2d_stem=True), model_path="p.onnx")
        feeds = cg._build_ramp_inputs()
        s_full, s_lean = cg.initial_states(), cg.initial_states()
        full = cg._forward_pass(dict(feeds), states=s_full)
        outs = [t.onnx_name for t in cg._graph.output_tensors]
        lean = cg._forward_pass(dict(feeds), states=s_lean, keep=outs)
        self.assertEqual(set(lean), set(outs) | set(s_lean))
        for n in outs:
            np.testing.assert_array_equal(lean[n], full[n])
        for n in s_full:
            np.testing.assert_array_equal(s_lean[n], s_full[n])


class TestWeightlessInference(unittest.TestCase):
    def test_caller_model_untouched(self):
        m = fe().entry("decode")
        before = m.SerializeToString()
        OnnxGraph(m, fuse_act=True, s2d_stem=True)
        self.assertEqual(m.SerializeToString(), before)

    def test_same_code_as_full_inference(self):
        m = fe().entry("prefill", 16, with_head=True)
        big = [i for i in m.graph.initializer
               if int(np.prod(i.dims)) > graph_mod._LIGHT_INIT_ELEMS]
        self.assertTrue(big)                           # the placeholders are exercised
        lean = _code(m)
        old = graph_mod._LIGHT_INIT_ELEMS
        graph_mod._LIGHT_INIT_ELEMS = 1 << 60          # every initializer in the checked model
        try:
            full = _code(m)
        finally:
            graph_mod._LIGHT_INIT_ELEMS = old
        self.assertEqual(lean, full)


class TestLazySafetensors(unittest.TestCase):
    def test_equals_eager(self):
        rng = np.random.default_rng(1)
        a = rng.normal(size=(3, 5)).astype(np.float32)
        tensors = {
            "bf": ("BF16", (a.view(np.uint32) >> 16).astype(np.uint16).tobytes(), [3, 5]),
            "h": ("F16", a.astype(np.float16).tobytes(), [3, 5]),
            "f": ("F32", a.tobytes(), [15]),
        }
        hdr, blob = {"__metadata__": {"format": "pt"}}, b""
        for name, (dt, raw, shape) in tensors.items():
            hdr[name] = {"dtype": dt, "shape": shape, "data_offsets": [len(blob), len(blob) + len(raw)]}
            blob += raw
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.safetensors")
            h = json.dumps(hdr).encode()
            with open(p, "wb") as f:
                f.write(struct.pack("<Q", len(h)) + h + blob)
            eager, lazy = load_safetensors(p), LazySafetensors(p)
            self.assertEqual(list(eager), list(lazy))
            for k in eager:
                self.assertEqual((lazy[k].dtype, lazy[k].shape), (eager[k].dtype, eager[k].shape))
                np.testing.assert_array_equal(lazy[k].view(np.uint32), eager[k].view(np.uint32))
            sub = lazy.select({"x": "bf"})
            self.assertEqual(list(sub), ["x"])
            self.assertNotIn("bf", sub)
            np.testing.assert_array_equal(sub["x"], eager["bf"])


class TestEntryModels(unittest.TestCase):
    def test_built_when_taken(self):
        calls = []
        em = EntryModels()
        em.add("a", lambda: calls.append("a") or onnx.ModelProto())
        self.assertEqual((list(em), calls), (["a"], []))
        self.assertIsInstance(em.pop("a"), onnx.ModelProto)
        self.assertEqual(calls, ["a"])
        self.assertNotIn("a", em)

    def test_entries_share_constant_arrays(self):
        em = EntryModels()
        em.add("decode", lambda: fe().entry("decode"))
        em.add("prefill_16", lambda: fe().entry("prefill", 16))
        em.add("head", lambda: fe().entry("head"))
        gs = dict(entry_graphs(em))
        self.assertEqual(len(em), 0)
        d, p = gs["decode"], gs["prefill_16"]
        n_shared = 0
        for name, t in d._tensors.items():
            u = p._tensors.get(name)
            if u is None:
                continue
            for attr in ("data", "wexp", "init_data"):
                a, b = getattr(t, attr, None), getattr(u, attr, None)
                if isinstance(a, np.ndarray) and a.nbytes >= 4096:
                    self.assertIs(a, b, f"{name}.{attr}")
                    n_shared += 1
        self.assertGreater(n_shared, 0)


if __name__ == "__main__":
    unittest.main()
