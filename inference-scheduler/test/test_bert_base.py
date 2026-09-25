"""BERT-base (bertsquad-12) gates — OPTIONAL, needs the 435 MB model.

Skipped unless BERT_SQUAD_MODEL points to bertsquad-12-simplified.onnx
(e.g. inference-scheduler/bertsquad-12-simplified.onnx in the main
checkout).  The SQuAD comparison also needs BERT_SQUAD_ASSETS (the directory
with vocab.txt and dev-v1.1.json).  ~2 min, ~6 GB RAM:

  BERT_SQUAD_MODEL=... BERT_SQUAD_ASSETS=... .venv/bin/python -m pytest test/test_bert_base.py -v

Gate (a): the project generates and its C compiles; gate (b): the
scheduler simulation equals the study's independent emulation of the same
partition bit for bit (demo/bert_squad/scripts/bert_sched_check.py runs it on
more examples); plus the generated test_inference.c passing on the host
against the software kernel models.
"""

import collections
import os
import sys
import tempfile
import unittest

import numpy as np

import host_emu
from src.codegen import CodeGenerator
from src.graph import OnnxGraph
from src.host_nodes import GeluNode, LayerNormNode, SoftmaxNode, TransposeNode
from src.nodes import MatmulNode, ScheduledNode

MODEL = os.environ.get("BERT_SQUAD_MODEL", "")
ASSETS = os.environ.get("BERT_SQUAD_ASSETS", "")
_STUDY = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "demo", "bert_squad", "scripts")


@unittest.skipUnless(MODEL and os.path.isfile(MODEL), "set BERT_SQUAD_MODEL to run")
class TestBertBase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.g = OnnxGraph(MODEL, fuse_act=True, s2d_stem=True)
        cls.cg = CodeGenerator(cls.g, model_path=MODEL)

    def test_partition(self):
        k = collections.Counter(type(sn) for sn in self.g.nodes)
        self.assertEqual(self.g.fusion_counts, {"layernorm": 25, "gelu": 12, "const_bcast": 15})
        self.assertEqual((k[LayerNormNode], k[GeluNode], k[SoftmaxNode], k[TransposeNode]),
                         (25, 12, 12, 49))
        self.assertEqual((k[MatmulNode], k[ScheduledNode]), (98, 126))
        att = [sn for sn in self.g.nodes if isinstance(sn, MatmulNode) and sn.batch == 12]
        self.assertEqual(len(att), 24)
        self.assertLessEqual(max(sn.k for sn in self.g.nodes if isinstance(sn, MatmulNode)), 3072)

    def test_generated_c_compiles(self):
        with tempfile.TemporaryDirectory() as td:
            from test_s2d_stem import _host_compile
            rc, log = _host_compile(self.cg, td)
        self.assertEqual(rc, 0, log)
        h = self.cg.generate_header()
        self.assertIn("inference_buf_t *input_ids_0,", h)

    @unittest.skipUnless(host_emu.which_cc(), "C compiler not available on host")
    def test_host_emulated_run(self):
        with tempfile.TemporaryDirectory() as td:
            rc, out = host_emu.build_and_run(self.cg, td, timeout=1800)
        self.assertEqual(rc, 0, out[-3000:])
        self.assertIn("test_inference PASSED", out)

    @unittest.skipUnless(ASSETS and os.path.isdir(ASSETS), "set BERT_SQUAD_ASSETS to run")
    def test_bit_exact_vs_study_on_squad(self):
        if _STUDY not in sys.path:
            sys.path.insert(0, _STUDY)
        import bert_study as bs
        import bert_sched_check as chk
        bs.HERE = ASSETS
        _, pick = bs.pick_examples(1)
        bert = bs.Bert(MODEL, bs.POLS["sched"])
        res, a, b, (n_cmp, n_bad) = chk.compare(self.cg, bert, bs.feeds_of(pick[0][1]))
        self.assertEqual(n_bad, 0)
        self.assertGreater(n_cmp, 300)
        for o, (n_diff, _m) in res.items():
            self.assertEqual(n_diff, 0, o)
            np.testing.assert_array_equal(a[o], b[o])


if __name__ == "__main__":
    unittest.main()
