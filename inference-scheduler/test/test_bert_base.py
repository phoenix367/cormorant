"""BERT-base (bertsquad-12) gates on the real 435 MB model.

The model is $BERT_SQUAD_MODEL, else
demo/bert_squad/assets/models/bertsquad-12-simplified.onnx (or an older
checkout's inference-scheduler/bertsquad-12-simplified.onnx); vocab.txt and
dev-v1.1.json come from $BERT_SQUAD_ASSETS, else demo/bert_squad/assets.
Missing files are downloaded on the first run by
demo/bert_squad/scripts/fetch_assets.py (the model from Google Drive, 435 MB,
md5-checked).  BERT_SQUAD_DOWNLOAD=0 skips the class instead of downloading;
a failed download skips it with the error as the reason.  ~40 s with 4+
cores, ~6 GB RAM:

  .venv/bin/python -m pytest test/test_bert_base.py -v

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
from src.nodes import ACT_GELU_TANH, MatmulConvNode, MatmulNode, ScheduledNode
from src.smx_nodes import SoftmaxVopNode, enabled as softmax_unit

_STUDY = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "demo", "bert_squad", "scripts")
if _STUDY not in sys.path:
    sys.path.append(_STUDY)
import fetch_assets                                   # noqa: E402  (demo/bert_squad/scripts)

MODEL = ASSETS = ""                                   # set by setUpClass


def _assets():
    """(model, assets dir), downloading what is missing; SkipTest when that
    is switched off or fails."""
    model, assets = fetch_assets.default_model_path(), fetch_assets.default_assets_dir()
    need = [model] + [assets / fetch_assets.ASSETS[n].rel for n in ("vocab", "squad_dev")]
    if all(p.exists() for p in need):
        return str(model), str(assets)
    if os.environ.get("BERT_SQUAD_DOWNLOAD", "1") == "0":
        raise unittest.SkipTest("bertsquad-12 assets missing and BERT_SQUAD_DOWNLOAD=0: "
                                + ", ".join(str(p) for p in need if not p.exists()))
    try:
        model, assets = fetch_assets.ensure_all(model, assets)
    except fetch_assets.FetchError as e:
        raise unittest.SkipTest(f"bertsquad-12 assets could not be downloaded: {e}") from e
    return str(model), str(assets)


class TestBertBase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        global MODEL, ASSETS
        MODEL, ASSETS = _assets()
        cls.g = OnnxGraph(MODEL, fuse_act=True, s2d_stem=True)
        cls.cg = CodeGenerator(cls.g, model_path=MODEL)

    def test_partition(self):
        k = collections.Counter(type(sn) for sn in self.g.nodes)
        self.assertEqual(self.g.fusion_counts,
                         {"layernorm": 25, "gelu": 12, "silu": 0, "const_bcast": 15})
        # the 12 GELUs run on VectorOP's activation unit, fused into the FFN
        # bias Adds (bit-identical to the host op: test_activations.py); the 12
        # softmaxes on its softmax unit where the platform has it
        smx = 12 if softmax_unit() else 0
        self.assertEqual((k[LayerNormNode], k[GeluNode], k[SoftmaxNode], k[SoftmaxVopNode],
                          k[TransposeNode]), (25, 0, 12 - smx, smx, 49))
        self.assertEqual(sum(1 for sn in self.g.nodes if isinstance(sn, ScheduledNode)
                             and sn.act == ACT_GELU_TANH), 12)
        self.assertEqual((k[MatmulNode] + k[MatmulConvNode], k[ScheduledNode]), (98, 126))
        mms = [sn for sn in self.g.nodes if isinstance(sn, (MatmulNode, MatmulConvNode))]
        att = [sn for sn in mms if sn.batch == 12]
        self.assertEqual(len(att), 24)
        self.assertLessEqual(max(sn.k for sn in mms), 3072)
        # BERT_PLAN 2A: the 72 encoder linears and the 24 attention MatMuls run on
        # ConvKernel (one call per head; the RTL ConvKernel, CONV_RTL_PLAN: a
        # per-head P·V call is 0.21 ms on the board, the batched MatmulKernel
        # call 4.45 ms — on the HLS ConvKernel's bitstreams, 0.61 ms per head,
        # the 12 P·V ran on MatmulKernel), and the K = 2 token-type MatMul and
        # the M = 2 span head stay on MatmulKernel.
        self.assertEqual(k[MatmulConvNode], 96)
        self.assertEqual(sorted((sn.k, sn.m, sn.batch) for sn in mms if isinstance(sn, MatmulNode)),
                         [(2, 768, 1), (768, 2, 1)])
        conv_att = [sn for sn in att if isinstance(sn, MatmulConvNode)]
        self.assertEqual(len(conv_att), 24)
        self.assertTrue(all(sn.kw == 1 and sn.calls == 12 for sn in conv_att))
        self.assertEqual(sorted({sn.m for sn in conv_att}), [64, 256])
        self.assertEqual(self.g.matmul_conv_stats["conv_calls"], 72 + 24 * 12)

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

    def test_bit_exact_vs_study_on_squad(self):
        import bert_study as bs
        import bert_sched_check as chk
        bs.HERE = ASSETS
        _, pick = bs.pick_examples(1)
        bert = bs.Bert(MODEL, bs.POLS["sched+vsmx" if softmax_unit() else "sched"])
        res, a, b, (n_cmp, n_bad) = chk.compare(self.cg, bert, bs.feeds_of(pick[0][1]))
        self.assertEqual(n_bad, 0)
        self.assertGreater(n_cmp, 300)
        for o, (n_diff, _m) in res.items():
            self.assertEqual(n_diff, 0, o)
            np.testing.assert_array_equal(a[o], b[o])


if __name__ == "__main__":
    unittest.main()
