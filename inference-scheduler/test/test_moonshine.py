"""Moonshine speech to text (doc/plans/ASR_PLAN.md §3): the encode_<T> and decode
entries of src/moonshine.py through the scheduler (the stem host ops and the
row-ops nodes of src/asr_nodes.py, the encoder's non-causal softmax on
VectorOPKernel's unit) against the specification demo/asr/scripts/moonshine_q.py
(QM with the same formats), bit for bit: the encoder output over a bucket
longer than the audio (the padded frames masked), the decoder's logits step
by step over the cross caches the encode entry wrote; the generated C against
the simulation (host_emu).  A small random Moonshine — width 80, 4 heads of 20
(padded to 32: the head-dim padding and the partial interleaved RoPE's
permutation), 2 + 2 layers, a 128-token vocabulary — with formats the
specification calibrates on synthetic audio (the checkpoints are untracked
assets); the real moonshine-tiny on one LibriSpeech utterance when its assets
are there."""

import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

import host_emu
from src.codegen import CodeGenerator
from src.graph import OnnxGraph
from src.moonshine import (MoonshineConfig, MoonshineFormats, MoonshineFrontend, head_map, rope_half_tables,
                           samples, stem_frames)
from src.rowops_nodes import enabled as rowops_unit
from src.smx_nodes import enabled as softmax_unit

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SCRIPTS = os.path.join(_REPO, "demo", "asr", "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
import moonshine_np as mn  # noqa: E402
import moonshine_q as mq  # noqa: E402

CFG = {"hidden_size": 80, "intermediate_size": 64, "encoder_num_hidden_layers": 2,
       "decoder_num_hidden_layers": 2, "encoder_num_attention_heads": 4, "decoder_num_attention_heads": 4,
       "encoder_num_key_value_heads": 4, "decoder_num_key_value_heads": 4, "vocab_size": 128,
       "partial_rotary_factor": 0.9, "rope_theta": 10000.0, "decoder_start_token_id": 1, "eos_token_id": 2,
       "max_position_embeddings": 64, "bos_token_id": 1, "pad_token_id": 2}
POLICY = {"p_bits": 12, "in_cap": 8,
          "caps": {c: 7 for c in ("x", "x2", "xf", "gn", "y", "y2", "y3", "yf")} | {("a", "d"): 7, "g2": 6}}


def random_weights(seed=0):
    """The checkpoint's tensors (Hugging Face names) at CFG's shapes, scaled by fan-in."""
    rng = np.random.default_rng(seed)
    d, ff, V = CFG["hidden_size"], CFG["intermediate_size"], CFG["vocab_size"]
    W = {}

    def w(name, *shape, fan=None, scale=1.0):
        fan = fan or shape[-1]
        W[name] = (rng.standard_normal(shape) * scale / np.sqrt(fan)).astype(np.float32)

    w("model.encoder.conv1.weight", d, 1, 127, scale=3.0)
    W["model.encoder.groupnorm.weight"] = (1.0 + 0.2 * rng.standard_normal(d)).astype(np.float32)
    W["model.encoder.groupnorm.bias"] = (0.1 * rng.standard_normal(d)).astype(np.float32)
    w("model.encoder.conv2.weight", 2 * d, d, 7, fan=7 * d)
    W["model.encoder.conv2.bias"] = (0.1 * rng.standard_normal(2 * d)).astype(np.float32)
    w("model.encoder.conv3.weight", d, 2 * d, 3, fan=6 * d)
    W["model.encoder.conv3.bias"] = (0.1 * rng.standard_normal(d)).astype(np.float32)
    for part, L in (("encoder", CFG["encoder_num_hidden_layers"]), ("decoder", CFG["decoder_num_hidden_layers"])):
        for li in range(L):
            p = f"model.{part}.layers.{li}."
            for k in ("q", "k", "v", "o"):
                w(p + f"self_attn.{k}_proj.weight", d, d)
            norms = ("input_layernorm", "post_attention_layernorm") + (("final_layernorm",) if part == "decoder" else ())
            for nm in norms:
                W[p + nm + ".weight"] = (1.0 + 0.1 * rng.standard_normal(d)).astype(np.float32)
            if part == "encoder":
                w(p + "mlp.fc1.weight", ff, d)
                W[p + "mlp.fc1.bias"] = (0.05 * rng.standard_normal(ff)).astype(np.float32)
            else:
                for k in ("q", "k", "v", "o"):
                    w(p + f"encoder_attn.{k}_proj.weight", d, d)
                w(p + "mlp.fc1.weight", 2 * ff, d)
                W[p + "mlp.fc1.bias"] = (0.05 * rng.standard_normal(2 * ff)).astype(np.float32)
            w(p + "mlp.fc2.weight", d, ff)
            W[p + "mlp.fc2.bias"] = (0.05 * rng.standard_normal(d)).astype(np.float32)
    W["model.encoder.layer_norm.weight"] = (1.0 + 0.1 * rng.standard_normal(d)).astype(np.float32)
    W["model.decoder.norm.weight"] = (1.0 + 0.1 * rng.standard_normal(d)).astype(np.float32)
    w("model.decoder.embed_tokens.weight", V, d, scale=4.0)
    return W


def audio(seconds, seed):
    """Synthetic speech-like int16: a few sweeping tones with an envelope, plus noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * 16000)) / 16000.0
    x = sum(np.sin(2 * np.pi * (f0 + 300 * t) * t + rng.uniform(0, 6)) for f0 in rng.uniform(120, 900, 3))
    x = x * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)) + 0.1 * rng.standard_normal(t.size)
    return np.clip(np.rint(x / np.abs(x).max() * 12000), -32768, 32767).astype(np.int16)


_SETUP = None


def setup():
    """(cfg, W, spec, fmt json): the specification calibrated (float) on synthetic audio,
    the opt formats."""
    global _SETUP
    if _SETUP is None:
        W = random_weights()
        W64 = {k.removeprefix("model."): v.astype(np.float64) for k, v in W.items()}
        rec = mq.QM(CFG, W64, record=True)
        for s, seed in ((1.1, 11), (0.7, 12)):
            mq.transcribe(rec, CFG, audio(s, seed), forced=[5, 9, 17, 33])
        fmt = mq.make_formats(mq.QM(CFG, W64), rec.ch, POLICY, ms={k: sq / n for k, (sq, n) in rec.ms.items()})
        js = {"exponents": {f"{c}@{p}{l}": (int(v) if np.ndim(v) == 0 else [int(x) for x in np.ravel(v)])
                            for (c, p, l), v in fmt.items()}}
        spec = mq.QM(CFG, W64, pol=POLICY, fmt=mq.load_formats_dict(js))
        _SETUP = (MoonshineConfig.from_dict(CFG), W, spec, js)
    return _SETUP


def frontend():
    cfg, W, _, js = setup()
    return MoonshineFrontend(cfg, W, MoonshineFormats(js, cfg))


@unittest.skipUnless(rowops_unit() and softmax_unit(), "needs the row-ops unit and the softmax unit")
class TestMoonshineEntries(unittest.TestCase):
    T = 64

    def test_layout_helpers(self):
        cfg = setup()[0]
        self.assertEqual((cfg.hd, cfg.HDp, cfg.rot), (20, 32, 18))
        m = head_map(cfg, rope=True).reshape(cfg.H, cfg.HDp)
        order = mq.rope_perm(cfg.hd, cfg.rot)
        np.testing.assert_array_equal(m[1, :10], 20 + order[:10])
        np.testing.assert_array_equal(m[1, 16:26], 20 + order[10:])
        self.assertTrue((m[:, 10:16] < 0).all() and (m[:, 26:] < 0).all())
        c, s = rope_half_tables(cfg, 40)
        sc, ss = mq.rope_half_tables(CFG, cfg.H, np.arange(40))
        np.testing.assert_array_equal(c[:, :10], sc)
        np.testing.assert_array_equal(s[:, :10], ss)
        self.assertTrue((c[:, 10:] == 1).all() and (s[:, 10:] == 0).all())

    def _encode(self, cg, pcm, states):
        T = self.T
        x = np.zeros(samples(T))
        x[:len(pcm)] = pcm / 32768.0
        out = cg._forward_pass({f"enc{T}.pcm": x, f"enc{T}.n": np.array([len(pcm)]),
                                f"enc{T}.pos": np.array([0])}, states=states, keep=[f"enc{T}.xf_out"])
        return np.asarray(out[f"enc{T}.xf_out"]).reshape(T, CFG["hidden_size"])

    def test_simulation_matches_specification(self):
        _, _, spec, _ = setup()
        fe = frontend()
        cge = CodeGenerator(OnnxGraph(fe.encode_entry(self.T, output_xf=True)), model_path="enc.onnx")
        cgd = CodeGenerator(OnnxGraph(fe.decode_entry()), model_path="dec.onnx")
        states = cge.initial_states()
        for k, v in cgd.initial_states().items():
            states.setdefault(k, v)
        for secs, seed in ((0.9, 3), (0.45, 4)):
            pcm = audio(secs, seed)
            nf = stem_frames(len(pcm))[2]
            self.assertLess(nf, self.T)
            xf = self._encode(cge, pcm, states)
            xs = spec.encode(pcm)
            np.testing.assert_array_equal(xf[:nf], xs, err_msg=f"xf {secs} s")
            spec.start(xs)
            tok = CFG["decoder_start_token_id"]
            for i, nxt in enumerate([7, 40, 3, 99]):
                o = cgd._forward_pass({"decode.ids": np.array([tok]), "decode.pos": np.array([i]),
                                       "decode.xpos": np.array([nf - 1])}, states=states,
                                      keep=["decode.id", "decode.logits"])
                lg = spec.step(tok)
                np.testing.assert_array_equal(np.asarray(o["decode.logits"]).reshape(-1), lg,
                                              err_msg=f"logits {secs} s step {i}")
                self.assertEqual(int(np.asarray(o["decode.id"]).reshape(-1)[0]), int(np.argmax(lg)))
                tok = nxt

    def test_graph(self):
        fe = frontend()
        kinds = {}
        for sn in OnnxGraph(fe.encode_entry(self.T)).nodes + OnnxGraph(fe.decode_entry()).nodes:
            kinds[type(sn).__name__] = kinds.get(type(sn).__name__, 0) + 1
        H, L, Ld = 4, 2, 2
        self.assertEqual(kinds.get("AsrStemGeluNode"), 2)
        self.assertEqual(kinds.get("ConvNode"), 3)
        self.assertEqual(kinds.get("LlmAttnSoftmaxVopNode"), H * L)          # the encoder's, on the unit
        self.assertEqual(kinds.get("LlmAttnSoftmaxNode"), 2 * H * Ld)        # a decode step's, on the host
        self.assertEqual(kinds.get("AttnPrepRoNode"), L + Ld)
        self.assertEqual(kinds.get("AsrCrossPrepRoNode"), Ld)
        self.assertEqual(kinds.get("AsrSwiGLURoNode"), Ld)
        host = {k for k in kinds if k in ("AsrFramesNode", "AsrStemNormNode", "AsrStemGeluNode",
                                          "AsrFrameCountNode", "LlmEmbedNode", "AsrArgmaxNode",
                                          "LlmAttnSoftmaxNode", "LlmAttnMergeNode")}
        self.assertEqual(host, {"AsrFramesNode", "AsrStemNormNode", "AsrStemGeluNode", "AsrFrameCountNode",
                                "LlmEmbedNode", "AsrArgmaxNode", "LlmAttnSoftmaxNode", "LlmAttnMergeNode"})

    def test_bucket_bounds(self):
        fe = frontend()
        with self.assertRaises(ValueError):
            fe.encode_entry(96)                    # not a multiple of 64
        with self.assertRaises(ValueError):
            fe.encode_entry(1088)                  # beyond the softmax unit's 1024 keys

    @unittest.skipUnless(shutil.which("cc"), "needs cc")
    def test_generated_c_matches_simulation(self):
        fe = frontend()
        for m in (fe.encode_entry(self.T, output_xf=True), fe.decode_entry()):
            cg = CodeGenerator(OnnxGraph(m), model_path="m.onnx")
            with tempfile.TemporaryDirectory() as td:
                rc, out = host_emu.build_and_run(cg, os.path.join(td, "p"), timeout=1800)
            self.assertEqual(rc, 0, out[-3000:])
            self.assertIn("PASSED", out)


_TINY = os.path.join(_REPO, "demo", "asr", "assets", "moonshine-tiny")
_TINY_FMT = os.path.join(_REPO, "demo", "asr", "moonshine-tiny_opt_formats.json")


@unittest.skipUnless(os.path.exists(os.path.join(_TINY, "model.safetensors")) and os.path.exists(_TINY_FMT)
                     and rowops_unit() and softmax_unit(), "needs the moonshine-tiny assets (asr_study.py fetch)")
class TestMoonshineTiny(unittest.TestCase):
    def test_encoder_and_first_steps_match_specification(self):
        from src.llama import load_safetensors
        cfgd, W64 = mn.load(_TINY)
        spec = mq.QM(cfgd, W64, pol=POLICY, fmt=mq.load_formats(_TINY_FMT))
        cfg = MoonshineConfig.from_dict(cfgd)
        fe = MoonshineFrontend(cfg, load_safetensors(os.path.join(_TINY, "model.safetensors")),
                               MoonshineFormats.from_file(_TINY_FMT, cfg))
        T = 128
        pcm = audio(1.6, 21)
        nf = stem_frames(len(pcm))[2]
        cge = CodeGenerator(OnnxGraph(fe.encode_entry(T, output_xf=True)), model_path="enc.onnx")
        cgd = CodeGenerator(OnnxGraph(fe.decode_entry()), model_path="dec.onnx")
        states = cge.initial_states()
        for k, v in cgd.initial_states().items():
            states.setdefault(k, v)
        x = np.zeros(samples(T))
        x[:len(pcm)] = pcm / 32768.0
        out = cge._forward_pass({f"enc{T}.pcm": x, f"enc{T}.n": np.array([len(pcm)]), f"enc{T}.pos": np.array([0])},
                                states=states, keep=[f"enc{T}.xf_out"])
        xs = spec.encode(pcm)
        np.testing.assert_array_equal(np.asarray(out[f"enc{T}.xf_out"]).reshape(T, cfg.d)[:nf], xs)
        spec.start(xs)
        tok = cfg.bos
        for i in range(2):
            o = cgd._forward_pass({"decode.ids": np.array([tok]), "decode.pos": np.array([i]),
                                   "decode.xpos": np.array([nf - 1])}, states=states, keep=["decode.logits"])
            lg = spec.step(tok)
            np.testing.assert_array_equal(np.asarray(o["decode.logits"]).reshape(-1), lg)
            tok = int(np.argmax(lg))


if __name__ == "__main__":
    unittest.main()
