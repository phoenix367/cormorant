# Copyright 2026 Ivan Gubochkin
# SPDX-License-Identifier: Apache-2.0

"""The guards of an unseen checkpoint (doc/plans/CHAT_PLAN.md §28): the config
parser src/llm_arch.py refuses what the Llama block does not implement, the
safetensors reader src/checkpoint.py reads sharded checkpoints and reports the
tensors nobody read, the formats refuse a missing exponent, the LM head
follows tie_word_embeddings."""

import json
import os
import struct
import tempfile
import unittest

import numpy as np

import gen_llama_models as G
from src.checkpoint import (LazySafetensors, Tracked, check_consumed, checkpoint_files,
                            load_safetensors)
from src.llama import Formats
from src.llm_arch import ArchError, LlamaConfig, problems

# config.json of SmolLM2-135M-Instruct (the keys and values the guard sees)
SMOLLM2 = {
    "architectures": ["LlamaForCausalLM"], "attention_bias": False, "attention_dropout": 0.0,
    "bos_token_id": 1, "eos_token_id": 2, "hidden_act": "silu", "hidden_size": 576,
    "initializer_range": 0.041666666666666664, "intermediate_size": 1536, "is_llama_config": True,
    "max_position_embeddings": 8192, "mlp_bias": False, "model_type": "llama",
    "num_attention_heads": 9, "num_hidden_layers": 30, "num_key_value_heads": 3, "pad_token_id": 2,
    "pretraining_tp": 1, "rms_norm_eps": 1e-05, "rope_interleaved": False, "rope_scaling": None,
    "rope_theta": 100000, "tie_word_embeddings": True, "torch_dtype": "bfloat16",
    "transformers.js_config": {"kv_cache_dtype": {"q4f16": "float16"}},
    "transformers_version": "4.42.3", "use_cache": True, "vocab_size": 49152}


def write_safetensors(path, tensors, dtype="F32"):
    """A safetensors file of {name: float32 array} at dtype F32 / F16 / BF16."""
    hdr, blobs, off = {}, [], 0
    for name, a in tensors.items():
        a = np.ascontiguousarray(a, np.float32)
        raw = {"F32": a.view(np.uint8), "F16": a.astype(np.float16).view(np.uint8),
               "BF16": (a.view(np.uint32) >> 16).astype(np.uint16).view(np.uint8)}[dtype].tobytes()
        hdr[name] = {"dtype": dtype, "shape": list(a.shape), "data_offsets": [off, off + len(raw)]}
        blobs.append(raw)
        off += len(raw)
    h = json.dumps(hdr).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h + b"".join(blobs))


class TestConfigGuard(unittest.TestCase):

    def test_shipped_configs_pass(self):
        self.assertEqual(problems(SMOLLM2), [])
        self.assertEqual(problems(G.TINY), [])
        c = LlamaConfig.from_dict(SMOLLM2)
        self.assertEqual((c.L, c.D, c.H, c.KV, c.HD, c.FF, c.V), (30, 576, 9, 3, 64, 1536, 49152))
        self.assertEqual((c.theta, c.tied, c.max_pos), (100000.0, True, 8192))

    def test_unimplemented_math_refused(self):
        for key, value in (("attention_bias", True), ("mlp_bias", True), ("hidden_act", "gelu"),
                           ("rope_scaling", {"rope_type": "su", "factor": 32.0}),
                           ("rope_scaling", {"rope_type": "llama3", "factor": 32.0}),   # parameters missing
                           ("partial_rotary_factor", 0.3), ("sliding_window", 512),
                           ("rope_interleaved", True), ("final_logit_softcapping", 30.0)):
            with self.subTest(key=key):
                p = problems({**SMOLLM2, key: value})
                self.assertEqual(len(p), 1, p)
                self.assertIn(key, p[0])
                with self.assertRaises(ArchError):
                    LlamaConfig.from_dict({**SMOLLM2, key: value})

    def test_other_block_and_unknown_keys_refused(self):
        p = problems({**SMOLLM2, "model_type": "qwen2"})
        self.assertTrue(any("qwen2" in x for x in p), p)
        p = problems({**SMOLLM2, "query_pre_attn_scalar": 256})           # Gemma's
        self.assertTrue(any("query_pre_attn_scalar" in x and "unknown" in x for x in p), p)
        c = dict(SMOLLM2)
        del c["hidden_size"]
        self.assertTrue(any("hidden_size" in x and "missing" in x for x in problems(c)))
        # every problem at once, not the first only
        self.assertEqual(len(problems({**SMOLLM2, "mlp_bias": True, "foo": 1})), 2)

    def test_foldable_extras(self):
        """What src/llm_weights.py / src/rope.py fold into the plain block."""
        c = LlamaConfig.from_dict(SMOLLM2)
        self.assertTrue(c.plain)
        llama3 = {"rope_type": "llama3", "factor": 32.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
                  "original_max_position_embeddings": 8192}
        c = LlamaConfig.from_dict({**SMOLLM2, "rope_scaling": llama3})
        self.assertEqual(c.rope_scaling["rope_type"], "llama3")
        self.assertTrue(c.plain)                                   # tables only
        # transformers 5's format
        c = LlamaConfig.from_dict({**{k: v for k, v in SMOLLM2.items() if k not in ("rope_theta", "rope_scaling")},
                                   "rope_parameters": {"rope_type": "default", "rope_theta": 500000.0}})
        self.assertEqual((c.theta, c.rope_scaling), (500000.0, None))
        c = LlamaConfig.from_dict({**SMOLLM2, "model_type": "granite", "embedding_multiplier": 12.0,
                                   "residual_multiplier": 0.22, "attention_multiplier": 0.0078125,
                                   "logits_scaling": 8.0})
        self.assertEqual((c.emb_mult, c.res_mult, c.attn_mult, c.logit_div), (12.0, 0.22, 0.0078125, 8.0))
        self.assertFalse(c.plain)
        c = LlamaConfig.from_dict({**SMOLLM2, "model_type": "phi3", "partial_rotary_factor": 0.75,
                                   "sliding_window": 2047, "original_max_position_embeddings": 4096,
                                   "rope_scaling": {"type": "longrope", "short_factor": [1.0] * 24,
                                                    "long_factor": [2.0] * 24}})
        self.assertEqual((c.rot, c.window), (48, 2047))
        self.assertEqual(c.rope_scaling["original_max_position_embeddings"], 4096)
        self.assertFalse(c.plain)
        self.assertEqual(problems({**SMOLLM2, "model_type": "mistral", "sliding_window": 4096}), [])

    def test_defaults_and_unchecked(self):
        c = {k: v for k, v in SMOLLM2.items() if k not in ("num_key_value_heads", "rope_theta")}
        cfg = LlamaConfig.from_dict(c)
        self.assertEqual((cfg.KV, cfg.theta), (9, 10000.0))
        cfg = LlamaConfig.from_dict({**SMOLLM2, "attention_bias": True}, check=False)
        self.assertEqual(cfg.L, 30)


class TestCheckpoint(unittest.TestCase):

    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.dir = self.td.name
        rng = np.random.default_rng(0)
        self.T = {f"t{i}": rng.normal(0, 1, (3, 4)).astype(np.float32) for i in range(5)}

    def tearDown(self):
        self.td.cleanup()

    def test_dtypes_exact(self):
        for dtype in ("F32", "F16", "BF16"):
            with self.subTest(dtype=dtype):
                p = os.path.join(self.dir, f"{dtype}.safetensors")
                write_safetensors(p, self.T, dtype)
                W = load_safetensors(p)
                ref = {"F32": lambda a: a, "F16": lambda a: a.astype(np.float16).astype(np.float32),
                       "BF16": lambda a: (a.view(np.uint32) & np.uint32(0xFFFF0000)).view(np.float32)}[dtype]
                for n, a in self.T.items():
                    np.testing.assert_array_equal(W[n], ref(a))
                    np.testing.assert_array_equal(LazySafetensors(p)[n], W[n])

    def test_sharded_equals_single(self):
        names = sorted(self.T)
        write_safetensors(os.path.join(self.dir, "model-00001-of-00002.safetensors"),
                          {n: self.T[n] for n in names[:2]})
        write_safetensors(os.path.join(self.dir, "model-00002-of-00002.safetensors"),
                          {n: self.T[n] for n in names[2:]})
        index = {"weight_map": {n: f"model-0000{1 + (i >= 2)}-of-00002.safetensors"
                                for i, n in enumerate(names)}}
        with open(os.path.join(self.dir, "model.safetensors.index.json"), "w") as f:
            json.dump(index, f)
        self.assertEqual(len(checkpoint_files(self.dir)), 2)
        W, L = load_safetensors(self.dir), LazySafetensors(self.dir)
        self.assertEqual(sorted(W), names)
        self.assertEqual(sorted(L), names)
        for n in names:
            np.testing.assert_array_equal(W[n], self.T[n])
            np.testing.assert_array_equal(L[n], self.T[n])
        # a single model.safetensors takes precedence
        write_safetensors(os.path.join(self.dir, "model.safetensors"), {"only": self.T["t0"]})
        self.assertEqual(sorted(load_safetensors(self.dir)), ["only"])

    def test_missing_checkpoint(self):
        with self.assertRaises(FileNotFoundError):
            checkpoint_files(self.dir)

    def test_unread_tensors(self):
        p = os.path.join(self.dir, "model.safetensors")
        write_safetensors(p, {**self.T, "model.layers.0.self_attn.rotary_emb.inv_freq": self.T["t0"]})
        for W in (LazySafetensors(p), Tracked(load_safetensors(p))):
            with self.subTest(view=type(W).__name__):
                W["t0"]
                W.get("t1")
                with self.assertRaises(ValueError) as cm:
                    check_consumed(W, "test")
                self.assertIn("t2", str(cm.exception))
                self.assertNotIn("inv_freq", str(cm.exception))         # NEVER_READ
                check_consumed(W, "test", ignore=("t[234]",))
                for n in ("t2", "t3", "t4"):
                    W[n]
                check_consumed(W, "test")

    def test_select_shares_the_record(self):
        p = os.path.join(self.dir, "model.safetensors")
        write_safetensors(p, self.T)
        L = LazySafetensors(p)
        a, b = L.select({"x": "t0", "y": "t1"}), L.select({"z": "t2"})
        a["x"], a["y"], b["z"]
        self.assertEqual(L.unread(), ["t3", "t4"])


class TestFrontendGuards(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cfg, cls.W, cls.formats, cls.fe = G.tiny(0)

    def test_missing_exponent_refused(self):
        fd = dict(self.formats, exponents={k: v for k, v in self.formats["exponents"].items()
                                           if k != "q@1"})
        fmt = Formats(fd, self.fe.cfg)
        self.assertEqual(len(fmt.get("q", 0, 4)), 4)
        with self.assertRaises(KeyError) as cm:
            fmt.get("q", 1, 4)
        self.assertIn("q@1", str(cm.exception))

    def test_lm_head_follows_tie(self):
        W = dict(self.W, **{"lm_head.weight": self.W["model.embed_tokens.weight"] + 1})
        fe = G.frontend(self.cfg, W, self.formats)                        # tied: lm_head ignored
        np.testing.assert_array_equal(fe.lm_weight(), self.W["model.embed_tokens.weight"])
        cfg = dict(self.cfg, tie_word_embeddings=False)
        np.testing.assert_array_equal(G.frontend(cfg, W, self.formats).lm_weight(),
                                      W["lm_head.weight"])
        with self.assertRaises(ValueError):
            G.frontend(cfg, self.W, self.formats).lm_weight()


if __name__ == "__main__":
    unittest.main()
