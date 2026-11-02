# Copyright 2026 Ivan Gubochkin
# SPDX-License-Identifier: Apache-2.0

"""
The Llama block a checkpoint's config.json describes, and the guard that
refuses what this stack does not implement (doc/plans/CHAT_PLAN.md §28).

One parser for every reader of a Llama config — the scheduler frontend
(src/llama.py), the study that specifies it (demo/chat/scripts/llm_study.py,
which loads this file by path: the standard library only), the VLM text
model (demo/chat/scripts/vlm_project.py) and the model-study / llm-onboard
checkers.  The study and the frontend implement the same block, so a config
key either of them ignored would make both compute the same wrong model and
every bit-exact gate pass; ``problems()`` therefore lists every key that is
not read, not known to be harmless (generation defaults, bookkeeping) and
not at the plain block's value — an unknown key is refused until someone
has checked what it does.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List

# The model types whose block this stack implements.
MODEL_TYPES = ("llama",)

# Keys LlamaConfig reads.
READ = frozenset((
    "num_hidden_layers", "hidden_size", "num_attention_heads", "num_key_value_heads",
    "head_dim", "intermediate_size", "vocab_size", "rms_norm_eps", "rope_theta",
    "tie_word_embeddings", "max_position_embeddings", "model_type"))

# Keys accepted only at the plain Llama block's value (any other value changes
# the math: biases, another activation, RoPE scaling / partial / interleaved
# rotation, a sliding window, softcapping, extra norms, pruned heads).
PLAIN = {
    "hidden_act": ("silu",),
    "attention_bias": (False,),
    "mlp_bias": (False,),
    "rope_scaling": (None,),
    "rope_interleaved": (False,),
    "partial_rotary_factor": (1, 1.0),
    "sliding_window": (None,),
    "use_sliding_window": (False,),
    "attn_logit_softcapping": (None,),
    "final_logit_softcapping": (None,),
    "qk_layer_norms": (False,),
    "use_resampler": (False,),
    "pruned_heads": ({},),
    "add_cross_attention": (False,),
    "is_encoder_decoder": (False,),
    "tie_encoder_decoder": (False,),
}

# Keys that do not change the forward pass: Hugging Face's PretrainedConfig
# bookkeeping and generation defaults, training-only settings, and the
# vision side of a VLM's text config (src/vit.py reads those).
HARMLESS = frozenset((
    "_name_or_path", "_attn_implementation_autoset", "_flash_attn_2_enabled", "architectures",
    "attention_dropout", "bad_words_ids", "begin_suppress_tokens", "bos_token_id",
    "chunk_size_feed_forward", "cross_attention_hidden_size", "decoder_start_token_id",
    "diversity_penalty", "do_sample", "dtype", "early_stopping", "encoder_no_repeat_ngram_size",
    "eos_token_id", "exponential_decay_length_penalty", "finetuning_task", "forced_bos_token_id",
    "forced_eos_token_id", "id2label", "initializer_range", "is_decoder", "is_llama_config",
    "label2id", "length_penalty", "max_length", "min_length", "neftune_noise_alpha",
    "no_repeat_ngram_size", "num_beam_groups", "num_beams", "num_return_sequences",
    "output_attentions", "output_hidden_states", "output_scores", "pad_token_id",
    "perceiver_config", "pixel_shuffle_factor", "prefix", "pretraining_tp", "problem_type",
    "remove_invalid_values", "repetition_penalty", "return_dict", "return_dict_in_generate",
    "sep_token_id", "suppress_tokens", "task_specific_params", "temperature", "tf_legacy_loss",
    "tokenizer_class", "top_k", "top_p", "torch_dtype", "torchscript", "transformers.js_config",
    "transformers_version", "typical_p", "use_bfloat16", "use_cache"))


class ArchError(ValueError):
    """A config this stack does not implement (``problems()`` lists why)."""


def problems(c: dict) -> List[str]:
    """Everything in a config.json dict that this stack does not implement:
    another model type, a key at a non-plain value, an unknown key."""
    out = []
    if c.get("model_type") not in MODEL_TYPES:
        out.append(f"model_type {c.get('model_type')!r}: only {', '.join(MODEL_TYPES)} blocks are implemented")
    for k in sorted(c):
        if k in READ or k in HARMLESS:
            continue
        if k in PLAIN:
            if c[k] not in PLAIN[k]:
                out.append(f"{k} = {c[k]!r}: not implemented (only {PLAIN[k][0]!r})")
        else:
            out.append(f"{k} = {c[k]!r}: unknown config key — if it does not change the "
                       f"forward pass, add it to llm_arch.HARMLESS, else implement it")
    for k in ("num_hidden_layers", "hidden_size", "num_attention_heads", "intermediate_size",
              "vocab_size", "rms_norm_eps"):
        if k not in c:
            out.append(f"{k}: missing")
    return out


@dataclass
class LlamaConfig:
    L: int                  # layers
    D: int                  # hidden size
    H: int                  # attention heads
    KV: int                 # key / value heads
    HD: int                 # head dim
    FF: int                 # MLP intermediate size
    V: int                  # vocabulary
    eps: float
    theta: float
    tied: bool = True
    max_pos: int = 0        # max_position_embeddings (0: not given)

    @classmethod
    def from_dict(cls, c: dict, check: bool = True) -> "LlamaConfig":
        """Hugging Face's defaults for the optional keys (num_key_value_heads =
        heads, head_dim = hidden / heads, rope_theta 10000, tied); ``check``
        raises ArchError on any of ``problems(c)``."""
        if check:
            p = problems(c)
            if p:
                raise ArchError("config.json describes a model this stack does not implement:\n  "
                                + "\n  ".join(p))
        H = int(c["num_attention_heads"])
        D = int(c["hidden_size"])
        return cls(L=int(c["num_hidden_layers"]), D=D, H=H,
                   KV=int(c.get("num_key_value_heads") or H),
                   HD=int(c.get("head_dim") or D // H),
                   FF=int(c["intermediate_size"]), V=int(c["vocab_size"]),
                   eps=float(c["rms_norm_eps"]), theta=float(c.get("rope_theta", 10000.0)),
                   tied=bool(c.get("tie_word_embeddings", True)),
                   max_pos=int(c.get("max_position_embeddings") or 0))

    @classmethod
    def from_file(cls, path: str, check: bool = True) -> "LlamaConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f), check)


__all__ = ("ArchError", "HARMLESS", "LlamaConfig", "MODEL_TYPES", "PLAIN", "READ", "problems")
