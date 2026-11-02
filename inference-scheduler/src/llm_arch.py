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

Model types whose extras are weight or table transforms of the plain block
(src/llm_weights.py, src/rope.py; CHAT_PLAN §29): "mistral" (a sliding
window no shorter than the longest context), "granite" (the embedding,
residual and attention multipliers, the logits divisor), "phi3" (fused
q/k/v and gate/up projections, partial rotation); for every type RoPE
scaling (linear, llama3, dynamic, yarn, longrope) and partial rotation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Optional

# The model types whose block this stack implements.
MODEL_TYPES = ("llama", "mistral", "granite", "phi3")
# The longest context a library is built for (the softmax unit's keys): a
# sliding window at least this long never cuts a key.
MAX_CONTEXT = 1024
# RoPE scalings (src/rope.py)
ROPE_TYPES = ("default", "linear", "llama3", "dynamic", "yarn", "longrope")
ROPE_NEEDS = {"linear": ("factor",), "dynamic": ("factor",),
              "llama3": ("factor", "low_freq_factor", "high_freq_factor", "original_max_position_embeddings"),
              "yarn": ("original_max_position_embeddings",),
              "longrope": ("short_factor", "long_factor", "original_max_position_embeddings")}

# Keys LlamaConfig reads.
READ = frozenset((
    "num_hidden_layers", "hidden_size", "num_attention_heads", "num_key_value_heads",
    "head_dim", "intermediate_size", "vocab_size", "rms_norm_eps", "rope_theta",
    "tie_word_embeddings", "max_position_embeddings", "model_type", "rope_scaling",
    "rope_parameters", "partial_rotary_factor", "sliding_window", "use_sliding_window",
    "original_max_position_embeddings", "embedding_multiplier", "residual_multiplier",
    "attention_multiplier", "logits_scaling"))

# Keys accepted only at the plain Llama block's value (any other value changes
# the math: biases, another activation, RoPE scaling / partial / interleaved
# rotation, a sliding window, softcapping, extra norms, pruned heads).
PLAIN = {
    "hidden_act": ("silu",),
    "attention_bias": (False,),
    "mlp_bias": (False,),
    "rope_interleaved": (False,),          # Hugging Face's Llama ignores it (rotate-half)
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
    "transformers_version", "typical_p", "use_bfloat16", "use_cache",
    # dropout (training only); custom-code pointers of a type transformers implements natively
    # (the native implementation is the reference: llm_study.py validate compares with it)
    "embd_pdrop", "resid_pdrop", "auto_map"))


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
    rp = _rope_params(c)
    kind = rp.get("rope_type") or rp.get("type") or "default"
    if kind not in ROPE_TYPES:
        out.append(f"rope_scaling {kind!r}: not implemented ({', '.join(ROPE_TYPES)})")
    else:
        miss = [k for k in ROPE_NEEDS.get(kind, ()) if rp.get(k) is None and c.get(k) is None]
        if miss:
            out.append(f"rope_scaling {kind!r}: {', '.join(miss)} missing")
    prf = rp.get("partial_rotary_factor", c.get("partial_rotary_factor", 1.0))
    if "hidden_size" in c and "num_attention_heads" in c:
        hd = int(c.get("head_dim") or c["hidden_size"] // c["num_attention_heads"])
        rot = int(hd * float(prf))
        if rot <= 0 or rot % 2 or rot > hd:
            out.append(f"partial_rotary_factor {prf}: {rot} rotated dims of {hd}")
    w = c.get("sliding_window")
    if w is not None and c.get("use_sliding_window", True) and int(w) < MAX_CONTEXT:
        out.append(f"sliding_window {w}: shorter than the context ({MAX_CONTEXT}); windowed "
                   f"attention is not implemented")
    return out


def _rope_params(c: dict) -> dict:
    """The RoPE parameters in either format: transformers 5's rope_parameters
    {rope_type, rope_theta, ...} or rope_theta + rope_scaling {rope_type | type, ...}."""
    rp = c.get("rope_parameters")
    if isinstance(rp, dict):
        return dict(rp)
    return dict(c.get("rope_scaling") or {})


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
    rot: int = 0            # rotated dims per head (0: all, partial_rotary_factor 1)
    rope_scaling: Optional[dict] = None   # rope_type + its parameters (src/rope.py), None: unscaled
    emb_mult: float = 1.0   # Granite: embedding, residual-branch and score multipliers, logits divisor
    res_mult: float = 1.0
    attn_mult: Optional[float] = None     # the score scale instead of head_dim^-0.5
    logit_div: float = 1.0
    window: Optional[int] = None          # a sliding window (>= MAX_CONTEXT: no effect)

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
        HD = int(c.get("head_dim") or D // H)
        rp = _rope_params(c)
        kind = rp.get("rope_type") or rp.get("type") or "default"
        scaling = None
        if kind != "default":
            scaling = {**rp, "rope_type": kind}
            if c.get("original_max_position_embeddings") and "original_max_position_embeddings" not in rp:
                scaling["original_max_position_embeddings"] = c["original_max_position_embeddings"]
        rot = int(HD * float(rp.get("partial_rotary_factor", c.get("partial_rotary_factor", 1.0))))
        window = c.get("sliding_window") if c.get("use_sliding_window", True) else None
        return cls(L=int(c["num_hidden_layers"]), D=D, H=H,
                   KV=int(c.get("num_key_value_heads") or H), HD=HD,
                   FF=int(c["intermediate_size"]), V=int(c["vocab_size"]),
                   eps=float(c["rms_norm_eps"]),
                   theta=float(rp.get("rope_theta", c.get("rope_theta", 10000.0))),
                   tied=bool(c.get("tie_word_embeddings", True)),
                   max_pos=int(c.get("max_position_embeddings") or 0),
                   rot=0 if rot == HD else rot, rope_scaling=scaling,
                   emb_mult=float(c.get("embedding_multiplier", 1.0)),
                   res_mult=float(c.get("residual_multiplier", 1.0)),
                   attn_mult=None if c.get("attention_multiplier") is None else float(c["attention_multiplier"]),
                   logit_div=float(c.get("logits_scaling", 1.0)),
                   window=None if window is None else int(window))

    @property
    def plain(self) -> bool:
        """No weight transform (src/llm_weights.py): the block as it is stored."""
        return (not self.rot and self.emb_mult == 1.0 and self.res_mult == 1.0
                and self.attn_mult is None and self.logit_div == 1.0
                and (self.rope_scaling or {}).get("rope_type") not in ("yarn", "longrope"))

    @classmethod
    def from_file(cls, path: str, check: bool = True) -> "LlamaConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f), check)


__all__ = ("ArchError", "HARMLESS", "LlamaConfig", "MAX_CONTEXT", "MODEL_TYPES", "PLAIN", "READ",
           "ROPE_TYPES", "problems")
