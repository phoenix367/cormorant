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
window), "granite" (the embedding, residual and attention multipliers, the
logits divisor), "phi3" (fused q/k/v and gate/up projections, partial
rotation); for every type RoPE scaling (linear, llama3, dynamic, yarn,
longrope) and partial rotation.  Types with new numerics (CHAT_PLAN §31):
"gemma3_text" (RMSNorm weights stored as w - 1, the embedding times
sqrt(hidden), RMSNorms of every q / k head and of the attention and MLP
outputs, a GELU-tanh gated MLP, sliding-window layers with their own RoPE
theta), "qwen3" (the q / k head norms).
"""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from typing import List, Optional, Tuple

# The model types whose block this stack implements.
MODEL_TYPES = ("llama", "mistral", "granite", "phi3", "gemma3_text", "qwen3")
# The longest context a library is built for (the softmax unit's keys): a
# sliding window at least this long never cuts a key.
MAX_CONTEXT = 1024
# The MLP's gate activations (config hidden_act / hidden_activation -> the block's name)
ACTS = {"silu": "silu", "gelu_pytorch_tanh": "gelu_tanh"}
LAYER_TYPES = ("full_attention", "sliding_attention")
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
    "attention_multiplier", "logits_scaling", "hidden_act"))
# Keys only some model types read (another type's Hugging Face class ignores them:
# unknown there, refused)
TYPE_READ = {
    "gemma3_text": frozenset(("query_pre_attn_scalar", "rope_local_base_freq", "layer_types",
                              "_sliding_window_pattern", "sliding_window_pattern", "hidden_activation")),
    "qwen3": frozenset(("layer_types", "max_window_layers")),
}
# The model types whose Hugging Face class applies sliding_window (the others ignore it:
# a window shorter than MAX_CONTEXT is refused there)
WINDOW_TYPES = ("mistral", "phi3", "gemma3_text", "qwen3")

# Keys accepted only at the plain Llama block's value (any other value changes
# the math: biases, another activation, RoPE scaling / partial / interleaved
# rotation, a sliding window, softcapping, extra norms, pruned heads).
PLAIN = {
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
    "use_bidirectional_attention": (False, None),
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
    "embd_pdrop", "resid_pdrop", "auto_map",
    # a marker of unsloth's re-uploads (chat-template fixes)
    "unsloth_fixed"))


class ArchError(ValueError):
    """A config this stack does not implement (``problems()`` lists why)."""


def problems(c: dict) -> List[str]:
    """Everything in a config.json dict that this stack does not implement:
    another model type, a key at a non-plain value, an unknown key."""
    out = []
    mt = c.get("model_type")
    if mt not in MODEL_TYPES:
        out.append(f"model_type {mt!r}: only {', '.join(MODEL_TYPES)} blocks are implemented")
    for k in sorted(c):
        if k in READ or k in HARMLESS or k in TYPE_READ.get(mt, ()):
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
    act = _act_key(c)
    if act not in ACTS:
        key = "hidden_activation" if mt == "gemma3_text" else "hidden_act"
        out.append(f"{key} {act!r}: not implemented ({', '.join(ACTS)})")
    for lt, rp in _rope_by_type(c).items():
        kind = rp.get("rope_type") or rp.get("type") or "default"
        what = "rope_scaling" + ("" if lt == "full_attention" else f" ({lt})")
        if kind not in ROPE_TYPES:
            out.append(f"{what} {kind!r}: not implemented ({', '.join(ROPE_TYPES)})")
        else:
            miss = [k for k in ROPE_NEEDS.get(kind, ()) if rp.get(k) is None and c.get(k) is None]
            if miss:
                out.append(f"{what} {kind!r}: {', '.join(miss)} missing")
    prf = _rope_params(c).get("partial_rotary_factor", c.get("partial_rotary_factor", 1.0))
    if "hidden_size" in c and "num_attention_heads" in c:
        hd = int(c.get("head_dim") or c["hidden_size"] // c["num_attention_heads"])
        rot = int(hd * float(prf))
        if rot <= 0 or rot % 2 or rot > hd:
            out.append(f"partial_rotary_factor {prf}: {rot} rotated dims of {hd}")
    lts = c.get("layer_types")
    if lts is not None and not isinstance(lts, (list, tuple)):
        out.append(f"layer_types {lts!r}: not a list")
    elif lts is not None:
        bad = sorted(set(map(str, lts)) - set(LAYER_TYPES))
        if bad:
            out.append(f"layer_types {bad}: not implemented ({', '.join(LAYER_TYPES)})")
        if "num_hidden_layers" in c and len(lts) != int(c["num_hidden_layers"]):
            out.append(f"layer_types: {len(lts)} entries for {c['num_hidden_layers']} layers")
    if c.get("use_sliding_window") and c.get("max_window_layers") is not None:
        # Qwen2-style: the window on layers >= max_window_layers only (not implemented)
        out.append(f"use_sliding_window with max_window_layers {c['max_window_layers']}: not implemented")
    w = c.get("sliding_window")
    if w is not None and c.get("use_sliding_window", True):
        if int(w) < 1:
            out.append(f"sliding_window {w}")
        elif mt not in WINDOW_TYPES and int(w) < MAX_CONTEXT:
            out.append(f"sliding_window {w}: model_type {mt!r} has no sliding window (only "
                       f"{', '.join(WINDOW_TYPES)})")
    return out


def _act_key(c: dict) -> str:
    """The MLP activation's config value (Gemma 3's hidden_activation, else hidden_act)."""
    if c.get("model_type") == "gemma3_text":
        return c.get("hidden_activation") or "gelu_pytorch_tanh"
    return c.get("hidden_act") or "silu"


def _rope_params(c: dict) -> dict:
    """The RoPE parameters of the full-attention layers in either format: transformers 5's
    rope_parameters {rope_type, rope_theta, ...} (per layer type: {"full_attention": {...},
    ...}) or rope_theta + rope_scaling {rope_type | type, ...}."""
    rp = c.get("rope_parameters")
    if isinstance(rp, dict):
        return dict(rp["full_attention"]) if isinstance(rp.get("full_attention"), dict) else dict(rp)
    return dict(c.get("rope_scaling") or {})


def _rope_by_type(c: dict) -> dict:
    """{layer type: its RoPE parameters}: the full layers' (``_rope_params``), and the
    sliding layers' of a model that gives them their own (rope_parameters per type,
    Gemma 3's rope_local_base_freq: unscaled)."""
    out = {"full_attention": _rope_params(c)}
    rp = c.get("rope_parameters")
    if isinstance(rp, dict) and isinstance(rp.get("sliding_attention"), dict):
        out["sliding_attention"] = dict(rp["sliding_attention"])
    elif c.get("rope_local_base_freq") is not None:
        out["sliding_attention"] = {"rope_type": "default", "rope_theta": c["rope_local_base_freq"]}
    return out


def _layer_types(c: dict, L: int) -> Optional[Tuple[str, ...]]:
    """Per layer "full" or "sliding": config layer_types, else Gemma 3's pattern (every
    pattern-th layer full), else None (every layer as sliding_window says)."""
    lts = c.get("layer_types")
    if lts is None and c.get("model_type") == "gemma3_text":
        pat = int(c.get("_sliding_window_pattern") or c.get("sliding_window_pattern") or 6)
        lts = ["sliding_attention" if (i + 1) % pat else "full_attention" for i in range(L)]
    if lts is None:
        return None
    return tuple("sliding" if t == "sliding_attention" else "full" for t in lts)


def _f32(x: float) -> float:
    """x rounded to float32 (as torch keeps a float32 constant)."""
    return struct.unpack("<f", struct.pack("<f", x))[0]


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
    emb_mult: float = 1.0   # Granite: embedding (float32), residual-branch and score multipliers, logits divisor
    res_mult: float = 1.0
    attn_mult: Optional[float] = None     # the score scale instead of head_dim^-0.5
    logit_div: float = 1.0
    window: Optional[int] = None          # a sliding window (>= the context: no effect)
    # new numerics (CHAT_PLAN §31)
    act: str = "silu"                     # the gate activation: "silu" | "gelu_tanh"
    norm_offset: bool = False             # RMSNorm weights stored as w - 1: x_hat * (1 + w) (Gemma)
    qk_norm: bool = False                 # an RMSNorm over every q / k head before RoPE
    post_norms: bool = False              # RMSNorms of the attention / MLP outputs before their adds
    layer_types: Optional[Tuple[str, ...]] = None   # per layer "full" | "sliding" (None: all as window)
    local_theta: Optional[float] = None   # the sliding layers' RoPE theta (unscaled; None: theta)

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
        window = (c.get("sliding_window") if c.get("use_sliding_window", True)
                  and c.get("model_type") in WINDOW_TYPES else None)
        L = int(c["num_hidden_layers"])
        gemma = c.get("model_type") == "gemma3_text"
        attn_mult = None if c.get("attention_multiplier") is None else float(c["attention_multiplier"])
        qpas = c.get("query_pre_attn_scalar")
        if qpas is not None and float(qpas) != HD:          # the score scale qpas^-1/2, not head_dim's
            attn_mult = float(qpas) ** -0.5
        local = _rope_by_type(c).get("sliding_attention")
        local_theta = None if local is None else float(local.get("rope_theta", c.get("rope_theta", 10000.0)))
        return cls(L=L, D=D, H=H,
                   KV=int(c.get("num_key_value_heads") or H), HD=HD,
                   FF=int(c["intermediate_size"]), V=int(c["vocab_size"]),
                   eps=float(c["rms_norm_eps"]),
                   theta=float(rp.get("rope_theta", c.get("rope_theta", 10000.0))),
                   tied=bool(c.get("tie_word_embeddings", True)),
                   max_pos=int(c.get("max_position_embeddings") or 0),
                   rot=0 if rot == HD else rot, rope_scaling=scaling,
                   # a float32 product at the lookup, as torch's (Gemma: sqrt(hidden) as a float32)
                   emb_mult=_f32(math.sqrt(D) if gemma else float(c.get("embedding_multiplier", 1.0))),
                   res_mult=float(c.get("residual_multiplier", 1.0)),
                   attn_mult=attn_mult,
                   logit_div=float(c.get("logits_scaling", 1.0)),
                   window=None if window is None else int(window),
                   act=ACTS.get(_act_key(c), _act_key(c)), norm_offset=gemma,
                   qk_norm=c.get("model_type") in ("gemma3_text", "qwen3"), post_norms=gemma,
                   layer_types=_layer_types(c, L), local_theta=local_theta)

    def layer_window(self, li: int) -> Optional[int]:
        """Layer li's sliding window (None: full attention, or a window no library's
        context reaches: MAX_CONTEXT)."""
        if self.window is None or (self.layer_types is not None and self.layer_types[li] != "sliding"):
            return None
        return self.window if self.window < MAX_CONTEXT else None

    def local_layer(self, li: int) -> bool:
        """Layer li rotates with the sliding layers' own RoPE (local_theta)."""
        return (self.local_theta is not None and self.layer_types is not None
                and self.layer_types[li] == "sliding")

    @property
    def plain(self) -> bool:
        """No weight transform (src/llm_weights.py): the block as it is stored."""
        return (not self.rot and self.res_mult == 1.0 and not self.norm_offset
                and self.attn_mult is None and self.logit_div == 1.0
                and (self.rope_scaling or {}).get("rope_type") not in ("yarn", "longrope"))

    @classmethod
    def from_file(cls, path: str, check: bool = True) -> "LlamaConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f), check)


__all__ = ("ACTS", "ArchError", "HARMLESS", "LAYER_TYPES", "LlamaConfig", "MAX_CONTEXT", "MODEL_TYPES",
           "PLAIN", "READ", "ROPE_TYPES", "TYPE_READ", "WINDOW_TYPES", "problems")
