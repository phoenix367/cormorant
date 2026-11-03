# Copyright 2026 Ivan Gubochkin
# SPDX-License-Identifier: Apache-2.0

"""
A checkpoint's tensors under the plain Llama block's names, with the
architecture's extras folded into the weights (doc/plans/CHAT_PLAN.md §29) —
one view for the frontend (src/llama.py) and the study
(demo/chat/scripts/llm_study.py, which loads this file by path), so both
compute the same block from the same numbers.  numpy only.

  fused projections   qkv_proj -> q / k / v rows, gate_up_proj -> gate / up (Phi-3)
  partial rotation    q / k rows of every head in src/rope.py perm's order
  attention factor    yarn / longrope's factor on cos and sin: factor² on the
                      rotated rows of W_q
  attention_multiplier  the score scale instead of head_dim^-0.5: the ratio on W_q
  residual_multiplier   on W_o and W_down (the two residual branches)
  embedding_multiplier  on the embedding table (not on a tied LM head)
  logits_scaling        1 / it on the LM head
  the LM head         lm_head.weight, or the embedding when tied (a stored
                      lm_head of a tied checkpoint is ignored, as Hugging Face does)

A plain model (LlamaConfig.plain) reads every tensor unchanged: the same
arrays as before, so its libraries regenerate byte-identical.  Transforms are
applied in float64 and rounded once to float32.  Reads go through to the
checkpoint view, whose record of the tensors read (src/checkpoint.py) stays
the one check_consumed looks at.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Iterator

import numpy as np

try:
    from . import rope
except ImportError:                     # loaded by path (llm_study): the sibling module
    import importlib.util
    import os
    import sys
    _spec = importlib.util.spec_from_file_location(
        "_axi_src_rope", os.path.join(os.path.dirname(os.path.abspath(__file__)), "rope.py"))
    rope = sys.modules.get(_spec.name) or importlib.util.module_from_spec(_spec)
    if _spec.name not in sys.modules:
        sys.modules[_spec.name] = rope
        _spec.loader.exec_module(rope)

LAYER = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
         "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "input_layernorm",
         "post_attention_layernorm")

# The block's linears: (key, tensor name in the layer, input class, output class) —
# the classes are the formats' (llm_study.py calibrates them, src/llama.py reads them)
LINEARS = (("q", "self_attn.q_proj", "x", "q0"), ("k", "self_attn.k_proj", "x", "k0"),
           ("v", "self_attn.v_proj", "x", "v"), ("o", "self_attn.o_proj", "pv", "o"),
           ("g", "mlp.gate_proj", "x2", "g"), ("u", "mlp.up_proj", "x2", "u"),
           ("d", "mlp.down_proj", "a", "d"))


def rope_spec(cfg) -> "rope.RopeSpec":
    """The RoPE of a LlamaConfig (or the study's Cfg)."""
    return rope.RopeSpec(cfg.HD, cfg.theta, getattr(cfg, "rot", 0),
                         getattr(cfg, "rope_scaling", None), getattr(cfg, "max_pos", 0))


class Prepared(Mapping):
    """The plain block's tensors of ``W`` (see the module docstring)."""

    def __init__(self, cfg, W: Mapping):
        self.cfg, self.W = cfg, W
        if hasattr(W, "unread"):        # a recording checkpoint view: the tensors not read
            self.unread = W.unread      # (src/checkpoint.check_consumed)
        self.fused_qkv = "model.layers.0.self_attn.qkv_proj.weight" in W
        self.fused_mlp = "model.layers.0.mlp.gate_up_proj.weight" in W
        if not getattr(cfg, "tied", True) and "lm_head.weight" not in W:
            raise ValueError("tie_word_embeddings is false but the checkpoint has no lm_head.weight")
        spec = rope_spec(cfg)
        hd, rot = cfg.HD, spec.dims
        self.perm = None if rot == hd else rope.perm(spec)
        # per head-dim row scale of W_q: factor² on the rotated pairs, the score-scale ratio on all
        q = np.ones(hd, np.float64)
        af = rope.attention_factor(spec)
        if af != 1.0:
            npair, half = rot // 2, hd // 2
            q[:npair] *= af * af
            q[half:half + npair] *= af * af
        if getattr(cfg, "attn_mult", None) is not None:
            q *= cfg.attn_mult * np.sqrt(hd)
        self.q_scale = None if (q == 1.0).all() else q
        self._names = (["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"]
                       + [f"model.layers.{li}.{n}.weight" for li in range(cfg.L) for n in LAYER])

    # ── Mapping ──
    def __iter__(self) -> Iterator[str]:
        return iter(self._names)

    def __len__(self) -> int:
        return len(self._names)

    def __contains__(self, name) -> bool:
        return name in self._names

    def __getitem__(self, name: str) -> np.ndarray:
        cfg = self.cfg
        if name == "model.embed_tokens.weight":
            return _scaled(self.W[name], getattr(cfg, "emb_mult", 1.0))
        if name == "lm_head.weight":
            w = self.W["model.embed_tokens.weight"] if getattr(cfg, "tied", True) else self.W[name]
            return _scaled(w, 1.0 / getattr(cfg, "logit_div", 1.0))
        if not name.startswith("model.layers."):
            return self.W[name]
        li, part = name[len("model.layers."):].split(".", 1)
        part = part[:-len(".weight")]
        if part in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"):
            w = self._qkv(li, part)
            if part != "self_attn.v_proj":
                w = self._permuted(w)
            if part == "self_attn.q_proj" and self.q_scale is not None:
                w = self._row_scaled(w, self.q_scale)
            return w
        if part in ("mlp.gate_proj", "mlp.up_proj") and self.fused_mlp:
            w = self.W[f"model.layers.{li}.mlp.gate_up_proj.weight"]
            ff = getattr(cfg, "FF", None) or cfg.F                 # the frontend's / the study's name
            return w[:ff] if part == "mlp.gate_proj" else w[ff:]
        if part in ("self_attn.o_proj", "mlp.down_proj"):
            return _scaled(self.W[name], getattr(cfg, "res_mult", 1.0))
        return self.W[name]

    # ── transforms ──
    def _qkv(self, li: str, part: str) -> np.ndarray:
        if not self.fused_qkv:
            return self.W[f"model.layers.{li}.{part}.weight"]
        cfg = self.cfg
        w = self.W[f"model.layers.{li}.self_attn.qkv_proj.weight"]
        nq, nk = cfg.H * cfg.HD, cfg.KV * cfg.HD
        return {"self_attn.q_proj": w[:nq], "self_attn.k_proj": w[nq:nq + nk],
                "self_attn.v_proj": w[nq + nk:]}[part]

    def _permuted(self, w: np.ndarray) -> np.ndarray:
        if self.perm is None:
            return w
        hd = self.cfg.HD
        return w.reshape(-1, hd, w.shape[-1])[:, self.perm, :].reshape(w.shape)

    def _row_scaled(self, w: np.ndarray, s: np.ndarray) -> np.ndarray:
        hd = self.cfg.HD
        return (w.reshape(-1, hd, w.shape[-1]).astype(np.float64) * s[None, :, None]
                ).reshape(w.shape).astype(np.float32)


def _scaled(w: np.ndarray, s: float) -> np.ndarray:
    return w if s == 1.0 else (np.asarray(w, np.float64) * s).astype(np.float32)


def prepare(cfg, W: Mapping) -> Mapping:
    """``W`` as the plain block's tensors (idempotent)."""
    return W if isinstance(W, Prepared) or getattr(W, "_axi_prepared", False) else Prepared(cfg, W)


Prepared._axi_prepared = True         # recognised across the two module copies (package / by path)

__all__ = ("LAYER", "Prepared", "prepare", "rope_spec")
