#!/usr/bin/env python3
"""Tiny random Llama fixtures (doc/CHAT_PLAN.md phase 3, gate 1).

A Llama-architecture model small enough to schedule, simulate and run in
seconds — hidden 64, 2 layers, 4 heads over 2 KV heads (GQA), head_dim 16,
FFN 128, vocab 256, context 32 — with random bf16 weights.  Its numeric
formats (power-of-two exponents per class and layer, the position-0 sink
rows) come from the study's own calibration code
(demo/chat/scripts/llm_study.py: ``Model`` with ``record_ch``,
``make_formats``, ``_write_sink``), exactly as ``llm_study.py formats``
makes them for SmolLM2, so the fixtures exercise the same machinery.

Writes test/models/llama_tiny_{decode,prefill8,head}.onnx (the prefill
fixture with the head, so its smoke test checks logits) and
test/models/llama_tiny.json (config + formats; the test suite rebuilds the
weights from the seed).

    .venv/bin/python test/gen_llama_models.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REPO = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(REPO, "demo", "chat", "scripts"))

import llm_study as study                      # noqa: E402
from src.llama import (Formats, LlamaConfig,   # noqa: E402
                       LlamaFrontend)

OUT_DIR = os.path.join(HERE, "models")   # gen_all_models.py sets it
POLICY = "pow2+sink+p12+xattn"
TINY = dict(hidden_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            head_dim=16, intermediate_size=128, vocab_size=256, rms_norm_eps=1e-5,
            rope_theta=10000.0, tie_word_embeddings=True)
TINY_CTX = 32


def bf16(x):
    u = np.asarray(x, np.float32).view(np.uint32)
    u = (u + np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))) & np.uint32(0xFFFF0000)
    return u.view(np.float32)


def tiny_weights(cfg: dict, seed: int = 0) -> dict:
    """Random bf16 weights under Hugging Face Llama names (tied embedding)."""
    rng = np.random.default_rng(seed)
    D, F, L, V = cfg["hidden_size"], cfg["intermediate_size"], cfg["num_hidden_layers"], \
        cfg["vocab_size"]
    H, KV, HD = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
    W = {"model.embed_tokens.weight": bf16(rng.normal(0, 0.6, (V, D))),
         "model.norm.weight": bf16(1 + rng.normal(0, 0.2, D))}
    shapes = {"self_attn.q_proj": (H * HD, D), "self_attn.k_proj": (KV * HD, D),
              "self_attn.v_proj": (KV * HD, D), "self_attn.o_proj": (D, H * HD),
              "mlp.gate_proj": (F, D), "mlp.up_proj": (F, D), "mlp.down_proj": (D, F)}
    for l in range(L):
        for name, (o, i) in shapes.items():
            W[f"model.layers.{l}.{name}.weight"] = bf16(rng.normal(0, 1.2 / np.sqrt(i), (o, i)))
        W[f"model.layers.{l}.input_layernorm.weight"] = bf16(1 + rng.normal(0, 0.2, D))
        W[f"model.layers.{l}.post_attention_layernorm.weight"] = bf16(1 + rng.normal(0, 0.2, D))
    return W


def study_cfg(cfg: dict):
    """llm_study.Cfg for an in-memory config dict."""
    c = study.Cfg.__new__(study.Cfg)
    c.L = cfg["num_hidden_layers"]; c.D = cfg["hidden_size"]; c.H = cfg["num_attention_heads"]
    c.KV = cfg["num_key_value_heads"]; c.F = cfg["intermediate_size"]; c.V = cfg["vocab_size"]
    c.HD = cfg.get("head_dim") or c.D // c.H
    c.eps = float(cfg["rms_norm_eps"]); c.theta = float(cfg["rope_theta"])
    c.tied = cfg.get("tie_word_embeddings", True)
    return c


def calibrate(cfg: dict, W: dict, policy: str = POLICY, seed: int = 1, n_seq: int = 4,
              length: int = 24) -> dict:
    """llm_study.py `formats` for an in-memory model: per-channel calibration
    maxima of a float run (sink at position 0 = token 1), make_formats, and
    the sink K / V rows at the cache exponents.  Returns the formats JSON."""
    sc = study_cfg(cfg)
    rng = np.random.default_rng(seed)
    calib = [np.concatenate([[1], rng.integers(2, sc.V, length - 1)]) for _ in range(n_seq)]
    pol = study.POLICIES[policy]
    fm = study.Model(W, sc, None)
    cm = study.Model(W, sc, None, base=fm, sink=True, sink_model=fm, record_ch=True)
    for c in calib:
        study.teacher_forced(cm, c)
    fmt = study.make_formats(pol, cm.stats.ch, W, sc)
    out = {"policy": policy, "spec": pol, "margin": study.MARGIN,
           "exponents": {f"{k[0]}@{k[1]}": np.asarray(v).tolist() for k, v in fmt.items()}}
    m = study.Model(W, sc, pol, fmt, sink_model=fm)
    s = study.Seq(sc, 1)
    m._write_sink(s, 1)
    out["sink_token"] = 1
    out["sink_k_raw"] = [np.rint(s.k[l][:, 0] * study.p2(m.Eh("k", l, sc.KV))[:, None])
                         .astype(int).tolist() for l in range(sc.L)]
    out["sink_v_raw"] = [np.rint(s.v[l][:, 0] * study.p2(m.Ev("vc", l, sc.KV * sc.HD)
                                                        .reshape(sc.KV, sc.HD)))
                         .astype(int).tolist() for l in range(sc.L)]
    return out


def study_model(cfg: dict, W: dict, formats: dict, policy: str = POLICY):
    """The study's emulation of `policy` with the given formats (keys
    "cls@layer" -> (cls, layer))."""
    sc = study_cfg(cfg)
    fmt = {}
    for k, v in formats["exponents"].items():
        cls_, l = k.split("@")
        fmt[(cls_, int(l))] = v if isinstance(v, int) else np.asarray(v, np.int64)
    fm = study.Model(W, sc, None)
    return study.Model(W, sc, study.POLICIES[policy], fmt, sink_model=fm), sc


def frontend(cfg: dict, W: dict, formats: dict, ctx: int = TINY_CTX, name: str = "llama_tiny"):
    lc = LlamaConfig.from_dict(cfg)
    return LlamaFrontend(lc, W, Formats(formats, lc), ctx=ctx, name=name)


def tiny(seed: int = 0):
    """(config, weights, formats, frontend) of the tiny fixture."""
    W = tiny_weights(TINY, seed)
    formats = calibrate(TINY, W)
    return TINY, W, formats, frontend(TINY, W, formats)


def main():
    import onnx
    os.makedirs(OUT_DIR, exist_ok=True)
    cfg, W, formats, fe = tiny(0)
    for fname, (kind, T, wh) in {"llama_tiny_decode.onnx": ("decode", 1, False),
                                 "llama_tiny_prefill8.onnx": ("prefill", 8, True),
                                 "llama_tiny_head.onnx": ("head", 1, False)}.items():
        onnx.save(fe.entry(kind, T, with_head=wh), os.path.join(OUT_DIR, fname))
        print(os.path.join(OUT_DIR, fname))
    with open(os.path.join(OUT_DIR, "llama_tiny.json"), "w") as f:
        json.dump({"config": cfg, "context": TINY_CTX, "seed": 0, "formats": formats}, f)
    print(os.path.join(OUT_DIR, "llama_tiny.json"))


if __name__ == "__main__":
    main()
