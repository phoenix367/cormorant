#!/usr/bin/env python3
"""llama_fit.py — first gate of a Llama-family decoder study (model-study
skill, route A), from config.json alone, before downloading the weights:

  * route A eligibility: what llm_study.py / src/llama.py implement — a plain
    Llama block (RMSNorm, rotate-half RoPE without scaling, GQA, SwiGLU with
    SiLU, no q/k/v/o or MLP biases) and one model.safetensors file;
  * parameters, the DMA pool (int16 weights incl. the LM head, int16 KV
    caches at the context, intermediates) against cma=1000M and the models
    already served (they swap under --resident auto);
  * decode ms / token and 256-token prefill, scaled from the shipped
    SmolLM2 libraries (decode streams every weight once per token; prefill
    grows with the MACs).

Calibration (board, CHAT_PLAN §20): pool 286 / 740 MiB, decode at position
32 99 / 256 ms per token, prefill 256 tokens 1.28 / 2.90 s for SmolLM2-135M /
360M; the constants below reproduce both.  Decode slows with the position
(360M: 306 ms at 1000).  Estimates only — the numeric study decides.

usage: python3 .claude/skills/model-study/scripts/llama_fit.py CONFIG.json|DIR [--context 1024]
       [--files FILE ...]   (checkpoint file names, e.g. from the Hugging Face file list)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CMA_MB = 1000.0
RESIDENT = {"bert-squad": 216, "smollm2-135m": 286, "smollm2-360m": 740, "smolvlm-256m": 495}  # MiB
INTERM_MIB_PER_HIDDEN = 0.0108  # intermediates: +6.2 MiB (135M, hidden 576), +10.4 (360M, 960)
DECODE_GBS = 2.89               # weight bytes streamed per second in decode (GEMV, both ports)
DECODE_FIXED_MS = 5.8           # host ops per token beyond the weight stream
PREFILL256_S = (0.321, 0.00713)  # s = a + b * M params (1.28 s at 134.5 M, 2.90 s at 361.8 M)
GEN_RAM_GB = (12.6, 134.5, 31.9, 361.8)   # host RAM peak generating the library (135M, 360M)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--context", type=int, default=1024)
    ap.add_argument("--files", nargs="*", default=None,
                    help="the checkpoint's file names (default: the files next to config.json)")
    args = ap.parse_args(argv)
    p = Path(args.config)
    p = p / "config.json" if p.is_dir() else p
    c = json.loads(p.read_text())
    if "text_config" in c:                         # a VLM: the text model's block
        print("note: a multimodal config; checking its text_config (the vision tower is route C)")
        c = {**c["text_config"], "tie_word_embeddings": c.get("tie_word_embeddings",
                                                              c["text_config"].get("tie_word_embeddings"))}
    files = args.files if args.files is not None else [f.name for f in p.parent.iterdir()]

    L, D, H = c["num_hidden_layers"], c["hidden_size"], c["num_attention_heads"]
    KV = c.get("num_key_value_heads", H)
    F, V = c["intermediate_size"], c["vocab_size"]
    HD = c.get("head_dim") or D // H
    tied = c.get("tie_word_embeddings", True)

    issues = []
    arch = (c.get("architectures") or ["?"])[0]
    if c.get("model_type") != "llama" and arch != "LlamaForCausalLM":
        issues.append(f"model_type {c.get('model_type')!r} / {arch}: not a Llama block "
                      f"(the frontend and the study implement Llama only)")
    if c.get("rope_scaling"):
        issues.append(f"rope_scaling {c['rope_scaling']}: not implemented (plain rotate-half RoPE)")
    if c.get("hidden_act", "silu") != "silu":
        issues.append(f"hidden_act {c['hidden_act']!r}: SwiGLU with SiLU only")
    for k in ("attention_bias", "mlp_bias"):
        if c.get(k):
            issues.append(f"{k}: biases are not implemented")
    if D % H or HD * H != D and "head_dim" not in c:
        issues.append(f"hidden {D} / heads {H}: head_dim not integral")
    if H % KV:
        issues.append(f"{H} heads over {KV} KV heads: not a GQA grouping")
    if files and "model.safetensors" not in files:
        shards = [f for f in files if f.endswith(".safetensors")]
        issues.append("no single model.safetensors" + (f" (sharded: {len(shards)} files; merge them "
                      "or extend load_safetensors)" if shards else ""))

    per_layer = D * H * HD + 2 * D * KV * HD + H * HD * D + 3 * D * F + 2 * D
    params = V * D + L * per_layer + D + (0 if tied else V * D)
    w_bytes = 2 * params                       # int16; a tied head reads the embedding table
    kv_bytes = 2 * L * KV * HD * args.context * 2
    pool_mib = (w_bytes + kv_bytes) / 2 ** 20 + INTERM_MIB_PER_HIDDEN * D
    decode_ms = w_bytes / (DECODE_GBS * 1e9) * 1e3 + DECODE_FIXED_MS
    prefill_ms = (PREFILL256_S[0] + PREFILL256_S[1] * params / 1e6) * 1e3
    r0, p0, r1, p1 = GEN_RAM_GB
    gen_gb = r0 + (r1 - r0) * (params / 1e6 - p0) / (p1 - p0)

    print(f"== {p}")
    print(f"block      {arch}: {L} layers, hidden {D}, {H} heads / {KV} KV (head_dim {HD}), "
          f"FFN {F}, vocab {V}, {'tied' if tied else 'untied'} head, "
          f"rope_theta {c.get('rope_theta', 10000.0):g}, ctx {c.get('max_position_embeddings', '?')}")
    print("route A    " + ("eligible" if not issues else "NOT eligible as is:"))
    for i in issues:
        print(f"  - {i}")
    print(f"params     {params / 1e6:.1f} M  (int16 weights {w_bytes / 2 ** 20:.0f} MiB, "
          f"KV caches at {args.context} tokens {kv_bytes / 2 ** 20:.1f} MiB)")
    free = CMA_MB * 1e6 / 2 ** 20
    beside = [k for k, v in RESIDENT.items() if pool_mib + v <= free - 32]
    verdict = ("does NOT fit cma=1000M" if pool_mib > free - 32 else
               f"fits; stays resident beside: {', '.join(beside) or 'none (swaps with every model)'}")
    print(f"pool       ~{pool_mib:.0f} MiB of {free:.0f} MiB CMA: {verdict}")
    print(f"speed      decode ~{decode_ms:.0f} ms / token (~{1e3 / decode_ms:.1f} tok/s) at position 32, "
          f"prefill 256 ~{prefill_ms / 1e3:.2f} s  (fitted on SmolLM2-135M / 360M on the board)")
    print(f"host RAM   generating the library ~{gen_gb:.0f} GB peak (135M 12.6, 360M 31.9; "
          f"the dev PC has 46 GB)")
    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main())
