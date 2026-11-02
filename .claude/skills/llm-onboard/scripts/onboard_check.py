#!/usr/bin/env python3
# Copyright 2026 Ivan Gubochkin
# SPDX-License-Identifier: Apache-2.0

"""onboard_check.py — llm-onboard's intake gate for a Llama-family checkpoint
that passed model-study (llama_fit.py eligible, numerics GO): which steps of
the SmolLM2 path it can take UNCHANGED and which need code, before an hour of
calibration and a 30 GB project generation.  Host only, stdlib only, reads the
assets directory (after `llm_calibrate.py add` / `fetch`).

What llama_fit.py does not check, and where the path hard-codes SmolLM2:
  * tokenizer.json   the server's pure-Python tokenizer (demo/chat/
                     hf_tokenizer.py) interprets the pipeline (BPE with the
                     usual normalizers, pre-tokenizers, decoders and added
                     tokens) and raises ValueError naming a part it does not
                     support (a WordPiece / Unigram model, a Precompiled
                     normalizer, ...); validate_text.py --assets checks a
                     loaded one against `tokenizers`;
  * special ids      <|im_start|> = 1 is the attention sink everywhere
                     (llm_study.py sink_token 1 and held-out windows,
                     llm_sched_check.py's assert), <|im_end|> = 2 is
                     llm_study.EOS; the backend finds them by name;
  * chat template    llm_study.chatml() and demo/chat/chatml.py are SmolLM2's
                     ChatML with its default system prompt;
  * prompt ids       llm_project.tokenize_prompts() / second_turn_ids() — the
                     gates' prompts — always use the SmolLM2-135M tokenizer and
                     cache the ids in assets/study/phase3_prompt_ids.json /
                     phase5_second_turn_ids.json; llm_lib_check.py hard-codes
                     SmolLM2 ids; valid only for SmolLM2's exact tokenizer.json;
  * vocabulary       the backend refuses a library whose llm_vocab_size() is
                     not the tokenizer's size;
  * K bound          every linear's K (hidden, heads x head_dim, FFN) must fit
                     MatmulKernel's max_k (platforms/<p>.json): the decode GEMV
                     runs there and src/llama.py does not split K;
  * names            --model-name -> project dir, board library, weights dir,
                     served model id; a name equal to a served one replaces it.

usage: python3 .claude/skills/llm-onboard/scripts/onboard_check.py demo/chat/assets/<name>
           [--name MODEL_NAME] [--platform kv260]
exit: 0 = the SmolLM2 text path applies unchanged (serving needs only a
backend slot), 1 = code needed (listed), 2 = missing input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
CHAT = REPO / "demo" / "chat"
sys.path.insert(0, str(CHAT / "scripts"))
import llm_model                                    # noqa: E402  (the scripts' names; stdlib only)
MANIFEST = CHAT / "scripts" / "llm_models.json"
# SHA-256 of SmolLM2-135M / 360M-Instruct's tokenizer_config.json chat_template
# (identical in both; demo/chat/chatml.py and llm_study.chatml() render it).
SMOLLM2_TEMPLATE_SHA = "872be49dbb638044ad01b60388f48d469ff2980e5f0dccdc22ec907db54d0788"
SERVED = {                                             # model id -> backend (kv260_chat_server.py)
    "smollm2-135m-instruct": "smollm2 (--llm-*, chat_config.json smollm2 block)",
    "smollm2-360m-instruct": "smollm2-360m (--llm-360m-*, smollm2_360m block)",
    "smolvlm-256m-instruct": "smolvlm (--vlm-*, smolvlm block)",
    "stories15m": "stories15m (--stories-*, stories15m block)",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("assets", help="the checkpoint directory (config.json, tokenizer.json, ...)")
    ap.add_argument("--name", default=None,
                    help="the model name (--model-name, llm_models.json key); default: the "
                         "directory name")
    ap.add_argument("--platform", default="kv260")
    args = ap.parse_args(argv)
    d = Path(args.assets)
    name = args.name or d.resolve().name
    reuse, code, notes = [], [], []

    cfg_path = d / "config.json"
    if not cfg_path.exists():
        print(f"error: {cfg_path} missing — `llm_calibrate.py add` / `fetch` first", file=sys.stderr)
        return 2
    c = json.loads(cfg_path.read_text())
    vlm = "vision_config" in c
    tc = c.get("text_config", c) if vlm else c

    # ── the block's K bound (llama_fit.py covers the rest of the block) ──
    plat = json.loads((REPO / "platforms" / f"{args.platform}.json").read_text())
    max_k = int(plat["kernels"]["matmul"]["max_k"])
    D, H = int(tc["hidden_size"]), int(tc["num_attention_heads"])
    HD = int(tc.get("head_dim") or D // H)
    F = int(tc["intermediate_size"])
    ks = {"hidden (q/k/v/gate/up)": D, "heads x head_dim (o_proj)": H * HD, "FFN (down_proj)": F}
    over = {k: v for k, v in ks.items() if v > max_k}
    if over:
        code.append("K > MatmulKernel max_k %d: %s — the decode GEMV cannot run it; split K in "
                    "src/llama.py (as src/vit.py splits the SmolVLM connector) or raise the "
                    "bound (a bitstream change)" % (max_k, ", ".join(f"{k} {v}" for k, v in over.items())))
    else:
        reuse.append(f"K bound: hidden {D}, o_proj {H * HD}, FFN {F} <= max_k {max_k}")
    if not (d / "model.safetensors").exists() and not (d / "model.safetensors.index.json").exists():
        code.append("no model.safetensors / model.safetensors.index.json in the checkpoint")

    bp = llm_model.board_paths(name, "<remote.dir>")
    notes.append(f"names: --model-name {name} -> demo/chat/build/"
                 f"{Path(llm_model.project_dir(name)).name}, board {bp['lib']}, weights "
                 f"{bp['weights']}, served id {name}")
    if name in SERVED:
        notes.append(f"{name} is served today by backend {SERVED[name]}: onboarding it replaces "
                     f"that model's library")
    elif not vlm:
        notes.append(f"serving: no backend serves {name} — add a backend entry (the smollm2-360m "
                     f"pattern) or repoint an existing slot's lib / tokenizer / cma_mb (replaces "
                     f"that model)")
    if vlm and name == "smolvlm-256m-instruct":
        reuse.append("SmolVLM-256M has its own text / image path: pins vlm_study_inputs.json, "
                     "vlm_study.py fetch / formats, vlm_sched_check.py --text, vlm_host_emu.py, "
                     "idefics3.py + the smolvlm backend")
        return report(name, d, reuse, code, notes, "the SmolVLM path")
    if vlm:
        code.append(f"a VLM ({c.get('model_type')}): vlm_study.py, vlm_project.py, "
                    f"vlm_sched_check.py, vlm_host_emu.py and smolvlm_backend.py are wired to "
                    f"SmolVLM-256M (vlm_study_inputs.json, vp.MODEL, idefics3.py, 512 x 512 tile) "
                    f"— model-study route C; the text checks below are for its text model")

    # ── text side ──
    tj, tcj = d / "tokenizer.json", d / "tokenizer_config.json"
    if not tj.exists() or not tcj.exists():
        print(f"error: {tj.name} / {tcj.name} missing in {d}", file=sys.stderr)
        return 2
    sys.path.insert(0, str(CHAT))
    from hf_tokenizer import Tokenizer                             # noqa: E402  (stdlib only)
    tok = None
    try:
        tok = Tokenizer(str(tj))
        reuse.append(f"tokenizer.json pipeline: hf_tokenizer.py, {tok.vocab_size} tokens — check it "
                     f"against `tokenizers`: $SPY demo/chat/scripts/validate_text.py --assets {d}")
    except ValueError as e:
        code.append(f"tokenizer.json: hf_tokenizer.py refuses it — {e}; the server's tokenizer "
                    f"(demo/chat/hf_tokenizer.py) needs that part, validate_text.py its check")
    V = int(tc["vocab_size"])
    if tok is not None and tok.vocab_size > V:
        code.append(f"tokenizer size {tok.vocab_size} > config vocab_size {V}: ids beyond the "
                    f"library's logits")
    # the sink, the end-of-generation ids and the context, as the study and the gates derive them
    try:
        prof = llm_model.Profile(str(d))
        reuse.append(f"profile {prof.kind}: sink token {prof.sink}, end of generation "
                     f"{list(prof.eog)}, context {prof.ctx} (llm_model.Profile)")
    except llm_model.ProfileError as e:
        code.append(f"profile: {e}")
        prof = None
    if prof is not None and prof.chat:
        tmpl = prof.template
        if hashlib.sha256(tmpl.encode()).hexdigest() == SMOLLM2_TEMPLATE_SHA:
            reuse.append("chat template: SmolLM2's ChatML with its default system prompt "
                         "(chatml.py, llm_study.chatml)")
        elif "<|im_start|>" in tmpl and "<|im_end|>" in tmpl:
            code.append("a ChatML template other than SmolLM2's: llm_study.chatml() and "
                        "demo/chat/chatml.py render SmolLM2's (its default system prompt); "
                        "`llm_calibrate.py validate` fails where they differ")
        else:
            code.append("the chat template is not ChatML: llm_study.chatml() (study, prompts, "
                        "gates) and demo/chat/chatml.py (server) render ChatML only")
        if tok is not None and tok.special.get("<|im_start|>") != prof.sink:
            code.append(f"the sink is token {prof.sink}, but the chat backend (smollm2_backend.py) "
                        f"takes <|im_start|> = {tok.special.get('<|im_start|>')}")
    elif prof is not None:
        notes.append("no chat template: a base model — the text profile (llm_study.STORY_PROMPTS "
                     "and the TinyStories texts, the stories backend's chat mapping) assumes "
                     "TinyStories-style stories")

    try:
        pinned = name in json.loads(MANIFEST.read_text())["models"]
    except (OSError, ValueError, KeyError):
        pinned = False
    (reuse if pinned else notes).append(
        f"llm_models.json: {name} {'pinned' if pinned else 'NOT pinned — llm_calibrate.py add'}")
    return report(name, d, reuse, code, notes)


def report(name, d, reuse, code, notes, path="the SmolLM2 path") -> int:
    print(f"== {name}  ({d})")
    for x in reuse:
        print(f"  reuse  {x}")
    for x in code:
        print(f"  CODE   {x}")
    for x in notes:
        print(f"  note   {x}")
    print("verdict: " + (f"{path} applies unchanged (serving: a backend slot)" if not code
                         else f"{len(code)} item(s) need code before the path applies"))
    return 1 if code else 0


if __name__ == "__main__":
    sys.exit(main())
