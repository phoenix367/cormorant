#!/usr/bin/env python3
"""onboard_check.py — llm-onboard's intake gate for a Llama-family checkpoint
that passed model-study (llama_fit.py eligible, numerics GO): which steps of
the SmolLM2 path it can take UNCHANGED and which need code, before an hour of
calibration and a 30 GB project generation.  Host only, stdlib only, reads the
assets directory (after `llm_calibrate.py add` / `fetch`).

What llama_fit.py does not check, and where the path hard-codes SmolLM2:
  * tokenizer.json   the server's pure-Python tokenizer (demo/chat/
                     smollm2_tokenizer.py) accepts SmolLM2's pipeline only
                     (byte-level BPE, Digits(individual) + ByteLevel, no
                     normalizer) and raises otherwise;
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
MANIFEST = CHAT / "scripts" / "llm_models.json"
# SHA-256 of SmolLM2-135M / 360M-Instruct's tokenizer_config.json chat_template
# (identical in both; demo/chat/chatml.py and llm_study.chatml() render it).
SMOLLM2_TEMPLATE_SHA = "872be49dbb638044ad01b60388f48d469ff2980e5f0dccdc22ec907db54d0788"
SMOLLM2_TOKENIZER_MODEL = "smollm2-135m-instruct"     # whose tokenizer the gates' prompt ids use
SERVED = {                                             # model id -> backend (kv260_chat_server.py)
    "smollm2-135m-instruct": "smollm2 (--llm-*, chat_config.json smollm2 block)",
    "smollm2-360m-instruct": "smollm2-360m (--llm-360m-*, smollm2_360m block)",
    "smolvlm-256m-instruct": "smolvlm (--vlm-*, smolvlm block)",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def tag_of(name: str) -> str:
    """generate_llm_project.py / llm_board.board_paths naming."""
    return name.removesuffix("-instruct").replace("-", "_").replace(".", "_")


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
    shards = sorted(p.name for p in d.glob("*.safetensors"))
    if "model.safetensors" not in shards:
        code.append(f"no single model.safetensors ({shards or 'none'}): load_safetensors() and "
                    f"llm_calibrate.py's fixed file list read one file")

    tag = tag_of(name)
    default = name == "smollm2-135m-instruct"
    notes.append(f"names: --model-name {name} -> demo/chat/build/"
                 f"{'llm_project' if default else 'llm_project_' + tag}, board <remote.dir>/lib/"
                 f"{'libsmollm2' if default else 'lib' + tag}.so, weights "
                 f"/root/{'smollm2' if default else tag}_weights, served id {name}")
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
    from smollm2_tokenizer import Tokenizer               # noqa: E402  (stdlib only)
    tok = None
    try:
        tok = Tokenizer(str(tj))
        reuse.append(f"tokenizer.json pipeline: SmolLM2's (smollm2_tokenizer.py loads it; "
                     f"{tok.vocab_size} tokens)")
    except ValueError as e:
        code.append(f"tokenizer.json: {e} — the board server's tokenizer (smollm2_tokenizer.py) "
                    f"and its validation (validate_text.py) need a new pipeline")
    V = int(tc["vocab_size"])
    if tok is not None:
        s, e = tok.special.get("<|im_start|>"), tok.special.get("<|im_end|>")
        if (s, e) == (1, 2):
            reuse.append("special ids: <|im_start|> = 1 (the sink), <|im_end|> = 2 (EOS)")
        else:
            code.append(f"special ids <|im_start|> = {s}, <|im_end|> = {e}, SmolLM2's are 1 / 2: "
                        f"llm_study.py (sink_token 1, held-out windows, EOS 2), llm_sched_check.py "
                        f"(assert toks[0] == 1) and the backend's sink / stop tokens")
        if tok.vocab_size != V:
            code.append(f"tokenizer size {tok.vocab_size} != config vocab_size {V}: the backend "
                        f"refuses the library at load (llm_vocab_size() = {V})")
    tmpl = json.loads(tcj.read_text()).get("chat_template")
    if not isinstance(tmpl, str) and (d / "chat_template.json").exists():
        tmpl = json.loads((d / "chat_template.json").read_text()).get("chat_template")
    if isinstance(tmpl, str) and hashlib.sha256(tmpl.encode()).hexdigest() == SMOLLM2_TEMPLATE_SHA:
        reuse.append("chat template: SmolLM2's ChatML with its default system prompt "
                     "(chatml.py, llm_study.chatml)")
    else:
        code.append("chat template differs from SmolLM2's ChatML: llm_study.chatml() (study, "
                    "prompts, validate) and demo/chat/chatml.py (server) render SmolLM2's only; "
                    "`llm_calibrate.py validate` prints how many prompts match HF")
    try:
        m = json.loads(MANIFEST.read_text())
        ref = m["models"][SMOLLM2_TOKENIZER_MODEL]["files"]["tokenizer.json"]
    except (OSError, KeyError, ValueError):
        ref = None
    if ref and sha256(tj) == ref:
        reuse.append("tokenizer.json byte-identical to SmolLM2-135M's: the gates' prompt ids "
                     "(assets/study/phase3_prompt_ids.json, phase5_second_turn_ids.json) and "
                     "llm_lib_check.py's ids apply")
    else:
        code.append("tokenizer.json differs from SmolLM2-135M's: llm_project.tokenize_prompts() "
                    "/ second_turn_ids() (llm_sched_check, llm_host_emu, llm_board) would feed the "
                    "135M tokenizer's ids from a shared cache, llm_lib_check.py hard-codes 135M "
                    "ids — make the caches per model first")

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
