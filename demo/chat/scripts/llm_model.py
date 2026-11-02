# Copyright 2026 Ivan Gubochkin
# SPDX-License-Identifier: Apache-2.0

"""
One checkpoint's identity for the chat scripts (doc/plans/CHAT_PLAN.md §28):
its names — tag, project directory, board paths, study directory — and its
profile: how it is prompted and gated (chat or text), its attention-sink
token, its end-of-generation ids and its context, all derived from the
checkpoint's own files (config.json, generation_config.json,
tokenizer_config.json / chat_template.jinja / chat_template.json and
tokenizer.json's added tokens), so a new checkpoint needs no constants.

Standard library only: llm_calibrate.py, llm_board.py and the llm-onboard
checker import it without numpy; llm_study.Profile adds the tokenization.
"""

from __future__ import annotations

import json
import os
import subprocess
from typing import Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
CHAT = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(CHAT))
DEFAULT_MODEL = "smollm2-135m-instruct"
CTX = 1024          # the context limit (CHAT_PLAN decision; the softmax unit's key count)

# Models that keep the names they had before every model was named after
# itself: SmolLM2-135M, the first (the chat server's defaults point there).
LEGACY = {DEFAULT_MODEL: {"project": "llm_project", "lib": "libsmollm2.so",
                          "weights": "/root/smollm2_weights", "study": ""}}


class ProfileError(ValueError):
    """A checkpoint whose prompting this stack does not implement."""


# ── names ────────────────────────────────────────────────────────────────────

def tag(name: str) -> str:
    """smollm2-360m-instruct -> smollm2_360m: the library, project and weights name."""
    return name.removesuffix("-instruct").replace("-", "_").replace(".", "_")


def model_name(assets: str) -> str:
    """The model's name: its checkpoint directory's name."""
    return os.path.basename(os.path.normpath(assets))


def study_dir(assets: str) -> str:
    """Where the study results and formats of the model in ``assets`` live:
    <assets>/../study/<model dir> (SmolLM2-135M: <assets>/../study itself,
    its original place)."""
    base = os.path.join(os.path.dirname(os.path.abspath(assets)), "study")
    sub = LEGACY.get(model_name(assets), {}).get("study", model_name(assets))
    return os.path.join(base, sub) if sub else base


def project_dir(name: str, build: Optional[str] = None) -> str:
    """The generated library project: demo/chat/build/llm_project_<tag>."""
    build = build or os.path.join(CHAT, "build")
    return os.path.join(build, LEGACY.get(name, {}).get("project", f"llm_project_{tag(name)}"))


def board_paths(name: str, remote_dir: str, weights_dir: Optional[str] = None) -> Dict[str, str]:
    """Board locations of a model's library, project sources and weights:
    <remote dir>/lib/lib<tag>.so, <remote dir>/llm_project_<tag>,
    /root/<tag>_weights (``weights_dir`` overrides the last)."""
    leg = LEGACY.get(name, {})
    t = tag(name)
    return {"lib": f"{remote_dir}/lib/{leg.get('lib', f'lib{t}.so')}",
            "proj": f"{remote_dir}/{leg.get('project', f'llm_project_{t}')}",
            "weights": weights_dir or leg.get("weights", f"/root/{t}_weights")}


def default_assets() -> str:
    """SMOLLM_ASSETS, else demo/chat/assets/<DEFAULT_MODEL> of this checkout
    (or of the main checkout, which a git worktree shares the assets of)."""
    if os.environ.get("SMOLLM_ASSETS"):
        return os.environ["SMOLLM_ASSETS"]
    rel = os.path.join("demo", "chat", "assets", DEFAULT_MODEL)
    cands = [os.path.join(REPO, rel)]
    try:
        common = subprocess.check_output(["git", "-C", REPO, "rev-parse", "--git-common-dir"],
                                         text=True, stderr=subprocess.DEVNULL).strip()
        cands.append(os.path.join(os.path.dirname(os.path.abspath(os.path.join(REPO, common))), rel))
    except Exception:
        pass
    for c in cands:
        if os.path.exists(os.path.join(c, "config.json")):
            return c
    return cands[0]


# ── profile ──────────────────────────────────────────────────────────────────

def _json(assets: str, name: str) -> dict:
    try:
        with open(os.path.join(assets, name)) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def chat_template(assets: str) -> Optional[str]:
    """The checkpoint's chat template: tokenizer_config.json's chat_template (a
    string, or the "default" of a named list), chat_template.jinja, or
    chat_template.json — None for a base model."""
    t = _json(assets, "tokenizer_config.json").get("chat_template")
    if isinstance(t, list):
        t = next((x["template"] for x in t if x.get("name") == "default"), None)
    if t:
        return t
    p = os.path.join(assets, "chat_template.jinja")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return f.read()
    return _json(assets, "chat_template.json").get("chat_template") or None


def added_tokens(assets: str) -> Dict[str, int]:
    """{content: id} of tokenizer.json's added tokens (the special tokens)."""
    return {a["content"]: int(a["id"]) for a in _json(assets, "tokenizer.json").get("added_tokens", [])}


def _token_id(assets: str, tok, added: Dict[str, int]) -> Optional[int]:
    """A tokenizer_config.json special token (a string or an AddedToken dict) -> id."""
    if isinstance(tok, dict):
        tok = tok.get("content")
    return added.get(tok) if tok else None


def _ids(v) -> List[int]:
    if v is None:
        return []
    return [int(i) for i in (v if isinstance(v, list) else [v])]


class Profile:
    """How a checkpoint is prompted and gated.

    "chat" — it has a chat template (llm_study renders ChatML only and refuses
    another; the VLM's Idefics3 template is vlm_study's).  "text" — none: a
    base model, studied on text it continues (llm_study.STORY_PROMPTS for
    TinyStories).

    sink:  the token of position 0, whose K / V rows are precomputed (the
           attention sink): the special token a chat rendering starts with,
           a text model's BOS.
    eog:   the ids that end a generation: generation_config.json's and
           config.json's eos_token_id and tokenizer_config.json's eos_token;
           a text model also stops at its BOS (the next document).
    ctx:   max_position_embeddings, at most CTX.
    """

    def __init__(self, assets: str):
        self.assets = assets
        conf = _json(assets, "config.json")           # the checkpoint's (not chat_config.json)
        gen = _json(assets, "generation_config.json")
        tcf = _json(assets, "tokenizer_config.json")
        added = added_tokens(assets)
        self.template = chat_template(assets)
        self.chat = self.template is not None
        self.kind = "chat" if self.chat else "text"
        bos = _ids(gen.get("bos_token_id", conf.get("bos_token_id"))) or \
            _ids(_token_id(assets, tcf.get("bos_token"), added))
        self.bos: Optional[int] = bos[0] if bos else None
        if self.chat:
            self.sink = self._first_special(added)
        else:
            if self.bos is None:
                raise ProfileError(f"{assets}: a model without a chat template needs a BOS token "
                                   f"(the attention sink at position 0)")
            self.sink = self.bos
        eos = set(_ids(gen.get("eos_token_id"))) | set(_ids(conf.get("eos_token_id")))
        e = _token_id(assets, tcf.get("eos_token"), added)
        if e is not None:
            eos.add(e)
        if not self.chat:
            eos.add(self.bos)
        if not eos:
            raise ProfileError(f"{assets}: no end-of-generation token (eos_token_id / eos_token)")
        self.eog: Tuple[int, ...] = tuple(sorted(eos))
        mp = int(conf.get("max_position_embeddings") or CTX)
        self.ctx = min(CTX, mp)
        self.windows = 3 * (CTX // self.ctx)        # held-out windows: 3 x CTX tokens either way

    def _first_special(self, added: Dict[str, int]) -> int:
        """The first token of every rendering: the BOS where the template writes
        bos_token before any special token's text (Llama-3, Mistral, Gemma),
        else the added token whose text comes first (ChatML: <|im_start|>)."""
        pos = {c: self.template.find(c) for c in added if c and c in self.template}
        b = self.template.find("bos_token")
        if b >= 0 and self.bos is not None and (not pos or b < min(pos.values())):
            return self.bos
        if not pos:
            raise ProfileError(f"{self.assets}: no special token in the chat template")
        return added[min(pos, key=lambda c: (pos[c], -len(c)))]


__all__ = ("CTX", "DEFAULT_MODEL", "LEGACY", "Profile", "ProfileError", "added_tokens",
           "board_paths", "chat_template", "default_assets", "model_name", "project_dir",
           "study_dir", "tag")
