# Copyright 2026 Ivan Gubochkin
# SPDX-License-Identifier: Apache-2.0

"""
One checkpoint's identity for the chat scripts (doc/plans/CHAT_PLAN.md §28):
its names — tag, project directory, board paths, study directory — and its
profile (demo/chat/llm_profile.py, which the chat server uses too): how it is
prompted and gated (chat or text), its attention-sink token, its
end-of-generation ids and its context, all derived from the checkpoint's own
files, so a new checkpoint needs no constants.

The standard library and jinja2 (a chat model's template is rendered to find
its first token; demo/chat/chat_template.py): llm_calibrate.py, llm_board.py
and the llm-onboard checker import it without numpy; llm_study.Profile adds
the tokenization.
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Dict, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
CHAT = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(CHAT))
if CHAT not in sys.path:
    sys.path.insert(0, CHAT)
from chat_template import ChatTemplate  # noqa: E402,F401
from llm_profile import CTX, Profile, ProfileError, added_tokens, chat_template  # noqa: E402,F401
DEFAULT_MODEL = "smollm2-135m-instruct"

# Models that keep the names they had before every model was named after
# itself: SmolLM2-135M, the first (the chat server's defaults point there).
LEGACY = {DEFAULT_MODEL: {"project": "llm_project", "lib": "libsmollm2.so",
                          "weights": "/root/smollm2_weights", "study": ""}}


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


__all__ = ("CTX", "ChatTemplate", "DEFAULT_MODEL", "LEGACY", "Profile", "ProfileError", "added_tokens",
           "board_paths", "chat_template", "default_assets", "model_name", "project_dir",
           "study_dir", "tag")
