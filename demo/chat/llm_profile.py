# Copyright 2026 Ivan Gubochkin
# SPDX-License-Identifier: Apache-2.0

"""llm_profile.py — how a checkpoint is prompted, from its own files
(doc/plans/CHAT_PLAN.md §28, §29): chat or text, the attention-sink token of
position 0, the end-of-generation ids, the context — for the study and the
gates (scripts/llm_model.py, llm_study.Profile) and the chat server (the
backends read it from the tokenizer directory deploy.py uploads:
tokenizer.json, tokenizer_config.json, chat_template.*, generation_config.json,
config.json).  The standard library and jinja2 (chat_template.py)."""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

from chat_template import SPECIAL_TOKEN_KEYS, ChatTemplate, special_tokens, template_source

CTX = 1024          # the context limit (CHAT_PLAN decision; the softmax unit's key count)


class ProfileError(ValueError):
    """A checkpoint whose prompting this stack does not implement."""


def _json(assets: str, name: str) -> dict:
    try:
        with open(os.path.join(assets, name)) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def chat_template(assets: str) -> Optional[str]:
    """The checkpoint's chat template source (chat_template.template_source),
    None for a base model."""
    return template_source(assets)


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

    "chat" — it has a chat template (``chat_template``: rendered as
    apply_chat_template does).  "text" — none: a base model, studied on text it
    continues (llm_study.STORY_PROMPTS for TinyStories).

    sink:  the token of position 0, whose K / V rows are precomputed (the
           attention sink): the special token every rendering of the chat
           template starts with, a text model's BOS.
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
        self.chat_template = ChatTemplate(self.template, special_tokens(assets)) if self.chat else None
        self.kind = "chat" if self.chat else "text"
        bos = _ids(gen.get("bos_token_id", conf.get("bos_token_id"))) or \
            _ids(_token_id(assets, tcf.get("bos_token"), added))
        self.bos: Optional[int] = bos[0] if bos else None
        if self.chat:
            self.sink = self._first_token(added)
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

    def _first_token(self, added: Dict[str, int]) -> int:
        """The special token renderings start with — one for a conversation that
        starts with a system message and one that does not (the sink is
        precomputed for a single token)."""
        firsts = {}
        for probe in ([{"role": "user", "content": "Hi"}],
                      [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}]):
            try:
                text = self.chat_template.render(probe)
            except Exception as e:                             # noqa: BLE001  (no system role)
                if probe[0]["role"] == "system":
                    continue
                raise ProfileError(f"{self.assets}: the chat template does not render: {e}") from None
            tok = max((c for c in added if c and text.startswith(c)), key=len, default=None)
            if tok is None:
                raise ProfileError(f"{self.assets}: the chat template's rendering does not start with "
                                   f"a special token ({text[:40]!r}): no fixed token for position 0, "
                                   f"the attention sink")
            firsts[probe[0]["role"]] = added[tok]
        if len(set(firsts.values())) != 1:
            raise ProfileError(f"{self.assets}: the chat template starts conversations with different "
                               f"tokens ({firsts}); the attention sink is precomputed for one")
        return next(iter(firsts.values()))

    @property
    def stop_ids(self) -> Tuple[int, ...]:
        """What ends a served answer: the end-of-generation ids and every special
        token tokenizer_config.json names (bos / eos / unk / pad: SmolLM2's
        <|endoftext|>, <|im_start|>, <|im_end|>) — a turn never contains them."""
        added = added_tokens(self.assets)
        ids = set(self.eog)
        for name, content in special_tokens(self.assets).items():
            if name in SPECIAL_TOKEN_KEYS and content in added:
                ids.add(added[content])
        return tuple(sorted(ids))


__all__ = ("CTX", "Profile", "ProfileError", "added_tokens", "chat_template")
