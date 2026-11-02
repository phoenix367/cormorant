# Copyright 2026 Ivan Gubochkin
# SPDX-License-Identifier: Apache-2.0

"""chat_template.py — a checkpoint's own chat template, rendered as Hugging
Face's apply_chat_template does, and the prompt builder of the chat server
(doc/plans/CHAT_PLAN.md §29).

The template is the checkpoint's: tokenizer_config.json's chat_template (a
string, or the "default" of a named list), chat_template.jinja or
chat_template.json.  It renders as transformers' chat_template_utils does: a
sandboxed Jinja environment with trim_blocks / lstrip_blocks and loop
controls, transformers' tojson, raise_exception and strftime_now, a
passthrough {% generation %} tag, and the tokenizer's special tokens
(bos_token, eos_token, ...) as variables.  The rendered text is tokenized
whole, without special tokens added (the template writes them), as
apply_chat_template(tokenize=True) does.  jinja2 is the only dependency
outside the standard library (the board: Ubuntu 22.04's python3-jinja2 3.0;
imported when a template is first compiled).

Trimming (PromptBuilder.fit): the prompt must leave `reserve` positions of the
context for the answer.  The system message (if the conversation starts with
one) and the last message are always kept; older messages are dropped from
the front, and an assistant message left at the front is dropped with its
user turn, so the kept history starts with a user message.  If the system
message and the last message alone do not fit -> ContextTooLong.
"""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from datetime import datetime
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# tokenizer_config.json's special tokens transformers passes to the template
SPECIAL_TOKEN_KEYS = ("bos_token", "eos_token", "unk_token", "sep_token", "pad_token",
                      "cls_token", "mask_token")


class ContextTooLong(ValueError):
    def __init__(self, message: str, needed: int, budget: int):
        super().__init__(message)
        self.needed, self.budget = needed, budget


def _json(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def template_source(assets: str) -> Optional[str]:
    """The checkpoint's chat template: tokenizer_config.json's chat_template (a
    string, or the "default" of a named list), chat_template.jinja, or
    chat_template.json — None for a base model."""
    t = _json(os.path.join(assets, "tokenizer_config.json")).get("chat_template")
    if isinstance(t, list):
        t = next((x["template"] for x in t if x.get("name") == "default"), None)
    if t:
        return t
    p = os.path.join(assets, "chat_template.jinja")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return f.read()
    return _json(os.path.join(assets, "chat_template.json")).get("chat_template") or None


def special_tokens(assets: str) -> Dict[str, str]:
    """{bos_token: "<s>", ...}: tokenizer_config.json's special tokens (strings
    or AddedToken dicts), special_tokens_map.json's where it has none."""
    out = {}
    for name in ("special_tokens_map.json", "tokenizer_config.json"):
        for k, v in _json(os.path.join(assets, name)).items():
            if k in SPECIAL_TOKEN_KEYS and v is not None:
                out[k] = v.get("content") if isinstance(v, dict) else v
    return {k: v for k, v in out.items() if isinstance(v, str)}


def _environment():
    import jinja2
    import jinja2.ext
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    class Generation(jinja2.ext.Extension):
        """{% generation %}...{% endgeneration %}: transformers marks the
        assistant's tokens with it; rendering passes the body through."""
        tags = {"generation"}

        def parse(self, parser):
            lineno = next(parser.stream).lineno
            body = parser.parse_statements(["name:endgeneration"], drop_needle=True)
            return jinja2.nodes.CallBlock(self.call_method("_body"), [], [], body).set_lineno(lineno)

        def _body(self, caller):
            return caller()

    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators,
                          sort_keys=sort_keys)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                        extensions=[Generation, jinja2.ext.loopcontrols])
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = lambda fmt: datetime.now().strftime(fmt)
    return env


class ChatTemplate:
    """A Jinja chat template with the tokenizer's special tokens."""

    def __init__(self, source: str, specials: Optional[Dict[str, str]] = None):
        self.source = source
        self.specials = dict(specials or {})
        self._compiled = None

    @classmethod
    def from_checkpoint(cls, assets: str) -> Optional["ChatTemplate"]:
        """The template of the checkpoint (or tokenizer) directory, None without one."""
        src = template_source(assets)
        return cls(src, special_tokens(assets)) if src else None

    def render(self, messages: Sequence[Dict], add_generation_prompt: bool = True, **kw) -> str:
        """apply_chat_template(messages, tokenize=False, add_generation_prompt=...)."""
        if self._compiled is None:
            self._compiled = _environment().from_string(self.source)
        return self._compiled.render(messages=list(messages), tools=None, documents=None,
                                     add_generation_prompt=add_generation_prompt,
                                     **{**self.specials, **kw})

    def turn(self, messages: Sequence[Dict], followup: Sequence[Dict]) -> str:
        """The text that opens the next turn after a generated answer to
        ``messages``: what the template writes between the answer's text and
        the generation prompt of messages + answer + followup (ChatML:
        "<|im_end|>\\n<|im_start|>user\\n...<|im_end|>\\n<|im_start|>assistant\\n")."""
        mark = "\x00ANSWER\x00"
        text = self.render(list(messages) + [{"role": "assistant", "content": mark}] + list(followup))
        if text.count(mark) != 1:
            raise ValueError("the chat template does not write the answer verbatim")
        return text.split(mark, 1)[1]


class PromptBuilder:
    """A chat template + an encoder (tokenizer.encode without added special
    tokens): render, tokenize the whole text — the ids apply_chat_template
    gives.  Renderings are cached (clients resend the whole history every
    turn; the trimming renders shorter histories)."""

    def __init__(self, template: ChatTemplate, encode: Callable[[str], List[int]],
                 cache: int = 64):
        self.template = template
        self.encode = encode
        self._cache: "OrderedDict[str, Tuple[int, ...]]" = OrderedDict()
        self._size = cache

    def _ids(self, text: str) -> Tuple[int, ...]:
        ids = self._cache.get(text)
        if ids is None:
            ids = tuple(self.encode(text))
            self._cache[text] = ids
            while len(self._cache) > self._size:
                self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(text)
        return ids

    def ids(self, messages: Sequence[Dict], add_generation_prompt: bool = True) -> List[int]:
        return list(self._ids(self.template.render(messages, add_generation_prompt)))

    def fit(self, messages: Sequence[Dict], budget: int
            ) -> Tuple[List[int], List[Dict], int]:
        """(prompt ids, kept messages, number of dropped messages) with
        len(prompt ids) <= budget; see the module docstring."""
        msgs = list(messages)
        head = msgs[:1] if msgs and msgs[0]["role"] == "system" else []
        rest = msgs[len(head):]
        start = 0
        while True:
            kept = head + rest[start:]
            ids = self.ids(kept)
            if len(ids) <= budget:
                return ids, kept, start
            if start >= len(rest) - 1:
                what = ("The system message is" if not rest else
                        "The last message is" if not head else
                        "The system message and the last message are")
                raise ContextTooLong(
                    f"{what} too long: with the chat template the prompt needs {len(ids)} tokens, "
                    f"but at most {budget} fit (the context minus the tokens reserved for the "
                    f"answer).", len(ids), budget)
            start += 1
            while start < len(rest) - 1 and rest[start]["role"] == "assistant":
                start += 1


__all__ = ("ChatTemplate", "ContextTooLong", "PromptBuilder", "SPECIAL_TOKEN_KEYS",
           "special_tokens", "template_source")
