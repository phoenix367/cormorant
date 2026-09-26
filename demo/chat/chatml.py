"""chatml.py — SmolLM2's chat template (ChatML) and history trimming
(standard library only).

The template is the `chat_template` of SmolLM2-135M-Instruct's
tokenizer_config.json, rendered exactly as transformers'
`apply_chat_template` does:

    {% for message in messages %}
      {% if loop.first and messages[0]['role'] != 'system' %}
        <|im_start|>system\\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\\n
      {% endif %}
      <|im_start|>{role}\\n{content}<|im_end|>\\n
    {% endfor %}
    {% if add_generation_prompt %}<|im_start|>assistant\\n{% endif %}

i.e. a conversation that does not start with a system message gets the
default one.  Every block starts with the special token <|im_start|>, so the
token ids of the rendered prompt are the concatenation of the ids of its
blocks — blocks are tokenized (and cached) one by one, and trimming works
on whole blocks.

Trimming (fit_messages): the prompt must leave `reserve` positions of the
context for the answer.  The system message (if the conversation starts
with one) and the last message are always kept; older messages are dropped
from the front, and an assistant message left at the front is dropped with
its user turn, so the kept history starts with a user message.  If the
system message and the last message alone do not fit -> ContextTooLong.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Sequence, Tuple

DEFAULT_SYSTEM = "You are a helpful AI assistant named SmolLM, trained by Hugging Face"
IM_START, IM_END = "<|im_start|>", "<|im_end|>"
GENERATION_PROMPT = IM_START + "assistant\n"


class ContextTooLong(ValueError):
    def __init__(self, message: str, needed: int, budget: int):
        super().__init__(message)
        self.needed, self.budget = needed, budget


def block(role: str, content: str) -> str:
    return IM_START + role + "\n" + content + IM_END + "\n"


def blocks(messages: Sequence[Dict[str, str]]) -> List[str]:
    """The template's text, one string per rendered block (the default
    system block first when the conversation has no system message)."""
    out = []
    for i, m in enumerate(messages):
        if i == 0 and m["role"] != "system":
            out.append(block("system", DEFAULT_SYSTEM))
        out.append(block(m["role"], m["content"]))
    return out


def render(messages: Sequence[Dict[str, str]], add_generation_prompt: bool = True) -> str:
    """apply_chat_template(messages, tokenize=False, add_generation_prompt=...)."""
    return "".join(blocks(messages)) + (GENERATION_PROMPT if add_generation_prompt else "")


class PromptBuilder:
    """Template + tokenizer with a per-block token cache (clients resend the
    whole history every turn; only new blocks are tokenized)."""

    def __init__(self, encode: Callable[[str], List[int]], cache_blocks: int = 512):
        self.encode = encode
        self._cache: "OrderedDict[str, Tuple[int, ...]]" = OrderedDict()
        self._size = cache_blocks
        self.gen_ids = tuple(encode(GENERATION_PROMPT))

    def block_ids(self, text: str) -> Tuple[int, ...]:
        ids = self._cache.get(text)
        if ids is None:
            ids = tuple(self.encode(text))
            self._cache[text] = ids
            while len(self._cache) > self._size:
                self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(text)
        return ids

    def ids(self, messages: Sequence[Dict[str, str]], add_generation_prompt: bool = True) -> List[int]:
        out: List[int] = []
        for b in blocks(messages):
            out.extend(self.block_ids(b))
        if add_generation_prompt:
            out.extend(self.gen_ids)
        return out

    def fit(self, messages: Sequence[Dict[str, str]], budget: int
            ) -> Tuple[List[int], List[Dict[str, str]], int]:
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
