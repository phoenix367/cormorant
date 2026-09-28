"""idefics3.py — SmolVLM's chat template (Idefics3) with images, and history
trimming (standard library only; doc/plans/CHAT_PLAN.md §23).

The template is the `chat_template` of SmolVLM-256M-Instruct's
chat_template.json, rendered as transformers' Idefics3Processor does with
image splitting off (one 512 x 512 tile per image):

    <|im_start|>{% for message in messages %}
      {{ role | capitalize }}{{ ':' if the first part is an image else ': ' }}
      {% for part in message.content %}{{ text }} or <image>{% endfor %}<end_of_utterance>\\n
    {% endfor %}{% if add_generation_prompt %}Assistant:{% endif %}

and each <image> expanded to
<fake_token_around_image><global-img><image> x 64<fake_token_around_image>.
A string content is one text part.  There is no default system message.
Every block after the leading <|im_start|> starts with a role word after
the previous block's "\\n" (a pre-token boundary), so the prompt's ids are
the concatenation of the blocks' ids — tokenized and cached one by one;
image content does not enter a block's text (the caller keeps the images).

Trimming (fit): as chatml.py — the system message (if first) and the last
message stay, older messages go from the front (an assistant message left
at the front goes with its user turn) until the prompt fits the budget;
images inside dropped messages go with them.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Callable, Dict, List, Sequence, Tuple

from chatml import ContextTooLong

IM_START = "<|im_start|>"
END_OF_UTTERANCE = "<end_of_utterance>"
FAKE_IMAGE = "<fake_token_around_image>"
GLOBAL_IMAGE = "<global-img>"
IMAGE = "<image>"
IMAGE_SEQ = 64
GENERATION_PROMPT = "Assistant:"


def image_text(seq: int = IMAGE_SEQ) -> str:
    """An image's tokens in the prompt text (one tile, no splitting)."""
    return FAKE_IMAGE + GLOBAL_IMAGE + IMAGE * seq + FAKE_IMAGE


def parts(m: Dict[str, Any]) -> List[Dict[str, Any]]:
    """A message's parts: [{"type": "text", "text"} | {"type": "image", ...}]."""
    return m.get("parts") or [{"type": "text", "text": m.get("content", "")}]


def block(m: Dict[str, Any], seq: int = IMAGE_SEQ) -> str:
    ps = parts(m)
    sep = ":" if ps and ps[0]["type"] == "image" else ": "
    body = "".join(image_text(seq) if p["type"] == "image" else p.get("text", "") for p in ps)
    return m["role"].capitalize() + sep + body + END_OF_UTTERANCE + "\n"


def render(messages: Sequence[Dict[str, Any]], add_generation_prompt: bool = True,
           seq: int = IMAGE_SEQ) -> str:
    return (IM_START + "".join(block(m, seq) for m in messages)
            + (GENERATION_PROMPT if add_generation_prompt else ""))


def images_of(messages: Sequence[Dict[str, Any]]) -> List[Any]:
    """The image parts' payloads ("image"), in prompt order."""
    return [p.get("image") for m in messages for p in parts(m) if p["type"] == "image"]


class PromptBuilder:
    """Template + tokenizer with a per-block token cache."""

    def __init__(self, encode: Callable[[str], List[int]], seq: int = IMAGE_SEQ,
                 cache_blocks: int = 512):
        self.encode = encode
        self.seq = seq
        self._cache: "OrderedDict[str, Tuple[int, ...]]" = OrderedDict()
        self._size = cache_blocks
        self.start_ids = tuple(encode(IM_START))
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

    def ids(self, messages: Sequence[Dict[str, Any]], add_generation_prompt: bool = True) -> List[int]:
        out: List[int] = list(self.start_ids)
        for m in messages:
            out.extend(self.block_ids(block(m, self.seq)))
        if add_generation_prompt:
            out.extend(self.gen_ids)
        return out

    def fit(self, messages: Sequence[Dict[str, Any]], budget: int):
        """(prompt ids, kept messages, number of dropped messages) with
        len(prompt ids) <= budget."""
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
                    f"{what} too long: with the chat template (an image takes {self.seq + 3} "
                    f"tokens) the prompt needs {len(ids)} tokens, but at most {budget} fit (the "
                    f"context minus the tokens reserved for the answer).", len(ids), budget)
            start += 1
            while start < len(rest) - 1 and rest[start]["role"] == "assistant":
                start += 1


__all__ = ("IM_START", "END_OF_UTTERANCE", "IMAGE", "IMAGE_SEQ", "GENERATION_PROMPT",
           "image_text", "parts", "block", "render", "images_of", "PromptBuilder")
