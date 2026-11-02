# Copyright 2026 Ivan Gubochkin
# SPDX-License-Identifier: Apache-2.0

"""llama_tokenizer.py — the Llama-2 SentencePiece tokenizer (stories15m) under
the name the stories backend imports: hf_tokenizer.py, which interprets any
tokenizer.json (here BPE with byte fallback, a Prepend("▁") + Replace(" ", "▁")
normalizer, normalized added tokens matched as "▁<s>", <s> added by the
post-processor, the Replace / ByteFallback / Fuse / Strip decoder), token for
token as the `tokenizers` library.

encode(text, special=True, add_special_tokens=True): <s> first unless
add_special_tokens=False; decode(..., strip_first=False) and
incremental_decoder(strip_first=False) decode a continuation (the leading
space of its first word kept).  bos_id / eos_id are <s> / </s>.

    tok = Tokenizer("assets/stories15m/tokenizer.json")
    ids = tok.encode("Once upon a time")         # [1, 9038, 2501, 263, 931]
    tok.decode(ids)                              # 'Once upon a time'
"""

from __future__ import annotations

from hf_tokenizer import SPACE, IncrementalDecoder
from hf_tokenizer import Tokenizer as _Tokenizer

__all__ = ["Tokenizer", "IncrementalDecoder", "SPACE"]


class Tokenizer(_Tokenizer):
    """hf_tokenizer.Tokenizer with <s> / </s> as the BOS / EOS fallback."""

    def __init__(self, path: str, cache_size: int = 50_000):
        super().__init__(path, cache_size)
        if self.bos_id is None:
            self.bos_id = self.special.get("<s>")
        if self.eos_id is None:
            self.eos_id = self.special.get("</s>")
