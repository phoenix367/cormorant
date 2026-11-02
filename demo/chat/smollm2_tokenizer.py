# Copyright 2026 Ivan Gubochkin
# SPDX-License-Identifier: Apache-2.0

"""smollm2_tokenizer.py — the SmolLM2 / SmolVLM tokenizer under the name their
backends import: hf_tokenizer.py, which interprets any tokenizer.json (here
SmolLM2's byte-level BPE: no normalizer, Digits(individual) + ByteLevel, 17
special tokens), token for token as the `tokenizers` library.

bos_id / eos_id are <|im_start|> / <|im_end|> (tokenizer_config.json's, and
the same when no config sits beside tokenizer.json).

    tok = Tokenizer("assets/smollm2-135m-instruct/tokenizer.json")
    ids = tok.encode("<|im_start|>user\\nHi!<|im_end|>\\n")
    tok.decode(ids)                          # 'user\\nHi!\\n'
    dec = tok.incremental_decoder()
"""

from __future__ import annotations

from hf_tokenizer import IncrementalDecoder, bytes_to_unicode
from hf_tokenizer import Tokenizer as _Tokenizer

__all__ = ["Tokenizer", "IncrementalDecoder", "bytes_to_unicode"]


class Tokenizer(_Tokenizer):
    """hf_tokenizer.Tokenizer with SmolLM2's <|im_start|> / <|im_end|> as the
    BOS / EOS fallback."""

    def __init__(self, path: str, cache_size: int = 50_000):
        super().__init__(path, cache_size)
        if self.bos_id is None:
            self.bos_id = self.special.get("<|im_start|>")
        if self.eos_id is None:
            self.eos_id = self.special.get("<|im_end|>")
