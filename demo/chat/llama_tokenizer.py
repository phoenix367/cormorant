# Copyright 2026 Ivan Gubochkin
# SPDX-License-Identifier: Apache-2.0

"""llama_tokenizer.py — the Llama-2 SentencePiece tokenizer of stories15m in
pure Python (standard library only; runs on the board's Python 3.8+).

Reads a Hugging Face `tokenizer.json` of the Llama-2 kind (BPE with byte
fallback, 32 000 tokens, a Prepend("▁") + Replace(" ", "▁") normalizer, no
pre-tokenizer) and reproduces what the `tokenizers` library does with it,
token for token:

  1. normalize         "▁" prepended to the (non-empty) text, every space " "
                       replaced by "▁"
  2. added tokens      <unk>, <s>, </s> are normalized added tokens: their
                       normalized forms "▁<unk>", "▁<s>", "▁</s>" are matched in
                       the normalized text (leftmost, longest; so "<s>" is one
                       only after a space or at the start, and takes the "▁"
                       with it) and never split; the text between them is one
                       BPE word each (there is no pre-tokenizer)
  3. symbols           every character in the vocabulary is a symbol; any other
                       becomes its UTF-8 bytes as <0xXX> tokens (byte fallback)
  4. BPE               merges by rank, lowest first, leftmost on ties — with a
                       heap, as `tokenizers` merges a word (a whole prompt is
                       one word here)
  5. post-processor    <s> in front (add_special_tokens, the default)

decode() is the decoder chain: "▁" -> " ", runs of <0xXX> tokens -> their
bytes as UTF-8 (a run that is not valid UTF-8: one U+FFFD per byte), fused,
one leading space stripped; special tokens skipped by default (`tokenizers`
0.23 keeps them, as " </s>": their normalized form — so does
skip_special_tokens=False here).
IncrementalDecoder does the same for a token stream (a byte run is held back
until the next other token); strip_first=False decodes a continuation (the
text after a prompt keeps its leading space).

    tok = Tokenizer("assets/stories15m/tokenizer.json")
    ids = tok.encode("Once upon a time")         # [1, 9038, 2501, 263, 931]
    tok.decode(ids)                              # 'Once upon a time'

Validated against `tokenizers` (`scripts/validate_text.py --model stories15m`):
ids and decoding identical on the validation set.
"""

from __future__ import annotations

import heapq
import json
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = ["Tokenizer", "IncrementalDecoder"]

SPACE = "▁"                                       # "▁"
_BYTE_TOKEN = re.compile(r"<0x([0-9A-Fa-f]{2})>")


class Tokenizer:
    """A Llama-2 tokenizer.json (see the module docstring)."""

    def __init__(self, path: str):
        with open(path, encoding="utf-8") as f:
            spec = json.load(f)
        self._check(spec)
        model = spec["model"]
        self.vocab: Dict[str, int] = model["vocab"]
        self.id_to_token: List[str] = [""] * (max(self.vocab.values()) + 1)
        for t, i in self.vocab.items():
            self.id_to_token[i] = t
        self.special: Dict[str, int] = {}
        normalized: Dict[str, int] = {}
        for a in spec.get("added_tokens", []):
            self.special[a["content"]] = a["id"]
            if a["id"] >= len(self.id_to_token):
                self.id_to_token.extend([""] * (a["id"] + 1 - len(self.id_to_token)))
            # an added token is matched / decoded in its normalized form, "▁<s>"
            normalized[SPACE + a["content"].replace(" ", SPACE)] = a["id"]
            self.id_to_token[a["id"]] = SPACE + a["content"].replace(" ", SPACE)
        self.special_ids = frozenset(self.special.values())
        self.vocab_size = len(self.id_to_token)
        merges = model["merges"]
        if merges and isinstance(merges[0], str):
            merges = [m.split(" ") for m in merges]
        # (left id, right id) -> (rank, merged id)
        self.merges: Dict[Tuple[int, int], Tuple[int, int]] = {
            (self.vocab[a], self.vocab[b]): (r, self.vocab[a + b]) for r, (a, b) in enumerate(merges)}
        self._byte_id = [self.vocab["<0x%02X>" % b] for b in range(256)]
        self._byte_of: Dict[int, int] = {i: b for b, i in enumerate(self._byte_id)}
        self.id_bytes: List[bytes] = []
        for i, t in enumerate(self.id_to_token):
            b = self._byte_of.get(i)
            self.id_bytes.append(bytes([b]) if b is not None and i not in self.special_ids
                                 else t.replace(SPACE, " ").encode("utf-8"))
        self._normalized = normalized
        specials = sorted(normalized, key=lambda s: (-len(s), s))     # longest first
        self._special_re = re.compile("|".join(re.escape(s) for s in specials)) if specials else None
        self.bos_id = self.special.get("<s>")
        self.eos_id = self.special.get("</s>")

    @staticmethod
    def _check(spec: dict) -> None:
        m = spec.get("model", {})
        norm = spec.get("normalizer") or {}
        nl = [(n.get("type"), n.get("prepend"), (n.get("pattern") or {}).get("String"), n.get("content"))
              for n in norm.get("normalizers", [])]
        dec = spec.get("decoder") or {}
        dl = [d.get("type") for d in dec.get("decoders", [])]
        post = spec.get("post_processor") or {}
        single = [next(iter(x)) for x in post.get("single", [])]
        if (m.get("type") != "BPE" or not m.get("byte_fallback") or m.get("dropout")
                or m.get("continuing_subword_prefix") or m.get("end_of_word_suffix")
                or m.get("ignore_merges")
                or nl != [("Prepend", SPACE, None, None), ("Replace", None, " ", SPACE)]
                or spec.get("pre_tokenizer") is not None
                or dl != ["Replace", "ByteFallback", "Fuse", "Strip"]
                or post.get("type") != "TemplateProcessing" or single != ["SpecialToken", "Sequence"]
                or any(not a.get("normalized", True) or a.get("lstrip") or a.get("rstrip")
                       or a.get("single_word") for a in spec.get("added_tokens", []))):
            raise ValueError("unsupported tokenizer.json: this is the Llama-2 SentencePiece pipeline "
                             "only (Prepend + Replace normalizer, no pre-tokenizer, BPE with byte "
                             "fallback, <s> template, Replace / ByteFallback / Fuse / Strip decoder)")

    # ── encoding ──

    def _bpe(self, word: str, out: List[int]) -> None:
        """One word -> token ids appended to out (tokenizers' Word::merge_all)."""
        vocab, byte_id = self.vocab, self._byte_id
        sym: List[int] = []
        for ch in word:
            i = vocab.get(ch)
            if i is not None:
                sym.append(i)
            else:
                sym.extend(byte_id[b] for b in ch.encode("utf-8"))
        n = len(sym)
        if n < 2:
            out.extend(sym)
            return
        merges = self.merges
        nxt = list(range(1, n + 1))
        prv = list(range(-1, n - 1))
        alive = [True] * n
        heap = []
        for p in range(n - 1):
            m = merges.get((sym[p], sym[p + 1]))
            if m is not None:
                heap.append((m[0], p, m[1]))
        heapq.heapify(heap)
        while heap:
            _, p, new = heapq.heappop(heap)
            if not alive[p] or nxt[p] >= n:
                continue
            q = nxt[p]
            m = merges.get((sym[p], sym[q]))
            if m is None or m[1] != new:                 # an expired entry
                continue
            sym[p] = new
            alive[q] = False
            nxt[p] = nxt[q]
            if nxt[q] < n:
                prv[nxt[q]] = p
            if prv[p] >= 0:
                m = merges.get((sym[prv[p]], new))
                if m is not None:
                    heapq.heappush(heap, (m[0], prv[p], m[1]))
            if nxt[p] < n:
                m = merges.get((new, sym[nxt[p]]))
                if m is not None:
                    heapq.heappush(heap, (m[0], p, m[1]))
        out.extend(s for s, a in zip(sym, alive) if a)

    def encode(self, text: str, special: bool = True, add_special_tokens: bool = True) -> List[int]:
        """Token ids of text: <s> first (add_special_tokens, the `tokenizers`
        default), then the text.  special=True: special-token strings in the
        text become their ids; False: they are tokenized as ordinary text."""
        out: List[int] = [self.bos_id] if add_special_tokens else []
        if not text:
            return out
        norm = SPACE + text.replace(" ", SPACE)
        if not special or self._special_re is None:
            self._bpe(norm, out)
            return out
        pos = 0
        for m in self._special_re.finditer(norm):
            if m.start() > pos:
                self._bpe(norm[pos:m.start()], out)
            out.append(self._normalized[m.group()])
            pos = m.end()
        if pos < len(norm):
            self._bpe(norm[pos:], out)
        return out

    # ── decoding ──

    def token_bytes(self, i: int) -> bytes:
        """The token's text as bytes ("▁" as a space; a byte token its byte)."""
        return self.id_bytes[i]

    def is_special(self, i: int) -> bool:
        return i in self.special_ids

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = True,
               strip_first: bool = True) -> str:
        dec = IncrementalDecoder(self, skip_special_tokens, strip_first)
        return dec.add_many(list(ids)) + dec.flush()

    def incremental_decoder(self, skip_special_tokens: bool = True,
                            strip_first: bool = True) -> "IncrementalDecoder":
        return IncrementalDecoder(self, skip_special_tokens, strip_first)


class IncrementalDecoder:
    """Streaming decode: add(id) returns the text that is complete so far (a
    run of byte tokens is held back until a token that is not one ends it);
    flush() returns the rest.  The pieces concatenate to Tokenizer.decode() of
    the same ids.  strip_first=False: no leading space is stripped (the
    continuation of a prompt)."""

    def __init__(self, tok: Tokenizer, skip_special_tokens: bool = True, strip_first: bool = True):
        self.tok = tok
        self.skip = tok.special_ids if skip_special_tokens else frozenset()
        self._run = bytearray()
        self._started = not strip_first

    def _emit(self, s: str) -> str:
        if not self._started and s:
            self._started = True
            if s[0] == " ":
                s = s[1:]
        return s

    def _flush_run(self) -> str:
        if not self._run:
            return ""
        try:
            s = self._run.decode("utf-8")
        except UnicodeDecodeError:
            s = "�" * len(self._run)
        self._run.clear()
        return s

    def add(self, i: int) -> str:
        if i in self.skip:
            return ""
        b = self.tok._byte_of.get(i)
        if b is not None and i not in self.tok.special_ids:
            self._run.append(b)
            return ""
        return self._emit(self._flush_run() + self.tok.id_to_token[i].replace(SPACE, " "))

    def add_many(self, ids: Sequence[int]) -> str:
        return "".join(self.add(i) for i in ids)

    def flush(self) -> str:
        return self._emit(self._flush_run())

    @property
    def pending(self) -> bool:
        """True while byte tokens are held back."""
        return bool(self._run)


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        t = Tokenizer(sys.argv[1])
        ids = t.encode(" ".join(sys.argv[2:]))
        print(ids)
        print([t.id_to_token[i] for i in ids])
        print(repr(t.decode(ids)))
    else:
        print("usage: llama_tokenizer.py tokenizer.json TEXT...", file=sys.stderr)
        sys.exit(2)
