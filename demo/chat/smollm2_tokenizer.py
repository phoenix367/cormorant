"""smollm2_tokenizer.py — SmolLM2's byte-level BPE tokenizer in pure Python
(standard library only; runs on the board's Python 3.8+).

Reads the Hugging Face `tokenizer.json` of SmolLM2 (GPT-2-style byte-level
BPE, 49 152 tokens, 48 900 merges) and reproduces what the `tokenizers`
library does with it, token for token:

  1. added tokens      the 17 special tokens (<|endoftext|>, <|im_start|>,
                       <|im_end|>, <repo_name>, ...) are matched literally
                       anywhere in the text (leftmost, longest) and never split;
                       the text between them is tokenized as below
  2. Digits            every numeric character (Unicode N*: Nd, Nl, No) is its
                       own piece ("individual_digits": "2024" -> 2 0 2 4, and a
                       space before a digit stays with the text before it)
  3. ByteLevel regex   each piece is split with GPT-2's pattern
                       's|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+
                       (\\s = Unicode White_Space, as Oniguruma has it; \\p{L} /
                       \\p{N} from the Unicode 15.0 tables at the end of this
                       file, so the board's Python 3.10 (Unicode 13) splits
                       exactly like the host)
  4. bytes -> symbols  UTF-8 bytes mapped to GPT-2's printable byte alphabet
                       (bytes_to_unicode); symbols missing from the vocabulary
                       (21 control / never-in-UTF-8 bytes) are dropped, like
                       `tokenizers` does without an unk token
  5. BPE               merges by rank (lowest first, leftmost on ties)

decode() maps tokens back to bytes and decodes UTF-8 with U+FFFD for invalid
sequences (like Rust's from_utf8_lossy), special tokens skipped by default.
IncrementalDecoder does the same for a token stream: it holds back an
incomplete UTF-8 sequence until the tokens completing it arrive.

    tok = Tokenizer("assets/smollm2-135m-instruct/tokenizer.json")
    ids = tok.encode("<|im_start|>user\\nHi!<|im_end|>\\n")
    tok.decode(ids)                          # 'user\\nHi!\\n'
    dec = tok.incremental_decoder()
    for i in ids: print(dec.add(i), end="")
    print(dec.flush())

Validated against transformers (`scripts/validate_text.py`): ids and
round trips identical on the validation set.
"""

from __future__ import annotations

import codecs
import json
import re
from collections import OrderedDict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = ["Tokenizer", "IncrementalDecoder", "bytes_to_unicode"]


def bytes_to_unicode() -> Dict[int, str]:
    """GPT-2's byte -> printable character map (the ByteLevel alphabet)."""
    bs = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1))
          + list(range(ord("\xae"), ord("\xff") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


# Oniguruma's \s (Unicode White_Space): what `tokenizers` splits on
_WHITE_SPACE = ("\t-\r", " ", "\x85", "\xa0", " ", " - ", " - ",
                " ", " ", "　")


def _class_body(ranges: str) -> str:
    """'41-5a 61-7a aa' (hex code point ranges) -> a regex character-class body."""
    out = []
    for r in ranges.split():
        a, _, b = r.partition("-")
        lo = chr(int(a, 16))
        out.append(re.escape(lo) if not b else re.escape(lo) + "-" + re.escape(chr(int(b, 16))))
    return "".join(out)


def _build_patterns() -> Tuple["re.Pattern", "re.Pattern"]:
    L = _class_body("".join(LETTER_RANGES))
    N = _class_body("".join(NUMBER_RANGES))
    S = "".join(re.escape(x) if len(x) == 1 else re.escape(x[0]) + "-" + re.escape(x[2])
                for x in _WHITE_SPACE)
    split_num = re.compile("([" + N + "])")
    pieces = re.compile(
        "'s|'t|'re|'ve|'m|'ll|'d"
        "| ?[" + L + "]+"
        "| ?[" + N + "]+"
        "| ?[^" + S + L + N + "]+"
        "|[" + S + "]+(?![^" + S + "])"
        "|[" + S + "]+")
    return split_num, pieces


class Tokenizer:
    """SmolLM2's tokenizer from its tokenizer.json (see the module docstring)."""

    def __init__(self, path: str, cache_size: int = 50_000):
        with open(path, encoding="utf-8") as f:
            spec = json.load(f)
        self._check(spec)
        model = spec["model"]
        self.vocab: Dict[str, int] = model["vocab"]
        merges = model["merges"]
        if merges and isinstance(merges[0], str):
            merges = [m.split(" ") for m in merges]
        self.ranks: Dict[Tuple[str, str], int] = {(a, b): i for i, (a, b) in enumerate(merges)}
        self.id_to_token: List[str] = [""] * (max(self.vocab.values()) + 1)
        for t, i in self.vocab.items():
            self.id_to_token[i] = t
        self.special: Dict[str, int] = {}
        for a in spec.get("added_tokens", []):
            self.special[a["content"]] = a["id"]
            if a["id"] >= len(self.id_to_token):
                self.id_to_token.extend([""] * (a["id"] + 1 - len(self.id_to_token)))
            self.id_to_token[a["id"]] = a["content"]
        self.special_ids = frozenset(self.special.values())
        self.vocab_size = len(self.id_to_token)
        b2u = bytes_to_unicode()
        u2b = {c: b for b, c in b2u.items()}
        # bytes -> byte-level string in one str.translate (latin-1 code points are the bytes)
        self._byte_table = {b: c for b, c in b2u.items()}
        self.id_bytes: List[bytes] = []
        for i, t in enumerate(self.id_to_token):
            if i in self.special_ids:
                self.id_bytes.append(t.encode("utf-8"))
            else:
                self.id_bytes.append(bytes(u2b[c] for c in t) if all(c in u2b for c in t)
                                     else t.encode("utf-8"))
        specials = sorted(self.special, key=lambda s: (-len(s), s))   # longest first
        self._special_re = re.compile("|".join(re.escape(s) for s in specials)) if specials else None
        self._split_num, self._pieces = _build_patterns()
        self._cache: "OrderedDict[str, Tuple[int, ...]]" = OrderedDict()
        self._cache_size = cache_size
        self.bos_id = self.special.get("<|im_start|>")
        self.eos_id = self.special.get("<|im_end|>")

    @staticmethod
    def _check(spec: dict) -> None:
        m = spec.get("model", {})
        pre = spec.get("pre_tokenizer") or {}
        kinds = [p.get("type") for p in pre.get("pretokenizers", [pre])]
        if (m.get("type") != "BPE" or spec.get("normalizer") is not None
                or kinds != ["Digits", "ByteLevel"]
                or not pre["pretokenizers"][0].get("individual_digits")
                or pre["pretokenizers"][1].get("add_prefix_space")
                or not pre["pretokenizers"][1].get("use_regex", True)
                or m.get("byte_fallback") or m.get("continuing_subword_prefix")
                or m.get("end_of_word_suffix") or m.get("ignore_merges")
                or (spec.get("decoder") or {}).get("type") != "ByteLevel"):
            raise ValueError("unsupported tokenizer.json: this is SmolLM2's pipeline only "
                             "(no normalizer; Digits(individual) + ByteLevel(no prefix space); "
                             "plain byte-level BPE)")

    # ── encoding ──

    def _bpe(self, word: str) -> Tuple[int, ...]:
        """Byte-level word -> token ids (merges by rank, leftmost first)."""
        hit = self._cache.get(word)
        if hit is not None:
            return hit
        vocab, ranks = self.vocab, self.ranks
        parts = [c for c in word if c in vocab]            # unknown symbols are dropped
        while len(parts) > 1:
            best, best_rank = -1, 1 << 62
            for i in range(len(parts) - 1):
                r = ranks.get((parts[i], parts[i + 1]))
                if r is not None and r < best_rank:
                    best, best_rank = i, r
            if best < 0:
                break
            a, b = parts[best], parts[best + 1]
            merged, i, n = [], 0, len(parts)
            while i < n:
                if i < n - 1 and parts[i] == a and parts[i + 1] == b:
                    merged.append(a + b)
                    i += 2
                else:
                    merged.append(parts[i])
                    i += 1
            parts = merged
        ids = tuple(vocab[p] for p in parts)
        if self._cache_size:
            self._cache[word] = ids
            if len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return ids

    def _encode_plain(self, text: str, out: List[int]) -> None:
        """Text without special tokens -> ids appended to out."""
        table, bpe = self._byte_table, self._bpe
        for k, part in enumerate(self._split_num.split(text)):
            if not part:
                continue
            if k & 1:                                      # one numeric character
                words = (part,)
            else:
                words = self._pieces.findall(part)
            for w in words:
                out.extend(bpe(w.encode("utf-8").decode("latin-1").translate(table)))

    def encode(self, text: str, special: bool = True) -> List[int]:
        """Token ids of text.  special=True (the transformers default):
        special-token strings in the text become their ids; False: they are
        tokenized as ordinary text."""
        out: List[int] = []
        if not special or self._special_re is None:
            self._encode_plain(text, out)
            return out
        pos = 0
        for m in self._special_re.finditer(text):
            if m.start() > pos:
                self._encode_plain(text[pos:m.start()], out)
            out.append(self.special[m.group()])
            pos = m.end()
        if pos < len(text):
            self._encode_plain(text[pos:], out)
        return out

    # ── decoding ──

    def token_bytes(self, i: int) -> bytes:
        return self.id_bytes[i]

    def is_special(self, i: int) -> bool:
        return i in self.special_ids

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        sp = self.special_ids if skip_special_tokens else frozenset()
        return b"".join(self.id_bytes[i] for i in ids if i not in sp).decode("utf-8", "replace")

    def incremental_decoder(self, skip_special_tokens: bool = True) -> "IncrementalDecoder":
        return IncrementalDecoder(self, skip_special_tokens)


class IncrementalDecoder:
    """Streaming decode: add(id) returns the text that is complete so far
    (an incomplete UTF-8 sequence at the end is held back); flush() returns
    the rest (U+FFFD for a sequence that never completed).  The pieces
    concatenate to Tokenizer.decode() of the same ids."""

    def __init__(self, tok: Tokenizer, skip_special_tokens: bool = True):
        self.tok = tok
        self.skip = tok.special_ids if skip_special_tokens else frozenset()
        self._dec = codecs.getincrementaldecoder("utf-8")("replace")

    def add(self, i: int) -> str:
        if i in self.skip:
            return ""
        return self._dec.decode(self.tok.id_bytes[i], False)

    def add_many(self, ids: Sequence[int]) -> str:
        return "".join(self.add(i) for i in ids)

    def flush(self) -> str:
        return self._dec.decode(b"", True)

    @property
    def pending(self) -> bool:
        """True while bytes of an incomplete character are held back."""
        return bool(self._dec.getstate()[0])


# Unicode 15.0 general categories L* (letters) and N* (numbers) as hex code
# point ranges — regenerate with `python3 smollm2_tokenizer.py --unicode-tables`
# on a Python whose unicodedata is the Unicode version wanted.
LETTER_RANGES = (
    "41-5a 61-7a aa b5 ba c0-d6 d8-f6 f8-2c1 2c6-2d1 2e0-2e4 2ec 2ee 370-374 376-377 37a-37d 37f 386 "
    "388-38a 38c 38e-3a1 3a3-3f5 3f7-481 48a-52f 531-556 559 560-588 5d0-5ea 5ef-5f2 620-64a 66e-66f "
    "671-6d3 6d5 6e5-6e6 6ee-6ef 6fa-6fc 6ff 710 712-72f 74d-7a5 7b1 7ca-7ea 7f4-7f5 7fa 800-815 81a "
    "824 828 840-858 860-86a 870-887 889-88e 8a0-8c9 904-939 93d 950 958-961 971-980 985-98c 98f-990 "
    "993-9a8 9aa-9b0 9b2 9b6-9b9 9bd 9ce 9dc-9dd 9df-9e1 9f0-9f1 9fc a05-a0a a0f-a10 a13-a28 a2a-a30 "
    "a32-a33 a35-a36 a38-a39 a59-a5c a5e a72-a74 a85-a8d a8f-a91 a93-aa8 aaa-ab0 ab2-ab3 ab5-ab9 abd "
    "ad0 ae0-ae1 af9 b05-b0c b0f-b10 b13-b28 b2a-b30 b32-b33 b35-b39 b3d b5c-b5d b5f-b61 b71 b83 "
    "b85-b8a b8e-b90 b92-b95 b99-b9a b9c b9e-b9f ba3-ba4 ba8-baa bae-bb9 bd0 c05-c0c c0e-c10 c12-c28 "
    "c2a-c39 c3d c58-c5a c5d c60-c61 c80 c85-c8c c8e-c90 c92-ca8 caa-cb3 cb5-cb9 cbd cdd-cde ce0-ce1 "
    "cf1-cf2 d04-d0c d0e-d10 d12-d3a d3d d4e d54-d56 d5f-d61 d7a-d7f d85-d96 d9a-db1 db3-dbb dbd "
    "dc0-dc6 e01-e30 e32-e33 e40-e46 e81-e82 e84 e86-e8a e8c-ea3 ea5 ea7-eb0 eb2-eb3 ebd ec0-ec4 ec6 "
    "edc-edf f00 f40-f47 f49-f6c f88-f8c 1000-102a 103f 1050-1055 105a-105d 1061 1065-1066 106e-1070 "
    "1075-1081 108e 10a0-10c5 10c7 10cd 10d0-10fa 10fc-1248 124a-124d 1250-1256 1258 125a-125d "
    "1260-1288 128a-128d 1290-12b0 12b2-12b5 12b8-12be 12c0 12c2-12c5 12c8-12d6 12d8-1310 1312-1315 "
    "1318-135a 1380-138f 13a0-13f5 13f8-13fd 1401-166c 166f-167f 1681-169a 16a0-16ea 16f1-16f8 "
    "1700-1711 171f-1731 1740-1751 1760-176c 176e-1770 1780-17b3 17d7 17dc 1820-1878 1880-1884 "
    "1887-18a8 18aa 18b0-18f5 1900-191e 1950-196d 1970-1974 1980-19ab 19b0-19c9 1a00-1a16 1a20-1a54 "
    "1aa7 1b05-1b33 1b45-1b4c 1b83-1ba0 1bae-1baf 1bba-1be5 1c00-1c23 1c4d-1c4f 1c5a-1c7d 1c80-1c88 "
    "1c90-1cba 1cbd-1cbf 1ce9-1cec 1cee-1cf3 1cf5-1cf6 1cfa 1d00-1dbf 1e00-1f15 1f18-1f1d 1f20-1f45 "
    "1f48-1f4d 1f50-1f57 1f59 1f5b 1f5d 1f5f-1f7d 1f80-1fb4 1fb6-1fbc 1fbe 1fc2-1fc4 1fc6-1fcc "
    "1fd0-1fd3 1fd6-1fdb 1fe0-1fec 1ff2-1ff4 1ff6-1ffc 2071 207f 2090-209c 2102 2107 210a-2113 2115 "
    "2119-211d 2124 2126 2128 212a-212d 212f-2139 213c-213f 2145-2149 214e 2183-2184 2c00-2ce4 "
    "2ceb-2cee 2cf2-2cf3 2d00-2d25 2d27 2d2d 2d30-2d67 2d6f 2d80-2d96 2da0-2da6 2da8-2dae 2db0-2db6 "
    "2db8-2dbe 2dc0-2dc6 2dc8-2dce 2dd0-2dd6 2dd8-2dde 2e2f 3005-3006 3031-3035 303b-303c 3041-3096 "
    "309d-309f 30a1-30fa 30fc-30ff 3105-312f 3131-318e 31a0-31bf 31f0-31ff 3400-4dbf 4e00-a48c "
    "a4d0-a4fd a500-a60c a610-a61f a62a-a62b a640-a66e a67f-a69d a6a0-a6e5 a717-a71f a722-a788 "
    "a78b-a7ca a7d0-a7d1 a7d3 a7d5-a7d9 a7f2-a801 a803-a805 a807-a80a a80c-a822 a840-a873 a882-a8b3 "
    "a8f2-a8f7 a8fb a8fd-a8fe a90a-a925 a930-a946 a960-a97c a984-a9b2 a9cf a9e0-a9e4 a9e6-a9ef "
    "a9fa-a9fe aa00-aa28 aa40-aa42 aa44-aa4b aa60-aa76 aa7a aa7e-aaaf aab1 aab5-aab6 aab9-aabd aac0 "
    "aac2 aadb-aadd aae0-aaea aaf2-aaf4 ab01-ab06 ab09-ab0e ab11-ab16 ab20-ab26 ab28-ab2e ab30-ab5a "
    "ab5c-ab69 ab70-abe2 ac00-d7a3 d7b0-d7c6 d7cb-d7fb f900-fa6d fa70-fad9 fb00-fb06 fb13-fb17 fb1d "
    "fb1f-fb28 fb2a-fb36 fb38-fb3c fb3e fb40-fb41 fb43-fb44 fb46-fbb1 fbd3-fd3d fd50-fd8f fd92-fdc7 "
    "fdf0-fdfb fe70-fe74 fe76-fefc ff21-ff3a ff41-ff5a ff66-ffbe ffc2-ffc7 ffca-ffcf ffd2-ffd7 ffda- "
    "ffdc 10000-1000b 1000d-10026 10028-1003a 1003c-1003d 1003f-1004d 10050-1005d 10080-100fa "
    "10280-1029c 102a0-102d0 10300-1031f 1032d-10340 10342-10349 10350-10375 10380-1039d 103a0-103c3 "
    "103c8-103cf 10400-1049d 104b0-104d3 104d8-104fb 10500-10527 10530-10563 10570-1057a 1057c-1058a "
    "1058c-10592 10594-10595 10597-105a1 105a3-105b1 105b3-105b9 105bb-105bc 10600-10736 10740-10755 "
    "10760-10767 10780-10785 10787-107b0 107b2-107ba 10800-10805 10808 1080a-10835 10837-10838 1083c "
    "1083f-10855 10860-10876 10880-1089e 108e0-108f2 108f4-108f5 10900-10915 10920-10939 10980-109b7 "
    "109be-109bf 10a00 10a10-10a13 10a15-10a17 10a19-10a35 10a60-10a7c 10a80-10a9c 10ac0-10ac7 "
    "10ac9-10ae4 10b00-10b35 10b40-10b55 10b60-10b72 10b80-10b91 10c00-10c48 10c80-10cb2 10cc0-10cf2 "
    "10d00-10d23 10e80-10ea9 10eb0-10eb1 10f00-10f1c 10f27 10f30-10f45 10f70-10f81 10fb0-10fc4 "
    "10fe0-10ff6 11003-11037 11071-11072 11075 11083-110af 110d0-110e8 11103-11126 11144 11147 "
    "11150-11172 11176 11183-111b2 111c1-111c4 111da 111dc 11200-11211 11213-1122b 1123f-11240 "
    "11280-11286 11288 1128a-1128d 1128f-1129d 1129f-112a8 112b0-112de 11305-1130c 1130f-11310 "
    "11313-11328 1132a-11330 11332-11333 11335-11339 1133d 11350 1135d-11361 11400-11434 11447-1144a "
    "1145f-11461 11480-114af 114c4-114c5 114c7 11580-115ae 115d8-115db 11600-1162f 11644 11680-116aa "
    "116b8 11700-1171a 11740-11746 11800-1182b 118a0-118df 118ff-11906 11909 1190c-11913 11915-11916 "
    "11918-1192f 1193f 11941 119a0-119a7 119aa-119d0 119e1 119e3 11a00 11a0b-11a32 11a3a 11a50 "
    "11a5c-11a89 11a9d 11ab0-11af8 11c00-11c08 11c0a-11c2e 11c40 11c72-11c8f 11d00-11d06 11d08-11d09 "
    "11d0b-11d30 11d46 11d60-11d65 11d67-11d68 11d6a-11d89 11d98 11ee0-11ef2 11f02 11f04-11f10 "
    "11f12-11f33 11fb0 12000-12399 12480-12543 12f90-12ff0 13000-1342f 13441-13446 14400-14646 "
    "16800-16a38 16a40-16a5e 16a70-16abe 16ad0-16aed 16b00-16b2f 16b40-16b43 16b63-16b77 16b7d-16b8f "
    "16e40-16e7f 16f00-16f4a 16f50 16f93-16f9f 16fe0-16fe1 16fe3 17000-187f7 18800-18cd5 18d00-18d08 "
    "1aff0-1aff3 1aff5-1affb 1affd-1affe 1b000-1b122 1b132 1b150-1b152 1b155 1b164-1b167 1b170-1b2fb "
    "1bc00-1bc6a 1bc70-1bc7c 1bc80-1bc88 1bc90-1bc99 1d400-1d454 1d456-1d49c 1d49e-1d49f 1d4a2 "
    "1d4a5-1d4a6 1d4a9-1d4ac 1d4ae-1d4b9 1d4bb 1d4bd-1d4c3 1d4c5-1d505 1d507-1d50a 1d50d-1d514 "
    "1d516-1d51c 1d51e-1d539 1d53b-1d53e 1d540-1d544 1d546 1d54a-1d550 1d552-1d6a5 1d6a8-1d6c0 "
    "1d6c2-1d6da 1d6dc-1d6fa 1d6fc-1d714 1d716-1d734 1d736-1d74e 1d750-1d76e 1d770-1d788 1d78a-1d7a8 "
    "1d7aa-1d7c2 1d7c4-1d7cb 1df00-1df1e 1df25-1df2a 1e030-1e06d 1e100-1e12c 1e137-1e13d 1e14e "
    "1e290-1e2ad 1e2c0-1e2eb 1e4d0-1e4eb 1e7e0-1e7e6 1e7e8-1e7eb 1e7ed-1e7ee 1e7f0-1e7fe 1e800-1e8c4 "
    "1e900-1e943 1e94b 1ee00-1ee03 1ee05-1ee1f 1ee21-1ee22 1ee24 1ee27 1ee29-1ee32 1ee34-1ee37 1ee39 "
    "1ee3b 1ee42 1ee47 1ee49 1ee4b 1ee4d-1ee4f 1ee51-1ee52 1ee54 1ee57 1ee59 1ee5b 1ee5d 1ee5f "
    "1ee61-1ee62 1ee64 1ee67-1ee6a 1ee6c-1ee72 1ee74-1ee77 1ee79-1ee7c 1ee7e 1ee80-1ee89 1ee8b-1ee9b "
    "1eea1-1eea3 1eea5-1eea9 1eeab-1eebb 20000-2a6df 2a700-2b739 2b740-2b81d 2b820-2cea1 2ceb0-2ebe0 "
    "2f800-2fa1d 30000-3134a 31350-323af "
)
NUMBER_RANGES = (
    "30-39 b2-b3 b9 bc-be 660-669 6f0-6f9 7c0-7c9 966-96f 9e6-9ef 9f4-9f9 a66-a6f ae6-aef b66-b6f "
    "b72-b77 be6-bf2 c66-c6f c78-c7e ce6-cef d58-d5e d66-d78 de6-def e50-e59 ed0-ed9 f20-f33 "
    "1040-1049 1090-1099 1369-137c 16ee-16f0 17e0-17e9 17f0-17f9 1810-1819 1946-194f 19d0-19da "
    "1a80-1a89 1a90-1a99 1b50-1b59 1bb0-1bb9 1c40-1c49 1c50-1c59 2070 2074-2079 2080-2089 2150-2182 "
    "2185-2189 2460-249b 24ea-24ff 2776-2793 2cfd 3007 3021-3029 3038-303a 3192-3195 3220-3229 "
    "3248-324f 3251-325f 3280-3289 32b1-32bf a620-a629 a6e6-a6ef a830-a835 a8d0-a8d9 a900-a909 "
    "a9d0-a9d9 a9f0-a9f9 aa50-aa59 abf0-abf9 ff10-ff19 10107-10133 10140-10178 1018a-1018b "
    "102e1-102fb 10320-10323 10341 1034a 103d1-103d5 104a0-104a9 10858-1085f 10879-1087f 108a7-108af "
    "108fb-108ff 10916-1091b 109bc-109bd 109c0-109cf 109d2-109ff 10a40-10a48 10a7d-10a7e 10a9d-10a9f "
    "10aeb-10aef 10b58-10b5f 10b78-10b7f 10ba9-10baf 10cfa-10cff 10d30-10d39 10e60-10e7e 10f1d-10f26 "
    "10f51-10f54 10fc5-10fcb 11052-1106f 110f0-110f9 11136-1113f 111d0-111d9 111e1-111f4 112f0-112f9 "
    "11450-11459 114d0-114d9 11650-11659 116c0-116c9 11730-1173b 118e0-118f2 11950-11959 11c50-11c6c "
    "11d50-11d59 11da0-11da9 11f50-11f59 11fc0-11fd4 12400-1246e 16a60-16a69 16ac0-16ac9 16b50-16b59 "
    "16b5b-16b61 16e80-16e96 1d2c0-1d2d3 1d2e0-1d2f3 1d360-1d378 1d7ce-1d7ff 1e140-1e149 1e2f0-1e2f9 "
    "1e4f0-1e4f9 1e8c7-1e8cf 1e950-1e959 1ec71-1ecab 1ecad-1ecaf 1ecb1-1ecb4 1ed01-1ed2d 1ed2f-1ed3d "
    "1f100-1f10c 1fbf0-1fbf9 "
)


def _unicode_tables() -> str:
    import textwrap
    import unicodedata

    def ranges(major: str) -> str:
        out, start = [], None
        for cp in range(0x110000):
            if unicodedata.category(chr(cp))[0] == major:
                if start is None:
                    start = cp
            elif start is not None:
                out.append("%x" % start if start == cp - 1 else "%x-%x" % (start, cp - 1))
                start = None
        return " ".join(out)

    lines = [f"# unicodedata {unicodedata.unidata_version}"]
    for name, major in (("LETTER_RANGES", "L"), ("NUMBER_RANGES", "N")):
        lines.append(f"{name} = (")
        lines += [f'    "{w} "' for w in textwrap.wrap(ranges(major), 96)]
        lines.append(")")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["--unicode-tables"]:
        print(_unicode_tables())
    elif len(sys.argv) >= 3:
        t = Tokenizer(sys.argv[1])
        ids = t.encode(" ".join(sys.argv[2:]))
        print(ids)
        print([t.id_to_token[i] for i in ids])
        print(repr(t.decode(ids)))
    else:
        print("usage: smollm2_tokenizer.py tokenizer.json TEXT...  |  --unicode-tables", file=sys.stderr)
        sys.exit(2)
