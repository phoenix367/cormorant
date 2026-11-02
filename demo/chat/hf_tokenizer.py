# Copyright 2026 Ivan Gubochkin
# SPDX-License-Identifier: Apache-2.0

"""hf_tokenizer.py — a Hugging Face tokenizer.json in pure Python (standard
library only; runs on the board's Python 3.10).

Interprets the tokenizer.json pipeline as the `tokenizers` library (0.23)
runs it, so a new checkpoint brings its tokenizer as data, not code:

  added tokens     matched first, leftmost-longest: the non-normalized ones in
                   the raw text, then the normalized ones (their content run
                   through the normalizer) in each normalized piece; flags
                   special (decode skips it), lstrip / rstrip (the match takes
                   the White_Space around it), single_word (not inside a word);
                   encode(special=False) matches only the non-special ones
  normalizer       NFC, NFD, NFKC, NFKD, Lowercase, Strip, Prepend, Replace,
                   Sequence — applied to each piece between added tokens
  pre-tokenizer    ByteLevel (add_prefix_space, use_regex: GPT-2's pattern),
                   Split (String / Regex pattern; Isolated, Removed,
                   MergedWithPrevious, MergedWithNext, Contiguous; invert),
                   Digits (individual_digits), Metaspace (prepend_scheme
                   always / first / never, split), Sequence
  model            BPE: merges as "a b" or [a, b], byte_fallback, unk_token +
                   fuse_unk, ignore_merges, continuing_subword_prefix,
                   end_of_word_suffix; merges by rank, leftmost on ties (a heap
                   for long words); per-word cache
  post-processor   TemplateProcessing (the "single" template: Llama-2's <s>),
                   ByteLevel (offsets only: nothing here), RobertaProcessing,
                   BertProcessing, Sequence — encode(add_special_tokens=True)
  decoder          ByteLevel, Replace, ByteFallback, Fuse, Strip, Metaspace,
                   BPEDecoder, Sequence (none: tokens joined with " ")

Anything else (WordPiece / Unigram / WordLevel models, Precompiled / Bert
normalizers, Whitespace / Bert pre-tokenizers, regex constructs Python's re
cannot express exactly, truncation / padding, BPE dropout) raises ValueError
naming the part.

Regex patterns (Split, Replace) are Oniguruma's; onig_to_re() rewrites them for
Python's re with explicit character classes: \\p{..} from the Unicode 16.0
tables at the end of this file (what the onig in `tokenizers` 0.23 uses),
\\s = White_Space, \\d = Nd, (?i:...) as explicit ASCII case variants (Oniguruma
folds s / k with ſ / K but not i with İ / ı, unlike Python's re).  Digits
splits on Unicode 17.0's N* (Rust's char::is_numeric).  NFC / NFKC run through
unicodedata, with the characters the reference's Unicode 9.0 normalization data
does not know kept as they are — the same on any Python version.  Lowercase is
Rust's char::to_lowercase (no final-sigma rule; Unicode 17.0).

Deliberate differences from `tokenizers` 0.23: decode(skip_special_tokens=True)
skips special tokens by id — `tokenizers` compares their decoded string, so a
normalized special token (Llama-2's "▁<s>") stays; added-token single_word
checks use letters, marks, Nd / Nl, Pc and the joiners for Rust regex's \\w.

    tok = Tokenizer("assets/smollm2-135m-instruct/tokenizer.json")
    ids = tok.encode("<|im_start|>user\\nHi!<|im_end|>\\n")
    tok.decode(ids)                          # 'user\\nHi!\\n' (special tokens skipped)
    dec = tok.incremental_decoder()
    for i in ids: print(dec.add(i), end="")
    print(dec.flush())

Validated against `tokenizers` (scripts/validate_text.py --all): ids with and
without special tokens, decode and the incremental decoder identical for the
SmolLM2, stories15m and Moonshine tokenizers and the pinned Qwen2.5, TinyLlama,
Phi-3, Llama-3.2, Gemma-3 and Mistral ones.
"""

from __future__ import annotations

import codecs
import heapq
import json
import os
import re
import unicodedata
from collections import OrderedDict
try:
    from re import _parser as _sre_parse           # Python 3.11+
except ImportError:                                # Python 3.10
    import sre_parse as _sre_parse
from typing import Any, Callable, Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple, Union

__all__ = ["Tokenizer", "IncrementalDecoder", "AddedToken", "bytes_to_unicode", "onig_to_re"]

MAX_CP = 0x10FFFF
Ranges = List[Tuple[int, int]]
SPACE = "▁"                                   # "▁", SentencePiece's space


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
    return {b: chr(c) for b, c in zip(bs, cs, strict=True)}


_B2U = bytes_to_unicode()
_BYTES_TO_CHARS = {b: c for b, c in _B2U.items()}                 # str.translate of latin-1 text
_CHARS_TO_BYTES = {ord(c): chr(b) for b, c in _B2U.items()}       # back; .encode("latin-1") then


# ── Unicode classes (the tables are at the end of the file) ───────────────────

_WHITE_SPACE: Ranges = [(0x09, 0x0D), (0x20, 0x20), (0x85, 0x85), (0xA0, 0xA0), (0x1680, 0x1680),
                        (0x2000, 0x200A), (0x2028, 0x2029), (0x202F, 0x202F), (0x205F, 0x205F),
                        (0x3000, 0x3000)]
WHITE_SPACE = "".join(chr(c) for a, b in _WHITE_SPACE for c in range(a, b + 1))

_CATEGORIES = {"L": ("Lu", "Ll", "Lt", "Lm", "Lo"), "LC": ("Lu", "Ll", "Lt"), "M": ("Mn", "Mc", "Me"),
               "N": ("Nd", "Nl", "No"), "P": ("Pc", "Pd", "Ps", "Pe", "Pi", "Pf", "Po"),
               "S": ("Sm", "Sc", "Sk", "So"), "Z": ("Zs", "Zl", "Zp"),
               "C": ("Cc", "Cf", "Cs", "Co", "Cn")}
_LONG_NAMES = {
    "letter": "L", "casedletter": "LC", "l&": "LC", "uppercaseletter": "Lu", "lowercaseletter": "Ll",
    "titlecaseletter": "Lt", "modifierletter": "Lm", "otherletter": "Lo", "mark": "M",
    "combiningmark": "M", "nonspacingmark": "Mn", "spacingmark": "Mc", "enclosingmark": "Me",
    "number": "N", "decimalnumber": "Nd", "letternumber": "Nl", "othernumber": "No",
    "punctuation": "P", "connectorpunctuation": "Pc", "dashpunctuation": "Pd",
    "openpunctuation": "Ps", "closepunctuation": "Pe", "initialpunctuation": "Pi",
    "finalpunctuation": "Pf", "otherpunctuation": "Po", "symbol": "S", "mathsymbol": "Sm",
    "currencysymbol": "Sc", "modifiersymbol": "Sk", "othersymbol": "So", "separator": "Z",
    "spaceseparator": "Zs", "lineseparator": "Zl", "paragraphseparator": "Zp", "other": "C",
    "control": "Cc", "format": "Cf", "surrogate": "Cs", "privateuse": "Co", "unassigned": "Cn"}
_gc_runs: Optional[List[Tuple[int, str]]] = None
_class_cache: Dict[Any, Ranges] = {}


def _parse_ranges(text: str) -> Ranges:
    """'41-5a 61 ...' (hex code points) -> [(0x41, 0x5a), (0x61, 0x61), ...]."""
    out = []
    for r in text.split():
        a, _, b = r.partition("-")
        out.append((int(a, 16), int(b or a, 16)))
    return out


def _merge(ranges: Iterable[Tuple[int, int]]) -> Ranges:
    out: Ranges = []
    for a, b in sorted(ranges):
        if out and a <= out[-1][1] + 1:
            if b > out[-1][1]:
                out[-1] = (out[-1][0], b)
        else:
            out.append((a, b))
    return out


def _complement(ranges: Ranges) -> Ranges:
    out, lo = [], 0
    for a, b in _merge(ranges):
        if a > lo:
            out.append((lo, a - 1))
        lo = b + 1
    if lo <= MAX_CP:
        out.append((lo, MAX_CP))
    return out


def _category_ranges(cats: Sequence[str]) -> Ranges:
    """The code points of the general categories ``cats`` (Unicode 16.0)."""
    global _gc_runs
    key = ("gc",) + tuple(sorted(cats))
    hit = _class_cache.get(key)
    if hit is not None:
        return hit
    if _gc_runs is None:
        runs, cp = [], 0
        for d, cat in re.findall(r"([0-9a-f]+)([A-Z][a-z])", "".join(GC16_RUNS)):
            cp += int(d, 16)
            runs.append((cp, cat))
        _gc_runs = runs
    want = set(cats)
    out = []
    for k, (start, cat) in enumerate(_gc_runs):
        if cat in want:
            out.append((start, _gc_runs[k + 1][0] - 1 if k + 1 < len(_gc_runs) else MAX_CP))
    _class_cache[key] = out = _merge(out)
    return out


def _property(name: str) -> Ranges:
    """\\p{name}: a general category (short or long name, Oniguruma's
    spelling rules: case, spaces, '_' and '-' ignored) or Any."""
    key = re.sub(r"[\s_-]", "", name).lower()
    if key == "any":
        return [(0, MAX_CP)]
    short = _LONG_NAMES.get(key)
    if short is None:
        short = next((c for c in list(_CATEGORIES) + [x for v in _CATEGORIES.values() for x in v]
                      if c.lower() == key), None)
    if short is None:
        raise ValueError(f"the property \\p{{{name}}} (only general categories are)")
    return _category_ranges(_CATEGORIES.get(short, (short,)))


def _numeric_ranges() -> Ranges:
    """Rust's char::is_numeric (N*, Unicode 17.0): what Digits splits on."""
    hit = _class_cache.get("numeric")
    if hit is None:
        _class_cache["numeric"] = hit = _parse_ranges("".join(NUMERIC17))
    return hit


def _word_ranges() -> Ranges:
    """Rust regex's \\w for single_word added tokens: letters, marks, Nd / Nl,
    Pc, the joiners and the Other_Alphabetic symbols (circled / squared letters)."""
    hit = _class_cache.get("word")
    if hit is None:
        hit = _merge(_category_ranges(_CATEGORIES["L"] + _CATEGORIES["M"] + ("Nd", "Nl", "Pc"))
                     + [(0x200C, 0x200D), (0x24B6, 0x24E9), (0x1F130, 0x1F149), (0x1F150, 0x1F169),
                        (0x1F170, 0x1F189)])
        _class_cache["word"] = hit
    return hit


def _esc(cp: int) -> str:
    c = chr(cp)
    return "\\" + c if c in "\\]^-[&~|" else c


def _class(ranges: Ranges, ascii_only: bool = False) -> str:
    """A character class for Python's re (never negated: complements are
    explicit ranges); ascii_only: its ASCII part (the same matches on ASCII
    text, without the scan of the non-BMP ranges every miss costs)."""
    if ascii_only:
        ranges = [(a, min(b, 0x7F)) for a, b in ranges if a <= 0x7F]
    if not ranges:
        return r"[^\x00-\U0010ffff]"                   # matches nothing
    parts = []
    for a, b in _merge(ranges):
        parts.append(_esc(a) if a == b else _esc(a) + _esc(b) if b == a + 1 else _esc(a) + "-" + _esc(b))
    return "[" + "".join(parts) + "]"


# ── Oniguruma patterns -> Python re ───────────────────────────────────────────

_SIMPLE_ESC = {"n": 0x0A, "r": 0x0D, "t": 0x09, "f": 0x0C, "v": 0x0B, "a": 0x07, "e": 0x1B}
_FOLD_EXTRA = {"s": (0x17F,), "k": (0x212A,)}                # Oniguruma: ſ ~ s, K (Kelvin) ~ k


class _Unsupported(ValueError):
    pass


def _escape(p: str, i: int) -> Tuple[str, Any, int]:
    """The escape at p[i] == "\\" -> ("char", code point | "class", ranges |
    "raw", Python re text, next index)."""
    if i + 1 >= len(p):
        raise _Unsupported("a trailing backslash")
    c = p[i + 1]
    if c in "pP":
        neg = c == "P"
        if p.startswith("{", i + 2):
            k = p.find("}", i + 3)
            if k < 0:
                raise _Unsupported("an unterminated \\p{")
            name, j = p[i + 3:k], k + 1
        else:
            name, j = p[i + 2:i + 3], i + 3
        if name.startswith("^"):
            neg, name = not neg, name[1:]
        r = _property(name)
        return "class", _complement(r) if neg else r, j
    if c in "sS":
        return "class", (_complement(_WHITE_SPACE) if c == "S" else list(_WHITE_SPACE)), i + 2
    if c in "dD":
        r = _category_ranges(("Nd",))
        return "class", (_complement(r) if c == "D" else r), i + 2
    if c in "hH":
        r = [(0x30, 0x39), (0x41, 0x46), (0x61, 0x66)]
        return "class", (_complement(r) if c == "H" else r), i + 2
    if c in "wWbB":
        raise _Unsupported(f"\\{c} (Oniguruma's word class)")
    if c == "A":
        return "raw", r"\A", i + 2
    if c == "z":
        return "raw", r"\Z", i + 2
    if c == "Z":
        return "raw", r"(?=\n?\Z)", i + 2
    if c in "123456789":
        raise _Unsupported("a back-reference")
    if c == "x":
        if p.startswith("{", i + 2):
            k = p.find("}", i + 3)
            m = re.fullmatch(r"[0-9a-fA-F]{1,8}", p[i + 3:k] if k > 0 else "")
            if not m:
                raise _Unsupported("a malformed \\x{...}")
            return "char", int(m.group(), 16), k + 1
        m = re.match(r"[0-9a-fA-F]{1,2}", p[i + 2:])
        if not m:
            raise _Unsupported("a malformed \\x")
        return "char", int(m.group(), 16), i + 2 + m.end()
    if c == "u":
        m = re.match(r"[0-9a-fA-F]{4}", p[i + 2:])
        if not m:
            raise _Unsupported("a malformed \\u")
        return "char", int(m.group(), 16), i + 6
    if c == "0":
        m = re.match(r"[0-7]{0,2}", p[i + 2:])
        return "char", int(m.group() or "0", 8), i + 2 + m.end()
    if c in _SIMPLE_ESC:
        return "char", _SIMPLE_ESC[c], i + 2
    if c.isalnum() or ord(c) > 0x7F:
        raise _Unsupported(f"the escape \\{c}")
    return "char", ord(c), i + 2


def _parse_class(p: str, i: int) -> Tuple[Ranges, int]:
    """The bracket expression at p[i] == "[" -> (code point ranges, next index)."""
    j = i + 1
    neg = p.startswith("^", j)
    if neg:
        j += 1
    items: Ranges = []
    first = True
    while True:
        if j >= len(p):
            raise _Unsupported("an unterminated '['")
        c = p[j]
        if c == "]" and not first:
            j += 1
            break
        if c == "[":
            raise _Unsupported("a nested class or a POSIX bracket in a class")
        if p.startswith("&&", j):
            raise _Unsupported("a class intersection (&&)")
        first = False
        if c == "\\":
            kind, val, j = _escape(p, j)
            if kind == "class":
                items.extend(val)
                continue
            if kind != "char":
                raise _Unsupported("an anchor in a class")
            lo = val
        else:
            lo, j = ord(c), j + 1
        if p.startswith("-", j) and j + 1 < len(p) and p[j + 1] != "]":
            j += 1
            if p[j] == "\\":
                kind, val, j = _escape(p, j)
                if kind != "char":
                    raise _Unsupported("a range ending in a class")
                hi = val
            elif p[j] == "[":
                raise _Unsupported("a nested class")
            else:
                hi, j = ord(p[j]), j + 1
            if hi < lo:
                raise _Unsupported("an empty range")
            items.append((lo, hi))
        else:
            items.append((lo, lo))
    r = _merge(items)
    return (_complement(r) if neg else r), j


def _case_variants(cp: int) -> Tuple[int, ...]:
    """The code points a case-insensitive Oniguruma match of ASCII letter cp
    accepts."""
    c = chr(cp).lower()
    return (ord(c), ord(c.upper())) + _FOLD_EXTRA.get(c, ())


def _literal(cp: int, ci: bool, ascii_only: bool = False) -> str:
    c = chr(cp)
    if ci:
        if c.isascii() and c.isalpha():
            return _class([(x, x) for x in _case_variants(cp)], ascii_only)
        if cp in (0x17F, 0x212A):
            return _literal(ord("s" if cp == 0x17F else "k"), True, ascii_only)
        if c.lower() != c.upper():
            raise _Unsupported(f"a case-insensitive non-ASCII letter ({c!r})")
    return re.escape(c)


def _ci_class(ranges: Ranges, ci: bool) -> Ranges:
    """A class inside (?i): its ASCII letters with their case variants; any
    other character that has case (letters, Nl, So's circled letters, U+0345)
    is refused."""
    if not ci:
        return ranges
    cased = _class_cache.get("cased")
    if cased is None:
        cased = _merge(_category_ranges(_CATEGORIES["L"] + ("Nl", "So")) + [(0x345, 0x345)])
        _class_cache["cased"] = cased
    for a, b in ranges:
        lo = max(a, 0x80)
        if lo <= b and any(c <= b and d >= lo for c, d in cased):
            raise _Unsupported("a character class with non-ASCII letters inside (?i)")
    extra = [(x, x) for a, b in ranges for cp in range(a, min(b, 0x7F) + 1) if chr(cp).isalpha()
             for x in _case_variants(cp)]
    return _merge(list(ranges) + extra)


def onig_to_re(pattern: str, what: str = "regex", ascii_only: bool = False) -> str:
    """An Oniguruma (Ruby syntax) pattern as `tokenizers` compiles it ->
    the same pattern for Python's re: explicit Unicode 16.0 classes, every
    group non-capturing, ^ / $ line anchors.  Constructs without an exact
    Python 3.10 equivalent (\\w, \\b, back-references, atomic groups,
    possessive quantifiers, scripts, case-insensitive non-ASCII) raise
    ValueError naming them.  ascii_only: every class cut to its ASCII part —
    the same matches on ASCII text, faster."""
    out: List[str] = []
    i, n = 0, len(pattern)
    ci = [False]                                   # case-insensitive, per group level
    try:
        while i < n:
            c = pattern[i]
            if c == "\\":
                kind, val, i = _escape(pattern, i)
                out.append(_literal(val, ci[-1], ascii_only) if kind == "char" else
                           _class(_ci_class(val, ci[-1]), ascii_only) if kind == "class" else val)
            elif c == "[":
                ranges, i = _parse_class(pattern, i)
                out.append(_class(_ci_class(ranges, ci[-1]), ascii_only))
            elif c == "(":
                if not pattern.startswith("(?", i):
                    out.append("(?:")
                    ci.append(ci[-1])
                    i += 1
                    continue
                j = i + 2
                rest = pattern[j:]
                if rest.startswith("#"):
                    k = pattern.find(")", j)
                    if k < 0:
                        raise _Unsupported("an unterminated comment")
                    i = k + 1
                elif rest[:1] in (":", "=", "!") or rest.startswith(("<=", "<!")):
                    k = 1 if rest[:1] in (":", "=", "!") else 2
                    out.append("(?" + rest[:k])
                    ci.append(ci[-1])
                    i = j + k
                elif rest.startswith(("<", "P<", "'")):
                    m = re.match(r"P?<\w+>|'\w+'", rest)
                    if not m:
                        raise _Unsupported("a malformed named group")
                    out.append("(?:")
                    ci.append(ci[-1])
                    i = j + m.end()
                elif rest.startswith(">"):
                    raise _Unsupported("an atomic group (?>...)")
                else:
                    m = re.match(r"([a-zA-Z]*)(?:-([a-zA-Z]*))?([:)])", rest)
                    if not m:
                        raise _Unsupported(f"the group {pattern[i:i + 4]!r}...")
                    on, off = m.group(1), m.group(2) or ""
                    if set(on + off) - {"i"}:
                        raise _Unsupported(f"the option (?{on}{'-' + off if off else ''}) (only i is)")
                    flag = True if "i" in on else False if "i" in off else ci[-1]
                    if m.group(3) == ":":
                        out.append("(?:")
                        ci.append(flag)
                    else:
                        ci[-1] = flag                          # to the end of the enclosing group
                    i = j + m.end()
            elif c == ")":
                if len(ci) == 1:
                    raise _Unsupported("an unbalanced ')'")
                ci.pop()
                out.append(")")
                i += 1
            elif c in "*+?{":
                if c == "{":
                    m = re.match(r"\{(\d*)(,(\d*))?\}", pattern[i:])
                    if not m or not (m.group(1) or m.group(3)):
                        out.append(r"\{")                      # a literal brace
                        i += 1
                        continue
                    out.append(m.group())
                    i += m.end()
                else:
                    out.append(c)
                    i += 1
                if pattern.startswith("+", i):
                    raise _Unsupported("a possessive quantifier")
                if pattern.startswith("?", i):
                    out.append("?")
                    i += 1
            elif c == ".":
                out.append(".")                                # not \n, as Oniguruma's
                i += 1
            elif c in "^$":
                out.append("(?m:" + c + ")")                   # Ruby syntax: line anchors
                i += 1
            elif c == "|":
                out.append("|")
                i += 1
            else:
                out.append(_literal(ord(c), ci[-1], ascii_only))
                i += 1
        if len(ci) != 1:
            raise _Unsupported("an unbalanced '('")
    except _Unsupported as e:
        raise ValueError(f"{what} {pattern!r}: {e} is not supported (hf_tokenizer.onig_to_re)") from None
    except ValueError as e:
        raise ValueError(f"{what} {pattern!r}: {e} is not supported (hf_tokenizer.onig_to_re)") from None
    return "".join(out)


GPT2_PATTERN = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
_gpt2_re: Optional["_Regex"] = None


def _gpt2() -> "_Regex":
    global _gpt2_re
    if _gpt2_re is None:
        _gpt2_re = _onig_regex(GPT2_PATTERN, "ByteLevel regex")
    return _gpt2_re


class _Regex:
    """A translated pattern with tokenizers' find_matches: the onig crate's
    iteration (an empty match right after the previous match is skipped by
    moving on one character — Python's split / sub look for a longer match at
    the same position instead); an empty text has no match.  With
    ``ascii_src`` (the pattern's ASCII cut) an ASCII text runs that."""

    def __init__(self, src: str, ascii_src: Optional[str] = None):
        self.rx, self.split_rx, self.empty = self._compile(src)
        self._ascii = self._compile(ascii_src) if ascii_src is not None else None

    @staticmethod
    def _compile(src: str) -> Tuple["re.Pattern", "re.Pattern", bool]:
        return (re.compile(src), re.compile("(" + src + ")"),       # src has no capturing groups
                _sre_parse.parse(src).getwidth()[0] == 0)

    def _pick(self, s: str) -> Tuple["re.Pattern", "re.Pattern", bool]:
        return self._ascii if self._ascii is not None and s.isascii() else (self.rx, self.split_rx, self.empty)

    def findall(self, s: str) -> List[str]:
        """The matches (a pattern that matches no empty string)."""
        return self._pick(s)[0].findall(s)

    def parts(self, s: str) -> List[str]:
        """gap, match, gap, ..., gap (as split() with one group)."""
        if not s:
            return [s]
        rx, split_rx, empty = self._pick(s)
        if not empty:
            return split_rx.split(s)
        out, prev, start, last = [], 0, 0, -1
        while start <= len(s):
            m = rx.search(s, start)
            if m is None:
                break
            a, b = m.span()
            if a == b == last:
                start = b + 1
                continue
            out += [s[prev:a], s[a:b]]
            prev = start = last = b
        out.append(s[prev:])
        return out

    def sub(self, s: str, content: str) -> str:
        rx, _, empty = self._pick(s)
        if not empty:
            return rx.sub(lambda m: content, s)
        return "".join(content if k & 1 else p for k, p in enumerate(self.parts(s)))


def _onig_regex(pattern: str, what: str) -> _Regex:
    return _Regex(onig_to_re(pattern, what), onig_to_re(pattern, what, True))


# ── normalizers ───────────────────────────────────────────────────────────────

_nf_inert: Optional["re.Pattern"] = None
_lower_extra: Optional[Dict[int, str]] = None


def _nf(form: str, s: str) -> str:
    """unicodedata.normalize with the characters the reference's normalization
    data (Unicode 9.0) does not know left as they are: they split the text into
    runs that normalize on their own, as unassigned starters do."""
    global _nf_inert
    if s.isascii():
        return s
    if _nf_inert is None:
        _nf_inert = re.compile("(" + _class(_parse_ranges("".join(NF_INERT))) + ")")
    if not _nf_inert.search(s):
        return unicodedata.normalize(form, s)
    return "".join(p if k & 1 else unicodedata.normalize(form, p) for k, p in enumerate(_nf_inert.split(s)))


def _lowercase(s: str) -> str:
    """Rust's char::to_lowercase per character (Unicode 17.0: the newer
    letters from LOWER17, Σ always σ)."""
    global _lower_extra
    if s.isascii():
        return s.lower()
    if _lower_extra is None:
        _lower_extra = {}
        for a, b in re.findall(r"([0-9a-f]+):([0-9a-f]+)", "".join(LOWER17)):
            _lower_extra[int(a, 16)] = chr(int(b, 16))
    return s.replace("Σ", "σ").lower().translate(_lower_extra)


def _string_or_regex(pat: Dict[str, str], what: str) -> Tuple[Optional[str], Optional[_Regex]]:
    if "String" in pat:
        if not pat["String"]:
            raise ValueError(f"tokenizer.json: {what}: an empty pattern is not supported")
        return pat["String"], None
    if "Regex" in pat:
        return None, _onig_regex(pat["Regex"], what)
    raise ValueError(f"tokenizer.json: {what}: pattern {pat!r} is not supported")


def _normalizer(spec: Optional[dict]) -> Optional[Callable[[str], str]]:
    if spec is None:
        return None
    t = spec.get("type")
    if t == "Sequence":
        fs = [f for f in (_normalizer(x) for x in spec.get("normalizers", [])) if f is not None]
        if not fs:
            return None
        if len(fs) == 1:
            return fs[0]

        def seq(s: str) -> str:
            for f in fs:
                s = f(s)
            return s
        return seq
    if t in ("NFC", "NFD", "NFKC", "NFKD"):
        return lambda s: _nf(t, s)
    if t == "Lowercase":
        return _lowercase
    if t == "Strip":
        left, right = spec.get("strip_left", True), spec.get("strip_right", True)
        return lambda s: (s.strip(WHITE_SPACE) if left and right else s.lstrip(WHITE_SPACE) if left
                          else s.rstrip(WHITE_SPACE) if right else s)
    if t == "Prepend":
        pre = spec["prepend"]
        return lambda s: pre + s if s else s
    if t == "Replace":
        lit, rx = _string_or_regex(spec["pattern"], "the Replace normalizer's pattern")
        content = spec["content"]
        if rx is None:
            return lambda s: s.replace(lit, content)
        return lambda s: rx.sub(s, content)
    raise ValueError(f"tokenizer.json: normalizer {t!r} is not supported (hf_tokenizer.py: NFC, NFD, "
                     f"NFKC, NFKD, Lowercase, Strip, Prepend, Replace, Sequence)")


# ── pre-tokenizers: (pieces, first piece at offset 0?) -> the same ────────────

PreTokenizer = Callable[[List[str], bool], Tuple[List[str], bool]]
_BEHAVIORS = ("Isolated", "Removed", "MergedWithPrevious", "MergedWithNext", "Contiguous")


def _fold(segs: List[Tuple[str, bool, int]], behavior: str) -> List[Tuple[str, int]]:
    """tokenizers' NormalizedString::split: (segment, is_match, start) ->
    (piece, start); empty matches take part, empty pieces are dropped after."""
    if behavior == "Isolated":
        out = [(t, a) for t, _, a in segs]
    elif behavior == "Removed":
        out = [(t, a) for t, m, a in segs if not m]
    else:
        acc: List[List[Any]] = []
        prev = False
        if behavior == "MergedWithNext":
            for t, m, a in reversed(segs):
                if m and not prev and acc:
                    acc[-1] = [t + acc[-1][0], a]
                else:
                    acc.append([t, a])
                prev = m
            acc.reverse()
        else:
            for t, m, a in segs:
                if (m and not prev if behavior == "MergedWithPrevious" else m == prev) and acc:
                    acc[-1][0] += t
                else:
                    acc.append([t, a])
                prev = m
        out = [(t, a) for t, a in acc]
    return [(t, a) for t, a in out if t]


def _regex_splitter(rx: _Regex, behavior: str, invert: bool) -> Callable[[str], Tuple[List[str], bool]]:
    """A string -> (pieces, the first piece starts at 0)."""
    if behavior not in _BEHAVIORS:
        raise ValueError(f"tokenizer.json: Split behavior {behavior!r} is not supported")
    if behavior == "Isolated":
        return lambda s: ([p for p in rx.parts(s) if p], True)

    def split(s: str) -> Tuple[List[str], bool]:
        segs, pos = [], 0
        for k, p in enumerate(rx.parts(s)):            # gap, match, gap, ..., gap
            if p or k & 1:                             # non-empty gaps, every match
                segs.append((p, bool(k & 1) != invert, pos))
            pos += len(p)
        out = _fold(segs, behavior)
        return [t for t, _ in out], bool(out) and out[0][1] == 0
    return split


def _pieces_stage(split: Callable[[str], Tuple[List[str], bool]]) -> PreTokenizer:
    def stage(pieces: List[str], at0: bool) -> Tuple[List[str], bool]:
        out: List[str] = []
        for k, p in enumerate(pieces):
            parts, z = split(p)
            if k == 0:
                at0 = at0 and z
            out.extend(parts)
        return out, at0
    return stage


def _prepend_scheme(spec: dict) -> str:
    s = spec.get("prepend_scheme")
    if s is None:
        return "always" if spec.get("add_prefix_space", True) else "never"
    if s not in ("always", "first", "never"):
        raise ValueError(f"tokenizer.json: Metaspace prepend_scheme {s!r} is not supported")
    return s


def _pre_tokenizer(spec: Optional[dict]) -> List[PreTokenizer]:
    if spec is None:
        return []
    t = spec.get("type")
    if t == "Sequence":
        return [st for x in spec.get("pretokenizers", []) for st in _pre_tokenizer(x)]
    if t == "ByteLevel":
        prefix, use_regex = spec.get("add_prefix_space", True), spec.get("use_regex", True)
        rx = _gpt2() if use_regex else None
        table = _BYTES_TO_CHARS

        def byte_level(pieces: List[str], at0: bool) -> Tuple[List[str], bool]:
            out = []
            for p in pieces:
                if prefix and not p.startswith(" "):
                    p = " " + p
                for w in (rx.findall(p) if rx is not None else (p,)):   # the pattern covers every character
                    out.append(w.encode("utf-8").decode("latin-1").translate(table))
            return out, at0
        return [byte_level]
    if t == "Split":
        lit, rx = _string_or_regex(spec["pattern"], "the Split pre-tokenizer's pattern")
        return [_pieces_stage(_regex_splitter(_Regex(re.escape(lit)) if rx is None else rx,
                                              spec.get("behavior", "Isolated"), bool(spec.get("invert", False))))]
    if t == "Digits":
        more = "" if spec.get("individual_digits", False) else "+"
        rx = _Regex(_class(_numeric_ranges()) + more, _class(_numeric_ranges(), True) + more)
        return [_pieces_stage(_regex_splitter(rx, "Isolated", False))]
    if t == "Metaspace":
        repl = spec.get("replacement", SPACE)
        scheme, do_split = _prepend_scheme(spec), spec.get("split", True)
        split = _regex_splitter(_Regex(re.escape(repl)), "MergedWithNext", False)

        def metaspace(pieces: List[str], at0: bool) -> Tuple[List[str], bool]:
            out: List[str] = []
            for k, p in enumerate(pieces):
                p = p.replace(" ", repl)
                if not p.startswith(repl) and (scheme == "always" or scheme == "first" and k == 0 and at0):
                    p = repl + p
                if do_split:
                    out.extend(split(p)[0])
                else:
                    out.append(p)
            return out, at0
        return [metaspace]
    raise ValueError(f"tokenizer.json: pre-tokenizer {t!r} is not supported (hf_tokenizer.py: ByteLevel, "
                     f"Split, Digits, Metaspace, Sequence)")


# ── the BPE model ─────────────────────────────────────────────────────────────

class _BPE:
    """tokenizers' BPE: merge_word (byte fallback, unk fusing, the subword
    prefix / suffix) and Word::merge_all (lowest rank first, leftmost on ties)."""

    NAIVE = 24                                     # words up to this many symbols: a plain scan

    def __init__(self, spec: dict, cache_size: int):
        if spec.get("dropout"):
            raise ValueError("tokenizer.json: BPE dropout is not supported (non-deterministic)")
        self.vocab: Dict[str, int] = spec["vocab"]
        self.id_to_token: Dict[int, str] = {i: t for t, i in self.vocab.items()}
        self.unk_token: Optional[str] = spec.get("unk_token")
        self.byte_fallback = bool(spec.get("byte_fallback"))
        self.fuse_unk = bool(spec.get("fuse_unk"))
        self.ignore_merges = bool(spec.get("ignore_merges"))
        self.prefix = spec.get("continuing_subword_prefix") or None
        self.suffix = spec.get("end_of_word_suffix") or None
        vocab, cut = self.vocab, len(self.prefix or "")
        merges: Dict[Tuple[int, int], Tuple[int, int]] = {}
        for rank, m in enumerate(spec.get("merges") or []):
            if isinstance(m, str):
                m = m.split(" ")
            if len(m) != 2:
                raise ValueError(f"tokenizer.json: the merge {m!r} is not a pair")
            a, b = m
            try:
                merges[(vocab[a], vocab[b])] = (rank, vocab[a + b[cut:]])
            except KeyError as e:
                raise ValueError(f"tokenizer.json: the merge {a!r} {b!r} uses {e} outside the vocabulary") \
                    from None
        self.merges = merges
        self._byte_ids = [vocab.get("<0x%02X>" % b) for b in range(256)]
        self._plain = not (self.prefix or self.suffix or self.byte_fallback or self.unk_token)
        self._cache: "OrderedDict[str, Tuple[int, ...]]" = OrderedDict()
        self._cache_size = cache_size

    def _symbols(self, w: str) -> List[int]:
        vocab = self.vocab
        if self._plain:                            # characters outside the vocabulary are dropped
            return [i for i in map(vocab.get, w) if i is not None]
        sym: List[int] = []
        unk: Optional[int] = None                  # a pending (fused) unk
        last = len(w) - 1
        affix = bool(self.prefix or self.suffix)
        get = vocab.get
        for k, ch in enumerate(w):
            s = ch
            if affix:
                if k and self.prefix:
                    s = self.prefix + s
                if k == last and self.suffix:
                    s = s + self.suffix
            i = get(s)
            if i is not None:
                if unk is not None:
                    sym.append(unk)
                    unk = None
                sym.append(i)
                continue
            if self.byte_fallback:
                bs = [self._byte_ids[b] for b in s.encode("utf-8")]
                if None not in bs:
                    sym.extend(bs)                 # (a pending unk stays pending, as in tokenizers)
                    continue
            if self.unk_token is not None:
                u = vocab.get(self.unk_token)
                if u is None:
                    raise ValueError(f"tokenizer.json: unk_token {self.unk_token!r} is not in the vocabulary")
                if unk is not None and not self.fuse_unk:
                    sym.append(unk)
                unk = u
        if unk is not None:
            sym.append(unk)
        return sym

    def _merge(self, sym: List[int]) -> List[int]:
        n = len(sym)
        merges = self.merges
        if n <= self.NAIVE:
            while len(sym) > 1:
                best, best_rank, new = -1, 1 << 62, 0
                for p, pair in enumerate(zip(sym, sym[1:], strict=False)):
                    m = merges.get(pair)
                    if m is not None and m[0] < best_rank:
                        best, best_rank, new = p, m[0], m[1]
                if best < 0:
                    break
                a, b = sym[best], sym[best + 1]
                out, p, n = sym[:best], best, len(sym)
                while p < n:                       # every occurrence, left to right
                    if p < n - 1 and sym[p] == a and sym[p + 1] == b:
                        out.append(new)
                        p += 2
                    else:
                        out.append(sym[p])
                        p += 1
                sym = out
            return sym
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
            if m is None or m[1] != new:           # an expired entry
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
        return [s for s, a in zip(sym, alive, strict=True) if a]

    def tokenize(self, w: str) -> Tuple[int, ...]:
        if self.ignore_merges:
            i = self.vocab.get(w)
            if i is not None:
                return (i,)
        hit = self._cache.get(w)
        if hit is not None:
            return hit
        ids = tuple(self._merge(self._symbols(w)))
        if self._cache_size and len(w) < 256:
            self._cache[w] = ids
            if len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return ids


# ── post-processors ───────────────────────────────────────────────────────────

def _post_processor(spec: Optional[dict]) -> Optional[Callable[[List[int]], List[int]]]:
    if spec is None:
        return None
    t = spec.get("type")
    if t == "ByteLevel":                           # trims offsets only
        return None
    if t == "TemplateProcessing":
        specials = spec.get("special_tokens") or {}
        seq: List[Optional[int]] = []              # None: the text's ids
        for piece in spec.get("single", []):
            if "SpecialToken" in piece:
                name = piece["SpecialToken"]["id"]
                if name not in specials:
                    raise ValueError(f"tokenizer.json: TemplateProcessing names {name!r} without its ids")
                seq.extend(specials[name]["ids"])
            elif "Sequence" in piece:
                seq.append(None)
            else:
                raise ValueError(f"tokenizer.json: TemplateProcessing piece {piece!r} is not supported")

        def template(ids: List[int]) -> List[int]:
            out: List[int] = []
            for x in seq:
                if x is None:
                    out.extend(ids)
                else:
                    out.append(x)
            return out
        return template
    if t in ("RobertaProcessing", "BertProcessing"):
        cls, sep = spec["cls"][1], spec["sep"][1]
        return lambda ids: [cls] + ids + [sep]
    if t == "Sequence":
        fs = [f for f in (_post_processor(x) for x in spec.get("processors", [])) if f is not None]

        def seq_post(ids: List[int]) -> List[int]:
            for f in fs:
                ids = f(ids)
            return ids
        return seq_post if fs else None
    raise ValueError(f"tokenizer.json: post-processor {t!r} is not supported (hf_tokenizer.py: "
                     f"TemplateProcessing, ByteLevel, RobertaProcessing, BertProcessing, Sequence)")


# ── decoders ──────────────────────────────────────────────────────────────────
#
# A step has batch() — tokenizers' decode_chain on the whole token list — and
# stream(), a stateful copy fed one token at a time whose outputs concatenate to
# batch()'s.  After Fuse or ByteLevel the list is one string (fused): later steps
# stream its chunks.  ``cont`` (strip_first=False) decodes a continuation: the
# start-of-text rules (Strip after Fuse, Metaspace's first token, the join) are
# off.  tb() is a token's bytes in the middle of a text (token_bytes()).

_BL_ALPHABET = re.compile(_class(_merge((ord(c), ord(c)) for c in _B2U.values())) + "*")
_BYTE_TOKEN = re.compile(rb"<0x(\+?[0-9A-Fa-f]{1,2})>")


def _bl_bytes(t: str) -> bytes:
    """ByteLevel: a token's characters as bytes (all in the alphabet), else its UTF-8."""
    if _BL_ALPHABET.fullmatch(t):
        return t.translate(_CHARS_TO_BYTES).encode("latin-1")
    return t.encode("utf-8")


def _fallback_byte(t: str) -> Optional[int]:
    """ByteFallback: "<0xXX>" -> XX."""
    b = t.encode("utf-8")
    if len(b) != 6:
        return None
    m = _BYTE_TOKEN.fullmatch(b)
    return int(m.group(1), 16) if m else None


class _Stream:
    """A streaming step: push(item) / flush() -> items; pending: holds text."""

    def __init__(self, push: Callable[[str], List[str]], flush: Callable[[], List[str]] = lambda: [],
                 pending: Callable[[], bool] = lambda: False):
        self.push, self.flush, self._pending = push, flush, pending

    @property
    def pending(self) -> bool:
        return self._pending()


def _buffered(batch: Callable[[str], str]) -> _Stream:
    """A fused step that needs the whole text: held until flush."""
    buf: List[str] = []
    return _Stream(lambda s: buf.append(s) or [], lambda: [batch("".join(buf))] if buf else [],
                   lambda: bool(buf))


class _Step:
    fuses = False
    fused_in = False

    def batch(self, toks: List[str], cont: bool) -> List[str]:
        raise NotImplementedError

    def stream(self, cont: bool) -> _Stream:
        raise NotImplementedError

    def tb(self, x: Union[str, bytes]) -> Union[str, bytes]:
        return x


class _ReplaceDec(_Step):
    def __init__(self, spec: dict):
        self.lit, self.rx = _string_or_regex(spec["pattern"], "the Replace decoder's pattern")
        self.content = spec["content"]

    def one(self, t: str) -> str:
        return t.replace(self.lit, self.content) if self.rx is None else self.rx.sub(t, self.content)

    def batch(self, toks, cont):
        return [self.one(t) for t in toks]

    def stream(self, cont):
        if not self.fused_in or self.rx is None and len(self.lit) == 1:
            return _Stream(lambda s: [self.one(s)])
        return _buffered(self.one)

    def tb(self, x):
        if isinstance(x, str):
            return self.one(x)
        return x.replace(self.lit.encode("utf-8"), self.content.encode("utf-8")) if self.rx is None else x


class _ByteFallbackDec(_Step):
    @staticmethod
    def _run(bs: bytearray) -> List[str]:
        try:
            return [bs.decode("utf-8")]
        except UnicodeDecodeError:
            return ["�"] * len(bs)

    def batch(self, toks, cont):
        out: List[str] = []
        run = bytearray()
        for t in toks:
            b = _fallback_byte(t)
            if b is not None:
                run.append(b)
                continue
            if run:
                out += self._run(run)
                run = bytearray()
            out.append(t)
        if run:
            out += self._run(run)
        return out

    def stream(self, cont):
        if self.fused_in:
            return _buffered(lambda s: "".join(self.batch([s], cont)))
        run = bytearray()

        def push(t: str) -> List[str]:
            b = _fallback_byte(t)
            if b is not None:
                run.append(b)
                return []
            return flush() + [t]

        def flush() -> List[str]:
            if not run:
                return []
            out = self._run(run)
            run.clear()
            return out
        return _Stream(push, flush, lambda: bool(run))

    def tb(self, x):
        b = _fallback_byte(x) if isinstance(x, str) else None
        return x if b is None else bytes([b])


class _FuseDec(_Step):
    fuses = True

    def batch(self, toks, cont):
        return ["".join(toks)]

    def stream(self, cont):
        return _Stream(lambda s: [s])


class _StripDec(_Step):
    def __init__(self, spec: dict):
        self.content, self.start, self.stop = spec["content"], int(spec.get("start", 0)), int(spec.get("stop", 0))

    def one(self, t: str, lead: bool = True) -> str:
        a = 0
        if lead:
            while a < min(self.start, len(t)) and t[a] == self.content:
                a += 1
        b = len(t)
        while len(t) - b < self.stop and b > a and t[b - 1] == self.content:
            b -= 1
        return t[a:b]

    def batch(self, toks, cont):
        return [self.one(t, not (cont and self.fused_in)) for t in toks]

    def stream(self, cont):
        if not self.fused_in:
            return _Stream(lambda s: [self.one(s)])
        st = {"lead": 0 if cont else self.start, "held": ""}
        c, stop = self.content, self.stop

        def push(s: str) -> List[str]:
            if st["lead"]:
                k = 0
                while k < len(s) and st["lead"] and s[k] == c:
                    k += 1
                    st["lead"] -= 1
                if k < len(s):
                    st["lead"] = 0                 # the text has begun
                s = s[k:]
            if not stop:
                return [s]
            s = st["held"] + s
            k = 0
            while k < stop and k < len(s) and s[len(s) - 1 - k] == c:
                k += 1
            st["held"] = s[len(s) - k:]
            return [s[:len(s) - k]]

        def flush() -> List[str]:
            st["held"] = ""                        # the stripped tail
            return []
        return _Stream(push, flush, lambda: bool(st["held"]))

    def tb(self, x):
        if self.fused_in:
            return x
        if isinstance(x, str):
            return self.one(x)
        return self.one(x.decode("utf-8", "surrogateescape")).encode("utf-8", "surrogateescape")


class _MetaspaceDec(_Step):
    def __init__(self, spec: dict):
        self.repl, self.scheme = spec.get("replacement", SPACE), _prepend_scheme(spec)

    def one(self, t: str, first: bool) -> str:
        return t.replace(self.repl, "" if first and self.scheme != "never" else " ")

    def batch(self, toks, cont):
        return [self.one(t, k == 0 and not cont) for k, t in enumerate(toks)]

    def stream(self, cont):
        st = [not cont]                            # the next token is the first

        def push(s: str) -> List[str]:
            first = st[0]
            if not self.fused_in:
                st[0] = False
            return [self.one(s, first)]
        return _Stream(push)

    def tb(self, x):
        if isinstance(x, str):
            return self.one(x, False)
        return x.replace(self.repl.encode("utf-8"), b" ")


class _ByteLevelDec(_Step):
    fuses = True

    def batch(self, toks, cont):
        return [b"".join(_bl_bytes(t) for t in toks).decode("utf-8", "replace")]

    def stream(self, cont):
        if self.fused_in:
            return _buffered(lambda s: self.batch([s], cont)[0])
        dec = codecs.getincrementaldecoder("utf-8")("replace")
        return _Stream(lambda s: [dec.decode(_bl_bytes(s), False)], lambda: [dec.decode(b"", True)],
                       lambda: bool(dec.getstate()[0]))

    def tb(self, x):
        return _bl_bytes(x) if isinstance(x, str) else x


class _BPESuffixDec(_Step):
    def __init__(self, spec: dict):
        self.suffix = spec.get("suffix", "</w>")

    def batch(self, toks, cont):
        n = len(toks) - 1
        return [t.replace(self.suffix, "" if k == n else " ") for k, t in enumerate(toks)]

    def stream(self, cont):
        if self.fused_in:
            return _buffered(lambda s: s.replace(self.suffix, ""))
        held: List[str] = []

        def push(s: str) -> List[str]:
            out = [held.pop().replace(self.suffix, " ")] if held else []
            held.append(s)
            return out
        return _Stream(push, lambda: [held.pop().replace(self.suffix, "")] if held else [],
                       lambda: bool(held))

    def tb(self, x):
        return x.replace(self.suffix, " ") if isinstance(x, str) else x.replace(self.suffix.encode(), b" ")


class _JoinDec(_Step):
    """No decoder: the tokens joined with " "."""
    fuses = True

    def batch(self, toks, cont):
        return [(" " if cont and toks else "") + " ".join(toks)]

    def stream(self, cont):
        st = [cont]

        def push(s: str) -> List[str]:
            sep = st[0]
            st[0] = True
            return [" " + s if sep else s]
        return _Stream(push)

    def tb(self, x):
        return " " + x if isinstance(x, str) else b" " + x


def _decoder(spec: Optional[dict]) -> List[_Step]:
    """The steps of a decoder (none: the tokens joined with " "; an empty
    Sequence: joined with "")."""
    if spec is None:
        return [_JoinDec()]
    t = spec.get("type")
    if t == "Sequence":
        return [s for x in spec.get("decoders", []) for s in _decoder(x)]
    step = {"ByteLevel": _ByteLevelDec, "ByteFallback": _ByteFallbackDec, "Fuse": _FuseDec}.get(t)
    if step is not None:
        return [step()]
    step = {"Replace": _ReplaceDec, "Strip": _StripDec, "Metaspace": _MetaspaceDec,
            "BPEDecoder": _BPESuffixDec}.get(t)
    if step is not None:
        return [step(spec)]
    raise ValueError(f"tokenizer.json: decoder {t!r} is not supported (hf_tokenizer.py: ByteLevel, Replace, "
                     f"ByteFallback, Fuse, Strip, Metaspace, BPEDecoder, Sequence)")


# ── added tokens ──────────────────────────────────────────────────────────────

class AddedToken(NamedTuple):
    id: int
    content: str
    special: bool
    normalized: bool
    lstrip: bool
    rstrip: bool
    single_word: bool


def _trie_regex(strings: Iterable[str]) -> str:
    """Python re source matching the longest of ``strings`` at the leftmost
    position (Aho-Corasick's LeftmostLongest): a trie of nested optional groups."""
    trie: Dict[str, Any] = {}
    for s in strings:
        node = trie
        for ch in s:
            node = node.setdefault(ch, {})
        node[""] = True

    def build(node: Dict[str, Any]) -> str:
        alts = [re.escape(ch) + build(child) for ch, child in sorted(node.items()) if ch]
        if not alts:
            return ""
        body = alts[0] if len(alts) == 1 else "(?:" + "|".join(alts) + ")"
        return "(?:" + body + ")?" if "" in node else body
    return build(trie)


class _Matcher:
    """One of tokenizers' two added-token tries: find(s) -> [(start, end, id
    or None)] covering s, with the lstrip / rstrip / single_word rules."""

    def __init__(self, patterns: Dict[str, AddedToken]):
        self.tokens = patterns
        self.rx = re.compile(_trie_regex(patterns))
        self.flags = any(t.lstrip or t.rstrip or t.single_word for t in patterns.values())
        self.word = re.compile(_class(_word_ranges())) if any(t.single_word for t in patterns.values()) \
            else None

    def find(self, s: str, skip_special: bool) -> List[Tuple[int, int, Optional[int]]]:
        out: List[Tuple[int, int, Optional[int]]] = []
        pos = 0
        for m in self.rx.finditer(s):
            tok = self.tokens[m.group()]
            if skip_special and tok.special:
                continue
            start, stop = m.span()
            if self.flags:
                if tok.single_word and (start > 0 and self.word.match(s, start - 1)
                                        or stop < len(s) and self.word.match(s, stop)):
                    continue
                if tok.lstrip:
                    a = start
                    while a > 0 and s[a - 1] in WHITE_SPACE:
                        a -= 1
                    start = max(a, pos)
                if tok.rstrip:
                    while stop < len(s) and s[stop] in WHITE_SPACE:
                        stop += 1
            if pos < start:
                out.append((pos, start, None))
            out.append((start, stop, tok.id))
            pos = stop
        if pos < len(s):
            out.append((pos, len(s), None))
        return out


# ── the tokenizer ─────────────────────────────────────────────────────────────

def _config_tokens(directory: Optional[str]) -> Dict[str, Any]:
    """bos_token / eos_token / unk_token / pad_token from tokenizer_config.json
    (special_tokens_map.json's where it has none) beside tokenizer.json."""
    out: Dict[str, Any] = {}
    if not directory:
        return out
    for name in ("special_tokens_map.json", "tokenizer_config.json"):
        p = os.path.join(directory, name)
        try:
            with open(p, encoding="utf-8") as f:
                doc = json.load(f)
        except (OSError, ValueError):
            continue
        for key in ("bos_token", "eos_token", "unk_token", "pad_token"):
            v = doc.get(key)
            if isinstance(v, dict):
                v = v.get("content")
            if isinstance(v, str) and v:
                out[key] = v
    return out


class Tokenizer:
    """A tokenizer.json (a path or its parsed dict); see the module docstring.

    Attributes: vocab (the model's token -> id), id_to_token (id -> token
    string as tokenizers' id_to_token, None for unused ids), added (every added
    token's content -> id), special (the special ones), special_ids,
    added_tokens (id -> AddedToken), vocab_size (max id + 1), merges ((left id,
    right id) -> (rank, merged id)), bos_token / eos_token and bos_id / eos_id
    (tokenizer_config.json / special_tokens_map.json beside the file, else the
    post-processor's leading special token for BOS)."""

    def __init__(self, source: Union[str, "os.PathLike", Dict[str, Any]], cache_size: int = 50_000,
                 config: Optional[Dict[str, Any]] = None):
        if isinstance(source, dict):
            spec, self.path = source, None
        else:
            self.path = os.fspath(source)
            with open(self.path, encoding="utf-8") as f:
                spec = json.load(f)
        for part in ("truncation", "padding"):
            if spec.get(part):
                raise ValueError(f"tokenizer.json: {part} is set; it is not supported")
        model = spec.get("model") or {}
        mtype = model.get("type") or ("BPE" if "merges" in model else None)
        if mtype != "BPE":
            raise ValueError(f"tokenizer.json: model {mtype!r} is not supported (hf_tokenizer.py runs BPE)")
        self.model = _BPE(model, cache_size)
        self.vocab: Dict[str, int] = self.model.vocab
        self.merges = self.model.merges
        self._normalize = _normalizer(spec.get("normalizer"))
        self._pre = _pre_tokenizer(spec.get("pre_tokenizer"))
        self._post = _post_processor(spec.get("post_processor"))
        self._decoders = _decoder(spec.get("decoder"))
        fused = False
        for st in self._decoders:
            st.fused_in = fused
            fused = fused or st.fuses
        self._bl_only = len(self._decoders) == 1 and isinstance(self._decoders[0], _ByteLevelDec)
        self._add_tokens(spec.get("added_tokens") or [])
        n = max([*self.model.id_to_token, *self.added_tokens, -1]) + 1
        self.id_to_token: List[Optional[str]] = [None] * n
        for i, t in self.model.id_to_token.items():
            self.id_to_token[i] = t
        for i, t in self.added_tokens.items():
            self.id_to_token[i] = self._norm_cache.get(i, t.content)
        self.vocab_size = n
        self._id_bytes: List[Optional[bytes]] = [None] * n
        names = dict(config) if config is not None else _config_tokens(os.path.dirname(self.path or ""))
        self.bos_token: Optional[str] = names.get("bos_token")
        self.eos_token: Optional[str] = names.get("eos_token")
        self.bos_id = self.token_to_id(self.bos_token) if self.bos_token else None
        self.eos_id = self.token_to_id(self.eos_token) if self.eos_token else None
        if self.bos_id is None and self._post is not None and not self.bos_token:
            lead = self._post([])                  # the template's leading special token
            if lead:
                self.bos_id = lead[0]
                self.bos_token = self.id_to_token[lead[0]]

    def _add_tokens(self, tokens: List[dict]) -> None:
        """tokenizers' AddedVocabulary::add_tokens: an id from the added map or
        the model's vocabulary, else the next one after the vocabulary; the
        normalized form of a normalized token (matched, and decoded)."""
        amap: Dict[str, int] = {}
        arev: Dict[int, AddedToken] = {}
        specials: List[str] = []
        self._norm_cache: Dict[int, str] = {}
        next_id = len(self.vocab)
        for a in tokens:
            content = a.get("content") or ""
            if not content:
                continue
            special = bool(a.get("special", False))
            tok = AddedToken(-1, content, special, bool(a.get("normalized", not special)),
                             bool(a.get("lstrip", False)), bool(a.get("rstrip", False)),
                             bool(a.get("single_word", False)))
            i = amap.get(content)
            if i is not None and arev[i]._replace(id=-1) == tok:
                continue
            if i is None:
                i = self.vocab.get(content)
            if i is None:
                i, next_id = next_id, next_id + 1
            if tok.normalized and self._normalize is not None:
                nc = self._normalize(content)
                if nc != content:
                    self._norm_cache[i] = nc
            amap[content] = i
            if special and content not in specials:
                specials.append(content)
            arev[i] = tok._replace(id=i)
        self.added: Dict[str, int] = amap
        self.added_tokens: Dict[int, AddedToken] = arev
        self.special: Dict[str, int] = {c: amap[c] for c in specials}
        self.special_ids = frozenset(self.special.values())
        raw = {t.content: t for t in arev.values() if not t.normalized}
        norm = {self._norm_cache.get(t.id, t.content): t for t in arev.values() if t.normalized}
        self._raw = _Matcher(raw) if raw else None
        self._norm = _Matcher(norm) if norm else None

    # ── encoding ──

    def token_to_id(self, token: str) -> Optional[int]:
        i = self.added.get(token)
        return self.vocab.get(token) if i is None else i

    def _pieces(self, s: str, at0: bool, out: List[int]) -> None:
        """A piece without added tokens: pre-tokenized, then the model."""
        if not s:
            return
        tokenize = self.model.tokenize
        if not self._pre:
            out.extend(tokenize(s))
            return
        pieces = [s]
        for stage in self._pre:
            pieces, at0 = stage(pieces, at0)
        for p in pieces:
            out.extend(tokenize(p))

    def encode(self, text: str, special: bool = True, add_special_tokens: bool = True) -> List[int]:
        """Token ids of text, as tokenizers' encode(text, add_special_tokens).
        special=False: special-token strings in the text are tokenized as
        ordinary text (tokenizers' encode_special_tokens; non-special added
        tokens are still matched)."""
        out: List[int] = []
        skip = not special
        segs = self._raw.find(text, skip) if self._raw is not None else [(0, len(text), None)]
        for a, b, tid in segs:
            if tid is not None:
                out.append(tid)
                continue
            piece = text[a:b]
            if self._normalize is not None:
                piece = self._normalize(piece)
            if self._norm is None:
                self._pieces(piece, a == 0, out)
                continue
            for c, d, tid in self._norm.find(piece, skip):
                if tid is not None:
                    out.append(tid)
                else:
                    self._pieces(piece[c:d], a == 0 and c == 0, out)
        if add_special_tokens and self._post is not None:
            out = self._post(out)
        return out

    def tokens(self, text: str, **kw) -> List[Optional[str]]:
        """The token strings of encode(text, **kw)."""
        return [self.id_to_token[i] for i in self.encode(text, **kw)]

    # ── decoding ──

    def is_special(self, i: int) -> bool:
        return i in self.special_ids

    def token_bytes(self, i: int) -> bytes:
        """What token i adds to a decoded text in its middle: a byte-fallback /
        byte-level token its raw byte(s), "▁" a space, ... (b"" for unused ids)."""
        if not 0 <= i < self.vocab_size:
            return b""
        b = self._id_bytes[i]
        if b is None:
            t = self.id_to_token[i]
            if t is None:
                b = b""
            else:
                x: Union[str, bytes] = t
                for st in self._decoders:
                    x = st.tb(x)
                b = x if isinstance(x, bytes) else x.encode("utf-8")
            self._id_bytes[i] = b
        return b

    @property
    def id_bytes(self) -> List[bytes]:
        """token_bytes() of every id."""
        return [self.token_bytes(i) for i in range(self.vocab_size)]

    def _strings(self, ids: Iterable[int], skip_special_tokens: bool) -> List[str]:
        skip = self.special_ids if skip_special_tokens else frozenset()
        n, names = self.vocab_size, self.id_to_token
        return [names[i] for i in ids if 0 <= i < n and i not in skip and names[i] is not None]

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = True, strip_first: bool = True) -> str:
        """Text of ids (tokenizers' decode; special tokens skipped by id; ids
        outside the vocabulary skipped).  strip_first=False: decoded as the
        continuation of a text (SentencePiece's leading space kept)."""
        if self._bl_only:
            skip = self.special_ids if skip_special_tokens else frozenset()
            n, tb = self.vocab_size, self.token_bytes
            return b"".join(tb(i) for i in ids if 0 <= i < n and i not in skip).decode("utf-8", "replace")
        toks = self._strings(ids, skip_special_tokens)
        cont = not strip_first
        for st in self._decoders:
            toks = st.batch(toks, cont)
        return "".join(toks)

    def incremental_decoder(self, skip_special_tokens: bool = True,
                            strip_first: bool = True) -> "IncrementalDecoder":
        return IncrementalDecoder(self, skip_special_tokens, strip_first)


class IncrementalDecoder:
    """Streaming decode: add(id) returns the text complete so far — an
    incomplete UTF-8 sequence, a run of byte-fallback tokens, a strippable
    tail is held back — and flush() the rest.  The pieces concatenate to
    Tokenizer.decode() of the same ids.  strip_first=False: a continuation
    (the leading space of its first token kept)."""

    def __init__(self, tok: Tokenizer, skip_special_tokens: bool = True, strip_first: bool = True):
        self.tok = tok
        self.skip = tok.special_ids if skip_special_tokens else frozenset()
        self._bl = tok._bl_only
        if self._bl:
            self._dec = codecs.getincrementaldecoder("utf-8")("replace")
        else:
            self._steps = [st.stream(not strip_first) for st in tok._decoders]

    def add(self, i: int) -> str:
        tok = self.tok
        if i in self.skip or not 0 <= i < tok.vocab_size or tok.id_to_token[i] is None:
            return ""
        if self._bl:
            return self._dec.decode(tok.token_bytes(i), False)
        items = [tok.id_to_token[i]]
        for st in self._steps:
            nxt: List[str] = []
            for x in items:
                nxt.extend(st.push(x))
            items = nxt
            if not items:
                break
        return "".join(items)

    def add_many(self, ids: Sequence[int]) -> str:
        return "".join(self.add(i) for i in ids)

    def flush(self) -> str:
        if self._bl:
            return self._dec.decode(b"", True)
        items: List[str] = []
        for st in self._steps:
            nxt: List[str] = []
            for x in items:
                nxt.extend(st.push(x))
            nxt.extend(st.flush())
            items = nxt
        return "".join(items)

    @property
    def pending(self) -> bool:
        """True while text is held back (an incomplete character, byte tokens)."""
        if self._bl:
            return bool(self._dec.getstate()[0])
        return any(st.pending for st in self._steps)


# ── Unicode tables ────────────────────────────────────────────────────────────
# Regenerate with `python3 hf_tokenizer.py --unicode-tables UCD16 UCD17`, the
# directories holding https://www.unicode.org/Public/<16.0.0|17.0.0>/ucd/
# (extracted/DerivedGeneralCategory.txt; 17.0: also UnicodeData.txt and
# DerivedAge.txt).

GC16_RUNS = (
    "0Cc20Zs1Po3Sc1Po3Ps1Pe1Po1Sm1Po1Pd1Po2NdaPo2Sm3Po2Lu1aPs1Po1Pe1Sk1Pc1Sk1Ll1aPs1Sm1Pe1Sm1Cc21Zs"
    "1Po1Sc4So1Po1Sk1So1Lo1Pi1Sm1Cf1So1Sk1So1Sm1No2Sk1Ll1Po2Sk1No1Lo1Pf1No3Po1Lu17Sm1Lu7Ll18Sm1Ll8Lu"
    "1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu"
    "1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll2Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu"
    "1Ll1Lu1Ll1Lu1Ll1Lu1Ll2Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu"
    "1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu2Ll1Lu1Ll1Lu1Ll3Lu2Ll1Lu1Ll1Lu"
    "2Ll1Lu3Ll2Lu4Ll1Lu2Ll1Lu3Ll3Lu2Ll1Lu2Ll1Lu1Ll1Lu1Ll1Lu2Ll1Lu1Ll2Lu1Ll1Lu2Ll1Lu3Ll1Lu1Ll1Lu2Ll2Lo"
    "1Lu1Ll3Lo4Lu1Lt1Ll1Lu1Lt1Ll1Lu1Lt1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll2Lu1Ll1Lu1Ll"
    "1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll2Lu1Lt1Ll1Lu1Ll1Lu3Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu"
    "1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu"
    "1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll7Lu2Ll1Lu2Ll2Lu1Ll1Lu4Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll"
    "45Lo1Ll1bLm12Sk4LmcSkeLm5Sk7Lm1Sk1Lm1Sk11Mn70Lu1Ll1Lu1Ll1Lm1Sk1Lu1Ll1Cn2Lm1Ll3Po1Lu1Cn4Sk2Lu1Po"
    "1Lu3Cn1Lu1Cn1Lu2Ll1Lu11Cn1Lu9Ll23Lu1Ll2Lu3Ll3Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu"
    "1Ll1Lu1Ll1Lu1Ll1Lu1Ll5Lu1Ll1Sm1Lu1Ll1Lu2Ll2Lu33Ll30Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu"
    "1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1So1Mn5Me2Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll"
    "1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll"
    "1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu2Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll2Lu1Ll1Lu1Ll1Lu1Ll"
    "1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll"
    "1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll"
    "1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Cn1Lu26Cn2Lm1Po"
    "6Ll29Po1Pd1Cn2So2Sc1Cn1Mn2dPd1Mn1Po1Mn2Po1Mn2Po1Mn1Cn8Lo1bCn4Lo4Po2CnbCf6Sm3Po2Sc1Po2So2MnbPo1Cf"
    "1Po3Lo20Lm1LoaMn15NdaPo4Lo2Mn1Lo63Po1Lo1Mn7Cf1So1Mn6Lm2Mn2So1Mn4Lo2NdaLo3So2Lo1PoeCn1Cf1Lo1Mn1Lo"
    "1eMn1bCn2Lo59MnbLo1CneNdaLo21Mn9Lm2So1Po3Lm1Cn2Mn1Sc2Lo16Mn4Lm1Mn9Lm1Mn3Lm1Mn5Cn2PofCn1Lo19Mn3Cn"
    "2Po1Cn1LobCn5Lo18Sk1Lo6Cn1Cf2Cn5Mn9Lo29Lm1Mn18Cf1Mn20Mc1Lo36Mn1Mc1Mn1Lo1Mc3Mn8Mc4Mn1Mc2Lo1Mn7Lo"
    "aMn2Po2NdaPo1Lm1LofMn1Mc2Cn1Lo8Cn2Lo2Cn2Lo16Cn1Lo7Cn1Lo1Cn3Lo4Cn2Mn1Lo1Mc3Mn4Cn2Mc2Cn2Mc2Mn1Lo"
    "1Cn8Mc1Cn4Lo2Cn1Lo3Mn2Cn2NdaLo2Sc2No6So1Sc1Lo1Po1Mn1Cn2Mn2Mc1Cn1Lo6Cn4Lo2Cn2Lo16Cn1Lo7Cn1Lo2Cn"
    "1Lo2Cn1Lo2Cn2Mn1Cn1Mc3Mn2Cn4Mn2Cn2Mn3Cn3Mn1Cn7Lo4Cn1Lo1Cn7NdaMn2Lo3Mn1Po1CnaMn2Mc1Cn1Lo9Cn1Lo3Cn"
    "1Lo16Cn1Lo7Cn1Lo2Cn1Lo5Cn2Mn1Lo1Mc3Mn5Cn1Mn2Mc1Cn1Mc2Mn1Cn2Lo1CnfLo2Mn2Cn2NdaPo1Sc1Cn7Lo1Mn6Cn"
    "1Mn1Mc2Cn1Lo8Cn2Lo2Cn2Lo16Cn1Lo7Cn1Lo2Cn1Lo5Cn2Mn1Lo1Mc1Mn1Mc1Mn4Cn2Mc2Cn2Mc2Mn1Cn7Mn2Mc1Cn4Lo"
    "2Cn1Lo3Mn2Cn2NdaSo1Lo1No6CnaMn1Lo1Cn1Lo6Cn3Lo3Cn1Lo4Cn3Lo2Cn1Lo1Cn1Lo2Cn3Lo2Cn3Lo3Cn3LocCn4Mc2Mn"
    "1Mc2Cn3Mc3Cn1Mc3Mn1Cn2Lo1Cn6Mc1CneNdaNo3So6Sc1So1Cn5Mn1Mc3Mn1Lo8Cn1Lo3Cn1Lo17Cn1Lo10Cn2Mn1Lo1Mn"
    "3Mc4Cn1Mn3Cn1Mn4Cn7Mn2Cn1Lo3Cn2Lo1Cn2Lo2Mn2Cn2NdaCn7Po1No7So1Lo1Mn1Mc2Po1Lo8Cn1Lo3Cn1Lo17Cn1Lo"
    "aCn1Lo5Cn2Mn1Lo1Mc1Mn1Mc5Cn1Mn1Mc2Cn1Mc2Mn2Cn7Mc2Cn6Lo2Cn1Lo2Mn2Cn2NdaCn1Lo2Mc1CncMn2Mc2Lo9Cn1Lo"
    "3Cn1Lo29Mn2Lo1Mc3Mn4Cn1Mc3Cn1Mc3Mn1Lo1So1Cn4Lo3Mc1No7Lo3Mn2Cn2NdaNo9So1Lo6Cn1Mn1Mc2Cn1Lo12Cn3Lo"
    "18Cn1Lo9Cn1Lo1Cn2Lo7Cn3Mn1Cn4Mc3Mn3Cn1Mn1Cn1Mc8Cn6NdaCn2Mc2Po1CncLo30Mn1Lo2Mn7Cn4Sc1Lo6Lm1Mn8Po"
    "1NdaPo2Cn25Lo2Cn1Lo1Cn1Lo5Cn1Lo18Cn1Lo1Cn1LoaMn1Lo2Mn9Lo1Cn2Lo5Cn1Lm1Cn1Mn7Cn1NdaCn2Lo4Cn20Lo1So"
    "3PofSo1Po1So3Mn2So6NdaNoaSo1Mn1So1Mn1So1Mn1Ps1Pe1Ps1Pe1Mc2Lo8Cn1Lo24Cn4MneMc1Mn5Po1Mn2Lo5MnbCn"
    "1Mn24Cn1So8Mn1So6Cn1So2Po5So4Po2Cn25Lo2bMc2Mn4Mc1Mn6Mc1Mn2Mc2Mn2Lo1NdaPo6Lo6Mc2Mn2Lo4Mn3Lo1Mc3Lo"
    "2Mc7Lo3Mn4LodMn1Mc2Mn2Mc6Mn1Lo1Mc1NdaMc3Mn1So2Lu26Cn1Lu1Cn5Lu1Cn2Ll2bPo1Lm1Ll3Lo149Cn1Lo4Cn2Lo"
    "7Cn1Lo1Cn1Lo4Cn2Lo29Cn1Lo4Cn2Lo21Cn1Lo4Cn2Lo7Cn1Lo1Cn1Lo4Cn2LofCn1Lo39Cn1Lo4Cn2Lo43Cn2Mn3Po9No"
    "14Cn3Lo10SoaCn6Lu56Cn2Ll6Cn2Pd1Lo26cSo1Po1Lo11Zs1Lo1aPs1Pe1Cn3Lo4bPo3Nl3Lo8Cn7Lo12Mn3Mc1Cn9Lo"
    "13Mn2Mc1Po2Cn9Lo12Mn2CncLodCn1Lo3Cn1Mn2CncLo34Mn2Mc1Mn7Mc8Mn1Mc2MnbPo3Lm1Po3Sc1Lo1Mn1Cn2NdaCn6No"
    "aCn6Po6Pd1Po4Mn3Cf1Mn1NdaCn6Lo23Lm1Lo35Cn7Lo5Mn2Lo22Mn1Lo1Cn5Lo46CnaLo1fCn1Mn3Mc4Mn2Mc3Cn4Mc2Mn"
    "1Mc6Mn3Cn4So1Cn3Po2NdaLo1eCn2Lo5CnbLo2cCn4Lo1aCn6NdaNo1Cn3So22Lo17Mn2Mc2Mn1Cn2Po2Lo35Mc1Mn1Mc1Mn"
    "7Cn1Mn1Mc1Mn1Mc2Mn8Mc6MnaCn2Mn1NdaCn6NdaCn6Po7Lm1Po6Cn2MneMe1Mn10Cn31Mn4Mc1Lo2fMn1Mc1Mn5Mc1Mn1Mc"
    "5Mn1Mc2Lo8Cn1Po2NdaPo7SoaMn9So9Po3Mn2Mc1Lo1eMc1Mn4Mc2Mn2Mc1Mn3Lo2NdaLo2cMn1Mc1Mn2Mc3Mn1Mc1Mn3Mc"
    "2Cn8Po4Lo24Mc8Mn8Mc2Mn2Cn3Po5NdaCn3Lo3NdaLo1eLm6Po2Ll9Lu1Ll1Cn5Lu2bCn2Lu3Po8Cn8Mn3Po1MndMc1Mn7Lo"
    "4Mn1Lo6Mn1Lo2Mc1Mn2Lo1Cn5Ll2cLm3fLldLm1Ll22Lm25Mn40Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu"
    "1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu"
    "1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu"
    "1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu"
    "1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu"
    "1Ll1Lu1Ll1Lu1Ll1Lu1Ll9Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu"
    "1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu"
    "1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu"
    "1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll9Lu8Ll6Cn2Lu6Cn2Ll8Lu8Ll8Lu8Ll6Cn2Lu6Cn2Ll8Cn1Lu1Cn1Lu1Cn1Lu1Cn1Lu1Ll"
    "8Lu8LleCn2Ll8Lt8Ll8Lt8Ll8Lt8Ll5Cn1Ll2Lu4Lt1Sk1Ll1Sk3Ll3Cn1Ll2Lu4Lt1Sk3Ll4Cn2Ll2Lu4Cn1Sk3Ll8Lu5Sk"
    "3Cn2Ll3Cn1Ll2Lu4Lt1Sk2Cn1ZsbCf5Pd6Po2Pi1Pf1Ps1Pi2Pf1Ps1Pi1Po8Zl1Zp1Cf5Zs1Po9Pi1Pf1Po4Pc2Po3Sm1Ps"
    "1Pe1PobSm1Po1Pc1PoaZs1Cf5Cn1CfaNo1Lm1Cn2No6Sm3Ps1Pe1Lm1NoaSm3Ps1Pe1Cn1LmdCn3Sc21CnfMndMe4Mn1Me"
    "3MncCnfSo2Lu1So4Lu1So2Ll1Lu3Ll2Lu3Ll1So1Lu1So2Sm1Lu5So6Lu1So1Lu1So1Lu1So1Lu4So1Ll1Lu4Ll1Lo4Ll1So"
    "2Ll2Lu2Sm5Lu1Ll4So1Sm1So2Ll1So1No10Nl23Lu1Ll1Nl4No1So2Cn4Sm5So5Sm2So4Sm1So2Sm1So2Sm1So7Sm1So1fSm"
    "2So2Sm1So1Sm1So1fSm10cSo8Ps1Pe1Ps1Pe1So14Sm2So7Ps1Pe1So51Sm1So1eSm19So28Sm6So48Cn16SobCn15No3cSo"
    "4eNo16Sob7Sm1So9Sm1So36Sm8So6fSm1Sof8Ps1Pe1Ps1Pe1Ps1Pe1Ps1Pe1Ps1Pe1Ps1Pe1Ps1Pe1No1eSo2cSm5Ps1Pe"
    "1Sm1fPs1Pe1Ps1Pe1Ps1Pe1Ps1Pe1Ps1Pe1Sm10So100Sm83Ps1Pe1Ps1Pe1Ps1Pe1Ps1Pe1Ps1Pe1Ps1Pe1Ps1Pe1Ps1Pe"
    "1Ps1Pe1Ps1Pe1Ps1Pe1Sm3fPs1Pe1Ps1Pe1Sm20Ps1Pe1Sm102So30Sm15So2Sm6So27Cn2So20Cn1So69Lu30Ll30Lu1Ll"
    "1Lu3Ll2Lu1Ll1Lu1Ll1Lu1Ll1Lu4Ll1Lu1Ll2Lu1Ll6Lm2Lu3Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu"
    "1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu"
    "1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu"
    "1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll2So6Lu1Ll1Lu1Ll1Mn3Lu1Ll1Cn5Po4No1Po2Ll"
    "26Cn1Ll1Cn5Ll1Cn2Lo38Cn7Lm1Po1CneMn1Lo17Cn9Lo7Cn1Lo7Cn1Lo7Cn1Lo7Cn1Lo7Cn1Lo7Cn1Lo7Cn1Lo7Cn1Mn"
    "20Po2Pi1Pf1Pi1Pf1Po3Pi1Pf1Po1Pi1Pf1Po9Pd1Po2Pd1Po1Pi1Pf1Po2Pi1Pf1Ps1Pe1Ps1Pe1Ps1Pe1Ps1Pe1Po5Lm"
    "1PoaPd2Po4Pd1Po1Ps1PodSo2Po3Ps1Pe1Ps1Pe1Ps1Pe1Ps1Pe1Pd1Cn22So1aCn1So59CncSod6Cn1aSo10Zs1Po3So1Lm"
    "1Lo1Nl1Ps1Pe1Ps1Pe1Ps1Pe1Ps1Pe1Ps1Pe1So2Ps1Pe1Ps1Pe1Ps1Pe1Ps1Pe1Pd1Ps1Pe2So1Nl9Mn4Mc2Pd1Lm5So2Nl"
    "3Lm1Lo1Po1So2Cn1Lo56Cn2Mn2Sk2Lm2Lo1Pd1Lo5aPo1Lm3Lo1Cn5Lo2bCn1Lo5eCn1So2No4SoaLo20So26Cn9So1Lo"
    "10So1fCn1NoaSo1eNo8So1NofSo20NoaSo27NofSo140Lo19c0So40Lo5215Lm1Lo477Cn3So37Cn9Lo28Lm6Po2Lo10cLm"
    "1Po3Lo10NdaLo2Cn14Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll"
    "1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lo1Mn1Me3Po1MnaPo1Lm1Lu1Ll1Lu1Ll1Lu"
    "1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lm2Mn2Lo46NlaMn2Po6Cn8Sk"
    "17Lm9Sk2Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll3Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu"
    "1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu"
    "1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lm1Ll8Lu1Ll1Lu1Ll1Lu2Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lm"
    "1Sk2Lu1Ll1Lu1Ll1Lo1Lu1Ll1Lu1Ll3Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu5Ll"
    "1Lu5Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu1Ll1Lu4Ll1Lu1Ll1Lu2Ll1Cn2Lu1Ll1Cn1Ll1Cn1Ll1Lu1Ll1Lu"
    "1Ll1Lu1Ll1Lu1Cn15Lm3Lu1Ll1Lo1Lm2Ll1Lo7Mn1Lo3Mn1Lo4Mn1Lo17Mc2Mn2Mc1So4Mn1Cn3No6So2Sc1So1Cn6Lo34Po"
    "4Cn8Mc2Lo32Mc10Mn2Cn8Po2NdaCn6Mn12Lo6Po3Lo1Po1Lo2Mn1NdaLo1cMn8Po2Lo17MnbMc2CnbPo1Lo1dCn3Mn3Mc1Lo"
    "2fMn1Mc2Mn4Mc2Mn2Mc3PodCn1Lm1NdaCn4Po2Lo5Mn1Lm1Lo9NdaLo5Cn1Lo29Mn6Mc2Mn2Mc2Mn2Cn9Lo3Mn1Lo8Mn1Mc"
    "1Cn2NdaCn2Po4Lo10Lm1Lo6So3Lo1Mc1Mn1Mc1Lo32Mn1Lo1Mn3Lo2Mn2Lo5Mn2Lo1Mn1Lo1Cn18Lo2Lm1Po2LobMc1Mn2Mc"
    "2Po2Lo1Lm2Mc1Mn1CnaLo6Cn2Lo6Cn2Lo6Cn9Lo7Cn1Lo7Cn1Ll2bSk1Lm4Ll9Lm1Sk2Cn4Ll50Lo23Mc2Mn1Mc2Mn1Mc2Po"
    "1Mc1Mn1Cn2NdaCn6Lo2ba4CncLo17Cn4Lo31Cn4Cs800Co1900Lo16eCn2Lo6aCn26Ll7CncLl5Cn5Lo1Mn1LoaSm1LodCn"
    "1Lo5Cn1Lo1Cn1Lo2Cn1Lo2Cn1Lo6cSk11Cn10Lo16bPe1Ps1So10Lo40Cn2Lo36Cn7So1Cn20LocSc1So3Mn10Po7Ps1Pe"
    "1Po1Cn6Mn10Po1Pd2Pc2Ps1Pe1Ps1Pe1Ps1Pe1Ps1Pe1Ps1Pe1Ps1Pe1Ps1Pe1Ps1Pe1Po2Ps1Pe1Po4Pc3Po3Cn1Po4Pd"
    "1Ps1Pe1Ps1Pe1Ps1Pe1Po3Sm1Pd1Sm3Cn1Po1Sc1Po2Cn4Lo5Cn1Lo87Cn2Cf1Cn1Po3Sc1Po3Ps1Pe1Po1Sm1Po1Pd1Po"
    "2NdaPo2Sm3Po2Lu1aPs1Po1Pe1Sk1Pc1Sk1Ll1aPs1Sm1Pe1Sm1Ps1Pe1Po1Ps1Pe1Po2LoaLm1Lo2dLm2Lo1fCn3Lo6Cn"
    "2Lo6Cn2Lo6Cn2Lo3Cn3Sc2Sm1Sk1So1Sc2Cn1So1Sm4So2CnaCf3So2Cn2LocCn1Lo1aCn1Lo13Cn1Lo2Cn1LofCn2LoeCn"
    "22Lo7bCn5Po3Cn4No2dCn3So9Nl35No4So11No2So3Cn1SodCn3So1Cn2fSo2dMn1Cn82Lo1dCn3Lo31CnfMn1No1bCn4Lo"
    "20No4Cn9Lo14Nl1Lo8Nl1Cn5Lo26Mn5Cn5Lo1eCn1Po1Lo24Cn4Lo8Po1Nl5Cn2aLu28Ll28Lo4eCn2NdaCn6Lu24Cn4Ll"
    "24Cn4Lo28Cn8Lo34CnbPo1LubCn1LufCn1Lu7Cn1Lu2Cn1LlbCn1LlfCn1Ll7Cn1Ll2Cn3Lo34CncLo137Cn9Lo16CnaLo"
    "8Cn18Lm6Cn1Lm2aCn1Lm9Cn45Lo6Cn2Lo1Cn1Lo2cCn1Lo2Cn3Lo1Cn2Lo17Cn1Po1No8Lo17So2No7Lo1fCn8No9Cn30Lo"
    "13Cn1Lo2Cn5No5Lo16No6Cn3Po1Lo1aCn5Po1Cn40Lo38Cn4No2Lo2No10Cn2No2eLo1Mn3Cn1Mn2Cn5Mn4Lo4Cn1Lo3Cn"
    "1Lo1dCn2Mn3Cn4Mn1No9Cn7Po9Cn7Lo1dNo2Po1Lo1dNo3Cn20Lo8So1Lo1cMn2Cn4No5Po7Cn9Lo36Cn3Po7Lo16Cn2No"
    "8Lo13Cn5No8Lo12Cn7Po4CncNo7Cn50Lo49Cn37Lu33CndLl33Cn7No6Lo24Mn4Cn8NdaCn6NdaLo4Lm1Lo1Lu16Cn3Mn5Pd"
    "1Lm1Ll16Cn8Sm2Cnd0No1fCn1Lo2aCn1Mn2Pd1Cn2Lo2Cn10Lo3Cn37Mn4Lo1dNoaLo1Cn8Lo16MnbNo4Po5Cn16Lo12Mn"
    "4Po4Cn26Lo15No7Cn14Lo17Cn9Mc1Mn1Mc1Lo35MnfPo7Cn4No14NdaMn1Lo2Mn2Lo1Cn9Mn3Mc1Lo2dMc3Mn4Mc2Mn2Po"
    "2Cf1Po4Mn1CnaCf1Cn2Lo19Cn7NdaCn6Mn3Lo24Mn5Mc1Mn8Cn1NdaPo4Lo1Mc2Lo1Cn8Lo23Mn1Po2Lo1Cn9Mn2Mc1Lo"
    "30Mc3Mn9Mc2Lo4Po4Mn4Po1Mc1Mn1NdaLo1Po1Lo1Po3Cn1No14CnbLo12Cn1Lo19Mc3Mn3Mc2Mn1Mc1Mn2Po6Mn1Lo2Mn"
    "1Cn3eLo7Cn1Lo1Cn1Lo4Cn1LofCn1LoaPo1Cn6Lo2fMn1Mc3Mn8Cn5NdaCn6Mn2Mc2Cn1Lo8Cn2Lo2Cn2Lo16Cn1Lo7Cn1Lo"
    "2Cn1Lo5Cn1Mn2Lo1Mc2Mn1Mc4Cn2Mc2Cn2Mc3Cn2Lo1Cn6Mc1Cn5Lo5Mc2Cn2Mn7Cn3Mn5CnbLoaCn1Lo1Cn2Lo1Cn1Lo"
    "26Cn1Lo1Mc3Mn6Cn1Mc1Cn2Mc1Cn1Mc4Cn1Mc2Mn1Mc1Mn1Lo1Mn1Lo1Po2Cn1Po2Cn8Mn2Cn1dLo35Mc3Mn8Mc2Mn3Mc1Mn"
    "1Lo4Po5NdaPo2Cn1Po1Mn1Lo3Cn1eLo30Mc3Mn6Mc1Mn1Mc4Mn2Mc1Mn2Lo2Po1Lo1Cn8NdaCna6Lo2fMc3Mn4Cn2Mc4Mn"
    "2Mc1Mn2Po17Lo4Mn2Cn22Lo30Mc3Mn8Mc2Mn1Mc1Mn2Po3Lo1CnbNdaCn6PodCn13Lo2bMn1Mc1Mn1Mc2Mn6Mc1Mn1Lo1Po"
    "1Cn6NdaCn6Nd14Cn1cLo1bCn2Mn1Mc1Mn1Mc2Mn4Mc1Mn5Cn4NdaNo2Po3So1Lo7Cnb9Lo2cMc3Mn9Mc1Mn2Po1Cn64Lu"
    "20Ll20NdaNo9CncLo8Cn2Lo1Cn2Lo8Cn1Lo2Cn1Lo18Mc6Cn1Mc2Cn2Mn2Mc1Mn1Lo1Mc1Lo1Mc1Mn1Po3Cn9NdaCn46Lo"
    "8Cn2Lo27Mc3Mn4Cn2Mn2Mc4Mn1Lo1Po1Lo1Mc1Cn1bLo1MnaLo28Mn6Mc1Lo1Mn4Po8Mn1Cn8Lo1Mn6Mc2Mn3Lo2eMndMc"
    "1Mn2Po3Lo1Po5CndLo49Cn7PoaCnb6Lo21Po1CneNdaCn6Lo9Cn1Lo25Mc1Mn7Cn1Mn6Mc1Mn1Lo1Po5CnaNdaNo13Cn3Po"
    "2Lo1eCn2Mn16Cn1Mc1Mn7Mc1Mn2Mc1Mn2Cn49Lo7Cn1Lo2Cn1Lo26Mn6Cn3Mn1Cn1Mn2Cn1Mn7Lo1Mn1Cn8NdaCn6Lo6Cn"
    "1Lo2Cn1Lo20Mc5Cn1Mn2Cn1Mc2Mn1Mc1Mn1Lo1Cn7NdaCn136Lo13Mn2Mc2Po2Cn7Mn2Lo1Mc1LodCn1Lo22Mc2Mn5Cn3Mc"
    "2Mn1Mc1Mn1PodNdaMn1Cn55Lo1CnfNo15So8Sc4So11CndPo1Lo39aCn66Nl6fCn1Po5CnbLoc4Cna4cLo61Po2CndLo"
    "430Cf10Mn1Lo6MnfCnaLof9bCn5Lo247Cn1ab9Lo1eMncMc3Mn3NdaCn6c6Lo239Cn7Lo1fCn1NdaCn4Po2Lo4fCn1NdaCn"
    "6Lo1eCn2Mn5Po1CnaLo30Mn7Po5So4Lm4Po1So1CnaNdaCn1No7Cn1Lo15Cn5Lo13Cn1b0Lm3Lo28Lm2Po3NdaCnc6Lu20Ll"
    "20No17Po4Cn65Lo4bCn4Mn1Lo1Mc37Cn7Mn4LmdCn40Lm2Po1Lm1Mn1CnbMc2CneLo17f8Cn8Lo4d6Cn29LoaCn22e7Lm4Cn"
    "1Lm7Cn1Lm2Cn1Lo123CnfLo1Cn1dLo3Cn2Lo1CneLo4Cn8Lo18cCn904Lo6bCn5LodCn3Lo9Cn7LoaCn2So1Mn2Po1Cf4Cn"
    "f5cSof0NdaCn6So1b4Cn4cMn2eCn2Mn17Cn9So74Cn3cSof6CnaSo27Cn2So3cMc2Mn3So3Mc6Cf8Mn8So2Mn7So1eMn4So"
    "3dCn15So42Mn3So1Cn7aNo14CncNo14CncSo57Cn9No19Cn87Lu1aLl1aLu1aLl7Cn1Ll12Lu1aLl1aLu1Cn1Lu2Cn2Lu1Cn"
    "2Lu2Cn2Lu4Cn1Lu8Ll4Cn1Ll1Cn1Ll7Cn1LlbLu1aLl1aLu2Cn1Lu4Cn2Lu8Cn1Lu7Cn1Ll1aLu2Cn1Lu4Cn1Lu5Cn1Lu1Cn"
    "3Lu7Cn1Ll1aLu1aLl1aLu1aLl1aLu1aLl1aLu1aLl1aLu1aLl1aLu1aLl1cCn2Lu19Sm1Ll19Sm1Ll6Lu19Sm1Ll19Sm1Ll"
    "6Lu19Sm1Ll19Sm1Ll6Lu19Sm1Ll19Sm1Ll6Lu19Sm1Ll19Sm1Ll6Lu1Ll1Cn2Nd32So200Mn37So4Mn32So8Mn1SoeMn1So"
    "2Po5CnfMn5Cn1MnfCn450LlaLo1Ll14Cn6Ll6Cnd5Mn7Cn1Mn11Cn2Mn7Cn1Mn2Cn1Mn5Cn5Lm3eCn21Mn1Cn70Lo2dCn3Mn"
    "7Lm7Cn2NdaCn4Lo1So1Cn140Lo1eMn1Cn11Lo2cMn4NdaCn5Sc1Cn1d0Lo1bLm1Mn4NdaCnd6Lo1eMn2Lo1NdaCn4Po1Cn"
    "1e0Lo7Cn1Lo4Cn1Lo2Cn1LofCn1Loc5Cn2No9Mn7Cn29Lu22Ll22Mn7Lm1Cn4NdaCn4Po2Cn311No3bSo1No3Sc1No4Cn"
    "4cNo2dSo1NofCnc2Lo4Cn1Lo1bCn1Lo2Cn1Lo1Cn2Lo1Cn1LoaCn1Lo4Cn1Lo1Cn1Lo1Cn6Lo1Cn4Lo1Cn1Lo1Cn1Lo1Cn"
    "1Lo3Cn1Lo2Cn1Lo1Cn2Lo1Cn1Lo1Cn1Lo1Cn1Lo1Cn1Lo1Cn1Lo2Cn1Lo1Cn2Lo4Cn1Lo7Cn1Lo4Cn1Lo4Cn1Lo1Cn1LoaCn"
    "1Lo11Cn5Lo3Cn1Lo5Cn1Lo11Cn34Sm2Cn10eSo2cCn4So64CncSofCn2SofCn1SofCn1So25CnaNodSoa1Cn38So1dCndSo"
    "2cCn4So9Cn7So2CneSo6Cn9aSofbSk5So2d8Cn4So11Cn3SodCn3So77Cn4So5fCn6SocCn4So1CnfSocCn4So38Cn8SoaCn"
    "6So28Cn8So1eCn2SocCn4So2Cn3eSo154CncSoeCn2SodCn3SoaCn5So38Cn7SofCn2SobCn6So9Cn7So93Cn1So5cNdaCn"
    "406Loa6e0Cn20Lo103aCn6LodeCn2Lo1682CneLo1d31CnfLo26eCn9a2Lo21eCn5e2Lo134bCn5Lo1060Cnadc51Cf1Cn"
    "1eCf60Cn80Mnf0Cnfe10CofffeCn2CofffeCn"
)
NUMERIC17 = (
    "30-39 b2-b3 b9 bc-be 660-669 6f0-6f9 7c0-7c9 966-96f 9e6-9ef 9f4-9f9 a66-a6f ae6-aef b66-b6f "
    "b72-b77 be6-bf2 c66-c6f c78-c7e ce6-cef d58-d5e d66-d78 de6-def e50-e59 ed0-ed9 f20-f33 "
    "1040-1049 1090-1099 1369-137c 16ee-16f0 17e0-17e9 17f0-17f9 1810-1819 1946-194f 19d0-19da "
    "1a80-1a89 1a90-1a99 1b50-1b59 1bb0-1bb9 1c40-1c49 1c50-1c59 2070 2074-2079 2080-2089 2150-2182 "
    "2185-2189 2460-249b 24ea-24ff 2776-2793 2cfd 3007 3021-3029 3038-303a 3192-3195 3220-3229 "
    "3248-324f 3251-325f 3280-3289 32b1-32bf a620-a629 a6e6-a6ef a830-a835 a8d0-a8d9 a900-a909 "
    "a9d0-a9d9 a9f0-a9f9 aa50-aa59 abf0-abf9 ff10-ff19 10107-10133 10140-10178 1018a-1018b "
    "102e1-102fb 10320-10323 10341 1034a 103d1-103d5 104a0-104a9 10858-1085f 10879-1087f 108a7-108af "
    "108fb-108ff 10916-1091b 109bc-109bd 109c0-109cf 109d2-109ff 10a40-10a48 10a7d-10a7e 10a9d-10a9f "
    "10aeb-10aef 10b58-10b5f 10b78-10b7f 10ba9-10baf 10cfa-10cff 10d30-10d39 10d40-10d49 10e60-10e7e "
    "10f1d-10f26 10f51-10f54 10fc5-10fcb 11052-1106f 110f0-110f9 11136-1113f 111d0-111d9 111e1-111f4 "
    "112f0-112f9 11450-11459 114d0-114d9 11650-11659 116c0-116c9 116d0-116e3 11730-1173b 118e0-118f2 "
    "11950-11959 11bf0-11bf9 11c50-11c6c 11d50-11d59 11da0-11da9 11de0-11de9 11f50-11f59 11fc0-11fd4 "
    "12400-1246e 16130-16139 16a60-16a69 16ac0-16ac9 16b50-16b59 16b5b-16b61 16d70-16d79 16e80-16e96 "
    "16ff4-16ff6 1ccf0-1ccf9 1d2c0-1d2d3 1d2e0-1d2f3 1d360-1d378 1d7ce-1d7ff 1e140-1e149 1e2f0-1e2f9 "
    "1e4f0-1e4f9 1e5f1-1e5fa 1e8c7-1e8cf 1e950-1e959 1ec71-1ecab 1ecad-1ecaf 1ecb1-1ecb4 1ed01-1ed2d "
    "1ed2f-1ed3d 1f100-1f10c 1fbf0-1fbf9 "
)
NF_INERT = (
    "7fd 897-89f 8ca-8d3 9fe c3c d3b-d3c eba 1715 1abf-1add 1ae0-1aeb 1df6-1dfa 32ff a7f1-a7f4 a82c "
    "ab69 105c9 105d2 105da 105e4 10781-10785 10787-107b0 107b2-107ba 10d24-10d27 10d69-10d6d "
    "10eab-10eac 10efa-10efb 10efd-10eff 10f46-10f50 10f82-10f85 11070 1133b 11382-11385 1138b 1138e "
    "11390-11391 113b8 113bb 113c2 113c5 113c7-113c9 113ce-113d0 1145e 11839-1183a 11930 11935 11938 "
    "1193d-1193e 11943 119e0 11a34 11a47 11a99 11d42 11d44-11d45 11d97 11f41-11f42 1611e-16129 1612f "
    "16d63 16d67-16d6a 16ff0-16ff1 1ccd6-1ccf9 1e030-1e06d 1e08f 1e130-1e136 1e2ae 1e2ec-1e2ef "
    "1e4ec-1e4ef 1e5ee-1e5ef 1e6e3 1e6e6 1e6ee-1e6ef 1e6f5 1f16c 1fbf0-1fbf9 "
)
LOWER17 = (
    "1c89:1c8a 2c2f:2c5f a7c0:a7c1 a7cb:264 a7cc:a7cd a7ce:a7cf a7d0:a7d1 a7d2:a7d3 a7d4:a7d5 "
    "a7d6:a7d7 a7d8:a7d9 a7da:a7db a7dc:19b 10570:10597 10571:10598 10572:10599 10573:1059a "
    "10574:1059b 10575:1059c 10576:1059d 10577:1059e 10578:1059f 10579:105a0 1057a:105a1 1057c:105a3 "
    "1057d:105a4 1057e:105a5 1057f:105a6 10580:105a7 10581:105a8 10582:105a9 10583:105aa 10584:105ab "
    "10585:105ac 10586:105ad 10587:105ae 10588:105af 10589:105b0 1058a:105b1 1058c:105b3 1058d:105b4 "
    "1058e:105b5 1058f:105b6 10590:105b7 10591:105b8 10592:105b9 10594:105bb 10595:105bc 10d50:10d70 "
    "10d51:10d71 10d52:10d72 10d53:10d73 10d54:10d74 10d55:10d75 10d56:10d76 10d57:10d77 10d58:10d78 "
    "10d59:10d79 10d5a:10d7a 10d5b:10d7b 10d5c:10d7c 10d5d:10d7d 10d5e:10d7e 10d5f:10d7f 10d60:10d80 "
    "10d61:10d81 10d62:10d82 10d63:10d83 10d64:10d84 10d65:10d85 16ea0:16ebb 16ea1:16ebc 16ea2:16ebd "
    "16ea3:16ebe 16ea4:16ebf 16ea5:16ec0 16ea6:16ec1 16ea7:16ec2 16ea8:16ec3 16ea9:16ec4 16eaa:16ec5 "
    "16eab:16ec6 16eac:16ec7 16ead:16ec8 16eae:16ec9 16eaf:16eca 16eb0:16ecb 16eb1:16ecc 16eb2:16ecd "
    "16eb3:16ece 16eb4:16ecf 16eb5:16ed0 16eb6:16ed1 16eb7:16ed2 16eb8:16ed3 "
)


def _unicode_tables(ucd16: str, ucd17: str) -> str:
    """The tables above from the Unicode Character Database:
    GC16_RUNS    general categories, Unicode 16.0 (Oniguruma's \\p{..} in tokenizers
                 0.23): runs as <hex delta from the previous start><category>
    NUMERIC17    N* in Unicode 17.0 (Rust's char::is_numeric: Digits)
    NF_INERT     code points assigned after Unicode 9.0 (the reference's
                 normalization data) that normalization could touch: a canonical
                 combining class, a decomposition, or part of one
    LOWER17      the simple lowercase mappings of code points assigned after
                 Unicode 13.0 (the board's Python): <cp>:<lowercase>"""
    def ranges_file(path: str) -> Iterable[Tuple[int, int, str]]:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.split("#")[0].strip()
                if line:
                    r, v = [x.strip() for x in line.split(";")[:2]]
                    a, _, b = r.partition("..")
                    yield int(a, 16), int(b or a, 16), v

    def fmt(items: List[str], sep: str) -> List[str]:
        out, cur = [], ""
        for it in items:
            if cur and len(cur) + len(it) + len(sep) > 96:
                out.append(f'    "{cur}"')
                cur = ""
            cur += it + sep
        return out + ([f'    "{cur}"'] if cur else [])

    def as_ranges(cps: Iterable[int]) -> List[str]:
        out, start, prev = [], None, None
        for cp in sorted(cps):
            if start is not None and cp == prev + 1:
                prev = cp
                continue
            if start is not None:
                out.append("%x" % start if start == prev else "%x-%x" % (start, prev))
            start = prev = cp
        if start is not None:
            out.append("%x" % start if start == prev else "%x-%x" % (start, prev))
        return out

    gc = ["Cn"] * (MAX_CP + 1)
    for a, b, v in ranges_file(os.path.join(ucd16, "extracted", "DerivedGeneralCategory.txt")):
        gc[a:b + 1] = [v] * (b - a + 1)
    runs, prev_start, prev_cat = [], 0, None
    for cp in range(MAX_CP + 1):
        if gc[cp] != prev_cat:
            runs.append("%x%s" % (cp - prev_start, gc[cp]))
            prev_start, prev_cat = cp, gc[cp]
    numeric = [cp for a, b, v in ranges_file(os.path.join(ucd17, "extracted", "DerivedGeneralCategory.txt"))
               if v[0] == "N" for cp in range(a, b + 1)]
    age: Dict[int, Tuple[int, ...]] = {}
    for a, b, v in ranges_file(os.path.join(ucd17, "DerivedAge.txt")):
        for cp in range(a, b + 1):
            age[cp] = tuple(int(x) for x in v.split("."))
    ccc, dec, lower = {}, {}, {}
    with open(os.path.join(ucd17, "UnicodeData.txt"), encoding="utf-8") as f:
        for line in f:
            fld = line.rstrip("\n").split(";")
            cp = int(fld[0], 16)
            ccc[cp] = int(fld[3])
            if fld[5]:
                dec[cp] = fld[5]
            if fld[13]:
                lower[cp] = int(fld[13], 16)
    parts = {int(x, 16) for d in dec.values() if not d.startswith("<") for x in d.split()}
    inert = [cp for cp, v in age.items() if v > (9, 0) and (ccc.get(cp) or cp in dec or cp in parts)]
    low = ["%x:%x" % (cp, lower[cp]) for cp in sorted(lower) if age.get(cp, (0,)) > (13, 0)]
    lines = (["GC16_RUNS = ("] + fmt(runs, "") + [")", "NUMERIC17 = ("] + fmt(as_ranges(numeric), " ")
             + [")", "NF_INERT = ("] + fmt(as_ranges(inert), " ") + [")", "LOWER17 = ("] + fmt(low, " ") + [")"])
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 4 and sys.argv[1] == "--unicode-tables":
        print(_unicode_tables(sys.argv[2], sys.argv[3]))
    elif len(sys.argv) >= 3:
        t = Tokenizer(sys.argv[1])
        ids = t.encode(" ".join(sys.argv[2:]))
        print(ids)
        print([t.id_to_token[i] for i in ids])
        print(repr(t.decode(ids)))
    else:
        print("usage: hf_tokenizer.py tokenizer.json TEXT...  |  --unicode-tables UCD16_DIR UCD17_DIR",
              file=sys.stderr)
        sys.exit(2)
