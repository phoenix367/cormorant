# Copyright 2026 Ivan Gubochkin
# SPDX-License-Identifier: Apache-2.0

"""hf_tokenizer.py, the tokenizer.json interpreter every chat backend uses:
each pipeline part on small synthetic tokenizer.json dicts (the expected
values are what the `tokenizers` library 0.23 gives for the same dicts:
scripts/validate_text.py --synthetic / --regex run that comparison at large),
the refusals, the Unicode tables, the compatibility shims, and the pinned
tokenizers of other model families against the reference ids in
data/hf_tokenizer_cases.json (written by validate_text.py --all
--write-fixture; skipped when demo/chat/assets/tokenizers/ is absent —
validate_text.py --fetch)."""

import hashlib
import json
import os
import random
import tempfile
import unicodedata
import unittest

from _util import CHAT

import hf_tokenizer as H
import llama_tokenizer
import smollm2_tokenizer
from hf_tokenizer import Tokenizer

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "hf_tokenizer_cases.json")
QWEN = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+"
        r"|\s+(?!\S)|\s+")
LLAMA3 = QWEN.replace(r"|\p{N}|", r"|\p{N}{1,3}|")


def vocab():
    """A small BPE vocabulary: specials, <0xXX> bytes, ASCII, the byte-level
    alphabet, ##x pieces, x</w> ends and the merges of a few words."""
    toks = (["<unk>", "<s>", "</s>", "▁", "Ġ"] + [f"<0x{b:02X}>" for b in range(256)]
            + [chr(c) for c in range(0x21, 0x7F)] + list(H.bytes_to_unicode().values())
            + ["##" + c for c in "abcdefghijklmnopqrstuvwxyz"] + ["o</w>", "d</w>"])
    v = {}
    for t in toks:
        v.setdefault(t, len(v))
    merges, seen = [], set()
    for w in ("▁hello", "▁world", "▁the", "▁there", "hello", "Ġhello", "Ġworld", "ĠĠ", "▁▁"):
        cur = w[0]
        for ch in w[1:]:
            if (cur, ch) not in seen:
                seen.add((cur, ch))
                merges.append([cur, ch])
                v.setdefault(cur + ch, len(v))
            cur += ch
    return v, merges


def spec(norm=None, pre=None, dec=None, post=None, added=(), **model):
    v, merges = vocab()
    m = {"type": "BPE", "dropout": None, "unk_token": "<unk>", "continuing_subword_prefix": None,
         "end_of_word_suffix": None, "fuse_unk": True, "byte_fallback": True, "ignore_merges": False,
         "vocab": v, "merges": merges}
    m.update(model)
    return {"version": "1.0", "truncation": None, "padding": None, "added_tokens": list(added),
            "normalizer": norm, "pre_tokenizer": pre, "post_processor": post, "decoder": dec, "model": m}


def at(content, i, special=True, normalized=False, lstrip=False, rstrip=False, single_word=False):
    return {"id": i, "content": content, "single_word": single_word, "lstrip": lstrip, "rstrip": rstrip,
            "normalized": normalized, "special": special}


SP_NORM = {"type": "Sequence", "normalizers": [{"type": "Prepend", "prepend": "▁"},
                                               {"type": "Replace", "pattern": {"String": " "}, "content": "▁"}]}
LLAMA_DEC = {"type": "Sequence", "decoders": [
    {"type": "Replace", "pattern": {"String": "▁"}, "content": " "}, {"type": "ByteFallback"}, {"type": "Fuse"},
    {"type": "Strip", "content": " ", "start": 1, "stop": 0}]}
BOS = {"type": "TemplateProcessing", "single": [{"SpecialToken": {"id": "<s>", "type_id": 0}},
                                                {"Sequence": {"id": "A", "type_id": 0}}],
       "pair": [], "special_tokens": {"<s>": {"id": "<s>", "ids": [1], "tokens": ["<s>"]}}}
BL = {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": True}


def pre_split(pre, text):
    pieces, at0 = [text], True
    for stage in H._pre_tokenizer(pre):
        pieces, at0 = stage(pieces, at0)
    return pieces


def dec_chain(dec, tokens, cont=False):
    steps = H._decoder(dec)
    fused = False
    for st in steps:
        st.fused_in, fused = fused, fused or st.fuses
    for st in steps:
        tokens = st.batch(tokens, cont)
    return "".join(tokens)


class TestRegex(unittest.TestCase):
    """onig_to_re: Oniguruma patterns as tokenizers' Split runs them."""

    def split(self, pattern, text):
        return [p for p in H._onig_regex(pattern, "test").parts(text) if p]

    def test_qwen_and_llama3(self):
        self.assertEqual(self.split(QWEN, "IT'S it's 'ſt 'İ 'K 12345"),
                         ["IT", "'S", " it", "'s", " '", "ſt", " '", "İ", " '", "K", " ", "1", "2", "3", "4", "5"])
        self.assertEqual(self.split(QWEN, "héllo\r\n\r\n  wörld!!\n"),
                         ["héllo", "\r\n\r\n", " ", " wörld", "!!\n"])
        self.assertEqual(self.split(QWEN, "x\u0301y ǅa 中文 😀👍🏽"), ["x", "\u0301y", " ǅa", " 中文", " 😀👍🏽"])
        self.assertEqual(self.split(LLAMA3, "1234567 ٣٤٥٦"), ["123", "456", "7", " ", "٣٤٥", "٦"])
        self.assertEqual(self.split(LLAMA3, "a  b\t\tc   "), ["a", " ", " b", "\t", "\tc", "   "])

    def test_case_folding(self):
        # Oniguruma folds s ~ ſ and k ~ K (Kelvin), not i ~ İ / ı (Python's re would)
        ci = H._onig_regex(r"(?i:'s|'k|'i)", "test")
        for s, hit in (("'S", True), ("'ſ", True), ("'\u212a", True), ("'İ", False), ("'ı", False), ("'I", True)):
            self.assertEqual(bool(ci.rx.fullmatch(s)), hit, s)
        with self.assertRaisesRegex(ValueError, "non-ASCII letter"):
            H.onig_to_re("(?i:é)")
        H.onig_to_re(r"(?i:\s+[a-z])")                      # classes without other letters are fine

    def test_unicode_16_classes(self):
        self.assertEqual(self.split(r"\p{L}+", "a\U00010D4Ab \u1C89"), ["a\U00010D4Ab", " ", "\u1C89"])
        self.assertEqual(self.split(r"\p{Nd}", "\U00010D40x"), ["\U00010D40", "x"])          # Garay digit (16.0)
        self.assertEqual(self.split(r"\s+", "a\x1c\u00a0\u3000b"), ["a\x1c", "\u00a0\u3000", "b"])

    def test_empty_matches(self):
        # after an empty match the onig crate moves on one character
        self.assertEqual(self.split(r"[a-z]{,2}", "  my box"), [" ", " ", "my", " ", "bo", "x"])
        self.assertEqual(H._onig_regex(r"\s{2,}|x*", "test").sub("a  bxxc d", "_"), "_a_b_c_ _d_")

    def test_ascii_cut(self):
        rx = H._onig_regex(QWEN, "test")
        self.assertIsNotNone(rx._ascii)
        for s in ("it's 12 ab", "IT'S\r\n  x", "naïve 12"):
            self.assertEqual(rx.findall(s), rx.rx.findall(s))

    def test_refusals(self):
        for pat, what in ((r"\w+", r"\\w"), (r"\bx", r"\\b"), (r"(a)\1", "back-reference"),
                          (r"(?>ab)", "atomic"), (r"a++", "possessive"), (r"\p{Han}", "property"),
                          (r"(?m:.)", "option"), (r"[a&&b]", "intersection"), (r"[[:alpha:]]", "nested")):
            with self.assertRaisesRegex(ValueError, what):
                H.onig_to_re(pat)


class TestNormalizers(unittest.TestCase):
    CASES = {
        "NFC": ("e\u0301 a\u0323\u0302 \u1100\u1161\u11a8 \U00011935\U00011930 a\u0345\u1df6\u0301",
                "é ậ 각 \U00011935\U00011930 a\u0345\u1df6\u0301"),
        "NFKC": ("ﬁ ①Ⅻ ｶﾞ \u32ff \U0001fbf3 ²", "fi 1XII ガ \u32ff \U0001fbf3 2"),
        "NFD": ("é ǅ \U00011938", "e\u0301 ǅ \U00011938"),
        "Lowercase": ("ΣΑΣ İ ẞ \U00010570\u1C89\uA7CB ǅ Ω", "σασ i\u0307 ß \U00010597\u1C8A\u0264 ǆ ω"),
    }

    def test_unicode(self):
        # the reference's normalization data is Unicode 9.0: later characters stay as they are
        for kind, (text, want) in self.CASES.items():
            with self.subTest(kind=kind):
                self.assertEqual(H._normalizer({"type": kind})(text), want)

    def test_strip_prepend_replace_sequence(self):
        norm = H._normalizer
        self.assertEqual(norm({"type": "Strip", "strip_left": True, "strip_right": True})("\u3000\t x \x1c\u00a0"),
                         "x \x1c")                                       # \x1c is not White_Space
        self.assertEqual(norm({"type": "Strip", "strip_left": True, "strip_right": False})("  x  "), "x  ")
        self.assertEqual(norm({"type": "Prepend", "prepend": "▁"})(""), "")
        self.assertEqual(norm({"type": "Replace", "pattern": {"Regex": r"\s{2,}|x*"}, "content": "_"})("a  bxxc d"),
                         "_a_b_c_ _d_")
        self.assertEqual(norm({"type": "Sequence", "normalizers": [{"type": "NFKC"}, SP_NORM]})("ﬁx y"), "▁fix▁y")


class TestPreTokenizers(unittest.TestCase):

    def test_split_behaviors(self):
        want = {("Isolated", False): ["a", "1", "  ", "b", "2", "2", " ", "c", " ", "3"],
                ("Removed", False): ["a", "b", "c"],
                ("Removed", True): ["1", "  ", "2", "2", " ", " ", "3"],
                ("MergedWithPrevious", False): ["a1", "  ", "b2", "2", " ", "c ", "3"],
                ("MergedWithPrevious", True): ["a", "1", "  b", "2", "2", " c", " ", "3"],
                ("MergedWithNext", False): ["a", "1", "  b", "2", "2", " c", " ", "3"],
                ("MergedWithNext", True): ["a1", "  ", "b2", "2", " ", "c ", "3"],
                ("Contiguous", False): ["a", "1  ", "b", "22 ", "c", " 3"]}
        for (beh, inv), pieces in want.items():
            with self.subTest(behavior=beh, invert=inv):
                pre = {"type": "Split", "pattern": {"Regex": r"\s+|\p{N}"}, "behavior": beh, "invert": inv}
                self.assertEqual(pre_split(pre, "a1  b22 c 3"), pieces)

    def test_byte_level_and_digits(self):
        self.assertEqual(pre_split(BL, "Hello  world 2024!"), ["Hello", "Ġ", "Ġworld", "Ġ2024", "!"])
        self.assertEqual(pre_split(BL, "naïve\u00a0x\u3000y\x1c"), ["naÃ¯ve", "Âł", "x", "ãĢĢ", "y", "Ĝ"])
        self.assertEqual(pre_split({"type": "Digits", "individual_digits": True}, "in 2024, ٣ and \U00011DE0 x"),
                         ["in ", "2", "0", "2", "4", ", ", "٣", " and ", "\U00011DE0", " x"])   # 17.0's N
        self.assertEqual(pre_split({"type": "Digits", "individual_digits": False}, "in 2024, ٣٤ x9"),
                         ["in ", "2024", ", ", "٣٤", " x", "9"])

    def test_metaspace(self):
        first = {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "first", "split": True}
        self.assertEqual(pre_split(first, "hello  big world"), ["▁hello", "▁", "▁big", "▁world"])
        self.assertEqual(pre_split(first, " x"), ["▁x"])
        self.assertEqual(pre_split({"type": "Metaspace", "replacement": "▁", "prepend_scheme": "never",
                                    "split": False}, "hello world"), ["hello▁world"])
        # "first": only the piece at the start of the text (none after an added token)
        tok = Tokenizer(spec(pre=first, added=[at("<s>", 1)]))
        self.assertEqual(tok.tokens("<s>hello"), ["<s>", "hello"])
        self.assertEqual(tok.tokens("hello <s> hello"), ["▁hello", "▁", "<s>", "▁hello"])


class TestDecoders(unittest.TestCase):

    def test_chains(self):
        self.assertEqual(dec_chain(LLAMA_DEC, ["▁Hello", "<0xC3>", "<0xA9>", "▁", "<0xF0>", "<0x9F>", "▁x"]),
                         "Helloé \ufffd\ufffd x")                      # a broken run: one U+FFFD per byte
        self.assertEqual(dec_chain(LLAMA_DEC, ["▁▁two"]), " two")
        self.assertEqual(dec_chain(LLAMA_DEC, ["▁▁two"], cont=True), "  two")
        ms = {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": True}
        self.assertEqual(dec_chain(ms, ["▁a▁b", "▁c"]), "ab c")         # every ▁ of the first token
        self.assertEqual(dec_chain(ms, ["▁a▁b", "▁c"], cont=True), " a b c")
        self.assertEqual(dec_chain(BL, ["Hello", "Ġw", "Ã", "©", "ð", "Ł"]), "Hello wé\ufffd")
        self.assertEqual(dec_chain(BL, ["中文", "Ġ<|x|>"]), "中文 <|x|>")
        self.assertEqual(dec_chain({"type": "BPEDecoder", "suffix": "</w>"}, ["he", "llo</w>", "wor", "ld</w>"]),
                         "hello world")
        self.assertEqual(dec_chain({"type": "Strip", "content": "x", "start": 2, "stop": 1}, ["xxxaxx", "xbx", "xyz"]),
                         "xaxbyz")
        self.assertEqual(dec_chain({"type": "Replace", "pattern": {"Regex": "l+"}, "content": "L"},
                                   ["hello", "all", "x"]), "heLoaLx")
        self.assertEqual(dec_chain(None, ["a", "b"]), "a b")
        self.assertEqual(dec_chain({"type": "Sequence", "decoders": []}, ["a", "b"]), "ab")

    def test_incremental_every_split(self):
        rng = random.Random(1)
        for dec in (LLAMA_DEC, BL, {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always"},
                    {"type": "Sequence", "decoders": [{"type": "Fuse"},
                                                      {"type": "Strip", "content": "▁", "start": 2, "stop": 2}]},
                    {"type": "BPEDecoder", "suffix": "</w>"}, None):
            tok = Tokenizer(spec(dec=dec, added=[at("<s>", 1)]))
            for _ in range(60):
                ids = [rng.choice([1, 3, 4, 5, 200, 201, 233, 189, 178, 325, 326, 548, 556]) for _ in range(12)]
                for strip in (True, False):
                    want = tok.decode(ids, strip_first=strip)
                    for cut in range(len(ids) + 1):
                        d = tok.incremental_decoder(strip_first=strip)
                        self.assertEqual(d.add_many(ids[:cut]) + d.add_many(ids[cut:]) + d.flush(), want,
                                         (dec, ids, cut))

    def test_pending_and_token_bytes(self):
        tok = Tokenizer(spec(norm=SP_NORM, dec=LLAMA_DEC))
        d = tok.incremental_decoder()
        self.assertEqual(d.add(tok.vocab["<0xF0>"]), "")
        self.assertTrue(d.pending)
        self.assertEqual(tok.token_bytes(tok.vocab["<0x0A>"]), b"\n")
        self.assertEqual(tok.token_bytes(tok.vocab["▁hello"]), b" hello")
        self.assertEqual(tok.token_bytes(10 ** 6), b"")
        bl = Tokenizer(spec(pre=BL, dec=BL))
        self.assertEqual(bl.token_bytes(bl.vocab["Ġhello"]), b" hello")
        self.assertEqual(bl.token_bytes(bl.vocab["Ã"]), b"\xc3")


class TestTokenizer(unittest.TestCase):
    """Whole pipelines: ids as tokenizers gives them (ids, without the
    post-processor, special-token strings as text) and the decodings."""

    CASES = [
        ("llama normalized", spec(norm=SP_NORM, dec=LLAMA_DEC, post=BOS, added=[
            at("<unk>", 0, normalized=True), at("<s>", 1, normalized=True), at("</s>", 2, normalized=True)]), [
            ("a <s>b", [1, 3, 325, 1, 326], [3, 325, 1, 326], [1, 3, 325, 3, 288, 343, 290, 326], "ab",
             "<s> a <s>b"),
            ("a<s>b", [1, 3, 325, 288, 343, 290, 326], None, None, "a<s>b", "<s> a<s>b"),
            ("hello world é😀\x00", [1, 548, 553, 3, 426, 245, 164, 157, 133, 5], None, None,
             "hello world é😀\x00", None),
            ("the there", [1, 556, 558], [556, 558], None, "the there", "<s> the there")]),
        ("raw flags", spec(norm=SP_NORM, dec=LLAMA_DEC, post=BOS, added=[
            at("<s>", 1), at("</s>", 2), at("<|user|>", 7777, rstrip=True), at("<|end|>", 7778, lstrip=True, rstrip=True),
            at("<mask>", 7779, special=False, lstrip=True), at("hi", 7780, special=False, single_word=True)]), [
            ("<|user|>\n  hello<|end|>  \n x", [1, 575, 548, 576, 3, 348], [575, 548, 576, 3, 348],
             [1, 3, 288, 352, 345, 343, 329, 342, 352, 290, 15, 3, 548, 288, 352, 329, 338, 328, 352, 290, 574, 15, 3, 348],
             "hello x", "<s><|user|> hello<|end|> x"),
            ("a  <mask>b", [1, 3, 325, 577, 3, 326], None, None, "a<mask> b", "<s> a<mask> b"),
            ("hi high ahi hi_ hi.", [1, 578, 3, 544, 333, 331, 332, 3, 325, 332, 333, 544, 333, 323, 3, 578, 3, 274],
             None, None, "hi  high ahi hi_ hi .", None)]),
        ("byte-level prefix per piece", spec(pre=dict(BL, add_prefix_space=True), dec=BL, unk_token=None,
                                             byte_fallback=False, added=[at("<s>", 1), at("<tool>", 9999, special=False)]), [
            ("a<s>a b", [4, 325, 1, 4, 325, 4, 326], None, [4, 325, 288, 343, 290, 325, 4, 326], " a a b", " a<s> a b"),
            ("hello <tool>world", [4, 562, 4, 575, 572], None, None, " hello <tool> world", None)]),
        ("unk fused", spec(byte_fallback=False), [("a中中b€", [325, 0, 326, 0], None, None, "a <unk> b <unk>", None)]),
        ("unk not fused", spec(byte_fallback=False, fuse_unk=False), [("a中中b€", [325, 0, 0, 326, 0], None, None, None, None)]),
        ("byte fallback with a missing byte", spec(dec=LLAMA_DEC, merges=[],
                                                    vocab={k: v for k, v in vocab()[0].items() if k != "<0xAC>"}), [
            ("x€中y", [348, 233, 189, 178, 0, 349], None, None, "x中<unk>y", None)]),   # the pending unk after the bytes
        ("ignore_merges", spec(pre=BL, dec=BL, ignore_merges=True, unk_token=None, byte_fallback=False,
                               vocab=dict(vocab()[0], **{"Ġhxl": len(vocab()[0])})), [
            ("hello hxl", [562, 575], None, None, "hello hxl", None)]),
        ("merges", spec(pre=BL, dec=BL, unk_token=None, byte_fallback=False,
                        vocab=dict(vocab()[0], **{"Ġhxl": len(vocab()[0])})), [
            ("hello hxl", [562, 563, 348, 336], None, None, "hello hxl", None)]),
        ("continuing_subword_prefix", spec(continuing_subword_prefix="##", byte_fallback=False, merges=[["h", "##e"]],
                                           vocab=dict(vocab()[0], he=len(vocab()[0])),
                                           pre={"type": "Split", "pattern": {"String": " "}, "behavior": "Removed",
                                                "invert": False}), [
            ("hello he", [575, 527, 527, 530, 575], None, None, "he ##l ##l ##o he", None)]),
        ("end_of_word_suffix", spec(end_of_word_suffix="</w>", byte_fallback=False, merges=[],
                                    dec={"type": "BPEDecoder", "suffix": "</w>"},
                                    pre={"type": "Split", "pattern": {"String": " "}, "behavior": "Removed",
                                         "invert": False}), [
            ("hello wod", [332, 329, 336, 336, 542, 347, 339, 543], None, None, "hello wod", None)]),
        ("RobertaProcessing", spec(pre=BL, dec=BL, unk_token=None, byte_fallback=False, added=[at("<s>", 1), at("</s>", 2)],
                                   post={"type": "RobertaProcessing", "sep": ["</s>", 2], "cls": ["<s>", 1],
                                         "trim_offsets": True, "add_prefix_space": False}), [
            ("hello", [1, 562, 2], [562], None, "hello", "<s>hello</s>")]),
    ]

    def test_cases(self):
        for name, sp, rows in self.CASES:
            tok = Tokenizer(sp)
            for text, ids, plain, as_text, dec, dec_all in rows:
                with self.subTest(pipeline=name, text=text):
                    self.assertEqual(tok.encode(text), ids)
                    if plain is not None:
                        self.assertEqual(tok.encode(text, add_special_tokens=False), plain)
                    if as_text is not None:
                        self.assertEqual(tok.encode(text, special=False), as_text)
                    if dec is not None:
                        self.assertEqual(tok.decode(ids), dec)
                    if dec_all is not None:
                        self.assertEqual(tok.decode(ids, skip_special_tokens=False), dec_all)

    def test_added_tokens(self):
        tok = Tokenizer(spec(norm=SP_NORM, added=[at("<s>", 1, normalized=True), at("<new>", 9000),
                                                  at("<also>", 42, special=False)]))
        # an id from the vocabulary, else the next one after it (the file's id is not used)
        n = len(vocab()[0])
        self.assertEqual(tok.added, {"<s>": 1, "<new>": n, "<also>": n + 1})
        self.assertEqual((tok.special, tok.special_ids), ({"<s>": 1, "<new>": n}, frozenset((1, n))))
        self.assertEqual(tok.id_to_token[1], "▁<s>")                     # the normalized form decodes
        self.assertEqual(tok.vocab_size, n + 2)
        self.assertEqual(tok.token_to_id("<also>"), n + 1)
        self.assertIsNone(tok.bos_id)
        self.assertEqual(Tokenizer(spec(post=BOS, added=[at("<s>", 1)])).bos_id, 1)   # the template's
        cfg = Tokenizer(spec(added=[at("<s>", 1), at("</s>", 2)]), config={"bos_token": "<s>", "eos_token": "</s>"})
        self.assertEqual((cfg.bos_id, cfg.eos_id), (1, 2))

    def test_long_words_heap_equals_scan(self):
        tok = Tokenizer(spec(norm=SP_NORM, dec=LLAMA_DEC))
        rng = random.Random(4)
        for _ in range(50):
            word = "".join(rng.choice(["▁hello", "▁the", "re", "▁▁", "x", "l", "o"]) for _ in range(rng.randint(5, 30)))
            sym = tok.model._symbols(word)
            scan, H._BPE.NAIVE = H._BPE.NAIVE, 10 ** 9
            try:
                want = tok.model._merge(list(sym))
            finally:
                H._BPE.NAIVE = scan
            self.assertEqual(tok.model._merge(list(sym)), want)

    def test_refusals(self):
        bad = [({"model": {"type": "WordPiece", "vocab": {}}}, "model 'WordPiece'"),
               ({"model": {"type": "Unigram", "vocab": []}}, "model 'Unigram'"),
               (dict(spec(), normalizer={"type": "Precompiled"}), "normalizer 'Precompiled'"),
               (dict(spec(), pre_tokenizer={"type": "BertPreTokenizer"}), "pre-tokenizer 'BertPreTokenizer'"),
               (dict(spec(), decoder={"type": "CTC"}), "decoder 'CTC'"),
               (dict(spec(), post_processor={"type": "Custom"}), "post-processor 'Custom'"),
               (dict(spec(), truncation={"max_length": 8}), "truncation"),
               (spec(dropout=0.1), "dropout"),
               (spec(pre={"type": "Split", "pattern": {"Regex": r"\w+"}, "behavior": "Isolated", "invert": False}),
                r"\\w")]
        for sp, what in bad:
            with self.subTest(what=what):
                with self.assertRaisesRegex(ValueError, what):
                    Tokenizer(sp)


class TestUnicodeTables(unittest.TestCase):

    def test_tables(self):
        self.assertEqual(H._category_ranges(("Lu",))[0], (0x41, 0x5A))
        self.assertTrue(any(a <= 0x10D4A <= b for a, b in H._category_ranges(("Lo",))))     # Unicode 16.0
        self.assertTrue(any(a <= 0x11DE0 <= b for a, b in H._numeric_ranges()))       # Unicode 17.0 digits
        self.assertFalse(any(a <= 0x11DE0 <= b for a, b in H._category_ranges(H._CATEGORIES["N"])))

    def test_agrees_with_unicodedata(self):
        """Every code point this Python knows has the table's category, but
        for the two Unicode 16 changed (U+1734, U+1171E: Mn -> Mc)."""
        H._category_ranges(("Lu",))
        runs, bad = H._gc_runs, []
        for k, (a, c) in enumerate(runs):
            b = runs[k + 1][0] - 1 if k + 1 < len(runs) else H.MAX_CP
            for cp in range(a, b + 1):
                u = unicodedata.category(chr(cp))
                if u != "Cn" and u != c and cp not in (0x1734, 0x1171E):
                    bad.append(hex(cp))
        self.assertEqual(bad, [])


class TestShims(unittest.TestCase):

    def test_bos_eos_fallbacks(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "tokenizer.json")
            with open(p, "w", encoding="utf-8") as f:
                json.dump(spec(added=[at("<s>", 1), at("</s>", 2), at("<|im_start|>", 600), at("<|im_end|>", 601)]), f)
            s, ll = smollm2_tokenizer.Tokenizer(p), llama_tokenizer.Tokenizer(p)
            n = len(vocab()[0])
            self.assertEqual((s.bos_id, s.eos_id, ll.bos_id, ll.eos_id), (n, n + 1, 1, 2))
            self.assertIsInstance(s, Tokenizer)
            self.assertIs(smollm2_tokenizer.IncrementalDecoder, H.IncrementalDecoder)


def _sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


class TestPinnedTokenizers(unittest.TestCase):
    """The tokenizers validate_text.py --all checks, against the reference ids
    it wrote (skipped per tokenizer when its file is absent or differs)."""

    @classmethod
    def setUpClass(cls):
        with open(FIXTURE, encoding="utf-8") as f:
            cls.fx = json.load(f)

    def test_cases(self):
        seen = 0
        for name, entry in self.fx["tokenizers"].items():
            path = os.path.join(CHAT, entry["path"])
            if not os.path.exists(path) or _sha256(path) != entry["sha256"]:
                continue
            seen += 1
            tok = Tokenizer(path)
            for c in entry["cases"]:
                with self.subTest(tokenizer=name, text=c["text"][:40]):
                    self.assertEqual(tok.encode(c["text"]), c["ids"])
                    self.assertEqual(tok.encode(c["text"], add_special_tokens=False), c["plain"])
                    self.assertEqual(tok.decode(c["ids"]), c["decode"])
                    self.assertEqual(tok.decode(c["ids"], skip_special_tokens=False), c["decode_all"])
                    d = tok.incremental_decoder()
                    self.assertEqual(d.add_many(c["ids"]) + d.flush(), c["decode"])
        if not seen:
            self.skipTest("no pinned tokenizer present (validate_text.py --fetch)")


if __name__ == "__main__":
    unittest.main()
