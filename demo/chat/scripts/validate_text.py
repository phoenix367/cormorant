#!/usr/bin/env python3
# Copyright 2026 Ivan Gubochkin
# SPDX-License-Identifier: Apache-2.0

"""Validate the pure-Python tokenizer (hf_tokenizer.py, the tokenizer.json
pipeline every chat backend uses) and SmolLM2's chat template (chat_template.py)
against transformers / `tokenizers` (CHAT_PLAN phase 4, gate 1).

  * default — SmolLM2 against transformers: >= 2 000 deterministic, diverse
    strings — held-out / calibration text lines, many scripts, emoji and ZWJ
    sequences, combining marks, number forms, code, JSON / HTML, whitespace
    runs (tabs, CR LF, NBSP, ideographic space, control characters),
    contractions, special tokens inside text, random code points (all planes,
    unassigned, private use) — token ids, decode(encode(s)) and HF decode
    identical; plus random id sequences (broken UTF-8 included) decoded
    identically, with and without special tokens, and the incremental decoder
    equal to decode() under every split; and >= 30 conversations (system / no
    system, multi-turn, empty content, special roles) — rendered text and ids
    identical to apply_chat_template(add_generation_prompt=True / False).

    The reference is transformers' TokenizersBackend.from_pretrained (the
    tokenizer.json pipeline as the `tokenizers` library runs it — what the
    model was trained with, and what transformers 4.x AutoTokenizer returns).
    transformers 5.x AutoTokenizer builds a GPT2Tokenizer class that drops
    tokenizer.json's Digits pre-tokenizer; its ids are compared too, and every
    difference must be that omission (our tokenizer with the Digits step
    removed reproduces them) — only strings with a numeral after two or more
    whitespace characters, or with non-ASCII numerals (Arabic-Indic,
    Devanagari, Ethiopic, ...), are affected.  --write-fixture writes
    demo/chat/tests/data/smollm2_text_cases.json (a subset with the reference
    ids) for the stdlib unit tests.

  * --assets DIR — any tokenizer.json (DIR/tokenizer.json) against the
    `tokenizers` library: the strings above + a Unicode corpus (combining
    sequences, normalization- and case-sensitive characters, every script's
    digits, Unicode 16 / 17 letters, whitespace and invisible characters,
    emoji, CJK, RTL and bidi controls) + the tokenizer's own added tokens in
    context: ids with and without special tokens added and with special-token
    strings as text (encode_special_tokens), decode with and without special
    tokens, the incremental decoder token by token and split in two, decoding
    as a continuation (strip_first=False) against the reference decoding a
    word first; random id sequences (byte tokens) decoded identically.  One
    deliberate difference is counted, not failed: `tokenizers` keeps a
    normalized special token (Llama-2's "▁<s>") under skip_special_tokens.

  * --model NAME — the checkpoint in assets/NAME (stories15m) the same way,
    plus its TinyStories texts and the Llama special-token cases;
    --write-fixture then writes tests/data/NAME_text_cases.json.

  * --all — SmolLM2-135M, SmolVLM-256M, stories15m, Moonshine base / tiny
    (demo/asr/assets) and every tokenizer pinned in tokenizer_assets.json
    (Qwen2.5, TinyLlama, Phi-3, Llama-3.2, Gemma-3, Mistral; --fetch first),
    then --regex and --synthetic.  A missing one is reported as skipped.
    --write-fixture then writes tests/data/hf_tokenizer_cases.json (~40
    strings per pinned tokenizer, SmolVLM's and Moonshine's, with the
    reference ids and decodings) for the stdlib unit tests.

  * --regex — the Oniguruma -> Python re translation (onig_to_re) of the
    GPT-2, Qwen2, Llama-3 and GPT-4o split patterns and of patterns using the
    other classes: pieces identical to tokenizers' Split on the corpus and on
    one string of every code point.

  * --synthetic — small tokenizer.json pipelines (every normalizer,
    pre-tokenizer, Split behaviour, decoder and added-token flag hf_tokenizer
    supports) against `tokenizers` on a short corpus.

  * --fetch — download the files pinned in tokenizer_assets.json (revision +
    SHA-256; HF_ENDPOINT, HF_TOKEN) into demo/chat/assets/tokenizers/<name>/.

Usage (.venv-export: transformers, tokenizers):
  /home/ivan/projects/axi_demo/.venv-export/bin/python demo/chat/scripts/validate_text.py [--write-fixture]
  /home/ivan/projects/axi_demo/.venv-export/bin/python demo/chat/scripts/validate_text.py --model stories15m
  /home/ivan/projects/axi_demo/.venv-export/bin/python demo/chat/scripts/validate_text.py --fetch
  /home/ivan/projects/axi_demo/.venv-export/bin/python demo/chat/scripts/validate_text.py --all
"""
import argparse
import json
import unicodedata
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CHAT = os.path.dirname(HERE)
sys.path.insert(0, CHAT)

import chat_template                                        # noqa: E402
import hf_tokenizer                                         # noqa: E402
from hf_tokenizer import Tokenizer                          # noqa: E402

ASSETS = os.environ.get("SMOLLM_ASSETS", os.path.join(CHAT, "assets", "smollm2-135m-instruct"))
FIXTURE = os.path.join(CHAT, "tests", "data", "smollm2_text_cases.json")
PINS = os.path.join(HERE, "tokenizer_assets.json")
PINNED_DIR = os.path.join(CHAT, "assets", "tokenizers")
ASR_ASSETS = os.path.join(os.path.dirname(CHAT), "asr", "assets")

SCRIPTS = {
    "latin": "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs.",
    "latin-ext": "Zażółć gęślą jaźń. Příliš žluťoučký kůň úpěl ďábelské ódy. Árvíztűrő tükörfúrógép. Æøå ß ŉ ǅ",
    "french": "L'élève a dit qu'il n'aime pas les crêpes à l'huile d'olive, n'est-ce pas ?",
    "german": "Die Straße ist naß; Größe, Übermut und Ärger — „Anführungszeichen“ »so«.",
    "greek": "Ξεσκεπάζω την ψυχοφθόρα βδελυγμία. Τάχιστη αλώπηξ βαφής ψημένη γη.",
    "cyrillic": "Съешь же ещё этих мягких французских булок, да выпей чаю. Ґанок їжака є.",
    "hebrew": "דג סקרן שט בים מאוכזב ולפתע מצא חברה. שָׁלוֹם עֲלֵיכֶם",
    "arabic": "نص حكيم له سر قاطع وذو شأن عظيم مكتوب على ثوب أخضر ومغلف بجلد أزرق ١٢٣ ٤٥٦",
    "persian": "من می‌خواهم فارسی صحبت کنم ۱۴۰۲ ۷۸۹",
    "devanagari": "ऋषियों को सताने वाले दुष्ट राक्षसों के राजा रावण का सर्वनाश करने वाले विष्णुवतार भगवान श्रीराम। १२३",
    "bengali": "আমি বাংলায় গান গাই। ০১২৩",
    "tamil": "யாமறிந்த மொழிகளிலே தமிழ்மொழி போல் இனிதாவது எங்கும் காணோம்",
    "thai": "เป็นมนุษย์สุดประเสริฐเลิศคุณค่า กว่าบรรดาฝูงสัตว์เดรัจฉาน ๑๒๓",
    "chinese": "我能吞下玻璃而不伤身体。天地玄黄，宇宙洪荒。一二三四五六七八九十百千万亿",
    "japanese": "いろはにほへと ちりぬるを わかよたれそ つねならむ カタカナ ｶﾀｶﾅ 漢字。",
    "korean": "다람쥐 헌 쳇바퀴에 타고파. 키스의 고유조건은 입술끼리 만나야 하고",
    "georgian": "გთხოვთ ახლავე გაიაროთ რეგისტრაცია",
    "armenian": "Բարեւ աշխարհ, ինչպե՞ս ես։",
    "ethiopic": "ሰማይ አይታረስ ንጉሥ አይከሰስ። ፩፪፫",
    "emoji": "😀 😃 👍🏽 👨‍👩‍👧‍👦 🏳️‍🌈 🇫🇷🇩🇪 ❤️ ✨🎉 🤖💬 🧑🏿‍💻 ⌚ ☕ 1️⃣ #️⃣",
    "math": "∀x∈ℝ: x² ≥ 0; ∑ᵢ aᵢ = ½ · π ≈ 3.14159; √2 ≠ 1.41; ∞ ⊂ ℵ₀ ⅔ ⅞ Ⅻ ⅻ ①②③ ⁴⁵ ₆₇",
    "fullwidth": "ＡＢＣ　ａｂｃ　０１２３４５６７８９　！？（）［］",
    "combining": "é à ñ ö Z͓͑͒ क्ष 각",
    "symbols": "© ® ™ § ¶ † ‡ • … ‰ ′ ″ ‹ › « » € £ ¥ ₹ ₽ ¢ ° ± × ÷ ← ↑ → ↓ ↔ ⇒ ⇔ ♠ ♣ ♥ ♦ ♪ ☺ ☻",
    "box": "┌──┬──┐\n│ab│cd│\n├──┼──┤\n└──┴──┘ ░▒▓█ ▲▼◆◇○●",
}

CODE = [
    "def fib(n):\n    if n < 2:\n        return n\n    return fib(n - 1) + fib(n - 2)\n",
    "for (int i = 0; i < n; ++i) {\n\tsum += a[i] * b[i];\n}\n",
    "#include <stdio.h>\nint main(void) { printf(\"%d\\n\", 42); return 0; }",
    "SELECT name, COUNT(*) AS n FROM users WHERE age >= 18 GROUP BY name ORDER BY n DESC;",
    "const f = async (x) => { await fetch(`/api/${x}?q=1&r=2`); };",
    "{\"key\": [1, 2.5, -3e10, true, null], \"nested\": {\"a\": \"b\\u00e9\"}}",
    "<div class=\"x\"><a href='https://example.com/path?a=1#frag'>link</a></div>",
    "x = [i**2 for i in range(10) if i % 2 == 0]  # squares\nprint(x)",
    "fn main() { let v: Vec<u32> = (0..10).map(|x| x * 2).collect(); }",
    "    \n\t\t  if (a && b || !c) { return a->b.c[0]; }  \r\n",
    "$ ls -la /usr/bin | grep -E '^-rwx' | wc -l\n",
    "diff --git a/f.py b/f.py\n@@ -1,3 +1,4 @@\n-old line\n+new line\n",
    "matrix = [[1,2,3],[4,5,6]]\n\n\n\n# trailing newlines\n\n",
    "'''docstring''' \"\"\"triple\"\"\" 'single' \"double\" `backtick`",
    "0x7fffffff 0b1011 1_000_000 3.14e-10 -0.0 +1 1e+308 NaN inf",
]

WS = [" ", "  ", "   ", "\t", "\n", "\n\n", "\r\n", " ", "　", " ", "​", " ",
      "\x0b", "\x0c", "\x1c", "\x1f", "\x85", " \t ", "\t\n ", "      ", "\n \n"]

WORDS = ["hello", "world", "It", "'s", "'S", "don't", "I'm", "we'll", "they've", "you're", "he'd",
         "can't", "O'Neil", "rock'n'roll", "’s", "“quoted”", "(paren)", "[bracket]", "{brace}",
         "e-mail", "U.S.A.", "a.m.", "Dr.", "#hashtag", "@user", "path/to/file.txt", "C++", "C#",
         "naïve", "café", "Zürich", "東京", "Москва", "3.5", "1,000,000", "2024-09-26", "12:30pm",
         "+1-555-0100", "$19.99", "50%", "x86_64", "GPT-2", "COVID-19", "__init__", "snake_case",
         "CamelCase", "ALLCAPS", "mIxEd", "a", "I", "the", "The", "THE", "<b>", "</b>", "&amp;",
         "!!!", "???", "...", "--", "—", "–", "…", "<|im_start|>", "<|im_end|>", "<|endoftext|>",
         "<repo_name>", "<reponame>", "<|im_start", "im_end|>", "<|", "|>", "<file_sep>",
         "<jupyter_code>", "<empty_output>", "\\n", "\\t", "😀", "👍🏽", "中文", "٣", "²", "½", "Ⅻ"]


def random_codepoint(rng):
    r = rng.random()
    if r < 0.35:
        return chr(rng.randint(0x20, 0x7e))
    if r < 0.45:
        return chr(rng.randint(0x00, 0x1f))
    if r < 0.60:
        return chr(rng.randint(0xa0, 0x2fff))
    if r < 0.70:
        return chr(rng.randint(0x3000, 0x9fff))
    if r < 0.78:
        return chr(rng.randint(0xa000, 0xd7ff))
    if r < 0.84:
        return chr(rng.randint(0xe000, 0xffff))
    if r < 0.93:
        return chr(rng.randint(0x10000, 0x1ffff))
    if r < 0.97:
        return chr(rng.randint(0x20000, 0x3ffff))
    return chr(rng.randint(0x40000, 0x10ffff))


def make_strings(n_min=2000, seed=1234):
    rng = random.Random(seed)
    out = []
    for name in ("heldout_wikitext2_test.txt", "calib_wikitext2_valid.txt"):
        p = os.path.join(ASSETS, name)
        if os.path.exists(p):
            lines = [ln for ln in open(p, encoding="utf-8").read().split("\n") if ln.strip()]
            out += lines[:300]
    for name, text in SCRIPTS.items():
        out.append(text)
        words = text.split(" ")
        for _ in range(12):
            k = rng.randint(1, len(words))
            i = rng.randint(0, len(words) - k)
            out.append(rng.choice(["", " ", "  ", "\n"]) + " ".join(words[i:i + k]) + rng.choice(["", " ", "\n", "  "]))
    for c in CODE:
        out.append(c)
        for _ in range(8):
            a, b = sorted(rng.sample(range(len(c) + 1), 2))
            out.append(c[a:b])
    for w in WS:                                            # whitespace runs in every position
        for x in ("a", "1", "!", "é", "中", "😀", "'s", ""):
            out += [w + x, x + w, x + w + x, w * 3 + x + w * 2, x + w + w + x]
    for _ in range(700):                                    # word salad
        k = rng.randint(1, 14)
        parts = []
        for _ in range(k):
            parts.append(rng.choice(WORDS))
            parts.append(rng.choice(WS[:8] + [" "] * 12 + [""] * 4))
        out.append("".join(parts))
    for _ in range(300):                                    # numbers
        f = rng.choice(["{}", " {}", "{} ", "x{}", "{}x", "  {}", "{}.{}", "{},{}", "-{}", "{}e{}"])
        out.append(f.format(*[rng.randint(0, 10 ** rng.randint(1, 12)) for _ in range(f.count("{}"))]))
    for _ in range(400):                                    # random code points
        out.append("".join(random_codepoint(rng) for _ in range(rng.randint(1, 40))))
    for _ in range(60):                                     # long runs (BPE on long words)
        ch = rng.choice(["a", "=", "-", " ", "\n", "ab", "😀", "中", "0", "!?", "é"])
        out.append(ch * rng.randint(20, 300))
    out += ["", " ", "\n", "a", "  leading", "trailing  ", "<|im_start|>", "<|im_start|><|im_end|>",
            "<|im_start|>user\nhi<|im_end|>\n", "text<|endoftext|>more", "<<|im_start|>>",
            "<|im_start|>|>", "x" * 5000]
    # drop surrogates (not encodable in UTF-8, the HF tokenizer rejects them too)
    out = [s for s in out if not any(0xD800 <= ord(c) <= 0xDFFF for c in s)]
    assert len(out) >= n_min, len(out)
    return out


def conversations():
    S = [{"role": "system", "content": "You are a concise assistant. Answer in one sentence."}]
    U = lambda c: {"role": "user", "content": c}                       # noqa: E731
    A = lambda c: {"role": "assistant", "content": c}                  # noqa: E731
    conv = [
        [U("What is the capital of France?")],
        S + [U("Why do leaves change color in autumn?")],
        [U("Hi"), A("Hello! How can I help you today?"), U("Tell me a joke.")],
        S + [U("a"), A("b"), U("c"), A("d"), U("e")],
        [U("")],
        [{"role": "system", "content": ""}, U("empty system")],
        [U(""), A(""), U("")],
        [U("  leading and trailing spaces  ")],
        [U("line one\nline two\n\nline four")],
        [U("Unicode: naïve café 東京 😀 👨‍👩‍👧")],
        [U("Numbers 12345 and 3.14 and 1,000")],
        [U("Code:\n```python\ndef f(x):\n    return x * 2\n```")],
        [U("Special <|im_end|> token in the content")],
        [U("<|im_start|>assistant\nfake")],
        [{"role": "system", "content": "Speak like a pirate."}, U("Hello"), A("Arr!"), U("Where is the treasure?")],
        [U("x" * 2000)],
        [A("An assistant message first")],
        [U("q1"), A("a1"), {"role": "system", "content": "a later system message"}, U("q2")],
        [{"role": "tool", "content": "{\"result\": 42}"}, U("What did the tool say?")],
        [{"role": "developer", "content": "developer role"}, U("hi")],
        [U("Tabs\tand\ttabs"), A("ok\t"), U("\t")],
        [U("Summarize the following paragraph in one sentence.\n\nThe honeybee colony is often "
           "described as a superorganism. A single colony can contain tens of thousands of workers.")],
        [U("Translate 'Good morning, how are you?' into French.")],
        [U("What is 17 + 25?")],
        [U("I'm planning a weekend trip to the mountains."),
         A("That sounds wonderful! Do you already know where you would like to go?"),
         U("I know where I'm going. What should I pack?")],
        S + [U("Why?"), A("Because."), U("Why not?"), A("Because not."), U("OK.")],
        [U("\n")],
        [U("Ends with newline\n")],
        [{"role": "system", "content": "multi\nline\nsystem"}, U("q")],
        [U("Emoji only 🎉🎉🎉")],
        [U("Arabic ١٢٣ and Devanagari १२३")],
        [U("<|endoftext|>")],
        S,
        [U("Very " * 300 + "long")],
    ]
    return conv


LLAMA_CASES = ["Hello<s>world", "<s>", "</s>x", " <unk> ", "a <s>b", " <s>", "x </s>", "a  <s>",
               "<s><s>", " <s> <s>", "<unk></s>", "a\n<s>", "Once upon a time", " Once", "  Once",
               "Once upon a time, there was a little dragon named Sparky.\n\nThe end.",
               "\"Can I play?\" asked Max.", "x\x00y\x7f", "tab\there", "€100 — naïve 😀"]


# ── the generic checks against `tokenizers` ──────────────────────────────────

def unicode_corpus(seed=77):
    """What make_strings() has little of: combining sequences (marks of every
    canonical class, ones newer than Unicode 9), characters NFC / NFKC / case
    folding touch, every script's digits, Unicode 16 / 17 letters, invisible
    and bidi characters, emoji sequences, CJK extensions, contractions in
    every case, SentencePiece spaces, newline / tab runs — alone and mixed."""
    rng = random.Random(seed)
    marks = [chr(c) for c in (0x300, 0x301, 0x302, 0x303, 0x308, 0x30A, 0x316, 0x323, 0x327, 0x328, 0x331,
                              0x345, 0x35C, 0x5B0, 0x5B4, 0x5BC, 0x5C1, 0x64B, 0x650, 0x651, 0x670, 0x7FD,
                              0x8D3, 0x93C, 0x94D, 0x9BC, 0x9FE, 0xD3B, 0x1AB0, 0x1ABF, 0x1AC0, 0x1ACE,
                              0x1DF6, 0x1DFA, 0x20D0, 0x20E3, 0x302A, 0x3099, 0x309A, 0xFE0F, 0xFE20,
                              0x16FF0, 0x1E944, 0x1133B, 0x11930, 0x1E2EC, 0x1E4EC)]
    bases = list("aeiouyAEIOUcnszCNSZgkhwHWİıſKßΣσςΑΩωαικ가각ㄱ") + [
        "ᄀ", "ᅡ", "ᆨ", "ب", "א", "क", "ক", "ക", "\U00011935"]
    compat = list("ﬁﬂﬀﬃﬅﬆ㎏㍿①⑳Ⅻⅻ½¼²³⁴₅ｶﾞﾊﾟＡａ０。℃℉™№…‥⁇﹖︵ｰ〜Ω") + [
        "㋿", "\U0001f16c", "\U0001fbf0", "Å", "̀", "̈́", "ཱི", "क़", "יִ",
        "⫝̸", "\U0001d400", "\U0001d7ce", "ẛ̣", "Ά", "ά"]
    digits = [chr(a + rng.randint(0, min(9, b - a))) for a, b in hf_tokenizer._numeric_ranges()]
    new_letters = ["\U00010D4A", "\U00010D50", "\U00010D70", "Ᲊ", "ᲊ", "Ɤ", "Ꟍ", "Ƛ",
                   "\U000105C0", "\U00016D40", "\U00011BC0", "\U0001E5D0", "\U000116D0", "\U00016EA0",
                   "\U00016EBB", "\U00011DE0", "\U00016FF4", "\U00010570", "\U00010597", "Ⱟ", "Ꟁ"]
    ws = list(hf_tokenizer.WHITE_SPACE) + list("\x1c\x1d\x1e\x1f᠎​‌‍⁠﻿­")
    emoji = ["👨‍👩‍👧‍👦", "🏳️‍🌈", "🇫🇷🇩🇪🇯🇵", "1️⃣#️⃣*️⃣", "👍🏽👍🏿", "🏴󠁧󠁢󠁥󠁮󠁧󠁿", "🧑🏿‍💻", "❤️‍🔥", "🫠🪿🩷",
             "☕️", "🤌🏼"]
    rtl = ["مَرْحَبًا بِالْعَالَمِ", "שָׁלוֹם עוֹלָם", "‏مرحبا‎ abc", "‫نص‬", "⁧x⁩",
           "سلام ۱۲۳ و ١٢٣"]
    cjk = ["漢字かなカナ", "\U00020000\U0002a6d6\U00031350\U00032000", "한국어 텍스트", "각",
           "ｶﾀｶﾅ ｸﾞ"]
    words = ["it's IT'S It'S i'Ll WE'RE you've YOU'VE I'M I'm he'd HE'D 'ſ 'K 'k 'İ 'ı can'T", "'s's''s 'll'd",
             "O'Neil's", "'t'T", "'ſt 'Ks", "ΣΑΣ σας ΌΣΟΣ", "İstanbul ıi", "Straße STRASSE", "ǅemal ǈ ǋ"]
    runs = ["\r\n\r\n", "\n\n\n\n\n", "  \n  \n", "\t\t\n\t", "a\r\nb\rc\n", "x  \n\n  y", "def f():\n\treturn 1\n\n\n",
            "1234567890123", "3.14159265358979", "١٢٣٤٥٦٧٨٩٠", "१२३४५", "𝟏𝟐𝟑𝟒", "Ⅻ½²", "▁▁▁", "▁a▁", " ▁ ",
            "▁\n▁", "<0x41>", "Ġhello", "##ing", "</w>", "  ", "\t", "\n"]
    out = marks + compat + digits + new_letters + emoji + rtl + cjk + words + runs
    out.append("".join(digits))
    out.append(" ".join(new_letters))
    out.append("".join(ws))
    out += [w + "x" + w for w in ws]
    for _ in range(150):
        out.append(rng.choice(bases) + "".join(rng.sample(marks, rng.randint(1, 4)))
                   + rng.choice(["", " ", "a", rng.choice(bases)]))
    pools = [marks, bases, compat, digits, new_letters, ws, emoji, rtl, cjk, words, runs,
             ["hello", "World", "x", "the", "THE", "12", "!", "?!", "...", "--", "a.b", "<b>", "&amp;"]]
    for _ in range(300):
        parts = []
        for _ in range(rng.randint(2, 10)):
            parts.append(rng.choice(rng.choice(pools)))
            parts.append(rng.choice(["", "", " ", "  ", "\n", "\t", " \n"]))
        out.append("".join(parts))
    return out


def special_cases(tok, seed=3):
    """The tokenizer's added tokens in context (spaces, newlines, words around
    them, doubled, cut in half): lstrip / rstrip / single_word and the
    matching order."""
    rng = random.Random(seed)
    toks = sorted(tok.added_tokens.values(), key=lambda a: a.id)
    special = [a for a in toks if a.special]
    other = [a for a in toks if not a.special]
    flagged = [a for a in toks if a.lstrip or a.rstrip or a.single_word]
    pick = (special if len(special) <= 30 else special[:10] + rng.sample(special[10:], 20)) + \
        (other if len(other) <= 15 else other[:5] + rng.sample(other[5:], 10)) + flagged[:10]
    out = []
    for a in pick:
        c = a.content
        h = len(c) // 2
        out += [c, f" {c} ", f"x{c}y", f"\n{c}\n", f"  {c}\t\n", c + c, f"{c}hello {c} world", f"a {c}b",
                c[:-1], c[1:], f"{c[:h]} {c[h:]}"]
    for _ in range(40):
        out.append("".join(rng.choice([rng.choice(pick).content if pick else "", "Hi", " ", "\n", "x", "  "])
                           for _ in range(rng.randint(2, 8))))
    return out


def byte_pool(tok):
    """Ids of single-byte tokens (byte fallback's <0xXX>, or the byte-level
    alphabet's characters): random sequences of them make broken UTF-8."""
    ids = [tok.vocab[f"<0x{b:02X}>"] for b in range(256) if f"<0x{b:02X}>" in tok.vocab]
    if not ids:
        ids = [tok.vocab[c] for c in hf_tokenizer.bytes_to_unicode().values() if c in tok.vocab]
    return ids


def check_tokenizer(path, strings, label):
    """hf_tokenizer against `tokenizers` on one tokenizer.json (see the module
    docstring) -> (all identical, [(string, reference ids)])."""
    from tokenizers import Tokenizer as Ref
    ref = Ref.from_file(path)
    ref_text = Ref.from_file(path)                    # special-token strings as text
    ref_text.encode_special_tokens = True
    t0 = time.perf_counter()
    tok = Tokenizer(path)
    load_s = time.perf_counter() - t0
    strings = list(strings) + special_cases(tok)
    special = tok.special_ids
    kept = {i for i, a in tok.added_tokens.items() if a.special and tok.id_to_token[i] != a.content}
    word = tok.encode("Hello", add_special_tokens=False)
    word_text = ref.decode(word)
    rng = random.Random(11)
    bad = {"ids": 0, "decode": 0, "incremental": 0, "continuation": 0, "random ids": 0}
    n_kept = n_tok = n_cont = 0
    t_ours = 0.0
    cases = []

    def fail(kind, msg):
        bad[kind] += 1
        if bad[kind] <= 3:
            print(f"  MISMATCH ({kind}) {msg}")

    for s in strings:
        want = ref.encode(s).ids
        t = time.perf_counter()
        ours = tok.encode(s)
        t_ours += time.perf_counter() - t
        n_tok += len(want)
        plain = tok.encode(s, add_special_tokens=False)
        if (ours != want or plain != ref.encode(s, add_special_tokens=False).ids
                or tok.encode(s, special=False) != ref_text.encode(s).ids):
            fail("ids", f"{s[:60]!r}\n    ours {ours[:24]}\n    ref  {want[:24]}")
        cases.append((s, want))
        ids = want
        dec = tok.decode(ids)
        if dec != ref.decode(ids):
            if kept & set(ids) and dec == ref.decode([i for i in ids if i not in special],
                                                     skip_special_tokens=False):
                n_kept += 1
            else:
                fail("decode", f"{s[:60]!r}: {dec[:60]!r} != {ref.decode(ids)[:60]!r}")
        if tok.decode(ids, False) != ref.decode(ids, skip_special_tokens=False):
            fail("decode", f"{s[:60]!r} (special tokens kept)")
        for skip in (True, False):
            for strip in (True, False):
                full = tok.decode(ids, skip, strip)
                inc = tok.incremental_decoder(skip, strip)
                one = "".join(inc.add(i) for i in ids) + inc.flush()
                cut = rng.randint(0, len(ids))
                inc = tok.incremental_decoder(skip, strip)
                two = inc.add_many(ids[:cut]) + inc.add_many(ids[cut:]) + inc.flush()
                if not one == two == full:
                    fail("incremental", f"{s[:60]!r} skip={skip} strip_first={strip}")
        body = [i for i in plain if i not in special]
        both = ref.decode(word + body)
        if both.startswith(word_text):
            n_cont += 1
            if tok.decode(body, strip_first=False) != both[len(word_text):]:
                fail("continuation", f"{s[:60]!r}")
    pool = byte_pool(tok)
    specials = sorted(special) or [0]
    for k in range(2000):
        n = rng.randint(1, 30)
        ids = ([rng.randrange(tok.vocab_size) for _ in range(n)] if k % 3 == 0 else
               [rng.choice(pool) if rng.random() < 0.8 else rng.choice(specials) for _ in range(n)] if pool else
               [rng.randrange(tok.vocab_size) for _ in range(n)])
        ok = tok.decode(ids, False) == ref.decode(ids, skip_special_tokens=False)
        dec = tok.decode(ids)
        ok &= dec == ref.decode(ids) or dec == ref.decode([i for i in ids if i not in special],
                                                           skip_special_tokens=False)
        inc = tok.incremental_decoder()
        ok &= "".join(inc.add(i) for i in ids) + inc.flush() == dec
        if not ok:
            fail("random ids", f"{ids}")
    good = not any(bad.values())
    print(f"{label}: {'OK' if good else 'FAILED'} — {len(strings)} strings, {n_tok} tokens: "
          + ", ".join(f"{k} {len(strings) - v if k != 'random ids' else 2000 - v}/"
                      f"{len(strings) if k != 'random ids' else 2000}" for k, v in bad.items())
          + f" identical (continuation checked on {n_cont}); "
          + (f"{n_kept} decodes keep a normalized special token in `tokenizers` (dropped here); " if n_kept
             else "")
          + f"load {load_s:.2f} s, encode {n_tok / t_ours / 1e3:.0f} k tok/s")
    return good, cases


def main_llama(model: str, write_fixture: bool) -> int:
    """The tokenizer of a text-profile checkpoint (stories15m) against `tokenizers`."""
    assets = os.path.join(CHAT, "assets", model)
    path = os.path.join(assets, "tokenizer.json")
    strings = make_strings() + LLAMA_CASES
    for name in ("heldout_tinystories_valid.txt", "calib_tinystories_valid.txt"):
        p = os.path.join(assets, name)
        if os.path.exists(p):
            strings += open(p, encoding="utf-8").read().split("\n<|endoftext|>\n")
    good, cases = check_tokenizer(path, strings + unicode_corpus(), model)
    cases = cases[:len(strings)]
    if write_fixture:
        fixture = os.path.join(CHAT, "tests", "data", f"{model}_text_cases.json")
        rng = random.Random(5)
        sub = [c for c in cases if len(c[0]) <= 400]
        sub = sub[:120] + rng.sample(sub[120:], 380) + [c for c in cases if c[0] in LLAMA_CASES]
        with open(fixture, "w", encoding="utf-8") as f:
            json.dump({"source": f"tokenizers {model}/tokenizer.json via "
                                 f"demo/chat/scripts/validate_text.py --model {model} --write-fixture",
                       "strings": [{"text": s, "ids": ids} for s, ids in sub]},
                      f, ensure_ascii=True, indent=None, separators=(",", ":"))
        print(f"wrote {fixture} ({os.path.getsize(fixture) // 1024} KiB)")
    return 0 if good else 1


# ── the regex translation ────────────────────────────────────────────────────

CONTRACTIONS = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
PATTERNS = {
    "GPT-2 (ByteLevel)": hf_tokenizer.GPT2_PATTERN,
    "Qwen2": CONTRACTIONS + r"|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
    "Llama-3": CONTRACTIONS + r"|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)"
                              r"|\s+",
    "GPT-4o (o200k)": r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+"
                      + CONTRACTIONS + r"?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+"
                      r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]*" + CONTRACTIONS + r"?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*"
                      r"|\s*[\r\n]+|\s+(?!\S)|\s+",
    "categories": r"\p{Lu}+|\p{Ll}+|\p{Lt}|\p{Lm}+|\p{Lo}+|\p{Mn}+|\p{Mc}|\p{Me}|\p{Nd}+|\p{Nl}|\p{No}|\p{Pc}"
                  r"|\p{Pd}|\p{Ps}|\p{Pe}|\p{Pi}|\p{Pf}|\p{Po}+|\p{Sm}|\p{Sc}|\p{Sk}|\p{So}+|\p{Zs}+|\p{Zl}"
                  r"|\p{Zp}|\p{Cc}+|\p{Cf}+|\p{Co}+|\p{Cn}+",
    "negations": r"\P{L}+|\p{^N}\p{N}|[^\p{P}\p{S}\s]{2}|\S\s|\d+|\D\d|\h+",
    "case and anchors": r"(?i:[a-z]+'?s)|(?:(?i)st|\s+$|x{2,}|[\x41-\x5a]{,2})|^\s+|\x{e9}|\u00e8|.",
}


def check_regex() -> bool:
    """onig_to_re against tokenizers' Split (isolated) on the corpus and one
    string of every code point."""
    from tokenizers import Regex, pre_tokenizers
    every = "".join(chr(c) for c in range(0x110000) if not 0xD800 <= c <= 0xDFFF)
    corpus = make_strings() + unicode_corpus() + [every, every[::-1]]
    good = True
    for name, pat in PATTERNS.items():
        ref = pre_tokenizers.Split(Regex(pat), behavior="isolated")
        rx = hf_tokenizer._onig_regex(pat, name)       # tokenizers' match iteration (+ the ASCII cut)
        bad = 0
        for s in corpus:
            if [p for p, _ in ref.pre_tokenize_str(s)] != [p for p in rx.parts(s) if p]:
                bad += 1
                if bad <= 2:
                    print(f"  MISMATCH {name}: {s[:60]!r}")
        good &= bad == 0
        print(f"regex {name}: {len(corpus) - bad}/{len(corpus)} strings split identically "
              f"(incl. all {len(every)} code points in one string)")
    return good


# ── synthetic pipelines ──────────────────────────────────────────────────────

def synthetic_vocab():
    """A small BPE vocabulary + merges that cover ASCII, the byte-level
    alphabet, <0xXX> bytes, "▁" / "##" / "</w>" forms and a few words."""
    vocab = {}
    for t in ["<unk>", "<s>", "</s>", "<pad>", "▁", "Ġ"] + [f"<0x{b:02X}>" for b in range(256)] + \
            [chr(c) for c in range(0x20, 0x7F)] + list(hf_tokenizer.bytes_to_unicode().values()) + \
            ["##" + chr(c) for c in range(0x61, 0x7B)] + [chr(c) + "</w>" for c in range(0x61, 0x7B)] + \
            list("éü中😀Σσ"):
        vocab.setdefault(t, len(vocab))
    merges = []
    for w in ["▁the", "▁hello", "▁world", "▁a", "▁is", "ing", "Ġthe", "Ġhello", "Ġworld", "hello", "the",
              "▁▁", "▁▁▁", "ĠĠ", "12", "123", "Ċ", "ĊĊ", "ello", "orld"]:
        cur = w[0]
        for ch in w[1:]:
            if cur + ch not in vocab:
                vocab[cur + ch] = len(vocab)
                merges.append([cur, ch])
            cur += ch
    for t in ("he", "##ll", "hell", "th</w>"):
        vocab.setdefault(t, len(vocab))
    return vocab, merges


def synthetic_pipelines():
    """name -> tokenizer.json dict, one per supported part / option."""
    vocab, merges = synthetic_vocab()

    def at(c, i=None, special=True, normalized=False, lstrip=False, rstrip=False, single_word=False):
        return {"id": vocab.get(c, i), "content": c, "single_word": single_word, "lstrip": lstrip,
                "rstrip": rstrip, "normalized": normalized, "special": special}

    def tj(pre=None, norm=None, dec=None, post=None, added=(), **model):
        m = {"type": "BPE", "dropout": None, "unk_token": "<unk>", "continuing_subword_prefix": None,
             "end_of_word_suffix": None, "fuse_unk": True, "byte_fallback": True, "ignore_merges": False,
             "vocab": vocab, "merges": merges}
        m.update(model)
        return {"version": "1.0", "truncation": None, "padding": None, "added_tokens": list(added),
                "normalizer": norm, "pre_tokenizer": pre, "post_processor": post, "decoder": dec, "model": m}
    sp_norm = {"type": "Sequence", "normalizers": [{"type": "Prepend", "prepend": "▁"},
                                                   {"type": "Replace", "pattern": {"String": " "}, "content": "▁"}]}
    llama_dec = {"type": "Sequence", "decoders": [
        {"type": "Replace", "pattern": {"String": "▁"}, "content": " "}, {"type": "ByteFallback"},
        {"type": "Fuse"}, {"type": "Strip", "content": " ", "start": 1, "stop": 0}]}
    bos = {"type": "TemplateProcessing", "single": [{"SpecialToken": {"id": "<s>", "type_id": 0}},
                                                    {"Sequence": {"id": "A", "type_id": 0}}],
           "pair": [], "special_tokens": {"<s>": {"id": "<s>", "ids": [1], "tokens": ["<s>"]}}}
    bl = {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": True}
    p = {}
    p["llama2 normalized specials"] = tj(norm=sp_norm, dec=llama_dec, post=bos,
                                         added=[at("<unk>", normalized=True), at("<s>", normalized=True),
                                                at("</s>", normalized=True)])
    p["llama2 raw specials + flags"] = tj(norm=sp_norm, dec=llama_dec, post=bos, added=[
        at("<unk>"), at("<s>"), at("</s>", rstrip=True), at("<|user|>", 9000, rstrip=True),
        at("<|end|>", 9001, lstrip=True, rstrip=True), at("<mask>", 9002, special=False, lstrip=True),
        at("hello", None, special=False, single_word=True), at("<tool>", 9003, special=False)])
    for scheme in ("always", "first", "never"):
        for split in (True, False):
            p[f"metaspace {scheme} split={split}"] = tj(
                pre={"type": "Metaspace", "replacement": "▁", "prepend_scheme": scheme, "split": split},
                dec={"type": "Metaspace", "replacement": "▁", "prepend_scheme": scheme, "split": split},
                post=bos, added=[at("<s>"), at("</s>")])
    p["metaspace legacy add_prefix_space"] = tj(
        pre={"type": "Metaspace", "replacement": "▁", "add_prefix_space": True},
        dec={"type": "Metaspace", "replacement": "▁", "add_prefix_space": True})
    for prefix in (False, True):
        for regex in (False, True):
            p[f"byte-level prefix={prefix} regex={regex}"] = tj(
                pre={"type": "ByteLevel", "add_prefix_space": prefix, "trim_offsets": True, "use_regex": regex},
                dec=bl, byte_fallback=False, unk_token=None,
                added=[at("<s>"), at("<|im_end|>", 9000), at("<tool>", 9001, special=False),
                       at("中文", 9002, special=False)])
    for ind in (True, False):
        p[f"digits individual={ind} + byte-level"] = tj(
            pre={"type": "Sequence", "pretokenizers": [{"type": "Digits", "individual_digits": ind}, bl]},
            dec=bl, byte_fallback=False, unk_token=None)
    for beh in ("Isolated", "Removed", "MergedWithPrevious", "MergedWithNext", "Contiguous"):
        for inv in (False, True):
            p[f"split regex {beh} invert={inv}"] = tj(
                pre={"type": "Sequence", "pretokenizers": [
                    {"type": "Split", "pattern": {"Regex": r"\s+|\p{N}"}, "behavior": beh, "invert": inv},
                    dict(bl, use_regex=False)]}, dec=bl, byte_fallback=False, unk_token=None)
        p[f"split empty-matching regex {beh}"] = tj(
            pre={"type": "Split", "pattern": {"Regex": r"[a-z]{,2}|\s"}, "behavior": beh, "invert": False},
            norm={"type": "Replace", "pattern": {"Regex": "x*"}, "content": "-"}, dec=llama_dec)
        p[f"split string {beh}"] = tj(norm={"type": "Replace", "pattern": {"String": " "}, "content": "▁"},
                                      pre={"type": "Split", "pattern": {"String": "▁"}, "behavior": beh,
                                           "invert": False}, dec=llama_dec)
    for n in ("NFC", "NFD", "NFKC", "NFKD", "Lowercase"):
        p[f"normalizer {n}"] = tj(norm={"type": n}, dec=llama_dec,
                                  added=[at("ﬁX", 9000, special=False, normalized=True), at("<s>")])
    for left, right in ((True, True), (True, False), (False, True)):
        p[f"normalizer Strip {left}/{right}"] = tj(norm={"type": "Strip", "strip_left": left, "strip_right": right},
                                                   dec=llama_dec)
    p["normalizer Replace regex + Sequence"] = tj(norm={"type": "Sequence", "normalizers": [
        {"type": "NFKC"}, {"type": "Replace", "pattern": {"Regex": r"\s{2,}"}, "content": " "}, sp_norm]},
        dec=llama_dec)
    p["model continuing_subword_prefix"] = tj(continuing_subword_prefix="##", byte_fallback=False,
                                              merges=[["h", "##e"], ["##l", "##l"], ["he", "##ll"]],
                                              pre={"type": "Metaspace", "replacement": " ", "prepend_scheme": "never"})
    p["model end_of_word_suffix + BPEDecoder"] = tj(end_of_word_suffix="</w>", byte_fallback=False,
                                                    merges=merges + [["t", "h</w>"]],
                                                    dec={"type": "BPEDecoder", "suffix": "</w>"},
                                                    pre={"type": "Split", "pattern": {"String": " "},
                                                         "behavior": "Removed", "invert": False})
    for fuse in (True, False):
        p[f"model unk fuse={fuse}"] = tj(byte_fallback=False, fuse_unk=fuse, dec=None)
    p["model plain (unknown characters dropped)"] = tj(byte_fallback=False, unk_token=None,
                                                       dec={"type": "Sequence", "decoders": []})
    p["model ignore_merges"] = tj(ignore_merges=True, pre=bl, dec=bl, byte_fallback=False,
                                  unk_token=None)
    p["decoder Strip start/stop"] = tj(norm=sp_norm, dec={"type": "Sequence", "decoders": [
        {"type": "Replace", "pattern": {"String": "▁"}, "content": " "}, {"type": "ByteFallback"},
        {"type": "Fuse"}, {"type": "Strip", "content": " ", "start": 2, "stop": 2}]})
    p["decoder Strip per token, Replace regex"] = tj(norm=sp_norm, dec={"type": "Sequence", "decoders": [
        {"type": "Strip", "content": "▁", "start": 1, "stop": 0},
        {"type": "Replace", "pattern": {"Regex": "l+"}, "content": "L"}, {"type": "ByteFallback"}]})
    p["post Sequence + Roberta"] = tj(pre=bl, dec=bl, byte_fallback=False, unk_token=None,
                                      post={"type": "Sequence", "processors": [
                                          dict(bl, type="ByteLevel"),
                                          {"type": "RobertaProcessing", "sep": ["</s>", 2], "cls": ["<s>", 1],
                                           "trim_offsets": True, "add_prefix_space": False}]},
                                      added=[at("<s>"), at("</s>")])
    p["post Bert"] = tj(norm=sp_norm, dec=llama_dec, post={"type": "BertProcessing", "sep": ["</s>", 2],
                                                           "cls": ["<s>", 1]})
    return p


def check_synthetic() -> bool:
    """Every synthetic pipeline: ids (with / without special tokens, special
    strings as text), id_to_token, decode and the incremental decoder against
    `tokenizers`."""
    from tokenizers import Tokenizer as Ref
    rng = random.Random(21)
    corpus = ["", " ", "hello world", " the  hello\tworld\n", "Hello World 123 ٣4", "<s>hello</s>", "a <s> b",
              "<|user|>\n hi<|end|>  \n<|end|>x", "x<mask> y <mask>", "hellohello hello_ hello", "<tool><tool>",
              "ﬁX fix FIX ﬁx", "  lead and trail  ", "▁▁a▁ b", "éü中😀Σ ΣΑΣ", "1234 56", "\n\n\nx\n",
              "it's   IT'S", "l lll", "\x00\x01 control", "中文 中文中文"]
    corpus += rng.sample(unicode_corpus(), 80)
    good = True
    for name, spec in synthetic_pipelines().items():
        ref = Ref.from_str(json.dumps(spec))
        ref_text = Ref.from_str(json.dumps(spec))
        ref_text.encode_special_tokens = True
        tok = Tokenizer(spec)
        bad = 0
        for i in range(tok.vocab_size):
            if tok.id_to_token[i] != ref.id_to_token(i):
                bad += 1
        panics = 0
        for s in corpus:
            ids = ref.encode(s).ids
            ok = (tok.encode(s) == ids and tok.encode(s, add_special_tokens=False) == ref.encode(
                s, add_special_tokens=False).ids and tok.encode(s, special=False) == ref_text.encode(s).ids)
            kept = [i for i in ids if i in tok.special_ids and tok.id_to_token[i] != tok.added_tokens[i].content]
            try:
                want = ref.decode(ids) if not kept else ref.decode([i for i in ids if i not in tok.special_ids],
                                                                   skip_special_tokens=False)
                want_all = ref.decode(ids, skip_special_tokens=False)
            except BaseException as e:            # pyo3's PanicException: Strip(stop) on a short text
                if isinstance(e, (KeyboardInterrupt, SystemExit)):
                    raise
                panics += 1
                continue
            ok &= tok.decode(ids) == want and tok.decode(ids, False) == want_all
            for strip in (True, False):
                inc = tok.incremental_decoder(strip_first=strip)
                ok &= "".join(inc.add(i) for i in ids) + inc.flush() == tok.decode(ids, strip_first=strip)
            if not ok:
                bad += 1
                if bad <= 2:
                    print(f"  MISMATCH {name}: {s[:40]!r}\n    ours {tok.encode(s)[:16]}\n    ref  {ids[:16]}\n"
                          f"    {tok.decode(ids)[:60]!r} / {want[:60]!r}")
        good &= bad == 0
        print(f"synthetic {name}: {'OK' if not bad else f'{bad} FAILED'}"
              + (f" ({panics} strings skipped: tokenizers panics decoding them)" if panics else ""))
    return good


# ── pinned tokenizers ────────────────────────────────────────────────────────

def pinned():
    with open(PINS, encoding="utf-8") as f:
        return json.load(f)["tokenizers"]


def fetch() -> int:
    """The files pinned in tokenizer_assets.json -> assets/tokenizers/<name>/."""
    from pathlib import Path
    from llm_calibrate import HF, download, sha256_file
    for name, spec in pinned().items():
        for fname, sha in spec["files"].items():
            p = Path(PINNED_DIR) / name / fname
            if p.exists() and sha256_file(p) == sha:
                print(f"  {name}/{fname}: kept")
                continue
            download(f"{HF}/{spec['repo']}/resolve/{spec['revision']}/{fname}", p, sha)
            print(f"  {name}/{fname}: {sha[:16]}… ({spec['repo']} @ {spec['revision'][:12]})")
    return 0


FIXTURE_ALL = os.path.join(CHAT, "tests", "data", "hf_tokenizer_cases.json")
FIXTURE_LABELS = ("smolvlm-256m-instruct", "moonshine-base")      # + every pinned one


def fixture_entry(path):
    """~40 strings of one tokenizer with the reference ids, ids without the
    post-processor, decode (special tokens skipped by id) and decode with them."""
    from tokenizers import Tokenizer as Ref
    import hashlib
    ref, tok = Ref.from_file(path), Tokenizer(path)
    rng = random.Random(9)
    texts = (LLAMA_CASES[:12] + rng.sample(unicode_corpus(), 12) + rng.sample(special_cases(tok), 12)
             + [SCRIPTS["latin"], CODE[0], "  1234567 ٣٤٥ IT'S"])
    cases = []
    for s in texts:
        ids = ref.encode(s).ids
        dec = ref.decode(ids)
        if tok.decode(ids) != dec:                    # a normalized special token tokenizers keeps
            dec = ref.decode([i for i in ids if i not in tok.special_ids], skip_special_tokens=False)
        cases.append({"text": s, "ids": ids, "plain": ref.encode(s, add_special_tokens=False).ids,
                      "decode": dec, "decode_all": ref.decode(ids, skip_special_tokens=False)})
    with open(path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    return {"path": os.path.relpath(path, CHAT), "sha256": sha, "cases": cases}


def main_all(write_fixture: bool = False) -> int:
    strings = make_strings() + LLAMA_CASES + unicode_corpus()
    targets = [(m, os.path.join(CHAT, "assets", m)) for m in ("smollm2-135m-instruct", "smolvlm-256m-instruct",
                                                             "stories15m")]
    targets += [(m, os.path.join(ASR_ASSETS, m)) for m in ("moonshine-base", "moonshine-tiny")]
    targets += [(name, os.path.join(PINNED_DIR, name)) for name in pinned()]
    good, missing = True, []
    for label, d in targets:
        path = os.path.join(d, "tokenizer.json")
        if not os.path.exists(path):
            missing.append(label)
            print(f"{label}: SKIPPED ({path} not present)")
            continue
        try:
            good &= check_tokenizer(path, strings, label)[0]
        except ValueError as e:
            good = False
            print(f"{label}: REFUSED — {e}")
    good &= check_regex()
    good &= check_synthetic()
    print(("all identical" if good else "MISMATCHES") + (f"; skipped: {', '.join(missing)}" if missing else ""))
    if write_fixture:
        entries = {label: fixture_entry(os.path.join(d, "tokenizer.json")) for label, d in targets
                   if (label in FIXTURE_LABELS or label in pinned()) and label not in missing}
        with open(FIXTURE_ALL, "w", encoding="utf-8") as f:
            json.dump({"source": "tokenizers via demo/chat/scripts/validate_text.py --all --write-fixture",
                       "tokenizers": entries}, f, ensure_ascii=True, indent=None, separators=(",", ":"))
        print(f"wrote {FIXTURE_ALL} ({os.path.getsize(FIXTURE_ALL) // 1024} KiB, {len(entries)} tokenizers)")
    return 0 if good else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-fixture", action="store_true")
    ap.add_argument("--model", default=None,
                    help="a text-profile checkpoint in assets/ (stories15m): its tokenizer vs tokenizers")
    ap.add_argument("--assets", default=None, help="a directory with a tokenizer.json: hf_tokenizer vs tokenizers")
    ap.add_argument("--all", action="store_true", help="every tokenizer at hand, the regex and synthetic checks")
    ap.add_argument("--regex", action="store_true", help="onig_to_re vs tokenizers' Split")
    ap.add_argument("--synthetic", action="store_true", help="synthetic pipelines vs tokenizers")
    ap.add_argument("--fetch", action="store_true", help="download the tokenizers pinned in tokenizer_assets.json")
    args = ap.parse_args()
    if args.fetch:
        return fetch()
    if args.all:
        return main_all(args.write_fixture)
    if args.regex or args.synthetic:
        good = (not args.regex or check_regex()) & (not args.synthetic or check_synthetic())
        return 0 if good else 1
    if args.assets:
        return 0 if check_tokenizer(os.path.join(args.assets, "tokenizer.json"),
                                    make_strings() + LLAMA_CASES + unicode_corpus(),
                                    os.path.basename(os.path.normpath(args.assets)))[0] else 1
    if args.model:
        return main_llama(args.model, args.write_fixture)
    import transformers
    from transformers import AutoTokenizer
    hf = transformers.TokenizersBackend.from_pretrained(ASSETS)
    auto = AutoTokenizer.from_pretrained(ASSETS)
    t0 = time.time()
    tok = Tokenizer(os.path.join(ASSETS, "tokenizer.json"))
    print(f"pure-Python tokenizer loaded in {time.time() - t0:.2f} s; reference transformers "
          f"{transformers.__version__} {type(hf).__name__} (pre_tokenizer "
          f"{hf.backend_tokenizer.pre_tokenizer}); also AutoTokenizer = {type(auto).__name__}")
    with open(os.path.join(ASSETS, "tokenizer.json"), encoding="utf-8") as f:
        spec = json.load(f)
    spec["pre_tokenizer"]["pretokenizers"] = [p for p in spec["pre_tokenizer"]["pretokenizers"]
                                              if p["type"] != "Digits"]
    nodigits = Tokenizer(spec)                               # the Digits step removed

    strings = make_strings()
    bad = 0
    t_ours = t_hf = 0.0
    n_tok = 0
    cases = []
    auto_diff = auto_explained = 0
    for s in strings:
        t = time.perf_counter()
        ours = tok.encode(s)
        t_ours += time.perf_counter() - t
        t = time.perf_counter()
        ref = hf.encode(s, add_special_tokens=False)
        t_hf += time.perf_counter() - t
        n_tok += len(ref)
        ok = ours == ref
        dec, dec_ref = tok.decode(ours), hf.decode(ref, skip_special_tokens=True)
        dec2, dec2_ref = tok.decode(ours, False), hf.decode(ref, skip_special_tokens=False)
        ok &= dec == dec_ref and dec2 == dec2_ref
        if ok:
            inc = tok.incremental_decoder()
            ok &= "".join(inc.add(i) for i in ours) + inc.flush() == dec
        if not ok:
            bad += 1
            if bad <= 10:
                print(f"MISMATCH {s[:80]!r}\n  ours {ours[:30]}\n  hf   {ref[:30]}")
        cases.append((s, ref))
        a = auto.encode(s, add_special_tokens=False)
        if a != ref:
            auto_diff += 1
            auto_explained += int(a == nodigits.encode(s)
                                  and any(unicodedata.category(c)[0] == "N" for c in s))
    lossless = sum(1 for s, ids in cases if tok.decode(ids, False) == s)
    print(f"tokenizer: {len(strings) - bad}/{len(strings)} strings identical (ids, decode with and "
          f"without special tokens, incremental decode); {n_tok} tokens; round trip lossless on "
          f"{lossless}/{len(strings)} (the rest contain control / unmapped bytes the vocabulary "
          f"drops, identically in HF)")
    print(f"  transformers AutoTokenizer ({type(auto).__name__}) differs on {auto_diff} strings, "
          f"{auto_explained} of them explained by its missing Digits pre-tokenizer (numerals after "
          f"a whitespace run, non-ASCII numerals; our tokenizer without the Digits step gives its "
          f"ids)")
    print(f"  encode time: ours {t_ours * 1e3:.0f} ms ({n_tok / t_ours / 1e3:.0f} k tok/s, cache warm "
          f"after the first use of a word), HF {t_hf * 1e3:.0f} ms")

    rng = random.Random(99)
    bad_dec = 0
    V = tok.vocab_size
    for k in range(1000):
        ids = [rng.randrange(V) for _ in range(rng.randint(1, 30))]
        if k % 3 == 0:                                   # plenty of partial UTF-8 bytes
            ids = [rng.choice(range(100, 400)) for _ in range(rng.randint(1, 30))]
        for skip in (True, False):
            a, b = tok.decode(ids, skip), hf.decode(ids, skip_special_tokens=skip)
            inc = tok.incremental_decoder(skip)
            c = "".join(inc.add(i) for i in ids) + inc.flush()
            if not (a == b == c):
                bad_dec += 1
                if bad_dec <= 5:
                    print(f"DECODE MISMATCH {ids}: {a!r} / {b!r} / {c!r}")
    print(f"decode of random id sequences: {2000 - bad_dec}/2000 identical (incl. incremental)")

    convs = conversations()
    tmpl = chat_template.ChatTemplate.from_checkpoint(ASSETS)   # what the server renders
    bad_t = 0
    tcases = []
    for c in convs:
        for gen in (True, False):
            ref_text = hf.apply_chat_template(c, tokenize=False, add_generation_prompt=gen)
            r = hf.apply_chat_template(c, tokenize=True, add_generation_prompt=gen)
            ref_ids = list(r["input_ids"] if not isinstance(r, list) else r)
            ours_text = tmpl.render(c, gen)
            ours_ids = tok.encode(ours_text, add_special_tokens=False)
            pb = chat_template.PromptBuilder(tmpl, lambda t: tok.encode(t, add_special_tokens=False))
            if not (ours_text == ref_text and ours_ids == ref_ids and pb.ids(c, gen) == ref_ids):
                bad_t += 1
                print(f"TEMPLATE MISMATCH {c}\n  {ours_text!r}\n  {ref_text!r}")
            if gen:
                tcases.append((c, ref_ids))
    print(f"chat template: {2 * len(convs) - bad_t}/{2 * len(convs)} renderings identical "
          f"({len(convs)} conversations x add_generation_prompt True / False; text, ids, and "
          f"the server's prompt builder, chat_template.py)")

    if args.write_fixture:
        rng = random.Random(5)
        sub = [c for c in cases if len(c[0]) <= 400]
        sub = sub[:120] + rng.sample(sub[120:], 380)
        os.makedirs(os.path.dirname(FIXTURE), exist_ok=True)
        with open(FIXTURE, "w", encoding="utf-8") as f:
            json.dump({"source": "transformers AutoTokenizer(SmolLM2-135M-Instruct) via "
                                 "demo/chat/scripts/validate_text.py --write-fixture",
                       "strings": [{"text": s, "ids": ids} for s, ids in sub],
                       "conversations": [{"messages": c, "ids": ids} for c, ids in tcases
                                         if len(ids) < 1200]},
                      f, ensure_ascii=True, indent=None, separators=(",", ":"))
        print(f"wrote {FIXTURE} ({os.path.getsize(FIXTURE) // 1024} KiB)")
    return 0 if bad == bad_dec == bad_t == 0 and auto_diff == auto_explained else 1




if __name__ == "__main__":
    sys.exit(main())
