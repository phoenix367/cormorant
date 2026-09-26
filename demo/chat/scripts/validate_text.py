#!/usr/bin/env python3
"""Validate the pure-Python SmolLM2 tokenizer (smollm2_tokenizer.py) and
chat template (chatml.py) against transformers (CHAT_PLAN phase 4, gate 1).

  * tokenizer: >= 2 000 deterministic, diverse strings — held-out / calibration
    text lines, many scripts, emoji and ZWJ sequences, combining marks, number
    forms, code, JSON / HTML, whitespace runs (tabs, CR LF, NBSP, ideographic
    space, control characters), contractions, special tokens inside text,
    random code points (all planes, unassigned, private use) — token ids,
    decode(encode(s)) and HF decode identical; plus random id sequences (broken
    UTF-8 included) decoded identically, with and without special tokens, and
    the incremental decoder equal to decode() under every split;
  * template: >= 30 conversations (system / no system, multi-turn, empty
    content, special roles) — rendered text and ids identical to
    apply_chat_template(add_generation_prompt=True / False).

  The reference is transformers' TokenizersBackend.from_pretrained (the
  tokenizer.json pipeline as the `tokenizers` library runs it — what the model
  was trained with, and what transformers 4.x AutoTokenizer returns).
  transformers 5.x AutoTokenizer builds a GPT2Tokenizer class that drops
  tokenizer.json's Digits pre-tokenizer; its ids are compared too, and every
  difference must be that omission (our tokenizer with the Digits step
  disabled reproduces them) — only strings with a numeral after two or more
  whitespace characters, or with non-ASCII numerals (Arabic-Indic,
  Devanagari, Ethiopic, ...), are affected.

  --write-fixture writes demo/chat/tests/data/smollm2_text_cases.json (a
  subset with the reference ids) for the stdlib unit tests.

Usage (.venv-export: transformers, tokenizers):
  /home/ivan/projects/axi_demo/.venv-export/bin/python demo/chat/scripts/validate_text.py [--write-fixture]
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

import chatml                                               # noqa: E402
from smollm2_tokenizer import Tokenizer                      # noqa: E402

ASSETS = os.environ.get("SMOLLM_ASSETS", os.path.join(CHAT, "assets", "smollm2-135m-instruct"))
FIXTURE = os.path.join(CHAT, "tests", "data", "smollm2_text_cases.json")

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-fixture", action="store_true")
    args = ap.parse_args()
    import re
    import transformers
    from transformers import AutoTokenizer
    hf = transformers.TokenizersBackend.from_pretrained(ASSETS)
    auto = AutoTokenizer.from_pretrained(ASSETS)
    t0 = time.time()
    tok = Tokenizer(os.path.join(ASSETS, "tokenizer.json"))
    print(f"pure-Python tokenizer loaded in {time.time() - t0:.2f} s; reference transformers "
          f"{transformers.__version__} {type(hf).__name__} (pre_tokenizer "
          f"{hf.backend_tokenizer.pre_tokenizer}); also AutoTokenizer = {type(auto).__name__}")
    nodigits = Tokenizer(os.path.join(ASSETS, "tokenizer.json"))
    nodigits._split_num = re.compile("(?!)")               # the Digits step disabled

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
    bad_t = 0
    tcases = []
    for c in convs:
        for gen in (True, False):
            ref_text = hf.apply_chat_template(c, tokenize=False, add_generation_prompt=gen)
            r = hf.apply_chat_template(c, tokenize=True, add_generation_prompt=gen)
            ref_ids = list(r["input_ids"] if not isinstance(r, list) else r)
            ours_text = chatml.render(c, gen)
            ours_ids = tok.encode(ours_text)
            pb = chatml.PromptBuilder(tok.encode)
            if not (ours_text == ref_text and ours_ids == ref_ids and pb.ids(c, gen) == ref_ids):
                bad_t += 1
                print(f"TEMPLATE MISMATCH {c}\n  {ours_text!r}\n  {ref_text!r}")
            if gen:
                tcases.append((c, ref_ids))
    print(f"chat template: {2 * len(convs) - bad_t}/{2 * len(convs)} renderings identical "
          f"({len(convs)} conversations x add_generation_prompt True / False; text, ids, and "
          f"block-wise ids)")

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
