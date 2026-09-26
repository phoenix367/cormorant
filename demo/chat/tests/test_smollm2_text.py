"""smollm2_tokenizer.py and chatml.py: ids and decoding against the
transformers reference ids stored in data/smollm2_text_cases.json (written by
scripts/validate_text.py --write-fixture, which also runs the full >= 2 000
string / 34 conversation validation against transformers), the incremental
detokenizer, and history trimming."""

import json
import os
import random
import unittest

from _util import CHAT

import chatml
from smollm2_tokenizer import Tokenizer, bytes_to_unicode

TOKENIZER = os.path.join(CHAT, "assets", "smollm2-135m-instruct", "tokenizer.json")
FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "smollm2_text_cases.json")
HAVE = os.path.exists(TOKENIZER)


@unittest.skipUnless(HAVE, "demo/chat/assets/smollm2-135m-instruct/tokenizer.json not present")
class TestTokenizer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tok = Tokenizer(TOKENIZER)
        with open(FIXTURE, encoding="utf-8") as f:
            cls.fx = json.load(f)

    def test_fixture_ids_and_decode(self):
        self.assertGreaterEqual(len(self.fx["strings"]), 400)
        for c in self.fx["strings"]:
            with self.subTest(text=c["text"][:60]):
                ids = self.tok.encode(c["text"])
                self.assertEqual(ids, c["ids"])
                self.assertEqual(self.tok.decode(ids, False), self.tok.decode(c["ids"], False))

    def test_specials(self):
        t = self.tok
        self.assertEqual(t.vocab_size, 49152)
        self.assertEqual((t.special["<|endoftext|>"], t.bos_id, t.eos_id), (0, 1, 2))
        hi = t.encode("Hi")
        self.assertEqual(t.encode("<|im_start|>user\nHi<|im_end|>\n"), [1, 4093, 198] + hi + [2, 198])
        self.assertEqual(t.encode("<|im_start|>", special=False)[0], t.encode("<")[0])
        self.assertEqual(t.decode([1] + hi + [2]), "Hi")
        self.assertEqual(t.decode([1] + hi + [2], skip_special_tokens=False), "<|im_start|>Hi<|im_end|>")
        self.assertEqual(t.encode(""), [])

    def test_digits_and_whitespace(self):
        t = self.tok
        self.assertEqual([t.id_to_token[i] for i in t.encode("in 2024")], ["in", "Ġ", "2", "0", "2", "4"])
        self.assertEqual([t.id_to_token[i] for i in t.encode("a  1")], ["a", "ĠĠ", "1"])
        self.assertEqual([t.id_to_token[i] for i in t.encode("x  y")], ["x", "Ġ", "Ġy"])
        self.assertEqual(t.encode("in\x04g"), t.encode("in") + t.encode("g"))   # unmapped byte dropped

    def test_round_trip(self):
        rng = random.Random(7)
        alphabet = "abc XYZ 012 éü 中文 😀👍🏽 \n\t  ,.!?'s-_/\\\"<|>"
        for _ in range(300):
            s = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 60)))
            self.assertEqual(self.tok.decode(self.tok.encode(s)), s)

    def test_incremental_decoder(self):
        t = self.tok
        rng = random.Random(3)
        texts = ["Hello, world! 😀 中文 naïve", "👨‍👩‍👧‍👦 and 🇫🇷", "plain ascii"]
        for s in texts:
            ids = t.encode(s)
            dec = t.incremental_decoder()
            pieces = [dec.add(i) for i in ids]
            self.assertEqual("".join(pieces) + dec.flush(), s)
            for p in pieces:
                self.assertNotIn("�", p)                    # never a half character
        for _ in range(300):                                    # random ids: == decode()
            ids = [rng.randrange(t.vocab_size) for _ in range(rng.randint(1, 20))]
            if rng.random() < 0.5:
                ids = [rng.randrange(100, 400) for _ in ids]     # byte tokens: partial UTF-8
            dec = t.incremental_decoder()
            out = "".join(dec.add(i) for i in ids) + dec.flush()
            self.assertEqual(out, t.decode(ids))
        dec = t.incremental_decoder()
        emoji = t.encode("😀")
        self.assertGreater(len(emoji), 1)
        self.assertEqual(dec.add(emoji[0]), "")
        self.assertTrue(dec.pending)

    def test_byte_map(self):
        m = bytes_to_unicode()
        self.assertEqual(len(set(m.values())), 256)
        self.assertEqual(m[32], "Ġ")


@unittest.skipUnless(HAVE, "demo/chat/assets/smollm2-135m-instruct/tokenizer.json not present")
class TestChatML(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tok = Tokenizer(TOKENIZER)
        with open(FIXTURE, encoding="utf-8") as f:
            cls.fx = json.load(f)

    def test_render_matches_hf_ids(self):
        self.assertGreaterEqual(len(self.fx["conversations"]), 30)
        pb = chatml.PromptBuilder(self.tok.encode)
        for c in self.fx["conversations"]:
            with self.subTest(messages=str(c["messages"])[:80]):
                self.assertEqual(self.tok.encode(chatml.render(c["messages"])), c["ids"])
                self.assertEqual(pb.ids(c["messages"]), c["ids"])

    def test_render_text(self):
        self.assertEqual(chatml.render([{"role": "user", "content": "Hi"}]),
                         "<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by "
                         "Hugging Face<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n")
        self.assertEqual(chatml.render([{"role": "system", "content": "S"},
                                        {"role": "user", "content": ""}], add_generation_prompt=False),
                         "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\n<|im_end|>\n")

    def _conv(self, n_turns, words=20, system=True):
        m = [{"role": "system", "content": "Be brief."}] if system else []
        for i in range(n_turns):
            m.append({"role": "user", "content": f"question {i} " + "word " * words})
            m.append({"role": "assistant", "content": f"answer {i} " + "word " * words})
        m.append({"role": "user", "content": "last question"})
        return m

    def test_fit_no_trim(self):
        pb = chatml.PromptBuilder(self.tok.encode)
        m = self._conv(2)
        ids, kept, dropped = pb.fit(m, 10_000)
        self.assertEqual((ids, kept, dropped), (pb.ids(m), m, 0))

    def test_fit_drops_oldest_turns(self):
        pb = chatml.PromptBuilder(self.tok.encode)
        m = self._conv(6)
        full = len(pb.ids(m))
        ids, kept, dropped = pb.fit(m, full - 5)
        self.assertLessEqual(len(ids), full - 5)
        self.assertEqual(kept[0], m[0])                          # the system message stays
        self.assertEqual(kept[-1], m[-1])                        # the last message stays
        self.assertEqual(kept[1]["role"], "user")                # history starts with a user turn
        self.assertEqual(dropped, 2)                             # one whole turn
        self.assertEqual(kept[1:], m[1 + dropped:])
        self.assertEqual(ids, pb.ids(kept))
        # a budget just above system + last message keeps only those
        minimal = len(pb.ids([m[0], m[-1]]))
        ids, kept, dropped = pb.fit(m, minimal)
        self.assertEqual(kept, [m[0], m[-1]])
        self.assertEqual(dropped, len(m) - 2)
        with self.assertRaises(chatml.ContextTooLong) as cm:
            pb.fit(m, minimal - 1)
        self.assertEqual((cm.exception.needed, cm.exception.budget), (minimal, minimal - 1))

    def test_fit_default_system(self):
        pb = chatml.PromptBuilder(self.tok.encode)
        m = self._conv(3, system=False)
        full = pb.ids(m)
        ids, kept, dropped = pb.fit(m, len(full) - 1)
        self.assertEqual(kept[0]["role"], "user")
        self.assertEqual(ids[:len(pb.block_ids(chatml.block("system", chatml.DEFAULT_SYSTEM)))],
                         list(pb.block_ids(chatml.block("system", chatml.DEFAULT_SYSTEM))))
        self.assertEqual(ids, self.tok.encode(chatml.render(kept)))

    def test_fit_assistant_first_and_single_message(self):
        pb = chatml.PromptBuilder(self.tok.encode)
        m = [{"role": "assistant", "content": "x " * 50}, {"role": "user", "content": "a " * 50},
             {"role": "assistant", "content": "y " * 50}, {"role": "user", "content": "q"}]
        ids, kept, dropped = pb.fit(m, len(pb.ids(m)) - 1)
        self.assertEqual(kept, m[1:] if len(pb.ids(m[1:])) < len(pb.ids(m)) - 1 else m[3:])
        with self.assertRaises(chatml.ContextTooLong):
            pb.fit([{"role": "user", "content": "long " * 500}], 100)
        with self.assertRaises(chatml.ContextTooLong):
            pb.fit([{"role": "system", "content": "long " * 500}], 100)

    def test_block_cache(self):
        calls = []

        def enc(s):
            calls.append(s)
            return self.tok.encode(s)
        pb = chatml.PromptBuilder(enc)
        m = self._conv(3)
        pb.ids(m)
        n = len(calls)
        pb.ids(m + [{"role": "assistant", "content": "new"}, {"role": "user", "content": "more"}])
        self.assertEqual(len(calls), n + 2)                      # only the two new blocks


if __name__ == "__main__":
    unittest.main()
