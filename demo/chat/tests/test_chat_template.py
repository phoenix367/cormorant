# Copyright 2026 Ivan Gubochkin
# SPDX-License-Identifier: Apache-2.0

"""chat_template.py: a checkpoint's own chat template rendered as
apply_chat_template does, and the prompt builder (whole-text tokenization,
trimming).  Synthetic templates; SmolLM2's checkpoint where its assets are
present (its template must render as chatml.py did)."""

import json
import os
import tempfile
import unittest

from _util import CHAT

import chat_template as ct
import chatml

CHATML = ("{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}"
          "{{ '<|im_start|>system\\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face"
          "<|im_end|>\\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + "
          "'<|im_end|>' + '\\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}"
          "{% endif %}")
SMOLLM2 = os.path.join(CHAT, "assets", "smollm2-135m-instruct")
CONV = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "What is 2 + 2?"}]


def encode(text):
    """A stand-in tokenizer: one id per character (special tokens are text)."""
    return [ord(c) for c in text]


class TestRender(unittest.TestCase):

    def test_chatml_as_chatml_py(self):
        t = ct.ChatTemplate(CHATML)
        for msgs in (CONV, [{"role": "system", "content": "Be brief."}] + CONV, CONV[:1]):
            self.assertEqual(t.render(msgs), chatml.render(msgs))
            self.assertEqual(t.render(msgs, add_generation_prompt=False), chatml.render(msgs, False))

    def test_turn(self):
        t = ct.ChatTemplate(CHATML)
        follow = [{"role": "user", "content": "And 3 + 3?"}]
        self.assertEqual(t.turn(CONV, follow),
                         "<|im_end|>\n<|im_start|>user\nAnd 3 + 3?<|im_end|>\n<|im_start|>assistant\n")

    def test_transformers_environment(self):
        """Special tokens as variables, trim_blocks / lstrip_blocks, tojson
        without HTML escaping, raise_exception, loop controls, {% generation %}."""
        src = ("{{ bos_token }}{% for m in messages %}\n"
               "  {% if m['role'] == 'tool' %}{{ raise_exception('no tools') }}{% endif %}\n"
               "  {% if loop.index > 2 %}{% break %}{% endif %}\n"
               "{% generation %}{{ m['content'] | tojson }}{% endgeneration %}\n"
               "{% endfor %}")
        t = ct.ChatTemplate(src, {"bos_token": "<s>"})
        self.assertEqual(t.render([{"role": "user", "content": "<a & b>"}, {"role": "user", "content": "é"},
                                   {"role": "user", "content": "x"}]),
                         '<s>"<a & b>""é"')                 # transformers' own output
        with self.assertRaises(Exception) as cm:
            t.render([{"role": "tool", "content": "x"}])
        self.assertIn("no tools", str(cm.exception))

    def test_template_sources(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(ct.template_source(d))
            with open(os.path.join(d, "chat_template.json"), "w") as f:
                json.dump({"chat_template": "json"}, f)
            self.assertEqual(ct.template_source(d), "json")
            with open(os.path.join(d, "chat_template.jinja"), "w") as f:
                f.write("jinja")
            self.assertEqual(ct.template_source(d), "jinja")
            with open(os.path.join(d, "tokenizer_config.json"), "w") as f:
                json.dump({"chat_template": [{"name": "tool_use", "template": "t"},
                                             {"name": "default", "template": "config"}],
                           "bos_token": {"content": "<s>"}, "eos_token": "</s>", "pad_token": None}, f)
            self.assertEqual(ct.template_source(d), "config")
            self.assertEqual(ct.special_tokens(d), {"bos_token": "<s>", "eos_token": "</s>"})


class TestPromptBuilder(unittest.TestCase):

    def test_ids_and_trim(self):
        pb = ct.PromptBuilder(ct.ChatTemplate(CHATML), encode)
        self.assertEqual(pb.ids(CONV), encode(chatml.render(CONV)))
        long = CONV + [{"role": "assistant", "content": "4"}, {"role": "user", "content": "Bye"}]
        full = len(pb.ids(long))
        ids, kept, dropped = pb.fit(long, full)
        self.assertEqual((dropped, kept), (0, long))
        ids, kept, dropped = pb.fit(long, full - 1)          # the first user + assistant turn goes
        self.assertEqual(dropped, 2)
        self.assertEqual(kept[0], long[2])
        with self.assertRaises(ct.ContextTooLong):
            pb.fit(long, 10)

    @unittest.skipUnless(os.path.exists(os.path.join(SMOLLM2, "tokenizer.json")), "SmolLM2 assets")
    def test_smollm2_checkpoint_ids_as_chatml(self):
        """SmolLM2's own template + tokenizer give chatml.py's block-wise ids."""
        from smollm2_tokenizer import Tokenizer
        tok = Tokenizer(os.path.join(SMOLLM2, "tokenizer.json"))
        pb = ct.PromptBuilder(ct.ChatTemplate.from_checkpoint(SMOLLM2), tok.encode)
        old = chatml.PromptBuilder(tok.encode)
        for msgs in (CONV, [{"role": "system", "content": "Be brief. ünïcode ✓"}] + CONV):
            self.assertEqual(pb.ids(msgs), old.ids(msgs))


if __name__ == "__main__":
    unittest.main()
