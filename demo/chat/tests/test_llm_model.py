# Copyright 2026 Ivan Gubochkin
# SPDX-License-Identifier: Apache-2.0

"""scripts/llm_model.py: a checkpoint's names and profile (chat or text, the
sink token, the end-of-generation ids, the context) from its own files —
synthetic checkpoints shaped like SmolLM2's, stories15m's, a Llama-3-style
one and SmolVLM's (doc/plans/CHAT_PLAN.md §28)."""

import json
import os
import sys
import tempfile
import unittest

from _util import CHAT

sys.path.insert(0, os.path.join(CHAT, "scripts"))
import llm_model as lm  # noqa: E402

CHATML = ("{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}"
          "{{ '<|im_start|>system\nYou are a helpful AI assistant<|im_end|>\n' }}{% endif %}"
          "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
          "{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}")
LLAMA3 = ("{{- bos_token }}{% for message in messages %}{{ '<|start_header_id|>' + message['role'] "
          "+ '<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}{% endfor %}")


class _Ckpt(unittest.TestCase):

    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = self.td.name

    def tearDown(self):
        self.td.cleanup()

    def ckpt(self, name, config, added, tok_cfg=None, gen=None, extra=None):
        d = os.path.join(self.root, name)
        os.makedirs(d)
        files = {"config.json": config,
                 "tokenizer.json": {"added_tokens": [{"id": i, "content": c, "special": True}
                                                     for c, i in added.items()]}}
        if tok_cfg is not None:
            files["tokenizer_config.json"] = tok_cfg
        if gen is not None:
            files["generation_config.json"] = gen
        for n, v in {**files, **(extra or {})}.items():
            with open(os.path.join(d, n), "w") as f:
                f.write(v if isinstance(v, str) else json.dumps(v))
        return d


class TestProfile(_Ckpt):

    def test_smollm2_chat(self):
        d = self.ckpt("smollm2-135m-instruct",
                      {"max_position_embeddings": 8192, "bos_token_id": 1, "eos_token_id": 2},
                      {"<|endoftext|>": 0, "<|im_start|>": 1, "<|im_end|>": 2},
                      {"chat_template": CHATML, "bos_token": "<|im_start|>", "eos_token": "<|im_end|>"},
                      {"bos_token_id": 1, "eos_token_id": 2})
        p = lm.Profile(d)
        self.assertEqual((p.kind, p.sink, p.eog, p.ctx, p.windows), ("chat", 1, (2,), 1024, 3))

    def test_stories_text(self):
        d = self.ckpt("stories15m", {"max_position_embeddings": 256, "bos_token_id": 1, "eos_token_id": 2},
                      {"<unk>": 0, "<s>": 1, "</s>": 2},
                      {"bos_token": {"content": "<s>"}, "eos_token": {"content": "</s>"}},
                      {"bos_token_id": 1, "eos_token_id": 2})
        p = lm.Profile(d)
        # a base model stops at the next document's BOS too
        self.assertEqual((p.kind, p.sink, p.bos, p.eog, p.ctx, p.windows), ("text", 1, 1, (1, 2), 256, 12))

    def test_llama3_style_ids(self):
        """Other ids than SmolLM2's: the sink is the BOS the template writes
        first, the end ids the union of the configs' (several in a list)."""
        d = self.ckpt("llama-3-mini", {"max_position_embeddings": 131072, "bos_token_id": 128000,
                                       "eos_token_id": [128001, 128008, 128009]},
                      {"<|begin_of_text|>": 128000, "<|end_of_text|>": 128001, "<|eot_id|>": 128009,
                       "<|start_header_id|>": 128006, "<|end_header_id|>": 128007},
                      {"chat_template": LLAMA3, "bos_token": "<|begin_of_text|>", "eos_token": "<|eot_id|>"},
                      {"bos_token_id": 128000, "eos_token_id": [128001, 128009]})
        p = lm.Profile(d)
        self.assertEqual(p.kind, "chat")
        self.assertEqual(p.sink, 128000)              # {{- bos_token }} comes first
        self.assertEqual(p.eog, (128001, 128008, 128009))
        self.assertEqual(p.ctx, 1024)

    def test_template_files(self):
        """A template beside the tokenizer config: chat_template.jinja / .json."""
        base = ({"max_position_embeddings": 2048}, {"<|im_start|>": 1, "<|im_end|>": 2},
                {"eos_token": "<|im_end|>"})
        p = lm.Profile(self.ckpt("a", *base, extra={"chat_template.jinja": CHATML}))
        self.assertEqual((p.kind, p.sink, p.eog), ("chat", 1, (2,)))
        p = lm.Profile(self.ckpt("b", *base, extra={"chat_template.json": {"chat_template": CHATML}}))
        self.assertEqual(p.kind, "chat")
        p = lm.Profile(self.ckpt("c", *base[:2], {"chat_template": [{"name": "default", "template": CHATML}],
                                                  "eos_token": "<|im_end|>"}))
        self.assertEqual(p.kind, "chat")

    def test_refusals(self):
        d = self.ckpt("nobos", {"max_position_embeddings": 512, "eos_token_id": 2}, {"</s>": 2}, {})
        with self.assertRaises(lm.ProfileError):
            lm.Profile(d)                             # a text model needs a BOS for the sink
        d = self.ckpt("noeos", {"bos_token_id": 1}, {"<|im_start|>": 1},
                      {"chat_template": "<|im_start|>{{ m }}"})
        with self.assertRaises(lm.ProfileError):
            lm.Profile(d)

    def test_llm_study_renders_chatml_only(self):
        """llm_study.Profile refuses another template (it would prompt with ChatML)."""
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("numpy")
        import llm_study
        d = self.ckpt("llama-3-mini", {"max_position_embeddings": 4096, "bos_token_id": 128000,
                                       "eos_token_id": 128009},
                      {"<|begin_of_text|>": 128000, "<|eot_id|>": 128009, "<|start_header_id|>": 128006},
                      {"chat_template": LLAMA3})
        with self.assertRaises(lm.ProfileError):
            llm_study.Profile(d)


class TestNames(unittest.TestCase):

    def test_tag_and_paths(self):
        self.assertEqual(lm.tag("smollm2-360m-instruct"), "smollm2_360m")
        self.assertEqual(lm.tag("qwen2.5-0.5b-instruct"), "qwen2_5_0_5b")
        self.assertEqual(lm.board_paths("smollm2-360m-instruct", "/r"),
                         {"lib": "/r/lib/libsmollm2_360m.so", "proj": "/r/llm_project_smollm2_360m",
                          "weights": "/root/smollm2_360m_weights"})
        self.assertEqual(lm.board_paths("stories15m", "/r", "/w")["weights"], "/w")
        self.assertTrue(lm.project_dir("stories15m", "/b").endswith("/b/llm_project_stories15m"))

    def test_legacy_135m(self):
        name = lm.DEFAULT_MODEL
        self.assertEqual(lm.board_paths(name, "/r"),
                         {"lib": "/r/lib/libsmollm2.so", "proj": "/r/llm_project",
                          "weights": "/root/smollm2_weights"})
        self.assertEqual(lm.project_dir(name, "/b"), "/b/llm_project")
        self.assertEqual(lm.study_dir(f"/a/assets/{name}"), "/a/assets/study")
        self.assertEqual(lm.study_dir("/a/assets/stories15m"), "/a/assets/study/stories15m")


if __name__ == "__main__":
    unittest.main()
