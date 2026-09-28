"""Backend C (smolvlm_backend.py) with a fake engine: the Idefics3 template
against transformers' processor (tests/data/smolvlm_cases.json), image
decoding against the study's pixels, and the backend's calls — llm_image()
before each new image block, image tokens as vocab + row, prefix reuse keyed
by image content, the answer's leading space, trimming with images, the
server's image parts (and a text model's 400) over HTTP."""

import copy
import json
import os
import unittest

from _util import CHAT, RunningServer, sampler_lib, server_mod

import idefics3
from chat_backend import BackendError, CancelToken, ChatRequest, Delta, Finish
from fake_llm import FakeLibraryError, ScriptedEngine
from sampler import SamplerParams

TOKENIZER = os.path.join(CHAT, "assets", "smolvlm-256m-instruct", "tokenizer.json")
CASES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "smolvlm_cases.json")
HAVE = os.path.exists(TOKENIZER)
try:
    import PIL  # noqa: F401
    HAVE_PIL = True
except ImportError:
    HAVE_PIL = False
VOCAB = 49280
_TOK = []


def tok():
    from smollm2_tokenizer import Tokenizer
    if not _TOK:
        _TOK.append(Tokenizer(TOKENIZER))
    return _TOK[0]


def cases():
    return json.load(open(CASES))


class FakeVlmEngine(ScriptedEngine):
    """ScriptedEngine + llm_image(): answers `reply` after "Assistant:"."""

    image_tokens, image_size = 64, 512

    def __init__(self, reply, **kw):
        super().__init__(reply, vocab_size=VOCAB, **kw)
        self.gen = tuple(tok().encode(idefics3.GENERATION_PROMPT))
        self.next_token = self._vlm_script
        self.images = []
        self.image_rows = None

    def _vlm_script(self, seq):
        g = len(self.gen)
        for i in range(len(seq) - g, -1, -1):
            if tuple(seq[i:i + g]) == self.gen:
                k = len(seq) - (i + g)
                return self.reply[k] if k < len(self.reply) else tok().special["<end_of_utterance>"]
        return tok().special["<end_of_utterance>"]

    def image(self, im):
        self._need_open()
        self.images.append(im.sha)
        self.image_rows = im.sha

    def prefill(self, tokens):
        self._need_open()
        rows = [t for t in tokens if t >= self.vocab_size]
        if rows and self.image_rows is None:
            raise FakeLibraryError("image rows without llm_image()")
        # the fake keeps real ids: image rows -> the image token
        return super().prefill([tok().special["<image>"] if t >= self.vocab_size else t
                                for t in tokens]) if not rows or all(
            0 <= t - self.vocab_size < self.image_tokens for t in rows) else self._bad()

    def _bad(self):
        raise FakeLibraryError("image row out of range")


def backend(engine, **kw):
    from smolvlm_backend import SmolvlmBackend
    kw.setdefault("defaults", SamplerParams(temperature=0.0))
    kw.setdefault("loop_guard", False)
    b = SmolvlmBackend(engine, TOKENIZER, sampler_lib=sampler_lib() or "/nonexistent", **kw)
    b.load_host()
    b.load()
    return b


def image_msg(url, text, first=True):
    ps = [{"type": "image", "url": url}, {"type": "text", "text": text}]
    return {"role": "user", "content": text, "parts": ps if first else ps[::-1]}


def run(b, messages, **kw):
    job = b.prepare(ChatRequest(model="smolvlm-256m-instruct", messages=messages, **kw))
    ev = list(b.generate(job, CancelToken()))
    assert isinstance(ev[-1], Finish) and all(isinstance(e, Delta) for e in ev[:-1])
    return "".join(e.text for e in ev[:-1]), ev[-1], job


@unittest.skipUnless(HAVE, "demo/chat/assets/smolvlm-256m-instruct/tokenizer.json not present")
class TestTemplate(unittest.TestCase):
    """idefics3.py + the tokenizer == transformers' Idefics3Processor."""

    def test_ids(self):
        pb = idefics3.PromptBuilder(tok().encode)
        for c in cases()["template"]:
            with self.subTest(case=c["name"]):
                msgs = [{"role": m["role"], "content": "", "parts": m["parts"]} for m in c["messages"]]
                rendered = idefics3.render(msgs)
                self.assertEqual(rendered.replace(idefics3.image_text(), idefics3.IMAGE), c["text"])
                self.assertEqual(pb.ids(msgs), c["ids"])

    def test_string_content_is_one_text_part(self):
        pb = idefics3.PromptBuilder(tok().encode)
        c = next(c for c in cases()["template"] if c["name"] == "text_only")
        self.assertEqual(pb.ids([{"role": "user", "content": "Hello! Who are you?"}]), c["ids"])


@unittest.skipUnless(HAVE_PIL, "Pillow not installed")
class TestImages(unittest.TestCase):
    """vlm_image.py == vlm_study.load_pixels (the processor's resize)."""

    def test_pixels_match_the_study(self):
        from vlm_image import ImageCache
        c = cases()
        cache = ImageCache()
        for im in c["images"]:
            with self.subTest(size=im["size"]):
                got = cache.get(im["url"])
                self.assertEqual(got.size, 512)
                self.assertEqual(len(got.pixels), 512 * 512 * 3)
                self.assertEqual(got.sha, im["sha256"],
                                 f"Pillow {PIL.__version__} vs the fixture's {c['pillow']}")
        self.assertEqual(cache.get(c["images"][0]["url"]).sha, c["images"][0]["sha256"])
        self.assertEqual(cache.hits, 1)

    def test_errors(self):
        from vlm_image import ImageCache, ImageError
        cache = ImageCache()
        for url in ("http://example.com/a.png", "data:image/png,abc", "data:image/png;base64,!!!",
                    "data:image/png;base64,aGVsbG8="):
            with self.subTest(url=url), self.assertRaises(ImageError):
                cache.get(url)


@unittest.skipUnless(HAVE and HAVE_PIL, "SmolVLM tokenizer or Pillow missing")
class TestBackend(unittest.TestCase):
    def setUp(self):
        c = cases()
        self.A, self.B = c["images"][0]["url"], c["images"][1]["url"]
        self.shaA, self.shaB = c["images"][0]["sha256"], c["images"][1]["sha256"]
        self.reply = tok().encode(" Two cats sleeping.")

    def test_image_rows_and_answer(self):
        eng = FakeVlmEngine(self.reply)
        b = backend(eng)
        text, fin, job = run(b, [image_msg(self.A, "Can you describe this image?")])
        self.assertEqual(text, "Two cats sleeping.")               # the leading space dropped
        self.assertEqual(eng.images, [self.shaA])
        self.assertEqual(fin.prompt_tokens, 81)
        ids = [t for p in eng.prefills for t in p]
        img = tok().special["<image>"]
        self.assertEqual(sum(1 for t in ids if t == img), 64)
        self.assertEqual(fin.info["finish"], "eos")

    def test_prefix_reuse_by_image_content(self):
        eng = FakeVlmEngine(self.reply)
        b = backend(eng)
        m1 = [image_msg(self.A, "What is in this image?")]
        text, fin, _ = run(b, m1)
        n1 = eng.prefilled_total
        m2 = m1 + [{"role": "assistant", "content": text},
                   {"role": "user", "content": "What color are they?"}]
        _, fin2, _ = run(b, m2)
        # the same image: no new llm_image(); the answer's tokens were cached too
        self.assertEqual(eng.images, [self.shaA])
        self.assertLess(fin2.info["prefill_tokens"], 20)
        self.assertEqual(fin2.info["cached_tokens"] + fin2.info["prefill_tokens"], fin2.prompt_tokens)
        # a different image at the same place: re-encoded, the prefix breaks at its block
        m3 = copy.deepcopy(m2)
        m3[0]["parts"][0]["url"] = self.B
        _, fin3, _ = run(b, m3)
        self.assertEqual(eng.images, [self.shaA, self.shaB])
        # the sink and "User:<fake_token_around_image><global-img>" stay cached
        self.assertEqual(fin3.info["cached_tokens"], 5)
        self.assertGreater(eng.prefilled_total, n1)

    def test_two_images_encoded_in_order(self):
        eng = FakeVlmEngine(self.reply)
        b = backend(eng)
        msgs = [{"role": "user", "content": "", "parts": [
            {"type": "image", "url": self.A}, {"type": "image", "url": self.B},
            {"type": "text", "text": "What differs between these two images?"}]}]
        _, fin, _ = run(b, msgs)
        self.assertEqual(eng.images, [self.shaA, self.shaB])
        self.assertEqual(fin.prompt_tokens, 149)
        self.assertEqual(len(eng.prefills), 3)           # image A block, image B block, the rest

    def test_trimming_drops_old_images(self):
        eng = FakeVlmEngine(self.reply, context_size=180)
        b = backend(eng, reserve=60)
        msgs = [image_msg(self.A, "First?"), {"role": "assistant", "content": "A."},
                image_msg(self.B, "Second?")]
        _, fin, job = run(b, msgs)
        self.assertEqual(job.dropped, 2)
        self.assertEqual([im.sha for im in job.extra["images"]], [self.shaB])
        self.assertEqual(eng.images, [self.shaB])

    def test_markers_in_text_are_removed(self):
        eng = FakeVlmEngine(self.reply)
        b = backend(eng)
        _, fin, job = run(b, [image_msg(self.A, "describe <image><image> this")])
        img = tok().special["<image>"]
        self.assertEqual(sum(1 for t in job.ids if t == img), 64)

    def test_bad_image(self):
        b = backend(FakeVlmEngine(self.reply))
        with self.assertRaises(BackendError) as cm:
            b.prepare(ChatRequest(model="smolvlm-256m-instruct",
                                  messages=[image_msg("data:image/png;base64,aGVsbG8=", "hi")]))
        self.assertEqual(cm.exception.code, "invalid_image")


@unittest.skipUnless(HAVE and HAVE_PIL, "SmolVLM tokenizer or Pillow missing")
class TestServer(unittest.TestCase):
    """image_url parts over HTTP: the VLM answers, a text model refuses."""

    def test_http(self):
        url = cases()["images"][0]["url"]
        vlm = backend(FakeVlmEngine(tok().encode(" A gradient.")))
        srv = RunningServer({"smolvlm-256m-instruct": vlm, "echo": server_mod.EchoBackend()})
        try:
            body = {"model": "smolvlm-256m-instruct", "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": url}},
                {"type": "text", "text": "What is this?"}]}]}
            st, _h, raw = srv.post(body)
            out = json.loads(raw)
            self.assertEqual(st, 200, out)
            self.assertEqual(out["choices"][0]["message"]["content"], "A gradient.")
            body["model"] = "echo"
            st, _h, raw = srv.post(body)
            out = json.loads(raw)
            self.assertEqual(st, 400)
            self.assertEqual(out["error"]["code"], "images_not_supported")
            self.assertEqual(out["error"]["param"], "messages.[0].content.[0]")
        finally:
            srv.close()


if __name__ == "__main__":
    unittest.main()
