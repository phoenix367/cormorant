"""smolvlm_backend.py — backend C of the KV260 chat server: chat with images,
SmolVLM-256M-Instruct on the FPGA (libsmolvlm_256m.so: the §11 C API plus
llm_image(); doc/plans/CHAT_PLAN.md §23).  Standard library + Pillow (image
decoding, vlm_image.py); the text side is Smollm2Backend's (tokenizer,
sampler, stop strings, penalties, loop guard, streaming).

prepare()  messages (text and image parts; images as base64 data URLs) ->
           the Idefics3 prompt ids (idefics3.py; every image one 512 x 512
           tile = 67 tokens), history trimmed from the front, images decoded
           and resized (cached by URL).
generate() prefix-cache reuse keyed by content: the KV cache's positions are
           keyed by token id, an image's rows by (image SHA-256, row), so a
           follow-up question about the same image prefills only the new turn
           and a different image breaks the prefix where it starts.  The new
           positions are prefilled in segments: before a segment holding (part
           of) an image block the image goes through llm_image(), and its k-th
           image token is passed as vocab + k.  Then Smollm2Backend's decode
           loop; ends at <end_of_utterance> (also <|im_end|>, <|endoftext|>).
           The answer's first piece loses one leading space ("Assistant:" is
           followed by " The ..."): a client resending the answer then renders
           "Assistant: The ..." — the same tokens, so the next turn reuses them.
"""

from __future__ import annotations

import ctypes
from typing import Any, Dict, List, Optional

import idefics3
from chat_backend import BackendError, CancelToken, ChatRequest, Delta
from smollm2_backend import Job, LibLlmEngine, Smollm2Backend
from vlm_image import Image, ImageCache, ImageError

MODEL_ID = "smolvlm-256m-instruct"
EOG_TOKENS = ("<end_of_utterance>", "<|im_end|>", "<|endoftext|>")


class LibVlmEngine(LibLlmEngine):
    """LibLlmEngine + llm_image(): int llm_image(const uint8_t *rgb),
    int llm_image_tokens(void), int llm_image_size(void)."""

    image_tokens = 0
    image_size = 0

    def _bind(self):
        lib = super()._bind()
        for name, args, res in (("llm_image", [ctypes.c_char_p], ctypes.c_int),
                                ("llm_image_tokens", [], ctypes.c_int),
                                ("llm_image_size", [], ctypes.c_int)):
            if hasattr(lib, name):
                f = getattr(lib, name)
                f.argtypes, f.restype = args, res
        return lib

    def open(self) -> None:
        super().open()
        lib = self.lib
        self.image_tokens = int(lib.llm_image_tokens()) if hasattr(lib, "llm_image_tokens") else 0
        self.image_size = int(lib.llm_image_size()) if hasattr(lib, "llm_image_size") else 0
        if self.image_tokens <= 0:
            self.close()
            raise RuntimeError(f"{self.lib_path} has no vision encoder (llm_image_tokens() = 0)")

    def info(self) -> Dict[str, Any]:
        return dict(super().info(), image_tokens=self.image_tokens, image_size=self.image_size)

    def image(self, im: Image) -> None:
        if im.size != self.image_size:
            raise ValueError(f"image of {im.size} pixels, the library takes {self.image_size}")
        rc = self.lib.llm_image(im.pixels)
        if rc < 0:
            raise self._err("llm_image", rc)


class SmolvlmBackend(Smollm2Backend):
    model_id = MODEL_ID
    fingerprint = "kv260-smolvlm-256m-pow2+sink+p12"
    accepts_images = True
    cma_mb = 520.0                      # fallback; the 495 MiB pool BO + the image buffer

    def __init__(self, engine, tokenizer_path: str, *, image_cache: Optional[ImageCache] = None,
                 max_images: int = 8, **kw):
        kw.setdefault("model_id", MODEL_ID)
        super().__init__(engine, tokenizer_path, **kw)
        if self.model_id == MODEL_ID:
            self.fingerprint = "kv260-smolvlm-256m-pow2+sink+p12"
        self.images = image_cache or ImageCache()
        self.max_images = max_images
        self.stats["images"] = 0
        self.stats["image_encodes"] = 0

    # ── lifecycle ──

    def load_host(self) -> None:
        if self.tok is not None:
            return
        super().load_host()
        self.builder = idefics3.PromptBuilder(self.tok.encode)
        self.eog = frozenset(self.tok.special[t] for t in EOG_TOKENS if t in self.tok.special)
        self.image_tok = self.tok.special[idefics3.IMAGE]

    def load(self) -> None:
        super().load()
        if getattr(self.engine, "image_tokens", idefics3.IMAGE_SEQ) != idefics3.IMAGE_SEQ:
            n = self.engine.image_tokens
            self.engine.close()
            raise RuntimeError(f"the library takes {n} image tokens per image, the template "
                               f"{idefics3.IMAGE_SEQ}")

    def health(self) -> Dict[str, Any]:
        return dict(super().health(), image_cache={"hits": self.images.hits,
                                                    "misses": self.images.misses})

    # ── request -> prompt ids + images (no FPGA) ──

    def _messages(self, req: ChatRequest) -> List[Dict[str, Any]]:
        out, n_img = [], 0
        for i, m in enumerate(req.messages):
            role = "system" if m["role"] == "developer" else m["role"]
            ps = []
            for j, p in enumerate(m.get("parts") or [{"type": "text", "text": m["content"]}]):
                if p["type"] != "image":
                    ps.append({"type": "text", "text": _no_image_markers(p.get("text", ""))})
                    continue
                n_img += 1
                if n_img > self.max_images:
                    raise BackendError(f"At most {self.max_images} images per request.",
                                       f"messages.[{i}].content", code="invalid_value")
                try:
                    im = self.images.get(p["url"])
                except ImageError as e:
                    raise BackendError(f"Invalid image in messages[{i}].content[{j}]: {e}",
                                       f"messages.[{i}].content.[{j}]",
                                       code="invalid_image") from None
                ps.append({"type": "image", "image": im})
            out.append({"role": role, "content": m.get("content", ""), "parts": ps})
        return out

    def prepare(self, req: ChatRequest) -> Job:
        if self.tok is None:
            self.load_host()
        msgs = self._messages(req)
        job = super().prepare(ChatRequest(req.model, msgs, req.stream, req.max_tokens,
                                          req.temperature, req.top_p, req.stop, req.seed, req.raw))
        # the kept messages' images, in prompt order (super().prepare dropped job.dropped
        # messages after a leading system message)
        head = msgs[:1] if msgs and msgs[0]["role"] == "system" else []
        job.extra["images"] = idefics3.images_of(head + msgs[len(head) + job.dropped:])
        n_blocks = sum(1 for t in job.ids if t == self.image_tok) // idefics3.IMAGE_SEQ
        if n_blocks != len(job.extra["images"]):
            raise BackendError("internal: image blocks and images disagree", "messages")
        return job

    # ── prompt -> KV cache (FPGA) ──

    def _prompt_keys(self, job: Job) -> list:
        """Token ids; an image's rows keyed (sha, row)."""
        keys, imgs, k, j = [], job.extra.get("images", []), 0, -1
        for t in job.ids[1:]:
            if t == self.image_tok:
                if k == 0:
                    j += 1
                keys.append(("img", imgs[j].sha, k))
                k = (k + 1) % idefics3.IMAGE_SEQ
            else:
                keys.append(t)
        return keys

    def _prefill_new(self, job: Job, keys: list, reuse: int, cancel: CancelToken):
        """Segments of the new positions: each ends with an image block (the
        image encoded first, its rows passed as vocab + row), the last one with
        the prompt's end."""
        eng = self.engine
        imgs = {im.sha: im for im in job.extra.get("images", [])}
        V = eng.vocab_size
        new = keys[reuse:]
        logits, pos, cur = None, 0, None
        while pos < len(new):
            end = pos
            while end < len(new) and not isinstance(new[end], tuple):
                end += 1
            if end < len(new):                                  # an image block starts at `end`
                sha = new[end][1]
                while end < len(new) and isinstance(new[end], tuple) and new[end][1] == sha:
                    end += 1
                if sha != cur:
                    if pos:
                        cancel.check()
                    eng.image(imgs[sha])
                    self.stats["image_encodes"] += 1
                    cur = sha
            seg = new[pos:end]
            ids = [V + key[2] if isinstance(key, tuple) else key for key in seg]
            if pos:
                cancel.check()
            logits = eng.prefill(ids)
            self._cached.extend(seg)
            pos = end
        return logits

    def generate(self, job: Job, cancel: CancelToken):
        self.stats["images"] += len(job.extra.get("images", []))
        first = True
        for ev in super().generate(job, cancel):
            if first and isinstance(ev, Delta):
                first = False
                text = ev.text[1:] if ev.text.startswith(" ") else ev.text
                if text:
                    yield Delta(text)
                continue
            yield ev


def _no_image_markers(text: str) -> str:
    """A text part without the template's image markers (they would be
    tokenized as image tokens)."""
    for mark in (idefics3.IMAGE, idefics3.FAKE_IMAGE, idefics3.GLOBAL_IMAGE):
        text = text.replace(mark, "")
    return text


__all__ = ("MODEL_ID", "LibVlmEngine", "SmolvlmBackend", "EOG_TOKENS")
