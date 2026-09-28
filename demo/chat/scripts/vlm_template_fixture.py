#!/usr/bin/env python3
"""
vlm_template_fixture.py — writes demo/chat/tests/data/smolvlm_cases.json, the
references the stdlib tests of the SmolVLM backend compare against
(doc/plans/CHAT_PLAN.md §23):

  template  prompt ids of several conversations (images first / after text,
            two images in a message, multi-turn with an answer, a system
            message, text only) by transformers' Idefics3Processor (PIL
            backend, image splitting off) — idefics3.py + smollm2_tokenizer.py
            must produce the same ids;
  images    synthetic PNGs (as data URLs) and the SHA-256 of their 512 x 512
            pixels by vlm_study.load_pixels (= the processor's resize) —
            vlm_image.py must reproduce them (Pillow's LANCZOS across versions).

usage: .venv-export/bin/python demo/chat/scripts/vlm_template_fixture.py
"""
import base64
import hashlib
import io
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vlm_study as vs                                          # noqa: E402

OUT = os.path.join(os.path.dirname(HERE), "tests", "data", "smolvlm_cases.json")


def png(w, h, seed):
    from PIL import Image
    r = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w]
    a = np.stack([(x * 255 // max(w - 1, 1)), (y * 255 // max(h - 1, 1)),
                  ((x + y) * 7 + r.integers(0, 40, (h, w))) % 256], -1).astype(np.uint8)
    b = io.BytesIO()
    Image.fromarray(a, "RGB").save(b, "PNG")
    return b.getvalue()


def main():
    from PIL import Image
    assets = vs.default_assets()
    proc = vs.hf_processor(assets)
    tiny = Image.new("RGB", (8, 8))
    U = lambda t: {"role": "user", "content": t}
    I = {"type": "image"}
    T = lambda s: {"type": "text", "text": s}
    convs = {
        "image_first": [U([I, T("Can you describe this image?")])],
        "text_then_image": [U([T("What is shown here?"), I])],
        "two_images": [U([I, I, T("What differs between these two images?")])],
        "multi_turn": [U([I, T("What is in this image?")]),
                       {"role": "assistant", "content": [T("Two cats on a pink blanket.")]},
                       U([T("What color are they?")])],
        "second_image": [U([I, T("Describe it.")]),
                         {"role": "assistant", "content": [T("A red bus in a street.")]},
                         U([I, T("And this one?")])],
        "system": [{"role": "system", "content": [T("Answer in one sentence.")]},
                   U([I, T("What is happening?")])],
        "text_only": [U([T("Hello! Who are you?")])],
    }
    cases = []
    for name, msgs in convs.items():
        n_img = sum(1 for m in msgs for p in m["content"] if p.get("type") == "image")
        prompt = proc.apply_chat_template(msgs, add_generation_prompt=True)
        kw = dict(images=[tiny] * n_img, do_image_splitting=False) if n_img else {}
        ids = proc(text=prompt, return_tensors="np", **kw)["input_ids"][0].tolist()
        cases.append({"name": name, "messages": [
            {"role": m["role"], "parts": [{"type": "image"} if p["type"] == "image" else
                                          {"type": "text", "text": p["text"]} for p in m["content"]]}
            for m in msgs], "text": prompt, "ids": ids})
    images = []
    for w, h, seed in ((97, 61, 1), (160, 120, 2), (60, 180, 3)):
        data = png(w, h, seed)
        p = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"vlm_fixture_{seed}.png")
        with open(p, "wb") as f:
            f.write(data)
        px = vs.load_pixels(p)
        os.remove(p)
        images.append({"size": [w, h], "url": "data:image/png;base64," + base64.b64encode(data).decode(),
                       "sha256": hashlib.sha256(px.tobytes()).hexdigest()})
    import PIL
    import transformers
    json.dump({"note": "written by demo/chat/scripts/vlm_template_fixture.py",
               "pillow": PIL.__version__, "transformers": transformers.__version__,
               "template": cases, "images": images}, open(OUT, "w"), indent=1)
    print(f"{OUT}: {len(cases)} conversations, {len(images)} images")


if __name__ == "__main__":
    main()
