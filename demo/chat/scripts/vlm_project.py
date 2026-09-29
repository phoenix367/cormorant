"""Shared pieces of SmolVLM-256M-Instruct on the KV260 (doc/plans/CHAT_PLAN.md
§23): the checkpoint and its two calibrated formats (vlm_study.py formats:
the text model's, the vision encoder's), the frontends — the text model
through src/llama.py with image rows, the vision encoder + connector through
src/vit.py — the entry models of the multi-entry project (decode,
prefill_<T>, head, vision; one weight pool, the image features a shared
state), and the library simulation with images (VlmSession).

The library's image convention: llm_image() writes an image's features into
the state; the prompt passes its k-th image token as id vocab + k (an image
row of the prefill's LlmEmbed).  Runs in the scheduler's venv.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

import llm_project as lp
from src.codegen import CodeGenerator
from src.llama import Formats, LazySafetensors, LlamaConfig, LlamaFrontend, load_safetensors
from src.llm_entries import entry_graphs
from src.vit import VisionFormats, VitConfig, VitFrontend, patches

MODEL = "smolvlm-256m-instruct"
IMAGE_STATE = "vlm.img"
TEXT_FORMATS = "pow2+sink+p12"            # vlm_study.TEXT_FORMATS
VISION_FORMATS = "pow2+p12"               # vlm_study.VISION_SHIPPED


def default_assets() -> str:
    return os.path.join(os.path.dirname(lp.default_assets()), MODEL)


def is_vlm(assets: str) -> bool:
    try:
        with open(os.path.join(assets, "config.json")) as f:
            return "vision_config" in json.load(f)
    except OSError:
        return False


def default_formats(assets: Optional[str] = None):
    """(text formats, vision formats) paths in the model's study dir."""
    sd = lp.study.study_dir(assets or default_assets())
    return (os.path.join(sd, f"formats_{TEXT_FORMATS}.json"),
            os.path.join(sd, f"vision_formats_{VISION_FORMATS}.json"))


@dataclass
class Vlm:
    config: dict
    tcfg: LlamaConfig
    vcfg: VitConfig
    Wt: Dict[str, np.ndarray]           # text weights, Llama names (model.layers.*, lm_head.weight)
    Wv: Dict[str, np.ndarray]           # vision + connector weights, HF names
    tfmt: Formats
    vfmt: VisionFormats
    image_token: int
    formats_paths: tuple


def load(assets: Optional[str] = None, text_formats: Optional[str] = None,
         vision_formats: Optional[str] = None, lazy: bool = False) -> Vlm:
    """``lazy``: the weights as LazySafetensors mappings (llm_project.load_model)."""
    assets = assets or default_assets()
    with open(os.path.join(assets, "config.json")) as f:
        cj = json.load(f)
    tcfg = LlamaConfig.from_dict({**cj["text_config"],
                                  "tie_word_embeddings": cj.get("tie_word_embeddings", False)})
    vcfg = VitConfig.from_dict(cj)
    path = os.path.join(assets, "model.safetensors")
    W = LazySafetensors(path) if lazy else load_safetensors(path)
    tnames = {k.replace("model.text_model.", "model."): k for k in W
              if k.startswith("model.text_model.")}
    if "lm_head.weight" in W:
        tnames["lm_head.weight"] = "lm_head.weight"
    vnames = {k: k for k in W
              if k.startswith("model.vision_model.") or k.startswith("model.connector.")}
    if lazy:
        Wt, Wv = W.select(tnames), W.select(vnames)
    else:
        Wt = {n: W[k] for n, k in tnames.items()}
        Wv = {n: W[k] for n, k in vnames.items()}
    del W
    tp, vp = default_formats(assets)
    tp, vp = text_formats or tp, vision_formats or vp
    return Vlm(cj, tcfg, vcfg, Wt, Wv, Formats.from_file(tp, tcfg),
               VisionFormats.from_file(vp, vcfg), int(cj["image_token_id"]), (tp, vp))


def frontends(m: Vlm, ctx: int = lp.CONTEXT, name: str = MODEL):
    """(text frontend, vision frontend)."""
    fe_t = LlamaFrontend(m.tcfg, m.Wt, m.tfmt, ctx=ctx, name=name, prefill_attn="fpga",
                         image_rows=m.vcfg.n_img, image_state=IMAGE_STATE)
    fe_v = VitFrontend(m.vcfg, m.Wv, m.vfmt, name=name, image_state=IMAGE_STATE)
    return fe_t, fe_v


def entry_models(fe_t: LlamaFrontend, fe_v: VitFrontend, buckets: Sequence[int] = lp.BUCKETS):
    out = lp.entry_models(fe_t, buckets)
    out.add("vision", fe_v.entry)
    return out


def make_codegens(fe_t, fe_v, buckets: Sequence[int] = lp.BUCKETS) -> Dict[str, CodeGenerator]:
    return {n: CodeGenerator(g, model_path=f"{n}.onnx")
            for n, g in entry_graphs(entry_models(fe_t, fe_v, buckets))}


def library_ids(toks: Sequence[int], vocab: int, image_token: int) -> List[int]:
    """The library's ids of a prompt's token ids (incl. the leading
    <|im_start|>): the ids after it, the k-th image token as vocab + k."""
    out, k = [], 0
    for t in list(toks)[1:]:
        if int(t) == image_token:
            out.append(vocab + k)
            k += 1
        else:
            out.append(int(t))
    return out


class VlmSession(lp.SimSession):
    """SimSession with llm_image(): the vision entry writes the image state."""

    def __init__(self, cgs, patch: int, **kw):
        super().__init__(cgs, **kw)
        self.patch = patch

    def image(self, rgb: np.ndarray) -> np.ndarray:
        self._run("vision", {"vision.patches": patches(rgb, self.patch).astype(np.float64)})
        return self.states[IMAGE_STATE]


__all__ = ("MODEL", "IMAGE_STATE", "Vlm", "load", "frontends", "entry_models", "make_codegens",
           "library_ids", "VlmSession", "is_vlm", "default_assets", "default_formats")
