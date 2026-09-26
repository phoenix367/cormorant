"""Shared pieces of the SmolLM2 (Llama-family) decoder on the KV260 —
doc/CHAT_PLAN.md phase 3: model / formats loading, the entry graphs of the
multi-entry project, the prefill bucket split the library uses, and a
simulation of the library (``SimSession``: llm_prefill / llm_decode /
llm_truncate over the scheduler's own fixed-point simulation of every entry).

Used by llm_sched_check.py (bit-exactness against the study emulation),
generate_llm_project.py (the project / libsmollm2.so) and llm_board.py (the
board run's logits against the simulation).  Runs in the scheduler's venv
(inference-scheduler/.venv: onnx + numpy).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
SCHED = os.path.join(REPO, "inference-scheduler")
for _p in (HERE, SCHED):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import llm_study as study                                         # noqa: E402
from src.codegen import CodeGenerator                             # noqa: E402
from src.graph import OnnxGraph                                   # noqa: E402
from src.llama import Formats, LlamaConfig, LlamaFrontend, load_safetensors  # noqa: E402

POLICY = "pow2+sink+p12+xattn"          # the shipped numeric policy (llm_study.py)
FORMATS_POLICY = "pow2+sink+p12"        # its exponents (xattn uses the p12 formats)
CONTEXT = 1024                          # positions incl. the sink (CHAT_PLAN decision)
BUCKETS = (16, 64, 256)                 # prefill entries (rows per call)


def _main_checkout() -> str:
    """The main checkout (a git worktree shares its untracked .venv-export)."""
    try:
        common = subprocess.check_output(["git", "-C", REPO, "rev-parse", "--git-common-dir"],
                                         text=True, stderr=subprocess.DEVNULL).strip()
        return os.path.dirname(os.path.abspath(os.path.join(REPO, common)))
    except Exception:
        return REPO


EXPORT_PY = os.environ.get("LLM_EXPORT_PY",
                           os.path.join(_main_checkout(), ".venv-export", "bin", "python"))


def default_assets() -> str:
    return study.default_assets()


def default_formats(assets: Optional[str] = None) -> str:
    assets = assets or default_assets()
    return os.path.join(os.path.dirname(assets), "study", f"formats_{FORMATS_POLICY}.json")


def load_model(assets: Optional[str] = None, formats: Optional[str] = None):
    """(LlamaConfig, weights {HF name: float32}, Formats, formats dict)."""
    assets = assets or default_assets()
    cfg = LlamaConfig.from_file(os.path.join(assets, "config.json"))
    W = load_safetensors(os.path.join(assets, "model.safetensors"))
    fpath = formats or default_formats(assets)
    with open(fpath) as f:
        fd = json.load(f)
    return cfg, W, Formats(fd, cfg), fd


def frontend(cfg, W, fmt, ctx: int = CONTEXT, name: str = "smollm2") -> LlamaFrontend:
    return LlamaFrontend(cfg, W, fmt, ctx=ctx, name=name)


def entry_models(fe: LlamaFrontend, buckets: Sequence[int] = BUCKETS) -> Dict[str, object]:
    """{entry name: onnx.ModelProto}: decode, prefill_<P> per bucket, head."""
    out = {"decode": fe.entry("decode")}
    for p in sorted(buckets):
        out[f"prefill_{p}"] = fe.entry("prefill", p)
    out["head"] = fe.entry("head")
    return out


def split_prefill(n: int, buckets: Sequence[int] = BUCKETS) -> List[Tuple[int, int]]:
    """The library's split of n new tokens into prefill calls: [(rows, bucket)],
    the largest bucket that the remaining tokens fill completely, and the last
    remainder padded into the smallest bucket (llm_api.c llm_prefill)."""
    b = sorted(buckets)
    out = []
    while n > 0:
        fit = [x for x in b if x <= n]
        B = fit[-1] if fit else b[0]
        k = min(n, B)
        out.append((k, B))
        n -= k
    return out


def make_codegen(model, name: str, matmul_on_conv="auto") -> CodeGenerator:
    g = OnnxGraph(model, fuse_act=True, s2d_stem=True, matmul_on_conv=matmul_on_conv)
    return CodeGenerator(g, model_path=f"{name}.onnx")


class SimSession:
    """The C library's semantics over the scheduler simulation of each entry
    (llm_api.c): position 0 is the sink; prefill appends in bucket calls and
    then runs the head entry; decode appends one token."""

    def __init__(self, cgs: Dict[str, CodeGenerator], ctx: int = CONTEXT,
                 buckets: Sequence[int] = BUCKETS):
        self.cgs = cgs
        self.ctx = ctx
        self.buckets = tuple(sorted(buckets))
        self.states: Dict[str, np.ndarray] = {}
        for cg in cgs.values():
            for k, v in cg.initial_states().items():
                self.states.setdefault(k, v)
        self.pos = 1
        self.trace: List[dict] = []            # (entry, arrays) of the last call, optional

    def truncate(self, n: int) -> None:
        if not 1 <= n <= self.pos:
            raise ValueError(f"truncate({n}) with {self.pos} positions")
        self.pos = n

    def _run(self, entry: str, feeds: dict) -> dict:
        cg = self.cgs[entry]
        return cg._forward_pass({k: np.asarray(v, np.float64) for k, v in feeds.items()},
                                states=self.states)

    def prefill(self, tokens: Sequence[int]) -> np.ndarray:
        tokens = [int(t) for t in tokens]
        if not tokens or self.pos + len(tokens) > self.ctx:
            raise ValueError("prefill: empty or beyond the context")
        i = 0
        for k, B in split_prefill(len(tokens), self.buckets):
            ids = np.zeros(B)
            ids[:k] = tokens[i:i + k]
            e = f"prefill_{B}"
            self._run(e, {f"{e}.ids": ids, f"{e}.pos": [self.pos], f"{e}.n": [k]})
            self.pos += k
            i += k
        return self._run("head", {})["head.logits"].reshape(-1)

    def decode(self, token: int) -> np.ndarray:
        if self.pos + 1 > self.ctx:
            raise ValueError("decode: context full")
        out = self._run("decode", {"decode.ids": [int(token)], "decode.pos": [self.pos]})
        self.pos += 1
        return out["decode.logits"].reshape(-1)


def tokenize_prompts(names: Optional[Sequence[str]] = None, cache: Optional[str] = None
                     ) -> Dict[str, List[int]]:
    """Token ids of llm_study.PROMPTS (chat template, incl. the leading
    <|im_start|>) — tokenized once with the tokenizers package of
    .venv-export (subprocess) and cached as JSON."""
    cache = cache or os.path.join(os.path.dirname(default_assets()), "study",
                                  "phase3_prompt_ids.json")
    if not os.path.exists(cache):
        code = ("import json, sys; sys.path.insert(0, %r); import llm_study as s; "
                "t = s.Tok(s.default_assets()); "
                "json.dump({n: [int(i) for i in t.encode(s.chatml(m))] for n, m in s.PROMPTS}, "
                "open(%r, 'w'))" % (HERE, cache))
        subprocess.run([EXPORT_PY, "-c", code], check=True)
    with open(cache) as f:
        ids = json.load(f)
    if names:
        ids = {n: ids[n] for n in names}
    return ids
