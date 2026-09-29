"""Shared pieces of the SmolLM2 (Llama-family) decoder on the KV260 —
doc/plans/CHAT_PLAN.md phase 3: model / formats loading, the entry graphs of the
multi-entry project, the prefill bucket split the library uses, and a
simulation of the library (``SimSession``: llm_prefill / llm_decode /
llm_truncate over the scheduler's own fixed-point simulation of every entry).

Used by llm_sched_check.py (bit-exactness against the study emulation),
generate_llm_project.py (the project / libsmollm2.so) and llm_board.py (the
board run's logits against the simulation).  Runs in the scheduler's venv
(inference-scheduler/.venv: onnx + numpy).
"""

from __future__ import annotations

import functools
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
from src.llm_entries import EntryModels, entry_graphs            # noqa: E402
from src.llama import (Formats, LazySafetensors, LlamaConfig, LlamaFrontend,  # noqa: E402
                       load_safetensors)

# The shipped numeric policy (llm_study.py) per prefill-attention mode of the
# frontend (src/llama.py): "fpga" (phase 5, CHAT_PLAN §16) — q.K^T / P.V of the
# prefill on ConvKernel with the p12 host softmax, decode attention the xattn
# host region; "host" (phase 3) — xattn everywhere.
POLICIES = {"fpga": "pow2+sink+p12+mix", "host": "pow2+sink+p12+xattn"}
PREFILL_ATTN = "fpga"
POLICY = POLICIES[PREFILL_ATTN]
FORMATS_POLICY = "pow2+sink+p12"        # their exponents (both use the p12 formats)
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
    return os.path.join(study.study_dir(assets), f"formats_{FORMATS_POLICY}.json")


def load_model(assets: Optional[str] = None, formats: Optional[str] = None,
               lazy: bool = False):
    """(LlamaConfig, weights {HF name: float32}, Formats, formats dict);
    ``lazy``: the weights as a LazySafetensors mapping (converted when read,
    no float32 copy of the checkpoint held — generate_llm_project.py)."""
    assets = assets or default_assets()
    cfg = LlamaConfig.from_file(os.path.join(assets, "config.json"))
    W = (LazySafetensors if lazy else load_safetensors)(os.path.join(assets, "model.safetensors"))
    fpath = formats or default_formats(assets)
    with open(fpath) as f:
        fd = json.load(f)
    return cfg, W, Formats(fd, cfg), fd


def frontend(cfg, W, fmt, ctx: int = CONTEXT, name: str = "smollm2",
             prefill_attn: str = PREFILL_ATTN) -> LlamaFrontend:
    return LlamaFrontend(cfg, W, fmt, ctx=ctx, name=name, prefill_attn=prefill_attn)


def entry_models(fe: LlamaFrontend, buckets: Sequence[int] = BUCKETS) -> EntryModels:
    """{entry name: onnx.ModelProto}: decode, prefill_<P> per bucket, head —
    each built when first read (entry_graphs takes them one at a time)."""
    out = EntryModels()
    out.add("decode", functools.partial(fe.entry, "decode"))
    for p in sorted(buckets):
        out.add(f"prefill_{p}", functools.partial(fe.entry, "prefill", p))
    out.add("head", functools.partial(fe.entry, "head"))
    return out


# Board cost of one prefill call per bucket, ms, without the head (run once
# per llm_prefill): KV260, 100 MHz, hw_128 bitstream, FPGA prefill attention
# (CHAT_PLAN §16; a call at position 1, so the attention covers bucket + 1
# keys), profiled 2026-09-26.  The ConvKernel MatMuls stream the weights once
# per call, so a padded 64-row call costs little more than a 16-row one.
# (Phase 3's host-attention table, CHAT_PLAN §13.4: {16: 304, 64: 364, 256: 930}
# without the attention, which then covered only the valid rows.)
BUCKET_COST_MS = {16: 326, 64: 427, 256: 1275}


def bucket_costs(buckets: Sequence[int] = BUCKETS) -> List[int]:
    """Cost per call of each bucket (sorted order), ms: the measured table,
    linear inter- / extrapolation over it for other bucket sizes."""
    pts = sorted(BUCKET_COST_MS.items())
    out = []
    for b in sorted(buckets):
        if b in BUCKET_COST_MS:
            out.append(BUCKET_COST_MS[b])
            continue
        lo = max([p for p in pts if p[0] < b], default=pts[0])
        hi = min([p for p in pts if p[0] > b], default=pts[-1])
        if lo == hi:
            lo, hi = (pts[-2], pts[-1]) if b > pts[-1][0] else (pts[0], pts[1])
        out.append(max(1, round(lo[1] + (hi[1] - lo[1]) * (b - lo[0]) / (hi[0] - lo[0]))))
    return out


def split_plan(ctx: int = CONTEXT, buckets: Sequence[int] = BUCKETS,
               costs: Optional[Sequence[int]] = None) -> List[int]:
    """pick[r] = the bucket of the first prefill call for r remaining tokens
    (r = 1 .. ctx) on the least-cost split: cost[r] = min over buckets b of
    cost_b + cost[max(0, r - b)], larger buckets winning ties (llm_api.c
    plan_split, the same integer recurrence)."""
    b = sorted(buckets)
    c = list(costs) if costs is not None else bucket_costs(b)
    cost = [0] * (ctx + 1)
    pick = [b[-1]] * (ctx + 1)
    for r in range(1, ctx + 1):
        best = None
        for j in range(len(b) - 1, -1, -1):
            v = c[j] + cost[max(0, r - b[j])]
            if best is None or v < best:
                best, pick[r] = v, b[j]
        cost[r] = best
    return pick


def split_prefill(n: int, buckets: Sequence[int] = BUCKETS,
                  costs: Optional[Sequence[int]] = None) -> List[Tuple[int, int]]:
    """The library's split of n new tokens into prefill calls: [(rows, bucket)]
    on the least-cost plan (split_plan; the last call's rows padded to its
    bucket).  The logits do not depend on the split (every row is computed
    independently of the others in its call), only the time does."""
    pick = split_plan(max(n, 1), buckets, costs)
    out = []
    while n > 0:
        B = pick[n]
        k = min(n, B)
        out.append((k, B))
        n -= k
    return out


def make_codegen(model, name: str, matmul_on_conv="auto") -> CodeGenerator:
    g = OnnxGraph(model, fuse_act=True, s2d_stem=True, matmul_on_conv=matmul_on_conv)
    return CodeGenerator(g, model_path=f"{name}.onnx")


def make_codegens(fe: LlamaFrontend, buckets: Sequence[int] = BUCKETS) -> Dict[str, CodeGenerator]:
    """{entry: CodeGenerator} of every entry, scheduled like the generated
    library (src/llm_entries.entry_graphs): one model at a time and the
    constant arrays shared across entries — per-entry make_codegen holds
    every weight once per entry.  The simulation does not depend on the
    engine choices."""
    return {n: CodeGenerator(g, model_path=f"{n}.onnx")
            for n, g in entry_graphs(entry_models(fe, buckets))}


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
        """One entry call; returns its outputs (and the states)."""
        cg = self.cgs[entry]
        return cg._forward_pass({k: np.asarray(v, np.float64) for k, v in feeds.items()},
                                states=self.states,
                                keep=[t.onnx_name for t in cg._graph.output_tensors])

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


# A chat's second turn (phase 5 gates): appended to a prompt's decoded answer —
# the answer's <|im_end|>, a new user message and the assistant header — and
# prefilled at the position the decode steps reached.
SECOND_TURN = [{"role": "user", "content": "Tell me one more fact about it."}]


def second_turn_ids(cache: Optional[str] = None) -> List[int]:
    """Token ids of "<|im_end|>\\n" + the SECOND_TURN user block + the
    generation prompt (tokenized once with .venv-export's tokenizers, cached)."""
    cache = cache or os.path.join(os.path.dirname(default_assets()), "study",
                                  "phase5_second_turn_ids.json")
    if not os.path.exists(cache):
        code = ("import json, sys; sys.path.insert(0, %r); import llm_study as s; "
                "t = s.Tok(s.default_assets()); "
                "txt = '<|im_end|>\\n' + ''.join('<|im_start|>' + m['role'] + '\\n' + m['content'] "
                "+ '<|im_end|>\\n' for m in %r) + '<|im_start|>assistant\\n'; "
                "json.dump([int(i) for i in t.encode(txt)], open(%r, 'w'))"
                % (HERE, SECOND_TURN, cache))
        subprocess.run([EXPORT_PY, "-c", code], check=True)
    with open(cache) as f:
        return json.load(f)


def with_second_turns(ids: Dict[str, List[int]], names: Sequence[str]) -> Dict[str, List[int]]:
    """ids plus a "<name>/turn2" continuation right after each prompt in names."""
    out: Dict[str, List[int]] = {}
    t2 = second_turn_ids() if names else []
    for n, t in ids.items():
        out[n] = t
        if n in names:
            out[n + "/turn2"] = list(t2)
    return out


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
