# Copyright 2026 Ivan Gubochkin
# SPDX-License-Identifier: Apache-2.0

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

import llm_model                                                  # noqa: E402
import llm_study as study                                         # noqa: E402
from src.codegen import CodeGenerator                             # noqa: E402
from src.graph import OnnxGraph                                   # noqa: E402
from src.llm_entries import EntryModels, entry_graphs            # noqa: E402
from src.checkpoint import LazySafetensors, load_safetensors      # noqa: E402
from src.llama import Formats, LlamaConfig, LlamaFrontend        # noqa: E402

# The numeric policy (llm_study.py) per attention mode of the frontend
# (src/llama.py), (prefill_attn, decode_attn): ("fpga", "host") — phase 5,
# CHAT_PLAN §16: q.K^T / P.V of the prefill on ConvKernel with the p12 host
# softmax, decode attention the xattn host region; ("fpga", "fpga") — decode on
# ConvKernel too (doc/plans/KV_DECODE_PLAN.md); ("host", "host") — phase 3,
# xattn everywhere.
ATTN_POLICIES = {("fpga", "host"): "pow2+sink+p12+mix", ("fpga", "fpga"): "pow2+sink+p12",
                 ("host", "host"): "pow2+sink+p12+xattn"}
PREFILL_ATTN = "fpga"
DECODE_ATTN = "fpga"                   # the chat libraries since doc/plans/KV_DECODE_PLAN.md


def vsmx_default() -> bool:
    """The prefill softmax on VectorOPKernel's softmax unit (doc/plans/SOFTMAX_PLAN.md,
    policy <attention policy>+vsmx) where the platform has the unit."""
    from src.smx_nodes import enabled
    return enabled()


def rowops_default() -> bool:
    """The residual stream as int32 and the residual adds / RMSNorms / SiLU * up on
    the row-ops unit (doc/plans/HOSTOPS_PLAN.md, policy ...+rowops) where the
    platform has the unit."""
    from src.rowops_nodes import enabled
    return enabled()


def drope_default(rowops: Optional[bool] = None) -> bool:
    """The decode steps' attention prep on the row-ops unit too (RPU_PLAN §13.4,
    policy ...+rowops+drope: every step's RoPE the integer one) where the
    project uses the unit and the unit runs descriptor chains (one chain per
    layer; issued job by job it would cost the CPU what the host prep does)."""
    from src.rowops_nodes import chains_enabled
    rowops = rowops_default() if rowops is None else rowops
    return bool(rowops) and chains_enabled()


def policy(prefill_attn: str = PREFILL_ATTN, decode_attn: str = DECODE_ATTN,
           vsmx: Optional[bool] = None, rowops: Optional[bool] = None,
           drope: Optional[bool] = None) -> str:
    """The study policy a project of these attention modes computes (``vsmx`` /
    ``rowops`` / ``drope`` None: vsmx_default() / rowops_default() /
    drope_default())."""
    if (prefill_attn, decode_attn) not in ATTN_POLICIES:
        raise ValueError(f"no policy for prefill attention {prefill_attn!r} with decode "
                         f"attention {decode_attn!r} (decode on the FPGA needs FPGA prefill)")
    base = ATTN_POLICIES[(prefill_attn, decode_attn)]
    vsmx = vsmx_default() if vsmx is None else vsmx
    rowops = rowops_default() if rowops is None else rowops
    drope = drope_default(rowops) if drope is None else drope
    if (vsmx or rowops) and (prefill_attn, decode_attn) != ("fpga", "fpga"):
        raise ValueError("vsmx / rowops (the softmax and the row ops on the FPGA units) are studied "
                         "with the FPGA attention in prefill and decode only")
    if drope and not rowops:
        raise ValueError("drope (the decode prep on the row-ops unit) needs rowops")
    return base + ("+vsmx" if vsmx else "") + ("+rowops" if rowops else "") + ("+drope" if drope else "")


POLICY = policy()
FORMATS_POLICY = "pow2+sink+p12"        # their exponents (both use the p12 formats)
CONTEXT = 1024                          # positions incl. the sink (CHAT_PLAN decision)
BUCKETS = (16, 64, 256)                 # prefill entries (rows per call)


def context(assets: Optional[str] = None) -> int:
    """The library's context: CONTEXT, or a shorter max_position_embeddings
    (stories15m: 256; llm_model.Profile)."""
    return llm_model.Profile(assets or default_assets()).ctx


def gate_prompts(assets: Optional[str] = None) -> Tuple[str, str]:
    """(prompts, second-turn prompts) of the gates (llm_sched_check.py,
    llm_host_emu.py, llm_board.py): three chat prompts with a second turn on
    the first — a text profile's three story beginnings likewise."""
    if llm_model.Profile(assets or default_assets()).chat:
        return "factual,summarise,multi-turn", "factual"
    return "dragon,long,dialogue", "dragon"


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
    W = (LazySafetensors if lazy else load_safetensors)(assets)        # one file or shards
    fpath = formats or default_formats(assets)
    with open(fpath) as f:
        fd = json.load(f)
    return cfg, W, Formats(fd, cfg), fd


def frontend(cfg, W, fmt, ctx: int = CONTEXT, name: str = "smollm2",
             prefill_attn: str = PREFILL_ATTN, decode_attn: str = DECODE_ATTN,
             vsmx: Optional[bool] = None, rowops: Optional[bool] = None,
             drope: Optional[bool] = None) -> LlamaFrontend:
    vsmx = vsmx_default() if vsmx is None else vsmx
    rowops = rowops_default() if rowops is None else rowops
    drope = drope_default(rowops) if drope is None else drope
    policy(prefill_attn, decode_attn, vsmx, rowops, drope)      # a studied combination
    return LlamaFrontend(cfg, W, fmt, ctx=ctx, name=name, prefill_attn=prefill_attn,
                         decode_attn=decode_attn, vsmx=vsmx, rowops=rowops, drope=drope)


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


def _cached_ids(assets: Optional[str], cache: Optional[str], name: str, expr: str):
    """Token ids the study computes with the model's own tokenizer (``expr``
    over ``t`` = llm_study.Tok and ``p`` = llm_study.Profile of the assets), in
    .venv-export's tokenizers (subprocess) once, cached as JSON in the model's
    study directory."""
    assets = os.path.abspath(assets or default_assets())
    cache = cache or os.path.join(study.study_dir(assets), name)
    if not os.path.exists(cache):
        code = ("import json, sys; sys.path.insert(0, %r); import llm_study as s; "
                "a = %r; t = s.Tok(a); p = s.Profile(a); json.dump(%s, open(%r, 'w'))"
                % (HERE, assets, expr, cache))
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        subprocess.run([EXPORT_PY, "-c", code], check=True)
    with open(cache) as f:
        return json.load(f)


def second_turn_ids(cache: Optional[str] = None, assets: Optional[str] = None) -> List[int]:
    """Token ids of a second turn (llm_study.Profile.second_turn_ids: a chat's
    "<|im_end|>\\n" + the SECOND_TURN user block + the generation prompt, a text
    profile's STORY_SECOND), appended to a prompt's decoded answer and
    prefilled at the position the decode steps reached (the phase-5 gates)."""
    return _cached_ids(assets, cache, "second_turn_ids.json",
                       "[int(i) for i in p.second_turn_ids(t)]")


def with_second_turns(ids: Dict[str, List[int]], names: Sequence[str],
                      assets: Optional[str] = None) -> Dict[str, List[int]]:
    """ids plus a "<name>/turn2" continuation right after each prompt in names."""
    out: Dict[str, List[int]] = {}
    t2 = second_turn_ids(assets=assets) if names else []
    for n, t in ids.items():
        out[n] = t
        if n in names:
            out[n + "/turn2"] = list(t2)
    return out


def tokenize_prompts(names: Optional[Sequence[str]] = None, cache: Optional[str] = None,
                     assets: Optional[str] = None) -> Dict[str, List[int]]:
    """Token ids of the model's study prompts (llm_study.Profile.prompt_ids:
    PROMPTS in the chat template, from the sink on; a text profile's
    STORY_PROMPTS after the BOS), with its own tokenizer."""
    ids = _cached_ids(assets, cache, "prompt_ids.json",
                      "{n: [int(i) for i in p.prompt_ids(t, m)] for n, m in p.prompts}")
    if names:
        ids = {n: ids[n] for n in names}
    return ids
