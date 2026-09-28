"""
Planning options — the opt-in ``--plan`` mode of doc/plans/TACTICS_PLAN.md.

With planning off (the default) the scheduler decides as it always has:
the cost model's defaults, the preference rules and the frontends' orders,
and the generated projects are byte-identical.  With planning on, the
choices come from the performance model of the target bitstream
(``perf_models/<platform>/<bitstream-id>.json``, TACTICS_PLAN §4.2).

Every tool that generates a project takes the same options through
:func:`add_plan_args` / :func:`plan_options_from_args`, and forwards them to
``inference_scheduler.py`` with :meth:`PlanOptions.argv`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional


@dataclass(frozen=True)
class PlanOptions:
    enabled:         bool = False              # --plan
    perf_model:      Optional[str] = None      # --perf-model FILE (default: by bitstream id)
    report:          bool = False              # --plan-report (report only, no changes)
    pool_budget_mib: Optional[float] = None    # --pool-budget-mib (default: today's pool)
    entry_weights:   Mapping[str, float] = field(default_factory=dict)   # --entry-weights

    @property
    def active(self) -> bool:
        """True when anything planning-related was asked for."""
        return self.enabled or self.report

    def argv(self) -> List[str]:
        """These options as ``inference_scheduler.py`` arguments."""
        out: List[str] = []
        if self.enabled:
            out.append("--plan")
        if self.report:
            out.append("--plan-report")
        if self.perf_model:
            out += ["--perf-model", str(self.perf_model)]
        if self.pool_budget_mib is not None:
            out += ["--pool-budget-mib", f"{self.pool_budget_mib:g}"]
        if self.entry_weights:
            out += ["--entry-weights", format_entry_weights(self.entry_weights)]
        return out

    @classmethod
    def from_config(cls, cfg: Mapping) -> "PlanOptions":
        """From a demo config's ``"plan"`` value: ``true`` / ``false`` or an
        object with the keys ``enabled``, ``perf_model``, ``report``,
        ``pool_budget_mib``, ``entry_weights``."""
        v = cfg.get("plan", False) if cfg else False
        if isinstance(v, bool):
            return cls(enabled=v)
        if not isinstance(v, Mapping):
            raise ValueError(f"'plan' must be true / false or an object, got {v!r}")
        unknown = set(v) - {"enabled", "perf_model", "report", "pool_budget_mib",
                            "entry_weights"}
        if unknown:
            raise ValueError(f"unknown 'plan' keys: {sorted(unknown)}")
        ew = v.get("entry_weights") or {}
        return cls(enabled=bool(v.get("enabled", True)), perf_model=v.get("perf_model"),
                   report=bool(v.get("report", False)),
                   pool_budget_mib=(float(v["pool_budget_mib"])
                                    if v.get("pool_budget_mib") is not None else None),
                   entry_weights=parse_entry_weights(ew) if isinstance(ew, str)
                   else {str(k): float(x) for k, x in ew.items()})


def parse_entry_weights(text: str) -> Dict[str, float]:
    """``"decode=64,prefill_256=1"`` -> ``{"decode": 64.0, "prefill_256": 1.0}``."""
    out: Dict[str, float] = {}
    for part in filter(None, (p.strip() for p in (text or "").split(","))):
        name, sep, val = part.partition("=")
        if not sep or not name.strip():
            raise ValueError(f"entry weight {part!r} is not NAME=WEIGHT")
        w = float(val)
        if w < 0:
            raise ValueError(f"entry weight {part!r} is negative")
        out[name.strip()] = w
    return out


def format_entry_weights(w: Mapping[str, float]) -> str:
    return ",".join(f"{k}={v:g}" for k, v in w.items())


def add_plan_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("planning (doc/plans/TACTICS_PLAN.md)")
    g.add_argument("--plan", action="store_true",
                   help="choose tactics and the issue order from the target bitstream's "
                        "performance model instead of the cost model's defaults and rules; "
                        "bit-identical results either way")
    g.add_argument("--perf-model", metavar="FILE", default=None,
                   help="performance model file (default: perf_models/<platform>/"
                        "<bitstream-id>.json for the configured bitstream)")
    g.add_argument("--plan-report", action="store_true",
                   help="write the planner's report (predicted times, where the CPU "
                        "waits) without changing the project")
    g.add_argument("--pool-budget-mib", type=float, default=None, metavar="MIB",
                   help="--plan: a reordered schedule's intermediates' pool (slot reuse) "
                        "may not grow past this (default: its size in the model order)")
    g.add_argument("--entry-weights", default="", metavar="NAME=W,...",
                   help="--plan: relative frequency of each entry of a Llama multi-entry "
                        "project, for the prefill width shared with decode (default "
                        "decode=64,head=64, others 1)")


def plan_options_from_args(args: argparse.Namespace) -> PlanOptions:
    return PlanOptions(enabled=bool(getattr(args, "plan", False)),
                       perf_model=getattr(args, "perf_model", None),
                       report=bool(getattr(args, "plan_report", False)),
                       pool_budget_mib=getattr(args, "pool_budget_mib", None),
                       entry_weights=parse_entry_weights(getattr(args, "entry_weights", "")))


_MODELS: Dict[str, object] = {}


class PlanError(RuntimeError):
    """Planning was asked for but cannot run (no performance model)."""


def resolve_perf_model(opts: PlanOptions):
    """The PerfModel planning uses: ``opts.perf_model``, else the model of the
    bitstream named in bitstream_config_kv260.json
    (perf_models/kv260/<bitstream-id>.json).  Loaded once per path."""
    from .perf_calls import local_bitstream_id
    from .perf_model import PerfModel, default_model_path
    path = opts.perf_model
    if not path:
        bid = local_bitstream_id()
        p = default_model_path(bid)
        if p is None:
            raise PlanError(
                "--plan: no performance model"
                + (f" for bitstream {bid} (perf_models/kv260/{bid}.json)" if bid else
                   " (bitstream_config_kv260.json names no built bitstream)")
                + "; run perf_calibrate.py all, or pass --perf-model")
        path = str(p)
    if path not in _MODELS:
        _MODELS[path] = PerfModel.load(path)
    return _MODELS[path]


def plan_summary(graph) -> Optional[str]:
    """One line on what planning did to ``graph`` (None without --plan):
    the model, the choices changed and the predicted kernel time."""
    pm = getattr(graph, "perf_model", None)
    log = getattr(graph, "plan_log", None) or []
    if pm is None:
        return None
    changed = sum(e["decision"] == "planned" for e in log)
    base = sum(e["baseline_us"] or 0.0 for e in log)
    chosen = sum((e.get("chosen_us") if e.get("chosen_us") is not None else e["baseline_us"]) or 0.0
                 for e in log)
    out = (f"performance model {pm.platform}/{pm.bitstream}: {changed} of {len(log)} MatMul "
           f"choices changed, predicted MatMul time (one run of every entry) "
           f"{base / 1e3:.1f} -> {chosen / 1e3:.1f} ms")
    orders = [o for o in ([getattr(graph, "order_log", None)]
                          + [getattr(g, "order_log", None) for _, g in getattr(graph, "entries", [])])
              if o and o.get("applied")]
    if orders:
        out += (f"; issue order changed in {len(orders)} graph(s), simulated "
                + ", ".join(f"{o['before_us'] / 1e3:.1f} -> {o['after_us'] / 1e3:.1f} ms"
                            for o in orders))
    return out


# Default relative frequencies of a Llama project's entries for choices
# shared across entries (a chat answer decodes tens of tokens per prefill).
DEFAULT_ENTRY_WEIGHTS = {"decode": 64.0, "head": 64.0}


def entry_weight(opts: PlanOptions, name: str) -> float:
    if name in opts.entry_weights:
        return float(opts.entry_weights[name])
    return DEFAULT_ENTRY_WEIGHTS.get(name, 1.0)


__all__ = ("PlanOptions", "PlanError", "add_plan_args", "plan_options_from_args",
           "parse_entry_weights", "format_entry_weights", "resolve_perf_model",
           "entry_weight", "DEFAULT_ENTRY_WEIGHTS", "plan_summary")
