#!/usr/bin/env python3
"""calib_status.py — is a perf_calibrate.py campaign complete?

Reads <dir>/<id>.cases.json and <id>.calib.json (read-only unless
--drop-failed) and prints, per pass, how many cases are measured and which
failed (calib_runner "ok": 0 — "kernel X not available", "alloc",
"timeout"), the pass-to-pass spread (what `fit` reports as determinism), and
whether <id>.json covers every good measurement.  Exit 0 only when both
passes cover every case with no failure.

`run --resume` SKIPS failed cases (they are stored like measurements) and
`fit` silently ignores them; --drop-failed deletes them from the calib file
so the next `run --resume` measures them again.

--coverage CASES.json lists the calls of another case list (a `cases` dry
run of today's scheduler into a temp dir) that this campaign never measured:
the shipped calls a scheduler change added since the campaign.

usage (from inference-scheduler/):
  .venv/bin/python ../.claude/skills/perf-calibrate/scripts/calib_status.py
      [--bitstream-id ID] [--dir perf_models/kv260] [--drop-failed]
      [--coverage /tmp/x/<id>.cases.json]
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
from pathlib import Path

SCHED = Path(__file__).resolve().parents[4] / "inference-scheduler"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--bitstream-id", default=None,
                    help="default: the id of the .bit named in bitstream_config_kv260.json")
    ap.add_argument("--dir", default=str(SCHED / "perf_models" / "kv260"))
    ap.add_argument("--drop-failed", action="store_true",
                    help="delete failed cases from <id>.calib.json (then: run --resume)")
    ap.add_argument("--coverage", metavar="CASES_JSON", default=None,
                    help="list the calls of this case list the campaign never measured")
    args = ap.parse_args(argv)
    bid = args.bitstream_id
    if not bid:
        sys.path.insert(0, str(SCHED))
        from src.perf_calls import local_bitstream_id
        bid = local_bitstream_id()
        if not bid:
            print("error: no bitstream id (bitstream_config_kv260.json names no built .bit); "
                  "pass --bitstream-id")
            return 2
    d = Path(args.dir)
    cpath, dpath, mpath = d / f"{bid}.cases.json", d / f"{bid}.calib.json", d / f"{bid}.json"
    if not cpath.exists():
        print(f"error: {cpath} missing (perf_calibrate.py cases)")
        return 2
    cases = json.loads(cpath.read_text())["cases"]
    keys = {e["key"] for e in cases}
    by = collections.Counter((e["kernel"], e["set"]) for e in cases)
    print(f"bitstream {bid}: {len(cases)} cases ("
          + ", ".join(f"{k} {s} {n}" for (k, s), n in sorted(by.items())) + ")")
    if not dpath.exists():
        print(f"no measurements yet: {dpath} missing (perf_calibrate.py run)")
        return 1
    data = json.loads(dpath.read_text())
    passes = data.get("passes", {})
    complete = True
    for p in ("1", "2"):
        v = passes.get(p, {})
        bad = collections.Counter(r.get("err", "?") for k, r in v.items() if not r.get("ok"))
        n_ok = sum(1 for k in keys if v.get(k, {}).get("ok"))
        missing = sum(1 for k in keys if k not in v)
        print(f"pass {p}: {n_ok}/{len(keys)} measured, {missing} not run, "
              f"failed: {dict(bad) if bad else 'none'}")
        complete &= n_ok == len(keys)
    p1, p2 = passes.get("1", {}), passes.get("2", {})
    spreads = []
    for k in keys:
        a, b = p1.get(k, {}), p2.get(k, {})
        if a.get("ok") and b.get("ok"):
            m = (a["mean_us"] + b["mean_us"]) / 2
            spreads.append(abs(a["mean_us"] - b["mean_us"]) / m)
    if spreads:
        print(f"spread pass 1 vs 2: median {statistics.median(spreads) * 100:.3f} %, "
              f"max {max(spreads) * 100:.2f} %, {sum(s > 0.005 for s in spreads)} above 0.5 %")
    if mpath.exists():
        model = json.loads(mpath.read_text())
        good = {k for k, r in p1.items() if r.get("ok")}
        stale = len(good - set(model.get("exact", {})))
        print(f"model {mpath.name}: {len(model.get('exact', {}))} exact calls"
              + (f", {stale} measured calls not in it (re-run fit)" if stale else ", up to date"))
    else:
        print(f"no model yet: {mpath.name} missing (perf_calibrate.py fit)")
    if args.coverage:
        other = json.loads(Path(args.coverage).read_text())["cases"]
        unmeasured = [e for e in other if not p1.get(e["key"], {}).get("ok")]
        cnt = collections.Counter((e["kernel"], e["set"]) for e in unmeasured)
        print(f"coverage of {args.coverage}: {len(other) - len(unmeasured)}/{len(other)} measured"
              + (" — unmeasured: " + ", ".join(f"{k} {s} {n}" for (k, s), n in sorted(cnt.items()))
                 if unmeasured else ""))
        for e in unmeasured:
            if e["set"] == "shipped":
                print(f"  {e['key']}  {(e.get('sources') or ['?'])[0]}")
    if args.drop_failed:
        n = 0
        for v in passes.values():
            for k in [k for k, r in v.items() if not r.get("ok")]:
                del v[k]
                n += 1
        if n:
            dpath.write_text(json.dumps(data, indent=0) + "\n")
        print(f"dropped {n} failed entries from {dpath}" + (" (now: run --resume)" if n else ""))
    return 0 if complete else 1


if __name__ == "__main__":
    sys.exit(main())
