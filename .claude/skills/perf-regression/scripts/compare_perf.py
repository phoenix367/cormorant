#!/usr/bin/env python3
"""Compare KV260 performance results against the baseline of the bitstream.

Inputs (at least one):
  --run  RUN.json            run_remote_perf.py --json output (the 60 kernel cases)
  --demo [NAME=]RESULTS.json a demo's build/results.json (mnist, image_classification,
                             bert_squad); NAME defaults to the directory above build/

Baseline: <skill>/baselines/<platform>-<bitstream-id>.json (or --baseline FILE).
The bitstream id is the first 12 hex digits of the SHA-256 of the flat .bin the
board loads (`sha256sum /lib/firmware/pl.bin` on the board); without
--bitstream-id it is the local one (src.perf_calls.local_bitstream_id(), the
.bit named in inference-scheduler/bitstream_config_kv260.json).

Per case: latency delta % (and throughput delta %).  A latency increase above
--threshold % (and above --min-delta-ms) is a REGRESSION, a decrease an
improvement.  A failed case (ok=false) and a changed demo result (MNIST
correct count, top-1 class / logit, BERT EM / F1) count as regressions too.
Missing, new and redefined cases (same label, other fields) are reported.

--record writes the inputs into the baseline (a clean run only); the section
it replaces (kernels, or one demo) moves to the file's history.

Exit: 0 no regression, 1 regression, 2 input error, 3 no baseline for this
bitstream id (nothing compared; --record creates it).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

SKILL = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[4]
BASELINES = SKILL / "baselines"
KERNEL_ORDER = ("VectorOPKernel", "MatmulKernel", "ConvKernel", "PoolingKernel")
EXIT_OK, EXIT_REGRESSION, EXIT_INPUT, EXIT_NO_BASELINE = 0, 1, 2, 3


class InputError(Exception):
    pass


# ── inputs ───────────────────────────────────────────────────────────────────

def _load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise InputError(f"{path}: {exc}") from exc


def _mtime(path: Path) -> str:
    return dt.datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")


def load_run(path: Path) -> dict:
    """run_remote_perf.py --json → {kernel/label: entry}."""
    data = _load_json(path)
    if not isinstance(data, list):
        raise InputError(f"{path}: not a run_remote_perf.py --json list")
    cases = {}
    for r in data:
        try:
            key = f"{r['kernel']}/{r['label']}"
            mkey = "gbs" if "gbs" in r else "gops"
            cases[key] = {"fields": r.get("fields", {}), "ok": bool(r["ok"]),
                          "lat_ms": r.get("lat_ms"), "thr": r.get(mkey), "thr_unit": mkey}
        except KeyError as exc:
            raise InputError(f"{path}: entry without {exc}") from exc
    return cases


def _demo_checks(metrics: dict, accuracy: dict) -> dict:
    """The bit-exact result values of a demo run (they must not move)."""
    checks = {}
    for k in ("correct", "timed", "examples", "uid_ok"):
        if k in metrics:
            checks[k] = metrics[k]
    for res in metrics.get("results", []):
        if res.get("top"):
            top = res["top"][0]
            checks[f"top1[{res.get('name', '?')}]"] = f"{top.get('class_id')}:{top.get('logit')}"
    board = (accuracy or {}).get("board", {})
    for k in ("em", "f1"):
        if k in board:
            checks[k] = round(float(board[k]), 4)
    return checks


def _demo_entry(metrics: dict, accuracy: dict, ok: bool) -> dict:
    return {"ok": ok, "lat_ms": metrics.get("mean_ms"), "p50_ms": metrics.get("p50_ms"),
            "thr": metrics.get("throughput_ips"), "thr_unit": "ips",
            "checks": _demo_checks(metrics, accuracy)}


def load_demo(path: Path) -> dict:
    """A demo's build/results.json → {model: entry}.  Two formats: a list of
    {name, ok, metrics} (mnist, image_classification) or one dict with
    metrics / accuracy / steps (bert_squad)."""
    data = _load_json(path)
    models = {}
    if isinstance(data, list):
        for r in data:
            if "name" not in r:
                raise InputError(f"{path}: list entry without 'name'")
            models[r["name"]] = _demo_entry(r.get("metrics") or {}, {}, bool(r.get("ok")))
    elif isinstance(data, dict) and "metrics" in data:
        m = data["metrics"]
        ok = all(s.get("ok") for s in data.get("steps", [])) and m.get("mean_ms") is not None
        models[m.get("model", path.parent.parent.name)] = _demo_entry(m, data.get("accuracy"), ok)
    else:
        raise InputError(f"{path}: not a demo results.json")
    return models


def parse_demo_arg(arg: str):
    name, sep, path = arg.partition("=")
    if not sep:
        path = arg
        p = Path(path).resolve()
        name = p.parent.parent.name if p.parent.name == "build" else p.stem
    return name, Path(path)


# ── comparison ───────────────────────────────────────────────────────────────

def classify(base: dict, now: dict, threshold: float, min_ms: float) -> tuple:
    """(flag, dlat %, dthr %) for one case."""
    if not now["ok"] or now.get("lat_ms") is None:
        return "FAILED", None, None
    if "fields" in base and base["fields"] != now.get("fields", base["fields"]):
        return "REDEFINED", None, None
    b, n = base["lat_ms"], now["lat_ms"]
    dlat = 100.0 * (n / b - 1.0) if b else 0.0
    dthr = (100.0 * (now["thr"] / base["thr"] - 1.0)
            if base.get("thr") and now.get("thr") is not None else None)
    flag = ""
    if dlat > threshold and n - b > min_ms:
        flag = "REGRESSION"
    elif dlat < -threshold and b - n > min_ms:
        flag = "improved"
    if base.get("checks") and now.get("checks") != base["checks"]:
        flag = "RESULT CHANGED" + (f" + {flag}" if flag else "")
    return flag, dlat, dthr


def _fmt(v, spec, width):
    return format(v, spec) if v is not None else "—".rjust(width)


def compare(sections: list, threshold: float, min_ms: float, brief: bool) -> dict:
    """sections: [(title, base cases, now cases)].  Prints the table, returns counts."""
    def full(title, key):      # kernel cases are already "<kernel>/<label>"
        return key if title == "kernels" else f"{title}/{key}"

    rows, missing, new = [], [], []
    for title, base, now in sections:
        for key in base:
            if key not in now:
                missing.append(full(title, key))
        for key, n in now.items():
            if key not in base:
                new.append(full(title, key))
                continue
            flag, dlat, dthr = classify(base[key], n, threshold, min_ms)
            rows.append((full(title, key), base[key], n, flag, dlat, dthr))

    if not brief:
        sw = max([len("section")] + [len(r[0].split("/", 1)[0]) for r in rows]) + 1
        w = max([len("case")] + [len(r[0].split("/", 1)[1]) for r in rows]) + 2
        print(f"  {'section':<{sw}} {'case':<{w}} {'base ms':>10} {'now ms':>10} "
              f"{'Δlat %':>8} {'Δthr %':>8}  flag")
        last = None
        for name, b, n, flag, dlat, dthr in rows:
            sec, case = name.split("/", 1)
            if sec != last and last is not None:
                print()
            last = sec
            print(f"  {sec:<{sw}} {case:<{w}} {_fmt(b['lat_ms'], '10.4f', 10)} "
                  f"{_fmt(n.get('lat_ms'), '10.4f', 10)} {_fmt(dlat, '+8.2f', 8)} "
                  f"{_fmt(dthr, '+8.2f', 8)}  {flag}")
            if flag.startswith("RESULT CHANGED"):
                for k in sorted(set(b["checks"]) | set(n["checks"])):
                    if b["checks"].get(k) != n["checks"].get(k):
                        print(f"  {'':<{sw}}   {k}: {b['checks'].get(k)} -> {n['checks'].get(k)}")
        print()

    def names(pred):
        return [name for name, _, _, f, _, _ in rows if pred(f)]

    reg = names(lambda f: "REGRESSION" in f)
    changed = names(lambda f: f.startswith("RESULT CHANGED"))
    failed = names(lambda f: f == "FAILED")
    redefined = names(lambda f: f == "REDEFINED")
    improved = names(lambda f: "improved" in f)
    timed = [r for r in rows if r[4] is not None]
    worst = sorted(timed, key=lambda r: -r[4])[:5]

    print(f"compared {len(rows)} cases (threshold ±{threshold:g} %, min Δ {min_ms:g} ms): "
          f"{len(reg)} regressions, {len(improved)} improved, {len(failed)} failed, "
          f"{len(changed)} result changes, {len(missing)} missing, {len(new)} new, "
          f"{len(redefined)} redefined")
    for label, items in (("REGRESSION", reg), ("FAILED", failed), ("RESULT CHANGED", changed),
                         ("improved", improved), ("missing (in the baseline, not run)", missing),
                         ("new (not in the baseline)", new),
                         ("redefined (same label, other fields: not compared)", redefined)):
        if items:
            print(f"  {label}: {', '.join(items)}")
    if worst:
        print("worst 5 (largest latency increase):")
        for name, b, n, _, dlat, _ in worst:
            print(f"  {name}: {b['lat_ms']:.4f} -> {n['lat_ms']:.4f} ms ({dlat:+.2f} %)")
    bad = bool(reg or failed or changed)
    print(f"verdict: {'REGRESSION' if bad else 'PASS'}")
    return {"bad": bad}


# ── baseline file ────────────────────────────────────────────────────────────

def _strip(entry: dict) -> dict:
    return {k: v for k, v in entry.items() if k != "ok"}


def record(path: Path, baseline: dict | None, platform: str, bitstream_id: str,
           run: tuple | None, demos: list) -> None:
    today = dt.date.today().isoformat()
    b = baseline or {"platform": platform, "bitstream_id": bitstream_id,
                     "kernels": None, "demos": {}, "history": []}
    b.setdefault("demos", {})
    b.setdefault("history", [])

    def retire(section: str, old: dict | None, cases_key: str):
        if old:
            b["history"].append({"section": section, "run": old.get("run"),
                                 "source": old.get("source"), "replaced": today,
                                 "lat_ms": {k: v["lat_ms"] for k, v in old[cases_key].items()}})

    if run:
        src, cases = run
        retire("kernels", b.get("kernels"), "cases")
        b["kernels"] = {"run": _mtime(src), "source": str(src.resolve()),
                        "cases": {k: _strip(v) for k, v in cases.items()}}
    for name, src, models in demos:
        retire(f"demo:{name}", b["demos"].get(name), "models")
        b["demos"][name] = {"run": _mtime(src), "source": str(src.resolve()),
                            "models": {k: _strip(v) for k, v in models.items()}}
    b["updated"] = today
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(b, indent=1, ensure_ascii=False) + "\n")
    print(f"recorded baseline: {path}")


def local_bitstream_id() -> str | None:
    sys.path.insert(0, str(REPO / "inference-scheduler"))
    try:
        from src.perf_calls import local_bitstream_id as lbi
    except ImportError as exc:
        raise InputError(f"cannot import src.perf_calls ({exc}); pass --bitstream-id") from exc
    return lbi()


# ── main ─────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, help="run_remote_perf.py --json output")
    ap.add_argument("--demo", action="append", default=[], metavar="[NAME=]RESULTS.json",
                    help="a demo's build/results.json (repeatable)")
    ap.add_argument("--platform", default="kv260")
    ap.add_argument("--bitstream-id", help="board: sha256sum /lib/firmware/pl.bin | cut -c1-12 "
                                           "(default: the local .bit's id)")
    ap.add_argument("--baseline", type=Path, help="baseline file (default: "
                    "baselines/<platform>-<bitstream-id>.json next to this script's skill)")
    ap.add_argument("--threshold", type=float, default=2.0, help="flag |Δlat| above this %% (2)")
    ap.add_argument("--min-delta-ms", type=float, default=0.001,
                    help="and above this many ms (0.001: short calls jitter by up to 0.5 us)")
    ap.add_argument("--record", action="store_true",
                    help="write the inputs into the baseline (after comparing)")
    ap.add_argument("--brief", action="store_true", help="summary only, no per-case table")
    args = ap.parse_args(argv)

    try:
        if not args.run and not args.demo:
            raise InputError("nothing to compare: pass --run and / or --demo")
        run = (args.run, load_run(args.run)) if args.run else None
        demos = [(name, p, load_demo(p)) for name, p in map(parse_demo_arg, args.demo)]

        bid = args.bitstream_id
        if not bid and args.baseline is None:
            bid = local_bitstream_id()
            if not bid:
                raise InputError("no local bitstream (bitstream_config_kv260.json / .bit); "
                                 "pass --bitstream-id")
            print(f"bitstream id {bid} (local .bit — pass the board's with --bitstream-id)")
        path = args.baseline or BASELINES / f"{args.platform}-{bid}.json"
        baseline = _load_json(path) if path.exists() else None
        if baseline and bid and baseline.get("bitstream_id") != bid:
            print(f"baseline {path} is for bitstream {baseline.get('bitstream_id')}, not {bid}: "
                  f"not compared")
            return EXIT_NO_BASELINE
        bid = bid or (baseline or {}).get("bitstream_id")
        if args.record and run and any(not c["ok"] or c["lat_ms"] is None
                                       for c in run[1].values()):
            raise InputError("the run has failed cases: not recorded")
        if args.record and any(not m["ok"] for _, _, ms in demos for m in ms.values()):
            raise InputError("a demo run failed: not recorded")
    except InputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INPUT

    if baseline is None:
        others = sorted(p.name for p in BASELINES.glob(f"{args.platform}-*.json"))
        print(f"no baseline for {args.platform} bitstream {bid} ({path}); "
              f"recorded ones: {', '.join(others) or 'none'}")
        if not args.record:
            print("nothing compared — rerun with --record to make these results the baseline")
            return EXIT_NO_BASELINE
        record(path, None, args.platform, bid, run, demos)
        return EXIT_OK

    sections, skipped = [], []
    if run:
        if baseline.get("kernels"):
            k = baseline["kernels"]
            print(f"kernels: baseline run {k.get('run')} ({k.get('source')})")
            order = {n: i for i, n in enumerate(KERNEL_ORDER)}
            now = dict(sorted(run[1].items(), key=lambda kv: order.get(kv[0].split("/")[0], 9)))
            sections.append(("kernels", k["cases"], now))
        else:
            skipped.append("kernels")
    for name, _, models in demos:
        d = baseline["demos"].get(name)
        if d:
            print(f"demo {name}: baseline run {d.get('run')} ({d.get('source')})")
            sections.append((name, d["models"], models))
        else:
            skipped.append(f"demo {name}")
    if skipped:
        print(f"not in the baseline (not compared; --record adds them): {', '.join(skipped)}")
    print(f"platform {args.platform}, bitstream {bid}\n")
    result = compare(sections, args.threshold, args.min_delta_ms, args.brief)
    if args.record:
        record(path, baseline, args.platform, bid, run, demos)
    return EXIT_REGRESSION if result["bad"] else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
