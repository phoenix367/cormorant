#!/usr/bin/env python3
"""Compare a MatmulKernel / VectorOPKernel behaviour-test report against the
previous run.

Reads the freshly-written test-stand report
(``<build>/kernels/<k>/kv260/{matmul_op,vector_op}_test_report.json``),
compares every case's ``duration_ns`` with the baseline, prints a delta table
sorted by absolute movement, then overwrites the baseline snapshot
(``<build>/kernels/<k>/kv260/<k>_timing_last.json``) with the current run so
the next invocation compares against it.  ``--no-save`` leaves the baseline
untouched (a one-off comparison, or keeping a known-good reference).

Cases are matched by label AND geometry: the VectorOP report repeats labels
(``ADD`` is 11 cases of different ``size``), so a label-only key would collapse
them.  Same-label cases are shown as ``label field=value`` with the geometry
fields that tell them apart.

``--baseline`` accepts a snapshot written by this script or a raw
``*_test_report.json`` (e.g. a report kept from an earlier run).

Exit codes: 0 compared (or first run), 1 the report records failures,
2 report / baseline missing, unreadable, or for another kernel.
"""
import argparse
import datetime
import json
import os
import sys
from collections import defaultdict

KERNELS = {
    "matmul": ("MatmulKernel", "matmul_op_test_report.json", "matmul_timing_last.json"),
    "vectorop": ("VectorOPKernel", "vector_op_test_report.json", "vectorop_timing_last.json"),
}
NAME_MAX = 64


def case_key(label, geometry):
    return label + " " + json.dumps(geometry or {}, sort_keys=True)


def cases_from_report(report):
    """[(key, label, geometry, index, duration_ns)] of a raw test-stand report."""
    out, seen = [], defaultdict(int)
    for t in report["tests"]:
        key = case_key(t["label"], t.get("geometry"))
        seen[key] += 1
        if seen[key] > 1:  # identical label + geometry: keep them apart
            key = f"{key} #{seen[key]}"
        out.append((key, t["label"], t.get("geometry") or {}, t.get("index"), t["duration_ns"]))
    return out


def load_baseline(path):
    """(kernel, cases, sim_time_ns, description) from a snapshot or a raw report."""
    with open(path) as f:
        data = json.load(f)
    if isinstance(data.get("tests"), list):  # raw test-stand report
        mtime = datetime.datetime.fromtimestamp(os.path.getmtime(path))
        return (data.get("kernel"), cases_from_report(data), data.get("sim_time_ns"),
                f"raw report, written {mtime:%Y-%m-%d %H:%M}")
    cases = [(c["key"], c["label"], c.get("geometry") or {}, c.get("index"), c["duration_ns"])
             for c in data["cases"]]
    return (data.get("kernel"), cases, data.get("sim_time_ns"),
            f"snapshot saved {data.get('saved_at', '?')} from a report written "
            f"{data.get('report_written', '?')}")


def display_names(case_lists):
    """key -> readable name; same-label cases get the geometry fields that differ."""
    by_label = defaultdict(dict)
    for cases in case_lists:
        for key, label, geom, _idx, _dur in cases:
            by_label[label][key] = geom
    names = {}
    for label, members in by_label.items():
        if len(members) == 1:
            (key,) = members
            names[key] = label
            continue
        fields = sorted({f for g in members.values() for f in g})
        varying = [f for f in fields if len({json.dumps(g.get(f)) for g in members.values()}) > 1]
        for key, geom in members.items():
            extra = " ".join(f"{f}={geom.get(f)}" for f in varying)
            suffix = key.rsplit(" #", 1)[1] if " #" in key else ""
            names[key] = (f"{label} {extra}".strip() + (f" #{suffix}" if suffix else ""))
    return {k: (v if len(v) <= NAME_MAX else v[:NAME_MAX - 2] + "..") for k, v in names.items()}


def pct(prev, now):
    return 100.0 * (now - prev) / prev if prev else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kernel", choices=sorted(KERNELS))
    ap.add_argument("--build-dir", default="build",
                    help="CMake build tree (default: ./build); sets the default paths")
    ap.add_argument("--report", help="behaviour-test report (default: "
                    "<build>/kernels/<k>/kv260/<stand>_test_report.json)")
    ap.add_argument("--baseline", help="snapshot or raw report to compare against "
                    "(default: <build>/kernels/<k>/kv260/<k>_timing_last.json)")
    ap.add_argument("--no-save", action="store_true",
                    help="do not overwrite the baseline snapshot with this run")
    ap.add_argument("--top", type=int, default=0,
                    help="print only the N largest movers (default: every changed case)")
    ap.add_argument("--slower-pct", type=float, default=1.0,
                    help="list matched cases slower than this %% as regressions (default 1.0)")
    args = ap.parse_args()

    kname, report_file, snap_file = KERNELS[args.kernel]
    kdir = os.path.join(args.build_dir, "kernels", args.kernel, "kv260")
    report_path = args.report or os.path.join(kdir, report_file)
    baseline_path = args.baseline or os.path.join(kdir, snap_file)

    try:
        with open(report_path) as f:
            report = json.load(f)
        cur = cases_from_report(report)
    except FileNotFoundError:
        print(f"ERROR: no behaviour-test report at {report_path}\n"
              f"       run `make behavior_test_{args.kernel}` first.", file=sys.stderr)
        return 2
    except (ValueError, KeyError, TypeError) as exc:
        print(f"ERROR: cannot parse {report_path}: {exc}", file=sys.stderr)
        return 2
    if report.get("kernel") != kname:
        print(f"ERROR: {report_path} is a {report.get('kernel')} report, not {kname}",
              file=sys.stderr)
        return 2

    summary = report.get("summary", {})
    written = datetime.datetime.fromtimestamp(os.path.getmtime(report_path))
    print(f"{kname} behaviour test: {summary.get('passed')}/{summary.get('total')} PASS"
          + ("" if summary.get("all_passed") else f"  ({summary.get('failed')} FAILED)")
          + f"   [{report_path}, written {written:%Y-%m-%d %H:%M}]")
    if not summary.get("all_passed"):
        print("ERROR: the behaviour test reported failures; fix them before comparing timings.",
              file=sys.stderr)
        return 1

    cur_sim = report.get("sim_time_ns")
    cur_by_key = {c[0]: c for c in cur}

    if os.path.exists(baseline_path):
        try:
            b_kernel, prev, prev_sim, b_desc = load_baseline(baseline_path)
        except (ValueError, KeyError, TypeError) as exc:
            print(f"ERROR: cannot parse baseline {baseline_path}: {exc}", file=sys.stderr)
            return 2
        if b_kernel and b_kernel != kname:
            print(f"ERROR: baseline {baseline_path} is for {b_kernel}, not {kname}",
                  file=sys.stderr)
            return 2
        prev_by_key = {c[0]: c for c in prev}
        names = display_names([prev, cur])
        print(f"Baseline: {baseline_path} ({b_desc})")

        matched = [k for k in cur_by_key if k in prev_by_key]
        new = [k for k in cur_by_key if k not in prev_by_key]
        gone = [k for k in prev_by_key if k not in cur_by_key]
        rows = [(names[k], prev_by_key[k][4], cur_by_key[k][4]) for k in matched]
        moved = sorted((r for r in rows if r[1] != r[2]),
                       key=lambda r: abs(r[2] - r[1]), reverse=True)
        shown = moved[:args.top] if args.top > 0 else moved
        width = max([len(r[0]) for r in rows] + [36])
        head = f"  {'case':<{width}} {'prev':>10} {'now':>10} {'delta':>10} {'pct':>7}"
        print()
        print(head)
        print("  " + "-" * (len(head) - 2))
        for name, p, n in shown:
            print(f"  {name:<{width}} {p:>10} {n:>10} {n - p:>+10} {pct(p, n):>+6.1f}%")
        if len(shown) < len(moved):
            print(f"  ... {len(moved) - len(shown)} more changed case(s) (--top 0 lists all)")
        print(f"  ({len(rows) - len(moved)} of {len(rows)} matched cases unchanged)")
        print("  " + "-" * (len(head) - 2))
        p_tot, n_tot = sum(r[1] for r in rows), sum(r[2] for r in rows)
        label = f"TOTAL ({len(rows)} matched cases)"
        print(f"  {label:<{width}} {p_tot:>10} {n_tot:>10} {n_tot - p_tot:>+10} "
              f"{pct(p_tot, n_tot):>+6.1f}%")
        if prev_sim is not None and cur_sim is not None and not new and not gone:
            print(f"  {'sim_time_ns':<{width}} {prev_sim:>10} {cur_sim:>10} "
                  f"{cur_sim - prev_sim:>+10} {pct(prev_sim, cur_sim):>+6.1f}%")
        for title, keys, src in (("New cases (not in the baseline)", new, cur_by_key),
                                 ("Removed cases (only in the baseline)", gone, prev_by_key)):
            if keys:
                print(f"\n{title}: {len(keys)}")
                for k in keys:
                    print(f"  {names[k]:<{width}} {src[k][4]:>10}")
        slower = [r for r in rows if pct(r[1], r[2]) > args.slower_pct]
        print(f"\nSlower than +{args.slower_pct:g} %: {len(slower)} case(s)"
              + ("" if slower else " -- none"))
        for name, p, n in sorted(slower, key=lambda r: pct(r[1], r[2]), reverse=True):
            print(f"  {name}: {p} -> {n} ({pct(p, n):+.1f} %)")
    else:
        names = display_names([cur])
        width = max([len(n) for n in names.values()] + [36])
        print(f"\nNo baseline at {baseline_path} -- absolute values only "
              "(the next run will produce a diff).")
        for key, _label, _g, _i, dur in cur:
            print(f"  {names[key]:<{width}} {dur:>12}")
        print(f"  {'TOTAL (' + str(len(cur)) + ' cases)':<{width}} "
              f"{sum(c[4] for c in cur):>12}")
        if cur_sim is not None:
            print(f"  {'sim_time_ns':<{width}} {cur_sim:>12}")

    if args.no_save:
        print("\n(--no-save: baseline left untouched.)")
        return 0
    snapshot = {
        "kernel": kname,
        "saved_at": f"{datetime.datetime.now():%Y-%m-%d %H:%M}",
        "report": os.path.abspath(report_path),
        "report_written": f"{written:%Y-%m-%d %H:%M}",
        "sim_time_ns": cur_sim,
        "summary": summary,
        "cases": [{"key": k, "label": lab, "geometry": g, "index": i, "duration_ns": d}
                  for k, lab, g, i, d in cur],
    }
    os.makedirs(os.path.dirname(os.path.abspath(baseline_path)), exist_ok=True)
    with open(baseline_path, "w") as f:
        json.dump(snapshot, f, indent=1)
    print(f"\nSaved this run to {baseline_path} as the next baseline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
