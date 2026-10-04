#!/usr/bin/env python3
"""Summarise a VectorOPKernel HLS synthesis and diff it against the previous
one (MatmulKernel is SystemVerilog since MATMUL_RTL_PLAN phase 4: its gate is
``make synth_matmul_rtl``).

Reads ``csynth.rpt`` and the vitis-run log of ``make synthesize_<k>_kv260`` and
prints the top-level slack / resources, every pipelined loop that is not II=1,
every row with an Issue / Violation type or negative slack, the m_axi port
table and the s_axi_ctrl register map, plus the log's ERROR / SCHED 204-65 /
II-violation / partial-write counts.  Anything that needs the user's attention
is printed as a ``FLAG:`` line.

With a baseline (a copy of the previous run's csynth.rpt, by default
``<build>/kernels/<k>/kv260/<k>_csynth_last.rpt``) it also diffs the resources
(>10 % moves flagged), the pipelined-loop count, the m_axi table and the
register map; a register-map or port change is an INTERFACE change.  The
current report is then copied over the baseline (``--no-save`` keeps it).

Exit codes: 0 summarised (flags or not), 2 report missing or unparseable.
"""
import argparse
import os
import re
import shutil
import sys

KERNELS = {  # kernel -> (top function, csynth.rpt below <build>/kernels/<k>/kv260)
    "vectorop": ("VectorOPKernel", "vadd_kv260/solution1/syn/report/csynth.rpt"),
}
PERF_COLS = ["name", "issue", "violation", "iter_lat", "ii", "trip", "pipelined",
             "lat_cycles", "lat_ns", "slack", "BRAM", "DSP", "FF", "LUT", "URAM"]
LOG_PATTERNS = [("ERROR", r"^ERROR:"),
                ("SCHED 204-65 (pipeline directive not honoured)",
                 r"SCHED 204-65|Unable to satisfy pipeline directive"),
                ("II Violation", r"II Violation"),
                ("Inferring partial write", r"Inferring partial write")]


def section(lines, start_re, stop_re):
    out, inside = [], False
    for line in lines:
        if not inside and re.search(start_re, line):
            inside = True
            continue
        if inside:
            if re.search(stop_re, line):
                break
            out.append(line)
    return out


def table_rows(lines, first_cell_re):
    rows = []
    for line in lines:
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells and re.match(first_cell_re, cells[0]):
            rows.append(cells)
    return rows


def number(cell):
    m = re.match(r"-?\d+(\.\d+)?", cell or "")
    return float(m.group(0)) if m else None


def parse(path):
    with open(path) as f:
        lines = f.read().splitlines()
    date = next((ln.split(":", 1)[1].strip() for ln in lines if "* Date:" in ln), "?")
    perf = [dict(zip(PERF_COLS, c, strict=False))
            for c in table_rows(section(lines, r"Performance & Resource Estimates",
                                        r"Name Prefix"), r"^[+o] ")]
    maxi = table_rows(section(lines, r"^\* M_AXI", r"^\s*$"), r"^m_axi_")
    regs = table_rows(section(lines, r"^\* S_AXILITE Registers", r"^\s*$"), r"^s_axi_")
    if not perf or not regs:
        raise ValueError("no performance table or register map found")
    return {"date": date, "perf": perf, "maxi": maxi,
            "regs": [(r[1], r[2], r[3], r[4]) for r in regs]}  # name, offset, width, access


def loops(perf):
    return [r for r in perf if r["name"].startswith("o ")]


def pipelined(perf):
    return [r for r in loops(perf) if r["pipelined"] == "yes"]


def fmt_top(top):
    return (f"slack {top['slack']} ns  BRAM {top['BRAM']}  DSP {top['DSP']}  "
            f"FF {top['FF']}  LUT {top['LUT']}  URAM {top['URAM']}")


def fmt_maxi(r):
    # Interface | R/W | width | addr | latency | offset | register | widen |
    # max rd burst | max wr burst | rd outstanding | wr outstanding | resource
    return (f"{r[0]:<12} {r[1]:<10} {r[2]:<11} burst rd {r[8]}/wr {r[9]}  "
            f"outstanding rd {r[10]}/wr {r[11]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kernel", choices=sorted(KERNELS))
    ap.add_argument("--build-dir", default="build",
                    help="CMake build tree (default: ./build); sets the default paths")
    ap.add_argument("--report", help="csynth.rpt to check (default: this kernel's in the build)")
    ap.add_argument("--log", help="vitis-run log (default: <build>/kernels/<k>/kv260/logs/"
                    "hls_run_tcl.log)")
    ap.add_argument("--baseline", help="previous csynth.rpt copy (default: "
                    "<build>/kernels/<k>/kv260/<k>_csynth_last.rpt)")
    ap.add_argument("--no-save", action="store_true",
                    help="do not copy this report over the baseline")
    args = ap.parse_args()

    top_fn, rel = KERNELS[args.kernel]
    kdir = os.path.join(args.build_dir, "kernels", args.kernel, "kv260")
    report = args.report or os.path.join(kdir, rel)
    log = args.log or os.path.join(kdir, "logs", "hls_run_tcl.log")
    baseline = args.baseline or os.path.join(kdir, f"{args.kernel}_csynth_last.rpt")

    try:
        cur = parse(report)
    except FileNotFoundError:
        print(f"ERROR: no csynth report at {report}\n"
              f"       run `make synthesize_{args.kernel}_kv260` first.", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"ERROR: cannot parse {report}: {exc}", file=sys.stderr)
        return 2

    flags = []
    top = cur["perf"][0]
    print(f"{top_fn} csynth ({cur['date']})  {report}")
    print(f"  top: {fmt_top(top)}")
    worst = min((s for s in (number(r["slack"]) for r in cur["perf"]) if s is not None),
                default=None)
    pl = pipelined(cur["perf"])
    not_ii1 = [r for r in pl if r["ii"] != "1"]
    print(f"  worst slack {worst} ns; {len(pl)} pipelined loops, "
          + ("all II=1" if not not_ii1 else f"{len(not_ii1)} with II != 1"))
    for r in not_ii1:
        flags.append(f"loop {r['name'][2:]} pipelined at II={r['ii']}")
    if worst is not None and worst < 0:
        flags.append(f"negative slack {worst} ns")
    for r in cur["perf"]:
        if r["issue"] not in ("-", "") or r["violation"] not in ("-", ""):
            flags.append(f"{r['name']}: issue '{r['issue']}' violation '{r['violation']}'")
    print("  m_axi:")
    for r in cur["maxi"]:
        print(f"    {fmt_maxi(r)}")
    print(f"  s_axi_ctrl: {len(cur['regs'])} registers, "
          + ", ".join(f"{n}@{o}" for n, o, _w, _a in cur["regs"][4:]))

    if os.path.exists(log):
        with open(log, errors="replace") as f:
            text = f.read().splitlines()
        fmax = next((ln.split("Estimated Fmax:")[1].strip() for ln in text
                     if "Estimated Fmax:" in ln), None)
        counts = {name: sum(1 for ln in text if re.search(pat, ln))
                  for name, pat in LOG_PATTERNS}
        print(f"  log: Estimated Fmax {fmax}; "
              + ", ".join(f"{n.split(' (')[0]} x{c}" for n, c in counts.items()))
        for name, c in counts.items():
            if c:
                flags.append(f"log has {c} '{name}' line(s): grep {log}")
    else:
        print(f"  log: not found at {log}")

    if os.path.exists(baseline):
        try:
            prev = parse(baseline)
        except ValueError as exc:
            print(f"ERROR: cannot parse baseline {baseline}: {exc}", file=sys.stderr)
            return 2
        ptop = prev["perf"][0]
        print(f"\nvs baseline {baseline} (synthesised {prev['date']}):")
        print(f"  was: {fmt_top(ptop)}")
        for res in ("BRAM", "DSP", "FF", "LUT", "URAM"):
            a, b = number(ptop[res]), number(top[res])
            if a and b is not None and abs(b - a) > 0.10 * a:
                flags.append(f"{res} {ptop[res]} -> {top[res]} (>10 %)")
        a, b = number(ptop["slack"]), number(top["slack"])
        if a is not None and b is not None and b < a:
            flags.append(f"top-level slack worsened {a} -> {b} ns")
        npl, ppl = len(pl), len(pipelined(prev["perf"]))
        print(f"  pipelined loops {ppl} -> {npl}")
        if npl < ppl:
            flags.append(f"{ppl - npl} fewer pipelined loop(s): a PIPELINE loop may be "
                         "left as FSM states (Pipelined = no, no II entry)")
        pmaxi = [fmt_maxi(r) for r in prev["maxi"]]
        cmaxi = [fmt_maxi(r) for r in cur["maxi"]]
        if pmaxi != cmaxi:
            flags.append("m_axi table changed (port width / burst / outstanding) -- "
                         "INTERFACE change")
            for row in pmaxi:
                if row not in cmaxi:
                    print(f"  - {row}")
            for row in cmaxi:
                if row not in pmaxi:
                    print(f"  + {row}")
        if prev["regs"] != cur["regs"]:
            flags.append("s_axi_ctrl register map changed -- INTERFACE change")
            for reg in prev["regs"]:
                if reg not in cur["regs"]:
                    print(f"  - register {reg[0]} {reg[1]} w{reg[2]} {reg[3]}")
            for reg in cur["regs"]:
                if reg not in prev["regs"]:
                    print(f"  + register {reg[0]} {reg[1]} w{reg[2]} {reg[3]}")
        else:
            print("  register map and m_axi table unchanged" if pmaxi == cmaxi
                  else "  register map unchanged")
    else:
        print(f"\nNo baseline at {baseline} -- nothing to diff (saved for next time).")

    print(f"\nFLAGS: {len(flags)}" + ("" if flags else " -- none"))
    for fl in flags:
        print(f"  FLAG: {fl}")

    if args.no_save:
        print("(--no-save: baseline left untouched.)")
    elif os.path.abspath(report) != os.path.abspath(baseline):
        shutil.copyfile(report, baseline)
        print(f"Saved this report to {baseline} as the next baseline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
