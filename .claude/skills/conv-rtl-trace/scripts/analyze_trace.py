#!/usr/bin/env python3
"""analyze_trace.py <verbose tb log> — per-beat analysis of the conv testbench's
+VERBOSE probes (CK_AXI gmem0 AR/R, gmem3 AW/W/B, CK_ACC acc_stream, process
done markers).  One block per test in the log."""
import re, sys, bisect, collections

def hist(vals, n=6):
    return collections.Counter(vals).most_common(n)

def analyse(lines):
    aw, w, b, ar, r = [], [], [], [], []
    acc_push_gaps, acc_summ, done = [], [], []
    scb = []
    for line in lines:
        m = re.match(r"\[(\d+) ns\]\[CK_AXI\] gmem3 AW: AWVALID=1 AWREADY=1  ADDR=(\S+)  LEN=(\d+)", line)
        if m: aw.append((int(m[1]), int(m[2], 16), int(m[3]))); continue
        m = re.match(r"\[(\d+) ns\]\[CK_AXI\] gmem3 W:  beat#\d+  WSTRB=\S+  WLAST=(\d)", line)
        if m: w.append((int(m[1]), int(m[2]))); continue
        m = re.match(r"\[(\d+) ns\]\[CK_AXI\] gmem3 B:", line)
        if m: b.append(int(m[1])); continue
        m = re.match(r"\[(\d+) ns\]\[CK_AXI\] gmem0 AR: ARVALID=1 ARREADY=1  ADDR=\S+  LEN=(\d+)", line)
        if m: ar.append((int(m[1]), int(m[2]))); continue
        m = re.match(r"\[(\d+) ns\]\[CK_AXI\] gmem0 R:  beat#", line)
        if m: r.append(int(m[1])); continue
        m = re.match(r"\[(\d+) ns\]\[CK_ACC\] push gap (\d+) ns before push #(\d+) \(pops=(\d+) occupancy=(\d+)\)", line)
        if m: acc_push_gaps.append(tuple(int(x) for x in m.groups())); continue
        m = re.match(r"\[(\d+) ns\]\[CK_ACC\] pushes=(\d+) pops=(\d+) occupancy=(\d+)", line)
        if m: acc_summ.append(tuple(int(x) for x in m.groups())); continue
        m = re.match(r"\[(\d+) ns\]\[CK_ACC\] (.*ap_done.*)", line)
        if m: done.append((int(m[1]), m[2])); continue
        m = re.match(r"\[(\d+) ns\]\[SCB\] (PASS|FAIL) +(\S+)", line)
        if m: scb.append((int(m[1]), m[2], m[3]))
    return aw, w, b, ar, r, acc_push_gaps, acc_summ, done, scb

def report(name, t0, t1, aw, w, b, ar, r, gaps, summ, done):
    sel = lambda xs, key=lambda x: x[0]: [x for x in xs if t0 < key(x) <= t1]
    aw, w, b, ar, r = sel(aw), sel(w), sel(b, lambda x: x), sel(ar), sel(r, lambda x: x)
    gaps, summ, done = sel(gaps), sel(summ), sel(done)
    print(f"\n=== {name}  ({(t1 - t0)/1e4:.1f} k cycles @100 MHz) ===")
    if aw:
        print(f"gmem3 write: {len(aw)} AW, {len(w)} W beats, {len(b)} B; burst LEN hist {sorted(collections.Counter(l+1 for *_, l in aw).items())[:6]}")
        d = [y[0]-x[0] for x, y in zip(aw, aw[1:])]
        print(f"  AW→AW spacing (ns): {hist(d)}")
        if w:
            intra = [y[0]-x[0] for x, y in zip(w, w[1:]) if x[1] == 0]
            inter = [y[0]-x[0] for x, y in zip(w, w[1:]) if x[1] == 1]
            print(f"  W intra-burst spacing: {hist(intra, 3)}   inter-burst gaps: {hist(inter, 4)}")
            big = [(y[0]-x[0], i) for i, (x, y) in enumerate(zip(w, w[1:])) if y[0]-x[0] > 1000]
            if big: print(f"  W gaps > 1 µs: {len(big)} (first: {big[:3]})")
            print(f"  write phase: first W {w[0][0]} ns → last {w[-1][0]} ns = {(w[-1][0]-w[0][0])/1e4:.1f} k cycles for {len(w)} beats ({(w[-1][0]-w[0][0])/10/len(w):.2f} cyc/beat)")
        if b:
            outs = [i - bisect.bisect_right(b, t) for i, (t, *_) in enumerate(aw)]
            lat = [bb - a[0] for a, bb in zip(aw, b)]
            print(f"  bursts in flight at AW: max {max(outs)} {sorted(collections.Counter(outs).items())[:6]};  AW→B latency ns: min {min(lat)} median {sorted(lat)[len(lat)//2]} max {max(lat)}")
    if ar:
        d = [y[0]-x[0] for x, y in zip(ar, ar[1:])]
        print(f"gmem0 read: {len(ar)} AR, {len(r)} R beats; LEN hist {sorted(collections.Counter(l+1 for _, l in ar).items())[:6]}; AR→AR spacing {hist(d, 4)}")
    if summ:
        rates = [(s2[1]-s1[1]) / ((s2[0]-s1[0])/10) for s1, s2 in zip(summ, summ[1:]) if s2[0] > s1[0]]
        print(f"acc_stream: {summ[-1][1]} pushes, {summ[-1][2]} pops; push rate per 1024-block: min {min(rates):.2f} max {max(rates):.2f} elem/cycle; occupancy max {max(s[3] for s in summ)}")
    for t, g, n, pops, occ in gaps:
        if n > 1: print(f"  push gap {g/1e3:.1f} µs before push #{n} (pops {pops}, occupancy {occ}) — chunk boundary or stall")
    for t, s in done: print(f"  {t} ns: {s}")

def main():
    lines = open(sys.argv[1], errors="ignore").read().splitlines()
    aw, w, b, ar, r, gaps, summ, done, scb = analyse(lines)
    if not scb:
        print("no [SCB] result lines — did the run reach a verdict? (deadlock → look for 'forward_progress' / 'Pending AW')")
        scb = [(max([x[0] for x in aw] + [x[0] for x in w] + [1]), "?", "(incomplete)")]
    prev = 0
    for t, status, name in scb:
        report(f"{name} {status}", prev, t, aw, w, b, ar, r, gaps, summ, done)
        prev = t

if __name__ == "__main__":
    main()
