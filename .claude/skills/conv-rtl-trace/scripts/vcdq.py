#!/usr/bin/env python3
r"""vcdq.py VCD 'regex1' 'regex2' ... [--from C] [--to C] [--list] [--max N]

Query a VCD of the RTL ConvKernel's Verilator testbench (vlt/Vtb --trace F.fst,
then fst2vcd F.fst > F.vcd).  The testbench dumps twice per cycle: time 2c
after the inputs settle, 2c+1 after the rising edge.  Prints one row per dump
time at which a selected signal changed — the cycle (``+`` = after the edge),
then name=value (hex) of every signal whose hierarchical name matches one of
the regexes (``u_en\.(s_act|eq_ready)$``).  --list prints the matching names
and widths instead; --from / --to bound the cycles, --max the rows (400).
"""
import re
import sys

args = sys.argv[1:]
if not args or args[0] in ("-h", "--help"):
    print(__doc__)
    sys.exit(0 if args else 2)
path = args.pop(0)
t_from, t_to, do_list, maxrows = 0, 1 << 62, False, 400
pats = []
i = 0
while i < len(args):
    a = args[i]
    if a == "--from": t_from = int(args[i + 1]); i += 2; continue
    if a == "--to": t_to = int(args[i + 1]); i += 2; continue
    if a == "--max": maxrows = int(args[i + 1]); i += 2; continue
    if a == "--list": do_list = True; i += 1; continue
    pats.append(re.compile(a)); i += 1

ids = {}        # id -> list of names
scope = []
f = open(path)
for line in f:
    line = line.strip()
    if line.startswith("$scope"):
        scope.append(line.split()[2])
    elif line.startswith("$upscope"):
        scope.pop()
    elif line.startswith("$var"):
        p = line.split()
        width, vid, name = int(p[2]), p[3], p[4]
        full = ".".join(scope[1:] + [name])
        if do_list:
            if not pats or any(r.search(full) for r in pats):
                print(width, full)
            continue
        if any(r.search(full) for r in pats):
            ids.setdefault(vid, []).append((full, width))
    elif line.startswith("$enddefinitions"):
        break
if do_list:
    sys.exit(0)

names = sorted({n for v in ids.values() for n, _ in v})
val = {n: "x" for n in names}
t = 0
changed = False
rows = 0


def emit():
    global rows
    if t_from <= t // 2 <= t_to and rows < maxrows:
        print(f"{t//2:7d}{'+' if t % 2 else ' '} " + " ".join(f"{n.split('.')[-1] if len(names) < 12 else n}={val[n]}" for n in names))
        rows += 1


def tohex(b):
    if any(c in "xz" for c in b):
        return b
    return format(int(b, 2), "x")


for line in f:
    if not line:
        continue
    c = line[0]
    if c == "#":
        nt = int(line[1:])
        if changed:
            emit()
            changed = False
        t = nt
        if t // 2 > t_to:
            break
    elif c == "b":
        b, vid = line[1:].split()
        if vid in ids:
            for n, _ in ids[vid]:
                v = tohex(b)
                if val[n] != v:
                    val[n] = v
                    changed = True
    elif c in "01xz":
        vid = line[1:].strip()
        if vid in ids:
            for n, _ in ids[vid]:
                if val[n] != c:
                    val[n] = c
                    changed = True
