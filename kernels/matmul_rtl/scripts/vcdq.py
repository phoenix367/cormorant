#!/usr/bin/env python3
"""vcdq.py — tiny VCD query tool for debugging without a waveform viewer.

  vcdq.py FILE list [PATTERN]              list signals whose path matches PATTERN
  vcdq.py FILE at TIME PATTERN...          values of matching signals at TIME
  vcdq.py FILE hist PATTERN [T0 T1]        value changes of matching signals
  vcdq.py FILE table T0 T1 PATTERN...      one row per clock edge (odd times)

Patterns are Python regexes matched against the full dotted path.  Values are
printed in hex.  Times are VCD time units (the testbench dumps 2 per cycle).
"""
import re
import sys


def parse(path):
    sigs = {}          # id -> list of names
    widths = {}
    scope = []
    changes = {}       # id -> list[(t, value)]
    t = 0
    with open(path) as f:
        in_defs = True
        for line in f:
            line = line.strip()
            if not line:
                continue
            if in_defs:
                if line.startswith('$scope'):
                    scope.append(line.split()[2])
                elif line.startswith('$upscope'):
                    scope.pop()
                elif line.startswith('$var'):
                    p = line.split()
                    w, ident, name = int(p[2]), p[3], p[4]
                    full = '.'.join(scope + [name])
                    sigs.setdefault(ident, []).append(full)
                    widths[ident] = w
                    changes.setdefault(ident, [])
                elif line.startswith('$enddefinitions'):
                    in_defs = False
                continue
            c = line[0]
            if c == '#':
                t = int(line[1:])
            elif c in '01xz':
                changes[line[1:]].append((t, line[0]))
            elif c == 'b':
                v, ident = line[1:].split()
                changes[ident].append((t, v))
    return sigs, widths, changes


def fmt(v):
    if v is None:
        return '-'
    if any(ch in v for ch in 'xz'):
        return v
    return hex(int(v, 2))


def value_at(ch, t):
    v = None
    for (tt, vv) in ch:
        if tt > t:
            break
        v = vv
    return v


def main():
    path, cmd = sys.argv[1], sys.argv[2]
    sigs, widths, changes = parse(path)
    names = [(n, i) for i, ns in sigs.items() for n in ns]
    names.sort()

    def match(pats):
        out = []
        for pat in pats:
            r = re.compile(pat)
            out += [(n, i) for (n, i) in names if r.search(n)]
        return out

    if cmd == 'list':
        pat = sys.argv[3] if len(sys.argv) > 3 else '.'
        for n, i in match([pat]):
            print(f'{n} [{widths[i]}]')
    elif cmd == 'at':
        t = int(sys.argv[3])
        for n, i in match(sys.argv[4:]):
            print(f'{n:70s} {fmt(value_at(changes[i], t))}')
    elif cmd == 'hist':
        pat = sys.argv[3]
        t0 = int(sys.argv[4]) if len(sys.argv) > 4 else 0
        t1 = int(sys.argv[5]) if len(sys.argv) > 5 else 1 << 62
        for n, i in match([pat]):
            print(n)
            for (tt, vv) in changes[i]:
                if t0 <= tt <= t1:
                    print(f'   {tt:8d}  {fmt(vv)}')
    elif cmd == 'table':
        t0, t1 = int(sys.argv[3]), int(sys.argv[4])
        sel = match(sys.argv[5:])
        short = [n.split('.')[-1] if len(n.split('.')[-1]) > 2 else '.'.join(n.split('.')[-2:])
                 for n, _ in sel]
        print('time    ' + ' '.join(f'{s:>10s}' for s in short))
        for t in range(t0 | 1, t1 + 1, 2):
            vals = [fmt(value_at(changes[i], t - 1)) for _, i in sel]
            print(f'{t // 2:7d} ' + ' '.join(f'{v:>10s}' for v in vals))


if __name__ == '__main__':
    main()
