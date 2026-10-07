#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# gen_act_rom.py — writes rtl/vo_act_rom.sv, the table ROM of the RTL
# VectorOPKernel's SiLU and GELU activations (vo_act), and checks it.
#
# SiLU and both GELUs are x * F(x) with F(-x) = 1 - F(x), so
#
#     f(x) = max(x, 0) + f(-|x|)
#
# and, as f(x) - f(-|x|) = max(x, 0) is a whole number of LSBs, the rounded
# values obey the same identity.  The ROM holds m(t) = -round(256 * f(-t / 256))
# for t = |x| in LSBs (0 <= m <= 71: 7 bits) over the t where it is non-zero —
# SiLU 2141 entries, GELU 829, GELU tanh 816; 3786 of a 4096 x 7 ROM, one
# BRAM36 (4K x 9) per two lanes — and vo_act computes f(x) = max(x, 0) - m(|x|)
# (0 past a function's segment).  Rounding: to nearest, ties to even, the C++
# model's round_cast (kernels/vectorop/include/VectorOP.h).
#
# The values come from the C++ model's formulas in IEEE double
# (kernels/vectorop/kernel/VectorOP.cpp act_fn).  Every run asserts that no
# Q8.8 input lies within 1e-9 LSB of a rounding tie — so any implementation
# within a few ulp rounds alike (the nearest is 1.6e-5 LSB away) — and that the
# identity above reproduces the direct formula for all 65 536 inputs; --exact
# also recomputes every value with mpmath (40 digits).
#
#   gen_act_rom.py               write rtl/vo_act_rom.sv
#   gen_act_rom.py --check       exit 1 if rtl/vo_act_rom.sv is not what it writes
#   gen_act_rom.py --exact       also check the tables against mpmath (~20 s)
# ---------------------------------------------------------------------------
import argparse
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "rtl", "vo_act_rom.sv")
GENERATOR = "kernels/vectorop_rtl/scripts/gen_act_rom.py"

FRAC = 8                    # Q8.8
SCALE = 1 << FRAC
ROM_D = 4096
ROM_W = 7
MIN_TIE_MARGIN = 1e-9       # LSB

# The table functions in ROM order (name, VectorOP.h Act code, formula in double).
FUNCS = [
    ("SILU", 4, lambda x: x / (1.0 + math.exp(-x))),
    ("GELU", 5, lambda x: x * (0.5 * (1.0 + math.erf(x / math.sqrt(2.0))))),
    ("GELU_TANH", 6, lambda x: x * (0.5 * (1.0 + math.tanh(
        math.sqrt(2.0 / math.pi) * (x + 0.044715 * (x * x * x)))))),
]


def rounded(f, raw):
    """round(256 * f(raw / 256)) to nearest, ties to even, with its distance
    from the nearest tie (LSB)."""
    v = f(raw / SCALE) * SCALE
    margin = abs(abs(v - math.floor(v)) - 0.5)
    return round(v), margin


def table(name, f):
    """m(t) for t = 0 .. 32768 (|x| of Q8.8, -32768 included), trimmed to the
    non-zero segment; checks the tie margin and the identity on every input."""
    m = []
    for t in range(0, 32769):
        r, margin = rounded(f, -t)
        if margin < MIN_TIE_MARGIN:
            sys.exit(f"{name}: x = {-t}/256 is {margin:.2e} LSB from a rounding tie")
        if r > 0 or -r >= 1 << ROM_W:
            sys.exit(f"{name}: f(-{t}/256) = {r} LSB does not fit the {ROM_W}-bit magnitude")
        m.append(-r)
    n = max(i for i, v in enumerate(m) if v) + 1
    for raw in range(1, 32768):                       # positive x: x + f(-x)
        r, margin = rounded(f, raw)
        if margin < MIN_TIE_MARGIN:
            sys.exit(f"{name}: x = {raw}/256 is {margin:.2e} LSB from a rounding tie")
        if r != raw - (m[raw] if raw < n else 0):
            sys.exit(f"{name}: f({raw}/256) = {r} LSB, the ROM identity gives "
                     f"{raw - (m[raw] if raw < n else 0)}")
    return m[:n]


def check_exact(tables):
    import mpmath
    mpmath.mp.dps = 40
    exact = {
        "SILU": lambda x: x / (1 + mpmath.exp(-x)),
        "GELU": lambda x: x * (1 + mpmath.erf(x / mpmath.sqrt(2))) / 2,
        "GELU_TANH": lambda x: x * (1 + mpmath.tanh(
            mpmath.sqrt(2 / mpmath.pi) * (x + mpmath.mpf("0.044715") * x ** 3))) / 2,
    }
    for name, m in tables.items():
        for t in range(0, 32769):
            r = int(mpmath.nint(exact[name](-mpmath.mpf(t) / SCALE) * SCALE))
            if -r != (m[t] if t < len(m) else 0):
                sys.exit(f"{name}: t = {t}: mpmath {-r}, the table {m[t] if t < len(m) else 0}")
        print(f"{name}: {len(m)} entries equal mpmath's", file=sys.stderr)


def render(tables):
    segs, base = [], 0
    for name, act, _ in FUNCS:
        segs.append((name, act, base, len(tables[name])))
        base += len(tables[name])
    if base > ROM_D:
        sys.exit(f"the tables need {base} entries, the ROM has {ROM_D}")
    rom = [0] * ROM_D
    for name, _, b, n in segs:
        rom[b:b + n] = tables[name]

    w = max(len(n) for n, *_ in segs)
    lines = []
    for name, act, b, n in segs:
        bv, nv = "12'd%d," % b, "12'd%d;" % n
        lines.append(f"  localparam logic [11:0] {name + '_BASE':<{w + 5}} = {bv:<9} "
                     f"{name + '_LEN':<{w + 4}} = {nv:<9} // act {act}")
    consts = "\n".join(lines)
    inits, line = [], []
    for a, v in enumerate(rom):
        if v:
            line.append(f"rom[{a:4d}] = 7'd{v:<2d};")
            if len(line) == 6:
                inits.append("    " + " ".join(line))
                line = []
    if line:
        inits.append("    " + " ".join(line))
    used = sum(n for *_, n in segs)
    return f"""// ---------------------------------------------------------------------------
// vo_act_rom — GENERATED by {GENERATOR}; do not edit.
//
// The SiLU / GELU / GELU tanh tables of vo_act: rom[BASE + t] = -round(256 *
// f(-t / 256)) for |x| = t LSB < LEN (round to nearest, ties to even), so
// f(x) = max(x, 0) - rom[BASE + |x|], and f(x) = max(x, 0) for |x| >= LEN.
// {used} of {ROM_D} entries used.  Two read ports, two cycles of latency (the
// block RAM and its output register): one BRAM36 (4K x 9) per two lanes.
// ---------------------------------------------------------------------------
package vo_act_tab;
  localparam int ROM_D = {ROM_D};
  localparam int ROM_W = {ROM_W};
{consts}
endpackage

module vo_act_rom
  import vo_act_tab::*;
(
  input  logic             clk,
  input  logic [11:0]      addr_a,
  input  logic [11:0]      addr_b,
  output logic [ROM_W-1:0] q_a,
  output logic [ROM_W-1:0] q_b
);
  (* rom_style = "block" *) logic [ROM_W-1:0] rom [ROM_D];
  logic [ROM_W-1:0] r_a, r_b;

  initial begin
    for (int i = 0; i < ROM_D; i++) rom[i] = '0;
{chr(10).join(inits)}
  end

  always_ff @(posedge clk) begin
    r_a <= rom[addr_a];
    r_b <= rom[addr_b];
    q_a <= r_a;                             // the block RAM's output register
    q_b <= r_b;
  end
endmodule
"""


def main():
    ap = argparse.ArgumentParser(description="Write / check rtl/vo_act_rom.sv (vo_act's SiLU / GELU tables).")
    ap.add_argument("--check", action="store_true", help="compare with rtl/vo_act_rom.sv")
    ap.add_argument("--exact", action="store_true", help="check the tables against mpmath")
    args = ap.parse_args()
    tables = {name: table(name, f) for name, _, f in FUNCS}
    if args.exact:
        check_exact(tables)
    text = render(tables)
    if args.check:
        have = open(OUT).read() if os.path.exists(OUT) else ""
        if have != text:
            print(f"{OUT} is stale: run {GENERATOR}", file=sys.stderr)
            return 1
        print(f"{OUT}: up to date ({', '.join(f'{n} {len(t)}' for n, t in tables.items())})")
        return 0
    with open(OUT, "w") as f:
        f.write(text)
    print(f"wrote {OUT} ({', '.join(f'{n} {len(t)}' for n, t in tables.items())})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
