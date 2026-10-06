#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# vl_unisim.py — a copy of Vivado's DSP48E2 unisim model that Verilator takes.
#
# The model treats an undriven (high-impedance) input as its reset value:
# `(X !== 1'bz) && ...` / `(X === 1'bz) || ...` on every input.  Verilator
# does not take tristate comparisons on top-level-style inputs, and every
# input of the DSPs in cv_mac_chain is driven, so the comparisons are
# replaced by their driven-value results (1'b1 / 1'b0).  Used by the
# ConvRtlDsp check (tb/verilator/dsp_check.*) only.
#
#   vl_unisim.py <Vivado data/verilog/src/unisims/DSP48E2.v> <out.v>
# ---------------------------------------------------------------------------
import re
import sys

src, out = sys.argv[1], sys.argv[2]
text = open(src).read()
text, n1 = re.subn(r"\(([A-Za-z_0-9]+(?:\[\d+\])?) !== 1'bz\)", "1'b1", text)
text, n2 = re.subn(r"\(([A-Za-z_0-9]+(?:\[\d+\])?) === 1'bz\)", "1'b0", text)
if n1 == 0 or n2 == 0:
    sys.exit(f"vl_unisim.py: no high-impedance input tests found in {src} — another model version?")
open(out, "w").write(text)
print(f"vl_unisim.py: {n1 + n2} input tests replaced -> {out}")
