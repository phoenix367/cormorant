#!/usr/bin/env python3
# Copyright 2026 Ivan Gubochkin
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# gen_driver.py — writes the C driver of the RTL RowOpsKernel and checks its
# register table against the RTL.
#
# The driver has the API the Vitis HLS generators give every kernel
# (xrowopskernel.h: XRowopskernel_Initialize / _Start / _IsDone / _Set_<arg> /
# _Get_<arg> / interrupt calls; Linux UIO and bare-metal back ends), so the
# generated inference projects and the benchmarks drive it like the other
# four.  The generator is the RTL MatmulKernel's
# (kernels/matmul_rtl/scripts/gen_driver.py) with this kernel's name and
# register table (doc/kernels/ROWOPS_RTL_KERNEL.md §1.2); rtl/ro_ctrl_s_axi.sv
# must agree with REGS (--check).
#
#   gen_driver.py --out DIR            write DIR/RowOpsKernel_v1_0/{src,data}/
#   gen_driver.py --check [--json]
# ---------------------------------------------------------------------------
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "matmul_rtl", "scripts"))
import gen_driver as g  # noqa: E402  (the MatmulKernel generator, not this file)

g.KERNEL_DIR = os.path.dirname(HERE)
g.RTL_CTRL = os.path.join(g.KERNEL_DIR, "rtl", "ro_ctrl_s_axi.sv")
g.NAME = "RowOpsKernel"
g.PREFIX = "XRowopskernel"
g.FILE = "xrowopskernel"
g.MACRO = "XROWOPSKERNEL"
g.GENERATOR = "kernels/rowops_rtl/scripts/gen_driver.py"

# Arguments in register order: name, offset, bits, RTL localparams (lo[, hi]).
g.REGS = [
    ("op",     0x10, 32, ("A_OP",)),
    ("n",      0x14, 32, ("A_N",)),
    ("rows",   0x18, 32, ("A_ROWS",)),
    ("blocks", 0x1C, 32, ("A_BLOCKS",)),
    ("a",      0x20, 64, ("A_A0", "A_A1")),
    ("a_s1",   0x28, 32, ("A_AS1",)),
    ("a_s2",   0x2C, 32, ("A_AS2",)),
    ("b",      0x30, 64, ("A_B0", "A_B1")),
    ("b_s1",   0x38, 32, ("A_BS1",)),
    ("b_s2",   0x3C, 32, ("A_BS2",)),
    ("c",      0x40, 64, ("A_C0", "A_C1")),
    ("c_s1",   0x48, 32, ("A_CS1",)),
    ("c_s2",   0x4C, 32, ("A_CS2",)),
    ("p",      0x50, 64, ("A_P0", "A_P1")),
    ("eps",    0x58, 64, ("A_E0", "A_E1")),
    ("aux",    0x60, 32, ("A_AUX",)),
    ("bq_sh",  0x64, 32, ("A_BQSH",)),
    ("win_lo", 0x68, 32, ("A_WLO",)),
    ("win_hi", 0x6C, 32, ("A_WHI",)),
    # descriptor chains (doc/kernels/ROWOPS_RTL_KERNEL.md §1.6): the list's base, its job
    # count (0: the single job of the registers above)
    ("chain",   0x70, 64, ("A_CH0", "A_CH1")),
    ("chain_n", 0x78, 32, ("A_CHN",)),
]

if __name__ == "__main__":
    sys.exit(g.main())
