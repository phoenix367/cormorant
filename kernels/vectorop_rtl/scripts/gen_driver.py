#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# gen_driver.py — writes the C driver of the RTL VectorOPKernel and checks its
# register table against the RTL.
#
# The driver has the API of the one Vitis HLS generates for the HLS
# VectorOPKernel (xvectoropkernel.h: XVectoropkernel_Initialize / _Start /
# _IsDone / _Set_<arg> / _Get_<arg> / interrupt calls; Linux UIO and
# bare-metal back ends), so generated inference projects, the benchmarks and
# the calibration runner build against either IP unchanged.  The generator is
# the RTL MatmulKernel's (kernels/matmul_rtl/scripts/gen_driver.py) with this
# kernel's name and register table; rtl/vo_ctrl_s_axi.sv must agree with REGS
# (--check).  So did the HLS export's xvectoropkernel_hw.h (--hls-driver DIR,
# checked before the HLS kernel's synthesis was retired; VECTOROP_RTL_PLAN).
#
#   gen_driver.py --out DIR            write DIR/VectorOPKernel_v1_0/{src,data}/
#   gen_driver.py --check [--hls-driver DIR] [--json]
# ---------------------------------------------------------------------------
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "matmul_rtl", "scripts"))
import gen_driver as g  # noqa: E402  (the MatmulKernel generator, not this file)

g.KERNEL_DIR = os.path.dirname(HERE)
g.RTL_CTRL = os.path.join(g.KERNEL_DIR, "rtl", "vo_ctrl_s_axi.sv")
g.NAME = "VectorOPKernel"
g.PREFIX = "XVectoropkernel"
g.FILE = "xvectoropkernel"
g.MACRO = "XVECTOROPKERNEL"
g.GENERATOR = "kernels/vectorop_rtl/scripts/gen_driver.py"

# Arguments in register order: name, offset, bits, RTL localparams (lo[, hi]).
g.REGS = [
    ("a",     0x10, 64, ("A_A0", "A_A1")),
    ("b",     0x1C, 64, ("A_B0", "A_B1")),
    ("c",     0x28, 64, ("A_C0", "A_C1")),
    ("size",  0x34, 32, ("A_SIZE",)),
    ("op",    0x3C, 32, ("A_OP",)),
    ("outer", 0x44, 32, ("A_OUTER",)),
    ("a_inc", 0x4C, 32, ("A_AINC",)),
    ("b_inc", 0x54, 32, ("A_BINC",)),
    ("act",   0x5C, 32, ("A_ACT",)),
]

if __name__ == "__main__":
    sys.exit(g.main())
