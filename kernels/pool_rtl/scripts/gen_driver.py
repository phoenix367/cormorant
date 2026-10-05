#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# gen_driver.py — writes the C driver of the RTL PoolingKernel and checks its
# register table against the RTL.
#
# The driver has the API of the one Vitis HLS generates for the HLS
# PoolingKernel (xpoolingkernel.h: XPoolingkernel_Initialize / _Start /
# _IsDone / _Set_<arg> / _Get_<arg> / interrupt calls; Linux UIO and
# bare-metal back ends), so generated inference projects, the benchmarks and
# the calibration runner build against either IP unchanged.  The generator is
# the RTL MatmulKernel's (kernels/matmul_rtl/scripts/gen_driver.py) with this
# kernel's name and register table; rtl/pl_ctrl_s_axi.sv must agree with REGS
# (--check).  So did the HLS export's xpoolingkernel_hw.h (--hls-driver DIR,
# checked before the HLS kernel's synthesis was retired; VECTOROP_RTL_PLAN).
#
#   gen_driver.py --out DIR            write DIR/PoolingKernel_v1_0/{src,data}/
#   gen_driver.py --check [--hls-driver DIR] [--json]
# ---------------------------------------------------------------------------
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "matmul_rtl", "scripts"))
import gen_driver as g  # noqa: E402  (the MatmulKernel generator, not this file)

g.KERNEL_DIR = os.path.dirname(HERE)
g.RTL_CTRL = os.path.join(g.KERNEL_DIR, "rtl", "pl_ctrl_s_axi.sv")
g.NAME = "PoolingKernel"
g.PREFIX = "XPoolingkernel"
g.FILE = "xpoolingkernel"
g.MACRO = "XPOOLINGKERNEL"
g.GENERATOR = "kernels/pool_rtl/scripts/gen_driver.py"

# Arguments in register order: name, offset, bits, RTL localparams (lo[, hi]).
g.REGS = [
    ("x",                 0x10, 64, ("A_X0", "A_X1")),
    ("y",                 0x1C, 64, ("A_Y0", "A_Y1")),
    ("batch",             0x28, 32, ("A_BATCH",)),
    ("channels",          0x30, 32, ("A_CHANNELS",)),
    ("in_h",              0x38, 32, ("A_IN_H",)),
    ("in_w",              0x40, 32, ("A_IN_W",)),
    ("out_h",             0x48, 32, ("A_OUT_H",)),
    ("out_w",             0x50, 32, ("A_OUT_W",)),
    ("pool_h",            0x58, 32, ("A_POOL_H",)),
    ("pool_w",            0x60, 32, ("A_POOL_W",)),
    ("stride_h",          0x68, 32, ("A_STRIDE_H",)),
    ("stride_w",          0x70, 32, ("A_STRIDE_W",)),
    ("pad_top",           0x78, 32, ("A_PAD_TOP",)),
    ("pad_left",          0x80, 32, ("A_PAD_LEFT",)),
    ("dil_h",             0x88, 32, ("A_DIL_H",)),
    ("dil_w",             0x90, 32, ("A_DIL_W",)),
    ("pool_type",         0x98, 32, ("A_POOL_TYPE",)),
    ("lp_order",          0xA0, 32, ("A_LP_ORDER",)),
    ("count_include_pad", 0xA8, 32, ("A_CIP",)),
]

if __name__ == "__main__":
    sys.exit(g.main())
