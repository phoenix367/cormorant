#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# gen_driver.py — writes the C driver of the RTL ConvKernel and checks its
# register table against the RTL.
#
# The driver has the API of the one Vitis HLS generates for the HLS
# ConvKernel (xconvkernel.h: XConvkernel_Initialize / _Start / _IsDone /
# _Set_<arg> / _Get_<arg> / interrupt calls; Linux UIO and bare-metal back
# ends), so generated inference projects, the benchmarks and the calibration
# runner build against either IP unchanged.  The generator is the RTL
# MatmulKernel's (kernels/matmul_rtl/scripts/gen_driver.py) with this
# kernel's name and register table; rtl/cv_ctrl_s_axi.sv must agree with REGS
# (--check), and so must the HLS export's xconvkernel_hw.h (--hls-driver DIR).
#
#   gen_driver.py --out DIR            write DIR/ConvKernel_v1_0/{src,data}/
#   gen_driver.py --check [--hls-driver DIR] [--json]
# ---------------------------------------------------------------------------
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "matmul_rtl", "scripts"))
import gen_driver as g  # noqa: E402  (the MatmulKernel generator, not this file)

g.KERNEL_DIR = os.path.dirname(HERE)
g.RTL_CTRL = os.path.join(g.KERNEL_DIR, "rtl", "cv_ctrl_s_axi.sv")
g.NAME = "ConvKernel"
g.PREFIX = "XConvkernel"
g.FILE = "xconvkernel"
g.MACRO = "XCONVKERNEL"
g.GENERATOR = "kernels/conv_rtl/scripts/gen_driver.py"

# Arguments in register order: name, offset, bits, RTL localparams (lo[, hi]).
g.REGS = [
    ("x",            0x10, 64, ("A_X0", "A_X1")),
    ("weight",       0x1C, 64, ("A_WEIGHT0", "A_WEIGHT1")),
    ("bias",         0x28, 64, ("A_BIAS0", "A_BIAS1")),
    ("y",            0x34, 64, ("A_Y0", "A_Y1")),
    ("batch",        0x40, 32, ("A_BATCH",)),
    ("in_ch",        0x48, 32, ("A_IN_CH",)),
    ("in_h",         0x50, 32, ("A_IN_H",)),
    ("in_w",         0x58, 32, ("A_IN_W",)),
    ("out_ch",       0x60, 32, ("A_OUT_CH",)),
    ("out_h",        0x68, 32, ("A_OUT_H",)),
    ("out_w",        0x70, 32, ("A_OUT_W",)),
    ("kh",           0x78, 32, ("A_KH",)),
    ("kw",           0x80, 32, ("A_KW",)),
    ("stride_h",     0x88, 32, ("A_STRIDE_H",)),
    ("stride_w",     0x90, 32, ("A_STRIDE_W",)),
    ("dilation_h",   0x98, 32, ("A_DIL_H",)),
    ("dilation_w",   0xA0, 32, ("A_DIL_W",)),
    ("pad_top",      0xA8, 32, ("A_PAD_TOP",)),
    ("pad_left",     0xB0, 32, ("A_PAD_LEFT",)),
    ("has_bias",     0xB8, 32, ("A_HAS_BIAS",)),
    ("is_depthwise", 0xC0, 32, ("A_IS_DW",)),
]

if __name__ == "__main__":
    sys.exit(g.main())
