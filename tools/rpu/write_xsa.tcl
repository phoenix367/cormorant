# Copyright 2026 Ivan Gubochkin
# SPDX-License-Identifier: Apache-2.0

# write_xsa.tcl — export the hardware platform (XSA) of the implemented block design
# for the Cortex-R5 BSP (doc/plans/RPU_PLAN.md track B1):
#
#   vivado -mode batch -nolog -nojournal -source tools/rpu/write_xsa.tcl \
#          -tclargs hw/cormorant_hw_128/cormorant_hw_128.xpr <out>/cormorant.xsa
#
# then tools/rpu/make_r5_bsp.py builds the standalone BSP from it.
if {[llength $argv] != 2} {
    puts "usage: write_xsa.tcl <project.xpr> <out.xsa>"
    exit 1
}
open_project [lindex $argv 0]
write_hw_platform -fixed -include_bit -force [lindex $argv 1]
puts "XSA_DONE"
