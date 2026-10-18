# Copyright 2026 Ivan Gubochkin
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# package_ip.tcl — packages the RTL RowOpsKernel as a Vivado IP for the
# block design (doc/kernels/ROWOPS_RTL_KERNEL.md §1.1):
#
#   VLNV            xilinx.com:hls:RowOpsKernel:1.0
#   interfaces      s_axi_ctrl (64 KiB register block "Reg"), m_axi_gmem0
#                   (address space Data_m_axi_gmem0, 64-bit; read + write,
#                   16 outstanding bursts per direction, 64-beat bursts),
#                   ap_clk, ap_rst_n (active low), interrupt (level high)
#   drivers         RowOpsKernel_v1_0, written by scripts/gen_driver.py
#
#   vivado -mode batch -source syn/package_ip.tcl \
#          -tclargs <kernel dir> <driver dir> [part]
#
# <driver dir> holds RowOpsKernel_v1_0/ (gen_driver.py --out <driver dir>).
# Output: ./RowOpsKernel_ip (IP repository root) and ./RowOpsKernel_ip.zip.
# ---------------------------------------------------------------------------
set root    [file normalize [lindex $argv 0]]
set drvroot [file normalize [lindex $argv 1]]
set part    [expr {[llength $argv] > 2 ? [lindex $argv 2] : "xck26-sfvc784-2LV-c"}]
set ipdir   [file normalize ./RowOpsKernel_ip]

file delete -force $ipdir ./proj ./RowOpsKernel_ip.zip
source $root/syn/rtl_files.tcl

create_project -force pkg ./proj -part $part
add_files -norecurse $rtl_files
set_property file_type SystemVerilog [get_files -of_objects [get_filesets sources_1] *.sv]
set_property top RowOpsKernel [current_fileset]
update_compile_order -fileset sources_1

ipx::package_project -root_dir $ipdir -vendor xilinx.com -library hls \
    -taxonomy /VITIS_HLS_IP -import_files -set_current true
set core [ipx::current_core]
set_property name         RowOpsKernel $core
set_property version      1.0 $core
set_property display_name RowOpsKernel $core
set_property description  "Row-ops unit (SystemVerilog RTL): the transformer graphs' row ops —\
 norms, residual adds, SiLU*up, RoPE, dequant, copies and block transposes — in fixed point;\
 128-bit gmem0 (read + write)" $core
set_property supported_families {zynquplus Production} $core
set_property core_revision 1 $core

# Interface widths are fixed by the RTL: expose the HLS parameter names, read only.
foreach p [ipx::get_user_parameters -of_objects $core] {
  set_property enablement_value false $p
}

# Like the HLS exports: *USER ports exist in the HDL but are hidden unless
# C_M_AXI_GMEM0_ENABLE_USER_PORTS is set (inputs then read 0); ID ports stay.
foreach g {GMEM0} {
  set gl [string tolower $g]
  foreach {pn def} [list ENABLE_USER_PORTS false ENABLE_ID_PORTS true] {
    set up [ipx::add_user_parameter C_M_AXI_${g}_$pn $core]
    set_property value_resolve_type user $up
    set_property value_format bool $up
    set_property value $def $up
    set_property enablement_value false $up
  }
  foreach ch {AWUSER WUSER ARUSER RUSER BUSER} {
    set port [ipx::get_ports m_axi_${gl}_$ch -of_objects $core]
    set_property enablement_dependency "\$C_M_AXI_${g}_ENABLE_USER_PORTS = 1" $port
    if {[get_property direction $port] eq "in"} { set_property driver_value 0 $port }
  }
}

# Clock / reset / interrupt ----------------------------------------------------------
foreach bi {s_axi_ctrl m_axi_gmem0} {
  ipx::associate_bus_interfaces -busif $bi -clock ap_clk $core
}
# m_axi properties (ro_pkg RD_BURST / RD_OUTS / WR_BURST / WR_OUTS): the block
# design sizes the crossbar slot's acceptance from the outstanding counts.
set bi [ipx::get_bus_interfaces m_axi_gmem0 -of_objects $core]
foreach {n v} [list NUM_READ_OUTSTANDING 16 NUM_WRITE_OUTSTANDING 16 \
                    MAX_READ_BURST_LENGTH 64 MAX_WRITE_BURST_LENGTH 64 MAX_BURST_LENGTH 256 \
                    PROTOCOL AXI4 READ_WRITE_MODE READ_WRITE HAS_BURST 0 \
                    SUPPORTS_NARROW_BURST 0 ADDR_WIDTH 64] {
  set bp [ipx::get_bus_parameters $n -of_objects $bi]
  if {$bp eq ""} { set bp [ipx::add_bus_parameter $n $bi] }
  set_property value $v $bp
}
set rst [ipx::get_bus_interfaces ap_rst_n -of_objects $core]
set pol [ipx::get_bus_parameters POLARITY -of_objects $rst]
if {$pol eq ""} { set pol [ipx::add_bus_parameter POLARITY $rst] }
set_property value ACTIVE_LOW $pol
set irq [ipx::get_bus_interfaces interrupt -of_objects $core]
if {$irq ne ""} {
  set sens [ipx::get_bus_parameters SENSITIVITY -of_objects $irq]
  if {$sens eq ""} { set sens [ipx::add_bus_parameter SENSITIVITY $irq] }
  set_property value LEVEL_HIGH $sens
}

# Register block --------------------------------------------------------------------------
foreach mm [ipx::get_memory_maps -of_objects $core] { ipx::remove_memory_map [get_property name $mm] $core }
set mm [ipx::add_memory_map s_axi_ctrl $core]
set_property slave_memory_map_ref s_axi_ctrl [ipx::get_bus_interfaces s_axi_ctrl -of_objects $core]
set blk [ipx::add_address_block Reg $mm]
set_property range 65536 $blk
set_property width 32 $blk
set_property usage register $blk
set_property access read-write $blk

# Master address space --------------------------------------------------------------------
foreach as [ipx::get_address_spaces -of_objects $core] { ipx::remove_address_space [get_property name $as] $core }
set as [ipx::add_address_space Data_m_axi_gmem0 $core]
set_property range 16E $as
set_property width 64 $as
set_property master_address_space_ref Data_m_axi_gmem0 \
    [ipx::get_bus_interfaces m_axi_gmem0 -of_objects $core]

# Software driver ----------------------------------------------------------------------------
set drv $drvroot/RowOpsKernel_v1_0
if {![file isdirectory $drv]} {
  error "no driver at $drv (run scripts/gen_driver.py --out $drvroot)"
}
file mkdir $ipdir/drivers
file copy -force $drv $ipdir/drivers/
set fg [ipx::add_file_group -type software_driver {} $core]
foreach f [lsort [glob -directory $ipdir/drivers/RowOpsKernel_v1_0 -tails data/* src/*]] {
  set rel drivers/RowOpsKernel_v1_0/$f
  set fo [ipx::add_file $rel $fg]
  switch -glob $f {
    *.mdd  { set_property type mdd $fo }
    *.tcl  { set_property type tclSource $fo }
    *.c    { set_property type cSource $fo }
    *.h    { set_property type cSource $fo }
    default { set_property type unknown $fo }
  }
}
puts "INFO: packaged the driver from $drv"

ipx::create_xgui_files $core
ipx::update_checksums $core
ipx::check_integrity $core
ipx::save_core $core
ipx::archive_core ./RowOpsKernel_ip.zip $core
puts "RESULT packaged [get_property vlnv $core] -> $ipdir"
