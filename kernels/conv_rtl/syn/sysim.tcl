# ---------------------------------------------------------------------------
# sysim.tcl — system-level drop-in check: runs the test stand's
# ConvKernel block design (Zynq PS VIP + AXI interconnect + DDR model +
# conv_tb.sv) with the packaged RTL IP in place of the HLS export.
#
# Operates on a COPY of hw/cormorant_test_stand/kernels/conv_test
# (make sysim_conv_rtl creates it under build/kernels/conv_rtl/sysim/),
# never on the original.
#
#   vivado -mode batch -source syn/sysim.tcl -tclargs <xpr> <ip_repo> <data_dir> <report>
# ---------------------------------------------------------------------------
set xpr      [file normalize [lindex $argv 0]]
set ip_repo  [file normalize [lindex $argv 1]]
set data_dir [file normalize [lindex $argv 2]]
set report   [file normalize [lindex $argv 3]]

open_project $xpr
set_property ip_repo_paths [list $ip_repo] [current_project]
update_ip_catalog -rebuild

set bd_file [lindex [get_files -of_objects [get_filesets sources_1] \
                         -filter {FILE_TYPE == "Block Designs"}] 0]
open_bd_design $bd_file

# Swap the HLS core for the RTL one (same VLNV; all four ports 128 bits).
set mm [get_bd_cells -hierarchical -filter {VLNV =~ "xilinx.com:hls:ConvKernel:*"}]
puts "\[sysim\] ConvKernel cell: $mm"
report_ip_status
# every locked IP, as the test stand's own flow does (ts_prepare_bd): the
# ConvKernel is locked by the new IP, others by a project saved elsewhere
upgrade_ip [get_ips -all -quiet -filter {IS_LOCKED == 1}]
# an interconnect's crossbar can stay locked in the session that upgraded the
# kernel; a new session reads the saved design and upgrades it (ts_prepare_bd)
if {[llength [get_ips -all -quiet -filter {IS_LOCKED == 1}]] > 0} {
  puts "\[sysim\] still locked after upgrade_ip: reopening the project"
  save_bd_design
  close_project
  open_project $xpr
  open_bd_design [lindex [get_files -of_objects [get_filesets sources_1] \
                              -filter {FILE_TYPE == "Block Designs"}] 0]
  set again [get_ips -all -quiet -filter {IS_LOCKED == 1}]
  if {[llength $again] > 0} { upgrade_ip $again }
}
set still [get_ips -all -quiet -filter {IS_LOCKED == 1}]
if {[llength $still] > 0} { error "sysim: IPs still locked after upgrade_ip and a reopen: $still" }
set bd_file [lindex [get_files -of_objects [get_filesets sources_1] \
                         -filter {FILE_TYPE == "Block Designs"}] 0]
# the upgrade invalidates cell handles: look the cell up again
set mm [get_bd_cells -hierarchical -filter {VLNV =~ "xilinx.com:hls:ConvKernel:*"}]
foreach p {C_M_AXI_GMEM0_DATA_WIDTH C_M_AXI_GMEM1_DATA_WIDTH C_M_AXI_GMEM2_DATA_WIDTH C_M_AXI_GMEM3_DATA_WIDTH} {
  puts "\[sysim\] $p = [get_property CONFIG.$p $mm]"
}
foreach p {C_M_AXI_GMEM0_DATA_WIDTH C_M_AXI_GMEM1_DATA_WIDTH C_M_AXI_GMEM2_DATA_WIDTH C_M_AXI_GMEM3_DATA_WIDTH} {
  if {[get_property CONFIG.$p $mm] != 128} { error "sysim: $p is not 128 bits after the upgrade" }
}
validate_bd_design
save_bd_design

generate_target all $bd_file
set wrapper [make_wrapper -files $bd_file -top]
if {[llength [get_files -quiet $wrapper]] == 0} { add_files -norecurse $wrapper }

set sim_set [get_filesets sim_1]
set_property top conv_tb $sim_set
set_property -name {xsim.simulate.xsim.more_options} \
    -value "-testplusarg DATA_DIR=$data_dir -testplusarg REPORT=$report" -objects $sim_set
set batch_tcl [file normalize [file join [file dirname $xpr] sysim_batch.tcl]]
set fh [open $batch_tcl w]; puts $fh "run 0us"; close $fh
set_property -name {xsim.simulate.custom_tcl} -value $batch_tcl -objects $sim_set
set_property -name {xsim.simulate.log_all_signals} -value false -objects $sim_set

launch_simulation -mode behavioral -simset $sim_set
run -all
puts "\[sysim\] DONE report=$report"
close_sim
close_project
