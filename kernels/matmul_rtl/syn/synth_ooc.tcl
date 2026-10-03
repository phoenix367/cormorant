# ---------------------------------------------------------------------------
# synth_ooc.tcl — out-of-context synthesis, placement and routing of the
# MatmulKernel RTL on the KV260 part, for utilisation and timing reports.
#
#   vivado -mode batch -source syn/synth_ooc.tcl -tclargs <kernel dir> [period_ns] [part]
#
# Reports land in the current directory (make synth_matmul_rtl ->
# <build>/kernels/matmul_rtl/synth/).
# ---------------------------------------------------------------------------
set root   [file normalize [lindex $argv 0]]
set period [expr {[llength $argv] > 1 ? [lindex $argv 1] : 3.333}]
set part   [expr {[llength $argv] > 2 ? [lindex $argv 2] : "xck26-sfvc784-2LV-c"}]

source $root/syn/rtl_files.tcl
read_verilog -sv $rtl_files

synth_design -top MatmulKernel -part $part -mode out_of_context \
             -flatten_hierarchy rebuilt -directive PerformanceOptimized

create_clock -name ap_clk -period $period [get_ports ap_clk]

report_utilization -file utilization_synth.rpt
report_utilization -hierarchical -hierarchical_depth 3 -file utilization_synth_hier.rpt

opt_design
place_design -directive ExtraTimingOpt
phys_opt_design
route_design
phys_opt_design

report_utilization -file utilization.rpt
report_utilization -hierarchical -hierarchical_depth 3 -file utilization_hier.rpt
report_timing_summary -max_paths 20 -file timing.rpt
report_design_analysis -logic_level_distribution -file logic_levels.rpt

set wns [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]]
puts "RESULT period=${period}ns WNS=${wns}ns Fmax~[format %.1f [expr {1000.0 / ($period - $wns)}]]MHz"
