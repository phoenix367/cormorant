# Set AxPROT = 010 (non-secure, unprivileged, data) on every kernel m_axi
# master of the cormorant_hw_128 block design.  The HLS default "000" is a
# Secure access, which the SMMU does not match against the non-secure stream
# table Linux programs: the PL traffic would bypass translation.
#
#   vivado -mode batch -nojournal -nolog -source set_prot_ns.tcl \
#       -tclargs <ip repo: build/kernels> <cormorant_hw_128.xpr>
#
# then build the bitstream as usual (hw/cormorant_hw_128: build.sh all).
set ip_repo [lindex $argv 0]
set xpr     [lindex $argv 1]
open_project $xpr
set_property ip_repo_paths [list $ip_repo] [current_project]
update_ip_catalog -rebuild
set locked [get_ips -quiet -filter {IS_LOCKED == 1}]
if {[llength $locked] > 0} { upgrade_ip $locked }
open_bd_design [get_files design_cormorant.bd]
foreach {cell ports} {VectorOPKernel_0 {0 1 2} MatmulKernel_0 {0 1 2} ConvKernel_0 {0 1 2 3} PoolingKernel_0 {0 1}} {
    set c [get_bd_cells $cell]
    foreach p $ports {
        set_property CONFIG.C_M_AXI_GMEM${p}_PROT_VALUE {"010"} $c
        puts "=== $cell gmem$p PROT_VALUE = [get_property CONFIG.C_M_AXI_GMEM${p}_PROT_VALUE $c]"
    }
}
validate_bd_design
save_bd_design
close_project
