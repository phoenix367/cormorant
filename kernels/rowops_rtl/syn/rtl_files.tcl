# Copyright 2026 Ivan Gubochkin
# SPDX-License-Identifier: Apache-2.0

# RTL source list (compile order) shared by the Vivado scripts.
set rtl_files [list \
  $root/rtl/ro_pkg.sv \
  $root/rtl/ro_fifo.sv \
  $root/rtl/ro_rs.sv \
  $root/rtl/ro_ctrl_s_axi.sv \
  $root/rtl/ro_burstgen.sv \
  $root/rtl/ro_rd_port.sv \
  $root/rtl/ro_wr_port.sv \
  $root/rtl/ro_rd_arb.sv \
  $root/rtl/ro_prm_ram.sv \
  $root/rtl/ro_param.sv \
  $root/rtl/ro_sig_ram.sv \
  $root/rtl/ro_stream.sv \
  $root/rtl/ro_rsqrt_ram.sv \
  $root/rtl/ro_mul32.sv \
  $root/rtl/ro_norm.sv \
  $root/rtl/ro_rope.sv \
  $root/rtl/ro_xpose.sv \
  $root/rtl/ro_fold.sv \
  $root/rtl/ro_core.sv \
  $root/rtl/RowOpsKernel.v \
]
