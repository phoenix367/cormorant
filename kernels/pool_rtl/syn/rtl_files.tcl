# RTL source list (compile order) shared by the Vivado scripts.
set rtl_files [list \
  $root/rtl/pl_pkg.sv \
  $root/rtl/pl_fifo.sv \
  $root/rtl/pl_rs.sv \
  $root/rtl/pl_lutram.sv \
  $root/rtl/pl_ctrl_s_axi.sv \
  $root/rtl/pl_loader.sv \
  $root/rtl/pl_emit.sv \
  $root/rtl/pl_reduce.sv \
  $root/rtl/pl_writer.sv \
  $root/rtl/pl_core.sv \
  $root/rtl/PoolingKernel.v \
]
