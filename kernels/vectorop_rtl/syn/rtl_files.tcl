# RTL source list (compile order) shared by the Vivado scripts.
set rtl_files [list \
  $root/rtl/vo_pkg.sv \
  $root/rtl/vo_fifo.sv \
  $root/rtl/vo_ctrl_s_axi.sv \
  $root/rtl/vo_burstgen.sv \
  $root/rtl/vo_rd_port.sv \
  $root/rtl/vo_wr_port.sv \
  $root/rtl/vo_div.sv \
  $root/rtl/vo_compute.sv \
  $root/rtl/vo_core.sv \
  $root/rtl/VectorOPKernel.v \
]
