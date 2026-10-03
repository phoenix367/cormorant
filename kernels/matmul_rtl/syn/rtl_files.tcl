# RTL source list (compile order) shared by the Vivado scripts.
set rtl_files [list \
  $root/rtl/mm_pkg.sv \
  $root/rtl/mm_fifo.sv \
  $root/rtl/mm_gearbox.sv \
  $root/rtl/mm_axi_rd.sv \
  $root/rtl/mm_axi_wr.sv \
  $root/rtl/mm_packer.sv \
  $root/rtl/mm_ctrl_s_axi.sv \
  $root/rtl/mm_walker.sv \
  $root/rtl/mm_rungen.sv \
  $root/rtl/mm_abuf.sv \
  $root/rtl/mm_awr.sv \
  $root/rtl/mm_xpf.sv \
  $root/rtl/mm_mac.sv \
  $root/rtl/mm_lane.sv \
  $root/rtl/mm_drain.sv \
  $root/rtl/mm_core.sv \
  $root/rtl/MatmulKernel.v \
]
