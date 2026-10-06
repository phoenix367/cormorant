# RTL source list (compile order) shared by the Vivado scripts.
set rtl_files [list \
  $root/rtl/cv_pkg.sv \
  $root/rtl/cv_fifo.sv \
  $root/rtl/cv_lutram.sv \
  $root/rtl/cv_div.sv \
  $root/rtl/cv_ctrl_s_axi.sv \
  $root/rtl/cv_mac_chain.sv \
  $root/rtl/cv_engine.sv \
  $root/rtl/cv_xload.sv \
  $root/rtl/cv_patch.sv \
  $root/rtl/cv_wload.sv \
  $root/rtl/cv_bias.sv \
  $root/rtl/cv_drain.sv \
  $root/rtl/cv_ywriter.sv \
  $root/rtl/cv_core.sv \
  $root/rtl/ConvKernel.v \
]
