// ---------------------------------------------------------------------------
// cv_core — the RTL ConvKernel: the HLS kernel's ports, registers and
// results (VLNV xilinx.com:hls:ConvKernel:1.0; ConvKernel.v is the
// Verilog-2001 IP top level around it).
//
//   ctrl ─► config ─► sweep sequencer ─┬► x loader (gmem0) ─► column FIFO ─► patch producer ─┐
//                                      ├► weight loader (gmem1) ─────────────────────────────┤
//                                      ├► engine: weight cache, MAC grid, accumulators ◄──────┘ ◄─ bias (gmem2)
//                                      └► (fill / issue)        └► drain ─► y writer (gmem3)
// ---------------------------------------------------------------------------
module cv_core
  import cv_pkg::*;
#(
  parameter int C_S_AXI_CTRL_DATA_WIDTH    = 32,
  parameter int C_S_AXI_CTRL_ADDR_WIDTH    = 8,
  parameter int C_M_AXI_GMEM0_ID_WIDTH    = 1,
  parameter int C_M_AXI_GMEM0_ADDR_WIDTH  = 64,
  parameter int C_M_AXI_GMEM0_DATA_WIDTH  = 128,
  parameter int C_M_AXI_GMEM0_AWUSER_WIDTH= 1,
  parameter int C_M_AXI_GMEM0_ARUSER_WIDTH= 1,
  parameter int C_M_AXI_GMEM0_WUSER_WIDTH = 1,
  parameter int C_M_AXI_GMEM0_RUSER_WIDTH = 1,
  parameter int C_M_AXI_GMEM0_BUSER_WIDTH = 1,
  parameter int C_M_AXI_GMEM0_USER_VALUE  = 0,
  parameter int C_M_AXI_GMEM0_PROT_VALUE  = 0,
  parameter int C_M_AXI_GMEM0_CACHE_VALUE = 3,
  parameter int C_M_AXI_GMEM1_ID_WIDTH    = 1,
  parameter int C_M_AXI_GMEM1_ADDR_WIDTH  = 64,
  parameter int C_M_AXI_GMEM1_DATA_WIDTH  = 128,
  parameter int C_M_AXI_GMEM1_AWUSER_WIDTH= 1,
  parameter int C_M_AXI_GMEM1_ARUSER_WIDTH= 1,
  parameter int C_M_AXI_GMEM1_WUSER_WIDTH = 1,
  parameter int C_M_AXI_GMEM1_RUSER_WIDTH = 1,
  parameter int C_M_AXI_GMEM1_BUSER_WIDTH = 1,
  parameter int C_M_AXI_GMEM1_USER_VALUE  = 0,
  parameter int C_M_AXI_GMEM1_PROT_VALUE  = 0,
  parameter int C_M_AXI_GMEM1_CACHE_VALUE = 3,
  parameter int C_M_AXI_GMEM2_ID_WIDTH    = 1,
  parameter int C_M_AXI_GMEM2_ADDR_WIDTH  = 64,
  parameter int C_M_AXI_GMEM2_DATA_WIDTH  = 128,
  parameter int C_M_AXI_GMEM2_AWUSER_WIDTH= 1,
  parameter int C_M_AXI_GMEM2_ARUSER_WIDTH= 1,
  parameter int C_M_AXI_GMEM2_WUSER_WIDTH = 1,
  parameter int C_M_AXI_GMEM2_RUSER_WIDTH = 1,
  parameter int C_M_AXI_GMEM2_BUSER_WIDTH = 1,
  parameter int C_M_AXI_GMEM2_USER_VALUE  = 0,
  parameter int C_M_AXI_GMEM2_PROT_VALUE  = 0,
  parameter int C_M_AXI_GMEM2_CACHE_VALUE = 3,
  parameter int C_M_AXI_GMEM3_ID_WIDTH    = 1,
  parameter int C_M_AXI_GMEM3_ADDR_WIDTH  = 64,
  parameter int C_M_AXI_GMEM3_DATA_WIDTH  = 128,
  parameter int C_M_AXI_GMEM3_AWUSER_WIDTH= 1,
  parameter int C_M_AXI_GMEM3_ARUSER_WIDTH= 1,
  parameter int C_M_AXI_GMEM3_WUSER_WIDTH = 1,
  parameter int C_M_AXI_GMEM3_RUSER_WIDTH = 1,
  parameter int C_M_AXI_GMEM3_BUSER_WIDTH = 1,
  parameter int C_M_AXI_GMEM3_USER_VALUE  = 0,
  parameter int C_M_AXI_GMEM3_PROT_VALUE  = 0,
  parameter int C_M_AXI_GMEM3_CACHE_VALUE = 3
) (
  input  logic ap_clk,
  input  logic ap_rst_n,

  // gmem0: x (read only)
  output logic                                  m_axi_gmem0_AWVALID,
  input  logic                                  m_axi_gmem0_AWREADY,
  output logic [C_M_AXI_GMEM0_ADDR_WIDTH-1:0]   m_axi_gmem0_AWADDR,
  output logic [C_M_AXI_GMEM0_ID_WIDTH-1:0]     m_axi_gmem0_AWID,
  output logic [7:0]                            m_axi_gmem0_AWLEN,
  output logic [2:0]                            m_axi_gmem0_AWSIZE,
  output logic [1:0]                            m_axi_gmem0_AWBURST,
  output logic [1:0]                            m_axi_gmem0_AWLOCK,
  output logic [3:0]                            m_axi_gmem0_AWCACHE,
  output logic [2:0]                            m_axi_gmem0_AWPROT,
  output logic [3:0]                            m_axi_gmem0_AWQOS,
  output logic [3:0]                            m_axi_gmem0_AWREGION,
  output logic [C_M_AXI_GMEM0_AWUSER_WIDTH-1:0] m_axi_gmem0_AWUSER,
  output logic                                  m_axi_gmem0_WVALID,
  input  logic                                  m_axi_gmem0_WREADY,
  output logic [C_M_AXI_GMEM0_DATA_WIDTH-1:0]   m_axi_gmem0_WDATA,
  output logic [C_M_AXI_GMEM0_DATA_WIDTH/8-1:0] m_axi_gmem0_WSTRB,
  output logic                                  m_axi_gmem0_WLAST,
  output logic [C_M_AXI_GMEM0_ID_WIDTH-1:0]     m_axi_gmem0_WID,
  output logic [C_M_AXI_GMEM0_WUSER_WIDTH-1:0]  m_axi_gmem0_WUSER,
  output logic                                  m_axi_gmem0_ARVALID,
  input  logic                                  m_axi_gmem0_ARREADY,
  output logic [C_M_AXI_GMEM0_ADDR_WIDTH-1:0]   m_axi_gmem0_ARADDR,
  output logic [C_M_AXI_GMEM0_ID_WIDTH-1:0]     m_axi_gmem0_ARID,
  output logic [7:0]                            m_axi_gmem0_ARLEN,
  output logic [2:0]                            m_axi_gmem0_ARSIZE,
  output logic [1:0]                            m_axi_gmem0_ARBURST,
  output logic [1:0]                            m_axi_gmem0_ARLOCK,
  output logic [3:0]                            m_axi_gmem0_ARCACHE,
  output logic [2:0]                            m_axi_gmem0_ARPROT,
  output logic [3:0]                            m_axi_gmem0_ARQOS,
  output logic [3:0]                            m_axi_gmem0_ARREGION,
  output logic [C_M_AXI_GMEM0_ARUSER_WIDTH-1:0] m_axi_gmem0_ARUSER,
  input  logic                                  m_axi_gmem0_RVALID,
  output logic                                  m_axi_gmem0_RREADY,
  input  logic [C_M_AXI_GMEM0_DATA_WIDTH-1:0]   m_axi_gmem0_RDATA,
  input  logic                                  m_axi_gmem0_RLAST,
  input  logic [C_M_AXI_GMEM0_ID_WIDTH-1:0]     m_axi_gmem0_RID,
  input  logic [C_M_AXI_GMEM0_RUSER_WIDTH-1:0]  m_axi_gmem0_RUSER,
  input  logic [1:0]                            m_axi_gmem0_RRESP,
  input  logic                                  m_axi_gmem0_BVALID,
  output logic                                  m_axi_gmem0_BREADY,
  input  logic [1:0]                            m_axi_gmem0_BRESP,
  input  logic [C_M_AXI_GMEM0_ID_WIDTH-1:0]     m_axi_gmem0_BID,
  input  logic [C_M_AXI_GMEM0_BUSER_WIDTH-1:0]  m_axi_gmem0_BUSER,

  // gmem1: weight (read only)
  output logic                                  m_axi_gmem1_AWVALID,
  input  logic                                  m_axi_gmem1_AWREADY,
  output logic [C_M_AXI_GMEM1_ADDR_WIDTH-1:0]   m_axi_gmem1_AWADDR,
  output logic [C_M_AXI_GMEM1_ID_WIDTH-1:0]     m_axi_gmem1_AWID,
  output logic [7:0]                            m_axi_gmem1_AWLEN,
  output logic [2:0]                            m_axi_gmem1_AWSIZE,
  output logic [1:0]                            m_axi_gmem1_AWBURST,
  output logic [1:0]                            m_axi_gmem1_AWLOCK,
  output logic [3:0]                            m_axi_gmem1_AWCACHE,
  output logic [2:0]                            m_axi_gmem1_AWPROT,
  output logic [3:0]                            m_axi_gmem1_AWQOS,
  output logic [3:0]                            m_axi_gmem1_AWREGION,
  output logic [C_M_AXI_GMEM1_AWUSER_WIDTH-1:0] m_axi_gmem1_AWUSER,
  output logic                                  m_axi_gmem1_WVALID,
  input  logic                                  m_axi_gmem1_WREADY,
  output logic [C_M_AXI_GMEM1_DATA_WIDTH-1:0]   m_axi_gmem1_WDATA,
  output logic [C_M_AXI_GMEM1_DATA_WIDTH/8-1:0] m_axi_gmem1_WSTRB,
  output logic                                  m_axi_gmem1_WLAST,
  output logic [C_M_AXI_GMEM1_ID_WIDTH-1:0]     m_axi_gmem1_WID,
  output logic [C_M_AXI_GMEM1_WUSER_WIDTH-1:0]  m_axi_gmem1_WUSER,
  output logic                                  m_axi_gmem1_ARVALID,
  input  logic                                  m_axi_gmem1_ARREADY,
  output logic [C_M_AXI_GMEM1_ADDR_WIDTH-1:0]   m_axi_gmem1_ARADDR,
  output logic [C_M_AXI_GMEM1_ID_WIDTH-1:0]     m_axi_gmem1_ARID,
  output logic [7:0]                            m_axi_gmem1_ARLEN,
  output logic [2:0]                            m_axi_gmem1_ARSIZE,
  output logic [1:0]                            m_axi_gmem1_ARBURST,
  output logic [1:0]                            m_axi_gmem1_ARLOCK,
  output logic [3:0]                            m_axi_gmem1_ARCACHE,
  output logic [2:0]                            m_axi_gmem1_ARPROT,
  output logic [3:0]                            m_axi_gmem1_ARQOS,
  output logic [3:0]                            m_axi_gmem1_ARREGION,
  output logic [C_M_AXI_GMEM1_ARUSER_WIDTH-1:0] m_axi_gmem1_ARUSER,
  input  logic                                  m_axi_gmem1_RVALID,
  output logic                                  m_axi_gmem1_RREADY,
  input  logic [C_M_AXI_GMEM1_DATA_WIDTH-1:0]   m_axi_gmem1_RDATA,
  input  logic                                  m_axi_gmem1_RLAST,
  input  logic [C_M_AXI_GMEM1_ID_WIDTH-1:0]     m_axi_gmem1_RID,
  input  logic [C_M_AXI_GMEM1_RUSER_WIDTH-1:0]  m_axi_gmem1_RUSER,
  input  logic [1:0]                            m_axi_gmem1_RRESP,
  input  logic                                  m_axi_gmem1_BVALID,
  output logic                                  m_axi_gmem1_BREADY,
  input  logic [1:0]                            m_axi_gmem1_BRESP,
  input  logic [C_M_AXI_GMEM1_ID_WIDTH-1:0]     m_axi_gmem1_BID,
  input  logic [C_M_AXI_GMEM1_BUSER_WIDTH-1:0]  m_axi_gmem1_BUSER,

  // gmem2: bias (read only)
  output logic                                  m_axi_gmem2_AWVALID,
  input  logic                                  m_axi_gmem2_AWREADY,
  output logic [C_M_AXI_GMEM2_ADDR_WIDTH-1:0]   m_axi_gmem2_AWADDR,
  output logic [C_M_AXI_GMEM2_ID_WIDTH-1:0]     m_axi_gmem2_AWID,
  output logic [7:0]                            m_axi_gmem2_AWLEN,
  output logic [2:0]                            m_axi_gmem2_AWSIZE,
  output logic [1:0]                            m_axi_gmem2_AWBURST,
  output logic [1:0]                            m_axi_gmem2_AWLOCK,
  output logic [3:0]                            m_axi_gmem2_AWCACHE,
  output logic [2:0]                            m_axi_gmem2_AWPROT,
  output logic [3:0]                            m_axi_gmem2_AWQOS,
  output logic [3:0]                            m_axi_gmem2_AWREGION,
  output logic [C_M_AXI_GMEM2_AWUSER_WIDTH-1:0] m_axi_gmem2_AWUSER,
  output logic                                  m_axi_gmem2_WVALID,
  input  logic                                  m_axi_gmem2_WREADY,
  output logic [C_M_AXI_GMEM2_DATA_WIDTH-1:0]   m_axi_gmem2_WDATA,
  output logic [C_M_AXI_GMEM2_DATA_WIDTH/8-1:0] m_axi_gmem2_WSTRB,
  output logic                                  m_axi_gmem2_WLAST,
  output logic [C_M_AXI_GMEM2_ID_WIDTH-1:0]     m_axi_gmem2_WID,
  output logic [C_M_AXI_GMEM2_WUSER_WIDTH-1:0]  m_axi_gmem2_WUSER,
  output logic                                  m_axi_gmem2_ARVALID,
  input  logic                                  m_axi_gmem2_ARREADY,
  output logic [C_M_AXI_GMEM2_ADDR_WIDTH-1:0]   m_axi_gmem2_ARADDR,
  output logic [C_M_AXI_GMEM2_ID_WIDTH-1:0]     m_axi_gmem2_ARID,
  output logic [7:0]                            m_axi_gmem2_ARLEN,
  output logic [2:0]                            m_axi_gmem2_ARSIZE,
  output logic [1:0]                            m_axi_gmem2_ARBURST,
  output logic [1:0]                            m_axi_gmem2_ARLOCK,
  output logic [3:0]                            m_axi_gmem2_ARCACHE,
  output logic [2:0]                            m_axi_gmem2_ARPROT,
  output logic [3:0]                            m_axi_gmem2_ARQOS,
  output logic [3:0]                            m_axi_gmem2_ARREGION,
  output logic [C_M_AXI_GMEM2_ARUSER_WIDTH-1:0] m_axi_gmem2_ARUSER,
  input  logic                                  m_axi_gmem2_RVALID,
  output logic                                  m_axi_gmem2_RREADY,
  input  logic [C_M_AXI_GMEM2_DATA_WIDTH-1:0]   m_axi_gmem2_RDATA,
  input  logic                                  m_axi_gmem2_RLAST,
  input  logic [C_M_AXI_GMEM2_ID_WIDTH-1:0]     m_axi_gmem2_RID,
  input  logic [C_M_AXI_GMEM2_RUSER_WIDTH-1:0]  m_axi_gmem2_RUSER,
  input  logic [1:0]                            m_axi_gmem2_RRESP,
  input  logic                                  m_axi_gmem2_BVALID,
  output logic                                  m_axi_gmem2_BREADY,
  input  logic [1:0]                            m_axi_gmem2_BRESP,
  input  logic [C_M_AXI_GMEM2_ID_WIDTH-1:0]     m_axi_gmem2_BID,
  input  logic [C_M_AXI_GMEM2_BUSER_WIDTH-1:0]  m_axi_gmem2_BUSER,

  // gmem3: y (write only)
  output logic                                  m_axi_gmem3_AWVALID,
  input  logic                                  m_axi_gmem3_AWREADY,
  output logic [C_M_AXI_GMEM3_ADDR_WIDTH-1:0]   m_axi_gmem3_AWADDR,
  output logic [C_M_AXI_GMEM3_ID_WIDTH-1:0]     m_axi_gmem3_AWID,
  output logic [7:0]                            m_axi_gmem3_AWLEN,
  output logic [2:0]                            m_axi_gmem3_AWSIZE,
  output logic [1:0]                            m_axi_gmem3_AWBURST,
  output logic [1:0]                            m_axi_gmem3_AWLOCK,
  output logic [3:0]                            m_axi_gmem3_AWCACHE,
  output logic [2:0]                            m_axi_gmem3_AWPROT,
  output logic [3:0]                            m_axi_gmem3_AWQOS,
  output logic [3:0]                            m_axi_gmem3_AWREGION,
  output logic [C_M_AXI_GMEM3_AWUSER_WIDTH-1:0] m_axi_gmem3_AWUSER,
  output logic                                  m_axi_gmem3_WVALID,
  input  logic                                  m_axi_gmem3_WREADY,
  output logic [C_M_AXI_GMEM3_DATA_WIDTH-1:0]   m_axi_gmem3_WDATA,
  output logic [C_M_AXI_GMEM3_DATA_WIDTH/8-1:0] m_axi_gmem3_WSTRB,
  output logic                                  m_axi_gmem3_WLAST,
  output logic [C_M_AXI_GMEM3_ID_WIDTH-1:0]     m_axi_gmem3_WID,
  output logic [C_M_AXI_GMEM3_WUSER_WIDTH-1:0]  m_axi_gmem3_WUSER,
  output logic                                  m_axi_gmem3_ARVALID,
  input  logic                                  m_axi_gmem3_ARREADY,
  output logic [C_M_AXI_GMEM3_ADDR_WIDTH-1:0]   m_axi_gmem3_ARADDR,
  output logic [C_M_AXI_GMEM3_ID_WIDTH-1:0]     m_axi_gmem3_ARID,
  output logic [7:0]                            m_axi_gmem3_ARLEN,
  output logic [2:0]                            m_axi_gmem3_ARSIZE,
  output logic [1:0]                            m_axi_gmem3_ARBURST,
  output logic [1:0]                            m_axi_gmem3_ARLOCK,
  output logic [3:0]                            m_axi_gmem3_ARCACHE,
  output logic [2:0]                            m_axi_gmem3_ARPROT,
  output logic [3:0]                            m_axi_gmem3_ARQOS,
  output logic [3:0]                            m_axi_gmem3_ARREGION,
  output logic [C_M_AXI_GMEM3_ARUSER_WIDTH-1:0] m_axi_gmem3_ARUSER,
  input  logic                                  m_axi_gmem3_RVALID,
  output logic                                  m_axi_gmem3_RREADY,
  input  logic [C_M_AXI_GMEM3_DATA_WIDTH-1:0]   m_axi_gmem3_RDATA,
  input  logic                                  m_axi_gmem3_RLAST,
  input  logic [C_M_AXI_GMEM3_ID_WIDTH-1:0]     m_axi_gmem3_RID,
  input  logic [C_M_AXI_GMEM3_RUSER_WIDTH-1:0]  m_axi_gmem3_RUSER,
  input  logic [1:0]                            m_axi_gmem3_RRESP,
  input  logic                                  m_axi_gmem3_BVALID,
  output logic                                  m_axi_gmem3_BREADY,
  input  logic [1:0]                            m_axi_gmem3_BRESP,
  input  logic [C_M_AXI_GMEM3_ID_WIDTH-1:0]     m_axi_gmem3_BID,
  input  logic [C_M_AXI_GMEM3_BUSER_WIDTH-1:0]  m_axi_gmem3_BUSER,

  // AXI4-Lite control
  input  logic                                  s_axi_ctrl_AWVALID,
  output logic                                  s_axi_ctrl_AWREADY,
  input  logic [C_S_AXI_CTRL_ADDR_WIDTH-1:0]    s_axi_ctrl_AWADDR,
  input  logic                                  s_axi_ctrl_WVALID,
  output logic                                  s_axi_ctrl_WREADY,
  input  logic [C_S_AXI_CTRL_DATA_WIDTH-1:0]    s_axi_ctrl_WDATA,
  input  logic [C_S_AXI_CTRL_DATA_WIDTH/8-1:0]  s_axi_ctrl_WSTRB,
  input  logic                                  s_axi_ctrl_ARVALID,
  output logic                                  s_axi_ctrl_ARREADY,
  input  logic [C_S_AXI_CTRL_ADDR_WIDTH-1:0]    s_axi_ctrl_ARADDR,
  output logic                                  s_axi_ctrl_RVALID,
  input  logic                                  s_axi_ctrl_RREADY,
  output logic [C_S_AXI_CTRL_DATA_WIDTH-1:0]    s_axi_ctrl_RDATA,
  output logic [1:0]                            s_axi_ctrl_RRESP,
  output logic                                  s_axi_ctrl_BVALID,
  input  logic                                  s_axi_ctrl_BREADY,
  output logic [1:0]                            s_axi_ctrl_BRESP,
  output logic                                  interrupt
);

  // ===========================================================================
  // Control slave and the job's registers
  // ===========================================================================
  logic        clk, rst;
  assign clk = ap_clk;
  assign rst = !ap_rst_n;

  logic        ap_start, ap_done, ap_idle, ap_ready;
  logic [63:0] r_x, r_w, r_b, r_y;
  logic [31:0] r_batch, r_in_ch, r_in_h, r_in_w, r_out_ch, r_out_h, r_out_w, r_kh, r_kw;
  logic [31:0] r_sh, r_sw, r_dh, r_dw, r_pt, r_pl, r_has_bias, r_dwm;

  cv_ctrl_s_axi u_ctrl (
    .clk, .rst,
    .awvalid (s_axi_ctrl_AWVALID), .awready (s_axi_ctrl_AWREADY), .awaddr (s_axi_ctrl_AWADDR),
    .wvalid  (s_axi_ctrl_WVALID),  .wready  (s_axi_ctrl_WREADY),  .wdata  (s_axi_ctrl_WDATA),
    .wstrb   (s_axi_ctrl_WSTRB),
    .bvalid  (s_axi_ctrl_BVALID),  .bready  (s_axi_ctrl_BREADY),  .bresp  (s_axi_ctrl_BRESP),
    .arvalid (s_axi_ctrl_ARVALID), .arready (s_axi_ctrl_ARREADY), .araddr (s_axi_ctrl_ARADDR),
    .rvalid  (s_axi_ctrl_RVALID),  .rready  (s_axi_ctrl_RREADY),  .rdata  (s_axi_ctrl_RDATA),
    .rresp   (s_axi_ctrl_RRESP),
    .interrupt,
    .ap_start, .ap_done, .ap_ready, .ap_idle,
    .x (r_x), .weight (r_w), .bias (r_b), .y (r_y), .batch (r_batch), .in_ch (r_in_ch),
    .in_h (r_in_h), .in_w (r_in_w), .out_ch (r_out_ch), .out_h (r_out_h), .out_w (r_out_w),
    .kh (r_kh), .kw (r_kw), .stride_h (r_sh), .stride_w (r_sw), .dilation_h (r_dh),
    .dilation_w (r_dw), .pad_top (r_pt), .pad_left (r_pl), .has_bias (r_has_bias),
    .is_depthwise (r_dwm)
  );

  // ===========================================================================
  // Job configuration (the HLS compute_conv_geometry): the contract, the
  // products the sweeps need (one pipelined multiplier: operands registered
  // in cycle i, product in mp3 in cycle i + 3), and three divisions — the
  // chunk height, the ow-tile width and the m-group residency cap.
  // ===========================================================================
  typedef enum logic [3:0] {T_IDLE, T_PA, T_PB, T_DIV, T_GEO, T_GEO2, T_PC, T_SEQ, T_WAIT, T_DONE} tst_t;
  tst_t        tstate;
  logic [3:0]  ci;
  job_t        j;
  logic [31:0] q_batch, q_in_ch, q_in_h, q_in_w, q_out_ch, q_out_h, q_out_w;
  logic [31:0] q_sh, q_sw, q_dh, q_dw, q_pt, q_pl;
  logic [2:0]  q_kh, q_kw;
  logic        q_bias, q_dwm, q_kh_ok, q_kw_ok;
  logic [59:0] q_x, q_w, q_b, q_y;
  logic [31:0] ma, mb, ma_q, mb_q, mp1, mp2, mp3;
  logic [31:0] in_hw, out_hw, row_words, lastw, reach_h, reach_w, n_pos, ch_in, ch_out, pmw;
  logic [31:0] per_sh, per_ow, owpt_sw, owpt_mt, gstep_q;
  // for the sweep sequencer: reach_w - pad_left, the last input column of
  // the first ow-tile and of the last one, in_w - 1
  logic [31:0] q_rpl, q_iwe0, q_iwl, q_inw1;
  logic [12:0] per;                        // output rows per chunk
  logic [16:0] geo_p;                      // per before the line-buffer cap
  logic [6:0]  owpt;                       // output columns per ow-tile
  logic [6:0]  m_tiles, ic_tiles;
  logic [2:0]  mtpg;                       // m-tiles per group
  logic        last_half;                  // the last ic-tile is a half tile
  logic        job_start, start_q, urst, seq_done, all_idle;

  // registers: the tile counts a cycle after the q_* registers, mtpg and
  // last_half two (first used in T_PA cycle 2, T_PB, T_PC)
  always_ff @(posedge clk) begin
    m_tiles   <= 7'((q_out_ch + 32'd15) >> 4);
    ic_tiles  <= 7'((q_in_ch + 32'd15) >> 4);
    mtpg      <= (m_tiles > 7'(MPG)) ? 3'(MPG) : m_tiles[2:0];
    last_half <= ((q_in_ch - {21'b0, ic_tiles - 7'd1, 4'b0}) <= 32'(E));
  end

  always_ff @(posedge clk) begin
    ma_q <= ma;
    mb_q <= mb;
    mp1  <= ma_q * mb_q;
    mp2  <= mp1;
    mp3  <= mp2;
  end

  always_comb begin
    ma = '0; mb = '0;
    case (tstate)
      T_PA: case (ci)
        4'd0: begin ma = q_in_h;                   mb = q_in_w;               end
        4'd1: begin ma = q_out_h;                  mb = q_out_w;              end
        4'd2: begin ma = q_out_w;                  mb = {25'b0, m_tiles};     end
        4'd3: begin ma = q_out_w - 32'd1;          mb = q_sw;                 end
        4'd4: begin ma = {29'b0, q_kh} - 32'd1;    mb = q_dh;                 end
        4'd5: begin ma = {29'b0, q_kw} - 32'd1;    mb = q_dw;                 end
        4'd6: begin ma = {29'b0, q_kh};            mb = {29'b0, q_kw};        end
        default: ;
      endcase
      T_PB: case (ci)
        4'd0: begin ma = q_in_ch;                  mb = in_hw;                end
        4'd1: begin ma = q_out_ch;                 mb = out_hw;               end
        4'd2: begin ma = n_pos;                    mb = {24'b0, ic_tiles - 7'd1, 1'b0} + (last_half ? 32'd1 : 32'd2); end
        default: ;
      endcase
      T_PC: case (ci)
        4'd0: begin ma = {19'b0, per};             mb = q_sh;                 end
        4'd1: begin ma = {19'b0, per};             mb = q_out_w;              end
        4'd2: begin ma = {25'b0, owpt};            mb = q_sw;                 end
        4'd3: begin ma = {25'b0, owpt};            mb = {25'b0, m_tiles};     end
        4'd4: begin ma = {29'b0, mtpg};            mb = pmw;                  end
        default: ;
      endcase
      default: ;
    endcase
  end

  // divisions: chunk rows 65536 / (out_w * m_tiles * TM), ow-tile (LBC - window_w) / stride_w,
  // residency cap (LBR - window_h) / stride_h
  logic        dv_start, dv0_done, dv1_done, dv2_done;
  logic [16:0] dv0_q;
  logic [6:0]  dv1_q;
  logic [4:0]  dv2_q;
  logic        dv_pend0, dv_pend1, dv_pend2;
  cv_div #(.N(17)) u_dv0 (.clk, .start (dv_start), .dividend (17'h10000),
                          .divisor ({row_words[27:0], 4'b0}), .quotient (dv0_q), .done (dv0_done));
  cv_div #(.N(7))  u_dv1 (.clk, .start (dv_start), .dividend (7'(32'(LBC) - reach_w - 32'd1)),
                          .divisor (q_sw), .quotient (dv1_q), .done (dv1_done));
  cv_div #(.N(5))  u_dv2 (.clk, .start (dv_start), .dividend (5'(32'(LBR) - reach_h - 32'd1)),
                          .divisor (q_sh), .quotient (dv2_q), .done (dv2_done));

  // the contract (beyond it the job reads and writes nothing)
  logic contract;
  assign contract = (q_batch != '0) && (q_in_ch != '0) && (q_in_ch <= 32'(MAXIC)) &&
                    (q_out_ch != '0) && (q_out_ch <= 32'(MAXOC)) && (q_out_h != '0) &&
                    (q_out_w != '0) && (q_out_w <= 32'(ACCN / TM)) && q_kh_ok && q_kw_ok &&
                    (q_sh != '0) && (q_sw != '0) && (q_dh != '0) && (q_dw != '0);

  always_ff @(posedge clk) begin
    if (rst) begin
      tstate    <= T_IDLE;
      job_start <= 1'b0;
      dv_start  <= 1'b0;
    end else begin
      job_start <= 1'b0;
      dv_start  <= 1'b0;
      case (tstate)
        T_IDLE: if (ap_start) begin
          q_x <= r_x[63:4];  q_w <= r_w[63:4];  q_b <= r_b[63:4];  q_y <= r_y[63:4];
          {q_batch, q_in_ch, q_in_h, q_in_w} <= {r_batch, r_in_ch, r_in_h, r_in_w};
          {q_out_ch, q_out_h, q_out_w}       <= {r_out_ch, r_out_h, r_out_w};
          {q_sh, q_sw, q_dh, q_dw, q_pt, q_pl} <= {r_sh, r_sw, r_dh, r_dw, r_pt, r_pl};
          q_kh    <= r_kh[2:0];
          q_kw    <= r_kw[2:0];
          // kh, kw in 1..MAXK; a multi-tap dilated window within the line buffer
          q_kh_ok <= (r_kh != '0) && (r_kh <= 32'(MAXK)) && ((r_kh == 32'd1) || (r_dh < 32'(LBR)));
          q_kw_ok <= (r_kw != '0) && (r_kw <= 32'(MAXK)) && ((r_kw == 32'd1) || (r_dw < 32'(LBC)));
          q_bias  <= (r_has_bias != '0);
          q_dwm   <= (r_dwm != '0);
          ci      <= '0;
          tstate  <= T_PA;
        end
        T_PA: begin
          ci <= ci + 4'd1;
          case (ci)
            4'd4: in_hw     <= mp3;
            4'd5: out_hw    <= mp3;
            4'd6: row_words <= mp3;
            4'd7: lastw     <= mp3;
            4'd8: reach_h   <= mp3;
            4'd9: reach_w   <= mp3;
            4'd10: begin
              n_pos  <= mp3;
              ci     <= '0;
              tstate <= T_PB;
            end
            default: ;
          endcase
        end
        T_PB: begin
          ci <= ci + 4'd1;
          case (ci)
            4'd4: ch_in  <= mp3;
            4'd5: ch_out <= mp3;
            4'd6: begin
              pmw <= mp3;
              if (contract && reach_h < 32'(LBR) && reach_w < 32'(LBC) &&
                  row_words <= 32'(ACCN / TM)) begin
                dv_start <= 1'b1;
                {dv_pend0, dv_pend1, dv_pend2} <= 3'b111;
                tstate   <= T_DIV;
              end else begin
                tstate <= T_DONE;
              end
            end
            default: ;
          endcase
        end
        T_DIV: begin
          if (dv0_done) dv_pend0 <= 1'b0;
          if (dv1_done) dv_pend1 <= 1'b0;
          if (dv2_done) dv_pend2 <= 1'b0;
          if (!(dv_pend0 && !dv0_done) && !(dv_pend1 && !dv1_done) && !(dv_pend2 && !dv2_done) &&
              !dv_start)
            tstate <= T_GEO;
        end
        T_GEO: begin
          // chunk rows: at least 1, at most out_h (T_GEO2: capped by the line
          // buffer when more than one m-group replays the chunk, standard)
          logic [16:0] p;
          logic [6:0]  w;
          p = (dv0_q == '0) ? 17'd1 : dv0_q;
          if (32'(p) > q_out_h) p = 17'(q_out_h);
          geo_p <= p;
          // ow-tile: (LBC - window_w) / stride_w + 1, even, at most out_w
          w = (reach_w + 32'd1 >= 32'(LBC)) ? 7'd1 : dv1_q + 7'd1;
          if (w > 7'd1) w = w & ~7'd1;
          if (32'(w) > q_out_w) w = 7'(q_out_w);
          owpt   <= w;
          tstate <= T_GEO2;
        end
        T_GEO2: begin
          per    <= (!q_dwm && m_tiles > 7'(MPG) && 17'(dv2_q) + 17'd1 < geo_p) ? 13'(dv2_q) + 13'd1
                                                                                : 13'(geo_p);
          ci     <= '0;
          tstate <= T_PC;
        end
        T_PC: begin
          ci <= ci + 4'd1;
          case (ci)
            4'd0: begin
              q_rpl  <= reach_w - q_pl;
              q_inw1 <= q_in_w - 32'd1;
            end
            4'd1: q_iwl   <= lastw + q_rpl;
            4'd4: per_sh  <= mp3;
            4'd5: per_ow  <= mp3;
            4'd6: owpt_sw <= mp3;
            4'd7: begin
              owpt_mt <= mp3;
              q_iwe0  <= owpt_sw + q_rpl;
            end
            4'd8: begin
              gstep_q   <= mp3;
              q_iwe0    <= q_iwe0 - q_sw;
              tstate    <= T_SEQ;
              job_start <= 1'b1;
            end
            default: ;
          endcase
        end
        T_SEQ:  if (seq_done && !job_start) tstate <= T_WAIT;
        T_WAIT: if (all_idle) tstate <= T_DONE;
        T_DONE: tstate <= T_IDLE;
        default: tstate <= T_IDLE;
      endcase
    end
  end

  assign ap_done  = (tstate == T_DONE);
  assign ap_idle  = (tstate == T_IDLE);
  assign ap_ready = ap_done;

  // the job's constants, from T_SEQ on
  always_ff @(posedge clk) begin
    if (tstate == T_PC && ci == 4'd8) begin
      j.x_w <= q_x;  j.w_w <= q_w;  j.b_w <= q_b;  j.y_w <= q_y;
      j.batch <= q_batch;  j.in_ch <= q_in_ch;  j.in_h <= q_in_h;  j.in_w <= q_in_w;
      j.out_ch <= q_out_ch;  j.out_h <= q_out_h;  j.out_w <= q_out_w;
      j.sh <= q_sh;  j.sw <= q_sw;  j.dh <= q_dh;  j.dw <= q_dw;  j.pt <= q_pt;  j.pl <= q_pl;
      j.kh <= q_kh;  j.kw <= q_kw;
      j.has_bias <= q_bias;
      j.dwm      <= q_dwm;
      j.in_hw    <= in_hw;
      j.out_hw   <= out_hw;
      j.ch_out   <= ch_out;
      j.m_tiles  <= m_tiles;
      j.ic_tiles <= ic_tiles;
      j.n_pos    <= n_pos[5:0];
      j.n_win    <= (n_pos[5:0] < 6'd2) ? 6'd2 : n_pos[5:0];
      j.reach_h  <= reach_h[3:0];
      j.row_words   <= row_words[12:0];
      j.per_m_words <= pmw[12:0];
      j.dw_words    <= 4'((n_pos[5:0] + 6'd7) >> 3);
    end
  end

  always_ff @(posedge clk) start_q <= job_start;
  always_ff @(posedge clk) urst <= rst || job_start;

  // ===========================================================================
  // Sweep sequencer: (ni, chunk, input tile, ow-tile, m-group) for standard,
  // (ni, chunk, m-tile, ow-tile) for depthwise — the HLS loop nest — every
  // quantity by increments.  Each sweep goes to the five units at once.
  // ===========================================================================
  // Three stages per sweep (at most one carry chain per value and stage), then
  // the push: a sweep is at least 2 instants, the shortest real ones dozens.
  typedef enum logic [2:0] {Q_OFF, Q_K1, Q_K2, Q_K3, Q_PUSH} qst_t;
  qst_t        qs;
  logic [31:0] s_ni, s_xn, s_yn;           // image: index, x / y element offsets
  logic [31:0] s_oh, s_ih0, s_yoh;         // chunk: first row, its input row, y offset
  logic        s_par;
  logic [6:0]  s_ct;                       // channel tile (standard: input tile, depthwise: m-tile)
  logic [31:0] s_xct, s_cwo, s_chrem;      // its x offset, weight offset, channels left
  logic [12:0] s_ow, s_owe;                // ow-tile: first column, s_ow + owpt
  logic [31:0] s_iws, s_iwe, s_owm;        // its first and (not the last tile) last input column (signed), accumulator word
  logic [6:0]  s_mtb;                      // m-group: first m-tile
  logic [31:0] s_mgw;                      // its weight offset (standard)
  logic        k_last_mg, k_last_owt, k_last_ct, k_last_chunk;
  logic [12:0] k_owe;                      // the tile's end column
  logic [31:0] k_iwe, k_hi;                // its last input column, clamped to the input
  logic [31:0] k_ohrem, k_yrem;            // output rows / y elements from the chunk on
  logic [6:0]  k_mrem;                     // m-tiles from the group on
  sweep_t      k;

  logic sq_x_rdy, sq_p_rdy, sq_w_rdy, sq_f_rdy, sq_e_rdy, s_push;
  assign s_push = (qs == Q_PUSH) && sq_x_rdy && sq_p_rdy && sq_w_rdy && sq_f_rdy && sq_e_rdy;

  always_ff @(posedge clk) begin
    if (urst) begin
      qs       <= start_q ? Q_K1 : Q_OFF;
      seq_done <= 1'b0;
      {s_ni, s_xn, s_yn, s_oh, s_yoh, s_xct, s_cwo, s_owm, s_mgw} <= '0;
      s_ih0    <= 32'd0 - q_pt;
      s_par    <= 1'b0;
      s_ct     <= '0;
      s_chrem  <= q_dwm ? q_out_ch : q_in_ch;
      s_ow     <= '0;
      s_owe    <= {6'b0, owpt};
      s_iws    <= 32'd0 - q_pl;
      s_iwe    <= q_iwe0;
      s_mtb    <= '0;
    end else if (job_start) begin
      seq_done <= 1'b0;
    end else if (start_q) begin
      qs <= Q_K1;
    end else begin
      case (qs)
        Q_K1: begin
          logic last_owt;
          last_owt = ({19'b0, s_owe} >= j.out_w);
          k_last_owt    <= last_owt;
          k_owe         <= last_owt ? 13'(j.out_w) : s_owe;
          // the tile's last input column: (ow_end - 1) * stride_w + reach_w - pad_left
          k_iwe         <= last_owt ? q_iwl : s_iwe;
          k_ohrem       <= j.out_h - s_oh;
          k_yrem        <= j.out_hw - s_yoh;
          k_mrem        <= j.m_tiles - s_mtb;
          k.iw_lo       <= $signed(s_iws) < 0 ? 32'd0 : s_iws;
          k.load        <= j.dwm || (s_mtb == '0);
          k.seed_bias   <= j.dwm || (s_ct == '0);
          k.par         <= s_par;
          k.half        <= !j.dwm && (s_ct + 7'd1 == j.ic_tiles) && last_half;
          k.x_cbase     <= s_xn + s_xct;
          k.ch_valid    <= (s_chrem >= 32'(TIC)) ? 5'(TIC) : s_chrem[4:0];
          k.ih0         <= s_ih0;
          k.ow_start    <= s_ow;
          k.pair_base   <= {s_ow[12:1], 1'b0};
          k.iwb0        <= s_iws - (s_ow[0] ? j.sw : 32'd0);
          k.mt_base     <= j.dwm ? s_ct : s_mtb;
          k.wrow0       <= 13'(s_owm - (s_ow[0] ? {25'b0, j.m_tiles} : 32'd0)) + (j.dwm ? {6'b0, s_ct} : {6'b0, s_mtb});
          k.w_off       <= j.dwm ? s_cwo : s_mgw + s_cwo;
          k.y_cbase     <= s_yn + s_yoh;
          k_last_mg     <= j.dwm || (s_mtb + {4'b0, mtpg} >= j.m_tiles);
          k_last_ct     <= j.dwm ? (s_ct + 7'd1 == j.m_tiles) : (s_ct + 7'd1 == j.ic_tiles);
          k.chunk_first <= (s_ct == '0) && (s_ow == '0) && (j.dwm || s_mtb == '0);
          qs            <= Q_K2;
        end
        Q_K2: begin
          k_hi          <= ($signed(k_iwe) >= $signed(j.in_w)) ? q_inw1 : k_iwe;
          k.ow_end      <= k_owe;
          k.n_pairs     <= ((k_owe - 13'd1) >> 1) - {1'b0, s_ow[12:1]} + 13'd1;
          k.coh         <= (k_ohrem >= {19'b0, per}) ? per : 13'(k_ohrem);
          k.run_len     <= (k_ohrem >= {19'b0, per}) ? 13'(per_ow) : 13'(k_yrem);
          k.g_n         <= j.dwm ? 3'd1 : ((k_mrem >= {4'b0, mtpg}) ? mtpg : 3'(k_mrem));
          k_last_chunk  <= (k_ohrem <= {19'b0, per});
          k.chunk_last  <= k_last_ct && k_last_owt && k_last_mg;
          qs            <= Q_K3;
        end
        Q_K3: begin
          k.iw_cnt      <= ($signed(k_hi) >= $signed(k.iw_lo)) ? 7'(k_hi - k.iw_lo + 32'd1) : 7'd0;
          qs            <= Q_PUSH;
        end
        Q_PUSH: if (s_push) begin
          qs <= Q_K1;
          if (!k_last_mg) begin
            s_mtb <= s_mtb + {4'b0, mtpg};
            s_mgw <= s_mgw + {gstep_q[27:0], 4'b0};
          end else begin
            s_mtb <= '0;
            s_mgw <= '0;
            if (!k_last_owt) begin
              s_ow  <= s_ow + {6'b0, owpt};
              s_owe <= s_owe + {6'b0, owpt};
              s_iws <= s_iws + owpt_sw;
              s_iwe <= s_iwe + owpt_sw;
              s_owm <= s_owm + owpt_mt;
            end else begin
              s_ow  <= '0;
              s_owe <= {6'b0, owpt};
              s_iws <= 32'd0 - j.pl;
              s_iwe <= q_iwe0;
              s_owm <= '0;
              if (!k_last_ct) begin
                s_ct    <= s_ct + 7'd1;
                s_xct   <= s_xct + {j.in_hw[27:0], 4'b0};
                s_cwo   <= s_cwo + (j.dwm ? {24'b0, j.dw_words, 4'b0} : {25'b0, j.n_pos, 1'b0});
                s_chrem <= s_chrem - 32'(TIC);
              end else begin
                s_ct    <= '0;
                s_xct   <= '0;
                s_cwo   <= '0;
                s_chrem <= j.dwm ? j.out_ch : j.in_ch;
                s_par   <= !s_par;
                if (!k_last_chunk) begin
                  s_oh  <= s_oh + {19'b0, per};
                  s_ih0 <= s_ih0 + per_sh;
                  s_yoh <= s_yoh + per_ow;
                end else begin
                  s_oh  <= '0;
                  s_ih0 <= 32'd0 - j.pt;
                  s_yoh <= '0;
                  if (s_ni + 32'd1 == j.batch) begin
                    seq_done <= 1'b1;
                    qs       <= Q_OFF;
                  end else begin
                    s_ni <= s_ni + 32'd1;
                    s_xn <= s_xn + ch_in;
                    s_yn <= s_yn + j.ch_out;
                  end
                end
              end
            end
          end
        end
        default: ;
      endcase
    end
  end

  // ===========================================================================
  // Units
  // ===========================================================================
  logic   xq_v, pq_v, wq_v, fq_v, eq_v, xq_r, pq_r, wq_r, fq_r, eq_r;
  sweep_t xq, pq, wq, fq, eq;
  logic [2:0] xq_n, pq_n, wq_n, fq_n, eq_n;
  cv_fifo #(.W($bits(sweep_t)), .D(4), .BRAM(1'b0)) u_xq (.clk, .rst (urst),
    .in_valid (s_push), .in_ready (sq_x_rdy), .in_data (k), .out_valid (xq_v), .out_ready (xq_r), .out_data (xq), .count (xq_n));
  cv_fifo #(.W($bits(sweep_t)), .D(4), .BRAM(1'b0)) u_pq (.clk, .rst (urst),
    .in_valid (s_push), .in_ready (sq_p_rdy), .in_data (k), .out_valid (pq_v), .out_ready (pq_r), .out_data (pq), .count (pq_n));
  cv_fifo #(.W($bits(sweep_t)), .D(4), .BRAM(1'b0)) u_wq (.clk, .rst (urst),
    .in_valid (s_push), .in_ready (sq_w_rdy), .in_data (k), .out_valid (wq_v), .out_ready (wq_r), .out_data (wq), .count (wq_n));
  cv_fifo #(.W($bits(sweep_t)), .D(4), .BRAM(1'b0)) u_fq (.clk, .rst (urst),
    .in_valid (s_push), .in_ready (sq_f_rdy), .in_data (k), .out_valid (fq_v), .out_ready (fq_r), .out_data (fq), .count (fq_n));
  cv_fifo #(.W($bits(sweep_t)), .D(4), .BRAM(1'b0)) u_eq (.clk, .rst (urst),
    .in_valid (s_push), .in_ready (sq_e_rdy), .in_data (k), .out_valid (eq_v), .out_ready (eq_r), .out_data (eq), .count (eq_n));

  logic                 col_v, col_r, colq_v, colq_r;
  logic [TIC*EW-1:0]    col_d, colq_d;
  logic [8:0]           colq_n;
  logic                 pt_v, pt_r;
  logic [PATCH_W-1:0]   pt_d;
  logic                 wv_v, wv_r;
  logic [TIC*EW-1:0]    wv_d;
  logic                 bias_ok;
  logic [6:0]           bias_tile;
  logic [TM*EW-1:0]     bias_vec;
  logic                 dr_v, dr_r;
  dreq_t                dr;
  logic                 d_ren, d_par, d_free, d_free_par;
  logic [11:0]          d_addr;
  logic [TM*32-1:0]     d_rdata;
  logic                 run_v, run_r, yw_v, yw_r;
  logic [31:0]          run_e;
  logic [8:0]           run_len;
  logic [BW-1:0]        yw_d;
  logic xl_idle, pa_idle, wl_idle, bi_idle, en_idle, dr_idle, yw_idle;

  cv_xload u_xl (
    .clk, .rst (urst), .j,
    .xq_valid (xq_v), .xq_ready (xq_r), .xq,
    .arvalid (m_axi_gmem0_ARVALID), .arready (m_axi_gmem0_ARREADY),
    .araddr  (m_axi_gmem0_ARADDR),  .arlen   (m_axi_gmem0_ARLEN),
    .rvalid  (m_axi_gmem0_RVALID),  .rready  (m_axi_gmem0_RREADY),
    .rdata   (m_axi_gmem0_RDATA),   .rlast   (m_axi_gmem0_RLAST),
    .col_valid (col_v), .col_ready (col_r), .col_data (col_d),
    .idle (xl_idle)
  );

  // the column FIFO: the loader runs up to 4 rows ahead of the producer
  cv_fifo #(.W(TIC * EW), .D(256), .BRAM(1'b1)) u_colq (
    .clk, .rst (urst), .in_valid (col_v), .in_ready (col_r), .in_data (col_d),
    .out_valid (colq_v), .out_ready (colq_r), .out_data (colq_d), .count (colq_n)
  );

  cv_patch u_pa (
    .clk, .rst (urst), .j,
    .pq_valid (pq_v), .pq_ready (pq_r), .pq,
    .col_valid (colq_v), .col_ready (colq_r), .col_data (colq_d),
    .pt_valid (pt_v), .pt_ready (pt_r), .pt_data (pt_d),
    .idle (pa_idle)
  );

  cv_wload u_wl (
    .clk, .rst (urst), .j,
    .wq_valid (wq_v), .wq_ready (wq_r), .wq,
    .arvalid (m_axi_gmem1_ARVALID), .arready (m_axi_gmem1_ARREADY),
    .araddr  (m_axi_gmem1_ARADDR),  .arlen   (m_axi_gmem1_ARLEN),
    .rvalid  (m_axi_gmem1_RVALID),  .rready  (m_axi_gmem1_RREADY),
    .rdata   (m_axi_gmem1_RDATA),   .rlast   (m_axi_gmem1_RLAST),
    .wv_valid (wv_v), .wv_ready (wv_r), .wv_data (wv_d),
    .idle (wl_idle)
  );

  cv_bias u_bi (
    .clk, .rst (urst), .start (start_q), .j,
    .arvalid (m_axi_gmem2_ARVALID), .arready (m_axi_gmem2_ARREADY),
    .araddr  (m_axi_gmem2_ARADDR),  .arlen   (m_axi_gmem2_ARLEN),
    .rvalid  (m_axi_gmem2_RVALID),  .rready  (m_axi_gmem2_RREADY),
    .rdata   (m_axi_gmem2_RDATA),   .rlast   (m_axi_gmem2_RLAST),
    .ok (bias_ok), .tile (bias_tile), .vec (bias_vec),
    .idle (bi_idle)
  );

  cv_engine u_en (
    .clk, .rst (urst), .j,
    .fq_valid (fq_v), .fq_ready (fq_r), .fq,
    .wv_valid (wv_v), .wv_ready (wv_r), .wv_data (wv_d),
    .eq_valid (eq_v), .eq_ready (eq_r), .eq,
    .pt_valid (pt_v), .pt_ready (pt_r), .pt_data (pt_d),
    .bias_ok, .bias_tile, .bias_vec,
    .dr_valid (dr_v), .dr_ready (dr_r), .dr,
    .d_ren, .d_par, .d_addr, .d_rdata, .d_free, .d_free_par,
    .idle (en_idle)
  );

  cv_drain u_dr (
    .clk, .rst (urst), .j,
    .dr_valid (dr_v), .dr_ready (dr_r), .dr,
    .d_ren, .d_par, .d_addr, .d_rdata, .d_free, .d_free_par,
    .run_valid (run_v), .run_ready (run_r), .run_e, .run_len,
    .yw_valid (yw_v), .yw_ready (yw_r), .yw_data (yw_d),
    .idle (dr_idle)
  );

  cv_ywriter u_yw (
    .clk, .rst (urst), .j,
    .run_valid (run_v), .run_ready (run_r), .run_e, .run_len,
    .yw_valid (yw_v), .yw_ready (yw_r), .yw_data (yw_d),
    .awvalid (m_axi_gmem3_AWVALID), .awready (m_axi_gmem3_AWREADY),
    .awaddr  (m_axi_gmem3_AWADDR),  .awlen   (m_axi_gmem3_AWLEN),
    .wvalid  (m_axi_gmem3_WVALID),  .wready  (m_axi_gmem3_WREADY),
    .wdata   (m_axi_gmem3_WDATA),   .wstrb   (m_axi_gmem3_WSTRB),
    .wlast   (m_axi_gmem3_WLAST),   .bvalid  (m_axi_gmem3_BVALID),
    .bready  (m_axi_gmem3_BREADY),
    .idle (yw_idle)
  );

  assign all_idle = !xq_v && !pq_v && !wq_v && !fq_v && !eq_v && !colq_v &&
                    xl_idle && pa_idle && wl_idle && bi_idle && en_idle && dr_idle && yw_idle;

  // AXI constants and unused channels --------------------------------------------------------
  assign m_axi_gmem0_ARID     = '0;
  assign m_axi_gmem0_ARSIZE   = 3'b100;
  assign m_axi_gmem0_ARBURST  = 2'b01;
  assign m_axi_gmem0_ARLOCK   = 2'b00;
  assign m_axi_gmem0_ARCACHE  = 4'(C_M_AXI_GMEM0_CACHE_VALUE);
  assign m_axi_gmem0_ARPROT   = 3'(C_M_AXI_GMEM0_PROT_VALUE);
  assign m_axi_gmem0_ARQOS    = 4'b0;
  assign m_axi_gmem0_ARREGION = 4'b0;
  assign m_axi_gmem0_ARUSER   = C_M_AXI_GMEM0_ARUSER_WIDTH'(C_M_AXI_GMEM0_USER_VALUE);
  assign m_axi_gmem0_AWVALID  = 1'b0;
  assign m_axi_gmem0_AWADDR   = '0;
  assign m_axi_gmem0_AWID     = '0;
  assign m_axi_gmem0_AWLEN    = '0;
  assign m_axi_gmem0_AWSIZE   = 3'b100;
  assign m_axi_gmem0_AWBURST  = 2'b01;
  assign m_axi_gmem0_AWLOCK   = 2'b00;
  assign m_axi_gmem0_AWCACHE  = 4'(C_M_AXI_GMEM0_CACHE_VALUE);
  assign m_axi_gmem0_AWPROT   = 3'(C_M_AXI_GMEM0_PROT_VALUE);
  assign m_axi_gmem0_AWQOS    = 4'b0;
  assign m_axi_gmem0_AWREGION = 4'b0;
  assign m_axi_gmem0_AWUSER   = C_M_AXI_GMEM0_AWUSER_WIDTH'(C_M_AXI_GMEM0_USER_VALUE);
  assign m_axi_gmem0_WVALID   = 1'b0;
  assign m_axi_gmem0_WDATA    = '0;
  assign m_axi_gmem0_WSTRB    = '0;
  assign m_axi_gmem0_WLAST    = 1'b0;
  assign m_axi_gmem0_WID      = '0;
  assign m_axi_gmem0_WUSER    = C_M_AXI_GMEM0_WUSER_WIDTH'(C_M_AXI_GMEM0_USER_VALUE);
  assign m_axi_gmem0_BREADY   = 1'b1;

  assign m_axi_gmem1_ARID     = '0;
  assign m_axi_gmem1_ARSIZE   = 3'b100;
  assign m_axi_gmem1_ARBURST  = 2'b01;
  assign m_axi_gmem1_ARLOCK   = 2'b00;
  assign m_axi_gmem1_ARCACHE  = 4'(C_M_AXI_GMEM1_CACHE_VALUE);
  assign m_axi_gmem1_ARPROT   = 3'(C_M_AXI_GMEM1_PROT_VALUE);
  assign m_axi_gmem1_ARQOS    = 4'b0;
  assign m_axi_gmem1_ARREGION = 4'b0;
  assign m_axi_gmem1_ARUSER   = C_M_AXI_GMEM1_ARUSER_WIDTH'(C_M_AXI_GMEM1_USER_VALUE);
  assign m_axi_gmem1_AWVALID  = 1'b0;
  assign m_axi_gmem1_AWADDR   = '0;
  assign m_axi_gmem1_AWID     = '0;
  assign m_axi_gmem1_AWLEN    = '0;
  assign m_axi_gmem1_AWSIZE   = 3'b100;
  assign m_axi_gmem1_AWBURST  = 2'b01;
  assign m_axi_gmem1_AWLOCK   = 2'b00;
  assign m_axi_gmem1_AWCACHE  = 4'(C_M_AXI_GMEM1_CACHE_VALUE);
  assign m_axi_gmem1_AWPROT   = 3'(C_M_AXI_GMEM1_PROT_VALUE);
  assign m_axi_gmem1_AWQOS    = 4'b0;
  assign m_axi_gmem1_AWREGION = 4'b0;
  assign m_axi_gmem1_AWUSER   = C_M_AXI_GMEM1_AWUSER_WIDTH'(C_M_AXI_GMEM1_USER_VALUE);
  assign m_axi_gmem1_WVALID   = 1'b0;
  assign m_axi_gmem1_WDATA    = '0;
  assign m_axi_gmem1_WSTRB    = '0;
  assign m_axi_gmem1_WLAST    = 1'b0;
  assign m_axi_gmem1_WID      = '0;
  assign m_axi_gmem1_WUSER    = C_M_AXI_GMEM1_WUSER_WIDTH'(C_M_AXI_GMEM1_USER_VALUE);
  assign m_axi_gmem1_BREADY   = 1'b1;

  assign m_axi_gmem2_ARID     = '0;
  assign m_axi_gmem2_ARSIZE   = 3'b100;
  assign m_axi_gmem2_ARBURST  = 2'b01;
  assign m_axi_gmem2_ARLOCK   = 2'b00;
  assign m_axi_gmem2_ARCACHE  = 4'(C_M_AXI_GMEM2_CACHE_VALUE);
  assign m_axi_gmem2_ARPROT   = 3'(C_M_AXI_GMEM2_PROT_VALUE);
  assign m_axi_gmem2_ARQOS    = 4'b0;
  assign m_axi_gmem2_ARREGION = 4'b0;
  assign m_axi_gmem2_ARUSER   = C_M_AXI_GMEM2_ARUSER_WIDTH'(C_M_AXI_GMEM2_USER_VALUE);
  assign m_axi_gmem2_AWVALID  = 1'b0;
  assign m_axi_gmem2_AWADDR   = '0;
  assign m_axi_gmem2_AWID     = '0;
  assign m_axi_gmem2_AWLEN    = '0;
  assign m_axi_gmem2_AWSIZE   = 3'b100;
  assign m_axi_gmem2_AWBURST  = 2'b01;
  assign m_axi_gmem2_AWLOCK   = 2'b00;
  assign m_axi_gmem2_AWCACHE  = 4'(C_M_AXI_GMEM2_CACHE_VALUE);
  assign m_axi_gmem2_AWPROT   = 3'(C_M_AXI_GMEM2_PROT_VALUE);
  assign m_axi_gmem2_AWQOS    = 4'b0;
  assign m_axi_gmem2_AWREGION = 4'b0;
  assign m_axi_gmem2_AWUSER   = C_M_AXI_GMEM2_AWUSER_WIDTH'(C_M_AXI_GMEM2_USER_VALUE);
  assign m_axi_gmem2_WVALID   = 1'b0;
  assign m_axi_gmem2_WDATA    = '0;
  assign m_axi_gmem2_WSTRB    = '0;
  assign m_axi_gmem2_WLAST    = 1'b0;
  assign m_axi_gmem2_WID      = '0;
  assign m_axi_gmem2_WUSER    = C_M_AXI_GMEM2_WUSER_WIDTH'(C_M_AXI_GMEM2_USER_VALUE);
  assign m_axi_gmem2_BREADY   = 1'b1;

  assign m_axi_gmem3_AWID     = '0;
  assign m_axi_gmem3_AWSIZE   = 3'b100;
  assign m_axi_gmem3_AWBURST  = 2'b01;
  assign m_axi_gmem3_AWLOCK   = 2'b00;
  assign m_axi_gmem3_AWCACHE  = 4'(C_M_AXI_GMEM3_CACHE_VALUE);
  assign m_axi_gmem3_AWPROT   = 3'(C_M_AXI_GMEM3_PROT_VALUE);
  assign m_axi_gmem3_AWQOS    = 4'b0;
  assign m_axi_gmem3_AWREGION = 4'b0;
  assign m_axi_gmem3_AWUSER   = C_M_AXI_GMEM3_AWUSER_WIDTH'(C_M_AXI_GMEM3_USER_VALUE);
  assign m_axi_gmem3_WID      = '0;
  assign m_axi_gmem3_WUSER    = C_M_AXI_GMEM3_WUSER_WIDTH'(C_M_AXI_GMEM3_USER_VALUE);
  assign m_axi_gmem3_ARVALID  = 1'b0;
  assign m_axi_gmem3_ARADDR   = '0;
  assign m_axi_gmem3_ARID     = '0;
  assign m_axi_gmem3_ARLEN    = '0;
  assign m_axi_gmem3_ARSIZE   = 3'b100;
  assign m_axi_gmem3_ARBURST  = 2'b01;
  assign m_axi_gmem3_ARLOCK   = 2'b00;
  assign m_axi_gmem3_ARCACHE  = 4'(C_M_AXI_GMEM3_CACHE_VALUE);
  assign m_axi_gmem3_ARPROT   = 3'(C_M_AXI_GMEM3_PROT_VALUE);
  assign m_axi_gmem3_ARQOS    = 4'b0;
  assign m_axi_gmem3_ARREGION = 4'b0;
  assign m_axi_gmem3_ARUSER   = C_M_AXI_GMEM3_ARUSER_WIDTH'(C_M_AXI_GMEM3_USER_VALUE);
  assign m_axi_gmem3_RREADY   = 1'b1;

  logic unused_top;
  assign unused_top = ^{m_axi_gmem0_AWREADY, m_axi_gmem0_WREADY, m_axi_gmem0_RID,
                        m_axi_gmem0_RUSER, m_axi_gmem0_RRESP, m_axi_gmem0_BVALID,
                        m_axi_gmem0_BRESP, m_axi_gmem0_BID, m_axi_gmem0_BUSER, m_axi_gmem1_AWREADY,
                        m_axi_gmem1_WREADY, m_axi_gmem1_RID, m_axi_gmem1_RUSER, m_axi_gmem1_RRESP,
                        m_axi_gmem1_BVALID, m_axi_gmem1_BRESP, m_axi_gmem1_BID, m_axi_gmem1_BUSER,
                        m_axi_gmem2_AWREADY, m_axi_gmem2_WREADY, m_axi_gmem2_RID,
                        m_axi_gmem2_RUSER, m_axi_gmem2_RRESP, m_axi_gmem2_BVALID,
                        m_axi_gmem2_BRESP, m_axi_gmem2_BID, m_axi_gmem2_BUSER, m_axi_gmem3_ARREADY,
                        m_axi_gmem3_RVALID, m_axi_gmem3_RDATA, m_axi_gmem3_RLAST, m_axi_gmem3_RID,
                        m_axi_gmem3_RUSER, m_axi_gmem3_RRESP, m_axi_gmem3_BRESP, m_axi_gmem3_BID,
                        m_axi_gmem3_BUSER, xq_n, pq_n, wq_n, fq_n, eq_n, colq_n};

endmodule
