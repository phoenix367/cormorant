// ---------------------------------------------------------------------------
// mm_core — Q8.8 GEMM / GEMV kernel for the Kria KV260 (RTL).  The IP top
// level is the Verilog wrapper MatmulKernel.v (generated from this header by
// scripts/gen_top_wrapper.py); all logic lives here.
//
// Drop-in replacement for the Vitis-HLS MatmulKernel (kernels/matmul): same
// module / port names, AXI-Lite register map and DDR layouts (row-major B,
// tile-major packed B, GEMV kernel-width image), bit-identical results.
// All three m_axi ports are 128 bits wide.  See doc/kernels/MATMUL_RTL_KERNEL.md.
//
//   ap_start ─► walker ─► step FIFOs ─┬► rungen0 ─► axi_rd0 ─► gearbox0 ─┬► A writer 0 ─┐
//                                     │            (gmem0)               └► lane 0 ◄─ xpf0 ◄┤ A buffer
//                                     ├► rungen1 ─► axi_rd1 ─► gearbox1 ─┬► A writer 1 ─┘
//                                     │            (gmem1)               └► lane 1 ◄─ xpf1
//                                     └► drain ◄── lane 0/1 accumulators ─► packer ─► axi_wr (gmem2)
// ---------------------------------------------------------------------------
module mm_core #(
  parameter int C_S_AXI_CTRL_DATA_WIDTH    = 32,
  parameter int C_S_AXI_CTRL_ADDR_WIDTH    = 8,
  parameter int C_M_AXI_GMEM0_ID_WIDTH     = 1,
  parameter int C_M_AXI_GMEM0_ADDR_WIDTH   = 64,
  parameter int C_M_AXI_GMEM0_DATA_WIDTH   = 128,
  parameter int C_M_AXI_GMEM0_AWUSER_WIDTH = 1,
  parameter int C_M_AXI_GMEM0_ARUSER_WIDTH = 1,
  parameter int C_M_AXI_GMEM0_WUSER_WIDTH  = 1,
  parameter int C_M_AXI_GMEM0_RUSER_WIDTH  = 1,
  parameter int C_M_AXI_GMEM0_BUSER_WIDTH  = 1,
  parameter int C_M_AXI_GMEM0_USER_VALUE   = 0,
  parameter int C_M_AXI_GMEM0_PROT_VALUE   = 0,
  parameter int C_M_AXI_GMEM0_CACHE_VALUE  = 3,
  parameter int C_M_AXI_GMEM1_ID_WIDTH     = 1,
  parameter int C_M_AXI_GMEM1_ADDR_WIDTH   = 64,
  parameter int C_M_AXI_GMEM1_DATA_WIDTH   = 128,
  parameter int C_M_AXI_GMEM1_AWUSER_WIDTH = 1,
  parameter int C_M_AXI_GMEM1_ARUSER_WIDTH = 1,
  parameter int C_M_AXI_GMEM1_WUSER_WIDTH  = 1,
  parameter int C_M_AXI_GMEM1_RUSER_WIDTH  = 1,
  parameter int C_M_AXI_GMEM1_BUSER_WIDTH  = 1,
  parameter int C_M_AXI_GMEM1_USER_VALUE   = 0,
  parameter int C_M_AXI_GMEM1_PROT_VALUE   = 0,
  parameter int C_M_AXI_GMEM1_CACHE_VALUE  = 3,
  parameter int C_M_AXI_GMEM2_ID_WIDTH     = 1,
  parameter int C_M_AXI_GMEM2_ADDR_WIDTH   = 64,
  parameter int C_M_AXI_GMEM2_DATA_WIDTH   = 128,
  parameter int C_M_AXI_GMEM2_AWUSER_WIDTH = 1,
  parameter int C_M_AXI_GMEM2_ARUSER_WIDTH = 1,
  parameter int C_M_AXI_GMEM2_WUSER_WIDTH  = 1,
  parameter int C_M_AXI_GMEM2_RUSER_WIDTH  = 1,
  parameter int C_M_AXI_GMEM2_BUSER_WIDTH  = 1,
  parameter int C_M_AXI_GMEM2_USER_VALUE   = 0,
  parameter int C_M_AXI_GMEM2_PROT_VALUE   = 0,
  parameter int C_M_AXI_GMEM2_CACHE_VALUE  = 3
) (
  input  logic ap_clk,
  input  logic ap_rst_n,

  // gmem0: A and half of B (read only)
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

  // gmem1: A and half of B (read only)
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

  // gmem2: C (write only)
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
  import mm_pkg::*;

  // The data path is built for 128-bit ports and the HLS register map.
  if (C_M_AXI_GMEM0_DATA_WIDTH != BW || C_M_AXI_GMEM1_DATA_WIDTH != BW ||
      C_M_AXI_GMEM2_DATA_WIDTH != BW || C_M_AXI_GMEM0_ADDR_WIDTH != 64 ||
      C_M_AXI_GMEM1_ADDR_WIDTH != 64 || C_M_AXI_GMEM2_ADDR_WIDTH != 64 ||
      C_S_AXI_CTRL_DATA_WIDTH != 32 || C_S_AXI_CTRL_ADDR_WIDTH != 8) begin : g_bad_params
    $error("mm_core: unsupported interface parameters");
  end

  logic clk, rst;
  assign clk = ap_clk;
  always_ff @(posedge clk) rst <= !ap_rst_n;

  // Control slave ------------------------------------------------------------------
  logic        ap_start, ap_done, ap_idle;
  logic [63:0] r_a, r_b, r_c, r_a_to_b;
  logic [31:0] r_n, r_k, r_m, r_batch, r_as, r_bs, r_cs, r_bp, r_kw;

  mm_ctrl_s_axi u_ctrl (
    .clk, .rst,
    .awvalid (s_axi_ctrl_AWVALID), .awready (s_axi_ctrl_AWREADY), .awaddr (s_axi_ctrl_AWADDR),
    .wvalid  (s_axi_ctrl_WVALID),  .wready  (s_axi_ctrl_WREADY),  .wdata  (s_axi_ctrl_WDATA),
    .wstrb   (s_axi_ctrl_WSTRB),
    .bvalid  (s_axi_ctrl_BVALID),  .bready  (s_axi_ctrl_BREADY),  .bresp  (s_axi_ctrl_BRESP),
    .arvalid (s_axi_ctrl_ARVALID), .arready (s_axi_ctrl_ARREADY), .araddr (s_axi_ctrl_ARADDR),
    .rvalid  (s_axi_ctrl_RVALID),  .rready  (s_axi_ctrl_RREADY),  .rdata  (s_axi_ctrl_RDATA),
    .rresp   (s_axi_ctrl_RRESP),
    .interrupt,
    .ap_start, .ap_done, .ap_ready (ap_done), .ap_idle,
    .a (r_a), .b (r_b), .c (r_c), .n (r_n), .k (r_k), .m (r_m), .batch (r_batch),
    .a_batch_stride (r_as), .b_batch_stride (r_bs), .c_batch_stride (r_cs),
    .b_packed (r_bp), .gemv_kw (r_kw), .a_to_b (r_a_to_b)
  );

  // Job sequencing ---------------------------------------------------------------------
  typedef enum logic [1:0] {T_IDLE, T_CFG, T_RUN, T_DONE} tst_t;
  tst_t        tstate;
  logic        job_start, urst;
  cfg_t        cfg;
  logic [63:0] j_a, j_b, j_c;
  logic [31:0] j_n, j_k, j_m, j_batch, j_as, j_bs, j_cs, j_bp, j_kw;
  logic        job_done;

  always_ff @(posedge clk) begin
    if (rst) begin
      tstate    <= T_IDLE;
      job_start <= 1'b0;
    end else begin
      job_start <= 1'b0;
      case (tstate)
        T_IDLE: if (ap_start) begin
          j_a <= r_a; j_b <= r_b; j_c <= r_c;
          j_n <= r_n; j_k <= r_k; j_m <= r_m; j_batch <= r_batch;
          j_as <= r_as; j_bs <= r_bs; j_cs <= r_cs; j_bp <= r_bp; j_kw <= r_kw;
          tstate <= T_CFG;
        end
        T_CFG: begin
          logic [12:0] kc, pl;
          logic [1:0]  lk;
          logic        pk;
          logic [35:0] lf;
          kc = (j_k > 32'(K_MAX)) ? 13'(K_MAX) : j_k[12:0];
          lk = (j_kw == 32'd2) ? 2'd1 : (j_kw == 32'd4) ? 2'd2 : (j_kw == 32'd8) ? 2'd3 : 2'd0;
          pk = (j_kw == '0) && (j_bp != '0);
          pl = kc >> lk;
          lf = {4'b0, j_m} << lk;
          cfg.n      <= j_n;
          cfg.m      <= j_m;
          cfg.batch  <= j_batch;
          cfg.k      <= kc;
          cfg.pk     <= pk;
          cfg.lk     <= lk;
          cfg.planes <= pl;
          cfg.nblk   <= 9'((pl + 13'd15) >> 4);
          cfg.lfull  <= lf;
          cfg.contig <= !pk && (lf <= 36'(W_EL));
          cfg.mc_max <= pk ? 10'(W_EL)
                      : (lf <= 36'(W_EL)) ? j_m[9:0]
                      : 10'(10'(W_EL) >> lk);
          job_start <= 1'b1;
          tstate    <= T_RUN;
        end
        T_RUN:  if (job_done) tstate <= T_DONE;
        T_DONE: tstate <= T_IDLE;
        default: tstate <= T_IDLE;
      endcase
    end
  end

  assign ap_idle = (tstate == T_IDLE);
  assign ap_done = (tstate == T_DONE);
  assign urst    = rst || job_start;      // every unit starts a job from reset

  // Walker and step queues ----------------------------------------------------------
  logic  w_valid, w_ready, w_done;
  step_t w_step;
  logic [31:0] n_steps;
  logic [2:0]  sq_in_ready, sq_out_valid, sq_out_ready;
  step_t       sq_out [3];

  mm_walker u_walker (
    .clk, .rst, .start (job_start), .cfg,
    .a_base (j_a), .b_base (j_b), .c_base (j_c),
    .a_stride (j_as), .b_stride (j_bs), .c_stride (j_cs),
    .step_valid (w_valid), .step_ready (w_ready), .step (w_step),
    .done (w_done), .n_steps
  );
  assign w_ready = &sq_in_ready;

  for (genvar i = 0; i < 3; i++) begin : g_sq
    logic [2:0] cnt;
    mm_fifo #(.W($bits(step_t)), .D(4), .BRAM(1'b0)) u_sq (
      .clk, .rst (urst),
      .in_valid (w_valid && w_ready), .in_ready (sq_in_ready[i]), .in_data (w_step),
      .out_valid(sq_out_valid[i]), .out_ready(sq_out_ready[i]), .out_data(sq_out[i]),
      .count    (cnt)
    );
  end

  // Per port: run generator, read engine, gearbox, A writer, x prefetcher, lane ----
  logic [1:0][31:0] awr_cnt, xpf_cnt, cmp_cnt;
  logic [1:0]       lane_active;
  logic [31:0]      drn_cnt;
  logic             drd_en;
  logic [ACC_AW-1:0] drd_addr;
  logic [2:0]       drd_row;
  logic [1:0][E-1:0][31:0] drd_data;
  logic [1:0]       rg_idle, rd_idle;

  logic [1:0]            ab_we, ab_wlane, ab_re;
  logic [1:0][1:0]       ab_wrow;
  logic [1:0][A_AW-1:0]  ab_waddr, ab_raddr;
  logic [1:0][BW-1:0]    ab_wdata;
  logic [1:0][R-1:0][BW-1:0] ab_rdata;

  // AXI read channel bundles
  logic [1:0]        ar_valid, ar_ready, r_valid, r_last;
  logic [1:0][63:0]  ar_addr;
  logic [1:0][7:0]   ar_len;
  logic [1:0][BW-1:0] r_data;
  assign ar_ready = {m_axi_gmem1_ARREADY, m_axi_gmem0_ARREADY};
  assign r_valid  = {m_axi_gmem1_RVALID,  m_axi_gmem0_RVALID};
  assign r_last   = {m_axi_gmem1_RLAST,   m_axi_gmem0_RLAST};
  assign r_data   = {m_axi_gmem1_RDATA,   m_axi_gmem0_RDATA};

  for (genvar p = 0; p < 2; p++) begin : g_port
    // run generator -> queues
    logic    gq_v, gq_r, rq_v, rq_r, xq_v, xq_r;
    gb_run_t gq_d;
    rd_run_t rq_d;
    x_run_t  xq_d;
    mm_rungen #(.P(p)) u_rg (
      .clk, .rst (urst), .start (job_start), .cfg,
      .step_valid (sq_out_valid[p]), .step_ready (sq_out_ready[p]), .step (sq_out[p]),
      .gb_valid (gq_v), .gb_ready (gq_r), .gb_run (gq_d),
      .rd_valid (rq_v), .rd_ready (rq_r), .rd_run (rq_d),
      .x_valid  (xq_v), .x_ready  (xq_r), .x_run  (xq_d),
      .idle     (rg_idle[p])
    );

    logic    gb_v, gb_r, rd_v, rd_r, x_v, x_r;
    gb_run_t gb_d;
    rd_run_t rd_d;
    x_run_t  x_d;
    logic [5:0] c0, c1, c2;
    mm_fifo #(.W($bits(gb_run_t)), .D(32), .BRAM(1'b0)) u_gq (
      .clk, .rst (urst), .in_valid (gq_v), .in_ready (gq_r), .in_data (gq_d),
      .out_valid (gb_v), .out_ready (gb_r), .out_data (gb_d), .count (c0));
    mm_fifo #(.W($bits(rd_run_t)), .D(32), .BRAM(1'b0)) u_rq (
      .clk, .rst (urst), .in_valid (rq_v), .in_ready (rq_r), .in_data (rq_d),
      .out_valid (rd_v), .out_ready (rd_r), .out_data (rd_d), .count (c1));
    mm_fifo #(.W($bits(x_run_t)), .D(32), .BRAM(1'b0)) u_xq (
      .clk, .rst (urst), .in_valid (xq_v), .in_ready (xq_r), .in_data (xq_d),
      .out_valid (x_v), .out_ready (x_r), .out_data (x_d), .count (c2));

    // read engine
    logic          rw_v, rw_r;
    logic [BW-1:0] rw_d;
    mm_axi_rd u_rd (
      .clk, .rst (urst),
      .desc_valid (rd_v), .desc_ready (rd_r), .desc (rd_d),
      .arvalid (ar_valid[p]), .arready (ar_ready[p]), .araddr (ar_addr[p]), .arlen (ar_len[p]),
      .rvalid (r_valid[p]), .rready (), .rdata (r_data[p]), .rlast (r_last[p]),
      .out_valid (rw_v), .out_ready (rw_r), .out_data (rw_d),
      .idle (rd_idle[p])
    );

    // gearbox
    logic          go_v, go_r, go_mk, go_rf, go_rl, go_uf, go_ul;
    logic [BW-1:0] go_d;
    logic [8:0]    go_beat;
    logic [4:0]    go_row;
    gb_run_t       go_meta;
    mm_gearbox u_gb (
      .clk, .rst (urst),
      .run_valid (gb_v), .run_ready (gb_r), .run (gb_d),
      .in_valid (rw_v), .in_ready (rw_r), .in_data (rw_d),
      .out_valid (go_v), .out_ready (go_r), .out_data (go_d), .out_marker (go_mk),
      .out_row_first (go_rf), .out_row_last (go_rl), .out_run_first (go_uf),
      .out_run_last (go_ul), .out_beat (go_beat), .out_row (go_row), .out_meta (go_meta)
    );

    // A writer (A beats)
    logic aw_r;
    mm_awr u_awr (
      .clk, .rst (urst), .cfg,
      .in_valid (go_v && !go_meta.dest_b), .in_ready (aw_r), .in_data (go_d),
      .in_marker (go_mk), .in_run_last (go_ul), .in_beat (go_beat), .in_row (go_row),
      .xpf_cnt, .awr_cnt (awr_cnt[p]),
      .we (ab_we[p]), .wrow (ab_wrow[p]), .wlane (ab_wlane[p]),
      .waddr (ab_waddr[p]), .wdata (ab_wdata[p])
    );

    // x prefetcher of lane p
    logic tp_v, tp_pop;
    logic [E-1:0][R-1:0][EW-1:0] tp_d;
    mm_xpf u_xpf (
      .clk, .rst (urst), .cfg,
      .xq_valid (x_v), .xq_ready (x_r), .xq (x_d),
      .awr_cnt, .xpf_cnt (xpf_cnt[p]),
      .re (ab_re[p]), .raddr (ab_raddr[p]), .rdata (ab_rdata[p]),
      .tap_valid (tp_v), .tap_pop (tp_pop), .tap_data (tp_d)
    );

    // MAC lane p (B beats)
    logic ln_r;
    mm_lane u_lane (
      .clk, .rst (urst),
      .in_valid (go_v && go_meta.dest_b), .in_ready (ln_r), .in_data (go_d),
      .in_marker (go_mk), .in_row_first (go_rf), .in_row_last (go_rl), .in_run_last (go_ul),
      .in_beat (go_beat), .in_row (go_row), .in_meta (go_meta),
      .tap_valid (tp_v), .tap_pop (tp_pop), .tap_data (tp_d),
      .drn_cnt, .cmp_cnt (cmp_cnt[p]), .active (lane_active[p]),
      .drd_en, .drd_addr, .drd_row, .drd_data (drd_data[p])
    );

    assign go_r = go_meta.dest_b ? ln_r : aw_r;

    logic unused;
    assign unused = go_uf ^ ^c0 ^ ^c1 ^ ^c2;
  end

  mm_abuf u_abuf (
    .clk,
    .we (ab_we), .wrow (ab_wrow), .wlane (ab_wlane), .waddr (ab_waddr), .wdata (ab_wdata),
    .re (ab_re), .raddr (ab_raddr), .rdata (ab_rdata)
  );

  // Drain, packer, write engine ------------------------------------------------------
  logic          cr_v, cr_r, crq_v, crq_r;
  logic [63:0]   cr_addr, crq_addr;
  logic [12:0]   cr_n, crq_n;
  logic          el_v, el_r;
  logic [BW-1:0] el_d;
  logic [3:0]    el_c;
  logic          dr_idle, pk_idle, wr_idle;

  mm_drain u_drain (
    .clk, .rst (urst), .start (job_start), .cfg,
    .step_valid (sq_out_valid[2]), .step_ready (sq_out_ready[2]), .step (sq_out[2]),
    .cmp_cnt, .lane_active, .drn_cnt,
    .drd_en, .drd_addr, .drd_row, .drd_data,
    .crun_valid (cr_v), .crun_ready (cr_r), .crun_addr (cr_addr), .crun_n (cr_n),
    .el_valid (el_v), .el_ready (el_r), .el_data (el_d), .el_cnt (el_c),
    .idle (dr_idle)
  );

  logic [2:0] crq_cnt;
  mm_fifo #(.W(77), .D(4), .BRAM(1'b0)) u_crq (
    .clk, .rst (urst), .in_valid (cr_v), .in_ready (cr_r), .in_data ({cr_n, cr_addr}),
    .out_valid (crq_v), .out_ready (crq_r), .out_data ({crq_n, crq_addr}), .count (crq_cnt));

  logic          pa_v, pa_r, pw_v, pw_r;
  logic [59:0]   pa_waddr;
  logic [11:0]   pa_nw;
  logic [BW-1:0] pw_d;
  logic [2*E-1:0] pw_s;
  mm_packer u_pk (
    .clk, .rst (urst),
    .run_valid (crq_v), .run_ready (crq_r), .run_addr (crq_addr), .run_n (crq_n),
    .el_valid (el_v), .el_ready (el_r), .el_data (el_d), .el_cnt (el_c),
    .aw_valid (pa_v), .aw_ready (pa_r), .aw_waddr (pa_waddr), .aw_nw (pa_nw),
    .w_valid (pw_v), .w_ready (pw_r), .w_data (pw_d), .w_strb (pw_s),
    .idle (pk_idle)
  );

  mm_axi_wr u_wr (
    .clk, .rst (urst),
    .desc_valid (pa_v), .desc_ready (pa_r), .desc_waddr (pa_waddr), .desc_nw (pa_nw),
    .in_valid (pw_v), .in_ready (pw_r), .in_data (pw_d), .in_strb (pw_s),
    .awvalid (m_axi_gmem2_AWVALID), .awready (m_axi_gmem2_AWREADY),
    .awaddr (m_axi_gmem2_AWADDR), .awlen (m_axi_gmem2_AWLEN),
    .wvalid (m_axi_gmem2_WVALID), .wready (m_axi_gmem2_WREADY),
    .wdata (m_axi_gmem2_WDATA), .wstrb (m_axi_gmem2_WSTRB), .wlast (m_axi_gmem2_WLAST),
    .bvalid (m_axi_gmem2_BVALID), .bready (m_axi_gmem2_BREADY),
    .idle (wr_idle)
  );

  assign job_done = (tstate == T_RUN) && !job_start && w_done && (drn_cnt == n_steps) &&
                    dr_idle && !crq_v && pk_idle && wr_idle;

  // AXI constant fields and unused channels ---------------------------------------------
  // gmem0 / gmem1: read only
  assign m_axi_gmem0_ARVALID  = ar_valid[0];
  assign m_axi_gmem0_ARADDR   = ar_addr[0];
  assign m_axi_gmem0_ARLEN    = ar_len[0];
  assign m_axi_gmem1_ARVALID  = ar_valid[1];
  assign m_axi_gmem1_ARADDR   = ar_addr[1];
  assign m_axi_gmem1_ARLEN    = ar_len[1];
  assign m_axi_gmem0_RREADY   = 1'b1;
  assign m_axi_gmem1_RREADY   = 1'b1;

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

  // gmem2: write only
  assign m_axi_gmem2_AWID     = '0;
  assign m_axi_gmem2_AWSIZE   = 3'b100;
  assign m_axi_gmem2_AWBURST  = 2'b01;
  assign m_axi_gmem2_AWLOCK   = 2'b00;
  assign m_axi_gmem2_AWCACHE  = 4'(C_M_AXI_GMEM2_CACHE_VALUE);
  assign m_axi_gmem2_AWPROT   = 3'(C_M_AXI_GMEM2_PROT_VALUE);
  assign m_axi_gmem2_AWQOS    = 4'b0;
  assign m_axi_gmem2_AWREGION = 4'b0;
  assign m_axi_gmem2_AWUSER   = C_M_AXI_GMEM2_AWUSER_WIDTH'(C_M_AXI_GMEM2_USER_VALUE);
  assign m_axi_gmem2_WID      = '0;
  assign m_axi_gmem2_WUSER    = C_M_AXI_GMEM2_WUSER_WIDTH'(C_M_AXI_GMEM2_USER_VALUE);
  assign m_axi_gmem2_ARVALID  = 1'b0;
  assign m_axi_gmem2_ARADDR   = '0;
  assign m_axi_gmem2_ARID     = '0;
  assign m_axi_gmem2_ARLEN    = '0;
  assign m_axi_gmem2_ARSIZE   = 3'b100;
  assign m_axi_gmem2_ARBURST  = 2'b01;
  assign m_axi_gmem2_ARLOCK   = 2'b00;
  assign m_axi_gmem2_ARCACHE  = 4'(C_M_AXI_GMEM2_CACHE_VALUE);
  assign m_axi_gmem2_ARPROT   = 3'(C_M_AXI_GMEM2_PROT_VALUE);
  assign m_axi_gmem2_ARQOS    = 4'b0;
  assign m_axi_gmem2_ARREGION = 4'b0;
  assign m_axi_gmem2_ARUSER   = C_M_AXI_GMEM2_ARUSER_WIDTH'(C_M_AXI_GMEM2_USER_VALUE);
  assign m_axi_gmem2_RREADY   = 1'b1;

  logic unused_top;
  assign unused_top = ^{m_axi_gmem0_AWREADY, m_axi_gmem0_WREADY, m_axi_gmem0_RID,
                        m_axi_gmem0_RUSER, m_axi_gmem0_RRESP, m_axi_gmem0_BVALID,
                        m_axi_gmem0_BRESP, m_axi_gmem0_BID, m_axi_gmem0_BUSER,
                        m_axi_gmem1_AWREADY, m_axi_gmem1_WREADY, m_axi_gmem1_RID,
                        m_axi_gmem1_RUSER, m_axi_gmem1_RRESP, m_axi_gmem1_BVALID,
                        m_axi_gmem1_BRESP, m_axi_gmem1_BID, m_axi_gmem1_BUSER,
                        m_axi_gmem2_ARREADY, m_axi_gmem2_RVALID, m_axi_gmem2_RDATA,
                        m_axi_gmem2_RLAST, m_axi_gmem2_RID, m_axi_gmem2_RUSER,
                        m_axi_gmem2_RRESP, m_axi_gmem2_BRESP, m_axi_gmem2_BID,
                        m_axi_gmem2_BUSER, r_a_to_b, rg_idle, rd_idle, crq_cnt};

endmodule
