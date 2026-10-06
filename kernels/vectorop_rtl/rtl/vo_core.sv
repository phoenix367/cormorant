// ---------------------------------------------------------------------------
// vo_core — element-wise Q8.8 vector kernel for the Kria KV260 (RTL).  The IP
// top level is the Verilog wrapper VectorOPKernel.v (generated from this
// header by scripts/gen_top_wrapper.py); all logic lives here.
//
// Drop-in replacement for the Vitis-HLS VectorOPKernel (kernels/vectorop):
// same module / port names, AXI-Lite register map, DDR access pattern and
// bit-identical results.  See doc/plans/VECTOROP_RTL_PLAN.md.
//
//   ap_start ─► config ─┬► rd_port a (gmem0) ─┐
//                       ├► rd_port b (gmem1) ─┴► compute (8 lanes; DIV 1 lane) ─► wr_port (gmem2)
//                       └──────────────────────────────────────────────────────────┘
// ---------------------------------------------------------------------------
module vo_core #(
  parameter int C_S_AXI_CTRL_DATA_WIDTH    = 32,
  parameter int C_S_AXI_CTRL_ADDR_WIDTH    = 7,
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

  // gmem0: a (read only)
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

  // gmem1: b (read only; idle for unary ops)
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

  // gmem2: c (write only)
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
  import vo_pkg::*;

  // The data path is built for 128-bit ports and the HLS register map.
  if (C_M_AXI_GMEM0_DATA_WIDTH != BW || C_M_AXI_GMEM1_DATA_WIDTH != BW ||
      C_M_AXI_GMEM2_DATA_WIDTH != BW || C_M_AXI_GMEM0_ADDR_WIDTH != 64 ||
      C_M_AXI_GMEM1_ADDR_WIDTH != 64 || C_M_AXI_GMEM2_ADDR_WIDTH != 64 ||
      C_S_AXI_CTRL_DATA_WIDTH != 32 || C_S_AXI_CTRL_ADDR_WIDTH != 7) begin : g_bad_params
    $error("vo_core: unsupported interface parameters");
  end

  // The reset, registered: one copy for the control slave, one for the rest.
  // keep: the four kernels' copies are equivalent registers, and the block
  // design's global synthesis would otherwise merge them into one that
  // drives all four kernels across the device (FMAX_250_PLAN).
  logic clk;
  (* keep = "true" *) logic rst;
  (* keep = "true", max_fanout = 128 *) logic rst_c;
  assign clk = ap_clk;
  always_ff @(posedge clk) begin
    rst   <= !ap_rst_n;
    rst_c <= !ap_rst_n;
  end

  // Control slave ------------------------------------------------------------------
  logic        ap_start, ap_done, ap_idle;
  logic [63:0] r_a, r_b, r_c;
  logic [31:0] r_size, r_op, r_outer, r_ainc, r_binc, r_act;

  vo_ctrl_s_axi u_ctrl (
    .clk, .rst (rst_c),
    .awvalid (s_axi_ctrl_AWVALID), .awready (s_axi_ctrl_AWREADY), .awaddr (s_axi_ctrl_AWADDR),
    .wvalid  (s_axi_ctrl_WVALID),  .wready  (s_axi_ctrl_WREADY),  .wdata  (s_axi_ctrl_WDATA),
    .wstrb   (s_axi_ctrl_WSTRB),
    .bvalid  (s_axi_ctrl_BVALID),  .bready  (s_axi_ctrl_BREADY),  .bresp  (s_axi_ctrl_BRESP),
    .arvalid (s_axi_ctrl_ARVALID), .arready (s_axi_ctrl_ARREADY), .araddr (s_axi_ctrl_ARADDR),
    .rvalid  (s_axi_ctrl_RVALID),  .rready  (s_axi_ctrl_RREADY),  .rdata  (s_axi_ctrl_RDATA),
    .rresp   (s_axi_ctrl_RRESP),
    .interrupt,
    .ap_start, .ap_done, .ap_ready (ap_done), .ap_idle,
    .a (r_a), .b (r_b), .c (r_c), .size (r_size), .op (r_op), .outer (r_outer),
    .a_inc (r_ainc), .b_inc (r_binc), .act (r_act)
  );

  // Job configuration ------------------------------------------------------------------
  // Mirrors VectorOP.cpp: n_words = ceil(size / 8); a stride-0 input of at most
  // REP_D words is replayed; outer == 1 or inc == size (whole words) is one
  // contiguous range of outer * n_words words; otherwise outer runs of n_words
  // words, run o at word o * (inc / 8).  The output uses c_inc = a_inc + b_inc.
  typedef enum logic [3:0] {T_IDLE, T_LAT, T_CFG0, T_CFG1, T_CFG2, T_CFG3, T_CFG4, T_CFG5,
                            T_RUN, T_DONE} tst_t;
  tst_t        tstate;
  logic [63:0] j_a, j_b, j_c;
  logic [31:0] j_size, j_op, j_outer, j_ainc, j_binc, j_act;
  logic [31:0] nw, c_inc, pm;
  logic [3:0]  tail;
  logic        go;
  geom_t       ga, gb, gc;
  logic        job_start, start_q, start_q2, job_done;

  function automatic geom_t mk_geom(input logic en, input logic [63:0] base,
                                    input logic [31:0] inc, input logic can_replay);
    geom_t g;
    logic  contig, rep;
    contig = (j_outer == 32'd1) || (inc == j_size && j_size[2:0] == 3'd0);
    rep    = can_replay && (j_outer > 32'd1) && (inc == '0) && (nw <= 32'(REP_D));
    g.en       = en && go;
    g.base_w   = base[63:4];
    g.tail     = tail;
    g.replay   = rep;
    g.reps     = j_outer;
    g.n_runs   = (rep || contig) ? 32'd1 : j_outer;
    g.run_words = rep ? nw : contig ? pm : nw;
    g.stride_w = (rep || contig) ? '0 : {3'b0, inc[31:3]};
    return g;
  endfunction

  // outer * n_words (low 32 bits, as the HLS kernel) from three 16 x 16
  // partial products, every stage registered (DSP A/B, M and P registers,
  // then one 32-bit add):  o * n mod 2^32
  //   = oL*nL + ((oL*nH + oH*nL) mod 2^16) << 16
  // nw is valid from T_CFG1; pm from T_CFG5 (four stages).
  logic [31:0] po, pn, m0, s0;
  logic [15:0] m1, m2, s12;
  always_ff @(posedge clk) begin
    po  <= j_outer;
    pn  <= nw;
    m0  <= po[15:0] * pn[15:0];
    m1  <= 16'(po[15:0] * pn[31:16]);
    m2  <= 16'(po[31:16] * pn[15:0]);
    s0  <= m0;
    s12 <= m1 + m2;
    pm  <= s0 + {s12, 16'b0};
  end

  // The argument latch and the geometry loads are enabled by registered
  // strobes (a state bit each, copied per geometry), not by a state decode:
  // these enables reach several hundred registers across the kernel.
  logic                ld_j;              // T_LAT: latch the arguments
  (* keep = "true" *) logic ld_ga, ld_gb, ld_gc;   // T_CFG5: load the geometries

  always_ff @(posedge clk) begin
    if (rst) begin
      tstate    <= T_IDLE;
      job_start <= 1'b0;
      ld_j      <= 1'b0;
      ld_ga     <= 1'b0;
      ld_gb     <= 1'b0;
      ld_gc     <= 1'b0;
    end else begin
      job_start <= 1'b0;
      ld_j      <= (tstate == T_IDLE) && ap_start;
      ld_ga     <= (tstate == T_CFG4);
      ld_gb     <= (tstate == T_CFG4);
      ld_gc     <= (tstate == T_CFG4);
      case (tstate)
        T_IDLE: if (ap_start) tstate <= T_LAT;
        T_LAT:  tstate <= T_CFG0;
        T_CFG0: tstate <= T_CFG1;
        T_CFG1: tstate <= T_CFG2;           // T_CFG1 .. T_CFG4: the product's four stages
        T_CFG2: tstate <= T_CFG3;
        T_CFG3: tstate <= T_CFG4;
        T_CFG4: tstate <= T_CFG5;
        T_CFG5: begin
          job_start <= 1'b1;
          tstate    <= T_RUN;
        end
        T_RUN:  if (job_done) tstate <= T_DONE;
        T_DONE: tstate <= T_IDLE;
        default: tstate <= T_IDLE;
      endcase
    end
  end

  // Job arguments and derived values (no reset; loaded before every use).
  always_ff @(posedge clk) begin
    if (ld_j) begin
      j_a <= r_a; j_b <= r_b; j_c <= r_c;
      j_size <= r_size; j_op <= r_op; j_outer <= r_outer;
      j_ainc <= r_ainc; j_binc <= r_binc; j_act <= r_act;
    end
    if (tstate == T_CFG0) begin
      nw    <= 32'((33'(j_size) + 33'd7) >> 3);
      tail  <= (j_size[2:0] == 3'd0) ? 4'd8 : {1'b0, j_size[2:0]};
      c_inc <= j_ainc + j_binc;
      go    <= (j_size != '0) && (j_outer != '0);
    end
    if (ld_ga) ga <= mk_geom(1'b1,           j_a, j_ainc, 1'b1);
    if (ld_gb) gb <= mk_geom(j_op < OP_RELU, j_b, j_binc, 1'b1);
    if (ld_gc) gc <= mk_geom(1'b1,           j_c, c_inc,  1'b0);
  end

  // Every unit starts a job from reset: the job reset is registered and
  // copied per unit (start_q2 follows it by one cycle, as start_q followed
  // the combinational reset before).
  (* keep = "true" *) logic urst_a, urst_b, urst_c, urst_w;
  always_ff @(posedge clk) begin
    start_q  <= !rst && job_start;
    start_q2 <= !rst && start_q;
    urst_a   <= rst || job_start;
    urst_b   <= rst || job_start;
    urst_c   <= rst || job_start;
    urst_w   <= rst || job_start;
  end

  assign ap_idle = (tstate == T_IDLE);
  assign ap_done = (tstate == T_DONE);

  // Operand ports, compute, output -----------------------------------------------------
  logic          a_valid, a_ready, b_valid, b_ready, c_valid, c_ready;
  logic [BW-1:0] a_data, b_data, c_data;
  logic          a_idle, b_idle, cp_idle, wr_idle;

  vo_rd_port u_rd_a (
    .clk, .rst (urst_a), .start (start_q2), .g (ga),
    .arvalid (m_axi_gmem0_ARVALID), .arready (m_axi_gmem0_ARREADY),
    .araddr  (m_axi_gmem0_ARADDR),  .arlen   (m_axi_gmem0_ARLEN),
    .rvalid  (m_axi_gmem0_RVALID),  .rready  (m_axi_gmem0_RREADY),
    .rdata   (m_axi_gmem0_RDATA),   .rlast   (m_axi_gmem0_RLAST),
    .out_valid (a_valid), .out_ready (a_ready), .out_data (a_data),
    .idle (a_idle)
  );

  vo_rd_port u_rd_b (
    .clk, .rst (urst_b), .start (start_q2), .g (gb),
    .arvalid (m_axi_gmem1_ARVALID), .arready (m_axi_gmem1_ARREADY),
    .araddr  (m_axi_gmem1_ARADDR),  .arlen   (m_axi_gmem1_ARLEN),
    .rvalid  (m_axi_gmem1_RVALID),  .rready  (m_axi_gmem1_RREADY),
    .rdata   (m_axi_gmem1_RDATA),   .rlast   (m_axi_gmem1_RLAST),
    .out_valid (b_valid), .out_ready (b_ready), .out_data (b_data),
    .idle (b_idle)
  );

  vo_compute u_cp (
    .clk, .rst (urst_c), .op (j_op), .act (j_act),
    .a_valid, .a_ready, .a_data, .b_valid, .b_ready, .b_data,
    .c_valid, .c_ready, .c_data,
    .idle (cp_idle)
  );

  vo_wr_port u_wr (
    .clk, .rst (urst_w), .start (start_q2), .g (gc),
    .in_valid (c_valid), .in_ready (c_ready), .in_data (c_data),
    .awvalid (m_axi_gmem2_AWVALID), .awready (m_axi_gmem2_AWREADY),
    .awaddr  (m_axi_gmem2_AWADDR),  .awlen   (m_axi_gmem2_AWLEN),
    .wvalid  (m_axi_gmem2_WVALID),  .wready  (m_axi_gmem2_WREADY),
    .wdata   (m_axi_gmem2_WDATA),   .wlast   (m_axi_gmem2_WLAST),
    .bvalid  (m_axi_gmem2_BVALID),  .bready  (m_axi_gmem2_BREADY),
    .idle (wr_idle)
  );
  assign m_axi_gmem2_WSTRB = '1;

  assign job_done = (tstate == T_RUN) && !job_start && !start_q && !start_q2 &&
                    a_idle && b_idle && cp_idle && wr_idle;

  // AXI constant fields and unused channels ---------------------------------------------
  // gmem0 / gmem1: read only

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
                        m_axi_gmem2_BUSER};

endmodule
