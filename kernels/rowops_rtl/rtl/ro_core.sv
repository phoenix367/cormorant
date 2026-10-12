// ---------------------------------------------------------------------------
// ro_core — the RowOpsKernel (doc/kernels/ROWOPS_RTL_KERNEL.md): the row ops
// of the transformer graphs on streams of rows, in fixed point.  The IP top
// level is the Verilog wrapper RowOpsKernel.v (generated from this header by
// scripts/gen_top_wrapper.py); all logic lives here.
//
//   ap_start ─► config ─► [P load: rd_port B ─► ro_param]
//                      ─► rd_port A (ARID 0) ─┐
//                         rd_port B (ARID 1) ─┴► ro_stream / ro_norm / ro_rope / ro_xpose ─► wr_port
//                         (one AXI read channel through ro_rd_arb; one write channel)
// ---------------------------------------------------------------------------
module ro_core #(
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
  parameter int C_M_AXI_GMEM0_CACHE_VALUE  = 3
) (
  input  logic ap_clk,
  input  logic ap_rst_n,

  // gmem0: streams A and B (read, ARID 0 / 1), stream C (write)
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
  import ro_pkg::*;

  if (C_M_AXI_GMEM0_DATA_WIDTH != BW || C_M_AXI_GMEM0_ADDR_WIDTH != 64 ||
      C_M_AXI_GMEM0_ID_WIDTH != 1 || C_S_AXI_CTRL_DATA_WIDTH != 32 ||
      C_S_AXI_CTRL_ADDR_WIDTH != 7) begin : g_bad_params
    $error("ro_core: unsupported interface parameters");
  end

  // The reset, registered: one copy for the control slave, one for the rest
  // (keep: the kernels' copies must not be merged across the device).
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
  logic [31:0] r_op, r_n, r_rows, r_blocks, r_as1, r_as2, r_bs1, r_bs2, r_cs1, r_cs2;
  logic [31:0] r_aux, r_bqsh, r_wlo, r_whi;
  logic [63:0] r_a, r_b, r_c, r_p, r_eps;

  ro_ctrl_s_axi u_ctrl (
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
    .op (r_op), .n (r_n), .rows (r_rows), .blocks (r_blocks),
    .a (r_a), .a_s1 (r_as1), .a_s2 (r_as2), .b (r_b), .b_s1 (r_bs1), .b_s2 (r_bs2),
    .c (r_c), .c_s1 (r_cs1), .c_s2 (r_cs2), .p (r_p), .eps (r_eps), .aux (r_aux),
    .bq_sh (r_bqsh), .win_lo (r_wlo), .win_hi (r_whi)
  );

  // Job configuration ------------------------------------------------------------------
  typedef enum logic [3:0] {T_IDLE, T_LAT, T_CFG0, T_CFG1, T_CFG2, T_PLD, T_PWAIT, T_RUN, T_DONE} tst_t;
  tst_t        tstate;
  logic [31:0] j_op, j_n, j_rows, j_blocks, j_as1, j_as2, j_bs1, j_bs2, j_cs1, j_cs2, j_aux, j_bqsh;
  logic [31:0] j_wlo, j_whi;
  logic [63:0] j_a, j_b, j_c, j_p, j_eps;
  logic [3:0]  opc;
  logic [1:0]  ak, bk, ck;
  logic        f_ben, f_pbc, f_pkeep, f_relu, f_ln;
  logic        is_stream, is_norm, is_rope, is_xpose, is_tload, known, xp_contig;
  logic [35:0] ln_prod;                       // TRANSPOSE: L * R (T_CFG0)
  logic [31:0] lr_words;                      // TRANSPOSE: words of a tile's contiguous run
  logic [3:0]  lr_tail;
  logic        go;                            // the job has work
  logic [31:0] nrows;                         // rows * blocks (low 32 bits)
  logic [31:0] wa, wb, wc;                    // words per row of A, B, C
  logic [3:0]  ta, tb, tc;                    // tails
  logic        job_start, start_q, start_q2, job_done;
  logic        pld_start, pld_q, pld_q2;      // the P load's port start
  geom_t       ga, gb, gc, gp, gb_cur;

  function automatic void row_words(input logic [1:0] kind, input logic [31:0] n,
                                    output logic [31:0] words, output logic [3:0] tail);
    if (kind == K_I16) begin
      words = (n + 32'd7) >> 3;
      tail  = (n[2:0] == 3'd0) ? 4'd8 : {1'b0, n[2:0]};
    end else begin
      words = (n + 32'd3) >> 2;
      tail  = (n[1:0] == 2'd0) ? 4'd8 : {1'b0, n[1:0], 1'b0};
    end
  endfunction

  function automatic geom_t mk_geom(input logic en, input logic [63:0] base, input logic [31:0] s1,
                                    input logic [31:0] c1, input logic [31:0] s2, input logic [31:0] c2,
                                    input logic [31:0] words, input logic [3:0] tail,
                                    input logic [31:0] win_lo, input logic [31:0] win_hi,
                                    input logic [31:0] n_tiles = 32'd1, input logic [31:0] tile_b = 32'd0);
    geom_t g;
    g.en        = en;
    g.base_w    = base[63:4];
    g.n_blk     = c2;
    g.blk_w     = {4'b0, s2[31:4]};
    g.n_tiles   = n_tiles;
    g.tile_w    = {4'b0, tile_b[31:4]};
    g.n_runs    = c1;
    g.run_words = words;
    g.stride_w  = {4'b0, s1[31:4]};
    g.tail      = tail;
    g.replay    = 1'b0;
    g.reps      = 32'd1;
    g.win_lo    = win_lo;
    g.win_hi    = win_hi;
    return g;
  endfunction

  // rows * blocks (low 32 bits), as the other kernels' products: registered stages
  logic [31:0] m0, s0;
  logic [15:0] m1, m2, s12;
  always_ff @(posedge clk) begin
    m0  <= j_rows[15:0] * j_blocks[15:0];
    m1  <= 16'(j_rows[15:0] * j_blocks[31:16]);
    m2  <= 16'(j_rows[31:16] * j_blocks[15:0]);
    s0  <= m0;
    s12 <= m1 + m2;
    nrows <= s0 + {s12, 16'b0};
  end

  logic ld_j;
  (* keep = "true" *) logic ld_g;
  logic prm_busy;

  // ROPE's segments per row, n / HD: a restoring divider, one bit per cycle, started
  // with the geometries (ld_g); T_CFG2 waits for it (32 cycles, ROPE jobs only).
  logic        div_busy;
  logic [5:0]  div_cnt;
  logic [32:0] div_rem;
  logic [31:0] div_q, div_num, div_den;
  always_ff @(posedge clk) begin
    if (rst) begin
      div_busy <= 1'b0;
    end else if (ld_g && is_rope) begin
      div_busy <= (j_aux[11:0] != '0);
      div_cnt  <= '0;
      div_rem  <= '0;
      div_q    <= '0;
      div_num  <= j_n;
      div_den  <= 32'(j_aux[11:0]);
    end else if (div_busy) begin
      logic [32:0] t;
      t = {div_rem[31:0], div_num[31]};
      div_num <= {div_num[30:0], 1'b0};
      if (t >= {1'b0, div_den}) begin
        div_rem <= t - {1'b0, div_den};
        div_q   <= {div_q[30:0], 1'b1};
      end else begin
        div_rem <= t;
        div_q   <= {div_q[30:0], 1'b0};
      end
      div_cnt <= div_cnt + 6'd1;
      if (div_cnt == 6'd31) div_busy <= 1'b0;
    end
  end

  always_ff @(posedge clk) begin
    if (rst) begin
      tstate    <= T_IDLE;
      job_start <= 1'b0;
      pld_start <= 1'b0;
      ld_j      <= 1'b0;
      ld_g      <= 1'b0;
    end else begin
      job_start <= 1'b0;
      pld_start <= 1'b0;
      ld_j      <= (tstate == T_IDLE) && ap_start;
      ld_g      <= (tstate == T_CFG1);
      case (tstate)
        T_IDLE:  if (ap_start) tstate <= T_LAT;
        T_LAT:   tstate <= T_CFG0;
        T_CFG0:  tstate <= T_CFG1;
        T_CFG1:  tstate <= T_CFG2;
        T_CFG2:  begin                          // geometries loaded (ld_g); nrows valid
          if (div_busy || (ld_g && is_rope)) begin
            tstate <= T_CFG2;                   // ROPE: n / HD in flight (started this cycle)
          end else if (go && !f_pkeep && (is_stream || is_norm || is_rope)) begin
            pld_start <= 1'b1;
            tstate    <= T_PLD;
          end else begin
            job_start <= 1'b1;
            tstate    <= T_RUN;
          end
        end
        T_PLD:   tstate <= T_PWAIT;
        T_PWAIT: if (!pld_q && !pld_q2 && !prm_busy && b_idle) begin
          job_start <= 1'b1;
          tstate    <= T_RUN;
        end
        T_RUN:   if (job_done) tstate <= T_DONE;
        T_DONE:  tstate <= T_IDLE;
        default: tstate <= T_IDLE;
      endcase
    end
  end

  always_ff @(posedge clk) begin
    if (ld_j) begin
      j_op <= r_op; j_n <= r_n; j_rows <= r_rows; j_blocks <= r_blocks;
      j_a <= r_a; j_as1 <= r_as1; j_as2 <= r_as2;
      j_b <= r_b; j_bs1 <= r_bs1; j_bs2 <= r_bs2;
      j_c <= r_c; j_cs1 <= r_cs1; j_cs2 <= r_cs2;
      j_p <= r_p; j_eps <= r_eps; j_aux <= r_aux; j_bqsh <= r_bqsh;
      j_wlo <= r_wlo; j_whi <= r_whi;
    end
    if (tstate == T_CFG0) begin
      opc     <= j_op[3:0];
      ak      <= j_op[5:4];
      bk      <= j_op[7:6];
      ck      <= j_op[9:8];
      f_ben   <= j_op[F_B_EN];
      f_pbc   <= j_op[F_P_BCAST];
      f_pkeep <= j_op[F_P_KEEP];
      f_relu  <= j_op[F_RELU];
      f_ln    <= j_op[F_NORM_LN];
      is_stream <= (j_op[3:0] == OP_COPY) || (j_op[3:0] == OP_ADDX) || (j_op[3:0] == OP_DEQUANT) ||
                   (j_op[3:0] == OP_MULACT);
      is_norm   <= (j_op[3:0] == OP_NORM);
      is_rope   <= (j_op[3:0] == OP_ROPE);
      is_xpose  <= (j_op[3:0] == OP_XPOSE);
      is_tload  <= (j_op[3:0] == OP_TLOAD);
      ln_prod   <= 36'(j_n) * 36'(j_aux[3:0]);
      row_words(j_op[5:4], j_n, wa, ta);
      row_words(j_op[7:6], j_n, wb, tb);
      row_words(j_op[9:8], j_n, wc, tc);
    end
    if (tstate == T_CFG1) begin
      known <= is_stream || is_norm || is_rope || is_xpose || is_tload;
      go    <= (is_stream || is_norm || is_rope || is_xpose || is_tload) && (j_n != '0) && (j_rows != '0) &&
               (j_blocks != '0) && (j_n <= 32'(N_MAX)) &&
               (!is_xpose || (j_aux[3:0] != '0 && j_n <= 32'd1024));
      xp_contig <= (j_bs2 == {27'b0, j_aux[3:0], 1'b0});
      lr_words  <= 32'((ln_prod + 36'd7) >> 3);
      lr_tail   <= (ln_prod[2:0] == 3'd0) ? 4'd8 : {1'b0, ln_prod[2:0]};
    end
    if (ld_g) begin
      if (is_xpose) begin
        // A: blocks / tiles / the tile's R rows at b_s1; C: one run of L*R elements per
        // tile (contiguous granules), or nothing (the transposer's own bursts)
        ga <= mk_geom(go, j_a, j_bs1, 32'(j_aux[3:0]), j_as2, j_blocks, wa, ta, 32'd0, 32'hFFFFFFFF,
                      j_rows, j_as1);
        gc <= mk_geom(go && xp_contig, j_c, j_cs1, j_rows, j_cs2, j_blocks, lr_words, lr_tail,
                      32'd0, 32'hFFFFFFFF);
        gb <= '0;                                   // no stream B (not the previous job's)
      end else if (is_tload) begin
        // the table image: rows of A words (its kinds int32), no B, no C
        ga <= mk_geom(go, j_a, j_as1, j_rows, j_as2, j_blocks, wa, ta, 32'd0, 32'hFFFFFFFF);
        gb <= '0;
        gc <= '0;
      end else begin
      ga <= mk_geom(go, j_a, j_as1, j_rows, j_as2, j_blocks, wa, ta, j_wlo, j_whi);
      gb <= mk_geom(go && f_ben, j_b, j_bs1, j_rows, j_bs2, j_blocks,
                    is_rope ? 32'(j_aux[11:1]) : wb, 4'd8, 32'd0, 32'hFFFFFFFF);
      gc <= mk_geom(go, j_c, j_cs1, j_rows, j_cs2, j_blocks, wc, tc, 32'd0, 32'hFFFFFFFF);
      end
      gp <= mk_geom(go, j_p, 32'd0, 32'd1, 32'd0, 32'd1, f_pbc ? 32'd1 : j_n, 4'd8, 32'd0, 32'hFFFFFFFF);
    end
  end

  // Port B serves the parameter load first (gp), then stream B (gb): its
  // geometry register follows the phase.
  always_ff @(posedge clk) begin
    if (pld_start)      gb_cur <= gp;
    else if (job_start) gb_cur <= gb;
  end

  // Every unit starts a job from reset: the job reset is registered and copied per unit.
  (* keep = "true" *) logic urst_a, urst_b, urst_c, urst_d, urst_p;
  always_ff @(posedge clk) begin
    start_q  <= !rst && job_start;
    start_q2 <= !rst && start_q;
    pld_q    <= !rst && pld_start;
    pld_q2   <= !rst && pld_q;
    urst_a   <= rst || job_start;
    urst_b   <= rst || job_start || pld_start;
    urst_c   <= rst || job_start;
    urst_d   <= rst || job_start;
    urst_p   <= rst || pld_start;
  end

  assign ap_idle = (tstate == T_IDLE);
  assign ap_done = (tstate == T_DONE);

  // Ports ------------------------------------------------------------------------------
  logic          a_valid, a_ready, b_valid, b_ready, w_valid, w_ready;
  logic [BW-1:0] a_data, b_data, w_data;
  logic [BW/8-1:0] w_strb;
  logic          a_idle, b_idle, wr_idle;
  logic          a_arvalid, a_arready, b_arvalid, b_arready, a_rvalid, a_rready, b_rvalid, b_rready;
  logic [63:0]   a_araddr, b_araddr;
  logic [7:0]    a_arlen, b_arlen;

  ro_rd_port u_rd_a (
    .clk, .rst (urst_a), .start (start_q2), .g (ga),
    .arvalid (a_arvalid), .arready (a_arready), .araddr (a_araddr), .arlen (a_arlen),
    .rvalid  (a_rvalid),  .rready  (a_rready),
    .rdata   (m_axi_gmem0_RDATA), .rlast (m_axi_gmem0_RLAST),
    .out_valid (a_valid), .out_ready (a_ready), .out_data (a_data),
    .idle (a_idle)
  );

  ro_rd_port u_rd_b (
    .clk, .rst (urst_b), .start (start_q2 || pld_q2), .g (gb_cur),
    .arvalid (b_arvalid), .arready (b_arready), .araddr (b_araddr), .arlen (b_arlen),
    .rvalid  (b_rvalid),  .rready  (b_rready),
    .rdata   (m_axi_gmem0_RDATA), .rlast (m_axi_gmem0_RLAST),
    .out_valid (b_valid), .out_ready (b_ready), .out_data (b_data),
    .idle (b_idle)
  );

  ro_rd_arb u_arb (
    .clk, .rst,
    .a_arvalid, .a_arready, .a_araddr, .a_arlen, .a_rvalid, .a_rready,
    .b_arvalid, .b_arready, .b_araddr, .b_arlen, .b_rvalid, .b_rready,
    .arvalid (m_axi_gmem0_ARVALID), .arready (m_axi_gmem0_ARREADY),
    .araddr  (m_axi_gmem0_ARADDR),  .arlen   (m_axi_gmem0_ARLEN),  .arid (m_axi_gmem0_ARID[0]),
    .rvalid  (m_axi_gmem0_RVALID),  .rready  (m_axi_gmem0_RREADY), .rid  (m_axi_gmem0_RID[0])
  );

  logic        xp_ext_en, xp_ext_valid, xp_ext_ready, xp_ext_done;
  logic [59:0] xp_ext_waddr;
  logic [8:0]  xp_ext_len;
  ro_wr_port u_wr (
    .clk, .rst (urst_c), .start (start_q2), .g (gc),
    .ext_en (is_xpose && !xp_contig), .ext_valid (xp_ext_valid), .ext_ready (xp_ext_ready),
    .ext_waddr (xp_ext_waddr), .ext_len (xp_ext_len), .ext_done (xp_ext_done),
    .in_valid (w_valid), .in_ready (w_ready), .in_data (w_data), .in_strb (w_strb),
    .awvalid (m_axi_gmem0_AWVALID), .awready (m_axi_gmem0_AWREADY),
    .awaddr  (m_axi_gmem0_AWADDR),  .awlen   (m_axi_gmem0_AWLEN),
    .wvalid  (m_axi_gmem0_WVALID),  .wready  (m_axi_gmem0_WREADY),
    .wdata   (m_axi_gmem0_WDATA),   .wstrb   (m_axi_gmem0_WSTRB), .wlast (m_axi_gmem0_WLAST),
    .bvalid  (m_axi_gmem0_BVALID),  .bready  (m_axi_gmem0_BREADY),
    .idle (wr_idle)
  );

  // Parameters -------------------------------------------------------------------------
  // During the load, port B's words go to ro_param; during the job, to the datapath.
  logic   loading;
  logic   prm_in_ready;
  logic [PIW-1:0] prm_idx;
  param_t prm_q [PB];

  // The table load (OP_TLOAD, ROWOPS_RTL_KERNEL.md §1.4): A words 0 .. 2047 hold two
  // sigmoid entries each, 2048 .. 4095 four rsqrt entries each (two table words);
  // every word takes two beats, the second consumes it.  Words past the image are
  // consumed and ignored.
  logic             tl_run, tl_beat, tl_a_ready, tl_idle;
  logic [12:0]      tl_k;
  logic             tl_sig_we, tl_rsq_we;
  logic [11:0]      tl_addr;
  logic [SIG_W-1:0] tl_sig_d;
  logic [63:0]      tl_rsq_d;
  assign tl_a_ready = is_tload && (!tl_run || tl_beat);
  assign tl_idle    = !tl_sig_we && !tl_rsq_we;
  always_ff @(posedge clk) begin
    if (urst_d) begin
      tl_run  <= 1'b0;
      tl_beat <= 1'b0;
    end else if (start_q2) begin
      tl_run  <= go && is_tload;
      tl_beat <= 1'b0;
      tl_k    <= '0;
    end else if (tl_run && a_valid && is_tload) begin
      tl_beat <= !tl_beat;
      if (tl_beat) begin
        tl_k <= tl_k + 13'd1;
        if (tl_k == 13'd4095) tl_run <= 1'b0;
      end
    end
    tl_sig_we <= tl_run && a_valid && is_tload && !tl_k[11] && !tl_k[12];
    tl_rsq_we <= tl_run && a_valid && is_tload &&  tl_k[11] && !tl_k[12];
    tl_addr   <= {tl_k[10:0], tl_beat};
    tl_sig_d  <= tl_beat ? a_data[64 +: SIG_W] : a_data[0 +: SIG_W];
    tl_rsq_d  <= tl_beat ? a_data[127:64] : a_data[63:0];
  end
  always_ff @(posedge clk) begin
    if (rst) loading <= 1'b0;
    else if (pld_start) loading <= 1'b1;
    else if (job_start) loading <= 1'b0;
  end

  ro_param u_prm (
    .clk, .rst (urst_p),
    .ld_start (pld_q2), .ld_bcast (f_pbc), .ld_n (f_pbc ? 32'd1 : j_n), .ld_busy (prm_busy),
    .in_valid (b_valid && loading), .in_ready (prm_in_ready), .in_data (b_data),
    .rd_idx (prm_idx), .rd_q (prm_q)
  );

  // Datapath -----------------------------------------------------------------------------
  logic st_a_ready, st_b_ready, st_idle, st_valid;
  logic [BW-1:0] st_data;

  ro_stream u_st (
    .clk, .rst (urst_d), .start (start_q2), .en (go && is_stream),
    .op (opc), .ak, .bk, .ck, .b_en (f_ben), .relu (f_relu), .bq_sh (j_bqsh[5:0]),
    .n (j_n), .nrows,
    .a_valid (a_valid && is_stream), .a_ready (st_a_ready), .a_data,
    .b_valid (b_valid && !loading && is_stream), .b_ready (st_b_ready), .b_data,
    .prm_idx (st_prm_idx), .prm_q,
    .tbl_we (tl_sig_we), .tbl_addr (tl_addr), .tbl_data (tl_sig_d),
    .c_valid (st_valid), .c_ready (w_ready), .c_data (st_data),
    .idle (st_idle)
  );

  logic nm_a_ready, nm_idle, nm_valid;
  logic [BW-1:0] nm_data;
  logic [PIW-1:0] st_prm_idx, nm_prm_idx;

  ro_norm u_nm (
    .clk, .rst (urst_d), .start (start_q2), .en (go && is_norm),
    .ln (f_ln), .a32 (ak == K_I32), .n (j_n), .nrows, .words (wa), .eps (j_eps), .bq_sh (j_bqsh[5:0]),
    .a_valid (a_valid && is_norm), .a_ready (nm_a_ready), .a_data,
    .prm_idx (nm_prm_idx), .prm_q,
    .tbl_we (tl_rsq_we), .tbl_addr (tl_addr), .tbl_data (tl_rsq_d),
    .c_valid (nm_valid), .c_ready (w_ready), .c_data (nm_data),
    .idle (nm_idle)
  );

  logic rp_a_ready, rp_b_ready, rp_idle, rp_valid;
  logic [BW-1:0] rp_data;
  logic [PIW-1:0] rp_prm_idx;

  ro_rope u_rp (
    .clk, .rst (urst_d), .start (start_q2), .en (go && is_rope),
    .n (j_n), .nrows, .hd (j_aux[11:0]), .segs (div_q),
    .a_valid (a_valid && is_rope), .a_ready (rp_a_ready), .a_data,
    .b_valid (b_valid && !loading && is_rope), .b_ready (rp_b_ready), .b_data,
    .prm_idx (rp_prm_idx), .prm_q,
    .c_valid (rp_valid), .c_ready (w_ready), .c_data (rp_data),
    .idle (rp_idle)
  );

  logic xp_a_ready, xp_idle, xp_valid;
  logic [BW-1:0] xp_data;
  logic [BW/8-1:0] xp_strb;

  ro_xpose u_xp (
    .clk, .rst (urst_d), .start (start_q2), .en (go && is_xpose),
    .L (j_n), .R (j_aux[3:0]), .c1 (j_rows), .c2 (j_blocks), .c_base (j_c), .c_s1 (j_cs1), .c_s2 (j_cs2),
    .g_stride (j_bs2),
    .a_valid (a_valid && is_xpose), .a_ready (xp_a_ready), .a_data,
    .c_valid (xp_valid), .c_ready (w_ready), .c_data (xp_data), .c_strb (xp_strb),
    .ext_en (xp_ext_en), .ext_valid (xp_ext_valid), .ext_ready (xp_ext_ready),
    .ext_waddr (xp_ext_waddr), .ext_len (xp_ext_len), .ext_done (xp_ext_done),
    .idle (xp_idle)
  );

  assign prm_idx = is_norm ? nm_prm_idx : is_rope ? rp_prm_idx : st_prm_idx;
  assign a_ready = is_norm ? nm_a_ready : is_rope ? rp_a_ready : is_xpose ? xp_a_ready :
                   is_tload ? tl_a_ready : st_a_ready;
  assign b_ready = loading ? prm_in_ready : is_rope ? rp_b_ready : st_b_ready;
  assign w_valid = is_norm ? nm_valid : is_rope ? rp_valid : is_xpose ? xp_valid : is_tload ? 1'b0 : st_valid;
  assign w_data  = is_norm ? nm_data : is_rope ? rp_data : is_xpose ? xp_data : st_data;
  assign w_strb  = is_xpose ? xp_strb : '1;

  assign job_done = (tstate == T_RUN) && !job_start && !start_q && !start_q2 &&
                    a_idle && b_idle && st_idle && nm_idle && rp_idle && xp_idle && tl_idle && wr_idle;

  // AXI constant fields ------------------------------------------------------------------
  assign m_axi_gmem0_ARSIZE   = 3'b100;
  assign m_axi_gmem0_ARBURST  = 2'b01;
  assign m_axi_gmem0_ARLOCK   = 2'b00;
  assign m_axi_gmem0_ARCACHE  = 4'(C_M_AXI_GMEM0_CACHE_VALUE);
  assign m_axi_gmem0_ARPROT   = 3'(C_M_AXI_GMEM0_PROT_VALUE);
  assign m_axi_gmem0_ARQOS    = 4'b0;
  assign m_axi_gmem0_ARREGION = 4'b0;
  assign m_axi_gmem0_ARUSER   = C_M_AXI_GMEM0_ARUSER_WIDTH'(C_M_AXI_GMEM0_USER_VALUE);
  assign m_axi_gmem0_AWID     = '0;
  assign m_axi_gmem0_AWSIZE   = 3'b100;
  assign m_axi_gmem0_AWBURST  = 2'b01;
  assign m_axi_gmem0_AWLOCK   = 2'b00;
  assign m_axi_gmem0_AWCACHE  = 4'(C_M_AXI_GMEM0_CACHE_VALUE);
  assign m_axi_gmem0_AWPROT   = 3'(C_M_AXI_GMEM0_PROT_VALUE);
  assign m_axi_gmem0_AWQOS    = 4'b0;
  assign m_axi_gmem0_AWREGION = 4'b0;
  assign m_axi_gmem0_AWUSER   = C_M_AXI_GMEM0_AWUSER_WIDTH'(C_M_AXI_GMEM0_USER_VALUE);
  assign m_axi_gmem0_WID      = '0;
  assign m_axi_gmem0_WUSER    = C_M_AXI_GMEM0_WUSER_WIDTH'(C_M_AXI_GMEM0_USER_VALUE);

  logic unused_top;
  assign unused_top = ^{m_axi_gmem0_RUSER, m_axi_gmem0_RRESP, m_axi_gmem0_BRESP, m_axi_gmem0_BID,
                        m_axi_gmem0_BUSER, j_aux[31:12], tb, wb, xp_ext_en,
                        is_xpose, known, j_bqsh[31:6]};

endmodule
