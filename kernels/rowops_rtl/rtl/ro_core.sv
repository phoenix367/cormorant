// Copyright 2026 Ivan Gubochkin
// SPDX-License-Identifier: Apache-2.0

// ---------------------------------------------------------------------------
// ro_core — the RowOpsKernel (doc/kernels/ROWOPS_RTL_KERNEL.md): the row ops
// of the transformer graphs on streams of rows, in fixed point.  The IP top
// level is the Verilog wrapper RowOpsKernel.v (generated from this header by
// scripts/gen_top_wrapper.py); all logic lives here.
//
//   ap_start ─► [chain: descriptors, rd_port B ─► descriptor RAM] ─► for each job:
//               config ─► [P load: rd_port B ─► ro_param]
//                      ─► rd_port A (ARID 0) ─┐
//                         rd_port B (ARID 1) ─┴► ro_stream / ro_norm / ro_rope / ro_xpose / ro_fold ─► wr_port
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
  logic [31:0] r_aux, r_bqsh, r_wlo, r_whi, r_chain_n;
  logic [63:0] r_a, r_b, r_c, r_p, r_eps, r_chain;

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
    .bq_sh (r_bqsh), .win_lo (r_wlo), .win_hi (r_whi), .chain (r_chain), .chain_n (r_chain_n)
  );

  // The ports' stream handshakes (declared here: the job FSM below uses them)
  logic          a_valid, a_ready, b_valid, b_ready, w_valid, w_ready;
  logic [BW-1:0] a_data, b_data, w_data;
  logic [BW/8-1:0] w_strb;
  logic          a_idle, b_idle, wr_idle;

  // Job configuration ------------------------------------------------------------------
  // T_IDLE -> T_LAT (the job registers j_* load) -> T_CFG0 .. T_CFG2 -> [T_PLD, T_PWAIT]
  // -> T_RUN -> T_DONE (ap_done).  A chain (chain_n != 0, §1.6) goes T_LAT -> T_DF /
  // T_DFW (the descriptors fetched into the descriptor RAM) -> T_DLD -> T_LAT (j_* load
  // from the RAM's read at the job index, which moves on) -> ... -> T_RUN, and from T_RUN
  // back to T_DLD for the next job; T_DONE only after the last.
  typedef enum logic [3:0] {T_IDLE, T_LAT, T_CFG0, T_CFG1, T_CFG2, T_PLD, T_PWAIT, T_RUN, T_DONE,
                            T_DF, T_DFW, T_DLD} tst_t;
  tst_t        tstate;
  logic [31:0] j_op, j_n, j_rows, j_blocks, j_as1, j_as2, j_bs1, j_bs2, j_cs1, j_cs2, j_aux, j_bqsh;
  logic [31:0] j_wlo, j_whi;
  logic [63:0] j_a, j_b, j_c, j_p, j_eps;
  logic [3:0]  opc;
  logic [1:0]  ak, bk, ck;
  logic        f_ben, f_pbc, f_pkeep, f_relu, f_ln, f_scale;
  logic        is_stream, is_norm, is_rope, is_xpose, is_tload, is_fold, known, xp_contig;
  logic [35:0] ln_prod;                       // TRANSPOSE: L * R (T_CFG0)
  logic [31:0] lr_words;                      // TRANSPOSE: words of a tile's contiguous run
  logic [3:0]  lr_tail;
  logic        go;                            // the job has work
  logic [31:0] nrows;                         // rows * blocks (low 32 bits)
  logic [31:0] wa, wb, wc;                    // words per row of A, B, C
  logic [3:0]  ta, tb, tc;                    // tails
  logic        job_start, start_q, start_q2, job_done;
  logic        pld_start, pld_q, pld_q2;      // the P load's port start
  geom_t       ga, gc, gb_cur;
  logic        gb_en;                         // stream B is read (ld_g)
  logic [31:0] gb_words, gp_words;            // words per row of stream B, entries of the P load (T_CFG1)

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

  // ROPE's segments per row, n / HD, and FOLD's reciprocal ceil(2^24 / m) =
  // (2^24 + m - 1) / m: a restoring divider, one bit per cycle, started with the
  // geometries (ld_g); T_CFG2 waits for it (32 cycles, ROPE and FOLD jobs only).
  // (the divisor is below 2^12, so is the remainder)
  logic        div_busy;
  logic [5:0]  div_cnt;
  logic [11:0] div_rem, div_den;
  logic [31:0] div_q, div_num;
  always_ff @(posedge clk) begin
    if (rst) begin
      div_busy <= 1'b0;
    end else if (ld_g && (is_rope || is_fold)) begin
      div_busy <= is_fold ? (j_aux[10:4] != '0) : (j_aux[11:0] != '0);
      div_cnt  <= '0;
      div_rem  <= '0;
      div_q    <= '0;
      div_num  <= is_fold ? 32'h0100_0000 + 32'(j_aux[10:4]) - 32'd1 : j_n;
      div_den  <= is_fold ? 12'(j_aux[10:4]) : j_aux[11:0];
    end else if (div_busy) begin
      logic [12:0] t;
      t = {div_rem, div_num[31]};
      div_num <= {div_num[30:0], 1'b0};
      if (t >= {1'b0, div_den}) begin
        div_rem <= 12'(t - {1'b0, div_den});
        div_q   <= {div_q[30:0], 1'b1};
      end else begin
        div_rem <= t[11:0];
        div_q   <= {div_q[30:0], 1'b0};
      end
      div_cnt <= div_cnt + 6'd1;
      if (div_cnt == 6'd31) div_busy <= 1'b0;
    end
  end

  // Descriptor chains (§1.6): the chain's state, latched with ap_start.
  logic        ch_on;                         // this start runs a chain (chain_n != 0)
  logic        ch_pend;                       // its descriptors are still to be fetched
  logic        ch_more;                       // a job of the chain follows the current one
  logic [3:0]  ch_last;                       // min(chain_n, CHAIN_MAX) - 1
  logic [59:0] ch_base;                       // word address of the descriptor list
  logic [3:0]  jidx;                          // the descriptor RAM's read index: the next job to load
  logic        dfe_start, dfe_q, dfe_q2;      // the descriptor fetch's port start
  logic        dfetch;                        // port B's words are descriptor beats
  logic [6:0]  dw;                            // beats taken: descriptor [6:3], beat [2:0]
  geom_t       gd;                            // the list: one run of min(chain_n, 16) * 8 words

  always_ff @(posedge clk) begin
    if (rst) begin
      tstate    <= T_IDLE;
      job_start <= 1'b0;
      pld_start <= 1'b0;
      dfe_start <= 1'b0;
      ld_j      <= 1'b0;
      ld_g      <= 1'b0;
      ch_on     <= 1'b0;
      ch_pend   <= 1'b0;
      ch_more   <= 1'b0;
      jidx      <= '0;
    end else begin
      job_start <= 1'b0;
      pld_start <= 1'b0;
      dfe_start <= 1'b0;
      ld_j      <= ((tstate == T_IDLE) && ap_start) || (tstate == T_DLD);
      ld_g      <= (tstate == T_CFG1);
      if ((tstate == T_IDLE) && ap_start) begin
        ch_on   <= (r_chain_n != '0);
        ch_pend <= (r_chain_n != '0);
        ch_more <= 1'b0;
        ch_last <= (r_chain_n > 32'(CHAIN_MAX)) ? 4'(CHAIN_MAX - 1) : 4'(r_chain_n - 32'd1);
        ch_base <= r_chain[63:4];
        jidx    <= '0;
      end
      // A chain's job loads its descriptor (ld_j in T_LAT, from the RAM's read at jidx): the
      // read index moves on to the next one at the same edge, so job_done reaches the state
      // alone, as for a single job.
      if ((tstate == T_LAT) && ch_on && !ch_pend) begin
        jidx    <= jidx + 4'd1;
        ch_more <= jidx != ch_last;
      end
      case (tstate)
        T_IDLE:  if (ap_start) tstate <= T_LAT;
        T_LAT:   if (ch_pend) begin                 // a chain: its descriptors first
          ch_pend   <= 1'b0;
          dfe_start <= 1'b1;
          tstate    <= T_DF;
        end else begin
          tstate <= T_CFG0;
        end
        T_DF:    tstate <= T_DFW;
        T_DFW:   if (!dfe_q && !dfe_q2 && !dfetch && b_idle) tstate <= T_DLD;
        T_DLD:   tstate <= T_LAT;                   // ld_j next (jidx stable from here to T_LAT's end)
        T_CFG0:  tstate <= T_CFG1;
        T_CFG1:  tstate <= T_CFG2;
        T_CFG2:  begin                          // geometries loaded (ld_g); nrows valid
          if (div_busy || (ld_g && (is_rope || is_fold))) begin
            tstate <= T_CFG2;                   // ROPE: n / HD, FOLD: 2^24 / m in flight (started this cycle)
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
        T_RUN:   if (job_done) tstate <= ch_more ? T_DLD : T_DONE;   // every unit idle, every B response back
        T_DONE:  tstate <= T_IDLE;
        default: tstate <= T_IDLE;
      endcase
    end
  end

  // The descriptor fetch: port B reads the list as one run (bursts of <= 64 beats within
  // 4 KiB, ARID 1) while `loading` keeps its words from the units and ro_param (whose load
  // is idle: no entry is written); beats 0 .. 5 of every descriptor go to the RAM as they
  // are taken (the last write, beat 5 of the last descriptor, comes two beats before
  // dfetch drops).
  assign gd = mk_geom(1'b1, {ch_base, 4'b0}, 32'd0, 32'd1, 32'd0, 32'd1, {24'b0, ({1'b0, ch_last} + 5'd1), 3'b0},
                      4'd8, 32'd0, 32'hFFFFFFFF);
  always_ff @(posedge clk) begin
    if (rst) begin
      dfetch <= 1'b0;
    end else begin
      if (dfe_start) begin
        dfetch <= 1'b1;
        dw     <= '0;
      end else if (dfetch && b_valid) begin           // b_ready is high while fetching
        dw <= dw + 7'd1;
        if (dw == {ch_last, 3'b111}) dfetch <= 1'b0;
      end
    end
  end

  // The descriptor RAM: one LUTRAM of CHAIN_MAX x 128 bits per used beat (its own write
  // enable), read asynchronously at the job index; the job registers load from it in T_LAT
  // (no write after the fetch, jidx stable from T_DLD on).
  logic [BW-1:0] dq [DESC_USED];
  for (genvar k = 0; k < DESC_USED; k++) begin : g_dram
    (* ram_style = "distributed" *) logic [BW-1:0] mem [CHAIN_MAX];
    always_ff @(posedge clk)
      if (dfetch && b_valid && dw[2:0] == 3'(k)) mem[dw[6:3]] <= b_data;
    assign dq[k] = mem[jidx];
  end

  always_ff @(posedge clk) begin
    if (ld_j) begin
      if (ch_on) begin                              // the chain's job: its descriptor words
        j_op <= dq[0][31:0];  j_n   <= dq[0][63:32]; j_rows <= dq[0][95:64]; j_blocks <= dq[0][127:96];
        j_a  <= dq[1][63:0];  j_as1 <= dq[1][95:64]; j_as2  <= dq[1][127:96];
        j_b  <= dq[2][63:0];  j_bs1 <= dq[2][95:64]; j_bs2  <= dq[2][127:96];
        j_c  <= dq[3][63:0];  j_cs1 <= dq[3][95:64]; j_cs2  <= dq[3][127:96];
        j_p  <= dq[4][63:0];  j_eps <= dq[4][127:64];
        j_aux <= dq[5][31:0]; j_bqsh <= dq[5][63:32]; j_wlo <= dq[5][95:64]; j_whi <= dq[5][127:96];
      end else begin
        j_op <= r_op; j_n <= r_n; j_rows <= r_rows; j_blocks <= r_blocks;
        j_a <= r_a; j_as1 <= r_as1; j_as2 <= r_as2;
        j_b <= r_b; j_bs1 <= r_bs1; j_bs2 <= r_bs2;
        j_c <= r_c; j_cs1 <= r_cs1; j_cs2 <= r_cs2;
        j_p <= r_p; j_eps <= r_eps; j_aux <= r_aux; j_bqsh <= r_bqsh;
        j_wlo <= r_wlo; j_whi <= r_whi;
      end
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
      f_scale <= j_op[F_SCALE];
      is_stream <= (j_op[3:0] == OP_COPY) || (j_op[3:0] == OP_ADDX) || (j_op[3:0] == OP_DEQUANT) ||
                   (j_op[3:0] == OP_MULACT) || (j_op[3:0] == OP_GATE);
      is_norm   <= (j_op[3:0] == OP_NORM);
      is_rope   <= (j_op[3:0] == OP_ROPE);
      is_xpose  <= (j_op[3:0] == OP_XPOSE);
      is_tload  <= (j_op[3:0] == OP_TLOAD);
      is_fold   <= (j_op[3:0] == OP_FOLD);
      ln_prod   <= 36'(j_n) * 36'(j_aux[3:0]);
      row_words(j_op[5:4], j_n, wa, ta);
      row_words(j_op[7:6], j_n, wb, tb);
      row_words(j_op[9:8], j_n, wc, tc);
    end
    if (tstate == T_CFG1) begin
      known <= is_stream || is_norm || is_rope || is_xpose || is_tload || is_fold;
      go    <= ((is_stream || is_norm || is_rope || is_xpose || is_tload) && (j_n != '0) && (j_rows != '0) &&
                (j_blocks != '0) && (j_n <= 32'(N_MAX)) &&
                (!is_xpose || (j_aux[3:0] != '0 && j_n <= 32'd1024))) ||
               (is_fold && (j_n != '0) && (j_rows != '0) && (j_blocks != '0) && (j_aux[10:4] != '0));
      xp_contig <= (j_bs2 == {27'b0, j_aux[3:0], 1'b0});
      lr_words  <= 32'((ln_prod + 36'd7) >> 3);
      lr_tail   <= (ln_prod[2:0] == 3'd0) ? 4'd8 : {1'b0, ln_prod[2:0]};
      gb_words  <= is_rope ? 32'(j_aux[11:1]) : wb;
      gp_words  <= f_pbc ? 32'd1 : j_n;
    end
    if (ld_g) begin
      if (is_xpose) begin
        // A: blocks / tiles / the tile's R rows at b_s1; C: one run of L*R elements per
        // tile (contiguous granules), or nothing (the transposer's own bursts)
        ga <= mk_geom(go, j_a, j_bs1, 32'(j_aux[3:0]), j_as2, j_blocks, wa, ta, 32'd0, 32'hFFFFFFFF,
                      j_rows, j_as1);
        gc <= mk_geom(go && xp_contig, j_c, j_cs1, j_rows, j_cs2, j_blocks, lr_words, lr_tail,
                      32'd0, 32'hFFFFFFFF);
      end else if (is_fold) begin
        // the read runs and the write bursts come from ro_fold (external)
        ga <= mk_geom(1'b0, j_a, 32'd0, 32'd1, 32'd0, 32'd1, 32'd1, 4'd8, 32'd0, 32'hFFFFFFFF);
        gc <= '0;
      end else if (is_tload) begin
        // the table image: rows of A words (its kinds int32), no B, no C
        ga <= mk_geom(go, j_a, j_as1, j_rows, j_as2, j_blocks, wa, ta, 32'd0, 32'hFFFFFFFF);
        gc <= '0;
      end else begin
      ga <= mk_geom(go, j_a, j_as1, j_rows, j_as2, j_blocks, wa, ta, j_wlo, j_whi);
      gc <= mk_geom(go, j_c, j_cs1, j_rows, j_cs2, j_blocks, wc, tc, 32'd0, 32'hFFFFFFFF);
      end
      // no stream B in TRANSPOSE, FOLD and TLOAD (not the previous job's)
      gb_en <= go && f_ben && !is_xpose && !is_fold && !is_tload;
    end
  end

  // Port B serves a chain's descriptor fetch (gd), the parameter load, then stream B: its
  // geometry register follows the phase, the P load's and stream B's built from the job
  // registers (stable through the job) and the counts registered in T_CFG1.
  always_ff @(posedge clk) begin
    if (pld_start)
      gb_cur <= mk_geom(go, j_p, 32'd0, 32'd1, 32'd0, 32'd1, gp_words, 4'd8, 32'd0, 32'hFFFFFFFF);
    else if (dfe_start)
      gb_cur <= gd;
    else if (job_start)
      gb_cur <= mk_geom(gb_en, j_b, j_bs1, j_rows, j_bs2, j_blocks, gb_words, 4'd8, 32'd0, 32'hFFFFFFFF);
  end

  // Every unit starts a job from reset: the job reset is registered and copied per unit.
  (* keep = "true" *) logic urst_a, urst_b, urst_c, urst_d, urst_p;
  always_ff @(posedge clk) begin
    start_q  <= !rst && job_start;
    start_q2 <= !rst && start_q;
    pld_q    <= !rst && pld_start;
    pld_q2   <= !rst && pld_q;
    dfe_q    <= !rst && dfe_start;
    dfe_q2   <= !rst && dfe_q;
    urst_a   <= rst || job_start;
    urst_b   <= rst || job_start || pld_start || dfe_start;
    urst_c   <= rst || job_start;
    urst_d   <= rst || job_start;
    urst_p   <= rst || pld_start;
  end

  assign ap_idle = (tstate == T_IDLE);
  assign ap_done = (tstate == T_DONE);

  // Ports ------------------------------------------------------------------------------
  logic          a_arvalid, a_arready, b_arvalid, b_arready, a_rvalid, a_rready, b_rvalid, b_rready;
  logic [63:0]   a_araddr, b_araddr;
  logic [7:0]    a_arlen, b_arlen;

  logic        fd_rr_valid, fd_rr_ready;
  logic [WA-1:0] fd_rr_waddr;
  logic [31:0] fd_rr_words;
  ro_rd_port u_rd_a (
    .clk, .rst (urst_a), .start (start_q2), .g (ga),
    .ext_en (is_fold), .ext_valid (fd_rr_valid), .ext_ready (fd_rr_ready),
    .ext_waddr (60'(fd_rr_waddr)), .ext_words (fd_rr_words),
    .arvalid (a_arvalid), .arready (a_arready), .araddr (a_araddr), .arlen (a_arlen),
    .rvalid  (a_rvalid),  .rready  (a_rready),
    .rdata   (m_axi_gmem0_RDATA), .rlast (m_axi_gmem0_RLAST),
    .out_valid (a_valid), .out_ready (a_ready), .out_data (a_data),
    .idle (a_idle)
  );

  // port B's geometries (gd, gp, gb) have one tile and no window
  logic b_ext_ready;
  ro_rd_port #(.WIN(1'b0), .TILES(1'b0)) u_rd_b (
    .clk, .rst (urst_b), .start (start_q2 || pld_q2 || dfe_q2), .g (gb_cur),
    .ext_en (1'b0), .ext_valid (1'b0), .ext_ready (b_ext_ready), .ext_waddr ('0), .ext_words ('0),
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
  logic [WA-1:0] xp_ext_waddr;
  logic [8:0]  xp_ext_len;
  logic        fd_wb_valid, fd_wb_ready, fd_wb_done, wr_ext_ready;
  logic [WA-1:0] fd_wb_waddr;
  logic [8:0]  fd_wb_len;
  assign xp_ext_ready = wr_ext_ready && !is_fold;
  assign fd_wb_ready  = wr_ext_ready && is_fold;
  ro_wr_port u_wr (
    .clk, .rst (urst_c), .start (start_q2), .g (gc),
    .ext_en ((is_xpose && !xp_contig) || is_fold),
    .ext_valid (is_fold ? fd_wb_valid : xp_ext_valid), .ext_ready (wr_ext_ready),
    .ext_waddr (is_fold ? fd_wb_waddr : xp_ext_waddr), .ext_len (is_fold ? fd_wb_len : xp_ext_len),
    .ext_done (is_fold ? fd_wb_done : xp_ext_done),
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
  // `loading` also covers a chain's descriptor fetch: port B's words reach neither
  // the units nor ro_param (no load in progress: it accepts nothing).
  always_ff @(posedge clk) begin
    if (rst) loading <= 1'b0;
    else if (pld_start || dfe_start) loading <= 1'b1;
    else if (job_start) loading <= 1'b0;
  end

  ro_param u_prm (
    .clk, .rst (urst_p),
    .ld_start (pld_q2), .ld_bcast (f_pbc), .ld_n (gp_words), .ld_busy (prm_busy),
    .in_valid (b_valid && loading), .in_ready (prm_in_ready), .in_data (b_data),
    .rd_idx (prm_idx), .rd_q (prm_q)
  );

  // Datapath -----------------------------------------------------------------------------
  logic st_a_ready, st_b_ready, st_idle, st_valid;
  logic [BW-1:0] st_data;
  logic [PIW-1:0] st_prm_idx, nm_prm_idx;

  ro_stream u_st (
    .clk, .rst (urst_d), .start (start_q2), .en (go && is_stream),
    .op (opc), .ak, .bk, .ck, .b_en (f_ben), .relu (f_relu), .scale (f_scale), .bq_sh (j_bqsh[5:0]),
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

  ro_norm u_nm (
    .clk, .rst (urst_d), .start (start_q2), .en (go && is_norm),
    .ln (f_ln), .a32 (ak == K_I32), .n (j_n), .nrows, .words (wa), .eps (j_eps), .bq_sh (j_bqsh[5:0]),
    .a_valid (a_valid && is_norm), .a_ready (nm_a_ready), .a_data,
    .prm_idx (nm_prm_idx), .prm_q,
    .tbl_we (tl_rsq_we), .tbl_addr (tl_addr), .tbl_data (tl_rsq_d),
    .c_valid (nm_valid), .c_ready (w_ready), .c_data (nm_data),
    .idle (nm_idle)
  );

  // The output FIFO of ROPE, TRANSPOSE and FOLD: one unit runs per job, so the three
  // share one FIFO of 32 beats (TRANSPOSE's strobes beside the data; FOLD's last-word
  // flag in strobe bit 0), written by the job's unit and reset with the units; each
  // unit keeps its own credits on the FIFO's count.  FOLD pops it through its burst
  // former (which sees the head in FOLD jobs only); ROPE's and TRANSPOSE's beats go
  // straight to the write port.
  localparam int OQ_D = 32;
  logic           oq_in_valid, oq_in_ready, oq_valid, oq_ready;
  logic [BW+15:0] oq_in, oq_out;
  logic [5:0]     oq_count;
  logic           rp_o_valid, xp_o_valid, fd_o_valid;
  logic [BW-1:0]  rp_o_data;
  logic [BW+15:0] xp_o_data;
  logic [BW:0]    fd_o_data;
  logic           fd_q_ready;
  assign oq_in_valid = rp_o_valid || xp_o_valid || fd_o_valid;
  assign oq_in       = {xp_o_data[BW+15:BW+1], is_fold ? fd_o_data[BW] : xp_o_data[BW],
                        is_rope ? rp_o_data : is_fold ? fd_o_data[BW-1:0] : xp_o_data[BW-1:0]};
  assign oq_ready    = is_fold ? fd_q_ready : w_ready;
  ro_fifo #(.W(BW + 16), .D(OQ_D), .BRAM(1'b0)) u_oq (
    .clk, .rst (urst_d || start_q2),
    .in_valid (oq_in_valid), .in_ready (oq_in_ready), .in_data (oq_in),
    .out_valid (oq_valid), .out_ready (oq_ready), .out_data (oq_out),
    .count (oq_count)
  );

  logic rp_a_ready, rp_b_ready, rp_idle;
  logic [PIW-1:0] rp_prm_idx;

  ro_rope u_rp (
    .clk, .rst (urst_d), .start (start_q2), .en (go && is_rope),
    .n (j_n), .nrows, .hd (j_aux[11:0]), .segs (div_q[12:0]),
    .a_valid (a_valid && is_rope), .a_ready (rp_a_ready), .a_data,
    .b_valid (b_valid && !loading && is_rope), .b_ready (rp_b_ready), .b_data,
    .prm_idx (rp_prm_idx), .prm_q,
    .o_valid (rp_o_valid), .o_data (rp_o_data), .o_count (oq_count),
    .idle (rp_idle)
  );

  logic xp_a_ready, xp_idle;

  ro_xpose u_xp (
    .clk, .rst (urst_d), .start (start_q2), .en (go && is_xpose),
    .L (j_n), .R (j_aux[3:0]), .c1 (j_rows), .c2 (j_blocks), .c_base (j_c), .c_s1 (j_cs1), .c_s2 (j_cs2),
    .g_stride (j_bs2),
    .a_valid (a_valid && is_xpose), .a_ready (xp_a_ready), .a_data,
    .o_valid (xp_o_valid), .o_data (xp_o_data), .o_count (oq_count),
    .ext_en (xp_ext_en), .ext_valid (xp_ext_valid), .ext_ready (xp_ext_ready),
    .ext_waddr (xp_ext_waddr), .ext_len (xp_ext_len), .ext_done (xp_ext_done),
    .idle (xp_idle)
  );

  logic fd_a_ready, fd_idle, fd_valid;

  ro_fold u_fd (
    .clk, .rst (urst_d), .start (start_q2), .en (go && is_fold),
    .a (j_a), .a_s2 (j_as2), .w0 (j_bs1), .s0 (j_bs2), .n (j_n), .rows (j_rows), .blocks (j_blocks),
    .win_lo (j_wlo), .win_hi (j_whi), .s (j_aux[3:0]), .rcp (div_q[24:0]), .c (j_c),
    .rr_valid (fd_rr_valid), .rr_ready (fd_rr_ready), .rr_waddr (fd_rr_waddr), .rr_words (fd_rr_words),
    .a_valid (a_valid && is_fold), .a_ready (fd_a_ready), .a_data,
    .o_valid (fd_o_valid), .o_data (fd_o_data), .o_count (oq_count),
    .q_valid (oq_valid && is_fold), .q_last (oq_out[BW]), .q_ready (fd_q_ready),
    .c_valid (fd_valid), .c_ready (w_ready),
    .wb_valid (fd_wb_valid), .wb_ready (fd_wb_ready), .wb_waddr (fd_wb_waddr), .wb_len (fd_wb_len),
    .wb_done (fd_wb_done),
    .idle (fd_idle)
  );

  assign prm_idx = is_norm ? nm_prm_idx : is_rope ? rp_prm_idx : st_prm_idx;
  assign a_ready = is_norm ? nm_a_ready : is_rope ? rp_a_ready : is_xpose ? xp_a_ready :
                   is_tload ? tl_a_ready : is_fold ? fd_a_ready : st_a_ready;
  assign b_ready = loading ? (prm_in_ready || dfetch) : is_rope ? rp_b_ready : st_b_ready;
  // (TLOAD and the ops not known write nothing: the output FIFO stays empty)
  assign w_valid = is_norm ? nm_valid : is_stream ? st_valid : is_fold ? fd_valid : oq_valid;
  assign w_data  = is_norm ? nm_data : is_stream ? st_data : oq_out[BW-1:0];
  assign w_strb  = is_xpose ? oq_out[BW+15:BW] : '1;

  assign job_done = (tstate == T_RUN) && !job_start && !start_q && !start_q2 &&
                    a_idle && b_idle && st_idle && nm_idle && rp_idle && xp_idle && tl_idle && fd_idle && wr_idle;

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
                        known, j_bqsh[31:6], b_ext_ready, oq_in_ready};

endmodule
