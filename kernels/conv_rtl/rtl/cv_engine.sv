// ---------------------------------------------------------------------------
// cv_engine — weight cache, MAC grid and accumulators (the HLS
// process_conv_kernel_tile's Phases 1-2).
//
//   weight vectors ─► fill ─► weight cache (2 banks: slab k+1 fills while sweep k reads)
//                                    │ one TIC-lane word per column and instant
//   patch beats ─► issue ─► operand pipeline ─► skew ─► 2 x TM DSP chains ─► write-back
//                    │   (replay file for m-tiles 1..G-1)        ▲ seeds        │
//                    └► seed reads (accumulators or bias) ────────┘             ▼
//                                   accumulator buffers (one per chunk parity) ─► drain
//
// A sweep walks (output row, pixel pair, m-tile g, window position) at one
// instant per cycle, exactly the HLS flat sweep: each instant multiplies the
// two pixels' TIC-lane patch columns against one weight word per output
// column.  The seed of an accumulator word — the bias for the first input
// tile, else the stored partial sum — enters pixel 0's chains at position 0
// and pixel 1's at position 1; the window's sum leaves the chains TIC + 2
// cycles after its last instant and is stored, pixel 0 then pixel 1.  A
// depthwise sweep is a standard one with a single m-tile and diagonal weight
// words (column m1 holds only lane m1).
//
// The grid's 512 DSPs and the 48 UltraRAMs fill a large part of the device,
// so every signal that reaches all chains or all RAMs is a fan-out tree of
// registers — one copy near the source, one per column group, one per column
// (or chain) next to its DSPs or RAMs — and every RAM read has its output
// register: no register drives loads that lie far apart.
// ---------------------------------------------------------------------------
module cv_engine
  import cv_pkg::*;
(
  input  logic          clk,
  input  logic          rst,
  input  job_t          j,

  input  logic          fq_valid,          // sweep descriptors for the fill
  output logic          fq_ready,
  input  sweep_t        fq,
  input  logic          wv_valid,          // weight vectors (standard) / beats (depthwise)
  output logic          wv_ready,
  input  logic [TIC*EW-1:0] wv_data,

  input  logic          eq_valid,          // sweep descriptors for the issue
  output logic          eq_ready,
  input  sweep_t        eq,

  input  logic          pt_valid,          // patch beats: pixel x lane l at [(x*TIC + l)*EW +: EW]
  output logic          pt_ready,
  input  logic [PATCH_W-1:0] pt_data,

  input  logic          bias_ok,           // the bias RAM is loaded
  output logic [6:0]    bias_tile,         // the m-tile of this instant's bias
  input  logic [TM*EW-1:0] bias_vec,       // the bias of the previous cycle's bias_tile

  output logic          dr_valid,          // finished chunks
  input  logic          dr_ready,
  output dreq_t         dr,
  input  logic          d_ren,             // the drain's reads
  input  logic          d_par,
  input  logic [11:0]   d_addr,
  output logic [TM*32-1:0] d_rdata,        // three cycles after d_ren
  input  logic          d_free,            // the drain is done with buffer d_free_par
  input  logic          d_free_par,

  output logic          idle
);
  // ---- the operand pipeline (issue cycle I) ---------------------------------------------
  //   I+1  weight-cache address (one copy), I+2 one copy per column; the RAM
  //   latch I+3, its output register I+4, the column mask I+5, lane 0's
  //   operand register next to the DSPs: E = I + D_E.
  localparam int D_E  = 6;                 // issue -> E (lane 0's operands at the DSPs)
  localparam int LAT  = D_E + TIC + 2;     // issue -> chain output
  localparam int WBS  = 3;                 // write-back registers: the pixel mux, a middle one, a copy per buffer
  localparam int LSTO = LAT + WBS + 1;     // issue -> the cycle of an instant's last store (pixel 1)
  localparam int NG   = 4;                 // column groups of the fan-out trees
  localparam int GW   = TM / NG;           // columns per group
  localparam int WW   = TIC * EW;          // one weight / patch column
  // A sweep that reads stored partial sums may start right behind one at least
  // HAZ instants long: a word's store (issue + LAT + WBS) lands before the
  // next sweep reads it (issue + 1), at least L - n_win + 2 cycles later.
  localparam int HAZ  = 80;

  // the weight cache address {bank, tile, khi, kwi} is a bit concatenation
  if (WC_ROW != 8 || MAXK > WC_ROW || MPG != 4 || WC_WORDS != 512) begin : g_wc_check
    $error("cv_engine: the weight cache address assumes WC_ROW = 8, MPG = 4");
  end
  if (HAZ < LAT + WBS - 2 + MAXK * MAXK) begin : g_haz_check
    $error("cv_engine: HAZ too short for the store latency");
  end
  if (D_E != 6 || TM % NG != 0) begin : g_pipe_check
    $error("cv_engine: the operand pipeline assumes D_E = 6");
  end

  // the job constants the engine uses, registered here: j is stable from one
  // cycle before the units' reset, so the copy is valid from the reset on
  logic [2:0]  jq_kh, jq_kw;
  logic [5:0]  jq_n_pos, jq_n_win;
  logic [31:0] jq_out_ch;
  logic [6:0]  jq_m_tiles;
  logic [12:0] jq_row_words;
  logic        jq_dwm, jq_has_bias;
  always_ff @(posedge clk) begin
    jq_kh        <= j.kh;
    jq_kw        <= j.kw;
    jq_n_pos     <= j.n_pos;
    jq_n_win     <= j.n_win;
    jq_out_ch    <= j.out_ch;
    jq_m_tiles   <= j.m_tiles;
    jq_row_words <= j.row_words;
    jq_dwm       <= j.dwm;
    jq_has_bias  <= j.has_bias;
  end

  // ===========================================================================
  // Slab bookkeeping.  Slab k (= sweep k) lives in bank k mod 2.  cB fills
  // begun, cF slabs filled, cS sweeps started, cR sweeps done reading (all
  // mod 4): the fill may begin slab cB when bank cB mod 2 is no longer read
  // (cB - cR < 2), a sweep may start when its slab is in (cF != cS).
  // ===========================================================================
  logic [1:0] cB, cF, cS, cR;
  logic       f_done_pulse, s_start, r_done;

  // ===========================================================================
  // Fill.  The weight vectors pass a two-entry FIFO first: its count is a
  // register, so neither side's handshake reaches the other's logic.
  // ===========================================================================
  logic        wf_valid, wf_ready;
  logic [WW-1:0] wf_data;
  logic [1:0]  wf_n;
  cv_fifo #(.W(WW), .D(2), .BRAM(1'b0)) u_wv (
    .clk, .rst, .in_valid (wv_valid), .in_ready (wv_ready), .in_data (wv_data),
    .out_valid (wf_valid), .out_ready (wf_ready), .out_data (wf_data), .count (wf_n)
  );

  logic        f_act, f_dw, f_bank;
  logic [2:0]  f_t, f_gn;
  logic [4:0]  f_m1, f_mv;
  logic [31:0] f_mrem;
  logic [2:0]  f_khi, f_kwi;
  logic [5:0]  f_pos;
  logic [2:0]  f_q;
  logic        f_have;
  logic        f_init;                     // first fill cycle: f_mv from f_mrem
  logic [127:0] f_beat;
  logic        wc_we;
  logic [3:0]  wc_col;
  logic [8:0]  wc_addr;
  logic [WW-1:0] wc_data;

  logic [1:0]  f_ahead;
  assign f_ahead  = cB - cR;
  assign fq_ready = !f_act && (f_ahead != 2'd2);

  function automatic logic [4:0] mvalid(input logic [31:0] rem);
    return (rem >= 32'(TM)) ? 5'(TM) : rem[4:0];
  endfunction

  // the fill's write this cycle
  logic f_std_w, f_dw_w, f_dw_ld;
  assign f_std_w  = f_act && !f_init && !f_dw && wf_valid;
  assign f_dw_ld  = f_act && !f_init && f_dw && !f_have && wf_valid;
  assign f_dw_w   = f_act && !f_init && f_dw && f_have;
  assign wf_ready = f_std_w || f_dw_ld;

  always_comb begin
    wc_we   = f_std_w || f_dw_w;
    wc_col  = f_m1[3:0];
    wc_addr = {f_bank, f_t[1:0], f_khi, f_kwi};
    if (f_dw) begin
      wc_data = '0;
      for (int l = 0; l < TIC; l++)
        if (l == int'(f_m1)) wc_data[l * EW +: EW] = f_beat[f_q * EW +: EW];
    end else begin
      wc_data = wf_data;
    end
  end

  always_ff @(posedge clk) begin
    f_done_pulse <= 1'b0;
    if (rst) begin
      f_act <= 1'b0;
      f_have <= 1'b0;
    end else if (!f_act) begin
      if (fq_valid && fq_ready) begin
        f_act  <= 1'b1;
        f_dw   <= jq_dwm;
        f_bank <= cB[0];
        f_gn   <= fq.g_n;
        f_t    <= '0;
        f_m1   <= '0;
        f_mrem <= jq_out_ch - 32'({fq.mt_base, 4'b0});
        f_init <= 1'b1;
        f_khi  <= '0;
        f_kwi  <= '0;
        f_pos  <= '0;
        f_q    <= '0;
        f_have <= 1'b0;
      end
    end else if (f_init) begin
      f_mv   <= mvalid(f_mrem);
      f_init <= 1'b0;
    end else if (!f_dw) begin
      if (f_std_w) begin
        // order (t, m1, khi, kwi)
        if (f_kwi != jq_kw - 3'd1) f_kwi <= f_kwi + 3'd1;
        else begin
          f_kwi <= '0;
          if (f_khi != jq_kh - 3'd1) f_khi <= f_khi + 3'd1;
          else begin
            f_khi <= '0;
            if (f_m1 + 5'd1 != f_mv) f_m1 <= f_m1 + 5'd1;
            else begin
              f_m1 <= '0;
              if (f_t + 3'd1 == f_gn) begin
                f_act <= 1'b0;
                f_done_pulse <= 1'b1;
              end else begin
                f_t    <= f_t + 3'd1;
                f_mrem <= f_mrem - 32'(TM);
                f_mv   <= mvalid(f_mrem - 32'(TM));
              end
            end
          end
        end
      end
    end else begin
      // depthwise: one beat (8 positions) at a time, one position per cycle
      if (f_dw_ld) begin
        f_beat <= wf_data[127:0];
        f_have <= 1'b1;
        f_q    <= '0;
      end
      if (f_dw_w) begin
        if (f_kwi != jq_kw - 3'd1) f_kwi <= f_kwi + 3'd1;
        else begin
          f_kwi <= '0;
          f_khi <= f_khi + 3'd1;
        end
        f_q <= f_q + 3'd1;
        if (f_pos + 6'd1 == jq_n_pos) begin
          // the channel's last position: its remaining beat lanes are padding
          f_have <= 1'b0;
          f_pos  <= '0;
          f_khi  <= '0;
          f_kwi  <= '0;
          if (f_m1 + 5'd1 == f_mv) begin
            f_act <= 1'b0;
            f_done_pulse <= 1'b1;
          end else begin
            f_m1 <= f_m1 + 5'd1;
          end
        end else begin
          f_pos <= f_pos + 6'd1;
          if (f_q == 3'd7) f_have <= 1'b0;
        end
      end
    end
  end

  // the write reaches the RAMs through two registers: one copy, then one per
  // column group (it lands the cycle after f_done_pulse; the first read of a
  // sweep that waits for the slab is three cycles later)
  logic          wc_we_q;
  logic [3:0]    wc_col_q;
  logic [8:0]    wc_addr_q;
  logic [WW-1:0] wc_data_q;
  (* keep = "true" *) logic          wg_we   [NG];
  (* keep = "true" *) logic [1:0]    wg_col  [NG];
  (* keep = "true" *) logic [8:0]    wg_addr [NG];
  (* keep = "true" *) logic [WW-1:0] wg_data [NG];
  always_ff @(posedge clk) begin
    wc_we_q   <= wc_we && !rst;
    wc_col_q  <= wc_col;
    wc_addr_q <= wc_addr;
    wc_data_q <= wc_data;
    for (int g = 0; g < NG; g++) begin
      wg_we[g]   <= wc_we_q && (int'(wc_col_q) / GW == g) && !rst;
      wg_col[g]  <= wc_col_q[1:0];
      wg_addr[g] <= wc_addr_q;
      wg_data[g] <= wc_data_q;
    end
  end

  // ===========================================================================
  // Weight cache: TM columns of WC_WORDS words of TIC lanes; columns
  // [0, TM/2) in block RAM, [TM/2, TM) in UltraRAM.  Address {bank, tile,
  // khi, kwi}.  From the issue: the address register (one copy) at I + 1,
  // its copy per column at I + 2 (each drives its column's RAMs only: 4 block
  // RAMs or 4 UltraRAMs), the RAM latch, the RAM's output register, then the
  // column mask (a column past the m-tile's channels reads zero) at I + 5.
  // ===========================================================================
  logic [8:0]    wa1;                      // I + 1
  (* keep = "true" *) logic [8:0] wa_q [TM];   // I + 2, per column
  (* keep = "true" *) logic [TM-1:0] mk1;  // the m-tile's valid columns, I + 1
  (* keep = "true" *) logic [TM-1:0] mk2, mk3, mk4;   // per column, I + 2 .. I + 4
  logic [WW-1:0] wr_data [TM];             // the masked word, I + 5

  for (genvar m = 0; m < TM; m++) begin : g_wc
    localparam int G = m / GW;
    logic [WW-1:0] rd0, rd1, rd2;
    if (m < TM / 2) begin : g_b
      (* ram_style = "block" *) logic [WW-1:0] mem [WC_WORDS];
      always_ff @(posedge clk) begin
        if (wg_we[G] && wg_col[G] == 2'(m % GW)) mem[wg_addr[G]] <= wg_data[G];
        rd0 <= mem[wa_q[m]];
      end
    end else begin : g_u
      (* ram_style = "ultra" *) logic [WW-1:0] mem [WC_WORDS];
      always_ff @(posedge clk) begin
        if (wg_we[G] && wg_col[G] == 2'(m % GW)) mem[wg_addr[G]] <= wg_data[G];
        rd0 <= mem[wa_q[m]];
      end
    end
    always_ff @(posedge clk) begin
      rd1 <= rd0;                          // the RAM's output register
      rd2 <= mk4[m] ? rd1 : '0;
    end
    assign wr_data[m] = rd2;
  end

  // ===========================================================================
  // Issue
  // ===========================================================================
  logic        s_act, s_bank;
  sweep_t      sd;
  logic [12:0] s_oh, s_pair, s_owa, s_word, s_wrow;
  logic [2:0]  s_g, s_khi, s_kwi;
  logic [5:0]  s_pos;
  logic [31:0] s_mrem0;                    // out_ch - mt_base * TM
  logic [15:0] s_iters, prev_iters;
  logic        pipe_empty;

  typedef enum logic [1:0] {B_FREE, B_ENG, B_DRAIN} own_t;
  own_t        own [2];

  logic need, adv, s_lastit, s_first, s_lastp, s_v0, s_v1, s_atpos;
  assign s_atpos  = (s_pos < jq_n_pos);
  assign need     = s_act && s_atpos && (s_g == 3'd0);
  assign adv      = s_act && (!need || pt_valid);
  assign pt_ready = adv && need;
  assign s_first  = (s_pos == 6'd0);
  assign s_lastp  = (s_pos + 6'd1 == jq_n_win);
  assign s_v0     = (s_owa >= sd.ow_start);
  assign s_v1     = (s_owa + 13'd1 < sd.ow_end);
  assign s_lastit = s_lastp && (s_g + 3'd1 == sd.g_n) && (s_pair + 13'd1 == sd.n_pairs) &&
                    (s_oh + 13'd1 == sd.coh);

  logic slab_in, bias_in, buf_in, haz_ok;
  assign slab_in  = (cF != cS);
  assign bias_in  = !eq.seed_bias || !jq_has_bias || bias_ok;
  assign buf_in   = !eq.chunk_first || (own[eq.par] == B_FREE);
  assign haz_ok   = eq.seed_bias || pipe_empty || (prev_iters >= 16'(HAZ));
  assign eq_ready = !s_act && slab_in && bias_in && buf_in && haz_ok;
  assign s_start  = eq_valid && eq_ready;

  // replay file: the window's patch beats, written at tile 0 (a cycle after
  // the issue, from the first patch register), read at tiles 1..G-1 — at
  // least n_win >= 2 instants later
  logic [PATCH_W-1:0] rf_rdata;
  logic [PATCH_W-1:0] pd [1:D_E];           // the patch root chain: pd[k] at I + k
  (* max_fanout = 64 *) logic rf_we_q;
  logic [5:0] rf_wa_q;
  always_ff @(posedge clk) begin
    rf_we_q <= pt_ready;
    rf_wa_q <= s_pos;
  end
  for (genvar x = 0; x < PIX2; x++) begin : g_rf
    cv_lutram #(.W(WW), .D(64)) u_rf (
      .clk, .we (rf_we_q), .waddr (rf_wa_q), .wdata (pd[1][x * WW +: WW]),
      .raddr (s_pos), .rdata (rf_rdata[x * WW +: WW])
    );
  end

  // m-tile g's valid columns
  logic [31:0] s_mrem;
  logic [TM-1:0] s_mask;
  always_comb begin
    s_mrem = s_mrem0 - {25'b0, s_g, 4'b0};
    for (int m = 0; m < TM; m++) s_mask[m] = (s_mrem > 32'(m));
  end

  // the bias RAM's read address (cv_bias registers it: the word arrives at
  // I + 1, registered at I + 2)
  assign bias_tile = sd.mt_base + {4'b0, s_g};

  always_ff @(posedge clk) begin
    if (rst) begin
      s_act      <= 1'b0;
      prev_iters <= 16'hFFFF;
      own[0]     <= B_FREE;
      own[1]     <= B_FREE;
    end else begin
      if (s_start) begin
        s_act   <= 1'b1;
        sd      <= eq;
        s_bank  <= cS[0];
        s_oh    <= '0;
        s_pair  <= '0;
        s_g     <= '0;
        s_pos   <= '0;
        s_khi   <= '0;
        s_kwi   <= '0;
        s_owa   <= eq.pair_base;
        s_word  <= eq.wrow0;
        s_wrow  <= eq.wrow0;
        s_mrem0 <= jq_out_ch - 32'({eq.mt_base, 4'b0});
        s_iters <= '0;
        if (eq.chunk_first) own[eq.par] <= B_ENG;
      end else if (adv) begin
        if (s_iters != 16'hFFFF) s_iters <= s_iters + 16'd1;
        if (s_lastit) begin
          s_act      <= 1'b0;
          prev_iters <= (s_iters == 16'hFFFF) ? s_iters : s_iters + 16'd1;
        end
        if (s_lastp) begin
          s_pos <= '0;
          s_khi <= '0;
          s_kwi <= '0;
          if (s_g + 3'd1 == sd.g_n) begin
            s_g <= '0;
            if (s_pair + 13'd1 == sd.n_pairs) begin
              s_pair <= '0;
              s_owa  <= sd.pair_base;
              s_wrow <= s_wrow + jq_row_words;
              s_word <= s_wrow + jq_row_words;
              s_oh   <= s_oh + 13'd1;
            end else begin
              s_pair <= s_pair + 13'd1;
              s_owa  <= s_owa + 13'd2;
              s_word <= s_word + {5'b0, jq_m_tiles, 1'b0} - {10'b0, sd.g_n} + 13'd1;
            end
          end else begin
            s_g    <= s_g + 3'd1;
            s_word <= s_word + 13'd1;
          end
        end else begin
          s_pos <= s_pos + 6'd1;
          if (s_pos + 6'd1 < jq_n_pos) begin
            if (s_kwi == jq_kw - 3'd1) begin
              s_kwi <= '0;
              s_khi <= s_khi + 3'd1;
            end else begin
              s_kwi <= s_kwi + 3'd1;
            end
          end
        end
      end
      if (dr_valid && dr_ready) own[dr.par] <= B_DRAIN;
      if (d_free) own[d_free_par] <= B_FREE;
    end
  end

  // a sweep's bank is released the cycle after its last weight read reached
  // the RAMs (issue + 2)
  logic [2:0] r_pipe;
  always_ff @(posedge clk) begin
    if (rst) r_pipe <= '0;
    else     r_pipe <= {r_pipe[1:0], adv && s_lastit};
  end
  assign r_done = r_pipe[2];

  always_ff @(posedge clk) begin
    if (rst) begin
      cB <= '0; cF <= '0; cS <= '0; cR <= '0;
    end else begin
      if (!f_act && fq_valid && fq_ready) cB <= cB + 2'd1;
      if (f_done_pulse) cF <= cF + 2'd1;
      if (s_start)      cS <= cS + 2'd1;
      if (r_done)       cR <= cR + 2'd1;
    end
  end

  // ===========================================================================
  // Operand pipeline.  Issue cycle I; weights and patches meet at E = I + D_E;
  // seeds at E + 1; the chains' last DSP takes wfb at E + TIC; sums at
  // E + TIC + 2 = I + LAT; stored at I + LAT + WBS (pixel 0) and one cycle
  // later (pixel 1).
  // ===========================================================================
  typedef struct packed {
    logic        v;                        // a real instant (not a bubble)
    logic        first, last, v0, v1;
    logic        sp0, sp1;                 // pixel 0 / 1 takes its seed now
    logic        sbias;                    // seeds from the bias
    logic        par;
    logic [11:0] w0, w1;
    logic        cdone;                    // the last instant of a chunk
  } ctl_t;

  ctl_t ctl0;
  logic [PATCH_W-1:0] patch0;
  logic [11:0]        aaddr0;
  logic               aren0;
  always_comb begin
    ctl0.v     = adv;
    ctl0.first = s_first;
    ctl0.last  = s_lastp;
    ctl0.v0    = s_v0;
    ctl0.v1    = s_v1;
    ctl0.sp0   = s_first;
    ctl0.sp1   = (s_pos == 6'd1);
    ctl0.sbias = sd.seed_bias;
    ctl0.par   = sd.par;
    ctl0.w0    = s_word[11:0];
    ctl0.w1    = s_word[11:0] + {5'b0, jq_m_tiles};
    ctl0.cdone = s_lastit && sd.chunk_last;
    patch0     = (s_g == 3'd0) ? pt_data : rf_rdata;   // zeroed at I + 2 (pgate)
    aren0      = adv && !sd.seed_bias && (ctl0.sp0 || ctl0.sp1);
    aaddr0     = (ctl0.sp1 && s_v1) ? ctl0.w1 : ctl0.w0;
  end

  // control delay line: ctl_d[k] is the instant issued k cycles ago
  ctl_t ctl_d [1:LSTO];
  always_ff @(posedge clk) begin
    if (rst) begin
      for (int k = 1; k <= LSTO; k++) ctl_d[k] <= '0;
    end else begin
      ctl_d[1] <= ctl0;
      for (int k = 2; k <= LSTO; k++) ctl_d[k] <= ctl_d[k - 1];
    end
  end

  always_comb begin
    pipe_empty = !s_act;
    for (int k = 1; k <= LSTO; k++) if (ctl_d[k].v) pipe_empty = 1'b0;
  end

  // weights: the address and mask trees (one copy, then one per column)
  always_ff @(posedge clk) begin
    wa1 <= {s_bank, s_g[1:0], s_khi, s_kwi};
    mk1 <= s_mask;
    for (int m = 0; m < TM; m++) wa_q[m] <= wa1;
    mk2 <= mk1;
    mk3 <= mk2;
    mk4 <= mk3;
  end

  // patches: the root chain pd[1..D_E], zeroed at I + 2 for bubbles and
  // positions past the window (pgate: one copy per 64 bits)
  localparam int PGN = PATCH_W / 64;
  (* keep = "true" *) logic pgate [PGN];
  always_ff @(posedge clk) begin
    for (int k = 0; k < PGN; k++) pgate[k] <= adv && s_atpos;
    pd[1] <= patch0;
    for (int k = 0; k < PGN; k++) pd[2][k * 64 +: 64] <= pgate[k] ? pd[1][k * 64 +: 64] : '0;
    for (int k = 3; k <= D_E; k++) pd[k] <= pd[k - 1];
  end

  // ===========================================================================
  // Accumulator buffers: AWORDS words of TM x 32 bits per chunk parity.  Port
  // A reads (the issue's seeds, or the drain when it owns the buffer), port B
  // writes (the write-back).  Read latency 3: address register, the RAM latch
  // and its output register.
  // ===========================================================================
  logic [TM*32-1:0] acc_rd [2];
  logic [11:0]      a_ra [2];
  (* keep = "true" *) logic             wq2_we   [2];  // the stores: a copy per buffer
  (* keep = "true" *) logic [11:0]      wq2_addr [2];
  (* keep = "true" *) logic [TM*32-1:0] wq2_data [2];

  always_ff @(posedge clk)
    for (int b = 0; b < 2; b++)
      a_ra[b] <= (own[b] == B_DRAIN) ? d_addr : aaddr0;

  for (genvar b = 0; b < 2; b++) begin : g_acc
    (* ram_style = "ultra" *) logic [TM*32-1:0] mem [AWORDS];
    logic [TM*32-1:0] r1, r2;
    always_ff @(posedge clk) begin
      if (wq2_we[b]) mem[wq2_addr[b]] <= wq2_data[b];
      r1 <= mem[a_ra[b]];
      r2 <= r1;
    end
    assign acc_rd[b] = r2;
  end

  // the drain's data: three cycles after its address
  logic [2:0] dpar_d;
  always_ff @(posedge clk) dpar_d <= {dpar_d[1:0], d_par};
  assign d_rdata = acc_rd[dpar_d[2]];

  logic unused_ren;
  assign unused_ren = aren0 ^ d_ren;

  // ===========================================================================
  // Seeds.  I + 3: the accumulator word (both buffers); I + 4: the chunk's
  // buffer; I + 5: bias or partial sum, masked (one copy); I + 6: again, near
  // the columns; E + 1 = I + 7: per chain, zero unless the pixel seeds now
  // (pixel 0 at its position 0, pixel 1 at its position 1).
  // ===========================================================================
  logic [TM*EW-1:0] bias_d2, bias_d3, bias_d4;   // I + 2 .. I + 4
  (* keep = "true" *) logic [TM-1:0] ms1, ms2, ms3, ms4;   // the column mask, I + 1 .. I + 4
  (* keep = "true" *) logic par3 [NG];            // ctl_d[3].par, I + 3
  (* keep = "true" *) logic sb4  [NG];            // ctl_d[4].sbias, I + 4
  logic             hb4;                          // the job has a bias
  logic [TM*32-1:0] acc_r3;                       // I + 4
  logic [31:0]      sw5 [TM];                     // I + 5
  logic [31:0]      sw6 [TM];                     // I + 6
  (* keep = "true" *) logic sp5 [PIX2][NG];       // pixel x seeds now: per group, I + 5
  (* keep = "true" *) logic sp6 [PIX2][TM];       // per chain, I + 6
  logic [31:0]      seed [PIX2][TM];              // E + 1

  always_ff @(posedge clk) begin
    bias_d2 <= bias_vec;
    bias_d3 <= bias_d2;
    bias_d4 <= bias_d3;
    ms1 <= s_mask;
    ms2 <= ms1;
    ms3 <= ms2;
    ms4 <= ms3;
    hb4 <= jq_has_bias;
    for (int g = 0; g < NG; g++) begin
      par3[g]   <= ctl_d[2].par;
      sb4[g]    <= ctl_d[3].sbias;
      sp5[0][g] <= ctl_d[4].v && ctl_d[4].sp0;
      sp5[1][g] <= ctl_d[4].v && ctl_d[4].sp1;
    end
    for (int m = 0; m < TM; m++) begin
      acc_r3[m * 32 +: 32] <= par3[m / GW] ? acc_rd[1][m * 32 +: 32] : acc_rd[0][m * 32 +: 32];
      if (!ms4[m])          sw5[m] <= '0;
      else if (sb4[m / GW]) sw5[m] <= hb4 ? {{8{bias_d4[m * EW + EW - 1]}}, bias_d4[m * EW +: EW], 8'b0} : '0;
      else                  sw5[m] <= acc_r3[m * 32 +: 32];
      sw6[m] <= sw5[m];
      sp6[0][m] <= sp5[0][m / GW];
      sp6[1][m] <= sp5[1][m / GW];
      seed[0][m] <= sp6[0][m] ? sw6[m] : '0;
      seed[1][m] <= sp6[1][m] ? sw6[m] : '0;
    end
  end

  // ===========================================================================
  // Skew and the grid.  Lane l of every operand is delayed l cycles past E.
  // ===========================================================================
  logic [TIC-1:0][15:0] a_sk [PIX2][TM];   // patch lanes, per chain
  logic [TIC-1:0][15:0] b_sk [TM];         // weight lanes, per column

  // Lane l of a pixel's patch reaches the DSPs of all TM columns at E + l
  // through a tree: one register at E + l - 2 (lanes 0..2: the root chain;
  // lanes 3..: a shift register from pd[D_E] — a register in front of the
  // SRL and one after it), NG copies at E + l - 1, one per column at E + l.
  for (genvar x = 0; x < PIX2; x++) begin : g_ask
    for (genvar l = 0; l < TIC; l++) begin : g_l
      logic [15:0] lv0;
      (* keep = "true" *) logic [15:0] lv1 [NG];
      (* keep = "true" *) logic [15:0] lv2 [TM];
      if (l < 3) begin : g_root
        assign lv0 = pd[D_E - 2 + l][(x * TIC + l) * EW +: EW];
      end else begin : g_sr
        (* srl_style = "srl_reg" *) logic [15:0] sr [l - 2];
        always_ff @(posedge clk) begin
          sr[0] <= pd[D_E][(x * TIC + l) * EW +: EW];
          for (int k = 1; k < l - 2; k++) sr[k] <= sr[k - 1];
        end
        assign lv0 = sr[l - 3];
      end
      always_ff @(posedge clk) begin
        for (int c = 0; c < NG; c++) lv1[c] <= lv0;
        for (int m = 0; m < TM; m++) lv2[m] <= lv1[m / GW];
      end
      for (genvar m = 0; m < TM; m++) begin : g_m
        assign a_sk[x][m][l] = lv2[m];
      end
    end
  end

  // A weight lane drives only the two pixels' DSPs of its column: the masked
  // word (a register) delayed l + 1 cycles — a shift register ending in a
  // register next to the DSPs.
  for (genvar m = 0; m < TM; m++) begin : g_bsk
    for (genvar l = 0; l < TIC; l++) begin : g_l
      (* srl_style = "srl_reg" *) logic [15:0] sr [l + 1];
      always_ff @(posedge clk) begin
        sr[0] <= wr_data[m][l * EW +: EW];
        for (int k = 1; k <= l; k++) sr[k] <= sr[k - 1];
      end
      assign b_sk[m][l] = sr[l];
    end
  end

  // wfb: accumulate unless the instant is a window's first (at E + TIC): one
  // register at E + TIC - 2, NG copies at E + TIC - 1, one per chain at E + TIC
  logic wfb0;
  (* keep = "true" *) logic wfb1 [NG];
  (* keep = "true" *) logic wfb2 [PIX2][TM];
  always_ff @(posedge clk) begin
    wfb0 <= !(ctl_d[D_E + TIC - 3].v && ctl_d[D_E + TIC - 3].first);
    for (int g = 0; g < NG; g++) wfb1[g] <= wfb0;
    for (int x = 0; x < PIX2; x++)
      for (int m = 0; m < TM; m++) wfb2[x][m] <= wfb1[m / GW];
  end

  logic [31:0] psum [PIX2][TM];
  for (genvar x = 0; x < PIX2; x++) begin : g_px
    for (genvar m = 0; m < TM; m++) begin : g_col
      cv_mac_chain u_chain (
        .clk, .a (a_sk[x][m]), .b (b_sk[m]), .c (seed[x][m]), .wfb (wfb2[x][m]), .p (psum[x][m])
      );
    end
  end

  // ===========================================================================
  // Write-back.  The window's sums leave the chains at I_last + LAT; the
  // first register (next to the chains) takes pixel 0's word, the cycle after
  // pixel 1's (held a cycle); then a middle register and a copy per buffer
  // next to its UltraRAMs.  Stores at I_last + LAT + WBS and one cycle later.
  // ===========================================================================
  ctl_t             wbc;
  assign wbc = ctl_d[LAT];

  (* keep = "true" *) logic sel_g [NG];   // the instant at I + LAT is a window's last: per group, I + LAT - 1
  (* keep = "true" *) logic sel_c [TM];   // per column, I + LAT
  logic [31:0]      h1 [TM];              // pixel 1's word, a cycle later
  logic [TM*32-1:0] wq1_data, wqm_data;
  logic             wq1_we [2], wqm_we [2];
  logic [11:0]      wq1_addr, wqm_addr;
  logic             hold_en, hold_par;
  logic [11:0]      hold_addr;

  always_ff @(posedge clk) begin
    for (int g = 0; g < NG; g++) sel_g[g] <= ctl_d[LAT - 2].v && ctl_d[LAT - 2].last;
    for (int m = 0; m < TM; m++) begin
      sel_c[m] <= sel_g[m / GW];
      h1[m]    <= psum[1][m];
      wq1_data[m * 32 +: 32] <= sel_c[m] ? psum[0][m] : h1[m];
    end
    hold_par  <= wbc.par;
    hold_addr <= wbc.w1;
    if (wbc.v && wbc.last) wq1_addr <= wbc.w0;
    else                   wq1_addr <= hold_addr;
    if (rst) begin
      hold_en   <= 1'b0;
      wq1_we[0] <= 1'b0;
      wq1_we[1] <= 1'b0;
    end else begin
      hold_en <= wbc.v && wbc.last && wbc.v1;
      for (int b = 0; b < 2; b++)
        wq1_we[b] <= (wbc.v && wbc.last) ? (wbc.v0 && wbc.par == 1'(b))
                                         : (hold_en && hold_par == 1'(b));
    end
    for (int b = 0; b < 2; b++) begin
      wqm_we[b]   <= wq1_we[b] && !rst;
      wq2_we[b]   <= wqm_we[b] && !rst;
      wq2_addr[b] <= wqm_addr;
      wq2_data[b] <= wqm_data;
    end
    wqm_addr <= wq1_addr;
    wqm_data <= wq1_data;
  end

  // ===========================================================================
  // Finished chunks: the request is queued when the chunk's last sweep
  // starts and released once its last store is done (the cycle of the
  // store of the chunk's last instant's pixel 1).
  // ===========================================================================
  dreq_t pend_in;
  logic  pend_valid, pend_ready, pend_in_ready;
  logic [2:0] pend_n;
  assign pend_in = '{par: eq.par, y_cbase: eq.y_cbase, run_len: eq.run_len};
  cv_fifo #(.W($bits(dreq_t)), .D(4), .BRAM(1'b0)) u_pend (
    .clk, .rst,
    .in_valid (s_start && eq.chunk_last), .in_ready (pend_in_ready), .in_data (pend_in),
    .out_valid (pend_valid), .out_ready (pend_ready), .out_data (dr),
    .count (pend_n)
  );

  logic [2:0] c_ready;                     // chunks whose stores are done
  logic       c_done;
  assign c_done     = ctl_d[LSTO].v && ctl_d[LSTO].cdone;
  assign dr_valid   = pend_valid && (c_ready != '0);
  assign pend_ready = dr_valid && dr_ready;
  always_ff @(posedge clk) begin
    if (rst) c_ready <= '0;
    else     c_ready <= c_ready + 3'(c_done) - 3'(pend_ready);
  end

  assign idle = !f_act && !s_act && pipe_empty && !hold_en && !pend_valid && !wf_valid;

  logic unused;
  assign unused = ^pend_n ^ ^sd ^ ^j ^ f_t[2] ^ s_g[2] ^ pend_in_ready ^ ^wf_n;

endmodule
