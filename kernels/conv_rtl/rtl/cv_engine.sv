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
  output logic [6:0]    bias_tile,
  input  logic [TM*EW-1:0] bias_vec,       // the bias of m-tile bias_tile (asynchronous)

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
  localparam int HAZ  = 80;                // a sweep this long keeps its stores ahead of the next one's loads
  localparam int LAT  = 3 + TIC + 2;       // issue -> chain output
  localparam int WW   = TIC * EW;          // one weight / patch column

  // the weight cache address {bank, tile, khi, kwi} is a bit concatenation
  if (WC_ROW != 8 || MAXK > WC_ROW || MPG != 4 || WC_WORDS != 512) begin : g_wc_check
    $error("cv_engine: the weight cache address assumes WC_ROW = 8, MPG = 4");
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
  // Fill
  // ===========================================================================
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
  logic        wc_we, wc_we_q;
  logic [3:0]  wc_col, wc_col_q;
  logic [8:0]  wc_addr, wc_addr_q;
  logic [WW-1:0] wc_data, wc_data_q;      // _q: the write, a cycle later (it lands with f_done_pulse)

  logic [1:0]  f_ahead;
  assign f_ahead  = cB - cR;
  assign fq_ready = !f_act && (f_ahead != 2'd2);

  function automatic logic [4:0] mvalid(input logic [31:0] rem);
    return (rem >= 32'(TM)) ? 5'(TM) : rem[4:0];
  endfunction

  // the fill's write this cycle
  logic f_std_w, f_dw_w, f_dw_ld;
  assign f_std_w = f_act && !f_init && !f_dw && wv_valid;
  assign f_dw_ld = f_act && !f_init && f_dw && !f_have && wv_valid;
  assign f_dw_w  = f_act && !f_init && f_dw && f_have;
  assign wv_ready = f_std_w || f_dw_ld;

  logic f_last_pos;                        // standard: (khi, kwi) is the window's last
  assign f_last_pos = (f_khi == j.kh - 3'd1) && (f_kwi == j.kw - 3'd1);

  always_comb begin
    wc_we   = f_std_w || f_dw_w;
    wc_col  = f_m1[3:0];
    wc_addr = {f_bank, f_t[1:0], f_khi, f_kwi};
    if (f_dw) begin
      wc_data = '0;
      for (int l = 0; l < TIC; l++)
        if (l == int'(f_m1)) wc_data[l * EW +: EW] = f_beat[f_q * EW +: EW];
    end else begin
      wc_data = wv_data;
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
        f_dw   <= j.dwm;
        f_bank <= cB[0];
        f_gn   <= fq.g_n;
        f_t    <= '0;
        f_m1   <= '0;
        f_mrem <= j.out_ch - 32'({fq.mt_base, 4'b0});
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
        if (f_kwi != j.kw - 3'd1) f_kwi <= f_kwi + 3'd1;
        else begin
          f_kwi <= '0;
          if (f_khi != j.kh - 3'd1) f_khi <= f_khi + 3'd1;
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
        f_beat <= wv_data[127:0];
        f_have <= 1'b1;
        f_q    <= '0;
      end
      if (f_dw_w) begin
        if (f_kwi != j.kw - 3'd1) f_kwi <= f_kwi + 3'd1;
        else begin
          f_kwi <= '0;
          f_khi <= f_khi + 3'd1;
        end
        f_q <= f_q + 3'd1;
        if (f_pos + 6'd1 == j.n_pos) begin
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

  always_ff @(posedge clk) begin
    wc_we_q   <= wc_we && !rst;
    wc_col_q  <= wc_col;
    wc_addr_q <= wc_addr;
    wc_data_q <= wc_data;
  end

  logic unused_f;
  assign unused_f = f_last_pos;

  // ===========================================================================
  // Weight cache: TM columns of WC_WORDS words of TIC lanes; columns
  // [0, TM/2) in block RAM, [TM/2, TM) in UltraRAM.  Address {bank, tile,
  // khi, kwi}; read latency 3 from the issue (address and two data registers,
  // the second one zeroing a column past the m-tile's channels).
  // ===========================================================================
  // the read address register, one copy per column (each drives its column's
  // RAMs only: 4 block RAMs or 4 UltraRAMs)
  (* keep = "true" *) logic [8:0] wa_q [TM];
  logic [TM-1:0] mask_d1, mask_d2, mask_e, mask_e1;   // the m-tile's valid columns, issue + 1 .. E + 1
  logic [WW-1:0] wr_data [TM];             // valid in instant E

  for (genvar m = 0; m < TM; m++) begin : g_wc
    logic [WW-1:0] rd1, rd2;
    if (m < TM / 2) begin : g_b
      (* ram_style = "block" *) logic [WW-1:0] mem [WC_WORDS];
      always_ff @(posedge clk) begin
        if (wc_we_q && wc_col_q == 4'(m)) mem[wc_addr_q] <= wc_data_q;
        rd1 <= mem[wa_q[m]];
        rd2 <= mask_d2[m] ? rd1 : '0;
      end
    end else begin : g_u
      (* ram_style = "ultra" *) logic [WW-1:0] mem [WC_WORDS];
      always_ff @(posedge clk) begin
        if (wc_we_q && wc_col_q == 4'(m)) mem[wc_addr_q] <= wc_data_q;
        rd1 <= mem[wa_q[m]];
        rd2 <= mask_d2[m] ? rd1 : '0;
      end
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
  assign s_atpos  = (s_pos < j.n_pos);
  assign need     = s_act && s_atpos && (s_g == 3'd0);
  assign adv      = s_act && (!need || pt_valid);
  assign pt_ready = adv && need;
  assign s_first  = (s_pos == 6'd0);
  assign s_lastp  = (s_pos + 6'd1 == j.n_win);
  assign s_v0     = (s_owa >= sd.ow_start);
  assign s_v1     = (s_owa + 13'd1 < sd.ow_end);
  assign s_lastit = s_lastp && (s_g + 3'd1 == sd.g_n) && (s_pair + 13'd1 == sd.n_pairs) &&
                    (s_oh + 13'd1 == sd.coh);

  logic slab_in, bias_in, buf_in, haz_ok;
  assign slab_in  = (cF != cS);
  assign bias_in  = !eq.seed_bias || !j.has_bias || bias_ok;
  assign buf_in   = !eq.chunk_first || (own[eq.par] == B_FREE);
  assign haz_ok   = eq.seed_bias || pipe_empty || (prev_iters >= 16'(HAZ));
  assign eq_ready = !s_act && slab_in && bias_in && buf_in && haz_ok;
  assign s_start  = eq_valid && eq_ready;

  // replay file: the window's patch beats, written at tile 0, read at tiles 1..G-1
  logic [PATCH_W-1:0] rf_rdata;
  for (genvar x = 0; x < PIX2; x++) begin : g_rf
    cv_lutram #(.W(WW), .D(64)) u_rf (
      .clk, .we (pt_ready), .waddr (s_pos), .wdata (pt_data[x * WW +: WW]),
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
        s_mrem0 <= j.out_ch - 32'({eq.mt_base, 4'b0});
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
              s_wrow <= s_wrow + j.row_words;
              s_word <= s_wrow + j.row_words;
              s_oh   <= s_oh + 13'd1;
            end else begin
              s_pair <= s_pair + 13'd1;
              s_owa  <= s_owa + 13'd2;
              s_word <= s_word + {5'b0, j.m_tiles, 1'b0} - {10'b0, sd.g_n} + 13'd1;
            end
          end else begin
            s_g    <= s_g + 3'd1;
            s_word <= s_word + 13'd1;
          end
        end else begin
          s_pos <= s_pos + 6'd1;
          if (s_pos + 6'd1 < j.n_pos) begin
            if (s_kwi == j.kw - 3'd1) begin
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

  // a sweep's bank is released two cycles after its last weight read was issued
  logic [1:0] r_pipe;
  always_ff @(posedge clk) begin
    if (rst) r_pipe <= '0;
    else     r_pipe <= {r_pipe[0], adv && s_lastit};
  end
  assign r_done = r_pipe[1];

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
  // Operand pipeline.  Issue cycle I; weights and patches meet at E = I + 3;
  // seeds at E + 1; the chains' last DSP takes wfb at E + TIC; sums at
  // E + TIC + 2 = I + LAT.
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
    ctl0.w1    = s_word[11:0] + {5'b0, j.m_tiles};
    ctl0.cdone = s_lastit && sd.chunk_last;
    patch0     = !adv || !s_atpos ? '0 : (s_g == 3'd0) ? pt_data : rf_rdata;
    aren0      = adv && !sd.seed_bias && (ctl0.sp0 || ctl0.sp1);
    aaddr0     = (ctl0.sp1 && s_v1) ? ctl0.w1 : ctl0.w0;
  end

  // control delay line: ctl_d[k] is the instant issued k cycles ago
  ctl_t ctl_d [1:LAT+1];
  always_ff @(posedge clk) begin
    if (rst) begin
      for (int k = 1; k <= LAT + 1; k++) ctl_d[k] <= '0;
    end else begin
      ctl_d[1] <= ctl0;
      for (int k = 2; k <= LAT + 1; k++) ctl_d[k] <= ctl_d[k - 1];
    end
  end

  always_comb begin
    pipe_empty = !s_act;
    for (int k = 1; k <= LAT + 1; k++) if (ctl_d[k].v) pipe_empty = 1'b0;
  end

  // weights: address register, two RAM data registers -> instant E
  always_ff @(posedge clk)
    for (int m = 0; m < TM; m++) wa_q[m] <= {s_bank, s_g[1:0], s_khi, s_kwi};

  // patches and masks to E, biases to E + 1
  localparam int NREP = 4;                 // copies of a patch lane, each for TM / NREP columns
  logic [PATCH_W-1:0] patch_d1, patch_d2, patch_e;
  (* keep = "true" *) logic [PATCH_W-1:0] patch_er [NREP];   // patch_e, for lane 0
  logic [TM*EW-1:0]   bias_d1, bias_d2, bias_d3, bias_e1;
  always_ff @(posedge clk) begin
    patch_d1 <= patch0;
    patch_d2 <= patch_d1;
    patch_e  <= patch_d2;
    for (int c = 0; c < NREP; c++) patch_er[c] <= patch_d2;
    mask_d1  <= s_mask;
    mask_d2  <= mask_d1;
    mask_e   <= mask_d2;
    mask_e1  <= mask_e;
    bias_d1  <= bias_vec;
    bias_d2  <= bias_d1;
    bias_d3  <= bias_d2;
    bias_e1  <= bias_d3;
  end

  // ===========================================================================
  // Accumulator buffers: AWORDS words of TM x 32 bits per chunk parity.  Port
  // A reads (the issue's seeds, or the drain when it owns the buffer), port B
  // writes (the write-back).  Read latency 3: address register, two data
  // registers.
  // ===========================================================================
  logic [TM*32-1:0] acc_rd [2];
  logic             wb_we [2];
  logic [11:0]      wb_addr;
  logic [TM*32-1:0] wb_data;
  logic [11:0]      a_ra [2];

  always_ff @(posedge clk)
    for (int b = 0; b < 2; b++)
      a_ra[b] <= (own[b] == B_DRAIN) ? d_addr : aaddr0;

  for (genvar b = 0; b < 2; b++) begin : g_acc
    (* ram_style = "ultra" *) logic [TM*32-1:0] mem [AWORDS];
    logic [TM*32-1:0] r1, r2;
    always_ff @(posedge clk) begin
      if (wb_we[b]) mem[wb_addr] <= wb_data;
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

  // seeds at E + 1: pixel 0 at its position 0, pixel 1 at its position 1
  logic [TM*32-1:0] aseed_e1;
  always_ff @(posedge clk) aseed_e1 <= acc_rd[ctl_d[3].par];

  logic [31:0] seed [PIX2][TM];
  always_comb begin
    for (int m = 0; m < TM; m++) begin
      logic [31:0] s;
      s = ctl_d[4].sbias ? {{8{bias_e1[m * EW + EW - 1]}}, bias_e1[m * EW +: EW], 8'b0}
                         : aseed_e1[m * 32 +: 32];
      if (!mask_e1[m]) s = '0;
      seed[0][m] = (ctl_d[4].v && ctl_d[4].sp0) ? s : '0;
      seed[1][m] = (ctl_d[4].v && ctl_d[4].sp1) ? s : '0;
    end
  end

  // ===========================================================================
  // Skew and the grid.  Lane l of every operand is delayed l cycles past E.
  // ===========================================================================
  logic [TIC-1:0][15:0] a_sk [PIX2][NREP]; // patch lanes, a copy per TM / NREP columns
  logic [TIC-1:0][15:0] b_sk [TM];         // weight lanes, per column

  // Lane l of a pixel's patch reaches the DSPs of all TM columns at E + l: a
  // shift register of l - 1 stages, then NREP copies of the last stage, each
  // driving TM / NREP columns (lane 0: the copies of patch_e).  A weight lane
  // drives only the two pixels' DSPs of its column: a shift register ending
  // in a flip-flop.
  for (genvar x = 0; x < PIX2; x++) begin : g_ask
    for (genvar l = 0; l < TIC; l++) begin : g_l
      if (l == 0) begin : g0
        for (genvar c = 0; c < NREP; c++) begin : g_c
          assign a_sk[x][c][l] = patch_er[c][(x * TIC + l) * EW +: EW];
        end
      end else begin : gd
        logic [15:0] src;
        (* keep = "true" *) logic [15:0] fin [NREP];
        if (l == 1) begin : g1
          assign src = patch_e[(x * TIC + l) * EW +: EW];
        end else begin : gs
          logic [15:0] sr [l - 1];
          always_ff @(posedge clk) begin
            sr[0] <= patch_e[(x * TIC + l) * EW +: EW];
            for (int k = 1; k < l - 1; k++) sr[k] <= sr[k - 1];
          end
          assign src = sr[l - 2];
        end
        always_ff @(posedge clk)
          for (int c = 0; c < NREP; c++) fin[c] <= src;
        for (genvar c = 0; c < NREP; c++) begin : g_c
          assign a_sk[x][c][l] = fin[c];
        end
      end
    end
  end

  for (genvar m = 0; m < TM; m++) begin : g_bsk
    for (genvar l = 0; l < TIC; l++) begin : g_l
      logic [15:0] w_e;
      assign w_e = wr_data[m][l * EW +: EW];
      if (l == 0) begin : g0
        assign b_sk[m][l] = w_e;
      end else begin : gd
        (* srl_style = "srl_reg" *) logic [15:0] sr [l];
        always_ff @(posedge clk) begin
          sr[0] <= w_e;
          for (int k = 1; k < l; k++) sr[k] <= sr[k - 1];
        end
        assign b_sk[m][l] = sr[l - 1];
      end
    end
  end

  // wfb: accumulate unless the instant is a window's first (at E + TIC)
  logic wfb;
  assign wfb = !(ctl_d[3 + TIC].v && ctl_d[3 + TIC].first);

  logic [31:0] psum [PIX2][TM];
  for (genvar x = 0; x < PIX2; x++) begin : g_px
    for (genvar m = 0; m < TM; m++) begin : g_col
      cv_mac_chain u_chain (
        .clk, .a (a_sk[x][m / (TM / NREP)]), .b (b_sk[m]), .c (seed[x][m]), .wfb, .p (psum[x][m])
      );
    end
  end

  // ===========================================================================
  // Write-back: pixel 0's word at E_last + TIC + 2, pixel 1's one cycle later.
  // ===========================================================================
  ctl_t             wbc;
  logic             hold_en, hold_par;
  logic [11:0]      hold_addr;
  logic [TM*32-1:0] hold_data, p0_word, p1_word;
  assign wbc = ctl_d[LAT];

  always_comb
    for (int m = 0; m < TM; m++) begin
      p0_word[m * 32 +: 32] = psum[0][m];
      p1_word[m * 32 +: 32] = psum[1][m];
    end

  logic wb_now;
  assign wb_now = wbc.v && wbc.last && wbc.v0;
  always_comb begin
    wb_we[0] = 1'b0;
    wb_we[1] = 1'b0;
    if (wb_now) begin
      wb_we[wbc.par] = 1'b1;
      wb_addr = wbc.w0;
      wb_data = p0_word;
    end else begin
      wb_we[hold_par] = hold_en;
      wb_addr = hold_addr;
      wb_data = hold_data;
    end
  end

  always_ff @(posedge clk) begin
    if (rst) begin
      hold_en <= 1'b0;
    end else begin
      hold_en <= wbc.v && wbc.last && wbc.v1;
    end
    if (wbc.v && wbc.last) begin
      hold_par  <= wbc.par;
      hold_addr <= wbc.w1;
      hold_data <= p1_word;
    end
  end

  // ===========================================================================
  // Finished chunks: the request is queued when the chunk's last sweep
  // starts and released once its last store is done (two cycles after the
  // write-back of the chunk's last instant).
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
  assign c_done     = ctl_d[LAT + 1].v && ctl_d[LAT + 1].cdone;
  assign dr_valid   = pend_valid && (c_ready != '0);
  assign pend_ready = dr_valid && dr_ready;
  always_ff @(posedge clk) begin
    if (rst) c_ready <= '0;
    else     c_ready <= c_ready + 3'(c_done) - 3'(pend_ready);
  end

  assign idle = !f_act && !s_act && pipe_empty && !hold_en && !pend_valid;

  logic unused;
  assign unused = ^pend_n ^ ^sd ^ ^j ^ f_t[2] ^ s_g[2] ^ pend_in_ready;

endmodule
