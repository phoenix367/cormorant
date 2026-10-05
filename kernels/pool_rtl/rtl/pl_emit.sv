// ---------------------------------------------------------------------------
// pl_emit — the line buffer and the window emitter (the HLS window_emitter).
//
// Line buffer: TC channels x E column banks of LBR rows x LBW words (16-bit
// LUTRAMs, one write and one read port each).  A word of a (row, channel) run
// is written in one cycle, its lanes rotated by the run's alignment so bank b
// takes the lane whose local column is b (mod E); lanes outside the run are
// dropped.  A window beat reads, for each of the OWP positions, one column of
// all TC channels (the positions of a group land in different banks: PoolGeometry::gw).
//
// Per chunk, per output row ("step"): load the input rows output row oh+1
// needs while emitting row oh (prefetch: the new rows never alias the window in
// use), or load row oh's rows first and then emit (a window too tall for that).
// A step ends when its loads are written and its beats have read the buffer.
// Emission: per group of OWP positions, pool_h x pool_w beats; out-of-bounds
// taps, positions past the W-tile and channels past c_valid carry the pool
// type's identity.  The AVG denominators (in-bounds rows x columns, or
// pool_h * pool_w with count_include_pad) are counted over the group's taps
// and travel with its last beat.
// ---------------------------------------------------------------------------
module pl_emit
  import pl_pkg::*;
(
  input  logic          clk,
  input  logic          rst,
  input  job_t          j,

  input  logic          cq_valid,
  output logic          cq_ready,
  input  chunk_t        cq,

  input  logic          dq_valid,
  output logic          dq_ready,
  input  lbd_t          dq_data,
  input  logic          wq_valid,
  output logic          wq_ready,
  input  logic [BW-1:0] wq_data,

  output logic          bt_valid,
  input  logic          bt_ready,
  output beat_t         bt,

  output logic          idle
);
  localparam int BD = 16;                  // beat FIFO depth

  // Chunk / step control ----------------------------------------------------------------
  typedef enum logic [1:0] {S_IDLE, S_SET0, S_SET1, S_RUN} st_t;
  st_t         st;
  chunk_t      ck;
  logic [31:0] oh, ih0;                    // step's output row and its first input row (signed)
  logic [31:0] tgt, lrow, loaded;          // rows allowed / started / written this chunk
  logic        emit;                       // the step emits a row (oh >= 0)
  // the row the step loads for: oh_t = oh + 1 with prefetch, else oh; it needs
  // input rows up to oh_t * stride_h - pad_top + reach_h, i.e. nh rows (signed)
  logic [31:0] oht, nh;
  logic [31:0] lc1;                        // SET0 -> SET1: min(nh, in_h)
  logic        lh_ok;
  logic        step_end, wg_act, w0_v, w1_v, lw_v, cur_v;

  assign cq_ready = (st == S_IDLE);

  always_ff @(posedge clk) begin
    if (rst) begin
      st <= S_IDLE;
    end else begin
      case (st)
        S_IDLE: if (cq_valid) begin
          ck     <= cq;
          oh     <= j.prefetch ? 32'hFFFF_FFFF : 32'd0;
          ih0    <= (j.prefetch ? (32'd0 - j.stride_h) : 32'd0) - j.pad_top;
          oht    <= '0;
          nh     <= j.reach_h + 32'd1 - j.pad_top;
          tgt    <= '0;
          st     <= S_SET0;
        end
        S_SET0: begin
          lc1   <= ($signed(nh) > $signed(j.in_h)) ? j.in_h : nh;
          lh_ok <= (oht < j.out_h);
          emit  <= !oh[31];
          st    <= S_SET1;
        end
        S_SET1: begin
          if (lh_ok && ($signed(lc1) > $signed(tgt))) tgt <= lc1;
          st <= S_RUN;
        end
        S_RUN: if (step_end) begin
          if (oh + 32'd1 == j.out_h) begin
            st <= S_IDLE;
          end else begin
            oh  <= oh + 32'd1;
            ih0 <= ih0 + j.stride_h;
            oht <= oht + 32'd1;
            nh  <= nh + j.stride_h;
            st  <= S_SET0;
          end
        end
        default: st <= S_IDLE;
      endcase
    end
  end

  // Line-buffer writer ------------------------------------------------------------------
  lbd_t       cur;                         // the run being written
  logic [3:0] cq_w;                        // its next word
  logic       ld_desc, ld_word, last_word;

  assign last_word = cur_v && (cq_w + 4'd1 == cur.nw);
  assign ld_word   = cur_v && wq_valid;
  assign ld_desc   = (st == S_RUN) && dq_valid && ($signed(lrow) < $signed(tgt)) &&
                     (!cur_v || (ld_word && last_word));
  assign dq_ready  = ld_desc;
  assign wq_ready  = ld_word;

  logic [BW-1:0] lw_word;
  lbd_t          lw_d;
  logic [3:0]    lw_q;

  always_ff @(posedge clk) begin
    if (rst || st == S_IDLE) begin
      cur_v  <= 1'b0;
      lw_v   <= 1'b0;
      lrow   <= '0;
      loaded <= '0;
    end else begin
      if (ld_desc) begin
        cur   <= dq_data;
        cur_v <= 1'b1;
        cq_w  <= '0;
        if (dq_data.row_last) lrow <= lrow + 32'd1;
      end else if (ld_word && last_word) begin
        cur_v <= 1'b0;
      end else if (ld_word) begin
        cq_w <= cq_w + 4'd1;
      end
      lw_v <= ld_word;
      if (ld_word) begin
        lw_word <= wq_data;
        lw_d    <= cur;
        lw_q    <= cq_w;
      end
      if (lw_v && (lw_q + 4'd1 == lw_d.nw) && lw_d.row_last) loaded <= loaded + 32'd1;
    end
  end

  // write port of every bank: bank b takes lane (b + shift) mod E
  logic          lb_we   [TC][E];
  logic [6:0]    lb_wa   [E];
  logic [EW-1:0] lb_wd   [E];
  logic          lb_ok   [E];
  always_comb begin
    for (int b = 0; b < E; b++) begin
      logic [3:0] bs;
      logic [7:0] col;
      logic [3:0] w;
      bs  = 4'(b) + 4'(lw_d.shift);
      col = {lw_q, 3'(b)} - (bs[3] ? 8'd8 : 8'd0);
      w   = bs[3] ? ((lw_q == 0) ? 4'd0 : lw_q - 4'd1) : lw_q;
      lb_ok[b] = lw_v && !col[7] && (col < {1'b0, ck.run_len});
      lb_wa[b] = {lw_d.slot, w[2:0]};
      lb_wd[b] = lw_word[EW*bs[2:0] +: EW];
      for (int c = 0; c < TC; c++) lb_we[c][b] = lb_ok[b] && (lw_d.ch == 3'(c));
    end
  end

  // Window generator ---------------------------------------------------------------------
  logic [6:0]  g, gpos;
  logic [2:0]  khi, kwi, kwc0, kwc1, nvk;
  logic [31:0] gcol, colk, colk1, ih;      // signed: group / tap columns, tap row
  logic [31:0] gw_sw;
  logic [4:0]  bt_count;
  logic        adv, en;

  assign gw_sw = j.gw2 ? {j.stride_w[30:0], 1'b0} : j.stride_w;
  assign en    = (st == S_RUN) && emit && wg_act &&
                 (j.prefetch || ((loaded == tgt) && !lw_v && !cur_v));
  assign adv   = en && (32'(bt_count) + 32'(w0_v) + 32'(w1_v) < BD);

  // this tap
  logic        ih_ok, ok0, ok1, kt0, kt1, first, last;
  logic [2:0]  kw0_n, kw1_n, nvk_n;
  always_comb begin
    ih_ok = !ih[31] && (ih < j.in_h);
    ok0   = ih_ok && (colk  < 32'(ck.run_len));
    ok1   = ih_ok && (colk1 < 32'(ck.run_len)) && j.gw2 && (gpos + 7'd1 < ck.ow_span);
    kt0   = ($signed(colk)  >= $signed(ck.lo_lim)) && ($signed(colk)  < $signed(ck.hi_lim));
    kt1   = ($signed(colk1) >= $signed(ck.lo_lim)) && ($signed(colk1) < $signed(ck.hi_lim));
    first = (khi == 3'd0) && (kwi == 3'd0);
    last  = (khi == j.pool_h - 3'd1) && (kwi == j.pool_w - 3'd1);
    kw0_n = (first ? 3'd0 : kwc0) + 3'((khi == 3'd0) && kt0);
    kw1_n = (first ? 3'd0 : kwc1) + 3'((khi == 3'd0) && kt1);
    nvk_n = (first ? 3'd0 : nvk)  + 3'((kwi == 3'd0) && ih_ok);
  end

  // W0: the tap's banks, words, validity
  logic [2:0] w0_bank0, w0_bank1, w0_word0, w0_word1;
  logic [3:0] w0_slot;
  logic       w0_ok0, w0_ok1, w0_first, w0_last;
  logic [5:0] w0_d0, w0_d1;

  always_ff @(posedge clk) begin
    if (rst || st == S_IDLE) begin
      wg_act <= 1'b0;
      w0_v   <= 1'b0;
    end else begin
      w0_v <= adv;
      if (st == S_SET1) begin
        wg_act <= (ck.n_groups != '0);
        g      <= '0;
        gpos   <= '0;
        khi    <= '0;
        kwi    <= '0;
        gcol   <= ck.col_base;
        colk   <= ck.col_base;
        colk1  <= ck.col_base + j.stride_w;
        ih     <= ih0;
      end else if (adv) begin
        w0_bank0 <= colk[2:0];   w0_word0 <= colk[5:3];
        w0_bank1 <= colk1[2:0];  w0_word1 <= colk1[5:3];
        w0_slot  <= ih_ok ? ih[3:0] : 4'd0;
        w0_ok0   <= ok0;
        w0_ok1   <= ok1;
        w0_first <= first;
        w0_last  <= last;
        w0_d0    <= j.cip ? j.denom_all : 6'(nvk_n * kw0_n);
        w0_d1    <= j.cip ? j.denom_all : 6'(nvk_n * kw1_n);
        kwc0 <= kw0_n;
        kwc1 <= kw1_n;
        nvk  <= nvk_n;
        if (kwi == j.pool_w - 3'd1) begin
          kwi <= '0;
          if (khi == j.pool_h - 3'd1) begin
            khi   <= '0;
            ih    <= ih0;
            g     <= g + 7'd1;
            gpos  <= gpos + (j.gw2 ? 7'd2 : 7'd1);
            gcol  <= gcol + gw_sw;
            colk  <= gcol + gw_sw;
            colk1 <= gcol + gw_sw + j.stride_w;
            if (g + 7'd1 == ck.n_groups) wg_act <= 1'b0;
          end else begin
            khi   <= khi + 3'd1;
            ih    <= ih + j.dil_h;
            colk  <= gcol;
            colk1 <= gcol + j.stride_w;
          end
        end else begin
          kwi   <= kwi + 3'd1;
          colk  <= colk + j.dil_w;
          colk1 <= colk1 + j.dil_w;
        end
      end
    end
  end

  // W1: one read per bank (position 0 wins a shared bank: then position 1 is padded)
  logic [EW-1:0] rv     [TC][E];
  logic [6:0]    lb_ra  [E];
  logic [EW-1:0] lb_rd  [TC][E];
  always_comb
    for (int b = 0; b < E; b++)
      lb_ra[b] = (w0_bank0 == 3'(b)) ? {w0_slot, w0_word0} : {w0_slot, w0_word1};

  for (genvar c = 0; c < TC; c++) begin : g_ch
    for (genvar b = 0; b < E; b++) begin : g_bank
      pl_lutram #(.W(EW), .D(LBR * LBW)) u_lb (
        .clk, .we (lb_we[c][b]), .waddr (lb_wa[b]), .wdata (lb_wd[b]),
        .raddr (lb_ra[b]), .rdata (lb_rd[c][b])
      );
    end
  end

  logic [2:0] w1_bank0, w1_bank1;
  logic       w1_ok0, w1_ok1, w1_first, w1_last;
  logic [5:0] w1_d0, w1_d1;
  always_ff @(posedge clk) begin
    if (rst || st == S_IDLE) w1_v <= 1'b0;
    else                     w1_v <= w0_v;
    if (w0_v) begin
      rv       <= lb_rd;
      w1_bank0 <= w0_bank0;  w1_bank1 <= w0_bank1;
      w1_ok0   <= w0_ok0;    w1_ok1   <= w0_ok1;
      w1_first <= w0_first;  w1_last  <= w0_last;
      w1_d0    <= w0_d0;     w1_d1    <= w0_d1;
    end
  end

  // W2: pick each position's bank, pad the rest; into the beat FIFO
  beat_t         bt_in;
  logic [EW-1:0] pad;
  assign pad = (j.mode == M_MAX) ? DATA_MIN : 16'h0000;
  always_comb begin
    for (int c = 0; c < TC; c++) begin
      logic cv;
      cv = (4'(c) < ck.c_valid);
      bt_in.v[(0 * TC + c) * EW +: EW] = (w1_ok0 && cv) ? rv[c][w1_bank0] : pad;
      bt_in.v[(1 * TC + c) * EW +: EW] = (w1_ok1 && cv) ? rv[c][w1_bank1] : pad;
    end
    bt_in.first = w1_first;
    bt_in.last  = w1_last;
    bt_in.d0    = w1_d0;
    bt_in.d1    = w1_d1;
  end

  logic bt_in_ready;
  pl_fifo #(.W($bits(beat_t)), .D(BD), .BRAM(1'b0)) u_bt (
    .clk, .rst,
    .in_valid (w1_v),     .in_ready (bt_in_ready), .in_data (bt_in),
    .out_valid(bt_valid), .out_ready(bt_ready),    .out_data(bt),
    .count    (bt_count)
  );

  // The step ends when its rows are written and its beats have read the buffer.
  assign step_end = (st == S_RUN) && (!emit || !wg_act) && (loaded == tgt) &&
                    !cur_v && !lw_v && !w0_v && !w1_v;

  assign idle = (st == S_IDLE) && !bt_valid;

  logic unused;
  assign unused = bt_in_ready ^ ^g ^ ^j;

endmodule
