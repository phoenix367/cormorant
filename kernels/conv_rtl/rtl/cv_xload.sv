// ---------------------------------------------------------------------------
// cv_xload — the input reader (gmem0, the HLS x_row_loader).
//
// For every sweep that loads (the first m-group of a (chunk, channel tile,
// ow-tile)), it walks the output rows and works out, exactly as the patch
// producer does, the input rows the line buffer still lacks; each such row
// is read as one run per channel (the whole words covering the tile's input
// columns) into one half of a ping-pong row buffer, then emitted as one
// column of TIC channels per cycle.  The next row's runs are requested as
// soon as its half has been emitted, so the reads of row r + 1 overlap the
// emission of row r.
// ---------------------------------------------------------------------------
module cv_xload
  import cv_pkg::*;
(
  input  logic          clk,
  input  logic          rst,
  input  job_t          j,

  input  logic          xq_valid,
  output logic          xq_ready,
  input  sweep_t        xq,

  output logic          arvalid,
  input  logic          arready,
  output logic [63:0]   araddr,
  output logic [7:0]    arlen,
  input  logic          rvalid,
  output logic          rready,
  input  logic [BW-1:0] rdata,
  input  logic          rlast,

  output logic          col_valid,         // one input column: channel c in lane c
  input  logic          col_ready,
  output logic [TIC*EW-1:0] col_data,

  output logic          idle
);
  localparam int OW = $clog2(X_OUTS) + 1;

  // ---- row generator: the rows each output row adds to the line buffer ----------
  typedef struct packed {
    logic [31:0] ih;
    logic [31:0] x_cbase;
    logic [31:0] iw_lo;
    logic [6:0]  iw_cnt;
    logic [4:0]  ch_valid;
  } row_t;

  logic        g_act;
  sweep_t      gd;
  logic [12:0] g_oh;
  logic [31:0] g_last, g_ihmin, g_ih, g_le;
  logic        g_rows;                     // emitting rows ls..le of output row g_oh
  logic        rq_in_ready, rq_valid, rq_pop;
  row_t        rq_in, rq;
  logic [2:0]  rq_n;

  assign xq_ready = !g_act;

  // window of output row g_oh: input rows [ihmin, ihmin + reach_h]; the rows
  // it adds are [max(last + 1, ihmin, 0), min(ihmin + reach_h, in_h - 1)] —
  // two pipeline stages, valid two cycles after g_ihmin / g_last last changed
  // (g_settle)
  logic [31:0] s1_ih, s1_le, s1_last1, ls, le, g_inh1;
  logic [1:0]  g_settle;
  always_ff @(posedge clk) begin
    s1_ih    <= g_ihmin;
    s1_le    <= g_ihmin + {28'b0, j.reach_h};
    s1_last1 <= g_last + 32'd1;
    ls       <= ($signed(s1_last1) > $signed(s1_ih)) ? (s1_last1[31] ? '0 : s1_last1)
                                                     : (s1_ih[31] ? '0 : s1_ih);
    le       <= ($signed(s1_le) >= $signed(j.in_h)) ? g_inh1 : s1_le;
  end

  assign rq_in = '{ih: g_ih, x_cbase: gd.x_cbase, iw_lo: gd.iw_lo, iw_cnt: gd.iw_cnt,
                   ch_valid: gd.ch_valid};

  always_ff @(posedge clk) begin
    if (rst) begin
      g_act <= 1'b0;
    end else if (!g_act) begin
      if (xq_valid && xq.load && xq.iw_cnt != '0) begin
        g_act   <= 1'b1;
        gd      <= xq;
        g_oh    <= '0;
        g_last  <= xq.ih0 - 32'd1;
        g_ihmin <= xq.ih0;
        g_rows  <= 1'b0;
        g_inh1  <= j.in_h - 32'd1;
        g_settle <= 2'd2;
      end
    end else if (g_settle != 2'd0) begin
      g_settle <= g_settle - 2'd1;
    end else if (!g_rows) begin
      // output row g_oh: its new input rows, if any
      if ($signed(ls) <= $signed(le)) begin
        g_rows <= 1'b1;
        g_ih   <= ls;
        g_le   <= le;
      end else begin
        // nothing new; the resident-row bound still follows the HLS loader
        if ($signed(le) > $signed(g_last)) g_last <= le;
        if (g_oh + 13'd1 == gd.coh) g_act <= 1'b0;
        g_oh    <= g_oh + 13'd1;
        g_ihmin <= g_ihmin + j.sh;
        g_settle <= 2'd2;
      end
    end else if (rq_in_ready) begin
      // one row task per cycle
      g_ih <= g_ih + 32'd1;
      if (g_ih == g_le) begin
        g_rows  <= 1'b0;
        g_last  <= g_le;
        if (g_oh + 13'd1 == gd.coh) g_act <= 1'b0;
        g_oh    <= g_oh + 13'd1;
        g_ihmin <= g_ihmin + j.sh;
        g_settle <= 2'd2;
      end
    end
  end

  cv_fifo #(.W($bits(row_t)), .D(4), .BRAM(1'b0)) u_rq (
    .clk, .rst, .in_valid (g_act && g_rows), .in_ready (rq_in_ready), .in_data (rq_in),
    .out_valid (rq_valid), .out_ready (rq_pop), .out_data (rq), .count (rq_n)
  );

  // a run (<= LBC columns at any alignment) is at most LBC / E + 1 words
  if (X_BURST < LBC / E + 1) begin : g_burst_check
    $error("cv_xload: X_BURST must hold a run of LBC / E + 1 words");
  end

  // ---- issuer: one run per channel of the row, into a free half ------------------
  // A row is in half h = rows issued mod 2; a half is free once the emitter
  // has emitted the row it held.
  logic [1:0]  h_busy;                     // half holds a row not yet emitted
  logic        i_half;                     // half of the next row
  logic        i_act;
  row_t        ir;
  logic [2:0]  i_mul;                      // row base multiply in flight
  logic [31:0] i_prod, i_runoff;
  logic [4:0]  i_c;
  logic        i_second;                   // the run's second burst (4 KiB split) is next
  logic [OW-1:0] outs;
  logic [59:0] i_waddr;
  logic [3:0]  i_nw;                       // words of the current run

  // per-half row info for the emitter
  logic [TIC-1:0][2:0] h_shift [2];
  logic [4:0]  h_chv [2];
  logic [6:0]  h_cnt [2];
  logic [7:0]  h_words [2];                // words of the row
  logic [7:0]  h_got [2];                  // words received

  // run queue: which channel / half each arriving word belongs to
  typedef struct packed {
    logic        h;
    logic [3:0]  c;
    logic [3:0]  nw;
  } run_t;
  logic run_in_ready, run_valid, run_pop;
  run_t run_in, run_out;
  assign run_in = '{h: i_half, c: i_c[3:0], nw: i_nw};
  logic [5:0] run_n;

  assign rq_pop = !i_act && rq_valid && !h_busy[i_half];

  logic [8:0]  to4k;
  logic [3:0]  b_len;
  logic        b_last;
  always_comb begin
    to4k = 9'd256 - {1'b0, i_waddr[7:0]};
    if (!i_second) begin
      b_len  = (9'(i_nw) > to4k) ? to4k[3:0] : i_nw;
      b_last = (9'(i_nw) <= to4k);
    end else begin
      b_len  = i_nw - to4k[3:0];
      b_last = 1'b1;
    end
  end

  logic issue;
  assign issue = i_act && (i_mul == '0) && (!arvalid || arready) && (outs != OW'(X_OUTS)) &&
                 (i_second || run_in_ready);

  // the run's geometry follows its offset
  logic [31:0] runoff_nw_sum;
  assign runoff_nw_sum = {29'b0, i_runoff[2:0]} + {25'b0, ir.iw_cnt} + 32'd7;
  assign i_waddr = j.x_w + 60'(i_runoff[31:3]);
  assign i_nw    = runoff_nw_sum[6:3];

  always_ff @(posedge clk) begin
    if (rst) begin
      i_act    <= 1'b0;
      i_half   <= 1'b0;
      arvalid  <= 1'b0;
      outs     <= '0;
      i_second <= 1'b0;
    end else begin
      if (arvalid && arready) arvalid <= 1'b0;
      if (rq_pop) begin
        i_act   <= 1'b1;
        ir      <= rq;
        i_mul   <= 3'd5;
        i_c     <= '0;
        h_chv[i_half]   <= rq.ch_valid;
        h_cnt[i_half]   <= rq.iw_cnt;
        h_words[i_half] <= '0;
      end
      if (i_act && i_mul != '0) begin
        i_mul <= i_mul - 3'd1;
        if (i_mul == 3'd1) begin
          i_runoff <= ir.x_cbase + i_prod + ir.iw_lo;
        end
      end
      if (issue) begin
        arvalid  <= 1'b1;
        araddr   <= {(i_second ? i_waddr + 60'(to4k) : i_waddr), 4'b0};
        arlen    <= {4'b0, b_len - 4'd1};
        i_second <= !b_last;
        if (b_last) begin
          // the run is out: next channel / row done
          h_words[i_half] <= h_words[i_half] + {4'b0, i_nw};
          if (i_c + 5'd1 == ir.ch_valid) begin
            i_act  <= 1'b0;
            i_half <= !i_half;
          end else begin
            i_c      <= i_c + 5'd1;
            i_runoff <= i_runoff + j.in_hw;
          end
        end
      end
      outs <= outs + (issue ? OW'(1) : '0) - ((rvalid && rlast) ? OW'(1) : '0);
    end
    for (int c = 0; c < TIC; c++)
      if (issue && !i_second && i_c == 5'(c)) h_shift[i_half][c] <= i_runoff[2:0];
  end

  // ih * in_w for the row base: i_prod is valid 5 cycles after the pop
  logic [31:0] m_a, m_b, m_p1, m_p2;
  always_ff @(posedge clk) begin
    m_a    <= ir.ih;
    m_b    <= j.in_w;
    m_p1   <= m_a * m_b;
    m_p2   <= m_p1;
    i_prod <= m_p2;
  end

  cv_fifo #(.W($bits(run_t)), .D(32), .BRAM(1'b0)) u_run (
    .clk, .rst, .in_valid (issue && !i_second), .in_ready (run_in_ready),
    .in_data (run_in),
    .out_valid (run_valid), .out_ready (run_pop), .out_data (run_out), .count (run_n)
  );

  // ---- data: R beats into the row buffer ------------------------------------------
  logic [3:0] d_q;
  assign rready  = 1'b1;
  assign run_pop = rvalid && run_valid && (d_q + 4'd1 == run_out.nw);

  logic [TIC-1:0] rb_we;
  always_comb
    for (int c = 0; c < TIC; c++) rb_we[c] = rvalid && run_valid && (run_out.c == 4'(c));

  // The write reaches the row buffer through a register stage — the address
  // a copy per bank (it reaches every LUT of the bank), the data a copy per
  // RBG banks — and is counted when it lands: the emitter reads a half only
  // once all its words are in.
  localparam int RBG = 4;
  logic [TIC-1:0]   rbw_we;
  (* keep = "true" *) logic [4:0]    rbw_addr [TIC];
  (* keep = "true" *) logic [BW-1:0] rbw_data [TIC / RBG];
  logic             rbw_v, rbw_h;
  always_ff @(posedge clk) begin
    rbw_we <= rst ? '0 : rb_we;
    rbw_v  <= rvalid && run_valid && !rst;
    rbw_h  <= run_out.h;
    for (int c = 0; c < TIC; c++)       rbw_addr[c] <= {run_out.h, d_q};
    for (int g = 0; g < TIC / RBG; g++) rbw_data[g] <= rdata;
  end

  always_ff @(posedge clk) begin
    if (rst) begin
      d_q <= '0;
      h_got[0] <= '0;
      h_got[1] <= '0;
    end else begin
      if (rvalid) d_q <= run_pop ? 4'd0 : d_q + 4'd1;
      if (rbw_v) h_got[rbw_h] <= h_got[rbw_h] + 8'd1;
      if (e_done) h_got[e_half] <= '0;
    end
  end

  // ---- emitter: one column of all channels per cycle ------------------------------
  logic        e_half, e_done;
  logic [6:0]  e_i;
  logic [TIC*EW-1:0] e_col;
  logic [BW-1:0] rb_rd [TIC];

  for (genvar c = 0; c < TIC; c++) begin : g_rb
    logic [2:0] e_e;
    logic [6:0] e_pos;
    assign e_pos = {4'b0, h_shift[e_half][c]} + e_i;
    cv_lutram #(.W(BW), .D(32)) u_rb (
      .clk, .we (rbw_we[c]), .waddr (rbw_addr[c]), .wdata (rbw_data[c / RBG]),
      .raddr ({e_half, e_pos[6:3]}), .rdata (rb_rd[c])
    );
    assign e_e = e_pos[2:0];
    assign e_col[c * EW +: EW] = (5'(c) < h_chv[e_half]) ? rb_rd[c][e_e * EW +: EW] : '0;
  end

  // a half is ready to emit once all its words are in; a column goes through
  // an output register (o_v / o_d), which takes the next one while it drains
  logic h_full, e_go, o_v;
  logic [TIC*EW-1:0] o_d;
  assign h_full = h_busy[e_half] && !(i_act && i_half == e_half) &&
                  (h_got[e_half] == h_words[e_half]);
  assign e_go      = h_full && (!o_v || col_ready);
  assign col_valid = o_v;
  assign col_data  = o_d;
  assign e_done    = e_go && (e_i + 7'd1 == h_cnt[e_half]);

  always_ff @(posedge clk) begin
    if (rst)                    o_v <= 1'b0;
    else if (!o_v || col_ready) o_v <= h_full;
    if (e_go) o_d <= e_col;
  end

  always_ff @(posedge clk) begin
    if (rst) begin
      e_half <= 1'b0;
      e_i    <= '0;
      h_busy <= '0;
    end else begin
      if (rq_pop) h_busy[i_half] <= 1'b1;
      if (e_go) begin
        if (e_done) begin
          e_i            <= '0;
          e_half         <= !e_half;
          h_busy[e_half] <= 1'b0;
        end else begin
          e_i <= e_i + 7'd1;
        end
      end
    end
  end

  assign idle = !g_act && !rq_valid && !i_act && !arvalid && (outs == '0) && (h_busy == '0) && !o_v;

  logic unused;
  assign unused = ^rq_n ^ ^run_n ^ ^j ^ ^gd ^ ^ir;

endmodule
