// Copyright 2026 Ivan Gubochkin
// SPDX-License-Identifier: Apache-2.0

// ---------------------------------------------------------------------------
// ro_fold — OP_FOLD (op 9, ROWOPS_RTL_KERNEL.md §1.4, HOSTOPS_PLAN.md §11):
// Piper's TtsPrep.  Output channel k reads the int16 source row at element
// address A_k = a / 2 + k * a_s2 / 2; output row r, column j reads its column
// base_r + j, base_r = s0 + r * w0, where that column lies in [win_lo, win_hi),
// else the element is 0; every element read becomes
//
//   y = rne(x / (m_x * 2^s)),  m_x = m for x < 0, 1 for x >= 0
//
// (rowops_spec.fold_elem; LeakyReLU with alpha = 1/m and the round half to
// even s bits coarser).  The output is one contiguous run of blocks x rows x
// n int16 at c, written as whole words (the last word's lanes past the end 0).
//
//   sequencer ─► row descriptors ─┐
//       └──────► read runs ─► stream A's port (external runs) ─┐
//                                 ▼                             ▼
//            step control (a two-word window per row) ─► lanes ─► fold ─► packer
//                                                                          ▼
//                       out FIFO ─► burst former ─► write port (external bursts)
//
// Sequencer: per row (six cycles, 64-bit element arithmetic) the valid part
// [lo_e, hi_e) = [max(win_lo, base), min(win_hi, base + n)) and the words it
// lies in, [f, last] — one read run (none when the part is empty) — and the
// datapath's descriptor: the start lane delta = (A_k + base) mod 8, the
// column range [jlo, jhi) = [lo_e, hi_e) - base, the steps before the first
// word is needed and the word count.
// Datapath: a row of n columns is G = ceil(n / 8) groups of 8; group g needs
// the words Q0 + g and Q0 + g + 1 (Q0 = (A_k + base) >> 3): each step shifts
// the next word of the run (or a zero word outside it) into a two-word
// window, lane l takes window element delta + l, the lanes outside [jlo, jhi)
// are zeroed.  A row whose first word is Q0 itself takes one extra step to
// load it.  The fold without a divider: u = 2|x|, R_x = R = ceil(2^24 / m)
// for x < 0 (2^24 for x >= 0), P = u * R_x (one DSP), q1 = P >> 24 =
// floor(u / m_x) and (P mod 2^24) < R_x says the remainder is 0 (exact for
// |x| <= 32768 and m <= 127), then the round at s + 1 bits and the sign
// (rowops_spec.fold_recip).  The packer appends each group (8 columns, the
// row's last n mod 8) to the contiguous output.
// Bursts: the output words are cut into bursts of at most WR_BURST words
// within 4 KiB, each offered to the write port after its last word.
// ---------------------------------------------------------------------------
module ro_fold
  import ro_pkg::*;
(
  input  logic          clk,
  input  logic          rst,
  input  logic          start,            // one cycle; the job constants are stable from here on
  input  logic          en,               // a FOLD job with work (n, rows, blocks, m non-zero)
  input  logic [63:0]   a,                // byte address of source column 0 of output channel 0
  input  logic [31:0]   a_s2,             // signed bytes between the channels' source rows
  input  logic [31:0]   w0,               // source columns between rows
  input  logic [31:0]   s0,               // signed source column of row 0, column 0
  input  logic [31:0]   n,                // the output row
  input  logic [31:0]   rows,             // rows per channel
  input  logic [31:0]   blocks,           // channels
  input  logic [31:0]   win_lo,           // the source columns read: [win_lo, win_hi)
  input  logic [31:0]   win_hi,
  input  logic [3:0]    s,                // the rounding shift
  input  logic [24:0]   rcp,              // ceil(2^24 / m)
  input  logic [63:0]   c,                // the output (16-byte aligned)

  // the read runs of stream A (ro_rd_port's external runs)
  output logic          rr_valid,
  input  logic          rr_ready,
  output logic [59:0]   rr_waddr,
  output logic [31:0]   rr_words,

  input  logic          a_valid,
  output logic          a_ready,
  input  logic [BW-1:0] a_data,

  // the output words and their bursts (ro_wr_port's external bursts)
  output logic          c_valid,
  input  logic          c_ready,
  output logic [BW-1:0] c_data,
  output logic          wb_valid,
  input  logic          wb_ready,
  output logic [59:0]   wb_waddr,
  output logic [8:0]    wb_len,
  output logic          wb_done,

  output logic          idle
);
  // Job constants ---------------------------------------------------------------------------
  logic        k_en;
  logic signed [63:0] k_cs;                // a_s2 / 2, sign-extended
  logic signed [63:0] k_w0, k_s0, k_n, k_wlo, k_whi;
  logic [31:0] k_rows;
  logic [3:0]  k_s;
  logic [24:0] k_rcp;
  logic [16:0] k_lowm;                     // 2^(s + 1) - 1
  logic [15:0] k_half;                     // 2^s
  logic [29:0] k_g;                        // groups per row: ceil(n / 8)
  logic [3:0]  k_clast;                    // columns of a row's last group: (n - 1) mod 8 + 1
  always_ff @(posedge clk)
    if (start) begin
      k_en     <= en;
      k_cs     <= {{33{a_s2[31]}}, a_s2[31:1]};
      k_w0     <= {32'b0, w0};
      k_s0     <= {{32{s0[31]}}, s0};
      k_n      <= {32'b0, n};
      k_wlo    <= {32'b0, win_lo};
      k_whi    <= {32'b0, win_hi};
      k_rows   <= rows;
      k_s      <= s;
      k_rcp    <= rcp;
      k_lowm   <= 17'((18'd2 << s) - 18'd1);
      k_half   <= 16'd1 << s;
      k_g      <= 30'(({1'b0, n} + 33'd7) >> 3);
      k_clast  <= {1'b0, 3'(n - 32'd1)} + 4'd1;
    end

  // Sequencer -----------------------------------------------------------------------------
  // Row descriptor: {last row of the job, pre, delta, pstart, words, jlo, jhi}.
  localparam int RDW = 1 + 1 + 3 + 32 + 32 + 32 + 32;
  typedef enum logic [2:0] {Q_IDLE, Q_S0, Q_S1, Q_S2, Q_S3, Q_S4, Q_PUSH} qst_t;
  qst_t               qs;
  logic [31:0]        q_kleft, q_rleft;    // channels / rows of the channel not finished, current included
  logic signed [63:0] q_A, q_base;         // the channel's element address, the row's source column
  logic signed [63:0] q_e0, q_hend, q_lo, q_hi, q_f, q_hm1, q_lx;
  logic signed [63:0] q_jlo, q_jhi;
  logic               q_vld, q_lastrow;
  logic [63:0]        q_fo, q_nw;
  logic               rd_in_ready, rq_in_ready, q_push;
  logic [RDW-1:0]     q_desc;

  assign q_push = (qs == Q_PUSH) && rd_in_ready && (q_nw == '0 || rq_in_ready);

  always_ff @(posedge clk) begin
    if (rst) begin
      qs <= Q_IDLE;
    end else if (start) begin
      qs <= en ? Q_S0 : Q_IDLE;
    end else begin
      case (qs)
        Q_S0: qs <= Q_S1;
        Q_S1: qs <= Q_S2;
        Q_S2: qs <= Q_S3;
        Q_S3: qs <= Q_S4;
        Q_S4: qs <= Q_PUSH;
        Q_PUSH: if (q_push) qs <= q_lastrow ? Q_IDLE : Q_S0;
        default: qs <= Q_IDLE;
      endcase
    end
  end

  always_ff @(posedge clk) begin
    if (start) begin
      q_A     <= {1'b0, a[63:1]};
      q_base  <= {{32{s0[31]}}, s0};
      q_kleft <= blocks;
      q_rleft <= rows;
    end else begin
      case (qs)
        Q_S0: begin
          q_e0      <= q_A + q_base;
          q_hend    <= q_base + k_n;
          q_lo      <= (q_base > k_wlo) ? q_base : k_wlo;
          q_lastrow <= (q_rleft == 32'd1) && (q_kleft == 32'd1);
        end
        Q_S1: q_hi <= (q_hend < k_whi) ? q_hend : k_whi;
        Q_S2: begin
          q_vld <= q_hi > q_lo;
          q_f   <= q_A + q_lo;
          q_hm1 <= q_hi - 64'sd1;
          q_jlo <= q_lo - q_base;
          q_jhi <= q_hi - q_base;
        end
        Q_S3: begin
          q_lx <= q_A + q_hm1;
          q_fo <= 64'((q_f >>> 3) - (q_e0 >>> 3));
        end
        Q_S4: q_nw <= q_vld ? 64'((q_lx >>> 3) - (q_f >>> 3) + 64'sd1) : 64'd0;
        Q_PUSH: if (q_push) begin
          if (q_rleft == 32'd1) begin            // the channel ends
            q_rleft <= k_rows;
            q_kleft <= q_kleft - 32'd1;
            q_A     <= q_A + k_cs;
            q_base  <= k_s0;
          end else begin
            q_rleft <= q_rleft - 32'd1;
            q_base  <= q_base + k_w0;
          end
        end
        default: ;
      endcase
    end
  end

  // pre: the row's first word is Q0 (one step loads it); pops start at step pstart
  always_comb begin
    logic        pre;
    logic [31:0] pstart;
    pre    = q_vld && (q_fo == '0);
    pstart = (q_fo == '0) ? 32'd0 : 32'(q_fo - 64'd1);
    q_desc = {q_lastrow, pre, q_e0[2:0], pstart, 32'(q_nw),
              q_vld ? 32'(q_jlo) : 32'd0, q_vld ? 32'(q_jhi) : 32'd0};
  end

  logic           rd_valid, rd_pop;
  logic [RDW-1:0] rd_data;
  logic [4:0]     rd_count;
  logic [2:0]     rq_count;
  ro_fifo #(.W(RDW), .D(16), .BRAM(1'b0)) u_rdq (
    .clk, .rst (rst || start),
    .in_valid (q_push), .in_ready (rd_in_ready), .in_data (q_desc),
    .out_valid (rd_valid), .out_ready (rd_pop), .out_data (rd_data),
    .count (rd_count)
  );

  ro_fifo #(.W(92), .D(4), .BRAM(1'b0)) u_rrq (
    .clk, .rst (rst || start),
    .in_valid (q_push && q_nw != '0), .in_ready (rq_in_ready),
    .in_data ({60'(q_f >>> 3), 32'(q_nw)}),
    .out_valid (rr_valid), .out_ready (rr_ready), .out_data ({rr_waddr, rr_words}),
    .count (rq_count)
  );

  // Step control ---------------------------------------------------------------------------
  logic          row_on;                   // a row is being stepped
  logic          r_pre, r_last;            // the load step is next; the row is the job's last
  logic [2:0]    r_dl;                     // delta
  logic [31:0]   r_skip, r_pops;           // steps before the first pop; words still to pop
  logic [29:0]   r_gleft;                  // groups still to emit
  logic signed [32:0] r_dlo, r_dhi;        // jlo / 8 - g, jhi / 8 - g
  logic [7:0]    r_lom, r_him;             // lanes >= jlo mod 8, lanes < jhi mod 8
  logic [BW-1:0] r_hi;                     // the window's newer word

  logic          room, need_pop, step, emit;
  assign need_pop = (r_skip == '0) && (r_pops != '0);
  assign step     = row_on && (!need_pop || a_valid) && room;
  assign emit     = step && !r_pre && (r_gleft != '0);
  assign a_ready  = step && need_pop;
  assign rd_pop   = rd_valid && !row_on && k_en;

  logic [BW-1:0] new_w;
  assign new_w = need_pop ? a_data : '0;

  logic [7:0] g_mask;
  always_comb begin
    logic [7:0] lower, upper;
    lower  = r_dlo[32] ? 8'hFF : (r_dlo == '0) ? r_lom : 8'h00;
    upper  = (!r_dhi[32] && r_dhi != '0) ? 8'hFF : (r_dhi == '0) ? r_him : 8'h00;
    g_mask = lower & upper;
  end

  always_ff @(posedge clk) begin
    if (rst || start) begin
      row_on <= 1'b0;
    end else if (rd_pop) begin
      logic [31:0] jlo, jhi;
      jlo     = rd_data[63:32];
      jhi     = rd_data[31:0];
      row_on  <= 1'b1;
      r_last  <= rd_data[RDW-1];
      r_pre   <= rd_data[RDW-2];
      r_dl    <= rd_data[RDW-3 -: 3];
      r_skip  <= rd_data[127:96];
      r_pops  <= rd_data[95:64];
      r_gleft <= k_g;
      r_dlo   <= {4'b0, jlo[31:3]};
      r_dhi   <= {4'b0, jhi[31:3]};
      r_lom   <= 8'hFF << jlo[2:0];
      r_him   <= 8'((9'd1 << jhi[2:0]) - 9'd1);
      r_hi    <= '0;
    end else if (step) begin
      r_hi   <= new_w;
      r_skip <= r_skip - ((r_skip != '0) ? 32'd1 : 32'd0);
      r_pops <= r_pops - (need_pop ? 32'd1 : 32'd0);
      if (r_pre) begin
        r_pre <= 1'b0;
      end else if (r_gleft != '0) begin
        r_gleft <= r_gleft - 30'd1;
        r_dlo   <= r_dlo - 33'sd1;
        r_dhi   <= r_dhi - 33'sd1;
      end
      // the row ends with its last group and its last word (a word left over
      // would only come from inconsistent registers: it is drained, never left
      // in the port)
      if ((r_gleft == '0 || (!r_pre && r_gleft == 30'd1)) &&
          (r_pops == '0 || (need_pop && r_pops == 32'd1)))
        row_on <= 1'b0;
    end
  end

  // Pipeline (v[i]: stage s(i+1) holds a group): s1 window, s2 lanes, s3 |x|, s4 - s6 the product, s7 q1 / z,
  // s8 the shift, s9 the rounding, s10 the sign -> the packer.
  localparam int NS = 10;
  logic [NS-1:0] v;
  logic [3:0]    cnt  [NS];               // columns of the group (1..8)
  logic [NS-1:0] jend;                    // the job's last group

  logic [2*BW-1:0] win1;
  logic [2:0]      dl1;
  logic [7:0]      msk1;
  logic signed [15:0] x2 [E];
  logic        neg3 [E], neg4 [E], neg5 [E], neg6 [E], neg7 [E], neg8 [E], neg9 [E];
  logic [16:0] u3 [E];
  logic [24:0] rx3 [E];
  logic [41:0] p4 [E], p5 [E], p6 [E];
  logic [17:0] q1_7 [E];
  logic        z7 [E], z8 [E];
  logic [16:0] q8 [E];
  logic [16:0] low8 [E];
  logic [15:0] y9 [E];
  logic [15:0] y10 [E];

  always_ff @(posedge clk) begin
    if (rst || start) v <= '0;
    else              v <= {v[NS-2:0], emit};
    cnt[0]  <= (r_gleft == 30'd1) ? k_clast : 4'd8;
    jend[0] <= r_last && (r_gleft == 30'd1);
    for (int i = 1; i < NS; i++) begin
      cnt[i]  <= cnt[i-1];
      jend[i] <= jend[i-1];
    end
    // s1: the window, its lanes, the mask
    win1 <= {new_w, r_hi};
    dl1  <= r_dl;
    msk1 <= g_mask;
    // s2: lane l = window element delta + l, masked
    for (int l = 0; l < E; l++) begin
      logic [3:0] t;
      t = 4'(dl1) + 4'(l);
      x2[l] <= msk1[l] ? signed'(win1[EW*t +: EW]) : 16'sd0;
    end
    for (int l = 0; l < E; l++) begin
      // s3: u = 2|x|, R_x
      neg3[l] <= x2[l][15];
      u3[l]   <= x2[l][15] ? 17'(-(17'(x2[l])) << 1) : 17'({1'b0, x2[l]} << 1);
      rx3[l]  <= x2[l][15] ? k_rcp : 25'h1000000;
      // s4 - s6: P = u * R_x (17 x 25, DSP), registered twice more
      p4[l]   <= 42'(u3[l]) * 42'(rx3[l]);
      p5[l]   <= p4[l];
      p6[l]   <= p5[l];
      neg4[l] <= neg3[l];
      neg5[l] <= neg4[l];
      neg6[l] <= neg5[l];
      // s7: q1 = floor(u / m_x); z: the remainder is 0
      q1_7[l] <= p6[l][41:24];
      z7[l]   <= !neg6[l] || ({1'b0, p6[l][23:0]} < k_rcp);
      neg7[l] <= neg6[l];
      // s8: q = q1 >> (s + 1), the bits below
      q8[l]   <= 17'(q1_7[l] >> (5'(k_s) + 5'd1));
      low8[l] <= 17'(q1_7[l]) & k_lowm;
      z8[l]   <= z7[l];
      neg8[l] <= neg7[l];
      // s9: round half to even
      y9[l]   <= 16'(q8[l]) + ((low8[l] > 17'(k_half)) ||
                               ((low8[l] == 17'(k_half)) && (!z8[l] || q8[l][0])) ? 16'd1 : 16'd0);
      neg9[l] <= neg8[l];
      // s10: the sign
      y10[l]  <= neg9[l] ? 16'(-y9[l]) : y9[l];
    end
  end

  // Packer ------------------------------------------------------------------------------------
  // acc holds o columns (lanes 0 .. o-1; the rest 0); a group of cnt columns
  // (its lanes past cnt are 0) is placed at lane o of a 16-lane line.
  logic [EW-1:0] acc [E];
  logic [3:0]    o;
  logic          pk_v, pk_last, flush;
  logic [BW-1:0] pk_d;

  always_ff @(posedge clk) begin
    if (rst || start) begin
      pk_v  <= 1'b0;
      flush <= 1'b0;
      o     <= '0;
      for (int l = 0; l < E; l++) acc[l] <= '0;
    end else begin
      pk_v <= 1'b0;
      if (v[NS-1]) begin
        logic [EW-1:0] line [2*E];
        logic [4:0]    fill;
        for (int t = 0; t < 2 * E; t++) begin
          logic [4:0] i;
          i = 5'(t) - 5'(o);
          line[t] = (t < int'(o)) ? acc[t] : (i < 5'd8) ? y10[i[2:0]] : '0;
        end
        fill = 5'(o) + 5'(cnt[NS-1]);
        if (fill >= 5'd8) begin
          pk_v    <= 1'b1;
          pk_last <= jend[NS-1] && (fill == 5'd8);
          for (int l = 0; l < E; l++) begin
            pk_d[EW*l +: EW] <= line[l];
            acc[l] <= line[E + l];
          end
          o     <= 4'(fill - 5'd8);
          flush <= jend[NS-1] && (fill != 5'd8);
        end else begin
          for (int l = 0; l < E; l++) acc[l] <= line[l];
          o     <= 4'(fill);
          flush <= jend[NS-1];
        end
      end else if (flush) begin                 // the job's last, partial word
        pk_v    <= 1'b1;
        pk_last <= 1'b1;
        for (int l = 0; l < E; l++) pk_d[EW*l +: EW] <= acc[l];
        flush <= 1'b0;
        o     <= '0;
      end
    end
  end

  // Output FIFO and credits: every emitted group reserves two entries (the job's
  // last group may leave two words) until it has left the pipeline.
  localparam int CF_D = 32;
  localparam int FW   = $clog2(CF_D);
  logic        cf_valid, cf_pop, cf_last, cf_in_ready;
  logic [FW:0] cf_count, inflight;
  logic [BW-1:0] cf_data;
  ro_fifo #(.W(BW + 1), .D(CF_D), .BRAM(1'b0)) u_cf (
    .clk, .rst (rst || start),
    .in_valid (pk_v), .in_ready (cf_in_ready), .in_data ({pk_last, pk_d}),
    .out_valid (cf_valid), .out_ready (cf_pop), .out_data ({cf_last, cf_data}),
    .count (cf_count)
  );
  always_ff @(posedge clk) begin
    if (rst || start) inflight <= '0;
    else inflight <= inflight + (FW+1)'(emit) - (FW+1)'(v[NS-1]);
  end
  assign room = ({1'b0, cf_count} + {1'b0, inflight}) < (FW+2)'(CF_D - 4);

  // Bursts ---------------------------------------------------------------------------------------
  logic [59:0] b_next, b_start;            // the next word's address, the open burst's first
  logic [6:0]  b_len;                      // words of the open burst
  logic        bq_in_ready, b_end, b_done;
  assign b_end   = (b_len == 7'(WR_BURST - 1)) || (b_next[7:0] == 8'hFF) || cf_last;
  assign cf_pop  = cf_valid && c_ready && (!b_end || bq_in_ready);
  assign c_valid = cf_valid && (!b_end || bq_in_ready);
  assign c_data  = cf_data;

  always_ff @(posedge clk) begin
    if (rst) begin
      b_done <= 1'b1;
    end else if (start) begin
      b_next  <= c[63:4];
      b_start <= c[63:4];
      b_len   <= '0;
      b_done  <= !en;
    end else if (cf_pop) begin
      b_next <= b_next + 60'd1;
      if (b_end) begin
        b_len   <= '0;
        b_start <= b_next + 60'd1;
        if (cf_last) b_done <= 1'b1;
      end else begin
        b_len <= b_len + 7'd1;
      end
    end
  end

  logic       bq_valid;
  logic [3:0] bq_count;
  ro_fifo #(.W(69), .D(8), .BRAM(1'b0)) u_bq (
    .clk, .rst (rst || start),
    .in_valid (cf_pop && b_end), .in_ready (bq_in_ready), .in_data ({b_start, 9'(b_len) + 9'd1}),
    .out_valid (bq_valid), .out_ready (wb_ready), .out_data ({wb_waddr, wb_len}),
    .count (bq_count)
  );
  assign wb_valid = bq_valid;
  assign wb_done  = b_done && !bq_valid;

  assign idle = (qs == Q_IDLE) && !rd_valid && !rr_valid && !row_on && (v == '0) && !pk_v && !flush &&
                !cf_valid && wb_done;

  logic unused;
  assign unused = ^{cf_in_ready, rd_count, rq_count, bq_count};

endmodule
