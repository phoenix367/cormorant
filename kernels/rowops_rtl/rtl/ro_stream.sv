// ---------------------------------------------------------------------------
// ro_stream — the element-wise ops of the RowOpsKernel (ROWOPS_RTL_KERNEL.md
// §1.4): COPY, ADDX, DEQUANT, MULACT.  Stream A (and B) words in, C words
// out, one group of four lanes per cycle (an int16 word in two halves, an
// int32 / float32 word whole), with every column's parameters from ro_param.
// Per lane (64-bit integer arithmetic, rne = round half to even):
//
//   COPY / ADDX  y = sat_C(rne_shr((a << sh_a) [+ (b << sh_b)] + (bq << bq_sh), sh_c)), relu
//   DEQUANT      y = float32(a * 2^-sh_c)
//   MULACT       y = sat16(rne_shr(silu_q(a << sh_a) * b, sh_c)),
//                silu_q(gv) = rne_shr(gv * sig, 18), sig the interpolated sigmoid
//                table of |gv| (ro_sig_ram), mirrored for gv < 0, 2^24 for |gv| >= 16
//
// Lanes past the row's n produce 0 (the output rows' tail lanes).  Rows are
// consumed and produced back to back: a row starts a new input word and a
// new output word (the ports' tails handle the ends).
//
// Pipeline (one group per stage; stage s data is registered at the end of
// cycle t + s for a group issued in cycle t):
//   s0 lanes, s1 -, s2 parameters, s3 the shifts / gv, s4 a + b / |gv|,
//   s5 + bq / the table address, s6 the rounding shift (sum) / the float
//   normalisation, s7 the rounding add / the table entry, s8 saturation /
//   the entry's register, s9 relu / float / the interpolation, s10 sigma's
//   mirror, s11 gv * sigma, s12 - s13 the product's registers, s14 silu,
//   s15 silu * u, s16 - s17 the product's registers, s18 - s19 the
//   rounding, s20 saturation -> pack.  COPY / ADDX / DEQUANT results are
//   carried to s20 so every op leaves in order.
// ---------------------------------------------------------------------------
module ro_stream
  import ro_pkg::*;
(
  input  logic          clk,
  input  logic          rst,
  input  logic          start,            // one cycle; the job constants are stable from here on
  input  logic          en,               // a stream job with work (rows and n non-zero)
  input  logic [3:0]    op,
  input  logic [1:0]    ak, bk, ck,       // kinds of A, B, C
  input  logic          b_en,
  input  logic          relu,
  input  logic [5:0]    bq_sh,
  input  logic [31:0]   n,                // elements per row
  input  logic [31:0]   nrows,            // rows in the job

  input  logic          a_valid,
  output logic          a_ready,
  input  logic [BW-1:0] a_data,
  input  logic          b_valid,
  output logic          b_ready,
  input  logic [BW-1:0] b_data,

  output logic [PIW-1:0] prm_idx,
  input  param_t        prm_q [PB],

  // the sigmoid table's load (OP_TLOAD)
  input  logic             tbl_we,
  input  logic [11:0]      tbl_addr,
  input  logic [SIG_W-1:0] tbl_data,

  output logic          c_valid,
  input  logic          c_ready,
  output logic [BW-1:0] c_data,

  output logic          idle
);
  localparam int NS   = 21;               // pipeline stages s0 .. s20
  localparam int CF_D = 32;               // output FIFO depth (groups in flight are counted against it)
  localparam int FW   = $clog2(CF_D);

  // Job constants -------------------------------------------------------------------------
  logic        k_a32, k_b32, k_c16, k_ben, k_relu, k_mul, k_deq;
  logic [5:0]  k_bqsh;
  logic [31:0] k_n;
  always_ff @(posedge clk)
    if (start) begin
      k_a32  <= (ak == K_I32);
      k_b32  <= b_en && (bk == K_I32);
      k_c16  <= (ck == K_I16);
      k_ben  <= b_en;
      k_relu <= relu;
      k_bqsh <= bq_sh;
      k_mul  <= (op == OP_MULACT);
      k_deq  <= (op == OP_DEQUANT);
      k_n    <= n;
    end

  // Input holders: one word each, consumed in halves when the stream is int16.
  logic          ah_v, bh_v, ah_half, bh_half;
  logic [BW-1:0] ah_d, bh_d;
  logic          a_take, b_take, a_done_w, b_done_w;

  logic [31:0]   rows_left;
  logic [31:0]   col;                     // first column of the next group
  logic          running;
  logic          grp_last;                // the next group ends its row
  logic [31:0]   col_next;
  assign col_next = col + 32'd4;
  assign grp_last = (col_next >= k_n);

  logic issue, room;
  assign issue    = running && ah_v && (!k_ben || bh_v) && room;
  // a word is finished by this group when it is int32 (4 lanes = the word), the
  // second half of an int16 word, or the row ends
  assign a_done_w = k_a32 || ah_half || grp_last;
  assign b_done_w = k_b32 || bh_half || grp_last;
  assign a_take   = !ah_v || (issue && a_done_w);     // the holder takes a new word this cycle
  assign b_take   = k_ben && (!bh_v || (issue && b_done_w));
  assign a_ready  = running && a_take;
  assign b_ready  = running && b_take;

  always_ff @(posedge clk) begin
    if (rst || start) begin
      ah_v <= 1'b0; bh_v <= 1'b0; ah_half <= 1'b0; bh_half <= 1'b0;
    end else begin
      if (a_ready && a_valid) begin
        ah_v <= 1'b1; ah_d <= a_data; ah_half <= 1'b0;
      end else if (issue) begin
        if (a_done_w) ah_v <= 1'b0;
        else          ah_half <= 1'b1;
      end
      if (b_ready && b_valid) begin
        bh_v <= 1'b1; bh_d <= b_data; bh_half <= 1'b0;
      end else if (issue && k_ben) begin
        if (b_done_w) bh_v <= 1'b0;
        else          bh_half <= 1'b1;
      end
    end
  end

  always_ff @(posedge clk) begin
    if (rst) begin
      running <= 1'b0;
    end else if (start) begin
      running   <= en;
      rows_left <= nrows;
      col       <= '0;
    end else if (issue) begin
      if (grp_last) begin
        col <= '0;
        rows_left <= rows_left - 32'd1;
        if (rows_left == 32'd1) running <= 1'b0;
      end else begin
        col <= col_next;
      end
    end
  end

  // Lane extraction ----------------------------------------------------------------------
  function automatic logic signed [31:0] lane_of(input logic [BW-1:0] w, input logic is32,
                                                 input logic half, input int l);
    logic [2:0] li;
    li = half ? 3'(l + 4) : 3'(l);
    if (is32) return w[32*l +: 32];
    return 32'(signed'(w[EW*li +: EW]));
  endfunction

  // Stage registers -----------------------------------------------------------------------
  logic [NS-1:0]      v;                  // stage valids: v[s] = a group is in stage s
  logic [NS-1:0]      lastq;              // the group ends its output row
  logic [2:0]         nval [NS];          // valid lanes of the group (1..4)
  logic signed [31:0] x0 [PB], y0 [PB], x1 [PB], y1 [PB], x2 [PB], y2 [PB];
  param_t             p2 [PB];
  logic [2:0]         nv_in;

  always_comb begin
    logic [31:0] remain;
    remain = k_n - col;
    nv_in  = (remain >= 32'd4) ? 3'd4 : remain[2:0];
  end

  assign prm_idx = col[PIW+1:2];          // read with the group's issue: data two cycles later

  always_ff @(posedge clk) begin
    if (rst) v <= '0;
    else     v <= {v[NS-2:0], issue};
    lastq   <= {lastq[NS-2:0], grp_last};
    nval[0] <= nv_in;
    for (int s = 1; s < NS; s++) nval[s] <= nval[s-1];
    for (int l = 0; l < PB; l++) begin
      x0[l] <= lane_of(ah_d, k_a32, ah_half, l);
      y0[l] <= k_ben ? lane_of(bh_d, k_b32, bh_half, l) : 32'sd0;
      x1[l] <= x0[l];
      y1[l] <= y0[l];
      x2[l] <= x1[l];
      y2[l] <= y1[l];
      p2[l] <= prm_q[l];
    end
  end

  // The sigmoid table: two UltraRAM copies, two ports each, one lane per port;
  // the address issued at s5 answers at s7.
  logic [11:0]      sig_idx [PB];
  logic [SIG_W-1:0] sig_q0, sig_q1, sig_q2, sig_q3;
  ro_sig_ram u_sig0 (.clk, .we (tbl_we), .waddr (tbl_addr), .wdata (tbl_data),
                     .addr_a (sig_idx[0]), .addr_b (sig_idx[1]), .q_a (sig_q0), .q_b (sig_q1));
  ro_sig_ram u_sig1 (.clk, .we (tbl_we), .waddr (tbl_addr), .wdata (tbl_data),
                     .addr_a (sig_idx[2]), .addr_b (sig_idx[3]), .q_a (sig_q2), .q_b (sig_q3));

  // Per-lane arithmetic -----------------------------------------------------------------------
  logic [31:0] res [PB];                  // the lane results at s20

  for (genvar l = 0; l < PB; l++) begin : g_lane
    // COPY / ADDX
    logic signed [63:0] xs3, ys3, bq3, t4, bq4, sum5;
    logic [5:0]         shc3, shc4, shc5;
    rne_t               ra6;
    logic signed [63:0] r7;
    logic signed [31:0] c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19;
    // DEQUANT
    logic signed [31:0] fv3;
    logic               fneg4, fneg5, fneg6, fneg7;
    logic [31:0]        fmag4, fmag5, fsh6;
    logic [5:0]         flz5;
    logic [7:0]         fexp6, fexp7;
    logic [22:0]        fman7;
    logic               fzero5, fzero6, fzero7;
    logic [31:0]        f8;
    // MULACT
    logic signed [31:0] gv3, gv4, gv5, gv6, gv7, gv8, gv9, gv10;
    logic [31:0]        ga4;                        // |gv|
    logic               big5, big6, big7, big8, big9, neg4, neg5, neg6, neg7, neg8, neg9;
    logic [11:0]        idx5;
    logic [5:0]         frac5, frac6, frac7, frac8;
    logic [SIG_W-1:0]   q7, q8;                     // {D, T}
    logic [24:0]        si9, sig10;                 // the interpolation, sigma
    logic signed [56:0] ps11, ps12, ps13;           // gv * sigma at 2^-38
    logic signed [38:0] s14;                        // silu at 2^-20
    logic signed [55:0] pu15, pu16, pu17;           // silu * u
    rne_t               ra18;
    logic signed [63:0] r19;
    logic signed [31:0] u3, u4, u5, u6, u7, u8, u9, u10, u11, u12, u13, u14;
    logic [5:0]         shc6, shc7, shc8, shc9, shc10, shc11, shc12, shc13, shc14, shc15, shc16, shc17;

    assign sig_idx[l] = idx5;
    assign q7 = (l == 0) ? sig_q0 : (l == 1) ? sig_q1 : (l == 2) ? sig_q2 : sig_q3;

    always_ff @(posedge clk) begin
      // ---- s3: the shifts
      xs3  <= 64'(x2[l]) <<< p2[l].sh_a;
      ys3  <= 64'(y2[l]) <<< p2[l].sh_b;
      bq3  <= 64'(p2[l].bq) <<< k_bqsh;
      shc3 <= p2[l].sh_c;
      gv3  <= x2[l] <<< p2[l].sh_a[4:0];
      u3   <= y2[l];
      fv3  <= x2[l];
      // ---- s4
      t4    <= xs3 + ys3;
      bq4   <= bq3;
      shc4  <= shc3;
      ga4   <= gv3[31] ? 32'(-gv3) : 32'(gv3);
      neg4  <= gv3[31];
      gv4   <= gv3;
      u4    <= u3;
      fneg4 <= fv3[31];
      fmag4 <= fv3[31] ? 32'(-fv3) : 32'(fv3);
      // ---- s5
      sum5  <= t4 + bq4;
      shc5  <= shc4;
      big5  <= (ga4 >= (32'd1 << 18));
      idx5  <= ga4[17:6];
      frac5 <= ga4[5:0];
      neg5  <= neg4;
      gv5   <= gv4;
      u5    <= u4;
      begin
        logic [5:0] lz;
        lz = 6'd63;
        for (int i = 0; i < 32; i++) if (fmag4[i]) lz = 6'(i);
        flz5   <= lz;
        fzero5 <= (fmag4 == '0);
        fmag5  <= fmag4;
        fneg5  <= fneg4;
      end
      // ---- s6
      ra6   <= rne_a(sum5, shc5);
      shc6  <= shc5;
      big6  <= big5; neg6 <= neg5; frac6 <= frac5; gv6 <= gv5; u6 <= u5;
      fsh6   <= fmag5 << (6'd31 - flz5[4:0]);
      fexp6  <= 8'd127 + 8'(flz5) - 8'(shc5);       // exponent of 2^lz * 2^-sh_c
      fzero6 <= fzero5;
      fneg6  <= fneg5;
      // ---- s7  (the table answers idx5 here: q7)
      r7    <= rne_b(ra6);
      shc7  <= shc6;
      big7  <= big6; neg7 <= neg6; frac7 <= frac6; gv7 <= gv6; u7 <= u6;
      begin
        // mantissa: bits 30..8 of the normalised magnitude, round half to even on bits 7..0
        logic [23:0] m;
        logic        up;
        m  = {1'b0, fsh6[30:8]};
        up = fsh6[7] && ((fsh6[6:0] != '0) || fsh6[8]);
        m  = m + 24'(up);
        fman7  <= m[22:0];
        fexp7  <= fexp6 + 8'(m[23]);                 // the round-up carried into the hidden bit
        fzero7 <= fzero6;
        fneg7  <= fneg6;
      end
      // ---- s8
      c8    <= k_c16 ? 32'(signed'(sat16(r7))) : sat32(r7);
      shc8  <= shc7;
      q8    <= q7;                                    // a fabric register after the UltraRAM
      frac8 <= frac7;
      big8  <= big7;
      neg8  <= neg7;
      gv8   <= gv7;
      u8    <= u7;
      f8    <= fzero7 ? 32'd0 : {fneg7, fexp7, fman7};
      // ---- s9
      c9    <= k_deq ? f8 : (k_relu && c8[31]) ? 32'sd0 : c8;
      si9   <= 25'(((31'(q8[24:0]) << 6) + 31'(q8[48:25]) * 31'(frac8)) >> 6);   // T + floor(D * frac / 64): one DSP multiply-add
      big9  <= big8;
      neg9  <= neg8;
      gv9   <= gv8;
      shc9  <= shc8;
      u9    <= u8;
      // ---- s10
      c10   <= c9;
      begin
        logic [24:0] s;
        s = big9 ? (25'd1 << 24) : si9;                 // |gv| >= 16: sigma = 1
        sig10 <= neg9 ? ((25'd1 << 24) - s) : s;        // mirrored for gv < 0
      end
      gv10  <= gv9;
      shc10 <= shc9;
      u10   <= u9;
      // ---- s11
      c11   <= c10;
      ps11  <= 57'(gv10) * 57'(signed'({1'b0, sig10}));           // 32 x 25 (DSP), registered twice more
      shc11 <= shc10;
      u11   <= u10;
      // ---- s12, s13
      c12 <= c11; ps12 <= ps11; shc12 <= shc11; u12 <= u11;
      c13 <= c12; ps13 <= ps12; shc13 <= shc12; u13 <= u12;
      // ---- s14: silu = rne_shr(ps, 18)
      begin
        logic signed [38:0] q;
        logic               up;
        q  = 39'(ps13 >>> 18);
        up = ps13[17] && ((ps13[16:0] != '0) || q[0]);
        s14 <= q + 39'(up);
      end
      c14 <= c13; shc14 <= shc13; u14 <= u13;
      // ---- s15: silu * u (39 x 17, DSP), registered twice more
      pu15 <= 56'(s14) * 56'(signed'(u14[16:0]));
      c15 <= c14; shc15 <= shc14;
      pu16 <= pu15; c16 <= c15; shc16 <= shc15;
      pu17 <= pu16; c17 <= c16; shc17 <= shc16;
      // ---- s18, s19: the rounding
      ra18 <= rne_a(64'(pu17), shc17);
      c18  <= c17;
      r19  <= rne_b(ra18);
      c19  <= c18;
      // ---- s20
      res[l] <= k_mul ? 32'(signed'(sat16(r19))) : c19;
    end

    logic unused_lane;
    assign unused_lane = ^{flz5[5], fexp6[7], u14[31:17]};
  end

  // Pack --------------------------------------------------------------------------------------
  // s20 results -> words.  int16 out: two groups per word (the first fills lanes
  // 0..3, the second 4..7; a row's last group flushes a half-filled word with the
  // rest 0); int32 / float32: one group = a word.
  logic          wv, half_pend;
  logic [BW-1:0] wd;
  logic [63:0]   wd_lo;
  logic [BW-1:0] packed_full;
  logic [63:0]   packed_half;

  always_comb begin
    packed_full = '0;
    packed_half = '0;
    for (int l = 0; l < PB; l++) begin
      if (l < int'(nval[NS-1])) begin
        packed_full[32*l +: 32] = res[l];
        packed_half[EW*l +: EW] = res[l][15:0];
      end
    end
  end

  always_ff @(posedge clk) begin
    if (rst || start) begin
      wv <= 1'b0;
      half_pend <= 1'b0;
    end else begin
      wv <= 1'b0;
      if (v[NS-1]) begin
        if (k_c16) begin
          if (half_pend) begin                         // second half of the word
            wv <= 1'b1;
            wd <= {packed_half, wd_lo};
            half_pend <= 1'b0;
          end else if (lastq[NS-1]) begin              // a row ending on a half word
            wv <= 1'b1;
            wd <= {64'b0, packed_half};
          end else begin
            wd_lo <= packed_half;
            half_pend <= 1'b1;
          end
        end else begin
          wv <= 1'b1;
          wd <= packed_full;
        end
      end
    end
  end

  // Output FIFO and credits: every issued group reserves one entry until it
  // has left the pipeline.
  logic        cf_in_ready;
  logic [FW:0] cf_count, inflight;
  logic        cf_rst;
  assign cf_rst = rst || start;
  ro_fifo #(.W(BW), .D(CF_D), .BRAM(1'b0)) u_cf (
    .clk, .rst (cf_rst),
    .in_valid (wv), .in_ready (cf_in_ready), .in_data (wd),
    .out_valid (c_valid), .out_ready (c_ready), .out_data (c_data),
    .count (cf_count)
  );
  always_ff @(posedge clk) begin
    if (rst || start) inflight <= '0;
    else inflight <= inflight + (FW+1)'(issue) - (FW+1)'(v[NS-1]);
  end
  assign room = ({1'b0, cf_count} + {1'b0, inflight}) < (FW+2)'(CF_D - 2);

  assign idle = !running && (v == '0) && !wv && !half_pend && (cf_count == '0) && !ah_v && !bh_v;

  logic unused;
  assign unused = ^{cf_in_ready, rows_left};

endmodule
