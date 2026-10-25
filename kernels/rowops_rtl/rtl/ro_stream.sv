// Copyright 2026 Ivan Gubochkin
// SPDX-License-Identifier: Apache-2.0

// ---------------------------------------------------------------------------
// ro_stream — the element-wise ops of the RowOpsKernel (ROWOPS_RTL_KERNEL.md
// §1.4): COPY, ADDX, DEQUANT, MULACT, GATE.  Stream A (and B) words in, C words
// out, one group of four lanes per cycle (an int16 word in two halves, an
// int32 / float32 word whole), with every column's parameters from ro_param.
// Per lane (64-bit integer arithmetic, rne = round half to even):
//
//   COPY / ADDX  y = sat_C(rne_shr(S, sh_c)), relu; S = (a << sh_a) [+ (b << sh_b)] + (bq << bq_sh)
//                with SCALE: y = sat_C(rne_shr(S * G, sh_c)), relu (|S| < 2^31, 0 <= G < 2^25)
//   DEQUANT      y = float32(a * 2^-sh_c)
//   MULACT       y = sat16(rne_shr(silu_q(a << sh_a) * b, sh_c)),
//                silu_q(gv) = rne_shr(gv * sig, 18), sig the interpolated sigmoid
//                table of |gv| (ro_sig_ram), mirrored for gv < 0, 2^24 for |gv| >= 16
//   GATE         y = sat16(rne_shr(th * sg, sh_c)), th = 2 sig(a << sh_a) - 2^24 (tanh),
//                sg = sig(b << sh_b) — two table reads per lane (four UltraRAM copies);
//                MULACT's sig takes the B side's read and mirror
//
// SCALE and GATE share MULACT's gv * sigma multiplier (its operands chosen at s10)
// and its rounding (s18 - s19); their product bypasses the silu stages (s14 - s16).
// COPY / ADDX's sum is carried to s17 and takes the same rounding and saturation:
// one rounding unit per lane (one op per job).
//
// Lanes past the row's n produce 0 (the output rows' tail lanes).  Rows are
// consumed and produced back to back: a row starts a new input word and a
// new output word (the ports' tails handle the ends).
//
// Pipeline (one group per stage; stage s data is registered at the end of
// cycle t + s for a group issued in cycle t):
//   s0 lanes, s1 -, s2 parameters, s3 the shifts, s4 a + b, bq's shift / gv, |gv|,
//   s5 + bq / the table address, s6 the float normalisation, s7 the table
//   entry / the float's rounding, s8 the entry's register / float, s9 the
//   interpolation, s10 sigma's mirror, s11 gv * sigma, s12 - s13 the
//   product's registers, s14 silu, s15 silu * u, s16 the product's
//   register, s17 the product or COPY / ADDX's sum (carried from s5),
//   s18 - s19 the rounding, s20 saturation, relu -> pack.  DEQUANT's float
//   is carried to s20 so every op leaves in order.
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
  input  logic          scale,            // COPY / ADDX: the sum times G (op bit 17)
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
  logic        k_a32, k_b32, k_c16, k_s16, k_ben, k_relu, k_mul, k_deq, k_gate, k_scale, k_alt;
  logic [5:0]  k_bqsh;
  logic [31:0] k_n;
  always_ff @(posedge clk)
    if (start) begin
      k_a32   <= (ak == K_I32);
      k_b32   <= b_en && (bk == K_I32);
      k_c16   <= (ck == K_I16);
      k_s16   <= (ck == K_I16) || (op == OP_MULACT) || (op == OP_GATE);   // sat16 at s20
      k_ben   <= b_en;
      k_relu  <= relu && (op != OP_MULACT) && (op != OP_GATE);          // COPY / ADDX
      k_bqsh  <= bq_sh;
      k_mul   <= (op == OP_MULACT);                                    // s17: silu * u
      k_deq   <= (op == OP_DEQUANT);
      k_gate  <= (op == OP_GATE);
      k_scale <= scale && ((op == OP_COPY) || (op == OP_ADDX));
      k_alt   <= (op == OP_GATE) || (scale && ((op == OP_COPY) || (op == OP_ADDX)));   // the bypass
      k_n     <= n;
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
      if (k_deq) p2[l].sh_a <= '0;        // DEQUANT: x itself through gv's path (the float's magnitude)
    end
  end

  // The sigmoid table: four UltraRAM copies, two ports each, one lane per port —
  // u_sig0 / u_sig1 answer the A side (GATE's 2a), u_sig2 / u_sig3 the B side
  // (GATE's b, MULACT's gv); the address issued at s5 answers at s7.
  logic [11:0]      sig_idx [PB], sigb_idx [PB];
  logic [SIG_W-1:0] sig_q0, sig_q1, sig_q2, sig_q3, sigb_q0, sigb_q1, sigb_q2, sigb_q3;
  ro_sig_ram u_sig0 (.clk, .we (tbl_we), .waddr (tbl_addr), .wdata (tbl_data),
                     .addr_a (sig_idx[0]), .addr_b (sig_idx[1]), .q_a (sig_q0), .q_b (sig_q1));
  ro_sig_ram u_sig1 (.clk, .we (tbl_we), .waddr (tbl_addr), .wdata (tbl_data),
                     .addr_a (sig_idx[2]), .addr_b (sig_idx[3]), .q_a (sig_q2), .q_b (sig_q3));
  ro_sig_ram u_sig2 (.clk, .we (tbl_we), .waddr (tbl_addr), .wdata (tbl_data),
                     .addr_a (sigb_idx[0]), .addr_b (sigb_idx[1]), .q_a (sigb_q0), .q_b (sigb_q1));
  ro_sig_ram u_sig3 (.clk, .we (tbl_we), .waddr (tbl_addr), .wdata (tbl_data),
                     .addr_a (sigb_idx[2]), .addr_b (sigb_idx[3]), .q_a (sigb_q2), .q_b (sigb_q3));

  // Per-lane arithmetic -----------------------------------------------------------------------
  logic [31:0] res [PB];                  // the lane results at s20

  for (genvar l = 0; l < PB; l++) begin : g_lane
    // COPY / ADDX
    logic signed [63:0] xs3, ys3, t4, bq4, sum5;
    logic signed [21:0] bq3;                        // Bq' (shifted into bq4)
    logic signed [63:0] sd [11];                    // sum5 carried to s16 (sd[i]: stage 6 + i)
    logic [5:0]         shc3, shc4, shc5;
    logic signed [31:0] c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19;   // DEQUANT's float
    // DEQUANT (|x| and its sign from gv's path: sh_a is 0)
    logic               fneg5, fneg6, fneg7;
    logic [31:0]        fmag5, fsh6;
    logic [5:0]         flz5;
    logic [7:0]         fexp6, fexp7;
    logic [22:0]        fman7;
    logic               fzero5, fzero6, fzero7;
    logic [31:0]        f8;
    // MULACT
    logic               sah3, sbh3;                 // sh_a[5], sh_b[5]
    logic signed [31:0] gv4, gv5, gv6, gv7, gv8, gv9, gv10;
    logic [31:0]        ga4;                        // |gv|
    logic               big5, big6, big7, big8, big9, neg4, neg5, neg6, neg7, neg8, neg9;
    logic [11:0]        idx5;
    logic [5:0]         frac5, frac6, frac7, frac8;
    logic [SIG_W-1:0]   q7, q8;                     // {D, T}
    logic [24:0]        si9, sig10;                 // the interpolation, sigma
    logic signed [56:0] ps11, ps12, ps13;           // gv * sigma at 2^-38
    logic signed [38:0] s14;                        // silu at 2^-20
    logic signed [55:0] pu15, pu16;                 // silu * u
    logic signed [56:0] pg14, pg15, pg16;           // GATE / SCALE: the product past the silu stages
    logic signed [63:0] pm17;                       // the value to round: a product or the sum
    rne_t               ra18;
    logic signed [63:0] r19;
    logic signed [31:0] u3, u4, u5, u6, u7, u8, u9, u10, u11, u12, u13, u14;
    logic [5:0]         shc6, shc7, shc8, shc9, shc10, shc11, shc12, shc13, shc14, shc15, shc16, shc17;
    // the sigma side: GATE's b, MULACT's gv
    logic [31:0]        gab4;
    logic               bigb5, bigb6, bigb7, bigb8, bigb9, negb4, negb5, negb6, negb7, negb8, negb9;
    logic [11:0]        idxb5;
    logic [5:0]         fracb5, fracb6, fracb7, fracb8;
    logic [SIG_W-1:0]   qb7, qb8;
    logic [24:0]        sib9;
    // SCALE: the sum S and G carried to the multiplier
    logic signed [31:0] sc6, sc7, sc8, sc9;
    logic [24:0]        g3, g4, g5, g6, g7, g8, g9;

    assign sig_idx[l]  = idx5;
    assign sigb_idx[l] = idxb5;
    assign q7  = (l == 0) ? sig_q0 : (l == 1) ? sig_q1 : (l == 2) ? sig_q2 : sig_q3;
    assign qb7 = (l == 0) ? sigb_q0 : (l == 1) ? sigb_q1 : (l == 2) ? sigb_q2 : sigb_q3;

    always_ff @(posedge clk) begin
      // ---- s3: the shifts
      xs3  <= 64'(x2[l]) <<< p2[l].sh_a;
      ys3  <= 64'(y2[l]) <<< p2[l].sh_b;
      bq3  <= p2[l].bq;
      shc3 <= p2[l].sh_c;
      sah3 <= p2[l].sh_a[5];
      sbh3 <= p2[l].sh_b[5];
      g3   <= p2[l].g[24:0];
      u3   <= y2[l];
      // ---- s4
      t4    <= xs3 + ys3;
      bq4   <= 64'(bq3) <<< k_bqsh;
      shc4  <= shc3;
      begin
        // gv = x << sh_a[4:0] and y << sh_b[4:0] in 32 bits: the word of the 64-bit shifts
        // that holds them (the upper one for a shift >= 32)
        logic signed [31:0] gv, gvb;
        gv    = sah3 ? xs3[63:32] : xs3[31:0];
        gvb   = sbh3 ? ys3[63:32] : ys3[31:0];
        ga4   <= gv[31] ? 32'(-gv) : 32'(gv);
        neg4  <= gv[31];
        gab4  <= gvb[31] ? 32'(-gvb) : 32'(gvb);
        negb4 <= gvb[31];
        gv4   <= gv;
      end
      g4    <= g3;
      u4    <= u3;
      // ---- s5
      sum5  <= t4 + bq4;
      shc5  <= shc4;
      big5  <= (ga4 >= (32'd1 << 18));
      idx5  <= ga4[17:6];
      frac5 <= ga4[5:0];
      neg5  <= neg4;
      bigb5  <= k_mul ? (ga4 >= (32'd1 << 18)) : (gab4 >= (32'd1 << 18));
      idxb5  <= k_mul ? ga4[17:6] : gab4[17:6];
      fracb5 <= k_mul ? ga4[5:0] : gab4[5:0];
      negb5  <= k_mul ? neg4 : negb4;
      g5     <= g4;
      gv5   <= gv4;
      u5    <= u4;
      begin
        logic [5:0] lz;
        lz = 6'd63;
        for (int i = 0; i < 32; i++) if (ga4[i]) lz = 6'(i);
        flz5   <= lz;
        fzero5 <= (ga4 == '0);
        fmag5  <= ga4;
        fneg5  <= neg4;
      end
      // ---- s6 .. s16: the sum carried to s17
      sd[0] <= sum5;
      for (int i = 1; i < 11; i++) sd[i] <= sd[i-1];
      // ---- s6
      shc6  <= shc5;
      big6  <= big5; neg6 <= neg5; frac6 <= frac5; gv6 <= gv5; u6 <= u5;
      bigb6 <= bigb5; negb6 <= negb5; fracb6 <= fracb5;
      sc6   <= 32'(sum5);                              // SCALE: |S| < 2^31 (the contract)
      g6    <= g5;
      fsh6   <= fmag5 << (6'd31 - flz5[4:0]);
      fexp6  <= 8'd127 + 8'(flz5) - 8'(shc5);       // exponent of 2^lz * 2^-sh_c
      fzero6 <= fzero5;
      fneg6  <= fneg5;
      // ---- s7  (the table answers idx5 here: q7)
      shc7  <= shc6;
      big7  <= big6; neg7 <= neg6; frac7 <= frac6; gv7 <= gv6; u7 <= u6;
      bigb7 <= bigb6; negb7 <= negb6; fracb7 <= fracb6;
      sc7   <= sc6;
      g7    <= g6;
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
      shc8  <= shc7;
      q8    <= q7;                                    // a fabric register after the UltraRAM
      qb8   <= qb7;
      frac8 <= frac7;
      fracb8 <= fracb7; bigb8 <= bigb7; negb8 <= negb7;
      sc8   <= sc7;
      g8    <= g7;
      big8  <= big7;
      neg8  <= neg7;
      gv8   <= gv7;
      u8    <= u7;
      f8    <= fzero7 ? 32'd0 : {fneg7, fexp7, fman7};
      // ---- s9
      c9    <= f8;
      si9   <= 25'(((31'(q8[24:0]) << 6) + 31'(q8[48:25]) * 31'(frac8)) >> 6);   // T + floor(D * frac / 64): one DSP multiply-add
      sib9  <= 25'(((31'(qb8[24:0]) << 6) + 31'(qb8[48:25]) * 31'(fracb8)) >> 6);
      big9  <= big8;
      neg9  <= neg8;
      bigb9 <= bigb8;
      negb9 <= negb8;
      sc9   <= sc8;
      g9    <= g8;
      gv9   <= gv8;
      shc9  <= shc8;
      u9    <= u8;
      // ---- s10
      c10   <= c9;
      // the multiplier's operands: MULACT gv * sigma(gv); GATE th * sg; SCALE S * G
      begin
        logic [24:0]        sbb, sb;
        logic signed [25:0] u, th;
        // tanh = 2 sigma(2a) - 1: si9 lies in [2^23, 2^24), so 2 si9 - 2^24 is its low 23 bits doubled
        u   = big9 ? (26'sd1 <<< 24) : 26'({1'b0, si9[22:0], 1'b0});
        th  = neg9 ? -u : u;
        sbb = bigb9 ? (25'd1 << 24) : sib9;             // |arg| >= 16: sigma = 1
        sb  = negb9 ? ((25'd1 << 24) - sbb) : sbb;      // mirrored for arg < 0
        sig10 <= k_scale ? g9 : sb;
        gv10  <= k_scale ? sc9 : k_gate ? 32'(th) : gv9;
      end
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
      pg14 <= ps13;
      // ---- s15: silu * u (39 x 17, DSP), registered twice more
      pu15 <= 56'(s14) * 56'(signed'(u14[16:0]));
      c15 <= c14; shc15 <= shc14; pg15 <= pg14;
      pu16 <= pu15; c16 <= c15; shc16 <= shc15; pg16 <= pg15;
      // GATE / SCALE: the product of s11; MULACT: silu * u; COPY / ADDX: the sum
      pm17 <= k_alt ? 64'(pg16) : k_mul ? 64'(pu16) : sd[10];
      c17 <= c16; shc17 <= shc16;
      // ---- s18, s19: the rounding
      ra18 <= rne_a(pm17, shc17);
      c18  <= c17;
      r19  <= rne_b(ra18);
      c19  <= c18;
      // ---- s20: sat_C, relu (saturation keeps the sign: relu of r19's sign)
      if (k_deq) res[l] <= c19;
      else       res[l] <= (k_relu && r19[63]) ? 32'd0 : k_s16 ? 32'(signed'(sat16(r19))) : sat32(r19);
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
