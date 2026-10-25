// Copyright 2026 Ivan Gubochkin
// SPDX-License-Identifier: Apache-2.0

// ---------------------------------------------------------------------------
// ro_rope — the ROPE op of the RowOpsKernel (ROWOPS_RTL_KERNEL.md §1.4):
// rotate-half RoPE on rows of m head segments of HD int16 elements, bit for
// bit rowops_spec.rope with per-column exponents:
//
//   d = c mod HD;  rot = d < HD/2 ? -x[c + HD/2] : x[c - HD/2]
//   y = sat16(rne_shr(((x * cos[d]) << sh_a[c]) + ((rot * sin[d]) << sh_b[c]), sh_c[c]))
//
// Per row: the B row (cos[0..HD) then sin[0..HD) as int32 words, HD/2 words)
// is loaded into four lane banks (element e in bank e mod 4 at index
// e / 4), then the A row's segments stream through two 16-word buffers:
// segment s fills one while segment s - 1 is emitted from the other, four
// elements (half a word) per cycle through the four lanes.  HD is a
// multiple of 16 (16 .. 128), so an element's rotation partner sits in the
// same lane HD/16 words away.
// ---------------------------------------------------------------------------
module ro_rope
  import ro_pkg::*;
(
  input  logic          clk,
  input  logic          rst,
  input  logic          start,
  input  logic          en,
  input  logic [31:0]   n,                // elements per row (a multiple of HD)
  input  logic [31:0]   nrows,
  input  logic [11:0]   hd,
  input  logic [12:0]   segs,             // n / HD (the core's divider; n <= 4096)

  input  logic          a_valid,
  output logic          a_ready,
  input  logic [BW-1:0] a_data,
  input  logic          b_valid,
  output logic          b_ready,
  input  logic [BW-1:0] b_data,

  output logic [PIW-1:0] prm_idx,
  input  param_t        prm_q [PB],

  // the output words, into the core's output FIFO (shared with TRANSPOSE and FOLD)
  output logic          o_valid,
  output logic [BW-1:0] o_data,
  input  logic [5:0]    o_count,          // the FIFO's entries

  output logic          idle
);
  localparam int SW   = 16;               // words per segment buffer (HD <= 128)
  localparam int NS   = 10;               // pipeline stages s0 .. s9
  localparam int CF_D = 32;               // the output FIFO's depth
  localparam int FW   = $clog2(CF_D);

  // Job constants ---------------------------------------------------------------------------
  logic [31:0] k_n;
  logic [4:0]  k_sw, k_hw;                // words per segment (HD / 8), partner offset (HD / 16)
  logic [6:0]  k_bw;                      // B words per row (HD / 2, up to 64)
  logic [12:0] k_segs;                    // segments per row (n / HD)
  always_ff @(posedge clk)
    if (start) begin
      k_n    <= n;
      k_sw   <= 5'(hd >> 3);
      k_hw   <= 5'(hd >> 4);
      k_bw   <= 7'(hd >> 1);
      k_segs <= segs;
    end

  // Row sequencing --------------------------------------------------------------------------
  typedef enum logic [1:0] {R_IDLE, R_B, R_A} rst_t;
  rst_t        rs;
  logic [31:0] rows_left;
  logic [6:0]  b_left;                    // B words still to take for this row
  logic [12:0] segs_left;                 // A segments still to take for this row
  logic [4:0]  fw;                        // words filled in the current segment
  logic        fpar;                      // the buffer being filled
  logic        full [2];                  // segment buffers ready to emit

  // the next row's cos / sin may only replace the banks once the row's last segments
  // have been emitted (the banks are read at issue)
  assign b_ready = (rs == R_B) && !full[0] && !full[1];
  assign a_ready = (rs == R_A) && !full[fpar];

  // Emitter ------------------------------------------------------------------------------------
  logic        epar;                      // the buffer being emitted
  logic [4:0]  ew;                        // word within the segment
  logic        eh;                        // the word's half (lanes 0..3 / 4..7)
  logic [12:0] seg_out;                   // the segment's index within its row
  logic [PIW-1:0] cw;                     // the group's column / 4 within the row
  logic        issue, room, em_active, seg_last_w, row_last_s;
  assign em_active  = full[epar];
  assign seg_last_w = (ew + 5'd1 == k_sw) && eh;
  assign row_last_s = (seg_out + 13'd1 == k_segs);
  assign issue      = em_active && room;
  assign prm_idx    = cw;

  // cos / sin lane banks: element e of cos in bank e % 4 at index e / 4 (32 deep: HD <= 128),
  // so lane l of word w's half h reads index {w, h} (element 8w + 4h + l).
  // A B word carries elements 4 off .. 4 off + 3 of cos (k < hb) or sin: index off of the four banks.
  logic        b_take, bk_cos;
  logic [4:0]  bk_idx;
  logic [31:0] cs_q [PB], sn_q [PB];
  always_comb begin
    logic [6:0] k, hb, off;
    k      = k_bw - b_left;
    hb     = {1'b0, k_bw[6:1]};
    bk_cos = (k < hb);
    off    = bk_cos ? k : k - hb;
    bk_idx = off[4:0];
  end
  assign b_take = b_valid && b_ready;
  for (genvar b = 0; b < PB; b++) begin : g_bank
    (* ram_style = "distributed" *) logic [31:0] csm [2 * SW], snm [2 * SW];
    always_ff @(posedge clk)
      if (b_take) begin
        if (bk_cos) csm[bk_idx] <= b_data[32*b +: 32];
        else        snm[bk_idx] <= b_data[32*b +: 32];
      end
    assign cs_q[b] = csm[{a0_w, a0_h}];
    assign sn_q[b] = snm[{a0_w, a0_h}];
  end
  // segment buffers: buffer p's word w at sb[{p, w}]; read at ew and at the partner word
  (* ram_style = "distributed" *) logic [BW-1:0] sb [2 * SW];
  logic [BW-1:0] sb_x, sb_p;
  logic [3:0]    pw;
  // the reads of an issued group happen a cycle later, from registered addresses (stage 0)
  logic          a0_par, a0_h, a0_neg;
  logic [3:0]    a0_w, a0_pw;
  always_ff @(posedge clk) if (a_valid && a_ready) sb[{fpar, fw[3:0]}] <= a_data;
  assign pw   = (ew < k_hw) ? 4'(ew + k_hw) : 4'(ew - k_hw);
  always_ff @(posedge clk) begin
    a0_par <= epar; a0_w <= ew[3:0]; a0_pw <= pw; a0_h <= eh; a0_neg <= (ew < k_hw);
  end
  assign sb_x = sb[{a0_par, a0_w}];
  assign sb_p = sb[{a0_par, a0_pw}];

  always_ff @(posedge clk) begin
    if (rst || start) begin
      rs      <= R_IDLE;
      fpar    <= 1'b0;
      epar    <= 1'b0;
      full[0] <= 1'b0;
      full[1] <= 1'b0;
      fw      <= '0;
      ew      <= '0;
      eh      <= 1'b0;
      cw      <= '0;
      seg_out <= '0;
      if (start) begin
        rows_left <= nrows;
        rs        <= (en && nrows != '0) ? R_B : R_IDLE;
        b_left    <= 7'(hd >> 1);
        segs_left <= segs;
      end
    end else begin
      // ---- B row: cos then sin, 4 int32 per word (the banks take the word above)
      if (b_take) begin
        b_left <= b_left - 7'd1;
        if (b_left == 7'd1) begin
          rs <= R_A;
          fw <= '0;
        end
      end
      // ---- A row: segments into the buffers (sb takes the word above)
      if (a_valid && a_ready) begin
        if (fw + 5'd1 == k_sw) begin
          full[fpar] <= 1'b1;
          fw         <= '0;
          fpar       <= !fpar;
          segs_left  <= segs_left - 13'd1;
          if (segs_left == 13'd1) begin
            rows_left <= rows_left - 32'd1;
            segs_left <= k_segs;
            b_left    <= k_bw;
            rs        <= (rows_left == 32'd1) ? R_IDLE : R_B;
          end
        end else begin
          fw <= fw + 5'd1;
        end
      end
      // ---- emit: one half word per cycle
      if (issue) begin
        cw <= (seg_last_w && row_last_s) ? '0 : cw + PIW'(1);
        eh <= !eh;
        if (seg_last_w) begin
          full[epar] <= 1'b0;
          epar       <= !epar;
          ew         <= '0;
          seg_out    <= row_last_s ? 13'd0 : seg_out + 13'd1;
        end else if (eh) begin
          ew <= ew + 5'd1;
        end
      end
    end
  end

  // Pipeline: stage s data registered at the end of cycle t + s (issue at t).
  //   0 the addresses; 1 x / partner / cos / sin read; 2 parameters; 3 products;
  //   4 the products' registers; 5 shifts; 6 sum (rot = -partner: the partner's
  //   product subtracted); 7 - 8 rounding; 9 saturation -> the group
  logic [NS-1:0] v;
  logic          h0;                        // the group's half (the output word's lanes)
  logic          neg1, neg2;                // rot = -partner (d < HD/2)
  logic signed [15:0] x1 [PB];
  logic signed [15:0] r1 [PB];              // the partner
  logic signed [31:0] c1 [PB], s1 [PB];
  logic signed [15:0] x2 [PB];
  logic signed [15:0] r2 [PB];
  logic signed [31:0] c2 [PB], s2 [PB];
  param_t p2 [PB];
  logic [15:0] y9 [PB];

  always_ff @(posedge clk) begin
    if (rst || start) v <= '0;
    else v <= {v[NS-2:0], issue};
    h0 <= eh;
    neg1 <= a0_neg;
    neg2 <= neg1;
    for (int l = 0; l < PB; l++) begin
      x1[l] <= sb_x[EW*(l + (a0_h ? 4 : 0)) +: EW];
      r1[l] <= sb_p[EW*(l + (a0_h ? 4 : 0)) +: EW];
      c1[l] <= cs_q[l];
      s1[l] <= sn_q[l];
      x2[l] <= x1[l]; r2[l] <= r1[l]; c2[l] <= c1[l]; s2[l] <= s1[l];
      p2[l] <= prm_q[l];
    end
  end

  for (genvar l = 0; l < PB; l++) begin : g_lane
    logic signed [47:0] px3, pr3, px4, pr4;
    logic signed [63:0] xs5, rs5, sum6;
    rne_t               ra7;
    logic signed [63:0] r8;
    param_t p3, p4, p5;
    logic       neg3, neg4, neg5;             // the lane's copy of the partner's sign
    logic [5:0] shc6;
    always_ff @(posedge clk) begin
      px3  <= 48'(x2[l]) * 48'(c2[l]);                // 16 x 32 (DSP), registered once more
      pr3  <= 48'(r2[l]) * 48'(s2[l]);
      p3   <= p2[l];
      neg3 <= neg2;
      px4  <= px3;
      pr4  <= pr3;
      p4   <= p3;
      neg4 <= neg3;
      xs5  <= 64'(px4) <<< p4.sh_a;
      rs5  <= 64'(pr4) <<< p4.sh_b;
      p5   <= p4;
      neg5 <= neg4;
      // ((-p) s) << k = -((p s) << k) mod 2^64
      sum6 <= neg5 ? xs5 - rs5 : xs5 + rs5;
      shc6 <= p5.sh_c;
      ra7  <= rne_a(sum6, shc6);
      r8   <= rne_b(ra7);
      y9[l] <= sat16(r8);
    end
  end

  // Output words: two groups per word (lanes 0..3 then 4..7; HD is a multiple of
  // 16, so every word is whole) and credits ------------------------------------------------
  logic          wv;
  logic [BW-1:0] wd;
  logic [NS-1:0] hq;                        // the half along the pipeline
  always_ff @(posedge clk) begin
    hq <= {hq[NS-2:0], h0};
    if (rst || start) wv <= 1'b0;
    else wv <= v[NS-1] && hq[NS-2];         // y9 is stage 9 data (v[9]); its half is hq[8]
    if (v[NS-1]) begin
      if (hq[NS-2]) for (int l = 0; l < PB; l++) wd[EW*(l + 4) +: EW] <= y9[l];
      else          for (int l = 0; l < PB; l++) wd[EW*l +: EW]       <= y9[l];
    end
  end

  assign o_valid = wv;
  assign o_data  = wd;
  logic [FW:0] cf_count, inflight;
  assign cf_count = o_count;
  // every issued group (a half word) holds a credit until its word leaves the pipeline
  always_ff @(posedge clk) begin
    if (rst || start) inflight <= '0;
    else inflight <= inflight + (FW+1)'(issue) - (FW+1)'(v[NS-1]);
  end
  assign room = ({1'b0, cf_count} + {1'b0, inflight}) < (FW+2)'(CF_D - 2);

  assign idle = (rs == R_IDLE) && !full[0] && !full[1] && (v == '0) && !wv && (cf_count == '0);

  logic unused;
  assign unused = ^{k_n, hq[NS-1]};

endmodule
