// Copyright 2026 Ivan Gubochkin
// SPDX-License-Identifier: Apache-2.0

// ---------------------------------------------------------------------------
// ro_norm — the NORM op of the RowOpsKernel (ROWOPS_RTL_KERNEL.md §1.4, §2):
// RMSNorm / LayerNorm rows of int16 or int32 elements -> int16, bit for bit
// rowops_spec.rms_norm / layer_norm:
//
//   S1 = sum x, S2 = sum x^2 (exact);  D = S2 + E  (rms)  |  n*S2 - S1^2 + E  (ln)
//   (Rm, sh) = rsqrt_bf(D):  Dm = D block-floated to [2^30, 2^32) with an even
//     exponent d, y0 = ROM[Dm[31:19]], t = Dm*y0 >> 32, u = t*y0 >> 32,
//     Rm = clamp(y0 + (y0*(2^30 - u) >>> 31)), sh = 47 + d/2
//   A = x (rms) | n*x - S1 (ln);  P1 = floor(A * Rm / 2^(sh - 24))
//   y = sat16(rne_shr(P1 * G[c] + (Bq'[c] << bq_sh), sh_c[c]))
//
// Three stages work on different rows at once:
//   LOADER   A words into the ring (2 UltraRAMs: 4096 x 128) while 4 lanes
//            accumulate S1 and S2 (an int16 word in two halves); a finished
//            row's descriptor goes to rq
//   SCALAR   one row at a time (a sequential FSM): V, D, the block float, the
//            table, one Newton step -> eq
//   EMITTER  the row's words back from the ring, 4 elements per cycle through
//            the per-lane pipeline (n*x - S1, * Rm, the shift, * G, + Bq', the
//            rounding), packed 8 per word into the output FIFO
// Rows queue in rq / eq (8 deep each) and in the ring (credits of words), so
// the loader runs ahead and the scalar latency hides behind the emitter.
// ---------------------------------------------------------------------------
module ro_norm
  import ro_pkg::*;
(
  input  logic          clk,
  input  logic          rst,
  input  logic          start,            // one cycle; the job constants are stable from here on
  input  logic          en,
  input  logic          ln,               // LayerNorm (0: RMSNorm)
  input  logic          a32,              // A elements are int32 (0: int16)
  input  logic [31:0]   n,                // elements per row
  input  logic [31:0]   nrows,
  input  logic [31:0]   words,            // A words per row
  input  logic [63:0]   eps,              // E
  input  logic [5:0]    bq_sh,

  input  logic          a_valid,
  output logic          a_ready,
  input  logic [BW-1:0] a_data,

  output logic [PIW-1:0] prm_idx,
  input  param_t        prm_q [PB],

  // the rsqrt table's load (OP_TLOAD)
  input  logic          tbl_we,
  input  logic [11:0]   tbl_addr,
  input  logic [63:0]   tbl_data,

  output logic          c_valid,
  input  logic          c_ready,
  output logic [BW-1:0] c_data,

  output logic          idle
);
  localparam int RING_D = 4096;
  localparam int RW     = 12;             // ring address
  localparam int CF_D   = 32;
  localparam int FW     = $clog2(CF_D);
  localparam int NS     = 17;             // emitter pipeline stages

  // Job constants ---------------------------------------------------------------------------
  logic        k_ln, k_a32;
  logic [31:0] k_n, k_words;
  logic [63:0] k_eps;
  logic [5:0]  k_bqsh;
  logic [12:0] k_n13;
  always_ff @(posedge clk)
    if (start) begin
      k_ln <= ln; k_a32 <= a32; k_n <= n; k_words <= words; k_eps <= eps; k_bqsh <= bq_sh;
      k_n13 <= n[12:0];
    end

  // Ring --------------------------------------------------------------------------------------
  (* ram_style = "ultra" *) logic [BW-1:0] ring [RING_D];
  logic [RW-1:0] wptr, rptr;
  logic [RW:0]   used;                     // words written, not yet read back
  logic          ring_we;
  logic [BW-1:0] ring_q0, ring_q1;
  always_ff @(posedge clk) begin
    if (ring_we) ring[wptr] <= a_data;
    ring_q0 <= ring[rptr];
    ring_q1 <= ring_q0;
  end

  // ===================================== LOADER =============================================
  // A words into the ring, four elements per cycle into the sums (an int16 word in
  // two halves, an int32 word whole); the word is consumed with its last half.
  logic          ld_run, ld_ok, ld_step, ld_half;
  logic [31:0]   ld_rows_left, ld_wleft;  // rows left; words left in the current row
  logic          rq_in_ready;
  logic          ld_take, ld_last;         // a word accepted / it ends its row
  logic [3:0]    rq_count, rq_pend;        // descriptors queued / still in the sums pipeline
  // a row's descriptor is pushed five cycles after its last group: keep room for the
  // rows already in the pipeline plus this one
  assign ld_ok   = ld_run && (used < (RW+1)'(RING_D - 8)) && ((rq_count + rq_pend) < 4'd8);
  assign ld_step = a_valid && ld_ok;
  assign a_ready = ld_ok && (k_a32 || ld_half);
  assign ld_take = a_valid && a_ready;
  assign ld_last = (ld_wleft == 32'd1);
  assign ring_we = ld_take;

  logic [RW-1:0] ld_base;                  // ring address of the current row's first word
  always_ff @(posedge clk) begin
    if (rst) begin
      ld_run  <= 1'b0;
      ld_half <= 1'b0;
      wptr    <= '0;
    end else if (start) begin
      ld_run       <= en;
      ld_half      <= 1'b0;
      ld_rows_left <= nrows;
      ld_wleft     <= words;
      ld_base      <= wptr;
    end else begin
      if (ld_step && !k_a32) ld_half <= !ld_half;
      if (ld_take) begin
        wptr <= wptr + RW'(1);
        if (ld_last) begin
          ld_wleft     <= k_words;
          ld_base      <= wptr + RW'(1);
          ld_rows_left <= ld_rows_left - 32'd1;
          if (ld_rows_left == 32'd1) ld_run <= 1'b0;
        end else begin
          ld_wleft <= ld_wleft - 32'd1;
        end
      end
    end
  end

  // sums: s0 the group, s1 lanes, s2 squares (DSP), s3 the squares' register,
  // s4 lane sums, s5 accumulate
  logic          sv1, sv2, sv3, sv4, sv5, sl1, sl2, sl3, sl4, sl5;   // valid / last-of-row
  logic [RW-1:0] sb1, sb2, sb3, sb4, sb5;                            // row base
  logic signed [31:0] sx1 [PB];
  logic signed [63:0] sq2 [PB], sq3 [PB];
  logic signed [31:0] sx2 [PB], sx3 [PB];
  logic signed [33:0] sum1_4;
  logic signed [65:0] sum2_4;
  logic signed [47:0] acc1;                 // S1 (int32 rows of 4096: < 2^44)
  logic signed [79:0] acc2;                 // S2 (< 2^75)
  logic signed [47:0] row_s1;
  logic signed [79:0] row_s2;
  logic          rq_push;
  logic [RW-1:0] rq_base;

  function automatic logic signed [31:0] lane_ld(input logic [BW-1:0] w, input logic is32,
                                                 input logic half, input int l);
    logic [2:0] li;
    li = half ? 3'(l + 4) : 3'(l);
    if (is32) return w[32*l +: 32];
    return 32'(signed'(w[EW*li +: EW]));
  endfunction

  always_ff @(posedge clk) begin
    sv1 <= ld_step; sl1 <= ld_last && (k_a32 || ld_half); sb1 <= ld_base;
    for (int l = 0; l < PB; l++) sx1[l] <= lane_ld(a_data, k_a32, ld_half, l);
    sv2 <= sv1; sl2 <= sl1; sb2 <= sb1;
    for (int l = 0; l < PB; l++) begin
      sq2[l] <= 64'(sx1[l]) * 64'(sx1[l]);
      sx2[l] <= sx1[l];
    end
    sv3 <= sv2; sl3 <= sl2; sb3 <= sb2;
    for (int l = 0; l < PB; l++) begin
      sq3[l] <= sq2[l];
      sx3[l] <= sx2[l];
    end
    sv4 <= sv3; sl4 <= sl3; sb4 <= sb3;
    begin
      logic signed [33:0] a1;
      logic signed [65:0] a2;
      a1 = '0; a2 = '0;
      for (int l = 0; l < PB; l++) begin a1 = a1 + 34'(sx3[l]); a2 = a2 + 66'(sq3[l]); end
      sum1_4 <= a1;
      sum2_4 <= a2;
    end
    sv5 <= sv4; sl5 <= sl4; sb5 <= sb4;
    if (rst || start) begin
      acc1 <= '0; acc2 <= '0;
    end else if (sv4) begin
      if (sl4) begin
        acc1 <= '0; acc2 <= '0;
      end else begin
        acc1 <= acc1 + 48'(sum1_4);
        acc2 <= acc2 + 80'(sum2_4);
      end
      row_s1 <= acc1 + 48'(sum1_4);
      row_s2 <= acc2 + 80'(sum2_4);
    end
    if (rst) begin sv1 <= 1'b0; sv2 <= 1'b0; sv3 <= 1'b0; sv4 <= 1'b0; sv5 <= 1'b0; end
  end
  assign rq_push = sv5 && sl5;
  assign rq_base = sb5;

  // the finished rows: {base, S1, S2}
  localparam int RQW = RW + 48 + 80;
  logic          rq_valid, rq_pop;
  logic [RQW-1:0] rq_q;
  assign rq_pend = 4'(sv1 && sl1) + 4'(sv2 && sl2) + 4'(sv3 && sl3) + 4'(sv4 && sl4) + 4'(sv5 && sl5);
  ro_fifo #(.W(RQW), .D(8), .BRAM(1'b0)) u_rq (
    .clk, .rst (rst || start),
    .in_valid (rq_push), .in_ready (rq_in_ready), .in_data ({rq_base, row_s1, row_s2}),
    .out_valid (rq_valid), .out_ready (rq_pop), .out_data (rq_q),
    .count (rq_count)
  );

  // ===================================== SCALAR =============================================
  // One row at a time: V, D, the block float, the table (two cycles), one Newton
  // step.  The 32 x 32 products go through ro_mul32 (three cycles each): the
  // square S1^2 = (H 2^32 + L)^2 as L*L, H*L (both ro_mul32) and H*H.
  typedef enum logic [4:0] {C_IDLE, C_LOAD, C_MULN, C_SQ, C_SQ2, C_V, C_D, C_D2, C_LZ, C_DM,
                            C_ROM, C_ROM2, C_T, C_T1, C_T2, C_T3, C_U, C_U1, C_U2, C_U3,
                            C_Y, C_Y1, C_Y2, C_Y3, C_Y4, C_Y5, C_Y6, C_PUSH} cst_t;
  cst_t          cs;
  logic [RW-1:0] c_base;
  logic signed [47:0] c_s1;
  logic signed [79:0] c_s2;
  logic [95:0]   c_v, c_d, c_dsum, c_ns2;  // n*S2 / V / D (unsigned; D >= 1)
  logic [95:0]   c_sq, c_sqt;              // S1^2
  logic [47:0]   c_sa;                      // |S1|: H = [47:32], L = [31:0]
  logic [31:0]   c_hh;                      // H * H
  logic [63:0]   c_sql, c_sqhl;             // L * L, H * L
  logic [3:0]    c_bit;                     // n's bit being added (n*S2 = sum n[i] * S2 << i)
  logic [6:0]    c_e;                       // bit length of D (1 .. 96)
  logic signed [7:0] c_dd;                  // d (even, -32 .. 64)
  logic [31:0]   c_dm, c_y0, c_t, c_u, c_rm;
  logic          c_eneg;                    // 2^30 - u < 0
  logic signed [65:0] c_pr;                 // y0 * (2^30 - u)
  logic signed [33:0] c_y1;
  logic [31:0]   mul_a, mul_b;
  logic [63:0]   mul_p;
  logic [12:0]   rom_addr;
  logic [31:0]   rom_q;
  logic          eq_in_ready, eq_push;
  logic [5:0]    c_k;                       // sh - 24

  ro_mul32 u_mul (.clk, .a (mul_a), .b (mul_b), .p (mul_p));
  ro_rsqrt_ram u_rom (.clk, .we (tbl_we), .waddr (tbl_addr), .wdata (tbl_data), .addr (rom_addr), .q (rom_q));
  assign rom_addr = c_dm[31:19];
  assign rq_pop   = (cs == C_IDLE) && rq_valid && eq_in_ready;
  assign eq_push  = (cs == C_PUSH);

  function automatic logic [6:0] bitlen96(input logic [95:0] v);
    logic [6:0] r;
    r = '0;
    for (int i = 0; i < 96; i++) if (v[i]) r = 7'(i + 1);
    return r;
  endfunction

  always_ff @(posedge clk) begin
    if (rst || start) begin
      cs <= C_IDLE;
    end else begin
      case (cs)
        C_IDLE: if (rq_pop) begin
          {c_base, c_s1, c_s2} <= rq_q;
          cs <= C_LOAD;
        end
        C_LOAD: begin
          c_ns2 <= '0;
          c_bit <= '0;
          c_sa  <= c_s1[47] ? 48'(-c_s1) : 48'(c_s1);
          cs <= k_ln ? C_MULN : C_V;
        end
        C_MULN: begin                               // n * S2 over the 13 bits of n; the square beside it
          if (k_n13[c_bit]) c_ns2 <= c_ns2 + (96'(c_s2) << c_bit);
          c_bit <= c_bit + 4'd1;
          case (c_bit)
            4'd0: begin mul_a <= c_sa[31:0]; mul_b <= c_sa[31:0]; c_hh <= c_sa[47:32] * c_sa[47:32]; end
            4'd4: begin c_sql <= mul_p; mul_a <= {16'b0, c_sa[47:32]}; mul_b <= c_sa[31:0]; end
            4'd8: c_sqhl <= mul_p;
            default: ;
          endcase
          if (c_bit == 4'd12) cs <= C_SQ;
        end
        C_SQ:  begin c_sqt <= (96'(c_sqhl) << 33) + 96'(c_sql); cs <= C_SQ2; end
        C_SQ2: begin c_sq <= c_sqt + (96'(c_hh) << 64); cs <= C_V; end
        C_V: begin
          c_v <= k_ln ? (c_ns2 - c_sq) : 96'(c_s2);
          cs  <= C_D;
        end
        C_D:  begin c_dsum <= c_v + 96'(k_eps); cs <= C_D2; end
        C_D2: begin
          c_d <= (c_dsum == '0 || c_v[95]) ? 96'd1 : c_dsum;   // V < 0 cannot happen for exact sums; guard
          cs  <= C_LZ;
        end
        C_LZ: begin
          c_e <= bitlen96(c_d);
          cs  <= C_DM;
        end
        C_DM: begin
          begin
            logic signed [7:0] d;
            d = 8'(signed'({1'b0, c_e})) - 8'sd32;
            if (d[0]) d = d + 8'sd1;
            c_dd <= d;
            c_dm <= (d >= 8'sd0) ? 32'(c_d >> d[6:0]) : 32'(c_d << (-d));
          end
          cs <= C_ROM;
        end
        C_ROM:  cs <= C_ROM2;                       // the table answers two cycles after the address
        C_ROM2: cs <= C_T;
        C_T: begin
          c_y0  <= rom_q;
          mul_a <= c_dm;
          mul_b <= rom_q;                           // Dm * y0
          cs    <= C_T1;
        end
        C_T1: cs <= C_T2;
        C_T2: cs <= C_T3;
        C_T3: cs <= C_U;
        C_U: begin
          c_t   <= mul_p[63:32];
          mul_a <= mul_p[63:32];
          mul_b <= c_y0;                            // t * y0
          cs    <= C_U1;
        end
        C_U1: cs <= C_U2;
        C_U2: cs <= C_U3;
        C_U3: cs <= C_Y;
        C_Y: begin
          begin
            logic signed [33:0] e;                  // 2^30 - u (signed)
            e = 34'sd1073741824 - 34'(signed'({2'b0, mul_p[63:32]}));
            c_eneg <= e[33];
            mul_a  <= c_y0;
            mul_b  <= e[33] ? 32'(-e) : 32'(e);     // y0 * |2^30 - u|
          end
          c_u <= mul_p[63:32];
          cs  <= C_Y1;
        end
        C_Y1: cs <= C_Y2;
        C_Y2: cs <= C_Y3;
        C_Y3: cs <= C_Y4;
        C_Y4: begin
          c_pr <= c_eneg ? -66'(signed'({2'b0, mul_p})) : 66'(signed'({2'b0, mul_p}));
          cs   <= C_Y5;
        end
        C_Y5: begin
          c_y1 <= 34'(signed'({2'b0, c_y0})) + 34'(c_pr >>> 31);
          cs   <= C_Y6;
        end
        C_Y6: begin
          if (c_y1 < 34'sd2147483648)      c_rm <= 32'd2147483648;
          else if (c_y1 > 34'sd4294967295) c_rm <= 32'd4294967295;
          else                             c_rm <= c_y1[31:0];
          c_k <= 6'(8'sd47 + (c_dd >>> 1) - 8'sd24);
          cs  <= C_PUSH;
        end
        C_PUSH: cs <= C_IDLE;
        default: cs <= C_IDLE;
      endcase
    end
  end

  // the rows ready to emit: {base, S1, Rm, k}
  localparam int EQW = RW + 48 + 32 + 6;
  logic          eq_valid, eq_pop;
  logic [EQW-1:0] eq_q;
  logic [3:0]    eq_count;
  ro_fifo #(.W(EQW), .D(8), .BRAM(1'b0)) u_eq (
    .clk, .rst (rst || start),
    .in_valid (eq_push), .in_ready (eq_in_ready), .in_data ({c_base, c_s1, c_rm, c_k}),
    .out_valid (eq_valid), .out_ready (eq_pop), .out_data (eq_q),
    .count (eq_count)
  );

  // ===================================== EMITTER ============================================
  // One row at a time: words read from the ring, 4 elements per cycle (an int16
  // word in two halves).  Groups issue when the output FIFO has room.
  logic          em_run, em_half;          // a row in progress / the second half of an int16 word
  logic [31:0]   em_wleft, em_col;
  logic signed [47:0] em_s1;
  logic [31:0]   em_rm;
  logic [5:0]    em_k;
  logic          issue, room, grp_last;
  logic [31:0]   col_next;
  assign col_next = em_col + 32'd4;
  assign grp_last = (col_next >= k_n) || ((em_wleft == 32'd1) && (k_a32 || em_half));
  assign eq_pop   = !em_run && eq_valid;
  assign issue    = em_run && room;
  assign prm_idx  = em_col[PIW+1:2];

  always_ff @(posedge clk) begin
    if (rst || start) begin
      em_run <= 1'b0;
      rptr   <= '0;
      em_half <= 1'b0;
    end else if (eq_pop) begin
      {rptr, em_s1, em_rm, em_k} <= eq_q;
      em_run   <= 1'b1;
      em_wleft <= k_words;
      em_col   <= '0;
      em_half  <= 1'b0;
    end else if (issue) begin
      if (k_a32 || em_half || grp_last) begin       // the word is consumed (a row may end on a half)
        rptr     <= rptr + RW'(1);
        em_wleft <= em_wleft - 32'd1;
        em_half  <= 1'b0;
      end else begin
        em_half  <= 1'b1;
      end
      em_col <= col_next;
      if (grp_last) em_run <= 1'b0;
    end
  end

  // ring credits: a word is freed when its last read has issued
  always_ff @(posedge clk) begin
    if (rst || start) used <= '0;
    else used <= used + (RW+1)'(ring_we) - (RW+1)'(issue && (k_a32 || em_half || grp_last));
  end

  // Emitter pipeline.  A group issued in cycle t: stage s data is registered at
  // the end of cycle t + s and valid with v[s] in cycle t + s + 1.  The ring
  // word read at t is ring_q0 (stage 0), ring_q1 (stage 1); the parameters
  // read at t (prm_idx) are valid in cycle t + 2; lanes x3 and p3 are stage 2.
  logic [NS-1:0] v, lq;                     // valid, last group of the row
  logic [3:0]    nval [NS];
  logic          hf1, hf2;                  // the group's half (int16): stages 0, 1
  logic [3:0]    nv_in;
  (* shreg_extract = "no" *) logic signed [47:0] s1q [NS];   // S1 along the pipeline (changes per row)
  (* shreg_extract = "no" *) logic [31:0]   rmq [NS];
  (* shreg_extract = "no" *) logic [5:0]    kq [NS];

  always_comb begin
    logic [31:0] remain;
    remain = k_n - em_col;
    nv_in  = (remain >= 32'd4) ? 4'd4 : remain[3:0];
  end

  logic signed [31:0] x3 [E32];
  param_t             p3 [E32];

  always_ff @(posedge clk) begin
    if (rst || start) v <= '0;
    else v <= {v[NS-2:0], issue};
    lq <= {lq[NS-2:0], grp_last};
    nval[0] <= nv_in; s1q[0] <= em_s1; rmq[0] <= em_rm; kq[0] <= em_k;
    for (int s = 1; s < NS; s++) begin
      nval[s] <= nval[s-1]; s1q[s] <= s1q[s-1]; rmq[s] <= rmq[s-1]; kq[s] <= kq[s-1];
    end
    hf1 <= em_half; hf2 <= hf1;
    // stage 2: lanes from ring_q1, the parameters
    for (int l = 0; l < E32; l++) begin
      x3[l] <= k_a32 ? ring_q1[32*l +: 32]
                     : 32'(signed'(hf2 ? ring_q1[EW*(l+4) +: EW] : ring_q1[EW*l +: EW]));
      p3[l] <= prm_q[l];
    end
  end

  // Per-lane arithmetic (stages 3 .. 14)
  logic [15:0] res [E32];
  for (genvar l = 0; l < E32; l++) begin : g_lane
    logic signed [47:0] nx4, a5, a6;          // n*x, A
    logic signed [47:0] s1_4;
    logic signed [79:0] pm7, pm8;             // A * Rm
    logic signed [47:0] p1_9;                 // floor(A*Rm / 2^k) (|P1| <= 2^31 for valid data)
    logic signed [79:0] q10, q11;             // P1 * G
    logic signed [63:0] b10, b11;
    logic signed [79:0] s12;
    rne_t               ra13;
    logic signed [63:0] r14;
    (* shreg_extract = "no" *) param_t p4, p5, p6, p7, p8, p9, p10, p11, p12;
    logic [31:0] rm5, rm6;
    logic [5:0]  k8;
    always_ff @(posedge clk) begin
      // e4
      nx4  <= 48'(x3[l]) * 48'(signed'({1'b0, k_n13}));
      s1_4 <= s1q[2];
      p4   <= p3[l];
      // e5
      a5   <= k_ln ? (nx4 - s1_4) : 48'(x3_d[l]);
      p5   <= p4;
      rm5  <= rmq[3];                        // the group's Rm (set at t + 3), with a5
      // e6: the product's input registers
      a6   <= a5;
      rm6  <= rm5;
      p6   <= p5;
      // e7: A * Rm
      pm7  <= a6 * 80'(signed'({1'b0, rm6}));
      p7   <= p6;
      // e8
      pm8  <= pm7;
      p8   <= p7;
      k8   <= kq[6];                         // valid in t + 7, with pm8
      // e9: the floor shift
      p1_9 <= 48'(pm8 >>> k8);
      p9   <= p8;
      // e10: * G, the bias
      q10  <= p1_9 * 80'(p9.g);
      b10  <= 64'(p9.bq) <<< k_bqsh;
      p10  <= p9;
      // e11
      q11  <= q10;
      b11  <= b10;
      p11  <= p10;
      // e12
      s12  <= q11 + 80'(b11);
      p12  <= p11;
      // e13, e14: the rounding
      ra13 <= rne_a(64'(s12), p12.sh_c);
      r14  <= rne_b(ra13);
      // e15
      res[l] <= sat16(r14);
    end
  end
  // x delayed one stage for the rms path (A = x at e5)
  logic signed [31:0] x3_d [E32];
  always_ff @(posedge clk) for (int l = 0; l < E32; l++) x3_d[l] <= x3[l];

  // Pack: 4 results per group, 8 per output word (two groups), a row's last
  // group flushes a half word with zeros.
  localparam int PS = 15;                   // res is stage 14 data: v[14]
  logic          wv, half_pend;
  logic [BW-1:0] wd;
  logic [63:0]   wd_lo, packed_half;
  always_comb begin
    packed_half = '0;
    for (int l = 0; l < E32; l++)
      if (l < int'(nval[PS-1])) packed_half[EW*l +: EW] = res[l];
  end
  always_ff @(posedge clk) begin
    if (rst || start) begin
      wv <= 1'b0; half_pend <= 1'b0;
    end else begin
      wv <= 1'b0;
      if (v[PS-1]) begin
        if (half_pend) begin
          wv <= 1'b1; wd <= {packed_half, wd_lo}; half_pend <= 1'b0;
        end else if (lq[PS-1]) begin
          wv <= 1'b1; wd <= {64'b0, packed_half};
        end else begin
          wd_lo <= packed_half; half_pend <= 1'b1;
        end
      end
    end
  end

  logic        cf_in_ready;
  logic [FW:0] cf_count, inflight;
  ro_fifo #(.W(BW), .D(CF_D), .BRAM(1'b0)) u_cf (
    .clk, .rst (rst || start),
    .in_valid (wv), .in_ready (cf_in_ready), .in_data (wd),
    .out_valid (c_valid), .out_ready (c_ready), .out_data (c_data),
    .count (cf_count)
  );
  always_ff @(posedge clk) begin
    if (rst || start) inflight <= '0;
    else inflight <= inflight + (FW+1)'(issue) - (FW+1)'(v[PS-1]);
  end
  assign room = ({1'b0, cf_count} + {1'b0, inflight}) < (FW+2)'(CF_D - 2);

  assign idle = !ld_run && !sv1 && !sv2 && !sv3 && !sv4 && !sv5 && !rq_valid && (cs == C_IDLE) && !eq_valid &&
                !em_run && (v == '0) && !wv && !half_pend && (cf_count == '0);

  logic unused;
  assign unused = ^{cf_in_ready, eq_count, c_t, c_u, c_dd[7], c_e[6], rq_in_ready};

endmodule
