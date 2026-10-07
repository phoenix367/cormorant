// ---------------------------------------------------------------------------
// vo_smx — the softmax of the RTL VectorOPKernel (OP_SOFTMAX row mode,
// OP_SOFTMAX_T column mode; doc/plans/SOFTMAX_PLAN.md §2): operand a's word
// stream -> P words, bit for bit the integer specification of VectorOP.h
// smx_vector.  Per vector (n inputs x, v of them valid):
//
//   m = max_{j<v} x_j;  y_j = ((m - x_j) * Cm) >> Cs;
//   e_j = TAB[y_j mod 4096] >> (y_j div 4096)   (0 when y_j >= 17 * 4096: TAB < 2^17)
//   S = sum_{j<v} e_j;  R = floor(2^40 / S);
//   P_j = min((e_j * R + 2^(39 - f_p)) >> (40 - f_p), 32767), 0 for j >= v
//
// One unit at a time: a row (row mode), or a block of 16 query columns
// (column mode: the input is K keys x 2 words, the output 16 rows of K):
//   V     the valid length of each vector (smx_mask; 1 or 16 vectors)
//   LOAD  the unit's words into the buffer; the maxima (row: per lane)
//   MRED  (row) the lane maxima reduced to m
//   EXP   the buffer read once: e, summed per vector (row: per lane)
//   SUM   the sums complete; SRED (row) the lane sums reduced to S
//   DIV   R = floor(2^40 / S) per vector: restoring, 25 cycles (S >= 2^16
//         whenever v > 0: the maximum's e is 2^16)
//   OUT   the buffer read again: e recomputed (the same pipeline), then P;
//         one word per cycle while the output FIFO has room for it and for
//         every word in flight
//   NEXT  the last OUT words past the stage that reads m and v (the next
//         unit's V and LOAD overwrite them)
// e is recomputed rather than stored: the buffer keeps the inputs only and
// EXP and OUT share one pipeline (r0 .. r20 below).
//
// The buffer: 8 banks of 2048 x 16 (block RAM, two cycles of read latency).
//   row mode     element 8w + i of the row in bank i at address w;
//   column mode  element (key k, query 8w + i) in bank (k + i) mod 8 at
//                address 2k + w: a key's word is one write (one address, the
//                lanes rotated by k), and an output word — keys 8g .. 8g + 7
//                of query 8w + i — one read (bank b at address
//                16g + 2((b - i) mod 8) + w, the lanes rotated back by i).
// Limits (VectorOP.h): rows of at most 2048 elements, at most 1024 keys.
// Longer ones wrap the buffer (wrong results, never a hang): every count is
// 32 bits wide, so a job always consumes and produces its words.
// ---------------------------------------------------------------------------
module vo_smx
  import vo_pkg::*;
(
  input  logic          clk,
  input  logic          rst,
  input  logic          start,            // one cycle; the job constants are stable from here on
  input  logic          en,               // a softmax job with at least one unit
  input  logic          col,              // column mode (OP_SOFTMAX_T)
  input  logic [31:0]   size,             // row length / keys
  input  logic [31:0]   nw,               // ceil(size / 8): output words per vector
  input  logic [31:0]   units,            // rows / blocks of 16 queries
  input  logic [31:0]   cm,               // smx_cm: Cm [23:0]
  input  logic [31:0]   cfg,              // smx_cfg: Cs [5:0], f_p [12:8]
  input  logic [31:0]   mask,             // smx_mask: valid0 [15:0], period [31:16]

  input  logic          in_valid,
  output logic          in_ready,
  input  logic [BW-1:0] in_data,

  output logic          out_valid,
  input  logic          out_ready,
  output logic [BW-1:0] out_data,

  output logic          idle
);
  localparam int NQ = 16;                   // vectors of a column-mode unit
  localparam int BA = 11;                   // buffer address bits
  localparam int NP = 21;                   // pipeline stages r0 .. r20
  localparam int FW = $clog2(CF_D);

  // stages of the read pipeline
  localparam int P_X  = 3;                  // unrotated x, m, the valid bound
  localparam int P_E  = 14;                 // e (EXP: summed here)
  localparam int P_O  = 20;                 // P (OUT: to the FIFO)

  function automatic logic [EW-1:0] lane(input logic [BW-1:0] w, input logic [2:0] i);
    return w[EW*i +: EW];
  endfunction

  function automatic logic [2:0] add3(input logic [2:0] a, input logic [2:0] b);
    return a + b;
  endfunction

  function automatic logic [2:0] sub3(input logic [2:0] a, input logic [2:0] b);
    return a - b;
  endfunction

  // Job constants ----------------------------------------------------------------------
  logic        k_col;
  logic [23:0] k_cm;
  logic [5:0]  k_cs;
  logic [4:0]  k_sft;                       // 31 - f_p: P = (e R + k_rnd) >> (9 + k_sft)
  logic [47:0] k_rnd;                       // 2^(39 - f_p)
  logic [15:0] k_v0, k_per;
  logic [16:0] k_size;                      // min(size, 2^17 - 1): bounds a valid length
  logic [31:0] k_win, k_nw;                 // words in per unit, out per vector
  logic        k_nw1;                       // k_nw == 1
  logic [3:0]  k_vl;                        // vectors per unit - 1 (15 / 0)

  always_ff @(posedge clk)
    if (start) begin
      k_col  <= col;
      k_cm   <= cm[23:0];
      k_cs   <= cfg[5:0];
      k_sft  <= 5'd31 - cfg[12:8];
      k_rnd  <= 48'd1 << (6'd39 - {1'b0, cfg[12:8]});
      k_v0   <= mask[15:0];
      k_per  <= mask[31:16];
      k_size <= (size > 32'h1FFFF) ? 17'h1FFFF : size[16:0];
      k_win  <= col ? {size[30:0], 1'b0} : nw;
      k_nw   <= nw;
      k_nw1  <= (nw == 32'd1);
      k_vl   <= col ? 4'd15 : 4'd0;
    end

  // State ----------------------------------------------------------------------------------
  typedef enum logic [3:0] {S_IDLE, S_V, S_LOAD, S_MRED, S_EXP, S_SUM, S_SRED, S_DIV, S_OUT,
                            S_NEXT} st_t;
  st_t         st;
  logic [31:0] units_left;
  logic [1:0]  red_c;                       // MRED / SRED step

  // V: valid lengths, v = min(size, valid0 + (period ? q mod period : 0))
  logic [4:0]  vcnt;
  logic        va_v;
  logic [3:0]  va_i;
  logic [16:0] va_s;
  logic [15:0] qmod;                        // the next vector's q mod period
  logic [16:0] vq [NQ];

  // LOAD: in -> l1 (word, its index) -> l2 (bank data, lane masks) -> buffer, maxima
  logic [31:0] ld_left;
  logic        ld_on, take;
  logic [15:0] ld_t;
  logic        l1_v, l2_v;
  logic [BW-1:0] l1_d;
  logic [15:0] l1_t;
  logic [BA-1:0] l2_a;
  logic [EW-1:0] l2_bd [E];
  logic signed [EW-1:0] l2_x [E];
  logic [E-1:0] l2_m;
  logic        l2_w;
  logic signed [EW-1:0] mx [NQ];            // row: lane maxima (m in mx[0] after MRED); column: per query

  // EXP / OUT issue
  logic [31:0] rd_left;
  logic        rd_last;
  logic [15:0] rd_t;
  logic [15:0] og;                          // OUT: word of the vector
  logic [31:0] og_left;
  logic        og_last;
  logic [3:0]  oq;                          // OUT: vector of the unit
  logic        oq_last;
  logic        iss_exp, iss_out, room;

  // DIV
  logic        dv_busy;
  logic [3:0]  dv_q;
  logic [4:0]  dv_it;
  logic [27:0] dv_s;
  logic [28:0] dv_rem, dv_r2;
  logic [24:0] dv_quo;
  logic        dv_ge;
  logic [27:0] acc [NQ];                    // row: lane sums (S in acc[0] after SRED); column: per query
  logic [24:0] rr [NQ];                     // R per vector

  // pipeline valid / OUT tags
  logic [NP-1:0] pv, po;
  logic [FW:0]   inflight;                  // OUT words issued, not yet in the FIFO

  assign in_ready = ld_on;
  assign take     = in_valid && ld_on;
  assign iss_exp  = (st == S_EXP);
  assign iss_out  = (st == S_OUT) && room;
  assign dv_r2    = {dv_rem[27:0], 1'b0};
  assign dv_ge    = dv_r2 >= {1'b0, dv_s};

  // Control ------------------------------------------------------------------------------------
  always_ff @(posedge clk) begin
    if (rst) begin
      st      <= S_IDLE;
      va_v    <= 1'b0;
      ld_on   <= 1'b0;
      l1_v    <= 1'b0;
      l2_v    <= 1'b0;
      dv_busy <= 1'b0;
      pv      <= '0;
      po      <= '0;
    end else begin
      va_v <= 1'b0;
      l1_v <= take;
      l2_v <= l1_v;
      pv   <= {pv[NP-2:0], iss_exp || iss_out};
      po   <= {po[NP-2:0], iss_out};
      case (st)
        S_IDLE: if (start && en) begin
          st         <= S_V;
          vcnt       <= '0;
          units_left <= units;
          qmod       <= '0;
        end
        S_V: begin
          if (vcnt <= {1'b0, k_vl}) begin
            va_v <= 1'b1;
            va_i <= vcnt[3:0];
            va_s <= {1'b0, k_v0} + ((k_per != '0) ? {1'b0, qmod} : 17'd0);
            qmod <= ({1'b0, qmod} + 17'd1 == {1'b0, k_per}) ? '0 : qmod + 16'd1;
          end
          vcnt <= vcnt + 5'd1;
          if (vcnt == {1'b0, k_vl} + 5'd1) begin   // the last length is written this cycle
            st      <= S_LOAD;
            ld_left <= k_win;
            ld_on   <= 1'b1;
            ld_t    <= '0;
          end
        end
        S_LOAD: begin
          if (take) begin
            ld_left <= ld_left - 32'd1;
            ld_t    <= ld_t + 16'd1;
            if (ld_left == 32'd1) ld_on <= 1'b0;
          end
          if (!ld_on && !l1_v) begin               // the last word is written this cycle
            st      <= k_col ? S_EXP : S_MRED;
            red_c   <= '0;
            rd_left <= k_win;
            rd_last <= (k_win == 32'd1);
            rd_t    <= '0;
          end
        end
        S_MRED: begin
          red_c <= red_c + 2'd1;
          if (red_c == 2'd2) st <= S_EXP;
        end
        S_EXP: begin                               // one read per cycle
          rd_t    <= rd_t + 16'd1;
          rd_left <= rd_left - 32'd1;
          rd_last <= (rd_left == 32'd2);
          if (rd_last) st <= S_SUM;
        end
        S_SUM: if (pv[P_E:0] == '0) begin          // every e summed
          st    <= k_col ? S_DIV : S_SRED;
          red_c <= '0;
          dv_q  <= '0;
        end
        S_SRED: begin
          red_c <= red_c + 2'd1;
          if (red_c == 2'd2) st <= S_DIV;
        end
        S_DIV: begin
          if (!dv_busy) begin
            dv_busy <= 1'b1;
            dv_s    <= acc[dv_q];
            dv_rem  <= 29'h8000;                   // 2^40 / 2^25: quotient bits 24 .. 0 follow
            dv_quo  <= '0;
            dv_it   <= 5'd25;
          end else begin
            dv_rem <= dv_ge ? dv_r2 - {1'b0, dv_s} : dv_r2;
            dv_quo <= {dv_quo[23:0], dv_ge};
            dv_it  <= dv_it - 5'd1;
            if (dv_it == 5'd1) begin
              dv_busy <= 1'b0;
              for (int q = 0; q < NQ; q++)
                if (dv_q == 4'(q)) rr[q] <= (dv_s == '0) ? '0 : {dv_quo[23:0], dv_ge};
              dv_q <= dv_q + 4'd1;
              if (dv_q == k_vl) begin
                st      <= S_OUT;
                og      <= '0;
                og_left <= k_nw;
                og_last <= k_nw1;
                oq      <= '0;
                oq_last <= (k_vl == '0);
              end
            end
          end
        end
        S_OUT: if (room) begin
          if (og_last) begin
            og      <= '0;
            og_left <= k_nw;
            og_last <= k_nw1;
            oq      <= oq + 4'd1;
            oq_last <= (oq + 4'd1 == k_vl);
            if (oq_last) st <= S_NEXT;
          end else begin
            og      <= og + 16'd1;
            og_left <= og_left - 32'd1;
            og_last <= (og_left == 32'd2);
          end
        end
        S_NEXT: if (pv[P_X-1:0] == '0) begin       // m and v no longer read
          if (units_left == 32'd1) begin
            st <= S_IDLE;
          end else begin
            units_left <= units_left - 32'd1;
            st         <= S_V;
            vcnt       <= '0;
          end
        end
        default: st <= S_IDLE;
      endcase
    end
  end

  // Valid lengths, maxima, sums -------------------------------------------------------------------
  logic newu;                               // a unit begins: maxima and sums cleared
  assign newu = (st == S_V) && (vcnt == '0);

  always_ff @(posedge clk) begin
    for (int q = 0; q < NQ; q++)
      if (va_v && va_i == 4'(q)) vq[q] <= (va_s < k_size) ? va_s : k_size;

    // LOAD stage 1 / 2
    if (take) begin
      l1_d <= in_data;
      l1_t <= ld_t;
    end
    begin
      logic [2:0] r;
      r    = k_col ? l1_t[3:1] : 3'd0;
      l2_a <= l1_t[BA-1:0];
      l2_w <= k_col && l1_t[0];
      for (int b = 0; b < E; b++) l2_bd[b] <= lane(l1_d, sub3(3'(b), r));
      for (int i = 0; i < E; i++) begin
        l2_x[i] <= lane(l1_d, 3'(i));
        l2_m[i] <= k_col ? ({4'b0, l1_t[15:1]} < {2'b0, vq[{l1_t[0], 3'(i)}]})
                         : ({l1_t, 3'(i)} < {2'b0, vq[0]});
      end
    end

    // maxima: cleared, LOAD stage 3, MRED
    if (newu) begin
      for (int q = 0; q < NQ; q++) mx[q] <= 16'sh8000;
    end else if (l2_v) begin
      for (int q = 0; q < NQ; q++)
        if (l2_w == q[3] && l2_m[q[2:0]] && l2_x[q[2:0]] > mx[q]) mx[q] <= l2_x[q[2:0]];
    end else if (st == S_MRED) begin
      case (red_c)
        2'd0:    for (int i = 0; i < 4; i++) if (mx[i+4] > mx[i]) mx[i] <= mx[i+4];
        2'd1:    for (int i = 0; i < 2; i++) if (mx[i+2] > mx[i]) mx[i] <= mx[i+2];
        default: if (mx[1] > mx[0]) mx[0] <= mx[1];
      endcase
    end
  end

  // Issue (r0) and the buffer ------------------------------------------------------------------------
  logic [BA-1:0] addr0 [E];
  logic [2:0]    rot [P_X];                 // r0 .. r2: lane rotation
  logic [15:0]   idx [P_X];                 // r0 .. r2: row word / key / output word
  logic          ws [P_E+1];                // r0 .. r14: column EXP: the word's query half
  logic [3:0]    qs [P_E+1];                // r0 .. r14: OUT: the vector

  always_ff @(posedge clk) begin
    if (st == S_OUT) begin
      for (int b = 0; b < E; b++)
        addr0[b] <= k_col ? {og[6:0], sub3(3'(b), oq[2:0]), oq[3]} : og[BA-1:0];
      rot[0] <= k_col ? oq[2:0] : 3'd0;
      idx[0] <= og;
      ws[0]  <= 1'b0;
      qs[0]  <= oq;
    end else begin
      for (int b = 0; b < E; b++) addr0[b] <= rd_t[BA-1:0];
      rot[0] <= k_col ? rd_t[3:1] : 3'd0;
      idx[0] <= k_col ? {1'b0, rd_t[15:1]} : rd_t;
      ws[0]  <= k_col && rd_t[0];
      qs[0]  <= '0;
    end
    for (int s = 1; s < P_X; s++) begin
      rot[s] <= rot[s-1];
      idx[s] <= idx[s-1];
    end
    for (int s = 1; s <= P_E; s++) begin
      ws[s] <= ws[s-1];
      qs[s] <= qs[s-1];
    end
  end

  logic [EW-1:0] bq2 [E];                   // r2: the banks' words
  for (genvar b = 0; b < E; b++) begin : g_bank
    (* ram_style = "block" *) logic [EW-1:0] mem [1 << BA];
    logic [EW-1:0] q1, q2;
    always_ff @(posedge clk) begin
      if (l2_v) mem[l2_a] <= l2_bd[b];
      q1 <= mem[addr0[b]];
      q2 <= q1;                             // the block RAM's output register
    end
    assign bq2[b] = q2;
  end

  // e: r3 .. r14 -----------------------------------------------------------------------------------------
  logic signed [EW-1:0] x3 [E], m3 [E];
  logic [16:0] lim3 [E];
  logic [18:0] pos3 [E];
  logic [15:0] d4 [E], da5 [E];
  logic [39:0] dm6 [E], dp7 [E], s8 [E], y9 [E];
  logic [E-1:0] mk4, mk5, mk6, mk7, mk8, mk9;
  logic [11:0] ra10 [E];
  logic [4:0]  sh10 [E], sh11 [E], sh12 [E];
  logic [1:0]  sh13 [E];
  logic [E-1:0] z10, z11, z12;
  logic [16:0] tab12 [E];
  logic [16:0] e13 [E], e14 [E];

  always_ff @(posedge clk) begin
    for (int l = 0; l < E; l++) begin
      // r3: unrotate; this vector's maximum and valid length, the element's position
      x3[l]   <= bq2[add3(3'(l), rot[2])];
      m3[l]   <= !k_col ? mx[0] : po[2] ? mx[qs[2]] : mx[{ws[2], 3'(l)}];
      lim3[l] <= !k_col ? vq[0] : po[2] ? vq[qs[2]] : vq[{ws[2], 3'(l)}];
      pos3[l] <= (k_col && !po[2]) ? {3'b0, idx[2]} : {idx[2], 3'(l)};
      // r4: d = m - x (0 .. 65535 on the valid lanes), the lane mask
      d4[l]   <= 16'(m3[l] - x3[l]);
      mk4[l]  <= pos3[l] < {2'b0, lim3[l]};
      // r5 .. r7: d * Cm (DSP: A / B, M, P registers)
      da5[l]  <= d4[l];
      dm6[l]  <= da5[l] * k_cm;
      dp7[l]  <= dm6[l];
      // r8, r9: y = d Cm >> Cs
      s8[l]   <= dp7[l] >> {k_cs[5:3], 3'b0};
      y9[l]   <= s8[l] >> k_cs[2:0];
      // r10: the table address; e = 0 past the table's 17 bits or on a masked lane
      ra10[l] <= y9[l][11:0];
      sh10[l] <= y9[l][16:12];
      z10[l]  <= !mk9[l] || (y9[l][39:17] != '0) || (y9[l][16:12] > 5'd16);
      // r11, r12: the table (vo_smx_rom)
      sh11[l] <= sh10[l];
      sh12[l] <= sh11[l];
      // r13, r14: e = TAB >> sh
      // (a masked lane may come from a buffer word this unit never wrote: its
      // shift is zeroed too, so e stays 0 — and known — in a 4-state simulation)
      e13[l]  <= z12[l] ? '0 : tab12[l] >> {sh12[l][4:2], 2'b0};
      sh13[l] <= z12[l] ? 2'b0 : sh12[l][1:0];
      e14[l]  <= e13[l] >> sh13[l];
    end
    mk5 <= mk4; mk6 <= mk5; mk7 <= mk6; mk8 <= mk7; mk9 <= mk8;
    z11 <= z10; z12 <= z11;
  end

  for (genvar r = 0; r < E / 2; r++) begin : g_rom
    vo_smx_rom u_rom (
      .clk,
      .addr_a (ra10[2*r]),  .addr_b (ra10[2*r+1]),
      .q_a    (tab12[2*r]), .q_b    (tab12[2*r+1])
    );
  end

  // sums (EXP at r14), cleared per unit, reduced (SRED)
  always_ff @(posedge clk) begin
    if (newu) begin
      for (int q = 0; q < NQ; q++) acc[q] <= '0;
    end else if (pv[P_E] && !po[P_E]) begin
      for (int q = 0; q < NQ; q++)
        if (ws[P_E] == q[3]) acc[q] <= acc[q] + 28'(e14[q[2:0]]);
    end else if (st == S_SRED) begin
      case (red_c)
        2'd0:    for (int i = 0; i < 4; i++) acc[i] <= acc[i] + acc[i+4];
        2'd1:    for (int i = 0; i < 2; i++) acc[i] <= acc[i] + acc[i+2];
        default: acc[0] <= acc[0] + acc[1];
      endcase
    end
  end

  // P: r15 .. r20 ----------------------------------------------------------------------------------------
  logic [16:0] ea15 [E];
  logic [24:0] rr15;
  logic [41:0] em16 [E];
  logic [47:0] ep17 [E];
  logic [38:0] o18 [E], o19 [E];
  logic [EW-1:0] p20 [E];

  always_ff @(posedge clk) begin
    rr15 <= rr[qs[P_E]];
    for (int l = 0; l < E; l++) begin
      // r15 .. r17: e * R + 2^(39 - f_p) (DSP: A / B, M, P with C)
      ea15[l] <= e14[l];
      em16[l] <= ea15[l] * rr15;
      ep17[l] <= 48'(em16[l]) + k_rnd;
      // r18, r19: >> (40 - f_p) = >> 9 >> k_sft;  r20: saturate
      o18[l]  <= ep17[l][47:9] >> {k_sft[4:3], 3'b0};
      o19[l]  <= o18[l] >> k_sft[2:0];
      p20[l]  <= (o19[l][38:15] != '0) ? 16'h7FFF : {1'b0, o19[l][14:0]};
    end
  end

  // Output FIFO and credits -----------------------------------------------------------------------------
  logic          cf_in_valid, cf_in_ready;
  logic [BW-1:0] cf_in_data;
  logic [FW:0]   cf_count;
  assign cf_in_valid = pv[P_O] && po[P_O];
  always_comb for (int l = 0; l < E; l++) cf_in_data[EW*l +: EW] = p20[l];

  vo_fifo #(.W(BW), .D(CF_D), .BRAM(1'b0)) u_cf (
    .clk, .rst,
    .in_valid (cf_in_valid), .in_ready (cf_in_ready), .in_data (cf_in_data),
    .out_valid,              .out_ready,              .out_data,
    .count    (cf_count)
  );

  assign room = ({1'b0, cf_count} + {1'b0, inflight}) < (FW+2)'(CF_D);
  always_ff @(posedge clk) begin
    if (rst) inflight <= '0;
    else     inflight <= inflight + (FW+1)'(iss_out) - (FW+1)'(cf_in_valid);
  end

  assign idle = (st == S_IDLE) && (pv == '0) && (inflight == '0) && !out_valid;

  logic unused;
  assign unused = ^{cf_in_ready, cm[31:24], cfg[31:13], cfg[7:6]};

endmodule
