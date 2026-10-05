// ---------------------------------------------------------------------------
// pl_reduce — accumulate window beats and finalise (the HLS
// process_pool_kernel_tile).
//
//   beats ─► R0 ─► R1 (contribution: x, |x| or x·x) ─► accumulate OWP x TC lanes
//                                               └ last tap: snapshot ─► finaliser,
//                                                 one channel (OWP lanes) per cycle
//
// Accumulators are ap_fixed<32,16> (raw Q16.16, wrapping).  The first tap of a
// group replaces the accumulator (max(x, -32768) = x for MAX).  The finaliser
// (fixed latency FL) produces, per lane, the HLS kernel's result:
//   MAX, LP-1      a
//   AVG            floor(a * round(2^23 / d) / 2^23)
//   LP-2           poly_sqrt(a): range reduction, a cubic in ap_fixed<24,4>
//                  (each Horner step floored), scaled by 2^k
// floored to Q8.8 and saturated.  The finaliser takes a new snapshot only once
// the previous one is done (TC cycles), so groups of fewer than TC taps wait.
// ---------------------------------------------------------------------------
module pl_reduce
  import pl_pkg::*;
(
  input  logic       clk,
  input  logic       rst,
  input  job_t       j,

  input  logic       bt_valid,
  output logic       bt_ready,
  input  beat_t      bt,

  output logic       fb_valid,
  input  logic       fb_ready,
  output logic [OWP*EW-1:0] fb,            // one channel: position p at [p*EW +: EW]

  output logic       idle
);
  localparam int NL = OWP * TC;            // accumulator lanes
  localparam int FD = 32;                  // finalised-bundle FIFO depth
  localparam int FL = 9;                   // finaliser latency

  // the poly_sqrt coefficients, ap_fixed<16,1> raw (0.4434, 0.6432, -0.0943, 0.0077)
  localparam logic signed [15:0] C0 = 16'sd14529, C1 = 16'sd21076, C2 = -16'sd3091, C3 = 16'sd252;

  // AVG reciprocals for d = 0 .. 63 (the contract bounds d by MAXK * MAXK)
  logic [23:0] INV [64];
  for (genvar i = 0; i < 64; i++) begin : g_inv
    assign INV[i] = inv_of(6'(i));
  end

  // Accumulate ----------------------------------------------------------------------------
  // R0 holds the beat, R1 its contributions; R1 enters the accumulators on adv.
  // A group's last tap needs a free finaliser (its snapshot), else all stall.
  logic        r0_v, r1_v, adv;
  beat_t       r0;
  logic        r1_first, r1_last;
  logic [5:0]  r1_d0, r1_d1;
  logic [31:0] r1_c [NL];                  // contributions
  logic [31:0] acc  [NL];
  logic [31:0] accd [NL];                  // snapshot of the finished group
  logic [23:0] invd [OWP];
  logic        snap, fin_free;

  assign adv      = !(r1_v && r1_last && !fin_free);
  assign bt_ready = adv;
  assign snap     = adv && r1_v && r1_last;

  always_ff @(posedge clk) begin
    if (rst) begin
      r0_v <= 1'b0;
      r1_v <= 1'b0;
    end else if (adv) begin
      r0_v <= bt_valid;
      r1_v <= r0_v;
    end
    if (adv) begin
      r0       <= bt;
      r1_first <= r0.first;
      r1_last  <= r0.last;
      r1_d0    <= r0.d0;
      r1_d1    <= r0.d1;
      for (int l = 0; l < NL; l++) begin
        logic signed [15:0] x;
        logic signed [31:0] x32;
        x   = signed'(r0.v[l * EW +: EW]);
        x32 = 32'(x) <<< 8;
        case (j.mode)
          M_LP1:   r1_c[l] <= x[15] ? -x32 : x32;
          M_LP2:   r1_c[l] <= 32'(x * x);
          default: r1_c[l] <= x32;
        endcase
      end
    end
    if (adv && r1_v) begin
      for (int l = 0; l < NL; l++) begin
        logic [31:0] n;
        if (r1_first)             n = r1_c[l];
        else if (j.mode == M_MAX) n = ($signed(r1_c[l]) > $signed(acc[l])) ? r1_c[l] : acc[l];
        else                      n = acc[l] + r1_c[l];
        acc[l] <= n;
        if (r1_last) accd[l] <= n;
      end
      if (r1_last) begin
        invd[0] <= INV[r1_d0];
        invd[1] <= INV[r1_d1];
      end
    end
  end

  // Finaliser -------------------------------------------------------------------------------
  logic [3:0]  f_rem;                      // channels left of the snapshot
  logic [2:0]  f_c;
  logic [FL-1:0] f_v;
  logic [5:0]  fb_count;
  logic        f_issue;

  assign f_issue  = (f_rem != '0) &&
                    (32'(fb_count) + 32'($countones(f_v)) < FD);
  assign fin_free = (f_rem == '0) || ((f_rem == 4'd1) && f_issue);

  always_ff @(posedge clk) begin
    if (rst) begin
      f_rem <= '0;
      f_v   <= '0;
    end else begin
      f_v <= {f_v[FL-2:0], f_issue};
      if (snap) begin
        f_rem <= 4'(TC);
        f_c   <= '0;
      end else if (f_issue) begin
        f_rem <= f_rem - 4'd1;
        f_c   <= f_c + 3'd1;
      end
    end
  end

  // per lane: 1 operands; 2 AVG product, LP-2 range reduction; 3-8 the three
  // Horner steps, each a product (DSP M register) then the add and floor (P
  // register); the AVG product's second cycle in 3; 9 scale and output
  logic [EW-1:0] f_out [OWP];
  for (genvar p = 0; p < OWP; p++) begin : g_fin
    logic signed [31:0] a1, a2, a3;
    logic        [23:0] inv1;
    logic signed [55:0] pr2, pr3;          // AVG: a * inv
    logic        [17:0] m2, m3, m4, m5, m6;   // LP-2: m in [1, 4), 16 fraction bits
    logic signed [4:0]  k2, k3, k4, k5, k6, k7, k8;
    logic               z2, z3, z4, z5, z6, z7, z8;   // LP-2: a <= 0
    logic signed [26:0] h3;                // c3 * m
    logic signed [41:0] h5, h7;            // t * m
    logic signed [23:0] t4, t6, t8;        // Horner steps, ap_fixed<24,4> raw
    logic signed [31:0] r4, r5, r6, r7, r8;   // the result of the other modes

    always_ff @(posedge clk) begin
      // 1
      a1   <= signed'(accd[p * TC + int'(f_c)]);
      inv1 <= invd[p];
      // 2
      pr2 <= a1 * signed'({1'b0, inv1});
      a2  <= a1;
      begin
        logic [4:0]        P;
        logic signed [5:0] tk;
        logic [31:0]       mr;
        P = '0;
        for (int i = 0; i < 31; i++) if (a1[i]) P = 5'(i);
        tk = (6'(P) - 6'sd16) & ~6'sd1;
        mr = (tk >= 0) ? (32'(a1) >> tk) : (32'(a1) << (-tk));
        m2 <= mr[17:0];
        k2 <= 5'(tk >>> 1);
        z2 <= (a1 <= 0);
      end
      // 3-4: t1 = floor(c3 * m + c2)
      h3  <= 27'(C3) * signed'({1'b0, m2});
      pr3 <= pr2;  a3 <= a2;  m3 <= m2;  k3 <= k2;  z3 <= z2;
      t4  <= 24'((48'(h3) + (48'(C2) <<< 16)) >>> 11);
      r4  <= (j.mode == M_AVG) ? 32'(pr3 >>> 23) : a3;
      m4  <= m3;  k4 <= k3;  z4 <= z3;
      // 5-6: t2 = floor(t1 * m + c1)
      h5  <= 42'(t4) * signed'({1'b0, m4});
      m5  <= m4;  k5 <= k4;  z5 <= z4;  r5 <= r4;
      t6  <= 24'((48'(h5) + (48'(C1) <<< 21)) >>> 16);
      m6  <= m5;  k6 <= k5;  z6 <= z5;  r6 <= r5;
      // 7-8: sqrt(m) = floor(t2 * m + c0)
      h7  <= 42'(t6) * signed'({1'b0, m6});
      k7  <= k6;  z7 <= z6;  r7 <= r6;
      t8  <= 24'((48'(h7) + (48'(C0) <<< 21)) >>> 16);
      k8  <= k7;  z8 <= z7;  r8 <= r7;
      // 9: AccData_t(sqrt(m)) scaled by 2^k; Q8.8
      begin
        logic signed [31:0] res, rf;
        res = 32'(t8) >>> 4;
        res = (k8 >= 0) ? (res <<< k8) : (res >>> (-k8));
        rf  = (j.mode == M_LP2) ? (z8 ? 32'sd0 : res) : r8;
        f_out[p] <= sat16(rf);
      end
    end
  end

  logic fb_in_ready;
  pl_fifo #(.W(OWP * EW), .D(FD), .BRAM(1'b0)) u_fb (
    .clk, .rst,
    .in_valid (f_v[FL-1]), .in_ready (fb_in_ready), .in_data ({f_out[1], f_out[0]}),
    .out_valid(fb_valid),  .out_ready(fb_ready),    .out_data(fb),
    .count    (fb_count)
  );

  assign idle = !r0_v && !r1_v && (f_rem == '0) && (f_v == '0) && !fb_valid;

  logic unused;
  assign unused = fb_in_ready ^ ^j;

endmodule
