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
  //
  // The 16 lanes (their DSPs for x * x) spread over several DSP columns, so
  // the stall decision is not one net to all of them: each group of LG lanes
  // keeps its own copy of the few registers it depends on (beat valid / last
  // flags, the finaliser's channel count, the FIFO room) and computes an
  // identical adv locally.  Copy NG is the central one (bt_ready, the AVG
  // reciprocals, the finaliser).  The room for the finaliser's output is
  // registered twice (central, then per copy), so it is reserved for two more
  // issues than are in flight: f_issue never overruns the bundle FIFO.
  localparam int NG = 4;                   // lane groups
  localparam int LG = NL / NG;             // lanes per group

  logic [5:0]  fb_count;
  logic [FL-1:0] f_v;
  logic        room_c;
  always_ff @(posedge clk)
    room_c <= !rst && (32'(fb_count) + 32'($countones(f_v)) + 32'd2 < FD);

  // Every register below lives in exactly one always_ff of one generate
  // scope; the scopes only export wires (g_adv, g_r1v, ..., the lanes' next
  // accumulator values) and the shared snapshot accd is written by a single
  // process.  (A variable written from several generate scopes is not
  // synthesised as simulated: Vivado mapped the shared accd array wrongly.)
  wire [NG:0] g_adv, g_snap, g_issue, g_r0v, g_r1v, g_fz;

  for (genvar g = 0; g <= NG; g++) begin : g_ctl
    (* keep = "true" *) logic r0v, r0l, r1v, r1l, room;
    (* keep = "true" *) logic [3:0] frem;
    logic fin_free;
    assign fin_free   = (frem == 4'd0) || ((frem == 4'd1) && room);
    assign g_issue[g] = (frem != 4'd0) && room;
    assign g_adv[g]   = !(r1v && r1l && !fin_free);
    assign g_snap[g]  = g_adv[g] && r1v && r1l;
    assign g_r0v[g]   = r0v;
    assign g_r1v[g]   = r1v;
    assign g_fz[g]    = (frem == 4'd0);
    always_ff @(posedge clk) begin
      if (rst) begin
        r0v  <= 1'b0;
        r1v  <= 1'b0;
        frem <= '0;
        room <= 1'b0;
      end else begin
        if (g_adv[g]) begin
          r0v <= bt_valid;
          r1v <= r0v;
        end
        if (g_snap[g])       frem <= 4'(TC);
        else if (g_issue[g]) frem <= frem - 4'd1;
        room <= room_c;
      end
      if (g_adv[g]) begin
        r0l <= bt.last;
        r1l <= r0l;
      end
    end
  end

  logic adv, snap, f_issue;
  assign adv      = g_adv[NG];
  assign snap     = g_snap[NG];
  assign f_issue  = g_issue[NG];
  assign bt_ready = adv;

  // the lanes, LG per group; nx: each lane's next accumulator value
  wire [NL-1:0][31:0] nx;
  for (genvar g = 0; g < NG; g++) begin : g_lanes
    logic [LG*EW-1:0] r0x;
    logic r0f, r1f;
    logic [31:0] r1c [LG];
    logic [31:0] acc [LG];
    logic [LG-1:0][31:0] n;
    // the job's mode, copied per group (it steers the DSPs' operation); it is
    // stable from long before the job's first beat
    (* keep = "true" *) mode_t mode;
    always_ff @(posedge clk) mode <= j.mode;
    always_comb
      for (int i = 0; i < LG; i++) begin
        if (r1f)                n[i] = r1c[i];
        else if (mode == M_MAX) n[i] = ($signed(r1c[i]) > $signed(acc[i])) ? r1c[i] : acc[i];
        else                    n[i] = acc[i] + r1c[i];
      end
    assign nx[g * LG +: LG] = n;
    always_ff @(posedge clk) begin
      if (g_adv[g]) begin
        r0x <= bt.v[g * LG * EW +: LG * EW];
        r0f <= bt.first;
        r1f <= r0f;
        for (int i = 0; i < LG; i++) begin
          logic signed [15:0] x;
          logic signed [31:0] x32;
          x   = signed'(r0x[i * EW +: EW]);
          x32 = 32'(x) <<< 8;
          case (mode)
            M_LP1:   r1c[i] <= x[15] ? -x32 : x32;
            M_LP2:   r1c[i] <= 32'(x * x);
            default: r1c[i] <= x32;
          endcase
        end
      end
      if (g_adv[g] && g_r1v[g])
        for (int i = 0; i < LG; i++) acc[i] <= n[i];
    end
  end

  // the snapshot of a finished group, for the finaliser: one process, one
  // enable (the central copy's; every copy's snap is the same), and a packed
  // vector, so synthesis builds registers and a read mux, not a RAM
  logic [NL-1:0][31:0] accd;
  always_ff @(posedge clk)
    if (snap) accd <= nx;

  // the AVG reciprocals of the group (central)
  logic [5:0]  r0_d0, r0_d1, r1_d0, r1_d1;
  logic [23:0] invd [OWP];
  always_ff @(posedge clk) begin
    if (adv) begin
      r0_d0 <= bt.d0;  r0_d1 <= bt.d1;
      r1_d0 <= r0_d0;  r1_d1 <= r0_d1;
    end
    if (snap) begin
      invd[0] <= INV[r1_d0];
      invd[1] <= INV[r1_d1];
    end
  end

  // Finaliser -------------------------------------------------------------------------------
  logic [2:0]  f_c;

  always_ff @(posedge clk) begin
    if (rst) begin
      f_v <= '0;
    end else begin
      f_v <= {f_v[FL-2:0], f_issue};
    end
    if (snap)         f_c <= '0;
    else if (f_issue) f_c <= f_c + 3'd1;
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

  assign idle = !g_r0v[NG] && !g_r1v[NG] && g_fz[NG] && (f_v == '0) && !fb_valid;

  logic unused;
  assign unused = fb_in_ready ^ ^j;

endmodule
