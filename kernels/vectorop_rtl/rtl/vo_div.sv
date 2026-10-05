// ---------------------------------------------------------------------------
// vo_div — Q8.8 division, one lane per cycle, fully pipelined (26 cycles).
//
//   q = b == 0 ? 0 : sat16(trunc((a << 8) / b)),  then the act activation
//
// exactly as the HLS kernel's ap_fixed<16,8> division (VectorOP.cpp sub_div):
// the 24-bit dividend a << 8 divided by b with C semantics (the quotient
// truncated toward zero), then saturated to int16.  Computed on magnitudes
// with a restoring radix-2 divider (|a| << 8 <= 2^23, |b| <= 2^15).
// ---------------------------------------------------------------------------
module vo_div
  import vo_pkg::*;
(
  input  logic        clk,
  input  logic        rst,
  input  logic [1:0]  act,

  input  logic        in_valid,
  input  logic [15:0] in_a,
  input  logic [15:0] in_b,
  input  logic [2:0]  in_lane,

  output logic        out_valid,
  output logic [15:0] out_q,
  output logic [2:0]  out_lane
);
  localparam int NQ = 24;                   // quotient bits

  // Stage 0: magnitudes and signs.
  logic        v   [NQ+1];
  logic [15:0] rem [NQ+1];                  // partial remainder (< divisor)
  logic [23:0] num [NQ+1];                  // dividend bits not yet consumed (MSB next)
  logic [23:0] quo [NQ+1];
  logic [15:0] den [NQ+1];
  logic        neg [NQ+1];
  logic        bz  [NQ+1];
  logic [2:0]  ln  [NQ+1];

  always_ff @(posedge clk) begin
    logic [15:0] ua, ub;
    ua = in_a[15] ? 16'(-in_a) : in_a;      // |-32768| = 0x8000 as unsigned
    ub = in_b[15] ? 16'(-in_b) : in_b;
    v[0]   <= rst ? 1'b0 : in_valid;
    rem[0] <= '0;
    num[0] <= {ua, 8'b0};
    quo[0] <= '0;
    den[0] <= ub;
    neg[0] <= in_a[15] ^ in_b[15];
    bz[0]  <= (in_b == '0);
    ln[0]  <= in_lane;
  end

  // Stages 1..NQ: one quotient bit each.
  for (genvar i = 1; i <= NQ; i++) begin : g_step
    always_ff @(posedge clk) begin
      logic [16:0] t;
      t = {rem[i-1], num[i-1][23]};
      v[i]   <= rst ? 1'b0 : v[i-1];
      num[i] <= {num[i-1][22:0], 1'b0};
      den[i] <= den[i-1];
      neg[i] <= neg[i-1];
      bz[i]  <= bz[i-1];
      ln[i]  <= ln[i-1];
      if (t >= {1'b0, den[i-1]}) begin
        rem[i] <= 16'(t - {1'b0, den[i-1]});
        quo[i] <= {quo[i-1][22:0], 1'b1};
      end else begin
        rem[i] <= t[15:0];
        quo[i] <= {quo[i-1][22:0], 1'b0};
      end
    end
  end

  // Final stage: sign, saturation, zero divisor, activation.
  always_ff @(posedge clk) begin
    logic [23:0] q;
    logic [15:0] s;
    q = quo[NQ];
    if (bz[NQ])                 s = '0;
    else if (!neg[NQ])          s = (q > 24'd32767) ? 16'h7FFF : q[15:0];
    else                        s = (q > 24'd32768) ? 16'h8000 : 16'(-q[15:0]);
    out_valid <= rst ? 1'b0 : v[NQ];
    out_q     <= activate(s, act);
    out_lane  <= ln[NQ];
  end

endmodule
