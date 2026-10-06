// ---------------------------------------------------------------------------
// mm_mac — one DSP48E2 multiply-accumulate slice, P = C + A * B, with the
// accumulator kept outside (the lane's LUTRAM).  Every operand arrives from
// a register outside the slice over a long route (the MAC array spans many
// DSP columns), so it enters the DSP's own input registers (AREG = BREG = 2,
// CREG, MREG, PREG):
//
//   a (B beat element, every cycle)  -> B port: B1 -> B2
//   b (tap, held for a whole B row)  -> A port: A1 -> A2 (b_ce)
//   M  = A2 * B2 -> MREG                c -> CREG (c_rst: 0 on a run's first row)
//   P  = CREG + MREG -> PREG            (32-bit, wraps like ap_fixed<32,16>)
//
// The B beat element is shared by the 8 MACs of a beat lane, so it goes to
// the 18-bit B port, where its sign bit drives 3 pins per slice; on the
// 27-bit A port it would drive 12 (a net of ~100 pins across the group).
// The tap is sign-extended to 19 bits, which only the A port takes; the
// value, and so the product, is unchanged.
//
// Latency: a / b at cycle t -> M register at the end of t + 2; c at t -> C
// register at the end of t; P = C + M at the end of the next cycle.
// ---------------------------------------------------------------------------
module mm_mac (
  input  logic               clk,
  input  logic signed [15:0] a,
  input  logic signed [15:0] b,
  input  logic               b_ce,       // A2 enable (A1 loads every cycle)
  input  logic signed [31:0] c,
  input  logic               c_rst,      // CREG synchronous reset (RSTC)
  output logic signed [31:0] p
);
  (* use_dsp = "yes" *) logic signed [15:0] a1, a2;
  logic signed [18:0] b1, b2;
  logic signed [31:0] m_r, c_r, p_r;

  always_ff @(posedge clk) begin
    a1 <= a;
    a2 <= a1;
    b1 <= 19'(b);
    if (b_ce) b2 <= b1;
    m_r <= 32'(a2 * b2);
    if (c_rst) c_r <= '0;
    else       c_r <= c;
    p_r <= c_r + m_r;
  end

  assign p = p_r;

endmodule
