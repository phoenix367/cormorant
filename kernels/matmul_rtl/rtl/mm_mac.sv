// ---------------------------------------------------------------------------
// mm_mac — one DSP48E2 multiply-accumulate slice, P = C + A * B, with the
// accumulator kept outside (the lane's LUTRAM):
//
//   a  -> AREG (every cycle)          b -> BREG (held for a whole B row)
//   M  = AREG * BREG -> MREG          c -> CREG (accumulator read, or 0)
//   P  = CREG + MREG -> PREG          (32-bit, wraps like ap_fixed<32,16>)
// ---------------------------------------------------------------------------
module mm_mac (
  input  logic               clk,
  input  logic signed [15:0] a,
  input  logic               b_ce,
  input  logic signed [15:0] b,
  input  logic signed [31:0] c,
  output logic signed [31:0] p
);
  (* use_dsp = "yes" *) logic signed [15:0] a_r, b_r;
  logic signed [31:0] m_r, c_r, p_r;

  always_ff @(posedge clk) begin
    a_r <= a;
    if (b_ce) b_r <= b;
    m_r <= a_r * b_r;
    c_r <= c;
    p_r <= c_r + m_r;
  end

  assign p = p_r;

endmodule
