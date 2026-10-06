// ---------------------------------------------------------------------------
// pl_rs — register slice (two-entry skid buffer), valid/ready on both sides.
//
// Both directions are registered: out_valid / out_data come straight from
// the main register and in_ready is a register (no skid entry held), so the
// consumer's ready never reaches the producer's logic in the same cycle.  It
// sustains one transfer per cycle.  Used on the m_axi AR / AW / W outputs
// (the xREADY inputs then only enable this slice's registers).
// ---------------------------------------------------------------------------
module pl_rs #(
  parameter int W = 8
) (
  input  logic         clk,
  input  logic         rst,

  input  logic         in_valid,
  output logic         in_ready,
  input  logic [W-1:0] in_data,

  output logic         out_valid,
  input  logic         out_ready,
  output logic [W-1:0] out_data
);
  logic         m_v, s_v;
  logic [W-1:0] s_d;
  logic         m_take;               // the main register takes a word this cycle

  assign in_ready  = !s_v;
  assign out_valid = m_v;
  assign m_take    = out_ready || !m_v;

  always_ff @(posedge clk) begin
    if (rst) begin
      m_v <= 1'b0;
      s_v <= 1'b0;
    end else if (m_take) begin
      m_v <= s_v || in_valid;
      s_v <= 1'b0;
    end else if (in_valid && !s_v) begin
      s_v <= 1'b1;
    end
  end

  // Data registers: no reset (the valid bits qualify them).
  always_ff @(posedge clk) begin
    if (m_take) out_data <= s_v ? s_d : in_data;
    if (!m_take && !s_v) s_d <= in_data;
  end

endmodule
