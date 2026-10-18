// Copyright 2026 Ivan Gubochkin
// SPDX-License-Identifier: Apache-2.0

// ---------------------------------------------------------------------------
// ro_mul32 — an unsigned 32 x 32 -> 64 multiplier in three pipeline stages
// (four 16 x 16 DSP products, then two adds), for the NORM scalar unit:
// p is valid three cycles after a / b.
// ---------------------------------------------------------------------------
module ro_mul32 (
  input  logic        clk,
  input  logic [31:0] a,
  input  logic [31:0] b,
  output logic [63:0] p
);
  logic [31:0] ll, lh, hl, hh;
  logic [63:0] s_lo, s_hi;
  always_ff @(posedge clk) begin
    ll   <= a[15:0]  * b[15:0];
    lh   <= a[15:0]  * b[31:16];
    hl   <= a[31:16] * b[15:0];
    hh   <= a[31:16] * b[31:16];
    s_lo <= {hh, ll};
    s_hi <= 64'({32'b0, lh} + {32'b0, hl}) << 16;
    p    <= s_lo + s_hi;
  end
endmodule
