// Copyright 2026 Ivan Gubochkin
// SPDX-License-Identifier: Apache-2.0

// ---------------------------------------------------------------------------
// ro_rd_arb — the two read ports (stream A, stream B) on the one AXI read
// channel of m_axi_gmem0: AR requests are merged round robin and tagged with
// the port as ARID (0 = A, 1 = B); R beats are routed back by RID.  Each
// port's bursts stay in order (same-ID ordering), so the ports' credit FIFOs
// work as on their own channels.  AR is registered through a slice (ARREADY
// only enables it); R fans out combinationally — the ports register R
// themselves before their FIFOs.
// ---------------------------------------------------------------------------
module ro_rd_arb
  import ro_pkg::*;
(
  input  logic          clk,
  input  logic          rst,

  // port A
  input  logic          a_arvalid,
  output logic          a_arready,
  input  logic [63:0]   a_araddr,
  input  logic [7:0]    a_arlen,
  output logic          a_rvalid,
  input  logic          a_rready,
  // port B
  input  logic          b_arvalid,
  output logic          b_arready,
  input  logic [63:0]   b_araddr,
  input  logic [7:0]    b_arlen,
  output logic          b_rvalid,
  input  logic          b_rready,

  // AXI
  output logic          arvalid,
  input  logic          arready,
  output logic [63:0]   araddr,
  output logic [7:0]    arlen,
  output logic          arid,
  input  logic          rvalid,
  output logic          rready,
  input  logic          rid
);
  // AR: round robin between the two ports (the last served loses a tie).
  logic last_b, pick_b, in_valid, in_ready;
  assign pick_b    = b_arvalid && (!a_arvalid || last_b == 1'b0);
  assign in_valid  = a_arvalid || b_arvalid;
  assign a_arready = in_ready && in_valid && !pick_b;
  assign b_arready = in_ready && in_valid && pick_b;

  always_ff @(posedge clk) begin
    if (rst) last_b <= 1'b1;
    else if (in_valid && in_ready) last_b <= pick_b;
  end

  ro_rs #(.W(73)) u_ar (
    .clk, .rst,
    .in_valid, .in_ready,
    .in_data   (pick_b ? {1'b1, b_araddr, b_arlen} : {1'b0, a_araddr, a_arlen}),
    .out_valid (arvalid), .out_ready (arready), .out_data ({arid, araddr, arlen}),
    .head_data ()
  );

  // R: by ID.  The ports keep RREADY high (credit FIFOs), so rready is their AND.
  assign a_rvalid = rvalid && (rid == 1'b0);
  assign b_rvalid = rvalid && (rid == 1'b1);
  assign rready   = a_rready && b_rready;

endmodule
