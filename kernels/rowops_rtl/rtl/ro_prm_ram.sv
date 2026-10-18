// Copyright 2026 Ivan Gubochkin
// SPDX-License-Identifier: Apache-2.0

// ---------------------------------------------------------------------------
// ro_prm_ram — one of ro_param's two UltraRAMs (4096 x 72: two parameter
// banks): port A writes during the load (keeping its output: the write mode
// an UltraRAM infers from) and reads otherwise, port B reads; reads answer
// two cycles later.
// ---------------------------------------------------------------------------
module ro_prm_ram
  import ro_pkg::*;
(
  input  logic             clk,
  input  logic             we,
  input  logic [11:0]      waddr,
  input  logic [PRM_W-1:0] wdata,
  input  logic [11:0]      addr_a,
  input  logic [11:0]      addr_b,
  output logic [PRM_W-1:0] q_a,
  output logic [PRM_W-1:0] q_b
);
  (* ram_style = "ultra" *) logic [PRM_W-1:0] mem [4096];
  logic [PRM_W-1:0] qa0, qb0;
  logic [11:0]      aa;
  assign aa = we ? waddr : addr_a;
  always_ff @(posedge clk) begin
    if (we) mem[aa] <= wdata;
    else    qa0 <= mem[aa];
    qb0 <= mem[addr_b];
    q_a <= qa0;
    q_b <= qb0;
  end
endmodule
