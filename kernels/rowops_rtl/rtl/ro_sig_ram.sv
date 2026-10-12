// ---------------------------------------------------------------------------
// ro_sig_ram — the sigmoid table of the MULACT op (ROWOPS_RTL_KERNEL.md §2):
// entry i = {D[i], T[i]} with T[i] = round(sigma(i / 2^8) * 2^24) and
// D[i] = T[i + 1] - T[i] (rowops_spec.sigmoid_table), one read giving both
// ends of the interpolation.  An UltraRAM, loaded by OP_TLOAD (an UltraRAM
// has no initial contents): port A writes during the load and reads
// otherwise, port B reads.  Reads answer two cycles later.
// ---------------------------------------------------------------------------
module ro_sig_ram
  import ro_pkg::*;
(
  input  logic             clk,
  input  logic             we,
  input  logic [11:0]      waddr,
  input  logic [SIG_W-1:0] wdata,
  input  logic [11:0]      addr_a,
  input  logic [11:0]      addr_b,
  output logic [SIG_W-1:0] q_a,
  output logic [SIG_W-1:0] q_b
);
  (* ram_style = "ultra" *) logic [SIG_W-1:0] mem [4096];
  logic [SIG_W-1:0] qa0, qb0;
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
