// ---------------------------------------------------------------------------
// ro_rsqrt_ram — the reciprocal square root table of the NORM op
// (ROWOPS_RTL_KERNEL.md §2): entry i = round(2^47 / sqrt((i + 0.5) * 2^19))
// for i = 2048 .. 8191 (rowops_spec.rsqrt_table; the entries below 2048 are
// never read).  An UltraRAM of 4096 x 64 holding two entries per word,
// loaded by OP_TLOAD (an UltraRAM has no initial contents): the one port
// writes during the load and reads otherwise, answering two cycles later.
// ---------------------------------------------------------------------------
module ro_rsqrt_ram (
  input  logic        clk,
  input  logic        we,
  input  logic [11:0] waddr,
  input  logic [63:0] wdata,
  input  logic [12:0] addr,
  output logic [31:0] q
);
  (* ram_style = "ultra" *) logic [63:0] mem [4096];
  logic [63:0] q0, q1;
  logic        s0, s1;
  logic [11:0] aa;
  assign aa = we ? waddr : addr[12:1];
  always_ff @(posedge clk) begin
    if (we) mem[aa] <= wdata;
    else    q0 <= mem[aa];
    s0 <= addr[0];
    q1 <= q0;
    s1 <= s0;
  end
  assign q = s1 ? q1[63:32] : q1[31:0];
endmodule
