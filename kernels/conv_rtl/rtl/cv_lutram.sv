// ---------------------------------------------------------------------------
// cv_lutram — distributed RAM, one synchronous write port and one
// asynchronous read port (simple dual port).
// (The x loader's row buffer and the engine's patch replay file are banks of these.)
// ---------------------------------------------------------------------------
module cv_lutram #(
  parameter int W = 16,
  parameter int D = 128
) (
  input  logic                 clk,
  input  logic                 we,
  input  logic [$clog2(D)-1:0] waddr,
  input  logic [W-1:0]         wdata,
  input  logic [$clog2(D)-1:0] raddr,
  output logic [W-1:0]         rdata
);
  (* ram_style = "distributed" *) logic [W-1:0] mem [D];

  always_ff @(posedge clk)
    if (we) mem[waddr] <= wdata;

  assign rdata = mem[raddr];

endmodule
