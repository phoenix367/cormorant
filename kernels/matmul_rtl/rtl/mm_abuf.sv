// ---------------------------------------------------------------------------
// mm_abuf — the A panel buffer: one 128-bit x A_D block RAM per (lane, panel
// row).  RAM (h, r) holds the K elements of panel row r that lane h owns
// (its K blocks, packed back to back), eight per word.
//
// Write port p (one per read port / A writer) covers panel rows
// [p*RH, p*RH + RH) of both lanes; read port h reads word `raddr` of all R
// rows of lane h with a latency of two cycles (RAM + output register).
// ---------------------------------------------------------------------------
module mm_abuf
  import mm_pkg::*;
(
  input  logic                 clk,

  input  logic [1:0]           we,           // per write port
  input  logic [1:0][1:0]      wrow,         // row within the port's half
  input  logic [1:0]           wlane,
  input  logic [1:0][A_AW-1:0] waddr,
  input  logic [1:0][BW-1:0]   wdata,

  input  logic [1:0]           re,           // per lane
  input  logic [1:0][A_AW-1:0] raddr,
  output logic [1:0][R-1:0][BW-1:0] rdata
);

  for (genvar h = 0; h < 2; h++) begin : g_lane
    for (genvar r = 0; r < R; r++) begin : g_row
      localparam int PW = r / RH;           // write port owning this row
      (* ram_style = "block" *) logic [BW-1:0] mem [A_D];
      logic [BW-1:0] q1, q2;
      logic          w;
      assign w = we[PW] && (wlane[PW] == 1'(h)) && (wrow[PW] == 2'(r % RH));
      always_ff @(posedge clk) begin
        if (w)     mem[waddr[PW]] <= wdata[PW];
        if (re[h]) q1 <= mem[raddr[h]];
        q2 <= q1;
      end
      assign rdata[h][r] = q2;
    end
  end

endmodule
