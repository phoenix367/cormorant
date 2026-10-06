// ---------------------------------------------------------------------------
// cv_mac_chain — one column of the MAC grid for one output pixel: a cascade
// of TIC DSP48E2s computing, per instant E,
//
//     P(E) = W + seed(E) + sum_l a_l(E) * b_l(E)        (48 bits, read as 32)
//
// DSP l takes lane l's operands l cycles late (the caller skews them), adds
// its product to the partial sum of DSP l-1 (PCIN) and passes it on (PCOUT):
// DSP 0 adds the seed on its C port instead, DSP TIC-1 adds its own P (W =
// P) unless `wfb` is 0 — so it accumulates over the instants of a kernel
// window and restarts at the window's first one.
//
// Timing (every DSP: AREG = BREG = MREG = PREG = 1; CREG = 1 on DSP 0,
// OPMODEREG = 1 on DSP TIC-1):
//   a[l], b[l]   presented at E + l
//   c            presented at E + 1
//   wfb          presented at E + TIC (the cycle DSP TIC-1's M register loads)
//   p            valid at E + TIC + 2 (the sum of instant E and, with wfb,
//                of the window's earlier instants)
// The sum wraps at 48 bits; its low 32 bits are the HLS kernel's
// ap_fixed<32,16> accumulator (wrap-around), whatever the order of the terms.
//
// PRIM = 1 builds the DSP48E2 primitive (Vivado synthesis and xsim), PRIM = 0
// an equivalent behavioural model (the Verilator testbench).  The default
// follows the tool; tb/verilator/dsp_check.cpp runs both on the same stimulus
// against the unisim DSP48E2 model.
// ---------------------------------------------------------------------------
`ifdef VERILATOR
 `ifndef CV_UNISIM_DSP
  `define CV_NO_PRIM
 `endif
`endif

module cv_mac_chain
  import cv_pkg::*;
#(
`ifdef CV_NO_PRIM
  parameter bit PRIM = 1'b0
`else
  parameter bit PRIM = 1'b1
`endif
) (
  input  logic                 clk,
  input  logic [TIC-1:0][15:0] a,        // lane l at E + l
  input  logic [TIC-1:0][15:0] b,
  input  logic [31:0]          c,        // seed, at E + 1
  input  logic                 wfb,      // accumulate (0 at a window's first instant), at E + TIC
  output logic [31:0]          p         // at E + TIC + 2
);
  logic [47:0] pc [TIC];                 // PCOUT of each DSP
  logic [47:0] p_last;

  for (genvar l = 0; l < TIC; l++) begin : g_dsp
    localparam bit FIRST = (l == 0);
    localparam bit LAST  = (l == TIC - 1);
    logic [47:0] pcin, pout;
    assign pcin = FIRST ? 48'd0 : pc[(l == 0) ? 0 : l - 1];

    if (!PRIM) begin : g_beh
    // behavioural model: the DSP48E2 configuration below
    logic signed [15:0] a2, b2;
    logic signed [31:0] m;
    logic        [31:0] c1;
    logic               wfb1;
    logic        [47:0] pr;
    always_ff @(posedge clk) begin
      a2 <= signed'(a[l]);
      b2 <= signed'(b[l]);
      m  <= a2 * b2;
      if (FIRST) c1   <= c;
      if (LAST)  wfb1 <= wfb;
      pr <= ((LAST && wfb1) ? pr : 48'd0) + 48'(m)
          + (FIRST ? 48'(signed'(c1)) : pcin);
    end
    assign pout  = pr;
    assign pc[l] = pr;
    end else begin : g_prim
`ifndef CV_NO_PRIM
    // OPMODE: X = Y = M; Z = C (DSP 0) or PCIN; W = P (DSP TIC-1 accumulating) or 0
    logic [8:0] opmode;
    assign opmode = {(LAST && wfb) ? 2'b01 : 2'b00, FIRST ? 3'b011 : 3'b001, 2'b01, 2'b01};
    DSP48E2 #(
      .AMULTSEL ("A"), .BMULTSEL ("B"), .A_INPUT ("DIRECT"), .B_INPUT ("DIRECT"),
      .PREADDINSEL ("A"), .USE_MULT ("MULTIPLY"), .USE_SIMD ("ONE48"),
      .USE_PATTERN_DETECT ("NO_PATDET"), .AUTORESET_PATDET ("NO_RESET"),
      .AREG (1), .ACASCREG (1), .BREG (1), .BCASCREG (1), .ADREG (0), .DREG (0),
      .MREG (1), .PREG (1), .CREG (FIRST ? 1 : 0), .OPMODEREG (1), .ALUMODEREG (1),
      .INMODEREG (1), .CARRYINREG (1), .CARRYINSELREG (1)
    ) u_dsp (
      .CLK (clk),
      .A ({{14{a[l][15]}}, a[l]}), .B ({{2{b[l][15]}}, b[l]}),
      .C (FIRST ? {{16{c[31]}}, c} : 48'd0), .D (27'd0), .PCIN (pcin),
      .ACIN (30'd0), .BCIN (18'd0), .CARRYCASCIN (1'b0), .MULTSIGNIN (1'b0),
      .OPMODE (opmode), .ALUMODE (4'b0000), .INMODE (5'b00000),
      .CARRYINSEL (3'b000), .CARRYIN (1'b0),
      .CEA1 (1'b0), .CEA2 (1'b1), .CEAD (1'b0), .CEALUMODE (1'b1), .CEB1 (1'b0), .CEB2 (1'b1),
      .CEC (1'b1), .CECARRYIN (1'b1), .CECTRL (1'b1), .CED (1'b0), .CEINMODE (1'b1),
      .CEM (1'b1), .CEP (1'b1),
      .RSTA (1'b0), .RSTALLCARRYIN (1'b0), .RSTALUMODE (1'b0), .RSTB (1'b0), .RSTC (1'b0),
      .RSTCTRL (1'b0), .RSTD (1'b0), .RSTINMODE (1'b0), .RSTM (1'b0), .RSTP (1'b0),
      .P (pout), .PCOUT (pc[l]), .ACOUT (), .BCOUT (), .CARRYCASCOUT (), .CARRYOUT (),
      .MULTSIGNOUT (), .OVERFLOW (), .PATTERNBDETECT (), .PATTERNDETECT (), .UNDERFLOW (),
      .XOROUT ()
    );
`else
    assign pout  = '0;
    assign pc[l] = '0;
`endif
    end
    if (LAST) begin : g_out
      assign p_last = pout;
    end
  end

  assign p = p_last[31:0];

endmodule
