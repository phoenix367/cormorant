// dsp_check — the behavioural cv_mac_chain against the DSP48E2 primitive
// (Vivado's unisim simulation model), on the same stimulus.
module dsp_check
  import cv_pkg::*;
(
  input  logic                 clk,
  input  logic [TIC-1:0][15:0] a,
  input  logic [TIC-1:0][15:0] b,
  input  logic [31:0]          c,
  input  logic                 wfb,
  output logic [31:0]          p_beh,
  output logic [31:0]          p_prim
);
  glbl glbl ();
  cv_mac_chain #(.PRIM (1'b0)) u_beh  (.clk, .a, .b, .c, .wfb, .p (p_beh));
  cv_mac_chain #(.PRIM (1'b1)) u_prim (.clk, .a, .b, .c, .wfb, .p (p_prim));
endmodule
