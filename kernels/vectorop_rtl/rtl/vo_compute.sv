// ---------------------------------------------------------------------------
// vo_compute — the element-wise op on the a / b word streams, 8 lanes per
// cycle (a 6-stage pipeline: input registers, then add / sub / DSP multiply,
// product register, op select + saturation, activation), and OP_DIV one lane
// per cycle through vo_div (as the HLS kernel).  Unary ops (op >= 4) take no
// b words.  Results go to a small FIFO; a word enters a pipeline only when
// the FIFO has room for it and for every word already in flight.
// ---------------------------------------------------------------------------
module vo_compute
  import vo_pkg::*;
(
  input  logic          clk,
  input  logic          rst,
  input  logic [31:0]   op,                 // job constants
  input  logic [31:0]   act,

  input  logic          a_valid,
  output logic          a_ready,
  input  logic [BW-1:0] a_data,
  input  logic          b_valid,
  output logic          b_ready,
  input  logic [BW-1:0] b_data,

  output logic          c_valid,
  input  logic          c_ready,
  output logic [BW-1:0] c_data,

  output logic          idle
);
  localparam int FW = $clog2(CF_D);

  // Decoded job constants (op / act do not change during a job).
  logic       binary, is_div;
  logic [2:0] sel;                          // 0 add 1 sub 2 mul 3 relu 4 relu6 5 pass
  logic [1:0] actc;
  always_ff @(posedge clk) begin
    binary <= (op < OP_RELU);
    is_div <= (op == OP_DIV);
    sel    <= (op == OP_ADD) ? 3'd0 : (op == OP_SUB) ? 3'd1 : (op == OP_MUL) ? 3'd2 :
              (op == OP_RELU) ? 3'd3 : (op == OP_RELU6) ? 3'd4 : 3'd5;
    actc   <= (act == ACT_RELU) ? 2'd1 : (act == ACT_RELU6) ? 2'd2 : 2'd0;
  end

  // Output FIFO and credits ----------------------------------------------------------
  logic          cf_in_valid, cf_in_ready;
  logic [BW-1:0] cf_in_data;
  logic [FW:0]   cf_count;
  vo_fifo #(.W(BW), .D(CF_D), .BRAM(1'b0)) u_cf (
    .clk, .rst,
    .in_valid (cf_in_valid), .in_ready (cf_in_ready), .in_data (cf_in_data),
    .out_valid(c_valid),     .out_ready(c_ready),     .out_data(c_data),
    .count    (cf_count)
  );

  logic [5:0] v;                            // valid bits of ALU stages 1..5 (v[0] unused)
  logic [3:0] dw_inflight;                  // DIV words taken, not yet pushed
  logic [5:0] alu_inflight;
  always_comb begin
    alu_inflight = '0;
    for (int s = 1; s <= 5; s++) alu_inflight += 6'(v[s]);
  end
  logic room;
  assign room = (7'(cf_count) + 7'(alu_inflight) + 7'(dw_inflight)) < 7'(CF_D);

  // ALU pipeline -----------------------------------------------------------------------
  logic alu_take;
  assign alu_take = !is_div && a_valid && (b_valid || !binary) && room;

  logic [BW-1:0] a1, b1;
  logic [15:0]   sum2 [E], dif2 [E], a2 [E], a3 [E], sum3 [E], dif3 [E], r4 [E], o5 [E];
  logic signed [31:0] p2 [E], p3 [E];

  always_ff @(posedge clk) begin
    if (rst) v <= '0;
    else     v <= {v[4:1], alu_take, 1'b0};
    a1 <= a_data;
    b1 <= binary ? b_data : '0;
  end

  for (genvar l = 0; l < E; l++) begin : g_lane
    logic signed [15:0] la, lb;
    assign la = a1[EW*l +: EW];
    assign lb = b1[EW*l +: EW];
    always_ff @(posedge clk) begin
      // stage 2
      sum2[l] <= sat17(17'(la) + 17'(lb));
      dif2[l] <= sat17(17'(la) - 17'(lb));
      p2[l]   <= la * lb;
      a2[l]   <= la;
      // stage 3
      p3[l]   <= p2[l];
      sum3[l] <= sum2[l];
      dif3[l] <= dif2[l];
      a3[l]   <= a2[l];
      // stage 4: op select (MUL: floor to Q8.8, then saturate)
      case (sel)
        3'd0: r4[l] <= sum3[l];
        3'd1: r4[l] <= dif3[l];
        3'd2: r4[l] <= (p3[l][31:23] == {9{p3[l][31]}}) ? p3[l][23:8]
                       : (p3[l][31] ? 16'h8000 : 16'h7FFF);
        3'd3: r4[l] <= relu(a3[l]);
        3'd4: r4[l] <= relu6(a3[l]);
        default: r4[l] <= a3[l];
      endcase
      // stage 5: activation
      o5[l]   <= activate(r4[l], actc);
    end
  end

  // DIV: one lane per cycle --------------------------------------------------------------
  logic          d_busy;
  logic [2:0]    d_li;
  logic [BW-1:0] d_a, d_b;
  logic          div_take;
  assign div_take = is_div && a_valid && b_valid && room && (!d_busy || d_li == 3'd7);

  always_ff @(posedge clk) begin
    if (rst) begin
      d_busy <= 1'b0;
      d_li   <= '0;
    end else if (div_take) begin
      d_busy <= 1'b1;
      d_li   <= '0;
    end else if (d_busy) begin
      d_li <= d_li + 3'd1;
      if (d_li == 3'd7) d_busy <= 1'b0;
    end
  end

  always_ff @(posedge clk)
    if (div_take) begin
      d_a <= a_data;
      d_b <= b_data;
    end

  logic        q_valid;
  logic [15:0] q;
  logic [2:0]  q_lane;
  vo_div u_div (
    .clk, .rst, .act (actc),
    .in_valid (d_busy), .in_a (d_a[EW*d_li +: EW]), .in_b (d_b[EW*d_li +: EW]), .in_lane (d_li),
    .out_valid (q_valid), .out_q (q), .out_lane (q_lane)
  );

  logic [BW-EW-1:0] d_acc;                  // lanes 0..6 of the word being assembled
  logic             d_push;
  assign d_push = q_valid && (q_lane == 3'd7);
  always_ff @(posedge clk) begin
    if (q_valid && q_lane != 3'd7) d_acc[EW*q_lane +: EW] <= q;
    if (rst) dw_inflight <= '0;
    else     dw_inflight <= dw_inflight + 4'(div_take) - 4'(d_push);
  end

  // Results -> FIFO ----------------------------------------------------------------------
  logic [BW-1:0] o5w;
  always_comb for (int l = 0; l < E; l++) o5w[EW*l +: EW] = o5[l];

  assign cf_in_valid = v[5] || d_push;
  assign cf_in_data  = d_push ? {q, d_acc} : o5w;

  assign a_ready = alu_take || div_take;
  assign b_ready = (alu_take && binary) || div_take;

  assign idle = (v == '0) && !d_busy && (dw_inflight == '0) && !c_valid;

  logic unused;
  assign unused = cf_in_ready;

endmodule
