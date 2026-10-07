// ---------------------------------------------------------------------------
// vo_compute — the element-wise op on the a / b word streams, 8 lanes per
// cycle (a 6-stage pipeline: input registers, then add / sub / DSP multiply,
// product register, op select + saturation, ReLU / ReLU6 activation), and
// OP_DIV one lane per cycle through vo_div (as the HLS kernel); then both
// through vo_act (LeakyReLU / SiLU / GELU / GELU tanh, 7 cycles).  Unary ops
// (op >= 4, the activation ops 6..9 among them: the pass op plus their
// activation) take no b words.  Results go to a small FIFO; a word enters a
// pipeline only when the FIFO has room for it and for every word already in
// flight.
// ---------------------------------------------------------------------------
module vo_compute
  import vo_pkg::*;
(
  input  logic          clk,
  input  logic          rst,
  input  logic [31:0]   op,                 // job constants
  input  logic [31:0]   act,
  input  logic [31:0]   alpha,

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

  // Decoded job constants (op / act do not change during a job).  The job's
  // activation (VectorOP.h job_act): an activation op's own (op - 3), else the
  // act register's (codes past ACT_GELU_TANH: none); ReLU / ReLU6 are applied
  // in the ALU's stage 5 and by vo_div, the others by vo_act.
  logic       binary, is_div;
  logic [2:0] sel;                          // 0 add 1 sub 2 mul 3 relu 4 relu6 5 pass
  logic       actop;
  logic [2:0] op3, act3, jact;
  logic [1:0] actc;
  logic [2:0] fnc;                          // vo_act: 0 none, 3..6
  always_ff @(posedge clk) begin
    binary <= (op < OP_RELU);
    is_div <= (op == OP_DIV);
    sel    <= (op == OP_ADD) ? 3'd0 : (op == OP_SUB) ? 3'd1 : (op == OP_MUL) ? 3'd2 :
              (op == OP_RELU) ? 3'd3 : (op == OP_RELU6) ? 3'd4 : 3'd5;
    actop  <= (op >= OP_LEAKY_RELU) && (op <= OP_GELU_TANH);
    op3    <= op[2:0];
    act3   <= (act <= ACT_GELU_TANH) ? act[2:0] : 3'd0;
    actc   <= (jact == 3'(ACT_RELU)) ? 2'd1 : (jact == 3'(ACT_RELU6)) ? 2'd2 : 2'd0;
    fnc    <= (jact >= 3'(ACT_LEAKY_RELU)) ? jact : 3'd0;
  end
  assign jact = actop ? op3 - 3'(OP_LEAKY_RELU - ACT_LEAKY_RELU) : act3;   // 6..9 -> 3..6 (mod 8)

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
  logic [FW:0] inflight;                    // words taken (ALU or DIV), not yet in the FIFO
  logic       take;
  logic room;
  assign room = (7'(cf_count) + 7'(inflight)) < 7'(CF_D);

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
  always_ff @(posedge clk)
    if (q_valid && q_lane != 3'd7) d_acc[EW*q_lane +: EW] <= q;

  // Results -> vo_act -> FIFO --------------------------------------------------------------
  logic [BW-1:0] o5w;
  always_comb for (int l = 0; l < E; l++) o5w[EW*l +: EW] = o5[l];

  vo_act u_act (
    .clk, .rst, .fn (fnc), .alpha (alpha[15:0]),
    .in_valid  (v[5] || d_push),             // one job is either ALU or DIV
    .in_data   (d_push ? {q, d_acc} : o5w),
    .out_valid (cf_in_valid), .out_data (cf_in_data)
  );

  assign take = alu_take || div_take;
  always_ff @(posedge clk) begin
    if (rst) inflight <= '0;
    else     inflight <= inflight + (FW+1)'(take) - (FW+1)'(cf_in_valid);
  end

  assign a_ready = alu_take || div_take;
  assign b_ready = (alu_take && binary) || div_take;

  assign idle = (inflight == '0) && !d_busy && !c_valid;

  logic unused;
  assign unused = ^{cf_in_ready, alpha[31:16]};

endmodule
