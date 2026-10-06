// ---------------------------------------------------------------------------
// cv_ywriter — the output writer (gmem3, the HLS write_output_tile).
//
// Each run (channel, segment) of len pixels starts at element e of y: its
// words (8 pixels each, from the drain) are shifted onto DDR words — word k
// of the run is lanes [8k, 8k + 8) of the shifted run — and written as one
// burst (two where the run crosses 4 KiB) with byte strobes on the run's own
// lanes only, so a neighbouring channel's lanes in the first and last words
// stay intact.  At most Y_OUTS bursts await their B.
// ---------------------------------------------------------------------------
module cv_ywriter
  import cv_pkg::*;
(
  input  logic          clk,
  input  logic          rst,
  input  job_t          j,

  input  logic          run_valid,
  output logic          run_ready,
  input  logic [31:0]   run_e,
  input  logic [8:0]    run_len,
  input  logic          yw_valid,
  output logic          yw_ready,
  input  logic [BW-1:0] yw_data,

  output logic          awvalid,
  input  logic          awready,
  output logic [63:0]   awaddr,
  output logic [7:0]    awlen,
  output logic          wvalid,
  input  logic          wready,
  output logic [BW-1:0] wdata,
  output logic [E*2-1:0] wstrb,
  output logic          wlast,
  input  logic          bvalid,
  output logic          bready,

  output logic          idle
);
  localparam int WD = 32;                  // W FIFO depth

  // a run (<= DSEG pixels at any alignment) is one burst unless it crosses 4 KiB
  if (Y_BURST < DSEG / E + 1) begin : g_burst_check
    $error("cv_ywriter: Y_BURST must hold a run of DSEG / E + 1 words");
  end
  localparam int OW = $clog2(Y_OUTS) + 1;

  logic        r_act;
  logic [2:0]  r_sh;                       // lanes before the run in its first word
  logic [8:0]  r_len;
  logic [5:0]  r_nin, r_nout, r_k;         // input words, output words, next output word
  logic [59:0] r_waddr;                    // word address of output word r_k
  logic [5:0]  r_brem;                     // words left in the current burst
  logic [BW-1:0] r_prev;

  logic [5:0]  wf_n;
  logic [3:0]  aw_n;
  logic [OW-1:0] outs;
  logic        wf_in_ready, aw_in_ready;

  // the burst starting at word r_k: up to the 4 KiB boundary (runs are <= 33 words)
  logic [8:0]  to4k;
  logic [5:0]  b_len;
  assign to4k  = 9'd256 - {1'b0, r_waddr[7:0]};
  assign b_len = ((9'(r_nout) - 9'(r_k)) > to4k) ? 6'(to4k) : (r_nout - r_k);

  logic go, need_in, bstart;
  assign need_in = (r_k < r_nin);
  assign bstart  = (r_brem == 6'd0);
  // the word goes through a register (p_v / p_d) into the W FIFO: room for it too
  logic p_v;
  assign go = r_act && (!need_in || yw_valid) && (32'(wf_n) + 32'(p_v) < WD - 1) &&
              (!bstart || ((outs != OW'(Y_OUTS)) && aw_in_ready));
  assign yw_ready  = go && need_in;
  assign run_ready = !r_act;

  logic [BW-1:0] in_w, out_w;
  logic [E*2-1:0] strb;
  always_comb begin
    in_w  = need_in ? yw_data : '0;
    out_w = BW'({in_w, r_prev} >> (EW * (E - int'(r_sh))));
    for (int l = 0; l < E; l++) begin
      logic [9:0] pos;
      logic       ok;
      pos = 10'({r_k, 3'(l)});
      ok  = (pos >= {7'b0, r_sh}) && (pos < {7'b0, r_sh} + {1'b0, r_len});
      strb[2 * l +: 2] = {ok, ok};
      if (!ok) out_w[l * EW +: EW] = '0;
    end
  end

  always_ff @(posedge clk) begin
    if (rst) begin
      r_act <= 1'b0;
    end else if (!r_act) begin
      if (run_valid) begin
        r_act   <= 1'b1;
        r_sh    <= run_e[2:0];
        r_len   <= run_len;
        r_nin   <= 6'((run_len + 9'd7) >> 3);
        r_nout  <= 6'((9'(run_e[2:0]) + run_len + 9'd7) >> 3);
        r_k     <= '0;
        r_waddr <= j.y_w + 60'(run_e[31:3]);
        r_brem  <= '0;
        r_prev  <= '0;
      end
    end else if (go) begin
      r_k     <= r_k + 6'd1;
      r_waddr <= r_waddr + 60'd1;
      r_prev  <= in_w;
      r_brem  <= (bstart ? b_len : r_brem) - 6'd1;
      if (r_k + 6'd1 == r_nout) r_act <= 1'b0;
    end
  end

  // AW and W queues
  logic        wlast_in;
  logic [BW+E*2:0] p_d;
  assign wlast_in = (bstart ? b_len : r_brem) == 6'd1;
  always_ff @(posedge clk) begin
    p_v <= go && !rst;
    p_d <= {wlast_in, strb, out_w};
  end
  cv_fifo #(.W(BW + E * 2 + 1), .D(WD), .BRAM(1'b0)) u_wf (
    .clk, .rst, .in_valid (p_v), .in_ready (wf_in_ready), .in_data (p_d),
    .out_valid (wvalid), .out_ready (wready), .out_data ({wlast, wstrb, wdata}), .count (wf_n)
  );
  logic [67:0] aw_q;
  cv_fifo #(.W(68), .D(8), .BRAM(1'b0)) u_aw (
    .clk, .rst, .in_valid (go && bstart), .in_ready (aw_in_ready),
    .in_data ({r_waddr, 2'b0, b_len}), .out_valid (awvalid), .out_ready (awready),
    .out_data (aw_q), .count (aw_n)
  );
  assign awaddr = {aw_q[67:8], 4'b0};
  assign awlen  = {2'b0, 6'(aw_q[5:0] - 6'd1)};

  assign bready = 1'b1;
  always_ff @(posedge clk) begin
    if (rst) outs <= '0;
    else     outs <= outs + ((go && bstart) ? OW'(1) : '0) - (bvalid ? OW'(1) : '0);
  end

  assign idle = !r_act && !run_valid && !p_v && !wvalid && !awvalid && (outs == '0);

  logic unused;
  assign unused = wf_in_ready ^ ^aw_n ^ ^j ^ aw_q[7] ^ aw_q[6];

endmodule
