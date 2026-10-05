// ---------------------------------------------------------------------------
// pl_loader — the DDR reader (gmem0): every input row a chunk needs, as one
// (row, channel) run of words each, in the HLS row_loader's order (rows
// outer, channels inner) and word ranges.
//
//   chunks ─► run generator ─► run FIFO ─► bursts (<= RD_BURST, split at 4 KiB)
//                                            │ AR                 │ run descriptors
//                                            ▼                    ▼
//                                   R ─► word FIFO ───────────► emitter (line buffer)
//
// A burst's AR is only issued when the word FIFO can take all of its beats, so
// RREADY stays high; at most RD_OUTS bursts await their data.  A run's
// descriptor (slot, channel, lane shift, word count) is queued with its first
// burst; the emitter consumes the descriptors and the words in the same order.
// ---------------------------------------------------------------------------
module pl_loader
  import pl_pkg::*;
(
  input  logic          clk,
  input  logic          rst,
  input  job_t          j,

  input  logic          cq_valid,
  output logic          cq_ready,
  input  chunk_t        cq,

  output logic          arvalid,
  input  logic          arready,
  output logic [63:0]   araddr,
  output logic [7:0]    arlen,
  input  logic          rvalid,
  output logic          rready,
  input  logic [BW-1:0] rdata,
  input  logic          rlast,

  output logic          dq_valid,
  input  logic          dq_ready,
  output lbd_t          dq_data,
  output logic          wq_valid,
  input  logic          wq_ready,
  output logic [BW-1:0] wq_data,

  output logic          idle
);
  // Run generator -------------------------------------------------------------------
  // run_off = base_in + ih * in_w + c * in_hw, kept incrementally.
  logic        g_act;
  logic [31:0] g_ih, g_rows, g_row_off, g_run_off;
  logic [2:0]  g_c, g_clast;
  logic [6:0]  g_len;

  typedef struct packed {
    logic [59:0] waddr;
    lbd_t        d;
  } run_t;

  logic  rq_in_valid, rq_in_ready, rq_valid, rq_ready;
  run_t  rq_in, rq;
  logic [2:0] rq_count;

  logic [3:0]  g_nw;
  logic [9:0]  g_sum;
  assign g_sum = 10'(g_run_off[2:0]) + 10'(g_len) + 10'd7;
  assign g_nw  = g_sum[6:3];

  assign rq_in_valid = g_act;
  assign rq_in.waddr = j.x_w + 60'(g_run_off[31:3]);
  assign rq_in.d     = '{slot: g_ih[3:0], ch: g_c, shift: g_run_off[2:0], nw: g_nw,
                         row_last: (g_c == g_clast)};
  assign cq_ready    = !g_act;

  always_ff @(posedge clk) begin
    if (rst) begin
      g_act <= 1'b0;
    end else if (cq_valid && cq_ready) begin
      g_act     <= (j.rows != '0);
      g_ih      <= '0;
      g_c       <= '0;
      g_rows    <= j.rows;
      g_clast   <= 3'(cq.c_valid - 4'd1);
      g_len     <= cq.run_len;
      g_row_off <= cq.base_in;
      g_run_off <= cq.base_in;
    end else if (rq_in_valid && rq_in_ready) begin
      if (g_c == g_clast) begin
        g_c       <= '0;
        g_ih      <= g_ih + 32'd1;
        g_row_off <= g_row_off + j.in_w;
        g_run_off <= g_row_off + j.in_w;
        if (g_ih + 32'd1 == g_rows) g_act <= 1'b0;
      end else begin
        g_c       <= g_c + 3'd1;
        g_run_off <= g_run_off + j.in_hw;
      end
    end
  end

  pl_fifo #(.W($bits(run_t)), .D(4), .BRAM(1'b0)) u_rq (
    .clk, .rst,
    .in_valid (rq_in_valid), .in_ready (rq_in_ready), .in_data (rq_in),
    .out_valid(rq_valid),    .out_ready(rq_ready),    .out_data(rq),
    .count    (rq_count)
  );

  // a run (<= LBC columns) is at most LBW + 1 words: one burst unless it crosses 4 KiB
  if (RD_BURST < LBW + 1) begin : g_burst_check
    $error("pl_loader: RD_BURST must hold a run of LBW + 1 words");
  end

  // Bursts and AR ---------------------------------------------------------------------
  // A run of nw words from waddr: one burst, or two where it crosses 4 KiB.
  // The run FIFO's head is registered with its bursts worked out, so the
  // issue decision reads registers only.
  localparam int SW = $clog2(RD_FIFO_D) + 1;
  localparam int OW = $clog2(RD_OUTS) + 1;

  logic [SW-1:0] space;            // word-FIFO entries not reserved by a burst
  logic [OW-1:0] outs;             // bursts issued, last beat not yet received
  logic          second;           // the run's second burst is next
  logic          h_v, h_split, h_take;
  logic [59:0]   h_waddr, h_waddr2;
  logic [3:0]    h_len1, h_len2;
  lbd_t          h_d;
  logic [59:0]   b_waddr;
  logic [3:0]    b_len;
  logic          b_lastb;          // the run's last burst
  logic          dq_in_ready, issue, wf_pop;

  assign rq_ready = !h_v || h_take;

  always_ff @(posedge clk) begin
    if (rst) begin
      h_v <= 1'b0;
    end else if (rq_ready) begin
      h_v <= rq_valid;
    end
    if (rq_ready && rq_valid) begin
      logic [8:0] to_4k;
      to_4k    = 9'd256 - {1'b0, rq.waddr[7:0]};
      h_waddr  <= rq.waddr;
      h_waddr2 <= {rq.waddr[59:8] + 52'd1, 8'd0};      // = waddr + to_4k
      h_split  <= (9'(rq.d.nw) > to_4k);
      h_len1   <= (9'(rq.d.nw) > to_4k) ? to_4k[3:0] : rq.d.nw;
      h_len2   <= rq.d.nw - to_4k[3:0];
      h_d      <= rq.d;
    end
  end

  assign b_waddr = second ? h_waddr2 : h_waddr;
  assign b_len   = second ? h_len2 : h_len1;
  assign b_lastb = second || !h_split;
  assign issue   = h_v && (!arvalid || arready) && (space >= SW'(b_len)) &&
                   (outs != OW'(RD_OUTS)) && (second || dq_in_ready);
  assign h_take  = issue && b_lastb;

  always_ff @(posedge clk) begin
    if (rst) begin
      arvalid <= 1'b0;
      second  <= 1'b0;
      space   <= SW'(RD_FIFO_D);
      outs    <= '0;
    end else begin
      if (arvalid && arready) arvalid <= 1'b0;
      if (issue) begin
        arvalid <= 1'b1;
        araddr  <= {b_waddr, 4'b0};
        arlen   <= 8'(4'(b_len - 4'd1));
        second  <= !b_lastb;
      end
      space <= space - (issue ? SW'(b_len) : '0) + (wf_pop ? SW'(1) : '0);
      outs  <= outs + (issue ? OW'(1) : '0) - ((rvalid && rlast) ? OW'(1) : '0);
    end
  end

  // Run descriptors and words to the emitter ---------------------------------------------
  logic [6:0] dq_count;
  pl_fifo #(.W($bits(lbd_t)), .D(64), .BRAM(1'b0)) u_dq (
    .clk, .rst,
    .in_valid (issue && !second), .in_ready (dq_in_ready), .in_data (h_d),
    .out_valid(dq_valid),         .out_ready(dq_ready),    .out_data(dq_data),
    .count    (dq_count)
  );

  assign rready = 1'b1;
  logic          wf_in_ready;
  logic [SW-1:0] wf_count;
  pl_fifo #(.W(BW), .D(RD_FIFO_D), .BRAM(1'b1)) u_wf (
    .clk, .rst,
    .in_valid (rvalid),   .in_ready (wf_in_ready), .in_data (rdata),
    .out_valid(wq_valid), .out_ready(wq_ready),    .out_data(wq_data),
    .count    (wf_count)
  );
  assign wf_pop = wq_valid && wq_ready;

  assign idle = !g_act && !rq_valid && !h_v && !arvalid && (outs == '0) &&
                (space == SW'(RD_FIFO_D)) && !dq_valid;

  // The word FIFO never fills (every burst reserved its space).
  logic unused;
  assign unused = wf_in_ready ^ ^wf_count ^ ^dq_count ^ ^rq_count;

endmodule
