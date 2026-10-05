// ---------------------------------------------------------------------------
// vo_wr_port — the output c on gmem2: word stream -> DDR runs.
//
// Words are buffered in a block-RAM FIFO; the output geometry's bursts (at
// most WR_BURST words, never crossing 4 KiB or a run) are issued on AW once
// all of a burst's words are buffered, so W streams without gaps; at most
// WR_OUTS bursts await their B.  Every beat is written whole (the tail lanes
// of a run's last word carry op(0, 0) = 0, the alignment contract).
// `idle` means every burst has been written and acknowledged.
// ---------------------------------------------------------------------------
module vo_wr_port
  import vo_pkg::*;
(
  input  logic            clk,
  input  logic            rst,
  input  logic            start,
  input  geom_t           g,

  input  logic            in_valid,
  output logic            in_ready,
  input  logic [BW-1:0]   in_data,

  output logic            awvalid,
  input  logic            awready,
  output logic [63:0]     awaddr,
  output logic [7:0]      awlen,
  output logic            wvalid,
  input  logic            wready,
  output logic [BW-1:0]   wdata,
  output logic            wlast,
  input  logic            bvalid,
  output logic            bready,

  output logic            idle
);
  localparam int CW = $clog2(WR_FIFO_D) + 1;
  localparam int OW = $clog2(WR_OUTS) + 1;

  // Beat FIFO -----------------------------------------------------------------------
  logic          wf_valid, wf_ready;
  logic [BW-1:0] wf_data;
  logic [CW-1:0] wf_count;
  vo_fifo #(.W(BW), .D(WR_FIFO_D), .BRAM(1'b1)) u_wf (
    .clk, .rst,
    .in_valid (in_valid), .in_ready (in_ready), .in_data (in_data),
    .out_valid(wf_valid), .out_ready(wf_ready), .out_data(wf_data),
    .count    (wf_count)
  );

  // Bursts and AW -------------------------------------------------------------------
  logic        bg_valid, bg_ready, bg_done;
  logic [59:0] bg_waddr;
  logic [8:0]  bg_len;

  vo_burstgen #(.MAXB(WR_BURST)) u_bg (
    .clk, .rst, .start, .g,
    .b_valid (bg_valid), .b_ready (bg_ready), .b_waddr (bg_waddr), .b_len (bg_len),
    .done    (bg_done)
  );

  logic [CW-1:0] pend_w;                    // beats of issued bursts not yet sent on W
  logic [OW-1:0] outstanding;               // bursts issued, B not yet received
  logic          lq_in_ready, lq_valid, lq_ready;
  logic [7:0]    lq_data;
  logic [OW-1:0] lq_count;
  logic          issue, w_fire, w_end, b_fire;

  // every beat of the next burst is buffered and not promised to an earlier one
  logic [CW-1:0] avail;
  assign avail = wf_count - pend_w;
  assign issue = bg_valid && (!awvalid || awready) && (avail >= CW'(bg_len)) &&
                 lq_in_ready && (outstanding != OW'(WR_OUTS));
  assign bg_ready = issue;

  always_ff @(posedge clk) begin
    if (rst) begin
      awvalid     <= 1'b0;
      pend_w      <= '0;
      outstanding <= '0;
    end else begin
      if (awvalid && awready) awvalid <= 1'b0;
      if (issue) begin
        awvalid <= 1'b1;
        awaddr  <= {bg_waddr, 4'b0};
        awlen   <= 8'(bg_len - 9'd1);
      end
      pend_w      <= pend_w + (issue ? CW'(bg_len) : '0) - (w_fire ? CW'(1) : '0);
      outstanding <= outstanding + (issue ? OW'(1) : '0) - (b_fire ? OW'(1) : '0);
    end
  end

  // Burst lengths, AW -> W (in issue order).
  vo_fifo #(.W(8), .D(WR_OUTS), .BRAM(1'b0)) u_lq (
    .clk, .rst,
    .in_valid (issue),    .in_ready (lq_in_ready), .in_data (8'(bg_len - 9'd1)),
    .out_valid(lq_valid), .out_ready(lq_ready),    .out_data(lq_data),
    .count    (lq_count)
  );

  // W ---------------------------------------------------------------------------------
  logic [7:0] wcnt;
  assign wvalid   = lq_valid && wf_valid;
  assign wdata    = wf_data;
  assign wlast    = (wcnt == lq_data);
  assign w_fire   = wvalid && wready;
  assign w_end    = w_fire && wlast;
  assign wf_ready = lq_valid && wready;
  assign lq_ready = w_end;

  always_ff @(posedge clk) begin
    if (rst) wcnt <= '0;
    else if (w_fire) wcnt <= wlast ? 8'd0 : wcnt + 8'd1;
  end

  assign bready = 1'b1;
  assign b_fire = bvalid;

  assign idle = bg_done && !awvalid && (wf_count == '0) && !lq_valid && (outstanding == '0);

  logic unused;
  assign unused = ^lq_count;

endmodule
