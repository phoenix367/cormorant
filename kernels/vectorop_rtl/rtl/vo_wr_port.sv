// ---------------------------------------------------------------------------
// vo_wr_port — the output c on gmem2: word stream -> DDR runs.
//
// Words are buffered in a block-RAM FIFO; the output geometry's bursts (at
// most WR_BURST words, never crossing 4 KiB or a run) are issued on AW once
// all of a burst's words are buffered, so W streams without gaps; at most
// WR_OUTS bursts await their B.  Every beat is written whole (the tail lanes
// of a run's last word carry op(0, 0) = 0, the alignment contract).
// `idle` means every burst has been written and acknowledged.  The AXI side
// is registered both ways: AW and W leave through register slices (AWREADY /
// WREADY only enable them) and BVALID is registered before it is counted.
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

  logic [OW-1:0] outstanding;               // bursts issued, B not yet received
  logic          lq_in_ready, lq_valid, lq_ready;
  logic [7:0]    lq_data;
  logic [OW-1:0] lq_count;
  logic          issue, w_fire, w_end, b_fire;
  logic          aw_in_ready, w_in_ready, bv_q;

  // The next bursts wait in a register slice, so the burst generator's
  // advance (the enable of its ~150 address / count registers) depends on
  // the slice's registered in_ready only, and the issue decision is an AND
  // of registers:
  //   avail    beats buffered and not promised to an issued burst, kept as a
  //            register (wf_count - pend_w: a W pop lowers both, a push
  //            raises it, an issue lowers it by the burst)
  //   av_ok    avail >= the head burst's length, computed one cycle ahead
  //            for the head of the next cycle; it ignores this cycle's push,
  //            so it can only be late (one cycle after a burst's last beat)
  //   ob_ok    fewer than WR_OUTS bursts await B (one cycle late on a B)
  logic          bq_valid;
  logic [59:0]   bq_waddr, hd_waddr;
  logic [8:0]    bq_len, hd_len;
  vo_rs #(.W(69)) u_bq (
    .clk, .rst,
    .in_valid  (bg_valid), .in_ready  (bg_ready), .in_data ({bg_waddr, bg_len}),
    .out_valid (bq_valid), .out_ready (issue),    .out_data ({bq_waddr, bq_len}),
    .head_data ({hd_waddr, hd_len})
  );

  logic [CW-1:0] avail;
  logic          av_ok, ob_ok, in_fire;
  assign in_fire = in_valid && in_ready;
  assign issue   = bq_valid && av_ok && ob_ok && aw_in_ready && lq_in_ready;

  always_ff @(posedge clk) begin
    if (rst) begin
      avail       <= '0;
      outstanding <= '0;
      av_ok       <= 1'b0;
      ob_ok       <= 1'b0;
    end else begin
      logic [CW-1:0] a_left;               // avail after this cycle's issue (no push)
      a_left = avail - CW'(bq_len);
      avail  <= issue ? avail - CW'(bq_len) + CW'(in_fire) : avail + CW'(in_fire);
      outstanding <= outstanding + (issue ? OW'(1) : '0) - (b_fire ? OW'(1) : '0);
      // the head of the next cycle: hd after an issue or into an empty slice
      if (issue)          av_ok <= (a_left >= CW'(hd_len));
      else if (!bq_valid) av_ok <= (avail >= CW'(hd_len));
      else                av_ok <= (avail >= CW'(bq_len));
      ob_ok <= issue ? (outstanding + OW'(1) < OW'(WR_OUTS)) : (outstanding < OW'(WR_OUTS));
    end
  end
  logic unused_hd;
  assign unused_hd = ^hd_waddr;

  vo_rs #(.W(72)) u_aw (
    .clk, .rst,
    .in_valid  (issue),   .in_ready  (aw_in_ready), .in_data ({bq_waddr, 4'b0, 8'(bq_len - 9'd1)}),
    .out_valid (awvalid), .out_ready (awready),     .out_data ({awaddr, awlen}),
    .head_data ()
  );

  // Burst lengths, AW -> W (in issue order).
  vo_fifo #(.W(8), .D(WR_OUTS), .BRAM(1'b0)) u_lq (
    .clk, .rst,
    .in_valid (issue),    .in_ready (lq_in_ready), .in_data (8'(bq_len - 9'd1)),
    .out_valid(lq_valid), .out_ready(lq_ready),    .out_data(lq_data),
    .count    (lq_count)
  );

  // W ---------------------------------------------------------------------------------
  logic [7:0] wcnt;
  logic       wv_i, wl_i;
  assign wv_i     = lq_valid && wf_valid;
  assign wl_i     = (wcnt == lq_data);
  assign w_fire   = wv_i && w_in_ready;
  assign w_end    = w_fire && wl_i;
  assign wf_ready = lq_valid && w_in_ready;
  assign lq_ready = w_end;

  vo_rs #(.W(BW + 1)) u_w (
    .clk, .rst,
    .in_valid  (wv_i),   .in_ready  (w_in_ready), .in_data ({wl_i, wf_data}),
    .out_valid (wvalid), .out_ready (wready),     .out_data ({wlast, wdata}),
    .head_data ()
  );

  always_ff @(posedge clk) begin
    if (rst) wcnt <= '0;
    else if (w_fire) wcnt <= wl_i ? 8'd0 : wcnt + 8'd1;
  end

  // BVALID registered before it is counted.
  assign bready = 1'b1;
  always_ff @(posedge clk) begin
    if (rst) bv_q <= 1'b0;
    else     bv_q <= bvalid;
  end
  assign b_fire = bv_q;

  assign idle = bg_done && !bq_valid && bg_ready && !awvalid && aw_in_ready && (wf_count == '0) && !lq_valid &&
                !wvalid && w_in_ready && (outstanding == '0);

  logic unused;
  assign unused = ^lq_count;

endmodule
