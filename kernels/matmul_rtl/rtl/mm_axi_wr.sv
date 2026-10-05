// ---------------------------------------------------------------------------
// mm_axi_wr — AXI4 write master for the 128-bit C port.
//
// Beats (data + strobes) are buffered in a block-RAM FIFO.  Word-range
// descriptors are split into INCR bursts of at most WR_BURST beats that never
// cross a 4 KiB boundary; a burst's AW is only issued once all of its beats
// are in the FIFO, so W streams without gaps; at most WR_OUTS bursts await
// their B.  `idle` means every accepted beat has been written and acknowledged.
// ---------------------------------------------------------------------------
module mm_axi_wr
  import mm_pkg::*;
(
  input  logic            clk,
  input  logic            rst,

  input  logic            desc_valid,
  output logic            desc_ready,
  input  logic [59:0]     desc_waddr,
  input  logic [11:0]     desc_nw,

  input  logic            in_valid,
  output logic            in_ready,
  input  logic [BW-1:0]   in_data,
  input  logic [2*E-1:0]  in_strb,

  output logic            awvalid,
  input  logic            awready,
  output logic [63:0]     awaddr,
  output logic [7:0]      awlen,
  output logic            wvalid,
  input  logic            wready,
  output logic [BW-1:0]   wdata,
  output logic [2*E-1:0]  wstrb,
  output logic            wlast,
  input  logic            bvalid,
  output logic            bready,

  output logic            idle
);
  localparam int FW = BW + 2*E;
  localparam int CW = $clog2(WR_FIFO_D) + 1;

  // Descriptor queue -----------------------------------------------------------
  logic        dq_valid, dq_ready;
  logic [71:0] dq_data;
  logic [2:0]  dq_count;
  mm_fifo #(.W(72), .D(4), .BRAM(1'b0)) u_dq (
    .clk, .rst,
    .in_valid (desc_valid), .in_ready (desc_ready), .in_data ({desc_waddr, desc_nw}),
    .out_valid(dq_valid),   .out_ready(dq_ready),   .out_data(dq_data),
    .count    (dq_count)
  );

  // Beat FIFO ------------------------------------------------------------------
  logic          wf_valid, wf_ready;
  logic [FW-1:0] wf_data;
  logic [CW-1:0] wf_count;
  mm_fifo #(.W(FW), .D(WR_FIFO_D), .BRAM(1'b1)) u_wf (
    .clk, .rst,
    .in_valid (in_valid), .in_ready (in_ready), .in_data ({in_strb, in_data}),
    .out_valid(wf_valid), .out_ready(wf_ready), .out_data(wf_data),
    .count    (wf_count)
  );

  // AW side ----------------------------------------------------------------------
  logic        busy;
  logic [59:0] cur_w;
  logic [11:0] rem;
  logic [CW:0] pend_w;      // beats of issued bursts not yet sent on W

  assign dq_ready = !busy;

  logic [8:0]  to_4k;
  logic [11:0] blen, blen_c;
  logic        bl_v;       // blen holds the next burst of cur_w / rem
  always_comb begin
    to_4k = 9'd256 - {1'b0, cur_w[7:0]};
    blen_c = rem;
    if (blen_c > 12'(WR_BURST)) blen_c = 12'(WR_BURST);
    if (blen_c > {3'b0, to_4k}) blen_c = {3'b0, to_4k};
  end

  // Burst-length queue from AW to W.
  logic        lq_in_ready, lq_valid, lq_ready;
  logic [7:0]  lq_data;
  logic [$clog2(WR_OUTS):0] lq_count;

  logic [CW:0] avail;
  assign avail = {1'b0, wf_count} - pend_w;

  logic issue;
  // Registered "all beats of the next burst are buffered" and "fewer than
  // WR_OUTS bursts await B".  Safe although one cycle old: only an AW issue
  // can lower `avail` or raise `outstanding`, and after an issue bl_v blocks
  // the next one until blen and these flags are recomputed.
  localparam int OW = $clog2(WR_OUTS) + 1;
  logic [OW-1:0] outstanding;  // bursts awaiting B
  logic av_ok, ob_ok;
  always_ff @(posedge clk) begin
    av_ok <= !rst && (avail >= (CW+1)'(bl_v ? blen : blen_c));
    ob_ok <= !rst && (outstanding < OW'(WR_OUTS));
  end
  assign issue = busy && bl_v && av_ok && ob_ok && (!awvalid || awready) && lq_in_ready;

  logic       w_fire, b_fire, w_end;

  always_ff @(posedge clk) begin
    if (rst) begin
      busy        <= 1'b0;
      bl_v        <= 1'b0;
      awvalid     <= 1'b0;
      pend_w      <= '0;
      outstanding <= '0;
    end else begin
      if (dq_valid && dq_ready) begin
        busy  <= (dq_data[11:0] != '0);
        cur_w <= dq_data[71:12];
        rem   <= dq_data[11:0];
      end
      if (awvalid && awready) awvalid <= 1'b0;

      // One cycle to size the next burst after every change of cur_w / rem.
      if (issue) bl_v <= 1'b0;
      else if (busy && !bl_v) begin blen <= blen_c; bl_v <= 1'b1; end
      if (issue) begin
        awvalid <= 1'b1;
        awaddr  <= {cur_w, 4'b0};
        awlen   <= 8'(blen - 12'd1);
        cur_w   <= cur_w + 60'(blen);
        rem     <= rem - blen;
        if (rem == blen) busy <= 1'b0;
      end
      pend_w      <= pend_w + (issue ? (CW+1)'(blen) : '0) - (w_fire ? (CW+1)'(1) : '0);
      outstanding <= outstanding + (issue ? OW'(1) : '0) - (b_fire ? OW'(1) : '0);
    end
  end

  mm_fifo #(.W(8), .D(WR_OUTS), .BRAM(1'b0)) u_lq (
    .clk, .rst,
    .in_valid (issue),   .in_ready (lq_in_ready), .in_data (8'(blen - 12'd1)),
    .out_valid(lq_valid), .out_ready(lq_ready),   .out_data(lq_data),
    .count    (lq_count)
  );

  // W side -----------------------------------------------------------------------
  logic [7:0] wcnt;
  assign wvalid   = lq_valid && wf_valid;
  assign wdata    = wf_data[BW-1:0];
  assign wstrb    = wf_data[FW-1:BW];
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

  assign idle = !busy && !dq_valid && !awvalid && (wf_count == '0) &&
                !lq_valid && (outstanding == '0);

endmodule
