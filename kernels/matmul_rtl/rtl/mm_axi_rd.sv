// ---------------------------------------------------------------------------
// mm_axi_rd — AXI4 read master for one 128-bit port.
//
// Takes word-range descriptors (mm_pkg::rd_run_t), splits them into INCR
// bursts of at most RD_BURST beats that never cross a 4 KiB boundary, and
// buffers the returned beats in a block-RAM FIFO.  A burst is only issued when
// the FIFO has room for all of it, so RREADY is tied high and the port never
// back-pressures the interconnect; at most RD_OUTS bursts await their data.
// Single ID, so data returns in order.
//
// The AXI side is registered both ways: AR leaves through a register slice
// (ARREADY only enables the slice), and R is registered before the FIFO
// write.  Both add a cycle of latency; the FIFO reservation is unchanged.
// ---------------------------------------------------------------------------
module mm_axi_rd
  import mm_pkg::*;
(
  input  logic          clk,
  input  logic          rst,

  input  logic          desc_valid,
  output logic          desc_ready,
  input  rd_run_t       desc,

  // AXI4 AR / R
  output logic          arvalid,
  input  logic          arready,
  output logic [63:0]   araddr,
  output logic [7:0]    arlen,
  input  logic          rvalid,
  output logic          rready,
  input  logic [BW-1:0] rdata,
  input  logic          rlast,

  output logic          out_valid,
  input  logic          out_ready,
  output logic [BW-1:0] out_data,

  output logic          idle
);
  localparam int CW = $clog2(RD_FIFO_D) + 1;

  // Current descriptor.
  logic        busy;
  logic [59:0] cur_w;       // next word address
  logic [11:0] rem;         // words left to request

  logic [CW:0] space;       // FIFO entries not yet reserved by a burst
  logic        out_fire;

  // Next burst: min(rem, RD_BURST, words to the next 4 KiB boundary).
  logic [8:0]  to_4k;
  logic [11:0] blen, blen_c;
  logic        bl_v;       // blen holds the next burst of cur_w / rem
  always_comb begin
    to_4k = 9'd256 - {1'b0, cur_w[7:0]};
    blen_c = rem;
    if (blen_c > 12'(RD_BURST)) blen_c = 12'(RD_BURST);
    if (blen_c > {3'b0, to_4k}) blen_c = {3'b0, to_4k};
  end

  // AR register slice and the registered R channel.
  logic          ar_in_ready;
  logic          rv_q, rl_q;
  logic [BW-1:0] rd_q;
  always_ff @(posedge clk) begin
    if (rst) rv_q <= 1'b0;
    else     rv_q <= rvalid;
    rl_q <= rlast;
    rd_q <= rdata;
  end

  logic issue;
  // Registered FIFO-space and outstanding checks (see mm_axi_wr): only an
  // issue lowers `space` or raises `outs`, and bl_v blocks the cycle after
  // every issue.
  localparam int OW = $clog2(RD_OUTS) + 1;
  logic [OW-1:0] outs;      // bursts issued, last beat not yet received
  logic sp_ok, os_ok;
  always_ff @(posedge clk) begin
    sp_ok <= !rst && (space >= (CW+1)'(bl_v ? blen : blen_c));
    os_ok <= !rst && (outs < OW'(RD_OUTS));
  end
  assign issue      = busy && bl_v && sp_ok && os_ok && ar_in_ready;
  assign desc_ready = !busy;

  always_ff @(posedge clk) begin
    if (rst) begin
      busy    <= 1'b0;
      bl_v    <= 1'b0;
      space   <= (CW+1)'(RD_FIFO_D);
      outs    <= '0;
    end else begin
      if (desc_valid && desc_ready) begin
        busy  <= (desc.nw != '0);
        cur_w <= desc.waddr;
        rem   <= desc.nw;
      end
      // One cycle to size the next burst after every change of cur_w / rem.
      if (issue) bl_v <= 1'b0;
      else if (busy && !bl_v) begin blen <= blen_c; bl_v <= 1'b1; end
      if (issue) begin
        cur_w   <= cur_w + 60'(blen);
        rem     <= rem - blen;
        if (rem == blen) busy <= 1'b0;
      end
      space <= space - (issue ? (CW+1)'(blen) : '0) + (out_fire ? (CW+1)'(1) : '0);
      outs  <= outs + (issue ? OW'(1) : '0) - ((rv_q && rl_q) ? OW'(1) : '0);
    end
  end

  mm_rs #(.W(72)) u_ar (
    .clk, .rst,
    .in_valid  (issue),   .in_ready  (ar_in_ready), .in_data ({cur_w, 4'b0, 8'(blen - 12'd1)}),
    .out_valid (arvalid), .out_ready (arready),     .out_data ({araddr, arlen})
  );

  assign rready = 1'b1;

  logic       f_in_ready;
  logic [CW-1:0] f_count;
  mm_fifo #(.W(BW), .D(RD_FIFO_D), .BRAM(1'b1)) u_fifo (
    .clk, .rst,
    .in_valid (rv_q), .in_ready (f_in_ready), .in_data (rd_q),
    .out_valid, .out_ready, .out_data,
    .count    (f_count)
  );
  assign out_fire = out_valid && out_ready;

  assign idle = !busy && !arvalid && ar_in_ready && (space == (CW+1)'(RD_FIFO_D));

  logic unused;
  assign unused = f_in_ready;

endmodule
