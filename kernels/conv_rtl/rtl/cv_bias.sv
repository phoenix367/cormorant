// ---------------------------------------------------------------------------
// cv_bias — the bias reader (gmem2, the HLS bias_producer): at job start the
// whole bias vector (ceil(out_ch / 8) words, one burst — two across 4 KiB)
// into a RAM of one TM-lane word per m-tile; the engine reads a tile's word
// asynchronously.  Without a bias the words read as zero.
// ---------------------------------------------------------------------------
module cv_bias
  import cv_pkg::*;
(
  input  logic          clk,
  input  logic          rst,                // unit reset: the job's start
  input  logic          start,              // load now (comes with rst)
  input  job_t          j,

  output logic          arvalid,
  input  logic          arready,
  output logic [63:0]   araddr,
  output logic [7:0]    arlen,
  input  logic          rvalid,
  output logic          rready,
  input  logic [BW-1:0] rdata,
  input  logic          rlast,

  output logic          ok,                 // loaded (or no bias)
  input  logic [6:0]    tile,
  output logic [TM*EW-1:0] vec,

  output logic          idle
);
  logic [8:0]  n_words, nw, issued, got;   // nw: n_words latched at the start
  logic        second;
  logic [59:0] waddr;
  logic        loading;

  // the bias words: ceil(out_ch / 8) <= MAXOC / 8 — one burst, or two across 4 KiB,
  // both in flight; one RAM word per m-tile
  if (B_BURST < MAXOC / E || B_OUTS < 2 || MAXMT > 128) begin : g_check
    $error("cv_bias: B_BURST / B_OUTS / the bias RAM do not cover MAXOC");
  end

  assign n_words = 9'((j.out_ch + 32'd7) >> 3);

  logic [8:0] to4k, b_len;
  assign to4k  = 9'd256 - {1'b0, waddr[7:0]};
  assign b_len = second ? nw - to4k : ((nw > to4k) ? to4k : nw);

  always_ff @(posedge clk) begin
    if (rst) begin
      // j changes when the next job is configured, while this unit may still
      // hold the last one's state: everything it needs is latched here
      loading <= start && j.has_bias;
      waddr   <= j.b_w;
      nw      <= n_words;
      arvalid <= 1'b0;
      second  <= 1'b0;
      got     <= '0;
      issued  <= '0;
    end else begin
      if (arvalid && arready) arvalid <= 1'b0;
      // up to two bursts, issued back to back (B_OUTS = 2)
      if (loading && (!arvalid || arready) && issued != nw) begin
        arvalid <= 1'b1;
        araddr  <= {(second ? waddr + 60'(to4k) : waddr), 4'b0};
        arlen   <= 8'(b_len - 9'd1);
        issued  <= issued + b_len;
        second  <= 1'b1;
      end
      if (rvalid) got <= got + 9'd1;
    end
  end

  assign rready = 1'b1;
  assign ok     = !j.has_bias || (loading && got == nw);

  // lanes 0-7 and 8-15 of each tile
  logic [E*EW-1:0] lo, hi;
  cv_lutram #(.W(E * EW), .D(128)) u_lo (
    .clk, .we (rvalid && !got[0]), .waddr (got[7:1]), .wdata (rdata), .raddr (tile), .rdata (lo)
  );
  cv_lutram #(.W(E * EW), .D(128)) u_hi (
    .clk, .we (rvalid && got[0]), .waddr (got[7:1]), .wdata (rdata), .raddr (tile), .rdata (hi)
  );
  assign vec = j.has_bias ? {hi, lo} : '0;

  assign idle = !arvalid && (!loading || got == nw);

  logic unused;
  assign unused = rlast ^ got[8] ^ ^j;

endmodule
