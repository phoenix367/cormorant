// ---------------------------------------------------------------------------
// mm_fifo — synchronous first-word-fall-through FIFO, valid/ready on both
// sides.
//
//   BRAM = 0: distributed RAM with an asynchronous read (small FIFOs).
//   BRAM = 1: block RAM with a registered read, followed by a three-entry skid
//             stage so the output still sustains one pop per cycle.
//
// `count` is the number of accepted entries not yet popped, including the
// ones in the BRAM read pipeline (usable for credit schemes).
// ---------------------------------------------------------------------------
module mm_fifo #(
  parameter int W    = 32,
  parameter int D    = 16,          // power of two
  parameter bit BRAM = 1'b0
) (
  input  logic         clk,
  input  logic         rst,

  input  logic         in_valid,
  output logic         in_ready,
  input  logic [W-1:0] in_data,

  output logic         out_valid,
  input  logic         out_ready,
  output logic [W-1:0] out_data,

  output logic [$clog2(D):0] count
);
  localparam int AW = $clog2(D);

  logic [AW-1:0] wptr, rptr;
  logic [AW:0]   cnt;          // entries in the RAM array
  logic          in_fire;

  assign in_ready = (cnt != D[AW:0]);
  assign in_fire  = in_valid && in_ready;

  if (!BRAM) begin : g_lut
    (* ram_style = "distributed" *) logic [W-1:0] mem [D];
    logic out_fire;

    always_ff @(posedge clk) if (in_fire) mem[wptr] <= in_data;

    assign out_valid = (cnt != '0);
    assign out_data  = mem[rptr];
    assign out_fire  = out_valid && out_ready;
    assign count     = cnt;

    always_ff @(posedge clk) begin
      if (rst) begin
        wptr <= '0; rptr <= '0; cnt <= '0;
      end else begin
        if (in_fire)  wptr <= wptr + 1'b1;
        if (out_fire) rptr <= rptr + 1'b1;
        cnt <= cnt + (AW+1)'(in_fire) - (AW+1)'(out_fire);
      end
    end
  end else begin : g_bram
    (* ram_style = "block" *) logic [W-1:0] mem [D];
    logic [W-1:0] rdata;
    logic         rd_issue, inflight;
    logic [W-1:0] q [3];
    logic [1:0]   qn;          // entries in the skid stage (0..3)
    logic         out_fire;

    // Issue a RAM read whenever the skid stage can absorb it without
    // looking at this cycle's pop: with three skid entries one RAM read per
    // cycle keeps up with one pop per cycle.
    assign rd_issue = (cnt != '0) && ({1'b0, qn} + {2'b0, inflight} < 3'd3);

    always_ff @(posedge clk) begin
      if (in_fire)  mem[wptr] <= in_data;
      if (rd_issue) rdata <= mem[rptr];
    end

    assign out_valid = (qn != 2'd0);
    assign out_data  = q[0];
    assign out_fire  = out_valid && out_ready;
    assign count     = cnt + {{AW{1'b0}}, inflight} + {{(AW-1){1'b0}}, qn};

    logic [1:0] qn_pop;        // entries left after this cycle's pop
    assign qn_pop = qn - {1'b0, out_fire};

    always_ff @(posedge clk) begin
      if (rst) begin
        wptr <= '0; rptr <= '0; cnt <= '0; inflight <= 1'b0; qn <= 2'd0;
      end else begin
        if (in_fire)  wptr <= wptr + 1'b1;
        if (rd_issue) rptr <= rptr + 1'b1;
        cnt      <= cnt + (AW+1)'(in_fire) - (AW+1)'(rd_issue);
        inflight <= rd_issue;
        // Skid stage: pop from q[0], append the arriving RAM word.
        if (out_fire) begin
          q[0] <= q[1];
          q[1] <= q[2];
        end
        if (inflight) q[qn_pop] <= rdata;
        qn <= qn_pop + {1'b0, inflight};
      end
    end
  end

endmodule
