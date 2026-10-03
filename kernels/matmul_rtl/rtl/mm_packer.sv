// ---------------------------------------------------------------------------
// mm_packer — packs the output element stream of one C run into 128-bit
// write beats with byte strobes.
//
// A run is a contiguous range of `n` elements starting at byte address
// `addr` (2-byte aligned).  Groups of 1..8 elements arrive in order; element
// j of the run lands in lane (addr / 2 + j) % 8 of its word.  Only the run's
// first and last beat can carry partial strobes.  For every run the packer
// also emits one word-range descriptor for the AW side.
// ---------------------------------------------------------------------------
module mm_packer
  import mm_pkg::*;
(
  input  logic          clk,
  input  logic          rst,

  input  logic          run_valid,
  output logic          run_ready,
  input  logic [63:0]   run_addr,
  input  logic [12:0]   run_n,

  input  logic          el_valid,
  output logic          el_ready,
  input  logic [BW-1:0] el_data,
  input  logic [3:0]    el_cnt,       // 1..8

  output logic          aw_valid,
  input  logic          aw_ready,
  output logic [59:0]   aw_waddr,
  output logic [11:0]   aw_nw,

  output logic          w_valid,
  input  logic          w_ready,
  output logic [BW-1:0] w_data,
  output logic [E*2-1:0] w_strb,

  output logic          idle
);
  logic            active, flushing;
  logic [4:0]      f;              // filled positions (incl. leading holes)
  logic [EW-1:0]   buf_q [2*E];
  logic [2*E-1:0]  vm;             // valid element positions
  logic [12:0]     rem;

  // Run start: also hand the word range to the AW side.
  logic [2:0]  s;
  assign s         = run_addr[3:1];
  assign aw_valid  = run_valid && !active;
  assign aw_waddr  = run_addr[63:4];
  assign aw_nw     = 12'((13'(s) + run_n + 13'd7) >> 3);
  assign run_ready = !active && aw_ready;

  // Word emission.
  logic emit, stall;
  assign emit  = active && ((f >= 5'd8) || (flushing && f != 5'd0));
  assign stall = emit && !w_ready;

  assign w_valid = emit;
  always_comb begin
    for (int i = 0; i < E; i++) begin
      w_data[i*EW +: EW] = buf_q[i];
      w_strb[2*i +: 2]   = {2{vm[i]}};
    end
  end

  // Insertion point after this cycle's emission.
  logic [4:0] f1;
  assign f1 = emit ? ((f >= 5'd8) ? f - 5'd8 : 5'd0) : f;

  assign el_ready = active && !flushing && !stall;
  logic ins;
  assign ins = el_valid && el_ready;

  always_ff @(posedge clk) begin
    if (rst) begin
      active   <= 1'b0;
      flushing <= 1'b0;
      f        <= '0;
      vm       <= '0;
    end else begin
      if (run_valid && run_ready) begin
        active   <= 1'b1;
        flushing <= 1'b0;
        f        <= {2'b0, s};
        vm       <= '0;
        rem      <= run_n;
      end else if (active && !stall) begin
        // Shift out the emitted word.
        logic [EW-1:0]  b1 [2*E];
        logic [2*E-1:0] vm1;
        for (int i = 0; i < 2*E; i++) begin
          b1[i]  = buf_q[i];
          vm1[i] = vm[i];
        end
        if (emit) begin
          for (int i = 0; i < E; i++) begin
            b1[i]      = buf_q[i+E];
            vm1[i]     = (f >= 5'd8) ? vm[i+E] : 1'b0;
            vm1[i+E]   = 1'b0;
          end
        end
        // Insert the arriving group at f1.
        if (ins) begin
          for (int i = 0; i < 2*E; i++) begin
            logic [4:0] d;
            d = 5'(i) - f1;
            if (5'(i) >= f1 && d < {1'b0, el_cnt}) begin
              b1[i]  = el_data[d[2:0]*EW +: EW];
              vm1[i] = 1'b1;
            end
          end
        end
        for (int i = 0; i < 2*E; i++) buf_q[i] <= b1[i];
        vm <= vm1;
        f  <= f1 + (ins ? {1'b0, el_cnt} : 5'd0);
        if (ins) begin
          rem <= rem - {9'b0, el_cnt};
          if (rem == {9'b0, el_cnt}) flushing <= 1'b1;
        end
        if (flushing && emit && f <= 5'd8) begin
          active   <= 1'b0;
          flushing <= 1'b0;
          f        <= '0;
        end
      end
    end
  end

  assign idle = !active;

endmodule
