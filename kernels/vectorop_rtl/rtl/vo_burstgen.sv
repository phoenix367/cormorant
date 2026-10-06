// ---------------------------------------------------------------------------
// vo_burstgen — walks one operand's run geometry (vo_pkg::geom_t) and emits
// its INCR bursts: at most MAXB words each, never crossing a 4 KiB boundary,
// never spanning two runs.  One burst per cycle (registered output).
//
// `start` (one cycle, with `g` stable from then on) begins the walk; `done`
// is high once the last burst has been taken (and before any start).
// ---------------------------------------------------------------------------
module vo_burstgen
  import vo_pkg::*;
#(
  parameter int MAXB = 64                  // power of two, <= 256
) (
  input  logic        clk,
  input  logic        rst,

  input  logic        start,
  input  geom_t       g,

  output logic        b_valid,
  input  logic        b_ready,
  output logic [59:0] b_waddr,
  output logic [8:0]  b_len,               // 1 .. MAXB words

  output logic        done
);
  logic        active;
  logic [31:0] runs_left;                   // runs not finished, current included
  logic [31:0] rem;                         // words left in the current run
  logic [59:0] cur_w;                       // next word to request
  logic [59:0] next_run_w;                  // first word of the next run

  // Next burst: min(rem, MAXB, words to the next 4 KiB boundary).  No
  // full-width carry chain follows `len` (300 MHz): the run end is decided
  // beside it, and the address and the remainder add `len` to their low
  // bits only, the high part's +1 / -1 being computed from the registers.
  // The high parts' carry and borrow are decided beside `len` too, from the
  // registers (not from lo_sum / rem_lo), which keeps the remainder's update
  // within 300 MHz:
  //   carry  = cur_w[7:0] + len >= 256  <=>  len == to_4k  <=>  lim >= to_4k
  //   borrow = rem[8:0] < len  <=>  big && rem[8:0] < MAXB && rem[8:0] < to_4k
  //            (without big, len <= lim = rem[8:0])
  logic [8:0]  to_4k, lim, len;
  logic        big, last, carry, borrow;
  logic [8:0]  lo_sum;                      // cur_w[7:0] + len  (low 8 bits used)
  logic [9:0]  rem_lo;                      // rem[8:0] - len    (low 9 bits used)
  logic [51:0] hi_inc;                      // cur_w[59:8] + 1
  logic [22:0] rem_hi_dec;                  // rem[31:9] - 1
  always_comb begin
    to_4k      = 9'd256 - {1'b0, cur_w[7:0]};
    big        = rem > 32'(MAXB);
    lim        = big ? 9'(MAXB) : rem[8:0];
    len        = (lim > to_4k) ? to_4k : lim;
    last       = !big && (rem[8:0] <= to_4k);  // len == rem: the run ends with this burst
    carry      = (lim >= to_4k);
    borrow     = big && ({1'b0, rem[8:0]} < 10'(MAXB)) && (rem[8:0] < to_4k);
    lo_sum     = {1'b0, cur_w[7:0]} + len;
    rem_lo     = {1'b0, rem[8:0]} - {1'b0, len};
    hi_inc     = cur_w[59:8] + 52'd1;
    rem_hi_dec = rem[31:9] - 23'd1;
  end

  logic load;                               // the output register takes the next burst
  assign load = active && (!b_valid || b_ready);

  always_ff @(posedge clk) begin
    if (rst) begin
      active  <= 1'b0;
      b_valid <= 1'b0;
    end else if (start) begin
      active     <= g.en && (g.n_runs != '0) && (g.run_words != '0);
      runs_left  <= g.n_runs;
      rem        <= g.run_words;
      cur_w      <= g.base_w;
      next_run_w <= g.base_w + 60'(g.stride_w);
      b_valid    <= 1'b0;
    end else begin
      if (b_valid && b_ready) b_valid <= 1'b0;
      if (load) begin
        b_valid <= 1'b1;
        b_waddr <= cur_w;
        b_len   <= len;
        if (last) begin
          if (runs_left == 32'd1) active <= 1'b0;
          runs_left  <= runs_left - 32'd1;
          rem        <= g.run_words;
          cur_w      <= next_run_w;
          next_run_w <= next_run_w + 60'(g.stride_w);
        end else begin
          rem   <= {borrow ? rem_hi_dec : rem[31:9], rem_lo[8:0]};
          cur_w <= {carry ? hi_inc : cur_w[59:8], lo_sum[7:0]};
        end
      end
    end
  end

  assign done = !active && !b_valid;

endmodule
