// ---------------------------------------------------------------------------
// ro_burstgen — walks one operand's run geometry (ro_pkg::geom_t) and emits
// its INCR bursts: at most MAXB words each, never crossing a 4 KiB boundary,
// never spanning two runs.  One burst per cycle (registered output).  Blocks
// (g.n_blk) repeat the run pattern from base + b * blk_w.
//
// `start` (one cycle, with `g` stable from then on) begins the walk; `done`
// is high once the last burst has been taken (and before any start).
// ---------------------------------------------------------------------------
module ro_burstgen
  import ro_pkg::*;
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
  output logic        b_zero,              // the burst's run is outside the window: zero words, no AR

  output logic        done
);
  logic        active;
  logic [31:0] blks_left;                   // blocks not finished, current included
  logic [31:0] tiles_left;                  // tiles of the block not finished, current included
  logic [31:0] runs_left;                   // runs of the tile not finished, current included
  logic        run1, tile1, blk1, nr1, nt1; // runs_left == 1, tiles_left == 1, blks_left == 1, n_runs == 1, n_tiles == 1
  logic [31:0] rem;                         // words left in the current run
  logic [59:0] cur_w;                       // next word to request
  logic [59:0] next_run_w;                  // first word of the next run
  logic [59:0] tile_next;                   // first word of the next tile
  logic [59:0] blk_next;                    // first word of the next block
  logic [31:0] run_i;                       // index of the current run within its block
  logic        zero_cur, zero_next;         // the current / next run is outside the window
  assign zero_next = ((run_i + 32'd1) < g.win_lo) || ((run_i + 32'd1) >= g.win_hi);

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
      active     <= g.en && (g.n_blk != '0) && (g.n_tiles != '0) && (g.n_runs != '0) && (g.run_words != '0);
      blks_left  <= g.n_blk;
      blk1       <= (g.n_blk == 32'd1);
      tiles_left <= g.n_tiles;
      tile1      <= (g.n_tiles == 32'd1);
      nt1        <= (g.n_tiles == 32'd1);
      runs_left  <= g.n_runs;
      run1       <= (g.n_runs == 32'd1);
      nr1        <= (g.n_runs == 32'd1);
      rem        <= g.run_words;
      cur_w      <= g.base_w;
      next_run_w <= g.base_w + 60'(g.stride_w);
      tile_next  <= g.base_w + 60'(g.tile_w);
      blk_next   <= g.base_w + 60'(g.blk_w);
      run_i      <= '0;
      zero_cur   <= (32'd0 < g.win_lo) || (32'd0 >= g.win_hi);
      b_valid    <= 1'b0;
    end else begin
      if (b_valid && b_ready) b_valid <= 1'b0;
      if (load) begin
        b_valid <= 1'b1;
        b_waddr <= cur_w;
        b_len   <= len;
        b_zero  <= zero_cur;
        if (last) begin
          rem <= g.run_words;
          if (run1 && tile1) begin          // the block ends
            if (blk1) active <= 1'b0;
            blks_left  <= blks_left - 32'd1;
            blk1       <= (blks_left == 32'd2);
            tiles_left <= g.n_tiles;
            tile1      <= nt1;
            runs_left  <= g.n_runs;
            run1       <= nr1;
            cur_w      <= blk_next;
            next_run_w <= blk_next + 60'(g.stride_w);
            tile_next  <= blk_next + 60'(g.tile_w);
            blk_next   <= blk_next + 60'(g.blk_w);
            run_i      <= '0;
            zero_cur   <= (32'd0 < g.win_lo) || (32'd0 >= g.win_hi);
          end else if (run1) begin          // the tile ends
            tiles_left <= tiles_left - 32'd1;
            tile1      <= (tiles_left == 32'd2);
            runs_left  <= g.n_runs;
            run1       <= nr1;
            cur_w      <= tile_next;
            next_run_w <= tile_next + 60'(g.stride_w);
            tile_next  <= tile_next + 60'(g.tile_w);
            run_i      <= '0;
            zero_cur   <= (32'd0 < g.win_lo) || (32'd0 >= g.win_hi);
          end else begin
            runs_left  <= runs_left - 32'd1;
            run1       <= (runs_left == 32'd2);
            cur_w      <= next_run_w;
            next_run_w <= next_run_w + 60'(g.stride_w);
            run_i      <= run_i + 32'd1;
            zero_cur   <= zero_next;
          end
        end else begin
          rem   <= {borrow ? rem_hi_dec : rem[31:9], rem_lo[8:0]};
          cur_w <= {carry ? hi_inc : cur_w[59:8], lo_sum[7:0]};
        end
      end
    end
  end

  assign done = !active && !b_valid;

endmodule
