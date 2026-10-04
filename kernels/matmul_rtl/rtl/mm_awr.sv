// ---------------------------------------------------------------------------
// mm_awr — A writer of read port P: stores the port's A-panel rows (gearbox
// beats of A runs) into the A buffer, splitting K between the two lanes.
//
// K block b (16 planes = 16 << lk elements) belongs to lane b % 2; within a
// lane the blocks are packed back to back, so beat i of a row (elements
// 8i..8i+7, never straddling a block) goes to
//     lane  (i >> (lk + 1)) & 1
//     word  ((i >> (lk + 2)) << (lk + 1)) | (i & ((2 << lk) - 1)).
// A split last block (cfg.split, beats from cfg.bsplit on) is shared: beat
// 2j + h of the block holds planes 8h..8h+7 of tap j, so lane = i & 1; the
// word formula is unchanged (both lanes keep it at their next local block).
//
// Panel handshake: the A buffer is single-buffered, so panel p may only be
// written after both lanes' x prefetchers have finished reading panel p - 1
// (xpf_cnt == awr_cnt).  Every panel has exactly one A run or A marker per
// port; awr_cnt counts the panels this port has finished.
// ---------------------------------------------------------------------------
module mm_awr
  import mm_pkg::*;
(
  input  logic            clk,
  input  logic            rst,
  input  cfg_t            cfg,

  input  logic            in_valid,
  output logic            in_ready,
  input  logic [BW-1:0]   in_data,
  input  logic            in_marker,
  input  logic            in_run_last,
  input  logic [8:0]      in_beat,
  input  logic [4:0]      in_row,

  input  logic [1:0][31:0] xpf_cnt,
  output logic [31:0]     awr_cnt,

  output logic            we,
  output logic [1:0]      wrow,
  output logic            wlane,
  output logic [A_AW-1:0] waddr,
  output logic [BW-1:0]   wdata
);
  logic passed;               // this panel's barrier has been passed
  logic bar_q, last_q;
  assign in_ready = !last_q && (passed || bar_q);

  // Registered barrier, evaluated against awr_cnt's next value: it can only
  // open late (xpf_cnt only grows), never early.
  logic [31:0] awr_cnt_n;
  assign awr_cnt_n = awr_cnt + {31'b0, last_q};
  always_ff @(posedge clk)
    bar_q <= !rst && (xpf_cnt[0] == awr_cnt_n) && (xpf_cnt[1] == awr_cnt_n);

  logic fire;
  assign fire = in_valid && in_ready;

  // The panel is only announced once its last write (registered below) has
  // reached the RAM, so the x prefetchers never read a stale word.
  always_ff @(posedge clk) begin
    if (rst) begin
      passed  <= 1'b0;
      awr_cnt <= '0;
      we      <= 1'b0;
      last_q  <= 1'b0;
    end else begin
      we     <= fire && !in_marker;
      last_q <= fire && in_run_last;
      if (fire) passed <= !in_run_last;
      if (last_q) awr_cnt <= awr_cnt + 32'd1;
    end
  end

  always_ff @(posedge clk) begin
    if (fire) begin
      wrow  <= in_row[1:0];
      wlane <= (cfg.split && {1'b0, in_beat} >= cfg.bsplit) ? in_beat[0]
                                                            : in_beat[{2'b0, cfg.lk} + 4'd1];
      waddr <= A_AW'(((in_beat >> ({2'b0, cfg.lk} + 4'd2)) << ({2'b0, cfg.lk} + 4'd1)) |
                     (in_beat & ((9'd2 << cfg.lk) - 9'd1)));
      wdata <= in_data;
    end
  end

endmodule
