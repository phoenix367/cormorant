// ---------------------------------------------------------------------------
// mm_drain — empties the accumulators after every step and produces C.
//
// Waits until both lanes have finished step s (= drn_cnt), then reads the
// accumulator words of every valid panel row from both lanes in lockstep:
//
//   sum[l]  = lane0[l] + lane1[l]      (a lane that had no K block adds 0)
//   col[g]  = sum of the kw lanes of column g (GEMV image), else sum[l]
//   C       = saturate(col >>> 8)      (floor, then clamp to Q8.8)
//
// All adds wrap modulo 2^32, so the result is bit-identical to the
// reference's single ap_fixed<32,16> accumulator.  C runs go to the packer:
// one run for the whole panel when the chunk spans all m columns (the rows
// are contiguous), else one run per row.
// ---------------------------------------------------------------------------
module mm_drain
  import mm_pkg::*;
(
  input  logic              clk,
  input  logic              rst,
  input  logic              start,
  input  cfg_t              cfg,

  input  logic              step_valid,
  output logic              step_ready,
  input  step_t             step,

  input  logic [1:0][31:0]  cmp_cnt,
  input  logic [1:0]        lane_active,
  output logic [31:0]       drn_cnt,

  output logic              drd_en,
  output logic [ACC_AW-1:0] drd_addr,
  output logic [2:0]        drd_row,
  input  logic [1:0][E-1:0][31:0] drd_data,

  output logic              crun_valid,
  input  logic              crun_ready,
  output logic [63:0]       crun_addr,
  output logic [12:0]       crun_n,

  output logic              el_valid,
  input  logic              el_ready,
  output logic [BW-1:0]     el_data,
  output logic [3:0]        el_cnt,

  output logic              idle
);
  typedef enum logic [1:0] {S_IDLE, S_WAIT, S_RUN} st_t;
  st_t   state;
  step_t st;

  logic        act0, act1, contig;
  logic [2:0]  r;
  logic [6:0]  w, dw;         // word within the row / words per row
  logic [3:0]  last_cnt;      // elements of the last word (row-major)
  logic [63:0] row_addr;
  logic        run_pushed;    // this row's C run is queued

  // Output FIFO and credits.
  localparam int OD = 16;
  logic       of_in_valid, of_in_ready;
  logic [4:0] of_count;
  localparam int PIPE = 9;     // request -> output FIFO (see the arithmetic pipeline)
  logic [PIPE-1:0] pipe_v;    // requests in the read / arithmetic pipeline
  logic [3:0] inflight;
  always_comb begin
    inflight = '0;
    for (int i = 0; i < PIPE; i++) inflight += {3'b0, pipe_v[i]};
  end

  logic need_run, run_ok, credit_ok, rd_fire, last_rd;
  assign need_run  = (w == '0) && !run_pushed && (!contig || r == 3'd0);
  assign run_ok    = !need_run || crun_ready;
  assign credit_ok = ({1'b0, of_count} + {2'b0, inflight} < 6'(OD - 1));
  assign rd_fire   = (state == S_RUN) && run_ok && credit_ok;
  assign last_rd   = (w + 7'd1 == dw) && ({1'b0, r} + 4'd1 == st.n_valid);

  assign crun_valid = (state == S_RUN) && need_run && credit_ok;
  assign crun_addr  = row_addr;
  assign crun_n     = contig ? 13'(st.n_valid * st.mcc) : {3'b0, st.mcc};

  assign drd_en   = rd_fire;
  assign drd_addr = w[ACC_AW-1:0];
  assign drd_row  = r;

  assign step_ready = (state == S_IDLE);

  // words per row and the element count of the last one
  logic [9:0] mcc_lk;
  assign mcc_lk = 10'(({st.mcc, 3'b0} >> (2'd3 - cfg.lk)));   // mcc << lk (<= 512)

  always_ff @(posedge clk) begin
    if (rst || start) begin
      state   <= S_IDLE;
      drn_cnt <= '0;
    end else begin
      case (state)
        S_IDLE: if (step_valid) begin
          st    <= step;
          state <= S_WAIT;
        end
        S_WAIT: if (cmp_cnt[0] != drn_cnt && cmp_cnt[1] != drn_cnt) begin
          act0       <= lane_active[0];
          act1       <= lane_active[1];
          contig     <= ({22'b0, st.mcc} == cfg.m);
          r          <= '0;
          w          <= '0;
          dw         <= 7'((mcc_lk + 10'd7) >> 3);
          last_cnt   <= (cfg.lk != 2'd0) ? 4'(4'd8 >> cfg.lk)
                      : ((st.mcc[2:0] == 3'd0) ? 4'd8 : {1'b0, st.mcc[2:0]});
          row_addr   <= st.c_addr;
          run_pushed <= 1'b0;
          state      <= S_RUN;
        end
        S_RUN: begin
          if (crun_valid && crun_ready) run_pushed <= 1'b1;
          if (rd_fire) begin
            if (w + 7'd1 == dw) begin
              w          <= '0;
              r          <= r + 3'd1;
              row_addr   <= row_addr + {31'b0, cfg.m, 1'b0};
              run_pushed <= 1'b0;
            end else begin
              w <= w + 7'd1;
            end
            if (last_rd) begin
              state   <= S_IDLE;
              drn_cnt <= drn_cnt + 32'd1;
            end
          end
        end
        default: state <= S_IDLE;
      endcase
    end
  end

  // Arithmetic pipeline ---------------------------------------------------------------
  // request (0) -> the lanes' address copies (1, 2) -> accumulator read and
  // row select next to each beat lane's DSPs (3) -> lane word in drd_data
  // (4) -> registered here (5) -> lane sum (6) -> pair / quad sums (7) ->
  // column select (8) -> saturate (9) -> FIFO.  Stage k holds a request
  // made k cycles earlier; the lane words cross the MAC array between two
  // registers (mm_lane's drd_data and dq5).
  // tags: element count of the word and the lanes' activity, aligned with
  // the data stages (act0 / act1 change at the next step's S_WAIT, while
  // this step's words are still in the pipeline)
  logic [3:0] cnt0;
  logic [PIPE:1][3:0] cnt;
  logic [PIPE:1][1:0] act;
  assign cnt0 = (cfg.lk != 2'd0) ? 4'(4'd8 >> cfg.lk)
              : ((w + 7'd1 == dw) ? last_cnt : 4'd8);

  logic [1:0][E-1:0][31:0] dq5;
  logic [E-1:0][31:0] sum6, sum7, col8;
  logic [3:0][31:0]   t1_7;
  logic [1:0][31:0]   t2_7;
  logic [E-1:0][15:0] sat9;

  always_ff @(posedge clk) begin
    if (rst || start) pipe_v <= '0;
    else pipe_v <= {pipe_v[PIPE-2:0], rd_fire};
    cnt <= {cnt[PIPE-1:1], cnt0};
    act <= {act[PIPE-1:1], {act1, act0}};
  end

  // stage 5: the lane words, registered on this side of the MAC array
  always_ff @(posedge clk) dq5 <= drd_data;

  // stage 6: lane sum
  always_ff @(posedge clk)
    for (int l = 0; l < E; l++)
      sum6[l] <= (act[5][0] ? dq5[0][l] : 32'd0) + (act[5][1] ? dq5[1][l] : 32'd0);

  // stages 7-8: GEMV tap reduction (sums of 2 / 4 / 8 neighbouring lanes)
  logic [3:0][31:0] t1;
  always_comb for (int g = 0; g < 4; g++) t1[g] = sum6[2*g] + sum6[2*g+1];
  always_ff @(posedge clk) begin
    sum7 <= sum6;
    t1_7 <= t1;
    for (int g = 0; g < 2; g++) t2_7[g] <= t1[2*g] + t1[2*g+1];
  end
  always_ff @(posedge clk) begin
    col8 <= sum7;
    case (cfg.lk)
      2'd1: for (int g = 0; g < 4; g++) col8[g] <= t1_7[g];
      2'd2: for (int g = 0; g < 2; g++) col8[g] <= t2_7[g];
      2'd3: col8[0] <= t2_7[0] + t2_7[1];
      default: ;
    endcase
  end

  // stage 9: floor(acc / 256), saturated to Q8.8
  always_ff @(posedge clk)
    for (int l = 0; l < E; l++) begin
      if (col8[l][31:23] == 9'h000 || col8[l][31:23] == 9'h1FF) sat9[l] <= col8[l][23:8];
      else sat9[l] <= col8[l][31] ? 16'h8000 : 16'h7FFF;
    end

  assign of_in_valid = pipe_v[PIPE-1];   // sat9 of a request made PIPE cycles ago

  mm_fifo #(.W(BW + 4), .D(OD), .BRAM(1'b0)) u_of (
    .clk, .rst,
    .in_valid (of_in_valid), .in_ready (of_in_ready), .in_data ({cnt[PIPE], sat9}),
    .out_valid(el_valid),    .out_ready(el_ready),    .out_data({el_cnt, el_data}),
    .count    (of_count)
  );

  assign idle = (state == S_IDLE) && (pipe_v == '0) && !el_valid;

  logic unused;
  assign unused = of_in_ready;

endmodule
