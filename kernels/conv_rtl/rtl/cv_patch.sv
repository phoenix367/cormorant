// ---------------------------------------------------------------------------
// cv_patch — the line buffer and the patch producer (the HLS
// input_patch_producer).
//
// Line buffer: TIC channel banks of LBR rows x LBC columns (block RAM, row
// ih mod LBR, column iw mod LBC), two copies per bank so that loads and
// window reads never share a port: each copy is written through its port A
// (one column of all channels per cycle from the x loader) and read through
// its port B (copy 0 for pixel 0 of a pair, copy 1 for pixel 1).
//
// Two sequencers walk the sweep's output rows.  The writer loads, per
// output row, the input rows the window still lacks (a loading sweep only);
// the emitter reads, per pixel pair and window position, one beat of the two
// pixels' TIC-lane columns, once the writer has finished that output row.
// The writer may run one output row ahead of the emitter when the rows it
// then writes cannot alias the rows the emitter reads (stride_h + reach_h
// < LBR: the loads of output row e + 1 overlap the emission of row e —
// DEPTHWISE_PLAN option B3); otherwise it waits, as the HLS producer did.
// Lanes outside the input, past the tile's channels, or of a pixel outside
// the ow-tile are zero.
// ---------------------------------------------------------------------------
module cv_patch
  import cv_pkg::*;
(
  input  logic          clk,
  input  logic          rst,
  input  job_t          j,

  input  logic          pq_valid,
  output logic          pq_ready,
  input  sweep_t        pq,

  input  logic          col_valid,
  output logic          col_ready,
  input  logic [TIC*EW-1:0] col_data,

  output logic          pt_valid,
  input  logic          pt_ready,
  output logic [PATCH_W-1:0] pt_data,

  output logic          idle
);
  localparam int PD = 64;                  // patch FIFO depth

  typedef enum logic [1:0] {W_IDLE, W_ROW, W_LOAD} wst_t;
  typedef enum logic [1:0] {E_IDLE, E_ROW, E_EMIT} est_t;
  wst_t        ws;
  est_t        es;
  sweep_t      sd;
  logic        ovl;                        // the writer may run a row ahead

  // writer
  logic [12:0] w_oh, w_done;               // row being prepared; rows resident
  logic [31:0] w_last, w_ihmin, w_ih, w_le;
  logic [1:0]  w_settle;                   // ls / le valid after two cycles
  logic [6:0]  w_j;
  logic [5:0]  w_col;                      // (iw_lo + w_j) mod LBC
  logic [31:0] inh1;                       // in_h - 1

  // emitter
  logic [12:0] e_oh;
  logic [31:0] e_ihmin;
  logic [12:0] e_pair, e_owa;
  logic [2:0]  e_khi, e_kwi;
  logic [31:0] e_ih, e_iwb, e_iw0;        // pixel 0's: pair base, current column
  logic [31:0] e_iwb1, e_iw1;              // pixel 1's (+ stride_w)

  assign pq_ready = (ws == W_IDLE) && (es == E_IDLE);

  // The window of output row w_oh decides about: input rows [ls, le] =
  // [max(last + 1, ihmin, 0), min(ihmin + reach_h, in_h - 1)].  Two
  // pipeline stages from w_ihmin / w_last (w_settle).
  logic [31:0] s1_ih, s1_le, s1_last1, ls, le;
  always_ff @(posedge clk) begin
    s1_ih    <= w_ihmin;
    s1_le    <= w_ihmin + {28'b0, j.reach_h};
    s1_last1 <= w_last + 32'd1;
    ls       <= ($signed(s1_last1) > $signed(s1_ih)) ? (s1_last1[31] ? '0 : s1_last1)
                                                     : (s1_ih[31] ? '0 : s1_ih);
    le       <= ($signed(s1_le) >= $signed(j.in_h)) ? inh1 : s1_le;
  end

  // patch FIFO room: entries + beats in the read pipeline
  logic [$clog2(PD):0] pf_n;
  logic [1:0]          inflight;
  logic                e_go, l_go, w_may;
  // the writer's row may be loaded: the emitter waits for it (same row), or
  // emits the row before and the new rows alias none of its window's
  assign w_may     = (w_oh == e_oh) || (ovl && (w_oh == e_oh + 13'd1));
  assign l_go      = (ws == W_LOAD) && col_valid;
  assign col_ready = (ws == W_LOAD);
  assign e_go      = (es == E_EMIT) && (32'(pf_n) + 32'(inflight) < PD - 1);

  logic e_last_beat;
  assign e_last_beat = (e_kwi == j.kw - 3'd1) && (e_khi == j.kh - 3'd1) &&
                       (e_pair + 13'd1 == sd.n_pairs);

  logic accept;
  assign accept = pq_valid && (ws == W_IDLE) && (es == E_IDLE);

  // ---- the writer ------------------------------------------------------------------
  always_ff @(posedge clk) begin
    if (rst) begin
      ws <= W_IDLE;
    end else begin
      case (ws)
        W_IDLE: if (accept) begin
          sd       <= pq;
          ovl      <= (j.sh + {28'b0, j.reach_h}) < 32'(LBR);
          w_oh     <= '0;
          w_last   <= pq.ih0 - 32'd1;
          w_ihmin  <= pq.ih0;
          inh1     <= j.in_h - 32'd1;
          w_settle <= 2'd2;
          if (pq.load && pq.iw_cnt != '0) begin
            w_done <= '0;
            ws     <= W_ROW;
          end else begin
            w_done <= pq.coh;                // the rows are resident already
          end
        end
        W_ROW: begin
          if (w_settle != 2'd0) begin
            w_settle <= w_settle - 2'd1;
          end else if (w_may) begin
            if ($signed(ls) <= $signed(le)) begin
              w_ih  <= ls;
              w_le  <= le;
              w_j   <= '0;
              w_col <= sd.iw_lo[5:0];
              ws    <= W_LOAD;
            end else begin
              // nothing new; the resident-row bound still follows the HLS loader
              if ($signed(le) > $signed(w_last)) w_last <= le;
              w_done   <= w_done + 13'd1;
              w_oh     <= w_oh + 13'd1;
              w_ihmin  <= w_ihmin + j.sh;
              w_settle <= 2'd2;
              if (w_oh + 13'd1 == sd.coh) ws <= W_IDLE;
            end
          end
        end
        W_LOAD: if (l_go) begin
          w_col <= w_col + 6'd1;
          if (w_j + 7'd1 == sd.iw_cnt) begin
            w_j   <= '0;
            w_col <= sd.iw_lo[5:0];
            w_ih  <= w_ih + 32'd1;
            if (w_ih == w_le) begin
              w_last   <= w_le;
              w_done   <= w_done + 13'd1;
              w_oh     <= w_oh + 13'd1;
              w_ihmin  <= w_ihmin + j.sh;
              w_settle <= 2'd2;
              ws       <= (w_oh + 13'd1 == sd.coh) ? W_IDLE : W_ROW;
            end
          end else begin
            w_j <= w_j + 7'd1;
          end
        end
        default: ws <= W_IDLE;
      endcase
    end
  end

  // ---- the emitter -----------------------------------------------------------------
  always_ff @(posedge clk) begin
    if (rst) begin
      es <= E_IDLE;
    end else begin
      case (es)
        E_IDLE: if (accept) begin
          e_oh    <= '0;
          e_ihmin <= pq.ih0;
          es      <= E_ROW;
        end
        E_ROW: begin
          e_pair <= '0;
          e_owa  <= sd.pair_base;
          e_khi  <= '0;
          e_kwi  <= '0;
          e_ih   <= e_ihmin;
          e_iwb  <= sd.iwb0;
          e_iw0  <= sd.iwb0;
          e_iwb1 <= sd.iwb0 + j.sw;
          e_iw1  <= sd.iwb0 + j.sw;
          if (w_done > e_oh) es <= E_EMIT;
        end
        E_EMIT: if (e_go) begin
          if (e_kwi != j.kw - 3'd1) begin
            e_kwi <= e_kwi + 3'd1;
            e_iw0 <= e_iw0 + j.dw;
            e_iw1 <= e_iw1 + j.dw;
          end else begin
            e_kwi <= '0;
            if (e_khi != j.kh - 3'd1) begin
              e_khi <= e_khi + 3'd1;
              e_ih  <= e_ih + j.dh;
              e_iw0 <= e_iwb;
              e_iw1 <= e_iwb1;
            end else begin
              e_khi  <= '0;
              e_ih   <= e_ihmin;
              e_pair <= e_pair + 13'd1;
              e_owa  <= e_owa + 13'd2;
              e_iwb  <= e_iwb + {j.sw[30:0], 1'b0};
              e_iw0  <= e_iwb + {j.sw[30:0], 1'b0};
              e_iwb1 <= e_iwb1 + {j.sw[30:0], 1'b0};
              e_iw1  <= e_iwb1 + {j.sw[30:0], 1'b0};
            end
          end
          if (e_last_beat) begin
            e_oh    <= e_oh + 13'd1;
            e_ihmin <= e_ihmin + j.sh;
            es      <= (e_oh + 13'd1 == sd.coh) ? E_IDLE : E_ROW;
          end
        end
        default: es <= E_IDLE;
      endcase
    end
  end

  // ---- the read: addresses and masks of this beat --------------------------------
  logic        ih_ok, ok0, ok1;
  assign ih_ok = !e_ih[31] && (e_ih < j.in_h);
  assign ok0   = ih_ok && (e_owa >= sd.ow_start) && !e_iw0[31] && (e_iw0 < j.in_w);
  assign ok1   = ih_ok && (e_owa + 13'd1 < sd.ow_end) && !e_iw1[31] && (e_iw1 < j.in_w);

  logic [9:0] w_addr, a_addr, b_addr;
  assign w_addr = {w_ih[3:0], w_col};
  assign a_addr = {e_ih[3:0], e_iw0[5:0]};
  assign b_addr = {e_ih[3:0], e_iw1[5:0]};

  // The banks are spread over several block-RAM columns: their ports take a
  // register stage (the write address, enable and the two read addresses a
  // copy per LBG banks — each copy drives both copies of its banks), the
  // reads the RAM's output register — a beat is pushed three cycles after
  // its e_go, a column written one cycle after its l_go (writes and reads
  // keep their order: the emitter starts a row only after its writes went in).
  localparam int LBG = 2;
  (* keep = "true" *) logic [9:0] pw_addr [TIC / LBG];
  (* keep = "true" *) logic [9:0] pa_addr [TIC / LBG];
  (* keep = "true" *) logic [9:0] pb_addr [TIC / LBG];
  (* keep = "true" *) logic       pw_we   [TIC / LBG];
  logic [TIC*EW-1:0] pw_wd;
  always_ff @(posedge clk) begin
    for (int g = 0; g < TIC / LBG; g++) begin
      pw_addr[g] <= w_addr;
      pa_addr[g] <= a_addr;
      pb_addr[g] <= b_addr;
      pw_we[g]   <= l_go && !rst;
    end
    pw_wd <= col_data;
  end

  logic [TIC*EW-1:0] rd_a, rd_b;
  for (genvar c = 0; c < TIC; c++) begin : g_lb
    localparam int G = c / LBG;
    (* ram_style = "block" *) logic [EW-1:0] mem0 [LBR * LBC];   // read by pixel 0
    (* ram_style = "block" *) logic [EW-1:0] mem1 [LBR * LBC];   // read by pixel 1
    logic [EW-1:0] ra0, rb0;
    always_ff @(posedge clk) begin
      if (pw_we[G]) mem0[pw_addr[G]] <= pw_wd[c * EW +: EW];
      ra0 <= mem0[pa_addr[G]];
    end
    always_ff @(posedge clk) begin
      if (pw_we[G]) mem1[pw_addr[G]] <= pw_wd[c * EW +: EW];
      rb0 <= mem1[pb_addr[G]];
    end
    always_ff @(posedge clk) begin           // the RAMs' output registers
      rd_a[c * EW +: EW] <= ra0;
      rd_b[c * EW +: EW] <= rb0;
    end
  end

  // masks travel with the read (three cycles), then the beat is pushed
  logic        r_v1, r_v2, r_v, r_ok0, r_ok1;
  logic [1:0]  r_ok0_d, r_ok1_d;
  logic [4:0]  r_chv, r_chv1, r_chv2;
  always_ff @(posedge clk) begin
    if (rst) {r_v1, r_v2, r_v} <= '0;
    else     {r_v1, r_v2, r_v} <= {e_go, r_v1, r_v2};
    r_ok0_d <= {r_ok0_d[0], ok0};
    r_ok1_d <= {r_ok1_d[0], ok1};
    r_ok0   <= r_ok0_d[1];
    r_ok1   <= r_ok1_d[1];
    r_chv1  <= sd.ch_valid;
    r_chv2  <= r_chv1;
    r_chv   <= r_chv2;
  end
  assign inflight = 2'(r_v1) + 2'(r_v2) + 2'(r_v);

  logic [PATCH_W-1:0] beat;
  always_comb
    for (int c = 0; c < TIC; c++) begin
      logic ch_ok;
      ch_ok = (5'(c) < r_chv);
      beat[c * EW +: EW]         = (ch_ok && r_ok0) ? rd_a[c * EW +: EW] : '0;
      beat[(TIC + c) * EW +: EW] = (ch_ok && r_ok1) ? rd_b[c * EW +: EW] : '0;
    end

  logic pf_in_ready;
  cv_fifo #(.W(PATCH_W), .D(PD), .BRAM(1'b0)) u_pf (
    .clk, .rst, .in_valid (r_v), .in_ready (pf_in_ready), .in_data (beat),
    .out_valid (pt_valid), .out_ready (pt_ready), .out_data (pt_data), .count (pf_n)
  );

  assign idle = (ws == W_IDLE) && (es == E_IDLE) && !r_v1 && !r_v2 && !r_v && !pt_valid;

  logic unused;
  assign unused = pf_in_ready ^ ^j ^ ^sd;

endmodule
