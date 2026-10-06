// ---------------------------------------------------------------------------
// cv_patch — the line buffer and the patch producer (the HLS
// input_patch_producer).
//
// Line buffer: TIC channel banks of LBR rows x LBC columns (block RAM, row
// ih mod LBR, column iw mod LBC).  Per output row of a sweep: when the sweep
// loads, the input rows the window still lacks are written from the x
// loader's columns (one column of all channels per cycle, port A); then, per
// pixel pair and window position, one beat of the two pixels' TIC-lane
// columns is read (pixel 0 through port A, pixel 1 through port B).  Lanes
// outside the input, past the tile's channels, or of a pixel outside the
// ow-tile are zero.
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

  typedef enum logic [2:0] {P_IDLE, P_PRE, P_ROW, P_LOAD, P_EMIT} pst_t;
  pst_t        st;
  sweep_t      sd;
  logic [12:0] p_oh;
  logic [31:0] p_last, p_ihmin;
  logic [31:0] p_ihn, p_len, p_lenn;       // ihmin + sh, ihmin + reach_h, + sh
  logic [31:0] inh1;                       // in_h - 1
  logic        pre2;                       // second P_PRE cycle
  logic        e_age;                      // P_EMIT has run a cycle already

  // load
  logic [31:0] l_ih, l_le;
  logic [6:0]  l_j;
  logic [5:0]  l_col;                      // (iw_lo + l_j) mod LBC

  // emission
  logic [12:0] e_pair, e_owa;
  logic [2:0]  e_khi, e_kwi;
  logic [31:0] e_ih, e_iwb, e_iw0;        // pixel 0's: pair base, current column
  logic [31:0] e_iwb1, e_iw1;              // pixel 1's (+ stride_w)

  assign pq_ready = (st == P_IDLE);

  // The window of the output row P_ROW decides about: input rows [ls, le] =
  // [max(last + 1, ihmin, 0), min(ihmin + reach_h, in_h - 1)].  Two pipeline
  // stages; during P_EMIT they work on the next row (ihmin + sh), which the
  // row's emission leaves time for when it takes two cycles or more, so P_ROW
  // finds them ready; a sweep's first row, and the row after a one-beat
  // emission (a 1 x 1 kernel, one pixel pair), wait in P_PRE instead.
  logic [31:0] s1_ih, s1_le, s1_last1, ls, le;
  always_ff @(posedge clk) begin
    e_age    <= (st == P_EMIT);
    s1_ih    <= (st == P_EMIT) ? p_ihn : p_ihmin;
    s1_le    <= (st == P_EMIT) ? p_lenn : p_len;
    s1_last1 <= p_last + 32'd1;
    ls       <= ($signed(s1_last1) > $signed(s1_ih)) ? (s1_last1[31] ? '0 : s1_last1)
                                                     : (s1_ih[31] ? '0 : s1_ih);
    le       <= ($signed(s1_le) >= $signed(j.in_h)) ? inh1 : s1_le;
  end

  // patch FIFO room: entries + beats in the read pipeline
  logic [$clog2(PD):0] pf_n;
  logic [1:0]          inflight;
  logic                e_go, l_go;
  assign l_go      = (st == P_LOAD) && col_valid;
  assign col_ready = l_go;
  assign e_go      = (st == P_EMIT) && (32'(pf_n) + 32'(inflight) < PD - 1);

  logic e_last_beat;
  assign e_last_beat = (e_kwi == j.kw - 3'd1) && (e_khi == j.kh - 3'd1) &&
                       (e_pair + 13'd1 == sd.n_pairs);

  always_ff @(posedge clk) begin
    if (rst) begin
      st <= P_IDLE;
    end else begin
      case (st)
        P_IDLE: if (pq_valid) begin
          sd      <= pq;
          p_oh    <= '0;
          p_last  <= pq.ih0 - 32'd1;
          p_ihmin <= pq.ih0;
          p_ihn   <= pq.ih0 + j.sh;
          p_len   <= pq.ih0 + {28'b0, j.reach_h};
          inh1    <= j.in_h - 32'd1;
          pre2    <= 1'b0;
          st      <= P_PRE;
        end
        P_PRE: begin
          p_lenn <= p_len + j.sh;
          pre2   <= 1'b1;
          if (pre2) st <= P_ROW;
        end
        P_ROW: begin
          // the output row's new input rows (only a loading sweep loads)
          e_pair <= '0;
          e_owa  <= sd.pair_base;
          e_khi  <= '0;
          e_kwi  <= '0;
          e_ih   <= p_ihmin;
          e_iwb  <= sd.iwb0;
          e_iw0  <= sd.iwb0;
          e_iwb1 <= sd.iwb0 + j.sw;
          e_iw1  <= sd.iwb0 + j.sw;
          if (sd.load && sd.iw_cnt != '0 && $signed(ls) <= $signed(le)) begin
            l_ih  <= ls;
            l_le  <= le;
            l_j   <= '0;
            l_col <= sd.iw_lo[5:0];
            st    <= P_LOAD;
          end else begin
            if (sd.load && $signed(le) > $signed(p_last)) p_last <= le;
            st <= P_EMIT;
          end
        end
        P_LOAD: if (l_go) begin
          l_col <= l_col + 6'd1;
          if (l_j + 7'd1 == sd.iw_cnt) begin
            l_j   <= '0;
            l_col <= sd.iw_lo[5:0];
            l_ih  <= l_ih + 32'd1;
            if (l_ih == l_le) begin
              p_last <= l_le;
              st     <= P_EMIT;
            end
          end else begin
            l_j <= l_j + 7'd1;
          end
        end
        P_EMIT: if (e_go) begin
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
              e_ih   <= p_ihmin;
              e_pair <= e_pair + 13'd1;
              e_owa  <= e_owa + 13'd2;
              e_iwb  <= e_iwb + {j.sw[30:0], 1'b0};
              e_iw0  <= e_iwb + {j.sw[30:0], 1'b0};
              e_iwb1 <= e_iwb1 + {j.sw[30:0], 1'b0};
              e_iw1  <= e_iwb1 + {j.sw[30:0], 1'b0};
            end
          end
          if (e_last_beat) begin
            p_oh    <= p_oh + 13'd1;
            p_ihmin <= p_ihn;
            p_ihn   <= p_ihn + j.sh;
            p_len   <= p_lenn;
            p_lenn  <= p_lenn + j.sh;
            pre2    <= 1'b0;
            st      <= (p_oh + 13'd1 == sd.coh) ? P_IDLE : (e_age ? P_ROW : P_PRE);
          end
        end
        default: st <= P_IDLE;
      endcase
    end
  end

  // ---- the read: addresses and masks of this beat --------------------------------
  logic        ih_ok, ok0, ok1;
  assign ih_ok = !e_ih[31] && (e_ih < j.in_h);
  assign ok0   = ih_ok && (e_owa >= sd.ow_start) && !e_iw0[31] && (e_iw0 < j.in_w);
  assign ok1   = ih_ok && (e_owa + 13'd1 < sd.ow_end) && !e_iw1[31] && (e_iw1 < j.in_w);

  logic [9:0] a_addr, b_addr;
  assign a_addr = (st == P_LOAD) ? {l_ih[3:0], l_col} : {e_ih[3:0], e_iw0[5:0]};
  assign b_addr = {e_ih[3:0], e_iw1[5:0]};

  // The banks are spread over several block-RAM columns: their ports take a
  // register stage (the address and write enable a copy per LBG banks), the
  // reads the RAM's output register — a beat is pushed three cycles after
  // its e_go, a column written one cycle after its l_go (writes and reads
  // keep their order).
  localparam int LBG = 4;
  (* keep = "true" *) logic [9:0] pa_addr [TIC / LBG];
  (* keep = "true" *) logic [9:0] pb_addr [TIC / LBG];
  (* keep = "true" *) logic       pa_we   [TIC / LBG];
  logic [TIC*EW-1:0] pa_wd;
  always_ff @(posedge clk) begin
    for (int g = 0; g < TIC / LBG; g++) begin
      pa_addr[g] <= a_addr;
      pb_addr[g] <= b_addr;
      pa_we[g]   <= l_go && !rst;
    end
    pa_wd <= col_data;
  end

  logic [TIC*EW-1:0] rd_a, rd_b;
  for (genvar c = 0; c < TIC; c++) begin : g_lb
    localparam int G = c / LBG;
    (* ram_style = "block" *) logic [EW-1:0] mem [LBR * LBC];
    logic [EW-1:0] ra0, rb0;
    always_ff @(posedge clk) begin
      if (pa_we[G]) mem[pa_addr[G]] <= pa_wd[c * EW +: EW];
      ra0 <= mem[pa_addr[G]];
    end
    always_ff @(posedge clk) rb0 <= mem[pb_addr[G]];
    always_ff @(posedge clk) begin           // the RAM's output registers
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

  assign idle = (st == P_IDLE) && !r_v1 && !r_v2 && !r_v && !pt_valid;

  logic unused;
  assign unused = pf_in_ready ^ ^j ^ ^sd;

endmodule
