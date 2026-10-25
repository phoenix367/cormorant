// Copyright 2026 Ivan Gubochkin
// SPDX-License-Identifier: Apache-2.0

// ---------------------------------------------------------------------------
// mm_walker — walks the job's loop nest and emits one step descriptor per
// (batch slice, panel of R rows, column chunk):
//
//   for bi < batch:  for n_off < n step R:  for c0 < m step mc_max:  step
//
// Every step goes to both run generators and to the drain (three FIFOs, all
// pushed together).  Byte addresses advance incrementally (no multipliers).
// ---------------------------------------------------------------------------
module mm_walker
  import mm_pkg::*;
(
  input  logic        clk,
  input  logic        rst,

  input  logic        start,       // pulse: cfg and base addresses are valid
  input  cfg_t        cfg,
  input  logic [63:0] a_base,
  input  logic [63:0] b_base,
  input  logic [63:0] c_base,
  input  logic [31:0] a_stride,    // elements
  input  logic [31:0] b_stride,
  input  logic [31:0] c_stride,

  output logic        step_valid,
  input  logic        step_ready,  // all three consumers have room
  output step_t       step,

  output logic        done,        // all steps emitted
  output logic [HS-1:0] n_steps    // steps emitted so far (mod 2^HS)
);
  logic        run;
  logic [31:0] bi_left, n_left, m_left;  // batch slices / panel rows / chunk columns left, the current one included
  logic        ch_first, bi_first;  // the panel's first chunk, the job's first batch slice
  logic [AB-1:0] a_bi, b_bi, c_bi;  // batch slice bases
  logic [AB-1:0] a_pn, c_pn;        // panel bases
  logic [AB-1:0] b_ch, c_st;        // chunk bases
  logic [AB-1:0] b_inc;             // B bytes from a chunk to the next: packed 16 tiles of 32 x k, image mc_max << lk
  logic        reuse;               // A is identical for every batch slice

  logic        last_bi, last_pn, last_ch;
  assign last_bi = (bi_left <= 32'd1);
  assign last_pn = (n_left <= R);
  assign last_ch = (m_left <= {22'b0, cfg.mc_max});

  always_comb begin
    step.a_addr      = a_pn;
    step.b_addr      = b_ch;
    step.c_addr      = c_st;
    step.mcc         = last_ch ? m_left[9:0] : cfg.mc_max;
    step.n_valid     = last_pn ? n_left[3:0] : 4'(R);
    step.panel_first = ch_first;
    step.panel_last  = last_ch;
    step.a_load      = ch_first && !(reuse && !bi_first);
    step.last        = last_bi && last_pn && last_ch;
  end

  assign step_valid = run;

  always_ff @(posedge clk) begin
    if (rst) begin
      run     <= 1'b0;
      done    <= 1'b0;
      n_steps <= '0;
    end else if (start) begin
      run     <= (cfg.batch != '0) && (cfg.n != '0) && (cfg.m != '0);
      done    <= (cfg.batch == '0) || (cfg.n == '0) || (cfg.m == '0);
      n_steps <= '0;
      bi_left <= cfg.batch;
      n_left  <= cfg.n;
      m_left  <= cfg.m;
      ch_first <= 1'b1;
      bi_first <= 1'b1;
      a_bi    <= a_base[AB-1:0];  a_pn <= a_base[AB-1:0];
      b_bi    <= b_base[AB-1:0];  b_ch <= b_base[AB-1:0];
      b_inc   <= cfg.pk ? AB'(cfg.k) << 10 : AB'(cfg.mc_max) << ({2'b0, cfg.lk} + 4'd1);
      c_bi    <= c_base[AB-1:0];  c_pn <= c_base[AB-1:0];  c_st <= c_base[AB-1:0];
      reuse   <= (a_stride == '0) && (cfg.n <= R);
    end else if (run && step_ready) begin
      n_steps <= n_steps + HS'(1);
      if (!last_ch) begin
        m_left   <= m_left - {22'b0, cfg.mc_max};
        ch_first <= 1'b0;
        b_ch     <= b_ch + b_inc;
        c_st     <= c_st + AB'({cfg.mc_max, 1'b0});
      end else begin
        m_left   <= cfg.m;
        ch_first <= 1'b1;
        b_ch     <= b_bi;
        if (!last_pn) begin
          n_left <= n_left - 32'(R);
          // next panel: R rows further in A (R * k elements) and C (R * m)
          a_pn  <= a_pn + AB'({cfg.k, 1'b0}) * AB'(R);
          c_pn  <= c_pn + AB'({cfg.m, 1'b0}) * AB'(R);
          c_st  <= c_pn + AB'({cfg.m, 1'b0}) * AB'(R);
        end else begin
          n_left <= cfg.n;
          if (!last_bi) begin
            bi_left  <= bi_left - 32'd1;
            bi_first <= 1'b0;
            a_bi <= a_bi + AB'({a_stride, 1'b0});
            a_pn <= a_bi + AB'({a_stride, 1'b0});
            b_bi <= b_bi + AB'({b_stride, 1'b0});
            b_ch <= b_bi + AB'({b_stride, 1'b0});
            c_bi <= c_bi + AB'({c_stride, 1'b0});
            c_pn <= c_bi + AB'({c_stride, 1'b0});
            c_st <= c_bi + AB'({c_stride, 1'b0});
          end else begin
            run  <= 1'b0;
            done <= 1'b1;
          end
        end
      end
    end
  end

endmodule
