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
  output logic [31:0] n_steps      // steps emitted so far
);
  logic        run;
  logic [31:0] bi, n_off, c0;
  logic [63:0] a_bi, b_bi, c_bi;    // batch slice bases
  logic [63:0] a_pn, c_pn;          // panel bases
  logic [63:0] c_st;                // chunk base
  logic        reuse;               // A is identical for every batch slice

  logic [31:0] n_left, m_left;
  logic        last_bi, last_pn, last_ch;
  assign n_left  = cfg.n - n_off;
  assign m_left  = cfg.m - c0;
  assign last_bi = (bi + 32'd1 >= cfg.batch);
  assign last_pn = (n_left <= R);
  assign last_ch = (m_left <= {22'b0, cfg.mc_max});

  always_comb begin
    step.a_addr      = a_pn;
    step.b_addr      = b_bi;
    step.c_addr      = c_st;
    step.c0          = c0;
    step.mcc         = last_ch ? m_left[9:0] : cfg.mc_max;
    step.n_valid     = last_pn ? n_left[3:0] : 4'(R);
    step.panel_first = (c0 == '0);
    step.panel_last  = last_ch;
    step.a_load      = (c0 == '0) && !(reuse && bi != '0);
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
      bi      <= '0;
      n_off   <= '0;
      c0      <= '0;
      a_bi    <= a_base;  a_pn <= a_base;
      b_bi    <= b_base;
      c_bi    <= c_base;  c_pn <= c_base;  c_st <= c_base;
      reuse   <= (a_stride == '0) && (cfg.n <= R);
    end else if (run && step_ready) begin
      n_steps <= n_steps + 32'd1;
      if (!last_ch) begin
        c0   <= c0 + {22'b0, cfg.mc_max};
        c_st <= c_st + {53'b0, cfg.mc_max, 1'b0};
      end else begin
        c0 <= '0;
        if (!last_pn) begin
          n_off <= n_off + 32'(R);
          // next panel: R rows further in A (R * k elements) and C (R * m)
          a_pn  <= a_pn + {50'b0, cfg.k, 1'b0} * 64'(R);
          c_pn  <= c_pn + {31'b0, cfg.m, 1'b0} * 64'(R);
          c_st  <= c_pn + {31'b0, cfg.m, 1'b0} * 64'(R);
        end else begin
          n_off <= '0;
          if (!last_bi) begin
            bi   <= bi + 32'd1;
            a_bi <= a_bi + {31'b0, a_stride, 1'b0};
            a_pn <= a_bi + {31'b0, a_stride, 1'b0};
            b_bi <= b_bi + {31'b0, b_stride, 1'b0};
            c_bi <= c_bi + {31'b0, c_stride, 1'b0};
            c_pn <= c_bi + {31'b0, c_stride, 1'b0};
            c_st <= c_bi + {31'b0, c_stride, 1'b0};
          end else begin
            run  <= 1'b0;
            done <= 1'b1;
          end
        end
      end
    end
  end

endmodule
