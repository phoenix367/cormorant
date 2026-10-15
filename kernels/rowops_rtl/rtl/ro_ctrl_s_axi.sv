// ---------------------------------------------------------------------------
// ro_ctrl_s_axi — AXI4-Lite control slave of the RowOpsKernel: the HLS
// register-block layout and ap_ctrl_hs behaviour (as the other RTL kernels'
// control slaves, derived from the MatmulKernel's mm_ctrl_s_axi):
//
//   0x00 ctrl  b0 ap_start (R/W, cleared on ap_ready unless auto_restart)
//              b1 ap_done (clear on read)  b2 ap_idle  b3 ap_ready (COR)
//              b7 auto_restart  b9 interrupt
//   0x04 GIE   0x08 IER (b0 done, b1 ready)   0x0C ISR (toggle on write)
//   0x10 op    0x14 n      0x18 rows (c1)  0x1C blocks (c2)
//   0x20/24 a  0x28 a_s1   0x2C a_s2
//   0x30/34 b  0x38 b_s1   0x3C b_s2
//   0x40/44 c  0x48 c_s1   0x4C c_s2
//   0x50/54 p  0x58/5C eps 0x60 aux  0x64 bq_sh  0x68 win_lo  0x6C win_hi
//   (doc/kernels/ROWOPS_RTL_KERNEL.md §1.2; scripts/gen_driver.py checks the table)
//
// Like the HLS slave, AW and W are taken one after the other and every
// access gets an OKAY response.
// ---------------------------------------------------------------------------
module ro_ctrl_s_axi (
  input  logic        clk,
  input  logic        rst,

  input  logic        awvalid,
  output logic        awready,
  input  logic [6:0]  awaddr,
  input  logic        wvalid,
  output logic        wready,
  input  logic [31:0] wdata,
  input  logic [3:0]  wstrb,
  output logic        bvalid,
  input  logic        bready,
  output logic [1:0]  bresp,
  input  logic        arvalid,
  output logic        arready,
  input  logic [6:0]  araddr,
  output logic        rvalid,
  input  logic        rready,
  output logic [31:0] rdata,
  output logic [1:0]  rresp,

  output logic        interrupt,

  // Kernel handshake.
  output logic        ap_start,
  input  logic        ap_done,
  input  logic        ap_ready,
  input  logic        ap_idle,

  // Arguments.
  output logic [31:0] op,
  output logic [31:0] n,
  output logic [31:0] rows,
  output logic [31:0] blocks,
  output logic [63:0] a,
  output logic [31:0] a_s1,
  output logic [31:0] a_s2,
  output logic [63:0] b,
  output logic [31:0] b_s1,
  output logic [31:0] b_s2,
  output logic [63:0] c,
  output logic [31:0] c_s1,
  output logic [31:0] c_s2,
  output logic [63:0] p,
  output logic [63:0] eps,
  output logic [31:0] aux,
  output logic [31:0] bq_sh,
  output logic [31:0] win_lo,
  output logic [31:0] win_hi
);
  localparam logic [6:0] A_CTRL = 7'h00, A_GIE = 7'h04, A_IER = 7'h08, A_ISR = 7'h0C,
                         A_OP = 7'h10, A_N = 7'h14, A_ROWS = 7'h18, A_BLOCKS = 7'h1C,
                         A_A0 = 7'h20, A_A1 = 7'h24, A_AS1 = 7'h28, A_AS2 = 7'h2C,
                         A_B0 = 7'h30, A_B1 = 7'h34, A_BS1 = 7'h38, A_BS2 = 7'h3C,
                         A_C0 = 7'h40, A_C1 = 7'h44, A_CS1 = 7'h48, A_CS2 = 7'h4C,
                         A_P0 = 7'h50, A_P1 = 7'h54, A_E0 = 7'h58, A_E1 = 7'h5C,
                         A_AUX = 7'h60, A_BQSH = 7'h64, A_WLO = 7'h68, A_WHI = 7'h6C;

  // Write channel ------------------------------------------------------------------
  // Every AXI-Lite input is registered before it is decoded: the address and
  // the write data are captured on their handshakes and the register update
  // (w_hs) happens the cycle after the W handshake, together with BVALID; a
  // read address is decoded the cycle after its handshake (RD_ADDR).
  typedef enum logic [1:0] {WR_IDLE, WR_DATA, WR_RESP} wst_t;
  wst_t        wst;
  logic [6:0]  waddr;
  logic        w_hs;                 // registered W handshake: apply wdata / wstrb
  logic [31:0] wdata_q;
  logic [3:0]  wstrb_q;

  assign awready = (wst == WR_IDLE);
  assign wready  = (wst == WR_DATA);
  assign bvalid  = (wst == WR_RESP);
  assign bresp   = 2'b00;

  always_ff @(posedge clk) begin
    if (rst) begin
      wst  <= WR_IDLE;
      w_hs <= 1'b0;
    end else begin
      w_hs <= wvalid && wready;
      case (wst)
        WR_IDLE: if (awvalid) wst <= WR_DATA;
        WR_DATA: if (wvalid)  wst <= WR_RESP;
        WR_RESP: if (bready)  wst <= WR_IDLE;
        default: wst <= WR_IDLE;
      endcase
    end
    if (awvalid && awready) waddr <= {awaddr[6:2], 2'b00};
    if (wvalid && wready) begin
      wdata_q <= wdata;
      wstrb_q <= wstrb;
    end
  end



  // Control / status -------------------------------------------------------------
  logic       int_ap_start, int_task_done, int_ap_idle, int_ap_ready;
  logic       int_auto_restart, auto_restart_status, int_gie, int_irq;
  logic [1:0] int_ier, int_isr;
  logic       ar_hs;               // registered AR handshake: decode raddr
  logic [6:0] raddr;
  logic       task_ap_done, task_ap_ready, auto_restart_done;

  assign auto_restart_done = auto_restart_status && ap_idle && !int_ap_idle;
  assign task_ap_done      = (ap_done && !auto_restart_status) || auto_restart_done;
  assign task_ap_ready     = ap_ready && !int_auto_restart;

  assign ap_start  = int_ap_start;
  assign interrupt = int_irq;

  always_ff @(posedge clk) begin
    if (rst) begin
      int_ap_start        <= 1'b0;
      int_task_done       <= 1'b0;
      int_ap_idle         <= 1'b0;
      int_ap_ready        <= 1'b0;
      int_auto_restart    <= 1'b0;
      auto_restart_status <= 1'b0;
      int_gie             <= 1'b0;
      int_ier             <= 2'b0;
      int_isr             <= 2'b0;
      int_irq             <= 1'b0;
      op <= '0; n <= '0; rows <= '0; blocks <= '0;
      a <= '0; a_s1 <= '0; a_s2 <= '0; b <= '0; b_s1 <= '0; b_s2 <= '0;
      c <= '0; c_s1 <= '0; c_s2 <= '0; p <= '0; eps <= '0; aux <= '0; bq_sh <= '0;
      win_lo <= '0; win_hi <= 32'hFFFFFFFF;
    end else begin
      int_irq <= int_gie && (|int_isr);

      if (w_hs && waddr == A_CTRL && wstrb_q[0] && wdata_q[0]) int_ap_start <= 1'b1;
      else if (ap_ready)                                  int_ap_start <= int_auto_restart;

      if (task_ap_done)                   int_task_done <= 1'b1;
      else if (ar_hs && raddr == A_CTRL)  int_task_done <= 1'b0;

      int_ap_idle <= ap_idle;

      if (task_ap_ready)                  int_ap_ready <= 1'b1;
      else if (ar_hs && raddr == A_CTRL)  int_ap_ready <= 1'b0;

      if (w_hs && waddr == A_CTRL && wstrb_q[0]) int_auto_restart <= wdata_q[7];

      if (int_auto_restart) auto_restart_status <= 1'b1;
      else if (ap_idle)     auto_restart_status <= 1'b0;

      if (w_hs && waddr == A_GIE && wstrb_q[0]) int_gie <= wdata_q[0];
      if (w_hs && waddr == A_IER && wstrb_q[0]) int_ier <= wdata_q[1:0];

      if (int_ier[0] && ap_done)                    int_isr[0] <= 1'b1;
      else if (w_hs && waddr == A_ISR && wstrb_q[0])  int_isr[0] <= int_isr[0] ^ wdata_q[0];
      if (int_ier[1] && ap_ready)                   int_isr[1] <= 1'b1;
      else if (w_hs && waddr == A_ISR && wstrb_q[0])  int_isr[1] <= int_isr[1] ^ wdata_q[1];

      if (w_hs) begin
        case (waddr)
          A_OP: for (int k_ = 0; k_ < 4; k_++) if (wstrb_q[k_]) op[8*k_ +: 8] <= wdata_q[8*k_ +: 8];
          A_N: for (int k_ = 0; k_ < 4; k_++) if (wstrb_q[k_]) n[8*k_ +: 8] <= wdata_q[8*k_ +: 8];
          A_ROWS: for (int k_ = 0; k_ < 4; k_++) if (wstrb_q[k_]) rows[8*k_ +: 8] <= wdata_q[8*k_ +: 8];
          A_BLOCKS: for (int k_ = 0; k_ < 4; k_++) if (wstrb_q[k_]) blocks[8*k_ +: 8] <= wdata_q[8*k_ +: 8];
          A_A0: for (int k_ = 0; k_ < 4; k_++) if (wstrb_q[k_]) a[8*k_ +: 8] <= wdata_q[8*k_ +: 8];
          A_A1: for (int k_ = 0; k_ < 4; k_++) if (wstrb_q[k_]) a[32 + 8*k_ +: 8] <= wdata_q[8*k_ +: 8];
          A_AS1: for (int k_ = 0; k_ < 4; k_++) if (wstrb_q[k_]) a_s1[8*k_ +: 8] <= wdata_q[8*k_ +: 8];
          A_AS2: for (int k_ = 0; k_ < 4; k_++) if (wstrb_q[k_]) a_s2[8*k_ +: 8] <= wdata_q[8*k_ +: 8];
          A_B0: for (int k_ = 0; k_ < 4; k_++) if (wstrb_q[k_]) b[8*k_ +: 8] <= wdata_q[8*k_ +: 8];
          A_B1: for (int k_ = 0; k_ < 4; k_++) if (wstrb_q[k_]) b[32 + 8*k_ +: 8] <= wdata_q[8*k_ +: 8];
          A_BS1: for (int k_ = 0; k_ < 4; k_++) if (wstrb_q[k_]) b_s1[8*k_ +: 8] <= wdata_q[8*k_ +: 8];
          A_BS2: for (int k_ = 0; k_ < 4; k_++) if (wstrb_q[k_]) b_s2[8*k_ +: 8] <= wdata_q[8*k_ +: 8];
          A_C0: for (int k_ = 0; k_ < 4; k_++) if (wstrb_q[k_]) c[8*k_ +: 8] <= wdata_q[8*k_ +: 8];
          A_C1: for (int k_ = 0; k_ < 4; k_++) if (wstrb_q[k_]) c[32 + 8*k_ +: 8] <= wdata_q[8*k_ +: 8];
          A_CS1: for (int k_ = 0; k_ < 4; k_++) if (wstrb_q[k_]) c_s1[8*k_ +: 8] <= wdata_q[8*k_ +: 8];
          A_CS2: for (int k_ = 0; k_ < 4; k_++) if (wstrb_q[k_]) c_s2[8*k_ +: 8] <= wdata_q[8*k_ +: 8];
          A_P0: for (int k_ = 0; k_ < 4; k_++) if (wstrb_q[k_]) p[8*k_ +: 8] <= wdata_q[8*k_ +: 8];
          A_P1: for (int k_ = 0; k_ < 4; k_++) if (wstrb_q[k_]) p[32 + 8*k_ +: 8] <= wdata_q[8*k_ +: 8];
          A_E0: for (int k_ = 0; k_ < 4; k_++) if (wstrb_q[k_]) eps[8*k_ +: 8] <= wdata_q[8*k_ +: 8];
          A_E1: for (int k_ = 0; k_ < 4; k_++) if (wstrb_q[k_]) eps[32 + 8*k_ +: 8] <= wdata_q[8*k_ +: 8];
          A_AUX: for (int k_ = 0; k_ < 4; k_++) if (wstrb_q[k_]) aux[8*k_ +: 8] <= wdata_q[8*k_ +: 8];
          A_BQSH: for (int k_ = 0; k_ < 4; k_++) if (wstrb_q[k_]) bq_sh[8*k_ +: 8] <= wdata_q[8*k_ +: 8];
          A_WLO: for (int k_ = 0; k_ < 4; k_++) if (wstrb_q[k_]) win_lo[8*k_ +: 8] <= wdata_q[8*k_ +: 8];
          A_WHI: for (int k_ = 0; k_ < 4; k_++) if (wstrb_q[k_]) win_hi[8*k_ +: 8] <= wdata_q[8*k_ +: 8];
          default: ;
        endcase
      end
    end
  end

  // Read channel -------------------------------------------------------------------
  typedef enum logic [1:0] {RD_IDLE, RD_ADDR, RD_DATA} rst_t;
  rst_t rst_q;

  assign arready = (rst_q == RD_IDLE);
  assign rvalid  = (rst_q == RD_DATA);
  assign rresp   = 2'b00;
  assign ar_hs   = (rst_q == RD_ADDR);

  always_ff @(posedge clk) begin
    if (rst) rst_q <= RD_IDLE;
    else case (rst_q)
      RD_IDLE: if (arvalid) rst_q <= RD_ADDR;
      RD_ADDR: rst_q <= RD_DATA;
      RD_DATA: if (rready)  rst_q <= RD_IDLE;
      default: rst_q <= RD_IDLE;
    endcase
    if (arvalid && arready) raddr <= {araddr[6:2], 2'b00};
    if (ar_hs) begin
      rdata <= '0;
      case (raddr)
        A_CTRL:  begin
                   rdata[0] <= int_ap_start;
                   rdata[1] <= int_task_done;
                   rdata[2] <= int_ap_idle;
                   rdata[3] <= int_ap_ready;
                   rdata[7] <= int_auto_restart;
                   rdata[9] <= int_irq;
                 end
        A_GIE:    rdata <= {31'b0, int_gie};
        A_IER:    rdata <= {30'b0, int_ier};
        A_ISR:    rdata <= {30'b0, int_isr};
        A_OP:     rdata <= op;
        A_N:      rdata <= n;
        A_ROWS:   rdata <= rows;
        A_BLOCKS: rdata <= blocks;
        A_A0:     rdata <= a[31:0];
        A_A1:     rdata <= a[63:32];
        A_AS1:    rdata <= a_s1;
        A_AS2:    rdata <= a_s2;
        A_B0:     rdata <= b[31:0];
        A_B1:     rdata <= b[63:32];
        A_BS1:    rdata <= b_s1;
        A_BS2:    rdata <= b_s2;
        A_C0:     rdata <= c[31:0];
        A_C1:     rdata <= c[63:32];
        A_CS1:    rdata <= c_s1;
        A_CS2:    rdata <= c_s2;
        A_P0:     rdata <= p[31:0];
        A_P1:     rdata <= p[63:32];
        A_E0:     rdata <= eps[31:0];
        A_E1:     rdata <= eps[63:32];
        A_AUX:    rdata <= aux;
        A_BQSH:   rdata <= bq_sh;
        A_WLO:    rdata <= win_lo;
        A_WHI:    rdata <= win_hi;
        default: ;
      endcase
    end
  end

endmodule
