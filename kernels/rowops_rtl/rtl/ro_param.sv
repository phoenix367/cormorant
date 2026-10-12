// ---------------------------------------------------------------------------
// ro_param — the per-column parameter block P (ROWOPS_RTL_KERNEL.md §1.3).
//
// Load: the 16-byte DDR entries arrive as words on `in` (stream B's port,
// before the job's streams start); entry c goes to bank c % 4 at index
// c / 4, or — P_BCAST — entry 0 to index 0 of every bank.  Read: index i
// returns the four entries of columns 4i .. 4i + 3, two cycles later (the
// four arithmetic lanes of every unit).
//
// Storage: two 4096 x 72 UltraRAMs, each holding two banks (bank 2k at rows
// 0 .. 1023, bank 2k + 1 at rows 2048 .. 3071) read through its two ports;
// the load writes through port A while nothing reads (the port keeps its
// output on a write: the write mode an UltraRAM infers from).
// ---------------------------------------------------------------------------
module ro_param
  import ro_pkg::*;
(
  input  logic          clk,
  input  logic          rst,

  input  logic          ld_start,         // one cycle: a load of `ld_n` entries begins
  input  logic          ld_bcast,         // the one entry fills index 0 of every bank
  input  logic [31:0]   ld_n,             // entries (columns)
  output logic          ld_busy,          // from ld_start until the last entry is written

  input  logic          in_valid,
  output logic          in_ready,
  input  logic [BW-1:0] in_data,

  input  logic [PIW-1:0] rd_idx,          // column / 4
  output param_t        rd_q [PB]         // columns 4 rd_idx .. + 3, two cycles after rd_idx
);
  localparam int BANK1 = 2048;            // bank 2k + 1's rows in RAM k

  function automatic param_t unpack(input logic [BW-1:0] w);
    param_t p;
    p.g    = w[31:0];
    p.bq   = w[53:32];
    p.sh_a = w[69:64];
    p.sh_b = w[77:72];
    p.sh_c = w[85:80];
    return p;
  endfunction

  // Load ------------------------------------------------------------------------------
  logic [31:0]    ld_left;
  logic           bcast;                  // the loaded block is one entry: every read returns index 0
  logic [PIW-1:0] rd_i;
  assign rd_i = bcast ? '0 : rd_idx;
  logic [1:0]     bc_step;                // broadcast: the entry written to bank 1 .. 3
  logic [1:0]     wr_bank, w_bank;        // the next entry's bank / the bank written this cycle
  logic [PIW-1:0] wr_idx, w_idx;
  param_t         w_data, bc_data;
  logic [PRM_W-1:0] w_vec;
  logic           bc_on, accept, we;
  assign w_vec = w_data;

  assign in_ready = ld_busy && !bc_on;
  assign accept   = in_valid && in_ready;
  // the write happens in the accepting cycle (the counters still name its slot)
  assign we     = accept || bc_on;
  assign w_bank = bc_on ? bc_step : wr_bank;
  assign w_idx  = bc_on ? '0 : wr_idx;
  assign w_data = bc_on ? bc_data : unpack(in_data);

  always_ff @(posedge clk) begin
    if (rst) begin
      ld_busy <= 1'b0;
      bc_on   <= 1'b0;
    end else begin
      if (ld_start) begin
        ld_busy <= (ld_n != '0);
        ld_left <= ld_n;
        bcast   <= ld_bcast;
        wr_bank <= '0;
        wr_idx  <= '0;
        bc_on   <= 1'b0;
      end else if (bc_on) begin                     // the broadcast entry into banks 1 .. 3
        bc_step <= bc_step + 2'd1;
        if (bc_step == 2'd3) begin
          bc_on   <= 1'b0;
          ld_busy <= 1'b0;
        end
      end else if (accept) begin
        if (bcast) begin                            // bank 0 written now; the rest follow
          bc_data <= unpack(in_data);
          bc_step <= 2'd1;
          bc_on   <= 1'b1;
          ld_left <= '0;
        end else begin
          ld_left <= ld_left - 32'd1;
          if (ld_left == 32'd1) ld_busy <= 1'b0;
          if (wr_bank == 2'd3) begin
            wr_bank <= 2'd0;
            wr_idx  <= wr_idx + PIW'(1);
          end else begin
            wr_bank <= wr_bank + 2'd1;
          end
        end
      end
    end
  end

  // Storage: RAM k holds banks 2k (rows 0..) and 2k+1 (rows BANK1..).  Port A
  // writes during a load and reads bank 2k otherwise; port B reads bank 2k+1.
  for (genvar k = 0; k < PB / 2; k++) begin : g_ram
    logic [11:0]      waddr;
    logic [PRM_W-1:0] qa, qb;
    logic             we_k;
    assign we_k  = we && (w_bank[1] == 1'(k));
    assign waddr = (w_bank[0] ? 12'(BANK1) : 12'd0) | 12'(w_idx);
    ro_prm_ram u_ram (.clk, .we (we_k), .waddr, .wdata (w_vec),
                      .addr_a ({2'b0, rd_i}), .addr_b (12'(BANK1) | {2'b0, rd_i}), .q_a (qa), .q_b (qb));
    assign rd_q[2 * k]     = qa;
    assign rd_q[2 * k + 1] = qb;
  end

endmodule
