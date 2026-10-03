// ---------------------------------------------------------------------------
// mm_pkg — shared parameters and descriptor types of the MatmulKernel RTL.
//
// See doc/kernels/MATMUL_RTL_KERNEL.md for how the units use them.
// ---------------------------------------------------------------------------
package mm_pkg;

  // Data path geometry -------------------------------------------------------
  localparam int E       = 8;          // Q8.8 elements per 128-bit AXI beat
  localparam int EW      = 16;         // element width
  localparam int BW      = E * EW;     // beat width (128)
  localparam int R       = 8;          // A rows per panel = DSP rows per lane
  localparam int RH      = R / 2;      // panel rows loaded by each read port
  localparam int NLANE   = 2;          // lanes = read ports (gmem0, gmem1)

  // Accumulators: every DSP keeps ACC_D 32-bit partial sums in LUTRAM, so one
  // plane chunk (a B row segment) is at most ACC_D beats = W_EL elements.
  localparam int ACC_D   = 64;
  localparam int ACC_AW  = 6;
  localparam int W_EL    = ACC_D * E;  // 512
  localparam int W_EL_LG = 9;

  // A buffer: per lane and panel row, K_MAX / 2 elements as 128-bit words.
  localparam int K_MAX   = 4096;
  localparam int A_D     = K_MAX / 2 / E;   // 256 words
  localparam int A_AW    = 8;

  // K is split between the two lanes in interleaved blocks of 2^BLK_LG
  // planes (block b goes to lane b % 2).
  localparam int BLK_LG  = 4;

  // Minimum distance (cycles) between two MACs into the same accumulator:
  // LUTRAM read -> CREG -> PREG -> LUTRAM write.
  localparam int DMIN    = 3;
  // Cycles after the last issue before a lane's accumulators are final.
  localparam int FLUSH   = 6;

  // Tap FIFO between the x prefetcher and the MAC array (row sets).
  localparam int TAP_D   = 16;
  localparam int TAP_AW  = 4;

  // AXI engines.
  localparam int RD_FIFO_D  = 512;   // beats buffered per read port
  localparam int RD_BURST   = 64;    // max beats per AR burst (1 KiB)
  localparam int WR_FIFO_D  = 512;   // beats buffered before AW issue
  localparam int WR_BURST   = 64;    // max beats per AW burst
  localparam int WR_OUTS    = 16;    // AW bursts awaiting W / B

  // Run descriptor for a gearbox: a contiguous element range of `rows` rows
  // of `len` elements starting at lane `s` of the first word.  rows == 0 is a
  // marker (no data, one flagged output beat).
  typedef struct packed {
    logic        dest_b;      // 0: A panel rows -> A writer, 1: B -> MAC lane
    logic        marker;
    logic [2:0]  s;           // first element's lane in the first word
    logic [12:0] len;         // elements per row (1..4096)
    logic [4:0]  rows;        // 0..16
    logic [11:0] nw;          // 128-bit words the run spans
    logic [5:0]  wb;          // B: accumulator word of the row's first beat
    logic        init;        // B: the run's first row starts the accumulators
    logic        step_last;   // B: last run of this lane's step
    logic        panel_last;  // B: last run of this lane's panel
  } gb_run_t;

  // Read engine descriptor: nw words from word address waddr.
  typedef struct packed {
    logic [59:0] waddr;
    logic [11:0] nw;
  } rd_run_t;

  // x prefetcher descriptor (B runs only).
  typedef struct packed {
    logic        marker;
    logic [4:0]  rows;
    logic [10:0] cl0;         // lane-local plane of the first row
    logic        panel_last;
  } x_run_t;

  // Step = one (batch slice, panel, column chunk) of the job.
  typedef struct packed {
    logic [63:0] a_addr;      // byte address of A row n_off
    logic [63:0] b_addr;      // byte address of this batch slice of B
    logic [63:0] c_addr;      // byte address of C[n_off][c0]
    logic [31:0] c0;          // first column of the chunk
    logic [9:0]  mcc;         // columns in the chunk (1..512)
    logic [3:0]  n_valid;     // panel rows (1..R)
    logic        a_load;      // this step (re)loads the A panel
    logic        panel_first;
    logic        panel_last;
    logic        last;        // last step of the job
  } step_t;

  // Job constants, latched at ap_start and derived once.
  typedef struct packed {
    logic [31:0] n, m, batch;
    logic [12:0] k;           // clamped to K_MAX
    logic        pk;          // packed (tile-major) B
    logic [1:0]  lk;          // log2 of the GEMV kernel width (0 = row-major)
    logic [12:0] planes;      // k >> lk
    logic [8:0]  nblk;        // ceil(planes / 16)
    logic        contig;      // image mode, one chunk: planes are contiguous
    logic [35:0] lfull;       // elements per plane (m << lk)
    logic [9:0]  mc_max;      // columns per chunk
  } cfg_t;

  function automatic logic [31:0] umin32(input logic [31:0] a, input logic [31:0] b);
    return (a < b) ? a : b;
  endfunction

endpackage
