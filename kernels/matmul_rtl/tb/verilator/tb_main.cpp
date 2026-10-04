// ---------------------------------------------------------------------------
// tb_main.cpp — Verilator testbench for the RTL MatmulKernel.
//
// Runs matrix products through the real AXI interfaces against sparse-memory
// slave models with randomised timing and checks, per case:
//   * every C element bit-exactly against a reference model that reads A and
//     B straight from the DDR image through the layout formulas of
//     MatmulKernel.h (row-major, tile-major packed, GEMV kernel-width image);
//   * that no byte outside the C elements was written;
//   * the AXI protocol on all ports, ap_done / interrupt behaviour.
//
// Usage:
//   Vtb [--fixtures DIR] [--random N] [--seed S] [--timing fast|rand|slow]
//       [--case "n k m batch as bs cs packed kw [mode]"] [--perf] [--quiet]
//       [--max-cycles N] [--trace FILE]
// ---------------------------------------------------------------------------
#include <cstdarg>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>

#include "VMatmulKernel.h"
#include "axi_models.h"
#if VM_TRACE
#ifdef VCD_TRACE
#include "verilated_vcd_c.h"
using TraceT = VerilatedVcdC;
#else
#include "verilated_fst_c.h"
using TraceT = VerilatedFstC;
#endif
#endif

uint64_t g_cycle = 0;

[[noreturn]] void tb_fatal(const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  std::fprintf(stderr, "FATAL @%llu: ", (unsigned long long)g_cycle);
  std::vfprintf(stderr, fmt, ap);
  std::fprintf(stderr, "\n");
  va_end(ap);
  std::exit(2);
}

// Register map (xmatmulkernel_hw.h)
enum : uint8_t {
  R_CTRL = 0x00, R_GIE = 0x04, R_IER = 0x08, R_ISR = 0x0C,
  R_A = 0x10, R_B = 0x1C, R_C = 0x28, R_N = 0x34, R_K = 0x3C, R_M = 0x44,
  R_BATCH = 0x4C, R_AS = 0x54, R_BS = 0x5C, R_CS = 0x64, R_BP = 0x6C, R_KW = 0x74,
  R_AB = 0x7C,
};

// ---------------------------------------------------------------------------
// Test case description
// ---------------------------------------------------------------------------
struct Case {
  std::string label;
  uint32_t n = 1, k = 1, m = 1, batch = 1;
  uint32_t as = 0, bs = 0, cs = 0;       // element strides
  uint32_t packed = 0, kw = 0;
  uint64_t a = 0, b = 0, c = 0;          // byte addresses
  // Explicit data (fixtures); empty -> random fill of the needed footprint.
  std::vector<uint16_t> av, bv, cv;
  int data_mode = 0;                      // 0 small, 1 full int16, 2 mixed
};

// B element (kk, p) of batch slice bi, as an element index into the B image.
static uint64_t b_index(const Case& t, uint32_t kk, uint32_t p) {
  if (t.kw >= 1) {
    const uint32_t kw = t.kw;
    if (kw == 1) return (uint64_t)kk * t.m + p;
    const uint64_t c = (kk / (16 * kw)) * 16 + kk % 16;
    const uint64_t j = (kk / 16) % kw;
    return (c * t.m + p) * kw + j;
  }
  if (t.packed) return ((uint64_t)(p / 32) * t.k + kk) * 32 + p % 32;
  return (uint64_t)kk * t.m + p;
}

static uint64_t packed_m(uint32_t m) { return (m + 31) / 32 * 32; }

// Elements one batch slice of B spans in its layout.
static uint64_t b_slice_elems(const Case& t) {
  if (t.kw == 0 && t.packed) return (uint64_t)t.k * packed_m(t.m);
  return (uint64_t)t.k * t.m;
}

static int16_t sat16(int32_t acc) {
  int32_t v = acc >> 8;                  // arithmetic: floor
  if (v > 32767) v = 32767;
  if (v < -32768) v = -32768;
  return (int16_t)v;
}

// Reference: reads the DDR image written for the case.
static void reference(Memory& mem, const Case& t, std::vector<int16_t>& out) {
  out.assign((size_t)t.batch * t.n * t.m, 0);
  std::vector<int16_t> bcol(t.k);
  for (uint32_t bi = 0; bi < t.batch; bi++) {
    const uint64_t ab = t.a + 2ull * bi * t.as, bb = t.b + 2ull * bi * t.bs;
    for (uint32_t p = 0; p < t.m; p++) {
      for (uint32_t kk = 0; kk < t.k; kk++)
        bcol[kk] = (int16_t)mem.rd16(bb + 2 * b_index(t, kk, p));
      for (uint32_t r = 0; r < t.n; r++) {
        uint32_t acc = 0;
        const uint64_t arow = ab + 2ull * r * t.k;
        for (uint32_t kk = 0; kk < t.k; kk++)
          acc += (uint32_t)((int32_t)(int16_t)mem.rd16(arow + 2 * kk) * (int32_t)bcol[kk]);
        out[((size_t)bi * t.n + r) * t.m + p] = sat16((int32_t)acc);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Simulation harness
// ---------------------------------------------------------------------------
struct Tb {
  std::unique_ptr<VerilatedContext> ctx;
  std::unique_ptr<VMatmulKernel>    top;
  Memory        mem;
  Timing        tim;
  std::mt19937  rng{1};
  AxiReadSlave  rd0{}, rd1{};
  AxiWriteSlave wr{};
  AxiLiteMaster lite{};
  uint64_t      max_cycles = 50'000'000;
  bool          quiet = false;
  bool          in_reset = false;    // AXI slaves ignore the bus during reset
#if VM_TRACE
  std::unique_ptr<TraceT> tfp;
#endif

  Tb(int argc, char** argv) {
    ctx.reset(new VerilatedContext);
    ctx->commandArgs(argc, argv);    // +verilator+rand+reset+2 +verilator+seed+N
    top.reset(new VMatmulKernel{ctx.get()});
    auto* T = top.get();
    rd0 = {"gmem0", &T->m_axi_gmem0_ARVALID, &T->m_axi_gmem0_ARREADY, &T->m_axi_gmem0_ARLEN,
           &T->m_axi_gmem0_ARSIZE, &T->m_axi_gmem0_ARBURST, &T->m_axi_gmem0_ARID,
           &T->m_axi_gmem0_ARADDR, &T->m_axi_gmem0_RVALID, &T->m_axi_gmem0_RREADY,
           &T->m_axi_gmem0_RLAST, &T->m_axi_gmem0_RRESP, &T->m_axi_gmem0_RID,
           &T->m_axi_gmem0_RDATA, &mem, &tim, &rng};
    rd1 = {"gmem1", &T->m_axi_gmem1_ARVALID, &T->m_axi_gmem1_ARREADY, &T->m_axi_gmem1_ARLEN,
           &T->m_axi_gmem1_ARSIZE, &T->m_axi_gmem1_ARBURST, &T->m_axi_gmem1_ARID,
           &T->m_axi_gmem1_ARADDR, &T->m_axi_gmem1_RVALID, &T->m_axi_gmem1_RREADY,
           &T->m_axi_gmem1_RLAST, &T->m_axi_gmem1_RRESP, &T->m_axi_gmem1_RID,
           &T->m_axi_gmem1_RDATA, &mem, &tim, &rng};
    wr = {"gmem2", &T->m_axi_gmem2_AWVALID, &T->m_axi_gmem2_AWREADY, &T->m_axi_gmem2_AWLEN,
          &T->m_axi_gmem2_AWSIZE, &T->m_axi_gmem2_AWBURST, &T->m_axi_gmem2_AWADDR,
          &T->m_axi_gmem2_WVALID, &T->m_axi_gmem2_WREADY, &T->m_axi_gmem2_WLAST,
          &T->m_axi_gmem2_WSTRB, &T->m_axi_gmem2_WDATA, &T->m_axi_gmem2_BVALID,
          &T->m_axi_gmem2_BREADY, &T->m_axi_gmem2_BRESP, &T->m_axi_gmem2_BID,
          &mem, &tim, &rng};
    lite.awvalid = &T->s_axi_ctrl_AWVALID; lite.awready = &T->s_axi_ctrl_AWREADY;
    lite.awaddr  = &T->s_axi_ctrl_AWADDR;  lite.wvalid  = &T->s_axi_ctrl_WVALID;
    lite.wready  = &T->s_axi_ctrl_WREADY;  lite.wstrb   = &T->s_axi_ctrl_WSTRB;
    lite.wdata   = &T->s_axi_ctrl_WDATA;   lite.bvalid  = &T->s_axi_ctrl_BVALID;
    lite.bready  = &T->s_axi_ctrl_BREADY;  lite.arvalid = &T->s_axi_ctrl_ARVALID;
    lite.arready = &T->s_axi_ctrl_ARREADY; lite.araddr  = &T->s_axi_ctrl_ARADDR;
    lite.rvalid  = &T->s_axi_ctrl_RVALID;  lite.rready  = &T->s_axi_ctrl_RREADY;
    lite.rdata   = &T->s_axi_ctrl_RDATA;
    lite.tick    = [this] { tick(); };
    lite.idle_inputs();

    // Unused inputs of the read-only / write-only ports.
    T->m_axi_gmem0_AWREADY = 0; T->m_axi_gmem0_WREADY = 0; T->m_axi_gmem0_BVALID = 0;
    T->m_axi_gmem1_AWREADY = 0; T->m_axi_gmem1_WREADY = 0; T->m_axi_gmem1_BVALID = 0;
    T->m_axi_gmem2_ARREADY = 0; T->m_axi_gmem2_RVALID = 0;
  }

  void tick() {
    auto* T = top.get();
    rd0.drive(); rd1.drive(); wr.drive();
    T->ap_clk = 0;
    T->eval();
#if VM_TRACE
    if (tfp) tfp->dump(2 * g_cycle);
#endif
    if (in_reset) {
      T->ap_clk = 1;
      T->eval();
      g_cycle++;
      return;
    }
    rd0.sample(); rd1.sample(); wr.sample(); lite.sample();
    if (T->m_axi_gmem0_AWVALID || T->m_axi_gmem0_WVALID || T->m_axi_gmem1_AWVALID ||
        T->m_axi_gmem1_WVALID || T->m_axi_gmem2_ARVALID)
      tb_fatal("activity on an unused AXI channel");
    T->ap_clk = 1;
    T->eval();
#if VM_TRACE
    if (tfp) tfp->dump(2 * g_cycle + 1);
#endif
    g_cycle++;
  }

  void reset() {
    in_reset = true;
    top->ap_rst_n = 0;
    for (int i = 0; i < 8; i++) tick();
    in_reset = false;
    auto* T = top.get();
    if (T->m_axi_gmem0_ARVALID || T->m_axi_gmem1_ARVALID || T->m_axi_gmem2_AWVALID ||
        T->m_axi_gmem2_WVALID || T->s_axi_ctrl_BVALID || T->s_axi_ctrl_RVALID || T->__SYM__interrupt)
      tb_fatal("AXI VALID or interrupt still high after reset");
    top->ap_rst_n = 1;
    for (int i = 0; i < 4; i++) tick();
  }

  void wr64(uint8_t reg, uint64_t v) {
    lite.write(reg, (uint32_t)v);
    lite.write(reg + 4, (uint32_t)(v >> 32));
  }

  // Runs one case; returns the kernel cycle count (0 on failure).
  uint64_t run(Case& t) {
    // ---- place data --------------------------------------------------------------
    const uint64_t slices  = t.batch ? t.batch - 1 : 0;
    const uint64_t a_elems = t.av.size() ? t.av.size()
        : (t.as ? slices * t.as : 0) + (uint64_t)t.n * t.k;
    const uint64_t b_elems = t.bv.size() ? t.bv.size()
        : (t.bs ? slices * t.bs : 0) + b_slice_elems(t);
    auto rnd16 = [&](int mode) -> uint16_t {
      if (mode == 1) return (uint16_t)rng();
      if (mode == 2 && (rng() & 3) == 0) return (uint16_t)rng();
      return (uint16_t)(int16_t)std::uniform_int_distribution<int>(-256, 256)(rng);
    };
    for (uint64_t i = 0; i < a_elems; i++)
      mem.wr16(t.a + 2 * i, t.av.size() ? t.av[i] : rnd16(t.data_mode));
    if (t.bv.size()) {
      for (uint64_t i = 0; i < b_elems; i++) mem.wr16(t.b + 2 * i, t.bv[i]);
    } else {
      for (uint64_t i = 0; i < b_elems; i++) mem.wr16(t.b + 2 * i, rnd16(t.data_mode));
      // packed layout: the column padding of every tile is zero
      if (t.kw == 0 && t.packed) {
        const uint32_t slices = t.bs ? t.batch : 1;
        for (uint32_t s = 0; s < slices; s++)
          for (uint32_t kk = 0; kk < t.k; kk++)
            for (uint32_t p = t.m; p < packed_m(t.m); p++)
              mem.wr16(t.b + 2 * ((uint64_t)s * t.bs + b_index(t, kk, p)), 0);
      }
    }

    std::vector<int16_t> ref;
    reference(mem, t, ref);
    if (t.cv.size()) {
      if (t.cv.size() != ref.size()) tb_fatal("%s: c.hex has %zu elements, expected %zu",
                                              t.label.c_str(), t.cv.size(), ref.size());
      for (size_t i = 0; i < ref.size(); i++)
        if ((uint16_t)ref[i] != t.cv[i])
          tb_fatal("%s: reference model disagrees with the fixture at %zu (%04x vs %04x)",
                   t.label.c_str(), i, (uint16_t)ref[i], t.cv[i]);
    }

    // ---- run ------------------------------------------------------------------------------
    mem.written.clear();
    wr64(R_A, t.a); wr64(R_B, t.b); wr64(R_C, t.c);
    lite.write(R_N, t.n); lite.write(R_K, t.k); lite.write(R_M, t.m);
    lite.write(R_BATCH, t.batch);
    lite.write(R_AS, t.as); lite.write(R_BS, t.bs); lite.write(R_CS, t.cs);
    lite.write(R_BP, t.packed); lite.write(R_KW, t.kw);
    wr64(R_AB, t.b - t.a);
    lite.write(R_GIE, 1); lite.write(R_IER, 1);

    if (lite.read(R_K) != t.k || lite.read(R_A) != (uint32_t)t.a)
      tb_fatal("register read-back mismatch");
    if (!(lite.read(R_CTRL) & 4)) tb_fatal("kernel not idle before start");

    lite.write(R_CTRL, 1);
    const uint64_t t0 = g_cycle;
    uint64_t done_at = 0;
    while (!top->__SYM__interrupt) {
      tick();
      if (g_cycle - t0 > max_cycles) {
        std::fprintf(stderr, "TIMEOUT %s after %llu cycles\n", t.label.c_str(),
                     (unsigned long long)(g_cycle - t0));
        return 0;
      }
    }
    done_at = g_cycle;
    const uint32_t ctrl = lite.read(R_CTRL);
    if (!(ctrl & 2)) tb_fatal("interrupt without ap_done (ctrl %x)", ctrl);
    if (!(ctrl & 4)) tb_fatal("not idle after done (ctrl %x)", ctrl);
    if (lite.read(R_CTRL) & 2) tb_fatal("ap_done not cleared on read");
    if (!(lite.read(R_ISR) & 1)) tb_fatal("ISR done bit not set");
    lite.write(R_ISR, 1);
    for (int i = 0; i < 4; i++) tick();
    if (top->__SYM__interrupt) tb_fatal("interrupt still asserted after ISR clear");
    if (!rd0.idle() || !rd1.idle() || !wr.idle())
      tb_fatal("AXI traffic outstanding after ap_done");

    // ---- check --------------------------------------------------------------------------
    size_t bad = 0;
    for (uint32_t bi = 0; bi < t.batch; bi++)
      for (uint32_t r = 0; r < t.n; r++)
        for (uint32_t p = 0; p < t.m; p++) {
          const uint64_t addr = t.c + 2 * ((uint64_t)bi * t.cs + (uint64_t)r * t.m + p);
          const uint16_t got = mem.rd16(addr);
          const uint16_t exp = (uint16_t)ref[((size_t)bi * t.n + r) * t.m + p];
          if (got != exp || !mem.was_written(addr) || !mem.was_written(addr + 1)) {
            if (bad < 5)
              std::fprintf(stderr, "  MISMATCH %s bi=%u r=%u p=%u got=%04x exp=%04x%s\n",
                           t.label.c_str(), bi, r, p, got, exp,
                           mem.was_written(addr) ? "" : " (not written)");
            bad++;
          }
          mem.unmark(addr);                                  // accounted for
          mem.unmark(addr + 1);
        }
    size_t stray = 0;
    for (auto& kv : mem.written)
      for (int i = 0; i < 4096; i++)
        if (kv.second[i]) {
          if (stray < 3)
            std::fprintf(stderr, "  STRAY WRITE %s at %llx\n", t.label.c_str(),
                         (unsigned long long)((kv.first << 12) + i));
          stray++;
        }
    const uint64_t cyc = done_at - t0;
    if (bad || stray) {
      std::printf("FAIL  %-44s  %zu/%zu wrong, %zu stray bytes\n", t.label.c_str(), bad,
                  ref.size(), stray);
      return 0;
    }
    if (!quiet) {
      const double macs = (double)t.batch * t.n * t.k * t.m;
      std::printf("PASS  %-44s  n=%u k=%u m=%u b=%u %s  %llu cyc  %.1f MAC/cyc\n",
                  t.label.c_str(), t.n, t.k, t.m, t.batch,
                  t.kw ? ("kw" + std::to_string(t.kw)).c_str() : (t.packed ? "pk" : "rm"),
                  (unsigned long long)cyc, macs / (double)cyc);
    }
    return cyc;
  }
};

// ---------------------------------------------------------------------------
// Case construction
// ---------------------------------------------------------------------------
static uint64_t g_region = 0x0000'0008'0000'0000ull;   // above 4 GiB

// misalign: shift the C base (and, beyond the kernel contract, the A / B
// bases) by a random number of elements within the first 16-byte word.
static void place(Case& t, std::mt19937& rng, bool misalign) {
  auto next = [&](uint64_t bytes) {
    uint64_t base = g_region;
    g_region += (bytes + 0x10000) & ~0xFFFull;
    g_region += 0x1000 * (rng() % 4) + 16 * (rng() % 256);
    g_region &= ~15ull;
    return base;
  };
  const uint64_t slices  = t.batch ? t.batch - 1 : 0;
  const uint64_t a_elems = (t.as ? slices * t.as : 0) + (uint64_t)t.n * t.k;
  const uint64_t b_elems = (t.bs ? slices * t.bs : 0) + b_slice_elems(t);
  const uint64_t c_elems = slices * t.cs + (uint64_t)t.n * t.m;
  auto skew = [&] { return misalign ? 2 * (rng() % 8) : 0; };
  t.a = next(2 * std::max<uint64_t>(a_elems, t.av.size()) + 64) + ((rng() & 3) == 0 ? skew() : 0);
  t.b = next(2 * std::max<uint64_t>(b_elems, t.bv.size()) + 64) + ((rng() & 3) == 0 ? skew() : 0);
  t.c = next(2 * c_elems + 64) + skew();
}

static Case make_case(uint32_t n, uint32_t k, uint32_t m, uint32_t batch, uint32_t packed,
                      uint32_t kw, std::mt19937& rng, int broadcast = 0, bool pad = false) {
  Case t;
  t.n = n; t.k = k; t.m = m; t.batch = batch; t.packed = packed; t.kw = kw;
  const uint32_t al = kw ? 8 : 1;            // GEMV strides are multiples of 8
  auto padded = [&](uint64_t v) {
    uint64_t p = pad ? (rng() % 40) : 0;
    return (uint32_t)((v + p + al - 1) / al * al);
  };
  t.as = (broadcast & 1) ? 0 : padded((uint64_t)n * k);
  t.bs = (broadcast & 2) ? 0 : padded(b_slice_elems(t));
  t.cs = (uint32_t)((uint64_t)n * m + (pad ? rng() % 24 : 0));
  if (batch == 1 && !pad) { t.as = n * k; t.bs = (uint32_t)b_slice_elems(t); }
  t.data_mode = (int)(rng() % 3);
  return t;
}

// A random case that satisfies the kernel's documented constraints.
static Case random_case(std::mt19937& rng) {
  auto U = [&](uint32_t lo, uint32_t hi) {
    return std::uniform_int_distribution<uint32_t>(lo, hi)(rng);
  };
  const uint32_t mode = U(0, 5);   // 0,1 row-major  2 packed  3..5 GEMV
  uint32_t n = (U(0, 3) == 0) ? U(1, 3) : U(1, 24);
  uint32_t batch = (U(0, 3) == 0) ? U(2, 3) : 1;
  int bc = (batch > 1) ? (int)U(0, 3) : 0;
  Case t;
  if (mode <= 2) {
    uint32_t k = (U(0, 5) == 0) ? U(1, 40) : U(1, 700);
    uint32_t m = (U(0, 5) == 0) ? U(1, 40) : U(1, 1200);
    if (U(0, 30) == 0) k = 4096 - U(0, 9);
    t = make_case(n, k, m, batch, mode == 2, 0, rng, bc, U(0, 1));
  } else {
    const uint32_t kw = 1u << U(0, 3);
    const uint32_t kq = (kw == 1) ? 8 : 16 * kw;
    uint32_t k = kq * U(1, std::max(1u, 1200 / kq));
    if (U(0, 30) == 0) k = 4096 / kq * kq;
    uint32_t m = 8 * U(1, 150);
    while (m * kw < 64) m += 8;
    t = make_case(n, k, m, batch, U(0, 1), kw, rng, bc, U(0, 1));
  }
  char buf[128];
  std::snprintf(buf, sizeof buf, "rand_n%u_k%u_m%u_b%u_%s%u", t.n, t.k, t.m, t.batch,
                t.kw ? "kw" : (t.packed ? "pk" : "rm"), t.kw);
  t.label = buf;
  return t;
}

// ---------------------------------------------------------------------------
// Fixtures: manifest.txt + test_NN_{a,b,c}.hex (kernels/matmul TestMatmulRef --dump-data)
// ---------------------------------------------------------------------------
static std::vector<uint16_t> read_hex(const std::string& path) {
  std::ifstream f(path);
  if (!f) tb_fatal("cannot open %s", path.c_str());
  std::vector<uint16_t> v;
  std::string line;
  while (std::getline(f, line))
    if (!line.empty()) v.push_back((uint16_t)std::stoul(line, nullptr, 16));
  return v;
}

static std::vector<Case> load_fixtures(const std::string& dir) {
  std::ifstream f(dir + "/manifest.txt");
  if (!f) tb_fatal("cannot open %s/manifest.txt", dir.c_str());
  std::vector<Case> out;
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream is(line);
    std::vector<std::string> tok;
    std::string s;
    while (is >> s) tok.push_back(s);
    if (tok.size() < 9) continue;
    Case t;
    const int idx = std::stoi(tok[0]);
    t.n = std::stoul(tok[1]); t.k = std::stoul(tok[2]); t.m = std::stoul(tok[3]);
    t.batch = std::stoul(tok[4]); t.as = std::stoul(tok[5]); t.bs = std::stoul(tok[6]);
    t.packed = std::stoul(tok[7]);
    const bool has_kw = tok.size() >= 10;
    t.kw = has_kw ? std::stoul(tok[8]) : 0;
    t.label = "fx" + tok[0] + "_" + tok.back();
    if (t.label.size() > 44) t.label.resize(44);
    t.cs = t.n * t.m;
    char pre[32];
    std::snprintf(pre, sizeof pre, "/test_%02d_", idx);
    t.av = read_hex(dir + pre + "a.hex");
    t.bv = read_hex(dir + pre + "b.hex");
    t.cv = read_hex(dir + pre + "c.hex");
    out.push_back(std::move(t));
  }
  return out;
}

// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
  std::string fixtures, trace, one_case, timing = "rand";
  int n_random = 0;
  unsigned seed = 1;
  bool perf = false, quiet = false;
  uint64_t max_cycles = 50'000'000;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto next = [&] { if (i + 1 >= argc) tb_fatal("missing value for %s", a.c_str()); return std::string(argv[++i]); };
    if (a == "--fixtures") fixtures = next();
    else if (a == "--random") n_random = std::stoi(next());
    else if (a == "--seed") seed = (unsigned)std::stoul(next());
    else if (a == "--timing") timing = next();
    else if (a == "--case") one_case = next();
    else if (a == "--perf") perf = true;
    else if (a == "--quiet") quiet = true;
    else if (a == "--max-cycles") max_cycles = std::stoull(next());
    else if (a == "--trace") trace = next();
    else if (a.rfind("+verilator", 0) == 0) {}
    else tb_fatal("unknown argument %s", a.c_str());
  }

  Tb tb(argc, argv);
  tb.rng.seed(seed);
  tb.max_cycles = max_cycles;
  tb.quiet = quiet;
  std::mt19937 crng(seed * 7919u + 13);
  auto set_timing = [&](const std::string& mode) {
    if (mode == "fast") tb.tim = Timing{1, 1, 1, 1, 1, 24, 24};
    else if (mode == "slow") tb.tim = Timing{0.3, 0.4, 0.3, 0.4, 0.5, 40, 200};
    else {
      auto P = [&] { return std::uniform_real_distribution<double>(0.25, 1.0)(crng); };
      int lmin = (int)(crng() % 60) + 2;
      tb.tim = Timing{P(), P(), P(), P(), P(), lmin, lmin + (int)(crng() % 120)};
    }
  };
#if VM_TRACE
  if (!trace.empty()) {
    Verilated::traceEverOn(true);
    tb.tfp.reset(new TraceT);
    tb.top->trace(tb.tfp.get(), 99);
    tb.tfp->open(trace.c_str());
  }
#endif
  tb.reset();

  std::vector<Case> cases;
  if (!fixtures.empty())
    for (auto& t : load_fixtures(fixtures)) cases.push_back(std::move(t));
  if (!one_case.empty()) {
    std::istringstream is(one_case);
    Case t;
    is >> t.n >> t.k >> t.m >> t.batch >> t.as >> t.bs >> t.cs >> t.packed >> t.kw;
    if (!(is >> t.data_mode)) t.data_mode = 0;
    t.label = "case";
    cases.push_back(t);
  }
  if (perf) {
    auto add = [&](const char* l, uint32_t n, uint32_t k, uint32_t m, uint32_t pk, uint32_t kw) {
      Case t = make_case(n, k, m, 1, pk, kw, crng);
      t.label = l;
      cases.push_back(t);
    };
    add("perf_gemv_576x1536_kw1", 1, 576, 1536, 0, 1);
    add("perf_gemv_512x1536_kw8", 1, 512, 1536, 0, 8);
    add("perf_gemv_1536x576_kw4", 1, 1536, 576, 0, 4);
    add("perf_gemv_576x1536_kw4", 1, 576, 1536, 0, 4);       // 9 K blocks: the last is split
    add("perf_gemv4_576x1536_kw1", 4, 576, 1536, 0, 1);
    add("perf_fc_1x4096x512_rm", 1, 4096, 512, 0, 0);
    add("perf_gemm_64x576x576_pk", 64, 576, 576, 1, 0);
    add("perf_gemm_64x576x576_rm", 64, 576, 576, 0, 0);
    add("perf_gemm_128x256x2048_pk", 128, 256, 2048, 1, 0);
    add("perf_gemm_256x64x64_rm", 256, 64, 64, 0, 0);
    add("perf_gemm_64x144x576_pk", 64, 144, 576, 1, 0);      // 9 K blocks: the last is split
  }
  for (int i = 0; i < n_random; i++) cases.push_back(random_case(crng));
  if (cases.empty()) tb_fatal("nothing to run (use --fixtures, --random, --case or --perf)");

  int pass = 0, fail = 0;
  for (auto& t : cases) {
    set_timing(perf ? "fast" : timing);
    if (t.label.rfind("fx", 0) == 0 || t.label == "case")
      place(t, crng, false);
    else
      place(t, crng, (crng() & 1) != 0);
    tb.mem.clear();
    if (tb.run(t)) pass++;
    else fail++;
  }
  std::printf("\n%d passed, %d failed  (%llu cycles simulated)\n", pass, fail,
              (unsigned long long)g_cycle);
#if VM_TRACE
  if (tb.tfp) tb.tfp->close();
#endif
  tb.top->final();
  return fail ? 1 : 0;
}
