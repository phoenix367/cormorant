// ---------------------------------------------------------------------------
// tb_main.cpp — Verilator testbench of the RTL PoolingKernel (kernels/pool_rtl).
//
// Drives the IP top level PoolingKernel through its AXI4-Lite slave and serves
// gmem0 (x, read) / gmem1 (y, write) from a sparse byte-addressed DDR model
// with randomised ready / valid timing and read latency.  Checks per job:
//   * with the HLS oracle (PL_HLS_ORACLE: the HLS kernel's own C++,
//     kernels/pool/kernel/PoolingKernel.cpp, built with the Vitis HLS headers):
//     every byte of the output region equals what the HLS kernel leaves there,
//     run on the same DDR image — the written elements and the untouched
//     lanes between runs alike;
//   * fixtures (kernels/pool TestPoolingSim --dump-data): every output element
//     equals y.hex;
//   * that nothing outside the output elements was written, the AXI protocol
//     (bursts never cross 4 KiB, WLAST, at most the declared outstanding
//     bursts), ap_done / interrupt behaviour.
// Jobs outside the contract (pool_h / pool_w > 7, a dilated window past the
// line buffer) must write nothing; jobs with a zero size likewise (the HLS
// kernel's behaviour there is not defined, so no oracle).
//
// Usage:
//   Vtb [--fixtures DIR] [--random N] [--seed S] [--timing fast|rand|slow]
//       [--case "N C H W oh ow ph pw sh sw pt pl dh dw type lp cip"] [--perf]
//       [--quiet] [--max-cycles N] [--trace FILE]
// ---------------------------------------------------------------------------
#include <algorithm>
#include <cstdarg>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <fcntl.h>
#include <unistd.h>

#include "VPoolingKernel.h"
#include "axi_models.h"
#ifdef PL_HLS_ORACLE
#include "PoolingKernel.h"
#endif
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

// Register map (xpoolingkernel_hw.h)
enum : uint8_t {
  R_CTRL = 0x00, R_GIE = 0x04, R_IER = 0x08, R_ISR = 0x0C,
  R_X = 0x10, R_Y = 0x1C, R_BATCH = 0x28, R_CH = 0x30, R_IN_H = 0x38, R_IN_W = 0x40,
  R_OUT_H = 0x48, R_OUT_W = 0x50, R_POOL_H = 0x58, R_POOL_W = 0x60, R_STRIDE_H = 0x68,
  R_STRIDE_W = 0x70, R_PAD_TOP = 0x78, R_PAD_LEFT = 0x80, R_DIL_H = 0x88, R_DIL_W = 0x90,
  R_TYPE = 0x98, R_LP = 0xA0, R_CIP = 0xA8,
};

// ---------------------------------------------------------------------------
// Test case
// ---------------------------------------------------------------------------
struct Case {
  std::string label;
  uint32_t n = 1, c = 1, h = 1, w = 1, oh = 1, ow = 1, ph = 1, pw = 1, sh = 1, sw = 1;
  uint32_t pt = 0, pl = 0, dh = 1, dw = 1, type = 0, lp = 1, cip = 0;
  uint64_t x = 0, y = 0;                 // byte addresses
  std::vector<uint16_t> xv, yv;          // fixture data (empty: random)
  int data_mode = 0;
  bool no_oracle = false;                // zero-size job: must write nothing
};

static uint64_t x_elems(const Case& t) { return (uint64_t)t.n * t.c * t.h * t.w; }
static uint64_t y_elems(const Case& t) { return (uint64_t)t.n * t.c * t.oh * t.ow; }
static uint64_t words_of_elems(uint64_t e) { return (e + 7) / 8; }

static bool in_contract(const Case& t) {
  return t.ph >= 1 && t.ph <= 7 && t.pw >= 1 && t.pw <= 7 &&
         (t.ph == 1 || t.dh < 16) && (t.pw == 1 || t.dw < 64) &&
         (uint64_t)(t.ph - 1) * t.dh < 16 && (uint64_t)(t.pw - 1) * t.dw < 64;
}

// ---------------------------------------------------------------------------
// Simulation harness
// ---------------------------------------------------------------------------
struct Tb {
  std::unique_ptr<VerilatedContext> ctx;
  std::unique_ptr<VPoolingKernel>   top;
  Memory        mem;
  Timing        tim;
  std::mt19937  rng{1};
  AxiReadSlave  rd{};
  AxiWriteSlave wr{};
  AxiLiteMaster lite{};
  uint64_t      max_cycles = 50'000'000;
  bool          quiet = false;
  bool          in_reset = false;
#if VM_TRACE
  std::unique_ptr<TraceT> tfp;
#endif

  Tb(int argc, char** argv) {
    ctx.reset(new VerilatedContext);
    ctx->commandArgs(argc, argv);
    top.reset(new VPoolingKernel{ctx.get()});
    auto* T = top.get();
    rd = {"gmem0", &T->m_axi_gmem0_ARVALID, &T->m_axi_gmem0_ARREADY, &T->m_axi_gmem0_ARLEN,
          &T->m_axi_gmem0_ARSIZE, &T->m_axi_gmem0_ARBURST, &T->m_axi_gmem0_ARID,
          &T->m_axi_gmem0_ARADDR, &T->m_axi_gmem0_RVALID, &T->m_axi_gmem0_RREADY,
          &T->m_axi_gmem0_RLAST, &T->m_axi_gmem0_RRESP, &T->m_axi_gmem0_RID,
          &T->m_axi_gmem0_RDATA, &mem, &tim, &rng};
    wr = {"gmem1", &T->m_axi_gmem1_AWVALID, &T->m_axi_gmem1_AWREADY, &T->m_axi_gmem1_AWLEN,
          &T->m_axi_gmem1_AWSIZE, &T->m_axi_gmem1_AWBURST, &T->m_axi_gmem1_AWADDR,
          &T->m_axi_gmem1_WVALID, &T->m_axi_gmem1_WREADY, &T->m_axi_gmem1_WLAST,
          &T->m_axi_gmem1_WSTRB, &T->m_axi_gmem1_WDATA, &T->m_axi_gmem1_BVALID,
          &T->m_axi_gmem1_BREADY, &T->m_axi_gmem1_BRESP, &T->m_axi_gmem1_BID,
          &mem, &tim, &rng};
    // the outstanding bursts the IP declares (pl_pkg RD_OUTS / WR_OUTS, package_ip.tcl;
    // fact rtl.axi_masters)
    rd.outs_limit = 16;  wr.outs_limit = 8;
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

    T->m_axi_gmem0_AWREADY = 0; T->m_axi_gmem0_WREADY = 0; T->m_axi_gmem0_BVALID = 0;
    T->m_axi_gmem1_ARREADY = 0; T->m_axi_gmem1_RVALID = 0;
  }

  void tick() {
    auto* T = top.get();
    rd.drive(); wr.drive();
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
    rd.sample(); wr.sample(); lite.sample();
    if (T->m_axi_gmem0_AWVALID || T->m_axi_gmem0_WVALID || T->m_axi_gmem1_ARVALID)
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
    if (T->m_axi_gmem0_ARVALID || T->m_axi_gmem1_AWVALID || T->m_axi_gmem1_WVALID ||
        T->s_axi_ctrl_BVALID || T->s_axi_ctrl_RVALID || T->__SYM__interrupt)
      tb_fatal("AXI VALID or interrupt still high after reset");
    top->ap_rst_n = 1;
    for (int i = 0; i < 4; i++) tick();
  }

  void wr64(uint8_t reg, uint64_t v) {
    lite.write(reg, (uint32_t)v);
    lite.write(reg + 4, (uint32_t)(v >> 32));
  }

  uint16_t rnd16(int mode) {
    static const uint16_t edge[] = {0x8000, 0x7FFF, 0x0000, 0x0001, 0xFFFF, 0x0100, 0xFF00,
                                    0x8001, 0x00FF, 0xFF01, 0x4000, 0xC000};
    switch (mode) {
      case 1:  return (uint16_t)rng();
      case 2:  if ((rng() & 3) == 0) return edge[rng() % (sizeof edge / sizeof edge[0])];
               return (uint16_t)rng();
      default: return (uint16_t)(int16_t)std::uniform_int_distribution<int>(-2048, 2048)(rng);
    }
  }

  // Runs one case; returns the kernel cycle count (0 on failure).
  uint64_t run(Case& t) {
    const uint64_t xw = words_of_elems(x_elems(t)) + 1;       // a run reads up to 7 past the end
    const uint64_t yw = words_of_elems(y_elems(t)) + 1;

    // ---- place data -----------------------------------------------------------------
    for (uint64_t i = 0; i < 8 * xw; i++)
      mem.wr16(t.x + 2 * i, i < t.xv.size() ? t.xv[i] : rnd16(t.data_mode));
    for (uint64_t i = 0; i < 8 * yw; i++) mem.wr16(t.y + 2 * i, 0xDEAD);   // poison

#ifdef PL_HLS_ORACLE
    // ---- the HLS kernel on the same image ------------------------------------------
    std::vector<PoolWord> ox(xw + 1), oy(yw + 1);
    for (uint64_t w = 0; w < xw; w++)
      for (int l = 0; l < 8; l++) ox[w].range(16 * l + 15, 16 * l) = mem.rd16(t.x + 16 * w + 2 * l);
    for (uint64_t w = 0; w < yw; w++)
      for (int l = 0; l < 8; l++) oy[w].range(16 * l + 15, 16 * l) = mem.rd16(t.y + 16 * w + 2 * l);
    if (!t.no_oracle) {
      // the C simulation reports duplicated reads (W-tiles) on stdout: keep it quiet
      std::fflush(stdout);
      const int saved = dup(1);
      const int devnull = open("/dev/null", O_WRONLY);
      dup2(devnull, 1);
      {
        hls::burst_maxi<PoolWord> px(ox.data()), py(oy.data());
        PoolingKernel(px, py, t.n, t.c, t.h, t.w, t.oh, t.ow, t.ph, t.pw, t.sh, t.sw,
                      t.pt, t.pl, t.dh, t.dw, t.type, t.lp, t.cip);
      }
      std::fflush(stdout);
      dup2(saved, 1);
      close(saved);
      close(devnull);
    }
#endif

    // ---- run ------------------------------------------------------------------------------
    mem.written.clear();
    wr64(R_X, t.x); wr64(R_Y, t.y);
    lite.write(R_BATCH, t.n); lite.write(R_CH, t.c); lite.write(R_IN_H, t.h); lite.write(R_IN_W, t.w);
    lite.write(R_OUT_H, t.oh); lite.write(R_OUT_W, t.ow); lite.write(R_POOL_H, t.ph);
    lite.write(R_POOL_W, t.pw); lite.write(R_STRIDE_H, t.sh); lite.write(R_STRIDE_W, t.sw);
    lite.write(R_PAD_TOP, t.pt); lite.write(R_PAD_LEFT, t.pl); lite.write(R_DIL_H, t.dh);
    lite.write(R_DIL_W, t.dw); lite.write(R_TYPE, t.type); lite.write(R_LP, t.lp);
    lite.write(R_CIP, t.cip);
    lite.write(R_GIE, 1); lite.write(R_IER, 1);
    if (lite.read(R_CH) != t.c || lite.read(R_Y) != (uint32_t)t.y || lite.read(R_CIP) != t.cip ||
        lite.read(R_Y + 4) != (uint32_t)(t.y >> 32) || lite.read(R_DIL_W) != t.dw)
      tb_fatal("register read-back mismatch");
    if (!(lite.read(R_CTRL) & 4)) tb_fatal("kernel not idle before start");

    lite.write(R_CTRL, 1);
    const uint64_t t0 = g_cycle;
    while (!top->__SYM__interrupt) {
      tick();
      if (g_cycle - t0 > max_cycles) {
        std::fprintf(stderr, "TIMEOUT %s after %llu cycles\n", t.label.c_str(),
                     (unsigned long long)(g_cycle - t0));
        return 0;
      }
    }
    const uint64_t done_at = g_cycle;
    const uint32_t ctrl = lite.read(R_CTRL);
    if (!(ctrl & 2)) tb_fatal("interrupt without ap_done (ctrl %x)", ctrl);
    if (!(ctrl & 4)) tb_fatal("not idle after done (ctrl %x)", ctrl);
    if (lite.read(R_CTRL) & 2) tb_fatal("ap_done not cleared on read");
    if (!(lite.read(R_ISR) & 1)) tb_fatal("ISR done bit not set");
    lite.write(R_ISR, 1);
    for (int i = 0; i < 4; i++) tick();
    if (top->__SYM__interrupt) tb_fatal("interrupt still asserted after ISR clear");
    if (!rd.idle() || !wr.idle())
      tb_fatal("AXI traffic outstanding after ap_done (read: %d bursts%s; write: %d bursts, %d B due%s)",
               (int)rd.q.size(), rd.rv ? ", R valid" : "", (int)wr.aw.size(), (int)wr.b_due.size(),
               wr.bv ? ", B valid" : "");

    // ---- check --------------------------------------------------------------------------
    size_t bad = 0;
    auto report = [&](const char* what, uint64_t e, uint16_t got, uint16_t exp) {
      if (bad < 5)
        std::fprintf(stderr, "  MISMATCH %s %s elem %llu got=%04x exp=%04x\n", t.label.c_str(),
                     what, (unsigned long long)e, got, exp);
      bad++;
    };
    const bool expect_none = t.no_oracle || !in_contract(t);
#ifdef PL_HLS_ORACLE
    if (!t.no_oracle)
      for (uint64_t w = 0; w < yw; w++)
        for (int l = 0; l < 8; l++) {
          const uint16_t exp = (uint16_t)oy[w].range(16 * l + 15, 16 * l).to_uint();
          const uint16_t got = mem.rd16(t.y + 16 * w + 2 * l);
          if (got != exp) report("hls", 8 * w + l, got, exp);
        }
#endif
    if (!t.yv.empty())                                  // fixture: every element vs y.hex
      for (uint64_t e = 0; e < y_elems(t) && e < t.yv.size(); e++) {
        const uint16_t got = mem.rd16(t.y + 2 * e);
        if (got != t.yv[e]) report("fixture", e, got, t.yv[e]);
      }
    // nothing written outside the output elements (nothing at all when no job)
    if (!expect_none)
      for (uint64_t i = 0; i < 2 * y_elems(t); i++) mem.unmark(t.y + i);
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
      std::printf("FAIL  %-44s  %zu wrong, %zu stray bytes  (--case \"%u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u\")\n",
                  t.label.c_str(), bad, stray, t.n, t.c, t.h, t.w, t.oh, t.ow, t.ph, t.pw, t.sh, t.sw,
                  t.pt, t.pl, t.dh, t.dw, t.type, t.lp, t.cip);
      return 0;
    }
    if (!quiet)
      std::printf("PASS  %-44s  %ux%ux%ux%u -> %ux%u  k%ux%u s%ux%u p%u,%u d%ux%u t%u lp%u cip%u  %llu cyc\n",
                  t.label.c_str(), t.n, t.c, t.h, t.w, t.oh, t.ow, t.ph, t.pw, t.sh, t.sw, t.pt, t.pl,
                  t.dh, t.dw, t.type, t.lp, t.cip, (unsigned long long)cyc);
    return std::max<uint64_t>(cyc, 1);
  }
};

// ---------------------------------------------------------------------------
// Case construction
// ---------------------------------------------------------------------------
static uint64_t g_region = 0x0000'0008'0000'0000ull;   // above 4 GiB

static void place(Case& t, std::mt19937& rng) {
  auto next = [&](uint64_t words) {
    uint64_t base = g_region;
    g_region += (16 * (words + 2) + 0x10000) & ~0xFFFull;
    g_region += 0x1000 * (rng() % 4) + 16 * (rng() % 256);
    g_region &= ~15ull;
    return base;
  };
  t.x = next(words_of_elems(x_elems(t)) + 1);
  t.y = next(words_of_elems(y_elems(t)) + 1);
}

static uint32_t out_dim(uint32_t in, uint32_t pad0, uint32_t pad1, uint32_t reach, uint32_t s) {
  const int64_t v = (int64_t)in + pad0 + pad1 - reach - 1;
  return v < 0 ? 0 : (uint32_t)(v / s + 1);
}

// A random job inside the contract (windows that overlap the input, as ONNX
// pooling has), sometimes outside it or of zero size.
static Case random_case(std::mt19937& rng) {
  auto U = [&](uint32_t lo, uint32_t hi) { return std::uniform_int_distribution<uint32_t>(lo, hi)(rng); };
  Case t;
  for (int tries = 0; tries < 1000; tries++) {
    const uint32_t kind = U(0, 39);
    t.type = (U(0, 19) == 0) ? U(3, 5) : U(0, 2);
    t.lp   = (U(0, 9) == 0) ? U(0, 3) : U(1, 2);
    t.cip  = U(0, 1);
    t.n    = (U(0, 4) == 0) ? U(2, 3) : 1;
    t.c    = (U(0, 2) == 0) ? U(9, 20) : U(1, 8);
    t.ph   = U(1, 7);
    t.pw   = U(1, 7);
    t.dh   = (t.ph == 1) ? U(1, 4) : ((U(0, 2) == 0) ? U(1, 15 / (t.ph - 1)) : 1);
    t.dw   = (t.pw == 1) ? U(1, 4) : ((U(0, 2) == 0) ? U(1, 63 / (t.pw - 1)) : 1);
    t.sh   = (U(0, 5) == 0) ? U(5, 12) : U(1, 4);
    t.sw   = (U(0, 5) == 0) ? 8 * U(1, 2) : ((U(0, 5) == 0) ? U(5, 20) : U(1, 4));
    const uint32_t rh = (t.ph - 1) * t.dh, rw = (t.pw - 1) * t.dw;
    t.pt   = U(0, rh);
    t.pl   = U(0, rw);
    const uint32_t pb = U(0, rh), pr = U(0, rw);
    t.h    = (U(0, 3) == 0) ? U(1, 6) : U(1, 30);
    t.w    = (U(0, 3) == 0) ? U(1, 9) : ((U(0, 2) == 0) ? U(60, 200) : U(1, 64));
    if (U(0, 9) == 0) {                                  // a global pool
      t.ph = std::min<uint32_t>(t.h, 7); t.pw = std::min<uint32_t>(t.w, 7);
      t.h = t.ph; t.w = t.pw; t.sh = t.sw = 1; t.dh = t.dw = 1; t.pt = t.pl = 0;
      t.oh = t.ow = 1;
    } else {
      t.oh = out_dim(t.h, t.pt, pb, rh, t.sh);
      t.ow = out_dim(t.w, t.pl, pr, rw, t.sw);
      if (t.oh == 0 || t.ow == 0) continue;
      // every window must touch the input (ONNX: pads smaller than the window)
      if ((uint64_t)(t.oh - 1) * t.sh >= (uint64_t)t.h + t.pt) continue;
      if ((uint64_t)(t.ow - 1) * t.sw >= (uint64_t)t.w + t.pl) continue;
    }
    if (kind == 0) {                                     // outside the contract
      if (U(0, 1)) t.ph = U(8, 9); else t.pw = U(8, 9);
      if (U(0, 3) == 0) { t.ph = 3; t.dh = 8; }
    } else if (kind == 1) {                              // a zero size: no job
      switch (U(0, 3)) { case 0: t.n = 0; break; case 1: t.c = 0; break;
                         case 2: t.oh = 0; break; default: t.ow = 0; }
      t.no_oracle = true;
    }
    if (x_elems(t) + y_elems(t) > 400000) continue;
    break;
  }
  t.data_mode = (int)U(0, 2);
  char buf[128];
  std::snprintf(buf, sizeof buf, "rand_%ux%ux%ux%u_k%ux%u_s%ux%u_t%u", t.n, t.c, t.h, t.w, t.ph,
                t.pw, t.sh, t.sw, t.type);
  t.label = buf;
  return t;
}

// ---------------------------------------------------------------------------
// Fixtures: manifest.txt + test_NN_{x,y}.hex (kernels/pool TestPoolingSim --dump-data)
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
    if (tok.size() < 19) continue;
    Case t;
    const int idx = std::stoi(tok[0]);
    uint32_t* f17[] = {&t.n, &t.c, &t.h, &t.w, &t.oh, &t.ow, &t.ph, &t.pw, &t.sh, &t.sw,
                       &t.pt, &t.pl, &t.dh, &t.dw, &t.type, &t.lp, &t.cip};
    for (int i = 0; i < 17; i++) *f17[i] = (uint32_t)std::stoul(tok[1 + i]);
    t.label = "fx" + tok[0] + "_" + tok.back();
    if (t.label.size() > 44) t.label.resize(44);
    char pre[32];
    std::snprintf(pre, sizeof pre, "/test_%02d_", idx);
    t.xv = read_hex(dir + pre + "x.hex");
    t.yv = read_hex(dir + pre + "y.hex");
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
#ifndef PL_HLS_ORACLE
  std::printf("note: built without the HLS oracle (no Vitis HLS headers): fixtures only\n");
  if (n_random) tb_fatal("--random needs the HLS oracle");
#endif

  std::vector<Case> cases;
  if (!fixtures.empty())
    for (auto& t : load_fixtures(fixtures)) cases.push_back(std::move(t));
  if (!one_case.empty()) {
    std::istringstream is(one_case);
    Case t;
    is >> t.n >> t.c >> t.h >> t.w >> t.oh >> t.ow >> t.ph >> t.pw >> t.sh >> t.sw >> t.pt >> t.pl
       >> t.dh >> t.dw >> t.type >> t.lp >> t.cip;
    t.label = "case";
    cases.push_back(t);
  }
  if (perf) {
    auto add = [&](const char* l, uint32_t c, uint32_t h, uint32_t w, uint32_t k, uint32_t s,
                   uint32_t p, uint32_t type) {
      Case t;
      t.label = l; t.c = c; t.h = h; t.w = w; t.ph = t.pw = k; t.sh = t.sw = s; t.pt = t.pl = p;
      t.type = type;
      t.oh = out_dim(h, p, p, k - 1, s); t.ow = out_dim(w, p, p, k - 1, s);
      cases.push_back(t);
    };
    add("perf_max3x3s2_64x112x112", 64, 112, 112, 3, 2, 1, 0);   // the ResNet-18 stem pool
    add("perf_max2x2s2_64x56x56", 64, 56, 56, 2, 2, 0, 0);
    add("perf_max3x3s1_32x28x28", 32, 28, 28, 3, 1, 1, 0);
    add("perf_gavg7x7_512", 512, 7, 7, 7, 1, 0, 1);              // ResNet-18 global average
    add("perf_gavg7x7_1024", 1024, 7, 7, 7, 1, 0, 1);            // MobileNet v1
    add("perf_avg2x2s2_16x28x28", 16, 28, 28, 2, 2, 0, 1);       // LeNet
  }
  for (int i = 0; i < n_random; i++) cases.push_back(random_case(crng));
  if (cases.empty()) tb_fatal("nothing to run (use --fixtures, --random, --case or --perf)");

  int pass = 0, fail = 0;
  for (auto& t : cases) {
    set_timing(perf ? "fast" : timing);
    place(t, crng);
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
