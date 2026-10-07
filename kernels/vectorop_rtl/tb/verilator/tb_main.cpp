// ---------------------------------------------------------------------------
// tb_main.cpp — Verilator testbench for the RTL VectorOPKernel.
//
// Runs element-wise jobs through the real AXI interfaces against sparse-memory
// slave models with randomised timing and checks, per case:
//   * with the HLS oracle (VO_HLS_ORACLE: the HLS kernel's own C++,
//     kernels/vectorop/kernel/VectorOP.cpp, built with the Vitis HLS headers):
//     every byte of the output region equals what the HLS kernel leaves there,
//     run on the same DDR image — the written elements, the zeroed tail
//     lanes and the untouched gaps alike;
//   * fixtures (kernels/vectorop TestSimulation --dump-data): every element
//     inside a run equals c.hex, every tail lane is 0;
//   * that nothing outside the output words was written, that a unary op
//     issued nothing on gmem1, the AXI protocol on all ports, ap_done /
//     interrupt behaviour.
//
// Usage:
//   Vtb [--fixtures DIR] [--random N] [--seed S] [--timing fast|rand|slow]
//       [--case "size op outer a_inc b_inc act [mode [alpha [smx_cm smx_cfg smx_mask]]]"]
//       [--perf] [--quiet] [--max-cycles N] [--trace FILE]
// (--case may be repeated; data mode 0: values in +-8, 1: any, 2: any with
// edge values, 3: a[i] = i mod 2^16 — size 65536 covers every Q8.8 input.)
// The softmax ops (10 row mode, 11 column mode; SOFTMAX_PLAN) write with
// c_inc = b_inc; column mode reads size key rows of outer queries and writes
// outer & ~15 rows of size.
// ---------------------------------------------------------------------------
#include <algorithm>
#include <cstdarg>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>

#include "VVectorOPKernel.h"
#include "axi_models.h"
#ifdef VO_HLS_ORACLE
#include "VectorOP.h"
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

// Register map (xvectoropkernel_hw.h)
enum : uint8_t {
  R_CTRL = 0x00, R_GIE = 0x04, R_IER = 0x08, R_ISR = 0x0C,
  R_A = 0x10, R_B = 0x1C, R_C = 0x28, R_SIZE = 0x34, R_OP = 0x3C, R_OUTER = 0x44,
  R_AINC = 0x4C, R_BINC = 0x54, R_ACT = 0x5C, R_ALPHA = 0x64,
  R_SCM = 0x6C, R_SCFG = 0x74, R_SMASK = 0x7C,
};

// ---------------------------------------------------------------------------
// Test case and the kernel's geometry (VectorOP.cpp run_geometry / load_words)
// ---------------------------------------------------------------------------
struct Case {
  std::string label;
  uint32_t size = 1, op = 0, outer = 1, a_inc = 0, b_inc = 0, act = 0, alpha = 0;
  uint32_t smx_cm = 0, smx_cfg = 0, smx_mask = 0;
  uint64_t a = 0, b = 0, c = 0;          // byte addresses
  std::vector<uint16_t> av, bv, cv;      // fixture data (empty: random)
  int data_mode = 0;
};

static uint64_t n_words(uint32_t size) { return ((uint64_t)size + 7) / 8; }
static bool unary(uint32_t op) { return op >= 4; }
static bool softmax(uint32_t op) { return op == 10 || op == 11; }
static uint32_t c_inc_of(const Case& t) { return softmax(t.op) ? t.b_inc : t.a_inc + t.b_inc; }
// rows of the output: column-mode softmax writes outer & ~15 of them
static uint32_t out_rows(const Case& t) { return t.op == 11 ? (t.outer & ~15u) : t.outer; }

// Words of an operand / output that the kernel touches, from its base.
static uint64_t span_words(const Case& t, uint32_t inc, bool input) {
  if (t.size == 0 || t.outer == 0) return 0;
  const uint64_t nw = n_words(t.size);
  if (t.op == 11) {             // column softmax: the C++ reads size rows of outer queries
    if (input) return (uint64_t)(t.size - 1) * (inc / 8) + n_words(t.outer);
    const uint32_t rows = out_rows(t);
    return rows ? (uint64_t)(rows - 1) * (inc / 8) + nw : 0;
  }
  if (input && t.outer > 1 && inc == 0 && nw <= 256) return nw;
  if (t.outer == 1 || (inc == t.size && t.size % 8 == 0)) return (uint64_t)t.outer * nw;
  return (uint64_t)(t.outer - 1) * (inc / 8) + nw;
}

// The word streams of a job, in order (VectorOP.cpp load_words / store_words):
// for every word the kernel processes, its word index from the operand's base
// and its valid lanes (the rest of a run's last word is zeroed on input).
struct WordRef { uint64_t w; int lanes; };
static std::vector<WordRef> word_stream(const Case& t, uint32_t inc, bool input) {
  std::vector<WordRef> out;
  if (t.size == 0 || t.outer == 0) return out;
  const uint32_t nw = (uint32_t)n_words(t.size);
  const int tail = (t.size % 8) ? (int)(t.size % 8) : 8;
  auto run = [&](uint64_t base, uint32_t words) {
    for (uint32_t w = 0; w < words; w++) out.push_back({base + w, (w + 1 == words) ? tail : 8});
  };
  if (input && t.outer > 1 && inc == 0 && nw <= 256) {
    for (uint32_t o = 0; o < t.outer; o++) run(0, nw);
  } else if (t.outer == 1 || (inc == t.size && t.size % 8 == 0)) {
    run(0, t.outer * nw);                               // 32-bit product, as the kernel
  } else {
    uint32_t base = 0;
    for (uint32_t o = 0; o < t.outer; o++, base += inc / 8) run(base, nw);
  }
  return out;
}

// The act of a DIV job on a positive value (the corner below: 0x7FFF): ReLU6
// clips it; ReLU, LeakyReLU, SiLU and both GELUs leave values >= 8.36 as they are.
static uint16_t activate(uint16_t v, uint32_t act) {
  const bool neg = v & 0x8000;
  if (act == 1) return neg ? 0 : v;
  if (act == 2) return neg ? 0 : (v > 0x0600 ? 0x0600 : v);
  return v;
}

// ---------------------------------------------------------------------------
// Simulation harness
// ---------------------------------------------------------------------------
struct Tb {
  std::unique_ptr<VerilatedContext> ctx;
  std::unique_ptr<VVectorOPKernel>  top;
  Memory        mem;
  Timing        tim;
  std::mt19937  rng{1};
  AxiReadSlave  rd0{}, rd1{};
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
    top.reset(new VVectorOPKernel{ctx.get()});
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
    // the outstanding bursts the IP declares (vo_pkg RD_OUTS / WR_OUTS, package_ip.tcl;
    // fact rtl.axi_masters)
    rd0.outs_limit = rd1.outs_limit = 16;  wr.outs_limit = 16;
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

  uint16_t rnd16(int mode) {
    static const uint16_t edge[] = {0x8000, 0x7FFF, 0x0000, 0x0001, 0xFFFF, 0x0100, 0xFF00,
                                    0x0600, 0x05FF, 0x0601, 0x8001, 0x00FF, 0xFF01, 0x4000};
    switch (mode) {
      case 1:  return (uint16_t)rng();
      case 2:  if ((rng() & 3) == 0) return edge[rng() % (sizeof edge / sizeof edge[0])];
               return (uint16_t)rng();
      default: return (uint16_t)(int16_t)std::uniform_int_distribution<int>(-2048, 2048)(rng);
    }
  }

  // Runs one case; returns the kernel cycle count (0 on failure).
  uint64_t run(Case& t) {
    const uint32_t c_inc = c_inc_of(t);
    const uint64_t a_w = span_words(t, t.a_inc, true) + 1;
    const uint64_t b_w = span_words(t, t.b_inc, true) + 1;
    const uint64_t c_w = span_words(t, c_inc, false);

    // ---- place data: every word the kernel may read --------------------------------
    for (uint64_t i = 0; i < 8 * a_w; i++)
      mem.wr16(t.a + 2 * i, i < t.av.size() ? t.av[i]
                            : t.data_mode == 3 ? (uint16_t)i : rnd16(t.data_mode));
    for (uint64_t i = 0; i < 8 * b_w; i++)
      mem.wr16(t.b + 2 * i, i < t.bv.size() ? t.bv[i] : rnd16(t.data_mode));
    for (uint64_t i = 0; i < 8 * c_w; i++) mem.wr16(t.c + 2 * i, 0xDEAD);   // poison

#ifdef VO_HLS_ORACLE
    // ---- the HLS kernel on the same image ------------------------------------------
    auto words_of = [&](uint64_t base, uint64_t n) {
      std::vector<VecWord> v(std::max<uint64_t>(n, 1) + 1);
      for (uint64_t w = 0; w < n; w++)
        for (int l = 0; l < 8; l++) v[w].range(16 * l + 15, 16 * l) = mem.rd16(base + 16 * w + 2 * l);
      return v;
    };
    std::vector<VecWord> oa = words_of(t.a, a_w), ob = words_of(t.b, b_w), oc = words_of(t.c, c_w);
    {
      hls::burst_maxi<VecWord> pa(oa.data()), pb(ob.data()), pc(oc.data());
      VectorOPKernel(pa, pb, pc, t.size, t.op, t.outer, t.a_inc, t.b_inc, t.act, t.alpha,
                     t.smx_cm, t.smx_cfg, t.smx_mask);
    }
    // HLS C simulation and the synthesised HLS kernel differ in one DIV case:
    // a = -128 (0x8000), b = -1/256 (0xFFFF).  C simulation divides in a 24-bit
    // sdiv, where (a << 8) / b = +2^23 wraps to -2^23 (saturating to 0x8000);
    // the hardware's divider is 25 bits wide (VectorOPKernel_sdiv_25s_16s_25),
    // so the quotient saturates to 0x7FFF.  The board runs the hardware: expect
    // 0x7FFF (then the activation) where the last write of an element had it.
    if (t.op == 3) {
      const auto sa = word_stream(t, t.a_inc, true), sb = word_stream(t, t.b_inc, true),
                 sc = word_stream(t, t.a_inc + t.b_inc, false);
      if (sa.size() != sc.size() || sb.size() != sc.size()) tb_fatal("word stream lengths differ");
      std::unordered_map<uint64_t, bool> corner;        // c element -> its last write had it
      for (size_t k = 0; k < sc.size(); k++)
        for (int l = 0; l < 8; l++) {
          const bool v = l < sa[k].lanes;               // masked lanes are 0 / 0 -> 0
          corner[8 * sc[k].w + l] =
              v && mem.rd16(t.a + 16 * sa[k].w + 2 * l) == 0x8000 &&
                   mem.rd16(t.b + 16 * sb[k].w + 2 * l) == 0xFFFF;
        }
      for (auto& kv : corner)
        if (kv.second) oc[kv.first / 8].range(16 * (kv.first % 8) + 15, 16 * (kv.first % 8)) =
                           activate(0x7FFF, t.act);
    }
#endif

    // ---- run ------------------------------------------------------------------------------
    mem.written.clear();
    const uint64_t rd1_bursts0 = rd1.bursts_total;
    wr64(R_A, t.a); wr64(R_B, t.b); wr64(R_C, t.c);
    lite.write(R_SIZE, t.size); lite.write(R_OP, t.op); lite.write(R_OUTER, t.outer);
    lite.write(R_AINC, t.a_inc); lite.write(R_BINC, t.b_inc); lite.write(R_ACT, t.act);
    lite.write(R_ALPHA, t.alpha);
    lite.write(R_SCM, t.smx_cm); lite.write(R_SCFG, t.smx_cfg); lite.write(R_SMASK, t.smx_mask);
    lite.write(R_GIE, 1); lite.write(R_IER, 1);
    if (lite.read(R_SIZE) != t.size || lite.read(R_C) != (uint32_t)t.c ||
        lite.read(R_ACT) != t.act || lite.read(R_ALPHA) != t.alpha ||
        lite.read(R_C + 4) != (uint32_t)(t.c >> 32) || lite.read(R_SCM) != t.smx_cm ||
        lite.read(R_SCFG) != t.smx_cfg || lite.read(R_SMASK) != t.smx_mask)
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
    if (!rd0.idle() || !rd1.idle() || !wr.idle()) tb_fatal("AXI traffic outstanding after ap_done");
    if (unary(t.op) && rd1.bursts_total != rd1_bursts0) tb_fatal("%s: unary op read gmem1", t.label.c_str());

    // ---- check --------------------------------------------------------------------------
    size_t bad = 0;
    auto report = [&](const char* what, uint64_t e, uint16_t got, uint16_t exp) {
      if (bad < 5)
        std::fprintf(stderr, "  MISMATCH %s %s elem %llu got=%04x exp=%04x\n", t.label.c_str(),
                     what, (unsigned long long)e, got, exp);
      bad++;
    };
#ifdef VO_HLS_ORACLE
    for (uint64_t w = 0; w < c_w; w++)
      for (int l = 0; l < 8; l++) {
        const uint16_t exp = (uint16_t)oc[w].range(16 * l + 15, 16 * l).to_uint();
        const uint16_t got = mem.rd16(t.c + 16 * w + 2 * l);
        if (got != exp) report("hls", 8 * w + l, got, exp);
      }
#endif
    if (!t.cv.empty()) {                                // fixture: runs vs c.hex, tails 0
      const uint64_t nw = n_words(t.size);
      const uint32_t rows = out_rows(t);
      for (uint32_t o = 0; o < rows; o++)
        for (uint64_t i = 0; i < 8 * nw; i++) {
          const uint64_t e = (uint64_t)o * c_inc + i;
          const uint16_t got = mem.rd16(t.c + 2 * e);
          if (i < t.size) {
            if (e < t.cv.size() && got != t.cv[e]) report("fixture", e, got, t.cv[e]);
          } else if (o + 1 == rows && got != 0) {
            report("tail", e, got, 0);
          }
        }
    }
    // nothing written outside the output words
    for (uint64_t w = 0; w < c_w; w++)
      for (int i = 0; i < 16; i++) mem.unmark(t.c + 16 * w + i);
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
      std::printf("FAIL  %-40s  %zu wrong, %zu stray bytes\n", t.label.c_str(), bad, stray);
      return 0;
    }
    if (!quiet) {
      const double words = (double)out_rows(t) * (double)n_words(t.size);
      std::printf("PASS  %-40s  size=%u op=%u outer=%u inc=%u/%u act=%u  %llu cyc  %.2f word/cyc\n",
                  t.label.c_str(), t.size, t.op, t.outer, t.a_inc, t.b_inc, t.act,
                  (unsigned long long)cyc, words / (double)std::max<uint64_t>(cyc, 1));
    }
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
  const uint32_t c_inc = c_inc_of(t);
  t.a = next(span_words(t, t.a_inc, true) + 1);
  t.b = next(span_words(t, t.b_inc, true) + 1);
  t.c = next(span_words(t, c_inc, false));
}

static uint32_t round8(uint32_t v) { return (v + 7) / 8 * 8; }

// A softmax case (op 10 / 11) within VectorOP.h's limits (rows <= 2048,
// keys <= 1024): scales around the models' (Cs 10 .. 30, f_p 8 .. 15), at
// times any register value (Cm's ignored high bits, Cs to 63, f_p to 31);
// valid lengths from smx_mask, causal (period) or not.
static void random_softmax(Case& t, std::mt19937& rng) {
  auto U = [&](uint32_t lo, uint32_t hi) { return std::uniform_int_distribution<uint32_t>(lo, hi)(rng); };
  t.act = 0;
  if (t.op == 10) {
    t.size  = (U(0, 2) == 0) ? U(1, 24) : U(1, 2048);
    t.outer = (t.size > 1024) ? U(1, 4) : U(1, 24);
    const uint32_t step = (t.size % 8 == 0 && U(0, 1)) ? t.size : round8(t.size) + 8 * U(0, 2);
    t.a_inc = (U(0, 7) == 0) ? 0 : step;                // 0: every row the same input (replay)
    t.b_inc = (t.size % 8 == 0 && U(0, 1)) ? t.size : round8(t.size) + 8 * U(0, 2);
  } else {
    t.size  = (U(0, 2) == 0) ? U(1, 40) : U(1, 1024);
    t.outer = 16 * U(1, t.size > 256 ? 3 : 8) + ((U(0, 5) == 0) ? U(1, 15) : 0);
    t.a_inc = round8(t.outer) + 8 * U(0, 2);
    t.b_inc = (t.size % 8 == 0 && U(0, 1)) ? t.size : round8(t.size) + 8 * U(0, 2);
  }
  t.smx_cm  = (U(0, 9) == 0) ? U(0, 0xFFFFFFFFu) : U(1u << 23, (1u << 24) - 1);
  const uint32_t cs = (U(0, 9) == 0) ? U(0, 63) : U(10, 30);
  const uint32_t fp = (U(0, 9) == 0) ? U(0, 31) : U(8, 15);
  t.smx_cfg = cs | (fp << 8) | ((U(0, 9) == 0) ? (U(0, 0xFFFF) << 16) : 0);
  const uint32_t valid0 = (U(0, 3) == 0) ? U(0, t.size + 4) : (U(0, 9) == 0 ? 0xFFFF : t.size);
  const uint32_t period = (U(0, 2) == 0) ? U(1, 40) : 0;
  t.smx_mask = valid0 | (period << 16);
}

// A random case: mostly within the documented contract (bases 16-byte
// aligned, a_inc / b_inc 0 or multiples of 8), sometimes outside it (other
// increments, op / act codes beyond the enums, alpha's ignored high bits) —
// the oracle is exact either way.
static Case random_case(std::mt19937& rng) {
  auto U = [&](uint32_t lo, uint32_t hi) { return std::uniform_int_distribution<uint32_t>(lo, hi)(rng); };
  Case t;
  if (U(0, 6) == 0) {                                  // softmax
    t.op = U(10, 11);
    random_softmax(t, rng);
    t.data_mode = (int)U(0, 2);
    char buf[128];
    std::snprintf(buf, sizeof buf, "rand_smx%u_s%u_o%u_i%u_%u_%x_%x_%x", t.op, t.size, t.outer,
                  t.a_inc, t.b_inc, t.smx_cm, t.smx_cfg, t.smx_mask);
    t.label = buf;
    return t;
  }
  t.op    = (U(0, 19) == 0) ? U(12, 14) : U(0, 9);
  t.act   = (U(0, 19) == 0) ? U(7, 9) : U(0, 6);
  t.alpha = (U(0, 9) == 0) ? U(0, 0xFFFFFFFFu) : U(0, 0xFFFF);
  const uint32_t shape = U(0, 9);
  if (shape <= 2) {                                    // one run
    t.size = (U(0, 2) == 0) ? U(1, 40) : U(1, 6000);
    t.outer = 1;
  } else if (shape <= 5) {                             // broadcast: one operand repeats
    t.size = (U(0, 2) == 0) ? U(1, 24) : U(1, 400);
    t.outer = U(2, 60);
    const uint32_t step = (U(0, 2) == 0 && t.size % 8 == 0) ? t.size : round8(t.size) + 8 * U(0, 3);
    if (U(0, 1)) { t.a_inc = step; t.b_inc = 0; } else { t.a_inc = 0; t.b_inc = step; }
  } else if (shape <= 7) {                             // both advance (strided)
    t.size = U(1, 300);
    t.outer = U(2, 40);
    t.a_inc = t.b_inc = (U(0, 1) && t.size % 8 == 0) ? t.size : round8(t.size) + 8 * U(0, 2);
    if (U(0, 3) == 0) t.b_inc = round8(t.size) + 8 * U(0, 4);
  } else if (shape == 8) {                             // the replay bound (256 words)
    t.size = 8 * U(253, 259) - U(0, 7);
    t.outer = U(2, 4);
    t.a_inc = 0; t.b_inc = 8 * U(0, 1) * (uint32_t)n_words(t.size);
  } else {                                             // outside the contract / corner
    t.size = U(0, 3) == 0 ? 0 : U(1, 100);
    t.outer = U(0, 6);
    t.a_inc = U(0, 120); t.b_inc = U(0, 120);
  }
  t.data_mode = (int)U(0, 2);
  char buf[96];
  std::snprintf(buf, sizeof buf, "rand_s%u_op%u_o%u_i%u_%u_act%u_al%x", t.size, t.op, t.outer,
                t.a_inc, t.b_inc, t.act, t.alpha);
  t.label = buf;
  return t;
}

// ---------------------------------------------------------------------------
// Fixtures: manifest.txt + test_NN_{a,b,c}.hex (kernels/vectorop TestSimulation --dump-data)
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
    if (tok.size() < 8) continue;
    Case t;
    const int idx = std::stoi(tok[0]);
    t.size = std::stoul(tok[1]); t.op = std::stoul(tok[2]); t.outer = std::stoul(tok[3]);
    t.a_inc = std::stoul(tok[4]); t.b_inc = std::stoul(tok[5]); t.act = std::stoul(tok[6]);
    if (tok.size() >= 9) t.alpha = std::stoul(tok[7]);   // manifests before alpha: 0
    if (tok.size() >= 12) {                              // manifests before softmax: 0
      t.smx_cm = std::stoul(tok[8]); t.smx_cfg = std::stoul(tok[9]); t.smx_mask = std::stoul(tok[10]);
    }
    t.label = "fx" + tok[0] + "_" + tok.back();
    if (t.label.size() > 40) t.label.resize(40);
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
  std::string fixtures, trace, timing = "rand";
  std::vector<std::string> one_cases;
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
    else if (a == "--case") one_cases.push_back(next());
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
#ifndef VO_HLS_ORACLE
  std::printf("note: built without the HLS oracle (no Vitis HLS headers): fixtures only\n");
  if (n_random) tb_fatal("--random needs the HLS oracle");
#endif

  std::vector<Case> cases;
  if (!fixtures.empty())
    for (auto& t : load_fixtures(fixtures)) cases.push_back(std::move(t));
  for (const auto& one_case : one_cases) {
    std::istringstream is(one_case);
    Case t;
    is >> t.size >> t.op >> t.outer >> t.a_inc >> t.b_inc >> t.act;
    if (!(is >> t.data_mode)) t.data_mode = 0;
    if (!(is >> t.alpha)) t.alpha = 0;
    if (!(is >> t.smx_cm >> t.smx_cfg >> t.smx_mask)) t.smx_cm = t.smx_cfg = t.smx_mask = 0;
    char buf[96];
    std::snprintf(buf, sizeof buf, "case_s%u_op%u_act%u_m%d_al%x_%x_%x_%x", t.size, t.op, t.act,
                  t.data_mode, t.alpha, t.smx_cm, t.smx_cfg, t.smx_mask);
    t.label = buf;
    cases.push_back(t);
  }
  if (perf) {
    auto add = [&](const char* l, uint32_t size, uint32_t op, uint32_t outer, uint32_t ai,
                   uint32_t bi, uint32_t act = 0) {
      Case t;
      t.label = l; t.size = size; t.op = op; t.outer = outer; t.a_inc = ai; t.b_inc = bi;
      t.act = act;
      if (softmax(op)) {                                // q88 scores, P at 2^-15, every key valid
        t.smx_cm = 12102203; t.smx_cfg = 19 | (15 << 8); t.smx_mask = size;
      }
      cases.push_back(t);
    };
    add("perf_add_64k", 65536, 0, 1, 0, 0);
    add("perf_mul_64k", 65536, 2, 1, 0, 0);
    add("perf_relu_64k", 65536, 4, 1, 0, 0);
    add("perf_div_8k", 8192, 3, 1, 0, 0);
    add("perf_bias_64x1024", 1024, 0, 64, 1024, 0);        // replay of a 1024-element b
    add("perf_bcast_12_stride16", 12, 0, 1000, 16, 0);    // two-word runs
    add("perf_bcast_8_x4096", 8, 0, 4096, 8, 0);          // contiguous a, one-word b replay
    add("perf_bcast_rows_5000", 5000, 0, 16, 5008, 0);    // b over the replay bound
    add("perf_gelu_64k", 65536, 8, 1, 0, 0);              // an activation op (vo_act)
    add("perf_add_silu_64k", 65536, 0, 1, 0, 0, 4);       // ADD + act SILU
    add("perf_smx_bert_256x256", 256, 10, 256, 256, 256);  // BERT: a head's rows
    add("perf_smx_t_vit_1024x64", 1024, 11, 64, 1024, 1024);   // SmolVLM: 1024 keys, 4 blocks
    add("perf_smx_t_llm_256x256", 256, 11, 256, 256, 256);     // a prefill-256 head
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
