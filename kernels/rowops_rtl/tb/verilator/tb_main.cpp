// ---------------------------------------------------------------------------
// tb_main.cpp — Verilator testbench for the RTL RowOpsKernel.
//
// Runs row-op jobs through the real AXI interfaces against the sparse-memory
// slave models with randomised timing and checks, per case:
//   * every byte the oracle (rowops_ref.h: the C++ port of
//     inference-scheduler/src/rowops_spec.py) says the job writes equals the
//     memory after the run — the elements, the zeroed tail lanes, the strobed
//     granules — and nothing else was written;
//   * fixtures (scripts/gen_fixtures.py, from the Python specification): the
//     whole C region equals test_NN_c.hex (the untouched bytes keep the poison);
//   * the AXI protocol on the port, the outstanding limits, ap_done /
//     interrupt behaviour.
//
// Usage:
//   Vtb [--fixtures DIR] [--random N] [--seed S] [--timing fast|rand|slow]
//       [--case "op n c1 c2 [mode]"] [--perf] [--quiet] [--max-cycles N] [--trace FILE]
// (--case: an op code with the kinds / flags in its upper bits as the register
// takes them; strides, parameters and data are chosen as for a random case.)
// ---------------------------------------------------------------------------
#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>

#include "VRowOpsKernel.h"
#include "axi_models.h"
#include "rowops_ref.h"
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

// Register map (xrowopskernel_hw.h)
enum : uint8_t {
  R_CTRL = 0x00, R_GIE = 0x04, R_IER = 0x08, R_ISR = 0x0C,
  R_OP = 0x10, R_N = 0x14, R_ROWS = 0x18, R_BLOCKS = 0x1C,
  R_A = 0x20, R_AS1 = 0x28, R_AS2 = 0x2C, R_B = 0x30, R_BS1 = 0x38, R_BS2 = 0x3C,
  R_C = 0x40, R_CS1 = 0x48, R_CS2 = 0x4C, R_P = 0x50, R_EPS = 0x58,
  R_AUX = 0x60, R_BQSH = 0x64, R_WLO = 0x68, R_WHI = 0x6C,
};

static const uint8_t POISON = 0xDE;

// ---------------------------------------------------------------------------
// A case: the job plus the DDR images of its regions (fixtures) or how to fill
// them (random).  Region extents are derived from the geometry.
// ---------------------------------------------------------------------------
struct Region { uint64_t bytes = 0; std::vector<uint8_t> img; };   // img empty: random fill

struct Case {
  std::string label;
  ro::Job j;
  Region ra, rb, rp, rc;
  std::vector<uint8_t> expect_c;          // fixture: the whole C region after the run
  int data_mode = 0;                      // 0 small values, 1 any, 2 any + edges
  // random-case parameter choices (kept so a P image can be built once the job is placed)
  std::vector<ro::Param> params;
  bool fixture = false;
};

static uint64_t round16(uint64_t b) { return (b + 15) & ~15ull; }
static int kbytes(ro::Kind k) { return k == ro::I16 ? 2 : 4; }

// bytes of a stream from its base to the end of its last row's last word
static uint64_t stream_extent(const ro::Job& j, uint64_t s1, uint64_t s2, uint64_t row_bytes) {
  if (j.n == 0 || j.c1 == 0 || j.c2 == 0) return 0;
  return (uint64_t)(j.c2 - 1) * s2 + (uint64_t)(j.c1 - 1) * s1 + round16(row_bytes);
}

static void extents(Case& t) {
  ro::Job& j = t.j;
  const uint32_t op = j.code();
  if (op == ro::TLOAD) {
    t.ra.bytes = stream_extent(j, j.a_s1, j.a_s2, (uint64_t)kbytes(j.ak()) * j.n);
    t.rb.bytes = 0; t.rc.bytes = 0; t.rp.bytes = 0;
    return;
  }
  if (op == ro::XPOSE) {
    const uint32_t R = j.aux & 15, L = j.n;
    t.ra.bytes = stream_extent(j, j.a_s1, j.a_s2, 0) + (uint64_t)(R ? R - 1 : 0) * j.b_s1 + round16(2ull * L);
    t.rb.bytes = 0;
    t.rc.bytes = (j.n && j.c1 && j.c2) ? ((uint64_t)(j.c2 - 1) * j.c_s2 + (uint64_t)(j.c1 - 1) * j.c_s1 +
                                          (uint64_t)(L ? L - 1 : 0) * j.b_s2 + 2ull * R + 15) & ~15ull : 0;
    t.rp.bytes = 0;
    return;
  }
  t.ra.bytes = stream_extent(j, j.a_s1, j.a_s2, (uint64_t)kbytes(j.ak()) * j.n);
  if (op == ro::ROPE)
    t.rb.bytes = stream_extent(j, j.b_s1, j.b_s2, 8ull * (j.aux & 0xFFF));
  else
    t.rb.bytes = j.b_en() ? stream_extent(j, j.b_s1, j.b_s2, (uint64_t)kbytes(j.bk()) * j.n) : 0;
  const ro::Kind ck = (op == ro::NORM || op == ro::ROPE) ? ro::I16 : j.ck();
  t.rc.bytes = stream_extent(j, j.c_s1, j.c_s2, (uint64_t)kbytes(ck) * j.n);
  t.rp.bytes = (op == ro::XPOSE) ? 0 : (j.p_bcast() ? 16 : 16ull * j.n);
}

// ---------------------------------------------------------------------------
// Simulation harness
// ---------------------------------------------------------------------------
struct Tb {
  std::unique_ptr<VerilatedContext> ctx;
  std::unique_ptr<VRowOpsKernel>    top;
  Memory        mem;
  Timing        tim;
  std::mt19937  rng{1};
  AxiReadSlave  rd{};
  AxiWriteSlave wr{};
  AxiLiteMaster lite{};
  uint64_t      max_cycles = 50'000'000;
  bool          quiet = false;
  bool          dump = false;             // --dump: a failing case's whole output, got / expected
  bool          in_reset = false;
#if VM_TRACE
  std::unique_ptr<TraceT> tfp;
#endif

  Tb(int argc, char** argv) {
    ctx.reset(new VerilatedContext);
    ctx->commandArgs(argc, argv);
    top.reset(new VRowOpsKernel{ctx.get()});
    auto* T = top.get();
    rd = {"gmem0", &T->m_axi_gmem0_ARVALID, &T->m_axi_gmem0_ARREADY, &T->m_axi_gmem0_ARLEN,
          &T->m_axi_gmem0_ARSIZE, &T->m_axi_gmem0_ARBURST, &T->m_axi_gmem0_ARID,
          &T->m_axi_gmem0_ARADDR, &T->m_axi_gmem0_RVALID, &T->m_axi_gmem0_RREADY,
          &T->m_axi_gmem0_RLAST, &T->m_axi_gmem0_RRESP, &T->m_axi_gmem0_RID,
          &T->m_axi_gmem0_RDATA, &mem, &tim, &rng};
    wr = {"gmem0", &T->m_axi_gmem0_AWVALID, &T->m_axi_gmem0_AWREADY, &T->m_axi_gmem0_AWLEN,
          &T->m_axi_gmem0_AWSIZE, &T->m_axi_gmem0_AWBURST, &T->m_axi_gmem0_AWADDR,
          &T->m_axi_gmem0_WVALID, &T->m_axi_gmem0_WREADY, &T->m_axi_gmem0_WLAST,
          &T->m_axi_gmem0_WSTRB, &T->m_axi_gmem0_WDATA, &T->m_axi_gmem0_BVALID,
          &T->m_axi_gmem0_BREADY, &T->m_axi_gmem0_BRESP, &T->m_axi_gmem0_BID,
          &mem, &tim, &rng};
    // two read IDs of 16 outstanding bursts each (ro_pkg RD_OUTS), 16 write bursts
    rd.outs_limit = 32;  wr.outs_limit = 16;
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
    if (T->m_axi_gmem0_ARVALID || T->m_axi_gmem0_AWVALID || T->m_axi_gmem0_WVALID ||
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
                                    0x0600, 0x05FF, 0x0601, 0x8001, 0x00FF, 0xFF01, 0x4000};
    switch (mode) {
      case 1:  return (uint16_t)rng();
      case 2:  if ((rng() & 3) == 0) return edge[rng() % (sizeof edge / sizeof edge[0])];
               return (uint16_t)rng();
      default: return (uint16_t)(int16_t)std::uniform_int_distribution<int>(-2048, 2048)(rng);
    }
  }
  uint32_t rnd32(int mode) {
    switch (mode) {
      case 1:  return (uint32_t)rng();
      case 2:  if ((rng() & 7) == 0) return (rng() & 1) ? 0x7FFFFFFFu : 0x80000000u;
               return (uint32_t)rng();
      default: return (uint32_t)(int32_t)std::uniform_int_distribution<int>(-(1 << 20), 1 << 20)(rng);
    }
  }

  void fill(uint64_t base, const Region& r, ro::Kind k, int mode) {
    if (!r.img.empty()) {
      for (uint64_t i = 0; i < r.img.size(); i++) mem.wr8(base + i, r.img[i]);
      return;
    }
    if (k == ro::I16) for (uint64_t i = 0; i < r.bytes / 2; i++) mem.wr16(base + 2 * i, rnd16(mode));
    else for (uint64_t i = 0; i < r.bytes / 4; i++) {
      uint32_t v = rnd32(mode);
      for (int b = 0; b < 4; b++) mem.wr8(base + 4 * i + b, (uint8_t)(v >> (8 * b)));
    }
  }

  void write_params(uint64_t base, const std::vector<ro::Param>& ps) {
    for (size_t i = 0; i < ps.size(); i++) {
      uint64_t a = base + 16 * i;
      for (int b = 0; b < 4; b++) mem.wr8(a + b, (uint8_t)((uint32_t)ps[i].g >> (8 * b)));
      for (int b = 0; b < 4; b++) mem.wr8(a + 4 + b, (uint8_t)((uint32_t)ps[i].bq >> (8 * b)));
      mem.wr8(a + 8, ps[i].sh_a); mem.wr8(a + 9, ps[i].sh_b); mem.wr8(a + 10, ps[i].sh_c);
      for (int b = 11; b < 16; b++) mem.wr8(a + b, 0);
    }
  }

  // Runs one case; returns the kernel cycle count (0 on failure).
  uint64_t run(Case& t) {
    ro::Job& j = t.j;
    const uint32_t op = j.code();
    // ---- place data ---------------------------------------------------------------
    fill(j.a, t.ra, op == ro::XPOSE || op == ro::ROPE ? ro::I16 : j.ak(), t.data_mode);
    if (t.rb.bytes) fill(j.b, t.rb, op == ro::ROPE ? ro::I32 : j.bk(), t.data_mode);
    if (t.rp.bytes) {
      if (!t.rp.img.empty()) fill(j.p, t.rp, ro::I16, 0);
      else write_params(j.p, t.params);
    }
    for (uint64_t i = 0; i < t.rc.bytes; i++) mem.wr8(j.c + i, POISON);

    // ---- the oracle on the same image ------------------------------------------------
    ro::Mem rm{[this](uint64_t a) { return mem.rd8(a); }};
    std::vector<ro::Write> exp = ro::run(j, rm);

    // ---- run --------------------------------------------------------------------------
    mem.written.clear();
    lite.write(R_OP, j.op); lite.write(R_N, j.n); lite.write(R_ROWS, j.c1); lite.write(R_BLOCKS, j.c2);
    wr64(R_A, j.a); lite.write(R_AS1, j.a_s1); lite.write(R_AS2, j.a_s2);
    wr64(R_B, j.b); lite.write(R_BS1, j.b_s1); lite.write(R_BS2, j.b_s2);
    wr64(R_C, j.c); lite.write(R_CS1, j.c_s1); lite.write(R_CS2, j.c_s2);
    wr64(R_P, j.p); wr64(R_EPS, j.eps);
    lite.write(R_AUX, j.aux); lite.write(R_BQSH, j.bq_sh); lite.write(R_WLO, j.win_lo); lite.write(R_WHI, j.win_hi);
    lite.write(R_GIE, 1); lite.write(R_IER, 1);
    if (lite.read(R_N) != j.n || lite.read(R_C) != (uint32_t)j.c || lite.read(R_C + 4) != (uint32_t)(j.c >> 32) ||
        lite.read(R_OP) != j.op || lite.read(R_WHI) != j.win_hi || lite.read(R_EPS + 4) != (uint32_t)(j.eps >> 32))
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
    if (!rd.idle() || !wr.idle()) tb_fatal("AXI traffic outstanding after ap_done");

    // ---- check --------------------------------------------------------------------------
    size_t bad = 0, bad_fx = 0;
    auto report = [&](const char* what, uint64_t a, unsigned got, unsigned expv) {
      if (bad + bad_fx < 6)
        std::fprintf(stderr, "  MISMATCH %s %s @%llx (+%lld) got=%02x exp=%02x\n", t.label.c_str(), what,
                     (unsigned long long)a, (long long)(a - j.c), got, expv);
      if (what[0] == 'f') bad_fx++; else bad++;
    };
    for (const auto& w : exp)
      for (size_t i = 0; i < w.bytes.size(); i++) {
        const uint8_t got = mem.rd8(w.addr + i);
        if (got != w.bytes[i]) report("oracle", w.addr + i, got, w.bytes[i]);
        mem.unmark(w.addr + i);
      }
    if (!t.expect_c.empty())
      for (uint64_t i = 0; i < t.expect_c.size(); i++) {
        const uint8_t got = mem.rd8(j.c + i);
        if (got != t.expect_c[i]) report("fixture", j.c + i, got, t.expect_c[i]);
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
    if (bad || bad_fx || stray) {
      std::printf("FAIL  %-44s  %zu wrong vs the oracle, %zu vs the fixture, %zu stray bytes\n", t.label.c_str(),
                  bad, bad_fx, stray);
      if (dump) {                                     // every expected write, as 16-bit lanes
        for (const auto& w : exp) {
          std::printf("  +%-6lld", (long long)(w.addr - j.c));
          for (size_t i = 0; i + 1 < w.bytes.size(); i += 2) {
            unsigned g = mem.rd8(w.addr + i) | (mem.rd8(w.addr + i + 1) << 8);
            unsigned e = w.bytes[i] | (w.bytes[i + 1] << 8);
            std::printf(" %04x%s%04x", g, g == e ? "=" : "!", e);
          }
          std::printf("\n");
        }
      }
      return 0;
    }
    if (!quiet) {
      uint64_t out_bytes = 0;
      for (const auto& w : exp) out_bytes += w.bytes.size();
      std::printf("PASS  %-44s  op=%u n=%u rows=%ux%u  %llu cyc  %.2f B/cyc\n", t.label.c_str(), op, j.n,
                  j.c1, j.c2, (unsigned long long)cyc, (double)out_bytes / (double)std::max<uint64_t>(cyc, 1));
    }
    return std::max<uint64_t>(cyc, 1);
  }
};

// ---------------------------------------------------------------------------
// Case construction
// ---------------------------------------------------------------------------
static uint64_t g_region = 0x0000'0008'0000'0000ull;   // above 4 GiB

static void place(Case& t, std::mt19937& rng) {
  extents(t);
  auto next = [&](uint64_t bytes) {
    uint64_t base = g_region;
    g_region += (bytes + 0x10000) & ~0xFFFull;
    g_region += 0x1000 * (rng() % 4) + 16 * (rng() % 256);
    g_region &= ~15ull;
    return base;
  };
  t.j.a = next(t.ra.bytes + 64);
  t.j.b = next(t.rb.bytes + 64);
  t.j.c = next(t.rc.bytes + 64);
  t.j.p = next(t.rp.bytes + 64);
  // scattered granules may start anywhere (2-byte aligned); a contiguous run is a regular
  // row geometry and needs the 16-byte alignment
  if (t.j.code() == ro::XPOSE && t.j.b_s2 != 2 * (t.j.aux & 15)) t.j.c += 2 * (rng() % 8);
}

static uint32_t opword(uint32_t code, ro::Kind ak, ro::Kind bk, ro::Kind ck, bool ben, bool bcast,
                       bool relu = false, bool ln = false) {
  return code | (ak << 4) | (bk << 6) | (ck << 8) | (ben ? ro::F_B_EN : 0) | (bcast ? ro::F_P_BCAST : 0) |
         (relu ? ro::F_RELU : 0) | (ln ? ro::F_NORM_LN : 0);
}

static int gamma_scale(double gmax) {
  if (gmax == 0.0) return 0;
  return 30 - (int)std::floor(std::log2(gmax));
}

// A random job within the contract (and at its edges: n up to 4096, tails of
// every length, rows of 1, windows, broadcast parameters).
static Case random_case(std::mt19937& rng) {
  auto U = [&](uint32_t lo, uint32_t hi) { return std::uniform_int_distribution<uint32_t>(lo, hi)(rng); };
  auto D = [&](double lo, double hi) { return std::uniform_real_distribution<double>(lo, hi)(rng); };
  Case t;
  ro::Job& j = t.j;
  const uint32_t pick = U(0, 99);
  const uint32_t code = pick < 20 ? ro::ADDX : pick < 30 ? ro::COPY : pick < 38 ? ro::DEQUANT :
                        pick < 58 ? ro::MULACT : pick < 78 ? ro::NORM : pick < 90 ? ro::ROPE : ro::XPOSE;
  // shape
  const uint32_t shape = U(0, 9);
  j.n  = shape <= 1 ? U(1, 24) : shape <= 6 ? U(1, 800) : shape <= 8 ? U(1, 4096) : 8 * U(1, 128);
  j.c1 = shape <= 1 ? U(1, 40) : shape <= 6 ? U(1, 12) : U(1, 4);
  j.c2 = (U(0, 3) == 0) ? U(2, 4) : 1;
  t.data_mode = (int)U(0, 2);
  const bool bcast = U(0, 2) == 0;
  auto stride = [&](uint64_t row_bytes) {           // a row stride that covers the row's words
    return (uint32_t)(round16(row_bytes) + 16 * U(0, 3));
  };
  auto block_stride = [&](uint32_t s1) { return (uint32_t)((uint64_t)s1 * j.c1 + 16 * U(0, 2)); };
  const uint32_t np = bcast ? 1 : j.n;
  t.params.assign(np, ro::Param{});

  if (code == ro::ADDX || code == ro::COPY) {
    const ro::Kind ak = U(0, 1) ? ro::I32 : ro::I16, bk = U(0, 1) ? ro::I32 : ro::I16,
                   ck = U(0, 1) ? ro::I32 : ro::I16;
    const bool ben = code == ro::ADDX && U(0, 5) != 0;
    const bool relu = U(0, 3) == 0;
    j.op = opword(code, ak, bk, ck, ben, bcast, relu);
    j.bq_sh = U(0, 36);
    for (auto& p : t.params) {
      p.sh_a = ak == ro::I32 ? U(0, 20) : U(0, 36);
      p.sh_b = bk == ro::I32 ? U(0, 20) : U(0, 36);
      p.sh_c = U(0, 50);
      p.g = 0;
      p.bq = (U(0, 2) == 0) ? 0 : (int32_t)std::uniform_int_distribution<int>(-(1 << 21), (1 << 21) - 1)(rng);
    }
    if (U(0, 4) == 0) { j.win_lo = U(0, j.c1 / 2 + 1); j.win_hi = U(j.win_lo, j.c1 + 1); }
    j.a_s1 = stride((uint64_t)kbytes(ak) * j.n); j.a_s2 = block_stride(j.a_s1);
    j.b_s1 = stride((uint64_t)kbytes(bk) * j.n); j.b_s2 = block_stride(j.b_s1);
    j.c_s1 = stride((uint64_t)kbytes(ck) * j.n); j.c_s2 = block_stride(j.c_s1);
  } else if (code == ro::DEQUANT) {
    const ro::Kind ak = U(0, 1) ? ro::I32 : ro::I16;
    j.op = opword(code, ak, ro::I16, ro::F32, false, bcast);
    for (auto& p : t.params) p.sh_c = U(0, 31);
    j.a_s1 = stride((uint64_t)kbytes(ak) * j.n); j.a_s2 = block_stride(j.a_s1);
    j.c_s1 = stride(4ull * j.n); j.c_s2 = block_stride(j.c_s1);
  } else if (code == ro::MULACT) {
    j.op = opword(code, ro::I16, ro::I16, ro::I16, true, bcast);
    for (auto& p : t.params) {
      const uint32_t fg = U(8, 14), fu = U(8, 15), fo = U(5, 12);
      p.sh_a = 14 - fg;
      p.sh_c = 20 + fu - fo;
    }
    j.a_s1 = stride(2ull * j.n); j.a_s2 = block_stride(j.a_s1);
    j.b_s1 = stride(2ull * j.n); j.b_s2 = block_stride(j.b_s1);
    j.c_s1 = stride(2ull * j.n); j.c_s2 = block_stride(j.c_s1);
  } else if (code == ro::NORM) {
    const bool ln = U(0, 1);
    const ro::Kind ak = U(0, 1) ? ro::I32 : ro::I16;
    const int f_in = ak == ro::I32 ? 15 : 8;
    j.op = opword(code, ak, ro::I16, ro::I16, false, bcast, false, ln);
    const double eps = ln ? (U(0, 1) ? 1e-6 : 1e-12) : 1e-5;
    const double n = j.n;
    j.eps = (uint64_t)(ln ? std::llround(eps * n * n * std::ldexp(1.0, 2 * f_in))
                          : std::llround(eps * n * std::ldexp(1.0, 2 * f_in)));
    std::vector<double> gamma(np), beta(np);
    double gmax = 0;
    for (uint32_t i = 0; i < np; i++) {
      gamma[i] = D(0.02, 4.0) * (U(0, 1) ? 1 : -1);
      beta[i]  = ln ? D(-1.0, 1.0) : 0.0;
      double gs = ln ? gamma[i] : gamma[i] * std::sqrt(n);
      gmax = std::max(gmax, std::fabs(gs));
    }
    const int g = gamma_scale(gmax);
    j.bq_sh = g + 4;
    for (uint32_t i = 0; i < np; i++) {
      double gs = ln ? gamma[i] : gamma[i] * std::sqrt(n);
      t.params[i].g = (int32_t)std::llround(gs * std::ldexp(1.0, g));
      t.params[i].bq = (int32_t)std::llround(beta[i] * std::ldexp(1.0, 20));
      const int f_out = (int)U(6, 12);
      t.params[i].sh_c = (uint8_t)(24 + g - f_out);
    }
    j.a_s1 = stride((uint64_t)kbytes(ak) * j.n); j.a_s2 = block_stride(j.a_s1);
    j.c_s1 = stride(2ull * j.n); j.c_s2 = block_stride(j.c_s1);
  } else if (code == ro::ROPE) {
    const uint32_t HD = 16 * U(1, 8);                       // a multiple of 16 (the contract)
    j.n = HD * U(1, std::max<uint32_t>(1, std::min<uint32_t>(16, 4096 / HD)));
    t.params.assign(bcast ? 1 : j.n, ro::Param{});
    j.op = opword(code, ro::I16, ro::I32, ro::I16, true, bcast);
    j.aux = HD;
    std::vector<int> fx(j.n);
    for (uint32_t c = 0; c < j.n; c++) fx[c] = bcast ? 12 : (int)U(9, 15);
    for (uint32_t c = 0; c < t.params.size(); c++) {
      const uint32_t d = c % HD, half = HD / 2;
      const int f_rot = fx[(d < half) ? c + half : c - half];
      const int fp = std::max(fx[c], f_rot);
      t.params[c].sh_a = (uint8_t)(fp - fx[c]);
      t.params[c].sh_b = (uint8_t)(fp - f_rot);
      t.params[c].sh_c = (uint8_t)(fp + 23 - (int)U(5, 9));
    }
    j.a_s1 = stride(2ull * j.n); j.a_s2 = block_stride(j.a_s1);
    j.b_s1 = stride(8ull * HD);  j.b_s2 = block_stride(j.b_s1);
    j.c_s1 = stride(2ull * j.n); j.c_s2 = block_stride(j.c_s1);
  } else {                                                      // XPOSE
    const uint32_t R = U(1, 8);
    j.n = shape <= 1 ? U(1, 16) : U(1, 512);
    j.c1 = U(1, 6); j.c2 = U(1, 3);
    j.op = opword(code, ro::I16, ro::I16, ro::I16, false, false);
    j.aux = R;
    j.b_s1 = stride(2ull * j.n);                                // the tile's source rows
    j.a_s1 = (uint32_t)(j.b_s1 * R + 16 * U(0, 2));
    j.a_s2 = block_stride(j.a_s1);
    const bool contiguous = U(0, 1);
    j.b_s2 = contiguous ? 2 * R : 2 * R + 2 * U(0, 6);          // granule stride (bytes)
    j.c_s1 = (uint32_t)round16((uint64_t)j.b_s2 * j.n + 2 * R) + 16 * U(0, 1);
    j.c_s2 = block_stride(j.c_s1);
    t.params.clear();
  }
  char buf[128];
  std::snprintf(buf, sizeof buf, "rand_op%u_%x_n%u_%ux%u_m%d", code, j.op >> 4, j.n, j.c1, j.c2, t.data_mode);
  t.label = buf;
  return t;
}

// ---------------------------------------------------------------------------
// Fixtures: manifest.txt + test_NN_{a,b,p,c}.hex (scripts/gen_fixtures.py): 16-byte
// words as 32 hex digits per line, byte 0 first; c.hex = the whole C region after
// the job (untouched bytes 0xDE).
// ---------------------------------------------------------------------------
static std::vector<uint8_t> read_hex_bytes(const std::string& path) {
  std::ifstream f(path);
  if (!f) tb_fatal("cannot open %s", path.c_str());
  std::vector<uint8_t> v;
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    for (size_t i = 0; i + 1 < line.size(); i += 2)
      v.push_back((uint8_t)std::stoul(line.substr(i, 2), nullptr, 16));
  }
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
    Case t;
    ro::Job& j = t.j;
    int idx; unsigned long long eps;
    std::string label;
    is >> idx >> j.op >> j.n >> j.c1 >> j.c2 >> j.a_s1 >> j.a_s2 >> j.b_s1 >> j.b_s2 >> j.c_s1 >> j.c_s2
       >> eps >> j.aux >> j.bq_sh >> j.win_lo >> j.win_hi >> label;
    if (!is) continue;
    j.eps = eps;
    t.label = "fx" + std::to_string(idx) + "_" + label;
    if (t.label.size() > 44) t.label.resize(44);
    char pre[32];
    std::snprintf(pre, sizeof pre, "/test_%02d_", idx);
    t.ra.img = read_hex_bytes(dir + pre + "a.hex");
    t.rb.img = read_hex_bytes(dir + pre + "b.hex");
    t.rp.img = read_hex_bytes(dir + pre + "p.hex");
    t.expect_c = read_hex_bytes(dir + pre + "c.hex");
    t.fixture = true;
    out.push_back(std::move(t));
  }
  return out;
}

// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
  std::string fixtures, trace, timing = "rand", only;
  std::vector<std::string> one_cases;
  int n_random = 0;
  unsigned seed = 1;
  bool perf = false, quiet = false, dump = false;
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
    else if (a == "--dump") dump = true;
    else if (a == "--only") only = next();           // run only the cases whose label contains this
    else if (a == "--max-cycles") max_cycles = std::stoull(next());
    else if (a == "--trace") trace = next();
    else if (a.rfind("+verilator", 0) == 0) {}
    else tb_fatal("unknown argument %s", a.c_str());
  }

  Tb tb(argc, argv);
  tb.rng.seed(seed);
  tb.max_cycles = max_cycles;
  tb.quiet = quiet;
  tb.dump = dump;
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
  for (const auto& one_case : one_cases) {
    // a random case of the given op code / kinds / flags, with the given shape
    std::istringstream is(one_case);
    uint32_t op, n, c1, c2; int mode = 0;
    is >> op >> n >> c1 >> c2;
    if (!(is >> mode)) mode = 0;
    Case t;
    for (int tries = 0; tries < 10000; tries++) {
      t = random_case(crng);
      if (t.j.code() == (op & 15)) break;
    }
    t.j.op = op; t.j.n = n; t.j.c1 = c1; t.j.c2 = c2; t.data_mode = mode;
    char buf[96];
    std::snprintf(buf, sizeof buf, "case_op%x_n%u_%ux%u_m%d", op, n, c1, c2, mode);
    t.label = buf;
    cases.push_back(t);
  }
  if (perf) {
    // the shipped models' job shapes (one layer each): SmolLM2-135M prefill 256,
    // BERT's LayerNorm, SmolVLM's vision LayerNorm
    struct P { const char* l; uint32_t op; uint32_t n, c1; };
    const P ps[] = {
      {"perf_resadd_256x576_i32", opword(ro::ADDX, ro::I32, ro::I16, ro::I32, true, true), 576, 256},
      {"perf_silu_256x1536",      opword(ro::MULACT, ro::I16, ro::I16, ro::I16, true, false), 1536, 256},
      {"perf_copy_256x576_i16",   opword(ro::COPY, ro::I16, ro::I16, ro::I16, false, true), 576, 256},
      {"perf_dequant_1x49152",    opword(ro::DEQUANT, ro::I16, ro::I16, ro::F32, false, true), 49152 / 12, 12},
      {"perf_rms_256x576_i32",    opword(ro::NORM, ro::I32, ro::I16, ro::I16, false, false), 576, 256},
      {"perf_ln_256x768_i16",     opword(ro::NORM, ro::I16, ro::I16, ro::I16, false, false, false, true), 768, 256},
      {"perf_ln_1024x768_i32",    opword(ro::NORM, ro::I32, ro::I16, ro::I16, false, false, false, true), 768, 1024},
      {"perf_rope_256x576",       opword(ro::ROPE, ro::I16, ro::I32, ro::I16, true, false), 576, 256},
      {"perf_xpose_q_8x64_x96",   opword(ro::XPOSE, ro::I16, ro::I16, ro::I16, false, false), 64, 96},
    };
    for (const P& p : ps) {
      Case t;
      for (int tries = 0; tries < 20000; tries++) {
        t = random_case(crng);
        if (t.j.code() == (p.op & 15)) break;
      }
      t.j.op = p.op; t.j.n = p.n; t.j.c1 = p.c1; t.j.c2 = 1; t.data_mode = 0;
      if ((p.op & 15) == ro::ROPE) t.j.aux = 64;
      if ((p.op & 15) == ro::XPOSE) { t.j.aux = 8; t.j.b_s2 = 2 * 768; t.j.c_s1 = 16; t.j.a_s1 = 8 * 1152; t.j.b_s1 = 1152; }
      t.label = p.l;
      cases.push_back(t);
    }
  }
  for (int i = 0; i < n_random; i++) cases.push_back(random_case(crng));
  if (cases.empty()) tb_fatal("nothing to run (use --fixtures, --random, --case or --perf)");

  // The tables first (OP_TLOAD, as the driver does once after the bitstream): the image
  // the oracle builds; with fixtures, checked against the Python specification's tables.hex.
  {
    Case t;
    t.label = "table_load";
    t.fixture = true;
    t.j.op = opword(ro::TLOAD, ro::I32, ro::I32, ro::I32, false, false);
    t.j.n = 4096; t.j.c1 = 4; t.j.c2 = 1; t.j.a_s1 = 16384; t.j.a_s2 = 65536;
    t.ra.img = ro::table_image();
    if (!fixtures.empty()) {
      std::ifstream f(fixtures + "/tables.hex");
      if (f) {
        std::vector<uint8_t> ref;
        std::string line;
        while (std::getline(f, line))
          for (size_t i = 0; i + 1 < line.size(); i += 2)
            ref.push_back((uint8_t)std::stoul(line.substr(i, 2), nullptr, 16));
        if (ref != t.ra.img) tb_fatal("%s/tables.hex differs from the oracle's table image", fixtures.c_str());
      }
    }
    set_timing(perf ? "fast" : timing);
    place(t, crng);
    tb.mem.clear();
    if (!tb.run(t)) tb_fatal("the table load failed");
  }

  int pass = 0, fail = 0;
  for (auto& t : cases) {
    set_timing(perf ? "fast" : timing);
    if (!only.empty() && t.label.find(only) == std::string::npos) { place(t, crng); continue; }
    // a case reshaped by --case / --perf keeps its random parameters: regenerate
    // them for the new column count where they are per column
    if (!t.fixture && !t.params.empty() && !t.j.p_bcast() && t.params.size() != t.j.n &&
        t.j.code() != ro::XPOSE) {
      std::vector<ro::Param> p2(t.j.n);
      for (uint32_t c = 0; c < t.j.n; c++) p2[c] = t.params[c % t.params.size()];
      t.params = p2;
    }
    if (!t.fixture) {
      // strides for the (possibly reshaped) job
      ro::Job& j = t.j;
      const uint32_t op = j.code();
      if (op != ro::XPOSE) {
        auto rb = [&](ro::Kind k) { return (uint32_t)round16((uint64_t)kbytes(k) * j.n); };
        if (j.a_s1 < rb(j.ak())) j.a_s1 = rb(j.ak());
        if (op == ro::ROPE) { if (j.b_s1 < round16(8ull * (j.aux & 0xFFF))) j.b_s1 = (uint32_t)round16(8ull * (j.aux & 0xFFF)); }
        else if (j.b_s1 < rb(j.bk())) j.b_s1 = rb(j.bk());
        const ro::Kind ck = (op == ro::NORM || op == ro::ROPE) ? ro::I16 : j.ck();
        if (j.c_s1 < rb(ck)) j.c_s1 = rb(ck);
        if (j.a_s2 < (uint64_t)j.a_s1 * j.c1) j.a_s2 = j.a_s1 * j.c1;
        if (j.b_s2 < (uint64_t)j.b_s1 * j.c1) j.b_s2 = j.b_s1 * j.c1;
        if (j.c_s2 < (uint64_t)j.c_s1 * j.c1) j.c_s2 = j.c_s1 * j.c1;
      }
    }
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
