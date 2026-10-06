// ---------------------------------------------------------------------------
// tb_main.cpp — Verilator testbench of the RTL ConvKernel (kernels/conv_rtl).
//
// Drives the IP top level ConvKernel through its AXI4-Lite slave and serves
// gmem0 (x), gmem1 (weight), gmem2 (bias) (reads) and gmem3 (y, write) from a
// sparse byte-addressed DDR model with randomised ready / valid timing and
// read latency.  Checks per job:
//   * with the HLS oracle (CV_HLS_ORACLE: the HLS kernel's own C++,
//     kernels/conv/kernel/ConvKernel.cpp, built with the Vitis HLS headers):
//     every byte of the output region equals what the HLS kernel leaves there,
//     run on the same DDR image — the written elements and the untouched
//     lanes past the tensor alike;
//   * fixtures (kernels/conv TestConvRef --dump-data): every output element
//     equals y.hex;
//   * that nothing outside the output elements was written, the AXI protocol
//     (bursts never cross 4 KiB, WLAST, at most the declared outstanding
//     bursts), ap_done / interrupt behaviour.
// Jobs outside the contract (a window past the line buffer, kh or kw > 7, a
// padded accumulator row past 65 536 entries) and jobs with a zero size must
// write nothing (the HLS kernel's behaviour there is not defined, so no
// oracle).
//
// Usage:
//   Vtb [--fixtures DIR] [--random N] [--seed S] [--timing fast|rand|slow]
//       [--case "N IC H W OC OH OW KH KW SH SW DH DW PT PL BIAS DW"] [--perf]
//       [--case-file F] [--only I,J,...] [--no-oracle] [--quiet] [--max-cycles N]
//       [--trace FILE]
//   --case-file reads one --case geometry per line (17 numbers, then an
//   optional label; '#' comments).  --only runs the listed cases (0-based
//   positions in the list built above).  --no-oracle skips the HLS oracle
//   (cycle counts only: the protocol and stray-write checks stay).
// ---------------------------------------------------------------------------
#include <algorithm>
#include <cstdarg>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <fcntl.h>
#include <unistd.h>

#include "VConvKernel.h"
#include "axi_models.h"
#ifdef CV_HLS_ORACLE
#include "ConvKernel.h"
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

// Register map (xconvkernel_hw.h)
enum : uint8_t {
  R_CTRL = 0x00, R_GIE = 0x04, R_IER = 0x08, R_ISR = 0x0C,
  R_X = 0x10, R_W = 0x1C, R_B = 0x28, R_Y = 0x34, R_BATCH = 0x40, R_IN_CH = 0x48,
  R_IN_H = 0x50, R_IN_W = 0x58, R_OUT_CH = 0x60, R_OUT_H = 0x68, R_OUT_W = 0x70,
  R_KH = 0x78, R_KW = 0x80, R_SH = 0x88, R_SW = 0x90, R_DH = 0x98, R_DW = 0xA0,
  R_PT = 0xA8, R_PL = 0xB0, R_BIAS = 0xB8, R_DWM = 0xC0,
};

// ---------------------------------------------------------------------------
// Test case
// ---------------------------------------------------------------------------
struct Case {
  std::string label;
  uint32_t n = 1, ic = 1, h = 1, w = 1, oc = 1, oh = 1, ow = 1, kh = 1, kw = 1;
  uint32_t sh = 1, sw = 1, dh = 1, dw = 1, pt = 0, pl = 0, bias = 0, dwm = 0;
  uint64_t x = 0, wt = 0, b = 0, y = 0;  // byte addresses
  std::vector<uint16_t> xv, wv, bv, yv;  // fixture data, packed (empty: random)
  int  data_mode = 0;
  bool no_oracle = false;                // zero-size job: must write nothing
};

// the packed layouts (ConvKernel.h)
static uint32_t ic_tiles(uint32_t ic) { return (ic + 15) / 16; }
static uint32_t last_lanes(uint32_t ic) {
  const uint32_t rem = ic - (ic_tiles(ic) - 1) * 16;
  return rem <= 8 ? 8 : 16;
}
static uint64_t per_m(const Case& t) {
  return (uint64_t)t.kh * t.kw * ((ic_tiles(t.ic) - 1) * 16 + last_lanes(t.ic));
}
static uint64_t dw_stride(const Case& t) { return ((uint64_t)t.kh * t.kw + 7) / 8 * 8; }
static uint64_t x_elems(const Case& t) { return (uint64_t)t.n * t.ic * t.h * t.w; }
static uint64_t w_elems(const Case& t) { return t.dwm ? t.oc * dw_stride(t) : t.oc * per_m(t); }
static uint64_t b_elems(const Case& t) { return ((uint64_t)t.oc + 7) / 8 * 8; }
static uint64_t y_elems(const Case& t) { return (uint64_t)t.n * t.oc * t.oh * t.ow; }
static uint64_t words_of(uint64_t e) { return (e + 7) / 8; }

static bool in_contract(const Case& t) {
  const uint64_t mt = (t.oc + 15) / 16;
  return t.ic >= 1 && t.ic <= 1024 && t.oc >= 1 && t.oc <= 1280 && t.ow <= 4096 &&
         t.kh >= 1 && t.kh <= 7 && t.kw >= 1 && t.kw <= 7 && t.sh >= 1 && t.sw >= 1 &&
         t.dh >= 1 && t.dw >= 1 && (t.kh == 1 || t.dh < 16) && (t.kw == 1 || t.dw < 64) &&
         (uint64_t)(t.kh - 1) * t.dh < 16 && (uint64_t)(t.kw - 1) * t.dw < 64 &&
         (uint64_t)t.ow * mt * 16 <= 65536;
}

// ---------------------------------------------------------------------------
// Simulation harness
// ---------------------------------------------------------------------------
struct Tb {
  std::unique_ptr<VerilatedContext> ctx;
  std::unique_ptr<VConvKernel>      top;
  Memory        mem;
  Timing        tim;
  std::mt19937  rng{1};
  AxiReadSlave  rx{}, rw{}, rb{};
  AxiWriteSlave wy{};
  AxiLiteMaster lite{};
  uint64_t      max_cycles = 200'000'000;
  bool          quiet = false;
  bool          skip_oracle = false;
  bool          in_reset = false;
#if VM_TRACE
  std::unique_ptr<TraceT> tfp;
#endif

  Tb(int argc, char** argv) {
    ctx.reset(new VerilatedContext);
    ctx->commandArgs(argc, argv);
    top.reset(new VConvKernel{ctx.get()});
    auto* T = top.get();
#define RD(port, name) AxiReadSlave{name, &T->m_axi_##port##_ARVALID, &T->m_axi_##port##_ARREADY, \
      &T->m_axi_##port##_ARLEN, &T->m_axi_##port##_ARSIZE, &T->m_axi_##port##_ARBURST,          \
      &T->m_axi_##port##_ARID, &T->m_axi_##port##_ARADDR, &T->m_axi_##port##_RVALID,            \
      &T->m_axi_##port##_RREADY, &T->m_axi_##port##_RLAST, &T->m_axi_##port##_RRESP,            \
      &T->m_axi_##port##_RID, &T->m_axi_##port##_RDATA, &mem, &tim, &rng}
    rx = RD(gmem0, "gmem0");
    rw = RD(gmem1, "gmem1");
    rb = RD(gmem2, "gmem2");
#undef RD
    wy = {"gmem3", &T->m_axi_gmem3_AWVALID, &T->m_axi_gmem3_AWREADY, &T->m_axi_gmem3_AWLEN,
          &T->m_axi_gmem3_AWSIZE, &T->m_axi_gmem3_AWBURST, &T->m_axi_gmem3_AWADDR,
          &T->m_axi_gmem3_WVALID, &T->m_axi_gmem3_WREADY, &T->m_axi_gmem3_WLAST,
          &T->m_axi_gmem3_WSTRB, &T->m_axi_gmem3_WDATA, &T->m_axi_gmem3_BVALID,
          &T->m_axi_gmem3_BREADY, &T->m_axi_gmem3_BRESP, &T->m_axi_gmem3_BID,
          &mem, &tim, &rng};
    // the outstanding bursts the IP declares (cv_pkg *_OUTS, package_ip.tcl;
    // fact rtl.axi_masters)
    rx.outs_limit = 16;  rw.outs_limit = 8;  rb.outs_limit = 2;  wy.outs_limit = 8;
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
    T->m_axi_gmem2_AWREADY = 0; T->m_axi_gmem2_WREADY = 0; T->m_axi_gmem2_BVALID = 0;
    T->m_axi_gmem3_ARREADY = 0; T->m_axi_gmem3_RVALID = 0;
  }

  void tick() {
    auto* T = top.get();
    rx.drive(); rw.drive(); rb.drive(); wy.drive();
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
    rx.sample(); rw.sample(); rb.sample(); wy.sample(); lite.sample();
    if (T->m_axi_gmem0_AWVALID || T->m_axi_gmem0_WVALID || T->m_axi_gmem1_AWVALID ||
        T->m_axi_gmem1_WVALID || T->m_axi_gmem2_AWVALID || T->m_axi_gmem2_WVALID ||
        T->m_axi_gmem3_ARVALID)
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
    if (T->m_axi_gmem0_ARVALID || T->m_axi_gmem1_ARVALID || T->m_axi_gmem2_ARVALID ||
        T->m_axi_gmem3_AWVALID || T->m_axi_gmem3_WVALID || T->s_axi_ctrl_BVALID ||
        T->s_axi_ctrl_RVALID || T->__SYM__interrupt)
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
               return (uint16_t)(int16_t)std::uniform_int_distribution<int>(-512, 512)(rng);
      default: return (uint16_t)(int16_t)std::uniform_int_distribution<int>(-256, 256)(rng);
    }
  }

  // logical weights -> the packed layout (TestConvSim pack_conv_weights)
  std::vector<uint16_t> pack(const Case& t) {
    std::vector<uint16_t> out(w_elems(t), 0);
    if (t.dwm) {
      for (uint32_t m = 0; m < t.oc; m++)
        for (uint32_t q = 0; q < t.kh * t.kw; q++) out[m * dw_stride(t) + q] = rnd16(t.data_mode);
    } else {
      const uint32_t nt = ic_tiles(t.ic);
      for (uint32_t m = 0; m < t.oc; m++)
        for (uint32_t c = 0; c < t.ic; c++)
          for (uint32_t khi = 0; khi < t.kh; khi++)
            for (uint32_t kwi = 0; kwi < t.kw; kwi++) {
              const uint32_t ict = c / 16, l = c % 16;
              const uint32_t lanes = (ict + 1 == nt) ? last_lanes(t.ic) : 16;
              const uint64_t idx = m * per_m(t) + (uint64_t)ict * t.kh * t.kw * 16 +
                                   (khi * t.kw + kwi) * lanes + l;
              out[idx] = rnd16(t.data_mode);
            }
    }
    return out;
  }

  // Runs one case; returns the kernel cycle count (0 on failure).
  uint64_t run(Case& t) {
    const uint64_t xw = words_of(x_elems(t)) + 1;            // a run reads up to 7 past the end
    const uint64_t ww = words_of(w_elems(t)) + 1;
    const uint64_t bw = words_of(b_elems(t)) + 1;
    const uint64_t yw = words_of(y_elems(t)) + 1;

    // ---- place data -----------------------------------------------------------------
    if (t.wv.empty() && in_contract(t) && !t.no_oracle) t.wv = pack(t);
    for (uint64_t i = 0; i < 8 * xw; i++)
      mem.wr16(t.x + 2 * i, i < t.xv.size() ? t.xv[i] : (i < x_elems(t) ? rnd16(t.data_mode) : 0));
    for (uint64_t i = 0; i < 8 * ww; i++)
      mem.wr16(t.wt + 2 * i, i < t.wv.size() ? t.wv[i] : 0);
    for (uint64_t i = 0; i < 8 * bw; i++)
      mem.wr16(t.b + 2 * i, i < t.bv.size() ? t.bv[i] : (i < t.oc ? rnd16(t.data_mode) : 0));
    for (uint64_t i = 0; i < 8 * yw; i++) mem.wr16(t.y + 2 * i, 0xDEAD);   // poison

#ifdef CV_HLS_ORACLE
    // ---- the HLS kernel on the same image ------------------------------------------
    std::vector<XWord> ox(xw + 1), oy(yw + 1);
    std::vector<WeightWord> ow_(ww + 1), ob(bw + 1);
    auto load = [&](auto& v, uint64_t base, uint64_t words) {
      for (uint64_t w = 0; w < words; w++)
        for (int l = 0; l < 8; l++) v[w].range(16 * l + 15, 16 * l) = mem.rd16(base + 16 * w + 2 * l);
    };
    load(ox, t.x, xw); load(ow_, t.wt, ww); load(ob, t.b, bw); load(oy, t.y, yw);
    if (!skip_oracle && !t.no_oracle && in_contract(t)) {
      std::fflush(stdout);
      const int saved = dup(1);
      const int devnull = open("/dev/null", O_WRONLY);
      dup2(devnull, 1);
      {
        hls::burst_maxi<XWord> px(ox.data());
        hls::burst_maxi<WeightWord> pw(ow_.data()), pb(ob.data());
        hls::burst_maxi<YWord> py(oy.data());
        ConvKernel(px, pw, pb, py, t.n, t.ic, t.h, t.w, t.oc, t.oh, t.ow, t.kh, t.kw, t.sh, t.sw,
                   t.dh, t.dw, t.pt, t.pl, t.bias, t.dwm);
      }
      std::fflush(stdout);
      dup2(saved, 1);
      close(saved);
      close(devnull);
    }
#endif

    // ---- run ------------------------------------------------------------------------------
    mem.written.clear();
    wr64(R_X, t.x); wr64(R_W, t.wt); wr64(R_B, t.b); wr64(R_Y, t.y);
    lite.write(R_BATCH, t.n); lite.write(R_IN_CH, t.ic); lite.write(R_IN_H, t.h);
    lite.write(R_IN_W, t.w); lite.write(R_OUT_CH, t.oc); lite.write(R_OUT_H, t.oh);
    lite.write(R_OUT_W, t.ow); lite.write(R_KH, t.kh); lite.write(R_KW, t.kw);
    lite.write(R_SH, t.sh); lite.write(R_SW, t.sw); lite.write(R_DH, t.dh); lite.write(R_DW, t.dw);
    lite.write(R_PT, t.pt); lite.write(R_PL, t.pl); lite.write(R_BIAS, t.bias);
    lite.write(R_DWM, t.dwm);
    lite.write(R_GIE, 1); lite.write(R_IER, 1);
    if (lite.read(R_IN_CH) != t.ic || lite.read(R_Y) != (uint32_t)t.y || lite.read(R_DWM) != t.dwm ||
        lite.read(R_W + 4) != (uint32_t)(t.wt >> 32) || lite.read(R_DW) != t.dw)
      tb_fatal("register read-back mismatch");
    if (!(lite.read(R_CTRL) & 4)) tb_fatal("kernel not idle before start");

    char geo[160];
    std::snprintf(geo, sizeof geo, "%u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u",
                  t.n, t.ic, t.h, t.w, t.oc, t.oh, t.ow, t.kh, t.kw, t.sh, t.sw, t.dh, t.dw,
                  t.pt, t.pl, t.bias, t.dwm);
    lite.write(R_CTRL, 1);
    const uint64_t t0 = g_cycle;
    while (!top->__SYM__interrupt) {
      tick();
      if (g_cycle - t0 > max_cycles) {
        // the kernel is stuck mid-job: nothing after this is meaningful
        std::printf("TIMEOUT %s after %llu cycles  (--case \"%s\")\n", t.label.c_str(),
                    (unsigned long long)(g_cycle - t0), geo);
        std::exit(1);
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
    if (!rx.idle() || !rw.idle() || !rb.idle() || !wy.idle())
      tb_fatal("AXI traffic outstanding after ap_done (x %d, w %d, b %d bursts; y %d bursts, %d B due)",
               (int)rx.q.size(), (int)rw.q.size(), (int)rb.q.size(), (int)wy.aw.size(),
               (int)wy.b_due.size());

    // ---- check --------------------------------------------------------------------------
    size_t bad = 0;
    auto report = [&](const char* what, uint64_t e, uint16_t got, uint16_t exp) {
      if (bad < 5)
        std::fprintf(stderr, "  MISMATCH %s %s elem %llu got=%04x exp=%04x\n", t.label.c_str(),
                     what, (unsigned long long)e, got, exp);
      bad++;
    };
    const bool expect_none = t.no_oracle || !in_contract(t);
#ifdef CV_HLS_ORACLE
    if (!expect_none && !skip_oracle)
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
      std::printf("FAIL  %-44s  %zu wrong, %zu stray bytes  (--case \"%s\")\n", t.label.c_str(), bad,
                  stray, geo);
      return 0;
    }
    if (!quiet)
      std::printf("PASS  %-44s  %s  %llu cyc\n", t.label.c_str(), geo, (unsigned long long)cyc);
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
  t.x  = next(words_of(x_elems(t)) + 1);
  t.wt = next(words_of(w_elems(t)) + 1);
  t.b  = next(words_of(b_elems(t)) + 1);
  t.y  = next(words_of(y_elems(t)) + 1);
}

static uint32_t out_dim(uint32_t in, uint32_t pad0, uint32_t pad1, uint32_t k, uint32_t d, uint32_t s) {
  const int64_t v = (int64_t)in + pad0 + pad1 - (int64_t)(k - 1) * d - 1;
  return v < 0 ? 0 : (uint32_t)(v / s + 1);
}

// A random job inside the contract — standard, depthwise, MatMul-on-ConvKernel
// shapes, chunked, m-grouped, ow-tiled — sometimes outside it or of zero size.
static Case random_case(std::mt19937& rng) {
  auto U = [&](uint32_t lo, uint32_t hi) { return std::uniform_int_distribution<uint32_t>(lo, hi)(rng); };
  Case t;
  for (int tries = 0; tries < 10000; tries++) {
    t = Case();
    const uint32_t kind = U(0, 39);
    t.bias = U(0, 1);
    t.n    = (U(0, 5) == 0) ? U(2, 3) : 1;
    const uint32_t shape = U(0, 11);
    if (shape == 0) {                                    // MatMul on ConvKernel: 1 x kw, stride (1, kw)
      t.kh = 1; t.kw = U(1, 4); t.sh = 1; t.sw = t.kw;
      t.ic = 16 * U(1, 6) / t.kw; if (t.ic == 0) t.ic = 1;
      t.oc = U(2, 90);
      t.h  = U(1, 12); t.w = t.kw * U(1, 40);
      t.bias = 0;
    } else if (shape == 1) {                             // the same, wide: up to the channel bounds
      t.kh = 1; t.kw = U(1, 4); t.sh = 1; t.sw = t.kw;
      t.ic = (U(0, 1) ? 16 * U(1, 64) : U(17, 1024)) / t.kw; if (t.ic == 0) t.ic = 1;
      t.oc = (U(0, 3) == 0) ? U(1100, 1280) : U(65, 600);
      t.h  = U(1, 3); t.w = t.kw * U(1, 4);
      t.bias = U(0, 1);
    } else {
      t.dwm = (shape <= 3);
      t.kh  = (U(0, 2) == 0) ? U(1, 7) : (U(0, 1) ? 3 : 1);
      t.kw  = (U(0, 3) == 0) ? U(1, 7) : t.kh;
      t.dh  = (t.kh > 1 && U(0, 4) == 0) ? U(2, 15 / (t.kh - 1)) : 1;
      t.dw  = (t.kw > 1 && U(0, 4) == 0) ? U(2, std::min<uint32_t>(10, 63 / (t.kw - 1))) : 1;
      t.sh  = (U(0, 5) == 0) ? U(3, 6) : U(1, 2);
      t.sw  = (U(0, 5) == 0) ? U(3, 9) : U(1, 2);
      if (t.dwm) { t.ic = t.oc = (U(0, 2) == 0) ? U(17, 40) : U(1, 16); }
      else {
        t.ic = (U(0, 2) == 0) ? U(17, 70) : U(1, 16);
        t.oc = (U(0, 3) == 0) ? U(65, 100) : ((U(0, 1)) ? U(17, 64) : U(1, 16));
      }
      t.h = (U(0, 3) == 0) ? U(1, 6) : U(4, 24);
      t.w = (U(0, 5) == 0) ? U(65, 140) : ((U(0, 3) == 0) ? U(1, 9) : U(4, 40));
      const uint32_t rh = (t.kh - 1) * t.dh, rw = (t.kw - 1) * t.dw;
      t.pt = U(0, rh); t.pl = U(0, rw);
      const uint32_t pb = U(0, rh), pr = U(0, rw);
      t.oh = out_dim(t.h, t.pt, pb, t.kh, t.dh, t.sh);
      t.ow = out_dim(t.w, t.pl, pr, t.kw, t.dw, t.sw);
    }
    if (shape <= 1) { t.oh = t.h; t.ow = t.w / t.kw; }
    if (t.oh == 0 || t.ow == 0) continue;
    if (kind == 0) {                                     // outside the contract
      switch (U(0, 3)) {
        case 0: t.kh = 8; break;
        case 1: t.kw = 9; break;
        case 2: t.kh = 3; t.dh = 8; break;
        default: t.oc = 70; t.ow = 1000; break;
      }
    } else if (kind == 1) {                              // a zero size: no job
      switch (U(0, 3)) { case 0: t.n = 0; break; case 1: t.ic = 0; break;
                         case 2: t.oh = 0; break; default: t.ow = 0; }
      t.no_oracle = true;
    }
    if (!t.no_oracle && in_contract(t) && (uint64_t)t.ow * ((t.oc + 15) / 16) * 16 > 65536) continue;
    // keep the job small: elements and MAC instants
    const uint64_t inst = (uint64_t)t.n * t.oh * ((t.ow + 1) / 2) * t.kh * t.kw *
                          (t.dwm ? (t.oc + 15) / 16 : ((t.oc + 15) / 16) * ((t.ic + 15) / 16));
    if (x_elems(t) + y_elems(t) > 300000 || inst > 400000 ||
        (in_contract(t) && w_elems(t) > 1500000)) continue;
    break;
  }
  t.data_mode = (int)U(0, 2);
  char buf[128];
  std::snprintf(buf, sizeof buf, "rand_%s%ux%ux%ux%u_%u_k%ux%u_s%ux%u", t.dwm ? "dw" : "",
                t.n, t.ic, t.h, t.w, t.oc, t.kh, t.kw, t.sh, t.sw);
  t.label = buf;
  return t;
}

// ---------------------------------------------------------------------------
// Fixtures: manifest.txt + test_NN_{x,w,b,y}.hex (kernels/conv TestConvRef --dump-data)
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
    // idx batch in_ch in_h in_w out_ch out_h out_w kh kw sh sw dh dw pt pl has_bias is_dw label
    uint32_t* f17[] = {&t.n, &t.ic, &t.h, &t.w, &t.oc, &t.oh, &t.ow, &t.kh, &t.kw, &t.sh, &t.sw,
                       &t.dh, &t.dw, &t.pt, &t.pl, &t.bias, &t.dwm};
    for (int i = 0; i < 17; i++) *f17[i] = (uint32_t)std::stoul(tok[1 + i]);
    t.label = "fx" + tok[0] + "_" + tok.back();
    if (t.label.size() > 44) t.label.resize(44);
    char pre[32];
    std::snprintf(pre, sizeof pre, "/test_%02d_", idx);
    t.xv = read_hex(dir + pre + "x.hex");
    t.wv = read_hex(dir + pre + "w.hex");
    t.bv = read_hex(dir + pre + "b.hex");
    t.yv = read_hex(dir + pre + "y.hex");
    out.push_back(std::move(t));
  }
  return out;
}

// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
  std::string fixtures, trace, one_case, case_file, only, timing = "rand";
  bool no_oracle = false;
  int n_random = 0;
  unsigned seed = 1;
  bool perf = false, quiet = false;
  uint64_t max_cycles = 200'000'000;
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
    else if (a == "--only") only = next();
    else if (a == "--case-file") case_file = next();
    else if (a == "--no-oracle") no_oracle = true;
    else if (a == "--max-cycles") max_cycles = std::stoull(next());
    else if (a == "--trace") trace = next();
    else if (a.rfind("+verilator", 0) == 0) {}
    else tb_fatal("unknown argument %s", a.c_str());
  }

  Tb tb(argc, argv);
  tb.rng.seed(seed);
  tb.max_cycles = max_cycles;
  tb.quiet = quiet;
  tb.skip_oracle = no_oracle;
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
#ifndef CV_HLS_ORACLE
  std::printf("note: built without the HLS oracle (no Vitis HLS headers): fixtures only\n");
  if (n_random) tb_fatal("--random needs the HLS oracle");
#endif

  std::vector<Case> cases;
  if (!fixtures.empty())
    for (auto& t : load_fixtures(fixtures)) cases.push_back(std::move(t));
  auto parse_case = [](const std::string& line, const char* label) {
    std::istringstream is(line);
    Case t;
    is >> t.n >> t.ic >> t.h >> t.w >> t.oc >> t.oh >> t.ow >> t.kh >> t.kw >> t.sh >> t.sw >> t.dh
       >> t.dw >> t.pt >> t.pl >> t.bias >> t.dwm;
    if (!is) tb_fatal("bad case geometry: %s", line.c_str());
    std::string l;
    t.label = (is >> l) ? l : label;
    if (t.label.size() > 44) t.label.resize(44);
    return t;
  };
  if (!one_case.empty()) cases.push_back(parse_case(one_case, "case"));
  if (!case_file.empty()) {
    std::ifstream f(case_file);
    if (!f) tb_fatal("cannot open %s", case_file.c_str());
    std::string line;
    while (std::getline(f, line))
      if (line.find_first_not_of(" \t") != std::string::npos && line[line.find_first_not_of(" \t")] != '#')
        cases.push_back(parse_case(line, "case"));
  }
  if (perf) {
    // the ConvKernel benchmarks of run_remote_perf.py (perf_config.json.example)
    auto add = [&](const char* l, uint32_t n, uint32_t ic, uint32_t hw, uint32_t oc, uint32_t k,
                   uint32_t s, uint32_t p, uint32_t dwm) {
      Case t;
      t.label = l; t.n = n; t.ic = ic; t.h = t.w = hw; t.oc = oc; t.kh = t.kw = k;
      t.sh = t.sw = s; t.pt = t.pl = p; t.dwm = dwm;
      t.oh = out_dim(hw, p, p, k, 1, s); t.ow = out_dim(hw, p, p, k, 1, s);
      cases.push_back(t);
    };
    add("perf_3x3-1ch-28x28-32out",     1,   1, 28,  32, 3, 1, 1, 0);
    add("perf_3x3-1ch-28x28-32out-b16", 16,  1, 28,  32, 3, 1, 1, 0);
    add("perf_3x3-64ch-56x56",          1,  64, 56,  64, 3, 1, 1, 0);
    add("perf_3x3-64ch-56x56-s2",       1,  64, 56,  64, 3, 2, 1, 0);
    add("perf_3x3-64ch-28x28",          1,  64, 28,  64, 3, 1, 1, 0);
    add("perf_1x1-64to128-56x56",       1,  64, 56, 128, 1, 1, 0, 0);
    add("perf_1x1-128to256-28x28",      1, 128, 28, 256, 1, 1, 0, 0);
    add("perf_5x5-16ch-28x28",          1,  16, 28,  16, 5, 1, 2, 0);
    add("perf_dw-3x3-32ch-28x28",       1,  32, 28,  32, 3, 1, 1, 1);
    add("perf_dw-3x3-64ch-56x56",       1,  64, 56,  64, 3, 1, 1, 1);
  }
  for (int i = 0; i < n_random; i++) cases.push_back(random_case(crng));
  if (cases.empty()) tb_fatal("nothing to run (use --fixtures, --random, --case or --perf)");

  if (!only.empty()) {
    std::vector<Case> sel;
    std::istringstream is(only);
    std::string tok;
    while (std::getline(is, tok, ','))
      if (!tok.empty()) {
        const size_t k = std::stoul(tok);
        if (k >= cases.size()) tb_fatal("--only %zu: only %zu cases", k, cases.size());
        sel.push_back(cases[k]);
      }
    cases.swap(sel);
  }

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
