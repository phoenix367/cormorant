// Copyright 2026 Ivan Gubochkin
// SPDX-License-Identifier: Apache-2.0

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
// Descriptor chains (ROWOPS_RTL_KERNEL.md §1.6): one start runs a list of jobs
// read from DDR; the oracle runs them in order on the memory (ro::run_chain) and
// every byte the chain writes is checked as above, the descriptor list and the
// inputs being read-only.  Random chains mix every op, chain jobs that read what
// an earlier job wrote (and overwrite what one wrote or read), P_KEEP across
// jobs and chains, TLOAD, descriptor tails of junk, chain_n above 16 (clamped);
// the argument registers hold decoys during a chain and must keep them.  Chain
// fixtures: DIR/chains/manifest.txt + chain_NN_{mem,out}.hex (one segment).
//
// Usage:
//   Vtb [--fixtures DIR] [--random N] [--chains N] [--seed S] [--timing fast|rand|slow]
//       [--case "op n c1 c2 [mode]"] [--perf] [--quiet] [--max-cycles N] [--trace FILE]
// (--case: an op code with the kinds / flags in its upper bits as the register
// takes them; strides, parameters and data are chosen as for a random case.)
// The single jobs (fixtures, --case, --perf, --random) run first, then the chains.
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
  R_CHAIN = 0x70, R_CHN = 0x78,
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
  // FOLD: the source rows' length (random cases) and the offset of a within region A
  // (a reversed stride points a at the last channel; any element alignment)
  uint64_t fold_L = 0, a_off = 0;
  // a job of a chain: A / B read an earlier job's C region (not filled here); C is not a
  // fresh region (it overwrites an earlier job's output or input: not poisoned); P_KEEP
  // wanted where the parameter RAM allows it (decided when the chain runs)
  bool dep_a = false, dep_b = false, own_c = true, want_keep = false;
  int  link_c = 0;                        // C over an earlier job's output (1, WAW) or input (2, WAR)
};

// A descriptor chain: its jobs, the descriptor list, chain_n as written.
struct Chain {
  std::string label;
  std::vector<Case> jobs;
  uint64_t desc = 0;
  uint32_t n_reg = 0;                     // above 16: clamped by the unit (only 16 descriptors exist)
  bool junk_tail = false;                 // descriptor words 24 .. 31 random (ignored by the unit)
  // fixtures: one segment (inputs, parameters, outputs at the manifest's offsets) before / after
  bool fixture = false;
  uint64_t seg = 0;
  std::vector<uint8_t> seg_img, seg_expect;
  bool perf = false;                      // --perf: also run the jobs one by one (cycles compared)
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
  if (op == ro::FOLD) {
    // the source: c2 rows of fold_L int16 |a_s2| apart (a fixture: its image), plus the
    // element offset; the output one contiguous run
    const uint64_t s2 = (uint64_t)std::llabs((int64_t)(int32_t)j.a_s2);
    const uint64_t L = t.fold_L ? t.fold_L : std::min<uint64_t>(j.win_hi, 1u << 20);
    t.ra.bytes = std::max<uint64_t>(round16((uint64_t)(j.c2 ? j.c2 - 1 : 0) * s2 + 2 * L + 16), t.ra.img.size());
    t.rb.bytes = 0; t.rp.bytes = 0;
    t.rc.bytes = round16(2ull * j.c2 * j.c1 * j.n);
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
  // The parameter RAM as the jobs so far left it (the oracle's P_KEEP), and the job that
  // loaded it last (which later jobs may keep: the same op, kinds and flags, n no larger)
  ro::PState    pst;
  struct PLoad { bool valid = false, bcast = false; uint32_t sig = 0, n = 0, bq_sh = 0; } pld;
  bool          chain_dirty = false;      // chain_n != 0 in the unit: a single job writes 0 first
  // what the chains run covered
  struct ChainStats { uint64_t chains = 0, jobs = 0, raw_a = 0, raw_b = 0, waw = 0, war = 0, keep = 0, keep_first = 0,
                      tload = 0, clamped = 0, junk = 0, ops[11] = {}; } cst;
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

  // P_KEEP bookkeeping (ro_core T_CFG2: the stream ops, NORM and ROPE with work load P unless P_KEEP)
  static bool p_go(const ro::Job& j) {
    const uint32_t op = j.code();
    return (op == ro::COPY || op == ro::ADDX || op == ro::DEQUANT || op == ro::MULACT || op == ro::GATE ||
            op == ro::NORM || op == ro::ROPE) && j.n && j.c1 && j.c2 && j.n <= 4096;
  }
  static uint32_t p_sig(const ro::Job& j) { return j.op & ~(uint32_t)(ro::F_P_KEEP | ro::F_P_BCAST); }
  static void p_note(PLoad& s, const ro::Job& j) {
    if (p_go(j) && !(j.op & ro::F_P_KEEP)) s = {true, j.p_bcast(), p_sig(j), j.n, j.bq_sh};
  }
  // A chain's P_KEEP flags: kept where the job wants it and the RAM holds a block of the
  // same op (kinds, flags) and at least its columns — taking the loader's bq_sh with it.
  void resolve_keep(std::vector<Case>& jobs) const {
    PLoad s = pld;
    for (Case& t : jobs) {
      ro::Job& j = t.j;
      j.op &= ~(uint32_t)ro::F_P_KEEP;
      if (t.want_keep && p_go(j) && s.valid && s.sig == p_sig(j) && (s.bcast || j.n <= s.n)) {
        j.op |= ro::F_P_KEEP;
        j.bq_sh = s.bq_sh;
      }
      p_note(s, j);
    }
  }

  // Places a job's inputs (A, B, P) — unless an earlier job of its chain writes them.
  void place_inputs(const Case& t) {
    const ro::Job& j = t.j;
    const uint32_t op = j.code();
    if (!t.dep_a) fill(j.a - t.a_off, t.ra, op == ro::XPOSE || op == ro::ROPE ? ro::I16 : j.ak(), t.data_mode);
    if (t.rb.bytes && !t.dep_b) fill(j.b, t.rb, op == ro::ROPE ? ro::I32 : j.bk(), t.data_mode);
    if (t.rp.bytes) {
      if (!t.rp.img.empty()) fill(j.p, t.rp, ro::I16, 0);
      else write_params(j.p, t.params);
    }
  }

  // ap_start, the wait for the interrupt and the ap_ctrl_hs checks; false on a timeout.
  bool start_and_wait(const std::string& label, uint64_t& cyc) {
    lite.write(R_CTRL, 1);
    const uint64_t t0 = g_cycle;
    while (!top->__SYM__interrupt) {
      tick();
      if (g_cycle - t0 > max_cycles) {
        std::fprintf(stderr, "TIMEOUT %s after %llu cycles\n", label.c_str(), (unsigned long long)(g_cycle - t0));
        return false;
      }
    }
    cyc = g_cycle - t0;
    const uint32_t ctrl = lite.read(R_CTRL);
    if (!(ctrl & 2)) tb_fatal("interrupt without ap_done (ctrl %x)", ctrl);
    if (!(ctrl & 4)) tb_fatal("not idle after done (ctrl %x)", ctrl);
    if (lite.read(R_CTRL) & 2) tb_fatal("ap_done not cleared on read");
    if (!(lite.read(R_ISR) & 1)) tb_fatal("ISR done bit not set");
    lite.write(R_ISR, 1);
    for (int i = 0; i < 4; i++) tick();
    if (top->__SYM__interrupt) tb_fatal("interrupt still asserted after ISR clear");
    if (!rd.idle() || !wr.idle()) tb_fatal("AXI traffic outstanding after ap_done");
    return true;
  }

  // Every byte written that no expected write accounts for (they are unmarked first).
  size_t stray_writes(const std::string& label) {
    size_t stray = 0;
    for (auto& kv : mem.written)
      for (int i = 0; i < 4096; i++)
        if (kv.second[i]) {
          if (stray < 3)
            std::fprintf(stderr, "  STRAY WRITE %s at %llx\n", label.c_str(), (unsigned long long)((kv.first << 12) + i));
          stray++;
        }
    return stray;
  }

  // Runs one case; returns the kernel cycle count (0 on failure).
  uint64_t run(Case& t) {
    ro::Job& j = t.j;
    const uint32_t op = j.code();
    // ---- place data ---------------------------------------------------------------
    place_inputs(t);
    for (uint64_t i = 0; i < t.rc.bytes; i++) mem.wr8(j.c + i, POISON);

    // ---- the oracle on the same image ------------------------------------------------
    ro::Mem rm{[this](uint64_t a) { return mem.rd8(a); }};
    std::vector<ro::Write> exp = ro::run(j, rm, &pst);
    p_note(pld, j);

    // ---- run --------------------------------------------------------------------------
    mem.written.clear();
    if (chain_dirty) { lite.write(R_CHN, 0); chain_dirty = false; }
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

    uint64_t cyc = 0;
    if (!start_and_wait(t.label, cyc)) return 0;

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
    const size_t stray = stray_writes(t.label);

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

  // Runs one chain; returns the kernel cycle count (0 on failure).
  uint64_t run_chain(Chain& ch) {
    const size_t N = ch.jobs.size();
    // ---- place data: every job's inputs, then the fresh C regions poisoned ---------------
    if (ch.fixture) {
      for (size_t i = 0; i < ch.seg_img.size(); i++) mem.wr8(ch.seg + i, ch.seg_img[i]);
    } else {
      resolve_keep(ch.jobs);
      for (const Case& t : ch.jobs) place_inputs(t);
      for (const Case& t : ch.jobs)
        if (t.own_c)
          for (uint64_t i = 0; i < t.rc.bytes; i++) mem.wr8(t.j.c + i, POISON);
    }
    // ---- the descriptors ----------------------------------------------------------------
    std::vector<ro::Job> js;
    for (size_t k = 0; k < N; k++) {
      js.push_back(ch.jobs[k].j);
      std::array<uint32_t, ro::DESC_WORDS> d = ro::descriptor(ch.jobs[k].j);
      if (ch.junk_tail)
        for (uint32_t w = 24; w < ro::DESC_WORDS; w++) d[w] = (uint32_t)rng();
      for (uint32_t w = 0; w < ro::DESC_WORDS; w++)
        for (int b = 0; b < 4; b++) mem.wr8(ch.desc + ro::DESC_BYTES * k + 4 * w + b, (uint8_t)(d[w] >> (8 * b)));
    }

    cst.chains++;
    cst.clamped += ch.n_reg > ro::CHAIN_MAX;
    cst.junk += ch.junk_tail;
    for (size_t k = 0; k < N; k++) {
      const Case& t = ch.jobs[k];
      cst.jobs++;
      cst.ops[t.j.code() % 11]++;
      cst.raw_a += t.dep_a; cst.raw_b += t.dep_b;
      cst.waw += t.link_c == 1; cst.war += t.link_c == 2;
      cst.tload += t.j.code() == ro::TLOAD;
      if (t.j.op & ro::F_P_KEEP) { cst.keep++; cst.keep_first += (k == 0); }
    }

    // ---- the oracle: the jobs in order on the same image ----------------------------------
    ro::Mem rm{[this](uint64_t a) { return mem.rd8(a); }};
    const std::unordered_map<uint64_t, uint8_t> exp = ro::run_chain(js, rm, &pst);
    for (const ro::Job& j : js) p_note(pld, j);

    // ---- run: decoys in the argument registers (a chain neither uses nor changes them) ----
    mem.written.clear();
    uint32_t decoy[24];
    for (int i = 0; i < 24; i++) {
      decoy[i] = (uint32_t)rng();
      lite.write((uint8_t)(R_OP + 4 * i), decoy[i]);
    }
    wr64(R_CHAIN, ch.desc);
    lite.write(R_CHN, ch.n_reg);
    chain_dirty = true;
    lite.write(R_GIE, 1); lite.write(R_IER, 1);
    if (lite.read(R_CHAIN) != (uint32_t)ch.desc || lite.read(R_CHAIN + 4) != (uint32_t)(ch.desc >> 32) ||
        lite.read(R_CHN) != ch.n_reg)
      tb_fatal("chain register read-back mismatch");
    if (!(lite.read(R_CTRL) & 4)) tb_fatal("kernel not idle before start");

    uint64_t cyc = 0;
    if (!start_and_wait(ch.label, cyc)) return 0;

    // ---- check --------------------------------------------------------------------------
    size_t bad = 0, bad_fx = 0, bad_reg = 0;
    auto report = [&](const char* what, uint64_t a, unsigned got, unsigned expv) {
      if (bad + bad_fx < 6)
        std::fprintf(stderr, "  MISMATCH %s %s @%llx got=%02x exp=%02x\n", ch.label.c_str(), what,
                     (unsigned long long)a, got, expv);
      if (what[0] == 'f') bad_fx++; else bad++;
    };
    for (const auto& kv : exp) {
      const uint8_t got = mem.rd8(kv.first);
      if (got != kv.second) report("oracle", kv.first, got, kv.second);
      mem.unmark(kv.first);
    }
    if (ch.fixture)
      for (uint64_t i = 0; i < ch.seg_expect.size(); i++) {
        const uint8_t got = mem.rd8(ch.seg + i);
        if (got != ch.seg_expect[i]) report("fixture", ch.seg + i, got, ch.seg_expect[i]);
      }
    const size_t stray = stray_writes(ch.label);
    for (int i = 0; i < 24; i++)
      if (lite.read((uint8_t)(R_OP + 4 * i)) != decoy[i]) bad_reg++;

    if (bad || bad_fx || stray || bad_reg) {
      std::printf("FAIL  %-44s  %zu wrong vs the oracle, %zu vs the fixture, %zu stray bytes, %zu registers changed\n",
                  ch.label.c_str(), bad, bad_fx, stray, bad_reg);
      if (dump)
        for (size_t k = 0; k < N; k++) {
          const ro::Job& j = js[k];
          std::printf("  job %zu: op %x n %u %ux%u a %llx b %llx c %llx p %llx s %u %u %u %u %u %u aux %x bq %u win %u %u%s%s%s\n",
                      k, j.op, j.n, j.c1, j.c2, (unsigned long long)j.a, (unsigned long long)j.b,
                      (unsigned long long)j.c, (unsigned long long)j.p, j.a_s1, j.a_s2, j.b_s1, j.b_s2, j.c_s1, j.c_s2,
                      j.aux, j.bq_sh, j.win_lo, j.win_hi, ch.jobs[k].dep_a ? " depA" : "",
                      ch.jobs[k].dep_b ? " depB" : "", ch.jobs[k].own_c ? "" : " sharedC");
        }
      return 0;
    }
    if (!quiet) {
      std::string ops;
      for (const ro::Job& j : js) ops += (char)('0' + j.code() % 10);
      std::printf("PASS  %-44s  %zu jobs (%s)  %zu bytes  %llu cyc\n", ch.label.c_str(), N, ops.c_str(), exp.size(),
                  (unsigned long long)cyc);
    }
    return std::max<uint64_t>(cyc, 1);
  }
};

// ---------------------------------------------------------------------------
// Case construction
// ---------------------------------------------------------------------------
static uint64_t g_region = 0x0000'0008'0000'0000ull;   // above 4 GiB

static uint64_t next_region(uint64_t bytes, std::mt19937& rng) {
  uint64_t base = g_region;
  g_region += (bytes + 0x10000) & ~0xFFFull;
  g_region += 0x1000 * (rng() % 4) + 16 * (rng() % 256);
  g_region &= ~15ull;
  return base;
}

static void place(Case& t, std::mt19937& rng) {
  extents(t);
  auto next = [&](uint64_t bytes) { return next_region(bytes, rng); };
  t.j.a = next(t.ra.bytes + 64) + t.a_off;
  t.j.b = next(t.rb.bytes + 64);
  t.j.c = next(t.rc.bytes + 64);
  t.j.p = next(t.rp.bytes + 64);
  // scattered granules may start anywhere (2-byte aligned); a contiguous run is a regular
  // row geometry and needs the 16-byte alignment
  if (t.j.code() == ro::XPOSE && t.j.b_s2 != 2 * (t.j.aux & 15)) t.j.c += 2 * (rng() % 8);
}

static uint32_t opword(uint32_t code, ro::Kind ak, ro::Kind bk, ro::Kind ck, bool ben, bool bcast,
                       bool relu = false, bool ln = false, bool scale = false) {
  return code | (ak << 4) | (bk << 6) | (ck << 8) | (ben ? ro::F_B_EN : 0) | (bcast ? ro::F_P_BCAST : 0) |
         (relu ? ro::F_RELU : 0) | (ln ? ro::F_NORM_LN : 0) | (scale ? ro::F_SCALE : 0);
}

static int gamma_scale(double gmax) {
  if (gmax == 0.0) return 0;
  return 30 - (int)std::floor(std::log2(gmax));
}

// A random job within the contract (and at its edges: n up to 4096, tails of
// every length, rows of 1, windows, broadcast parameters).  `small`: a job of a
// chain — the same mix with smaller shapes (the single jobs cover the large ones).
static Case random_case(std::mt19937& rng, bool small = false) {
  auto U = [&](uint32_t lo, uint32_t hi) { return std::uniform_int_distribution<uint32_t>(lo, hi)(rng); };
  auto D = [&](double lo, double hi) { return std::uniform_real_distribution<double>(lo, hi)(rng); };
  Case t;
  ro::Job& j = t.j;
  const uint32_t pick = U(0, 99);
  const uint32_t code = pick < 17 ? ro::ADDX : pick < 25 ? ro::COPY : pick < 31 ? ro::DEQUANT :
                        pick < 43 ? ro::MULACT : pick < 54 ? ro::GATE : pick < 68 ? ro::NORM :
                        pick < 78 ? ro::ROPE : pick < 88 ? ro::XPOSE : ro::FOLD;
  // shape
  const uint32_t shape = U(0, 9);
  j.n  = shape <= 1 ? U(1, 24) : shape <= 6 ? U(1, small ? 320 : 800) : shape <= 8 ? U(1, small ? 1024 : 4096) :
         8 * U(1, small ? 48 : 128);
  j.c1 = shape <= 1 ? U(1, small ? 12 : 40) : shape <= 6 ? U(1, small ? 6 : 12) : U(1, small ? 2 : 4);
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
    // SCALE (RPU_PLAN §14.3): the sum times G inside the contract, |S| < 2^31 and
    // 0 <= G < 2^25 — int32 data from the small mode, shifts and the bias bounded
    const bool scale = U(0, 2) == 0;
    j.op = opword(code, ak, bk, ck, ben, bcast, relu, false, scale);
    j.bq_sh = scale ? U(0, 8) : U(0, 36);
    if (scale && (ak == ro::I32 || (ben && bk == ro::I32))) t.data_mode = 0;
    for (auto& p : t.params) {
      if (scale) {
        p.sh_a = ak == ro::I32 ? U(0, 8) : U(0, 14);
        p.sh_b = bk == ro::I32 ? U(0, 8) : U(0, 14);
        p.sh_c = U(0, 56);
        static const int32_t gs[] = {0, 1, 3, 174763, (1 << 25) - 1, 1 << 24};
        p.g = U(0, 2) ? (int32_t)gs[U(0, 5)] : (int32_t)U(0, (1u << 25) - 1);
      } else {
        p.sh_a = ak == ro::I32 ? U(0, 20) : U(0, 36);
        p.sh_b = bk == ro::I32 ? U(0, 20) : U(0, 36);
        p.sh_c = U(0, 50);
        p.g = 0;
      }
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
  } else if (code == ro::GATE) {
    // tanh(a << sh_a) * sigma(b << sh_b) (RPU_PLAN §14.3): the shifts up to 16 (a << sh_a
    // within int32, the contract), the output shift around 48 - f_out (saturating below 33)
    j.op = opword(code, ro::I16, ro::I16, ro::I16, true, bcast);
    for (auto& p : t.params) {
      p.sh_a = U(0, 2) ? U(4, 7) : U(0, 16);
      p.sh_b = U(0, 2) ? U(3, 6) : U(0, 16);
      p.sh_c = U(0, 2) ? U(36, 42) : U(26, 52);
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
    j.n = HD * U(1, std::max<uint32_t>(1, std::min<uint32_t>(small ? 4 : 16, 4096 / HD)));
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
  } else if (code == ro::FOLD) {
    // TtsPrep's shapes and their edges: rows of W = n columns w0 apart from s0 (halos,
    // negative starts), windows inside / across / outside the rows, reversed channels,
    // unaligned channel rows, m 1 .. 127, s 0 .. 15
    const uint32_t L = shape <= 1 ? U(1, 40) : shape <= 7 ? U(8, small ? 400 : 1200) :
                       U(small ? 300 : 1000, small ? 1200 : 5000);
    j.n  = shape <= 1 ? U(1, 20) : U(1, std::min<uint32_t>(L + 40, small ? 320 : 2100));
    j.c1 = U(1, shape <= 1 ? (small ? 12 : 30) : (small ? 4 : 8));
    j.c2 = U(1, 4);
    j.b_s1 = U(0, 2) ? std::max<uint32_t>(1, j.n - U(0, std::min<uint32_t>(j.n - 1, 40))) : U(1, 3 * j.n);
    j.b_s2 = (uint32_t)((int32_t)U(0, 80) - 40);
    uint32_t lo = U(0, 3) ? U(0, L / 3) : 0, hi = U(0, 3) ? U(lo, L) : L;
    if (U(0, 9) == 0) std::swap(lo, hi);                        // an inverted window: nothing read
    j.win_lo = lo; j.win_hi = hi;
    static const uint32_t ms[] = {1, 10, 100, 7, 127, 2};
    const uint32_t m = U(0, 3) ? ms[U(0, 5)] : U(1, 127);
    const uint32_t sh = U(0, 2) ? U(0, 6) : U(0, 15);
    j.aux = sh | (m << 4);
    const uint32_t s2 = 2 * (L + (U(0, 3) ? U(0, 9) : 0));
    const bool rev = U(0, 4) == 0;
    j.a_s2 = rev ? (uint32_t)(-(int32_t)s2) : s2;
    j.op = code;
    t.fold_L = L;
    t.a_off = (rev ? (uint64_t)(j.c2 - 1) * s2 : 0) + 2 * U(0, 7);
    t.data_mode = (int)U(1, 2);
    t.params.clear();
  } else {                                                      // XPOSE
    const uint32_t R = U(1, 8);
    j.n = shape <= 1 ? U(1, 16) : U(1, small ? 160 : 512);
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

// The strides of a reshaped job (--case, --perf): every row stride covers its row,
// every block stride its rows.
static void fixup_strides(Case& t) {
  ro::Job& j = t.j;
  const uint32_t op = j.code();
  if (op == ro::XPOSE || op == ro::FOLD || op == ro::TLOAD) return;
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

// The table load (OP_TLOAD): the oracle's image, as the driver runs it once after the bitstream.
static Case tload_case() {
  Case t;
  t.label = "table_load";
  t.fixture = true;
  t.j.op = opword(ro::TLOAD, ro::I32, ro::I32, ro::I32, false, false);
  t.j.n = 4096; t.j.c1 = 4; t.j.c2 = 1; t.j.a_s1 = 16384; t.j.a_s2 = 65536;
  t.ra.img = ro::table_image();
  return t;
}

// ---------------------------------------------------------------------------
// Chains (ROWOPS_RTL_KERNEL.md §1.6)
// ---------------------------------------------------------------------------
static bool loads_p(const ro::Job& j) {
  const uint32_t op = j.code();
  return (op == ro::COPY || op == ro::ADDX || op == ro::DEQUANT || op == ro::MULACT || op == ro::GATE ||
          op == ro::NORM || op == ro::ROPE) && j.n && j.c1 && j.c2 && j.n <= 4096;
}

// A job that may keep the parameter RAM of `src` (the same op, kinds, flags, n and
// constants; other rows): its own parameter block a decoy (G, Bq' inverted, sh_c's low bit
// flipped — valid values, other results), so a block read from p instead shows.
static Case keep_clone(const Case& src, std::mt19937& rng) {
  auto U = [&](uint32_t lo, uint32_t hi) { return std::uniform_int_distribution<uint32_t>(lo, hi)(rng); };
  Case t = src;
  t.want_keep = true;
  t.j.c1 = U(1, 6);
  t.j.c2 = U(0, 3) == 0 ? U(2, 3) : 1;
  t.j.win_lo = 0; t.j.win_hi = 0xFFFFFFFFu;
  if (t.j.c2 > 1) {
    t.j.a_s2 = 0; t.j.b_s2 = 0; t.j.c_s2 = 0;            // fixup_strides: the rows' extent
  }
  fixup_strides(t);
  for (ro::Param& p : t.params) {
    p.g = t.j.scale() ? (1 << 25) - 1 - p.g : ~p.g;     // SCALE: G stays in [0, 2^25)
    p.bq = ~p.bq;
    p.sh_c ^= 1;
  }
  return t;
}

// A random chain: N = 1 .. 16 jobs of the random mix (small shapes), now and then a TLOAD;
// some want P_KEEP — among them clones of an earlier loader of the chain or of the previous
// chain's last one (`prev`), which the parameter RAM then holds; chain_n sometimes above 16
// with 16 jobs; junk descriptor tails.
static Chain random_chain(std::mt19937& rng, int idx, const Case* prev) {
  auto U = [&](uint32_t lo, uint32_t hi) { return std::uniform_int_distribution<uint32_t>(lo, hi)(rng); };
  Chain ch;
  const uint32_t pick = U(0, 9);
  const uint32_t N = pick == 0 ? 1 : pick <= 2 ? ro::CHAIN_MAX : U(2, ro::CHAIN_MAX);
  int last_k = -1;                                       // the chain's last loader so far (else prev)
  for (uint32_t k = 0; k < N; k++) {
    Case t;
    if ((last_k >= 0 || prev) && U(0, 4) == 0) t = keep_clone(last_k >= 0 ? ch.jobs[last_k] : *prev, rng);
    else {
      t = (U(0, 39) == 0) ? tload_case() : random_case(rng, true);
      t.want_keep = U(0, 2) == 0;
    }
    t.fixture = false;
    ch.jobs.push_back(std::move(t));
    if (loads_p(ch.jobs.back().j) && !ch.jobs.back().want_keep) last_k = (int)k;
  }
  ch.n_reg = N;
  if (N == ro::CHAIN_MAX && U(0, 2) == 0) ch.n_reg = U(0, 3) ? ro::CHAIN_MAX + U(1, 100) : 0xFFFFFFFFu;
  ch.junk_tail = U(0, 1);
  std::string ops;
  for (const Case& t : ch.jobs) ops += (char)('0' + t.j.code() % 10);
  ch.label = "chain" + std::to_string(idx) + "_n" + std::to_string(ch.n_reg) + "_" + ops;
  if (ch.label.size() > 44) ch.label.resize(44);
  return ch;
}

// The last job of a random chain that loads the parameter RAM (for the next chain's clones;
// a fixture's jobs carry no parameter values to clone).
static const Case* last_loader(const Chain& ch) {
  if (ch.fixture) return nullptr;
  for (size_t k = ch.jobs.size(); k-- > 0;)
    if (loads_p(ch.jobs[k].j) && !ch.jobs[k].want_keep) return &ch.jobs[k];
  return nullptr;
}

// Places a random chain's jobs and its descriptor list, then links jobs: A or B reads an
// earlier job's output (RAW), C overwrites an earlier job's output (WAW) or input (WAR) —
// each only where the region is large enough, so no job reads or writes beyond it (no
// job's inputs overlap its own output).
static void place_chain(Chain& ch, std::mt19937& rng, bool link = true) {
  auto U = [&](uint32_t lo, uint32_t hi) { return std::uniform_int_distribution<uint32_t>(lo, hi)(rng); };
  for (Case& t : ch.jobs) place(t, rng);
  const size_t N = ch.jobs.size();
  std::vector<int> src_a(N, -1), src_b(N, -1);
  for (size_t k = 1; link && k < N; k++) {
    Case& t = ch.jobs[k];
    const uint32_t op = t.j.code();
    if (op == ro::TLOAD) continue;
    auto pick = [&](auto ok) {                         // a random earlier job that fits, or -1
      std::vector<int> c;
      for (size_t i = 0; i < k; i++)
        if (ch.jobs[i].j.code() != ro::TLOAD && ok(ch.jobs[i], (int)i)) c.push_back((int)i);
      return c.empty() ? -1 : c[U(0, (uint32_t)c.size() - 1)];
    };
    // a SCALE job's int32 input keeps its own (small) data: |S| < 2^31 is the contract
    const bool a_free = !(t.j.scale() && t.j.ak() == ro::I32), b_free = !(t.j.scale() && t.j.bk() == ro::I32);
    if (U(0, 9) < 4 && a_free) {                       // A <- an earlier job's C
      const int i = pick([&](const Case& s, int) { return s.own_c && s.rc.bytes >= t.ra.bytes; });
      if (i >= 0) { t.j.a = (ch.jobs[i].j.c & ~15ull) + t.a_off; t.dep_a = true; src_a[k] = i; }
    }
    if (t.rb.bytes && U(0, 9) < 3 && b_free) {         // B <- an earlier job's C
      const int i = pick([&](const Case& s, int) { return s.own_c && s.rc.bytes >= t.rb.bytes; });
      if (i >= 0) { t.j.b = ch.jobs[i].j.c & ~15ull; t.dep_b = true; src_b[k] = i; }
    }
    const uint32_t r = U(0, 15);
    if (r == 0) {                                      // WAW: C over an earlier job's output
      const int i = pick([&](const Case& s, int ii) {
        return s.own_c && ii != src_a[k] && ii != src_b[k] && s.rc.bytes >= t.rc.bytes + 16;
      });
      if (i >= 0) { t.j.c = (ch.jobs[i].j.c & ~15ull) + (t.j.c & 15); t.own_c = false; t.link_c = 1; }
    } else if (r == 1) {                               // WAR: C over an earlier job's input A
      const int i = pick([&](const Case& s, int) { return !s.dep_a && s.ra.bytes >= t.rc.bytes + 16; });
      if (i >= 0) { t.j.c = (ch.jobs[i].j.a - ch.jobs[i].a_off) + (t.j.c & 15); t.own_c = false; t.link_c = 2; }
    }
  }
  // the list: 16-byte aligned, now and then across a 4 KiB boundary (two bursts)
  const uint64_t bytes = (uint64_t)ro::DESC_BYTES * N;
  uint64_t base = next_region(bytes + 0x1000, rng) & ~0xFFFull;
  ch.desc = U(0, 2) == 0 ? base + 0x1000 - 16 * U(1, (uint32_t)std::min<uint64_t>(bytes / 16, 255)) : base + 16 * U(0, 64);
}

// The perf chains: a decode-step attention prep of SmolLM2-135M (H 9, KV 3, HD 64, G 3;
// HOSTOPS_PLAN §10, RPU_PLAN §13.4) as ten jobs — RoPE of q and k, the k row into the K
// cache (the groups as blocks), v re-rounded, the q image per group (TRANSPOSE R = G,
// contiguous granules) and the V cache's granules per group (R = 1, byte strobes); and
// LlmDequant's logits row of 49 152 as 12 DEQUANT jobs of 4 096 (RPU_PLAN §13.1).
static Chain perf_chain_dequant(std::mt19937& rng) {
  Chain ch;
  ch.label = "perf_chain_dequant_x12";
  ch.perf = true;
  for (int k = 0; k < 12; k++) {
    Case t;
    for (int tries = 0; tries < 20000; tries++) {
      t = random_case(rng, true);
      if (t.j.code() == ro::DEQUANT) break;
    }
    t.j.op = opword(ro::DEQUANT, ro::I16, ro::I16, ro::F32, false, true);
    t.j.n = 4096; t.j.c1 = 1; t.j.c2 = 1; t.data_mode = 0; t.want_keep = false; t.fixture = false;
    fixup_strides(t);
    ch.jobs.push_back(t);
  }
  ch.n_reg = 12;
  return ch;
}

static Chain perf_chain(std::mt19937& rng) {
  auto like = [&](uint32_t op) {
    Case t;
    for (int tries = 0; tries < 20000; tries++) {
      t = random_case(rng, true);
      if (t.j.code() == (op & 15)) break;
    }
    t.j.op = op; t.j.c2 = 1; t.data_mode = 0; t.want_keep = false; t.fixture = false;
    return t;
  };
  const uint32_t H = 9, KV = 3, HD = 64, G = 3;
  Chain ch;
  ch.label = "perf_chain_decode_prep_x10";
  ch.perf = true;
  for (uint32_t heads : {H, KV}) {                    // RoPE q, k: one row of heads * HD
    Case t = like(opword(ro::ROPE, ro::I16, ro::I32, ro::I16, true, false));
    t.j.n = heads * HD; t.j.c1 = 1; t.j.aux = HD;
    std::vector<ro::Param> p2(t.j.n);
    for (uint32_t c = 0; c < t.j.n; c++) p2[c] = t.params[c % t.params.size()];
    t.params = p2;
    ch.jobs.push_back(t);
  }
  {                                                   // k into the K cache: HD per group, the groups as blocks
    Case t = like(opword(ro::COPY, ro::I16, ro::I16, ro::I16, false, true));
    t.j.n = HD; t.j.c1 = 1; t.j.c2 = KV; t.j.win_lo = 0; t.j.win_hi = 0xFFFFFFFFu;
    t.j.a_s1 = 2 * HD; t.j.a_s2 = 2 * HD; t.j.c_s1 = 2 * HD; t.j.c_s2 = 2 * HD * 256;
    t.params.assign(1, ro::Param{}); t.j.bq_sh = 0;
    ch.jobs.push_back(t);
  }
  {                                                   // v re-rounded into its staging row
    Case t = like(opword(ro::COPY, ro::I16, ro::I16, ro::I16, false, true));
    t.j.n = KV * HD; t.j.c1 = 1; t.j.win_lo = 0; t.j.win_hi = 0xFFFFFFFFu;
    ch.jobs.push_back(t);
  }
  for (uint32_t g = 0; g < KV; g++) {                 // the q image of group g: R = G heads, contiguous
    Case t = like(opword(ro::XPOSE, ro::I16, ro::I16, ro::I16, false, false));
    t.j.n = HD; t.j.c1 = 1; t.j.c2 = 1; t.j.aux = G;
    t.j.b_s1 = 2 * HD; t.j.a_s1 = 2 * HD * G; t.j.a_s2 = t.j.a_s1; t.j.b_s2 = 2 * G; t.j.c_s1 = 16 * 64; t.j.c_s2 = t.j.c_s1;
    ch.jobs.push_back(t);
  }
  for (uint32_t g = 0; g < KV; g++) {                 // the V granules of group g: R = 1, strobed
    Case t = like(opword(ro::XPOSE, ro::I16, ro::I16, ro::I16, false, false));
    t.j.n = HD; t.j.c1 = 1; t.j.c2 = 1; t.j.aux = 1;
    t.j.b_s1 = 2 * HD; t.j.a_s1 = 2 * HD; t.j.a_s2 = t.j.a_s1; t.j.b_s2 = 2 * 16; t.j.c_s1 = 16 * 2 * HD + 16;
    t.j.c_s2 = t.j.c_s1;
    ch.jobs.push_back(t);
  }
  for (Case& t : ch.jobs) fixup_strides(t);
  ch.n_reg = (uint32_t)ch.jobs.size();
  return ch;
}

// Chain fixtures (scripts/gen_fixtures.py, from the Python specification):
// DIR/chains/manifest.txt — one line per job, `chain job op n c1 c2 a a_s1 a_s2 b b_s1 b_s2
// c c_s1 c_s2 p eps aux bq_sh win_lo win_hi label` with a, b, c, p byte offsets into the
// chain's segment — and chain_NN_mem.hex / chain_NN_out.hex, the segment before / after.
static std::vector<uint8_t> read_hex_bytes(const std::string& path);
static std::vector<Chain> load_chain_fixtures(const std::string& dir) {
  std::vector<Chain> out;
  std::ifstream f(dir + "/chains/manifest.txt");
  if (!f) return out;
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream is(line);
    int ci, ji;
    unsigned long long a, b, c, p, eps;
    Case t;
    ro::Job& j = t.j;
    std::string label;
    is >> ci >> ji >> j.op >> j.n >> j.c1 >> j.c2 >> a >> j.a_s1 >> j.a_s2 >> b >> j.b_s1 >> j.b_s2 >> c >> j.c_s1 >>
        j.c_s2 >> p >> eps >> j.aux >> j.bq_sh >> j.win_lo >> j.win_hi >> label;
    if (!is) continue;
    if (ci < 0 || ji < 0) tb_fatal("chain fixture: bad indices in '%s'", line.c_str());
    if ((size_t)ci >= out.size()) out.resize(ci + 1);
    Chain& ch = out[ci];
    if ((size_t)ji != ch.jobs.size()) tb_fatal("chain fixture %d: job %d out of order", ci, ji);
    j.a = a; j.b = b; j.c = c; j.p = p; j.eps = eps;   // offsets until the segment is placed
    t.label = label;
    t.fixture = true;
    ch.jobs.push_back(t);
  }
  for (size_t ci = 0; ci < out.size(); ci++) {
    Chain& ch = out[ci];
    char pre[48];
    std::snprintf(pre, sizeof pre, "/chains/chain_%02zu_", ci);
    ch.seg_img = read_hex_bytes(dir + pre + "mem.hex");
    ch.seg_expect = read_hex_bytes(dir + pre + "out.hex");
    if (ch.seg_img.size() != ch.seg_expect.size()) tb_fatal("chain fixture %zu: mem / out sizes differ", ci);
    ch.fixture = true;
    ch.n_reg = (uint32_t)ch.jobs.size();
    ch.label = "fxchain" + std::to_string(ci) + "_" + (ch.jobs.empty() ? std::string() : ch.jobs[0].label);
    if (ch.label.size() > 44) ch.label.resize(44);
  }
  return out;
}

static void place_chain_fixture(Chain& ch, std::mt19937& rng) {
  ch.seg = next_region(ch.seg_img.size() + 64, rng);
  for (Case& t : ch.jobs) {
    t.j.a += ch.seg; t.j.b += ch.seg; t.j.c += ch.seg; t.j.p += ch.seg;
  }
  const uint64_t bytes = (uint64_t)ro::DESC_BYTES * ch.jobs.size();
  ch.desc = (next_region(bytes + 0x1000, rng) & ~0xFFFull) + 0x1000 - 16 * 3;   // across 4 KiB
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
  int n_random = 0, n_chains = 0;
  unsigned seed = 1;
  bool perf = false, quiet = false, dump = false;
  uint64_t max_cycles = 50'000'000;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto next = [&] { if (i + 1 >= argc) tb_fatal("missing value for %s", a.c_str()); return std::string(argv[++i]); };
    if (a == "--fixtures") fixtures = next();
    else if (a == "--random") n_random = std::stoi(next());
    else if (a == "--chains") n_chains = std::stoi(next());
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
  std::mt19937 chrng(seed * 104729u + 7);           // the chains: their own stream (the single jobs' unchanged)
  auto set_timing = [&](const std::string& mode, std::mt19937& r) {
    if (mode == "fast") tb.tim = Timing{1, 1, 1, 1, 1, 24, 24};
    else if (mode == "slow") tb.tim = Timing{0.3, 0.4, 0.3, 0.4, 0.5, 40, 200};
    else {
      auto P = [&] { return std::uniform_real_distribution<double>(0.25, 1.0)(r); };
      int lmin = (int)(r() % 60) + 2;
      tb.tim = Timing{P(), P(), P(), P(), P(), lmin, lmin + (int)(r() % 120)};
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
      // Piper's chunk (RPU_PLAN §14): one TtsGate, the last stage average's (T + y2) * G
      {"perf_gate_12x4096",       opword(ro::GATE, ro::I16, ro::I16, ro::I16, true, true), 4096, 12},
      {"perf_avg3_384x4096",      opword(ro::ADDX, ro::I32, ro::I16, ro::I16, true, true, false, false, true),
                                  4096, 384},
    };
    // Piper's TtsPrep calls as FOLD jobs: the last upsampling stage, the one before, a flow coupling
    struct F { const char* l; uint32_t L, nch, rows, w0, hl, hr, m, s; };
    const F fs[] = {
      {"perf_fold_32x24x2050",  49152, 32, 24, 2048, 1, 1, 10, 0},
      {"perf_fold_64x12x1026",  12288, 64, 12, 1024, 1, 1, 10, 1},
      {"perf_fold_192x2x132",     256, 192, 2, 128, 2, 2, 1, 4},
    };
    for (const F& f : fs) {
      Case t;
      ro::Job& j = t.j;
      j.op = ro::FOLD; j.n = f.w0 + f.hl + f.hr; j.c1 = f.rows; j.c2 = f.nch;
      j.a_s2 = 2 * f.L; j.b_s1 = f.w0; j.b_s2 = (uint32_t)(-(int32_t)f.hl);
      j.win_lo = 0; j.win_hi = f.L; j.aux = f.s | (f.m << 4);
      t.fold_L = f.L; t.data_mode = 1; t.label = f.l;
      cases.push_back(t);
    }
    for (const P& p : ps) {
      Case t;
      for (int tries = 0; tries < 20000; tries++) {
        t = random_case(crng);
        if (t.j.code() == (p.op & 15)) break;
      }
      t.j.op = p.op; t.j.n = p.n; t.j.c1 = p.c1; t.j.c2 = 1; t.data_mode = 0;
      if ((p.op & 15) == ro::ROPE) t.j.aux = 64;
      if ((p.op & 15) == ro::XPOSE) { t.j.aux = 8; t.j.b_s2 = 2 * 768; t.j.c_s1 = 16; t.j.a_s1 = 8 * 1152; t.j.b_s1 = 1152; }
      if (p.op & ro::F_SCALE) {                     // the stage average: G = round(2^19 / 3), >> 18
        t.params.assign(1, ro::Param{});
        t.params[0].g = 174763; t.params[0].sh_c = 18;
        t.j.bq_sh = 0; t.j.win_lo = 0; t.j.win_hi = 0xFFFFFFFFu;
      }
      t.label = p.l;
      cases.push_back(t);
    }
  }
  for (int i = 0; i < n_random; i++) cases.push_back(random_case(crng));

  std::vector<Chain> chains;
  if (!fixtures.empty())
    for (auto& c : load_chain_fixtures(fixtures)) chains.push_back(std::move(c));
  if (perf) {
    chains.push_back(perf_chain(chrng));
    chains.push_back(perf_chain_dequant(chrng));
  }
  for (int i = 0; i < n_chains; i++) {
    const Case* prev = chains.empty() ? nullptr : last_loader(chains.back());
    Chain ch = random_chain(chrng, i, prev);
    chains.push_back(std::move(ch));
  }
  if (cases.empty() && chains.empty())
    tb_fatal("nothing to run (use --fixtures, --random, --chains, --case or --perf)");

  // The tables first (OP_TLOAD, as the driver does once after the bitstream): the image
  // the oracle builds; with fixtures, checked against the Python specification's tables.hex.
  {
    Case t = tload_case();
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
    set_timing(perf ? "fast" : timing, crng);
    place(t, crng);
    tb.mem.clear();
    if (!tb.run(t)) tb_fatal("the table load failed");
  }

  int pass = 0, fail = 0;
  for (auto& t : cases) {
    set_timing(perf ? "fast" : timing, crng);
    if (!only.empty() && t.label.find(only) == std::string::npos) { place(t, crng); continue; }
    // a case reshaped by --case / --perf keeps its random parameters: regenerate
    // them for the new column count where they are per column
    if (!t.fixture && !t.params.empty() && !t.j.p_bcast() && t.params.size() != t.j.n &&
        t.j.code() != ro::XPOSE) {
      std::vector<ro::Param> p2(t.j.n);
      for (uint32_t c = 0; c < t.j.n; c++) p2[c] = t.params[c % t.params.size()];
      t.params = p2;
    }
    if (!t.fixture) fixup_strides(t);              // strides for the (possibly reshaped) job
    place(t, crng);
    tb.mem.clear();
    if (tb.run(t)) pass++;
    else fail++;
  }

  // The chains: fixtures, the perf chain (its jobs one by one first), random chains.
  for (auto& ch : chains) {
    set_timing(perf ? "fast" : timing, chrng);
    if (ch.fixture) place_chain_fixture(ch, chrng);
    else place_chain(ch, chrng, !ch.perf);
    if (!only.empty() && ch.label.find(only) == std::string::npos) continue;
    if (ch.perf) {
      uint64_t sum = 0;
      bool ok = true;
      for (size_t k = 0; k < ch.jobs.size(); k++) {
        Case t = ch.jobs[k];
        t.label = ch.label + "_job" + std::to_string(k);
        place(t, chrng);
        tb.mem.clear();
        const uint64_t c = tb.run(t);
        ok = ok && c;
        sum += c;
      }
      tb.mem.clear();
      const uint64_t c = tb.run_chain(ch);
      if (ok && c) {
        pass++;
        std::printf("CHAIN %-44s  %llu cyc as one chain, %llu cyc as %zu jobs one by one (%+lld: the fetch, "
                    "the job starts)\n", ch.label.c_str(), (unsigned long long)c, (unsigned long long)sum,
                    ch.jobs.size(), (long long)c - (long long)sum);
      } else {
        fail++;
      }
      continue;
    }
    tb.mem.clear();
    if (tb.run_chain(ch)) pass++;
    else fail++;
  }
  if (tb.cst.chains) {
    const auto& c = tb.cst;
    std::printf("\nchains: %llu (%llu jobs: COPY %llu ADDX %llu DEQUANT %llu MULACT %llu NORM %llu ROPE %llu "
                "TRANSPOSE %llu TLOAD %llu FOLD %llu GATE %llu); A from an earlier job %llu, B %llu, C over an earlier "
                "output %llu / input %llu; P_KEEP %llu (%llu first in their chain); chain_n > 16: %llu; "
                "junk tails %llu\n",
                (unsigned long long)c.chains, (unsigned long long)c.jobs, (unsigned long long)c.ops[1],
                (unsigned long long)c.ops[2], (unsigned long long)c.ops[3], (unsigned long long)c.ops[4],
                (unsigned long long)c.ops[5], (unsigned long long)c.ops[6], (unsigned long long)c.ops[7],
                (unsigned long long)c.ops[8], (unsigned long long)c.ops[9], (unsigned long long)c.ops[10],
                (unsigned long long)c.raw_a,
                (unsigned long long)c.raw_b, (unsigned long long)c.waw, (unsigned long long)c.war,
                (unsigned long long)c.keep, (unsigned long long)c.keep_first, (unsigned long long)c.clamped,
                (unsigned long long)c.junk);
  }
  std::printf("\n%d passed, %d failed  (%llu cycles simulated)\n", pass, fail,
              (unsigned long long)g_cycle);
#if VM_TRACE
  if (tb.tfp) tb.tfp->close();
#endif
  tb.top->final();
  return fail ? 1 : 0;
}
