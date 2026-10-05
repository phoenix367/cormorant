// ---------------------------------------------------------------------------
// axi_models.h — cycle-level AXI4 slave models (sparse DDR, two read ports,
// one write port) and an AXI4-Lite master for the PoolingKernel testbench
// (the RTL MatmulKernel's models, kernels/matmul_rtl/tb/verilator).
//
// Every model has drive() (set the DUT inputs for this cycle, from state) and
// sample() (observe the handshakes that happen at the coming rising edge).
// All slaves randomise their READY / VALID timing and read latency, and check
// the DUT's side of the protocol (stable VALID payloads, INCR bursts of
// 16-byte beats, no 4 KiB crossings, correct WLAST).
// ---------------------------------------------------------------------------
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <functional>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "verilated.h"

extern uint64_t g_cycle;
[[noreturn]] void tb_fatal(const char* fmt, ...);

// ---------------------------------------------------------------------------
// Sparse byte-addressed memory.  Bytes never written read back as a
// deterministic address hash, so over-read padding is never accidentally 0.
// ---------------------------------------------------------------------------
struct Memory {
  std::unordered_map<uint64_t, std::vector<uint8_t>> pages;
  std::unordered_map<uint64_t, std::vector<uint8_t>> written;   // write masks

  static uint8_t junk(uint64_t a) {
    uint64_t x = a * 0x9E3779B97F4A7C15ull;
    x ^= x >> 29;
    return (uint8_t)(x >> 17);
  }
  std::vector<uint8_t>& page(uint64_t a) {
    auto it = pages.find(a >> 12);
    if (it != pages.end()) return it->second;
    std::vector<uint8_t> p(4096);
    for (int i = 0; i < 4096; i++) p[i] = junk((a & ~0xFFFull) + i);
    return pages.emplace(a >> 12, std::move(p)).first->second;
  }
  uint8_t rd8(uint64_t a) { return page(a)[a & 0xFFF]; }
  void wr8(uint64_t a, uint8_t v) { page(a)[a & 0xFFF] = v; }
  uint16_t rd16(uint64_t a) { return (uint16_t)(rd8(a) | (rd8(a + 1) << 8)); }
  void wr16(uint64_t a, uint16_t v) { wr8(a, v & 0xFF); wr8(a + 1, v >> 8); }
  void mark(uint64_t a) {
    auto& m = written[a >> 12];
    if (m.empty()) m.assign(4096, 0);
    m[a & 0xFFF] = 1;
  }
  void unmark(uint64_t a) {
    auto it = written.find(a >> 12);
    if (it != written.end()) it->second[a & 0xFFF] = 0;
  }
  bool was_written(uint64_t a) const {
    auto it = written.find(a >> 12);
    return it != written.end() && it->second[a & 0xFFF];
  }
  void clear() { pages.clear(); written.clear(); }
};

struct Timing {
  double p_arready = 1.0, p_rvalid = 1.0, p_awready = 1.0, p_wready = 1.0, p_bvalid = 1.0;
  int    lat_min = 20, lat_max = 20;   // AR accept -> first R beat
};

// ---------------------------------------------------------------------------
// Read slave for one 128-bit port.
// ---------------------------------------------------------------------------
struct AxiReadSlave {
  const char* name;
  // DUT signals
  CData *arvalid, *arready, *arlen, *arsize, *arburst, *arid;
  QData *araddr;
  CData *rvalid, *rready, *rlast, *rresp, *rid;
  VlWide<4>* rdata;

  Memory*      mem;
  Timing*      t;
  std::mt19937* rng;

  struct Req { uint64_t addr; int beats; uint64_t ready_at; };
  std::deque<Req> q;
  int      beat = 0;
  bool     rv = false;          // RVALID currently asserted
  bool     ar_pend = false;     // ARVALID seen without ARREADY
  uint64_t ar_addr_q = 0;
  int      ar_len_q = 0;
  uint64_t beats_total = 0, bursts_total = 0;
  int      max_outstanding = 0;
  int      outs_limit = 0;      // > 0: the master's declared NUM_READ_OUTSTANDING

  bool coin(double p) { return std::uniform_real_distribution<double>(0, 1)(*rng) < p; }

  void drive() {
    *arready = coin(t->p_arready);
    if (!rv && !q.empty() && q.front().ready_at <= g_cycle && coin(t->p_rvalid)) rv = true;
    *rvalid = rv;
    *rresp  = 0;
    *rid    = 0;
    if (rv) {
      const Req& r = q.front();
      uint64_t a = r.addr + 16ull * beat;
      for (int w = 0; w < 4; w++) {
        uint32_t v = 0;
        for (int b = 0; b < 4; b++) v |= (uint32_t)mem->rd8(a + 4 * w + b) << (8 * b);
        (*rdata)[w] = v;
      }
      *rlast = (beat == r.beats - 1);
    } else {
      *rlast = 0;
    }
  }

  void sample() {
    // DUT master checks: ARVALID must hold with a stable payload.
    if (ar_pend) {
      if (!*arvalid) tb_fatal("%s: ARVALID dropped before ARREADY", name);
      if (*araddr != ar_addr_q || *arlen != ar_len_q)
        tb_fatal("%s: AR payload changed while waiting", name);
    }
    if (*arvalid && *arready) {
      uint64_t a = *araddr;
      int beats = *arlen + 1;
      if (*arsize != 4)  tb_fatal("%s: ARSIZE %d", name, *arsize);
      if (*arburst != 1) tb_fatal("%s: ARBURST %d", name, *arburst);
      if (a & 15)        tb_fatal("%s: unaligned ARADDR %llx", name, (unsigned long long)a);
      if ((a >> 12) != ((a + 16ull * beats - 1) >> 12))
        tb_fatal("%s: burst crosses 4 KiB: %llx + %d", name, (unsigned long long)a, beats);
      int lat = std::uniform_int_distribution<int>(t->lat_min, t->lat_max)(*rng);
      q.push_back({a, beats, g_cycle + (uint64_t)lat});
      bursts_total++;
      if ((int)q.size() > max_outstanding) max_outstanding = (int)q.size();
      if (outs_limit > 0 && (int)q.size() > outs_limit)
        tb_fatal("%s: %d bursts outstanding, more than %d", name, (int)q.size(), outs_limit);
      ar_pend = false;
    } else if (*arvalid) {
      ar_pend = true; ar_addr_q = *araddr; ar_len_q = *arlen;
    }
    if (rv && *rready) {
      beats_total++;
      rv = false;
      if (++beat == q.front().beats) { beat = 0; q.pop_front(); }
    }
  }
  bool idle() const { return q.empty() && !rv; }
};

// ---------------------------------------------------------------------------
// Write slave for the 128-bit C port.
// ---------------------------------------------------------------------------
struct AxiWriteSlave {
  const char* name;
  CData *awvalid, *awready, *awlen, *awsize, *awburst;
  QData *awaddr;
  CData *wvalid, *wready, *wlast;
  SData *wstrb;
  VlWide<4>* wdata;
  CData *bvalid, *bready, *bresp, *bid;

  Memory*       mem;
  Timing*       t;
  std::mt19937* rng;

  struct Burst { uint64_t addr; int beats; };
  std::deque<Burst>    aw;
  std::deque<uint64_t> b_due;
  int  wbeat = 0;
  bool bv = false, aw_pend = false;
  uint64_t aw_addr_q = 0; int aw_len_q = 0;
  uint64_t beats_total = 0, bursts_total = 0, partial_beats = 0;
  int      max_outstanding = 0;  // bursts accepted, B not yet taken
  int      outs_limit = 0;       // > 0: the master's declared NUM_WRITE_OUTSTANDING

  bool coin(double p) { return std::uniform_real_distribution<double>(0, 1)(*rng) < p; }

  void drive() {
    *awready = coin(t->p_awready);
    *wready  = !aw.empty() && coin(t->p_wready);
    if (!bv && !b_due.empty() && b_due.front() <= g_cycle && coin(t->p_bvalid)) bv = true;
    *bvalid = bv;
    *bresp  = 0;
    *bid    = 0;
  }

  void sample() {
    if (aw_pend) {
      if (!*awvalid) tb_fatal("%s: AWVALID dropped before AWREADY", name);
      if (*awaddr != aw_addr_q || *awlen != aw_len_q)
        tb_fatal("%s: AW payload changed while waiting", name);
    }
    if (*awvalid && *awready) {
      uint64_t a = *awaddr;
      int beats = *awlen + 1;
      if (*awsize != 4)  tb_fatal("%s: AWSIZE %d", name, *awsize);
      if (*awburst != 1) tb_fatal("%s: AWBURST %d", name, *awburst);
      if (a & 15)        tb_fatal("%s: unaligned AWADDR %llx", name, (unsigned long long)a);
      if ((a >> 12) != ((a + 16ull * beats - 1) >> 12))
        tb_fatal("%s: burst crosses 4 KiB", name);
      aw.push_back({a, beats});
      bursts_total++;
      const int outs = (int)(aw.size() + b_due.size());
      if (outs > max_outstanding) max_outstanding = outs;
      if (outs_limit > 0 && outs > outs_limit)
        tb_fatal("%s: %d bursts outstanding, more than %d", name, outs, outs_limit);
      aw_pend = false;
    } else if (*awvalid) {
      aw_pend = true; aw_addr_q = *awaddr; aw_len_q = *awlen;
    }
    if (*wvalid && *wready) {
      const Burst& bu = aw.front();
      uint64_t a = bu.addr + 16ull * wbeat;
      uint16_t s = *wstrb;
      if (s != 0xFFFF) partial_beats++;
      for (int i = 0; i < 16; i++) {
        if (s & (1u << i)) {
          uint8_t v = (uint8_t)((*wdata)[i / 4] >> (8 * (i % 4)));
          mem->wr8(a + i, v);
          mem->mark(a + i);
        }
      }
      bool last = (wbeat == bu.beats - 1);
      if ((bool)*wlast != last)
        tb_fatal("%s: WLAST=%d on beat %d of %d", name, *wlast, wbeat, bu.beats);
      beats_total++;
      if (last) {
        wbeat = 0;
        aw.pop_front();
        b_due.push_back(g_cycle + 3 + (g_cycle % 7));
      } else {
        wbeat++;
      }
    }
    if (bv && *bready) { bv = false; b_due.pop_front(); }
  }
  bool idle() const { return aw.empty() && b_due.empty() && !bv; }
};

// ---------------------------------------------------------------------------
// AXI4-Lite master (blocking helpers call `tick` until the access completes).
// ---------------------------------------------------------------------------
struct AxiLiteMaster {
  CData *awvalid, *awready, *awaddr, *wvalid, *wready, *wstrb, *bvalid, *bready;
  IData *wdata;
  CData *arvalid, *arready, *araddr, *rvalid, *rready;
  IData *rdata;
  std::function<void()> tick;

  // Handshakes observed at the last rising edge (set by sample()).
  bool aw_hs = false, w_hs = false, b_hs = false, ar_hs = false, r_hs = false;
  uint32_t r_data = 0;

  void idle_inputs() {
    *awvalid = 0; *wvalid = 0; *bready = 0; *arvalid = 0; *rready = 0;
    *awaddr = 0; *wdata = 0; *wstrb = 0; *araddr = 0;
  }
  void sample() {
    aw_hs = *awvalid && *awready;
    w_hs  = *wvalid && *wready;
    b_hs  = *bvalid && *bready;
    ar_hs = *arvalid && *arready;
    r_hs  = *rvalid && *rready;
    if (r_hs) r_data = *rdata;
  }
  void write(uint8_t addr, uint32_t data) {
    *awvalid = 1; *awaddr = addr; *wvalid = 1; *wdata = data; *wstrb = 0xF; *bready = 1;
    for (int guard = 0; ; guard++) {
      if (guard > 1000) tb_fatal("AXI-Lite write %02x timed out", addr);
      tick();
      if (aw_hs) *awvalid = 0;
      if (w_hs)  *wvalid = 0;
      if (b_hs)  break;
    }
    *bready = 0;
  }
  uint32_t read(uint8_t addr) {
    *arvalid = 1; *araddr = addr; *rready = 1;
    for (int guard = 0; ; guard++) {
      if (guard > 1000) tb_fatal("AXI-Lite read %02x timed out", addr);
      tick();
      if (ar_hs) *arvalid = 0;
      if (r_hs)  break;
    }
    *rready = 0;
    return r_data;
  }
};
