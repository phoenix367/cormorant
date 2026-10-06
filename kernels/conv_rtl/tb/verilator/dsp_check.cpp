// dsp_check — drives dsp_check.sv with random operands, seeds and accumulate
// patterns and compares the behavioural chain with the DSP48E2 model every
// cycle; also checks the chain against a scalar model of the documented timing.
#include <cstdint>
#include <cstdio>
#include <deque>
#include <random>
#include "Vdsp_check.h"
#include "verilated.h"

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  Vdsp_check top;
  std::mt19937 rng(7);
  const int TIC = 16, N = 200000;
  struct In { int16_t a[16], b[16]; int32_t c; bool wfb; };
  std::deque<In> hist;                     // instants E, newest at the back
  long bad = 0, checked = 0;
  int64_t acc = 0;                         // scalar model of DSP TIC-1's P
  auto clk = [&] { top.clk = 0; top.eval(); top.clk = 1; top.eval(); };
  for (int t = 0; t < N; t++) {
    In in;
    const int mode = (t / 1000) % 3;       // full-range, small, zero-heavy operands
    for (int l = 0; l < TIC; l++) {
      in.a[l] = mode == 0 ? (int16_t)rng() : mode == 1 ? (int16_t)(rng() % 64 - 32) : ((rng() % 4) ? 0 : (int16_t)rng());
      in.b[l] = mode == 0 ? (int16_t)rng() : mode == 1 ? (int16_t)(rng() % 64 - 32) : ((rng() % 4) ? 0 : (int16_t)rng());
    }
    in.c   = (rng() % 3 == 0) ? (int32_t)rng() : 0;
    in.wfb = (rng() % 7) != 0;
    hist.push_back(in);
    // instant E = t: lane l takes instant t - l; c of instant t - 1; wfb of instant t - TIC
    // pack lanes into the 256-bit ports (lane l at bits [16l+15:16l])
    for (int w = 0; w < 8; w++) {
      uint32_t va = 0, vb = 0;
      for (int h = 0; h < 2; h++) {
        const int l = 2 * w + h;
        const int16_t aa = (size_t)l < hist.size() ? hist[hist.size() - 1 - l].a[l] : 0;
        const int16_t bb = (size_t)l < hist.size() ? hist[hist.size() - 1 - l].b[l] : 0;
        va |= (uint32_t)(uint16_t)aa << (16 * h);
        vb |= (uint32_t)(uint16_t)bb << (16 * h);
      }
      top.a[w] = va;
      top.b[w] = vb;
    }
    top.c   = hist.size() >= 2 ? (uint32_t)hist[hist.size() - 2].c : 0;
    top.wfb = hist.size() >= (size_t)TIC + 1 ? hist[hist.size() - 1 - TIC].wfb : 0;
    clk();
    // the model: instant E's sum is valid in cycle E + TIC + 2, i.e. after the
    // edge of iteration E + TIC + 1
    if ((int)hist.size() > TIC + 1) {
      const In& s = hist[hist.size() - 1 - (TIC + 1)];
      int64_t sum = s.c;
      for (int l = 0; l < TIC; l++) sum += (int64_t)s.a[l] * s.b[l];
      acc = (s.wfb ? acc : 0) + sum;
    }
    if (t > 100) {
      checked++;
      if (top.p_beh != top.p_prim) { if (bad < 10) std::printf("t=%d beh %08x prim %08x\n", t, top.p_beh, top.p_prim); bad++; }
      if (top.p_beh != (uint32_t)acc) { if (bad < 10) std::printf("t=%d beh %08x model %08x\n", t, top.p_beh, (uint32_t)acc); bad++; }
    }
    if (hist.size() > 64) hist.pop_front();
  }
  std::printf("%ld cycles compared, %ld mismatches (behavioural vs DSP48E2 and vs the timing model)\n", checked, bad);
  return bad ? 1 : 0;
}
