// Copyright 2026 Ivan Gubochkin
// SPDX-License-Identifier: Apache-2.0

// dsp_check — drives dsp_check.sv with random operands, seeds and accumulate
// patterns and compares the behavioural chain with the DSP48E2 model every
// cycle; also checks the chain against a scalar model of the documented timing.
#include <cstdint>
#include <cstdio>
#include <deque>
#include <vector>
#include <random>
#include "Vdsp_check.h"
#include "verilated.h"

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  Vdsp_check top;
  std::mt19937 rng(7);
  const int TIC = 16, N = 200000;
  struct In { int16_t a[16], b[16]; int32_t c; bool wfb; };
  // every instant generated up front: the weight lanes lead (cv_mac_chain:
  // lane l at E + l - 1 for l % 4 == 0, else E + l - 2)
  std::vector<In> ins(N + 4);
  for (int t = 0; t < N + 4; t++) {
    In& in = ins[t];
    const int mode = (t / 1000) % 3;       // full-range, small, zero-heavy operands
    for (int l = 0; l < TIC; l++) {
      in.a[l] = mode == 0 ? (int16_t)rng() : mode == 1 ? (int16_t)(rng() % 64 - 32) : ((rng() % 4) ? 0 : (int16_t)rng());
      in.b[l] = mode == 0 ? (int16_t)rng() : mode == 1 ? (int16_t)(rng() % 64 - 32) : ((rng() % 4) ? 0 : (int16_t)rng());
    }
    in.c   = (rng() % 3 == 0) ? (int32_t)rng() : 0;
    in.wfb = (rng() % 7) != 0;
  }
  long bad = 0, checked = 0;
  int64_t acc = 0;                         // scalar model of DSP TIC-1's P
  auto clk = [&] { top.clk = 0; top.eval(); top.clk = 1; top.eval(); };
  for (int t = 0; t < N; t++) {
    // instant E = t: patch lane l takes instant t - l, weight lane l instant
    // t - l + lead(l); c of instant t - 1; wfb of instant t - TIC
    // pack lanes into the 256-bit ports (lane l at bits [16l+15:16l])
    for (int w = 0; w < 8; w++) {
      uint32_t va = 0, vb = 0;
      for (int h = 0; h < 2; h++) {
        const int l = 2 * w + h;
        const int ia = t - l, ib = t - l + ((l % 4 == 0) ? 1 : 2);
        const int16_t aa = ia >= 0 ? ins[ia].a[l] : 0;
        const int16_t bb = ib >= 0 ? ins[ib].b[l] : 0;
        va |= (uint32_t)(uint16_t)aa << (16 * h);
        vb |= (uint32_t)(uint16_t)bb << (16 * h);
      }
      top.a[w] = va;
      top.b[w] = vb;
    }
    top.c   = t >= 1 ? (uint32_t)ins[t - 1].c : 0;
    top.wfb = t >= TIC ? ins[t - TIC].wfb : 0;
    clk();
    // the model: instant E's sum is valid in cycle E + TIC + 2, i.e. after the
    // edge of iteration E + TIC + 1
    if (t >= TIC + 1) {
      const In& s = ins[t - (TIC + 1)];
      int64_t sum = s.c;
      for (int l = 0; l < TIC; l++) sum += (int64_t)s.a[l] * s.b[l];
      acc = (s.wfb ? acc : 0) + sum;
    }
    if (t > 100) {
      checked++;
      if (top.p_beh != top.p_prim) { if (bad < 10) std::printf("t=%d beh %08x prim %08x\n", t, top.p_beh, top.p_prim); bad++; }
      if (top.p_beh != (uint32_t)acc) { if (bad < 10) std::printf("t=%d beh %08x model %08x\n", t, top.p_beh, (uint32_t)acc); bad++; }
    }
  }
  std::printf("%ld cycles compared, %ld mismatches (behavioural vs DSP48E2 and vs the timing model)\n", checked, bad);
  return bad ? 1 : 0;
}
