# On-chip scratchpad between the kernels (URAM) — study

**Status (2026-10-10): study only — NO-GO.**  The latency-bound layers
this would have served reached the ideal-memory bound the same day without
it: the ConvKernel's x port moved off the CCI onto `S_AXI_HP2_FPD` plus the
producer's overlapped loads (`DEPTHWISE_PLAN.md` §8.5: the x-bound jobs
−27…−67 %, on the Verilator ideal column), so the scratch's remaining case
is §4's right column only (≤ 8 % on the MobileNets), against all 16 URAMs,
a riskier crossbar and a second pool.  Kept as the record.  The bound of what an on-chip scratch
can save is 8–20 % on the depthwise CNNs and ~1 % on the compute- and
weight-bound models, and most of it needs tiled producer → consumer
scheduling because the tensors that matter do not fit the 576 KB that is
free; the kernel-side rework reaches the same bound on the same layers
without a capacity limit or a new interconnect.

The user asked: "we have a memory bandwidth limit — what if we use URAM
blocks as an internal cache and implement an interconnect between our
blocks, so they communicate at 250 MHz, 128-bit, without the DDR delay, at
least for small data chunks?"

## 1. What the DDR path costs today

Measured in `PS_PORTS_PLAN.md` §8–§11 and `DEPTHWISE_PLAN.md` §3:

- A PS port is 128-bit at the PL clock: **4 GB/s per direction**, the same
  as a 128-bit on-chip slave at 250 MHz would give.  A scratch does not add
  bandwidth per port; it removes latency (~20 cycles against ~150–200
  through the CCI and the DDRC), DRAM row conflicts and DDR traffic.
- The streaming kernels hide the latency: the standard 3 × 3 ConvKernel
  jobs and the MatmulKernel are within 1–7 % of their ideal-memory time on
  the board.  The binary VectorOP is at 75–90 % of its ceiling (the rest
  DRAM: §11's bank phases took the largest part).
- **What is latency-bound** is the ConvKernel's x path on depthwise and
  few-output-channel 1 × 1 jobs — 144-byte runs, two rows in flight: 1.4–
  3.3× their ideal-memory time on the board.  The Verilator "fast" memory
  model (24-cycle latency, one beat per cycle) is a fair model of a URAM
  scratch, so that column is the scratch's upper bound per job:

| job (LightStereo-S shapes) | board | ideal memory | bound |
|---|---|---|---|
| dw 3 × 3, 16 ch, 120 × 160 | 0.635 ms | 0.461 | −27 % |
| dw 3 × 3 stride 2, 16 ch | 1.926 | 0.736 | −62 % |
| dw 1 × 7 stripe, 16 ch, 60 × 80 | 0.142 | 0.118 | −17 % |
| 1 × 1, 16 → 16 ch, 120 × 160 | 0.564 | 0.196 | −65 % |
| 1 × 1, 144 → 16 ch, 120 × 160 | 4.70 | 1.42 | −70 % |
| standard 3 × 3, 64 ch, 56 × 56 | 0.918 | 0.915 | 0 |

## 2. What is free on the device

Production bitstream `8599aa7a5f12`, placed: URAM **48 / 64 → 16 free =
576 KB** (the ConvKernel's accumulators and caches hold the 48); BRAM 123 /
144 RAMB36 → ~21 free ≈ 94 KB (a few needed by any controller); LUTs 53 %,
registers 37 %, DSP 57 %; timing closes at WNS +0.114 ns at 250 MHz.
A URAM is 72 bits × 4 K; two side by side make the 128-bit (+ parity) word
the ports carry, so 576 KB is 8 × 128-bit × 36 K words, one clock domain,
registered outputs.

## 3. What would fit

Kernel-to-kernel intermediates — produced by a kernel and read only by
kernels (a host op cannot see the scratch; graph inputs and outputs are the
caller's) — against 576 KB, from the graphs:

| model | kernel → kernel tensors | bytes | fit whole (≤ 576 KB) |
|---|---|---|---|
| MobileNet v2 | 97 | 24.7 MB | 83 tensors, 10.7 MB, **43 %** of the bytes |
| MobileNet v1 | 53 | 19.1 MB | 41, 8.4 MB, 44 % |
| ResNet-18 | 44 | 9.0 MB | 43, 7.5 MB, 83 % |
| LightStereo-S | 364 | 586 MB | 119, 41 MB, **7 %** (its quarter-resolution maps are 14 MB each) |

So for the model where the latency-bound layers dominate, whole-tensor
residency covers almost nothing; using the scratch there means tiling:
the producer (a 1 × 1 conv) writes a channel slice or a row band into the
scratch, the consumer (the depthwise conv) reads it, the result goes back
to DDR or to the next consumer — with double buffering in 2 × 288 KB and
one more kernel call (431 cycles of overhead each) per tile.

## 4. The bound per model

Per kernel call of each model, priced with the board's performance model
(`perf_models/kv260/8599aa7a5f12.json`; LightStereo-S from its board
profile `demo/stereo_depth/build/profile_by_kind.json`), with the
ideal-memory factors of §1 by kind (depthwise stride 1 ×0.72, stride 2
×0.38, stripes ×0.83, 1 × 1 with ≤ 32 output channels ×0.32, binary
VectorOP ×0.8, unary ×0.9, everything else ×1):

| model | board | ideal memory for every byte | …for the tensors that fit whole |
|---|---|---|---|
| MobileNet v1 | 21.2 ms | −2.1 ms (**−10 %**) | −0.9 ms (−4 %) |
| MobileNet v2 | 19.6 ms | −3.9 ms (**−20 %**) | −1.5 ms (−8 %) |
| ResNet-18 | 18.5 ms | −0.2 ms (−1 %) | −0.2 ms |
| LightStereo-S | 599 ms | ≈ −95 ms (**−16 %**: dw 3 × 3 159 → 115, stripes 76 → 63, rescales 23 → 16, the 1 × 1 convs with ≤ 32 output channels 27 → 9, ADD 38 → 30, ReLU6 55 → 50) | ≈ −10 ms (−2 %) |
| BERT, SmolLM2, SmolVLM, Piper | | ~0: MatMul / 3 × 3-conv compute and weight streams from DDR; the VectorOP share is 4–8 % and the bank phases took most of its gap | |

These are upper bounds: every byte of the listed calls on a 24-cycle,
1-beat-per-cycle memory, no tiling overhead, no scratch bandwidth sharing
(three streams of one VectorOP call through one 4 GB/s slave would be
slower than today's three ports).

## 5. The design, if built

- **Hardware.**  `axi_interconnect_0` (HPC0) gets a second MI to an AXI
  BRAM controller (128-bit, 250 MHz) in front of a `blk_mem_gen` of 16
  URAMs (`PRIM_type_to_Implement URAM`), 576 KB at `0xB000_0000`; the
  kernels' address spaces include the segment, so a kernel reads or writes
  the scratch exactly as it reads DDR (the registers hold 64-bit
  addresses) — no kernel change.  Two slaves (two MIs, 2 × 288 KB) would
  keep a VectorOP's a and c on separate slaves.  Risk: the crossbar grows
  (the HPC0 one is 4 SI × 1 MI today), the design has 0.114 ns of slack;
  URAM cascades and the controller's registered outputs are 250 MHz
  friendly.  LUT +2–4 k.
- **Host visibility.**  The scratch is not cache-coherent with the APU
  and `M_AXI_HPM0_FPD` reaches it only through the control interconnect
  (32-bit AXI-Lite speed): only kernel → kernel tensors qualify; a tensor
  a host op reads stays in DDR.
- **Scheduler.**  A second pool with its own liveness colouring restricted
  to kernel → kernel tensors ≤ cap (the colouring of
  `SCHEDULER_DAG.md` §6 reused; the address table gains a base per pool);
  the gain of §4's right column.  For the left column: tiled issue of
  producer → consumer pairs — the ConvKernel's output-channel and row-band
  splits already exist as call parameters (`dw_slice`, the sweep
  geometry), so the tiling is a scheduler transform: split a (1 × 1 → dw)
  pair into tiles whose intermediate fits half the scratch, alternate the
  halves, order the calls so the consumer of tile i runs while the producer
  makes tile i + 1 (different lanes are needed for overlap — both are
  ConvKernel calls, so they serialise; the gain is then latency only).
  Bit-exact by construction.  This is the real cost: a new pass over the
  DAG, the cost model's tile choice, tests, the planner's call lists.
- **Testbench / simulation.**  The Verilator testbenches' memory model
  gains a scratch region with its own timing; `sim_hw_kv260` gets the
  controller in the block design.

## 6. Verdict and order

1. **Not now.**  The same layers reach the same ideal-memory bound through
   the kernel's own x path (`DEPTHWISE_PLAN.md` option B: longer runs,
   prefetch, loads beside the sweep) — no capacity limit, no tiling, no
   interconnect change, and it helps every depthwise / few-channel job
   whatever its size.  B first.
2. **After B**, re-measure the table of §1: what is left of the 1.4–3.3×
   is the scratch's remaining case.  If the depthwise jobs are then within
   ~10 % of ideal, the scratch's value is the VectorOP chains and the
   mid-/late-layer residency of the MobileNets (§4 right column, ≤ 8 %),
   against +2–4 k LUT, a riskier crossbar and a second pool in the
   scheduler — a modest GO at best.
3. The cheapest sibling of the idea, worth doing on its own: **fusing the
   residual add into the ConvKernel** (a fourth read port `r`, added to
   y before the write) removes ResNet-18's and BERT's binary VectorOP
   calls and their DDR round trip entirely; and the ConvKernel's x path is
   where the latency lives — B is the fix.
