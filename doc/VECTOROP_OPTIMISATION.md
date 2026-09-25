# VectorOPKernel — Optimization Log

This document records performance investigation and optimization work on
`kernels/vectorop/kernel/VectorOP.cpp`.  Each section describes one change,
the rationale, the HLS traps met on the way, and the measured HW
behaviour-simulation (`make behavior_test_vectorop`) impact on the kv260
RTL test stand (`hw/cormorant_test_stand/kernels/vector_op_test`).

For the kernel description see [VECTOROP_KERNEL.md](VECTOROP_KERNEL.md);
the plan this work executes is THROUGHPUT_PLAN.md §4 (Track C).

All numbers are `duration_ns` per case from
`build/kernels/vectorop/kv260/vector_op_test_report.json`.  The testbench
runs at a fixed 100 MHz sim basis, so they are directly comparable between
runs regardless of the HLS target clock; a case's duration includes the
testbench's DDR fill and register programming (~4 µs per case).

---

## 1. Baseline (2026-09-25, main 56fc2eb)

One `ap_fixed<16,8>` element per beat on every port: the three `m_axi`
ports were plain `Data_t*` with `max_widen_bitwidth=512`, but HLS reported
`Widen Fail` on the runtime-strided access (`o * stride + i`), so every
stream moved 16 bits per cycle — the ceiling the on-board numbers showed
(ADD 256K: 0.59 GB/s).  Each stage was an `(outer, size)` loop nest whose
inner II=1 loop restarted per outer iteration (~33 cycles), which is what
made the broadcast `MUL 12544 × 16` case cost 6.1 ms on the board.

Synthesis: II=1 on every loop, estimated 4.867 ns, BRAM 17, DSP 26,
FF 8 520, LUT 8 155.  RTL: 61/61, `sim_time_ns = 400 295`.  The test-stand
block design at that point still had a 32-bit `S_AXI_HPC0_FPD` / crossbar
(the vector_op project had not been part of the 2026-09-24 width change),
so the baseline column below is the kernel *and* the narrow PS port —
i.e. what the board ran before this change.

---

## 2. 128-bit ports, 8 lanes per cycle, flattened stages, `act` register (C1–C4)

**Change.**  `a`, `b`, `c` are `hls::burst_maxi<VecWord>` with
`VecWord = ap_uint<128>` (`kVecLanes = 8` elements per beat; `VectorOP.h`).
The DDR layout is unchanged; the kernel requests whole word ranges and
extracts / packs lanes itself, exactly like MatmulKernel's A/B ports
(MATMUL_OPTIMISATION.md §3) and ConvKernel's `y` writer (§2.27 / §2.30):

* **`load_words`** — one flattened `PIPELINE II=1` loop over all
  `outer × ceil(size/8)` words.  The operand is classified once:
  `outer == 1`, or `inc == size` on whole words → one contiguous range
  (the dw chunk-16 pattern); `inc == 0` with ≤ 256 words → read once into
  `VecWord rep_buf[256]` and replayed; otherwise each run is its own word
  range at stride `inc / 8`.  Requests are ≤ `kReadReqWords = 64` words
  and `kReadAhead = 8` of them are kept in flight ahead of the `read()`
  cursor (a prologue issues the first 8, then one request is issued at
  every piece boundary).  Lanes past `size` in a run's last word are
  masked to zero.  For unary ops the `b` loader does nothing.
* **`compute_words`** — 8 unrolled lanes per cycle: `switch (op)` on
  each lane, then `apply_act` (the new `act` register: 0 none / 1 relu /
  2 relu6).  `OP_DIV` is routed to `compute_div`, a lane-serial II=1 loop
  around ONE divider (1 element per cycle, as before) — 8 pipelined
  dividers were not worth ~7 k LUT for an op no shipped model uses.
* **`store_words`** — one flattened II=1 loop: a `write_request` per
  ≤ 256-word piece issued in the same iteration as the piece's first
  `write()`, responses collected in a sliding window of 8 (the §2.30
  bound: ≤ 8 unacknowledged requests < `num_write_outstanding = 16`).
  The last word of every run is written whole — the **alignment contract**
  documented in `VectorOP.h`: every run start is 16-byte aligned (`a_inc`
  / `b_inc` are 0 or a multiple of 8; the scheduler's `CHUNK_STRIDE` and
  64-byte buffer bases guarantee it), `size` is arbitrary, and `c[size ..
  ceil8(size))` of each run receives `op(0, 0) = 0`.  `TestSimulation`
  asserts `inc % 8 == 0`, checks the tail lanes read 0 and the rest of the
  stride gap is untouched.
* **`act`** is appended LAST in the argument list, so every existing
  register offset is unchanged (new register at 0x5C).

Port pragmas: `max_read_burst_length=64 num_read_outstanding=16` on a / b,
`max_write_burst_length=256 num_write_outstanding=16` on c;
`max_widen_bitwidth` dropped.  The read-side choice is a BRAM decision:
the HLS m_axi adapter's read buffer is `num_read_outstanding ×
max_read_burst_length` beats, so 256 × 16 at 128 bits would be 32 BRAM18
per port (64 for a + b); 64 × 16 is 8 per port, and 16 requests in flight
suit the short-run broadcast pattern better than 4 long ones.  The
prologue + per-piece issue keeps ≤ 9 requests (≤ 576 words) outstanding,
below both the outstanding count and the 1024-word buffer, so a
`read_request()` can never block ahead of the `read()` that would drain
the buffer (the read-side twin of the §2.30 deadlock).

**HLS traps hit.**

| Form | Result |
|---|---|
| `hls::stream<VecWord>` depth 64 with default storage | 8 BRAM18 per FIFO (width-limited packing of a 64 × 128-bit FIFO), 24 in total — `bind_storage type=fifo impl=lutram` brings it to 0 |
| `op < OP_RELU` written inline as a dataflow-call argument | HLS 214-113 dataflow-form warning; hoisted into a scalar before the `dataflow` pragma (the 200-1449 "reads an input from its caller" note on the `b` loader remains and is benign) |
| `read_request` + `read` in the same II=1 loop iteration | schedules at II=1 (AR and R are different channels) — no need for the separate request loop the other kernels use |
| `write_request` + `write` + conditional `write_response` in one II=1 iteration | II=1 as well |
| lane-serial `compute_div` assembling the output word through a shift register | would put the divider latency into the loop-carried chain; per-lane result registers (`res[8]`, partitioned) written at the lane index and assembled at lane 7 give II=1 |

**Synthesis** (kv260, 150 MHz target): II=1 on every loop, slack ≥ 0
(0.00 on the stage loops, estimated 4.87 ns, same as the baseline), ports
`m_axi_gmem0/1/2` 128 → 128 in the interface table.  Resources
baseline → new: BRAM18 17 → **32** (3 adapters × 8 + 2 × 4 for `rep_buf`),
DSP 26 → 33, FF 8 520 → 13 725, LUT 8 155 → **22 657** (HLS estimate;
the 128-bit adapters, two `stream_runs` instances at ~3 k, the 8-lane
compute at 3.6 k and the lane-serial divider at 1.8 k).

**Test stand.**  `vector_op_test` upgraded to the new IP;
`PSU__SAXIGP0__DATA_WIDTH` and `XBAR_DATA_WIDTH` 32 → 128 and
`C_M_AXI_GMEM0/1/2_DATA_WIDTH` set to the IP defaults (128) — the same
state the other three projects reached on 2026-09-24.  `vectorop_tb.sv`
programs the `act` register (0x5C), derives the a / b / c extents from
`(outer, a_inc, b_inc, size)` and compares only the valid positions of a
strided output.  Fixtures regenerated: 61 → **119** cases (sizes 1, 3, 8,
9, 13, 64, 255, 256, 1023, 1024, 4097 × 6 ops; 31 saturation cases; 22
geometry / `act` cases including `run 5000 words` (> 16 × 256 words, the
§2.30 bound) and the broadcast, replay and multi-piece paths).

**Result (RTL, 119/119 PASS, bit-exact against the C-sim oracle; the 61
baseline geometries all present).**  Per-element slope between the 1024-
and 4097-element cases (kernel + the testbench's proportional DDR fill):

| Op | baseline ns/elem (256→1024) | new ns/elem (1024→4097) |
|---|---:|---:|
| ADD | 13.32 | 3.05 |
| SUB | 13.40 | 2.81 |
| MUL | 13.80 | 3.00 |
| DIV | 13.92 | 10.50 |
| RELU | 10.82 | 1.75 |
| RELU6 | 11.33 | 1.69 |

(`RELU` has no `b` fill, so its slope is the closest to the kernel alone:
10.8 → 1.75 ns per element ≈ 6.2×; the 8-lane ceiling is 1.25 ns at the
100 MHz sim clock.  DIV keeps one element per cycle by design.)

Per case, baseline → new `duration_ns` (each includes ~4 µs of testbench
DDR fill / register programming, which is why the 1- and 8-element cases
are flat and scatter ±15 % — the baseline's own eight identical 8-element
`SUB` cases already ranged 3 940–5 220 ns):

| # | Case | op | size | outer | baseline | new | Δ |
|--:|---|---|---:|---:|---:|---:|---:|
| 0 | ADD | ADD | 1 | 1 | 4,915 | 4,765 | -3.1 % |
| 1 | ADD | ADD | 3 | 1 | — | 4,660 | new |
| 2 | ADD | ADD | 8 | 1 | 4,690 | 4,440 | -5.3 % |
| 3 | ADD | ADD | 9 | 1 | — | 4,180 | new |
| 4 | ADD | ADD | 13 | 1 | — | 4,560 | new |
| 5 | ADD | ADD | 64 | 1 | 5,700 | 4,390 | -23.0 % |
| 6 | ADD | ADD | 255 | 1 | — | 5,270 | new |
| 7 | ADD | ADD | 256 | 1 | 9,520 | 5,130 | -46.1 % |
| 8 | ADD | ADD | 1,023 | 1 | — | 8,030 | new |
| 9 | ADD | ADD | 1,024 | 1 | 19,750 | 8,030 | -59.3 % |
| 10 | ADD | ADD | 4,097 | 1 | — | 17,390 | new |
| 11 | SUB | SUB | 1 | 1 | 4,470 | 3,890 | -13.0 % |
| 12 | SUB | SUB | 3 | 1 | — | 4,200 | new |
| 13 | SUB | SUB | 8 | 1 | 5,040 | 4,700 | -6.7 % |
| 14 | SUB | SUB | 9 | 1 | — | 4,440 | new |
| 15 | SUB | SUB | 13 | 1 | — | 4,370 | new |
| 16 | SUB | SUB | 64 | 1 | 5,360 | 4,860 | -9.3 % |
| 17 | SUB | SUB | 255 | 1 | — | 4,870 | new |
| 18 | SUB | SUB | 256 | 1 | 9,480 | 5,260 | -44.5 % |
| 19 | SUB | SUB | 1,023 | 1 | — | 8,260 | new |
| 20 | SUB | SUB | 1,024 | 1 | 19,770 | 8,320 | -57.9 % |
| 21 | SUB | SUB | 4,097 | 1 | — | 16,940 | new |
| 22 | MUL | MUL | 1 | 1 | 4,800 | 4,650 | -3.1 % |
| 23 | MUL | MUL | 3 | 1 | — | 4,070 | new |
| 24 | MUL | MUL | 8 | 1 | 4,380 | 4,340 | -0.9 % |
| 25 | MUL | MUL | 9 | 1 | — | 4,340 | new |
| 26 | MUL | MUL | 13 | 1 | — | 4,580 | new |
| 27 | MUL | MUL | 64 | 1 | 5,290 | 4,200 | -20.6 % |
| 28 | MUL | MUL | 255 | 1 | — | 5,040 | new |
| 29 | MUL | MUL | 256 | 1 | 9,330 | 5,810 | -37.7 % |
| 30 | MUL | MUL | 1,023 | 1 | — | 7,940 | new |
| 31 | MUL | MUL | 1,024 | 1 | 19,930 | 8,390 | -57.9 % |
| 32 | MUL | MUL | 4,097 | 1 | — | 17,620 | new |
| 33 | DIV | DIV | 1 | 1 | 4,410 | 4,900 | +11.1 % |
| 34 | DIV | DIV | 3 | 1 | — | 4,900 | new |
| 35 | DIV | DIV | 8 | 1 | 4,610 | 4,290 | -6.9 % |
| 36 | DIV | DIV | 9 | 1 | — | 4,410 | new |
| 37 | DIV | DIV | 13 | 1 | — | 5,480 | new |
| 38 | DIV | DIV | 64 | 1 | 5,640 | 5,440 | -3.5 % |
| 39 | DIV | DIV | 255 | 1 | — | 7,540 | new |
| 40 | DIV | DIV | 256 | 1 | 9,160 | 7,630 | -16.7 % |
| 41 | DIV | DIV | 1,023 | 1 | — | 16,700 | new |
| 42 | DIV | DIV | 1,024 | 1 | 19,850 | 16,520 | -16.8 % |
| 43 | DIV | DIV | 4,097 | 1 | — | 48,780 | new |
| 44 | RELU | RELU | 1 | 1 | 4,570 | 5,010 | +9.6 % |
| 45 | RELU | RELU | 3 | 1 | — | 4,160 | new |
| 46 | RELU | RELU | 8 | 1 | 4,880 | 4,410 | -9.6 % |
| 47 | RELU | RELU | 9 | 1 | — | 4,340 | new |
| 48 | RELU | RELU | 13 | 1 | — | 3,820 | new |
| 49 | RELU | RELU | 64 | 1 | 5,290 | 4,480 | -15.3 % |
| 50 | RELU | RELU | 255 | 1 | — | 4,650 | new |
| 51 | RELU | RELU | 256 | 1 | 8,440 | 4,660 | -44.8 % |
| 52 | RELU | RELU | 1,023 | 1 | — | 6,720 | new |
| 53 | RELU | RELU | 1,024 | 1 | 16,750 | 6,760 | -59.6 % |
| 54 | RELU | RELU | 4,097 | 1 | — | 12,140 | new |
| 55 | RELU6 | RELU6 | 1 | 1 | 4,450 | 4,420 | -0.7 % |
| 56 | RELU6 | RELU6 | 3 | 1 | — | 4,260 | new |
| 57 | RELU6 | RELU6 | 8 | 1 | 4,760 | 4,270 | -10.3 % |
| 58 | RELU6 | RELU6 | 9 | 1 | — | 4,720 | new |
| 59 | RELU6 | RELU6 | 13 | 1 | — | 4,870 | new |
| 60 | RELU6 | RELU6 | 64 | 1 | 5,430 | 4,550 | -16.2 % |
| 61 | RELU6 | RELU6 | 255 | 1 | — | 5,040 | new |
| 62 | RELU6 | RELU6 | 256 | 1 | 8,160 | 5,000 | -38.7 % |
| 63 | RELU6 | RELU6 | 1,023 | 1 | — | 6,550 | new |
| 64 | RELU6 | RELU6 | 1,024 | 1 | 16,860 | 7,000 | -58.5 % |
| 65 | RELU6 | RELU6 | 4,097 | 1 | — | 12,190 | new |
| 66 | ADD  100 100 200        sat max | ADD | 8 | 1 | 4,770 | 4,090 | -14.3 % |
| 67 | ADD   64 64 128         sat max | ADD | 8 | 1 | 5,190 | 4,680 | -9.8 % |
| 68 | ADD  max 1LSB 128       sat max | ADD | 8 | 1 | 4,920 | 4,330 | -12.0 % |
| 69 | ADD -100-100 -200       sat min | ADD | 8 | 1 | 4,680 | 4,250 | -9.2 % |
| 70 | ADD  min-1LSB           sat min | ADD | 8 | 1 | 4,870 | 4,290 | -11.9 % |
| 71 | ADD   64 63 996 max  no clip | ADD | 8 | 1 | 4,640 | 4,140 | -10.8 % |
| 72 | ADD  -64-64 -128 min  no clip | ADD | 8 | 1 | 4,950 | 4,120 | -16.8 % |
| 73 | SUB  100- -100  200     sat max | SUB | 8 | 1 | 3,940 | 4,490 | +14.0 % |
| 74 | SUB  max- -1LSB         sat max | SUB | 8 | 1 | 4,390 | 4,310 | -1.8 % |
| 75 | SUB -100-100 -200       sat min | SUB | 8 | 1 | 5,220 | 4,310 | -17.4 % |
| 76 | SUB  min-1LSB           sat min | SUB | 8 | 1 | 4,610 | 4,320 | -6.3 % |
| 77 | MUL  16  16 256          sat max | MUL | 8 | 1 | 4,560 | 4,670 | +2.4 % |
| 78 | MUL  12  12 144          sat max | MUL | 8 | 1 | 4,340 | 4,300 | -0.9 % |
| 79 | MUL -16  -16 256         sat max | MUL | 8 | 1 | 4,720 | 4,200 | -11.0 % |
| 80 | MUL -16  16 -256         sat min | MUL | 8 | 1 | 4,690 | 4,630 | -1.3 % |
| 81 | MUL  11  11 121  no clip | MUL | 8 | 1 | 4,410 | 4,580 | +3.9 % |
| 82 | RELU -1LSB     0 | RELU | 8 | 1 | 4,480 | 4,480 | +0.0 % |
| 83 | RELU -1 0      0 | RELU | 8 | 1 | 4,640 | 4,660 | +0.4 % |
| 84 | RELU  0        0 | RELU | 8 | 1 | 4,320 | 4,580 | +6.0 % |
| 85 | RELU  1LSB      1LSB  no clip | RELU | 8 | 1 | 4,300 | 4,570 | +6.3 % |
| 86 | RELU  3 5      3 5  no clip | RELU | 8 | 1 | 4,390 | 4,140 | -5.7 % |
| 87 | RELU6 -1        0 | RELU6 | 8 | 1 | 4,700 | 4,560 | -3.0 % |
| 88 | RELU6 -1LSB     0 | RELU6 | 8 | 1 | 4,530 | 4,510 | -0.4 % |
| 89 | RELU6  0        0  no clip | RELU6 | 8 | 1 | 4,310 | 4,620 | +7.2 % |
| 90 | RELU6  1LSB      1LSB  no clip | RELU6 | 8 | 1 | 4,720 | 4,240 | -10.2 % |
| 91 | RELU6  3 0      3 0  no clip | RELU6 | 8 | 1 | 4,620 | 4,480 | -3.0 % |
| 92 | RELU6  6-1LSB     6-1LSB  no clip | RELU6 | 8 | 1 | 4,870 | 4,300 | -11.7 % |
| 93 | RELU6  6 0      6 0  no clip | RELU6 | 8 | 1 | 4,170 | 4,590 | +10.1 % |
| 94 | RELU6  6 1LSB     6 | RELU6 | 8 | 1 | 4,700 | 4,660 | -0.9 % |
| 95 | RELU6  8        6 | RELU6 | 8 | 1 | 4,480 | 4,580 | +2.2 % |
| 96 | RELU6 20        6 | RELU6 | 8 | 1 | 4,280 | 4,590 | +7.2 % |
| 97 | bcast b chunk12 stride16 ADD | ADD | 12 | 5 | — | 4,360 | new |
| 98 | bcast a chunk12 stride16 MUL | MUL | 12 | 5 | — | 5,100 | new |
| 99 | bcast b chunk13 stride16 SUB | SUB | 13 | 9 | — | 5,230 | new |
| 100 | bcast a chunk9 stride16 DIV | DIV | 9 | 4 | — | 5,260 | new |
| 101 | outer1000 x size16 MUL | MUL | 16 | 1000 | — | 26,310 | new |
| 102 | outer1000 x size16 ADD a-bcast | ADD | 16 | 1000 | — | 26,630 | new |
| 103 | stride0 a 2048  replay max | ADD | 2,048 | 3 | — | 18,660 | new |
| 104 | stride0 a 2100    replay  re-read | SUB | 2,100 | 3 | — | 22,050 | new |
| 105 | stride0 b 2100    replay  re-read | MUL | 2,100 | 3 | — | 21,770 | new |
| 106 | size1000 stride1008  2 read pieces | ADD | 1,000 | 3 | — | 10,890 | new |
| 107 | size3000 stride3008  2 write pieces | SUB | 3,000 | 2 | — | 21,450 | new |
| 108 | unary bcast-shaped RELU | RELU | 12 | 4 | — | 4,410 | new |
| 109 | unary outer3 size20 RELU6 | RELU6 | 20 | 3 | — | 4,620 | new |
| 110 | run 5000 words ADD | ADD | 40,000 | 1 | — | 106,770 | new |
| 111 | run 5000 words RELU | RELU | 40,000 | 1 | — | 57,910 | new |
| 112 | ADD   act RELU | ADD | 255 | 1 | — | 5,240 | new |
| 113 | ADD   act RELU6 | ADD | 1,023 | 1 | — | 8,020 | new |
| 114 | SUB   act RELU | SUB | 64 | 1 | — | 4,270 | new |
| 115 | MUL bcast   act RELU6 | MUL | 12 | 5 | — | 4,690 | new |
| 116 | DIV   act RELU6 | DIV | 100 | 1 | — | 5,430 | new |
| 117 | RELU   act RELU6 | RELU | 33 | 1 | — | 4,370 | new |
| 118 | outer1000 x size16 ADD   act RELU | ADD | 16 | 1000 | — | 26,620 | new |

Sum over the 61 matched cases: 398,095 → 307,785 ns (-22.7 %); the
1024-element cases −58…−60 % (ADD 19 750 → 8 030 ns), 256-element −38…−46 %.
The new geometry cases: `outer1000 x size16` (16 000 elements in 1 000
16-element runs — the dw chunk-16 pattern that cost 6.1 ms for 12 544 runs
on the board) runs in 26.3 µs including its 64 KB testbench fill, i.e. well
under 1.5 ns per element instead of the baseline's ~49 cycles per run
(≈ 490 µs modelled); `run 5000 words ADD` (40 000 elements, > 16 × 256
words) 106.8 µs, `RELU` 57.9 µs; the stride-0 replay (2 048) and re-read
(2 100) paths and the 2-piece read / write runs all pass.

**Board.**  Not measured here (the coordinator integrates the bitstream);
the projection from THROUGHPUT_PLAN.md §4 is ADD 256K 2.65 → ~0.8–0.95 ms,
RELU 64K 0.67 → ~0.12–0.17 ms, MUL bcast 12544×16 6.1 → ~0.3 ms.  In
`hw/cormorant_hw_128` the `VectorOPKernel_0` instance parameters
`C_M_AXI_GMEM0/1/2_DATA_WIDTH` must be set to 128 (the new IP defaults)
after the IP upgrade — an upgrade keeps user-set values (§2.36) — and the
new `act` register (0x5C) proven with a write-then-read before trusting
results.  Demo projects must be regenerated (the scheduler now emits `act`;
`run_op()` writes it every call, so a stale project would still be
correct, just unfused).

---

### On board (2026-09-25, bitstream WNS +1.88 ns, VectorOPKernel_0 instance widths 128)

144/144 scheduler models PASS; the `act` register (0x5C) proven by a
write-then-read.  `run_remote_perf.py`, before → after:

| Case | before | after | |
|---|---:|---:|---:|
| ADD-1K | 19 µs | 9 µs | 2.1× |
| ADD-16K | 0.175 ms | 0.048 ms | 3.6× |
| ADD-256K | 2.653 ms / 0.59 GB/s | 0.663 ms / 2.37 GB/s | 4.0× |
| MUL-64K | 0.671 ms | 0.171 ms | 3.9× |
| RELU-64K | 0.668 ms / 0.39 GB/s | 0.090 ms / 2.90 GB/s | 7.4× |
| RELU6-16K | 0.173 ms | 0.028 ms | 6.2× |
| ADD-bcast-8x16K | 1.376 ms | 0.338 ms | 4.1× |
| MUL-bcast-dw-12544x16 | 6.135 ms / 0.20 GB/s | 0.261 ms / 4.61 GB/s | 23.5× |
| DIV-4K | 0.051 ms | 0.049 ms | (lane-serial by design) |

Binary ops reach 2.3–2.4 GB/s of the 2.4 GB/s port-bound ideal (the two
read streams share the 1.6 GB/s read channel); unary ops 2.9–3.0 GB/s.
Demos with Relu/Clip fusion enabled in their generators (`fuse_act=True`):
MobileNet v2 400 → **344 ms**, ResNet-18 372 → **345 ms**, MobileNet v1
489 → **444 ms**, MNIST convnet 0.898 → **0.813 ms**, LeNet 7.99 →
**7.65 ms**; predictions and logits identical.

## 3. Scheduler: `act` register and Relu / Clip(0,6) fusion (C4)

`inference-scheduler`: `run_op()` now writes `act = VECTOROP_ACT_NONE`
every call (the register keeps its last value across runs); a node with a
fused activation is emitted as `run_op_act(…, VECTOROP_ACT_RELU|RELU6)`.
`OnnxGraph(fuse_act=True)` (the CLI default, `--no-fuse-act` to disable;
`OnnxGraph()` itself defaults to off so library callers and the existing
1 306 tests see unchanged code) folds a `Relu` / `Clip(0,6)` into the
VectorOP node that produces its input when that tensor has no other
consumer and is not a graph output; the producer writes the activation's
output tensor, the intermediate disappears, node indices are renumbered,
the simulation applies the activation before quantisation (it commutes
with the saturating truncation) and the report lists the fusions.  Conv /
MatMul / Pool producers are not fused (no `act` register).  The alignment
contract is now explicit on the scheduler side too: `CHUNK_STRIDE %
INFERENCE_ALIGN_ELEMS == 0` and both `inference_buf_alloc()` implementations
round to 64 bytes so the whole-word tail write stays inside the allocation.
18 new tests (`test/test_act_fusion.py`): 1 324 pass.  ResNet-18's residual
`Add → Relu` pairs and every `Add → Clip` become one pass each once the
demo projects are regenerated with the CLI default (the demos'
`generate_project.py` call `OnnxGraph()` directly and need `fuse_act=True`).

