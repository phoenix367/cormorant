# VectorOPKernel in SystemVerilog

**Status (2026-10-05):** the hardware build's VectorOPKernel since
[VECTOROP_RTL_PLAN](../plans/VECTOROP_RTL_PLAN.md) phase 3 (bitstream
`68665fc1833a`); the Vitis HLS kernel's synthesis is retired, its C++
([VECTOROP_KERNEL](VECTOROP_KERNEL.md)) is the reference model and fixture
generator.  **2026-10-06:** the activation unit (`vo_act`: LeakyReLU, SiLU,
GELU, GELU tanh as ops 6–9 and acts 3–6, the `alpha` register;
[ACTIVATIONS_PLAN](../plans/ACTIVATIONS_PLAN.md)) — verified out of context
at 300 MHz; in the production bitstream `6436623029f7` (250 MHz, WNS
+0.061 ns) since 2026-10-06.  **2026-10-07:** the softmax unit (`vo_smx`: ops
10 / 11, registers `smx_cm` / `smx_cfg` / `smx_mask`;
[SOFTMAX_PLAN](../plans/SOFTMAX_PLAN.md)) — in the bitstreams since
`588d721997cb` (2026-10-07); the production `8599aa7a5f12` (250 MHz, WNS +0.114 ns,
since 2026-10-08) reads b through its own PS port, HPC1 (a and c on HPC0;
[PS_PORTS_PLAN](../plans/PS_PORTS_PLAN.md) §5: binary ops 24–38 % faster).

A drop-in replacement for the HLS kernel: the same IP (VLNV
`xilinx.com:hls:VectorOPKernel:1.0`, 155 ports and 35 parameters with the
HLS export's names and defaults), the same AXI-Lite register map (plus
`alpha` since the activation unit), the same DDR access pattern and
bit-identical results.  All three m_axi ports are
128 bits wide, as on the HLS kernel.

## 1. Contract

**Registers** (`s_axi_ctrl`, 7 address bits): `0x00` ap_ctrl (b0 ap_start,
b1 ap_done clear-on-read, b2 ap_idle, b3 ap_ready clear-on-read, b7
auto_restart, b9 interrupt), `0x04` GIE, `0x08` IER, `0x0C` ISR (toggle on
write), `0x10/14` a, `0x1C/20` b, `0x28/2C` c, `0x34` size, `0x3C` op,
`0x44` outer, `0x4C` a_inc, `0x54` b_inc, `0x5C` act, `0x64` alpha
(LeakyReLU's slope: bits 15:0 / 65536; bits 31:16 are stored and read back
but unused), `0x6C` smx_cm (the softmax's Cm, bits 23:0), `0x74` smx_cfg (Cs
5:0, f_p 12:8), `0x7C` smx_mask (valid0 15:0, period 31:16).  `interrupt` is level high while GIE and a set ISR bit.

**Arithmetic** per Q8.8 lane (raw int16):

| op | result |
|---|---|
| 0 ADD, 1 SUB | `sat16(a ± b)` |
| 2 MUL | `sat16((a · b) >>> 8)` — floor, then saturate |
| 3 DIV | `b == 0 ? 0 : sat16(trunc((a << 8) / b))` — the quotient truncated toward zero |
| 4 RELU | `max(a, 0)` |
| 5 RELU6 | `min(max(a, 0), 0x0600)` |
| 6 LEAKY_RELU, 7 SILU, 8 GELU, 9 GELU_TANH | `a`, then the activation op − 3 (the act register is not applied) |
| 10 SOFTMAX, 11 SOFTMAX_T | the softmax below (act not applied) |
| ≥ 12 | `a` |

then `act` on the result: 1 RELU, 2 RELU6, 3 LEAKY_RELU, 4 SILU, 5 GELU,
6 GELU_TANH, anything else none.  Ops ≥ 4 read no b (nothing is issued on
gmem1).  The activations 3–6 give the exact function of the Q8.8 value
rounded to nearest, ties to even:

| act | result (raw int16 x) |
|---|---|
| 3 LEAKY_RELU | `x ≥ 0 ? x : round(x · alpha / 2¹⁶)` |
| 4 SILU, 5 GELU (erf), 6 GELU_TANH | `max(x, 0) − rom[BASE + |x|]` for `|x| < LEN`, else `max(x, 0)` |

SiLU and both GELUs are `x · F(x)` with `F(−x) = 1 − F(x)`, so
`f(x) = max(x, 0) + f(−|x|)`, and — `max(x, 0)` being a whole number of LSBs
— the rounded values obey the same identity: the ROM holds
`−round(256 · f(−t/256))` (0…71, 7 bits) for the `t` where it is non-zero:
SiLU 2 141 entries (|x| < 8.36), GELU 829 (|x| < 3.24), GELU tanh 816
(|x| < 3.19), 3 786 of 4 096 (`rtl/vo_act_rom.sv`, written and checked by
`scripts/gen_act_rom.py`).

**The softmax** (ops 10 / 11; VectorOP.h `smx_vector`, the specification
`inference-scheduler/src/vectorop_smx.py`).  Per vector of n raw inputs x, v
valid (`v(q) = min(n, valid0 + (period ? q mod period : 0))` for vector q):
`m = max_{j<v} x_j`, `y_j = ((m − x_j) · Cm) >> Cs`,
`e_j = TAB[y_j mod 4096] >> (y_j div 4096)` (`TAB[k] = round(2¹⁶ · 2^(−k/4096))`,
0 past 17 · 4096), `S = Σ e_j`, `R = ⌊2⁴⁰ / S⌋`,
`P_j = min((e_j · R + 2^(39 − f_p)) >> (40 − f_p), 32767)`, 0 for j ≥ v —
within 1 LSB of the exact softmax.  Row mode (10): `outer` rows of `size`
(≤ 2048) at input stride `a_inc`, written at stride `b_inc`.  Column mode
(11): the input `s[size keys][outer queries]` at row stride `a_inc`, the
output `P[outer & ~15][size]` at row stride `b_inc` (≤ 1024 keys; the kernel
reads blocks of 16 query columns).  Both read no b; every output row's last
word is written whole (P = 0 past `size`).

**DIV and the HLS kernel's C simulation.**  The RTL matches the synthesised
HLS kernel, whose divider (`VectorOPKernel_sdiv_25s_16s_25`) is 25 bits
wide.  HLS C simulation divides in 24 bits instead, so for a = 0x8000 (−128)
and b = 0xFFFF (−1/256) — quotient +2²³ — C simulation wraps and saturates
to 0x8000, while the hardware (and the RTL) gives 0x7FFF.  The testbench's
oracle (the HLS C++) is corrected for exactly that input pair; the
checked-in fixtures never divide by a negative number.

**Geometry** — VectorOP.cpp's, word for word (8 lanes per 16-byte word):
`n_words = ceil(size / 8)`; per input, a stride-0 operand (`outer > 1`,
`inc == 0`, `n_words ≤ 256`) is read once and replayed; `outer == 1` or
`inc == size` with `size % 8 == 0` is one contiguous range of
`outer · n_words` words (the 32-bit product); otherwise `outer` runs of
`n_words` words, run `o` at word `o · ⌊inc / 8⌋`.  The output uses
`c_inc = a_inc + b_inc` (no replay); the softmax ops `c_inc = b_inc`, and
column mode reads `outer / 16` blocks of `size` two-word runs (block `k` at
word `2k`, run `r` at `r · ⌊a_inc / 8⌋` — `vo_burstgen`'s block loop) and
writes `outer & ~15` runs of `n_words`.  The lanes of a run's last word past
`size` read as 0, so every output run's last word is written whole with 0 in
its tail lanes.  `size == 0` or `outer == 0`: nothing is read or written.
Bases are 16-byte aligned (the low four address bits are ignored).

## 2. Architecture

```
ctrl ─► config (7 cycles) ─┬► rd_port a: burstgen ─► AR ─► R FIFO (512) ─► tail mask ─┬─► out FIFO ─┐
                           │                                              └─► replay RAM (256) ─┘    │
                           ├► rd_port b (gmem1; disabled for ops >= 4)                               ├► compute ─► act ─► wr_port c
                           └──────────────────────────────────────────────────────────────────────────┘
```

| module | role |
|---|---|
| `vo_ctrl_s_axi` | the AXI-Lite slave (derived from the RTL MatmulKernel's); its inputs are registered before they are decoded (a write takes effect with BVALID, a read is answered one cycle later) |
| `vo_core` | job FSM: latch the registers, derive the three port geometries (`outer · n_words` as three 16 × 16 partial products, four registered stages), start the units, `ap_done` when all are idle; the latch and the geometry loads are enabled by registered strobes, and the job reset is registered and copied per unit |
| `vo_burstgen` | one port's bursts: one per cycle, ≤ MAXB words, never across 4 KiB or a run |
| `vo_rd_port` | AR issued only when the 512-beat read FIFO can take the whole burst (RREADY tied high); tail mask (its run-end flag and lane mask kept in registers); the replay RAM; a small output FIFO |
| `vo_rs` | a register slice (two entries, both directions registered) on each AR, AW and W output |
| `vo_compute` | 8 lanes per cycle in a 6-stage pipeline (add / sub, a DSP multiply per lane, select + saturate, ReLU / ReLU6); DIV through `vo_div`; both paths then through `vo_act`; a 32-word output FIFO, entered only with room for every word in flight (one counter from the take to the FIFO); the job's activation (`job_act`) decoded in two registered steps |
| `vo_act` | the activations 3–6 on the merged 8-lane words, a fixed 7-cycle pipeline (other jobs pass through): per lane `|x|`, the ROM address and past-the-table flag, the ROM read and its output register, `max(x, 0) − m`; LeakyReLU in one DSP48E2 per lane (A / B registers, M, P; the 33-bit product rounded in fabric) |
| `vo_act_rom` | generated: the 4 096 × 7 table, two read ports with the block RAM's output register — one BRAM36 (4K × 9) per two lanes |
| `vo_div` | restoring radix-2 divider on magnitudes, one lane per cycle, 26 stages |
| `vo_smx` | the softmax (ops 10 / 11), on the a stream and into the write port in place of `vo_compute`: one unit at a time (a row, or 16 query columns), load into an 8-bank buffer (2048 × 16 per bank; column mode skewed so a key row is one write and an output word one read) with the maxima on the way in, then EXP (read, e, the sums), 25-cycle restoring divisions R = ⌊2⁴⁰ / S⌋, OUT (read again, e recomputed, e · R, round, saturate) — EXP and OUT share one 21-stage pipeline (2 DSPs per lane); the 32-word output FIFO with credits as `vo_compute`'s |
| `vo_smx_rom` | generated: the 4 096 × 17 exponential table, two read ports with the output register — 4 instances (8 lanes) |
| `vo_wr_port` | 512-beat FIFO; bursts ≤ 256 words issued on AW once their beats are buffered; ≤ 16 bursts awaiting B; the next bursts wait in a register slice and the issue decision is an AND of registers (`avail`, `av_ok` computed a cycle ahead, `ob_ok`) |

The m_axi ports are registered both ways: AR, AW and W leave through
`vo_rs` slices (ARREADY / AWREADY / WREADY only enable a slice's registers),
and RVALID / RDATA / RLAST and BVALID are registered before they are used —
so the block design's interconnect never sees a combinational path into the
kernel's logic or out of it.

Burst sizes and outstanding counts follow the HLS interface: reads ≤ 64
beats with ≤ 16 bursts awaiting data per port, writes ≤ 256 beats with ≤ 16
awaiting B.  The IP declares them on its m_axi interfaces as the HLS export
does (`NUM_READ_OUTSTANDING` / `NUM_WRITE_OUTSTANDING` 16,
`MAX_READ_BURST_LENGTH` / `MAX_WRITE_BURST_LENGTH`, `READ_WRITE_MODE`
read-only for gmem0/1 and write-only for gmem2; `syn/package_ip.tcl`): the
block design sizes each crossbar slot's acceptance from these.  Without them
(the first package; the RTL MatmulKernel's IP until MATMUL_RTL_PLAN phase 5) a port counts as
read-write with 2 outstanding bursts, and the crossbar held the reads of a
job of nine 2-word runs to about four in flight — slower than the HLS
kernel in the test stand.

Resources (out of context, xck26 −2LV, `make synth_vectorop_rtl`), with the
softmax unit (2026-10-07): 11 672 LUT (491 LUTRAM), 13 751 FF, 30 BRAM36, 35
DSP; timing met at 300 MHz (WNS +0.175 ns, Fmax ≈ 317 MHz; the worst path the
argument latch's enable, routing only).  Before it: 6 173
LUT (345 LUTRAM), 8 640 FF, 14 BRAM36, 19 DSP; timing met at 300 MHz (WNS
+0.277 ns, Fmax ≈ 327 MHz; the block design runs the kernels at 250 MHz
since FMAX_250_PLAN).  The worst paths are the read ports' replay-RAM inputs
and the AW slice's enables (one LUT, ~92 % route), none in `vo_act`.
Before the activation unit (2026-10-06, the same script on main
`c5c5d83`): 5 418 LUT (270 LUTRAM), 7 431 FF, 10 BRAM36, 11 DSP, WNS
+0.453 ns (+0.191 in an earlier run of the same sources: placement varies
by a few tenths).  Before the registered AXI ports and job reset
(2026-10-06): 5 243 LUT, 6 453 FF, WNS +0.248 ns.  The
HLS kernel: about 22.7k LUT, 13.7k FF, 32 BRAM18 and 33 DSP (its synthesis
estimate).

In the block design (the routed 100 MHz `cormorant_hw_128`, 2026-10-05) the
kernel's long paths were not inside its arithmetic: the job reset
(`rst || job_start`, combinational, 285 loads, plus `rst` with 417) spanning
five clock regions, the geometry-register enable (504 loads), and the AXI
boundary — the crossbar's WREADY through the W FIFO's skid logic (5 levels,
5.7 ns), RDATA into the read FIFOs, `araddr` into the crossbar's arbiter.
The registered resets, strobes and AXI slices above address them; cycle
counts move by at most 12 cycles per job (`perf_vectorop_rtl`).  At 4 ns
(the 250 MHz trial of 2026-10-06) the one logic-bound path of the whole
design was this kernel's geometry product `outer · n_words`, a 32 × 32
multiply in two cascaded DSPs and a carry chain (10 levels, 3.4 ns): it is
now three 16 × 16 partial products with every stage registered (one more
configuration cycle per job).  In the full 250 MHz design (four kernels,
WNS +0.074 ns) the last cluster below 0.1 ns was the write port's issue
decision: the FIFO count through `wf_count − pend_w ≥ len` (9 levels) into
the enables of the burst generator's registers.  The bursts now wait in a
register slice and the decision reads registers only.

## 3. Performance

Cycles from `ap_start` to `ap_done` with ideal memory (`make
perf_vectorop_rtl`, Verilator; the registered AXI ports and job start of
2026-10-06 added 2–12 cycles per job, the activation unit's 7 stages another
4–11):

| job | cycles | words / cycle |
|---|---:|---:|
| ADD / MUL / RELU, 65 536 elements | 8 514 | 0.96 |
| GELU (op 8); ADD + act SILU, 65 536 elements | 8 516; 8 518 | 0.96 |
| DIV, 8 192 elements | 8 362 | 0.12 (one lane per cycle, as HLS) |
| bias add, 64 × 1 024 (b replayed) | 8 518 | 0.96 |
| 12-element runs, stride 16, × 1 000 | 2 071 | 0.97 |
| 8-element rows × 4 096 (one-word b replayed) | 4 422 | 0.93 |
| 5 000-element rows × 16 (b over the replay bound) | 10 323 | 0.97 |

An activation costs no throughput: SiLU / GELU / LeakyReLU run at the
memory-bound rate of any unary op, as a separate op or fused after one.

The softmax (`vo_smx`) runs its phases one after the other per unit — load,
EXP, the divisions, OUT — at about a third of that rate (ideal memory):

| job | cycles | board (`run_remote_perf`, 250 MHz) |
|---|---:|---:|
| row mode, rows of n | 75 + rows · (58 + 3 ⌈n / 8⌉) | |
| BERT head: 256 rows of 256 | 39 056 (0.21 words / cycle) | 0.160 ms |
| column mode, K keys | 60 + (outer / 16) · (455 + 6 K) | |
| SmolVLM head: 1024 keys × 1024 queries | 6 600 per 16 queries | 3.77 ms |
| SmolLM2 prefill-256 head: 256 keys × 256 queries (row stride 768) | 31 947 | 0.182 ms |

Column mode reads a two-beat run per key row: on the board it reaches about
half the ideal-memory rate (SmolVLM's head 3.77 ms against 1.69).

On the board (bitstream `68665fc1833a`, the 15 VectorOP benchmarks of
`run_remote_perf.py` against the HLS kernel's bitstream back to back): no
case slower; the large binary jobs are equal (bound by the shared HPC0 read
channel), the unary, DIV, broadcast and short jobs 0.2–3.4 % faster.  In the
bitstream the kernel takes 5 160 LUT, 6 469 FF, 10 BRAM36 and 11 DSP; the
design as a whole 3.6k LUT, 4.8k FF, 4 BRAM tiles and 22 DSP less than with
the HLS kernel (VECTOROP_RTL_PLAN phases 1–2).

## 4. Verification

| check | where | result |
|---|---|---|
| Verilator lint (`-Wall`) | `lint_vectorop_rtl` | clean |
| the 212 checked-in fixtures (`hw/test_data/vecop_test_data`; 82 activation cases since 2026-10-06, 11 softmax cases since 2026-10-07) | `TestVectorOpRtl` (ctest) | 212 / 212: c.hex in every run, tail lanes 0, and byte-identical to the HLS oracle on the whole output region |
| the softmax: every Q8.8 input through a 2048-element row at two scales (ctest `--case … 3`); ~40 random softmax jobs per 300 (row and column mode, masks, any register value); 2 700 more by hand (seeds 1–3 × rand / slow / fast) and 1 200 after the masked-lane fix (seeds 5–6) | `TestVectorOpRtl`, `Vtb --random` | all bit-exact against the C++ model (`smx_vector`) |
| the exponential table | ctest `VectorOpRtlSmxRom`; `gen_smx_rom.py --check` | current; equal to `vectorop_smx.TAB` |
| every Q8.8 input of each activation (`a` = 0…65 535: LeakyReLU at slopes 0.01 and 0.5, SiLU, GELU, GELU tanh; ADD + SiLU, ADD + GELU tanh, DIV + GELU) | ctest `TestVectorOpRtl` (`--case … 3`) | bit-exact against the C++ model |
| the generated table | ctest `VectorOpRtlActRom`; `gen_act_rom.py --exact` | current; all 3 786 entries equal 40-digit mpmath |
| random jobs against the HLS kernel's C++, randomised AXI timing, protocol checks, no stray write, no gmem1 traffic for unary ops | ctest: 300 (seed 1; since 2026-10-06 ops 0–12, acts 0–9, random `alpha`); by hand: 2 × 500 (seeds 1, 2026) + 200 slow timing (seed 4711); after the registered ports and resets (2026-10-06): 9 × 700 (seeds 2, 2026, 4711 × random / fast / slow timing), repeated after the write-issue change | all bit-exact |
| register table vs RTL (and, until it was retired, the HLS driver) | `VectorOpRtlDriver` (ctest), `gen_driver.py --check` | 13 arguments (`alpha` since 2026-10-06, the three softmax registers since 2026-10-07) + 4 control registers agree |
| the synthesised netlist against the RTL, cycle by cycle in xsim | `neteq_vectorop_rtl` | 30 jobs over ops 0–9 and acts 0–6 (2026-10-06): 0 mismatches; with 12 softmax jobs (2026-10-07): 0 mismatches after the masked-lane fix (§6) |
| the full `cormorant_hw_128` block design through the PS VIP | `sim_hw_kv260` | 75 / 75 (VectorOPKernel 29: five activation cases, two softmax cases; 2026-10-07) |
| the test stand's VectorOP block design in xsim with the RTL IP (upgraded in place of the HLS IP) | `sysim_vectorop_rtl`; `behavior_test_vectorop` | 212 / 212 with the softmax unit (2026-10-07, 33 min on a loaded host); 201 / 201 with the activation unit (2026-10-06, 4 min 35 s).  Phase 1 (119 cases): 984 µs of kernel time against the HLS kernel's 1 000 µs, faster in 115 of the 119 cases, at most 2.4 % slower in the other four (1 000 × 16-element jobs with a replayed operand) |

On the board (VECTOROP_RTL_PLAN phase 2): registers written and read back,
DIV of every int16 value by −1/256, the 148 models of `run_remote_tests`, the
MNIST / image-classification / BERT demos and the SmolLM2-135M / 360M,
SmolVLM and Piper library gates — all bit-exact.  With the activation unit
(bitstream `6436623029f7`, ACTIVATIONS_PLAN §6): `alpha` written and read
back, 156 / 156 models (the eight `act_*` models among them, four over every
Q8.8 input), the demos (BERT's GELUs on the unit: 541 → 525 ms) and the four
library gates — all bit-exact.  With the softmax unit (bitstream
`588d721997cb`, SOFTMAX_PLAN §4.2b): the three registers written and read
back, 159 / 159 models (the four tiny BERTs and three `smx_*` models on the
unit), the 63 benchmarks, the demos (BERT's Softmaxes on the unit: 525 →
427 ms) and the five library gates (SmolLM2-135M / 360M and SmolVLM with
their softmaxes on the unit, BERT, Piper) — all bit-exact.

The random jobs cover every op (also codes 10–12) and act (also 7–9), sizes
up to 6 000, broadcasts in both directions, both-advancing strides, the
replay bound (253–259 words) and jobs outside the contract (increments that
are not multiples of 8, `size` or `outer` 0).  The AXI models also fail a
port with more than 16 bursts outstanding (the declared counts; the random
jobs reach 16).

## 5. Build targets

`TestVectorOpRtl`, `lint_vectorop_rtl`, `perf_vectorop_rtl`,
`vectorop_rtl_tb_fst`, `driver_vectorop_rtl` (`build/kernels/vectorop_rtl/driver/VectorOPKernel_v1_0/src`),
`package_vectorop_rtl` (`build/rtl_ip/VectorOPKernel_ip`), `synth_vectorop_rtl`,
`xsim_vectorop_rtl`, `neteq_vectorop_rtl` (the synthesised netlist against the RTL in
lockstep, `tools/neteq`), `sysim_vectorop_rtl` — `kernels/vectorop_rtl/CMakeLists.txt`.

## 6. Invariants for changes

- The read FIFO never fills: every AR reserves its beats (`space`), so
  RREADY can stay high.  Keep the reservation if the FIFO depth or burst
  size changes.
- At most `RD_OUTS` / `WR_OUTS` bursts are outstanding per port — the counts
  `package_ip.tcl` declares; change both together.
- `vo_burstgen` keeps the full-width adds off the path after `len` (the run
  end is decided beside it; the address and remainder add `len` to their low
  bits only, and the high parts' carry and borrow are decided from the
  registers, not from those sums).  The plain form missed 300 MHz by
  0.29 ns, the one with the carry / borrow taken from the sums by 0.04 ns.
- The compute output FIFO is only entered with room for every word in
  flight: `inflight` counts the words taken (ALU or DIV) and not yet pushed
  into the FIFO, so `vo_act` (or any further stage) is covered; the FIFO
  (32 words) must stay deeper than the pipeline from the take to the FIFO
  (5 + 7 stages) for a full word per cycle.
- `vo_act` decodes its job constants (mode, ROM segment, `alpha`) two
  register stages after `j_op` / `j_act` / `j_alpha`; the first word of a
  job reaches it many cycles later (AR / R, the FIFOs, the ALU), so the
  constants are always settled.  Keep it so if the job start is shortened.
- `rtl/vo_act_rom.sv` is generated: change `scripts/gen_act_rom.py` and
  re-run it (ctest `VectorOpRtlActRom` fails on a stale file).  A table
  function must be odd-symmetric in the `x · F(x)`, `F(−x) = 1 − F(x)`
  sense for `max(x, 0) + f(−|x|)` to hold (the generator checks the identity
  on every input); and the C++ model (`act_fn`) must give the same values —
  the exhaustive ctest cases compare the two on all 65 536 inputs.
- An AW is issued only when all of its beats are buffered and not promised
  to an earlier burst, so W never stalls mid-burst.  `avail` (= the FIFO's
  count minus the beats promised to issued bursts) is a register updated by
  pushes and issues — a W pop lowers both terms — and `av_ok` / `ob_ok` are
  computed one cycle ahead for the next cycle's head burst, ignoring that
  cycle's push and B: they may only be late, never early.
- The job reset (`urst_*`, one registered copy per unit) is asserted the
  cycle after `job_start`, and the units' `start` (`start_q2`) the cycle
  after that; the geometry registers are loaded (T_CFG5) before either.
- `vo_smx` handles one unit at a time and its buffer holds one unit: the next
  unit's load starts only after OUT has issued every read, and only once the
  issued words are past the stage that reads the maxima and valid lengths
  (`S_NEXT` waits for r0–r2 to be empty) — the next unit's V / LOAD
  overwrite them; R is written by the next unit's DIV, which follows its EXP
  (in order behind every OUT word).  e is recomputed in OUT from the
  buffer, so EXP and OUT must use the same pipeline and constants.
- `vo_smx`'s masked lanes must produce a known 0, not only a 0 on hardware:
  in column mode the last output word of a query reads buffer words this unit
  never wrote (keys past `size`), whose X in a 4-state simulation reached P
  through `e = 0 >> sh` (sh from the unknown input) until `sh13` was zeroed on
  masked lanes too — `neteq_vectorop_rtl` caught it (RTL X, netlist 0 in the
  tail lanes); Verilator, being 2-state, cannot.
- `vo_smx`'s counts are 32 bits wide and its buffer addresses wrap: a job
  outside the limits (rows > 2048, keys > 1024) gives wrong P but always
  consumes and produces its words.
- `rtl/vo_smx_rom.sv` is generated (`scripts/gen_smx_rom.py`; ctest
  `VectorOpRtlSmxRom`); the table must stay `vectorop_smx.TAB`, which the C++
  model (`smx_table`) and the scheduler share.
- The three word streams (a, b, c) have the same length by construction
  (`outer · n_words`); the geometry must stay identical to VectorOP.cpp's
  for the RTL to read and write the same words.
