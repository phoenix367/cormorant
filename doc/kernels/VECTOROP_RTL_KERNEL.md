# VectorOPKernel in SystemVerilog

**Status (2026-10-05):** the hardware build's VectorOPKernel since
[VECTOROP_RTL_PLAN](../plans/VECTOROP_RTL_PLAN.md) phase 3 (bitstream
`68665fc1833a`); the Vitis HLS kernel's synthesis is retired, its C++
([VECTOROP_KERNEL](VECTOROP_KERNEL.md)) is the reference model and fixture
generator.

A drop-in replacement for the HLS kernel: the same IP (VLNV
`xilinx.com:hls:VectorOPKernel:1.0`, 155 ports and 35 parameters with the
HLS export's names and defaults), the same AXI-Lite register map, the same
DDR access pattern and bit-identical results.  All three m_axi ports are
128 bits wide, as on the HLS kernel.

## 1. Contract

**Registers** (`s_axi_ctrl`, 7 address bits): `0x00` ap_ctrl (b0 ap_start,
b1 ap_done clear-on-read, b2 ap_idle, b3 ap_ready clear-on-read, b7
auto_restart, b9 interrupt), `0x04` GIE, `0x08` IER, `0x0C` ISR (toggle on
write), `0x10/14` a, `0x1C/20` b, `0x28/2C` c, `0x34` size, `0x3C` op,
`0x44` outer, `0x4C` a_inc, `0x54` b_inc, `0x5C` act.  `interrupt` is level
high while GIE and a set ISR bit.

**Arithmetic** per Q8.8 lane (raw int16):

| op | result |
|---|---|
| 0 ADD, 1 SUB | `sat16(a ± b)` |
| 2 MUL | `sat16((a · b) >>> 8)` — floor, then saturate |
| 3 DIV | `b == 0 ? 0 : sat16(trunc((a << 8) / b))` — the quotient truncated toward zero |
| 4 RELU | `max(a, 0)` |
| 5 RELU6 | `min(max(a, 0), 0x0600)` |
| ≥ 6 | `a` |

then `act` on the result: 1 RELU, 2 RELU6, anything else none.  Ops ≥ 4 read
no b (nothing is issued on gmem1).

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
`c_inc = a_inc + b_inc` (no replay).  The lanes of a run's last word past
`size` read as 0, so every output run's last word is written whole with 0 in
its tail lanes.  `size == 0` or `outer == 0`: nothing is read or written.
Bases are 16-byte aligned (the low four address bits are ignored).

## 2. Architecture

```
ctrl ─► config (7 cycles) ─┬► rd_port a: burstgen ─► AR ─► R FIFO (512) ─► tail mask ─┬─► out FIFO ─┐
                           │                                              └─► replay RAM (256) ─┘    │
                           ├► rd_port b (gmem1; disabled for ops >= 4)                               ├► compute ─► wr_port c
                           └──────────────────────────────────────────────────────────────────────────┘
```

| module | role |
|---|---|
| `vo_ctrl_s_axi` | the AXI-Lite slave (derived from the RTL MatmulKernel's); its inputs are registered before they are decoded (a write takes effect with BVALID, a read is answered one cycle later) |
| `vo_core` | job FSM: latch the registers, derive the three port geometries (`outer · n_words` as three 16 × 16 partial products, four registered stages), start the units, `ap_done` when all are idle; the latch and the geometry loads are enabled by registered strobes, and the job reset is registered and copied per unit |
| `vo_burstgen` | one port's bursts: one per cycle, ≤ MAXB words, never across 4 KiB or a run |
| `vo_rd_port` | AR issued only when the 512-beat read FIFO can take the whole burst (RREADY tied high); tail mask (its run-end flag and lane mask kept in registers); the replay RAM; a small output FIFO |
| `vo_rs` | a register slice (two entries, both directions registered) on each AR, AW and W output |
| `vo_compute` | 8 lanes per cycle in a 6-stage pipeline (add / sub, a DSP multiply per lane, select + saturate, activation); DIV through `vo_div`; a 16-word output FIFO, entered only with room for every word in flight |
| `vo_div` | restoring radix-2 divider on magnitudes, one lane per cycle, 26 stages |
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

Resources (out of context, xck26 −2LV, `make synth_vectorop_rtl`): 5 412
LUT (270 LUTRAM), 7 429 FF, 10 BRAM36, 11 DSP; timing met at 300 MHz (WNS
+0.191 ns, Fmax ≈ 318 MHz; the block design runs the kernels at 250 MHz since FMAX_250_PLAN).  Before the registered AXI
ports and job reset (2026-10-06): 5 243 LUT, 6 453 FF, WNS +0.248 ns.  The
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
2026-10-06 added 2–12 cycles per job):

| job | cycles | words / cycle |
|---|---:|---:|
| ADD / MUL / RELU, 65 536 elements | 8 507–8 510 | 0.96 |
| DIV, 8 192 elements | 8 356 | 0.12 (one lane per cycle, as HLS) |
| bias add, 64 × 1 024 (b replayed) | 8 507 | 0.96 |
| 12-element runs, stride 16, × 1 000 | 2 064 | 0.97 |
| 8-element rows × 4 096 (one-word b replayed) | 4 416 | 0.93 |
| 5 000-element rows × 16 (b over the replay bound) | 10 319 | 0.97 |

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
| the 119 checked-in fixtures (`hw/test_data/vecop_test_data`) | `TestVectorOpRtl` (ctest) | 119 / 119: c.hex in every run, tail lanes 0, and byte-identical to the HLS oracle on the whole output region |
| random jobs against the HLS kernel's C++, randomised AXI timing, protocol checks, no stray write, no gmem1 traffic for unary ops | ctest: 300 (seed 1); by hand: 2 × 500 (seeds 1, 2026) + 200 slow timing (seed 4711); after the registered ports and resets (2026-10-06): 9 × 700 (seeds 2, 2026, 4711 × random / fast / slow timing), repeated after the write-issue change | all bit-exact |
| register table vs RTL and the HLS driver | `VectorOpRtlDriver` (ctest), `gen_driver.py --check --hls-driver` | 9 arguments + 4 control registers agree |
| the test stand's VectorOP block design in xsim with the RTL IP (upgraded in place of the HLS IP) | `sysim_vectorop_rtl`; `behavior_test_vectorop` | 119 / 119 (`check_test_report.py` PASS); 984 µs of kernel time against the HLS kernel's 1 000 µs, faster in 115 of the 119 cases, at most 2.4 % slower in the other four (1 000 × 16-element jobs with a replayed operand) |

On the board (VECTOROP_RTL_PLAN phase 2): registers written and read back,
DIV of every int16 value by −1/256, the 148 models of `run_remote_tests`, the
MNIST / image-classification / BERT demos and the SmolLM2-135M / 360M,
SmolVLM and Piper library gates — all bit-exact.

The random jobs cover every op (also codes 6–9) and act (also 3–5), sizes
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
  flight (`alu_inflight`, `dw_inflight`); a deeper pipeline must count its
  stages.
- An AW is issued only when all of its beats are buffered and not promised
  to an earlier burst, so W never stalls mid-burst.  `avail` (= the FIFO's
  count minus the beats promised to issued bursts) is a register updated by
  pushes and issues — a W pop lowers both terms — and `av_ok` / `ob_ok` are
  computed one cycle ahead for the next cycle's head burst, ignoring that
  cycle's push and B: they may only be late, never early.
- The job reset (`urst_*`, one registered copy per unit) is asserted the
  cycle after `job_start`, and the units' `start` (`start_q2`) the cycle
  after that; the geometry registers are loaded (T_CFG5) before either.
- The three word streams (a, b, c) have the same length by construction
  (`outer · n_words`); the geometry must stay identical to VectorOP.cpp's
  for the RTL to read and write the same words.
