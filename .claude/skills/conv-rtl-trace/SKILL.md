---
description: Trace ONE ConvKernel case — inside the SystemVerilog kernel with the Verilator testbench's waveforms (sweep issue, weight-cache fills, accumulator write-back, drain, every unit's state), or at its AXI ports on the test stand with the testbench's +VERBOSE probes (burst lengths, AW→AW and W-beat spacing, bursts in flight, read-request spacing). Use when a conv result or timing is unexplained, a layer is slower than the cycle model predicts, or a kernel hangs on the board; run on a scaled-down case, never the whole suite.
allowed-tools: Bash Read
---

# conv-rtl-trace

ConvKernel is the SystemVerilog kernel in `kernels/conv_rtl`
(CONV_RTL_KERNEL.md; its C++ reference model is `kernels/conv`).  Two
views of one case:

- **Inside the kernel** — the Verilator testbench with waveforms (§1):
  every signal, any job size, seconds per run.  How the RTL's bugs were
  found in CONV_RTL_PLAN phase 0.
- **At the AXI ports, in the real block design** — the test stand's conv
  testbench with its `+VERBOSE` probes (§2): the PS VIP, interconnect and
  DDR model, xsim with the DSP48E2 models (~20 µs of simulated time per
  wall second: keep the case small).

## 1. Inside the kernel: Verilator waveforms

```bash
cd build                                        # the verification tree
make conv_rtl_tb_fst                            # vlt/Vtb: the testbench with --trace
T=/mnt/data/cormorant_repro/tmp/cvtrace         # a roomy disk: a 30 k-cycle job is ~1 GB as VCD
kernels/conv_rtl/vlt/Vtb --case "N IC H W OC OH OW KH KW SH SW DH DW PT PL BIAS DW" \
    --timing fast --max-cycles 300000 --trace $T/c.fst      # or --fixtures DIR --only I
fst2vcd $T/c.fst > $T/c.vcd
Q=../.claude/skills/conv-rtl-trace/scripts/vcdq.py
python3 $Q $T/c.vcd --list 'u_en\.[a-z_]*$'                 # the engine's signals
python3 $Q $T/c.vcd 'u_en\.(s_act|eq_valid|eq_ready|slab_in|bias_in|buf_in|haz_ok)$' --from 150 --to 400
```

- A failing random job prints its `--case "…"` line (the data differ —
  the testbench re-draws them — but a structural bug reproduces); a
  timeout prints it and stops.  `--only I,J` runs fixtures by position.
- `vcdq.py` prints a row per dump time at which a selected signal changed:
  the cycle (`+` = after the rising edge), then each signal in hex.
  Hierarchy: `ConvKernel.u_core.` then `u_xl` (x loader), `u_pa` (patch
  producer), `u_wl` (weight loader), `u_bi` (bias), `u_en` (engine:
  fill `f_*`, issue `s_*`, write-back `wb_*`, `hold_*`), `u_dr` (drain:
  fill `f_*`, transposer `t_*`, emitter `e_*`), `u_yw` (y writer), the
  sweep sequencer `qs`, `k`, `s_*` and the job FSM `tstate` in `u_core`.
- **Where a job stops:** the units' `idle` and the engine's `eq_ready`
  terms (`slab_in`, `bias_in`, `buf_in`, `haz_ok`) name the unit it waits
  for.  **Where a value goes wrong:** follow it — patch beats `pt_data`,
  the write-back `wb_data` / `wb_addr` (one 16-lane accumulator word per
  pixel), the drain's `d_rdata` → `sat_lo` → the emitter's `yw_data`.
- Rebuild `vlt/Vtb` after an RTL edit (`make conv_rtl_tb_fst`; `make
  TestConvRtl` builds only `vl/Vtb`).

## 2. At the AXI ports: the test stand's verbose probes

```bash
S=.claude/skills/conv-rtl-trace/scripts
make -C build gen_conv_test_data                  # refresh dumps
$S/make_fixture.sh build 21 /tmp/trace_fixture    # one case → dir
make -C build package_conv_rtl                    # the IP under build/rtl_ip/ConvKernel_ip
$S/run_trace.sh build /tmp/trace_fixture /tmp/trace.log
python3 $S/analyze_trace.py /tmp/trace.log
```

`make_fixture.sh <build-dir> <test-index> <out-dir>` copies that case's
four hex files (`TestConvRef --dump-data`) and a 2-line-header manifest;
add a named case to `kernels/conv/test/TestConvSim.cpp` first if none
fits (keep it SMALL).  `run_trace.sh` runs `make -C
hw/cormorant_test_stand tb-conv` with `TS_VERBOSE=1` against
`<build-dir>/rtl_ip/ConvKernel_ip` — the last packaged IP; re-package
after a source change.  The report goes next to the log.

`analyze_trace.py` prints, per test in the log:

- **gmem3 (y)**: write bursts (LEN histogram), AW→AW spacing, W-beat
  spacing split into intra-burst vs inter-burst gaps, bursts in flight at
  each AW, AW→B latency, gaps > 1 µs.
- **gmem0 (x)**: AR bursts and AR→AR spacing.
- The acc_stream and per-process sections are the HLS kernel's
  (`CONV_TB_HLS_PROBES`, the retired IP's dataflow internals): empty for
  the RTL IP.

## Reading the numbers

Compare against CONV_RTL_KERNEL §3: one grid instant per cycle, a pixel
pair costing `G · max(kh·kw, 2)` instants per m-group and input tile; the
next sweep's weight slab fills beside the current sweep (two 128-bit
beats per 16-lane vector), the drain of chunk n beside chunk n + 1 (8
outputs per cycle), only the last chunk's drain after the last sweep.
`kernels/conv_rtl/vl/Vtb --case "…" --timing fast` gives the cycle count
with ideal memory; the gap to the board time is the DDR.

## Board hangs

If the kernel hung ON THE BOARD, first read its registers
(`board-deploy` skill, `read_kernel_regs.sh`) to get the layer geometry,
then run that geometry in the Verilator testbench (§1, `--case`, random
and slow timing): a hang there prints `TIMEOUT … (--case "…")` and the
units' `idle` signals show which one waits.  If it passes there, scale it
down to a test-stand fixture that keeps the suspect structure (run
length, chunk count, tile count) and trace §2: the AXI VIP's
forward-progress watchdog turns a deadlock into a `$finish` with "Pending
AW ... no progress" and no report.
