---
description: Verify a MatmulKernel, VectorOPKernel or PoolingKernel change end-to-end on the KV260 platform (argument `matmul`, `vectorop` or `pool`) — all three are SystemVerilog kernels (kernels/matmul_rtl, kernels/vectorop_rtl, kernels/pool_rtl) with a C++ reference model (kernels/matmul, kernels/vectorop, kernels/pool): C-simulation, the Verilator testbench and lint, Vivado out-of-context synthesis (timing at 300 MHz, resources), the test-stand behaviour test, then a per-case timing diff against the previous run; plus the project rules for optimisations (baseline first, revert regressions, log the result) and for interface changes (block-design testbench + sim_hw_kv260). Use after any edit to kernels/matmul_rtl/, kernels/vectorop_rtl/, kernels/pool_rtl/, kernels/matmul/, kernels/vectorop/ or kernels/pool/ (the C++ reference models and fixture generators), their CMakeLists or RTL fixtures, or the platform JSON's kernels.matmul / kernels.pool block.
allowed-tools: Bash Read
---

# kernel-verify `<matmul|vectorop|pool>`

Four sequential gates (plus 1b and 5 when they apply).  **If any gate
fails, stop immediately and report the failure** — do not run later gates
on a broken build.  `K` below is the argument: `matmul`, `vectorop` or
`pool`.  ConvKernel, the one HLS kernel left, has its own skill
(`conv-verify`).

| | `matmul` | `vectorop` | `pool` |
|---|---|---|---|
| C-sim targets | `TestMatmulRef`, `TestMatmulBlas` (the C++ reference model; Blas only when configure found a BLAS), `TestMatmulRtl` (Verilator), `lint_matmul_rtl` | `TestSimulation` (the C++ reference model), `TestVectorOpRtl` (Verilator), `lint_vectorop_rtl` | `TestPoolingSim` (the C++ reference model), `TestPoolRtl` (Verilator), `lint_pool_rtl` |
| ctest filter | `ctest -R Matmul` (TestMatmulRef, TestMatmulBlas, TestMatmulRtl, MatmulRtlDriver) | `ctest -R 'TestSimulation\|VectorOp'` (TestSimulation, TestVectorOpRtl, VectorOpRtlDriver) | `ctest -R Pool` (TestPoolingSim, TestPoolRtl, PoolRtlDriver) |
| synthesis | `synth_matmul_rtl` (Vivado out of context at 300 MHz; the IP for the hardware is `package_matmul_rtl`) | `synth_vectorop_rtl` (likewise; the IP is `package_vectorop_rtl`) | `synth_pool_rtl` (likewise; the IP is `package_pool_rtl`) |
| synthesis report | `$BUILD_DIR/kernels/matmul_rtl/synth/` (`timing.rpt`, `utilization.rpt`, the make output's `RESULT` line) | `$BUILD_DIR/kernels/vectorop_rtl/synth/` (the same files) | `$BUILD_DIR/kernels/pool_rtl/synth/` (the same files) |
| RTL behaviour test | `behavior_test_matmul` | `behavior_test_vectorop` | `behavior_test_pool` |
| report (under `$BUILD_DIR/kernels/K/kv260/`) | `matmul_op_test_report.json` | `vector_op_test_report.json` | `pooling_test_report.json` |
| checked-in RTL fixtures | `hw/test_data/matmul_test_data/` (50 cases, 11 GEMV) | `hw/test_data/vecop_test_data/` (119 cases) | `hw/test_data/pool_test_data/` (45 cases) |
| fixture target → output | `gen_matmul_test_data` → `$BUILD_DIR/matmul_test_data/` | `gen_vectorop_test_data` → `$BUILD_DIR/vectorop_test_data/` | `gen_pool_test_data` → `$BUILD_DIR/pool_test_data/` |
| test-stand testbench | `hw/cormorant_test_stand/kernels/matmul_op_test/matmul_op_test.srcs/sim_1/new/matmul_tb.sv` | `hw/cormorant_test_stand/kernels/vector_op_test/vector_op_test.srcs/sim_1/new/vectorop_tb.sv` | `hw/cormorant_test_stand/kernels/pooling_test/pooling_test.srcs/sim_1/new/pooling_tb.sv` |
| block-design testbench (`hw/cormorant_hw_128/cormorant_hw_128.srcs/sim_1/new/`) | `mm_regmap.svh`, `mm_classes.svh`, `tb_functions.svh` | `vop_regmap.svh`, `vop_classes.svh`, `tb_functions.svh` | `pk_regmap.svh`, `pk_classes.svh`, `tb_functions.svh` |
| optimisation log | `doc/kernels/MATMUL_RTL_KERNEL.md` (Performance / Resources) and `doc/plans/MATMUL_RTL_PLAN.md` (`MATMUL_OPTIMISATION.md` is the retired HLS kernel's log) | `doc/kernels/VECTOROP_RTL_KERNEL.md` and `doc/plans/VECTOROP_RTL_PLAN.md` (`VECTOROP_OPTIMISATION.md` is the retired HLS kernel's log) | `doc/kernels/POOL_RTL_KERNEL.md` and `doc/plans/POOL_RTL_PLAN.md` (`POOL_OPTIMISATION.md` is the retired HLS kernel's log) |

## Locate the build directory and set up the shell

The user may have placed `build/` inside the repo or anywhere else
(out-of-source builds are common).  From the repo root, locate it once:

```bash
# Prefer the conventional in-repo location; fall back to a shallow find.
if [ -f build/CMakeCache.txt ]; then
    BUILD_DIR=$(pwd)/build
else
    BUILD_DIR=$(find . -maxdepth 4 -name CMakeCache.txt -path '*/build/CMakeCache.txt' | head -n 1 | xargs -r dirname)
fi
echo "BUILD_DIR=$BUILD_DIR"
grep -E '^AXI_BUS_WIDTH:|^CMAKE_HOME_DIRECTORY:' "$BUILD_DIR/CMakeCache.txt"
```

If `BUILD_DIR` ends up empty, ask the user where the build tree is and
stop.  `CMAKE_HOME_DIRECTORY` must be this checkout (a build tree of
another clone runs another clone's sources); `AXI_BUS_WIDTH` should be
`128` (the block designs are 128-bit).  Every shell that runs a gate needs
Vitis, and — `/` is nearly full on the maintainer's machine — a temp dir
elsewhere:

```bash
source /mnt/data/xilinx/2025.2/Vitis/settings64.sh   # vitis-run, vivado
export TMPDIR=/mnt/data/cormorant_repro/tmp          # only if / is short of space (df -h /)
K=matmul                                            # or vectorop, pool
S="${CLAUDE_SKILL_DIR}/scripts"
cd "$BUILD_DIR"
```

## Step 0 — baseline BEFORE the change (optimisations)

A speed-up is claimed only against a behaviour-test run of the
unmodified kernel on the same fixtures.  Before editing, check that the
snapshot Gate 4 compares against exists and is from the current HEAD:

```bash
ls -l "kernels/$K/kv260/${K}_timing_last.json" "kernels/$K/kv260/"*_test_report.json
```

If it is missing or older than the last kernel commit, run Gates 2–4 on
the clean tree first.  Then keep that run as a fixed reference — Gate 4's
default baseline moves with every run, and `behavior_test_K` deletes the
old report before it starts, so copy it now:

```bash
cp "kernels/$K/kv260/"*_test_report.json "kernels/$K/kv260/${K}_report_before.json"
```

## Gate 1 — C-simulation

```bash
make TestMatmulRef TestMatmulBlas TestMatmulRtl lint_matmul_rtl   # K=matmul  (drop TestMatmulBlas if make says no rule: no BLAS)
make TestSimulation TestVectorOpRtl lint_vectorop_rtl             # K=vectorop
make TestPoolingSim TestPoolRtl lint_pool_rtl                     # K=pool
ctest -R Matmul --output-on-failure                   # K=matmul (~50 s: TestMatmulRtl)
ctest -R 'TestSimulation|VectorOp' --output-on-failure   # K=vectorop (~2 s)
ctest -R Pool --output-on-failure                     # K=pool (~3 s)
```

- `lint_matmul_rtl` (`verilator --lint-only -Wall` with the waivers) must
  pass with no warning; `TestMatmulRtl` runs the 50 checked-in fixtures and
  200 random cases (seed 1) under randomised memory timing, with AXI
  protocol checks; `MatmulRtlDriver` checks the generated driver's register
  table against `rtl/mm_ctrl_s_axi.sv`.  Edits to `kernels/matmul/` (the C++
  reference model) change the oracle: then Gate 1b decides whether the
  fixtures move.
- `lint_vectorop_rtl` likewise; `TestVectorOpRtl` runs the 119 checked-in
  fixtures and 300 random cases (seed 1) against the HLS C++ under
  randomised memory timing, with AXI protocol checks and at most 16 bursts
  outstanding per port; `VectorOpRtlDriver` checks the driver table against
  `rtl/vo_ctrl_s_axi.sv`.  For more coverage run the testbench by hand:
  `kernels/vectorop_rtl/vl/Vtb --random 500 --seed 2026 --timing rand --quiet`
  (also `--timing slow`, `--perf` for cycle counts with ideal memory).
- `lint_pool_rtl` likewise; `TestPoolRtl` runs the 45 checked-in fixtures
  and 200 random jobs (seed 1) against the HLS C++ on the whole output
  region (every byte; no stray write) under randomised memory timing, with
  AXI protocol checks and the declared outstanding limits (gmem0 16, gmem1
  8); `PoolRtlDriver` checks the driver table against
  `rtl/pl_ctrl_s_axi.sv`.  By hand: `kernels/pool_rtl/vl/Vtb --random 300
  --seed 2 --timing slow --quiet` (a failing case prints its `--case "…"`
  line to re-run alone; `--perf` for cycle counts).  Configure checks that
  the platform JSON's `kernels.pool` bounds equal `pl_pkg`'s constants.

- ctest must end `100% tests passed`.  To see the per-case lines run the
  executable, e.g. `./kernels/matmul/TestMatmulRef | grep -E 'FAIL|passed|PASSED'`:
  `./kernels/matmul/TestMatmulRef` (ends `58/58 tests passed`
  / `TestMatmulSim PASSED`; 19 of the 58 are GEMV cases),
  `./kernels/matmul/TestMatmulBlas` (`24/24 tests passed` /
  `TestMatmulBlas PASSED`), `./kernels/vectorop/TestSimulation`
  (`All 119 tests passed.`), `./kernels/pool/TestPoolingSim`
  (`51 / 51 tests passed.`; `[FAIL]` / `failures=N/M` lines with N > 0 are
  regressions).  Counts grow with the suites — the passed count must equal
  the total.
- **Any `FAIL` line is a regression**, even when the summary looks right.
  A failed `assert` in `TestSimulation` (alignment contract: `a_inc` /
  `b_inc` multiples of 8, tail lanes 0, stride gaps untouched) aborts the
  run — also a regression.

## Gate 1b — RTL fixtures (only when a C-sim case or a DDR layout changed)

The RTL run reads the CHECKED-IN fixtures (`hw/test_data/...`, table
above), not the build tree.  They are the C-sim's `--dump-data` output;
regenerate and copy them only when a case was added / changed or the
buffer layout moved:

```bash
SRC=$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' CMakeCache.txt)
FIX=$SRC/hw/test_data/$([ "$K" = vectorop ] && echo vecop || echo $K)_test_data
make gen_${K}_test_data                   # writes $BUILD_DIR/${K}_test_data/
diff -rq "${K}_test_data" "$FIX" && echo "fixtures unchanged"
# only when the diff is intended:
rm -f "$FIX"/test_*.hex "$FIX"/manifest.txt
cp "${K}_test_data"/* "$FIX"/
```

State as of 2026-09-29: both dumps are identical to the checked-in
fixtures — VectorOP 119 cases, MatMul 50 (the GEMV path's 11 cases, with
the `gemv_kw` manifest column, were added to the RTL suite that day; 50/50
PASS in 893 s).  Pool: 45 cases since 2026-10-05 (POOL_RTL_PLAN phase 0
added the two bank-collision cases as 43–44; the first 43 unchanged).  A dump that differs changes the suite, not just refreshes
it: say so, and expect Gate 4 to list the new cases.
If the manifest gains a column or the DDR layout changes, the test-stand
testbench (table above) must parse / lay it out the same way.  Keep
fixtures small — xsim time grows with every element.

## Gate 2 — synthesis

**MatmulKernel (RTL):**

```bash
make synth_matmul_rtl 2>&1 | grep -E "^RESULT|ERROR"      # ~13-18 min
grep -E "^\| (CLB LUTs|CLB Registers|Block RAM Tile|DSPs) " kernels/matmul_rtl/synth/utilization.rpt
```

- `RESULT period=3.333ns WNS=…` must show WNS ≥ 0: the kernel closes
  timing at 300 MHz out of context (phase 5, 2026-10-05: WNS +0.015 ns,
  18 379 LUT, 9 493 FF, 38 BRAM36, 0 URAM, 130 DSP — MATMUL_RTL_PLAN).  The PL clock is
  100 MHz, so a small miss at 300 MHz does not break the bitstream; report
  it, and a clear loss of Fmax is a finding (phase 2b's first fix lost
  0.26 ns and was redone with precomputed config fields).
- Resources against the record above; > 10 % more is a finding.
- Its register map is checked by ctest `MatmulRtlDriver` and the fact
  registry (`registers.MatmulKernel`); its m_axi bus parameters (outstanding
  counts, burst lengths, read / write only) are declared in
  `kernels/matmul_rtl/syn/package_ip.tcl` and must match `mm_pkg` (fact
  `rtl.axi_masters`).

**VectorOPKernel (RTL):**

```bash
make synth_vectorop_rtl 2>&1 | grep -E "^RESULT|ERROR"    # ~5 min
grep -E "^\| (CLB LUTs|CLB Registers|Block RAM Tile|DSPs) " kernels/vectorop_rtl/synth/utilization.rpt
```

- WNS ≥ 0 at 300 MHz, as for MatmulKernel.  Record (VECTOROP_RTL_PLAN
  phase 0, 2026-10-05): WNS +0.248 ns (Fmax ≈ 324 MHz), 5 243 LUT,
  6 453 FF, 10 BRAM36, 11 DSP.  The burst generators are the critical
  paths (VECTOROP_RTL_KERNEL §6: no full-width add after the burst length).
- Its register map: ctest `VectorOpRtlDriver` and the fact registry
  (`registers.VectorOPKernel`); its m_axi bus parameters (outstanding
  counts, burst lengths, read / write only) are declared in
  `kernels/vectorop_rtl/syn/package_ip.tcl` and must match `vo_pkg` (fact
  `rtl.axi_masters`).
**PoolingKernel (RTL):**

```bash
make synth_pool_rtl 2>&1 | grep -E "^RESULT|ERROR"        # ~10 min
grep -E "^\| (CLB LUTs|CLB Registers|Block RAM Tile|DSPs) " kernels/pool_rtl/synth/utilization.rpt
```

- WNS ≥ 0 at 300 MHz.  Record (POOL_RTL_PLAN phase 0, 2026-10-05): WNS
  +0.077 ns (Fmax ≈ 307 MHz), 11 956 LUT (3 446 LUTRAM), 8 483 FF, 2 RAMB36
  + 2 RAMB18, 29 DSP.  The critical paths were control arithmetic (the
  config FSM's multiplier operands, the chunk sequencer, the emitter's step
  setup, the loader's burst decision) and the LP-2 Horner steps — each got
  a register (POOL_RTL_KERNEL §6).
- Its register map: ctest `PoolRtlDriver` and the fact registry
  (`registers.PoolKernel`); its m_axi bus parameters in
  `kernels/pool_rtl/syn/package_ip.tcl` must match `pl_pkg` (fact
  `rtl.axi_masters`).
- A **register-map, `m_axi` or bus-parameter change is an interface
  change** → Gate 5.

## Gate 3 — RTL behaviour test

Never start one while another behaviour test or `sim_hw_kv260` runs:
the stand's DDR model and projects are shared state.  Check first (exact
process names — `pgrep -f 'vivado|xsim'` would match its own shell):

```bash
pgrep -a -x 'vivado|xsimk|xelab|xvlog|vitis-run|vitis_hls' || echo "no Vivado / HLS job running"
```

Run the test in the background and wait for the completion notice:

```bash
make behavior_test_matmul             # or behavior_test_vectorop
```

- `behavior_test_vectorop` depends on `package_vectorop_rtl`,
  `behavior_test_matmul` on `package_matmul_rtl`, `behavior_test_pool` on
  `package_pool_rtl` (~20 s each; the test stand upgrades the IP and puts
  the instance widths back to the IP's defaults — MatmulKernel's gmem2
  becomes 128).
- Wall time: MatMul (RTL) ≈ 5 min (50 cases, 2.52 ms simulated, 297 s on
  2026-10-04), VectorOP (RTL) ≈ 4 min (119 cases, 0.986 ms simulated, 237 s
  on 2026-10-05), Pool (RTL) ≈ 3 min (45 cases, 0.606 ms simulated,
  2026-10-05) — the xsim simulated time, not the case count, sets the
  cost.  The output must end
  with these two lines followed by
  `[100%] Built target behavior_test_K`:

  ```
  [ts] kernel=MatmulKernel  total=50  passed=50  failed=0  all_passed=True
  [ck] MatmulKernel: PASS  (50/50)  …/matmul_op_test_report.json
  ```

  (`VectorOPKernel … total=119 passed=119`, `vector_op_test_report.json`;
  `PoolingKernel … total=45 passed=45`, `[ck] PoolingKernel: PASS  (45/45)`,
  `pooling_test_report.json`).
  `failed=0` and `all_passed=True` are mandatory; the make exits non-zero
  otherwise.  **If anything else, stop here** and show the failing cases:

  ```bash
  python3 -c "import json,sys; r=json.load(open(sys.argv[1])); [print(t['index'], t['label'], t['geometry'], t['errors'], t['mismatches'][:3]) for t in r['tests'] if t['status']!='PASS']" kernels/$K/kv260/*_test_report.json
  ```

- The run modifies tracked `.bd` / `.xci` / `.xpr` files of
  `hw/cormorant_test_stand` (IP upgrade into the stand's block design).
  **Do not commit them**, and do not blanket-revert the submodule either
  if it carries real testbench edits.

## Gate 4 — timing diff vs the previous run

```bash
python3 "$S/compare_timing.py" "$K" --build-dir "$BUILD_DIR"
```

It reads the fresh report, compares every case's `duration_ns` with
`kernels/K/kv260/K_timing_last.json` (the previous run's snapshot),
prints the changed cases sorted by absolute movement, a
`TOTAL (n matched cases)` row and `sim_time_ns`, new / removed cases and
`Slower than +1 %` (`--slower-pct` to change), then saves this run as the
next baseline.

- Cases are matched by label **and geometry**: VectorOP repeats labels
  (`ADD` is 11 sizes), shown as `ADD size=4097`.
- First run: no snapshot yet — absolute values are printed and saved; say
  so (the next run gives a real diff).
- `--no-save` compares without moving the baseline (the user says "don't
  update the baseline", or a speculative change).  `--baseline <file>`
  takes a snapshot or a raw `*_test_report.json`, e.g. the Step 0
  reference: `--baseline kernels/$K/kv260/${K}_report_before.json --no-save`.
  `--top N` limits the table.
- Durations are at the testbench's fixed 100 MHz sim basis, comparable
  between runs whatever the synthesis clock.  Each includes the testbench's DDR
  fill and register programming (VectorOP's 1-element `ADD` is 4.8 µs,
  MatMul `1x1x1` 5.8 µs), so tiny cases hardly move — judge a kernel
  change by the large cases and the TOTAL.
- There is **no run-to-run noise**: re-running an unchanged kernel
  reproduced every `duration_ns` and `sim_time_ns` exactly (2026-09-29,
  both kernels).  Every delta comes from the change — kernel RTL, fixture
  data or the test stand — so a +0.3 % on a targeted case is real.
- Exit 1 = the report has failures, 2 = report / baseline missing or of
  the other kernel.

## Gate 5 — interface changes only: block-design testbench + `sim_hw_kv260`

An INTERFACE change is anything the drivers or the block design see: an
`s_axi_ctrl` register added / moved / removed or its meaning changed, an
`m_axi` port's width / bundle / direction, or the buffer contract
(alignment, layout, tail-word writes — `VectorOP.h` alignment contract,
MatMul packed-B / GEMV image).  Then, besides the test-stand testbench:

1. Update the block-design testbench in
   `hw/cormorant_hw_128/cormorant_hw_128.srcs/sim_1/new/` — `vop_regmap.svh`
   / `mm_regmap.svh` (offsets, op / act codes), `vop_classes.svh` /
   `mm_classes.svh` (what each test programs; every register written on
   every call, since the IP keeps its last value), `tb_functions.svh`
   (how buffers are laid out: whole 16-byte words, VectorOP strides 0 or
   multiples of 8 elements).  It was stale for months until 2026-09-29
   because nobody ran it after interface changes.  Today the MatMul tests
   there never write `b_packed` / `gemv_kw` / `a_to_b` (reset 0 = the
   row-major tiled path).
   Also the kernel's control block (`kernels/matmul_rtl/rtl/mm_ctrl_s_axi.sv`
   / `kernels/vectorop_rtl/rtl/vo_ctrl_s_axi.sv` /
   `kernels/pool_rtl/rtl/pl_ctrl_s_axi.sv`), its driver generator's table
   (`scripts/gen_driver.py`) and the fact registry
   (`python3 tools/facts/facts.py check registers.MatmulKernel` /
   `registers.VectorOPKernel` / `registers.PoolKernel`).
2. Run it (all four kernels through the PS VIP, ~4 min):

   ```bash
   make sim_hw_kv260
   ```

   Pass: `##  TOTAL: 68 / 68 passed` and `##  ALL TESTS PASSED`
   (VectorOPKernel 22, ConvKernel 17, MatmulKernel 10, PoolingKernel 19)
   and exit 0 — `scripts/sim.tcl` exits 1 without `ALL TESTS PASSED`.
   Measured 2026-09-29: 68/68 in 237 s wall (again with the RTL MatmulKernel
   in phase 1; 68/68, 241 s, with the RTL VectorOPKernel on 2026-10-05).
   It uses `$BUILD_DIR/ip_repo_kv260` (the Conv HLS export and the three
   RTL IPs), so all four must have been built in this build tree.  It upgrades the IPs in `hw/cormorant_hw_128` and
   modifies its tracked `.bd` / `.xci` files: do not commit those.
3. Board: the new register must be proven with a write-then-read before
   trusting results, and generated projects / libraries regenerated (a
   stale project never writes a new register) — the `board-deploy` skill.

## After the gates

- **Regression → revert.**  If the change makes the timing worse (total
  or cases the change targets) and it is not a deliberate, documented
  trade-off, revert your kernel edit and re-run Gates 2–4 to get back to
  the baseline numbers.  A rejected attempt is still logged (MATMUL §2 is
  the model: "Tried and rejected").
- **Log the result** in the optimisation log (table above): a new
  section with the change, the RTL traps met, synthesis before → after
  (WNS and the utilization report), and the
  behaviour-test result — pass count and per-case / total `duration_ns`
  before → after against the Step 0 reference.  Board numbers go in later,
  after `board-deploy` (and `perf-regression` for the benchmark diff).
- Do not commit anything the tools modified in `hw/cormorant_test_stand`
  / `hw/cormorant_hw_128` (`.bd`, `.xci`, `.xpr`).

## Reporting back to the user

1. **Pass/fail** of each gate run (one line each, with counts).
2. **Total delta** vs the previous run (the TOTAL row, ns and %), and vs
   the Step 0 reference when there is one.
3. **Top movers** — 3–5 cases: `name: prev → now (Δns, ±X %)`; plus any
   `Slower than` cases and new / removed cases.
4. **Synthesis** — the WNS and resource moves; say explicitly whether the
   interface changed (and whether Gate 5 ran).

## When `make` reports an unknown target

`make: *** No rule to make target '<X>'` means the build tree was
configured for another branch.  Refresh once from `$BUILD_DIR` and retry
(if the target is still missing: `TestMatmulBlas` needs a BLAS at
configure time — skip it; `behavior_test_*` / `sim_hw_kv260` need the
`hw/cormorant_test_stand` / `hw/cormorant_hw_128` submodules):

```bash
cmake .
```

Don't delete the build tree.

## What to skip

- Don't rebuild or synthesise kernels the user hasn't touched.
- Don't re-run cmake unless a CMakeLists or `platforms/*.json` changed
  (`CMAKE_CONFIGURE_DEPENDS` handles those automatically).
- Don't poll long runs with sleep loops — run them in the background and
  wait for the completion notice.

## Validation record (2026-09-29, scratch clone, `AXI_BUS_WIDTH=128`)

Both columns are the retired HLS kernels' (their synthesis was removed in
MATMUL_RTL_PLAN phase 4 and VECTOROP_RTL_PLAN phase 3).  The RTL kernels'
gates ran in their plans: MatmulKernel phases 0–2b (`TestMatmulRtl` 50
fixtures + 200 random, lint clean, `synth_matmul_rtl` 13 min, behaviour test
50/50 in 297 s, `sim_hw_kv260` 68/68); VectorOPKernel phases 0–1
(`TestVectorOpRtl` 119 fixtures + 300 random, lint clean, `synth_vectorop_rtl`
4 min, behaviour test 119/119 in 237 s, `sim_hw_kv260` 68/68); PoolingKernel
phases 0–1 (`TestPoolRtl` 45 fixtures + 200 random, lint clean,
`synth_pool_rtl` 10 min, behaviour test 45/45, `sim_hw_kv260` 68/68; the
retired `pool-verify` skill covered the HLS PoolingKernel).

Every command above was run on unchanged kernels (`kernels/`,
`hw/test_data/`, `platforms/` identical to main 481e7a1), sequentially,
one Vivado job at a time:

| Gate | MatMul | VectorOP |
|---|---|---|
| 1 C-sim | `TestMatmulRef` 58/58, `TestMatmulBlas` 24/24; build 9 s (`make -B`), ctest 1 s | `TestSimulation` 119/119; build 4 s, ctest < 1 s |
| 1b fixtures | dump 50 cases ≠ checked-in 39 at the time (the 11 GEMV cases, since added — 50/50 PASS, 893 s) | dump identical to the checked-in 119 |
| 2 synthesis | 70 s; `FLAGS: 0`; perf table identical to the previous run | 38 s; `FLAGS: 0` |
| 3 behaviour test | 39/39 PASS, 901 s wall (incl. re-synthesis), `sim_time_ns` 5 170 915 | 119/119 PASS, 236 s, `sim_time_ns` 1 002 075 |
| 4 timing diff | 0 of 39 cases moved vs a run 2 h earlier (Σ 4 388 715 ns) | 0 of 119 moved (Σ 999 875 ns) |
| 5 `sim_hw_kv260` | 68/68, 237 s (all four kernels) | (same run) |

`compare_timing.py` was also checked against perturbed report copies
(slower / faster / +0.5 % / removed / new cases, a failed report, the
other kernel's report or snapshot) — each case is reported as described
above.  (The HLS synthesis summariser `csynth_check.py` went with the HLS
VectorOP synthesis.)
