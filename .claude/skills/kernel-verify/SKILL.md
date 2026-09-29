---
description: Verify a MatmulKernel or VectorOPKernel change end-to-end on the KV260 platform (argument `matmul` or `vectorop`) — C-simulation, HLS synthesis with an II / slack / interface diff, the RTL behaviour test, then a per-case timing diff against the previous run; plus the project rules for optimisations (baseline first, revert regressions, log the result) and for interface changes (block-design testbench + sim_hw_kv260). Use after any edit to kernels/matmul/ or kernels/vectorop/, their CMakeLists, Synthesis.tcl.in or RTL fixtures, or the platform JSON's kernels.matmul block.
allowed-tools: Bash Read
---

# kernel-verify `<matmul|vectorop>`

Four sequential gates (plus 1b and 5 when they apply).  **If any gate
fails, stop immediately and report the failure** — do not run later gates
on a broken build.  `K` below is the argument: `matmul` or `vectorop`.
ConvKernel and PoolingKernel have their own skills (`conv-verify`,
`pool-verify`).

| | `matmul` | `vectorop` |
|---|---|---|
| C-sim targets | `TestMatmulRef`, `TestMatmulBlas` (only when configure found a BLAS) | `TestSimulation` |
| ctest filter | `ctest -R Matmul` | `ctest -R TestSimulation` |
| synthesis | `synthesize_matmul_kv260` | `synthesize_vectorop_kv260` |
| csynth.rpt (under `$BUILD_DIR/kernels/K/kv260/`) | `matmul_kv260/hls/syn/report/csynth.rpt` (unified component flow) | `vadd_kv260/solution1/syn/report/csynth.rpt` (legacy `open_project` flow) |
| RTL behaviour test | `behavior_test_matmul` | `behavior_test_vectorop` |
| report (under `$BUILD_DIR/kernels/K/kv260/`) | `matmul_op_test_report.json` | `vector_op_test_report.json` |
| checked-in RTL fixtures | `hw/test_data/matmul_test_data/` (50 cases, 11 GEMV) | `hw/test_data/vecop_test_data/` (119 cases) |
| fixture target → output | `gen_matmul_test_data` → `$BUILD_DIR/matmul_test_data/` | `gen_vectorop_test_data` → `$BUILD_DIR/vectorop_test_data/` |
| test-stand testbench | `hw/cormorant_test_stand/kernels/matmul_op_test/matmul_op_test.srcs/sim_1/new/matmul_tb.sv` | `hw/cormorant_test_stand/kernels/vector_op_test/vector_op_test.srcs/sim_1/new/vectorop_tb.sv` |
| block-design testbench (`hw/cormorant_hw_128/cormorant_hw_128.srcs/sim_1/new/`) | `mm_regmap.svh`, `mm_classes.svh`, `tb_functions.svh` | `vop_regmap.svh`, `vop_classes.svh`, `tb_functions.svh` |
| optimisation log | `doc/kernels/MATMUL_OPTIMISATION.md` | `doc/kernels/VECTOROP_OPTIMISATION.md` |

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
K=matmul                                            # or vectorop
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
make TestMatmulRef TestMatmulBlas     # K=matmul  (drop TestMatmulBlas if make says no rule: no BLAS)
make TestSimulation                   # K=vectorop
ctest -R Matmul --output-on-failure           # K=matmul
ctest -R TestSimulation --output-on-failure   # K=vectorop
```

- ctest must end `100% tests passed`.  To see the per-case lines run the
  executable, e.g. `./kernels/matmul/TestMatmulRef | grep -E 'FAIL|passed|PASSED'`:
  `./kernels/matmul/TestMatmulRef` (ends `58/58 tests passed`
  / `TestMatmulSim PASSED`; 19 of the 58 are GEMV cases),
  `./kernels/matmul/TestMatmulBlas` (`24/24 tests passed` /
  `TestMatmulBlas PASSED`), `./kernels/vectorop/TestSimulation`
  (`All 119 tests passed.`).  Counts grow with the suites — the passed
  count must equal the total.
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
FIX=$SRC/hw/test_data/$([ "$K" = vectorop ] && echo vecop || echo matmul)_test_data
make gen_${K}_test_data                   # writes $BUILD_DIR/${K}_test_data/
diff -rq "${K}_test_data" "$FIX" && echo "fixtures unchanged"
# only when the diff is intended:
rm -f "$FIX"/test_*.hex "$FIX"/manifest.txt
cp "${K}_test_data"/* "$FIX"/
```

State as of 2026-09-29: both dumps are identical to the checked-in
fixtures — VectorOP 119 cases, MatMul 50 (the GEMV path's 11 cases, with
the `gemv_kw` manifest column, were added to the RTL suite that day; 50/50
PASS in 893 s).  A dump that differs changes the suite, not just refreshes
it: say so, and expect Gate 4 to list the new cases.
If the manifest gains a column or the DDR layout changes, the test-stand
testbench (table above) must parse / lay it out the same way.  Keep
fixtures small — xsim time grows with every element.

## Gate 2 — HLS synthesis

```bash
make synthesize_matmul_kv260          # or synthesize_vectorop_kv260
python3 "$S/csynth_check.py" "$K" --build-dir "$BUILD_DIR"
```

- The make must exit 0 and end `[100%] Built target synthesize_<K>_kv260`
  (MatMul ~70 s, VectorOP ~40 s).  Any `ERROR:` line is a hard failure.
  The target always re-runs and starts by deleting its HLS project, so
  the C driver directory (`…/impl/ip/drivers`) is missing until it
  finishes — never synthesise while something else (a scheduler project
  generation, another agent) reads it.
- `csynth_check.py` prints the top-level slack / resources, the pipelined
  loops that are not II=1, rows with an Issue / Violation type or negative
  slack, the `m_axi` table, the `s_axi_ctrl` register map, and the log's
  `ERROR` / `SCHED 204-65` / `II Violation` / `Inferring partial write`
  counts; then diffs against the previous synthesis
  (`kernels/K/kv260/K_csynth_last.rpt`, a copy it saves each run —
  `--no-save` keeps it) and ends with `FLAGS: n`.  Report every `FLAG:`
  line; do not fail the gate on them.  A `SCHED 204-65` count matters even
  with all loops II=1: a PIPELINE loop that HLS could not pipeline shows
  up only as `Pipelined = no` with no II entry (the "fewer pipelined
  loops" flag).
- Baselines (2026-09-29, 150 MHz target, estimated Fmax 205.47 MHz both):
  - **MatMul**: slack 0.00 ns, 20 pipelined loops all II=1, BRAM 88 (30 %),
    DSP 86, FF 35 628, LUT 60 891, URAM 8; `gmem0` (A) and `gmem1` (B)
    READ_ONLY `128 -> 128`, `gmem2` (C) WRITE_ONLY `16 -> 16` (by design —
    C is the one 16-bit element port left); registers up to `gemv_kw`
    0x74 and `a_to_b` 0x7C/0x80.
  - **VectorOP**: slack 0.00 ns, 14 pipelined loops all II=1, BRAM 32
    (11 %), DSP 33, FF 13 725, LUT 22 657; `gmem0/1` (a, b) READ_ONLY and
    `gmem2` (c) WRITE_ONLY, all `128 -> 128`; registers up to `act` 0x5C.
  The HLS 200-1449 "reads an input from its caller" warning on VectorOP's
  `b` loader is known and benign.
- A **register-map or `m_axi` change is an interface change** → Gate 5.

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

- `behavior_test_K` depends on `synthesize_K_kv260`, so it re-synthesises
  first (the same ~40–70 s) — Gate 2's report is regenerated identically.
- Wall time incl. that synthesis (2026-09-29): MatMul ≈ 15 min (39
  cases, 5.17 ms simulated), VectorOP ≈ 4 min (119 cases, 1.00 ms
  simulated) — the xsim simulated time, not the case count, sets the
  cost.  The output must end with these two lines followed by
  `[100%] Built target behavior_test_K`:

  ```
  [ts] kernel=MatmulKernel  total=50  passed=50  failed=0  all_passed=True
  [ck] MatmulKernel: PASS  (50/50)  …/matmul_op_test_report.json
  ```

  (`VectorOPKernel … total=119 passed=119`, `vector_op_test_report.json`).
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
  between runs whatever the HLS clock.  Each includes the testbench's DDR
  fill and register programming (VectorOP's 1-element `ADD` is 4.8 µs,
  MatMul `1x1x1` 6.7 µs), so tiny cases hardly move — judge a kernel
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
2. Run it (all four kernels through the PS VIP, ~4 min):

   ```bash
   make sim_hw_kv260
   ```

   Pass: `##  TOTAL: 68 / 68 passed` and `##  ALL TESTS PASSED`
   (VectorOPKernel 22, ConvKernel 17, MatmulKernel 10, PoolingKernel 19)
   and exit 0 — `scripts/sim.tcl` exits 1 without `ALL TESTS PASSED`.
   Measured 2026-09-29: 68/68 in 237 s wall.  It uses every kernel's IP
   under `$BUILD_DIR/kernels`, so all four must have been synthesised in
   this build tree.  It upgrades the IPs in `hw/cormorant_hw_128` and
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
  section with the change, the HLS traps met, synthesis before → after
  (slack, II, BRAM / DSP / FF / LUT from `csynth_check.py`), and the
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
4. **Synthesis flags** — every `FLAG:` line of `csynth_check.py`; say
   explicitly whether the interface changed (and whether Gate 5 ran).

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
- Don't read the whole `csynth.rpt` (150–330 KB) — `csynth_check.py` or
  `sed -n` / `grep`.
- Don't poll long runs with sleep loops — run them in the background and
  wait for the completion notice.

## Validation record (2026-09-29, scratch clone, `AXI_BUS_WIDTH=128`)

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
other kernel's report or snapshot), `csynth_check.py` against a
perturbed report (II=2 loop, negative slack, Issue type, lost pipelined
loop, LUT +26 %, moved register, port width) — each case is reported as
described above.
