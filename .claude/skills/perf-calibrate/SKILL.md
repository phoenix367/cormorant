---
description: Make the per-bitstream performance model that the scheduler's opt-in --plan mode prices its choices with — perf_calibrate.py cases → run (board, two passes) → fit → refinement loop, then the host-op model from board profiles (host), the simulator check against measured workloads (simulate) and what to commit. Use after every new bitstream (a new bitstream id: --plan refuses to run without its model), when --plan says "no performance model for bitstream …", after a scheduler change that alters the kernel calls of the shipped models, or when a planned prediction disagrees with the board.
allowed-tools: Bash Read
---

# perf-calibrate

`--plan` (doc/plans/TACTICS_PLAN.md §4, §9; INFERENCE_SCHEDULER.md
§Planning; `inference-scheduler/perf_models/README.md`) reads
`perf_models/kv260/<bitstream-id>.json`. The id is the first 12 hex digits
of the SHA-256 of the flat `.bin` the board loads (`src/perf_calls.py`).
The script is `inference-scheduler/perf_calibrate.py`, and its board runner is
`bench_src/calib_runner.c`. Run every command from `inference-scheduler/`
with `.venv/bin/python`. Put the subcommand FIRST, because `--models` and
`--profile` take several values.

## 0. When

| trigger | do |
|---|---|
| new bitstream: any Vivado build whose flat `.bin` differs | §2 in full (a new id), then §4; §3 only if host code changed |
| `--plan: no performance model for bitstream X (perf_models/kv260/X.json)` | the bitstream in `bitstream_config_kv260.json` has no campaign: §2 |
| scheduler change that alters shipped kernel calls (same bitstream) | coverage check §6, then a top-up: `cases`, `run --resume`, `fit` |
| host-op C code, `INFERENCE_HOST_THREADS` or the model set changed | §3, then §4 |
| hw submodule bump that leaves the `.bit` unchanged (sim / testbench only) | nothing, because the id is the hash of the `.bin`, not the commit. hw_128 bbfacf6 changed only the testbench, and the `.bit` built at d7ce129 still hashes to caa67f49a5a3 |

Current model: **kv260/588d721997cb** (2026-10-07), the 250 MHz bitstream with
VectorOPKernel's softmax unit (SOFTMAX_PLAN): a fresh case list (1443 cases, the softmax calls
among them), three refinement rounds (139 + 126 + 78; a fourth would add 27), then the prefill
softmax at every runtime key count (72 calls) to 1858 exact calls, and
`host.json` merged to 151 signatures / 39 kinds (unchanged); repeat spread median 0.057 %.  Note: `calib_runner.c` writes the softmax registers for ops
10 / 11 only, as the generated `run_softmax()` — written on every VectorOP call they cost
~80 ns per call (the first pass of this campaign; its 99 element-wise calls were re-measured).
Before it, **kv260/6436623029f7** (2026-10-06), the 250 MHz bitstream with
VectorOPKernel's activation unit (ACTIVATIONS_PLAN): a fresh case list (BERT's GELUs became
VectorOP calls; 1402 cases), one pass and three refinement rounds (157 + 103 + 67); topped up
on 2026-10-07 (OFFLOAD_PLAN: SmolVLM's GELU and Piper's decoder sums on VectorOP) with a fresh
case list (1406), `run --resume` and three rounds (125 + 29 + 5) to 1852, then for the chat
libraries' decode attention on ConvKernel (KV_DECODE_PLAN) a fresh case list (1438) and two
rounds (118 + 26): 1930 exact calls, `host.json` as above; repeat spread median 0.078 %,
`clock_mhz` 250 (perf_fit takes it from the local HWH).  Its family fits are loose (held-out
p90 9–62 %): the ConvKernel features are the engine cost model's walk with the frozen 100 MHz
parameters (`cost_model.RTL_CONV_FEATURES`, so refitting the cost model —
`tools/fit_cost_model.py`, done at 250 MHz in OFFLOAD_PLAN §4.3 — leaves the stored models
alone).  Before it, kv260/986cef4866a0 (the same design without
the unit, FMAX_250_PLAN): the case list of kv260/c2b2a6e5e50e (same shipped calls), one pass
and five refinement rounds (108 + 59 + 33 + 69 + 13) gave 2011 exact calls, median spread 0.099 %.
kv260/c2b2a6e5e50e (hw_128 850cc88, 2026-10-06), the 100 MHz bitstream
with all four kernels in SystemVerilog (CONV_RTL_PLAN phase 3).  Its campaign started from
the converged case list of kv260/dbb320fb7297 (one pass, one refinement round of 136); the
scheduler's RTL ConvKernel cost model then changed the shipped ConvKernel calls, so the
list was rebuilt (`cases`: 1402) and refined to convergence — 181 + 93 + 32 + 16 + 5 calls,
each round a `cases --refine` of 15–30 min and a `run --resume` of seconds — to 2343 exact calls; `host.json` stayed at 151 signatures / 39 kinds (unchanged; repeat spread median 0.017 %).  Against dbb320fb7297 the 1042
ConvKernel calls are a median 28.4 % faster, the others unchanged (median 0.00 %); conv /
conv-dw held-out p90 9.4 / 12.1 % with the RTL walk's terms, conv-mm 42.8 %.
kv260/dbb320fb7297 (hw_128 7d8eefe, 2026-10-05), the bitstream
with the SystemVerilog MatmulKernel, VectorOPKernel and PoolingKernel (POOL_RTL_PLAN
phase 3; the HLS ConvKernel, `AXI_CONV_IMPL=hls`).  Its campaign started from the converged
case list of kv260/b3309f424562 (the scheduler's kernel calls had not changed): one pass
(repeat spread median 0.016 %), then a refinement round that added 0,
to 1581 exact calls; `host.json` then had 151 signatures / 39 kinds (§3; the host
model does not depend on the bitstream).  Against b3309f424562 the 30 PoolingKernel calls are a median
19.5 % faster (2.7–36.1 %), the others unchanged (median 0.00 %).
kv260/b3309f424562 (the same design with the HLS PoolingKernel and its out-of-contract
guard; the MatmulKernel IP declaring its m_axi bus parameters, MATMUL_RTL_PLAN phase 5)
started from the converged list of kv260/68665fc1833a: a refinement round that added 0,
1581 exact calls; against 68665fc1833a the MatmulKernel calls are unchanged at the median
and up to 15 % faster (90 of 414 by more than 1 %).
kv260/68665fc1833a (the same design with every MatmulKernel crossbar slot at 2
outstanding bursts) started from the converged list of kv260/bbb9a37f73f8 (the HLS
VectorOPKernel): refinement rounds of 1 and 0 calls to 1581 exact calls.  kv260/bbb9a37f73f8 itself
took a full campaign and four refinement rounds (122, 31, 8, 3 calls; the fifth added 0)
to 1580 exact calls.  Before it, kv260/1d28630fbfa4
(the same design without the guard, one refinement round to 1383 calls; the 1251 calls
both measured agree to a median 0.002 %).  The previous HLS model,
kv260/caa67f49a5a3 (hw_128 d7ce129, the HLS MatmulKernel; topped up on 2026-10-01
with Piper to 1505 exact calls), stays for projects on that bitstream
(`AXI_MATMUL_IMPL=hls`); the records below are mostly its. The perf-regression skill keeps its own baseline for each bitstream id, so a new
bitstream needs one there too.

**Adding a model** (the 2026-10-01 top-up, about 45 min of chat-server downtime):
1. Add it to `perf_calibrate.SHIPPED`, with a loader in `shipped_graphs()` that
   builds the entries exactly as its generator does. For Piper that is
   `generate_tts_project.entry_graphs()`, so the calls are the library's.
2. Run the coverage check (§6) on a copy.
3. Run `cases`, `cases --refine`, `run --resume` and `fit`, repeating the
   refinement while it adds calls.
4. Profile the model and run `host --merge` (§3).
5. Run `simulate` (§4).

## 1. Preconditions (board steps: from the sources, not re-run here)

1. **Ids agree.** Get the local id (the `.bit` named by the untracked
   `bitstream_config_kv260.json`, converted exactly as `upload_bitstream.py` does):
   ```bash
   .venv/bin/python -c "from src.perf_calls import local_bitstream_id as f; print(f())"   # 588d721997cb today
   ssh -i ~/.ssh/kv260-testkey root@192.168.100.8 'sha256sum /lib/firmware/pl.bin' | cut -c1-12
   ```
   `run` makes this check itself on `--board-bin`, by default `/lib/firmware/<overlay>.bin`
   with the overlay name `upload_bitstream.py` uses for the same config (`overlay_name`,
   else the `.dtbo` stem: `pl.bin` today).
2. **The bitstream is actually loaded.** Follow the board-deploy skill §2. The id
   check only hashes the file. After a reboot the PL can hold the starter-kit overlay. Run
   `cat /sys/class/uio/uio*/name` on the board. It must list `fabric_vecop fabric_matmul fabric_conv fabric_pool`.
3. **Config** `perf_config.json`: it is local and untracked. Copy it from
   `perf_config.json.example` if it is missing. It needs:
   - `remote.uio_devices` keyed **`PoolingKernel`** (the run_remote_perf naming). A
     `remote_config*.json` says `PoolKernel`, and then every pool case fails;
   - all four `local.driver_dirs`, because `calib_runner` is only built with all four. They
     must hold the register maps of the LOADED bitstream. They point at `build/` (the
     verification tree), while the bitstream comes from `build_hw128`. All four must print `same`:
     ```bash
     for f in $(cd ../build_hw128 && ls kernels/{vectorop,conv,matmul,pool}_rtl/driver/*/src/x*_hw.h); do
         cmp -s ../build/$f ../build_hw128/$f && echo "same $f" || echo "DIFF $f"; done
     ```
4. **Chat server stopped.** It owns the kernels and the CMA.
   `../demo/chat/deploy.py --status` / `--stop`, or pass `run --stop-server`.
5. **Board memory.** `ssh … 'sync; echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory'`.
   `run` repeats this itself after building. Every case allocates its buffers as
   XRT BOs from CMA, and a failed allocation records the case as `"err":"alloc"`.
6. **One board job at a time.** `perf_calibrate.py run` takes the per-board lock
   (`/tmp/kv260-board-<host>.lock`, `src/remote/lock.py`) for its session and waits
   while another job holds it; the `deploy.py` it starts for `--stop-server`
   re-enters the lock (the holder marks its environment).  Do not wrap it in a
   shell `flock` on that file — that lock is not marked and it would wait forever.
7. **Host**: `cases` and `cases --refine` load all ten shipped models (assets under
   `demo/*/assets`, `demo/chat/assets/<model>`, `demo/tts/assets`).
   - Measured 2026-10-01: 2.2 min for `cases` and 11.4 min for `cases --refine`, with
     an 8.8 GB peak RSS for each.
   - On 2026-09-29 the peak was 23 GB, before the generator's memory diet.
   - Run them in the background. `fit` takes seconds.

## 2. Kernel campaign

```bash
STATUS=../.claude/skills/perf-calibrate/scripts/calib_status.py
.venv/bin/python perf_calibrate.py cases                                          # host
.venv/bin/python perf_calibrate.py run --config perf_config.json --stop-server    # board
.venv/bin/python $STATUS                                                          # complete?
.venv/bin/python perf_calibrate.py fit                                            # host
# refinement round: repeat until it adds 0
.venv/bin/python perf_calibrate.py cases --refine
.venv/bin/python perf_calibrate.py run --config perf_config.json --stop-server --resume
.venv/bin/python $STATUS && .venv/bin/python perf_calibrate.py fit
```
`all --config perf_config.json --stop-server` runs the first block in one command
(cases, run, fit). `all --refine --resume --config perf_config.json --stop-server` runs one
refinement round. `--bitstream-id ID` overrides the local id for every subcommand.
Use the same id throughout.

**cases**: prints one line per model (`bert: 66 new calls (6 s)` …), then
`space-filling set: 569 new calls`, then
`N cases -> perf_models/kv260/<id>.cases.json (…); about N min of calls per pass`.
The grid (seed 2026, a quarter held out) is 569 calls.
- **The first record** had 1087 cases (518 shipped + 569 grid).
- **Today's scheduler** gives 1249:
  - the LeNet FC convs have run as MatMul since 8e537bc (+3);
  - Piper adds 159 calls, 139 ConvKernel and 20 MatMul tactics (2026-10-01).
- The file is rewritten from scratch, so run `cases --refine` after it.
Keep the default `--seed`. Keep all models too: `--models` is for experiments only, because a
model made from part of the list lacks the exact entries of the models left out.

**run**: builds the bench project on the board (`upload / cmake / make → OK`).
It refuses with `error: the board runs bitstream X, the cases are for Y`,
`error: the chat server is running (it owns the kernels); pass --stop-server`
or `error: calib_runner was not built (all four kernels' drivers are needed)`.
It then runs two passes (`pass 1: N cases (M done)` … `pass 2: N/N (T s)`,
`measurements -> …calib.json`). Pass 2 uses a shuffled order and buffers filled with 0x5A
instead of 1. That is the determinism check. calib.json is saved after every chunk (`--chunk-seconds`, default 60).
Record: **two passes of 90 s each, about 10 min of chat-server downtime with the build**.
Refinement: 101 calls in 10 s. (The 30–60 min of §4.3 was the design estimate.)
**Pass:** `calib_status.py` exits 0, meaning both passes cover every case and nothing failed.

**fit**: writes `<id>.json` and prints the validation. For caa67f49a5a3, re-fitting the
committed data reproduces the file exactly, apart from `date`:
```
performance model kv260 / caa67f49a5a3: 1188 exact calls
determinism: 1188 calls measured twice, median spread 0.017 %, max 3.83 %, 27 above 0.5 %
family      k    n train med     max | held-out n     med     p90     max
conv        0   64     4.07%  76.08% |         16   6.34%  36.89%  59.21%
conv-dw     1   39     3.03%  39.60% |          8  10.77%  25.23%  38.28%
conv-mm     3  648     5.13% 144.81% |         74  10.34%  35.88%  52.24%
mm-gemv     5   83     0.01%   0.78% |         13   0.01%   0.21%   0.64%
mm-tiled    1   98     5.30%  85.20% |         21   9.28%  19.79%  32.38%
pool        8   27     3.28%  26.83% |          3   7.77%  13.57%  15.02%
vecop       8   80     0.24%  43.56% |          5   0.15%  12.67%  18.09%
vecop-div   1    7     0.54%   8.22% |          2   5.42%   9.62%  10.67%
note: held-out p90 above 3 % in ['conv', 'conv-dw', 'conv-mm', 'mm-tiled', 'pool', 'vecop', 'vecop-div']
```
For 1d28630fbfa4 (2026-10-04, the RTL MatmulKernel; reproduced apart from `date`):
```
performance model kv260 / 1d28630fbfa4: 1383 exact calls
determinism: 1383 calls measured twice, median spread 0.023 %, max 7.60 %, 47 above 0.5 %
family      k    n train med     max | held-out n     med     p90     max
mm-gemv     3   94     1.00%   6.80% |         13   1.14%   3.22%   5.62%
mm-tiled    5  121     0.39%  13.62% |         21   0.79%   3.98%   8.35%
```
(the other families as above within a few points): both MatmulKernel families are
under the planner's 5 % trust threshold.
**Pass criteria:**
- The determinism median should be in the 0.01 % range, because the kernels are deterministic
  (TACTICS_PLAN §1). The record's 27 noisy points are all calls a few tens of µs long.
  A large median means something else ran on the board. Measure again.
- The family errors should be close to the table above. The `note:` line is EXPECTED:
  only GEMV met the §4.3 target of 3 % (caa67f49a5a3; on 1d28630fbfa4 none did, mm-gemv 3.22 %; on bbb9a37f73f8 mm-gemv 2.89 %, on 68665fc1833a 2.91 %, on b3309f424562 0.89 %, on dbb320fb7297 0.79 %).
- The planner trusts a family prediction only when the family's held-out p90 is at most 5 %
  (`perf_model.MAX_MODEL_ERROR`). Today (dbb320fb7297: 0.79 / 3.22 %, as on the bitstreams before it) that is mm-gemv and mm-tiled. Every other tactic needs an
  exact entry. That is why the refinement exists.
- `call_overhead_us` (the minimum measured call) was 3.121 µs.

**Refinement loop.** `cases --refine` appends the unmeasured tactics that the fitted
model ranks within 15 % of each shipped MatMul's best (at most 6 per MatMul). It prints
`N refinement calls added -> … (measure them: run --resume)`. Next run
`run --resume` and `fit`. Repeat until it adds 0. Record: one pass added 101 (all
ConvKernel), which makes 1188 exact calls. That campaign stopped after one round, and it was
not converged. A second `--refine` on a copy (2026-09-29) adds **42**: 38 conv-mm,
3 mm-gemv and 1 mm-tiled, from lenet (4), BERT (3), SmolLM2-135M (18),
SmolVLM (1) and SmolLM2-360M (16). The estimate is 1.4 s of calls per pass. `cases --refine` counts calls that are
already measured but missing from the list (after a top-up `cases`) as "added". `run --resume` then
measures only the missing ones, as its `pass 1: N cases` line shows.
The 2026-10-01 top-up, with Piper, made three rounds:

| round | calls added | for | after `fit` |
|---|---:|---|---|
| 1 | 135 | BERT 7, SmolLM2-135M 36, SmolVLM 2, SmolLM2-360M 37, Piper 53 | 1444 exact calls |
| 2 | 63 | BERT 3, 135M 19, 360M 21, Piper 20 | 1505 exact calls |
| 3 | 18 | BERT 1, 135M 7, 360M 5, Piper 5 | checked on a copy; not measured yet |

The rounds converge. Each one costs a short `run --resume`, but also a stop of the
chat server.

## 3. Host-op model: `host.json`, independent of the bitstream

Its inputs are per-layer board profiles. Record: SmolVLM, SmolLM2-135M, BERT and the three CNNs
gave **56 exact signatures and 27 op kinds**. Piper, merged in on 2026-10-01, made that
151 signatures and 39 kinds. Chat server stopped, one job at a time:
```bash
# chat models: llm_board.py --profile --out (results.profile_layers per phase: vision, prefill_<n>, decode)
.venv/bin/python ../demo/chat/scripts/llm_board.py --project ../demo/chat/build/llm_project_smolvlm_256m --profile --out /tmp/vlm.json
.venv/bin/python ../demo/chat/scripts/llm_board.py --project ../demo/chat/build/llm_project --profile --out /tmp/135m.json
# BERT / CNNs: deploy_and_run.py --profile-layers (metrics.layer_stats)
(cd ../demo/bert_squad && ../../inference-scheduler/.venv/bin/python scripts/deploy_and_run.py --profile-layers)   # -> build/results.json (overwritten)
(cd ../demo/image_classification && ../../inference-scheduler/.venv/bin/python scripts/deploy_and_run.py --profile-layers --results build/results_prof.json)
# fit into a copy first; the old exact entries carry over only from the file you point at
cp perf_models/kv260/host.json /tmp/host.json
IC=../demo/image_classification/build/results_prof.json
.venv/bin/python perf_calibrate.py host --host-model /tmp/host.json --profile \
    smolvlm-256m-instruct=/tmp/vlm.json smollm2-135m-instruct=/tmp/135m.json \
    bert=../demo/bert_squad/build/results.json resnet18=$IC mobilenet_v1=$IC mobilenet_v2=$IC
```
Output: `  <model>: N host-op timings` per spec, then
`host model: N exact signatures, K kinds -> …` and one fit line per kind.
Check the counts against the record, then copy `/tmp/host.json` over the committed file.
- `llm_board.py` also runs its bit-exact gate. It builds and installs the project's library
  where the chat server loads that model (`board_paths`, keyed by `project.json` `model`).
  Profile the project that is deployed, or a copy generated under its own `--model-name`.
- **Pass ALL the profiles in one call.** The per-kind fits are rebuilt from this
  call's profiles only. Only the exact entries carry over. Verified: two CNN specs without
  layer stats gave `56 exact signatures, 0 kinds`. `--fresh` also drops the old exact entries.
- **Adding one model without re-profiling the others: `--merge`.**
  - **What it keeps.**  Every kind of the file you point at keeps its fit. Only kinds
    that are new get fitted, and every measured signature becomes an exact entry.
    It prints `merge: N new kinds …; M kept as fitted before …`.
  - **Start from the right file.**  Merge into a copy of the file *before* the new
    model. Merging again into a file that already has the model's kinds keeps their
    old fits.
  - **Piper (2026-10-01):**

    ```bash
    $PY ../demo/tts/scripts/tts_board.py --profile --out /tmp/piper.json      # gate + profile, ~3 min
    cp perf_models/kv260/host.json /tmp/host.json                              # (the pre-Piper file)
    .venv/bin/python perf_calibrate.py host --host-model /tmp/host.json --merge \
        --profile piper-lessac-medium=/tmp/piper.json
    ```

    The output: `piper-lessac-medium: 397 host-op timings`, 12 new `Tts*` kinds, and
    `VitAttnPrepNode` kept (SmolVLM's fit).
  - **Profile every bucket.**  `tts_bench` profiles each encode bucket as its own
    phase (`encode_<T>`) since 2026-10-01. With only the 400 bucket profiled (one
    size per kind), the smaller buckets were under-predicted by 6–16 %.
  - **The board library.**  `tts_board.py` also installs the library it gates, in
    the chat server's `lib/`.
- `0 host-op timings` means the results file has no per-layer profile. Run it again with `--profile` or `--profile-layers`.
- `MODEL` must be a shipped name (`perf_calibrate.SHIPPED`), not a project's `--model-name`.
  Append `:plan` when the profiled project was generated with `--plan`. The graphs are rebuilt
  from the assets with default frontend options, so non-default buckets or attention
  options will not match.
- Order search needs every host op priced. Otherwise the report says
  `N nodes not priced`, and the order is unchanged.

## 4. Simulator check (host only)

This compares predicted and measured totals per phase. `simulate` needs only
`llm_board.py --out` (its `bench`) for chat models, and any demo `results.json`:
```bash
IC=../demo/image_classification/build/results.json; MN=../demo/mnist/build/results.json
.venv/bin/python perf_calibrate.py simulate --profile resnet18=$IC mobilenet_v1=$IC mobilenet_v2=$IC \
    mnist_convnet=$MN mnist_lenet=$MN bert=../demo/bert_squad/build/results.json \
    smolvlm-256m-instruct=/tmp/vlm.json smollm2-135m-instruct=/tmp/135m.json
```
Verified on 2026-09-29 against the demo results on disk. It took about 1 s for the CNNs, about 6 s for BERT and about 40 s for `bert:plan`:
```
project                      phase         predicted  measured   error  CPU waits  unpriced
resnet18                     inference         60.0ms     59.9ms   +0.1%      59.3ms  0
mobilenet_v1                 inference         81.4ms     81.0ms   +0.5%      80.6ms  0
mobilenet_v2                 inference         62.9ms     62.9ms   -0.1%      62.0ms  0
mnist_convnet                inference          0.3ms      0.3ms   -1.4%       0.2ms  0
mnist_lenet                  inference          2.8ms      2.8ms   -0.1%       2.8ms  0
bert                         inference        965.5ms    962.3ms   +0.3%     589.4ms  0
```
Records (TACTICS_PLAN §9 T3):

| workload | predicted | measured | error |
|---|---:|---:|---:|
| SmolVLM `llm_image` | 3914.5 ms | 3885.4 ms | +0.7 % |
| SmolVLM prefill 16 / 64 / 256 | 339.8 / 445.5 / 1281.9 ms | 339.2 / 440.9 / 1278.9 ms | +0.2 / +1.0 / +0.2 % |
| SmolVLM decode step | 100.4 ms | 100.8 ms | −0.4 % |
| SmolLM2-135M prefill 16 / 64 / 256, decode | 339.8 / 445.5 / 1281.9 / 100.3 ms | 339.8 / 442.0 / 1284.7 / 101.4 ms | 0.0 / +0.8 / −0.2 / −1.0 % |
| BERT (p50 of the profiled run) | 965.5 ms | 963.0 ms | +0.3 % |
| ResNet-18, MobileNet v1 / v2 | 60.0, 81.4 / 62.9 ms | 60.3, 81.0 / 63.9 ms | −0.5, +0.5 / −1.6 % |
| Piper chunk (median of the gate's chunks) | 708.4 ms | 707.6 ms | +0.1 % |
| Piper `encode_<T>`, T = 32 / 64 / 128 / 256 / 400 (20 / 60 / 88 / 162 / 400 ids) | 25.4 / 38.1 / 67.5 / 139.3 / 259.7 ms | 25.6 / 37.7 / 67.9 / 139.9 / 260.2 ms | −0.6 / +1.1 / −0.6 / −0.5 / −0.2 % |

Piper rows come from `piper-lessac-medium=<tts_board.py --profile --out JSON>`.
- **The chunk** is compared with the median chunk time.
- **Each `encode_<T>` bucket** is compared with its fullest sequence, the largest
  id count in that bucket.  The graph prices T rows, and some host ops scale with
  the real count: 400 ids take 260 ms, 268 ids 239 ms.

`--html FILE` also writes the simulated phases as one interactive timeline
(src/timeline_html.py): a tab per phase, with the measured time beside each
prediction.  Each phase that has a per-layer profile also gets every node's
measured time, its "Measured vs predicted" table and the "color: error vs
measured" mode.  The profile comes from `profile_layers` (`llm_board.py` /
`tts_board.py --profile`) or `layer_stats` (the demos' `--profile-layers`).
The log line counts the nodes that have a measured time.  It is the quickest
way to see which node a prediction gets wrong.  Example, 2026-10-02, Piper
`chunk`: four of the sixteen TtsGate host ops are priced at 1511 µs.  On the
board, two of them take about 0.5 ms each.  The other two, gates #14 and
#28, average 5.8 and 7.2 ms (min about 0.52 ms) over the two profiled
chunks, so they carry the outliers.  `host.json`'s exact entry is the
plain mean, outliers included.  The total is still right.

**Pass:** every row within ±2 % and `unpriced` 0. A `:plan` spec must be paired with
results from the planned build. `bert:plan` against the unplanned results gives
949.9 vs 962.3 ms (−1.3 %). On the board the planned build measures 951.3.

## 5. Commit (only when the user asks)

- `perf_models/kv260/<id>.cases.json`, `<id>.calib.json` and `<id>.json` (the three model files);
- `perf_models/kv260/host.json`, if §3 was run;
- `perf_models/README.md`: its `kv260/<id>` paragraph names the current bitstream (the previous one follows it). Keep its form:
  "`kv260/<id>` is hw_128 <commit> (<date>): N calls, repeat spread median X %."

Commit them together with the hw submodule bump (TACTICS_PLAN §8). A re-fit of unchanged
data changes only `date`, so do not commit that. Planned projects (`--plan`,
`"plan"` in demo configs, `deploy.py --regenerate --plan`) must be regenerated to
use the new model. Unplanned projects do not change.

## 6. Pitfalls

- **Protect the committed files.** `MODELS_DIR` is hard-wired to `perf_models/kv260/`, and there is
  no output option. `cases` or `all` for an id that is already committed rewrites its list and drops the
  refinement entries. `all` without `--config` writes the case list BEFORE it fails
  with `run needs --config`. `fit` rewrites the model. `host` rewrites `host.json` unless
  `--host-model` is given. For experiments, redirect the directory:
  ```bash
  mkdir -p $TMPDIR/pc && cp perf_models/kv260/<id>.* $TMPDIR/pc/     # when refining / re-fitting
  .venv/bin/python -c "import sys,pathlib,perf_calibrate as pc; pc.MODELS_DIR=pathlib.Path('$TMPDIR/pc'); sys.exit(pc.main(sys.argv[1:]))" cases --bitstream-id <id>
  .venv/bin/python ../.claude/skills/perf-calibrate/scripts/calib_status.py --coverage $TMPDIR/pc/<id>.cases.json
  ```
  The second line with today's scheduler, followed by `--coverage`, is the coverage check.
  For caa67f49a5a3 it finds 6 unmeasured shipped calls, all from mnist_lenet since 8e537bc.
  The family models price them, and `simulate` still gives −0.1 %.
- **`run` without `--resume` starts calib.json afresh.** It is overwritten at the first chunk.
  Always pass `--resume` after `cases --refine` or a top-up `cases`, and after an interruption.
- **Failed cases are sticky.** calib_runner's `"ok":0` lines (`kernel X not available`,
  `alloc`, `timeout`) are stored like measurements. `--resume` skips them, and `fit`
  ignores them silently. Families with fewer than 6 points are dropped. Run `calib_status.py`,
  fix the cause, then run `calib_status.py --drop-failed` and `run --resume`.
- **A call that is not done after 20 s** ends the batch: `error: calib_runner rc=2`. The kernel
  is wedged. Reboot the board before reloading the PL (board-deploy §4), load the bitstream
  again, then `run --resume`. The hung call is recorded as failed, so it is skipped. Expect it to hang
  again if you `--drop-failed` it, because the kernels are deterministic. Report its key.
- **The server check is only `systemctl is-active kv260-chat`.** A server started with
  `launcher: "nohup"` is not seen. Use `deploy.py --status`. `--stop-server` restarts
  the server only if it had stopped it, and then runs a full `deploy.py`.
- **`--perf-model FILE` is not checked against the bitstream.** `planning.resolve_perf_model`
  loads any file. Only the default lookup is keyed by id. Do not use an old model to avoid a
  campaign.
- A new shipped model needs an entry in `perf_calibrate.SHIPPED` (a code change) before `cases`
  sees it.
