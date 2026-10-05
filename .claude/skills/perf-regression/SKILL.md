---
description: Run the KV260 performance tests and say whether anything regressed — the 60 kernel benchmarks of run_remote_perf.py (VectorOP, Matmul, Conv, Pool) and, optionally, the MNIST / image-classification / BERT-SQuAD demo latencies — compared case by case with the recorded baseline of the loaded bitstream (baselines/<platform>-<bitstream-id>.json): latency moves above 2 %, failed cases and changed demo results are flagged, a new baseline is recorded on request. Use when asked to run the perf tests or check for a performance regression, after a kernel, bitstream or scheduler change that is already correct on the board, and before quoting board numbers in an optimisation log.
allowed-tools: Bash Read
---

# perf-regression

One board session, one comparison.  The reference numbers live in
`baselines/` (one file per platform + bitstream), not in the four
`*_OPTIMISATION.md` logs.  Commands run from the repo root unless stated.

```bash
export TMPDIR=/mnt/data/cormorant_repro/tmp      # host / is nearly full
PY=inference-scheduler/.venv/bin/python
CMP=.claude/skills/perf-regression/scripts/compare_perf.py
SSH="ssh -o BatchMode=yes -o ConnectTimeout=10 -i $HOME/.ssh/kv260-testkey root@192.168.100.8"
STAMP=$(date +%Y%m%d_%H%M)
```

## 0. Preconditions

- `inference-scheduler/perf_config.json` exists (local, gitignored copy of
  `perf_config.json.example`: the board's UIO names `fabric_vecop` /
  `fabric_matmul` / `fabric_conv` / `fabric_pool`, the local driver dirs:
  the Conv and Pool HLS exports and the `driver_vectorop_rtl` /
  `driver_matmul_rtl` outputs).
- The kernels are CORRECT on the board (board-deploy §3).  This skill
  measures speed only; a wrong result is not a perf question.
- Nothing else uses the board.  **Never two board jobs at once** — every
  board tool (`run_remote_perf.py`, the demos' `deploy_and_run.py`,
  `run_remote_tests.py`, `perf_calibrate.py run`, `upload_bitstream.py`,
  the chat / BERT scripts) takes the per-board lock
  `/tmp/kv260-board-<host>.lock` itself (`inference-scheduler/src/remote/lock.py`)
  and prints `waiting for board lock …` while another job holds it.  Do
  not wrap them in a shell `flock` on that file (they would wait for it
  forever); for other commands use
  `(cd inference-scheduler && .venv/bin/python -m src.remote.locked --host 192.168.100.8 -- CMD …)`.

## 1. Free the FPGA — the chat server owns it

```bash
$PY demo/chat/deploy.py --status      # exit 0 = running and healthy
```
```
  kv260-chat: inactive
  http://192.168.100.8:8000/health: no answer          (exit 1: not running)
```
`active` (and a /health answer) = running: note it (step 6 restarts it),
then `$PY demo/chat/deploy.py --stop` (demo/chat/doc/DEPLOY.md; `--stop` and
the restart were not exercised when this skill was validated — the server
was already stopped).  Never touch `/root/kv260_chat` or other production
directories on the board.

## 2. Bitstream id — which baseline

```bash
BID=$($SSH 'sha256sum /lib/firmware/pl.bin' | cut -c1-12); echo $BID      # b3309f424562
(cd inference-scheduler && .venv/bin/python -c "from src.perf_calls import local_bitstream_id as f; print(f())")
ls .claude/skills/perf-regression/baselines/                               # kv260-b3309f424562.json; kv260-68665fc1833a.json (MatmulKernel without bus parameters), kv260-bbb9a37f73f8.json (HLS VectorOPKernel), kv260-1d28630fbfa4.json (no pool guard), kv260-caa67f49a5a3.json (HLS MatmulKernel)
```
The board's id is what gets measured; always pass it (`--bitstream-id
$BID`).  A local id that differs means the board runs another bitstream
than `bitstream_config_kv260.json` names — say so.  No
`baselines/kv260-$BID.json` → nothing to compare against; run anyway,
show the numbers, offer to record them as the new baseline (step 7):
```
no baseline for kv260 bitstream 0123456789ab (…/baselines/kv260-0123456789ab.json); recorded ones: kv260-1d28630fbfa4.json, kv260-68665fc1833a.json, kv260-b3309f424562.json, kv260-bbb9a37f73f8.json, kv260-caa67f49a5a3.json
nothing compared — rerun with --record to make these results the baseline      (exit 3)
```

## 3. Kernel benchmarks (~30 s)

```bash
$SSH 'sync; echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory'
cd inference-scheduler
.venv/bin/python run_remote_perf.py --config perf_config.json \
    --json $TMPDIR/perf_$STAMP.json > $TMPDIR/perf_$STAMP.log 2>&1; echo "exit $?"
grep -E "OVERALL|FAIL" $TMPDIR/perf_$STAMP.log; cd ..
```
Pass = `exit 0` and `── OVERALL: All 60 cases passed ──` (26 s wall
clock on 2026-09-29: connect + upload + build ≈ 10 s, the cases ≈ 15 s).
Exit 1 = a case failed (its `ok` is false in the JSON; compare flags it
FAILED).

## 4. Compare

```bash
$PY $CMP --run $TMPDIR/perf_$STAMP.json --bitstream-id $BID     # --brief: summary only
```
Output shape (validation run 2026-09-29 14:31 against the 09:53 run, same bitstream;
the id shown is today's bitstream):
```
kernels: baseline run 2026-09-29 09:53 (…/perf_20260929.json)
platform kv260, bitstream b3309f424562

  section         case                         base ms     now ms   Δlat %   Δthr %  flag
  VectorOPKernel  ADD-1K                        0.0087     0.0088    +1.15    -0.57
  …
  ConvKernel      3x3-64ch-56x56                2.6830     2.6830    +0.00    +0.00
  …
compared 60 cases (threshold ±2 %, min Δ 0.001 ms): 0 regressions, 0 improved, 0 failed, 0 result changes, 0 missing, 0 new, 0 redefined
worst 5 (largest latency increase):
  VectorOPKernel/ADD-1K: 0.0087 -> 0.0088 ms (+1.15 %)
  PoolingKernel/GlobalAvgPool-7x7-64: 0.0254 -> 0.0255 ms (+0.39 %)
  …
verdict: PASS
```
Flags: `REGRESSION` (latency up > `--threshold` %, default 2, AND by more
than `--min-delta-ms`, default 0.001 ms — the shortest calls jitter by
up to 0.5 µs, e.g. ADD-4K 0.0175 → 0.0180 ms in one run of four), `improved` (down by as much), `FAILED` (ok = false),
`RESULT CHANGED` (a demo's bit-exact result moved), `REDEFINED` (same
label, other fields — not compared); missing / new cases are listed.
Exit: **0** no regression, **1** regression / failed / result changed,
**2** input error, **3** no baseline for this bitstream.

Noise on an unchanged bitstream (five runs on 2026-09-29): every case
within ±1.2 % except the ~10–20 µs VectorOP calls, which moved by up to
0.5 µs (ADD-4K +2.9 % once, +1.1 / +0.6 % in the next runs) — hence the
1 µs floor; the demos' MNIST means identical to 0.1 µs.  A flagged case is
real; only one just above the threshold is worth one re-run of step 3
before reporting it.

## 5. Demo latencies (optional)

MNIST (regenerate + 2 models × 10 000 images) took 51 s on 2026-09-29;
the image-classification and BERT commands were checked with
`--check-only` (preflight passes) but not timed here — BERT alone is
10 × ~1 s inferences plus a 17 s board build.

```bash
$SSH 'sync; echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory'
(cd demo/mnist && ../../inference-scheduler/.venv/bin/python run_demo.py --skip-download)
(cd demo/image_classification && ../../inference-scheduler/.venv/bin/python run_demo.py --skip-download)
(cd demo/bert_squad && ../../inference-scheduler/.venv/bin/python run_demo.py)
$PY $CMP --bitstream-id $BID --demo demo/mnist/build/results.json \
    --demo demo/image_classification/build/results.json --demo demo/bert_squad/build/results.json
```
Each writes `demo/<name>/build/results.json`; `--demo` takes the demo
name from the directory above `build/` (`NAME=PATH` to override) and
combines with `--run` in one call.  Compared per model: `mean_ms`
(latency), `throughput_ips`, and the bit-exact results — MNIST `correct`
of `timed`, top-1 `class_id:logit` per image, BERT `examples` / `uid_ok`
/ EM / F1.  Unlike the kernel cases, demo latency depends on the
scheduler and the regenerated project too (the mnist / image
`run_demo.py` regenerates it; bert_squad reuses `build/project` unless
`--regenerate`): a demo delta with flat kernels is a scheduler change.
The image demo times one image (single sample); MNIST 9 950.

## 6. Restore

Restart the chat server ONLY if step 1 found it running:
`$PY demo/chat/deploy.py` (waits for /health).

## 7. Record a baseline — only when the user asks

After a new bitstream, or when the user accepts a change as the new
reference:
```bash
$PY $CMP --run $TMPDIR/perf_$STAMP.json [--demo …] --bitstream-id $BID --record
```
It compares first (exit status = the comparison's), then writes
`baselines/<platform>-<id>.json`.  Only a clean run is recorded (a
failed case or demo → error, exit 2).  Recording kernels leaves the demo
entries alone and vice versa; a replaced section (kernels, or one demo)
moves to `history` with its run time, source and latencies.  Format:
`kernels.cases["<Kernel>/<label>"] = {fields, lat_ms, thr, thr_unit}`,
`demos.<name>.models.<model> = {lat_ms, p50_ms, thr, thr_unit, checks}`,
each section with `run` (the results file's mtime) and `source`.  The
baselines are checked-in files: tell the user to commit them.

## Rules and pitfalls

- **Board hang**: if ssh stops answering, a case times out, or
  `run_remote_perf.py` stalls, STOP and tell the user — no retries, no
  reboot attempts; the board needs a power cycle.  Never start a second
  board command while one runs.
- Memory: drop caches + compact before each board run (3.9 GB RAM, no
  swap, `cma=1000M`).
- `perf_config.json` driver dirs must be the ones the loaded bitstream
  was built from, or the benchmark programs stale register offsets
  (board-deploy §3: a register that is not there reads 0).
- A new or renamed case in `perf_config.json` shows as `new` / `missing`
  until recorded; a changed geometry under an old label shows `REDEFINED`
  — rename the label instead.
- Report: verdict, counts, every flagged case (base → now, Δ %), the
  worst five, and whether a baseline was recorded.  Quote the numbers
  from the compare output, not from memory or the optimisation logs.
