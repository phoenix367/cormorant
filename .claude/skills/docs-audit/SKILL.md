---
description: Audit the project documentation — (A) static audit, docs vs code: one agent per doc area compares every factual claim (paths, names, flags, defaults, counts, numbers) with the current code and fixes stale facts, a second agent reviews the combined diff claim by claim, the test suites confirm the counts; (B) fresh-clone reproduction: clone from GitHub as the README says, overlay the unpushed work, one background agent follows only the docs through C-sim, scheduler tests, HW build, board setup, on-board tests, RTL tests and demos and writes GAPS.md, then restore the board and fix code + docs. Use when asked to audit / check / refresh the docs, after a large change or before telling anyone to clone the repo, or to "reproduce from a fresh clone" / test the docs as a new user.
allowed-tools: Bash Read Write Edit Agent
---

# docs-audit

Both modes were first run by the maintainer: A on 2026-09-28 (commit
172ad45), B on 2026-09-29 (23 gaps, among them `cmake/FindVitis.cmake` never
committed → ea3eeed code, 0c08fb8 docs, 481e7a1 hw_128 testbench).  A after
code changes; B when build / setup paths, repo contents (new files,
`.gitignore`, example configs, script defaults) or the board flow changed.

`S=.claude/skills/docs-audit` (scripts in `$S/scripts`, briefs in
`$S/templates`); commands run from the repo root.  `/` is nearly full:
`export TMPDIR=/mnt/data/cormorant_repro/tmp` (`mkdir -p` it) in every shell.

## A. Static audit — docs vs code (host only, ~1 h)

1. **Measure once, write the briefs.**  `git status --short -- '*.md'` — commit
   or keep in mind pre-existing doc edits (the review brief lists them).
   ```bash
   python3 $S/scripts/audit_facts.py areas          # areas, files, line counts; UNASSIGNED files
   python3 $S/scripts/audit_facts.py counts --run --build build --briefs $TMPDIR/docs_audit
   ```
   `counts --run` runs both suites (~4 min) and measures ctest, RTL fixture
   manifests, the board model set and the perf cases; it prints every
   count-like claim of the current-state docs that matches no measured number
   (`CHECK` — a lead, not a verdict: dated numbers, sample output, per-module
   counts show up too) and writes `area_<name>.md` + `review.md` with the
   counts filled in.  Areas: root (README / CLAUDE.md / doc/README),
   scheduler-ref (doc/scheduler, inference-scheduler/CLAUDE.md),
   scheduler-guides (inference-scheduler/doc — the scheduler docs are ~7 500
   lines, hence two agents), kernels (references in full, optimisation logs
   only their current-state parts), build-test (doc/build-and-test,
   .claude/skills, hw/ submodule READMEs), demos, plans (status lines only).
2. **One agent per area, in parallel** — a single message with one Agent call
   per `area_*.md` (general-purpose; the file's text is the prompt).  Each edits
   only its files and reports `| file:line | was | now | evidence |`.
3. **Review** — one agent with `review.md` over the combined
   `git diff -- '*.md'`: every changed claim confirmed against code
   (`file:line`), corrected, or reverted when unprovable; edits to dated
   history reverted; old values left elsewhere listed.  Fix what it lists.
4. **Tests and counts** — a total in the docs is what the suite COLLECTS,
   a pass / skip split is what a run prints.  The registered ones
   (`facts.yaml`: test counts, the bitstream id, pool sizes, …) are checked
   and fixed by `python3 tools/facts/facts.py check` / `fix`; add a count
   you find repeated in several docs to the registry:
   ```bash
   cd inference-scheduler && .venv/bin/python -m pytest test/ -q                # 1633: all pass, 0 skip
   cd .. && inference-scheduler/.venv/bin/python -m pytest demo/chat/tests -q   # 188
   python3 $S/scripts/audit_facts.py counts --run   # no CHECK line left that is a stale count
   git diff --stat -- ':!*.md'                      # empty: docs only
   ```
5. Summarise the diff for the user; commit only on request ("Docs: stale facts
   compared with the current code").  hw/ submodule doc edits are a commit in
   the submodule (pushed first), then the pointer.

Rules: history stays (dated plan sections, optimisation-log entries,
before / after tables); a doc that is right about buggy code is reported, not
bent to the bug; agents do not run syntheses, board commands or the suites
(seven agents running pytest at once = inconsistent numbers and ~30 min of
CPU).  2026-09-28 found stale port widths, register offsets, fixture and test
counts, script paths and plan status lines.

## B. Fresh-clone reproduction — a new user follows the docs (~2 h 15 min + restore + fixes)

Needs: the board to itself for ~2.5 h (ask the user — no other agent or job on
it), ≥ 10 GB free on /mnt/data (the clone grows to ~7 GB), GitHub ssh.

**B1–B2. Clone and overlay the unpushed work.**
```bash
df -h /mnt/data /
D=$(date +%F); B=/mnt/data/cormorant_repro_$D
$S/scripts/make_clone.sh --list-untracked "$PWD"    # untracked, not-ignored paths + sizes
# write the ones that belong to the work (new sources, tests, docs, skills) into a list —
# not models, assets, build trees, venvs, locks, private board/ dirs
$S/scripts/make_clone.sh --files $TMPDIR/overlay_files_$D.txt "$PWD" "$D"
```
It clones `git@github.com:phoenix367/cormorant.git` + `git submodule update
--init` exactly as README Quick start §1 (47 MB, ~12 s), applies
`git diff <origin HEAD> --binary` of the working copy (unpushed commits
included; the submodules excluded — Vivado dirties them and the clone keeps
the pushed hw; `hw/test_data` is included) as the commit "local: unpushed
working-copy changes (not for push)", copies the listed untracked files
("local: unpushed new files"; ignored files are skipped and named — a fresh
clone lacks them too, that was FindVitis.cmake), disables the push URLs and
writes `$B/{env.sh,brief.md,GAPS.md,clone_info.txt,overlay.patch}`.  It stops
if the working copy is behind or diverged from origin.

**B3. Board snapshot (read-only).**
```bash
$S/scripts/board_snapshot.sh $B/board_before.txt
```
pl.bin SHA-256 (its first 12 hex digits = the bitstream id in
`perf_models/kv260/`), overlays and status, UIO names, the /root listing,
`_repro` leftovers, `kv260-chat`, the tmpfiles rule, cpuidle, cma.  If
`chat_service: active=active`, that is the production server — ask before
stopping it (`demo/chat/deploy.py --stop`) and restart it in B5.

**B4. Launch ONE general-purpose agent in the background** with the text of
`$B/brief.md` as its prompt (from `templates/new_user_brief.md`: docs only,
the user-known facts, env, the production paths it must never touch, `_repro`
names, ≥ 1 GB free on the board, removal of everything it created, own
bitstream allowed but never `--overlay-name pl`, board unresponsive > 2 min →
stop and report, no host sudo, long jobs with `run_in_background` and NEVER
`until … sleep` loops, skip SmolLM2-360M generation, SmolVLM only if time
allows, the order of paths, what counts as a gap, RUNLOG.md + GAPS.md written
incrementally).  Then leave the board and the clone alone and do not poll; if
the user asks, it is still running.  If it stops before its deliverables are
complete, continue it with SendMessage (context intact) instead of a new agent.

**B5. After it finishes.**
1. Leftover processes — on 2026-09-29 six `until … sleep` loops outlived the
   agent by hours, waiting for an `rc=` line a log never got:
   ```bash
   ps -eo pid,ppid,etime,cmd --forest | less
   $S/scripts/leftover_procs.sh $B          # rc 1 = leftovers listed; then --kill
   ```
   Polling loops that do not mention `$B` are listed but never killed.
2. Read its final message, `$B/RUNLOG.md` (board-cleanup section) and `$B/GAPS.md`.
3. Restore the production bitstream and prove it:
   ```bash
   cd inference-scheduler && source /mnt/data/xilinx/2025.2/Vitis/settings64.sh \
       && .venv/bin/python upload_bitstream.py --config bitstream_config_kv260.json
   cd ../demo/mnist && ../../inference-scheduler/.venv/bin/python run_demo.py --skip-download
   ```
   Expect ConvMNIST 98.92 % / 0.268 ms, LeNet 97.35 % / 2.81 ms.  "Overlay
   'pl' did not apply" → the agent's `design_cormorant` overlay is still
   there: `rmdir /sys/kernel/config/device-tree/overlays/design_cormorant` on
   the board, run again.
4. Compare the board with the snapshot — stable lines must be identical:
   ```bash
   $S/scripts/board_snapshot.sh $B/board_after.txt
   $S/scripts/board_snapshot.sh --compare $B/board_before.txt $B/board_after.txt   # "none — board state restored"
   ```

**B6. Fix.**
1. Triage GAPS.md: repo / code defects (a file never committed, hard-coded
   paths, a false success, a non-hermetic test, a wrong example config) → fix
   in code, one commit; the rest → docs.
2. A doc agent gets the GAPS list plus the code-side decisions (so the docs
   describe the fixed behaviour) and mode A's rules.
3. Re-run both suites (fixes add tests: the counts move), `audit_facts.py
   counts`, then the mode-A review brief over the diff.
4. Commit on request: code first, docs second; submodule fixes in the
   submodule first.  The clone can go afterwards (`rm -rf $B`, ~7 GB) — ask;
   keep RUNLOG.md / GAPS.md if the user wants them.

## Reference: the 2026-09-29 run

Clone 47 MB; whole reproduction ~2 h 15 min with overlap (demo preparation
during Vivado, board tests during RTL tests on a driver snapshot).

| Path | Result | Wall |
|---|---|---|
| C simulation | PASS after `make TestMatmulBlas` (ctest *Not Run* with a BLAS) | ~1.5 min |
| Scheduler venv / tests / CLI | 1558 pass, 1 fail (planning test needed a local bitstream config), 5 skip; pytest not in requirements | pytest 2 m 45 s (venv 3.5 min: slow pip index) |
| Chat tests | 153 collected, ~60 skip without tokenizers (README said 149) | 29 s |
| HLS × 4 | PASS only with `-DVitis_HLS=… -DVitis_HLS_TCL_FLAG=--tcl` (FindVitis.cmake missing) | 4 m 23 s |
| Vivado `build_hw_kv260` | PASS, WNS +1.067 ns (re-synthesises all four first) | 71 min |
| Board setup + upload | PASS; "applied" printed for a rejected overlay | 4 s |
| `run_remote_tests.py` / `run_remote_perf.py` | 148/148; perf 60/60 after fixing the example's UIO names | 11 min / 21 s |
| RTL behaviour tests | VectorOP 119, Pool 43, Matmul 39, Conv 63 — all pass | 45 min |
| `sim_hw_kv260` | stale testbench: 19/25 VectorOP, Conv fatal — yet exit 0 | 3 min |
| Demos | MNIST 98.92 % / 0.268 ms, LeNet 97.35 % / 2.810 ms; MobileNet v1 / v2 / ResNet-18 81.0 / 62.9 / 59.9 ms; BERT 965.8 ms, EM/F1 88.0/90.3, bit-exact; SmolLM2-135M 9.4–9.8 tok/s, board_gate 50/50; SmolVLM 9.5 tok/s | generation RSS: SmolLM2 12.8 GB, SmolVLM 19.3 GB |

Gaps: 2 blocker (FindVitis.cmake ignored by `*.cmake` → no VectorOP synthesis
target, so `synthesize_kv260`, `build_hw_kv260` and the VectorOP RTL test
failed; `llm_board.py` hard-coded `/root/kv260_chat` + `/root/smollm2_weights`),
5 major (ctest with BLAS, pytest missing, perf example UIO names, `.venv-export`
missing from the SmolLM2 steps, `sim_hw_kv260` failing with exit 0), 10 minor,
6 nit.

## Pitfalls

- **Polling loops outlive agents** — the brief forbids them; still run
  `leftover_procs.sh` after every agent run.
- **Masked by this host**: the login profile sources Vitis (a doc missing
  `source` shows no error); pip.conf's extra index is slow (`env.sh` sets
  `PIP_CONFIG_FILE=/dev/null`).  Neither is a doc gap.
- **Production on the board**: example configs default to production paths
  (`/root/bert_squad_weights`), `deploy.py` reuses the unit name
  `kv260-chat`, `--overlay-name pl` would overwrite `/lib/firmware/pl.bin` —
  hence the `_repro` names and B3 / B5.
- **The same dtbo under a second overlay name** is rejected by the kernel
  (`err=-22`) while configfs reads "applied"; the PL stays programmed with the
  agent's bitstream until B5.
- **`build_hw_kv260` and `behavior_test_*` re-synthesise and wipe the driver
  dirs** — no project generation meanwhile; the brief's driver snapshot lets
  board work overlap with RTL tests.  Never two test-stand jobs at once.
- **Documented steps dirty tracked files** in the clone (`.bd` / `.xci` /
  `.xpr` of both submodules, the SmolLM2 provenance JSON) — expected.
- **Excluding all of `hw/`** from the overlay would drop `hw/test_data`
  (tracked in this repo); `make_clone.sh` excludes only the submodule paths.
