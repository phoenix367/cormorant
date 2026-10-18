# Brief: reproduce cormorant from a fresh clone, as a new user

You are a new user of **cormorant** (FPGA neural-network inference on the
Xilinx KV260).  A fresh clone is at `{{CLONE}}`.  Follow the repository's
documentation — and only its documentation — through every main path, and
record every place where it is wrong, incomplete or misleading.  You find
gaps; you do not fix them.

## What you know (nothing else)

- Host: this Linux PC (8 cores, 46 GB RAM, Python 3, gcc, cmake), internet access (GitHub, Hugging Face,
  the datasets the demos download).
- Xilinx Vitis / Vivado 2025.2 are installed at `{{XILINX}}`
  (`{{XILINX}}/Vitis/settings64.sh`).
- A KV260 at `{{BOARD}}`: `ssh -i {{KEY}} root@{{BOARD}}`, Kria Ubuntu 22.04
  with XRT and `cma=1400M`, already in use by its owner (see the board rules).
  A RealSense D435 is plugged into it (USB).  No microphone or speaker on the
  host: audio paths run on files.
- Nothing about the project's internals.  Start at `{{CLONE}}/README.md` and
  follow the links.  When the docs do not say how to go on you may read the
  clone's source to get unstuck — every time you had to is a gap.  Never read
  or use anything on the host outside `{{BASE}}` (not the maintainer's working
  copy, build trees, venvs or configs): a new user does not have them.

## Environment (every shell)

- `source {{BASE}}/env.sh` first: TMPDIR, XDG_CACHE_HOME, PIP_CACHE_DIR, HF_HOME
  under `{{BASE}}`, and `PIP_CONFIG_FILE=/dev/null` (the host's pip.conf has an
  unreachable extra index — host-specific, not a gap).  `/` is nearly full:
  never write to `/tmp` or `~/.cache` on the host.
- Logs: `{{BASE}}/logs/NN_short_name.log`, numbered in the order you run them.
- The login profile already sources Vitis (`XILINX_HLS` is set).  Run the docs'
  `source …/settings64.sh` steps anyway; note in RUNLOG that a missing `source`
  step would be masked here.
- No sudo on the host.

## Board rules (hard — breaking one ends the run)

- Production data is off limits — never write, move or delete anything that
  exists at the start: every entry of `/root` listed in
  `{{BASE}}/board_before.txt` (`root_entries`: `kv260_chat`, the `*_weights`
  directories, `rpu_firmware.elf`, …), `/lib/firmware/pl.bin`,
  `/lib/firmware/rpu_firmware.elf`, `/tmp/pl.dtbo`, the overlays `rpu` and
  `rpu-ipi` (you may reuse them as they are).  The owner has stopped the
  services `kv260-chat` and `kv260-miranda` and removes the production overlay
  `pl` only if the docs tell you to (see below).
- Your names end in `_repro`: work dirs `/tmp/repro_<name>`, weights
  `/root/<name>_weights_repro`, the chat install `/root/kv260_chat_repro`.  Set
  them in EVERY config copy before its first run — `.example` defaults are the
  owner's production paths (e.g. `weights_dir` `/root/bert_squad_weights`).
- If `kv260-chat` or `kv260-miranda` is active before you have deployed
  anything, it is the owner's: stop board work and report (your `deploy.py`
  would replace it).  Your chat install is `/root/kv260_chat_repro` (Miranda
  installs into `<chat dir>/miranda`).
- The Cortex-R5 firmware: build it as the docs say; on the board always pass
  `--firmware-name rpu_firmware_repro.elf` (`rpu_board.py`; for the chat
  server `rpu.firmware` in your config copy) — never the default name, which is
  the owner's file.  remoteproc0 may run your firmware afterwards; the owner
  restores it.
- Loading your own bitstream is allowed; the owner restores production
  afterwards.  If `upload_bitstream.py` refuses because the overlay `pl` is
  applied, remove that overlay as the docs say — never pass `--overlay-name pl`
  (it overwrites `/lib/firmware/pl.bin`).
- Keep at least 1 GB free on the board's `/` (`df -h /` before every weight
  upload: BERT 217 MB, SmolLM2-135M 335 MB, SmolVLM 514 MB).
- One board job at a time (they share the kernels), and none while a chat
  server of yours holds the FPGA (`deploy.py --stop` first).
- Board unresponsive (ssh / ping failing) for more than 2 min → stop all board
  work, do not reboot it, write up and report.  A kernel hang → kill your
  process, stop board work and report (reloading the PL under a hung kernel
  wedges the AXI port).
- At the end remove everything you created on the board, by explicit path —
  never a glob that could reach a production name: your `_repro` dirs,
  `/lib/firmware/design_cormorant.bin`, `/lib/firmware/rpu_firmware_repro.elf`,
  `/tmp/design_cormorant.dtbo`, your overlay directory, the empty work dirs the
  runners leave; stop your chat server and Miranda (`deploy.py --stop`) and your
  R5 firmware (`rpu_board.py --stop`).  Record what you removed and what is left (the PL stays programmed
  with your bitstream — say so).

## Long jobs

- Anything longer than ~2 min (venvs, syntheses, Vivado, RTL tests, model
  generation, board runs) runs with `run_in_background: true`, output to a log
  that ends with the exit code:
  `( cmd ) > {{BASE}}/logs/NN_x.log 2>&1; echo "rc=$?" >> {{BASE}}/logs/NN_x.log`.
  You are notified when it ends; do other work meanwhile.
- NEVER write a polling loop (`until grep …; do sleep …; done`,
  `while …; do sleep …`).  On 2026-09-29 six of them outlived their agent by
  hours, waiting for an `rc=` line a log never got.  To check on a job, read
  its log once.
- Before your final report make sure nothing you started still runs
  (`ps -eo pid,ppid,etime,cmd --forest`); stop what is left.

## Order of paths (each one as the clone's docs describe it)

1. The clone: `git log --oneline -3` (local commits, if any, are the owner's
   unpushed work — part of the docs under test), its size; the README's clone
   and submodule steps exactly as written (HTTPS).
2. Kernel C simulation (README Quick start §2, TESTING §2).
3. Scheduler: venv, test models, unit tests, the CLI on a test model (Quick
   start §3, TESTING §1, USER_GUIDE); the chat app tests.
4. Hardware build (Quick start §4, BUILD_TARGETS): the IP packaging, the Vivado
   bitstream (~1–2 h — start it early in the background and do host-only demo
   preparation meanwhile), the device-tree overlay.  The RTL tests re-package
   the IPs but leave the driver dirs (`build/kernels/<k>_rtl/driver/`), so they
   can run beside board work.
5. Board setup (Quick start §5): the CMA size check (`cma_boot`, report only —
   the board already has `cma=1400M`; never `--set` or `--reboot`), the tmpfiles
   rule, `upload_bitstream.py`, the UIO listing.
5b. The Cortex-R5 firmware (`tools/rpu/README.md`, RPU docs): the XSA, the BSP,
   then `rpu_board.py` on the demos' generated projects (after path 8 has
   generated them) with `--firmware-name rpu_firmware_repro.elf`, and
   `--stop` at the end.
6. On-board tests and benchmarks (Quick start §6, TESTING §4–5,
   REMOTE_TESTING): `run_remote_tests.py`, `run_remote_perf.py`; the USER_GUIDE's
   build-by-hand of a generated project on the board.
7. RTL tests (TESTING §3): `make behavior_test_<k>` for the four kernels that have one, one at
   a time (the test stand must never run two jobs at once; ~45 min in all), then
   `make sim_hw_kv260` (block design, ~8 min).  Read the summaries, not only the
   exit codes.
8. Demos, each by its README (`demo/README.md` lists them): `demo/mnist`,
   `demo/image_classification`, `demo/object_detection`, `demo/stereo_depth`
   (also `--capture` with the D435), `demo/bert_squad`, `demo/tts` (the
   library and its board gate), `demo/asr` (Moonshine base: the library and its
   board gate), `demo/chat` (the bert-squad, SmolLM2-135M, Piper and Moonshine
   backends: deploy, the clients, the curl examples, `board_gate.py`),
   `demo/miranda` (deploy on top of your chat server; the client with an audio
   file as input), `demo/camera` (as far as it goes without a display).  Only if
   time allows: SmolVLM-256M (its generation peaks at ~3.3 GiB RAM — never
   beside another generation).  Skip the SmolLM2-360M generation (7.7 GiB RAM).
9. The test suites as the docs give them (TESTING, README §Testing): the
   scheduler and chat pytest suites, the run-tests helper, the fact registry.

The 2026-09-29 run took ~2 h 15 min with overlap (Vivado 71 min, RTL tests
45 min); with today's paths expect 4–6 h (Vivado 1–2 h, the Conv behaviour
test ~75 min).  Keep going until every path is done or blocked.

## What counts as a gap

Following the docs literally fails; it needs an undocumented step, install,
flag or config edit; you had to read source or guess; the output shows another
name, number, count or file than the doc (beyond run-to-run noise, ~3 % on
latencies); a tool reports success while it failed; a doc points at the
owner's own paths; a documented step silently modifies tracked files; a step
run as written would touch production data on the board.  Severities and entry
format: `{{BASE}}/GAPS.md` (already holds the format — keep it).

## Deliverables — write them incrementally, after every step

- `{{BASE}}/RUNLOG.md`: a header (clone HEAD, host, env), a summary table
  `| Path | Result | Notes |`, then per path a table
  `| Step | Command | Result | Wall |` with the exact commands, the key output
  lines, rc and the log file; at the end "Board cleanup" (removed / still
  there) and "Host state left behind".
- `{{BASE}}/GAPS.md`: one entry per gap (id, severity, doc file:line, says,
  happened, workaround, suggested fix) and the index by severity.
- Every config you used, copied to `{{BASE}}/configs_used/`; any wrapper you had
  to write in `{{BASE}}/workarounds/`.
- Final message: the summary table, the gap index (id — severity — title), what
  is left on the board and on the host.

Never commit, push or open PRs, and never edit the clone's docs or code to
"fix" a gap — work around it outside the tracked tree and record it.
