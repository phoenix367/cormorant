---
description: Run one board session for the shipped libraries after a scheduler change on the production bitstream — regenerate the libraries, gate / install / profile each one on the KV260 (BERT demo, Piper, SmolLM2-135M / 360M, SmolVLM), top up the performance model and merge the host model, run the kernel benchmark and record the baseline, bring the chat server back and check the simulator — with scripts/board_window.sh, which keeps the order and the per-board lock. Use after a scheduler, frontend or host-op change that regenerates the chat / TTS / BERT libraries (a row-ops offload, a new node, a cost-model change) and before quoting board numbers; not for a new bitstream (board-deploy, then perf-calibrate §2 in full).
allowed-tools: Bash Read
---

# board-window

What the HOSTOPS phases 4a / 4b and RPU_PLAN A1 each ran by hand, as one
script.  It is sequential and takes 2–3 hours with every model: the
libraries' gates are 4 min (Piper) to 35 min (SmolLM2-360M) each, the
perf-model refinement rounds about 17 min of host time each.  Commands run
from the repo root; the script finds the root itself.

```bash
export TMPDIR=/mnt/data/cormorant_repro/tmp
S=.claude/skills/board-window/scripts/board_window.sh
OUT=$CLAUDE_JOB_DIR/tmp/window_$(date +%Y%m%d_%H%M)        # logs + the profiles JSON

nohup $S --out $OUT --all > $OUT.log 2>&1 &                 # everything, in the background
$S --out $OUT --regenerate --models piper                    # one library's regate
$S --out $OUT --models smollm2-135m,smolvlm --topup --baseline --deploy --simulate
```

Stages, in the order the script keeps:

1. **Regenerate** (`--regenerate`, host): `generate_llm_project.py` per chat
   model, `generate_tts_project.py`; the BERT demo regenerates inside its
   own run.  A traceback in any generation log stops the script before it
   touches the board — the chat gates *install* the library they gate, so
   a stale build must never reach them.  Grep the generated `inference.c`
   for the new node's comment first when the change is a new node.
2. **Gates** (`--models`): the chat server is stopped, then the BERT demo
   (`run_demo.py --n 20 --profile-layers`: latency mean, bit-exact against
   the simulation and the emulation, the per-kind profile), `tts_board.py
   --profile` (PCM, encode and duration bit-exact per sequence, the chunk
   and encode timings), `llm_board.py --profile` per chat model (logits
   bit-exact on the gate prompts, `llm_image` for SmolVLM, the per-kind
   profiles).  A MISMATCH is a bug in the generated C against the spec, not
   noise: fix, regenerate and regate that library only.
3. **Top-up** (`--topup`): `perf_calibrate.py cases` / `run --resume` /
   `fit`, refinement rounds until one adds no call (at most three), then `host --merge`
   with every profile the window produced (a new host-op kind is unpriced
   until this step).
4. **Baseline** (`--baseline`): `run_remote_perf.py` and `compare_perf.py
   --record` for the loaded bitstream (perf-regression skill: a clean run
   records even when cases are flagged; the small MatmulKernel cases move
   2–7 % between runs minutes apart).
5. **Server** (`--deploy`, or always after gates): `deploy.py`, with
   `--regenerate` when BERT was in the model list (the server's BERT
   project is separate from the demo's).
6. **Simulate** (`--simulate`, host): `perf_calibrate.py simulate` against
   the window's profiles; every phase within a few percent and no
   unpriced node.

Rules:

- One board job at a time: the script runs the tools one after another and
  each takes the per-board lock; do not start another board tool while it
  runs, and never wrap it in a shell `flock`.
- Do not interrupt a gate mid-way (the board may hold a half-installed
  library); a stopped script leaves the server down — run `deploy.py`.
- The results go to the plan's results section and the memory: the
  numbers from the logs, not from memory.
- Board hang (ssh silent, a tool stalled): stop, report, no retries.
