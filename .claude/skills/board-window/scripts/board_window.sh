#!/usr/bin/env bash
# Copyright 2026 Ivan Gubochkin
# SPDX-License-Identifier: Apache-2.0

# board_window.sh — one board session for the shipped libraries after a scheduler change:
# regenerate (host), gate + install + profile each library (board), top up the performance
# model, benchmark + baseline, bring the chat server back, check the simulator.  Every board
# tool takes the per-board lock itself; the chat server is stopped first and restarted at the
# end (or by --deploy).  Logs and JSON profiles land in --out.
#
#   board_window.sh --out DIR [--regenerate] [--models bert,piper,smollm2-135m,smollm2-360m,smolvlm]
#                   [--topup] [--baseline] [--deploy] [--simulate] [--all]
#
# --all = --regenerate --models <all five> --topup --baseline --deploy --simulate.
# The order of the models is fixed (BERT, Piper, 135M, 360M, SmolVLM): the chat libraries
# are installed where the server loads them, so gate them only from a regenerated build.
set -u
R=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
P=$R/inference-scheduler/.venv/bin/python
SSH="ssh -o BatchMode=yes -o ConnectTimeout=10 -i $HOME/.ssh/kv260-testkey root@192.168.100.8"
export TMPDIR=${TMPDIR:-/mnt/data/cormorant_repro/tmp}
OUT=""; REGEN=0; MODELS=""; TOPUP=0; BASELINE=0; DEPLOY=0; SIMULATE=0
ALL="bert,piper,smollm2-135m,smollm2-360m,smolvlm"
while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT=$2; shift 2;;
    --regenerate) REGEN=1; shift;;
    --models) MODELS=$2; shift 2;;
    --topup) TOPUP=1; shift;;
    --baseline) BASELINE=1; shift;;
    --deploy) DEPLOY=1; shift;;
    --simulate) SIMULATE=1; shift;;
    --all) REGEN=1; MODELS=$ALL; TOPUP=1; BASELINE=1; DEPLOY=1; SIMULATE=1; shift;;
    *) echo "unknown option $1" >&2; exit 2;;
  esac
done
[ -n "$OUT" ] || { echo "--out DIR is required" >&2; exit 2; }
mkdir -p "$OUT"
has() { case ",$MODELS," in *",$1,"*) return 0;; *) return 1;; esac; }
step() { echo "[$(date +%T)] $*"; }
rc() { echo "[$(date +%T)]   -> rc=$1"; }
drop() { $SSH 'sync; echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory' 2>/dev/null; }
BID=$(cd "$R/inference-scheduler" && $P -c "from src.perf_calls import local_bitstream_id as f; print(f())")
step "bitstream $BID; models: ${MODELS:-none}; regenerate=$REGEN topup=$TOPUP baseline=$BASELINE deploy=$DEPLOY simulate=$SIMULATE"

# ---- host: regenerate ------------------------------------------------------------------
if [ $REGEN = 1 ]; then
  cd "$R/demo/chat"
  has smollm2-135m && { step "generate SmolLM2-135M"; $P scripts/generate_llm_project.py > "$OUT/gen_135m.log" 2>&1; rc $?; tail -1 "$OUT/gen_135m.log"; }
  has smollm2-360m && { step "generate SmolLM2-360M"; $P scripts/generate_llm_project.py --assets assets/smollm2-360m-instruct --model-name smollm2-360m-instruct > "$OUT/gen_360m.log" 2>&1; rc $?; tail -1 "$OUT/gen_360m.log"; }
  has smolvlm && { step "generate SmolVLM-256M"; $P scripts/generate_llm_project.py --model-name smolvlm-256m-instruct > "$OUT/gen_vlm.log" 2>&1; rc $?; tail -1 "$OUT/gen_vlm.log"; }
  cd "$R/demo/tts"
  has piper && { step "generate Piper"; $P scripts/generate_tts_project.py > "$OUT/gen_piper.log" 2>&1; rc $?; tail -1 "$OUT/gen_piper.log"; }
  for f in gen_135m gen_360m gen_vlm gen_piper; do
    [ -f "$OUT/$f.log" ] && grep -qiE "Traceback|Error" "$OUT/$f.log" && { step "generation failed ($f.log); not touching the board"; exit 1; }
  done
fi
cd "$R"

# ---- board: the gates, in a fixed order -----------------------------------------------------
if [ -n "$MODELS" ]; then
  step "stop the chat server"; $P demo/chat/deploy.py --stop 2>&1 | tail -1
  if has bert; then
    drop; step "BERT demo (regenerate + 20 examples + profile)"
    (cd demo/bert_squad && $P run_demo.py $([ $REGEN = 1 ] && echo --regenerate) --n 20 --profile-layers) > "$OUT/bert.log" 2>&1; rc $?
    grep -E "latency mean|BIT-EXACT|bit-exact|FAIL" "$OUT/bert.log" | tail -4
  fi
  if has piper; then
    drop; step "tts_board (gate + install + profile)"
    $P demo/tts/scripts/tts_board.py --profile --out "$OUT/piper.json" > "$OUT/tts_board.log" 2>&1; rc $?
    grep -E "BOARD GATE|MISMATCH|differ|encode [0-9]+:" "$OUT/tts_board.log" | tail -10
  fi
  for m in smollm2-135m smollm2-360m smolvlm; do
    has $m || continue
    case $m in
      smollm2-135m) proj=demo/chat/build/llm_project; tag=135m;;
      smollm2-360m) proj=demo/chat/build/llm_project_smollm2_360m; tag=360m;;
      smolvlm) proj=demo/chat/build/llm_project_smolvlm_256m; tag=vlm;;
    esac
    drop; step "llm_board $m (gate + install + profile)"
    $P demo/chat/scripts/llm_board.py --project $proj --profile --out "$OUT/$tag.json" > "$OUT/llm_board_$tag.log" 2>&1; rc $?
    grep -E "BOARD GATE|MISMATCH|llm_image|prefill_256" "$OUT/llm_board_$tag.log" | tail -4
  done
fi

# ---- board: the performance model ---------------------------------------------------------
if [ $TOPUP = 1 ]; then
  cd "$R/inference-scheduler"
  step "perf_calibrate cases (top-up)"; $P perf_calibrate.py cases > "$OUT/pc_cases.log" 2>&1; rc $?; tail -1 "$OUT/pc_cases.log"
  drop; step "perf_calibrate run --resume"; $P perf_calibrate.py run --config perf_config.json --resume --stop-server > "$OUT/pc_run.log" 2>&1; rc $?; tail -1 "$OUT/pc_run.log"
  step "perf_calibrate fit"; $P perf_calibrate.py fit > "$OUT/pc_fit.log" 2>&1; rc $?; head -3 "$OUT/pc_fit.log"
  for i in 1 2 3; do
    step "refine round $i: cases"; $P perf_calibrate.py cases --refine > "$OUT/pc_refine$i.log" 2>&1; rc $?; tail -1 "$OUT/pc_refine$i.log"
    grep -q "^0 refinement calls" "$OUT/pc_refine$i.log" && break
    drop; step "refine round $i: run"; $P perf_calibrate.py run --config perf_config.json --resume --stop-server > "$OUT/pc_run_r$i.log" 2>&1; rc $?
    step "refine round $i: fit"; $P perf_calibrate.py fit > "$OUT/pc_fit_r$i.log" 2>&1; rc $?; head -1 "$OUT/pc_fit_r$i.log"
  done
  profiles=""
  for spec in smollm2-135m-instruct:135m smollm2-360m-instruct:360m smolvlm-256m-instruct:vlm piper-lessac-medium:piper; do
    [ -f "$OUT/${spec#*:}.json" ] && profiles="$profiles ${spec%:*}=$OUT/${spec#*:}.json"
  done
  [ -f "$R/demo/bert_squad/build/results.json" ] && profiles="$profiles bert=$R/demo/bert_squad/build/results.json"
  if [ -n "$profiles" ]; then
    step "host model: merge the profiles"; cp perf_models/kv260/host.json "$OUT/host_before.json"
    $P perf_calibrate.py host --host-model perf_models/kv260/host.json --merge --profile $profiles > "$OUT/host_merge.log" 2>&1; rc $?
    grep -E "^merge:|^host model:" "$OUT/host_merge.log"
  fi
  cd "$R"
fi

# ---- board: the benchmark and its baseline -------------------------------------------------
if [ $BASELINE = 1 ]; then
  drop; step "run_remote_perf"
  (cd inference-scheduler && $P run_remote_perf.py --config perf_config.json --json "$OUT/perf.json") > "$OUT/perf.log" 2>&1; rc $?
  grep -E "OVERALL|FAIL" "$OUT/perf.log" | tail -2
  step "perf baseline (kernels, --record)"
  $P .claude/skills/perf-regression/scripts/compare_perf.py --run "$OUT/perf.json" --bitstream-id "$BID" --record > "$OUT/compare.log" 2>&1; rc $?
  grep -E "verdict|compared|REGRESSION|improved|recorded" "$OUT/compare.log" | head -8
fi

# ---- the server back, the simulator check ---------------------------------------------------
if [ $DEPLOY = 1 ] || [ -n "$MODELS" ]; then
  step "chat server up$([ $DEPLOY = 1 ] && has bert && echo ' (BERT project regenerated)')"
  $P demo/chat/deploy.py $([ $DEPLOY = 1 ] && has bert && echo --regenerate) > "$OUT/deploy.log" 2>&1; rc $?; tail -1 "$OUT/deploy.log"
fi
if [ $SIMULATE = 1 ]; then
  cd "$R/inference-scheduler"
  profiles=""
  for spec in smollm2-135m-instruct:135m smollm2-360m-instruct:360m smolvlm-256m-instruct:vlm piper-lessac-medium:piper; do
    [ -f "$OUT/${spec#*:}.json" ] && profiles="$profiles ${spec%:*}=$OUT/${spec#*:}.json"
  done
  [ -f "$R/demo/bert_squad/build/results.json" ] && profiles="$profiles bert=$R/demo/bert_squad/build/results.json"
  step "simulate (host)"; $P perf_calibrate.py simulate --profile $profiles > "$OUT/simulate.log" 2>&1; rc $?
  grep -E "project|prefill|decode|vision|encode|chunk|inference" "$OUT/simulate.log" | head -24
  cd "$R"
fi
step "window done"
