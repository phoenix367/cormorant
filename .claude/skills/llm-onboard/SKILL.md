---
description: Take a Llama-family decoder that passed model-study (GO) all the way to a served chat model on the KV260 — pinned checkpoint and calibrated exponents (llm_calibrate.py), the multi-entry library project (generate_llm_project.py), the bit-exact gates on the host (llm_sched_check.py, llm_host_emu.py) and on the board (llm_board.py, llm_lib_check.py), a chat-server backend (kv260_chat_server.py / deploy.py) and the write-up — with the steps a new checkpoint reuses unchanged separated from the ones that need code, and the extra steps of a vision-language model like SmolVLM. Use after a GO verdict, when asked to add / onboard / serve a new LLM on the board, or to rebuild and redeploy an existing chat model's library.
allowed-tools: Bash Read Write Edit
---

# llm-onboard

`model-study` decides GO / NO-GO; this is what follows a GO for a Llama-family
decoder.  SmolLM2-135M (CHAT_PLAN §11–§16, §19), SmolLM2-360M (§20, §21) and
SmolVLM-256M (§22–§24) took this path — read §20 first, it is the closest
template.  Order: **intake → pin + calibrate → generate → host gates → board
gates → serve → document**; each step has a gate, do not start the next on a
failed one.

Commands run from the repo root:

```bash
PY=inference-scheduler/.venv/bin/python     # generation, gates, board, deploy
SPY=.venv-export/bin/python                 # study env (numpy, tokenizers, torch, transformers; Pillow for VLMs)
# missing: python3 -m venv .venv-export && PIP_CONFIG_FILE=/dev/null .venv-export/bin/pip install \
#   --extra-index-url https://download.pytorch.org/whl/cpu -r demo/chat/scripts/requirements-study.txt
N=<name>          # llm_models.json key = demo/chat/assets/<name>/ = --model-name = served model id
T=<tag>           # N without "-instruct", "-" and "." -> "_" (smollm2-360m-instruct -> smollm2_360m)
BOARD=root@192.168.100.8; KEY=~/.ssh/kv260-testkey    # ssh block of demo/bert_squad/bert_squad_config.json
```

SmolLM2-135M keeps legacy names: `build/llm_project`, `lib/libsmollm2.so`,
`/root/smollm2_weights`, study files in `assets/study/` itself; every other
model gets `build/llm_project_$T`, `lib/lib$T.so`, `/root/${T}_weights`,
`assets/study/$N/` (`llm_board.board_paths`, `llm_study.study_dir`).

## 0. Intake — what is reused, what needs code

Preconditions: a GO section in CHAT_PLAN (model-study route A; `llama_fit.py`
eligible — plain Llama block, one `model.safetensors`, pool < ~920 MiB).
After step 1's `add` (it downloads the checkpoint):

```bash
python3 .claude/skills/llm-onboard/scripts/onboard_check.py demo/chat/assets/$N   # stdlib, seconds
```

It checks what `llama_fit.py` does not: every linear's K ≤ MatmulKernel
`max_k` (4096), a single safetensors file, the tokenizer pipeline the board's
`smollm2_tokenizer.py` accepts, `<|im_start|>` = 1 / `<|im_end|>` = 2,
SmolLM2's chat template, tokenizer.json byte-identical to 135M's, tokenizer
size = `vocab_size`, the pin, and the names the tools derive from `$N`.
Exit 0 = the table's left column applies; exit 1 = do the CODE items first.

| step | SmolLM2-text checkpoint (exit 0: SmolLM2 tokenizer + ChatML, e.g. the 360M) | otherwise |
|---|---|---|
| 1 pin / calibrate / study | unchanged | `add` fetches a fixed 8-file list; `llm_study.py` hard-codes ChatML (`chatml()`), sink id 1, EOS 2 |
| 2 generate | unchanged (`--assets --model-name`) | K > 4096: split K in `src/llama.py`; shards: `load_safetensors` |
| 3–4 gates | unchanged | `llm_project.tokenize_prompts()` / `second_turn_ids()` always tokenize with 135M's tokenizer into shared caches (`assets/study/phase3_prompt_ids.json`, `phase5_second_turn_ids.json`); `llm_lib_check.py` hard-codes 135M ids and `context == 1024` |
| 5 serve | **needs a backend entry (code, §5B)** — or repoint an existing slot (config only, replaces that model, §5A) | `smollm2_tokenizer.py` (raises on another pipeline), `chatml.py`, the backend's `<|im_start|>` sink |

Not possible on this path today (model-study route C, or a bitstream / CMA
change): non-Llama blocks (biases, LayerNorm, GELU MLP, partial rotary,
`rope_scaling`, sliding window, MoE — the frontend ignores unknown config
keys silently, so trust `llama_fit.py`, not a successful generation); pools
above ~920 MiB (≈ 440 M int16 params with the KV caches); K > 4096; a context
other than 1024 (`llm_lib_check.py` requires it); another VLM
(the `vlm_*` scripts and the `smolvlm` backend are wired to SmolVLM-256M).

## 1. Pin + calibrate (`llm_calibrate.py`, stdlib; study steps in `$SPY`)

model-study's route A usually did `add` / `fetch` / `validate` /
`calibrate --record` already — then confirm with `check` and record the
shipped-policy metrics with `study --record`.

```bash
python3 demo/chat/scripts/llm_calibrate.py add $N --repo <org/Repo> [--revision <commit|tag|branch>]
python3 demo/chat/scripts/llm_calibrate.py fetch $N          # texts copied from another model (hash-checked) or fetched
python3 demo/chat/scripts/llm_calibrate.py validate $N       # float64 numpy vs torch f32 (135M ~2 min)
python3 demo/chat/scripts/llm_calibrate.py calibrate $N --record   # formats + provenance, hash stored
python3 demo/chat/scripts/llm_calibrate.py check $N          # recompute, install nothing
python3 demo/chat/scripts/llm_calibrate.py study $N --record # bf16 / p12 / p12+mix vs float (360M ~40 min)
```

- `add` records the resolved commit, license and SHA-256 of `config.json,
  generation_config.json, model.safetensors, tokenizer.json,
  tokenizer_config.json, special_tokens_map.json, vocab.json, merges.txt`
  (a missing one is an unhandled HTTP error) and policy `pow2+sink+p12` with
  no hash.  `HF_ENDPOINT` selects a mirror.  Fetch 135M 58 s, 360M 135 s.
- **Gates:** `validate` prints `chat template + tokenizer: 15/15 prompts
  identical`, greedy `12/12`, max |logit diff| ~1e-4 (360M 1.5e-4).
  `calibrate --record` → `RECORDED`, then `check` → `REPRODUCED` (exit 0;
  1 = differs / stale, 2 = error).  `study` metrics must repeat the model-study
  verdict (360M: top-1 0.978 / 0.994 / 0.976, KL 0.0018, ppl 12.394 vs float
  12.373); `--record` stores them in `llm_models.json`.
- **Outputs:** `demo/chat/assets/study/$N/formats_pow2+sink+p12.json` (135M:
  `assets/study/`), `…/$N/formats_pow2+sink+p12.provenance.json` (tracked),
  `…/$N/shipped/{results.json,generations.txt}`.  Git: only
  `llm_models.json` and the provenance.  `all $N [--validate] [--study]
  [--record]` chains fetch → validate → calibrate → study.
- Another CPU may round a sink value differently: `calibrate` then leaves
  `formats_*.new.json` and lists the differing exponents; `--force` installs
  it — a different (not wrong) library, gated against its own emulation.

## 2. Generate the library project (host RAM!)

```bash
$PY demo/chat/scripts/generate_llm_project.py --assets demo/chat/assets/$N --model-name $N \
    [--plan [--perf-model FILE] [--plan-report] [--pool-budget-mib MIB] [--entry-weights decode=64,head=64]]
# -> demo/chat/build/llm_project_$T/ (135M: build/llm_project): weights/*.dat, project.json, layers.json
```

- The model name is the checkpoint directory's name: `--assets` alone is
  enough (`--model-name`, if given, must match it — `--allow-name-mismatch`
  overrides — and `--model-name` alone looks for `assets/<name>`).  It becomes
  `llm_model_name()` = the served id and names the out dir, `lib/lib$T.so` and
  `/root/${T}_weights`.  The out dir is wiped (except `weights/` with
  `--no-weights`), but only if it is empty or holds this model's project —
  another model's project or other files need `--force`.
- Defaults are the shipped design — keep them: buckets `16,64,256`, context
  1024, `--prefill-engine conv`, `--prefill-attn fpga` (policy
  `pow2+sink+p12+mix`).  `--plan` is optional and bit-identical (needs
  `inference-scheduler/perf_models/kv260/<bitstream-id>.json` of the board's
  bitstream; gains ≤ ~1 %, TACTICS_PLAN §9).
- **Host RAM peak** (dev PC 46 GB): 135M 3.3 GB / ~60 s, 360M 8.2 GB /
  ~2.5 min, SmolVLM 3.6 GB / 90 s (CHAT_PLAN §25; about 23 bytes per
  parameter); `llama_fit.py` interpolates.  `llm_sched_check.py`,
  `llm_host_emu.py` and the host side of `llm_board.py` rebuild the
  simulation and hold the study model too (`llm_sched_check.py` on 135M:
  5.1 GB); run one such job at a time and check `free -g` first.
- **Gate:** the summary lines — pool MiB (weights / KV caches /
  intermediates) close to `llama_fit.py`'s estimate and < ~920; no
  `missing driver files` warning (drivers come from
  `demo/bert_squad/bert_squad_config.json` `local.driver_dirs`; never generate
  while a conv synthesis rewrites `build*/…/drivers/`).  360M: pool 740.2 MiB,
  weight files 818 MB, 224 / 224 MatMuls of `prefill_64` / `prefill_256` on
  ConvKernel (kw 4) and of `prefill_16` on MatmulKernel, 225 decode on GEMV.

## 3. Host gates — bit-exact before any board time

```bash
$PY demo/chat/scripts/llm_sched_check.py --assets demo/chat/assets/$N \
    --save demo/chat/assets/study/$N/gate2.json        # 360M: 23 min, 27 GB
$PY demo/chat/scripts/llm_host_emu.py --project demo/chat/build/llm_project_$T \
    --study-json demo/chat/assets/study/$N/gate2.json [--incoherent]
```

- `llm_sched_check.py`: scheduler simulation == study emulation of
  `pow2+sink+p12+mix`, every logits vector, for `factual, summarise,
  multi-turn` + a second `factual` turn at position 69, 32 greedy decode steps
  each.  **Gate:** `3/3 prompts bit-exact`, exit 0.  `setup` prints `weights
  saturated` (0 so far).  Keep `--prompts / --second-turn / --decode` at the
  defaults — `llm_board.py` uses the same ones against `gate2.json`.
- `llm_host_emu.py`: the generated `inference.c` + `llm_api.c` +
  `llm_bench.c`, compiled `-Werror` against software kernels.  **Gate:**
  `HOST EMULATION: logits bit-exact with the simulation`, re-open identical.
  `--incoherent` (separate CPU / DDR copies) catches a missing cache sync.
  Mandatory when frontend, host-op or codegen code changed (§20 lists no
  host-emulation run for 360M; 135M and SmolVLM ran it).

## 4. Board — gate, install, memory

The board is shared: **one board job at a time**; `llm_board.py` (like every
board tool) holds the per-board lock `/tmp/kv260-board-<host>.lock`
(`src/remote/lock.py`; a path in the BERT config's `board_lock` overrides it)
for its session.  The running
chat server owns the FPGA and the CMA — stop it first.

```bash
$PY demo/chat/deploy.py --stop
(cd inference-scheduler && .venv/bin/python -m src.remote.locked --host 192.168.100.8 -- \
    ssh -i $KEY $BOARD 'sync; echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory;
    grep -E "CmaFree|MemAvailable" /proc/meminfo; df -h /root')
$PY demo/chat/scripts/llm_board.py --project demo/chat/build/llm_project_$T \
    --study-json demo/chat/assets/study/$N/gate2.json --reopen \
    [--profile] [--decode-at 32,256,1000] [--out /tmp/${T}_board.json]
```

- It uploads the sources, syncs changed `weights/*.dat` to `/root/${T}_weights`
  (135M `/root/smollm2_weights`), builds `-j1` under a memory guard (starts
  only with MemAvailable ≥ 1.5 GB, kills make below 400 MB or PSI > 50),
  **installs** `<remote.dir>/lib/lib$T.so` (`remote.dir` of
  `demo/chat/chat_config.json`, else `/root/kv260_chat`), runs `llm_bench`,
  downloads the logits and replays them on the host simulation.
- **Gate** (read the lines, the exit code covers only the logits):
  `[factual] logits bit-exact 33/33, greedy tokens == study emulation` for
  all 4 entries (`factual`, `factual/turn2`, `summarise`, `multi-turn`);
  `re-open: {… identical: true}`; `lib check (ctypes)`: `only_llm_exports`,
  `chunked_prefill_identical`, `threads_identical`, `reopen_identical` true,
  `vocab` = V, `context` 1024; last line `BOARD GATE: logits bit-exact`.
  lib check's `"ok": false` from the `abs(CmaFree after close − before open)
  < 4 MB` rule alone is page cache (missed by 0.4–10 MB in §19 / §20 / §23),
  not a leak.
- Numbers for the write-up: `decode … ms/token`, `prefill 16/64/256`,
  `llm_open … ms, CmaFree A -> B kB (C MB)` — cold when the weights are not
  in the page cache (after a reboot or the drop above: 360M 58.8 s, 135M
  21 s), warm right after the upload and in the `--reopen` / lib-check
  cycles (360M 1.5–1.7 s); `--decode-at` per position; `--profile` per-kind
  breakdown (`--out` keeps `profile_layers` for `perf_calibrate.py host /
  simulate --profile`).
- The gate run already installed the library.  To (re)install without the
  bench: `llm_board.py --project demo/chat/build/llm_project_$T --install-only`,
  `[--remote-dir DIR] [--weights-dir DIR]` for a second install beside an
  existing one (defaults: `chat_config.json` `remote.dir`,
  `/root/${T}_weights`).
- Reference (360M, bitstream `1d28630fbfa4`): decode 254 / 315 ms at 32 /
  1000, prefill 0.58 / 1.04 / 3.08 s, CMA used 736–740 MB.

**CMA budget** (`cma=1000M`, ~954 MiB usable, idle CmaFree ~1011 MB, less
with page cache in CMA).  Served pools: BERT 216, SmolLM2-135M 286,
SmolLM2-360M 740, SmolVLM 495 MiB.  The backend's `cma_mb` = measured CMA
drop + margin (286 → 330, 740 → 760, 495 → 540); `--resident auto` evicts
least-recently-used models while CmaFree < `cma_mb` + 32 and retries a failed
load once after evicting the rest — a pool that fits beside nothing swaps
with every model (warm swap 1–2 s, cold from SD card 21–59 s).  An
`xclAllocBO` failure with enough CmaFree is page migration: drop caches +
compact memory (command above) and retry; do the same before a deploy so
`auto` can keep more models resident (§16.5).

## 5. Serve it

**A. Zero code — repoint a slot** (how 360M was first served, §20
"Deploying it"; replaces that slot's model).  In `demo/chat/chat_config.json`
set the `smollm2_360m` (or `smollm2`) block: `lib` →
`<remote.dir>/lib/lib$T.so`, `tokenizer` → `assets/$N/tokenizer.json`,
`cma_mb`, `weights_dir` null (the build's).  The server serves the
library's `llm_model_name()` (= `$N`); `model_id` overrides it.

**B. A backend of its own** — the `smollm2-360m` pattern (commit 693cfb4);
`smollm2_backend.py` and the test helpers (`fake_llm_lib(model_name)`) need
nothing — the id and fingerprint come from the library:
- `demo/chat/kv260_chat_server.py`: docstring (backends, residency CMA
  line, usage); a tuple like `SMOLLM2_360M = ("smollm2-360m",
  "smollm2-360m-instruct")`; its branch in `build_backends()` →
  `smollm2_backend(args, llm_engine(args, args.llm_X_lib, args.llm_X_weights),
  args.llm_X_tokenizer, args.llm_X_cma_mb, args.llm_X_model_id,
  default_model_id="$N")`; `--backend` choices and the unknown-backend message;
  an argument group `--llm-X-lib` (default `HERE/lib/lib$T.so`),
  `--llm-X-model-id`, `--llm-X-weights`, `--llm-X-tokenizer`
  (`_first_existing(HERE/$T/tokenizer.json, HERE/assets/$N/tokenizer.json)`),
  `--llm-X-cma-mb`.
- `demo/chat/deploy.py`: docstring; the block's defaults in `load_config()`
  (`lib` → `<dir>/lib/lib$T.so`, `weights_dir`, `model_id`, `cma_mb`,
  `tokenizer`); the tuple + `uses_llm_X()`; the `preflight()` entry (library
  and tokenizer); `upload_server()` tokenizer → `<dir>/$T/tokenizer.json`;
  `server_argv()` — its options, **and add `uses_llm_X(cfg)` to the
  condition that passes the shared `--llm-*` sampling defaults and
  `--llm-sampler-lib`**; `print_ready()` hints only ids starting `smollm2-`.
- `demo/chat/chat_config.json.example`: the block and `_backends_note`;
  your local `chat_config.json` (untracked): the block + `server.backends`.
- `demo/chat/tests/test_smollm2_backend.py`: a `TestTwoSizes`-style class
  (two `fake_llm_lib(model_name)` copies side by side, `build_backends`
  order / `cma_mb` / fingerprint, default ids and overrides, one id twice →
  `SystemExit`).  **Gate:** `cd demo/chat/tests && python3 -m unittest -v`
  (188 before, ~60 skip without tokenizers).

**Deploy and talk to it:**

```bash
$PY demo/chat/deploy.py --check-only          # preflight: library + tokenizer OK
$PY demo/chat/deploy.py                       # [--hold]: keep the board lock, follow the log
$PY demo/chat/deploy.py --status              # /health: loaded, cma_free_mb, loads / unloads
python3 demo/chat/chat.py --url http://192.168.100.8:8000/v1 --model $N --temperature 0 -v
curl -s http://192.168.100.8:8000/v1/chat/completions -H 'Content-Type: application/json' -d \
 '{"model":"'$N'","messages":[{"role":"user","content":"What is the capital of France?"}],
   "temperature":0,"repetition_penalty":1,"dry_multiplier":0,"loop_guard":false,"max_tokens":128}'
```

- `deploy.py` always generates (if missing), builds and uploads the BERT
  project too, whether or not `bert-squad` is in `server.backends` — the
  `demo/bert_squad` prerequisites must be in place.
- **Gate (API = study):** with the penalties off, the greedy answer equals the
  `pow2+sink+p12+mix` text of `[0] factual` in
  `demo/chat/assets/study/$N/shipped/generations.txt` (the server's default
  system prompt is the study's).  `chat.py` cannot switch the penalties off —
  use curl.  Then the §19 4-turn conversation at temperature 0 for the table
  (`kv260.cached_tokens` / `prefill_tokens` show the prefix reuse, `ttft_ms`,
  `decode_tok_s`), alternating with the other models (`/health` `loads` /
  `unloads`, swap times).
- `demo/chat/tests/board_gate.py --url http://192.168.100.8:8000/v1` tests
  only `bert-squad` (12 / 12 demo spans): run it when BERT is served, to show
  the swaps left it intact.  Leave the server as the board setup expects
  (`deploy.py`, or `--stop` before handing the board over).

## 6. Document (commit only when asked)

- `doc/plans/CHAT_PLAN.md`: a new section on §20's outline — model (repo @
  revision, license, architecture, assets size), numerics table (`study`),
  `validate`, formats (entries, saturation), project (pool split, host
  tables, weight files, generation time / RAM), gates (sched check, host emu,
  board), board table (decode at 32 / 256 / 1000, prefill 16 / 64 / 256,
  `llm_open` cached / cold, CMA), chat API turns, deploying it, residency.
- `demo/chat/README.md`: a row in the "What it serves" table.
  `demo/chat/doc/`: the model's guide (a section like "SmolLM2-360M-Instruct"
  in `SMOLLM2.md`, or a new file linked from the README) with the commands;
  in `DEPLOY.md` the backends table, the pool table under "The FPGA and the
  board lock" and the by-hand server command; the test count in the README
  and `DEVELOPMENT.md`.
- `README.md`: the results table (~line 32) and the supported-models table
  (~line 286); `doc/README.md`'s CHAT_PLAN line.  Pools are also quoted in
  `kv260_chat_server.py`'s docstring, `chat_config.json.example`, and
  model-study (`SKILL.md` §2.2, `llama_fit.py` `RESIDENT`).
- Commit set: code, tests, docs, `llm_models.json`, the provenance JSON.
  Never `chat_config.json`, `demo/chat/assets/`, `demo/chat/build/`.

## VLM (SmolVLM-256M) — what differs

```bash
$SPY demo/chat/scripts/vlm_study.py fetch       # pins: vlm_study_inputs.json (not llm_models.json); + COCO images
$SPY demo/chat/scripts/vlm_study.py validate
$SPY demo/chat/scripts/vlm_study.py formats     # text + vision formats -> assets/study/smolvlm-256m-instruct/ (~3 min)
$PY demo/chat/scripts/generate_llm_project.py --model-name smolvlm-256m-instruct   # assets implied; 90 s, 3.3 GiB (CHAT_PLAN §25)
$PY demo/chat/scripts/vlm_sched_check.py --text  # vision 2/2 images + text 2/2 prompts x 16 steps bit-exact
$PY demo/chat/scripts/vlm_host_emu.py [--incoherent]           # HOST EMULATION: logits bit-exact, re-open identical
$PY demo/chat/scripts/llm_board.py --project demo/chat/build/llm_project_smolvlm_256m --decode 16 --reopen [--profile]
```

- No `llm_calibrate.py` hash / `check` for the VLM formats (reruns are
  byte-identical, §23); `--prefill-attn fpga` is required.
- Board gate: image prompts `--images 39769,1268` (COCO), 2 × 17 / 17 logits
  vs the simulation (no `--study-json`); `llm_image` ~2.3 s at 250 MHz (3.4 s at 100 MHz, 3.9 s before the RTL ConvKernel).
- Serving: the `smolvlm` backend exists (`smolvlm_backend.py`, `idefics3.py`,
  `vlm_image.py`; `--vlm-*`; config block `smolvlm`, `cma_mb` 540); the board
  needs `python3-pil` (deploy preflight checks it).  API gate: 48-token
  greedy answers for COCO 39769 and 1268 equal to the study emulation;
  `chat.py --image FILE`.

## Rules

- Host RAM: one heavy host job at a time (generation, sched check, host emu,
  the board gate's replay); board: one job at a time, server stopped, under
  the board lock.
- Never skip a gate: board time only after `llm_sched_check.py` is
  bit-exact; serving only after the board gate.
- Every library served side by side must be generated after the kernels'
  last AXI-Lite register change (a stale library inherits registers such as
  `gemv_kw` from the previous run — board-deploy skill).
- Assets and projects stay untracked; don't run `validate_text.py
  --write-fixture` for another model (it rewrites 135M's tracked fixture);
  `SMOLLM_ASSETS=demo/chat/assets/$N` points `validate_text.py`,
  `e2e_check.py` and the float fake at another checkpoint.
- Do not commit or push unless the user asks.
