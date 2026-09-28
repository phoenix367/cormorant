# Tactics plan — planning from calibrated performance models

Date: 2026-09-28.  Status: **proposed, not started.**

Goal: an **optional** planning mode (`--plan`) in which the scheduler
chooses each node's implementation (its *tactic*) and the issue order
from performance models of this bitstream, instead of the cost model's
defaults, preference rules and orders written by hand in the frontends.

- **The models come from measurements taken once per bitstream** (a
  calibration campaign on the board).  Planning itself runs entirely on
  the host, in seconds, with no board access at deploy time.  This works
  because the system is deterministic (§1).
- **Without `--plan`**, the scheduler behaves as today and generated
  projects are byte-identical.
- **Every tactic is bit-identical.**  Both kernels multiply Q8.8 exactly
  and accumulate in `ap_fixed<32,16>`, and alternative host code must pass
  the same bit-exact gates.  A choice is therefore judged on speed only.
  The existing gates (simulation == study, host emulation, board logits)
  stay the correctness guard and are unchanged.

## 0. Why

| evidence | where |
|---|---|
| The ConvKernel cycle model is 1.06–1.30× optimistic on plans whose accumulator holds fewer output rows than the line buffer, and 0.8–1.0× on the rest.  The 1024-row vision linears were found by hand with a 27-case board sweep and fixed with a rule. | CHAT_PLAN §24 |
| At 512 rows the model rates kw 4 and kw 6 equal; on the board kw 4 is 6–18 % slower (q 7.93 vs 6.93 ms, fc1 31.2 vs 29.3, fc2 29.7 vs 25.3). | CHAT_PLAN §24 |
| The rule picks 2 × 512 rows (164.5 ms per layer); the board's best is 256 / 128 rows (158.6 ms). | CHAT_PLAN §24 |
| MatmulKernel's GEMV cost is not board-calibrated yet. | `src/cost_model.py` |
| Host ops have no cost model at all; attention issue orders are written by hand (`llama.py`, `vit.py`). | §2 |
| The CPU still waits ~3.4 ms per head for P.V to free ConvKernel before the next q.Kᵀ: up to 0.49 s of SmolVLM's 3.9 s per image. | CHAT_PLAN §24 |

## 1. Determinism: why models measured once are enough

The kernels are fixed-function HLS pipelines with no data-dependent
control flow.  Their run time is a function of the call's register values
(geometry) and the bitstream.  The board measurements agree:

| same work, repeated | results | spread |
|---|---|---:|
| ConvKernel fc1, 1024 rows, kw 6 (two perf-runner sessions, rebuilt, 13 min apart with the chat server running in between) | 100.6372 / 100.6360 ms | 0.001 % |
| ConvKernel q, 256 rows, kw 6 (same two sessions) | 3.4387 / 3.4386 ms | 0.003 % |
| ConvKernel fc2, 256 rows, kw 6 (same) | 12.5535 / 12.5521 ms | 0.01 % |
| ConvKernel fc1, 256 rows, kw 6 (same) | 15.4487 / 15.4523 ms | 0.02 % |
| SmolVLM `llm_image`, kernels + host ops, two *different* images, one build | 3896.7 / 3895.3 ms | 0.04 % |
| SmolVLM decode step (host-heavy), 32 steps | 100.7–101.5 ms | ~0.8 % |

So a kernel call measured once is known exactly, and the run time does not
depend on the data (two different images take the same time).  A model
fitted to a well-chosen set of calls predicts the rest.  Planning can then use only
the models, on the host.  What is less deterministic, and how the plan
handles it:

- **Memory contention.**  All kernels share one 128-bit HPC port, and host
  ops stream DDR while kernels run.  Single-call timings do not include
  this.  A contention term is fitted from whole-graph profiles (§4.5), and
  small predicted gains are ignored (§4.1).
- **Host ops.**  Linux scheduling and caches add ~1 % spread.  Host-op
  models carry this as an error bar.
- **Extrapolation.**  A call outside the calibrated range gets today's
  choice, and the report lists it (§4.1).

Calibrating once and planning on the host is much cheaper than measuring
at deploy time.
- **Board time:** one campaign per bitstream, instead of a sweep and
  chat-server downtime at every deploy.
- **Order search needs a simulator anyway:** it evaluates thousands of
  orders, far more than could ever be run on the board.

## 2. How the scheduler decides today

| decision | where | how |
|---|---|---|
| node order | `OnnxGraph.nodes` | ONNX file order; passes replace nodes in place and never reorder |
| attention order (LLM prefill, ViT) | `llama.py`, `vit.py` | written by hand (e.g. softmax(g), pv(g), qk(g + 2)) |
| issue / wait | `_compute_event_stream` (`codegen/_core.py`) | one pass over the list: wait for producers still on a lane, wait for the node's own lane if busy, then start (non-blocking) or run a host op (blocking) |
| MatMul engine | `matmul_lowering.lower_matmuls` | ConvKernel when the model says ≥ 10 % faster (`LOWER_MARGIN`) |
| conv geometry (kw, out_w) | `matmul_lowering.conv_plans` | cheapest by `cost_model.conv_cycles`; `_MIN_OUT_W` rule |
| row split | `conv_plans` | rule: only when every one-call plan is accumulator-limited |
| GEMV path | `matmul_gemv.py` | cost model, single-row MatMuls |
| shared weight layout | `matmul_conv_kw` / `matmul_conv_kws` (`llm_entries.py`) | pinned so prefill buckets and decode share one image |
| attention kernel widths | `choose_qk_kw` (`llama.py`, `vit.py`), `pv_kw` | cost model / fixed |
| host threads and grain | `INFERENCE_HOST_THREADS` (4), `host_row_grain` | fixed |
| rewrites | `fuse_act`, `s2d_stem`, `fuse_patterns` | CLI rules |

The DAG (`src/schedule.py`) has producer → consumer edges only.  Persistent
states are "external" and nodes that write them in place (`state_writes()`,
e.g. `VitAttnPrep` writing the vision K / V caches that all 12 layers share)
add no edges; today their ordering is safe only because the list order is
fixed and the next writer depends on the last reader through real data.

## 3. Terms

- **Tactic** — one implementation of one node that yields the same bits:
  engine, geometry, number of calls, chunking of a host op.
- **Call signature** — the kernel plus the register values that set its
  run time; for the four kernels, exactly `run_remote_perf.py`'s
  `_CASE_FIELDS` without `iters` (buffer offsets excluded: every call start
  is 16-byte aligned).
- **Performance model** — per bitstream: the exact measured calls, a fitted
  model per kernel for the rest, host-op models, and a contention term.
  Stored in one versioned data file.
- **Plan** — the chosen tactics plus the issue order, host-op chunks
  included.

## 4. Design

### 4.1 Tactics and the planner

`src/tactics.py`: per node kind, `candidates(node) -> [Tactic]`.  A tactic
carries its kind and parameters, the call signatures it issues (with
counts), the weight image it needs (if any) and the model's estimate.

- **MatMul:** today's `conv_plans` (row splits included), MatmulKernel
  tiled, and GEMV when `n == 1`.
- **Attention:** q.Kᵀ / P.V kernel widths.
- **Host ops:** chunk count for row-parallel ops (§4.4).

The legality checks stay hard filters (`ineligible_reason`, kernel
bounds).  Under `--plan`, the preference rules (`LOWER_MARGIN`, the
row-split trigger, `_MIN_OUT_W`) are replaced by the performance model.
Two guards stay:

- **Minimum gain:** a candidate replaces today's choice only when it is
  predicted at least 3 % faster, so noise causes no churn.
- **Calibrated range only:** a node whose candidates fall outside the
  model's calibrated range keeps today's choice.  The plan report lists
  it, so the next campaign can add it.

**Shared weights choose jointly.**  All nodes that read one constant B
(prefill buckets, decode GEMV, other entries) must agree on its image.  The
group takes the layout with the least weighted sum of predicted times; the
weights are per entry (`--entry-weights decode=64,prefill_256=1,…`,
default 1).

### 4.2 Performance models

`inference-scheduler/perf_models/<platform>/<bitstream-id>.json`, versioned
in git and produced by the campaign (§4.3).  It holds:

- **Header:** platform, bitstream id (hw submodule commit or bitstream
  hash), PL clock, AXI bus width, campaign date.
- **Exact entries:** measured ms per call signature, e.g.
  `"ConvKernel:1,128,96,48,512,1,6,1,6,1,1,0,0,0,0": {"ms": 6.927}` (the
  512-row q projection of SmolVLM's vision encoder, CHAT_PLAN §24).  With
  deterministic kernels these are exact for that call.
- **Kernel models:** for ConvKernel, MatmulKernel (tiled and GEMV),
  VectorOP and Pool.  Each keeps the structure of the existing analytic
  model, which already splits a call into terms: for ConvKernel the sweep,
  weight fill, phases 1 / 3, loads, chunks, M-groups.  The measurements
  fit each term's coefficient, plus terms for regimes the model misses.
  The accumulator-limited chunk found in CHAT_PLAN §24 is the first known
  one.  The model stays interpretable, and its residuals show a new
  missing term.  Each model records its calibrated range, i.e. the extent
  of every parameter.
- **Host-op models:** per op kind and thread count,
  `t = a + b·rows + c·elements` (softmax, LayerNorm, GELU, attention prep,
  residual add, RMSNorm, SiLU·mul, the LLM attention helpers, …).  Each
  carries its error bar.
- **Contention:** one factor per (kernel, concurrent host-op kind) pair,
  fitted from whole-graph profiles (§4.5); 1.0 until then.
- **Call overhead:** a kernel start plus completion poll through UIO,
  measured.

`--perf-model FILE` overrides the default path.  A file for another
bitstream is rejected under `--plan`.

### 4.3 Calibration campaign (once per bitstream)

`inference-scheduler/perf_calibrate.py`: board time roughly 30–60 minutes,
with the chat server stopped (the stop, drop_caches + compact_memory and
restart sequence of CHAT_PLAN §23 / §24, automated).

1. **Shipped models:** every call signature that any candidate tactic of
   the models we ship would issue — BERT, SmolLM2-135M / 360M, SmolVLM,
   ResNet-18, MobileNet v1 / v2, MNIST.  The uncalibrated model prunes each
   node to today's choice plus the top K (default 4).  These become exact
   entries.
2. **Space-filling set:** a grid per kernel over its geometry parameters
   (out_ch, in_ch, kw, out_w, out_h, M-group and chunk regimes, n / k / m
   for MatmulKernel, sizes and strides for VectorOP / Pool).  It fits the
   models and exposes structural gaps.  A quarter of the points is held out
   for validation.
3. **Host ops:** the generated C helpers cut out into a board harness (the
   method of CHAT_PLAN §24), swept over shapes and thread counts 1–4.
4. **Determinism check:** each point is measured in two separate sessions.
   A spread above 0.5 % marks the point as noisy; it is kept, with its
   spread.

Needed in `run_remote_perf.py`: `--json OUT` (mean, min and stdev per case)
and cases for calls at offsets.

**Validation report** (in the model file and on stdout):

| check | target |
|---|---|
| held-out error, kernels | ≤ 3 % |
| held-out error, host ops | ≤ 5 % |
| residuals per regime | no regime biased beyond these bounds |

A missed target blocks the file from `--plan` until it is fixed (a missing
term or more points).

### 4.4 Simulator and order search (host only)

`src/codegen/timing.py` replays `_compute_event_stream` with the model's
durations.  It returns the makespan, per-lane busy time, the CPU's waits
(which node, waiting for which lane) and the critical path.

- **Semantics** are exactly the event stream's: one CPU issues in order; a
  kernel start does not block; starting on a busy lane blocks the CPU; a
  host op blocks the CPU; a node waits for its producers' lanes.
- **Report:** a "where the CPU waits" section in the scheduler report,
  available with `--plan` or `--plan-report` (report only, no changes).

Order search, under `--plan`:

- **Prerequisite:** state edges in the DAG.  A state read after a write is
  RAW, a write after a read is WAR, two writes are WAW.  They come from
  `state_writes()` and the state inputs, in today's list order.  Without
  them no order may change.
- **Search:** list scheduling from today's order, prioritised by the
  longest remaining path; then local moves (bring a kernel start earlier,
  swap independent neighbours).  A move is kept only if the simulated
  makespan drops and the pool stays within `--pool-budget-mib` (default:
  today's pool size, so CMA budgets such as the 360M's 740 MiB don't grow).
  Ties are broken by node index, so the result is deterministic.
- **Host-op chunking tactic:** a row-parallel host op (`VitAttnSoftmax` by
  query block, `LlmAttnSoftmax`, LayerNorm by rows) can become k chunk nodes
  over row ranges.  This lets the CPU start the next kernel between chunks,
  which is what hides the P.V wait.  Bit-exact, since rows are independent;
  the numpy reference runs per slice.
- **Hand orders** in `llama.py` / `vit.py` become the starting order, and
  the search result must never simulate slower than them.

### 4.5 Verification (optional, board)

- **`--plan-verify`:** after deploying a planned project, one profiled run
  in serial profiling mode (`INFERENCE_PROF_SERIAL=1`: every kernel waited
  on right after its start, so each bracket is the call's run time; today's
  brackets span start → drain).  It reports predicted vs measured per node
  and in total.
- **What it feeds:** mismatches and out-of-range calls go into the
  calibration data for the next campaign.  Whole-graph profiles fit the
  contention factors.
- **Never used to decide a plan:** planning stays host-only and
  reproducible.

### 4.6 Turning it on

| entry point | switch |
|---|---|
| `inference_scheduler.py`, `OnnxGraph` | `--plan` / `plan=True`, `--perf-model FILE`, `--plan-report`, `--pool-budget-mib`, `--entry-weights` |
| `demo/chat/scripts/generate_llm_project.py` | `--plan` (passed to every entry) |
| `demo/chat/deploy.py --regenerate` (BERT) | `--plan` |
| demo `deploy_and_run.py` scripts | `--plan`, or `"plan": true` in the demo's config JSON |
| chat server configs | `"plan": true` per backend, used when its project is (re)generated |

The generated project records the plan: the model file's id and the chosen
tactics and order, in the report and the `inference.c` header.  A build can
be traced to the models it was planned with.

## 5. Phases

| phase | work | done when |
|---|---|---|
| **T0** prerequisites | `--plan` plumbing (accepted, no effect yet); `run_remote_perf.py --json` and offset cases; one call-signature function shared by the lowering and the runner; state RAW / WAR / WAW edges in `Dag` + tests (the vision caches, LLM K / V states); bitstream id | all projects byte-identical; new DAG tests pass |
| **T1** calibration campaign + kernel models | `perf_calibrate.py`; fitted ConvKernel / MatmulKernel / VectorOP / Pool models with exact entries; model file for the current bitstream | held-out error ≤ 3 %; determinism check passes; the model reproduces the CHAT_PLAN §24 numbers (1024-row penalty, kw 4 vs kw 6 at 512) |
| **T2** planned tactics | tactic registry for MatMuls and attention widths under `--plan`; joint choice for shared weights | without `--plan` byte-identical; with it, SmolVLM's vision linears go to 256 / 128 rows (~6 ms per layer); on the board no model gets slower (BERT, prefill 16 / 64 / 256, decode, `llm_image`, ResNet-18); gates bit-exact |
| **T3** host-op models + simulator + report | host-op campaign; `codegen/timing.py`; `--plan-report` | predicted vs measured within 10 % on BERT, SmolVLM `llm_image`, SmolLM2 prefill 256, ResNet-18 |
| **T4** order search | list scheduling + local search under the pool budget; host-op chunking (softmax first) | simulated ≤ hand order for every model; board: SmolVLM `llm_image` recovers most of the ~0.49 s P.V wait; no regression elsewhere; gates bit-exact |
| **T5** verification and more tactics | `--plan-verify`, serial profiling mode, contention fit; host threads / grain and conv rewrites (s2d stem, pointwise) as tactics | predicted vs measured within 5 % on the same four workloads |

Each phase gets a section here with its board numbers, like CHAT_PLAN.

## 6. Risks

| risk | mitigation |
|---|---|
| A structural gap in a kernel model (a regime it gets wrong, like the accumulator-limited chunks) | The space-filling set and per-regime residuals in the validation report; exact entries for every call of the shipped models |
| A call outside the calibrated range | Keep today's choice for that node; list it in the report |
| Single-call timings miss memory contention | The 3 % minimum gain; contention factors from profiles (T5); the T3 check within 10 % |
| A stale model after a new bitstream | Keyed by bitstream id; rejected under `--plan` |
| Reordering breaks a state hazard | T0 edges first; unit tests with tiny models that share a state between layers; host emulation of planned projects |
| Reordering grows the pool past CMA | `--pool-budget-mib`, default today's pool size |
| One layout for weights shared across entries | Joint choice per weight group (§4.1) |
| Test time | Tiny models in unit tests; the existing gates for the big models |

## 7. Unchanged

Numerics and every gate, the kernel IPs and bitstream, the C API, the
generated projects without `--plan`, and the analytic cost model: without
`--plan` it decides as today, with it it gives the kernel models their
structure, and it stays equal to the conv-cycle-model skill script
(`test_matmul_on_conv.py`).

## 8. Open questions

- **Entry weights for joint choices:** decode steps per prefill differ by
  use (a chat answer is ~50–200 tokens).  Proposed default: decode 64,
  prefill 1.
- **The shipped model list** for the campaign's exact entries (§4.3):
  proposed, every model with a demo in `demo/`.
- **Who runs the campaign:** proposed, a `make` target / script run
  together with the bitstream build and board bring-up, its model file
  committed with the hw submodule bump.
