# CLAUDE.md — inference-scheduler

Python code-generator that parses an ONNX model and emits a complete C project
that runs inference on the Xilinx KV260 FPGA using up to four hardware kernels:
VectorOPKernel (element-wise), MatmulKernel (matmul/FC), ConvKernel (2-D conv,
incl. depthwise, and MatMuls lowered with swapped operand roles), and
PoolingKernel (2-D pooling). Ops no kernel implements (Softmax, LayerNorm,
Gelu, Transpose, Slice / Split copies, Gather, OneHot, Cast, SpaceToDepth and
the `axi.llm` Llama decoder, vision-encoder and text-to-speech ops) run as
host-CPU code inside `inference_run()`. Reshape-class ops are buffer aliases with no hardware call;
Gemm is decomposed to MatMul + Add at load time. Several graphs can share one
library and weight pool (multi-entry projects, `--entry`). The opt-in `--plan`
mode picks MatMul tactics and the issue order from the bitstream's measured
performance model (`perf_models/`, `../doc/plans/TACTICS_PLAN.md`).

The technical reference is `../doc/scheduler/INFERENCE_SCHEDULER.md`; the user guide is
`doc/USER_GUIDE.md`.

## Quick Start

```bash
cd inference-scheduler
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# Generate all test ONNX models (required before running tests)
.venv/bin/python test/gen_all_models.py      # runs every gen_*_models.py below
# or individually:
.venv/bin/python test/gen_test_models.py
.venv/bin/python test/gen_matmul_models.py
.venv/bin/python test/gen_mixed_kernel_models.py
.venv/bin/python test/gen_conv_models.py
.venv/bin/python test/gen_pool_models.py
.venv/bin/python test/gen_reshape_gemm_models.py
.venv/bin/python test/gen_mixed_all_kernels_models.py
.venv/bin/python test/gen_parallel_models.py
.venv/bin/python test/gen_bert_models.py
.venv/bin/python test/gen_llama_models.py    # imports ../demo/chat/scripts/llm_study.py

# Run all tests
.venv/bin/python -m pytest test/ -v

# Lint — same invocation CI uses; configuration lives in pyproject.toml.
# Run from inference-scheduler/ (the dot scopes ruff to the whole package,
# including the top-level run_remote_*.py / upload_bitstream.py scripts).
.venv/bin/pip install 'ruff>=0.15,<0.16'
.venv/bin/ruff check .

# Generate a C inference project
.venv/bin/python inference_scheduler.py test/models/mixed_ops.onnx --out-dir /tmp/out
```

## CLI

```
python inference_scheduler.py <model.onnx> [options]
python inference_scheduler.py --entry NAME=MODEL.onnx [--entry ...] [options]

Options:
  --entry NAME=MODEL.onnx    Multi-entry project (src/codegen/multi.py): one library,
                             inference_run_NAME() per graph, one weight pool; repeat
                             per entry, replaces the positional model
  --out-dir DIR              Output directory (default: ./<stem>_inference/,
                             ./multi_inference/ with --entry)
  --driver-dir DIR           Copy the drivers of every active kernel from this path
  --embed-large-weights      Inline all weights as C arrays (skip .dat files)
  --embed-large-expected     Inline all GT arrays in test_inference.c
  --no-report                Skip report.md
  --no-fuse-act              Keep Relu / Clip(0,6) as separate VectorOP calls
  --no-s2d-stem              Keep stride-2 Convs on <= 4 channels as they are
  --fc-conv {auto,always,off}
                             Fully-connected Convs (kernel = whole input, one output
                             pixel) as MatMul on MatmulKernel (default auto: >= 20 %
                             faster by the cost model; src/fc_conv.py)
  --no-fuse-patterns         No LayerNorm / GELU fusion, no constant-broadcast
                             normalisation
  --matmul-on-conv {auto,always,off}
                             Run MatMuls on ConvKernel with swapped operand roles
                             (default auto: where the cost model says it is faster)
  --no-matmul-on-conv        Same as --matmul-on-conv off
  --matmul-gemv {auto,always,off}
                             Single-row MatMuls on MatmulKernel's GEMV streaming path
                             (both read ports; default auto: where the cost model says
                             it is faster)

Planning (src/planning.py, ../doc/plans/TACTICS_PLAN.md; bit-identical results):
  --plan                     MatMul tactics and the issue order from the performance model
  --perf-model FILE          Model file (default: perf_models/kv260/<bitstream-id>.json of
                             the bitstream named in bitstream_config_kv260.json)
  --plan-report              Add the "Planning" section to report.md, change nothing
                             (--plan / --plan-report also write timeline.html: the predicted
                             execution as an interactive Nsight-style timeline)
  --pool-budget-mib MIB      --plan: pool limit for a new order (default: the unplanned pool)
  --entry-weights NAME=W,... --plan: entry frequencies for the kernel widths a Llama
                             project's entries share (llm_entries.entry_graphs, i.e.
                             generate_llm_project.py; default decode 64, head 64, others 1)
  --timeline-profile [PHASE=]FILE
                             a board profile (LAYERS_JSON: output, a demo results.json,
                             llm_board.py / tts_board.py --out): measured per-layer times
                             in timeline.html, next to the prediction
```

The CLI enables `fuse_act`, `s2d_stem`, `fuse_patterns`,
`matmul_on_conv="auto"`, `matmul_gemv="auto"` and `fc_conv="auto"`; the
`OnnxGraph` library defaults are `fuse_act=False`, `s2d_stem=False`,
`fuse_patterns=True`, `matmul_on_conv="auto"`, `matmul_gemv="auto"`,
`fc_conv="auto"`, `plan=None` (off).

## Preprocessing ONNX models — `simplify_onnx.py`

Most pre-trained ONNX models ship with a dynamic batch dim (`'N'`) and
trailing BatchNormalization layers that the scheduler can't consume
directly.  `simplify_onnx.py` is the one-shot fix:

```bash
# Pin batch=1, run onnxsim, fuse BN→Conv, write <stem>-simplified.onnx
python simplify_onnx.py model.onnx --batch 1

# Multi-input or non-batch dynamic dims need explicit shape(s) (repeatable)
python simplify_onnx.py bertsquad-12.onnx \
    --input-shape input_ids:0=1,256 --input-shape input_mask:0=1,256 \
    --input-shape segment_ids:0=1,256 --input-shape unique_ids_raw_output___9:0=1

# Custom output + post-save onnxruntime smoke test (random input)
python simplify_onnx.py model.onnx -o out.onnx --check

# Keep BN nodes for inspection (skips fuse_bn_into_conv; onnxsim may still fold them)
python simplify_onnx.py model.onnx --no-fuse-bn
```

Pipeline: `onnxsim.simplify(overwrite_input_shapes=…)` → optional
`onnxoptimizer.fuse_bn_into_conv` → `onnx.checker.check_model` → save.
Default output path is `<stem>-simplified.onnx` next to the input.  The
script prints a node-count delta with per-op-type changes highlighted —
e.g. resnet50-v1-12 collapses from 175 to 122 nodes with all 53 BNs
absorbed into the preceding Convs.

`--batch N` errors out on non-batch dynamic dims rather than guessing —
use `--input-shape NAME=D1,D2,…` for those.  The `*-simplified.onnx` and
`*_simplified.onnx` model files in this directory (local, not tracked) were
produced by (or can be regenerated with) this script; `resnet18-simplified-fused.onnx`
came from a different BN-fusion pipeline (see `doc/MODEL_PREPARATION.md` §1).

See `doc/MODEL_PREPARATION.md` for the full workflow, including handling
of unsupported ops that survive simplification (e.g. `BatchNormalization`
left in place, non-depthwise grouped Conv) and a worked example on
`resnet50-v1-12.onnx`.

## Generated Project Layout

```
<out_dir>/
├── CMakeLists.txt            INFERENCE_TARGET=BARE_METAL (default)|LINUX
├── include/
│   ├── inference.h           Public API: Data_t, size macros, DMA buffer API,
│   │                         init/run/deinit declarations, UIO instance defaults
│   ├── inference_prof.h      Per-layer profiler (-DINFERENCE_PROFILING=ON)
│   └── inference_ddr.h       DDR-traffic counters for the profiler
├── src/
│   ├── inference.c           Weight ROM arrays, run_*() helpers, kernel_wait(),
│   │                         host-op helpers, init/deinit/run bodies
│   ├── inference_buf.c       DMA buffer alloc/sync (Linux XRT BOs or bare-metal Xil)
│   ├── inference_prof.c, inference_ddr.c, inference_ddr_backend.h, ddr/zuplus_apm.c
├── test/test_inference.c     On-device test: ramp fill → run → compare vs GT
├── scripts/check_inference_setup.sh
├── driver/                   Kernel driver sources, flat (copied or stub README.md)
├── weights/                  External .dat files: weights > 4096 elems, host tables > 64 KiB
├── expected/                 External .dat files for large GT expected arrays (> 4096 elems)
└── report.md                 Model summary (unless --no-report; single-entry only)
```

## Source Layout

```
inference_scheduler.py   CLI entry point — ONNX → C project
simplify_onnx.py         CLI entry point — ONNX → ONNX (onnxsim + BN-fusion)
run_remote_tests.py      On-board correctness runner (SSH)
run_remote_perf.py       On-board kernel benchmark runner (SSH; bench_src/; --json OUT)
upload_bitstream.py      Load .bit + xclbin + .dtbo on the board (src/bitstream/)
perf_calibrate.py        Performance-model campaign for --plan: cases / run / fit / host /
                         simulate / all (--refine, --resume)
requirements.txt, pyproject.toml (ruff)
runtime/                 inference_prof / inference_ddr sources copied into projects
bench_src/               C benchmark project used by run_remote_perf.py; calib_runner.c
                         times perf_calibrate.py's kernel-call batches on the board
perf_models/<platform>/  <bitstream-id>.cases / .calib / .json (kernel model), host.json
                         (host-op model); perf_models/README.md
src/
  dtype.py               DataType abstraction (ap_fixed<W,I>, float32)
  layout.py              TensorLayout frozen dataclass (numel, alloc, n_chunks, chunk, stride)
  tensor.py              TensorInfo: metadata + C declaration emitters
  kernels.py             KERNEL_REGISTRY: per-kernel driver files, UIO default, init param
  nodes.py               ScheduledNode (VectorOP), MatmulNode, ConvNode, MatmulConvNode,
                         PoolNode, ReshapeNode, SpaceToDepthNode
  host_nodes.py          HostNode family (Softmax, LayerNorm, Gelu, Transpose, Slice,
                         Gather, OneHot, Cast): numpy reference + C helpers
  llm_nodes.py           axi.llm ops (LlmEmbed, LlmRMSNorm, LlmResAdd, LlmAttention,
                         LlmSiluMul, LlmSelectRow, LlmDequant, LlmAttnPrep /
                         LlmAttnSoftmax / LlmAttnMerge) + LlmAttnConvNode (FPGA q·Kᵀ / P·V;
                         static keys for the vision encoder)
  llama.py               Llama frontend: config.json + safetensors + formats → entry graphs
                         (image_rows: a VLM text model's prefill reads image-feature rows)
  vit.py                 Vision-encoder frontend (SmolVLM: SigLIP ViT + Idefics3 connector)
                         → the `vision` entry of a multi-entry project (attn_split=R:
                         softmax / P·V in R query-row parts, default 1)
  vit_nodes.py           the vision host ops (VitLayerNorm / AttnPrep / AttnSoftmax / Gelu /
                         ResAdd / EmbedAdd / PixelShuffle / SumDequant) + VIT_C helpers
  piper.py               Piper (VITS) text-to-speech frontend: the flow + HiFi-GAN decoder as the
                         fixed-size `chunk` entry (Conv exponents, folded 1-D convs, polyphase);
                         the text encoder as `encode_<T>` entries (length buckets, MatMuls and
                         attention on ConvKernel, one weight image shared by the buckets)
  tts_nodes.py           the TTS host ops (TtsPrep / Gate / Sum / FlowOut / Interleave / Pcm; the
                         encoder's TtsEmbed / RowPrep / AttnSoftmax / AttnMerge / ResNorm / EncOut)
                         + TTS_C / TTS_ENC_C
  numeric.py             axi.numeric metadata: power-of-two exponents (MatMul and Conv weight
                         encoding), host tensors, states
  fusion.py              Constant folding, Split lowering, LayerNorm / GELU fusion,
                         constant-broadcast normalisation
  matmul_lowering.py     MatMul → ConvKernel lowering pass (engine choice, geometry, row
                         split; _plan_matmul under --plan)
  planning.py            the opt-in --plan mode (options, model lookup, report line) —
                         ../doc/plans/TACTICS_PLAN.md
  perf_calls.py          KernelCall: the register values of a kernel call (every kernel node's
                         kernel_calls()), the bitstream id
  perf_model.py          per-bitstream kernel model (exact calls + fitted families), error bands
  perf_fit.py            fitting it from a calibration campaign (NNLS + k-NN correction)
  host_model.py          host-op timing model from board profiles
  tactics.py             a MatMul's tactics (conv geometries / row splits, tiled, GEMV)
  order_search.py        issue-order search on the timed event-stream replay (codegen/timing.py)
  matmul_gemv.py         MatmulKernel GEMV streaming pass (single-row MatMuls, B image)
  fc_conv.py             fully-connected Convs -> Flatten + MatMul + Reshape (+ bias Add)
  llm_entries.py         Llama entry graphs: prefill kernel widths shared with the
                         GEMV decode, one copy of every weight
  cost_model.py          ConvKernel / MatmulKernel (tiled + GEMV) cycle estimates
  _conv_hw_config.py, _matmul_hw_config.py, _pool_hw_config.py
                         platform JSON resolvers (kernels.{conv,matmul,pool})
  graph.py               OnnxGraph: ONNX parsing, shape inference, tensor registry
  schedule.py            Dag: data-flow DAG over scheduled nodes (+ state RAW / WAR / WAW
                         edges); topological order, predecessors/successors,
                         independent-pair queries
  report.py              ReportGenerator → report.md
  timeline_html.py       timeline.html: the predicted execution (codegen/timing.simulate's spans;
                         with a board profile, measured per-layer times vs Timeline.windows)
                         as a self-contained Nsight-style page (canvas, LOD merging, search, deps)
  bitstream/             upload_bitstream.py implementation (convert, hwh, xclbin,
                         board, loader, platforms/kv260.py)
  remote/                SSH session, config defaults, preflight checks (shared by
                         run_remote_tests.py / run_remote_perf.py / upload_bitstream.py)
  codegen/
    __init__.py          CodeGenerator (assembles all mixins)
    _core.py             _compute_event_stream() → list of comment/start/wait/drain/cpu events
                         _compute_live_intervals() → event-stream-based intervals
                         _compute_tensor_layouts() → TensorLayout; pool slot colouring
    _header.py           generate_header()  → include/inference.h
    _source.py           generate_source()  → src/inference.c
    _buf_impl.py         generate_buf_impl() → src/inference_buf.c; generate_setup_script()
    _simulate.py         Fixed-point forward simulation; generate_expected_dat()
    _test.py             generate_test()    → test/test_inference.c
    _cmake.py            generate_cmake()   → CMakeLists.txt
    _banners.py          File-header banner helpers
    multi.py             MultiEntryGenerator (--entry): several graphs, one weight pool
    timing.py            simulate(): timed replay of the event stream (lane / CPU busy
                         times, CPU waits) for --plan and perf_calibrate.py simulate
test/
  gen_*_models.py        Test ONNX model generators; gen_all_models.py runs them all
  helpers.py             _model(), _models_exist() shared by test modules
  host_emu.py            Builds a generated project on the host against software
                         VectorOP / Matmul / Conv kernels and runs test_inference
  models/                Generated ONNX models (single_add.onnx, etc.)
  c/                     C harness for test_profiler_overlap.py
  test_*.py              75 pytest modules, 1625 tests, all pass (test_bert_base.py
                         downloads bertsquad-12, 435 MB, on its first run) — includes
                         test_dag.py (DAG correctness), test_parallel_waits.py (split
                         start/wait emission), test_nop_corner_cases.py (NOP-layer
                         corner cases), test_profiler_overlap.py (overlapping brackets),
                         test_cache_coherency.py (sync audit), test_host_ops.py,
                         test_llm_ops.py, test_llama.py, test_vit.py, test_matmul_on_conv.py,
                         test_planning.py (--plan options, state edges, reordered code),
                         test_perf_calls.py (kernel_calls() == emitted C), test_timing.py,
                         test_generator_memory.py (lean simulation, weightless shape
                         inference, lazy entry models / checkpoints, shared arrays),
                         test_conv_exp.py (Conv with power-of-two exponents),
                         test_tts_ops.py (the TTS C helpers), test_piper.py (the Piper
                         chunk and encode entries == the specification, stitching,
                         buckets, host_emu; the C duration predictor == its spec),
                         test_bert_base.py (BERT-base on the real model),
                         test_fetch_assets.py (the BERT demo's downloader, local server)
```

## Key Abstractions

### DataType (`src/dtype.py`)

Encapsulates everything type-specific. The rest of the codebase is type-agnostic.

```python
from src.dtype import AP_FIXED_16_8, FLOAT32, ApFixed

g  = OnnxGraph("model.onnx", dtype=AP_FIXED_16_8)   # default
cg = CodeGenerator(g, "model.onnx", dtype=AP_FIXED_16_8)
```

| Type | `name` | `bytes_per_elem` | `align_elems` | Range |
|------|--------|-----------------|----------------|-------|
| `AP_FIXED_16_8` | `ap_fixed<16,8>` | 2 | 8 | [-128, 127.996] |
| `FLOAT32` | `float32` | 4 | 4 | IEEE 754 |

Key methods:
- `quantize(x)` — round-to-nearest onto the representable grid (weights, inputs)
- `truncate(x)` / `truncate_div(x)` — floor (AP_TRN) / toward zero (Div): kernel outputs in simulation
- `encode_weight(data)` → list of C literal strings (`"0x0100"`)
- `float_to_storage(x)` → numpy array with `np_storage` dtype
- `dat_bytes(data)` → little-endian bytes for external `.dat` files
- `c_display(ptr, idx)` → C expression for `printf("%.4f")` display
- `c_fill_rhs(pos_expr)` → RHS of test ramp-fill assignment

Adding a new type: subclass `DataType`, implement all abstract methods, pass
the instance to `OnnxGraph` and `CodeGenerator`.

### Node Classes (`src/nodes.py`, `src/host_nodes.py`, `src/llm_nodes.py`, `src/vit_nodes.py`)

**ScheduledNode** — VectorOPKernel element-wise ops:

| ONNX op | Opcode | Arity | Notes |
|---------|--------|-------|-------|
| `Add` | `OP_ADD` (0) | binary | |
| `Sub` | `OP_SUB` (1) | binary | |
| `Mul` | `OP_MUL` (2) | binary | |
| `Div` | `OP_DIV` (3) | binary | |
| `Relu` | `OP_RELU` (4) | unary | b=NULL |
| `Clip(min=0,max=6)` | `OP_RELU6` (5) | unary | exact bounds required |

With `OnnxGraph(fuse_act=True)` (CLI default) a following `Relu` /
`Clip(0,6)` is folded into the producing ScheduledNode (`act`, emitted as
`run_op_act()`).

**MatmulNode** — MatmulKernel: `MatMul`. Tiled 2-D matrix multiply; supports
batched matmul and row-strided decomposition for alignment-gapped buffers.
A constant B read only by tiled MatMuls is packed tile-major (`b_packed = 1`).
Single-row MatMuls take the kernel's GEMV streaming path (`gemv_kw`,
`src/matmul_gemv.py`): B through both read ports, row-major or — with
`OnnxGraph(matmul_gemv_kw={name: kw})` — in ConvKernel's kw image, which is
how a Llama project's decode shares the prefill weights
(`src/llm_entries.py`).  `run_matmul()` writes `gemv_kw` and `a_to_b` on
every call; `kernels.matmul.gemv_max_m = 0` turns the path (and those
register writes) off.

**ConvNode** — ConvKernel: `Conv`. NCHW 2-D convolution with optional bias,
configurable kernel/stride/pad (incl. `auto_pad`)/dilation. `group=1` or
depthwise (`group=in_ch`); other grouped convolutions are rejected.

**MatmulConvNode** — ConvKernel: a `MatMul` lowered by
`src/matmul_lowering.py` (`OnnxGraph(matmul_on_conv="auto")`, the default)
with swapped operand roles: `C[N][M] = A[N][K]·B[K][M]` is a conv with
`out_ch = N`, `in_ch = K/kw`, a `1×kw` kernel with stride `(1, kw)`, output
`out_h × out_w = M`, A (row-major = the packed weight layout when
`K % (16·kw) == 0`) as the weight and B as `x` (as is for `kw = 1`; a
constant B read only by this MatMul is re-imaged by
`nodes.conv_lowered_b_image` for `kw > 1`).  Engine and `(kw, out_w)` are
chosen with `src/cost_model.py` (conv-cycle-model port vs a board-calibrated
MatmulKernel model).  Batched MatMuls with per-item weights (attention)
emit one `run_conv_at()` per item.  When every one-call plan is
accumulator-limited (fewer than 16 output rows per chunk), the rows are
split over several `run_conv_at()` calls, with B shared.  Without `--plan`
only SmolVLM's 1024-token vision linears qualify.  The simulator treats it exactly like a
`MatmulNode` — the two kernels are bit-identical.  See
`../doc/scheduler/INFERENCE_SCHEDULER.md` §"MatMul on ConvKernel".

**PoolNode** — PoolingKernel: `MaxPool`, `AveragePool`, `LpPool` (p=1 or 2),
`GlobalMaxPool`, `GlobalAveragePool`, `GlobalLpPool`. Full 2-D NCHW geometry
including dilation and `count_include_pad`; `ceil_mode=1` is rejected.

`PoolNode.from_onnx_node` validates the model against the kernel's
compile-time bounds (`pool_h ≤ POOL_MAX_KH`, `pool_w ≤ POOL_MAX_KW`,
`(pool_h - 1) * dil_h + 1 ≤ POOL_MAX_LINE_BUF_ROWS`,
`(pool_w - 1) * dil_w + 1 ≤ POOL_MAX_LINE_BUF_COLS`) and raises `SchedulerError`
naming the violated bound + the JSON field to bump.  The bounds come
from the **same platform JSON the C++ build reads**
(`platforms/<AXI_PLATFORM>.json`, `kernels.pool` object — see
[`../doc/build-and-test/PLATFORM_CONFIGURATION.md`](../doc/build-and-test/PLATFORM_CONFIGURATION.md)
for the full field reference across all three kernels, and
[`../doc/kernels/POOL_OPTIMISATION.md`](../doc/kernels/POOL_OPTIMISATION.md) §4 for the
pool-specific knobs).
`src/_pool_hw_config.py::resolve(platform_name)` is the resolver:

- `platform_name=None` (default) reads `AXI_PLATFORM` env var (defaults
  to `kv260`).  Mirrors the CMake cache var of the same name so CLI
  invocations targeting a non-default board can stay in sync with
  `cmake -DAXI_PLATFORM=<name>`.
- Raises `PoolHwConfigError` on missing file, missing `kernels.pool`
  object, missing required field, or wrong field type — no silent
  fallback to defaults.

`tile_c` and `ow_parallel` are not validated: any C runs (channel
tiling) and any out_w runs (residual-lane padding) inside the kernel,
so models cannot violate them.  `MatmulNode` checks `K ≤ MATMUL_MAX_K`
(`kernels.matmul.max_k`) the same way; ConvNode checks the
`kernels.conv` bounds.

**ReshapeNode** — zero-cost buffer alias: `Reshape`, `Squeeze`, `Unsqueeze`,
`Flatten`, `Dropout`, `Identity` (`RESHAPE_OP_TYPES`) and a `Cast` within one
storage kind. `emit_call()` returns `""`. Output pointer is assigned
`= source` in `inference_init()`; NULLed without free in `inference_deinit()`.
Requires equal `numel` between source and output.

**SpaceToDepthNode** — host-CPU reorder: ONNX `SpaceToDepth`, and the
stride-2 stem rewrite (`OnnxGraph(s2d_stem=True)`, CLI default).

**HostNode** (`src/host_nodes.py`) — `Softmax`, `LayerNormalization`, `Gelu`,
`Transpose`, `Slice` (every `Split` output), `Gather`, `OneHot`, `Cast`:
C helpers run inside `inference_run()`, numpy `reference()` for the
simulator; a contiguous 64-byte-aligned `Slice` piece becomes a zero-cost
sub-buffer view instead (`OnnxGraph._choose_slice_views`).
**LlmNode** / **LlmAttnConvNode** (`src/llm_nodes.py`) — the `axi.llm`
domain ops of the Llama frontend; **VitNode** (`src/vit_nodes.py`, an
`LlmNode`) — those of the vision encoder.

Every kernel node lists the calls it issues with `kernel_calls(layouts)`
(`src/perf_calls.py` `KernelCall`: the register values), which the
performance models, `perf_calibrate.py` and `--plan` key on.

`Gemm` is decomposed to `MatMul` + optional `Add` by `OnnxGraph._preprocess_model()`
at load time, before any node class sees it (`alpha=1, beta=1, transA=0`;
`transB=1` only with a constant 2-D B, transposed into a `<B>_T` initializer).
`Constant` nodes are folded into initializers and `Split` is lowered to one
`Slice` per output before dispatch.

**Broadcasting**: One input per binary op may broadcast. Rules:
- Right-align input shape to output shape
- All broadcast dimensions (size 1) must form a **contiguous leading block**
  (size-1 output dims are neutral)
- Valid: `[1, 1, 64]` broadcasts to `[4, 32, 64]` (dims 0,1 broadcast)
- Invalid: `[4, 1, 64]` to `[4, 32, 64]` (broadcast dim between matching dims)

When broadcasting, `emit_call()` emits one `run_op()` call; the kernel runs
the outer loop itself (`outer`, `a_inc`, `b_inc` registers; an input that
repeats has increment 0):
```c
run_op(X, bias, Y, INFERENCE_Y_CHUNK, VECTOROP_ADD,
       128u, INFERENCE_Y_CHUNK_STRIDE, 0u);
```

### CodeGenerator (`src/codegen/`)

Multi-mixin class. `_CoreMixin.__init__` computes padded allocation sizes for all
tensors accounting for broadcast alignment gaps. All other mixins read `self._alloc_sizes`.

Allocation rules (`_compute_tensor_layouts` → `TensorLayout`):
- Phase 1: all tensors seeded as `TensorLayout.flat(numel)`
- Phase 2: broadcast VectorOP nodes set advancing/repeating layouts with alignment-padded strides
- Phase 3: layout propagates forward through non-broadcast, non-MatmulNode chains
- MatmulNode reads row strides directly from `TensorLayout.gap` at emit time

### Memory layout — event-stream liveness

Pool-slot colouring uses interval-graph greedy first-fit. The
**intervals are derived from the event stream**, not from node-index
order, so two tensors share a slot only when one is fully drained
before the other's producer starts under the parallel-wait emission.

For each intermediate tensor `T` that is not an alias (Reshape alias or
Slice view):

- `start_event` = event index of `T`'s producer Start (or `cpu` event)
- `end_event`   = max event index of any `kernel_wait` that drains a
  consumer on its lane (a host consumer's own `cpu` event)
- Consumers reached via a ReshapeNode chain (`RESHAPE_OP_TYPES`) or a
  Slice view extend `T`'s interval through the alias: a
  `Pool → Squeeze → MatMul` chain keeps Pool's output buffer live until the
  Matmul lane drains.

This is what makes parallel branches correct: in `parallel_two_chains`,
`ca1` (consumed by Pool) and `cb0` (written by Conv-B in parallel) have
overlapping event intervals, so the colouring places them in different
slots even though their node indices look disjoint. Coloring is still
valid for the strictly-sequential case — every node-index interval is
also an event-index interval.  Weights, then DMA states, precede the slot
region; host-memory intermediates get their own arena coloured the same way
(`_compute_host_layout`).

The Dag invariant *"if two tensors share a pool offset, their event
intervals must be strictly disjoint"* is enforced by
`test/test_nop_corner_cases.py::TestNopFixturesNoSlotAliasing` across
every NOP fixture.

### Schedule (DAG) — `src/schedule.py`

See [`doc/SCHEDULER_DAG.md`](doc/SCHEDULER_DAG.md) for the full
algorithm reference (DAG construction, event-stream walk,
event-timeline liveness, slot coloring, worked examples on
`parallel_two_chains`, `squeeze_then_matmul`,
`asymmetric_nested_branches`, and the invariants tested in CI).

`Dag.from_graph(OnnxGraph)` builds a producer/consumer DAG over the
scheduled node list:

- Edge `u → v` iff some intermediate tensor produced by `u` is consumed
  by `v`. Graph inputs, constant initializers and persistent states no
  node of the graph produces are **external** — they impose no producer
  edges (they're already available before `inference_run()` enters its body).
- Persistent states add ordering edges in list order
  (`from_graph(state_edges=True)`, the default): a read after a write (RAW),
  a write after reads (WAR) and a write after a write (WAW), from the
  nodes' state inputs and `state_updates()` / `state_writes()` — so any
  order that respects the DAG (the `--plan` order search) keeps every state
  access where the list put it.
- ReshapeNodes appear as ordinary DAG nodes (`kernel_name == ""`), so
  consumers of an alias are correctly ordered after the producer of the
  underlying source. The event-stream walker traverses through them
  when it needs the *real* producing kernel for wait emission.

Public API: `predecessors(idx)`, `successors(idx)`, `roots()`,
`leaves()`, `topological_order()` (deterministic Kahn), `ancestors()`,
`independent_pairs()`. The DAG is the foundation for the parallel-wait
scheduler — it answers "which nodes have all their data ready" and
"which nodes are concurrency-independent".

### Parallel kernel execution (`kernel_wait`)

Each kernel-bearing node lives on exactly one hardware lane (`Conv`,
`Pool`, `Matmul`, `VectorOP`); only one of each IP exists on the FPGA.
The codegen overlaps work across **different** lanes.  Host nodes have no
lane: they wait for their producers, run inline and are never waited on.

**Helpers** are non-blocking: `run_op()` / `run_op_act()` / `run_matmul()` /
`run_conv()` / `run_pool()` program the AXI-Lite registers, call
`XKernel_Start()`, and return immediately. (`run_matmul_at()` — used only
inside the 4D×3D outer loop where iterations would race on the same Matmul
registers — remains synchronous; the per-item `run_conv_at()` loop of a
batched MatMul on ConvKernel waits on `KERNEL_CONV` before each call after
the first and leaves the last in flight.)

**Sync** funnels through a single weak-symbol primitive emitted into the
generated source:

```c
typedef enum {
    KERNEL_VECTOROP, KERNEL_MATMUL, KERNEL_CONV, KERNEL_POOL, KERNEL_COUNT
} kernel_id_t;

__attribute__((weak))
void kernel_wait(kernel_id_t k);   // default: poll IsDone in a tight loop
```

Override at link time to swap in IRQ-driven waiting (UIO under Linux,
GIC under bare-metal) without touching the generated code.

**Event stream** — `CodeGenerator._compute_event_stream()` is the single
source of truth for the schedule. It walks `graph.nodes` once (or a
candidate `order` against a given `dag`, for the `--plan` order search),
tracks `pending[lane] → node_idx`, and emits a deterministic sequence:

```
('comment', node_idx)              ─ node header
('start',   node_idx)              ─ non-blocking Start
('start_sync', node_idx)           ─ Start whose helper drains internally
('wait',    kid, drained_idx)      ─ kernel_wait(KERNEL_*) call
('drain',   kid, drained_idx)      ─ final wait before output cache sync
('reshape', node_idx)              ─ ReshapeNode / Slice view (no kernel work)
('cpu',     node_idx)              ─ SpaceToDepthNode / HostNode (host code, inline)
```

Both the body emitter (`_inference_function`) and the live-interval
analyser (`_compute_live_intervals`) consume this stream verbatim, so
the buffer-slot colouring and the emitted waits cannot disagree about
which lanes are in flight at any point; `codegen/timing.py` replays the
same stream with modelled durations.

A wait fires only when (a) some predecessor is still in flight on its
lane, or (b) the target lane has a different op pending. Predecessor
analysis walks **through** ReshapeNode chains to reach the real
producing kernel.

### Cache Coherency Model

The kernel's AXI master reads/writes DDR using **physical addresses** programmed
into AXI-Lite registers. CPU cache must be explicitly managed (on Linux the
XRT buffer objects are mapped cacheable by default):

- `inference_buf_sync_to_device(buf)` — clean CPU cache → DDR (before a kernel reads **or writes**)
- `inference_buf_sync_from_device(buf)` — invalidate CPU cache (after a kernel wrote)

**Contract** (full table: `../doc/scheduler/INFERENCE_SCHEDULER.md` §Cache coherency):
- `inference_init()`: syncs the weight pool **once** after `memcpy` from ROM
- `inference_run()`: cleans all graph **inputs and outputs** at the top, invalidates all graph **outputs** at the bottom; `kernel_wait` calls drain in-flight kernels before the output sync
- Host ops: invalidate a kernel-written input after its lane drained, flush their output before a kernel reads it; KV-cache DMA states are flushed with `llm_cache_flush`
- `run_*()` helpers: pure AXI-Lite register writes + Start (no sync, no poll)
- Internal kernel-to-kernel buffers (intermediates): **no sync ever needed**

`test/test_cache_coherency.py` audits every test model's `inference_run()`
against these rules; `test/host_emu.py` (`incoherent=True`) checks them
dynamically.

### Large Tensor Handling

Tensors exceeding the threshold are written to external binary `.dat` files and
loaded at runtime via `fread()`:

| Threshold | Constant | Files |
|-----------|----------|-------|
| `LARGE_WEIGHT_THRESHOLD = 4096` | in `tensor.py` | `weights/<c_name>.dat` |
| `LARGE_EXPECTED_THRESHOLD = 4096` | in `codegen/_simulate.py` | `expected/<c_name>.dat` |
| `TABLE_FILE_BYTES = 64 KiB` | in `llm_nodes.py` | `weights/<name>.dat` (host tables) |

Both `.dat` files contain little-endian elements in the DMA-buffer layout
(packed images for Conv weights / MatMul B; alignment gaps are zero-filled
for broadcast tensors).

## Generated C API

```c
// inference.h
typedef uint16_t Data_t;              // ap_fixed<16,8> raw bits
#define INFERENCE_BYTES_PER_ELEM  2u
#define INFERENCE_ALIGN_BYTES     16u
#define INFERENCE_ALIGN_ELEMS     (INFERENCE_ALIGN_BYTES / INFERENCE_BYTES_PER_ELEM)
#define INFERENCE_<TENSOR>_SIZE   N   // alloc elements, one per graph input / output
#define INFERENCE_BUF_POOL_SIZE_BYTES N  // advisory upper bound (no slot reuse)

// For broadcast nodes only:
#define INFERENCE_<TENSOR>_CHUNK        chunk_size
#define INFERENCE_<TENSOR>_CHUNK_STRIDE INFERENCE_ALIGN_UP(INFERENCE_<TENSOR>_CHUNK)

// One instance name per active kernel, registry order (VectorOP, Matmul, Conv, Pool);
// defaults INFERENCE_<KERNEL>_INSTANCE ("VectorOPKernel_0", ...)
int  inference_init(const char *vectoropkernel_instance /*, ... */);
void inference_run(inference_buf_t *in, inference_buf_t *out);
// signature lists all graph inputs then all graph outputs; host-memory tensors
// (axi.numeric) are plain pointers; --entry projects: inference_run_<name>()
void inference_deinit(void);
unsigned           inference_num_layers(void);
const char *const *inference_layer_names_ptr(void);

// inference_buf.c (platform-specific)
inference_buf_t *inference_buf_alloc(unsigned n_elem);
void             inference_buf_free(inference_buf_t *buf);     // == release (refcounted)
void             inference_buf_init_view(inference_buf_t *view, inference_buf_t *base,
                                         unsigned offset_elems, unsigned count_elems);
Data_t          *inference_buf_ptr(inference_buf_t *buf);         // virtual address (CPU)
uint64_t         inference_buf_phys(const inference_buf_t *buf);  // physical address (AXI)
unsigned         inference_buf_count(const inference_buf_t *buf);
int              inference_buf_is_cached(const inference_buf_t *buf);  // static inline
void             inference_buf_sync_to_device(inference_buf_t *buf);
void             inference_buf_sync_from_device(inference_buf_t *buf);
```

Build knobs: `-DINFERENCE_TARGET=BARE_METAL|LINUX`, `-DINFERENCE_BUF_CACHEABLE=ON|OFF`
(env `INFERENCE_BUF_CACHEABLE`), `-DINFERENCE_HOST_THREADS=N` (env
`INFERENCE_HOST_THREADS`, host ops only), `-DINFERENCE_PROFILING=ON`,
`-DINFERENCE_WEIGHTS_DIR=...`, `-DINFERENCE_<KERNEL>_INSTANCE=...`.

## Testing

Tests live in `test/`. Run with `pytest`:

```bash
.venv/bin/python -m pytest test/ -v                        # all tests
.venv/bin/python -m pytest test/test_source.py -v          # generated inference.c
.venv/bin/python -m pytest test/test_broadcast.py -v       # broadcast logic
.venv/bin/python -m pytest test/ -k "test_relu" -v         # filter by name
.venv/bin/python -m pytest test/test_bert_base.py -v       # BERT-base on the real model (downloads it once;
                                                           # BERT_SQUAD_MODEL / BERT_SQUAD_ASSETS override the paths)
```

Most test classes are decorated `@unittest.skipUnless(_models_exist(), ...)` —
run all `gen_*.py` scripts first if tests are skipped (see Quick Start).

**Hardware-bound fixtures.** The `gen_conv_models.py`, `gen_matmul_models.py`
and `gen_pool_models.py` "must raise" / "at_limit" generators read their
geometry constants from the platform JSON via the same `_<kernel>_hw_config`
resolvers the scheduler uses (`MATMUL_MAX_K`, `POOL_MAX_KH/KW/LINE_BUF_*`,
`CONV_MAX_IN_CH/OUT_CH/KH/KW/LINE_BUF_*/ACC_PERSIST_ENTRIES`). The matching
`test_*.py` assertions read the same constants. Don't hard-code a bound
literal in a fixture or its assertion — a JSON bump (e.g. `max_k: 2048 →
4096`) would otherwise silently turn a "must raise" model into a legal
one, masking validator regressions. Re-run the generator after any
`platforms/<name>.json` change.

## On-Device Testing and Benchmarking

Four scripts drive KV260 hardware over SSH:

| Script | Purpose | Config |
|--------|---------|--------|
| `upload_bitstream.py` | Load the bitstream, xclbin and device-tree overlay | `bitstream_config_kv260.json.example` |
| `run_remote_tests.py` | **Correctness** — generates a C project per model, builds on board, compares every output element against Python GT | `remote_config.json.example` (148 models) |
| `run_remote_perf.py` | **Performance** — builds one benchmark project for all four kernels, runs parametric cases and reports latency (ms) and throughput (GB/s / GOps/s); `--json OUT` also writes the results as JSON | `perf_config.json` (60 cases) |
| `perf_calibrate.py run` | **Calibration** for `--plan` — times batches of kernel calls with `bench_src/calib_runner.c` (two passes), then `fit` writes `perf_models/<platform>/<bitstream-id>.json` (`../doc/plans/TACTICS_PLAN.md` §4.3) | `--config` in the `run_remote_perf.py` format |

All share the same SSH/driver config schema. See `doc/REMOTE_TESTING.md` for
the full reference including the `benchmarks` config section and per-kernel
case field definitions.  `run_remote_tests.py` keys `remote.uio_devices` by
the `KERNEL_REGISTRY` names (`PoolKernel`); `run_remote_perf.py` uses
`PoolingKernel`.

`run_remote_perf.py` validates each `VectorOPKernel` case's `op` against the
kernel-supported set (`OP_ADD..OP_RELU6`, 0..5) when it loads the cases and
exits with `config error: VectorOPKernel case '<label>': unsupported op=…`
before it connects to the board (`_load_cases()` runs right after the config
is loaded), so a bad case costs no remote build.

## Driver Sources

Each hardware kernel has its own driver generated by Vitis HLS synthesis.
When `--driver-dir` is omitted, `driver/` is left empty except for a stub
`README.md` listing the files the model's kernels need.

| Kernel | Driver prefix | Required files |
|--------|--------------|----------------|
| VectorOPKernel | `xvectoropkernel` | `xvectoropkernel.h`, `xvectoropkernel_hw.h`, `xvectoropkernel.c`, `xvectoropkernel_sinit.c`, `xvectoropkernel_linux.c` |
| MatmulKernel | `xmatmulkernel` | `xmatmulkernel.h`, `xmatmulkernel_hw.h`, `xmatmulkernel.c`, … |
| ConvKernel | `xconvkernel` | `xconvkernel.h`, `xconvkernel_hw.h`, `xconvkernel.c`, … |
| PoolKernel | `xpoolingkernel` | `xpoolingkernel.h`, `xpoolingkernel_hw.h`, `xpoolingkernel.c`, … |
