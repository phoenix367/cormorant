# CLAUDE.md — inference-scheduler

Python code-generator that parses an ONNX model and emits a complete C project
that runs inference on the Xilinx KV260 FPGA using up to four hardware kernels:
VectorOPKernel (element-wise), MatmulKernel (matmul/FC), ConvKernel (2-D conv),
and PoolingKernel (2-D pooling). Reshape is handled as a buffer alias with no
hardware call; Gemm is decomposed to MatMul + Add at load time.

## Quick Start

```bash
cd inference-scheduler
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# Generate all test ONNX models (required before running tests)
.venv/bin/python test/gen_test_models.py
.venv/bin/python test/gen_matmul_models.py
.venv/bin/python test/gen_mixed_kernel_models.py
.venv/bin/python test/gen_conv_models.py
.venv/bin/python test/gen_pool_models.py
.venv/bin/python test/gen_reshape_gemm_models.py
.venv/bin/python test/gen_mixed_all_kernels_models.py

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

Options:
  --out-dir DIR              Output directory (default: ./<stem>_inference/)
  --driver-dir DIR           Copy XVectoropkernel driver sources from this path
  --embed-large-weights      Inline all weights as C arrays (skip .dat files)
  --embed-large-expected     Inline all GT arrays in test_inference.c
  --matmul-on-conv {auto,always,off}
                             Run MatMuls on ConvKernel with swapped operand roles
                             (default auto: where the cost model says it is faster)
  --no-matmul-on-conv        Same as --matmul-on-conv off
```

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
use `--input-shape NAME=D1,D2,…` for those.  All `*-simplified.onnx` and
`*_simplified.onnx` files in this directory were produced by (or can be
regenerated with) this script.

See `doc/MODEL_PREPARATION.md` for the full workflow, including handling
of unsupported ops that survive simplification (e.g. tail `Softmax` /
`Cast`, grouped Conv) and a worked example on `resnet50-v1-12.onnx`.

## Generated Project Layout

```
<out_dir>/
├── CMakeLists.txt            INFERENCE_TARGET=BARE_METAL|LINUX
├── include/inference.h       Public API: Data_t, size macros, init/run declarations
├── src/
│   ├── inference.c           Weight ROM arrays, run_op() helper, init/run bodies
│   └── inference_buf.c       DMA buffer alloc/sync (Linux XRT or bare-metal Xil)
├── test/test_inference.c     On-device test: ramp fill → run → compare vs GT
├── scripts/check_inference_setup.sh
├── driver/                   XVectoropkernel sources (copied or stub README)
├── weights/                  External .dat files for large weight tensors (> 4096 elems)
└── expected/                 External .dat files for large GT expected arrays (> 4096 elems)
```

## Source Layout

```
inference_scheduler.py   CLI entry point — ONNX → C project
simplify_onnx.py         CLI entry point — ONNX → ONNX (onnxsim + BN-fusion)
requirements.txt
src/
  dtype.py               DataType abstraction (ap_fixed<W,I>, float32)
  layout.py              TensorLayout frozen dataclass (numel, alloc, n_chunks, chunk, stride)
  tensor.py              TensorInfo: metadata + C declaration emitters
  nodes.py               ScheduledNode: ONNX op → VectorOPKernel call (+ MatmulNode,
                         ConvNode, MatmulConvNode, PoolNode, ReshapeNode, …)
  matmul_lowering.py     MatMul → ConvKernel lowering pass (engine choice, geometry)
  cost_model.py          ConvKernel / MatmulKernel cycle estimates
  graph.py               OnnxGraph: ONNX parsing, shape inference, tensor registry
  schedule.py            Dag: data-flow DAG over scheduled nodes; topological order,
                         predecessors/successors, independent-pair queries
  codegen/
    __init__.py          CodeGenerator (assembles all mixins)
    _core.py             _compute_event_stream() → list of Start/Wait/Drain events
                         _compute_live_intervals() → event-stream-based intervals
                         _compute_tensor_layouts() → TensorLayout; pool slot colouring
    _header.py           generate_header()  → include/inference.h
    _source.py           generate_source()  → src/inference.c
    _buf_impl.py         generate_buf_impl() → src/inference_buf.c
    _simulate.py         Fixed-point forward simulation; generate_expected_dat()
    _test.py             generate_test()    → test/test_inference.c
    _cmake.py            generate_cmake()   → CMakeLists.txt
    _banners.py          File-header banner helpers
test/
  gen_test_models.py     Build all test ONNX models
  helpers.py             _model(), _models_exist() shared by test modules
  models/                Pre-generated ONNX models (single_add.onnx, etc.)
  test_*.py              pytest test modules (1301 tests total)
                         — includes test_dag.py (DAG correctness),
                           test_parallel_waits.py (split start/wait emission),
                           test_nop_corner_cases.py (NOP-layer corner cases),
                           test_profiler_overlap.py (overlapping bracket support)
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
- `quantize(x)` — round float64 array to representable grid (used in simulation)
- `encode_weight(data)` → list of C literal strings (`"0x0100"`)
- `float_to_storage(x)` → numpy array with `np_storage` dtype
- `dat_bytes(data)` → little-endian bytes for external `.dat` files
- `c_display(ptr, idx)` → C expression for `printf("%.4f")` display
- `c_fill_rhs(pos_expr)` → RHS of test ramp-fill assignment

Adding a new type: subclass `DataType`, implement all abstract methods, pass
the instance to `OnnxGraph` and `CodeGenerator`.

### Node Classes (`src/nodes.py`)

Five node classes cover all supported ONNX operators:

**ScheduledNode** — VectorOPKernel element-wise ops:

| ONNX op | Opcode | Arity | Notes |
|---------|--------|-------|-------|
| `Add` | `OP_ADD` (0) | binary | |
| `Sub` | `OP_SUB` (1) | binary | |
| `Mul` | `OP_MUL` (2) | binary | |
| `Div` | `OP_DIV` (3) | binary | |
| `Relu` | `OP_RELU` (4) | unary | b=NULL |
| `Clip(min=0,max=6)` | `OP_RELU6` (5) | unary | exact bounds required |

**MatmulNode** — MatmulKernel: `MatMul`. Tiled 2-D matrix multiply; supports
batched matmul and row-strided decomposition for alignment-gapped buffers.

**ConvNode** — ConvKernel: `Conv`. NCHW 2-D convolution with optional bias,
configurable kernel/stride/pad/dilation. `groups=1` only.

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
emit one `run_conv_at()` per item.  The simulator treats it exactly like a
`MatmulNode` — the two kernels are bit-identical.  See
`../doc/INFERENCE_SCHEDULER.md` §"MatMul on ConvKernel".

**PoolNode** — PoolingKernel: `MaxPool`, `AveragePool`, `LpPool` (p=1 or 2),
`GlobalMaxPool`, `GlobalAveragePool`, `GlobalLpPool`. Full 2-D NCHW geometry
including dilation and `count_include_pad`.

`PoolNode.from_onnx_node` validates the model against the kernel's
compile-time bounds (`pool_h ≤ kMaxPoolH`, `pool_w ≤ kMaxPoolW`,
`(pool_h - 1) * dil_h + 1 ≤ kMaxLineBufRows`,
`(pool_w - 1) * dil_w + 1 ≤ kMaxLineBufCols`) and raises `SchedulerError`
naming the violated bound + the JSON field to bump.  The bounds come
from the **same platform JSON the C++ build reads**
(`platforms/<AXI_PLATFORM>.json`, `kernels.pool` object — see
[`../doc/PLATFORM_CONFIGURATION.md`](../doc/PLATFORM_CONFIGURATION.md)
for the full field reference across all three kernels, and
`doc/POOL_OPTIMIZATION.md` §4 for the pool-specific architectural
context).
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
so models cannot violate them.

**ReshapeNode** — zero-cost buffer alias: `Reshape`. `emit_call()` returns `""`.
Output pointer is assigned `= source` in `inference_init()`; NULLed without free
in `inference_deinit()`. Requires equal `numel` between source and output.

`Gemm` is decomposed to `MatMul` + optional `Add` by `OnnxGraph._preprocess_model()`
at load time, before any node class sees it.

**Broadcasting**: One input per binary op may broadcast. Rules:
- Right-align input shape to output shape
- All broadcast dimensions (size 1) must form a **contiguous leading block**
- Valid: `[1, 1, 64]` broadcasts to `[4, 32, 64]` (dims 0,1 broadcast)
- Invalid: `[4, 1, 64]` to `[4, 32, 64]` (broadcast dim between matching dims)

When broadcasting, `emit_call()` generates a for-loop calling `run_op_at()`:
```c
for (unsigned _i = 0u; _i < 4u; _i++) {
    run_op_at(X, _i * INFERENCE_Y_CHUNK_STRIDE, bias, 0u,
              Y, _i * INFERENCE_Y_CHUNK_STRIDE, INFERENCE_Y_CHUNK, VECTOROP_ADD);
}
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

For each non-Reshape intermediate tensor `T`:

- `start_event` = event index of `T`'s producer Start
- `end_event`   = max event index of any `kernel_wait` that drains a
  consumer on its lane
- Consumers reached via a ReshapeNode chain (Squeeze, Unsqueeze,
  Reshape, Dropout, Flatten — all `RESHAPE_OP_TYPES`) extend `T`'s
  interval through the alias: a `Pool → Squeeze → MatMul` chain keeps
  Pool's output buffer live until the Matmul lane drains.

This is what makes parallel branches correct: in `parallel_two_chains`,
`ca1` (consumed by Pool) and `cb0` (written by Conv-B in parallel) have
overlapping event intervals, so the colouring places them in different
slots even though their node indices look disjoint. Coloring is still
valid for the strictly-sequential case — every node-index interval is
also an event-index interval.

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
  by `v`. Graph inputs and constant initializers are **external** —
  they impose no edges (they're already available before
  `inference_run()` enters its body).
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
The codegen overlaps work across **different** lanes.

**Helpers** are non-blocking: `run_op()` / `run_matmul()` / `run_conv()`
/ `run_pool()` program the AXI-Lite registers, call `XKernel_Start()`,
and return immediately. (`run_matmul_at()` — used only inside the 4D×3D
outer loop where iterations would race on the same Matmul registers —
remains synchronous.)

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
source of truth for the schedule. It walks `graph.nodes` once, tracks
`pending[lane] → node_idx`, and emits a deterministic sequence:

```
('comment', node_idx)              ─ node header
('start',   node_idx)              ─ non-blocking Start
('start_sync', node_idx)           ─ Start whose helper drains internally
('wait',    kid, drained_idx)      ─ kernel_wait(KERNEL_*) call
('drain',   kid, drained_idx)      ─ final wait before output cache sync
('reshape', node_idx)              ─ ReshapeNode (no kernel work)
```

Both the body emitter (`_inference_function`) and the live-interval
analyser (`_compute_live_intervals`) consume this stream verbatim, so
the buffer-slot colouring and the emitted waits cannot disagree about
which lanes are in flight at any point.

A wait fires only when (a) some predecessor is still in flight on its
lane, or (b) the target lane has a different op pending. Predecessor
analysis walks **through** ReshapeNode chains to reach the real
producing kernel.

### Cache Coherency Model

The kernel's AXI master reads/writes DDR using **physical addresses** programmed
into AXI-Lite registers. CPU cache must be explicitly managed:

- `inference_buf_sync_to_device(buf)` — flush CPU cache → DDR (before kernel reads)
- `inference_buf_sync_from_device(buf)` — invalidate CPU cache (after kernel writes)

**Contract**:
- `inference_init()`: syncs each weight buffer **once** after `memcpy` from ROM
- `inference_run()`: syncs all graph **inputs** at the top, all graph **outputs** at the bottom; `kernel_wait` calls drain in-flight kernels before the output sync
- `run_*()` helpers: pure AXI-Lite register writes + Start (no sync, no poll)
- Internal kernel-to-kernel buffers (intermediates): **no sync ever needed**

### Large Tensor Handling

Tensors exceeding the threshold are written to external binary `.dat` files and
loaded at runtime via `fread()`:

| Threshold | Constant | Files |
|-----------|----------|-------|
| `LARGE_WEIGHT_THRESHOLD = 4096` | in `tensor.py` | `weights/<c_name>.dat` |
| `LARGE_EXPECTED_THRESHOLD = 4096` | in `codegen/_simulate.py` | `expected/<c_name>.dat` |

Both `.dat` files contain little-endian elements in the strided DMA-buffer layout
(alignment gaps are zero-filled for broadcast tensors).

## Generated C API

```c
// inference.h
typedef uint16_t Data_t;              // ap_fixed<16,8>
#define INFERENCE_BYTES_PER_ELEM  2u
#define INFERENCE_ALIGN_BYTES     16u
#define INFERENCE_ALIGN_ELEMS     8u
#define INFERENCE_INPUT_SIZE      N   // alloc size for graph input(s)
#define INFERENCE_OUTPUT_SIZE     N   // alloc size for graph output(s)

// For broadcast nodes only:
#define INFERENCE_<TENSOR>_CHUNK        chunk_size
#define INFERENCE_<TENSOR>_CHUNK_STRIDE INFERENCE_ALIGN_UP(chunk_size, align)

int  inference_init(const char *instance_name);  // alloc DMA bufs, load weights
void inference_run(inference_buf_t *in, inference_buf_t *out);
// signature lists all graph inputs then all graph outputs;
// models may have multiple of each.
void inference_deinit(void);

// inference_buf.c (platform-specific)
inference_buf_t *inference_buf_alloc(unsigned size_elements);
void             inference_buf_free(inference_buf_t *buf);
void            *inference_buf_ptr(inference_buf_t *buf);   // virtual address (CPU)
uint64_t         inference_buf_phys(inference_buf_t *buf);  // physical address (AXI)
void             inference_buf_sync_to_device(inference_buf_t *buf);
void             inference_buf_sync_from_device(inference_buf_t *buf);
```

## Testing

Tests live in `test/`. Run with `pytest`:

```bash
.venv/bin/python -m pytest test/ -v                        # all tests
.venv/bin/python -m pytest test/test_source.py -v          # generated inference.c
.venv/bin/python -m pytest test/test_broadcast.py -v       # broadcast logic
.venv/bin/python -m pytest test/ -k "test_relu" -v         # filter by name
```

Most test classes are decorated `@unittest.skipUnless(_models_exist(), ...)` —
run all `gen_*.py` scripts first if tests are skipped (see Quick Start).

**Hardware-bound fixtures.** The `gen_conv_models.py`, `gen_matmul_models.py`
and `gen_pool_models.py` "must raise" / "at_limit" generators read their
geometry constants from the platform JSON via the same `_<kernel>_hw_config`
resolvers the scheduler uses (`MATMUL_MAX_K`, `POOL_MAX_KH/KW/LINE_BUF_*`,
`CONV_MAX_IN_CH/OUT_CH/LINE_BUF_*/ACC_PERSIST_ENTRIES`). The matching
`test_*.py` assertions read the same constants. Don't hard-code a bound
literal in a fixture or its assertion — a JSON bump (e.g. `max_k: 2048 →
4096`) would otherwise silently turn a "must raise" model into a legal
one, masking validator regressions. Re-run the generator after any
`platforms/<name>.json` change.

## On-Device Testing and Benchmarking

Two scripts drive KV260 hardware over SSH:

| Script | Purpose | Config |
|--------|---------|--------|
| `run_remote_tests.py` | **Correctness** — generates a C project per model, builds on board, compares every output element against Python GT | `remote_config_*.json` |
| `run_remote_perf.py` | **Performance** — builds one benchmark project for all four kernels, runs parametric cases and reports latency (ms) and throughput (GB/s / GOps/s) | `perf_config.json` |

Both share the same SSH/driver config schema. See `doc/REMOTE_TESTING.md` for
the full reference including the `benchmarks` config section and per-kernel
case field definitions.

`run_remote_perf.py` validates each `VectorOPKernel` case's `op` against the
kernel-supported set (`OP_ADD..OP_RELU6`, 0..5) at config-load time and exits
with `config error: VectorOPKernel case '<label>': unsupported op=…` before
any SSH upload or remote build — same fail-fast contract as the scheduler's
hardware-bound check, but for the perf benchmark cases.

## Driver Sources

Each hardware kernel has its own driver generated by Vitis HLS synthesis.
When `--driver-dir` is omitted, `driver/` sub-directories are left empty with
stub `README.md` files.

| Kernel | Driver prefix | Required files |
|--------|--------------|----------------|
| VectorOPKernel | `xvectoropkernel` | `xvectoropkernel.h`, `xvectoropkernel_hw.h`, `xvectoropkernel.c`, `xvectoropkernel_sinit.c`, `xvectoropkernel_linux.c` |
| MatmulKernel | `xmatmulkernel` | `xmatmulkernel.h`, `xmatmulkernel_hw.h`, `xmatmulkernel.c`, … |
| ConvKernel | `xconvkernel` | `xconvkernel.h`, `xconvkernel_hw.h`, `xconvkernel.c`, … |
| PoolKernel | `xpoolingkernel` | `xpoolingkernel.h`, `xpoolingkernel_hw.h`, `xpoolingkernel.c`, … |
