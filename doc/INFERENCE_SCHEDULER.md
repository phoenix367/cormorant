# Inference Scheduler — Technical Reference

The inference scheduler is a Python code-generator that bridges the gap between
trained neural network models (ONNX format) and the hardware accelerators running
on the KV260.  It reads an ONNX graph, validates that every operator can be
executed by one of the supported hardware kernels, and emits a self-contained C
project that drives the IP through the auto-generated Xilinx driver APIs.

> **Full documentation** lives in
> [`inference-scheduler/doc/INFERENCE_SCHEDULER.md`](../inference-scheduler/doc/INFERENCE_SCHEDULER.md)
> and
> [`inference-scheduler/doc/ARCHITECTURE.md`](../inference-scheduler/doc/ARCHITECTURE.md).
> This file provides a quick orientation. For the DAG / event-stream /
> liveness / slot-coloring algorithms in detail, see
> [`SCHEDULER_DAG.md`](../inference-scheduler/doc/SCHEDULER_DAG.md).

---

## Hardware Kernels

| Kernel | ONNX ops handled | Notes |
|--------|-----------------|-------|
| **VectorOPKernel** | `Add`, `Sub`, `Mul`, `Div`, `Relu`, `Clip(0,6)` | 1-D element-wise, 8 elements/cycle on 128-bit ports; `act` register fuses a following `Relu` / `Clip(0,6)` |
| **MatmulKernel** | `MatMul` | Tiled 2-D matrix multiply — the MatMuls the ConvKernel lowering does not take (batch-1 FC layers, `K % 16 ≠ 0`, `M % 8 ≠ 0`, fewer than 16 rows, 4D×3D outer loops, or not estimated faster) |
| **ConvKernel** | `Conv`; `MatMul` (lowered) | 2-D NCHW convolution with optional bias; also runs MatMuls with swapped operand roles ([§MatMul on ConvKernel](#matmul-on-convkernel)) |
| **PoolingKernel** | `MaxPool`, `AveragePool`, `LpPool`, `GlobalMaxPool`, `GlobalAveragePool`, `GlobalLpPool` | 2-D NCHW pooling |

**Zero-cost transformations (no hardware call):**
- `Reshape` — output pointer is aliased to the source buffer; no data copy.
- `Gemm` — decomposed to `MatMul` + optional `Add` at model load time
  (`alpha=1, beta=1, transA=0, transB=0` required).
- `Relu` / `Clip(0,6)` after a VectorOP node — folded into that node's
  call via the kernel's `act` register (`run_op_act()`), when the producer's
  output has no other consumer and is not a graph output
  (`OnnxGraph(fuse_act=True)`, the CLI default; `--no-fuse-act` disables
  it; `OnnxGraph.act_fused_count` reports the number folded).  A `Relu`
  after a Conv / MatMul / Pool node or on a graph input stays a call.

**Host-CPU ops (C code inside `inference_run()`, no hardware call):**
- **Softmax, LayerNormalization, Gelu, Transpose, Slice / Split, Gather,
  OneHot, Cast** — `src/host_nodes.py`, see [§Host-CPU ops](#host-cpu-ops)
  below.  TensorFlow-style LayerNorm and GELU (tanh / erf) subgraphs are
  fused into single host nodes first ([§Pattern fusion](#pattern-fusion));
  integer tensors (token ids, masks) are supported as raw int16
  ([§Integer tensors](#integer-tensors)).  This is what makes BERT-base
  (bertsquad-12) schedulable — [`BERT_PLAN.md`](BERT_PLAN.md).
- **Space-to-depth stem** — a stride-2 `Conv` whose input has
  `4·C ≤ kTileIC` channels (C ≤ 4 on the KV260; the RGB stem of ResNet-18 /
  MobileNet-style nets) is rewritten as `SpaceToDepth(blocksize=2)` +
  stride-1 `Conv` over `4·C` channels with re-indexed weights
  (`OnnxGraph(s2d_stem=True)`, the CLI default; `--no-s2d-stem` disables
  it; `OnnxGraph.s2d_stem_count` reports the number rewritten).
  ConvKernel multiplies 16 input-channel lanes per cycle, so the 3-channel
  7×7 stem ran at 3/16 lane utilisation; the rewrite gives it 12 lanes and
  16 taps instead of 49.  The reorder runs on the host CPU as a
  `SpaceToDepthNode` (a C loop inside `inference_run()`, ~150 k elements
  for 224², staged through cached host memory because the DMA buffers are
  mapped non-cacheable — measured 16.3 ms when it read the BO directly);
  the model's public input stays `[1, C, H, W]`.  Details in
  [§Space-to-depth stem](#space-to-depth-stem) below.  A model that
  already contains an ONNX `SpaceToDepth` node is accepted the same way.

---

## Usage

```bash
cd inference-scheduler

# One-time setup
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Generate all test ONNX models
.venv/bin/python test/gen_test_models.py
.venv/bin/python test/gen_matmul_models.py
.venv/bin/python test/gen_mixed_kernel_models.py
.venv/bin/python test/gen_conv_models.py
.venv/bin/python test/gen_pool_models.py
.venv/bin/python test/gen_reshape_gemm_models.py
.venv/bin/python test/gen_mixed_all_kernels_models.py
.venv/bin/python test/gen_parallel_models.py    # parallel + NOP corner-case fixtures
.venv/bin/python test/gen_bert_models.py        # tiny BERT-like models (host ops, fusion)

# Generate a complete C inference project from an ONNX model
.venv/bin/python inference_scheduler.py model.onnx --out-dir /tmp/out

# Run the full test suite
.venv/bin/python -m pytest test/ -v
```

---

## Architecture

```
inference_scheduler.py          CLI, argument parsing
└── src/
    ├── graph.py    OnnxGraph   load, shape inference, Gemm preprocessing,
    │                           tensor registry, node dispatch
    ├── tensor.py   TensorInfo  weight encoding, C declarations
    ├── nodes.py                ScheduledNode  (VectorOPKernel)
    │                           MatmulNode     (MatmulKernel)
    │                           ConvNode       (ConvKernel)
    │                           MatmulConvNode (a MatMul on ConvKernel)
    │                           PoolNode       (PoolingKernel)
    │                           ReshapeNode    (buffer alias)
    │                           SpaceToDepthNode (host reorder)
    ├── host_nodes.py           HostNode family (Softmax, LayerNorm, Gelu,
    │                           Transpose, Slice, Gather, OneHot, Cast):
    │                           numpy reference + C helper library
    ├── fusion.py               Constant folding, Split lowering, LayerNorm /
    │                           GELU fusion, constant-broadcast normalisation
    ├── matmul_lowering.py      MatMul -> ConvKernel engine choice and geometry
    ├── cost_model.py           ConvKernel / MatmulKernel cycle estimates
    ├── schedule.py Dag         data-flow DAG: predecessors, successors,
    │                           topological order, independent pairs
    └── codegen/    CodeGenerator
                    _core.py    event stream, tensor layout, DMA pool sizing,
                                event-stream liveness intervals
                    _header.py  include/inference.h
                    _source.py  src/inference.c  (weights, init, run, kernel_wait)
                    _simulate.py  fixed-point forward simulation
                    _test.py    test/test_inference.c  (on-device smoke test)
                    _cmake.py   CMakeLists.txt
```

### OnnxGraph loading sequence

1. `onnx.load()` + `onnx.checker.check_model()` — structural validation.
2. `shape_inference.infer_shapes()` — fills intermediate tensor shapes.
3. `fusion.fold_constant_nodes()` — `Constant` nodes become initializers.
4. `_preprocess_model()` — rewrites `Gemm` → `MatMul` + optional `Add`.
5. `fusion.lower_split()` — `Split` becomes one `Slice` per output.
6. `_space_to_depth_stems()` (when `s2d_stem=True`, see below).
7. `fusion.fuse_patterns()` (when `fuse_patterns=True`, the default) —
   LayerNorm / GELU fusion and VectorOP constant-broadcast normalisation.
8. Build tensor registry (weights, inputs, intermediates, outputs).
9. Dispatch each node to `MatmulNode` / `ConvNode` / `PoolNode` / `ReshapeNode`
   / `ScheduledNode` / a host node (`host_nodes.HOST_OP_FACTORIES`) based on
   `op_type`; kernel nodes reading an integer tensor are rejected.
10. `_fuse_activations()` (when `fuse_act=True`) — folds `Relu` / `Clip(0,6)`
   into the producing `ScheduledNode` (`act`, `fused_nodes`, output tensor
   re-pointed) and renumbers node indices.
11. `matmul_lowering.lower_matmuls()` (`matmul_on_conv="auto"`, the
   default) — MatMuls estimated faster on ConvKernel become
   `MatmulConvNode`s, their constant B re-laid out when `kw > 1`
   ([§MatMul on ConvKernel](#matmul-on-convkernel)).
12. `_pack_matmul_weights()` (the remaining MatmulNodes), then
   `_choose_slice_views()` (contiguous Slice pieces that may alias their
   source).

`_space_to_depth_stems()` (when `s2d_stem=True`) runs on the ONNX model
like the Gemm rewrite: it inserts the `SpaceToDepth`
node, appends the `<W>_s2d` initializer and replaces the Conv, so the
tensor registry, `ConvNode` validation / weight packing and the report see
an ordinary graph.

**VectorOPKernel alignment contract** (kernel ports are 128-bit words):
every DMA buffer base is 64-byte aligned, every broadcast `CHUNK_STRIDE`
is `INFERENCE_ALIGN_UP(CHUNK)` (a multiple of 8 elements), and
`inference_buf_alloc()` rounds allocations up to 64 bytes, so the kernel's
whole-word reads and its whole last-word write per run stay inside the
buffer / stride gap (`test/test_act_fusion.py::TestAlignmentContract`).

### Key data flow

```
model.onnx  →  OnnxGraph  →  Dag  →  CodeGenerator  →  C project
                                          │
                       ┌──────────────────┼──────────────────┐
                       │                  │                  │
                  event stream      tensor layout         emit:
                  (Start/Wait      (alloc sizes,         run_*()      Start kernel
                   per node)        strides,             kernel_wait  Block on lane
                                    pool slots from
                                    event-stream
                                    liveness intervals)
```

### Parallel kernel execution

Each hardware lane (`Conv`, `Pool`, `Matmul`, `VectorOP`) is a single IP
on the FPGA, and the codegen overlaps work across **different** lanes.
The `run_*()` helpers are non-blocking — they program AXI-Lite
registers, call `XKernel_Start()`, and return. Synchronisation is a
single weak-symbol primitive emitted into the generated source:

```c
typedef enum { KERNEL_VECTOROP, KERNEL_MATMUL, KERNEL_CONV, KERNEL_POOL,
               KERNEL_COUNT } kernel_id_t;
__attribute__((weak))
void kernel_wait(kernel_id_t k);   /* default: poll IsDone */
```

Override at link time to swap in IRQ-driven waiting (UIO/Linux,
GIC/bare-metal) without touching the generated code.

`inference_run()` emits a `kernel_wait` only when (a) some predecessor
is still in flight on its lane, or (b) the target lane has a different
op pending. Predecessor analysis walks **through** ReshapeNode chains
to reach the real producing kernel, so a `Pool → Squeeze → MatMul`
chain still drains the Pool lane before MatMul reads the alias.

### DMA buffer management

- All buffers allocated in `inference_init()` from a contiguous DMA pool.
- `Reshape` output buffers are pointer-assigned (= source), never independently allocated.
- Pool slots are coloured by **event-stream liveness intervals**: a tensor
  is live from its producer's Start event until the latest `kernel_wait`
  that drains a consumer's lane (consumers reached via Reshape aliases
  count). Two tensors share a slot only if their event intervals are
  strictly disjoint — necessary for correctness under cross-lane parallelism.
- `inference_run()` flushes all graph inputs to DDR at the top, drains
  every still-pending lane, and invalidates all graph outputs at the bottom.
  Internal intermediate buffers are never synced — the PL kernels access
  DDR directly via their AXI master ports — except around a host op
  (`SpaceToDepthNode`, `HostNode`), which invalidates a kernel-written
  source before reading it and flushes its own output before the consuming
  kernel starts.
- Weights are synced once at init; they never change.

### Space-to-depth stem

`OnnxGraph._space_to_depth_stems` (`src/graph.py`; maths in
`nodes._s2d_stem_geometry` / `nodes._s2d_stem_weight`).

Per axis, a stride-2 tap `t ∈ 0..k-1` of the original Conv with pad `p`
reads input row `2o + t − p`.  After a block-2 space-to-depth the same row
is `(r, ph)` with `2r + ph = 2o + t − p`.  A stride-1 Conv over the
reordered tensor with pad `P'` reads rows `o + R − P'`, hence
`t = 2R + ph + p − 2P'`.  Choosing

```
P'  = ceil(p / 2)              off = p − 2P'  (0 or −1)
K'  = (k − 1 − off) // 2 + 1   t   = 2R + ph + off
```

covers every original tap exactly once (taps outside `0..k-1` are zero
weights), so both convolutions accumulate the same products and the
scheduler's fixed-point simulation truncates once after the whole sum in
both cases — the outputs are bit-identical (`test/test_s2d_stem.py`).

| original | rewritten (per axis) |
|---|---|
| 7×7 s2 p3 (ResNet-18) | 4×4 s1 P'=2, off −1 (one zero tap row / column) |
| 5×5 s2 p2 | 3×3 s1 P'=1 |
| 3×3 s2 p1 | 2×2 s1 P'=1 |
| 3×3 s2 p0 | 2×2 s1 P'=0 |

Tensors (ONNX `SpaceToDepth` channel order, so the rewritten model is a
valid ONNX graph):

```
x'[n][(ph·2 + pw)·C + c][r][cc]  = x[n][c][2r + ph][2cc + pw]
w'[m][(ph·2 + pw)·C + c][R][Cc]  = w[m][c][2R + ph + off_h][2Cc + pw + off_w]   (0 outside kh×kw)
bias, output shape               unchanged
```

The rewritten Conv's ONNX `pads` are `[P't, P'l, Pb', Pr']` with the
bottom / right values chosen so shape inference reproduces the original
`out_h`/`out_w` (`[2, 2, 1, 1]` for 224² → 112²).  ConvKernel takes only
`pad_top` / `pad_left` and zero-pads bottom / right by its bounds check
(`ConvKernel.h`), which is exactly the implicit padding the identity
relies on; the geometry is covered by C-sim (`TestConvSim.cpp`, "s2d
stem: 12ch 4x4 s1 pad [2,2,1,1]").

Applicability (all required): `group = 1`, dilations 1, `strides = [2, 2]`,
explicit pads (`auto_pad` absent / NOTSET), constant 4-D weight, 4-D input
of known shape with **even H and W** (odd sizes are left untouched — the
ONNX op requires divisibility and the reorder loop stays branch-free; the
maths would also hold with a zero-padded odd edge), and `4·C ≤ kTileIC`
(`platforms/<name>.json kernels.conv.tile_ic`, 16 → C ≤ 4).  Any input
tensor qualifies; when the source is a kernel output the event stream
waits for that lane first.  One `SpaceToDepth` output is shared by every
rewritten Conv reading the same tensor.

Codegen: `SpaceToDepthNode` has no lane (`kernel_name == ""`), emits a
`('cpu', idx)` event (waits on its producers like a kernel start, runs
inline, never waited on), and its output is a normal pool buffer whose
liveness interval starts and ends at that event, so it never shares a
slot with its source.  Profiling brackets the loop like a synchronous
node.  The `<W>_s2d` weight goes through the normal ConvNode tile-major
packing and ROM / `.dat` emission.

The loop never reads the DMA buffers element-wise.  On the KV260 the DMA
pool is an XRT BO whose CPU mapping is non-cacheable, so the strided
2-byte source loads of the reorder cost ~100 ns each (16.3 ms for the
ResNet-18 stem on the board).  `inference_init()` mallocs one cached
staging block per node (`_s2d_stage_<out>`, 2 × source numel, freed in
`inference_deinit()`); at run time the node does `memcpy(BO → stage_in)`,
reorders `stage_in → stage_out` entirely in cached memory, `memcpy(stage_out
→ BO)` and then `inference_buf_sync_to_device(out)` — the only DMA-memory
traffic is two wide sequential copies (~0.1–0.3 ms for 300 KB).

### Host-CPU ops

`src/host_nodes.py`.  Ops no PL kernel implements run on the A53 inside
`inference_run()`:

| ONNX op | Node | Semantics / restrictions |
|---|---|---|
| `Softmax` | `SoftmaxNode` | opset ≥ 13: last axis only; opset < 13: "coerce to 2-D", i.e. rows of `prod(shape[axis:])` (BERT's `axis = 3` on rank 4 is the last axis) |
| `LayerNormalization` | `LayerNormNode` | over `prod(shape[axis:])`; scale / bias must be constants (kept float32, emitted as C arrays, never DMA weights); only the `Y` output |
| `Gelu` | `GeluNode` | `approximate = "tanh"` / `"none"` (erf) |
| `Transpose` | `TransposeNode` | any perm, ≤ 5 non-mergeable dims |
| `Slice` (and every `Split` output) | `SliceNode` | constant starts / ends / axes, positive steps; zero-cost view when possible (below) |
| `Gather` | `GatherNode` | axis 0, runtime integer indices, any table (a constant table stays a DMA weight; rows are copied straight out of it, one memcpy per row).  Negative indices count from the end; an index still outside `[0, rows)` is **clamped** (ONNX leaves it undefined) |
| `OneHot` | `OneHotNode` | axis −1, runtime integer indices, constant depth and `[off, on]`; out-of-range index → all-off row |
| `Cast` | `CastNode` | integer → Data_t, Data_t → integer (truncate toward zero), → bool; a cast within one storage kind (float → float, int → int) is a zero-cost `ReshapeNode` alias |

**Numeric contract** (the generated C and `HostNode.reference()`, which
`_simulate` runs, implement the same operations in the same order):

- inputs are read as double — Data_t via `host_ld` (`(int16_t)bits / 256.0`,
  exact), integer tensors via `host_ld_int` (raw int16);
- all arithmetic in IEEE double; reductions accumulate left to right
  (`np.cumsum` in the simulator); the host section of `inference.c` starts
  with `#pragma GCC optimize ("fp-contract=off")` (GNU C modes otherwise
  fuse `a*b + c` into an FMA on the A53) and CMake adds `-ffp-contract=off`;
- `exp` / `tanh` / `erf` come from libm; the simulator calls Python's
  `math` module — the platform's glibc — elementwise, not numpy: `np.tanh`
  differs from glibc's `tanh` by up to 3 ulp on ~26 % of inputs (numpy 2.x),
  and `np.exp` switches to SVML on AVX-512 hosts;
- outputs are written back with `host_st`: `nearbyint(v * 256)` under the
  default FE_TONEAREST mode — **round half to even**, identical to numpy's
  `np.round` — then saturation to `[-128, 127.996]`; NaN → 0.  Integer
  outputs use `host_st_int` (truncate toward zero, saturate to int16);
- formulas: Softmax `exp(x − max) / Σ exp(x − max)`; LayerNorm TF form
  (fused BERT pattern) `g = gamma / sqrt(var + eps)`,
  `y = x·g + (beta − mean·g)`, ONNX form `(x − mean)·inv·gamma + beta`, with
  `mean = Σx/n`, `var = Σ(x − mean)²/n`; GELU tanh
  `x·(0.5·(1 + tanh(c2·(x + c1·x³))))`, erf `x·(0.5·(1 + erf(x/k)))` with the
  constants found in the graph (native `Gelu`: exact `√(2/π)`, `0.044715`,
  `√2`).

Kernel ops keep their semantics (products / sums exact, AP_TRN floor +
AP_SAT on the output), so a model's data path mixes floor (kernels) and
round-half-even (host) write-backs; the BERT study's `sched` policy emulates
exactly this mix.

**Staging.**  On the KV260 the buffer pool is an XRT BO mapped
non-cacheable (a strided 2-byte read costs ~100 ns).  Every host op does one
wide `memcpy` per input from the BO into `s_host_stage` — a malloc'd
(cached) arena shared by all host ops, since they run one at a time — then
computes stage → stage and copies the result back with one `memcpy`
followed by `inference_buf_sync_to_device()`; a kernel-written input is
invalidated (`inference_buf_sync_from_device`) first.  `host_load` /
`host_store` also compact / re-expand an advancing-strided layout (a
broadcast VectorOP neighbour with an unaligned chunk), e.g. BERT's
`[256, 2]` logits (chunk 2, stride 8) before the final Transpose.  Gather
copies whole table rows straight from the (weight) BO instead.  The arena
is sized to the largest op (BERT-base: 1 573 888 elements, 3 MiB — a
softmax input + output + one row of doubles).

**Scheduling.**  Like `SpaceToDepthNode`: no lane, one synchronous
`('cpu', idx)` event that first waits for in-flight producers, liveness
interval that starts and ends at that event (so its output never shares a
pool slot with its input), profiled with `INFERENCE_PROF_BEGIN/END`.  Host
ops run in graph order; kernel work that does not depend on them is not
yet hoisted around them (phase-2 item).

**Slice views.**  `OnnxGraph._choose_slice_views` turns a `Slice` piece
into a zero-cost `inference_buf_init_view()` of its source (set up once in
`inference_init()`, no pool slot, liveness through the root like a Reshape
alias) when the piece is contiguous, its byte offset is a multiple of 64,
the source's root buffer is an internal pool buffer (not a graph input /
weight / output and not redirected to a graph output at run time), and the
piece itself never reaches a graph output.  The codegen demotes a view to
a host copy when a broadcast consumer gives the piece or its root a strided
layout.  BERT's final `Split` feeds the graph outputs, so it is copied.

### Pattern fusion

`src/fusion.py`, `OnnxGraph(fuse_patterns=True)` — ON by default in the
library and the CLI (`--no-fuse-patterns`); it only changes graphs that
contain these patterns (all 147 generatable existing test models produce
byte-identical projects).  Matching is structural — producer / consumer
links, op types and constant values — never node names; every
intermediate of a match must have no consumer outside it and must not be a
graph output.

| Pattern | Matched arrangement | Fused to |
|---|---|---|
| TF LayerNorm (bertsquad-12, 12 nodes) | `mean = ReduceMean(x, last, keepdims)`, `d = Sub(x, mean)`, `Mul(d, d)` \| `Pow(d, 2)`, `ReduceMean`, `Add(eps)`, `Sqrt`, `Reciprocal` \| `Div(1, ·)`, `g = Mul(inv, gamma)`, `Mul(mean, g)`, `Sub(beta, ·)`, `Mul(x, g)`, `Add` — commutative operands in any order, gamma / beta constant `[n]`, eps constant scalar | `LayerNormalization` (TF form) |
| GELU tanh (BERT, 8 nodes) | `Pow(x, 3)` \| `x·(x·x)`, `Mul(c1≈0.044715)`, `Add(x)`, `Mul(c2≈√(2/π))`, `Tanh`, `Add(1)`, then `x·(0.5·a)` \| `(x·a)·0.5` \| `(x·0.5)·a` | `Gelu(approximate="tanh")` with the graph's c1 / c2 |
| GELU erf (PyTorch export) | `Div(x, k≈√2)` \| `Mul(x, k≈1/√2)`, `Erf`, `Add(1)`, then the same three tails | `Gelu(approximate="none")` with the graph's k |
| native `LayerNormalization` / `Gelu` | — | dispatched directly (ONNX form / exact constants) |

Constants are compared with a relative tolerance of 1e-5 (c1, c2, k) or
exactly (0.5, 1, 2, 3).  A near miss is left alone and its `ReduceMean` /
`Pow` / `Sqrt` / `Reciprocal` / `Tanh` / `Erf` fails node dispatch with a
hint (lowering them op by op would saturate Q8.8: x² and x³ overflow at
|x| ≥ 11.3 / 5.04).  Because the fused nodes carry the graph's float32
constants and evaluate the graph's own formula in double, a fused region
computes exactly what the subgraph computes op by op in float64
(`test/test_fusion.py` checks it bit for bit).

**Constant broadcast normalisation** (same flag; values unchanged):
a scalar constant operand of a VectorOP `Add` / `Sub` / `Mul` / `Div` on a
tensor whose last dim L is a multiple of 8 and ≤ 2048 becomes an `[L]`
vector — one repeating chunk of L instead of L one-element chunks at
stride 8 (BERT's ×1/8 score scale, `1 − mask`, `× −10000`); and when the
runtime operand is itself broadcast (the kernel repeats one side only), a
constant the kernel cannot broadcast is pre-broadcast to the output shape
(BERT's `ones[1,S,1] · mask[1,1,S]`).  VectorOP's broadcast rule also
treats size-1 output dims as neutral now, so `[1,1,S,S]` onto `[1,H,S,S]`
(the attention mask) is a plain repeating chunk.

### Integer tensors

Integer / bool ONNX tensors (`TensorInfo.is_int`) — BERT's `input_ids`,
`segment_ids`, `input_mask`, `unique_ids` — are stored in `inference_buf_t`
as **raw signed integers** of the element width (`int16_t` for
ap_fixed<16,8>, i.e. `(Data_t)(int16_t)id`), not in the fixed-point
encoding; values must fit (a 30 522-entry vocabulary does).  The generated
`inference.h` lists them.  Only host ops (Gather, OneHot, Cast, data
movement) and buffer aliases may read them; a kernel node reading one is
rejected ("must go through a Cast").  `Identity` of an integer input to an
output is a copy (`unique_ids` passthrough).  The generated test harness
fills an integer input with `p[i] = i % R` — R the smallest Gather table /
OneHot depth that reads it, else 2 (a 0/1 mask) — and compares integer
outputs exactly (printed with `%d`).

### MatMul on ConvKernel

[`BERT_PLAN.md`](BERT_PLAN.md) §2 2A.  ConvKernel's 16 × 16 MAC grid runs
two output pixels per cycle (512 MACs, CONV_OPTIMISATION §2.42) against
MatmulKernel's 32; a MatMul runs on it with **swapped operand roles**.  For
`C[N][M] = A[N][K] · B[K][M]` (per batch item) one ConvKernel call has

| conv | = |
|---|---|
| `out_ch` | `N` (tokens, for a transformer linear) |
| `in_ch`, kernel, stride | `K / kw`, `1 × kw`, `(1, kw)`; no padding, dilation 1, no bias |
| output `out_h × out_w` | `M`; `in_h = out_h`, `in_w = kw · out_w` |
| weight | **A** — row-major `[N][K]` *is* the packed tile-major `[N][K/(16 kw)][1][kw][16]` when `K % (16 kw) == 0`: no packing, no copy |
| x | **B** — `x[c][kw·p + j] = B[(c/16)·16·kw + j·16 + c%16][p]` (`p = h·out_w + ow`); for `kw = 1` that is B's own row-major `[K][M]` |
| y | **C** row-major `[N][M]` |

A Gemm's bias stays the VectorOP Add.  Both kernels multiply Q8.8 operands
exactly, sum in `ap_fixed<32,16>` and floor + saturate, so the result is
**bit-identical** to MatmulKernel's: `_simulate` runs a `MatmulConvNode`
exactly like a `MatmulNode` (`np.matmul`, truncate), and
`test/test_matmul_on_conv.py` proves the mapping (A read through
ConvKernel.h's packed-weight formula and B's image read as NCHW `x` give
`A · B` through the conv reference; the generated C run on the host
against a software ConvKernel reproduces the simulation bit for bit —
also for BERT-base, `test/test_bert_base.py`).  The kernel side is
covered by `TestConvSim.cpp`'s `mm-on-conv` cases (C-sim and RTL: 1×2 /
1×3 / 1×4 with stride = kw, 1×1 with out_h > 1, M-groups and oh-chunks,
in_ch 1024, weights read at offsets inside a shared buffer).

**Eligibility** (`matmul_lowering.ineligible_reason`): ap_fixed<16,8>;
`outer_count == 1`; `N > 1`; `K % 16 == 0` (no pad lanes in the weight
tile) and `M % 8 == 0` (every row / batch slice of B and C starts on a
16-byte word, so no broadcast consumer can give A, B or C a gapped
layout — `_compute_tensor_layouts` still checks); ConvKernel's bounds
(`out_ch ≤ max_out_ch`, `K/kw ≤ max_in_ch`, `out_w · ceil(N/16)·16 ≤
max_acc_persist_entries`, `kw ≤ max_kw`).

**Kernel width.**  B is used as is (`kw = 1`) when it is an activation or
a constant something else also reads.  A constant B read only by this
MatMul is emitted in the `kw` layout above (`nodes.conv_lowered_b_image`
→ `TensorInfo.packed_data`, the `.dat` / ROM image; `data` stays logical)
and every `kw` with `K % (16 kw) == 0` is a candidate: the §2.42 sweep
spends `max(kh·kw, 2)` cycles per pixel pair, so `kw ≥ 2` halves a 1×1's
sweep (which pays a dummy second position).

**Engine choice** (`--matmul-on-conv auto`, the default): every
`(kw, out_w | M)` geometry is costed with `cost_model.conv_cycles` — the
standard path of the conv-cycle-model skill (§2.42), kept equal to the
skill script by a test — plus `CALL_OVERHEAD` (1 500 cycles) per call;
the cheapest is compared with `cost_model.matmul_cycles`, a block model of
MatmulKernel calibrated on the board (256³, the BERT linears, attention
QKᵀ within 1–4 %).  The MatMul is lowered when the conv estimate is below
0.9 × MatmulKernel's and the conv has at least one full 16-row output tile
(`out_ch ≥ kTileM`; below that both kernels are dominated by fixed
per-call costs).  `--matmul-on-conv always` lowers every eligible MatMul
(tests), `--no-matmul-on-conv` / `off` none.  The CNN models' only
MatMuls are batch-1 classifier FCs (`N = 1`), so their generated projects
are byte-identical with and without the pass.

**Batches** (`matmul_lowering._batch_mode`, always `outer_count == 1`):
A shared and B batched → one call with ConvKernel's `batch` (its batch
shares the weights); B shared and A batched → the batch folds into the rows
(`out_ch = batch·N`, one call); both batched (attention, one weight matrix
per head) → one call per item, `run_conv_at()` with element offsets, each
waiting on `KERNEL_CONV` for the previous one; the last call is left in
flight like any other start, so the event stream / liveness are unchanged.

BERT-base (bertsquad-12, 386 nodes): 96 of the 98 MatMuls run on ConvKernel
— the 72 encoder linears as 1×4 convs (`in_ch` 192 / 768, output 48×16 or
192×16) and the 24 attention MatMuls as 12 per-head 1×1 calls each — in 360
ConvKernel calls; the K = 2 token-type MatMul and the M = 2 span head stay
on MatmulKernel.  Cost model at 100 MHz: 0.62 s of MatMul per inference
against 8.37 s on MatmulKernel (the phase-1 board measured 8.41 s).
Board results: BERT_PLAN §3.

---

## Weight layouts

Constant weights are emitted in the layout the target kernel reads
fastest, not in ONNX row-major order; `TensorInfo.packed_data` holds the
image and `numel` / the `.dat` file / the DMA buffer follow it, while
`data` / `shape` stay logical for the simulator:

| Kernel | Tensor | Layout |
|---|---|---|
| ConvKernel | weight, bias | tile-major `[M][ceil(C/16)][kH][kW][lanes]`, bias padded to 8 (CONV_OPTIMISATION §2.32 / §2.34); a space-to-depth stem's re-indexed `<W>_s2d` initializer is packed the same way |
| MatmulKernel | B (constant only) | tile-major `[ceil(M/32)][K][32]`, `b_packed = 1` on every consumer (MATMUL_OPTIMISATION §3b, §8) |
| ConvKernel (MatMul on ConvKernel) | B (constant, read only by that MatMul, `kw > 1`) | `x[c][kw·p + j] = B[(c/16)·16·kw + j·16 + c%16][p]` per batch slice ([§MatMul on ConvKernel](#matmul-on-convkernel)); A needs none |

A MatMul B is packed only when every reader of the tensor is a MatMul
using it as B with the same `(k, m)` (`OnnxGraph._pack_matmul_weights`);
activations and shared constants stay row-major and the kernel reads them
through its per-row path.

## Data Type

Default: `ap_fixed<16,8>` — 16-bit two's complement, 8 integer bits, 8
fractional bits. Encoding: `1.0 = 0x0100`, `0.5 = 0x0080`, range `[-128, 127.996]`.

The `DataType` abstraction in `src/dtype.py` allows other types (e.g. `float32`)
to be plugged in without changing any other source file.

---

## Generated C API

```c
// inference.h
typedef uint16_t Data_t;              // ap_fixed<16,8>
#define INFERENCE_BYTES_PER_ELEM  2u
#define INFERENCE_ALIGN_BYTES     16u
#define INFERENCE_BUF_POOL_SIZE_BYTES  N

// One per active kernel (only present kernels appear):
int  inference_init(const char *vectoropkernel_instance
                    [, const char *matmulkernel_instance]
                    [, const char *convkernel_instance]
                    [, const char *poolkernel_instance]);

// All graph inputs, then all graph outputs (integer tensors hold raw int16):
void inference_run(inference_buf_t *<input...>, inference_buf_t *<output...>);
void inference_deinit(void);
```

BERT-base (`bertsquad-12-simplified.onnx`) for example:

```c
int  inference_init(const char *vectoropkernel_instance,
                    const char *matmulkernel_instance);
void inference_run(inference_buf_t *unique_ids_raw_output_9_0,   /* int64 [1]      */
                   inference_buf_t *segment_ids_0,               /* int64 [1, 256] */
                   inference_buf_t *input_mask_0,                /* int64 [1, 256] */
                   inference_buf_t *input_ids_0,                 /* int64 [1, 256] */
                   inference_buf_t *unstack_1,                   /* end logits   [1, 256] */
                   inference_buf_t *unstack_0,                   /* start logits [1, 256] */
                   inference_buf_t *unique_ids_0);               /* int64 [1]      */
```
