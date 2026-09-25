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
| **MatmulKernel** | `MatMul` | Tiled 2-D matrix multiply |
| **ConvKernel** | `Conv` | 2-D NCHW convolution with optional bias |
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

**Host-side transformation (CPU loop, no hardware call):**
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
    │                           PoolNode       (PoolingKernel)
    │                           ReshapeNode    (buffer alias)
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
3. `_preprocess_model()` — rewrites `Gemm` → `MatMul` + optional `Add`.
4. Build tensor registry (weights, inputs, intermediates, outputs).
5. Dispatch each node to `MatmulNode` / `ConvNode` / `PoolNode` / `ReshapeNode`
   / `ScheduledNode` based on `op_type`.
6. `_fuse_activations()` (when `fuse_act=True`) — folds `Relu` / `Clip(0,6)`
   into the producing `ScheduledNode` (`act`, `fused_nodes`, output tensor
   re-pointed) and renumbers node indices.

`_space_to_depth_stems()` (when `s2d_stem=True`) runs between steps 3 and
4, on the ONNX model like the Gemm rewrite: it inserts the `SpaceToDepth`
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
  (`SpaceToDepthNode`), which invalidates a kernel-written source before
  reading it and flushes its own output before the consuming kernel starts.
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

---

## Weight layouts

Constant weights are emitted in the layout the target kernel reads
fastest, not in ONNX row-major order; `TensorInfo.packed_data` holds the
image and `numel` / the `.dat` file / the DMA buffer follow it, while
`data` / `shape` stay logical for the simulator:

| Kernel | Tensor | Layout |
|---|---|---|
| ConvKernel | weight, bias | tile-major `[M][ceil(C/16)][kH][kW][lanes]`, bias padded to 8 (CONV_OPTIMISATION §2.32 / §2.34); a space-to-depth stem's re-indexed `<W>_s2d` initializer is packed the same way |
| MatmulKernel | B (constant only) | tile-major `[ceil(M/16)][K][16]`, `b_packed = 1` on every consumer (MATMUL_OPTIMISATION §3b) |

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

// All graph inputs, then all graph outputs:
void inference_run(inference_buf_t *<input...>, inference_buf_t *<output...>);
void inference_deinit(void);
```
