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
  DDR directly via their AXI master ports.
- Weights are synced once at init; they never change.

---

## Weight layouts

Constant weights are emitted in the layout the target kernel reads
fastest, not in ONNX row-major order; `TensorInfo.packed_data` holds the
image and `numel` / the `.dat` file / the DMA buffer follow it, while
`data` / `shape` stay logical for the simulator:

| Kernel | Tensor | Layout |
|---|---|---|
| ConvKernel | weight, bias | tile-major `[M][ceil(C/16)][kH][kW][lanes]`, bias padded to 8 (CONV_OPTIMISATION §2.32 / §2.34) |
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
