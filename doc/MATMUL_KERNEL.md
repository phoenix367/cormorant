# Matmul Kernel — Detailed Implementation Description

## Overview

`MatmulKernel` is a Vitis HLS kernel implementing tiled matrix multiplication
`C = A × B` on the Xilinx KV260 FPGA. It is one of four hardware kernels in
the `axi_demo` project. The kernel computes a batch of independent 2-D
matrix products on row-major matrices, with optional per-operand batch
broadcasting (a stride of 0 reuses `A` or `B` across every batch step).

A three-level tiling strategy (output rows `kTileN` × output columns
`kTileM` × inner dimension `kTileK`) keeps the working set on-chip and feeds
an II=1 K-reduction loop. Row-lane rotation (`n1 = ki % kTileN`) breaks the
accumulator read-after-write hazard so the inner loop sustains one K-step per
clock across `kTileM` parallel MAC lanes.

---

## 1. AXI Interface

**Memory ports (m_axi):**

| Bundle | Port | Direction | Description |
|--------|------|-----------|-------------|
| `gmem0` | `a` | Read | Matrix A `[n][k]`, row-major — `hls::burst_maxi<ap_uint<128>>`, 8 elements per beat (MATMUL_OPTIMISATION §3) |
| `gmem1` | `b` | Read | Matrix B `[k][m]`, row-major — `hls::burst_maxi<ap_uint<128>>`, 8 elements per beat |
| `gmem2` | `c` | Write | Matrix C `[n][m]`, row-major — 16-bit element port |

Keeping A, B, and C on separate AXI buses lets HLS issue their reads and
writes concurrently.

**AXI-Lite control registers (`s_axilite bundle=ctrl`):**

| Register | Type | Description |
|----------|------|-------------|
| `a`, `b`, `c` | `uint64_t` | Physical DDR base addresses |
| `n` | `unsigned` | Rows of A / rows of C |
| `k` | `unsigned` | Inner dimension (cols of A = rows of B); `k ≤ kMaxK` |
| `m` | `unsigned` | Cols of B / cols of C |
| `batch` | `unsigned` | Number of independent 2-D products |
| `a_batch_stride` | `unsigned` | Elements to advance `a` per batch step (0 = broadcast) |
| `b_batch_stride` | `unsigned` | Elements to advance `b` per batch step (0 = broadcast) |
| `c_batch_stride` | `unsigned` | Elements to advance `c` per batch step |
| `return` | — | `ap_ctrl_hs` (start / done / idle / ready) |

Memory layout is row-major: `A[row·k + col]`, `B[row·m + col]`,
`C[row·m + col]`.  `a` and `b` must be 16-byte aligned base addresses;
the kernel derives every row's covering 128-bit word range itself, so
batch strides and tile offsets are ordinary element offsets.

---

## 2. Compile-Time Configuration (`Config.h.in`)

CMake substitutes the data types and tile constants into `Config.h`:

| Constant | Default | Purpose |
|----------|---------|---------|
| `Data_t` | `ap_fixed<16,8>` | Element type (2-byte, range \[-128, 127.996\]) |
| `AccData_t` | `ap_fixed<32,16>` | Accumulator type (wider range, avoids overflow) |
| `kTileN` | 4 | Output-row tile / accumulator-lane interleave depth. Power of 2; must be ≥ MAC latency (≈3) for II=1 |
| `kTileM` | 16 | Output columns processed per cycle — one DSP accumulator lane each. Power of 2 |
| `kTileK` | 256 | On-chip B-buffer K-slice depth. Power of 2 (so `k_tile` indexing needs no divider) |
| `kMaxK` | 2048 | Compile-time upper bound on the inner dimension `K`; sizes `a_buf`. Models with `K > kMaxK` are rejected by the scheduler |

If Vitis HLS headers are unavailable at configure time, the types fall back
to `float` / `double`.

**`AccData_t` overflow budget.** With `ap_fixed<16,8>` operands the
worst-case product is `127.996² ≈ 16383`; `ap_fixed<32,16>` saturates near
`32767`, so accumulation is exact for `K ≤ 2` at full-scale inputs and for
`K ≤ ~200` at the typical neural-net range `|v| ≤ 8`. Larger `K` budgets
need a wider accumulator (`-DMM_ACC_DATA_TYPE=ap_fixed<40,24>`).

**Runtime constraint validated by the scheduler:** `k ≤ kMaxK`. `n`, `m`,
and `batch` are unbounded — they are handled by the `n_tile` / `m_tile` /
`batch` loops.

---

## 3. On-Chip Memory

Three `static` on-chip buffers stage the tiles between DDR and the MAC array
(declared `static` so HLS infers BRAM/registers; in C-sim the storage
persists across calls, which is safe because every element is written before
it is read within a call):

| Buffer | Shape | Storage | Holds |
|--------|-------|---------|-------|
| `a_buf` | `[kTileN][kMaxK]` | `kTileN` BRAMs (partition dim 1) | `kTileN` full rows of A for the current `n_tile`; loaded once, reused across all `m_tile`s and `k_tile`s |
| `b_tile` | `[kTileK][kTileM]` | `kTileM` BRAMs (partition dim 2) | One `kTileK × kTileM` block of B; reloaded from DDR per `(m_tile, k_tile)` |
| `acc` | `[kTileN][kTileM]` | registers (partition dim 0) | `kTileN × kTileM` partial dot products; cleared per `m_tile` |

```cpp
static Data_t    a_buf [kTileN][kMaxK];
static Data_t    b_tile[kTileK][kTileM];
static AccData_t acc   [kTileN][kTileM];
#pragma HLS ARRAY_PARTITION variable=a_buf  complete dim=1   // kTileN parallel row banks
#pragma HLS ARRAY_PARTITION variable=b_tile complete dim=2   // kTileM parallel column banks
#pragma HLS ARRAY_PARTITION variable=acc    complete dim=0   // all kTileN·kTileM in registers
```

`a_buf` is partitioned on dim 1 so all `kTileN` rows can be read in the same
cycle — the II=1 K-reduction reads `a_buf[n1][…]` for a different `n1` each
iteration. `b_tile` is partitioned on dim 2 so the `kTileM`-wide unrolled
inner loop reads one element per column bank per cycle. `acc` is fully
partitioned so all `kTileN·kTileM` accumulators are independent registers.

The kernel is **not** a `DATAFLOW` design — it is a single sequential loop
nest with each load / reduce / write loop pipelined at II=1.

---

## 4. Loop Structure and HLS Pragmas

```
for bi in [0, batch)                              // a/b/c advanced by *_batch_stride
  for n_tile in [0, ceil(n / kTileN))
    n_off, n_valid = n_tile·kTileN, min(kTileN, n - n_off)

    // LOAD a_buf — n_valid rows × k columns, one burst read per row   PIPELINE II=1
    for n1 in [0, n_valid): for ki in [0, k):
      a_buf[n1][ki] = a[(n_off + n1)·k + ki]

    for m_tile in [0, ceil(m / kTileM))
      m_off, m_valid = m_tile·kTileM, min(kTileM, m - m_off)

      // CLEAR acc — kTileN × kTileM registers, fully unrolled → 1 cycle
      for n1, m1 (UNROLL): acc[n1][m1] = 0

      for k_tile in [0, ceil(k / kTileK))
        k_off, k_valid = k_tile·kTileK, min(kTileK, k - k_off)

        // LOAD b_tile — k_valid rows × m_valid cols, burst read per row  PIPELINE II=1
        for k1 in [0, k_valid): for m1 in [0, m_valid):
          b_tile[k1][m1] = b[(k_off + k1)·m + (m_off + m1)]

        // K-REDUCTION — iterates k_valid·kTileN times                    PIPELINE II=1
        for ki in [0, k_valid·kTileN):
          n1 = ki % kTileN          // row lane — rotates 0..kTileN-1
          kk = ki / kTileN          // K index local to this k_tile
          a_val = a_buf[n1][k_off + kk]
          for m1 in [0, kTileM) UNROLL:
            acc[n1][m1] += AccData_t(a_val) · AccData_t(b_tile[kk][m1])

      // WRITE C — saturate_cast acc → C, burst write per row             PIPELINE II=1
      for n1 in [0, n_valid): for m1 in [0, m_valid):
        c[(n_off + n1)·m + (m_off + m1)] = saturate_cast<Data_t>(acc[n1][m1])
```

`A` is loaded once per `n_tile` and reused across every `m_tile`/`k_tile`;
`B` is reloaded per `(m_tile, k_tile)`. Partial last tiles load only the
valid rows/columns — unused `a_buf`/`b_tile` lanes hold stale data but feed
`acc` lanes that are never written out to `C`.

### HLS pragmas applied

| Pragma | Location | Effect |
|--------|----------|--------|
| `INTERFACE m_axi … bundle=gmem0/1/2` | top-level | AXI memory ports for A / B / C |
| `INTERFACE s_axilite … bundle=ctrl` | every scalar + `return` | AXI-Lite register file |
| `ARRAY_PARTITION variable=a_buf complete dim=1` | `a_buf[kTileN][kMaxK]` | `kTileN` parallel row banks |
| `ARRAY_PARTITION variable=b_tile complete dim=2` | `b_tile[kTileK][kTileM]` | `kTileM` parallel column banks |
| `ARRAY_PARTITION variable=acc complete dim=0` | `acc[kTileN][kTileM]` | all accumulators in registers |
| `PIPELINE II=1` | a_buf load / b_tile load / K-reduction / C write | One iteration per clock |
| `UNROLL` | inner `m1` loop + the `acc` clear | `kTileM` parallel MAC lanes |

---

## 5. II=1 Strategy — Accumulator Lane Rotation

The K-reduction loop iterates `k_valid × kTileN` times. Each group of
`kTileN` consecutive iterations processes **one** K element across all
`kTileN` row lanes:

```
n1 = ki % kTileN     // which row lane  — rotates 0, 1, …, kTileN-1
kk = ki / kTileN     // K index local to this K-tile
```

Because the lane index rotates, the same `acc[n1][m1]` register is written
only once every `kTileN` cycles. That distance (`kTileN ≥ MAC latency ≈ 3`)
covers the multiply-accumulate pipeline depth, so the read-after-write hazard
that would otherwise force II ≥ 3 is broken and HLS schedules the loop at
II=1. Since `kTileN` is a power of two, `ki % kTileN` is a bitwise AND and
`ki / kTileN` a right shift — no dividers appear in RTL.

The inner `m1` loop is fully unrolled, so `kTileM` MAC units fire every
cycle (one per output column). **Inner-loop throughput is `kTileM`
MACs/cycle**, sustained at II=1.

---

## 6. Batch Broadcasting

`batch` independent 2-D products are computed by the outer `bi` loop, which
advances each pointer by its `*_batch_stride`:

- `a_batch_stride = 0` → `A` stays fixed and broadcasts across every batch
  step (e.g. a shared weight matrix against a batch of activations).
- `b_batch_stride = 0` → `B` broadcasts.
- `c_batch_stride` is normally `n·m` (each product writes a fresh output).

This matches ONNX `MatMul` broadcasting on the leading batch dimension
without copying the broadcast operand in DDR.

---

## 7. Data Types and Saturation

`saturate_cast<Data_t>(v)` (defined in `MatmulKernel.h`) converts an
`AccData_t` accumulator back to `Data_t`. For `ap_fixed` it routes through
`ap_fixed<W,I,AP_TRN,AP_SAT>` — truncation toward zero, then saturation
clamping — matching ONNX fixed-point semantics; it is applied in the C-write
loop. The primary template is an identity pass-through for `float` / `double`
builds. The `ap_fixed` specialisation is guarded by `MATMUL_HAVE_APFIXED` so
the matmul subdirectory stays self-contained (it does not depend on the
VectorOPKernel headers).

MAC operands are widened to `AccData_t` before the multiply
(`AccData_t(a_val) · AccData_t(b_tile[kk][m1])`).

---

## 8. Test Coverage (`TestMatmulSim.cpp`)

C-simulation tests compiled with GCC. Each case runs `MatmulKernel` against
`ref_matmul_2d()` — a naive triple-nested-loop oracle that uses the same
`AccData_t` accumulation and `saturate_cast<Data_t>` output, so fixed-point
results are bitwise-identical (exact comparison); `float` builds allow a
1-ULP tolerance for reordered tile sums.

| Category | Cases |
|----------|-------|
| Degenerate | `1×1×1`; unit-N (`1×K×M`); unit-M (`N×K×1`) |
| Exact tiles | `kTileN × kTileK × kTileM` — one full tile in every dimension |
| Partial last tile | partial N (`kTileN+2`); partial M (`kTileM+3`); partial K (`kTileK+5`, spans 2 K-tiles) |
| Multi-tile | all dims span 2 tiles; `kTileN·… × kTileK·2+7 × kTileM·2+1` |
| Arbitrary | `7×13×5` (all dims below the tile sizes) |
| Batch | `batch=3` without broadcast |

A second test, **`TestMatmulBlas.cpp`**, validates the `float` build of the
kernel against `cblas_sgemm` when a BLAS library is found at configure time.
`make gen_matmul_test_data` re-runs the reference in `--dump-data` mode to
emit hex fixtures for the HDL testbench.

---

## 9. Inference Scheduler Integration

**`MatmulNode` (`nodes.py`, `kernel_name = "MatmulKernel"`)** maps the ONNX
`MatMul` operator to `XMatmulkernel` invocations:

- Validates the 2-D / batched shapes and the `n`, `k`, `m` dimensions.
- Enforces `k ≤ kMaxK`.
- Supports batched matmul and stride-0 batch broadcasting; a row-strided
  decomposition handles alignment-gapped buffers.

`Gemm` is **not** a matmul node directly — `OnnxGraph._preprocess_model()`
decomposes `Gemm` into `MatMul` + optional `Add` at model-load time, so the
MatmulKernel only ever sees plain `MatMul`.

The code-generated `run_matmul()` writes the AXI-Lite registers and calls
`XMatmulkernel_Start()` non-blocking; `run_matmul_at()` (used inside the
4-D × 3-D outer loop, where iterations would otherwise race on the shared
Matmul registers) is the synchronous variant that polls internally. A
`kernel_wait(KERNEL_MATMUL)` drains the lane only when a dependent op needs
the result.

---

## 10. Build Targets

```bash
# C simulation (GCC, no Vitis)
make TestMatmulRef && ctest

# HLS synthesis + IP export for KV260
make synthesize_matmul_kv260
```

The synthesis target reads a `platforms/<name>.json` (part, optional board
and clock) and invokes Vitis HLS via `Synthesis.tcl.in`, which configures the
project, sets 64-bit AXI and the bus width, runs `csynth_design`, and exports
an IP-catalog archive.

---

## 11. Key Source Files

| File | Purpose |
|------|---------|
| `kernels/matmul/kernel/MatmulKernel.cpp` | HLS kernel — tiled loop nest |
| `kernels/matmul/include/MatmulKernel.h` | Kernel declaration, `saturate_cast<T>` |
| `kernels/matmul/include/Config.h.in` | CMake template → `Config.h` (`Data_t`, `AccData_t`, tile constants) |
| `kernels/matmul/test/TestMatmulSim.cpp` | C simulation tests (GCC) |
| `kernels/matmul/test/TestMatmulBlas.cpp` | `float` build validated against `cblas_sgemm` |
| `kernels/matmul/scripts/Synthesis.tcl.in` | Vitis HLS TCL template |
| `inference-scheduler/src/nodes.py` | `MatmulNode` class (ONNX → kernel params) |
| `inference-scheduler/src/codegen/_source.py` | `run_matmul()` / `run_matmul_at()` code generation |
| `inference-scheduler/src/graph.py` | `Gemm` → `MatMul` + `Add` decomposition |

---

## 12. Summary

| Aspect | Details |
|--------|---------|
| **Supported ONNX op** | `MatMul` (`Gemm` decomposed to `MatMul` + `Add` at load time) |
| **Operation** | `C = A × B`, batched, row-major |
| **Data type** | `ap_fixed<16,8>` (default) or `float` |
| **Accumulator type** | `ap_fixed<32,16>` (default) or `double` |
| **Tiling** | `kTileN=4` rows × `kTileM=16` columns × `kTileK=256` inner |
| **Inner-loop parallelism** | `kTileM=16` MACs/cycle (unrolled `m1` lanes) |
| **Initiation interval** | II=1 in every load / reduce / write loop |
| **II=1 mechanism** | Accumulator lane rotation `n1 = ki % kTileN` (RAW distance = `kTileN`) |
| **Architecture** | Single sequential tiled loop nest (not `DATAFLOW`) |
| **On-chip buffers** | `a_buf` (BRAM), `b_tile` (BRAM), `acc` (registers) |
| **A reuse** | `a_buf` loaded once per `n_tile`, reused across all `m_tile`/`k_tile` |
| **Batch broadcasting** | `a_batch_stride` / `b_batch_stride` = 0 reuses A / B |
| **Inner-dimension limit** | `k ≤ kMaxK` (2048, compile-time); `n` / `m` / `batch` unbounded |
| **AXI master ports** | 3 (gmem0 `a`, gmem1 `b`, gmem2 `c`) |
| **AXI-Lite registers** | 10 scalars/pointers + `return` |
| **Saturation** | `saturate_cast` with `AP_TRN` + `AP_SAT` at the C-write |
| **AXI-Lite base address** | `0xA001_0000` |
| **Driver prefix** | `xmatmulkernel` |
| **UIO device name** | `MatmulKernel_0` |
