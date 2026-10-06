# Platform Configuration

Everything platform-specific in this repository lives in a single JSON
file: `platforms/<platform>.json`. The file is the **single source of
truth** for the FPGA part and the compile-time bounds of the kernels
(constants of the RTL kernels, checked against it at configure time).

Two distinct consumers read the same JSON:

| Consumer | Source | How it reads |
|----------|--------|--------------|
| **C++ / RTL build** (IP packaging part, the RTL kernels' bound checks, C-sim Config.h) | `kernels/<k>/CMakeLists.txt::<k>_load_constants()` | `string(JSON … GET … kernels <k> <field>)`; missing field → `FATAL_ERROR` |
| **Python scheduler** (model validation before codegen) | `inference-scheduler/src/_<k>_hw_config.py::resolve()` | `json.load`; missing field → `<Kernel>HwConfigError` |

Keeping a single source means the Python validator and the C++ kernel
build cannot drift apart — there are no implicit defaults. If a field
is missing, **CMake configure fails immediately**, naming the JSON
file and the offending key.

`platforms/kv260.json` (the only platform shipped today) is the
reference. New platforms are added by dropping a similarly-shaped file
next to it and re-running CMake.

---

## Platform selection

The active platform is controlled by the top-level CMake cache
variable `AXI_PLATFORM` (default `kv260`):

```bash
cmake .. -DAXI_PLATFORM=<platform>
```

This picks `platforms/<platform>.json` as the source for the **default
C-sim build** — the `Config.h` consumed by `make TestConvRef`,
`make TestPoolingSim`, etc. is generated from this platform's bounds.

Every JSON file in `platforms/` gets a `synthesize_<platform>` aggregate
(the four kernel IPs) regardless of `AXI_PLATFORM`; the IPs themselves are
packaged for the `AXI_PLATFORM` platform's part, and the RTL kernels hold
its bounds as constants (checked at configure time).  Picking a different
platform with `-DAXI_PLATFORM=foo` affects which JSON drives the `Config.h`
file used by C-sim tests and the RTL kernels' part and bound check.

The Python side mirrors the same convention: setting the
`AXI_PLATFORM` environment variable selects the JSON the scheduler
resolvers read (`_<k>_hw_config.resolve()` falls back to
`os.environ.get("AXI_PLATFORM", "kv260")`).

```bash
AXI_PLATFORM=zcu102 .venv/bin/python inference_scheduler.py model.onnx
```

---

## File schema

### Top-level fields

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `description` | no | *(none)* | Informational only; not read by CMake or the scheduler |
| `part` | yes | — | Xilinx device part string passed to `set_part` |
| `board` | no | *(none)* | Board identifier passed to `set_part -board` |
| `clock` | no | *(none)* | Informational: the target clock (MHz) of the retired Vitis HLS synthesis.  The RTL IPs' out-of-context checks use their own period (`<K>_RTL_PERIOD`, 3.333 ns); the block design runs the kernels at 250 MHz from an MMCM (`hw/cormorant_hw_128/scripts/bd_kernel_clock.tcl`, [FMAX_250_PLAN](../plans/FMAX_250_PLAN.md)) |
| `kernels.conv` | yes | — | ConvKernel compile-time bounds — [§ConvKernel](#kernelsconv) |
| `kernels.matmul` | yes | — | MatmulKernel compile-time bounds — [§MatmulKernel](#kernelsmatmul) |
| `kernels.pool` | yes | — | PoolingKernel compile-time bounds — [§PoolingKernel](#kernelspool) |

VectorOPKernel has no per-platform constants: it is a runtime-sized,
element-wise kernel and has no compile-time bounds to validate (its
SystemVerilog IP reads only `part`, for packaging and out-of-context
synthesis).

> **`AXI_BUS_WIDTH` is not a JSON field.** It is a top-level CMake
> cache variable (default `32`) that set the HLS synthesis's
> `config_interface -m_axi_max_widen_bitwidth`.  With every kernel in
> SystemVerilog (fixed 128-bit data ports, the block design's width) it
> affects no IP. See the configure-time cache variables table in
> [BUILD_TARGETS.md](BUILD_TARGETS.md#setup).

---

### `kernels.conv`

Describes the ConvKernel the bitstream carries — since CONV_RTL_PLAN
phase 3 the SystemVerilog kernel
([`CONV_RTL_KERNEL.md`](../kernels/CONV_RTL_KERNEL.md)), whose
`rtl/cv_pkg.sv` holds every bound below as a constant (configure stops with
a `FATAL_ERROR` when they differ: changing a bound means changing the RTL).
The same fields size the HLS kernel's C++ model (`kernels/conv`, the
reference and fixture generator): its line buffer, weight cache, bias
buffer and persistent accumulator
([`CONV_KERNEL.md`](../kernels/CONV_KERNEL.md) §3).

| Field | Constraint | Description |
|-------|------------|-------------|
| `impl` | `"rtl"` (the CMake build refuses anything else); the scheduler also takes `"hls"` | Which ConvKernel the platform's bitstream carries: the SystemVerilog one (`kernels/conv_rtl`), or the Vitis HLS kernel of the bitstreams built before phase 3 (`dbb320fb7297` and older).  Same calls, same results, different speed: it selects the scheduler's ConvKernel cycle model (`cost_model.rtl_conv_walk` or the HLS walk) behind the MatMul-on-ConvKernel engine and geometry choices.  The `AXI_CONV_IMPL` environment variable overrides it for the Python tools (`AXI_CONV_IMPL=hls` for a project on an HLS bitstream) |
| `tile_m` | power of 2, multiple of 8, `≤ tile_ic`; any `out_ch` works (residual-padded) | Output-channel tile = M dimension of the `tile_ic × tile_m` MAC grid |
| `tile_ic` | power of 2; any `in_ch` works (residual-padded) | Input-channel tile = IC dimension of the MAC grid; also the lane count of the packed weight layout |
| `max_kh` | `kh ≤ max_kh` | Hard upper bound on kernel height |
| `max_kw` | `kw ≤ max_kw` | Hard upper bound on kernel width |
| `max_in_ch` | `in_ch ≤ max_in_ch` | Hard upper bound on input channels |
| `max_out_ch` | `out_ch ≤ max_out_ch` | Hard upper bound on output channels; sizes the bias buffer |
| `max_line_buf_cols` | power of 2; `(kw-1)·dilation_w + 1 ≤ max_line_buf_cols`; caps `ow_per_tile`, **not** `in_w` | Line-buffer column capacity |
| `max_line_buf_rows` | power of 2; `(kh-1)·dilation_h + 1 ≤ max_line_buf_rows` | Line-buffer row capacity |
| `max_acc_persist_entries` | `out_w · ceil(out_ch / tile_m) · tile_m ≤ max_acc_persist_entries` | URAM persistent-accumulator capacity across in-channel tiles (one `tile_m`-padded output row must fit; stored as 512-bit URAM words — 8 URAM blocks at 65536 — and the `acc_stream` FIFO scales with it) |
| `max_m_per_group` | — | Number of M-tiles cached together in the weight slab |

Models cannot violate `tile_m` / `tile_ic` (any `out_ch` / `in_ch` is
residual-padded), but the scheduler still reads them: `tile_ic` sets the
packed weight layout it emits (CONV_OPTIMISATION.md §2.32), `tile_m` the
accumulator-row padding check, and both — with `max_kw` and
`max_m_per_group` — feed the MatMul-on-ConvKernel lowering and the
engine cost model (`matmul_lowering.py`, `cost_model.py`). Model
acceptance is gated by `ConvNode.from_onnx_node()`, which raises
`SchedulerError` naming the violated bound for `max_in_ch`,
`max_out_ch`, `max_kh`, `max_kw`, `max_line_buf_rows`,
`max_line_buf_cols` and `max_acc_persist_entries`.

---

### `kernels.matmul`

Describes the MatmulKernel the bitstream carries — since MATMUL_RTL_PLAN
phase 4 the SystemVerilog kernel
([`MATMUL_RTL_KERNEL.md`](../kernels/MATMUL_RTL_KERNEL.md)), which shares
`tile_m` (the packed-B DDR tile), `max_k` (its `K_MAX`) and `gemv_max_m > 0`
(the image path) with the retired HLS kernel.  `tile_n`, `tile_k` and the
value of `gemv_max_m` size the HLS kernel's C++ model (`kernels/matmul`, the
reference and fixture generator): its row-staging buffer
`a_buf[tile_n][max_k]` and B block buffer `b_tile[tile_m][2·tile_k]`
([`MATMUL_KERNEL.md`](../kernels/MATMUL_KERNEL.md) §2–§3).

| Field | Constraint | Description |
|-------|------------|-------------|
| `impl` | `"rtl"` (the CMake build refuses anything else); the scheduler also takes `"hls"` | Which MatmulKernel the platform's bitstream carries: the SystemVerilog one (`kernels/matmul_rtl`, [MATMUL_RTL_KERNEL](../kernels/MATMUL_RTL_KERNEL.md)), or the Vitis HLS kernel of the bitstreams built before phase 4 (`caa67f49a5a3` and older).  Same calls, same results, different speed: it selects the scheduler's MatmulKernel cost model and engine-choice rules.  The `AXI_MATMUL_IMPL` environment variable overrides it for the Python tools (`AXI_MATMUL_IMPL=hls` for a project on an HLS bitstream) |
| `tile_n` | power of 2; any `N` works (residual-padded) | Row tile / unroll factor (HLS kernel; the RTL kernel works on panels of 8 rows) |
| `tile_m` | power of 2, multiple of 8; any `M` works (residual-padded) | Column tile / unroll factor; also the width of the packed-B DDR layout |
| `tile_k` | power of 2; any `K` works (residual-padded) | K-loop tile (B block rows; HLS kernel) |
| `max_k` | `k ≤ max_k`; multiple of 8 | Hard upper bound on the inner dimension; sizes the row staging buffer (the RTL kernel's `K_MAX` in `rtl/mm_pkg.sv` must equal it — fact `platform.kv260_bounds`). `MatMul` nodes with `k > max_k` are rejected at scheduling |
| `gemv_max_m` | multiple of 8; 0 = no GEMV path | GEMV streaming mode (`gemv_kw`, MATMUL_KERNEL.md §1): output columns one pass over B accumulates on chip per read stream (one URAM word per 8 columns); a wider `m` is split into column chunks inside the kernel, so it bounds nothing.  The RTL kernel reads the image at any `m` (512-column chunks); for it only `> 0` matters |

The Python side reads `impl` (the engine cost model), `max_k` (validation),
`tile_m` (the packed-B layout the scheduler emits for constant B operands),
`tile_n` (the HLS kernel's engine cost model and the `--plan` performance
model's HLS features only) and `gemv_max_m` (whether the GEMV path exists,
and the HLS cost model's column chunks). `tile_k` is a pure C++ tiling factor.
Setting `gemv_max_m` to 0 also changes the generated `run_matmul()`: it
then does not write the `gemv_kw` / `a_to_b` registers, so projects
build against drivers without them.

---

### `kernels.pool`

Sizes the PoolingKernel's line buffer and unrolled per-position
adders at compile time. See [`POOLING_KERNEL.md`](../kernels/POOLING_KERNEL.md) §3
and [`POOL_OPTIMISATION.md`](../kernels/POOL_OPTIMISATION.md) §4 for the full
architectural context including bank/port topology.

| Field | Constraint | Description |
|-------|------------|-------------|
| `tile_c` | power of 2; any `C` works (channel-tiled) | Channel tile width / II=1 lane rotation depth |
| `max_kh` | `pool_h ≤ max_kh` | Compile-time pool window height limit |
| `max_kw` | `pool_w ≤ max_kw` | Compile-time pool window width limit |
| `max_line_buf_rows` | power of 2; `(pool_h-1)·dil_h + 1 ≤ max_line_buf_rows` | Line-buffer row capacity |
| `max_line_buf_cols` | multiple of 8 (whole 128-bit words); `(pool_w-1)·dil_w + 1 ≤ max_line_buf_cols`; W-tiling kicks in for `in_w > this` | Line-buffer column capacity |
| `ow_parallel` | power of 2, `≤ 8` (lanes per word); any `out_w` works (residual-padded) | Output-position unroll factor |

`tile_c` and `ow_parallel` are not exported to the Python validator —
both have unconditional runtime fallbacks (channel tiling for any C;
residual-lane padding for any `out_w`).

The bitstream's PoolingKernel is the SystemVerilog one (`kernels/pool_rtl`,
[`POOL_RTL_KERNEL.md`](../kernels/POOL_RTL_KERNEL.md)); it holds these six
fields as constants in `rtl/pl_pkg.sv`, and configure compares them with
the `AXI_PLATFORM` JSON and stops with a `FATAL_ERROR` when they differ —
changing a `kernels.pool` field means changing the RTL.  The fields also size
the retired HLS kernel's C++ (`kernels/pool`, the reference model and
fixture generator).

---

## Example — kv260.json

```jsonc
{
  "description": "Xilinx KV260 Starter Kit",
  "part":  "xck26-sfvc784-2LV-c",
  "board": "xilinx.com:kv260_som:part0:1.4",
  "clock": 150,
  "kernels": {
    "conv": {
      "impl":   "rtl",
      "tile_m":                  16, "tile_ic":               16,
      "max_kh":                  7,  "max_kw":                 7,
      "max_in_ch":            1024,  "max_out_ch":          1280,
      "max_line_buf_cols":      64,  "max_line_buf_rows":     16,
      "max_acc_persist_entries": 65536,
      "max_m_per_group":         4
    },
    "matmul": {
      "impl":   "rtl",
      "tile_n":   4, "tile_m":  32, "tile_k": 256, "max_k": 4096,
      "gemv_max_m": 4096
    },
    "pool": {
      "tile_c":            8,
      "max_kh":            7,  "max_kw":            7,
      "max_line_buf_rows": 16, "max_line_buf_cols": 64,
      "ow_parallel":       2
    }
  }
}
```

---

## Adding a new platform

1. Copy `platforms/kv260.json` to `platforms/<platform>.json` and
   edit the FPGA-specific fields:

    ```jsonc
    {
      "description": "Xilinx ZCU102 dev board",
      "part":  "xczu9eg-ffvb1156-2-e",
      "board": "xilinx.com:zcu102:part0:3.4",
      "clock": 250,
      "kernels": { /* tune to taste; see per-kernel tables above */ }
    }
    ```

2. Re-run CMake. `CMAKE_CONFIGURE_DEPENDS` is set on every platform
   JSON, so subsequent `make` invocations auto-rerun configure when
   the JSON changes:

    ```bash
    cd build
    cmake ..                            # picks up the new file
    ```

3. The new platform now has:
    - `synthesize_<platform>` — roll-up target that packages all four kernel IPs
    - `dtbo_<platform>_<stem>` — for any `<stem>.dts` file under `dts/<platform>/`

    The Vivado / behavioural-test targets (`build_hw_kv260`,
    `sim_hw_kv260`, `behavior_test_<kernel>`) stay KV260-only.

4. To make the platform the **default** for C-sim tests and the
   inference scheduler, pass `AXI_PLATFORM` to CMake (and export the
   same value when running Python tools):

    ```bash
    cmake .. -DAXI_PLATFORM=<platform>
    export AXI_PLATFORM=<platform>
    ```

---

## Editing an existing platform's bounds

Any change to a `max_*` field affects both the synthesised kernel and
the models the Python scheduler accepts. Two things need re-running:

```bash
# 1. Re-run CMake to pick up the new bounds in Config.h.
cd build && cmake ..

# 2. Re-generate the hardware-bound test fixtures so the boundary
#    geometries match the new bounds (otherwise "must raise" / "at
#    limit" tests can either start passing on the wrong values or
#    start hitting unrelated limits).
cd ../inference-scheduler
AXI_PLATFORM=<platform> .venv/bin/python test/gen_conv_models.py
AXI_PLATFORM=<platform> .venv/bin/python test/gen_matmul_models.py
AXI_PLATFORM=<platform> .venv/bin/python test/gen_pool_models.py

# 3. Confirm the validator/test fixtures still agree.
AXI_PLATFORM=<platform> .venv/bin/python -m pytest test/ -q
```

The `TestConvHwConfigResolver` (`test/test_conv.py`),
`TestMatmulHwConfigResolver` (`test/test_matmul.py`) and
`TestPoolHwConfigResolver` (`test/test_pool.py`) classes cross-check the
resolved Python constants against the JSON, so a typo or shape error
surfaces immediately at `pytest` time.

`tile_*` / `ow_parallel` / `tile_c` changes don't require fixture
regeneration — those fields are not validated against models — but the
`kernels.conv` and `kernels.pool` fields, and `kernels.matmul.max_k`, are
constants of the RTL kernels: change the RTL with them, then rebuild the
bitstream. `kernels.conv.tile_ic` and
`kernels.matmul.tile_m` also change the packed weight layout the
scheduler emits, so every generated project must be regenerated against
the same JSON as the bitstream.

---

## Related references

| Document | Coverage |
|----------|----------|
| [`CONV_RTL_KERNEL.md`](../kernels/CONV_RTL_KERNEL.md) | The SystemVerilog ConvKernel: contract (the bounds, `rtl/cv_pkg.sv`), architecture |
| [`CONV_KERNEL.md`](../kernels/CONV_KERNEL.md) §3 | The retired HLS kernel's architecture and tiling (its C++ is the reference model), including how each `kernels.conv.*` field maps to its resources |
| [`MATMUL_KERNEL.md`](../kernels/MATMUL_KERNEL.md) §2–§3 | MatmulKernel tiling, `max_k` rationale |
| [`POOL_RTL_KERNEL.md`](../kernels/POOL_RTL_KERNEL.md) | The SystemVerilog PoolingKernel: contract (the bounds, `rtl/pl_pkg.sv`), architecture |
| [`POOLING_KERNEL.md`](../kernels/POOLING_KERNEL.md) §3 | PoolingKernel compile-time configuration (the retired HLS kernel's C++, the reference model) |
| [`POOL_OPTIMISATION.md`](../kernels/POOL_OPTIMISATION.md) §4 | PoolingKernel field-by-field reference, bank topology, `ow_parallel` interaction with stride |
| [`VECTOROP_RTL_KERNEL.md`](../kernels/VECTOROP_RTL_KERNEL.md) | VectorOPKernel architecture (no compile-time bounds; the retired HLS kernel: [`VECTOROP_KERNEL.md`](../kernels/VECTOROP_KERNEL.md)) |
| [`inference-scheduler/CLAUDE.md`](../../inference-scheduler/CLAUDE.md) | Python resolver pattern (`_<k>_hw_config.resolve()`) and validator flow |
