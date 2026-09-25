# Platform Configuration

Everything platform-specific in this repository lives in a single JSON
file: `platforms/<platform>.json`. The file is the **single source of
truth** for the FPGA part, the synthesis target clock, and the
compile-time bounds each kernel synthesises against.

Two distinct consumers read the same JSON:

| Consumer | Source | How it reads |
|----------|--------|--------------|
| **C++ build** (Vitis HLS synthesis, IP export, C-sim Config.h) | `kernels/<k>/CMakeLists.txt::<k>_load_constants()` | `string(JSON … GET … kernels <k> <field>)`; missing field → `FATAL_ERROR` |
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

The per-platform synthesis loop is independent. Every JSON file in
`platforms/` gets its own `synthesize_<kernel>_<platform>` target
regardless of `AXI_PLATFORM`. Picking a different platform with
`-DAXI_PLATFORM=foo` only affects which JSON drives the `Config.h`
file used by C-sim tests.

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
| `description` | no | *(none)* | Informational only; surfaces in CMake status logs |
| `part` | yes | — | Xilinx device part string passed to `set_part` |
| `board` | no | *(none)* | Board identifier passed to `set_part -board` |
| `clock` | no | `300` | Target clock in MHz (HLS `create_clock -period`) |
| `kernels.conv` | yes | — | ConvKernel compile-time bounds — [§ConvKernel](#kernelsconv) |
| `kernels.matmul` | yes | — | MatmulKernel compile-time bounds — [§MatmulKernel](#kernelsmatmul) |
| `kernels.pool` | yes | — | PoolingKernel compile-time bounds — [§PoolingKernel](#kernelspool) |

VectorOPKernel has no per-platform constants: it is a runtime-sized,
element-wise kernel and has no compile-time bounds to validate.

> **`AXI_BUS_WIDTH` is not a JSON field.** It is a top-level CMake
> cache variable (`cmake -DAXI_BUS_WIDTH=128`) so that a single
> platform JSON can be synthesised against multiple bus widths
> independently. See the *Key CMake parameters* table in the top-level
> [README.md](../README.md#key-cmake-parameters).

---

### `kernels.conv`

Sizes the ConvKernel's line buffer, weight cache, bias buffer, and
persistent-accumulator at compile time. See
[`CONV_KERNEL.md`](CONV_KERNEL.md) §3 for the full architectural
context.

| Field | Constraint | Description |
|-------|------------|-------------|
| `tile_m` | power of 2; any `out_ch` works (residual-padded) | Output-channel tile / unroll factor |
| `tile_ic` | power of 2; any `in_ch` works (residual-padded) | Input-channel tile / unroll factor |
| `max_kh` | `kh ≤ max_kh` | Hard upper bound on kernel height |
| `max_kw` | `kw ≤ max_kw` | Hard upper bound on kernel width |
| `max_in_ch` | `in_ch ≤ max_in_ch` | Hard upper bound on input channels |
| `max_out_ch` | `out_ch ≤ max_out_ch` | Hard upper bound on output channels; sizes the bias buffer |
| `max_line_buf_cols` | power of 2; caps `ow_per_tile`, **not** `in_w` | Line-buffer column capacity |
| `max_line_buf_rows` | power of 2; `(kh-1)·dilation_h + 1 ≤ max_line_buf_rows` | Line-buffer row capacity |
| `max_acc_persist_entries` | `out_w · out_ch ≤ max_acc_persist_entries` | URAM persistent-accumulator capacity across in-channel tiles |
| `max_m_per_group` | — | Number of M-tiles cached together in the weight slab |

`tile_m`, `tile_ic` are read by the C++ build but **not** exported to
the Python validator — any `out_ch`/`in_ch` is residual-padded inside
the kernel, so models cannot violate them. The other fields gate model
acceptance: `ConvNode.from_onnx_node()` raises `SchedulerError`
naming the violated bound.

---

### `kernels.matmul`

Sizes the MatmulKernel's row-staging buffer `a_buf[tile_n][max_k]` at
compile time. See [`MATMUL_KERNEL.md`](MATMUL_KERNEL.md) §3.

| Field | Constraint | Description |
|-------|------------|-------------|
| `tile_n` | power of 2; any `N` works (residual-padded) | Row tile / unroll factor |
| `tile_m` | power of 2; any `M` works (residual-padded) | Column tile / unroll factor |
| `tile_k` | any `K` works (residual-padded) | K-loop tile |
| `max_k` | `k ≤ max_k` | Hard upper bound on the inner dimension; sizes the row staging buffer. `MatMul` nodes with `k > max_k` are rejected at scheduling |

Only `max_k` is exported to the Python validator. `tile_n`/`tile_m`/
`tile_k` are pure unrolling factors with runtime residual-tile
fallbacks.

---

### `kernels.pool`

Sizes the PoolingKernel's line buffer and unrolled per-position
adders at compile time. See [`POOL_OPTIMIZATION.md`](POOL_OPTIMIZATION.md)
§4 for the full architectural context including bank/port topology.

| Field | Constraint | Description |
|-------|------------|-------------|
| `tile_c` | power of 2; any `C` works (channel-tiled) | Channel tile width / II=1 lane rotation depth |
| `max_kh` | `pool_h ≤ max_kh` | Compile-time pool window height limit |
| `max_kw` | `pool_w ≤ max_kw` | Compile-time pool window width limit |
| `max_line_buf_rows` | power of 2; `(pool_h-1)·dil_h + 1 ≤ max_line_buf_rows` | Line-buffer row capacity |
| `max_line_buf_cols` | `(pool_w-1)·dil_w + 1 ≤ max_line_buf_cols`; W-tiling kicks in for `in_w > this` | Line-buffer column capacity |
| `ow_parallel` | power of 2; any `out_w` works (residual-padded) | Output-position unroll factor |

`tile_c` and `ow_parallel` are not exported to the Python validator —
both have unconditional runtime fallbacks (channel tiling for any C;
residual-lane padding for any `out_w`).

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
      "tile_m":                  16, "tile_ic":               16,
      "max_kh":                  7,  "max_kw":                 7,
      "max_in_ch":            1024,  "max_out_ch":          1280,
      "max_line_buf_cols":      64,  "max_line_buf_rows":     16,
      "max_acc_persist_entries": 65536,
      "max_m_per_group":         4
    },
    "matmul": {
      "tile_n":   4, "tile_m":  32, "tile_k": 256, "max_k": 2048
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
    - `synthesize_<kernel>_<platform>` — per-kernel HLS synthesis + IP export
    - `synthesize_<platform>` — roll-up target that builds all four kernels
    - `dtbo_<platform>_<design>` — for any `.dts` file under `dts/<platform>/`

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

The `test_<kernel>_hw_config.py` modules cross-check the resolved
Python constants against the JSON, so a typo or shape error surfaces
immediately at `pytest` time.

`tile_*` / `ow_parallel` / `tile_c` changes don't require fixture
regeneration — those fields are not validated against models — but
they do affect HLS resource usage / II, so a re-synthesis is still
needed before deploying.

---

## Related references

| Document | Coverage |
|----------|----------|
| [`CONV_KERNEL.md`](CONV_KERNEL.md) §3 | Full ConvKernel architecture and tiling, including how each `kernels.conv.*` field maps to hardware resources |
| [`MATMUL_KERNEL.md`](MATMUL_KERNEL.md) §3 | MatmulKernel tiling, `max_k` rationale |
| [`POOL_OPTIMIZATION.md`](POOL_OPTIMIZATION.md) §4 | PoolingKernel field-by-field reference, bank topology, `ow_parallel` interaction with stride |
| [`VECTOROP_KERNEL.md`](VECTOROP_KERNEL.md) | VectorOPKernel architecture (no compile-time bounds) |
| [`../inference-scheduler/CLAUDE.md`](../inference-scheduler/CLAUDE.md) | Python resolver pattern (`_<k>_hw_config.resolve()`) and validator flow |
