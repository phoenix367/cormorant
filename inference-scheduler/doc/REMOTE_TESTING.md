# Remote Hardware Testing

`run_remote_tests.py` automates end-to-end hardware validation of ONNX models on
a physical KV260 board. For each model it:

1. Generates a C inference project locally (`inference_scheduler.py`)
2. Uploads the project to the board via SSH/SFTP
3. Builds on the board (`cmake -DINFERENCE_TARGET=LINUX` + `make`)
4. Runs the generated `test_inference` binary as root
5. Collects pass/fail results and prints a summary

The test binary fills inputs with a deterministic ramp pattern, runs the FPGA
kernel, and compares every output element against gold-standard values
pre-computed by the Python fixed-point simulator. A mismatch in any element
fails the test.

---

## Prerequisites

### Local machine

- Python virtual environment with `paramiko`:
  ```bash
  cd inference-scheduler
  .venv/bin/pip install paramiko
  ```
- Driver sources (for `local.driver_dirs` in the config): the Vitis HLS exports
  of `make synthesize_kv260` and the RTL MatmulKernel's driver (`make driver_matmul_rtl`):
  ```
  <repo>/build/kernels/vectorop/kv260/vadd_kv260/solution1/impl/ip/drivers/VectorOPKernel_v1_0/src/
  <repo>/build/kernels/matmul_rtl/driver/MatmulKernel_v1_0/src/
  <repo>/build/kernels/conv/kv260/conv_kv260/hls/impl/ip/drivers/ConvKernel_v1_0/src/
  <repo>/build/kernels/pool/kv260/pool_kv260/hls/impl/ip/drivers/PoolingKernel_v1_0/src/
  ```

### Remote board (KV260)

| Requirement | How to verify |
|-------------|---------------|
| `gcc` ≥ 11, `cmake` ≥ 3.19, `make` | `gcc --version`, `cmake --version` |
| XRT runtime (`xrt.h` + `libxrt_core.so`) | `pkg-config --exists xrt && echo ok` |
| Kernel overlay(s) loaded | `cat /sys/class/uio/uio*/name` — must print the names in `remote.uio_devices` |
| Passwordless sudo, or SSH as root | `sudo -n true && echo ok` |

The `--check-only` flag verifies all of these remotely before running any tests
(see [Preflight Check](#preflight-check)).

---

## Setup

### 1. SSH key authentication (recommended)

```bash
# Generate a dedicated key if you don't have one
ssh-keygen -t ed25519 -f ~/.ssh/kv260-testkey -N ""

# Copy it to the board
ssh-copy-id -i ~/.ssh/kv260-testkey root@192.168.100.8
```

Password authentication is also supported; set `ssh.password` in the config and
leave `ssh.key_file` null.

### 2. Config files

Copy the appropriate example(s) and set your board's IP (the `*.example`
files are the tracked templates; the copies are gitignored):

```bash
# Bitstream upload
cp bitstream_config_kv260.json.example bitstream_config_kv260.json
$EDITOR bitstream_config_kv260.json       # set ssh.host and check bitstream paths

# Correctness tests
cp remote_config.json.example remote_config.json
$EDITOR remote_config.json

# Performance benchmarks
cp perf_config.json.example perf_config.json
$EDITOR perf_config.json
```

Minimum required change in all configs: set `ssh.host` to your board's IP or
hostname.  The remote / perf examples' `remote.uio_devices` are the
`fabric_*` names of the Cormorant overlay and their `local.driver_dirs`
(`../build/kernels/…`, relative to `inference-scheduler/`) the driver
directories of a build in `<repo>/build`.

---

## Config File Reference

| Key | Default | Description |
|-----|---------|-------------|
| `ssh.host` | *(required)* | Board IP address or hostname |
| `ssh.user` | `"root"` | SSH login user |
| `ssh.port` | `22` | SSH port |
| `ssh.key_file` | `null` | Path to SSH private key; `null` to use password |
| `ssh.password` | `null` | SSH password; `null` when using key auth |
| `ssh.connect_timeout` | `15` | TCP connect timeout in seconds |
| `remote.work_dir` | `"/tmp/inference_hw_tests"` | Base directory on the board; created automatically, cleaned up after the run |
| `remote.uio_devices` | `{}` | Per-kernel UIO sysfs names — `{"KernelName": "sysfs_name"}`. The string in `/sys/class/uio/uio*/name`, **not** a `/dev/uioN` path. Each entry becomes `-DINFERENCE_<KERNELNAME>_INSTANCE`, so for `run_remote_tests.py` the keys must be the `KERNEL_REGISTRY` names `VectorOPKernel`, `MatmulKernel`, `ConvKernel`, **`PoolKernel`** (a `PoolingKernel` key is ignored); `run_remote_perf.py` uses `PoolingKernel`. Empty dict uses the defaults in the generated `inference.h` (`VectorOPKernel_0`, …). |
| `remote.uio_device` | `null` | **Deprecated.** Old single-string UIO name; treated as `{"VectorOPKernel": value}`. Use `uio_devices` instead. |
| `remote.driver_dir` | `null` | Path **on the board** to copy all kernel driver sources from |
| `remote.driver_dirs` | `{}` | Per-kernel paths **on the board**: `{"VectorOPKernel": "/path", ...}` (registry names, as for `uio_devices`) |
| `remote.cmake_args` | `[]` | Extra `-D` flags appended to the `cmake` invocation |
| `local.driver_dir` | `null` | Single local directory with all driver files to bundle into each project before upload |
| `local.driver_dirs` | `{}` | Per-kernel local paths; files are merged before upload: `{"VectorOPKernel": "/path1", "MatmulKernel": "/path2"}` |
| `build.jobs` | `4` | Parallel `make -j` jobs on the board |
| `build.timeout` | `180` | Combined cmake + make timeout in seconds |
| `run.timeout` | `120` | Per-model `test_inference` execution timeout in seconds |
| `run.use_sudo` | `true` | Prefix the test binary with `sudo -n`; requires passwordless sudo |
| `cleanup` | `true` | After all tests finish, remove the run's directories under `remote.work_dir`, then `remote.work_dir` itself when it is empty |
| `board_lock` | `null` | The board lock file (see [Board lock](#board-lock)); `null` = `/tmp/kv260-board-<ssh.host>.lock`, `false` disables it |
| `models` | `[]` | List of model paths relative to the `inference-scheduler` directory |

### Board lock

Two jobs on one board at the same time corrupt each other, so every board
tool — `run_remote_tests.py`, `run_remote_perf.py`, `perf_calibrate.py run`,
`upload_bitstream.py`, the demos' `deploy_and_run.py`, the chat
`deploy.py` / `llm_board.py` — holds an exclusive `flock` on one file per
board for its whole session: `/tmp/kv260-board-<ssh.host>.lock` (the
config's `board_lock` overrides it; `false` disables it;
`src/remote/lock.py`).  A second job prints `waiting for board lock …`
and starts when the first exits.  The lock is re-entrant for tools a
holder starts (`perf_calibrate.py run --stop-server` → `deploy.py`): the
holder marks its environment (`KV260_BOARD_LOCK_HELD`).  Do not wrap these
tools in a shell `flock` on the same file — that lock is not marked and the
tool inside would wait forever; run other commands under the lock with

```bash
.venv/bin/python -m src.remote.locked --host <board> -- CMD ARG ...   # or --config CFG.json
```

### UIO device names

Each hardware kernel has a UIO sysfs name determined by the **node label** in the
device tree overlay (DTS).  The canonical DTS file for the KV260 is:

```
dts/kv260/cormorant.dts
```

The four UIO nodes and their names as they appear in `/sys/class/uio/uio*/name`:

| Kernel | DTS node label | UIO sysfs name |
|--------|----------------|----------------|
| `VectorOPKernel` | `fabric_vecop`  | `fabric_vecop`  |
| `MatmulKernel`   | `fabric_matmul` | `fabric_matmul` |
| `ConvKernel`     | `fabric_conv`   | `fabric_conv`   |
| `PoolingKernel`  | `fabric_pool`   | `fabric_pool`   |

To look up or confirm names, grep the DTS for `generic-uio`:

```bash
grep -B3 'generic-uio' dts/kv260/cormorant.dts
# fabric_vecop: fabric_vecop@a0000000 {
# fabric_matmul: fabric_matmul@a0010000 {
# fabric_conv: fabric_conv@a0020000 {
# fabric_pool: fabric_pool@a0030000 {
```

Verify on the board after loading the DTBO (the Kria Ubuntu image's four
`axi-pmon` performance monitors come first, uio0–3):

```bash
cat /sys/class/uio/uio*/name
# axi-pmon
# axi-pmon
# axi-pmon
# axi-pmon
# fabric_vecop
# fabric_matmul
# fabric_conv
# fabric_pool
```

These names are passed to cmake as `-DINFERENCE_VECTOROPKERNEL_INSTANCE`,
`-DINFERENCE_MATMULKERNEL_INSTANCE`, etc., which override the matching
`#ifndef` defaults in `inference.h`.

> **Note**: an older or third-party bitstream may use different node labels
> (e.g. `VectorOPKernel_0`).  Always derive `remote.uio_devices` from the
> DTS/DTSI that was used to build the overlay, not from kernel driver defaults.

### Driver source resolution

The runner places kernel driver sources (`.c`/`.h` files) in the project's
`driver/` directory before uploading. Four options, evaluated in priority order:

1. **`local.driver_dir`** — single local directory with all driver files. Use for
   single-kernel models or when you have pre-merged the files.
2. **`local.driver_dirs`** — per-kernel local paths; the runner merges them into
   one directory and passes it to `inference_scheduler.py --driver-dir`. Best for
   multi-kernel models where each kernel's HLS output lives separately.
3. **`remote.driver_dirs`** — per-kernel paths **on the board**; the runner copies
   each kernel's files via SSH after upload.
4. **`remote.driver_dir`** — single board-side directory for all driver files.

When `local.*` is set, the drivers are bundled before upload and `remote.*`
driver options are skipped. When none is set, `driver/` contains only a
`README.md` and the build will fail until you populate it manually.

---

## Bitstream Upload (`upload_bitstream.py`)

Before running any hardware tests the Cormorant bitstream must be loaded onto
the KV260.  `upload_bitstream.py` automates the full loading sequence, mirroring
what PYNQ's `Overlay` class does internally:

1. Parse the `.bit` header, byte-swap the payload → raw `.bin`
2. Parse the `.hwh` — extract PS AXI port-width parameters and memory topology
3. Build a minimal xclbin (MEM_TOPOLOGY) locally with `xclbinutil`
4. Upload `.bin` → `/lib/firmware/<name>.bin` on the board
5. Remove any existing configfs DTBO overlay with the same name
6. Write to `fpga_manager` sysfs — triggers PL reconfiguration
7. Verify `fpga_manager` state == `operating`
8. Load the xclbin into the zocl DRM driver (`xclLoadXclBin`) — registers
   memory topology so `xclAllocBO` resolves to a named DDR bank
9. Upload the `.dtbo`; drop a stale `pynq` overlay and unbind any foreign
   UIO device at one of the kernel addresses; apply the overlay via configfs
10. Verify overlay status == `applied` **and** no overlay error in `dmesg`
    since step 9 (configfs reads `applied` even when the kernel rejected the
    overlay)
11. Write PS SLCR / AXIFM registers to match the bitstream's AXI bus widths
    (last, because the overlay's `afi0` node resets the AXIFM width fields)
12. List `/dev/uio*` devices to confirm UIO nodes are up

Step 5 removes only an overlay of the same name (and step 9 `pynq`).  If the
same design is already loaded under another overlay name (e.g. `pl`), the
kernel refuses the new one (`create_overlay: Failed to create overlay
(err=-22)` in `dmesg`: the other overlay owns the nodes) and step 10 fails
with `Overlay '<name>' did not apply …`, listing the kernel errors and the
other loaded overlays.  Remove the other one on the board and run again,
or reuse its name:

```bash
ls /sys/kernel/config/device-tree/overlays/          # on the board
sudo rmdir /sys/kernel/config/device-tree/overlays/<name>
# or: upload_bitstream.py --config bitstream_config_kv260.json --overlay-name <name>
```

### Prerequisites

- `xclbinutil` on `PATH` (source the Vitis `settings64.sh`):
  ```bash
  source <Xilinx>/2025.2/Vitis/settings64.sh   # e.g. /opt/Xilinx/2025.2/Vitis/settings64.sh
  which xclbinutil   # should print the path
  ```
- SSH access to the board as root (or passwordless sudo) — same requirement
  as `run_remote_tests.py`.

### Config file

`bitstream_config_kv260.json.example` is the template for the Cormorant
design.  Copy it to `bitstream_config_kv260.json` and set your board address
and paths once:

```json
{
  "ssh": {
    "host": "192.168.100.8",
    "key_file": "~/.ssh/kv260-testkey"
  },
  "bitstream": {
    "bit":  "../hw/cormorant_hw_128/cormorant_hw_128.runs/impl_1/design_cormorant_wrapper.bit",
    "hwh":  "../hw/cormorant_hw_128/cormorant_hw_128.gen/sources_1/bd/design_cormorant/hw_handoff/design_cormorant.hwh",
    "dtbo": "../build/dts/kv260/design_cormorant.dtbo"
  }
}
```

The `.bit` comes from `make build_hw_kv260`, the `.dtbo` from
`make dtbo_kv260_cormorant` (compiles `dts/kv260/cormorant.dts`; needs `dtc`).

All paths under `"bitstream"` are resolved relative to the config file, so the
config is portable across checkouts.  The `hwh` key may be omitted only when
`<bit_stem>.hwh` sits alongside the `.bit` file; `make build_hw_kv260` writes
none there (the hwh is
`cormorant_hw_128.gen/sources_1/bd/design_cormorant/hw_handoff/design_cormorant.hwh`),
so keep the key.

#### `bitstream` config keys

| Key | Default | Description |
|-----|---------|-------------|
| `bitstream.bit` | *(required)* | Local path to the Vivado `.bit` file |
| `bitstream.hwh` | auto-detect | Local path to the hardware-handoff `.hwh`; auto-detected as `<bit_stem>.hwh` when omitted |
| `bitstream.dtbo` | *(required)* | Local path to the compiled device tree overlay `.dtbo` |
| `bitstream.overlay_name` | `.dtbo` stem | Configfs directory name and `/lib/firmware/<name>.bin` filename |
| `bitstream.xclbinutil` | `"xclbinutil"` | `xclbinutil` command or absolute path |

### Usage

```bash
cd inference-scheduler

# Load bitstream — all paths from config
.venv/bin/python upload_bitstream.py --config bitstream_config_kv260.json

# Override individual paths on the CLI (takes precedence over config)
.venv/bin/python upload_bitstream.py --config bitstream_config_kv260.json \
    --bit   ../hw/cormorant_hw_128/.../design_cormorant_wrapper.bit \
    --dtbo  ../build/dts/kv260/design_cormorant.dtbo

# Check board readiness without loading anything
.venv/bin/python upload_bitstream.py --config bitstream_config_kv260.json \
    --check-only
```

`--bit`, `--hwh`, `--dtbo`, `--overlay-name` and `--xclbinutil` override the
matching `bitstream.*` config keys.

The `bitstream.bit` of `bitstream_config_kv260.json` also names the
bitstream the scheduler plans for: its id (first 12 hex digits of the
SHA-256 of the flat image) selects the performance model
`perf_models/kv260/<bitstream-id>.json` that `--plan` and
`perf_calibrate.py` use (see [Calibration Campaign](#calibration-campaign-perf_calibratepy)).

After a successful run, verify the UIO devices are up:

```bash
cat /sys/class/uio/uio*/name
# axi-pmon            (×4, uio0–3: the Kria image's performance monitors)
# fabric_vecop
# fabric_matmul
# fabric_conv
# fabric_pool
```

### Implementation

The script and its helper modules live in `src/bitstream/`:

| Module | Contents |
|--------|----------|
| `src/bitstream/convert.py` | `.bit` → `.bin` conversion |
| `src/bitstream/hwh.py` | HWH parsing — PS params and memory topology |
| `src/bitstream/xclbin.py` | xclbin synthesis via `xclbinutil` |
| `src/bitstream/board.py` | All remote SSH/SFTP board operations |
| `src/bitstream/loader.py` | `upload_bitstream()` orchestration |
| `src/bitstream/platforms/kv260.py` | KV260 register tables (`FPD_SLCR_REG`, `AXIFM_REG`) and `BLANK_METADATA` XML |
| `src/remote/` | SSH session, config defaults and preflight checks shared with the test runners |

---

## Running Tests

### Full test run

```bash
.venv/bin/python run_remote_tests.py --config remote_config.json
```

### Preflight check (no tests)

Verify SSH connectivity and all board prerequisites without running any tests:

```bash
.venv/bin/python run_remote_tests.py --config remote_config.json --check-only
```

Example output (VectorOPKernel-only model):

```
Connecting to root@192.168.100.8:22 …
  Connected

Remote prerequisites
    OK      cmake                        cmake version 3.22.1
    OK      make                         GNU Make 4.3
    OK      gcc                          gcc (Ubuntu 11.4.0-1ubuntu1~22.04.3) 11.4.0
    OK      xrt headers                  xrt via pkg-config
    OK      sudo / root                  passwordless sudo OK
    OK      uio (VectorOPKernel: fabric_vecop)   /dev/uio4
```

For a model that uses all four kernels:

```
    OK      uio (VectorOPKernel: fabric_vecop)   /dev/uio4
    OK      uio (MatmulKernel: fabric_matmul)    /dev/uio5
    OK      uio (ConvKernel: fabric_conv)        /dev/uio6
    OK      uio (PoolKernel: fabric_pool)        /dev/uio7
```

### Subset of models

Override the model list on the command line (paths relative to `inference-scheduler/`):

```bash
.venv/bin/python run_remote_tests.py --config remote_config.json \
    --models test/models/single_add.onnx test/models/mixed_ops.onnx
```

### Verbose output

Show full cmake/make and test binary output for every model, not just failures:

```bash
.venv/bin/python run_remote_tests.py --config remote_config.json --verbose
```

### Keep remote directories (debugging)

Preserve remote build directories after the run so you can SSH in and inspect:

```bash
.venv/bin/python run_remote_tests.py --config remote_config.json --no-cleanup
# then: ssh root@192.168.100.8
#       ls /tmp/inference_hw_tests/
#       cat /tmp/inference_hw_tests/single_add/build/CMakeFiles/CMakeError.log
```

### Stop on first failure

```bash
.venv/bin/python run_remote_tests.py --config remote_config.json --fail-fast
```

---

## Reading the Summary

```
  Model                   Status                 Time  Steps
  ──────────────────────────────────────────────────────────
  single_add              PASSED                  3.4s  generate  upload  build  run
  mixed_ops               PASSED                  3.3s  generate  upload  build  run
  sat_div_pos             BUILD_ERROR             2.4s  generate  upload  build
```

| Status | Meaning |
|--------|---------|
| `PASSED` | All steps completed; `test_inference` printed `PASSED` |
| `GENERATE_ERROR` | `inference_scheduler.py` failed locally |
| `UPLOAD_ERROR` | SFTP transfer to the board failed |
| `DRIVERS_ERROR` | A `local.driver_dir(s)` path does not exist, or copying `remote.driver_dir(s)` on the board failed |
| `BUILD_ERROR` | `cmake` or `make` failed on the board |
| `RUN_ERROR` | `test_inference` ran but printed `FAILED` or exited non-zero |

For any non-`PASSED` result the full step output is printed below the table.

---

## Debugging Failures

### BUILD_ERROR

Run with `--no-cleanup`, SSH into the board, and check the build log:

```bash
.venv/bin/python run_remote_tests.py --config remote_config.json \
    --models test/models/my_model.onnx --no-cleanup

ssh root@192.168.100.8
cd /tmp/inference_hw_tests/my_model/build
make test_inference          # re-run make interactively
cat CMakeFiles/CMakeError.log
```

Common causes:

- **C name conflict**: a tensor named after a C standard library function (e.g.,
  `div`, `log`, `exp`) generates a variable that shadows the library symbol.
  Fix: rename the initializer tensor in the model generator.
- **Missing driver files**: `driver/` is empty because neither `local.driver_dir`
  nor `remote.driver_dir` was set, or the path does not contain the expected
  `.c`/`.h` files.
- **XRT not found**: `xrt.h` is not in the system include path and `XRT_DIR` was
  not set. Add `"-DXRT_DIR=/opt/xilinx/xrt"` to `remote.cmake_args`.

### RUN_ERROR — value mismatch

The test binary prints the failing element index and both the expected and actual
values. Run with `--verbose` to see the full output:

```bash
.venv/bin/python run_remote_tests.py --config remote_config.json --verbose \
    --models test/models/my_model.onnx
```

A systematic 1-LSB difference between expected and actual usually indicates a
quantization rounding mismatch between the Python simulator and the hardware. See
`src/dtype.py` for the three-mode fixed-point rounding model:
- `quantize()` — round-to-nearest (weight initialisation)
- `truncate()` — floor toward −∞ (Add/Sub/Mul output narrowing, AP_TRN)
- `truncate_div()` — truncate toward zero (Div, C integer division semantics)

### UIO device not found

The prerequisite check reports `MISSING uio (VectorOPKernel: fabric_vecop)`.

The UIO sysfs name comes from the **node label** in the DTS overlay.  Check what
names the loaded overlay actually exported:

```bash
# On the board: list UIO device names
cat /sys/class/uio/uio*/name

# Expected for cormorant.dts (all four kernels), after the image's four axi-pmon:
# fabric_vecop
# fabric_matmul
# fabric_conv
# fabric_pool
```

If the names differ, find the `.dts` that was used to build the overlay:

```bash
grep -B3 'generic-uio' dts/kv260/cormorant.dts
```

Update `remote.uio_devices` in your config to match the labels found there.
The driver scans `/sys/class/uio/uio*/name` for an exact string match — these
must be sysfs names, not `/dev/uioN` paths.

If no UIO devices appear at all, the DTBO overlay may not be loaded.
Load it with `upload_bitstream.py` (see [Bitstream Upload](#bitstream-upload-upload_bitstreampy)):

```bash
.venv/bin/python upload_bitstream.py --config bitstream_config_kv260.json

# Verify
cat /sys/class/uio/uio*/name
```

---

## Performance Benchmarking (`run_remote_perf.py`)

`run_remote_perf.py` measures raw kernel throughput and latency on a physical
KV260. Unlike `run_remote_tests.py`, it does **not** check numerical correctness
— it only cares about how fast each kernel runs for a given set of parameters.

The script:
1. Assembles a self-contained C benchmark project locally from `bench_src/`
   (four standalone binaries, one per kernel, plus `calib_runner` for
   `perf_calibrate.py` when all four kernels' drivers are present)
2. Uploads it to the board once, builds everything in one `cmake` + `make` pass
3. Runs each test case as a separate binary invocation and parses the JSON output
4. Prints a formatted table of latency (ms) and throughput (GB/s or GOps/s)

Kernels whose driver files are absent are not built (the runner warns about
the missing files; their cases then fail with `ERR`) — disable them with
`enabled: false` or select the deployed ones with `--kernels`.

---

### Prerequisites

Same SSH, toolchain, and XRT requirements as `run_remote_tests.py`. See
[Prerequisites](#prerequisites) above. Run `--check-only` to verify before
starting a long benchmark run.

Driver files must be available either locally (`local.driver_dirs`) or on the
board (`remote.driver_dirs`). The local paths are the standard Vitis HLS output
(see [Prerequisites](#prerequisites); VectorOPKernel under
`<hls_project>/solution1/impl/ip/`, the other three under
`<hls_project>/hls/impl/ip/`):

```
<repo>/build/kernels/<kernel>/kv260/<hls_project>/{solution1,hls}/impl/ip/drivers/<KernelName>_v1_0/src/
```

---

### Config File — `perf_config.json`

`perf_config.json` extends the same schema as the correctness-test configs with
one additional top-level section: `benchmarks`.

#### Full key reference

All keys from the correctness-test config apply (see [Config File Reference](#config-file-reference)). Additional and modified keys:

| Key | Default | Description |
|-----|---------|-------------|
| `remote.work_dir` | `"/tmp/inference_hw_tests"` | Temporary build directory on the board |
| `remote.uio_devices` | `{}` | Per-kernel UIO sysfs names (keys `VectorOPKernel`, `MatmulKernel`, `ConvKernel`, `PoolingKernel`); a missing entry falls back to `<Kernel>_0` |
| `local.driver_dirs` | `{}` | Per-kernel local HLS driver source paths; merged before upload |
| `build.timeout` | `300` | Combined cmake + make timeout in seconds |
| `run.timeout` | `60` | Per-benchmark binary execution timeout in seconds |
| `benchmarks.<Kernel>.enabled` | `true` | Set `false` to skip that kernel entirely |
| `benchmarks.<Kernel>.warmup` | `10` | Default warmup iterations for all cases in this kernel group |
| `benchmarks.<Kernel>.cases` | *(required)* | List of named-parameter case dicts; see [Case Fields](#case-fields). Kernels without a `cases` array are skipped. |

#### `benchmarks` section

Each kernel group has three keys:

```json
"benchmarks": {
  "VectorOPKernel": {
    "enabled": true,
    "warmup": 10,
    "cases": [ ... ]
  },
  "MatmulKernel": { "enabled": true, "warmup": 10, "cases": [ ... ] },
  "ConvKernel":   { "enabled": true, "warmup": 10, "cases": [ ... ] },
  "PoolingKernel":{ "enabled": true, "warmup": 10, "cases": [ ... ] }
}
```

If `"cases"` is omitted or empty that kernel is skipped entirely. If
`"enabled"` is `false` no cases from that kernel are loaded or run
(equivalent to `--kernels` without that kernel).

#### Case fields

Each element of `"cases"` is a JSON object with a `"label"` string plus the
kernel-specific numeric fields listed below. All numeric values are integers.
An optional `"warmup"` field overrides the per-kernel warmup for that case.

**VectorOPKernel** — element-wise op benchmark

| Field | Description |
|-------|-------------|
| `label` | Display name in the report |
| `op` | Opcode: 0=ADD 1=SUB 2=MUL 3=DIV 4=RELU 5=RELU6. Any other value is rejected by `run_remote_perf.py` when it loads the cases (right after the config, before it connects to the board) with `config error: VectorOPKernel case '<label>': unsupported op=…`. |
| `size` | Elements per inner kernel call |
| `outer` | Outer loop count; `outer=1` is the non-broadcast case |
| `a_inc` | Stride for A between outer iterations (`size` to advance, `0` to repeat) |
| `b_inc` | Stride for B between outer iterations (`size` to advance, `0` to repeat) |
| `iters` | Timed iterations (after warmup) |

Binary ops touch 3 memory ports (A, B, C); unary ops (RELU, RELU6) touch 2.
The reported **GB/s** accounts for this: `ports × size × outer × 2B / lat`.

**MatmulKernel** — matrix multiply benchmark

| Field | Description |
|-------|-------------|
| `n` | Rows of A and C |
| `k` | Columns of A / rows of B (accumulation dimension) |
| `m` | Columns of B and C |
| `batch` | Batch size; `1` = single matrix multiply |
| `a_stride` | Elements between A batch slices (`n×k` for batched, `0` to broadcast A) |
| `b_stride` | Elements between B batch slices (`k×m` for batched, `0` to broadcast B) |
| `b_packed` | Optional (default 0): 1 = B is in the packed tile-major layout `[ceil(m/32)][k][32]` (MATMUL_OPTIMISATION §3b) |
| `gemv_kw` | Optional (default 0): 1 / 2 / 4 / 8 = the GEMV streaming path with B in the ConvKernel image of that kernel width (MATMUL_OPTIMISATION §8b); `b_packed` is then ignored |
| `iters` | Timed iterations |

Reported metric: **GOps/s** = `2 × batch × n × k × m / lat`.

**ConvKernel** — 2-D NCHW convolution benchmark

| Field | Description |
|-------|-------------|
| `batch` | Batch size |
| `in_ch` | Input channels |
| `in_h`, `in_w` | Input spatial dimensions |
| `out_ch` | Output channels |
| `kh`, `kw` | Kernel (filter) size |
| `stride_h`, `stride_w` | Convolution stride |
| `dilation_h`, `dilation_w` | Dilation factor (1 = standard conv) |
| `pad_top`, `pad_left` | Zero-padding; use `(k−1)/2` for same-padding |
| `has_bias` | 1 to include a bias buffer in the benchmark, 0 to skip |
| `is_dw` | 1 for depthwise (grouped, `in_ch=out_ch`), 0 for standard |
| `iters` | Timed iterations |

Output size is computed by the benchmark binary: `out_h = (in_h + 2×pad_top − dil×(kh−1) − 1) / stride_h + 1`.
Reported metric: **GOps/s** = `2 × MACs / lat` where MACs = `batch × out_ch × out_h × out_w × ic × kh × kw`
(standard conv) or `batch × out_ch × out_h × out_w × kh × kw` (depthwise).

**PoolingKernel** — 2-D NCHW pooling benchmark

| Field | Description |
|-------|-------------|
| `batch`, `channels` | Batch and channel count |
| `in_h`, `in_w` | Input spatial dimensions |
| `pool_h`, `pool_w` | Pooling window size; set to `in_h×in_w` for global pool |
| `stride_h`, `stride_w` | Pooling stride |
| `pad_top`, `pad_left` | Zero-padding |
| `dil_h`, `dil_w` | Dilation (1 = standard) |
| `pool_type` | 0=MaxPool 1=AveragePool 2=LpPool |
| `lp_order` | P value for LpPool (1 or 2); ignored for other types |
| `count_include_pad` | 1 to include padding in the average denominator |
| `iters` | Timed iterations |

Reported metric: **GB/s** = `(x_bytes + y_bytes) / lat`.

---

### Running Benchmarks

#### Full benchmark run (all enabled kernels)

```bash
.venv/bin/python run_remote_perf.py --config perf_config.json
```

#### Specific kernels only (overrides `enabled` in config)

```bash
.venv/bin/python run_remote_perf.py --config perf_config.json \
    --kernels VectorOPKernel MatmulKernel
```

#### Override iteration and warmup counts

```bash
.venv/bin/python run_remote_perf.py --config perf_config.json \
    --iters 500 --warmup 20
```

`--iters` overrides the per-case `iters` field for every case. `--warmup`
overrides the per-kernel `warmup` for every case.

#### Machine-readable results

```bash
.venv/bin/python run_remote_perf.py --config perf_config.json --json results.json
```

`--json OUT` also writes the results as a JSON list, one object per case:
`kernel`, `label`, `fields` (the case's fields), `warmup`, `ok`, `lat_ms`
and the metric (`gbs` or `gops`; `null` for a failed case).

#### Preflight check (no benchmarks)

```bash
.venv/bin/python run_remote_perf.py --config perf_config.json --check-only
```

#### Verbose — show error output for failed cases

```bash
.venv/bin/python run_remote_perf.py --config perf_config.json --verbose
```

#### Keep remote build for manual inspection

```bash
.venv/bin/python run_remote_perf.py --config perf_config.json --no-cleanup
# then: ssh root@192.168.100.8
#       ls /tmp/kernel_perf/kv260_perf/build/
#       sudo /tmp/kernel_perf/kv260_perf/build/bench_vectorop fabric_vecop ADD-1K 0 1024 1 0 0 100 5
#       (args: instance label op size outer a_inc b_inc iters warmup)
```

---

### Reading the Report

After all cases run, the script prints a per-kernel table.  The sample below
is the 60-case `perf_config.json.example` set on 2026-09-29 (hw_128 d7ce129,
100 MHz; the `perf-regression` skill compares a run with this kind of
baseline):

```
  VectorOPKernel
  ───────────────────────────────────────────────────────────────────────────────────
  Label                    Parameters                        Lat(ms)      GB/s
  ───────────────────────────────────────────────────────────────────────────────────
  ADD-1K                   ADD    size=1024    outer=1        0.0087     0.703
  ADD-4K                   ADD    size=4096    outer=1        0.0175     1.406
  ADD-16K                  ADD    size=16384   outer=1        0.0483     2.034
  ADD-64K                  ADD    size=65536   outer=1        0.1711     2.298
  ADD-256K                 ADD    size=262144  outer=1        0.6627     2.373
  MUL-16K                  MUL    size=16384   outer=1        0.0482     2.039
  MUL-64K                  MUL    size=65536   outer=1        0.1714     2.294
  DIV-4K                   DIV    size=4096    outer=1        0.0492     0.500
  RELU-16K                 RELU   size=16384   outer=1        0.0281     2.334
  RELU-64K                 RELU   size=65536   outer=1        0.0901     2.910
  RELU6-16K                RELU6  size=16384   outer=1        0.0282     2.321
  ADD-bcast-8x16K          ADD    size=16384   outer=8        0.3351     2.347
  MUL-bcast-8x16K          MUL    size=16384   outer=8        0.3351     2.347
  RELU-bcast-8x16K         RELU   size=16384   outer=8        0.1731     3.028
  MUL-bcast-dw-12544x16    MUL    size=16      outer=12544     0.2615     4.604
  ───────────────────────────────────────────────────────────────────────────────────
                                                 peak GB/s                4.604
                                               min latency     0.0087
  15/15 OK

  MatmulKernel
  ────────────────────────────────────────────────────────────────────────────────────
  Label                     Parameters                        Lat(ms)    GOps/s
  ────────────────────────────────────────────────────────────────────────────────────
  8x8x8                     N=8    K=8    M=8    batch=1       0.0114     0.090
  16x16x16                  N=16   K=16   M=16   batch=1       0.0209     0.391
  32x32x32                  N=32   K=32   M=32   batch=1       0.0499     1.313
  64x64x64                  N=64   K=64   M=64   batch=1       0.2245     2.335
  128x128x128               N=128  K=128  M=128  batch=1       1.1912     3.521
  256x256x256               N=256  K=256  M=256  batch=1       7.3934     4.538
  FC-1x256x256              N=1    K=256  M=256  batch=1       0.1042     1.258
  FC-4x256x256              N=4    K=256  M=256  batch=1       0.1306     4.016
  batch4-64x64x64           N=64   K=64   M=64   batch=4       0.8746     2.398
  batch4-A-bcast            N=64   K=64   M=64   batch=4       0.8746     2.398
  dw-12544x16x1             N=12544 K=16   M=1    batch=1      6.9414     0.058
  dw-12544x16x3             N=12544 K=16   M=1    batch=3     20.8133     0.058
  FC-1x256x256-packed       N=1    K=256  M=256  batch=1       0.1009     1.299
  FC-4x256x256-packed       N=4    K=256  M=256  batch=1       0.1270     4.128
  256x256x256-packed        N=256  K=256  M=256  batch=1       7.1711     4.679
  FC-1x512x1000             N=1    K=512  M=1000 batch=1       0.9782     1.047
  FC-1x512x1000-packed      N=1    K=512  M=1000 batch=1       0.7201     1.422
  FC-1x1280x1001            N=1    K=1280 M=1001 batch=1       2.4396     1.050
  FC-1x1280x1001-packed     N=1    K=1280 M=1001 batch=1       1.7475     1.466
  FC-1x512x1000-gemv        N=1    K=512  M=1000 batch=1 GEMV kw=1     0.3384     3.026
  FC-1x576x1536-packed      N=1    K=576  M=1536 batch=1       1.2189     1.452
  FC-1x576x1536-gemv-kw4    N=1    K=576  M=1536 batch=1 GEMV kw=4     0.5772     3.066
  FC-1x1536x576-packed      N=1    K=1536 M=576  batch=1       1.1801     1.499
  FC-1x1536x576-gemv-kw4    N=1    K=1536 M=576  batch=1 GEMV kw=4     0.5675     3.118
  ────────────────────────────────────────────────────────────────────────────────────
                                                peak GOps/s                4.679
                                                min latency     0.0114
  24/24 OK

  ConvKernel
  ─────────────────────────────────────────────────────────────────────────────────────
  Label                      Parameters                        Lat(ms)    GOps/s
  ─────────────────────────────────────────────────────────────────────────────────────
  3x3-1ch-28x28-32out        1ch 28x28→32ch 3x3k                0.1490     3.031
  3x3-1ch-28x28-32out-b16    1ch 28x28→32ch 3x3k                2.0737     3.484
  3x3-64ch-56x56             64ch 56x56→64ch 3x3k               2.6830    86.175
  3x3-64ch-56x56-s2          64ch 56x56→64ch 3x3k               0.7104    81.362
  3x3-64ch-28x28             64ch 28x28→64ch 3x3k               0.7096    81.454
  1x1-64to128-56x56          64ch 56x56→128ch 1x1k              1.8381    27.953
  1x1-128to256-28x28         128ch 28x28→256ch 1x1k             1.6417    31.298
  5x5-16ch-28x28             16ch 28x28→16ch 5x5k               0.1562    64.231
  dw-3x3-32ch-28x28          32ch 28x28→32ch 3x3k               0.1540     2.933
  dw-3x3-64ch-56x56          64ch 56x56→64ch 3x3k               1.0129     3.567
  ─────────────────────────────────────────────────────────────────────────────────────
                                                 peak GOps/s               86.175
                                                 min latency     0.1490
  10/10 OK

  PoolingKernel
  ────────────────────────────────────────────────────────────────────────────────────
  Label                     Parameters                        Lat(ms)      GB/s
  ────────────────────────────────────────────────────────────────────────────────────
  MaxPool-2x2-56x56         MaxPool 2x2 64ch 56x56             0.3352     1.497
  MaxPool-2x2-28x28         MaxPool 2x2 64ch 28x28             0.1161     1.080
  MaxPool-3x3-56x56         MaxPool 3x3 64ch 56x56             0.3385     1.482
  MaxPool-3x3-14x14         MaxPool 3x3 64ch 14x14             0.0502     0.625
  AvgPool-2x2-56x56         AvgPool 2x2 64ch 56x56             0.3350     1.498
  AvgPool-3x3-28x28         AvgPool 3x3 64ch 28x28             0.1164     1.078
  GlobalMaxPool-7x7-256     MaxPool 7x7 256ch 7x7              0.0770     0.332
  GlobalAvgPool-7x7-64      AvgPool 7x7 64ch 7x7               0.0254     0.252
  GlobalAvgPool-7x7-1024    AvgPool 7x7 1024ch 7x7             0.2828     0.362
  AvgPool-2x2-7x7-1024      AvgPool 7x7 1024ch 7x7             0.2828     0.362
  AvgPool-2x2-3x3-32-112    AvgPool 3x3 32ch 112x112           0.6793     1.467
  ────────────────────────────────────────────────────────────────────────────────────
                                                  peak GB/s                1.498
                                                min latency     0.0254
  11/11 OK

  ── OVERALL: All 60 cases passed ──
```

| Column | Meaning |
|--------|---------|
| `Lat(ms)` | Mean kernel execution time per call (wall-clock, after warmup) |
| `GB/s` | Memory bandwidth for VectorOPKernel and PoolingKernel |
| `GOps/s` | Arithmetic throughput for MatmulKernel and ConvKernel |
| `peak GB/s` / `peak GOps/s` | Best metric across all passing cases in the group |
| `min latency` | Shortest latency across all passing cases in the group |
| `ERR` | Case failed; run with `--verbose` to see the error message |

The latency includes `inference_buf_sync_from_device()` (cache invalidation after
each kernel write). For large buffers this adds a measurable but consistent
overhead; it is included because it is part of the real inference path.

---

### Disabling Kernels

To skip a kernel group without removing its cases from the config:

```json
"benchmarks": {
  "ConvKernel": { "enabled": false, "warmup": 10, "cases": [ ... ] }
}
```

The CLI `--kernels` flag takes precedence over `enabled`: passing
`--kernels VectorOPKernel` runs only VectorOPKernel regardless of which kernels
are marked `enabled` in the config.

---

### Adding Custom Test Cases

Append entries to the kernel's `"cases"` array in `perf_config.json`. Use the
field tables above to set the parameters. Example — a large square matmul:

```json
{"label": "512x512x512", "n": 512, "k": 512, "m": 512,
 "batch": 1, "a_stride": 0, "b_stride": 0, "iters": 5}
```

The `"cases"` array is required; a kernel with no `"cases"` key (or an empty
array) is skipped.

---

### Debugging Failed Cases

A case shows `ERR` when the benchmark binary exits non-zero. Common causes:

| Error | Cause | Fix |
|-------|-------|-----|
| `xclOpen(0) failed` | XRT not loaded / no bitstream active | Load the bitstream: `python upload_bitstream.py --config bitstream_config_kv260.json` |
| `bench_<kernel>: init '<name>' failed` | UIO device name mismatch | Check `cat /sys/class/uio/uio*/name` and update `remote.uio_devices` |
| `alloc failed` | DMA buffer allocation failed | Reduce `size` or number of concurrent allocations; check `dmesg` for CMA |
| `No such file` (binary missing) | Driver files not found at build time | Verify `local.driver_dirs` paths exist and contain all `x<kernel>*.c/.h` files; re-run with `--no-cleanup` and inspect cmake output |

Run with `--verbose` to see the first 5 lines of stderr from the failing binary:

```bash
.venv/bin/python run_remote_perf.py --config perf_config.json --verbose
```

Run with `--no-cleanup` to keep the build on the board and invoke the binary
manually for interactive debugging.

---

## Calibration Campaign (`perf_calibrate.py`)

The optional planning mode (`inference_scheduler.py --plan`) prices its
choices with a performance model of the loaded bitstream,
`perf_models/kv260/<bitstream-id>.json` (`perf_models/README.md`).
`perf_calibrate.py` makes it, once per bitstream: it builds the benchmark
project of `run_remote_perf.py` (same config format; `calib_runner` needs all
four kernels' drivers) and measures every kernel call of the shipped models
and their MatMul tactics plus a space-filling set, twice.

```bash
.venv/bin/python perf_calibrate.py cases                  # the case list
.venv/bin/python perf_calibrate.py run --config perf_config.json --stop-server
.venv/bin/python perf_calibrate.py fit                    # the model + validation report
# or all three: perf_calibrate.py all --config perf_config.json --stop-server
```

`run` checks that the board runs the bitstream the cases are for
(`--board-bin`, default `/lib/firmware/pl.bin`), refuses to run while the
chat server owns the kernels unless `--stop-server` (stops it and restarts
it afterwards), and `--resume` keeps the measurements already taken.
`cases --refine` adds the tactics the fitted model ranks near the best;
`host` / `simulate` build the host-op model and compare predictions with
board profiles.  Full workflow:
[`doc/scheduler/INFERENCE_SCHEDULER.md` §Planning](../../doc/scheduler/INFERENCE_SCHEDULER.md#planning---plan).

---

## Multi-Kernel Models (MatmulKernel + VectorOPKernel)

Models that combine kernels (e.g., `MatMul → Relu`, `Add → MatMul`) need
every UIO device they use active on the board.  The Cormorant bitstream
carries all four kernels, so one bitstream covers every test model.

### 1. Generate mixed-kernel test models

```bash
.venv/bin/python test/gen_mixed_kernel_models.py
# Creates 16 models in test/models/:
#   mixed_matmul_relu.onnx          mixed_add_matmul.onnx
#   mixed_matmul_add_relu.onnx      mixed_two_layer_mlp.onnx
#   mixed_add_matmul_unaligned.onnx mixed_matmul_scale_bias.onnx
#   mixed_outer_matmul_relu.onnx    mixed_relu6_matmul_add.onnx
#   mixed_residual.onnx             mixed_batch_matmul_relu.onnx
#   mixed_sub_div_matmul.onnx       mixed_two_input_matmul.onnx
#   mixed_two_output.onnx           mixed_two_input_two_output.onnx
#   mixed_spatial_matmul_relu.onnx  mixed_skip_connection.onnx
# Three models exercise multiple graph inputs and/or outputs:
#   mixed_two_input_matmul:      two inputs (X1, X2), one output
#   mixed_two_output:            one input, two outputs (Yadd, Yrelu)
#   mixed_two_input_two_output:  two inputs (X1, X2), two outputs (Yadd, Yrelu)
```

### 2. Load the bitstream on the board

```bash
# From the inference-scheduler directory on the local machine
.venv/bin/python upload_bitstream.py --config bitstream_config_kv260.json

# Verify UIO devices are up
cat /sys/class/uio/uio*/name   # run on the board
# axi-pmon (×4), then
# fabric_vecop
# fabric_matmul
# fabric_conv
# fabric_pool
```

See [Bitstream Upload](#bitstream-upload-upload_bitstreampy) for config reference and CLI options.

### 3. Run them

```bash
# remote_config.json (from remote_config.json.example) with local.driver_dirs
# pointing at your HLS output directories

# Preflight check
.venv/bin/python run_remote_tests.py --config remote_config.json --check-only

# Run only the mixed-kernel models
.venv/bin/python run_remote_tests.py --config remote_config.json \
    --models test/models/mixed_*.onnx
```

The runner merges driver files from `local.driver_dirs.VectorOPKernel` and
`local.driver_dirs.MatmulKernel` before uploading, and passes both UIO names
to `cmake` as compile definitions:

```
cmake … -DINFERENCE_VECTOROPKERNEL_INSTANCE=\"fabric_vecop\" \
         -DINFERENCE_MATMULKERNEL_INSTANCE=\"fabric_matmul\"
```

### 4. Config files by purpose

Tracked templates (copy, then edit the copy):

| Config file | Script | Purpose |
|-------------|--------|---------|
| `bitstream_config_kv260.json.example` | `upload_bitstream.py` | Load Cormorant bitstream + xclbin + DTBO onto the board |
| `remote_config.json.example` | `run_remote_tests.py` | Correctness tests — 148 models over all four kernels |
| `perf_config.json.example` | `run_remote_perf.py`, `perf_calibrate.py run` | Performance benchmarks — 60 cases; the board config of the calibration campaign |

Per-subset copies such as `remote_config_vectorop.json`, `remote_config_conv.json`
or `remote_config_all_models.json` are local working files (not tracked):
the same schema with a narrower `models` list.

---

## Full Example: Adding a New Model

1. Create the ONNX model in `test/gen_test_models.py` and re-generate:
   ```bash
   .venv/bin/python test/gen_test_models.py
   ```

2. Verify the Python simulation locally:
   ```bash
   .venv/bin/python -m pytest test/test_saturation.py -v
   ```

3. Add the model path to `remote_config.json`:
   ```json
   "models": [
     "test/models/sat_add_pos.onnx",
     "test/models/my_new_model.onnx"
   ]
   ```

4. Run on hardware:
   ```bash
   .venv/bin/python run_remote_tests.py --config remote_config.json \
       --models test/models/my_new_model.onnx
   ```
