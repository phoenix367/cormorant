# MNIST KV260 demo

End-to-end MNIST inference demo for the KV260 FPGA platform.  Downloads the
MNIST test split (from the `ossci-datasets` S3 mirror of LeCun's files) and
two pre-trained ONNX models (an MNIST convnet and a LeNet variant) from a
shared Google Drive folder, generates a self-contained KV260 inference
project for each model with `inference-scheduler`, copies in the kernel driver
sources, builds the project on the board over SSH, and runs a benchmark
that reports top-1 accuracy and per-image latency for each.

```mermaid
flowchart LR
    A["download_assets.py<br/>MNIST + .onnx"]
      --> B["generate_project.py<br/>per-model CMake project<br/>+ bench_glue.h"]
      --> C["deploy_and_run.py<br/>SSH upload, build,<br/>run bench_mnist"]
```

## Layout

```
demo/mnist/
├── README.md
├── requirements.txt
├── mnist_config.json.example       — copy to mnist_config.json and edit
├── run_demo.py                     — one-shot orchestrator
├── scripts/
│   ├── download_assets.py          — fetch MNIST IDX + ONNX models
│   ├── generate_project.py         — schedule each ONNX → CMake project
│   └── deploy_and_run.py           — upload, build, run on KV260 over SSH
├── src/
│   └── bench_mnist.c               — benchmark host (compiled on the board)
├── assets/                         — populated by download_assets.py
│   ├── data/                       — MNIST IDX files
│   └── models/                     — ONNX files from Google Drive
└── build/                          — generator output and results.json
    ├── projects/projects.json      — summary of the generated projects (read by deploy)
    ├── projects/<model>/           — generated CMake project per model
    │   ├── CMakeLists.txt          —   patched to build bench_mnist
    │   ├── include/inference*.h    —   from inference-scheduler
    │   ├── src/inference*.c        —   from inference-scheduler
    │   ├── weights/*.dat           —   large weight tensors, read at runtime
    │   ├── driver/                 —   kernel driver sources copied in
    │   └── test/
    │       ├── bench_mnist.c       —   copied from demo/mnist/src/
    │       └── bench_glue.h        —   generated; per-model glue + macros
    ├── logs/<model>.<step>.log     — per-step build/run output
    └── results.json                — final benchmark summary
```

`test/bench_glue.h` is **generated per model** by `scripts/generate_project.py`
— not committed in the repo.  Each copy bakes in the right
`INFERENCE_<INPUT>_SIZE` / `INFERENCE_<OUTPUT>_SIZE` macros and a
`bench_inference_init()` shim that calls `inference_init()` with the correct
number of UIO arguments for the kernels that model actually uses (both
models use all 4 today; a model without a MatMul would get 3).  That's why it's generated rather than
static: the I/O names and active-kernel set differ per model.

## Prerequisites

### Host (workstation that orchestrates the demo)

* Python 3.10+
* `pip install -r requirements.txt`
* Generated driver sources for the four kernels (HLS; the RTL MatmulKernel's
  by `driver_matmul_rtl`).  These are produced by the top-level CMake build:

  ```bash
  # from the repo root
  mkdir -p build && cd build
  cmake ..
  make synthesize_kv260
  ```

  The driver sources only describe the kernels' AXI-Lite registers, so they
  are the same for every `AXI_BUS_WIDTH`; the bitstream on the board is the
  128-bit block design (`-DAXI_BUS_WIDTH=128`, `make build_hw_kv260`), which
  the sample numbers below were measured with.
  The default `mnist_config.json.example` assumes the standard build paths;
  override `local.driver_dirs` if you keep the build elsewhere.

### KV260 board

* Linux with the cormorant bitstream and overlay loaded
  (`inference-scheduler/upload_bitstream.py`, see the root README), so the
  four UIO devices appear.  Verify on the board:

  ```bash
  cat /sys/class/uio/uio*/name
  # → fabric_vecop, fabric_matmul, fabric_conv, fabric_pool
  ```

* `gcc`, `cmake ≥ 3.19`, `make`
* XRT runtime available via `pkg-config xrt` or under `/opt/xilinx/xrt`
* Passwordless `sudo` for the SSH user (XRT requires root for
  buffer allocation).

## First-time setup

```bash
cd demo/mnist
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

cp mnist_config.json.example mnist_config.json
$EDITOR mnist_config.json          # set ssh.host, key_file, etc.
```

## Run the demo

One-liner that does download → generate → deploy:

```bash
.venv/bin/python run_demo.py
```

Or step-by-step (lets you iterate without re-downloading):

```bash
.venv/bin/python scripts/download_assets.py
.venv/bin/python scripts/generate_project.py
.venv/bin/python scripts/deploy_and_run.py --verbose
```

Sample output (host `~/projects/axi_demo/demo/mnist`, board at
`192.168.100.8`, full 10 000-image MNIST test set):

```
$ ./run_demo.py

=== download_assets ===
MNIST test split → demo/mnist/assets/data
  cached t10k-images-idx3-ubyte.gz (1648877 B)
  cached t10k-labels-idx1-ubyte.gz (4542 B)
  ✓ 10000 test images
ONNX models → demo/mnist/assets/models
  ✓ mnist-simplified.onnx  (25,853 B)
  ✓ lenet_simplified.onnx  (13,100,799 B)
done

=== generate_project ===
[mnist_convnet] scheduling mnist-simplified.onnx
Model      : demo/mnist/assets/models/mnist-simplified.onnx
Inputs     : ['Input3[1, 1, 28, 28]']
Outputs    : ['Plus214_Output_0[1, 10]']
Nodes      : 9
  [  0] Conv         [1, 1, 28, 28] x [8, 1, 5, 5] x [8] -> [1, 8, 28, 28]
  [  1] Relu         [1, 8, 28, 28] -> [1, 8, 28, 28]
  [  2] MaxPool      [1, 8, 28, 28] -> [1, 8, 14, 14]
  [  3] Conv         [1, 8, 14, 14] x [16, 8, 5, 5] x [16] -> [1, 16, 14, 14]
  [  4] Relu         [1, 16, 14, 14] -> [1, 16, 14, 14]
  [  5] MaxPool      [1, 16, 14, 14] -> [1, 16, 4, 4]
  [  6] Reshape      [1, 16, 4, 4] -> [1, 256]
  [  7] MatMul       [1, 256] x [256, 10] -> [1, 10]
  [  8] Add          [1, 10] x [1, 10] -> [1, 10]
[mnist_convnet] active kernels: VectorOPKernel, MatmulKernel, ConvKernel, PoolKernel

[mnist_lenet] scheduling lenet_simplified.onnx
Model      : demo/mnist/assets/models/lenet_simplified.onnx
Inputs     : ['import/Placeholder:0[1, 1, 28, 28]']
Outputs    : ['import/conv4last/BiasAdd:0[1, 10, 1, 1]']
Nodes      : 14
  [  0] Conv         [1, 1, 28, 28] x [32, 1, 5, 5] x [32] -> [1, 32, 28, 28]
  [  1] Relu         [1, 32, 28, 28] -> [1, 32, 28, 28]
  [  2] MaxPool      [1, 32, 28, 28] -> [1, 32, 14, 14]
  [  3] Conv         [1, 32, 14, 14] x [64, 32, 5, 5] x [64] -> [1, 64, 14, 14]
  [  4] Relu         [1, 64, 14, 14] -> [1, 64, 14, 14]
  [  5] MaxPool      [1, 64, 14, 14] -> [1, 64, 7, 7]
  [  6] Flatten      [1, 64, 7, 7] -> [1, 3136]
  [  7] MatMul       [1, 3136] x [3136, 1024] -> [1, 1024]
  [  8] Reshape      [1, 1024] -> [1, 1024, 1, 1]
  [  9] Add          [1, 1024, 1, 1] x [1, 1024, 1, 1] -> [1, 1024, 1, 1]
  [ 10] Flatten      [1, 1024, 1, 1] -> [1, 1024]
  [ 11] MatMul       [1, 1024] x [1024, 10] -> [1, 10]
  [ 12] Reshape      [1, 10] -> [1, 10, 1, 1]
  [ 13] Add          [1, 10, 1, 1] x [1, 10, 1, 1] -> [1, 10, 1, 1]
Weights    : 4 large weight(s) written to build/projects/mnist_lenet/weights/
[mnist_lenet] active kernels: VectorOPKernel, MatmulKernel, ConvKernel, PoolKernel
wrote 2 project(s) under demo/mnist/build/projects

=== deploy_and_run ===

Preflight (local)
    OK      data/t10k-images-idx3-ubyte  7,840,016 B
    OK      data/t10k-labels-idx1-ubyte  10,008 B
    OK      ssh.host configured  192.168.100.8
    OK      project 'mnist_convnet' on disk
    OK        driver/xvectoropkernel.h  (VectorOPKernel)
    OK        driver/xmatmulkernel.h    (MatmulKernel)
    OK        driver/xconvkernel.h      (ConvKernel)
    OK        driver/xpoolingkernel.h   (PoolKernel)
    OK      project 'mnist_lenet' on disk
    OK        driver/xvectoropkernel.h  (VectorOPKernel)
    OK        driver/xmatmulkernel.h    (MatmulKernel)
    OK        driver/xconvkernel.h      (ConvKernel)
    OK        driver/xpoolingkernel.h   (PoolKernel)

Connecting to root@192.168.100.8:22 …
  connected

Preflight (remote)
    OK      cmake                                cmake version 3.22.1
    OK      make                                 GNU Make 4.3
    OK      gcc                                  gcc (Ubuntu 11.4.0-1ubuntu1~22.04.3) 11.4.0
    OK      xrt headers                          xrt via pkg-config
    OK      sudo / root                          passwordless sudo OK
    OK      uio (VectorOPKernel: fabric)         /dev/uio4
    OK      uio (MatmulKernel: fabric_matmul)    /dev/uio5
    OK      uio (ConvKernel:    fabric_conv)     /dev/uio6
    OK      uio (PoolKernel:    fabric_pool)     /dev/uio7
    OK      work_dir parent writable (/tmp)      /tmp/mnist_demo

Uploading dataset → /tmp/mnist_demo/data
  dataset  → OK       2.4s

mnist_convnet
  upload   → OK       0.4s
  cmake    → OK       1.1s
  make     → OK       3.0s
    bench_mnist: dataset=10000 images, iters=10000, warmup=50
                 input_numel=784, output_numel=10, classes=10
    progress: 10000/10000 (100.0%) acc=98.92% mean=4.546ms rate=219.8ips
  run      → OK                3.0s
    accuracy = 98.92%   mean = 0.266 ms   throughput = 3766.7 img/s

mnist_lenet
  upload   → OK       2.0s
  cmake    → OK       0.9s
  make     → OK       2.7s
    bench_mnist: dataset=10000 images, iters=10000, warmup=50
                 input_numel=784, output_numel=10, classes=10
  run      → OK               28.3s
    accuracy = 97.35%   mean = 2.810 ms   throughput = 355.9 img/s

cleanup /tmp/mnist_demo
per-step logs written to demo/mnist/build/logs

  ── MNIST KV260 BENCHMARK ──

  Model          Status       Acc   mean(ms)    p50(ms)    p99(ms)        IPS
  ───────────────────────────────────────────────────────────────────────────
  mnist_convnet  OK       98.92%      0.266      0.265      0.272     3766.7
  mnist_lenet    OK       97.35%      2.810      2.809      2.816      355.9
```

Notable behaviour visible in the run:

- **LeNet's fully-connected Convs run as MatMuls.** Its conv3 (7×7 over
  the 7×7 map, 1024 outputs) and conv4last (1×1 on 1×1) each compute one
  output pixel; the scheduler rewrites them as Flatten + `MatMul` +
  Reshape + bias `Add` (`--fc-conv`, default `auto`), so conv3's 6.4 MB
  weight streams through MatmulKernel's GEMV path (both read ports):
  **5.44 → 2.81 ms per image** (183.8 → 355.9 img/s), accuracy unchanged —
  see [`doc/plans/LENET_PLAN.md`](../../doc/plans/LENET_PLAN.md).  Both
  models now use all four kernels; `generate_project.py` still emits a
  `bench_glue.h` per model so `inference_init()` gets the right number of
  UIO arguments.
- **LeNet's 97.35 % is the model's own.** In float (onnxruntime, the same
  p/256 inputs) it scores 97.37 %; the board agrees with float on 99.91 %
  of the images.
- **Large weights go to `weights/`.** Tensors over the inline-array
  threshold end up as external `.dat` files loaded at runtime by
  `fread()`: all four `mnist_lenet` conv weights (the 1024×64×7×7
  fully-connected-equivalent weight alone is 6.4 MB) and the convnet's
  `MatMul` weight.
- The full per-image timing series is also written to
  `build/results.json`.

## Useful options

| Flag | Effect |
|------|--------|
| `--models mnist_lenet` | restrict generate/deploy to a subset of models (names from `models[].name`) |
| `--skip-download` | reuse cached MNIST + ONNX files |
| `--skip-generate` | reuse the generated projects under `build/projects/` |
| `--skip-deploy` | only regenerate the local CMake projects |
| `--force-download` | re-fetch all assets |
| `--check-only` | run preflight checks for every stage and exit (no SSH uploads, no inference) |
| `--profile-layers` | build with `INFERENCE_PROFILING=ON` and print the per-layer wall-clock stats (also `run.profile_layers`) |
| `--no-cleanup` (deploy script) | keep the remote work_dir for inspection |
| `--plan` (generate script; also `--perf-model`, `--plan-report`, `--pool-budget-mib`) | planning mode ([`doc/plans/TACTICS_PLAN.md`](../../doc/plans/TACTICS_PLAN.md) §9): tactics and issue order from the bitstream's performance model, bit-identical results; or `"plan": true` in the config |
| `--verbose` | also print the cmake / make output of steps that succeed (a failed step always prints its output) |

Each script also accepts `--check-only` on its own, useful for quickly
verifying just one stage:

```bash
.venv/bin/python scripts/deploy_and_run.py --check-only
# Preflight (local)
#     OK   data/t10k-images-idx3-ubyte  7,840,016 B
#     OK   project 'mnist_convnet' on disk
#     OK     driver/xconvkernel.h  (ConvKernel)
#     ...
# Preflight (remote)
#     OK   cmake                             cmake version 3.22.1
#     OK   gcc                               gcc (Ubuntu 11.4.0) …
#     OK   xrt headers                       xrt at /opt/xilinx/xrt
#     OK   uio (VectorOPKernel: fabric_vecop) /dev/uio0
#     ...
```

## Tuning the run

Edit `mnist_config.json`:

* **`run.iters`** — number of test images to evaluate.  `0` means the full
  10 000-image test set.
* **`run.warmup`** — how many inferences to discard before the timed window.
* **`run.use_sudo`** — set to `false` if you SSH in directly as root.
* **`run.env`** — environment variables forwarded to `bench_mnist` on the
  board (e.g. the `INFERENCE_DDR_*` profiler overrides, see
  [`doc/scheduler/PROFILER.md`](../../doc/scheduler/PROFILER.md)).
* **`remote.cmake_args`** — extra `-D…` flags forwarded to the cross-build,
  e.g. `["-DBENCH_INPUT_BIAS=-128"]` if your model expects centred input.

## Troubleshooting

* **`ImportError: paramiko`** — `pip install -r requirements.txt`.
* **`gdown failed`** — Google Drive sometimes throttles.  Run
  `python3 -m gdown --folder <url> -O assets/models/` manually, or download
  the `.onnx` files via a browser and place them in `assets/models/`.
* **`Driver file not found: driver/xconvkernel.h`** during cmake — the HLS
  driver sources weren't on this host.  Either run `make synthesize_kv260`
  in the repo build, or point `local.driver_dirs` at an existing build
  output.
* **`UIO device 'fabric_pool' not found`** — the loaded overlay does not
  expose all four kernels.  Update `remote.uio_devices` to the names that
  appear in `cat /sys/class/uio/uio*/name`.
* **Accuracy near 10 %** — almost always an input encoding mismatch.  Try
  `BENCH_INPUT_BIAS=-128` in `remote.cmake_args` (centred input) or
  re-train / re-export the model with the expected input range.
