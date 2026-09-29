# LeNet on the KV260 — study and the fully-connected Conv lowering

Status: **done 2026-09-29** — numerics GO as is; LeNet **5.44 → 2.81 ms**
per image on the board (1.94×), accuracy unchanged at 97.35 %.

## 1. Study (2026-09-29, the `model-study` skill, route B)

**Verdict: GO, and the datapath is not what limits LeNet.** Its 97.35 % on
the board is the model's own float accuracy (97.37 %); the gap to the MNIST
convnet is the model.  Where LeNet was weak is speed: 85 % of its time was
one fully-connected layer running as a Conv on ConvKernel (§2).

**Model.** `demo/mnist/assets/models/lenet_simplified.onnx` (the demo's
Google Drive folder; a TensorFlow export — the weights are also listed as
graph inputs), 3.27 M parameters:

| layer | op | shape | weights max \|w\| |
|---|---|---|---|
| conv1first | Conv 5×5 p2 + Relu + MaxPool 2 | 1 → 32, 28×28 → 14×14 | 68.2 |
| conv2 | Conv 5×5 p2 + Relu + MaxPool 2 | 32 → 64, 14×14 → 7×7 | 0.29 |
| conv3 | Conv 7×7 p0 + Relu | 64×7×7 → 1024×1×1 (a fully-connected layer) | 0.31 |
| conv4last | Conv 1×1 | 1024×1×1 → 10×1×1 (logits) | 0.22 |

Operators all supported (Conv 4, Relu 3, MaxPool 2); DMA pool 6.4 MiB.

**Method.** `onnx_study.py` on 200 test images encoded as the board does
(`bench_mnist.c`: pixel p → p / 256), then the task metric on all 10 000
MNIST test images: onnxruntime float32 against the scheduler's simulation
(bit-exact with the board: its 97.35 % equals the board's), 8 processes
(~9 min).

**Results.**

| | float | fixed point (board) |
|---|---|---|
| top-1 accuracy, 10 000 images | 97.37 % | 97.35 % |
| agreement with float | — | 99.91 % (9 images) |
| logits rel. L2 vs float (200 images) | — | 7.1 % mean |

- **Input encoding checked** (float, 3 000 images): p / 256 96.47 %, p / 255
  96.43 %, raw 0..255 96.27 % (the ReLU net is nearly scale-invariant);
  centred or normalised inputs break it (21.8 %, 49.3 %), and so do
  transposed or flipped images (11.8 %, 36 %).  The demo's encoding is right.
- **Value ranges.** conv1's pre-ReLU output reaches −369 (12 % beyond
  Q8.8's ±128) and conv2's −461 (22 %) — harmless, the ReLU removes the
  negative side (post-ReLU drift 0.03 % / 3.5 %).  The **logits** reach
  ±298 and 20 % of them saturate at ±128: that is where the 9
  disagreements and the one argmax tie come from.  A per-tensor exponent on
  the logits would remove them — worth at most 0.02 points, not pursued.
- **Speed** (performance model, `perf_models/kv260/caa67f49a5a3.json`):
  5.44 ms predicted = board 5.44; conv3 alone 4.63 ms (ConvKernel streams
  its 3.2 M weights through one 128-bit port for a single output pixel).

## 2. Fully-connected Convs → MatMul (`src/fc_conv.py`, `--fc-conv`)

A Conv whose kernel covers its whole unpadded input computes one output
pixel per image — a fully-connected layer.  The scheduler now rewrites it
at load time:

```
y = Conv(x[N,C,H,W], W[M,C,H,W], b)   ->   Reshape(MatMul(Flatten(x), W'[C·H·W, M]), [N,M,1,1]) + b
```

`W'[(c·H + h)·W + w][m] = W[m][c][h][w]` is Flatten's order, so the MatMul
reads the same products.  The bias becomes a VectorOP Add after the
Reshape, where a following Relu still fuses (`act`).  One image makes it a
one-row MatMul: MatmulKernel's GEMV path streams the weight through both
read ports.

- **Eligibility:** group 1, dilations 1, all pads 0, the kernel equal to
  the input's H × W, a constant weight (and bias), and C·H·W ≤ `max_k`
  (4096).
- **Decision (`auto`, the default in the library and the CLI):** the
  engine cost model, i.e. the ConvKernel cycle model against the GEMV
  model, or the tiled MatMul where GEMV does not apply (m % 8, m < 64,
  N > 1), plus one VectorOP call for the bias.  It rewrites only when the
  MatMul is estimated ≥ 20 % faster.  `always` / `off` (`--fc-conv`).
- **Numerics:** bit-identical except where the sum before the bias
  saturates.  The Conv adds the bias inside its accumulator and floors
  once, which equals floor(acc / 2⁸) + b_raw, but the MatMul saturates
  first.  On LeNet, 27 logits of 26 of the 10 000 images differ, all of
  them saturated in both paths (float ≥ 127.8), by at most 0.109 (≤ the
  bias).  No prediction changes and the accuracy is identical.
- **Other models:** MobileNet v1's classifier (1×1 on 1×1, 1001 outputs)
  is a candidate. GEMV needs m % 8 == 0 and the tiled MatMul is not
  faster (1.48 vs 1.41 ms estimated), so `auto` keeps the Conv.  ResNet-18,
  MobileNet v2 and the MNIST convnet have none; their projects are
  unchanged.

**Board (2026-09-29, hw_128 d7ce129, 100 MHz, 10 000 images):**

| | before | after |
|---|---|---|
| LeNet mean / p99 | 5.442 / 5.449 ms | **2.810 / 2.816 ms** |
| throughput | 183.8 img/s | **355.9 img/s** |
| accuracy | 97.35 % | 97.35 % |
| predicted (timing simulator) | 5.44 ms | 2.81 ms |

conv3 moved from ConvKernel (4.63 ms) to a GEMV (2.03 ms, priced by the
GEMV-family model, error band ±1 %); conv4last from 0.092 to 0.048 ms (the
tiled MatMul).  What is left: conv1 / conv2 on ConvKernel (0.28 / 0.27 ms),
and conv3's GEMV is bandwidth-bound — 6.4 MB per image at ~3.1 GB/s.

## 3. Commands

```bash
# study (host only)
inference-scheduler/.venv/bin/python .claude/skills/model-study/scripts/onnx_study.py \
    demo/mnist/assets/models/lenet_simplified.onnx --inputs mnist_200.npz
#   mnist_200.npz: {"import/Placeholder:0": t10k images [200, 1, 28, 28] / 256}
# the rewrite on / off
inference-scheduler/.venv/bin/python inference-scheduler/inference_scheduler.py \
    demo/mnist/assets/models/lenet_simplified.onnx [--fc-conv off]
# board
cd demo/mnist && ../../inference-scheduler/.venv/bin/python run_demo.py --skip-download
```

Tests: `inference-scheduler/test/test_fc_conv.py` (the rewrite, eligibility,
`auto` on LeNet's conv3 and on MobileNet's classifier, bit-identical
simulation with and without Relu fusion, the CLI flag).
