# YOLOv5 object detection on the KV260

**Status (2026-10-08):** phases 0–3 done.
- **The study: GO with plain Q8.8.**  On COCO128 the FPGA datapath gives
  mAP@0.5:0.95 0.343 against float's 0.349 (mAP@0.5 0.543 vs 0.545).
- **The demo.**  `demo/object_detection` runs YOLOv5n on the board at 63.9 ms
  per image (15.6 FPS).  The head maps are bit-exact with the scheduler's
  simulation, and the boxes are drawn on the images.  Concat and Resize are
  new host ops.

The user asked to "try to add a YOLO v5 demo for object detection".  The
model is YOLOv5n (Ultralytics v7.0, 640 × 640, COCO's 80 classes), the
smallest YOLOv5 and the usual edge choice; YOLOv5s is the step up if the
board has room.

## 1. Model

`yolov5n.onnx` from the v7.0 release
(`https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.onnx`):
- size and hash: 3 981 910 bytes, SHA-256
  `04f0e55c26f58d17145b36045780fe1250d5bd2187543e11568e5141d05b3262`;
- opset 17, 1 867 405 parameters, input `images` [1, 3, 640, 640] (RGB, 0 … 1);
- license AGPL-3.0, so the assets are downloaded by the demo, never
  committed.

The export ends in the Detect layer's decode, `output0` [1, 25200, 85]: Split
/ Sigmoid / Pow / grid and anchor arithmetic.  The FPGA graph stops at the
three head convs (`/model.24/m.{0,1,2}/Conv`, [1, 255, 80 / 40 / 20, …]).
The decode (sigmoid, grid, anchors) and NMS run on the host, as detection
post-processing usually does.

| part | ops (the cut graph, 201 nodes) | where |
|---|---|---|
| backbone + neck | Conv 60 (BN folded), SiLU 57 (Sigmoid · Mul), Add 7 (bottleneck residuals) | ConvKernel; VectorOPKernel (SiLU on the activation unit, Add) |
| SPPF | MaxPool 3 (5 × 5, stride 1, pad 2) | PoolingKernel (within its 7 × 7) |
| C3 / neck joins | Concat 13 (channels) | **host** (new op) |
| neck upsampling | Resize 2 (nearest × 2) | **host** (new op) |
| stem | Conv 6 × 6 stride 2 on 3 channels | the scheduler's stride-2 stem rewrite: SpaceToDepth (host) + Conv 4 × 4 |

## 2. Phases

| phase | work | done when |
|---|---|---|
| 0 study | census and coverage; Concat and Resize as host ops (the enablers); memory; numerics on COCO128 (mAP@0.5 and mAP@0.5:0.95 against its labels: float, float on the Q8.8 input, the scheduler's simulation); latency projection | GO / NO-GO |
| 1 scheduler | tests of Concat / Resize (against onnxruntime), docs; zero-copy Concat if the profile asks | suite passes |
| 2 demo | `demo/object_detection`: assets (the pinned model, COCO128), letterbox preprocessing → Q8.8, project generation, a board runner (inference + decode + NMS in C, timings), host side (boxes drawn, mAP), `run_demo.py`, bit-exact check board vs simulation | runs end to end on the host emulation |
| 3 board | the demo on the KV260: latency, FPS, mAP against float, profile | bit-exact, numbers recorded |
| 4 docs | README, demo README, facts | — |

## 3. Results

### 3.1 Phase 0: census, coverage, memory (2026-10-08)

`onnx_study.py` on the cut graph: Concat × 13 and Resize × 2 were MISSING (no
kernel or host op).  Both are pure data movement, so they don't touch the
numerics, and they are now host ops (`src/host_nodes.py`):
- **`ConcatNode`**: block copies (`host_concat`).
- **`ResizeNode`**: nearest by integer scales, a `host_copy_nd` with zero
  strides.

With them the graph schedules: ConvKernel 60, VectorOPKernel 64 (SiLU 57,
Add 7), PoolingKernel 3, host 16 (Concat 13, Resize 2, the stem's
SpaceToDepth).  The DMA pool is 14.5 MiB (weights 3.6, intermediates 10.9):
1.5 % of CMA.

One random sample (std 1.0, not the verdict): the three head maps are within
2.4–2.6 % relative L2 of onnxruntime, and no tensor saturates.

The prediction is 35 ms per image (ConvKernel 30 ms), from family models
only (±51 %).  79 nodes are unpriced (SiLU calls, the host copies), so it is
a floor.

### 3.2 Phase 0: numerics on COCO128 (2026-10-08)

**Verdict: GO with plain Q8.8.**  The FPGA graph loses 0.006 mAP@0.5:0.95
(0.003 mAP@0.5) against float; no per-tensor exponents are needed.

**The model in float32.**  The v7.0 `yolov5n.onnx` is a float16 graph.
`yolo_study.py fetch` cuts it at the Detect convs and widens it to float32,
which is exact.  The result is `yolov5n_raw.onnx`, SHA-256
`3cde25c3d36f9f04dd17d95c4656a77ae353d7ef59c987307c87a36041198c11`.

**The host side.**  `demo/object_detection/scripts/yolo_post.py` does the
letterbox, the Detect decode, class-aware NMS and val.py's mAP (101-point
interpolation, IoU 0.5 : 0.95).  `yolo_study.py validate` checks the decode
of the cut graph's head maps against the export's own `output0`: within
4.4·10⁻⁴ relative, which is float16 against float32.

**The data.**  COCO128 (Ultralytics' assets v0.0.0, `coco128.zip` SHA-256
`61e5e302…aeb8e`): 128 images, 929 labels in 71 classes.  These are COCO
train images, so the absolute mAP is optimistic; the comparison between
policies is what counts.  NMS at val.py's settings (conf 0.001, IoU 0.6,
multi-label, 300 detections).

| policy | mAP@0.5 | mAP@0.5:0.95 | detections |
|---|---:|---:|---:|
| float (onnxruntime, float32) | 0.5454 | 0.3486 | 28 141 |
| q88-in: float on the input rounded to Q8.8 | 0.5459 | 0.3486 | 28 161 |
| **fpga: the scheduler's simulation** (the board's numbers) | **0.5427** | **0.3429** | 27 151 |

- **Head maps.**  The FPGA's are within 2.50 % relative L2 of float (mean
  over 384 maps; max 5.12 %).
- **The input.**  The Q8.8 input costs nothing; the loss is the datapath's
  (int16 weights at 2⁻⁸, int16 activations, floor(acc / 2⁸)).
- **At the display threshold (conf 0.25).**  Float finds 561 boxes and the
  FPGA 532, of which 510 match a float box (same class, IoU ≥ 0.5): 91 % of
  float's boxes and 96 % of the FPGA's.
- **Saturation.**  The random-input census found none (3.1).

**Cost.**  The study's 35 ms is a floor: 79 nodes are unpriced and
ConvKernel is priced by family models (±51 %).  Counted by volume:
- ConvKernel about 30 ms;
- the VectorOP SiLU / Add passes 13.0 M elements, about 52 MB of DDR
  traffic, roughly 8 ms;
- the host copies 13.7 MB: Concat 8.4, the stem's SpaceToDepth 2.5,
  Resize 1.2;
- so about 45–60 ms per image (17–22 FPS), before pre- and post-processing.
  The board measures it in phase 3.

**Levers, if the board asks.**
- A zero-copy Concat: the producers write into the joined buffer
  (sub-buffer views, as Split / Slice already have).
- SiLU fused into the producing conv: ConvKernel has no activation stage
  today.

**What the implementation needs.**
- Concat and Resize, now host ops with tests against onnxruntime.
- The demo: assets, preprocessing, a board runner, host post-processing,
  orchestration.
- No new kernel, frontend or exponent metadata; CMA 14.5 MiB.

Commands:

```bash
PY=inference-scheduler/.venv/bin/python
$PY demo/object_detection/scripts/yolo_study.py fetch       # model + COCO128 (SHA-256 checked), the cut graph (2 s)
$PY demo/object_detection/scripts/yolo_study.py validate    # decode vs the export (4 images, 3 s)
$PY demo/object_detection/scripts/yolo_study.py study       # 128 images, 3 policies (11 min, 6 workers)
.claude/skills/model-study/scripts/onnx_study.py demo/object_detection/assets/models/yolov5n_raw.onnx   # census, memory
```

### 3.3 Phases 2–3: the demo on the board (2026-10-08)

**`demo/object_detection`** (README there):
- `run_demo.py` runs the five steps: fetch, prepare, generate, board,
  results.
- `scripts/` holds `yolo_post.py`, `yolo_study.py`, `prepare.py`,
  `generate_project.py`, `deploy_and_run.py` and `postprocess.py`.
- `src/detect_images.c` is the board host: `inference_run` per image, the
  three raw head maps to `heads.bin`, per-image latency, optional per-layer
  times.
- The project comes from `inference-scheduler` with the image demos' options
  (activations fused, the stride-2 stem rewrite).
- The chat server owns the FPGA, so `--stop-server` stops it for the run and
  restarts it.

The first run, on 16 COCO128 images with bitstream `8599aa7a5f12`, worked
first time (4 min 17 s with the build and the chat-server restart):

| | result |
|---|---|
| latency (`inference_run`, the FPGA graph) | **63.9 ms mean, 63.9 p50 (15.6 FPS)** |
| head maps vs the scheduler's simulation (2 images) | **bit-exact** |
| mAP on those 16 images: board / float | 0.771 / 0.762 @0.5, 0.540 / 0.552 @0.5:0.95 |
| detections at conf 0.25 | 41 (e.g. person 0.73 + umbrella 0.26 on `000000000036`) |

The board equals the simulation bit for bit, so on all of COCO128 it gives
the study's FPGA numbers (§3.2).

**Profile** (`deploy_and_run.py --profile`: the per-layer mean; lanes
overlap, so the parts add up to 73.3 ms against the 63.9 ms total):

| part | layers | ms | share |
|---|---:|---:|---:|
| ConvKernel | 60 | 45.1 | 62 % |
| VectorOP SiLU | 57 | 12.5 | 17 % |
| SpaceToDepth (the stem rewrite, host) | 1 | 4.9 | 7 % |
| Concat (host) | 13 | 4.2 | 6 % |
| VectorOP Add | 7 | 3.7 | 5 % |
| Resize (host) | 2 | 1.8 | 2 % |
| PoolingKernel | 3 | 1.0 | 1 % |

**Levers**, if the demo should go faster:
- **SiLU's own pass (12.5 ms).**  Fusing it into the conv's write-back
  needs an activation stage in ConvKernel: a bitstream change.
- **The host copies (10.9 ms).**
  - A zero-copy Concat: the producers write into the joined buffer.
  - The stem without the SpaceToDepth rewrite (`--no-s2d-stem`): measure
    it; the rewrite was chosen for ResNet's 7 × 7 stem.
- **A board-side decode + NMS in C.**  Needed for a live camera mode
  (`demo/camera`); the still-image demo does it on the host.
- **YOLOv5s.**  It is about 4× the MACs of YOLOv5n.

**Scheduler.**  `ConcatNode` and `ResizeNode` (`src/host_nodes.py`, helper
`host_concat`) are in the supported-op tables (CLAUDE.md, the scheduler
reference, the user guide).  Their tests in `test_host_ops.py`:
- the C equals the reference, bitwise, on 1 / 3 / 4 threads;
- Concat on four axis / input-count combinations;
- Resize in every accepted coordinate mode, against onnxruntime;
- bilinear and downscaling are rejected.
