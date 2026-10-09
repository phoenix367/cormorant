# KV260 demos

End-to-end demos that take an ONNX model (or, for the chat, a Hugging Face
checkpoint), compile it to a C inference project with `inference-scheduler`,
build it on a KV260 board over SSH, and run it on this repo's FPGA kernels
(Conv / Pool / MatMul / VectorOP) plus host-CPU ops on the board's A53 cores.

The CNN demos (`mnist/`, `image_classification/`, `camera/`) follow the same
three stages — **download → generate → deploy** — driven by a one-shot
`run_demo.py` orchestrator and configured by a single `<demo>_config.json`
(copy the bundled `.example` and fill in your board's SSH details).
`bert_squad/` runs **prepare → generate → deploy** the same way, with the
model, vocabulary and SQuAD file downloaded on first use (see its README).
`chat/` installs a long-running server on the board with `deploy.py` and
takes its board settings from `bert_squad/bert_squad_config.json`; `tts/`
builds the speech library the server's `piper` backend loads.

| Demo | Model | Input | What it shows |
|------|-------|-------|---------------|
| [`mnist/`](mnist/) | MNIST convnet + LeNet | 10 000 MNIST test images | Top-1 accuracy and per-image latency over the full test split (0.111 / 1.249 ms per image) |
| [`image_classification/`](image_classification/) | MobileNetV1 1.0/224, MobileNetV2, ResNet-18 | static JPG/PNG files | Top-5 ImageNet predictions per image, with latency (ResNet-18 20.2 ms = 49.6 FPS at 250 MHz) |
| [`stereo_depth/`](stereo_depth/) | LightStereo-S (OpenStereo, StereoAnything weights; 640 × 480) | Middlebury MiddEval3-Q / ETH3D pairs, your rectified pairs, a RealSense D4xx on the board (`--capture`) | Disparity maps beside the left image: the network on ConvKernel + VectorOPKernel, the correlation / norms / upsampling on the host (659 ms per pair, 1.52 FPS); EPE 0.667 px on 42 pairs against float's 0.650 on the same input, maps bit-exact vs the scheduler simulation; RealSense pairs: depth within 10 % of the camera's own on 98 % of the pixels (projector on) |
| [`object_detection/`](object_detection/) | YOLOv5n (Ultralytics v7.0, 640 × 640) | COCO128 images, your JPG/PNG files | Boxes and classes drawn on each image: the network on ConvKernel + VectorOPKernel + PoolingKernel (64 ms per image, 15.6 FPS), decode + NMS on the host; COCO128 mAP@0.5:0.95 0.343 against float's 0.349, head maps bit-exact vs the scheduler simulation |
| [`camera/`](camera/) | MobileNetV1 1.0/224 | live Intel RealSense feed | Live classification on the board; annotated frames stream back over SSH with inference latency and whole-board power |
| [`bert_squad/`](bert_squad/) | BERT-base (bertsquad-12) | SQuAD 1.1 dev questions | Extractive QA on ConvKernel + MatmulKernel + VectorOPKernel + host ops: EM / F1 vs the float model, board logits bit-exact vs the scheduler simulation, per-layer time by kind (418 ms per inference) |
| [`chat/`](chat/) | BERT-base (bertsquad-12), SmolLM2-135M / 360M-Instruct, SmolVLM-256M-Instruct, Piper lessac-medium | chat messages (and images), text to speak, over HTTP | OpenAI-compatible server running on the board (`/v1/chat/completions` and `/v1/audio/speech`, streaming), backends `bert-squad`, `smollm2`, `smollm2-360m`, `smolvlm` and `piper`: question answering over a user-supplied document (~1 s per 256-token window), generative multi-turn chat (SmolLM2-135M ~18.3 tokens/s, 360M ~7.3 tokens/s), questions about images sent as OpenAI `image_url` parts (3.9 s per image, then ~9.5 tokens/s) and text to speech (first sound after 1.0–1.5 s, faster than real time); works with `curl`, the `openai` SDK, `llm`, `aichat` and the bundled `chat.py` (which can read answers aloud); [demo video](https://youtu.be/VVS7ExW0XYQ) |
| [`tts/`](tts/) | Piper en_US-lessac-medium (VITS) | sentences | The numeric study, `libpiper_tts.so` (flow + HiFi-GAN on ConvKernel, 0.7 s per 1.49 s of audio; the text encoder on the FPGA and the duration predictor in C too) and its board gate: audio bit-exact with the specification; the library serves the chat server's `piper` backend |

## Common workflow

CNN demos:

```bash
cd demo/<name>
cp <name>_config.json.example <name>_config.json
$EDITOR <name>_config.json          # set ssh.host, key_file, driver paths

python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python run_demo.py
```

`bert_squad/` and `chat/` have no `requirements.txt`: they run with the
scheduler's virtualenv, `../../inference-scheduler/.venv/bin/python`.

`run_demo.py --check-only` runs the preflight checks (SSH reachability,
kernel UIO devices, board-side dependencies) without uploading anything —
the fastest way to confirm a board is ready (`deploy.py --check-only` for
`chat/`).

Planning (optional, [TACTICS_PLAN §9](../doc/plans/TACTICS_PLAN.md)): set
`"plan": true` in `<demo>_config.json`, or pass `--plan` to
`scripts/generate_project.py`, to generate the project from the bitstream's
performance model; for `chat/` use `deploy.py --regenerate --plan` (BERT) and
`scripts/generate_llm_project.py --plan` (the LLM libraries).

## Prerequisites (shared)

- **Host:** Python 3.10+ and each demo's `requirements.txt` (or the
  scheduler's `.venv`, see above).
- **Kernel drivers:** build the kernel IP from the repo root first —
  `cmake .. && make synthesize_kv260` in `build/` (the RTL MatmulKernel's,
  VectorOPKernel's and PoolingKernel's drivers: `build/kernels/matmul_rtl/driver/`,
  `build/kernels/vectorop_rtl/driver/`, `build/kernels/pool_rtl/driver/`) — and point
  `local.driver_dirs` at the result.  The driver sources only describe the
  AXI-Lite registers, so they are the same for every `AXI_BUS_WIDTH`.
- **KV260 board:** Linux with the bitstream and overlay loaded
  (`inference-scheduler/upload_bitstream.py`, see *Quick start* step 5 in
  the [root README](../README.md#quick-start); UIO devices `fabric_vecop` /
  `fabric_matmul` / `fabric_conv` / `fabric_pool`), `gcc` / `cmake` /
  `make`, the XRT runtime, and passwordless `sudo`.  The numbers in the
  demo READMEs were measured with the 128-bit block design
  (`hw/cormorant_hw_128`) at 100 MHz, the dated transcripts; since
  FMAX_250_PLAN the kernels run at 250 MHz (the current numbers are in the
  root README's results).  The
  `camera/` demo and `stereo_depth/`'s `--capture` additionally need an
  Intel RealSense camera plus `pyrealsense2` / OpenCV on the board (the
  stereo pair over USB 2: firmware 5.17.3.10) — see their READMEs.

See each demo's own `README.md` for model sources, config-field reference,
sample output, and troubleshooting. The generated-project internals are
documented in [`../doc/scheduler/INFERENCE_SCHEDULER.md`](../doc/scheduler/INFERENCE_SCHEDULER.md)
(scheduler technical reference),
[`../inference-scheduler/doc/USER_GUIDE.md`](../inference-scheduler/doc/USER_GUIDE.md)
(CLI, generated project layout and C API) and
[`../doc/scheduler/PROFILER.md`](../doc/scheduler/PROFILER.md).
