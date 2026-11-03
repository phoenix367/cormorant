# KV260 demos

End-to-end demos that take an ONNX model (or, for the chat, a Hugging Face
checkpoint), compile it to a C inference project with `inference-scheduler`,
build it on a KV260 board over SSH, and run it on this repo's FPGA kernels
(Conv / Pool / MatMul / VectorOP / RowOps) plus host-CPU ops on the board's
A53 cores.

The CNN demos (`mnist/`, `image_classification/`, `camera/`) follow the same
three stages — **download → generate → deploy** — driven by a one-shot
`run_demo.py` orchestrator and configured by a single `<demo>_config.json`
(copy the bundled `.example` and fill in your board's SSH details).
`bert_squad/` runs **prepare → generate → deploy** the same way, with the
model, vocabulary and SQuAD file downloaded on first use (see its README).
`chat/` installs a long-running server on the board with `deploy.py` and
takes its board settings from `bert_squad/bert_squad_config.json`; `tts/`
builds the speech library the server's `piper` backend loads, `asr/` the
speech-to-text library of its `moonshine` backend.

| Demo | Model | Input | What it shows |
|------|-------|-------|---------------|
| [`mnist/`](mnist/) | MNIST convnet + LeNet | 10 000 MNIST test images | Top-1 accuracy and per-image latency over the full test split (0.115 / 1.232 ms per image) on bitstream `04259a7af917` |
| [`image_classification/`](image_classification/) | MobileNetV1 1.0/224, MobileNetV2, ResNet-18 | static JPG/PNG files | Top-5 ImageNet predictions per image, with latency (ResNet-18 20.0 ms = 50 FPS at 250 MHz, bitstream `04259a7af917`) |
| [`stereo_depth/`](stereo_depth/) | LightStereo-S (OpenStereo, StereoAnything weights; 640 × 480) | Middlebury MiddEval3-Q / ETH3D pairs, your rectified pairs, a RealSense D4xx on the board (`--capture`) | Disparity maps beside the left image: the network on ConvKernel + VectorOPKernel, the correlation / norms / upsampling on the host (469 ms per pair, 2.13 FPS) on bitstream `04259a7af917`; EPE 0.667 px on 42 pairs against float's 0.650 on the same input, maps bit-exact vs the scheduler simulation; RealSense pairs: depth within 10 % of the camera's own on 91 % of the pixels (projector on, the capture of 2026-10-18) |
| [`object_detection/`](object_detection/) | YOLOv5n (Ultralytics v7.0, 640 × 640) | COCO128 images, your JPG/PNG files | Boxes and classes drawn on each image: the network on ConvKernel + VectorOPKernel + PoolingKernel (48 ms per image, 21.0 FPS) on bitstream `04259a7af917`, decode + NMS on the host; COCO128 mAP@0.5:0.95 0.343 against float's 0.349, head maps bit-exact vs the scheduler simulation |
| [`camera/`](camera/) | MobileNetV1 1.0/224 | live Intel RealSense feed | Live classification on the board; annotated frames stream back over SSH with inference latency and whole-board power |
| [`bert_squad/`](bert_squad/) | BERT-base (bertsquad-12) | SQuAD 1.1 dev questions | Extractive QA on ConvKernel + MatmulKernel + VectorOPKernel + RowOpsKernel + host ops: EM / F1 vs the float model, board logits bit-exact vs the scheduler simulation, per-layer time by kind (267 ms per inference) on bitstream `04259a7af917` |
| [`chat/`](chat/) | BERT-base (bertsquad-12), SmolLM2-135M / 360M-Instruct, SmolVLM-256M-Instruct, Gemma-3-270M-it, stories15M, Piper lessac-medium, Moonshine base | chat messages (and images), text to speak, audio to transcribe, over HTTP | OpenAI-compatible server running on the board (`/v1/chat/completions`, `/v1/audio/speech` and `/v1/audio/transcriptions`, streaming), backends `bert-squad`, `smollm2`, `smollm2-360m`, `smolvlm`, `piper` and `moonshine`: question answering over a user-supplied document (~0.27 s per 256-token window on bitstream `04259a7af917`), generative multi-turn chat (SmolLM2-135M ~20.4 tokens/s, 360M ~7.8 tokens/s on bitstream `04259a7af917`), questions about images sent as OpenAI `image_url` parts (1.6 s per image, then ~20 tokens/s on bitstream `04259a7af917`), text to speech (first sound after 1.0–1.5 s at 100 MHz, faster than real time) and speech to text; works with `curl`, the `openai` SDK, `llm`, `aichat` and the bundled `chat.py` (which can read answers aloud); [demo video](https://youtu.be/VVS7ExW0XYQ) |
| [`tts/`](tts/) | Piper en_US-lessac-medium (VITS) | sentences | The numeric study, `libpiper_tts.so` (flow + HiFi-GAN on ConvKernel, 0.15 s per 1.49 s of audio; the text encoder on the FPGA and the duration predictor in C too) and its board gate: audio bit-exact with the specification; the library serves the chat server's `piper` backend |
| [`asr/`](asr/) | Moonshine base / tiny (UsefulSensors) | LibriSpeech test-clean utterances, your WAV files (through the chat server) | The numeric study, `libmoonshine_<size>.so` (the encoder and the greedy decoder on the FPGA, a 10 s recording in 1.3 s through the server) and its board gate: every token id bit-exact with the specification, WER 2.93 % (base); the library serves the chat server's `moonshine` backend |
| [`miranda/`](miranda/) | Moonshine base, SmolLM2-360M-Instruct, Piper lessac-medium (through the chat server) | the microphone (RTP from a console client) | A voice assistant: "Hello Miranda" and a request; voice activity detection on the board's CPU, the key phrase found in the transcripts, the answer streamed as text and spoken back; the microphone muted while Miranda thinks and speaks; a dialogue mode without the key phrase (by voice, `--dialogue` or `d` + Enter) that ends after silence |
| [`vla/`](vla/) | SmolVLA (lerobot/smolvla_base) | 1–3 camera frames, a task, the robot's state | Vision-language-action: 50 actions per chunk (3.96 s from two cameras); the study, the bit-level specification, `libsmolvla.so` (with a ctypes wrapper) and its host-emulation and board gates (actions bit-exact with the specification) |

## Models

Every model the demos run, validated end to end (generate → run on the board → compare
with the Python ground truth).  The CNN links are already simplified with
[`simplify_onnx.py`](../inference-scheduler/simplify_onnx.py); see
[MODEL_PREPARATION.md](../inference-scheduler/doc/MODEL_PREPARATION.md) for your
own models.

| Model | Input | Task |
|---|---|---|
| [ConvMNIST](https://drive.google.com/file/d/1a-A-t2JBC9r5IaEjpWp9915n0wBoIj0y) | `1×1×28×28` | MNIST digit classifier (small convnet) |
| [LeNet](https://drive.google.com/file/d/1tNQe_wDvVzuPnEMJvIzeULMZrUlYS7VF) | `1×1×28×28` | LeNet-5 on MNIST |
| [MobileNet V1](https://drive.google.com/file/d/1PzFSPkXpkIpiKfyo8tORl2AkfXgjdQvw) | `1×3×224×224` | ImageNet classifier (depthwise-separable) |
| [MobileNet V2](https://drive.google.com/file/d/1ti97y2P_Fc8TRUk0oVm_AG7yrmk5Fuw1) | `1×3×224×224` | ImageNet classifier (inverted residuals) |
| [ResNet-18](https://drive.google.com/file/d/1DKyALYam5jAzMSK8ulgFQuSbQ62-EVvr) | `1×3×224×224` | ImageNet classifier (residual blocks) |
| [YOLOv5n](https://github.com/ultralytics/yolov5/releases/tag/v7.0) (Ultralytics v7.0 ONNX; the demo cuts it at its Detect convs) | `1×3×640×640` | object detection, COCO's 80 classes ([demo/object_detection](object_detection/README.md)) |
| [LightStereo-S](https://huggingface.co/XiandaGuo/OpenStereo) (OpenStereo, StereoAnything weights; PyTorch checkpoint, compiled by `src/stereo.py`) | two `1×3×480×640` images | stereo depth: disparity of a rectified pair ([demo/stereo_depth](stereo_depth/README.md)) |
| bertsquad-12 (ONNX model zoo) | 256 tokens | extractive QA ([demo/bert_squad](bert_squad/README.md)) |
| SmolLM2-135M-Instruct (Hugging Face safetensors) | context 1024 | chat ([demo/chat](chat/README.md)) |
| SmolLM2-360M-Instruct (Hugging Face safetensors) | context 1024 | chat ([demo/chat](chat/README.md)) |
| SmolVLM-256M-Instruct (Hugging Face safetensors) | one 512×512 image + text, context 1024 | chat about images ([demo/chat](chat/README.md)) |
| stories15M ([nickypro/tinyllama-15M](https://huggingface.co/nickypro/tinyllama-15M): Karpathy's llama2.c model, Hugging Face safetensors) | text, up to 256 tokens of context | story writing: text completion ([demo/chat](chat/doc/STORIES.md)) |
| Gemma-3-270M-it ([unsloth/gemma-3-270m-it](https://huggingface.co/unsloth/gemma-3-270m-it): Google's Gemma 3, Gemma terms of use; safetensors) | context 1024 | chat ([demo/chat](chat/README.md)) |
| Piper en_US-lessac-medium ([rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices) ONNX) | text, up to 4096 characters (espeak-ng phonemes) | text to speech ([demo/tts](tts/README.md), [demo/chat](chat/doc/TEXT_TO_SPEECH.md)) |
| Moonshine base / tiny ([UsefulSensors/moonshine-base](https://huggingface.co/UsefulSensors/moonshine-base), safetensors) | 16 kHz audio, any length (parts of up to 24.6 s) | speech to text ([demo/asr](asr/scripts/), [demo/chat](chat/doc/SPEECH_TO_TEXT.md)) |
| SmolVLA ([lerobot/smolvla_base](https://huggingface.co/lerobot/smolvla_base), safetensors) | 1–3 camera frames (any size), a task's tokens, the robot's state | robot actions: a chunk of 50 ([demo/vla](vla/README.md)) |

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
  `cmake .. && make synthesize_kv260` in `build/` (the five RTL kernels'
  drivers: `build/kernels/conv_rtl/driver/`, `build/kernels/matmul_rtl/driver/`,
  `build/kernels/vectorop_rtl/driver/`, `build/kernels/pool_rtl/driver/`,
  `build/kernels/rowops_rtl/driver/`) — and point
  `local.driver_dirs` at the result.  The driver sources only describe the
  AXI-Lite registers, so they are the same for every `AXI_BUS_WIDTH`.
- **KV260 board:** Linux with the bitstream and overlay loaded
  (`inference-scheduler/upload_bitstream.py`, see *Quick start* step 5 in
  the [root README](../README.md#quick-start); UIO devices `fabric_vecop` /
  `fabric_matmul` / `fabric_conv` / `fabric_pool` / `fabric_rowops`),
  `gcc` / `cmake` / `make`, the XRT runtime, and passwordless `sudo`.  The
  numbers in the demo READMEs were measured with the 128-bit block design
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
