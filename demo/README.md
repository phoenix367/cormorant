# KV260 demos

End-to-end demos that take an ONNX model, compile it to a self-contained C
inference project with `inference-scheduler`, build it on a KV260 board over
SSH, and run it on this repo's HLS kernels (Conv / Pool / MatMul / VectorOP).

Each demo follows the same three stages — **download → generate → deploy** —
driven by a one-shot `run_demo.py` orchestrator and configured by a single
`<demo>_config.json` (copy the bundled `.example` and fill in your board's
SSH details).

| Demo | Model | Input | What it shows |
|------|-------|-------|---------------|
| [`mnist/`](mnist/) | MNIST convnet + LeNet | 10 000 MNIST test images | Top-1 accuracy and per-image latency over the full test split |
| [`image_classification/`](image_classification/) | MobileNetV1 1.0/224 | static JPG/PNG files | Top-5 ImageNet predictions per image, with latency |
| [`bert_squad/`](bert_squad/) | BERT-base (bertsquad-12) | SQuAD 1.1 dev questions | Extractive QA on MatmulKernel + VectorOPKernel + host ops: EM / F1 vs the float model, board logits bit-exact vs the scheduler simulation, per-layer time by kind (12.1 s per inference) |
| [`chat/`](chat/) | BERT-base (bertsquad-12) | chat messages over HTTP | OpenAI-compatible chat server running on the board (`/v1/chat/completions`, streaming): question answering over a user-supplied document with sliding 256-token windows (~1 s each); works with `curl`, the `openai` SDK, `llm`, `aichat` and the bundled `chat.py` |
| [`camera/`](camera/) | MobileNetV1 1.0/224 | live Intel RealSense feed | Real-time classification; annotated frames stream back over SSH with inference latency and whole-board power |

## Common workflow

```bash
cd demo/<name>
cp <name>_config.json.example <name>_config.json
$EDITOR <name>_config.json          # set ssh.host, key_file, driver paths

python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python run_demo.py
```

`run_demo.py --check-only` runs every stage's preflight (SSH reachability,
kernel UIO devices, board-side dependencies) without uploading anything —
the fastest way to confirm a board is ready.

## Prerequisites (shared)

- **Host:** Python 3.10+ and each demo's `requirements.txt`.
- **HLS drivers:** build the kernel IP from the repo root first —
  `cmake -DAXI_BUS_WIDTH=32 .. && make synthesize_kv260` — and point
  `local.driver_dirs` at the result.
- **KV260 board:** Linux with the cormorant overlay loaded (`/dev/uio*`
  kernels), `gcc` / `cmake` / `make`, the XRT runtime, and passwordless
  `sudo`. The `camera/` demo additionally needs an Intel RealSense camera
  plus `pyrealsense2` / OpenCV on the board — see its README.

See each demo's own `README.md` for model sources, config-field reference,
sample output, and troubleshooting. The generated-project internals are
documented in
[`../inference-scheduler/doc/INFERENCE_SCHEDULER.md`](../inference-scheduler/doc/INFERENCE_SCHEDULER.md)
(canonical scheduler reference; `doc/INFERENCE_SCHEDULER.md` is a thin
pointer to it) and [`../doc/PROFILER.md`](../doc/PROFILER.md).
