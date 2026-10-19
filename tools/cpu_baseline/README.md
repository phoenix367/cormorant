# tools/cpu_baseline — the FPGA against onnxruntime on the board's CPU

`cpu_baseline.py` runs the models of the FPGA results with onnxruntime on
the KV260's four Cortex-A53 cores. It mirrors the FPGA's measurements: the
same inputs, prompts and timing, on the same board. The results are in
[BERT_PLAN §4](../../doc/plans/BERT_PLAN.md) (BERT-SQuAD) and
[CHAT_PLAN §26](../../doc/plans/CHAT_PLAN.md) (SmolLM2-135M / 360M, SmolVLM-256M).

| | BERT-SQuAD | SmolLM2-135M / 360M | SmolVLM-256M |
|---|---|---|---|
| CPU model | `bertsquad-12.onnx` of the ONNX model zoo (fp32, 435 MB): the demo's assets, else downloaded | the export `onnx/model.onnx` at the revision [`llm_models.json`](../../demo/chat/scripts/llm_models.json) pins (fp32), with its head on the last position only (below) | the exports `onnx/vision_encoder.onnx` (with the connector), `embed_tokens.onnx` and `decoder_model_merged.onnx` at the revision [`vlm_study_inputs.json`](../../demo/chat/scripts/vlm_study_inputs.json) pins (fp32), the decoder's head on the last position only |
| int8 copy | onnxruntime's `quantize_dynamic` (`QInt8` weights) | the same, of the last-row model, its tied head's weight first folded into an initializer of its own (below) | the same, of all three; the vision encoder's patch-embedding Conv stays fp32 (onnxruntime's CPU `ConvInteger` takes uint8 weights only) |
| inputs | the BERT demo's `inputs.bin` (50 SQuAD examples, sequence 256) | the chat prompts as `llm_board.py` tokenizes them (factual, summarise, multi-turn, the factual prompt's second turn) | `llm_board.py`'s image prompts: COCO 39769 and 1268, one 512 × 512 image each (the gate's cached pixels), its evaluation questions |
| timing | a warm-up run, then each example's run; 4 threads × 50 examples, 1 thread × 5 | `llm_bench.c`'s protocol: the `<\|im_start\|>` sink cached; each prompt prefilled from position 1, then 32 greedy steps, timed one by one; prefill of 16 / 64 / 256 tokens, best of 3 | the same, each prompt after its image (`llm_image`: the vision encoder, timed); the embedding lookup inside the timed prefill and decode steps |
| scored | EM / F1 against the gold answers (`deploy_and_run.py`'s scoring), spans against fp32's | greedy tokens against fp32's | greedy tokens against fp32's |
| FPGA side | `demo/bert_squad/run_demo.py` → `build/results.json` (read by default) | `demo/chat/scripts/llm_board.py --out JSON` (`--fpga JSON`) | `llm_board.py --project demo/chat/build/llm_project_smolvlm_256m --out JSON` (`--fpga JSON`) |

Every source model's SHA-256 is pinned in `cpu_baseline.py` (`PINS`).

## Run

From the repository root, with the board set up for the BERT demo
(`demo/bert_squad/bert_squad_config.json`: its `ssh`, as `llm_board.py`
uses).  The chat server owns the FPGA and most of the board's 3.9 GB, so stop
it first or pass `--stop-server` (it is restarted afterwards).  Miranda, if
deployed, keeps running; it has a small footprint.

```bash
PY=inference-scheduler/.venv/bin/python

# BERT: the FPGA first (results.json), then the CPU
(cd demo/bert_squad && ../../$PY run_demo.py)
$PY tools/cpu_baseline/cpu_baseline.py bert --stop-server

# SmolLM2: the FPGA library's run (llm_board.py installs it into --remote-dir: keep the chat
# server's own paths out of it), then the CPU
$PY demo/chat/scripts/llm_board.py --rpu r5 --remote-dir /root/cmp_chat \
    --weights-dir /root/cmp_smollm2_weights --out /tmp/fpga_135m.json
$PY tools/cpu_baseline/cpu_baseline.py smollm2 --model 135m --fpga /tmp/fpga_135m.json
$PY demo/chat/scripts/llm_board.py --rpu r5 --project demo/chat/build/llm_project_smollm2_360m \
    --remote-dir /root/cmp_chat --weights-dir /root/cmp_smollm2_360m_weights --out /tmp/fpga_360m.json
$PY tools/cpu_baseline/cpu_baseline.py smollm2 --model 360m --fpga /tmp/fpga_360m.json

# SmolVLM, likewise
$PY demo/chat/scripts/llm_board.py --rpu r5 --project demo/chat/build/llm_project_smolvlm_256m \
    --remote-dir /root/cmp_chat --weights-dir /root/cmp_smolvlm_weights --out /tmp/fpga_vlm.json
$PY tools/cpu_baseline/cpu_baseline.py smolvlm --fpga /tmp/fpga_vlm.json
```

`--rpu r5` has the Cortex-R5 drive the kernels, as the chat server does: it
needs the firmware running (`inference-scheduler/rpu_board.py --start-only
--firmware-name rpu_cmp.elf --bsp-dir BSP_DIR`; `rpu_board.py --stop` after).
Without the firmware, leave `--rpu` out and the A53 drives them.  Afterwards
remove what `llm_board.py` installed (`ssh root@BOARD 'rm -rf /root/cmp_chat
/root/cmp_smollm2_weights /root/cmp_smollm2_360m_weights /root/cmp_smolvlm_weights
/lib/firmware/rpu_cmp.elf'`).

Each run prints a table and writes `tools/cpu_baseline/build/<model>_results.json`
(`--out`):

```
smollm2-135m-instruct: decode ms / token (mean of 128 steps), prefill best of 3
  engine                   threads  decode ms (tok/s)    prefill 16    prefill 64   prefill 256  FPGA speed-up: decode, prefill
  FPGA (fpga_135m.json)          -     48.32 (20.70)         80 ms         97 ms        392 ms
  onnxruntime fp32               4    118.81 ( 8.42)        277 ms        913 ms       3952 ms  2.5x, 3.4x / 9.4x / 10.1x
  onnxruntime int8               4     56.10 (17.83)        176 ms        615 ms       2726 ms  1.2x, 2.2x / 6.3x / 6.9x
  greedy tokens equal to onnxruntime fp32's before the first difference (of 33):
    FPGA                     factual 7, factual/turn2 0, summarise 27, multi-turn 33
    onnxruntime int8 t4      factual 7, factual/turn2 0, summarise 6, multi-turn 5
```

| Option | |
|---|---|
| `--engines fp32,int8` | which models run |
| `--threads 4,1` / `4` | intra-op thread counts (bert: `--n` examples at the first, `--n-rest` at the others) |
| `--stop-server` | stop the chat server for the run, restart it after |
| `--keep` | keep `--remote-dir` (the venv and the models: a re-run uploads nothing) |
| `--remote-dir DIR` | the board directory (default `/root/ort_bench`) |
| `--prepare-only` | only the host part: the models in `build/` |
| `--work DIR` | the host's work directory (default `tools/cpu_baseline/build`): the venv, the models (~6 GB with both SmolLM2s and SmolVLM), the results |
| `--images 39769,1268` | SmolVLM: the COCO images, one prompt each (`llm_board.py --images`) |
| `--config CFG` | the board config (default the BERT demo's) |

## What it does

- **Host.**
  - **Venv:** `build/venv` with onnxruntime 1.23.2, the board's version (the
    newest with an aarch64 wheel for its Python 3.10). The quantizer also
    needs onnx and sympy.
  - **Models:** downloaded and edited into `build/`. `prepare.py` makes
    the int8 and last-row copies.
- **The last-row edit (SmolLM2, SmolVLM's decoder).**  The exports compute logits for every
  prompt position. The FPGA library computes them only for the last one
  (`LlmSelectRow`), and the head is 21 % (135M) / 13 % (360M) of the work per
  token.  `prepare.py last-row` inserts a Slice of the last position in front
  of the head's MatMul. It checks that the result's logits equal the export's
  last row, and fails otherwise.
- **The tied head (SmolLM2, int8).**  SmolLM2's head weight is
  Transpose(the embedding), computed in the graph.  Quantized as it is, the
  embedding becomes int8 for its Gather, and the head's weight becomes
  DequantizeLinear + Transpose of it.  onnxruntime does not fold
  DequantizeLinear, so every step dequantizes and transposes the whole
  embedding and runs the head in fp32 (decode 160 / 280 ms per token,
  slower than fp32).  `prepare.py int8` first folds the transposed weight
  into an initializer of its own, which is exact.  The head then becomes
  an int8 MatMul (56 / 103 ms).
- **Board.**
  - **Venv:** onnxruntime in a venv in `--remote-dir` (`pip` needs PyPI),
    beside the system Python the chat server uses.
  - **Uploads:** `board_bench.py`, the inputs and the models, each only where
    its SHA-256 differs.
  - **Runs:** the page cache is dropped before each run. Each run is in a
    systemd scope capped 300 MB below MemAvailable, so a model that does not
    fit is killed alone. SmolLM2-360M fp32 peaks at 2.3 GB.
  - **Cleanup:** `--remote-dir` is removed unless `--keep`.  The board lock
    (`src/remote/lock.py`) is held throughout.
- **Report.**  The table, the speed-ups against the FPGA's results, and the
  agreement with fp32: BERT's spans, SmolLM2's greedy tokens.  A SmolLM2
  second turn continues after each engine's own first answer, so it is
  comparable only where those answers were the same.
