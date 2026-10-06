# Chat about images — SmolVLM

`smolvlm-256m-instruct` answers questions about images (CHAT_PLAN §22,
§23).  It is SmolVLM-256M-Instruct: a SigLIP-style vision encoder, an
Idefics3 connector and a small text model.  All three run on the FPGA
kernels and the board's A53 cores, from one library, `libsmolvlm_256m.so`.

- **Timing.**  The vision encoder takes 2.3 s per image at 250 MHz (3.4 s at 100 MHz, 3.9 s before the RTL ConvKernel: CHAT_PLAN §24, CONV_RTL_PLAN, FMAX_250_PLAN),
  then text comes at ~18 tokens/s.  A follow-up question about the same
  image reuses its encoding, so its first token comes after ~0.4 s.
- **Images.**  Each image becomes one 512 × 512 tile (67 prompt tokens).
- **Delivery.**  Images are sent as OpenAI `image_url` parts with base64
  data URLs; the server fetches nothing from the network.
- **Text-only models** answer an image with 400 `images_not_supported`.

## Install

```bash
cd demo/chat
../../.venv-export/bin/python scripts/vlm_study.py fetch     # checkpoint + COCO images (pinned hashes)
../../.venv-export/bin/python scripts/vlm_study.py formats   # text + vision exponents (~3 min)
$PY scripts/generate_llm_project.py --model-name smolvlm-256m-instruct
                                          # -> build/llm_project_smolvlm_256m (~90 s, 3.3 GiB RAM)
$PY deploy.py --stop                      # the server owns the FPGA
$PY scripts/llm_board.py --project build/llm_project_smolvlm_256m --install-only
                                          # -> <dir>/lib/libsmolvlm_256m.so, /root/smolvlm_256m_weights
                                          # (--remote-dir / --weights-dir as for SmolLM2)
```

`.venv-export` is the study environment described in
[Generative chat](SMOLLM2.md#install-smollm2-135m); `$PY` is the
scheduler's venv, `../../inference-scheduler/.venv/bin/python`.

**Enable it:**
1. Add `"smolvlm"` to `server.backends` in `chat_config.json`.  The
   `smolvlm` block sets the library, the tokenizer and `cma_mb` (540).  The
   `smollm2` block's sampling defaults apply.
2. The board needs `python3-pil` (Pillow) to decode images.
3. Run `deploy.py`.

**The board gate.**  To check the library on the board — image prompts,
logits against the simulation — run the same command without
`--install-only`:
`$PY scripts/llm_board.py --project build/llm_project_smolvlm_256m`.

## Sending images

**With `chat.py`**, `--image` attaches a picture to the first question, and
`/image FILE` in the REPL to the next message:

```bash
python3 chat.py --url http://<board>:8000/v1 --model smolvlm-256m-instruct --image photo.jpg \
    -q "What is in this image?"
python3 chat.py --url http://<board>:8000/v1 --model smolvlm-256m-instruct   # then: /image photo.jpg, and ask
```

**With any OpenAI client**, put an `image_url` part next to the text part
of a user message:

```json
{"model": "smolvlm-256m-instruct",
 "messages": [{"role": "user", "content": [
     {"type": "text", "text": "What is in this image?"},
     {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}]}]}
```

The request body may be up to 32 MB when an image model is served
(`--max-body-mb`).

## Memory and other models

The SmolVLM pool is 495 MiB (`cma_mb` 540 with the image buffer).  Under
`--resident auto`:
- **Beside BERT:** it stays loaded.
- **Beside SmolLM2-135M:** only when the CMA is not fragmented.  Otherwise
  its load fails and is retried after evicting SmolLM2-135M.
- **With SmolLM2-360M:** they swap.

So the board serves it next to both SmolLM2 sizes and BERT (CHAT_PLAN §23),
loading it on demand when needed.

See [Several models on one FPGA](DEPLOY.md#several-models-on-one-fpga).

## How images are handled

- **Template.**  `idefics3.py` renders SmolVLM's chat template with the
  image blocks.
- **Pixels.**  `vlm_image.py` turns a data URL into 512 × 512 pixels with
  Pillow's LANCZOS, the same pixels as the numeric study's.
- **Encoding.**  Before each new image block the backend calls the
  library's `llm_image()`, which runs the vision encoder and keeps the
  image features in the library.  In the prompt, the image's positions use
  token ids past the end of the vocabulary (vocab + k for the k-th row).
  The text model reads the k-th row of image features for them, instead
  of a word embedding.
- **The prefix cache** is keyed by the image content, so a follow-up
  question about the same image skips the vision encoder.
