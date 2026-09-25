# BERT-SQuAD on the KV260 — plan

Date: 2026-09-26.  Model: `inference-scheduler/bertsquad-12-simplified.onnx`
(ONNX model zoo bertsquad-12, BERT-base uncased fine-tuned on SQuAD 1.1,
opset 12): 12 layers, hidden 768, 12 heads, FFN 3072, sequence 256,
108.7 M parameters.  Inputs `input_ids`, `segment_ids`, `input_mask`
(int64 [1,256]) and `unique_ids_raw_output___9` (int64 [1], passed
through); outputs `unstack:0` / `unstack:1` = start / end logits [1,256].
Status: **phase 1 in progress.**

## 0. Feasibility (measured 2026-09-26)

**Numerics — no blocker.**  `scratchpad/bert/bert_study.py` (to be moved
to `demo/bert_squad/scripts/`) runs the model through a numpy interpreter
(matches onnxruntime to 4e-6) and a bit-level emulation of this project's
datapath: every tensor that lives in DDR is `ap_fixed<16,8>`; MatMul/Gemm
take Q8.8 inputs, accumulate exactly (`ap_fixed<32,16>`) and floor +
saturate on output (AP_TRN, AP_SAT); elementwise Add/Mul likewise; host
regions (embeddings, LayerNorm, GELU, softmax, mask prep) compute in float
and round + saturate on write-back; weights rounded to Q8.8.  60
single-window SQuAD 1.1 dev questions:

| policy | EM | F1 | same span as float |
|---|---:|---:|---:|
| float reference | 93.3 | 96.5 | – |
| **plain Q8.8 (phase-1 partition)** | **93.3** | **97.1** | **59/60** |
| + residual add in the LN region, mask add + scale in softmax, 1/8 folded into W_q | 91.7 | 94.8 | 59/60 |
| + per-weight power-of-two scale (needs an acc-shift register) | 95.0 | 97.1 | 59/60 |
| + softmax output scaled 2^7 | 95.0 | 97.1 | 59/60 |
| float weights, Q8.8 activations | 95.0 | 97.1 | 59/60 |

One example is 1.7 points, so every policy is equivalent to float.  Ranges:
LN outputs ≤ 102, GELU ≤ 109, Q/K/V ≤ 12, P·V ≤ 6.3, FFN-out BiasAdd hits
128 on 0.0001 % of elements, residual sums reach 229 (0.0004 % saturate),
raw QKᵀ reaches 386 (0.1 % saturate before the ×1/8), matmul accumulators
≤ 821 (wrap at 32768).  None of it moves the answers.  What *is* required:
LayerNorm and GELU must run as fused float regions (their intermediates —
x², x³ — saturate Q8.8 if lowered op by op).

**Memory.**  Weights 217 MB at 16 bit (word embeddings 47 MB of it); the
board has 1 GB CMA (1013 MB free) and 4 GB DDR.  Fits.

**Operators.**  Supported today: MatMul, Gemm (→ MatMul + Add), Add, Sub,
Mul, Reshape, Squeeze.  Missing: ReduceMean ×50 / Pow ×12 / Sqrt ×25 /
Reciprocal ×25 / Tanh ×12 (all inside LayerNorm and GELU patterns),
Softmax ×12, Transpose ×49 (perm 0213 ×36, 0231 ×12, 201 ×1), Gather ×1
(word embeddings), OneHot ×1 (token types), Cast ×1, Identity ×1,
Split ×1.  FFN down-projection has K = 3072 > `kMaxK` = 2048.

**Compute.**  22.9 GMAC per inference (linears 21.7, attention 1.2).
MatmulKernel sustains ~2.3 GMAC/s (4.7 GOPS on 256³) → **~10 s per
inference in phase 1**.  ConvKernel's 512-MAC grid does 1×1 convs at
15–40 GMAC/s, so phase 2 moves the linears there.

## 1. Phase 1 — functional BERT (row-major, as the ONNX graph)

| # | Item | Where |
|---|---|---|
| 1a | `kernels.matmul.max_k` 2048 → 4096 (a_buf banks 256 → 512 deep, same BRAM18 count; request-window assert still holds) | JSON, C-sim/RTL fixtures, bitstream |
| 1b | Host-op framework: generalise `SpaceToDepthNode` into host nodes with C implementations — Softmax (last axis), LayerNormalization (fused), Gelu (tanh / erf, fused), Transpose (rank ≤ 5), Gather (axis 0, integer indices), OneHot, Cast, Identity / Squeeze / Unsqueeze (aliases where the layout allows), Split.  Float (double) arithmetic, Q8.8 read / round + saturate write-back, staged through cached host memory (the BOs are mapped non-cacheable). | `src/nodes.py`, `src/codegen/*` |
| 1c | Pattern fusion pre-pass: TF-style LayerNorm (`moments/mean → SquaredDifference → variance → batchnorm/*`) and ONNX `LayerNormalization`; GELU tanh approximation (`x·0.5·(1+tanh(√(2/π)(x+0.044715x³)))`) and erf form and ONNX `Gelu` | `src/graph.py` |
| 1d | Integer tensors (token ids, segment ids, mask): stored as raw int16 in `inference_buf_t` (values must fit; vocab 30522 does), documented in `inference.h` | graph / codegen / harness |
| 1e | `_simulate` covers every new op with the same semantics as the C host code, so generated `test_inference.c` expected outputs stay meaningful | `src/codegen/_simulate.py` |
| 1f | Tiny transformer fixtures (hidden 32–64, 1–2 layers, seq 8–16, both LN/GELU pattern styles) in a new `test/gen_bert_models.py`; unit tests per host op; added to the on-board model suite | tests, `remote_config_all_models.json` |
| 1g | `demo/bert_squad/`: WordPiece tokenizer + SQuAD feature builder (from the study), project generation, deploy-and-run, span decoding and EM/F1 against the emulation | new demo |

**Gates.**  Scheduler simulation of BERT-base reproduces the study's
plain-Q8.8 emulation (logits within a few LSB; identical spans on the
study set); all scheduler tests pass; the 144-model board suite plus the
tiny-BERT cases pass; BERT-base runs on the board with logits matching the
simulation and EM/F1 matching the emulation on the demo set; latency
measured.

## 2. Phase 2 — performance (after phase 1 runs on the board)

Candidates, to be sized with the phase-1 per-layer profile:
- Linears on ConvKernel as 1×1 convs in a feature-major activation layout
  (Xᵀ [768][256]: every Gemm is a 1×1 conv with packed constant weights,
  LayerNorm reduces over channels, per-head Q/K/V slices are contiguous);
  needs `max_in_ch` ≥ 3072 or K-split.  ~21.7 GMAC at 15–40 GMAC/s.
- Attention matmuls without host transposes: row-stride registers and a
  transposed-B mode on MatmulKernel (removes all 48 Transposes).
- Host ops multithreaded (4 × A53) and a cacheable buffer pool.
- MatmulKernel B-stationary 4-row array (128 MACs/cycle) if attention
  stays on it.

## 3. Measured outcome

(to be filled)
