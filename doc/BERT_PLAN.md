# BERT-SQuAD on the KV260 — plan

Date: 2026-09-26.  Model: `inference-scheduler/bertsquad-12-simplified.onnx`
(ONNX model zoo bertsquad-12, BERT-base uncased fine-tuned on SQuAD 1.1,
opset 12): 12 layers, hidden 768, 12 heads, FFN 3072, sequence 256,
108.7 M parameters.  Inputs `input_ids`, `segment_ids`, `input_mask`
(int64 [1,256]) and `unique_ids_raw_output___9` (int64 [1], passed
through); outputs `unstack:0` / `unstack:1` = start / end logits [1,256].
Status: **phase 1 done and merged to main (2026-09-26)** — scheduler side
(1b–1f), demo (1g) and the `max_k` 4096 bitstream (1a); BERT-base
runs on the KV260 in **12.13 s per inference**, logits bit-exact with the
scheduler simulation, EM / F1 equal to the float model on the demo set (§3).
Phase 2 (performance) in progress — 2A conv lowering, 2B host ops (§2).

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
| 1b | Host-op framework: generalise `SpaceToDepthNode` into host nodes with C implementations — Softmax (last axis), LayerNormalization (fused), Gelu (tanh / erf, fused), Transpose (rank ≤ 5), Gather (axis 0, integer indices), OneHot, Cast, Identity / Squeeze / Unsqueeze (aliases where the layout allows), Split.  Float (double) arithmetic, Q8.8 read / round + saturate write-back, staged through cached host memory (the BOs are mapped non-cacheable). | `src/host_nodes.py`, `src/codegen/*` — **done** (§3) |
| 1c | Pattern fusion pre-pass: TF-style LayerNorm (`moments/mean → SquaredDifference → variance → batchnorm/*`) and ONNX `LayerNormalization`; GELU tanh approximation (`x·0.5·(1+tanh(√(2/π)(x+0.044715x³)))`) and erf form and ONNX `Gelu` | `src/fusion.py` — **done** (§3) |
| 1d | Integer tensors (token ids, segment ids, mask): stored as raw int16 in `inference_buf_t` (values must fit; vocab 30522 does), documented in `inference.h` | graph / codegen / harness — **done** (§3) |
| 1e | `_simulate` covers every new op with the same semantics as the C host code, so generated `test_inference.c` expected outputs stay meaningful | `src/codegen/_simulate.py` — **done** (§3) |
| 1f | Tiny transformer fixtures (hidden 32–64, 1–2 layers, seq 8–16, both LN/GELU pattern styles) in a new `test/gen_bert_models.py`; unit tests per host op; added to the on-board model suite | tests, `remote_config_all_models.json` — **done** except the on-board suite entry (§3) |
| 1g | `demo/bert_squad/`: WordPiece tokenizer + SQuAD feature builder (from the study), project generation, deploy-and-run, span decoding and EM/F1 against the emulation | new demo — **done** (§3) |

**Gates.**  Scheduler simulation of BERT-base reproduces the study's
plain-Q8.8 emulation (logits within a few LSB; identical spans on the
study set); all scheduler tests pass; the 144-model board suite plus the
tiny-BERT cases pass; BERT-base runs on the board with logits matching the
simulation and EM/F1 matching the emulation on the demo set; latency
measured.

## 2. Phase 2 — performance (started 2026-09-26)

Phase-1 board breakdown (12.13 s): MatMul linears 7.66 s, host GELU 2.05 s,
host softmax 1.07 s, attention MatMuls 0.75 s, LayerNorm 0.34 s, host
transposes 0.17 s, VectorOP 0.10 s.  Host ops move 127.5 MiB of BO data per
inference through a non-cacheable mapping (~0.1–0.2 GB/s).

**2A — MatMul on ConvKernel with swapped operand roles.**  For
C[N][M] = A[N][K]·B[K][M]: conv `out_ch := N` (tokens), `in_ch := K/kw`,
kernel 1×kw, stride (1, kw), output spatial := M (out_h × out_w), **conv
weights := A** (row-major [N][K] *is* the packed tile-major
`[N][ict][1][kw][16]` layout when K is a multiple of 16·kw), **conv input
:= B** arranged `[K/kw][out_h][kw·out_w]` (free at codegen for constant B;
kw = 1 is B's natural row-major layout, used for activation×activation
attention).  Output = C row-major.  Activations stay row-major as in the
ONNX graph; the Gemm bias stays a VectorOP Add; both kernels accumulate
exactly in `ap_fixed<32,16>` and floor + saturate the same way, so results
stay **bit-identical** with phase 1.  Batch-1 FC layers (N = 1) stay on
MatmulKernel.  Cycle model (§2.42 kernel), per call, converted at the board's
100 MHz (the model prints ms at the platform JSON's 150 MHz — divide its ms by 1.5):

| layer | mapping | model | MatmulKernel today |
|---|---|---:|---:|
| Q/K/V/out 768→768 | C=384 M=256, 1×2 s2, out 12×64 | 3.84 ms (79 % MAC) | ~53 ms |
| FFN up 768→3072 | C=384 M=256, out 48×64 | 15.3 ms | ~212 ms |
| FFN down 3072→768 | C=1024 M=256, 1×3 s3 | 17.0 ms (C=1536 1×2: 14.2 ms, needs max_in_ch 2048) | ~212 ms |
| QKᵀ per head | C=64 M=256, 1×1, out 4×64 | 0.32 ms | ~2.6 ms |
| P·V per head | C=256 M=256, 1×1, out 1×64 | 0.23 ms | ~1.5 ms |

→ linears ≈ 0.57 s, attention ≈ 0.08 s per inference (from 8.4 s).

**2B — host ops.**  Cacheable buffer pool (the generated code already
syncs at every CPU↔kernel hand-off); 65 536-entry GELU table and a Softmax
`exp` table indexed by the 16-bit input / the exact 1/256-grid argument
(filled at init with the same double formula → bit-identical); rows split
over the 4 A53 cores (per-row arithmetic unchanged → bit-identical).

**Later:** row-stride / transposed-B MatmulKernel modes or conv-friendly
transposes (host transposes are ~0.17 s now, less once memory is
cacheable); GELU fused into the conv drain; KV-cache / step graphs for
autoregressive decoders (a different, bandwidth-bound problem).

## 3. Measured outcome

### Phase 1 scheduler (1b–1f) — implemented 2026-09-25, branch `feat/bert-sched`

`inference_scheduler.py bertsquad-12-simplified.onnx` generates the project
in 33 s (3.2 GB peak RSS).  After Gemm decomposition (73), Split
lowering, fusion (25 LayerNorm, 12 GELU) and constant-broadcast
normalisation (15 operands: the
twelve ×1/8 scalars, `1 − mask`, `× −10000` → `[256]` vectors, `ones[1,256,1]`
→ `[1,256,256]`) the 671-node graph schedules as 386 nodes:

| Engine | Nodes | Work per inference |
|---|---:|---|
| MatmulKernel | 98 | 73 linears (constant packed B, K ≤ 3072): 21.74 GMAC; 24 attention MatMuls (QKᵀ 256×64·64×256 and P·V 256×256·256×64, batch 12): 1.21 GMAC; OneHot · token-type (K = 2) |
| VectorOPKernel | 126 | 73 Gemm bias Adds, 24 residual + 2 embedding Adds, 12 mask Adds (65 536-element chunk × 12 heads), 12 × 1/8 scale Muls, mask `ones·Cast` Mul, `1 − ·` Sub, `× −10000` Mul: 45.4 M element ops |
| host CPU | 103 | 25 LayerNorm (TF form), 12 GELU-tanh, 12 Softmax, 49 Transpose (36 × 0213, 12 × 0231, 1 × 201), Gather (word embeddings), OneHot (segments), Cast (mask), 2 Slice (Split): 33.4 M elements, 127.5 MiB of BO copies |
| zero-cost | 59 | 56 Reshape, 2 Squeeze, 1 Identity (`unique_ids`, an input → output copy) |

**Gate (a) — passed.**  Pool BO 215.3 MiB (153 weight tensors, 108.7 M
parameters = 207.4 MiB: 76 external `weights/*.dat` + 77 inline;
383 intermediates in 6 pool slots = 7.9 MiB),
host staging arena 3 MiB (malloc), `inference.c` 1.9 MB (the LayerNorm
γ / β are float32 C arrays).  `inference.c` and `test_inference.c` compile
with `-Wall -Wextra -Werror`, and the generated `test_inference.c` passes
on the host against software models of the two kernels
(`test/host_emu.py`, `test/test_bert_base.py`).  API:

```c
int  inference_init(const char *vectoropkernel_instance, const char *matmulkernel_instance);
void inference_run(inference_buf_t *unique_ids_raw_output_9_0,  /* int64 [1]      raw int16 */
                   inference_buf_t *segment_ids_0,              /* int64 [1, 256] raw int16 */
                   inference_buf_t *input_mask_0,               /* int64 [1, 256] raw int16 */
                   inference_buf_t *input_ids_0,                /* int64 [1, 256] raw int16 */
                   inference_buf_t *unstack_1,                  /* end logits   [1, 256] Q8.8 */
                   inference_buf_t *unstack_0,                  /* start logits [1, 256] Q8.8 */
                   inference_buf_t *unique_ids_0);              /* int64 [1]      raw int16 */
```

**Gate (b) — passed.**  `demo/bert_squad/scripts/bert_sched_check.py`: the
scheduler's `_simulate` start / end logits equal the study's independent
op-by-op emulation of the same partition (`bert_study.py` policy `sched`)
bit for bit on 20 / 20 SQuAD examples (`--n 20`; also 5 / 5 with
`--n 5`) — and so does every one of the 313 DDR tensors both sides
materialise.  Study, 60 single-window dev questions:

| policy | EM | F1 | same span as float | mean max \|logit err\| |
|---|---:|---:|---:|---:|
| float reference | 93.3 | 96.5 | – | – |
| `q88` (§0 plain Q8.8 partition) | 93.3 | 97.1 | 59/60 | 1.289 |
| **`sched` (what the scheduler generates)** | **93.3** | **97.1** | **59/60** | 1.298 |

`sched` differs from `q88` only where the scheduler put work on the
kernels that §0 kept in float host regions — the embedding adds and
OneHot · token-type MatMul (with the word-embedding table read as Q8.8),
the mask preparation — and it rounds host write-backs half to even.  The
one example whose span moves (#52) moves under both policies.

**Gate (c) — estimate** (`bert_schedule_stats.py`): 98 MatmulKernel calls,
22.95 GMAC → ~10.0 s at 2.3 GMAC/s; 126 VectorOP calls, 45 M element ops
→ ~0.06 s; host ops ~0.7 s single-threaded (placeholder rates) → **~10.8 s
per inference** before any phase-2 work.

**Board steps left for phase 1** (all done 2026-09-26, next subsection)**:** bitstream with `max_k` 4096 (1a); a demo
runner (1g) that fills the four integer inputs as raw int16
(`inference_buf_ptr(input_ids_0)[i] = (Data_t)(int16_t)id`) and reads the
logits as `(int16_t)bits / 256.0`; a 215 MiB contiguous BO plus
`weights/*.dat` (208 MB) on the board; the four `bert_tiny_*.onnx`
fixtures added to the on-board model suite.

**Phase-2 notes from the schedule.**  The 49 host transposes move 9.44 M
elements (36 MiB of BO copies) per inference — 48 disappear with row-stride
/ transposed-B modes on MatmulKernel.  Host ops run in graph order and
block the kernel work that does not depend on them (e.g. the Q and K
Gemms could run during V's transpose).  All host ops together copy
127.5 MiB between the non-cacheable BO and the staging arena per
inference.  Gather reads the 47 MB word-embedding table in place (256 rows,
384 KB per inference); the table could live in host memory to save CMA.
`INFERENCE_BUF_POOL_SIZE_BYTES` in `inference.h` is the naive
no-reuse sum (462 MB); the real pool BO is 215 MiB.

### Phase 1 on the board (1a, 1g) — 2026-09-26, branch `feat/bert-demo`

KV260 at 100 MHz, bitstream with `kernels.matmul.max_k` 4096 (the same
bitstream passes the 148-model board suite including the four
`bert_tiny_*` fixtures), `demo/bert_squad/` (README there): the first 50
single-window SQuAD 1.1 dev questions (all from the "Super Bowl 50"
article), 208 MB of weights uploaded once to `/root/bert_squad_weights`,
project built on the board in 16 s.

* **Smoke test** — the generated `test_inference` (ramp inputs, expected
  logits from the simulation): **PASSED** (12.6 s including init).
* **Bit-exactness** — board start / end logits equal the scheduler
  simulation (`CodeGenerator._forward_pass`) **bit for bit on 3 / 3**
  checked examples and the study's `sched` emulation on **50 / 50**.  No
  tolerance anywhere: the aarch64 glibc 2.35 `exp` / `tanh` under
  `-ffp-contract=off` and the kernels' integer arithmetic reproduce the
  host's numpy + glibc 2.39 simulation exactly (a last-ulp libm difference
  would only matter where a host-op result lies within an ulp of a Q8.8
  rounding boundary — none reached the logits of the 50 inferences).
* **Accuracy** (official SQuAD normalisation):

| N | float32 EM / F1 | Q8.8 emulation EM / F1 | **KV260 EM / F1** | same span as float |
|---:|---:|---:|---:|---:|
| 20 | 90.0 / 91.7 | 90.0 / 91.7 | **90.0 / 91.7** | 19 / 20 |
| 50 | 88.0 / 90.3 | 88.0 / 90.3 | **88.0 / 90.3** | 49 / 50 |

  The one moved span (#16) answers "2015" where float says "2016," — both
  gold answers.
* **Latency** — **12.13 s per inference** (mean 12130.9 ms, min 12128.0,
  max 12134.0 over 50; per-layer profiling costs < 0.1 %),
  `inference_init` 0.3 s.  Per-layer profile (`--profile-layers`, N = 20,
  per inference; the sum, 12.14 s, equals the wall time — the schedule is
  effectively serial):

| kind | layers | time | share | measured rate |
|---|---:|---:|---:|---|
| MatMul linears | 74 | 7.66 s | 63.1 % | 2.78 GMAC/s (768², 768×3072), 2.97 (3072×768) |
| GELU (host) | 12 | 2.05 s | 16.9 % | 217 ns / element (171 ms per layer) |
| Softmax (host) | 12 | 1.07 s | 8.8 % | 114 ns / element (89 ms per layer) |
| attention MatMuls | 24 | 0.75 s | 6.2 % | QKᵀ 1.28 GMAC/s (K = 64), P·V 2.16 |
| LayerNorm (host) | 25 | 0.34 s | 2.8 % | 69 ns / element |
| Transpose (host) | 49 | 0.17 s | 1.4 % | 18 ns / element (≈ 220 MB/s BO read + write) |
| VectorOP | 126 | 0.10 s | 0.8 % | 0.4–0.8 G element/s |
| other host (Gather, OneHot, Cast, Slice) | 5 | 0.003 s | 0.0 % | |

  Against the gate (c) estimate (10.8 s): the MatmulKernel is faster than
  assumed (8.4 s at 2.8 GMAC/s instead of 10.0 s at 2.3) and the host ops
  are 5× slower (3.6 s instead of 0.7 s: double-precision `tanh` / `exp`
  on one A53 plus reads from the non-cacheable BOs).

**What phase 2 should attack first, by the measured profile:**

1. **The linears (7.66 s, 63 %).**  73 Gemms at 2.8 GMAC/s.  As 1×1 convs on
   ConvKernel (15–40 GMAC/s on 1×1, §0) they would take 0.55–1.45 s —
   −6.2 to −7.1 s, the single biggest item; needs `max_in_ch` ≥ 3072 (or a
   K-split) and the feature-major activation layout.
2. **GELU + Softmax on the host (3.12 s, 26 %) — cheap and bit-exact.**
   `host_gelu_tanh` is a pure function of its 16-bit Q8.8 input, so a
   65 536-entry `Data_t` table filled at init with the same double formula
   gives the same bits by construction (−~1.9 s).  Softmax's `exp` argument
   `x − max` is an exact multiple of 1/256 in [−256, 0], so a 65 536-entry
   `double` table of `exp` is exact too (the per-row sum and division stay).
   The simulation needs no change.  Then split rows of LayerNorm / Softmax
   over the four A53 cores (same per-element arithmetic, still bit-exact).
3. **Attention (0.75 s MatMul + 0.17 s Transpose, 7.6 %)** — row-stride /
   transposed-B modes on MatmulKernel remove the 48 transposes and the short
   K = 64 QKᵀ calls run at half the linears' rate.

VectorOP (0.8 %) is not worth touching.  With 1 and 2 done the model would
be at roughly 2.5–3.5 s per inference.
