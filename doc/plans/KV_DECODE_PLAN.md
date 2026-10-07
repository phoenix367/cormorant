# Decode attention on the FPGA: the chat models' KV cache read by ConvKernel

**Status (2026-10-07):** done — decode attention on the FPGA in all three chat libraries (bit-exact on the board; at position 1000 −9 to −11 % per token, +1–1.5 ms below ~130 positions); the chat server runs them; performance model topped up to 1930 exact calls.

## 1. Where it stands

- **The cache is already on the FPGA side.**  Every chat library keeps its K / V
  caches as raw int16 DMA states in the CMA pool, group-major `[KV][C][HD]`
  (V interleaved by `pv_kw` = 4 for ConvKernel's P·V), starting with the sink
  row (`src/llama.py` `_caches`).
- **Prefill attention runs on the FPGA** (CHAT_PLAN §16, policy
  `pow2+sink+p12+mix`): per layer `LlmAttnPrep` (host: RoPE, the new cache
  rows, the q image, a flush of rows [0, keys)), per KV group q·Kᵀ and P·V on
  ConvKernel over keys = roundup(pos + n, 64) (`LlmAttnConvNode`, a run-time
  dimension), the p12 softmax and `LlmAttnMerge` on the host.
- **Decode attention does not.**  `LlmAttention` (policy xattn: exact double
  on the host, 4 threads) reads every cached key per token.  Its cost grows
  with the position, ~30 ms per token at position 1000 (CHAT_PLAN §17); the
  rest of a 135M decode step is 54 ms at 250 MHz, so at long context the
  attention is about a third of the token.  CHAT_PLAN §16 kept it on the host
  when the kernels ran at 100 MHz and the FPGA alternative (MatmulKernel with a
  transposed K cache) saved 13 % at full context only.
- **On-chip storage is out of reach**: 135M's caches are 23.6 MB at 1024
  positions (360M ~42 MB); the device has ~2.3 MB of URAM and 0.6 MB of BRAM.

## 2. Design

Decode reuses the prefill path with one row (T = 1): `LlmAttnPrep` → q·Kᵀ
(ConvKernel, weight = the K cache rows, out_ch = keys, x = the q image of the
G heads of a group: G pixels) → p12 softmax (host) → P·V (ConvKernel) →
`LlmAttnMerge`, per layer and KV group: 2 · KV ConvKernel calls and 2 + KV host
ops per layer.

- **Numerics.**  The study's policy `pow2+sink+p12` is exactly this: the p12
  kernel attention in prefill and decode, the shipped formats.  It measured as
  the shipped `+mix` on teacher-forced metrics (135M §16.2: top-1 0.976 / 0.969,
  KL 0.0024, ppl 15.616; 360M §20: 0.9777 / 0.9759, 0.0018, 12.394); only the
  greedy answers differ (135M 2 / 12 identical against 3 / 12).
- **The decode node has no `n` input**: an empty `n` means one row
  (T = 1), as `LlmAttention` already does.
- **Cache flushes.**  Today a decode step writes its rows without a flush
  and every prefill flushes rows [0, keys) — 23 MB of cache maintenance per
  call at full context; the decode prep must flush only its own row of each
  cache (K: HD elements per group; V: the 16·VK-row block spanning it).
- **Option, not default.**  `LlamaFrontend(decode_attn="fpga" | "host")`,
  `generate_llm_project.py --decode-attn`, recorded in `project.json` and
  selecting the gates' study policy (`pow2+sink+p12` / `+mix`); the default
  flips only if the board says so (§3, phase 3).

## 3. Phases

| phase | work | done when |
|---|---|---|
| 0 study | numerics recap (§2); SmolVLM's text model under all-FPGA attention (`vlm_study.py` combo `pow2+p12+vgelu/pow2+sink+p12`); decode cost per token at 64…1024 keys priced with the 250 MHz performance and host models against today's xattn | a GO / NO-GO on cost and quality |
| 1 scheduler | `decode_attn` in `llama.py`; empty `n` in `LlmAttnPrep` / `LlmAttnScores` / `LlmAttnSoftmax` / `LlmAttnPV`; the row-only flush; tests: simulation = the study emulation of `pow2+sink+p12` bit for bit, the generated C in the incoherent emulator | scheduler tests |
| 2 host gates | `generate_llm_project.py` / `llm_project.py` / `llm_sched_check.py` / `llm_host_emu.py` / the VLM scripts with `--decode-attn`; the real 135M, 360M and SmolVLM projects bit-exact against the study on the host | every host gate bit-exact |
| 3 board | the three libraries in scratch dirs: `llm_board.py` gate (logits bit-exact) and `--decode-at 32,128,256,512,768,1000` against today's libraries | the crossover measured; the decision (FPGA always, or a length switch) |
| 4 production | the libraries the decision picks in the chat server, the performance-model top-up, docs, facts | the server healthy, every gate bit-exact |

Rules: every library change is proven on the board (its gate bit-exact against
the scheduler's simulation) before it reaches the chat server; scratch
`--remote-dir` for gates; one board job at a time; commit only when asked.

## 4. Results

### 4.1 Phase 0 — the estimate

Numerics: §2 (the measured studies).  Cost of the 135M decode step's attention
with the FPGA path (per token; 30 layers × 3 KV groups: 180 ConvKernel calls and
150 host ops), the calls priced by the engine cost model refitted at 250 MHz
(the performance model has no family for a q·Kᵀ of 3 output pixels; this call
family's model error is ±45–60 %), the host ops by the host model:

| keys | ConvKernel | host ops (prep, softmax, merge) | FPGA path | host xattn today (CHAT_PLAN §17) |
|---:|---:|---:|---:|---:|
| 64 | 1.7 ms | ~6.5 ms | ~8 ms | ~4 ms |
| 256 | 5.4 ms | ~6.5 ms | ~12 ms | ~9 ms |
| 1024 | 20.0 ms | ~6.5 ms | ~26.5 ms | ~33 ms |

Marginal: a q·Kᵀ with G = 3 output pixels leaves most of ConvKernel's grid idle,
and each decode step pays ~16 more `xclSyncBO` calls per layer (the softmax and
merge buffers, the cache rows).  The implementation exists (phase 1), so the
board measures it (phase 3).

### 4.2 Phases 1–2 — the scheduler and the host gates

- `LlamaFrontend(decode_attn="fpga")`: the decode entry takes `_fpga_attention`
  with T = 1 and an empty `n`; `LlmAttnPrep` / `LlmAttnScores` / `LlmAttnPV` /
  `LlmAttnSoftmax` treat a missing `n` as T rows (`has_n`); the decode prep
  flushes only its rows (`llm_cache_flush_rows`: each group's row span as whole
  64-byte lines).
- Tests (`test/test_llama.py`): `TestSimVsStudyFpgaDecode` — prefill + decode,
  truncate + re-prefill, a second turn on decoded rows — bit-exact against
  `pow2+sink+p12`; the decode graph's structure; the generated C of every entry
  in the emulator and the incoherent library sequence (prefill → decode → second
  turn → truncate).  Dropping the K-row flush fails that sequence.
- Scripts: `llm_project.policy(prefill_attn, decode_attn)`;
  `generate_llm_project.py` / `llm_sched_check.py` / `vlm_sched_check.py`
  `--decode-attn`; `project.json` records it; `llm_board.py` and
  `vlm_host_emu.py` rebuild the simulation from it.
- SmolLM2-135M (`llm_sched_check.py --decode-attn fpga`): 3 / 3 prompts and the
  second turn bit-exact, 32 decode steps each (25 min).  The project: decode 694
  nodes, pool 286 → 300 MB.
- SmolVLM's text model under `pow2+sink+p12` (`vlm_study.py` combo
  `pow2+p12+vgelu/pow2+sink+p12`, 24 images): top-1 0.9667, KL 0.0034, ppl
  12.223 — as `+mix` (teacher-forced metrics are prefill-only); identical greedy
  answers 7 / 24 (6 / 24 with `+mix`).

### 4.3 Phase 3 — SmolLM2-135M on the board

`llm_board.py --decode-at 32,128,256,512,768,1000 --decode-at-steps 8`, both
projects in scratch dirs sharing one weights dir (identical weight files),
bitstream 6436623029f7:

| position | host xattn (today) | FPGA attention | change |
|---:|---:|---:|---:|
| 32 | 51.7 ms / token | 52.7 | +1.0 ms |
| 128 | 55.5 | 55.2 | −0.3 |
| 256 | 58.9 | 57.9 | −1.0 |
| 512 | 66.5 | 63.3 | −3.2 |
| 768 | 74.7 | 68.6 | −6.1 |
| 1000 | 81.5 | 72.5 | −9.0 (−11 %) |

Gate: logits bit-exact (4 prompts × 33, `threads_identical`, re-open identical);
prefill 16 / 64 / 256 unchanged (156 / 297 / 811 ms); CMA 282 MB.  The crossover
is ~100 positions (a chat starts at ~40–170); the growth over the context
falls from ~30 to ~20 ms per token — the kernels' share, as the cost model
predicted (§4.1); the host ops cost less than the host model's 6.5 ms (≈ 1 ms
more than xattn's at 32 positions).  A length switch would save at most ~1 ms
below 128 positions at the price of a second policy in the emulation: not
worth it.

### 4.4 Phase 3 — SmolLM2-360M and SmolVLM

Host gates (`--decode-attn fpga`): 360M `llm_sched_check.py` 3 / 3 prompts and
the second turn bit-exact (32 decode steps each, 32 min); SmolVLM
`vlm_sched_check.py --text` 2 / 2 image prompts bit-exact (16 decode steps),
the vision entry unchanged.  Pools unchanged (360M 740 MiB, SmolVLM 494 MiB).
Board (`llm_board.py --decode-at …`, scratch dirs, one weights dir per pair):

| position | 360M host xattn → FPGA (ms / token) | SmolVLM host xattn → FPGA |
|---:|---:|---:|
| 32 | 133.3 → 134.8 (+1.5) | 51.6 → 52.5 (+0.9) |
| 128 | 138.1 → 138.5 (+0.4) | 54.6 → 55.0 (+0.4) |
| 256 | 145.4 → 143.8 (−1.6) | 58.4 → 58.0 (−0.4) |
| 512 | 159.7 → 153.4 (−6.3) | 66.5 → 62.8 (−3.7) |
| 768 | 173.6 → 163.2 (−10.4) | 74.5 → 68.0 (−6.5) |
| 1000 | 187.7 → 170.9 (−16.8, −9 %) | 81.3 → 72.2 (−9.1, −11 %) |

Every gate's logits bit-exact; prefill and `llm_image` (2.19 / 2.18 s) unchanged.

**Decision: FPGA decode in every chat library, no length switch.**  Below
~130 positions it costs ≤ 1.5 ms (≤ 1.1 %); from there it saves up to 9–11 %
and frees the host cores during decode.

### 4.5 Phase 4 — production

- `llm_project.DECODE_ATTN = "fpga"`: `generate_llm_project.py`,
  `llm_sched_check.py`, `vlm_sched_check.py` and `perf_calibrate.py`'s
  shipped graphs default to it; `llm_calibrate.SHIPPED` / `vlm_study.TEXT_SHIPPED`
  = `pow2+sink+p12` (the calibration records of `llm_models.json` keep the
  `+mix` metrics they were measured with — identical on the teacher-forced
  metrics, §2).
- The gated projects replaced `demo/chat/build/llm_project{,_smollm2_360m,_smolvlm_256m}`
  (the previous ones in `/mnt/data/act/kvdec/backup/`); `llm_board.py
  --install-only` (no weight changed: 0 files uploaded) and `deploy.py`; the
  three models answer through the server.
- Performance model `kv260/6436623029f7`: a fresh case list (1438 cases: the
  decode entries' ConvKernel calls), `run --resume`, two refinement rounds
  (118 + 26 calls): 1852 → 1930 exact calls.
- Tests: 1678 scheduler tests, the chat suite (`test_llm_calibrate` checks the
  manifest's study records against `SHIPPED`: re-recorded from the stored study
  results under `pow2+sink+p12` — identical metrics but 135M's identical answers,
  2 instead of 3), C simulation, the TTS host emulation; 60 facts — Piper's pool
  51.1 MiB (`chat.pool_mib`, from the OFFLOAD change).
