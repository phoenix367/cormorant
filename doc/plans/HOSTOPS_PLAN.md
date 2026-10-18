# The host row ops on the FPGA — a row-ops unit (study)

**Status (2026-10-15): phase 5a DONE — FOLD (op 9, Piper's `TtsPrep`) is in
production.**  Bitstream `de968ccc45da` (§11.7, after the kernels' area pass
of `AREA_PLAN.md`): WNS +0.029 ns, 89 547 LUTs; on the board 159 / 159 models
and every library bit-exact, the 72 kernel benchmarks equal to
`1cedaf2ed816`'s; the Piper chunk ≈ 260 → 183 ms (RTF 0.19 → 0.13); the
chat server ran it (with the R5) until 2026-10-16; FOLD is in every later
bitstream (production since 2026-10-17: `04259a7af917`, RPU_PLAN §14).
RPU_PLAN track A1 (2026-10-13) moved BERT's transposes, the Piper sums /
interleaves / relative-key band, the dequants, the row select and pixel
shuffle onto the unit (BERT 288 → 266 ms).

**Status (2026-10-12): phase 3 DONE — the unit is in production.  Bitstream
`1cedaf2ed816` (§9): the RowOpsKernel on `S_AXI_HP0_FPD`, WNS +0.020 ns at
4 ns, the design at 80 % LUT / 136 of 144 BRAM; on the board 159 / 159
models and every library bit-exact, the unit's calls at their modelled
rate; SmolLM2-135M prefill 256 776 → 448 ms (−42 %) and decode 54.9 → 49.4
ms per token, SmolLM2-360M prefill 256 1701 → 1125 ms (−34 %), BERT 364 →
288 ms (−21 %), SmolVLM 2.07 → 1.69 s per image, the Piper encoder 162 → 146
ms at 400 ids.  **Phase 4a DONE 2026-10-12** (§10.8): the prefill attention
prep on the unit as ROPE / TRANSPOSE / COPY jobs, no bitstream change —
SmolLM2-135M prefill 256 448 → 398 ms (0.51 × of 776: the 0.55 × gate met),
SmolLM2-360M 1 125 → 1 050 ms, SmolVLM's text prefill 449 → 399, every
library bit-exact.  **Phase 4b DONE 2026-10-12** (§10.9): the vision / Piper
encoder prep and the TTS row prep on the unit — SmolVLM 1.70 → 1.64 s per
image, the Piper encoder 146 → 96 ms at 400 ids.  BERT's 0.79 × (gate 0.75 ×)
stays: its transposes and the merge are phase 5.  Phase 2 DONE
2026-10-12: the scheduler lane (§7).  Phase 1 DONE 2026-10-11:
`kernels/rowops_rtl`, bit-exact, +0.025 ns at 300 MHz out of context (§7).
Phase 0 is done: the int32 residual
(§5.1) and the integer specifications of every op (`src/rowops_spec.py`,
§5.2) pass the gates of all five models (and again on the final specs,
2026-10-11 evening); the replay is
`perf_calibrate.py rowops`; the decode host-overhead measurement on the
board is the one phase-0 item left; no hardware yet.**  The user asked for "sigmoid + memory
transfer functions (transpose / concat / scatter / gather) in a separate
block on its own PS port (HP0)".  Priced against the shipped models, the
memory moves are at most 8 % of any model and the standalone sigmoid 0 %;
what the host spends its time on is the **float row ops** of the
transformer graphs — RMSNorm / LayerNorm, SiLU · up, the residual adds, the
attention prep / merge, the TTS encoder's row prep — 55 % of a SmolLM2
256-token prefill, 46 % of a 360M one, 30 % of BERT, 19 % of the SmolVLM
vision pass, 80 % of the Piper text encoder.  A streaming row engine on
`S_AXI_HP0_FPD` that runs those ops (and carries the movers as its address
generator) is worth, by the timed replay of the shipped graphs with the
unit priced at one PS port's rate (§3):

| workload (bitstream `f70ce632a97a`) | today | with the unit | gain |
|---|---:|---:|---:|
| SmolLM2-135M prefill 16 / 64 / 256 | 138 / 201 / 690 ms | 66 / 80 / 359 | **−52 / −60 / −48 %** |
| SmolLM2-135M decode step | 55.6 ms | 42.9 | **−23 %** (18 → 23 tokens/s; most of it per-op overhead, §1) |
| SmolLM2-360M prefill 16 / 64 / 256 | 225 / 385 / 1642 ms | 168 / 205 / 965 | −25 / −47 / **−41 %** |
| SmolLM2-360M decode step | 137.9 ms | 119.2 | −14 % |
| BERT-base (256 tokens) | 366 ms | 265 | **−28 %** |
| SmolVLM vision (1024 patches) | 1779 ms | 1546 | −13 % |
| Piper text encoder, 400 ids | 159 ms | 49 | **−69 %** (first audio −0.1 s) |
| Piper decoder chunk | 254 ms | 211 | −17 % |
| CNNs, YOLO, stereo | | | ≤ 3 % (SpaceToDepth, Concat, Resize) |

The cost is a fifth kernel (`kernels/rowops_rtl`, fixed point throughout:
~80 DSP / ~15 k LUT / 10 URAM of the 536 / 54 k / 16 free), a fourth data
interconnect on HP0, a fifth lane in the scheduler with a DMA int32 storage
kind for the residual stream, and the numerics work: every op gets an
integer specification (the softmax unit's way, `SOFTMAX_PLAN.md`) and every
model is re-gated against it — §5.  The residual stream itself moves from
float32 to int32 at 2^−15, which makes its adds exact; §5.1 has the
ablation (phase 0's first item, run 2026-10-11).

## 1. Where the host time is

The host ops of every shipped model, priced with the board's host model
(`perf_models/kv260/host.json`, fitted on board profiles) and the kernel
model of `f70ce632a97a` (`perf_calibrate.py`'s `HostModel` / `PerfModel`
on the shipped graphs; sums per entry, not the critical path):

| model / entry | kernel calls | host ops | of which data moves | the costly ops |
|---|---:|---:|---:|---|
| ResNet-18, MobileNet v1 / v2 | 18.7 / 19.8 / 16.8 ms | 0.6 | 0.6 (3 %) | SpaceToDepth (the stem rewrite) |
| BERT-base | 254.5 | 110.5 | 29.6 (8 %) | LayerNorm 80.9 (25 calls), Transpose 29.4 (49) |
| SmolLM2-135M decode | 55.5 | 13.9 | 0 | AttnPrep 5.8, SiLU·up 4.2, RMSNorm 1.4 |
| SmolLM2-135M prefill_256 | 622.3 | 379.0 | 0 | SiLU·up 160.4, RMSNorm 82.3, AttnPrep 81.5, ResAdd 49.1, AttnMerge 5.4 |
| SmolLM2-360M prefill_256 | 1441.7 | 761.2 | 0 | SiLU·up 308.6, RMSNorm 189.6, AttnPrep 141.0, ResAdd 111.9 |
| SmolVLM vision | 1597.7 | 332.4 | 1.2 | VitAttnPrep 132.5, VitLayerNorm 102.4, VitResAdd 57.1, VitGelu 20.1 (the one layer left on the host), AttnMerge 12.2 |
| Piper chunk | 125.6 | 139.8 | 4.7 (2 %) | TtsPrep 77.5, TtsSum 30.5, TtsGate 20.0 |
| Piper encode_400 | 29.6 | 126.5 | 56.9 (36 %) | TtsRowPrep 56.9, AttnMerge 23.2, ResNorm 17.7, RelAdd 17.3 |

Three things follow.  (1) A pure data mover (transpose / concat / gather /
scatter) pays on BERT (8 %) and the Piper encoder (36 %, but 57 ms) and
nowhere else; the LLM graphs have no moves — their "moves" are already
folded into AttnPrep / AttnMerge, which also round and scatter.  (2) The
standalone `Sigmoid` appears in no shipped graph: SiLU (`x · Sigmoid(x)`)
is fused onto VectorOPKernel's activation unit already; YOLOv5n's head
applies its sigmoids in the demo's post-processing, outside the graph.
(3) The ops that cost are per-row arithmetic over float32 or exponent
tensors — the same data the kernels already stream, at a rate an A53
cannot reach: SmolLM2-135M's SiLU·up spends 13.6 ns per element over four
cores (a table lookup, a double multiply and a rounding), BERT's LayerNorm
16.5 ns per element (two double passes over each row).

The host ops' time at small sizes is mostly overhead, not arithmetic: a
decode step's 212 host ops move 0.6 MB in 15 ms (70 µs per op: cache
maintenance of the kernel-written inputs, the thread fan-out of
`host_parallel`, the flush of the outputs).  A host-side fix (fewer, fused
host ops; no fan-out below a size) could take a third of that; it is the
cheap first step for decode (§7 phase 0) and does nothing for prefill.

## 2. What the ops compute — the numerics the unit must reproduce

Every host op has a numpy `reference()` (the specification; the generated
C computes the same bit for bit, and the board gates compare every
element) and writes its outputs with `_st`: round half to even at the
tensor's exponent 2^f, saturate to int16, NaN → 0.  Inputs are raw int16
at per-tensor or per-channel exponents, or the float32 residual stream
(`host = "f32"`, a plain host array today, never a DMA buffer).

| op (model) | inputs → output | the arithmetic (as the reference does it today) | the unit's fixed-point form |
|---|---|---|---|
| `LlmRMSNormNode` (SmolLM2) | h f32 [T][n], gamma f32 → int16 at f_y[c] | `ss = Σ_c h²` left to right in double, `r = 1/sqrt(ss/n + eps)`, `y = (h·r)·gamma`, round at 2^f_y[c] | h int32 at 2^−15; Σ h² exact in a 96-bit accumulator; `r` from an rsqrt table + one Newton step at 2^−30 (≤ 1 LSB of the exact value); y = round(h · r · gamma) with gamma int24 |
| `VitLayerNormNode` (SmolVLM), `LayerNormNode` (BERT, int16 in), `TtsResNormNode` (Piper, adds x + y + by first), `StereoInstanceNormNode` | row f32 or int16, gamma / beta f32 → int16 | mean, var (sums left to right in double), `((h − mu)/sqrt(var + eps))·gamma + beta`; BERT's TF form `x·g + (beta − mean·g)` | two passes: Σ h exact, then Σ (h − mu)² exact (no cancellation: integer sums); rsqrt as above; gamma / beta int24 |
| `LlmResAddNode`, `VitResAddNode` | h f32, d int16 at f (+ bias f32 per channel) → f32 | `float32(h + d)`; Vit: `float32(h + (d + b[c]))` | h int32 at 2^−15, d shifted to 2^−15 (exact: every delta exponent ≤ 15), a 32-bit add with saturation — **exact**; the Vit bias as int32 at 2^−15 |
| `LlmSiluMulNode` | g int16 at f_g[c], u int16 → int16 at f_out[c] | `silu(g)` from a 65 536-entry double table per gate exponent (libm `exp`), `· u` in double, round | `silu(g)` tabulated as int32 at 2^−20 per gate exponent (a 256 KB table the unit loads on exponent change), integer product with u, round at f_out |
| `VitGeluNode` | raw int16 (+ bias raw) → int16 | `gelu_f[c][sat16(raw + braw[c])]` — a per-channel-exponent table of the exact GELU rounded at f | already VectorOPKernel's act unit's scheme (`OFFLOAD_PLAN.md` §2.1 moved 11 of 12 layers there); the last layer's outputs are at 2^−9 — a second 4 096 × 7 table |
| `LlmAttnPrepNode`, `VitAttnPrepNode`, Piper's `VitAttnPrepNode` | q0, k0, v int16; cos / sin f32 tables → the q·Kᵀ conv image int16 at f_q[h], K / V cache rows | RoPE in double: `q·cos − rot(q)·sin` per pair, round at f_q[h]; k likewise into the cache; v re-rounded; the output scattered into ConvKernel's `1 × kw` image layout (`nodes.conv_lowered_b_image`) | cos / sin as int24 at 2^−23 (exact to 2^−24, the output rounds at ≤ 2^−15), two 16 × 24 products, one subtraction, round; the image address generator; cache-row writes; Vit: bias add + round |
| `LlmAttnMergeNode`, `TtsAttnMergeNode` | the P·V outputs per group → [T][H·HD] | a head-order gather (raw copy); Tts adds the relative-value term `P·ev` (double) and rounds | the address generator; Tts: an integer MAC over 2W + 1 taps (P at 2^−12, ev int16) |
| `TtsRowPrepNode` | x [T][C], bias → [T][taps·C] | tap stacking (rows t + k − off, zero outside [0, n)), + bias, ReLU, round | the address generator + int add + ReLU |
| `TtsAttnRelAddNode`, `TtsSumNode`, `TtsGateNode` | raw int16 rows | integer / double adds at exponent changes, a tanh · sigmoid gate (double) | exponent shifts + round; the gate as a table or an integer spec |
| `TransposeNode`, `ConcatNode`, `SliceNode`, `SpaceToDepthNode` | any dtype | strided copies, bit for bit | the address generator (16-byte granules; BERT's transposes move 128-byte head rows) |
| `TtsPrepNode` (Piper chunk, 77 ms) | the flow's coupling prep | several passes, per-frame geometry | **stays on the host** in this plan (its own op; a later candidate) |

Two facts decide the design.  **The float32 residual stream** (SmolLM2,
SmolVLM): h lives in host memory as float32; the norms read it, the
residual adds read and write it.  It is float32 only because the numerics
study rejected an int16 residual (`CHAT_PLAN.md` §3.2: the position-0 sink
reaches 25 982, the deltas have exponents up to 2^−15); an int32 residual
at 2^−15 holds both — a range of ±65 536 and every delta exactly — and is
what the unit streams (§5.1).  **Double arithmetic in the references**: the
norms, RoPE and SiLU·up compute in double and round once.  That double was
a consequence of writing the ops in C on the A53, not a requirement: the
outputs are int16 at an exponent, the inputs int16 or the residual, and
every op has a fixed-point form whose only approximations are a table
(rsqrt, silu, gelu, the gate) and the final rounding — the sums themselves
become exact.  The references change to those integer algorithms and
every model is re-gated for quality (`llm_study.py`, `vlm_study.py`, the
BERT SQuAD set, Piper's log-mel, the stereo EPE), as the softmax unit was.

## 3. The gain — a timed replay of the shipped graphs

Method: `src/codegen/timing.simulate` on every shipped entry (the graphs
`perf_calibrate.py` builds), the kernel calls at the `f70ce632a97a`
performance model, the host ops at `host.json`, once as today and once
with every op of §2 (except `TtsPrepNode`) priced as a streaming unit
call: `max(bytes read, bytes written) / 3.4 GB/s + 2.5 µs` — one 128-bit
PS port at 250 MHz at 85 % (what the binary VectorOP reaches on its ports)
plus the issue / completion latency of a kernel call (the measured minimum
is 1.8 µs).  Bytes are the inputs' and the output's numel × 4 (f32) or 2
(int16); the KV caches a prefill writes count as written rows only.  The
same replay predicts today's BERT at 366.5 ms (measured 363.9) and the
135M prefill_256 at 690 ms (measured 776: the LLM entries' host ops run
under memory pressure the model does not see — the gains below are
relative to the replay's own baseline).

| entry | today (replay) | with the unit | host ops today → unit | the unit's calls |
|---|---:|---:|---:|---:|
| BERT | 366.5 ms | 264.9 (−27.7 %) | 111.2 → 8.6 ms | 76 (LayerNorm 25, Transpose 49) |
| SmolLM2-135M decode | 55.6 | 42.9 (−22.8 %) | 15.0 → 0.8 | 212 |
| SmolLM2-135M prefill_16 / 64 / 256 | 138.3 / 201.3 / 689.8 | 66.1 / 80.0 / 358.5 (−52 / −60 / −48 %) | 77 → 3.5, 135 → 12, 380 → 47 (SiLU·up 160 → 14, RMSNorm 82 → 11, AttnPrep 82 → 4, ResAdd 49 → 16) | 210 |
| SmolLM2-360M decode | 137.9 | 119.2 (−13.6 %) | 22.0 → 0.9 | 226 |
| SmolLM2-360M prefill_16 / 64 / 256 | 224.5 / 385.2 / 1641.8 | 167.7 / 204.5 / 965.1 (−25 / −47 / −41 %) | 65 → 6, 204 → 21, 763 → 84 | 224 |
| SmolVLM vision | 1778.6 | 1546.0 (−13.1 %) | 334 → 92 (AttnPrep 133 → 28, LayerNorm 102 → 23, ResAdd 57 → 33, Gelu 20 → 2) | 74 |
| Piper chunk | 253.5 | 210.5 (−17.0 %) | 140 → 97 (TtsSum 31 → 6, TtsGate 20 → 1; TtsPrep stays) | 35 |
| Piper encode 32 / 64 / 128 / 256 / 400 | 15.8 / 27.0 / 50.5 / 99.4 / 159.2 | 5.0 / 6.5 / 13.8 / 27.9 / 48.8 (−69…−76 %) | 127 → 10 at 400 ids | 55 |

What the table says: the prefill entries halve — their host ops are
arithmetic over 35–95 MB per entry that the unit streams in 10–28 ms
each; the vision pass gains 13 % because its kernels dominate (1.6 s of
ConvKernel MatMuls); the decode steps gain 14–23 %, which is the per-op
overhead of §1 (0.6 MB moved in 15 ms) — the unit's 212 calls sit at the
2.5 µs floor, and the host fix of phase 0 can take part of the same time
without hardware; BERT's gain is its 25 LayerNorms (81 → 3 ms) plus the
49 transposes (29 → 6).  The replay is a bound in one respect — the unit's
calls are serialised on the CPU timeline like every kernel call, with no
overlap modelled beyond what the event stream allows — and optimistic in
another: 3.4 GB/s assumes long rows; the Piper encoder's and the decode
step's rows are short, and their calls will sit at the 2.5 µs floor
(already in the price).

## 4. The unit

**RowOpsKernel** (`kernels/rowops_rtl`, VLNV `xilinx.com:hls:RowOpsKernel:1.0`
in the project's pattern): a streaming row engine with one AXI4 read
master and one AXI4 write master (128-bit, bursts ≤ 64 beats, 16
outstanding, as the other RTL kernels declare), an AXI-Lite control port
with the HLS register-block layout (`ap_ctrl_hs`, GIE / IER / ISR, the
generated driver from `scripts/gen_driver.py` as the other kernels), one
job per `ap_start`.

- **The job**: `op`, `rows`, `n` (elements per row, ≤ 4096), up to three
  sources and two destinations as (address, row stride, element kind:
  i16 / f32 / i32, exponent) tuples, a per-column parameter row (gamma,
  beta as f32; the per-channel output exponent as int8; the per-channel
  gate exponent) read once per job into a parameter RAM, `eps` (f64 bits),
  a table id, and the address generator's geometry (§4.2).
- **The engine**: rows are read in bursts into a 16-row ring in URAM (16
  × 16 KB: int32 rows of 4096 — 2 URAM pairs), processed by 8 lanes at
  250 MHz (one 128-bit beat of int16 per cycle; int32 rows take two cycles
  per beat on the same lanes), written back in bursts.  The reductions
  (Σ h, Σ h², Σ (h − mu)²) are exact integer sums in wide accumulators
  (a DSP48E2 cascade, 96 bits for Σ h² of 4096 values at 2^−15), one per
  lane, combined per row; order does not matter for an exact sum, so the
  spec is simply the sum.  A two-pass op (mean then var, or Σ h² then the
  scale) rereads the row from the ring.  The per-row scalar
  `1/sqrt(D)` comes from an 8 192-entry rsqrt table on the leading 13 bits
  of the block-floated denominator plus one Newton step in 32-bit integers
  (the softmax unit's exp-table pattern; relative error below 2^−24), one
  result per row, a few DSPs; no division anywhere — `src/rowops_spec.py`
  folds n into the denominator (`V = n·S2 − S1²`) and sqrt(n) into gamma
  (§5.2).
- **Ops**: `NORM` (variants rms / mean-var, input i16 or f32, TF form;
  output i16 at per-column f or i32), `RESADD` (i32 ← i32 + i16 · 2^(15 − f)
  [+ bias], saturating), `MULACT` (i16 ← act_table[g] · u at exponents), `ACT`
  (a table lookup with a bias: GELU, tanh · sigmoid gate), `DEQUANT`
  (f32 ← i16·2^−f[c]), `ROPE` (i16 ← round((q·cos − rot(q)·sin)·2^f),
  cos / sin rows f32 from DDR per position; the k rows to the cache
  destination), `SUMEXP` (i16 ← Σ_i x_i·2^(f_i − f_out), the TTS sums and
  the relative-key add), `COPY` (any kind, the address generator alone:
  transpose, concat, gather, tap stacking with zero rows outside a range,
  ReLU on the way).
- **Tables**: GELU and the gate as 4 096 × 7 ROMs in BRAM (VectorOPKernel's
  activation unit's scheme, `scripts/gen_act_rom.py`); SiLU at a gate
  exponent as a 65 536-entry RAM (int32 at 2^−20: 256 KB, 8 URAMs) that
  the unit loads from DDR when the job's table id changes — a 256 KB
  burst, 80 µs; SmolLM2-135M's 30 layers have a few distinct gate
  exponents, SmolLM2-360M's 32 likewise, so a prefill reloads a handful of
  times.
- **Resources** (fixed point throughout — no floating-point operators):
  8 lanes × (a 32 × 24 product, 2 DSP48E2; a 32-bit square, 4; the RoPE
  pair, 2) + the accumulators and the rsqrt step ≈ **80–100 DSP**; the AXI
  masters, the address generator, the ring control, the roundings ≈
  **12–15 k LUT**; **10 URAM** (ring 2, SiLU table 8), **7 BRAM** (the
  rsqrt and GELU tables, parameter RAM, FIFOs).  Free after
  `f70ce632a97a`: 536 DSP, 54 502 LUT, 16 URAM, 13 BRAM — it fits with
  room.  Risk: the design closes 4 ns by
  0.098 ns; a fifth kernel adds placement pressure near the HP0 pins
  (bottom of the device, by the PS).  Mitigation as before: register the
  reset, keep the AXI masters' outputs registered, synthesise out of
  context at 300 MHz first; the datapath is DSP cascades and roundings,
  the kind the other kernels close at 300 MHz out of context.
- **Port**: `S_AXI_HP0_FPD` (free; DDRC port 3, shared only with the
  unused DisplayPort), through its own one-slave interconnect
  (`axi_r_intercon`, a register slice on the SI, none on the MI —
  `scripts/bd_conv_x_hp.tcl`'s pattern, which also showed that moving a
  master to an HP port costs nothing when its traffic is long bursts).
  The unit's reads and writes thus never share a crossbar with the
  kernels; the HP port's 4 GB/s per direction is the unit's full rate
  (an HP port keeps one DDRC channel — `PS_PORTS_PLAN.md` §8 — which cost
  the kernels' 64-beat writes 0.5–1 %; the unit's writes are the same kind
  of long bursts, so the same 1 % applies to it, not to the kernels).
  Row reads of 2–4 KB and writes of the same length are the CCI-friendly
  shape; the short rows (decode, head dim 64 gathers) are latency-bound
  and sit at the call floor whatever the port.

### 4.1 Why a unit and not VectorOPKernel

VectorOPKernel is int16-only (Q8.8 lanes, 128-bit ports, the activation and
softmax units); the ops of §2 need float32 operands, per-column exponents
and gammas, double reductions and a scatter / gather address generator.
Grafting them onto it would double its register map and put f32 paths into
a kernel whose critical paths are its burst generators (`VECTOROP_RTL_KERNEL.md`
§6).  The softmax unit was the right thing to put there: int16 in, int16
out, one row at a time.  The row-ops unit is the float / exponent sibling,
on its own port, and leaves VectorOPKernel's timing alone.

### 4.2 The data mover — the unit's address generator

The user's "memory transfer block" is the unit's `COPY` op: a 3-level
nested address pattern on each side (count, stride) × 3 with a 16-byte
granule, an optional per-row window (rows outside [r0, r1) read as
zero — tap stacking), an element kind on each side (i16 ↔ f32 with an
exponent — `DEQUANT` and the int16 cast are the same op) and ReLU.  It
covers every mover in the graphs: Transpose of a [T][H][HD] head split
(128-byte rows), Concat (row ranges), the head-order AttnMerge gather,
TtsRowPrep's tap stacking, SpaceToDepth (4-byte granules of the 2-pixel
pairs — a 2-level pattern with a 4-byte granule runs at a quarter rate;
0.6 ms today, not worth more), Resize (nearest, integer factors: row and
element repeats).  A separate "DMA block" would be the same address
generator without the datapath — hence one unit, not two.

### 4.3 Sigmoid

No shipped graph has a standalone `Sigmoid`; SiLU is on VectorOPKernel's
activation unit already.  For completeness (gates of GRU / LSTM cells,
squeeze-excitation blocks of EfficientNet / MobileNetV3 — which use
hard-sigmoid) the cheapest home is VectorOPKernel's activation unit: act
7 / op 12 = sigmoid from a 4 096 × 7 table of `σ(−|x|)` with `σ(x) = 1 −
σ(−x)`, exact to the Q8.8 rounding like the other acts
(`ACTIVATIONS_PLAN.md`), one BRAM and a few LUTs, no port or scheduler
change beyond the op code — to be bundled with the next VectorOPKernel
bitstream, not a bitstream of its own.

## 5. The numerics: fixed point throughout

The first draft of this study offered IEEE fp64 operators as a way to keep
today's double references bit for bit without re-gating.  The user asked
why fp64 at all, and the answer is that nothing needs it: the double was
how the C helpers were written for the A53, the gate is bit-exactness
between the simulation and the board, and that gate holds by construction
whenever the simulation models the hardware's integer algorithm — the
softmax unit's way (`src/vectorop_smx.py`: a Python function that is the
spec, the C fallback, the RTL).  So every op gets an integer specification
(§2's right column), and the models are re-gated once:

- **The residual stream becomes int32 at 2^−15** (`res_bits = 15`).  The
  int16 residual failed in `CHAT_PLAN.md` §3.2 (the sink at 25 982 and the
  fine deltas cannot share 16 bits); int32 at 2^−15 has the range (±65 536)
  and every delta's resolution (the calibrated delta exponents are ≤ 15 —
  `o` 15 at layer 0, `d` 13 — so the adds are exact; no rounding happens
  in the stream at all).  Against float32 it is coarser only below a
  magnitude of 2^8 and finer above (float32 at 25 982 has a resolution of
  2^−9).  §5.1 measures it.
- **The norms**: exact integer sums, an rsqrt table + Newton step at
  2^−30, int24 gamma / beta; gate: ≤ 1 LSB against the exact function on
  every row of every shipped model's simulation, and the task metrics.
- **RoPE**: int24 cos / sin; **SiLU · up, GELU, the TTS gate**: int24
  tables; **the TTS sums and relative-key add**: exact as they are.
- **Dequant** to float32 for the caller (the logits): an exact conversion
  of int16 · 2^−f, unchanged.

The gates per model: `llm_study.py study` on SmolLM2-135M / 360M (top-1,
KL, perplexity against float; the bf16 yardstick), `vlm_study.py study` on
SmolVLM (image features, answer top-1, KL), the 60-example SQuAD set for
BERT, Piper's log-mel distance, the stereo EPE.  Phase 0 runs them with
each op's integer spec substituted in the study models, the residual
first.  Nothing in hardware starts before they pass.

### 5.1 The int32 residual — ablation (2026-10-11)

`llm_study.py` policies `pow2+sink+p12+r15` / `r14` / `r12` / `r10`
(`res_bits`: the residual rounded half-even to int32 at 2^−N after the
embedding rows and every add, saturating) against the shipped
`pow2+sink+p12`; `vlm_study.py` combos with the vision residual at
`r15` / `r12` as well (its residual peaks at 753).

SmolLM2-135M (12 prompts, 128 new tokens, 3 held-out windows of 1024
WikiText-2 tokens; 5 737 s):

| policy | top-1 all / responses / held-out | top-5 held-out | KL held-out | ppl (float 15.598) | identical generations |
|---|---|---|---|---|---|
| shipped `pow2+sink+p12` | 0.9756 / 0.9798 / 0.9690 | 0.9997 | 0.0024 | 15.616 | 2 / 12 |
| + **r15** | 0.9689 / 0.9808 / 0.9687 | 0.9997 | 0.0025 | 15.611 | 2 / 12 |
| + r14 | 0.9744 / 0.9808 / 0.9671 | 0.9997 | 0.0025 | 15.623 | 2 / 12 |
| + r12 | 0.9750 / 0.9808 / 0.9710 | 0.9997 | 0.0024 | 15.607 | 4 / 12 |
| + r10 | 0.9738 / 0.9778 / 0.9687 | 1.0000 | 0.0024 | 15.618 | 4 / 12 |

Every int32 variant sits inside the shipped policy's own spread (KL
0.0024–0.0025, perplexity 15.607–15.623 around 15.616, held-out top-1
within 0.2 points; the "all" column moves with a single long prompt's
greedy path), no element saturates, the exponents and accumulator peaks
are unchanged.  2^−15 is the choice because it also makes the adds exact;
the stream could go as coarse as 2^−10 without a measurable loss.

SmolLM2-360M (the same prompts and windows; its residual peaks at 22 887;
4 307 s):

| policy | top-1 all / responses / held-out | top-5 held-out | KL held-out | ppl (float 12.373) | identical generations |
|---|---|---|---|---|---|
| shipped `pow2+sink+p12` | 0.9777 / 0.9936 / 0.9759 | 0.9997 | 0.0018 | 12.394 | 8 / 12 |
| + **r15** | 0.9763 / 0.9936 / 0.9707 | 0.9997 | 0.0018 | 12.392 | 8 / 12 |
| + r14 | 0.9777 / 0.9911 / 0.9726 | 0.9997 | 0.0018 | 12.391 | 8 / 12 |
| + r12 | 0.9589 / 0.9898 / 0.9778 | 0.9997 | 0.0018 | 12.394 | 6 / 12 |
| + r10 | 0.9638 / 0.9975 / 0.9756 | 0.9997 | 0.0018 | 12.407 | 10 / 12 |

The same picture: KL and perplexity unchanged at every width, the top-1
columns moving by a few tenths of a point in both directions (the
generations diverge at different tokens, not earlier or more often), no
saturation.

SmolVLM-256M (24 images, 96 new tokens, teacher-forced answer positions
against the float64 reference; 8 046 s):

| vision / text policy | features rel (max) | cos min | answer top-1 | KL | identical | ppl |
|---|---|---|---|---|---|---|
| float | | | | | | 12.221 |
| shipped `pow2+p12+vgelu+vsmx` / `pow2+sink+p12` | 0.0548 (0.0687) | 0.99788 | 0.9704 | 0.0035 | 8 / 24 | 12.223 |
| … + **r15** / … + **r15** | 0.0546 (0.0678) | 0.99794 | **0.9734** | 0.0036 | 7 / 24 | 12.222 |
| … + r12 / … + r12 | 0.0547 (0.0689) | 0.99787 | 0.9660 | 0.0031 | 5 / 24 | 12.225 |

The int32 residual at 2^−15 is indistinguishable from float32 on every
metric (the differences are the greedy paths' noise: 7 vs 8 identical
answers, top-1 +0.3 points); no int32 element saturates (the float run's
45 613 "saturated" h values are its counter of what int16 could not hold).
Even 2^−12 passes on this model.

**Verdict of the ablation: the residual stream goes to int32 at 2^−15 in
all three models** — the yardstick (quality indistinguishable from the
float residual on KL, perplexity and top-1; the bf16 yardstick of
`CHAT_PLAN.md` is far looser) is met with no saturation and, at 2^−15,
with exact adds.  Commands (host only, `.venv-export`; the three runs
took 5 737 + 4 307 + 8 046 s on the dev PC):

```bash
PY=.venv-export/bin/python
$PY demo/chat/scripts/llm_study.py --assets demo/chat/assets/smollm2-135m-instruct study \
    --policies pow2+sink+p12,pow2+sink+p12+r15,pow2+sink+p12+r14,pow2+sink+p12+r12,pow2+sink+p12+r10 --out OUT.json
$PY demo/chat/scripts/llm_study.py --assets demo/chat/assets/smollm2-360m-instruct study --policies ... (the same)
$PY demo/chat/scripts/vlm_study.py --assets demo/chat/assets/smolvlm-256m-instruct study \
    --combos "pow2+p12+vgelu+vsmx/pow2+sink+p12,pow2+p12+vgelu+vsmx+r15/pow2+sink+p12+r15,pow2+p12+vgelu+vsmx+r12/pow2+sink+p12+r12"
```

### 5.2 The integer specifications (2026-10-11)

`inference-scheduler/src/rowops_spec.py` is the unit's arithmetic, written
as plain integer operations (numpy int64, Python ints where a product
passes 63 bits), with one constant table per function computed in double
and rounded once — the file the simulator's references, the C fallback and
the RTL are all checked against (as `vectorop_smx.py` is for the softmax):

- `rsqrt_bf(D)`: the block-floated denominator's mantissa `Dm` in
  [2^30, 2^32) (even exponent), the 8 192-entry table on its leading 13
  bits (`round(2^47 / sqrt((idx + ½)·2^19))`), one Newton step with every
  intermediate truncated as a right shift truncates: `t = Dm·y0 >> 32`,
  `u = t·y0 >> 32`, `y1 = y0 + (y0·(2^30 − u) >> 31)`; relative error
  below 2^−24 over 5 000 random denominators of 1 to 90 bits (a 12-bit
  index reached 2^−23.4 at the bottom of the range).
- `rms_norm(h_raw, f_h, gamma, eps, f_out)`: `D = Σ h² + round(eps·n·2^2f)`
  (exact), `P1 = floor(h_raw · Rm · 2^(24 − sh))` (the normalised value at
  2^−24; an exact 64-bit product split at 2^20), `y = rne((P1 · G) /
  2^(24 + g − f_out[c]))` with `G = round(gamma · sqrt(n) · 2^g)` as int32
  at a per-tensor g, saturated.
- `layer_norm(x_raw, f_x, gamma, beta, eps, f_out)`: rows wider than 24
  bits block-floated per row (the int32 residual; int16 inputs untouched),
  `S1 = Σ x`, `S2 = Σ x²`, `V = n·S2 − S1²` and `A = n·x − S1` exact, `D =
  V + round(eps·n²·2^2f)`, then as the RMSNorm with `Bq = round(beta ·
  2^(24 + g))` added before the rounding.  BERT's TF form `x·g + (beta −
  mean·g)` and the plain form are the same function here.
- `rope(x_raw, f_x, c_raw, s_raw, f_out)`: int24 cos / sin at 2^−23 from
  the float32 tables, per-channel input exponents (a rotation pair at two
  exponents is shifted up to the finer one, exact), one rounding of the
  integer sum.
- `silu_mul(g_raw, f_g, u_raw, f_u, f_out)`: `silu_table_raw(f_g)[raw] =
  round(silu(raw·2^−f_g)·2^20)` (glibc `exp`, as the host table), the
  integer product with u, one rounding.

`test/test_rowops_spec.py` (8 tests, 0.3 s) checks every function against
the exactly rounded double computation on rows of mixed scale, with an
outlier channel and a near-constant row: **every output within 1 LSB, fewer
than 1 % of the elements differing at all**, the helpers exact.

The gates: the study scripts run the specs in place of their double ops —
`llm_study.py` policies `pow2+sink+p12+r15+inorm` / `+irope` / `+isilu` /
`+iops` (all three), `vlm_study.py` `pow2+p12+vgelu+vsmx+r15+inorm` (the
vision LayerNorms on the int32 residual) with the text model at `+iops`,
`bert_study.py` `sched+vsmx+iln` (every LayerNorm region replaced by the
spec on the Q8.8 residual-add output; 60 SQuAD examples), `piper_study.py
encoder-vsmx` with a third column "VectorOP softmax + unit LN"
(`piper_vits.ROWOPS_LN`: the encoder's residual LayerNorms, the sum exact
at the finer exponent).  Results:

**Piper encoder** (`encoder-vsmx`, the 28 evaluation sentences, 1 138
phoneme durations, log-mel distance of the audio synthesised with the
encoder under test and the float durations kept):

| encoder | durations changed vs float | log-mel vs float | max |x − float| |
|---|---|---|---|
| library (host softmax) | 4 / 1138 (max 1 frame) | 0.388 dB | 0.0318 |
| softmax on VectorOPKernel (shipped) | 3 / 1138 | 0.391 dB | 0.0271 |
| + the unit's LayerNorm | 4 / 1138 | **0.384 dB** | 0.0286 |
| + the unit's LayerNorm, the final specs (rerun 2026-10-11 evening) | 4 / 1138 | **0.362 dB** | 0.0264 |

Pass: the same distance as the two encoders in production (the yardstick
is 2.64 dB between two float runs of the decoder, `TTS_PLAN.md`), one
duration differing between the unit-LN and the shipped encoder.

**BERT-base** (`bert_study.py --n 60`, the 25 LayerNorms per inference as
the spec on the Q8.8 residual-add outputs; 75 min):

| policy | EM | F1 | same span as float | mean max logit error |
|---|---|---|---|---|
| float | 93.3 | 96.5 | | |
| `sched+vsmx` (shipped) | 93.3 | 96.5 | 60 / 60 | 1.289 |
| `sched+vsmx+iln` | 93.3 | **97.1** | 59 / 60 | 1.293 |
| `sched+vsmx+iln`, the final specs (rerun 2026-10-11 evening) | 93.3 | **97.1** | 59 / 60 | 1.290 |

Pass: one example's span moves by a token (to the better-scoring one — the
same 59 / 60 the pre-FPGA policies of `BERT_PLAN.md` §0 showed), the
logit error unchanged.

**SmolVLM-256M** (24 images; the vision tower's 25 LayerNorms as the spec on
its int32 residual, the text model with all three specs; 8 073 s):

| vision / text policy | features rel (max) | cos min | answer top-1 | KL | identical | ppl |
|---|---|---|---|---|---|---|
| r15 / r15 (§5.1) | 0.0546 (0.0678) | 0.99794 | 0.9734 | 0.0036 | 7 / 24 | 12.222 |
| r15 + **inorm** / r15 + **iops** (the final specs, rerun 2026-10-11 evening) | 0.0547 (0.0681) | 0.99793 | 0.9734 | 0.0036 | 9 / 24 | 12.229 |

Pass: the image features and KL unchanged, top-1 within the greedy
paths' spread (the shipped float-residual policy itself sits at 0.9704),
no saturation.

**SmolLM2-135M** (the §5.1 prompts and windows; 7 437 s) and
**SmolLM2-360M** (2 930 s), each spec alone and all three together on the
int32 residual:

| policy | 135M: top-1 all / resp / held | KL | ppl (float 15.598) | ident | 360M: top-1 all / resp / held | KL | ppl (float 12.373) | ident |
|---|---|---|---|---|---|---|---|---|
| r15 (§5.1) | 0.9689 / 0.9808 / 0.9687 | 0.0025 | 15.611 | 2 | 0.9763 / 0.9936 / 0.9707 | 0.0018 | 12.392 | 8 |
| r15 + inorm | 0.9677 / 0.9788 / 0.9668 | 0.0024 | 15.603 | 3 | 0.9700 / 0.9911 / 0.9756 | 0.0018 | 12.379 | 7 |
| r15 + irope | 0.9671 / 0.9788 / 0.9677 | 0.0024 | 15.623 | 4 | 0.9679 / 0.9924 / 0.9752 | 0.0018 | 12.398 | 7 |
| r15 + isilu | 0.9695 / 0.9808 / 0.9674 | 0.0024 | 15.609 | 4 | 0.9603 / 0.9924 / 0.9733 | 0.0019 | 12.387 | 7 |
| r15 + **iops** (all three) | 0.9695 / 0.9828 / 0.9661 | 0.0024 | 15.605 | 3 | 0.9596 / 0.9911 / 0.9752 | 0.0018 | 12.404 | 6 |
| r15 + **iops**, the final specs (Bq' << (g + 4), the {T, ΔT} sigmoid table; rerun 2026-10-11 evening) | 0.9689 / 0.9818 / 0.9684 | 0.0024 | 15.613 | 6 | 0.9533 / 0.9911 / 0.9707 | 0.0018 | 12.396 | 6 |

Pass: KL and perplexity inside the shipped policy's own spread
(135M 15.603–15.623 around 15.611; 360M 12.379–12.404 around 12.392),
held-out top-1 within 0.5 points either way, the accumulator peaks
unchanged, no saturation.  The 360M "all" column (0.9596 with every spec)
is the long prompt's greedy path diverging at a different token — its
responses column is 0.9911 against 0.9936, its held-out column higher
than the float-residual policy's.

**Verdict of phase 0's numerics: every op of §2 has an integer
specification that passes every model's gate** — the int32 residual at
2^−15, the norms, RoPE and SiLU·up on SmolLM2-135M / 360M and SmolVLM, the
LayerNorms on BERT and the Piper encoder.  The unit needs no floating-point
operator.  What remains of phase 0 is engineering, not numerics: the
scheduler's references switched to the specs (phase 2), the C fallbacks
rewritten to the same algorithms, and the decode host-overhead measurement
on the board (§1: 141 µs per SiLU·up call on a cache-cold 512 KB table,
193 µs per attention prep, 0.75 ms in the single-threaded logits dequant —
the unit's price for all of it is 0.8 ms per token).

Commands (host only; the chat studies with `.venv-export`, BERT and Piper
with `inference-scheduler/.venv`):

```bash
PY=.venv-export/bin/python; P=pow2+sink+p12+r15
$PY demo/chat/scripts/llm_study.py --assets demo/chat/assets/smollm2-135m-instruct study \
    --policies $P,$P+inorm,$P+irope,$P+isilu,$P+iops --out OUT.json        # 360M: the same
$PY demo/chat/scripts/vlm_study.py --assets demo/chat/assets/smolvlm-256m-instruct study \
    --combos "pow2+p12+vgelu+vsmx+r15/pow2+sink+p12+r15,pow2+p12+vgelu+vsmx+r15+inorm/pow2+sink+p12+r15+iops"
inference-scheduler/.venv/bin/python demo/bert_squad/scripts/bert_study.py --policies sched+vsmx,sched+vsmx+iln --n 60
inference-scheduler/.venv/bin/python demo/tts/scripts/piper_study.py encoder-vsmx
(cd inference-scheduler && .venv/bin/python -m pytest test/test_rowops_spec.py -q && .venv/bin/python perf_calibrate.py rowops)
```

## 6. Scheduler and runtime

- **A fifth lane** `KERNEL_ROWOPS` in `KERNEL_REGISTRY` / `kernel_id_t`,
  `run_rowops()` helpers (non-blocking, as the others), the event stream
  and liveness unchanged in kind; `kernel_wait` covers it.  The host ops of
  §2 become kernel nodes behind a platform capability
  (`kernels.rowops`, `AXI_ROWOPS=0` for bitstreams without the unit — the
  pattern of `kernels.vectorop.activations` / `softmax`): the node keeps
  its `reference()` (N1) or gets the unit's spec (N2), `kernel_calls()`
  lists its register values (`src/perf_calls.py`) so the performance model
  and `--plan` price it, the C helper stays as the host fallback.
- **DMA int32** — the real work: the residual tensors (`host = "f32"`
  today) that a rowops node reads or writes move into the DMA pool as
  int32 at 2^−15 (a `TensorInfo` storage kind "DMA i32 at exponent":
  4 bytes per element, 64-byte-aligned slots, the pool-slot colouring
  unchanged; `inference_buf_t` already carries byte counts) — with the
  cache-coherency contract applied to them (they are written by the unit
  and read by the remaining host ops: invalidate after the lane drains,
  flush before the unit reads what the host wrote).  The residual stream
  of a 256-token SmolLM2-360M prefill is 256 × 960 × 4 = 1 MB per tensor;
  the pool grows by the live residual intermediates (≤ a few MB), not by
  the weights.  The Llama / ViT frontends' `numeric.py` metadata gains the
  kind and the residual exponent; `test_cache_coherency.py` audits it; the
  host ops that stay (`LlmEmbed`, `LlmSelectRow`, the stereo ops) read
  and write the int32 form.
- **The parameter rows** (gamma, beta, the per-column exponents) are
  constant tensors of the weight pool (f32 and int8 kinds, encoded by
  `tensor.py` as today's host tables are, `HostTable`).
- **Tests**: every op's spec against the reference (`test_rowops_ops.py`),
  `host_emu` with a software model of the unit (the C fallback is that
  model), `test_perf_calls.py` (register values = emitted C), the chat /
  BERT / Piper sched-check scripts bit for bit.
- **Planning**: a `rowops` family in `perf_fit.py` (bytes-linear, like
  `vecop`), `perf_calibrate.py cases` adds the shipped calls, `host.json`
  loses the ops that moved.

## 7. Phases and gates

| phase | what | gate | cost |
|---|---|---|---|
| **0 — spec and host fix** | DONE 2026-10-11: the int32 residual ablation (§5.1); the integer spec of every op (`src/rowops_spec.py`, `test/test_rowops_spec.py`, §5.2) substituted into the study models; the gates on 135M / 360M / SmolVLM / BERT / Piper (the stereo InstanceNorm's rows are H·W long — beyond the unit's 4 096-column row; it stays a host op); `perf_calibrate.py rowops` (the replay of §3 from the repo).  LEFT: the decode host-op overhead measured on the board (one session with the chat server stopped), and whether a host-side fix is worth it before the unit | every op ≤ 1 LSB of the exact function ✓; every model within its yardstick ✓; decode host 15 → ≤ 8 ms (open) | done in a day; one board session left |
| **1 — RTL** | DONE 2026-10-11 — `kernels/rowops_rtl` (`doc/kernels/ROWOPS_RTL_KERNEL.md`): the engine, the seven ops of §2 plus TLOAD (the 64 KiB table image into the UltraRAM tables, once per bitstream), the driver generator, the Verilator testbench against the specification's C++ port (`rowops_ref.h`) and the Python fixtures (`gen_fixtures.py`, 21 cases + `tables.hex`), lint, `synth_rowops_rtl`.  **Bit-exact ✓** (every fixture at fast / random / slow memory timing, 300 random jobs × 4 seeds).  **Synthesis ✓**: WNS +0.025 ns at 3.333 ns, 30 947 LUT, 24 774 FF, 6 RAMB36, 7 URAM, 121 DSP — above the estimate of 12–15 k LUT (the four-lane 64-bit shift / add / round-to-even arithmetic the estimate priced as DSP work), inside the device with the production design (80 % LUT, 137 / 144 BRAM, 55 / 64 URAM; phase 3's in-context build decides 250 MHz).  The road: the first synthesis gave 81 k LUT / 170 k FF (memories as flip-flops, eight sigmoid copies, the P RAM in 32 BRAM) and −22 ns (ROPE's n / HD); memories recoded, a sequential divider and a four-lane MULACT gave 42 k LUT / 32 BRAM / −4.8 ns; the decisions then taken — **four lanes for every op** (ADDX, MULACT, DEQUANT and NORM lose nothing against the port; COPY, ROPE and TRANSPOSE's reads run at 2 GB/s: COPY 256 × 576 74 → 148 µs, RoPE 118 → 192 µs, the movers ≤ 8 % of any model) and **the tables in UltraRAM** (an UltraRAM has no initial contents: TLOAD loads the sigmoid and rsqrt tables from DDR, 33 µs once per bitstream; the parameter RAM moved there too, BRAM 32 → 6) — with the two-stage rounding, the DSP pipeline registers, `ro_mul32` in the NORM scalar unit and the transposer's precomputed block step gave 31 k LUT and timing closure.  Cycle counts of the shipped shapes: ROWOPS_RTL_KERNEL §3 | bit-exact on every case ✓; WNS ≥ 0 at 300 MHz ✓ (+0.025); resources ≤ §4's estimate ✗ (LUT 2 ×, DSP 1.2 ×; BRAM 1.5 ×, URAM 7 of 8) — accepted: the device holds it | one day |
| **2 — scheduler** | DONE 2026-10-12 (one day, not the 1–2 weeks: the int32 residual of phase 0 replaced "DMA f32") — **the lane**: `KERNEL_REGISTRY["RowOpsKernel"]` (UIO `RowOpsKernel_0`, 0xA004_0000, the generated driver), `KERNEL_ROWOPS` in `kernel_id_t`, the `run_rowops()` helper (byte strides, NULL → address 0), the 64 KiB table image as the weight tensor `rowops_tables` loaded by TLOAD in `inference_init()`, `perf_calls.FIELDS` / `perf_model.family() == "rowops"`, the timeline lane, `perf_calibrate` sizing; **the capability** `platforms/<p>.json kernels.rowops.unit` (`src/_rowops_hw_config.py`, `AXI_ROWOPS`; the KV260 says false until phase 3); **the int32 residual**: `axi.numeric` `dma: i32` (`TensorInfo.dma`, twice the pool elements), host ops take `int32_t *` buffers with the kernel-buffer syncs (`LlmEmbed` / `llm_embed32`, `LlmSelectRow`, `VitEmbedAdd` / `vit_embed_add32`, `LlmAttnPrep`, `LlmDequant`); **the nodes** (`src/rowops_nodes.py`, the factories merged over the host ops in `graph.py`): `LayerNormRoNode` (ONNX `LayerNormalization`, aligned rows ≤ 4 096, else the host op — BERT's 25), `ResAddRoNode` / `RMSNormRoNode` / `SiluMulRoNode` (`llama.py rowops=True`: `res_bits = 15` + `inorm` + `isilu`), `ResAddBiasRoNode` / `LayerNorm32RoNode` (`vit.py rowops=True`: the bias rounded once at 2^-15 — `rowops_spec.quant_bias`, the study's `ibias`; `piper.py rowops=True`: `TtsResAdd` at max(f_x, f_y) + `TtsLayerNorm`), each with the spec's reference, `kernel_calls()` and the per-column parameter block (`P_BCAST` when every column agrees; β at `beta_bits`: 2^-20, coarser only for |β| ≥ 2, which 6 of BERT's LayerNorms need); `LlmAttnPrep` / `LlmAttnMerge` / `LlmDequant` stay host ops (the four-lane ROPE / TRANSPOSE would not beat them; the dequant writes a host output); **the studies**: `llm_study` `pow2+sink+p12(+vsmx)+rowops`, `vlm_study` `pow2+p12+vgelu+vsmx+rowops`, `bert_study` `+iln`, `piper_vits` `encoder_forward(rowops_unit=True)`, and `--rowops auto / on / off` on `generate_llm_project.py`, `generate_tts_project.py`, the sched-checks; **`host_emu`** carries the unit's C model (`xrowopskernel.h`: every op, TLOAD, the tail-word writes) | every suite ✓ (1 789 scheduler tests, the chat tests, ruff, facts); the sched-checks bit for bit ✓ — BERT 3 / 3, SmolLM2-135M 3 / 3 (`AXI_ROWOPS=1 llm_sched_check.py --rowops on`), SmolVLM image features + text (`vlm_sched_check.py --rowops on`); the generated projects equal the simulation on the host emulation ✓ (test_rowops_unit, test_vit `TestVisionRowops`, test_piper `TestPiperEncoderRowops`); the libraries regenerated — after phase 3 | one day |
| **3 — hardware** | DONE 2026-10-12 (§9): `hw/cormorant_hw_128/scripts/bd_rowops_hp0.tcl` (the kernel, `axi_rowops_intercon` with register slices on both sides into `S_AXI_HP0_FPD`, SmartConnect M04 at `0xA004_0000`, interrupt `pl_ps_irq0[4]`), the testbench's `ro_classes.svh` (the table load + the 21 fixtures through the PS VIP, byte for byte) — `sim_hw_kv260` 97 / 97; bitstream `1cedaf2ed816`, WNS +0.020 / WHS +0.010 ns, 93 969 LUT (80 %), 136 / 144 BRAM, 55 / 64 URAM, 833 DSP; the board: overlay `fabric_rowops`, 159 / 159 models, SmolLM2-135M / 360M, SmolVLM, Piper and BERT bit-exact, the perf baseline `kv260-1cedaf2ed816.json` (69 cases: 6 of the unit), the performance model `kv260/1cedaf2ed816` with the `rowops` family, the chat server redeployed with the regenerated libraries | WNS ≥ 0 ✓; 159 / 159 and every library bit-exact ✓; prefill_256 0.58 × (gate 0.55 ×) ✗ and BERT 0.79 × (gate 0.75 ×) ✗ — the attention prep / merge still on the host (§9.4) | one day + 2 h 20 min of Vivado |
| **4 — attention prep on the unit** | **4a DONE 2026-10-12** (§10.8: 135M prefill 256 448 → 398 ms, 360M 1 125 → 1 050, bit-exact, the model topped up to 2 073 calls); **4b DONE 2026-10-12** (§10.9: SmolVLM 1.70 → 1.64 s per image, the Piper encoder 146 → 96 ms at 400 ids, bit-exact, 2 131 calls); PLANNED (§10): 4a the Llama prefill prep as ROPE / TRANSPOSE / COPY jobs on the production bitstream (`irope` joins the policy, `qk_kw = 1`, the V image's boundary blocks as R = 1 granules), 4b the vision / Piper prep (`iqkv`) and `TtsRowPrep` as windowed COPYs, 4c the RTL tweaks only on evidence (ROPE B reuse, TRANSPOSE emit rate, descriptor chains) | 135M prefill 256 ≤ 427 ms, 360M ≤ 1 030, SmolVLM ≤ 1.60 s, Piper encode 400 ≤ 110 ms, all bit-exact | 3 + 2 (+ 3) days |
| **5 — the rest** | TtsPrep onto the unit (77 ms of the chunk): op 9 FOLD, §11 — DONE 2026-10-15 (5a, bitstream `de968ccc45da`); the merge gone through a ConvKernel y channel stride (§10.4); sigmoid as act 7 with the next VectorOPKernel bitstream (§4.3); the perf-regression and docs-audit passes | | as they come |

The decode step gets its 10 % from phase 0's host fix as much as from the
unit; the prefill, BERT and the Piper encoder need the unit.

## 8. Verdict and order

1. **GO for phase 0 now.**  It is host-only, it settles the numerics
   route per op (the one decision that can sink the rest), it gives the
   replay a home in `perf_calibrate.py`, and its decode fix pays on its
   own.
2. **The unit after that**, in the order 1 → 2 → 3, with the prefill as
   the headline: SmolLM2-135M prefill_256 ~690 → ~360 ms, SmolLM2-360M
   1.64 → 0.97 s, BERT 366 → 265 ms, the Piper encoder 159 → 49 ms, all
   by the replay of §3.  The vision pass gains 13 %, the decode steps
   14–23 % (overhead the host fix of phase 0 shares).
3. **Not a separate DMA block, not a sigmoid bitstream**: the mover is the
   unit's address generator (§4.2), sigmoid rides the next VectorOPKernel
   build (§4.3).

Commands (host only, from `inference-scheduler/`, ~10 min with the
SmolLM2-360M graphs; `TMPDIR=/mnt/data/cormorant_repro/tmp`): the host-op
pricing of §1 iterates `perf_calibrate.SHIPPED` through `_entry_graphs`,
`HostModel.load(default_host_model_path()).us(sn)` and
`PerfModel.load(...)`; the replay of §3 is `codegen.timing.simulate(cg,
kernel_duration_fn(pm, cg._layouts[, keys_of]), host_us)` with `host_us`
returning the unit price for the kinds of §2 and `HostModel.us` otherwise —
phase 0 makes it `perf_calibrate.py simulate --rowops`.

## 9. Phase 3 results (2026-10-12)

### 9.1 The design

`bd_rowops_hp0.tcl` adds the kernel to the block design: its one read + write
port through `axi_rowops_intercon` (one SI, one MI, 128-bit, register slices
on both sides as HPC0's crossbar has) into `S_AXI_HP0_FPD` (SAXIGP2, 128-bit,
the FPD switch → DDRC port 3), its control block as the SmartConnect's fifth
master at `0xA004_0000`, its interrupt on `ilconcat_0` In4 (`pl_ps_irq0[4]`,
device-tree node `fabric_rowops@a0040000`, `dts/kv260/cormorant.dts`).  The
other kernels and ports are untouched.

| | `f70ce632a97a` | `1cedaf2ed816` |
|---|---:|---:|
| WNS / WHS at 4 ns | +0.098 / +0.010 ns | **+0.020 / +0.010 ns** |
| CLB LUTs | 62 618 (53.5 %) | 93 969 (80.2 %) |
| CLB registers | 87 832 | 113 020 |
| BRAM tiles / URAM / DSP | 131 / 48 / 712 | 136 / 55 / 833 |

Synthesis 865 s, the build 8 217 s.  The whole-design simulation
(`sim_hw_kv260`) runs the unit's 21 checked-in fixtures after its table load
through the PS VIP's DDR model on HP0 — 97 / 97 with the four kernels' 75.
Two things it found: the PS VIP's DDRC write path races and clobbers the
bytes just past a burst (the previous transfer's byte count applied at the
new address — a 32-byte word of zeros or stale data after a 48-byte COPY
row), which the other kernels' scoreboards never see because they read only
the written positions; the testbench now restores every byte the racing
write can touch from a shadow of everything written (`tb_infra.svh`).  And
`ro_xpose`'s contiguous packer started a job with whatever its accumulator
held — zero in Verilator and in hardware after power-up, X in a 4-state
simulator and the previous job's leftovers after a flush: it is cleared at
job start now (the Verilator suite and lint unchanged).

### 9.2 The board

| gate | result |
|---|---|
| overlay, ports, registers | `fabric_rowops` as uio8; every PS slave port 128-bit; the unit idle; `n` / `win_hi` written and read back |
| `run_remote_tests` (159 models) | **159 / 159**, every output equal to the simulation — the four BERT fixtures run 7–9 unit calls each (their LayerNorms and the table load) |
| SmolLM2-135M (`llm_board`, the library regenerated with `--rowops`) | logits **bit-exact** 4 prompts × 33 / 33; decode **49.4 ms / token** (54.9), prefill 16 / 64 / 256 **88 / 111 / 448 ms** (155 / 226 / 776) |
| SmolLM2-360M | bit-exact; decode **129.4 ms / token** (137.8), prefill **200 / 257 / 1125 ms** (1701 at 256) |
| SmolVLM-256M | bit-exact; `llm_image` **1.69 s** (2.07), text as the 135M |
| Piper (`tts_board`) | PCM, encode and duration bit-exact; RTF 0.224 / 0.194 (unchanged: the chunk has no unit op); `tts_encode` 20 / 88 / 400 ids **14.2 / 45.5 / 146.0 ms** (15.7 / 50.6 / 162.0) |
| BERT-base (`deploy_and_run`, 50 examples) | board = simulation 3 / 3, EM / F1 88.0 / 90.3 = float; **p50 288 ms** (364) |
| `run_remote_perf` | 69 / 69 (the six RowOps cases below); baseline `kv260-1cedaf2ed816.json` recorded with the MNIST / image / BERT demos |
| the CNN demos | MNIST 0.108 / 1.251 ms, ResNet-18 19.99 ms, MobileNet v1 / v2 20.59 / 17.49 ms — unchanged (no unit op) |
| YOLOv5n, LightStereo-S | 46.8 ms per image (21.4 FPS; 46.9 before), 466.0 ms mean / 464.9 p50 per pair (2.15 FPS; 469 before), both bit-exact — no unit op in their graphs |

The unit's calls run at the rate the Verilator cycle counts predicted (§3 of
ROWOPS_RTL_KERNEL.md), DDR behind HP0 included:

| call (board, 250 MHz) | board | modelled |
|---|---:|---:|
| NORM LN 256 × 768 i16 (BERT's LayerNorm; `run_remote_perf` LN-768x256) | 206 µs | 202 µs |
| NORM RMS 256 × 576 i32 → i16 (RMS-i32-576x256; the 135M's `norm_in` 155 µs) | 157 µs | 152 µs |
| ADDX 256 × 576 i32 + i16 (the residual add; `res_attn` 232 µs) | 234 µs | 222 µs |
| MULACT 256 × 1536 (SiLU · up; `silu_mul` 416 µs) | 418 µs | 400 µs |
| COPY 1024 × 576 i16 | 595 µs | 4 × 148 µs |
| the one-row calls of a decode step (norm / add / SiLU·up) | 6–11 µs | the per-call floor |

The performance model `kv260/1cedaf2ed816` (`perf_calibrate.py`: 1 530
cases, two passes of about 2 min, three refinement rounds of +209 / +100 /
+99 → **1 938 exact calls**, repeat spread median 0.065 %) carries a
`rowops` family (the 37 shipped unit calls, train median 6.5 %, no
held-out); the planner uses the exact entries.  The perf-regression baseline
`kv260-1cedaf2ed816.json` holds the 69 cases and the three demos.  The
simulator on this model (`perf_calibrate.py simulate` against the window's
profiles) prices every workload within ±3 %: BERT +0.8 %, SmolLM2-135M
prefill 16 / 64 / 256 −2.9 / +0.9 / −1.1 % and decode +1.9 %, SmolLM2-360M
−0.1 / +0.7 / −1.4 % and +1.6 %, SmolVLM `llm_image` −1.6 %, the Piper chunk
−2.4 % and its encoder buckets within ±1.7 %, the CNNs within ±0.5 % — no
unpriced node.

**The engine cost model** (`tools/fit_cost_model.py fit 1cedaf2ed816`):
the constants as used score ConvKernel 2.8 / 20.3 % (median / p90) over the
campaign's 1 303 calls and MatmulKernel 4.7 / 17.6 % over 414 — the same as
on `f70ce632a97a`, whose refit they are.  The plain refit's proposal scores
2.9 / 19.1 % and would move SmolLM2-135M's and SmolVLM's `prefill_16` 60
MatMuls back from ConvKernel to the GEMV (+1.3 ms per run each by the
measured entries; `diff` with the decision files rebuilt — the ones cached
from 2026-10-11 replay the old comparison), nothing else, so it is not
taken; only the call floor moves, `CALL_OVERHEAD` 443 → 440 cycles with
`ONE` / `one` +3.  The libraries of this window carry the `f70ce632a97a`
refit's choices, the `prefill_16` MatMuls on ConvKernel, which closes that
open item of DEPTHWISE_PLAN §8.6.

### 9.3 Where the time went

SmolLM2-135M prefill 256 by op (the board profile, 30 layers): MatMul
linears 240 ms, the attention softmax on VectorOPKernel's unit 66 ms (90
calls of 733 µs), **attention prep on the host 80 ms (2.67 ms per layer)**,
P·V / q·Kᵀ 76 ms, SiLU·up 12.5 ms, the residual adds 14 ms, the norms 9 ms,
attention merge (host) 5.4 ms.  The 360M's prefill 256: attention prep 142
ms of 1 125; the vision pass: attention prep 144 ms of 1 690.  BERT: the 49
host transposes 29.5 ms of 288, the 25 LayerNorms about 5 ms.  The Piper
encoder at 400 ids: `TtsRowPrep` 57 ms of 146.

### 9.4 Open after phase 3

- **Attention prep / merge on the unit.**  Phase 2 kept `LlmAttnPrep` /
  `LlmAttnMerge` (and `VitAttnPrep`) on the host.  The board says
  otherwise: the host prep costs 2.67 ms per layer for 256 rows where the
  unit's ROPE (192 µs for 256 × 576) + TRANSPOSE (74 µs for the q image) +
  the cache copies would take about 0.45 ms — the 80 / 142 / 144 ms above,
  the 15 % that separates the prefill from the 0.55 × gate.  The numerics
  are the study's `irope` policy; the C side is the `rowops_spec.rope` /
  transposes already in the specification.  **Done as phase 4a, §10.8**
  (448 → 398 ms); the merge stays on the host.
- `TtsRowPrep` (57 ms of the 146 ms encoder at 400 ids) and BERT's
  transposes (29.5 ms) — the TRANSPOSE / COPY ops, phase 4 as planned.
- The decode step's host overhead (§1, the phase-0 item still open): with
  the unit the 135M decode is 49.4 ms of which the kernels' calls are ~22
  ms.
- `neteq_rowops_rtl` (a lockstep job set, `tools/neteq`): the bitstream's
  synthesis log has no multi-driven-signal warning for the unit, the board
  is bit-exact, but the lockstep check the other kernels have is still to
  be written.
- The host-op model (`perf_calibrate.py host --merge` with this window's
  profiles: the ops that moved are gone from the host) is to be refit; the
  engine cost model was re-scored and kept (§9.2).

## 10. Phase 4 plan — attention prep on the unit (2026-10-12)

**Verdict: GO, scheduler and numerics only — the production bitstream
`1cedaf2ed816` already has every op the attention prep needs.**  The host
prep is 80 / 142 / 144 ms of the 135M prefill 256 / the 360M prefill 256 /
the vision pass (§9.3); the jobs below put it at ≈ 0.9 / 1.1 / 4.7 ms per
layer against 2.67 / 4.45 / 11.0 on the host, so the 135M prefill 256 goes
448 → ≈ 395 ms (0.51 × of 776: the 0.55 × gate of §7), the 360M 1 125 →
≈ 1 010, the vision pass 1.69 → ≈ 1.60 s.  No RTL change is required
(§10.2); the RTL tweaks of §10.6 are a second bitstream only if the
measured per-layer time misses the budget.

### 10.1 What the host prep does (the contract to keep)

`LlmAttnPrepNode` (`src/llm_nodes.py`), one call per layer and entry, T
rows (16 / 64 / 256; decode T = 1), `pos` and `n` runtime i32 tensors:

- q: `q0 [T][H·HD]` (int16, per-channel exponents) → rotate-half RoPE
  with the float32 cos / sin table rows `pos + t` → round at `f_q[h]` →
  the q·Kᵀ x image `qx [KV][HD / kw][G·T·kw]` of ConvKernel's `1 × kw`
  lowering (`nodes.conv_lowered_b_image`); columns of rows `t ≥ n` are 0.
- k: `k0 [T][KV·HD]` → RoPE → round at `f_k[g]` → the K cache rows
  `[pos, pos + n)`, group-major `[KV][C][HD]`, read in place as the q·Kᵀ
  weight (row-major A, `group_kw = 1`).
- v: `v [T][KV·HD]` → re-round at the calibrated `f_vc[c]` (`vc ≤ v` where
  P·V would overflow at p12, `llm_study.py:992`) → the V cache rows, the
  P·V x image with `K = pv_kw` (4): element `(k, d)` of group g at
  `g·C·HD + ((k / 16K)·16 + k % 16)·HD·K + d·K + (k / 16) % K` — the keys
  of one residue `k % 16` inside a 64-row block interleave at 2 bytes.
- rows `t ≥ n` write no cache row; the flush of the written rows.
- `LlmAttnMergeNode`: a raw 128-byte-row gather of the P·V outputs into
  head order (0.18 ms per layer at T = 256; 1.0 ms in the vision pass).

`VitAttnPrepNode` (SmolVLM's vision, Piper's encoder — Piper reuses it):
no RoPE, `T = C`, `pos = 0`; `q = round((q0 + bq)·2^f_q[h])`, k and v
likewise with their biases, into the same image and cache layouts (one
cache pair per entry, shared by the layers).  The decode step (T = 1)
stays on the host in this phase: nine unit jobs at the 6–11 µs job floor
cost what the 89 µs host prep costs (§10.5).

### 10.2 The jobs — every one an existing op

Per layer and prefill entry (SmolLM2-135M: H 9, KV 3, G 3, HD 64, T 256,
K 4; the cycle counts from `ROWOPS_RTL_KERNEL.md` §3 at 250 MHz, which the
board reproduced in phase 3):

| job | op | geometry | cycles → µs |
|---|---|---|---|
| q RoPE | ROPE, `n = H·HD` = 576, `rows = n`, `aux = HD`; A = q0, B = the int24 cos / sin table at `pos·8·HD` (a C expression in `run_rowops`), P per column (`sh_a = fp − f_x[c]`, `sh_b = fp − f_rot`, `sh_c = fp + 23 − f_q[h]`) → the staging row `qs [T][H·HD]` | 256 × (144 + 32 + 11) + 32 | 47.9 k → 192 |
| q image | TRANSPOSE, one job per KV group: tiles of R = 8 rows × L = HD from `qs` (`b_s1` = the row stride, `a_s1 = 8` rows, `a_s2` = HD: the G heads as blocks), granules at `b_s2 = 2·G·T` (the channel row of the image), `c_s1 = 16`, `c_s2 = 2T` — the scattered mode, 16-byte granules | 3 × 96 tiles × (64 + 128) | 55 k → 222 |
| k RoPE | ROPE, `n = KV·HD` = 192, `rows = n` → `ks [T][KV·HD]` at `f_k[g]` | 256 × (48 + 32 + 11) | 23 k → 93 |
| K cache | COPY, `n = HD`, `rows = n`, `blocks = KV` (`a_s2` = HD, `c_s2 = C·HD`), c at row `pos` | 768 rows × 16 | 12 k → 49 |
| v re-round | COPY with P (`sh_c = f_v[c] − f_vc[c]`: the same round-half-even the host does, bit for bit) → `vs [T][KV·HD]` | 256 × 48 | 12 k → 49 |
| V image | TRANSPOSE per group, §10.3: R = 4 contiguous granules for the 64-row blocks inside `[pos, pos + n)`, R = 1 scattered granules for the two boundary blocks | ≤ 3 × (64 × 96 + 2 × 63 × 128) | 18–67 k → 74–270 |
| (merge) | stays on the host (§10.4) | | |

Total 0.68–0.88 ms of unit time plus 10–12 jobs; the host prep is 2.67 ms
and the merge 0.18.  The 360M (H 15, KV 5): q RoPE 290, q image 370, k
RoPE 126, K cache 82, v 82, V image 125–450 µs → 1.1–1.4 ms against 4.45.
The vision pass (H = KV = 12, G 1, HD 64, T = C = 1 024, no RoPE): q and
k and v as COPY jobs with bias and rounding (`Bq'` per column, §10.4's
`iqkv` spec), 786 µs each (1 024 rows × 96 words × 2 cycles — the i16
stream is datapath-bound at 4 elements per cycle), k straight into the K
cache (`blocks = H`), the q image 12 × 128 tiles × 192 = 1.18 ms, the V
image 12 × 256 × 96 = 1.18 ms (R = 4, every block aligned: `pos = 0`) →
4.7 ms against 11.0; Piper's encoder at 400 ids (H = KV = 2, HD 96,
`pv_kw = 1`: the V cache is plain rows, a COPY) ≈ 0.3 ms against 0.9.

**The q·Kᵀ image at `kw = 1`.**  The TRANSPOSE writes granules of one
column; the `kw = 2` image interleaves two head dimensions at 2 bytes
(`x_g[c][2p + j]` ← dims `(c / 16)·32 + j·16 + c % 16`), which no
existing op writes at a useful rate (R = 1 granules: 2 cycles each, 1.2 ms
per layer).  With the unit's prep the frontend sets `qk_kw = 1`
(`LlamaFrontend.qk_kw`; decode uses 1 already): `cost_model.conv_board_cycles`
prices the q·Kᵀ conv at 490 / 34 / 13 µs per group for T = 256 / 64 / 16
at `kw = 1` against 507 / 35 / 13 at `kw = 2` — the shipped choice came
from `choose_qk_kw`'s simulation-model search over 2-D geometries and is
within noise either way (the 90 q·Kᵀ calls are 34 ms of the prefill 256).
The vision pass likewise (2 600 against 2 688 µs per head).

### 10.3 The one design problem: the V cache at an arbitrary `pos`

A V-image granule holds the four keys `16·kk + r` (kk < K) of one residue
inside a 64-row block; a prefill that starts at an arbitrary cache
position (a chat turn after decode steps) writes a block whose other keys
are already in the cache.  The host writes 2-byte elements and never sees
the problem; a contiguous R = 4 TRANSPOSE would read the rows below `pos`
from outside the staging buffer and clobber valid keys.  Options weighed:

- a row-major V mirror per layer (`[C][KV·HD]` int16 at `f_vc`, a DMA
  state the prefill and decode both write): exact and fast (R = 4 for
  every block), but +11.5 MB (135M) / +20 MB (360M) of pool — the 360M's
  CMA budget is 775 MB against its 754 MiB pool: rejected for memory;
- aligning the prefill start to 64 keys: not possible, `pos` is the cache
  length after decode;
- a permuted key order inside the blocks (consecutive keys per tap
  group, so a row's elements are contiguous): breaks the softmax unit's
  causal mask, which keeps a prefix of keys: rejected;
- **R = 1 scattered granules for the two boundary blocks** (2-byte
  writes with byte strobes, 1 granule per 2 cycles: ≤ 63 rows × 128
  cycles per group each) **and R = 4 contiguous granules for the whole
  blocks between them** — the host's exact semantics (rows below `pos`
  and at or beyond `pos + n` untouched), no new state, 74–270 µs per
  layer.  Chosen.  The vision and Piper caches start at 0 and are whole
  (or, Piper at 400 ids, `pv_kw = 1`): R = 4 or a plain COPY throughout.

The node's `reference()` reproduces every byte the unit writes (the
boundary blocks element by element, the whole blocks tile by tile), so
the sched-checks and the board gates compare the cache states as well.

### 10.4 Numerics and the specifications

- **RoPE**: the study's `irope` policy (§5.2: `rowops_spec.rope`, cos /
  sin as int24 at 2^−23, the pair's products at the finer exponent, one
  round-half-even at `f_q` / `f_k`) passed its gate on both models (135M
  top-1 0.967 / 0.979 / 0.968, KL 0.0024, ppl 15.623 against float 15.598;
  360M 0.968 / 0.992 / 0.975, 0.0018, 12.398 against 12.373).  It joins the
  shipped policy: `+rowops` = `res_bits 15 + inorm + isilu + irope`
  (`llm_study.py`, `vlm_study.py`'s text model), re-run once as the
  record.  The libraries' numerics change; every other op is unchanged.
- **The V re-round** is the host's rounding exactly (an integer
  shift-round of the int16 value): no policy change.  A later option,
  `ivc` — the v projection's output exponent set to `f_vc` so the COPY
  disappears — is a one-line ablation in `llm_study.py` (the kernel's
  floor replaces the host's round: a gate, not a given).
- **The vision / Piper bias add** `round((x + b)·2^f)` becomes
  `rne_shr((x << sh_a) + Bq', sh_c)` with the bias rounded once at a fixed
  exponent — the residual's `quant_bias` / `ibias` scheme (§5.2) applied
  to q, k, v: `rowops_spec.bias_round`, the policy `iqkv` in `vlm_study.py`
  and `piper_vits.encoder_forward`, gated like `ibias` was (the vision
  features' relative error and the answer top-1; Piper's PCM bit-exact
  against its own emulation and the log-mel distance).
- The cos / sin tables: `rowops_spec.rope_tables_raw` as one int32 weight
  tensor `[C][2·HD]` per model (512 KB at C 1 024, HD 64; the float32
  tables go), the B stream of both RoPE jobs at `pos·8·HD`.
- The merge stays a host op: a COPY would take 147 µs (2 304 rows × 16
  cycles) against the host's 180 — not worth a job; the real fix is a y
  channel stride on ConvKernel (the P·V output written in head order:
  no merge at all, −5.4 ms per prefill 256, −12 ms per image), a ConvKernel
  register for a later bitstream.

### 10.5 Scheduler, runtime, coherence

- `src/rowops_nodes.py`: `AttnPrepRoNode` (Llama; `LLM_OVERRIDES["LlmAttnPrep"]`,
  taken when the unit is on, `rowops=True` and the entry has `n` — the
  prefill entries; decode keeps `LlmAttnPrepNode`) and `VitAttnPrepRoNode`
  (`VIT_OVERRIDES`, Piper through it).  Each emits its job list in order
  (q RoPE → q image per group; k RoPE → K cache; v → V image per group)
  with `kernel_wait(KERNEL_ROWOPS)` between dependent jobs — the unit is
  one engine, the jobs serialise anyway; `kernel_calls()` lists them all
  (`test_perf_calls` parses `run_rowops`); `rows = n` and the B offset
  `pos·8·HD` as C expressions.
- The staging rows `qs` / `ks` / `vs` are pool intermediates per entry
  (0.5 MB at T = 256); the q image's columns `t ≥ n` must be 0 (the
  host's contract): one COPY with the window `[0, 0)` zeroes `qs` at the
  entry start (49 µs per run, not per layer) and the RoPE writes rows
  `< n` over it, so the TRANSPOSE of all T rows leaves zeros past n.
- The P blocks per layer (q: 576 columns, k: 192; the vision's q / k / v
  biases) are weight tensors as the norms' are (`param_tensor`).
- `LlamaFrontend(qk_kw=1)` when the unit prep is on; `choose_qk_kw` is
  bypassed for those entries.
- **Coherence.**  The unit writes through `S_AXI_HP0_FPD`, which does not
  snoop the CPU caches; ConvKernel reads the K cache through HPC1 and the
  V image through HP2.  Rule 1 (today's): a buffer the unit writes is
  never dirty in a CPU cache — the prep's inputs and outputs are
  kernel-written DMA tensors the host does not touch.  Rule 2 (new): the
  decode step's host prep writes 2-byte V slots into lines the unit
  filled during the prefill, so `llm_cache_flush_rows` invalidates the
  8 lines of each written key row before the write (a stale clean line
  would otherwise be written back over the unit's keys); the K rows are
  whole 128-byte lines and need nothing.
- `perf_calibrate.py`: the `rowops` family's range grows to ROPE,
  TRANSPOSE (both granule modes) and `blocks > 1` — the shipped graphs'
  calls become exact entries through `cases`, and `bench_rowops.c` /
  `_rowops_detail` learn the B stream and the transpose geometry; the
  host model loses `LlmAttnPrepNode` / `VitAttnPrepNode` for the prefill
  entries (`host --merge` with the new profiles — together with the
  phase-3 merge still owed, §9.4).
- `host_emu._ROWOPS` already models ROPE and TRANSPOSE; the fixtures of
  phase 1 cover the V interleave (R = 4 contiguous), the q image (R = 8 at
  stride `2·G·T`) and R = 1 straddling words (`gen_fixtures.py` case 16
  onward) — phase 4 adds the exact job shapes at small T (ROPE with
  `blocks = KV` and a B offset, the K-cache COPY with blocks, the R = 1
  boundary blocks at an unaligned `pos`, the window-zero COPY) so that
  `TestRowOpsRtl`, the block-design testbench (`ro_classes.svh`,
  manifest-driven, images ≤ 64 KiB) and the C model all see them.

### 10.6 If the budget is missed: the RTL tweaks (a second bitstream)

Measured first; each is small and keeps the register map:

1. **ROPE B-row reuse**: the B row (cos / sin, 32 words) is reloaded per
   A row and costs more than the 8–24 words of A; with the job's blocks
   as the inner loop (`b_s1 = 0`) and a "same B address as the previous
   row" check, the load happens once per position: q RoPE 192 → ≈ 85 µs,
   k 93 → ≈ 45.
2. **TRANSPOSE scattered emit at one granule per cycle** (today one per
   two: the AW / W issue of single-beat bursts): the q image 222 → ≈ 130
   µs, the boundary V blocks halve.
3. **Descriptor chains**: a job list in DDR (the 19 registers per
   descriptor) that one Start runs back to back — the 6–11 µs per job
   becomes one issue per layer, which is what the decode step needs (nine
   jobs at the floor ≈ the 89 µs host prep today; chained ≈ 20 µs:
   −2 ms per token, 4 %) and what the 212 unit calls of a decode step sit
   on (§3).
4. The ConvKernel y channel stride (§10.4) — a ConvKernel change, with
   the next ConvKernel bitstream.

All four fit the free resources (≈ 23 k LUT, 8 BRAM, 9 URAM) but the
design's +0.020 ns margin makes every bitstream a timing risk; hence the
order: 4a measures on `1cedaf2ed816` first.

### 10.7 Phases and gates

| phase | what | gate | effort |
|---|---|---|---|
| **4a — Llama prefill prep** | `irope` into `+rowops` (re-gate 135M / 360M); `AttnPrepRoNode` + the job emission, `qk_kw = 1`, the window-zero, the decode invalidate; the fixtures of §10.5; `llm_sched_check --rowops on` bit for bit; the libraries regenerated; the board: `llm_board` gates, profiles, the perf top-up (`cases`, `run --resume`, `fit`), baseline, chat server | fixtures + 300 random + sim_hw 100 %; sched-checks 3 / 3; board bit-exact; **135M prefill 256 ≤ 427 ms** (0.55 × of 776; expected ≈ 395), prefill 16 / 64 and decode not slower; 360M prefill 256 ≤ 1 030; simulate within ±3 % | 3 days (numerics ½, scheduler 1, fixtures ½, board window 1) |
| **4b — vision and Piper prep, TtsRowPrep** | `bias_round` spec + `iqkv` gate; `VitAttnPrepRoNode`; `TtsRowPrep` as three windowed COPY jobs per call (§4.2: the taps are the window; 57 → ≈ 6 ms at 400 ids); `vlm_sched_check`, the TTS emulation; `tts_board`, `llm_board` SmolVLM | bit-exact; **SmolVLM `llm_image` ≤ 1.60 s**; **Piper encode 400 ids ≤ 110 ms**, the chunk unchanged | 2 days |
| **4c — the RTL tweaks of §10.6** | only what 4a / 4b measured short; a bitstream, `sim_hw`, the campaign, the baseline | WNS ≥ 0 at 4 ns; the per-layer prep at the §10.2 budget; decode ≤ 47 ms per token with the chains | 2–3 days incl. the build |

Order: 4a, then 4b; 4c on evidence.  Nothing in 4a / 4b touches the
bitstream, the register map or the other kernels.

### 10.8 Phase 4a results (2026-10-12)

**Numerics.**  `irope` joined the shipped policies as `irope = "prefill"`
(`llm_study.py`: the prefill's RoPE on q and k is the unit's integer one,
the decode steps keep the host's double RoPE — the decode prep stays on the
host).  The study re-run on the final policy `pow2+sink+p12+vsmx+rowops`
(the same prompts, windows and yardsticks as §5.2):

| model | top-1 all / responses / held-out | KL tf / held-out | ppl (float) | identical | time |
|---|---|---|---|---|---|
| SmolLM2-135M | 0.9750 / 0.9818 / 0.9684 | 0.0023 / 0.0025 | 15.600 (15.598) | 5 / 12 | 414 s |
| SmolLM2-360M | 0.9554 / 0.9949 / 0.9762 | 0.0016 / 0.0018 | 12.387 (12.373) | 8 / 12 | 1 h 10 min |

Both inside the shipped policy's own spread (§5.2: KL 0.0024–0.0025 /
0.0018, perplexity 15.607–15.623 / 12.391–12.407; the 360M's "all" column
moves with one long prompt's greedy path, as r12's 0.9589 did, while its
responses, held-out windows and perplexity improve).  No element saturates.

**The scheduler** (`src/rowops_nodes.py` `AttnPrepRoNode`, `from_llm_attn_prep`
in `LLM_OVERRIDES`; `src/llama.py` `rowops=True`: `qk_kw = 1`, the staging
states `<entry>.ro.qs / ks / vs`, the decode nodes' `rowops_cache` attribute,
and `decode_attn "fpga"` required): the jobs of §10.2 as written, the V
image through `vimage_jobs` (Python, the priced calls) and its C twin
`llm_ro_vimage` (emitted after `run_rowops`), `rows = n` and the B offset
`pos · 8 HD` as C expressions, every job but the first waiting for the one
before it, the last left in flight for the schedule's wait.  The cos / sin
table is one weight tensor per model (`rowops_rope_<crc>`, 512 KB at C 1 024).
The decode prep invalidates the V-cache lines it writes 2-byte slots into
(`llm_cache_inval_rows`).  The coherency audit starts a project's DMA
states clean when its prefill prep runs on the unit, and requires them clean
at the end of every run.  `perf_calibrate.buffer_sizes` / `bench_rowops.c`
size TRANSPOSE (the tile's rows, the granule run) and ROPE (the 8 HD-byte B
row) buffers — before, a TRANSPOSE case's C buffer was sized as a row of n
elements, far short of the image it writes.  Three benchmark cases
(`ROPE-576x256-hd64`, `XPOSE-qimage-g3-t256`, `XPOSE-vimage-k4-256`) join
`perf_config.json`.

**Verification.**  27 fixtures (6 new: the window-zero COPY, the K-cache
COPY with blocks, the whole-block and the two boundary-block V transposes,
the q image) — Verilator 100 % with the random cases; `sim_hw_kv260`
103 / 103 (RowOps 28 / 28); the scheduler suite 1 791 tests, ruff clean;
the tiny Llama's prefill (T 16 and 32, an unaligned `pos`) bit for bit on
the host emulation of the unit; `llm_sched_check --rowops on` SmolLM2-135M
3 / 3 prompts bit-exact (prefills at positions 1 and 69: the boundary
blocks); `vlm_sched_check --rowops on` the vision features bit-exact.
The regenerated libraries: 90 / 96 / 90 prep blocks (the 3 prefill entries
x the layers), pools 294 / 758 / 508 MiB (+4 MB: the staging states, the
table).

**The board** (the window of 2026-10-12 13:23 on `1cedaf2ed816`, unchanged:
`llm_board.py --profile` gates, the perf-model top-up, the 72-case benchmark):

| | phase 3 | phase 4a | gate (§10.7) |
|---|---|---|---|
| SmolLM2-135M prefill 16 / 64 / 256 | 88 / 111 / 448 ms | **81 / 98 / 398 ms** | ≤ 427 ✓ — 0.51 × of the 776 ms start |
| SmolLM2-135M decode | 49.4 ms / token | 49.4 | not slower ✓ |
| SmolLM2-360M prefill 16 / 64 / 256 | 200 / 257 / 1 125 ms | **191 / 236 / 1 050 ms** | ≤ 1 030 ✗ by 2 % (0.62 × of 1 701) |
| SmolLM2-360M decode | 129.4 ms / token | 129.5 | ✓ |
| SmolVLM text prefill 256 / decode / `llm_image` | 449 ms / 49.1 / 1.69 s | 399 / 49.2 / 1.70 s | the image pass is phase 4b |

Every library bit-exact (SmolLM2 4 prompts × 33 / 33 logits, SmolVLM 2
images × 33 / 33).  The prep on the unit per layer, from the board profiles:
815 µs at T 256 (the host 2 670), 98 µs at T 16 (343); the 360M 1 290 µs
(4 450) — inside the §10.2 budget (0.68–0.88 ms) for the 135M; the 360M's
1 030 ms gate was 2 % too tight (its prep is 41 of the 1 050 ms, the
linears 780).  The benchmark cases measure what §10.2 modelled:
`ROPE-576x256-hd64` 196 µs (192), `XPOSE-qimage-g3-t256` 78 (74),
`XPOSE-vimage-k4-256` 29 (25); 72 / 72, the baseline re-recorded (one
PoolingKernel case +3 % on the first run — the jitter of a 93 µs call).
The performance model: +90 shipped calls, three refinement rounds (+184,
+24, +8) → 2 073 exact calls, the `rowops` family 79 calls (train median
7.8 %).  `simulate` prices every prefill 2.0–4.3 % low (phase 3: within
±1.5 %) and decode 0.9–2.0 % high, nothing unpriced: the 10–15 unit jobs
per layer cost ≈ 40 µs each beyond the calibration's back-to-back loop —
the issue after a wait; a per-lane issue constant for the timing model is
open.  The chat server is redeployed with the three libraries.

### 10.9 Phase 4b results (2026-10-12)

**The vision and Piper encoder prep, the TTS row prep.**  `VitAttnPrepRoNode`
(`VIT_OVERRIDES["VitAttnPrep"]`, `src/vit.py` / `src/piper.py` rowops=True:
`qk_kw = 1`, the staging states `vision.ro.qs / vs`, `enc<T>.ro.qs / vs`): q0 +
bq and v + bv as COPYs with the bias in the parameter block (the new
`rowops_spec.bias_round`: Bq' = round(b·2^fp) at fp = `bias_exp` — max(f_x,
f_out) + 8, at most 24, lowered where the int22 would overflow — then one
round-half-even at f_out), one TRANSPOSE of the q rows into every head's image
(the heads as blocks), k0 + bk per head straight into the K cache (its own
parameter block), the V image per head as the whole-block TRANSPOSEs (T = C
is a multiple of 16K in every bucket: pos 0, no boundary rows) or, at
`pv_kw` 1 (Piper's 400-id bucket), one COPY with the heads as blocks: 3 + 2H
jobs per layer, all static.  `TtsRowPrepRoNode` (`TTS_OVERRIDES["TtsRowPrep"]`):
a COPY per tap of the rows it reads (`rows = n`, the output shifted by the
tap's offset) with the bias and `RELU` in the parameter block, the rows the
host zeroes — the first tap's row 0, the last tap's row n − 1, every row ≥ n
— as window-[0, 0) COPYs: 6 jobs for 3 taps, 2 for one.  `TtsAttnRelAdd`,
which recomputes q from q0 + bq, now does so with the unit's integer formula
(`rowops = 1`: the parameter arrays in the C) — the one place the host had
to follow the unit's numerics.

**Numerics.**  The policies gained `iqkv` (`vlm_study.py`
`pow2+p12+vgelu+vsmx+rowops`; `piper_vits.encoder_forward(rowops_unit=True)`
and `piper_study.py encoder-vsmx`'s "VectorOP softmax + unit" policy):

| gate | before (the unit's norms alone) | with the prep on the unit |
|---|---|---|
| SmolVLM, 24 images: features rel (max) / cos min / answer top-1 / KL / identical / ppl | 0.0548 (0.0690) / 0.99787 / 0.9712 / 0.0033 / 8 of 24 / 12.208 | identical to the digit |
| Piper encoder, 1 138 ids: durations changed vs float / log-mel distance / max \|x − float\| | 4 (0.35 %, 1 frame) / 0.362 dB / 0.0264 | 2 (0.18 %) / 0.372 dB / 0.0274 (the host: 4 / 0.388 / 0.0318) |

**Verification.**  28 fixtures (+ the per-column bias + ReLU COPY), Verilator
and `sim_hw_kv260` 104 / 104 (RowOps 29 / 29); the suite 1 791, ruff clean;
`test_vit` / `test_piper` against the study models (the Piper encoder's
emulation now carries the integer prep); `vlm_sched_check --rowops on --text`
vision and text bit-exact.

**The board** (the window of 2026-10-12 14:46 on `1cedaf2ed816`):

| | phase 4a | phase 4b | gate (§10.7) |
|---|---|---|---|
| SmolVLM `llm_image` | 1.70 s | **1.64 s** | ≤ 1.60 ✗ by 2.5 % |
| Piper encode 20 / 60 / 88 / 162 / 268 / 400 ids | 14.2 / 23.6 / 45.5 / 89.8 / 143.5 / 146.0 ms | **9.5 / 14.4 / 28.4 / 56.8 / 92.1 / 95.8 ms** | ≤ 110 ✓ |
| Piper chunk, RTF | 259 ms, 0.224 / 0.194 | 260 ms, 0.224 / 0.194 | unchanged ✓ |

Every output bit-exact (SmolVLM 2 images × 33 / 33 logits; Piper's encoder,
durations and PCM).  The vision prep on the unit costs 4.81 ms per layer
(the host 12.8; §10.2 said 4.7) — but the q·Kᵀ conv at `kw = 1` costs 3.86
ms per head-call against 3.69 at `kw = 2` (+24 ms per image; the cost model
had priced `kw = 1` cheaper) and the softmax +15 ms, so the image gains 55
of the 96 ms the prep saved.  The TRANSPOSE kw-interleave of §10.6 (a second
bitstream) would give the 24 ms back; the gate stays open by it.  The
performance model: +58 shipped calls, one refinement round +179 → 2 131
exact calls, the `rowops` family 131 calls (train median 3.1 %); the
benchmark 72 / 72, the baseline re-recorded (PASS: two cases improved by the
usual jitter); `simulate`: the image pass −2.3 %, the Piper encoder buckets
−0.0…−3.6 %, the chunk −2.3 %, the text prefills −3.2…−4.2 % as in §10.8.
The chat server is redeployed with both libraries.

## 11. Phase 5a plan — TtsPrep on the unit: FOLD, op 9 (2026-10-14)

**Verdict: GO — a new op and a bitstream.**  `TtsPrep` is the Piper
chunk's largest host op: 77 ms of the 260 ms chunk on the board (RPU_PLAN
§1), 80.7 ms by the host model's 46 entries.  None of the unit's ops fits
it (a 2-D windowed gather at unaligned element offsets, a runtime column
window, LeakyReLU and a re-rounding per element), so it is one new op —
FOLD — with its own row sequencer and an 8-lane datapath, ≈ 11–14 ms for
the chunk's 41 int16 calls: the chunk 260 → ≈ 195 ms, RTF 0.194 → ≈ 0.15,
and 77 ms of A53 time per chunk freed.

### 11.1 What `TtsPrep` does (the contract to keep)

`TtsPrepNode` (`src/tts_nodes.py`): output `y [nch][rows][W]` int16 at
f_out, `W = w0 + hl + hr` (contiguous: rows start at any element, `W · 2`
bytes apart); element `(k, r, j)` reads source channel `ch0 + k`
(`reverse`: `ch0 + nch − 1 − k`) at column `s = r·w0 + j − hl + t0` of a
`[C][L]` int16 (or float32) tensor at f_in, LeakyReLU with `alpha`, rounded
half to even at f_out and saturated — or 0 where `s` lies outside `[slo,
shi)`: the columns inside `[wlo, whi) ∩ [0, L)` and inside the utterance,
`lo · rate ≤ s + off · rate < hi · rate`, with `lo` / `hi` runtime int32
graph inputs (the C helper `tts_prep` clamps them into `[slo, shi)` once
per call).

The chunk's 46 calls (`generate_tts_project.entry_graphs`, the host
model's prices):

| source | calls | W | rows | alpha | f_in − f_out | host |
|---|---:|---|---:|---|---|---:|
| `[32][49152]` i16 (the last upsampling stage) | 8 | 2050 … 4102 | 24 / 12 | 0.1 (0.01 for the last) | 0 … 2 | 44.6 ms |
| `[64][12288]` i16 | 8 | 514 … 1060 | 12 / 24 | 0.1 | 1, 2 | 24.5 ms |
| `[128][1536]` i16 | 8 | 130 … 548 | 3 / 12 | 0.1 | 1 … 3 | 6.4 ms |
| `[192][256]` i16 (the flow's couplings) | 16 | 132 | 2 | 1 | 4, 5 | 4.2 ms |
| `[256][192]` i16 | 1 | 66 | 3 | 0.1 | 3 | 0.3 ms |
| float32 sources (the flow's reversed halves, the encoder output) | 5 | 70, 256 | 1 / 3 | 1 | — | 0.8 ms |

The 41 int16 calls write 21.5 M elements per chunk.  The five float32 calls
stay on the host.

### 11.2 The exact rule

The C computes `v = x · 2^-f_in` (exact), `alpha · v` in double (one
rounding) for `v ≤ 0`, then `nearbyint(· 2^f_out)`.  With `s = f_in −
f_out` and `m = 1 / alpha`, that is **`rne(x / (m_x · 2^s))` with `m_x = m`
for x < 0 and 1 for x ≥ 0** — checked over every negative int16 for alpha
0.1 (m 10) and 0.01 (m 100) and s −2 … 8: 0 mismatches of 32 768 each.  It
is not the exact product with the double constant (0.1 is not 1 / 10 in
binary; that product differs from the C on some inputs): the double's single
rounding happens to land on the exact quotient.  So the op is specified with
an integer divisor, and the scheduler checks each call's (alpha, s) over all
65 536 inputs at generation time (`rowops_spec.fold_divisor`): a pair that
fails stays on the host.  Positive x is a plain `rne_shr(x, s)`; for s ≥ 0
no value saturates.

The datapath divides by multiplying: with `R = ceil(2^24 / m)` (2^24 for
x ≥ 0), `u = 2|x|` and `P = u · R`, `q1 = P >> 24` is `floor(u / m)` and
`P mod 2^24 < R` says the remainder is 0 — exact for every |x| ≤ 32 768 and
m ≤ 127 (the error of R stays below 1 / m over that range); the round at
`s + 1` bits (u carries one more) is then a shift with round-to-even
decided from the low bits and that remainder flag, and the sign comes back
last.  The spec states both forms; its test runs the reciprocal form
against the quotient for every m 1 … 127, s 0 … 15 and every int16.

### 11.3 The op

| register | FOLD |
|---|---|
| `op` | [3:0] = 9 (A and C int16; kinds and flags ignored) |
| `n` | W, the output row (1 … 65 535) |
| `c1` / `c2` | rows per channel / channels |
| `a` | the byte address of source column 0 of the first output channel (2-byte aligned) |
| `a_s2` | the bytes from one output channel's source row to the next (signed, even: reversal is a negative stride) |
| `b_s1` | w0, the source columns between rows |
| `b_s2` | `s0 = t0 − hl`, the source column of row 0, column 0 (signed) |
| `c` | the output, one contiguous run of `ceil(c2·c1·n / 8)` words (16-byte aligned; the last word's lanes past the end 0) |
| `aux` | [3:0] s (0 … 15), [10:4] m (1 … 127) |
| `win_lo` / `win_hi` | `[slo, shi)`: the source columns that are read; every other column is 0; the unit never reads outside it |

Row `(k, r)` covers source columns `[base, base + W)`, `base = s0 + r·w0`;
the valid part `[max(slo, base), min(shi, base + W))` is one read of whole
words at element address `a / 2 + k · a_s2 / 2 + column` — any alignment.

- **Reads**: a row sequencer (64-bit element addresses, a few cycles per
  row, running ahead through a descriptor FIFO) gives each row's word range
  to stream A's read port through a new **external-run input**: the port's
  burst generator is restarted per run (the 4 KiB and 64-beat splits, the
  credits and the outstanding count unchanged); a row with no valid column
  reads nothing.
- **Datapath, eight 16-bit lanes** (a word per cycle; at four lanes, the
  unit's width for the other ops, FOLD would take 21 ms instead of 11): a
  two-word window per row slides one word per group of 8 output columns, lane
  l takes window element `δ + l` (δ = the row's start mod 8), the columns
  outside the valid part are zeroed (group / lane masks from the row's
  descriptor), the fold of §11.2 per lane (one DSP for `u · R`), then a
  packer appends each group (8 columns, the row's last W mod 8) to the
  contiguous output.  One group per cycle; a row of W columns costs
  `ceil(W / 8)` cycles plus one to fill the window.
- **Core**: R from the ROPE divider (`(2^24 + m − 1) / m`, 32 cycles in
  T_CFG2), the output's word count from `rows · blocks · n`; no parameter
  block.  Every counter is sized for its register; m = 0 or n = 0 is an
  empty job.
- **Cost**: ≈ 2.5–3 k LUT, ≈ 2 k FF, 8 DSP, no BRAM / URAM.  The design is
  at 80.2 % LUT and closes 4 ns by +0.020 ns: in-context timing is the risk
  (a register between every stage of the sequencer, the lane muxes and the
  packer).

### 11.4 What it buys

Cycles ≈ the words read (one per cycle through the port): Σ rows ·
ceil((δ + W) / 8) ≈ 2.7 M for the chunk = 10.7 ms at 250 MHz; at the
read + write rate ADDX reaches on HP0 (6.3 GB/s both ways, §9.2) ≈ 13.5 ms
— against 80 ms on the host.  The chunk 260 → ≈ 195 ms (RTF 0.194 →
≈ 0.15); the chunk's A53 time 139 → ≈ 62 ms.  Nothing else in the shipped
models uses TtsPrep.

### 11.5 Scheduler

- **Capability** `platforms/kv260.json kernels.rowops.fold` (false until a
  bitstream with op 9 is in production; `AXI_ROWOPS_FOLD=0 / 1`), read by
  `src/_rowops_hw_config.py`.
- **`FoldRoNode`** (`TTS_OVERRIDES["TtsPrep"]`): int16 source, `reverse` or
  not, `fold_divisor(alpha, s)` defined, `s ≥ 0`, W ≤ 65 535 — else the host
  op.  Its C computes `slo` / `shi` from `lo` / `hi` as `tts_prep` does and
  starts one `run_rowops()`; the reference is `rowops_spec.fold`, equal to
  `TtsPrepNode.reference` by §11.2 (tested on every chunk call);
  `kernel_calls()` with the full window.
- **Host emulation**: the unit's C model (`xrowopskernel.h` of `host_emu`)
  gains op 9; the Piper chunk on the emulation equals the simulation.

### 11.6 Phases and gates

| phase | what | gate |
|---|---|---|
| **5a.1 — spec + scheduler** (host only) | `rowops_spec.fold` / `fold_recip` / `fold_divisor`, `FoldRoNode`, the capability, the C model, the tests | fold = TtsPrep's reference on all 41 int16 calls (random + edge windows); the reciprocal form = the quotient exhaustively; the Piper chunk on `host_emu` = the simulation; every suite |
| **5a.2 — RTL** | `ro_fold.sv`, the read port's external runs, op 9 in `ro_core`, the C++ oracle, fixtures (W not a multiple of 8, halos, empty / partial windows, negative stride, m 1 / 10 / 100, s 0 … 5), random jobs, `--perf` shapes | fixtures + 300 random × 4 seeds at every memory timing; lint clean; WNS ≥ 0 at 300 MHz out of context |
| **5a.3 — bitstream** | `build_hw_kv260`, the block-design testbench's FOLD fixtures | WNS ≥ 0 at 4 ns; `sim_hw_kv260` 100 % |
| **5a.4 — board** | load, the 159 models, every library bit-exact (the chat ones unchanged), the Piper gate with FOLD, the perf campaign and baseline for the new id | bit-exact; **Piper chunk ≤ 205 ms**; simulate within ±3 % |
| **5a.5 — production** | `pl.bin` and the archive, `kernels.rowops.fold = true`, the libraries regenerated, the chat server, docs and facts | the chat server healthy on the new bitstream |

### 11.7 Phase 5a results (2026-10-14)

**5a.1 — spec and scheduler.**  `rowops_spec.fold` / `fold_elem` (the
quotient), `fold_recip` (the datapath's form: equal to the quotient for
every int16, m 1 … 127, s 0 … 15 — 133 M cases), `fold_divisor` (accepts
alpha 1, 0.1, 0.01 at s 0 … 8; rejects 0.3, m > 127, s outside 0 … 15) and
`fold_aux`; `test_rowops_spec.py TestFold` checks fold against TtsPrep's
double arithmetic on the chunk's shapes with windows inside, across and
outside the rows.  `FoldRoNode` (`TTS_OVERRIDES["TtsPrep"]`) where
`kernels.rowops.fold` (false until the bitstream is in production;
`AXI_ROWOPS_FOLD=1` generates for it): on the real voice **all 41 int16
calls** (21.5 M elements; m 1 / 10 / 100, s 0 … 5), the five float32 ones
stay host ops.  The chunk's simulation equals `piper_vits.chunk_forward`
at five utterance windows, the generated C equals the simulation on
`host_emu` (whose unit model gained op 9), and the real Piper project with
FOLD runs bit-exact on `tts_host_emu.py` (PCM, encoder, durations).  The
pricing: `perf_model` features, `rpu_policy.estimate_us` (`rows · (ceil(n /
8) + 1) + 500` cycles), `perf_calibrate`'s buffer sizes.  The suite 1 840
passed, ruff clean.

**5a.2 — RTL.**  `ro_fold.sv` (row sequencer, eight lanes, packer, its own
write bursts), external runs on `ro_rd_port`, op 9 in `ro_core` (R from the
ROPE divider); the C++ oracle (`rowops_ref.h`, the quotient form — the RTL
takes the reciprocal one), 8 new fixtures from the Python spec (36 in all),
random FOLD jobs (reversed and unaligned channel rows, empty / inverted
windows, any m and s).  One bug on the way: the packer read the lane
results one stage late (the pipeline had 10 stages, the valids 11), so each
group took its successor's values.  **Bit-exact ✓**: the 36 fixtures at
fast / random / slow memory timing, 1 112 random FOLD jobs over three seeds
at each timing, ctest `TestRowOpsRtl` (fixtures + 300 random jobs), lint
clean.  Cycles with ideal memory: 32 × 24 × 2050 in 199 051 (0.80 ms, 0.99
words per cycle; the host 5.5 ms), 64 × 12 × 1026 in 100 713, a flow
coupling 192 × 2 × 132 in 7 273.  **Synthesis** at 3.333 ns: WNS 0.000 ns
(no FOLD path among the 20 worst: the sigmoid UltraRAMs, as before),
34 675 LUT (+3 728), 28 259 FF, 129 DSP (+8); `ro_fold` 3 546 LUT.

**5a.3 — the block design.**  `sim_hw_kv260` **112 / 112** (RowOps 37 / 37:
the table load and the 36 fixtures through the PS VIP on HP0; the
testbench parses the signed FOLD registers as 64-bit).  The first bitstream
with FOLD did not route: it placed at 97 729 LUTs (83.4 %, 99.9 % of the
CLBs) and the router stayed at WNS −2.18 ns.  An area pass over the five
kernels at equal throughput (`AREA_PLAN.md`: the FIFOs' skid stages as
LUTRAM rings, per-byte control writes, 48-bit address arithmetic, Matmul's
packed accumulators, Conv's weight skew in the RAM addresses and the DSPs'
A registers) gave bitstream **`de968ccc45da`**: 89 547 LUTs (76.5 %, 4 422
below the production `1cedaf2ed816`), WNS +0.029 ns, WHS +0.010 ns; the
block-design simulation again 112 / 112.

**5a.4 — the board** (2026-10-15, `de968ccc45da`, deep idle on as the soak
of RPU_PLAN §11 leaves it):

| gate | result |
|---|---|
| `run_remote_tests` (159 models) | **159 / 159**, every output equal to the simulation |
| `run_remote_perf` (72 cases) | 72 / 72; against `1cedaf2ed816` run twice in the same session every case within the run-to-run noise (the 20–25 µs calls move by 1–2 µs, FC-1x1280x1001-packed is 0.36 or 0.40 ms by its buffers' placement on either bitstream); baseline `kv260-de968ccc45da.json` recorded |
| SmolLM2-135M / 360M, SmolVLM (`llm_board`, the libraries unchanged) | logits **bit-exact** (4 prompts × 33 / 33; SmolVLM 2 images × 33 / 33); decode 50.5 / 131.8 / 50.2 ms per token, prefill 256 408 / 1 063 / 408 ms, `llm_image` 1.64 s |
| BERT-base (`deploy_and_run`, 50 examples) | 50 / 50 bit-exact with the emulation, EM / F1 88.0 / 90.3; p50 266.3 ms |
| the CNN demos | MNIST 0.114 / 1.266 ms (the same on `1cedaf2ed816` in the same session: the 0.108 / 1.251 of its baseline predate the A53's interrupt waits), ResNet-18 20.02 ms, MobileNet v1 / v2 20.51 / 17.52 ms |
| **Piper with FOLD** (`tts_board`, the library regenerated with `kernels.rowops.fold`) | PCM, encoder and durations **bit-exact**; **chunk 179–187 ms** (≈ 260 before; gate ≤ 205 ✓), RTF 0.134 (6.88 s utterance) / 0.169 (2.15 s) |
| YOLOv5n, LightStereo-S (`run_demo.py`) | head maps / disparity bit-exact with the simulation; 47.9 ms per image (47.9 on `1cedaf2ed816` in the same session), 468.9 ms per pair |
| performance model `kv260/de968ccc45da` | `cases` 1703 + `1cedaf2ed816`'s 179 ConvKernel refinement calls, three refinement rounds (+103, +84, +31) → 2100 exact calls, repeat spread median 0.069 %; FOLD's 41 calls of the chunk 11.3 ms predicted, 11.6 measured |
| `perf_calibrate.py simulate` (gate ±3 %) | the CNNs and BERT within 0.4 % ✓; the Piper chunk +4.3 % ✗ (every node priced within a few %: the lanes' overlap); the chat libraries −2.5 … −6.3 % ✗ — the same with `1cedaf2ed816`'s model: the host-op model predates the A53's interrupt waits and the deep-idle soak (a `host --merge` of new profiles is open) |

The ctypes check of the 360M library reports CmaFree 6.4 MB short after its
first close on both bitstreams (after the close 1 007 788 kB to the kB on
each; only the value before the open moves with the board's page cache): its
4 MiB rule compares against a moving start, not a leak.

**5a.5 — production** (2026-10-15): `/lib/firmware/pl.bin` is
`de968ccc45da` (archive `/mnt/data/bitstreams/kv260_250_de968ccc45da/`, the
previous one stays in `kv260_250_1cedaf2ed816/`); `platforms/kv260.json`
`kernels.rowops.fold = true`; the R5 BSP rebuilt from the design
(`xparameters.h` unchanged); the Piper library regenerated with FOLD and
installed by `tts_board.py`; the chat server redeployed (`deploy.py`, the
R5 driving): healthy, speech at RTF 0.137 through the server; the deep-idle
soak re-enabled after the deploy.  The other libraries are unchanged (no
FOLD in their graphs; bit-exact on the new bitstream above).
