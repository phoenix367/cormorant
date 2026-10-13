# RowOpsKernel — the row-ops unit in SystemVerilog (`kernels/rowops_rtl`)

The fifth kernel of `doc/plans/HOSTOPS_PLAN.md`: a streaming row engine
that runs the transformer graphs' host row ops — RMSNorm / LayerNorm, the
residual adds, SiLU · up, RoPE, dequantisation, strided copies and the
small-block transposes behind the attention prep / merge — on the FPGA, in
fixed point, bit for bit the integer specifications of
`inference-scheduler/src/rowops_spec.py`.  One AXI4 master (reads on two
IDs, writes on one) on its own PS port — `S_AXI_HP0_FPD` through
`axi_rowops_intercon` in the block design since phase 3 (bitstream
`1cedaf2ed816`, 2026-10-12: control block at `0xA004_0000`, interrupt
`pl_ps_irq0[4]`, UIO `fabric_rowops`; `hw/cormorant_hw_128/scripts/bd_rowops_hp0.tcl`)
— and an AXI4-Lite control slave with the HLS register-block layout,
`ap_ctrl_hs`.

Read this before changing the RTL: §1 is the contract the scheduler and
the drivers program against, §6 the invariants a change must keep.

## 1. Contract

### 1.1 Interfaces

| interface | role | bus parameters (declared by the IP, `syn/package_ip.tcl`) |
|---|---|---|
| `s_axi_ctrl` | AXI4-Lite, 7-bit address (128 B), the register map of §1.2 | |
| `m_axi_gmem0` | AXI4, 128-bit data, 64-bit address, ID width 1: reads of stream A on ARID 0 and stream B on ARID 1, writes on AWID 0 | READ_WRITE; NUM_READ_OUTSTANDING 16, NUM_WRITE_OUTSTANDING 16; MAX_READ_BURST_LENGTH 64, MAX_WRITE_BURST_LENGTH 64 (beats of 16 bytes, never across 4 KiB) |
| `ap_clk`, `ap_rst_n`, `interrupt` | as the other kernels (250 MHz in the block design; the reset registered inside, `keep`) | |

VLNV `xilinx.com:hls:RowOpsKernel:1.0`, driver `RowOpsKernel_v1_0`
(`XRowopskernel_*`, the HLS driver API, written by `scripts/gen_driver.py`).

### 1.2 Register map

HLS layout: `0x00` ctrl (b0 ap_start, b1 ap_done COR, b2 ap_idle, b3
ap_ready COR, b7 auto_restart, b9 interrupt), `0x04` GIE, `0x08` IER (b0
done, b1 ready), `0x0C` ISR (toggle on write).  Arguments
(32-bit words; 64-bit values low word first):

| offset | name | meaning |
|---|---|---|
| `0x10` | `op` | [3:0] op code (§1.4); [5:4] A kind; [7:6] B kind; [9:8] C kind (0 i16, 1 i32, 2 f32 — C only); [12] `B_EN` stream B present; [13] `P_BCAST` one parameter entry for every column; [14] `P_KEEP` keep the parameter RAM of the previous job (no read of `p`); [15] `RELU` (COPY / ADDX: max(y, 0) after saturation); [16] `NORM_LN` (NORM: LayerNorm; 0 RMSNorm) |
| `0x14` | `n` | elements per row, 1 … 4096 (TRANSPOSE: L, the tile's row length) |
| `0x18` | `c1` | rows per block (TRANSPOSE: tiles per block) |
| `0x1C` | `c2` | blocks (≥ 1) |
| `0x20`/`0x24` | `a` | stream A base (byte address, 16-byte aligned) |
| `0x28` | `a_s1` | bytes between A's consecutive rows (multiple of 16) |
| `0x2C` | `a_s2` | bytes between A's blocks |
| `0x30`/`0x34` | `b` | stream B base |
| `0x38` | `b_s1` | B row stride (TRANSPOSE: bytes between the tile's R source rows) |
| `0x3C` | `b_s2` | B block stride (TRANSPOSE: bytes between destination granules) |
| `0x40`/`0x44` | `c` | stream C base (16-byte aligned; TRANSPOSE: 2-byte aligned) |
| `0x48` | `c_s1` | C row stride (TRANSPOSE: tile stride) |
| `0x4C` | `c_s2` | C block stride |
| `0x50`/`0x54` | `p` | the parameter block (§1.3), 16-byte aligned |
| `0x58`/`0x5C` | `eps` | NORM: the integer E of `rowops_spec.norm_params` (`round(eps·n·2^2f)` for RMSNorm, `round(eps·n²·2^2f)` for LayerNorm) |
| `0x60` | `aux` | ROPE: [11:0] HD (the head segment, a multiple of 16, 16 … 128); TRANSPOSE: [3:0] R (rows per tile, 1 … 8; L = `n` ≤ 1024) |
| `0x64` | `bq_sh` | [5:0] the left shift of every column's Bq' before it is added (NORM: 24 + g − bb, bb = beta's scale; COPY / ADDX: the bias's scale) |
| `0x68` | `win_lo` | COPY / ADDX: A rows with index i1 < win_lo read as zeros (default 0) |
| `0x6C` | `win_hi` | … and rows with i1 ≥ win_hi (default 0xFFFFFFFF): tap stacking's rows outside the sequence |

Registers are sampled when `ap_start` is taken; they may be rewritten
while the job runs.

### 1.3 Streams, rows and the parameter block

A **stream** is `c2` blocks of `c1` rows; row (i2, i1) starts at
`base + i2·s2 + i1·s1` and holds `n` elements of the stream's kind
(i16: 2 bytes, i32 / f32: 4) — `ceil(n·bytes / 16)` words.  Streams A and B
are read, C is written.  Rows of A and B are read whole words; the lanes
past `n` of a row's last word read as 0.  Rows of C are written whole words
(every beat of a run, WSTRB all ones): the lanes past `n` of the last word
are written 0, so the caller's buffer or stride gap must cover the whole
last word — the alignment contract of the other kernels (the scheduler's
64-byte allocations and `CHUNK_STRIDE` do).  TRANSPOSE is the one op that
writes with byte strobes (§1.4).

The **parameter block** P gives every column c < n (or one entry for all,
`P_BCAST`) the constants the op needs.  In DDR an entry is 16 bytes:

| bytes | field | type |
|---|---|---|
| 0–3 | `G` | int32 — NORM: `round(gamma[c] · sqrt(n) · 2^g)` (RMSNorm) or `round(gamma[c] · 2^g)` (LayerNorm), `g` the per-tensor scale of `rowops_spec.gamma_scale` |
| 4–7 | `Bq'` | int22 in an int32 — NORM: `round(beta[c] · 2^bb)` with bb = 20 while every |beta| < 2, one less per doubling beyond (`rowops_spec.beta_bits`; BERT's attention LayerNorms reach 4.2), added as `Bq' << bq_sh`, `bq_sh = 24 + g − bb`; COPY / ADDX: the bias, added as `Bq' << bq_sh` |
| 8 | `sh_a` | 0 … 63: the left shift of A's element |
| 9 | `sh_b` | the left shift of B's element (ROPE: of the rotated partner's product) |
| 10 | `sh_c` | the right shift of the sum before the output rounding (NORM: `24 + g − f_out[c]`) |
| 11–15 | | reserved, 0 |

Inside the unit an entry is 72 bits (`G` 32, `Bq'` 22, three 6-bit shifts).
P is read once per job (n words), before the streams start, unless
`P_KEEP` says the previous job's block still applies (the same `p` and `n`;
the driver's responsibility).

### 1.4 The ops

Every op reads A (and B) row by row and writes C row by row, one C word
per cycle at most; `rne` is round half to even, `sat_K` saturation to the
output kind.  The exact definitions are the functions named in
`inference-scheduler/src/rowops_spec.py`; the testbench's oracle is their
C++ port.

| code | op | definition (per row, per column c) | kinds |
|---|---|---|---|
| 1 | `COPY` | `y = sat_C(rne_shr((A[c] << sh_a[c]) + (Bq'[c] << bq_sh), sh_c[c]))`, then `max(y, 0)` with `RELU`; rows outside `[win_lo, win_hi)` read A as 0 | A i16 / i32 → C i16 / i32 |
| 2 | `ADDX` | `y = sat_C(rne_shr((A[c] << sh_a[c]) + (B[c] << sh_b[c]) + (Bq'[c] << bq_sh), sh_c[c]))`, `RELU` as COPY | A, B, C i16 / i32 (the residual add: A i32, B i16 with `sh_b = 15 − f_d`, `sh_c = 0`) |
| 3 | `DEQUANT` | `y = float32(A[c] · 2^−sh_c[c])` (IEEE round to nearest even; exact for |A| < 2^24) | A i16 / i32 → C f32 |
| 4 | `MULACT` | `y = sat16(rne_shr(silu_q(A[c] << sh_a[c]) · B[c], sh_c[c]))` — `silu_q` of `rowops_spec` (the sigmoid table, §2.4); `sh_a = 14 − f_g[c]`, `sh_c = 20 + f_u − f_out` | A, B, C i16 |
| 5 | `NORM` | `rowops_spec.rms_norm` (`NORM_LN` = 0) or `layer_norm` (= 1) of the row with `E = eps`, `G`, `Bq' << bq_sh`, `sh_c`: exact `S1 = Σ x`, `S2 = Σ x²`, `D = Σ h² + E` or `n·S2 − S1² + E`, `(Rm, sh) = rsqrt_bf(D)`, `P1 = floor(A_c · Rm · 2^(24 − sh))` with `A_c = h` or `n·x − S1`, `y = sat16(rne_shr(P1 · G[c] + (Bq'[c] << bq_sh), sh_c[c]))` | A i16 / i32 → C i16 |
| 6 | `ROPE` | rows of `n = m · HD` i16: `d = c mod HD`, `rot = −A[c + HD/2]` for `d < HD/2` else `A[c − HD/2]`; B's row holds `cos[0..HD)` then `sin[0..HD)` as int32 words (int24 values at 2^−23, `rowops_spec.rope_tables_raw`); `y = sat16(rne_shr(((A[c]·cos[d]) << sh_a[c]) + ((rot·sin[d]) << sh_b[c]), sh_c[c]))` — `rowops_spec.rope` with per-column exponents | A, C i16; B i32 |
| 7 | `TRANSPOSE` | tile (i2, i1): R rows (`aux`) of L = `n` i16 at `a + i2·a_s2 + i1·a_s1 + r·b_s1`; the tile's column d (< L) becomes a granule of R int16 — `out[d][r] = in[r][d]` — at byte address `c + i2·c_s2 + i1·c_s1 + d·b_s2`, written with byte strobes covering exactly its 2R bytes (a granule may straddle two words; `b_s2 = 2R` makes the granules one contiguous run of L·R elements, written as whole words with the lanes past the last element zero — the stream ops' contract — at a 16-byte-aligned `c`) | i16 |

| 8 | `TLOAD` | the tables: A is the 64 KiB table image as rows of int32 (`n = 4096`, `c1 = 4`, `a_s1 = 16384` — any geometry that delivers its 4096 words in order): words 0 … 2047 hold two sigmoid entries {D[i], T[i]} (bits [48:25] / [24:0] of each 64-bit half, i = 2w and 2w + 1; `rowops_spec.sigmoid_table`), words 2048 … 4095 four rsqrt entries (int32, i = 4(w − 2048) …; `rowops_spec.rsqrt_table`); no B, P or C.  Run once after the bitstream, before any MULACT or NORM: the tables live in UltraRAM, which has no initial contents.  `scripts/gen_fixtures.py` writes the image as `tables.hex`; the testbench's oracle builds the same one | A i32 |

Rows are independent; the counts are 32-bit; `n = 0`, `c1 = 0` or `c2 = 0`
is an empty job (`ap_done` without traffic).  Values outside the
contract (unaligned strides, a kind the op does not take, `HD` not a
multiple of 16 or above 128, `R` above 8, `n` above 4096 — 1024 for
TRANSPOSE) give undefined results but never a hang:
every counter is sized for the register value.

### 1.5 How the scheduler's ops map onto it

| host op today | unit jobs |
|---|---|
| `LlmResAddNode`, `VitResAddNode` | ADDX (A h i32, B d i16, `P_BCAST` or per-column bias) |
| `LlmRMSNormNode`, `VitLayerNormNode`, `LayerNormNode` (BERT), `TtsResNormNode` | NORM (TtsResNorm: ADDX into an i32 row, then NORM) |
| `LlmSiluMulNode` | MULACT |
| `LlmDequantNode` (`DequantRoNode`, RPU_PLAN A1: the output a float32 DMA tensor or state, `dma: f32`) | DEQUANT (A i16 at per-column exponents, or the int32 residual) |
| `VitSumDequantNode` (RPU_PLAN A1, the vision frontend's `rowops`) | the parts summed by ADDX into an int32 row at the finest part exponent (`ResAddBiasRoNode` with a zero bias, then `ResAddRoNode`), DEQUANT into the float32 image-feature state |
| `LlmSelectRowNode` (`SelectRowRoNode`, the int32 residual) | one COPY (A / C i32) of row n − 1, the base from the runtime n |
| `VitPixelShuffleNode` (`PixelShuffleRoNode`) | one COPY per source block the column window touches: the output rows (i, j) as rows j × blocks i, the source rows (i·s + b)·n + j·s + a |
| `LlmAttnPrepNode` (the prefill entries — `AttnPrepRoNode`, HOSTOPS_PLAN §10) | ROPE on q0 (→ a staging row at f_q[h]) and k0 (→ a staging row at f_k[g]), one COPY of the k rows into the K cache (`blocks` = the KV groups), a COPY re-rounding v where the V cache's exponents differ, one TRANSPOSE per KV group of the q rows into the q·Kᵀ x image (tiles of 8 query rows, `kw` 1), the V image per group as TRANSPOSEs: the 16K-row blocks inside [pos, pos + n) with R = group_kw (contiguous granules), the two boundary blocks as R = 1 granules (byte-strobed); a COPY with the window [0, 0) zeroes the q staging once per run |
| `VitAttnPrepNode`, Piper's (`VitAttnPrepRoNode`, phase 4b) | q0 + bq and v + bv as COPYs with the bias in the parameter block (`rowops_spec.bias_round`: Bq' = round(b·2^fp), fp = `bias_exp`) into staging rows, k0 + bk per head straight into the K cache, one TRANSPOSE of the q rows into every head's image (the heads as blocks), the V image per head as the whole-block TRANSPOSEs (T = C = a multiple of 16K) or, group_kw 1, one COPY with the heads as blocks |
| `LlmAttnMergeNode`, `TtsAttnMergeNode` | COPY per group (2-level: rows t, heads hh) |
| `TtsRowPrepNode` (`TtsRowPrepRoNode`, phase 4b) | one COPY per tap of the rows it reads (`rows = n`, the output shifted by the tap's offset) with the bias and `RELU` in the parameter block; the rows the host zeroes (the first tap's row 0, the last tap's row n − 1, every row ≥ n) as COPYs with the window [0, 0) |
| `TransposeNode` (`TransposeRoNode`, RPU_PLAN A1: BERT's head splits / merges and Kᵀ) | a head split / merge (the collapsed [D2][D1][n] with strides [n, D2·n, 1]) as one COPY with `rows = D1`, `blocks = D2`; Kᵀ ([B][L][M] with strides [sB, 1, sM]) as one TRANSPOSE of 8-row tiles into granules at the output's row stride M |
| `TtsSumNode` (two inputs, `TtsSumRoNode`) / `TtsInterleaveNode` (`TtsInterleaveRoNode`) | ADDX over the channel slices with the shifts fp − f at the max exponent / one TRANSPOSE of R = the phase count s, tile = output channel, the columns in ≤ 1 024-wide chunks as blocks |
| `TtsAttnRelAddNode` (`TtsRelBandRoNode`, `PiperEncoderFrontend(rel_kernel=True)`) | the relative keys as 16 extra K cache rows (a COPY in `VitAttnPrepRoNode`; the q·Kᵀ call yields q·ekᵀ as rows T ..), one TRANSPOSE (R = 1, granule stride 2(T + 1): byte strobes) per key offset onto the diagonals of a zero band state [W + T][T] (W guard rows take the granules of i < −d, so every job reads its scores row from its aligned start), one ADDX of the band onto the scores |
| `VitEmbedAddNode` (`EmbedAddRoNode`, opt-in: `VitFrontend(iembed=True)`) | one ADDX of the patch embedding (i16, per-column shifts) and the position table as an int32 stream at 2^-15 (shipped as a Q8.8 weight whose words are the int32 image) into the int32 residual |
| `ConcatNode`, `SliceNode` | COPY |

## 2. Architecture

```
 s_axi_ctrl ─► ro_ctrl_s_axi ─► ro_core (job FSM, geometries, op mux)
                                   │
   m_axi_gmem0 AR/R ◄─ ro_rd_arb ◄─┼─ ro_rd_port A (ARID 0) ─► A words ─┐
                                   ├─ ro_rd_port B (ARID 1) ─► B words ─┤
                                   ├─ ro_param  (P: 2 URAM 4096 x 72, 4 banks, 4 entries / cycle)
                                   │                                     ▼
                                   │   ro_stream (COPY / ADDX / DEQUANT / MULACT; sigmoid table, 2 URAM)
                                   │   ro_norm   (ring 2 URAM, sums, ro_rsqrt, emit)
                                   │   ro_rope   (head-segment buffer)
                                   │   ro_xpose  (tile ≤ 8 rows x 1024 elements, granule strobes)
                                   │                                     │
   m_axi_gmem0 AW/W/B ◄─────────────┴─ ro_wr_port (strobes) ◄─ C words ◄─┘
```

- **Ports** are the VectorOPKernel's (`vo_rd_port`, `vo_wr_port`,
  `vo_burstgen`, `vo_fifo`, `vo_rs`, renamed `ro_*`): credit-based read
  FIFOs so RREADY never drops, bursts of at most 64 beats within 4 KiB,
  registered AXI outputs through register slices, 16 outstanding bursts
  per direction.  The write port gains a per-beat strobe input (all ones
  for every op but TRANSPOSE).  `ro_rd_arb` merges the two read ports' AR
  streams onto the one channel (round robin, ARID = port) and routes R
  beats by RID; each port's beats stay in order.
- **Geometry** is the VectorOPKernel's `geom_t` (blocks × runs × words
  with a tail), built per stream from `n`, `c1`, `c2` and the strides; the
  window of COPY / ADDX marks runs that are not read and are replaced by
  zero words at the port's output.
- **Lanes.**  Every datapath works on four lanes of 32 bits per cycle:
  an int32 / float32 word whole, an int16 word in two halves.  ADDX,
  MULACT and DEQUANT lose nothing against the port (their two reads, or
  their two output cycles, per word); NORM runs at four elements per
  cycle anyway; COPY and ROPE run at half the port rate (2 GB/s at
  250 MHz) — the choice that halved the LUT count (§2 Resources).  Column
  c's parameters come from `ro_param`'s RAM: bank c mod 4 at index c / 4,
  two UltraRAMs of 4096 × 72 holding two banks each — port A writes during
  the load (keeping its output, the write mode an UltraRAM infers from) and
  reads otherwise, port B reads: four entries per cycle, two cycles after
  the index.
- **Rounding** (`rne_shr`, the 64-bit round half to even) takes two
  stages everywhere (`ro_pkg::rne_a`: the shift, the round bit and the
  sticky bit; `rne_b`: the increment); the DSP products carry two
  registers after the multiply (what Vivado folds into the DSP48E2's
  pipeline), the NORM scalar unit's 32 × 32 products go through
  `ro_mul32` (four 16 × 16 products and two adds, three cycles) and the
  transposer's block step is a delta precomputed after start.
- **NORM** keeps two passes per row and never a third: pass 1 streams
  the row from DDR into the ring (2 URAMs: 4096 words of 128 bits — 4 rows
  of 4096 int32, more of shorter ones) while four lanes accumulate S1 and S2
  (an int16 word in two halves)
  (24 × 24 … 32 × 32 products, 80-bit adders in carry-save), pass 2
  reads the ring and emits.  Between them a sequential scalar unit forms
  `V = n·S2 − S1²` (or `Σh² + E`), block-floats D, reads the rsqrt table
  (`ro_rsqrt_ram`: 8 192 × 32-bit entries, two per word of one UltraRAM
  loaded by TLOAD, index = the 13 leading bits of the mantissa, two
  cycles) and takes one Newton step with 32-bit products.
  Rows ahead are loaded while a row is emitted (the ring holds ≥ 2 rows),
  so the unit runs at the slower of the two ports.  The emit lanes compute
  `floor(A · Rm · 2^(24 − sh))` (44 × 32-bit product, then a variable
  right shift), `· G` (32 × 32), `+ Bq'<<bq_sh`, the per-column rounding
  shift and the saturation.
- **MULACT**'s sigmoid table (4 096 entries of {T[i], T[i+1] − T[i]},
  49 bits, `rtl/ro_sig_ram.sv`, loaded by TLOAD) gives both ends of the
  interpolation in one read: two UltraRAM copies, two ports each, serve
  the four lanes, answering two cycles after the address; the
  interpolation and the two products are DSP48E2 work (`gv · σ` 32 × 25,
  `s · u` 39 × 17).
- **ROPE** buffers one head segment (HD ≤ 128 elements = 16 words) of A
  and the row's cos / sin from B (eight lane banks of LUTRAM, one write
  enable each), then emits the segment half a word per cycle through the
  four lanes: the partner element is HD/2 away, so the rotation needs the
  whole segment first; HD a multiple of 16 keeps the partner in the same
  lane HD/16 words away.  Two segment buffers alternate (fill one, emit
  the other); the next row's cos / sin wait for the row's last segment to
  leave the buffers (the banks are read at issue), so a row of one
  segment costs a bubble of its B load.
- **TRANSPOSE** reads the tile's R rows into eight row RAMs of 128
  words (LUTRAM, one asynchronous read port per row; L ≤ 1024), then emits L granules of R elements with their byte
  strobes (two beats when a granule straddles a word, split at a 4 KiB
  boundary; one contiguous run of full words, zero tail lanes, when
  `b_s2 = 2R`).
- **Resources** — `synth_rowops_rtl` on 2026-10-11 (phase 1, out of
  context at 3.333 ns): **WNS +0.025 ns** (Fmax ≈ 302 MHz), **30 947 LUT**
  (27 107 logic, 3 840 memory), 24 774 FF, 6 RAMB36 (the port FIFOs),
  7 URAM (the norm ring 2, the parameters 2, the sigmoid table 2, the
  rsqrt table 1), 121 DSP.  Per unit: the stream ops 7.7 k LUT / 20 DSP,
  ROPE 6.5 k / 16, NORM 6.1 k / 77, TRANSPOSE 5.0 k (2.5 k of it the tile
  buffer), the ports 4.2 k.  The plan's estimate was 12–15 k LUT, 4 BRAM
  and 80–100 DSP: the four-lane 64-bit shift / add / round-to-even
  arithmetic is the cost the estimate priced as DSP work.  How it got
  here: the first synthesis gave 81 k LUT / 170 k FF (the tile buffer and
  ROPE's banks as flip-flops, the parameter RAM in 32 BRAM, eight sigmoid
  copies) and WNS −22 ns (ROPE's n / HD); recoding the memories, the
  sequential divider and the four-lane MULACT gave 42 k LUT / 32 BRAM /
  −4.8 ns; four lanes for every op, the tables in UltraRAM (loaded by
  TLOAD), the two-stage rounding, the DSP pipeline registers, `ro_mul32`
  in the scalar unit and the transposer's precomputed step gave 31 k LUT /
  6 BRAM / +0.025 ns.  In context (phase 3, bitstream `1cedaf2ed816`) the
  whole design closes 4 ns with WNS +0.020 ns / WHS +0.010 ns at 93 969
  LUT (80.2 %), 113 020 FF, 136 / 144 BRAM tiles, 55 / 64 URAM and 833
  DSP — the unit and its interconnect cost 31 351 LUT, 25 188 FF, 5 BRAM
  tiles, 7 URAM and 121 DSP over `f70ce632a97a`.

## 3. Performance

One output word per cycle: 4 GB/s on either direction of a 128-bit PS
port at 250 MHz; a job's time is `max(words read, words written) / 1 +
~2 µs` of issue and completion.  The replay of HOSTOPS_PLAN §3 priced the
unit at 85 % of that rate plus 2.5 µs per call.  NORM on int16 rows runs
at 4 elements per cycle (half a word): BERT's 25 LayerNorms of 256 × 768
take 25 × 49 k cycles ≈ 5 ms at 250 MHz against 81 ms on the host.
Cycle counts of the shipped models' job shapes with ideal memory
(`make perf_rowops_rtl`, 2026-10-11, the four-lane design; one read+write
port, 128 bits each way):

| job | cycles | µs at 250 MHz | bound |
|---|---:|---:|---|
| ADDX 256 × 576, A i32 + B i16 → C i32 (the residual add) | 55 471 | 222 | i32 rows: 4 elements per cycle |
| MULACT 256 × 1536 (SiLU · up) | 100 004 | 400 | A and B through the one read port: 98 304 words |
| COPY 256 × 576 i16 | 37 034 | 148 | four lanes: half a word per cycle |
| DEQUANT 12 × 4096 i16 → f32 | 12 467 | 50 | 12 288 output words |
| NORM (RMS) 256 × 576 i32 | 38 013 | 152 | 4 elements per cycle |
| NORM (LN) 256 × 768 i16 | 50 447 | 202 | BERT's 25 LayerNorms ≈ 5.0 ms (81 ms on the host) |
| NORM (LN) 1024 × 768 i32 | 198 757 | 795 | |
| ROPE 256 × 576, HD 64 | 47 933 | 192 | four lanes + the B load per row |
| TRANSPOSE R 8, L 64, 96 tiles (the q image) | 18 491 | 74 | one granule per cycle out |
| TLOAD (the 64 KiB table image, once per bitstream) | 8 236 | 33 | two beats per word |

Board figures (the DDR behind the port, the PS port's own rate) follow
with phase 3.

## 4. Verification

- `tb/verilator/tb_main.cpp` — the AXI slave models of the other RTL
  testbenches (sparse DDR, randomised READY / VALID / latency, protocol
  checks, 4 KiB and outstanding limits) on one read+write port and the
  AXI-Lite master; the C++ oracle `tb/verilator/rowops_ref.h` — a port of
  `rowops_spec.py` in 64- / 128-bit integer arithmetic — checked per case on
  the whole output region (every element, the zeroed tail lanes, no stray
  write), ap_done / interrupt behaviour, no traffic on the unused ID.
- **Fixtures** `hw/test_data/rowops_test_data/` (28 cases: `manifest.txt`,
  `test_NN_{a,b,p,c}.hex` — every op, kinds, tails, the window, the
  broadcast and per-column parameter block, both norms, ROPE at HD 64,
  contiguous and strobed transposes): written by `scripts/gen_fixtures.py`
  (run it from the scheduler's venv: it imports `rowops_spec`) from the
  Python specification itself (`inference-scheduler/src/rowops_spec.py`),
  so the C++ oracle is checked against the Python one by the fixtures and
  the RTL against both.  CTest `TestRowOpsRtl` runs them and 300 random
  jobs (seed 1, `RO_RTL_RANDOM_CASES`) under randomised AXI timing; the
  RTL passed them at every timing and 1 200 random jobs of four seeds on
  2026-10-11.
- `--random N`: random jobs (all ops, kinds, shapes inside and at the
  contract's edges) against the C++ oracle; `--case "..."`; `--perf`
  (ideal memory cycle counts of the shipped models' job shapes);
  `--timing fast|rand|slow`.
- `lint_rowops_rtl` (`verilator --lint-only -Wall` with
  `scripts/lint_waivers.vlt`), `synth_rowops_rtl` (300 MHz out of
  context), `xsim_rowops_rtl` (xvlog / xelab), `RowOpsRtlDriver` (the driver's register
  table against `rtl/ro_ctrl_s_axi.sv`).  The testbench runs a TLOAD job
  before its cases (the oracle's table image, checked against the
  fixtures' `tables.hex`), as the driver does once after the bitstream.
- **The block design** (`make sim_hw_kv260`, `hw/cormorant_hw_128`): the
  testbench's `ro_classes.svh` runs the table load and the 21 checked-in
  fixtures through the PS VIP's DDR model on the HP0 port — every A / B / P
  image written, the C region poisoned and compared byte for byte with the
  fixture — 22 of the design's 97 cases.  Two lessons from bringing it up:
  the PS VIP's DDRC write path races and clobbers bytes just past a burst
  (the previous transfer's byte count at the new address), which the
  testbench undoes from a shadow of every byte written (`tb_infra.svh`);
  and a 4-state simulator shows an accumulator that a job never cleared
  (`ro_xpose`'s contiguous packer, X on the first word — fixed: a job
  starts with it empty).

## 5. Build targets

`TestRowOpsRtl`, `lint_rowops_rtl`, `perf_rowops_rtl`, `rowops_rtl_tb_fst`,
`driver_rowops_rtl`, `package_rowops_rtl`, `synth_rowops_rtl`,
`xsim_rowops_rtl` — `kernels/rowops_rtl/CMakeLists.txt`, the
VectorOPKernel's pattern (`doc/build-and-test/BUILD_TARGETS.md`);
`synthesize_<platform>` packages this IP with the other four, `ip_repo_kv260`
links it for the block design, `sim_hw_kv260` / `build_hw_kv260` simulate and
build the design with it.  A `neteq_rowops_rtl` lockstep run (`tools/neteq`, a
job set for this kernel) is still to be written; the bitstream's synthesis
log has no multi-driven-signal warning (`Synth 8-4767`) for the unit.

## 6. Invariants for changes

1. **The specification is `rowops_spec.py`.**  A numerics change is made
   there first (with its 1-LSB tests and the model gates of
   HOSTOPS_PLAN §5), then in the C++ oracle, then in the RTL; the fixtures
   are regenerated and checked in.
2. **Bit-exact and no hang.**  Every op's output equals the oracle's on
   every element; every counter is sized for the register value;
   `ap_done` follows every job, empty or not.
3. **The contract's layouts** (§1.3): whole-word rows with zero tails,
   strobes only in TRANSPOSE, 16-byte-aligned bases and strides (TRANSPOSE's
   C 2-byte).
4. **250 MHz in context**: the reset registered and `keep`-copied per
   unit; no AXI handshake input reaching a wide enable combinationally
   (the register slices); wide fanouts through register trees; no variable
   written from two generate scopes (Vivado mis-synthesises it, Verilator
   does not — FMAX_250_PLAN); Vivado also rejects a part select that
   can leave the vector for an unused lane (Verilator does not): index
   through a sized variable.  Memories: a 2-D array written with a
   variable first index, or read by several ports, becomes flip-flops
   (the first synthesis: 133 k FF for the tile buffer) — one RAM per
   bank / row with its own write enable; an UltraRAM takes no initial
   contents (the tables are loaded by TLOAD) and a port that both writes
   and reads must keep its output on a write (`if (we) mem[a] <= d; else
   q <= mem[a];` — `ro_param`, `ro_sig_ram`, `ro_rsqrt_ram`).  A 64-bit
   `rne_shr` or a 32 × 32 product never sits in one stage (§2 Rounding).
5. **The AXI side**: RREADY never drops once a burst is issued (credit
   FIFOs), bursts within 4 KiB and ≤ 64 beats, at most 16 outstanding per
   direction, the two read IDs' beats in order per ID.
6. **Register map changes** go through `scripts/gen_driver.py`'s table
   (`RowOpsRtlDriver`), the testbench, the block-design testbench and the
   fact registry (`registers.RowOpsKernel`, registered with phase 2's
   writers), as the other kernels'.
