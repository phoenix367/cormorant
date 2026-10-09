# Depthwise convolutions on ConvKernel: where the time goes, and what would help

**Status (2026-10-09): analysis only — no kernel change yet.**  The
scheduler's depthwise channel slices ([STEREO_PLAN](STEREO_PLAN.md) §4.5,
INFERENCE_SCHEDULER.md §Depthwise channel slices) are in; this report
explains the measurements behind them and ranks the kernel-side options.

**Verdict in one paragraph.**  A depthwise job is slow not because the grid
does its multiplies slowly but because it moves its input slowly.  On the
board the kernel's x path delivers about 1–2 GB/s on these jobs, out of a
port that carries 4 GB/s and a DDR that the PL can drive at 6–7 GB/s; the
Verilator testbench with ideal memory runs the same jobs 1.4–3.3× faster
than the board, and its "slow memory" setting reproduces the board almost
exactly.  The weights were never the cost: a 3 × 3 depthwise layer of 144
channels on 120 × 160 reads 2.6 KB of weights and 5.5 MB of input.  The
first kernel change to make is in the x loader and patch producer (longer
runs, deeper prefetch, loads that do not stall the sweep); it would also
speed up the 1 × 1 convs, which have the same problem.  A faster depthwise
*datapath* (several taps per instant) only pays after that, and is best done
as a small separate engine in the free DSPs rather than inside the grid.

## 1. The question

"Why is the depthwise mode not *faster* — it needs 16× fewer weights?"

Because ConvKernel's time is instants and bytes, not weights:

| 3 × 3 job on 120 × 160 | MACs | weights | x + y bytes | bytes per MAC | useful MACs per instant |
|---|---:|---:|---:|---:|---:|
| standard, 64 → 64 ch (56 × 56, the perf benchmark) | 115 M | 73 KB | 0.8 MB | 0.007 | 512 |
| depthwise, 144 ch | 25 M | 2.6 KB | 11.1 MB | 0.44 | 32 |

The grid multiplies a 16-input-channel patch column against 16 output
channels for two pixels per instant: 512 MACs.  A depthwise instant uses one
lane per column (the diagonal weight word, `cv_engine`: "column m1 holds
only lane m1"): 32 MACs, 6 %.  And per useful MAC a depthwise job moves
60× more bytes.  So the depthwise mode runs at the *pixel rate* of a
16-input-channel standard conv — `out_h · ⌈out_w / 2⌉ · kh·kw` instants per
16 channels — while reading its whole input through the same x path that
the standard conv keeps busy 2 % of the time.

## 2. How the kernel runs a depthwise job (`kernels/conv_rtl/rtl`)

- **Sweeps** (`cv_core`'s sequencer): `(image, chunk, m-tile, ow-tile)`; every
  depthwise sweep loads its own input tile (`k.load <= j.dwm || …`) because
  each m-tile's 16 channels are different input channels.
- **Chunk height**: `65 536 / (out_w · m_tiles · 16)` output rows, the same
  formula as a standard conv — although depthwise m-tiles do not share
  accumulators.  144 channels at width 160 → 2-row chunks (every chunk
  re-reads the window's halo rows and pays a sweep ramp per m-tile).  This is
  what the scheduler's channel slices work around.
- **x loader** (`cv_xload`): per output row of a loading sweep, one run per
  channel of the rows the line buffer lacks — a run covers the ow-tile's
  columns only, **at most `LBC / 8 + 1 = 9` words (144 bytes)**, 16 of them
  per row 2·in_h·in_w bytes apart (38 KB at 120 × 160); runs land in one half
  of a two-row ping-pong buffer, so at most two rows are in flight; ≤ 16
  bursts outstanding.
- **Patch producer** (`cv_patch`): a 16-row × 64-column line buffer per
  channel bank (block RAM, two ports); per output row the FSM goes
  `P_ROW → P_LOAD → P_EMIT`: the new rows are written (one column of 16
  channels per cycle, port A) **before** the row's window reads start, so
  within a sweep loading and emission do not overlap — only the DDR fetch of
  the next row overlaps, through the ping-pong.
- **Weights** (`cv_wload`, `cv_engine` fill): `⌈kh·kw / 8⌉` beats per channel,
  written as one diagonal word per window position (9 cycles per channel):
  negligible except on tiny jobs.

## 3. Measurements

Board: bitstream `8599aa7a5f12`, `run_remote_perf.py`, single ConvKernel
calls (the configs in the session's job directory; the depthwise ones are
also in STEREO_PLAN §4.5).  Verilator: `build/kernels/conv_rtl/vl/Vtb
--no-oracle --timing fast|slow` (fast: 1 beat per cycle after 24 cycles;
slow: 0.3–0.5 beats per cycle, 40–200 cycles latency).  Grid bound:
`out_h · ⌈out_w / 2⌉ · max(kh·kw, 2)` instants per m-tile (and per input tile).

| job | grid bound | Verilator ideal | Verilator slow | board | board ÷ ideal | x + y at the board's rate |
|---|---:|---:|---:|---:|---:|---:|
| dw 3 × 3, 16 ch, 120 × 160 | 0.346 ms | 0.461 | 0.531 | **0.635** | 1.4 | 1.9 GB/s |
| dw 3 × 3, 16 ch, 240 × 320 → 120 × 160 (stride 2) | 0.346 | 0.736 | 1.896 | **1.926** | 2.6 | 1.6 |
| dw 3 × 3, 144 ch, 120 × 160 | 3.11 | 4.59 | 7.78 | **10.26** | 2.2 | 1.1 |
| dw 1 × 7, 16 ch, 60 × 80 | 0.067 | 0.118 | 0.207 | **0.142** | 1.2 | 2.2 |
| standard 1 × 1, 16 → 16 ch, 120 × 160 | 0.077 | 0.196 | 0.483 | **0.564** | 2.9 | 2.2 |
| standard 1 × 1, 144 → 16 ch, 120 × 160 | 0.691 | 1.420 | 3.712 | **4.701** | 3.3 | 1.3 |
| standard 3 × 3, 64 → 64 ch, 56 × 56 (control) | 0.903 | 0.915 | 0.975 | 0.918 | 1.0 | 0.9 |

What the table says:

1. **The standard conv is grid-bound and insensitive to memory** (ideal,
   slow and board within 7 %): its loads are 2 columns per 9 × 4 instants.
2. **Depthwise and few-input-channel 1 × 1 jobs are memory-bound on the
   board**, 1.4–3.3× their ideal-memory time, and the slow-memory model
   reproduces the board (the stride-2 case to 2 %).  The board behaves, for
   these access patterns, like a memory that returns ~0.4 words per cycle
   with 100+ cycles of latency.
3. Even with ideal memory the kernel's own serialisation costs 25–100 %
   over the grid bound (the `P_LOAD` phase, the per-m-tile sweep ramps, the
   halo rows): dw 16 ch 120 × 160 0.346 → 0.461 ms, the stride-2 one
   0.346 → 0.736 (its input is 4× its output, all of it emitted through the
   line-buffer write port one column per cycle).
4. The x-path problem is not depthwise-specific: the 1 × 1 projections of a
   MobileNetV2 block (144 → 16 at 120 × 160) run at 3.3× their ideal time.
   In LightStereo-S the 1 × 1 convs are 165 ms of 599 — more than the
   depthwise 3 × 3s.

Depthwise per-channel cost on the board, 3 × 3 at 120 × 160: 40 µs in a
16-channel call, 71 µs in a 144-channel one (the chunk-height effect, §2;
now handled by the scheduler), 30 µs when the map is one column tile wide
(120 × 62: no second tile re-loading the rows).

Why the x path is slow is consistent with its access pattern — 144-byte
runs, 16 per row, each in a different DRAM page, at most two rows in flight —
but the share between DRAM page misses, the two-row ping-pong depth and the
AR issue rate is **not pinned down**.  The design has AXI performance
monitors (the `axi-pmon` UIO devices); reading their byte and transaction
counters during a depthwise job is the first step below.

Traffic of the whole stereo pair, ConvKernel only (x + y bytes from the
graph): depthwise 3 × 3 215 MB, 1 × 1 261 MB, stripes 97 MB, 3 × 3 66 MB,
rescales 53 MB — about 700 MB in 599 ms, 1.2 GB/s on average.  The port
is not the limit; the pattern is.

## 4. Options

### A. Scheduler: channel slices — done

A depthwise conv with many channels as 16- / 32- / 64-channel calls
(taller chunks, fewer halo rows and sweep ramps): LightStereo-S 659 →
599 ms, bit-identical, no bitstream.  It does not change the x path's
bytes per cycle, and the `P_LOAD` / `P_EMIT` serialisation stays.

### B. Kernel: the x path (the next change)

| change | what | expected (board) | cost |
|---|---|---|---|
| B1 — longer runs | fetch a channel's whole input row once per `(chunk, channel tile)` instead of once per ow-tile: a per-channel full-row buffer (in_w ≤ 512 columns: 16 ch × 64 words × 2 halves = 32 KB block RAM) that the ow-tiles' column windows are emitted from; or the loop order `(row, tile)` inside the chunk | run length 9 → up to 64 words; 3 (120 × 160) to 10 (640-wide) times fewer requests per row | `cv_xload` issuer / row buffer; the sweep sequencer if the loop order moves |
| B2 — deeper prefetch | 4 row halves instead of 2 (kev-gpt's window-granular ping-pong, §5): more rows in flight while a sweep runs | hides the per-row latency; needed for B1's longer rows | row buffer ×2 (+16 KB), the `h_busy` logic |
| B3 — loads beside emission | write the line buffer without taking the window reads' port: a second line-buffer copy (one for loads + pixel 0, one for pixel 1 — both copies written), or emission at one pixel per instant during loads | removes the `P_LOAD` serialisation: the ideal-memory overhead of §3 point 3 (stride 2: 0.736 → ~0.4 ms) | +32 KB block RAM, `cv_patch` FSM |
| B4 — depthwise chunk height | chunk rows from `65 536 / (out_w · 16)` for depthwise (its m-tiles are independent) | what the scheduler's 16-channel slices give, without the call overheads (~5 % of a 0.6 ms call) | `cv_core` geometry; frees the scheduler from slicing |

Bound: the ideal-memory column of §3 — dw 16 ch 120 × 160 0.635 → ~0.46 ms
(→ ~0.35 with B3), the stride-2 one 1.93 → ~0.5, 1 × 1 144 → 16 4.7 → ~1.4.
For LightStereo-S: depthwise 176 → ~100 ms, 1 × 1 165 → ~90 ms, pair
**599 → ~450 ms** (an estimate from the ideal-memory simulations; B1's real
DDR efficiency is the unknown — the APM measurement first).  Every job with
few input-channel instants per pixel gains; MobileNet v1 / v2 (22 / 20 ms)
by less, their maps being smaller.  Verification: `TestConvRtl` (bit-exact
against the HLS C++), the slow-timing testbench, `perf-regression`, and a
refit of `RTL_CONV_BOARD` (`tools/fit_cost_model.py`).

### C. Kernel: a depthwise datapath

The grid's shape — a patch column shared by 16 output columns, weights per
column — is the opposite of depthwise, where every channel has a private
input.  Putting a window's 9 taps on 9 lanes of column m's chain would sum
the window in one instant (the cascade already adds the lanes), but column
m then needs channel m's 9 patch values while column m + 1 needs channel
m + 1's: the patch bus and its register trees (8 192 of the engine's flip
flops are the per-chain patch copies) grow 16×.  Not worth it inside the grid.

A **separate depthwise engine** fits the free resources: 16 channels × 2
pixels × 9 taps = 288 DSP48E2 (712 of 1 248 used today, 536 free), its own
sliding-window line buffer (three row banks per channel so a 3 × 3 window is
one read per new column), sharing `cv_xload`, the bias RAM and the drain.
Gain: the sweep part 9× (one instant per pixel pair per 16 channels),
which after B is the limiter on stride-1 jobs — dw 16 ch 120 × 160 → ~0.2 ms
(the DDR floor at 4 GB/s is 0.15).  LightStereo-S depthwise → ~50 ms, pair
→ **~400 ms** with B.  Cost: a new unit, 288 DSPs across three clock
regions at 250 MHz in a design that closes 4 ns by +0.114 ns (FMAX_250_PLAN)
— the timing risk is the real one; kernels ≤ 7 × 7 would need 49 taps per
pixel or several instants per pixel.  Only after B, and only if the
in-context timing can take it.

### C′. Reusing the grid: the transposed depthwise mode

The grid does have a per-column operand: every column reads its own 16-lane
word from its own weight-cache RAM each instant.  Swapping the roles makes
depthwise fit:

- **lane l = tap l** of one channel c on the *patch* bus — shared by all
  columns, correctly, since every pixel uses the same filter;
- **column m = output pixel m** of channel c, its "weight word" the 16 input
  values of c's window around pixel m, fed into the column's weight-lane skew
  registers in place of the cache word;
- the cascade sums the lanes = the window sum, `wfb` 0: **one instant per
  32 pixels of a channel** (5 × 5 = 25 taps: two accumulating instants,
  7 × 7: four).

dw 3 × 3, 16 ch, 120 × 160: 9 600 instants instead of 86 400 — the 9× of a
separate engine with no new DSPs (9 of 16 lanes busy, 56 % of the grid
instead of 6 %) and no new global fan-out: the tap word is the patch tree
holding a constant per channel pass, the window vectors are per column,
local like the weights today.  That is the timing argument for reuse.

| piece | change |
|---|---|
| `cv_patch` | a per-channel window generator: the line buffer reshaped to row-wide words (64 columns × 16 bit per row, so 16 consecutive windows come from three row reads) with a shift / select network for stride 1–2 and the tap count; written column by column by today's loader (byte enables) |
| `cv_engine` operand path | a bypass mux per column before the weight skew registers: cache word or window vector (16 × 16 × 16 bit) |
| `cv_engine` accumulators | a depthwise word layout (channel, 16 pixels) instead of (pixel, 16 channels) — otherwise one instant's 16 pixels of one channel would be 16 lane-masked writes |
| `cv_drain` | a bypass for that layout: a word is 16 contiguous pixels of one channel, two y words, no transposer |
| `cv_core` | sweeps per (chunk, channel, ow-tile) instead of per m-tile; the ~113-cycle sweep ramp against ~600 instants per sweep at 120 × 160 (19 %) |

Everything the generator does not cover (horizontal dilation > 2, odd tap
counts above 16) keeps the diagonal path, so correctness never depends on the
new mode; the stereo stripe pieces (1 × 7 dilation 3, 2 × 1 dilation 13) may
stay on it — vertical dilation is free (rows are separate words), horizontal
dilation widens the select network.  It does not touch the x path: alone it
would take dw 16 ch 120 × 160 from 0.64 to roughly 0.4 ms on the board; with
B to ~0.2 (the 4 GB/s floor is 0.15).

### D. Kernel: per-channel output shift (the rescales)

The per-channel exponent scheme (`chexp`) spends 28 identity depthwise 1 × 1
calls — 24 ms and 53 MB of traffic in LightStereo-S — to rescale a conv's
output per channel.  A per-output-channel right shift at the drain (4 bits
per channel, a table read with the bias) removes them.  Small RTL (`cv_drain`
+ a register / table), a scheduler change (`chexp` emits the shift instead of
the rescale conv), bit-identical by construction (floor of a floor).  Also in
this family, outside the depthwise question: ReLU6 at the drain (82 VectorOP
calls, 55 ms, 100 MB).

### E. What kev-gpt suggests ([MichaelAyles/kev-gpt](https://github.com/MichaelAyles/kev-gpt))

The repository is an on-chip-resident INT4 GPT on the same board, not a
convolution engine; what transfers is its treatment of memory:

- **The bandwidth wall as the design centre.**  Its thesis — the PL only
  wins by leaving DDR — is the extreme form of §3: our conv jobs run at
  1–2 GB/s because of *how* they read, far under the 6–7.5 GB/s it quotes as
  the sustained HP ceiling.  Fix the pattern first (B); on-chip residency
  second: a MobileNetV2 block's expand → depthwise → project could run per
  row band with the depthwise input and output never written to DDR (138 KB
  for 3 rows × 144 channels × 160), saving 2 of the 4 big tensors per block —
  a kernel chaining feature, longer term than B or C.
- **Window-granular double-buffered prefetch** (`kv_prefetch.sv`: read each
  DDR row once, demux it to all consumers, fill the next window behind the
  current compute): B1 + B2 exactly — one long run per channel row, two or
  more rows ahead.
- **A DDR latency / cadence model in simulation** (`ddr_latency_model.sv`,
  swept first-word latency and beat gap): our Verilator `--timing slow`
  already plays this role and reproduced the board here; worth keeping as
  the gate for B (a change that only helps with ideal memory is not a change).
- **Wide words as addresses, not muxes**; dual-ported URAM split between two
  consumers; a 250 MHz hard wall they hit on this part — all consistent with
  how `cv_engine` is built and with the margin we have left.

## 5. Recommended order

0. **Measure the x port** on a depthwise job with the design's AXI performance
   monitors (bytes, transactions, average latency): an afternoon on the board,
   decides B1 against B2 / B3.
1. **B1–B3 (+ B4)** in `cv_xload` / `cv_patch` / `cv_core`: the only change
   that also helps the 1 × 1 convs.  Gate: Verilator fast and slow timing,
   `TestConvRtl`, the board benchmarks, cost-model refit.  Expected
   LightStereo-S ~450 ms.
2. **D**: the rescales (−24 ms) and, if the drain is being touched, ReLU6.
3. **C′**, the transposed depthwise mode on the existing grid, if B leaves the
   sweep as the limiter: no new DSPs and no new global fan-out, at the cost
   of five coordinated changes (`cv_patch`, the operand bypass, the
   accumulator layout, `cv_drain`, the sequencer).  A separate engine (C)
   only if that proves unworkable.

## 6. The transposed depthwise mode — design, implementation and measurement (2026-10-09; reverted)

**Status:** implemented in `kernels/conv_rtl` (about 900 lines over six
files), verified bit-exact, measured slower (the result and synthesis
paragraphs below), and **reverted from the tree** at the maintainer's
request: the kernel is the one of CONV_RTL_PLAN phase 3.  The patch is kept
outside the repository at
`/mnt/data/cormorant_repro/dwt_mode_2026-10-09.patch` (against commit
4878ade), with its testbench aid (`TB_MAX_MISMATCH=N` prints every
mismatch) for whoever picks the row-reuse design up.

The design: the grid serves both — a depthwise job that fits the rule below
runs in the transposed mode ("dwt"), every other depthwise job on the
diagonal path as before.  No register, layout or driver changes; bit-exact
by construction (the cascade sums the same products in a wrapping 32-bit
accumulator).

**Eligibility** (`cv_core`, from the registers): `is_depthwise`, `stride_w ≤ 2`,
the lane table below has `L ≤ 16` entries, and one accumulator row of the
dwt layout fits (`RW ≤ 4096` words).  Everything else is unchanged.

**Lane table.**  Column m of the grid is output pixel pair m of a 32-pixel
group; its chain's lane l multiplies the window element at row `r_l`, column
offset `o_l` (relative to the pair's window origin) by the tap on the patch
bus.  Pixel 0 of the pair needs offsets `{kwi·dw}`, pixel 1 `{sw + kwi·dw}`;
per window row the table lists the union in increasing order (a two-pointer
merge at job start), each entry carrying the tap index of pixel 0 and / or
pixel 1 at that offset: `L = kh · U`, `U = kw + sw` for dilation 1, `2·kw`
for `dw > sw`, `kw + 1` for `dw = sw`.  3 × 3 stride 1: 12 lanes; stride 2:
15; 7 × 1: 14; 1 × 7 dilation 3: 14; 2 × 1 dilation 13: 4.

**Taps on the patch bus** (`cv_engine` fill): the depthwise weight run of a
channel (8 taps per beat, as today) is scattered into two 16-lane patterns —
pixel 0's and pixel 1's, lane l taking the tap the table names — and written
to a tap RAM `[bank][channel of the m-tile]` (LUT RAM, 2 × 16 × 512 bits).
During a channel's instants the patch root chain carries its two patterns
(`patch0 = taps[bank][c]`): one register load per channel, the trees hold.

**Windows on the weight port** (`cv_patch` → `cv_engine`): in dwt the line
buffer is column-interleaved — element (channel c, row ih, position p) in
bank `p mod 16` at address `{c, ih mod 16, p / 16}`, positions relative to
the tile's window origin `iwb0` (`p = column − iwb0`, 0..63; the loaded
columns are `[pad_eff, pad_eff + iw_cnt)`, `pad_eff = iw_lo − iwb0`).  The
loader (`cv_xload`) emits 16 consecutive positions of one channel per cycle
(a 16-of-24 select over three row-buffer words; 64 beats per row, as many as
today's 64 columns); the patch producer writes them to the 16 banks at one
address.  Per instant (row, channel c, group gi) the generator reads, lane by
lane, the 16 columns' elements: positions `32·sw·gi + o_l + 2·sw·m` — for
stride 1 one cycle per lane (8 banks through port A, 8 through port B),
stride 2 two — masks the elements outside the loaded columns or the input
rows, and streams each lane (256 bits) to the engine, which captures it in a
per-column register bank `b_dw[m][l]`; the instant issues when the last lane
is in, `b_dw` replaces the weight-cache word at the column mask stage
(`rd2`), and the existing skew registers and chains do the rest.  One instant
per 32 pixels of a channel: `L·sw + ~8` cycles against `9` per *pair* today —
3 × 3 stride 1 ≈ 0.6 cycles per pixel instead of 4.5.

**Sweeps and instants** (`cv_core`, `cv_engine`): the sweep stays
(chunk, m-tile, ow-tile); its instants are (output row, channel of the tile,
group), `G = ⌈n_pairs / 16⌉` groups (1 or 2: a tile is ≤ 62 pixels), a
one-instant window: seeds from the bias of channel c at both pixels, `wfb`
0.  The patch producer and the engine walk the same (row, channel, group)
order; the generator blocks on the engine's capture buffer.

**Accumulators and drain**: a dwt word holds 16 pixels of one channel —
the even pixels of the group at `w`, the odd at `w + 1`:
`w = row · RW + (tile · m_tiles + m-tile) · 32 · Gmax + (c · Gmax + gi) · 2`,
`RW = n_tiles · m_tiles · 32 · Gmax` (`n_tiles = ⌈out_w / owpt⌉`, a fourth
divider at job start; `Gmax` from `owpt`).  Chunk rows: `4096 / RW`.  The
drain's dwt path walks (m-tile, channel, row, tile, group), reads the word
pair, saturates and interleaves it into 32 pixel slots, keeps the previous
group's slots beside the current one and selects the next 8 consecutive
valid pixels of the row (pixels of a tile are `[ow_start, ow_end)`, slot
`2m + x` is pixel `pair_base + 32·gi + 2m + x`) — one y word per cycle into
the same run / word FIFOs as today (runs of ≤ 256 pixels per channel).

**What is not in v1**: windows that do not fit 16 lanes as a pixel-pair
union (5 × 5 and up, 7 × 1 at stride 2, …) and stride_w > 2 — they keep the
diagonal path; double-buffered lane capture (a ~5-cycle bubble per instant
today's design accepts); the loads still precede a row's instants in the
patch FSM (B3).

**Result (2026-10-09): correct, and slower.**  The mode is bit-exact
against the HLS C++ on the fixtures, 800 random jobs (three seeds, fast /
random / slow memory), the 60 depthwise geometries of LightStereo-S and 33
hand-made edge cases (odd tile starts, stride 2, dilations, 7 × 1, 17–960
channels, batch 2, one pixel, 4096-wide rows).  Two bugs found on the way
are worth remembering: the lane table must be cleared per job (lanes past
the new table carried an earlier job's tap flags), and `cv_div` reads its
divisor every cycle (a one-cycle divisor select gave the chunk height of the
wrong layout, and rows past the buffer wrapped onto row 0).  But it is slower
than the diagonal path on every depthwise shape of LightStereo-S, 2.1–3.9×
with ideal memory and 1.1–2.7× with the slow-memory model (which reproduced
the board in §3), so the code was first gated off (`DWT_ENABLE = 0`: the shipped kernel
behaviourally identical, no bitstream change) and then reverted.  Cycles from `ap_start` to `ap_done` (`Vtb --timing fast` /
`slow`, `--no-oracle`; the mode's binary is kept as `vl/Vtb_dwt_on`):

| job | diagonal, ideal | transposed, ideal | × | diagonal, slow | transposed, slow | × |
|---|---:|---:|---:|---:|---:|---:|
| 3 × 3, 32 ch, 240 × 320 | 914 k | 1 940 k | 2.1 | 1 258 k | 1 949 k | 1.55 |
| 3 × 3 s2, 96 ch, 240 × 320 → 120 × 160 | 1 184 k | 3 257 k | 2.8 | 3 045 k | 3 264 k | 1.07 |
| 3 × 3, 144 ch, 120 × 160 | 1 147 k | 2 478 k | 2.2 | 1 949 k | 2 485 k | 1.28 |
| 3 × 3, 192 ch, 60 × 80 | 361 k | 871 k | 2.4 | 565 k | 877 k | 1.55 |
| 3 × 3, 384 ch, 30 × 40 | 178 k | 553 k | 3.1 | 244 k | 557 k | 2.29 |
| 3 × 3, 960 ch, 15 × 20 | 119 k | 399 k | 3.3 | 284 k | 403 k | 1.42 |
| 1 × 7, 96 ch, 60 × 80 | 140 k | 325 k | 2.3 | 206 k | 332 k | 1.61 |
| 7 × 1, 96 ch, 60 × 80 | 158 k | 480 k | 3.0 | 320 k | 526 k | 1.64 |
| 1 × 7 dilation 3, 96 ch, 60 × 80 | 144 k | 557 k | 3.9 | 207 k | 562 k | 2.72 |

Why — the operand bandwidth, not the mapping.  The diagonal mode computes
32 outputs (16 channels × 2 pixels) per instant at one instant per cycle,
reading 2 columns × 16 channels = 32 elements per cycle from the line
buffer: 9 cycles per 32 outputs of a 3 × 3.  The transposed mode computes
32 outputs (1 channel × 32 pixels) per instant too, but its windows reach
the grid lane by lane at 16 elements per cycle — `L` lanes plus the capture
handshake, ~20 cycles per 32 outputs.  Both are bound by what the line
buffer delivers: 2 ports × 16 banks × 1 element.  The diagonal path already
sits at that bound, and no re-mapping of the same 512 DSPs changes it.  A
real gain needs *reuse* of what is read: the 32 outputs of a 3 × 3 need only
3 rows × 34 columns = 102 distinct elements, not 288 — read each window row
once per instant (16 banks of 4-element words: a whole 64-column row per
port and cycle), rotate it once to the instant's origin (one 64-element
barrel shifter, pipelined), and take every lane of every column from the
rotated rows by fixed wiring (per-element 8 : 1 muxes for `dw` ∈ 1..4,
`sw` ∈ 1..2).  That gives ~3 cycles per 32 outputs for a 3 × 3 (3× the
diagonal rate), 1–2 for a 1 × k, about 7 for a 7 × 1; the estimate is
+15 k LUT (a 6 k rotator, 7 k selection muxes, the row and capture
registers) on a kernel of 18 k, in a design at 62 k of 117 k.  It is a
second rewrite of `cv_patch` (the loader writes 4-element words; the
standard path selects its element from a word) and of the capture side of
`cv_engine`; the sequencer, accumulator layout and drain of this
implementation stay.  Before it, the x path (§4 B) still bounds everything
on the board.

**Synthesis of the enabled datapath** (`synth_conv_rtl`, out of context,
300 MHz): 47 828 LUT and 48 930 FF against the shipped kernel's 17 835 /
39 630 (routed), the same 52 block RAM tiles, 48 UltraRAMs and 518 DSPs; WNS
−3.969 ns (Fmax ≈ 137 MHz).  Of the +30 k LUT, 20 k are `cv_drain`'s pixel
packer: the variable-position insert of a 32-element group into the
48-element queue (`q_px[n + i] = in[i]`) synthesises to a mux network per
element, and every worst path is `q_n` → `q_px` through it.  `cv_engine`'s
lane capture and bypass cost 3.2 k LUT and 9 k FF, `cv_xload`'s channel
select 2.5 k, `cv_patch`'s bank select 2 k.  Any future enablement replaces
the packer: a 64-slot ring with per-lane read indices (8 × 64 : 1 muxes,
~2.7 k LUT) or a two-stage pipelined shifter, and the capture double-buffered
behind a register stage.  (The gated-off synthesis was not completed: the
code was reverted instead.)

**Verification done**: with the mode on — the 63 fixtures, 800 random jobs
(seeds 1–3; fast, random and slow memory; the generator draws depthwise
strides 1–9 and dilations up to 10, so both paths ran), the 60 depthwise
geometries of LightStereo-S and 33 edge cases, all bit-exact against the
HLS C++; with the mode off — the fixtures + 200 random and 200 random
slow, bit-exact; `lint_conv_rtl` clean; `ConvRtlDsp` untouched.  After the
revert the tree's kernel passes the same regression (fixtures + 200 random).

## 7. Reproduction

```bash
# board probes (chat server stopped; the configs: ConvKernel cases with is_dw 1 / 0)
cd inference-scheduler && .venv/bin/python run_remote_perf.py --config CASES.json --json OUT.json
# ideal / slow memory
build/kernels/conv_rtl/vl/Vtb --no-oracle --timing fast --case "1 16 120 160 16 120 160 3 3 1 1 1 1 1 1 1 1"
build/kernels/conv_rtl/vl/Vtb --no-oracle --timing slow --case "1 144 120 160 16 120 160 1 1 1 1 1 1 0 0 1 0"
# the cost model's split (sweep / loads) and the slice choice
.venv/bin/python -c "from src import cost_model as cm; print(cm.rtl_conv_walk(16,16,120,160,120,160,3,3,dwise=True,board=True))"
```
