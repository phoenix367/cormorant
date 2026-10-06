"""
Engine cost model — estimated cycles of one ConvKernel / MatmulKernel call.

Used by the MatMul-on-ConvKernel lowering (``matmul_lowering.py``,
doc/plans/BERT_PLAN.md §2 2A) to choose, per MatMul, between MatmulKernel and a
ConvKernel call with swapped operand roles, and to choose the lowered
conv's kernel width / output shape.  Both models count kernel clock cycles
(the board runs the fabric at 100 MHz); the absolute numbers matter only
relative to each other.

ConvKernel in SystemVerilog
---------------------------
With ``kernels.conv.impl == "rtl"`` (the KV260's since CONV_RTL_PLAN phase 3;
env ``AXI_CONV_IMPL``) the bitstream carries the RTL kernel
(``kernels/conv_rtl``, doc/kernels/CONV_RTL_KERNEL.md), and ``conv_cycles`` /
``conv_board_cycles`` use ``rtl_conv_walk``: the kernel's sweeps as a
pipeline recurrence (weight loader, the two weight-cache banks, patch
producer and x loader, the sweep, the drain beside the next chunk; the
comment above ``RTL_CONV_SIM``), with one parameter set tuned to the
Verilator testbench and one to the board (``RTL_CONV_BOARD``, the
calibration campaign of c2b2a6e5e50e).  The conv-cycle-model skill's script
calls it for ``--arch rtl`` (its default).

ConvKernel (HLS, ``kernels.conv.impl == "hls"``)
------------------------------------------------
The kernel of the bitstreams built before CONV_RTL_PLAN phase 3
(dbb320fb7297 and older).  ``conv_cycles`` is the standard-convolution path
of the ConvKernel cycle model,
``.claude/skills/conv-cycle-model/scripts/conv_cycle_model.py --arch 42``
(architecture of CONV_OPTIMISATION.md §2.42: flat II=1 sweep over output
pixel PAIRS, ``G · max(kh·kw, 2)`` cycles per pair and M-group, one ramp per
``(ict, ow_tile, M-group)``, the §2.35 weight-slab prefetch, the §2.38
8-lane drain, the §2.39 row loader) with the same constants, so the two
cannot drift apart silently: ``test/test_matmul_on_conv.py`` compares them
on a geometry sweep.  That model tracked the RTL behavior test within ~5 %
on the > 20 k-cycle cases at every kernel step.  On the board a weight
slab's fetch is also bound by its requests — one per output channel, 2
words for a 1x1 kernel —, which the simulation's DDR model hides:
``conv_board_cycles`` adds that (``CONV_WEIGHT_REQ_CYCLES``,
``CONV_PREFETCH_HIDE``, fitted to the board).

For either kernel the engine choices price ConvKernel against MatmulKernel
with ``conv_board_cycles`` (``matmul_lowering``: the geometry
``conv_cycles`` ranks first; ``fc_conv``), while the geometries themselves —
a lowered MatMul's (kw, out_w), the frontends' attention widths — are
ranked by ``conv_cycles``.

MatmulKernel (HLS, ``kernels.matmul.impl == "hls"``)
-----------------------------------------------------
The kernel of the bitstreams built before MATMUL_RTL_PLAN phase 4.
``matmul_cycles`` is a block model of MatmulKernel after Track A
(MATMUL_OPTIMISATION.md §4–§8): the output is computed in
``ceil(n / kTileN) × ceil(m / kTileM)`` blocks, each a K-loop of
``kTileN · k`` cycles (``kTileN`` row lanes × ``kTileM`` = 32 columns, 32
MACs per cycle) plus a fixed per-block cost (A-row requests, C write,
ramps).  Calibrated on the board (KV260, 100 MHz, BERT_PLAN.md §3 phase-1
profile and MATMUL_OPTIMISATION.md §9): 256³ 7.24 ms (model 7.36),
128³ 1.17 (1.18), 64³ 0.219 (0.21), BERT 256×768·768×768 54.3 ms (54.4),
256×3072·3072×768 203 ms (200), attention QKᵀ 256×64·64×256 3.28 ms
(3.30), P·V 256×256·256×64 1.94 ms (1.84).  For ``n < kTileN`` (K-split
over the lanes) the same formula matches the B-port bound of the one-row
FC layers (1×1280·1280×1001: 1.75 ms, model 1.82).

MatmulKernel in SystemVerilog
-----------------------------
With ``kernels.matmul.impl == "rtl"`` in the platform JSON (the KV260's
since MATMUL_RTL_PLAN phase 4) the bitstream carries the RTL kernel
(``kernels/matmul_rtl``, doc/kernels/MATMUL_RTL_KERNEL.md), and
``matmul_cycles`` / ``gemv_cycles`` use ``rtl_matmul_cycles``: a structural
model of its job walk — panels of 8 A rows × column chunks of at most 512
accumulator columns; per step each lane streams its half of the K planes
(blocks of 16, the odd last one split) at one 128-bit beat per cycle and at
least ``DMIN`` cycles per row, then the drain reads the accumulators; the
A panel loads before its first step.  ``rtl_matmul_terms`` lists the terms,
weighted by ``RTL_COEF``: a non-negative fit to the 414 MatmulKernel calls
of the calibration campaign of b3309f424562 (the IP with its m_axi bus
parameters, MATMUL_RTL_PLAN phase 5; leave-one-out median error 0.35 %,
p90 1.7 %; the previous fit, to 240 calls of 1d28630fbfa4 with every
crossbar slot at 2 outstanding bursts, predicted it at 1.3 / 3.1 % and gave
the shipped models the same engine choices).  The same terms are the perf
models' MatmulKernel features (``perf_model.features``, ``rtl_*``).
"""

from __future__ import annotations

from functools import lru_cache

from ._conv_hw_config import (
    CONV_IMPL,
    CONV_MAX_ACC_PERSIST_ENTRIES,
    CONV_MAX_LINE_BUF_COLS,
    CONV_MAX_LINE_BUF_ROWS,
    CONV_MAX_M_PER_GROUP,
    CONV_TILE_IC,
    CONV_TILE_M,
    CONV_WEIGHT_PORT_ELEMS,
)
from ._matmul_hw_config import MATMUL_GEMV_MAX_M, MATMUL_IMPL, MATMUL_TILE_M, MATMUL_TILE_N

# --------------------------------------------------------------------------
# ConvKernel — constants of conv_cycle_model.py (ARCH 42), keep in sync.
# --------------------------------------------------------------------------
CONV_INVOKE_OVERHEAD = 1000   # geometry dividers, bias load, DATAFLOW start-up
# The board only (conv_board_cycles): cycles per weight request (one per
# output channel of a slab, 8 in flight) and the share of a slab's sweep
# that hides the next slab's fetch (one word per iteration); the RTL
# simulation's model (conv_cycles, the skill's) uses 0 and 0.5.
CONV_WEIGHT_REQ_CYCLES = 12
CONV_PREFETCH_HIDE   = 1.0
DRAIN_SEG            = 256    # §2.38 Phase-3 transposer segment (pixels)
DRAIN_STEP_RAMP      = 6
ROW_FILL_LATENCY     = 12     # §2.39 per-row Phase-1 entry
ROW_LOADER_SETUP     = 64     # §2.39 per-row request loop + first-data latency
PIXEL_OVERHEAD       = 12     # one sweep ramp per (ict, ow_tile, M-group)

# --------------------------------------------------------------------------
# MatmulKernel block model (board-calibrated, see the module docstring).
# --------------------------------------------------------------------------
MM_K_CYCLE           = 1.03   # cycles per (row lane, k) step of the K-loop
MM_BLOCK_OVERHEAD    = 380    # per (n_tile, m_tile) block
MM_INVOKE_OVERHEAD   = 1000

# MatmulKernel GEMV streaming mode (MATMUL_OPTIMISATION §8b): B streamed once
# per A row, half through each read port at one 8-element word per cycle.
# Not yet board-calibrated: the per-word cost is the tiled path's (the same
# port, the same request shape), the job overhead the x load, the tap
# preload and the pipeline ramps; the writer then drains m columns.
GEMV_WORD_CYCLE      = 1.03
GEMV_JOB_OVERHEAD    = 200

# Host-side cost of one kernel call on the board (AXI-Lite register writes
# through the UIO mapping, Start, the IsDone poll loop), in kernel cycles.
# Charged per call to both engines, so it only matters where the lowering
# issues several conv calls for one MatMul (batched attention: one per head).
CALL_OVERHEAD        = 1500


def _conv_geom(in_ch, out_ch, oh, ow, kh, kw, sh, sw, dh, dw):
    m_tiles  = -(-out_ch // CONV_TILE_M)
    ic_tiles = -(-in_ch // CONV_TILE_IC)
    mtg      = min(CONV_MAX_M_PER_GROUP, m_tiles)
    groups   = -(-m_tiles // mtg)
    per = max(1, min(oh, CONV_MAX_ACC_PERSIST_ENTRIES // (ow * m_tiles * CONV_TILE_M)))
    if groups > 1:
        win = (kh - 1) * dh + 1
        cap = (CONV_MAX_LINE_BUF_ROWS - win) // sh + 1 if win < CONV_MAX_LINE_BUF_ROWS else 1
        per = min(per, cap)
    chunks = -(-oh // per)
    winw = (kw - 1) * dw + 1
    owpt = (CONV_MAX_LINE_BUF_COLS - winw) // sw + 1 if winw < CONV_MAX_LINE_BUF_COLS else 1
    if owpt > 1:
        owpt &= ~1                           # §2.42: even tile widths ...
    owpt = min(ow, owpt)                     # ... then at most out_w (compute_ow_tiling)
    owt = -(-ow // owpt)
    return m_tiles, ic_tiles, mtg, groups, per, chunks, owpt, owt


def _tile_pixel_units(ow, owpt, t, kh, kw):
    """Sweep iterations per (row, ow-tile) per m-tile: output columns in
    pairs aligned to even ow, max(kh·kw, 2) positions each (§2.42)."""
    ow_start = t * owpt
    ow_end   = min(ow, ow_start + owpt)
    n_pairs  = ((ow_end - 1) >> 1) - (ow_start >> 1) + 1
    return n_pairs * max(kh * kw, 2)


def _loader_row_cycles(ch, cols):
    return ch * (-(-cols // 8) + 1) + ch + ROW_LOADER_SETUP


# --------------------------------------------------------------------------
# ConvKernel in SystemVerilog (platform ``kernels.conv.impl == "rtl"``).
# --------------------------------------------------------------------------
# The kernel's sweeps as a pipeline recurrence (CONV_RTL_KERNEL.md §2-§3).
# Per sweep k, in the sequencer's order, with s_k grid instants (output rows
# x pixel pairs x max(kh*kw, 2) x the group's m-tiles), b_k weight beats and
# w_k weight-cache writes of its slab, and p_k the patch producer's cycles
# (its beats, a cost per output row, and on a loading sweep the input rows at
# the pace of the line buffer or of the x loader, whichever is slower):
#
#   weight loader  L_k = max(L_{k-1}, F_{k-1} - CAP) + b_k      (its FIFO runs ahead)
#   cache fill     F_k = max(max(F_{k-1}, S_{k-2}) + w_k, L_k)  (two banks)
#   producer       P_k = P_{k-1} + p_k
#   sweep          S_k = max(max(S_{k-1} + H (+ HZW), F_k + FL, D_{c-2}) + s_k, P_k + PL)
#   drain          D_c = max(D_{c-1}, S_last(c) + LAT) + DPX * (1 or 2 cycles per pixel and m-tile)
#
# and the job ends with the last chunk's drain plus ONE.  The DDR's runs: an x
# run (one per input row and channel) costs its words plus RUN, a weight run
# (one per output channel of the slab) its beats; with XLAT / WLAT, 16 x and
# 8 weight runs in flight bound them too.  RTL_CONV_SIM is tuned to the
# Verilator testbench (ideal memory) on the 1 042 ConvKernel calls of the
# calibration case list — median error 0.65 %, p90 5.3 % —, RTL_CONV_BOARD to
# the board's (bitstream c2b2a6e5e50e, the same calls: median 3.1 %, p90
# 14.6 %, against 18.7 / 37 % for the HLS model on its bitstream).  The worst
# misses are narrow, tall MatMul-on-ConvKernel jobs (a few short x runs per
# row whose DDR latency depends on the channel-plane stride: up to 2.5x
# slower than modelled).  ONE excludes the host's call overhead (CALL_OVERHEAD).
RTL_CONV_SIM = {"CAP": 256, "XL": 0, "PL": 0, "FL": 0, "H": 2, "HAZ": 80, "HZW": 50,
                "LAT": 10, "ROW": 1, "ONE": 400, "XBEAT": 0.9, "XROW": 12, "RUN": 0,
                "XLAT": 0, "WLAT": 0, "WBEAT": 1.0, "DPX": 1.1}
RTL_CONV_BOARD = {"CAP": 256, "XL": 0, "PL": 20, "FL": 50, "H": 20, "HAZ": 80, "HZW": 50,
                  "LAT": 10, "ROW": 2, "ONE": 290, "XBEAT": 0.9, "XROW": 12, "RUN": 1,
                  "XLAT": 50, "WLAT": 30, "WBEAT": 1.05, "DPX": 1.1}


@lru_cache(maxsize=16384)
def rtl_conv_walk(in_ch: int, out_ch: int, in_h: int, in_w: int, oh: int, ow: int,
                  kh: int, kw: int, sh: int = 1, sw: int = 1, dh: int = 1, dw: int = 1,
                  pt: int = 0, pl: int = 0, dwise: bool = False, board: bool = False) -> dict:
    """The RTL kernel's job (batch 1) through the recurrence above:
    ``{total, sweep, fill, ph1 (0: no bias pass), ph3 (the drain where it is not
    hidden), loads (where the producer bounds a sweep), chunks, rows, groups,
    ic_tiles, owt}`` — the keys of the HLS model's walk.  ``dwise``: a
    depthwise job (one m-tile per sweep, its own input tile)."""
    p = RTL_CONV_BOARD if board else RTL_CONV_SIM
    m_tiles, ic_tiles, mtg, groups, per, chunks, owpt, owt = _conv_geom(
        in_ch, out_ch, oh, ow, kh, kw, sh, sw, dh, dw)
    if dwise:                                # no m-group cap on the chunk height
        per = max(1, min(oh, CONV_MAX_ACC_PERSIST_ENTRIES // (ow * m_tiles * CONV_TILE_M)))
        chunks, groups, mtg = -(-oh // per), 1, 1
    E, T = CONV_WEIGHT_PORT_ELEMS, CONV_TILE_IC
    npos = kh * kw
    nwin = max(npos, 2)
    L = F = Pd = S = S2 = 0.0
    D = [0.0, 0.0]
    prev_s = 0
    sweep = fill = loads = dwait = 0.0
    for c in range(chunks):
        rows = min(per, oh - c * per)
        r0 = c * per * sh - pt
        r1 = (c * per + rows - 1) * sh + (kh - 1) * dh - pt
        in_rows = max(0, min(r1, in_h - 1) - max(r0, 0) + 1)
        if sh > (kh - 1) * dh + 1:           # rows between the windows are not read
            in_rows = sum(1 for r in range(max(r0, 0), min(r1, in_h - 1) + 1)
                          if (r + pt) % sh <= (kh - 1) * dh)
        first = True
        for ct in range(m_tiles if dwise else ic_tiles):
            icv = min(T, (out_ch if dwise else in_ch) - ct * T)
            half = not dwise and ct == ic_tiles - 1 and icv <= E
            for t in range(owt):
                cols = min(in_w, (min(owpt, ow - t * owpt) - 1) * sw + (kw - 1) * dw + 1)
                ow0 = t * owpt
                npairs = ((min(ow, ow0 + owpt) - 1) >> 1) - (ow0 >> 1) + 1
                for g in range(groups):
                    if dwise:
                        G, mv = 1, icv
                        b = mv * -(-npos // E)
                    else:
                        G = min(mtg, m_tiles - g * mtg)
                        mv = sum(min(CONV_TILE_M, out_ch - (g * mtg + i) * CONV_TILE_M)
                                 for i in range(G))
                        b = mv * npos * (1 if half else 2)
                    w = mv * npos
                    s = rows * npairs * nwin * G
                    b = max(b, (p["WLAT"] + b / mv) * mv / 8) * p["WBEAT"]
                    pk = rows * (npairs * npos + p["ROW"])
                    if dwise or g == 0:              # the m-group's first sweep loads
                        words = -(-cols // E) + p["RUN"]
                        xrow = max(icv * words * p["XBEAT"] + p["XROW"],
                                   (p["XLAT"] + words) * icv / 16)
                        pk = max(pk + in_rows * cols, in_rows * xrow) + p["XL"]
                    L = max(L, F - p["CAP"]) + b
                    Fn = max(max(F, S2) + w, L)
                    Pd += pk
                    h = p["H"] + (p["HZW"] if not dwise and ct > 0 and prev_s < p["HAZ"] else 0)
                    S0 = max(S + h, Fn + p["FL"])
                    fill += max(0.0, Fn + p["FL"] - (S + h))
                    if first and c >= 2:             # chunk c reuses chunk c-2's buffer
                        dwait += max(0.0, D[0] - S0)
                        S0 = max(S0, D[0])
                    Sn = max(S0 + s, Pd + p["PL"])
                    loads += Sn - (S0 + s)
                    sweep += s
                    S2, S, F = S, Sn, Fn
                    prev_s = s
                    first = False
        dr = p["DPX"] * sum(rows * ow * (2 if min(CONV_TILE_M, out_ch - mt * CONV_TILE_M) > E else 1)
                            for mt in range(m_tiles))
        D = [D[1], max(D[1], S + p["LAT"]) + dr]
    return dict(total=D[1] + p["ONE"], sweep=sweep, fill=fill, ph1=0.0, ph3=dwait + D[1] - S,
                loads=loads, chunks=chunks, rows=per, groups=groups, ic_tiles=ic_tiles, owt=owt)


def conv_invoke_overhead() -> float:
    """The fixed cycles of one ConvKernel call (its configuration, first
    loads, last drain), the part that does not repeat per image."""
    return RTL_CONV_SIM["ONE"] if CONV_IMPL == "rtl" else CONV_INVOKE_OVERHEAD


def conv_cycles(in_ch: int, out_ch: int, in_h: int, in_w: int,
                oh: int, ow: int, kh: int, kw: int,
                sh: int = 1, sw: int = 1, dh: int = 1, dw: int = 1,
                pt: int = 0, pl: int = 0) -> dict:
    """Estimated cycles of one standard (group = 1) ConvKernel invocation
    with batch 1, split like conv_cycle_model.py's buckets:
    ``{total, sweep, fill, ph1, ph3, loads, chunks, rows, groups,
    ic_tiles, owt}`` — the kernel with ideal memory (the skill's model): the
    SystemVerilog kernel's recurrence (``rtl_conv_walk``) with
    ``kernels.conv.impl == "rtl"``, the HLS kernel's walk with ``"hls"``."""
    if CONV_IMPL == "rtl":
        return rtl_conv_walk(in_ch, out_ch, in_h, in_w, oh, ow, kh, kw, sh, sw, dh, dw, pt, pl)
    return _conv_walk(in_ch, out_ch, in_h, in_w, oh, ow, kh, kw, sh, sw, dh, dw, pt, pl,
                      0, 0.5)


def conv_board_cycles(in_ch: int, out_ch: int, in_h: int, in_w: int,
                      oh: int, ow: int, kh: int, kw: int,
                      sh: int = 1, sw: int = 1, dh: int = 1, dw: int = 1,
                      pt: int = 0, pl: int = 0) -> dict:
    """:func:`conv_cycles` on the board: a weight slab's fetch is bound by
    its requests too (one per output channel of the slab, ``kh·kw·16``
    elements — 2 words for a 1x1 kernel —, 8 in flight:
    ``CONV_WEIGHT_REQ_CYCLES`` each), and the sweep hides it one word per
    iteration (``CONV_PREFETCH_HIDE``).  Fitted to the 962 measured
    ConvKernel calls of the RTL MatmulKernel's bitstream (1d28630fbfa4):
    median error 20.5 → 18.7 %, p90 46 → 37 %; a 1x1 kernel on <= 256
    pixels and > 64 output channels (BERT's per-head attention P·V:
    model 0.23 → 0.55 ms, board 0.61 ms) median |log error| 0.60 → 0.13.
    With ``kernels.conv.impl == "rtl"``: ``rtl_conv_walk`` with the board's
    parameters (RTL_CONV_BOARD)."""
    if CONV_IMPL == "rtl":
        return rtl_conv_walk(in_ch, out_ch, in_h, in_w, oh, ow, kh, kw, sh, sw, dh, dw, pt, pl,
                             board=True)
    return _conv_walk(in_ch, out_ch, in_h, in_w, oh, ow, kh, kw, sh, sw, dh, dw, pt, pl,
                      CONV_WEIGHT_REQ_CYCLES, CONV_PREFETCH_HIDE)


@lru_cache(maxsize=8192)
def _conv_walk(in_ch, out_ch, in_h, in_w, oh, ow, kh, kw, sh, sw, dh, dw, pt, pl,
               req: float, hide: float) -> dict:
    """The model's walk; ``req`` (cycles per weight request) and ``hide``
    (the share of a slab's sweep that absorbs the next slab's fetch) are
    the RTL model's 0 and 0.5, or the board's."""
    m_tiles, ic_tiles, mtg, groups, per, chunks, owpt, owt = _conv_geom(
        in_ch, out_ch, oh, ow, kh, kw, sh, sw, dh, dw)
    E, T = CONV_WEIGHT_PORT_ELEMS, CONV_TILE_IC
    sweep = fill = ph1 = ph3 = loads = 0
    fill_steps = sum(-(-min(CONV_TILE_M, out_ch - t * CONV_TILE_M) // E)
                     for t in range(m_tiles))
    for c in range(chunks):
        rows = min(per, oh - c * per)
        ph1 += rows * ow * m_tiles
        L = rows * ow
        nseg = -(-L // DRAIN_SEG)
        ph3 += fill_steps * L + min(L, DRAIN_SEG) + DRAIN_STEP_RAMP * (m_tiles * nseg + 1)
        r0 = c * per * sh - pt
        r1 = (c * per + rows - 1) * sh + (kh - 1) * dh - pt
        in_rows = max(0, min(r1, in_h - 1) - max(r0, 0) + 1)
        if sh > (kh - 1) * dh + 1:
            in_rows = sum(1 for r in range(max(r0, 0), min(r1, in_h - 1) + 1)
                          if (r + pt) % sh <= (kh - 1) * dh)
        for ict in range(ic_tiles):
            icv   = min(T, in_ch - ict * T)
            lanes = E if (ict == ic_tiles - 1 and icv <= E) else T
            for t in range(owt):
                tw    = min(owpt, ow - t * owpt)
                cols  = min(in_w, (tw - 1) * sw + (kw - 1) * dw + 1)
                units = _tile_pixel_units(ow, owpt, t, kh, kw)
                sweep_blk = sum(rows * units * min(mtg, m_tiles - g * mtg) + PIXEL_OVERHEAD
                                for g in range(groups))
                fill_c = in_rows * (cols + ROW_FILL_LATENCY)
                ldr    = in_rows * _loader_row_cycles(icv, cols)
                loads += fill_c + max(0, ldr - (sweep_blk + fill_c))
                for g in range(groups):
                    mt0 = g * mtg
                    G   = min(mtg, m_tiles - mt0)
                    mv_sum = sum(min(CONV_TILE_M, out_ch - (mt0 + i) * CONV_TILE_M)
                                 for i in range(G))
                    f  = max(mv_sum * kh * kw * lanes / E, mv_sum * req)
                    s_ = rows * units * G + PIXEL_OVERHEAD
                    first = c == 0 and ict == 0 and t == 0 and g == 0
                    fill  += f if first else max(0.0, f - s_ * hide)
                    sweep += s_
    total = sweep + fill + ph1 + ph3 + loads + CONV_INVOKE_OVERHEAD
    return dict(total=total, sweep=sweep, fill=fill, ph1=ph1, ph3=ph3, loads=loads,
                chunks=chunks, rows=per, groups=groups, ic_tiles=ic_tiles, owt=owt)


def conv_batch_cycles(batch: int, board: bool = False, **geom) -> float:
    """A ConvKernel call with ``batch`` images: the per-image work repeats,
    the invocation overhead does not (the conv-cycle-model --validate rule).
    ``board``: :func:`conv_board_cycles` instead of :func:`conv_cycles`."""
    t = (conv_board_cycles if board else conv_cycles)(**geom)["total"]
    one = (RTL_CONV_BOARD if board else RTL_CONV_SIM)["ONE"] if CONV_IMPL == "rtl" \
        else CONV_INVOKE_OVERHEAD
    return (t - one) * batch + one


def matmul_cycles(n: int, k: int, m: int, batch: int = 1, b_packed: bool = True) -> float:
    """Estimated cycles of one MatmulKernel call (``batch`` slices) on its
    tiled path, B packed (constant weights) or row-major; the model of the
    platform's MatmulKernel (``MATMUL_IMPL``)."""
    if MATMUL_IMPL == "rtl":
        return rtl_matmul_cycles(n, k, m, batch, b_packed, 0)
    blocks = batch * -(-n // MATMUL_TILE_N) * -(-m // MATMUL_TILE_M)
    return blocks * (MM_K_CYCLE * MATMUL_TILE_N * k + MM_BLOCK_OVERHEAD) + MM_INVOKE_OVERHEAD


def gemv_cycles(n: int, k: int, m: int, batch: int = 1, kw: int = 1) -> float:
    """Estimated cycles of one MatmulKernel GEMV call (``batch`` slices of
    ``n`` A rows; B is ``k x m``, ``kw`` its image's kernel width)."""
    if MATMUL_IMPL == "rtl":
        return rtl_matmul_cycles(n, k, m, batch, False, max(1, kw))
    chunks = max(1, -(-m // MATMUL_GEMV_MAX_M)) if MATMUL_GEMV_MAX_M else 1
    words  = k * m / 8
    per_row = (GEMV_WORD_CYCLE * words / 2
               + chunks * (k / 8 + kw + GEMV_JOB_OVERHEAD) + m)
    return batch * n * per_row + MM_INVOKE_OVERHEAD


# --------------------------------------------------------------------------
# MatmulKernel in SystemVerilog (platform ``kernels.matmul.impl == "rtl"``).
# --------------------------------------------------------------------------
RTL_ROWS   = 8      # A rows per panel (DSP rows per lane)
RTL_W_EL   = 512    # accumulator columns of one step (ACC_D x 8), << lk
RTL_DMIN   = 3      # minimum cycles per streamed B row (accumulator distance)
RTL_TILE   = 32     # packed-B DDR tile width (beats per row = 4 per tile)

# Cycles per term (rtl_matmul_terms), fitted on the board (see the module
# docstring); "one" is the job's fixed cost without the host's call
# overhead (3.15 us measured, charged by the callers as CALL_OVERHEAD).
RTL_COEF = {"stream": 0.986, "drain": 1.000, "aload": 0.733,
            "steps": 2.97, "runs": 0.325, "one": 96.4}


def _rtl_lane_planes(planes: int) -> int:
    """Planes the busier lane streams: blocks of 16 interleaved between the
    two lanes, an odd last block of more than 8 planes split 8 / rest."""
    if planes <= 0:
        return 0
    nb = -(-planes // 16)
    rl = planes - 16 * (nb - 1)
    l0 = 16 * (-(-nb // 2)) - (16 - rl if nb % 2 else 0)
    l1 = 16 * (nb // 2) - (0 if nb % 2 else 16 - rl)
    if nb % 2 and rl > 8:
        l0, l1 = l0 - (rl - 8), l1 + (rl - 8)
    return max(l0, l1)


@lru_cache(maxsize=65536)
def rtl_matmul_terms(n: int, k: int, m: int, batch: int = 1,
                     b_packed: bool = True, gemv_kw: int = 0) -> dict:
    """The RTL kernel's job walk as cycle terms: ``stream`` (B beats of the
    busier lane, at least RTL_DMIN per row), ``drain`` (accumulator words
    read), ``aload`` (A panel beats per port), ``steps``, ``runs`` (B read
    runs: per tile and block, per block or per plane) and ``one``.
    ``gemv_kw`` 1/2/4/8 is the GEMV image layout (1 = row-major B);
    ``b_packed`` applies to the tiled path only."""
    lk = gemv_kw.bit_length() - 1 if gemv_kw else 0
    kw = 1 << lk
    packed = bool(b_packed) and not gemv_kw
    lf = m << lk
    mc_max = RTL_W_EL if packed else (m if lf <= RTL_W_EL else RTL_W_EL >> lk)
    chunks = [mc_max] * (m // mc_max) + ([m % mc_max] if m % mc_max else [])
    planes = k >> lk
    lp = _rtl_lane_planes(planes)
    nblk = -(-planes // 16)
    contig = not packed and lf <= RTL_W_EL
    rows = [RTL_ROWS] * (n // RTL_ROWS) + ([n % RTL_ROWS] if n % RTL_ROWS else [])
    stream = drain = runs = 0.0
    for mcc in chunks:
        if packed:
            bpr, nr = 4 * -(-mcc // RTL_TILE), nblk * -(-mcc // RTL_TILE)
        else:
            bpr, nr = -(-(mcc * kw) // 8), (nblk if contig else planes)
        stream += len(rows) * lp * max(bpr, RTL_DMIN)
        drain += sum(rows) * -(-(mcc * kw) // 8)
        runs += len(rows) * nr
    return {"stream": batch * stream, "drain": batch * drain,
            "aload": batch * len(rows) * -(-(4 * k) // 8),
            "steps": batch * len(rows) * len(chunks), "runs": batch * runs, "one": 1.0}


def rtl_matmul_cycles(n: int, k: int, m: int, batch: int = 1,
                      b_packed: bool = True, gemv_kw: int = 0) -> float:
    """Estimated cycles of one call of the RTL MatmulKernel."""
    t = rtl_matmul_terms(n, k, m, batch, bool(b_packed), gemv_kw)
    return sum(RTL_COEF[name] * v for name, v in t.items())


def cycles_to_ms(cycles: float, mhz: float = 100.0) -> float:
    return cycles / (mhz * 1e3)


__all__ = (
    "CALL_OVERHEAD",
    "conv_cycles",
    "conv_board_cycles",
    "conv_batch_cycles",
    "conv_invoke_overhead",
    "rtl_conv_walk",
    "matmul_cycles",
    "gemv_cycles",
    "rtl_matmul_cycles",
    "rtl_matmul_terms",
    "cycles_to_ms",
)
