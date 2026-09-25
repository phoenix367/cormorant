"""
Engine cost model — estimated cycles of one ConvKernel / MatmulKernel call.

Used by the MatMul-on-ConvKernel lowering (``matmul_lowering.py``,
doc/BERT_PLAN.md §2 2A) to choose, per MatMul, between MatmulKernel and a
ConvKernel call with swapped operand roles, and to choose the lowered
conv's kernel width / output shape.  Both models count kernel clock cycles
(the board runs the fabric at 100 MHz); the absolute numbers matter only
relative to each other.

ConvKernel
----------
``conv_cycles`` is the standard-convolution path of the ConvKernel cycle
model, ``.claude/skills/conv-cycle-model/scripts/conv_cycle_model.py``
(architecture of CONV_OPTIMISATION.md §2.42: flat II=1 sweep over output
pixel PAIRS, ``G · max(kh·kw, 2)`` cycles per pair and M-group, one ramp per
``(ict, ow_tile, M-group)``, the §2.35 weight-slab prefetch, the §2.38
8-lane drain, the §2.39 row loader) with the same constants, so the two
cannot drift apart silently: ``test/test_matmul_on_conv.py`` compares them
on a geometry sweep.  That model tracked the RTL behavior test within ~5 %
on the > 20 k-cycle cases at every kernel step.

MatmulKernel
------------
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
"""

from __future__ import annotations

from functools import lru_cache

from ._conv_hw_config import (
    CONV_MAX_ACC_PERSIST_ENTRIES,
    CONV_MAX_LINE_BUF_COLS,
    CONV_MAX_LINE_BUF_ROWS,
    CONV_MAX_M_PER_GROUP,
    CONV_TILE_IC,
    CONV_TILE_M,
    CONV_WEIGHT_PORT_ELEMS,
)
from ._matmul_hw_config import MATMUL_TILE_M, MATMUL_TILE_N

# --------------------------------------------------------------------------
# ConvKernel — constants of conv_cycle_model.py (ARCH 42), keep in sync.
# --------------------------------------------------------------------------
CONV_INVOKE_OVERHEAD = 1000   # geometry dividers, bias load, DATAFLOW start-up
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
    owpt = min(ow, (CONV_MAX_LINE_BUF_COLS - winw) // sw + 1
               if winw < CONV_MAX_LINE_BUF_COLS else 1)
    if owpt > 1:
        owpt = min(ow, owpt & ~1)            # §2.42: even tile widths
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


@lru_cache(maxsize=4096)
def conv_cycles(in_ch: int, out_ch: int, in_h: int, in_w: int,
                oh: int, ow: int, kh: int, kw: int,
                sh: int = 1, sw: int = 1, dh: int = 1, dw: int = 1,
                pt: int = 0, pl: int = 0) -> dict:
    """Estimated cycles of one standard (group = 1) ConvKernel invocation
    with batch 1, split like conv_cycle_model.py's buckets:
    ``{total, sweep, fill, ph1, ph3, loads, chunks, rows, groups,
    ic_tiles, owt}``."""
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
                    f  = mv_sum * kh * kw * lanes / E
                    s_ = rows * units * G + PIXEL_OVERHEAD
                    first = c == 0 and ict == 0 and t == 0 and g == 0
                    fill  += f if first else max(0.0, f - s_ / 2)
                    sweep += s_
    total = sweep + fill + ph1 + ph3 + loads + CONV_INVOKE_OVERHEAD
    return dict(total=total, sweep=sweep, fill=fill, ph1=ph1, ph3=ph3, loads=loads,
                chunks=chunks, rows=per, groups=groups, ic_tiles=ic_tiles, owt=owt)


def conv_batch_cycles(batch: int, **geom) -> float:
    """A ConvKernel call with ``batch`` images: the per-image work repeats,
    the invocation overhead does not (the conv-cycle-model --validate rule)."""
    t = conv_cycles(**geom)["total"]
    return (t - CONV_INVOKE_OVERHEAD) * batch + CONV_INVOKE_OVERHEAD


def matmul_cycles(n: int, k: int, m: int, batch: int = 1) -> float:
    """Estimated cycles of one MatmulKernel call (``batch`` slices)."""
    blocks = batch * -(-n // MATMUL_TILE_N) * -(-m // MATMUL_TILE_M)
    return blocks * (MM_K_CYCLE * MATMUL_TILE_N * k + MM_BLOCK_OVERHEAD) + MM_INVOKE_OVERHEAD


def cycles_to_ms(cycles: float, mhz: float = 100.0) -> float:
    return cycles / (mhz * 1e3)


__all__ = (
    "CALL_OVERHEAD",
    "conv_cycles",
    "conv_batch_cycles",
    "matmul_cycles",
    "cycles_to_ms",
)
