"""DRAM bank phases for the DMA pool layout (doc/plans/PS_PORTS_PLAN.md §11).

The KV260's DDR controller maps byte address bits 14–15 to the DRAM bank
(bit 6 the bank group, 7–13 the column of an 8 KB row, 16+ the row), so a
sequential stream returns to the same bank every 64 KB, and two streams
whose start addresses differ by a multiple of 64 KB sit in the same bank
with different rows at every moment — a row miss on every switch between
them.  A binary VectorOP keeps three such streams open (a, b, c) and loses
~20 % to it when they share a bank; the other kernels do not care.

The pool allocator therefore gives the slots that a VectorOP call reads or
writes at full rate start addresses of different *bank phase*
(``(byte_offset >> 14) & 3``): ``place_slots`` lays the slots end to end as
before, but pads a slot that has placed partners up to the next 16 KB
boundary whose phase collides with none of them (≤ 48 KB per constrained
slot; unconstrained slots keep their 64-byte packing).  Bit-exact: only
addresses move.
"""

from typing import Dict, Iterable, List, Sequence, Tuple

BANK_SHIFT = 14                         # byte address bits 14–15: the bank
BANK_COUNT = 4
BANK_STRIDE = 1 << BANK_SHIFT           # 16 KB between phases
BANK_PERIOD = BANK_STRIDE * BANK_COUNT  # 64 KB: the same bank again

BANK_MIN_STREAM = BANK_STRIDE           # a stream shorter than a bank phase is not worth padding for

__all__ = ["BANK_COUNT", "BANK_MIN_STREAM", "BANK_PERIOD", "BANK_STRIDE", "bank_phase",
           "place_slots", "vectorop_groups"]


def bank_phase(byte_off: int) -> int:
    """The DRAM bank phase (0..3) of a byte offset."""
    return (byte_off >> BANK_SHIFT) & (BANK_COUNT - 1)


def vectorop_groups(nodes: Iterable, root_of, in_pool, bpe: int = 2) -> List[List[Tuple[str, int]]]:
    """The operand streams of every VectorOPKernel node, as groups of
    ``(pool buffer name, byte shift)``: the output c and the input a always,
    the input b when the kernel streams it (a binary op whose b advances; a
    broadcast b with ``b_inc`` 0 is replayed on chip) — each only when it is
    at least ``BANK_MIN_STREAM`` bytes long.  ``root_of(name)`` resolves a
    tensor to ``(owning buffer name, byte offset)`` through Reshape aliases
    and Slice views; ``in_pool(name)`` says whether that buffer is in the DMA
    pool (graph inputs / outputs are the caller's).  Groups with fewer than
    two distinct pool buffers are dropped."""
    from .nodes import ScheduledNode
    groups = []
    for sn in nodes:
        if not isinstance(sn, ScheduledNode):
            continue
        streams = [sn.inputs[0], sn.output]
        if sn.arity == 2 and (sn.outer_count == 1 or sn.b_advances):
            streams.append(sn.inputs[1])
        g = []
        for t in streams:
            if t.numel * bpe < BANK_MIN_STREAM:
                continue
            root, shift = root_of(t.onnx_name)
            if in_pool(root) and all(root != r for r, _s in g):
                g.append((root, shift))
        if len(g) >= 2:
            groups.append(g)
    return groups


def place_slots(slots: Sequence[Tuple[int, Sequence[str]]],
                groups: Sequence[Sequence[Tuple[str, int]]],
                bpe: int, base_elems: int = 0,
                fixed: Dict[str, int] = None) -> List[int]:
    """Start offsets (elements, relative to the region's start) of ``slots``
    laid end to end — each ``(alloc_elems, names)`` already a multiple of the
    64-byte alignment — with the bank phases of ``groups`` (``vectorop_groups``)
    kept apart.  ``base_elems`` is the region's offset in the pool (the
    phases are absolute within the pool); ``fixed`` gives the absolute byte
    offsets of buffers placed before (the weights, for an entry's
    intermediates).  Returns one start per slot, in order."""
    fixed = fixed or {}
    slot_of: Dict[str, int] = {}
    for i, (_alloc, names) in enumerate(slots):
        for n in names:
            slot_of[n] = i
    # partners[i] = [(shift_i, other, shift_other)], other = slot index or ("fixed", byte_off)
    partners: Dict[int, List[Tuple[int, object, int]]] = {}
    for g in groups:
        for name, shift in g:
            if name not in slot_of:
                continue
            i = slot_of[name]
            for oname, oshift in g:
                if oname == name:
                    continue
                if oname in slot_of:
                    if slot_of[oname] != i:
                        partners.setdefault(i, []).append((shift, slot_of[oname], oshift))
                elif oname in fixed:
                    partners.setdefault(i, []).append((shift, ("fixed", fixed[oname]), oshift))

    starts: List[int] = []
    cursor = 0
    for i, (alloc, _names) in enumerate(slots):
        cons = partners.get(i, ())
        if not cons:
            starts.append(cursor)
            cursor += alloc
            continue
        byte0 = (base_elems + cursor) * bpe
        best = None
        for p in range(BANK_COUNT):
            delta = (p - bank_phase(byte0)) % BANK_COUNT
            if delta == 0:
                start_byte = byte0
            else:
                start_byte = ((byte0 >> BANK_SHIFT) + delta) << BANK_SHIFT
            start = (start_byte // bpe) - base_elems
            conflicts = 0
            for shift, other, oshift in cons:
                if isinstance(other, tuple):
                    obyte = other[1] + oshift
                elif other < i:
                    obyte = (base_elems + starts[other]) * bpe + oshift
                else:
                    continue                      # placed later: it will see us
                if bank_phase(start_byte + shift) == bank_phase(obyte):
                    conflicts += 1
            key = (conflicts, start - cursor)
            if best is None or key < best[0]:
                best = (key, start)
        starts.append(best[1])
        cursor = best[1] + alloc
    return starts
