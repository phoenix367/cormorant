# Copyright 2025 GradeBuilder SL
# SPDX-License-Identifier: Apache-2.0

"""
TensorLayout: unified DMA buffer layout descriptor for all kernel types.

A TensorLayout captures how a tensor's logical elements are laid out in its
physical DMA buffer, including any alignment gaps inserted to satisfy the
AXI burst-alignment requirement.

Layout kinds
------------
Flat (n_chunks == 1, stride == chunk == numel, alloc == numel):
    Contiguous elements with no padding.  alloc == numel.

Single padded block (n_chunks == 1, stride > chunk):
    One data block followed by alignment padding.  Used for repeating
    inputs of broadcast VectorOP nodes (bias at offset 0 each iteration).
    alloc == stride.

Advancing-strided (n_chunks > 1, stride >= chunk):
    n_chunks data blocks, each of `chunk` data elements followed by
    (stride - chunk) gap/padding elements.  alloc == n_chunks * stride.
    Used for advancing inputs and outputs of broadcast VectorOP nodes
    and for tensors whose alloc was propagated from such nodes.
    gap > 0 iff stride > chunk (i.e. alignment padding is present).
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class TensorLayout:
    """DMA buffer layout for one tensor.

    Invariants:
        alloc == n_chunks * stride          (when n_chunks > 1)
        alloc == stride                     (when n_chunks == 1)
        stride >= chunk
        chunk <= numel
    """

    numel:    int   # logical element count (data only, no gaps)
    alloc:    int   # DMA buffer elements (>= numel, includes alignment gaps)
    n_chunks: int   # >1 = advancing-strided; 1 = flat or single padded block
    chunk:    int   # data elements per chunk
    stride:   int   # buffer elements per chunk (stride >= chunk)

    @property
    def is_strided(self) -> bool:
        """True when the buffer has alignment gaps (stride > chunk)."""
        return self.stride > self.chunk

    @property
    def gap(self) -> int:
        """Alignment gap elements appended after each chunk."""
        return self.stride - self.chunk

    @staticmethod
    def flat(numel: int) -> "TensorLayout":
        """Contiguous single-block layout with no padding."""
        return TensorLayout(numel=numel, alloc=numel,
                            n_chunks=1, chunk=numel, stride=numel)

    @staticmethod
    def advancing(numel: int, n_chunks: int, stride: int) -> "TensorLayout":
        """Advancing-strided layout: n_chunks blocks each of stride elements."""
        chunk = numel // n_chunks
        return TensorLayout(numel=numel, alloc=n_chunks * stride,
                            n_chunks=n_chunks, chunk=chunk, stride=stride)

    @staticmethod
    def repeating(numel: int, alloc: int) -> "TensorLayout":
        """Single padded block; repeats at offset 0 each broadcast iteration."""
        return TensorLayout(numel=numel, alloc=alloc,
                            n_chunks=1, chunk=numel, stride=alloc)
