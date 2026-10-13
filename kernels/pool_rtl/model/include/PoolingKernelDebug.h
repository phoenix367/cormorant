#pragma once

// ---------------------------------------------------------------------------
// PoolingKernelDebug.h — C-simulation-only debug interface for PoolingKernel.
//
// When PoolingKernel.cpp is compiled with DEBUG_LOAD_DATA_CACHING (auto-
// enabled for non-synthesis builds), the kernel tracks every DDR address it
// reads from `x[]` during a single PoolingKernel invocation.  Any address
// read more than once in that invocation is counted here as a "duplicate
// read" — a regression in the on-chip caching of pool-window pixels (or, in
// the current pool kernel, an opportunity for future caching to claw back).
//
// Usage from a testbench:
//   pool_debug_reset_duplicate_reads();
//   PoolingKernel(...);
//   if (pool_debug_duplicate_read_count() != 0) { /* fail or report */ }
//
// In synthesis builds these symbols still link (no-op / always 0) so the
// header is safe to include unconditionally — same convention as
// ConvKernelDebug.h in the conv subdirectory.
// ---------------------------------------------------------------------------

void     pool_debug_reset_duplicate_reads();
unsigned pool_debug_duplicate_read_count();
