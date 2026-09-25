"""Mixin that generates src/inference_buf.c and scripts/check_inference_setup.sh."""

from __future__ import annotations
import os

from ._banners import _file_banner


# inference_buf.c — dual-platform DMA buffer allocator.
# The file banner (with model info / timestamp) is prepended by
# generate_buf_impl().  INFERENCE_BYTES_PER_ELEM comes from inference.h.
_BUF_IMPL_TEMPLATE = r"""
/* inference_buf_t struct is defined in inference.h so that inference.c
 * can declare static view instances.  This file provides the implementation. */
#include "inference.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

////////////////////////////////////////////////////////////////////////
/* Platform-independent accessors                                      */
////////////////////////////////////////////////////////////////////////

Data_t *inference_buf_ptr(inference_buf_t *buf)
{
    return (Data_t *)buf->virt;
}

uint64_t inference_buf_phys(const inference_buf_t *buf)
{
    return buf->phys;
}

unsigned inference_buf_count(const inference_buf_t *buf)
{
    return buf->count;
}

void inference_buf_init_view(inference_buf_t *view,
                              inference_buf_t *base,
                              unsigned offset_elems,
                              unsigned count_elems)
{
    uint64_t byte_off  = (uint64_t)offset_elems * INFERENCE_BYTES_PER_ELEM;
    view->virt         = (char *)base->virt + (size_t)byte_off;
    view->phys         = base->phys + byte_off;
    view->count        = count_elems;
    view->refcount     = 0u;
    view->is_owner     = 0u;
    view->cached       = base->cached;
#ifdef __linux__
    view->bo           = base->bo;
    view->bo_offset    = base->bo_offset + byte_off;
#endif
}

////////////////////////////////////////////////////////////////////////
/* Reference counting                                                  */
/*                                                                     */
/* inference_buf_retain / inference_buf_release implement a simple     */
/* reference-counted ownership model for owner buffers (is_owner==1). */
/* Views (is_owner==0) are backed by a parent owner buffer and their   */
/* refcount is always 0; retain/release are no-ops for them.           */
/*                                                                     */
/* Usage in inference_run():                                            */
/*   inference_buf_retain(out);     // borrow caller's output buffer   */
/*   internal_buf = out;            // redirect kernel output          */
/*   ... run kernels ...                                                */
/*   internal_buf = prev;           // restore                         */
/*   inference_buf_release(out);    // end borrow                      */
/*                                                                     */
/* This prevents premature free if the caller calls                    */
/* inference_buf_free() while inference_run() holds a reference.       */
////////////////////////////////////////////////////////////////////////

/* Forward declaration — platform-specific dealloc defined below. */
static void _inference_buf_dealloc(inference_buf_t *buf);

void inference_buf_retain(inference_buf_t *buf)
{
    if (buf && buf->is_owner)
        buf->refcount++;
}

void inference_buf_release(inference_buf_t *buf)
{
    if (!buf || !buf->is_owner)
        return;
    if (--buf->refcount == 0u)
        _inference_buf_dealloc(buf);
}

void inference_buf_free(inference_buf_t *buf)
{
    inference_buf_release(buf);
}

////////////////////////////////////////////////////////////////////////
/* Float cast helpers (platform-independent)                           */
/*                                                                     */
/* Uses C's built-in implicit conversion to/from Data_t so the        */
/* implementation is correct for any numeric Data_t (float, double,   */
/* int8_t, int16_t, uint8_t, …) without type-specific constants.      */
////////////////////////////////////////////////////////////////////////

void inference_buf_fill_float(inference_buf_t *buf,
                              const float *src, unsigned n)
{
    Data_t  *dst = (Data_t *)buf->virt;
    unsigned i;
    for (i = 0; i < n; i++)
        dst[i] = (Data_t)src[i];
}

void inference_buf_read_float(const inference_buf_t *buf,
                              float *dst, unsigned n)
{
    const Data_t *src = (const Data_t *)buf->virt;
    unsigned i;
    for (i = 0; i < n; i++)
        dst[i] = (float)src[i];
}

////////////////////////////////////////////////////////////////////////
/* Linux — XRT Buffer Object (BO) API                                 */
/*                                                                     */
/* xclAllocBO allocates physically contiguous (CMA) memory managed by  */
/* the XRT zocl driver; xclGetBOProperties.paddr is the physical       */
/* address to program into the kernels' AXI-Lite address registers.   */
/*                                                                     */
/* The PL kernels' AXI masters do not snoop the CPU caches.  The BOs   */
/* are mapped CACHEABLE by default (XCL_BO_FLAGS_CACHEABLE, as PYNQ's  */
/* allocate(cacheable=True)): CPU access runs at cached speed and      */
/* xclSyncBO does the cache maintenance on the requested range only:   */
/*   XCL_BO_SYNC_BO_TO_DEVICE   — clean: CPU writes reach DDR; must    */
/*                                precede any kernel read or write of  */
/*                                the range                            */
/*   XCL_BO_SYNC_BO_FROM_DEVICE — invalidate: drop stale lines after a */
/*                                kernel wrote the range, before the   */
/*                                CPU reads it                         */
/* Views (inference_buf_init_view) sync their own byte range of the    */
/* parent BO (size / offset arguments of xclSyncBO).                   */
/*                                                                     */
/* Fallback: -DINFERENCE_BUF_CACHEABLE=0 at build time, or the         */
/* environment variable INFERENCE_BUF_CACHEABLE=0 at run time, maps    */
/* the BOs non-cacheable (write-combine) as before: every CPU read of  */
/* a BO then goes to DDR (~7x slower sequential, ~100 ns per scattered */
/* 2-byte load), and the host ops stage through malloc'd memory.       */
/*                                                                     */
/* Mirrors PYNQ xrt_device.py allocate_bo / map_bo /                  */
/*   get_device_address / flush / invalidate.                          */
/*                                                                     */
/* Requires XRT runtime (/opt/xilinx/xrt) and access to the XRT       */
/* device node.                                                        */
////////////////////////////////////////////////////////////////////////

#ifdef __linux__

#include <stdio.h>
#include <xrt.h>

#ifndef INFERENCE_BUF_CACHEABLE
#  define INFERENCE_BUF_CACHEABLE 1
#endif
#ifndef XCL_BO_FLAGS_CACHEABLE
#  define XCL_BO_FLAGS_CACHEABLE (1U << 24)       /* xrt_mem.h */
#endif

static xclDeviceHandle s_xrt_dev   = NULL;
static int             s_cacheable = INFERENCE_BUF_CACHEABLE;

int inference_buf_pool_init(void)
{
    const char *env = getenv("INFERENCE_BUF_CACHEABLE");
    if (env && *env)
        s_cacheable = (strcmp(env, "0") != 0);
    if (s_xrt_dev)
        return 0;
    s_xrt_dev = xclOpen(0, NULL, (enum xclVerbosityLevel)XCL_QUIET);
    if (s_xrt_dev == NULL) {
        fprintf(stderr, "inference: xclOpen(0) failed — is XRT loaded?\n");
        return -1;
    }
    return 0;
}

void inference_buf_pool_deinit(void)
{
    if (s_xrt_dev) {
        xclClose(s_xrt_dev);
        s_xrt_dev = NULL;
    }
}

static void _inference_buf_sync(inference_buf_t *buf, enum xclBOSyncDirection dir)
{
    static int reported = 0;
    size_t     bytes = (size_t)buf->count * INFERENCE_BYTES_PER_ELEM;
    int        rc;
    if (bytes == 0u)
        return;
    rc = xclSyncBO(s_xrt_dev, (xclBufferHandle)buf->bo, dir, bytes, (size_t)buf->bo_offset);
    if (rc != 0 && !reported) {
        reported = 1;
        fprintf(stderr, "inference: xclSyncBO(%s, %zu bytes at offset %llu) failed: %d"
                " — CPU / kernel data may be incoherent\n",
                dir == XCL_BO_SYNC_BO_TO_DEVICE ? "to device" : "from device",
                bytes, (unsigned long long)buf->bo_offset, rc);
    }
}

inference_buf_t *inference_buf_alloc(unsigned n_elem)
{
    /* Round up to 64 bytes: the kernels read / write whole 16-byte words
     * (VectorOPKernel writes the last word of every run whole), so the
     * allocation must cover the tail word past n_elem. */
    size_t                 bytes = ((size_t)n_elem * INFERENCE_BYTES_PER_ELEM + 63u) & ~(size_t)63u;
    unsigned               flags = s_cacheable ? XCL_BO_FLAGS_CACHEABLE : 0u;
    xclBufferHandle        bo;
    void                  *virt;
    struct xclBOProperties props;
    inference_buf_t       *buf;

    buf = (inference_buf_t *)malloc(sizeof(inference_buf_t));
    if (!buf) return NULL;

    /* flags: memory bank 0 | XCL_BO_FLAGS_CACHEABLE (default) */
    bo = xclAllocBO(s_xrt_dev, bytes ? bytes : 64u, 0, flags);
    if (bo == (xclBufferHandle)NULLBO) {
        fprintf(stderr, "inference: xclAllocBO(%zu bytes) failed\n", bytes);
        free(buf);
        return NULL;
    }

    /* Map buffer into CPU virtual address space */
    virt = xclMapBO(s_xrt_dev, bo, /*write=*/1);
    if (!virt || virt == (void *)(uintptr_t)(-1)) {
        fprintf(stderr, "inference: xclMapBO failed\n");
        xclFreeBO(s_xrt_dev, bo);
        free(buf);
        return NULL;
    }

    /* Retrieve physical (device) address from BO properties */
    memset(&props, 0, sizeof(props));
    if (xclGetBOProperties(s_xrt_dev, bo, &props) != 0) {
        fprintf(stderr, "inference: xclGetBOProperties failed\n");
        xclUnmapBO(s_xrt_dev, bo, virt);
        xclFreeBO(s_xrt_dev, bo);
        free(buf);
        return NULL;
    }

    buf->virt      = virt;
    buf->phys      = props.paddr;
    buf->count     = n_elem;
    buf->bo        = (unsigned)bo;
    buf->refcount  = 1u;
    buf->is_owner  = 1u;
    buf->cached    = (uint8_t)(s_cacheable != 0);
    buf->bo_offset = 0;
    /* Start clean: no CPU-dirty line may exist in a range a kernel writes
     * before the CPU ever touched it (~15 ms per 216 MiB on the KV260). */
    if (buf->cached)
        _inference_buf_sync(buf, XCL_BO_SYNC_BO_TO_DEVICE);
    return buf;
}

static void _inference_buf_dealloc(inference_buf_t *buf)
{
    xclUnmapBO(s_xrt_dev, (xclBufferHandle)buf->bo, buf->virt);
    xclFreeBO(s_xrt_dev, (xclBufferHandle)buf->bo);
    free(buf);
}

/* Flush (clean): write dirty CPU cache lines of the range to DDR — before a
 * kernel reads the range, and before a kernel writes it (a dirty line
 * evicted later would overwrite the kernel's data). */
void inference_buf_sync_to_device(inference_buf_t *buf)
{
    _inference_buf_sync(buf, XCL_BO_SYNC_BO_TO_DEVICE);
}

/* Invalidate: drop the CPU cache lines of the range after a kernel wrote it
 * (and its lane drained), before the CPU reads it. */
void inference_buf_sync_from_device(inference_buf_t *buf)
{
    _inference_buf_sync(buf, XCL_BO_SYNC_BO_FROM_DEVICE);
}

#else  /* bare-metal */

////////////////////////////////////////////////////////////////////////
/* Bare-metal — heap allocation (virtual address == physical address)  */
/*                                                                     */
/* On Xilinx standalone/FreeRTOS BSP the MMU uses a flat (identity)   */
/* mapping for DDR, so virtual and physical addresses are equal.       */
////////////////////////////////////////////////////////////////////////

#include "xil_cache.h"
#include "xil_types.h"

int  inference_buf_pool_init(void)   { return 0; }
void inference_buf_pool_deinit(void) {}

inference_buf_t *inference_buf_alloc(unsigned n_elem)
{
    /* 64-byte multiple: covers the tail word the kernels access past n_elem. */
    size_t bytes = ((size_t)n_elem * INFERENCE_BYTES_PER_ELEM + 63u) & ~(size_t)63u;
    inference_buf_t *buf;
    void *mem;

    buf = (inference_buf_t *)malloc(sizeof(inference_buf_t));
    if (!buf) return NULL;

    mem = malloc(bytes);
    if (!mem) { free(buf); return NULL; }

    buf->virt     = mem;
    /* VA == PA on Xilinx standalone with identity-mapped DDR */
    buf->phys     = (uint64_t)(uintptr_t)mem;
    buf->count    = n_elem;
    buf->refcount = 1u;
    buf->is_owner = 1u;
    buf->cached   = 1u;          /* standalone DDR is cached (Xil_DCache*) */
    Xil_DCacheFlushRange((INTPTR)mem, (INTPTR)bytes);
    return buf;
}

static void _inference_buf_dealloc(inference_buf_t *buf)
{
    free(buf->virt);
    free(buf);
}

void inference_buf_sync_to_device(inference_buf_t *buf)
{
    Xil_DCacheFlushRange((INTPTR)buf->virt,
                         (INTPTR)(buf->count * INFERENCE_BYTES_PER_ELEM));
}

void inference_buf_sync_from_device(inference_buf_t *buf)
{
    Xil_DCacheInvalidateRange((INTPTR)buf->virt,
                              (INTPTR)(buf->count * INFERENCE_BYTES_PER_ELEM));
}

#endif /* __linux__ */
"""

# scripts/check_inference_setup.sh — preflight check for the inference library.
# {model_name} and {pool_bytes} are substituted by generate_setup_script().
_SETUP_SCRIPT_TEMPLATE = """\
#!/bin/sh
# check_inference_setup.sh — verify XRT prerequisites for the inference library
#
# Auto-generated by inference-scheduler
# Model   : {model_name}
# Required: {pool_bytes} bytes of DMA-capable memory
#
# The Linux inference library uses the XRT runtime (xclAllocBO / xclSyncBO)
# for DMA buffer allocation and cache coherency.
#
# Usage:
#   sh scripts/check_inference_setup.sh

POOL_BYTES={pool_bytes}
ERRORS=0

echo "Checking XRT inference library prerequisites..."
echo "  Required DMA pool : $POOL_BYTES bytes"

# -----------------------------------------------------------------------
# 1. Locate xrt.h
#    Try (in order):
#      a) pkg-config xrt               (preferred — works on Kria Ubuntu)
#      b) /usr/include/xrt/xrt.h       (Kria apt package layout)
#      c) $INFERENCE_XRT_DIR/include/xrt.h  (user override / cross-compile)
#      d) /opt/xilinx/xrt/include/xrt.h    (classic XRT install)
# -----------------------------------------------------------------------
XRT_HEADER=""

if pkg-config --exists xrt 2>/dev/null; then
    _inc=$(pkg-config --variable=includedir xrt 2>/dev/null)
    if [ -f "$_inc/xrt.h" ]; then
        XRT_HEADER="$_inc/xrt.h"
    elif [ -f "$_inc/xrt/xrt.h" ]; then
        XRT_HEADER="$_inc/xrt/xrt.h"
    fi
fi

if [ -z "$XRT_HEADER" ]; then
    for _candidate in \\
        /usr/include/xrt/xrt.h \\
        ${{INFERENCE_XRT_DIR:-/opt/xilinx/xrt}}/include/xrt.h \\
        /opt/xilinx/xrt/include/xrt.h
    do
        if [ -f "$_candidate" ]; then
            XRT_HEADER="$_candidate"
            break
        fi
    done
fi

if [ -n "$XRT_HEADER" ]; then
    echo "  xrt.h             : $XRT_HEADER  OK"
else
    echo "ERROR: xrt.h not found. Install XRT:" >&2
    echo "         sudo apt install xrt  (Kria/Ubuntu)" >&2
    echo "       or set INFERENCE_XRT_DIR to your XRT prefix." >&2
    ERRORS=$((ERRORS + 1))
fi

# -----------------------------------------------------------------------
# 2. Locate libxrt_core.so
# -----------------------------------------------------------------------
XRT_LIB=""

if pkg-config --exists xrt 2>/dev/null; then
    _libdir=$(pkg-config --variable=libdir xrt 2>/dev/null)
    [ -f "$_libdir/libxrt_core.so" ] && XRT_LIB="$_libdir/libxrt_core.so"
fi

if [ -z "$XRT_LIB" ]; then
    for _candidate in \\
        /usr/lib/libxrt_core.so \\
        /usr/lib/aarch64-linux-gnu/libxrt_core.so \\
        ${{INFERENCE_XRT_DIR:-/opt/xilinx/xrt}}/lib/libxrt_core.so \\
        /opt/xilinx/xrt/lib/libxrt_core.so
    do
        if [ -f "$_candidate" ]; then
            XRT_LIB="$_candidate"
            break
        fi
    done
fi

if [ -n "$XRT_LIB" ]; then
    echo "  libxrt_core.so    : $XRT_LIB  OK"
else
    echo "ERROR: libxrt_core.so not found." >&2
    echo "       Install XRT or set INFERENCE_XRT_DIR." >&2
    ERRORS=$((ERRORS + 1))
fi

# -----------------------------------------------------------------------
# 3. XRT device node
# -----------------------------------------------------------------------
if ls /dev/dri/renderD* >/dev/null 2>&1; then
    echo "  DRI render node   : $(ls /dev/dri/renderD* | head -1)  OK"
else
    echo "WARNING: no /dev/dri/renderD* device found — is the XRT driver loaded?" >&2
fi

# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------
if [ "$ERRORS" -eq 0 ]; then
    echo "XRT prerequisites OK"
else
    echo "$ERRORS prerequisite(s) missing — see errors above." >&2
    exit 1
fi
"""


class _BufImplMixin:
    """Generates src/inference_buf.c and scripts/check_inference_setup.sh."""

    def generate_buf_impl(self) -> str:
        """Content for src/inference_buf.c.

        Platform-independent public API (accessors) plus two platform
        implementations selected at compile time by __linux__:
          - Linux:      XRT buffer objects (xclAllocBO / xclMapBO), mapped
                        cacheable by default (INFERENCE_BUF_CACHEABLE), with
                        xclSyncBO range flush / invalidate.
          - Bare-metal: malloc (virtual address == physical address on Xilinx
                        standalone/FreeRTOS BSP), Xil_DCache* maintenance.
        """
        return (
            _file_banner("inference_buf.c", self._graph, self._model_path) +
            _BUF_IMPL_TEMPLATE
        )

    def generate_setup_script(self) -> str:
        """Content for scripts/check_inference_setup.sh."""
        pool_bytes = self._compute_pool_bytes()
        model_name = os.path.basename(self._model_path)
        return _SETUP_SCRIPT_TEMPLATE.format(
            model_name=model_name,
            pool_bytes=pool_bytes,
        )
