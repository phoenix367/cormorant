/* inference_buf.c — DMA-capable buffer implementation for kernel benchmarks */
#include "inference.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

Data_t *inference_buf_ptr(inference_buf_t *buf)       { return (Data_t *)buf->virt; }
uint64_t inference_buf_phys(const inference_buf_t *b) { return b->phys; }
unsigned inference_buf_count(const inference_buf_t *b){ return b->count; }

void inference_buf_init_view(inference_buf_t *view, inference_buf_t *base,
                              unsigned offset_elems, unsigned count_elems)
{
    uint64_t byte_off = (uint64_t)offset_elems * INFERENCE_BYTES_PER_ELEM;
    view->virt        = (char *)base->virt + (size_t)byte_off;
    view->phys        = base->phys + byte_off;
    view->count       = count_elems;
    view->is_owner    = 0u;
#ifdef __linux__
    view->bo          = base->bo;
    view->bo_offset   = base->bo_offset + byte_off;
#endif
}

int inference_buf_alloc_layout(const char *layout, unsigned n, const unsigned *counts,
                               inference_buf_t *bufs[], inference_buf_t **pool)
{
    unsigned i;
    *pool = NULL;
    if (!layout) {
        for (i = 0; i < n; i++) if (!(bufs[i] = inference_buf_alloc(counts[i]))) return -1;
        return 0;
    }
    unsigned long long off[16];
    const char *s = layout;
    size_t need = 0, e = INFERENCE_BYTES_PER_ELEM;
    for (i = 0; i < n && i < 16; i++) {
        char *end;
        off[i] = strtoull(s, &end, 0);
        if (end == s || (off[i] & 63ull) || (*end != ',' && *end != '\0')) return -1;
        if (off[i] + (size_t)counts[i] * e > need) need = off[i] + (size_t)counts[i] * e;
        s = (*end == ',') ? end + 1 : end;
    }
    if (i < n) return -1;
    *pool = inference_buf_alloc((unsigned)((need + e - 1) / e));
    if (!*pool) return -1;
    for (i = 0; i < n; i++) {
        bufs[i] = (inference_buf_t *)malloc(sizeof(inference_buf_t));
        if (!bufs[i]) return -1;
        inference_buf_init_view(bufs[i], *pool, (unsigned)(off[i] / e), counts[i]);
    }
    return 0;
}

void inference_buf_free_layout(unsigned n, inference_buf_t *bufs[], inference_buf_t *pool)
{
    unsigned i;
    if (pool) { for (i = 0; i < n; i++) free(bufs[i]); inference_buf_free(pool); }
    else      { for (i = 0; i < n; i++) inference_buf_free(bufs[i]); }
}

const char *inference_buf_phys_json(unsigned n, inference_buf_t *const bufs[])
{
    static char out[16 * 24 + 4];
    size_t len = 0; unsigned i;
    out[len++] = '[';
    for (i = 0; i < n && i < 16; i++)
        len += (size_t)snprintf(out + len, sizeof(out) - len, "%s\"0x%llx\"", i ? "," : "",
                                (unsigned long long)inference_buf_phys(bufs[i]));
    out[len++] = ']'; out[len] = '\0';
    return out;
}

/* Data_t is ap_fixed<16,8> bits: value = (int16_t)bits / 256.  Rounds half
 * to even and saturates (the scheduler's float_to_storage); NaN -> 0. */
static Data_t buf_from_float(float v)
{
    double    r = (double)v * 256.0;
    long long i;
    double    f;
    if (r != r) return (Data_t)0u;
    if (r > 32767.0) r = 32767.0;
    if (r < -32768.0) r = -32768.0;
    i = (long long)r;                                /* toward zero */
    f = r - (double)i;
    if (f > 0.5 || (f == 0.5 && (i & 1))) i++;
    else if (f < -0.5 || (f == -0.5 && (i & 1))) i--;
    return (Data_t)(int16_t)i;
}

void inference_buf_fill_float(inference_buf_t *buf, const float *src, unsigned n)
{
    Data_t  *dst = (Data_t *)buf->virt;
    unsigned i;
    for (i = 0; i < n; i++) dst[i] = buf_from_float(src[i]);
}

void inference_buf_read_float(const inference_buf_t *buf, float *dst, unsigned n)
{
    const Data_t *src = (const Data_t *)buf->virt;
    unsigned i;
    for (i = 0; i < n; i++) dst[i] = (float)((double)(int16_t)src[i] / 256.0);
}

#ifdef __linux__
#include <stdio.h>
#include <xrt.h>

static xclDeviceHandle s_xrt_dev = NULL;

int inference_buf_pool_init(void)
{
    s_xrt_dev = xclOpen(0, NULL, (enum xclVerbosityLevel)XCL_QUIET);
    if (!s_xrt_dev) {
        fprintf(stderr, "bench: xclOpen(0) failed — is XRT loaded?\n");
        return -1;
    }
    return 0;
}

void inference_buf_pool_deinit(void)
{
    if (s_xrt_dev) { xclClose(s_xrt_dev); s_xrt_dev = NULL; }
}

inference_buf_t *inference_buf_alloc(unsigned n_elem)
{
    size_t bytes = (size_t)n_elem * INFERENCE_BYTES_PER_ELEM;
    inference_buf_t *buf = (inference_buf_t *)malloc(sizeof(inference_buf_t));
    if (!buf) return NULL;

    xclBufferHandle bo = xclAllocBO(s_xrt_dev, bytes, 0, 0);
    if (bo == (xclBufferHandle)NULLBO) {
        fprintf(stderr, "bench: xclAllocBO(%zu B) failed\n", bytes);
        free(buf); return NULL;
    }
    void *virt = xclMapBO(s_xrt_dev, bo, 1);
    if (!virt || virt == (void *)(uintptr_t)(-1)) {
        fprintf(stderr, "bench: xclMapBO failed\n");
        xclFreeBO(s_xrt_dev, bo); free(buf); return NULL;
    }
    struct xclBOProperties props;
    memset(&props, 0, sizeof(props));
    if (xclGetBOProperties(s_xrt_dev, bo, &props) != 0) {
        fprintf(stderr, "bench: xclGetBOProperties failed\n");
        xclUnmapBO(s_xrt_dev, bo, virt);
        xclFreeBO(s_xrt_dev, bo); free(buf); return NULL;
    }
    buf->virt = virt; buf->phys = props.paddr;
    buf->count = n_elem; buf->bo = (unsigned)bo;
    buf->is_owner = 1u; buf->bo_offset = 0;
    return buf;
}

void inference_buf_free(inference_buf_t *buf)
{
    if (!buf || !buf->is_owner) return;
    xclUnmapBO(s_xrt_dev, (xclBufferHandle)buf->bo, buf->virt);
    xclFreeBO(s_xrt_dev,  (xclBufferHandle)buf->bo);
    free(buf);
}

void inference_buf_sync_to_device(inference_buf_t *buf)
{
    xclSyncBO(s_xrt_dev, (xclBufferHandle)buf->bo, XCL_BO_SYNC_BO_TO_DEVICE,
              buf->count * INFERENCE_BYTES_PER_ELEM, buf->bo_offset);
}

void inference_buf_sync_from_device(inference_buf_t *buf)
{
    xclSyncBO(s_xrt_dev, (xclBufferHandle)buf->bo, XCL_BO_SYNC_BO_FROM_DEVICE,
              buf->count * INFERENCE_BYTES_PER_ELEM, buf->bo_offset);
}

#else  /* bare-metal stub — not used in benchmark builds */
int  inference_buf_pool_init(void)   { return 0; }
void inference_buf_pool_deinit(void) {}
inference_buf_t *inference_buf_alloc(unsigned n) { (void)n; return NULL; }
void inference_buf_free(inference_buf_t *b)      { (void)b; }
void inference_buf_sync_to_device(inference_buf_t *b)   { (void)b; }
void inference_buf_sync_from_device(inference_buf_t *b) { (void)b; }
#endif /* __linux__ */
