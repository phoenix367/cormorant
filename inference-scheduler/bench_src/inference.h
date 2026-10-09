/* inference.h — minimal DMA buffer API for kernel benchmarks */
#pragma once
#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

typedef uint16_t Data_t;
#define INFERENCE_BYTES_PER_ELEM  2u

struct inference_buf {
    void     *virt;
    uint64_t  phys;
    unsigned  count;
    uint8_t   is_owner;
#ifdef __linux__
    unsigned  bo;
    uint64_t  bo_offset;
#endif
};
typedef struct inference_buf inference_buf_t;

int      inference_buf_pool_init(void);
void     inference_buf_pool_deinit(void);
inference_buf_t *inference_buf_alloc(unsigned n_elem);
void     inference_buf_free(inference_buf_t *buf);
Data_t  *inference_buf_ptr(inference_buf_t *buf);
uint64_t inference_buf_phys(const inference_buf_t *buf);
unsigned inference_buf_count(const inference_buf_t *buf);
void     inference_buf_init_view(inference_buf_t *view, inference_buf_t *base,
                                  unsigned offset_elems, unsigned count_elems);
void     inference_buf_sync_to_device(inference_buf_t *buf);
void     inference_buf_sync_from_device(inference_buf_t *buf);
/* The benchmarks' buffers at chosen addresses (DDR bank / page experiments,
 * the parent repo's doc/plans/PS_PORTS_PLAN.md §11): layout "OFF0,OFF1,..."
 * gives the byte offset of each of the n buffers inside ONE allocation
 * (multiples of 64); NULL gives n separate allocations.  bufs[i] are views
 * or owners, *pool the one allocation or NULL.  0 on success. */
int      inference_buf_alloc_layout(const char *layout, unsigned n, const unsigned *counts,
                                    inference_buf_t *bufs[], inference_buf_t **pool);
void     inference_buf_free_layout(unsigned n, inference_buf_t *bufs[], inference_buf_t *pool);
/* "[\"0x..\",...]": the buffers' physical addresses as a JSON array (static buffer) */
const char *inference_buf_phys_json(unsigned n, inference_buf_t *const bufs[]);
void     inference_buf_fill_float(inference_buf_t *buf,
                                   const float *src, unsigned n);
void     inference_buf_read_float(const inference_buf_t *buf,
                                   float *dst, unsigned n);

#ifdef __cplusplus
}
#endif
