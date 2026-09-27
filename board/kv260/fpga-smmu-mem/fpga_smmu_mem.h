/* SPDX-License-Identifier: GPL-2.0-only WITH Linux-syscall-note */
/*
 * fpga_smmu_mem — user-space interface.
 *
 * One buffer per open file of /dev/fpga_smmu_mem:
 *   FSM_IOC_ALLOC  allocate `size` bytes (zeroed); `iova` returns the device
 *                  address the PL kernels use — one contiguous range behind
 *                  the SMMU, whatever the physical layout
 *   mmap(fd, 0)    the CPU mapping: cacheable, or write-combined with
 *                  FSM_ALLOC_WC
 *   FSM_IOC_SYNC   cache maintenance on a byte range: FSM_SYNC_TO_DEVICE
 *                  cleans (CPU writes reach DDR), FSM_SYNC_FROM_DEVICE
 *                  invalidates (drop stale lines after a kernel wrote)
 *   close(fd)      unmaps and frees
 *
 * The generated inference_buf.c (inference-scheduler/src/codegen/_buf_impl.py)
 * carries a copy of these definitions; test_buf_impl.py checks they match.
 */
#ifndef FPGA_SMMU_MEM_H
#define FPGA_SMMU_MEM_H

#include <linux/ioctl.h>
#include <linux/types.h>

#define FSM_DEVICE_PATH       "/dev/fpga_smmu_mem"

#define FSM_ALLOC_WC          0x1u     /* write-combined CPU mapping */

struct fsm_alloc {
	__u64 size;                    /* in: bytes */
	__u32 flags;                   /* in: FSM_ALLOC_* */
	__u32 pad;
	__u64 iova;                    /* out: device address */
};

#define FSM_SYNC_TO_DEVICE    0u
#define FSM_SYNC_FROM_DEVICE  1u

struct fsm_sync {
	__u64 offset;                  /* bytes from the buffer start */
	__u64 size;                    /* bytes */
	__u32 dir;                     /* FSM_SYNC_* */
	__u32 pad;
};

#define FSM_IOC_MAGIC         'F'
#define FSM_IOC_ALLOC         _IOWR(FSM_IOC_MAGIC, 1, struct fsm_alloc)
#define FSM_IOC_SYNC          _IOW(FSM_IOC_MAGIC, 2, struct fsm_sync)

#endif /* FPGA_SMMU_MEM_H */
