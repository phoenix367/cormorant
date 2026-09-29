// SPDX-License-Identifier: GPL-2.0-only
/*
 * fsm_kernel_test — the PL reading and writing one /dev/fpga_smmu_mem buffer
 * larger than the whole CMA region (run as root, NS bitstream + SMMU stack).
 *
 *   ./fsm_kernel_test [MiB] [--cma]
 *
 * One buffer of MiB (default 1100) is split into three equal regions a | b |
 * c; VectorOPKernel computes c = a + b over all of them in one call (OP_ADD,
 * outer 1), so its reads and writes span the whole buffer — scattered
 * ordinary pages at one IOVA range.  Every element of c is checked, the
 * kernel call is timed (bytes read + written / time).  --cma runs the same
 * on three XRT (CMA) buffer objects instead: the baseline, on a bitstream
 * whose traffic bypasses the SMMU (or without the SMMU stack loaded).
 *
 * Build on the board, with the VectorOPKernel driver of a generated project:
 *   cc -O2 -Wall -I DRV -o test/fsm_kernel_test test/fsm_kernel_test.c \
 *      DRV/xvectoropkernel.c DRV/xvectoropkernel_linux.c
 *   add -DFSM_WITH_XRT $(pkg-config --cflags --libs xrt) for --cma
 */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include "../fpga_smmu_mem.h"
#include "xvectoropkernel.h"
#ifdef FSM_WITH_XRT
#include <xrt.h>
#endif

#define MiB      ((uint64_t)1024 * 1024)
#define OP_ADD   0u

static double now_s(void)
{
	struct timespec t;

	clock_gettime(CLOCK_MONOTONIC, &t);
	return t.tv_sec + t.tv_nsec * 1e-9;
}

static long meminfo_kib(const char *key)
{
	char line[128];
	long v = -1;
	FILE *f = fopen("/proc/meminfo", "r");

	if (!f)
		return -1;
	while (fgets(line, sizeof(line), f))
		if (!strncmp(line, key, strlen(key)) && line[strlen(key)] == ':')
			v = strtol(line + strlen(key) + 1, NULL, 10);
	fclose(f);
	return v;
}

/* Values that keep a + b inside int16: no saturation, c == a + b exactly. */
static inline int16_t val_a(uint64_t i) { return (int16_t)((i * 37u) & 0x1fff) - 0x1000; }
static inline int16_t val_b(uint64_t i) { return (int16_t)((i * 101u + 7u) & 0x1fff) - 0x1000; }

/* One region: CPU pointer, device address, and how to clean / invalidate it. */
struct region {
	int16_t  *p;
	uint64_t  dev;
	int       fd;                  /* SMMU: the buffer's file; -1 for XRT */
	uint64_t  off;                 /* SMMU: byte offset in that buffer */
#ifdef FSM_WITH_XRT
	xclDeviceHandle xrt;
	xclBufferHandle bo;
#endif
};

static int region_sync(const struct region *r, uint64_t bytes, int to_device)
{
	if (r->fd >= 0) {
		struct fsm_sync s = { .offset = r->off, .size = bytes,
				      .dir = to_device ? FSM_SYNC_TO_DEVICE : FSM_SYNC_FROM_DEVICE };

		return ioctl(r->fd, FSM_IOC_SYNC, &s) ? errno : 0;
	}
#ifdef FSM_WITH_XRT
	return xclSyncBO(r->xrt, r->bo, to_device ? XCL_BO_SYNC_BO_TO_DEVICE
						   : XCL_BO_SYNC_BO_FROM_DEVICE, bytes, 0);
#else
	return -1;
#endif
}

int main(int argc, char **argv)
{
	uint64_t mib = 1100, total, reg_bytes, n, i, bad = 0, first_bad = 0;
	long cma0 = meminfo_kib("CmaFree"), free0 = meminfo_kib("MemFree");
	struct region r[3];
	XVectoropkernel k;
	int cma = 0, a, pass;
	double t0, t_fill, t_run[2], t_check;

	for (a = 1; a < argc; a++) {
		if (!strcmp(argv[a], "--cma"))
			cma = 1;
		else
			mib = strtoull(argv[a], NULL, 0);
	}
	total = mib * MiB;
	reg_bytes = (total / 3) & ~(2 * MiB - 1);          /* 2 MiB-aligned regions */
	n = reg_bytes / sizeof(int16_t);
	memset(r, 0, sizeof(r));

	if (!cma) {
		struct fsm_alloc req = { .size = total };
		int fd = open(FSM_DEVICE_PATH, O_RDWR | O_CLOEXEC);
		uint8_t *base;

		if (fd < 0 || ioctl(fd, FSM_IOC_ALLOC, &req)) {
			perror("fpga_smmu_mem alloc");
			return 1;
		}
		base = mmap(NULL, total, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
		if (base == MAP_FAILED) {
			perror("mmap");
			return 1;
		}
		printf("SMMU buffer: %" PRIu64 " MiB at IOVA 0x%09" PRIx64 "..0x%09" PRIx64 "\n",
		       mib, (uint64_t)req.iova, (uint64_t)req.iova + total);
		for (a = 0; a < 3; a++) {
			r[a].fd = fd;
			r[a].off = a * reg_bytes;
			r[a].p = (int16_t *)(base + r[a].off);
			r[a].dev = (uint64_t)req.iova + r[a].off;
		}
	} else {
#ifdef FSM_WITH_XRT
		xclDeviceHandle dev = xclOpen(0, NULL, XCL_QUIET);

		if (!dev) {
			fprintf(stderr, "xclOpen failed\n");
			return 1;
		}
		printf("XRT (CMA) buffers: 3 x %" PRIu64 " MiB\n", reg_bytes / MiB);
		for (a = 0; a < 3; a++) {
			struct xclBOProperties props;

			r[a].fd = -1;
			r[a].xrt = dev;
			r[a].bo = xclAllocBO(dev, reg_bytes, 0, 1u << 24 /* XCL_BO_FLAGS_CACHEABLE */);
			if (r[a].bo == (xclBufferHandle)NULLBO) {
				fprintf(stderr, "xclAllocBO(%" PRIu64 " MiB) failed\n", reg_bytes / MiB);
				return 1;
			}
			r[a].p = xclMapBO(dev, r[a].bo, 1);
			memset(&props, 0, sizeof(props));
			if (!r[a].p || xclGetBOProperties(dev, r[a].bo, &props)) {
				fprintf(stderr, "xclMapBO / xclGetBOProperties failed\n");
				return 1;
			}
			r[a].dev = props.paddr;
		}
#else
		fprintf(stderr, "--cma needs a build with -DFSM_WITH_XRT\n");
		return 2;
#endif
	}
	for (a = 0; a < 3; a++)
		printf("  %c: %" PRIu64 " MiB at 0x%09" PRIx64 "\n", "abc"[a], reg_bytes / MiB, r[a].dev);

	if (XVectoropkernel_Initialize(&k, "fabric_vecop") != XST_SUCCESS) {
		fprintf(stderr, "XVectoropkernel_Initialize(fabric_vecop) failed\n");
		return 1;
	}

	t0 = now_s();
	for (i = 0; i < n; i++) {
		r[0].p[i] = val_a(i);
		r[1].p[i] = val_b(i);
		r[2].p[i] = 0x5a5a;
	}
	for (a = 0; a < 3; a++)
		if (region_sync(&r[a], reg_bytes, 1)) {
			fprintf(stderr, "clean of region %c failed\n", "abc"[a]);
			return 1;
		}
	t_fill = now_s() - t0;

	for (pass = 0; pass < 2; pass++) {
		XVectoropkernel_Set_a(&k, r[0].dev);
		XVectoropkernel_Set_b(&k, r[1].dev);
		XVectoropkernel_Set_c(&k, r[2].dev);
		XVectoropkernel_Set_size(&k, (u32)n);
		XVectoropkernel_Set_op(&k, OP_ADD);
		XVectoropkernel_Set_outer(&k, 1);
		XVectoropkernel_Set_a_inc(&k, 0);
		XVectoropkernel_Set_b_inc(&k, 0);
		XVectoropkernel_Set_act(&k, 0);
		t0 = now_s();
		XVectoropkernel_Start(&k);
		while (!XVectoropkernel_IsDone(&k)) {
			if (now_s() - t0 > 30) {
				fprintf(stderr, "kernel did not finish in 30 s\n");
				return 1;
			}
		}
		t_run[pass] = now_s() - t0;
	}

	t0 = now_s();
	if (region_sync(&r[2], reg_bytes, 0)) {
		fprintf(stderr, "invalidate of region c failed\n");
		return 1;
	}
	for (i = 0; i < n; i++) {
		int16_t want = (int16_t)(val_a(i) + val_b(i));

		if (r[2].p[i] != want) {
			if (!bad++)
				first_bad = i;
		}
	}
	t_check = now_s() - t0;

	printf("  %" PRIu64 " elements (c = a + b), fill + clean %.2f s, check %.2f s\n",
	       n, t_fill, t_check);
	printf("  kernel: %.1f ms / %.1f ms (2 calls), %.2f GB/s moved (2 reads + 1 write)\n",
	       t_run[0] * 1e3, t_run[1] * 1e3, 3.0 * reg_bytes / t_run[1] / 1e9);
	printf("  MemFree %+ld MiB, CmaFree %+ld MiB\n",
	       (meminfo_kib("MemFree") - free0) / 1024, (meminfo_kib("CmaFree") - cma0) / 1024);
	if (bad) {
		printf("FAIL: %" PRIu64 " of %" PRIu64 " elements wrong, first at %" PRIu64
		       " (got %d, want %d)\n", bad, n, first_bad, r[2].p[first_bad],
		       (int16_t)(val_a(first_bad) + val_b(first_bad)));
		return 1;
	}
	printf("PASS\n");
	XVectoropkernel_Release(&k);
	return 0;
}
