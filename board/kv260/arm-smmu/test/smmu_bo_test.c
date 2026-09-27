// SPDX-License-Identifier: GPL-2.0-only
/*
 * With zocl behind the SMMU: where does an XRT buffer's memory come from,
 * and does zocl accept ordinary (malloc) memory?
 *   1. xclAllocBO (cacheable, as the generated code uses): CmaFree before /
 *      after, device address (paddr), physical contiguity of its pages;
 *   2. xclAllocUserPtrBO on malloc memory: accepted?  device address.
 * build: gcc -O2 -o smmu_bo_test smmu_bo_test.c $(pkg-config --cflags --libs xrt)
 */
#define _GNU_SOURCE
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <xrt/xrt.h>

#ifndef XCL_BO_FLAGS_CACHEABLE
#define XCL_BO_FLAGS_CACHEABLE (1U << 24)
#endif

static long cma_free_kb(void)
{
	char line[256];
	long v = -1;
	FILE *f = fopen("/proc/meminfo", "r");

	while (f && fgets(line, sizeof(line), f))
		if (sscanf(line, "CmaFree: %ld kB", &v) == 1)
			break;
	if (f)
		fclose(f);
	return v;
}

/* contiguous runs of the physical pages behind [p, p + n) */
static void phys_runs(int pm, void *p, size_t n, const char *what)
{
	size_t pages = n / 4096, i, runs = 1;
	uint64_t prev = 0, first = 0;

	for (i = 0; i < pages; i++) {
		uint64_t e = 0, f;

		if (pread(pm, &e, 8, (off_t)(((uintptr_t)p + i * 4096) / 4096) * 8) != 8)
			return;
		f = e & ((1ULL << 55) - 1);
		if (i == 0)
			first = f;
		else if (f != prev + 1)
			runs++;
		prev = f;
	}
	printf("  %s: first page phys 0x%llx, %zu pages in %zu physically contiguous runs\n",
	       what, (unsigned long long)(first * 4096), pages, runs);
}

int main(void)
{
	const size_t n = 64u << 20;
	xclDeviceHandle dev = xclOpen(0, NULL, XCL_QUIET);
	int pm = open("/proc/self/pagemap", O_RDONLY);
	struct xclBOProperties pr;
	xclBufferHandle bo;
	long cma0, cma1;
	void *p;

	if (!dev || pm < 0) {
		fprintf(stderr, "open failed\n");
		return 1;
	}

	/* 1. an ordinary XRT buffer */
	cma0 = cma_free_kb();
	bo = xclAllocBO(dev, n, 0, XCL_BO_FLAGS_CACHEABLE);
	if (bo == NULLBO) {
		printf("xclAllocBO 64 MiB: FAILED\n");
		return 1;
	}
	p = xclMapBO(dev, bo, true);
	memset(p, 1, n);
	cma1 = cma_free_kb();
	memset(&pr, 0, sizeof(pr));
	xclGetBOProperties(dev, bo, &pr);
	printf("xclAllocBO 64 MiB (cacheable): ok, device address 0x%llx, CmaFree %ld -> %ld kB (%s)\n",
	       (unsigned long long)pr.paddr, cma0, cma1,
	       cma0 - cma1 > (long)(n >> 11) ? "from CMA" : "not from CMA");
	phys_runs(pm, p, n, "its pages");
	munmap(p, n);
	xclFreeBO(dev, bo);

	/* 2. ordinary malloc memory */
	if (posix_memalign(&p, 4096, n))
		return 1;
	memset(p, 2, n);
	phys_runs(pm, p, n, "malloc 64 MiB");
	bo = xclAllocUserPtrBO(dev, p, n, 0);
	if (bo == NULLBO) {
		printf("xclAllocUserPtrBO on malloc memory: FAILED\n");
	} else {
		memset(&pr, 0, sizeof(pr));
		xclGetBOProperties(dev, bo, &pr);
		printf("xclAllocUserPtrBO on malloc memory: ok, device address 0x%llx\n",
		       (unsigned long long)pr.paddr);
		xclFreeBO(dev, bo);
	}
	free(p);
	close(pm);
	xclClose(dev);
	return 0;
}
