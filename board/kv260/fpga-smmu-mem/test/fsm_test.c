// SPDX-License-Identifier: GPL-2.0-only
/*
 * fsm_test — checks /dev/fpga_smmu_mem without the PL (run as root: pagemap).
 *
 *   ./fsm_test [BIG_MiB]
 *
 * Buffers come out zeroed at one page-aligned IOVA in the PL window
 * [FSM_IOVA_BASE, FSM_IOVA_END) (the kernels' DDR_HIGH segment), backed by
 * scattered ordinary pages (not CMA: CmaFree stays put), readable and
 * writable through the cacheable and the write-combined mapping; the ioctls
 * reject bad arguments; buffers do not overlap in IOVA space and their memory
 * returns on close.  BIG_MiB (default 1100) is one buffer larger than the
 * 1000 MiB CMA region — 0 skips it.
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
#include <unistd.h>

#include "../fpga_smmu_mem.h"

#define MiB ((uint64_t)1024 * 1024)

static int failures;

#define CHECK(cond, ...) do {                                             \
	if (!(cond)) {                                                    \
		failures++;                                               \
		printf("  FAIL %s:%d: ", __FILE__, __LINE__);             \
		printf(__VA_ARGS__);                                      \
		printf("\n");                                             \
	}                                                                 \
} while (0)

static long meminfo_kib(const char *key)
{
	FILE *f = fopen("/proc/meminfo", "r");
	char line[256];
	long v = -1;
	size_t n = strlen(key);

	if (!f)
		return -1;
	while (fgets(line, sizeof(line), f))
		if (!strncmp(line, key, n) && line[n] == ':') {
			v = strtol(line + n + 1, NULL, 10);
			break;
		}
	fclose(f);
	return v;
}

struct buf {
	int fd;
	uint64_t size, iova;
	uint8_t *p;
};

static int buf_alloc(struct buf *b, uint64_t size, uint32_t flags)
{
	struct fsm_alloc a = { .size = size, .flags = flags };

	b->fd = open(FSM_DEVICE_PATH, O_RDWR | O_CLOEXEC);
	if (b->fd < 0) {
		perror("open " FSM_DEVICE_PATH);
		return -1;
	}
	if (ioctl(b->fd, FSM_IOC_ALLOC, &a)) {
		printf("  FSM_IOC_ALLOC(%" PRIu64 " bytes): %s\n", size, strerror(errno));
		close(b->fd);
		return -1;
	}
	b->size = size;
	b->iova = a.iova;
	b->p = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, b->fd, 0);
	if (b->p == MAP_FAILED) {
		printf("  mmap: %s\n", strerror(errno));
		close(b->fd);
		return -1;
	}
	return 0;
}

static void buf_free(struct buf *b)
{
	munmap(b->p, b->size);
	close(b->fd);
}

static int sync_range(int fd, uint64_t off, uint64_t size, uint32_t dir)
{
	struct fsm_sync s = { .offset = off, .size = size, .dir = dir };

	return ioctl(fd, FSM_IOC_SYNC, &s) ? errno : 0;
}

/* Physical layout through /proc/self/pagemap: the number of physically
 * contiguous runs and the longest one (pages). */
static int phys_runs(const struct buf *b, unsigned long *runs, unsigned long *longest)
{
	long pg = sysconf(_SC_PAGESIZE);
	unsigned long n = b->size / pg, i, len = 0;
	uint64_t prev = 0;
	int fd = open("/proc/self/pagemap", O_RDONLY);

	*runs = *longest = 0;
	if (fd < 0)
		return -1;
	for (i = 0; i < n; i++) {
		uint64_t e, pfn;

		if (pread(fd, &e, sizeof(e), ((uintptr_t)b->p / pg + i) * sizeof(e)) != sizeof(e)) {
			close(fd);
			return -1;
		}
		if (!(e >> 63) || !(pfn = e & ((1ull << 55) - 1))) {
			close(fd);
			return -1;              /* not present, or PFNs hidden (not root) */
		}
		if (i && pfn == prev + 1) {
			len++;
		} else {
			(*runs)++;
			len = 1;
		}
		if (len > *longest)
			*longest = len;
		prev = pfn;
	}
	close(fd);
	return 0;
}

static void check_buffer(const char *what, uint64_t size, uint32_t flags)
{
	long cma0 = meminfo_kib("CmaFree"), free0 = meminfo_kib("MemFree");
	unsigned long runs, longest;
	struct buf b;
	uint64_t i, bad = 0;

	printf("%s: %" PRIu64 " MiB%s\n", what, size / MiB, flags & FSM_ALLOC_WC ? ", write-combined" : "");
	if (buf_alloc(&b, size, flags)) {
		failures++;
		return;
	}
	printf("  iova 0x%09" PRIx64 "..0x%09" PRIx64 "\n", b.iova, b.iova + size);
	CHECK(b.iova && !(b.iova & 0xfff), "iova 0x%" PRIx64 " not page aligned", b.iova);
	CHECK(b.iova >= FSM_IOVA_BASE && b.iova + size <= FSM_IOVA_END,
	      "iova range outside the PL window 0x%llx-0x%llx", FSM_IOVA_BASE, FSM_IOVA_END);

	for (i = 0; i < size; i += 64)
		bad |= *(volatile uint64_t *)(b.p + i);
	CHECK(!bad, "buffer not zeroed");

	for (i = 0; i < size / 8; i++)
		((uint64_t *)b.p)[i] = i * 0x9e3779b97f4a7c15ull;
	CHECK(!sync_range(b.fd, 0, size, FSM_SYNC_TO_DEVICE), "full clean");
	CHECK(!sync_range(b.fd, 0, size, FSM_SYNC_FROM_DEVICE), "full invalidate");
	for (i = 0, bad = 0; i < size / 8; i++)
		bad += ((uint64_t *)b.p)[i] != i * 0x9e3779b97f4a7c15ull;
	CHECK(!bad, "%" PRIu64 " words differ after clean + invalidate", bad);

	if (phys_runs(&b, &runs, &longest) == 0) {
		printf("  %lu physically contiguous runs, longest %lu KiB\n", runs, longest * 4);
		CHECK(size <= 2 * MiB || runs > 1 || longest * 4096ull < size,
		      "one physical run — expected ordinary scattered pages");
	} else {
		printf("  pagemap unavailable (not root?) — layout not checked\n");
	}
	{
		long cma1 = meminfo_kib("CmaFree"), free1 = meminfo_kib("MemFree");
		long used = free0 - free1, cma_used = cma0 - cma1;

		printf("  MemFree -%ld MiB, CmaFree -%ld MiB\n", used / 1024, cma_used / 1024);
		CHECK(cma_used * 1024L < (long)(size / 8), "CmaFree dropped by %ld KiB", cma_used);
	}
	buf_free(&b);
}

static void check_errors(void)
{
	struct fsm_alloc a = { .size = MiB };
	struct buf b;
	void *p;
	int fd, e;

	printf("argument checks\n");
	fd = open(FSM_DEVICE_PATH, O_RDWR | O_CLOEXEC);
	if (fd < 0) {
		perror("open");
		failures++;
		return;
	}
	p = mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
	CHECK(p == MAP_FAILED && errno == EINVAL, "mmap before alloc");
	CHECK(sync_range(fd, 0, 64, FSM_SYNC_TO_DEVICE) == EINVAL, "sync before alloc");
	a.size = 0;
	CHECK(ioctl(fd, FSM_IOC_ALLOC, &a) && errno == EINVAL, "size 0");
	a.size = 4ull << 30;
	CHECK(ioctl(fd, FSM_IOC_ALLOC, &a) && errno == EINVAL, "size 4 GiB");
	a.size = MiB;
	a.flags = 0x80;
	CHECK(ioctl(fd, FSM_IOC_ALLOC, &a) && errno == EINVAL, "unknown flag");
	a.flags = 0;
	CHECK(!ioctl(fd, FSM_IOC_ALLOC, &a), "alloc 1 MiB: %s", strerror(errno));
	CHECK(ioctl(fd, FSM_IOC_ALLOC, &a) && errno == EBUSY, "second alloc on one file");
	CHECK(ioctl(fd, _IO('F', 9)) && errno == ENOTTY, "unknown ioctl");
	CHECK(sync_range(fd, 0, MiB, 2) == EINVAL, "bad direction");
	CHECK(sync_range(fd, MiB, 1, FSM_SYNC_TO_DEVICE) == EINVAL, "range past the end");
	CHECK(sync_range(fd, 64, MiB, FSM_SYNC_TO_DEVICE) == EINVAL, "range past the end");
	CHECK(sync_range(fd, ~0ull, 2, FSM_SYNC_TO_DEVICE) == EINVAL, "offset overflow");
	CHECK(sync_range(fd, MiB, 0, FSM_SYNC_TO_DEVICE) == 0, "empty range at the end");
	CHECK(sync_range(fd, 4000, 200000, FSM_SYNC_FROM_DEVICE) == 0, "partial range");
	p = mmap(NULL, MiB, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
	CHECK(p == MAP_FAILED && errno == EINVAL, "MAP_PRIVATE");
	p = mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 4096);
	CHECK(p == MAP_FAILED && errno == EINVAL, "mmap at an offset");
	p = mmap(NULL, 2 * MiB, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
	CHECK(p == MAP_FAILED && errno == EINVAL, "mmap longer than the buffer");
	p = mmap(NULL, 4096, PROT_READ, MAP_SHARED, fd, 0);  /* a prefix is fine */
	CHECK(p != MAP_FAILED, "mmap of a prefix: %s", strerror(errno));
	if (p != MAP_FAILED) {
		e = mremap(p, 4096, 8192, MREMAP_MAYMOVE) == MAP_FAILED;
		CHECK(e, "mremap grew a VM_DONTEXPAND mapping");
		munmap(p, 4096);
	}
	close(fd);

	/* Several buffers at once: disjoint IOVA ranges. */
	{
		struct buf v[8];
		int i, j, n = 0;

		for (i = 0; i < 8; i++)
			if (!buf_alloc(&v[n], (uint64_t)(i + 1) * 3 * MiB + 4096 * i, 0))
				n++;
		CHECK(n == 8, "only %d of 8 buffers", n);
		for (i = 0; i < n; i++)
			for (j = i + 1; j < n; j++)
				CHECK(v[i].iova + v[i].size <= v[j].iova || v[j].iova + v[j].size <= v[i].iova,
				      "buffers %d and %d overlap in IOVA space", i, j);
		for (i = 0; i < n; i++)
			buf_free(&v[i]);
	}
	/* The mapping outlives close(): the pages stay until munmap. */
	if (!buf_alloc(&b, MiB, 0)) {
		close(b.fd);
		b.p[MiB - 1] = 0x5a;
		CHECK(b.p[MiB - 1] == 0x5a, "mapping after close");
		munmap(b.p, MiB);
	}
}

int main(int argc, char **argv)
{
	uint64_t big = (argc > 1 ? strtoull(argv[1], NULL, 0) : 1100) * MiB;
	long free0, free1;

	setvbuf(stdout, NULL, _IOLBF, 0);
	free0 = meminfo_kib("MemFree");
	check_errors();
	check_buffer("cacheable buffer", 64 * MiB, 0);
	check_buffer("write-combined buffer", 16 * MiB, FSM_ALLOC_WC);
	check_buffer("odd size", 5 * MiB + 123, 0);
	if (big)
		check_buffer("buffer larger than the CMA region", big, 0);
	free1 = meminfo_kib("MemFree");
	printf("MemFree after all buffers closed: %+ld MiB\n", (free1 - free0) / 1024);
	CHECK(free0 - free1 < 32 * 1024, "memory not returned: %ld KiB", free0 - free1);
	printf(failures ? "FAILED: %d check(s)\n" : "PASS\n", failures);
	return failures != 0;
}
