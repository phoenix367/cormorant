# arm-smmu kernel module for the KV260 (experimental)

The ZynqMP has an ARM MMU-500 SMMU (`iommu@fd800000` in the board's device
tree, `status = "disabled"`), but the Ubuntu `xilinx-zynqmp` 5.15 kernel is
built without `CONFIG_ARM_SMMU`.  This directory builds that driver as an
out-of-tree module, so the FPGA's AXI masters can be put behind the SMMU:
zocl (XRT 2.13, `zocl_iommu_map_bo`) then maps ordinary, scattered pages to
one contiguous device address instead of needing CMA memory.

**License.** `arm-smmu.c`, `arm-smmu.h` and `arm-smmu-impl.c` are copied
unchanged from Linux v5.15.139 (`drivers/iommu/arm/arm-smmu/`);
`arm-smmu-stubs.c` stubs the NVIDIA / Qualcomm hooks the MMU-500 never takes.
All files in this directory are **GPL-2.0-only** (see their SPDX headers),
not covered by the repository's Apache 2.0 license.

## Build (on the board)

Needs the running kernel's headers (`/lib/modules/$(uname -r)/build`), gcc
and make — all present on the Kria Ubuntu 22.04 image.

```bash
make                     # -> arm_smmu.ko (vermagic 5.15.0-1079-xilinx-zynqmp)
```

The kernel does not enforce module signatures (`sig_enforce=N`), so the
unsigned module loads; the kernel is then tainted `E`.

## Facts for using it

- Stream ID = TBU number `[14:10]` + master ID `[9:0]` (UG1085 §16, Tables
  16-3 and 16-11).  `S_AXI_HPC0_FPD` and `S_AXI_HPC1_FPD` are on TBU0 with
  master IDs `1000` / `1001` + AXI ID `[5:0]`: stream IDs `0x200`–`0x23F` and
  `0x240`–`0x27F`.  One SMR entry covers both: `iommus = <&smmu 0x200>` on the
  zocl node with `stream-match-mask = <0x7f>` on the SMMU node (the MMU-500
  has 48 SMRs, fewer than the 128 IDs).
- Load with `disable_bypass=0` (the default here) so devices without an
  `iommus` entry — Ethernet, SD, USB — keep bypassing the SMMU.

## Status (2026-09-27)

Works on the board: `insmod arm_smmu.ko disable_bypass=0`, then the
`smmu-enable.dtbo` overlay (`make overlays` builds it and `cormorant-smmu.dtbo`
on the host).  The driver probes an SMMUv2 (48 stream-match groups, 16 context
banks, 4K/64K/2M/32M/512M/1G pages); with `cormorant-smmu.dtbo` loaded through
`upload_bitstream.py --dtbo ... --overlay-name pl`, zocl joins IOMMU group 0
(type DMA).  Load order matters: a `pynq` overlay stacked on `pl` (PYNQ's
login script inserts one) makes `pl` unremovable; reboot for a clean stack.

Blocked in zocl, not the SMMU.  With an IOMMU domain, zocl 2.13 ("SVM" mode)
allocates shmem pages and maps them into the SMMU at the process's virtual
address when the buffer is `mmap`ed, but:

1. `xclGetBOProperties` oopses: `zocl_bo_describe()` has no SVM case and
   reads the NULL `mm_node` (fix: report `bo->uaddr`);
2. `xclSyncBO` treats every non-coherent buffer as CMA and flushes
   `cma_obj->paddr`, a garbage address for SVM buffers;
3. SVM buffers are mapped to user space write-combined (uncached).

Ways on: patch and rebuild zocl from the Ubuntu kernel source (1–3), or keep
zocl on CMA and allocate SMMU-mapped memory through a separate driver (e.g.
u-dma-buf with the PL stream IDs) behind `inference_buf.c`.

The second way is `../fpga-smmu-mem/` (u-dma-buf's cache sync assumes
physically contiguous memory).  It works as an allocator, but the PL traffic
of the current bitstream is not translated: the HLS masters issue Secure
accesses (`AxPROT = 000`), which bypass the non-secure stream-match table —
see that README before loading anything with the PL running.
