# fpga_smmu_mem — SMMU-mapped DMA buffers for the PL kernels (experimental)

A platform driver that gives the KV260 kernels large DMA buffers without CMA:
ordinary pages (allocated in chunks of up to 2 MiB) mapped by the ZynqMP SMMU
at one contiguous IO virtual address.  One buffer per open file of
`/dev/fpga_smmu_mem`:

| Call | Effect |
|------|--------|
| `FSM_IOC_ALLOC` | allocate `size` bytes, zeroed; returns the device address (`iova`, in `[FSM_IOVA_BASE, FSM_IOVA_END)` = 32–64 GiB) |
| `mmap(fd, 0)` | CPU mapping: cacheable, or write-combined with `FSM_ALLOC_WC` |
| `FSM_IOC_SYNC` | clean (to device) / invalidate (from device) a byte range, split at the physically contiguous chunks |
| `close` / `munmap` | the last one frees the buffer |

The user interface is `fpga_smmu_mem.h`.  The generated `inference_buf.c`
(`inference-scheduler/src/codegen/_buf_impl.py`) carries a copy of it
(`test_buf_impl.py` checks they match) and uses the driver instead of XRT
buffer objects when `/dev/fpga_smmu_mem` exists; `INFERENCE_BUF_BACKEND=xrt|smmu`
overrides the choice.

## Requirements

- **A non-secure bitstream.**  HLS `m_axi` ports drive `AxPROT = 000`
  (Secure) by default, and the SMMU does not match Secure transactions
  against the non-secure stream table Linux programs: the PL traffic then
  bypasses translation.  `set_prot_ns.tcl` sets `C_M_AXI_GMEMn_PROT_VALUE =
  "010"` on all twelve kernel masters of `hw/cormorant_hw_128`; build the
  bitstream as usual.  Nothing else changes (timing, resources, results — the
  CMA mode runs as before).
- **IOVAs inside the kernels' address map.**  Vivado's address editor gives
  every kernel master `DDR_LOW` (0–2 GiB), `QSPI` (3–3.5 GiB) and `DDR_HIGH`
  (32–64 GiB) on HPC0 / HPC1; the interconnect answers any other address
  with DECERR before the SMMU sees it (the kernel reads zeros, no fault).  The
  driver sets a 36-bit DMA mask, so iommu-dma allocates top-down from 64 GiB,
  and checks every buffer lies in `DDR_HIGH` (`FSM_IOVA_BASE` /
  `FSM_IOVA_END`); the generated `inference_buf.c` checks it too.

## Build and load

```bash
# on the board (needs ../arm-smmu/arm_smmu.ko, see its README), after
# upload_bitstream.py has loaded a non-secure bitstream
make                                   # fpga_smmu_mem.ko
insmod ../arm-smmu/arm_smmu.ko disable_bypass=0
insmod fpga_smmu_mem.ko
# on the host: make overlay DTC=/mnt/data/xilinx/2025.2/Vitis/bin/dtc
mkdir /sys/kernel/config/device-tree/overlays/smmu
cat smmu-mem.dtbo > /sys/kernel/config/device-tree/overlays/smmu/dtbo
cc -O2 -o test/fsm_test test/fsm_test.c && test/fsm_test 1100
# the PL on one buffer larger than CMA (DRV = a generated project's driver/)
cc -O2 -I DRV -o test/fsm_kernel_test test/fsm_kernel_test.c \
   DRV/xvectoropkernel.c DRV/xvectoropkernel_linux.c && test/fsm_kernel_test 1100
```

`smmu-mem.dtbo` enables the SMMU (`stream-match-mask = <0x7f>`) and adds the
`fpga_smmu_mem` node with `iommus = <&smmu 0x200>`: stream IDs
`0x200`–`0x27F`, `S_AXI_HPC0_FPD` and `S_AXI_HPC1_FPD` (TBU0, UG1085 §16) —
the only PS slave ports the block design enables, so every kernel master is
behind it.  The probe refuses to bind without an IOMMU domain.

Replacing the module by a build with another IOVA window needs a reboot:
iommu-dma keeps freed IOVAs in per-CPU caches of the SMMU domain, which
outlives the module, and hands them out again (the window check then
rejects the allocation).

## Status (2026-09-29)

Works on the board with the non-secure bitstream `98de80db9cc8`
(`hw/cormorant_hw_128` at `bbfacf6` + `set_prot_ns.tcl`; WNS +0.767 ns, LUT 93 429,
BRAM 115.5, URAM 56, DSP 1 058 — as the production build):

- **Without the SMMU stack** it is the production bitstream: MNIST 98.92 % /
  0.268 ms, LeNet 97.35 % / 2.810 ms, the probe models pass.
- **The SMMU translates the PL traffic.**  A run on CMA physical addresses
  (`INFERENCE_BUF_BACKEND=xrt`) fails with `Unhandled context fault` on
  stream IDs `0x200`, `0x206`, `0x20e`, `0x240`, `0x242` — VectorOP, Conv,
  Pool and MatMul, HPC0 and HPC1.  (With the Secure production bitstream the
  same run passed: no translation.)
- **The kernels run on SMMU memory.**  The same binaries on the default
  (SMMU) backend pass, with no faults and CmaFree unchanged; MNIST from the
  generated library on SMMU buffers: 98.92 % / 0.269 ms, LeNet 97.35 % /
  2.808 ms — no measurable translation cost.  The 148-model board suite
  (`run_remote_tests.py`, every kernel path incl. GEMV and packed B) passes
  148 / 148 on SMMU memory, no context fault.
- **More than CMA.**  `test/fsm_kernel_test 1100`: one 1100 MiB buffer (the
  CMA region is 1000 MiB) at IOVA `0xF_8000_0000`, VectorOP computes
  c = a + b over three 366 MiB regions in one call — all 191 889 408
  elements right, 2.40 GB/s (the HPC0 port limit: 1.6 GB/s read shared by
  a and b, plus the write), CmaFree unchanged.  At 3 × 96 MiB: 125.8 ms on
  SMMU memory, 125.8 ms on CMA (`fsm_kernel_test_xrt 288 --cma`, built with
  `-DFSM_WITH_XRT`, on the production bitstream).
- `test/fsm_test 1100`: zeroed, page-aligned buffers in the window, clean +
  invalidate, argument checks, disjoint buffers, memory returned on close.

**Never load `fpga_smmu_mem` with a Secure (`PROT_VALUE "000"`) bitstream**
— the production one: the library would pick the SMMU backend and hand the
kernels IOVAs they cannot reach (zeros, or with a `DDR_LOW` window, other
memory).  Before trusting a new bitstream, repeat the probe: a run with
`INFERENCE_BUF_BACKEND=xrt` must fail with `Unhandled context fault` in
`dmesg`.
