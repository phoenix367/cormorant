# fpga_smmu_mem — SMMU-mapped DMA buffers for the PL kernels (experimental)

A platform driver that gives the KV260 kernels large DMA buffers without CMA:
ordinary pages (allocated in chunks of up to 2 MiB) mapped by the ZynqMP SMMU
at one contiguous IO virtual address.  One buffer per open file of
`/dev/fpga_smmu_mem`:

| Call | Effect |
|------|--------|
| `FSM_IOC_ALLOC` | allocate `size` bytes, zeroed; returns the device address (`iova`, below 4 GiB) |
| `mmap(fd, 0)` | CPU mapping: cacheable, or write-combined with `FSM_ALLOC_WC` |
| `FSM_IOC_SYNC` | clean (to device) / invalidate (from device) a byte range, split at the physically contiguous chunks |
| `close` / `munmap` | the last one frees the buffer |

The user interface is `fpga_smmu_mem.h`.  The generated `inference_buf.c`
(`inference-scheduler/src/codegen/_buf_impl.py`) carries a copy of it
(`test_buf_impl.py` checks they match) and uses the driver instead of XRT
buffer objects when `/dev/fpga_smmu_mem` exists; `INFERENCE_BUF_BACKEND=xrt|smmu`
overrides the choice.

## Build and load

```bash
# on the board (needs ../arm-smmu/arm_smmu.ko, see its README)
make                                   # fpga_smmu_mem.ko
insmod ../arm-smmu/arm_smmu.ko disable_bypass=0
insmod fpga_smmu_mem.ko
# on the host: make overlay DTC=/mnt/data/xilinx/2025.2/Vitis/bin/dtc
mkdir /sys/kernel/config/device-tree/overlays/smmu
cat smmu-mem.dtbo > /sys/kernel/config/device-tree/overlays/smmu/dtbo
cc -O2 -o test/fsm_test test/fsm_test.c && test/fsm_test 1100
```

`smmu-mem.dtbo` enables the SMMU (`stream-match-mask = <0x7f>`) and adds the
`fpga_smmu_mem` node with `iommus = <&smmu 0x200>`: stream IDs
`0x200`–`0x27F`, `S_AXI_HPC0_FPD` and `S_AXI_HPC1_FPD` (TBU0, UG1085 §16).
The probe refuses to bind without an IOMMU domain.

## Status (2026-09-27)

- The allocator works on the board (`test/fsm_test`: zeroed, page-aligned
  IOVAs below 4 GiB, clean + invalidate, argument checks, disjoint buffers; an
  1100 MiB buffer — more than the whole 1000 MiB CMA region — in 24 618
  physical runs at one IOVA range, CmaFree unchanged).
- The SMMU is programmed as intended (SMR0 = `0x200` / mask `0x7f` →
  context bank 0, translate; read back through `/dev/mem`).
- **The PL traffic is not translated.**  With the SMMU active, a model run on
  CMA physical addresses (`INFERENCE_BUF_BACKEND=xrt`) still passes and raises
  no context fault, so the kernels' accesses bypass the SMMU.  The likely
  cause: every HLS `m_axi` port drives `AxPROT = 000`, a Secure access, and
  Secure transactions are not matched against the non-secure stream table
  Linux programs.  Untested fix: a bitstream with
  `C_M_AXI_*_PROT_VALUE = "010"` (Non-secure data) on all twelve kernel
  masters; Linux memory is Non-secure, so the CMA mode would not change.
- `llm_bench` with the generated library in CMA mode is bit-exact with the
  previous build (FNV 186711104 / 3694798025 / 1483111907 at positions
  32 / 256 / 1000).

**Do not load `fpga_smmu_mem` while the current bitstream runs kernels.**  The
library would pick the SMMU backend and hand the kernels IOVAs
(`0x8000_0000`–`0xFFFF_FFFF`), which they would use as physical addresses.
Before trusting a new bitstream, repeat the probe: a run with
`INFERENCE_BUF_BACKEND=xrt` must fail with `Unhandled context fault` in
`dmesg`.
