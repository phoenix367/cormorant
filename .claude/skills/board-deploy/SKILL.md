---
description: Build the KV260 bitstream for the current kernels, load it on the board, verify the PS AXI port widths, then run the on-device correctness set and the demos — with the board pitfalls encoded (reboot before reloading the PL after a hang, UIO names, project regeneration, self-matching pkill). Use after any kernel change that must be verified on hardware, or when a model hangs or mispredicts on the board.
allowed-tools: Bash Read
---

# board-deploy

## 0. Preconditions

- The kernel passed `kernel-verify`.  Do not put an unverified IP on the
  board.
- Board reachable: `ping 192.168.100.8`; SSH key `~/.ssh/kv260-testkey`.
- Vitis env for `xclbinutil`: `source /mnt/data/xilinx/2025.2/Vitis/settings64.sh`.

## 1. Build the bitstream (~30 min, Vivado)

All four kernels are SystemVerilog IPs (`package_<k>_rtl`; the last Vitis
HLS kernel, ConvKernel, was retired in CONV_RTL_PLAN phase 3).  The block
design's `S_AXI_HPC0_FPD` and interconnect crossbar are 128-bit; each
kernel instance's `C_M_AXI_*_DATA_WIDTH` must equal the IP's own default
(128 for every data port; the block designs still say 32 for MatmulKernel
`c`, the retired HLS kernel's width, and the build scripts reset it to the
IP's 128 after the upgrade, `scripts/ip_defaults.tcl`).  The test stand's
four block designs use the same widths with a 128-bit PS port since
2026-09-24, so RTL timing matches the board.  (With the HLS IPs, widening
an instance's width in IP integrator or `config_interface
-m_axi_min_bitwidth` corrupted writes — CONV_OPTIMISATION.md §2.31; the
RTL IPs fix their widths, the parameters are read-only.)  The bitstream
build uses its own tree, `build_hw128`, so it never disturbs the
verification tree `build/` and its timing baselines:

```bash
cmake -S . -B build_hw128 -DAXI_BUS_WIDTH=128     # once (the bus width no longer affects any IP)
make -C build_hw128 build_hw_kv260 > /tmp/hw.log 2>&1   # package the four RTL IPs + Vivado
grep -E "Timing summary|write_bitstream completed|^ERROR" /tmp/hw.log
```

WNS must be positive — at **4.000 ns**: the kernels run at 250 MHz from an
MMCM in the block design (`clk_wiz_0`, `scripts/bd_kernel_clock.tcl`;
FMAX_250_PLAN), with impl_1's directives set by `scripts/build.tcl`
(`set_impl_directives`; the default strategy missed by tens of ps).  The
signed-off bitstream `986cef4866a0` had +0.105 ns; the margin moves by
±0.1 ns from build to build, so a kernel change that leaves < +0.05 ns
deserves a look at the worst paths.  Two checks before any bitstream goes
to the board: `grep -c 'Synth 8-4767' <runs>/synth_1/runme.log` must be 0
(a variable written from several generate scopes: Vivado builds something
other than what the simulators run — the PoolingKernel bug of
FMAX_250_PLAN), and the `neteq_<k>_rtl` targets (each kernel's RTL in
lockstep with its synthesised netlist) after a kernel change.  `synth_1` must NOT run incrementally
(`INCREMENTAL_CHECKPOINT` empty, `AUTO_INCREMENTAL_CHECKPOINT` 0): an
incremental run against a stale reference checkpoint reused the old
MatmulKernel control block and dropped a newly added AXI-Lite register
(2026-09-25) while the HWH and drivers showed it.  After an IP register
change, prove it on the board: write the new register via /dev/mem and
read it back (a register that is not there reads 0).  Post-synthesis utilisation:
`hw/cormorant_hw_128/cormorant_hw_128.runs/synth_1/design_cormorant_wrapper_utilization_synth.rpt`;
per-kernel numbers need `report_utilization -hierarchical` on the synth
checkpoint, or each kernel's own out-of-context run (`synth_<k>_rtl`).

Never run a scheduler project generation (demo `generate_project.py`,
`run_remote_tests.py`) while a driver target rewrites
`build*/kernels/<k>_rtl/driver/` — the generated project would miss its
driver headers and fail preflight.

## 2. Load it

```bash
cd inference-scheduler
.venv/bin/python upload_bitstream.py --config bitstream_config_kv260.json
../.claude/skills/board-deploy/scripts/read_kernel_regs.sh   # widths + kernel states
```

The loader checks that PL0 (`/sys/devices/platform/fclk0/set_rate`) is at
the HWH's 100 MHz before programming (Step 5b; it sets it when another loader
changed it) and again after the xclbin load: the MMCM needs it to lock, and
a design that does not lock stays in reset.  `read_kernel_regs.sh` must show the HPC0 **and HPC1** width fields = **0 (128-bit)** (HPC1 carries conv w/b and matmul B since 2026-09-26; the loader derives both from the HWH)
— the AFIFM encoding is 0 = 128, 1 = 64, 2 = 32 (the loader's PYNQ
table), NOT the other way round.  Until 2026-09-24 the block design
left `S_AXI_HPC0_FPD` at 32 bits (`PSU__SAXIGP0__DATA_WIDTH`), so the
field read 2 and every kernel's traffic was squeezed through one 32-bit
port at 100 MHz (400 MB/s: a 128-bit weight word cost 4 cycles).  The
loader writes the widths LAST on purpose: the overlay's `afi0` node
resets the fields to 0 when applied (§2.31); a reading that does not
match the handoff file's `C_SAXIGP0_DATA_WIDTH` means every kernel will read/write
garbage.

After a reboot the board applies the starter-kit overlay (dfx-mgr) and,
during the xclbin load, a `pynq` overlay whose `fabric@A0000000` node
takes IRQ 61 and outlives its overlay.  The loader now removes `pynq`
and unbinds such a device before applying ours (2026-09-24); if
`inference_init() failed: 2` still shows up, check
`ls /sys/class/uio/*/name` — it must list `fabric_vecop`, not `fabric`
(`rmdir` the foreign overlay, `echo a0000000.fabric >
/sys/bus/platform/drivers/uio_pdrv_genirq/unbind`, re-apply).

## 3. Run

```bash
.venv/bin/python run_remote_tests.py --config remote_config_all_models.json   # 156 models, ~10 min
cd ../demo/image_classification && ../../inference-scheduler/.venv/bin/python scripts/generate_project.py \
    && ../../inference-scheduler/.venv/bin/python scripts/deploy_and_run.py --verbose
cd ../mnist && ../../inference-scheduler/.venv/bin/python scripts/generate_project.py \
    && ../../inference-scheduler/.venv/bin/python scripts/deploy_and_run.py --verbose
```

Regenerate demo projects whenever the scheduler's emitted layouts
changed (weights are packed for the kernel since §2.32; MatMul B since
MATMUL_OPTIMISATION §3b) — and ALWAYS after a kernel gains an AXI-Lite
register: a register keeps its last written value across runs, so a
project generated before the register exists inherits whatever the
previous project left there (2026-09-25: stale demo projects ran with
`b_packed = 1` left by the model set and mispredicted).  The chat
server's libraries too: `demo/chat/scripts/generate_llm_project.py` +
`scripts/llm_board.py --install-only` per model, `demo/chat/deploy.py
--regenerate` for its BERT project (demo/chat/doc/DEPLOY.md).  Run the demos
and the model set SEQUENTIALLY — they share the kernels.  Local configs
must name the VectorOP UIO `fabric_vecop` (the overlay's name after a
clean boot; older boards showed `fabric`).

Record results in CONV_OPTIMISATION.md (§2.31/§2.33 format) and refresh
the demo READMEs' transcripts if latencies moved.

## 4. When a model hangs or mispredicts

1. **Identify the kernel and layer without killing anything:**
   `scripts/read_kernel_regs.sh` prints each kernel's ctrl (idle/done)
   and ConvKernel's argument registers — a hung conv shows idle=0 and its
   full geometry (that is how the §2.30 deadlock was found).  Reproduce
   with a scaled fixture in `conv-rtl-trace`.
2. **Kill the inference, then REBOOT the board before reloading the PL.**
   Reprogramming the PL under a kernel's in-flight AXI transactions
   wedges the HPC port and the next kernel to use it hangs (§2.31).
   `ssh ... 'systemctl reboot'`, wait ~60 s, upload again.
3. In remote `pkill -f`, use a bracket pattern (`"[c]lassify_images"`) —
   a plain pattern matches the ssh shell's own command line and kills it.
4. Everything failing / mispredicting right after a boot → step 2's width
   registers (they must match the handoff file's port width; the
   overlay resets them to 0 = 128-bit).
