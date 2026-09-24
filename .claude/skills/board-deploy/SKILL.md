---
description: Build the KV260 bitstream for the current kernels, load it on the board, verify the PS AXI port widths, then run the on-device correctness set and the demos — with the board pitfalls encoded (reboot before reloading the PL after a hang, UIO names, project regeneration, self-matching pkill). Use after any kernel change that must be verified on hardware, or when a model hangs or mispredicts on the board.
allowed-tools: Bash Read
---

# board-deploy

## 0. Preconditions

- The kernel passed `conv-verify` (or the pool equivalent).  Do not put
  an unverified IP on the board.
- Board reachable: `ping 192.168.100.8`; SSH key `~/.ssh/kv260-testkey`.
- Vitis env for `xclbinutil`: `source /mnt/data/xilinx/2025.2/Vitis/settings64.sh`.

## 1. Build the bitstream (~30 min, Vivado)

The block design's `S_AXI_HPC0_FPD` and interconnect crossbar are
128-bit; each kernel instance's `C_M_AXI_*_DATA_WIDTH` must equal the
exported IP's own default (32 for the 16-bit element ports; 128 for the
ports that are `ap_uint<128>` in C++: conv weight/bias, matmul a/b, pool
x).  The test stand's three block designs use the same widths with a
128-bit PS port since 2026-09-24, so RTL timing matches the board.  Two things that
look like shortcuts and are not (2026-09-24): (a) widening
`C_M_AXI_*_DATA_WIDTH` on an instance in IP integrator — the HLS wrapper
hard-codes `C_M_AXI_*_WSTRB_WIDTH = (<HLS bus width> / 8)` as a literal,
the strobe port stays 4 bits (`0xzzzf` in sim) and only the low 4 bytes
of every beat reach DDR (outputs [0],[1] right, rest zero); (b)
`config_interface -m_axi_min_bitwidth 128` — the 16-bit ports then emit
one single-beat partial-strobe write per element and the PS kept only
lane-0 beats (every 8th output right, rest zero).  A port that must be
wide is widened in the C++.  Keep that in a SEPARATE build tree so the 32-bit
tree used by conv-verify and its timing baseline stay intact:

```bash
cmake -S . -B build_hw128 -DAXI_BUS_WIDTH=128     # once
make -C build_hw128 build_hw_kv260 > /tmp/hw.log 2>&1   # synth all 4 kernels + Vivado
grep -E "Timing summary|write_bitstream completed|^ERROR" /tmp/hw.log
```

WNS must be positive.  Post-synthesis utilisation:
`hw/cormorant_hw_128/cormorant_hw_128.runs/synth_1/design_cormorant_wrapper_utilization_synth.rpt`;
per-kernel numbers need `report_utilization -hierarchical` on the synth
checkpoint (see CONV_OPTIMISATION.md §2.31 for the last set).

Never run a scheduler project generation (demo `generate_project.py`,
`run_remote_tests.py`) while a conv synthesis is rewriting
`build*/kernels/conv/.../drivers/` — the generated project will miss its
driver headers and fail preflight.

## 2. Load it

```bash
cd inference-scheduler
.venv/bin/python upload_bitstream.py --config bitstream_config_kv260.json
scripts/read_kernel_regs.sh            # from this skill: widths + kernel states
```

`read_kernel_regs.sh` must show the HPC0 width fields = **0 (128-bit)**
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
.venv/bin/python run_remote_tests.py --config remote_config_all_models.json   # 126 models, ~9 min
cd ../demo/image_classification && ../../inference-scheduler/.venv/bin/python scripts/generate_project.py \
    && ../../inference-scheduler/.venv/bin/python scripts/deploy_and_run.py --verbose
cd ../mnist && ../../inference-scheduler/.venv/bin/python scripts/generate_project.py \
    && ../../inference-scheduler/.venv/bin/python scripts/deploy_and_run.py --verbose
```

Regenerate demo projects whenever the scheduler's emitted layouts
changed (weights are packed for the kernel since §2.32).  Run the demos
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
