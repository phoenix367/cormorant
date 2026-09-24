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

The block design is 128-bit, so the kernels must be synthesised with
`AXI_BUS_WIDTH=128`.  Keep that in a SEPARATE build tree so the 32-bit
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

`read_kernel_regs.sh` must show the HPC0 width fields = 2 (128-bit) for
the four ports the design uses.  The loader writes the widths LAST on
purpose: the overlay's `afi0` node resets them to 32-bit when applied
(§2.31); a 32-bit reading here means every kernel will read/write
garbage.

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
   registers (they read 0 when the overlay reset them).
