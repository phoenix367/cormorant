# tools/rpu — the Cortex-R5 BSP of a bitstream

The kernel-driver firmware on the R5 (`inference-scheduler/runtime/rpu/`,
`doc/plans/RPU_PLAN.md` §9) links against a standalone BSP built from the
implemented design, so it knows the kernels' AXI-Lite bases and PL interrupt
lines (`xparameters.h`).  The firmware knows no model: one build per
bitstream.  Rebuild the BSP after every bitstream whose kernels, addresses
or interrupts changed.

```bash
source /mnt/data/xilinx/2025.2/Vitis/settings64.sh
vivado -mode batch -nolog -nojournal -source tools/rpu/write_xsa.tcl \
       -tclargs hw/cormorant_hw_128/cormorant_hw_128.xpr $TMPDIR/rpu/cormorant.xsa   # ~1 min
vitis -s tools/rpu/make_r5_bsp.py $TMPDIR/rpu/cormorant.xsa $TMPDIR/rpu/ws          # ~3 min
# -> $TMPDIR/rpu/ws/cormorant_r5/psu_cortexr5_0/standalone_r5/bsp
```

Then, with the chat server stopped (`demo/chat/deploy.py --stop`), gate any
generated projects with DMA inputs and outputs (the demos' `build/projects`):

```bash
cd inference-scheduler
.venv/bin/python rpu_board.py --config remote_config.json \
    --bsp-dir $TMPDIR/rpu/ws/cormorant_r5/psu_cortexr5_0/standalone_r5/bsp --runs 20 \
    --project ../demo/mnist/build/projects/mnist_convnet \
    --project ../demo/image_classification/build/projects/resnet18
```

`rpu_board.py` builds the firmware, uploads it with the overlays
(`board/kv260/rpu/rpu.dts`: RPU core 0, its TCMs, `zynqmp_r5_remoteproc`;
`rpu-ipi.dts`: the A53's doorbell, IPI channel 7 as the `rpu_ipi` UIO device,
so the A53 sleeps while the R5 runs a batch) and the projects, starts the
remote processor, and per project builds the
Linux library and the harness `rpu/rpu_host.c` on the board and runs N
inferences: each first by the library driving the kernels itself (the
reference), then with `inference_rpu_attach()` — the library records every
kernel call's register block, Start and wait into a command ring, the R5
replays them with interrupt-driven waits, the host ops and the outputs stay
on the A53 — compared bit for bit and timed; it reports the R5's batches,
calls, waits and interrupts per lane.  `--mode auto` lets each project's
policy choose instead (`INFERENCE_RPU_RECOMMENDED`, `src/rpu_policy.py`:
small models stay on the A53), `--mode both` (the default) runs one pass of
each.

`--start-only` brings the firmware up and leaves it running (every library
on the board then attaches by itself, `INFERENCE_RPU=0` / `1` overrides its
policy), `--stop` stops it.  For the chat server, `demo/chat/deploy.py --rpu
BSP_DIR` (or `rpu.enabled` + `rpu.bsp_dir` in `chat_config.json`) builds and
starts the firmware before the server and stops it with the server.
