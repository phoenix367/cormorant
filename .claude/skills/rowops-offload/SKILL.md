---
description: Move a host-CPU op of the inference scheduler onto the row-ops unit (RowOpsKernel) as a kernel node — the node class (jobs, reference, kernel_calls, emit), its factory in the override tables, the frontend flag and study policy, the specification function, the tests (simulation == spec, host emulation, kind counts), the regenerated libraries and the board gate. Written from HOSTOPS_PLAN phases 4a / 4b and RPU_PLAN track A1, with the traps each of them hit. Use when a plan maps a host op onto COPY / ADDX / DEQUANT / MULACT / NORM / ROPE / TRANSPOSE, or when a unit node disagrees with the board.
allowed-tools: Bash Read Write Edit
---

# rowops-offload

The row-ops unit (`kernels/rowops_rtl`, `doc/kernels/ROWOPS_RTL_KERNEL.md`)
takes a host op when the op is a bit copy or an integer formula the unit
computes (`inference-scheduler/src/rowops_spec.py`).  Paths are relative to
`inference-scheduler/` unless noted.  A unit node is an `LlmKernelNode` on
the `RowOpsKernel` lane: one or more `run_rowops()` jobs, non-blocking,
`kernel_wait(KERNEL_ROWOPS)` between dependent jobs of the same node.

## 0. Decide the mapping first (§1.4 of the unit's reference)

| the op does | unit job |
|---|---|
| copies rows / tiles with a 3-level address pattern (rows `s1`, blocks `s2`), re-rounds, adds a per-column bias, ReLU, zeroes rows (`win_lo`, `win_hi`) | COPY |
| adds two rows with per-column shifts (i16 / i32 in, i16 / i32 out, exact) | ADDX |
| int16 / int32 → float32 (`y = float32(raw · 2^-sh_c[c])`) | DEQUANT into a `dma: f32` tensor |
| gathers a matrix's diagonals, transposes rows into granules, interleaves phases | TRANSPOSE (R ≤ 8 rows per tile, L ≤ 1024; `b_s2 = 2R` contiguous, else byte-strobed granules) |
| needs a runtime-indexed gather, an unaligned row start, float math, tanh / sigmoid, a non-power-of-two scale, rows > 4096 | stays on the host (RPU_PLAN §3.3 lists the kernel features it would need) |

Hard rules of the contract that the host emulation did not check until
RPU_PLAN A1 (it aborts now):

- **A stream A / B / P and a non-TRANSPOSE C start 16-byte aligned, and so
  do the strides the job uses** (`a_s1` with `rows > 1`, `a_s2` with
  `blocks > 1`).  Reading a diagonal, or "row k from element j", is
  impossible; write diagonals instead (TRANSPOSE R = 1 granules at a
  2(T + 1)-byte stride) and give the destination guard rows so every
  source row is read from its aligned start (`TtsRelBandRoNode`).
- A TRANSPOSE's destination (`c`, `c_s1`, `c_s2`, the granule stride
  `b_s2`) is 2-byte aligned; nothing else is.
- A raw-integer constant the unit reads (a parameter block, a table, key
  rows) ships as a **Q8.8 weight whose words are the raw image**
  (`rowops_nodes.raw_as_data`, `param_tensor`): an initializer with
  `exp = 0` is still encoded at Q8.8 by the weight ROM (the K-cache
  key-rows bug of A1 — the board read ±256).
- A row of `n > 4096` is several jobs; with per-column exponents each
  chunk needs its own parameter block (`DequantRoNode`: blocks named by
  the chunk's exponents, so a project's entries share them).
- A state written only by the unit starts at zero (the generated init
  memsets DMA states); a float32 DMA state may have no initial value.
- An int32 DMA tensor is read by host ops as `(int32_t *)inference_buf_ptr`
  and never through `host_in`; a float32 DMA state read by a host op is
  invalidated first (`_emit_llm_block`).  An int32 / float32 DMA *graph
  input* has no consistent test-harness fill — do not write such a test.

## 1. The node (`src/rowops_nodes.py`)

```python
@dataclass
class XRoNode(RowOpsNode):
    """What it computes, in the host op's terms; the job list."""
    rows: int = 1
    n: int = 1
    pz: object = field(default=None, repr=False)      # the all-zero P_BCAST entry for plain copies

    def state_writes(self): return [...]              # DMA states the jobs write (the DAG's WAR / WAW edges)
    def state_updates(self): return self.state_writes()
    @property
    def opword(self): return rs.opword(rs.OP_COPY, rs.K_I16, rs.K_I16, rs.K_I16, p_bcast=True)
    def _jobs(self, layouts): ...                     # [(comment, a, a_off, b, b_off, c, c_off, p, op, n, rows,
                                                      #   blocks, a_s1, a_s2, b_s1, b_s2, c_s1, c_s2, aux, win_lo, win_hi)]
    def reference(self, ins, dtype): ...              # the host op's value, bit for bit, on logical arrays
    def kernel_calls(self, layouts): return merge([KernelCall.of("RowOpsKernel", ...) for j in self._jobs(layouts)])
    def describe(self) / emit_comment(self) / emit_call(self, layouts)   # emit_call: self.call(...) per job,
                                                                         # "kernel_wait(KERNEL_ROWOPS);" between jobs
```

- Row strides come from `row_stride_bytes(t, layouts, n, elem_bytes, node, align)`
  (a chunked layout's stride), never from `n` alone, unless the node
  re-rows a flat tensor itself.
- `reference()` writes a state in place (`arrays[state][...] = ...`: the
  simulator passes the logical state array) and returns the output;
  `_simulate` writes a kernel node's state output in place too.
- The reference uses the spec: `rs.sat16`, `rs.rne_shr_vec`,
  `rs.bias_round`, `rs.layer_norm`, … — never a reimplementation.
- The host op keeps its class: the factory returns the host node when the
  platform has no unit (`enabled()`), when the op's attribute `rowops`
  is absent (`_rowops_attr`), or when a geometry rule fails; `_need_unit`
  raises when a frontend asked for the unit without it.
- Register the factory in `LLM_OVERRIDES` / `VIT_OVERRIDES` /
  `TTS_OVERRIDES` (merged over the host factories in `graph.py`), or in
  `graph.py` before `HOST_OP_FACTORIES` for a standard ONNX op
  (`from_onnx_transpose`); a dict entry that names a function defined later
  in the file is a `NameError` at import.

## 2. The frontend and the specification

- The frontend (`src/llama.py`, `src/vit.py`, `src/piper.py`) decides with
  its `rowops` flag (and a sub-flag with a default constant when the move
  is a numerics change: `vit.IEMBED_DEFAULT`, `piper rel_kernel`): it
  declares the DMA tensors / states the node needs (`dma="i32"` /
  `"f32"`, `state=True` with a zero init, `layout` for group-major
  caches), the constants as Q8.8 images, and the node with `rowops=1`.
- Where values change, the study script is the specification and must
  compute the same numbers first: a policy in `demo/chat/scripts/llm_study.py`
  / `vlm_study.py` / `demo/tts/scripts/piper_vits.py` (+ `piper_study.py`
  encoder policies), gated as before (model-study skill §2.3: the shipped
  policy's metrics are the yardstick; a step down on the greedy answers is
  a NO-GO even when top-1 / KL barely move — A1's `iembed`).
- `vision_policy()` / `lp.policy()` / `project.json` carry the flags so
  `*_sched_check.py`, `tts_board.py`, `tts_host_emu.py` and the tests pick
  the matching spec.

## 3. Tests (the order that finds bugs cheapest)

1. `cg._forward_pass(...)` of the frontend's graph against the spec for
   several lengths (`T`, `T − 5`, `17`, `1`): the simulation equals the
   specification bit for bit.
2. `host_emu.build_and_run(cg, td)`: the generated C equals the simulation
   (the unit's C model, `xrowopskernel.h`, checks the alignment contract;
   use `incoherent=True` when the node adds a host read of a DMA tensor).
   To localise a divergence, append intermediates to the ONNX graph's
   outputs (`m.graph.output.append(vi[name])`) and rerun.
3. `test_perf_calls`-style: `emitted_calls(sn.emit_call(layouts)) ==
   sn.kernel_calls(layouts)`.
4. Kind counts in the frontend tests (`test_piper`, `test_vit`,
   `test_rowops_unit`, `test_bert_base`), the `run_rowops(` counts of a
   generated source, "needs the unit" / "flag off keeps the host op".
5. `python3 ../.claude/agents/run-tests/run_tests.py --suite all`, then
   `python3 tools/facts/facts.py check` / `fix` (the test count moves).

## 4. Regenerate, gate, price

- Regenerate every library the node touches (`generate_llm_project.py` per
  model, `generate_tts_project.py`, the BERT demo with `--regenerate`); grep
  the generated `inference.c` for the node's comment before a board run.
- The board gates are the libraries' own (`llm_board.py --profile --out`,
  `tts_board.py --profile --out`, `run_demo.py --profile-layers`), one job
  at a time — the `board-window` skill runs the whole sequence.
- The perf model needs a top-up (perf-calibrate §0: `cases`, `run
  --resume`, `fit`, refinement), and the host model a `host --merge` with
  every profile when a host-op kind appeared or vanished (A1's
  `LlmToHostNode` was unpriced until then); `simulate` must show no
  unpriced node.
- Document: the unit reference's §1.5 table row, `INFERENCE_SCHEDULER.md`'s
  unit paragraph, both `CLAUDE.md` tables, the plan's results section.

## Pitfalls seen so far

- A job offset computed from the wrong row count: the V-image jobs used
  the K cache's rows after the K cache grew (A1) — take each cache's own
  `shape[0]`.
- The softmax masks keys / queries ≥ n, so a static band add over all T
  rows is fine; the host op's `n`-bounded loop need not be reproduced.
- A profile's kind map (`generate_llm_project.KINDS`, the BERT demo's
  `node_kind`) keys on node classes: map the new class or it lands in
  "other".
- Two unit nodes writing the same state (the band per head, the K cache
  per layer) need no extra sync: the lane is one engine and the DAG orders
  state writes; a host reader of a unit-written state does need the
  invalidate.
