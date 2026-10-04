---
description: Add an operator that runs on the KV260's A53 CPU inside the generated C project, for ops no kernel can do (model-study route C). Covers a standard ONNX host op (src/host_nodes.py), a pattern fusion (src/fusion.py) or an axi.llm custom-domain op (src/llm_nodes.py / src/vit_nodes.py). The numpy reference and the C helper live side by side and must agree bit for bit, with correct cache coherency, A53-fast loops, the host timing model, tests and docs. Templates for both node kinds and their tests are proven to pass. Use when a model-study census lists an op MISSING, when a frontend needs a new host region, or when rewriting an existing host op for speed.
allowed-tools: Bash Read Write Edit
---

# add-host-op

A host op runs inline in `inference_run()` on the A53. It has no lane
(`kernel_name == ""`) and emits one synchronous `('cpu', idx)` event that
first waits for its in-flight producers. Its liveness interval starts and
ends at that event, and the profiler brackets it as a layer. Each op lives
in one Python module as two parts that must agree:

- **The C helper**, a string that the codegen pastes into `inference.c`.
- **`reference()`**, which the simulator (`src/codegen/_simulate.py`) runs.
  Its outputs become `test_inference.c`'s expected values.

So every generated project checks C against the reference, bit for bit.
Paths below are relative to `inference-scheduler/` unless noted.

## 0. Pick the route

| | standard ONNX op | pattern fusion | `axi.llm` op |
|---|---|---|---|
| when | the op_type appears in exported graphs; tensors at the element type (Q8.8) | a subgraph of primitive ops whose intermediates saturate Q8.8 op by op (x², x³) | a region a frontend (`src/llama.py`, `src/vit.py`) emits; per-channel exponents, f32 / i32 / i16 host tensors, states |
| code | `HostNode` subclass + C string in `src/host_nodes.py`; `HOST_OP_FACTORIES` | matcher in `src/fusion.py`, producing a standard host op | `LlmNode` in `src/llm_nodes.py` (`LLM_C`, `LLM_OP_FACTORIES`), `VitNode` in `src/vit_nodes.py` (`VIT_C`, `VIT_OP_FACTORIES`) or `TtsNode` in `src/tts_nodes.py` (`TTS_C`, `TTS_OP_FACTORIES`) |
| spec | ONNX semantics, plus documented deviations (Gather clamps its index) | the subgraph evaluated in float64 | the study emulation: `demo/chat/scripts/llm_study.py` `Model`, `vlm_study.py` `VisionModel` |
| examples | Softmax, LayerNormalization, Gelu, Transpose, Slice, Gather, OneHot, Cast | TF LayerNorm, GELU tanh / erf | LlmRMSNorm, LlmSiluMul, LlmAttention, VitLayerNorm, VitAttnSoftmax |
| template | `templates/host_op.py.tmpl`, `test_host_op.py.tmpl` | none (§3) | `templates/llm_op.py.tmpl`, `test_llm_op.py.tmpl` |

**Dispatch** (`OnnxGraph.__init__` in `src/graph.py`):

- Domain `axi.llm` goes to
  `{**LLM_OP_FACTORIES, **VIT_OP_FACTORIES, **TTS_OP_FACTORIES}`.
- Otherwise `HOST_OP_FACTORIES` is checked **before** MatMul, Conv, Pool,
  Reshape and VectorOP. Never register an op_type that a kernel runs.
- A factory may return another node class. For example, `make_cast_node`
  returns a `ReshapeNode` for a same-kind Cast.

**Two restrictions:**

- `numeric.check` allows exponent tensors only on MatMuls, on Convs (one
  exponent per tensor), on Reshape aliases with equal exponents, and on
  `LlmNode` subclasses and
  `is_llm_op` kernel nodes. Host tensors and states are allowed only on the
  last two. A standard `HostNode` therefore sees Q8.8 values only.
- `_check_integer_inputs`: only host ops and buffer aliases may read an
  integer tensor (`t.is_int`, raw int16). A float-only op calls
  `self._reject_int(t)`.

## 1. The contract: C == `reference()`, bit for bit

**Reading inputs**
- Standard ops: `Data_t` via `host_ld` (raw / 256, exact); integers via
  `host_ld_int`.
- `axi.llm` ops: `llm_ld(raw, sc[c])` with `sc = 2^-f[c]`, exact.
- `reference()` receives these values as float64, never raw bit
  patterns. An integer tensor's values are its raw integers.

**Arithmetic**
- IEEE double throughout, and every sum runs left to right. In
  `reference()` write sums as `np.cumsum(v, axis)[..., -1:]`. Never use
  `np.sum`, `mean`, `var`, `@` or `dot`: they use pairwise or BLAS order.
- If the spec fixes a different order, implement it literally on both
  sides. Example: `llm_nodes.dot8` uses 8 lane sums with `a` ascending,
  combined as `((0+1)+(2+3))+((4+5)+(6+7))`. See the xattn text in the
  `llm_study.py` docstring.

**No FMA**
- The host section of `inference.c` starts with
  `#pragma GCC optimize ("fp-contract=off")`, and CMake adds
  `-ffp-contract=off`.
- A standalone test harness must pass `-ffp-contract=off` itself: aarch64
  GCC contracts `a*b + c` by default.

**Transcendentals**
- C uses libm. Python uses `host_nodes.libm("exp"|"tanh"|"erf", x)`,
  which calls Python's `math` module (the same glibc).
- Never use `np.exp` or `np.tanh`. `np.tanh` is up to 3 ulp off on 26 % of
  inputs, and `np.exp` switches to SVML on AVX-512 hosts.
- For another function (log, sin, …), add it to the tuple of `_LIBM`.
- `sqrt`, `+`, `−`, `×` and `÷` are correctly rounded, so numpy matches C
  for these.

**Write-back** (the C function and its Python counterpart)

| output | C | Python |
|---|---|---|
| standard, Data_t | `host_st(v)`: `nearbyint(v·256)`, round half to even (= `np.round`), saturate, NaN → 0 | `dtype.host_quantize(y)` |
| standard, integer | `host_st_int(v)`: truncate, saturate | `dtype.int_quantize(y)` |
| `axi.llm`, int16 | `llm_st(v, 2^f[c])` | `_st(dtype, y, self.output.exp_channels(self.F)[None, :])` |
| `axi.llm`, f32 host | `(float)v` | `_f32(v)` |

**Lookup tables.** A function of one 16-bit input can be a 65 536-entry
table, filled at init by the same code path. It is then bit-identical by
construction. Existing examples: `host_gelu_*_lut`, `host_exp_lut_init`,
`llm_silu_table`, `vit_gelu_table`. The Python side builds its table with
the same scalar `math` calls (`silu_table`, `gelu_table`).

**Constants**
- Emit float32 constants with `_c_float` and doubles with `_c_double`;
  both round-trip exactly.
- A fused node carries the graph's constant as a `repr()` string attribute
  (`axi_epsilon`, `axi_c1`, …); read it back with `_num()`.

**Threads.** `host_parallel` splits rows or elements into contiguous
ranges, and every row runs the same code. The result is therefore
identical for 1 … N threads. Never split one reduction across threads.

## 2. Checklist: standard ONNX op (`src/host_nodes.py`)

To instantiate the template:
`sed -e 's/@OP@/<OpType>/g' -e 's/@KIND@/<kind>/g' .claude/skills/add-host-op/templates/host_op.py.tmpl`.
Paste its three blocks where each says. The test template becomes
`test/test_<kind>.py`.

1. **C helper `_<kind>_C`**, placed above `# Helper kinds in emission order.`
   - An argument struct.
   - `host_<kind>_rows(void *p, unsigned r0, unsigned r1)`.
   - `host_<kind>(…)`, which calls
     `host_parallel(fn, &a, rows, host_row_grain(n), 1u)`. For an
     elementwise op use `host_parallel(fn, &a, n, INFERENCE_HOST_MIN_ELEMS, 64u)`.
2. **Register the kind.** Add it to `HOST_C_HELPER_ORDER` and to the dict
   in the last return of `host_c_helper()`. `_source.py::_host_ops_section`
   emits only the kinds that some node's `c_helpers()` names.
3. **Node class** `@dataclass class <Op>Node(HostNode)` with
   `helpers = ("<kind>",)`:
   - `from_onnx_node(cls, node, tensors, index, align_elems, ctx: HostContext)`:
     - Look up tensors with `_resolve` and attributes with `_attrs`.
     - Read constant inputs with `_const(ctx, tensors, name, node, what)`.
       `ctx.consts` holds raw initializers of up to 65 536 elements.
     - Raise a `SchedulerError` for every attribute value the C does not
       implement.
     - Call `_reject_int`, then check numel and shapes.
   - `c_call(ins, out, scratch, direct, dtype)` returns the C lines of the
     call. `ins` are the input pointers (`in0`, …) and `out` the output
     pointer.
   - `reference(ins, dtype)` returns the output values in the logical
     shape, as float64.
   - `describe()` supplies the report.md note and the C comment.
   - Optional hooks:
     - `staged_inputs()` / `direct_inputs()`: a big table read in place
       from its buffer (as `GatherNode` does); it must have a flat layout.
     - `scratch_bytes()`: scratch space, available as `tmp` in the stage
       arena.
     - `c_file_consts(dtype)`: per-node constant arrays, named
       `self.c_prefix + "_…"`.
     - `c_luts(dtype)`: returns `[(c_name, init_call)]` when
       `dtype.host_lut_bits` is set. `init_call` mallocs and fills the
       table and returns 0 (see `host_gelu_*_lut`); the
       `static Data_t *` declaration and the free are generated. Tables
       with the same name are shared. If the op calls `host_lut_map`, add
       its kind to the
       `{"gelu_tanh", "gelu_erf"}` set in `_host_ops_section` so that
       `lut_map` gets emitted.
     - `host_features()`: see §6.
4. **Factory.** Add `HOST_OP_FACTORIES["<Op>"] = <Op>Node.from_onnx_node`.
   That entry flows into `HOST_OP_TYPES` and then
   `graph._ALL_SUPPORTED_OP_TYPES`. As a result, the op census in
   `.claude/skills/model-study/scripts/onnx_study.py` lists the op as
   supported with no further edit.
5. **Integer index inputs** (Gather-like ops): add the node to
   `_simulate.py::_int_fill_range`. Otherwise the test harness's `i % R`
   fill may produce invalid indices.
6. **Fixture.** Add a fixture model to a `test/gen_*_models.py` generator.
   This puts the op into `test_cache_coherency.TestAllModels`, and makes
   it listable in `remote_config.json.example` for the board run.

## 3. Checklist: pattern fusion (`src/fusion.py`)

1. **Matcher** `_match_<x>(ix: _Index, anchor)`. It is structural only;
   never match on node names.
   - Walk producers and sole consumers (`ix.sole`) and check op types.
   - Check constants with `ix.scalar`. Use `_close` (rtol 1e-5) for
     transcendental constants, and exact equality for 0.5, 1, 2 and 3.
   - Every intermediate must be internal: no consumer outside the pattern,
     not a graph output. On any near miss, leave the subgraph alone.
2. **Emit** `oh.make_node(<standard op>, …, axi_<c>=repr(v))`. Carry the
   graph's own constants and arrangement (`axi_tf_form`, `axi_c1`, …), so
   the node evaluates exactly that subgraph in double. `_apply` places the
   node where the pattern's last node was.
3. **Wire it up:**
   - Add the matcher to the `fuse_patterns()` loop and its counts dict.
   - Add the key to the fallback counts dict in `graph.py`.
   - Add a bullet in `report.py::_transformations`.
4. **Pattern-only ops.** Ops that are legal only inside the pattern go into
   `PATTERN_ONLY_OPS`. The dispatch error then gives a hint, and the
   census marks them "pattern-only".
5. **Tests** in `test/test_fusion.py`:
   - The fused node is bit-exact with the subgraph run op by op in float64.
   - Near misses are left alone.
   - `fuse_patterns=False` is honoured.

## 4. Checklist: `axi.llm` op (`src/llm_nodes.py`, vision ops in `src/vit_nodes.py`)

1. **Spec first.** Add the op to the study emulation
   (`llm_study.py` `Model`, or `vlm_study.py` `VisionModel` / `TextModel`)
   with every operation order fixed. The C must reproduce it.
2. **Node class** (template `templates/llm_op.py.tmpl`): `<Op>Node(LlmNode)`.
   A vision op uses `VitNode`, whose `helpers` are `("llm", "vit")`.
   - Declare every tensor's storage with `_want(t, None|"f32"|"i32"|"i16", what)`,
     where `None` means a DMA tensor. Pass `F=ctx.frac_bits`.
   - Exponents come from `t.exp_channels(self.F)`.
     `self._scales(t)` returns `(2^-f name, 2^f name, RuntimeItem)`.
   - List every runtime item in `c_runtime()`; items are deduplicated by
     key project-wide.
   - Tables:
     - `HostTable(name, "bf16"|"f32", data)`, returned by `tables()`,
       becomes a C array. Above `TABLE_FILE_BYTES` (64 KiB) it becomes
       `weights/<name>.dat`, loaded at init by `llm_load_table`.
     - Many per-exponent tables are rows of a `RUNTIME_GROUPS` group
       (`silu_item`, `sexp_item`, `gelu_item`). Rows, not straight-line init
       calls: hundreds of those made gcc's register allocator need 2.6 GB.
   - How each tensor reaches the C call:

     | tensor | passed as |
     |---|---|
     | DMA input (`dma_inputs()`) | a staged pointer |
     | host tensor | its `c_name` (`const float *`, `const int32_t *`, `int16_t *`) |
     | DMA state | `(int16_t *)inference_buf_ptr(name)` |
     | host output | written directly; no `host_out_done` |

   - States:
     - `state_writes()` lists the DMA states the op writes in place. The
       coherency audit then treats them as CPU-dirty, and a kernel may read
       them only after `llm_cache_flush(state, rows, G, C, D)`
       (`LLM_C_DMA`).
     - `state_updates()` lists every state written, host states included.
       These become the DAG's RAW / WAR / WAW edges (`src/schedule.py`).
   - C code goes at the end of the `LLM_C` (or `VIT_C`, `TTS_C`) string. The whole
     string is emitted whenever the project contains any op of that family.
3. **Register** the factory in `LLM_OP_FACTORIES` (or `VIT_OP_FACTORIES`, `TTS_OP_FACTORIES`)
   and add the class to `__all__`.
4. **Frontend** (`src/llama.py` or `src/vit.py`):
   - Emit the node with `self._node(op, ins, outs, name, domain=LLM_DOMAIN, **attrs)`.
   - Declare **every output** with `self._t(name, shape, exp=…, host=…, state=…)`.
     ONNX shape inference does not know custom ops, so an undeclared output
     is unknown to the graph (verified: "axi.numeric.exp: tensor 'y' not in
     the graph").
   - Exponents must lie in [-30, 40] (`numeric.apply`).
5. **Gates, in this order:**
   1. Tiny fixtures (`test/gen_llama_models.py`, or `Tiny` in
      `test_vit.py`), then `test_llama.py` / `test_vit.py`: the simulation
      equals the study, and the host emulation passes.
   2. On the real model, `llm_sched_check.py` / `vlm_sched_check.py`: the
      simulation equals the study bit for bit.
   3. `llm_host_emu.py --incoherent` / `vlm_host_emu.py --incoherent`: the
      generated C on the PC equals the simulation.
   4. `scripts/a64_check.py`: see §5.
   5. `llm_board.py --profile [--images …]`: logits bit-exact on the board.
      The llm-onboard and board-deploy skills cover this step.

## 5. Cache coherency is automatic; keep it that way

The codegen (`_source.py`: `_emit_host_block`, `_emit_llm_block`) does the
following for every host op:

1. Invalidates each input a kernel wrote (`_written_by_kernel`, followed
   through aliases). This happens after the `('cpu')` event has waited for
   that kernel's lane.
2. `host_in` returns the buffer pointer itself when the buffer is cached
   and its layout flat. Otherwise it copies into `s_host_stage`; strided
   layouts are compacted.
3. `host_out` / `host_out_done` hand the output back and flush it (a staged
   output goes through `host_store` first).

**Your helper must:**
- Read only through `in*` and write only through `out` and its declared
  states.
- Never touch any other DMA buffer.
- Read a DMA weight in place only through `direct_inputs()`.

**Checks:**
- The static audit: `CoherencyChecker(cg).check() == []` (from
  `test/test_cache_coherency.py`).
- The dynamic check: `host_emu.build_and_run(..., incoherent=True)`, which
  keeps separate CPU and DDR copies of every buffer.
- In the template proof, a dropped output flush and a dropped invalidate
  were caught only by these two checks.

**The aarch64 code is never compiled on the PC.** The x86 host emulation
does not build `#if defined(__aarch64__) && defined(__ARM_NEON)` code. Run:

`.venv/bin/python ../.claude/skills/add-host-op/scripts/a64_check.py <project>`

- It compiles `src/inference.c` with `aarch64-linux-gnu-gcc` (shipped with
  Vitis) using `-O2 -Werror`.
- It counts NEON double-vector instructions, and exits 1 if any function
  contains an FMA.
- It accepts any generated project: a CLI `--out-dir`, a `host_emu`
  workdir, or `demo/chat/build/llm_project*` (SmolVLM: 18 s).
- It proves compilation and no-FMA only. A NEON branch is proven bit-exact
  only on the board.

## 6. A53 performance: never at the cost of a bit

1. **Measure first.**
   - Build with `-DINFERENCE_PROFILING=ON` for a per-layer profile (host ops
     are bracketed), or use `llm_board.py --profile`.
   - Run with `INFERENCE_HOST_THREADS=1` to see the bare loop cost.
2. **Threads.** `host_parallel(fn, arg, n, grain, align)`.
   - Every range holds at least `grain` items, and its boundaries are
     multiples of `align`.
   - `host_row_grain(n)` keeps at least `INFERENCE_HOST_MIN_ELEMS` (16 384)
     elements per range.
   - A pool wake-up costs about 80 µs, so dispatch once per op. Inside one
     dispatch, separate phases with spin barriers (as `llm_attn_decode`
     does).
   - Order work items so that contiguous ranges get equal work
     (LlmAttention: head-major, rows interleaved).
3. **Latency, not throughput, limits these loops.** The in-order A53 does
   about one double operation per cycle, and with no FMA a multiply-add
   takes at least 2 cycles. gcc -O2 neither unrolls nor vectorises these
   loops, even with `#pragma GCC optimize ("tree-vectorize")`. A
   convert → scale → add → round chain per element waits out every latency.
   Measured fixes (CHAT_PLAN §24, single thread):

   | op | before | after |
   |---|---|---|
   | LayerNorm | 50 ms | 16 ms |
   | residual add | 17.6 ms | 6.8 ms |
   | attention prep | 26.7 ms | 8.1 ms |

   - **4 independent chains.** Process 4 rows (`vit_layernorm_rows`) or 4
     elements per step (`vit_resadd_rows`, `vit_prep_*`). Each sum keeps its
     own left-to-right order. Interleave different rows or columns; never
     split one sum into partial sums unless the spec defines lanes (as
     `dot8` does).
   - **Branch-free stores.** Where the value is provably finite, store with
     `vit_st16f` (an fmin / fmax clamp of `nearbyint`). The NaN test in
     `llm_st` costs a branch per element.
   - **Reciprocal multiply** instead of a divide, only with an exact
     fallback near a rounding tie. Examples: `vit_ln_st`
     (`VIT_LN_TIE = 0.5 − 2^-20`, with the error bound argued in its
     comment) and `vit_smx_rnd` (margin 1e-7).
   - **Tables.** A one-input function becomes a 65 536-entry table.
     Splitting a table into L1-resident parts (`T_hi · T_lo`, the vision
     softmax) changes the spec, so the study must adopt it too. Random
     lookups into 128 KB tables stay memory-bound: GELU with 8 lookups per
     step gained nothing on the board.
   - **Memory.** Pad 2 KB-strided tiles (`VIT_SMX_TP`: the rows otherwise
     share L1 sets), prefetch strided DRAM rows (`VIT_PREFETCH`), and
     transpose through tiles.
   - **NEON.** Put it under `#if defined(__aarch64__) && defined(__ARM_NEON)`
     and keep the scalar fallback. Use lane-wise operations only:
     `vmulq_f64` / `vaddq_f64`, `vrndiq_f64` (frinti, which equals
     `nearbyint`), the saturating `vqmovn_*` as the clamp, `vmaxq_s16`.
     When a vector test finds a near-tie, redo that block with the scalar
     code (`vit_smx_round`).
     Result: the vision softmax went from 16.2 to 6.5 ms per head.
4. **Prove every rewrite.** Build an old-vs-new harness: cut both versions
   out of the C strings, test_host_ops-style.
   - Require bit-identical output at 1, 3 and 4 threads, over several value
     distributions: exact ties and near-ties, saturation, zero and NaN rows,
     all-equal rows.
   - For an op with one int16 input, test all 65 536 inputs for every
     exponent pair in use (`test_exhaustive_c` in the template).
   - Random inputs do not catch a 1-ulp change. In the template proof,
     `x / nrm` → `x * (1.0 / nrm)` passed every random-input test.
   - Then time it **on the board**: compile the harness there with the
     board's gcc 11, `-O2 -ffp-contract=off -pthread`. The PC says nothing
     about A53 speed. Then re-run the gates.

## 7. Host timing model and report

**The model** (`src/host_model.py`, stored in `perf_models/kv260/host.json`)
is keyed by class name:
- Exact entries per `signature(sn)` (the kind plus its shapes).
- A per-kind NNLS fit over `in_elems`, `out_elems`, `rows` and `one`, taken
  from `features(sn)`.
- A node that reads only part of its inputs overrides `host_features()`
  and returns those keys (as `VitAttnSoftmaxNode` does).

**A new kind stays unpriced until it is measured.** Until then:
- `--plan` refuses the order search (`order_log["why"]` says
  "N nodes not priced").
- report.md's Planning section says "could not be priced (counted as 0)".
- `onnx_study.py` lists the node as unpriced.

**To measure it:**
1. Take a board profile: `llm_board.py --profile --out RESULTS`, or the
   `layer_stats` in a demo's `results.json`.
2. Run `perf_calibrate.py host --profile MODEL[:plan]=RESULTS`. MODEL must
   be listed in `SHIPPED` and `_entry_graphs` of `perf_calibrate.py`. See
   the perf-calibrate skill.

**report.md needs no change for a new op.** `_node_notes` prints
`describe()`, and `_transformations` counts every HostNode. A new fusion
does need its own count bullet.

## 8. Tests and verification

| file | extend it for |
|---|---|
| `test/test_<kind>.py` (from the template) | the new op on its own |
| `test_host_ops.py` | standard ops: `_Base.check` compiles `_host_ops_section()` plus a harness, and compares with `reference()` bitwise at 1 / 3 / 4 threads |
| `test_fusion.py` | pattern fusions |
| `test_bert_tiny.py` + `gen_bert_models.py` | BERT-like end to end (partition, onnxruntime, study emulation, host emulation) |
| `test_llm_ops.py` | `axi.llm` ops: `emulate()` runs cached / staged × threads × incoherent; standalone C via `HOST_C_POOL + llm_c_helpers()` |
| `test_llama.py`, `test_vit.py` | simulation == study; entry and multi-entry projects on the host emulation |
| `test_cache_coherency.py` | automatic for every model in `test/models` |

The host emulation has no Pool model (`host_emu` raises). Chain tests
therefore use VectorOP, MatMul or Conv.

Run from `inference-scheduler/`:

```bash
.venv/bin/python -m pytest test/test_<kind>.py -v
.venv/bin/python -m pytest test/test_host_ops.py test/test_fusion.py test/test_cache_coherency.py \
    test/test_bert_tiny.py test/test_report.py test/test_cli.py -q          # standard ops
.venv/bin/python -m pytest test/test_llm_ops.py test/test_llama.py test/test_vit.py -q   # axi.llm
.venv/bin/python -m pytest test/ -q && .venv/bin/ruff check .                 # before handing over
.venv/bin/python ../.claude/skills/add-host-op/scripts/a64_check.py <generated project>
```

**Standalone harness flags:**
`-std=gnu99 -O2 -Wall -Wextra -Werror -Wno-unused-function -pthread -ffp-contract=off -DINFERENCE_HOST_MIN_ELEMS=1u … -lm`.
Run the binary under `INFERENCE_HOST_THREADS` = 1, 3 and 4.
`-DINFERENCE_HOST_MIN_ELEMS=1u` makes even tiny inputs split across the
threads; without it only thread 0 runs.

**On the board:**
- Standard ops: add the fixture to `remote_config.json.example` `models`
  and run `run_remote_tests.py`. The board compares every output with the
  Python ground truth (board-deploy skill).
- `axi.llm` ops: `llm_board.py` (§4).

## 9. Docs to update

- `doc/scheduler/INFERENCE_SCHEDULER.md`:
  - the "Host-CPU ops" bullets (~l. 48);
  - §Host-CPU ops table, §Pattern fusion table, the §Llama-family decoders
    "Host ops" table and §Vision encoders;
  - the test count.
- `CLAUDE.md` (root): the "(host CPU)" rows of the Inference Scheduler
  table, and the key source files.
- `inference-scheduler/CLAUDE.md`: the intro list, the Source Layout lines,
  the Node Classes paragraph, and the test count.
- Other op lists:
  - `README.md` (~l. 99)
  - `inference-scheduler/doc/USER_GUIDE.md` (~l. 181)
  - `inference-scheduler/doc/ARCHITECTURE.md` (l. 123, plus the mermaid
    edge at ~l. 322)
  - `inference-scheduler/doc/SCHEDULER_DAG.md` (~l. 414)
- The plan doc that motivated the op (BERT_PLAN, CHAT_PLAN or
  `<MODEL>_PLAN`): the op, its gates and its board time.
- No census edit is needed for a standard op. A pattern-only op goes into
  `PATTERN_ONLY_OPS`. `axi.llm` ops never appear in exported ONNX graphs.

## Rules and pitfalls

- **Spec first.** Change the C and `reference()` together, in the same
  commit and in the same operation order.
- In `reference()`, never use `np.sum`, `np.mean`, `np.var`, `np.dot`, `@`,
  `np.exp` or `np.tanh`. Use `np.cumsum` for sums and `libm()` for
  transcendentals.
- **Template BLOCK headers are one line.** In the proof, a wrapped header
  comment line was pasted into the C string and broke compilation. Keep
  pasted blocks free of stray `#` lines.
- **Custom-domain outputs need value_info.**
- **No exponents outside `LlmNode`**, and no op_type that a kernel runs in
  `HOST_OP_FACTORIES`.
- **Unpriced ops.** A new kind disables the `--plan` order search until
  `host.json` has it (§7).
- **Board discipline.** Stop the chat server before board runs, and never
  run two board jobs at once (board-deploy skill). Do not commit unless the
  user asks.

The templates were proven by instantiating them in a scratch copy of the
scheduler:
- **Standard op**, as LpNormalization (p = 2): the C matches the reference
  at 1 / 3 / 4 threads, including a zero row (NaN → 0). A VectorOP chain
  covered the staged path, cacheable and incoherent runs, and the audit.
- **`axi.llm` op**, as LlmSoftsign: alone and between two MatMuls on the
  host emulation, plus all 65 536 inputs at four exponent pairs.
- **Regression and lint:** the related existing suites still passed and
  ruff was clean.
- **Mutations:** a wrong formula, a truncating store, a dropped flush and a
  dropped invalidate were each caught.
