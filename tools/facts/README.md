# facts — the fact registry

Many facts in this repo are stated in more than one place. The scheduler's test count is
quoted in eleven places. The bitstream id is in ten files. A kernel's registers are in its
HLS source, its driver, the scheduler, two benchmarks, the calibration runner
and the timeline viewer.

`facts.yaml` (at the repo root) records, for each such fact, **where it is
true** and **every place that repeats it**. `tools/facts/facts.py` checks that
those places agree, fixes the stale ones it may fix, and tells you what else
to touch when you change one of them.

The registry stores pointers, not copies. A derived fact has no value in
`facts.yaml`: its value is read from the code on every run. Only measured
facts (`recorded`) carry a value, and they also carry its provenance.

```bash
python3 tools/facts/facts.py check              # every fact; exit 1 when one fails (--strict: warnings too)
python3 tools/facts/facts.py check 'registers.*' -v
python3 tools/facts/facts.py fix                # rewrite the stale mentions of `fix: auto` facts (--dry-run)
python3 tools/facts/facts.py changed            # the facts this branch touches, checked, + their files you did not change
python3 tools/facts/facts.py impact kernels/matmul/kernel/MatmulKernel.cpp   # what a file is part of
python3 tools/facts/facts.py impact registers.MatmulKernel                   # what a fact ties together
python3 tools/facts/facts.py verify             # re-measure recorded facts where they can be measured (local builds)
python3 tools/facts/facts.py install-hook       # git commit checks the facts the staged files touch
python3 -m unittest tools/facts/test_facts.py   # the tool's own tests
```

It needs PyYAML: the system `python3` usually has it, and the scheduler's venv
gets it from `inference-scheduler/requirements.txt`. `py:` sources and `{python}` in
commands use the scheduler's venv, `inference-scheduler/.venv/bin/python`,
which is set by `python:` in `facts.yaml`. A full check takes about 5 s. The
run-tests helper runs it as the `facts` suite
(`.claude/agents/run-tests/run_tests.py --suite facts`, part of the default
set).

## Pre-commit hook

`python3 tools/facts/facts.py install-hook` puts a small shim in
`.git/hooks/pre-commit`. Hooks are not cloned, so run it once per clone.
On every `git commit`, the shim runs `tools/facts/pre-commit` from the
checked-out tree, which runs `facts.py changed --staged --quiet`:

- **What it checks.** The facts that involve a staged file: a source,
  producer, consumer, plugin argument, mention or `watch` glob. When
  `facts.yaml` or `tools/facts/` is staged, it checks every fact.
- **How long it takes.** A commit that touches no fact costs well under a
  second. A staged test file re-collects that suite (1–2 s). A registry
  change checks all of them (about 7 s).
- **When it refuses.** A failing fact refuses the commit. The hook prints
  the errors, the `facts.py fix …` command for stale `fix: auto` counts
  (then `git add` them), and how to skip it. Warnings, such as a stale
  recorded fact, are printed but do not block.
- **Skipping it.**
  - `git commit --no-verify` or `FACTS_SKIP=1 git commit` skips it once;
  - `FACTS_HOOK=all git commit` checks every fact;
  - `FACTS_PYTHON` picks the interpreter, which defaults to the
    scheduler's venv, else `python3`.
- **Without PyYAML** the hook skips the check with a note; it never blocks
  a commit for a missing tool.
- **It reads the working tree**, as most hooks do. When an involved file
  also has unstaged changes, the report says so: the committed version may
  differ from what was checked. `commit -a` and `commit PATH` are handled,
  because git sets `GIT_INDEX_FILE` for them.
- **Other branches and worktrees.** The hooks directory is shared by every
  worktree of the repository. On a branch without `tools/facts/` the shim
  does nothing.
- **Removing it.** `install-hook --uninstall` removes it. An existing
  pre-commit hook is never replaced without `--force`; instead, add
  `"$(git rev-parse --show-toplevel)/tools/facts/pre-commit" || exit 1`
  to it.

## Kinds

| kind | true where | checked |
|---|---|---|
| `value` | a `source`: a command's output, a file (regex or JSON path), a glob, a Python expression in the venv | every `mentions` locator and every marker quotes the value; `exists` files are present |
| `interface` | a `producer` extracted to a mapping, or a `check` plugin | each `consumers` entry extracts to the same mapping. Per consumer: `keys` (a glob) restricts it, `exclude` drops keys it does not handle, `subset: true` lets it name only some keys, though never an unknown one. A plugin returns its own findings |
| `recorded` | `value` + `provenance` (commit, date, how, `facts`) | The mentions quote it, within `tolerance`: a number, or `{abs: x, rel: y}`. It is **stale** (a warning) in two cases: a fact named in `provenance.facts` (e.g. the bitstream id it was measured on) now has another value, or commits touched its `depends_on` since the provenance commit. `verify` re-reads it from where it can be measured |

A fact marked `optional: true` has a source that only some checkouts
have, such as the `hw/` submodule or a configured `build/`. Where its
source is missing it is **skipped**, not failed. Its mentions are still
checked wherever the source exists.

## Sources

A source is evaluated, then `transform` (a Python expression on `v`), then
`type` (`int`, `float` or `str`).

| spec | value |
|---|---|
| `{cmd: "...", cwd: DIR, parse: REGEX}` | group 1 of the command's output (`{python}` = the venv interpreter) |
| `{file: F, regex: R}` | group 1 of the first match. A regex with more groups gives the tuple. `all: true` gives every match. `as: dict` maps group 1 to group 2 (`swap: true`: 2 to 1) |
| `{file: F, json: a.b}` | that JSON value |
| `{glob: PATTERN}` | the matching repo paths (tracked files; on-disk build artifacts as a fallback) |
| `{py: "module:expr", cwd: DIR}` | `expr` evaluated in the module's namespace (private names too) by the venv, as JSON |
| `{parts: {key: spec, ...}}` | a mapping, one source per key |

`{fact:ID}` inside a string is replaced by another fact's value. That is
how `perf.model_size` opens the model of `perf.bitstream_id`.

Transforms have `ast` (`ast.literal_eval`), `json`, `re`, `one(xs)` (exactly
one item), `basename`, `stem` and the plain builtins.

## Mentions

A regex locator, `{file: F, regex: R}`, quotes the value in group 1. For a
mapping value it uses named groups, one per key:
`(?P<smollm2_135m>\d+)`. Each match is checked. Optional fields:
- `count: N`: the locator must match exactly N times;
- `dotall: true`: `.` also matches newlines;
- `format: EXPR`: how this text renders the value, an expression on `v`
  (e.g. `10` as "ten"), used to compare and to fix;
- `fix: auto | report`: overrides the fact's `fix` for this one mention.

A locator that matches nothing is an error ("locator lost"). It means
someone edited the text, and nothing passes silently.

In prose, a marker works better than a regex: wrap the value in
`<!-- fact:ID -->1625<!-- /fact -->`, or `fact:ID.KEY` for one key of a
mapping. The marker is invisible on GitHub. It is found in any tracked `.md`,
`.txt`, `.html` or `.rst` file without being listed in `facts.yaml`. Markers
do not work inside code blocks, where they would render; use a regex there.

The `fix` policy:
- `fix: auto`: `facts.py fix` rewrites stale quotes. Use it for numbers
  whose sentence stays true, such as test counts.
- `fix: report`: only flags them. Use it for sentences with a date or a
  story ("topped up on 2026-10-01 to 1505 calls"), which a person has to
  rewrite.

## Check plugins (`plugins.py`)

| plugin | args | checks |
|---|---|---|
| `register_map` | `kernel`, `prefix`, `hls`, `driver` (glob, a build artifact), `fields_alias`, `not_keyed` {register: why}, `decode`, `writers_all` | It checks five things, sketched below the table. |
| `cli_flags` | `script`, `cwd`, `docs` [{file, start, end, pattern}] | the flags of argparse's `usage:` against the flags that start the lines (or table rows) of each doc section, or that `pattern` finds in it (a usage synopsis), both ways |
| `ctypes_bindings` | `files`, `external` | `.claude/agents/code-audit/ctypes_check.py`. Mismatches, inconsistencies and undeclared non-int restypes are errors. A symbol without a prototype that is not listed in `external` is a warning. `loose` (`c_void_p` for an opaque handle) is info |
| `config_keys` | `example`, `code`, `waive` {key: why} | `.claude/agents/code-audit/config_keys.py`. An unread example key or an undocumented key that is read is an error unless waived. A waiver that matches nothing is a warning |
| `names_in_docs` | `names` (a source giving a list), `missing` (how to word an unknown name), `docs` [{file, start, end, pattern, forward, reverse, extra_ok, skip}] | Every name shows in each doc section (`pattern`'s group 1; `skip`: names a section may omit; `forward: false`: none required). With `reverse`, every name the section shows must exist, unless listed in `extra_ok` |
| `script_flags` | `pairs` [{caller, function, callee (one or a list)}] | The `--flag` literals a function passes (`flags_check.py`) against what the callee's argparse `usage:` accepts, which includes flags added by helpers such as `add_plan_args` |
| `json_excerpt` | `json`, `doc`, `start`, `complete` | A doc's copy of a JSON file (the first ```` ```json ```` / ```` ```jsonc ```` block after `start`, comments allowed): every value it shows equals the file's; with `complete` it shows every key |

`register_map` checks that:
- the HLS `s_axilite` ports equal the driver header's registers, where the
  driver is built (a mismatch is a warning, because the header is a local
  artifact);
- every port is a `src/perf_calls.FIELDS` field (through `fields_alias`) or
  is `not_keyed` with a reason;
- the timeline's `DECODE` table decodes only FIELDS;
- every `writers_all` file sets every register through `<prefix>_Set_*`,
  and no file sets a register that does not exist.

A plugin is `fn(args, ctx) -> [(level, message, where)]`. Register it in
`PLUGINS`.

## The facts (2026-10-02)

| id | kind | what |
|---|---|---|
| `scheduler.test_count`, `scheduler.test_modules`, `chat.test_count` | value | the suites' collected counts, quoted in 11, 2 and 8 places |
| `perf.bitstream_id` | value | the committed performance model's bitstream. `exists`: its cases / calib files and the perf-regression baseline |
| `perf.model_size` | value | exact calls of the model; host-model signatures and kinds (`fix: report`: dated sentences) |
| `cli.inference_scheduler` | interface | the CLI flags against `inference-scheduler/CLAUDE.md` § CLI and USER_GUIDE § Options |
| `registers.{VectorOP,Matmul,Conv,Pool}Kernel` | interface | the register maps (`register_map`) |
| `vectorop.codes` | interface | the `Op` / `Act` enums against `nodes.py`, the CLAUDE.md table, `run_remote_perf._OP_NAMES` and the timeline `DECODE` |
| `ctypes.tts`, `ctypes.llm` | interface | the chat server's ctypes bindings against the C headers |
| `config.chat` | interface | `chat_config.json.example` against the code reading it, with 5 waivers |
| `chat.pool_mib` | recorded | each chat model's DMA pool (MiB). It is verified against the generated projects in `demo/*/build`, and stale when the layout code changes |
| `scheduler.onnx_ops`, `scheduler.axi_llm_ops` | interface | the ops the scheduler accepts (32 ONNX ops, 32 `axi.llm` ops) against the op lists of USER_GUIDE §2, CLAUDE.md and INFERENCE_SCHEDULER.md (`names_in_docs`) |
| `scheduler.event_kinds` | interface | the event-stream kinds against the C emitter, the liveness pass, the timed replay and the two docs that list them |
| `scheduler.file_thresholds` | value | when weights, expected outputs and host tables go to `.dat` files |
| `planning.thresholds` | value | `--plan`'s minimum gain (3 %) and largest trusted family error (5 %) |
| `perf.shipped_models` | value | how many models `perf_calibrate.py` measures, quoted as a word |
| `remote_tests.model_count` | value | the on-board correctness suite's models (`remote_config.json.example`) |
| `platform.kv260_excerpt`, `platform.kv260_clock` | interface, value | PLATFORM_CONFIGURATION.md's copy of `kv260.json`; the 150 MHz HLS target in six docs |
| `chat.http_routes` | interface | the server's routes against API.md, and against the routes five other docs name |
| `cli.script_calls` | interface | the flags of five script-to-script calls (`deploy.py` → the server, the BERT generator; `e2e_check.py`; `llm_calibrate.py`; `run_remote_tests.py`) |
| `tts.max_ids` | value | Piper's encoder buckets; the largest (400) is the server's packing size in two files |
| `board.results` | recorded | the README's headline board results, quoted 24 times in 9 files, within 3 %; stale on another bitstream; verified against the demos' `results.json` |
| `platform.kv260_bounds` | value | the tiles and bounds of the README's kernel table, from `platforms/kv260.json` |
| `board.pl_clock_mhz` | value, optional | the PL clock (the block design's PL0, 100 MHz) in the README, two docs and the two `MHZ` constants of the performance model |
| `hw.utilization` | recorded | the bitstream's hw commit (d7ce129) and its DSP / LUT / BRAM / URAM use, verified against the local Vivado placed-utilization report |
| `board.uio_labels`, `board.uio_map` | interface | the overlay's UIO names (`dts/kv260/cormorant.dts`) and which kernel uses which, in the six example configs, two doc tables, the README, a skill and the MNIST README |
| `bert.model_mb` | value | the 435 MB BERT download (`fetch_assets.py`), quoted in 11 files |
| `chat.limits` | value | the 1024-position context and the 4096-character speech input |
| `piper.chunk` | value | Piper's chunk: 128 frames = 1.49 s of audio, 192 decoder frames |
| `build.make_targets` | interface, optional | every make target that the README and CLAUDE.md name exists in a configured `build/` (`make help`) |
| `tests.suites` | value | the run-tests helper's `default` / `all` suite sets, in the README, CLAUDE.md, the agent and the helper's docstring |
| `toolchain.vivado` | value, optional | the Vitis / Vivado release (2025.2, from the hw project) in 11 files |
| `cli.chat_client` | interface | `chat.py`'s flags against its own usage synopsis (a `cli_flags` doc with a `pattern`) |
| `chat.banner_columns` | value | the narrowest terminal `chat.py` draws its banner in (60), from `LOGO` |
| `tts.sample_lengths`, `tts.sample_links` | value, interface | the voice samples' lengths (`samples/manifest.json`) in the TTS README; every sample a doc links to exists, and the TTS README lists them all |
| `docs.demo_video` | value | the demo video's link. Its value lives in `facts.yaml`: change it there, then `facts.py fix docs.demo_video` rewrites the six copies |

What the first run found:
- five stale scheduler test counts (README ×2, TESTING ×2, the docs-audit
  skill); the last one was found after its locator was added;
- `bench_vectorop.c` never wrote the `act` register, so a benchmark
  inherited the fused activation of the previous program on the board;
- `smollm2.model_id` and `piper.model_id` were read by `deploy.py` but
  missing from the example config.

The second batch found gaps in INFERENCE_SCHEDULER.md:
- its "Hardware Kernels" summary listed no text-to-speech op;
- its vision-encoder section described `VitEmbedAdd`, `VitLayerNorm`,
  `VitGelu` and `VitResAdd` without naming them.

All are fixed. Each fact of the second batch was also checked by breaking
its source or a doc on purpose (16 mutations, all caught).

The third batch registered the main README's own claims. All of its numbers
were current. The README itself was out of date in two ways:
- its "All host layers" row did not list the `facts` suite;
- it never mentioned the fact registry or the hook.

Fourteen mutations of these facts were all caught.

The fourth batch registered what the chat demo added: the client's flags, the
banner's width, the voice samples and the video link. `chat.py`'s usage
synopsis lacked `--timeout`; it has it now.

## Adding a fact

1. **When to add one.** Add an entry when a review, an audit or a
   `docs-audit` pass finds the same fact out of date in two places. Do not
   register facts preemptively.
2. **Pick the kind.** If the value can be derived from the code, it is a
   `value` or `interface` fact. Use `recorded` only for measurements.
3. **Find every quote.**
   `git grep -n '<the value>'` — then decide which hits are current statements
   (mentions) and which are dated records (leave those alone; note them in a
   comment, as `perf.bitstream_id` does).
4. **Anchor each locator** on the words around the value, not the value
   itself, so the locator survives a change of value.
5. **Run it.** `facts.py check ID -v`, then `facts.py fix ID --dry-run`.
6. **Big features.** Before you start, run `facts.py impact <file you will
   change>` to get the files that must move with it. Before you commit, run
   `facts.py changed`: it checks the touched facts and lists the files of
   those facts that you have not changed.
