# Brief: static docs audit — area {{AREA}}

Repository `{{REPO}}`; read its CLAUDE.md first for the layout.  Your files —
edit only these:

{{FILES}}

**Rule: compare every factual claim in your files with the current code and
repository, and fix the stale ones in place.**  Factual claims: paths, file /
function / class / script / target names, CLI flags and their defaults, config
keys and defaults, register offsets, port widths, bounds, counts (tests,
modules, fixtures, models, cases), numbers stated as current.

Counts measured on {{DATE}} — use these, do not re-run the suites:

```
{{COUNTS}}
```

- **Evidence first.**  For every edit know the code location that proves it
  (`file:line`).  If you cannot prove it, leave the text and list the claim as
  unverified.
- **History stays.**  Dated results and logs — plan sections, optimisation-log
  entries ("§2.31 …", "on 2026-09-25 …"), before / after tables, sample
  transcripts labelled with a date — describe the past and stay as written,
  even where the code has moved on.  Current-state text (references, quick
  starts, tables of current behaviour, option lists, plan status lines) must
  match the code.  In `doc/plans/` only the status line at the top changes.
- **Facts, not style.**  No rewording, restructuring or new sections; keep the
  doc's voice and line wrapping.  A fact a reader needs and cannot find (a new
  flag, a new file, a new default) gets one line where a reader looks for it.
- **Docs only.**  Do not touch code, tests, configs or other areas' files.  If
  a doc is right and the CODE is wrong (a bug, a missing file), do not bend the
  doc to the bug — report it.
- **Where to look.**  Kernels: `kernels/<k>/kernel/*.cpp`, `include/*.h`,
  `kernels/<k>/CMakeLists.txt`, `platforms/kv260.json`, the generated register
  maps `build*/kernels/<k>/kv260/**/drivers/*/src/x*_hw.h` (MatmulKernel and
  VectorOPKernel: `kernels/{matmul,vectorop}_rtl/rtl/*.sv`,
  `build*/kernels/{matmul,vectorop}_rtl/driver/`).  Scheduler:
  `inference-scheduler/src/`, `inference_scheduler.py --help`, `test/`.  Build:
  the CMakeLists, `make help` in a configured build dir.  Demos: the scripts'
  `--help` and the `*.json.example` files.  History of a fact: `git log -S`.
- **Cheap and safe commands only**: `--help`, `grep`, `git log`, `python -c`
  imports.  No syntheses, Vivado, board commands or full test suites.
- **The fact registry** (`facts.yaml`, `tools/facts/README.md`;
  `python3 tools/facts/facts.py list` / `impact FILE`) checks values that
  several files repeat.  Do not edit `facts.yaml` — report candidates: a value
  your files state that is stale here and also stated elsewhere, or quoted in
  two or more places and derivable from code (a count, a path, a name, a
  default, a bound, a measured result with its source).  For each: the value,
  its source of truth (file + how to read it), every place that quotes it,
  and whether a registered fact already covers it (then: which locator is
  missing).

Final message: a table `| file:line | was | now | evidence (code file:line) |`
for every edit; then "unverified" (claims you could not check), "code
suspects" (the doc is right, the code is not) and "registry candidates".
