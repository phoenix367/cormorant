# GAPS — documentation gaps found while reproducing cormorant from a fresh clone

Clone: `{{CLONE}}` (HEAD {{CLONE_HEAD}}, local overlay commits over origin {{ORIGIN_HEAD}}).
Date: {{DATE}}.  Paths are relative to the clone root.  Evidence (logs) in `{{BASE}}/logs/`.

Severity:
- **blocker** — cannot proceed without reading the source or guessing (a missing
  file, a step that cannot work as written, a command that would touch production).
- **major** — fails as documented and needs an undocumented step (install,
  flag, config edit) to get through.
- **minor** — a wrong number / name / path / count, a small omission, a step
  whose output differs from the doc in a way a user would notice.
- **nit** — cosmetic: wording, ordering, output formatting, expected warnings
  not mentioned.

Not gaps (list them in RUNLOG instead): host-specific trouble (the slow extra pip
index, Vitis already on PATH from the login profile), run-to-run noise
(latency within ~3 %), anything caused by an instruction of the brief.

## Index (by severity)

| Severity | Gaps |
|---|---|
| blocker | G1 (one-line title) |
| major | |
| minor | |
| nit | |

## G1 — blocker — <one-line title: what fails, as the user sees it>
- Doc: `<file>:<line>` (and every other place that says the same thing), quoted
  or paraphrased: what the doc says to do / claims.
- Actual: what happened — the exact command, the exact error or output line,
  rc, the log file (`logs/NN_name.log`).  For a count or number: doc value vs
  measured value.
- Workaround: what got you through (the command), and what you had to read to
  find it (`<source file>:<line>`) — or "none, path abandoned".
- Fix: the smallest change that would have made the doc work — in the doc
  (exact replacement text) or in the code / repo (file, what to change).

## G2 — major — …
