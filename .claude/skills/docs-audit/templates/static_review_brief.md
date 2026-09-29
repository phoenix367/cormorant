# Brief: review the docs-audit diff

Repository `{{REPO}}`; read its CLAUDE.md first.  Several agents compared the
docs with the code and fixed stale facts.  The combined change:
`git diff -- '*.md'`, plus `git -C hw/<submodule> diff` for the hw/ submodules.
{{PREEXISTING}}

For EVERY changed claim (each removed / added pair):

1. **Against the code.**  Find the code location that proves the new text.
   Verdict: *confirmed* (`file:line`), *wrong* (correct it to what the code
   says), or *unprovable* (restore the old text unless the old text is
   provably wrong too — then say so).
2. **Current state only.**  An edit inside dated history (plan sections,
   optimisation-log entries, before / after measurements, dated transcripts)
   is reverted; in `doc/plans/` only the status line may change.
3. **Nothing else.**  `git diff --stat -- ':!*.md'` must be empty (docs-only
   audit), and no hunk may reword or restructure beyond the fact it fixes.
4. **Consistency.**  The same fact (test counts, paths, flag names, defaults)
   must read the same in every doc: `grep -rn` each replaced value across all
   `*.md` files and list the places still holding it.

Measured counts ({{DATE}}):

```
{{COUNTS}}
```

Final message: a table `| file:line | claim | verdict | evidence | action |`,
then the leftovers of old values elsewhere, then everything you reverted or
corrected.
