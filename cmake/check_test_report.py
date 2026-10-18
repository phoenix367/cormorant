#!/usr/bin/env python3
# Copyright 2025 GradeBuilder SL
# SPDX-License-Identifier: Apache-2.0

"""Verify a cormorant_test_stand JSON scoreboard report.

Used by the per-kernel `behavior_test_<k>` targets in the top-level
CMakeLists.txt.  Exits:
    0 — report exists, `summary.all_passed` is true
    1 — report exists, at least one test failed
    2 — report missing or unparseable (treat as test infra failure)

Output is one line per invocation, prefixed with `[ck]` so it stands out
from the surrounding Vivado / make logs.
"""
import json
import sys


def main(path):
    try:
        with open(path) as f:
            report = json.load(f)
    except FileNotFoundError:
        print(f"[ck] ERROR: report not written: {path}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"[ck] ERROR: report parse failed ({path}): {exc}",
              file=sys.stderr)
        return 2

    summary = report.get("summary", {})
    total   = summary.get("total", "?")
    passed  = summary.get("passed", "?")
    failed  = summary.get("failed", "?")
    all_ok  = bool(summary.get("all_passed", False))
    kernel  = report.get("kernel", "?")

    if all_ok:
        print(f"[ck] {kernel}: PASS  ({passed}/{total})  {path}")
        return 0
    print(f"[ck] {kernel}: FAIL  ({passed}/{total}, {failed} failed)  {path}",
          file=sys.stderr)
    return 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: check_test_report.py <report.json>", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
