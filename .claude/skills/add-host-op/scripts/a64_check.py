#!/usr/bin/env python3
"""a64_check.py — compile a generated project's src/inference.c for the
board's CPU (aarch64) on the dev PC and inspect the object code.

The host emulation (test/host_emu.py, llm_host_emu.py, vlm_host_emu.py)
builds with the x86 compiler, so every ``#if defined(__aarch64__) &&
defined(__ARM_NEON)`` branch of the host helpers is never compiled there —
this is the only pre-board check of that code.  It also proves that no FMA
was contracted in (the host section's ``#pragma GCC optimize
("fp-contract=off")``; aarch64 GCC fuses a*b + c by default in GNU C modes,
which breaks bit-exactness with the numpy reference).  It does NOT prove the
NEON path bit-exact: that needs the board (or a harness run on it).

  PROJECT   a generated project directory (include/inference.h,
            src/inference.c): inference_scheduler.py --out-dir, a
            host_emu.build_and_run workdir, demo/chat/build/llm_project*

Driver headers are the host emulation's (test/host_emu.py), so the project
needs no --driver-dir.  Prints the NEON (vector double) instruction count and
every function that contains an FMA; exit 1 on a compile error or an FMA.

usage: inference-scheduler/.venv/bin/python .claude/skills/add-host-op/scripts/a64_check.py PROJECT
           [--cc aarch64-linux-gnu-gcc] [--keep DIR]
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
SCHED = os.path.join(REPO, "inference-scheduler")
sys.path.insert(0, SCHED)
sys.path.insert(0, os.path.join(SCHED, "test"))

FMA = re.compile(r"\s(fmadd|fmsub|fnmadd|fnmsub|fmla|fmls)\s")
NEON_F64 = re.compile(r"\s(f\w+)\s+v\d+\.2d")          # double-precision vector ops


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("project")
    ap.add_argument("--cc", default="aarch64-linux-gnu-gcc",
                    help="aarch64 C compiler (Vitis ships one: <Vitis>/gnu/aarch64/lin/aarch64-linux/bin)")
    ap.add_argument("--keep", default=None, help="write the object file here instead of a temp dir")
    args = ap.parse_args(argv)
    cc = shutil.which(args.cc)
    if cc is None:
        print(f"error: {args.cc} not found (source the Vitis settings64.sh or pass --cc)")
        return 2
    objdump = shutil.which(os.path.basename(cc).replace("gcc", "objdump")) or \
        os.path.join(os.path.dirname(cc), os.path.basename(cc).replace("gcc", "objdump"))
    proj = os.path.abspath(args.project)
    src = os.path.join(proj, "src", "inference.c")
    if not os.path.isfile(src):
        print(f"error: {src} not found")
        return 2

    import host_emu                                              # noqa: E402
    from src._conv_hw_config import CONV_TILE_IC                 # noqa: E402
    from src._matmul_hw_config import MATMUL_TILE_M              # noqa: E402

    with tempfile.TemporaryDirectory() as td:
        emu = os.path.join(td, "emu")
        os.makedirs(emu)
        for name, text in (("emu_common.h", host_emu._COMMON), ("xvectoropkernel.h", host_emu._VOP),
                           ("xmatmulkernel.h", host_emu._MM), ("xconvkernel.h", host_emu._CONV)):
            with open(os.path.join(emu, name), "w") as f:
                f.write(text)
        inc = [os.path.join(proj, "include")]
        if not os.path.isfile(os.path.join(inc[0], "inference_prof.h")):
            inc.append(os.path.join(SCHED, "runtime"))
        out_dir = args.keep or td
        os.makedirs(out_dir, exist_ok=True)
        obj = os.path.join(out_dir, "inference_a64.o")
        cmd = [cc, "-std=gnu99", "-O2", "-Wall", "-Wextra", "-Werror", "-Wno-unused-function",
               "-Wno-error=parentheses", "-pthread", f"-DEMU_TILE_M={MATMUL_TILE_M}",
               f"-DEMU_CONV_TILE_IC={CONV_TILE_IC}",
               *[a for d in inc for a in ("-I", d)], "-I", emu, "-c", src, "-o", obj]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            print("COMPILE FAILED (aarch64, -Werror)\n" + r.stderr[-6000:])
            return 1
        dis = subprocess.run([objdump, "-d", obj], capture_output=True, text=True, check=True).stdout

    fn, fma, neon = "?", Counter(), Counter()
    for line in dis.splitlines():
        m = re.match(r"^[0-9a-f]+ <(.+)>:$", line)
        if m:
            fn = m.group(1)
            continue
        if FMA.search(line):
            fma[fn] += 1
        if NEON_F64.search(line):
            neon[fn] += 1
    print(f"aarch64 compile OK ({os.path.basename(cc)}, -O2 -Werror)")
    print(f"NEON double-vector instructions: {sum(neon.values())} in {len(neon)} function(s)"
          + (": " + ", ".join(f"{k} {v}" for k, v in neon.most_common(8)) if neon else ""))
    if fma:
        print("FMA CONTRACTED (breaks bit-exactness with the numpy reference) in: "
              + ", ".join(f"{k} x{v}" for k, v in fma.most_common()))
        return 1
    print("no FMA instructions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
