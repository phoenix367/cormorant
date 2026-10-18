# Copyright 2026 Ivan Gubochkin
# SPDX-License-Identifier: Apache-2.0

"""The runtime files every generated project carries, copied verbatim from
inference-scheduler/runtime/ — one list for both project writers (the CLI,
inference_scheduler.py, and multi-entry projects, multi.write_project), so a
file the generated CMakeLists.txt names cannot be missing from one of them."""
from __future__ import annotations

import os
import shutil
from typing import Optional

RUNTIME_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                           "runtime")

# (source under runtime/, destination in the project)
RUNTIME_FILES = (
    # the per-layer profiler and its DDR-traffic counters (-DINFERENCE_PROFILING=ON)
    ("inference_prof.h", "include/inference_prof.h"),
    ("inference_prof.c", "src/inference_prof.c"),
    ("inference_ddr.h", "include/inference_ddr.h"),
    ("inference_ddr.c", "src/inference_ddr.c"),
    ("inference_ddr_backend.h", "src/inference_ddr_backend.h"),
    ("ddr/zuplus_apm.c", "src/ddr/zuplus_apm.c"),          # one .c per platform under src/ddr/
    # the kernels driven from the Cortex-R5 (RPU_PLAN §9): the recorder in every Linux
    # library, the protocol with the firmware (runtime/rpu/, one build per bitstream)
    # and rpu_host.c, the gate of a single-entry project on the R5 (make rpu_host)
    # the A53's waits on the kernels' interrupts (UIO) while it drives them
    ("inference_irq.h", "include/inference_irq.h"),
    ("inference_irq.c", "src/inference_irq.c"),
    ("inference_rpu.h", "include/inference_rpu.h"),
    ("inference_rpu.c", "src/inference_rpu.c"),
    ("rpu/rpu_ctrl.h", "include/rpu_ctrl.h"),
    ("rpu/rpu_host.c", "rpu/rpu_host.c"),
)


def copy_runtime(out_dir: str, runtime_dir: Optional[str] = None) -> None:
    rt = runtime_dir or RUNTIME_DIR
    for src_name, rel in RUNTIME_FILES:
        dst = os.path.join(out_dir, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(os.path.join(rt, src_name), dst)


__all__ = ("RUNTIME_DIR", "RUNTIME_FILES", "copy_runtime")
