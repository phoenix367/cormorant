"""Tests for the kernel registry (src/kernels.py)."""

import os
import unittest

from src.kernels import KERNEL_REGISTRY
class TestRpuLanes(unittest.TestCase):
    """The Cortex-R5 recorder copies each kernel's argument registers from
    offset 0x10 (runtime/inference_rpu.c): the word count must end exactly at
    the register map's end (the control blocks decode 7 or 8 address bits, so
    a write past it would alias onto a register), and the lanes are the
    firmware's (runtime/rpu/rpu_ctrl.h RPU_LANE_*)."""

    _GEN = {   # kernel -> its driver generator's register table
        "VectorOPKernel": "kernels/vectorop_rtl/scripts/gen_driver.py",
        "MatmulKernel":   "kernels/matmul_rtl/scripts/gen_driver.py",
        "ConvKernel":     "kernels/conv_rtl/scripts/gen_driver.py",
        "PoolKernel":     "kernels/pool_rtl/scripts/gen_driver.py",
        "RowOpsKernel":   "kernels/rowops_rtl/scripts/gen_driver.py",
    }

    def test_reg_words_match_driver_generators(self):
        import re
        root = os.path.join(os.path.dirname(__file__), "..", "..")
        for kd in KERNEL_REGISTRY.values():
            src = open(os.path.join(root, self._GEN[kd.name])).read()
            regs = re.findall(r'\(\s*"[a-z_0-9]+",\s*(0x[0-9a-fA-F]+),\s*(\d+)', src)
            self.assertTrue(regs, kd.name)
            end = max(int(o, 16) + (8 if int(w) > 32 else 4) for o, w in regs)
            self.assertEqual(kd.rpu_reg_words, (end - 0x10) // 4, kd.name)

    def test_lanes_are_the_registry_order(self):
        lanes = [kd.rpu_lane for kd in KERNEL_REGISTRY.values()]
        self.assertEqual(lanes, list(range(len(lanes))))
        header = open(os.path.join(os.path.dirname(__file__), "..", "runtime", "rpu", "rpu_ctrl.h")).read()
        for kd in KERNEL_REGISTRY.values():
            name = {"PoolKernel": "POOL"}.get(kd.name, kd.name.replace("Kernel", "").upper())
            self.assertIn(f"#define RPU_LANE_{name}  ", header.replace("   ", "  "), kd.name)
        self.assertIn(f"#define RPU_LANES          {len(lanes)}u", header)

    def test_control_block_layout(self):
        """rpu_ctrl.h compiles on the host with its 512-byte layout assert: the
        A53 library and the R5 firmware share the block in the BTCM."""
        import shutil
        import subprocess
        import tempfile
        cc = shutil.which("cc") or shutil.which("gcc")
        if not cc:
            self.skipTest("no C compiler")
        hdr = os.path.join(os.path.dirname(__file__), "..", "runtime", "rpu")
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "t.c")
            with open(src, "w") as f:
                f.write('#include <stddef.h>\n#include "rpu_ctrl.h"\n'
                        'int main(void) { return (int)offsetof(struct rpu_ctrl, gic) == 152 ? 0 : 1; }\n')
            r = subprocess.run([cc, "-std=gnu99", "-Wall", "-Werror", "-I", hdr, src, "-o",
                                os.path.join(td, "t")], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(subprocess.run([os.path.join(td, "t")]).returncode, 0)
