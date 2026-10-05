"""Tests for src.kernels — KernelDesc registry and derived helpers."""

import unittest

from src.kernels import (
    KernelDesc,
    KERNEL_REGISTRY,
    all_driver_files,
    mixed_driver_readme,
)


_EXPECTED_KERNELS = ("VectorOPKernel", "MatmulKernel", "ConvKernel", "PoolKernel")


# ---------------------------------------------------------------- #
# Registry contents and invariants                                  #
# ---------------------------------------------------------------- #

class TestKernelRegistry(unittest.TestCase):
    def test_contains_all_four_expected_kernels(self):
        for name in _EXPECTED_KERNELS:
            self.assertIn(name, KERNEL_REGISTRY)
        self.assertEqual(len(KERNEL_REGISTRY), len(_EXPECTED_KERNELS))

    def test_each_entry_is_a_kerneldesc_with_matching_name(self):
        for name, kd in KERNEL_REGISTRY.items():
            self.assertIsInstance(kd, KernelDesc)
            self.assertEqual(kd.name, name)

    def test_driver_prefixes_are_unique(self):
        prefixes = [kd.driver_prefix for kd in KERNEL_REGISTRY.values()]
        self.assertEqual(len(prefixes), len(set(prefixes)),
                         f"driver_prefix collision: {prefixes}")

    def test_axi_bases_unique_and_64kb_aligned(self):
        bases = [kd.axi_base for kd in KERNEL_REGISTRY.values()]
        self.assertEqual(len(bases), len(set(bases)),
                         f"axi_base collision: {[hex(b) for b in bases]}")
        for b in bases:
            self.assertEqual(b & 0xFFFF, 0,
                             f"axi_base 0x{b:08X} not 64 KB-aligned")

    def test_uio_default_ends_with_underscore_zero(self):
        # Convention: UIO device-tree label is "<KernelName>_0".
        for kd in KERNEL_REGISTRY.values():
            self.assertTrue(kd.uio_default.endswith("_0"),
                            f"{kd.name}.uio_default = {kd.uio_default!r}")

    def test_driver_files_nonempty(self):
        for kd in KERNEL_REGISTRY.values():
            self.assertGreater(len(kd.driver_files), 0,
                               f"{kd.name} has no driver_files")


# ---------------------------------------------------------------- #
# KernelDesc derived @property values                               #
# ---------------------------------------------------------------- #

class TestKernelDescDerivedProperties(unittest.TestCase):
    def test_c_var_format(self):
        # 's_' + lowercase(name)
        self.assertEqual(KERNEL_REGISTRY["VectorOPKernel"].c_var,
                         "s_vectoropkernel")
        self.assertEqual(KERNEL_REGISTRY["ConvKernel"].c_var,
                         "s_convkernel")

    def test_init_param_format(self):
        self.assertEqual(KERNEL_REGISTRY["MatmulKernel"].init_param,
                         "matmulkernel_instance")

    def test_instance_macro_format(self):
        self.assertEqual(KERNEL_REGISTRY["ConvKernel"].instance_macro,
                         "INFERENCE_CONVKERNEL_INSTANCE")
        self.assertEqual(KERNEL_REGISTRY["PoolKernel"].instance_macro,
                         "INFERENCE_POOLKERNEL_INSTANCE")


# ---------------------------------------------------------------- #
# all_driver_files — deduped union across all kernels               #
# ---------------------------------------------------------------- #

class TestAllDriverFiles(unittest.TestCase):
    def test_includes_every_kernel_driver_file(self):
        files = all_driver_files()
        for kd in KERNEL_REGISTRY.values():
            for f in kd.driver_files:
                self.assertIn(f, files)

    def test_no_duplicate_filenames(self):
        files = all_driver_files()
        self.assertEqual(len(files), len(set(files)),
                         f"duplicate driver-file names: {files}")

    def test_returns_a_list(self):
        self.assertIsInstance(all_driver_files(), list)


# ---------------------------------------------------------------- #
# mixed_driver_readme — multi-kernel README composer                #
# ---------------------------------------------------------------- #

class TestMixedDriverReadme(unittest.TestCase):
    def test_section_per_named_kernel(self):
        names  = ["VectorOPKernel", "ConvKernel"]
        readme = mixed_driver_readme(names)
        for n in names:
            self.assertIn(f"## {n} driver", readme)

    def test_includes_hex_axi_base(self):
        # VectorOP = 0xA0000000; Conv = 0xA0020000.
        readme = mixed_driver_readme(["VectorOPKernel", "ConvKernel"])
        self.assertIn("0xA0000000", readme)
        self.assertIn("0xA0020000", readme)

    def test_excludes_unnamed_kernels(self):
        readme = mixed_driver_readme(["VectorOPKernel"])
        self.assertNotIn("## MatmulKernel driver", readme)
        self.assertNotIn("## ConvKernel driver",   readme)
        self.assertNotIn("## PoolKernel driver",   readme)

    def test_lists_driver_files(self):
        readme = mixed_driver_readme(["MatmulKernel"])
        for f in KERNEL_REGISTRY["MatmulKernel"].driver_files:
            self.assertIn(f, readme)

    def test_driver_sources(self):
        # the SystemVerilog kernels' drivers come from make driver_<k>_rtl,
        # ConvKernel's from its HLS synthesis
        readme = mixed_driver_readme(["VectorOPKernel", "MatmulKernel", "ConvKernel", "PoolKernel"])
        for target in ("driver_vectorop_rtl", "driver_matmul_rtl", "driver_pool_rtl"):
            self.assertIn(f"make {target}", readme)
        self.assertIn("Vitis HLS synthesis output for ConvKernel", readme)


if __name__ == "__main__":
    unittest.main()
