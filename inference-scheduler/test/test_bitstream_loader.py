# Copyright 2026 Ivan Gubochkin
# SPDX-License-Identifier: Apache-2.0

"""upload_bitstream.py's board steps that guard the overlay stack (src/bitstream/):
a refused overlay removal stops the upload, and a UIO device that kept an old
interrupt is rebound — both against a fake SSH session."""

import shutil
import subprocess
import unittest
from pathlib import Path

from src.bitstream.board import parse_proc_interrupts, remove_overlay
from src.bitstream.loader import check_kernels_free, check_uio_irqs, dtbo_uio_irqs, dtbo_uio_nodes

REPO = Path(__file__).resolve().parents[2]
DTS = REPO / "dts" / "kv260" / "cormorant.dts"


class FakeSession:
    """exec() answers the commands the board steps send, from a small state."""

    def __init__(self, refuse=False, irqs_before=None, irqs_after=None, overlays=("pl",), uio=()):
        self.refuse = refuse
        self.uio = list(uio)
        self.irqs_before = irqs_before or {}
        self.irqs_after = irqs_after if irqs_after is not None else self.irqs_before
        self.overlays = list(overlays)
        self.removed = False
        self.rebound = []
        self.dmesg = ["[  100.000000] zocl-drm axi:zyxclmm_drm: client exits"]

    def exec(self, cmd, timeout=None):
        if cmd.startswith("dmesg"):
            n = int(cmd.split()[-1])
            return "\n".join(self.dmesg[-n:]) + "\n", "", 0
        if "rmdir" in cmd:
            name = cmd.split("/overlays/")[1].split("'")[0]
            if name not in self.overlays:
                return "absent\n", "", 0
            self.overlays.remove(name)
            self.removed = True
            if self.refuse:
                self.dmesg += ["[  101.000000] OF: overlay: node_overlaps_later_cs: #2 overlaps with #5 @/axi/afi0",
                               "[  101.000100] OF: overlay: overlay #2 is not topmost"]
            return "removed\n", "", 0
        if cmd.startswith("ls '/sys/kernel/config/device-tree/overlays'"):
            return "\n".join(self.overlays) + "\n", "", 0
        if cmd.startswith("for f in /sys/class/uio/uio*/name"):
            return "".join(f"/dev/uio{i} {name}\n" for i, name in enumerate(self.uio)), "", 0
        if cmd == "cat /proc/interrupts":
            irqs = self.irqs_after if self.rebound else self.irqs_before
            lines = [f" {60 + i}:  0  0  0  0  GICv2 {irq} Level  {name}" for i, (name, irq) in enumerate(irqs.items())]
            return "\n".join(lines) + "\n", "", 0
        if "uio_pdrv_genirq/unbind" in cmd:
            self.rebound.append(cmd.split("'")[1])
            return "", "", 0
        raise AssertionError(f"unexpected command: {cmd}")


class TestRemoveOverlay(unittest.TestCase):
    def test_absent(self):
        s = FakeSession(overlays=())
        remove_overlay(s, "pl")
        self.assertFalse(s.removed)

    def test_removed(self):
        s = FakeSession()
        remove_overlay(s, "pl")
        self.assertTrue(s.removed)

    def test_refused_stops(self):
        # configfs's rmdir succeeds, the kernel keeps the overlay: the upload must stop
        s = FakeSession(refuse=True, overlays=("pl", "rpu", "rpu-ipi"))
        with self.assertRaisesRegex(RuntimeError, "(?s)refused to remove overlay 'pl'.*not topmost.*Reboot"):
            remove_overlay(s, "pl")


class TestUioIrqs(unittest.TestCase):
    WANT = {"fabric_vecop@a0000000": 121, "fabric_matmul@a0010000": 122}

    def test_parse_proc_interrupts(self):
        text = (" 61:          1          0          0          0     GICv2 121 Level     fabric_vecop\n"
                " 14:          0          0          0          0     GICv2  67 Level     zynqmp-ipi\n"
                "IPI0:     12345     23456     34567     45678       Rescheduling interrupts\n")
        self.assertEqual(parse_proc_interrupts(text), {"fabric_vecop": 121, "zynqmp-ipi": 67})

    def test_already_right(self):
        s = FakeSession(irqs_before={"fabric_vecop": 121, "fabric_matmul": 122})
        check_uio_irqs(s, self.WANT)
        self.assertEqual(s.rebound, [])

    def test_stale_device_rebound(self):
        s = FakeSession(irqs_before={"fabric_vecop": 121, "fabric_matmul": 128},
                        irqs_after={"fabric_vecop": 121, "fabric_matmul": 122})
        check_uio_irqs(s, self.WANT)
        self.assertEqual(s.rebound, ["a0010000.fabric_matmul"])

    def test_still_stale_raises(self):
        s = FakeSession(irqs_before={"fabric_vecop": 121, "fabric_matmul": 128})
        with self.assertRaisesRegex(RuntimeError, "fabric_matmul@a0010000: GIC 128, overlay 122"):
            check_uio_irqs(s, self.WANT)


@unittest.skipUnless(shutil.which("dtc"), "dtc not installed")
class TestDtbo(unittest.TestCase):
    def test_overlay_irqs(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            dtbo = Path(d) / "pl.dtbo"
            subprocess.run(["dtc", "-q", "-I", "dts", "-O", "dtb", "-o", str(dtbo), str(DTS)], check=True)
            self.assertEqual(dtbo_uio_nodes(dtbo), ["fabric_vecop@a0000000", "fabric_matmul@a0010000",
                                                    "fabric_conv@a0020000", "fabric_pool@a0030000",
                                                    "fabric_rowops@a0040000"])
            # pl_ps_irq0[0..4] = SPI 89..93 = GIC 121..125 (facts.yaml board.irq_spi)
            self.assertEqual(list(dtbo_uio_irqs(dtbo).values()), [121, 122, 123, 124, 125])


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(shutil.which("dtc"), "needs dtc")
class TestKernelsFree(unittest.TestCase):
    """Another overlay with the same kernels stops the upload before the PL is
    reprogrammed (the 2026-10-18 reproduction: the kernel stacked the second
    overlay instead of refusing it)."""

    @classmethod
    def setUpClass(cls):
        import tempfile
        cls.tmp = tempfile.mkdtemp(prefix="kernels_free_")
        cls.dtbo = Path(cls.tmp) / "cormorant.dtbo"
        subprocess.run(["dtc", "-q", "-@", "-I", "dts", "-O", "dtb", "-o", str(cls.dtbo), str(DTS)], check=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_taken(self):
        s = FakeSession(overlays=("k26-starter-kits_image_1", "pl"), uio=("axi-pmon", "fabric_vecop", "fabric_conv"))
        with self.assertRaises(RuntimeError) as cm:
            check_kernels_free(s, self.dtbo, "design_cormorant")
        msg = str(cm.exception)
        self.assertIn("fabric_conv, fabric_vecop", msg)
        self.assertIn("(k26-starter-kits_image_1, pl)", msg)

    def test_free(self):
        s = FakeSession(overlays=("k26-starter-kits_image_1",), uio=("axi-pmon", "rpu_ipi"))
        check_kernels_free(s, self.dtbo, "pl")

    def test_upload_stops_before_upload(self):
        """Under another overlay name nothing reaches the board's /lib/firmware."""
        from unittest import mock
        from src.bitstream import loader
        s = FakeSession(overlays=("k26-starter-kits_image_1", "pl"), uio=("fabric_vecop", "fabric_conv"))
        with mock.patch.object(loader, "bit_to_bin", return_value=b"bin"), \
             mock.patch.object(loader, "parse_hwh_ps_params", return_value=("zynq_ultra_ps_e", {})), \
             mock.patch.object(loader.kv260, "axi_port_width_writes", return_value=[]), \
             mock.patch.object(loader, "parse_hwh_clocks", return_value={"pl0_mhz": 100, "kernel_mhz": 250}), \
             mock.patch.object(loader, "build_xclbin", return_value=b"xclbin"), \
             mock.patch.object(loader, "upload_file") as up:
            with self.assertRaises(RuntimeError) as cm:
                loader.upload_bitstream(s, Path("x.bit"), Path("x.hwh"), self.dtbo, "design_cormorant")
        self.assertIn("another overlay provides them", str(cm.exception))
        up.assert_not_called()
        self.assertEqual(s.overlays, ["k26-starter-kits_image_1", "pl"])
