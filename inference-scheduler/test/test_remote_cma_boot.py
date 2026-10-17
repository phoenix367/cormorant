"""The CMA boot-size tool (src/remote/cma_boot.py): the size, the boot script
edit (run with the host's sed on a copy of the board script's layout), the
board's report — no board."""

import os
import subprocess
import tempfile
import unittest

from src.remote import cma_boot

# the Kria Ubuntu image's bootscr.zynqmp, the part that sets cma
SCRIPT = """\t\techo "Configuring the cma value based on the board type"
\t\tif test $kria = "KD"; then
\t\t\tcma="512M"
\t\telif  test -n $kria; then
\t\t\tcma="1000M"
\t\telif setexpr model_test gsub '.*104.*' 104; then
\t\t\tcma="700M"
\t\telif setexpr model_test gsub '.*102.*' 102 || setexpr model_test gsub '.*111.*' 111; then
\t\t\tcma="1000M"
\t\tfi
"""


class TestCmaBoot(unittest.TestCase):

    def test_size(self):
        self.assertEqual([cma_boot.parse_size(s) for s in ("1400M", "1g", " 512M ")], [1400, 1024, 512])
        for bad in ("1400", "1.5G", "1400MB", ""):
            with self.assertRaises(ValueError):
                cma_boot.parse_size(bad)

    def test_edit_changes_the_kria_branch_only(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "bootscr.zynqmp")
            with open(p, "w") as f:
                f.write(SCRIPT)
            subprocess.run(cma_boot.edit_command(p, "1400M"), shell=True, check=True)
            text = open(p).read()
        self.assertEqual(cma_boot.kria_value(text), "1400M")
        self.assertEqual(text, SCRIPT.replace('$kria; then\n\t\t\tcma="1000M"', '$kria; then\n\t\t\tcma="1400M"'))
        self.assertEqual(text.count('cma="1000M"'), 1)                    # the ZCU102 branch kept
        self.assertEqual(cma_boot.kria_value(SCRIPT), "1000M")
        self.assertIsNone(cma_boot.kria_value("cma=1000M"))

    def test_state(self):
        st = cma_boot.parse_state(
            "cmdline=root=LABEL=writable rootwait cma=1400M quiet splash\ntotal_kb=1433600\nfree_kb=1400000\n"
            "reserved=cma: Reserved 1400 MiB at 0x000000001e400000\nunit=kv260-chat active\n"
            "unit=kv260-miranda inactive\n")
        self.assertEqual((st["cmdline_cma"], st["total_kb"], st["active"]), ("1400M", 1433600, ["kv260-chat"]))
        self.assertIn("CmaTotal 1400 MiB", cma_boot.describe(st))
        self.assertIsNone(cma_boot.parse_state("cmdline=root=LABEL=writable\n")["cmdline_cma"])


if __name__ == "__main__":
    unittest.main()
