"""The board's CMA clean-up helper (src/remote/cma.py): what it runs and how it
reads the board's answer — on a fake session, no board."""

import unittest

from src.remote import cma


class _FakeSession:
    def __init__(self, states):
        self.states = list(states)
        self.commands = []

    def exec(self, command, timeout=120):
        self.commands.append(command)
        if "meminfo" in command:
            return self.states.pop(0), "", 0
        return "", "", 0


BEFORE = ("total_kb=1024000\nfree_kb=880636\nmaxchunk_kb=300000\n"
          "holder=4242 /root/b2_chat/llm_project/build/llm_bench -i prompts.bin\n")
AFTER = "total_kb=1024000\nfree_kb=1014296\nmaxchunk_kb=1013760\n"


class TestCma(unittest.TestCase):

    def test_parse(self):
        st = cma.parse_state(BEFORE)
        self.assertEqual((st["total_kb"], st["free_kb"], st["maxchunk_kb"]), (1024000, 880636, 300000))
        self.assertEqual(st["holders"], ["4242 /root/b2_chat/llm_project/build/llm_bench -i prompts.bin"])
        self.assertIsNone(cma.parse_state("total_kb=1\nfree_kb=1\n")["maxchunk_kb"])   # no debugfs

    def test_clean_runs_drop_and_compact_between_two_reports(self):
        s = _FakeSession([BEFORE, AFTER])
        r = cma.clean_cma(s)
        self.assertEqual(len(s.commands), 3)
        self.assertIn("drop_caches", s.commands[1])
        self.assertIn("compact_memory", s.commands[1])
        self.assertEqual(r["before"]["free_kb"], 880636)
        self.assertEqual(r["after"]["maxchunk_kb"], 1013760)
        self.assertIn("largest free chunk 990 MB", cma.describe(r["after"]))

    def test_sudo_wraps_the_whole_command(self):
        s = _FakeSession([BEFORE, AFTER])
        cma.clean_cma(s, sudo="sudo ")
        self.assertTrue(all(c.startswith("sudo sh -c '") for c in s.commands))


if __name__ == "__main__":
    unittest.main()
