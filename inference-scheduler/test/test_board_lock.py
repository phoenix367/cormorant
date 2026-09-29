"""The shared board lock (src/remote/lock.py): one flock per board, taken by
every board tool, re-entrant for a tool started by a lock holder."""

import os
import subprocess
import sys
import tempfile
import textwrap
import unittest

from src.remote.lock import LOCK_ENV, board_lock, default_lock_path, hold_board_lock, lock_path

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Try to take the lock without blocking: exit 0 when free, 3 when held elsewhere.
_PROBE = textwrap.dedent("""
    import fcntl, sys
    f = open(sys.argv[1], "a")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit(3)
""")

# A child tool: takes the lock through hold_board_lock (it would block forever
# if it did not see that its parent holds it).
_CHILD = textwrap.dedent("""
    import sys
    sys.path.insert(0, sys.argv[2])
    from src.remote.lock import hold_board_lock
    print(hold_board_lock({"board_lock": sys.argv[1]}))
""")


def _probe(path, env=None):
    return subprocess.run([sys.executable, "-c", _PROBE, path], env=env, timeout=30).returncode


class TestBoardLock(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "board.lock")
        self._env = os.environ.pop(LOCK_ENV, None)

    def tearDown(self):
        os.environ.pop(LOCK_ENV, None)
        if self._env is not None:
            os.environ[LOCK_ENV] = self._env
        self.tmp.cleanup()

    def test_paths(self):
        self.assertEqual(default_lock_path("192.168.100.8"), "/tmp/kv260-board-192.168.100.8.lock")
        self.assertEqual(default_lock_path("kv260 a/b"), "/tmp/kv260-board-kv260_a_b.lock")
        cfg = {"ssh": {"host": "10.0.0.2"}}
        self.assertEqual(lock_path(cfg), "/tmp/kv260-board-10.0.0.2.lock")
        self.assertEqual(lock_path({**cfg, "board_lock": None}), "/tmp/kv260-board-10.0.0.2.lock")
        self.assertEqual(lock_path({**cfg, "board_lock": "~/x.lock"}),
                         os.path.expanduser("~/x.lock"))
        self.assertEqual(lock_path(cfg, override="/tmp/o.lock"), "/tmp/o.lock")
        for off in (False, "none", "off", "false"):
            self.assertIsNone(lock_path({**cfg, "board_lock": off}))

    def test_exclusive_and_released(self):
        self.assertEqual(_probe(self.path), 0)
        with board_lock(self.path):
            self.assertEqual(os.environ.get(LOCK_ENV), self.path)
            self.assertEqual(_probe(self.path), 3)          # another job would wait
        self.assertNotIn(LOCK_ENV, os.environ)
        self.assertEqual(_probe(self.path), 0)

    def test_child_tool_does_not_wait_for_its_parent(self):
        with board_lock(self.path):
            r = subprocess.run([sys.executable, "-c", _CHILD, self.path, _ROOT],
                               capture_output=True, text=True, timeout=30)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(r.stdout.strip(), self.path)
            # an unrelated process (no marker in its environment) is still excluded
            env = {k: v for k, v in os.environ.items() if k != LOCK_ENV}
            self.assertEqual(_probe(self.path, env), 3)

    def test_nested_in_process(self):
        with board_lock(self.path):
            with board_lock(self.path):                     # no self-deadlock
                self.assertEqual(_probe(self.path), 3)
            self.assertEqual(_probe(self.path), 3)          # still held by the outer block
        self.assertEqual(_probe(self.path), 0)

    def test_hold_for_process_lifetime(self):
        code = textwrap.dedent(f"""
            import sys, time
            sys.path.insert(0, {_ROOT!r})
            from src.remote.lock import hold_board_lock
            hold_board_lock({{"board_lock": {self.path!r}}})
            print("held", flush=True)
            time.sleep(30)
        """)
        p = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
        try:
            self.assertEqual(p.stdout.readline().strip(), "held")
            self.assertEqual(_probe(self.path), 3)
        finally:
            p.kill()
            p.wait(timeout=30)
        self.assertEqual(_probe(self.path), 0)             # released when the process exits

    def test_disabled(self):
        self.assertIsNone(hold_board_lock({"board_lock": False, "ssh": {"host": "x"}}))
        self.assertNotIn(LOCK_ENV, os.environ)
        with board_lock(None):
            self.assertNotIn(LOCK_ENV, os.environ)


if __name__ == "__main__":
    unittest.main()
