"""The size of the KV260's CMA area: the kernel's cma= boot argument.

Every FPGA library takes its DMA pool as one contiguous CMA buffer.  The
board's U-Boot script (/boot/firmware/boot.scr.uimg, made by flash-kernel from
/etc/flash-kernel/bootscript/bootscr.zynqmp and .kria) sets cma= per board
type; for Kria boards it says cma="1000M" in Ubuntu's image.  At 1000M,
SmolLM2-360M (a 795 MB pool) does not fit beside Moonshine base and Piper, so
a voice session swaps models every turn; cma="1400M" holds SmolLM2-360M,
SmolLM2-135M, Moonshine base and Piper at once (2026-10-17).  CMA pages serve
ordinary movable memory (page cache, processes) while no pool holds them.

    .venv/bin/python -m src.remote.cma_boot --config CFG                       # report
    .venv/bin/python -m src.remote.cma_boot --config CFG --set 1400M           # the boot script only
    .venv/bin/python -m src.remote.cma_boot --config CFG --set 1400M --reboot  # and reboot, check

--set backs up the boot image and both script sources to /root/cma_backup/
(<name>.<time>), rewrites the Kria branch's cma="..." in the sources, runs
flash-kernel and checks the new image's Kria branch.  --reboot (also alone,
after --set) refuses while the chat server or Miranda runs (their deploy.py
--stop), reboots, waits for ssh and checks /proc/cmdline and CmaTotal.  After
a reboot the PL holds the starter-kit overlay: load the bitstream again
(upload_bitstream.py) before the chat server (demo/chat/deploy.py).  The CMA
area must fit in the low 2 GB of DDR beside the kernel: at most 1600M unless
--force.  The board lock is held (src/remote/lock.py) unless --no-lock.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from typing import Dict, List, Optional

BOARD = "1400M"                                    # what the project's KV260 runs (Ubuntu's image: 1000M)
SCRIPTS = ("/etc/flash-kernel/bootscript/bootscr.zynqmp", "/etc/flash-kernel/bootscript/bootscr.zynqmp.kria")
IMAGE = "/boot/firmware/boot.scr.uimg"
BACKUP_DIR = "/root/cma_backup"
UNITS = ("kv260-chat", "kv260-miranda")             # services that hold FPGA pools
MAX_MB = 1600                                      # beyond: --force
KRIA_BRANCH = r"elif[[:space:]]+test -n \$kria; then"


def parse_size(s: str) -> int:
    """'1400M' / '1G' -> MiB."""
    m = re.fullmatch(r"(\d+)([MG])", s.strip().upper())
    if not m:
        raise ValueError(f"a size like 1400M or 2G expected, not {s!r}")
    return int(m.group(1)) * (1024 if m.group(2) == "G" else 1)


def edit_command(path: str, size: str) -> str:
    """sed: the Kria branch's cma="..." (the line after `elif test -n $kria; then`) := size."""
    return f"sed -E -i '/{KRIA_BRANCH}/{{n;s/cma=\"[0-9]+[MG]\"/cma=\"{size}\"/}}' {path}"


def kria_value(script: str) -> Optional[str]:
    """The cma value of the Kria branch of a boot script's text, else None."""
    m = re.search(r"elif\s+test -n \$kria; then\s*\n\s*cma=\"(\d+[MG])\"", script)
    return m.group(1) if m else None


STATE_SCRIPT = r"""
echo "cmdline=$(cat /proc/cmdline)"
awk '/^CmaTotal/{print "total_kb=" $2} /^CmaFree/{print "free_kb=" $2}' /proc/meminfo
dmesg 2>/dev/null | grep -o 'cma: Reserved [0-9]* MiB at 0x[0-9a-f]*' | tail -1 | sed 's/^/reserved=/'
for u in UNITS; do echo "unit=$u $(systemctl is-active $u 2>/dev/null)"; done
true
"""


def parse_state(text: str) -> Dict:
    """{cmdline_cma, total_kb, free_kb, reserved, active: [unit]}."""
    st: Dict = {"cmdline_cma": None, "total_kb": 0, "free_kb": 0, "reserved": None, "active": []}
    for line in text.splitlines():
        key, _, val = line.strip().partition("=")
        if key == "cmdline":
            m = re.search(r"\bcma=(\S+)", val)
            st["cmdline_cma"] = m.group(1) if m else None
        elif key in ("total_kb", "free_kb"):
            st[key] = int(val)
        elif key == "reserved":
            st["reserved"] = val
        elif key == "unit":
            unit, _, state = val.partition(" ")
            if state.strip() == "active":
                st["active"].append(unit)
    return st


class Board:
    def __init__(self, session, sudo: str = ""):
        self.s, self.sudo = session, sudo

    def run(self, cmd: str, timeout: int = 120):
        out, err, rc = self.s.exec(f"{self.sudo}sh -c {_quote(cmd)}" if self.sudo else cmd, timeout=timeout)
        return out, err, rc

    def state(self) -> Dict:
        return parse_state(self.run(STATE_SCRIPT.replace("UNITS", " ".join(UNITS)), 60)[0])

    def boot_value(self) -> Dict[str, Optional[str]]:
        """The Kria branch's cma of each script source and of the boot image."""
        out = {}
        for p in SCRIPTS:
            text, _, rc = self.run(f"cat {p}", 30)
            out[p] = kria_value(text) if rc == 0 else None
        text, _, rc = self.run(f"dumpimage -T script -p 0 -o /tmp/cma_boot.txt {IMAGE} >/dev/null 2>&1; "
                               "cat /tmp/cma_boot.txt", 60)
        out[IMAGE] = kria_value(text) if rc == 0 else None
        return out

    def set_size(self, size: str) -> None:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        cmds = [f"mkdir -p {BACKUP_DIR}", f"cp -p {IMAGE} {BACKUP_DIR}/boot.scr.uimg.{stamp}"]
        for p in SCRIPTS:
            cmds += [f"cp -p {p} {BACKUP_DIR}/{p.rsplit('/', 1)[1]}.{stamp}", edit_command(p, size)]
        out, err, rc = self.run(" && ".join(cmds), 60)
        if rc:
            raise RuntimeError(f"editing the boot script failed: {err.strip() or out.strip()}")
        out, err, rc = self.run("flash-kernel", 300)
        if rc:
            raise RuntimeError(f"flash-kernel failed ({rc}): {(err or out).strip()[-500:]}")


def _quote(s: str) -> str:
    return "'" + s.replace("'", "'\"'\"'") + "'"


def wait_for_ssh(cfg: dict, down_s: float = 20.0, timeout_s: float = 300.0):
    """A new session once the board answers after a reboot."""
    from .session import RemoteSession
    time.sleep(down_s)
    t0 = time.monotonic()
    while True:
        s = RemoteSession(cfg["ssh"])
        try:
            s.connect()
            return s
        except Exception:                                                  # noqa: BLE001
            if time.monotonic() - t0 > timeout_s:
                raise RuntimeError(f"the board did not answer ssh within {down_s + timeout_s:.0f} s") from None
            time.sleep(5)


def describe(st: Dict) -> str:
    s = (f"running cma={st['cmdline_cma'] or '(none)'}: CmaTotal {st['total_kb'] / 1024:.0f} MiB, "
         f"CmaFree {st['free_kb'] / 1024:.0f} MiB")
    return s + (f" ({st['reserved']})" if st["reserved"] else "")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Report or set the KV260's CMA size (the cma= boot argument).")
    ap.add_argument("--config", required=True, help="a board config JSON (its ssh block, board_lock)")
    ap.add_argument("--set", dest="size", default=None, help="the new size, e.g. 1400M")
    ap.add_argument("--reboot", action="store_true", help="reboot and check the size the kernel reserved")
    ap.add_argument("--force", action="store_true", help=f"allow more than {MAX_MB}M")
    ap.add_argument("--no-lock", action="store_true", help="do not take the board lock")
    args = ap.parse_args(argv)
    want = None
    if args.size:
        try:
            want = parse_size(args.size)
        except ValueError as e:
            ap.error(str(e))
        if want > MAX_MB and not args.force:
            ap.error(f"{args.size} is more than {MAX_MB}M: the CMA area must fit in the low 2 GB of DDR beside "
                     "the kernel (--force to try)")
        if want < 256:
            ap.error(f"{args.size}: the FPGA libraries need hundreds of MB")
        size = f"{want}M"

    from .config import load_config
    from .lock import hold_board_lock
    from .session import RemoteSession
    cfg = load_config(args.config)
    if not args.no_lock:
        hold_board_lock(cfg)
    sudo = "sudo " if cfg["ssh"].get("user", "root") != "root" else ""
    session = RemoteSession(cfg["ssh"])
    session.connect()
    try:
        b = Board(session, sudo)
        st = b.state()
        print(f"cma: {describe(st)}")
        if want is not None:
            b.set_size(size)
        boot = b.boot_value()
        for p, v in boot.items():
            print(f"cma: {p}: Kria branch cma={v or '?'}")
        if want is not None and any(v != size for v in boot.values()):
            print(f"cma: the boot script does not say cma=\"{size}\" everywhere (backups in {BACKUP_DIR})",
                  file=sys.stderr)
            return 1
        if not args.reboot:
            if want is not None:
                print(f"cma: cma=\"{size}\" takes effect at the next boot (--reboot)")
            return 0
        if st["active"]:
            print(f"cma: not rebooting: {', '.join(st['active'])} running (stop them first: demo/chat/deploy.py "
                  "--stop, demo/miranda/deploy.py --stop)", file=sys.stderr)
            return 1
        expect = boot[IMAGE]
        print("cma: rebooting ...", flush=True)
        session.exec(f"{sudo}sh -c 'sleep 1; systemctl reboot' >/dev/null 2>&1 &", timeout=15)
        session.close()
        session = wait_for_ssh(cfg)
        st = Board(session, sudo).state()
        print(f"cma: {describe(st)}")
        if expect and (st["cmdline_cma"] != expect or abs(st["total_kb"] / 1024 - parse_size(expect)) > 1):
            print(f"cma: the kernel did not reserve cma={expect} (CmaTotal {st['total_kb'] / 1024:.0f} MiB); "
                  f"set it back with --set and reboot (backups in {BACKUP_DIR})", file=sys.stderr)
            return 1
        print("cma: done.  Load the bitstream again (upload_bitstream.py), then the servers (demo/chat/deploy.py, "
              "demo/miranda/deploy.py)")
        return 0
    finally:
        session.close()


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ("parse_size", "edit_command", "kria_value", "parse_state", "Board", "describe", "main")
