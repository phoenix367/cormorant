#!/usr/bin/env python3
# Copyright 2026 Ivan Gubochkin
# SPDX-License-Identifier: Apache-2.0

"""
fpga_load.py — on the board, as root: make sure the FPGA runs the chat
server's bitstream and nothing else uses it.  The kv260-chat unit runs it
before the server (ExecStartPre); deploy.py installs it with the bundle.

  fpga_load.py BUNDLE_DIR            load the bundle's bitstream unless it is loaded
  fpga_load.py --check BUNDLE_DIR    report only (exit 0: loaded and free, 3: not)

The bundle (<dir>/fpga; inference-scheduler/src/bitstream/bundle.py builds it
on the host from the bitstream the server was deployed on): bitstream.bin,
overlay.dtbo, design.xclbin and load.json (its id, the overlay's name, the
PL0 frequency, the PS AXI port-width register writes, the UIO nodes and
their interrupts).

The FPGA is in use when a process holds its render device (the XRT buffers)
or a kernel's UIO device: a board job, or a server started by hand.  Then
nothing is loaded and the exit status is 1 — reprogramming the PL under a
kernel's AXI traffic wedges the PS port, and a second process driving the
kernels corrupts both.

The bitstream is loaded when the overlay is applied, /lib/firmware/<overlay>.bin
is the bundle's (its SHA-256: the bitstream id every board tool reads), the
FPGA manager is operating and every UIO node of the overlay exists; then PL0
and the port widths are only checked (and set when they are off).  Otherwise
— after a boot the board runs the starter-kit design; a board job may have
loaded another bitstream — it does upload_bitstream.py's board steps
(inference-scheduler/src/bitstream/loader.py): remove a `pynq` overlay, the
Cortex-R5's overlays applied after ours (they stay when its firmware stops; it
must not run, and its next start applies them again) and ours, check no other
overlay provides the kernels, set PL0, copy the .bin to
/lib/firmware and program it, try the xclbin (zocl has no device before the
overlay, so this fails as in upload_bitstream.py: logged), apply the overlay
(after unbinding a foreign UIO device at one of its addresses), write the
port widths last (the overlay's afi0 node resets them) and check each UIO
node's interrupt.  A refused overlay removal stops it: reboot the board.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import mmap
import os
import re
import struct
import subprocess
import sys
import time

FIRMWARE = "/lib/firmware"
OVERLAYS = "/sys/kernel/config/device-tree/overlays"
FPGA = "/sys/class/fpga_manager/fpga0"
FCLK0 = "/sys/devices/platform/fclk0/set_rate"
UIO_DRV = "/sys/bus/platform/drivers/uio_pdrv_genirq"
UIO_CLASS = "/sys/class/uio"
INTERRUPTS = "/proc/interrupts"
REMOTEPROC = "/sys/class/remoteproc"
LIVE_DT = "/sys/firmware/devicetree/base"
# the Cortex-R5's overlays (rpu_up.sh applies them when its firmware starts; they stay
# when it stops), each with a label its .dts defines
R5_OVERLAYS = (("rpu", "tcm_0a"), ("rpu-ipi", "rpu_ipi"))
XRT_CORE = ("/opt/xilinx/xrt/lib/libxrt_core.so", "/usr/lib/aarch64-linux-gnu/libxrt_core.so",
            "/usr/lib/libxrt_core.so")


class LoadError(RuntimeError):
    pass


def log(msg: str) -> None:
    print(f"fpga_load: {msg}", flush=True)


def read(path: str, default: str = "") -> str:
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return default


def write(path: str, text: str) -> None:
    with open(path, "w") as f:
        f.write(text)


def sha256(path: str) -> str:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    except OSError:
        return ""
    return h.hexdigest()


# ── what the board runs ──────────────────────────────────────────────────────

def uio_devices() -> dict:
    """{UIO name: /dev/uioN}."""
    devs = {}
    for d in sorted(os.listdir(UIO_CLASS)) if os.path.isdir(UIO_CLASS) else []:
        name = read(f"{UIO_CLASS}/{d}/name")
        if name:
            devs[name] = f"/dev/{d}"
    return devs


def holders(uio_names) -> list:
    """'pid cmdline' of every process holding the FPGA's render device or one
    of the kernels' UIO devices (not this process)."""
    devs = {p for n, p in uio_devices().items() if n in uio_names}
    found = []
    for pid in filter(str.isdigit, os.listdir("/proc")):
        if int(pid) == os.getpid():
            continue
        try:
            fds = [os.readlink(f"/proc/{pid}/fd/{fd}") for fd in os.listdir(f"/proc/{pid}/fd")]
        except OSError:
            continue
        if any("renderD" in t or t in devs for t in fds):
            cmd = read(f"/proc/{pid}/cmdline").replace("\0", " ").strip()
            found.append(f"{pid} {cmd[:100]}")
    return found


def overlay_status(name: str) -> str:
    return read(f"{OVERLAYS}/{name}/status", "absent")


def dmesg() -> list:
    """[(timestamp s, text)] of the kernel log."""
    out = subprocess.run(["dmesg"], capture_output=True, text=True).stdout
    lines = []
    for ln in out.splitlines():
        m = re.match(r"\[\s*(\d+\.\d+)\]\s*(.*)", ln)
        if m:
            lines.append((float(m.group(1)), m.group(2)))
    return lines


def dmesg_mark() -> float:
    lines = dmesg()
    return lines[-1][0] if lines else 0.0


def parse_proc_interrupts(text: str) -> dict:
    """{handler name: GIC id} (inference-scheduler/src/bitstream/board.py)."""
    irqs = {}
    for line in text.splitlines():
        m = re.match(r"\s*\d+:(?:\s+\d+)+\s+GIC\S*\s+(\d+)\s+\S+\s+(\S+)\s*$", line)
        if m:
            irqs[m.group(2)] = int(m.group(1))
    return irqs


def stale_irqs(want: dict) -> dict:
    """{'name@addr': (held, wanted)} of the UIO nodes whose interrupt is not the overlay's."""
    have = parse_proc_interrupts(read(INTERRUPTS))
    return {n: (have.get(n.split("@")[0]), irq) for n, irq in want.items()
            if have.get(n.split("@")[0]) != irq}


def loaded(load: dict) -> list:
    """Why the bundle's bitstream is not the loaded one ([]: it is)."""
    name = load["overlay"]
    why = []
    if overlay_status(name) != "applied":
        why.append(f"overlay '{name}' {overlay_status(name)}")
    have = sha256(f"{FIRMWARE}/{name}.bin")[:12]
    if have != load["id"]:
        why.append(f"{FIRMWARE}/{name}.bin is {have or 'missing'}, not {load['id']}")
    if read(f"{FPGA}/state") != "operating":
        why.append(f"FPGA manager {read(f'{FPGA}/state', 'unknown')}")
    missing = [n.split("@")[0] for n in load["uio_nodes"] if n.split("@")[0] not in uio_devices()]
    if missing:
        why.append(f"no UIO device {', '.join(missing)}")
    return why


# ── loading ──────────────────────────────────────────────────────────────────

def remove_overlay(name: str) -> None:
    """rmdir the overlay; a refusal the kernel logs ("not topmost") stops the load:
    its nodes stay in the live tree, and only a reboot removes them."""
    if not os.path.isdir(f"{OVERLAYS}/{name}"):
        return
    mark = dmesg_mark()
    os.rmdir(f"{OVERLAYS}/{name}")
    refused = [t for ts, t in dmesg() if ts > mark and ("not topmost" in t or "overlaps with" in t)]
    if refused:
        raise LoadError(f"the kernel refused to remove overlay '{name}' ({'; '.join(refused)}): "
                        "reboot the board")
    log(f"removed overlay '{name}'")


def live_phandle(label: str):
    """The live device tree's phandle of the node labelled ``label``, or None."""
    try:
        with open(f"{LIVE_DT}/__symbols__/{label}", "rb") as f:
            path = f.read().rstrip(b"\0").decode()
        with open(f"{LIVE_DT}{path}/phandle", "rb") as f:
            return struct.unpack(">I", f.read(4))[0]
    except (OSError, struct.error, UnicodeDecodeError):
        return None


def r5_overlays_above(name: str, label: str) -> list:
    """The Cortex-R5's overlays applied after ours (``name``, whose nodes
    include the one labelled ``label``), newest first: the ones to remove
    before ours, since every overlay adds to /__symbols__ and the kernel
    refuses to remove ours from under them ("node_overlaps_later_cs: #2
    overlaps with #4 @/__symbols__").  The firmware's next start applies them
    again.  configfs keeps no order, but the kernel gives an overlay's nodes
    phandles above every one before it, so the live phandles of the labelled
    nodes order the overlays.  Refuses while the firmware runs (it drives the
    kernels) and when a phandle is missing."""
    r5 = [(n, lb) for n, lb in R5_OVERLAYS if os.path.isdir(f"{OVERLAYS}/{n}")]
    if not os.path.isdir(f"{OVERLAYS}/{name}") or not r5:
        return []
    ph = {lb: live_phandle(lb) for lb in [label] + [lb for _, lb in r5]}
    missing = [lb for lb, p in ph.items() if p is None]
    if missing:
        raise LoadError(f"cannot tell whether the Cortex-R5's overlays ({', '.join(n for n, _ in r5)}) sit "
                        f"above '{name}': no live phandle for {', '.join(missing)}; reboot the board")
    above = [n for n, lb in sorted(r5, key=lambda r: ph[r[1]], reverse=True) if ph[lb] > ph[label]]
    states = ([read(f"{REMOTEPROC}/{d}/state") for d in sorted(os.listdir(REMOTEPROC))]
              if os.path.isdir(REMOTEPROC) else [])
    if above and "running" in states:
        raise LoadError(f"the Cortex-R5 firmware is running and its overlays ({', '.join(above)}) "
                        f"sit above '{name}': stop it first (echo stop > {REMOTEPROC}/remoteproc0/state)")
    return above


def check_kernels_free(load: dict) -> None:
    """No other overlay provides the kernels' UIO devices (the same design under another name)."""
    taken = sorted({n.split("@")[0] for n in load["uio_nodes"]} & set(uio_devices()))
    if taken:
        others = [o for o in os.listdir(OVERLAYS) if o != load["overlay"]]
        raise LoadError(f"the kernels' UIO devices exist without overlay '{load['overlay']}' "
                        f"({', '.join(taken)}; overlays: {', '.join(others) or 'none'}): remove that "
                        f"overlay (rmdir {OVERLAYS}/<name>), then start again")


def check_pl0(hz, set_if_off: bool) -> None:
    if hz is None:
        return
    try:
        have = int(read(FCLK0, "0"))
    except ValueError:
        have = 0
    if abs(have - hz) > 0.01 * hz and set_if_off:
        log(f"PL0 {have} Hz, the bitstream needs {hz}: setting it")
        write(FCLK0, str(hz))
        have = int(read(FCLK0, "0") or 0)
    if abs(have - hz) > 0.01 * hz:
        raise LoadError(f"PL0 (fclk0) is {have} Hz, the bitstream needs {hz} Hz")


def program(load: dict, bundle: str) -> None:
    name = load["overlay"]
    dst = f"{FIRMWARE}/{name}.bin"
    if sha256(dst)[:12] != load["id"]:
        tmp = f"{dst}.tmp"
        with open(f"{bundle}/bitstream.bin", "rb") as src, open(tmp, "wb") as out:
            out.write(src.read())
        os.replace(tmp, dst)
    write(f"{FPGA}/flags", "0")
    write(f"{FPGA}/firmware", f"{name}.bin")
    state = read(f"{FPGA}/state")
    if state != "operating":
        raise LoadError(f"FPGA manager state '{state}' after programming {dst} (dmesg)")
    log(f"programmed {dst} ({load['id']})")


def load_xclbin(path: str) -> None:
    """The xclbin's MEM_TOPOLOGY into zocl (xclLoadXclBin), as loader.py does."""
    lib = next((ctypes.CDLL(p) for p in XRT_CORE if os.path.exists(p)), None)
    if lib is None:
        raise LoadError("libxrt_core.so not found")
    lib.xclOpen.restype = ctypes.c_void_p
    lib.xclOpen.argtypes = [ctypes.c_uint, ctypes.c_char_p, ctypes.c_int]
    lib.xclLoadXclBin.restype = ctypes.c_int
    lib.xclLoadXclBin.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.xclClose.restype = None
    lib.xclClose.argtypes = [ctypes.c_void_p]
    with open(path, "rb") as f:
        data = f.read()
    handle = lib.xclOpen(0, None, 0)
    if not handle:
        raise LoadError("xclOpen failed")
    buf = ctypes.create_string_buffer(data, len(data))
    rc = lib.xclLoadXclBin(handle, ctypes.cast(buf, ctypes.c_void_p))
    lib.xclClose(handle)
    if rc != 0:
        raise LoadError(f"xclLoadXclBin failed (rc {rc})")


def unbind_foreign_uio(load: dict) -> None:
    """A UIO device at one of our addresses under another name (a `pynq`
    overlay's a0000000.fabric) keeps the interrupt our node needs: unbind it."""
    ours = {n.split("@")[1].lower(): n.split("@")[0] for n in load["uio_nodes"]}
    for dev in os.listdir(UIO_DRV) if os.path.isdir(UIO_DRV) else []:
        addr, _, name = dev.partition(".")
        if re.fullmatch(r"[0-9a-f]+", addr) and addr.lower() in ours and ours[addr.lower()] != name:
            write(f"{UIO_DRV}/unbind", dev)
            log(f"unbound foreign UIO device {dev}")


def apply_overlay(load: dict, bundle: str) -> None:
    name = load["overlay"]
    mark = dmesg_mark()
    os.mkdir(f"{OVERLAYS}/{name}")
    with open(f"{bundle}/overlay.dtbo", "rb") as src, open(f"{OVERLAYS}/{name}/dtbo", "wb") as out:
        out.write(src.read())
    errors = [t for ts, t in dmesg() if ts > mark and "overlay" in t.lower()
              and any(w in t.lower() for w in ("fail", "error", "err="))]
    if overlay_status(name) != "applied" or errors:
        raise LoadError(f"overlay '{name}' did not apply (status {overlay_status(name)}"
                        + (f": {'; '.join(errors)}" if errors else "") + ")")
    log(f"applied overlay '{name}'")


def port_widths(writes: list, fix: bool) -> int:
    """The PS AXI port-width fields (addr, mask, value) via /dev/mem: how many
    were off (written when fix)."""
    off = 0
    page = mmap.PAGESIZE
    fd = os.open("/dev/mem", os.O_RDWR | os.O_SYNC)
    try:
        for addr, mask, val in writes:
            base = addr & ~(page - 1)
            m = mmap.mmap(fd, page, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=base)
            try:
                v = struct.unpack_from("<I", m, addr - base)[0]
                if v & mask != val:
                    off += 1
                    if fix:
                        struct.pack_into("<I", m, addr - base, (v & ~mask) | val)
            finally:
                m.close()
    finally:
        os.close(fd)
    return off


def check_irqs(load: dict) -> None:
    bad = stale_irqs(load["uio_irqs"])
    if bad:
        log("stale interrupts: " + ", ".join(f"{n} GIC {h} (overlay {w})" for n, (h, w) in bad.items())
            + " - rebinding")
        for n in bad:
            dev = f"{n.split('@')[1]}.{n.split('@')[0]}"
            write(f"{UIO_DRV}/unbind", dev)
            write(f"{UIO_DRV}/bind", dev)
        bad = stale_irqs(load["uio_irqs"])
    if bad:
        raise LoadError("UIO interrupts disagree with the overlay after a rebind: "
                        + ", ".join(f"{n}: GIC {h}, overlay {w}" for n, (h, w) in bad.items())
                        + ": reboot the board")


def load_bundle(load: dict, bundle: str) -> None:
    t0 = time.monotonic()
    r5_above = r5_overlays_above(load["overlay"], load["uio_nodes"][0].split("@")[0])
    remove_overlay("pynq")
    for name in r5_above:
        remove_overlay(name)
    remove_overlay(load["overlay"])
    check_kernels_free(load)
    check_pl0(load["pl0_hz"], set_if_off=True)
    program(load, bundle)
    try:
        load_xclbin(f"{bundle}/design.xclbin")
    except LoadError as e:
        # upload_bitstream.py's step 8 at the same point: zocl's device comes with the
        # overlay (its zyxclmm_drm node), not applied yet, so after a boot or a reload
        # it fails there as well (its exit status is not the script's) and the buffers
        # come from CMA ("Allocating BO from CMA for invalid or unused memory index[0]"),
        # as in every board run so far
        log(f"xclbin not loaded ({e}): the buffers come from CMA, as after upload_bitstream.py")
    check_pl0(load["pl0_hz"], set_if_off=False)
    remove_overlay("pynq")
    unbind_foreign_uio(load)
    apply_overlay(load, bundle)
    port_widths(load["axi_writes"], fix=True)
    check_irqs(load)
    why = loaded(load)
    if why:
        raise LoadError(f"still not loaded: {'; '.join(why)}")
    log(f"bitstream {load['id']} loaded in {time.monotonic() - t0:.1f} s "
        f"({', '.join(n.split('@')[0] for n in load['uio_nodes'])})")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bundle", help="the bundle directory (<dir>/fpga)")
    ap.add_argument("--check", action="store_true", help="report only (exit 0: loaded and free, 3: not)")
    args = ap.parse_args(argv)
    try:
        with open(os.path.join(args.bundle, "load.json")) as f:
            load = json.load(f)
    except (OSError, ValueError) as e:
        log(f"error: no bundle in {args.bundle}: {e}")
        return 1
    names = {n.split("@")[0] for n in load["uio_nodes"]}
    busy = holders(names)
    why = loaded(load)
    if args.check:
        log(f"bitstream {load['id']}: " + ("loaded" if not why else "not loaded (" + "; ".join(why) + ")"))
        for h in busy:
            log(f"in use by pid {h}")
        return 0 if not why and not busy else 3
    if busy:
        log("error: the FPGA is in use (a board job?): " + "; ".join(f"pid {h}" for h in busy))
        return 1
    try:
        if not why:
            check_pl0(load["pl0_hz"], set_if_off=True)
            off = port_widths(load["axi_writes"], fix=True)
            check_irqs(load)
            log(f"bitstream {load['id']} loaded" + (f" ({off} port-width fields were off: set)" if off else ""))
            return 0
        log(f"loading bitstream {load['id']}: {'; '.join(why)}")
        load_bundle(load, args.bundle)
    except (LoadError, OSError) as e:
        log(f"error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
