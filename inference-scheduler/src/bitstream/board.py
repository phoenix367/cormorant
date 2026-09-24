"""Remote board operations over SSH for bitstream loading."""

from pathlib import Path

from ..remote import RemoteSession, _green, _red, _dim

_FPGA_FLAGS    = "/sys/class/fpga_manager/fpga0/flags"
_FPGA_FIRMWARE = "/sys/class/fpga_manager/fpga0/firmware"
_FPGA_STATE    = "/sys/class/fpga_manager/fpga0/state"
_FIRMWARE_DIR  = "/lib/firmware"
_OVERLAYS_DIR  = "/sys/kernel/config/device-tree/overlays"


def upload_file(session: RemoteSession, local_path: Path, remote_path: str) -> None:
    """Upload a single file to the board via SFTP."""
    sftp = session._client.open_sftp()
    try:
        sftp.put(str(local_path), remote_path)
    finally:
        sftp.close()


def upload_bytes(session: RemoteSession, data: bytes, remote_path: str) -> None:
    """Upload raw bytes to a file on the board via SFTP."""
    sftp = session._client.open_sftp()
    try:
        with sftp.open(remote_path, "wb") as fh:
            fh.write(data)
    finally:
        sftp.close()


def upload_text(session: RemoteSession, text: str, remote_path: str) -> None:
    """Upload a text string to a file on the board via SFTP."""
    sftp = session._client.open_sftp()
    try:
        with sftp.open(remote_path, "w") as fh:
            fh.write(text)
    finally:
        sftp.close()


def remove_overlay(session: RemoteSession, overlay_name: str) -> None:
    """Remove a device tree overlay from configfs if present."""
    sysfs_dir = f"{_OVERLAYS_DIR}/{overlay_name}"
    out, _, rc = session.exec(
        f"[ -d '{sysfs_dir}' ] && rmdir '{sysfs_dir}' && echo removed || echo absent",
        timeout=15,
    )
    status = out.strip()
    if status == "removed":
        print(f"  Removed existing overlay: {overlay_name}")
    elif status != "absent":
        raise RuntimeError(f"Failed to remove overlay '{overlay_name}': {out.strip()}")


def load_bitstream(session: RemoteSession, bin_name: str) -> None:
    """Write flags=0 and the .bin filename to the fpga_manager sysfs interface."""
    for cmd in [
        f"echo 0 > {_FPGA_FLAGS}",
        f"echo {bin_name} > {_FPGA_FIRMWARE}",
    ]:
        out, err, rc = session.exec(cmd, timeout=30)
        if rc != 0:
            raise RuntimeError(
                f"FPGA manager command failed (rc={rc}):\n  {cmd}\n{err}")


def fpga_state(session: RemoteSession) -> str:
    """Return the current fpga_manager state string."""
    out, _, _ = session.exec(
        f"cat {_FPGA_STATE} 2>/dev/null || echo unknown", timeout=10)
    return out.strip()


def set_axi_port_widths(
    session: RemoteSession,
    writes: list[tuple[int, int, int]],
) -> None:
    """
    Apply (addr, mask, value) RMW register writes via /dev/mem on the board.

    Uploads a small Python script to avoid shell quoting issues with one-liners.
    """
    if not writes:
        return

    script = "\n".join([
        "import mmap, os, struct",
        "PAGE = 4096",
        f"writes = {writes!r}",
        "fd = os.open('/dev/mem', os.O_RDWR | os.O_SYNC)",
        "for addr, mask, val in writes:",
        "    base = addr & ~(PAGE - 1)",
        "    off  = addr - base",
        "    m = mmap.mmap(fd, PAGE, mmap.MAP_SHARED,",
        "                  mmap.PROT_READ | mmap.PROT_WRITE, offset=base)",
        "    v = struct.unpack_from('<I', m, off)[0]",
        "    struct.pack_into('<I', m, off, (v & ~mask) | val)",
        "    m.close()",
        "os.close(fd)",
    ])

    remote_script = "/tmp/_axi_port_width.py"
    upload_text(session, script, remote_script)
    out, err, rc = session.exec(
        f"python3 '{remote_script}'; rm -f '{remote_script}'", timeout=15)
    if rc != 0:
        raise RuntimeError(f"AXI port-width register write failed (rc={rc}):\n{err}")


def load_xclbin(session: RemoteSession, xclbin_data: bytes) -> None:
    """
    Load an xclbin into the zocl DRM driver via xclLoadXclBin (ctypes).

    Uploads the xclbin and a loader script, executes on the board, then
    removes both temp files.  Registers MEM_TOPOLOGY so that xclAllocBO
    resolves to a named memory bank instead of falling back to CMA.
    """
    remote_xclbin = "/tmp/_design.xclbin"
    remote_script  = "/tmp/_load_xclbin.py"

    upload_bytes(session, xclbin_data, remote_xclbin)

    script = "\n".join([
        "import ctypes, os, sys",
        "",
        "data = open(sys.argv[1], 'rb').read()",
        "",
        "candidates = [",
        "    os.path.join(os.environ.get('XILINX_XRT', '/opt/xilinx/xrt'), 'lib', 'libxrt_core.so'),",
        "    '/opt/xilinx/xrt/lib/libxrt_core.so',",
        "    '/usr/lib/aarch64-linux-gnu/libxrt_core.so',",
        "    '/usr/lib/libxrt_core.so',",
        "]",
        "lib = next((ctypes.CDLL(p) for p in candidates if os.path.exists(p)), None)",
        "if lib is None:",
        "    print('ERROR: libxrt_core.so not found', file=sys.stderr); sys.exit(1)",
        "",
        "lib.xclOpen.restype  = ctypes.c_void_p",
        "lib.xclOpen.argtypes = [ctypes.c_uint, ctypes.c_char_p, ctypes.c_int]",
        "lib.xclLoadXclBin.restype  = ctypes.c_int",
        "lib.xclLoadXclBin.argtypes = [ctypes.c_void_p, ctypes.c_void_p]",
        "lib.xclClose.restype  = None",
        "lib.xclClose.argtypes = [ctypes.c_void_p]",
        "",
        "handle = lib.xclOpen(0, None, 0)",
        "if not handle:",
        "    print('ERROR: xclOpen failed', file=sys.stderr); sys.exit(1)",
        "",
        "buf = ctypes.create_string_buffer(data, len(data))",
        "rc  = lib.xclLoadXclBin(handle, ctypes.cast(buf, ctypes.c_void_p))",
        "lib.xclClose(handle)",
        "if rc != 0:",
        "    print(f'ERROR: xclLoadXclBin failed rc={rc}', file=sys.stderr); sys.exit(1)",
    ])

    upload_text(session, script, remote_script)
    out, err, rc = session.exec(
        f"python3 '{remote_script}' '{remote_xclbin}';"
        f" rm -f '{remote_script}' '{remote_xclbin}'",
        timeout=20,
    )
    if rc != 0:
        raise RuntimeError(f"xclLoadXclBin failed (rc={rc}):\n{err}")


def unbind_stale_uio(session: RemoteSession, node_names: list[str]) -> list[str]:
    """
    Unbind uio_pdrv_genirq platform devices that sit at one of our kernel
    addresses but are NOT ours (e.g. `a0000000.fabric` from the `pynq`
    overlay that appears during the xclbin load on PYNQ images).  Such a
    device survives the removal of its overlay and keeps the shared IRQ,
    so our node fails to probe with EBUSY ("genirq: Flags mismatch irq 61
    (fabric_vecop) vs (fabric)") and the kernel's UIO name is wrong.
    Returns the unbound device names.
    """
    drv = "/sys/bus/platform/drivers/uio_pdrv_genirq"
    out, _, _ = session.exec(f"ls '{drv}' 2>/dev/null | grep -E '^[0-9a-f]+\\.' || true", timeout=10)
    ours = set(node_names)
    unbound = []
    for dev in out.split():
        addr, _, name = dev.partition(".")
        if any(n.split("@")[0] == name and n.split("@")[1].lower() == addr.lower() for n in ours):
            continue
        if any(n.split("@")[1].lower() == addr.lower() for n in ours):
            _, err, rc = session.exec(f"echo '{dev}' > '{drv}/unbind'", timeout=10)
            if rc == 0:
                unbound.append(dev)
    return unbound


def apply_dtbo(session: RemoteSession, remote_dtbo: str, overlay_name: str) -> None:
    """Create the configfs overlay directory and write the .dtbo into it."""
    sysfs_dir = f"{_OVERLAYS_DIR}/{overlay_name}"
    out, err, rc = session.exec(f"mkdir -p '{sysfs_dir}'", timeout=10)
    if rc != 0:
        raise RuntimeError(f"Failed to create overlay directory '{sysfs_dir}':\n{err}")
    out, err, rc = session.exec(
        f"cat '{remote_dtbo}' > '{sysfs_dir}/dtbo'", timeout=15)
    if rc != 0:
        raise RuntimeError(f"Failed to write DTBO to '{sysfs_dir}/dtbo':\n{err}")


def overlay_status(session: RemoteSession, overlay_name: str) -> str:
    """Return the configfs overlay status string ('applied', 'error', etc.)."""
    sysfs_dir = f"{_OVERLAYS_DIR}/{overlay_name}"
    out, _, _ = session.exec(
        f"cat '{sysfs_dir}/status' 2>/dev/null || echo unknown", timeout=10)
    return out.strip()


def list_uio_devices(session: RemoteSession) -> list[tuple[str, str]]:
    """Return [(name, /dev/uioN), ...] for all UIO devices on the board."""
    out, _, _ = session.exec(
        "for f in /sys/class/uio/uio*/name; do "
        "  [ -f \"$f\" ] || continue; "
        "  name=$(cat \"$f\"); "
        "  dev=$(echo \"$f\" | grep -o 'uio[0-9]*'); "
        "  echo \"/dev/$dev $name\"; "
        "done",
        timeout=10,
    )
    devices = []
    for line in out.strip().splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2:
            devices.append((parts[1].strip(), parts[0].strip()))
    return devices


def check_board(session: RemoteSession) -> bool:
    """Verify the board exposes the sysfs interfaces required to load a bitstream."""
    checks = [
        (f"[ -f {_FPGA_FLAGS} ] && echo ok || echo missing",      "fpga_manager flags"),
        (f"[ -f {_FPGA_FIRMWARE} ] && echo ok || echo missing",    "fpga_manager firmware"),
        (f"[ -d {_OVERLAYS_DIR} ] && echo ok || echo missing",     "configfs overlays dir"),
        (f"[ -d {_FIRMWARE_DIR} ] && echo ok || echo missing",     "/lib/firmware dir"),
        ("[ -c /dev/mem ] && echo ok || echo missing",            "/dev/mem (AFIFM RMW)"),
        ("[ \"$(id -u)\" = '0' ] && echo 'root' || "
         "sudo -n true 2>/dev/null && echo 'passwordless sudo' || echo '__MISSING__'",
         "root / sudo access"),
    ]
    all_ok = True
    for cmd, label in checks:
        out, _, _ = session.exec(cmd, timeout=10)
        ok = "ok" in out or "root" in out or "sudo" in out
        sym = _green("OK") if ok else _red("MISSING")
        print(f"  {sym}  {label:<30}  {_dim(out.strip())}")
        if not ok:
            all_ok = False
    return all_ok
