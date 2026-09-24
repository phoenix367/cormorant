"""High-level bitstream upload orchestration."""

import tempfile
from pathlib import Path

from ..remote import RemoteSession, _green, _yellow, _bold
from .convert import bit_to_bin
from .hwh import parse_hwh_ps_params
from .xclbin import build_xclbin
from .board import (
    _FIRMWARE_DIR,
    upload_file,
    remove_overlay,
    load_bitstream,
    fpga_state,
    set_axi_port_widths,
    load_xclbin,
    apply_dtbo,
    overlay_status,
    list_uio_devices,
)
from .platforms import kv260


def upload_bitstream(
    session:      RemoteSession,
    bit_path:     Path,
    hwh_path:     Path,
    dtbo_path:    Path,
    overlay_name: str,
    xclbinutil:   str = "xclbinutil",
) -> None:
    """
    Convert, upload, and activate a bitstream + DTBO on the board.

    Loading sequence:
      1  .bit → .bin (header strip + 32-bit byteswap)
      2  Parse HWH — PS family + AXI port-width parameters
      3  Build xclbin with MEM_TOPOLOGY via xclbinutil
      4  Upload .bin → /lib/firmware/<overlay_name>.bin
      5  Remove any existing configfs DTBO overlay
      6  Load bitstream via fpga_manager
      7  Verify fpga_manager state == "operating"
      8  Load xclbin into zocl DRM driver
      9  Upload and apply DTBO via configfs
      10 Verify overlay status == "applied"
      11 Write PS SLCR / AXIFM registers (set_axi_port_width)
      12 List /dev/uio* devices

    The AXI port widths are written AFTER the overlay is applied on
    purpose: the overlay's `afi0` node (`xlnx,afi-fpga`, `config-afi`
    table) is itself applied by the kernel's AFI driver and resets the
    AFIFM width fields — on a freshly booted board the old order left
    every PS slave port at 32 bits under a 128-bit design, and every
    kernel then read/wrote garbage (all scheduler models failed, models
    that had passed earlier mispredicted).
    """
    bin_name    = f"{overlay_name}.bin"
    remote_bin  = f"{_FIRMWARE_DIR}/{bin_name}"
    remote_dtbo = f"/tmp/{overlay_name}.dtbo"

    print(f"\n{_bold('Step 1')}   Converting {bit_path.name} → {bin_name}")
    bin_data = bit_to_bin(bit_path)
    print(f"          {len(bin_data):,} bytes")

    print(f"\n{_bold('Step 2')}   Parsing HWH: {hwh_path.name}")
    family, ps_params = parse_hwh_ps_params(hwh_path)
    print(f"          PS family: {family}")
    axi_writes = kv260.axi_port_width_writes(family, ps_params)
    for param, width in sorted(ps_params.items()):
        print(f"          {param} = {width}")

    print(f"\n{_bold('Step 3')}   Building xclbin (MEM_TOPOLOGY from HWH)")
    xclbin_data = build_xclbin(hwh_path, kv260.BLANK_METADATA, xclbinutil)
    print(f"          {len(xclbin_data):,} bytes")

    print(f"\n{_bold('Step 4')}   Uploading → {remote_bin}")
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tf:
        tmp_bin = Path(tf.name)
        tmp_bin.write_bytes(bin_data)
    try:
        upload_file(session, tmp_bin, remote_bin)
    finally:
        tmp_bin.unlink(missing_ok=True)
    print("          done")

    print(f"\n{_bold('Step 5')}   Removing existing overlay '{overlay_name}' (if any)")
    remove_overlay(session, overlay_name)

    print(f"\n{_bold('Step 6')}   Loading bitstream via FPGA manager")
    load_bitstream(session, bin_name)

    print(f"\n{_bold('Step 7')}   Verifying FPGA manager state")
    state = fpga_state(session)
    if state == "operating":
        print(f"          {_green('operating')}  ✓")
    else:
        raise RuntimeError(
            f"FPGA manager state is '{state}' (expected 'operating').\n"
            f"Check dmesg on the board for details.")

    print(f"\n{_bold('Step 8')}   Loading xclbin into zocl DRM driver")
    load_xclbin(session, xclbin_data)
    print("          done")

    print(f"\n{_bold('Step 9')}   Uploading DTBO → {remote_dtbo}")
    upload_file(session, dtbo_path, remote_dtbo)
    print(f"          Applying overlay '{overlay_name}'")
    apply_dtbo(session, remote_dtbo, overlay_name)

    print(f"\n{_bold('Step 10')}  Verifying overlay status")
    status = overlay_status(session, overlay_name)
    if status == "applied":
        print(f"          {_green('applied')}  ✓")
    else:
        raise RuntimeError(
            f"Overlay status is '{status}' (expected 'applied').\n"
            f"Check dmesg on the board for device tree errors.")

    # After the overlay: its afi0 node resets the AFIFM width fields (see
    # the docstring), so the HWH-derived widths must be written last.
    print(f"\n{_bold('Step 11')}  Setting PS AXI port widths ({len(axi_writes)} register writes)")
    set_axi_port_widths(session, axi_writes)
    print("          done")

    print(f"\n{_bold('Step 12')}  UIO devices")
    devices = list_uio_devices(session)
    if devices:
        for name, dev in devices:
            print(f"          {_green(dev)}  {name}")
    else:
        print(f"          {_yellow('none found')}  (check dmesg for DT errors)")

    print(f"\n{_green(_bold('Done.'))}  Bitstream loaded and overlay applied.\n")
