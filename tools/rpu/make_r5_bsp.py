# Copyright 2026 Ivan Gubochkin
# SPDX-License-Identifier: Apache-2.0

"""make_r5_bsp.py — the Cortex-R5 standalone BSP of this bitstream (RPU_PLAN B1).

Run with the Vitis Python interpreter (it drives the Vitis server):

    source <Xilinx>/2025.2/Vitis/settings64.sh
    vitis -s tools/rpu/make_r5_bsp.py <cormorant.xsa> <workspace>

It creates a platform component for psu_cortexr5_0 (standalone, no boot
components) and builds its BSP; the result the firmware build needs is
<workspace>/cormorant_r5/psu_cortexr5_0/standalone_r5/bsp (its include/
holds the xparameters.h naming the kernels and their interrupt lines, lib/
the libxil / libxilstandalone / libxiltimer archives, Xilinx.spec the link
spec).  Pass that directory as -DBSP_DIR to a generated project's
rpu/CMakeLists.txt (rpu_board.py --bsp-dir).  Rebuild it after every new
bitstream whose kernels, addresses or interrupts changed.
"""
import os
import sys

import vitis


def main(argv):
    if len(argv) != 3:
        print("usage: vitis -s make_r5_bsp.py <design.xsa> <workspace>", file=sys.stderr)
        return 2
    xsa, ws = os.path.abspath(argv[1]), os.path.abspath(argv[2])
    client = vitis.create_client()
    client.set_workspace(path=ws)
    plat = client.create_platform_component(name="cormorant_r5", hw_design=xsa, os="standalone",
                                            cpu="psu_cortexr5_0", domain_name="standalone_r5",
                                            no_boot_bsp=True, generate_dtb=False)
    status = plat.build()
    print("BUILD_STATUS", status)
    bsp = os.path.join(ws, "cormorant_r5", "psu_cortexr5_0", "standalone_r5", "bsp")
    print("BSP_DIR", bsp)
    vitis.dispose()
    return 0 if os.path.isfile(os.path.join(bsp, "lib", "libxil.a")) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
