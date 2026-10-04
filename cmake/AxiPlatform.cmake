# AxiPlatform.cmake — the bus width and platform definitions every kernel
# build reads.
#
# Included by the top-level CMakeLists.txt and by each kernel's own
# CMakeLists.txt, so a kernel also configures on its own
# (cmake -S kernels/<k> -B <dir>).  Under the top-level project the kernels
# inherit the variables it set and the include is a no-op.
#
# Sets:
#   AXI_BUS_WIDTH               (cache) the kernels' AXI master bus width
#   AXI_PLATFORM_FILES          platforms/*.json — one synthesis target per file
#   AXI_PLATFORM                (cache) the default platform, kv260
#   AXI_DEFAULT_PLATFORM_JSON   platforms/<AXI_PLATFORM>.json — the bounds the
#                               C-sim builds (and the Python scheduler) target
include_guard(GLOBAL)

# AXI master bus data width.  Valid values: 32, 64, 128, 256, 512.
# Wider buses amortise burst overhead; must match the interconnect configured
# in the Vivado block design (MAXI_DATA_WIDTH for each kernel IP).
set(AXI_BUS_WIDTH 32 CACHE STRING
    "AXI master bus data width in bits (32, 64, 128, 256, 512)")
set_property(CACHE AXI_BUS_WIDTH PROPERTY STRINGS 32 64 128 256 512)

get_filename_component(_AXI_PLATFORM_DIR "${CMAKE_CURRENT_LIST_DIR}/../platforms" ABSOLUTE)

# Collect platform files once; kernels iterate over this list.
file(GLOB AXI_PLATFORM_FILES "${_AXI_PLATFORM_DIR}/*.json")
if(NOT AXI_PLATFORM_FILES)
    message(WARNING "axi_kernels: no platform JSON files found in ${_AXI_PLATFORM_DIR}")
endif()

# Default platform — drives the C-simulation builds (and the kernel
# constants embedded in their Config.h).  HLS synthesis targets are still
# generated per platform from AXI_PLATFORM_FILES; the default is just the
# one whose bounds the C-sim libraries (and the Python validator in
# inference-scheduler) target.  Override with -DAXI_PLATFORM=<name> at
# configure time when a project builds for a non-default board.
set(AXI_PLATFORM "kv260" CACHE STRING
    "Default platform whose platforms/<name>.json drives C-sim builds")
set(AXI_DEFAULT_PLATFORM_JSON "${_AXI_PLATFORM_DIR}/${AXI_PLATFORM}.json")
if(NOT EXISTS "${AXI_DEFAULT_PLATFORM_JSON}")
    message(FATAL_ERROR
        "axi_kernels: AXI_PLATFORM='${AXI_PLATFORM}' but "
        "${AXI_DEFAULT_PLATFORM_JSON} does not exist.")
endif()
