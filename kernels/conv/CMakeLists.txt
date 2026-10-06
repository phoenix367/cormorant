cmake_minimum_required(VERSION 3.19)
project(ConvKernel VERSION 1.0 LANGUAGES CXX)

# The platform JSONs (AXI_PLATFORM_FILES, AXI_DEFAULT_PLATFORM_JSON): set by the
# top-level project, or here when this kernel is configured on its own.
list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../cmake")
include(AxiPlatform)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Locate Vitis HLS headers (ap_fixed.h) for fixed-point C++ simulation.
# The headers are header-only — no HLS synthesis flow is needed for C-sim.
# If not found, Data_t falls back to float, but the kernel headers include
# ap_int.h / hls_burst_maxi.h unconditionally, so nothing builds without
# them (the float Data_t is only the configure default).
# ---------------------------------------------------------------------------
find_path(HLS_INCLUDE_DIR NAMES ap_fixed.h
    HINTS
        $ENV{XILINX_HLS}/include
        $ENV{XILINX_VITIS}/include
        /mnt/data/xilinx/2025.2/Vitis/include
    NO_DEFAULT_PATH
    DOC "Vitis HLS include directory containing ap_fixed.h"
)

if(HLS_INCLUDE_DIR)
    message(STATUS "ConvKernel: Vitis HLS headers found at ${HLS_INCLUDE_DIR}")
    set(_CONV_HAVE_HLS TRUE)
else()
    message(STATUS "ConvKernel: Vitis HLS headers not found — "
                   "the C-sim targets need them and will not build")
    set(_CONV_HAVE_HLS FALSE)
endif()

# ---------------------------------------------------------------------------
# Data type configuration
# ---------------------------------------------------------------------------
if(_CONV_HAVE_HLS)
    set(CONV_DATA_TYPE     "ap_fixed<16,8>"  CACHE STRING "Element type")
    set(CONV_ACC_DATA_TYPE "ap_fixed<32,16>" CACHE STRING "Accumulator type")
    set(CONV_DATA_TYPE_INCLUDE  "#include \"ap_fixed.h\"")
    set(CONV_HAVE_APFIXED_DEFINE "#define CONV_HAVE_APFIXED")
else()
    set(CONV_DATA_TYPE     "float"  CACHE STRING "Element type")
    set(CONV_ACC_DATA_TYPE "float"  CACHE STRING "Accumulator type")
    set(CONV_DATA_TYPE_INCLUDE  "")
    set(CONV_HAVE_APFIXED_DEFINE "")
endif()

# ---------------------------------------------------------------------------
# Compile-time kernel bounds — read from the platform JSON.
#
# These are the single source of truth, defined under "kernels.conv" in each
# platforms/<name>.json — there are NO CMake CACHE defaults; the JSON is
# authoritative.  conv_load_constants() errors out if any field is missing
# rather than silently defaulting.  Field reference:
#   tile_m / tile_ic            output / input channel tiles (power of 2)
#   max_kh / max_kw             kernel-size limits (Conv nodes above are
#                               rejected at scheduling)
#   max_in_ch / max_out_ch      channel-count limits (size bias_buf)
#   max_line_buf_cols / _rows   line-buffer column / circular-row capacity
#                               (power of 2; cols caps ow_per_tile, not in_w)
#   max_acc_persist_entries     URAM partial-output accumulator that survives
#                               across ic-tiles: one output chunk's
#                               chunk_oh x out_w x out_ch (padded to kTileM)
#   max_m_per_group             mt-tiles cached together in the weight slab
# ---------------------------------------------------------------------------
function(conv_load_constants platform_json out_var_prefix)
    if(NOT EXISTS "${platform_json}")
        message(FATAL_ERROR
            "ConvKernel: platform file not found: ${platform_json}")
    endif()
    file(READ "${platform_json}" _pjson)
    set(_keys "tile_m;tile_ic;max_kh;max_kw;max_in_ch;max_out_ch;max_line_buf_cols;max_line_buf_rows;max_acc_persist_entries;max_m_per_group")
    set(_var_names "TILE_M;TILE_IC;MAX_KH;MAX_KW;MAX_IN_CH;MAX_OUT_CH;MAX_LINE_BUF_COLS;MAX_LINE_BUF_ROWS;MAX_ACC_PERSIST_ENTRIES;MAX_M_PER_GROUP")
    foreach(_k _vn IN ZIP_LISTS _keys _var_names)
        string(JSON _val ERROR_VARIABLE _err GET "${_pjson}" kernels conv "${_k}")
        if(_err)
            message(FATAL_ERROR
                "ConvKernel: '${platform_json}' is missing "
                "kernels.conv.${_k} (CMake error: ${_err})")
        endif()
        set("${out_var_prefix}_${_vn}" "${_val}" PARENT_SCOPE)
    endforeach()
endfunction()
conv_load_constants("${AXI_DEFAULT_PLATFORM_JSON}" CONV)
message(STATUS "ConvKernel: constants from ${AXI_DEFAULT_PLATFORM_JSON}: "
               "TILE_M=${CONV_TILE_M} TILE_IC=${CONV_TILE_IC} "
               "MAX_KH=${CONV_MAX_KH} MAX_KW=${CONV_MAX_KW} "
               "MAX_IN_CH=${CONV_MAX_IN_CH} MAX_OUT_CH=${CONV_MAX_OUT_CH} "
               "MAX_LINE_BUF_COLS=${CONV_MAX_LINE_BUF_COLS} "
               "MAX_LINE_BUF_ROWS=${CONV_MAX_LINE_BUF_ROWS} "
               "MAX_ACC_PERSIST_ENTRIES=${CONV_MAX_ACC_PERSIST_ENTRIES} "
               "MAX_M_PER_GROUP=${CONV_MAX_M_PER_GROUP}")

# Re-run cmake configure automatically if a platform JSON changes, so the
# generated Config.h tracks JSON edits without a manual cmake re-run.
set_property(DIRECTORY APPEND PROPERTY
    CMAKE_CONFIGURE_DEPENDS "${AXI_DEFAULT_PLATFORM_JSON}")
foreach(_PF IN LISTS AXI_PLATFORM_FILES)
    set_property(DIRECTORY APPEND PROPERTY
        CMAKE_CONFIGURE_DEPENDS "${_PF}")
endforeach()

# ---------------------------------------------------------------------------
# Generate Config.h into the build directory
# ---------------------------------------------------------------------------
configure_file(
    include/Config.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/Config.h
    @ONLY
)

# ---------------------------------------------------------------------------
# Kernel library — ConvKernel.cpp (the RTL kernel's reference model, the
# retired HLS kernel) compiled as a static lib for the C++ simulation targets.
# The hardware build's ConvKernel is kernels/conv_rtl (doc/plans/CONV_RTL_PLAN.md).
# ---------------------------------------------------------------------------
add_library(conv_kernel STATIC
    kernel/ConvKernel.cpp
)
target_include_directories(conv_kernel
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include  # ConvKernel.h
        ${CMAKE_CURRENT_BINARY_DIR}          # Config.h
)
if(_CONV_HAVE_HLS)
    target_include_directories(conv_kernel SYSTEM PUBLIC ${HLS_INCLUDE_DIR})
endif()
target_compile_options(conv_kernel PRIVATE -Wall -Wextra -O2 -Wno-unknown-pragmas)

# No forced-float library: the 128-bit ports pack eight 16-bit elements per
# word (kXPortElems, the row / weight buffers and their static_asserts), so
# ConvKernel.cpp does not build with a 32-bit Data_t.

# ---------------------------------------------------------------------------
# TestConvRef — C++ simulation test, no Vitis HLS synthesis required.
#
# Compiles and runs entirely with standard GCC.  Links against conv_kernel
# (ap_fixed<16,8>; the Vitis HLS headers are required).
# ---------------------------------------------------------------------------
add_executable(TestConvRef
    test/TestConvSim.cpp
)
target_link_libraries(TestConvRef PRIVATE conv_kernel)
target_compile_options(TestConvRef PRIVATE -Wall -Wextra -O2)

enable_testing()
add_test(NAME TestConvRef   COMMAND TestConvRef)

# TestConvGrid — MAC-array unit test on include/ConvMacGrid.h alone (no
# geometry, no DDR model); the gate for changes to the grid itself.
add_executable(TestConvGrid test/TestConvGrid.cpp)
target_link_libraries(TestConvGrid PRIVATE conv_kernel)
target_compile_options(TestConvGrid PRIVATE -Wall -Wextra -O2 -Wno-unknown-pragmas)
add_test(NAME TestConvGrid  COMMAND TestConvGrid)
# Randomised-geometry sweep (bit-exact vs the naive oracle) — covers the
# interactions between chunking / M-grouping / ow-tiling / padding that the
# named cases exercise one at a time.  ~1 min.
add_test(NAME TestConvSweep COMMAND TestConvRef --sweep 300)

# ---------------------------------------------------------------------------
# gen_conv_test_data — regenerate hex fixtures for the HDL testbench.
#
# Runs TestConvRef in --dump-data mode, writing per-test x/w/b/y_ref hex
# files plus manifest.txt to ${CMAKE_BINARY_DIR}/conv_test_data/.  The HDL
# testbench loads these via $readmemh, so HW-level tests stay bit-identical
# to the C++ verification (same kSeed, same ref_conv() oracle).
#
# Requires the ap_fixed build (CONV_HAVE_APFIXED=ON via Vitis HLS headers).
# ---------------------------------------------------------------------------
set(CONV_TEST_DATA_DIR "${CMAKE_BINARY_DIR}/conv_test_data"
    CACHE PATH "Output directory for ConvKernel HDL testbench fixtures")

if(_CONV_HAVE_HLS)
    add_custom_target(gen_conv_test_data
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CONV_TEST_DATA_DIR}"
        COMMAND $<TARGET_FILE:TestConvRef> --dump-data "${CONV_TEST_DATA_DIR}"
        DEPENDS TestConvRef
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
        COMMENT "Generating ConvKernel HDL test fixtures → ${CONV_TEST_DATA_DIR}"
        VERBATIM)
    message(STATUS "ConvKernel: gen_conv_test_data → ${CONV_TEST_DATA_DIR}")
endif()

message(STATUS "ConvKernel: targets — conv_kernel, TestConvRef")
message(STATUS "ConvKernel: Data_t=${CONV_DATA_TYPE}  "
               "AccData_t=${CONV_ACC_DATA_TYPE}  "
               "TILE_M=${CONV_TILE_M}  TILE_IC=${CONV_TILE_IC}  "
               "MAX_KH=${CONV_MAX_KH}  MAX_KW=${CONV_MAX_KW}  "
               "MAX_IN_CH=${CONV_MAX_IN_CH}  MAX_OUT_CH=${CONV_MAX_OUT_CH}  "
               "MAX_ACC_PERSIST_ENTRIES=${CONV_MAX_ACC_PERSIST_ENTRIES}  "
               "MAX_M_PER_GROUP=${CONV_MAX_M_PER_GROUP}")
