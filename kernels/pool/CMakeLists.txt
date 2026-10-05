cmake_minimum_required(VERSION 3.19)
project(PoolingKernel VERSION 1.0 LANGUAGES CXX)

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
    message(STATUS "PoolingKernel: Vitis HLS headers found at ${HLS_INCLUDE_DIR}")
    set(_POOL_HAVE_HLS TRUE)
else()
    message(STATUS "PoolingKernel: Vitis HLS headers not found — "
                   "the C-sim targets need them and will not build")
    set(_POOL_HAVE_HLS FALSE)
endif()

# ---------------------------------------------------------------------------
# Data type configuration
# ---------------------------------------------------------------------------
if(_POOL_HAVE_HLS)
    set(POOL_DATA_TYPE     "ap_fixed<16,8>"  CACHE STRING "Element type")
    set(POOL_ACC_DATA_TYPE "ap_fixed<32,16>" CACHE STRING "Accumulator type")
    set(POOL_DATA_TYPE_INCLUDE  "#include \"ap_fixed.h\"")
    set(POOL_HAVE_APFIXED_DEFINE "#define POOL_HAVE_APFIXED")
    # Minimum representable values — used as MAX pool sentinels.
    # ap_fixed<16,8> min = -128; ap_fixed<32,16> min = -32768.
    # AP_WRAP (default) wraps on overflow, so we must stay in-range.
    set(POOL_DATA_MIN "-128.0f")
    set(POOL_ACC_MIN  "-32768.0f")
else()
    set(POOL_DATA_TYPE     "float"  CACHE STRING "Element type")
    set(POOL_ACC_DATA_TYPE "float"  CACHE STRING "Accumulator type")
    set(POOL_DATA_TYPE_INCLUDE  "")
    set(POOL_HAVE_APFIXED_DEFINE "")
    set(POOL_DATA_MIN "-1.0e30f")
    set(POOL_ACC_MIN  "-1.0e30f")
endif()

# ---------------------------------------------------------------------------
# Tile sizes and window limits — sourced from the platform JSON
# (platforms/<AXI_PLATFORM>.json, "kernels.pool" object).  These are the
# kernel's compile-time bounds: pool_h ≤ POOL_MAX_KH, line_buf rows ≥
# (pool_h-1)*dil_h+1, etc.  See doc/kernels/POOL_OPTIMISATION.md §4.
#
# No CACHE defaults — the JSON is the single source of truth so the C++
# kernel build and the Python validator (inference-scheduler) cannot drift
# apart.  To bump a value, edit the JSON; CMake re-reads on the next
# configure.  pool_load_constants() below errors out if any field is
# missing rather than silently defaulting.
# ---------------------------------------------------------------------------
function(pool_load_constants platform_json out_var_prefix)
    if(NOT EXISTS "${platform_json}")
        message(FATAL_ERROR
            "PoolingKernel: platform file not found: ${platform_json}")
    endif()
    file(READ "${platform_json}" _pjson)
    set(_keys "tile_c;max_kh;max_kw;max_line_buf_rows;max_line_buf_cols;ow_parallel")
    set(_var_names "TILE_C;MAX_KH;MAX_KW;MAX_LINE_BUF_ROWS;MAX_LINE_BUF_COLS;OW_PARALLEL")
    foreach(_k _vn IN ZIP_LISTS _keys _var_names)
        string(JSON _val ERROR_VARIABLE _err GET "${_pjson}" kernels pool "${_k}")
        if(_err)
            message(FATAL_ERROR
                "PoolingKernel: '${platform_json}' is missing "
                "kernels.pool.${_k} (CMake error: ${_err})")
        endif()
        set("${out_var_prefix}_${_vn}" "${_val}" PARENT_SCOPE)
    endforeach()
endfunction()

pool_load_constants("${AXI_DEFAULT_PLATFORM_JSON}" POOL)
message(STATUS "PoolingKernel: constants from ${AXI_DEFAULT_PLATFORM_JSON}: "
               "TILE_C=${POOL_TILE_C} MAX_KH=${POOL_MAX_KH} MAX_KW=${POOL_MAX_KW} "
               "MAX_LINE_BUF_ROWS=${POOL_MAX_LINE_BUF_ROWS} "
               "MAX_LINE_BUF_COLS=${POOL_MAX_LINE_BUF_COLS} "
               "OW_PARALLEL=${POOL_OW_PARALLEL}")

# Re-run cmake configure automatically if the platform JSON changes.  The
# generated Config.h and `make`-time rebuilds depend on JSON values; without
# this, edits would not regenerate Config.h until the user reruns cmake by
# hand.
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
# Kernel library — PoolingKernel.cpp (the RTL kernel's reference model)
# compiled as a static lib for the C++ simulation target.
# ---------------------------------------------------------------------------
add_library(pool_kernel STATIC
    kernel/PoolingKernel.cpp
)
target_include_directories(pool_kernel
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include  # PoolingKernel.h
        ${CMAKE_CURRENT_BINARY_DIR}          # Config.h
)
if(_POOL_HAVE_HLS)
    target_include_directories(pool_kernel SYSTEM PUBLIC ${HLS_INCLUDE_DIR})
endif()
target_compile_options(pool_kernel PRIVATE -Wall -Wextra -O2 -Wno-unknown-pragmas)

# ---------------------------------------------------------------------------
# Float kernel library — same PoolingKernel.cpp compiled with Data_t=float.
# Used by any future float-reference tests without the ap_fixed dependency.
# ---------------------------------------------------------------------------
set(_SAVE_DATA_TYPE           "${POOL_DATA_TYPE}")
set(_SAVE_ACC_DATA_TYPE       "${POOL_ACC_DATA_TYPE}")
set(_SAVE_DATA_TYPE_INCLUDE   "${POOL_DATA_TYPE_INCLUDE}")
set(_SAVE_HAVE_APFIXED_DEFINE "${POOL_HAVE_APFIXED_DEFINE}")
set(_SAVE_DATA_MIN            "${POOL_DATA_MIN}")
set(_SAVE_ACC_MIN             "${POOL_ACC_MIN}")

set(POOL_DATA_TYPE           "float")
set(POOL_ACC_DATA_TYPE       "float")
set(POOL_DATA_TYPE_INCLUDE   "")
set(POOL_HAVE_APFIXED_DEFINE "")
set(POOL_DATA_MIN            "-1.0e30f")
set(POOL_ACC_MIN             "-1.0e30f")

configure_file(
    include/Config.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/float_config/Config.h
    @ONLY
)

set(POOL_DATA_TYPE           "${_SAVE_DATA_TYPE}")
set(POOL_ACC_DATA_TYPE       "${_SAVE_ACC_DATA_TYPE}")
set(POOL_DATA_TYPE_INCLUDE   "${_SAVE_DATA_TYPE_INCLUDE}")
set(POOL_HAVE_APFIXED_DEFINE "${_SAVE_HAVE_APFIXED_DEFINE}")
set(POOL_DATA_MIN            "${_SAVE_DATA_MIN}")
set(POOL_ACC_MIN             "${_SAVE_ACC_MIN}")

add_library(pool_kernel_float STATIC
    kernel/PoolingKernel.cpp
)
# float_config must come before include/ so its Config.h shadows the primary one.
target_include_directories(pool_kernel_float
    PUBLIC
        ${CMAKE_CURRENT_BINARY_DIR}/float_config  # float Config.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include       # PoolingKernel.h
)
if(_POOL_HAVE_HLS)
    # PoolingKernel.h needs ap_int.h / hls_burst_maxi.h for the 128-bit x port.
    target_include_directories(pool_kernel_float SYSTEM PUBLIC ${HLS_INCLUDE_DIR})
endif()
target_compile_options(pool_kernel_float PRIVATE -Wall -Wextra -O2 -Wno-unknown-pragmas)

# ---------------------------------------------------------------------------
# TestPoolingSim — C++ simulation test, no Vitis HLS synthesis required.
#
# Compiles and runs entirely with standard GCC.  Links against pool_kernel
# (ap_fixed<16,8>; the Vitis HLS headers are required).
# ---------------------------------------------------------------------------
add_executable(TestPoolingSim
    test/TestPoolingSim.cpp
)
target_link_libraries(TestPoolingSim PRIVATE pool_kernel)
target_compile_options(TestPoolingSim PRIVATE -Wall -Wextra -O2)

enable_testing()
add_test(NAME TestPoolingSim COMMAND TestPoolingSim)

# ---------------------------------------------------------------------------
# gen_pool_test_data — regenerate hex fixtures for the HDL testbench.
#
# Runs TestPoolingSim in --dump-data mode, writing per-test x/y_ref hex
# files plus manifest.txt to ${CMAKE_BINARY_DIR}/pool_test_data/.  The
# HDL testbench loads these via $readmemh, so HW-level tests stay
# bit-identical to the C++ verification (same kSeed, same ref_pool_elem()
# oracle).
#
# Requires the ap_fixed build (POOL_HAVE_APFIXED=ON via Vitis HLS headers).
# ---------------------------------------------------------------------------
set(POOL_TEST_DATA_DIR "${CMAKE_BINARY_DIR}/pool_test_data"
    CACHE PATH "Output directory for PoolingKernel HDL testbench fixtures")

if(_POOL_HAVE_HLS)
    add_custom_target(gen_pool_test_data
        COMMAND ${CMAKE_COMMAND} -E make_directory "${POOL_TEST_DATA_DIR}"
        COMMAND $<TARGET_FILE:TestPoolingSim> --dump-data "${POOL_TEST_DATA_DIR}"
        DEPENDS TestPoolingSim
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
        COMMENT "Generating PoolingKernel HDL test fixtures → ${POOL_TEST_DATA_DIR}"
        VERBATIM)
    message(STATUS "PoolingKernel: gen_pool_test_data → ${POOL_TEST_DATA_DIR}")
endif()

message(STATUS "PoolingKernel: targets — pool_kernel, pool_kernel_float, TestPoolingSim")
message(STATUS "PoolingKernel: Data_t=${POOL_DATA_TYPE}  "
               "AccData_t=${POOL_ACC_DATA_TYPE}  "
               "TILE_C=${POOL_TILE_C}  "
               "MAX_KH=${POOL_MAX_KH}  MAX_KW=${POOL_MAX_KW}  "
               "MAX_LINE_BUF_ROWS=${POOL_MAX_LINE_BUF_ROWS}  "
               "MAX_LINE_BUF_COLS=${POOL_MAX_LINE_BUF_COLS}  "
               "OW_PARALLEL=${POOL_OW_PARALLEL}")

# The HLS kernel's synthesis (synthesize_pool_<platform>, cosim_pool_<platform>)
# was retired in doc/plans/POOL_RTL_PLAN.md phase 3: the hardware build's
# PoolingKernel is the SystemVerilog one (kernels/pool_rtl), which runs this
# C++ as its testbench oracle.
