# Copyright 2025 GradeBuilder SL
# Modifications copyright 2026 Ivan Gubochkin
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.19)
project(MatmulKernelModel VERSION 1.0 LANGUAGES CXX)

# ---------------------------------------------------------------------------
# MatmulKernel's C++ reference model (src/MatmulKernel.cpp): the oracle of the
# SystemVerilog kernel's testbench (kernels/matmul_rtl, TestMatmulRtl), the
# generator of the fixtures the RTL tests check (hw/test_data/matmul_test_data)
# and their C-simulation tests.  It began as the Vitis HLS kernel (its
# synthesis retired in doc/plans/MATMUL_RTL_PLAN.md phase 4) and still compiles
# against the Vitis HLS headers (ap_fixed, hls::stream).
# ---------------------------------------------------------------------------

# The platform JSONs (AXI_PLATFORM_FILES, AXI_DEFAULT_PLATFORM_JSON): set by the
# top-level project, or here when this kernel is configured on its own.
list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake")
include(AxiPlatform)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Locate Vitis HLS headers (ap_fixed.h) for fixed-point C++ simulation.
#
# The headers are header-only and work with standard GCC — no HLS synthesis
# flow is needed.  If not found, Data_t falls back to float, but the kernel
# headers include ap_int.h / hls_burst_maxi.h unconditionally, so nothing
# builds without them (the float Data_t is only the configure default).
# ---------------------------------------------------------------------------
find_path(HLS_INCLUDE_DIR NAMES ap_fixed.h
    HINTS
        $ENV{XILINX_HLS}/include
        $ENV{XILINX_VITIS}/include
    NO_DEFAULT_PATH
    DOC "Vitis HLS include directory containing ap_fixed.h"
)

if(HLS_INCLUDE_DIR)
    message(STATUS "MatmulKernel: Vitis HLS headers found at ${HLS_INCLUDE_DIR}")
    set(_MM_HAVE_HLS TRUE)
else()
    message(STATUS "MatmulKernel: Vitis HLS headers not found — "
                   "the C-sim targets need them and will not build "
                   "(source <Xilinx>/2025.2/Vitis/settings64.sh: it sets XILINX_HLS)")
    set(_MM_HAVE_HLS FALSE)
endif()

# ---------------------------------------------------------------------------
# Data type configuration
# ---------------------------------------------------------------------------
if(_MM_HAVE_HLS)
    set(MM_DATA_TYPE     "ap_fixed<16,8>"  CACHE STRING "Element type")
    set(MM_ACC_DATA_TYPE "ap_fixed<32,16>" CACHE STRING "Accumulator type")
    set(MM_DATA_TYPE_INCLUDE  "#include \"ap_fixed.h\"")
    set(MM_HAVE_APFIXED_DEFINE "#define MATMUL_HAVE_APFIXED")
else()
    set(MM_DATA_TYPE     "float"  CACHE STRING "Element type")
    set(MM_ACC_DATA_TYPE "double" CACHE STRING "Accumulator type")
    set(MM_DATA_TYPE_INCLUDE  "")
    set(MM_HAVE_APFIXED_DEFINE "")
endif()

# ---------------------------------------------------------------------------
# Compile-time kernel bounds — read from the platform JSON.
#
# These are the single source of truth, defined under "kernels.matmul" in
# each platforms/<name>.json — there are NO CMake CACHE defaults; the JSON is
# authoritative.  matmul_load_constants() errors out if any field is missing
# rather than silently defaulting.  Field reference (the C++ model's; the RTL
# kernel shares tile_m = the packed-B DDR tile, max_k and gemv_max_m > 0 =
# the image path; tile_n / tile_k are the C++ model's internal tiling):
#   tile_n   row interleave depth (power of 2; >= MAC latency for II=1)
#   tile_m   parallel output columns per cycle (power of 2); the packed-B
#            tile width of the DDR layout
#   tile_k   on-chip B-buffer row count (power of 2)
#   max_k    compile-time a_buf row-length limit; K > max_k is rejected at
#            scheduling
#   gemv_max_m  GEMV streaming mode (gemv_kw != 0): on-chip accumulator
#            depth, the widest column chunk one pass over B covers (a
#            multiple of 8; wider m is split into chunks).  0 = no GEMV path
# ---------------------------------------------------------------------------
function(matmul_load_constants platform_json out_var_prefix)
    if(NOT EXISTS "${platform_json}")
        message(FATAL_ERROR
            "MatmulKernel: platform file not found: ${platform_json}")
    endif()
    file(READ "${platform_json}" _pjson)
    set(_keys "tile_n;tile_m;tile_k;max_k;gemv_max_m")
    set(_var_names "TILE_N;TILE_M;TILE_K;MAX_K;GEMV_MAX_M")
    foreach(_k _vn IN ZIP_LISTS _keys _var_names)
        string(JSON _val ERROR_VARIABLE _err GET "${_pjson}" kernels matmul "${_k}")
        if(_err)
            message(FATAL_ERROR
                "MatmulKernel: '${platform_json}' is missing "
                "kernels.matmul.${_k} (CMake error: ${_err})")
        endif()
        set("${out_var_prefix}_${_vn}" "${_val}" PARENT_SCOPE)
    endforeach()
endfunction()
matmul_load_constants("${AXI_DEFAULT_PLATFORM_JSON}" MM)
message(STATUS "MatmulKernel: constants from ${AXI_DEFAULT_PLATFORM_JSON}: "
               "TILE_N=${MM_TILE_N} TILE_M=${MM_TILE_M} "
               "TILE_K=${MM_TILE_K} MAX_K=${MM_MAX_K} GEMV_MAX_M=${MM_GEMV_MAX_M}")

# Re-run cmake configure automatically if a platform JSON changes, so the
# generated Config.h tracks JSON edits without a manual cmake re-run.
set_property(DIRECTORY APPEND PROPERTY
    CMAKE_CONFIGURE_DEPENDS "${AXI_DEFAULT_PLATFORM_JSON}")

# ---------------------------------------------------------------------------
# Generate Config.h into the build directory
# ---------------------------------------------------------------------------
configure_file(
    include/Config.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/Config.h
    @ONLY
)

# ---------------------------------------------------------------------------
# The model library — src/MatmulKernel.cpp compiled as a static lib for the
# C++ tests.
# ---------------------------------------------------------------------------
add_library(matmul_model STATIC
    src/MatmulKernel.cpp
)
target_include_directories(matmul_model
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include  # MatmulKernel.h
        ${CMAKE_CURRENT_BINARY_DIR}          # Config.h
)
if(_MM_HAVE_HLS)
    target_include_directories(matmul_model SYSTEM PUBLIC ${HLS_INCLUDE_DIR})
    # MATMUL_HAVE_APFIXED is already defined in Config.h via MM_HAVE_APFIXED_DEFINE;
    # no need to pass it again via -D (that would cause a redefinition warning).
endif()
target_compile_options(matmul_model PRIVATE -Wall -Wextra -O2 -Wno-unknown-pragmas)

# ---------------------------------------------------------------------------
# TestMatmulModel — C++ simulation test: the kernel function (MatmulKernel.cpp)
# against ref_matmul_2d(); compiles and runs entirely with standard GCC.
# ---------------------------------------------------------------------------
add_executable(TestMatmulModel
    test/TestMatmulModel.cpp
)
target_link_libraries(TestMatmulModel PRIVATE matmul_model)
target_compile_options(TestMatmulModel PRIVATE -Wall -Wextra -O2)

# Register with CTest so "ctest" runs it.
enable_testing()
add_test(NAME TestMatmulModel COMMAND TestMatmulModel)

# ---------------------------------------------------------------------------
# gen_matmul_test_data — regenerate hex fixtures for the HDL testbench.
#
# Runs TestMatmulModel in --dump-data mode, writing per-test A/B/C_ref hex
# files plus manifest.txt to ${CMAKE_BINARY_DIR}/matmul_test_data/.  The
# HDL testbench loads these via $readmemh, so HW-level tests stay
# bit-identical to the C++ verification (same kSeed, same ref_matmul_2d()
# oracle).
#
# Requires the ap_fixed build (MATMUL_HAVE_APFIXED=ON via Vitis HLS headers).
# ---------------------------------------------------------------------------
set(MATMUL_TEST_DATA_DIR "${CMAKE_BINARY_DIR}/matmul_test_data"
    CACHE PATH "Output directory for MatmulKernel HDL testbench fixtures")

if(_MM_HAVE_HLS)
    add_custom_target(gen_matmul_test_data
        COMMAND ${CMAKE_COMMAND} -E make_directory "${MATMUL_TEST_DATA_DIR}"
        COMMAND $<TARGET_FILE:TestMatmulModel> --dump-data "${MATMUL_TEST_DATA_DIR}"
        DEPENDS TestMatmulModel
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
        COMMENT "Generating MatmulKernel HDL test fixtures → ${MATMUL_TEST_DATA_DIR}"
        VERBATIM)
    message(STATUS "MatmulKernel: gen_matmul_test_data → ${MATMUL_TEST_DATA_DIR}")
endif()

# ---------------------------------------------------------------------------
# TestMatmulModelBlas — validates the configured kernel (ap_fixed<16,8> by
# default) bit-exactly against cblas_sgemm on inputs chosen so that both
# sides accumulate exactly (see the header of test/TestMatmulModelBlas.cpp).
#
# Requires libblas-dev (provides cblas.h and libblas.so).  Skipped silently
# if BLAS is not found so the build remains self-contained on bare systems.
# ---------------------------------------------------------------------------
find_package(BLAS)
find_path(CBLAS_INCLUDE_DIR NAMES cblas.h
    HINTS
        /usr/include/x86_64-linux-gnu
        /usr/include
    DOC "Directory containing cblas.h"
)

if(BLAS_FOUND AND CBLAS_INCLUDE_DIR)
    message(STATUS "MatmulKernel: BLAS found — TestMatmulModelBlas enabled "
                   "(cblas.h at ${CBLAS_INCLUDE_DIR})")
    add_executable(TestMatmulModelBlas
        test/TestMatmulModelBlas.cpp
    )
    target_link_libraries(TestMatmulModelBlas PRIVATE matmul_model ${BLAS_LIBRARIES})
    target_include_directories(TestMatmulModelBlas PRIVATE ${CBLAS_INCLUDE_DIR})
    target_compile_options(TestMatmulModelBlas PRIVATE -Wall -Wextra -O2 -Wno-unknown-pragmas)
    add_test(NAME TestMatmulModelBlas COMMAND TestMatmulModelBlas)
else()
    message(STATUS "MatmulKernel: BLAS not found — TestMatmulModelBlas skipped")
endif()

message(STATUS "MatmulKernel: targets — matmul_model, TestMatmulModel")
message(STATUS "MatmulKernel: Data_t=${MM_DATA_TYPE}  "
               "AccData_t=${MM_ACC_DATA_TYPE}  "
               "TILE_N=${MM_TILE_N}  TILE_M=${MM_TILE_M}  "
               "TILE_K=${MM_TILE_K}  MAX_K=${MM_MAX_K}  GEMV_MAX_M=${MM_GEMV_MAX_M}")
