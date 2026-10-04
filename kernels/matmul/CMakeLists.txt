cmake_minimum_required(VERSION 3.19)
project(MatmulKernel VERSION 1.0 LANGUAGES CXX)

# The platform JSONs (AXI_PLATFORM_FILES, AXI_DEFAULT_PLATFORM_JSON): set by the
# top-level project, or here when this kernel is configured on its own.
list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../cmake")
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
        /mnt/data/xilinx/2025.2/Vitis/include
    NO_DEFAULT_PATH
    DOC "Vitis HLS include directory containing ap_fixed.h"
)

if(HLS_INCLUDE_DIR)
    message(STATUS "MatmulKernel: Vitis HLS headers found at ${HLS_INCLUDE_DIR}")
    set(_MM_HAVE_HLS TRUE)
else()
    message(STATUS "MatmulKernel: Vitis HLS headers not found — "
                   "the C-sim targets need them and will not build")
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
# Kernel library — MatmulKernel.cpp (the HLS kernel's C++, the reference
# model of the RTL kernel) compiled as a static lib for the C++ tests.
# ---------------------------------------------------------------------------
add_library(matmul_kernel STATIC
    kernel/MatmulKernel.cpp
)
target_include_directories(matmul_kernel
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include  # MatmulKernel.h
        ${CMAKE_CURRENT_BINARY_DIR}          # Config.h
)
if(_MM_HAVE_HLS)
    target_include_directories(matmul_kernel SYSTEM PUBLIC ${HLS_INCLUDE_DIR})
    # MATMUL_HAVE_APFIXED is already defined in Config.h via MM_HAVE_APFIXED_DEFINE;
    # no need to pass it again via -D (that would cause a redefinition warning).
endif()
target_compile_options(matmul_kernel PRIVATE -Wall -Wextra -O2 -Wno-unknown-pragmas)

# ---------------------------------------------------------------------------
# TestMatmulRef — C++ simulation test: the kernel function (MatmulKernel.cpp)
# against ref_matmul_2d(); compiles and runs entirely with standard GCC.
# ---------------------------------------------------------------------------
add_executable(TestMatmulRef
    test/TestMatmulSim.cpp
)
target_link_libraries(TestMatmulRef PRIVATE matmul_kernel)
target_compile_options(TestMatmulRef PRIVATE -Wall -Wextra -O2)

# Register with CTest so "ctest" runs it.
enable_testing()
add_test(NAME TestMatmulRef COMMAND TestMatmulRef)

# ---------------------------------------------------------------------------
# gen_matmul_test_data — regenerate hex fixtures for the HDL testbench.
#
# Runs TestMatmulRef in --dump-data mode, writing per-test A/B/C_ref hex
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
        COMMAND $<TARGET_FILE:TestMatmulRef> --dump-data "${MATMUL_TEST_DATA_DIR}"
        DEPENDS TestMatmulRef
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
        COMMENT "Generating MatmulKernel HDL test fixtures → ${MATMUL_TEST_DATA_DIR}"
        VERBATIM)
    message(STATUS "MatmulKernel: gen_matmul_test_data → ${MATMUL_TEST_DATA_DIR}")
endif()

# ---------------------------------------------------------------------------
# TestMatmulBlas — validates the configured kernel (ap_fixed<16,8> by
# default) bit-exactly against cblas_sgemm on inputs chosen so that both
# sides accumulate exactly (see the header of test/TestMatmulBlas.cpp).
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
    message(STATUS "MatmulKernel: BLAS found — TestMatmulBlas enabled "
                   "(cblas.h at ${CBLAS_INCLUDE_DIR})")
    add_executable(TestMatmulBlas
        test/TestMatmulBlas.cpp
    )
    target_link_libraries(TestMatmulBlas PRIVATE matmul_kernel ${BLAS_LIBRARIES})
    target_include_directories(TestMatmulBlas PRIVATE ${CBLAS_INCLUDE_DIR})
    target_compile_options(TestMatmulBlas PRIVATE -Wall -Wextra -O2 -Wno-unknown-pragmas)
    add_test(NAME TestMatmulBlas COMMAND TestMatmulBlas)
else()
    message(STATUS "MatmulKernel: BLAS not found — TestMatmulBlas skipped")
endif()

message(STATUS "MatmulKernel: targets — matmul_kernel, TestMatmulRef")
message(STATUS "MatmulKernel: Data_t=${MM_DATA_TYPE}  "
               "AccData_t=${MM_ACC_DATA_TYPE}  "
               "TILE_N=${MM_TILE_N}  TILE_M=${MM_TILE_M}  "
               "TILE_K=${MM_TILE_K}  MAX_K=${MM_MAX_K}  GEMV_MAX_M=${MM_GEMV_MAX_M}")

# The HLS synthesis (synthesize_matmul_<platform>, cosim_matmul_<platform>) was
# retired: the hardware build packages the SystemVerilog MatmulKernel
# (kernels/matmul_rtl, doc/plans/MATMUL_RTL_PLAN.md phase 4).  This C++ model
# stays as the oracle (TestMatmulRef, TestMatmulBlas) and writes the fixtures
# the RTL testbenches check (gen_matmul_test_data).
