cmake_minimum_required(VERSION 3.19)
project(VectorOPKernel)

# ---------------------------------------------------------------------------
# The Vitis HLS VectorOPKernel's C++ (kernel/VectorOP.cpp): the reference model
# of the SystemVerilog kernel the hardware build uses (kernels/vectorop_rtl),
# its testbench oracle, and the generator of the RTL fixtures
# (hw/test_data/vecop_test_data).  Its HLS synthesis was retired
# (doc/plans/VECTOROP_RTL_PLAN.md phase 3).
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Auto-detect Vitis HLS headers (ap_fixed.h)
# ---------------------------------------------------------------------------
find_path(VA_HLS_INCLUDE_DIR NAMES ap_fixed.h
    HINTS
        $ENV{XILINX_HLS}/include
        $ENV{XILINX_VITIS}/include
        /mnt/data/xilinx/2025.2/Vitis/include
    NO_DEFAULT_PATH
    DOC "Vitis HLS include directory containing ap_fixed.h"
)

if(VA_HLS_INCLUDE_DIR)
    message(STATUS "VectorOP: Vitis HLS headers found at ${VA_HLS_INCLUDE_DIR}")
    set(_VA_DATA_TYPE_DEFAULT "ap_fixed<16,8>")
else()
    message(STATUS "VectorOP: Vitis HLS headers not found — defaulting to float")
    set(_VA_DATA_TYPE_DEFAULT "float")
endif()

# Domain parameters
set(VA_DATA_TYPE "${_VA_DATA_TYPE_DEFAULT}" CACHE STRING
    "Element data type: float, double, half, uint8_t, or ap_fixed<W,I>.")

# ---------------------------------------------------------------------------
# Resolve data-type byte width
# ---------------------------------------------------------------------------
include(CheckTypeSize)
check_type_size(${VA_DATA_TYPE} _VA_TMP_WIDTH_${VA_DATA_TYPE})

if(NOT _VA_TMP_WIDTH_${VA_DATA_TYPE})
  if(VA_DATA_TYPE STREQUAL "half")
    set(_VA_TMP_WIDTH_${VA_DATA_TYPE} 2)
  elseif(VA_DATA_TYPE STREQUAL "uint8_t")
    set(_VA_TMP_WIDTH_${VA_DATA_TYPE} 1)
  elseif(VA_DATA_TYPE MATCHES "^ap_(u?)fixed<([0-9]+)")
    math(EXPR _bits  "${CMAKE_MATCH_2}")
    math(EXPR _bytes "(${_bits} + 7) / 8")
    set(_VA_TMP_WIDTH_${VA_DATA_TYPE} ${_bytes})
  else()
    message(FATAL_ERROR "Cannot determine size of data type '${VA_DATA_TYPE}'.")
  endif()
endif()

set(VA_DATA_WIDTH_BYTES "${_VA_TMP_WIDTH_${VA_DATA_TYPE}}")
math(EXPR VA_DATA_WIDTH_BITS "${VA_DATA_WIDTH_BYTES} * 8")

# Include directive for types that need a special header
if(VA_DATA_TYPE STREQUAL "half")
  set(VA_DATA_TYPE_INCLUDE "#include \"hls_half.h\"")
elseif(VA_DATA_TYPE MATCHES "^ap_(u?)fixed")
  set(VA_DATA_TYPE_INCLUDE "#include \"ap_fixed.h\"")
elseif(VA_DATA_TYPE STREQUAL "uint8_t")
  set(VA_DATA_TYPE_INCLUDE "#include <cstdint>")
else()
  set(VA_DATA_TYPE_INCLUDE "")
endif()

# ---------------------------------------------------------------------------
# Generate configuration header
# ---------------------------------------------------------------------------
configure_file(include/Config.h.in Config.h)

# ---------------------------------------------------------------------------
# Simulation (C++ test) build
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(VA_KERNEL_SRC ${CMAKE_CURRENT_SOURCE_DIR}/kernel/VectorOP.cpp)

add_library(vadd_kernel ${VA_KERNEL_SRC})
target_include_directories(vadd_kernel
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_BINARY_DIR})
if(VA_HLS_INCLUDE_DIR)
    target_include_directories(vadd_kernel SYSTEM PUBLIC ${VA_HLS_INCLUDE_DIR})
endif()
target_compile_options(vadd_kernel PRIVATE -Wall -Wextra -O2 -Wno-unknown-pragmas)

add_executable(TestSimulation test/TestSimulation.cpp)
target_link_libraries(TestSimulation PRIVATE vadd_kernel)
target_compile_options(TestSimulation PRIVATE -Wall -Wextra -O2)
if(VA_DATA_TYPE MATCHES "^ap_(u?)fixed<16")
    target_compile_definitions(TestSimulation PRIVATE VA_HAVE_APFIXED)
endif()

enable_testing()
add_test(NAME TestSimulation COMMAND TestSimulation)

# ---------------------------------------------------------------------------
# gen_vectorop_test_data — regenerate hex fixtures for the HDL testbench.
#
# Runs TestSimulation in --dump-data mode, writing per-test a/b/c_ref hex
# files plus manifest.txt to ${CMAKE_BINARY_DIR}/vectorop_test_data/.  The
# HDL testbench loads these via $readmemh, so HW-level tests stay
# bit-identical to the C++ verification (same kSeed, same ref_op() oracle).
#
# Requires the 16-bit ap_fixed build (VA_HAVE_APFIXED).
# ---------------------------------------------------------------------------
set(VA_TEST_DATA_DIR "${CMAKE_BINARY_DIR}/vectorop_test_data"
    CACHE PATH "Output directory for VectorOPKernel HDL testbench fixtures")

if(VA_DATA_TYPE MATCHES "^ap_(u?)fixed<16")
    add_custom_target(gen_vectorop_test_data
        COMMAND ${CMAKE_COMMAND} -E make_directory "${VA_TEST_DATA_DIR}"
        COMMAND $<TARGET_FILE:TestSimulation> --dump-data "${VA_TEST_DATA_DIR}"
        DEPENDS TestSimulation
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
        COMMENT "Generating VectorOPKernel HDL test fixtures → ${VA_TEST_DATA_DIR}"
        VERBATIM)
    message(STATUS "VectorOPKernel: gen_vectorop_test_data → ${VA_TEST_DATA_DIR}")
endif()

message(STATUS "VectorOPKernel: Data_t=${VA_DATA_TYPE}")
