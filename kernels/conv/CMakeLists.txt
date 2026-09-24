cmake_minimum_required(VERSION 3.19)
project(ConvKernel VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Locate Vitis HLS headers (ap_fixed.h) for fixed-point C++ simulation.
# The headers are header-only — no HLS synthesis flow is needed for C-sim.
# If not found, Data_t falls back to float and tests still verify correctness,
# but saturation behaviour won't be tested.
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
                   "using float fallback (no saturation tests)")
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
#   max_acc_persist_entries     out_w*out_ch bound for the URAM partial-output
#                               accumulator that survives across ic-tiles
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
# Kernel library — ConvKernel.cpp compiled as a static lib so that both
# the C++ simulation target and the future HLS synthesis target can link it.
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

# ---------------------------------------------------------------------------
# Float kernel library — same ConvKernel.cpp compiled with Data_t=float.
# Used by any future float-reference tests without the ap_fixed dependency.
#
# A separate float_config/Config.h is generated so the include path override
# works without touching the primary Config.h.
# ---------------------------------------------------------------------------
set(_SAVE_DATA_TYPE          "${CONV_DATA_TYPE}")
set(_SAVE_ACC_DATA_TYPE      "${CONV_ACC_DATA_TYPE}")
set(_SAVE_DATA_TYPE_INCLUDE  "${CONV_DATA_TYPE_INCLUDE}")
set(_SAVE_HAVE_APFIXED_DEFINE "${CONV_HAVE_APFIXED_DEFINE}")

set(CONV_DATA_TYPE          "float")
set(CONV_ACC_DATA_TYPE      "float")
set(CONV_DATA_TYPE_INCLUDE  "")
set(CONV_HAVE_APFIXED_DEFINE "")

configure_file(
    include/Config.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/float_config/Config.h
    @ONLY
)

set(CONV_DATA_TYPE          "${_SAVE_DATA_TYPE}")
set(CONV_ACC_DATA_TYPE      "${_SAVE_ACC_DATA_TYPE}")
set(CONV_DATA_TYPE_INCLUDE  "${_SAVE_DATA_TYPE_INCLUDE}")
set(CONV_HAVE_APFIXED_DEFINE "${_SAVE_HAVE_APFIXED_DEFINE}")

add_library(conv_kernel_float STATIC
    kernel/ConvKernel.cpp
)
# float_config must come before include/ so its Config.h shadows the primary one.
target_include_directories(conv_kernel_float
    PUBLIC
        ${CMAKE_CURRENT_BINARY_DIR}/float_config  # float Config.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include       # ConvKernel.h
)
target_compile_options(conv_kernel_float PRIVATE -Wall -Wextra -O2 -Wno-unknown-pragmas)

# ---------------------------------------------------------------------------
# TestConvRef — C++ simulation test, no Vitis HLS synthesis required.
#
# Compiles and runs entirely with standard GCC.  Links against conv_kernel
# (ap_fixed<16,8> when HLS headers are available, float otherwise).
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

message(STATUS "ConvKernel: targets — conv_kernel, conv_kernel_float, TestConvRef")
message(STATUS "ConvKernel: Data_t=${CONV_DATA_TYPE}  "
               "AccData_t=${CONV_ACC_DATA_TYPE}  "
               "TILE_M=${CONV_TILE_M}  TILE_IC=${CONV_TILE_IC}  "
               "MAX_KH=${CONV_MAX_KH}  MAX_KW=${CONV_MAX_KW}  "
               "MAX_IN_CH=${CONV_MAX_IN_CH}  MAX_OUT_CH=${CONV_MAX_OUT_CH}  "
               "MAX_ACC_PERSIST_ENTRIES=${CONV_MAX_ACC_PERSIST_ENTRIES}  "
               "MAX_M_PER_GROUP=${CONV_MAX_M_PER_GROUP}")

# ---------------------------------------------------------------------------
# HLS synthesis targets — one per platform JSON file in platforms/.
#
# Requires Vitis HLS (vitis-run for 2024.x+, vitis_hls for older versions).
# Each JSON must have a "part" field; "board" and "clock" are optional.
#
# Generated targets:
#   make synthesize_<name>   — C synthesis + ip_catalog export for that part
# ---------------------------------------------------------------------------
find_program(CONV_VITIS_HLS NAMES vitis-run vitis_hls
    HINTS
        $ENV{XILINX_HLS}/bin
        $ENV{XILINX_VITIS}/bin
        /mnt/data/xilinx/2025.2/Vitis/bin
    DOC "Vitis HLS executable (vitis-run for 2024.x+, vitis_hls for older)")

if(CONV_VITIS_HLS)
    if(CONV_VITIS_HLS MATCHES "vitis-run")
        set(CONV_HLS_TCL_FLAG "--tcl")
    else()
        set(CONV_HLS_TCL_FLAG "-f")
    endif()
    message(STATUS "ConvKernel: HLS synthesis enabled via ${CONV_VITIS_HLS} ${CONV_HLS_TCL_FLAG}")
else()
    message(STATUS "ConvKernel: vitis_hls / vitis-run not found — synthesis targets disabled")
endif()

if(CONV_VITIS_HLS)
    set(CONV_KERNEL_SRC "${CMAKE_CURRENT_SOURCE_DIR}/kernel/ConvKernel.cpp")

    set(_CONV_PLATFORM_FILES "${AXI_PLATFORM_FILES}")
    foreach(_CONV_PLATFORM_FILE IN LISTS _CONV_PLATFORM_FILES)
        get_filename_component(_pname "${_CONV_PLATFORM_FILE}" NAME_WE)
        file(READ "${_CONV_PLATFORM_FILE}" _pjson)

        # Required: FPGA part string
        string(JSON _p_part ERROR_VARIABLE _e1 GET "${_pjson}" part)
        if(_e1)
            message(WARNING "ConvKernel platform '${_pname}': missing 'part' — skipping.")
            continue()
        endif()

        # Optional: board identifier
        string(JSON _p_board ERROR_VARIABLE _e GET "${_pjson}" board)
        if(_e)
            set(_p_board "")
        endif()

        # Optional: target clock in MHz (default 300)
        string(JSON _p_clock ERROR_VARIABLE _e GET "${_pjson}" clock)
        if(_e)
            set(_p_clock "300")
        endif()

        # Per-platform build directory
        set(_p_build_dir "${CMAKE_CURRENT_BINARY_DIR}/${_pname}")
        file(MAKE_DIRECTORY "${_p_build_dir}")

        # Per-platform conv constants — drives this platform's Config.h
        # rather than the C-sim default platform's values, so each
        # synthesised IP uses bounds tuned for that platform.  Stash the
        # C-sim values so we can restore them after configure_file().
        set(_save_TILE_M                  "${CONV_TILE_M}")
        set(_save_TILE_IC                 "${CONV_TILE_IC}")
        set(_save_MAX_KH                  "${CONV_MAX_KH}")
        set(_save_MAX_KW                  "${CONV_MAX_KW}")
        set(_save_MAX_IN_CH               "${CONV_MAX_IN_CH}")
        set(_save_MAX_OUT_CH              "${CONV_MAX_OUT_CH}")
        set(_save_MAX_LINE_BUF_COLS       "${CONV_MAX_LINE_BUF_COLS}")
        set(_save_MAX_LINE_BUF_ROWS       "${CONV_MAX_LINE_BUF_ROWS}")
        set(_save_MAX_ACC_PERSIST_ENTRIES "${CONV_MAX_ACC_PERSIST_ENTRIES}")
        set(_save_MAX_M_PER_GROUP         "${CONV_MAX_M_PER_GROUP}")

        conv_load_constants("${_CONV_PLATFORM_FILE}" CONV)

        # Generate a platform-specific Config.h using these constants.
        configure_file(
            "${CMAKE_CURRENT_SOURCE_DIR}/include/Config.h.in"
            "${_p_build_dir}/Config.h"
            @ONLY)

        set(CONV_TILE_M                  "${_save_TILE_M}")
        set(CONV_TILE_IC                 "${_save_TILE_IC}")
        set(CONV_MAX_KH                  "${_save_MAX_KH}")
        set(CONV_MAX_KW                  "${_save_MAX_KW}")
        set(CONV_MAX_IN_CH               "${_save_MAX_IN_CH}")
        set(CONV_MAX_OUT_CH              "${_save_MAX_OUT_CH}")
        set(CONV_MAX_LINE_BUF_COLS       "${_save_MAX_LINE_BUF_COLS}")
        set(CONV_MAX_LINE_BUF_ROWS       "${_save_MAX_LINE_BUF_ROWS}")
        set(CONV_MAX_ACC_PERSIST_ENTRIES "${_save_MAX_ACC_PERSIST_ENTRIES}")
        set(CONV_MAX_M_PER_GROUP         "${_save_MAX_M_PER_GROUP}")

        if(_p_board)
            set(CONV_BOARD_OPTION "set_part {${_p_part}} -board {${_p_board}}")
        else()
            set(CONV_BOARD_OPTION "set_part {${_p_part}}")
        endif()

        set(CONV_HLS_PROJECT_NAME  "conv_${_pname}")
        set(CONV_SRC_SYNTHESIS     "${CONV_KERNEL_SRC}")
        set(CONV_SYNTHESIS_FLAGS
            "-O3 -I${CMAKE_CURRENT_SOURCE_DIR}/include -I${_p_build_dir}")
        set(CONV_TARGET_CLOCK_TCL  "${_p_clock}")
        set(CONV_IP_OUTPUT_DIR     "${_p_build_dir}/ip_catalog")

        configure_file(
            "${CMAKE_CURRENT_SOURCE_DIR}/scripts/Synthesis.tcl.in"
            "${_p_build_dir}/synthesize_${_pname}.tcl"
            @ONLY)

        # Clean stale HLS project + IP catalog so old Verilog from a prior
        # run isn't re-exported into the IP archive (which then breaks
        # downstream Vivado packaging when source files have been removed
        # or renamed since the last synthesis).
        add_custom_target(
            "synthesize_conv_${_pname}"
            COMMAND ${CMAKE_COMMAND} -E rm -rf
                    "${_p_build_dir}/${CONV_HLS_PROJECT_NAME}"
                    "${CONV_IP_OUTPUT_DIR}"
            COMMAND ${CONV_VITIS_HLS} ${CONV_HLS_TCL_FLAG}
                    "${_p_build_dir}/synthesize_${_pname}.tcl"
            WORKING_DIRECTORY "${_p_build_dir}"
            COMMENT "HLS synthesis: ConvKernel for '${_pname}' (${_p_part} @ ${_p_clock} MHz)"
            DEPENDS
                "${CONV_KERNEL_SRC}"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/ConvKernel.h"
                "${_p_build_dir}/Config.h")

        message(STATUS "ConvKernel: synthesis target 'synthesize_${_pname}' "
                       "(${_p_part} @ ${_p_clock} MHz)")

        # -------------------------------------------------------------------
        # cosim_conv_<name> — C synthesis + RTL co-simulation in ONE run.
        #
        # csynth_design and cosim_design are sequenced in a single TCL (see
        # Cosim.tcl.in), so one vitis-run invocation produces a component
        # whose run summary covers both — `vitis -a <component>` then shows
        # the cosim report (measured latency/II + actual hls::stream FIFO
        # depths from -enable_dataflow_profiling) alongside synthesis.
        #
        # Separate component (conv_<name>_cosim) so it never clobbers the
        # synthesize target's conv_<name> component / exported IP.  cosim
        # RTL-simulates the whole TestConvSim.cpp test bench, so this target
        # is slow (minutes) and is intentionally NOT a dependency of anything.
        # -------------------------------------------------------------------
        set(CONV_COSIM_COMPONENT "conv_${_pname}_cosim")
        set(CONV_TB_SYNTHESIS    "${CMAKE_CURRENT_SOURCE_DIR}/test/TestConvSim.cpp")

        configure_file(
            "${CMAKE_CURRENT_SOURCE_DIR}/scripts/Cosim.tcl.in"
            "${_p_build_dir}/cosim_${_pname}.tcl"
            @ONLY)

        add_custom_target(
            "cosim_conv_${_pname}"
            COMMAND ${CMAKE_COMMAND} -E rm -rf
                    "${_p_build_dir}/${CONV_COSIM_COMPONENT}"
            COMMAND ${CONV_VITIS_HLS} ${CONV_HLS_TCL_FLAG}
                    "${_p_build_dir}/cosim_${_pname}.tcl"
            WORKING_DIRECTORY "${_p_build_dir}"
            COMMENT "HLS synth + cosim: ConvKernel for '${_pname}' (${_p_part} @ ${_p_clock} MHz)"
            DEPENDS
                "${CONV_KERNEL_SRC}"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/ConvKernel.h"
                "${CMAKE_CURRENT_SOURCE_DIR}/test/TestConvSim.cpp"
                "${_p_build_dir}/Config.h")

        message(STATUS "ConvKernel: cosim target 'cosim_conv_${_pname}' "
                       "(${_p_part} @ ${_p_clock} MHz)")
    endforeach()
endif()
