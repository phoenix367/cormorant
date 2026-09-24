cmake_minimum_required(VERSION 3.19)
project(MatmulKernel VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Locate Vitis HLS headers (ap_fixed.h) for fixed-point C++ simulation.
#
# The headers are header-only and work with standard GCC — no HLS synthesis
# flow is needed.  If not found, Data_t falls back to float and tests still
# verify correctness, but saturation behaviour won't be tested.
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
                   "using float/double fallback (no saturation tests)")
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
# rather than silently defaulting.  Field reference:
#   tile_n   row interleave depth (power of 2; >= MAC latency for II=1)
#   tile_m   parallel output columns per cycle (power of 2)
#   tile_k   on-chip B-buffer row count (power of 2)
#   max_k    compile-time a_buf row-length limit; K > max_k is rejected at
#            scheduling
# ---------------------------------------------------------------------------
function(matmul_load_constants platform_json out_var_prefix)
    if(NOT EXISTS "${platform_json}")
        message(FATAL_ERROR
            "MatmulKernel: platform file not found: ${platform_json}")
    endif()
    file(READ "${platform_json}" _pjson)
    set(_keys "tile_n;tile_m;tile_k;max_k")
    set(_var_names "TILE_N;TILE_M;TILE_K;MAX_K")
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
               "TILE_K=${MM_TILE_K} MAX_K=${MM_MAX_K}")

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
# Kernel library — MatmulKernel.cpp compiled as a static lib so that both
# the C++ simulation target and the future HLS synthesis target can link it.
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
# Float kernel library — same MatmulKernel.cpp but always compiled with
# Data_t=float / AccData_t=float regardless of HLS availability.
#
# Used by TestMatmulBlas so that cblas_sgemm (float) can serve as the oracle.
# A separate float_config/Config.h is generated into the build directory so
# the include path override works without touching the primary Config.h.
# ---------------------------------------------------------------------------

# Generate float_config/Config.h by temporarily swapping the type variables.
set(_SAVE_DATA_TYPE          "${MM_DATA_TYPE}")
set(_SAVE_ACC_DATA_TYPE      "${MM_ACC_DATA_TYPE}")
set(_SAVE_DATA_TYPE_INCLUDE  "${MM_DATA_TYPE_INCLUDE}")
set(_SAVE_HAVE_APFIXED_DEFINE "${MM_HAVE_APFIXED_DEFINE}")

set(MM_DATA_TYPE          "float")
set(MM_ACC_DATA_TYPE      "float")
set(MM_DATA_TYPE_INCLUDE  "")
set(MM_HAVE_APFIXED_DEFINE "")

configure_file(
    include/Config.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/float_config/Config.h
    @ONLY
)

set(MM_DATA_TYPE          "${_SAVE_DATA_TYPE}")
set(MM_ACC_DATA_TYPE      "${_SAVE_ACC_DATA_TYPE}")
set(MM_DATA_TYPE_INCLUDE  "${_SAVE_DATA_TYPE_INCLUDE}")
set(MM_HAVE_APFIXED_DEFINE "${_SAVE_HAVE_APFIXED_DEFINE}")

add_library(matmul_kernel_float STATIC
    kernel/MatmulKernel.cpp
)
# float_config must come before include/ so its Config.h shadows the primary one.
target_include_directories(matmul_kernel_float
    PUBLIC
        ${CMAKE_CURRENT_BINARY_DIR}/float_config  # float Config.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include       # MatmulKernel.h
)
if(_MM_HAVE_HLS)
    # MatmulKernel.h needs ap_int.h / hls_burst_maxi.h for the 128-bit ports.
    target_include_directories(matmul_kernel_float SYSTEM PUBLIC ${HLS_INCLUDE_DIR})
endif()
target_compile_options(matmul_kernel_float PRIVATE -Wall -Wextra -O2 -Wno-unknown-pragmas)

# ---------------------------------------------------------------------------
# TestMatmulRef — C++ simulation test, no Vitis HLS synthesis required.
#
# Compiles and runs entirely with standard GCC.  The name "Ref" reflects that
# this is the Phase 1 reference; Phase 2 will rename/extend to TestMatmulSim
# once the HLS kernel function is wired in.
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
# TestMatmulBlas — validates the float kernel against cblas_sgemm.
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
    target_link_libraries(TestMatmulBlas PRIVATE matmul_kernel_float ${BLAS_LIBRARIES})
    target_include_directories(TestMatmulBlas PRIVATE ${CBLAS_INCLUDE_DIR})
    target_compile_options(TestMatmulBlas PRIVATE -Wall -Wextra -O2)
    add_test(NAME TestMatmulBlas COMMAND TestMatmulBlas)
else()
    message(STATUS "MatmulKernel: BLAS not found — TestMatmulBlas skipped")
endif()

message(STATUS "MatmulKernel: targets — matmul_kernel, matmul_kernel_float, TestMatmulRef")
message(STATUS "MatmulKernel: Data_t=${MM_DATA_TYPE}  "
               "AccData_t=${MM_ACC_DATA_TYPE}  "
               "TILE_N=${MM_TILE_N}  TILE_M=${MM_TILE_M}  "
               "TILE_K=${MM_TILE_K}  MAX_K=${MM_MAX_K}")

# ---------------------------------------------------------------------------
# HLS synthesis targets — one per platform JSON file in platforms/.
#
# Requires Vitis HLS (vitis-run for 2024.x+, vitis_hls for older versions).
# Each JSON must have a "part" field; "board" and "clock" are optional.
#
# Generated targets:
#   make synthesize_<name>   — C synthesis + ip_catalog export for that part
# ---------------------------------------------------------------------------
find_program(MM_VITIS_HLS NAMES vitis-run vitis_hls
    HINTS
        $ENV{XILINX_HLS}/bin
        $ENV{XILINX_VITIS}/bin
        /mnt/data/xilinx/2025.2/Vitis/bin
    DOC "Vitis HLS executable (vitis-run for 2024.x+, vitis_hls for older)")

if(MM_VITIS_HLS)
    # vitis-run (2024.x+) uses --tcl; the old vitis_hls used -f.
    if(MM_VITIS_HLS MATCHES "vitis-run")
        set(MM_HLS_TCL_FLAG "--tcl")
    else()
        set(MM_HLS_TCL_FLAG "-f")
    endif()
    message(STATUS "MatmulKernel: HLS synthesis enabled via ${MM_VITIS_HLS} ${MM_HLS_TCL_FLAG}")
else()
    message(STATUS "MatmulKernel: vitis_hls / vitis-run not found — synthesis targets disabled")
endif()

if(MM_VITIS_HLS)
    set(MM_KERNEL_SRC "${CMAKE_CURRENT_SOURCE_DIR}/kernel/MatmulKernel.cpp")

    set(_MM_PLATFORM_FILES "${AXI_PLATFORM_FILES}")
    foreach(_MM_PLATFORM_FILE IN LISTS _MM_PLATFORM_FILES)
        get_filename_component(_pname "${_MM_PLATFORM_FILE}" NAME_WE)
        file(READ "${_MM_PLATFORM_FILE}" _pjson)

        # Required: FPGA part string
        string(JSON _p_part ERROR_VARIABLE _e1 GET "${_pjson}" part)
        if(_e1)
            message(WARNING "MatmulKernel platform '${_pname}': missing 'part' — skipping.")
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

        # Per-platform matmul constants — drives this platform's Config.h
        # rather than the C-sim default platform's values, so each
        # synthesised IP uses bounds tuned for that platform.  Stash the
        # C-sim values so we can restore them after configure_file().
        set(_save_TILE_N "${MM_TILE_N}")
        set(_save_TILE_M "${MM_TILE_M}")
        set(_save_TILE_K "${MM_TILE_K}")
        set(_save_MAX_K  "${MM_MAX_K}")

        matmul_load_constants("${_MM_PLATFORM_FILE}" MM)

        # Generate a platform-specific Config.h using these constants.
        configure_file(
            "${CMAKE_CURRENT_SOURCE_DIR}/include/Config.h.in"
            "${_p_build_dir}/Config.h"
            @ONLY)

        set(MM_TILE_N "${_save_TILE_N}")
        set(MM_TILE_M "${_save_TILE_M}")
        set(MM_TILE_K "${_save_TILE_K}")
        set(MM_MAX_K  "${_save_MAX_K}")

        # set_part TCL command (with optional -board)
        if(_p_board)
            set(MM_BOARD_OPTION "set_part {${_p_part}} -board {${_p_board}}")
        else()
            set(MM_BOARD_OPTION "set_part {${_p_part}}")
        endif()

        # Variables consumed by Synthesis.tcl.in (@ONLY substitution)
        set(MM_HLS_PROJECT_NAME  "matmul_${_pname}")
        set(MM_SRC_SYNTHESIS     "${MM_KERNEL_SRC}")
        set(MM_SYNTHESIS_FLAGS
            "-O3 -I${CMAKE_CURRENT_SOURCE_DIR}/include -I${_p_build_dir}")
        set(MM_TARGET_CLOCK_TCL  "${_p_clock}")
        set(MM_IP_OUTPUT_DIR     "${_p_build_dir}/ip_catalog")

        configure_file(
            "${CMAKE_CURRENT_SOURCE_DIR}/scripts/Synthesis.tcl.in"
            "${_p_build_dir}/synthesize_${_pname}.tcl"
            @ONLY)

        # Clean stale HLS project + IP catalog so old Verilog from a prior
        # run isn't re-exported into the IP archive (which then breaks
        # downstream Vivado packaging when source files have been removed
        # or renamed since the last synthesis).
        add_custom_target(
            "synthesize_matmul_${_pname}"
            COMMAND ${CMAKE_COMMAND} -E rm -rf
                    "${_p_build_dir}/${MM_HLS_PROJECT_NAME}"
                    "${MM_IP_OUTPUT_DIR}"
            COMMAND ${MM_VITIS_HLS} ${MM_HLS_TCL_FLAG}
                    "${_p_build_dir}/synthesize_${_pname}.tcl"
            WORKING_DIRECTORY "${_p_build_dir}"
            COMMENT "HLS synthesis: MatmulKernel for '${_pname}' (${_p_part} @ ${_p_clock} MHz)"
            DEPENDS
                "${MM_KERNEL_SRC}"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/MatmulKernel.h"
                "${_p_build_dir}/Config.h")

        message(STATUS "MatmulKernel: synthesis target 'synthesize_${_pname}' "
                       "(${_p_part} @ ${_p_clock} MHz)")

        # -------------------------------------------------------------------
        # cosim_matmul_<name> — C synthesis + RTL co-simulation in ONE run.
        #
        # csynth_design and cosim_design are sequenced in a single TCL (see
        # Cosim.tcl.in), so one vitis-run invocation produces a component
        # whose run summary covers both — `vitis -a <component>` then shows
        # the cosim report (measured latency / II) alongside synthesis.
        #
        # Separate component (matmul_<name>_cosim) so it never clobbers the
        # synthesize target's matmul_<name> component / exported IP.  cosim
        # RTL-simulates the whole TestMatmulSim.cpp test bench, so this target
        # is slow (minutes) and is intentionally NOT a dependency of anything.
        # -------------------------------------------------------------------
        set(MM_COSIM_COMPONENT "matmul_${_pname}_cosim")
        set(MM_TB_SYNTHESIS    "${CMAKE_CURRENT_SOURCE_DIR}/test/TestMatmulSim.cpp")

        configure_file(
            "${CMAKE_CURRENT_SOURCE_DIR}/scripts/Cosim.tcl.in"
            "${_p_build_dir}/cosim_${_pname}.tcl"
            @ONLY)

        add_custom_target(
            "cosim_matmul_${_pname}"
            COMMAND ${CMAKE_COMMAND} -E rm -rf
                    "${_p_build_dir}/${MM_COSIM_COMPONENT}"
            COMMAND ${MM_VITIS_HLS} ${MM_HLS_TCL_FLAG}
                    "${_p_build_dir}/cosim_${_pname}.tcl"
            WORKING_DIRECTORY "${_p_build_dir}"
            COMMENT "HLS synth + cosim: MatmulKernel for '${_pname}' (${_p_part} @ ${_p_clock} MHz)"
            DEPENDS
                "${MM_KERNEL_SRC}"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/MatmulKernel.h"
                "${CMAKE_CURRENT_SOURCE_DIR}/test/TestMatmulSim.cpp"
                "${_p_build_dir}/Config.h")

        message(STATUS "MatmulKernel: cosim target 'cosim_matmul_${_pname}' "
                       "(${_p_part} @ ${_p_clock} MHz)")
    endforeach()
endif()
