cmake_minimum_required(VERSION 3.19)

# ---------------------------------------------------------------------------
# Reject in-source builds.  Configuring in the source tree scatters
# CMakeCache.txt / CMakeFiles / generated Config.h etc. across the repo, and a
# later `cmake <path>` silently re-targets that stale cache instead of the
# intended build directory.  Always configure into a separate build/ dir.
# ---------------------------------------------------------------------------
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
    message(FATAL_ERROR
        "In-source builds are not allowed. Run cmake from a separate build "
        "directory, e.g.:\n  mkdir build && cd build && cmake ..")
endif()

project(axi_kernels)

# ---------------------------------------------------------------------------
# All four kernels in SystemVerilog (kernels/vectorop_rtl, kernels/conv_rtl,
# kernels/matmul_rtl, kernels/pool_rtl); kernels/vectorop, kernels/conv,
# kernels/matmul and kernels/pool keep the retired Vitis HLS kernels' C++ as
# their reference models, testbench oracles and fixture generators.  Each
# subdirectory is a self-contained CMake project.
#
# Simulation test targets (no hardware needed):
#   make TestSimulation   — VectorOPKernel's C++ reference model
#   make TestConvRef      — ConvKernel's C++ reference model
#   make TestMatmulRef    — MatmulKernel   (TestMatmulBlas if BLAS available)
#   make TestPoolingSim   — PoolingKernel's C++ reference model
#   make TestMatmulRtl    — MatmulKernel in SystemVerilog (kernels/matmul_rtl,
#                           Verilator)
#   make TestVectorOpRtl  — VectorOPKernel in SystemVerilog (kernels/vectorop_rtl,
#                           Verilator)
#   make TestPoolRtl      — PoolingKernel in SystemVerilog (kernels/pool_rtl,
#                           Verilator)
#   make TestConvRtl      — ConvKernel in SystemVerilog (kernels/conv_rtl,
#                           Verilator)
#
# IP export targets (requires Vivado; every kernel's HLS synthesis was retired):
#   make synthesize_<platform>              — all four kernel IPs
#   make package_vectorop_rtl               — the VectorOPKernel IP
#   make package_conv_rtl                   — the ConvKernel IP
#   make package_matmul_rtl                 — the MatmulKernel IP
#   make package_pool_rtl                   — the PoolingKernel IP
#
# Platform definitions live in platforms/*.json (shared by all kernels).
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Global hardware parameters — shared by all kernels: the AXI_BUS_WIDTH cache
# var, the platform list and the default platform (AXI_PLATFORM_FILES, the
# AXI_PLATFORM cache var, AXI_DEFAULT_PLATFORM_JSON).  cmake/AxiPlatform.cmake,
# which every kernel's CMakeLists also includes so it configures on its own.
# ---------------------------------------------------------------------------
list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(AxiPlatform)

# Enable CTest at the top level so `ctest` from the build root discovers tests
# registered by every kernel subdirectory.
enable_testing()

add_subdirectory(kernels/vectorop)
add_subdirectory(kernels/conv)
add_subdirectory(kernels/matmul)
add_subdirectory(kernels/pool)
add_subdirectory(kernels/matmul_rtl)
add_subdirectory(kernels/vectorop_rtl)
add_subdirectory(kernels/pool_rtl)
add_subdirectory(kernels/conv_rtl)

# The hardware build's VectorOPKernel is the SystemVerilog one
# (kernels/vectorop_rtl; VLNV xilinx.com:hls:VectorOPKernel:1.0, the HLS
# kernel's ports, register map and m_axi bus parameters).  The Vitis HLS
# kernel's synthesis was retired (doc/plans/VECTOROP_RTL_PLAN.md phase 3);
# kernels/vectorop keeps its C++ as the oracle and fixture generator.  The
# phase-1 switch AXI_VECTOROP_IMPL is gone: a cache that still asks for hls
# is refused.
if(DEFINED CACHE{AXI_VECTOROP_IMPL})
    if(NOT AXI_VECTOROP_IMPL STREQUAL "rtl")
        message(FATAL_ERROR
            "AXI_VECTOROP_IMPL=${AXI_VECTOROP_IMPL}: the HLS VectorOPKernel's synthesis "
            "was retired (doc/plans/VECTOROP_RTL_PLAN.md); the hardware build always "
            "uses kernels/vectorop_rtl.  Re-run cmake with -UAXI_VECTOROP_IMPL.")
    endif()
    unset(AXI_VECTOROP_IMPL CACHE)
endif()

# The hardware build's MatmulKernel is the SystemVerilog one (kernels/matmul_rtl;
# VLNV xilinx.com:hls:MatmulKernel:1.0, the HLS kernel's register map, gmem2
# 128-bit).  The Vitis HLS kernel's synthesis was retired
# (doc/plans/MATMUL_RTL_PLAN.md phase 4); kernels/matmul keeps its C++ model as
# the oracle.  The platform JSON's kernels.matmul.impl tells the scheduler's
# cost model which kernel the bitstream carries, so it must say rtl here.
file(READ "${AXI_DEFAULT_PLATFORM_JSON}" _axi_pjson)
string(JSON _axi_mm_impl ERROR_VARIABLE _axi_mm_err GET "${_axi_pjson}" kernels matmul impl)
if(_axi_mm_err OR NOT _axi_mm_impl STREQUAL "rtl")
    message(FATAL_ERROR "axi_kernels: ${AXI_DEFAULT_PLATFORM_JSON}: kernels.matmul.impl must be "
                        "\"rtl\" (the hardware build packages the SystemVerilog MatmulKernel; "
                        "the HLS kernel's synthesis was retired), got '${_axi_mm_impl}'")
endif()

# The hardware build's PoolingKernel is the SystemVerilog one (kernels/pool_rtl;
# VLNV xilinx.com:hls:PoolingKernel:1.0, the HLS kernel's ports, register map
# and m_axi bus parameters).  The Vitis HLS kernel's synthesis was retired
# (doc/plans/POOL_RTL_PLAN.md phase 3); kernels/pool keeps its C++ as the
# oracle and fixture generator.  The phase-1 switch AXI_POOL_IMPL is gone: a
# cache that still asks for hls is refused.
if(DEFINED CACHE{AXI_POOL_IMPL})
    if(NOT AXI_POOL_IMPL STREQUAL "rtl")
        message(FATAL_ERROR
            "AXI_POOL_IMPL=${AXI_POOL_IMPL}: the HLS PoolingKernel's synthesis "
            "was retired (doc/plans/POOL_RTL_PLAN.md); the hardware build always "
            "uses kernels/pool_rtl.  Re-run cmake with -UAXI_POOL_IMPL.")
    endif()
    unset(AXI_POOL_IMPL CACHE)
endif()

# The hardware build's ConvKernel is the SystemVerilog one (kernels/conv_rtl;
# VLNV xilinx.com:hls:ConvKernel:1.0, the HLS kernel's ports, register map and
# m_axi bus parameters).  The Vitis HLS kernel's synthesis was retired
# (doc/plans/CONV_RTL_PLAN.md phase 3); kernels/conv keeps its C++ as the
# oracle and fixture generator.  No Vitis HLS kernel is left in the hardware
# build.  The phase-1 switch AXI_CONV_IMPL is gone: a cache that still asks
# for hls is refused.  The platform JSON's kernels.conv.impl tells the
# scheduler's cost model which kernel the bitstream carries, so it must say
# rtl here.
if(DEFINED CACHE{AXI_CONV_IMPL})
    if(NOT AXI_CONV_IMPL STREQUAL "rtl")
        message(FATAL_ERROR
            "AXI_CONV_IMPL=${AXI_CONV_IMPL}: the HLS ConvKernel's synthesis "
            "was retired (doc/plans/CONV_RTL_PLAN.md); the hardware build always "
            "uses kernels/conv_rtl.  Re-run cmake with -UAXI_CONV_IMPL.")
    endif()
    unset(AXI_CONV_IMPL CACHE)
endif()
string(JSON _axi_cv_impl ERROR_VARIABLE _axi_cv_err GET "${_axi_pjson}" kernels conv impl)
if(_axi_cv_err OR NOT _axi_cv_impl STREQUAL "rtl")
    message(FATAL_ERROR "axi_kernels: ${AXI_DEFAULT_PLATFORM_JSON}: kernels.conv.impl must be "
                        "\"rtl\" (the hardware build packages the SystemVerilog ConvKernel; "
                        "the HLS kernel's synthesis was retired), got '${_axi_cv_impl}'")
endif()

# ---------------------------------------------------------------------------
# run_tests — build every C-simulation test executable, then run ctest.
# `make test` (CTest's built-in target) only runs the tests; it does not
# rebuild their executables.  This aggregate target does both.
# ---------------------------------------------------------------------------
set(_AXI_TEST_TARGETS
    TestSimulation
    TestConvRef
    TestMatmulRef
    TestPoolingSim
)
if(TARGET TestMatmulBlas)
    list(APPEND _AXI_TEST_TARGETS TestMatmulBlas)
endif()
if(TARGET TestMatmulRtl)
    list(APPEND _AXI_TEST_TARGETS TestMatmulRtl)
endif()
if(TARGET TestVectorOpRtl)
    list(APPEND _AXI_TEST_TARGETS TestVectorOpRtl)
endif()
if(TARGET TestPoolRtl)
    list(APPEND _AXI_TEST_TARGETS TestPoolRtl)
endif()
if(TARGET TestConvRtl)
    list(APPEND _AXI_TEST_TARGETS TestConvRtl)
endif()
if(TARGET conv_rtl_dsp_check)
    list(APPEND _AXI_TEST_TARGETS conv_rtl_dsp_check)
endif()

add_custom_target(run_tests
    COMMAND "${CMAKE_CTEST_COMMAND}" --output-on-failure
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
    DEPENDS ${_AXI_TEST_TARGETS}
    COMMENT "Running all C-simulation tests"
    USES_TERMINAL
)
message(STATUS "axi_kernels: target 'run_tests' (${_AXI_TEST_TARGETS})")

# ---------------------------------------------------------------------------
# dtc — device tree compiler (required for dtbo targets)
# ---------------------------------------------------------------------------
find_program(DTC_EXECUTABLE dtc)
if(DTC_EXECUTABLE)
    message(STATUS "axi_kernels: dtc found at ${DTC_EXECUTABLE}")
else()
    message(STATUS "axi_kernels: dtc not found; dtbo targets will fail at build time")
    message(STATUS "             Install: sudo apt install device-tree-compiler")
endif()

# ---------------------------------------------------------------------------
# Convenience aggregate synthesis targets — one per platform file.
# Device tree overlay targets — one per .dts file under dts/<platform>/.
# ---------------------------------------------------------------------------
foreach(_PF IN LISTS AXI_PLATFORM_FILES)
    get_filename_component(_pname "${_PF}" NAME_WE)

    add_custom_target("synthesize_${_pname}"
        DEPENDS
            package_vectorop_rtl
            package_conv_rtl
            package_matmul_rtl
            package_pool_rtl
        COMMENT "IP export: all four kernels' IPs for platform '${_pname}' (SystemVerilog, Vivado)"
    )
    message(STATUS "axi_kernels: aggregate target 'synthesize_${_pname}'")

    file(GLOB _DTS_FILES "${CMAKE_CURRENT_SOURCE_DIR}/dts/${_pname}/*.dts")
    foreach(_DTS IN LISTS _DTS_FILES)
        get_filename_component(_stem "${_DTS}" NAME_WE)
        set(_DTBO_DIR "${CMAKE_CURRENT_BINARY_DIR}/dts/${_pname}")
        set(_DTBO     "${_DTBO_DIR}/design_${_stem}.dtbo")

        if(DTC_EXECUTABLE)
            add_custom_command(
                OUTPUT  "${_DTBO}"
                COMMAND "${CMAKE_COMMAND}" -E make_directory "${_DTBO_DIR}"
                COMMAND "${DTC_EXECUTABLE}" -@ -O dtb -o "${_DTBO}" "${_DTS}"
                DEPENDS "${_DTS}"
                WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/dts/${_pname}"
                COMMENT "DTC: ${_stem}.dts → design_${_stem}.dtbo"
                VERBATIM
            )
        else()
            add_custom_command(
                OUTPUT  "${_DTBO}"
                COMMAND "${CMAKE_COMMAND}" -E echo
                        "ERROR: dtc not found -- install device-tree-compiler"
                COMMAND "${CMAKE_COMMAND}" -E false
                DEPENDS "${_DTS}"
                VERBATIM
            )
        endif()

        add_custom_target("dtbo_${_pname}_${_stem}"
            DEPENDS "${_DTBO}"
            COMMENT "Device tree overlay: ${_stem} for platform '${_pname}'"
        )
        message(STATUS "axi_kernels: target 'dtbo_${_pname}_${_stem}'")
    endforeach()
endforeach()

# ---------------------------------------------------------------------------
# Hardware build target — Vivado synthesis + implementation + bitstream.
# Requires the cormorant_hw_128 submodule and Vivado 2025.2.  Every kernel IP
# has fixed 128-bit data ports (the block design's width); AXI_BUS_WIDTH no
# longer affects them.
#
# Usage:
#   make synthesize_kv260      # build all four kernel IPs first
#   make build_hw_kv260        # then run Vivado on the block design
#
# Or in one shot (synthesize_kv260 is listed as a dependency):
#   make build_hw_kv260
# ---------------------------------------------------------------------------
set(_HW_REPO "${CMAKE_SOURCE_DIR}/hw/cormorant_hw_128")
if(EXISTS "${_HW_REPO}/build.sh")
    # The IP repository Vivado scans: ip_repo_kv260/ links one IP per kernel,
    # the RTL IPs packaged in <build>/rtl_ip/ (outside <build>/kernels, so a
    # directory holding an old HLS export of the same VLNV is never scanned).
    set(_HW_IP_REPO "${CMAKE_BINARY_DIR}/ip_repo_kv260")
    add_custom_target(ip_repo_kv260
        COMMAND "${CMAKE_COMMAND}" -E rm -rf "${_HW_IP_REPO}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${_HW_IP_REPO}"
        COMMAND "${CMAKE_COMMAND}" -E create_symlink
                "${CMAKE_BINARY_DIR}/rtl_ip/VectorOPKernel_ip" "${_HW_IP_REPO}/vectorop"
        COMMAND "${CMAKE_COMMAND}" -E create_symlink
                "${CMAKE_BINARY_DIR}/rtl_ip/ConvKernel_ip" "${_HW_IP_REPO}/conv"
        COMMAND "${CMAKE_COMMAND}" -E create_symlink
                "${CMAKE_BINARY_DIR}/rtl_ip/PoolingKernel_ip" "${_HW_IP_REPO}/pool"
        COMMAND "${CMAKE_COMMAND}" -E create_symlink
                "${CMAKE_BINARY_DIR}/rtl_ip/MatmulKernel_ip" "${_HW_IP_REPO}/matmul"
        DEPENDS package_matmul_rtl package_vectorop_rtl package_pool_rtl package_conv_rtl
        COMMENT "IP repository for the KV260 build: the four RTL kernels"
        VERBATIM)

    add_custom_target(build_hw_kv260
        COMMAND bash "${_HW_REPO}/build.sh" all
                -ip-repo "${_HW_IP_REPO}"
        WORKING_DIRECTORY "${_HW_REPO}"
        DEPENDS synthesize_kv260 ip_repo_kv260
        COMMENT "Vivado: synthesis + implementation + bitstream (KV260 128-bit)"
    )
    message(STATUS "axi_kernels: target 'build_hw_kv260' (depends on synthesize_kv260)")

    add_custom_target(sim_hw_kv260
        COMMAND bash "${_HW_REPO}/sim.sh"
                -ip-repo "${_HW_IP_REPO}"
        WORKING_DIRECTORY "${_HW_REPO}"
        DEPENDS ip_repo_kv260
        COMMENT "Vivado: behavioral simulation (KV260)"
    )
    message(STATUS "axi_kernels: target 'sim_hw_kv260'")
else()
    message(STATUS "axi_kernels: 'build_hw_kv260' and 'sim_hw_kv260' not available — submodule not initialised")
    message(STATUS "             Run: git submodule update --init hw/cormorant_hw_128")
endif()

# ---------------------------------------------------------------------------
# Per-kernel behavioural-simulation targets — drive cormorant_test_stand's
# Vivado projects (one per kernel) through their xsim flow with the freshly-
# built IP catalogue and the **checked-in** golden fixtures under hw/test_data/.
#
# Requires the cormorant_test_stand submodule and Vivado 2025.2.
#
# Generated targets:
#   make behavior_test_conv         — ConvKernel testbench
#   make behavior_test_pool         — PoolingKernel testbench
#   make behavior_test_matmul       — MatmulKernel testbench
#   make behavior_test_vectorop     — VectorOPKernel testbench
#   make behavior_test              — all four kernels in sequence
#
# Each per-kernel target:
#   - depends on the kernel's `package_<k>_rtl` (the IP in
#     <build>/rtl_ip/<Kernel>_ip; the test stand upgrades its block design to
#     it and puts the instance widths back to the IP's defaults: matmul gmem2
#     32 -> 128)
#   - reads its fixtures from the source tree at hw/test_data/<dir>/ — the
#     same goldens are used on every machine, so any divergence is a kernel
#     change, not a fixture-regeneration drift.  `gen_<k>_test_data` is left
#     in place for ad-hoc fixture regeneration but is NOT a dependency here.
#   - invokes the test stand's `make tb-<name>` with DATA_DIR_<name>,
#     IP_REPO_<name>, and REPORT_<name> all pinned by this CMake target.
#   - finally runs cmake/check_test_report.py to turn summary.all_passed
#     into the target's exit code.
#
# Kernel-name asymmetry:
#   axi_kernel  | test-stand | IP (<build>/rtl_ip/) | fixture dir under hw/test_data/
#   conv        | conv       | ConvKernel_ip        | conv_test_data
#   pool        | pooling    | PoolingKernel_ip     | pool_test_data
#   matmul      | matmul_op  | MatmulKernel_ip      | matmul_test_data
#   vectorop    | vector_op  | VectorOPKernel_ip    | vecop_test_data
#
# Vivado's set_property ip_repo_paths needs the directory that holds the IP's
# component.xml itself; passing a parent does NOT work.
# ---------------------------------------------------------------------------
set(_TS_REPO "${CMAKE_SOURCE_DIR}/hw/cormorant_test_stand")
if(EXISTS "${_TS_REPO}/Makefile")

    # Columns: axi_kernels name | test-stand registry | IP name | fixture-dir name
    set(_BEHAVIOR_TESTS
        "conv|conv|ConvKernel|conv_test_data"
        "pool|pooling|PoolingKernel|pool_test_data"
        "matmul|matmul_op|MatmulKernel|matmul_test_data"
        "vectorop|vector_op|VectorOPKernel|vecop_test_data"
    )

    set(_BEHAVIOR_TEST_AGG_DEPS "")

    foreach(_BT IN LISTS _BEHAVIOR_TESTS)
        string(REPLACE "|" ";" _BT_LIST "${_BT}")
        list(GET _BT_LIST 0 _bt_kernel)    # axi_kernels-side name (package_<k>_rtl)
        list(GET _BT_LIST 1 _bt_tsname)    # test-stand registry name (for tb-<name>)
        list(GET _BT_LIST 2 _bt_ipname)    # the IP: <build>/rtl_ip/<name>_ip
        list(GET _BT_LIST 3 _bt_dataname)  # subdirectory under hw/test_data/

        set(_bt_datadir "${CMAKE_SOURCE_DIR}/hw/test_data/${_bt_dataname}")
        set(_bt_iprepo  "${CMAKE_BINARY_DIR}/rtl_ip/${_bt_ipname}_ip")
        set(_bt_ipdep   "package_${_bt_kernel}_rtl")

        # Pin the scoreboard JSON report to a known path in the build tree.
        # check_test_report.py turns summary.all_passed into the target's
        # exit code, so behavior_test_<k> fails whenever any mismatch is
        # recorded — independent of run_tb.sh's own exit code (which today
        # masks the Python parse-block failure with `|| true`).
        set(_bt_reportdir "${CMAKE_BINARY_DIR}/kernels/${_bt_kernel}/kv260")
        set(_bt_report "${_bt_reportdir}/${_bt_tsname}_test_report.json")

        add_custom_target("behavior_test_${_bt_kernel}"
            COMMAND ${CMAKE_COMMAND} -E echo
                    "behavior_test_${_bt_kernel}: tb-${_bt_tsname}  data=${_bt_datadir}  ip=${_bt_iprepo}  report=${_bt_report}"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${_bt_reportdir}"
            # Wipe any stale report from a prior run BEFORE the test runs —
            # otherwise a Vivado crash that never overwrites the JSON would
            # let check_test_report.py succeed against a leftover PASS file.
            COMMAND ${CMAKE_COMMAND} -E rm -f "${_bt_report}"
            COMMAND ${CMAKE_MAKE_PROGRAM} -C "${_TS_REPO}"
                    "tb-${_bt_tsname}"
                    "DATA_DIR_${_bt_tsname}=${_bt_datadir}"
                    "IP_REPO_${_bt_tsname}=${_bt_iprepo}"
                    "REPORT_${_bt_tsname}=${_bt_report}"
            COMMAND python3 "${CMAKE_SOURCE_DIR}/cmake/check_test_report.py"
                    "${_bt_report}"
            WORKING_DIRECTORY "${_TS_REPO}"
            DEPENDS
                "${_bt_ipdep}"
            COMMENT "Vivado xsim: behavioural test of ${_bt_kernel} via cormorant_test_stand"
            USES_TERMINAL
            VERBATIM
        )
        message(STATUS "axi_kernels: target 'behavior_test_${_bt_kernel}' "
                       "(data=hw/test_data/${_bt_dataname}, depends on ${_bt_ipdep})")
        list(APPEND _BEHAVIOR_TEST_AGG_DEPS "behavior_test_${_bt_kernel}")
    endforeach()

    # Aggregate — runs all four behavioural tests sequentially.  CMake doesn't
    # parallelise add_custom_target dependencies that share a working dir, so
    # the kernels are exercised one after another (which is what we want for
    # Vivado batch invocations anyway).
    add_custom_target(behavior_test
        DEPENDS ${_BEHAVIOR_TEST_AGG_DEPS}
        COMMENT "Vivado xsim: behavioural test of all four kernels"
    )
    message(STATUS "axi_kernels: aggregate target 'behavior_test' (${_BEHAVIOR_TEST_AGG_DEPS})")

else()
    message(STATUS "axi_kernels: 'behavior_test_<k>' not available — submodule not initialised")
    message(STATUS "             Run: git submodule update --init hw/cormorant_test_stand")
endif()
