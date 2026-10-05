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
# All four kernels — Conv and Pool in Vitis HLS, MatmulKernel and
# VectorOPKernel in SystemVerilog (kernels/matmul_rtl, kernels/vectorop_rtl;
# kernels/matmul and kernels/vectorop keep the retired HLS kernels' C++ as
# their reference models).  Each subdirectory is a self-contained CMake
# project.
#
# Simulation test targets (no hardware needed):
#   make TestSimulation   — VectorOPKernel's C++ reference model
#   make TestConvRef      — ConvKernel
#   make TestMatmulRef    — MatmulKernel   (TestMatmulBlas if BLAS available)
#   make TestPoolingSim   — PoolingKernel
#   make TestMatmulRtl    — MatmulKernel in SystemVerilog (kernels/matmul_rtl,
#                           Verilator)
#   make TestVectorOpRtl  — VectorOPKernel in SystemVerilog (kernels/vectorop_rtl,
#                           Verilator)
#
# IP export targets (requires Vitis / Vivado):
#   make synthesize_<platform>              — all four kernel IPs
#   make synthesize_<kernel>_<platform>     — one HLS kernel (conv, pool)
#   make package_matmul_rtl                 — the MatmulKernel IP (SystemVerilog,
#                                             Vivado; its HLS synthesis was retired)
#   make package_vectorop_rtl               — the VectorOPKernel IP (likewise)
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
            "synthesize_conv_${_pname}"
            package_matmul_rtl
            "synthesize_pool_${_pname}"
        COMMENT "IP export: all kernels for platform '${_pname}' (the RTL VectorOP and Matmul kernels, Conv and Pool HLS)"
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
# Requires the cormorant_hw_128 submodule and Vivado 2025.2.
# Must configure with -DAXI_BUS_WIDTH=128 to match the 128-bit block design.
#
# Usage:
#   cmake .. -DAXI_BUS_WIDTH=128
#   make synthesize_kv260      # build all four kernel IPs first
#   make build_hw_kv260        # then run Vivado on the block design
#
# Or in one shot (synthesize_kv260 is listed as a dependency):
#   make build_hw_kv260
# ---------------------------------------------------------------------------
set(_HW_REPO "${CMAKE_SOURCE_DIR}/hw/cormorant_hw_128")
if(EXISTS "${_HW_REPO}/build.sh")
    # The IP repository Vivado scans: ip_repo_kv260/ links one IP per kernel —
    # the HLS exports and the RTL IPs (packaged outside <build>/kernels, in
    # rtl_ip/, so no directory Vivado scans holds two IPs with one VLNV).
    set(_HW_IP_REPO "${CMAKE_BINARY_DIR}/ip_repo_kv260")
    set(_HW_K "${CMAKE_BINARY_DIR}/kernels")
    add_custom_target(ip_repo_kv260
        COMMAND "${CMAKE_COMMAND}" -E rm -rf "${_HW_IP_REPO}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${_HW_IP_REPO}"
        COMMAND "${CMAKE_COMMAND}" -E create_symlink
                "${CMAKE_BINARY_DIR}/rtl_ip/VectorOPKernel_ip" "${_HW_IP_REPO}/vectorop"
        COMMAND "${CMAKE_COMMAND}" -E create_symlink
                "${_HW_K}/conv/kv260/conv_kv260/hls/impl/ip" "${_HW_IP_REPO}/conv"
        COMMAND "${CMAKE_COMMAND}" -E create_symlink
                "${_HW_K}/pool/kv260/pool_kv260/hls/impl/ip" "${_HW_IP_REPO}/pool"
        COMMAND "${CMAKE_COMMAND}" -E create_symlink
                "${CMAKE_BINARY_DIR}/rtl_ip/MatmulKernel_ip" "${_HW_IP_REPO}/matmul"
        DEPENDS package_matmul_rtl package_vectorop_rtl
        COMMENT "IP repository for the KV260 build: the RTL VectorOP and Matmul kernels, Conv and Pool HLS"
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
#   - depends on `synthesize_<k>_kv260` (IP catalogue must exist at the
#     revision the test stand's .xpr's .xci references); the RTL kernels on
#     `package_matmul_rtl` / `package_vectorop_rtl` (the test stand puts the
#     instance widths back to the IP's defaults after the upgrade: matmul
#     gmem2 32 -> 128)
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
#   axi_kernel  | test-stand | hls_project    | fixture dir under hw/test_data/
#   conv        | conv       | conv_kv260     | conv_test_data
#   pool        | pooling    | pool_kv260     | pool_test_data
#   matmul      | matmul_op  | (RTL IP)       | matmul_test_data
#   vectorop    | vector_op  | (RTL IP)       | vecop_test_data
#
# IP repo path resolution: Vitis HLS exports the IP catalogue into a
# directory containing component.xml.  Vivado's set_property ip_repo_paths
# needs that exact directory; passing a parent does NOT work.  The path
# depends on the HLS flow: the Vitis unified component flow (conv and pool)
# puts it under <hls_project>/hls/impl/ip/, the legacy
# open_project/open_solution flow under <hls_project>/solution1/impl/ip/.  The
# per-kernel IP subpath is therefore the 5th column of _BEHAVIOR_TESTS below;
# the RTL IPs are packaged in <build>/rtl_ip/<Kernel>_ip.
# ---------------------------------------------------------------------------
set(_TS_REPO "${CMAKE_SOURCE_DIR}/hw/cormorant_test_stand")
if(EXISTS "${_TS_REPO}/Makefile")

    # Columns: axi_kernels name | test-stand registry | hls_project |
    #          fixture-dir name | IP-repo subpath (flow-dependent, see above)
    set(_BEHAVIOR_TESTS
        "conv|conv|conv_kv260|conv_test_data|hls/impl/ip"
        "pool|pooling|pool_kv260|pool_test_data|hls/impl/ip"
        "matmul|matmul_op|-|matmul_test_data|-"
        "vectorop|vector_op|-|vecop_test_data|-"
    )

    set(_BEHAVIOR_TEST_AGG_DEPS "")

    foreach(_BT IN LISTS _BEHAVIOR_TESTS)
        string(REPLACE "|" ";" _BT_LIST "${_BT}")
        list(GET _BT_LIST 0 _bt_kernel)    # axi_kernels-side name (matches synthesize_<k>_kv260)
        list(GET _BT_LIST 1 _bt_tsname)    # test-stand registry name (for tb-<name>)
        list(GET _BT_LIST 2 _bt_hlsproj)   # HLS project dir name produced by Vitis
        list(GET _BT_LIST 3 _bt_dataname)  # subdirectory under hw/test_data/
        list(GET _BT_LIST 4 _bt_ipsub)     # IP-repo subpath under <hls_project> (flow-dependent)

        set(_bt_datadir "${CMAKE_SOURCE_DIR}/hw/test_data/${_bt_dataname}")
        set(_bt_iprepo  "${CMAKE_BINARY_DIR}/kernels/${_bt_kernel}/kv260/${_bt_hlsproj}/${_bt_ipsub}")
        set(_bt_ipdep   "synthesize_${_bt_kernel}_kv260")
        if(_bt_kernel STREQUAL "matmul")
            set(_bt_iprepo "${CMAKE_BINARY_DIR}/rtl_ip/MatmulKernel_ip")
            set(_bt_ipdep  package_matmul_rtl)
        elseif(_bt_kernel STREQUAL "vectorop")
            set(_bt_iprepo "${CMAKE_BINARY_DIR}/rtl_ip/VectorOPKernel_ip")
            set(_bt_ipdep  package_vectorop_rtl)
        endif()

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
