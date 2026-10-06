cmake_minimum_required(VERSION 3.19)
project(PoolingKernelRtl VERSION 1.0 LANGUAGES C CXX)

# The platform JSONs (AXI_PLATFORM_FILES, AXI_DEFAULT_PLATFORM_JSON): set by the
# top-level project, or here when this kernel is configured on its own.
list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../cmake")
include(AxiPlatform)
get_filename_component(PL_RTL_REPO "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
enable_testing()

# ---------------------------------------------------------------------------
# PoolingKernel in SystemVerilog — a drop-in for the Vitis HLS PoolingKernel
# (kernels/pool, whose C++ is the reference model): same VLNV, register map,
# m_axi bus parameters, DDR access pattern and arithmetic.  See
# doc/plans/POOL_RTL_PLAN.md.
#
# Targets (each one only when its tool is found):
#   TestPoolRtl            Verilator testbench (vl/Vtb); ctest runs the
#                          fixtures + PL_RTL_RANDOM_CASES random jobs checked
#                          against the HLS kernel's C++ (with the Vitis HLS headers)
#   lint_pool_rtl          verilator --lint-only -Wall
#   perf_pool_rtl          cycle counts of typical jobs (ideal memory)
#   pool_rtl_tb_fst        testbench with FST tracing (vlt/Vtb --trace F.fst)
#   driver_pool_rtl        the C driver (scripts/gen_driver.py) -> driver/
#   package_pool_rtl       Vivado IP catalog -> <build>/rtl_ip/PoolingKernel_ip (+ .zip)
#   synth_pool_rtl         Vivado out-of-context synthesis + P&R -> synth/
#   xsim_pool_rtl          xvlog / xelab parse and elaboration
#   sysim_pool_rtl         the test stand's PoolingKernel block design with
#                          this IP in xsim, on a copy
# ---------------------------------------------------------------------------

set(PL_RTL_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(PL_RTL_SOURCES
    rtl/pl_pkg.sv rtl/pl_fifo.sv rtl/pl_rs.sv rtl/pl_lutram.sv rtl/pl_ctrl_s_axi.sv rtl/pl_loader.sv
    rtl/pl_emit.sv rtl/pl_reduce.sv rtl/pl_writer.sv rtl/pl_core.sv rtl/PoolingKernel.v)
list(TRANSFORM PL_RTL_SOURCES PREPEND "${PL_RTL_DIR}/")
set(PL_RTL_TB_SOURCES
    "${PL_RTL_DIR}/tb/verilator/tb_main.cpp"
    "${PL_RTL_DIR}/tb/verilator/axi_models.h")
set(PL_RTL_WAIVERS "${PL_RTL_DIR}/scripts/lint_waivers.vlt")
set(PL_RTL_FIXTURES "${PL_RTL_REPO}/hw/test_data/pool_test_data"
    CACHE PATH "PoolingKernel fixtures (manifest.txt + test_NN_{x,y}.hex) for TestPoolRtl")
set(PL_RTL_RANDOM_CASES 200 CACHE STRING "TestPoolRtl: random jobs after the fixtures")
set(PL_RTL_PERIOD 3.333 CACHE STRING "synth_pool_rtl: clock period in ns (OOC timing check)")

find_package(Python3 COMPONENTS Interpreter REQUIRED)

file(READ "${AXI_DEFAULT_PLATFORM_JSON}" _pl_rtl_pjson)
string(JSON PL_RTL_PART GET "${_pl_rtl_pjson}" part)

# pl_pkg.sv holds the platform's kernels.pool bounds as constants (the HLS
# kernel's Config.h from the same file): a platform with other bounds needs
# them changed there.
file(READ "${PL_RTL_DIR}/rtl/pl_pkg.sv" _pl_rtl_pkg)
foreach(_pair "tile_c;TC" "max_kh;MAXK" "max_kw;MAXK" "max_line_buf_rows;LBR"
              "max_line_buf_cols;LBC" "ow_parallel;OWP")
    list(GET _pair 0 _key)
    list(GET _pair 1 _const)
    string(JSON _json_v GET "${_pl_rtl_pjson}" kernels pool ${_key})
    string(REGEX MATCH "localparam int ${_const} +=  *([0-9]+)" _m "${_pl_rtl_pkg}")
    if(NOT CMAKE_MATCH_1 STREQUAL _json_v)
        message(FATAL_ERROR "PoolingKernel RTL: kernels.pool.${_key} = ${_json_v} in "
                            "${AXI_DEFAULT_PLATFORM_JSON}, but rtl/pl_pkg.sv has ${_const} = ${CMAKE_MATCH_1}")
    endif()
endforeach()

# ---------------------------------------------------------------------------
# The C driver: the API and register map of the HLS-generated xpoolingkernel,
# written by scripts/gen_driver.py (the RTL MatmulKernel's generator); the
# register table is checked against rtl/pl_ctrl_s_axi.sv by ctest
# (PoolRtlDriver) and compiled here.
# ---------------------------------------------------------------------------
set(PL_RTL_DRIVER_ROOT "${CMAKE_CURRENT_BINARY_DIR}/driver")
set(PL_RTL_DRIVER_SRC "${PL_RTL_DRIVER_ROOT}/PoolingKernel_v1_0/src")
set(_pl_rtl_gen "${PL_RTL_DIR}/scripts/gen_driver.py" "${PL_RTL_REPO}/kernels/matmul_rtl/scripts/gen_driver.py")
set(_pl_rtl_driver_files
    "${PL_RTL_DRIVER_SRC}/xpoolingkernel.c" "${PL_RTL_DRIVER_SRC}/xpoolingkernel_linux.c"
    "${PL_RTL_DRIVER_SRC}/xpoolingkernel_sinit.c" "${PL_RTL_DRIVER_SRC}/xpoolingkernel.h"
    "${PL_RTL_DRIVER_SRC}/xpoolingkernel_hw.h")
add_custom_command(
    OUTPUT ${_pl_rtl_driver_files}
    COMMAND "${Python3_EXECUTABLE}" "${PL_RTL_DIR}/scripts/gen_driver.py" --out "${PL_RTL_DRIVER_ROOT}"
    DEPENDS ${_pl_rtl_gen}
    COMMENT "PoolingKernel RTL: C driver -> ${PL_RTL_DRIVER_ROOT}"
    VERBATIM)
add_library(pool_rtl_driver OBJECT
    "${PL_RTL_DRIVER_SRC}/xpoolingkernel.c" "${PL_RTL_DRIVER_SRC}/xpoolingkernel_linux.c")
target_compile_options(pool_rtl_driver PRIVATE -Wall -Wextra -Werror)
add_custom_target(driver_pool_rtl DEPENDS pool_rtl_driver)
add_test(NAME PoolRtlDriver
         COMMAND "${Python3_EXECUTABLE}" "${PL_RTL_DIR}/scripts/gen_driver.py" --check)

# ---------------------------------------------------------------------------
# The HLS oracle: kernels/pool/kernel/PoolingKernel.cpp built with the Vitis
# HLS headers into the testbench (ap_fixed<16,8> / <32,16> and the platform's
# bounds, as the bitstream).
# ---------------------------------------------------------------------------
find_path(PL_RTL_HLS_INCLUDE NAMES ap_fixed.h hls_burst_maxi.h
    HINTS $ENV{XILINX_HLS}/include $ENV{XILINX_VITIS}/include /mnt/data/xilinx/2025.2/Vitis/include
    NO_DEFAULT_PATH
    DOC "Vitis HLS include directory (ap_fixed.h): the testbench's HLS oracle")
set(_pl_rtl_oracle_flags)
set(_pl_rtl_oracle_src)
if(PL_RTL_HLS_INCLUDE)
    set(_pl_rtl_cfg "${CMAKE_CURRENT_BINARY_DIR}/oracle")
    set(POOL_DATA_TYPE_INCLUDE   "#include \"ap_fixed.h\"")
    set(POOL_HAVE_APFIXED_DEFINE "#define POOL_HAVE_APFIXED")
    set(POOL_DATA_TYPE     "ap_fixed<16,8>")
    set(POOL_ACC_DATA_TYPE "ap_fixed<32,16>")
    set(POOL_DATA_MIN      "-128.0f")
    set(POOL_ACC_MIN       "-32768.0f")
    string(JSON POOL_TILE_C            GET "${_pl_rtl_pjson}" kernels pool tile_c)
    string(JSON POOL_MAX_KH            GET "${_pl_rtl_pjson}" kernels pool max_kh)
    string(JSON POOL_MAX_KW            GET "${_pl_rtl_pjson}" kernels pool max_kw)
    string(JSON POOL_MAX_LINE_BUF_ROWS GET "${_pl_rtl_pjson}" kernels pool max_line_buf_rows)
    string(JSON POOL_MAX_LINE_BUF_COLS GET "${_pl_rtl_pjson}" kernels pool max_line_buf_cols)
    string(JSON POOL_OW_PARALLEL       GET "${_pl_rtl_pjson}" kernels pool ow_parallel)
    configure_file("${PL_RTL_REPO}/kernels/pool/include/Config.h.in" "${_pl_rtl_cfg}/Config.h")
    set(_pl_rtl_oracle_src "${PL_RTL_REPO}/kernels/pool/kernel/PoolingKernel.cpp")
    set(_pl_rtl_oracle_flags -CFLAGS -DPL_HLS_ORACLE -CFLAGS -I${_pl_rtl_cfg}
        -CFLAGS -I${PL_RTL_REPO}/kernels/pool/include -CFLAGS -I${PL_RTL_HLS_INCLUDE}
        -CFLAGS -Wno-unused -CFLAGS -Wno-unknown-pragmas)
    message(STATUS "PoolingKernel RTL: testbench oracle = the HLS kernel's C++ (${PL_RTL_HLS_INCLUDE})")
    set(_pl_rtl_ctest_random --random ${PL_RTL_RANDOM_CASES})
else()
    message(STATUS "PoolingKernel RTL: Vitis HLS headers not found — the testbench checks the fixtures only")
    set(_pl_rtl_ctest_random)
endif()

# ---------------------------------------------------------------------------
# Verilator: lint and the C++ testbench
# ---------------------------------------------------------------------------
find_program(PL_RTL_VERILATOR verilator DOC "Verilator (5.x) for the RTL PoolingKernel testbench")
if(PL_RTL_VERILATOR)
    execute_process(COMMAND "${PL_RTL_VERILATOR}" --version
                    OUTPUT_VARIABLE _pl_rtl_vl_version OUTPUT_STRIP_TRAILING_WHITESPACE)
    string(REGEX MATCH "[0-9]+\\.[0-9]+" _pl_rtl_vl_version "${_pl_rtl_vl_version}")
    if(_pl_rtl_vl_version VERSION_LESS 5.0)
        message(WARNING "PoolingKernel RTL: Verilator ${_pl_rtl_vl_version} found, the testbench needs 5.x "
                        "— TestPoolRtl disabled")
        unset(PL_RTL_VERILATOR)
    endif()
endif()

if(PL_RTL_VERILATOR)
    message(STATUS "PoolingKernel RTL: Verilator ${_pl_rtl_vl_version} at ${PL_RTL_VERILATOR}")
    cmake_host_system_information(RESULT _pl_rtl_jobs QUERY NUMBER_OF_LOGICAL_CORES)
    set(PL_RTL_VFLAGS -Wall -Wno-DECLFILENAME "${PL_RTL_WAIVERS}" --top-module PoolingKernel)

    add_custom_target(lint_pool_rtl
        COMMAND "${PL_RTL_VERILATOR}" --lint-only ${PL_RTL_VFLAGS} ${PL_RTL_SOURCES}
        COMMENT "PoolingKernel RTL: Verilator lint"
        VERBATIM)

    # pl_rtl_tb(<dir> <opt> <extra flags...>) — one Verilator build of the testbench
    function(pl_rtl_tb dir opt)
        set(_out "${CMAKE_CURRENT_BINARY_DIR}/${dir}/Vtb")
        add_custom_command(
            OUTPUT "${_out}"
            COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/${dir}"
            COMMAND "${PL_RTL_VERILATOR}" --cc --exe --build -j ${_pl_rtl_jobs} ${opt}
                    --x-assign unique --x-initial unique ${ARGN} ${_pl_rtl_oracle_flags} ${PL_RTL_VFLAGS}
                    -Mdir "${CMAKE_CURRENT_BINARY_DIR}/${dir}" -o Vtb
                    "${PL_RTL_DIR}/tb/verilator/tb_main.cpp" ${_pl_rtl_oracle_src} ${PL_RTL_SOURCES}
            DEPENDS ${PL_RTL_SOURCES} ${PL_RTL_TB_SOURCES} "${PL_RTL_WAIVERS}" ${_pl_rtl_oracle_src}
                    "${PL_RTL_REPO}/kernels/pool/include/PoolingKernel.h"
            COMMENT "PoolingKernel RTL: Verilator testbench -> ${dir}/Vtb"
            VERBATIM)
    endfunction()
    pl_rtl_tb(vl -O3 -CFLAGS -O2 -CFLAGS -std=c++17)
    pl_rtl_tb(vlt -O2 --trace-fst -CFLAGS -O2 -CFLAGS -std=c++17)

    set(PL_RTL_TB "${CMAKE_CURRENT_BINARY_DIR}/vl/Vtb")
    add_custom_target(TestPoolRtl ALL DEPENDS "${PL_RTL_TB}")
    add_custom_target(pool_rtl_tb_fst DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/vlt/Vtb")
    add_custom_target(perf_pool_rtl
        COMMAND "${PL_RTL_TB}" --perf
        DEPENDS "${PL_RTL_TB}"
        COMMENT "PoolingKernel RTL: cycle counts (ideal memory)"
        USES_TERMINAL VERBATIM)

    add_test(NAME TestPoolRtl
             COMMAND "${PL_RTL_TB}" --fixtures "${PL_RTL_FIXTURES}"
                     ${_pl_rtl_ctest_random} --seed 1 --timing rand --quiet)
    message(STATUS "PoolingKernel RTL: targets TestPoolRtl, lint_pool_rtl, perf_pool_rtl, "
                   "pool_rtl_tb_fst")
else()
    message(STATUS "PoolingKernel RTL: Verilator not found — TestPoolRtl and lint_pool_rtl disabled "
                   "(sudo apt install verilator)")
endif()

# ---------------------------------------------------------------------------
# Vivado: IP packaging, out-of-context synthesis, xsim
# ---------------------------------------------------------------------------
find_program(PL_RTL_VIVADO vivado
    HINTS $ENV{XILINX_VIVADO}/bin /mnt/data/xilinx/2025.2/Vivado/bin
    DOC "Vivado, for packaging the RTL PoolingKernel IP")

if(PL_RTL_VIVADO)
    get_filename_component(_pl_rtl_vivado_bin "${PL_RTL_VIVADO}" DIRECTORY)
    # Outside <build>/kernels on purpose: a build that scans that tree as its
    # IP repository must not see two IPs with the same VLNV.
    set(PL_RTL_IP_DIR "${CMAKE_BINARY_DIR}/rtl_ip")
    set(PL_RTL_IP "${PL_RTL_IP_DIR}/PoolingKernel_ip")
    set(_pl_rtl_pkg "${CMAKE_CURRENT_BINARY_DIR}/package")
    file(MAKE_DIRECTORY "${PL_RTL_IP_DIR}" "${_pl_rtl_pkg}" "${CMAKE_CURRENT_BINARY_DIR}/synth"
                        "${CMAKE_CURRENT_BINARY_DIR}/xsim" "${CMAKE_CURRENT_BINARY_DIR}/sysim")

    # package_ip.tcl works in its current directory (./PoolingKernel_ip, ./proj):
    # a directory of its own, then the IP is moved next to the other RTL IPs.
    add_custom_command(
        OUTPUT "${PL_RTL_IP}/component.xml" "${PL_RTL_IP_DIR}/PoolingKernel_ip.zip"
        COMMAND "${PL_RTL_VIVADO}" -mode batch -nojournal -nolog
                -source "${PL_RTL_DIR}/syn/package_ip.tcl"
                -tclargs "${PL_RTL_DIR}" "${PL_RTL_DRIVER_ROOT}" "${PL_RTL_PART}"
        COMMAND "${CMAKE_COMMAND}" -E rm -rf "${PL_RTL_IP}" "${PL_RTL_IP_DIR}/PoolingKernel_ip.zip"
        COMMAND "${CMAKE_COMMAND}" -E rename "${_pl_rtl_pkg}/PoolingKernel_ip" "${PL_RTL_IP}"
        COMMAND "${CMAKE_COMMAND}" -E rename "${_pl_rtl_pkg}/PoolingKernel_ip.zip"
                "${PL_RTL_IP_DIR}/PoolingKernel_ip.zip"
        DEPENDS ${PL_RTL_SOURCES} "${PL_RTL_DIR}/syn/package_ip.tcl" "${PL_RTL_DIR}/syn/rtl_files.tcl"
                ${_pl_rtl_driver_files} pool_rtl_driver
        WORKING_DIRECTORY "${_pl_rtl_pkg}"
        COMMENT "PoolingKernel RTL: Vivado IP -> ${PL_RTL_IP}"
        VERBATIM)
    add_custom_target(package_pool_rtl DEPENDS "${PL_RTL_IP}/component.xml")

    add_custom_target(synth_pool_rtl
        COMMAND "${PL_RTL_VIVADO}" -mode batch -nojournal -nolog
                -source "${PL_RTL_DIR}/syn/synth_ooc.tcl"
                -tclargs "${PL_RTL_DIR}" ${PL_RTL_PERIOD} "${PL_RTL_PART}"
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/synth"
        COMMENT "PoolingKernel RTL: Vivado OOC synthesis + P&R (${PL_RTL_PART}, ${PL_RTL_PERIOD} ns)"
        USES_TERMINAL VERBATIM)

    # netlist equivalence: Vivado's OOC synthesis of the IP top against the RTL,
    # in lockstep in xsim on board-like jobs (tools/neteq) — catches RTL that
    # synthesises differently from how it simulates
    add_custom_target(neteq_pool_rtl
        COMMAND "${CMAKE_COMMAND}" -E env "PATH=${_pl_rtl_vivado_bin}:$ENV{PATH}"
                "${PL_RTL_DIR}/../../tools/neteq/run.sh" pool "${PL_RTL_DIR}/rtl"
                "${CMAKE_CURRENT_BINARY_DIR}/neteq"
        COMMENT "PoolingKernel RTL: synthesised netlist vs RTL in lockstep (xsim)"
        USES_TERMINAL VERBATIM)

    add_custom_target(xsim_pool_rtl
        COMMAND "${_pl_rtl_vivado_bin}/xvlog" -sv ${PL_RTL_SOURCES}
        COMMAND "${_pl_rtl_vivado_bin}/xelab" -debug off PoolingKernel -s pl_elab
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/xsim"
        COMMENT "PoolingKernel RTL: xvlog / xelab"
        VERBATIM)

    # System level: a copy of the test stand's pooling_test project with this
    # IP in place of the HLS export (syn/sysim.tcl).
    set(_pl_rtl_stand "${PL_RTL_REPO}/hw/cormorant_test_stand/kernels/pooling_test")
    if(EXISTS "${_pl_rtl_stand}")
        set(_pl_rtl_sysim "${CMAKE_CURRENT_BINARY_DIR}/sysim")
        set(_pl_rtl_sysim_proj "${_pl_rtl_sysim}/pooling_test")
        add_custom_target(sysim_pool_rtl
            COMMAND "${CMAKE_COMMAND}" -E rm -rf "${_pl_rtl_sysim_proj}"
            COMMAND "${CMAKE_COMMAND}" -E copy_directory "${_pl_rtl_stand}" "${_pl_rtl_sysim_proj}"
            COMMAND "${CMAKE_COMMAND}" -E rm -rf "${_pl_rtl_sysim_proj}/pooling_test.sim"
            COMMAND "${PL_RTL_VIVADO}" -mode batch -nojournal -log "${_pl_rtl_sysim}/sysim.log"
                    -source "${PL_RTL_DIR}/syn/sysim.tcl"
                    -tclargs "${_pl_rtl_sysim_proj}/pooling_test.xpr" "${PL_RTL_IP}"
                             "${PL_RTL_FIXTURES}" "${_pl_rtl_sysim}/sysim_report.json"
            COMMAND "${Python3_EXECUTABLE}" "${PL_RTL_REPO}/cmake/check_test_report.py"
                    "${_pl_rtl_sysim}/sysim_report.json"
            DEPENDS "${PL_RTL_IP}/component.xml"
            WORKING_DIRECTORY "${_pl_rtl_sysim}"
            COMMENT "PoolingKernel RTL: test-stand block design in xsim (log: ${_pl_rtl_sysim}/sysim.log)"
            USES_TERMINAL VERBATIM)
        set(_pl_rtl_sysim_msg ", sysim_pool_rtl")
    endif()
    message(STATUS "PoolingKernel RTL: Vivado targets package_pool_rtl, synth_pool_rtl, "
                   "xsim_pool_rtl${_pl_rtl_sysim_msg} (part ${PL_RTL_PART})")
else()
    message(STATUS "PoolingKernel RTL: vivado not found — package_pool_rtl / synth_pool_rtl disabled")
endif()
