# Copyright 2026 Ivan Gubochkin
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.19)
project(RowOpsKernelRtl VERSION 1.0 LANGUAGES C CXX)

# The platform JSONs (AXI_PLATFORM_FILES, AXI_DEFAULT_PLATFORM_JSON): set by the
# top-level project, or here when this kernel is configured on its own.
list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../cmake")
include(AxiPlatform)
get_filename_component(RO_RTL_REPO "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
enable_testing()

# ---------------------------------------------------------------------------
# RowOpsKernel in SystemVerilog — the row-ops unit of doc/plans/HOSTOPS_PLAN.md
# (doc/kernels/ROWOPS_RTL_KERNEL.md): the transformer graphs' host row ops in
# fixed point, bit for bit inference-scheduler/src/rowops_spec.py.
#
# Targets (each one only when its tool is found):
#   TestRowOpsRtl          Verilator testbench (vl/Vtb); ctest runs the checked-in
#                          fixtures (written from the Python specification, the chain
#                          fixtures among them), random cases and random descriptor
#                          chains against the C++ oracle (tb/verilator/rowops_ref.h)
#   lint_rowops_rtl        verilator --lint-only -Wall
#   perf_rowops_rtl        cycle counts of the shipped models' job shapes (ideal memory)
#   rowops_rtl_tb_fst      testbench with FST tracing (vlt/Vtb --trace F.fst)
#   driver_rowops_rtl      the C driver (scripts/gen_driver.py) -> driver/
#   package_rowops_rtl     Vivado IP catalog -> <build>/rtl_ip/RowOpsKernel_ip (+ .zip)
#   synth_rowops_rtl       Vivado out-of-context synthesis + P&R -> synth/
#   xsim_rowops_rtl        xvlog / xelab parse and elaboration
# ---------------------------------------------------------------------------

set(RO_RTL_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(RO_RTL_SOURCES
    rtl/ro_pkg.sv rtl/ro_fifo.sv rtl/ro_rs.sv rtl/ro_ctrl_s_axi.sv rtl/ro_burstgen.sv
    rtl/ro_rd_port.sv rtl/ro_wr_port.sv rtl/ro_rd_arb.sv rtl/ro_prm_ram.sv rtl/ro_param.sv rtl/ro_sig_ram.sv
    rtl/ro_stream.sv rtl/ro_rsqrt_ram.sv rtl/ro_mul32.sv rtl/ro_norm.sv rtl/ro_rope.sv rtl/ro_xpose.sv rtl/ro_fold.sv rtl/ro_core.sv rtl/RowOpsKernel.v)
list(TRANSFORM RO_RTL_SOURCES PREPEND "${RO_RTL_DIR}/")
set(RO_RTL_TB_SOURCES
    "${RO_RTL_DIR}/tb/verilator/tb_main.cpp"
    "${RO_RTL_DIR}/tb/verilator/rowops_ref.h"
    "${RO_RTL_DIR}/tb/verilator/axi_models.h")
set(RO_RTL_WAIVERS "${RO_RTL_DIR}/scripts/lint_waivers.vlt")
set(RO_RTL_FIXTURES "${RO_RTL_REPO}/hw/test_data/rowops_test_data"
    CACHE PATH "RowOpsKernel fixtures (manifest.txt + test_NN_*.hex) for TestRowOpsRtl")
set(RO_RTL_RANDOM_CASES 300 CACHE STRING "TestRowOpsRtl: random cases after the fixtures")
set(RO_RTL_CHAIN_CASES 300 CACHE STRING "TestRowOpsRtl: random descriptor chains after the random cases")
set(RO_RTL_PERIOD 3.333 CACHE STRING "synth_rowops_rtl: clock period in ns (OOC timing check)")

find_package(Python3 COMPONENTS Interpreter REQUIRED)

file(READ "${AXI_DEFAULT_PLATFORM_JSON}" _ro_rtl_pjson)
string(JSON RO_RTL_PART GET "${_ro_rtl_pjson}" part)

# ---------------------------------------------------------------------------
# The C driver (scripts/gen_driver.py, the RTL MatmulKernel's generator); the
# register table is checked against rtl/ro_ctrl_s_axi.sv by ctest (RowOpsRtlDriver).
# ---------------------------------------------------------------------------
set(RO_RTL_DRIVER_ROOT "${CMAKE_CURRENT_BINARY_DIR}/driver")
set(RO_RTL_DRIVER_SRC "${RO_RTL_DRIVER_ROOT}/RowOpsKernel_v1_0/src")
set(_ro_rtl_gen "${RO_RTL_DIR}/scripts/gen_driver.py" "${RO_RTL_REPO}/kernels/matmul_rtl/scripts/gen_driver.py")
set(_ro_rtl_driver_files
    "${RO_RTL_DRIVER_SRC}/xrowopskernel.c" "${RO_RTL_DRIVER_SRC}/xrowopskernel_linux.c"
    "${RO_RTL_DRIVER_SRC}/xrowopskernel_sinit.c" "${RO_RTL_DRIVER_SRC}/xrowopskernel.h"
    "${RO_RTL_DRIVER_SRC}/xrowopskernel_hw.h")
add_custom_command(
    OUTPUT ${_ro_rtl_driver_files}
    COMMAND "${Python3_EXECUTABLE}" "${RO_RTL_DIR}/scripts/gen_driver.py" --out "${RO_RTL_DRIVER_ROOT}"
    DEPENDS ${_ro_rtl_gen}
    COMMENT "RowOpsKernel RTL: C driver -> ${RO_RTL_DRIVER_ROOT}"
    VERBATIM)
add_library(rowops_rtl_driver OBJECT
    "${RO_RTL_DRIVER_SRC}/xrowopskernel.c" "${RO_RTL_DRIVER_SRC}/xrowopskernel_linux.c")
target_compile_options(rowops_rtl_driver PRIVATE -Wall -Wextra -Werror)
add_custom_target(driver_rowops_rtl DEPENDS rowops_rtl_driver)
add_test(NAME RowOpsRtlDriver
         COMMAND "${Python3_EXECUTABLE}" "${RO_RTL_DIR}/scripts/gen_driver.py" --check)

# ---------------------------------------------------------------------------
# Verilator: lint and the C++ testbench
# ---------------------------------------------------------------------------
find_program(RO_RTL_VERILATOR verilator DOC "Verilator (5.x) for the RTL RowOpsKernel testbench")
if(RO_RTL_VERILATOR)
    execute_process(COMMAND "${RO_RTL_VERILATOR}" --version
                    OUTPUT_VARIABLE _ro_rtl_vl_version OUTPUT_STRIP_TRAILING_WHITESPACE)
    string(REGEX MATCH "[0-9]+\\.[0-9]+" _ro_rtl_vl_version "${_ro_rtl_vl_version}")
    if(_ro_rtl_vl_version VERSION_LESS 5.0)
        message(WARNING "RowOpsKernel RTL: Verilator ${_ro_rtl_vl_version} found, the testbench needs 5.x "
                        "— TestRowOpsRtl disabled")
        unset(RO_RTL_VERILATOR)
    endif()
endif()

if(RO_RTL_VERILATOR)
    message(STATUS "RowOpsKernel RTL: Verilator ${_ro_rtl_vl_version} at ${RO_RTL_VERILATOR}")
    cmake_host_system_information(RESULT _ro_rtl_jobs QUERY NUMBER_OF_LOGICAL_CORES)
    set(RO_RTL_VFLAGS -Wall -Wno-DECLFILENAME "${RO_RTL_WAIVERS}" --top-module RowOpsKernel)

    add_custom_target(lint_rowops_rtl
        COMMAND "${RO_RTL_VERILATOR}" --lint-only ${RO_RTL_VFLAGS} ${RO_RTL_SOURCES}
        COMMENT "RowOpsKernel RTL: Verilator lint"
        VERBATIM)

    function(ro_rtl_tb dir opt)
        set(_out "${CMAKE_CURRENT_BINARY_DIR}/${dir}/Vtb")
        add_custom_command(
            OUTPUT "${_out}"
            COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/${dir}"
            COMMAND "${RO_RTL_VERILATOR}" --cc --exe --build -j ${_ro_rtl_jobs} ${opt}
                    --x-assign unique --x-initial unique ${ARGN} ${RO_RTL_VFLAGS}
                    -Mdir "${CMAKE_CURRENT_BINARY_DIR}/${dir}" -o Vtb
                    "${RO_RTL_DIR}/tb/verilator/tb_main.cpp" ${RO_RTL_SOURCES}
            DEPENDS ${RO_RTL_SOURCES} ${RO_RTL_TB_SOURCES} "${RO_RTL_WAIVERS}"
            COMMENT "RowOpsKernel RTL: Verilator testbench -> ${dir}/Vtb"
            VERBATIM)
    endfunction()
    ro_rtl_tb(vl -O3 -CFLAGS -O2 -CFLAGS -std=c++17)
    ro_rtl_tb(vlt -O2 --trace-fst -CFLAGS -O2 -CFLAGS -std=c++17)

    set(RO_RTL_TB "${CMAKE_CURRENT_BINARY_DIR}/vl/Vtb")
    add_custom_target(TestRowOpsRtl ALL DEPENDS "${RO_RTL_TB}")
    add_custom_target(rowops_rtl_tb_fst DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/vlt/Vtb")
    add_custom_target(perf_rowops_rtl
        COMMAND "${RO_RTL_TB}" --perf
        DEPENDS "${RO_RTL_TB}"
        COMMENT "RowOpsKernel RTL: cycle counts (ideal memory)"
        USES_TERMINAL VERBATIM)

    add_test(NAME TestRowOpsRtl
             COMMAND "${RO_RTL_TB}" --fixtures "${RO_RTL_FIXTURES}"
                     --random ${RO_RTL_RANDOM_CASES} --chains ${RO_RTL_CHAIN_CASES} --seed 1 --timing rand --quiet)
    message(STATUS "RowOpsKernel RTL: targets TestRowOpsRtl, lint_rowops_rtl, perf_rowops_rtl, rowops_rtl_tb_fst")
else()
    message(STATUS "RowOpsKernel RTL: Verilator not found — TestRowOpsRtl and lint_rowops_rtl disabled "
                   "(sudo apt install verilator)")
endif()

# ---------------------------------------------------------------------------
# Vivado: IP packaging, out-of-context synthesis, xsim
# ---------------------------------------------------------------------------
find_program(RO_RTL_VIVADO vivado
    HINTS $ENV{XILINX_VIVADO}/bin /mnt/data/xilinx/2025.2/Vivado/bin
    DOC "Vivado, for packaging the RTL RowOpsKernel IP")

if(RO_RTL_VIVADO)
    get_filename_component(_ro_rtl_vivado_bin "${RO_RTL_VIVADO}" DIRECTORY)
    set(RO_RTL_IP_DIR "${CMAKE_BINARY_DIR}/rtl_ip")
    set(RO_RTL_IP "${RO_RTL_IP_DIR}/RowOpsKernel_ip")
    set(_ro_rtl_pkg "${CMAKE_CURRENT_BINARY_DIR}/package")
    file(MAKE_DIRECTORY "${RO_RTL_IP_DIR}" "${_ro_rtl_pkg}" "${CMAKE_CURRENT_BINARY_DIR}/synth"
                        "${CMAKE_CURRENT_BINARY_DIR}/xsim")

    add_custom_command(
        OUTPUT "${RO_RTL_IP}/component.xml" "${RO_RTL_IP_DIR}/RowOpsKernel_ip.zip"
        COMMAND "${RO_RTL_VIVADO}" -mode batch -nojournal -nolog
                -source "${RO_RTL_DIR}/syn/package_ip.tcl"
                -tclargs "${RO_RTL_DIR}" "${RO_RTL_DRIVER_ROOT}" "${RO_RTL_PART}"
        COMMAND "${CMAKE_COMMAND}" -E rm -rf "${RO_RTL_IP}" "${RO_RTL_IP_DIR}/RowOpsKernel_ip.zip"
        COMMAND "${CMAKE_COMMAND}" -E rename "${_ro_rtl_pkg}/RowOpsKernel_ip" "${RO_RTL_IP}"
        COMMAND "${CMAKE_COMMAND}" -E rename "${_ro_rtl_pkg}/RowOpsKernel_ip.zip"
                "${RO_RTL_IP_DIR}/RowOpsKernel_ip.zip"
        DEPENDS ${RO_RTL_SOURCES} "${RO_RTL_DIR}/syn/package_ip.tcl" "${RO_RTL_DIR}/syn/rtl_files.tcl"
                ${_ro_rtl_driver_files} rowops_rtl_driver
        WORKING_DIRECTORY "${_ro_rtl_pkg}"
        COMMENT "RowOpsKernel RTL: Vivado IP -> ${RO_RTL_IP}"
        VERBATIM)
    add_custom_target(package_rowops_rtl DEPENDS "${RO_RTL_IP}/component.xml")

    add_custom_target(synth_rowops_rtl
        COMMAND "${RO_RTL_VIVADO}" -mode batch -nojournal -nolog
                -source "${RO_RTL_DIR}/syn/synth_ooc.tcl"
                -tclargs "${RO_RTL_DIR}" ${RO_RTL_PERIOD} "${RO_RTL_PART}"
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/synth"
        COMMENT "RowOpsKernel RTL: Vivado OOC synthesis + P&R (${RO_RTL_PART}, ${RO_RTL_PERIOD} ns)"
        USES_TERMINAL VERBATIM)

    # --relax: ro_norm / ro_rope / ro_xpose use a few signals above their declarations,
    # which strict xvlog rejects (Verilator and synthesis accept it)
    add_custom_target(xsim_rowops_rtl
        COMMAND "${_ro_rtl_vivado_bin}/xvlog" --relax -sv ${RO_RTL_SOURCES}
        COMMAND "${_ro_rtl_vivado_bin}/xelab" -relax -debug off RowOpsKernel -s ro_elab
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/xsim"
        COMMENT "RowOpsKernel RTL: xvlog / xelab"
        VERBATIM)
    message(STATUS "RowOpsKernel RTL: Vivado targets package_rowops_rtl, synth_rowops_rtl, xsim_rowops_rtl "
                   "(part ${RO_RTL_PART})")
else()
    message(STATUS "RowOpsKernel RTL: vivado not found — package_rowops_rtl / synth_rowops_rtl disabled")
endif()
