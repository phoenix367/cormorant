cmake_minimum_required(VERSION 3.19)
project(VectorOPKernelRtl VERSION 1.0 LANGUAGES C CXX)

# The platform JSONs (AXI_PLATFORM_FILES, AXI_DEFAULT_PLATFORM_JSON): set by the
# top-level project, or here when this kernel is configured on its own.
list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../cmake")
include(AxiPlatform)
get_filename_component(VO_RTL_REPO "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
enable_testing()

# ---------------------------------------------------------------------------
# VectorOPKernel in SystemVerilog — the hardware build's VectorOPKernel, a
# drop-in for the retired Vitis HLS kernel (kernels/vectorop keeps its C++ as
# the reference model): same VLNV, register map, m_axi bus parameters, DDR
# access pattern and arithmetic.  See doc/kernels/VECTOROP_RTL_KERNEL.md and
# doc/plans/VECTOROP_RTL_PLAN.md.
#
# Targets (each one only when its tool is found):
#   TestVectorOpRtl        Verilator testbench (vl/Vtb); ctest runs the
#                          HLS-oracle fixtures + 300 random cases checked
#                          against the HLS kernel's C++ (with the Vitis HLS headers)
#   lint_vectorop_rtl      verilator --lint-only -Wall
#   perf_vectorop_rtl      cycle counts of typical jobs (ideal memory)
#   vectorop_rtl_tb_fst    testbench with FST tracing (vlt/Vtb --trace F.fst)
#   driver_vectorop_rtl    the C driver (scripts/gen_driver.py) -> driver/
#   package_vectorop_rtl   Vivado IP catalog -> <build>/rtl_ip/VectorOPKernel_ip (+ .zip)
#   synth_vectorop_rtl     Vivado out-of-context synthesis + P&R -> synth/
#   xsim_vectorop_rtl      xvlog / xelab parse and elaboration
#   sysim_vectorop_rtl     the test stand's VectorOPKernel block design with
#                          this IP in xsim, on a copy
# ---------------------------------------------------------------------------

set(VO_RTL_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(VO_RTL_SOURCES
    rtl/vo_pkg.sv rtl/vo_fifo.sv rtl/vo_rs.sv rtl/vo_ctrl_s_axi.sv rtl/vo_burstgen.sv
    rtl/vo_rd_port.sv rtl/vo_wr_port.sv rtl/vo_div.sv rtl/vo_compute.sv
    rtl/vo_core.sv rtl/VectorOPKernel.v)
list(TRANSFORM VO_RTL_SOURCES PREPEND "${VO_RTL_DIR}/")
set(VO_RTL_TB_SOURCES
    "${VO_RTL_DIR}/tb/verilator/tb_main.cpp"
    "${VO_RTL_DIR}/tb/verilator/axi_models.h")
set(VO_RTL_WAIVERS "${VO_RTL_DIR}/scripts/lint_waivers.vlt")
set(VO_RTL_FIXTURES "${VO_RTL_REPO}/hw/test_data/vecop_test_data"
    CACHE PATH "VectorOPKernel fixtures (manifest.txt + test_NN_{a,b,c}.hex) for TestVectorOpRtl")
set(VO_RTL_RANDOM_CASES 300 CACHE STRING "TestVectorOpRtl: random cases after the fixtures")
set(VO_RTL_PERIOD 3.333 CACHE STRING "synth_vectorop_rtl: clock period in ns (OOC timing check)")

find_package(Python3 COMPONENTS Interpreter REQUIRED)

file(READ "${AXI_DEFAULT_PLATFORM_JSON}" _vo_rtl_pjson)
string(JSON VO_RTL_PART GET "${_vo_rtl_pjson}" part)

# ---------------------------------------------------------------------------
# The C driver: the API and register map of the HLS-generated xvectoropkernel,
# written by scripts/gen_driver.py (the RTL MatmulKernel's generator); the
# register table is checked against rtl/vo_ctrl_s_axi.sv by ctest
# (VectorOpRtlDriver) and compiled here.
# ---------------------------------------------------------------------------
set(VO_RTL_DRIVER_ROOT "${CMAKE_CURRENT_BINARY_DIR}/driver")
set(VO_RTL_DRIVER_SRC "${VO_RTL_DRIVER_ROOT}/VectorOPKernel_v1_0/src")
set(_vo_rtl_gen "${VO_RTL_DIR}/scripts/gen_driver.py" "${VO_RTL_REPO}/kernels/matmul_rtl/scripts/gen_driver.py")
set(_vo_rtl_driver_files
    "${VO_RTL_DRIVER_SRC}/xvectoropkernel.c" "${VO_RTL_DRIVER_SRC}/xvectoropkernel_linux.c"
    "${VO_RTL_DRIVER_SRC}/xvectoropkernel_sinit.c" "${VO_RTL_DRIVER_SRC}/xvectoropkernel.h"
    "${VO_RTL_DRIVER_SRC}/xvectoropkernel_hw.h")
add_custom_command(
    OUTPUT ${_vo_rtl_driver_files}
    COMMAND "${Python3_EXECUTABLE}" "${VO_RTL_DIR}/scripts/gen_driver.py" --out "${VO_RTL_DRIVER_ROOT}"
    DEPENDS ${_vo_rtl_gen}
    COMMENT "VectorOPKernel RTL: C driver -> ${VO_RTL_DRIVER_ROOT}"
    VERBATIM)
add_library(vectorop_rtl_driver OBJECT
    "${VO_RTL_DRIVER_SRC}/xvectoropkernel.c" "${VO_RTL_DRIVER_SRC}/xvectoropkernel_linux.c")
target_compile_options(vectorop_rtl_driver PRIVATE -Wall -Wextra -Werror)
add_custom_target(driver_vectorop_rtl DEPENDS vectorop_rtl_driver)
add_test(NAME VectorOpRtlDriver
         COMMAND "${Python3_EXECUTABLE}" "${VO_RTL_DIR}/scripts/gen_driver.py" --check)

# ---------------------------------------------------------------------------
# The HLS oracle: kernels/vectorop/kernel/VectorOP.cpp built with the Vitis
# HLS headers into the testbench (Data_t = ap_fixed<16,8>, as the bitstream).
# ---------------------------------------------------------------------------
find_path(VO_RTL_HLS_INCLUDE NAMES ap_fixed.h hls_burst_maxi.h
    HINTS $ENV{XILINX_HLS}/include $ENV{XILINX_VITIS}/include /mnt/data/xilinx/2025.2/Vitis/include
    NO_DEFAULT_PATH
    DOC "Vitis HLS include directory (ap_fixed.h): the testbench's HLS oracle")
set(_vo_rtl_oracle_flags)
set(_vo_rtl_oracle_src)
if(VO_RTL_HLS_INCLUDE)
    set(_vo_rtl_cfg "${CMAKE_CURRENT_BINARY_DIR}/oracle")
    set(VA_DATA_TYPE_INCLUDE "#include \"ap_fixed.h\"")
    set(VA_DATA_TYPE "ap_fixed<16,8>")
    set(VA_DATA_WIDTH_BITS 16)
    configure_file("${VO_RTL_REPO}/kernels/vectorop/include/Config.h.in" "${_vo_rtl_cfg}/Config.h")
    set(_vo_rtl_oracle_src "${VO_RTL_REPO}/kernels/vectorop/kernel/VectorOP.cpp")
    set(_vo_rtl_oracle_flags -CFLAGS -DVO_HLS_ORACLE -CFLAGS -I${_vo_rtl_cfg}
        -CFLAGS -I${VO_RTL_REPO}/kernels/vectorop/include -CFLAGS -I${VO_RTL_HLS_INCLUDE}
        -CFLAGS -Wno-unused -CFLAGS -Wno-unknown-pragmas)
    message(STATUS "VectorOPKernel RTL: testbench oracle = the HLS kernel's C++ (${VO_RTL_HLS_INCLUDE})")
    set(_vo_rtl_ctest_random --random ${VO_RTL_RANDOM_CASES})
else()
    message(STATUS "VectorOPKernel RTL: Vitis HLS headers not found — the testbench checks the fixtures only")
    set(_vo_rtl_ctest_random)
endif()

# ---------------------------------------------------------------------------
# Verilator: lint and the C++ testbench
# ---------------------------------------------------------------------------
find_program(VO_RTL_VERILATOR verilator DOC "Verilator (5.x) for the RTL VectorOPKernel testbench")
if(VO_RTL_VERILATOR)
    execute_process(COMMAND "${VO_RTL_VERILATOR}" --version
                    OUTPUT_VARIABLE _vo_rtl_vl_version OUTPUT_STRIP_TRAILING_WHITESPACE)
    string(REGEX MATCH "[0-9]+\\.[0-9]+" _vo_rtl_vl_version "${_vo_rtl_vl_version}")
    if(_vo_rtl_vl_version VERSION_LESS 5.0)
        message(WARNING "VectorOPKernel RTL: Verilator ${_vo_rtl_vl_version} found, the testbench needs 5.x "
                        "— TestVectorOpRtl disabled")
        unset(VO_RTL_VERILATOR)
    endif()
endif()

if(VO_RTL_VERILATOR)
    message(STATUS "VectorOPKernel RTL: Verilator ${_vo_rtl_vl_version} at ${VO_RTL_VERILATOR}")
    cmake_host_system_information(RESULT _vo_rtl_jobs QUERY NUMBER_OF_LOGICAL_CORES)
    set(VO_RTL_VFLAGS -Wall -Wno-DECLFILENAME "${VO_RTL_WAIVERS}" --top-module VectorOPKernel)

    add_custom_target(lint_vectorop_rtl
        COMMAND "${VO_RTL_VERILATOR}" --lint-only ${VO_RTL_VFLAGS} ${VO_RTL_SOURCES}
        COMMENT "VectorOPKernel RTL: Verilator lint"
        VERBATIM)

    # vo_rtl_tb(<dir> <opt> <extra flags...>) — one Verilator build of the testbench
    function(vo_rtl_tb dir opt)
        set(_out "${CMAKE_CURRENT_BINARY_DIR}/${dir}/Vtb")
        add_custom_command(
            OUTPUT "${_out}"
            COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/${dir}"
            COMMAND "${VO_RTL_VERILATOR}" --cc --exe --build -j ${_vo_rtl_jobs} ${opt}
                    --x-assign unique --x-initial unique ${ARGN} ${_vo_rtl_oracle_flags} ${VO_RTL_VFLAGS}
                    -Mdir "${CMAKE_CURRENT_BINARY_DIR}/${dir}" -o Vtb
                    "${VO_RTL_DIR}/tb/verilator/tb_main.cpp" ${_vo_rtl_oracle_src} ${VO_RTL_SOURCES}
            DEPENDS ${VO_RTL_SOURCES} ${VO_RTL_TB_SOURCES} "${VO_RTL_WAIVERS}" ${_vo_rtl_oracle_src}
                    "${VO_RTL_REPO}/kernels/vectorop/include/VectorOP.h"
            COMMENT "VectorOPKernel RTL: Verilator testbench -> ${dir}/Vtb"
            VERBATIM)
    endfunction()
    vo_rtl_tb(vl -O3 -CFLAGS -O2 -CFLAGS -std=c++17)
    vo_rtl_tb(vlt -O2 --trace-fst -CFLAGS -O2 -CFLAGS -std=c++17)

    set(VO_RTL_TB "${CMAKE_CURRENT_BINARY_DIR}/vl/Vtb")
    add_custom_target(TestVectorOpRtl ALL DEPENDS "${VO_RTL_TB}")
    add_custom_target(vectorop_rtl_tb_fst DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/vlt/Vtb")
    add_custom_target(perf_vectorop_rtl
        COMMAND "${VO_RTL_TB}" --perf
        DEPENDS "${VO_RTL_TB}"
        COMMENT "VectorOPKernel RTL: cycle counts (ideal memory)"
        USES_TERMINAL VERBATIM)

    add_test(NAME TestVectorOpRtl
             COMMAND "${VO_RTL_TB}" --fixtures "${VO_RTL_FIXTURES}"
                     ${_vo_rtl_ctest_random} --seed 1 --timing rand --quiet)
    message(STATUS "VectorOPKernel RTL: targets TestVectorOpRtl, lint_vectorop_rtl, perf_vectorop_rtl, "
                   "vectorop_rtl_tb_fst")
else()
    message(STATUS "VectorOPKernel RTL: Verilator not found — TestVectorOpRtl and lint_vectorop_rtl disabled "
                   "(sudo apt install verilator)")
endif()

# ---------------------------------------------------------------------------
# Vivado: IP packaging, out-of-context synthesis, xsim
# ---------------------------------------------------------------------------
find_program(VO_RTL_VIVADO vivado
    HINTS $ENV{XILINX_VIVADO}/bin /mnt/data/xilinx/2025.2/Vivado/bin
    DOC "Vivado, for packaging the RTL VectorOPKernel IP")

if(VO_RTL_VIVADO)
    get_filename_component(_vo_rtl_vivado_bin "${VO_RTL_VIVADO}" DIRECTORY)
    # Outside <build>/kernels on purpose: a build that scans that tree as its
    # IP repository must not see two IPs with the same VLNV.
    set(VO_RTL_IP_DIR "${CMAKE_BINARY_DIR}/rtl_ip")
    set(VO_RTL_IP "${VO_RTL_IP_DIR}/VectorOPKernel_ip")
    set(_vo_rtl_pkg "${CMAKE_CURRENT_BINARY_DIR}/package")
    file(MAKE_DIRECTORY "${VO_RTL_IP_DIR}" "${_vo_rtl_pkg}" "${CMAKE_CURRENT_BINARY_DIR}/synth"
                        "${CMAKE_CURRENT_BINARY_DIR}/xsim" "${CMAKE_CURRENT_BINARY_DIR}/sysim")

    # package_ip.tcl works in its current directory (./VectorOPKernel_ip, ./proj):
    # a directory of its own, then the IP is moved next to the MatmulKernel's.
    add_custom_command(
        OUTPUT "${VO_RTL_IP}/component.xml" "${VO_RTL_IP_DIR}/VectorOPKernel_ip.zip"
        COMMAND "${VO_RTL_VIVADO}" -mode batch -nojournal -nolog
                -source "${VO_RTL_DIR}/syn/package_ip.tcl"
                -tclargs "${VO_RTL_DIR}" "${VO_RTL_DRIVER_ROOT}" "${VO_RTL_PART}"
        COMMAND "${CMAKE_COMMAND}" -E rm -rf "${VO_RTL_IP}" "${VO_RTL_IP_DIR}/VectorOPKernel_ip.zip"
        COMMAND "${CMAKE_COMMAND}" -E rename "${_vo_rtl_pkg}/VectorOPKernel_ip" "${VO_RTL_IP}"
        COMMAND "${CMAKE_COMMAND}" -E rename "${_vo_rtl_pkg}/VectorOPKernel_ip.zip"
                "${VO_RTL_IP_DIR}/VectorOPKernel_ip.zip"
        DEPENDS ${VO_RTL_SOURCES} "${VO_RTL_DIR}/syn/package_ip.tcl" "${VO_RTL_DIR}/syn/rtl_files.tcl"
                ${_vo_rtl_driver_files} vectorop_rtl_driver
        WORKING_DIRECTORY "${_vo_rtl_pkg}"
        COMMENT "VectorOPKernel RTL: Vivado IP -> ${VO_RTL_IP}"
        VERBATIM)
    add_custom_target(package_vectorop_rtl DEPENDS "${VO_RTL_IP}/component.xml")

    add_custom_target(synth_vectorop_rtl
        COMMAND "${VO_RTL_VIVADO}" -mode batch -nojournal -nolog
                -source "${VO_RTL_DIR}/syn/synth_ooc.tcl"
                -tclargs "${VO_RTL_DIR}" ${VO_RTL_PERIOD} "${VO_RTL_PART}"
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/synth"
        COMMENT "VectorOPKernel RTL: Vivado OOC synthesis + P&R (${VO_RTL_PART}, ${VO_RTL_PERIOD} ns)"
        USES_TERMINAL VERBATIM)

    # netlist equivalence: Vivado's OOC synthesis of the IP top against the RTL,
    # in lockstep in xsim on board-like jobs (tools/neteq) — catches RTL that
    # synthesises differently from how it simulates
    add_custom_target(neteq_vectorop_rtl
        COMMAND "${CMAKE_COMMAND}" -E env "PATH=${_vo_rtl_vivado_bin}:$ENV{PATH}"
                "${VO_RTL_DIR}/../../tools/neteq/run.sh" vectorop "${VO_RTL_DIR}/rtl"
                "${CMAKE_CURRENT_BINARY_DIR}/neteq"
        COMMENT "VectorOPKernel RTL: synthesised netlist vs RTL in lockstep (xsim)"
        USES_TERMINAL VERBATIM)

    add_custom_target(xsim_vectorop_rtl
        COMMAND "${_vo_rtl_vivado_bin}/xvlog" -sv ${VO_RTL_SOURCES}
        COMMAND "${_vo_rtl_vivado_bin}/xelab" -debug off VectorOPKernel -s vo_elab
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/xsim"
        COMMENT "VectorOPKernel RTL: xvlog / xelab"
        VERBATIM)

    # System level: a copy of the test stand's vector_op_test project with this
    # IP in place of the HLS export (syn/sysim.tcl).
    set(_vo_rtl_stand "${VO_RTL_REPO}/hw/cormorant_test_stand/kernels/vector_op_test")
    if(EXISTS "${_vo_rtl_stand}")
        set(_vo_rtl_sysim "${CMAKE_CURRENT_BINARY_DIR}/sysim")
        set(_vo_rtl_sysim_proj "${_vo_rtl_sysim}/vector_op_test")
        add_custom_target(sysim_vectorop_rtl
            COMMAND "${CMAKE_COMMAND}" -E rm -rf "${_vo_rtl_sysim_proj}"
            COMMAND "${CMAKE_COMMAND}" -E copy_directory "${_vo_rtl_stand}" "${_vo_rtl_sysim_proj}"
            COMMAND "${CMAKE_COMMAND}" -E rm -rf "${_vo_rtl_sysim_proj}/vector_op_test.sim"
            COMMAND "${VO_RTL_VIVADO}" -mode batch -nojournal -log "${_vo_rtl_sysim}/sysim.log"
                    -source "${VO_RTL_DIR}/syn/sysim.tcl"
                    -tclargs "${_vo_rtl_sysim_proj}/vector_op_test.xpr" "${VO_RTL_IP}"
                             "${VO_RTL_FIXTURES}" "${_vo_rtl_sysim}/sysim_report.json"
            COMMAND "${Python3_EXECUTABLE}" "${VO_RTL_REPO}/cmake/check_test_report.py"
                    "${_vo_rtl_sysim}/sysim_report.json"
            DEPENDS "${VO_RTL_IP}/component.xml"
            WORKING_DIRECTORY "${_vo_rtl_sysim}"
            COMMENT "VectorOPKernel RTL: test-stand block design in xsim (log: ${_vo_rtl_sysim}/sysim.log)"
            USES_TERMINAL VERBATIM)
        set(_vo_rtl_sysim_msg ", sysim_vectorop_rtl")
    endif()
    message(STATUS "VectorOPKernel RTL: Vivado targets package_vectorop_rtl, synth_vectorop_rtl, "
                   "xsim_vectorop_rtl${_vo_rtl_sysim_msg} (part ${VO_RTL_PART})")
else()
    message(STATUS "VectorOPKernel RTL: vivado not found — package_vectorop_rtl / synth_vectorop_rtl disabled")
endif()
