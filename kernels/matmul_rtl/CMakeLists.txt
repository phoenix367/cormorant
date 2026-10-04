cmake_minimum_required(VERSION 3.19)
project(MatmulKernelRtl VERSION 1.0 LANGUAGES C CXX)

# The platform JSONs (AXI_PLATFORM_FILES, AXI_DEFAULT_PLATFORM_JSON): set by the
# top-level project, or here when this kernel is configured on its own.
list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../cmake")
include(AxiPlatform)
get_filename_component(MM_RTL_REPO "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
enable_testing()

# ---------------------------------------------------------------------------
# MatmulKernel in SystemVerilog — a drop-in replacement for the Vitis HLS
# MatmulKernel (kernels/matmul): same VLNV, register map, DDR layouts and
# arithmetic; gmem2 (C) is 128-bit.  See doc/kernels/MATMUL_RTL_KERNEL.md.
#
# Targets (each one only when its tool is found):
#   TestMatmulRtl          Verilator testbench (vl/Vtb); ctest runs the
#                          HLS-oracle fixtures + 200 random cases
#   lint_matmul_rtl        verilator --lint-only -Wall
#   perf_matmul_rtl        cycle counts of typical shapes (ideal memory)
#   matmul_rtl_tb_fst      testbench with FST tracing (vlt/Vtb --trace F.fst)
#   matmul_rtl_tb_vcd      testbench with VCD tracing (vcd/Vtb --trace F.vcd)
#   driver_matmul_rtl      the C driver (scripts/gen_driver.py) -> driver/
#   package_matmul_rtl     Vivado IP catalog -> <build>/rtl_ip/MatmulKernel_ip (+ .zip)
#   synth_matmul_rtl       Vivado out-of-context synthesis + P&R -> synth/
#   xsim_matmul_rtl        xvlog / xelab parse and elaboration
#   sysim_matmul_rtl       the test stand's MatmulKernel block design with this
#                          IP (gmem2 upgraded to 128) in xsim, on a copy
# ---------------------------------------------------------------------------

set(MM_RTL_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(MM_RTL_SOURCES
    rtl/mm_pkg.sv rtl/mm_fifo.sv rtl/mm_gearbox.sv rtl/mm_axi_rd.sv
    rtl/mm_axi_wr.sv rtl/mm_packer.sv rtl/mm_ctrl_s_axi.sv rtl/mm_walker.sv
    rtl/mm_rungen.sv rtl/mm_abuf.sv rtl/mm_awr.sv rtl/mm_xpf.sv rtl/mm_mac.sv
    rtl/mm_lane.sv rtl/mm_drain.sv rtl/mm_core.sv rtl/MatmulKernel.v)
list(TRANSFORM MM_RTL_SOURCES PREPEND "${MM_RTL_DIR}/")
set(MM_RTL_TB_SOURCES
    "${MM_RTL_DIR}/tb/verilator/tb_main.cpp"
    "${MM_RTL_DIR}/tb/verilator/axi_models.h")
set(MM_RTL_WAIVERS "${MM_RTL_DIR}/scripts/lint_waivers.vlt")
set(MM_RTL_FIXTURES "${MM_RTL_REPO}/hw/test_data/matmul_test_data"
    CACHE PATH "MatmulKernel fixtures (manifest.txt + test_NN_{a,b,c}.hex) for TestMatmulRtl")
set(MM_RTL_RANDOM_CASES 200 CACHE STRING "TestMatmulRtl: random cases after the fixtures")
set(MM_RTL_PERIOD 3.333 CACHE STRING "synth_matmul_rtl: clock period in ns (OOC timing check)")

find_package(Python3 COMPONENTS Interpreter REQUIRED)

# The FPGA part of the default platform (packaging and OOC synthesis).
file(READ "${AXI_DEFAULT_PLATFORM_JSON}" _mm_rtl_pjson)
string(JSON MM_RTL_PART GET "${_mm_rtl_pjson}" part)

# ---------------------------------------------------------------------------
# The C driver: the API and register map of the HLS-generated xmatmulkernel,
# written by scripts/gen_driver.py; the register table is checked against
# rtl/mm_ctrl_s_axi.sv by ctest (MatmulRtlDriver) and compiled here.
# ---------------------------------------------------------------------------
set(MM_RTL_DRIVER_ROOT "${CMAKE_CURRENT_BINARY_DIR}/driver")
set(MM_RTL_DRIVER_SRC "${MM_RTL_DRIVER_ROOT}/MatmulKernel_v1_0/src")
set(_mm_rtl_driver_files
    "${MM_RTL_DRIVER_SRC}/xmatmulkernel.c" "${MM_RTL_DRIVER_SRC}/xmatmulkernel_linux.c"
    "${MM_RTL_DRIVER_SRC}/xmatmulkernel_sinit.c" "${MM_RTL_DRIVER_SRC}/xmatmulkernel.h"
    "${MM_RTL_DRIVER_SRC}/xmatmulkernel_hw.h")
add_custom_command(
    OUTPUT ${_mm_rtl_driver_files}
    COMMAND "${Python3_EXECUTABLE}" "${MM_RTL_DIR}/scripts/gen_driver.py" --out "${MM_RTL_DRIVER_ROOT}"
    DEPENDS "${MM_RTL_DIR}/scripts/gen_driver.py"
    COMMENT "MatmulKernel RTL: C driver -> ${MM_RTL_DRIVER_ROOT}"
    VERBATIM)
# compile it (Linux back end), so a broken driver fails the build
add_library(matmul_rtl_driver OBJECT
    "${MM_RTL_DRIVER_SRC}/xmatmulkernel.c" "${MM_RTL_DRIVER_SRC}/xmatmulkernel_linux.c")
target_compile_options(matmul_rtl_driver PRIVATE -Wall -Wextra -Werror)
add_custom_target(driver_matmul_rtl DEPENDS matmul_rtl_driver)
add_test(NAME MatmulRtlDriver
         COMMAND "${Python3_EXECUTABLE}" "${MM_RTL_DIR}/scripts/gen_driver.py" --check)

# ---------------------------------------------------------------------------
# Verilator: lint and the C++ testbench
# ---------------------------------------------------------------------------
find_program(MM_RTL_VERILATOR verilator DOC "Verilator (5.x) for the RTL MatmulKernel testbench")
if(MM_RTL_VERILATOR)
    execute_process(COMMAND "${MM_RTL_VERILATOR}" --version
                    OUTPUT_VARIABLE _mm_rtl_vl_version OUTPUT_STRIP_TRAILING_WHITESPACE)
    string(REGEX MATCH "[0-9]+\\.[0-9]+" _mm_rtl_vl_version "${_mm_rtl_vl_version}")
    if(_mm_rtl_vl_version VERSION_LESS 5.0)
        message(WARNING "MatmulKernel RTL: Verilator ${_mm_rtl_vl_version} found, the testbench needs 5.x "
                        "— TestMatmulRtl disabled")
        unset(MM_RTL_VERILATOR)
    endif()
endif()

if(MM_RTL_VERILATOR)
    message(STATUS "MatmulKernel RTL: Verilator ${_mm_rtl_vl_version} at ${MM_RTL_VERILATOR}")
    cmake_host_system_information(RESULT _mm_rtl_jobs QUERY NUMBER_OF_LOGICAL_CORES)
    set(MM_RTL_VFLAGS -Wall -Wno-DECLFILENAME "${MM_RTL_WAIVERS}" --top-module MatmulKernel)

    add_custom_target(lint_matmul_rtl
        COMMAND "${MM_RTL_VERILATOR}" --lint-only ${MM_RTL_VFLAGS} ${MM_RTL_SOURCES}
        COMMENT "MatmulKernel RTL: Verilator lint"
        VERBATIM)

    # mm_rtl_tb(<dir> <opt> <extra flags...>) — one Verilator build of the testbench
    function(mm_rtl_tb dir opt)
        set(_out "${CMAKE_CURRENT_BINARY_DIR}/${dir}/Vtb")
        add_custom_command(
            OUTPUT "${_out}"
            COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/${dir}"
            COMMAND "${MM_RTL_VERILATOR}" --cc --exe --build -j ${_mm_rtl_jobs} ${opt}
                    --x-assign unique --x-initial unique ${ARGN} ${MM_RTL_VFLAGS}
                    -Mdir "${CMAKE_CURRENT_BINARY_DIR}/${dir}" -o Vtb
                    "${MM_RTL_DIR}/tb/verilator/tb_main.cpp" ${MM_RTL_SOURCES}
            DEPENDS ${MM_RTL_SOURCES} ${MM_RTL_TB_SOURCES} "${MM_RTL_WAIVERS}"
            COMMENT "MatmulKernel RTL: Verilator testbench -> ${dir}/Vtb"
            VERBATIM)
    endfunction()
    mm_rtl_tb(vl -O3 -CFLAGS "-O2 -std=c++17")
    mm_rtl_tb(vlt -O2 --trace-fst -CFLAGS "-O2 -std=c++17")
    mm_rtl_tb(vcd -O1 --trace -CFLAGS "-O1 -std=c++17 -DVCD_TRACE")

    set(MM_RTL_TB "${CMAKE_CURRENT_BINARY_DIR}/vl/Vtb")
    add_custom_target(TestMatmulRtl ALL DEPENDS "${MM_RTL_TB}")
    add_custom_target(matmul_rtl_tb_fst DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/vlt/Vtb")
    add_custom_target(matmul_rtl_tb_vcd DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/vcd/Vtb")
    add_custom_target(perf_matmul_rtl
        COMMAND "${MM_RTL_TB}" --perf
        DEPENDS "${MM_RTL_TB}"
        COMMENT "MatmulKernel RTL: cycle counts (ideal memory)"
        USES_TERMINAL VERBATIM)

    add_test(NAME TestMatmulRtl
             COMMAND "${MM_RTL_TB}" --fixtures "${MM_RTL_FIXTURES}"
                     --random ${MM_RTL_RANDOM_CASES} --seed 1 --timing rand --quiet)
    message(STATUS "MatmulKernel RTL: targets TestMatmulRtl, lint_matmul_rtl, perf_matmul_rtl, "
                   "matmul_rtl_tb_fst, matmul_rtl_tb_vcd")
else()
    message(STATUS "MatmulKernel RTL: Verilator not found — TestMatmulRtl and lint_matmul_rtl disabled "
                   "(sudo apt install verilator)")
endif()

# ---------------------------------------------------------------------------
# Vivado: IP packaging, out-of-context synthesis, xsim
# ---------------------------------------------------------------------------
find_program(MM_RTL_VIVADO vivado
    HINTS $ENV{XILINX_VIVADO}/bin /mnt/data/xilinx/2025.2/Vivado/bin
    DOC "Vivado, for packaging the RTL MatmulKernel IP")

if(MM_RTL_VIVADO)
    get_filename_component(_mm_rtl_vivado_bin "${MM_RTL_VIVADO}" DIRECTORY)
    # Outside <build>/kernels on purpose: the HLS hardware build scans that
    # whole tree as its IP repository, and both IPs have the same VLNV.
    set(MM_RTL_IP_DIR "${CMAKE_BINARY_DIR}/rtl_ip")
    set(MM_RTL_IP "${MM_RTL_IP_DIR}/MatmulKernel_ip")
    # the commands run in these (make enters WORKING_DIRECTORY first)
    file(MAKE_DIRECTORY "${MM_RTL_IP_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/synth"
                        "${CMAKE_CURRENT_BINARY_DIR}/xsim" "${CMAKE_CURRENT_BINARY_DIR}/sysim")

    add_custom_command(
        OUTPUT "${MM_RTL_IP}/component.xml" "${MM_RTL_IP_DIR}/MatmulKernel_ip.zip"
        COMMAND "${MM_RTL_VIVADO}" -mode batch -nojournal -nolog
                -source "${MM_RTL_DIR}/syn/package_ip.tcl"
                -tclargs "${MM_RTL_DIR}" "${MM_RTL_DRIVER_ROOT}" "${MM_RTL_PART}"
        DEPENDS ${MM_RTL_SOURCES} "${MM_RTL_DIR}/syn/package_ip.tcl" "${MM_RTL_DIR}/syn/rtl_files.tcl"
                ${_mm_rtl_driver_files} matmul_rtl_driver
        WORKING_DIRECTORY "${MM_RTL_IP_DIR}"
        COMMENT "MatmulKernel RTL: Vivado IP -> ${MM_RTL_IP}"
        VERBATIM)
    add_custom_target(package_matmul_rtl DEPENDS "${MM_RTL_IP}/component.xml")

    add_custom_target(synth_matmul_rtl
        COMMAND "${MM_RTL_VIVADO}" -mode batch -nojournal -nolog
                -source "${MM_RTL_DIR}/syn/synth_ooc.tcl"
                -tclargs "${MM_RTL_DIR}" ${MM_RTL_PERIOD} "${MM_RTL_PART}"
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/synth"
        COMMENT "MatmulKernel RTL: Vivado OOC synthesis + P&R (${MM_RTL_PART}, ${MM_RTL_PERIOD} ns)"
        USES_TERMINAL VERBATIM)

    add_custom_target(xsim_matmul_rtl
        COMMAND "${_mm_rtl_vivado_bin}/xvlog" -sv ${MM_RTL_SOURCES}
        COMMAND "${_mm_rtl_vivado_bin}/xelab" -debug off MatmulKernel -s mm_elab
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/xsim"
        COMMENT "MatmulKernel RTL: xvlog / xelab"
        VERBATIM)

    # System level: a copy of the test stand's matmul_op_test project with
    # this IP in place of the HLS export (syn/sysim.tcl); ~4 min.
    set(_mm_rtl_stand "${MM_RTL_REPO}/hw/cormorant_test_stand/kernels/matmul_op_test")
    if(EXISTS "${_mm_rtl_stand}")
        set(_mm_rtl_sysim "${CMAKE_CURRENT_BINARY_DIR}/sysim")
        set(_mm_rtl_sysim_proj "${_mm_rtl_sysim}/matmul_op_test")
        add_custom_target(sysim_matmul_rtl
            COMMAND "${CMAKE_COMMAND}" -E rm -rf "${_mm_rtl_sysim_proj}"
            COMMAND "${CMAKE_COMMAND}" -E copy_directory "${_mm_rtl_stand}" "${_mm_rtl_sysim_proj}"
            COMMAND "${CMAKE_COMMAND}" -E rm -rf "${_mm_rtl_sysim_proj}/matmul_op_test.sim"
            COMMAND "${MM_RTL_VIVADO}" -mode batch -nojournal -log "${_mm_rtl_sysim}/sysim.log"
                    -source "${MM_RTL_DIR}/syn/sysim.tcl"
                    -tclargs "${_mm_rtl_sysim_proj}/matmul_op_test.xpr" "${MM_RTL_IP}"
                             "${MM_RTL_FIXTURES}" "${_mm_rtl_sysim}/sysim_report.json"
            COMMAND "${Python3_EXECUTABLE}" "${MM_RTL_REPO}/cmake/check_test_report.py"
                    "${_mm_rtl_sysim}/sysim_report.json"
            DEPENDS "${MM_RTL_IP}/component.xml"
            WORKING_DIRECTORY "${_mm_rtl_sysim}"
            COMMENT "MatmulKernel RTL: test-stand block design in xsim (log: ${_mm_rtl_sysim}/sysim.log)"
            USES_TERMINAL VERBATIM)
        set(_mm_rtl_sysim_msg ", sysim_matmul_rtl")
    endif()
    message(STATUS "MatmulKernel RTL: Vivado targets package_matmul_rtl, synth_matmul_rtl, "
                   "xsim_matmul_rtl${_mm_rtl_sysim_msg} (part ${MM_RTL_PART})")
else()
    message(STATUS "MatmulKernel RTL: vivado not found — package_matmul_rtl / synth_matmul_rtl disabled")
endif()
