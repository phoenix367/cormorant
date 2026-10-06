cmake_minimum_required(VERSION 3.19)
project(ConvKernelRtl VERSION 1.0 LANGUAGES C CXX)

# The platform JSONs (AXI_PLATFORM_FILES, AXI_DEFAULT_PLATFORM_JSON): set by the
# top-level project, or here when this kernel is configured on its own.
list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../cmake")
include(AxiPlatform)
get_filename_component(CV_RTL_REPO "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
enable_testing()

# ---------------------------------------------------------------------------
# ConvKernel in SystemVerilog — a drop-in for the Vitis HLS ConvKernel
# (kernels/conv, whose C++ is the reference model): same VLNV, register map,
# m_axi bus parameters, DDR layouts and arithmetic.  See
# doc/plans/CONV_RTL_PLAN.md.
#
# Targets (each one only when its tool is found):
#   TestConvRtl            Verilator testbench (vl/Vtb); ctest runs the
#                          fixtures + CV_RTL_RANDOM_CASES random jobs checked
#                          against the HLS kernel's C++ (with the Vitis HLS headers)
#   lint_conv_rtl          verilator --lint-only -Wall
#   perf_conv_rtl          cycle counts of the board's ConvKernel benchmarks (ideal memory)
#   ConvRtlDsp             (ctest) cv_mac_chain's behavioural model against the
#                          DSP48E2 unisim model (with Vivado's data directory)
#   conv_rtl_tb_fst        testbench with FST tracing (vlt/Vtb --trace F.fst)
#   driver_conv_rtl        the C driver (scripts/gen_driver.py) -> driver/
#   package_conv_rtl       Vivado IP catalog -> <build>/rtl_ip/ConvKernel_ip (+ .zip)
#   synth_conv_rtl         Vivado out-of-context synthesis + P&R -> synth/
#   xsim_conv_rtl          xvlog / xelab parse and elaboration
#   sysim_conv_rtl         the test stand's ConvKernel block design with
#                          this IP in xsim, on a copy
# ---------------------------------------------------------------------------

set(CV_RTL_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(CV_RTL_SOURCES
    rtl/cv_pkg.sv rtl/cv_fifo.sv rtl/cv_lutram.sv rtl/cv_div.sv rtl/cv_ctrl_s_axi.sv
    rtl/cv_mac_chain.sv rtl/cv_engine.sv rtl/cv_xload.sv rtl/cv_patch.sv rtl/cv_wload.sv
    rtl/cv_bias.sv rtl/cv_drain.sv rtl/cv_ywriter.sv rtl/cv_core.sv rtl/ConvKernel.v)
list(TRANSFORM CV_RTL_SOURCES PREPEND "${CV_RTL_DIR}/")
set(CV_RTL_TB_SOURCES
    "${CV_RTL_DIR}/tb/verilator/tb_main.cpp"
    "${CV_RTL_DIR}/tb/verilator/axi_models.h")
set(CV_RTL_WAIVERS "${CV_RTL_DIR}/scripts/lint_waivers.vlt")
set(CV_RTL_FIXTURES "${CV_RTL_REPO}/hw/test_data/conv_test_data"
    CACHE PATH "ConvKernel fixtures (manifest.txt + test_NN_{x,w,b,y}.hex) for TestConvRtl")
set(CV_RTL_RANDOM_CASES 200 CACHE STRING "TestConvRtl: random jobs after the fixtures")
set(CV_RTL_PERIOD 3.333 CACHE STRING "synth_conv_rtl: clock period in ns (OOC timing check)")

find_package(Python3 COMPONENTS Interpreter REQUIRED)

file(READ "${AXI_DEFAULT_PLATFORM_JSON}" _cv_rtl_pjson)
string(JSON CV_RTL_PART GET "${_cv_rtl_pjson}" part)

# cv_pkg.sv holds the platform's kernels.conv bounds as constants (the HLS
# kernel's Config.h from the same file): a platform with other bounds needs
# them changed there.
file(READ "${CV_RTL_DIR}/rtl/cv_pkg.sv" _cv_rtl_pkg)
foreach(_pair "tile_m;TM" "tile_ic;TIC" "max_kh;MAXK" "max_kw;MAXK" "max_line_buf_rows;LBR"
              "max_line_buf_cols;LBC" "max_acc_persist_entries;ACCN" "max_m_per_group;MPG"
              "max_in_ch;MAXIC" "max_out_ch;MAXOC")
    list(GET _pair 0 _key)
    list(GET _pair 1 _const)
    string(JSON _json_v GET "${_cv_rtl_pjson}" kernels conv ${_key})
    string(REGEX MATCH "localparam int ${_const} +=  *([0-9]+)" _m "${_cv_rtl_pkg}")
    if(NOT CMAKE_MATCH_1 STREQUAL _json_v)
        message(FATAL_ERROR "ConvKernel RTL: kernels.conv.${_key} = ${_json_v} in "
                            "${AXI_DEFAULT_PLATFORM_JSON}, but rtl/cv_pkg.sv has ${_const} = ${CMAKE_MATCH_1}")
    endif()
endforeach()

# ---------------------------------------------------------------------------
# The C driver: the API and register map of the HLS-generated xconvkernel,
# written by scripts/gen_driver.py (the RTL MatmulKernel's generator); the
# register table is checked against rtl/cv_ctrl_s_axi.sv by ctest
# (ConvRtlDriver) and compiled here.
# ---------------------------------------------------------------------------
set(CV_RTL_DRIVER_ROOT "${CMAKE_CURRENT_BINARY_DIR}/driver")
set(CV_RTL_DRIVER_SRC "${CV_RTL_DRIVER_ROOT}/ConvKernel_v1_0/src")
set(_cv_rtl_gen "${CV_RTL_DIR}/scripts/gen_driver.py" "${CV_RTL_REPO}/kernels/matmul_rtl/scripts/gen_driver.py")
set(_cv_rtl_driver_files
    "${CV_RTL_DRIVER_SRC}/xconvkernel.c" "${CV_RTL_DRIVER_SRC}/xconvkernel_linux.c"
    "${CV_RTL_DRIVER_SRC}/xconvkernel_sinit.c" "${CV_RTL_DRIVER_SRC}/xconvkernel.h"
    "${CV_RTL_DRIVER_SRC}/xconvkernel_hw.h")
add_custom_command(
    OUTPUT ${_cv_rtl_driver_files}
    COMMAND "${Python3_EXECUTABLE}" "${CV_RTL_DIR}/scripts/gen_driver.py" --out "${CV_RTL_DRIVER_ROOT}"
    DEPENDS ${_cv_rtl_gen}
    COMMENT "ConvKernel RTL: C driver -> ${CV_RTL_DRIVER_ROOT}"
    VERBATIM)
add_library(conv_rtl_driver OBJECT
    "${CV_RTL_DRIVER_SRC}/xconvkernel.c" "${CV_RTL_DRIVER_SRC}/xconvkernel_linux.c")
target_compile_options(conv_rtl_driver PRIVATE -Wall -Wextra -Werror)
add_custom_target(driver_conv_rtl DEPENDS conv_rtl_driver)
add_test(NAME ConvRtlDriver
         COMMAND "${Python3_EXECUTABLE}" "${CV_RTL_DIR}/scripts/gen_driver.py" --check)

# ---------------------------------------------------------------------------
# The HLS oracle: kernels/conv/kernel/ConvKernel.cpp built with the Vitis
# HLS headers into the testbench (ap_fixed<16,8> / <32,16> and the platform's
# bounds, as the bitstream).
# ---------------------------------------------------------------------------
find_path(CV_RTL_HLS_INCLUDE NAMES ap_fixed.h hls_burst_maxi.h
    HINTS $ENV{XILINX_HLS}/include $ENV{XILINX_VITIS}/include /mnt/data/xilinx/2025.2/Vitis/include
    NO_DEFAULT_PATH
    DOC "Vitis HLS include directory (ap_fixed.h): the testbench's HLS oracle")
set(_cv_rtl_oracle_flags)
set(_cv_rtl_oracle_src)
if(CV_RTL_HLS_INCLUDE)
    set(_cv_rtl_cfg "${CMAKE_CURRENT_BINARY_DIR}/oracle")
    set(CONV_DATA_TYPE_INCLUDE   "#include \"ap_fixed.h\"")
    set(CONV_HAVE_APFIXED_DEFINE "#define CONV_HAVE_APFIXED")
    set(CONV_DATA_TYPE     "ap_fixed<16,8>")
    set(CONV_ACC_DATA_TYPE "ap_fixed<32,16>")
    if(NOT DEFINED AXI_BUS_WIDTH)
        set(AXI_BUS_WIDTH 128)
    endif()
    foreach(_k tile_m tile_ic max_kh max_kw max_in_ch max_out_ch max_line_buf_cols
               max_line_buf_rows max_acc_persist_entries max_m_per_group)
        string(TOUPPER "CONV_${_k}" _var)
        string(JSON ${_var} GET "${_cv_rtl_pjson}" kernels conv ${_k})
    endforeach()
    configure_file("${CV_RTL_REPO}/kernels/conv/include/Config.h.in" "${_cv_rtl_cfg}/Config.h")
    set(_cv_rtl_oracle_src "${CV_RTL_REPO}/kernels/conv/kernel/ConvKernel.cpp")
    set(_cv_rtl_oracle_flags -CFLAGS -DCV_HLS_ORACLE -CFLAGS -I${_cv_rtl_cfg}
        -CFLAGS -I${CV_RTL_REPO}/kernels/conv/include -CFLAGS -I${CV_RTL_HLS_INCLUDE}
        -CFLAGS -Wno-unused -CFLAGS -Wno-unknown-pragmas -CFLAGS -Wno-unused-parameter
        -CFLAGS -Wno-sign-compare)
    message(STATUS "ConvKernel RTL: testbench oracle = the HLS kernel's C++ (${CV_RTL_HLS_INCLUDE})")
    set(_cv_rtl_ctest_random --random ${CV_RTL_RANDOM_CASES})
else()
    message(STATUS "ConvKernel RTL: Vitis HLS headers not found — the testbench checks the fixtures only")
    set(_cv_rtl_ctest_random)
endif()

# ---------------------------------------------------------------------------
# Verilator: lint and the C++ testbench
# ---------------------------------------------------------------------------
find_program(CV_RTL_VERILATOR verilator DOC "Verilator (5.x) for the RTL ConvKernel testbench")
if(CV_RTL_VERILATOR)
    execute_process(COMMAND "${CV_RTL_VERILATOR}" --version
                    OUTPUT_VARIABLE _cv_rtl_vl_version OUTPUT_STRIP_TRAILING_WHITESPACE)
    string(REGEX MATCH "[0-9]+\\.[0-9]+" _cv_rtl_vl_version "${_cv_rtl_vl_version}")
    if(_cv_rtl_vl_version VERSION_LESS 5.0)
        message(WARNING "ConvKernel RTL: Verilator ${_cv_rtl_vl_version} found, the testbench needs 5.x "
                        "— TestConvRtl disabled")
        unset(CV_RTL_VERILATOR)
    endif()
endif()

if(CV_RTL_VERILATOR)
    message(STATUS "ConvKernel RTL: Verilator ${_cv_rtl_vl_version} at ${CV_RTL_VERILATOR}")
    cmake_host_system_information(RESULT _cv_rtl_jobs QUERY NUMBER_OF_LOGICAL_CORES)
    set(CV_RTL_VFLAGS -Wall -Wno-DECLFILENAME "${CV_RTL_WAIVERS}" --top-module ConvKernel)

    add_custom_target(lint_conv_rtl
        COMMAND "${CV_RTL_VERILATOR}" --lint-only ${CV_RTL_VFLAGS} ${CV_RTL_SOURCES}
        COMMENT "ConvKernel RTL: Verilator lint"
        VERBATIM)

    # cv_rtl_tb(<dir> <opt> <extra flags...>) — one Verilator build of the testbench
    function(cv_rtl_tb dir opt)
        set(_out "${CMAKE_CURRENT_BINARY_DIR}/${dir}/Vtb")
        add_custom_command(
            OUTPUT "${_out}"
            COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/${dir}"
            COMMAND "${CV_RTL_VERILATOR}" --cc --exe --build -j ${_cv_rtl_jobs} ${opt}
                    --x-assign unique --x-initial unique ${ARGN} ${_cv_rtl_oracle_flags} ${CV_RTL_VFLAGS}
                    -Mdir "${CMAKE_CURRENT_BINARY_DIR}/${dir}" -o Vtb
                    "${CV_RTL_DIR}/tb/verilator/tb_main.cpp" ${_cv_rtl_oracle_src} ${CV_RTL_SOURCES}
            DEPENDS ${CV_RTL_SOURCES} ${CV_RTL_TB_SOURCES} "${CV_RTL_WAIVERS}" ${_cv_rtl_oracle_src}
                    "${CV_RTL_REPO}/kernels/conv/include/ConvKernel.h"
                    "${CV_RTL_REPO}/kernels/conv/include/ConvMacGrid.h"
            COMMENT "ConvKernel RTL: Verilator testbench -> ${dir}/Vtb"
            VERBATIM)
    endfunction()
    cv_rtl_tb(vl -O3 -CFLAGS -O2 -CFLAGS -std=c++17)
    cv_rtl_tb(vlt -O2 --trace-fst -CFLAGS -O2 -CFLAGS -std=c++17)

    set(CV_RTL_TB "${CMAKE_CURRENT_BINARY_DIR}/vl/Vtb")
    add_custom_target(TestConvRtl ALL DEPENDS "${CV_RTL_TB}")
    add_custom_target(conv_rtl_tb_fst DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/vlt/Vtb")
    add_custom_target(perf_conv_rtl
        COMMAND "${CV_RTL_TB}" --perf
        DEPENDS "${CV_RTL_TB}"
        COMMENT "ConvKernel RTL: cycle counts of the board benchmarks (ideal memory)"
        USES_TERMINAL VERBATIM)

    add_test(NAME TestConvRtl
             COMMAND "${CV_RTL_TB}" --fixtures "${CV_RTL_FIXTURES}"
                     ${_cv_rtl_ctest_random} --seed 1 --timing rand --quiet)
    # cv_mac_chain's behavioural model (what the testbench simulates) against
    # Vivado's DSP48E2 model (what the hardware is): both chains on the same
    # random stimulus, compared every cycle (tb/verilator/dsp_check.*).
    find_file(CV_RTL_UNISIM_DSP DSP48E2.v
        HINTS $ENV{XILINX_VIVADO}/data/verilog/src/unisims /mnt/data/xilinx/2025.2/Vivado/data/verilog/src/unisims
        NO_DEFAULT_PATH DOC "Vivado's DSP48E2 unisim model, for the ConvRtlDsp check")
    if(CV_RTL_UNISIM_DSP)
        get_filename_component(_cv_rtl_glbl "${CV_RTL_UNISIM_DSP}/../../glbl.v" ABSOLUTE)
        set(_cv_rtl_dsp "${CMAKE_CURRENT_BINARY_DIR}/dsp")
        add_custom_command(
            OUTPUT "${_cv_rtl_dsp}/Vdsp"
            COMMAND "${CMAKE_COMMAND}" -E make_directory "${_cv_rtl_dsp}"
            COMMAND "${Python3_EXECUTABLE}" "${CV_RTL_DIR}/scripts/vl_unisim.py"
                    "${CV_RTL_UNISIM_DSP}" "${_cv_rtl_dsp}/DSP48E2_vl.v"
            COMMAND "${CV_RTL_VERILATOR}" --cc --exe --build -j ${_cv_rtl_jobs} -O2 --no-timing
                    -Wno-fatal -Wno-lint -Wno-style -DCV_UNISIM_DSP --top-module dsp_check
                    -Mdir "${_cv_rtl_dsp}" -o Vdsp "${CV_RTL_DIR}/tb/verilator/dsp_check.cpp"
                    "${_cv_rtl_glbl}" "${_cv_rtl_dsp}/DSP48E2_vl.v" "${CV_RTL_DIR}/rtl/cv_pkg.sv"
                    "${CV_RTL_DIR}/rtl/cv_mac_chain.sv" "${CV_RTL_DIR}/tb/verilator/dsp_check.sv"
            DEPENDS "${CV_RTL_DIR}/rtl/cv_mac_chain.sv" "${CV_RTL_DIR}/rtl/cv_pkg.sv"
                    "${CV_RTL_DIR}/tb/verilator/dsp_check.sv" "${CV_RTL_DIR}/tb/verilator/dsp_check.cpp"
                    "${CV_RTL_DIR}/scripts/vl_unisim.py"
            COMMENT "ConvKernel RTL: cv_mac_chain vs the DSP48E2 model -> dsp/Vdsp"
            VERBATIM)
        add_custom_target(conv_rtl_dsp_check ALL DEPENDS "${_cv_rtl_dsp}/Vdsp")
        add_test(NAME ConvRtlDsp COMMAND "${_cv_rtl_dsp}/Vdsp")
        set(_cv_rtl_dsp_msg ", ConvRtlDsp")
    endif()
    message(STATUS "ConvKernel RTL: targets TestConvRtl, lint_conv_rtl, perf_conv_rtl, "
                   "conv_rtl_tb_fst${_cv_rtl_dsp_msg}")
else()
    message(STATUS "ConvKernel RTL: Verilator not found — TestConvRtl and lint_conv_rtl disabled "
                   "(sudo apt install verilator)")
endif()

# ---------------------------------------------------------------------------
# Vivado: IP packaging, out-of-context synthesis, xsim
# ---------------------------------------------------------------------------
find_program(CV_RTL_VIVADO vivado
    HINTS $ENV{XILINX_VIVADO}/bin /mnt/data/xilinx/2025.2/Vivado/bin
    DOC "Vivado, for packaging the RTL ConvKernel IP")

if(CV_RTL_VIVADO)
    get_filename_component(_cv_rtl_vivado_bin "${CV_RTL_VIVADO}" DIRECTORY)
    # Outside <build>/kernels on purpose: a build that scans that tree as its
    # IP repository must not see two IPs with the same VLNV.
    set(CV_RTL_IP_DIR "${CMAKE_BINARY_DIR}/rtl_ip")
    set(CV_RTL_IP "${CV_RTL_IP_DIR}/ConvKernel_ip")
    set(_cv_rtl_pkg "${CMAKE_CURRENT_BINARY_DIR}/package")
    file(MAKE_DIRECTORY "${CV_RTL_IP_DIR}" "${_cv_rtl_pkg}" "${CMAKE_CURRENT_BINARY_DIR}/synth"
                        "${CMAKE_CURRENT_BINARY_DIR}/xsim" "${CMAKE_CURRENT_BINARY_DIR}/sysim")

    # package_ip.tcl works in its current directory (./ConvKernel_ip, ./proj):
    # a directory of its own, then the IP is moved next to the other RTL IPs.
    add_custom_command(
        OUTPUT "${CV_RTL_IP}/component.xml" "${CV_RTL_IP_DIR}/ConvKernel_ip.zip"
        COMMAND "${CV_RTL_VIVADO}" -mode batch -nojournal -nolog
                -source "${CV_RTL_DIR}/syn/package_ip.tcl"
                -tclargs "${CV_RTL_DIR}" "${CV_RTL_DRIVER_ROOT}" "${CV_RTL_PART}"
        COMMAND "${CMAKE_COMMAND}" -E rm -rf "${CV_RTL_IP}" "${CV_RTL_IP_DIR}/ConvKernel_ip.zip"
        COMMAND "${CMAKE_COMMAND}" -E rename "${_cv_rtl_pkg}/ConvKernel_ip" "${CV_RTL_IP}"
        COMMAND "${CMAKE_COMMAND}" -E rename "${_cv_rtl_pkg}/ConvKernel_ip.zip"
                "${CV_RTL_IP_DIR}/ConvKernel_ip.zip"
        DEPENDS ${CV_RTL_SOURCES} "${CV_RTL_DIR}/syn/package_ip.tcl" "${CV_RTL_DIR}/syn/rtl_files.tcl"
                ${_cv_rtl_driver_files} conv_rtl_driver
        WORKING_DIRECTORY "${_cv_rtl_pkg}"
        COMMENT "ConvKernel RTL: Vivado IP -> ${CV_RTL_IP}"
        VERBATIM)
    add_custom_target(package_conv_rtl DEPENDS "${CV_RTL_IP}/component.xml")

    add_custom_target(synth_conv_rtl
        COMMAND "${CV_RTL_VIVADO}" -mode batch -nojournal -nolog
                -source "${CV_RTL_DIR}/syn/synth_ooc.tcl"
                -tclargs "${CV_RTL_DIR}" ${CV_RTL_PERIOD} "${CV_RTL_PART}"
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/synth"
        COMMENT "ConvKernel RTL: Vivado OOC synthesis + P&R (${CV_RTL_PART}, ${CV_RTL_PERIOD} ns)"
        USES_TERMINAL VERBATIM)

    add_custom_target(xsim_conv_rtl
        COMMAND "${_cv_rtl_vivado_bin}/xvlog" -sv ${CV_RTL_SOURCES}
        COMMAND "${_cv_rtl_vivado_bin}/xelab" -debug off ConvKernel -s cv_elab
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/xsim"
        COMMENT "ConvKernel RTL: xvlog / xelab"
        VERBATIM)

    # System level: a copy of the test stand's conv_test project with this
    # IP in place of the HLS export (syn/sysim.tcl).
    set(_cv_rtl_stand "${CV_RTL_REPO}/hw/cormorant_test_stand/kernels/conv_test")
    if(EXISTS "${_cv_rtl_stand}")
        set(_cv_rtl_sysim "${CMAKE_CURRENT_BINARY_DIR}/sysim")
        set(_cv_rtl_sysim_proj "${_cv_rtl_sysim}/conv_test")
        add_custom_target(sysim_conv_rtl
            COMMAND "${CMAKE_COMMAND}" -E rm -rf "${_cv_rtl_sysim_proj}"
            COMMAND "${CMAKE_COMMAND}" -E copy_directory "${_cv_rtl_stand}" "${_cv_rtl_sysim_proj}"
            COMMAND "${CMAKE_COMMAND}" -E rm -rf "${_cv_rtl_sysim_proj}/conv_test.sim"
            COMMAND "${CV_RTL_VIVADO}" -mode batch -nojournal -log "${_cv_rtl_sysim}/sysim.log"
                    -source "${CV_RTL_DIR}/syn/sysim.tcl"
                    -tclargs "${_cv_rtl_sysim_proj}/conv_test.xpr" "${CV_RTL_IP}"
                             "${CV_RTL_FIXTURES}" "${_cv_rtl_sysim}/sysim_report.json"
            COMMAND "${Python3_EXECUTABLE}" "${CV_RTL_REPO}/cmake/check_test_report.py"
                    "${_cv_rtl_sysim}/sysim_report.json"
            DEPENDS "${CV_RTL_IP}/component.xml"
            WORKING_DIRECTORY "${_cv_rtl_sysim}"
            COMMENT "ConvKernel RTL: test-stand block design in xsim (log: ${_cv_rtl_sysim}/sysim.log)"
            USES_TERMINAL VERBATIM)
        set(_cv_rtl_sysim_msg ", sysim_conv_rtl")
    endif()
    message(STATUS "ConvKernel RTL: Vivado targets package_conv_rtl, synth_conv_rtl, "
                   "xsim_conv_rtl${_cv_rtl_sysim_msg} (part ${CV_RTL_PART})")
else()
    message(STATUS "ConvKernel RTL: vivado not found — package_conv_rtl / synth_conv_rtl disabled")
endif()
