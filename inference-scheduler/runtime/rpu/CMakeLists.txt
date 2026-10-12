# runtime/rpu/CMakeLists.txt — the kernel-driver firmware for the Cortex-R5
# (RPU_PLAN track B1): rpu_main.c (the command-stream loop behind the control
# block of rpu_ctrl.h) + rpu_irq.c (the GIC, the kernels' interrupts, the
# waits), linked for the TCMs (lscript.ld).  It knows no model: one build per
# bitstream (rpu_board.py builds it).
#
#   cmake -S runtime/rpu -B build-rpu -DCMAKE_TOOLCHAIN_FILE=runtime/rpu/toolchain-r5.cmake \
#         -DBSP_DIR=<platform>/psu_cortexr5_0/<domain>/bsp
#   cmake --build build-rpu            -> build-rpu/rpu_firmware.elf (remoteproc loads it)
#
# BSP_DIR is a standalone domain for psu_cortexr5_0 built by Vitis from this
# bitstream's XSA (tools/rpu/make_r5_bsp.py): its xparameters.h names the
# kernels, their bases and their PL interrupt lines.
cmake_minimum_required(VERSION 3.16)
project(rpu_firmware C ASM)

if(NOT BSP_DIR)
    message(FATAL_ERROR "set -DBSP_DIR=<platform>/psu_cortexr5_0/<domain>/bsp (the R5 standalone BSP)")
endif()
add_executable(rpu_firmware.elf rpu_main.c rpu_irq.c)
target_include_directories(rpu_firmware.elf PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${BSP_DIR}/include)
target_compile_options(rpu_firmware.elf PRIVATE -Wall -Wextra -O2 -ffunction-sections -fdata-sections
    -Wno-unused-parameter)
target_link_options(rpu_firmware.elf PRIVATE
    -Wl,-T,${CMAKE_CURRENT_SOURCE_DIR}/lscript.ld
    -Wl,--gc-sections
    -Wl,-Map,${CMAKE_CURRENT_BINARY_DIR}/rpu_firmware.map
    -specs=${BSP_DIR}/Xilinx.spec
    -L${BSP_DIR}/lib
)
target_link_libraries(rpu_firmware.elf PRIVATE
    -Wl,--start-group -lxil -lxilstandalone -lxiltimer -lgcc -lc -Wl,--end-group)
set_target_properties(rpu_firmware.elf PROPERTIES LINK_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/lscript.ld)
add_custom_command(TARGET rpu_firmware.elf POST_BUILD
    COMMAND ${CMAKE_SIZE} $<TARGET_FILE:rpu_firmware.elf>)
