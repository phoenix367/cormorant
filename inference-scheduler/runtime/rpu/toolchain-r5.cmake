# Copyright 2026 Ivan Gubochkin
# SPDX-License-Identifier: Apache-2.0

# toolchain-r5.cmake — cross-compile the inference firmware for the Cortex-R5
# with the Vitis GNU toolchain (RPU_PLAN track B1).
#
#   cmake -S rpu -B build-rpu -DCMAKE_TOOLCHAIN_FILE=rpu/toolchain-r5.cmake \
#         -DBSP_DIR=<platform>/psu_cortexr5_0/<domain>/bsp
#
# VITIS_DIR defaults to the Vitis installation this repository documents.
set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR arm)
if(NOT DEFINED VITIS_DIR)
    if(DEFINED ENV{XILINX_VITIS})
        set(VITIS_DIR $ENV{XILINX_VITIS})
    else()
        set(VITIS_DIR /mnt/data/xilinx/2025.2/Vitis)
    endif()
endif()
set(_R5_BIN ${VITIS_DIR}/gnu/armr5/lin/gcc-arm-none-eabi/bin)
set(CMAKE_C_COMPILER   ${_R5_BIN}/armr5-none-eabi-gcc)
set(CMAKE_CXX_COMPILER ${_R5_BIN}/armr5-none-eabi-g++)
set(CMAKE_ASM_COMPILER ${_R5_BIN}/armr5-none-eabi-gcc)
set(CMAKE_AR           ${_R5_BIN}/armr5-none-eabi-ar)
set(CMAKE_OBJCOPY      ${_R5_BIN}/armr5-none-eabi-objcopy)
set(CMAKE_SIZE         ${_R5_BIN}/armr5-none-eabi-size)
set(CMAKE_C_FLAGS_INIT   "-mcpu=cortex-r5 -mfloat-abi=hard -mfpu=vfpv3-d16 -DARMR5 -DSDT")
set(CMAKE_ASM_FLAGS_INIT "-mcpu=cortex-r5 -mfloat-abi=hard -mfpu=vfpv3-d16 -DARMR5 -DSDT")
set(CMAKE_EXE_LINKER_FLAGS_INIT "-mcpu=cortex-r5 -mfloat-abi=hard -mfpu=vfpv3-d16")
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
