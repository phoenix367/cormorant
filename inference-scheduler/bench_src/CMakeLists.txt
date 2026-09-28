cmake_minimum_required(VERSION 3.19)
project(kernel_benchmarks C)

# ── XRT ─────────────────────────────────────────────────────────────────────
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(XRT QUIET xrt)
endif()
if(XRT_FOUND)
    set(_XRT_INC  ${XRT_INCLUDE_DIRS})
    set(_XRT_LIBS ${XRT_LIBRARIES})
else()
    set(XRT_DIR "/opt/xilinx/xrt" CACHE PATH "XRT installation prefix")
    if(NOT EXISTS "${XRT_DIR}/include/xrt.h")
        message(WARNING "XRT header not found at ${XRT_DIR}/include/xrt.h")
    endif()
    set(_XRT_INC  "${XRT_DIR}/include")
    set(_XRT_LIBS "${XRT_DIR}/lib/libxrt_core.so")
endif()

# ── Shared buffer library ────────────────────────────────────────────────────
add_library(bench_infra STATIC inference_buf.c)
target_include_directories(bench_infra
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE ${_XRT_INC})
target_compile_options(bench_infra PRIVATE -Wall -O2)
target_link_libraries(bench_infra PUBLIC ${_XRT_LIBS})

# ── Helper: create one benchmark executable if its driver files exist ────────
function(add_bench_if name src drv_c drv_linux)
    set(drv_c_path     "${CMAKE_CURRENT_SOURCE_DIR}/driver/${drv_c}")
    set(drv_linux_path "${CMAKE_CURRENT_SOURCE_DIR}/driver/${drv_linux}")
    if(EXISTS "${drv_c_path}" AND EXISTS "${drv_linux_path}")
        add_executable(${name} ${src} ${drv_c_path} ${drv_linux_path})
        target_include_directories(${name}
            PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/driver")
        target_link_libraries(${name} PRIVATE bench_infra)
        target_compile_options(${name} PRIVATE -Wall -O2)
        message(STATUS "benchmark: ${name} enabled")
    else()
        message(STATUS "benchmark: ${name} SKIPPED (driver files missing)")
    endif()
endfunction()

add_bench_if(bench_vectorop bench_vectorop.c
             xvectoropkernel.c xvectoropkernel_linux.c)
add_bench_if(bench_matmul   bench_matmul.c
             xmatmulkernel.c   xmatmulkernel_linux.c)
add_bench_if(bench_conv     bench_conv.c
             xconvkernel.c     xconvkernel_linux.c)
add_bench_if(bench_pool     bench_pool.c
             xpoolingkernel.c  xpoolingkernel_linux.c)

# ── Calibration runner: every kernel in one process (perf_calibrate.py) ─────
set(_CALIB_DRV "")
foreach(_d xvectoropkernel xmatmulkernel xconvkernel xpoolingkernel)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/driver/${_d}.c"
       AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/driver/${_d}_linux.c")
        list(APPEND _CALIB_DRV "${CMAKE_CURRENT_SOURCE_DIR}/driver/${_d}.c"
                               "${CMAKE_CURRENT_SOURCE_DIR}/driver/${_d}_linux.c")
    endif()
endforeach()
list(LENGTH _CALIB_DRV _n_calib_drv)
if(_n_calib_drv EQUAL 8)
    add_executable(calib_runner calib_runner.c ${_CALIB_DRV})
    target_include_directories(calib_runner PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/driver")
    target_link_libraries(calib_runner PRIVATE bench_infra m)
    target_compile_options(calib_runner PRIVATE -Wall -O2)
    message(STATUS "benchmark: calib_runner enabled")
else()
    message(STATUS "benchmark: calib_runner SKIPPED (needs all four kernels' drivers)")
endif()
