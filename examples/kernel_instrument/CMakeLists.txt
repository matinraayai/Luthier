cmake_minimum_required(VERSION 3.2)
project(kernel_instrument)

set(CMAKE_CXX_STANDARD 20)

list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm)
find_package(hip)



list(APPEND CMAKE_MODULE_PATH ../../src/sibir)

add_library(kernel_instrument SHARED kernel_instrument.cpp)

set_property(TARGET kernel_instrument PROPERTY COMPILE_FLAGS "-fPIC")


target_link_libraries(kernel_instrument PUBLIC sibir hip::host hip::device amd_comgr)