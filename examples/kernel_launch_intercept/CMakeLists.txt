cmake_minimum_required(VERSION 3.2)
project(kernel_launch_intercept)

set(CMAKE_CXX_STANDARD 17)

list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm)
find_package(hip)



list(APPEND CMAKE_MODULE_PATH ../../src/sibir)

add_library(kernel_launch_intercept SHARED kernel_launch_intercept.cpp)

set_property(TARGET kernel_launch_intercept PROPERTY COMPILE_FLAGS "-fPIC")


target_link_libraries(kernel_launch_intercept PUBLIC sibir hip::host hip::device)