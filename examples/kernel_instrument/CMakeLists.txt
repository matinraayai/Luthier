cmake_minimum_required(VERSION 3.2)
project(kernel_instrument LANGUAGES HIP)

set(CMAKE_HIP_STANDARD 20)

list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm)

find_package(hip)



list(APPEND CMAKE_MODULE_PATH ../../src/sibir)

add_library(kernel_instrument SHARED kernel_instrument.cpp)

set_source_files_properties(kernel_instrument.cpp PROPERTIES LANGUAGE HIP)

set_property(TARGET kernel_instrument PROPERTY COMPILE_FLAGS "-fPIC")

target_include_directories(kernel_instrument PUBLIC ../../src/sibir/)

target_link_libraries(kernel_instrument PUBLIC sibir hip::host hip::device)