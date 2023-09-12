cmake_minimum_required(VERSION 3.2)
project(hip_intercept LANGUAGES HIP)

set(CMAKE_HIP_STANDARD 20)

list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm)

find_package(hip)



list(APPEND CMAKE_MODULE_PATH ../../src/luthier)

add_library(hip_intercept SHARED hip_intercept.cpp)

set_source_files_properties(hip_intercept.cpp PROPERTIES LANGUAGE HIP)

set_property(TARGET hip_intercept PROPERTY COMPILE_FLAGS "-fPIC")

target_include_directories(hip_intercept PUBLIC ../../src/luthier/)

target_link_libraries(hip_intercept PUBLIC luthier hip::host hip::device)