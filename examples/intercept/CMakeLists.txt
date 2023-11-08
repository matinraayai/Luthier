cmake_minimum_required(VERSION 3.2)
project(intercept LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm)

find_package(hip)


list(APPEND CMAKE_MODULE_PATH ../../src/luthier)

add_library(intercept_hsa SHARED intercept_hsa.cpp)

set_property(TARGET intercept_hsa PROPERTY COMPILE_FLAGS "-fPIC")

target_include_directories(intercept_hsa PUBLIC ../../src/luthier/)

target_link_libraries(intercept_hsa PUBLIC luthier hip::host)