cmake_minimum_required(VERSION 3.2)
project(iterate_all_executable_symbols LANGUAGES HIP)

set(CMAKE_HIP_STANDARD 20)

list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm)

find_package(hip)



list(APPEND CMAKE_MODULE_PATH ../../src/luthier)

add_library(iterate_all_executable_symbols SHARED iterate_all_executable_symbols.cpp)

set_source_files_properties(iterate_all_executable_symbols.cpp PROPERTIES LANGUAGE HIP)

set_property(TARGET iterate_all_executable_symbols PROPERTY COMPILE_FLAGS "-fPIC")

target_include_directories(iterate_all_executable_symbols PUBLIC ../../src/luthier/)

target_link_libraries(iterate_all_executable_symbols PUBLIC luthier hip::host hip::device)