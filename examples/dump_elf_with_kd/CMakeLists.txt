cmake_minimum_required(VERSION 3.2)
project(dump_elf_with_kd LANGUAGES HIP)

set(CMAKE_CXX_STANDARD 20)

list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm)
find_package(hip)



list(APPEND CMAKE_MODULE_PATH ../../src/luthier)

add_library(dump_elf SHARED dump_elf.cpp)

set_source_files_properties(dump_elf.cpp PROPERTIES LANGUAGE HIP)

set_property(TARGET dump_elf PROPERTY COMPILE_FLAGS "-fPIC")

target_include_directories(dump_elf PUBLIC ../../src/luthier/)

target_link_libraries(dump_elf PUBLIC luthier hip::host hip::device)