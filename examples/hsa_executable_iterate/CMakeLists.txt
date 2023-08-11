cmake_minimum_required(VERSION 3.2)
project(luthier_init_example)

set(CMAKE_CXX_STANDARD 17)

# Search for rocm in common locations
list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm)

find_package(hip)



list(APPEND CMAKE_MODULE_PATH ../../src/luthier)
# Create the library
add_library(hsa_executable_iterate SHARED hsa_executable_iterate.cpp)

set_property(TARGET hsa_executable_iterate PROPERTY COMPILE_FLAGS "-fPIC")


target_link_libraries(hsa_executable_iterate PUBLIC luthier hip::host hip::device)