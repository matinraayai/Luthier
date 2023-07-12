cmake_minimum_required(VERSION 3.2)
project(sibir_init_example)

set(CMAKE_CXX_STANDARD 17)

# Search for rocm in common locations
list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm)
# Find hip
find_package(hip)



list(APPEND CMAKE_MODULE_PATH ../../src/sibir)
# Create the library
add_library(sibir_init_example SHARED sibir_init_example.cpp)

set_property(TARGET sibir_init_example PROPERTY COMPILE_FLAGS "-fPIC")


target_link_libraries(sibir_init_example PUBLIC sibir hip::host hip::device)