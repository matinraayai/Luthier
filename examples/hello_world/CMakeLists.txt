cmake_minimum_required(VERSION 3.2)
project(luthier_init_example)

set(CMAKE_CXX_STANDARD 17)

# Search for rocm in common locations
list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm)
# Find hip
find_package(hip)



list(APPEND CMAKE_MODULE_PATH ../../src/luthier)
# Create the library
add_library(luthier_init_example SHARED luthier_init_example.cpp)

set_property(TARGET luthier_init_example PROPERTY COMPILE_FLAGS "-fPIC")


target_link_libraries(luthier_init_example PUBLIC luthier hip::host hip::device)