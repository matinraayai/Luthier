cmake_minimum_required(VERSION 3.2)
project(sibir_callback)

set(CMAKE_CXX_STANDARD 17)

# Search for rocm in common locations
list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm)
# Find hip
find_package(hip)
# Create the library
add_library(sibir_callback SHARED sibir_callback.hpp sibir_callback.cpp)
# Link with HIP
target_link_libraries(sibir_callback hip::host hip::device)