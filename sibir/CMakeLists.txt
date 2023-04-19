cmake_minimum_required(VERSION 3.2)
project(sibir)

set(CMAKE_CXX_STANDARD 17)

# Search for rocm in common locations
list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm)
# Find hip
find_package(hip)
# Create the library
add_library(sibir STATIC sibir_callback.hpp sibir_callback.cpp sibir.h)

set_property(TARGET sibir
        PROPERTY COMPILE_FLAGS "-fPIC")
# Link with HIP
target_link_libraries(sibir hip::host hip::device)

target_include_directories(sibir PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})