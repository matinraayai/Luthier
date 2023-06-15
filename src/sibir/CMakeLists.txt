cmake_minimum_required(VERSION 3.2)
project(sibir)

set(CMAKE_CXX_STANDARD 20)

# Search for rocm in common locations
list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm)

#list(APPEND CMAKE_MODULE_PATH ../disassembler)
# Find hip
find_package(hip)
# Create the library
add_library(sibir SHARED sibir_impl.hpp sibir_impl.cpp sibir.h hip_intercept.h hip_intercept.cpp
        sibir_types.h hsa_intercept.h hsa_intercept.cpp)

# Some HSA include files won't work unless this is defined
target_compile_definitions(sibir PUBLIC AMD_INTERNAL_BUILD)
set_property(TARGET sibir PROPERTY COMPILE_FLAGS "-fPIC")

# Link with HIP and ROCTracer
target_link_libraries(sibir hip::host hip::device roctracer64 stdc++fs rocm-dbgapi)

target_include_directories(sibir PUBLIC .)