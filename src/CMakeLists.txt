cmake_minimum_required(VERSION 3.2)
project(sibir)

set(CMAKE_CXX_STANDARD 20)

# Search for rocm in common locations
list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm)

# External Dependencies

find_package(hip)

find_package(amd_comgr)

include(FetchContent)

FetchContent_Declare(
        elfio
        DOWNLOAD_EXTRACT_TIMESTAMP 0
        URL "https://github.com/serge1/ELFIO/archive/refs/heads/main.zip"
        SOURCE_DIR ${CMAKE_SOURCE_DIR}/ext/elfio/
        BUILD_COMMAND ""
        CONFIGURE_COMMAND ""
)

FetchContent_MakeAvailable(elfio)

# Create the library
add_library(sibir SHARED sibir_impl.hpp sibir_impl.cpp sibir.h hip_intercept.hpp hip_intercept.cpp sibir_types.h
        hsa_intercept.h hsa_intercept.cpp code_object_manager.hpp code_object_manager.cpp hip_arg_types.h
        hip_private_api.h error_check.h disassembler.hpp disassembler.cpp amdgpu_elf.hpp amdgpu_elf.cpp
        context_manager.hpp context_manager.cpp code_generator.hpp code_generator.cpp instr.hpp Instr.cpp)


target_include_directories(sibir PRIVATE ${elfio_SOURCE_DIR})


# Some HSA include files won't work unless this is defined
target_compile_definitions(sibir PUBLIC AMD_INTERNAL_BUILD)

set_property(TARGET sibir PROPERTY COMPILE_FLAGS "-fPIC")

# Link with HIP and ROCTracer
target_link_libraries(sibir hip::host hip::device roctracer64 stdc++fs amd_comgr)

target_include_directories(sibir PUBLIC .)