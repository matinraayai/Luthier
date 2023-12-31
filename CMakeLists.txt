cmake_minimum_required(VERSION 3.21)

# Global CXX Setup =====================================================================================================
project(luthier VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

# Put the built libraries for all CMake projects under the lib/ folder
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# ======================================================================================================================

# Library Declaration ==================================================================================================
add_library(luthier SHARED
        src/luthier.cpp include/luthier.h src/hip_intercept.hpp src/hip_intercept.cpp
        include/luthier_types.h src/hsa_intercept.hpp src/hsa_intercept.cpp src/code_object_manager.hpp
        src/code_object_manager.cpp include/hip_arg_types.h include/hip_private_api.h include/error.h
        src/disassembler.hpp src/disassembler.cpp src/code_object_manipulation.hpp src/code_object_manipulation.cpp
        src/context_manager.hpp src/code_generator.hpp src/code_generator.cpp
        src/log.hpp
        src/instrumentation_function.hpp
        src/hsa_executable.hpp
        src/hsa_executable.cpp
        src/hsa_executable_symbol.hpp
        src/hsa_agent.hpp
        src/hsa_cache.hpp
        src/hsa_handle_type.hpp
        src/hsa_cache.cpp
        src/hsa_type.hpp
        src/hsa_agent.cpp
        src/hsa_executable_symbol.cpp
        src/hsa_loaded_code_object.hpp
        src/hsa_loaded_code_object.cpp
        src/hsa_isa.hpp
        src/hsa_isa.cpp
        src/code_view.hpp
        src/code_view.cpp
        src/hsa_kernel_descriptor.hpp
        src/llvm_mc.hpp
        src/llvm_mc.cpp
        src/context_manager.cpp
        src/hsa_instr.hpp
        src/hsa_instr.cpp)
# ======================================================================================================================

# User Options =========================================================================================================
set(LUTHIER_LOG_LEVEL "ERROR" CACHE STRING "Sets the LUTHIER log level; Options: ERROR, INFO, DEBUG")
if (${LUTHIER_LOG_LEVEL} STREQUAL "ERROR")
    target_compile_definitions(luthier PUBLIC "LUTHIER_LOG_ENABLE_ERROR")
elseif (${LUTHIER_LOG_LEVEL} STREQUAL "INFO")
    target_compile_definitions(luthier PUBLIC "LUTHIER_LOG_ENABLE_ERROR")
    target_compile_definitions(luthier PUBLIC "LUTHIER_LOG_ENABLE_INFO")
elseif (${LUTHIER_LOG_LEVEL} STREQUAL "DEBUG")
    target_compile_definitions(luthier PUBLIC "LUTHIER_LOG_ENABLE_ERROR")
    target_compile_definitions(luthier PUBLIC "LUTHIER_LOG_ENABLE_INFO")
    target_compile_definitions(luthier PUBLIC "LUTHIER_LOG_ENABLE_DEBUG")
else ()
    message(FATAL_ERROR "Invalid LUTHIER_LOG_LEVEL passed: ${LUTHIER_LOG_LEVEL}")
endif ()

option(LUTHIER_BUILD_EXAMPLES "Builds the example tools in the examples/ folder" ON)

set(BOOST_VERSION "1.83.0" CACHE STRING "C++ Boost library to download and build with Luthier")
# ======================================================================================================================

# External CXX Dependencies ============================================================================================
include(FetchContent)
include(ExternalProject)

# ELFIO Configuration ==================================================================================================
#TODO: Add an option to the user to set the path to these libraries if externally available
FetchContent_Declare(
        elfio
        DOWNLOAD_EXTRACT_TIMESTAMP 0
        URL "https://github.com/serge1/ELFIO/archive/refs/heads/main.zip"
        SOURCE_DIR ${CMAKE_SOURCE_DIR}/ext/elfio/
        BUILD_COMMAND ""
        CONFIGURE_COMMAND "cmake -DCMAKE_POSITION_INDEPENDENT_CODE=TRUE ..."
        INSTALL_COMMAND ""
)

FetchContent_MakeAvailable(elfio)

# ======================================================================================================================

# FMT Configuration ====================================================================================================
FetchContent_Declare(
        fmt
        DOWNLOAD_EXTRACT_TIMESTAMP 0
        URL "https://github.com/fmtlib/fmt/archive/refs/heads/master.zip"
        SOURCE_DIR ${CMAKE_SOURCE_DIR}/ext/fmt/
        CONFIGURE_COMMAND "cmake -DCMAKE_POSITION_INDEPENDENT_CODE=TRUE ..."
        INSTALL_COMMAND ""
)

FetchContent_MakeAvailable(fmt)

# ======================================================================================================================

# Boost Configuration ==================================================================================================
ExternalProject_add(boost
        DOWNLOAD_EXTRACT_TIMESTAMP 0
        URL https://github.com/boostorg/boost/releases/download/boost-${BOOST_VERSION}/boost-${BOOST_VERSION}.tar.xz
        CONFIGURE_COMMAND ./bootstrap.sh --with-libraries=iostreams --prefix=./build
        SOURCE_DIR ${CMAKE_SOURCE_DIR}/ext/boost/
        BUILD_COMMAND ./b2 install link=static
        BUILD_IN_SOURCE 1
        INSTALL_COMMAND ""
)

ExternalProject_Get_Property(boost SOURCE_DIR)

add_dependencies(luthier boost)

target_include_directories(luthier PRIVATE ${SOURCE_DIR}/build/include)

target_link_directories(luthier PUBLIC ${SOURCE_DIR}/build/lib)

# ======================================================================================================================

# ROCM Setup ===========================================================================================================

#TODO: make this take arbitrary paths for each ROCm library

find_package(hip REQUIRED)

find_package(amd_comgr REQUIRED)

find_package(hsa-runtime64 REQUIRED HINTS ${CMAKE_INSTALL_PREFIX} PATHS /opt/rocm/)

find_package(LLVM REQUIRED HINTS ${CMAKE_INSTALL_PREFIX} PATHS /opt/rocm/llvm/)
target_compile_definitions(luthier PUBLIC LLVM_DISABLE_ABI_BREAKING_CHECKS_ENFORCING=1)

llvm_map_components_to_libnames(support bitreader core mc)

include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

# Required since roctracer doesn't have a cmake config file
target_link_directories(luthier PUBLIC /opt/rocm/lib/)
target_include_directories(luthier PUBLIC /opt/rocm/include)

# Some HSA include files won't work unless this is defined
target_compile_definitions(luthier PUBLIC AMD_INTERNAL_BUILD)
# ======================================================================================================================

# Any Compilation Flags, Linking Configuration Left ====================================================================

target_compile_options(luthier INTERFACE "-fPIC")


# For LD_PRELOAD-ing the hip intercept functions
target_compile_definitions(luthier PUBLIC "-D_GLIBCXX_DEBUG")

target_include_directories(luthier PUBLIC ${CMAKE_SOURCE_DIR}/include)


target_link_libraries(luthier hip::host roctracer64 elfio::elfio stdc++fs amd_comgr fmt::fmt-header-only
        hsa-runtime64::hsa-runtime64 LLVMCore LLVMBitReader LLVMSupport LLVMMC LLVMAMDGPUTargetMCA LLVMAMDGPUInfo
        LLVMAMDGPUCodeGen LLVMAMDGPUDesc LLVMAMDGPUInfo LLVMAMDGPUDisassembler LLVMAMDGPUAsmParser
        boost_iostreams)
# ======================================================================================================================

# Example Tools Configuration (if enabled) =============================================================================
if (${LUTHIER_BUILD_EXAMPLES})
    set(SUB_DIRS examples/intercept examples/kernel_instrument examples/iterate_all_executable_symbols)
    foreach (DIR ${SUB_DIRS})
        add_subdirectory(${DIR})
    endforeach (DIR)
endif ()
# ======================================================================================================================
