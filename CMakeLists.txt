cmake_minimum_required(VERSION 3.21)

# Global CXX Setup =====================================================================================================
project(luthier VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

# Put results in the lib/ folder
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# ======================================================================================================================

# Library Declaration ==================================================================================================
add_library(luthier SHARED
        src/luthier_impl.hpp src/luthier_impl.cpp include/luthier.h src/hip_intercept.hpp src/hip_intercept.cpp
        include/luthier_types.hpp src/hsa_intercept.hpp src/hsa_intercept.cpp src/code_object_manager.hpp
        src/code_object_manager.cpp include/hip_arg_types.h include/hip_private_api.h include/error.h
        src/disassembler.hpp src/disassembler.cpp src/amdgpu_elf.hpp src/amdgpu_elf.cpp
        src/context_manager.hpp src/context_manager.cpp src/code_generator.hpp src/code_generator.cpp include/instr.hpp
        src/instr.cpp src/log.hpp)
# ======================================================================================================================

# User Options =========================================================================================================
set(LUTHIER_LOG_LEVEL "ERROR" CACHE STRING "Sets the LUTHIER log level; Options: ERROR, INFO, DEBUG")
if(${LUTHIER_LOG_LEVEL} STREQUAL "ERROR")
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
# ======================================================================================================================

# External CXX Dependencies ============================================================================================
include(FetchContent)

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
# GOTCHA Configuration ====================================================================================================
FetchContent_Declare(
        gotcha
        DOWNLOAD_EXTRACT_TIMESTAMP 0
        URL "https://github.com/LLNL/GOTCHA/archive/refs/heads/develop.zip"
        SOURCE_DIR ${CMAKE_SOURCE_DIR}/ext/gotcha/
        CONFIGURE_COMMAND "cmake -DCMAKE_POSITION_INDEPENDENT_CODE=TRUE ..."
        INSTALL_COMMAND ""
)

FetchContent_MakeAvailable(gotcha)

# ======================================================================================================================
# ======================================================================================================================


# ROCM Setup ===========================================================================================================

#TODO: make this take arbitrary paths for each ROCm library

find_package(hip REQUIRED)

find_package(amd_comgr REQUIRED)

find_package(hsa-runtime64 REQUIRED HINTS ${CMAKE_INSTALL_PREFIX} PATHS /opt/rocm/)

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
        hsa-runtime64::hsa-runtime64)
# ======================================================================================================================

# Example Tools Configuration (if enabled) =============================================================================
if (${LUTHIER_BUILD_EXAMPLES})
  set(SUB_DIRS examples/intercept)
  foreach (DIR ${SUB_DIRS})
    add_subdirectory(${DIR})
  endforeach (DIR)
endif()
# ======================================================================================================================



