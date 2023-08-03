#cmake_minimum_required(VERSION 3.16)
#
## CXX Setup ============================================================================================================
#project(sibir VERSION 0.0.1 LANGUAGES CXX)
#
#set(CMAKE_CXX_STANDARD 20)
## ======================================================================================================================
#
## Library Creation =====================================================================================================
#add_library(sibir SHARED sibir_impl.hpp sibir_impl.cpp sibir.h hip_intercept.hpp hip_intercept.cpp sibir_types.hpp
#        hsa_intercept.hpp hsa_intercept.cpp code_object_manager.hpp code_object_manager.cpp hip_arg_types.h
#        hip_private_api.h error_and_log.hpp disassembler.hpp disassembler.cpp amdgpu_elf.hpp amdgpu_elf.cpp
#        context_manager.hpp context_manager.cpp code_generator.hpp code_generator.cpp instr.hpp instr.cpp)
## ======================================================================================================================
#
## User Options =========================================================================================================
#set(SIBIR_LOG_LEVEL "ERROR" CACHE STRING "Sets the Sibir log level; Options: ERROR, INFO, WARNING, DEBUG")
#if(${SIBIR_LOG_LEVEL} STREQUAL "ERROR")
#    add_compile_definitions(sibir PUBLIC "SIBIR_LOG_ENABLE_ERROR=1")
#elseif (${SIBIR_LOG_LEVEL} STREQUAL "INFO")
#    add_compile_definitions(sibir PUBLIC "SIBIR_LOG_ENABLE_ERROR=1")
#    add_compile_definitions(sibir PUBLIC "SIBIR_LOG_ENABLE_INFO=1")
#elseif (${SIBIR_LOG_LEVEL} STREQUAL "WARNING")
#    add_compile_definitions(sibir PUBLIC "SIBIR_LOG_ENABLE_ERROR=1")
#    add_compile_definitions(sibir PUBLIC "SIBIR_LOG_ENABLE_INFO=1")
#    add_compile_definitions(sibir PUBLIC "SIBIR_LOG_ENABLE_WARNING=1")
#elseif (${SIBIR_LOG_LEVEL} STREQUAL "DEBUG")
#    add_compile_definitions(sibir PUBLIC "SIBIR_ENABLE_ERROR=1")
#    add_compile_definitions(sibir PUBLIC "SIBIR_ENABLE_INFO=1")
#    add_compile_definitions(sibir PUBLIC "SIBIR_LOG_ENABLE_WARNING=1")
#    add_compile_definitions(sibir PUBLIC "SIBIR_LOG_ENABLE_DEBUG=1")
#else ()
#    message(FATAL_ERROR "Invalid SIBIR_LOG_LEVEL passed: ${SIBIR_LOG_LEVEL}")
#endif ()
### TODO: Make this a cmake parameter
##target_compile_definitions(sibir PUBLIC "SIBIR_ENABLE_DEBUG=${SIBIR_LOG_LEVEL}")
#
## ======================================================================================================================
#
## External CXX Dependencies ============================================================================================
#include(FetchContent)
#
## ELFIO Configuration ==================================================================================================
##TODO: Add an option to the user to set the path to these libraries if externally available
#FetchContent_Declare(
#        elfio
#        DOWNLOAD_EXTRACT_TIMESTAMP 0
#        URL "https://github.com/serge1/ELFIO/archive/refs/heads/main.zip"
#        SOURCE_DIR ${CMAKE_SOURCE_DIR}/ext/elfio/
#        BUILD_COMMAND ""
#        CONFIGURE_COMMAND ""
#)
#
#FetchContent_MakeAvailable(elfio)
#
#target_include_directories(sibir PRIVATE ${elfio_SOURCE_DIR})
## ======================================================================================================================
#
## FMT Configuration ====================================================================================================
#FetchContent_Declare(
#        fmt
#        DOWNLOAD_EXTRACT_TIMESTAMP 0
#        URL "https://github.com/fmtlib/fmt/archive/refs/heads/master.zip"
#        SOURCE_DIR ${CMAKE_SOURCE_DIR}/ext/fmt/
#        CONFIGURE_COMMAND "cmake -DCMAKE_POSITION_INDEPENDENT_CODE=TRUE ..."
#)
#
#FetchContent_MakeAvailable(fmt)
## ======================================================================================================================
## ======================================================================================================================
#
## ROCM Setup ===========================================================================================================
#
##TODO: make this take arbitrary paths for each ROCm library
##list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm)
#
#find_package(hip REQUIRED)
#
#find_package(amd_comgr REQUIRED)
#
#target_link_directories(sibir PUBLIC /opt/rocm/lib/)
#target_include_directories(sibir PUBLIC /opt/rocm/include)
#
## Some HSA include files won't work unless this is defined
#target_compile_definitions(sibir PUBLIC AMD_INTERNAL_BUILD)
## ======================================================================================================================
#
#
#set_property(TARGET sibir PROPERTY COMPILE_FLAGS "-fPIC")
#
#target_link_libraries(sibir hip::host roctracer64 stdc++fs amd_comgr fmt::fmt)
