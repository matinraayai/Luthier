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
        src/disassembler.hpp src/disassembler.cpp
        src/target_manager.hpp src/code_generator.hpp src/code_generator.cpp src/log.hpp
        src/instrumentation_function.hpp src/hsa_executable.hpp src/hsa_executable.cpp src/hsa_executable_symbol.hpp
        src/hsa_agent.hpp src/hsa_handle_type.hpp src/hsa_type.hpp src/hsa_agent.cpp src/hsa_executable_symbol.cpp
        src/hsa_loaded_code_object.hpp src/hsa_loaded_code_object.cpp src/hsa_isa.hpp src/hsa_isa.cpp
        src/code_view.hpp src/code_view.cpp src/hsa_kernel_descriptor.hpp src/target_manager.cpp src/hsa_instr.hpp
        src/hsa_instr.cpp src/hsa_code_object_reader.hpp src/hsa_code_object_reader.cpp src/hsa_kernel_descriptor.cpp
        src/hsa.hpp src/hsa.cpp)
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
# Required Flags for the project =======================================================================================
set_property(GLOBAL PROPERTY POSITION_INDEPENDENT_CODE TRUE)

# External CXX Dependencies ============================================================================================
include(FetchContent)
include(ExternalProject)

set_directory_properties(PROPERTIES EP_BASE ${CMAKE_SOURCE_DIR}/ext/)

# ELFIO Configuration ==================================================================================================
#TODO: Add an option to the user to set the path to these libraries if externally available
FetchContent_Declare(
        elfio
        DOWNLOAD_EXTRACT_TIMESTAMP 1
        URL "https://github.com/serge1/ELFIO/archive/refs/heads/main.zip"
        SOURCE_DIR ${CMAKE_SOURCE_DIR}/ext/Source/elfio/
        INSTALL_COMMAND ""
)

FetchContent_MakeAvailable(elfio)

# ======================================================================================================================

# FMT Configuration ====================================================================================================
FetchContent_Declare(
        fmt
        DOWNLOAD_EXTRACT_TIMESTAMP 1
        URL "https://github.com/fmtlib/fmt/archive/refs/heads/master.zip"
        SOURCE_DIR ${CMAKE_SOURCE_DIR}/ext/Source/fmt/
        INSTALL_COMMAND ""
)

FetchContent_MakeAvailable(fmt)

# ======================================================================================================================

# Boost Configuration ==================================================================================================
ExternalProject_add(boost
        DOWNLOAD_EXTRACT_TIMESTAMP 1
        URL https://github.com/boostorg/boost/releases/download/boost-${BOOST_VERSION}/boost-${BOOST_VERSION}.tar.xz
        CONFIGURE_COMMAND ./bootstrap.sh --with-libraries=iostreams --prefix=./build
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
# Some HSA include files won't work unless this is defined
target_compile_definitions(luthier PUBLIC AMD_INTERNAL_BUILD)

# LLVM AMDGPU Configuration (for tablegen and code generation) =========================================================
# TODO: Wait for ROCm 6.1 to be released to see if compiling LLVM from scratch can be avoided + match the rocm version
ExternalProject_add(
        llvm-project
        DOWNLOAD_EXTRACT_TIMESTAMP 1
        URL https://github.com/ROCm/llvm-project/archive/refs/heads/amd-staging.zip
#        URL https://github.com/ROCm/llvm-project/archive/refs/tags/rocm-${hip_VERSION_MAJOR}.${hip_VERSION_MINOR}.${hip_VERSION_TWEAK}.zip
        CMAKE_ARGS -DCMAKE_BUILD_TYPE=RelWithDebInfo -DLLVM_TARGETS_TO_BUILD=AMDGPU\;
        -DCMAKE_POSITION_INDEPENDENT_CODE=TRUE -DLLVM_ENABLE_RTTI=ON
        SOURCE_SUBDIR llvm
        BUILD_COMMAND ${CMAKE_COMMAND} --build . --target AMDGPUCommonTableGen InstCombineTableGen LLVMCore
        LLVMBitReader LLVMSupport LLVMMC LLVMAMDGPUTargetMCA LLVMAMDGPUInfo LLVMAMDGPUCodeGen LLVMAMDGPUDesc
        LLVMAMDGPUInfo LLVMAMDGPUDisassembler LLVMAMDGPUAsmParser
#        BUILD_COMMAND ${CMAKE_COMMAND} --build . --target AMDGPUCommonTableGen InstCombineTableGen
        INSTALL_COMMAND ""
)

add_dependencies(luthier llvm-project)

ExternalProject_Get_Property(llvm-project BINARY_DIR)

ExternalProject_Get_Property(llvm-project SOURCE_DIR)

target_include_directories(luthier PRIVATE ${BINARY_DIR}/lib/Target/AMDGPU/)

target_link_directories(luthier PUBLIC ${BINARY_DIR}/lib)

target_include_directories(luthier PUBLIC ${SOURCE_DIR}/llvm/lib/Target/AMDGPU/)

target_include_directories(luthier PUBLIC ${SOURCE_DIR}/llvm/include)

target_include_directories(luthier PUBLIC ${BINARY_DIR}/include)

#set(LLVM_DIR ${BINARY_DIR})

# ======================================================================================================================
# TODO: Fix the issue with ExternalProjectAdd
#find_package(LLVM REQUIRED PATHS ${LLVM_DIR} NO_DEFAULT_PATH)
target_compile_definitions(luthier PUBLIC LLVM_DISABLE_ABI_BREAKING_CHECKS_ENFORCING=1)

#include_directories(${LLVM_INCLUDE_DIRS})
#add_definitions(${LLVM_DEFINITIONS})
# ======================================================================================================================

# Any Compilation Flags, Linking Configuration Left ====================================================================

target_compile_options(luthier INTERFACE "-fPIC")


# For LD_PRELOAD-ing the hip intercept functions
target_compile_definitions(luthier PUBLIC "-D_GLIBCXX_DEBUG")

target_include_directories(luthier PUBLIC ${CMAKE_SOURCE_DIR}/include)


target_link_libraries(luthier hip::host elfio::elfio stdc++fs amd_comgr fmt::fmt-header-only
        hsa-runtime64::hsa-runtime64
        LLVMAMDGPUAsmParser LLVMAMDGPUCodeGen LLVMAMDGPUDesc LLVMAMDGPUDisassembler LLVMAMDGPUInfo LLVMAMDGPUTargetMCA
        LLVMAMDGPUUtils LLVMAggressiveInstCombine LLVMAnalysis LLVMAsmParser LLVMAsmPrinter LLVMBinaryFormat
        LLVMBitReader LLVMBitWriter LLVMBitstreamReader LLVMCFGuard LLVMCodeGen LLVMCodeGenTypes LLVMCore LLVMCoroutines
        LLVMDebugInfoBTF LLVMDebugInfoCodeView LLVMDebugInfoDWARF LLVMDebugInfoMSF LLVMDebugInfoPDB LLVMDemangle
        LLVMFrontendOffloading LLVMFrontendOpenMP LLVMGlobalISel LLVMHipStdPar LLVMIRPrinter LLVMIRReader LLVMInstCombine
        LLVMInstrumentation LLVMLinker LLVMMC LLVMMCA LLVMMCDisassembler LLVMMCParser LLVMMIRParser LLVMObjCARCOpts
        LLVMObject LLVMPasses LLVMProfileData LLVMRemarks LLVMScalarOpts LLVMSelectionDAG LLVMSupport LLVMSymbolize
        LLVMTableGen LLVMTableGenGlobalISel LLVMTarget LLVMTargetParser LLVMTextAPI LLVMTransformUtils LLVMVectorize
        LLVMipo boost_iostreams)
# ======================================================================================================================

# Example Tools Configuration (if enabled) =============================================================================
if (${LUTHIER_BUILD_EXAMPLES})
    set(SUB_DIRS examples/intercept examples/kernel_instrument examples/iterate_all_executable_symbols)
    foreach (DIR ${SUB_DIRS})
        add_subdirectory(${DIR})
    endforeach (DIR)
endif ()
# ======================================================================================================================
