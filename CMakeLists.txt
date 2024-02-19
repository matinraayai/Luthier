cmake_minimum_required(VERSION 3.21)

# Global CXX Setup =====================================================================================================
project(luthier VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

# Put the built libraries for all CMake projects under the lib/ folder
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# ======================================================================================================================

# Library Declaration ==================================================================================================
add_library(luthier STATIC
        src/luthier.cpp include/luthier.h src/hip_intercept.hpp src/hip_intercept.cpp
        include/luthier_types.h src/hsa_intercept.hpp src/hsa_intercept.cpp src/code_object_manager.hpp
        src/code_object_manager.cpp include/hip_arg_types.h include/hip_private_api.h src/error.hpp
        src/disassembler.hpp src/disassembler.cpp src/target_manager.hpp src/code_generator.hpp
        src/code_generator.cpp src/log.hpp src/instrumentation_function.hpp src/hsa_executable.hpp
        src/hsa_executable.cpp src/hsa_executable_symbol.hpp src/hsa_agent.hpp src/hsa_handle_type.hpp
        src/hsa_type.hpp src/hsa_agent.cpp src/hsa_executable_symbol.cpp src/hsa_loaded_code_object.hpp
        src/hsa_loaded_code_object.cpp src/hsa_isa.hpp src/hsa_isa.cpp src/object_utils.hpp src/object_utils.cpp
        src/hsa_kernel_descriptor.hpp src/target_manager.cpp src/hsa_instr.hpp src/hsa_instr.cpp
        src/hsa_code_object_reader.hpp src/hsa_code_object_reader.cpp src/hsa_kernel_descriptor.cpp
        src/hsa.hpp src/hsa.cpp
)
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
# ======================================================================================================================
# Required Flags for the project =======================================================================================
set_property(GLOBAL PROPERTY POSITION_INDEPENDENT_CODE TRUE)

# ======================================================================================================================
# ROCM Setup ===========================================================================================================
#TODO: make this take arbitrary paths for each ROCm library
find_package(hip REQUIRED)

find_package(amd_comgr REQUIRED)

find_package(hsa-runtime64 REQUIRED HINTS ${CMAKE_INSTALL_PREFIX} PATHS /opt/rocm/)
# Some HSA include files won't work unless this is defined
target_compile_definitions(luthier PUBLIC AMD_INTERNAL_BUILD)

# LLVM AMDGPU Configuration (for tablegen and code generation) =========================================================
find_package(LLVM REQUIRED CONFIG PATHS ${LUTHIER_LLVM_BIN_DIR} NO_DEFAULT_PATH)

target_include_directories(luthier PUBLIC ${LLVM_INCLUDE_DIRS})

target_include_directories(luthier PUBLIC ${LUTHIER_LLVM_BIN_DIR}/lib/Target/AMDGPU)

target_include_directories(luthier PUBLIC ${LUTHIER_LLVM_SRC_DIR}/llvm/lib/Target/AMDGPU/)

target_include_directories(luthier PUBLIC ${LUTHIER_LLVM_BIN_DIR}/lib/Target/AMDGPU)

# ======================================================================================================================
target_compile_definitions(luthier PUBLIC LLVM_DISABLE_ABI_BREAKING_CHECKS_ENFORCING=1)

#include_directories(${LLVM_INCLUDE_DIRS})
#add_definitions(${LLVM_DEFINITIONS})
# ======================================================================================================================

# Any Compilation Flags, Linking Configuration Left ====================================================================
target_compile_options(luthier PUBLIC "-fPIC")

#target_compile_options(luthier PUBLIC "-fsanitize=address")

#target_compile_options(luthier PUBLIC "-static-libsan")

#target_compile_options(luthier PUBLIC "-g3")

#target_compile_options(luthier PUBLIC "-fno-omit-frame-pointer")

# For LD_PRELOAD-ing the hip intercept functions
target_compile_definitions(luthier PUBLIC "-D_GLIBCXX_DEBUG")

#target_link_options(luthier PUBLIC "-static-libsan")

#target_compile_options(luthier PUBLIC "-static-libsan")

#set_property(TARGET luthier PROPERTY HIP_ARCHITECTURES OFF)

#target_compile_options(luthier PUBLIC "--offload-arch=gfx908:xnack+")

target_include_directories(luthier PUBLIC ${CMAKE_SOURCE_DIR}/include)

#target_link_options(luthier PUBLIC "-fsanitize=address")
#target_link_options(luthier PUBLIC "-static-libsan")

#target_link_options(luthier PUBLIC "--offload-arch=gfx908:xnack+")

target_link_libraries(luthier hip::host stdc++fs hsa-runtime64::hsa-runtime64
        LLVMAMDGPUAsmParser LLVMAMDGPUCodeGen LLVMAMDGPUDesc LLVMAMDGPUDisassembler LLVMAMDGPUInfo LLVMAMDGPUTargetMCA
        LLVMAMDGPUUtils LLVMAsmParser LLVMAsmPrinter
        LLVMCodeGen LLVMCodeGenTypes LLVMCore
        LLVMMC LLVMMCA LLVMMCDisassembler LLVMMCParser LLVMMIRParser
        LLVMObject LLVMPasses LLVMSupport
        LLVMTarget LLVMTargetParser)
# ======================================================================================================================

# Example Tools Configuration (if enabled) =============================================================================
if (${LUTHIER_BUILD_EXAMPLES})
    set(SUB_DIRS examples/intercept examples/kernel_instrument examples/iterate_all_executable_symbols)
    foreach (DIR ${SUB_DIRS})
        add_subdirectory(${DIR})
    endforeach (DIR)
endif ()
# ======================================================================================================================
