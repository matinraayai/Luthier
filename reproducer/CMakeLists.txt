cmake_minimum_required(VERSION 3.21)

project(llvm-reproducer VERSION 0.0.1 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)

include(FetchContent)

FetchContent_Declare(
        elfio
        DOWNLOAD_EXTRACT_TIMESTAMP 1
        URL "https://github.com/serge1/ELFIO/archive/refs/heads/main.zip"
        SOURCE_DIR ${CMAKE_SOURCE_DIR}/ext/Source/elfio/
        INSTALL_COMMAND ""
)

FetchContent_MakeAvailable(elfio)

FetchContent_Declare(
        fmt
        DOWNLOAD_EXTRACT_TIMESTAMP 1
        URL "https://github.com/fmtlib/fmt/archive/refs/heads/master.zip"
        SOURCE_DIR ${CMAKE_SOURCE_DIR}/ext/Source/fmt/
        INSTALL_COMMAND ""
)

FetchContent_MakeAvailable(fmt)


add_executable(reproducer reproducer.cpp
        code_view.hpp
        code_view.cpp)

find_package(LLVM PATHS /opt/rocm/llvm)

find_package(hip)

find_package(amd_comgr REQUIRED)



target_compile_definitions(reproducer PUBLIC LLVM_DISABLE_ABI_BREAKING_CHECKS_ENFORCING=1)

target_include_directories(reproducer PUBLIC ${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

target_compile_definitions(reproducer PUBLIC LLVM_DISABLE_ABI_BREAKING_CHECKS_ENFORCING=1)

#target_include_directories(reproducer PUBLIC ../src/)
#target_include_directories(reproducer PUBLIC ../include/)

target_link_libraries(reproducer
        fmt::fmt-header-only
        amd_comgr elfio::elfio hip::host
        LLVMAMDGPUAsmParser LLVMAMDGPUCodeGen LLVMAMDGPUDesc LLVMAMDGPUDisassembler LLVMAMDGPUInfo LLVMAMDGPUTargetMCA
        LLVMAMDGPUUtils LLVMAggressiveInstCombine LLVMAnalysis LLVMAsmParser LLVMAsmPrinter LLVMBinaryFormat
        LLVMBitReader LLVMBitWriter LLVMBitstreamReader LLVMCFGuard LLVMCodeGen LLVMCore LLVMCoroutines
        LLVMDebugInfoCodeView LLVMDebugInfoDWARF LLVMDebugInfoMSF LLVMDebugInfoPDB LLVMDemangle
        LLVMFrontendOpenMP LLVMGlobalISel LLVMIRReader LLVMInstCombine
        LLVMInstrumentation LLVMLinker LLVMMC LLVMMCA LLVMMCDisassembler LLVMMCParser LLVMMIRParser LLVMObjCARCOpts
        LLVMObject LLVMPasses LLVMProfileData LLVMRemarks LLVMScalarOpts LLVMSelectionDAG LLVMSupport LLVMSymbolize
        LLVMTableGen LLVMTableGenGlobalISel LLVMTarget LLVMTextAPI LLVMTransformUtils LLVMVectorize
        LLVMipo)