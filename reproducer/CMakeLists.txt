cmake_minimum_required(VERSION 3.21)

project(luthier VERSION 0.0.1 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)


add_executable(reproducer reproducer.cpp)

find_package(LLVM PATHS /opt/rocm/llvm)
target_compile_definitions(reproducer PUBLIC LLVM_DISABLE_ABI_BREAKING_CHECKS_ENFORCING=1)

target_include_directories(reproducer PUBLIC ${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

target_compile_definitions(reproducer PUBLIC LLVM_DISABLE_ABI_BREAKING_CHECKS_ENFORCING=1)

target_link_libraries(reproducer
        LLVMAMDGPUAsmParser LLVMAMDGPUCodeGen LLVMAMDGPUDesc LLVMAMDGPUDisassembler LLVMAMDGPUInfo LLVMAMDGPUTargetMCA
        LLVMAMDGPUUtils LLVMAggressiveInstCombine LLVMAnalysis LLVMAsmParser LLVMAsmPrinter LLVMBinaryFormat
        LLVMBitReader LLVMBitWriter LLVMBitstreamReader LLVMCFGuard LLVMCodeGen LLVMCore LLVMCoroutines
        LLVMDebugInfoCodeView LLVMDebugInfoDWARF LLVMDebugInfoMSF LLVMDebugInfoPDB LLVMDemangle
        LLVMFrontendOpenMP LLVMGlobalISel LLVMIRReader LLVMInstCombine
        LLVMInstrumentation LLVMLinker LLVMMC LLVMMCA LLVMMCDisassembler LLVMMCParser LLVMMIRParser LLVMObjCARCOpts
        LLVMObject LLVMPasses LLVMProfileData LLVMRemarks LLVMScalarOpts LLVMSelectionDAG LLVMSupport LLVMSymbolize
        LLVMTableGen LLVMTableGenGlobalISel LLVMTarget LLVMTextAPI LLVMTransformUtils LLVMVectorize
        LLVMipo)