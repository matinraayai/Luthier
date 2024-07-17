add_library(luthier_tooling_common STATIC
        code_generator.cpp
        code_lifter.cpp
        instrumentation_task.cpp
        luthier.cpp
        target_manager.cpp
        tool_executable_manager.cpp
        lifted_representation.cpp
        instrumentation_module.cpp
)

target_include_directories(luthier_tooling_common PUBLIC ${LUTHIER_PRIVATE_INCLUDE_DIR})
target_include_directories(luthier_tooling_common PUBLIC ${LUTHIER_PUBLIC_INCLUDE_DIR})

target_compile_options(luthier_tooling_common PUBLIC "-fPIC")
add_hsa_compile_definitions(luthier_tooling_common)
add_llvm_compile_definitions(luthier_tooling_common)

target_link_libraries(luthier_tooling_common PUBLIC luthier_common luthier_hip luthier_hsa
        hip::host rocprofiler-sdk::rocprofiler-sdk amd_comgr
        hsa-runtime64::hsa-runtime64 LLVMAMDGPUCodeGen LLVMAMDGPUDesc
        LLVMAMDGPUDisassembler LLVMAMDGPUInfo LLVMAMDGPUTargetMCA LLVMAMDGPUUtils LLVMAsmPrinter LLVMCodeGen
        LLVMAMDGPUAsmParser LLVMCodeGenTypes LLVMCore LLVMMC LLVMMCA LLVMMCDisassembler LLVMObject LLVMPasses
        LLVMSupport LLVMTarget LLVMTargetParser)