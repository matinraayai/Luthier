get_target_property(HSA_INCLUDE_DIRS hsa-runtime64::hsa-runtime64 INTERFACE_INCLUDE_DIRECTORIES)

if (${LUTHIER_GEN_INTERCEPTOR_IMPL_DURING_CONFIG})
    execute_process(COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/hsa_intercept_gen.py
            --hsa-include-dir ${HSA_INCLUDE_DIRS}/hsa
            --cpp-implementation-save-path ${CMAKE_CURRENT_SOURCE_DIR}/HsaRuntimeInterceptor.cpp
            --h-api-enums-structs-save-path ${CMAKE_SOURCE_DIR}/include/luthier/hsa/TraceApi.h
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/scripts)
endif ()

add_custom_command(OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/HsaRuntimeInterceptor.cpp
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/hsa_intercept_gen.py
        --hsa-include-dir ${HSA_INCLUDE_DIRS}/hsa
        --cpp-implementation-save-path ${CMAKE_CURRENT_SOURCE_DIR}/HsaRuntimeInterceptor.cpp
        --h-api-enums-structs-save-path ${CMAKE_SOURCE_DIR}/include/luthier/hsa/TraceApi.h
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/scripts
        COMMENT "Generating HSA API interceptor implementation")

add_library(luthier_hsa STATIC
        ${CMAKE_CURRENT_SOURCE_DIR}/HsaRuntimeInterceptor.cpp
        Executable.cpp
        GpuAgent.cpp
        ExecutableSymbol.cpp
        LoadedCodeObject.cpp
        ISA.cpp
        Instr.cpp
        CodeObjectReader.cpp
        KernelDescriptor.cpp
        hsa.cpp
        ExecutableBackedObjectsCache.cpp
        LoadedCodeObjectSymbol.cpp
        LoadedCodeObjectKernel.cpp
        LoadedCodeObjectVariable.cpp
        LoadedCodeObjectDeviceFunction.cpp
        LoadedCodeObjectExternSymbol.cpp
)

target_include_directories(luthier_hsa PUBLIC ${LUTHIER_PRIVATE_INCLUDE_DIR})
target_include_directories(luthier_hsa PUBLIC ${LUTHIER_PUBLIC_INCLUDE_DIR})

target_compile_options(luthier_hsa PUBLIC "-fPIC")
add_hsa_compile_definitions(luthier_hsa)
add_llvm_compile_definitions(luthier_hsa)

target_link_libraries(luthier_hsa PUBLIC luthier_common hsa-runtime64::hsa-runtime64 LLVMAMDGPUCodeGen LLVMAMDGPUDesc
        LLVMAMDGPUDisassembler LLVMAMDGPUInfo LLVMAMDGPUTargetMCA LLVMAMDGPUUtils LLVMAsmPrinter LLVMCodeGen
        LLVMCodeGenTypes LLVMCore LLVMMC LLVMMCA LLVMMCDisassembler LLVMObject LLVMPasses
        LLVMSupport LLVMTarget LLVMTargetParser)