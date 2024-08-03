if (${LUTHIER_GEN_INTERCEPTOR_IMPL_DURING_CONFIG})
    execute_process(COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/hsa_intercept_gen.py
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/scripts)
endif ()

add_custom_command(OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/hsa_intercept.cpp
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/hsa_intercept_gen.py
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/scripts
        COMMENT "Generating HSA API interceptor implementation")

add_library(luthier_hsa STATIC
        ${CMAKE_CURRENT_SOURCE_DIR}/hsa_intercept.cpp
        hsa_executable.cpp
        hsa_agent.cpp
        hsa_executable_symbol.cpp
        hsa_loaded_code_object.cpp
        hsa_isa.cpp
        instr.cpp
        hsa_code_object_reader.cpp
        kernel_descriptor.cpp
        hsa.cpp
        hsa_platform.cpp
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