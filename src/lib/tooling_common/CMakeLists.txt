# Add target to generate real to pseudo opcode map and the function to query it
set(luthier_tooling_common_TABLEGEN_EXE luthier-tblgen)
set(LLVM_TARGET_DEFINITIONS "${LUTHIER_LLVM_SRC_DIR}/llvm/lib/Target/AMDGPU/AMDGPU.td")
tablegen(luthier_tooling_common LuthierRealToPseudoOpcodeMap.hpp -gen-si-real-to-pseudo-opcode-map EXTRA_INCLUDES
        "${LUTHIER_LLVM_SRC_DIR}/llvm/lib/Target/AMDGPU/;${LUTHIER_LLVM_SRC_DIR}/llvm/include/")
add_public_tablegen_target(LuthierRealToPseudoOpcodeMap)

add_library(luthier_tooling_common STATIC
        code_generator.cpp
        code_lifter.cpp
        instrumentation_task.cpp
        luthier.cpp
        target_manager.cpp
        tool_executable_manager.cpp
        lifted_representation.cpp
        instrumentation_module.cpp
        intrinsic/IntrinsicLoweringInfo.cpp
        intrinsic/ReadReg.cpp
)

# Add explicit dependency between the tablegen target and the tooling common target to force cmake to run
# tablegen-ed target first
add_dependencies(luthier_tooling_common LuthierRealToPseudoOpcodeMap)

target_include_directories(luthier_tooling_common PUBLIC ${LUTHIER_PRIVATE_INCLUDE_DIR})
target_include_directories(luthier_tooling_common PUBLIC ${LUTHIER_PUBLIC_INCLUDE_DIR})
# Add the binary folder to the include directory to use LuthierRealToPseudoOpcodeMap.inc
target_include_directories(luthier_tooling_common PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

target_compile_options(luthier_tooling_common PUBLIC "-fPIC")
add_hsa_compile_definitions(luthier_tooling_common)
add_llvm_compile_definitions(luthier_tooling_common)

target_link_libraries(luthier_tooling_common PUBLIC luthier_common luthier_hip luthier_hsa
        hip::host rocprofiler-sdk::rocprofiler-sdk amd_comgr
        hsa-runtime64::hsa-runtime64 LLVMAMDGPUCodeGen LLVMAMDGPUDesc
        LLVMAMDGPUDisassembler LLVMAMDGPUInfo LLVMAMDGPUTargetMCA LLVMAMDGPUUtils LLVMAsmPrinter LLVMCodeGen
        LLVMAMDGPUAsmParser LLVMCodeGenTypes LLVMCore LLVMMC LLVMMCA LLVMMCDisassembler LLVMObject LLVMPasses
        LLVMSupport LLVMTarget LLVMTargetParser)

if (${LUTHIER_BUILD_UNIT_TESTS})
    add_subdirectory(unittest)
endif()
