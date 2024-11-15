# Add target to generate real to pseudo opcode map and the function to query it
set(LuthierToolingCommon_TABLEGEN_EXE luthier-tblgen)
set(LLVM_TARGET_DEFINITIONS "${LUTHIER_LLVM_SRC_DIR}/llvm/lib/Target/AMDGPU/AMDGPU.td")
tablegen(LuthierToolingCommon LuthierRealToPseudoOpcodeMap.hpp -gen-si-real-to-pseudo-opcode-map EXTRA_INCLUDES
        "${LUTHIER_LLVM_SRC_DIR}/llvm/lib/Target/AMDGPU/;${LUTHIER_LLVM_SRC_DIR}/llvm/include/")
add_public_tablegen_target(LuthierRealToPseudoOpcodeMap)

set(LuthierToolingCommon_TABLEGEN_EXE luthier-tblgen)
set(LLVM_TARGET_DEFINITIONS "${LUTHIER_LLVM_SRC_DIR}/llvm/lib/Target/AMDGPU/AMDGPU.td")
tablegen(LuthierToolingCommon LuthierRealToPseudoRegEnumMap.hpp -gen-si-real-to-pseudo-reg-enum-map EXTRA_INCLUDES
        "${LUTHIER_LLVM_SRC_DIR}/llvm/lib/Target/AMDGPU/;${LUTHIER_LLVM_SRC_DIR}/llvm/include/")
add_public_tablegen_target(LuthierRealToPseudoRegEnumMap)

add_library(LuthierToolingCommon STATIC
        CodeGenerator.cpp
        CodeLifter.cpp
        InstrumentationTask.cpp
        luthier.cpp
        TargetManager.cpp
        ToolExecutableManager.cpp
        LiftedRepresentation.cpp
        InstrumentationModule.cpp
        intrinsic/ReadReg.cpp
        intrinsic/WriteReg.cpp
        PhysicalRegAccessVirtualizationPass.cpp
        LRStateValueStorageAndLoadLocations.cpp
        LRCallGraph.cpp
        LRRegisterLiveness.cpp
        IntrinsicMIRLoweringPass.cpp
        InjectedPayloadPEIPass.cpp
        PreKernelEmitter.cpp
        intrinsic/WriteExec.cpp
        intrinsic/IntrinsicProcessor.cpp
        StateValueArraySpecs.cpp
        VectorCFG.cpp
)

# Add explicit dependency between the tablegen target and the tooling common target to force cmake to run
# tablegen-ed target first
add_dependencies(LuthierToolingCommon LuthierRealToPseudoOpcodeMap)
add_dependencies(LuthierToolingCommon LuthierRealToPseudoRegEnumMap)

target_include_directories(LuthierToolingCommon PUBLIC ${LUTHIER_PRIVATE_INCLUDE_DIR})
target_include_directories(LuthierToolingCommon PUBLIC ${LUTHIER_PUBLIC_INCLUDE_DIR})
# Add the binary folder to the include directory to use LuthierRealToPseudoOpcodeMap.inc
target_include_directories(LuthierToolingCommon PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

target_compile_options(LuthierToolingCommon PUBLIC "-fPIC")
add_hsa_compile_definitions(LuthierToolingCommon)
add_llvm_compile_definitions(LuthierToolingCommon)

target_link_libraries(LuthierToolingCommon PUBLIC LuthierCommon LuthierHip LuthierHSA
        hip::host rocprofiler-sdk::rocprofiler-sdk amd_comgr
        hsa-runtime64::hsa-runtime64 LLVMAMDGPUCodeGen LLVMAMDGPUDesc
        LLVMAMDGPUDisassembler LLVMAMDGPUInfo LLVMAMDGPUTargetMCA LLVMAMDGPUUtils LLVMAsmPrinter LLVMCodeGen
        LLVMAMDGPUAsmParser LLVMCodeGenTypes LLVMCore LLVMMC LLVMMCA LLVMMCDisassembler LLVMObject LLVMPasses
        LLVMSupport LLVMTarget LLVMTargetParser LLVMTransformUtils LLVMBitReader LLVMAnalysis)