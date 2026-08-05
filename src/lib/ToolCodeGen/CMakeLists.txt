# Add target to generate real to pseudo opcode and register map and functions to query them
set(LuthierRealToPseudoOpcodeMap_TABLEGEN_EXE luthier-tblgen)
set(LLVM_TARGET_DEFINITIONS "${LUTHIER_LLVM_SRC_DIR}/llvm/lib/Target/AMDGPU/AMDGPU.td")
tablegen(LuthierRealToPseudoOpcodeMap LuthierRealToPseudoOpcodeMap.hpp -gen-si-real-to-pseudo-opcode-map EXTRA_INCLUDES
        "${LUTHIER_LLVM_SRC_DIR}/llvm/lib/Target/AMDGPU/;${LUTHIER_LLVM_SRC_DIR}/llvm/include/")
add_public_tablegen_target(LuthierRealToPseudoOpcodeMap)

set(LuthierRealToPseudoRegEnumMap_TABLEGEN_EXE luthier-tblgen)
set(LLVM_TARGET_DEFINITIONS "${LUTHIER_LLVM_SRC_DIR}/llvm/lib/Target/AMDGPU/AMDGPU.td")
tablegen(LuthierRealToPseudoRegEnumMap LuthierRealToPseudoRegEnumMap.hpp -gen-si-real-to-pseudo-reg-map EXTRA_INCLUDES
        "${LUTHIER_LLVM_SRC_DIR}/llvm/lib/Target/AMDGPU/;${LUTHIER_LLVM_SRC_DIR}/llvm/include/")
add_public_tablegen_target(LuthierRealToPseudoRegEnumMap)

# Tablegen target for generating IR translation of MIR pseudo instructions
set(LuthierSIInstrSemanticsGen_TABLEGEN_EXE luthier-tblgen)
set(LLVM_TARGET_DEFINITIONS "${CMAKE_SOURCE_DIR}/include/luthier/InstSemantics/InstSISemantics.td")
tablegen(LuthierSIInstrSemanticsGen SIInstrSemantics.inc -gen-si-instr-semantics EXTRA_INCLUDES
        "${LUTHIER_LLVM_SRC_DIR}/llvm/lib/Target/AMDGPU/;${LUTHIER_LLVM_SRC_DIR}/llvm/include/;"
        "${CMAKE_SOURCE_DIR}/include/luthier/InstSemantics/")
add_public_tablegen_target(LuthierSIInstrSemanticsGen)

add_library(LuthierToolCodeGen OBJECT
        DebugInfoPass.cpp
        PseudoOpcodeAndRegMapper.cpp
        TargetMachineInstrMDNode.cpp
        CodeDiscoveryPass.cpp
        InstructionTracesAnalysis.cpp
        MemoryAllocationAccessor.cpp
        SVStorageAndLoadLocations.cpp
        SVAPhysVGPRPinPass.cpp
        TargetModulePatcherPass.cpp
        LuthierRegScavenger.cpp
        LuthierBranchRelaxation.cpp
        IntrinsicMIRLoweringPass.cpp
        InjectedPayloadPEIPass.cpp
        PrePostAmbleEmitter.cpp
        StateValueArraySpecs.cpp
        IPPredicatedCFG.cpp
        StateValueArrayStorage.cpp
        ProcessIntrinsicsAtIRLevelPass.cpp
        ForwardISAStateToCalleesPass.cpp
        IPPredicatedLivenessIModulePass.cpp
        InjectedPayloadPreserveLiveRegsPass.cpp
        WrapperAnalysisPasses.cpp
        MIRConvenience.cpp
        MetadataParserAnalysis.cpp
        IntrinsicProcessorRegistry.cpp
        IntrinsicProcessorsAnalysis.cpp
        InjectedPayloadAndInstPointAnalysis.cpp
        InjectedPayloadAccessedRegsAnalysis.cpp
        InstrumentationPMDriver.cpp
        InitialEntryPointAnalysis.cpp
        InitialExecutionPointAnalysis.cpp
        Metadata.cpp
        MIRToIRTranslator.cpp
        RegValueMetadata.cpp
        MIInlineAsmEmitter.cpp
        NewPMAsmPrinter.cpp
        FunctionAnnotations.cpp
        TraceCallGraph.cpp
        PredicatedMachineBasicBlock.cpp

)

add_dependencies(LuthierToolCodeGen LuthierRealToPseudoOpcodeMap)
add_dependencies(LuthierToolCodeGen LuthierRealToPseudoRegEnumMap)
add_dependencies(LuthierToolCodeGen LuthierSIInstrSemanticsGen)

target_compile_definitions(LuthierToolCodeGen PRIVATE AMD_INTERNAL_BUILD ${LLVM_DEFINITIONS})

target_include_directories(LuthierToolCodeGen
        PRIVATE
        ${LLVM_INCLUDE_DIRS}
        ${CMAKE_CURRENT_BINARY_DIR}
        ${hsa-runtime64_INCLUDE_DIRS}
        PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>"
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

target_link_libraries(LuthierToolCodeGen PRIVATE LuthierAMDGPU)