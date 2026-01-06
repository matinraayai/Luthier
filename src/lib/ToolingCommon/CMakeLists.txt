# Add target to generate real to pseudo opcode and register map and functions to query them
set(LuthierToolingCommon_TABLEGEN_EXE luthier-tblgen)
set(LLVM_TARGET_DEFINITIONS "${LUTHIER_LLVM_SRC_DIR}/llvm/lib/Target/AMDGPU/AMDGPU.td")
tablegen(LuthierToolingCommon LuthierRealToPseudoOpcodeMap.hpp -gen-si-real-to-pseudo-opcode-map EXTRA_INCLUDES
        "${LUTHIER_LLVM_SRC_DIR}/llvm/lib/Target/AMDGPU/;${LUTHIER_LLVM_SRC_DIR}/llvm/include/")
add_public_tablegen_target(LuthierRealToPseudoOpcodeMap)

set(LuthierToolingCommon_TABLEGEN_EXE luthier-tblgen)
set(LLVM_TARGET_DEFINITIONS "${LUTHIER_LLVM_SRC_DIR}/llvm/lib/Target/AMDGPU/AMDGPU.td")
tablegen(LuthierToolingCommon LuthierRealToPseudoRegEnumMap.hpp -gen-si-real-to-pseudo-reg-map EXTRA_INCLUDES
        "${LUTHIER_LLVM_SRC_DIR}/llvm/lib/Target/AMDGPU/;${LUTHIER_LLVM_SRC_DIR}/llvm/include/")
add_public_tablegen_target(LuthierRealToPseudoRegEnumMap)

add_library(LuthierToolingCommon OBJECT
        AnnotatedMachineInstr.cpp
        CodeDiscoveryPass.cpp
        MachineFunctionEntryPoints.cpp
        MachineToTraceInstrAnalysis.cpp
        InstructionTracesAnalysis.cpp
        MemoryAllocationAccessor.cpp
        InstructionModel.cpp
        LiftedRepresentation.cpp
        PhysicalRegAccessVirtualizationPass.cpp
        SVStorageAndLoadLocations.cpp
        LRCallGraph.cpp
        AMDGPURegisterLiveness.cpp
        IntrinsicMIRLoweringPass.cpp
        InjectedPayloadPEIPass.cpp
        PrePostAmbleEmitter.cpp
        StateValueArraySpecs.cpp
        VectorCFG.cpp
        StateValueArrayStorage.cpp
        IModuleIRGeneratorPass.cpp
        RunIRPassesOnIModulePass.cpp
        MMISlotIndexesAnalysis.cpp
        ProcessIntrinsicsAtIRLevelPass.cpp
        PhysRegsNotInLiveInsAnalysis.cpp
        WrapperAnalysisPasses.cpp
        RunMIRPassesOnIModulePass.cpp
        PatchLiftedRepresentationPass.cpp
        MIRConvenience.cpp
        MockAMDGPULoader.cpp
        MetadataParserAnalysis.cpp
        ImmutableMachineInstr.cpp
        IntrinsicProcessorRegistry.cpp
        IntrinsicProcessorsAnalysis.cpp
        InstrumentationPMDriver.cpp
        MockLoadAMDGPUCodeObjects.cpp
        CodeObjectManagerAnalysis.cpp
        InitialEntryPointAnalysis.cpp
        PseudoOpcodeAndRegMapper.cpp
)

add_dependencies(LuthierToolingCommon LuthierRealToPseudoOpcodeMap)
add_dependencies(LuthierToolingCommon LuthierRealToPseudoRegEnumMap)

target_compile_definitions(LuthierToolingCommon PRIVATE AMD_INTERNAL_BUILD ${LLVM_DEFINITIONS})

target_include_directories(LuthierToolingCommon
        PRIVATE
        ${LLVM_INCLUDE_DIRS}
        ${CMAKE_CURRENT_BINARY_DIR}
        ${hsa-runtime64_INCLUDE_DIRS}
        PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>"
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

target_link_libraries(LuthierToolingCommon PRIVATE LuthierAMDGPU hip::host)