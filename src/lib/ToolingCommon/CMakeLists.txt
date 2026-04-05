# Tablegen target for generating IR translation of MIR pseudo instructions
set(LuthierSIInstrSemanticsGen_TABLEGEN_EXE luthier-tblgen)
set(LLVM_TARGET_DEFINITIONS "${CMAKE_SOURCE_DIR}/include/luthier/InstSemantics/InstSISemantics.td")
tablegen(LuthierSIInstrSemanticsGen SIInstrSemantics.inc -gen-si-instr-semantics EXTRA_INCLUDES
        "${LUTHIER_LLVM_SRC_DIR}/llvm/lib/Target/AMDGPU/;${LUTHIER_LLVM_SRC_DIR}/llvm/include/;"
        "${CMAKE_SOURCE_DIR}/include/luthier/InstSemantics/")
add_public_tablegen_target(LuthierSIInstrSemanticsGen)

add_library(LuthierToolingCommon OBJECT
        TargetMachineInstrMDNode.cpp
        CodeDiscoveryPass.cpp
        MachineFunctionEntryPoint.cpp
        InstructionTracesAnalysis.cpp
        MemoryAllocationAccessor.cpp
        PhysicalRegAccessVirtualizationPass.cpp
        #        SVStorageAndLoadLocations.cpp
        #        LRCallGraph.cpp
        IPPredicatedLiveness.cpp
        IntrinsicMIRLoweringPass.cpp
        #        InjectedPayloadPEIPass.cpp
        #        PrePostAmbleEmitter.cpp
        StateValueArraySpecs.cpp
        IPPredicatedCFG.cpp
        #        StateValueArrayStorage.cpp
        #        IModuleIRGeneratorPass.cpp
        #        RunIRPassesOnIModulePass.cpp
        #        MMISlotIndexesAnalysis.cpp
        ProcessIntrinsicsAtIRLevelPass.cpp
        #        PhysRegsNotInLiveInsAnalysis.cpp
        WrapperAnalysisPasses.cpp
        #        RunMIRPassesOnIModulePass.cpp
        #        PatchLiftedRepresentationPass.cpp
        MIRConvenience.cpp
        MockAMDGPULoader.cpp
        MetadataParserAnalysis.cpp
        IntrinsicProcessorRegistry.cpp
        IntrinsicProcessorsAnalysis.cpp
        InstrumentationPMDriver.cpp
        MockLoadAMDGPUCodeObjects.cpp
        CodeObjectManagerAnalysis.cpp
        InitialEntryPointAnalysis.cpp
        AMDGPUMockLoaderPrinter.cpp
        MockLoaderMemoryAccessor.cpp
        Metadata.cpp
        MIRToIRTranslator.cpp
        NewPMAsmPrinter.cpp
        FunctionAnnotations.cpp
        IPReachingRefAnalysis.cpp
        IPPredicatedLoopTraversal.cpp
        #        IPRegScavenger.cpp
        IndirectBranchResolverAnalysis.cpp
        PredicatedMachineBasicBlock.cpp
        LinearMachineBasicBlock.cpp
        PredicatedMachineFunction.cpp

)

#add_dependencies(LuthierToolingCommon LuthierRealToPseudoOpcodeMap)
#add_dependencies(LuthierToolingCommon LuthierRealToPseudoRegEnumMap)

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