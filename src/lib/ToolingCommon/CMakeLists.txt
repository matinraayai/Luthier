add_library(LuthierToolingCommon OBJECT
        AnnotatedMachineInstr.cpp
        CodeDiscoveryPass.cpp
        CodeGenerator.cpp
        CodeLifter.cpp
        MachineFunctionEntryPoints.cpp
        MachineToTraceInstrAnalysis.cpp
        InstructionTracesAnalysis.cpp
        MemoryAllocationAccessor.cpp
        InstrumentationTask.cpp
        InstructionModel.cpp
        TargetManager.cpp
        ToolExecutableLoader.cpp
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
        Context.cpp
        luthier.cpp
        IntrinsicProcessorRegistry.cpp
        IntrinsicProcessorsAnalysis.cpp
        InstrumentationPMDriver.cpp
        MockLoadAMDGPUCodeObjects.cpp
        CodeObjectManagerAnalysis.cpp
)

target_compile_definitions(LuthierToolingCommon PRIVATE AMD_INTERNAL_BUILD ${LLVM_DEFINITIONS})

target_include_directories(LuthierToolingCommon
        PRIVATE
        ${LLVM_INCLUDE_DIRS}
        ${hsa-runtime64_INCLUDE_DIRS}
        PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>"
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

target_link_libraries(LuthierToolingCommon PRIVATE LuthierAMDGPU hip::host)