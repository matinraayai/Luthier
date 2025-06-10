
add_subdirectory(AMDGPU)

add_library(LuthierInstrumentation OBJECT
        CodeGenerator.cpp
        Disassembler.cpp
        ELFObjectFileAnalysisPass.cpp
        ELFRelocationResolverAnalysisPass.cpp
        IModuleIRGeneratorPass.cpp
        InstrumentationContext.cpp
        InstrumentationModule.cpp
        InstrumentationTask.cpp
        IntrinsicMIRLoweringPass.cpp
        IntrinsicProcessor.cpp
        LLVMManager.cpp
        LRCallGraph.cpp
        MMISlotIndexesAnalysis.cpp
        PatchLiftedRepresentationPass.cpp
        PhysicalRegAccessVirtualizationPass.cpp
        PhysRegsNotInLiveInsAnalysis.cpp
        ProcessIntrinsicsAtIRLevelPass.cpp
        RunIRPassesOnIModulePass.cpp
        RunMIRPassesOnIModulePass.cpp
        TargetManager.cpp
        WrapperAnalysisPasses.cpp
)

target_compile_definitions(LuthierInstrumentation PRIVATE ${LLVM_DEFINITIONS})

target_include_directories(LuthierInstrumentation
        PRIVATE
        ${CMAKE_CURRENT_BINARY_DIR}
        ${LLVM_INCLUDE_DIRS}
        PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>"
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

target_link_libraries(LuthierInstrumentation PUBLIC LuthierInstrumentationAMDGPU)