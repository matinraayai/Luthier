
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
        CodeGenerator.cpp
        CodeLifter.cpp
        InstrumentationTask.cpp
        TargetManager.cpp
        ToolExecutableLoader.cpp
        LiftedRepresentation.cpp
        InstrumentationModule.cpp
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
)

# Add explicit dependency between the tablegen target and the tooling common target to force cmake to run
# tablegen-ed target first
add_dependencies(LuthierToolingCommon LuthierRealToPseudoOpcodeMap)
add_dependencies(LuthierToolingCommon LuthierRealToPseudoRegEnumMap)
add_dependencies(LuthierToolingCommon LuthierAMDGPUTableGen)

#Added External Project Dep
add_dependencies(LuthierToolingCommon LLVMSRC)

target_compile_definitions(LuthierToolingCommon PRIVATE AMD_INTERNAL_BUILD ${LLVM_DEFINITIONS})

target_include_directories(LuthierToolingCommon
        PRIVATE
        "${CMAKE_SOURCE_DIR}/src/include"
        ${CMAKE_CURRENT_BINARY_DIR}
        ${LLVM_INCLUDE_DIRS}
        ${hsa-runtime64_INCLUDE_DIRS}
        PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>"
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

target_link_libraries(LuthierToolingCommon PRIVATE LuthierAMDGPU)