# Add target to generate real to pseudo opcode and register map and functions to query them
set(LuthierToolingCommon_TABLEGEN_EXE luthier-tblgen)
set(LLVM_TARGET_DEFINITIONS "${LUTHIER_TOOLING_LLVM_SRC_DIR}/llvm/lib/Target/AMDGPU/AMDGPU.td")
tablegen(LuthierToolingCommon LuthierRealToPseudoOpcodeMap.hpp -gen-si-real-to-pseudo-opcode-map EXTRA_INCLUDES
        "${LUTHIER_TOOLING_LLVM_SRC_DIR}/llvm/lib/Target/AMDGPU/;${LUTHIER_TOOLING_LLVM_SRC_DIR}/llvm/include/")
add_public_tablegen_target(LuthierRealToPseudoOpcodeMap)

set(LuthierToolingCommon_TABLEGEN_EXE luthier-tblgen)
set(LLVM_TARGET_DEFINITIONS "${LUTHIER_TOOLING_LLVM_SRC_DIR}/llvm/lib/Target/AMDGPU/AMDGPU.td")
tablegen(LuthierToolingCommon LuthierRealToPseudoRegEnumMap.hpp -gen-si-real-to-pseudo-reg-map EXTRA_INCLUDES
        "${LUTHIER_TOOLING_LLVM_SRC_DIR}/llvm/lib/Target/AMDGPU/;${LUTHIER_TOOLING_LLVM_SRC_DIR}/llvm/include/")
add_public_tablegen_target(LuthierRealToPseudoRegEnumMap)

add_library(LuthierToolingCommon OBJECT
        CodeGenerator.cpp
        CodeLifter.cpp
        InstrumentationTask.cpp
        luthier.cpp
        TargetManager.cpp
        ToolExecutableLoader.cpp
        LiftedRepresentation.cpp
        InstrumentationModule.cpp
        intrinsic/ReadReg.cpp
        intrinsic/WriteReg.cpp
        PhysicalRegAccessVirtualizationPass.cpp
        SVStorageAndLoadLocations.cpp
        LRCallGraph.cpp
        LRRegisterLiveness.cpp
        IntrinsicMIRLoweringPass.cpp
        InjectedPayloadPEIPass.cpp
        PrePostAmbleEmitter.cpp
        intrinsic/WriteExec.cpp
        intrinsic/IntrinsicProcessor.cpp
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
        intrinsic/ImplicitArgPtr.cpp
        intrinsic/SAtomicAdd.cpp
)

# Add explicit dependency between the tablegen target and the tooling common target to force cmake to run
# tablegen-ed target first
add_dependencies(LuthierToolingCommon LuthierRealToPseudoOpcodeMap)
add_dependencies(LuthierToolingCommon LuthierRealToPseudoRegEnumMap)
add_dependencies(LuthierToolingCommon LuthierAMDGPUTableGen)

target_include_directories(LuthierToolingCommon PUBLIC "${CMAKE_SOURCE_DIR}/src/include")

target_include_directories(LuthierToolingCommon PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

# Add the binary folder to the include directory to use LuthierRealToPseudoOpcodeMap.inc and
# LuthierRealToPseudoRegEnumMap.inc
target_include_directories(LuthierToolingCommon PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

luthier_add_position_independence(LuthierToolingCommon)
luthier_add_hsa_compile_definitions(LuthierToolingCommon)
luthier_add_llvm_compile_definitions(LuthierToolingCommon)

target_link_libraries(LuthierToolingCommon PRIVATE hip::host amd_comgr luthier::Intrinsics LuthierAMDGPU)