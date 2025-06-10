set(BIN_DIR ${CMAKE_CURRENT_BINARY_DIR})
# Generate AMDGPU tablegen headers + copy over the LLVM AMDGPU backend headers into Luthier's headers
set(AMDGPU_INCLUDE_DIR ${CMAKE_BINARY_DIR}/include/luthier/amdgpu/)
# workaround to make the tablegen generate its output files under the ${AMDGPU_INCLUDE_DIR} dir
set(CMAKE_CURRENT_BINARY_DIR ${AMDGPU_INCLUDE_DIR})
# Find the llvm-tblgen tool executable
find_program(LLVM_TABLEGEN_EXE "llvm-tblgen" ${LLVM_TOOLS_BINARY_DIR} NO_DEFAULT_PATH)
# Re-generates the required AMDGPU tablegen files and puts them under ${AMDGPU_INCLUDE_DIR}, which
# is also accessible publicly to tool writers
set(LuthierToolingCommon_TABLEGEN_EXE ${LLVM_TABLEGEN_EXE})
set(LLVM_TARGET_DEFINITIONS "${LUTHIER_LLVM_SRC_DIR}/llvm/lib/Target/AMDGPU/AMDGPU.td")
tablegen(LuthierToolingCommon AMDGPUGenInstrInfo.inc -gen-instr-info EXTRA_INCLUDES
        "${LUTHIER_LLVM_SRC_DIR}/llvm/lib/Target/AMDGPU/;${LUTHIER_LLVM_SRC_DIR}/llvm/include/")
tablegen(LuthierToolingCommon AMDGPUGenRegisterBank.inc -gen-register-bank EXTRA_INCLUDES
        "${LUTHIER_LLVM_SRC_DIR}/llvm/lib/Target/AMDGPU/;${LUTHIER_LLVM_SRC_DIR}/llvm/include/")
tablegen(LuthierToolingCommon AMDGPUGenRegisterInfo.inc -gen-register-info EXTRA_INCLUDES
        "${LUTHIER_LLVM_SRC_DIR}/llvm/lib/Target/AMDGPU/;${LUTHIER_LLVM_SRC_DIR}/llvm/include/")
tablegen(LuthierToolingCommon AMDGPUGenSearchableTables.inc -gen-searchable-tables EXTRA_INCLUDES
        "${LUTHIER_LLVM_SRC_DIR}/llvm/lib/Target/AMDGPU/;${LUTHIER_LLVM_SRC_DIR}/llvm/include/")
tablegen(LuthierToolingCommon AMDGPUGenSubtargetInfo.inc -gen-subtarget EXTRA_INCLUDES
        "${LUTHIER_LLVM_SRC_DIR}/llvm/lib/Target/AMDGPU/;${LUTHIER_LLVM_SRC_DIR}/llvm/include/")
add_public_tablegen_target(LuthierAMDGPUTableGen)


# Copy all the header files under llvm/lib/Target/AMDGPU under ${AMDGPU_INCLUDE_DIR}
file(COPY ${LUTHIER_LLVM_SRC_DIR}/llvm/lib/Target/AMDGPU/ DESTINATION ${AMDGPU_INCLUDE_DIR} FILES_MATCHING
        PATTERN "*.h")

# Reset back the current binary directory to its original value
set(CMAKE_CURRENT_BINARY_DIR ${BIN_DIR})

# Add target to generate real to pseudo opcode and register map and functions to query them inside the code lifter
# passes
set(LuthierInstrumentationAMDGPU_TABLEGEN_EXE luthier-tblgen)
set(LLVM_TARGET_DEFINITIONS "${LUTHIER_LLVM_SRC_DIR}/llvm/lib/Target/AMDGPU/AMDGPU.td")
tablegen(LuthierInstrumentationAMDGPU LuthierRealToPseudoOpcodeMap.hpp -gen-si-real-to-pseudo-opcode-map EXTRA_INCLUDES
        "${LUTHIER_LLVM_SRC_DIR}/llvm/lib/Target/AMDGPU/;${LUTHIER_LLVM_SRC_DIR}/llvm/include/")
add_public_tablegen_target(LuthierRealToPseudoOpcodeMap)

set(LuthierInstrumentationAMDGPU_TABLEGEN_EXE luthier-tblgen)
set(LLVM_TARGET_DEFINITIONS "${LUTHIER_LLVM_SRC_DIR}/llvm/lib/Target/AMDGPU/AMDGPU.td")
tablegen(LuthierInstrumentationAMDGPU LuthierRealToPseudoRegEnumMap.hpp -gen-si-real-to-pseudo-reg-map EXTRA_INCLUDES
        "${LUTHIER_LLVM_SRC_DIR}/llvm/lib/Target/AMDGPU/;${LUTHIER_LLVM_SRC_DIR}/llvm/include/")
add_public_tablegen_target(LuthierRealToPseudoRegEnumMap)


add_library(LuthierInstrumentationAMDGPU OBJECT
        Intrinsics/ImplicitArgPtr.cpp
        Intrinsics/ReadReg.cpp
        Intrinsics/SAtomicAdd.cpp
        Intrinsics/WriteExec.cpp
        Intrinsics/WriteReg.cpp
        AMDGPURegisterLiveness.cpp
        CodeGenHelpers.cpp
        CodeLifter.cpp
        InjectedPayloadPEIPass.cpp
        Metadata.cpp
        MIRConvenience.cpp
        PrePostAmbleEmitter.cpp
        StateValueArraySpecs.cpp
        StateValueArrayStorage.cpp
        SVStorageAndLoadLocations.cpp
        VectorCFG.cpp
)

add_dependencies(LuthierInstrumentationAMDGPU LuthierRealToPseudoOpcodeMap)
add_dependencies(LuthierInstrumentationAMDGPU LuthierRealToPseudoRegEnumMap)
add_dependencies(LuthierInstrumentationAMDGPU LuthierAMDGPUTableGen)

target_compile_definitions(LuthierInstrumentationAMDGPU PRIVATE ${LLVM_DEFINITIONS})

target_include_directories(LuthierInstrumentationAMDGPU
        PRIVATE
        ${CMAKE_CURRENT_BINARY_DIR}
        ${LLVM_INCLUDE_DIRS}
        PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>"
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
        "$<BUILD_INTERFACE:${AMDGPU_INCLUDE_DIR}>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/luthier/amdgpu>"
)