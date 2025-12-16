set(AMDGPU_INCLUDE_DIR ${CMAKE_BINARY_DIR}/include/luthier/AMDGPU/)
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

# Create a target for all the internal amdgpu headers
add_library(LuthierAMDGPU INTERFACE)

target_include_directories(LuthierAMDGPU INTERFACE
        "$<BUILD_INTERFACE:${AMDGPU_INCLUDE_DIR}>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/luthier/AMDGPU>"
)
