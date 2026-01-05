set(AMDGPU_INCLUDE_DIR ${CMAKE_BINARY_DIR}/include/luthier/AMDGPU/)

# Copy all the header files under llvm/lib/Target/AMDGPU under ${AMDGPU_INCLUDE_DIR}
file(COPY ${LUTHIER_LLVM_SRC_DIR}/llvm/lib/Target/AMDGPU/ DESTINATION ${AMDGPU_INCLUDE_DIR} FILES_MATCHING
        PATTERN "*.h")

# Find the llvm-tblgen tool executable
find_program(LLVM_TABLEGEN_EXE "llvm-tblgen" ${LLVM_TOOLS_BINARY_DIR} NO_DEFAULT_PATH)
# Re-generates the required AMDGPU tablegen files and puts them under ${AMDGPU_INCLUDE_DIR}, which
# is also accessible publicly to tool writers
set(LuthierAMDGPUTableGen_TABLEGEN_EXE ${LLVM_TABLEGEN_EXE})
set(LLVM_TARGET_DEFINITIONS "${LUTHIER_LLVM_SRC_DIR}/llvm/lib/Target/AMDGPU/AMDGPU.td")
tablegen(LuthierAMDGPUTableGen AMDGPUGenInstrInfo.inc -gen-instr-info EXTRA_INCLUDES
        "${LUTHIER_LLVM_SRC_DIR}/llvm/lib/Target/AMDGPU/;${LUTHIER_LLVM_SRC_DIR}/llvm/include/")
tablegen(LuthierAMDGPUTableGen AMDGPUGenRegisterBank.inc -gen-register-bank EXTRA_INCLUDES
        "${LUTHIER_LLVM_SRC_DIR}/llvm/lib/Target/AMDGPU/;${LUTHIER_LLVM_SRC_DIR}/llvm/include/")
tablegen(LuthierAMDGPUTableGen AMDGPUGenRegisterInfo.inc -gen-register-info EXTRA_INCLUDES
        "${LUTHIER_LLVM_SRC_DIR}/llvm/lib/Target/AMDGPU/;${LUTHIER_LLVM_SRC_DIR}/llvm/include/")
tablegen(LuthierAMDGPUTableGen AMDGPUGenSearchableTables.inc -gen-searchable-tables EXTRA_INCLUDES
        "${LUTHIER_LLVM_SRC_DIR}/llvm/lib/Target/AMDGPU/;${LUTHIER_LLVM_SRC_DIR}/llvm/include/")
tablegen(LuthierAMDGPUTableGen AMDGPUGenSubtargetInfo.inc -gen-subtarget EXTRA_INCLUDES
        "${LUTHIER_LLVM_SRC_DIR}/llvm/lib/Target/AMDGPU/;${LUTHIER_LLVM_SRC_DIR}/llvm/include/")

#add_public_tablegen_target(LuthierAMDGPUTableGen)

# Create a target for all the internal amdgpu headers
add_library(LuthierAMDGPU INTERFACE ${TABLEGEN_OUTPUT})

#add_dependencies(LuthierAMDGPU LuthierAMDGPUTableGen)

target_include_directories(LuthierAMDGPU INTERFACE
        "$<BUILD_INTERFACE:${AMDGPU_INCLUDE_DIR}>"
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/luthier/AMDGPU>"
)
