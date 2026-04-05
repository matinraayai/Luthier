# Add target to generate real to pseudo opcode and register map and functions to query them
set(LuthierIRTranslator_TABLEGEN_EXE luthier-tblgen)
set(LLVM_TARGET_DEFINITIONS "${LUTHIER_LLVM_SRC_DIR}/llvm/lib/Target/AMDGPU/AMDGPU.td")
tablegen(LuthierIRTranslator LuthierRealToPseudoOpcodeMap.hpp -gen-si-real-to-pseudo-opcode-map EXTRA_INCLUDES
        "${LUTHIER_LLVM_SRC_DIR}/llvm/lib/Target/AMDGPU/;${LUTHIER_LLVM_SRC_DIR}/llvm/include/")
add_public_tablegen_target(LuthierRealToPseudoOpcodeMap)

set(LuthierIRTranslator_TABLEGEN_EXE luthier-tblgen)
set(LLVM_TARGET_DEFINITIONS "${LUTHIER_LLVM_SRC_DIR}/llvm/lib/Target/AMDGPU/AMDGPU.td")
tablegen(LuthierIRTranslator LuthierRealToPseudoRegEnumMap.hpp -gen-si-real-to-pseudo-reg-map EXTRA_INCLUDES
        "${LUTHIER_LLVM_SRC_DIR}/llvm/lib/Target/AMDGPU/;${LUTHIER_LLVM_SRC_DIR}/llvm/include/")
add_public_tablegen_target(LuthierRealToPseudoRegEnumMap)


add_library(LuthierIRTranslator OBJECT
        PseudoOpcodeAndRegMapper.cpp
)

add_dependencies(LuthierIRTranslator LuthierRealToPseudoOpcodeMap)
add_dependencies(LuthierIRTranslator LuthierRealToPseudoRegEnumMap)
add_dependencies(LuthierIRTranslator LuthierSIInstrSemanticsGen)

target_compile_definitions(LuthierIRTranslator PRIVATE ${LLVM_DEFINITIONS})

target_include_directories(LuthierIRTranslator
        PRIVATE
        ${LLVM_INCLUDE_DIRS}
        ${CMAKE_CURRENT_BINARY_DIR}
        PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>"
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)
target_link_libraries(LuthierIRTranslator PRIVATE LuthierAMDGPU)