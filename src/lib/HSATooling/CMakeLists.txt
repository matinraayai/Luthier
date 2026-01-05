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

add_library(LuthierHSATooling OBJECT
        CodeGenerator.cpp
        CodeLifter.cpp
        Context.cpp
        InstrumentationModule.cpp
        InstrumentationTask.cpp
        TargetManager.cpp
        ToolExecutableLoader.cpp
)


add_dependencies(LuthierHSATooling LuthierRealToPseudoOpcodeMap)
add_dependencies(LuthierHSATooling LuthierRealToPseudoRegEnumMap)

target_compile_definitions(LuthierHSATooling PRIVATE AMD_INTERNAL_BUILD ${LLVM_DEFINITIONS})

target_include_directories(LuthierHSATooling
        PRIVATE
        ${CMAKE_CURRENT_BINARY_DIR}
        ${LLVM_INCLUDE_DIRS}
        ${hsa-runtime64_INCLUDE_DIRS}
        PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>"
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

target_link_libraries(LuthierHSATooling PRIVATE LuthierAMDGPU hip::host)