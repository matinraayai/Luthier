add_library(LuthierHSATooling OBJECT
        CodeGenerator.cpp
        CodeLifter.cpp
        Context.cpp
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