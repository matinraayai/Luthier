add_library(LuthierExpr OBJECT
        MachineIRExpr.cpp
)

target_compile_definitions(LuthierExpr PRIVATE ${LLVM_DEFINITIONS})

target_include_directories(LuthierExpr
        PRIVATE
        "${CMAKE_SOURCE_DIR}/src/include"
        ${LLVM_INCLUDE_DIRS}
        PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>"
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

target_link_libraries(LuthierExpr PRIVATE LuthierAMDGPU)