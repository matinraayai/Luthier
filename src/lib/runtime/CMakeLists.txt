add_library(LuthierRuntime OBJECT
        HsaObjectCache.cpp
        LoadedInstrumentationModule.cpp
        ToolExecutableLoader.cpp
)

target_compile_definitions(LuthierRuntime PRIVATE AMD_INTERNAL_BUILD ${LLVM_DEFINITIONS})

target_include_directories(LuthierRuntime
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

target_link_libraries(LuthierRuntime PRIVATE LuthierAMDGPU)