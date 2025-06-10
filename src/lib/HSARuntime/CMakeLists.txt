add_library(LuthierHSARuntime OBJECT
        CodeObjectCache.cpp
        PacketMonitor.cpp
        LoadedInstrumentationModule.cpp
        ToolExecutableLoader.cpp
)

target_compile_definitions(LuthierHSARuntime PRIVATE AMD_INTERNAL_BUILD ${LLVM_DEFINITIONS} ${hip_DEFINES})

target_include_directories(LuthierHSARuntime
        PRIVATE
        ${LLVM_INCLUDE_DIRS}
        ${hsa-runtime64_INCLUDE_DIRS}
        ${hip_INCLUDE_DIRS}
        PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")