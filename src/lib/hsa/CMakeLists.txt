add_library(LuthierHSA OBJECT
        Agent.cpp
        ApiTable.cpp
        CodeObjectCache.cpp
        CodeObjectReader.cpp
        Executable.cpp
        ExecutableSymbol.cpp
        hsa.cpp
        HsaError.cpp
        HsaPacketMonitor.cpp
        ISA.cpp
        KernelDescriptor.cpp
        LoadedCodeObject.cpp
        LoadedInstrumentationModule.cpp
        ToolExecutableLoader.cpp
)

target_compile_definitions(LuthierHSA PRIVATE AMD_INTERNAL_BUILD ${LLVM_DEFINITIONS} ${hip_DEFINES})

target_include_directories(LuthierHSA
        PRIVATE
        ${LLVM_INCLUDE_DIRS}
        ${hsa-runtime64_INCLUDE_DIRS}
        ${hip_INCLUDE_DIRS}
        PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")