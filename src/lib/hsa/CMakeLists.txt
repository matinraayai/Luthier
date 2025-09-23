add_library(LuthierHSA OBJECT
        Agent.cpp
        CodeObjectReader.cpp
        Executable.cpp
        ExecutableSymbol.cpp
        hsa.cpp
        HsaError.cpp
        Instr.cpp
        ISA.cpp
        KernelDescriptor.cpp
        LoadedCodeObjectCache.cpp
        LoadedCodeObjectDeviceFunction.cpp
        LoadedCodeObjectExternSymbol.cpp
        LoadedCodeObjectKernel.cpp
        LoadedCodeObjectSymbol.cpp
        LoadedCodeObjectVariable.cpp
        Metadata.cpp
        PacketMonitor.cpp
)

target_compile_definitions(LuthierHSA PRIVATE AMD_INTERNAL_BUILD ${LLVM_DEFINITIONS})

target_include_directories(LuthierHSA
        PRIVATE
        "${CMAKE_SOURCE_DIR}/src/include"
        ${LLVM_INCLUDE_DIRS}
        ${hsa-runtime64_INCLUDE_DIRS}
        PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")

target_link_libraries(LuthierHSA PRIVATE LuthierAMDGPU hip::host)