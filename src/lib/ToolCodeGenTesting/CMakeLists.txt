add_library(LuthierToolCodeGenTesting OBJECT
        MockAMDGPULoader.cpp
        MockLoadAMDGPUCodeObjects.cpp
        AMDGPUMockLoaderPrinter.cpp
        MockLoaderMemoryAccessor.cpp
        CodeObjectManagerAnalysis.cpp
        LuthierFile.cpp)

target_compile_definitions(LuthierToolCodeGenTesting
        PRIVATE AMD_INTERNAL_BUILD ${LLVM_DEFINITIONS})

target_include_directories(LuthierToolCodeGenTesting
        PRIVATE
        ${LLVM_INCLUDE_DIRS}
        ${hsa-runtime64_INCLUDE_DIRS}
        PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
