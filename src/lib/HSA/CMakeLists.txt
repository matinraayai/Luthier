add_library(LuthierHSA OBJECT
        Agent.cpp
        CodeObjectReader.cpp
        Executable.cpp
        ExecutableSymbol.cpp
        hsa.cpp
        HsaError.cpp
        ISA.cpp
        Queue.cpp
        Region.cpp
        Memory.cpp
        MemoryPool.cpp
        Signal.cpp
        SVM.cpp
        VMEM.cpp
)

target_compile_definitions(LuthierHSA PRIVATE AMD_INTERNAL_BUILD ${LLVM_DEFINITIONS})

target_include_directories(LuthierHSA
        PRIVATE
        ${LLVM_INCLUDE_DIRS}
        ${hsa-runtime64_INCLUDE_DIRS}
        PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")

target_link_libraries(LuthierHSA PRIVATE LuthierAMDGPU hip::host)