if (${LUTHIER_GEN_INTERCEPTOR_IMPL_DURING_CONFIG})
    execute_process(COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/hsa_intercept_gen.py
            --hsa-include-dir ${hsa-runtime64_INCLUDE_DIRS}/hsa
            --cpp-implementation-save-path ${CMAKE_CURRENT_BINARY_DIR}/HsaRuntimeInterceptor.cpp
            --h-api-enums-structs-save-path ${CMAKE_BINARY_DIR}/include/luthier/hsa/TraceApi.h
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/scripts)
endif ()

add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/HsaRuntimeInterceptor.cpp
        ${CMAKE_BINARY_DIR}/include/luthier/hsa/TraceApi.h
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/hsa_intercept_gen.py
        --hsa-include-dir ${hsa-runtime64_INCLUDE_DIRS}/hsa
        --cpp-implementation-save-path ${CMAKE_CURRENT_BINARY_DIR}/HsaRuntimeInterceptor.cpp
        --h-api-enums-structs-save-path ${CMAKE_BINARY_DIR}/include/luthier/hsa/TraceApi.h
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/scripts
        COMMENT "Generating HSA API interceptor implementation")

add_library(LuthierHSA OBJECT
        ${CMAKE_CURRENT_BINARY_DIR}/HsaRuntimeInterceptor.cpp
        Executable.cpp
        GpuAgent.cpp
        ExecutableSymbol.cpp
        LoadedCodeObject.cpp
        HsaError.cpp
        ISA.cpp
        Instr.cpp
        CodeObjectReader.cpp
        KernelDescriptor.cpp
        hsa.cpp
        ExecutableBackedObjectsCache.cpp
        LoadedCodeObjectSymbol.cpp
        LoadedCodeObjectKernel.cpp
        LoadedCodeObjectVariable.cpp
        LoadedCodeObjectDeviceFunction.cpp
        LoadedCodeObjectExternSymbol.cpp
        DisableInterceptionScope.cpp
)

target_compile_definitions(LuthierHSA PRIVATE AMD_INTERNAL_BUILD ${LLVM_DEFINITIONS})

target_include_directories(LuthierHSA
        PRIVATE
        "${CMAKE_SOURCE_DIR}/src/include"
        ${LLVM_INCLUDE_DIRS}
        ${hsa-runtime64_INCLUDE_DIRS}
        PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>"
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")

target_link_libraries(LuthierHSA PRIVATE LuthierAMDGPU hip::host)