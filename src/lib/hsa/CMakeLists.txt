get_target_property(HSA_INCLUDE_DIRS hsa-runtime64::hsa-runtime64 INTERFACE_INCLUDE_DIRECTORIES)

if (${LUTHIER_GEN_INTERCEPTOR_IMPL_DURING_CONFIG})
    execute_process(COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/hsa_intercept_gen.py
            --hsa-include-dir ${HSA_INCLUDE_DIRS}/hsa
            --cpp-implementation-save-path ${CMAKE_CURRENT_BINARY_DIR}/HsaRuntimeInterceptor.cpp
            --h-api-enums-structs-save-path ${CMAKE_BINARY_DIR}/include/luthier/hsa/TraceApi.h
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/scripts)
endif ()

add_custom_command(OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/HsaRuntimeInterceptor.cpp
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/hsa_intercept_gen.py
        --hsa-include-dir ${HSA_INCLUDE_DIRS}/hsa
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

target_include_directories(LuthierHSA PUBLIC "${CMAKE_SOURCE_DIR}/src/include")

target_include_directories(LuthierHSA PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

luthier_add_position_independence(LuthierHSA)
luthier_add_hsa_compile_definitions(LuthierHSA)
luthier_add_llvm_compile_definitions(LuthierHSA)

target_link_libraries(LuthierHSA PUBLIC hsa-runtime64::hsa-runtime64 LuthierAMDGPU LuthierCommon)