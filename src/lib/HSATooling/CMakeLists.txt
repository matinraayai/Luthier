add_library(LuthierHSATooling OBJECT
        DeviceToolCodeParser.cpp
        HSAToolCompileCheck.cpp
        HostcallConsumer.cpp
        HsaMemoryAllocationAccessor.cpp
        InstrumentedKernelLoaderAndLauncher.cpp
        LoadedCodeObjectCache.cpp
)

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

# Optional SPIR-V -> AMDGCN JIT fallback in DeviceToolCodeParser: when no
# precompiled bitcode slice matches the requested ISA, translate the bundle's
# SPIR-V slice and run the Luthier device tool passes on it. Compiled only when
# the AMD SPIR-V translator was located (FindLLVMSPIRVTranslator). The translator
# is linked PUBLIC so the final shared objects / executables that consume
# LuthierHSATooling pull it in; the device passes live in LuthierToolIRCompilation
# (already linked by those consumers).
if (NOT DEFINED LUTHIER_LLVM_SPIRV_TRANSLATOR_FOUND)
    find_package(LLVMSPIRVTranslator QUIET)
endif ()
if (LUTHIER_LLVM_SPIRV_TRANSLATOR_FOUND)
    target_compile_definitions(LuthierHSATooling PRIVATE LUTHIER_HAS_SPIRV_TRANSLATOR)
    target_link_libraries(LuthierHSATooling PUBLIC Luthier::LLVMSPIRVTranslator)
endif ()
