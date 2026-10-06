add_library(LuthierTooling SHARED
        DevicePrintf.cpp
        HiddenArgBuffers.cpp
        HostcallHandler.cpp
        HsaMemoryAllocationAccessor.cpp
        InstrumentedKernelLoaderAndLauncher.cpp
        LoadedCodeObjectCache.cpp
        $<TARGET_OBJECTS:LuthierObject>
        $<TARGET_OBJECTS:LuthierRocprofilerSDK>
        $<TARGET_OBJECTS:LuthierHSA>
        $<TARGET_OBJECTS:LuthierHIP>
        $<TARGET_OBJECTS:LuthierLLVM>
        $<TARGET_OBJECTS:LuthierIntrinsic>
        $<TARGET_OBJECTS:LuthierCommon>
        $<TARGET_OBJECTS:LuthierToolCodeGen>
        $<TARGET_OBJECTS:LuthierToolIRCompilation>
        $<TARGET_OBJECTS:LuthierPassPlugin>
        $<TARGET_OBJECTS:LuthierLinker>
        $<TARGET_OBJECTS:LuthierPatchPCUsagesPass>
        $<TARGET_OBJECTS:LuthierRebaseAppScratchAccessesPass>
)

target_compile_definitions(LuthierTooling PUBLIC AMD_INTERNAL_BUILD
        ${LLVM_DEFINITIONS} ${hip_COMPILE_DEFINITIONS})

target_include_directories(LuthierTooling
        PRIVATE
        ${CMAKE_CURRENT_BINARY_DIR}
        PUBLIC
        ${hsa-runtime64_INCLUDE_DIRS}
        ${LLVM_INCLUDE_DIRS}
        ${hip_INCLUDE_DIRS}
        ${rocprofiler-sdk_INCLUDE_DIRS}
        "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>"
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>"
        "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>"
        "$<BUILD_INTERFACE:${AMDGPU_INCLUDE_DIR}>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/luthier/AMDGPU>"
)

target_link_libraries(LuthierTooling
        PUBLIC LuthierAMDGPU ${CMAKE_DL_LIBS}
        LLVMAMDGPUAsmParser LLVMAMDGPUDesc LLVMAMDGPUDisassembler LLVMAMDGPUInfo LLVMAMDGPUTargetMCA
        LLVMAMDGPUUtils LLVMAMDGPUCodeGen
        PRIVATE
        Threads::Threads
        LLVMSymbolize LLVMDebugInfoDWARF LLVMCodeGenTypes LLVMMC LLVMMCA LLVMMCDisassembler LLVMObject
        LLVMSupport LLVMTarget LLVMTargetParser LLVMTransformUtils LLVMBitReader LLVMAnalysis LLVMAsmPrinter
        LLVMCore LLVMPasses LLVMCodeGen LLVMLinker LLVMIRReader)

if (${LUTHIER_ENABLE_SPIRV})
    target_link_libraries(LuthierTooling PRIVATE Luthier::LLVMSPIRVTranslator)
endif ()

add_library(luthier::LuthierTooling ALIAS LuthierTooling)
