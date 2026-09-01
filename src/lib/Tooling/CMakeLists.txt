add_library(LuthierTooling SHARED
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
        $<TARGET_OBJECTS:LuthierHSATooling>
        $<TARGET_OBJECTS:LuthierLinker>
        # The KFD module's LLVM-facing half: the MemoryAllocationAccessor over
        # allocations observed at the ioctl boundary. Only this half -- LuthierKFD
        # itself is the preloadable wrapper's logic and must stay out of here, or
        # linking Luthier would drag in an ioctl interceptor.
        $<TARGET_OBJECTS:LuthierKFDTooling>
)

target_compile_definitions(LuthierTooling PUBLIC AMD_INTERNAL_BUILD ${LLVM_DEFINITIONS})

target_include_directories(LuthierTooling
        PUBLIC
        ${LLVM_INCLUDE_DIRS}
        "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>"
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>"
        "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>"
        "$<BUILD_INTERFACE:${AMDGPU_INCLUDE_DIR}>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/luthier/AMDGPU>"
)

target_link_libraries(LuthierTooling
        PUBLIC hip::host rocprofiler-sdk::rocprofiler-sdk hsa-runtime64::hsa-runtime64 LuthierAMDGPU
        LLVMAMDGPUAsmParser LLVMAMDGPUDesc LLVMAMDGPUDisassembler LLVMAMDGPUInfo LLVMAMDGPUTargetMCA
        LLVMAMDGPUUtils LLVMAMDGPUCodeGen
        PRIVATE
        LLVMSymbolize LLVMDebugInfoDWARF LLVMCodeGenTypes LLVMMC LLVMMCA LLVMMCDisassembler LLVMObject
        LLVMSupport LLVMTarget LLVMTargetParser LLVMTransformUtils LLVMBitReader LLVMAnalysis LLVMAsmPrinter
        LLVMCore LLVMPasses LLVMCodeGen LLVMLinker LLVMIRReader
        # LuthierKFDTooling resolves the KFD tracker's exports with dlsym, so that a
        # tool reads the map the preloaded wrapper actually filled rather than a
        # second, permanently empty one of its own.
        ${CMAKE_DL_LIBS})

if (${LUTHIER_ENABLE_SPIRV})
    target_link_libraries(LuthierTooling PRIVATE Luthier::LLVMSPIRVTranslator)
endif ()
