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
        $<TARGET_OBJECTS:LuthierPatchPCUsagesPass>
        $<TARGET_OBJECTS:LuthierRebaseAppScratchAccessesPass>
        # Both halves of the KFD module. LuthierKFD used to be excluded, because
        # it carried an ioctl interceptor that linking Luthier would then have
        # installed process-wide. It no longer does: interception is LD_AUDIT, so
        # nothing is installed until the tool is named in AH_PLUGINS and the
        # application is started under LD_AUDIT. What is left here is the
        # allocation map and the DRM descriptor sharing, which PacketMonitorTrait
        # calls directly.
        $<TARGET_OBJECTS:LuthierKFD>
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
        # PUBLIC, because PacketMonitorTrait.h includes the audit_hook header and
        # every tool includes that header: a Luthier tool *is* an audit_hook
        # plugin, and ah_register_hook lives in this library. Loaded out of
        # AH_PLUGINS into the auditor's linker namespace, where the LD_AUDIT
        # engine already sits, so this DT_NEEDED resolves to the engine itself
        # rather than to a second copy.
        PUBLIC LuthierAudit
        hip::host rocprofiler-sdk::rocprofiler-sdk hsa-runtime64::hsa-runtime64 LuthierAMDGPU
        LLVMAMDGPUAsmParser LLVMAMDGPUDesc LLVMAMDGPUDisassembler LLVMAMDGPUInfo LLVMAMDGPUTargetMCA
        LLVMAMDGPUUtils LLVMAMDGPUCodeGen
        PRIVATE
        Threads::Threads
        LLVMSymbolize LLVMDebugInfoDWARF LLVMCodeGenTypes LLVMMC LLVMMCA LLVMMCDisassembler LLVMObject
        LLVMSupport LLVMTarget LLVMTargetParser LLVMTransformUtils LLVMBitReader LLVMAnalysis LLVMAsmPrinter
        LLVMCore LLVMPasses LLVMCodeGen LLVMLinker LLVMIRReader
        # dlmopen, for loading the HSA runtime into the application's linker
        # namespace rather than the auditor's -- see
        # PacketMonitorTrait::ensureHsaInitializedInApplication.
        ${CMAKE_DL_LIBS})

if (${LUTHIER_ENABLE_SPIRV})
    target_link_libraries(LuthierTooling PRIVATE Luthier::LLVMSPIRVTranslator)
endif ()
