add_subdirectory(Comgr)
add_subdirectory(Common)
add_subdirectory(CompilerPlugins)
add_subdirectory(HIP)
add_subdirectory(HSA)
add_subdirectory(HSARuntime)
add_subdirectory(Instrumentation)
add_subdirectory(LLVM)
add_subdirectory(Object)
add_subdirectory(Rocprofiler)

add_library(LuthierTooling SHARED
        $<TARGET_OBJECTS:LuthierComgr>
        $<TARGET_OBJECTS:LuthierCommon>
        $<TARGET_OBJECTS:LuthierHIP>
        $<TARGET_OBJECTS:LuthierHSA>
        $<TARGET_OBJECTS:LuthierHSARuntime>
        $<TARGET_OBJECTS:LuthierInstrumentation>
        $<TARGET_OBJECTS:LuthierLLVM>
        $<TARGET_OBJECTS:LuthierObject>
        $<TARGET_OBJECTS:LuthierRocprofiler>
)

target_compile_definitions(LuthierTooling PUBLIC
        AMD_INTERNAL_BUILD
        ${hsa-runtime64_INCLUDE_DIRS}
        ${hip_DEFINES}
        ${LLVM_DEFINITIONS})

target_include_directories(LuthierTooling
        PUBLIC
        ${LLVM_INCLUDE_DIRS}
        "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>"
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

target_link_libraries(LuthierTooling
        PUBLIC rocprofiler-sdk::rocprofiler-sdk
        PRIVATE LLVMCodeGenTypes LLVMMC LLVMMCA LLVMMCDisassembler LLVMObject
        LLVMSupport LLVMTarget LLVMTargetParser LLVMTransformUtils LLVMBitReader LLVMAnalysis LLVMAsmPrinter
        LLVMAMDGPUAsmParser LLVMAMDGPUDesc LLVMAMDGPUDisassembler LLVMAMDGPUInfo LLVMAMDGPUTargetMCA
        LLVMAMDGPUUtils LLVMCore LLVMPasses LLVMCodeGen LLVMAMDGPUCodeGen)