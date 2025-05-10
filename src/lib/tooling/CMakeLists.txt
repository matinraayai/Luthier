add_library(LuthierTooling SHARED
        $<TARGET_OBJECTS:LuthierComgr>
        $<TARGET_OBJECTS:LuthierRocprofilerSDK>
        $<TARGET_OBJECTS:LuthierHSA>
        $<TARGET_OBJECTS:LuthierHIP>
        $<TARGET_OBJECTS:LuthierLLVM>
        $<TARGET_OBJECTS:LuthierIntrinsic>
        $<TARGET_OBJECTS:LuthierCommon>
        $<TARGET_OBJECTS:LuthierToolingCommon> luthier.cpp Controller.cpp
)

target_compile_definitions(LuthierTooling PUBLIC AMD_INTERNAL_BUILD ${LLVM_DEFINITIONS})

# target_include_directories(LuthierTooling
#         PRIVATE
#         "${CMAKE_SOURCE_DIR}/src/include"
#         PUBLIC
#         ${LLVM_INCLUDE_DIRS}
#         "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>"
#         "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
#         "$<BUILD_INTERFACE:${AMDGPU_INCLUDE_DIR}>"
#         "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/luthier/amdgpu>"
# )

target_include_directories(LuthierTooling
        PRIVATE
        "${CMAKE_SOURCE_DIR}/src/include"
        PUBLIC
        ${LLVM_INCLUDE_DIRS}
        "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>"
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>"
        "$<BUILD_INTERFACE:${AMDGPU_INCLUDE_DIR}>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/luthier/amdgpu>"
)

target_link_libraries(LuthierTooling
        PUBLIC hip::host rocprofiler-sdk::rocprofiler-sdk hsa-runtime64::hsa-runtime64 LuthierAMDGPU
        PRIVATE
        LuthierObject LLVMCodeGenTypes LLVMMC LLVMMCA LLVMMCDisassembler LLVMObject
        LLVMSupport LLVMTarget LLVMTargetParser LLVMTransformUtils LLVMBitReader LLVMAnalysis LLVMAsmPrinter
        LLVMAMDGPUAsmParser LLVMAMDGPUDesc LLVMAMDGPUDisassembler LLVMAMDGPUInfo LLVMAMDGPUTargetMCA
        LLVMAMDGPUUtils LLVMCore LLVMPasses LLVMCodeGen LLVMAMDGPUCodeGen)