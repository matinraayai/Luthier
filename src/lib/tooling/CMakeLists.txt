add_library(LuthierTooling Controller.cpp
)
# $<TARGET_OBJECTS:LuthierCommon> $<TARGET_OBJECTS:LuthierHIP> $<TARGET_OBJECTS:LuthierHSA>
#        $<TARGET_OBJECTS:LuthierToolingCommon>

target_include_directories(LuthierTooling PUBLIC
        "${CMAKE_SOURCE_DIR}/src/include"
)

target_include_directories(LuthierTooling PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

target_include_directories(LuthierTooling PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include/luthier/amdgpu>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/include/luthier/amdgpu>"
)


luthier_add_position_independence(LuthierTooling)
luthier_add_hsa_compile_definitions(LuthierTooling)
luthier_add_llvm_compile_definitions(LuthierTooling)

target_link_libraries(LuthierTooling PUBLIC luthier::Intrinsics rocprofiler-sdk::rocprofiler-sdk
        LuthierCommon LuthierHIP LuthierHSA LuthierToolingCommon
        LLVMCodeGenTypes LLVMMC LLVMMCA LLVMMCDisassembler LLVMObject
        LLVMSupport LLVMTarget LLVMTargetParser LLVMTransformUtils LLVMBitReader LLVMAnalysis LLVMAsmPrinter
        LLVMAMDGPUAsmParser LLVMAMDGPUDesc LLVMAMDGPUDisassembler LLVMAMDGPUInfo LLVMAMDGPUTargetMCA
        LLVMAMDGPUUtils LLVMCore LLVMPasses LLVMCodeGen LLVMAMDGPUCodeGen)
