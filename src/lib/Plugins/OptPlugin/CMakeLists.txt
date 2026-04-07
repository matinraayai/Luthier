add_llvm_pass_plugin(LuthierOptPlugin
        OptPlugin.cpp
        $<TARGET_OBJECTS:LuthierToolingCommon>
        $<TARGET_OBJECTS:LuthierCommon>
        $<TARGET_OBJECTS:LuthierObject>
        $<TARGET_OBJECTS:LuthierLLVM>
        #$<TARGET_OBJECTS:LuthierIntrinsic>
)

target_compile_definitions(LuthierOptPlugin PRIVATE AMD_INTERNAL_BUILD ${LLVM_DEFINITIONS})

target_include_directories(LuthierOptPlugin PRIVATE
        ${hsa-runtime64_INCLUDE_DIRS}
        "$<BUILD_INTERFACE:${LLVM_INCLUDE_DIRS}>"
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>")

target_link_libraries(LuthierOptPlugin PRIVATE LuthierAMDGPU LLVMCodeGenTypes LLVMMC LLVMMCA LLVMMCDisassembler
        LLVMObject LLVMSupport LLVMTarget LLVMTargetParser LLVMTransformUtils LLVMBitReader LLVMAnalysis LLVMAsmPrinter
        LLVMAMDGPUAsmParser LLVMAMDGPUDesc LLVMAMDGPUDisassembler LLVMAMDGPUInfo LLVMAMDGPUTargetMCA
        LLVMAMDGPUUtils LLVMCore LLVMPasses LLVMCodeGen LLVMAMDGPUCodeGen)