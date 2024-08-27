add_executable(code_lifter_test code_lifter_test.cpp)

add_hsa_compile_definitions(code_lifter_test)

target_link_libraries(
        code_lifter_test 
        PUBLIC
        rocprofiler-sdk::rocprofiler-sdk
        LuthierCommon
        LuthierToolingCommon 
        LuthierHSA
        LLVMBinaryFormat
        LLVMObject 
        LLVMSupport 
        LLVMTransformUtils
)

