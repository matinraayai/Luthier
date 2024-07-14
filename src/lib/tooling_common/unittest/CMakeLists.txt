add_executable(code_lifter_test code_lifter_test.cpp)

add_hsa_compile_definitions(code_lifter_test)

target_link_libraries(
        code_lifter_test 
        PUBLIC
        # doctest::doctest
        rocprofiler-sdk::rocprofiler-sdk
        luthier_common
        luthier_tooling_common 
        luthier_hsa
        LLVMSupport 
        LLVMTransformUtils
)

