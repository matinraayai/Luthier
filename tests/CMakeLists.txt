include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/heads/main.zip
)

FetchContent_MakeAvailable(googletest)


enable_testing()

add_executable(
        code_lifter_test
        code_lifter_test.cpp
)

target_include_directories(code_lifter_test PUBLIC ${CMAKE_SOURCE_DIR}/src/)

target_include_directories(code_lifter_test PUBLIC ${CMAKE_SOURCE_DIR}/include/)

target_compile_definitions(code_lifter_test PUBLIC AMD_INTERNAL_BUILD)

#target_include_directories(code_lifter_test PUBLIC ${LLVM_INCLUDE_DIRS})

#target_compile_definitions(code_lifter_test PUBLIC ${LLVM_DEFINITIONS})

#target_include_directories(code_lifter_test PUBLIC ${LUTHIER_LLVM_BIN_DIR}/lib/Target/AMDGPU)
#
#target_include_directories(code_lifter_test PUBLIC ${LUTHIER_LLVM_SRC_DIR}/llvm/lib/Target/AMDGPU/)
#
#target_include_directories(code_lifter_test PUBLIC ${LUTHIER_LLVM_BIN_DIR}/lib/Target/AMDGPU)

target_link_libraries(
        code_lifter_test
        luthier
        GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(code_lifter_test)