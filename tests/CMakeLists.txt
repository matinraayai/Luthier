include(FetchContent)
FetchContent_Declare(
        doctest
        URL https://github.com/doctest/doctest/archive/refs/heads/master.zip
)

FetchContent_MakeAvailable(doctest)
enable_testing()
include(CTest)

add_library(
        code_lifter_test SHARED
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
        doctest::doctest
)