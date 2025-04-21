add_executable(
        LuthierComgrTests
        ComgrErrorTest.cpp
)

target_include_directories(LuthierComgrTests PRIVATE
        ${CMAKE_SOURCE_DIR}/src/include
        ${LLVM_INCLUDE_DIRS})

target_link_libraries(
        LuthierComgrTests
        LuthierComgr
        LLVMSupport
        amd_comgr
        GTest::gtest_main
)

gtest_discover_tests(LuthierComgrTests)