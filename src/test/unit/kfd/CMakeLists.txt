# Unit tests for the KFD queue wrapper's GPU-free logic.
#
# These test the *checks* used by the hardware suite and the parts of the
# wrapper that do not need hardware, so they run anywhere. That matters: the
# recurring failure in this work has been checks that could not fail, so the
# checks get tested too.
#
# Two areas, one target, because LLVM's build system wants a single target per
# directory:
#   VerifyTest.cpp        - the four verification checks
#   CallbackChainTest.cpp - "last registered runs first", which is otherwise the
#                           kind of claim that lives only in a comment

set(LLVM_LINK_COMPONENTS Support)

add_llvm_executable(KFDUnitTests
        VerifyTest.cpp
        CallbackChainTest.cpp
        ${CMAKE_SOURCE_DIR}/src/test/kfd/Verify.cpp
        # The wrapper's logic, not the shared library. The interposer that
        # redefines ioctl lives in a separate file, so linking this in does not
        # interpose on the test process itself.
        ${CMAKE_SOURCE_DIR}/src/lib/KFD/QueueWrapper.cpp)

target_include_directories(KFDUnitTests PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src/test
        ${LLVM_INCLUDE_DIRS}
        ${hsa-runtime64_INCLUDE_DIRS})

target_link_libraries(KFDUnitTests PRIVATE
        gtest gtest_main ${CMAKE_DL_LIBS} pthread)

gtest_discover_tests(KFDUnitTests)
