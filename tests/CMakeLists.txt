#enable_testing()
#include(CTest)

#add_library(
#        code_lifter_test SHARED
#        code_lifter_test.cpp
#)

# GoogleTest requires at least C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#
include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
FetchContent_MakeAvailable(googletest)

add_library(CodeLifterTest SHARED
        $<TARGET_OBJECTS:LuthierComgr>
        $<TARGET_OBJECTS:LuthierRocprofilerSDK>
        $<TARGET_OBJECTS:LuthierHSA>
        $<TARGET_OBJECTS:LuthierHIP>
        $<TARGET_OBJECTS:LuthierLLVM>
        $<TARGET_OBJECTS:LuthierIntrinsic>
        $<TARGET_OBJECTS:LuthierCommon>
        $<TARGET_OBJECTS:LuthierToolingCommon>
        $<TARGET_OBJECTS:LuthierTooling> CodeLifterTest.cpp
)

find_package(LLVM REQUIRED CONFIG)
find_package(amd_comgr REQUIRED CONFIG)
#find_package(hip REQUIRED CONFIG)
find_package(rocprofiler-sdk REQUIRED CONFIG)

get_target_property(amd_comgr_INCLUDE_DIRS amd_comgr INTERFACE_INCLUDE_DIRECTORIES)

target_include_directories(CodeLifterTest
        PRIVATE
        "${CMAKE_SOURCE_DIR}/src"
        "${CMAKE_SOURCE_DIR}/src/include"
        ${amd_comgr_INCLUDE_DIRS}
        ${LLVM_INCLUDE_DIRS}
        PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>"
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")

#target_include_directories(code_lifter_test PUBLIC ${CMAKE_SOURCE_DIR}/src/)

#target_include_directories(code_lifter_test PUBLIC ${CMAKE_SOURCE_DIR}/include/)
#target_compile_definitions(code_lifter_test PUBLIC AMD_INTERNAL_BUILD)
#target_compile_definitions(code_lifter_test PUBLIC ${LLVM_DEFINITIONS})
#target_include_directories(code_lifter_test PUBLIC ${LLVM_INCLUDE_DIRS})

target_compile_definitions(CodeLifterTest PRIVATE AMD_INTERNAL_BUILD ${LLVM_DEFINITIONS})

#target_compile_definitions(LuthierToolingCommon PRIVATE AMD_INTERNAL_BUILD ${LLVM_DEFINITIONS})

#target_include_directories(code_lifter_test PUBLIC ${LUTHIER_LLVM_BIN_DIR}/lib/Target/AMDGPU)
##
#target_include_directories(code_lifter_test PUBLIC ${LUTHIER_LLVM_SRC_DIR}/llvm/lib/Target/AMDGPU/)
##
#target_include_directories(code_lifter_test PUBLIC ${LUTHIER_LLVM_BIN_DIR}/lib/Target/AMDGPU)

enable_testing()

target_link_libraries(
#        code_lifter_test
        CodeLifterTest
        luthier
#        Gtest::gtest_main
        # doctest::doctest
)

include(GoogleTest)
#gtest_discover_tests(CodeLifterTest)