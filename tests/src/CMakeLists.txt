
add_library(TestingUtils
        $<TARGET_OBJECTS:LuthierComgr>
        $<TARGET_OBJECTS:LuthierRocprofilerSDK>
        $<TARGET_OBJECTS:LuthierHSA>
        $<TARGET_OBJECTS:LuthierHIP>
        $<TARGET_OBJECTS:LuthierLLVM>
        $<TARGET_OBJECTS:LuthierIntrinsic>
        $<TARGET_OBJECTS:LuthierCommon>
        $<TARGET_OBJECTS:LuthierToolingCommon>
        $<TARGET_OBJECTS:LuthierTooling> TestingUtils.cpp
)

find_package(LLVM REQUIRED CONFIG)
find_package(amd_comgr REQUIRED CONFIG)
#find_package(hip REQUIRED CONFIG)
find_package(rocprofiler-sdk REQUIRED CONFIG)

get_target_property(amd_comgr_INCLUDE_DIRS amd_comgr INTERFACE_INCLUDE_DIRECTORIES)

target_include_directories(TestingUtils
        PRIVATE
        "${CMAKE_SOURCE_DIR}/src"
        "${CMAKE_SOURCE_DIR}/src/include"
        "${CMAKE_SOURCE_DIR}/tests/include"
        ${amd_comgr_INCLUDE_DIRS}
        ${LLVM_INCLUDE_DIRS}
        PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>"
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")

target_compile_definitions(TestingUtils PRIVATE AMD_INTERNAL_BUILD ${LLVM_DEFINITIONS})