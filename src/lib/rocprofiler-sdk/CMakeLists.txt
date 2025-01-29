add_library(LuthierRocprofilerSDK OBJECT
        RocprofilerSDKError.cpp
)

target_compile_definitions(LuthierRocprofilerSDK PRIVATE ${LLVM_DEFINITIONS})

target_include_directories(LuthierRocprofilerSDK
        PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
        PRIVATE
        ${CMAKE_SOURCE_DIR}/src/include
        ${rocprofiler-sdk_INCLUDE_DIR}
        ${LLVM_INCLUDE_DIRS}
)

target_link_libraries(LuthierRocprofilerSDK PRIVATE hip::host)