add_library(LuthierRocprofilerSDK OBJECT
        RocprofilerError.cpp
        RocprofilerSdkLibrary.cpp
)

target_compile_definitions(LuthierRocprofilerSDK PRIVATE ${LLVM_DEFINITIONS} ${hip_COMPILE_DEFINITIONS})

target_compile_options(LuthierRocprofilerSDK
        PUBLIC ${LUTHIER_RTTI_FLAG})

target_include_directories(LuthierRocprofilerSDK
        PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
        PRIVATE
        ${rocprofiler-sdk_INCLUDE_DIRS}
        ${LLVM_INCLUDE_DIRS}
        ${hip_INCLUDE_DIRS}
)