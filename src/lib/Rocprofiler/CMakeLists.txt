add_library(LuthierRocprofiler OBJECT
        HsaApiTable.cpp
        RocprofilerError.cpp
)

target_compile_definitions(LuthierRocprofiler PRIVATE
        AMD_INTERNAL_BUILD ${hip_DEFINES} ${LLVM_DEFINITIONS})

target_include_directories(LuthierRocprofiler
        PRIVATE
        ${hsa-runtime64_INCLUDE_DIRS}
        ${rocprofiler-sdk_INCLUDE_DIR}
        ${hip_INCLUDE_DIRS}
        ${LLVM_INCLUDE_DIRS}
        PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)