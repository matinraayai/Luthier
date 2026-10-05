set(LLVM_LINK_COMPONENTS Support Symbolize DebugInfoDWARF Object)

set(LUTHIER_ROCPROFILER_TEST_SUPPORT
        ${CMAKE_SOURCE_DIR}/src/lib/LLVM/LLVMError.cpp
        ${CMAKE_SOURCE_DIR}/src/lib/LLVM/streams.cpp
        ${CMAKE_SOURCE_DIR}/src/lib/HSA/HsaError.cpp
        ${CMAKE_SOURCE_DIR}/src/lib/HIP/HipError.cpp
        ${CMAKE_SOURCE_DIR}/src/lib/Rocprofiler/RocprofilerError.cpp
        ${CMAKE_SOURCE_DIR}/src/lib/Rocprofiler/RocprofilerSdkLibrary.cpp)

#===------------------------------------------------------------------------===#
# GPU-free unit tests (always run): traits, callback logic, death tests.
#===------------------------------------------------------------------------===#
add_llvm_executable(RocprofilerTest
        PARTIAL_SOURCES_INTENDED
        RocprofilerTest.cpp
        ${LUTHIER_ROCPROFILER_TEST_SUPPORT})

target_compile_definitions(RocprofilerTest PRIVATE AMD_INTERNAL_BUILD ${LLVM_DEFINITIONS})

target_include_directories(RocprofilerTest PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src/test/unit
        ${LLVM_INCLUDE_DIRS}
        ${hsa-runtime64_INCLUDE_DIRS})

target_link_libraries(RocprofilerTest PRIVATE
        gtest gtest_main LuthierCommon ${CMAKE_DL_LIBS}
        hip::host hsa-runtime64::hsa-runtime64 rocprofiler-sdk::rocprofiler-sdk)

gtest_discover_tests(RocprofilerTest)

#===------------------------------------------------------------------------===#
# GPU-gated end-to-end tests: this binary acts as an in-process rocprofiler
# tool. GPU-dependent tests skip when no AMD GPU / HSA runtime is available.
#===------------------------------------------------------------------------===#
add_llvm_executable(RocprofilerToolE2ETest
        PARTIAL_SOURCES_INTENDED
        RocprofilerToolE2ETest.cpp
        ${LUTHIER_ROCPROFILER_TEST_SUPPORT})

target_compile_definitions(RocprofilerToolE2ETest PRIVATE AMD_INTERNAL_BUILD ${LLVM_DEFINITIONS})

target_include_directories(RocprofilerToolE2ETest PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src/test/unit
        ${LLVM_INCLUDE_DIRS}
        ${hsa-runtime64_INCLUDE_DIRS})

target_link_libraries(RocprofilerToolE2ETest PRIVATE
        gtest gtest_main LuthierCommon ${CMAKE_DL_LIBS}
        hip::host hsa-runtime64::hsa-runtime64 rocprofiler-sdk::rocprofiler-sdk)

gtest_discover_tests(RocprofilerToolE2ETest)
