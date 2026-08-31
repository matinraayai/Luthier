set(LLVM_LINK_COMPONENTS Support Object Symbolize DebugInfoDWARF BinaryFormat MC TargetParser)

# GPU-free tests for HsaMemoryAllocationAccessor's source-selection logic.
#
# No GPU, no HSA runtime and no rocprofiler are involved, and that is the point:
# an HsaApiTableSnapshot whose registration callback never fires is exactly the
# state the accessor is in inside an application that drives KFD itself, where
# HSA cannot be initialized at all. Constructing the snapshots and leaving them
# unregistered reproduces that state faithfully, so the branch that matters most
# is the one this suite can reach most cheaply.
#
# Sources are compiled in directly rather than linking LuthierHSATooling, which
# would drag in the instrumented-kernel loader and the AMDGPU target for a test
# that needs neither -- the same trade ../rocprofiler/CMakeLists.txt makes.
add_llvm_executable(HSAToolingUnitTests
        PARTIAL_SOURCES_INTENDED
        HsaMemoryAllocationAccessorTest.cpp
        ${CMAKE_SOURCE_DIR}/src/lib/HSATooling/HsaMemoryAllocationAccessor.cpp
        ${CMAKE_SOURCE_DIR}/src/lib/HSATooling/LoadedCodeObjectCache.cpp
        ${CMAKE_SOURCE_DIR}/src/lib/ToolCodeGen/MemoryAllocationAccessor.cpp
        ${CMAKE_SOURCE_DIR}/src/lib/Object/ObjectFileUtils.cpp
        ${CMAKE_SOURCE_DIR}/src/lib/HSA/hsa.cpp
        ${CMAKE_SOURCE_DIR}/src/lib/HSA/HsaError.cpp
        ${CMAKE_SOURCE_DIR}/src/lib/HIP/HipError.cpp
        ${CMAKE_SOURCE_DIR}/src/lib/Rocprofiler/RocprofilerError.cpp
        ${CMAKE_SOURCE_DIR}/src/lib/LLVM/LLVMError.cpp
        ${CMAKE_SOURCE_DIR}/src/lib/LLVM/streams.cpp)

target_compile_definitions(HSAToolingUnitTests PRIVATE AMD_INTERNAL_BUILD ${LLVM_DEFINITIONS})

target_include_directories(HSAToolingUnitTests PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src/test/unit
        ${CMAKE_BINARY_DIR}/include
        ${LLVM_INCLUDE_DIRS}
        ${hsa-runtime64_INCLUDE_DIRS})

target_link_libraries(HSAToolingUnitTests PRIVATE
        gtest gtest_main LuthierCommon ${CMAKE_DL_LIBS}
        hip::host hsa-runtime64::hsa-runtime64 rocprofiler-sdk::rocprofiler-sdk)

gtest_discover_tests(HSAToolingUnitTests)
