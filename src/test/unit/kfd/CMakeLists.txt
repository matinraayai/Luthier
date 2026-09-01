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
#   AllocationTrackerTest.cpp          - the interval map's boundary behaviour
#   KfdAllocationResolverTest.cpp - found / absent / unknowable, which are
#                           easy to collapse into one another
#   KfdTargetMachineTest.cpp - naming a GPU's ISA from the driver alone, checked
#                           against HSA as an oracle where a GPU is present
#
# The accessor pulls in AMDGCNObjectFile and hence LLVM's Object library and
# Luthier's error types, so the two error sources are compiled in directly rather
# than linking LuthierLLVM whole -- the same approach as
# ../rocprofiler/CMakeLists.txt.

# AMDGPU* are here for the one test that builds a TargetMachine end to end. That
# is the deliverable of the KFD ISA path, so leaving it uncovered because the
# components are bulky would be testing everything except the thing that ships.
set(LLVM_LINK_COMPONENTS Support Object Symbolize DebugInfoDWARF TargetParser
        MC Target CodeGen AMDGPUCodeGen AMDGPUDesc AMDGPUInfo)

add_llvm_executable(KFDUnitTests
        # Two targets in this directory: the tests, and the compile-only check
        # below that instantiates KFDTool.
        PARTIAL_SOURCES_INTENDED
        VerifyTest.cpp
        CallbackChainTest.cpp
        AllocationTrackerTest.cpp
        KfdAllocationResolverTest.cpp
        KfdTargetMachineTest.cpp
        AllocationChainTest.cpp
        ${CMAKE_SOURCE_DIR}/src/test/kfd/Verify.cpp
        # The wrapper's logic, not the shared library. The interposer that
        # redefines ioctl lives in a separate file, so linking this in does not
        # interpose on the test process itself.
        ${CMAKE_SOURCE_DIR}/src/lib/KFD/QueueWrapper.cpp
        ${CMAKE_SOURCE_DIR}/src/lib/KFD/AllocationTracker.cpp
        ${CMAKE_SOURCE_DIR}/src/lib/KFD/KfdAllocationResolver.cpp
        ${CMAKE_SOURCE_DIR}/src/lib/KFD/Topology.cpp
        ${CMAKE_SOURCE_DIR}/src/lib/KFD/IsaInfo.cpp
        ${CMAKE_SOURCE_DIR}/src/lib/KFD/KfdAgent.cpp
        ${CMAKE_SOURCE_DIR}/src/lib/KFD/KfdTargetMachine.cpp
        ${CMAKE_SOURCE_DIR}/src/lib/HSA/HsaError.cpp
        ${CMAKE_SOURCE_DIR}/src/lib/LLVM/LLVMError.cpp
        ${CMAKE_SOURCE_DIR}/src/lib/LLVM/streams.cpp)

# AMD_INTERNAL_BUILD selects the flat include layout inside hsa_api_trace.h.
# Without it that header reaches for "inc/hsa_ext_image.h", which exists only in
# an HSA source tree and not in an install -- so anything here that pulls in the
# API-table headers fails to compile. Every other target in the tree defines it;
# this one did not need to until it started using them.
target_compile_definitions(KFDUnitTests PRIVATE
        AMD_INTERNAL_BUILD ${LLVM_DEFINITIONS})

target_include_directories(KFDUnitTests PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src/test
        ${CMAKE_SOURCE_DIR}/src/test/unit
        ${LLVM_INCLUDE_DIRS}
        ${hsa-runtime64_INCLUDE_DIRS})

# hsa-runtime64 is linked for KfdTargetMachineTest's oracle only: it compares the
# driver-derived ISA against what HSA reports on the same device. Nothing under
# test calls HSA -- in the applications this work targets, it cannot.
target_link_libraries(KFDUnitTests PRIVATE
        gtest gtest_main LuthierCommon ${CMAKE_DL_LIBS} pthread
        hip::host hsa-runtime64::hsa-runtime64)

gtest_discover_tests(KFDUnitTests)

#===------------------------------------------------------------------------===#
# KFDTool is a CRTP base, so it is only checked when something instantiates it.
# This object library exists to be that something. It is compiled and never
# linked: the offload-parser trait's static fields need a HIP translation unit to
# define, which a plain C++ test cannot provide, but instantiation is what the
# question actually is.
#===------------------------------------------------------------------------===#
add_library(KFDToolCompileCheck OBJECT KFDToolCompileCheck.cpp)

target_compile_definitions(KFDToolCompileCheck PRIVATE
        AMD_INTERNAL_BUILD ${LLVM_DEFINITIONS})

target_include_directories(KFDToolCompileCheck PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_BINARY_DIR}/include
        ${LLVM_INCLUDE_DIRS}
        ${hsa-runtime64_INCLUDE_DIRS})

# For the AMDGPU tablegen headers the pipeline trait pulls in, and for hip::host's
# __HIP_PLATFORM_AMD__ -- usage requirements only, nothing is linked.
target_link_libraries(KFDToolCompileCheck PRIVATE LuthierAMDGPU hip::host)
