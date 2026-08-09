set(LLVM_LINK_COMPONENTS
        Object
        Support
        TargetParser)

#===------------------------------------------------------------------------===#
# GPU-free unit tests: the hostcall protocol, the printf encodings, and the
# hidden-argument fill are all host logic over fixed ABIs, so they run
# everywhere.
#===------------------------------------------------------------------------===#
add_llvm_executable(HSAToolingTest
        PARTIAL_SOURCES_INTENDED
        DevicePrintfTest.cpp
        HiddenArgumentsTest.cpp
        HostcallProtocolTest.cpp)

target_compile_definitions(HSAToolingTest PRIVATE AMD_INTERNAL_BUILD ${LLVM_DEFINITIONS})

target_include_directories(HSAToolingTest PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src/test/unit
        ${LLVM_INCLUDE_DIRS}
        ${hsa-runtime64_INCLUDE_DIRS})

target_link_libraries(HSAToolingTest PRIVATE
        gtest gtest_main LuthierTooling ${CMAKE_DL_LIBS}
        hip::host hsa-runtime64::hsa-runtime64)

gtest_discover_tests(HSAToolingTest)

#===------------------------------------------------------------------------===#
# Device relocatable for the end-to-end loader test.
#
# loadInstrumented takes an AMDGCN relocatable, and prying one out of the HIP
# driver takes two steps. In the default non-RDC mode the device pipeline runs
# all the way through lld, so `--cuda-device-only -c` hands back a finished
# ET_DYN code object, which the loader's own lld invocation would then treat as
# a shared-library input and link nothing out of. `-fgpu-rdc -c` is no better:
# it defers codegen and emits bitcode. Stopping the HIP driver at `-S` and
# assembling the result for the bare amdgcn target is what leaves an ET_REL,
# with the device libraries already linked in.
#
#   --cuda-device-only -S     : device assembly only, before lld runs
#   --no-gpu-bundle-output    : a bare .s rather than an offload bundle
#   -fgpu-allow-device-init   : without it clang drops the dynamic initializers
#                               for __device__ globals and emits no
#                               amdgcn.device.init at all
#
# Only one architecture is built; the test skips on a machine whose GPU does
# not match it.
#===------------------------------------------------------------------------===#
set(_LUTHIER_TEST_HIP_ARCH "")
if (CMAKE_HIP_ARCHITECTURES)
    list(GET CMAKE_HIP_ARCHITECTURES 0 _LUTHIER_TEST_HIP_ARCH)
endif ()

if (_LUTHIER_TEST_HIP_ARCH AND CMAKE_HIP_COMPILER)
    set(_LUTHIER_TEST_INIT_FINI_OBJECT
            "${CMAKE_CURRENT_BINARY_DIR}/InitFiniKernels-${_LUTHIER_TEST_HIP_ARCH}.o")
    set(_LUTHIER_TEST_INIT_FINI_ASM
            "${CMAKE_CURRENT_BINARY_DIR}/InitFiniKernels-${_LUTHIER_TEST_HIP_ARCH}.s")

    add_custom_command(
            OUTPUT "${_LUTHIER_TEST_INIT_FINI_OBJECT}"
            COMMAND "${CMAKE_HIP_COMPILER}"
            -x hip --cuda-device-only -S --no-gpu-bundle-output
            -fgpu-allow-device-init -O3 -g0
            --offload-arch=${_LUTHIER_TEST_HIP_ARCH}
            -o "${_LUTHIER_TEST_INIT_FINI_ASM}"
            "${CMAKE_CURRENT_SOURCE_DIR}/InitFiniKernels.hip"
            COMMAND "${CMAKE_HIP_COMPILER}"
            --target=amdgcn-amd-amdhsa -mcpu=${_LUTHIER_TEST_HIP_ARCH} -c
            -o "${_LUTHIER_TEST_INIT_FINI_OBJECT}"
            "${_LUTHIER_TEST_INIT_FINI_ASM}"
            BYPRODUCTS "${_LUTHIER_TEST_INIT_FINI_ASM}"
            DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/InitFiniKernels.hip"
            COMMENT "Building the init/fini device relocatable for ${_LUTHIER_TEST_HIP_ARCH}"
            VERBATIM)

    add_custom_target(HSAToolingTestInitFiniObject
            DEPENDS "${_LUTHIER_TEST_INIT_FINI_OBJECT}")
else ()
    # Without a HIP compiler or a target architecture there is nothing to load;
    # the test skips on the missing file.
    set(_LUTHIER_TEST_INIT_FINI_OBJECT
            "${CMAKE_CURRENT_BINARY_DIR}/InitFiniKernels-unavailable.o")
    set(_LUTHIER_TEST_HIP_ARCH "none")
    message(STATUS
            "HSATooling: no HIP compiler / CMAKE_HIP_ARCHITECTURES; the "
            "instrumented-kernel-loader end-to-end tests will skip.")
endif ()

#===------------------------------------------------------------------------===#
# GPU-gated end-to-end loader tests. This binary registers itself as an
# in-process rocprofiler tool to obtain the HSA table snapshots the loader
# needs, so it must be its own executable.
#===------------------------------------------------------------------------===#
add_llvm_executable(InstrumentedKernelLoaderE2ETest
        PARTIAL_SOURCES_INTENDED
        InstrumentedKernelLoaderE2ETest.cpp)

target_compile_definitions(InstrumentedKernelLoaderE2ETest PRIVATE
        AMD_INTERNAL_BUILD ${LLVM_DEFINITIONS}
        LUTHIER_TEST_INIT_FINI_OBJECT="${_LUTHIER_TEST_INIT_FINI_OBJECT}"
        LUTHIER_TEST_INIT_FINI_ARCH="${_LUTHIER_TEST_HIP_ARCH}")

target_include_directories(InstrumentedKernelLoaderE2ETest PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src/test/unit
        ${LLVM_INCLUDE_DIRS}
        ${hsa-runtime64_INCLUDE_DIRS})

target_link_libraries(InstrumentedKernelLoaderE2ETest PRIVATE
        gtest gtest_main LuthierTooling ${CMAKE_DL_LIBS}
        hip::host hsa-runtime64::hsa-runtime64 rocprofiler-sdk::rocprofiler-sdk)

if (TARGET HSAToolingTestInitFiniObject)
    add_dependencies(InstrumentedKernelLoaderE2ETest HSAToolingTestInitFiniObject)
endif ()

gtest_discover_tests(InstrumentedKernelLoaderE2ETest)
