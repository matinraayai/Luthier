# Hardware tests for the KFD queue wrapper.
#
# Off by default: these need a real GPU, so they cannot run everywhere.
#
#   cmake -DLUTHIER_BUILD_KFD_HW_TESTS=ON ...
#   ctest -R kfd
#
# The binary here must NOT link the HSA runtime. It stands in for an application
# that never uses HSA, which is the entire case issue #85 is about -- if HSA ever
# creeps in through a transitive dependency, every result becomes meaningless.
# A test enforces that rather than trusting review.

option(LUTHIER_BUILD_KFD_HW_TESTS
        "Build the KFD queue wrapper hardware tests (needs an AMD GPU)" OFF)

if (NOT LUTHIER_BUILD_KFD_HW_TESTS)
    return()
endif ()

# --- Tools from the ROCm install -------------------------------------------
find_program(ROCM_CLANG clang
        HINTS ${ROCM_PATH}/llvm/bin /opt/rocm/llvm/bin
        DOC "clang from the ROCm install, used to assemble the test kernel")
find_program(ROCM_OBJCOPY llvm-objcopy
        HINTS ${ROCM_PATH}/llvm/bin /opt/rocm/llvm/bin)
find_program(ROCM_AMDGPU_ARCH amdgpu-arch
        HINTS ${ROCM_PATH}/llvm/bin /opt/rocm/llvm/bin)

if (NOT ROCM_CLANG OR NOT ROCM_OBJCOPY)
    message(FATAL_ERROR
            "LUTHIER_BUILD_KFD_HW_TESTS needs clang and llvm-objcopy from a ROCm "
            "install; set ROCM_PATH if it is not in the usual place")
endif ()

# --- Which GPU architectures to build the test kernel for -------------------
# Detected by default so a collaborator gets a kernel for their own hardware
# without configuring anything. The kernel avoids architecture-specific
# instructions, so extra targets are cheap and harmless.
if (NOT LUTHIER_KFD_TEST_ARCHS)
    if (ROCM_AMDGPU_ARCH)
        execute_process(COMMAND ${ROCM_AMDGPU_ARCH}
                OUTPUT_VARIABLE DETECTED_ARCHS
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET)
        string(REPLACE "\n" ";" DETECTED_ARCHS "${DETECTED_ARCHS}")
        list(REMOVE_DUPLICATES DETECTED_ARCHS)
        list(REMOVE_ITEM DETECTED_ARCHS "")
        set(LUTHIER_KFD_TEST_ARCHS "${DETECTED_ARCHS}")
    endif ()
endif ()

if (NOT LUTHIER_KFD_TEST_ARCHS)
    # No GPU visible at configure time (a container without devices, say). Build
    # for a spread of generations so the binary is still useful somewhere.
    set(LUTHIER_KFD_TEST_ARCHS gfx906 gfx908 gfx90a gfx942 gfx1030 gfx1100)
    message(WARNING
            "Could not detect a GPU; building KFD test kernels for a default set: "
            "${LUTHIER_KFD_TEST_ARCHS}. Set LUTHIER_KFD_TEST_ARCHS to override.")
endif ()

message(STATUS "KFD test kernels will be built for: ${LUTHIER_KFD_TEST_ARCHS}")

# --- Assemble the test kernel for each architecture -------------------------
set(KERNEL_BINS)
foreach (ARCH IN LISTS LUTHIER_KFD_TEST_ARCHS)
    set(OBJ ${CMAKE_CURRENT_BINARY_DIR}/TestKernel_${ARCH}.o)
    set(BIN ${CMAKE_CURRENT_BINARY_DIR}/TestKernel_${ARCH}.bin)
    add_custom_command(OUTPUT ${BIN}
            COMMAND ${ROCM_CLANG} -x assembler -target amdgcn-amd-amdhsa
            -mcpu=${ARCH} -c ${CMAKE_CURRENT_SOURCE_DIR}/TestKernel.s -o ${OBJ}
            COMMAND ${ROCM_OBJCOPY} -O binary --only-section=.text ${OBJ} ${BIN}
            DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/TestKernel.s
            COMMENT "Assembling the KFD test kernel for ${ARCH}"
            VERBATIM)
    list(APPEND KERNEL_BINS ${BIN})
endforeach ()

# Embed them, so the suite is one self-contained binary with nothing to find at
# run time.
set(KERNEL_DATA ${CMAKE_CURRENT_BINARY_DIR}/TestKernelData.cpp)
add_custom_command(OUTPUT ${KERNEL_DATA}
        COMMAND ${CMAKE_COMMAND}
        -DARCHS=${LUTHIER_KFD_TEST_ARCHS}
        -DBINDIR=${CMAKE_CURRENT_BINARY_DIR}
        -DOUT=${KERNEL_DATA}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/EmbedTestKernels.cmake
        DEPENDS ${KERNEL_BINS} ${CMAKE_CURRENT_SOURCE_DIR}/EmbedTestKernels.cmake
        COMMENT "Embedding the KFD test kernels"
        VERBATIM)

# --- The non-HSA harness ----------------------------------------------------
add_executable(kfd-nonhsa-tests
        NonHsaMain.cpp
        AqlTestQueue.cpp
        Scenarios.cpp
        Verify.cpp
        ${KERNEL_DATA})

target_include_directories(kfd-nonhsa-tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/include
        ${hsa-runtime64_INCLUDE_DIRS})

# libhsakmt is the thin driver-call library. Deliberately NOT hsa-runtime64.
find_library(HSAKMT_LIBRARY NAMES hsakmt libhsakmt.a
        HINTS ${ROCM_PATH}/lib /opt/rocm/lib)
if (NOT HSAKMT_LIBRARY)
    message(FATAL_ERROR "could not find libhsakmt in the ROCm install")
endif ()

target_link_libraries(kfd-nonhsa-tests PRIVATE
        ${HSAKMT_LIBRARY} ${CMAKE_DL_LIBS} pthread numa drm drm_amdgpu)

# --- The HSA oracle ---------------------------------------------------------
# The opposite requirement to the harness above: this one MUST link the HSA
# runtime, because the case it checks only exists when a tool has HSA loaded.
# Deliberately excluded from the no-HSA-linkage test below.
#
# No AqlTestQueue here, and so no libhsakmt: the runtime does not export the
# hsaKmt symbols (checked), so linking libhsakmt alongside it would put two
# copies of that library's state in one process, each with its own /dev/kfd
# descriptor. The second one fails ACQUIRE_VM, as the Phase 0.1 spike measured.
add_executable(kfd-hsa-oracle
        HsaOracleMain.cpp
        Scenarios.cpp
        Verify.cpp)

target_include_directories(kfd-hsa-oracle PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/include
        ${hsa-runtime64_INCLUDE_DIRS})

target_link_libraries(kfd-hsa-oracle PRIVATE
        hsa-runtime64::hsa-runtime64 ${CMAKE_DL_LIBS} pthread)

# --- Tests ------------------------------------------------------------------

# The property the whole suite rests on. Checked as a test rather than assumed,
# because a transitive dependency could introduce HSA quietly and nothing else
# would notice.
add_test(NAME kfd-nonhsa-links-no-hsa
        COMMAND ${CMAKE_COMMAND}
        -DBINARY=$<TARGET_FILE:kfd-nonhsa-tests>
        -P ${CMAKE_CURRENT_SOURCE_DIR}/CheckNoHsaLinkage.cmake)

# Unwrapped: shows the workloads themselves are sound. Only their results are
# checked, since there is no interception to observe.
add_test(NAME kfd-scenarios-baseline
        COMMAND kfd-nonhsa-tests)

# Wrapped: the real check. --require-wrapper makes this fail rather than quietly
# fall back to the weaker baseline verification if the wrapper does not attach.
add_test(NAME kfd-scenarios-wrapped
        COMMAND kfd-nonhsa-tests --require-wrapper)
set_tests_properties(kfd-scenarios-wrapped PROPERTIES
        ENVIRONMENT
        "LD_PRELOAD=$<TARGET_FILE:luthier-kfd-queue-wrapper>")

# Self-exclusion: the wrapper must leave alone the queues the HSA runtime
# creates for the tool itself.
#
# Matching the output as well as the exit code. An exit code alone is too weak
# here: a run where the wrapper failed to load exits non-zero for a completely
# different reason, and this test would then be reporting on a check that never
# ran -- the same trap --require-wrapper closes on the other harness.
#
# This test was written while the gap was still open and required the FAIL line.
# Flipping it to PASS is what closing the gap looked like.
add_test(NAME kfd-oracle-self-exclusion
        COMMAND kfd-hsa-oracle)
set_tests_properties(kfd-oracle-self-exclusion PROPERTIES
        PASS_REGULAR_EXPRESSION "S19-tool-own-queues +PASS"
        ENVIRONMENT
        "LD_PRELOAD=$<TARGET_FILE:luthier-kfd-queue-wrapper>")
