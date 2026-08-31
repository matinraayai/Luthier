# KFD-level AQL queue interception.
#
# Two targets on purpose:
#
#   LuthierKFD                  the logic, as an OBJECT library. Can be linked
#                               into tests or into Luthier without replacing
#                               ioctl for the whole process.
#   luthier-kfd-queue-wrapper   the preloadable shared library. Adds the GOTCHA
#                               interception layer, which is the only part that
#                               touches ioctl process-wide.
#   LuthierKFDTooling           the parts that use LLVM. Linked into
#                               LuthierTooling, and kept OUT of the preloadable
#                               library on purpose (see below).
#
# Keeping the first two apart means the interposer is opt-in: you get it by
# preloading this library, not merely by linking the logic.
#
# LuthierKFDTooling is separate for a different reason. The preloadable library is
# injected into arbitrary applications, so it must not require LLVM to be present
# at load time; LuthierKFD's sources are LLVM-free for exactly that reason, and
# luthier-kfd-queue-wrapper below takes ALL of its objects. Anything that links
# LLVM therefore needs its own target, or preloading the wrapper would start
# demanding LLVM of every process it is injected into.

add_library(LuthierKFD OBJECT
        QueueWrapper.cpp
        AllocationTracker.cpp
        FdSharing.cpp
)

target_compile_definitions(LuthierKFD PRIVATE AMD_INTERNAL_BUILD ${LLVM_DEFINITIONS})

target_include_directories(LuthierKFD
        PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
        PRIVATE
        ${hsa-runtime64_INCLUDE_DIRS}
)

# Required: the objects are linked into a shared library below.
set_property(TARGET LuthierKFD PROPERTY POSITION_INDEPENDENT_CODE ON)

#===----------------------------------------------------------------------===#
# GOTCHA. Ships inside rocprofiler-sdk's private library directory and exposes no
# CMake target and no headers -- see GotchaAbi.h for why we declare the ABI rather
# than fetching or vendoring it.
#
# The directory is derived from rocprofiler-sdk_DIR, which find_package sets to
# <prefix>/lib/cmake/rocprofiler-sdk, so two levels up is the lib directory the
# .so sits under. Derived rather than hardcoded so a different ROCm prefix or
# version needs no edit here.
#===----------------------------------------------------------------------===#
get_filename_component(LUTHIER_ROCPROFILER_LIB_DIR
        "${rocprofiler-sdk_DIR}/../.." ABSOLUTE)

# libgotcha.so.3 in NAMES as well as gotcha: a runtime-only ROCm install has the
# versioned object but not the development symlink.
find_library(LUTHIER_GOTCHA_LIBRARY
        NAMES gotcha libgotcha.so.3
        HINTS "${LUTHIER_ROCPROFILER_LIB_DIR}" "${ROCM_PATH}/lib" /opt/rocm/lib
        PATH_SUFFIXES rocprofiler-sdk
        DOC "libgotcha, shipped inside rocprofiler-sdk's private lib directory")

if (NOT LUTHIER_GOTCHA_LIBRARY)
    message(FATAL_ERROR
            "libgotcha was not found. It ships with rocprofiler-sdk under "
            "<rocm>/lib/rocprofiler-sdk and is what the KFD wrapper uses to "
            "intercept ioctl. Set LUTHIER_GOTCHA_LIBRARY explicitly if your "
            "install puts it elsewhere.")
endif ()
message(STATUS "KFD ioctl interception will use ${LUTHIER_GOTCHA_LIBRARY}")

add_library(luthier-kfd-queue-wrapper SHARED
        GotchaIoctlInterceptor.cpp
        $<TARGET_OBJECTS:LuthierKFD>
)

target_compile_definitions(luthier-kfd-queue-wrapper PRIVATE AMD_INTERNAL_BUILD ${LLVM_DEFINITIONS})

target_include_directories(luthier-kfd-queue-wrapper
        PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
        PRIVATE
        ${hsa-runtime64_INCLUDE_DIRS}
)

# libgotcha for interception; dl for finding the tracker's exports; pthreads for
# the polling thread. libgotcha goes on this target only, never on LuthierKFD --
# that object library is linked into KFDUnitTests, and a unit test process must not
# interpose on itself.
target_link_libraries(luthier-kfd-queue-wrapper PRIVATE
        ${LUTHIER_GOTCHA_LIBRARY} ${CMAKE_DL_LIBS} pthread)

#===----------------------------------------------------------------------===#
# The LLVM-facing half: bridges the tracker to Luthier's code generation
# pipeline. Deliberately NOT part of luthier-kfd-queue-wrapper -- see the note at
# the top of this file. It finds the tracker with dlsym at run time rather than
# linking it, so that the records it reads are the ones the preloaded wrapper
# actually collected instead of a second, permanently empty map.
#===----------------------------------------------------------------------===#
add_library(LuthierKFDTooling OBJECT
        KfdAllocationResolver.cpp
        IsaInfo.cpp
        KfdAgent.cpp
        KfdTargetMachine.cpp
        Topology.cpp
)

target_compile_definitions(LuthierKFDTooling PRIVATE AMD_INTERNAL_BUILD ${LLVM_DEFINITIONS})

target_include_directories(LuthierKFDTooling
        PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
        PRIVATE
        ${LLVM_INCLUDE_DIRS}
        ${hsa-runtime64_INCLUDE_DIRS}
)

set_property(TARGET LuthierKFDTooling PROPERTY POSITION_INDEPENDENT_CODE ON)
