# User Options =========================================================================================================
option(BUILD_SHARED_LIBS "Builds the components as a shared library" OFF)
option(LUTHIER_GEN_INTERCEPTOR_IMPL_DURING_CONFIG "Generates the interceptor implementations
                                                   during the first cmake configuration" ON)
# ======================================================================================================================

# Set the bin and lib dirs
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

find_package(hip REQUIRED CONFIG)

find_package(amd_comgr REQUIRED CONFIG)

find_package(rocprofiler-sdk REQUIRED CONFIG)

find_package(hsa-runtime64 REQUIRED CONFIG)

find_package(Python3 COMPONENTS Interpreter REQUIRED)

find_package(LLVM REQUIRED CONFIG)

# Get the include directory of amd_comgr
get_target_property(amd_comgr_INCLUDE_DIRS amd_comgr INTERFACE_INCLUDE_DIRECTORIES)

# Get the include directory of hsa-runtime64
get_target_property(hsa-runtime64_INCLUDE_DIRS hsa-runtime64::hsa-runtime64 INTERFACE_INCLUDE_DIRECTORIES)

if (${LUTHIER_BUILD_UNIT_TESTS})
    include(FetchContent)
    FetchContent_Declare(
            doctest
            URL https://github.com/doctest/doctest/archive/refs/heads/master.zip
    )
    FetchContent_MakeAvailable(doctest)
    enable_testing()
endif ()

add_subdirectory(amdgpu)
add_subdirectory(llvm)
add_subdirectory(comgr)
add_subdirectory(rocprofiler-sdk)
add_subdirectory(common)
add_subdirectory(hsa)
add_subdirectory(hip)
add_subdirectory(tooling_common)
add_subdirectory(tooling)