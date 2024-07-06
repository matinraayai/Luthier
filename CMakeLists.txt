cmake_minimum_required(VERSION 3.21)

# Global CXX Setup =====================================================================================================
project(luthier
        VERSION 0.0.5
        LANGUAGES CXX
        DESCRIPTION "Dynamic Binary Instrumentation for AMD GPUs on ROCm"
)

set(CMAKE_CXX_STANDARD 20)
# ======================================================================================================================
# User Options =========================================================================================================
option(LUTHIER_BUILD_INTEGRATION_TESTS "Builds the integration tests" OFF)
option(LUTHIER_BUILD_UNIT_TESTS "Builds the unit test" OFF)
option(LUTHIER_BUILD_EXAMPLES "Builds the example tools in the examples/ folder" ON)

option(LUTHIER_BUILD_LATEX_DOCS "Builds Luthier documentation with Doxygen in PDF format with Latex" OFF)
option(LUTHIER_BUILD_HTML_DOCS "Builds Luthier documentation with Doxygen in HTML format" OFF)
# ======================================================================================================================
# Required Flags for the project =======================================================================================
set_property(GLOBAL PROPERTY POSITION_INDEPENDENT_CODE TRUE)

# ======================================================================================================================
# Dependency Setup =====================================================================================================
#TODO: make this take arbitrary paths for each ROCm library for Spack
find_package(hip REQUIRED)

find_package(amd_comgr REQUIRED)

find_package(rocprofiler-sdk REQUIRED)

find_package(hsa-runtime64 REQUIRED HINTS ${CMAKE_INSTALL_PREFIX} PATHS /opt/rocm/)

find_package(LLVM REQUIRED CONFIG PATHS ${LUTHIER_LLVM_BIN_DIR} NO_DEFAULT_PATH)

if (${LUTHIER_BUILD_UNIT_TESTS})
    include(FetchContent)
    FetchContent_Declare(
            doctest
            URL https://github.com/doctest/doctest/archive/refs/heads/master.zip
    )

    FetchContent_MakeAvailable(doctest)
    enable_testing()
endif ()

# Adds HSA as a dependency to a target
macro(add_hsa_compile_definitions target_name)
    # Some HSA include files won't work unless this is defined
    add_compile_definitions(${target_name} PUBLIC AMD_INTERNAL_BUILD)
endmacro()

# Adds LLVM as a dependency to a target
macro(add_llvm_compile_definitions target_name)
    target_include_directories(${target_name} PUBLIC ${LLVM_INCLUDE_DIRS})

    target_compile_definitions(${target_name} PUBLIC ${LLVM_DEFINITIONS})

    target_include_directories(${target_name} PUBLIC ${LUTHIER_LLVM_BIN_DIR}/lib/Target/AMDGPU)

    target_include_directories(${target_name} PUBLIC ${LUTHIER_LLVM_SRC_DIR}/llvm/lib/Target/AMDGPU/)

    target_include_directories(${target_name} PUBLIC ${LUTHIER_LLVM_BIN_DIR}/lib/Target/AMDGPU)
endmacro()

# ======================================================================================================================

# Library Declaration ==================================================================================================
set(LUTHIER_PUBLIC_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)
set(LUTHIER_PRIVATE_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/src/include)
add_subdirectory(src)
# ======================================================================================================================

# Any Compilation Flags, Linking Configuration Left ====================================================================
#target_compile_options(luthier PUBLIC "-fPIC")

# TODO: Add option to build with sanitizers
#target_compile_options(luthier PUBLIC "-fsanitize=address")

# ======================================================================================================================

# Example Tools Configuration (if enabled) =============================================================================
if (${LUTHIER_BUILD_EXAMPLES})
    set(SUB_DIRS examples/kernel_instrument)
    foreach (DIR ${SUB_DIRS})
        add_subdirectory(${DIR})
    endforeach (DIR)
endif ()
# ======================================================================================================================

# Tests (if enabled) ===================================================================================================
if (${LUTHIER_BUILD_INTEGRATION_TESTS})
    add_subdirectory(tests)
endif ()

# Documentation Build With Doxygen =====================================================================================
if (${LUTHIER_BUILD_HTML_DOCS} OR ${LUTHIER_BUILD_LATEX_DOCS})
    find_package(Doxygen REQUIRED)
    set(DOXYGEN_GENERATE_HTML ${LUTHIER_BUILD_HTMLS_DOCS})
    set(DOXYGEN_GENERATE_LATEX ${LUTHIER_BUILD_LATEX_DOCS})
    doxygen_add_docs(luthier_docs docs src compiler-plugins)
endif ()


