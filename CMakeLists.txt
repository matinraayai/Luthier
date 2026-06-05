cmake_minimum_required(VERSION 3.21)

project(luthier
        VERSION 0.1.0
        LANGUAGES CXX
        DESCRIPTION "Dynamic Binary Instrumentation for AMD GPUs"
        HOMEPAGE_URL "https://github.com/matinraayai/Luthier"
)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# User Options =========================================================================================================
option(LUTHIER_BUILD_INTEGRATION_TESTS "Builds the integration tests" OFF)
option(LUTHIER_BUILD_UNIT_TESTS "Builds the unit tests" OFF)
option(LUTHIER_BUILD_EXAMPLES "Builds the example tools in the examples/ folder" ON)
option(LUTHIER_BUILD_LATEX_DOCS "Builds Luthier documentation with Doxygen in PDF format with Latex" OFF)
option(LUTHIER_BUILD_HTML_DOCS "Builds Luthier documentation with Doxygen in HTML format" OFF)
set(LUTHIER_LLVM_SRC_DIR "" CACHE STRING "Path to the source code of the ROCm LLVM project used to build Luthier")
set(LUTHIER_HIP_TARGETS "" CACHE STRING "Semicolon-separated list of target ISAs \
    (triple + processor + subtarget features) to precompile Luthier HIP offload bundle files for; \
    When empty, it is derived from CMAKE_HIP_ARCHITECTURES. Excludes SPIR-V.")
set(LUTHIER_LLVM_SPIRV_TRANSLATOR_PREFIX_PATH "" CACHE STRING "Semicolon-separated list of candidate \
    install prefixes searched (under bin/include/lib) for the AMD SPIR-V translator (amd-llvm-spirv) \
    and its LLVMSPIRVLib headers/library. Optional; Will disable SPIR-V to LLVM AMD HSA translation \
    if not found.")
# ======================================================================================================================

# Append the modules
list(INSERT CMAKE_MODULE_PATH 0 "${PROJECT_SOURCE_DIR}/cmake/modules")

# Enable CTest from the build root so `ctest` aggregates tests registered in
# all subdirectories
enable_testing()

# Add the src sub projects
add_subdirectory(src)

# Add cmake installation config
add_subdirectory(cmake/modules)

# Any Compilation Flags, Linking Configuration Left
# TODO: Add option to build with sanitizers
#target_compile_options(luthier PUBLIC "-fsanitize=address")

# Build examples if enabled
if (${LUTHIER_BUILD_EXAMPLES})
    add_subdirectory(examples)
endif ()

# Build documentation with Doxygen if enabled
if (${LUTHIER_BUILD_HTML_DOCS} OR ${LUTHIER_BUILD_LATEX_DOCS})
    find_package(Doxygen REQUIRED)
    set(DOXYGEN_GENERATE_HTML ${LUTHIER_BUILD_HTMLS_DOCS})
    set(DOXYGEN_GENERATE_LATEX ${LUTHIER_BUILD_LATEX_DOCS})
    doxygen_add_docs(LuthierDocs docs src)
endif ()



