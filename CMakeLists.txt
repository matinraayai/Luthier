cmake_minimum_required(VERSION 3.21)

project(luthier
        VERSION 0.1.0
        LANGUAGES CXX
        DESCRIPTION "Dynamic Binary Instrumentation for AMD GPUs"
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(FetchContent)

# Conditional to check if LLVM_SRC_DIR exists somewhere or not
# Check if LUTHIER_LLVM_SRC_DIR is empty string or just a folder
if (NOT IS_DIRECTORY LUTHIER_LLVM_SRC_DIR)

  # Fetches GIT_HASH variable from VCS_Revision.h
  set(HEADER_FILE "/opt/luthier/llvm/include/llvm/Support/VCSRevision.h")
  file(STRINGS ${HEADER_FILE} LLVM_REVISION_LINE REGEX "^#define LLVM_REVISION")
  string(REGEX MATCH "LLVM_REVISION[ \t]+\"([^\"]+)\"" _ ${LLVM_REVISION_LINE})
  set(GIT_HASH ${CMAKE_MATCH_1})
  
  # Checks GIT_HASH
  message(STATUS "Git hash retrieved: ${GIT_HASH}")
  
  # Clones llvm-project branch used for build
  FetchContent_Declare(
    llvm_src
    URL https://github.com/llvm/llvm-project/archive/${GIT_HASH}.zip
  )
  FetchContent_MakeAvailable(llvm_src)

  # Provide source directory location
  FetchContent_GetProperties(llvm_src)

  # Checks SOURCE_DIR name
  message(STATUS "VARIABLE_NAME is set to ${llvm_src_SOURCE_DIR}")
    
  # Sets env variable, caching it for installation
  set(LUTHIER_LLVM_SRC_DIR ${llvm_src_SOURCE_DIR} CACHE STRING "Path to the LLVM source directory")
  message(STATUS "VARIABLE_NAME is set to ${LUTHIER_LLVM_SRC_DIR}")
  
endif()

# User Options =========================================================================================================
option(LUTHIER_BUILD_INTEGRATION_TESTS "Builds the integration tests" OFF)
option(LUTHIER_BUILD_EXAMPLES "Builds the example tools in the examples/ folder" ON)
option(LUTHIER_BUILD_LATEX_DOCS "Builds Luthier documentation with Doxygen in PDF format with Latex" OFF)
option(LUTHIER_BUILD_HTML_DOCS "Builds Luthier documentation with Doxygen in HTML format" OFF)
# TODO: Besides manual specification of the LLVM src code, also provide the option of cloning the correct source
# depending on the LLVM version imported by the user
set(LUTHIER_TOOLING_LLVM_SRC_DIR "" CACHE STRING "Path to the source code of the ROCm LLVM project used; Required.")
# ======================================================================================================================

# Import modules
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake ${PROJECT_SOURCE_DIR}/cmake/module ${CMAKE_MODULE_PATH})

# Add the sub projects
add_subdirectory(luthier-tblgen)
add_subdirectory(compiler-plugins)
add_subdirectory(src)

#TESTED ADDITION TO SEE IF IT WORKS WITH CMAKE --INSTALL
add_subdirectory(cmake/modules)

# Any Compilation Flags, Linking Configuration Left
# TODO: Add option to build with sanitizers
#target_compile_options(luthier PUBLIC "-fsanitize=address")

# Build examples if enabled
if (${LUTHIER_BUILD_EXAMPLES})
    add_subdirectory(examples)
endif ()

# Build integration tests if enabled
if (${LUTHIER_BUILD_INTEGRATION_TESTS})
    add_subdirectory(tests)
endif ()

# Build documentation with Doxygen if enabled
if (${LUTHIER_BUILD_HTML_DOCS} OR ${LUTHIER_BUILD_LATEX_DOCS})
    find_package(Doxygen REQUIRED)
    set(DOXYGEN_GENERATE_HTML ${LUTHIER_BUILD_HTMLS_DOCS})
    set(DOXYGEN_GENERATE_LATEX ${LUTHIER_BUILD_LATEX_DOCS})
    doxygen_add_docs(luthier_docs docs src compiler-plugins luthier-tblgen)
endif ()



