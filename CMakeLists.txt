cmake_minimum_required(VERSION 3.21)

project(luthier
        VERSION 0.1.0
        LANGUAGES CXX
        DESCRIPTION "Dynamic Binary Instrumentation for AMD GPUs"
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
#Include the ExternalProject regardless if necessary or not
include(ExternalProject)
#Conditional to check if LLVM_SRC_DIR exists somewhere or not
if(NOT DEFINED LUTHIER_LLVM_SRC_DIR)
  #Case where LLVM_SRC_DIR is not given either
  if(NOT DEFINED LLVM_SRC_DIR)
    #Set LLVM_SRC_DIR if not already
    set(LLVM_SRC_DIR /usr/local/install_proc)

    #Execute Process at Compile Time
    #find_package(Git REQUIRED)  # Ensure Git is available
    #execute_process(
    #  COMMAND ${GIT_EXECUTABLE} clone --branch 5e51f7702e2703df95f7a3d57284a1fdef4766b7 git@github.com:ROCm/llvm-project.git ${LLVM_SRC_DIR}
    #  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    #  RESULT_VARIABLE GIT_CLONE_RESULT
    #)
    #This is at build time
    ExternalProject_Add(llvm-project
      GIT_REPOSITORY    git@github.com:ROCm/llvm-project.git
      GIT_TAG           5e51f7702e2703df95f7a3d57284a1fdef4766b7
      GIT_SHALLOW       1
      PREFIX ${LLVM_SRC_DIR}
    )

    #Add Custom Target which all are dependent on
    add_custom_target(llvm-project-clone ALL DEPENDS llvm-project)

    #Set LuthierLLVMSRCDIR to something else?
    message(STATUS "VARIABLE_NAME is set to ${LLVM_SRC_DIR}")
    set(LUTHIER_LLVM_SRC_DIR ${LLVM_SRC_DIR})
    message(STATUS "VARIABLE_NAME is set to ${LUTHIER_LLVM_SRC_DIR}")

    #Set CMAKE_PREFIX_PATH to something else?
    #set(CMAKE_PREFIX_PATH "/opt/luthier;/opt/rocm")

    #message(STATUS "VARIABLE_NAME is set to ${CMAKE_PREFIX_PATH}")
  else()
    #Case where LLVM_SRC_DIR is already set, so LUTHIER gets set to it
    set(LUTHIER_LLVM_SRC_DIR ${LLVM_SRC_DIR})
  endif()
  
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

#Adding dependencies to custom target
add_dependencies(LUTHIERAMDGPU llvm-project-clone)

