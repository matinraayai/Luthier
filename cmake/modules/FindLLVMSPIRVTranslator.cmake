#===- FindLLVMSPIRVTranslator.cmake -------------------------------------===#
# Copyright @ Northeastern University Computer Architecture Lab
#
# Licensed under the Apache License, Version 2.0.
#===---------------------------------------------------------------------===#
#
# Locates the AMD SPIR-V translator header and library files, since it
# doesn't install any itself.
#
# Search roots are taken from the high-level, similar to how a normal
# `find_package` would perform a lookup: Every prefix listed in
# CMAKE_PREFIX_PATH is searched. If LUTHIER_LLVM_SPIRV_TRANSLATOR_DIR is
# specified, it will be given priority.
#
# * Result variables:
# - LUTHIER_LLVM_SPIRV_TRANSLATOR_FOUND: TRUE iff the translator binary, the
# LLVMSPIRVLib headers, and the translator library were all located.
# - LUTHIER_LLVM_SPIRV_TRANSLATOR: path to the
# amd-llvm-spirv (or llvm-spirv) binary.
# - LUTHIER_LLVM_SPIRV_TRANSLATOR_INCLUDE_DIR: directory containing
# LLVMSPIRVLib/LLVMSPIRVLib.h
# - LUTHIER_LLVM_SPIRV_TRANSLATOR_LIBRARY: path to the translator
# static/shared library.
#
# * Imported target (only defined when found):
# - Luthier::LLVMSPIRVTranslator: links
# LUTHIER_LLVM_SPIRV_TRANSLATOR_LIBRARY and adds
# LUTHIER_LLVM_SPIRV_TRANSLATOR_INCLUDE_DIR to the include path.
#===----------------------------------------------------------------------===#

# Allow the SPIR-V path to be turned off explicitly
if (DEFINED LUTHIER_ENABLE_SPIRV AND NOT LUTHIER_ENABLE_SPIRV)
    set(LLVMSPIRVTranslator_FOUND FALSE)
    set(LUTHIER_LLVM_SPIRV_TRANSLATOR_FOUND FALSE CACHE INTERNAL
            "Whether the AMD SPIR-V translator + LLVMSPIRVLib were found")
    return()
endif ()

# List of Candidate roots: the user-specified dir first, then every prefix
# listed in CMAKE_PREFIX_PATH.
set(_luthier_llvm_spirv_translator_roots)
if (LUTHIER_LLVM_SPIRV_TRANSLATOR_DIR)
    list(APPEND _luthier_llvm_spirv_translator_roots ${LUTHIER_LLVM_SPIRV_TRANSLATOR_DIR})
endif ()
if (CMAKE_PREFIX_PATH)
    list(APPEND _luthier_llvm_spirv_translator_roots ${CMAKE_PREFIX_PATH})
endif ()

find_program(LUTHIER_LLVM_SPIRV_TRANSLATOR
        NAMES amd-llvm-spirv llvm-spirv
        HINTS ${_luthier_llvm_spirv_translator_roots}
        PATH_SUFFIXES bin
        DOC "AMD SPIR-V translator binary (amd-llvm-spirv)")

find_path(LUTHIER_LLVM_SPIRV_TRANSLATOR_INCLUDE_DIR
        NAMES LLVMSPIRVLib/LLVMSPIRVLib.h
        HINTS ${_luthier_llvm_spirv_translator_roots}
        PATH_SUFFIXES include
        DOC "Directory containing the LLVMSPIRVLib headers")

find_library(LUTHIER_LLVM_SPIRV_TRANSLATOR_LIBRARY
        NAMES LLVMSPIRVAMDLib SPIRVAMDLib LLVMSPIRVLib
        HINTS ${_luthier_llvm_spirv_translator_roots}
        PATH_SUFFIXES lib lib64
        DOC "AMD SPIR-V translator library (LLVMSPIRVAMDLib)")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LLVMSPIRVTranslator
        REQUIRED_VARS
        LUTHIER_LLVM_SPIRV_TRANSLATOR
        LUTHIER_LLVM_SPIRV_TRANSLATOR_INCLUDE_DIR
        LUTHIER_LLVM_SPIRV_TRANSLATOR_LIBRARY)

set(LUTHIER_LLVM_SPIRV_TRANSLATOR_FOUND ${LLVMSPIRVTranslator_FOUND} CACHE INTERNAL
        "Whether the AMD SPIR-V translator + LLVMSPIRVLib were found")

if (LUTHIER_LLVM_SPIRV_TRANSLATOR_FOUND AND NOT TARGET Luthier::LLVMSPIRVTranslator)
    add_library(Luthier::LLVMSPIRVTranslator STATIC IMPORTED)
    set_target_properties(Luthier::LLVMSPIRVTranslator PROPERTIES
            IMPORTED_LOCATION "${LUTHIER_LLVM_SPIRV_TRANSLATOR_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${LUTHIER_LLVM_SPIRV_TRANSLATOR_INCLUDE_DIR}")
endif ()

mark_as_advanced(LUTHIER_LLVM_SPIRV_TRANSLATOR LUTHIER_LLVM_SPIRV_TRANSLATOR_INCLUDE_DIR
        LUTHIER_LLVM_SPIRV_TRANSLATOR_LIBRARY)
