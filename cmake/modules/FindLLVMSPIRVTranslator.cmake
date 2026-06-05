#===- FindLLVMSPIRVTranslator.cmake -------------------------------------===#
# Copyright @ Northeastern University Computer Architecture Lab
#
# Licensed under the Apache License, Version 2.0.
#===---------------------------------------------------------------------===#
#
# Locates the AMD SPIR-V translator and its LLVMSPIRVLib development files.
#
# Search roots are taken from the high-level, user-facing cache variable
# LUTHIER_LLVM_SPIRV_TRANSLATOR_PREFIX_PATH (a list of candidate install prefixes),
# plus the LLVM install tree as a fallback. Under each root the standard
# bin/include/lib layout is searched.
#
# Result variables:
#   LUTHIER_LLVM_SPIRV_TRANSLATOR_FOUND          - TRUE iff the translator binary, the
#                                  LLVMSPIRVLib headers, and the translator
#                                  library were all located.
#   LUTHIER_LLVM_SPIRV_TRANSLATOR     - path to the amd-llvm-spirv (or llvm-spirv)
#                                  binary (used to translate device IR to
#                                  SPIR-V during the offload-bundle compile).
#   LUTHIER_LLVM_SPIRV_TRANSLATOR_INCLUDE_DIR    - directory containing LLVMSPIRVLib/LLVMSPIRVLib.h
#   LUTHIER_LLVM_SPIRV_TRANSLATOR_LIBRARY        - path to the translator static/shared library.
#
# Imported target (only defined when found):
#   Luthier::LLVMSPIRVTranslator               - links LUTHIER_LLVM_SPIRV_TRANSLATOR_LIBRARY and adds
#                                  LUTHIER_LLVM_SPIRV_TRANSLATOR_INCLUDE_DIR to the include path.
#===----------------------------------------------------------------------===#

# Candidate roots: user-provided prefixes first, then the LLVM install tree.
set(_luthier_llvm_spirv_translator_roots)
if (LUTHIER_LLVM_SPIRV_TRANSLATOR_PREFIX_PATH)
    list(APPEND _luthier_llvm_spirv_translator_roots ${LUTHIER_LLVM_SPIRV_TRANSLATOR_PREFIX_PATH})
endif ()
if (DEFINED LLVM_TOOLS_BINARY_DIR)
    # …/<root>/bin -> <root>, and the bin dir itself.
    get_filename_component(_luthier_llvm_spirv_translator_llvm_root "${LLVM_TOOLS_BINARY_DIR}" DIRECTORY)
    get_filename_component(_luthier_llvm_spirv_translator_inst_root "${_luthier_llvm_spirv_translator_llvm_root}" DIRECTORY)
    list(APPEND _luthier_llvm_spirv_translator_roots
            "${_luthier_llvm_spirv_translator_llvm_root}" "${_luthier_llvm_spirv_translator_inst_root}")
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

# Re-export the FPHSA <pkg>_FOUND result under the LUTHIER_ name the rest of the
# build uses, and cache it so it is visible in subdirectories (e.g. examples/).
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
