#===- LuthierCreateOffloadBundle.cmake -----------------------------------------------------------------------------===#
# Copyright @ Northeastern University Computer Architecture Lab
#
# Licensed under the Apache License, Version 2.0.
#===----------------------------------------------------------------------------------------------------------------===#

include_guard(GLOBAL)


# Split one pipe-delimited target entry into three fields.
# - entry: target entry to be parsed, in the format of "<triple>|<cpu>[|<flags>]"
#   e.g. "amdgcn-amd-amdhsa-|gfx1036|-mwavefrontsize64"
# - out_triple: target triple of the entry e.g. amdgcn-amd-amdhsa-
# - out_cpu: Clang CPU specification e.g. gfx1036 (can be used to specify xnack and sramecc subtarget features)
# - out_flags: Additional compilation flags for the slice. Primarily used to specify subtarget features in clang that
#   have an cannot be controlled via the CPU spec e.g. -mwavefrontsize64. Can also be used to specify compilation flags
#   that have nothing to do with subtarget specifications.
function(luthier_split_target_entry entry out_triple out_cpu out_flags)
  string(REPLACE "|" ";" _FIELDS "${entry}")
  list(LENGTH _FIELDS _ENTRY_LIST_LEN)
  if (_ENTRY_LIST_LEN LESS 2)
    message(FATAL_ERROR
            "Target entry '${entry}' must be formatted as '<triple>|<cpu>[|<flags>]'.")
  endif ()
  list(GET _FIELDS 0 _TRIPLE)
  list(GET _FIELDS 1 _CPU)
  if (_TRIPLE STREQUAL "" OR _CPU STREQUAL "")
    message(FATAL_ERROR
            "Target entry '${entry}' has an empty triple or cpu field.")
  endif ()
  set(_FLAGS "")
  if (_ENTRY_LIST_LEN GREATER 2)
    list(GET _FIELDS 2 _FLAGS)
  endif ()
  # Split the flags field into individual, space-separated arguments.
  separate_arguments(_FLAG_LIST UNIX_COMMAND "${_FLAGS}")
  set(${out_triple} "${_TRIPLE}" PARENT_SCOPE)
  set(${out_cpu} "${_CPU}" PARENT_SCOPE)
  set(${out_flags} "${_FLAG_LIST}" PARENT_SCOPE)
endfunction()


# Builds an offload bundle object from a single hip source.
# Unlike normal HIP compilation:
# - The LLVM bitcode or SPIR-V file of the device logic is embedded in the FAT binary instead of its shared object.
#   SPIR-V file of the device logic if SPIR-V LLVM translator is enabled.
# - The Luthier IR compiler plugin is applied to the device code's compilation process for bundled LLVM bitcode slices
#   (SPIR-V files will have to apply them at runtime when JIT-ing for their concrete target).
# - The host portion is compiled with the Luthier CXX and the IR compiler plugins.
# This utility exists because there is no way to invoke clang directly to embed:
# - IR files instead of code objects.
# - Multiple entries for the same triple/cpu with different subtarget features (e.g. `wavefrontsize`, `cumode`).
# The HIP file will have the following targets generated for it:
#   - One HIP OBJECT library per specified device target, which generate the associated LLVM bitcode or SPIR-V file.
#     Note that CMake adds the `.o` suffix to the outputs of the device-side compilation despite them not being object
#     files.
#   - A bundle target, which packs every bitcode or SPIR-V slice into a single `.hipfb` clang offload bundle.
#   - A HIP OBJECT library host target, which compiles the host side of the same source with the produced `.hipfb`
#     spliced in.
# Both the device and the host targets are unlinked OBJECT libraries, and can be returned to the caller for further
# customization.
#
# - target: The host file's OBJECT library to be created by this function.
# - source: The HIP source file.
# - [TARGET_ISAS entry...]: Targets to compile for, one pipe-delimited entry per slice: `<triple>|<gpu>[|<flags>]`.
#   If not specified, this value will be taken from the parent scope's LUTHIER_HIP_TARGETS variable. If that is also
#   not set, this value is constructed automatically from the `CMAKE_HIP_ARCHITECTURES` without adding any additional
#   flags. If SPIRV is enabled, an AMD-flavored compute SPIR-V slice is also added to the offload bundle. Identical
#   target strings is accepted and will only be compiled once to keep the offload bundler happy. The flags are fed
#   "as is" to the compiler, and no de-duplication is performed on them (e.g. passing both
#   "amdgcn-amd-amdhsa-|gfx1036|-mwavefrontsize64 -mcumode" and "amdgcn-amd-amdhsa-|gfx1036|-mcumode -mwavefrontsize64"
#   will result in emission of two distinct slices). It is the loader's responsibility to take this behavior into
#   account.
# - [DEVICE_OBJECT_LIBRARIES var]: list of per-slice device OBJECT libraries.
# - [BUNDLE_TARGET <var>]: The custom target that builds the .hipfb.
function(luthier_create_offload_bundle target source)
  cmake_parse_arguments(OFFLOAD_BUNDLE_ARG ""
          "DEVICE_OBJECT_LIBRARIES;BUNDLE_TARGET"
          "TARGET_ISAS"
          ${ARGN})

  if (NOT source)
    message(FATAL_ERROR
            "luthier_create_offload_bundle(${target}): No HIP source file was specified.")
  endif ()
  # Reject any stray extra positional/keyword here.
  if (OFFLOAD_BUNDLE_ARG_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
            "luthier_create_offload_bundle(${target}): unexpected argument(s): "
            "${OFFLOAD_BUNDLE_ARG_UNPARSED_ARGUMENTS}.")
  endif ()

  # Resolve the AMDGCN target list: per-call TARGET_ISAS > LUTHIER_HIP_TARGETS > derived from CMAKE_HIP_ARCHITECTURES.
  # An empty result is fine as long as the amdgcnspirv slice is emitted (SPIR-V found); if both are empty we error out
  # below rather than bundle nothing.
  set(_DEVICE_ISA_TARGETS "")
  if (OFFLOAD_BUNDLE_ARG_TARGET_ISAS)
    set(_DEVICE_ISA_TARGETS "${OFFLOAD_BUNDLE_ARG_TARGET_ISAS}")
  elseif (LUTHIER_HIP_TARGETS)
    set(_DEVICE_ISA_TARGETS "${LUTHIER_HIP_TARGETS}")
  elseif (CMAKE_HIP_ARCHITECTURES)
    foreach (_ARCH IN LISTS CMAKE_HIP_ARCHITECTURES)
      list(APPEND _DEVICE_ISA_TARGETS "amdgcn-amd-amdhsa-|${_ARCH}")
    endforeach ()
  endif ()

  # Source-file naming → intermediates / fatbin.
  get_filename_component(_prefix "${source}" NAME_WE)
  set(_TARGET_FATBIN "${CMAKE_CURRENT_BINARY_DIR}/${target}.${_prefix}.hipfb")

  # Absolute source path for the downstream add_library / copy logic.
  if (IS_ABSOLUTE "${source}")
    set(_ABS_SOURCE "${source}")
  else ()
    set(_ABS_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/${source}")
  endif ()

  # The device-slice OBJECT libraries compile a COPY of the source, kept apart from the original the host compiles. The
  # host source carries an OBJECT_DEPENDS on the fat binary (so it recompiles when the bundle changes); source-file
  # properties are directory-scoped, so if the device slices shared that source they would inherit the OBJECT_DEPENDS
  # and form a build cycle. Compiling a copy breaks the share.
  get_filename_component(_SOURCE_NAME ${_ABS_SOURCE} NAME)
  set(_DEV_SOURCE "${CMAKE_CURRENT_BINARY_DIR}/${target}.dev_tu/${_SOURCE_NAME}")
  configure_file(${_ABS_SOURCE} ${_DEV_SOURCE} COPYONLY)
  # Compile the copy as HIP regardless of its extension (the host source is likewise marked LANGUAGE HIP at the
  # host-compile step below).
  set_source_files_properties(${_DEV_SOURCE} PROPERTIES LANGUAGE HIP)

  # Locate the plugins.
  if (TARGET LuthierToolIRCompilationPlugin)
    set(_LUTHIER_IR_PLUGIN "$<TARGET_FILE:LuthierToolIRCompilationPlugin>")
    set(_LUTHIER_IR_PLUGIN_TARGET LuthierToolIRCompilationPlugin)
  elseif (TARGET luthier::LuthierToolIRCompilationPlugin)
    set(_LUTHIER_IR_PLUGIN "$<TARGET_FILE:luthier::LuthierToolIRCompilationPlugin>")
    set(_LUTHIER_IR_PLUGIN_TARGET luthier::LuthierToolIRCompilationPlugin)
  else ()
    message(FATAL_ERROR
            "luthier_create_offload_bundle(${target}): "
            "LuthierToolIRCompilationPlugin not found.")
  endif ()

  if (TARGET LuthierToolCXXCompilationPlugin)
    set(_LUTHIER_CXX_PLUGIN "$<TARGET_FILE:LuthierToolCXXCompilationPlugin>")
    set(_LUTHIER_CXX_PLUGIN_TARGET LuthierToolCXXCompilationPlugin)
  elseif (TARGET luthier::LuthierToolCXXCompilationPlugin)
    set(_LUTHIER_CXX_PLUGIN "$<TARGET_FILE:luthier::LuthierToolCXXCompilationPlugin>")
    set(_LUTHIER_CXX_PLUGIN_TARGET luthier::LuthierToolCXXCompilationPlugin)
  else ()
    message(FATAL_ERROR
            "luthier_create_offload_bundle(${target}): "
            "LuthierToolCXXCompilationPlugin not found")
  endif ()


  # Resolve clang-offload-bundler path from LLVM's tool directory.
  find_program(LUTHIER_CLANG_OFFLOAD_BUNDLER
          NAMES clang-offload-bundler
          HINTS ${LLVM_TOOLS_BINARY_DIR}
          DOC "clang-offload-bundler used by luthier_create_offload_bundle")
  if (NOT LUTHIER_CLANG_OFFLOAD_BUNDLER)
    message(FATAL_ERROR
            "luthier_create_offload_bundle(${target}): clang-offload-bundler not found in LLVM_TOOLS_BINARY_DIR "
            "('${LLVM_TOOLS_BINARY_DIR}').")
  endif ()

  # Enable SPIR-V compilation if requested
  if (${LUTHIER_ENABLE_SPIRV})
    find_package(LLVMSPIRVTranslator)
    get_filename_component(_SPIRV_DIR "${LUTHIER_LLVM_SPIRV_TRANSLATOR}" DIRECTORY)
  endif ()

  #---------------------------------------------------------------------------------------------------------------------
  # Device Targets
  #---------------------------------------------------------------------------------------------------------------------
  set(_SLICE_OBJS "")
  set(_REBUNDLE_SLICE_INPUTS "")
  set(_REBUNDLE_TARGET_ISAS "")
  set(_DEV_TARGETS "")
  set(_SEEN_KEYS "")
  foreach (_ENTRY IN LISTS _DEVICE_ISA_TARGETS)
    luthier_split_target_entry("${_ENTRY}" _TRIPLE _GPU _EXTRA_FLAGS)

    # clang-offload-bundler checks the AMDGPU ISA labels handed to it. To keep it happy we encode the extra flags as
    # as synthetic target feature encoded as ":<sanitized-extra-flags>+".
    set(_FLAG_SUFFIX "")
    if (_EXTRA_FLAGS)
      string(JOIN "_" _FLAG_SUFFIX ${_EXTRA_FLAGS})
      string(REGEX REPLACE "[^A-Za-z0-9]+" "_" _FLAG_SUFFIX "${_FLAG_SUFFIX}")
      string(REGEX REPLACE "^_+|_+$" "" _FLAG_SUFFIX "${_FLAG_SUFFIX}")
      set(_FLAG_SUFFIX ":${_FLAG_SUFFIX}+")
    endif ()
    set(_BUNDLE_KEY "${_TRIPLE}-${_GPU}${_FLAG_SUFFIX}")
    if (_BUNDLE_KEY IN_LIST _SEEN_KEYS)
      continue()
    endif ()
    list(APPEND _SEEN_KEYS "${_BUNDLE_KEY}")

    # Sanitize the bundle key into a valid CMake target-name suffix: spell the subtarget feature signs as _on/_off and
    # turn the remaining ':' / '+' separators into '_'. The feature-anchored +/- replacements leave the triple's dashes
    # (amdgcn-amd-amdhsa-) intact.
    set(_SANITIZED "${_BUNDLE_KEY}")
    string(REGEX REPLACE "(xnack|sramecc|wavefrontsize64|cumode)\\+" "\\1_on"
            _SANITIZED "${_SANITIZED}")
    string(REGEX REPLACE "(xnack|sramecc|wavefrontsize64|cumode)-" "\\1_off"
            _SANITIZED "${_SANITIZED}")
    string(REGEX REPLACE "[:+]" "_" _SANITIZED "${_SANITIZED}")

    set(_SLICE_TGT "${target}-${_SANITIZED}")
    add_library(${_SLICE_TGT} OBJECT ${_DEV_SOURCE})
    set_target_properties(${_SLICE_TGT} PROPERTIES HIP_ARCHITECTURES "${_GPU}")
    # FIXME: -g0: disable debug info for now
    target_compile_options(${_SLICE_TGT} PRIVATE
            --cuda-device-only -emit-llvm --no-gpu-bundle-output -g0
            ${_EXTRA_FLAGS} -fpass-plugin=${_LUTHIER_IR_PLUGIN})
    add_dependencies(${_SLICE_TGT} ${_LUTHIER_IR_PLUGIN_TARGET})

    list(APPEND _DEV_TARGETS "${_SLICE_TGT}")
    list(APPEND _SLICE_OBJS "$<TARGET_OBJECTS:${_SLICE_TGT}>")
    list(APPEND _REBUNDLE_SLICE_INPUTS "--input=$<TARGET_OBJECTS:${_SLICE_TGT}>")
    list(APPEND _REBUNDLE_TARGET_ISAS "hipv4-${_BUNDLE_KEY}")
  endforeach ()

  # Add an AMD SPIR-V slice if requested
  if (${LUTHIER_LLVM_SPIRV_TRANSLATOR_FOUND})
    set(_SPIRV_TARGET_ISA "hip-spirv64-amd-amdhsa--amdgcnspirv")
    set(_SPIRV_TARGET "${target}-${_SPIRV_TARGET_ISA}")
    add_library(${_SPIRV_TARGET} OBJECT ${_DEV_SOURCE})
    set_target_properties(${_SPIRV_TARGET} PROPERTIES HIP_ARCHITECTURES "amdgcnspirv")
    # FIXME: -g0: disable debug info for now
    # -U SPIRV is added to undefine the SPIRV definition added by the SPIRV translator that clashes with LLVM's SPIRV
    # target Triple
    target_compile_options(${_SPIRV_TARGET} PRIVATE
            --cuda-device-only --no-gpu-bundle-output -g0 -B "${_SPIRV_DIR}"
            -fpass-plugin=${_LUTHIER_IR_PLUGIN} -U SPIRV)
    add_dependencies(${_SPIRV_TARGET} ${_LUTHIER_IR_PLUGIN_TARGET})

    list(APPEND _DEV_TARGETS "${_SPIRV_TARGET}")
    list(APPEND _SLICE_OBJS "$<TARGET_OBJECTS:${_SPIRV_TARGET}>")
    list(APPEND _REBUNDLE_SLICE_INPUTS "--input=$<TARGET_OBJECTS:${_SPIRV_TARGET}>")
    list(APPEND _REBUNDLE_TARGET_ISAS "${_SPIRV_TARGET_ISA}")
  endif ()

  if (NOT _SLICE_OBJS)
    message(FATAL_ERROR
            "luthier_create_offload_bundle(${target}): no device slices to bundle — the resolved target list is empty "
            "and the SPIR-V slice is unavailable. Set CMAKE_HIP_ARCHITECTURES / LUTHIER_HIP_TARGETS / TARGET_ISAS, or "
            "enable SPIR-V via LUTHIER_LLVM_SPIRV_TRANSLATOR_DIR.")
  endif ()

  # Join the list of target ISAs for the bundle target argument
  list(JOIN _REBUNDLE_TARGET_ISAS "," _REBUNDLE_TARGET_ISAS)
  #---------------------------------------------------------------------------------------------------------------------
  # Bundle the device slices (bitcode for the AMDGCN targets, SPIR-V for the amdgcnspirv target)  into the final
  #.hipfb — the one and only packing step. --bundle-align=8 is required by the LLVM offload parser.
  #---------------------------------------------------------------------------------------------------------------------

  add_custom_command(
          OUTPUT "${_TARGET_FATBIN}"
          COMMAND "${LUTHIER_CLANG_OFFLOAD_BUNDLER}" --type=o
          --targets=${_REBUNDLE_TARGET_ISAS}
          ${_REBUNDLE_SLICE_INPUTS}
          --output=${_TARGET_FATBIN} --bundle-align=8
          DEPENDS ${_SLICE_OBJS}
          COMMENT "luthier_create_offload_bundle(${target}): bundle .hipfb"
          VERBATIM COMMAND_EXPAND_LISTS)

  add_custom_target(${target}-fatbin DEPENDS "${_TARGET_FATBIN}" ${_DEV_TARGETS})

  #---------------------------------------------------------------------------------------------------------------------
  # Host compile → OBJECT library.
  #---------------------------------------------------------------------------------------------------------------------
  add_library(${target} OBJECT ${_ABS_SOURCE})
  set_source_files_properties(${_ABS_SOURCE} PROPERTIES
          LANGUAGE HIP
          OBJECT_DEPENDS "${_TARGET_FATBIN}")
  set_target_properties(${target} PROPERTIES HIP_ARCHITECTURES OFF)

  target_compile_options(${target} PRIVATE
          --cuda-host-only -fno-gpu-rdc -fuse-cuid=none
          "SHELL:-Xclang -fcuda-include-gpubinary -Xclang ${_TARGET_FATBIN}"
          -fpass-plugin=${_LUTHIER_IR_PLUGIN}
          -fplugin=${_LUTHIER_CXX_PLUGIN}
          "SHELL:-Xclang -add-plugin -Xclang luthier-emit-device-function-host-handle")

  add_dependencies(${target}
          ${target}-fatbin
          ${_LUTHIER_IR_PLUGIN_TARGET}
          ${_LUTHIER_CXX_PLUGIN_TARGET})

  # Hand the created targets back to the caller if requested.
  if (OFFLOAD_BUNDLE_ARG_DEVICE_OBJECT_LIBRARIES)
    set(${OFFLOAD_BUNDLE_ARG_DEVICE_OBJECT_LIBRARIES} "${_DEV_TARGETS}" PARENT_SCOPE)
  endif ()
  if (OFFLOAD_BUNDLE_ARG_BUNDLE_TARGET)
    set(${OFFLOAD_BUNDLE_ARG_BUNDLE_TARGET} "${target}-fatbin" PARENT_SCOPE)
  endif ()
endfunction()
