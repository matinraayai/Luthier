include_guard(GLOBAL)

#---------------------------------------------------------------------------
# Extract the following from the isa_string:
#   <isa_string> e.g. amdgcn-amd-amdhsa--gfx942:xnack+:wavefrontsize64-:cumode-
#   out_triple e.g. amdgcn-amd-amdhsa
#   out_offload_arch  offload architecture value for clang: proc[:sramecc±][:xnack±]
#   out_mflags   standalone clang flags: -m[no-]wavefrontsize64 / -m[no-]cumode
#---------------------------------------------------------------------------
function(luthier_parse_isa_string isa_string out_triple out_offload_arch out_mflags)
  # Split triple from the target-ID at the empty-env "--".
  string(FIND "${isa_string}" "--" _SEP)
  if (_SEP EQUAL -1)
    message(FATAL_ERROR
            "luthier_create_offload_bundle: target '${isa_string}' is missing the "
            "'--<processor>' suffix (expected e.g. amdgcn-amd-amdhsa--gfx942).")
  endif ()
  string(SUBSTRING "${isa_string}" 0 ${_SEP} _TRIPLE)
  math(EXPR _REST_START "${_SEP} + 2")
  string(SUBSTRING "${isa_string}" ${_REST_START} -1 _REST)
  # Fail anything other than amdhsa for now.
  if (NOT _TRIPLE STREQUAL "amdgcn-amd-amdhsa")
    message(FATAL_ERROR
            "luthier_create_offload_bundle: target '${isa_string}' has unsupported "
            "triple '${_TRIPLE}' (only amdgcn-amd-amdhsa is supported for now).")
  endif ()

  # proc + feature tokens.
  string(REPLACE ":" ";" _TOKS "${_REST}")
  list(GET _TOKS 0 _PROC)
  list(REMOVE_AT _TOKS 0)

  set(_XNACK "")
  set(_SRAMECC "")
  set(_MFLAGS "")
  foreach (_T IN LISTS _TOKS)
    string(REGEX MATCH "^(xnack|sramecc|wavefrontsize64|cumode)([+-])$" _M "${_T}")
    if (NOT _M)
      message(FATAL_ERROR
              "luthier_create_offload_bundle: target '${isa_string}' has unknown or "
              "malformed feature '${_T}' (expected <name>+ / <name>- where name "
              "is xnack | sramecc | wavefrontsize64 | cumode).")
    endif ()
    set(_NAME "${CMAKE_MATCH_1}")
    set(_SIGN "${CMAKE_MATCH_2}")
    if (_NAME STREQUAL "xnack")
      set(_XNACK ":xnack${_SIGN}")
    elseif (_NAME STREQUAL "sramecc")
      set(_SRAMECC ":sramecc${_SIGN}")
    elseif (_NAME STREQUAL "wavefrontsize64")
      if (_SIGN STREQUAL "+")
        list(APPEND _MFLAGS -mwavefrontsize64)
      else ()
        list(APPEND _MFLAGS -mno-wavefrontsize64)
      endif ()
    else () # cumode
      if (_SIGN STREQUAL "+")
        list(APPEND _MFLAGS -mcumode)
      else ()
        list(APPEND _MFLAGS -mno-cumode)
      endif ()
    endif ()
  endforeach ()

  set(${out_offload_arch} "${_PROC}${_SRAMECC}${_XNACK}" PARENT_SCOPE)
  set(${out_triple} "${_TRIPLE}" PARENT_SCOPE)
  set(${out_mflags} "${_MFLAGS}" PARENT_SCOPE)
endfunction()


# Builds an offload bundle object from a single hip source for use with
# instrumentation passes in Luthier.
# Unlike normal HIP compilation:
# - The LLVM bitcode or SPIR-V file of the device logic is embedded in the
#   FAT binary instead of its shared object. SPIR-V file of the device logic
#   is only emitted if the ROCm fork of the SPIR-V LLVM translator is found
#   or is provided to cmake.
# - The Luthier IR compiler plugin is applied to the device code's compilation
#   process for bundled LLVM bitcode slices but not the bundled SPIR-V files
#   (hence the parsing logic must first apply Luthier's tool device compilation
#   process itself).
# - The host portion is compiled with the Luthier CXX and the IR compiler plugins
#   applied.
# Note that this utility is necessary because:
# - There is no way to have clang automatically emit the bitcode and SPIR-V file
#   of the device logic into the FAT binary.
# - There is also no way to have clang automatically embed the device code for the
#   same architecture but with different wavefront or cu modes.
# The HIP file will have the following targets generated for it:
#   * Device: one HIP OBJECT library per Luthier target compiles the source via
#     CMake's native HIP language to generate the LLVM bitcode or SPIR-V file.
#     NOTE: CMake names these objects `*.o`, but they are NOT object files —
#     they are bitcode / SPIR-V, fed only to clang-offload-bundler (which keys on
#     content, not extension) and never linked.
#   * Bundle: packs every bitcode or SPIR-V slice into a single `.hipfb` clang
#     offload bundle.
#   * Host: compiles the host side of the same sources through CMake's native
#     HIP language with the produced `.hipfb` spliced in via
#     `-fcuda-include-gpubinary`, and the Luthier CXX and IR pass plugins.
# Both the device and the host targets are unlinked OBJECT libraries, and can
# be returned to the caller for further customization of their targets e.g.
# adding include directories and link libraries.
#
# Synopsis:
#
#   luthier_create_offload_bundle(<target> <source>
# --- inputs:
#     <target>:                   # The host file's OBJECT library to be created
#                                 # by this function.
#     <source>:                   # exactly one HIP source file,
#                                 # passed positionally right after <target>
#                                 # (single-TU tool).
#     [TARGET_ISAS <isa...>]      # complete offload target IDs to compile
#                                 # for, each `triple--proc[:feat±...]`, e.g.
#                                 # amdgcn-amd-amdhsa--gfx942:xnack-. Overrides
#                                 # LUTHIER_HIP_TARGETS for this call. When
#                                 # neither is set, derived from
#                                 # CMAKE_HIP_ARCHITECTURES (one bare target
#                                 # per arch). xnack/sramecc become target-ID
#                                 # feature suffixes on --offload-arch;
#                                 # wavefrontsize64/cumode become standalone
#                                 # -m flags.
#     [BUNDLER <path>]            # override the clang-offload-bundler
#                                 # path.
# --- outputs:
#     [DEVICE_OBJECT_LIBRARIES <var>] # list of per-slice device OBJECT libraries
#     [BUNDLE_TARGET <var>])          # the custom target that builds the .hipfb
#
#
# Requirements:
#   * `project(... LANGUAGES HIP)`
#   * The IR/CXX compilation plugins must be visible (in-tree via Luthier's
#     own build, or imported via `find_package(luthier ...)`).
#   * OPTIONAL: the AMD SPIR-V translator (FindLLVMSPIRVTranslator.cmake / the
#     LUTHIER_LLVM_SPIRV_TRANSLATOR_PREFIX_PATH cache var). When absent the
#     amdgcnspirv slice is simply omitted.
#===----------------------------------------------------------------------===#
function(luthier_create_offload_bundle target source)
  cmake_parse_arguments(OFFLOAD_BUNDLE_ARG ""
          "BUNDLER;DEVICE_OBJECT_LIBRARIES;BUNDLE_TARGET"
          "TARGET_ISAS"
          ${ARGN})

  if (NOT source)
    message(FATAL_ERROR
            "luthier_create_offload_bundle(${target}): a HIP source file must be "
            "passed immediately after <target>.")
  endif ()
  # Reject any stray extra positional/keyword here.
  if (OFFLOAD_BUNDLE_ARG_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
            "luthier_create_offload_bundle(${target}): unexpected argument(s): "
            "${OFFLOAD_BUNDLE_ARG_UNPARSED_ARGUMENTS}. Exactly one HIP source is "
            "passed positionally after <target>.")
  endif ()
  get_property(_ENABLED_LANGUAGES GLOBAL PROPERTY ENABLED_LANGUAGES)
  if (NOT "HIP" IN_LIST _ENABLED_LANGUAGES)
    message(FATAL_ERROR
            "luthier_create_offload_bundle(${target}): the HIP language is not "
            "enabled — add it via `project(... LANGUAGES HIP)` or "
            "`enable_language(HIP)` first.")
  endif ()

  # Resolve the AMDGCN target list: per-call TARGET_ISAS > LUTHIER_HIP_TARGETS >
  # derived from CMAKE_HIP_ARCHITECTURES (one bare target per arch). An empty
  # result is fine as long as the amdgcnspirv slice is emitted (SPIR-V found);
  # if both are empty we error out below rather than bundle nothing.
  set(_DEVICE_ISA_TARGETS "")
  if (OFFLOAD_BUNDLE_ARG_TARGET_ISAS)
    set(_DEVICE_ISA_TARGETS "${OFFLOAD_BUNDLE_ARG_TARGET_ISAS}")
  elseif (LUTHIER_HIP_TARGETS)
    set(_DEVICE_ISA_TARGETS "${LUTHIER_HIP_TARGETS}")
  elseif (CMAKE_HIP_ARCHITECTURES)
    foreach (_ARCH IN LISTS CMAKE_HIP_ARCHITECTURES)
      list(APPEND _DEVICE_ISA_TARGETS "amdgcn-amd-amdhsa--${_ARCH}")
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

  # The device-slice OBJECT libraries compile a COPY of the source, kept apart
  # from the original the host compiles. The host source carries an
  # OBJECT_DEPENDS on the fat binary (so it recompiles when the bundle changes);
  # source-file properties are directory-scoped, so if the device slices shared
  # that source they would inherit the OBJECT_DEPENDS and form a build cycle.
  # Compiling a copy breaks the share.
  get_filename_component(_SOURCE_NAME ${_ABS_SOURCE} NAME)
  set(_DEV_SOURCE "${CMAKE_CURRENT_BINARY_DIR}/${target}.dev_tu/${_SOURCE_NAME}")
  configure_file(${_ABS_SOURCE} ${_DEV_SOURCE} COPYONLY)
  # Compile the copy as HIP regardless of its extension (the host source is
  # likewise marked LANGUAGE HIP at the host-compile step below).
  set_source_files_properties(${_DEV_SOURCE} PROPERTIES LANGUAGE HIP)

  #---------------------------------------------------------------------------
  # Locate the plugins + LuthierTooling.
  #
  # All accept the in-tree-build naked target name OR the `luthier::...`
  # imported alias from find_package(luthier).
  #---------------------------------------------------------------------------

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

  #---------------------------------------------------------------------------
  # Resolve clang-offload-bundler.
  #
  # The bundler is an LLVM-project tool, so look in LLVM_TOOLS_BINARY_DIR
  # (exported by find_package(LLVM CONFIG)) first, then PATH.
  #---------------------------------------------------------------------------

  if (OFFLOAD_BUNDLE_ARG_BUNDLER)
    set(_OFFLOAD_BUNDLER "${OFFLOAD_BUNDLE_ARG_BUNDLER}")
  else ()
    find_program(LUTHIER_CLANG_OFFLOAD_BUNDLER
            NAMES clang-offload-bundler
            HINTS ${LLVM_TOOLS_BINARY_DIR}
            DOC "clang-offload-bundler used by luthier_create_offload_bundle")
    if (NOT LUTHIER_CLANG_OFFLOAD_BUNDLER)
      message(FATAL_ERROR
              "luthier_create_offload_bundle(${target}): clang-offload-bundler "
              "not found in LLVM_TOOLS_BINARY_DIR ('${LLVM_TOOLS_BINARY_DIR}'), "
              "or on PATH. Pass "
              "BUNDLER <path> to override.")
    endif ()
    set(_OFFLOAD_BUNDLER "${LUTHIER_CLANG_OFFLOAD_BUNDLER}")
  endif ()

  # AMD SPIR-V translator (amd-llvm-spirv) — OPTIONAL. Located by
  # FindLLVMSPIRVTranslator.cmake, which sets LUTHIER_LLVM_SPIRV_TRANSLATOR_FOUND /
  # LUTHIER_LLVM_SPIRV_TRANSLATOR. If the project hasn't run the search yet (e.g. an
  # installed find_package(luthier) consumer), try it now; treat absence as
  # "skip SPIR-V".
  if (NOT DEFINED LUTHIER_LLVM_SPIRV_TRANSLATOR_FOUND)
    find_package(LLVMSPIRVTranslator QUIET)
  endif ()
  if (LUTHIER_LLVM_SPIRV_TRANSLATOR_FOUND)
    get_filename_component(_SPIRV_DIR "${LUTHIER_LLVM_SPIRV_TRANSLATOR}" DIRECTORY)
  endif ()

  #---------------------------------------------------------------------------
  # Per target: a HIP OBJECT library that device-compiles the (copied) source
  # straight to LLVM bitcode with `-emit-llvm` + tool IR pass plugin.
  #---------------------------------------------------------------------------

  set(_SLICE_OBJS "")
  set(_REBUNDLE_SLICE_INPUTS "")
  set(_REBUNDLE_TARGET_ISAS "")
  set(_DEV_TARGETS "")
  set(_SEEN_LABELS "")
  foreach (_TARGET_ISA IN LISTS _DEVICE_ISA_TARGETS)
    luthier_parse_isa_string(${_TARGET_ISA} _TRIPLE _OFFLOAD _MFLAGS)

    # Canonical label of the target ISA, used for dedup. _OFFLOAD already lists
    # xnack/sramecc in canonical order; sorting the wave/cumode -m flags makes
    # feature reorderings of the same ISA (e.g. wavefrontsize64+:cumode- vs
    # cumode-:wavefrontsize64+) compare equal. Spaces (not ';') keep it a single
    # list element so IN_LIST / list(APPEND) treat it atomically.
    set(_SORTED_MFLAGS "${_MFLAGS}")
    list(SORT _SORTED_MFLAGS)
    string(REPLACE ";" " " _SORTED_MFLAGS "${_SORTED_MFLAGS}")
    set(_CANONICAL_LABEL "${_TRIPLE}--${_OFFLOAD} ${_SORTED_MFLAGS}")
    if (_CANONICAL_LABEL IN_LIST _SEEN_LABELS)
      continue()
    endif ()

    list(APPEND _SEEN_LABELS "${_CANONICAL_LABEL}")

    # Sanitize the ISA into a valid CMake target-name suffix: spell the subtarget
    # feature signs as _on/_off and turn the ':' separators into '_'. The +/-
    # replacements are anchored on the feature name, so the triple's dashes
    # (amdgcn-amd-amdhsa--) are left intact.
    set(_SANITIZED_TARGET_ISA "${_TARGET_ISA}")
    string(REGEX REPLACE "(xnack|sramecc|wavefrontsize64|cumode)\\+" "\\1_on"
            _SANITIZED_TARGET_ISA "${_SANITIZED_TARGET_ISA}")
    string(REGEX REPLACE "(xnack|sramecc|wavefrontsize64|cumode)-" "\\1_off"
            _SANITIZED_TARGET_ISA "${_SANITIZED_TARGET_ISA}")
    string(REGEX REPLACE ":" "_" _SANITIZED_TARGET_ISA "${_SANITIZED_TARGET_ISA}")

    set(_SLICE_TGT "${target}-${_SANITIZED_TARGET_ISA}")
    add_library(${_SLICE_TGT} OBJECT ${_DEV_SOURCE})
    set_target_properties(${_SLICE_TGT} PROPERTIES HIP_ARCHITECTURES "${_OFFLOAD}")
    # FIXME: -g0: disable debug info for now
    target_compile_options(${_SLICE_TGT} PRIVATE
            --cuda-device-only -emit-llvm --no-gpu-bundle-output -g0
            ${_MFLAGS} -fpass-plugin=${_LUTHIER_IR_PLUGIN})
    add_dependencies(${_SLICE_TGT} ${_LUTHIER_IR_PLUGIN_TARGET})

    list(APPEND _DEV_TARGETS "${_SLICE_TGT}")
    list(APPEND _SLICE_OBJS "$<TARGET_OBJECTS:${_SLICE_TGT}>")
    list(APPEND _REBUNDLE_SLICE_INPUTS "--input=$<TARGET_OBJECTS:${_SLICE_TGT}>")
    list(APPEND _REBUNDLE_TARGET_ISAS "hipv4-${_TARGET_ISA}")
  endforeach ()

  #---------------------------------------------------------------------------
  # Optionally add an AMD-flavored SPIR-V slice (amdgcnspirv), regardless of the
  # requested arch list, for the runtime SPIR-V -> AMDGCN JIT fallback. Skipped
  # when the SPIR-V translator is not found (LUTHIER_LLVM_SPIRV_TRANSLATOR_FOUND).
  #---------------------------------------------------------------------------

  if (LUTHIER_LLVM_SPIRV_TRANSLATOR_FOUND)
    set(_SPIRV_TARGET_ISA "hip-spirv64-amd-amdhsa--amdgcnspirv")
    set(_SPIRV_TARGET "${target}-${_SPIRV_TARGET_ISA}")
    add_library(${_SPIRV_TARGET} OBJECT ${_DEV_SOURCE})
    set_target_properties(${_SPIRV_TARGET} PROPERTIES HIP_ARCHITECTURES "amdgcnspirv")
    # FIXME: -g0: disable debug info for now
    # -U SPIRV is added to undefine the SPIRV definition added by the SPIRV translator
    # that clashes with LLVM's SPIRV target Triple
    target_compile_options(${_SPIRV_TARGET} PRIVATE
            --cuda-device-only --no-gpu-bundle-output -g0 -B "${_SPIRV_DIR}"
            -fpass-plugin=${_LUTHIER_IR_PLUGIN} -U SPIRV)
    add_dependencies(${_SPIRV_TARGET} ${_LUTHIER_IR_PLUGIN_TARGET})

    list(APPEND _DEV_TARGETS "${_SPIRV_TARGET}")
    list(APPEND _SLICE_OBJS "$<TARGET_OBJECTS:${_SPIRV_TARGET}>")
    list(APPEND _REBUNDLE_SLICE_INPUTS "--input=$<TARGET_OBJECTS:${_SPIRV_TARGET}>")
    list(APPEND _REBUNDLE_TARGET_ISAS "${_SPIRV_TARGET_ISA}")
  else ()
    message(STATUS
            "luthier_create_offload_bundle(${target}): SPIR-V translator not "
            "found; skipping the amdgcnspirv slice.")
  endif ()

  if (NOT _SLICE_OBJS)
    message(FATAL_ERROR
            "luthier_create_offload_bundle(${target}): no device slices to "
            "bundle — the resolved target list is empty and the SPIR-V slice is "
            "unavailable. Set CMAKE_HIP_ARCHITECTURES / LUTHIER_HIP_TARGETS / "
            "TARGET_ISAS, or enable SPIR-V via LUTHIER_LLVM_SPIRV_TRANSLATOR_PREFIX_PATH.")
  endif ()

  # Join the list of target ISAs for the bundle target argument
  list(JOIN _REBUNDLE_TARGET_ISAS "," _REBUNDLE_TARGET_ISAS)
  #---------------------------------------------------------------------------
  # Bundle the device slices (bitcode for the AMDGCN targets, SPIR-V for the
  # amdgcnspirv target)  into the final .hipfb — the one and only packing step.
  # --bundle-align=8 is required by the LLVM offload parser.
  #---------------------------------------------------------------------------

  add_custom_command(
          OUTPUT "${_TARGET_FATBIN}"
          COMMAND "${_OFFLOAD_BUNDLER}" --type=o
          --targets=${_REBUNDLE_TARGET_ISAS}
          ${_REBUNDLE_SLICE_INPUTS}
          --output=${_TARGET_FATBIN} --bundle-align=8
          DEPENDS ${_SLICE_OBJS}
          COMMENT "luthier_create_offload_bundle(${target}): bundle .hipfb"
          VERBATIM COMMAND_EXPAND_LISTS)

  add_custom_target(${target}-fatbin DEPENDS "${_TARGET_FATBIN}" ${_DEV_TARGETS})

  #---------------------------------------------------------------------------
  # Host compile → OBJECT library.
  #
  # The host side compiles through CMake's native HIP language (the .hip files
  # build as HIP).
  #
  # Flags (per-target; HIP language genex-guarded where multi-token):
  #   HIP_ARCHITECTURES OFF           : no --offload-arch is added — this is a
  #       host-only object; CMAKE_HIP_ARCHITECTURES is left untouched globally.
  #   --cuda-host-only / -fno-gpu-rdc : host-only, no separable device compile.
  #   -fuse-cuid=none                 : unsuffixed __hip_fatbin symbol names.
  #   -Xclang -fcuda-include-gpubinary -Xclang <fatbin> : embed the bundle
  #       bytes (SHELL: keeps the paired -Xclang from collapsing under de-dup).
  #   -fpass-plugin=<ir>              : IR tool compiler plugin.
  #   -fplugin=<cxx>                  : load the CXX tool compiler plugin.
  # OBJECT_DEPENDS on the fatbin makes each object wait for and rebuild with it.
  #---------------------------------------------------------------------------

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

  #---------------------------------------------------------------------------
  # Hand the created targets back to the caller if requested.
  #---------------------------------------------------------------------------
  if (OFFLOAD_BUNDLE_ARG_DEVICE_OBJECT_LIBRARIES)
    set(${OFFLOAD_BUNDLE_ARG_DEVICE_OBJECT_LIBRARIES} "${_DEV_TARGETS}" PARENT_SCOPE)
  endif ()
  if (OFFLOAD_BUNDLE_ARG_BUNDLE_TARGET)
    set(${OFFLOAD_BUNDLE_ARG_BUNDLE_TARGET} "${target}-fatbin" PARENT_SCOPE)
  endif ()
endfunction()
