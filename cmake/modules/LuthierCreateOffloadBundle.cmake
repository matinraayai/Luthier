#===- LuthierCreateOffloadBundle.cmake -----------------------------------===#
# Copyright @ Northeastern University Computer Architecture Lab
#
# Licensed under the Apache License, Version 2.0.
#===----------------------------------------------------------------------===#

include_guard(GLOBAL)

#---------------------------------------------------------------------------
# Parse one complete offload target ID into the pieces clang and the
# offload bundler want:
#   <entry>        e.g. amdgcn-amd-amdhsa--gfx942:xnack+:wavefrontsize64-:cumode-
#   out_offload    --offload-arch= value: proc[:sramecc±][:xnack±]
#   out_mflags     standalone clang flags: -m[no-]wavefrontsize64 / -m[no-]cumode
#   out_label      full bundle target-ID tail: proc[:sramecc±][:xnack±][:wavefrontsize64±][:cumode±]
#
# xnack/sramecc live in clang's target-ID feature whitelist, so they ride
# inside --offload-arch=. wavefrontsize64/cumode are NOT in that whitelist
# (clang rejects them in the target ID) so they are passed as -m flags; they
# still appear in the bundle label because the bundler treats labels opaquely
# and the Luthier loader keys wave/cumode matching off them + the per-slice
# __luthier_subtarget marker.
#---------------------------------------------------------------------------
function(_luthier_parse_hip_target entry out_offload out_mflags out_label)
  # Split triple ("amdgcn-amd-amdhsa") from the target-ID at the empty-env "--".
  string(FIND "${entry}" "--" _sep)
  if (_sep EQUAL -1)
    message(FATAL_ERROR
            "luthier_create_offload_bundle: target '${entry}' is missing the "
            "'--<processor>' suffix (expected e.g. amdgcn-amd-amdhsa--gfx942).")
  endif ()
  string(SUBSTRING "${entry}" 0 ${_sep} _triple)
  math(EXPR _rest_start "${_sep} + 2")
  string(SUBSTRING "${entry}" ${_rest_start} -1 _rest)
  if (NOT _triple STREQUAL "amdgcn-amd-amdhsa")
    message(FATAL_ERROR
            "luthier_create_offload_bundle: target '${entry}' has unsupported "
            "triple '${_triple}' (only amdgcn-amd-amdhsa is supported).")
  endif ()

  # proc + feature tokens.
  string(REPLACE ":" ";" _toks "${_rest}")
  list(GET _toks 0 _proc)
  list(REMOVE_AT _toks 0)

  set(_xnack "")
  set(_sramecc "")
  set(_wave "")
  set(_cumode "")
  set(_mflags "")
  foreach (_t IN LISTS _toks)
    string(REGEX MATCH "^(xnack|sramecc|wavefrontsize64|cumode)([+-])$" _m "${_t}")
    if (NOT _m)
      message(FATAL_ERROR
              "luthier_create_offload_bundle: target '${entry}' has unknown or "
              "malformed feature '${_t}' (expected <name>+ / <name>- where name "
              "is xnack | sramecc | wavefrontsize64 | cumode).")
    endif ()
    set(_name "${CMAKE_MATCH_1}")
    set(_sign "${CMAKE_MATCH_2}")
    if (_name STREQUAL "xnack")
      set(_xnack ":xnack${_sign}")
    elseif (_name STREQUAL "sramecc")
      set(_sramecc ":sramecc${_sign}")
    elseif (_name STREQUAL "wavefrontsize64")
      set(_wave ":wavefrontsize64${_sign}")
      if (_sign STREQUAL "+")
        list(APPEND _mflags -mwavefrontsize64)
      else ()
        list(APPEND _mflags -mno-wavefrontsize64)
      endif ()
    else () # cumode
      set(_cumode ":cumode${_sign}")
      if (_sign STREQUAL "+")
        list(APPEND _mflags -mcumode)
      else ()
        list(APPEND _mflags -mno-cumode)
      endif ()
    endif ()
  endforeach ()

  # AMDGPU canonical order: sramecc, xnack (offload-arch), then our wave/cumode.
  set(${out_offload} "${_proc}${_sramecc}${_xnack}" PARENT_SCOPE)
  set(${out_mflags} "${_mflags}" PARENT_SCOPE)
  set(${out_label} "${_proc}${_sramecc}${_xnack}${_wave}${_cumode}" PARENT_SCOPE)
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
# Both the device and the host targets are unliked OBJECT libraries, and can
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
#                                 # path. By default the sibling of
#                                 # CMAKE_HIP_COMPILER.
# --- outputs:
#     [DEVICE_OBJECT_LIBRARIES <var>] # list of per-slice device OBJECT libraries
#     [BUNDLE_TARGET <var>])          # the custom target that builds the .hipfb
#
# The target list is sourced from TARGET_ISAS, else LUTHIER_HIP_TARGETS, else
# synthesized from CMAKE_HIP_ARCHITECTURES.
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
          "TARGET_ISAS;"
          ${ARGN})

  if (NOT source)
    message(FATAL_ERROR
            "luthier_create_offload_bundle(${target}): a HIP source file must be "
            "passed immediately after <target>.")
  endif ()
  # A tool is a single HIP TU: -fuse-cuid=none forces the unsuffixed __hip_fatbin
  # symbol, so a second HIP source would collide on it (and the per-slice bundler
  # input assumes one object per device library). The single positional <source>
  # guarantees one TU; reject any stray extra positional/keyword here.
  if (OFFLOAD_BUNDLE_ARG_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
            "luthier_create_offload_bundle(${target}): unexpected argument(s): "
            "${OFFLOAD_BUNDLE_ARG_UNPARSED_ARGUMENTS}. Exactly one HIP source is "
            "passed positionally after <target>.")
  endif ()
  if (NOT CMAKE_HIP_COMPILER)
    message(FATAL_ERROR
            "luthier_create_offload_bundle(${target}): CMAKE_HIP_COMPILER not "
            "set — enable HIP via `project(... LANGUAGES HIP)` first.")
  endif ()

  # Resolve the AMDGCN target list: per-call TARGET_ISAS > LUTHIER_HIP_TARGETS >
  # derived from CMAKE_HIP_ARCHITECTURES (one bare target per arch). An empty
  # result is fine as long as the amdgcnspirv slice is emitted (SPIR-V found);
  # if both are empty we error out below rather than bundle nothing.
  set(_targets "")
  if (OFFLOAD_BUNDLE_ARG_TARGET_ISAS)
    set(_targets "${OFFLOAD_BUNDLE_ARG_TARGET_ISAS}")
  elseif (LUTHIER_HIP_TARGETS)
    set(_targets "${LUTHIER_HIP_TARGETS}")
  elseif (CMAKE_HIP_ARCHITECTURES)
    foreach (_a IN LISTS CMAKE_HIP_ARCHITECTURES)
      list(APPEND _targets "amdgcn-amd-amdhsa--${_a}")
    endforeach ()
  endif ()

  # Source-file naming → intermediates / fatbin.
  get_filename_component(_prefix "${source}" NAME_WE)
  set(_fatbin "${CMAKE_CURRENT_BINARY_DIR}/${target}.${_prefix}.hipfb")

  # Absolute source path (used by both the device and host compiles). Kept as a
  # one-element list (_abs_sources) for the downstream add_library / copy logic.
  if (IS_ABSOLUTE "${source}")
    set(_abs_sources "${source}")
  else ()
    set(_abs_sources "${CMAKE_CURRENT_SOURCE_DIR}/${source}")
  endif ()

  # The device-slice OBJECT libraries compile a COPY of the sources, kept apart
  # from the originals the host compiles. The host source carries an
  # OBJECT_DEPENDS on the fat binary (so it recompiles when the bundle changes);
  # source-file properties are directory-scoped, so if the device slices shared
  # that source they would inherit the OBJECT_DEPENDS and form a build cycle
  # (slice object -> fatbin -> slice object). Compiling a copy breaks the share.
  set(_dev_sources "")
  foreach (_s IN LISTS _abs_sources)
    get_filename_component(_sn "${_s}" NAME)
    set(_dev_copy "${CMAKE_CURRENT_BINARY_DIR}/${target}.dev_tu/${_sn}")
    configure_file("${_s}" "${_dev_copy}" COPYONLY)
    list(APPEND _dev_sources "${_dev_copy}")
  endforeach ()

  #---------------------------------------------------------------------------
  # Locate the plugins + LuthierTooling.
  #
  # All accept the in-tree-build naked target name OR the `luthier::...`
  # imported alias from find_package(luthier).
  #---------------------------------------------------------------------------

  if (TARGET LuthierToolIRCompilationPlugin)
    set(_ir_plugin "$<TARGET_FILE:LuthierToolIRCompilationPlugin>")
    set(_ir_plugin_target LuthierToolIRCompilationPlugin)
  elseif (TARGET luthier::LuthierToolIRCompilationPlugin)
    set(_ir_plugin "$<TARGET_FILE:luthier::LuthierToolIRCompilationPlugin>")
    set(_ir_plugin_target luthier::LuthierToolIRCompilationPlugin)
  else ()
    message(FATAL_ERROR
            "luthier_create_offload_bundle(${target}): "
            "LuthierToolIRCompilationPlugin not found.")
  endif ()

  if (TARGET LuthierToolCXXCompilationPlugin)
    set(_cxx_plugin "$<TARGET_FILE:LuthierToolCXXCompilationPlugin>")
    set(_cxx_plugin_target LuthierToolCXXCompilationPlugin)
  elseif (TARGET luthier::LuthierToolCXXCompilationPlugin)
    set(_cxx_plugin "$<TARGET_FILE:luthier::LuthierToolCXXCompilationPlugin>")
    set(_cxx_plugin_target luthier::LuthierToolCXXCompilationPlugin)
  else ()
    message(FATAL_ERROR
            "luthier_create_offload_bundle(${target}): "
            "LuthierToolCXXCompilationPlugin not found")
  endif ()

  #---------------------------------------------------------------------------
  # The helper injects ONLY the Luthier plugins onto each target (IR pass plugin
  # on every device slice; IR pass plugin + CXX clang plugin on the host) plus
  # the foundational mode flags that define each compile (--cuda-{device,host}-only,
  # -emit-llvm / --no-gpu-bundle-output, --offload-arch via HIP_ARCHITECTURES,
  # -m{wave,cumode}, -fcuda-include-gpubinary). Everything else — include dirs,
  # -O3, -std=, defines, and any extra flags — is the caller's responsibility on
  # the returned targets. NOTE: -O3 matters for the instrumentation pipeline
  # (without it HIP-Clang leaves out-of-line device helpers as pre-RA bodies that
  # trip resource-usage analysis), so callers will typically want it.
  #---------------------------------------------------------------------------

  #---------------------------------------------------------------------------
  # Resolve clang-offload-bundler + the host placeholder triple.
  #
  # The bundler is an LLVM-project tool, so look in LLVM_TOOLS_BINARY_DIR
  # (exported by find_package(LLVM CONFIG)) first, then next to
  # CMAKE_HIP_COMPILER, then PATH. The host placeholder slot's label uses
  # clang's own default target triple (the form `host-<triple>` the bundler
  # emits for the `--cuda-device-only` host stub).
  #---------------------------------------------------------------------------

  if (OFFLOAD_BUNDLE_ARG_BUNDLER)
    set(_bundler "${OFFLOAD_BUNDLE_ARG_BUNDLER}")
  else ()
    get_filename_component(_hipbin "${CMAKE_HIP_COMPILER}" DIRECTORY)
    find_program(LUTHIER_CLANG_OFFLOAD_BUNDLER
            NAMES clang-offload-bundler
            HINTS ${LLVM_TOOLS_BINARY_DIR} "${_hipbin}"
            DOC "clang-offload-bundler used by luthier_create_offload_bundle")
    if (NOT LUTHIER_CLANG_OFFLOAD_BUNDLER)
      message(FATAL_ERROR
              "luthier_create_offload_bundle(${target}): clang-offload-bundler "
              "not found in LLVM_TOOLS_BINARY_DIR ('${LLVM_TOOLS_BINARY_DIR}'), "
              "next to CMAKE_HIP_COMPILER ('${_hipbin}'), or on PATH. Pass "
              "BUNDLER <path> to override.")
    endif ()
    set(_bundler "${LUTHIER_CLANG_OFFLOAD_BUNDLER}")
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
    get_filename_component(_spv_dir "${LUTHIER_LLVM_SPIRV_TRANSLATOR}" DIRECTORY)
  endif ()

  #---------------------------------------------------------------------------
  # Per target: a HIP OBJECT library that device-compiles the (copied) source
  # straight to LLVM bitcode with `-emit-llvm`. The IR pass plugin still runs
  # (its EP-callback passes fire in the optimization pipeline, embedding the
  # IModule and the __luthier_subtarget marker), but the AMDGPU backend /
  # assembler / linker do not.
  #
  # NOTE: despite the `.o` filename CMake gives them, the objects produced by
  # these device OBJECT libraries are NOT object files — they are raw LLVM
  # bitcode (and SPIR-V for the amdgcnspirv slice, below). They must never be
  # linked; they are only ever fed to clang-offload-bundler, which dispatches on
  # the file's magic bytes rather than its extension. CMake has no per-target
  # knob to change the object extension, so the `.o` name is cosmetic.
  #
  # The arch + xnack/sramecc ride in HIP_ARCHITECTURES (-> --offload-arch=);
  # wave/cumode are -m flags (frontend target attributes).
  #---------------------------------------------------------------------------

  set(_slice_objs "")
  set(_slice_inputs "")
  set(_rebundle_targets "")
  set(_dev_targets "")
  set(_seen_labels "")
  set(_idx 0)
  foreach (_tgt IN LISTS _targets)
    _luthier_parse_hip_target("${_tgt}" _offload _mflags _label)

    # Reject duplicate targets up front. Keyed on the canonical label (the
    # parser normalizes feature order), so reordered-feature spellings of the
    # same ISA are caught too — exactly what clang-offload-bundler would reject
    # at bundle time ("Duplicate targets are not allowed").
    if (_label IN_LIST _seen_labels)
      message(FATAL_ERROR
              "luthier_create_offload_bundle(${target}): duplicate offload "
              "target '${_tgt}' (resolves to '${_label}'). Each target may "
              "appear only once in TARGETS / LUTHIER_HIP_TARGETS / "
              "CMAKE_HIP_ARCHITECTURES.")
    endif ()
    list(APPEND _seen_labels "${_label}")

    set(_slice_tgt "${target}.dev.${_idx}")
    add_library(${_slice_tgt} OBJECT ${_dev_sources})
    set_target_properties(${_slice_tgt} PROPERTIES HIP_ARCHITECTURES "${_offload}")
    # Only the IR pass plugin is injected here; the caller adds include dirs,
    # -O3/-std, defines, and any extra flags on this target itself.
    target_compile_options(${_slice_tgt} PRIVATE
            --cuda-device-only -emit-llvm --no-gpu-bundle-output
            ${_mflags} -fpass-plugin=${_ir_plugin})
    add_dependencies(${_slice_tgt} ${_ir_plugin_target})

    list(APPEND _dev_targets "${_slice_tgt}")
    list(APPEND _slice_objs "$<TARGET_OBJECTS:${_slice_tgt}>")
    list(APPEND _slice_inputs "--input=$<TARGET_OBJECTS:${_slice_tgt}>")
    string(APPEND _rebundle_targets ",hipv4-amdgcn-amd-amdhsa--${_label}")
    math(EXPR _idx "${_idx} + 1")
  endforeach ()

  #---------------------------------------------------------------------------
  # Optionally add an AMD-flavored SPIR-V slice (amdgcnspirv), regardless of the
  # requested arch list, for the runtime SPIR-V -> AMDGCN JIT fallback. Skipped
  # when the SPIR-V translator was not found (LUTHIER_LLVM_SPIRV_TRANSLATOR_FOUND).
  # `--no-gpu-bundle-output` makes clang emit raw SPIR-V (no __CLANG_OFFLOAD_BUNDLE__
  # wrapper); `-B${_spv_dir}` lets clang exec the SPIR-V translator. The object
  # is SPIR-V (again, not an object file despite the `.o` name). Its bundle label
  # is `hip-spirv64-amd-amdhsa--amdgcnspirv` (the `hip-` kind prefix + `spirv64`
  # triple, distinct from the native `hipv4-amdgcn-...` slices).
  #---------------------------------------------------------------------------

  if (LUTHIER_LLVM_SPIRV_TRANSLATOR_FOUND)
    set(_spv_target "hip-spirv64-amd-amdhsa--amdgcnspirv")
    set(_spv_tgt "${target}.dev.amdgcnspirv")
    add_library(${_spv_tgt} OBJECT ${_dev_sources})
    set_target_properties(${_spv_tgt} PROPERTIES HIP_ARCHITECTURES "amdgcnspirv")
    target_compile_options(${_spv_tgt} PRIVATE
            --cuda-device-only --no-gpu-bundle-output -B "${_spv_dir}"
            -fpass-plugin=${_ir_plugin})
    add_dependencies(${_spv_tgt} ${_ir_plugin_target})

    list(APPEND _dev_targets "${_spv_tgt}")
    list(APPEND _slice_objs "$<TARGET_OBJECTS:${_spv_tgt}>")
    list(APPEND _slice_inputs "--input=$<TARGET_OBJECTS:${_spv_tgt}>")
    string(APPEND _rebundle_targets ",${_spv_target}")
  else ()
    message(STATUS
            "luthier_create_offload_bundle(${target}): SPIR-V translator not "
            "found; skipping the amdgcnspirv slice.")
  endif ()

  if (NOT _slice_objs)
    message(FATAL_ERROR
            "luthier_create_offload_bundle(${target}): no device slices to "
            "bundle — the resolved target list is empty and the SPIR-V slice is "
            "unavailable. Set CMAKE_HIP_ARCHITECTURES / LUTHIER_HIP_TARGETS / "
            "TARGETS, or enable SPIR-V via LUTHIER_LLVM_SPIRV_TRANSLATOR_PREFIX_PATH.")
  endif ()

  #---------------------------------------------------------------------------
  # Bundle the device slices (bitcode for the AMDGCN targets, SPIR-V for the
  # amdgcnspirv target) + a /dev/null host placeholder into the final .hipfb —
  # the one and only packing step. --bundle-align=8 keeps each slice's offset
  # 8-byte aligned (bitcode itself has no alignment requirement, but the
  # alignment is harmless and keeps any code-object/SPIR-V slice readable in
  # place by the loader).
  #---------------------------------------------------------------------------

  add_custom_command(
          OUTPUT "${_fatbin}"
          COMMAND "${_bundler}" --type=o
          --targets=${_rebundle_targets}
          --input=/dev/null ${_slice_inputs}
          --output="${_fatbin}" --bundle-align=8
          DEPENDS ${_slice_objs}
          COMMENT "luthier_create_offload_bundle(${target}): bundle .hipfb"
          VERBATIM COMMAND_EXPAND_LISTS)

  add_custom_target(${target}-fatbin-dep DEPENDS "${_fatbin}")

  #---------------------------------------------------------------------------
  # Host compile → OBJECT library.
  #
  # The host side compiles through CMake's native HIP language (the .hip files
  # build as HIP). We do NOT link it against anything; the result is just the
  # object file(s), exposed as the OBJECT library `${target}` for the caller to
  # add to another target (e.g. `target_link_libraries(other PRIVATE ${target})`
  # or `target_sources(other PRIVATE $<TARGET_OBJECTS:${target}>)`). The caller
  # is responsible for linking hip::host (resolves __hipRegisterFatBinary et al.)
  # and any other dependencies.
  #
  # Flags (per-target; HIP language genex-guarded where multi-token):
  #   HIP_ARCHITECTURES OFF           : no --offload-arch is added — this is a
  #       host-only object; CMAKE_HIP_ARCHITECTURES is left untouched globally.
  #   --cuda-host-only / -fno-gpu-rdc : host-only, no separable device compile.
  #   -fuse-cuid=none                 : unsuffixed __hip_fatbin symbol names.
  #   -Xclang -fcuda-include-gpubinary -Xclang <fatbin> : embed the bundle
  #       bytes (SHELL: keeps the paired -Xclang from collapsing under de-dup).
  #   -fpass-plugin=<ir>              : LoadHIPFATBinaryInfoPass.
  #   -fplugin=<cxx>                  : LUTHIER_HOOK_* AST rewrites.
  # OBJECT_DEPENDS on the fatbin makes each object wait for and rebuild with it.
  #---------------------------------------------------------------------------

  add_library(${target} OBJECT ${_abs_sources})
  set_source_files_properties(${_abs_sources} PROPERTIES
          LANGUAGE HIP
          OBJECT_DEPENDS "${_fatbin}")
  set_target_properties(${target} PROPERTIES HIP_ARCHITECTURES OFF)

  # Both plugins are injected on the host: the IR pass plugin (LoadHIPFATBinaryInfoPass)
  # and the CXX clang plugin (LUTHIER_HOOK_* AST rewrites). The caller adds
  # include dirs, -O3/-std, defines, and any extra flags on this target itself.
  target_compile_options(${target} PRIVATE
          --cuda-host-only -fno-gpu-rdc -fuse-cuid=none
          "SHELL:-Xclang -fcuda-include-gpubinary -Xclang ${_fatbin}"
          -fpass-plugin=${_ir_plugin}
          -fplugin=${_cxx_plugin})

  # $<TARGET_FILE:...> compile options and the generated fatbin don't create
  # build-order edges on their own; add them explicitly.
  add_dependencies(${target}
          ${target}-fatbin-dep
          ${_ir_plugin_target}
          ${_cxx_plugin_target})

  #---------------------------------------------------------------------------
  # Hand the created targets back to the caller (all optional). The caller is
  # responsible for any target_include_directories / target_link_libraries on
  # these — the helper enforces none.
  #---------------------------------------------------------------------------

  if (OFFLOAD_BUNDLE_ARG_HOST_OBJECT_LIBRARY)
    set(${OFFLOAD_BUNDLE_ARG_HOST_OBJECT_LIBRARY} "${target}" PARENT_SCOPE)
  endif ()
  if (OFFLOAD_BUNDLE_ARG_DEVICE_OBJECT_LIBRARIES)
    set(${OFFLOAD_BUNDLE_ARG_DEVICE_OBJECT_LIBRARIES} "${_dev_targets}" PARENT_SCOPE)
  endif ()
  if (OFFLOAD_BUNDLE_ARG_BUNDLE_TARGET)
    set(${OFFLOAD_BUNDLE_ARG_BUNDLE_TARGET} "${target}-fatbin-dep" PARENT_SCOPE)
  endif ()
endfunction()
