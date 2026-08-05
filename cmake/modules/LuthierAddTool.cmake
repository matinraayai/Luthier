#===- LuthierAddTool.cmake -----------------------------------------------===#
# Copyright @ Northeastern University Computer Architecture Lab
#
# Licensed under the Apache License, Version 2.0.
#===----------------------------------------------------------------------===#
#
# luthier_add_tool: build a Luthier tool host shared object.
#
# The device-side fat binary is produced by the `luthier-tool-compile`
# driver, which enumerates the full subtarget-feature matrix from
# CMAKE_HIP_ARCHITECTURES + the helper's VARY/PIN options and fans out
# clang invocations per (arch, wave, cumode, xnack, sramecc) slice. The
# host TU compiles through CMake's HIP language and consumes the
# produced `.hipfb` via `-fcuda-include-gpubinary=`.
#
# Synopsis:
#
#   luthier_add_tool(<target>
#     SOURCES <files...>
#     [VARY <knob...>]            # opt knobs into the device matrix:
#                                 # xnack | sramecc | wavefrontsize64 |
#                                 # cumode. wavefrontsize64 and cumode
#                                 # default to the full matrix on archs
#                                 # that support them; xnack and sramecc
#                                 # default to "any" (one slice, ELF
#                                 # _ANY_V4 flag bits).
#     [PIN <knob>± ...]           # pin a knob to one value (e.g.
#                                 # cumode-, wavefrontsize64+, xnack-).
#     [EXTRA_CLANG_FLAGS <flag…>] # extra flags forwarded to every
#                                 # per-slice device-side clang
#                                 # invocation. Host-side flags go
#                                 # through stock CMake (`target_compile_options`
#                                 # etc.) — no helper channel needed.
#     [BUNDLER <path>]            # override the clang-offload-bundler
#                                 # path. By default the driver picks
#                                 # the sibling of CMAKE_HIP_COMPILER.
#     [LIBRARIES <libs...>])
#
# The arch list is sourced from CMAKE_HIP_ARCHITECTURES. Override at
# configure time with `-DCMAKE_HIP_ARCHITECTURES=...`.
#
# Requirements:
#   * `project(... LANGUAGES HIP)` (we read CMAKE_HIP_COMPILER /
#     CMAKE_HIP_ARCHITECTURES / CMAKE_HIP_STANDARD).
#   * `find_package(hip REQUIRED)` for `hip::host` and `hip_INCLUDE_DIR`.
#   * The `luthier-tool-compile` binary + the IR/CXX compilation
#     plugins must be visible (in-tree via Luthier's own build, or
#     imported via `find_package(luthier ...)`).
#===----------------------------------------------------------------------===#

include_guard(GLOBAL)

function(luthier_add_tool target)
  cmake_parse_arguments(LAT ""
          "BUNDLER"
          "SOURCES;VARY;PIN;EXTRA_CLANG_FLAGS;LIBRARIES"
          ${ARGN})

  if (NOT LAT_SOURCES)
    message(FATAL_ERROR "luthier_add_tool(${target}): SOURCES required")
  endif ()
  if (NOT CMAKE_HIP_ARCHITECTURES)
    message(FATAL_ERROR
            "luthier_add_tool(${target}): CMAKE_HIP_ARCHITECTURES is empty. "
            "Set it (e.g. via -DCMAKE_HIP_ARCHITECTURES=...) before calling "
            "this helper.")
  endif ()
  if (NOT CMAKE_HIP_COMPILER)
    message(FATAL_ERROR
            "luthier_add_tool(${target}): CMAKE_HIP_COMPILER not set — "
            "enable HIP via `project(... LANGUAGES HIP)` first.")
  endif ()

  # Source-file naming → intermediates / fatbin file.
  list(GET LAT_SOURCES 0 _first_source)
  get_filename_component(_prefix "${_first_source}" NAME_WE)
  set(_fatbin "${CMAKE_CURRENT_BINARY_DIR}/${target}.${_prefix}.hipfb")

  #---------------------------------------------------------------------------
  # Locate the driver + plugins + LuthierTooling.
  #
  # All three accept the in-tree-build naked target name OR the
  # `luthier::...` imported alias from find_package(luthier). Drop down
  # to find_program for the driver in the rare third case (Luthier
  # installed but no find_package call).
  #---------------------------------------------------------------------------

  if (TARGET luthier-tool-compile)
    set(_driver "$<TARGET_FILE:luthier-tool-compile>")
    set(_driver_dep luthier-tool-compile)
  elseif (TARGET luthier::luthier-tool-compile)
    set(_driver "$<TARGET_FILE:luthier::luthier-tool-compile>")
    set(_driver_dep luthier::luthier-tool-compile)
  else ()
    find_program(_driver luthier-tool-compile REQUIRED)
    set(_driver_dep "")
  endif ()

  if (TARGET LuthierToolIRCompilationPlugin)
    set(_ir_plugin "$<TARGET_FILE:LuthierToolIRCompilationPlugin>")
    set(_ir_plugin_target LuthierToolIRCompilationPlugin)
  elseif (TARGET luthier::LuthierToolIRCompilationPlugin)
    set(_ir_plugin "$<TARGET_FILE:luthier::LuthierToolIRCompilationPlugin>")
    set(_ir_plugin_target luthier::LuthierToolIRCompilationPlugin)
  else ()
    message(FATAL_ERROR
            "luthier_add_tool(${target}): LuthierToolIRCompilationPlugin "
            "not found. Did you find_package(luthier ...)?")
  endif ()

  if (TARGET LuthierToolCXXCompilationPlugin)
    set(_cxx_plugin "$<TARGET_FILE:LuthierToolCXXCompilationPlugin>")
    set(_cxx_plugin_target LuthierToolCXXCompilationPlugin)
  elseif (TARGET luthier::LuthierToolCXXCompilationPlugin)
    set(_cxx_plugin "$<TARGET_FILE:luthier::LuthierToolCXXCompilationPlugin>")
    set(_cxx_plugin_target luthier::LuthierToolCXXCompilationPlugin)
  else ()
    message(FATAL_ERROR
            "luthier_add_tool(${target}): LuthierToolCXXCompilationPlugin "
            "not found")
  endif ()

  if (TARGET LuthierTooling)
    set(_tooling_target LuthierTooling)
  elseif (TARGET luthier::LuthierTooling)
    set(_tooling_target luthier::LuthierTooling)
  else ()
    message(FATAL_ERROR
            "luthier_add_tool(${target}): LuthierTooling target not found")
  endif ()

  set(_tooling_aux_targets "")
  foreach (_aux LuthierAMDGPU luthier::LuthierAMDGPU)
    if (TARGET ${_aux})
      list(APPEND _tooling_aux_targets ${_aux})
      break ()
    endif ()
  endforeach ()

  set(_amdgpu_tablegen_dep "")
  foreach (_tg LuthierAMDGPUTableGen luthier::LuthierAMDGPUTableGen)
    if (TARGET ${_tg})
      set(_amdgpu_tablegen_dep ${_tg})
      break ()
    endif ()
  endforeach ()

  # HIP runtime headers (for -idirafter).
  if (DEFINED hip_INCLUDE_DIR)
    set(_hip_include "${hip_INCLUDE_DIR}")
  elseif (DEFINED CMAKE_HIP_COMPILER_ROCM_ROOT)
    set(_hip_include "${CMAKE_HIP_COMPILER_ROCM_ROOT}/include")
  else ()
    message(FATAL_ERROR
            "luthier_add_tool(${target}): cannot locate HIP headers — "
            "call find_package(hip REQUIRED) or enable LANGUAGES HIP first")
  endif ()

  #---------------------------------------------------------------------------
  # Assemble the driver argv.
  #---------------------------------------------------------------------------

  set(_driver_args
          --hip-clang=${CMAKE_HIP_COMPILER}
          --plugin-ir=${_ir_plugin}
          --hip-include-dir=${_hip_include})
  if (CMAKE_HIP_STANDARD)
    list(APPEND _driver_args --hip-std=${CMAKE_HIP_STANDARD})
  endif ()
  foreach (_a IN LISTS CMAKE_HIP_ARCHITECTURES)
    list(APPEND _driver_args --arch=${_a})
  endforeach ()
  foreach (_v IN LISTS LAT_VARY)
    list(APPEND _driver_args --vary=${_v})
  endforeach ()
  foreach (_p IN LISTS LAT_PIN)
    list(APPEND _driver_args --pin=${_p})
  endforeach ()
  foreach (_f IN LISTS LAT_EXTRA_CLANG_FLAGS)
    list(APPEND _driver_args --extra-clang-flag=${_f})
  endforeach ()
  if (LAT_BUNDLER)
    list(APPEND _driver_args --bundler=${LAT_BUNDLER})
  endif ()

  # Include dirs + compile defs from LuthierTooling + LuthierAMDGPU.
  # Strip $<BUILD_INTERFACE:...>; skip $<INSTALL_INTERFACE:...>; the
  # `-D` from LuthierTooling's bundled-LLVM_DEFINITIONS blob is filtered
  # because clang's -D wants one identifier per arg (see Phase 0 doc).
  set(_seen_includes "")
  foreach (_tgt ${_tooling_target} ${_tooling_aux_targets})
    get_target_property(_inc ${_tgt} INTERFACE_INCLUDE_DIRECTORIES)
    if (_inc)
      foreach (_dir IN LISTS _inc)
        if (_dir MATCHES "^\\$<INSTALL_INTERFACE:")
          continue ()
        endif ()
        string(REGEX REPLACE "^\\$<BUILD_INTERFACE:(.*)>$" "\\1" _dir "${_dir}")
        if (_dir AND NOT _dir IN_LIST _seen_includes)
          list(APPEND _seen_includes "${_dir}")
          list(APPEND _driver_args --include-dir=${_dir})
        endif ()
      endforeach ()
    endif ()
  endforeach ()
  get_target_property(_defs ${_tooling_target} INTERFACE_COMPILE_DEFINITIONS)
  if (_defs)
    foreach (_def IN LISTS _defs)
      if (_def MATCHES "[ \t]" OR _def MATCHES "^-D")
        continue ()
      endif ()
      list(APPEND _driver_args --define=${_def})
    endforeach ()
  endif ()

  list(APPEND _driver_args --output=${_fatbin})
  foreach (_s IN LISTS LAT_SOURCES)
    if (IS_ABSOLUTE "${_s}")
      list(APPEND _driver_args "${_s}")
    else ()
      list(APPEND _driver_args "${CMAKE_CURRENT_SOURCE_DIR}/${_s}")
    endif ()
  endforeach ()

  #---------------------------------------------------------------------------
  # Custom command: invoke the driver.
  #---------------------------------------------------------------------------

  add_custom_command(
          OUTPUT "${_fatbin}"
          COMMAND ${_driver} ${_driver_args}
          DEPENDS
          ${LAT_SOURCES}
          ${_ir_plugin_target}
          ${_driver_dep}
          ${_amdgpu_tablegen_dep}
          WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
          COMMENT "luthier_add_tool(${target}): luthier-tool-compile"
          VERBATIM
          COMMAND_EXPAND_LISTS)
  add_custom_target(${target}-fatbin-dep DEPENDS "${_fatbin}")

  #---------------------------------------------------------------------------
  # Host TU + final shared module.
  #
  # The host TU compiles through CMake's HIP language with the fatbin
  # bytes spliced in by `-Xclang -fcuda-include-gpubinary -Xclang ...`.
  # `-fuse-cuid=none` strips the per-TU CUID suffix so the symbols are
  # the plain `__hip_fatbin` / `__hip_gpubin_handle` names the loader
  # (and LoadHIPFATBinaryInfoPass) expect. The IR plugin runs on the
  # host side too — it's what populates HipFatBinaries / HipFunctions
  # / HipDeviceVars from the __hipRegister* call sites. The CXX plugin
  # is host-only by design.
  #---------------------------------------------------------------------------

  add_library(${target} MODULE ${LAT_SOURCES})

  set_target_properties(${target} PROPERTIES
          HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")

  set_source_files_properties(${LAT_SOURCES} PROPERTIES
          LANGUAGE HIP
          COMPILE_OPTIONS "--cuda-host-only;-fno-gpu-rdc;-fuse-cuid=none"
          OBJECT_DEPENDS "${_fatbin}")

  # Each `-Xclang`-paired cc1 flag must live in its own quoted `SHELL:`
  # entry, otherwise CMake de-duplicates the repeated `-Xclang` tokens.
  target_compile_options(${target} PRIVATE
          -fpass-plugin=${_ir_plugin}
          -fplugin=${_cxx_plugin}
          "SHELL:-Xclang -add-plugin -Xclang luthier-emit-device-function-host-handle"
          "SHELL:-Xclang -fcuda-include-gpubinary -Xclang ${_fatbin}")

  target_link_libraries(${target} PRIVATE
          hip::host ${_tooling_target} ${LAT_LIBRARIES})

  # Genexes above expand to file paths at build time, so CMake can't
  # auto-derive the build-order edges. Add them explicitly.
  add_dependencies(${target}
          ${target}-fatbin-dep
          ${_ir_plugin_target}
          ${_cxx_plugin_target})
endfunction()
