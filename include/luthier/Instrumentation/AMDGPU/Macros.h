//===-- Macros.h ------------------------------------------------*- C++ -*-===//
// Copyright 2022-2025 @ Northeastern University Computer Architecture Lab
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
///
/// \file Defines a set of macros for writing Luthier tools for the AMDGPU
/// backend.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_INSTRUMENTATION_AMDGPU_MACROS_H
#define LUTHIER_INSTRUMENTATION_AMDGPU_MACROS_H


/// \brief If a tool contains an instrumentation hook it \b must
/// use this macro once. Luthier hooks are annotated via the the
/// \p LUTHIER_HOOK_CREATE macro. \n
///
/// \p MARK_LUTHIER_DEVICE_MODULE macro defines a managed variable of
/// type \p char named \p __luthier_reserved in the tool device code.
/// This managed variable ensures that: \n
/// 1. <b>The HIP runtime is forced to load the tool code object before the
/// first HIP kernel is launched on the device, without requiring eager binary
/// loading to be enabled</b>: The Clang compiler embeds the device code of a
/// Luthier tool and its bitcode into a static HIP FAT binary bundled within the
/// tool's shared object. During runtime, the tool's FAT binary gets
/// registered with the HIP runtime; However, by default, the HIP runtime loads
/// FAT binaries in a lazy fashion, only loading it onto a device if:
/// a. a kernel is launched from it on the said device, or
/// b. it contains a managed variable. \n
/// Including a managed variable is the only way to ensure the tool's FAT binary
/// is loaded in time without interfering with the loading mechanism of HIP
/// runtime.
/// \n
/// 2. <b>Luthier can easily identify a tool's code object by a constant time
/// symbol hash lookup</b>.
/// \n
/// If the target application is not using the HIP runtime, then no kernel is
/// launched by the HIP runtime, meaning that the tool FAT binary does not ever
/// get loaded. In that scenario, as the HIP runtime is present solely for
/// Luthier's function, the `HIP_ENABLE_DEFERRED_LOADING` environment
/// variable must be set to zero to ensure Luthier tool code objects get loaded
/// right away on all devices.
/// \sa LUTHIER_HOOK_ANNOTATE
#define MARK_LUTHIER_DEVICE_MODULE                                             \
  __attribute__((managed, used)) char LUTHIER_RESERVED_MANAGED_VAR = 0;

#define LUTHIER_HOOK_ANNOTATE                                                  \
  __attribute__((                                                              \
      device, used,                                                            \
      annotate(LUTHIER_STRINGIFY(LUTHIER_HOOK_ATTRIBUTE)))) extern "C" void

#define LUTHIER_EXPORT_HOOK_HANDLE(HookName)                                   \
  __attribute__((global, used)) extern "C" void LUTHIER_CAT(                   \
      LUTHIER_HOOK_HANDLE_PREFIX, HookName)(){};

#define LUTHIER_GET_HOOK_HANDLE(HookName)                                      \
  reinterpret_cast<const void *>(                                              \
      LUTHIER_CAT(LUTHIER_HOOK_HANDLE_PREFIX, HookName))

#endif
