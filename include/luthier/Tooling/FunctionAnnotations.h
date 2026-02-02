//===-- FunctionAnnotations.h -------------------------------------*-C++-*-===//
// Copyright 2026 @ Northeastern University Computer Architecture Lab
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
/// \file FunctionAnnotations.h
/// Defines a set of function annotations used throughout Luthier's code
/// generation process, as well as methods to set/extract information related
/// to them from the IR function.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOLING_FUNCTION_ANNOTATIONS_H
#define LUTHIER_TOOLING_FUNCTION_ANNOTATIONS_H
#include "luthier/Tooling/EntryPoint.h"
#include <optional>

namespace llvm {

class Function;

}

namespace luthier {

//===----------------------------------------------------------------------===//
// Utility macros (see https://github.com/pfultz2/Cloak)
//===----------------------------------------------------------------------===//
#define LUTHIER_STRINGIFY(S) LUTHIER_PRIMITIVE_STR(S)
#define LUTHIER_PRIMITIVE_STR(S) #S

#define LUTHIER_CAT(a, ...) LUTHIER_PRIMITIVE_CAT(a, __VA_ARGS__)
#define LUTHIER_PRIMITIVE_CAT(a, ...) a##__VA_ARGS__

//===----------------------------------------------------------------------===//
// Luthier attributes and constants
//===----------------------------------------------------------------------===//

/// Prefix appended to all hook handle kernels
#define LUTHIER_HOOK_HANDLE_PREFIX __luthier_builtin_hook_handle_

/// All hooks in instrumentation modules must have this attribute
#define LUTHIER_HOOK_ATTRIBUTE luthier.function.hook

/// Name of the reserved managed variable defined in all Luthier tools so
/// that its device module can be easily identified at runtime
#define LUTHIER_RESERVED_MANAGED_VAR __luthier_builtin_reserved

/// All bindings to Luthier intrinsics must have this attribute
#define LUTHIER_INTRINSIC_ATTRIBUTE luthier.intrinsic

/// Prefix of the CUID symbol inside a HIP module
#define LUTHIER_HIP_CUID_PREFIX __hip_cuid_

/// All injected payload functions during instrumentation (i.e. functions that
/// their machine code will be inserted before an instrumentation point) must
/// have this attribute
#define LUTHIER_INJECTED_PAYLOAD_ATTRIBUTE luthier.function.injected_payload

static constexpr const char *HookHandlePrefix =
    LUTHIER_STRINGIFY(LUTHIER_HOOK_HANDLE_PREFIX);

static constexpr const char *ReservedManagedVar =
    LUTHIER_STRINGIFY(LUTHIER_RESERVED_MANAGED_VAR);

static constexpr const char *HipCUIDPrefix =
    LUTHIER_STRINGIFY(LUTHIER_HIP_CUID_PREFIX);

static constexpr const char *HookAttribute =
    LUTHIER_STRINGIFY(LUTHIER_HOOK_ATTRIBUTE);

static constexpr const char *IntrinsicAttribute =
    LUTHIER_STRINGIFY(LUTHIER_INTRINSIC_ATTRIBUTE);

static constexpr const char *InjectedPayloadAttribute =
    LUTHIER_STRINGIFY(LUTHIER_INJECTED_PAYLOAD_ATTRIBUTE);

static constexpr const char *EntryPointAddrAttr =
    "luthier.function.entrypoint.addr";

static constexpr const char *InitialEntryPointAttr =
    "luthier.function.initial_entrypoint";

/// \brief If a tool contains an instrumentation hook it \b must
/// use this macro once. Luthier hooks are annotated via the
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

void setFunctionEntryPoint(llvm::Function &F, EntryPoint EP);

std::optional<EntryPoint> getFunctionEntryPoint(llvm::Function &F);

} // namespace luthier

#endif