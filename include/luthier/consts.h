//===-- consts.h ------------------------------------------------*- C++ -*-===//
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
/// \file
/// This file defines some constants used throughout both Luthier and the
/// compiler plugins projects.
//===----------------------------------------------------------------------===//

#ifndef LUTHIER_CONSTS_H
#define LUTHIER_CONSTS_H

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
#define LUTHIER_HOOK_HANDLE_PREFIX __luthier_hook_handle_

/// All hooks in instrumentation modules must have this attribute
#define LUTHIER_HOOK_ATTRIBUTE luthier_hook

/// Name of the reserved managed variable defined in all Luthier tools so
/// that its device module can be easily identified at runtime
#define LUTHIER_RESERVED_MANAGED_VAR __luthier_reserved

/// All bindings to Luthier intrinsics must have this attribute
#define LUTHIER_INTRINSIC_ATTRIBUTE luthier_intrinsic

/// Prefix of the CUID symbol inside a HIP module
#define LUTHIER_HIP_CUID_PREFIX __hip_cuid_

/// All injected payload functions during instrumentation (i.e. functions that
/// their machine code will be inserted before an instrumentation point) must
/// have this attribute
#define LUTHIER_INJECTED_PAYLOAD_ATTRIBUTE luthier_injected_payload

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

} // namespace luthier

#endif