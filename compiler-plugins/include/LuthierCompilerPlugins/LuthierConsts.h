//===-- LuthierConsts.h ----------------------------------------*- C++ -*-===//
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

#ifndef LUTHIER_COMPILER_PLUGINS_LUTHIER_CONSTS_H
#define LUTHIER_COMPILER_PLUGINS_LUTHIER_CONSTS_H

namespace luthier {

/// Prefix appended to all hook handle kernels
#define LUTHIER_HOOK_HANDLE_PREFIX __luthier_hook_handle_

/// All hooks in instrumentation modules must have this attribute
#define LUTHIER_HOOK_ATTRIBUTE "luthier_hook"

/// Name of the reserved managed variable defined in all Luthier tools so
/// that its device module can be easily identified at runtime
#define LUTHIER_RESERVED_MANAGED_VAR __luthier_reserved


/// All bindings to Luthier intrinsics must have this attribute
#define LUTHIER_INTRINSIC_ATTRIBUTE "luthier_intrinsic"

/// All injected payload functions during instrumentation (i.e. functions that
/// their machine code will be inserted before an instrumentation point) must
/// have this attribute
#define LUTHIER_INJECTED_PAYLOAD_ATTRIBUTE "luthier_injected_payload"

#define STRINGIFY(S) #S

/// A macro to delay expansion of \c STRINGIFY
#define STRINGIFY_DELAY(S) STRINGIFY(S)

constexpr const char *HookHandlePrefix = STRINGIFY_DELAY(LUTHIER_HOOK_HANDLE_PREFIX);

constexpr const char *ReservedManagedVar =
    STRINGIFY_DELAY(LUTHIER_RESERVED_MANAGED_VAR);

static constexpr const char *HipCUIDPrefix = "__hip_cuid_";


#undef STRINGIFY_DELAY

#undef STRINGIFY

static constexpr const char *HookAttribute = "luthier_hook";

static constexpr const char *IntrinsicAttribute = "luthier_intrinsic";



}

#endif