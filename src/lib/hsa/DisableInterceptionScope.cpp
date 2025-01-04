//===-- DisableInterceptionScopt.h ----------------------------------------===//
// Copyright 2022-2024 @ Northeastern University Computer Architecture Lab
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
/// This file implements the \c hsa::DisableInterceptionScope class, a scoping
/// mechanism used to prevent re-interception of
/// HIP functions called inside Luthier tools inside HSA callbacks.
//===----------------------------------------------------------------------===//
#include "luthier/hsa/DisableInterceptionScope.h"
#include "hsa/HsaRuntimeInterceptor.hpp"

namespace luthier::hsa {

/// Keeps track of the number of times the \c DisableUserInterceptionScope class
/// has been created. It allows the disable user intercept scopes to be safely
/// nested
static thread_local unsigned int NumScopes{0};

DisableUserInterceptionScope::DisableUserInterceptionScope() {
  if (HsaRuntimeInterceptor::isInitialized() && NumScopes == 0) {
    HsaRuntimeInterceptor::instance()
        .toggleDisableUserCallbackInterceptionScope(true);
  }
  NumScopes++;
}

DisableUserInterceptionScope::~DisableUserInterceptionScope() {
  if (HsaRuntimeInterceptor::isInitialized()) {
    NumScopes--;
  }
  if (NumScopes == 0) {
    HsaRuntimeInterceptor::instance()
        .toggleDisableUserCallbackInterceptionScope(false);
  }
}

} // namespace luthier::hsa
