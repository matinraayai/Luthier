//===-- Executable.cpp - HSA Executable Interface Implementation ----------===//
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
/// This file implements the static \c hsa::Executable interface methods.
//===----------------------------------------------------------------------===//
#include "hsa/ExecutableImpl.hpp"
#include "luthier/hsa/HsaError.h"

namespace luthier::hsa {

llvm::Expected<std::unique_ptr<Executable>>
Executable::create(hsa_profile_t Profile,
                   hsa_default_float_rounding_mode_t DefaultFloatRoundingMode) {
  hsa_executable_t Exec;

  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
      hsa::HsaRuntimeInterceptor::instance()
          .getSavedApiTableContainer()
          .core.hsa_executable_create_alt_fn(Profile, DefaultFloatRoundingMode,
                                             "", &Exec)));

  return std::make_unique<ExecutableImpl>(Exec);
}

} // namespace luthier::hsa
