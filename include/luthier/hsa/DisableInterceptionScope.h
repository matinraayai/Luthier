//===-- DisableInterceptionScopt.h ------------------------------*- C++ -*-===//
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
/// This file describes \c hsa::DisableUserInterceptionScope, a scoping
/// mechanism used to temporarily disable tool HSA callbacks, so that the tool
/// can make calls to HIP functions.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_HSA_DISABLE_USER_INTERCEPTION_SCOPE_H
#define LUTHIER_HSA_DISABLE_USER_INTERCEPTION_SCOPE_H

namespace luthier::hsa {

class DisableUserInterceptionScope {
  public:
    DisableUserInterceptionScope();

   ~DisableUserInterceptionScope();
};

}


#endif
