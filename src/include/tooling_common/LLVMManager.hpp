//===-- LLVMManager.hpp - LLVM Library Lifetime Management ----------------===//
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
/// This file describes Luthier's LLVM Manager, a singleton in charge of
/// initializing and finalizing the LLVM library.
//===----------------------------------------------------------------------===//

#ifndef LUTHIER_TOOLING_COMMON_LLVM_MANAGER_HPP
#define LUTHIER_TOOLING_COMMON_LLVM_MANAGER_HPP
#include "common/Singleton.hpp"

namespace luthier {

class LLVMManager : public Singleton<LLVMManager> {
public:
  /// Initializes the LLVM library
  LLVMManager();

  /// Finalizes the LLVM library
  ~LLVMManager() override;
};

} // namespace luthier

#endif