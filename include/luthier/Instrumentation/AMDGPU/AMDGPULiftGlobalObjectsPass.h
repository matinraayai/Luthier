//===-- AMDGPULiftGlobalObjectsPass.h ---------------------------*- C++ -*-===//
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
/// \file
/// Implements the \c AMDGPULiftGlobalObjectsPass</tt> class in charge of
/// converting symbols inside the object file being lifted to
/// \c llvm::GlobalObject handles, as well as populating any analysis regarding
/// the object handles and their symbols.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_INSTRUMENTATION_AMDGPU_LIFT_GLOBAL_OBJECTS_PASS_H
#define LUTHIER_INSTRUMENTATION_AMDGPU_LIFT_GLOBAL_OBJECTS_PASS_H
#include <llvm/IR/PassManager.h>

namespace luthier {

class AMDGPULiftGlobalObjectsPass
    : public llvm::PassInfoMixin<AMDGPULiftGlobalObjectsPass> {
public:
  AMDGPULiftGlobalObjectsPass() = default;

  llvm::PreservedAnalyses run(llvm::Module &M,
                              llvm::ModuleAnalysisManager &MAM);
};

} // namespace luthier

#endif
