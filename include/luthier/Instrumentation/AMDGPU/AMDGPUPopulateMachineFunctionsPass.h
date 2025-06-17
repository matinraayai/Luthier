//===-- AMDGPUPopulateMachineFunctionsPass.h --------------------*- C++ -*-===//
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
/// Implements the <tt>AMDGPUPopulateMachineFunctionsPass</tt> pass in charge
/// of populating the machine functions inside the target module with
/// lifted machine instructions and basic blocks.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_INSTRUMENTATION_AMDGPU_AMDGPU_LIFT_FUNCTIONS_PASS_H
#define LUTHIER_INSTRUMENTATION_AMDGPU_AMDGPU_LIFT_FUNCTIONS_PASS_H
#include <llvm/IR/PassManager.h>

namespace luthier {
class AMDGPUPopulateMachineFunctionsPass
    : public llvm::PassInfoMixin<AMDGPUPopulateMachineFunctionsPass> {
public:
  AMDGPUPopulateMachineFunctionsPass() = default;

  llvm::PreservedAnalyses run(llvm::MachineFunction &MF,
                              llvm::MachineFunctionAnalysisManager &MFAM);
};

} // namespace luthier

#endif
