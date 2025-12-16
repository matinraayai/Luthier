//===-- PatchLiftedRepresentationPass.h -------------------------*- C++ -*-===//
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
/// This file describes the Patch lifted representation pass, used to
/// patch-in the generated machine code from the instrumentation module into
/// the target application.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOLING_PATCH_LIFTED_REPRESENTATION_H
#define LUTHIER_TOOLING_PATCH_LIFTED_REPRESENTATION_H
#include <llvm/CodeGen/MachineBasicBlock.h>
#include <llvm/CodeGen/MachineModuleInfo.h>
#include <llvm/IR/PassManager.h>

namespace luthier {

class PatchLiftedRepresentationPass
    : public llvm::PassInfoMixin<PatchLiftedRepresentationPass> {
public:
  enum PatchType {
    INLINE = 0,  ///< Patch the injected payload directly into the target app
    OUTLINE = 1, ///< Append the injected payload to the end/beginning
                 ///< of the function, jump to the injected
                 ///< payload using a short jump
  };

private:
  /// The instrumentation module
  llvm::Module &IModule;
  /// The instrumentation machine module info
  llvm::MachineModuleInfo &IMMI;
  /// Keeps track of the estimated size of each MF in bytes inside the
  /// instrumentation module
  llvm::SmallDenseMap<const llvm::MachineFunction *, uint64_t, 8>
      IModuleFuncSizes;

  llvm::DenseMap<const llvm::MachineFunction *,
                 PatchLiftedRepresentationPass::PatchType>
  decidePatchingMethod(llvm::Module &TargetAppM,
                       llvm::ModuleAnalysisManager &TargetMAM);

public:
  PatchLiftedRepresentationPass(llvm::Module &IModule,
                                llvm::MachineModuleInfo &IMMI)
      : IModule(IModule), IMMI(IMMI) {};

  llvm::PreservedAnalyses run(llvm::Module &TargetAppM,
                              llvm::ModuleAnalysisManager &);
};

} // namespace luthier

#endif