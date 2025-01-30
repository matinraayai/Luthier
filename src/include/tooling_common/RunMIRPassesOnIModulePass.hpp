//===-- RunMIRPassesOnIModulePass.hpp -------------------------------------===//
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
/// This file describes the <tt>RunMIRPassesOnIModulePass</tt>, which
/// runs the modified code gen pipeline on the instrumentation module to
/// generate machine IR that will later be patched into the target module.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOLING_COMMON_RUN_MIR_PASSES_ON_IMODULE_PASS_HPP
#define LUTHIER_TOOLING_COMMON_RUN_MIR_PASSES_ON_IMODULE_PASS_HPP
#include "luthier/intrinsic/IntrinsicProcessor.h"
#include <AMDGPUTargetMachine.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/PassManager.h>

namespace luthier {

/// \brief This pass runs the modified code gen pipeline on the instrumentation
/// module to generate machine IR that will later be patched into the target
/// module.
class RunMIRPassesOnIModulePass
    : public llvm::PassInfoMixin<RunMIRPassesOnIModulePass> {
private:
  /// The target machine of the module
  llvm::GCNTargetMachine &TM;
  /// The instrumentation module being worked on
  llvm::Module &IModule;
  /// A \c llvm::MachineModuleInfoWrapperPass that will house the final
  /// generated MIR of the instrumentation module
  llvm::MachineModuleInfoWrapperPass &IMMIWP;
  /// The legacy pass manager used to run the codegen pipeline
  llvm::legacy::PassManager &ILegacyPM;

public:
  RunMIRPassesOnIModulePass(llvm::GCNTargetMachine &TM, llvm::Module &IModule,
                            llvm::MachineModuleInfoWrapperPass &MMIWP,
                            llvm::legacy::PassManager &ILegacyPM)
      : TM(TM), IModule(IModule), IMMIWP(MMIWP), ILegacyPM(ILegacyPM) {};

  llvm::PreservedAnalyses run(llvm::Module &TargetAppM,
                              llvm::ModuleAnalysisManager &TargetMAM);
};

} // namespace luthier

#endif