//===-- RunIRPassesOnIModulePass.hpp --------------------------------------===//
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
/// This file describes the <tt>RunIRPassesOnIModulePass</tt>,
/// in charge of running all IR passes on the instrumentation module.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOLING_COMMON_RUN_IR_PASSES_ON_IMODULE_PASS_HPP
#define LUTHIER_TOOLING_COMMON_RUN_IR_PASSES_ON_IMODULE_PASS_HPP
#include "luthier/Intrinsic/IntrinsicProcessor.h"
#include "luthier/tooling/InstrumentationTask.h"
#include <llvm/IR/PassManager.h>
#include <llvm/Target/TargetMachine.h>

namespace luthier {

class RunIRPassesOnIModulePass
    : public llvm::PassInfoMixin<RunIRPassesOnIModulePass> {
private:
  llvm::GCNTargetMachine &TM;
  const InstrumentationTask &Task;
  const llvm::StringMap<IntrinsicProcessor> &IntrinsicProcessors;
  llvm::Module &IModule;

public:
  RunIRPassesOnIModulePass(
      const InstrumentationTask &Task,
      const llvm::StringMap<IntrinsicProcessor> &IntrinsicProcessors,
      llvm::GCNTargetMachine &TM, llvm::Module &IModule);

  llvm::PreservedAnalyses run(llvm::Module &TargetAppM,
                              llvm::ModuleAnalysisManager &);
};
} // namespace luthier

#endif