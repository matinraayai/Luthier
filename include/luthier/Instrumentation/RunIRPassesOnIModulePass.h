//===-- RunIRPassesOnIModulePass.h ------------------------------*- C++ -*-===//
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
/// Describes the \c RunIRPassesOnIModulePass in charge of running the
/// IR passes of the instrumentation module as part of the target application
/// instrumentation pipeline.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_INSTRUMENTATION_RUN_IR_PASSES_ON_IMODULE_PASS_H
#define LUTHIER_INSTRUMENTATION_RUN_IR_PASSES_ON_IMODULE_PASS_H
#include <llvm/IR/PassManager.h>
#include <llvm/Passes/OptimizationLevel.h>

namespace luthier {

/// \brief Runs the IR pipeline of the instrumentation module as part of the
/// target application pipeline
class RunIRPassesOnIModulePass
    : public llvm::PassInfoMixin<RunIRPassesOnIModulePass> {
private:
  /// The instrumentation module being operated on
  llvm::Module &IModule;

  /// The optimization level applied to the IR
  llvm::OptimizationLevel OptLevel;

public:
  RunIRPassesOnIModulePass(llvm::Module &IModule,
                           llvm::OptimizationLevel OptLevel)
      : IModule(IModule), OptLevel(OptLevel) {};

  static llvm::StringRef name() { return "LuthierRunIRPassesOnIModulePass"; }

  llvm::PreservedAnalyses run(llvm::Module &TargetAppM,
                              llvm::ModuleAnalysisManager &TargetMAM);
};
} // namespace luthier

#endif