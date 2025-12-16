//===-- ProcessIntrinsicsAtIRLevelPass.h ------------------------*- C++ -*-===//
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
/// This file describes the <tt>ProcessIntrinsicsAtIRLevelPass</tt>,
/// in charge of running the IR processing stage of Luthier intrinsics in
/// the instrumentation module.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOLING_PROCESS_INTRINSICS_AT_IR_LEVEL_PASS_H
#define LUTHIER_TOOLING_PROCESS_INTRINSICS_AT_IR_LEVEL_PASS_H
#include <GCNSubtarget.h>
#include <llvm/IR/PassManager.h>

namespace luthier {

/// \brief A pass that applies the IR processing stage of intrinsics
/// inside the instrumentation module
class ProcessIntrinsicsAtIRLevelPass
    : public llvm::PassInfoMixin<ProcessIntrinsicsAtIRLevelPass> {
private:
  const llvm::GCNTargetMachine &TM;

public:
  explicit ProcessIntrinsicsAtIRLevelPass(const llvm::GCNTargetMachine &TM)
      : TM(TM) {};

  llvm::PreservedAnalyses run(llvm::Module &IModule,
                              llvm::ModuleAnalysisManager &IMAM);
};

} // namespace luthier

#endif