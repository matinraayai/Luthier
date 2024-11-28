//===-- ProcessIntrinsicUsersAtIRLevelPass.hpp ----------------------------===//
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
/// This file describes the <tt>ProcessIntrinsicUsersAtIRLevelPass</tt>,
/// in charge of running the IR processing stage of Luthier intrinsics.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOLING_COMMON_PROCESS_INTRINSIC_USERS_AT_IR_LEVEL_PASS_HPP
#define LUTHIER_TOOLING_COMMON_PROCESS_INTRINSIC_USERS_AT_IR_LEVEL_PASS_HPP
#include <GCNSubtarget.h>
#include <llvm/IR/PassManager.h>

namespace luthier {

class ProcessIntrinsicUsersAtIRLevelPass
    : public llvm::PassInfoMixin<ProcessIntrinsicUsersAtIRLevelPass> {
private:
  const llvm::GCNTargetMachine &TM;

public:
  explicit ProcessIntrinsicUsersAtIRLevelPass(const llvm::GCNTargetMachine &TM)
      : TM(TM) {};

  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &);
};

} // namespace luthier

#endif