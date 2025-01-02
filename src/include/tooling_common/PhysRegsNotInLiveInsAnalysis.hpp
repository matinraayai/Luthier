//===-- PhysRegsNotInLiveInsAnalysis.hpp ----------------------------------===//
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
/// This file defines the analysis pass which aggregates the accessed
/// physical registers in injected payloads that are not preserved by the
/// live-in set.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOLING_COMMON_PHYS_REGS_NOT_IN_LIVE_INS_ANALYSIS_HPP
#define LUTHIER_TOOLING_COMMON_PHYS_REGS_NOT_IN_LIVE_INS_ANALYSIS_HPP
#include <llvm/CodeGen/LivePhysRegs.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Target/TargetMachine.h>

namespace luthier {
class PhysRegsNotInLiveInsAnalysis
    : public llvm::AnalysisInfoMixin<PhysRegsNotInLiveInsAnalysis> {
private:
  friend llvm::AnalysisInfoMixin<PhysRegsNotInLiveInsAnalysis>;

  static llvm::AnalysisKey Key;

public:
  class Result {
    std::unique_ptr<llvm::LivePhysRegs> Regs;

    explicit Result(std::unique_ptr<llvm::LivePhysRegs> Regs)
        : Regs(std::move(Regs)) {}

    friend class PhysRegsNotInLiveInsAnalysis;

  public:
    [[nodiscard]] const llvm::LivePhysRegs &getPhysRegsNotInLiveIns() const {
      return *Regs;
    }

    bool invalidate(llvm::Module &, const llvm::PreservedAnalyses &,
                    llvm::ModuleAnalysisManager::Invalidator &) {
      return false;
    }
  };

  PhysRegsNotInLiveInsAnalysis() = default;

  Result run(llvm::Module &IModule, llvm::ModuleAnalysisManager &IMAM);
};

} // namespace luthier

#endif