//===-- IPVectorRegLiveness.h ------------------------------------*- C++-*-===//
// Copyright 2022-2026 @ Northeastern University Computer Architecture Lab
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
/// \file IPVectorRegLiveness.h
/// Describes the \c IPVectorRegLiveness and its corresponding analysis,
/// which calculates the register live-in sets for each \c llvm::MachineInstr
/// of all <tt>llvm::MachineFunction</tt>s inside a <tt>llvm::MachineModuleInfo.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOLING_IP_VECTOR_REG_LIVENESS_H
#define LUTHIER_TOOLING_IP_VECTOR_REG_LIVENESS_H
#include "luthier/Tooling/IPPredicatedCFG.h"
#include <llvm/ADT/DenseMap.h>
#include <llvm/CodeGen/LivePhysRegs.h>
#include <llvm/CodeGen/MachineModuleInfo.h>
#include <llvm/IR/PassManager.h>

namespace luthier {

class IPVectorRegLiveness {
private:
  llvm::DenseMap<std::reference_wrapper<const PredicatedMachineBasicBlock>,
                 std::vector<llvm::MachineBasicBlock::RegisterMaskPair>>
      PredMBBLivenessMap{};

  /// Add live-out registers of basic block \p MBB to \p LiveUnits.
  void addBlockLiveOuts(const PredicatedMachineBasicBlock &MBB,
                        llvm::LiveRegUnits &LiveUnits) const;

  void addBlockLiveIns(const PredicatedMachineBasicBlock &MBB,
                       llvm::LiveRegUnits &LiveUnits) const;

  void addBlockLiveIns(llvm::LivePhysRegs &LPR,
                       const PredicatedMachineBasicBlock &PredMBB) const;

  void addLiveOutsNoPristines(llvm::LivePhysRegs &LPR,
                              const PredicatedMachineBasicBlock &MBB) const;

  void computeLiveIns(llvm::LivePhysRegs &LiveRegs,
                      const PredicatedMachineBasicBlock &MBB);

  std::vector<llvm::MachineBasicBlock::RegisterMaskPair>
  computeAndAddLiveIns(llvm::LivePhysRegs &LiveRegs,
                       const PredicatedMachineBasicBlock &MBB);

  void recomputePredMBBLiveIns(const IPPredicatedCFG &IPPredCFG);

  bool recomputeLiveIns(const PredicatedMachineBasicBlock &PredMBB);

public:
  llvm::ArrayRef<llvm::MachineBasicBlock::RegisterMaskPair>
  getPredMBBLiveIns(const PredicatedMachineBasicBlock &PredMBB) const {
    assert(PredMBBLivenessMap.contains(PredMBB) &&
           "Failed to find the predicated MBB in the liveness map");
    return PredMBBLivenessMap.at(PredMBB);
  }

  bool isLiveIn(const PredicatedMachineBasicBlock &PredMBB,
                llvm::MCRegister Reg,
                llvm::LaneBitmask LaneMask = llvm::LaneBitmask::getAll()) const;

  void addLiveIns(const PredicatedMachineBasicBlock &PredMBB,
                  llvm::LiveRegUnits &LRU) const;

  void addLiveOuts(const PredicatedMachineBasicBlock &PredMBB,
                   llvm::LiveRegUnits &LRU) const;

  [[nodiscard]] std::vector<llvm::MachineBasicBlock::RegisterMaskPair>
  getPredMBBLiveOuts(const PredicatedMachineBasicBlock &PredMBB) const;

  IPVectorRegLiveness(llvm::Module &M, llvm::ModuleAnalysisManager &MAM);

  bool invalidate(llvm::Module &M, const llvm::PreservedAnalyses &PA,
                  llvm::ModuleAnalysisManager::Invalidator &Inv);
};

/// \brief the analysis pass used to obtain the \c IPVectorRegLiveness
class IPVectorRegLivenessAnalysis
    : public llvm::AnalysisInfoMixin<IPVectorRegLivenessAnalysis> {
private:
  friend AnalysisInfoMixin<IPVectorRegLivenessAnalysis>;

  static llvm::AnalysisKey Key;

public:
  using Result = IPVectorRegLiveness;

  IPVectorRegLivenessAnalysis() = default;

  Result run(llvm::Module &M, llvm::ModuleAnalysisManager &MAM);
};

/// \brief Used to print and test the liveness analysis
class IPVectorRegLivenessPrinter
    : public llvm::PassInfoMixin<IPVectorRegLivenessPrinter> {

private:
  llvm::raw_ostream &OS;

public:
  explicit IPVectorRegLivenessPrinter(llvm::raw_ostream &OS) : OS(OS) {};

  llvm::PreservedAnalyses run(llvm::Module &M,
                              llvm::ModuleAnalysisManager &MAM);
};

} // namespace luthier

#endif