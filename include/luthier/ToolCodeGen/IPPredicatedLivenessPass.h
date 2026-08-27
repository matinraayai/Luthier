//===-- IPPredicatedLivenessPass.h ------------------------------*- C++ -*-===//
// Copyright @ Northeastern University Computer Architecture Lab
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
/// \c Prototype-level analysis that runs liveness analysis across
/// the target module's inter-procedural predicated control-flow graph,
/// tracking two live phys-reg sets at every predicated basic block: one
/// for lanes currently active under the EXEC mask and one for lanes
/// currently masked off.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOL_CODE_GEN_IP_PREDICATED_LIVENESS_PASS_H
#define LUTHIER_TOOL_CODE_GEN_IP_PREDICATED_LIVENESS_PASS_H
#include "luthier/ToolCodeGen/Prototype.h"
#include <llvm/ADT/DenseMap.h>
#include <llvm/CodeGen/LivePhysRegs.h>
#include <llvm/IR/PassManager.h>
#include <memory>

namespace llvm {
class Function;
class MachineInstr;
} // namespace llvm

namespace luthier {

class PredicatedMachineBasicBlock;

class IPPredicatedLivenessAnalysis;

/// \brief Result of \c IPPredicatedLivenessAnalysis.
///
/// Walks the target module's \c IPPredicatedCFG backward to fixed point,
/// tracking two physical-register live sets per PMBB:
///   - \em Active — regs live in the currently EXEC-on lanes.
///   - \em Inactive — regs live in the currently EXEC-off lanes.
class IPPredicatedLiveness {
public:
  /// Per-PMBB converged live-in set, keyed for one lane partition.
  using PMBBLiveInsMap =
      llvm::DenseMap<const PredicatedMachineBasicBlock *,
                     std::unique_ptr<llvm::LivePhysRegs>>;

private:
  friend class IPPredicatedLivenessAnalysis;
  PMBBLiveInsMap ActiveLiveInsByPMBB;
  PMBBLiveInsMap InactiveLiveInsByPMBB;
  /// True iff the dataflow ran in fully-discovered (inter-procedural) mode.
  /// False means it fell back to per-function local mode.
  bool ResultFullyDiscovered{false};

public:
  IPPredicatedLiveness() = default;

  /// \return true iff the IPPredCFG was fully discovered at analysis time.
  /// When false, results are produced by a per-function local fallback.
  [[nodiscard]] bool isFullyDiscovered() const { return ResultFullyDiscovered; }

  /// \return pointer to the converged per-PMBB live-in set for the
  /// EXEC-on (Active) lane partition, or \c nullptr if no entry was
  /// recorded for \p PMBB.
  [[nodiscard]] const llvm::LivePhysRegs *
  getPMBBActiveLiveIns(const PredicatedMachineBasicBlock &PMBB) const {
    auto It = ActiveLiveInsByPMBB.find(&PMBB);
    return It == ActiveLiveInsByPMBB.end() ? nullptr : It->second.get();
  }

  /// \return pointer to the converged per-PMBB live-in set for the
  /// EXEC-off (Inactive) lane partition, or \c nullptr if no entry was
  /// recorded for \p PMBB.
  [[nodiscard]] const llvm::LivePhysRegs *
  getPMBBInactiveLiveIns(const PredicatedMachineBasicBlock &PMBB) const {
    auto It = InactiveLiveInsByPMBB.find(&PMBB);
    return It == InactiveLiveInsByPMBB.end() ? nullptr : It->second.get();
  }

  [[nodiscard]] const PMBBLiveInsMap &getPMBBActiveLiveInsMap() const {
    return ActiveLiveInsByPMBB;
  }

  [[nodiscard]] const PMBBLiveInsMap &getPMBBInactiveLiveInsMap() const {
    return InactiveLiveInsByPMBB;
  }

  bool invalidate(Prototype &, const llvm::PreservedAnalyses &PA,
                  PrototypeAnalysisManager::Invalidator &);
};

/// \brief \c Prototype-level analysis that computes, for the target
/// module of an \c Prototype, the per-payload and per-PMBB live
/// physical-register sets.
class IPPredicatedLivenessAnalysis
    : public llvm::AnalysisInfoMixin<IPPredicatedLivenessAnalysis> {
  friend llvm::AnalysisInfoMixin<IPPredicatedLivenessAnalysis>;

  static llvm::AnalysisKey Key;

public:
  IPPredicatedLivenessAnalysis() = default;

  using Result = IPPredicatedLiveness;

  Result run(Prototype &IP,
             PrototypeAnalysisManager &IPAM);
};

/// \brief Prints the result of \c IPPredicatedLivenessAnalysis for \p IP.
class IPPredicatedLivenessPrinter
    : public llvm::PassInfoMixin<IPPredicatedLivenessPrinter> {
  llvm::raw_ostream &OS;

public:
  explicit IPPredicatedLivenessPrinter(llvm::raw_ostream &OS) : OS(OS) {}

  llvm::PreservedAnalyses run(Prototype &IP, PrototypeAnalysisManager &IPAM);
};

} // namespace luthier

#endif
