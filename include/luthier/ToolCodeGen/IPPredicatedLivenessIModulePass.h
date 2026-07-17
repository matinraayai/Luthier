//===-- IPPredicatedLivenessIModulePass.h ----------------------*- C++ -*-===//
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
/// \file IPPredicatedLivenessIModulePass.h
/// \c InstrumentPrototype-level analysis that runs liveness analysis across
/// the target module's inter-procedural predicated control-flow graph,
/// tracking separate active-lane and inactive-lane live phys-reg sets at every
/// program point of interest. Computed per-AppMI and surfaced per-injected-
/// payload so the downstream
/// \c InjectedPayloadPreserveLiveRegsPass can decide what physical
/// registers the injected payload must preserve.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOL_CODE_GEN_IP_PREDICATED_LIVENESS_IMODULE_PASS_H
#define LUTHIER_TOOL_CODE_GEN_IP_PREDICATED_LIVENESS_IMODULE_PASS_H
#include "luthier/ToolCodeGen/InstrumentPrototype.h"
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/PassManager.h>
#include <llvm/MC/MCRegister.h>

namespace llvm {
class Function;
class MachineInstr;
} // namespace llvm

namespace luthier {

class PredicatedMachineBasicBlock;

/// \brief Per-payload live-register record captured at the program point
/// just before the payload runs (within a chain of payloads at the same
/// AppMI, each payload's record is captured "after later payloads have
/// already been stepped over and before this payload's own effects are
/// applied").
///
/// \c Active is the live set across active lanes (lanes where EXEC=1 at
/// this point); \c Inactive is the live set across inactive lanes
/// (lanes where EXEC=0). For ordinary instrumentation under the C calling
/// convention, only \c Active is needed for preservation — see
/// \c project_sva_vgpr_wwm_preload memory note for the WWM-payload
/// considerations.
struct PayloadLiveSets {
  llvm::DenseSet<llvm::MCPhysReg> Active;
  llvm::DenseSet<llvm::MCPhysReg> Inactive;
};

/// \brief Per-PredicatedMachineBasicBlock live-in record.
///
/// Captures the same Active/Inactive lane partition as
/// \c PayloadLiveSets, but keyed by basic block rather than by payload.
/// Stored as \c SmallVector so consumers can borrow via \c ArrayRef
/// without copying.
struct PMBBLiveIns {
  llvm::SmallVector<llvm::MCPhysReg, 16> Active;
  llvm::SmallVector<llvm::MCPhysReg, 16> Inactive;
};

class IModuleIPPredicatedLivenessAnalysis;

/// \brief Result of \c IModuleIPPredicatedLivenessAnalysis.
///
/// Walks the target module's \c IPPredicatedCFG backward to fixed point,
/// tracking active/inactive lane liveness with the following per-PMBB
/// rules:
///   - Vector PMBB → step backward updates only the active set.
///   - Scalar PMBB → step backward updates both active and inactive sets.
///   - EXEC-modifying MI → "complete flip" (e.g. \c S_NOT_B64 exec,exec)
///     swaps active and inactive; any other EXEC write conservatively
///     unions both sets into both.
///   - Insertion-point AppMI → for each attached payload in reverse
///     execution order, snapshot the current sets (cached per-payload)
///     before stepping backward over the payload's declared
///     Reads/Writes from \c InjectedPayloadSideEffectsAnalysis.
///
/// If any PMBB has unresolved inter-procedural edges, the analysis falls
/// back to per-function local mode: every return-block live-out is
/// initialised to the function's allocatable GPR set (per the
/// \c amdgpu-num-{sgpr,vgpr} attributes plus reserved-but-not-read-only
/// registers from MRI) and dataflow is intra-procedural only.
class IModuleIPPredicatedLiveness {
public:
  using PayloadLiveSetsMap =
      llvm::DenseMap<const llvm::Function *, PayloadLiveSets>;
  using PMBBLiveInsMap =
      llvm::DenseMap<const PredicatedMachineBasicBlock *, PMBBLiveIns>;

private:
  friend class IModuleIPPredicatedLivenessAnalysis;

  PayloadLiveSetsMap LiveSetsByPayload;
  PMBBLiveInsMap LiveInsByPMBB;
  /// True iff the dataflow ran in fully-discovered (inter-procedural) mode.
  /// False means it fell back to per-function local mode.
  bool ResultFullyDiscovered{false};

public:
  IModuleIPPredicatedLiveness() = default;

  /// \return per-payload live-set record at the program point just before
  /// \p Payload's effects apply, or \c nullptr if the payload has no
  /// recorded entry.
  [[nodiscard]] const PayloadLiveSets *
  getLiveSetsForPayload(const llvm::Function &Payload) const {
    auto It = LiveSetsByPayload.find(&Payload);
    return It == LiveSetsByPayload.end() ? nullptr : &It->second;
  }

  /// \return true iff the IPPredCFG was fully discovered at analysis time.
  /// When false, results are produced by a per-function local fallback.
  [[nodiscard]] bool isFullyDiscovered() const { return ResultFullyDiscovered; }

  [[nodiscard]] const PayloadLiveSetsMap &getMap() const {
    return LiveSetsByPayload;
  }

  /// \return ArrayRef into the converged per-PMBB Active-lane live-in set,
  /// or an empty ArrayRef if no entry was recorded for \p PMBB.
  [[nodiscard]] llvm::ArrayRef<llvm::MCPhysReg>
  getPMBBLiveInsActive(const PredicatedMachineBasicBlock &PMBB) const {
    auto It = LiveInsByPMBB.find(&PMBB);
    return It == LiveInsByPMBB.end() ? llvm::ArrayRef<llvm::MCPhysReg>{}
                                     : llvm::ArrayRef<llvm::MCPhysReg>(
                                           It->second.Active);
  }

  /// \return ArrayRef into the converged per-PMBB Inactive-lane live-in set,
  /// or an empty ArrayRef if no entry was recorded for \p PMBB.
  [[nodiscard]] llvm::ArrayRef<llvm::MCPhysReg>
  getPMBBLiveInsInactive(const PredicatedMachineBasicBlock &PMBB) const {
    auto It = LiveInsByPMBB.find(&PMBB);
    return It == LiveInsByPMBB.end() ? llvm::ArrayRef<llvm::MCPhysReg>{}
                                     : llvm::ArrayRef<llvm::MCPhysReg>(
                                           It->second.Inactive);
  }

  [[nodiscard]] const PMBBLiveInsMap &getPMBBLiveInsMap() const {
    return LiveInsByPMBB;
  }

  bool invalidate(InstrumentPrototype &, const llvm::PreservedAnalyses &PA,
                  InstrumentPrototypeAnalysisManager::Invalidator &);
};

/// \brief \c InstrumentPrototype-level analysis that computes, for the target
/// module of an \c InstrumentPrototype, the per-payload and per-PMBB live
/// physical-register sets across active and inactive lane partitions.
class IModuleIPPredicatedLivenessAnalysis
    : public llvm::AnalysisInfoMixin<IModuleIPPredicatedLivenessAnalysis> {
  friend llvm::AnalysisInfoMixin<IModuleIPPredicatedLivenessAnalysis>;

  static llvm::AnalysisKey Key;

public:
  IModuleIPPredicatedLivenessAnalysis() = default;

  using Result = IModuleIPPredicatedLiveness;

  Result run(InstrumentPrototype &IP,
             InstrumentPrototypeAnalysisManager &IPAM);
};

} // namespace luthier

#endif
