//===-- SVStorageAndLoadLocations.hpp -------------------------------------===//
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
/// This file describes the Lifted Representation State Value Storage and Load
/// Locations, which calculates the storage locations of the state value
/// register at each slot index interval of a <tt>LiftedRepresentation</tt>, as
/// well as the VGPR it will be loaded to at each instrumentation point.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOLING_COMMON_SV_STORAGE_AND_LOAD_LOCATIONS_HPP
#define LUTHIER_TOOLING_COMMON_SV_STORAGE_AND_LOAD_LOCATIONS_HPP
#include "IModuleIRGeneratorPass.hpp"
#include "MMISlotIndexesAnalysis.hpp"
#include "hsa/LoadedCodeObject.hpp"
#include "luthier/tooling/AMDGPURegisterLiveness.h"
#include "luthier/tooling/LiftedRepresentation.h"
#include "tooling_common/PrePostAmbleEmitter.hpp"
#include "tooling_common/StateValueArrayStorage.hpp"
#include <llvm/CodeGen/SlotIndexes.h>
#include <llvm/IR/PassManager.h>
#include <luthier/hsa/LoadedCodeObjectDeviceFunction.h>
#include <luthier/hsa/LoadedCodeObjectKernel.h>

namespace luthier {

/// \brief Where the state value array is stored in an interval inside a
/// machine basic block
struct StateValueStorageSegment {
private:
  /// Start point of the interval (inclusive)
  llvm::SlotIndex Start;
  /// End point of the interval (exclusive)
  llvm::SlotIndex End;
  /// Where and how the state value is stored
  std::shared_ptr<StateValueArrayStorage> SVS;

public:
  StateValueStorageSegment(llvm::SlotIndex S, llvm::SlotIndex E,
                           std::shared_ptr<StateValueArrayStorage> SVAS)
      : Start(S), End(E), SVS(std::move(SVAS)) {
    if (S > E)
      llvm::report_fatal_error("Cannot create empty or backwards segment");
  }

  [[nodiscard]] llvm::SlotIndex begin() const { return Start; }

  [[nodiscard]] llvm::SlotIndex end() const { return End; }

  /// \returns the state value array storage of this MBB interval
  [[nodiscard]] const StateValueArrayStorage &getSVS() const { return *SVS; }

  /// \return true if the index is covered by this segment, false otherwise
  [[nodiscard]] bool contains(llvm::SlotIndex I) const {
    return Start <= I && I < End;
  }

  /// \return true if the given interval, [S, E), is covered by this segment,
  /// false otherwise
  [[nodiscard]] bool containsInterval(llvm::SlotIndex S,
                                      llvm::SlotIndex E) const {
    if (S > E)
      llvm::report_fatal_error("Backwards interval");
    return (Start <= S && S < End) && (Start < E && E <= End);
  }

  bool operator<(const StateValueStorageSegment &Other) const {
    return std::tie(Start, End) < std::tie(Other.Start, Other.End);
  }

  bool operator==(const StateValueStorageSegment &Other) const {
    return Start == Other.Start && End == Other.End;
  }

  bool operator!=(const StateValueStorageSegment &Other) const {
    return !(*this == Other);
  }
};

/// \brief describes where the state value storage will be loaded for use
/// by an injection payload, as well as where the state value array is stored
/// at the instrumentation point
struct InstPointSVALoadPlan {
  /// The VGPR where the state value will be loaded into
  llvm::MCRegister StateValueArrayLoadVGPR{};
  /// Whether or not the state value array load VGPR will clobber the app's
  /// live VGPR and must be stored somewhere
  bool LoadDestClobbersAppVGPR{};
  /// Where the state value is located before being loaded into the VGPR
  StateValueArrayStorage &StateValueStorageLocation;
};

/// \brief an analysis on a \c LiftedRepresentation that determines where the
/// state value array is stored at each instruction of the
/// \c LiftedRepresentation as well as where the state value will be loaded
/// at each instrumentation point.
class SVStorageAndLoadLocations {
private:
  /// Keeps track of how and where the state value array is stored in each
  /// function inside in the \c LR
  llvm::DenseMap<const llvm::MachineBasicBlock *,
                 llvm::SmallVector<StateValueStorageSegment>>
      StateValueStorageIntervals{};

  /// Contains a mapping between the instrumentation point MIs of the
  /// \c and the plan to load the state value array from its storage to
  /// the target VGPR to be used by the injected payload function
  llvm::DenseMap<const llvm::MachineInstr *, InstPointSVALoadPlan>
      InstPointSVSLoadPlans{};

public:
  SVStorageAndLoadLocations() = default;

  /// calculates the storage and load locations of the state value array
  /// \return an \c llvm::Error indication the success of failure of the
  /// operation
  llvm::Error calculate(
      const llvm::MachineModuleInfo &TargetMMI, const llvm::Module &TargetM,
      const MMISlotIndexesAnalysis::Result &SlotIndexes,
      const AMDGPURegisterLiveness &RegLiveness,
      const InjectedPayloadAndInstPoint &IPIP, FunctionPreambleDescriptor &FPD,
      const llvm::LivePhysRegs &AccessedPhysicalRegistersNotInLiveIns);

  /// Given the \p MBB of the \c LiftedRepresentation being worked on by this
  /// analysis, returns the state value array storage of every instruction
  /// interval inside the \p MBB
  /// \param MBB a basic block that belongs to the \c LiftedRepresentation
  /// of this analysis
  /// \return the state value array storage of every instruction
  /// interval inside the \p MBB or an empty \c llvm::ArrayRef if the \p MBB
  /// is not part of the \c LiftedRepresentation being analyzed
  [[nodiscard]] llvm::ArrayRef<StateValueStorageSegment>
  getStorageIntervals(const llvm::MachineBasicBlock &MBB) const;

  /// \return state value array load plan associated with instrumentation
  /// point \p MI or \c nullptr if the passed \p MI is not an instrumentation
  /// point
  [[nodiscard]] const InstPointSVALoadPlan *
  getStateValueArrayLoadPlanForInstPoint(const llvm::MachineInstr &MI) const;

  bool invalidate(llvm::Module &, const llvm::PreservedAnalyses &,
                  llvm::ModuleAnalysisManager::Invalidator &) {
    return false;
  }
};

class LRStateValueStorageAndLoadLocationsAnalysis
    : public llvm::AnalysisInfoMixin<
          LRStateValueStorageAndLoadLocationsAnalysis> {
  friend llvm::AnalysisInfoMixin<LRStateValueStorageAndLoadLocationsAnalysis>;

  static llvm::AnalysisKey Key;

public:
  using Result = SVStorageAndLoadLocations;

  Result run(llvm::Module &M, llvm::ModuleAnalysisManager &);
};

} // namespace luthier

#endif