//===-- StateValueLocationIntervalsPass.hpp -------------------------------===//
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
/// This file describes the Lifted Representation State Value Locations,
/// which calculates the location of the state value register at each slot index
/// interval of a <tt>LiftedRepresentation</tt>.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOLING_COMMON_LR_STATE_VALUE_LOCATIONS_HPP
#define LUTHIER_TOOLING_COMMON_LR_STATE_VALUE_LOCATIONS_HPP
#include "hsa/LoadedCodeObject.hpp"
#include "luthier/LRRegisterLiveness.h"
#include "luthier/LiftedRepresentation.h"
#include <llvm/CodeGen/SlotIndexes.h>
#include <luthier/hsa/LoadedCodeObjectDeviceFunction.h>
#include <luthier/hsa/LoadedCodeObjectKernel.h>
#include "tooling_common/PreKernelEmitter.hpp"

namespace luthier {

struct StateValueStorage
    : public std::enable_shared_from_this<StateValueStorage> {
public:
  enum StateValueStorageKind {
    SVS_SINGLE_VGPR, /// The state value is stored in a free VGPR
    SVS_TWO_AGPRs,   /// The state value is stored in an AGPR, with a free AGPR
                     /// to use as a temp storage for the app's VGPR when it
                     /// spills
    SVS_SINGLE_AGPR_WITH_THREE_SGPRS, /// The state value stored is in an AGPR,
                                      /// with two SGPRs holding the
                                      /// FLAT SCRATCH base address of the
                                      /// kernel, and one SGPR holding the
                                      /// instrumentation stack pointer; SGPRs
                                      /// are used to spill an app VGPR
                                      /// two SGPRs holding the base address to
                                      /// spill a live VGPR
    SVS_SPILLED_WITH_THREE_SGPRS,     /// The state value is spilled onto the
                                      /// instrumentation stack, and two SGPRs
                                      /// holding the flat scratch address and
                                      /// a single SGPR for the instrumentation
                                      /// stack pointer
  };

private:
  StateValueStorageKind Kind;

public:
  StateValueStorageKind getKind() const { return Kind; }

  explicit StateValueStorage(StateValueStorageKind Kind) : Kind(Kind){};

  virtual llvm::MCRegister getStateValueStorageReg() const = 0;
};

struct VGPRValueStorage : public StateValueStorage {
public:
  llvm::MCRegister StorageVGPR{};

  /// method for providing LLVM RTTI
  [[nodiscard]] static bool classof(const StateValueStorage *S) {
    return S->getKind() == SVS_SINGLE_VGPR;
  }

  explicit VGPRValueStorage(llvm::MCRegister StorageVGPR)
      : StorageVGPR(StorageVGPR), StateValueStorage(SVS_SINGLE_VGPR){};

  /// \return the register where the state value is stored; 0 means the
  /// state value is spilled on the stack
  llvm::MCRegister getStateValueStorageReg() const override {
    return StorageVGPR;
  }
};

struct TwoAGPRValueStorage : public StateValueStorage {
public:
  llvm::MCRegister StorageAGPR{};

  llvm::MCRegister TempAGPR{};

  /// method for providing LLVM RTTI
  [[nodiscard]] static bool classof(const StateValueStorage *S) {
    return S->getKind() == SVS_TWO_AGPRs;
  }

  TwoAGPRValueStorage(llvm::MCRegister StorageAGPR, llvm::MCRegister TempAGPR)
      : StorageAGPR(StorageAGPR), TempAGPR(TempAGPR),
        StateValueStorage(SVS_TWO_AGPRs){};

  llvm::MCRegister getStateValueStorageReg() const override {
    return StorageAGPR;
  }
};

struct AGPRWithThreeSGPRSValueStorage : public StateValueStorage {
public:
  llvm::MCRegister StorageAGPR{};

  llvm::MCRegister FlatScratchSGPRHigh{};

  llvm::MCRegister FlatScratchSGPRLow{};

  llvm::MCRegister InstrumentationStackPointer{};

  /// method for providing LLVM RTTI
  [[nodiscard]] static bool classof(const StateValueStorage *S) {
    return S->getKind() == SVS_SINGLE_AGPR_WITH_THREE_SGPRS;
  }

  AGPRWithThreeSGPRSValueStorage(llvm::MCRegister StorageAGPR,
                                 llvm::MCRegister FlatScratchSGPRHigh,
                                 llvm::MCRegister FlatScratchSGPRLow,
                                 llvm::MCRegister InstrumentationStackPointer)
      : StorageAGPR(StorageAGPR), FlatScratchSGPRHigh(FlatScratchSGPRHigh),
        FlatScratchSGPRLow(FlatScratchSGPRLow),
        InstrumentationStackPointer(InstrumentationStackPointer),
        StateValueStorage(SVS_SINGLE_AGPR_WITH_THREE_SGPRS){};

  llvm::MCRegister getStateValueStorageReg() const override {
    return StorageAGPR;
  }
};

struct SpilledWithThreeSGPRsValueStorage : public StateValueStorage {
public:
  llvm::MCRegister FlatScratchSGPRHigh{};

  llvm::MCRegister FlatScratchSGPRLow{};

  llvm::MCRegister InstrumentationStackPointer{};

  /// method for providing LLVM RTTI
  [[nodiscard]] static bool classof(const StateValueStorage *S) {
    return S->getKind() == SVS_SPILLED_WITH_THREE_SGPRS;
  }

  SpilledWithThreeSGPRsValueStorage(llvm::MCRegister FlatScratchSGPRHigh,
                                  llvm::MCRegister FlatScratchSGPRLow,
                                  llvm::MCRegister InstrumentationStackPointer)
      : FlatScratchSGPRHigh(FlatScratchSGPRHigh),
        FlatScratchSGPRLow(FlatScratchSGPRLow),
        InstrumentationStackPointer(InstrumentationStackPointer),
        StateValueStorage(SVS_SPILLED_WITH_THREE_SGPRS){};

  llvm::MCRegister getStateValueStorageReg() const override { return {}; }
};

/// \brief An interval inside the basic block
struct StateValueStorageSegment {
private:
  /// Start point of the interval (inclusive)
  llvm::SlotIndex Start;
  /// End point of the interval (exclusive)
  llvm::SlotIndex End;
  /// Where and how the state value is stored
  std::shared_ptr<StateValueStorage> SVS;

public:
  StateValueStorageSegment(llvm::SlotIndex S, llvm::SlotIndex E,
                           std::shared_ptr<StateValueStorage> SVS)
      : Start(S), End(E), SVS(std::move(SVS)) {
    if (S > E)
      llvm::report_fatal_error("Cannot create empty or backwards segment");
  }

  [[nodiscard]] const StateValueStorage &getSVS() const { return *SVS; }

  /// Return true if the index is covered by this segment.
  [[nodiscard]] bool contains(llvm::SlotIndex I) const {
    return Start <= I && I < End;
  }

  /// Return true if the given interval, [S, E), is covered by this segment.
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

/// \brief
struct InsertionPointStateValueDescriptor {
  /// The VGPR where the state value will be loaded into
  llvm::MCRegister StateValueVGPR{};
  /// Whether the State Value VGPR clobbers any live registers
  bool ClobbersAppRegister{false};
  /// Where the state value is located before being loaded into the VGPR
  StateValueStorageSegment StateValueLocation;
};

class LRStateValueLocations {

private:
  /// The Lifted Representation being processed
  const LiftedRepresentation &LR;

  /// The loaded code object being processed
  const hsa::LoadedCodeObject LCO;

  /// The reg liveness analysis of the \c LR
  const LRRegisterLiveness &RegLiveness;

  /// The kernels of the loaded code object
  llvm::SmallVector<
      std::pair<const hsa::LoadedCodeObjectKernel *, llvm::MachineFunction *>,
      1>
      Kernels;

  /// The device functions of the loaded code object
  llvm::SmallVector<std::pair<const hsa::LoadedCodeObjectDeviceFunction *,
                              llvm::MachineFunction *>,
                    1>
      DeviceFunctions;

  PreKernelEmissionDescriptor & PKInfo;

  /// A set of physical registers accessed by all hooks that weren't part
  /// of the hook insertion point's live-ins
  /// These are kept track of to ensure they don't get clobbered in
  /// instrumentation routines in order to give tools an
  /// "as correct as possible" view of the kernel
  /// In practice, this set should only become populated in very rare
  /// scenarios, as tools are more interested in registers that are live
  /// at each instruction
  const llvm::LivePhysRegs &HooksAccessedPhysicalRegistersNotInLiveIns;

  /// Slot index tracking for each machine instruction of each function in \c LR
  /// This allows us to create intervals to keep track of the location
  /// of the state value
  llvm::SmallDenseMap<llvm::Function *, std::unique_ptr<llvm::SlotIndexes>>
      FunctionsSlotIndexes{};

  /// Keeps track of where the state value and flat scratch registers will
  /// be stored in each function involved in the \c LR
  llvm::SmallDenseMap<llvm::MachineFunction *,
                      llvm::SmallVector<StateValueStorageSegment>>
      ValueStateRegAndFlatScratchIntervals{};

  /// Mapping between the MIs of the target app getting instrumented and their
  /// hooks
  const llvm::DenseMap<llvm::MachineInstr *, llvm::Function *> &MIToHookMap;

  /// Contains a mapping between the hook insertion point's MI and a
  /// struct describing which VGPR will the state value be loaded into,
  /// as well as its storage at that location
  llvm::DenseMap<const llvm::MachineInstr *, InsertionPointStateValueDescriptor>
      HookMIToValueStateInterval{};

  /// \brief Tries to find a fixed location to store the instrumentation value
  /// register in a VPGR or an AGPR, as well as an SGPR pair to store the flat
  /// address pointing to the beginning of the instrumentation private segment
  /// \details The order of searching for these fixed locations is as follows:
  /// 1. Find an unused VGPR. This is the ideal scenario, as an unused VGPR has
  /// enough space to accommodate the prologue/epilogue of an instrumentation
  /// hook. No additional code needs to be inserted inside the kernel itself
  /// for managing the value register, and the flat scratch register will be
  /// 0\n
  /// 2. If no unused VGPRs are found, then this routine will find the next
  /// unused AGPR. This usually comes at no cost to the occupancy, as the app
  /// will get the same amount of AGPRs as it gets VGPRs. In gfx90A-, since
  /// AGPRs cannot be used directly by vector instructions and have to be moved
  /// to a VGPR, it is likely that a single VGPR must be spilled to accommodate
  /// this issue. For this scenario, an unused SGPR pair must also be found to
  /// hold the address of the instrumentation's private segment. No other
  /// action is necessary by the value manager except generating
  /// prologue/epilogue code.\n
  /// 3. If no unused V/AGPRs are found in the kernel, then as a last resort,
  /// this function tries to find an unused SGPR pair for the instrumentation
  /// scratch address to be stored. In this scenario, besides emitting prologue
  /// /epiloge code, the value manager must also emit code in the kernel that
  /// moves the state value register around or spill it
  ///
  /// TODO: This function must take an argument indicating whether the tool
  /// writer wants to respect the original kernel's granulated register usage
  /// or not.
  [[nodiscard]] std::shared_ptr<StateValueStorage>
  findFixedStateValueStorageLocation(
      llvm::ArrayRef<llvm::MachineFunction *> RelatedFunctions) const;

  LRStateValueLocations(
      const luthier::LiftedRepresentation &LR, hsa::LoadedCodeObject LCO,
      const llvm::DenseMap<llvm::MachineInstr *, llvm::Function *> &MIToHookMap,
      const llvm::LivePhysRegs &HooksAccessedPhysicalRegistersNotInLiveIns,
      const luthier::LRRegisterLiveness &RegLiveness,
      PreKernelEmissionDescriptor &PKInfo);

  llvm::Error calculateStateValueLocations();

public:
  static llvm::Expected<std::unique_ptr<LRStateValueLocations>> create(
      const LiftedRepresentation &LR, const hsa::LoadedCodeObject &LCO,
      const llvm::DenseMap<llvm::MachineInstr *, llvm::Function *> &MIToHookMap,
      const llvm::LivePhysRegs &HooksAccessedPhysicalRegistersNotInLiveIns,
      const LRRegisterLiveness &RegLiveness,
      PreKernelEmissionDescriptor &PKInfo);

  const StateValueStorageSegment *
  getValueSegmentForInstr(llvm::MachineInstr &MI) const;

  [[nodiscard]] const InsertionPointStateValueDescriptor &
  getStateValueDescriptorOfHookInsertionPoint(
      const llvm::MachineInstr &MI) const;

  [[nodiscard]] hsa::LoadedCodeObject getLCO() const { return LCO; }
};

} // namespace luthier

#endif