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
/// This file describes the Lifted Representation State Value Storage and Load
/// Locations, which calculates the storage locations of the state value
/// register at each slot index interval of a <tt>LiftedRepresentation</tt>, as
/// well as the VGPR it will be loaded to at each instrumentation point.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOLING_COMMON_LR_STATE_VALUE_LOCATIONS_HPP
#define LUTHIER_TOOLING_COMMON_LR_STATE_VALUE_LOCATIONS_HPP
#include "hsa/LoadedCodeObject.hpp"
#include "luthier/LRRegisterLiveness.h"
#include "luthier/LiftedRepresentation.h"
#include "tooling_common/PreKernelEmitter.hpp"
#include <llvm/CodeGen/SlotIndexes.h>
#include <luthier/hsa/LoadedCodeObjectDeviceFunction.h>
#include <luthier/hsa/LoadedCodeObjectKernel.h>

namespace luthier {

/// \brief Contains information on the scheme used for storing with a way to
/// load the state value array into its destination VGPR
struct StateValueArrayStorage
    : public std::enable_shared_from_this<StateValueArrayStorage> {
public:
  enum SchemeKind {
    SVS_SINGLE_VGPR,          /// The state value array is stored in a
                              /// free VGPR
    SVS_ONE_AGPR_post_gfx908, /// The state value array is stored in a free
                              /// AGPR, and the target supports using AGPR as
                              /// an operand in vector instructions
    SVS_TWO_AGPRs_pre_gfx908, /// The state value array is stored in an
                              /// AGPR, with a free AGPR to use as a temp
                              /// spill slot for the app's VGPR. Only
                              /// applicable for pre-gfx908, since they don't
                              /// support using AGPRs as operands for vector
                              /// instructions
    SVS_SINGLE_AGPR_WITH_THREE_SGPRS_pre_gfx908, /// The state value array
                                                 /// is stored in in an AGPR,
                                                 /// with two SGPRs holding
                                                 /// the FLAT SCRATCH base
                                                 /// address of the thread,
                                                 /// and one SGPR
                                                 /// holding the pointer to
                                                 /// the VGPR emergency spill
                                                 /// slot at the beginning of
                                                 /// the instrumentation
                                                 /// private segment. SGPRs
                                                 /// are used to spill an app
                                                 /// VGPR for the state value
                                                 /// array to be loaded into.
                                                 /// For targets that don't
                                                 /// support using an AGPR
                                                 /// directly as a vector
                                                 /// operand
    SVS_SPILLED_WITH_THREE_SGPRS, /// The state value array is spilled into
                                  /// the emergency spill slot in the
                                  /// instrumentation private segment. Two
                                  /// SGPRs hold the thread's flat scratch
                                  /// base and a single SGPR points to the
                                  /// beginning of the instrumentation
                                  /// private segment AKA the emergency app
                                  /// VGPR spill slot
  };

private:
  /// Kind of scheme used to store and load the state value array
  const SchemeKind Kind;

public:
  /// \returns the scheme kind used for storing and loading the state value
  /// array
  SchemeKind getScheme() const { return Kind; }

  /// Constructor
  /// \param Scheme kind of storage and load scheme used for the state value
  /// array
  explicit StateValueArrayStorage(SchemeKind Scheme) : Kind(Scheme) {};

  /// \return if the state value array is stored in a V/AGPR, returns the
  /// the \c llvm::MCRegister associated with it; Otherwise, returns zero
  virtual llvm::MCRegister getStateValueStorageReg() const = 0;
};

/// \brief describes the state value array when stored in a VGPR
struct VGPRStateValueArrayStorage : public StateValueArrayStorage {
public:
  llvm::MCRegister StorageVGPR{};

  /// method for providing LLVM RTTI
  [[nodiscard]] static bool classof(const StateValueArrayStorage *S) {
    return S->getScheme() == SVS_SINGLE_VGPR;
  }

  /// Constructor
  /// \param StorageVGPR the VGPR where the state value array is stored
  explicit VGPRStateValueArrayStorage(llvm::MCRegister StorageVGPR)
      : StorageVGPR(StorageVGPR), StateValueArrayStorage(SVS_SINGLE_VGPR) {};

  llvm::MCRegister getStateValueStorageReg() const override {
    return StorageVGPR;
  }
};

/// \brief describes the state value array when stored in a Single AGPR for
/// targets that support using AGPRs as a vector instruction operand
struct SingleAGPRStateValueArrayStorage : public StateValueArrayStorage {
public:
  llvm::MCRegister StorageAGPR{};

  /// method for providing LLVM RTTI
  [[nodiscard]] static bool classof(const StateValueArrayStorage *S) {
    return S->getScheme() == SVS_SINGLE_VGPR;
  }

  /// Constructor
  /// \param StorageAGPR the AGPR where the state value array is stored
  explicit SingleAGPRStateValueArrayStorage(llvm::MCRegister StorageAGPR)
      : StorageAGPR(StorageAGPR),
        StateValueArrayStorage(SVS_ONE_AGPR_post_gfx908) {};

  llvm::MCRegister getStateValueStorageReg() const override {
    return StorageAGPR;
  }
};

/// \brief describes the state value array when stored in a single AGPR,
/// with a free AGPR for spilling an app VGPR. Only applicable to targets
/// that don't support AGPRs as operands to vector instructions
struct TwoAGPRValueStorage : public StateValueArrayStorage {
public:
  /// Where the state value is stored
  llvm::MCRegister StorageAGPR{};
  /// A free AGPR used for spilling an application VGPR
  llvm::MCRegister TempAGPR{};

  /// method for providing LLVM RTTI
  [[nodiscard]] static bool classof(const StateValueArrayStorage *S) {
    return S->getScheme() == SVS_TWO_AGPRs_pre_gfx908;
  }

  /// Constructor
  TwoAGPRValueStorage(llvm::MCRegister StorageAGPR, llvm::MCRegister TempAGPR)
      : StorageAGPR(StorageAGPR), TempAGPR(TempAGPR),
        StateValueArrayStorage(SVS_TWO_AGPRs_pre_gfx908) {};

  llvm::MCRegister getStateValueStorageReg() const override {
    return StorageAGPR;
  }
};

/// \brief Describes the state value storage scheme where a single AGPR is used
/// to store the state value array, with two SGPRs holding the base address of
/// the wave's flat scratch address, and another SGPR pointing to the
/// instrumentation private segment's emergency VGPR spill slot.
/// Only applicable to targets that don't support using AGPRs as an operand
/// to vector instructions
struct AGPRWithThreeSGPRSValueStorage : public StateValueArrayStorage {
public:
  /// Where the state value is stored
  llvm::MCRegister StorageAGPR{};
  /// Upper 32-bit address of the thread's flat scratch address
  llvm::MCRegister FlatScratchSGPRHigh{};
  /// Lower 32-bit address of the thread's flat scratch address
  llvm::MCRegister FlatScratchSGPRLow{};
  /// An SGPR holding the offset of the instrumentation private segment's
  /// emergency VGPR spill slot from the thread's flat scratch address
  llvm::MCRegister EmergencyVGPRSpillSlotOffset{};

  /// method for providing LLVM RTTI
  [[nodiscard]] static bool classof(const StateValueArrayStorage *S) {
    return S->getScheme() == SVS_SINGLE_AGPR_WITH_THREE_SGPRS_pre_gfx908;
  }

  AGPRWithThreeSGPRSValueStorage(llvm::MCRegister StorageAGPR,
                                 llvm::MCRegister FlatScratchSGPRHigh,
                                 llvm::MCRegister FlatScratchSGPRLow,
                                 llvm::MCRegister EmergencyVGPRSpillSlotOffset)
      : StorageAGPR(StorageAGPR), FlatScratchSGPRHigh(FlatScratchSGPRHigh),
        FlatScratchSGPRLow(FlatScratchSGPRLow),
        EmergencyVGPRSpillSlotOffset(EmergencyVGPRSpillSlotOffset),
        StateValueArrayStorage(SVS_SINGLE_AGPR_WITH_THREE_SGPRS_pre_gfx908) {};

  llvm::MCRegister getStateValueStorageReg() const override {
    return StorageAGPR;
  }
};

/// \brief State value array storage scheme where the SVA is spilled in
/// the thread's emergency SVA spill slot in the instrumentation's private
/// segment, and three SGPRs used to spill an app's VGPR to the
/// instrumentation's private segment before loading the state value array
/// in its place
struct SpilledWithThreeSGPRsValueStorage : public StateValueArrayStorage {
public:
  /// Upper 32-bit address of the thread's flat scratch address
  llvm::MCRegister FlatScratchSGPRHigh{};
  /// Lower 32-bit address of the thread's flat scratch address
  llvm::MCRegister FlatScratchSGPRLow{};
  /// An SGPR holding the offset of the instrumentation private segment's
  /// emergency VGPR spill slot from the thread's flat scratch address
  llvm::MCRegister EmergencyVGPRSpillSlotOffset{};

  /// method for providing LLVM RTTI
  [[nodiscard]] static bool classof(const StateValueArrayStorage *S) {
    return S->getScheme() == SVS_SPILLED_WITH_THREE_SGPRS;
  }

  SpilledWithThreeSGPRsValueStorage(
      llvm::MCRegister FlatScratchSGPRHigh, llvm::MCRegister FlatScratchSGPRLow,
      llvm::MCRegister EmergencyVGPRSpillSlotOffset)
      : FlatScratchSGPRHigh(FlatScratchSGPRHigh),
        FlatScratchSGPRLow(FlatScratchSGPRLow),
        EmergencyVGPRSpillSlotOffset(EmergencyVGPRSpillSlotOffset),
        StateValueArrayStorage(SVS_SPILLED_WITH_THREE_SGPRS) {};

  llvm::MCRegister getStateValueStorageReg() const override { return {}; }
};

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
  llvm::MCRegister StateValueVGPR{};
  /// Where the state value is located before being loaded into the VGPR
  StateValueStorageSegment StateValueLocation;
};

/// \brief an analysis on a \c LiftedRepresentation that determines where the
/// state value array is stored at each instruction of the
/// \c LiftedRepresentation as well as where the state value will be loaded
/// at each instrumentation point.
class LRStateValueStorageAndLoadLocations {

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

  /// Reference to the pre-kernel emission descriptor, in case we need to
  /// signal a need for setting up the state value array before the
  /// kerenl starts
  PreKernelEmissionDescriptor &PKInfo;

  /// A set of physical registers accessed by all injected payloads that
  /// weren't part of their insertion point's live-ins
  /// These are kept track of to ensure they don't get clobbered in
  /// instrumentation routines in order to give tools an
  /// "as correct as possible" view of the kernel
  /// In practice, this set should only become populated in very rare
  /// scenarios, as tools are more interested in registers that are live
  /// at each instruction \n
  /// The analysis will make sure not to use any of these registers when
  /// selecting storage and load locations for the state value array
  const llvm::LivePhysRegs &AccessedPhysicalRegistersNotInLiveIns;

  /// Slot index tracking for each machine instruction of each function in \c LR
  /// This allows us to create intervals to keep track of the location
  /// of the state value array
  llvm::SmallDenseMap<llvm::Function *, std::unique_ptr<llvm::SlotIndexes>>
      FunctionsSlotIndexes{};

  /// Keeps track of how and where the state value array is stored in each
  /// function inside in the \c LR
  llvm::SmallDenseMap<llvm::MachineFunction *,
                      llvm::SmallVector<StateValueStorageSegment>>
      StateValueStorageIntervals{};

  /// Mapping between the MIs of the target app getting instrumented and their
  /// injected payload functions
  const llvm::DenseMap<llvm::MachineInstr *, llvm::Function *>
      &InstPointToInjectedPayloadMap;

  /// Contains a mapping between the instrumentation point MIs of the
  /// \c and the plan to load the state value array from its storage to
  /// the target VGPR to be used by the injected payload function
  llvm::DenseMap<const llvm::MachineInstr *, InstPointSVALoadPlan>
      InstPointSVSLoadPlans{};

  /// \brief Tries to find a fixed location for storing the state value array
  /// \details The order of searching for the storage location is as follows:
  /// 1. Find an unused VGPR. This is the ideal scenario, as no further action
  /// is required in the prologue/epilogue of an injected payload to load/store
  /// the state value array\n
  /// 2. If no unused VGPRs are found, then this routine will find the next
  /// unused AGPR. This usually comes at no cost to the occupancy, as the app
  /// will get the same amount of AGPRs as it gets VGPRs. In gfx90A-, since
  /// AGPRs cannot be used directly by vector instructions and have to be moved
  /// to a VGPR, a single application VGPR must be spilled. Preference is
  /// given to finding another free AGPR to act as a spill slot. If no other
  /// free AGPR is found, then three free SGPRs must be found to spill the
  /// app's VGPR into an emergency spill slot in the instrumentation stack.\n
  /// 3. If no unused V/AGPRs are found in the kernel or a free AGPR is found
  /// but allocation of the spill registers is unsuccessful on gfx90A-,
  /// then as a last resort, this function tries to find three free SGPRs
  /// that can be used to spill an app's VGPR onto the stack, and load the
  /// state value array from the stack
  /// TODO: This function must take an argument indicating whether the tool
  /// writer wants to respect the original kernel's granulated register usage
  /// or not.
  [[nodiscard]] std::shared_ptr<StateValueArrayStorage>
  findFixedStateValueArrayStorage(
      llvm::ArrayRef<llvm::MachineFunction *> RelatedFunctions) const;

  /// Constructor
  /// \param LR the \c LiftedRepresentation being analyzed
  /// \param LCO the \c hsa::LoadedCodeObject in \p LR being analyzed
  /// \param InstPointToInjectedPayloadMap mapping between instrumentation
  /// points inside the \p LR and their injected payload \c llvm::Function
  /// \param AccessedPhysicalRegistersNotInLiveIns set of physical registers
  /// accessed in injected payloads that aren't live at the time of access
  /// \param RegLiveness register liveness analysis for the \p LR
  /// \param PKInfo reference to the pre-kernel emission descriptor
  LRStateValueStorageAndLoadLocations(
      const luthier::LiftedRepresentation &LR, hsa::LoadedCodeObject LCO,
      const llvm::DenseMap<llvm::MachineInstr *, llvm::Function *>
          &InstPointToInjectedPayloadMap,
      const llvm::LivePhysRegs &AccessedPhysicalRegistersNotInLiveIns,
      const LRRegisterLiveness &RegLiveness,
      PreKernelEmissionDescriptor &PKInfo);

  /// calculates the storage and load locations of the state value array
  /// \return an \c llvm::Error indication the success of failure of the
  /// operation
  llvm::Error calculateStateValueArrayStorageAndLoadLocations();

public:
  /// Factory method for creating and running the state value array storage
  /// and load location analysis
  /// \param LR the \c LiftedRepresentation being analyzed
  /// \param LCO the \c hsa::LoadedCodeObject in \p LR being analyzed
  /// \param InstPointToInjectedPayloadMap mapping between instrumentation
  /// points inside the \p LR and their injected payload \c llvm::Function
  /// \param AccessedPhysicalRegistersNotInLiveIns set of physical registers
  /// accessed in injected payloads that aren't live at the time of access
  /// \param RegLiveness register liveness analysis for the \p LR
  /// \param PKInfo reference to the pre-kernel emission descriptor
  /// \return an \c llvm::Error indicating whether or not the analysis was
  /// successful at allocating storage and load locations for the state value
  /// array
  static llvm::Expected<std::unique_ptr<LRStateValueStorageAndLoadLocations>>
  create(const LiftedRepresentation &LR, const hsa::LoadedCodeObject &LCO,
         const llvm::DenseMap<llvm::MachineInstr *, llvm::Function *>
             &InstPointToInjectedPayloadMap,
         const llvm::LivePhysRegs &AccessedPhysicalRegistersNotInLiveIns,
         const LRRegisterLiveness &RegLiveness,
         PreKernelEmissionDescriptor &PKInfo);

  /// Given the \p MBB of the \c LiftedRepresentation being worked on by this
  /// analysis, returns the state value array storage of every instruction
  /// interval inside the \p MBB
  /// \param MBB a basic block that belongs to the \c LiftedRepresentation
  /// of this analysis
  /// \return the state value array storage of every instruction
  /// interval inside the \p MBB or an empty \c llvm::ArrayRef if the \p MBB
  /// is not part of the \c LiftedRepresentation being analyzed
  [[nodiscard]] llvm::ArrayRef<StateValueStorageSegment>
  getStorageIntervalsOfBasicBlock(const llvm::MachineBasicBlock &MBB) const;

  /// \return state value array load plan associated with instrumentation
  /// point \p MI or \c nullptr if the passed \p MI is not an instrumentation
  /// point
  [[nodiscard]] const InstPointSVALoadPlan *
  getStateValueArrayLoadPlanForInstPoint(const llvm::MachineInstr &MI) const;

  /// \return the loaded code object of the \c LiftedRepresentation
  [[nodiscard]] hsa::LoadedCodeObject getLCO() const { return LCO; }
};

} // namespace luthier

#endif