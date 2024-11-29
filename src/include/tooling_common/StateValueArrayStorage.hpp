//===-- StateValueArrayStorage.hpp ----------------------------------------===//
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
/// This file describes different storages for the state value array.
//===----------------------------------------------------------------------===//

#ifndef LUTHIER_TOOLING_COMMON_STATE_VALUE_STATE_VALUE_ARRAY_STORAGE_HPP
#define LUTHIER_TOOLING_COMMON_STATE_VALUE_STATE_VALUE_ARRAY_STORAGE_HPP
#include "hsa/LoadedCodeObject.hpp"
#include "luthier/LRRegisterLiveness.h"
#include "luthier/LiftedRepresentation.h"
#include "luthier/hsa/LoadedCodeObjectDeviceFunction.h"
#include "luthier/hsa/LoadedCodeObjectKernel.h"
#include "tooling_common/PrePostAmbleEmitter.hpp"
#include <llvm/CodeGen/SlotIndexes.h>

namespace llvm {
class GCNSubtarget;
}

namespace luthier {

/// \brief Contains information on the scheme used for storing with a way to
/// load the state value array into its destination VGPR
struct StateValueArrayStorage
    : public std::enable_shared_from_this<StateValueArrayStorage> {
public:
  enum StorageKind {
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
    SVS_SPILLED_WITH_THREE_SGPRS_absolute_fs,    /// The state value array is
                                                 /// spilled into the emergency
                                                 /// spill slot in the
                                                 /// instrumentation private
                                                 /// segment. Two SGPRs hold the
    /// thread's flat scratch base and
    /// a single SGPR points to the
    /// beginning of the
    /// instrumentation private
    /// segment AKA the emergency app
    /// VGPR spill slot
    SVS_SPILLED_WITH_ONE_SGPR_architected_fs /// Same as \c
                                             /// SVS_SPILLED_WITH_THREE_SGPRS
                                             /// except for targets with
                                             /// architected FS which does not
                                             /// require a copy of wave's FS to
                                             /// be stored
  };

private:
  /// Kind of scheme used to store and load the state value array
  const StorageKind Kind;

public:
  /// \returns the scheme kind used for storing and loading the state value
  /// array
  StorageKind getScheme() const { return Kind; }

  /// Constructor
  /// \param Scheme kind of storage and load scheme used for the state value
  /// array
  explicit StateValueArrayStorage(StorageKind Scheme) : Kind(Scheme) {};

  virtual ~StateValueArrayStorage() = default;

  static llvm::Expected<std::unique_ptr<StateValueArrayStorage>>
  createSVAStorage(llvm::ArrayRef<llvm::MCRegister> VGPRs,
                   llvm::ArrayRef<llvm::MCRegister> AGPRs,
                   llvm::ArrayRef<llvm::MCRegister> SGPRs,
                   StateValueArrayStorage::StorageKind Scheme);

  /// \return if the state value array is stored in a V/AGPR, returns the
  /// the \c llvm::MCRegister associated with it; Otherwise, returns zero
  virtual llvm::MCRegister getStateValueStorageReg() const = 0;

  /// \return \c true if the storage requires to be loaded into a V/AGPR before
  /// being used
  virtual bool requiresLoadAndStoreBeforeUse() = 0;

  /// Emit a set of instructions before \p MI that loads the state value array
  /// from its storage to the \p DestVGPR
  virtual void emitCodeToLoadSVA(llvm::MachineInstr &MI,
                                 llvm::MCRegister DestVGPR) const = 0;

  /// Emit a set of instructions after \p MI that stores the state value array
  /// from \p SrcVGPR to the storage
  virtual void emitCodeToStoreSVA(llvm::MachineInstr &MI,
                                  llvm::MCRegister SrcVGPR) const = 0;

  /// Emit a set of instructions after \p MI that moves \c this
  /// the \p TargetSVS
  virtual void emitCodeToSwitchSVS(llvm::MachineInstr &MI,
                           StateValueArrayStorage &TargetSVS) const = 0;

  virtual void getAllStorageRegisters(
      llvm::SmallVectorImpl<llvm::MCRegister> &Regs) const = 0;

  static int getNumVGPRsUsed(StorageKind Kind);

  /// \return the number of VGPRs used by this storage
  int getNumVGPRsUsed() const { return getNumVGPRsUsed(Kind); };

  static int getNumAGPRsUsed(StorageKind Kind);

  /// \return the number of AGPRs used by this storage
  int getNumAGPRsUsed() const { return getNumAGPRsUsed(Kind); };

  static int getNumSGPRsUsed(StorageKind Kind);

  /// \return the number of SGPRs used by this storage
  int getNumSGPRsUsed() const { return getNumSGPRsUsed(Kind); };

  static bool isSupportedOnSubTarget(StorageKind Kind,
                                     const llvm::GCNSubtarget &ST);

  /// \return \c true if \p ST supports using this storage
  bool isSupportedOnSubTarget(const llvm::GCNSubtarget &ST) const {
    return isSupportedOnSubTarget(Kind, ST);
  }
};

/// \brief describes the state value array when stored in a free VGPR
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

  bool requiresLoadAndStoreBeforeUse() override { return false; }

  void emitCodeToLoadSVA(llvm::MachineInstr &MI,
                         llvm::MCRegister DestVGPR) const override {};

  void emitCodeToStoreSVA(llvm::MachineInstr &MI,
                          llvm::MCRegister SrcVGPR) const override {};

  void emitCodeToSwitchSVS(llvm::MachineInstr &MI,
                           StateValueArrayStorage &TargetSVS) const override;

  void getAllStorageRegisters(
      llvm::SmallVectorImpl<llvm::MCRegister> &Regs) const override {
    Regs.push_back(StorageVGPR);
  }
};

/// \brief describes the state value array when stored in a Single free AGPR for
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

  bool requiresLoadAndStoreBeforeUse() override { return false; }

  void emitCodeToLoadSVA(llvm::MachineInstr &MI,
                         llvm::MCRegister DestVGPR) const override {};

  void emitCodeToStoreSVA(llvm::MachineInstr &MI,
                          llvm::MCRegister SrcVGPR) const override {};

  void emitCodeToSwitchSVS(llvm::MachineInstr &MI,
                           StateValueArrayStorage &TargetSVS) const override;

  void getAllStorageRegisters(
      llvm::SmallVectorImpl<llvm::MCRegister> &Regs) const override {
    Regs.push_back(StorageAGPR);
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

  bool requiresLoadAndStoreBeforeUse() override { return true; }

  void emitCodeToLoadSVA(llvm::MachineInstr &MI,
                         llvm::MCRegister DestVGPR) const override;

  void emitCodeToStoreSVA(llvm::MachineInstr &MI,
                          llvm::MCRegister SrcVGPR) const override;

  void emitCodeToSwitchSVS(llvm::MachineInstr &MI,
                           StateValueArrayStorage &TargetSVS) const override;

  void getAllStorageRegisters(
      llvm::SmallVectorImpl<llvm::MCRegister> &Regs) const override {
    Regs.push_back(StorageAGPR);
    Regs.push_back(TempAGPR);
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

  bool requiresLoadAndStoreBeforeUse() override { return true; }

  void emitCodeToLoadSVA(llvm::MachineInstr &MI,
                         llvm::MCRegister DestVGPR) const override;

  void emitCodeToStoreSVA(llvm::MachineInstr &MI,
                          llvm::MCRegister SrcVGPR) const override;

  void emitCodeToSwitchSVS(llvm::MachineInstr &MI,
                           StateValueArrayStorage &TargetSVS) const override;

  void getAllStorageRegisters(
      llvm::SmallVectorImpl<llvm::MCRegister> &Regs) const override {
    Regs.push_back(StorageAGPR);
    Regs.push_back(FlatScratchSGPRHigh);
    Regs.push_back(FlatScratchSGPRLow);
    Regs.push_back(EmergencyVGPRSpillSlotOffset);
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
    return S->getScheme() == SVS_SPILLED_WITH_THREE_SGPRS_absolute_fs;
  }

  SpilledWithThreeSGPRsValueStorage(
      llvm::MCRegister FlatScratchSGPRHigh, llvm::MCRegister FlatScratchSGPRLow,
      llvm::MCRegister EmergencyVGPRSpillSlotOffset)
      : FlatScratchSGPRHigh(FlatScratchSGPRHigh),
        FlatScratchSGPRLow(FlatScratchSGPRLow),
        EmergencyVGPRSpillSlotOffset(EmergencyVGPRSpillSlotOffset),
        StateValueArrayStorage(SVS_SPILLED_WITH_THREE_SGPRS_absolute_fs) {};

  llvm::MCRegister getStateValueStorageReg() const override { return {}; }

  bool requiresLoadAndStoreBeforeUse() override { return true; }

  void emitCodeToLoadSVA(llvm::MachineInstr &MI,
                         llvm::MCRegister DestVGPR) const override;

  void emitCodeToStoreSVA(llvm::MachineInstr &MI,
                          llvm::MCRegister SrcVGPR) const override;

  void emitCodeToSwitchSVS(llvm::MachineInstr &MI,
                           StateValueArrayStorage &TargetSVS) const override;

  void getAllStorageRegisters(
      llvm::SmallVectorImpl<llvm::MCRegister> &Regs) const override {
    Regs.push_back(FlatScratchSGPRHigh);
    Regs.push_back(FlatScratchSGPRLow);
    Regs.push_back(EmergencyVGPRSpillSlotOffset);
  }
};

/// \brief State value array storage scheme for targets with architected
/// FS, where the SVA is spilled in the thread's emergency SVA spill slot in
/// the instrumentation's private segment, and only one SGPR is used with FS
/// to spill an app's VGPR to the instrumentation's private segment before
/// loading the state value array in its place
struct SpilledWithOneSGPRsValueStorage : public StateValueArrayStorage {
public:
  /// An SGPR holding the offset of the instrumentation private segment's
  /// emergency VGPR spill slot from the thread's flat scratch address
  llvm::MCRegister EmergencyVGPRSpillSlotOffset{};

  /// method for providing LLVM RTTI
  [[nodiscard]] static bool classof(const StateValueArrayStorage *S) {
    return S->getScheme() == SVS_SPILLED_WITH_ONE_SGPR_architected_fs;
  }

  explicit SpilledWithOneSGPRsValueStorage(
      llvm::MCRegister EmergencyVGPRSpillSlotOffset)
      : EmergencyVGPRSpillSlotOffset(EmergencyVGPRSpillSlotOffset),
        StateValueArrayStorage(SVS_SPILLED_WITH_ONE_SGPR_architected_fs) {};

  llvm::MCRegister getStateValueStorageReg() const override { return {}; }

  bool requiresLoadAndStoreBeforeUse() override { return true; }

  void emitCodeToLoadSVA(llvm::MachineInstr &MI,
                         llvm::MCRegister DestVGPR) const override;

  void emitCodeToStoreSVA(llvm::MachineInstr &MI,
                          llvm::MCRegister SrcVGPR) const override;

  void emitCodeToSwitchSVS(llvm::MachineInstr &MI,
                           StateValueArrayStorage &TargetSVS) const override;

  void getAllStorageRegisters(
      llvm::SmallVectorImpl<llvm::MCRegister> &Regs) const override {
    Regs.push_back(EmergencyVGPRSpillSlotOffset);
  }
};

/// \returns the set of storage <tt>SchemeKind</tt>s
/// that are supported on the <tt>ST</tt>. The ordering of schemes indicates
/// their preference, with lower-indexed storage kinds being preferred more
void getSupportedSVAStorageList(
    const llvm::GCNSubtarget &ST,
    llvm::SmallVectorImpl<StateValueArrayStorage::StorageKind>
        &SupportedStorageKinds);

} // namespace luthier

#endif