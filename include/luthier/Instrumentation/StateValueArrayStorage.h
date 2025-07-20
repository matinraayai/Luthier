//===-- StateValueArrayStorage.h --------------------------------*- C++ -*-===//
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
/// Describes different storage schemes for the state value array.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_INSTRUMENTATION_AMDGPU_STATE_VALUE_ARRAY_STORAGE_H
#define LUTHIER_INSTRUMENTATION_AMDGPU_STATE_VALUE_ARRAY_STORAGE_H
#include "luthier/Instrumentation/AMDGPURegisterLiveness.h"
#include <llvm/CodeGen/SlotIndexes.h>
#include <llvm/Support/ExtensibleRTTI.h>
#include <memory>

namespace llvm {
class GCNSubtarget;
}

namespace luthier {

/// \brief Struct used to describe the specifications of the state value array
/// storage scheme
struct SVAStorageScheme
    : public llvm::RTTIExtends<SVAStorageScheme, llvm::RTTIRoot> {
protected:
  const unsigned int NumVGPRsUsed;

  const unsigned int NumSGPRsUsed;

  const unsigned int NumAGPRsUsed;

  SVAStorageScheme(unsigned int NumVGPRsUsed, unsigned int NumSGPRsUsed,
                   unsigned int NumAGPRsUsed)
      : NumVGPRsUsed(NumVGPRsUsed), NumSGPRsUsed(NumSGPRsUsed),
        NumAGPRsUsed(NumAGPRsUsed) {}

public:
  /// LLVM RTTI ID
  static char ID;

  ~SVAStorageScheme() override = default;

  [[nodiscard]] virtual std::unique_ptr<SVAStorageScheme> clone() const = 0;

  /// \return the number of VGPRs used by this storage
  [[nodiscard]] unsigned int getNumVGPRsUsed() const { return NumVGPRsUsed; }

  /// \return the number of AGPRs used by this storage
  [[nodiscard]] unsigned int getNumAGPRsUsed() const { return NumAGPRsUsed; }

  /// \return the number of SGPRs used by this storage
  [[nodiscard]] unsigned int getNumSGPRsUsed() const { return NumSGPRsUsed; }

  /// \return \c true if \p ST supports using this storage
  [[nodiscard]] virtual bool
  isSupportedOnSubTarget(const llvm::GCNSubtarget &ST) const = 0;

  /// \return \c true if the storage requires to be loaded into a V/AGPR before
  /// being used
  virtual bool requiresLoadAndStoreBeforeUse() = 0;
};

/// \brief SVA is stored in a free VGPR
struct SingleVGPRSVAStorageScheme final
    : public llvm::RTTIExtends<SingleVGPRSVAStorageScheme, SVAStorageScheme> {
  /// LLVM RTTI ID
  static char ID;

  SingleVGPRSVAStorageScheme()
      : llvm::RTTIExtends<SingleVGPRSVAStorageScheme, SVAStorageScheme>(1, 0,
                                                                        0) {};

  ~SingleVGPRSVAStorageScheme() override = default;

  [[nodiscard]] std::unique_ptr<SVAStorageScheme> clone() const override {
    return std::make_unique<SingleVGPRSVAStorageScheme>();
  }

  [[nodiscard]] bool
  isSupportedOnSubTarget(const llvm::GCNSubtarget &) const override {
    return true;
  }

  bool requiresLoadAndStoreBeforeUse() override { return false; }
};

/// \brief The state value array is stored in a free AGPR, and the target
/// supports using AGPR as an operand in vector instructions
struct SingleAGPRPostGFX908SVAStorageScheme final
    : public llvm::RTTIExtends<SingleAGPRPostGFX908SVAStorageScheme,
                               SVAStorageScheme> {
  /// LLVM RTTI ID
  static char ID;

  SingleAGPRPostGFX908SVAStorageScheme()
      : llvm::RTTIExtends<SingleAGPRPostGFX908SVAStorageScheme,
                          SVAStorageScheme>(0, 0, 1) {};

  ~SingleAGPRPostGFX908SVAStorageScheme() override = default;

  [[nodiscard]] std::unique_ptr<SVAStorageScheme> clone() const override {
    return std::make_unique<SingleAGPRPostGFX908SVAStorageScheme>();
  }

  [[nodiscard]] bool
  isSupportedOnSubTarget(const llvm::GCNSubtarget &ST) const override;

  bool requiresLoadAndStoreBeforeUse() override { return false; }
};

/// \brief The state value array is stored in an AGPR, with a free AGPR to use
/// as a temp spill slot for the app's VGPR. Only applicable for pre-gfx908,
/// since they don't support using AGPRs as operands for vector instructions
struct TwoAGPRsPreGFX908SVAStorageScheme final
    : public llvm::RTTIExtends<TwoAGPRsPreGFX908SVAStorageScheme,
                               SVAStorageScheme> {
  /// LLVM RTTI ID
  static char ID;

  TwoAGPRsPreGFX908SVAStorageScheme()
      : llvm::RTTIExtends<TwoAGPRsPreGFX908SVAStorageScheme, SVAStorageScheme>(
            0, 0, 2) {};

  ~TwoAGPRsPreGFX908SVAStorageScheme() override = default;

  [[nodiscard]] std::unique_ptr<SVAStorageScheme> clone() const override {
    return std::make_unique<TwoAGPRsPreGFX908SVAStorageScheme>();
  }

  [[nodiscard]] bool
  isSupportedOnSubTarget(const llvm::GCNSubtarget &ST) const override;

  bool requiresLoadAndStoreBeforeUse() override { return true; }
};

/// The state value array is stored in an AGPR, with two SGPRs holding
/// the FLAT SCRATCH base address of the thread, and one SGPR holding the
/// pointer to the VGPR emergency spill slot at the beginning of the
/// instrumentation private segment. SGPRs are used to spill an app VGPR for the
/// state value array to be loaded into. For targets that don't support using an
/// AGPR directly as a vector operand
struct SingleAGPRWithThreeSGPRsPreGFX908SVAStorageScheme
    : public llvm::RTTIExtends<
          SingleAGPRWithThreeSGPRsPreGFX908SVAStorageScheme, SVAStorageScheme> {
  /// LLVM RTTI ID
  static char ID;

  SingleAGPRWithThreeSGPRsPreGFX908SVAStorageScheme()
      : llvm::RTTIExtends<SingleAGPRWithThreeSGPRsPreGFX908SVAStorageScheme,
                          SVAStorageScheme>(0, 3, 1) {};

  ~SingleAGPRWithThreeSGPRsPreGFX908SVAStorageScheme() override = default;

  [[nodiscard]] std::unique_ptr<SVAStorageScheme> clone() const override {
    return std::make_unique<
        SingleAGPRWithThreeSGPRsPreGFX908SVAStorageScheme>();
  }

  [[nodiscard]] bool
  isSupportedOnSubTarget(const llvm::GCNSubtarget &ST) const override;

  bool requiresLoadAndStoreBeforeUse() override { return true; }
};

/// The state value array is spilled into the emergency spill slot in the
/// instrumentation private segment. Two SGPRs hold the thread's flat scratch
/// base and a single SGPR points to the beginning of the instrumentation
/// private segment AKA the emergency app VGPR spill slot
struct SpilledWithThreeSGPRsAbsoluteFSSVAStorageScheme
    : public llvm::RTTIExtends<SpilledWithThreeSGPRsAbsoluteFSSVAStorageScheme,
                               SVAStorageScheme> {
public:
  /// LLVM RTTI ID
  static char ID;

  SpilledWithThreeSGPRsAbsoluteFSSVAStorageScheme()
      : llvm::RTTIExtends<SpilledWithThreeSGPRsAbsoluteFSSVAStorageScheme,
                          SVAStorageScheme>(0, 3, 0) {};

  ~SpilledWithThreeSGPRsAbsoluteFSSVAStorageScheme() override = default;

  [[nodiscard]] std::unique_ptr<SVAStorageScheme> clone() const override {
    return std::make_unique<SpilledWithThreeSGPRsAbsoluteFSSVAStorageScheme>();
  }

  [[nodiscard]] bool
  isSupportedOnSubTarget(const llvm::GCNSubtarget &ST) const override;

  bool requiresLoadAndStoreBeforeUse() override { return true; }
};

/// Same as \c SpilledWithThreeSGPRsPreGFX908SVAStorageScheme except for targets
/// with architected FS which does not require a copy of wave's FS to be stored
struct SpilledWithSingleSGPRArchitectedFSSVAStorageScheme
    : public llvm::RTTIExtends<
          SpilledWithSingleSGPRArchitectedFSSVAStorageScheme,
          SVAStorageScheme> {
  /// LLVM RTTI ID
  static char ID;

  SpilledWithSingleSGPRArchitectedFSSVAStorageScheme()
      : llvm::RTTIExtends<SpilledWithSingleSGPRArchitectedFSSVAStorageScheme,
                          SVAStorageScheme>(0, 1, 0) {};

  ~SpilledWithSingleSGPRArchitectedFSSVAStorageScheme() override = default;

  [[nodiscard]] std::unique_ptr<SVAStorageScheme> clone() const override {
    return std::make_unique<
        SpilledWithSingleSGPRArchitectedFSSVAStorageScheme>();
  }

  [[nodiscard]] bool
  isSupportedOnSubTarget(const llvm::GCNSubtarget &ST) const override;

  bool requiresLoadAndStoreBeforeUse() override { return true; }
};

/// \brief Contains information on the scheme used for storing with a way to
/// load the state value array into its destination VGPR
struct StateValueArrayStorage
    : public std::enable_shared_from_this<StateValueArrayStorage>,
      public llvm::RTTIExtends<StateValueArrayStorage, llvm::RTTIRoot> {

  static char ID;

  /// \returns the scheme kind used for storing and loading the state value
  /// array
  [[nodiscard]] virtual std::unique_ptr<SVAStorageScheme> getScheme() const = 0;

  /// Constructor
  StateValueArrayStorage() = default;

  ~StateValueArrayStorage() override = default;

  /// Factory method to create a storage scheme using its scheme descriptor and
  /// the registers it uses
  /// If the number of GPRs passed to this function is more than what the
  /// \p Scheme requires only the first few are used and the additional
  /// registers are ignored
  static llvm::Expected<std::unique_ptr<StateValueArrayStorage>>
  createSVAStorage(llvm::ArrayRef<llvm::MCRegister> VGPRs,
                   llvm::ArrayRef<llvm::MCRegister> AGPRs,
                   llvm::ArrayRef<llvm::MCRegister> SGPRs,
                   const SVAStorageScheme &Scheme);

  /// \return if the state value array is stored in a V/AGPR, returns the
  /// \c llvm::MCRegister associated with it; Otherwise, returns zero
  [[nodiscard]] virtual llvm::MCRegister getStateValueStorageReg() const = 0;

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
  virtual void emitCodeToSwitchSVS(
      llvm::MachineBasicBlock::iterator MI,
      const StateValueArrayStorage &TargetSVS,
      const StateValueArraySpecsAnalysis::Result &Specs) const = 0;

  virtual bool operator==(const StateValueArrayStorage &LHS) const = 0;

  bool operator!=(const StateValueArrayStorage &LHS) const {
    return !operator==(LHS);
  }

  /// Returns all the MCRegisters involved in this storage
  virtual void getAllStorageRegisters(
      llvm::SmallVectorImpl<llvm::MCRegister> &Regs) const = 0;
};

/// \brief describes the state value array when stored in a free VGPR
struct VGPRStateValueArrayStorage final
    : public llvm::RTTIExtends<VGPRStateValueArrayStorage,
                               StateValueArrayStorage> {
public:
  llvm::MCRegister StorageVGPR{};

  static char ID;

  /// Constructor
  /// \param StorageVGPR the VGPR where the state value array is stored
  explicit VGPRStateValueArrayStorage(llvm::MCRegister StorageVGPR)
      : llvm::RTTIExtends<VGPRStateValueArrayStorage, StateValueArrayStorage>(),
        StorageVGPR(StorageVGPR) {};

  [[nodiscard]] std::unique_ptr<SVAStorageScheme> getScheme() const override {
    return std::make_unique<SingleVGPRSVAStorageScheme>();
  }

  [[nodiscard]] llvm::MCRegister getStateValueStorageReg() const override {
    return StorageVGPR;
  }

  bool operator==(const StateValueArrayStorage &LHS) const override;

  void emitCodeToLoadSVA(llvm::MachineInstr &MI,
                         llvm::MCRegister DestVGPR) const override {};

  void emitCodeToStoreSVA(llvm::MachineInstr &MI,
                          llvm::MCRegister SrcVGPR) const override {};

  void emitCodeToSwitchSVS(
      llvm::MachineBasicBlock::iterator MI,
      const StateValueArrayStorage &TargetSVS,
      const StateValueArraySpecsAnalysis::Result &Specs) const override;

  void getAllStorageRegisters(
      llvm::SmallVectorImpl<llvm::MCRegister> &Regs) const override {
    Regs.push_back(StorageVGPR);
  }
};

/// \brief describes the state value array when stored in a Single free AGPR for
/// targets that support using AGPRs as a vector instruction operand
struct SingleAGPRStateValueArrayStorage final
    : public llvm::RTTIExtends<SingleAGPRStateValueArrayStorage,
                               StateValueArrayStorage> {
public:
  llvm::MCRegister StorageAGPR{};

  /// Constructor
  /// \param StorageAGPR the AGPR where the state value array is stored
  explicit SingleAGPRStateValueArrayStorage(llvm::MCRegister StorageAGPR)
      : llvm::RTTIExtends<SingleAGPRStateValueArrayStorage,
                          StateValueArrayStorage>(),
        StorageAGPR(StorageAGPR) {};

  [[nodiscard]] std::unique_ptr<SVAStorageScheme> getScheme() const override {
    return std::make_unique<SingleAGPRPostGFX908SVAStorageScheme>();
  }

  [[nodiscard]] llvm::MCRegister getStateValueStorageReg() const override {
    return StorageAGPR;
  }

  bool operator==(const StateValueArrayStorage &LHS) const override;

  void emitCodeToLoadSVA(llvm::MachineInstr &MI,
                         llvm::MCRegister DestVGPR) const override {};

  void emitCodeToStoreSVA(llvm::MachineInstr &MI,
                          llvm::MCRegister SrcVGPR) const override {};

  void emitCodeToSwitchSVS(
      llvm::MachineBasicBlock::iterator MI,
      const StateValueArrayStorage &TargetSVS,
      const StateValueArraySpecsAnalysis::Result &Specs) const override;

  void getAllStorageRegisters(
      llvm::SmallVectorImpl<llvm::MCRegister> &Regs) const override {
    Regs.push_back(StorageAGPR);
  }
};

/// \brief describes the state value array when stored in a single AGPR,
/// with a free AGPR for spilling an app VGPR. Only applicable to targets
/// that don't support AGPRs as operands to vector instructions
struct TwoAGPRValueStorage final
    : public llvm::RTTIExtends<SingleAGPRStateValueArrayStorage,
                               StateValueArrayStorage> {
public:
  /// Where the state value is stored
  llvm::MCRegister StorageAGPR{};
  /// A free AGPR used for spilling an application VGPR
  llvm::MCRegister TempAGPR{};

  /// Constructor
  TwoAGPRValueStorage(llvm::MCRegister StorageAGPR, llvm::MCRegister TempAGPR)
      : llvm::RTTIExtends<SingleAGPRStateValueArrayStorage,
                          StateValueArrayStorage>(),
        StorageAGPR(StorageAGPR), TempAGPR(TempAGPR) {};

  [[nodiscard]] std::unique_ptr<SVAStorageScheme> getScheme() const override {
    return std::make_unique<TwoAGPRsPreGFX908SVAStorageScheme>();
  }

  [[nodiscard]] llvm::MCRegister getStateValueStorageReg() const override {
    return StorageAGPR;
  }

  bool operator==(const StateValueArrayStorage &LHS) const override;

  void emitCodeToLoadSVA(llvm::MachineInstr &MI,
                         llvm::MCRegister DestVGPR) const override;

  void emitCodeToStoreSVA(llvm::MachineInstr &MI,
                          llvm::MCRegister SrcVGPR) const override;

  void emitCodeToSwitchSVS(
      llvm::MachineBasicBlock::iterator MI,
      const StateValueArrayStorage &TargetSVS,
      const StateValueArraySpecsAnalysis::Result &Specs) const override;

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
struct AGPRWithThreeSGPRSValueStorage final
    : public llvm::RTTIExtends<AGPRWithThreeSGPRSValueStorage,
                               StateValueArrayStorage> {
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

  AGPRWithThreeSGPRSValueStorage(llvm::MCRegister StorageAGPR,
                                 llvm::MCRegister FlatScratchSGPRHigh,
                                 llvm::MCRegister FlatScratchSGPRLow,
                                 llvm::MCRegister EmergencyVGPRSpillSlotOffset)
      : llvm::RTTIExtends<AGPRWithThreeSGPRSValueStorage,
                          StateValueArrayStorage>(),
        StorageAGPR(StorageAGPR), FlatScratchSGPRHigh(FlatScratchSGPRHigh),
        FlatScratchSGPRLow(FlatScratchSGPRLow),
        EmergencyVGPRSpillSlotOffset(EmergencyVGPRSpillSlotOffset) {};

  [[nodiscard]] std::unique_ptr<SVAStorageScheme> getScheme() const override {
    return std::make_unique<
        SingleAGPRWithThreeSGPRsPreGFX908SVAStorageScheme>();
  }

  [[nodiscard]] llvm::MCRegister getStateValueStorageReg() const override {
    return StorageAGPR;
  }

  bool operator==(const StateValueArrayStorage &LHS) const override;

  void emitCodeToLoadSVA(llvm::MachineInstr &MI,
                         llvm::MCRegister DestVGPR) const override;

  void emitCodeToStoreSVA(llvm::MachineInstr &MI,
                          llvm::MCRegister SrcVGPR) const override;

  void emitCodeToSwitchSVS(
      llvm::MachineBasicBlock::iterator MI,
      const StateValueArrayStorage &TargetSVS,
      const StateValueArraySpecsAnalysis::Result &Specs) const override;

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
struct SpilledWithThreeSGPRsValueStorage final
    : public llvm::RTTIExtends<SpilledWithThreeSGPRsValueStorage,
                               StateValueArrayStorage> {
public:
  /// Upper 32-bit address of the thread's flat scratch address
  llvm::MCRegister FlatScratchSGPRHigh{};
  /// Lower 32-bit address of the thread's flat scratch address
  llvm::MCRegister FlatScratchSGPRLow{};
  /// An SGPR holding the offset of the instrumentation private segment's
  /// emergency VGPR spill slot from the thread's flat scratch address
  llvm::MCRegister EmergencyVGPRSpillSlotOffset{};

  SpilledWithThreeSGPRsValueStorage(
      llvm::MCRegister FlatScratchSGPRHigh, llvm::MCRegister FlatScratchSGPRLow,
      llvm::MCRegister EmergencyVGPRSpillSlotOffset)
      : llvm::RTTIExtends<SpilledWithThreeSGPRsValueStorage,
                          StateValueArrayStorage>(),
        FlatScratchSGPRHigh(FlatScratchSGPRHigh),
        FlatScratchSGPRLow(FlatScratchSGPRLow),
        EmergencyVGPRSpillSlotOffset(EmergencyVGPRSpillSlotOffset) {};

  [[nodiscard]] std::unique_ptr<SVAStorageScheme> getScheme() const override {
    return std::make_unique<SpilledWithThreeSGPRsAbsoluteFSSVAStorageScheme>();
  }

  [[nodiscard]] llvm::MCRegister getStateValueStorageReg() const override {
    return {};
  }

  bool operator==(const StateValueArrayStorage &LHS) const override;

  void emitCodeToLoadSVA(llvm::MachineInstr &MI,
                         llvm::MCRegister DestVGPR) const override;

  void emitCodeToStoreSVA(llvm::MachineInstr &MI,
                          llvm::MCRegister SrcVGPR) const override;

  void emitCodeToSwitchSVS(
      llvm::MachineBasicBlock::iterator MI,
      const StateValueArrayStorage &TargetSVS,
      const StateValueArraySpecsAnalysis::Result &Specs) const override;

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
struct SpilledWithOneSGPRValueStorage
    : public llvm::RTTIExtends<SpilledWithOneSGPRValueStorage,
                               StateValueArrayStorage> {
public:
  /// An SGPR holding the offset of the instrumentation private segment's
  /// emergency VGPR spill slot from the thread's flat scratch address
  llvm::MCRegister EmergencyVGPRSpillSlotOffset{};

  explicit SpilledWithOneSGPRValueStorage(
      llvm::MCRegister EmergencyVGPRSpillSlotOffset)
      : llvm::RTTIExtends<SpilledWithOneSGPRValueStorage,
                          StateValueArrayStorage>(),
        EmergencyVGPRSpillSlotOffset(EmergencyVGPRSpillSlotOffset) {};

  [[nodiscard]] std::unique_ptr<SVAStorageScheme> getScheme() const override {
    return std::make_unique<
        SpilledWithSingleSGPRArchitectedFSSVAStorageScheme>();
  }

  [[nodiscard]] llvm::MCRegister getStateValueStorageReg() const override {
    return {};
  }

  bool operator==(const StateValueArrayStorage &LHS) const override;

  void emitCodeToLoadSVA(llvm::MachineInstr &MI,
                         llvm::MCRegister DestVGPR) const override;

  void emitCodeToStoreSVA(llvm::MachineInstr &MI,
                          llvm::MCRegister SrcVGPR) const override;

  void emitCodeToSwitchSVS(
      llvm::MachineBasicBlock::iterator MI,
      const StateValueArrayStorage &TargetSVS,
      const StateValueArraySpecsAnalysis::Result &Specs) const override;

  void getAllStorageRegisters(
      llvm::SmallVectorImpl<llvm::MCRegister> &Regs) const override {
    Regs.push_back(EmergencyVGPRSpillSlotOffset);
  }
};

/// \returns the st of storage schemes that are supported on the <tt>ST</tt>.
/// The ordering of schemes indicates their preference, with lower-indexed
/// storage kinds being preferred more
void getSupportedSVAStorageList(
    const llvm::GCNSubtarget &ST,
    llvm::SmallVectorImpl<std::unique_ptr<SVAStorageScheme>>
        &SupportedStorageKinds);

} // namespace luthier

#endif