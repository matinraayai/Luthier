//===-- StateValueArrayStorage.h --------------------------------*- C++ -*-===//
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
///
/// \file
/// This file describes different storages for the state value array.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOL_CODE_GEN_STATE_VALUE_ARRAY_STORAGE_H
#define LUTHIER_TOOL_CODE_GEN_STATE_VALUE_ARRAY_STORAGE_H
#include <llvm/CodeGen/SlotIndexes.h>

namespace llvm {
class GCNSubtarget;
}

namespace luthier {

class StateValueArraySpecs;

/// \brief Contains information on the scheme used for storing with a way to
/// load the state value array into its destination VGPR
struct StateValueArrayStorage
    : public std::enable_shared_from_this<StateValueArrayStorage> {
public:
  enum StorageKind {
    /// The state value array is stored in a free VGPR. Costs nothing beyond
    /// the VGPR itself: no load or store is needed before a payload uses it.
    SVS_SINGLE_VGPR,
    /// The state value array is stored in an AGPR, with a second free AGPR to
    /// use as a temp spill slot for the app's VGPR. Only applicable to targets
    /// with AGPRs that cannot use them as operands to vector instructions.
    SVS_TWO_AGPRs,
    /// The state value array is stored in an AGPR, with two SGPRs shadowing
    /// the wavefront's flat scratch base and one scratch SGPR. Used to spill
    /// an app VGPR for the state value array to be loaded into, on targets
    /// that cannot use an AGPR directly as a vector operand.
    ///
    /// GFX9 only, and in practice gfx908 only: gfx90a+ can use AGPRs as vector
    /// operands, and no GFX10 or later part has AGPRs at all, which is why
    /// there is no gfx10plus counterpart to this scheme.
    SVS_SINGLE_AGPR_WITH_THREE_SGPRS_absolute_fs_gfx9,
    /// The state value array is spilled into the emergency spill slot at the
    /// bottom of the wavefront's private segment. Two SGPRs shadow the
    /// wavefront's flat scratch base and a third is a scratch register.
    ///
    /// On GFX9 the flat scratch shadow is exchanged with \c FLAT_SCR by an
    /// XOR swap, which needs no temporary; the third SGPR is instead there to
    /// hold the zero SADDR, because GFX9 supports neither the ST addressing
    /// mode nor the \c null SGPR (see \c emitEmergencySlotAccess ).
    SVS_SPILLED_WITH_THREE_SGPRS_absolute_fs_gfx9,
    /// As \c SVS_SPILLED_WITH_THREE_SGPRS_absolute_fs_gfx9 , but for GFX10+
    /// absolute-flat-scratch targets, where the third SGPR pays for the
    /// opposite half of the problem: the emergency slots need no SADDR (\c
    /// SGPR_NULL or the ST form covers it), but \c FLAT_SCR is reachable only
    /// via \c S_GETREG_B32 / \c S_SETREG_B32 , so exchanging it with the
    /// shadow pair needs a temporary. See \c emitFlatScratchSwap .
    SVS_SPILLED_WITH_THREE_SGPRS_absolute_fs_gfx10plus,
    /// The state value array is spilled into the emergency spill slot at the
    /// bottom of the wavefront's private segment, at no register cost at all.
    /// Targets with architected flat scratch need no shadow of \c FLAT_SCR
    /// (the hardware initialises it and the shader cannot override it), and
    /// all of them support a SADDR-less \c SCRATCH_* , so no scratch SGPR is
    /// needed either. Being free, this scheme is always constructible.
    SVS_SPILLED_WITH_NO_SGPRS_architected_fs
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
  virtual bool requiresLoadAndStoreBeforeUse() const = 0;

  /// Emit a set of instructions before \p MI that loads the state value array
  /// from its storage to the \p DestVGPR
  virtual void emitCodeToLoadSVA(llvm::MachineInstr &MI,
                                 llvm::MCRegister DestVGPR) const = 0;

  /// Emit a set of instructions before \p MI that stores the state value array
  /// from \p SrcVGPR to the storage, and returns \p SrcVGPR 's application
  /// value to it.
  virtual void emitCodeToStoreSVA(llvm::MachineInstr &MI,
                                  llvm::MCRegister SrcVGPR) const = 0;

  /// Cross trace function handoff protocol — caller side. Emitted before
  /// a call/indirect-branch MI: spills \c VGPR0's app-value (all lanes) to
  /// the SVS's emergency slot and loads the SVA from \c this scheme into
  /// \c VGPR0 (all lanes) so the callee sees \c V0 == SVA on entry.
  virtual void handOffSVA(llvm::MachineInstr &MI,
                          const StateValueArraySpecs &Specs,
                          const llvm::GCNSubtarget &ST) const = 0;

  /// Cross trace function handoff protocol — callee side. Emitted at the
  /// first MI of a device-function entry block: \c VGPR0 arrives holding the
  /// SVA (from the caller's \c handOffSVA); this stores it into \c this
  /// scheme's storage (the entry-block SVS) and restores \c VGPR0's app value
  /// (all lanes) from the SVS's emergency slot.
  virtual void pickOffSVA(llvm::MachineInstr &MI,
                          const StateValueArraySpecs &Specs,
                          const llvm::GCNSubtarget &ST) const = 0;

  /// Emit a set of instructions after \p MI that moves \c this
  /// the \p TargetSVS
  virtual void
  emitCodeToSwitchSVS(llvm::MachineBasicBlock::iterator MI,
                      const StateValueArrayStorage &TargetSVS,
                      const StateValueArraySpecs &Specs) const = 0;

  virtual bool operator==(const StateValueArrayStorage &LHS) const = 0;

  bool operator!=(const StateValueArrayStorage &LHS) const {
    return !operator==(LHS);
  }

  virtual void getAllStorageRegisters(
      llvm::SmallVectorImpl<llvm::MCRegister> &Regs) const = 0;

  /// Largest number of 32-bit SGPRs \c emitLongJumpSGPRSpill /
  /// \c emitLongJumpSGPRRestore can carry: an \c SReg_64 pair plus the
  /// \c $scc save slot the patch-point call sequence needs when \c $scc is
  /// live across the site.
  static constexpr unsigned MaxLongJumpSpillSGPRs = 3;

  /// \return the SVA lane the \p Idx -th spilled SGPR occupies.
  ///
  /// The long-jump save borrows the three lanes that are only ever live
  /// inside an injected payload's own prologue/epilogue window --- the
  /// stack-pointer spill lane, the frame-pointer spill lane, and the lo half
  /// of the \c EXEC spill lane (see \c InjectedPayloadPEIPass ). At a
  /// target-module patch point or relaxed branch no payload is executing, so
  /// all three hold nothing live.
  static uint8_t getLongJumpSpillLane(const StateValueArraySpecs &Specs,
                                      unsigned Idx);

  /// Emit code before \p InsertPt that parks \p SGPRs --- between one and
  /// \c MaxLongJumpSpillSGPRs 32-bit SGPRs --- in the state value array,
  /// leaving the registers free for the caller to clobber while it builds a
  /// long jump out of them.
  ///
  /// Every scheme implements this against its own storage: the VGPR scheme
  /// writes the lanes directly, the AGPR schemes shuttle through a courier
  /// VGPR parked in the temp AGPR, and the spilled schemes shuttle through a
  /// courier VGPR parked on the instrumentation stack.
  ///
  /// \param MBB block \p InsertPt belongs to
  /// \param InsertPt where the save is emitted; on every scheme but
  /// \c SVS_SINGLE_VGPR this call splits \p MBB , so iterators into it do
  /// not survive
  /// \param SGPRs the 32-bit SGPRs to save, in lane order
  /// \param Specs the SVA lane layout
  virtual void
  emitLongJumpSGPRSpill(llvm::MachineBasicBlock &MBB,
                        llvm::MachineBasicBlock::iterator InsertPt,
                        llvm::ArrayRef<llvm::MCRegister> SGPRs,
                        const StateValueArraySpecs &Specs) const = 0;

  /// Inverse of \c emitLongJumpSGPRSpill : reload \p SGPRs from their SVA
  /// lanes, releasing the lanes for the next borrower. \p SGPRs must be the
  /// same list, in the same order, that was handed to the matching
  /// \c emitLongJumpSGPRSpill .
  virtual void
  emitLongJumpSGPRRestore(llvm::MachineBasicBlock &MBB,
                          llvm::MachineBasicBlock::iterator InsertPt,
                          llvm::ArrayRef<llvm::MCRegister> SGPRs,
                          const StateValueArraySpecs &Specs) const = 0;

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

protected:
  /// Declare this scheme's storage registers, plus \p Courier when it is
  /// non-zero, live-in on \p MBB . The lane moves emitted by
  /// \c emitLongJumpSGPRSpill read the courier as a tied operand and read
  /// the storage registers as addresses, and neither is in the block's
  /// live-in set as the patcher and the branch relaxer seed it.
  void addLongJumpSpillLiveIns(llvm::MachineBasicBlock &MBB,
                               llvm::MCRegister Courier) const;
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

  bool operator==(const StateValueArrayStorage &LHS) const override;

  bool requiresLoadAndStoreBeforeUse() const override { return false; }

  void emitCodeToLoadSVA(llvm::MachineInstr &MI,
                         llvm::MCRegister DestVGPR) const override {};

  void emitCodeToStoreSVA(llvm::MachineInstr &MI,
                          llvm::MCRegister SrcVGPR) const override {};

  void handOffSVA(llvm::MachineInstr &MI, const StateValueArraySpecs &Specs,
                  const llvm::GCNSubtarget &ST) const override;

  void pickOffSVA(llvm::MachineInstr &MI, const StateValueArraySpecs &Specs,
                  const llvm::GCNSubtarget &ST) const override;

  void
  emitCodeToSwitchSVS(llvm::MachineBasicBlock::iterator MI,
                      const StateValueArrayStorage &TargetSVS,
                      const StateValueArraySpecs &Specs) const override;

  void emitLongJumpSGPRSpill(llvm::MachineBasicBlock &MBB,
                             llvm::MachineBasicBlock::iterator InsertPt,
                             llvm::ArrayRef<llvm::MCRegister> SGPRs,
                             const StateValueArraySpecs &Specs) const override;

  void
  emitLongJumpSGPRRestore(llvm::MachineBasicBlock &MBB,
                          llvm::MachineBasicBlock::iterator InsertPt,
                          llvm::ArrayRef<llvm::MCRegister> SGPRs,
                          const StateValueArraySpecs &Specs) const override;

  void getAllStorageRegisters(
      llvm::SmallVectorImpl<llvm::MCRegister> &Regs) const override {
    Regs.push_back(StorageVGPR);
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
    return S->getScheme() == SVS_TWO_AGPRs;
  }

  /// Constructor
  TwoAGPRValueStorage(llvm::MCRegister StorageAGPR, llvm::MCRegister TempAGPR)
      : StorageAGPR(StorageAGPR), TempAGPR(TempAGPR),
        StateValueArrayStorage(SVS_TWO_AGPRs) {};

  llvm::MCRegister getStateValueStorageReg() const override {
    return StorageAGPR;
  }

  bool operator==(const StateValueArrayStorage &LHS) const override;

  bool requiresLoadAndStoreBeforeUse() const override { return true; }

  void emitCodeToLoadSVA(llvm::MachineInstr &MI,
                         llvm::MCRegister DestVGPR) const override;

  void emitCodeToStoreSVA(llvm::MachineInstr &MI,
                          llvm::MCRegister SrcVGPR) const override;

  void handOffSVA(llvm::MachineInstr &MI, const StateValueArraySpecs &Specs,
                  const llvm::GCNSubtarget &ST) const override;

  void pickOffSVA(llvm::MachineInstr &MI, const StateValueArraySpecs &Specs,
                  const llvm::GCNSubtarget &ST) const override;

  void
  emitCodeToSwitchSVS(llvm::MachineBasicBlock::iterator MI,
                      const StateValueArrayStorage &TargetSVS,
                      const StateValueArraySpecs &Specs) const override;

  void emitLongJumpSGPRSpill(llvm::MachineBasicBlock &MBB,
                             llvm::MachineBasicBlock::iterator InsertPt,
                             llvm::ArrayRef<llvm::MCRegister> SGPRs,
                             const StateValueArraySpecs &Specs) const override;

  void
  emitLongJumpSGPRRestore(llvm::MachineBasicBlock &MBB,
                          llvm::MachineBasicBlock::iterator InsertPt,
                          llvm::ArrayRef<llvm::MCRegister> SGPRs,
                          const StateValueArraySpecs &Specs) const override;

  void getAllStorageRegisters(
      llvm::SmallVectorImpl<llvm::MCRegister> &Regs) const override {
    Regs.push_back(StorageAGPR);
    Regs.push_back(TempAGPR);
  }
};

/// \brief Describes the state value storage scheme where a single AGPR is used
/// to store the state value array, with two SGPRs shadowing the base address
/// of the wave's flat scratch, and a third scratch SGPR. Only applicable to
/// GFX9 targets that don't support using AGPRs as an operand to vector
/// instructions.
struct AGPRWithThreeSGPRSValueStorage : public StateValueArrayStorage {
public:
  /// Where the state value is stored
  llvm::MCRegister StorageAGPR{};
  /// Upper 32-bit address of the thread's flat scratch address
  llvm::MCRegister FlatScratchSGPRHigh{};
  /// Lower 32-bit address of the thread's flat scratch address
  llvm::MCRegister FlatScratchSGPRLow{};
  /// A scratch SGPR, clobbered freely by this scheme's emission. On GFX9 it
  /// carries the zero SADDR every emergency-slot access needs; see
  /// \c SVS_SPILLED_WITH_THREE_SGPRS_absolute_fs_gfx9 .
  llvm::MCRegister ScratchSGPR{};

  /// method for providing LLVM RTTI
  [[nodiscard]] static bool classof(const StateValueArrayStorage *S) {
    return S->getScheme() == SVS_SINGLE_AGPR_WITH_THREE_SGPRS_absolute_fs_gfx9;
  }

  AGPRWithThreeSGPRSValueStorage(llvm::MCRegister StorageAGPR,
                                 llvm::MCRegister FlatScratchSGPRHigh,
                                 llvm::MCRegister FlatScratchSGPRLow,
                                 llvm::MCRegister ScratchSGPR)
      : StateValueArrayStorage(
            SVS_SINGLE_AGPR_WITH_THREE_SGPRS_absolute_fs_gfx9),
        StorageAGPR(StorageAGPR), FlatScratchSGPRHigh(FlatScratchSGPRHigh),
        FlatScratchSGPRLow(FlatScratchSGPRLow), ScratchSGPR(ScratchSGPR) {};

  llvm::MCRegister getStateValueStorageReg() const override {
    return StorageAGPR;
  }

  bool operator==(const StateValueArrayStorage &LHS) const override;

  bool requiresLoadAndStoreBeforeUse() const override { return true; }

  void emitCodeToLoadSVA(llvm::MachineInstr &MI,
                         llvm::MCRegister DestVGPR) const override;

  void emitCodeToStoreSVA(llvm::MachineInstr &MI,
                          llvm::MCRegister SrcVGPR) const override;

  void handOffSVA(llvm::MachineInstr &MI, const StateValueArraySpecs &Specs,
                  const llvm::GCNSubtarget &ST) const override;

  void pickOffSVA(llvm::MachineInstr &MI, const StateValueArraySpecs &Specs,
                  const llvm::GCNSubtarget &ST) const override;

  void
  emitCodeToSwitchSVS(llvm::MachineBasicBlock::iterator MI,
                      const StateValueArrayStorage &TargetSVS,
                      const StateValueArraySpecs &Specs) const override;

  void emitLongJumpSGPRSpill(llvm::MachineBasicBlock &MBB,
                             llvm::MachineBasicBlock::iterator InsertPt,
                             llvm::ArrayRef<llvm::MCRegister> SGPRs,
                             const StateValueArraySpecs &Specs) const override;

  void
  emitLongJumpSGPRRestore(llvm::MachineBasicBlock &MBB,
                          llvm::MachineBasicBlock::iterator InsertPt,
                          llvm::ArrayRef<llvm::MCRegister> SGPRs,
                          const StateValueArraySpecs &Specs) const override;

  void getAllStorageRegisters(
      llvm::SmallVectorImpl<llvm::MCRegister> &Regs) const override {
    Regs.push_back(StorageAGPR);
    Regs.push_back(FlatScratchSGPRHigh);
    Regs.push_back(FlatScratchSGPRLow);
    Regs.push_back(ScratchSGPR);
  }
};

/// \brief State value array storage scheme for absolute-flat-scratch targets,
/// where the SVA is spilled into the emergency SVA slot at the bottom of the
/// wavefront's private segment. Two SGPRs shadow the wavefront's flat scratch
/// base so it can be installed into \c FLAT_SCR around the spill, and a third
/// is a scratch register.
///
/// One class backs both the GFX9 and the GFX10+ scheme kinds: the registers
/// they hold are identical and only the emission differs, which
/// \c emitFlatScratchSwap and \c emitEmergencySlotAccess decide from the
/// subtarget. The two kinds are kept distinct so scheme selection names the
/// right one per target.
struct SpilledWithThreeSGPRsValueStorage : public StateValueArrayStorage {
public:
  /// Upper 32-bit address of the thread's flat scratch address
  llvm::MCRegister FlatScratchSGPRHigh{};
  /// Lower 32-bit address of the thread's flat scratch address
  llvm::MCRegister FlatScratchSGPRLow{};
  /// A scratch SGPR, clobbered freely by this scheme's emission: the zero
  /// SADDR on GFX9, the \c FLAT_SCR exchange temporary on GFX10+.
  llvm::MCRegister ScratchSGPR{};

  /// method for providing LLVM RTTI
  [[nodiscard]] static bool classof(const StateValueArrayStorage *S) {
    return S->getScheme() == SVS_SPILLED_WITH_THREE_SGPRS_absolute_fs_gfx9 ||
           S->getScheme() ==
               SVS_SPILLED_WITH_THREE_SGPRS_absolute_fs_gfx10plus;
  }

  SpilledWithThreeSGPRsValueStorage(StorageKind Kind,
                                    llvm::MCRegister FlatScratchSGPRHigh,
                                    llvm::MCRegister FlatScratchSGPRLow,
                                    llvm::MCRegister ScratchSGPR)
      : StateValueArrayStorage(Kind), FlatScratchSGPRHigh(FlatScratchSGPRHigh),
        FlatScratchSGPRLow(FlatScratchSGPRLow), ScratchSGPR(ScratchSGPR) {
    assert((Kind == SVS_SPILLED_WITH_THREE_SGPRS_absolute_fs_gfx9 ||
            Kind == SVS_SPILLED_WITH_THREE_SGPRS_absolute_fs_gfx10plus) &&
           "SpilledWithThreeSGPRsValueStorage backs only the two "
           "absolute-flat-scratch three-SGPR scheme kinds");
  };

  llvm::MCRegister getStateValueStorageReg() const override { return {}; }

  bool operator==(const StateValueArrayStorage &LHS) const override;

  bool requiresLoadAndStoreBeforeUse() const override { return true; }

  void emitCodeToLoadSVA(llvm::MachineInstr &MI,
                         llvm::MCRegister DestVGPR) const override;

  void emitCodeToStoreSVA(llvm::MachineInstr &MI,
                          llvm::MCRegister SrcVGPR) const override;

  void handOffSVA(llvm::MachineInstr &MI, const StateValueArraySpecs &Specs,
                  const llvm::GCNSubtarget &ST) const override;

  void pickOffSVA(llvm::MachineInstr &MI, const StateValueArraySpecs &Specs,
                  const llvm::GCNSubtarget &ST) const override;

  void
  emitCodeToSwitchSVS(llvm::MachineBasicBlock::iterator MI,
                      const StateValueArrayStorage &TargetSVS,
                      const StateValueArraySpecs &Specs) const override;

  void emitLongJumpSGPRSpill(llvm::MachineBasicBlock &MBB,
                             llvm::MachineBasicBlock::iterator InsertPt,
                             llvm::ArrayRef<llvm::MCRegister> SGPRs,
                             const StateValueArraySpecs &Specs) const override;

  void
  emitLongJumpSGPRRestore(llvm::MachineBasicBlock &MBB,
                          llvm::MachineBasicBlock::iterator InsertPt,
                          llvm::ArrayRef<llvm::MCRegister> SGPRs,
                          const StateValueArraySpecs &Specs) const override;

  void getAllStorageRegisters(
      llvm::SmallVectorImpl<llvm::MCRegister> &Regs) const override {
    Regs.push_back(FlatScratchSGPRHigh);
    Regs.push_back(FlatScratchSGPRLow);
    Regs.push_back(ScratchSGPR);
  }
};

/// \brief State value array storage scheme for targets with architected flat
/// scratch, where the SVA is spilled into the emergency SVA slot at the bottom
/// of the wavefront's private segment at no register cost.
///
/// The hardware initialises \c FLAT_SCRATCH on these targets and the shader
/// cannot override it, so there is nothing to shadow; and every architected-FS
/// target can encode a SADDR-less \c SCRATCH_* , so the emergency slots at
/// offsets 0 and 4 are reachable with no address register. Holding no
/// registers, this scheme can always be constructed.
struct SpilledWithNoSGPRsValueStorage : public StateValueArrayStorage {
public:
  /// method for providing LLVM RTTI
  [[nodiscard]] static bool classof(const StateValueArrayStorage *S) {
    return S->getScheme() == SVS_SPILLED_WITH_NO_SGPRS_architected_fs;
  }

  SpilledWithNoSGPRsValueStorage()
      : StateValueArrayStorage(SVS_SPILLED_WITH_NO_SGPRS_architected_fs) {};

  llvm::MCRegister getStateValueStorageReg() const override { return {}; }

  bool operator==(const StateValueArrayStorage &LHS) const override;

  bool requiresLoadAndStoreBeforeUse() const override { return true; }

  void emitCodeToLoadSVA(llvm::MachineInstr &MI,
                         llvm::MCRegister DestVGPR) const override;

  void emitCodeToStoreSVA(llvm::MachineInstr &MI,
                          llvm::MCRegister SrcVGPR) const override;

  void handOffSVA(llvm::MachineInstr &MI, const StateValueArraySpecs &Specs,
                  const llvm::GCNSubtarget &ST) const override;

  void pickOffSVA(llvm::MachineInstr &MI, const StateValueArraySpecs &Specs,
                  const llvm::GCNSubtarget &ST) const override;

  void
  emitCodeToSwitchSVS(llvm::MachineBasicBlock::iterator MI,
                      const StateValueArrayStorage &TargetSVS,
                      const StateValueArraySpecs &Specs) const override;

  void emitLongJumpSGPRSpill(llvm::MachineBasicBlock &MBB,
                             llvm::MachineBasicBlock::iterator InsertPt,
                             llvm::ArrayRef<llvm::MCRegister> SGPRs,
                             const StateValueArraySpecs &Specs) const override;

  void
  emitLongJumpSGPRRestore(llvm::MachineBasicBlock &MBB,
                          llvm::MachineBasicBlock::iterator InsertPt,
                          llvm::ArrayRef<llvm::MCRegister> SGPRs,
                          const StateValueArraySpecs &Specs) const override;

  void getAllStorageRegisters(
      llvm::SmallVectorImpl<llvm::MCRegister> &) const override {}
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