//===-- TargetModuleScavenger.h ---------------------------------*- C++ -*-===//
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
/// Target-module register scavenger — a custom version of
/// \c llvm::RegScavenger. Sole addition over stock: \c ReservedRegs —
/// a \c DenseSet of phys-regs the scavenger must never pick, regardless
/// of \c MRI.isReserved / \c LiveUnits / backward-walk state. Used by
/// \c TargetModuleBranchRelaxation to protect the SVA storage register(s).
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOL_CODE_GEN_TARGET_MODULE_SCAVENGER_H
#define LUTHIER_TOOL_CODE_GEN_TARGET_MODULE_SCAVENGER_H

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/BitVector.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/CodeGen/LiveRegUnits.h>
#include <llvm/CodeGen/MachineBasicBlock.h>
#include <llvm/CodeGen/Register.h>

namespace llvm {
class MachineFunction;
class MachineInstr;
class MachineRegisterInfo;
class TargetInstrInfo;
class TargetRegisterClass;
class TargetRegisterInfo;
} // namespace llvm

namespace luthier {

/// Sibling-class fork of \c llvm::RegScavenger
class TargetModuleScavenger {
public:
  TargetModuleScavenger() = default;

  /// Mark \p Regs as never-pick. The scavenger consults this in addition
  /// to \c MRI.isReserved, the in-flight \c LiveUnits, and the backward
  /// scan's \c Used set.
  void setReservedRegs(llvm::DenseSet<llvm::MCPhysReg> Regs) {
    ReservedRegs = std::move(Regs);
  }

  // ============ Stock RegScavenger API surface ============================
  //
  // These methods mirror their llvm::RegScavenger counterparts in
  // signature + behavior. Any deviation from stock is documented inline.

  /// See \c RegScavenger::assignRegToScavengingIndex.
  void assignRegToScavengingIndex(int FI, llvm::Register Reg,
                                  llvm::MachineInstr *Restore = nullptr);

  /// See \c RegScavenger::enterBasicBlock.
  void enterBasicBlock(llvm::MachineBasicBlock &MBB);

  /// See \c RegScavenger::enterBasicBlockEnd.
  void enterBasicBlockEnd(llvm::MachineBasicBlock &MBB);

  /// See \c RegScavenger::backward.
  void backward();

  /// See \c RegScavenger::backward(iterator).
  void backward(llvm::MachineBasicBlock::iterator I) {
    while (MBBI != I)
      backward();
  }

  /// See \c RegScavenger::isRegUsed. The extra reserved regs are reported
  /// as used regardless of \p IncludeReserved (they're always off-limits).
  bool isRegUsed(llvm::Register Reg, bool IncludeReserved = true) const;

  /// See \c RegScavenger::getRegsAvailable. The extra reserved regs
  /// never appear in the returned mask.
  llvm::BitVector getRegsAvailable(const llvm::TargetRegisterClass *RC);

  /// See \c RegScavenger::FindUnusedReg.
  llvm::Register FindUnusedReg(const llvm::TargetRegisterClass *RC) const;

  /// See \c RegScavenger::addScavengingFrameIndex.
  void addScavengingFrameIndex(int FI) { Scavenged.push_back(ScavengedInfo(FI)); }

  /// See \c RegScavenger::isScavengingFrameIndex.
  bool isScavengingFrameIndex(int FI) const {
    for (const ScavengedInfo &SI : Scavenged)
      if (SI.FrameIndex == FI)
        return true;
    return false;
  }

  /// See \c RegScavenger::getScavengingFrameIndices.
  void getScavengingFrameIndices(llvm::SmallVectorImpl<int> &A) const {
    for (const ScavengedInfo &I : Scavenged)
      if (I.FrameIndex >= 0)
        A.push_back(I.FrameIndex);
  }

  /// See \c RegScavenger::scavengeRegisterBackwards.
  llvm::Register
  scavengeRegisterBackwards(const llvm::TargetRegisterClass &RC,
                            llvm::MachineBasicBlock::iterator To,
                            bool RestoreAfter, int SPAdj,
                            bool AllowSpill = true);

  /// See \c RegScavenger::setRegUsed.
  void setRegUsed(llvm::Register Reg,
                  llvm::LaneBitmask LaneMask = llvm::LaneBitmask::getAll());

private:
  /// See \c RegScavenger::ScavengedInfo.
  struct ScavengedInfo {
    ScavengedInfo(int FI = -1) : FrameIndex(FI) {}
    int FrameIndex;
    llvm::Register Reg;
    const llvm::MachineInstr *Restore = nullptr;
  };

  bool isReserved(llvm::Register Reg) const;

  void init(llvm::MachineBasicBlock &MBB);

  ScavengedInfo &spill(llvm::Register Reg, const llvm::TargetRegisterClass &RC,
                       int SPAdj, llvm::MachineBasicBlock::iterator Before,
                       llvm::MachineBasicBlock::iterator &UseMI);

  const llvm::TargetRegisterInfo *TRI = nullptr;
  const llvm::TargetInstrInfo *TII = nullptr;
  llvm::MachineRegisterInfo *MRI = nullptr;
  llvm::MachineBasicBlock *MBB = nullptr;
  llvm::MachineBasicBlock::iterator MBBI;
  llvm::SmallVector<ScavengedInfo, 2> Scavenged;
  llvm::LiveRegUnits LiveUnits;

  /// Phys-regs the scavenger is forbidden to pick, on top of MRI's
  /// reserved set. Populated via \c setReservedRegs.
  llvm::DenseSet<llvm::MCPhysReg> ReservedRegs;
};

} // namespace luthier

#endif
