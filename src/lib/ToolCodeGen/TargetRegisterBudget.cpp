//===-- TargetRegisterBudget.cpp --------------------------------*- C++ -*-===//
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
/// Implements the application launch-budget queries declared in
/// TargetRegisterBudget.h.
//===----------------------------------------------------------------------===//
#include "luthier/ToolCodeGen/TargetRegisterBudget.h"

#include "luthier/LLVM/streams.h"
#include <AMDGPU.h>
#include <GCNSubtarget.h>
#include <SIRegisterInfo.h>
#include <algorithm>
#include <llvm/ADT/BitVector.h>
#include <llvm/CodeGen/LivePhysRegs.h>
#include <llvm/CodeGen/MachineFunction.h>
#include <llvm/CodeGen/MachineRegisterInfo.h>
#include <llvm/IR/Function.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/FormatVariadic.h>

#undef DEBUG_TYPE
#define DEBUG_TYPE "luthier-reg-budget"

namespace luthier {

void setTargetRegisterBudget(llvm::Function &F, unsigned NumSGPRs,
                             unsigned NumVGPRs) {
  F.removeFnAttr(AppNumSGPRsAttribute);
  F.removeFnAttr(AppNumVGPRsAttribute);
  F.addFnAttr(AppNumSGPRsAttribute, llvm::formatv("{0}", NumSGPRs).str());
  F.addFnAttr(AppNumVGPRsAttribute, llvm::formatv("{0}", NumVGPRs).str());
}

TargetRegisterBudget getTargetRegisterBudget(const llvm::MachineFunction &MF) {
  const llvm::Function &F = MF.getFunction();
  const auto &ST = MF.getSubtarget<llvm::GCNSubtarget>();

  const unsigned MaxSGPRs = ST.getAddressableNumSGPRs();
  const unsigned MaxVGPRs = ST.getAddressableNumVGPRs(0);

  // Prefer the launch budget CodeDiscoveryPass lifted from the kernel
  // descriptor. Fall back to the backend's allocation cap, then to the
  // subtarget maximum, so functions that never went through code discovery
  // (hand-written MIR in lit tests, for instance) still get a sane answer.
  auto readBudget = [&F](const char *Preferred, const char *Fallback,
                         unsigned Max) -> unsigned {
    if (F.hasFnAttribute(Preferred))
      return F.getFnAttributeAsParsedInteger(Preferred, Max);
    return F.getFnAttributeAsParsedInteger(Fallback, Max);
  };

  TargetRegisterBudget Budget;
  Budget.NumSGPRs = std::min<unsigned>(
      readBudget(AppNumSGPRsAttribute, "amdgpu-num-sgpr", MaxSGPRs), MaxSGPRs);
  Budget.NumVGPRs = std::min<unsigned>(
      readBudget(AppNumVGPRsAttribute, "amdgpu-num-vgpr", MaxVGPRs), MaxVGPRs);
  // A wave gets as many AGPRs as VGPRs on the subtargets that have them.
  Budget.NumAGPRs = ST.hasMAIInsts() ? Budget.NumVGPRs : 0;
  return Budget;
}

/// \return \c true if \p Reg belongs to one of the three general-purpose
/// register files (and is therefore covered by the launch budget), rather
/// than being a special-purpose register that happens to live in an
/// SGPR-shaped class such as \c VCC , \c EXEC or the \c TTMP family.
///
/// Checks the super-registers as well so that 16-bit sub-registers of a
/// VGPR/AGPR/SGPR -- which are what the reserved-set walk below actually
/// hands us on subtargets that model them -- are classified with their
/// parent.
static bool isGeneralPurposeReg(const llvm::SIRegisterInfo &TRI,
                                llvm::MCRegister Reg) {
  for (llvm::MCPhysReg Super : TRI.superregs_inclusive(Reg)) {
    if (llvm::AMDGPU::VGPR_32RegClass.contains(Super) ||
        llvm::AMDGPU::AGPR_32RegClass.contains(Super) ||
        llvm::AMDGPU::SGPR_32RegClass.contains(Super))
      return true;
  }
  return false;
}

void addAppOwnedRegisters(const llvm::MachineFunction &MF,
                          llvm::LivePhysRegs &Out) {
  const auto &ST = MF.getSubtarget<llvm::GCNSubtarget>();
  const auto *TRI = ST.getRegisterInfo();
  const llvm::MachineRegisterInfo &MRI = MF.getRegInfo();

  const TargetRegisterBudget Budget = getTargetRegisterBudget(MF);

  // The GPRs the wave was actually launched with. Everything at or above
  // these counts exists on the device but was never allocated to this wave,
  // so it holds no application state -- that is the whole point of asking
  // the budget instead of MRI.isReserved, which would report every one of
  // them as reserved (true cap) or none of them (widened cap).
  for (unsigned I = 0; I < Budget.NumSGPRs; ++I)
    Out.addReg(llvm::AMDGPU::SGPR0 + I);
  for (unsigned I = 0; I < Budget.NumVGPRs; ++I)
    Out.addReg(llvm::AMDGPU::VGPR0 + I);
  for (unsigned I = 0; I < Budget.NumAGPRs; ++I)
    Out.addReg(llvm::AMDGPU::AGPR0 + I);

  Out.addReg(llvm::AMDGPU::SCC);
  Out.addReg(TRI->getVCC());

  // Everything outside the GPR files that the runtime owns and that can
  // carry application-visible state across an instrumentation point --
  // EXEC, FLAT_SCR, M0, the TTMP/TBA/TMA family, the SRC_* hardware
  // registers. The reserved set is the only enumeration of those, but the
  // GPRs it also contains are filtered out: their ownership is decided by
  // the launch budget above.
  //
  // Only leaf physical registers (those with no proper sub-register) are
  // added. AMDGPU stacks each register file into a tower of tuple classes
  // (SGPR_32 -> SGPR_64 -> ... -> SGPR_1024), which puts the RegId space at
  // ~21k entries on gfx1036. Adding every reserved RegId makes each addReg
  // fan out through subregs_inclusive and lands LivePhysRegs' SparseSet at
  // ~19.5k distinct entries, so every later unionLivePhysRegs walks all of
  // them -- turning the liveness fixed point and
  // InjectedPayloadPreserveLiveRegsPass's per-MBB union into O(19.5k) work
  // per block. The leaves carry the same information, since LivePhysRegs'
  // consumers walk to sub-registers themselves.
  const llvm::BitVector &Reserved = MRI.getReservedRegs();
  for (unsigned RegId = 0, E = Reserved.size(); RegId < E; ++RegId) {
    if (!Reserved.test(RegId))
      continue;
    llvm::MCRegister Reg{static_cast<llvm::MCPhysReg>(RegId)};
    auto Subs = TRI->subregs(Reg);
    if (Subs.begin() != Subs.end())
      continue;
    if (isGeneralPurposeReg(*TRI, Reg))
      continue;
    Out.addReg(Reg);
  }

  LLVM_DEBUG(luthier::dbgs()
             << "[TargetRegisterBudget] app-owned seed for '" << MF.getName()
             << "': " << Budget.NumSGPRs << " SGPRs, " << Budget.NumVGPRs
             << " VGPRs, " << Budget.NumAGPRs << " AGPRs\n");
}

bool isAppOwnedGPR(const llvm::MachineFunction &MF, llvm::MCRegister Reg) {
  const auto &ST = MF.getSubtarget<llvm::GCNSubtarget>();
  const auto *TRI = ST.getRegisterInfo();
  const TargetRegisterBudget Budget = getTargetRegisterBudget(MF);

  // Each file's 32-bit registers are contiguous in the register enum -- the
  // same assumption every `AMDGPU::VGPR0 + I` in the backend makes -- so a
  // constituent's index within its file is just its distance from the file's
  // first register.
  auto inBudget = [](llvm::MCPhysReg Sub, unsigned First, unsigned Count) {
    return Sub >= First && Sub < First + Count;
  };

  // Mirrors SIRegisterInfo::reserveRegisterTuples: a tuple counts as owned
  // when any of its constituents lies inside the budget.
  for (llvm::MCPhysReg Sub : TRI->subregs_inclusive(Reg)) {
    if (llvm::AMDGPU::SGPR_32RegClass.contains(Sub) &&
        inBudget(Sub, llvm::AMDGPU::SGPR0, Budget.NumSGPRs))
      return true;
    if (llvm::AMDGPU::VGPR_32RegClass.contains(Sub) &&
        inBudget(Sub, llvm::AMDGPU::VGPR0, Budget.NumVGPRs))
      return true;
    if (llvm::AMDGPU::AGPR_32RegClass.contains(Sub) &&
        inBudget(Sub, llvm::AMDGPU::AGPR0, Budget.NumAGPRs))
      return true;
  }
  return false;
}

bool isReservedForApp(const llvm::MachineFunction &MF, llvm::MCRegister Reg) {
  const auto &ST = MF.getSubtarget<llvm::GCNSubtarget>();
  const auto *TRI = ST.getRegisterInfo();
  const llvm::MachineRegisterInfo &MRI = MF.getRegInfo();

  if (!MRI.isReserved(Reg))
    return false;
  // The subtarget reserves a register either because it is special-purpose
  // or because it sits above the function's \c amdgpu-num-* allocation cap.
  // Only the first kind is off limits: a GPR the wave was not launched with
  // holds nothing, and instrumentation grows the kernel's register request,
  // so the register will exist by the time the code runs.
  return !isGeneralPurposeReg(*TRI, Reg) || isAppOwnedGPR(MF, Reg);
}

bool isAvailableForInstrumentation(const llvm::MachineFunction &MF,
                                   llvm::MCRegister Reg) {
  const auto &ST = MF.getSubtarget<llvm::GCNSubtarget>();
  const auto *TRI = ST.getRegisterInfo();
  const llvm::MachineRegisterInfo &MRI = MF.getRegInfo();

  // Special-purpose registers are off limits no matter what the budget says.
  if (!TRI->isInAllocatableClass(Reg))
    return false;
  // Anything the application code in this MF touches is off limits.
  if (MRI.isPhysRegUsed(Reg))
    return false;
  return !isReservedForApp(MF, Reg);
}

} // namespace luthier
