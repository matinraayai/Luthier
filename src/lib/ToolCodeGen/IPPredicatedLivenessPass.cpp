//===-- IPPredicatedLivenessPass.cpp --------------------------------===//
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
/// \file IPPredicatedLivenessPass.cpp
/// Implements \c IPPredicatedLivenessAnalysis.
//===----------------------------------------------------------------------===//
#include "luthier/ToolCodeGen/IPPredicatedLivenessPass.h"
#include "luthier/Common/ErrorCheck.h"
#include "luthier/Common/GenericLuthierError.h"
#include "luthier/LLVM/streams.h"
#include "luthier/ToolCodeGen/IPPredicatedCFG.h"
#include "luthier/ToolCodeGen/InjectedPayloadAndInstPointAnalysis.h"
#include "luthier/ToolCodeGen/InjectedPayloadSideEffectsAnalysis.h"
#include "luthier/ToolCodeGen/MIRConvenience.h"
#include "luthier/ToolCodeGen/PredicatedMachineBasicBlock.h"
#include <AMDGPU.h>
#include <GCNSubtarget.h>
#include <SIInstrInfo.h>
#include <SIRegisterInfo.h>
#include <algorithm>
#include <llvm/ADT/PostOrderIterator.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/CodeGen/LivePhysRegs.h>
#include <llvm/CodeGen/MachineFunction.h>
#include <llvm/CodeGen/MachineInstr.h>
#include <llvm/CodeGen/MachineRegisterInfo.h>
#include <llvm/CodeGen/TargetRegisterInfo.h>
#include <llvm/CodeGen/TargetSubtargetInfo.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/FormatVariadic.h>

#undef DEBUG_TYPE
#define DEBUG_TYPE "luthier-ip-pred-liveness"

namespace luthier {

namespace {

//===----------------------------------------------------------------------===//
// EXEC-effect classification
//===----------------------------------------------------------------------===//

enum class ExecEffect {
  None,         ///< MI does not write EXEC / EXEC_LO / EXEC_HI.
  CompleteFlip, ///< MI fully flips EXEC (swap active and inactive sets).
  PartialFlip,  ///< MI writes EXEC in any other way (union both sets).
};

/// Returns true if \p Reg is EXEC, EXEC_LO, or EXEC_HI.
static bool isExecReg(llvm::MCRegister Reg) {
  return Reg == llvm::AMDGPU::EXEC || Reg == llvm::AMDGPU::EXEC_LO ||
         Reg == llvm::AMDGPU::EXEC_HI;
}

/// Returns true if MI's opcode + operand shape match a known
/// "complete-flip" EXEC update — currently only \c S_NOT_B{32,64} with
/// \c exec as both src and dst. Future work will refine this via the
/// IR-translator's EXEC-value tracking; for now opcode patterns suffice.
static bool isCompleteFlipPattern(const llvm::MachineInstr &MI) {
  unsigned Opc = MI.getOpcode();
  if (Opc != llvm::AMDGPU::S_NOT_B32 && Opc != llvm::AMDGPU::S_NOT_B64)
    return false;
  if (MI.getNumOperands() < 2)
    return false;
  const auto &Dst = MI.getOperand(0);
  const auto &Src = MI.getOperand(1);
  if (!Dst.isReg() || !Src.isReg())
    return false;
  return isExecReg(Dst.getReg()) && isExecReg(Src.getReg()) &&
         Dst.getReg() == Src.getReg();
}

/// Classify \p MI's effect on EXEC. Any MI whose def list (explicit or
/// implicit, via MCInstrDesc or operand list) includes EXEC, EXEC_LO, or
/// EXEC_HI is treated as an EXEC writer.
static ExecEffect classifyExecEffect(const llvm::MachineInstr &MI,
                                     const llvm::TargetRegisterInfo &TRI) {
  // Check explicit def operands.
  bool WritesExec = false;
  for (const llvm::MachineOperand &MO : MI.operands()) {
    if (!MO.isReg() || !MO.isDef() || !MO.getReg().isPhysical())
      continue;
    if (isExecReg(MO.getReg())) {
      WritesExec = true;
      break;
    }
    // Also account for super-regs of EXEC_LO/EXEC_HI via regs-overlap.
    if (TRI.regsOverlap(MO.getReg(), llvm::AMDGPU::EXEC)) {
      WritesExec = true;
      break;
    }
  }
  if (!WritesExec) {
    // Check the instruction descriptor's implicit defs.
    if (MI.getDesc().hasImplicitDefOfPhysReg(llvm::AMDGPU::EXEC, &TRI) ||
        MI.getDesc().hasImplicitDefOfPhysReg(llvm::AMDGPU::EXEC_LO, &TRI) ||
        MI.getDesc().hasImplicitDefOfPhysReg(llvm::AMDGPU::EXEC_HI, &TRI))
      WritesExec = true;
  }
  if (!WritesExec)
    return ExecEffect::None;
  return isCompleteFlipPattern(MI) ? ExecEffect::CompleteFlip
                                   : ExecEffect::PartialFlip;
}

//===----------------------------------------------------------------------===//
// Read-only register filter
//===----------------------------------------------------------------------===//

/// Is \p Reg conceptually read-only by user code (so we don't need to track
/// its liveness in the dataflow)? TTMP family, TBA/TMA, MODE — set by the
/// runtime/hardware and not part of preservable application state.
static bool isReadOnlyForLuthier(llvm::MCPhysReg Reg) {
  if (Reg >= llvm::AMDGPU::TTMP0 && Reg <= llvm::AMDGPU::TTMP15)
    return true;
  if (Reg == llvm::AMDGPU::TBA || Reg == llvm::AMDGPU::TBA_LO ||
      Reg == llvm::AMDGPU::TBA_HI || Reg == llvm::AMDGPU::TMA ||
      Reg == llvm::AMDGPU::TMA_LO || Reg == llvm::AMDGPU::TMA_HI)
    return true;
  if (Reg == llvm::AMDGPU::MODE)
    return true;
  return false;
}

//===----------------------------------------------------------------------===//
// Per-MF allocatable-GPR pool builder (for local-mode initial live-out)
//===----------------------------------------------------------------------===//

/// Build the initial "everything live" set for the not-fully-discovered
/// fallback: the union of the function's allocatable SGPR / VGPR / AGPR
/// pool (sized by \c amdgpu-num-sgpr / \c amdgpu-num-vgpr) plus any
/// reserved-but-not-read-only registers from MRI.
static void buildAllocatableSet(const llvm::MachineFunction &MF,
                                llvm::DenseSet<llvm::MCPhysReg> &Out) {
  const llvm::Function &F = MF.getFunction();
  const auto &ST = MF.getSubtarget<llvm::GCNSubtarget>();
  const llvm::MachineRegisterInfo &MRI = MF.getRegInfo();
  const llvm::TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();

  unsigned NumSGPRs = F.getFnAttributeAsParsedInteger("amdgpu-num-sgpr");
  unsigned NumVGPRs = F.getFnAttributeAsParsedInteger("amdgpu-num-vgpr");

  for (unsigned I = 0; I < NumSGPRs; ++I) {
    llvm::MCPhysReg Reg = llvm::AMDGPU::SGPR0 + I;
    if (!isReadOnlyForLuthier(Reg))
      Out.insert(Reg);
  }
  for (unsigned I = 0; I < NumVGPRs; ++I) {
    llvm::MCPhysReg Reg = llvm::AMDGPU::VGPR0 + I;
    if (!isReadOnlyForLuthier(Reg))
      Out.insert(Reg);
  }
  if (ST.hasMAIInsts()) {
    for (unsigned I = 0; I < NumVGPRs; ++I) {
      llvm::MCPhysReg Reg = llvm::AMDGPU::AGPR0 + I;
      if (!isReadOnlyForLuthier(Reg))
        Out.insert(Reg);
    }
  }
  // Add reserved-but-not-read-only regs (VCC, EXEC, FLAT_SCR, etc.) — they
  // can hold application state across the instrumentation point even though
  // regalloc doesn't allocate them.
  const llvm::BitVector &Reserved = MRI.getReservedRegs();
  for (unsigned RegId = 0, E = Reserved.size(); RegId < E; ++RegId) {
    if (!Reserved.test(RegId))
      continue;
    llvm::MCPhysReg Reg = RegId;
    if (isReadOnlyForLuthier(Reg))
      continue;
    Out.insert(Reg);
  }
  (void)TRI;
}

//===----------------------------------------------------------------------===//
// Live set primitives operating on DenseSet<MCPhysReg>
//===----------------------------------------------------------------------===//

/// Step backward over the MI \p MI w.r.t. live set \p L: kill defs and
/// add uses. Operates on physical-register granularity (no sub-reg lane
/// tracking — sufficient for our liveness purposes since the target MIR
/// is already in physreg form post-lifting). Reads-but-not-defs and
/// "kills" are handled by the standard LLVM operand semantics.
static void stepBackward(const llvm::MachineInstr &MI,
                         llvm::DenseSet<llvm::MCPhysReg> &L,
                         const llvm::TargetRegisterInfo &TRI) {
  // Pass 1: remove defs.
  for (const llvm::MachineOperand &MO : MI.operands()) {
    if (!MO.isReg() || !MO.isDef() || MO.isImplicit())
      continue;
    llvm::Register Reg = MO.getReg();
    if (!Reg.isPhysical())
      continue;
    L.erase(Reg.asMCReg());
    // Erase sub-regs and super-regs of the def to be conservative on
    // partial overlaps.
    for (llvm::MCRegAliasIterator AI(Reg.asMCReg(), &TRI, false); AI.isValid();
         ++AI)
      L.erase(*AI);
  }
  // Pass 2: explicit implicit defs.
  for (const llvm::MachineOperand &MO : MI.operands()) {
    if (!MO.isReg() || !MO.isDef() || !MO.isImplicit())
      continue;
    llvm::Register Reg = MO.getReg();
    if (!Reg.isPhysical())
      continue;
    L.erase(Reg.asMCReg());
    for (llvm::MCRegAliasIterator AI(Reg.asMCReg(), &TRI, false); AI.isValid();
         ++AI)
      L.erase(*AI);
  }
  // Pass 3: add uses. Mirror LLVM's LivePhysRegs::addReg by walking
  // `subregs_inclusive` and inserting each sub-reg. Without this, a USE
  // of a wider composite reg (e.g. `$vgpr0_vgpr1 = V_LSHLREV_B64 2,
  // $vgpr0_vgpr1`) only re-adds the pair to the live set after Pass 1's
  // alias-erase, leaving the leaf `$vgpr0` / `$vgpr1` out — even though
  // the wider use's bits cover them. A later (going backward = earlier
  // in program order) MI that uses only the leaf then doesn't see prior
  // liveness propagated past the wider USE+DEF and the analysis drops
  // the leaf from the Active set at IPs upstream of that point.
  for (const llvm::MachineOperand &MO : MI.operands()) {
    if (!MO.isReg() || !MO.readsReg())
      continue;
    llvm::Register Reg = MO.getReg();
    if (!Reg.isPhysical())
      continue;
    if (isReadOnlyForLuthier(Reg.asMCReg()))
      continue;
    for (llvm::MCPhysReg SubReg : TRI.subregs_inclusive(Reg.asMCReg())) {
      if (isReadOnlyForLuthier(SubReg))
        continue;
      L.insert(SubReg);
    }
  }
}

/// Apply the payload's declared effects in reverse: kill its writes from
/// \p L, then add its reads.
static void stepBackwardOverPayload(const InjectedPayloadSideEffects &Eff,
                                    llvm::DenseSet<llvm::MCPhysReg> &L,
                                    const llvm::TargetRegisterInfo &TRI) {
  for (llvm::MCRegister W : Eff.writes()) {
    L.erase(W);
    for (llvm::MCRegAliasIterator AI(W, &TRI, false); AI.isValid(); ++AI)
      L.erase(*AI);
  }
  for (llvm::MCRegister R : Eff.reads()) {
    // Mirror stepBackward Pass 3: insert the read reg AND its sub-regs
    // so the leaf entries from later uses survive a payload that
    // declares a wider read.
    for (llvm::MCPhysReg SubReg : TRI.subregs_inclusive(R)) {
      if (isReadOnlyForLuthier(SubReg))
        continue;
      L.insert(SubReg);
    }
  }
}

/// Union all of \p Src into \p Dst.
static bool unionInto(llvm::DenseSet<llvm::MCPhysReg> &Dst,
                      const llvm::DenseSet<llvm::MCPhysReg> &Src) {
  bool Changed = false;
  for (llvm::MCPhysReg R : Src)
    Changed |= Dst.insert(R).second;
  return Changed;
}

//===----------------------------------------------------------------------===//
// Per-PMBB liveness state used during the fixed-point iteration
//===----------------------------------------------------------------------===//

struct PMBBLive {
  llvm::DenseSet<llvm::MCPhysReg> LiveInActive;
  llvm::DenseSet<llvm::MCPhysReg> LiveInInactive;
};

using PayloadSideEffectsMap =
    llvm::DenseMap<const llvm::Function *, const InjectedPayloadSideEffects *>;

/// Walk \p PMBB backward from \p LiveOutA / \p LiveOutI, updating in place.
/// If \p CaptureMap is non-null, snapshot the pre-payload live sets into it
/// for each payload encountered (used for the post-convergence extraction
/// pass). Returns the resulting live-in pair on the caller's behalf via the
/// input parameters.
static void walkPMBBBackward(
    const PredicatedMachineBasicBlock &PMBB,
    llvm::DenseSet<llvm::MCPhysReg> &OutA,
    llvm::DenseSet<llvm::MCPhysReg> &OutI,
    const InjectedPayloadAndInstPoint &IPIP,
    const PayloadSideEffectsMap &SideEffectsByPayload,
    const llvm::TargetRegisterInfo &TRI,
    IPPredicatedLiveness::PayloadLiveSetsMap *CaptureMap) {
  const llvm::MachineBasicBlock &MBB = PMBB.getMBB();
  bool Vector = isVectorMBB(MBB);

  for (auto MIt = MBB.rbegin(), MEnd = MBB.rend(); MIt != MEnd; ++MIt) {
    const llvm::MachineInstr &MI = *MIt;

    // EXEC manipulation: apply swap/union BEFORE stepping backward over the
    // MI's own def/uses (the MI's def of EXEC is then a normal liveness
    // step).
    ExecEffect EE = classifyExecEffect(MI, TRI);
    if (EE == ExecEffect::CompleteFlip) {
      std::swap(OutA, OutI);
    } else if (EE == ExecEffect::PartialFlip) {
      unionInto(OutA, OutI);
      OutI = OutA;
    }

    stepBackward(MI, OutA, TRI);
    if (!Vector)
      stepBackward(MI, OutI, TRI);

    // Snapshot AFTER stepBackward(MI) so it represents "live just before
    // MI[X] starts forward" — i.e., what the kernel needs alive at the
    // payload's IP from the kernel's perspective. The patcher inlines
    // payloads BEFORE the target MI, so the payload sees exactly this
    // state. Capturing BEFORE stepBackward(MI) would yield "live after
    // MI[X] forward" which is off by one. Then step backward over the
    // payload's declared effects so its writes don't leak as "still
    // live" into MIs upstream of the IP.
    if (IPIP.contains(MI)) {
      auto Payloads = IPIP.at(MI);
      for (auto It = Payloads.rbegin(); It != Payloads.rend(); ++It) {
        const llvm::Function *PayloadFn = *It;
        if (CaptureMap) {
          auto &Slot = (*CaptureMap)[PayloadFn];
          Slot.Active = OutA;
          Slot.Inactive = OutI;
        }
        auto EffIt = SideEffectsByPayload.find(PayloadFn);
        if (EffIt != SideEffectsByPayload.end() && EffIt->second != nullptr) {
          stepBackwardOverPayload(*EffIt->second, OutA, TRI);
          if (!Vector)
            stepBackwardOverPayload(*EffIt->second, OutI, TRI);
        }
      }
    }
  }
}

} // namespace

bool IPPredicatedLiveness::invalidate(
    Prototype &, const llvm::PreservedAnalyses &PA,
    PrototypeAnalysisManager::Invalidator &) {
  auto PAC = PA.getChecker<IPPredicatedLivenessAnalysis>();
  return !PAC.preserved() &&
         !PAC.preservedSet<llvm::AllAnalysesOn<Prototype>>();
}

llvm::AnalysisKey IPPredicatedLivenessAnalysis::Key;

IPPredicatedLivenessAnalysis::Result
IPPredicatedLivenessAnalysis::run(
    Prototype &IP, PrototypeAnalysisManager &IPAM) {
  LLVM_DEBUG(luthier::dbgs()
             << "=== Luthier IModule IP-Predicated Liveness Analysis ===\n");

  Result Out;

  llvm::Module &IModule = IP.getInstrumentationModule();

  // This pass only reads the instrumentation module, so it goes through that
  // module's own managers.
  llvm::ModuleAnalysisManager &MAM =
      IPAM.getResult<IModuleAnalysisManagerPrototypeProxy>(IP).getManager();

  IPPredicatedCFG &CFG =
      IPAM.getResult<IPPredCFGAnalysis>(IP).getVecCFG();

  const InjectedPayloadAndInstPoint &IPIP =
      IPAM.getResult<InjectedPayloadAndInstPointAnalysis>(IP);

  // The IModule's payload side-effects come from a function-level analysis;
  // pre-collect the per-payload result for every payload named in the IPIP
  // so the backward walk can look them up by \c Function* without threading
  // the FAM through every helper.
  llvm::FunctionAnalysisManager &IFAM =
      MAM.getResult<llvm::FunctionAnalysisManagerModuleProxy>(IModule)
          .getManager();
  PayloadSideEffectsMap SideEffectsByPayload;
  for (auto It = IPIP.payload_mi_begin(), End = IPIP.payload_mi_end();
       It != End; ++It) {
    llvm::Function *PayloadFn = It->first;
    SideEffectsByPayload[PayloadFn] =
        &IFAM.getResult<InjectedPayloadSideEffectsAnalysis>(*PayloadFn);
  }

  // ---- Fully-discovered check ------------------------------------------
  bool IsFullyDiscovered = true;
  for (const PredicatedMachineBasicBlock &PMBB : CFG) {
    if (PMBB.hasUnresolvedEdges()) {
      IsFullyDiscovered = false;
      break;
    }
  }
  Out.ResultFullyDiscovered = IsFullyDiscovered;
  LLVM_DEBUG(luthier::dbgs()
             << "  IsFullyDiscovered=" << IsFullyDiscovered << "\n");

  if (CFG.empty())
    return Out;

  // ---- Initialize per-PMBB state --------------------------------------
  llvm::DenseMap<const PredicatedMachineBasicBlock *, PMBBLive> State;
  for (PredicatedMachineBasicBlock &PMBB : CFG)
    State[&PMBB] = PMBBLive{};

  // Local-mode: seed every exit PMBB's *live-out* with the function's
  // allocatable GPR pool. An "exit PMBB" here is any PMBB that has no
  // successors in the CFG — since the CFG is inter-procedural, function
  // exit blocks only end up here when their callee chain is incomplete or
  // they have no callee chain at all (true return blocks).
  //
  // The seed must live on the live-OUT side, not on the PMBB's recorded
  // LiveIn: the fixed-point loop recomputes the per-PMBB OutA/OutI from
  // its successors' converged LiveInActive/LiveInInactive on every
  // iteration. If we wrote the seed into State[PMBB].LiveInActive
  // directly, the first iteration would observe "no successors → OutA
  // empty," walk backward, and overwrite the seed — leaving local mode
  // no different from fully-discovered mode for an MF with no IP edges.
  llvm::DenseMap<const PredicatedMachineBasicBlock *, PMBBLive> ExitSeed;
  if (!IsFullyDiscovered) {
    llvm::DenseMap<const llvm::MachineFunction *,
                   llvm::DenseSet<llvm::MCPhysReg>>
        PerMFAllocSet;
    for (PredicatedMachineBasicBlock &PMBB : CFG) {
      if (PMBB.succs_begin() != PMBB.succs_end())
        continue;
      const llvm::MachineFunction *MF = PMBB.getMBB().getParent();
      auto It = PerMFAllocSet.find(MF);
      if (It == PerMFAllocSet.end()) {
        llvm::DenseSet<llvm::MCPhysReg> Set;
        buildAllocatableSet(*MF, Set);
        It = PerMFAllocSet.insert({MF, std::move(Set)}).first;
      }
      auto &Seed = ExitSeed[&PMBB];
      Seed.LiveInActive = It->second;
      Seed.LiveInInactive = It->second;
    }
  }

  // ---- Compute post-order traversal once -------------------------------
  // For backward dataflow we want to visit each PMBB after its successors
  // are settled, so we iterate in POST-order of the forward CFG (a.k.a.
  // reverse of reverse-post-order). LLVM's ReversePostOrderTraversal lets
  // us materialise this once and reuse it across fixed-point iterations.
  llvm::SmallVector<PredicatedMachineBasicBlock *, 16> POOrder;
  {
    llvm::ReversePostOrderTraversal<IPPredicatedCFG *> RPOT(&CFG);
    POOrder.assign(RPOT.begin(), RPOT.end());
    std::reverse(POOrder.begin(), POOrder.end());
  }

  // ---- Backward dataflow until fixed point -----------------------------
  // Per-payload snapshots are NOT captured during this loop — capturing
  // every iteration would be wasted work since later iterations overwrite
  // earlier ones. Instead, after convergence we do one final post-order
  // walk that calls the same backward routine with CaptureMap set.
  bool AnyChange = true;
  unsigned Iter = 0;
  auto computeLiveOut = [&](const PredicatedMachineBasicBlock *PMBB,
                            llvm::DenseSet<llvm::MCPhysReg> &OutA,
                            llvm::DenseSet<llvm::MCPhysReg> &OutI) {
    auto SeedIt = ExitSeed.find(PMBB);
    if (SeedIt != ExitSeed.end()) {
      OutA = SeedIt->second.LiveInActive;
      OutI = SeedIt->second.LiveInInactive;
    }
    for (const PredicatedMachineBasicBlock &Succ : PMBB->successors()) {
      auto SIt = State.find(&Succ);
      if (SIt == State.end())
        continue;
      unionInto(OutA, SIt->second.LiveInActive);
      unionInto(OutI, SIt->second.LiveInInactive);
    }
  };

  while (AnyChange) {
    AnyChange = false;
    ++Iter;
    LLVM_DEBUG(luthier::dbgs() << "  iter " << Iter << "\n");
    for (PredicatedMachineBasicBlock *PMBB : POOrder) {
      llvm::DenseSet<llvm::MCPhysReg> OutA, OutI;
      computeLiveOut(PMBB, OutA, OutI);
      const llvm::TargetRegisterInfo &TRI =
          *PMBB->getMBB().getParent()->getSubtarget().getRegisterInfo();
      walkPMBBBackward(*PMBB, OutA, OutI, IPIP, SideEffectsByPayload, TRI,
                       /*CaptureMap=*/nullptr);
      auto &Cur = State[PMBB];
      if (Cur.LiveInActive != OutA) {
        Cur.LiveInActive = std::move(OutA);
        AnyChange = true;
      }
      if (Cur.LiveInInactive != OutI) {
        Cur.LiveInInactive = std::move(OutI);
        AnyChange = true;
      }
    }
  }

  // ---- Post-convergence: extract per-payload pre-effects snapshots +
  // serialize per-PMBB live-ins for external consumers ----
  // Walk each PMBB once with its converged live-out (from successors'
  // converged live-ins + the exit seed if any), and let walkPMBBBackward
  // populate LiveSetsByPayload at every insertion point.
  for (PredicatedMachineBasicBlock *PMBB : POOrder) {
    llvm::DenseSet<llvm::MCPhysReg> OutA, OutI;
    computeLiveOut(PMBB, OutA, OutI);
    const llvm::TargetRegisterInfo &TRI =
        *PMBB->getMBB().getParent()->getSubtarget().getRegisterInfo();
    walkPMBBBackward(*PMBB, OutA, OutI, IPIP, SideEffectsByPayload, TRI,
                     /*CaptureMap=*/&Out.LiveSetsByPayload);
  }

  // Serialize the converged per-PMBB live-in sets into LiveInsByPMBB so
  // consumers (e.g. TargetModulePatcherPass for branch-relax scavenging)
  // can ArrayRef-borrow them without copying the DenseSets we use
  // internally for fixed-point iteration.
  Out.LiveInsByPMBB.reserve(State.size());
  for (const auto &[PMBB, Live] : State) {
    PMBBLiveIns &LiveIns = Out.LiveInsByPMBB[PMBB];
    LiveIns.Active.reserve(Live.LiveInActive.size());
    LiveIns.Inactive.reserve(Live.LiveInInactive.size());
    for (llvm::MCPhysReg R : Live.LiveInActive)
      LiveIns.Active.push_back(R);
    for (llvm::MCPhysReg R : Live.LiveInInactive)
      LiveIns.Inactive.push_back(R);
    // Deterministic order for ArrayRef consumers.
    llvm::sort(LiveIns.Active);
    llvm::sort(LiveIns.Inactive);
  }

  LLVM_DEBUG({
    for (const auto &[Fn, Sets] : Out.LiveSetsByPayload) {
      luthier::dbgs() << "  payload " << Fn->getName()
                      << " active=" << Sets.Active.size()
                   << " inactive=" << Sets.Inactive.size() << "\n";
    }
  });
  return Out;
}

} // namespace luthier
