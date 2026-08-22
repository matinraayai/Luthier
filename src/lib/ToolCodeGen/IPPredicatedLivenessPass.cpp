//===-- IPPredicatedLivenessPass.cpp --------------------------------------===//
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
#include "luthier/ToolCodeGen/PredicatedMachineBasicBlock.h"
#include <llvm/CodeGen/MachineBasicBlock.h>
#include <AMDGPU.h>
#include <GCNSubtarget.h>
#include <SIInstrInfo.h>
#include <SIRegisterInfo.h>
#include <algorithm>
#include <memory>
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
// LivePhysRegs helpers
//===----------------------------------------------------------------------===//

/// Copy contents from \p Src into \p Dst. Both must be initialised with a
/// \c TargetRegisterInfo.
static void copyLivePhysRegs(llvm::LivePhysRegs &Dst,
                             const llvm::LivePhysRegs &Src) {
  Dst.clear();
  for (llvm::MCPhysReg R : Src)
    Dst.addReg(R);
}

/// Union all registers in \p Src into \p Dst.
static void unionLivePhysRegs(llvm::LivePhysRegs &Dst,
                              const llvm::LivePhysRegs &Src) {
  for (llvm::MCPhysReg R : Src)
    Dst.addReg(R);
}

/// Equality check via double iteration — \c LivePhysRegs offers \c contains
/// and iteration but no built-in comparison operator.
static bool livePhysRegsEqual(const llvm::LivePhysRegs &A,
                              const llvm::LivePhysRegs &B) {
  size_t CountA = 0;
  for (llvm::MCPhysReg R : A) {
    ++CountA;
    if (!B.contains(R))
      return false;
  }
  size_t CountB = 0;
  for (llvm::MCPhysReg R : B) {
    (void)R;
    ++CountB;
  }
  return CountA == CountB;
}

//===----------------------------------------------------------------------===//
// Per-MF allocatable-GPR pool builder (for local-mode initial live-out)
//===----------------------------------------------------------------------===//

/// Build the initial "everything live" set for the not-fully-discovered
/// fallback: the union of the function's allocatable SGPR / VGPR / AGPR
/// pool (sized by \c amdgpu-num-sgpr / \c amdgpu-num-vgpr) plus every
/// reserved register from MRI (includes VCC, EXEC, FLAT_SCR, and the
/// runtime-owned TTMP/TBA/TMA/MODE family — all of which can hold
/// application-visible state across an instrumentation point).
static void buildAllocatableSet(const llvm::MachineFunction &MF,
                                llvm::LivePhysRegs &Out) {
  const llvm::Function &F = MF.getFunction();
  const auto &ST = MF.getSubtarget<llvm::GCNSubtarget>();
  const llvm::MachineRegisterInfo &MRI = MF.getRegInfo();

  unsigned NumSGPRs = F.getFnAttributeAsParsedInteger("amdgpu-num-sgpr");
  unsigned NumVGPRs = F.getFnAttributeAsParsedInteger("amdgpu-num-vgpr");

  for (unsigned I = 0; I < NumSGPRs; ++I)
    Out.addReg(llvm::AMDGPU::SGPR0 + I);
  for (unsigned I = 0; I < NumVGPRs; ++I)
    Out.addReg(llvm::AMDGPU::VGPR0 + I);
  if (ST.hasMAIInsts()) {
    for (unsigned I = 0; I < NumVGPRs; ++I)
      Out.addReg(llvm::AMDGPU::AGPR0 + I);
  }
  const llvm::BitVector &Reserved = MRI.getReservedRegs();
  for (unsigned RegId = 0, E = Reserved.size(); RegId < E; ++RegId) {
    if (!Reserved.test(RegId))
      continue;
    Out.addReg(static_cast<llvm::MCPhysReg>(RegId));
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

  IPPredicatedCFG &CFG =
      IPAM.getResult<IPPredCFGAnalysis>(IP).getVecCFG();

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

  // Grab a TRI once for LivePhysRegs initialisation. Every MF in the CFG
  // shares the same GCNSubtarget/SIRegisterInfo, so any PMBB's parent MF
  // is sufficient.
  const llvm::TargetRegisterInfo &TRI =
      *CFG.begin()->getMBB().getParent()->getSubtarget().getRegisterInfo();

  // ---- Initialize per-PMBB state --------------------------------------
  // \c LivePhysRegs deletes copy and has no implicit move constructor,
  // so it can't sit directly in a \c DenseMap value — the public result
  // map carries each set via \c unique_ptr.
  for (PredicatedMachineBasicBlock &PMBB : CFG)
    Out.LiveInsByPMBB[&PMBB] = std::make_unique<llvm::LivePhysRegs>(TRI);

  // Local-mode: seed every exit PMBB's *live-out* with the function's
  // allocatable GPR pool. An "exit PMBB" here is any PMBB that has no
  // successors in the CFG — since the CFG is inter-procedural, function
  // exit blocks only end up here when their callee chain is incomplete or
  // they have no callee chain at all (true return blocks).
  //
  // The seed must live on the live-OUT side, not on the PMBB's recorded
  // live-in: the fixed-point loop recomputes the per-PMBB live-out from
  // its successors' converged live-in on every iteration. If we wrote
  // the seed into the recorded live-in directly, the first iteration
  // would observe "no successors → Out empty," walk backward, and
  // overwrite the seed — leaving local mode no different from
  // fully-discovered mode for an MF with no IP edges.
  llvm::DenseMap<const PredicatedMachineBasicBlock *,
                 std::unique_ptr<llvm::LivePhysRegs>>
      ExitSeed;
  if (!IsFullyDiscovered) {
    llvm::DenseMap<const llvm::MachineFunction *,
                   std::unique_ptr<llvm::LivePhysRegs>>
        PerMFAllocSet;
    for (PredicatedMachineBasicBlock &PMBB : CFG) {
      if (PMBB.succs_begin() != PMBB.succs_end())
        continue;
      const llvm::MachineFunction *MF = PMBB.getMBB().getParent();
      auto &SetPtr = PerMFAllocSet[MF];
      if (!SetPtr) {
        SetPtr = std::make_unique<llvm::LivePhysRegs>(TRI);
        buildAllocatableSet(*MF, *SetPtr);
      }
      auto &Seed = ExitSeed[&PMBB];
      Seed = std::make_unique<llvm::LivePhysRegs>(TRI);
      copyLivePhysRegs(*Seed, *SetPtr);
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
  bool AnyChange = true;
  unsigned Iter = 0;
  auto computeLiveOut = [&](const PredicatedMachineBasicBlock *PMBB,
                            llvm::LivePhysRegs &Live) {
    Live.clear();
    auto SeedIt = ExitSeed.find(PMBB);
    if (SeedIt != ExitSeed.end())
      copyLivePhysRegs(Live, *SeedIt->second);
    for (const PredicatedMachineBasicBlock &Succ : PMBB->successors()) {
      auto SIt = Out.LiveInsByPMBB.find(&Succ);
      if (SIt == Out.LiveInsByPMBB.end())
        continue;
      unionLivePhysRegs(Live, *SIt->second);
    }
  };

  llvm::LivePhysRegs Live(TRI);
  while (AnyChange) {
    AnyChange = false;
    ++Iter;
    LLVM_DEBUG(luthier::dbgs() << "  iter " << Iter << "\n");
    for (PredicatedMachineBasicBlock *PMBB : POOrder) {
      computeLiveOut(PMBB, Live);
      const llvm::MachineBasicBlock &MBB = PMBB->getMBB();
      for (auto MIt = MBB.rbegin(), MEnd = MBB.rend(); MIt != MEnd; ++MIt)
        Live.stepBackward(*MIt);
      auto &Cur = *Out.LiveInsByPMBB[PMBB];
      if (!livePhysRegsEqual(Cur, Live)) {
        copyLivePhysRegs(Cur, Live);
        AnyChange = true;
      }
    }
  }

  return Out;
}

llvm::PreservedAnalyses
IPPredicatedLivenessPrinter::run(Prototype &IP,
                                 PrototypeAnalysisManager &IPAM) {
  const IPPredicatedLiveness &Liveness =
      IPAM.getResult<IPPredicatedLivenessAnalysis>(IP);
  const IPPredicatedCFG &CFG =
      IPAM.getResult<IPPredCFGAnalysis>(IP).getVecCFG();

  OS << "IPPredicatedLiveness for prototype '" << IP.getName() << "':\n";
  OS << "  fully-discovered: "
     << (Liveness.isFullyDiscovered() ? "true" : "false") << "\n";
  for (const PredicatedMachineBasicBlock &PMBB : CFG) {
    const llvm::MachineBasicBlock &MBB = PMBB.getMBB();
    const llvm::MachineFunction &MF = *MBB.getParent();
    const llvm::TargetRegisterInfo &TRI =
        *MF.getSubtarget().getRegisterInfo();
    OS << "  " << MF.getName() << ':' << llvm::printMBBReference(MBB)
       << "  live-ins: {";
    if (const llvm::LivePhysRegs *Live = Liveness.getPMBBLiveIns(PMBB)) {
      llvm::SmallVector<llvm::MCPhysReg, 32> Sorted(Live->begin(), Live->end());
      llvm::sort(Sorted);
      for (llvm::MCPhysReg R : Sorted)
        OS << ' ' << llvm::printReg(R, &TRI);
    }
    OS << " }\n";
  }
  return llvm::PreservedAnalyses::all();
}

} // namespace luthier
