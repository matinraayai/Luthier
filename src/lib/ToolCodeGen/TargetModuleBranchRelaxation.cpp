//===-- TargetModuleBranchRelaxation.cpp ----------------------------------===//
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
/// Target-module branch relaxer — fork of
/// \c llvm/lib/CodeGen/BranchRelaxation.cpp. Top-level
/// \c run + the offset-tracking machinery (\c scanFunction,
/// \c computeBlockSize, \c adjustBlockOffsets, \c isBlockInRange,
/// \c splitBlockBeforeInstr, \c fixupConditionalBranch,
/// \c relaxBranchInstructions) are transposed verbatim; the sole substantive
/// change is in \c fixupUnconditionalBranch, which calls
/// \c emitLongBranch instead of \c TII->insertIndirectBranch.
/// That helper mirrors \c SIInstrInfo::insertIndirectBranch's body but
/// scavenges via \c TargetModuleScavenger (which sees the \c ReservedRegs
/// set and the SVA-lane \c SpillSink the caller installed).
//===----------------------------------------------------------------------===//
#include "luthier/ToolCodeGen/TargetModuleBranchRelaxation.h"

#include "luthier/ToolCodeGen/IPPredicatedCFG.h"
#include "luthier/ToolCodeGen/IPPredicatedLivenessPass.h"
#include "luthier/ToolCodeGen/PredicatedMachineBasicBlock.h"
#include "luthier/ToolCodeGen/SVStorageAndLoadLocations.h"
#include "luthier/ToolCodeGen/StateValueArraySpecs.h"
#include "luthier/ToolCodeGen/StateValueArrayStorage.h"
#include "luthier/ToolCodeGen/TargetMachineInstrMDNode.h"

#include <AMDGPU.h>
#include <GCNSubtarget.h>
#include <SIInstrInfo.h>
#include <SIMachineFunctionInfo.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/CodeGen/LivePhysRegs.h>
#include <llvm/CodeGen/MachineBasicBlock.h>
#include <llvm/CodeGen/MachineFunction.h>
#include <llvm/CodeGen/MachineInstr.h>
#include <llvm/CodeGen/TargetInstrInfo.h>
#include <llvm/CodeGen/TargetRegisterInfo.h>
#include <llvm/CodeGen/TargetSubtargetInfo.h>
#include <llvm/IR/DebugLoc.h>
#include <llvm/MC/MCContext.h>
#include <llvm/MC/MCExpr.h>
#include <llvm/MC/MCSymbol.h>
#include <llvm/Support/Compiler.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>

#undef DEBUG_TYPE
#define DEBUG_TYPE "luthier-branch-relaxation"

namespace luthier {

namespace {

/// Worker class — fork of \c llvm::BranchRelaxation (anonymous namespace
/// class) with one substantive change: \c fixupUnconditionalBranch's
/// long-branch emission calls \c emitLongBranch with the target
/// module reg scavenger.
class TargetModuleBranchRelaxationWorker {
  struct BasicBlockInfo {
    unsigned Offset = 0;
    unsigned Size = 0;
    BasicBlockInfo() = default;
    unsigned postOffset(const llvm::MachineBasicBlock &MBB) const {
      const unsigned PO = Offset + Size;
      const llvm::Align Alignment = MBB.getAlignment();
      const llvm::Align ParentAlign = MBB.getParent()->getAlignment();
      if (Alignment <= ParentAlign)
        return llvm::alignTo(PO, Alignment);
      return llvm::alignTo(PO, Alignment) + Alignment.value() -
             ParentAlign.value();
    }
  };

  llvm::SmallVector<BasicBlockInfo, 16> BlockInfo;
  llvm::MachineBasicBlock *TrampolineInsertionPoint = nullptr;
  llvm::SmallDenseSet<
      std::pair<llvm::MachineBasicBlock *, llvm::MachineBasicBlock *>>
      RelaxedUnconditionals;
  TargetModuleScavenger RS;
  llvm::LivePhysRegs LiveRegs;

  llvm::MachineFunction *MF = nullptr;
  const llvm::TargetRegisterInfo *TRI = nullptr;
  const llvm::TargetInstrInfo *TII = nullptr;
  const llvm::TargetMachine *TM = nullptr;

  bool relaxBranchInstructions();
  void scanFunction();
  llvm::MachineBasicBlock *
  createNewBlockAfter(llvm::MachineBasicBlock &OrigMBB);
  llvm::MachineBasicBlock *createNewBlockAfter(llvm::MachineBasicBlock &OrigMBB,
                                               const llvm::BasicBlock *BB);
  llvm::MachineBasicBlock *
  splitBlockBeforeInstr(llvm::MachineInstr &MI,
                        llvm::MachineBasicBlock *DestBB);
  void adjustBlockOffsets(llvm::MachineBasicBlock &Start);
  void adjustBlockOffsets(llvm::MachineBasicBlock &Start,
                          llvm::MachineFunction::iterator End);
  bool isBlockInRange(const llvm::MachineInstr &MI,
                      const llvm::MachineBasicBlock &BB) const;
  bool fixupConditionalBranch(llvm::MachineInstr &MI);
  bool fixupUnconditionalBranch(llvm::MachineInstr &MI);
  uint64_t computeBlockSize(const llvm::MachineBasicBlock &MBB) const;
  unsigned getInstrOffset(const llvm::MachineInstr &MI) const;

  /// Only branches whose \c TargetMachineInstrMDNode carries
  /// \c canRelaxDirectBranch() = true (or which have no MDNode at
  /// all — the default for both lifted and relaxer-synthesized MIs)
  /// are eligible. A branch whose MDNode explicitly sets
  /// \c canRelaxDirectBranch=false stays as-is.
  static bool canRelaxBranch(const llvm::MachineInstr &MI);

  /// GFX12+ (\c ST.useAddPC64Inst()) long-branch emission
  void emitAddPCLongBranch(llvm::MachineBasicBlock &BranchBB,
                           llvm::MachineBasicBlock &DestBB,
                           const llvm::DebugLoc &DL);

  const IPPredicatedCFG &IPCFG;
  const IPPredicatedLiveness &IPLiveness;
  const SVStorageAndLoadLocations &SVLoc;
  const StateValueArraySpecs &Specs;

  /// Compute the SVS-storage registers that must be treated as reserved
  /// when scavenging for a long branch out of \p SourceMBB. Branches are
  /// terminators, so the enclosing \c StateValueStorageSegment is
  /// always the source MBB's LAST segment.
  llvm::DenseSet<llvm::MCPhysReg>
  getSVSReservedRegsAtBranch(const llvm::MachineBasicBlock &SourceMBB) const;

  /// AMDGPU-specific long-branch emission. Forked from
  /// \c SIInstrInfo::insertIndirectBranch (the pre-gfx12 path).
  /// Scavenges via \c RS so the SVA storage reg is excluded. When no
  /// free \c SReg_64 is available, emits an SVA-lane spill of a fixed
  /// pair (via \c emitSVALaneSpillForLongBranch) instead. \p MBB is
  /// the trampoline (BranchBB) — empty on entry. \p DestBB is the
  /// semantic branch destination (target when the scavenger finds a
  /// free reg). \p ReloadMBB is where the long jump lands when the
  /// scavenger has to spill: either DestBB itself (single-pred case,
  /// reload code prepended) or a dedicated MBB spliced in front of
  /// DestBB (multi-pred case). ReloadMBB may be non-empty in the
  /// single-pred case.
  void emitLongBranch(llvm::MachineBasicBlock &MBB,
                      llvm::MachineBasicBlock &DestBB,
                      llvm::MachineBasicBlock &ReloadMBB,
                      const llvm::DebugLoc &DL, int64_t BrOffset);

  /// Save \p Reg (an \c SReg_64) through two free SVA lanes so the
  /// long-branch trampoline can clobber it. Returns \c false if no
  /// two free lanes are available OR the source MBB's SVS segment is
  /// missing — the caller reports a hard error in that case. On \c
  /// true: emits the load-SVA + WRITELANEs in \p SpillMBB before
  /// \p SpillBefore, and the READLANEs + store-SVA in \p ReloadMBB
  /// before \p ReloadBefore.
  bool emitSVALaneSpillForLongBranch(
      llvm::MachineBasicBlock &SpillMBB,
      llvm::MachineBasicBlock::iterator SpillBefore,
      llvm::MachineBasicBlock &ReloadMBB,
      llvm::MachineBasicBlock::iterator ReloadBefore, llvm::MCRegister Reg);

public:
  TargetModuleBranchRelaxationWorker(const IPPredicatedCFG &IPCFG,
                                     const IPPredicatedLiveness &IPLiveness,
                                     const SVStorageAndLoadLocations &SVLoc,
                                     const StateValueArraySpecs &Specs)
      : IPCFG(IPCFG), IPLiveness(IPLiveness), SVLoc(SVLoc), Specs(Specs) {}

  bool run(llvm::MachineFunction &MF);
};

bool TargetModuleBranchRelaxationWorker::canRelaxBranch(
    const llvm::MachineInstr &MI) {
  const auto *MD = TargetMachineInstrMDNode::getInstrMDNodeIfExists(MI);
  if (!MD)
    return true;
  return MD->canRelaxDirectBranch();
}

void TargetModuleBranchRelaxationWorker::emitAddPCLongBranch(
    llvm::MachineBasicBlock &BranchBB, llvm::MachineBasicBlock &DestBB,
    const llvm::DebugLoc &DL) {
  assert(BranchBB.empty() && "trampoline MBB must start empty");
  auto &MCCtx = MF->getContext();
  auto *Offset = MCCtx.createTempSymbol("luthier_addpc_offset",
                                        /*AlwaysAddSuffix=*/true);
  auto *AddPC = llvm::BuildMI(BranchBB, BranchBB.end(), DL,
                              TII->get(llvm::AMDGPU::S_ADD_PC_I64))
                    .addSym(Offset, llvm::SIInstrInfo::MO_FAR_BRANCH_OFFSET)
                    .getInstr();
  auto *PostAddPCLabel = MCCtx.createTempSymbol("luthier_post_addpc",
                                                /*AlwaysAddSuffix=*/true);
  AddPC->setPostInstrSymbol(*MF, PostAddPCLabel);
  auto *OffsetExpr = llvm::MCBinaryExpr::createSub(
      llvm::MCSymbolRefExpr::create(DestBB.getSymbol(), MCCtx),
      llvm::MCSymbolRefExpr::create(PostAddPCLabel, MCCtx), MCCtx);
  Offset->setVariableValue(OffsetExpr);
}

llvm::DenseSet<llvm::MCPhysReg>
TargetModuleBranchRelaxationWorker::getSVSReservedRegsAtBranch(
    const llvm::MachineBasicBlock &SourceMBB) const {
  llvm::DenseSet<llvm::MCPhysReg> Out;
  // Walk up to the first MBB that SVLoc knows about. Both the
  // cross-MBB critical-edge splits emitted from
  // \c emitSVSSwitchesForMF and the relaxer's own \c NewBB/BranchBB
  // trampolines are absent from SVLoc; each of them has a single
  // predecessor that either is an original MBB (SVLoc'd) or itself
  // has one that is, and SVS is stable across cross-MBB edges by
  // \c emitSVSSwitchesForMF's invariant, so following the pred chain
  // yields the correct active SVS at this branch site.
  const llvm::MachineBasicBlock *Cur = &SourceMBB;
  llvm::ArrayRef<StateValueStorageSegment> Segments;
  llvm::SmallPtrSet<const llvm::MachineBasicBlock *, 4> Visited;
  while (Cur && Visited.insert(Cur).second) {
    Segments = SVLoc.getStorageIntervals(*Cur);
    if (!Segments.empty())
      break;
    if (Cur->pred_size() != 1)
      break;
    Cur = *Cur->pred_begin();
  }
  if (Segments.empty())
    return Out;
  llvm::SmallVector<llvm::MCRegister, 4> Regs;
  Segments.back().getSVS().getAllStorageRegisters(Regs);
  for (llvm::MCRegister R : Regs)
    Out.insert(R.id());
  return Out;
}

bool TargetModuleBranchRelaxationWorker::emitSVALaneSpillForLongBranch(
    llvm::MachineBasicBlock &SpillMBB,
    llvm::MachineBasicBlock::iterator SpillBefore,
    llvm::MachineBasicBlock &ReloadMBB,
    llvm::MachineBasicBlock::iterator ReloadBefore, llvm::MCRegister Reg) {
  // The pair is saved through two free SVA lanes; the SVA travels
  // through the long jump in a courier VGPR that gets stored back to
  // its permanent storage on the reload side.
  //
  //   Spill side (in BranchBB before S_GETPC):
  //     emitCodeToLoadSVA(anchor, Courier)  // no-op for VGPR schemes;
  //                                          // otherwise: spills
  //                                          // Courier's app value
  //                                          // to the emergency VGPR
  //                                          // slot, loads SVA into
  //                                          // Courier
  //     V_WRITELANE_B32 Sub0 -> Courier[l0]
  //     V_WRITELANE_B32 Sub1 -> Courier[l1]
  //     // Courier now carries the SVA-with-encoded-pair; the long
  //     // jump only touches SGPRs, so it survives across.
  //
  //   Reload side (at ReloadBefore in ReloadMBB):
  //     V_READLANE_B32 Sub0 <- Courier[l0]
  //     V_READLANE_B32 Sub1 <- Courier[l1]
  //     emitCodeToStoreSVA(anchor, Courier)  // no-op for VGPR
  //                                          // schemes; otherwise:
  //                                          // stores Courier (SVA)
  //                                          // back to permanent
  //                                          // storage, restores
  //                                          // Courier's app value
  //                                          // from the emergency
  //                                          // slot.
  //
  // SVS lookup: BranchBB (SpillMBB) is not in SVLoc. Its unique
  // predecessor IS an original MBB — the branch's source MBB — and
  // that MBB's terminator-enclosing (last) segment is the active SVS
  // at the branch site. SVS is stable across cross-MBB edges
  // (emitSVSSwitchesForMF invariant), so the SAME SVS is active on the
  // reload side regardless of whether ReloadMBB is the branch's DestBB
  // (single-pred case) or a synthetic block spliced in front of it
  // (multi-pred case).
  auto *MF = SpillMBB.getParent();
  const auto &ST = MF->getSubtarget<llvm::GCNSubtarget>();
  const auto *TII = ST.getInstrInfo();
  const auto *TRI = ST.getRegisterInfo();

  if (SpillMBB.pred_size() != 1)
    return false;
  llvm::MachineBasicBlock *SourceMBB = *SpillMBB.pred_begin();
  auto SpillSegs = SVLoc.getStorageIntervals(*SourceMBB);
  if (SpillSegs.empty())
    return false;
  const StateValueArrayStorage &SVS = SpillSegs.back().getSVS();

  unsigned WaveSize = ST.getWavefrontSize();
  auto FreeLanes = Specs.findLowestFreeLanes(2, WaveSize);
  if (FreeLanes.size() < 2)
    return false;

  llvm::MCRegister Sub0 = TRI->getSubReg(Reg, llvm::AMDGPU::sub0);
  llvm::MCRegister Sub1 = TRI->getSubReg(Reg, llvm::AMDGPU::sub1);
  if (!Sub0 || !Sub1)
    return false;

  // Courier VGPR:
  //   - VGPR SVS: the SVA VGPR itself. emitCodeToLoadSVA/StoreSVA are
  //     no-ops; the WRITELANE / READLANE run directly against the live
  //     SVA VGPR.
  //   - Spilled / AGPR SVS: V0 by convention. emitCodeToLoadSVA spills
  //     V0's app contents to the emergency VGPR slot (SP-8) and loads
  //     the SVA into V0. emitCodeToStoreSVA on the reload side stores
  //     V0 (SVA) back to its permanent storage and restores V0's app
  //     contents from SP-8.
  llvm::MCRegister Courier;
  {
    llvm::MCRegister SReg = SVS.getStateValueStorageReg();
    if (SReg && llvm::AMDGPU::VGPR_32RegClass.contains(SReg))
      Courier = SReg;
    else
      Courier = llvm::AMDGPU::VGPR0;
  }

  // Liveness on SpillMBB: MachineVerifier needs the SVS storage regs
  // and the courier live-in for the WRITELANE tied-def read.
  // SpillMBB (BranchBB) had its liveins seeded by the relaxer from
  // the source MBB's successors' liveins, which does not include
  // these. Reload-side liveness is handled by fixupUnconditionalBranch
  // after we return.
  llvm::SmallVector<llvm::MCRegister, 4> SVSStorageRegs;
  SVS.getAllStorageRegisters(SVSStorageRegs);
  for (llvm::MCRegister R : SVSStorageRegs)
    if (!SpillMBB.isLiveIn(R))
      SpillMBB.addLiveIn(R);
  if (!SpillMBB.isLiveIn(Courier))
    SpillMBB.addLiveIn(Courier);

  llvm::DebugLoc DL;

  // ---------- SPILL SIDE ----------
  // emitCodeToLoadSVA inserts BEFORE its anchor MI. SpillBefore is
  // the S_GETPC that emitLongBranch already placed at the top of
  // BranchBB, so *SpillBefore is safe to dereference. Insertion
  // order at the anchor position gives the final layout:
  // [load-SVA]  [WRITELANEs]  [S_GETPC...].
  {
    llvm::MachineInstr &Anchor = *SpillBefore;
    SVS.emitCodeToLoadSVA(Anchor, Courier);
    (void)llvm::BuildMI(SpillMBB, SpillBefore, DL,
                        TII->get(llvm::AMDGPU::V_WRITELANE_B32), Courier)
        .addReg(Sub0)
        .addImm(FreeLanes[0])
        .addReg(Courier);
    (void)llvm::BuildMI(SpillMBB, SpillBefore, DL,
                        TII->get(llvm::AMDGPU::V_WRITELANE_B32), Courier)
        .addReg(Sub1)
        .addImm(FreeLanes[1])
        .addReg(Courier);
  }

  // ---------- RELOAD SIDE ----------
  // ReloadMBB is either DestBB itself (single-pred case; ReloadBefore
  // points at DestBB's first original MI) or a fresh block spliced in
  // front of DestBB (multi-pred case; ReloadBefore == ReloadMBB.begin()
  // on an empty block). In the multi-pred case *ReloadBefore is
  // undefined; for VGPR schemes this is fine because emitCodeToStoreSVA
  // is a no-op and we just BuildMI the V_READLANEs directly. For
  // non-VGPR schemes we synthesize a placeholder S_NOP as an anchor,
  // run emitCodeToStoreSVA against it, then erase the placeholder.
  //
  // Final layout: [READLANEs] [store-SVA] [original ReloadMBB contents
  // if any].
  auto EmitReadlanes = [&](llvm::MachineBasicBlock::iterator Before) {
    (void)llvm::BuildMI(ReloadMBB, Before, DL,
                        TII->get(llvm::AMDGPU::V_READLANE_B32), Sub0)
        .addReg(Courier)
        .addImm(FreeLanes[0]);
    (void)llvm::BuildMI(ReloadMBB, Before, DL,
                        TII->get(llvm::AMDGPU::V_READLANE_B32), Sub1)
        .addReg(Courier)
        .addImm(FreeLanes[1]);
  };

  if (SVS.requiresLoadAndStoreBeforeUse()) {
    auto *Placeholder = llvm::BuildMI(ReloadMBB, ReloadBefore, DL,
                                      TII->get(llvm::AMDGPU::S_NOP))
                            .addImm(0)
                            .getInstr();
    EmitReadlanes(Placeholder->getIterator());
    SVS.emitCodeToStoreSVA(*Placeholder, Courier);
    Placeholder->eraseFromParent();
  } else {
    EmitReadlanes(ReloadBefore);
  }
  return true;
}

void TargetModuleBranchRelaxationWorker::emitLongBranch(
    llvm::MachineBasicBlock &MBB, llvm::MachineBasicBlock &DestBB,
    llvm::MachineBasicBlock &ReloadMBB, const llvm::DebugLoc &DL,
    int64_t BrOffset) {
  assert(MBB.empty() && "trampoline MBB must start empty");
  assert(MBB.pred_size() == 1 && "trampoline MBB must have exactly one pred");

  auto &MRI = MF->getRegInfo();
  const auto &ST = MF->getSubtarget<llvm::GCNSubtarget>();
  const auto *MFI = MF->getInfo<llvm::SIMachineFunctionInfo>();
  auto &MCCtx = MF->getContext();
  auto I = MBB.end();

  // FIXME (carried from stock SIInstrInfo): RegScavenger doesn't like
  // running on an empty MBB, so we materialize PCReg as a vreg first
  // and patch it up after scavenging.
  llvm::Register PCReg =
      MRI.createVirtualRegister(&llvm::AMDGPU::SReg_64RegClass);

  const bool FlushSGPRWrites = (ST.isWave64() && ST.hasVALUMaskWriteHazard()) ||
                               ST.hasVALUReadSGPRHazard();
  auto ApplyHazardWorkarounds = [&]() {
    if (FlushSGPRWrites)
      llvm::BuildMI(MBB, I, DL, TII->get(llvm::AMDGPU::S_WAITCNT_DEPCTR))
          .addImm(llvm::AMDGPU::DepCtr::encodeFieldSaSdst(0, ST));
  };

  // Build the S_GETPC / S_ADD_U32 / S_ADDC_U32 / S_SETPC_B64 sequence,
  // tagging the points the AsmPrinter resolves the offset against.
  llvm::MachineInstr *GetPC =
      llvm::BuildMI(MBB, I, DL, TII->get(llvm::AMDGPU::S_GETPC_B64), PCReg);
  ApplyHazardWorkarounds();

  auto *PostGetPCLabel =
      MCCtx.createTempSymbol("luthier_post_getpc", /*AlwaysAddSuffix=*/true);
  GetPC->setPostInstrSymbol(*MF, PostGetPCLabel);

  auto *OffsetLo =
      MCCtx.createTempSymbol("luthier_offset_lo", /*AlwaysAddSuffix=*/true);
  auto *OffsetHi =
      MCCtx.createTempSymbol("luthier_offset_hi", /*AlwaysAddSuffix=*/true);
  llvm::BuildMI(MBB, I, DL, TII->get(llvm::AMDGPU::S_ADD_U32))
      .addReg(PCReg, llvm::RegState::Define, llvm::AMDGPU::sub0)
      .addReg(PCReg, llvm::RegState::NoFlags, llvm::AMDGPU::sub0)
      .addSym(OffsetLo, llvm::SIInstrInfo::MO_FAR_BRANCH_OFFSET);
  llvm::BuildMI(MBB, I, DL, TII->get(llvm::AMDGPU::S_ADDC_U32))
      .addReg(PCReg, llvm::RegState::Define, llvm::AMDGPU::sub1)
      .addReg(PCReg, llvm::RegState::NoFlags, llvm::AMDGPU::sub1)
      .addSym(OffsetHi, llvm::SIInstrInfo::MO_FAR_BRANCH_OFFSET);
  ApplyHazardWorkarounds();

  (void)llvm::BuildMI(&MBB, DL, TII->get(llvm::AMDGPU::S_SETPC_B64))
      .addReg(PCReg);

  // Scavenge an SReg_64 to replace PCReg. If none available, fall
  // through to the SVA-lane spill of a fixed pair (SGPR0_SGPR1).
  llvm::Register LongBranchReservedReg = MFI->getLongBranchReservedReg();
  llvm::Register Scav;
  bool ScavengerSpilled = false;
  if (LongBranchReservedReg) {
    RS.enterBasicBlock(MBB);
    Scav = LongBranchReservedReg;
  } else {
    RS.enterBasicBlockEnd(MBB);
    Scav = RS.scavengeRegisterBackwards(
        llvm::AMDGPU::SReg_64RegClass, llvm::MachineBasicBlock::iterator(GetPC),
        /*RestoreAfter=*/false, /*SPAdj=*/0, /*AllowSpill=*/false);
    if (!Scav) {
      Scav = llvm::AMDGPU::SGPR0_SGPR1;
      ScavengerSpilled = true;
      if (!emitSVALaneSpillForLongBranch(MBB, GetPC->getIterator(), ReloadMBB,
                                         ReloadMBB.begin(), Scav)) {
        llvm::report_fatal_error(
            "TargetModuleBranchRelaxation: no free SReg_64 and SVA-lane "
            "spill could not be emitted (missing SVS segment or fewer than "
            "two free SVA lanes); cannot relax long branch",
            /*GenCrashDiag=*/false);
      }
    }
  }

  RS.setRegUsed(Scav);
  MRI.replaceRegWith(PCReg, Scav);
  MRI.clearVirtRegs();

  auto *DestLabel =
      !ScavengerSpilled ? DestBB.getSymbol() : ReloadMBB.getSymbol();
  auto *Offset = llvm::MCBinaryExpr::createSub(
      llvm::MCSymbolRefExpr::create(DestLabel, MCCtx),
      llvm::MCSymbolRefExpr::create(PostGetPCLabel, MCCtx), MCCtx);
  auto *Mask = llvm::MCConstantExpr::create(0xFFFFFFFFULL, MCCtx);
  OffsetLo->setVariableValue(
      llvm::MCBinaryExpr::createAnd(Offset, Mask, MCCtx));
  auto *ShAmt = llvm::MCConstantExpr::create(32, MCCtx);
  OffsetHi->setVariableValue(
      llvm::MCBinaryExpr::createAShr(Offset, ShAmt, MCCtx));
  (void)BrOffset;
}

void TargetModuleBranchRelaxationWorker::scanFunction() {
  BlockInfo.clear();
  BlockInfo.resize(MF->getNumBlockIDs());
  TrampolineInsertionPoint = nullptr;
  RelaxedUnconditionals.clear();
  for (auto &MBB : *MF) {
    BlockInfo[MBB.getNumber()].Size = computeBlockSize(MBB);
    if (MBB.getSectionID() != llvm::MBBSectionID::ColdSectionID)
      TrampolineInsertionPoint = &MBB;
  }
  adjustBlockOffsets(*MF->begin());
}

uint64_t TargetModuleBranchRelaxationWorker::computeBlockSize(
    const llvm::MachineBasicBlock &MBB) const {
  uint64_t Size = 0;
  for (const auto &MI : MBB)
    Size += TII->getInstSizeInBytes(MI);
  return Size;
}

unsigned TargetModuleBranchRelaxationWorker::getInstrOffset(
    const llvm::MachineInstr &MI) const {
  const auto *MBB = MI.getParent();
  unsigned Offset = BlockInfo[MBB->getNumber()].Offset;
  for (auto I = MBB->begin(); &*I != &MI; ++I)
    Offset += TII->getInstSizeInBytes(*I);
  return Offset;
}

void TargetModuleBranchRelaxationWorker::adjustBlockOffsets(
    llvm::MachineBasicBlock &Start) {
  adjustBlockOffsets(Start, MF->end());
}

void TargetModuleBranchRelaxationWorker::adjustBlockOffsets(
    llvm::MachineBasicBlock &Start, llvm::MachineFunction::iterator End) {
  unsigned PrevNum = Start.getNumber();
  for (auto &MBB : llvm::make_range(
           std::next(llvm::MachineFunction::iterator(Start)), End)) {
    unsigned Num = MBB.getNumber();
    BlockInfo[Num].Offset = BlockInfo[PrevNum].postOffset(MBB);
    PrevNum = Num;
  }
}

llvm::MachineBasicBlock *TargetModuleBranchRelaxationWorker::createNewBlockAfter(
    llvm::MachineBasicBlock &OrigBB) {
  return createNewBlockAfter(OrigBB, OrigBB.getBasicBlock());
}

llvm::MachineBasicBlock *TargetModuleBranchRelaxationWorker::createNewBlockAfter(
    llvm::MachineBasicBlock &OrigMBB, const llvm::BasicBlock *BB) {
  auto *NewBB = MF->CreateMachineBasicBlock(BB);
  MF->insert(++OrigMBB.getIterator(), NewBB);
  NewBB->setSectionID(OrigMBB.getSectionID());
  NewBB->setIsEndSection(OrigMBB.isEndSection());
  OrigMBB.setIsEndSection(false);
  BlockInfo.insert(BlockInfo.begin() + NewBB->getNumber(), BasicBlockInfo());
  return NewBB;
}

llvm::MachineBasicBlock *TargetModuleBranchRelaxationWorker::splitBlockBeforeInstr(
    llvm::MachineInstr &MI, llvm::MachineBasicBlock *DestBB) {
  auto *OrigBB = MI.getParent();
  auto *NewBB = MF->CreateMachineBasicBlock(OrigBB->getBasicBlock());
  MF->insert(++OrigBB->getIterator(), NewBB);
  NewBB->setSectionID(OrigBB->getSectionID());
  NewBB->setIsEndSection(OrigBB->isEndSection());
  OrigBB->setIsEndSection(false);
  NewBB->splice(NewBB->end(), OrigBB, MI.getIterator(), OrigBB->end());
  TII->insertUnconditionalBranch(*OrigBB, NewBB, llvm::DebugLoc());
  BlockInfo.insert(BlockInfo.begin() + NewBB->getNumber(), BasicBlockInfo());
  NewBB->transferSuccessors(OrigBB);
  OrigBB->addSuccessor(NewBB);
  OrigBB->addSuccessor(DestBB);
  OrigBB->updateTerminator(NewBB);
  BlockInfo[OrigBB->getNumber()].Size = computeBlockSize(*OrigBB);
  BlockInfo[NewBB->getNumber()].Size = computeBlockSize(*NewBB);
  adjustBlockOffsets(*OrigBB, std::next(NewBB->getIterator()));
  if (TRI->trackLivenessAfterRegAlloc(*MF))
    computeAndAddLiveIns(LiveRegs, *NewBB);
  return NewBB;
}

bool TargetModuleBranchRelaxationWorker::isBlockInRange(
    const llvm::MachineInstr &MI, const llvm::MachineBasicBlock &DestBB) const {
  int64_t BrOffset = getInstrOffset(MI);
  int64_t DestOffset = BlockInfo[DestBB.getNumber()].Offset;
  const auto *SrcBB = MI.getParent();
  return TII->isBranchOffsetInRange(
      MI.getOpcode(), SrcBB->getSectionID() != DestBB.getSectionID()
                          ? TM->getMaxCodeSize()
                          : DestOffset - BrOffset);
}

bool TargetModuleBranchRelaxationWorker::fixupConditionalBranch(
    llvm::MachineInstr &MI) {
  // Verbatim port from stock BranchRelaxation::fixupConditionalBranch.
  llvm::DebugLoc DL = MI.getDebugLoc();
  auto *MBB = MI.getParent();
  llvm::MachineBasicBlock *TBB = nullptr, *FBB = nullptr;
  llvm::MachineBasicBlock *NewBB = nullptr;
  llvm::SmallVector<llvm::MachineOperand, 4> Cond;

  auto insertUncondBranch = [&](llvm::MachineBasicBlock *MBB,
                                llvm::MachineBasicBlock *DestBB) {
    unsigned &BBSize = BlockInfo[MBB->getNumber()].Size;
    int NewBrSize = 0;
    TII->insertUnconditionalBranch(*MBB, DestBB, DL, &NewBrSize);
    BBSize += NewBrSize;
  };
  auto insertBranch = [&](llvm::MachineBasicBlock *MBB,
                          llvm::MachineBasicBlock *TBB,
                          llvm::MachineBasicBlock *FBB,
                          llvm::SmallVectorImpl<llvm::MachineOperand> &Cond) {
    unsigned &BBSize = BlockInfo[MBB->getNumber()].Size;
    int NewBrSize = 0;
    TII->insertBranch(*MBB, TBB, FBB, Cond, DL, &NewBrSize);
    BBSize += NewBrSize;
  };
  auto removeBranch = [&](llvm::MachineBasicBlock *MBB) {
    unsigned &BBSize = BlockInfo[MBB->getNumber()].Size;
    int RemovedSize = 0;
    TII->removeBranch(*MBB, &RemovedSize);
    BBSize -= RemovedSize;
  };
  auto updateOffsetAndLiveness = [&](llvm::MachineBasicBlock *NewBB) {
    adjustBlockOffsets(*std::prev(NewBB->getIterator()),
                       std::next(NewBB->getIterator()));
    if (TRI->trackLivenessAfterRegAlloc(*MF))
      computeAndAddLiveIns(LiveRegs, *NewBB);
  };

  bool Fail = TII->analyzeBranch(*MBB, TBB, FBB, Cond);
  assert(!Fail && "branches to be relaxed must be analyzable");
  (void)Fail;

  if (MBB->getSectionID() != TBB->getSectionID() &&
      TBB->getSectionID() == llvm::MBBSectionID::ColdSectionID &&
      TrampolineInsertionPoint != nullptr) {
    NewBB =
        createNewBlockAfter(*TrampolineInsertionPoint, MBB->getBasicBlock());
    if (isBlockInRange(MI, *NewBB)) {
      insertUncondBranch(NewBB, TBB);
      MBB->replaceSuccessor(TBB, NewBB);
      NewBB->addSuccessor(TBB);
      removeBranch(MBB);
      insertBranch(MBB, NewBB, FBB, Cond);
      TrampolineInsertionPoint = NewBB;
      updateOffsetAndLiveness(NewBB);
      return true;
    }
    TrampolineInsertionPoint->setIsEndSection(NewBB->isEndSection());
    MF->erase(NewBB);
    NewBB = nullptr;
  }

  bool ReversedCond = !TII->reverseBranchCondition(Cond);
  if (ReversedCond) {
    // NOTE: stock LLVM takes a clean-reverse optimization here when
    // FBB is in range — `cond → FBB; uncond → TBB`. That leaves the
    // uncond out-of-range, which then gets a trampoline that pushes
    // FBB further, which then makes the new cond out-of-range, ad
    // infinitum under tight `--amdgpu-s-branch-bits`. We skip the
    // optimization and always take the split-block path, which lands
    // a layout-adjacent NewBB and breaks the feedback loop. Slightly
    // worse code in the in-range case, dramatically better
    // convergence in the out-of-range case.
    if (FBB) {
      if (TBB == FBB) {
        removeBranch(MBB);
        insertUncondBranch(MBB, TBB);
        return true;
      }
      NewBB = createNewBlockAfter(*MBB);
      insertUncondBranch(NewBB, FBB);
      MBB->replaceSuccessor(FBB, NewBB);
      NewBB->addSuccessor(FBB);
      updateOffsetAndLiveness(NewBB);
    }
    auto &NextBB = *std::next(llvm::MachineFunction::iterator(MBB));
    removeBranch(MBB);
    insertBranch(MBB, &NextBB, TBB, Cond);
    return true;
  }
  if (!FBB)
    FBB = &(*std::next(llvm::MachineFunction::iterator(MBB)));
  NewBB = createNewBlockAfter(*MBB);
  insertUncondBranch(NewBB, TBB);
  MBB->replaceSuccessor(TBB, NewBB);
  NewBB->addSuccessor(TBB);
  removeBranch(MBB);
  insertBranch(MBB, NewBB, FBB, Cond);
  updateOffsetAndLiveness(NewBB);
  return true;
}

bool TargetModuleBranchRelaxationWorker::fixupUnconditionalBranch(
    llvm::MachineInstr &MI) {
  auto *MBB = MI.getParent();
  unsigned OldBrSize = TII->getInstSizeInBytes(MI);
  auto *DestBB = TII->getBranchDestBlock(MI);
  int64_t DestOffset = BlockInfo[DestBB->getNumber()].Offset;
  int64_t SrcOffset = getInstrOffset(MI);
  assert(!TII->isBranchOffsetInRange(
      MI.getOpcode(), MBB->getSectionID() != DestBB->getSectionID()
                          ? TM->getMaxCodeSize()
                          : DestOffset - SrcOffset));
  BlockInfo[MBB->getNumber()].Size -= OldBrSize;

  const auto &ST = MF->getSubtarget<llvm::GCNSubtarget>();

  // Install the SVS-storage regs active at THIS branch as the
  // scavenger's reserved set — narrower than a union over every
  // segment in the MF. The source MBB is an original MBB carried in
  // \c SVStorageAndLoadLocations; the branch is a terminator so its
  // enclosing segment is the MBB's last one. Trampoline MBBs
  // (BranchBB, RestoreBB) that the relaxer creates below inherit
  // this reserved set for the duration of \c emitLongBranch — the
  // scavenger call it makes runs against these regs. Not needed on
  // gfx12+ where \c S_ADD_PC_I64 doesn't scavenge, but it is cheap
  // to set unconditionally.
  RS.setReservedRegs(getSVSReservedRegsAtBranch(*MBB));

  llvm::MachineBasicBlock *BranchBB = MBB;
  if (!MBB->empty()) {
    BranchBB = createNewBlockAfter(*MBB);
    for (const auto *Succ : MBB->successors()) {
      for (const auto &LiveIn : Succ->liveins())
        BranchBB->addLiveIn(LiveIn);
    }
    BranchBB->sortUniqueLiveIns();
    BranchBB->addSuccessor(DestBB);
    MBB->replaceSuccessor(DestBB, BranchBB);
    if (TrampolineInsertionPoint == MBB)
      TrampolineInsertionPoint = BranchBB;
  }

  llvm::DebugLoc DL = MI.getDebugLoc();
  MI.eraseFromParent();

  // GFX12+ (\c useAddPC64Inst) short-circuits the whole SVA-lane
  // spill/reload apparatus
  if (ST.useAddPC64Inst()) {
    emitAddPCLongBranch(*BranchBB, *DestBB, DL);
    BlockInfo[BranchBB->getNumber()].Size = computeBlockSize(*BranchBB);
    adjustBlockOffsets(*MBB, std::next(BranchBB->getIterator()));
    RelaxedUnconditionals.insert({BranchBB, DestBB});
    return true;
  }

  // Choose reload-site placement.
  //
  //   Single-pred: DestBB's only predecessor (after the
  //   replaceSuccessor above) is our BranchBB. It's safe to prepend
  //   the reload code directly to DestBB — no other flow reaches it.
  //   No RestoreBB is created; the long jump lands directly at
  //   DestBB.
  //
  //   Multi-pred: prepending to DestBB would corrupt the SGPRs for
  //   every other predecessor's fall-through/branch. Create a fresh
  //   ReloadMBB adjacent to DestBB, route the long jump to it, put
  //   the reload code there, and fall through to DestBB.
  const bool SinglePred = DestBB->pred_size() == 1;
  llvm::MachineBasicBlock *ReloadMBB;
  if (SinglePred) {
    ReloadMBB = DestBB;
  } else {
    ReloadMBB = createNewBlockAfter(MF->back(), DestBB->getBasicBlock());
    std::prev(ReloadMBB->getIterator())
        ->setIsEndSection(ReloadMBB->isEndSection());
    ReloadMBB->setIsEndSection(false);
  }

  emitLongBranch(*BranchBB, *DestBB, *ReloadMBB, DL,
                 BranchBB->getSectionID() != DestBB->getSectionID()
                     ? TM->getMaxCodeSize()
                     : DestOffset - SrcOffset);

  BlockInfo[BranchBB->getNumber()].Size = computeBlockSize(*BranchBB);
  adjustBlockOffsets(*MBB, std::next(BranchBB->getIterator()));

  if (SinglePred) {
    // Reload code (if any) was prepended to DestBB. Update its size
    // and re-thread offsets across it. DestBB.liveins() already has
    // whatever the original CFG made live; the SpillSink added the
    // courier VGPR / SVA storage regs to SpillMBB's liveins but not
    // ours — computeAndAddLiveIns can't run here (asserts empty
    // livein list). Instead let the outer fixed-point loop's next
    // relaxBranchInstructions call see any stale-but-safe live-in
    // state; a subsequent full-MF liveness recompute (if any) will
    // reconcile.
    BlockInfo[DestBB->getNumber()].Size = computeBlockSize(*DestBB);
    adjustBlockOffsets(*DestBB, std::next(DestBB->getIterator()));
    RelaxedUnconditionals.insert({BranchBB, DestBB});
  } else if (!ReloadMBB->empty()) {
    if (MBB->getSectionID() == llvm::MBBSectionID::ColdSectionID &&
        DestBB->getSectionID() != llvm::MBBSectionID::ColdSectionID) {
      auto *NewBB = createNewBlockAfter(*TrampolineInsertionPoint);
      TII->insertUnconditionalBranch(*NewBB, DestBB, llvm::DebugLoc());
      BlockInfo[NewBB->getNumber()].Size = computeBlockSize(*NewBB);
      adjustBlockOffsets(*TrampolineInsertionPoint,
                         std::next(NewBB->getIterator()));
      TrampolineInsertionPoint = NewBB;
      BranchBB->replaceSuccessor(DestBB, NewBB);
      NewBB->addSuccessor(DestBB);
      DestBB = NewBB;
    }
    assert(!DestBB->isEntryBlock());
    auto *PrevBB = &*std::prev(DestBB->getIterator());
    if (auto *FT = PrevBB->getLogicalFallThrough()) {
      assert(FT == DestBB);
      (void)FT;
      TII->insertUnconditionalBranch(*PrevBB, DestBB, llvm::DebugLoc());
      BlockInfo[PrevBB->getNumber()].Size = computeBlockSize(*PrevBB);
    }
    MF->splice(DestBB->getIterator(), ReloadMBB->getIterator());
    ReloadMBB->addSuccessor(DestBB);
    BranchBB->replaceSuccessor(DestBB, ReloadMBB);
    if (TRI->trackLivenessAfterRegAlloc(*MF))
      computeAndAddLiveIns(LiveRegs, *ReloadMBB);
    BlockInfo[ReloadMBB->getNumber()].Size = computeBlockSize(*ReloadMBB);
    adjustBlockOffsets(*PrevBB, DestBB->getIterator());
    ReloadMBB->setSectionID(DestBB->getSectionID());
    ReloadMBB->setIsBeginSection(DestBB->isBeginSection());
    DestBB->setIsBeginSection(false);
    RelaxedUnconditionals.insert({BranchBB, ReloadMBB});
  } else {
    MF->erase(ReloadMBB);
    RelaxedUnconditionals.insert({BranchBB, DestBB});
  }
  return true;
}

bool TargetModuleBranchRelaxationWorker::relaxBranchInstructions() {
  bool Changed = false;
  for (auto &MBB : *MF) {
    auto Last = MBB.getLastNonDebugInstr();
    if (Last == MBB.end())
      continue;
    if (Last->isUnconditionalBranch()) {
      if (auto *DestBB = TII->getBranchDestBlock(*Last)) {
        if (!isBlockInRange(*Last, *DestBB) && !TII->isTailCall(*Last) &&
            !RelaxedUnconditionals.contains({&MBB, DestBB}) &&
            canRelaxBranch(*Last)) {
          fixupUnconditionalBranch(*Last);
          Changed = true;
        }
      }
    }
    llvm::MachineBasicBlock::iterator Next;
    for (auto J = MBB.getFirstTerminator(); J != MBB.end(); J = Next) {
      Next = std::next(J);
      auto &MI = *J;
      if (!MI.isConditionalBranch())
        continue;
      if (MI.getOpcode() == llvm::TargetOpcode::FAULTING_OP)
        continue;
      auto *DestBB = TII->getBranchDestBlock(MI);
      if (!isBlockInRange(MI, *DestBB) && canRelaxBranch(MI)) {
        if (Next != MBB.end() && Next->isConditionalBranch())
          splitBlockBeforeInstr(*Next, DestBB);
        else
          fixupConditionalBranch(MI);
        Changed = true;
        Next = MBB.getFirstTerminator();
      }
    }
  }
  if (Changed)
    adjustBlockOffsets(MF->front());
  return Changed;
}

bool TargetModuleBranchRelaxationWorker::run(llvm::MachineFunction &mf) {
  MF = &mf;
  const auto &ST = MF->getSubtarget();
  TII = ST.getInstrInfo();
  TM = &MF->getTarget();
  TRI = ST.getRegisterInfo();
  MF->RenumberBlocks();

  // Seed per-MBB live-ins from the cached prototype-level predicated
  // liveness (per-PMBB live-in sets from `IPPredicatedLivenessAnalysis`,
  // indexed via `IPPredicatedCFG`) — one source of truth for liveness
  // across the whole target module, no second full backward dataflow.
  // `TracksLiveness` must be set before seeding: `MBB::livein_begin`
  // asserts on it, and the scavenger reads it downstream via
  // `enterBasicBlockEnd` → `LiveUnits.addLiveOuts`.
  MF->getProperties().setTracksLiveness();
  for (llvm::MachineBasicBlock &MBB : *MF) {
    if (MBB.empty())
      continue;
    // Skip MBBs the CFG doesn't know about — MBBs synthesized by
    // pre-relaxer passes (cross-MBB critical-edge splits from
    // \c emitSVSSwitchesForMF, SCC-safe diamond blocks created by
    // SVS load/store emission, etc.) don't have PMBB entries. Their
    // liveins were populated by their creators; don't clobber them.
    if (!IPCFG.contains(MBB))
      continue;
    const PredicatedMachineBasicBlock &PMBB =
        const_cast<IPPredicatedCFG &>(IPCFG).getPredMBB(MBB.front());
    const llvm::LivePhysRegs *LI = IPLiveness.getPMBBLiveIns(PMBB);
    if (!LI)
      continue;
    // Clear any stale live-ins from prior runs, then seed from the
    // PMBB live-in set. LiveInVector uses per-lane masks — we can't
    // recover finer information from LivePhysRegs, so grant every
    // seeded reg the full lane mask (MBB::addLiveIn's default).
    MBB.clearLiveIns();
    for (llvm::MCPhysReg R : *LI)
      MBB.addLiveIn(R);
    MBB.sortUniqueLiveIns();
  }

  scanFunction();
  bool MadeChange = false;
  // Bound the relaxer's outer fixed-point loop. Stock LLVM converges
  // naturally because each fixup tightens distance; under tight
  // `--amdgpu-s-branch-bits` and our SVA-aware scavenger insertions
  // we can fail to converge. Bail safely rather than spin.
  constexpr int kRelaxIterLimit = 64;
  for (int I = 0; I < kRelaxIterLimit; ++I) {
    if (!relaxBranchInstructions())
      break;
    MadeChange = true;
  }
  BlockInfo.clear();
  RelaxedUnconditionals.clear();
  return MadeChange;
}

} // namespace

bool TargetModuleBranchRelaxation::run(llvm::MachineFunction &MF) {
  TargetModuleBranchRelaxationWorker Worker(IPCFG, IPLiveness, SVLoc, Specs);
  return Worker.run(MF);
}

} // namespace luthier
