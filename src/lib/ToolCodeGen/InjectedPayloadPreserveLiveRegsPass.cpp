//===-- InjectedPayloadPreserveLiveRegsPass.cpp ---------------------------===//
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
/// Implements \c InjectedPayloadPreserveLiveRegsPass.
//===----------------------------------------------------------------------===//
#include "luthier/ToolCodeGen/InjectedPayloadPreserveLiveRegsPass.h"
#include "luthier/LLVM/streams.h"
#include "luthier/ToolCodeGen/FunctionAnnotations.h"
#include "luthier/ToolCodeGen/IPPredicatedCFG.h"
#include "luthier/ToolCodeGen/IPPredicatedLivenessPass.h"
#include "luthier/ToolCodeGen/InjectedPayloadAndInstPointAnalysis.h"
#include "luthier/ToolCodeGen/InjectedPayloadSideEffectsAnalysis.h"
#include "luthier/ToolCodeGen/MIRConvenience.h"
#include "luthier/ToolCodeGen/PredicatedMachineBasicBlock.h"
#include "luthier/ToolCodeGen/TargetRegisterBudget.h"
#include <AMDGPU.h>
#include <GCNSubtarget.h>
#include <SIInstrInfo.h>
#include <SIMachineFunctionInfo.h>
#include <llvm/ADT/BitVector.h>
#include <llvm/CodeGen/LivePhysRegs.h>
#include <llvm/CodeGen/MachineBasicBlock.h>
#include <llvm/CodeGen/MachineFunction.h>
#include <llvm/CodeGen/MachineFunctionAnalysis.h>
#include <llvm/CodeGen/MachineInstrBuilder.h>
#include <llvm/CodeGen/MachineOperand.h>
#include <llvm/CodeGen/MachineRegisterInfo.h>
#include <llvm/CodeGen/TargetOpcodes.h>
#include <llvm/CodeGen/TargetRegisterInfo.h>
#include <llvm/CodeGen/TargetSubtargetInfo.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Support/Debug.h>

#undef DEBUG_TYPE
#define DEBUG_TYPE "luthier-payload-preserve-live-regs"

namespace luthier {

llvm::PreservedAnalyses
InjectedPayloadPreserveLiveRegsPass::run(Prototype &IP,
                                         PrototypeAnalysisManager &IPAM) {
  LLVM_DEBUG(luthier::dbgs()
             << "=== Luthier Injected Payload Preserve Live Regs Pass ===\n");

  llvm::Module &IModule = IP.getInstrumentationModule();
  llvm::Module &TargetModule = IP.getTargetModule();

  llvm::ModuleAnalysisManager &MAM =
      IPAM.getResult<IModuleAnalysisManagerPrototypeProxy>(IP).getManager();

  llvm::FunctionAnalysisManager &FAM =
      MAM.getResult<llvm::FunctionAnalysisManagerModuleProxy>(IModule)
          .getManager();

  llvm::FunctionAnalysisManager &TargetFAM =
      IPAM.getResult<TargetFunctionAnalysisManagerPrototypeProxy>(IP)
          .getManager();

  const IPPredicatedLiveness &Liveness =
      IPAM.getResult<IPPredicatedLivenessAnalysis>(IP);
  IPPredicatedCFG &CFG = IPAM.getResult<IPPredCFGAnalysis>(IP).getVecCFG();

  bool Changed = false;

  // Iterate PATCHPOINT markers in the target module directly. For each, compute
  // the live-out at the PATCHPOINT — that is the set the payload must preserve
  // — by unioning successor PMBBs' Active live-ins and stepping backward
  // through the parent MBB up to (but not including) the PATCHPOINT itself.
  for (llvm::Function &TF : TargetModule) {
    llvm::MachineFunctionAnalysis::Result *MFRes =
        TargetFAM.getCachedResult<llvm::MachineFunctionAnalysis>(TF);
    if (!MFRes)
      continue;
    llvm::MachineFunction &TMF = MFRes->getMF();
    const llvm::TargetRegisterInfo &TargetTRI =
        *TMF.getSubtarget().getRegisterInfo();
    const auto &TargetST = TMF.getSubtarget<llvm::GCNSubtarget>();
    const auto *TargetSIMFI = TMF.getInfo<llvm::SIMachineFunctionInfo>();

    // Build the target-MF-specific set of registers the preserve pass
    // deliberately skips.
    llvm::LivePhysRegs FrameOwnedRegs(TargetTRI);
    if (!TargetST.hasArchitectedFlatScratch() &&
        !TargetST.enableFlatScratch()) {
      if (llvm::MCRegister PSB = TargetSIMFI->getPreloadedReg(
              llvm::AMDGPUFunctionArgInfo::PRIVATE_SEGMENT_BUFFER))
        FrameOwnedRegs.addReg(PSB);
    }
    if (!TargetST.hasArchitectedFlatScratch())
      FrameOwnedRegs.addReg(llvm::AMDGPU::FLAT_SCR);
    if (llvm::MCRegister SP = TargetSIMFI->getStackPtrOffsetReg())
      FrameOwnedRegs.addReg(SP);
    if (llvm::MCRegister FP = TargetSIMFI->getFrameOffsetReg())
      FrameOwnedRegs.addReg(FP);

    for (llvm::MachineBasicBlock &MBB : TMF) {
      if (!CFG.contains(MBB))
        continue;
      const PredicatedMachineBasicBlock &PMBB = CFG.at(MBB);

      // MBB live-out (Active partition) = union of successor PMBBs' Active
      // and Inactive live-ins. Boundary semantics: control flow may converge
      // from paths whose EXEC-on/off partitions have swapped, so Active_out
      // gets both partitions of each successor.
      llvm::LivePhysRegs Live(TargetTRI);
      for (const PredicatedMachineBasicBlock &Succ : PMBB.successors()) {
        if (const llvm::LivePhysRegs *SuccLive =
                Liveness.getPMBBActiveLiveIns(Succ))
          for (llvm::MCPhysReg R : *SuccLive)
            Live.addReg(R);
        if (const llvm::LivePhysRegs *SuccLive =
                Liveness.getPMBBInactiveLiveIns(Succ))
          for (llvm::MCPhysReg R : *SuccLive)
            Live.addReg(R);
      }

      for (auto MIt = MBB.rbegin(), MEnd = MBB.rend(); MIt != MEnd; ++MIt) {
        llvm::MachineInstr &MI = *MIt;
        if (MI.getOpcode() != llvm::TargetOpcode::PATCHPOINT) {
          Live.stepBackward(MI);
          continue;
        }

        // PatchpointOpers layout: ID, NBytes, Target, NArgs, CC. Operand
        // 2 is the payload extern handle (a \c GlobalAddress \c Function *
        // in the target module). Its name matches the IModule payload
        // definition's name.
        const llvm::MachineOperand &TargetOp = MI.getOperand(2);
        assert(TargetOp.isGlobal() &&
               "PATCHPOINT target operand must be a GlobalAddress");
        const auto *ExternHandle =
            llvm::cast<llvm::Function>(TargetOp.getGlobal());
        llvm::Function *PayloadDef =
            IModule.getFunction(ExternHandle->getName());
        if (!PayloadDef ||
            !PayloadDef->hasFnAttribute(InjectedPayloadAttribute)) {
          Live.stepBackward(MI);
          continue;
        }
        auto *PayloadMFRes =
            FAM.getCachedResult<llvm::MachineFunctionAnalysis>(*PayloadDef);
        if (!PayloadMFRes) {
          Live.stepBackward(MI);
          continue;
        }
        llvm::MachineFunction *MF = &PayloadMFRes->getMF();

        LLVM_DEBUG({
          luthier::dbgs() << "  payload " << PayloadDef->getName()
                          << " live-out={";
          for (llvm::MCPhysReg R : Live)
            luthier::dbgs() << " " << llvm::printReg(R, &TargetTRI);
          luthier::dbgs() << " }\n";
        });

        // Compute Preserve at regunit granularity:
        //   preserveUnits = regunits(LiveOut(PATCHPOINT))
        //                 \ regunits(payload reads U writes)
        // then coalesce those regunits back into phys regs of at most
        // 64 bits, preferring the widest super-reg whose regunits all
        // remain in preserveUnits. Working at unit granularity avoids
        // preserving a live 64-bit pair whose halves the payload just
        // wrote, and deduplicates the subregs_inclusive expansion that
        // \c LivePhysRegs hands back (a live 64-bit pair also carries
        // its 32-bit halves and their 16-bit slices).
        const InjectedPayloadSideEffects &Acc =
            FAM.getResult<InjectedPayloadSideEffectsAnalysis>(*PayloadDef);
        llvm::BitVector LiveUnits(TargetTRI.getNumRegUnits());
        for (llvm::MCPhysReg R : Live)
          for (llvm::MCRegUnit U : TargetTRI.regunits(R))
            LiveUnits.set(static_cast<unsigned>(U));
        llvm::BitVector AccessedUnits(TargetTRI.getNumRegUnits());
        for (llvm::MCRegister R : Acc.reads())
          for (llvm::MCRegUnit U : TargetTRI.regunits(R))
            AccessedUnits.set(static_cast<unsigned>(U));
        for (llvm::MCRegister R : Acc.writes())
          for (llvm::MCRegUnit U : TargetTRI.regunits(R))
            AccessedUnits.set(static_cast<unsigned>(U));
        llvm::BitVector PreserveUnits = LiveUnits;
        PreserveUnits.reset(AccessedUnits);

        // A register the payload cannot write needs no save/restore.
        const llvm::MachineRegisterInfo &PayloadMRI = MF->getRegInfo();
        llvm::BitVector ReadOnlyUnits(TargetTRI.getNumRegUnits());
        for (llvm::MCPhysReg R : Live)
          if (PayloadMRI.isConstantPhysReg(R))
            for (llvm::MCRegUnit U : TargetTRI.regunits(R))
              ReadOnlyUnits.set(static_cast<unsigned>(U));
        PreserveUnits.reset(ReadOnlyUnits);

        // Keep only the half of \c VCC that the wavefront size actually makes a
        // condition register. \c luthier::addAppOwnedRegisters seeds liveness
        // with \c SIRegisterInfo::getVCC() precisely so that a wave32 target
        // claims \c $vcc_lo and not the 64-bit pair; the unit coalescing below
        // would undo that, because it widens a chosen register to the widest
        // super-register whose units are all still preserved, and nothing else
        // here removes \c $vcc_hi 's unit. The result is a
        // <tt>COPY $vcc</tt> save whose high half is never defined at the
        // payload entry, which the register coalescer rejects as
        // "Use not jointly dominated by defs".
        if (const llvm::MCRegister WaveVCC = TargetST.getRegisterInfo()->getVCC();
            WaveVCC != llvm::AMDGPU::VCC) {
          llvm::BitVector WaveVCCUnits(TargetTRI.getNumRegUnits());
          for (llvm::MCRegUnit U : TargetTRI.regunits(WaveVCC))
            WaveVCCUnits.set(static_cast<unsigned>(U));
          for (llvm::MCRegUnit U :
               TargetTRI.regunits(llvm::MCRegister(llvm::AMDGPU::VCC)))
            if (!WaveVCCUnits.test(static_cast<unsigned>(U)))
              PreserveUnits.reset(static_cast<unsigned>(U));
        }

        llvm::SmallVector<llvm::MCPhysReg, 16> Preserve;
        while (PreserveUnits.any()) {
          int UnitIdx = PreserveUnits.find_first();
          llvm::MCRegUnitRootIterator Roots(
              static_cast<llvm::MCRegUnit>(UnitIdx), &TargetTRI);
          if (!Roots.isValid()) {
            PreserveUnits.reset(static_cast<unsigned>(UnitIdx));
            continue;
          }
          llvm::MCPhysReg Chosen = *Roots;
          const llvm::TargetRegisterClass *ChosenRC =
              TargetTRI.getMinimalPhysRegClass(Chosen);
          unsigned ChosenBits =
              ChosenRC ? TargetTRI.getRegSizeInBits(*ChosenRC) : 0;
          for (llvm::MCPhysReg Super : TargetTRI.superregs(Chosen)) {
            const llvm::TargetRegisterClass *SuperRC =
                TargetTRI.getMinimalPhysRegClass(Super);
            if (!SuperRC)
              continue;
            unsigned SuperBits = TargetTRI.getRegSizeInBits(*SuperRC);
            if (SuperBits > 64 || SuperBits <= ChosenBits)
              continue;
            bool AllInPreserve = true;
            for (llvm::MCRegUnit U : TargetTRI.regunits(Super)) {
              if (!PreserveUnits.test(static_cast<unsigned>(U))) {
                AllInPreserve = false;
                break;
              }
            }
            if (AllInPreserve) {
              Chosen = Super;
              ChosenBits = SuperBits;
            }
          }
          // Skip frame-owned regs (PSB, FLAT_SCR, payload SP/FP): their
          // save/restore is delegated to InjectedPayloadPEIPass so it can
          // decide based on whether the payload actually needs
          // scratch/frame setup.
          if (!FrameOwnedRegs.contains(Chosen))
            Preserve.push_back(Chosen);
          for (llvm::MCRegUnit U : TargetTRI.regunits(Chosen))
            PreserveUnits.reset(static_cast<unsigned>(U));
        }

        LLVM_DEBUG({
          luthier::dbgs() << "    payload reads={";
          for (llvm::MCRegister R : Acc.reads())
            luthier::dbgs() << " " << llvm::printReg(R, &TargetTRI);
          luthier::dbgs() << " } writes={";
          for (llvm::MCRegister R : Acc.writes())
            luthier::dbgs() << " " << llvm::printReg(R, &TargetTRI);
          luthier::dbgs() << " }\n    preserve={";
          for (llvm::MCPhysReg R : Preserve)
            luthier::dbgs() << " " << llvm::printReg(R, &TargetTRI);
          luthier::dbgs() << " }\n";
          // What the payload has to hold all of that in.
          const llvm::Function &PF = MF->getFunction();
          auto attr = [&PF](const char *N) {
            return PF.hasFnAttribute(N)
                       ? PF.getFnAttribute(N).getValueAsString().str()
                       : std::string("<none>");
          };
          const auto &PST = MF->getSubtarget<llvm::GCNSubtarget>();
          auto [MaxV, MaxA] = PST.getMaxNumVectorRegs(PF);
          luthier::dbgs()
              << "    payload budget: amdgpu-num-vgpr=" << attr("amdgpu-num-vgpr")
              << " amdgpu-num-sgpr=" << attr("amdgpu-num-sgpr")
              << " getMaxNumVectorRegs=(vgpr " << MaxV << ", agpr " << MaxA
              << ") maxSGPRs=" << PST.getMaxNumSGPRs(PF)
              << "  preserveCount=" << Preserve.size() << "\n";
          // Does the payload know what the app actually launched with? These
          // are what addAppOwnedRegisters/getTargetRegisterBudget read to
          // decide which registers hold application state.
          const llvm::Function &TF = TMF.getFunction();
          luthier::dbgs()
              << "      payload app-budget attrs: " << AppNumVGPRsAttribute
              << "=" << attr(AppNumVGPRsAttribute) << " "
              << AppNumSGPRsAttribute << "=" << attr(AppNumSGPRsAttribute)
              << " waves-per-eu=" << attr("amdgpu-waves-per-eu") << "\n";
          auto tattr = [&TF](const char *N) {
            return TF.hasFnAttribute(N)
                       ? TF.getFnAttribute(N).getValueAsString().str()
                       : std::string("<none>");
          };
          luthier::dbgs()
              << "      target '" << TF.getName()
              << "' attrs: amdgpu-num-vgpr=" << tattr("amdgpu-num-vgpr")
              << " amdgpu-num-sgpr=" << tattr("amdgpu-num-sgpr") << " "
              << AppNumVGPRsAttribute << "=" << tattr(AppNumVGPRsAttribute)
              << " " << AppNumSGPRsAttribute << "="
              << tattr(AppNumSGPRsAttribute) << "\n";
        });

        // Advance the walk past the PATCHPOINT itself so that a later
        // PATCHPOINT MI upstream in the same MBB sees the correct
        // (post-payload-effects) live state.
        Live.stepBackward(MI);

        if (Preserve.empty())
          continue;

        const llvm::TargetInstrInfo *TII = MF->getSubtarget().getInstrInfo();
        const llvm::TargetRegisterInfo *TRI =
            MF->getSubtarget().getRegisterInfo();
        llvm::MachineRegisterInfo &MRI = MF->getRegInfo();

        // Splice a fresh entry block in front of the payload body and put the
        // save copies there.
        //
        // This is unconditional, and has to be: the copies are inserted at
        // \c getFirstTerminator() of \c MF->front() , which is only guaranteed
        // to precede the payload's own code when that block holds nothing but
        // the branch synthesized here. Reusing a body block as the entry puts
        // every save at the *end* of it, after the payload has already run —
        // and therefore after it may have clobbered the very register being
        // saved. A payload whose entry block defines \c $vcc (any divergent
        // compare does, e.g. the aperture test in the scratch-rebase buffer
        // payloads) then saves the compare result instead of the
        // application's \c $vcc , and leaves the physreg's live range with no
        // def on the path into the block, which fails as
        // "Use not jointly dominated by defs" in the register coalescer.
        //
        // Keeping it unconditional also preserves the original reason the
        // block was introduced: a single-block payload gives the AMDGPU
        // backend's SCC-related optimizations a separate block to work with.
        {
          llvm::MachineBasicBlock &Body = MF->front();
          llvm::MachineBasicBlock *NewEntry =
              MF->CreateMachineBasicBlock(Body.getBasicBlock());
          MF->insert(Body.getIterator(), NewEntry);
          for (const auto &LI : Body.liveins())
            NewEntry->addLiveIn(LI.PhysReg, LI.LaneMask);
          NewEntry->sortUniqueLiveIns();
          NewEntry->addSuccessor(&Body);
          (void)llvm::BuildMI(*NewEntry, NewEntry->end(), llvm::DebugLoc(),
                              TII->get(llvm::AMDGPU::S_BRANCH))
              .addMBB(&Body);
        }

        llvm::MachineBasicBlock &EntryMBB = MF->front();
        // Insert before the fresh entry's terminator (\c S_BRANCH) so
        // COPYs land inside the block, ahead of the branch to the old
        // entry.
        auto EntryInsertPt = EntryMBB.getFirstTerminator();

        // A payload marked \c ExecuteSingleLaneAttribute runs with
        // <tt>EXEC = 1</tt>: \c InjectedPayloadPEIPass parks the app's mask
        // in the SVA's exec-mask spill lanes and narrows the wave to lane 0
        // as the last step of its prologue — which is emitted at the *top* of
        // the entry block, i.e. strictly before the copies below. A plain
        // <tt>%save = COPY $vgprN</tt> is \c EXEC -masked, so under that mask
        // it would only capture lane 0 and the payload's clobber of lanes
        // 1..63 would leak back to the app. Such registers therefore need
        // whole-wave preservation: save under the current mask, flip \c EXEC ,
        // save the complement, flip back — the same idiom
        // \c VGPRStateValueArrayStorage::handOffSVA uses for the \c VGPR0
        // courier. Only the mask being *identical* at save and restore
        // matters, not what it is, and PEI guarantees that: it sets
        // <tt>EXEC = 1</tt> ahead of every copy below and restores the app's
        // \c EXEC after every restore below.
        const bool IsSingleLanePayload =
            PayloadDef->hasFnAttribute(ExecuteSingleLaneAttribute);
        const llvm::MCRegister ExecReg =
            MF->getSubtarget<llvm::GCNSubtarget>().getRegisterInfo()->getExec();

        /// One accepted entry of \c Preserve, resolved down to the virtual
        /// registers its save/restore copies use.
        struct PreservedReg {
          llvm::MCPhysReg PhysReg;
          /// Holds the lanes live under \c EXEC at the copy points.
          llvm::Register ActiveSaveVReg;
          /// Holds the complement; only valid when \c IsWholeWave .
          llvm::Register InactiveSaveVReg;
          /// Vector register in a single-lane payload — needs the
          /// \c EXEC -flip pair around it.
          bool IsWholeWave;
        };
        llvm::SmallVector<PreservedReg, 16> Preserved;
        bool AnyWholeWave = false;

        // Phase 1: validate each candidate and allocate its save vreg(s).
        for (llvm::MCPhysReg PhysReg : Preserve) {
          const llvm::TargetRegisterClass *RC =
              TRI->getPhysRegBaseClass(PhysReg);
          if (!RC) {
            LLVM_DEBUG(luthier::dbgs()
                       << "  skipping " << llvm::printReg(PhysReg, TRI)
                       << ": no reg class\n");
            continue;
          }
          // \c EXEC is never preserved here — it has no meaningful
          // save/restore as an opaque value from this pass's point of view,
          // and \c InjectedPayloadPEIPass owns it outright: it spills the
          // app's mask into the SVA's exec-mask spill lanes on entry and
          // reinstalls it on exit. This filter is what keeps the two passes
          // from both trying to save it, so it must stay unconditional —
          // in particular it already covers \c ExecuteSingleLaneAttribute
          // payloads, where PEI additionally rewrites \c EXEC to 1.
          bool IsUnpreservableArchReg = false;
          for (llvm::MCPhysReg ArchReg :
               {llvm::AMDGPU::EXEC, llvm::AMDGPU::XNACK_MASK}) {
            if (TRI->regsOverlap(PhysReg, ArchReg)) {
              IsUnpreservableArchReg = true;
              break;
            }
          }
          if (IsUnpreservableArchReg) {
            LLVM_DEBUG(luthier::dbgs()
                       << "  skipping " << llvm::printReg(PhysReg, TRI)
                       << ": architectural register (not preserved)\n");
            continue;
          }
          /// Only preserve flat scratch if the target doesn't have architected
          /// flat scratch
          if (TargetST.hasArchitectedFlatScratch() &&
              TRI->regsOverlap(PhysReg, llvm::AMDGPU::FLAT_SCR)) {
            continue;
          }

          const llvm::TargetRegisterClass *CrossCopyRC =
              TRI->getCrossCopyRegClass(RC);
          if (!CrossCopyRC) {
            LLVM_DEBUG(luthier::dbgs()
                       << "  skipping " << llvm::printReg(PhysReg, TRI)
                       << ": no cross-copy class\n");
            continue;
          }
          // A non-allocatable cross-copy class means the target has no way to
          // materialize this reg into a vreg. VCC / SCC do NOT hit this branch
          // on AMDGPU: SIRegisterInfo::getCrossCopyRegClass returns SReg_32 for
          // SCC and passes VCC through as SReg_64, both of which are
          // allocatable.
          if (!CrossCopyRC->isAllocatable()) {
            LLVM_DEBUG(luthier::dbgs()
                       << "  skipping " << llvm::printReg(PhysReg, TRI)
                       << ": cross-copy class not allocatable\n");
            continue;
          }
          // A register is lane-partitioned only if it is a vector one;
          // SGPRs / VCC / SCC hold a single wave-uniform value that a plain
          // COPY captures regardless of \c EXEC .
          const bool IsWholeWave =
              IsSingleLanePayload && (llvm::SIRegisterInfo::hasVGPRs(RC) ||
                                      llvm::SIRegisterInfo::hasAGPRs(RC));
          AnyWholeWave |= IsWholeWave;
          // Entry, per preserved register:
          //   $physreg  = IMPLICIT_DEF          ; dummy def
          //   %savevreg = COPY $physreg
          //
          // Both the live-in declaration and the dummy def are needed, and for
          // different reasons. The live-in is the truth: the application's
          // value arrives in the register. The \c IMPLICIT_DEF is live-range
          // bookkeeping — it gives the physreg a def that dominates the COPY on
          // every path into the block, without which the register coalescer
          // rejects the COPY's live range with "Use not jointly dominated by
          // defs". It costs nothing in the emitted code (\c IMPLICIT_DEF
          // expands to nothing), so the register still holds the caller's value
          // when the COPY executes.
          //
          // The block live-in alone is not enough. \c LiveIntervals only
          // manufactures a def from a block's live-in list for the *reg units*
          // it lists, and a use of a physreg whose unit is reached along an edge
          // from a predecessor that does not define it still has no reaching
          // def — which is exactly the shape here, since the copies sit in a
          // block with a predecessor-free entry only for the units the app
          // actually hands over.
          //
          // These are emitted in this phase, ahead of every COPY that phase 2
          // emits at the same insertion point, so that all the dummy defs
          // precede all the saves. \c BuildMI inserts before the position it is
          // handed (the entry block's terminator), so insertion order is
          // program order.
          if (!EntryMBB.isLiveIn(PhysReg))
            EntryMBB.addLiveIn(PhysReg);
          (void)llvm::BuildMI(EntryMBB, EntryInsertPt, llvm::DebugLoc(),
                              TII->get(llvm::AMDGPU::IMPLICIT_DEF))
              .addReg(PhysReg, llvm::RegState::Define);
          Preserved.push_back(
              {PhysReg, MRI.createVirtualRegister(CrossCopyRC),
               IsWholeWave ? MRI.createVirtualRegister(CrossCopyRC)
                           : llvm::Register(),
               IsWholeWave});
        }
        EntryMBB.sortUniqueLiveIns();

        // Phase 2: emit the copies.
        //
        // Ordering inside each block is load-bearing because
        // \c emitExecMaskFlip clobbers \c SCC , and \c SCC itself can be one
        // of the preserved registers. The flips are therefore hoisted out of
        // the per-register loop into a single pair per block, placed after
        // every entry-block save and before every return-block restore, so no
        // scalar copy ever straddles one.
        auto emitPreserveCopy = [&](llvm::MachineBasicBlock &MBB,
                                    llvm::MachineBasicBlock::iterator Pt,
                                    const PreservedReg &P, llvm::Register VReg,
                                    bool Restore, bool UnderFlippedExec) {
          auto MIB = llvm::BuildMI(MBB, Pt, llvm::DebugLoc(),
                                   TII->get(llvm::AMDGPU::COPY));
          if (Restore)
            MIB.addReg(P.PhysReg, llvm::RegState::Define).addReg(VReg);
          else
            MIB.addReg(VReg, llvm::RegState::Define).addReg(P.PhysReg);
          // A generic \c COPY carries no \c EXEC operand until it is expanded
          // to a \c V_MOV_B32_e32 after register allocation, so nothing would
          // otherwise stop the pre-RA scheduler from hoisting the complement
          // copy above the \c S_NOT that set up its mask. Spell the
          // dependency out.
          if (UnderFlippedExec)
            MIB.addReg(ExecReg, llvm::RegState::Implicit);
        };

        for (const PreservedReg &P : Preserved)
          emitPreserveCopy(EntryMBB, EntryInsertPt, P, P.ActiveSaveVReg,
                           /*Restore=*/false, /*UnderFlippedExec=*/false);
        if (AnyWholeWave) {
          emitExecMaskFlip(EntryMBB, EntryInsertPt);
          for (const PreservedReg &P : Preserved)
            if (P.IsWholeWave)
              emitPreserveCopy(EntryMBB, EntryInsertPt, P, P.InactiveSaveVReg,
                               /*Restore=*/false, /*UnderFlippedExec=*/true);
          emitExecMaskFlip(EntryMBB, EntryInsertPt);
        }

        // Return blocks: emit restore COPYs before the first terminator and
        // tag the terminator with an implicit-use of each physreg.
        for (llvm::MachineBasicBlock &MBB : *MF) {
          if (!MBB.isReturnBlock())
            continue;
          auto FirstTerm = MBB.getFirstTerminator();
          if (AnyWholeWave) {
            emitExecMaskFlip(MBB, FirstTerm);
            for (const PreservedReg &P : Preserved)
              if (P.IsWholeWave)
                emitPreserveCopy(MBB, FirstTerm, P, P.InactiveSaveVReg,
                                 /*Restore=*/true, /*UnderFlippedExec=*/true);
            emitExecMaskFlip(MBB, FirstTerm);
          }
          for (const PreservedReg &P : Preserved) {
            emitPreserveCopy(MBB, FirstTerm, P, P.ActiveSaveVReg,
                             /*Restore=*/true, /*UnderFlippedExec=*/false);
            // Add implicit use of $physreg on the terminator so the live-out
            // is visible to RA.
            if (FirstTerm != MBB.end()) {
              FirstTerm->addOperand(llvm::MachineOperand::CreateReg(
                  P.PhysReg, /*isDef=*/false, /*isImp=*/true));
            }
          }
        }
        Changed |= !Preserved.empty();
      } // end for each PATCHPOINT MI in reverse
    } // end for each MBB in TMF
  } // end for each target function

  if (!Changed)
    return llvm::PreservedAnalyses::all();

  // Preserve the outer MAM proxy so the Prototype adaptor doesn't
  // wipe every cached module-level analysis for both modules on the way out —
  // downstream passes still need the cached MachineFunctionAnalysis results
  // for the instrumentation module we just mutated.
  llvm::PreservedAnalyses PA = llvm::PreservedAnalyses::none();
  // Preservation copies are emitted into existing injected-payload MFs in the
  // instrumentation module. No MachineFunction is created or destroyed and the
  // target module is untouched, so the inner managers remain accurate; only
  // Prototype-level analyses over payload liveness are dropped.
  PA.preserve<TargetModuleAnalysisManagerPrototypeProxy>();
  PA.preserve<TargetFunctionAnalysisManagerPrototypeProxy>();
  PA.preserve<TargetMachineFunctionAnalysisManagerPrototypeProxy>();
  PA.preserve<IModuleAnalysisManagerPrototypeProxy>();
  PA.preserve<IModuleFunctionAnalysisManagerPrototypeProxy>();
  PA.preserve<IModuleMachineFunctionAnalysisManagerPrototypeProxy>();
  return PA;
}

} // namespace luthier
