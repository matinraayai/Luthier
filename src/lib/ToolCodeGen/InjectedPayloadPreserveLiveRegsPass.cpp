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

        llvm::MachineBasicBlock &EntryMBB = MF->front();
        // Insert before the fresh entry's terminator (\c S_BRANCH) so
        // COPYs land inside the block, ahead of the branch to the old
        // entry.
        auto EntryInsertPt = EntryMBB.getFirstTerminator();

        // For each preserved phys-reg, emit an entry-block COPY into a fresh
        // virtual register, and at every return block, emit a COPY back to
        // the phys-reg before the terminator + add an implicit-use of the
        // phys-reg on the terminator so the register allocator treats it as
        // a live-out of the function.
        for (llvm::MCPhysReg PhysReg : Preserve) {
          const llvm::TargetRegisterClass *RC =
              TRI->getPhysRegBaseClass(PhysReg);
          if (!RC) {
            LLVM_DEBUG(luthier::dbgs()
                       << "  skipping " << llvm::printReg(PhysReg, TRI)
                       << ": no reg class\n");
            continue;
          }
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
          llvm::Register SaveVReg = MRI.createVirtualRegister(CrossCopyRC);
          // Entry:
          // %physreg = IMPLICIT_DEF ; Introduce a dummy def besides the live-in
          // decl to keep the live intervals happy
          // %savevreg = COPY $physreg ; mark $physreg live-in.
          (void)llvm::BuildMI(EntryMBB, EntryInsertPt, llvm::DebugLoc(),
                              TII->get(llvm::AMDGPU::IMPLICIT_DEF))
              .addReg(PhysReg, llvm::RegState::Define);
          (void)llvm::BuildMI(EntryMBB, EntryInsertPt, llvm::DebugLoc(),
                              TII->get(llvm::AMDGPU::COPY))
              .addReg(SaveVReg, llvm::RegState::Define)
              .addReg(PhysReg);
          // if (!EntryMBB.isLiveIn(PhysReg))
          //   EntryMBB.addLiveIn(PhysReg);

          // Return blocks: emit restore COPY before the first terminator and
          // tag the terminator with an implicit-use of the physreg.
          for (llvm::MachineBasicBlock &MBB : *MF) {
            if (!MBB.isReturnBlock())
              continue;
            auto FirstTerm = MBB.getFirstTerminator();
            (void)llvm::BuildMI(MBB, FirstTerm, llvm::DebugLoc(),
                                TII->get(llvm::AMDGPU::COPY))
                .addReg(PhysReg, llvm::RegState::Define)
                .addReg(SaveVReg);
            // Add implicit use of $physreg on the terminator so the live-out
            // is visible to RA.
            if (FirstTerm != MBB.end()) {
              FirstTerm->addOperand(llvm::MachineOperand::CreateReg(
                  PhysReg, /*isDef=*/false, /*isImp=*/true));
            }
          }
          Changed = true;
        }
        EntryMBB.sortUniqueLiveIns();
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
